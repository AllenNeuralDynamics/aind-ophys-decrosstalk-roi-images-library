"""Register one plane's movie using its paired plane's motion transforms.

Crosstalk is estimated between two simultaneously-imaged planes, so the two
movies must first be brought into the same frame of reference: each plane's
raw movie is re-registered using the *other* plane's suite2p offsets. That is
why the suite2p ``block_size`` / ``nonrigid`` settings are read from the
upstream motion-correction metadata rather than re-chosen here.

Also provides the episodic-mean-FOV reduction used both as the input to
coefficient fitting and as the QC reference movie.
"""

import logging
from pathlib import Path
from typing import List, Optional

import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
from aind_ophys_utils.array_utils import normalize_array
from aind_ophys_utils.video_utils import encode_video
from suite2p.registration import nonrigid

mpl.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = (128, 128)
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_MAX_NUM_EPOCHS = 10
DEFAULT_NUM_FRAMES = 1000
PROJECTION_CHUNK_SIZE = 100


def get_s2p_motion_transform(
    csv_path: Path, non_rigid: bool = True
) -> pd.DataFrame:
    """Load a suite2p motion-transform CSV, restoring array columns.

    The non-rigid per-block offsets round-trip through CSV as bracketed
    strings, so they are parsed back into float arrays.

    Parameters
    ----------
    csv_path : Path
        Path to a suite2p rigid or non-rigid motion-transform CSV.
    non_rigid : bool, optional
        Whether the CSV carries non-rigid columns.

    Returns
    -------
    pandas.DataFrame
        One row per frame, with ``x`` / ``y`` rigid offsets and, when
        ``non_rigid``, ``nonrigid_x`` / ``nonrigid_y`` arrays.

    Raises
    ------
    KeyError
        If ``non_rigid`` is requested but the CSV has no non-rigid columns.
    """
    reg_df = pd.read_csv(csv_path)
    if not non_rigid:
        return reg_df

    missing = {"nonrigid_x", "nonrigid_y"} - set(reg_df.columns)
    if missing:
        raise KeyError(
            f"{csv_path} has no {sorted(missing)} column(s), but non-rigid "
            f"registration was requested."
        )
    if isinstance(reg_df.nonrigid_x[0], str):
        for column in ("nonrigid_x", "nonrigid_y"):
            reg_df[column] = reg_df[column].apply(_parse_offset_array)
    return reg_df


def _parse_offset_array(value: str) -> np.ndarray:
    """Parse a ``"[1.0, 2.0]"`` CSV cell into a float32 array.

    Parameters
    ----------
    value : str
        Bracketed, comma-separated numbers as written by ``to_csv``.

    Returns
    -------
    np.ndarray
        The parsed per-block offsets.
    """
    inner = value.split("[")[1].split("]")[0]
    return np.array([np.float32(v) for v in inner.split(",")])


def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Roll a single frame by the given rigid offsets.

    Parameters
    ----------
    frame : np.ndarray
        A 2D ``Ly x Lx`` frame.
    dy : int
        Vertical shift.
    dx : int
        Horizontal shift.

    Returns
    -------
    np.ndarray
        The shifted frame.

    Raises
    ------
    AssertionError
        If ``frame`` is not 2D or a shift exceeds the frame extent.
    """
    assert frame.ndim == 2, "frame must be 2D"
    assert np.abs(dy) < frame.shape[0], "dy must be less than frame height"
    assert np.abs(dx) < frame.shape[1], "dx must be less than frame width"
    return np.roll(frame, (-dy, -dx), axis=(0, 1))


def paired_plane_cached_movie(
    h5_file: Path,
    reg_df: pd.DataFrame,
    tmp_dir: Path = Path("../scratch/"),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    non_rigid: bool = True,
    block_size: Optional[List[int]] = None,
) -> Path:
    """Re-register a raw movie with another plane's offsets, chunk by chunk.

    The result is a large intermediate written to ``tmp_dir``; the caller
    deletes it once the decrosstalk movie has been produced.

    Parameters
    ----------
    h5_file : Path
        The *raw* (not motion-corrected) movie for this plane.
    reg_df : pandas.DataFrame
        Motion transforms of the *paired* plane, from
        :func:`get_s2p_motion_transform`.
    tmp_dir : Path, optional
        Directory for the intermediate movie.
    chunk_size : int, optional
        Frames transformed per chunk; bounds peak memory use.
    non_rigid : bool, optional
        Whether to apply the per-block non-rigid offsets after the rigid roll.
    block_size : list of int, optional
        Suite2p non-rigid block size; must match what suite2p used upstream.

    Returns
    -------
    Path
        Path to the written intermediate movie.

    Raises
    ------
    ValueError
        If the movie length and the offset table disagree.
    """
    block_size = tuple(block_size or DEFAULT_BLOCK_SIZE)
    temp_path = (
        Path(tmp_dir) / f'{Path(h5_file).name.split(".")[0]}'
        "_registered_to_pair.h5"
    )
    y_shifts = reg_df["y"].values
    x_shifts = reg_df["x"].values

    with h5py.File(h5_file, "r") as f:
        data_length, n_y, n_x = f["data"].shape
        if not data_length == len(y_shifts) == len(x_shifts):
            raise ValueError(
                f"{h5_file} has {data_length} frames but the paired motion "
                f"transform has {len(y_shifts)} rows; they must match."
            )
        if non_rigid:
            blocks = nonrigid.make_blocks(
                Ly=n_y, Lx=n_x, block_size=block_size
            )
            ymax1 = np.vstack(reg_df.nonrigid_y.values)
            xmax1 = np.vstack(reg_df.nonrigid_x.values)
        logger.info(
            "Registering %s (%s frames) to its pair -> %s",
            Path(h5_file).name,
            data_length,
            temp_path,
        )

        start_frames = np.arange(0, data_length, chunk_size)
        end_frames = np.append(start_frames[1:], data_length)
        with h5py.File(temp_path, "w") as cache_f:
            cache_f.create_dataset(
                "data",
                (data_length, n_y, n_x),
                dtype=np.int16,
                chunks=(min(1000, data_length), n_y, n_x),
            )
            for start_frame, end_frame in zip(start_frames, end_frames):
                frame_group = f["data"][start_frame:end_frame]
                r_frames = np.zeros_like(frame_group, dtype=np.int16)
                for frame_index, (frame, dy, dx) in enumerate(
                    zip(
                        frame_group,
                        y_shifts[start_frame:end_frame],
                        x_shifts[start_frame:end_frame],
                    )
                ):
                    r_frames[frame_index] = shift_frame(frame, dy=dy, dx=dx)
                if non_rigid:
                    # int16 rather than uint16 to match suite2p's own dtype and
                    # the rest of the pipeline's movies.
                    r_frames = nonrigid.transform_data(
                        r_frames,
                        yblock=blocks[0],
                        xblock=blocks[1],
                        nblocks=blocks[2],
                        ymax1=ymax1[start_frame:end_frame],
                        xmax1=xmax1[start_frame:end_frame],
                        bilinear=True,
                    )
                cache_f["data"][start_frame:end_frame] = r_frames
    return temp_path


def projection_process_chunked(
    h5_dataset: h5py.Dataset,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    projection: str = "max",
    chunk_size: int = PROJECTION_CHUNK_SIZE,
) -> np.ndarray:
    """Project a frame range without loading it all into memory.

    Parameters
    ----------
    h5_dataset : h5py.Dataset
        The movie dataset to reduce.
    start_idx : int, optional
        First frame; defaults to the start of the dataset.
    end_idx : int, optional
        Exclusive last frame; defaults to the end of the dataset.
    projection : str, optional
        ``"max"`` or ``"avg"``.
    chunk_size : int, optional
        Frames loaded per step.

    Returns
    -------
    np.ndarray
        The 2D projection.

    Raises
    ------
    ValueError
        If the range is empty or ``projection`` is not recognised.
    """
    start_idx = 0 if start_idx is None else max(0, start_idx)
    end_idx = (
        h5_dataset.shape[0]
        if end_idx is None
        else min(h5_dataset.shape[0], end_idx)
    )
    if start_idx >= end_idx:
        raise ValueError("Start index must be less than end index")
    if projection not in ("avg", "max"):
        raise ValueError(
            f'projection must be "max" or "avg", not {projection}'
        )

    running = None
    count = 0
    for i in range(start_idx, end_idx, chunk_size):
        chunk = h5_dataset[i : min(i + chunk_size, end_idx)]
        if projection == "avg":
            chunk_value = np.sum(chunk, axis=0, dtype=np.float64)
            running = chunk_value if running is None else running + chunk_value
            count += len(chunk)
        else:
            chunk_value = np.max(chunk, axis=0)
            running = (
                chunk_value
                if running is None
                else np.maximum(running, chunk_value)
            )
    return (running / count).astype(np.float32) if count else running


def _epoch_start_frames(
    data_length: int, max_num_epochs: int, num_frames: int
) -> tuple:
    """Choose evenly spaced epoch start frames inside a movie.

    Half an epoch is skipped at each end so the projections avoid the
    laser-settling frames at the start and any truncation at the end.

    Parameters
    ----------
    data_length : int
        Total frames in the movie.
    max_num_epochs : int
        Upper bound on the number of epochs.
    num_frames : int
        Requested frames per epoch.

    Returns
    -------
    tuple
        ``(start_frames, num_frames)``, with ``num_frames`` reduced to fit.

    Raises
    ------
    ValueError
        If the movie is too short for even one epoch.
    """
    num_epochs = min(max_num_epochs, data_length // num_frames)
    if num_epochs < 1:
        raise ValueError(
            f"Movie has {data_length} frames, too few for one epoch of "
            f"{num_frames} frames."
        )
    epoch_interval = data_length // (num_epochs + 1)
    num_frames = min(num_frames, epoch_interval)
    start_frames = [
        num_frames // 2 + i * epoch_interval for i in range(num_epochs)
    ]
    return start_frames, num_frames


def episodic_mean_fov(
    movie_fn: Path,
    save_dir: Path,
    max_num_epochs: int = DEFAULT_MAX_NUM_EPOCHS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    save_webm: bool = False,
) -> Path:
    """Write per-epoch mean FOV images, plus avg/max projection PNGs.

    Parameters
    ----------
    movie_fn : Path
        HDF5 movie to reduce.
    save_dir : Path
        Existing directory the outputs are written into.
    max_num_epochs : int, optional
        Upper bound on the number of epochs.
    num_frames : int, optional
        Frames averaged per epoch.
    save_webm : bool, optional
        Also encode the epoch stack as a webm for QC review.

    Returns
    -------
    Path
        Path to the written episodic-mean-FOV HDF5 file.

    Raises
    ------
    ValueError
        If ``movie_fn`` is not an HDF5 file or ``save_dir`` is not a directory.
    """
    movie_fn = Path(movie_fn)
    save_dir = Path(save_dir)
    if movie_fn.suffix != ".h5":
        raise ValueError("movie_fn must be an h5 file")
    if not save_dir.is_dir():
        raise ValueError("save_dir must be a directory")

    with h5py.File(movie_fn, "r") as f:
        start_frames, num_frames = _epoch_start_frames(
            f["data"].shape[0], max_num_epochs, num_frames
        )
        avg_img = projection_process_chunked(f["data"], projection="avg")
        max_img = projection_process_chunked(f["data"], projection="max")
        mean_fov = np.zeros(
            (len(start_frames), f["data"].shape[1], f["data"].shape[2])
        )
        for i, start_frame in enumerate(start_frames):
            mean_fov[i] = projection_process_chunked(
                f["data"],
                start_idx=start_frame,
                end_idx=start_frame + num_frames,
                projection="avg",
            )

    for image, name in (
        (avg_img, f"{movie_fn.stem}_avg_img.png"),
        (max_img, f"{movie_fn.stem}_max_img.png"),
    ):
        plt.imsave(save_dir / name, image, cmap="gray")

    save_path = save_dir / f"{movie_fn.stem}_episodic_mean_fov.h5"
    with h5py.File(save_path, "w") as f:
        f.create_dataset("data", data=mean_fov)
    if save_webm:
        encode_video(
            normalize_array(mean_fov),
            str(save_dir / f"{movie_fn.stem}_episodic_mean_fov.webm"),
            3,
        )
    return save_path
