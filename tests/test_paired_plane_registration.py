"""Tests for paired_plane_registration.

The movies, offset tables and projections here are built as the production
types -- HDF5 int16 movie datasets and the DataFrame ``read_csv`` returns from
a suite2p motion-transform CSV -- so the array-consuming code paths are
exercised as the pipeline calls them.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd
from suite2p.registration import nonrigid

from aind_ophys_decrosstalk_roi_images_library import (
    paired_plane_registration as ppr,
)

N_FRAMES = 50
FRAME_SHAPE = (64, 64)
BLOCK_SIZE = (32, 32)
MAX_RIGID_SHIFT = 3


def _write_movie(path: Path, n_frames: int = N_FRAMES) -> np.ndarray:
    """Write an int16 HDF5 movie and return the frames written.

    int16 is the dtype every movie in the pipeline carries.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2000, (n_frames, *FRAME_SHAPE)).astype(np.int16)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
    return data


def _write_motion_transform_csv(
    path: Path, n_frames: int = N_FRAMES, n_blocks: int = 0
) -> Path:
    """Write a suite2p motion-transform CSV as motion correction writes it.

    The per-block non-rigid offsets are serialised with ``np.array2string``,
    which is what puts them in the file as bracketed strings. ``n_blocks`` of
    0 writes a rigid-only CSV.
    """
    rng = np.random.default_rng(1)
    columns = {
        "framenumber": np.arange(n_frames),
        "x": rng.integers(-MAX_RIGID_SHIFT, MAX_RIGID_SHIFT + 1, n_frames),
        "y": rng.integers(-MAX_RIGID_SHIFT, MAX_RIGID_SHIFT + 1, n_frames),
        "correlation": rng.random(n_frames),
    }
    if n_blocks:
        offsets = rng.uniform(-2, 2, (n_frames, n_blocks)).astype(np.float32)
        for name, values in (
            ("nonrigid_x", offsets),
            ("nonrigid_y", offsets[::-1]),
        ):
            columns[name] = [
                np.array2string(
                    arr,
                    separator=",",
                    suppress_small=True,
                    max_line_width=4096,
                )
                for arr in values
            ]
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


def _n_blocks() -> int:
    """Number of suite2p non-rigid blocks for the test frame size."""
    blocks = nonrigid.make_blocks(
        Ly=FRAME_SHAPE[0], Lx=FRAME_SHAPE[1], block_size=BLOCK_SIZE
    )
    return int(np.prod(blocks[2]))


class TestGetS2pMotionTransform(unittest.TestCase):
    """Loading a suite2p motion-transform CSV back into arrays."""

    def test_nonrigid_columns_become_float_arrays(self) -> None:
        """The bracketed CSV cells parse back into per-block offsets."""
        n_blocks = _n_blocks()
        with TemporaryDirectory() as tmp:
            csv_path = _write_motion_transform_csv(
                Path(tmp) / "VISp_0_motion_transform.csv", n_blocks=n_blocks
            )
            reg_df = ppr.get_s2p_motion_transform(csv_path)
        for column in ("nonrigid_x", "nonrigid_y"):
            self.assertIsInstance(reg_df[column][0], np.ndarray)
            self.assertEqual(reg_df[column][0].shape, (n_blocks,))
        # Stacking is how paired_plane_cached_movie consumes them.
        self.assertEqual(
            np.vstack(reg_df.nonrigid_x.values).shape, (N_FRAMES, n_blocks)
        )

    def test_rigid_offsets_stay_integers(self) -> None:
        """np.roll rejects float shifts, so the rigid columns must be ints."""
        with TemporaryDirectory() as tmp:
            csv_path = _write_motion_transform_csv(Path(tmp) / "rigid.csv")
            reg_df = ppr.get_s2p_motion_transform(csv_path, non_rigid=False)
        for column in ("x", "y"):
            self.assertTrue(np.issubdtype(reg_df[column].dtype, np.integer))

    def test_missing_nonrigid_columns_raise(self) -> None:
        """A rigid-only CSV cannot satisfy a non-rigid request."""
        with TemporaryDirectory() as tmp:
            csv_path = _write_motion_transform_csv(Path(tmp) / "rigid.csv")
            with self.assertRaises(KeyError) as ctx:
                ppr.get_s2p_motion_transform(csv_path)
        self.assertIn("nonrigid_x", str(ctx.exception))


class TestShiftFrame(unittest.TestCase):
    """The rigid roll applied to a single movie frame."""

    def test_shifts_match_np_roll(self) -> None:
        """A real frame is rolled by the negated offsets."""
        rng = np.random.default_rng(2)
        frame = rng.integers(0, 2000, FRAME_SHAPE).astype(np.int16)
        shifted = ppr.shift_frame(frame, dy=2, dx=-3)
        np.testing.assert_array_equal(
            shifted, np.roll(frame, (-2, 3), axis=(0, 1))
        )
        self.assertEqual(shifted.dtype, np.int16)

    def test_a_movie_stack_is_rejected(self) -> None:
        """Only 2D frames are shiftable."""
        with self.assertRaises(AssertionError):
            ppr.shift_frame(np.zeros((2, *FRAME_SHAPE), np.int16), dy=0, dx=0)


class TestProjectionProcessChunked(unittest.TestCase):
    """Chunked reduction over an HDF5 movie dataset."""

    def test_matches_the_whole_array_reduction(self) -> None:
        """Chunking does not change the avg or max projection."""
        with TemporaryDirectory() as tmp:
            movie = Path(tmp) / "VISp_0_registered.h5"
            data = _write_movie(movie)
            with h5py.File(movie, "r") as f:
                avg = ppr.projection_process_chunked(
                    f["data"], projection="avg", chunk_size=7
                )
                maximum = ppr.projection_process_chunked(
                    f["data"], projection="max", chunk_size=7
                )
        np.testing.assert_allclose(
            avg, data.mean(axis=0).astype(np.float32), rtol=1e-5
        )
        np.testing.assert_array_equal(maximum, data.max(axis=0))
        self.assertEqual(avg.dtype, np.float32)

    def test_projects_a_frame_range(self) -> None:
        """An epoch projection reads only its own frames."""
        with TemporaryDirectory() as tmp:
            movie = Path(tmp) / "VISp_0_registered.h5"
            data = _write_movie(movie)
            with h5py.File(movie, "r") as f:
                avg = ppr.projection_process_chunked(
                    f["data"], start_idx=10, end_idx=20, projection="avg"
                )
        np.testing.assert_allclose(
            avg, data[10:20].mean(axis=0).astype(np.float32), rtol=1e-5
        )

    def test_range_is_clipped_to_the_dataset(self) -> None:
        """A request past the end reduces the available frames instead."""
        with TemporaryDirectory() as tmp:
            movie = Path(tmp) / "VISp_0_registered.h5"
            data = _write_movie(movie)
            with h5py.File(movie, "r") as f:
                avg = ppr.projection_process_chunked(
                    f["data"], start_idx=-5, end_idx=N_FRAMES + 100
                )
                maximum = ppr.projection_process_chunked(
                    f["data"], projection="max", end_idx=N_FRAMES + 100
                )
        np.testing.assert_array_equal(maximum, data.max(axis=0))
        self.assertEqual(avg.shape, FRAME_SHAPE)

    def test_empty_range_and_unknown_projection_raise(self) -> None:
        """Both misuses fail at the call rather than downstream."""
        with TemporaryDirectory() as tmp:
            movie = Path(tmp) / "VISp_0_registered.h5"
            _write_movie(movie)
            with h5py.File(movie, "r") as f:
                with self.assertRaises(ValueError):
                    ppr.projection_process_chunked(
                        f["data"], start_idx=10, end_idx=10
                    )
                with self.assertRaises(ValueError):
                    ppr.projection_process_chunked(
                        f["data"], projection="median"
                    )


class TestPairedPlaneCachedMovie(unittest.TestCase):
    """Re-registering a raw movie with the paired plane's offsets."""

    def test_rigid_registration_rolls_every_frame(self) -> None:
        """Each frame is rolled by that frame's offsets, in int16."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "VISp_0.h5"
            data = _write_movie(raw)
            reg_df = ppr.get_s2p_motion_transform(
                _write_motion_transform_csv(root / "pair.csv"),
                non_rigid=False,
            )
            cached = ppr.paired_plane_cached_movie(
                raw, reg_df, tmp_dir=root, non_rigid=False, chunk_size=13
            )
            with h5py.File(cached, "r") as f:
                registered = f["data"][:]
        self.assertEqual(registered.dtype, np.int16)
        self.assertEqual(registered.shape, data.shape)
        for i in (0, 20, N_FRAMES - 1):
            np.testing.assert_array_equal(
                registered[i],
                ppr.shift_frame(data[i], dy=reg_df["y"][i], dx=reg_df["x"][i]),
            )

    def test_nonrigid_registration_applies_the_block_offsets(self) -> None:
        """The non-rigid pass runs on the stacked per-block offsets."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "VISp_0.h5"
            data = _write_movie(raw)
            reg_df = ppr.get_s2p_motion_transform(
                _write_motion_transform_csv(
                    root / "pair.csv", n_blocks=_n_blocks()
                )
            )
            cached = ppr.paired_plane_cached_movie(
                raw,
                reg_df,
                tmp_dir=root,
                non_rigid=True,
                block_size=list(BLOCK_SIZE),
                chunk_size=13,
            )
            with h5py.File(cached, "r") as f:
                registered = f["data"][:]
        self.assertEqual(registered.shape, data.shape)
        self.assertEqual(registered.dtype, np.int16)
        self.assertEqual(cached.name, "VISp_0_registered_to_pair.h5")

    def test_block_size_may_be_an_array(self) -> None:
        """suite2p hands its own block size out as an array, not a list.

        Testing it for truthiness would raise ``ValueError: The truth value of
        an array with more than one element is ambiguous``.
        """
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "VISp_0.h5"
            _write_movie(raw)
            reg_df = ppr.get_s2p_motion_transform(
                _write_motion_transform_csv(
                    root / "pair.csv", n_blocks=_n_blocks()
                )
            )
            cached = ppr.paired_plane_cached_movie(
                raw,
                reg_df,
                tmp_dir=root,
                block_size=np.array(BLOCK_SIZE),
            )
            with h5py.File(cached, "r") as f:
                self.assertEqual(f["data"].shape[0], N_FRAMES)

    def test_an_empty_block_size_raises(self) -> None:
        """A malformed upstream value must not silently become the default.

        Registering with a block size suite2p did not use warps the paired
        movie wrongly, which shows up only as wrong mixing coefficients.
        """
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "VISp_0.h5"
            _write_movie(raw)
            reg_df = ppr.get_s2p_motion_transform(
                _write_motion_transform_csv(root / "pair.csv"), non_rigid=False
            )
            with self.assertRaises(ValueError) as ctx:
                ppr.paired_plane_cached_movie(
                    raw, reg_df, tmp_dir=root, non_rigid=False, block_size=[]
                )
        self.assertIn("exactly 2 values", str(ctx.exception))

    def test_length_mismatch_raises(self) -> None:
        """A transform table shorter than the movie is a fail-fast error."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "VISp_0.h5"
            _write_movie(raw)
            reg_df = ppr.get_s2p_motion_transform(
                _write_motion_transform_csv(
                    root / "pair.csv", n_frames=N_FRAMES - 1
                ),
                non_rigid=False,
            )
            with self.assertRaises(ValueError) as ctx:
                ppr.paired_plane_cached_movie(
                    raw, reg_df, tmp_dir=root, non_rigid=False
                )
        self.assertIn(str(N_FRAMES), str(ctx.exception))


class TestEpisodicMeanFov(unittest.TestCase):
    """The episodic mean FOV stack and its projection images."""

    def test_writes_the_stack_and_the_projection_images(self) -> None:
        """One mean image per epoch, plus the avg and max PNGs."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            movie = root / "VISp_0_registered.h5"
            _write_movie(movie)
            save_path = ppr.episodic_mean_fov(
                movie, root, max_num_epochs=4, num_frames=5
            )
            with h5py.File(save_path, "r") as f:
                mean_fov = f["data"][:]
            self.assertTrue((root / "VISp_0_registered_avg_img.png").exists())
            self.assertTrue((root / "VISp_0_registered_max_img.png").exists())
        self.assertEqual(
            save_path.name, "VISp_0_registered_episodic_mean_fov.h5"
        )
        self.assertEqual(mean_fov.shape, (4, *FRAME_SHAPE))

    def test_webm_is_encoded_from_the_stack(self) -> None:
        """The QC reference movie is encoded from the real float stack."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            movie = root / "VISp_0_decrosstalk.h5"
            _write_movie(movie)
            ppr.episodic_mean_fov(
                movie, root, max_num_epochs=4, num_frames=5, save_webm=True
            )
            webm = root / "VISp_0_decrosstalk_episodic_mean_fov.webm"
            self.assertTrue(webm.exists())
            self.assertGreater(webm.stat().st_size, 0)

    def test_rejects_a_non_h5_movie_and_a_missing_save_dir(self) -> None:
        """Both inputs are checked before any frame is read."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            movie = root / "VISp_0_registered.h5"
            _write_movie(movie)
            with self.assertRaises(ValueError):
                ppr.episodic_mean_fov(root / "movie.nwb", root)
            with self.assertRaises(ValueError):
                ppr.episodic_mean_fov(movie, root / "absent")


class TestEpochStartFrames(unittest.TestCase):
    """Epoch start/length selection across movie lengths."""

    def test_full_length_movie_uses_max_epochs(self) -> None:
        """A long movie is capped by max_num_epochs, not by its length."""
        starts, num_frames = ppr._epoch_start_frames(30000, 10, 1000)
        self.assertEqual(len(starts), 10)
        self.assertEqual(num_frames, 1000)

    def test_epoch_fits_inside_the_movie(self) -> None:
        """The last epoch never reads past the end of the movie."""
        for data_length in (2, 100, 999, 3000, 30000):
            starts, num_frames = ppr._epoch_start_frames(data_length, 10, 1000)
            self.assertLessEqual(starts[-1] + num_frames, data_length)

    def test_short_movie_yields_one_shortened_epoch(self) -> None:
        """A debug-truncated movie gets one epoch instead of raising.

        100 frames against a 300-frame request is what a --debug run asks
        for, so it must shorten the epoch rather than fail.
        """
        starts, num_frames = ppr._epoch_start_frames(100, 10, 300)
        self.assertEqual(starts, [25])
        self.assertEqual(num_frames, 50)

    def test_num_frames_never_drops_below_one(self) -> None:
        """A one-frame movie still asks for a single frame, not zero.

        num_frames of 0 makes the chunked projection return None, which fails
        far from the cause.
        """
        starts, num_frames = ppr._epoch_start_frames(1, 10, 1000)
        self.assertEqual(num_frames, 1)
        self.assertEqual(starts, [0])

    def test_epoch_count_is_at_least_one(self) -> None:
        """start_frames is never empty, whatever the ratio works out to."""
        for data_length in (1, 2, 50, 100, 299):
            starts, _ = ppr._epoch_start_frames(data_length, 10, 300)
            self.assertGreaterEqual(len(starts), 1)

    def test_empty_movie_raises(self) -> None:
        """Zero frames is unprojectable and fails with a clear message."""
        with self.assertRaises(ValueError) as ctx:
            ppr._epoch_start_frames(0, 10, 1000)
        self.assertIn("nothing to project", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
