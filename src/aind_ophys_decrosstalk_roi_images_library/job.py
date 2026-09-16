"""Top-level paired-plane decrosstalk orchestration.

``run()`` is the library entry point called by the thin runtime wrapper: read
metadata (v1 or v2) -> configure logging -> resolve the plane pair -> register
each plane's raw movie with its partner's motion transforms -> fit the mixing
coefficients on episodic mean FOVs -> write the unmixed movie -> emit a full
``processing.json`` and ``quality_control.json`` per plane.

Unlike the single-plane steps this one processes **two** planes per task, so
it writes one metadata document into each plane's output directory (which is
also what ``main.nf``'s ``capsule/results/*/*/processing.json`` glob expects).

``processing.json`` is written by the shared ``stage_guard``, one per plane, so
a failed run still leaves a document describing what was attempted with the
error recorded in ``notes``.
"""

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py as h5
import numpy as np
from aind_pophys_metadata import processing as shared_processing
from aind_pophys_metadata import find_one, relative_to_root
from aind_pophys_metadata.runner import StageContext, stage_guard
from log_schema.main import setup_logging

from . import decrosstalk as dri
from . import paired_plane_registration as ppr
from . import qc
from .settings import DecrosstalkSettings
from .utils import metadata_utils as mu

logger = logging.getLogger("Paired plane decrosstalk")

PROCESS_NAME = "aind-ophys-decrosstalk-roi-images"
PAIR_GLOB = "pair*/*"
MOTION_CORRECTION_DIR = "motion_correction"
DECROSSTALK_DIR = "decrosstalk"


def _locate_pair(input_dir: Path) -> Tuple[Path, Tuple[str, str]]:
    """Resolve the plane pair staged under ``input_dir``.

    The upstream ``decrosstalk_split_json`` step writes one ``pair*.txt``
    per plane pair; the thin runtime wrapper's ``code/run`` turns it into a
    ``pair*/``
    directory holding one symlink per plane.

    Parameters
    ----------
    input_dir : Path
        The input directory (``../data``).

    Returns
    -------
    tuple
        ``(pair_dir, (oeid, paired_oeid))``.

    Raises
    ------
    FileNotFoundError
        If no ``pair*/`` directory with exactly two planes is present.
    """
    plane_dirs = sorted(p for p in input_dir.glob(PAIR_GLOB) if p.is_dir())
    if len(plane_dirs) != 2:
        raise FileNotFoundError(
            f"Expected exactly 2 plane directories matching "
            f"'{PAIR_GLOB}' under {input_dir}, found "
            f"{[str(p) for p in plane_dirs]}. The upstream pair split must "
            f"run first."
        )
    return plane_dirs[0].parent, (plane_dirs[0].name, plane_dirs[1].name)


def _make_output_dir(results_root: Path, oeid: str) -> Path:
    """Create and return ``<results_root>/<oeid>/decrosstalk``.

    Parameters
    ----------
    results_root : Path
        The results root.
    oeid : str
        Ophys experiment (plane) id.

    Returns
    -------
    Path
        The created output directory.
    """
    output_dir = results_root / oeid / DECROSSTALK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _registered_movie(pair_dir: Path, oeid: str) -> Path:
    """Return the upstream motion-corrected movie for one plane.

    Parameters
    ----------
    pair_dir : Path
        Directory holding both planes' staged input.
    oeid : str
        Ophys experiment (plane) id.

    Returns
    -------
    Path
        Path to ``<oeid>/motion_correction/<oeid>_registered.h5``.
    """
    return pair_dir / oeid / MOTION_CORRECTION_DIR / f"{oeid}_registered.h5"


def _motion_transform_csv(pair_dir: Path, oeid: str) -> Path:
    """Return one plane's own suite2p motion-transform CSV.

    Parameters
    ----------
    pair_dir : Path
        Directory holding both planes' staged input.
    oeid : str
        Ophys experiment (plane) id.

    Returns
    -------
    Path
        Path to ``<oeid>/motion_correction/<oeid>_motion_transform.csv``.
    """
    return (
        pair_dir
        / oeid
        / MOTION_CORRECTION_DIR
        / f"{oeid}_motion_transform.csv"
    )


def _truncate_for_debug(
    h5_file: Path, tmp_dir: Path, frame_rate: float, duration_s: float
) -> Path:
    """Copy the leading ``duration_s`` of a movie into ``tmp_dir``.

    Parameters
    ----------
    h5_file : Path
        Movie to truncate.
    tmp_dir : Path
        Directory the truncated copy is written to.
    frame_rate : float
        Acquisition frame rate [Hz], used to convert seconds to frames.
    duration_s : float
        Duration [s] to retain.

    Returns
    -------
    Path
        Path to the truncated copy.
    """
    requested = int(duration_s * float(frame_rate))
    with h5.File(h5_file, "r") as f:
        available = f["data"].shape[0]
        # The slice clips silently, and the converter has already capped every
        # plane's movie in a debug run, so log what is actually retained.
        n_frames = min(requested, available)
        if requested > available:
            logger.warning(
                "Debug mode: %s holds %s frames, fewer than the %s requested "
                "(%.0f s at %.3f Hz) - movie already shorter than the debug "
                "target, upstream debug cap in effect.",
                h5_file.name,
                available,
                requested,
                duration_s,
                frame_rate,
            )
        logger.info(
            "Debug mode: truncating %s to %s of %s frames",
            h5_file.name,
            n_frames,
            available,
        )
        data = f["data"][:n_frames]
    truncated = Path(tmp_dir) / h5_file.name
    with h5.File(truncated, "w") as f:
        f.create_dataset("data", data=data)
    return truncated


def _prepare_cached_paired_movie(
    settings: DecrosstalkSettings,
    metadata: mu.DecrosstalkMetadata,
    pair_dir: Path,
    oeid: str,
    paired_oeid: str,
    suite2p_params: dict,
) -> Path:
    """Re-register ``oeid``'s raw movie with ``paired_oeid``'s transforms.

    Parameters
    ----------
    settings : DecrosstalkSettings
        Parsed settings.
    metadata : mu.DecrosstalkMetadata
        Metadata read at the start of the run.
    pair_dir : Path
        Directory holding both planes' staged input.
    oeid : str
        Plane whose raw movie is registered.
    paired_oeid : str
        Plane whose motion transforms are applied.
    suite2p_params : dict
        Resolved upstream ``nonrigid`` / ``block_size``.

    Returns
    -------
    Path
        Path to the intermediate registered movie in ``tmp_dir``.

    Raises
    ------
    FileNotFoundError
        If the raw movie or the paired motion transform is missing.
    """
    input_dir = Path(settings.input_dir).resolve()
    raw_movie = find_one(input_dir, f"{oeid}.h5")
    motion_csv = find_one(input_dir, f"{paired_oeid}_motion_transform.csv")

    non_rigid = suite2p_params["nonrigid"]
    transform_df = ppr.get_s2p_motion_transform(
        motion_csv, non_rigid=non_rigid
    )
    if settings.debug:
        raw_movie = _truncate_for_debug(
            raw_movie,
            settings.tmp_dir,
            metadata.frame_rate,
            settings.debug_duration_s,
        )
        with h5.File(raw_movie, "r") as f:
            n_frames = f["data"].shape[0]
        transform_df = transform_df.iloc[:n_frames].reset_index(drop=True)

    return ppr.paired_plane_cached_movie(
        raw_movie,
        transform_df,
        tmp_dir=settings.tmp_dir,
        chunk_size=settings.chunk_size,
        non_rigid=non_rigid,
        block_size=suite2p_params["block_size"],
    )


def _write_decrosstalk_movie(
    settings: DecrosstalkSettings,
    signal_movie: Path,
    paired_movie: Path,
    output_dir: Path,
    oeid: str,
    alpha: float,
    beta: float,
    alpha_list: list,
    beta_list: list,
    mean_norm_mi_list: list,
) -> Path:
    """Unmix the full movie chunk by chunk and write it to ``output_dir``.

    Parameters
    ----------
    settings : DecrosstalkSettings
        Parsed settings, providing ``chunk_size``.
    signal_movie : Path
        This plane's motion-corrected movie.
    paired_movie : Path
        The paired plane's movie registered into this plane's frame.
    output_dir : Path
        Directory the unmixed movie is written to.
    oeid : str
        Ophys experiment (plane) id, used in the output filename.
    alpha : float
        Mean fraction of the signal plane leaking into the paired plane.
    beta : float
        Mean fraction of the paired plane leaking into the signal plane.
    alpha_list, beta_list, mean_norm_mi_list : list
        Per-epoch fit results, stored alongside the movie for QC.

    Returns
    -------
    Path
        Path to the written decrosstalk movie.
    """
    with h5.File(signal_movie, "r") as f:
        signal_shape = f["data"].shape
    with h5.File(paired_movie, "r") as f:
        paired_length = f["data"].shape[0]

    # In debug mode the paired movie was truncated but the upstream registered
    # movie was not, so clamp to the shorter of the two.
    data_length = min(signal_shape[0], paired_length)
    if data_length < signal_shape[0]:
        logger.warning(
            "Paired movie is %s frames but the signal movie is %s; unmixing "
            "the overlapping %s frames only.",
            paired_length,
            signal_shape[0],
            data_length,
        )

    decrosstalk_fn = output_dir / f"{oeid}_decrosstalk.h5"
    start_frames = np.arange(0, data_length, settings.chunk_size)
    end_frames = np.append(start_frames[1:], data_length)

    with h5.File(decrosstalk_fn, "w") as out:
        out.create_dataset(
            "data",
            (data_length, signal_shape[1], signal_shape[2]),
            dtype=np.int16,
            chunks=(min(1000, data_length), signal_shape[1], signal_shape[2]),
        )
        out.create_dataset("alpha_list", data=alpha_list)
        out.create_dataset("beta_list", data=beta_list)
        out.create_dataset("mean_norm_mi_list", data=mean_norm_mi_list)
        out["data"].attrs["alpha"] = alpha
        out["data"].attrs["beta"] = beta

        for start_frame, end_frame in zip(start_frames, end_frames):
            with h5.File(paired_movie, "r") as f:
                paired_data = f["data"][start_frame:end_frame]
            with h5.File(signal_movie, "r") as f:
                signal_data = f["data"][start_frame:end_frame]
            recon = np.zeros_like(signal_data, dtype=np.int16)
            for i in range(signal_data.shape[0]):
                recon[i, :, :] = dri.apply_mixing_matrix(
                    alpha, beta, signal_data[i, :, :], paired_data[i, :, :]
                )[0]
            out["data"][start_frame:end_frame] = recon
    return decrosstalk_fn


def _decrosstalk_plane(
    settings: DecrosstalkSettings,
    pair_dir: Path,
    oeid: str,
    paired_oeid: str,
    output_dir: Path,
    paired_movie: Path,
    paired_emf: Path,
    suite2p_params: dict,
    plane_ctx: StageContext,
    model_files: List[Dict[str, str]],
) -> None:
    """Fit and apply the mixing coefficients for a single plane.

    Parameters
    ----------
    settings : DecrosstalkSettings
        Parsed settings.
    pair_dir : Path
        Directory holding both planes' staged input.
    oeid : str
        Plane being decrosstalked.
    paired_oeid : str
        The other plane of the pair.
    output_dir : Path
        This plane's output directory.
    paired_movie : Path
        The paired plane's full movie in this plane's frame of reference.
    paired_emf : Path
        Episodic mean FOV of ``paired_movie``.
    suite2p_params : dict
        Resolved upstream ``nonrigid`` / ``block_size``.
    plane_ctx : StageContext
        Populated in place so the ``processing.json`` the stage guard writes
        reports partial provenance even when this raises.
    model_files : list of dict
        Cellpose model provenance resolved before the run; empty when the
        weights were not present locally, in which case they are re-read
        afterwards and recorded as a network source.
    """
    num_frames = settings.resolved_num_frames()
    signal_movie = _registered_movie(pair_dir, oeid)

    signal_emf = ppr.episodic_mean_fov(
        signal_movie,
        output_dir,
        max_num_epochs=settings.max_num_epochs,
        num_frames=num_frames,
    )

    logger.info("Fitting mixing coefficients for %s", oeid)
    _, alpha_list, beta_list, mean_norm_mi_list = (
        dri.decrosstalk_roi_image_from_episodic_mean_fov(
            oeid,
            paired_oeid,
            paired_emf,
            pair_dir,
            signal_emf,
            pixel_size=settings.pixel_size,
            grid_interval=settings.grid_interval,
            max_grid_val=settings.max_grid_val,
            motion_buffer=settings.motion_buffer,
            dendrite_diameter_um=settings.dendrite_diameter_um,
            num_top_rois=settings.num_top_rois,
            overlap_threshold=settings.overlap_threshold,
            cellpose_model_type=settings.cellpose_model_type,
        )
    )
    alpha = float(np.mean(alpha_list))
    beta = float(np.mean(beta_list))
    logger.info(
        "%s mixing coefficients over %s epochs: alpha=%.4f beta=%.4f",
        oeid,
        len(alpha_list),
        alpha,
        beta,
    )
    # The resolved upstream suite2p settings ride here rather than in
    # code.parameters: they are read from the upstream document during the
    # run, after the Code block the stage guard was entered with.
    plane_ctx.output_parameters = {
        "alpha_mean": round(alpha, 4),
        "beta_mean": round(beta, 4),
        "num_epochs": len(alpha_list),
        "paired_episodic_mean_fov": relative_to_root(
            Path(settings.output_dir).resolve(), paired_emf
        ),
        **suite2p_params,
    }
    # Nothing was resolvable locally before the run, so whatever is there now
    # arrived over the network. Absent a models directory this stays empty and
    # the key is omitted rather than guessed at.
    if not model_files:
        downloaded = mu.cellpose_model_provenance(
            settings.cellpose_model_type, downloaded=True
        )
        if downloaded:
            plane_ctx.output_parameters[mu.MODEL_FILES_PARAM] = downloaded

    decrosstalk_fn = _write_decrosstalk_movie(
        settings,
        signal_movie,
        paired_movie,
        output_dir,
        oeid,
        alpha,
        beta,
        alpha_list,
        beta_list,
        mean_norm_mi_list,
    )
    ppr.episodic_mean_fov(
        decrosstalk_fn,
        output_dir,
        max_num_epochs=settings.max_num_epochs,
        num_frames=num_frames,
        save_webm=True,
    )
    _write_quality_control(settings, output_dir, oeid, decrosstalk_fn)


def _write_quality_control(
    settings: DecrosstalkSettings,
    output_dir: Path,
    oeid: str,
    decrosstalk_fn: Path,
) -> None:
    """Emit one full quality_control.json for this plane.

    Parameters
    ----------
    settings : DecrosstalkSettings
        Parsed settings, providing the results root.
    output_dir : Path
        This plane's output directory.
    oeid : str
        Ophys experiment (plane) id.
    decrosstalk_fn : Path
        The written decrosstalk movie, whose episodic-mean-FOV webm is the
        QC reference.
    """
    reference = relative_to_root(
        Path(settings.output_dir).resolve(),
        output_dir / f"{decrosstalk_fn.stem}_episodic_mean_fov.webm",
    )
    metric = qc.build_decrosstalk_metric(oeid, reference)
    qc.write_quality_control_json(
        qc.build_quality_control([metric]), output_dir
    )


def _run_decrosstalk_pair(
    settings: DecrosstalkSettings,
    metadata: mu.DecrosstalkMetadata,
    pair_dir: Path,
    oeids: Tuple[str, str],
    output_dirs: Dict[str, Path],
    contexts: Dict[str, StageContext],
    cached_movies: Dict[str, Path],
    model_files: List[Dict[str, str]],
) -> None:
    """Decrosstalk both planes of the staged pair.

    Parameters
    ----------
    settings : DecrosstalkSettings
        Parsed settings.
    metadata : mu.DecrosstalkMetadata
        Metadata read at the start of the run.
    pair_dir : Path
        Directory holding both planes' staged input.
    oeids : tuple of str
        The pair's two plane ids.
    output_dirs : dict of str to Path
        Each plane's output directory.
    contexts : dict of str to StageContext
        Each plane's stage context, populated as the run progresses so the
        emitted document reports partial provenance even when this raises.
    cached_movies : dict of str to Path
        Populated in place with the large paired-registered intermediates so
        the caller can delete them however the run ends.
    model_files : list of dict
        Cellpose model provenance resolved before the run.
    """
    oeid1, oeid2 = oeids
    Path(settings.tmp_dir).mkdir(parents=True, exist_ok=True)

    # Both planes were registered by the same upstream motion-correction run,
    # so one plane's own outputs supply the settings for the pair: block_size
    # from its registered movie's metadata, nonrigid from its own transform
    # CSV columns.
    suite2p_params = mu.resolve_suite2p_params(
        _registered_movie(pair_dir, oeid1),
        _motion_transform_csv(pair_dir, oeid1),
        nonrigid_override=settings.nonrigid,
        block_size_override=settings.block_size_override(),
    )
    logger.info("Upstream suite2p parameters: %s", suite2p_params)

    # cached_movies[X] is plane X's *raw* movie warped by the *other* plane's
    # motion transforms, i.e. X's fluorescence expressed in the other plane's
    # frame of reference. So decrosstalking plane P consumes
    # cached_movies[partner-of-P]. Its episodic mean FOV is written into X's
    # own output directory, matching the established results layout.
    cached_emf: Dict[str, Path] = {}
    for oeid, paired_oeid in ((oeid1, oeid2), (oeid2, oeid1)):
        cached_movies[oeid] = _prepare_cached_paired_movie(
            settings, metadata, pair_dir, oeid, paired_oeid, suite2p_params
        )
        cached_emf[oeid] = ppr.episodic_mean_fov(
            cached_movies[oeid],
            output_dirs[oeid],
            max_num_epochs=settings.max_num_epochs,
            num_frames=settings.resolved_num_frames(),
        )

    for oeid, paired_oeid in ((oeid1, oeid2), (oeid2, oeid1)):
        _decrosstalk_plane(
            settings,
            pair_dir,
            oeid,
            paired_oeid,
            output_dirs[oeid],
            cached_movies[paired_oeid],
            cached_emf[paired_oeid],
            suite2p_params,
            contexts[oeid],
            model_files,
        )


def _cleanup_cached_movies(cached_movies: Iterable[Path]) -> None:
    """Delete the large paired-registered intermediates. Never raises.

    Parameters
    ----------
    cached_movies : iterable of Path
        The intermediates to remove.
    """
    for cached_movie in cached_movies:
        try:
            Path(cached_movie).unlink(missing_ok=True)
            logger.info("Removed intermediate %s", cached_movie)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", cached_movie, exc)


def _enter_stage_guard(
    stack: ExitStack,
    settings: DecrosstalkSettings,
    output_dir: Path,
    plane_id: Optional[str],
    input_data: Optional[List[str]] = None,
    model_files: Optional[List[Dict[str, str]]] = None,
) -> StageContext:
    """Open one plane's stage guard on ``stack`` and return its context.

    Every ``processing.json`` staged into this task is a flat sibling
    upstream, so the dependency graph is collected here rather than inferred
    from a static stage order.

    Parameters
    ----------
    stack : contextlib.ExitStack
        Stack the guard is entered on; it writes the document as the stack
        unwinds, whether or not the run raised.
    settings : DecrosstalkSettings
        Parsed settings; supplies the roots and the recorded parameters.
    output_dir : Path
        Directory this plane's ``processing.json`` is written to.
    plane_id : str or None
        Plane/FOV id, or ``None`` when no plane could be resolved.
    input_data : list of str, optional
        Input data path(s) for provenance.
    model_files : list of dict, optional
        Cellpose model provenance records, recorded alongside the settings so
        the document names the model bytes the run loaded.

    Returns
    -------
    StageContext
        Handle for the run to record output parameters on.
    """
    results_root = Path(settings.output_dir).resolve()
    input_dir = Path(settings.input_dir).resolve()
    parameters = settings.model_dump(mode="json")
    if model_files:
        parameters[mu.MODEL_FILES_PARAM] = model_files
    code = mu.build_code(parameters, input_data=input_data)
    context = stack.enter_context(
        stage_guard(
            output_dir,
            mu.PROCESS_TYPE,
            code,
            plane_id=plane_id,
            logger=logger,
            output_path=relative_to_root(results_root, output_dir),
        )
    )
    context.upstream_names = shared_processing.collect_upstream_process_names(
        input_dir, exclude=mu.process_name(plane_id)
    )
    return context


def run(settings: Optional[DecrosstalkSettings] = None) -> None:
    """Run paired-plane decrosstalk end to end.

    Parameters
    ----------
    settings : DecrosstalkSettings, optional
        Parsed settings; constructed from CLI/env when omitted.
    """
    settings = settings or DecrosstalkSettings()
    # A frame rate is only needed to size the debug truncation, so it is
    # fatal in debug mode alone.
    metadata = mu.load_decrosstalk_metadata(
        settings.input_dir,
        cli_frame_rate=settings.frame_rate,
        require_frame_rate=settings.debug,
    )
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )

    results_root = Path(settings.output_dir).resolve()
    input_dir = Path(settings.input_dir).resolve()
    cached_movies: Dict[str, Path] = {}
    # Resolved before any segmentation runs, so a later download is
    # distinguishable from weights that were already local.
    model_files = mu.cellpose_model_provenance(settings.cellpose_model_type)

    with ExitStack() as stack:
        stack.callback(_cleanup_cached_movies, cached_movies.values())
        try:
            pair_dir, oeids = _locate_pair(input_dir)
        except BaseException:
            # No plane id is known yet, so the one document this run can
            # honestly write goes to the results root.
            _enter_stage_guard(
                stack,
                settings,
                results_root,
                None,
                model_files=model_files,
            )
            raise
        logger.info("Decrosstalking pair %s / %s from %s", *oeids, pair_dir)

        output_dirs = {
            oeid: _make_output_dir(results_root, oeid) for oeid in oeids
        }
        contexts = {
            oeid: _enter_stage_guard(
                stack,
                settings,
                output_dirs[oeid],
                oeid,
                input_data=[
                    relative_to_root(
                        input_dir, _registered_movie(pair_dir, oeid)
                    )
                ],
                model_files=model_files,
            )
            for oeid in oeids
        }
        _run_decrosstalk_pair(
            settings,
            metadata,
            pair_dir,
            oeids,
            output_dirs,
            contexts,
            cached_movies,
            model_files,
        )
