"""aind-data-schema v1/v2 metadata reads for paired-plane decrosstalk.

A thin, library-specific layer over the shared ``aind_pophys_metadata``
library. The v1/v2 dict paths, schema-version dispatch, and the ``Processing``
builders live in the shared library; this module assembles the
``DecrosstalkMetadata`` view, this library's ``Code`` identity, and the reader
for the upstream motion-correction parameters that paired-plane registration
has to reproduce.
"""

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import (
    MODEL_SOURCE_ASSET,
    MODEL_SOURCE_NETWORK,
    CoreMetadata,
    model_provenance,
)
from aind_pophys_metadata import processing as shared_processing

logger = logging.getLogger(__name__)

CODE_NAME = "aind-ophys-decrosstalk-roi-images"
LIBRARY_NAME = "aind-ophys-decrosstalk-roi-images-library"

# Only this repo knows where its own code lives, so the url is stated here
# rather than derived from the library name.
REPO_URL = (
    "https://github.com/AllenNeuralDynamics/"
    "aind-ophys-decrosstalk-roi-images-library"
)

PROCESS_TYPE = ProcessName.VIDEO_PLANE_DECROSSTALK

# Cellpose's own convention for where it looks for (and caches) weights. The
# thin runtime wrapper points it at the mounted model asset; the library only
# reads it, so no filesystem layout is baked in here.
CELLPOSE_MODELS_PATH_ENV = "CELLPOSE_LOCAL_MODELS_PATH"

# The key the model records land under in code.parameters.
MODEL_FILES_PARAM = "cellpose_model_files"


@dataclass(frozen=True)
class DecrosstalkMetadata:
    """Minimal metadata the decrosstalk step consumes.

    ``frame_rate`` is used only to size the debug truncation; the remaining
    fields are identifiers used for logging and provenance.
    """

    frame_rate: Optional[float] = None
    instrument_id: Optional[str] = None
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None
    fov_ids: Tuple[str, ...] = ()


def load_decrosstalk_metadata(
    input_dir: Path,
    cli_frame_rate: Optional[float] = None,
    require_frame_rate: bool = False,
) -> DecrosstalkMetadata:
    """Read v1/v2 metadata for decrosstalk from ``input_dir``.

    The core-file location, schema-version dispatch and sibling loads all come
    from the shared :meth:`CoreMetadata.load`. ``frame_rate`` falls back to
    ``platform.json`` and then to ``cli_frame_rate``; it is only fatal when
    ``require_frame_rate`` is set, since decrosstalk needs a rate solely to
    size the debug truncation.

    Parameters
    ----------
    input_dir : Path
        Directory searched recursively for the metadata files.
    cli_frame_rate : float, optional
        Operator-supplied frame rate, used when the metadata supplies none.
    require_frame_rate : bool, optional
        Raise when no source supplies a frame rate. Set in debug mode, where
        the truncation cannot be sized without one.

    Returns
    -------
    DecrosstalkMetadata
        The minimal fields this step consumes.

    Raises
    ------
    FileNotFoundError
        If no core metadata file is found.
    ValueError
        If both v1 and v2 core files are present (ambiguous input), or if
        ``require_frame_rate`` and no source supplies one.
    """
    core = CoreMetadata.load(Path(input_dir))
    return DecrosstalkMetadata(
        frame_rate=core.get_frame_rate(
            cli_override=cli_frame_rate,
            required=require_frame_rate,
        ),
        instrument_id=core.get_instrument_id(),
        subject_id=core.get_subject_id(),
        dataset_name=core.get_dataset_name(),
        fov_ids=core.get_fov_ids(),
    )


def process_name(plane_id: Optional[str]) -> Optional[str]:
    """Return the ``DataProcess.name`` this step emits for one plane.

    Mirrors the composition ``build_data_process`` performs, so a document
    can exclude itself from its own dependency graph before the process is
    built.

    Parameters
    ----------
    plane_id : str or None
        Plane/FOV id, or ``None`` when no plane was resolved.

    Returns
    -------
    str or None
        The composed process name, or ``None`` without a plane id (the
        schema then derives the name from the process type).
    """
    if not plane_id:
        return None
    return f"{plane_id}: {PROCESS_TYPE.value}"


def read_upstream_block_size(registered_movie: Path) -> Optional[List[int]]:
    """Read the suite2p ``block_size`` recorded on a registered movie.

    Motion correction stores its full ``suite2p_args`` on the registered
    movie's HDF5 ``metadata`` dataset, so paired-plane registration can
    re-apply the *other* plane's transforms with the exact block layout
    suite2p used. This dataset is the same regardless of schema version, so
    no v1/v2 handling is needed.

    ``nonrigid`` is deliberately NOT read from this dataset: motion
    correction's parameter-search step overwrites its own ``suite2p_args``
    dict with ``nonrigid=False`` before the args are dumped to this metadata,
    so the recorded flag does not reliably reflect what actually ran. See
    :func:`upstream_ran_nonrigid` for the trustworthy source.

    Parameters
    ----------
    registered_movie : Path
        The paired plane's ``*_registered.h5``.

    Returns
    -------
    list of int or None
        The recorded block size, or ``None`` if the movie carries no
        ``metadata`` dataset, no ``suite2p_args``, or no ``block_size``.
    """
    with h5py.File(registered_movie, "r") as f:
        if "metadata" not in f:
            return None
        metadata = json.loads(f["metadata"][()].decode())
    block_size = (metadata.get("suite2p_args") or {}).get("block_size")
    if block_size is None:
        return None
    logger.info(
        "Read upstream block_size %s from %s",
        block_size,
        Path(registered_movie).name,
    )
    return list(block_size)


def upstream_ran_nonrigid(motion_transform_csv: Path) -> bool:
    """Determine whether upstream registration was non-rigid.

    Reads only the CSV header rather than the whole file: the presence of
    the ``nonrigid_x`` / ``nonrigid_y`` columns is definitional (suite2p only
    writes them when non-rigid registration ran), so this is the trustworthy
    source rather than a ``nonrigid`` flag that motion correction's
    parameter-search step can leave stale (see
    :func:`read_upstream_block_size`).

    Parameters
    ----------
    motion_transform_csv : Path
        One plane's own ``*_motion_transform.csv``.

    Returns
    -------
    bool
        Whether the non-rigid offset columns are present.
    """
    with open(motion_transform_csv, newline="") as f:
        header = next(csv.reader(f), [])
    return "nonrigid_x" in header and "nonrigid_y" in header


def resolve_suite2p_params(
    registered_movie: Path,
    motion_transform_csv: Path,
    nonrigid_override: Optional[bool] = None,
    block_size_override: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Resolve ``nonrigid`` and ``block_size``, overrides winning.

    Each value has exactly one authoritative upstream source: ``block_size``
    from the registered movie's recorded ``suite2p_args``, ``nonrigid`` from
    whether the motion-transform CSV carries non-rigid columns. Fails fast:
    a value that is neither overridden nor resolvable upstream is an error,
    because silently guessing it would produce a wrongly registered paired
    movie and therefore wrong mixing coefficients.

    Parameters
    ----------
    registered_movie : Path
        The paired plane's ``*_registered.h5``, read for ``block_size``.
    motion_transform_csv : Path
        The paired plane's own ``*_motion_transform.csv``, read for
        ``nonrigid``.
    nonrigid_override : bool, optional
        Explicit setting; skips the CSV read for this key.
    block_size_override : list of int, optional
        Explicit setting; skips the movie read for this key.

    Returns
    -------
    dict
        ``{"nonrigid": bool, "block_size": list of int}``.

    Raises
    ------
    ValueError
        If ``block_size`` is available from neither source.
    """
    nonrigid = (
        nonrigid_override
        if nonrigid_override is not None
        else upstream_ran_nonrigid(motion_transform_csv)
    )
    block_size = (
        block_size_override
        if block_size_override is not None
        else read_upstream_block_size(registered_movie)
    )
    if block_size is None:
        raise ValueError(
            f"Could not resolve motion-correction parameter 'block_size': "
            f"the registered movie {registered_movie} recorded none, and no "
            f"CLI override was given."
        )
    return {
        "nonrigid": bool(nonrigid),
        "block_size": [int(x) for x in block_size],
    }


def cellpose_model_filenames(model_type: str) -> Tuple[str, ...]:
    """Return every weights file Cellpose resolves for ``model_type``.

    ``Cellpose.__init__`` builds the diameter-estimating ``SizeModel``
    unconditionally, so the size model is resolved alongside the segmentation
    net even when no diameter is requested. Both are named.

    Parameters
    ----------
    model_type : str
        Cellpose pretrained model name, e.g. ``"cyto"``.

    Returns
    -------
    tuple of str
        The segmentation net and size-model file names, in that order.
    """
    return (f"{model_type}torch_0", f"size_{model_type}torch_0.npy")


def cellpose_models_dir() -> Optional[Path]:
    """Return the directory Cellpose resolves its weights from, if declared.

    Read from :data:`CELLPOSE_MODELS_PATH_ENV`; the caller's environment is
    the only source, so the library never assumes a mount point.

    Returns
    -------
    Path or None
        The declared directory, or ``None`` when the variable is unset or
        empty.
    """
    value = os.environ.get(CELLPOSE_MODELS_PATH_ENV)
    return Path(value) if value else None


def cellpose_model_provenance(
    model_type: str,
    downloaded: bool = False,
) -> List[Dict[str, str]]:
    """Describe the Cellpose weights this run loaded, for ``code.parameters``.

    Provenance only: nothing is compared or verified, and a file that is not
    there is omitted rather than fabricated. ``source_name`` is the models
    directory's basename, a stable logical label rather than the resolved
    path.

    Parameters
    ----------
    model_type : str
        Cellpose pretrained model name, e.g. ``"cyto"``.
    downloaded : bool, optional
        ``True`` when the weights were absent before the run and Cellpose
        fetched them, which is recorded as a network source.

    Returns
    -------
    list of dict
        One record per resolved file; empty when no models directory is
        declared or none of the files are present.
    """
    models_dir = cellpose_models_dir()
    if models_dir is None:
        logger.warning(
            "%s is unset; recording no Cellpose model provenance",
            CELLPOSE_MODELS_PATH_ENV,
        )
        return []
    return model_provenance(
        models_dir,
        cellpose_model_filenames(model_type),
        source_name=models_dir.name,
        source=MODEL_SOURCE_NETWORK if downloaded else MODEL_SOURCE_ASSET,
    )


def build_code(
    parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Construct the decrosstalk ``Code`` block.

    Code identity comes from the backing library: the shared builder reads
    the installed ``LIBRARY_NAME`` distribution for the version, which is
    paired with this repo's own :data:`REPO_URL`.

    Parameters
    ----------
    parameters : dict
        Configuration parameters (from ``DecrosstalkSettings``).
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance.

    Returns
    -------
    Code
        The populated code block.
    """
    return shared_processing.build_code(
        name=CODE_NAME,
        library_name=LIBRARY_NAME,
        url=REPO_URL,
        parameters=parameters,
        input_data=input_data,
    )
