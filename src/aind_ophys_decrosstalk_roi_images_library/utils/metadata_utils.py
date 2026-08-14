"""aind-data-schema v1/v2 metadata reads for paired-plane decrosstalk.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The v1/v2 dict paths, schema-version dispatch, and the ``Processing``
builders live in the shared library; this module assembles the
``DecrosstalkMetadata`` view, this capsule's ``Code`` identity, and the reader
for the upstream motion-correction parameters that paired-plane registration
has to reproduce.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import (
    MODEL_SOURCE_ASSET,
    MODEL_SOURCE_NETWORK,
    CoreMetadata,
    io,
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

PROCESSING_JSON = "processing.json"
V1_DATA_PROCESS_GLOB = "*data_process.json"

# The two suite2p settings paired-plane registration must reproduce exactly.
_REQUIRED_SUITE2P_KEYS = ("block_size", "nonrigid")

# Cellpose's own convention for where it looks for (and caches) weights. The
# capsule wrapper points it at the mounted model asset; the library only reads
# it, so no filesystem layout is baked in here.
CELLPOSE_MODELS_PATH_ENV = "CELLPOSE_LOCAL_MODELS_PATH"

# The key the model records land under in code.parameters.
MODEL_FILES_PARAM = "cellpose_model_files"


@dataclass(frozen=True)
class DecrosstalkMetadata:
    """Minimal metadata the decrosstalk capsule consumes.

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
        The minimal fields this capsule consumes.

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
    """Return the ``DataProcess.name`` this capsule emits for one plane.

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


def _suite2p_params_from_v2(processing_raw: dict) -> Dict[str, Any]:
    """Extract the wanted suite2p settings from a v2 ``processing.json`` dict.

    The v2 motion-correction document records its settings flat on
    ``data_processes[*].code.parameters`` (there is no ``suite2p_args``
    nesting, which is where the v1 document kept them).

    Parameters
    ----------
    processing_raw : dict
        Raw ``processing.json`` dict. An empty dict (raw assets ship a
        two-byte ``{}``) yields no keys rather than raising.

    Returns
    -------
    dict
        The subset of :data:`_REQUIRED_SUITE2P_KEYS` that was found.
    """
    found: Dict[str, Any] = {}
    for process in processing_raw.get("data_processes") or []:
        parameters = (process.get("code") or {}).get("parameters") or {}
        for key in _REQUIRED_SUITE2P_KEYS:
            if key not in found and parameters.get(key) is not None:
                found[key] = parameters[key]
    return found


def _suite2p_params_from_v1(data_process_raw: dict) -> Dict[str, Any]:
    """Extract the wanted suite2p settings from a v1 ``DataProcess`` dict.

    v1 motion correction nested them under
    ``parameters.suite2p_args``; a flat ``parameters`` is also accepted.

    Parameters
    ----------
    data_process_raw : dict
        Raw ``*data_process.json`` dict.

    Returns
    -------
    dict
        The subset of :data:`_REQUIRED_SUITE2P_KEYS` that was found.
    """
    parameters = data_process_raw.get("parameters") or {}
    sources = [parameters.get("suite2p_args") or {}, parameters]
    found: Dict[str, Any] = {}
    for source in sources:
        for key in _REQUIRED_SUITE2P_KEYS:
            if key not in found and source.get(key) is not None:
                found[key] = source[key]
    return found


def read_upstream_suite2p_params(plane_input_dir: Path) -> Dict[str, Any]:
    """Read the upstream motion-correction suite2p settings for one plane.

    Paired-plane registration re-applies the *other* plane's suite2p motion
    transforms, so it must use the same ``block_size`` and ``nonrigid``
    settings suite2p itself used. Prefers the v2 ``processing.json`` and falls
    back to a v1 ``*data_process.json``.

    Parameters
    ----------
    plane_input_dir : Path
        A single plane's input directory (searched recursively), normally
        containing the upstream ``motion_correction`` output.

    Returns
    -------
    dict
        Mapping with whichever of ``block_size`` / ``nonrigid`` were found;
        may be empty when no upstream document exists.
    """
    plane_input_dir = Path(plane_input_dir)
    for name, extractor in (
        (PROCESSING_JSON, _suite2p_params_from_v2),
        (V1_DATA_PROCESS_GLOB, _suite2p_params_from_v1),
    ):
        for path in sorted(plane_input_dir.rglob(name)):
            params = extractor(io.load_json(path))
            if params:
                logger.info(
                    "Read upstream suite2p parameters %s from %s",
                    sorted(params),
                    path.name,
                )
                return params
    return {}


def resolve_suite2p_params(
    plane_input_dir: Path,
    nonrigid_override: Optional[bool] = None,
    block_size_override: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Resolve ``nonrigid`` and ``block_size``, overrides winning.

    Fails fast: a value that is neither overridden nor present upstream is an
    error, because silently guessing it would produce a wrongly registered
    paired movie and therefore wrong mixing coefficients.

    Parameters
    ----------
    plane_input_dir : Path
        A single plane's input directory, searched for upstream metadata.
    nonrigid_override : bool, optional
        Explicit setting; skips the upstream read for this key.
    block_size_override : list of int, optional
        Explicit setting; skips the upstream read for this key.

    Returns
    -------
    dict
        ``{"nonrigid": bool, "block_size": list of int}``.

    Raises
    ------
    ValueError
        If a required value is available from neither source.
    """
    overrides = {
        "nonrigid": nonrigid_override,
        "block_size": block_size_override,
    }
    missing_keys = [k for k, v in overrides.items() if v is None]
    upstream = (
        read_upstream_suite2p_params(plane_input_dir) if missing_keys else {}
    )

    resolved: Dict[str, Any] = {}
    unresolved = []
    for key in _REQUIRED_SUITE2P_KEYS:
        if overrides[key] is not None:
            resolved[key] = overrides[key]
        elif key in upstream:
            resolved[key] = upstream[key]
        else:
            unresolved.append(key)
    if unresolved:
        raise ValueError(
            f"Could not resolve upstream motion-correction parameter(s) "
            f"{unresolved}: no {PROCESSING_JSON} (v2) or "
            f"{V1_DATA_PROCESS_GLOB} (v1) under {plane_input_dir} supplied "
            f"them, and no CLI override was given."
        )
    resolved["block_size"] = [int(x) for x in resolved["block_size"]]
    resolved["nonrigid"] = bool(resolved["nonrigid"])
    return resolved


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
        Capsule configuration parameters (from ``DecrosstalkSettings``).
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
