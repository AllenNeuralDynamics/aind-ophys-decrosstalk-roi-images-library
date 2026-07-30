"""aind-data-schema v1/v2 metadata reads for paired-plane decrosstalk.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The v1/v2 dict paths, schema-version dispatch, and the ``Processing``
builders live in the shared library; this module assembles the
``DecrosstalkMetadata`` view, its capsule-specific ``DataProcess``, and the
reader for the upstream motion-correction parameters that paired-plane
registration has to reproduce.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz
from aind_data_schema.core.processing import DataProcess, Processing
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import fields, io
from aind_pophys_metadata import processing as shared_processing

logger = logging.getLogger(__name__)

_SEATTLE_TZ = pytz.timezone("America/Los_Angeles")

CODE_URL = (
    "https://github.com/AllenNeuralDynamics/"
    "aind-ophys-decrosstalk-roi-images/tree/main/code"
)

PROCESSING_JSON = "processing.json"
V1_DATA_PROCESS_GLOB = "*data_process.json"

# The two suite2p settings paired-plane registration must reproduce exactly.
_REQUIRED_SUITE2P_KEYS = ("block_size", "nonrigid")


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


def load_decrosstalk_metadata(input_dir: Path) -> DecrosstalkMetadata:
    """Read v1/v2 metadata for decrosstalk from ``input_dir``.

    Dispatches on which core file is present (``acquisition.json`` -> v2,
    ``session.json`` -> v1). ``frame_rate`` falls back to ``platform.json``
    when the core file lacks it. ``subject_id`` is read from the core file
    (canonical) and falls back to ``subject.json``. ``dataset_name`` comes
    from ``data_description.json``.

    Parameters
    ----------
    input_dir : Path
        Directory searched recursively for the metadata files.

    Returns
    -------
    DecrosstalkMetadata
        The minimal fields this capsule consumes.

    Raises
    ------
    FileNotFoundError
        If neither ``acquisition.json`` nor ``session.json`` is found.
    ValueError
        If both core files are present (ambiguous input).
    """
    input_dir = Path(input_dir)
    core_path = io.find_acquisition_file(input_dir)
    version = io.detect_schema_version(core_path)
    core_raw = io.load_json(core_path)

    platform_path = io.find(input_dir, "platform.json")
    platform_raw = io.load_json(platform_path) if platform_path else None
    subject_path = io.find(input_dir, "subject.json")
    subject_raw = io.load_json(subject_path) if subject_path else None
    dd_path = io.find(input_dir, "data_description.json")
    dd_raw = io.load_json(dd_path) if dd_path else None

    return DecrosstalkMetadata(
        frame_rate=fields.get_frame_rate(version, core_raw, platform_raw),
        instrument_id=fields.get_instrument_id(version, core_raw),
        subject_id=fields.get_subject_id(core_raw, subject_raw),
        dataset_name=fields.get_dataset_name(dd_raw),
    )


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


def build_data_process(
    parameters: dict,
    start_time: dt,
    end_time: dt,
    input_data: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    output_parameters: Optional[dict] = None,
    error: Optional[BaseException] = None,
) -> DataProcess:
    """Construct a v2 DataProcess for decrosstalk.

    Parameters
    ----------
    parameters : dict
        Capsule configuration parameters (from DecrosstalkSettings, plus the
        resolved upstream suite2p settings).
    start_time : dt
        Processing start time.
    end_time : dt
        Processing end time.
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance.
    output_path : str, optional
        Path to this process's output directory, relative to the results root.
    output_parameters : dict, optional
        Actual parameters used and output metrics (the fitted mixing
        coefficients).
    error : Exception, optional
        The exception that terminated the run, or ``None`` on success. The
        message is recorded in ``notes``.

    Returns
    -------
    DataProcess
        The populated decrosstalk data process.
    """
    notes = (
        None
        if error is None
        else f"Run failed with {type(error).__name__}: {error}"
    )
    code = shared_processing.build_code(
        url=CODE_URL,
        name="aind-ophys-decrosstalk-roi-images",
        version=os.getenv("VERSION", ""),
        parameters=parameters,
        input_data=input_data,
    )
    return shared_processing.build_data_process(
        process_type=ProcessName.VIDEO_PLANE_DECROSSTALK,
        code=code,
        start_time=start_time.astimezone(_SEATTLE_TZ),
        end_time=end_time.astimezone(_SEATTLE_TZ),
        output_path=output_path,
        output_parameters=output_parameters,
        notes=notes,
        resources=shared_processing.resource_usage(),
    )


def build_processing(data_processes: List[DataProcess]) -> Processing:
    """Wrap data processes into a full v2 Processing object."""
    return shared_processing.build_processing(data_processes)


def write_processing_json(processing: Processing, output_dir: Path) -> Path:
    """Serialize a Processing object to ``processing.json``.

    Parameters
    ----------
    processing : Processing
        The processing document to write.
    output_dir : Path
        Directory into which ``processing.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    return shared_processing.write_processing_json(processing, output_dir)
