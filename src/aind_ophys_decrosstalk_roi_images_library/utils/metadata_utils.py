"""aind-data-schema v1/v2 metadata reads for paired-plane decrosstalk.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The v1/v2 dict paths, schema-version dispatch, and the ``Processing``
builders live in the shared library; this module assembles the
``DecrosstalkMetadata`` view, this capsule's ``Code`` identity, and the reader
for the upstream motion-correction parameters that paired-plane registration
has to reproduce.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import fields, io
from aind_pophys_metadata import processing as shared_processing

logger = logging.getLogger(__name__)

CODE_NAME = "aind-ophys-decrosstalk-roi-images"
LIBRARY_NAME = "aind-ophys-decrosstalk-roi-images-library"

PROCESS_TYPE = ProcessName.VIDEO_PLANE_DECROSSTALK

PROCESSING_JSON = "processing.json"
V1_DATA_PROCESS_GLOB = "*data_process.json"

# The two suite2p settings paired-plane registration must reproduce exactly.
_REQUIRED_SUITE2P_KEYS = ("block_size", "nonrigid")


@dataclass(frozen=True)
class DecrosstalkMetadata:
    """Minimal metadata the decrosstalk capsule consumes.

    ``frame_rate`` is used only to size the debug truncation; the remaining
    fields are identifiers used for logging, provenance and the plane-id
    cross-check.
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
    from the shared ``load_common`` preamble. ``frame_rate`` falls back to
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
        If neither ``acquisition.json`` nor ``session.json`` is found.
    ValueError
        If both core files are present (ambiguous input), or if
        ``require_frame_rate`` and no source supplies one.
    """
    input_dir = Path(input_dir)
    common = io.load_common(input_dir)
    return DecrosstalkMetadata(
        frame_rate=fields.resolve_frame_rate(
            common.version,
            common.core_raw,
            common.platform_raw,
            core_file=common.core_path,
            input_dir=input_dir,
            cli_override=cli_frame_rate,
            required=require_frame_rate,
        ),
        instrument_id=fields.get_instrument_id(
            common.version, common.core_raw
        ),
        subject_id=fields.get_subject_id(
            common.core_raw, common.subject_raw
        ),
        dataset_name=fields.get_dataset_name(common.data_description_raw),
        fov_ids=fields.get_fov_ids(common.version, common.core_raw),
    )


def validate_pair_ids(
    scavenged: Iterable[str],
    canonical: Iterable[str],
) -> bool:
    """Warn when a pair's directory-scavenged plane ids are not canonical.

    Decrosstalk scavenges its two plane ids from the staged ``pair*/<id>``
    directory names. It sees one pair out of the session's planes, so the
    canonical set is compared *restricted to that pair*: a whole-session
    comparison would report a divergence on every multiplane run, which is
    noise rather than signal. An id absent from the canonical set survives the
    restriction and is what the warning names.

    Never raises: uniqueness is the hard requirement and the directory names
    already guarantee it, so canonicality only buys readability.

    Parameters
    ----------
    scavenged : iterable of str
        The pair's plane ids, taken from the input directory layout.
    canonical : iterable of str
        Every canonical plane id of the session (see
        :func:`aind_pophys_metadata.fields.get_fov_ids`).

    Returns
    -------
    bool
        ``True`` when the scavenged ids are all canonical, or when the
        metadata yielded no canonical ids to compare against.
    """
    scavenged = list(scavenged)
    canonical = list(canonical)
    if not canonical:
        logger.warning(
            "No canonical plane ids in the acquisition metadata; keeping the "
            "scavenged ids %s unchecked.",
            sorted(scavenged),
        )
        return True
    expected = [name for name in canonical if name in set(scavenged)]
    return fields.validate_scavenged_ids(scavenged, expected, logger)


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


def build_code(
    parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Construct the decrosstalk ``Code`` block.

    Code identity comes from the backing library: the shared builder reads
    the installed ``LIBRARY_NAME`` distribution for the version and derives
    the url from it, so the two always describe one artifact.

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
        parameters=parameters,
        input_data=input_data,
    )
