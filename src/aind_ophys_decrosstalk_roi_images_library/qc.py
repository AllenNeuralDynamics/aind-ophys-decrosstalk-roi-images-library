"""quality_control.json generation for paired-plane decrosstalk.

Capsule-specific metric content over the shared ``aind_pophys_metadata`` QC
builders: the metric skeleton, the PENDING status, the ``QualityControl``
wrapper with the project-standard ``default_grouping``, and the writer all
live in the shared library. ``job.run`` accumulates the metrics built here and
emits ONE full ``quality_control.json`` per plane (never per-metric files).
"""

from pathlib import Path
from typing import List

from aind_data_schema.core.quality_control import (
    QCMetric,
    QualityControl,
    Status,
)
from aind_pophys_metadata import quality_control as shared_qc


def build_decrosstalk_metric(
    unique_id: str,
    reference_filepath: str,
) -> QCMetric:
    """Construct the decrosstalk QCMetric for one plane.

    Parameters
    ----------
    unique_id : str
        Plane/FOV id used in the metric name.
    reference_filepath : str
        Path to the decrosstalk episodic-mean-FOV movie, relative to the
        asset.

    Returns
    -------
    QCMetric
        The decrosstalk metric (pending review).
    """
    return shared_qc.dropdown_metric(
        name=f"{unique_id} Decrosstalk",
        evaluation="Decrosstalk",
        options=["Reasonable", "Unreasonable"],
        status=[Status.PASS, Status.FAIL],
        reference=reference_filepath,
        description=(
            "Review the episodic mean FOV of the decrosstalk movie to ensure "
            "the paired-plane signal has been removed without introducing "
            "artifacts."
        ),
        value="Reasonable",
    )


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap metrics into a full QualityControl with the standard grouping."""
    return shared_qc.build_quality_control(metrics)


def write_quality_control_json(
    quality_control: QualityControl,
    output_dir: Path,
) -> Path:
    """Serialize a QualityControl object to ``quality_control.json``.

    Parameters
    ----------
    quality_control : QualityControl
        The QC document to write.
    output_dir : Path
        Directory into which ``quality_control.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    return shared_qc.write_quality_control_json(quality_control, output_dir)
