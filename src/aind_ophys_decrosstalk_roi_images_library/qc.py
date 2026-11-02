"""quality_control.json generation for paired-plane decrosstalk.

Library-specific metric content over the shared ``aind_pophys_metadata`` QC
builders: the metric skeleton, the PENDING status, the ``QualityControl``
wrapper with the project-standard ``default_grouping``, and the writer all
live in the shared library. ``job.run`` accumulates the metrics built here and
emits ONE full ``quality_control.json`` per plane (never per-metric files).

Each pair gets one coupled-FOV metric. It lives in the lower-indexed plane's
document only, so the run-level aggregate holds it exactly once; the
higher-indexed plane's document holds no metrics.
"""

from pathlib import Path
from typing import List, Sequence, Tuple

from aind_data_schema.core.quality_control import (
    QCMetric,
    QualityControl,
    Status,
)
from aind_pophys_metadata import quality_control as shared_qc
from matplotlib.figure import Figure
from matplotlib.image import imread

COUPLED_FOV_EVALUATION = "Crosstalk"
COUPLED_FOV_IMAGE_DPI = 150


def order_pair_by_plane_index(
    pair: Tuple[str, str], fov_ids: Sequence[str]
) -> Tuple[str, str]:
    """Return the pair as ``(lower_indexed, higher_indexed)``.

    Parameters
    ----------
    pair : tuple of str
        The two plane ids of the pair, in any order.
    fov_ids : sequence of str
        The session's plane ids, ordered by plane index.

    Returns
    -------
    tuple of str
        The pair, lower plane index first.

    Raises
    ------
    ValueError
        If either plane id is absent from ``fov_ids``.
    """
    missing = [plane for plane in pair if plane not in fov_ids]
    if missing:
        raise ValueError(
            f"Plane(s) {missing} of pair {pair} are not among the session's "
            f"plane ids {tuple(fov_ids)}, so the lower-indexed plane that "
            f"holds the coupled-FOV metric cannot be chosen."
        )
    lower, higher = sorted(pair, key=list(fov_ids).index)
    return lower, higher


def write_coupled_fov_image(
    max_images: Sequence[Tuple[str, Path]], output_path: Path
) -> Path:
    """Write the pair's max projections side by side as one PNG.

    Parameters
    ----------
    max_images : sequence of (str, Path)
        ``(plane_id, max_projection_png)`` for each plane, left to right.
    output_path : Path
        PNG to write; its parent directory is created if needed.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    FileNotFoundError
        If any max projection is missing.
    """
    for plane_id, image_path in max_images:
        if not Path(image_path).is_file():
            raise FileNotFoundError(
                f"Max projection for {plane_id} not found at {image_path}; "
                f"the coupled-FOV image needs both planes' projections."
            )
    figure = Figure(figsize=(5 * len(max_images), 5.4))
    axes = figure.subplots(1, len(max_images), squeeze=False)[0]
    for axis, (plane_id, image_path) in zip(axes, max_images):
        axis.imshow(imread(image_path), cmap="gray")
        axis.set_title(f"{plane_id} max projection")
        axis.axis("off")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=COUPLED_FOV_IMAGE_DPI)
    return output_path


def build_coupled_fov_metric(
    lower_id: str,
    higher_id: str,
    reference_filepath: str,
) -> QCMetric:
    """Construct the coupled-FOV crosstalk QCMetric for one plane pair.

    Parameters
    ----------
    lower_id : str
        Lower-indexed plane id; it leads the metric name.
    higher_id : str
        Higher-indexed plane id.
    reference_filepath : str
        Path to the side-by-side max-projection PNG, relative to the asset.

    Returns
    -------
    QCMetric
        The coupled-FOV metric (pending review).
    """
    return shared_qc.dropdown_metric(
        name=f"{lower_id}/{higher_id} Coupled FOV",
        evaluation=COUPLED_FOV_EVALUATION,
        options=[
            "Quality is sufficient",
            "Timeseries shuffled between planes",
            "Coupled FOV cross-talk",
        ],
        status=[Status.PASS, Status.FAIL, Status.FAIL],
        reference=reference_filepath,
        description=(
            "Maximum projections of the two coupled planes after "
            "decrosstalk. Check for residual crosstalk between them."
        ),
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
