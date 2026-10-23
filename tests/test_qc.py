"""Tests for decrosstalk quality_control.json generation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from aind_data_schema.core.quality_control import QualityControl, Status

from aind_ophys_decrosstalk_roi_images_library import qc


class TestOrderPairByPlaneIndex(unittest.TestCase):
    """Choosing the lower-indexed plane of a pair."""

    FOV_IDS = ("VISp_0", "VISp_1", "VISl_4", "VISl_5")

    def test_orders_by_plane_index_not_name(self) -> None:
        """The pair arrives in directory-name order, which is not index order.

        ``VISl_4`` sorts before ``VISp_0`` by name, but ``VISp_0`` is the
        lower-indexed plane.
        """
        self.assertEqual(
            qc.order_pair_by_plane_index(("VISl_4", "VISp_0"), self.FOV_IDS),
            ("VISp_0", "VISl_4"),
        )

    def test_already_ordered_pair_is_unchanged(self) -> None:
        """A pair already in index order comes back as given."""
        self.assertEqual(
            qc.order_pair_by_plane_index(("VISp_1", "VISl_5"), self.FOV_IDS),
            ("VISp_1", "VISl_5"),
        )

    def test_plane_absent_from_metadata_raises(self) -> None:
        """A plane the metadata does not list fails loudly, naming it."""
        with self.assertRaisesRegex(ValueError, "VISp_9"):
            qc.order_pair_by_plane_index(("VISp_0", "VISp_9"), self.FOV_IDS)


class TestCoupledFovImage(unittest.TestCase):
    """The side-by-side max-projection PNG."""

    def _write_max_projection(self, path: Path, seed: int) -> Path:
        """Write a max projection the way the registration module does."""
        image = np.random.default_rng(seed).random((32, 32))
        plt.imsave(path, image, cmap="gray")
        return path

    def test_writes_both_planes_side_by_side(self) -> None:
        """The PNG is written, creating its directory, and is landscape."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            max_images = [
                ("VISp_0", self._write_max_projection(tmp / "a.png", 0)),
                ("VISl_4", self._write_max_projection(tmp / "b.png", 1)),
            ]
            written = qc.write_coupled_fov_image(
                max_images, tmp / "VISp_0" / "decrosstalk" / "coupled.png"
            )
            self.assertTrue(written.is_file())
            height, width = plt.imread(written).shape[:2]
            self.assertGreater(width, height)

    def test_missing_projection_raises(self) -> None:
        """A missing projection fails loudly, naming the plane."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            max_images = [
                ("VISp_0", self._write_max_projection(tmp / "a.png", 0)),
                ("VISl_4", tmp / "absent.png"),
            ]
            with self.assertRaisesRegex(FileNotFoundError, "VISl_4"):
                qc.write_coupled_fov_image(max_images, tmp / "coupled.png")


class TestCoupledFovMetric(unittest.TestCase):
    """The per-pair coupled-FOV dropdown metric."""

    def setUp(self) -> None:
        """Build the metric once for the assertions below."""
        self.metric = qc.build_coupled_fov_metric(
            "VISp_0",
            "VISl_4",
            "VISp_0/decrosstalk/VISp_0_VISl_4_coupled_fov_max.png",
        )

    def test_name_leads_with_lower_plane(self) -> None:
        """The name starts with the lower plane id and names both planes."""
        self.assertEqual(self.metric.name, "VISp_0/VISl_4 Coupled FOV")

    def test_evaluation_and_reference(self) -> None:
        """The metric groups under Crosstalk as an operational check."""
        self.assertEqual(self.metric.tags["evaluation"], "Crosstalk")
        self.assertEqual(self.metric.tags["type"], "Operational QC")
        self.assertTrue(self.metric.reference.endswith(".png"))

    def test_description_names_no_plane(self) -> None:
        """The description is generic: it names neither plane of the pair."""
        self.assertEqual(
            self.metric.description,
            "Maximum projections of the two coupled planes after "
            "decrosstalk. Check for residual crosstalk between them.",
        )

    def test_options_map_to_status(self) -> None:
        """Only "Quality is sufficient" passes."""
        self.assertEqual(
            self.metric.value.status,
            [Status.PASS, Status.FAIL, Status.FAIL],
        )
        self.assertEqual(
            self.metric.value.options[0], "Quality is sufficient"
        )

    def test_awaits_review(self) -> None:
        """Pending with no preselected option, so it is not auto-passed."""
        self.assertEqual(self.metric.status_history[0].status, Status.PENDING)
        self.assertEqual(self.metric.value.value, "")

    def test_pair_metric_lives_in_the_lower_document_only(self) -> None:
        """The pair metric sits in the lower document; the higher is empty."""
        lower = qc.build_quality_control([self.metric])
        higher = qc.build_quality_control([])
        self.assertEqual(
            [m.name for m in lower.metrics], ["VISp_0/VISl_4 Coupled FOV"]
        )
        self.assertEqual(higher.metrics, [])


class TestQualityControlDocument(unittest.TestCase):
    """The full QualityControl wrapper and its writer."""

    def test_default_grouping(self) -> None:
        """The project-standard grouping is applied."""
        metric = qc.build_coupled_fov_metric("VISp_0", "VISl_4", "ref.png")
        document = qc.build_quality_control([metric])
        self.assertEqual(document.default_grouping, ["evaluation"])
        self.assertEqual(len(document.metrics), 1)

    def test_write_round_trips(self) -> None:
        """The written document validates back as a full v2 QualityControl."""
        metric = qc.build_coupled_fov_metric("VISp_0", "VISl_4", "ref.png")
        with TemporaryDirectory() as tmp:
            written = qc.write_quality_control_json(
                qc.build_quality_control([metric]), Path(tmp) / "nested"
            )
            self.assertTrue(written.exists())
            QualityControl.model_validate_json(written.read_text())

    def test_empty_document_round_trips(self) -> None:
        """A plane with no metrics still writes a valid, empty document."""
        with TemporaryDirectory() as tmp:
            written = qc.write_quality_control_json(
                qc.build_quality_control([]), Path(tmp)
            )
            document = QualityControl.model_validate_json(written.read_text())
        self.assertEqual(document.metrics, [])


if __name__ == "__main__":
    unittest.main()
