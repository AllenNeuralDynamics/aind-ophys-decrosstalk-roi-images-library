"""Tests for decrosstalk quality_control.json generation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aind_data_schema.core.quality_control import QualityControl, Status

from aind_ophys_decrosstalk_roi_images_library import qc


class TestDecrosstalkMetric(unittest.TestCase):
    """The per-plane decrosstalk dropdown metric."""

    def setUp(self):
        """Build the metric once for the assertions below."""
        self.metric = qc.build_decrosstalk_metric(
            "VISp_0",
            "VISp_0/decrosstalk/VISp_0_decrosstalk_episodic_mean_fov.webm",
        )

    def test_names_the_plane(self):
        """The metric name carries the plane id."""
        self.assertEqual(self.metric.name, "VISp_0 Decrosstalk")

    def test_carries_the_grouping_tags(self):
        """The evaluation tag is what default_grouping groups on."""
        self.assertEqual(self.metric.tags["evaluation"], "Decrosstalk")
        self.assertEqual(self.metric.tags["type"], "Operational QC")

    def test_reference_and_options(self):
        """The reference points at the QC movie; options are pass/fail."""
        self.assertTrue(self.metric.reference.endswith(".webm"))
        self.assertEqual(
            self.metric.value.options, ["Reasonable", "Unreasonable"]
        )
        self.assertEqual(self.metric.value.status, [Status.PASS, Status.FAIL])
        self.assertEqual(self.metric.value.value, "Reasonable")

    def test_awaits_review(self):
        """The metric lands in the reviewer queue rather than auto-passing."""
        self.assertEqual(len(self.metric.status_history), 1)
        self.assertEqual(self.metric.status_history[0].status, Status.PENDING)


class TestQualityControlDocument(unittest.TestCase):
    """The full QualityControl wrapper and its writer."""

    def test_default_grouping(self):
        """The project-standard grouping is applied."""
        metric = qc.build_decrosstalk_metric("VISp_0", "ref.webm")
        document = qc.build_quality_control([metric])
        self.assertEqual(document.default_grouping, ["evaluation"])
        self.assertEqual(len(document.metrics), 1)

    def test_write_round_trips(self):
        """The written document validates back as a full v2 QualityControl."""
        metric = qc.build_decrosstalk_metric("VISp_0", "ref.webm")
        with TemporaryDirectory() as tmp:
            written = qc.write_quality_control_json(
                qc.build_quality_control([metric]), Path(tmp) / "nested"
            )
            self.assertTrue(written.exists())
            QualityControl.model_validate_json(written.read_text())


if __name__ == "__main__":
    unittest.main()
