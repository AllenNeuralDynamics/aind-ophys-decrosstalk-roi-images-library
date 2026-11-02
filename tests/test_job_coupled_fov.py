"""Tests for the per-pair coupled-FOV QC wiring in ``job``."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from aind_data_schema.core.quality_control import QualityControl

from aind_ophys_decrosstalk_roi_images_library import job
from aind_ophys_decrosstalk_roi_images_library.settings import (
    DecrosstalkSettings,
)


def _make_settings(**kwargs: Any) -> DecrosstalkSettings:
    """Construct DecrosstalkSettings with a clean ``sys.argv``."""
    with patch.object(sys, "argv", ["run_capsule.py"]):
        return DecrosstalkSettings(**kwargs)


class TestCoupledFovMetric(unittest.TestCase):
    """The pair image lands with the lower-indexed plane's document."""

    def test_image_in_lower_plane_dir_with_root_relative_reference(
        self,
    ) -> None:
        """The PNG sits in the lower plane's dir; the reference is relative."""
        with TemporaryDirectory() as tmp:
            results_root = Path(tmp)
            output_dirs = {}
            for seed, plane in enumerate(("VISp_0", "VISl_4")):
                output_dirs[plane] = job._make_output_dir(results_root, plane)
                image = np.random.default_rng(seed).random((32, 32))
                plt.imsave(
                    output_dirs[plane] / f"{plane}_decrosstalk_max_img.png",
                    image,
                    cmap="gray",
                )
            settings = _make_settings(output_dir=results_root)

            metric = job._coupled_fov_metric(
                settings, output_dirs, "VISp_0", "VISl_4"
            )

            expected = "VISp_0/decrosstalk/VISp_0_VISl_4_coupled_fov_max.png"
            self.assertEqual(metric.reference, expected)
            self.assertTrue((results_root / expected).is_file())
            self.assertFalse(
                any(output_dirs["VISl_4"].glob("*coupled_fov*"))
            )


class TestPairQualityControl(unittest.TestCase):
    """The two documents ``_run_decrosstalk_pair`` writes for a pair."""

    def test_only_the_lower_plane_holds_the_coupled_fov_metric(self) -> None:
        """The higher-indexed plane's document is written without metrics.

        The pair arrives in directory-name order, which is not plane-index
        order: ``VISl_4`` sorts first but ``VISp_0`` is the lower plane.
        """
        planes = ("VISl_4", "VISp_0")
        with TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            output_dirs = {}
            for seed, plane in enumerate(planes):
                output_dirs[plane] = job._make_output_dir(results_root, plane)
                image = np.random.default_rng(seed).random((32, 32))
                plt.imsave(
                    output_dirs[plane] / f"{plane}_decrosstalk_max_img.png",
                    image,
                    cmap="gray",
                )
            settings = _make_settings(
                output_dir=results_root, tmp_dir=Path(tmp) / "scratch"
            )

            with (
                patch.object(
                    job.mu, "resolve_suite2p_params", return_value={}
                ),
                patch.object(
                    job, "_prepare_cached_paired_movie", return_value=Path("m")
                ),
                patch.object(
                    job.ppr, "episodic_mean_fov", return_value=Path("e")
                ),
                patch.object(job, "_decrosstalk_plane") as decrosstalk_plane,
            ):
                job._run_decrosstalk_pair(
                    settings,
                    SimpleNamespace(fov_ids=("VISp_0", "VISl_4")),
                    Path(tmp),
                    planes,
                    output_dirs,
                    dict.fromkeys(planes),
                    {},
                    [],
                )

            documents = {
                plane: QualityControl.model_validate_json(
                    (output_dirs[plane] / "quality_control.json").read_text()
                )
                for plane in planes
            }

        self.assertEqual(decrosstalk_plane.call_count, 2)
        self.assertEqual(
            [m.name for m in documents["VISp_0"].metrics],
            ["VISp_0/VISl_4 Coupled FOV"],
        )
        self.assertEqual(documents["VISl_4"].metrics, [])


if __name__ == "__main__":
    unittest.main()
