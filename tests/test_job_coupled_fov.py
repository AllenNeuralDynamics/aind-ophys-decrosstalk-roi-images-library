"""Tests for the per-pair coupled-FOV QC wiring in ``job``."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

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


if __name__ == "__main__":
    unittest.main()
