"""Tests for the decrosstalk movie writer.

The coefficient fit hands back numpy scalars off an ``np.arange`` grid and one
list of normalised MI values per epoch, and the movies are HDF5 int16
datasets, so the writer is exercised with those types rather than plain lists.
"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py as h5
import numpy as np

from aind_ophys_decrosstalk_roi_images_library import job
from aind_ophys_decrosstalk_roi_images_library.settings import (
    DecrosstalkSettings,
)

FRAME_SHAPE = (64, 64)
N_FRAMES = 50
N_EPOCHS = 4
# One value per (alpha, beta) point of the grid search.
N_GRID_POINTS = 9


def _make_settings(**kwargs) -> DecrosstalkSettings:
    """Construct DecrosstalkSettings with a clean ``sys.argv``."""
    with patch.object(sys, "argv", ["run_capsule.py"]):
        return DecrosstalkSettings(**kwargs)


def _write_movie(path: Path, n_frames: int = N_FRAMES) -> np.ndarray:
    """Write an int16 HDF5 movie and return the frames written."""
    rng = np.random.default_rng(abs(hash(path.name)) % 2**32)
    data = rng.integers(0, 2000, (n_frames, *FRAME_SHAPE)).astype(np.int16)
    with h5.File(path, "w") as f:
        f.create_dataset("data", data=data)
    return data


def _fit_results():
    """Build per-epoch fit results the way the grid search produces them.

    ``alpha``/``beta`` are numpy scalars taken off the search grid, and each
    epoch contributes a full row of normalised MI values.
    """
    grid = np.arange(0, 0.3 + 0.01, 0.01)
    alpha_list = [grid[i + 1] for i in range(N_EPOCHS)]
    beta_list = [grid[i + 2] for i in range(N_EPOCHS)]
    rng = np.random.default_rng(3)
    mean_norm_mi_list = [
        rng.uniform(0.5, 1.5, N_GRID_POINTS).tolist() for _ in range(N_EPOCHS)
    ]
    return alpha_list, beta_list, mean_norm_mi_list


class TestWriteDecrosstalkMovie(unittest.TestCase):
    """Chunked unmixing of the full movie and the QC datasets beside it."""

    def test_writes_the_unmixed_movie_and_the_fit_results(self):
        """Every frame is unmixed and the per-epoch fit rides along."""
        alpha_list, beta_list, mean_norm_mi_list = _fit_results()
        alpha = float(np.mean(alpha_list))
        beta = float(np.mean(beta_list))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal = root / "VISp_0_registered.h5"
            paired = root / "VISl_1_registered_to_pair.h5"
            signal_data = _write_movie(signal)
            _write_movie(paired)
            written = job._write_decrosstalk_movie(
                _make_settings(chunk_size=13),
                signal,
                paired,
                root,
                "VISp_0",
                alpha,
                beta,
                alpha_list,
                beta_list,
                mean_norm_mi_list,
            )
            with h5.File(written, "r") as f:
                data = f["data"][:]
                stored_alpha = f["alpha_list"][:]
                stored_mi = f["mean_norm_mi_list"][:]
                attrs = dict(f["data"].attrs)
        self.assertEqual(written.name, "VISp_0_decrosstalk.h5")
        self.assertEqual(data.shape, signal_data.shape)
        self.assertEqual(data.dtype, np.int16)
        np.testing.assert_allclose(stored_alpha, alpha_list)
        self.assertEqual(stored_mi.shape, (N_EPOCHS, N_GRID_POINTS))
        self.assertAlmostEqual(attrs["alpha"], alpha)
        self.assertAlmostEqual(attrs["beta"], beta)
        # Unmixing subtracts the paired plane's leak, so the result is not the
        # input movie back.
        self.assertFalse(np.array_equal(data, signal_data))

    def test_numpy_fit_results_are_supported(self):
        """The fit results are also accepted as arrays, not only lists.

        ``np.mean`` over the per-epoch values already returns numpy scalars,
        so nothing here may depend on the truthiness or the list-ness of the
        fitted coefficients.
        """
        alpha_list, beta_list, mean_norm_mi_list = _fit_results()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal = root / "VISp_0_registered.h5"
            paired = root / "VISl_1_registered_to_pair.h5"
            _write_movie(signal)
            _write_movie(paired)
            written = job._write_decrosstalk_movie(
                _make_settings(),
                signal,
                paired,
                root,
                "VISp_0",
                np.mean(alpha_list),
                np.mean(beta_list),
                np.array(alpha_list),
                np.array(beta_list),
                np.array(mean_norm_mi_list),
            )
            with h5.File(written, "r") as f:
                self.assertEqual(f["data"].shape[0], N_FRAMES)
                self.assertEqual(
                    f["mean_norm_mi_list"].shape, (N_EPOCHS, N_GRID_POINTS)
                )

    def test_a_shorter_paired_movie_is_clamped_and_warned(self):
        """A debug run truncates the paired movie but not the signal movie."""
        alpha_list, beta_list, mean_norm_mi_list = _fit_results()
        overlap = 20
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal = root / "VISp_0_registered.h5"
            paired = root / "VISl_1_registered_to_pair.h5"
            _write_movie(signal)
            _write_movie(paired, n_frames=overlap)
            with self.assertLogs(job.logger, level="WARNING") as logs:
                written = job._write_decrosstalk_movie(
                    _make_settings(chunk_size=7),
                    signal,
                    paired,
                    root,
                    "VISp_0",
                    float(np.mean(alpha_list)),
                    float(np.mean(beta_list)),
                    alpha_list,
                    beta_list,
                    mean_norm_mi_list,
                )
            with h5.File(written, "r") as f:
                self.assertEqual(f["data"].shape[0], overlap)
        self.assertIn("overlapping", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
