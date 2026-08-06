"""Tests for the debug-mode movie truncation.

The logged frame count has to be the one actually retained. In a debug run the
converter has already capped each plane's movie (100 frames), so a seconds-
based target computed from the frame rate can be far larger than the dataset;
``f["data"][:n]`` then clips silently and the log claimed up to 38x more frames
than any stage processed.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py as h5
import numpy as np

from aind_ophys_decrosstalk_roi_images_library import job

UPSTREAM_DEBUG_FRAMES = 100
FRAME_RATE_HZ = 10.710
DEBUG_DURATION_S = 360.0


def _write_movie(path: Path, n_frames: int) -> Path:
    """Write a tiny HDF5 movie of ``n_frames`` frames and return its path."""
    with h5.File(path, "w") as f:
        f.create_dataset("data", data=np.zeros((n_frames, 4, 4), np.int16))
    return path


class TestTruncateForDebug(unittest.TestCase):
    """The truncation reports the effective frame count, never the intent."""

    def test_target_beyond_the_dataset_is_capped_and_warned(self):
        """An already-capped movie logs its real length, not the request."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            movie = _write_movie(root / "VISl_6.h5", UPSTREAM_DEBUG_FRAMES)
            out_dir = root / "scratch"
            out_dir.mkdir()
            with self.assertLogs(job.logger, level="WARNING") as logs:
                truncated = job._truncate_for_debug(
                    movie, out_dir, FRAME_RATE_HZ, DEBUG_DURATION_S
                )
            with h5.File(truncated, "r") as f:
                self.assertEqual(f["data"].shape[0], UPSTREAM_DEBUG_FRAMES)

        message = "\n".join(logs.output)
        self.assertIn("upstream debug cap", message)
        self.assertIn(str(UPSTREAM_DEBUG_FRAMES), message)

    def test_target_inside_the_dataset_truncates_without_warning(self):
        """The ordinary case still retains duration_s of frames."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            movie = _write_movie(root / "VISp_0.h5", 200)
            out_dir = root / "scratch"
            out_dir.mkdir()
            with self.assertNoLogs(job.logger, level="WARNING"):
                truncated = job._truncate_for_debug(movie, out_dir, 10.0, 5.0)
            with h5.File(truncated, "r") as f:
                self.assertEqual(f["data"].shape[0], 50)


if __name__ == "__main__":
    unittest.main()
