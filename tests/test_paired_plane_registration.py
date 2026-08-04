"""Tests for epoch selection in paired_plane_registration."""

import unittest

from aind_ophys_decrosstalk_roi_images_library import (
    paired_plane_registration as ppr,
)


class TestEpochStartFrames(unittest.TestCase):
    """Epoch start/length selection across movie lengths."""

    def test_full_length_movie_uses_max_epochs(self):
        """A long movie is capped by max_num_epochs, not by its length."""
        starts, num_frames = ppr._epoch_start_frames(30000, 10, 1000)
        self.assertEqual(len(starts), 10)
        self.assertEqual(num_frames, 1000)

    def test_epoch_fits_inside_the_movie(self):
        """The last epoch never reads past the end of the movie."""
        for data_length in (2, 100, 999, 3000, 30000):
            starts, num_frames = ppr._epoch_start_frames(data_length, 10, 1000)
            self.assertLessEqual(starts[-1] + num_frames, data_length)

    def test_short_movie_yields_one_shortened_epoch(self):
        """A debug-truncated movie gets one epoch instead of raising.

        100 frames against a 300-frame request previously raised, which made
        the capsule unrunnable with --debug true.
        """
        starts, num_frames = ppr._epoch_start_frames(100, 10, 300)
        self.assertEqual(starts, [25])
        self.assertEqual(num_frames, 50)

    def test_num_frames_never_drops_below_one(self):
        """A one-frame movie still asks for a single frame, not zero.

        num_frames of 0 makes the chunked projection return None, which fails
        far from the cause.
        """
        starts, num_frames = ppr._epoch_start_frames(1, 10, 1000)
        self.assertEqual(num_frames, 1)
        self.assertEqual(starts, [0])

    def test_epoch_count_is_at_least_one(self):
        """start_frames is never empty, whatever the ratio works out to."""
        for data_length in (1, 2, 50, 100, 299):
            starts, _ = ppr._epoch_start_frames(data_length, 10, 300)
            self.assertGreaterEqual(len(starts), 1)

    def test_empty_movie_raises(self):
        """Zero frames is unprojectable and fails with a clear message."""
        with self.assertRaises(ValueError) as ctx:
            ppr._epoch_start_frames(0, 10, 1000)
        self.assertIn("nothing to project", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
