"""Tests for DecrosstalkSettings."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from aind_ophys_decrosstalk_roi_images_library.settings import (
    DecrosstalkSettings,
)


def _make_settings(**kwargs) -> DecrosstalkSettings:
    """Construct DecrosstalkSettings with a clean ``sys.argv``.

    ``cli_parse_args=True`` would otherwise consume the test runner's own
    arguments and abort with SystemExit.
    """
    with patch.object(sys, "argv", ["run_capsule.py"]):
        return DecrosstalkSettings(**kwargs)


class TestDecrosstalkSettings(unittest.TestCase):
    """Settings parsing, coercion and derived accessors."""

    def test_defaults(self):
        """Defaults match the Code Ocean capsule layout."""
        settings = _make_settings()
        self.assertEqual(settings.input_dir, Path("../data/"))
        self.assertEqual(settings.output_dir, Path("../results/"))
        self.assertEqual(settings.tmp_dir, Path("../scratch/"))
        self.assertFalse(settings.debug)
        self.assertFalse(settings.verify)
        self.assertIsNone(settings.nonrigid)
        self.assertIsNone(settings.frame_rate)
        self.assertIsNone(settings.block_size)

    def test_block_size_from_json_string(self):
        """A CLI/env string is parsed into a list of ints."""
        settings = _make_settings(block_size="[128, 256]")
        self.assertEqual(settings.block_size, [128, 256])
        self.assertEqual(settings.block_size_override(), [128, 256])

    def test_block_size_from_string_elements(self):
        """List elements that arrive as strings are coerced to ints."""
        settings = _make_settings(block_size=["64", "64"])
        self.assertEqual(settings.block_size, [64, 64])

    def test_block_size_empty_means_unset(self):
        """An empty block_size reads as "no override"."""
        for value in ([], "", None):
            with self.subTest(value=value):
                settings = _make_settings(block_size=value)
                self.assertIsNone(settings.block_size)
                self.assertIsNone(settings.block_size_override())

    def test_block_size_rejects_non_numeric(self):
        """A non-numeric block_size fails validation rather than silently."""
        with self.assertRaises((ValidationError, SystemExit)):
            _make_settings(block_size="[128, oops]")

    def test_resolved_num_frames(self):
        """Debug mode swaps in the shorter per-epoch frame count."""
        self.assertEqual(
            _make_settings(debug=False).resolved_num_frames(), 1000
        )
        self.assertEqual(_make_settings(debug=True).resolved_num_frames(), 300)
        self.assertEqual(
            _make_settings(
                debug=True, debug_num_frames=42
            ).resolved_num_frames(),
            42,
        )

    def test_nonrigid_override_accepted(self):
        """The upstream override is a real tri-state, not a bool default."""
        self.assertIs(_make_settings(nonrigid=False).nonrigid, False)
        self.assertIs(_make_settings(nonrigid=True).nonrigid, True)
        self.assertIsNone(_make_settings().nonrigid)

    def test_no_single_character_field_names(self):
        """No field may be 1 character: it renders as an unusable short flag.

        pydantic-settings turns a one-character field into ``-x``, which a Code
        Ocean app panel (``--name=value`` only) can never emit.
        """
        short = [f for f in DecrosstalkSettings.model_fields if len(f) < 2]
        self.assertEqual(short, [])

    def test_every_field_is_cli_addressable(self):
        """Every field parses as ``--name=value``, the only form CO emits.

        Guards the app-panel contract at the source rather than relying on
        auto-app-panel, which happily emits an unusable ``param_name``.
        """
        samples = {
            "input_dir": "../data/",
            "output_dir": "../results/",
            "tmp_dir": "../scratch/",
            "debug": "0",
            "debug_duration_s": "360.0",
            "frame_rate": "9.48",
            "num_frames": "1000",
            "debug_num_frames": "300",
            "max_num_epochs": "10",
            "nonrigid": "1",
            "block_size": "[128, 128]",
            "chunk_size": "5000",
            "pixel_size": "0.78",
            "grid_interval": "0.01",
            "max_grid_val": "0.3",
            "motion_buffer": "5",
            "dendrite_diameter_um": "10.0",
            "num_top_rois": "15",
            "overlap_threshold": "0.7",
            "cellpose_model_type": "cyto",
            "verify": "0",
        }
        self.assertEqual(
            sorted(samples), sorted(DecrosstalkSettings.model_fields)
        )
        argv = ["run_capsule.py"] + [
            f"--{name}={value}" for name, value in samples.items()
        ]
        with patch.object(sys, "argv", argv):
            settings = DecrosstalkSettings()
        self.assertEqual(settings.block_size, [128, 128])
        self.assertTrue(settings.nonrigid)


if __name__ == "__main__":
    unittest.main()
