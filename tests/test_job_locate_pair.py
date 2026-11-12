"""Tests for resolving the plane pair staged under the input directory.

A task stages one ``pair<k>.txt`` at the top of the input directory, naming the
two plane directories that sit beside it. A pre-staged ``pair*/`` directory
holding the two planes is still accepted when there is no pair file.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aind_ophys_decrosstalk_roi_images_library import job

# ``VISp_0`` is the lower-indexed plane but ``VISl_4`` sorts first by name,
# and the pair is returned in name order.
PAIR = ("VISp_0", "VISl_4")
NAME_ORDER = ("VISl_4", "VISp_0")
OTHER_PLANE = "VISm_2"
PAIR_FILE = "pair0.txt"


class StagedInput(unittest.TestCase):
    """Base class: an input directory holding one directory per plane."""

    def setUp(self) -> None:
        """Stage three plane directories under a temporary input directory."""
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.input_dir = Path(tmp.name)
        for plane in (*PAIR, OTHER_PLANE):
            (self.input_dir / plane / "motion_correction").mkdir(parents=True)

    def write_pair_file(self, text: str, name: str = PAIR_FILE) -> Path:
        """Write a pair file at the top of the input directory."""
        path = self.input_dir / name
        path.write_text(text)
        return path


class TestLocatePairFile(StagedInput):
    """The pair is read from the ``pair*.txt`` file."""

    def test_planes_come_back_in_name_order(self) -> None:
        """The input directory is the pair directory; planes are sorted."""
        self.write_pair_file("VISp_0\nVISl_4\n")

        pair_dir, planes = job._locate_pair(self.input_dir)

        self.assertEqual(pair_dir, self.input_dir)
        self.assertEqual(planes, NAME_ORDER)

    def test_absolute_entries_under_the_input_dir_name_the_same_planes(
        self,
    ) -> None:
        """An absolute path directly under ``input_dir`` names its plane."""
        first = self.input_dir / "VISp_0"
        second = self.input_dir.resolve() / "VISl_4"
        self.write_pair_file(f"{first}\n{second}\n")

        self.assertEqual(
            job._locate_pair(self.input_dir), (self.input_dir, NAME_ORDER)
        )

    def test_blank_lines_and_surrounding_whitespace_are_ignored(self) -> None:
        """Blank, whitespace-only and CRLF-terminated lines are tolerated."""
        self.write_pair_file("\n  VISp_0  \r\n\n\t\nVISl_4\n\n")

        self.assertEqual(
            job._locate_pair(self.input_dir), (self.input_dir, NAME_ORDER)
        )

    def test_wrong_plane_count_is_rejected(self) -> None:
        """Anything but two distinct planes names the file and the rule."""
        cases = {
            "empty": "",
            "one plane": "VISp_0\n",
            "three planes": f"VISp_0\nVISl_4\n{OTHER_PLANE}\n",
            "one plane twice": f"VISp_0\n{self.input_dir / 'VISp_0'}\n",
        }
        for label, text in cases.items():
            with self.subTest(label):
                self.write_pair_file(text)
                with self.assertRaises(ValueError) as ctx:
                    job._locate_pair(self.input_dir)
                self.assertIn(PAIR_FILE, str(ctx.exception))
                self.assertIn("2 distinct", str(ctx.exception))

    def test_plane_without_a_directory_is_rejected(self) -> None:
        """An absent plane, or one that is a plain file, is named."""
        (self.input_dir / "VISx_9").write_text("not a directory")
        for plane in ("VISq_8", "VISx_9"):
            with self.subTest(plane):
                self.write_pair_file(f"VISp_0\n{plane}\n")
                with self.assertRaises(FileNotFoundError) as ctx:
                    job._locate_pair(self.input_dir)
                self.assertIn(PAIR_FILE, str(ctx.exception))
                self.assertIn(str(self.input_dir / plane), str(ctx.exception))

    def test_absolute_path_outside_the_input_dir_is_rejected(self) -> None:
        """An absolute path must be a directory directly under input_dir."""
        self.write_pair_file("VISp_0\n/mnt/staging/VISl_4\n")

        with self.assertRaises(ValueError) as ctx:
            job._locate_pair(self.input_dir)

        self.assertIn(PAIR_FILE, str(ctx.exception))
        self.assertIn("/mnt/staging/VISl_4", str(ctx.exception))

    def test_entries_that_are_not_plane_names_are_rejected(self) -> None:
        """Nothing outside or below the input directory names a plane."""
        for entry in (
            "..",
            ".",
            "/data/VISl_4",
            "VISl_4/motion_correction",
            str(self.input_dir / "VISl_4" / "motion_correction"),
        ):
            with self.subTest(entry):
                self.write_pair_file(f"VISp_0\n{entry}\n")
                with self.assertRaises(ValueError) as ctx:
                    job._locate_pair(self.input_dir)
                self.assertIn(PAIR_FILE, str(ctx.exception))
                self.assertIn(repr(entry), str(ctx.exception))

    def test_more_than_one_pair_file_is_rejected(self) -> None:
        """Ambiguous staging names every pair file found."""
        self.write_pair_file("VISp_0\nVISl_4\n", "pair0.txt")
        self.write_pair_file(f"VISp_0\n{OTHER_PLANE}\n", "pair1.txt")

        with self.assertRaises(ValueError) as ctx:
            job._locate_pair(self.input_dir)

        self.assertIn("pair0.txt", str(ctx.exception))
        self.assertIn("pair1.txt", str(ctx.exception))


class TestLocatePairDirectory(StagedInput):
    """Without a pair file the pair is a staged ``pair*/`` directory."""

    def stage_pair_directory(self) -> Path:
        """Link each plane of the pair into ``<input_dir>/pair0/``."""
        pair_dir = self.input_dir / "pair0"
        pair_dir.mkdir()
        for plane in PAIR:
            (pair_dir / plane).symlink_to(Path("..") / plane)
        return pair_dir

    def test_pair_directory_is_used_without_a_pair_file(self) -> None:
        """The planes are the directory's entries, sorted by name."""
        pair_dir = self.stage_pair_directory()

        self.assertEqual(
            job._locate_pair(self.input_dir), (pair_dir, NAME_ORDER)
        )

    def test_pair_file_takes_precedence_over_a_pair_directory(self) -> None:
        """When both are staged, the file names the planes."""
        self.stage_pair_directory()
        self.write_pair_file(f"VISp_0\n{OTHER_PLANE}\n")

        self.assertEqual(
            job._locate_pair(self.input_dir),
            (self.input_dir, (OTHER_PLANE, "VISp_0")),
        )

    def test_nothing_staged_names_both_layouts(self) -> None:
        """With neither staged, the error says what was looked for."""
        with self.assertRaises(FileNotFoundError) as ctx:
            job._locate_pair(self.input_dir)

        self.assertIn("pair*.txt", str(ctx.exception))
        self.assertIn("pair*/*", str(ctx.exception))
        self.assertIn(str(self.input_dir), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
