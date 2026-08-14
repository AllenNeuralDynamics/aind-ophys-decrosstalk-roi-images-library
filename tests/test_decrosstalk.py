"""Tests for the mask filtering and unmixing steps of decrosstalk.

Everything a compute step hands to these functions is built as the production
type: cellpose returns an integer label image, the mean FOVs are float arrays
projected from the movie, the full-movie unmixing passes int16 frames straight
out of HDF5, and the motion-correction crop ranges come from the DataFrame
``read_csv`` returns. The cellpose segmentation itself is not exercised here --
it needs the local model weights and is covered by the Code Ocean data run.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from aind_ophys_decrosstalk_roi_images_library import decrosstalk as dri

FRAME_SHAPE = (64, 64)
# Wide enough to survive the dendrite area threshold (~129 px at the default
# 10 um diameter and 0.78 um pixel size).
ROI_WIDTH = 14


def _label_image(
    origins: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Build a cellpose-style label image with one square ROI per origin.

    Cellpose returns a 2D integer label image, ids running ``1..N``.
    """
    masks = np.zeros(FRAME_SHAPE, dtype=np.uint16)
    for roi_id, (y, x) in enumerate(origins, start=1):
        masks[y : y + ROI_WIDTH, x : x + ROI_WIDTH] = roi_id
    return masks


def _mean_fov(masks: np.ndarray) -> np.ndarray:
    """Build a mean-FOV image whose ROI brightness rises with the ROI id.

    Mirrors the float projection ``episodic_mean_fov`` writes.
    """
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 10, FRAME_SHAPE).astype(np.float32)
    for roi_id in range(1, int(masks.max()) + 1):
        image[masks == roi_id] += 100.0 * roi_id
    return image


def _write_motion_transform_csv(
    pair_dir: Path,
    oeid: str,
    y_offsets: Sequence[int],
    x_offsets: Sequence[int],
) -> Path:
    """Write one plane's motion-transform CSV where decrosstalk reads it."""
    directory = pair_dir / oeid / "motion_correction"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{oeid}_motion_transform.csv"
    pd.DataFrame(
        {
            "framenumber": np.arange(len(y_offsets)),
            "x": np.asarray(x_offsets),
            "y": np.asarray(y_offsets),
        }
    ).to_csv(path, index=False)
    return path


class TestMotionCorrectionCropRange(unittest.TestCase):
    """Crop ranges derived from the upstream rigid offsets."""

    def test_ranges_cover_the_observed_offsets(self) -> None:
        """The crop discards the band the frames were rolled through."""
        with TemporaryDirectory() as tmp:
            pair_dir = Path(tmp)
            _write_motion_transform_csv(
                pair_dir, "VISp_0", [-4, 0, 6], [-2, 0, 3]
            )
            range_y, range_x = dri.get_motion_correction_crop_xy_range(
                "VISp_0", pair_dir
            )
        self.assertEqual(range_y, [4, -6])
        self.assertEqual(range_x, [2, -3])

    def test_intersection_of_both_planes(self) -> None:
        """The pair is cropped to the band valid for both planes."""
        with TemporaryDirectory() as tmp:
            pair_dir = Path(tmp)
            _write_motion_transform_csv(pair_dir, "VISp_0", [-4, 6], [-2, 3])
            _write_motion_transform_csv(pair_dir, "VISl_1", [-7, 2], [-1, 8])
            range_y, range_x = (
                dri.get_motion_correction_crop_xy_range_from_both_planes(
                    "VISp_0", "VISl_1", pair_dir
                )
            )
        self.assertEqual(range_y, [7, -6])
        self.assertEqual(range_x, [2, -8])


class TestMaskFiltering(unittest.TestCase):
    """Dendrite, border and duplicate filtering of the label image."""

    def test_dendrite_sized_masks_are_dropped(self) -> None:
        """A mask smaller than a dendrite disc is zeroed, ids left as-is."""
        masks = _label_image([(20, 20), (40, 40)])
        masks[masks == 2] = 0
        masks[40:44, 40:44] = 2
        filtered = dri.filter_dendrite(masks)
        self.assertEqual(np.setdiff1d(np.unique(filtered), 0).tolist(), [1])
        self.assertEqual(filtered.dtype, masks.dtype)

    def test_border_masks_are_dropped(self) -> None:
        """An ROI touching the border band is zeroed."""
        masks = _label_image([(20, 20), (0, 40)])
        filtered = dri.filter_border_roi(masks)
        self.assertEqual(np.setdiff1d(np.unique(filtered), 0).tolist(), [1])

    def test_reorder_makes_the_ids_contiguous(self) -> None:
        """Filtering leaves gaps that the range-based loops would skip."""
        masks = _label_image([(10, 10), (30, 30), (10, 40)])
        masks[masks == 2] = 0
        reordered = dri.reorder_mask(masks)
        self.assertEqual(
            np.setdiff1d(np.unique(reordered), 0).tolist(), [1, 2]
        )
        np.testing.assert_array_equal(reordered == 2, masks == 3)


class TestIntensityRanking(unittest.TestCase):
    """Ranking and top-N selection over a real mean-FOV image."""

    def test_ranks_brightest_first(self) -> None:
        """Rank 0 is the brightest ROI in the image."""
        masks = _label_image([(10, 10), (30, 30), (10, 40)])
        ranks, roi_inds = dri.get_ranks_roi_inds(_mean_fov(masks), masks)
        np.testing.assert_array_equal(roi_inds, [1, 2, 3])
        np.testing.assert_array_equal(ranks, [2, 1, 0])

    def test_keeps_only_the_brightest_rois(self) -> None:
        """The dimmest ROI is zeroed when more than num_top_rois survive."""
        masks = _label_image([(10, 10), (30, 30), (10, 40)])
        top = dri.get_top_intensity_mask(
            _mean_fov(masks), masks, num_top_rois=2
        )
        self.assertEqual(np.setdiff1d(np.unique(top), 0).tolist(), [2, 3])

    def test_a_short_mask_is_returned_unchanged(self) -> None:
        """Fewer ROIs than requested means no selection to make."""
        masks = _label_image([(10, 10), (30, 30)])
        top = dri.get_top_intensity_mask(
            _mean_fov(masks), masks, num_top_rois=15
        )
        np.testing.assert_array_equal(top, masks)


class TestBoundingBox(unittest.TestCase):
    """Expansion of each ROI into the box the MI comparison uses."""

    def test_one_box_per_roi_extends_the_tight_mask(self) -> None:
        """The stack has an entry per ROI, each larger than its ROI."""
        masks = _label_image([(20, 20), (40, 40)])
        bb_masks = dri.get_bounding_box(masks)
        self.assertEqual(bb_masks.shape, (2, *FRAME_SHAPE))
        self.assertEqual(bb_masks.dtype, np.uint16)
        for i, roi_id in enumerate((1, 2)):
            self.assertGreater(
                np.sum(bb_masks[i] > 0), np.sum(masks == roi_id)
            )
            # np.where over a box is how the MI is sampled; a box is a
            # filled rectangle, so the sample count is its area.
            box_y, box_x = np.where(bb_masks[i])
            self.assertEqual(
                len(box_y),
                (box_y.max() - box_y.min() + 1)
                * (box_x.max() - box_x.min() + 1),
            )

    def test_boxes_are_clipped_to_the_image(self) -> None:
        """An ROI against the edge cannot extend past the FOV."""
        masks = _label_image([(0, 0)])
        bb_masks = dri.get_bounding_box(masks)
        self.assertEqual(np.where(bb_masks[0])[0].min(), 0)
        self.assertLess(np.where(bb_masks[0])[0].max(), FRAME_SHAPE[0])

    def test_an_empty_mask_yields_an_empty_stack(self) -> None:
        """No surviving ROI still concatenates with the other plane's boxes."""
        empty = np.zeros(FRAME_SHAPE, dtype=np.uint16)
        bb_masks = dri.get_bounding_box(empty)
        self.assertEqual(bb_masks.shape, (0, *FRAME_SHAPE))
        self.assertEqual(
            np.concatenate(
                [bb_masks, dri.get_bounding_box(_label_image([(20, 20)]))]
            ).shape,
            (1, *FRAME_SHAPE),
        )


class TestApplyMixingMatrix(unittest.TestCase):
    """Unmixing a pair of frames with the fitted coefficients."""

    def test_int16_movie_frames_are_unmixed(self) -> None:
        """The full-movie loop passes int16 frames read out of HDF5."""
        rng = np.random.default_rng(1)
        signal = rng.integers(0, 2000, FRAME_SHAPE).astype(np.int16)
        paired = rng.integers(0, 2000, FRAME_SHAPE).astype(np.int16)
        recon_signal, recon_paired = dri.apply_mixing_matrix(
            0.05, 0.07, signal, paired
        )
        self.assertEqual(recon_signal.shape, FRAME_SHAPE)
        self.assertEqual(recon_paired.shape, FRAME_SHAPE)
        # Round-tripping through the mixing matrix returns the input.
        remixed_signal = 0.95 * recon_signal + 0.07 * recon_paired
        np.testing.assert_allclose(remixed_signal, signal, atol=1e-6)

    def test_zero_coefficients_are_a_no_op(self) -> None:
        """An epoch that fitted nothing leaves the frames untouched."""
        rng = np.random.default_rng(2)
        signal = rng.uniform(0, 100, FRAME_SHAPE).astype(np.float32)
        paired = rng.uniform(0, 100, FRAME_SHAPE).astype(np.float32)
        recon_signal, recon_paired = dri.apply_mixing_matrix(
            0.0, 0.0, signal, paired
        )
        np.testing.assert_allclose(recon_signal, signal, atol=1e-4)
        np.testing.assert_allclose(recon_paired, paired, atol=1e-4)


class TestDrawMasksOnImage(unittest.TestCase):
    """The inspection overlay, on real image and mask arrays."""

    def test_creates_its_own_figure(self) -> None:
        """Called without axes it returns the figure it made."""
        masks = _label_image([(20, 20)])
        result = dri.draw_masks_on_image(_mean_fov(masks), masks)
        self.assertEqual(len(result), 2)

    def test_draws_on_supplied_axes(self) -> None:
        """Given axes it returns nothing and draws in place."""
        from matplotlib import pyplot as plt

        masks = _label_image([(20, 20)])
        _, ax = plt.subplots()
        self.assertIsNone(
            dri.draw_masks_on_image(_mean_fov(masks), masks, ax=ax)
        )


class TestModelDownloadFallback(unittest.TestCase):
    """Cellpose's own local-first resolution is left in place."""

    def test_downloader_is_not_patched(self) -> None:
        """Importing the module must not disable the network fallback."""
        from cellpose import utils as cp_utils

        self.assertEqual(
            cp_utils.download_url_to_file.__module__, "cellpose.utils"
        )


if __name__ == "__main__":
    unittest.main()
