"""Mixing-coefficient estimation and unmixing for paired-plane crosstalk.

Two simultaneously-imaged mesoscope planes bleed into each other. The mixing
is modelled as ``[[1-alpha, beta], [alpha, 1-beta]]``; ``alpha`` and ``beta``
are fitted per epoch by minimising the normalised mutual information between
the two planes' reconstructed ROI images, then averaged and applied to the
full movie.

ROIs are detected with Cellpose on the episodic mean FOV images. Cellpose
looks for its weights under ``CELLPOSE_LOCAL_MODELS_PATH`` first and downloads
them when they are absent. Which bytes were used is recorded in the emitted
``code.parameters`` by ``metadata_utils.cellpose_model_provenance``.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
import skimage
from cellpose import models as cp_models

mpl.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_PIXEL_SIZE_UM = 0.78
DEFAULT_DENDRITE_DIAMETER_UM = 10.0
DEFAULT_NUM_TOP_ROIS = 15
DEFAULT_OVERLAP_THRESHOLD = 0.7
DEFAULT_MOTION_BUFFER = 5
DEFAULT_GRID_INTERVAL = 0.01
DEFAULT_MAX_GRID_VAL = 0.3

# Border band, inside the already motion-cropped image, whose ROIs are
# discarded. Distinct from DEFAULT_MOTION_BUFFER (which crops the image
# itself) even though both happen to be 5 pixels.
DEFAULT_ROI_BORDER_BUFFER = 5


def get_motion_correction_crop_xy_range(
    oeid: str, pair_dir: Path
) -> Tuple[List[int], List[int]]:
    """Get x-y ranges that crop away motion-correction frame rolling.

    Parameters
    ----------
    oeid : str
        Ophys experiment (plane) id.
    pair_dir : Path
        Directory containing ``<oeid>/motion_correction/`` for both planes.

    Returns
    -------
    tuple of list of int
        ``(range_y, range_x)``, each ``[start, end]`` pixel index.
    """
    motion_transform_csv = (
        Path(pair_dir)
        / oeid
        / "motion_correction"
        / f"{oeid}_motion_transform.csv"
    )
    motion_df = pd.read_csv(motion_transform_csv)
    max_y = np.ceil(max(motion_df.y.max(), 1)).astype(int)
    min_y = np.floor(min(motion_df.y.min(), 0)).astype(int)
    max_x = np.ceil(max(motion_df.x.max(), 1)).astype(int)
    min_x = np.floor(min(motion_df.x.min(), 0)).astype(int)
    return [-min_y, -max_y], [-min_x, -max_x]


def get_motion_correction_crop_xy_range_from_both_planes(
    oeid: str, paired_id: str, pair_dir: Path
) -> Tuple[List[int], List[int]]:
    """Intersect both planes' motion-correction crop ranges.

    Parameters
    ----------
    oeid : str
        Ophys experiment (plane) id of the signal plane.
    paired_id : str
        Ophys experiment (plane) id of the paired plane.
    pair_dir : Path
        Directory containing both planes' ``motion_correction`` output.

    Returns
    -------
    tuple of list of int
        ``(range_y, range_x)`` valid for both planes, each ``[start, end]``
        pixel index.
    """
    yrange_og, xrange_og = get_motion_correction_crop_xy_range(oeid, pair_dir)
    yrange_paired, xrange_paired = get_motion_correction_crop_xy_range(
        paired_id, pair_dir
    )
    xrange = [
        max(xrange_og[0], xrange_paired[0]),
        min(xrange_og[1], xrange_paired[1]),
    ]
    yrange = [
        max(yrange_og[0], yrange_paired[0]),
        min(yrange_og[1], yrange_paired[1]),
    ]
    return yrange, xrange


def decrosstalk_roi_image_from_episodic_mean_fov(
    oeid: str,
    paired_id: str,
    paired_reg_emf_fn: Path,
    pair_dir: Path,
    signal_emf_fn: Path,
    pixel_size: float = DEFAULT_PIXEL_SIZE_UM,
    grid_interval: float = DEFAULT_GRID_INTERVAL,
    max_grid_val: float = DEFAULT_MAX_GRID_VAL,
    motion_buffer: int = DEFAULT_MOTION_BUFFER,
    dendrite_diameter_um: float = DEFAULT_DENDRITE_DIAMETER_UM,
    num_top_rois: int = DEFAULT_NUM_TOP_ROIS,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    cellpose_model_type: str = "cyto",
    return_recon: bool = False,
) -> Tuple[Optional[np.ndarray], list, list, list]:
    """Fit mixing coefficients across every epoch of the mean FOV movies.

    Parameters
    ----------
    oeid : str
        Ophys experiment (plane) id of the signal plane.
    paired_id : str
        Ophys experiment (plane) id of the paired plane.
    paired_reg_emf_fn : Path
        Episodic mean FOV of the paired plane registered to this plane.
    pair_dir : Path
        Directory containing both planes' ``motion_correction`` output, used
        for the motion-correction crop ranges.
    signal_emf_fn : Path
        Episodic mean FOV of this plane's own registered movie.
    pixel_size : float, optional
        Imaging plane pixel size [um].
    grid_interval : float, optional
        Step of the alpha/beta grid search.
    max_grid_val : float, optional
        Maximum alpha/beta value searched.
    motion_buffer : int, optional
        Extra pixels cropped from each edge.
    dendrite_diameter_um : float, optional
        Diameter [um] below which a mask is discarded as dendrite.
    num_top_rois : int, optional
        Number of highest-intensity ROIs retained per plane.
    overlap_threshold : float, optional
        IoU above which two masks are treated as the same duplicated cell.
    cellpose_model_type : str, optional
        Cellpose pretrained model name.
    return_recon : bool, optional
        Also reconstruct the unmixed mean FOV stack using the averaged
        coefficients. Off by default because the capsule unmixes the full
        movie instead.

    Returns
    -------
    tuple
        ``(recon_signal_data, alpha_list, beta_list, mean_norm_mi_list)``.
        ``recon_signal_data`` is ``None`` unless ``return_recon``.
    """
    with h5py.File(signal_emf_fn, "r") as f:
        data_length = f["data"].shape[0]

    alpha_list = []
    beta_list = []
    mean_norm_mi_list = []
    for start_frame in range(data_length):
        (
            alpha,
            beta,
            mean_norm_mi_values,
        ) = decrosstalk_roi_image_single_pair_from_episodic_mean_fov(
            oeid,
            paired_id,
            paired_reg_emf_fn,
            pair_dir,
            signal_emf_fn,
            start_frame,
            pixel_size,
            motion_buffer=motion_buffer,
            grid_interval=grid_interval,
            max_grid_val=max_grid_val,
            dendrite_diameter_um=dendrite_diameter_um,
            num_top_rois=num_top_rois,
            overlap_threshold=overlap_threshold,
            cellpose_model_type=cellpose_model_type,
        )
        alpha_list.append(alpha)
        beta_list.append(beta)
        mean_norm_mi_list.append(mean_norm_mi_values)

    recon_signal_data = None
    if return_recon:
        alpha = np.mean(alpha_list)
        beta = np.mean(beta_list)
        with h5py.File(signal_emf_fn, "r") as f:
            signal_data = f["data"][:]
        with h5py.File(paired_reg_emf_fn, "r") as f:
            paired_data = f["data"][:]
        recon_signal_data = np.zeros_like(signal_data)
        for i in range(data_length):
            recon_signal_data[i, :, :] = apply_mixing_matrix(
                alpha, beta, signal_data[i, :, :], paired_data[i, :, :]
            )[0]
    return recon_signal_data, alpha_list, beta_list, mean_norm_mi_list


def decrosstalk_roi_image_single_pair_from_episodic_mean_fov(
    oeid: str,
    paired_id: str,
    paired_reg_emf_fn: Path,
    pair_dir: Path,
    signal_emf_fn: Path,
    start_frame: int,
    pix_size: float,
    motion_buffer: int = DEFAULT_MOTION_BUFFER,
    grid_interval: float = DEFAULT_GRID_INTERVAL,
    max_grid_val: float = DEFAULT_MAX_GRID_VAL,
    dendrite_diameter_um: float = DEFAULT_DENDRITE_DIAMETER_UM,
    num_top_rois: int = DEFAULT_NUM_TOP_ROIS,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    cellpose_model_type: str = "cyto",
) -> Tuple[float, float, list]:
    """Fit mixing coefficients for a single epoch's pair of mean images.

    Parameters
    ----------
    oeid : str
        Ophys experiment (plane) id of the signal plane.
    paired_id : str
        Ophys experiment (plane) id of the paired plane.
    paired_reg_emf_fn : Path
        Episodic mean FOV of the paired plane registered to this plane.
    pair_dir : Path
        Directory containing both planes' ``motion_correction`` output.
    signal_emf_fn : Path
        Episodic mean FOV of this plane's own registered movie.
    start_frame : int
        Epoch index within the episodic mean FOV stacks.
    pix_size : float
        Imaging plane pixel size [um].
    motion_buffer : int, optional
        Extra pixels cropped from each edge.
    grid_interval : float, optional
        Step of the alpha/beta grid search.
    max_grid_val : float, optional
        Maximum alpha/beta value searched.
    dendrite_diameter_um : float, optional
        Diameter [um] below which a mask is discarded as dendrite.
    num_top_rois : int, optional
        Number of highest-intensity ROIs retained per plane.
    overlap_threshold : float, optional
        IoU above which two masks are treated as the same duplicated cell.
    cellpose_model_type : str, optional
        Cellpose pretrained model name.

    Returns
    -------
    tuple
        ``(alpha, beta, mean_norm_mi_values)`` for this epoch.
    """
    with h5py.File(signal_emf_fn, "r") as f:
        signal_mean = f["data"][start_frame : start_frame + 1].mean(axis=0)
    with h5py.File(paired_reg_emf_fn, "r") as f:
        paired_mean = f["data"][start_frame : start_frame + 1].mean(axis=0)

    p1y, p1x = get_motion_correction_crop_xy_range_from_both_planes(
        oeid, paired_id, pair_dir
    )
    crop = (
        slice(p1y[0] + motion_buffer, p1y[1] - motion_buffer),
        slice(p1x[0] + motion_buffer, p1x[1] - motion_buffer),
    )
    signal_mean = signal_mean[crop]
    paired_mean = paired_mean[crop]

    signal_top_masks, paired_top_masks = get_signal_paired_top_masks(
        signal_mean,
        paired_mean,
        dendrite_diameter_um=dendrite_diameter_um,
        pix_size=pix_size,
        overlap_threshold=overlap_threshold,
        num_top_rois=num_top_rois,
        cellpose_model_type=cellpose_model_type,
    )
    bb_masks = np.concatenate(
        [
            get_bounding_box(signal_top_masks),
            get_bounding_box(paired_top_masks),
        ]
    )
    if len(bb_masks) == 0:
        # Both planes segmented to nothing, so the grid search has no signal to
        # minimise and returns alpha=beta=0 (i.e. no unmixing at all). Loud,
        # because a silently uncorrected epoch drags the averaged coefficients
        # toward zero.
        logger.warning(
            "No ROIs survived filtering for %s epoch %s; this epoch "
            "contributes alpha=beta=0 to the averaged coefficients.",
            oeid,
            start_frame,
        )

    # Raw mutual information per ROI, used to normalise the grid search.
    mi_raw = np.zeros(len(bb_masks))
    for bi, mask in enumerate(bb_masks):
        bb_yx = np.where(mask)
        mi_raw[bi] = skimage.metrics.normalized_mutual_information(
            signal_mean[bb_yx], paired_mean[bb_yx]
        )

    data = np.vstack((signal_mean.ravel(), paired_mean.ravel()))
    grid = np.arange(0, max_grid_val + grid_interval, grid_interval)
    ab_pair = []
    mean_norm_mi_values = []
    for alpha in grid:
        for beta in grid:
            temp_mixing = np.array([[1 - alpha, beta], [alpha, 1 - beta]])
            temp_unmixed_data = np.dot(np.linalg.inv(temp_mixing), data)
            temp_recon_signal = temp_unmixed_data[0, :].reshape(
                signal_mean.shape
            )
            temp_recon_paired = temp_unmixed_data[1, :].reshape(
                signal_mean.shape
            )
            temp_mi = np.zeros(len(bb_masks))
            for bi, mask in enumerate(bb_masks):
                bb_yx = np.where(mask)
                temp_mi[bi] = skimage.metrics.normalized_mutual_information(
                    temp_recon_signal[bb_yx], temp_recon_paired[bb_yx]
                )
            mean_norm_mi_values.append((temp_mi / mi_raw).mean())
            ab_pair.append([alpha, beta])

    alpha, beta = ab_pair[np.argmin(mean_norm_mi_values)]
    return alpha, beta, np.array(mean_norm_mi_values).tolist()


def get_signal_paired_top_masks(
    signal_mean: np.ndarray,
    paired_mean: np.ndarray,
    dendrite_diameter_um: float = DEFAULT_DENDRITE_DIAMETER_UM,
    pix_size: float = DEFAULT_PIXEL_SIZE_UM,
    border_buffer: int = DEFAULT_ROI_BORDER_BUFFER,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    num_top_rois: int = DEFAULT_NUM_TOP_ROIS,
    cellpose_model_type: str = "cyto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment both mean images and keep the top-intensity ROIs.

    Cellpose masks are filtered for dendrites and border ROIs, then
    de-duplicated: excessive crosstalk can make the same cell appear in both
    planes, so where two masks overlap above ``overlap_threshold`` the one
    ranking lower in intensity within its own plane is dropped.

    Parameters
    ----------
    signal_mean : np.ndarray
        Mean image of the signal plane.
    paired_mean : np.ndarray
        Mean image of the paired plane.
    dendrite_diameter_um : float, optional
        Diameter [um] below which a mask is discarded as dendrite.
    pix_size : float, optional
        Imaging plane pixel size [um].
    border_buffer : int, optional
        Border width [pixels] within which ROIs are discarded.
    overlap_threshold : float, optional
        IoU above which two masks are treated as the same duplicated cell.
    num_top_rois : int, optional
        Number of highest-intensity ROIs retained per plane.
    cellpose_model_type : str, optional
        Cellpose pretrained model name.

    Returns
    -------
    tuple of np.ndarray
        ``(signal_top_masks, paired_top_masks)``.
    """
    model = cp_models.Cellpose(gpu=False, model_type=cellpose_model_type)
    dendrite_diameter_px = dendrite_diameter_um / pix_size

    def _masks(image: np.ndarray) -> np.ndarray:
        """Segment one mean image and filter dendrites and border ROIs."""
        masks, _, _, _ = model.eval(image, diameter=None, channels=[0, 0])
        masks = filter_dendrite(
            masks, dendrite_diameter_pix=dendrite_diameter_px
        )
        masks = filter_border_roi(masks, buffer_pix=border_buffer)
        return reorder_mask(masks)

    signal_masks_filtered = _masks(signal_mean)
    paired_masks_filtered = _masks(paired_mean)

    num_signal_masks = np.max(signal_masks_filtered)
    num_paired_masks = np.max(paired_masks_filtered)
    overlap_matrix = np.zeros((num_signal_masks, num_paired_masks))
    for i in range(1, num_signal_masks + 1):
        for j in range(1, num_paired_masks + 1):
            overlap_matrix[i - 1, j - 1] = np.sum(
                (signal_masks_filtered == i) & (paired_masks_filtered == j)
            ) / np.sum(
                (signal_masks_filtered == i) | (paired_masks_filtered == j)
            )

    signal_ranks, _ = get_ranks_roi_inds(signal_mean, signal_masks_filtered)
    paired_ranks, _ = get_ranks_roi_inds(paired_mean, paired_masks_filtered)

    signal_masks_overlap_filtered = signal_masks_filtered.copy()
    paired_masks_overlap_filtered = paired_masks_filtered.copy()
    for si, pi in zip(*np.where(overlap_matrix > overlap_threshold)):
        if signal_ranks[si] < paired_ranks[pi]:
            paired_masks_overlap_filtered[
                paired_masks_overlap_filtered == pi + 1
            ] = 0
        else:
            signal_masks_overlap_filtered[
                signal_masks_overlap_filtered == si + 1
            ] = 0

    signal_top_masks = get_top_intensity_mask(
        signal_mean, signal_masks_overlap_filtered, num_top_rois=num_top_rois
    )
    paired_top_masks = get_top_intensity_mask(
        paired_mean, paired_masks_overlap_filtered, num_top_rois=num_top_rois
    )
    return signal_top_masks, paired_top_masks


def filter_dendrite(
    masks: np.ndarray,
    dendrite_diameter_pix: float = (
        DEFAULT_DENDRITE_DIAMETER_UM / DEFAULT_PIXEL_SIZE_UM
    ),
) -> np.ndarray:
    """Discard masks smaller than a dendrite-sized disc.

    Parameters
    ----------
    masks : np.ndarray
        2D mask image; each ROI has a unique integer value.
    dendrite_diameter_pix : float, optional
        Dendrite diameter [pixels].

    Returns
    -------
    np.ndarray
        The mask image with sub-threshold ROIs zeroed. Remaining ROI ids are
        not renumbered (see :func:`reorder_mask`).
    """
    area_threshold = np.pi * (dendrite_diameter_pix / 2) ** 2
    filtered_mask = masks.copy()
    for roi_id in range(1, np.max(masks) + 1):
        roi_mask = masks == roi_id
        if np.sum(roi_mask) < area_threshold:
            filtered_mask[roi_mask] = 0
    return filtered_mask


def filter_border_roi(
    masks: np.ndarray, buffer_pix: int = DEFAULT_ROI_BORDER_BUFFER
) -> np.ndarray:
    """Discard masks touching a border band of the FOV.

    Parameters
    ----------
    masks : np.ndarray
        2D mask image; each ROI has a unique integer value.
    buffer_pix : int, optional
        Border band width [pixels].

    Returns
    -------
    np.ndarray
        The mask image with border-touching ROIs zeroed.
    """
    border_mask = np.zeros(masks.shape, dtype=bool)
    border_mask[:buffer_pix, :] = 1
    border_mask[-buffer_pix:, :] = 1
    border_mask[:, :buffer_pix] = 1
    border_mask[:, -buffer_pix:] = 1

    filtered_mask = masks.copy()
    for roi_id in range(1, np.max(masks) + 1):
        roi_mask = masks == roi_id
        if np.any(roi_mask & border_mask):
            filtered_mask[roi_mask] = 0
    return filtered_mask


def get_top_intensity_mask(
    img: np.ndarray, mask: np.ndarray, num_top_rois: int = DEFAULT_NUM_TOP_ROIS
) -> np.ndarray:
    """Keep only the ``num_top_rois`` brightest ROIs.

    Parameters
    ----------
    img : np.ndarray
        Image whose per-ROI mean intensity ranks the ROIs.
    mask : np.ndarray
        2D ROI mask image.
    num_top_rois : int, optional
        Number of ROIs to keep. When the mask holds fewer, it is returned
        unchanged.

    Returns
    -------
    np.ndarray
        The mask image with all but the brightest ROIs zeroed.
    """
    roi_inds = np.setdiff1d(np.unique(mask), 0)
    if len(roi_inds) < num_top_rois:
        return mask
    mean_intensities = [img[mask == i].mean() for i in roi_inds]
    top_ids = roi_inds[np.argsort(mean_intensities)[::-1][:num_top_rois]]
    top_mask = mask.copy()
    for i in roi_inds:
        if i not in top_ids:
            top_mask[mask == i] = 0
    return top_mask


def get_ranks_roi_inds(
    img: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank a mask's ROIs by mean intensity, brightest first.

    Parameters
    ----------
    img : np.ndarray
        2D image whose per-ROI mean intensity is ranked.
    mask : np.ndarray
        2D ROI mask image.

    Returns
    -------
    tuple of np.ndarray
        ``(ranks, roi_inds)``; ``ranks[k]`` is the 0-based rank of
        ``roi_inds[k]`` (0 = brightest).
    """
    roi_inds = np.setdiff1d(np.unique(mask), 0)
    mean_intensities = np.array([img[mask == i].mean() for i in roi_inds])
    ranks = np.zeros_like(mean_intensities)
    ranks[np.argsort(mean_intensities)[::-1]] = np.arange(
        len(mean_intensities)
    )
    return ranks, roi_inds


def reorder_mask(mask: np.ndarray) -> np.ndarray:
    """Renumber a mask's ROI ids to a contiguous ``1..N``.

    Run this after filtering, so downstream loops over ``range(1, max+1)``
    visit every surviving ROI.

    Parameters
    ----------
    mask : np.ndarray
        2D ROI mask image with possibly non-contiguous ids.

    Returns
    -------
    np.ndarray
        The renumbered mask image.
    """
    mask_reordered = np.zeros_like(mask)
    for i, ind in enumerate(np.setdiff1d(np.unique(mask), 0)):
        mask_reordered[mask == ind] = i + 1
    return mask_reordered


def get_bounding_box(
    masks: np.ndarray, area_extension_factor: float = 2
) -> np.ndarray:
    """Expand each ROI to a rectangular bounding box.

    The mutual-information comparison uses boxes rather than the tight masks
    so that surrounding background is included, which is what makes the
    crosstalk visible.

    Parameters
    ----------
    masks : np.ndarray
        2D ROI mask image.
    area_extension_factor : float, optional
        Approximate factor by which each box's area exceeds its ROI's,
        assuming a circular ROI.

    Returns
    -------
    np.ndarray
        3D stack of one box mask per ROI (boxes may overlap).
    """
    bb_extension = np.sqrt(area_extension_factor * np.pi / 4)
    mask_inds = np.setdiff1d(np.unique(masks), 0)

    bb_masks = np.zeros((len(mask_inds), *masks.shape), dtype=np.uint16)
    for i, mask_i in enumerate(mask_inds):
        y, x = np.where(masks == mask_i)
        bb_y = _extend_bounds(y.min(), y.max(), bb_extension, masks.shape[0])
        bb_x = _extend_bounds(x.min(), x.max(), bb_extension, masks.shape[1])
        bb_masks[i, bb_y[0] : bb_y[1], bb_x[0] : bb_x[1]] = mask_i
    return bb_masks


def _extend_bounds(
    low: int, high: int, extension: float, limit: int
) -> List[int]:
    """Symmetrically extend a tight ``[low, high]`` span, clipped to ``limit``.

    Parameters
    ----------
    low, high : int
        Tight bounds along one axis.
    extension : float
        Fraction of the tight span added to each side.
    limit : int
        Exclusive upper bound of the axis.

    Returns
    -------
    list of int
        The extended ``[start, end]`` indices.
    """
    span = high - low
    return [
        max(0, int(np.round(low - extension * span / 2))),
        min(limit, int(np.round(high + extension * span / 2))),
    ]


def apply_mixing_matrix(
    alpha: float,
    beta: float,
    signal_mean: np.ndarray,
    paired_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unmix a signal/paired image pair with the fitted coefficients.

    Parameters
    ----------
    alpha : float
        Fraction of the signal plane leaking into the paired plane.
    beta : float
        Fraction of the paired plane leaking into the signal plane.
    signal_mean : np.ndarray
        Image of the signal plane.
    paired_mean : np.ndarray
        Image of the paired plane.

    Returns
    -------
    tuple of np.ndarray
        ``(recon_signal, recon_paired)``.
    """
    mixing_mat = [[1 - alpha, beta], [alpha, 1 - beta]]
    unmixing_mat = np.linalg.inv(mixing_mat)
    raw_data = np.vstack([signal_mean.ravel(), paired_mean.ravel()])
    recon_data = np.dot(unmixing_mat, raw_data)
    return (
        recon_data[0, :].reshape(signal_mean.shape),
        recon_data[1, :].reshape(paired_mean.shape),
    )


def draw_masks_on_image(
    img: np.ndarray,
    masks: np.ndarray,
    ax: Optional[Axes] = None,
    color: str = "r",
    linewidth: int = 1,
) -> Optional[Tuple[Figure, Axes]]:
    """Overlay ROI contours on an image, for interactive inspection.

    Parameters
    ----------
    img : np.ndarray
        2D background image.
    masks : np.ndarray
        2D ROI mask image.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when omitted.
    color : str, optional
        Contour colour.
    linewidth : int, optional
        Contour line width.

    Returns
    -------
    tuple or None
        ``(fig, ax)`` when this call created the figure, else ``None``.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray")
    for i in range(1, np.max(masks) + 1):
        ax.contour(masks == i, colors=color, linewidths=linewidth)
    ax.axis("off")
    return (fig, ax) if created else None
