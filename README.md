# aind-ophys-decrosstalk-roi-images-library

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)

Paired-plane crosstalk removal for AIND population ophys, with
`aind-data-schema` v2 metadata I/O.

This is the backing library for the `aind-ophys-decrosstalk-roi-images` Code
Ocean capsule. The capsule is a thin wrapper that calls `job.run()`; all logic,
tests and dependency pins live here.

Two simultaneously-imaged mesoscope planes bleed into each other. The mixing is
modelled as `[[1-alpha, beta], [alpha, 1-beta]]`. Each plane's raw movie is
re-registered with the *other* plane's suite2p motion transforms so both sit in
the same frame of reference; `alpha` and `beta` are then fitted per epoch by
minimising the normalised mutual information between the two planes'
reconstructed ROI images, averaged, and applied to the full movie.

Unlike the single-plane capsules, one invocation processes **two** planes - the
pair named by the `pair*.txt` file in the input directory, one plane directory
name per line -
and writes a full set of outputs, including its own `processing.json` and
`quality_control.json`, into each plane's own output directory. Without a pair
file, a staged `../data/pair*/` directory holding the two plane directories is
used instead.

## Usage

```python
from aind_ophys_decrosstalk_roi_images_library.job import run

run()  # settings come from CLI args / environment
```

Settings can also be constructed explicitly:

```python
from aind_ophys_decrosstalk_roi_images_library.job import run
from aind_ophys_decrosstalk_roi_images_library.settings import (
    DecrosstalkSettings,
)

run(DecrosstalkSettings(debug=True))
```

## Modules

### `job.py`

Orchestration. `run()` is the entry point: metadata read, logging setup, pair
resolution (`_locate_pair`), the paired-plane re-registration of each raw movie,
the coefficient fit, the unmixed-movie write (`_write_decrosstalk_movie`), the
per-pair coupled-FOV image (`_coupled_fov_metric`), and the per-plane
`processing.json` (via the shared `stage_guard`) and `quality_control.json`
emission. The large paired-registered intermediates are written to `tmp_dir` and
deleted however the run ends.

### `decrosstalk.py`

The algorithm. Cellpose segmentation of the episodic mean FOV images and mask
filtering (`filter_dendrite`, `filter_border_roi`, `get_top_intensity_mask`,
`get_ranks_roi_inds`, `reorder_mask`, `get_signal_paired_top_masks`), the
motion-correction crop ranges
(`get_motion_correction_crop_xy_range`,
`get_motion_correction_crop_xy_range_from_both_planes`), the bounding boxes the
mutual-information comparison samples (`get_bounding_box`), the per-epoch grid
search (`decrosstalk_roi_image_from_episodic_mean_fov`,
`decrosstalk_roi_image_single_pair_from_episodic_mean_fov`), the unmixing itself
(`apply_mixing_matrix`), and an interactive overlay (`draw_masks_on_image`).

### `paired_plane_registration.py`

Re-registration of one plane's raw movie with its partner's suite2p offsets:
the motion-transform CSV reader (`get_s2p_motion_transform`), the rigid roll
(`shift_frame`), the chunked re-registration (`paired_plane_cached_movie`), and
the chunked projections and episodic-mean-FOV reduction
(`projection_process_chunked`, `episodic_mean_fov`) used as the fit input and
to write the unmixed movie's projections and episodic-mean-FOV video.

### `settings.py`

`DecrosstalkSettings` - the pydantic-settings model behind the CLI, the
environment (`DECROSSTALK_` prefix) and the Code Ocean app panel. Import-light
on purpose (pydantic plus stdlib only), so the app-panel generator can read it
without cellpose, suite2p or `aind-data-schema`.

### `qc.py`

Coupled-FOV QC metric content over the shared `aind_pophys_metadata` builders.
`build_coupled_fov_metric` makes one pending dropdown metric per pair
(`Crosstalk` group), referencing the two planes' decrosstalked max projections
side by side (`write_coupled_fov_image`) to check for residual cross-talk or
time series assigned to the wrong plane. It is written only into the
lower-indexed plane's document (`order_pair_by_plane_index`, by position in the
session's plane-index-ordered plane ids), so the run-level aggregate holds it
once.

One `quality_control.json` is written per plane; the higher-indexed plane's
holds no metrics.

### `utils/metadata_utils.py`

The `DecrosstalkMetadata` view over v1/v2 metadata
(`load_decrosstalk_metadata`), the upstream suite2p parameter reader
(`read_upstream_suite2p_params`, `resolve_suite2p_params`), the emitted
process name (`process_name`), the Cellpose model provenance
(`cellpose_model_filenames`, `cellpose_models_dir`,
`cellpose_model_provenance`) and the `Code` block (`build_code`).

## Upstream parameters

Paired-plane registration re-applies the other plane's suite2p transforms, so it
must use the `block_size` and `nonrigid` settings suite2p itself used. Both are
read from the upstream motion-correction document - a v2 `processing.json`
(flat on `data_processes[*].code.parameters`), falling back to a v1
`*data_process.json` (under `parameters.suite2p_args`). The `--nonrigid` and
`--block_size` settings override the read. A value available from neither source
is a fail-fast error rather than a guess, since a wrongly warped paired movie
yields wrong mixing coefficients.

## Cellpose model resolution

ROI detection uses the Cellpose pretrained model named by
`--cellpose_model_type` (default `cyto`). Cellpose resolves **two** files - the
segmentation net `<model>torch_0` and the diameter-estimating size model
`size_<model>torch_0.npy`, which `Cellpose.__init__` builds unconditionally.

Resolution is local-first: Cellpose reads
`CELLPOSE_LOCAL_MODELS_PATH` (set by the capsule's `code/run` to a mounted model
data asset) and downloads over the network when the files are not there. Which
bytes were actually used is recorded in the emitted
`code.parameters.cellpose_model_files`, one record per file with its `sha256`
and a `source` of either the mounted asset or the network. When
`CELLPOSE_LOCAL_MODELS_PATH` is unset the library warns and records no
provenance.

## Outputs

Written into `<results>/<plane_id>/decrosstalk/` for **each** plane of the pair:

| File | Contents |
|---|---|
| `<plane_id>_decrosstalk.h5` | unmixed int16 movie (`data`, with `alpha`/`beta` attributes) plus the per-epoch `alpha_list`, `beta_list` and `mean_norm_mi_list` datasets |
| `<plane_id>_decrosstalk_episodic_mean_fov.h5` | per-epoch mean images of the unmixed movie |
| `<plane_id>_decrosstalk_episodic_mean_fov.webm` | per-epoch mean images of the unmixed movie, as a video |
| `<plane_id>_registered_episodic_mean_fov.h5` | per-epoch mean images of this plane's own motion-corrected movie |
| `<plane_id>_registered_to_pair_episodic_mean_fov.h5` | per-epoch mean images of this plane's raw movie warped into its partner's frame |
| `*_avg_img.png`, `*_max_img.png` | average and maximum projections, one pair per reduced movie |
| `processing.json` | full `Processing` document for this plane |
| `quality_control.json` | full `QualityControl` document for this plane |

The lower-indexed plane of the pair also holds
`<lower_plane_id>_<higher_plane_id>_coupled_fov_max.png`, the coupled-FOV QC
reference, and its `quality_control.json` carries the pair's coupled-FOV metric.
The higher-indexed plane's `quality_control.json` holds no metrics.

The fitted coefficients also ride on the `processing.json` as `alpha_mean`,
`beta_mean`, `num_epochs` and the resolved upstream `block_size` / `nonrigid`.

## Installation

To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using [uv](https://docs.astral.sh/uv/), run
```bash
uv sync
```

This library is **Python 3.10 only**. `suite2p==0.14.3` gives a frozen-dataclass
field an ndarray default, which Python 3.11+ rejects, and the release that fixes
it requires a cellpose major bump that removes the
`models.Cellpose(model_type=...)` API this algorithm uses. See the comments in
`pyproject.toml`.

## Tests and linters

Tests are `unittest`. The CI gate is:
```bash
uv run flake8 . && uv run interrogate --verbose .
uv run coverage run -m unittest discover && uv run coverage report
```
Coverage is gated at 100%; `decrosstalk.py`, `paired_plane_registration.py` and
`job.py` are omitted, since the compute and orchestration layers need real
movies, cellpose weights and suite2p, and are exercised by the Code Ocean data
run rather than by local unit tests.

## Level of Support
We are not currently supporting this code, but simply releasing it to the
community AS IS but are not able to provide any guarantees of support. The
community is welcome to submit issues, but you should not expect an active
response.
