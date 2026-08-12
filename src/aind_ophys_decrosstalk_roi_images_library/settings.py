"""CLI/environment settings for paired-plane decrosstalk.

Kept import-light (pydantic + stdlib only) so it stays cheap to import and
free of cellpose / suite2p / aind-data-schema side effects.
"""

import json
from pathlib import Path
from typing import Any, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DecrosstalkSettings(BaseSettings, cli_parse_args=True):
    """Settings for the paired-plane decrosstalk job."""

    input_dir: Path = Field(
        default=Path("../data/"),
        description="Directory containing the paired plane input data.",
    )
    output_dir: Path = Field(
        default=Path("../results/"),
        description="Directory where output files will be saved.",
    )
    tmp_dir: Path = Field(
        default=Path("../scratch/"),
        description=(
            "Directory for the large intermediate paired-registered movies, "
            "which are deleted once decrosstalk finishes."
        ),
    )
    debug: bool = Field(
        default=False,
        description=(
            "Truncate each movie to debug_duration_s seconds before "
            "processing, for a fast end-to-end smoke test."
        ),
    )
    debug_duration_s: float = Field(
        default=360.0,
        description=(
            "Movie duration [s] retained when debug is enabled. Converted to "
            "a frame count using the acquisition frame rate."
        ),
    )
    frame_rate: Optional[float] = Field(
        default=None,
        description=(
            "Frame rate [Hz] fallback used only when it cannot be read from "
            "acquisition/session/platform metadata. Only needed for debug "
            "truncation."
        ),
    )

    # Episodic mean FOV
    num_frames: int = Field(
        default=1000,
        description=(
            "Number of frames averaged per epoch for the episodic mean FOV. "
            "Overridden by debug_num_frames when debug is enabled."
        ),
    )
    debug_num_frames: int = Field(
        default=300,
        description=(
            "Frames averaged per epoch for the episodic mean FOV when debug "
            "is enabled."
        ),
    )
    max_num_epochs: int = Field(
        default=10,
        description="Maximum number of epochs in the episodic mean FOV.",
    )

    # Upstream motion-correction parameters. Empty/None means "read them from
    # the upstream motion-correction metadata", which is the correct source:
    # the paired plane is registered by re-applying that plane's transforms,
    # so the block layout must match the one suite2p actually used.
    nonrigid: Optional[bool] = Field(
        default=None,
        description=(
            "Override for whether motion correction was non-rigid. Leave "
            "unset to read it from the upstream motion-correction metadata."
        ),
    )
    block_size: Optional[list] = Field(
        default=None,
        description=(
            "Override for the suite2p non-rigid block size, e.g. "
            "'[128, 128]'. Leave unset to read it from the upstream "
            "motion-correction metadata."
        ),
    )

    # Decrosstalk mixing-coefficient estimation
    chunk_size: int = Field(
        default=5000,
        description=(
            "Number of frames unmixed per chunk when writing the decrosstalk "
            "movie. Bounds peak memory use."
        ),
    )
    pixel_size: float = Field(
        default=0.78,
        description="Imaging plane pixel size [um].",
    )
    grid_interval: float = Field(
        default=0.01,
        description="Step of the alpha/beta grid search.",
    )
    max_grid_val: float = Field(
        default=0.3,
        description="Maximum alpha/beta value searched.",
    )
    motion_buffer: int = Field(
        default=5,
        description=(
            "Pixels cropped from each edge beyond the motion-correction crop "
            "range, to exclude non-rigid edge artifacts."
        ),
    )

    # Cellpose ROI detection on the mean images
    dendrite_diameter_um: float = Field(
        default=10.0,
        description=(
            "Diameter [um] below which a detected mask is treated as dendrite "
            "and discarded."
        ),
    )
    num_top_rois: int = Field(
        default=15,
        description=(
            "Number of highest-intensity ROIs per plane retained for the "
            "mutual-information grid search."
        ),
    )
    overlap_threshold: float = Field(
        default=0.7,
        description=(
            "Intersection-over-union above which a signal/paired mask pair is "
            "treated as the same cell duplicated by crosstalk."
        ),
    )
    cellpose_model_type: str = Field(
        default="cyto",
        description=(
            "Cellpose pretrained model used to segment the mean images. It "
            "is read from CELLPOSE_LOCAL_MODELS_PATH when present there, and "
            "downloaded otherwise."
        ),
    )

    verify: bool = Field(
        default=False,
        description=(
            "Round-trip the emitted processing.json / quality_control.json "
            "through the core v2 schema objects after writing, so bad output "
            "fails in-capsule rather than downstream."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DECROSSTALK_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("block_size", mode="before")
    @classmethod
    def _coerce_block_size(cls, value: Any) -> Optional[List[int]]:
        """Coerce a CLI/env block_size to a list of ints, or ``None``.

        Accepts a JSON string (``"[128, 128]"``) or a list whose elements may
        be strings. Kept a plain ``Optional[list]`` (not ``List[int]``, and
        with no ``default_factory``) so ``auto-app-panel`` renders it as one
        optional string parameter: ``List[int]`` becomes a single integer
        field, and an empty ``default_factory`` crashes the generator.
        """
        if isinstance(value, str):
            value = json.loads(value) if value else None
        # Normalise every "unset" spelling to None so callers test one thing.
        return [int(x) for x in value] if value else None

    def resolved_num_frames(self) -> int:
        """Return the per-epoch frame count for the active debug mode."""
        return self.debug_num_frames if self.debug else self.num_frames

    def block_size_override(self) -> Optional[List[int]]:
        """Return the block-size override, or ``None`` when unset."""
        return list(self.block_size) if self.block_size else None
