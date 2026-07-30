"""Tests for the decrosstalk metadata read layer and Processing builders."""

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from aind_data_schema.core.processing import Processing

from aind_ophys_decrosstalk_roi_images_library.utils import (
    metadata_utils as mu,
)

V1_SESSION = {
    "subject_id": "999999",
    "rig_id": "MESO.1",
    "data_streams": [
        {
            "ophys_fovs": [
                {
                    "index": 0,
                    "frame_rate": "9.48",
                    "targeted_structure": {"acronym": "VISp"},
                },
                {
                    "index": 1,
                    "frame_rate": "9.48",
                    "targeted_structure": {"acronym": "VISl"},
                },
            ]
        }
    ],
}

V2_ACQUISITION = {
    "subject_id": "999999",
    "instrument_id": "MESO.1",
    "data_streams": [
        {
            "configurations": [
                {
                    "object_type": "Imaging config",
                    "sampling_strategy": {"frame_rate": 9.48},
                }
            ]
        }
    ],
}

DATA_DESCRIPTION = {"name": "multiplane-ophys_999999_2026-07-29_00-00-00"}


def _write(directory: Path, name: str, payload: dict) -> Path:
    """Write ``payload`` as JSON into ``directory`` and return the path."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


class TestLoadDecrosstalkMetadata(unittest.TestCase):
    """v1/v2 dispatch for the fields decrosstalk consumes."""

    def test_reads_v1_session(self):
        """A v1 session.json supplies frame rate, rig and subject."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", V1_SESSION)
            _write(root, "data_description.json", DATA_DESCRIPTION)
            metadata = mu.load_decrosstalk_metadata(root)
        self.assertEqual(metadata.frame_rate, 9.48)
        self.assertEqual(metadata.instrument_id, "MESO.1")
        self.assertEqual(metadata.subject_id, "999999")
        self.assertEqual(metadata.dataset_name, DATA_DESCRIPTION["name"])

    def test_reads_v2_acquisition(self):
        """A v2 acquisition.json supplies the same fields."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", V2_ACQUISITION)
            metadata = mu.load_decrosstalk_metadata(root)
        self.assertEqual(metadata.frame_rate, 9.48)
        self.assertEqual(metadata.instrument_id, "MESO.1")
        self.assertIsNone(metadata.dataset_name)

    def test_missing_core_file_raises(self):
        """Neither core file present is a fail-fast FileNotFoundError."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mu.load_decrosstalk_metadata(Path(tmp))

    def test_both_core_files_raise(self):
        """Both core files present is ambiguous and rejected."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", V1_SESSION)
            _write(root, "acquisition.json", V2_ACQUISITION)
            with self.assertRaises(ValueError):
                mu.load_decrosstalk_metadata(root)

    def test_metadata_is_frozen(self):
        """The metadata view is immutable."""
        metadata = mu.DecrosstalkMetadata(frame_rate=9.48)
        with self.assertRaises(Exception):
            metadata.frame_rate = 1.0


class TestUpstreamSuite2pParams(unittest.TestCase):
    """Reading block_size / nonrigid from the upstream motion correction."""

    def test_reads_v2_processing_json(self):
        """v2 records the settings flat on code.parameters."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "motion_correction",
                "processing.json",
                {
                    "data_processes": [
                        {
                            "code": {
                                "parameters": {
                                    "block_size": [128, 128],
                                    "nonrigid": True,
                                    "other": 1,
                                }
                            }
                        }
                    ]
                },
            )
            params = mu.read_upstream_suite2p_params(root)
        self.assertEqual(params, {"block_size": [128, 128], "nonrigid": True})

    def test_reads_v1_data_process_json(self):
        """v1 nested them under parameters.suite2p_args."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "motion_correction",
                "plane_0_motion_correction_data_process.json",
                {
                    "parameters": {
                        "suite2p_args": {
                            "block_size": [64, 64],
                            "nonrigid": False,
                        }
                    }
                },
            )
            params = mu.read_upstream_suite2p_params(root)
        self.assertEqual(params, {"block_size": [64, 64], "nonrigid": False})

    def test_reads_v1_flat_parameters(self):
        """A v1 document without the suite2p_args nesting still resolves."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "x_data_process.json",
                {"parameters": {"block_size": [32, 32], "nonrigid": True}},
            )
            params = mu.read_upstream_suite2p_params(root)
        self.assertEqual(params, {"block_size": [32, 32], "nonrigid": True})

    def test_v2_preferred_over_v1(self):
        """With both present the v2 document wins."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "processing.json",
                {
                    "data_processes": [
                        {
                            "code": {
                                "parameters": {
                                    "block_size": [128, 128],
                                    "nonrigid": True,
                                }
                            }
                        }
                    ]
                },
            )
            _write(
                root,
                "x_data_process.json",
                {"parameters": {"block_size": [1, 1], "nonrigid": False}},
            )
            params = mu.read_upstream_suite2p_params(root)
        self.assertEqual(params["block_size"], [128, 128])

    def test_empty_processing_json_is_skipped(self):
        """Raw assets ship a two-byte '{}'; it must not raise."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "processing.json", {})
            self.assertEqual(mu.read_upstream_suite2p_params(root), {})

    def test_no_upstream_document(self):
        """Nothing upstream reads as no parameters."""
        with TemporaryDirectory() as tmp:
            self.assertEqual(mu.read_upstream_suite2p_params(Path(tmp)), {})

    def test_v2_skips_process_without_the_keys(self):
        """A data_process lacking the keys is passed over, not fatal."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "processing.json",
                {
                    "data_processes": [
                        {"code": {}},
                        {"code": {"parameters": {"nonrigid": True}}},
                    ]
                },
            )
            self.assertEqual(
                mu.read_upstream_suite2p_params(root), {"nonrigid": True}
            )


class TestResolveSuite2pParams(unittest.TestCase):
    """Override-wins resolution with fail-fast on the unresolvable."""

    def test_overrides_win_and_skip_the_read(self):
        """Both overrides given means no upstream read is attempted."""
        with TemporaryDirectory() as tmp:
            resolved = mu.resolve_suite2p_params(
                Path(tmp),
                nonrigid_override=False,
                block_size_override=[16, 16],
            )
        self.assertEqual(resolved, {"nonrigid": False, "block_size": [16, 16]})

    def test_falls_back_to_upstream(self):
        """Unset overrides are filled from the upstream document."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "processing.json",
                {
                    "data_processes": [
                        {
                            "code": {
                                "parameters": {
                                    "block_size": ["128", "128"],
                                    "nonrigid": True,
                                }
                            }
                        }
                    ]
                },
            )
            resolved = mu.resolve_suite2p_params(root)
        self.assertEqual(
            resolved, {"nonrigid": True, "block_size": [128, 128]}
        )

    def test_partial_override_mixes_sources(self):
        """One override plus one upstream value resolves both."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "processing.json",
                {
                    "data_processes": [
                        {"code": {"parameters": {"block_size": [8, 8]}}}
                    ]
                },
            )
            resolved = mu.resolve_suite2p_params(root, nonrigid_override=True)
        self.assertEqual(resolved, {"nonrigid": True, "block_size": [8, 8]})

    def test_unresolvable_raises(self):
        """Silently guessing would mis-register the paired plane, so raise."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                mu.resolve_suite2p_params(Path(tmp))
        message = str(ctx.exception)
        self.assertIn("block_size", message)
        self.assertIn("nonrigid", message)


class TestProcessingBuilders(unittest.TestCase):
    """The v2 DataProcess / Processing builders."""

    def setUp(self):
        """Fix the timestamps used by every builder assertion."""
        self.start = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
        self.end = datetime(2026, 7, 29, 12, 5, tzinfo=timezone.utc)

    def test_build_data_process(self):
        """A successful run yields a decrosstalk DataProcess, no notes."""
        data_process = mu.build_data_process(
            parameters={"block_size": [128, 128]},
            start_time=self.start,
            end_time=self.end,
            input_data=["plane_0/motion_correction/plane_0_registered.h5"],
            output_path="plane_0/decrosstalk",
            output_parameters={"alpha_mean": 0.1},
        )
        # process_type validates down to a plain str on the model.
        self.assertEqual(data_process.process_type, "Video plane decrosstalk")
        # output_path validates into an AssetPath (a str subclass).
        self.assertEqual(str(data_process.output_path), "plane_0/decrosstalk")
        self.assertIsNone(data_process.notes)
        self.assertEqual(data_process.experimenters, [])
        self.assertEqual(
            data_process.code.input_data[0].name,
            "plane_0/motion_correction/plane_0_registered.h5",
        )
        self.assertIsNotNone(data_process.resources)

    def test_build_data_process_records_error(self):
        """A failed run records the exception in notes."""
        data_process = mu.build_data_process(
            parameters={},
            start_time=self.start,
            end_time=self.end,
            error=ValueError("bad block size"),
        )
        self.assertIn("ValueError", data_process.notes)
        self.assertIn("bad block size", data_process.notes)

    @patch.dict(
        "os.environ",
        {
            "PIPELINE_NAME": "aind-pophys-pipeline",
            "PIPELINE_URL": "https://example.invalid/p",
            "PIPELINE_VERSION": "2.0.0",
        },
        clear=False,
    )
    def test_build_processing_populates_pipeline_from_env(self):
        """pipelines and pipeline_name come from the same env read."""
        data_process = mu.build_data_process(
            parameters={}, start_time=self.start, end_time=self.end
        )
        processing = mu.build_processing([data_process])
        self.assertEqual(len(processing.pipelines), 1)
        self.assertEqual(
            processing.data_processes[0].pipeline_name,
            processing.pipelines[0].name,
        )

    def test_write_processing_json_round_trips(self):
        """The written document validates back as a full v2 Processing."""
        data_process = mu.build_data_process(
            parameters={}, start_time=self.start, end_time=self.end
        )
        with TemporaryDirectory() as tmp:
            written = mu.write_processing_json(
                mu.build_processing([data_process]), Path(tmp) / "nested"
            )
            self.assertTrue(written.exists())
            Processing.model_validate_json(written.read_text())


if __name__ == "__main__":
    unittest.main()
