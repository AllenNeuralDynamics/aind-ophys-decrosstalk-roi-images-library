"""Tests for the decrosstalk metadata read layer and the Code builder."""

import json
import unittest
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
        """A v1 session.json supplies frame rate, rig, subject and planes."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", V1_SESSION)
            _write(root, "data_description.json", DATA_DESCRIPTION)
            metadata = mu.load_decrosstalk_metadata(root)
        self.assertEqual(metadata.frame_rate, 9.48)
        self.assertEqual(metadata.instrument_id, "MESO.1")
        self.assertEqual(metadata.subject_id, "999999")
        self.assertEqual(metadata.dataset_name, DATA_DESCRIPTION["name"])
        self.assertEqual(metadata.fov_ids, ("VISp_0", "VISl_1"))

    def test_reads_v2_acquisition(self):
        """A v2 acquisition.json supplies the same fields."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", V2_ACQUISITION)
            metadata = mu.load_decrosstalk_metadata(root)
        self.assertEqual(metadata.frame_rate, 9.48)
        self.assertEqual(metadata.instrument_id, "MESO.1")
        self.assertIsNone(metadata.dataset_name)

    def test_cli_frame_rate_fills_a_rateless_asset(self):
        """The CLI override is used when the metadata declares no rate."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", {"data_streams": []})
            metadata = mu.load_decrosstalk_metadata(root, cli_frame_rate=7.0)
        self.assertEqual(metadata.frame_rate, 7.0)

    def test_required_frame_rate_raises_when_unresolvable(self):
        """Debug mode cannot size its truncation without a rate."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", {"data_streams": []})
            with self.assertRaises(ValueError):
                mu.load_decrosstalk_metadata(root, require_frame_rate=True)

    def test_absent_frame_rate_is_not_fatal_outside_debug(self):
        """A rate-less asset still runs when the truncation is not needed."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", {"data_streams": []})
            metadata = mu.load_decrosstalk_metadata(root)
        self.assertIsNone(metadata.frame_rate)

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


class TestBuildCode(unittest.TestCase):
    """The v2 Code block this capsule stamps onto every document."""

    @patch.dict("os.environ", {"VERSION": "15.0.0"}, clear=False)
    def test_capsule_version_is_ignored(self):
        """The backing library, not the wrapper, identifies the artifact."""
        code = mu.build_code({"debug": False})
        self.assertNotEqual(code.version, "15.0.0")
        self.assertEqual(code.version, version(mu.LIBRARY_NAME))

    def test_version_and_url_identify_the_backing_library(self):
        """code.version and code.url must name one artifact."""
        code = mu.build_code({})
        self.assertEqual(code.version, version(mu.LIBRARY_NAME))
        self.assertIn(mu.LIBRARY_NAME, code.url)
        # The landing page, not a branch or a subdirectory.
        self.assertNotIn("/tree/", code.url)

    def test_records_input_data(self):
        """Input provenance is carried on the code block."""
        code = mu.build_code(
            {}, input_data=["plane_0/motion_correction/plane_0_registered.h5"]
        )
        self.assertEqual(
            code.input_data[0].name,
            "plane_0/motion_correction/plane_0_registered.h5",
        )


class TestPlaneIdentity(unittest.TestCase):
    """The scavenged-vs-canonical cross-check and the process name."""

    def test_canonical_pair_passes_quietly(self):
        """Two canonical ids out of a multiplane session are no divergence."""
        self.assertTrue(
            mu.validate_pair_ids(
                ["VISp_0", "VISl_1"],
                ["VISp_0", "VISl_1", "VISa_2", "VISal_3"],
            )
        )

    def test_unknown_id_warns_and_does_not_raise(self):
        """Canonicality is cosmetic, so a divergence only warns."""
        with self.assertLogs(mu.logger, level="WARNING") as logs:
            result = mu.validate_pair_ids(
                ["VISp_0", "plane_9"], ["VISp_0", "VISl_1"]
            )
        self.assertFalse(result)
        self.assertIn("plane_9", "\n".join(logs.output))

    def test_no_canonical_ids_warns_and_keeps_the_scavenged_pair(self):
        """Metadata without planes leaves nothing to check against."""
        with self.assertLogs(mu.logger, level="WARNING"):
            self.assertTrue(mu.validate_pair_ids(["a", "b"], []))

    def test_process_name_matches_the_shared_composition(self):
        """The name used to exclude self from the dependency graph."""
        self.assertEqual(
            mu.process_name("VISp_0"), "VISp_0: Video plane decrosstalk"
        )

    def test_process_name_without_a_plane(self):
        """Without a plane id the schema derives the name itself."""
        self.assertIsNone(mu.process_name(None))


class TestCellposeModelProvenance(unittest.TestCase):
    """The model records that land in ``code.parameters``."""

    def test_filenames_include_the_size_model(self):
        """Cellpose builds the SizeModel unconditionally, so both resolve."""
        self.assertEqual(
            mu.cellpose_model_filenames("cyto"),
            ("cytotorch_0", "size_cytotorch_0.npy"),
        )

    def test_models_dir_comes_from_the_environment(self):
        """The directory is read, never assumed, by the library."""
        with patch.dict(
            "os.environ", {mu.CELLPOSE_MODELS_PATH_ENV: "/models/cellpose"}
        ):
            self.assertEqual(
                mu.cellpose_models_dir(), Path("/models/cellpose")
            )

    def test_unset_models_dir_warns_and_records_nothing(self):
        """No declared directory means no provenance to state."""
        with patch.dict("os.environ", {mu.CELLPOSE_MODELS_PATH_ENV: ""}):
            with self.assertLogs(mu.logger, level="WARNING"):
                self.assertEqual(mu.cellpose_model_provenance("cyto"), [])
            self.assertIsNone(mu.cellpose_models_dir())

    def test_local_weights_are_recorded_as_an_asset(self):
        """Present files are hashed and labelled by the directory name."""
        with TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "cellpose-cyto"
            models_dir.mkdir()
            (models_dir / "cytotorch_0").write_bytes(b"net")
            (models_dir / "size_cytotorch_0.npy").write_bytes(b"size")
            with patch.dict(
                "os.environ",
                {mu.CELLPOSE_MODELS_PATH_ENV: str(models_dir)},
            ):
                records = mu.cellpose_model_provenance("cyto")
        self.assertEqual(
            [record["filename"] for record in records],
            ["cytotorch_0", "size_cytotorch_0.npy"],
        )
        self.assertTrue(all(record["sha256"] for record in records))
        self.assertEqual(
            {record["source_name"] for record in records}, {"cellpose-cyto"}
        )
        self.assertEqual(
            {record["source"] for record in records},
            {mu.MODEL_SOURCE_ASSET},
        )

    def test_downloaded_weights_are_recorded_as_a_network_source(self):
        """A post-run read describes bytes that arrived over the network."""
        with TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "cellpose-cyto"
            models_dir.mkdir()
            (models_dir / "cytotorch_0").write_bytes(b"net")
            with patch.dict(
                "os.environ",
                {mu.CELLPOSE_MODELS_PATH_ENV: str(models_dir)},
            ):
                with self.assertLogs(level="WARNING"):
                    records = mu.cellpose_model_provenance(
                        "cyto", downloaded=True
                    )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"], mu.MODEL_SOURCE_NETWORK)


if __name__ == "__main__":
    unittest.main()
