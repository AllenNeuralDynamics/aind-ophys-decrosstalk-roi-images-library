"""Tests for the decrosstalk metadata read layer and the Code builder."""

import json
import unittest
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest.mock import patch

import h5py

from aind_ophys_decrosstalk_roi_images_library.utils import (
    metadata_utils as mu,
)

LIBRARY = "aind-ophys-decrosstalk-roi-images-library"

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

    def test_reads_v1_session(self) -> None:
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

    def test_reads_v2_acquisition(self) -> None:
        """A v2 acquisition.json supplies the same fields."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", V2_ACQUISITION)
            metadata = mu.load_decrosstalk_metadata(root)
        self.assertEqual(metadata.frame_rate, 9.48)
        self.assertEqual(metadata.instrument_id, "MESO.1")
        self.assertIsNone(metadata.dataset_name)

    def test_cli_frame_rate_fills_a_rateless_asset(self) -> None:
        """The CLI override is used when the metadata declares no rate."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", {"data_streams": []})
            metadata = mu.load_decrosstalk_metadata(root, cli_frame_rate=7.0)
        self.assertEqual(metadata.frame_rate, 7.0)

    def test_required_frame_rate_raises_when_unresolvable(self) -> None:
        """Debug mode cannot size its truncation without a rate."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", {"data_streams": []})
            with self.assertRaises(ValueError):
                mu.load_decrosstalk_metadata(root, require_frame_rate=True)

    def test_absent_frame_rate_is_not_fatal_outside_debug(self) -> None:
        """A rate-less asset still runs when the truncation is not needed."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", {"data_streams": []})
            metadata = mu.load_decrosstalk_metadata(root)
        self.assertIsNone(metadata.frame_rate)

    def test_missing_core_file_raises(self) -> None:
        """Neither core file present is a fail-fast FileNotFoundError."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mu.load_decrosstalk_metadata(Path(tmp))

    def test_both_core_files_raise(self) -> None:
        """Both core files present is ambiguous and rejected."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", V1_SESSION)
            _write(root, "acquisition.json", V2_ACQUISITION)
            with self.assertRaises(ValueError):
                mu.load_decrosstalk_metadata(root)

    def test_metadata_is_frozen(self) -> None:
        """The metadata view is immutable."""
        metadata = mu.DecrosstalkMetadata(frame_rate=9.48)
        with self.assertRaises(Exception):
            metadata.frame_rate = 1.0


def _write_registered_movie(
    path: Path, suite2p_args: Optional[dict] = None
) -> Path:
    """Write a minimal registered movie, optionally with a metadata dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=[[[0]]])
        if suite2p_args is not None:
            f.create_dataset(
                "metadata",
                data=json.dumps({"suite2p_args": suite2p_args}).encode(),
            )
    return path


def _write_motion_transform_csv(path: Path, nonrigid: bool) -> Path:
    """Write a one-row motion-transform CSV with or without nonrigid cols."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "y,x" + (",nonrigid_y,nonrigid_x" if nonrigid else "")
    row = "0,0" + (',"[0.0]","[0.0]"' if nonrigid else "")
    path.write_text(f"{header}\n{row}\n")
    return path


class TestReadUpstreamBlockSize(unittest.TestCase):
    """Reading block_size from a registered movie's metadata dataset."""

    def test_reads_the_recorded_block_size(self) -> None:
        """block_size comes back exactly as suite2p_args recorded it."""
        with TemporaryDirectory() as tmp:
            movie = _write_registered_movie(
                Path(tmp) / "oeid_registered.h5",
                {"block_size": [128, 128], "nonrigid": False},
            )
            self.assertEqual(mu.read_upstream_block_size(movie), [128, 128])

    def test_no_metadata_dataset_is_none(self) -> None:
        """A movie with no metadata dataset yields None, not a crash."""
        with TemporaryDirectory() as tmp:
            movie = _write_registered_movie(Path(tmp) / "oeid_registered.h5")
            self.assertIsNone(mu.read_upstream_block_size(movie))

    def test_metadata_without_block_size_is_none(self) -> None:
        """suite2p_args present but lacking block_size still yields None."""
        with TemporaryDirectory() as tmp:
            movie = _write_registered_movie(
                Path(tmp) / "oeid_registered.h5", {"nonrigid": True}
            )
            self.assertIsNone(mu.read_upstream_block_size(movie))


class TestUpstreamRanNonrigid(unittest.TestCase):
    """Non-rigid is read from the CSV's own columns, never a flag.

    Motion correction's parameter-search step can leave a stale ``nonrigid``
    flag in the recorded suite2p_args (it force-sets ``nonrigid=False`` for
    the search, on the same dict later dumped to metadata), so column
    presence is the only trustworthy source.
    """

    def test_nonrigid_columns_present(self) -> None:
        """nonrigid_x/y columns present means non-rigid registration ran."""
        with TemporaryDirectory() as tmp:
            csv_path = _write_motion_transform_csv(
                Path(tmp) / "oeid_motion_transform.csv", nonrigid=True
            )
            self.assertTrue(mu.upstream_ran_nonrigid(csv_path))

    def test_nonrigid_columns_absent(self) -> None:
        """No nonrigid columns means rigid-only registration ran."""
        with TemporaryDirectory() as tmp:
            csv_path = _write_motion_transform_csv(
                Path(tmp) / "oeid_motion_transform.csv", nonrigid=False
            )
            self.assertFalse(mu.upstream_ran_nonrigid(csv_path))


class TestResolveSuite2pParams(unittest.TestCase):
    """Override-wins resolution with fail-fast on the unresolvable."""

    def test_overrides_win_and_skip_the_reads(self) -> None:
        """Both overrides given means no movie/CSV read is attempted."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolved = mu.resolve_suite2p_params(
                root / "missing_registered.h5",
                root / "missing_motion_transform.csv",
                nonrigid_override=False,
                block_size_override=[16, 16],
            )
        self.assertEqual(resolved, {"nonrigid": False, "block_size": [16, 16]})

    def test_falls_back_to_upstream(self) -> None:
        """Unset overrides are filled from the movie and the CSV."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            movie = _write_registered_movie(
                root / "oeid_registered.h5",
                {"block_size": ["128", "128"], "nonrigid": False},
            )
            csv_path = _write_motion_transform_csv(
                root / "oeid_motion_transform.csv", nonrigid=True
            )
            resolved = mu.resolve_suite2p_params(movie, csv_path)
        self.assertEqual(
            resolved, {"nonrigid": True, "block_size": [128, 128]}
        )

    def test_partial_override_mixes_sources(self) -> None:
        """One override plus one upstream value resolves both."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            movie = _write_registered_movie(
                root / "oeid_registered.h5", {"block_size": [8, 8]}
            )
            csv_path = root / "unused_motion_transform.csv"
            resolved = mu.resolve_suite2p_params(
                movie, csv_path, nonrigid_override=True
            )
        self.assertEqual(resolved, {"nonrigid": True, "block_size": [8, 8]})

    def test_unresolvable_block_size_raises(self) -> None:
        """Silently guessing would mis-register the paired plane, so raise."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            movie = _write_registered_movie(root / "oeid_registered.h5")
            csv_path = _write_motion_transform_csv(
                root / "oeid_motion_transform.csv", nonrigid=False
            )
            with self.assertRaises(ValueError) as ctx:
                mu.resolve_suite2p_params(movie, csv_path)
        self.assertIn("block_size", str(ctx.exception))


class TestBuildCode(unittest.TestCase):
    """The v2 Code block this library stamps onto every document."""

    @patch.dict("os.environ", {"VERSION": "15.0.0"}, clear=False)
    def test_wrapper_version_is_ignored(self) -> None:
        """The backing library, not the wrapper, identifies the artifact."""
        code = mu.build_code({"debug": False})
        self.assertNotEqual(code.version, "15.0.0")
        self.assertEqual(code.version, version(LIBRARY))

    def test_identity_is_the_installed_library(self) -> None:
        """name, version and url all describe this library's package."""
        code = mu.build_code({})
        self.assertEqual(code.name, LIBRARY)
        self.assertEqual(code.version, version(LIBRARY))
        self.assertEqual(
            code.url, f"https://github.com/AllenNeuralDynamics/{LIBRARY}"
        )

    def test_records_input_data(self) -> None:
        """Input provenance is carried on the code block."""
        code = mu.build_code(
            {}, input_data=["plane_0/motion_correction/plane_0_registered.h5"]
        )
        self.assertEqual(
            code.input_data[0].name,
            "plane_0/motion_correction/plane_0_registered.h5",
        )


class TestPlaneIdentity(unittest.TestCase):
    """The process name used for the dependency graph."""

    def test_process_name_matches_the_shared_composition(self) -> None:
        """The name used to exclude self from the dependency graph."""
        self.assertEqual(
            mu.process_name("VISp_0"), "VISp_0: Video plane decrosstalk"
        )

    def test_process_name_without_a_plane(self) -> None:
        """Without a plane id the schema derives the name itself."""
        self.assertIsNone(mu.process_name(None))


class TestCellposeModelProvenance(unittest.TestCase):
    """The model records that land in ``code.parameters``."""

    def test_filenames_include_the_size_model(self) -> None:
        """Cellpose builds the SizeModel unconditionally, so both resolve."""
        self.assertEqual(
            mu.cellpose_model_filenames("cyto"),
            ("cytotorch_0", "size_cytotorch_0.npy"),
        )

    def test_models_dir_comes_from_the_environment(self) -> None:
        """The directory is read, never assumed, by the library."""
        with patch.dict(
            "os.environ", {mu.CELLPOSE_MODELS_PATH_ENV: "/models/cellpose"}
        ):
            self.assertEqual(
                mu.cellpose_models_dir(), Path("/models/cellpose")
            )

    def test_unset_models_dir_warns_and_records_nothing(self) -> None:
        """No declared directory means no provenance to state."""
        with patch.dict("os.environ", {mu.CELLPOSE_MODELS_PATH_ENV: ""}):
            with self.assertLogs(mu.logger, level="WARNING"):
                self.assertEqual(mu.cellpose_model_provenance("cyto"), [])
            self.assertIsNone(mu.cellpose_models_dir())

    def test_local_weights_are_recorded_as_an_asset(self) -> None:
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

    def test_downloaded_weights_are_recorded_as_a_network_source(self) -> None:
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
