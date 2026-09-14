from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from cvae_sa.posterior_h50_action_replay_exact_init import (
    _first_threshold_crossings,
    _payload_sha256,
    _raw_from_processed,
    _recorded_hdf_identity,
    _write_exact_initialization,
    rotate_body_to_world,
)


class H50ExactInitializationReplayTests(unittest.TestCase):
    def test_body_to_world_rotation_uses_wxyz_quaternion(self) -> None:
        identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vector = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_allclose(rotate_body_to_world(identity, vector), vector)

        yaw_90 = np.asarray(
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32
        )
        np.testing.assert_allclose(
            rotate_body_to_world(yaw_90, np.asarray([1.0, 0.0, 0.0])),
            np.asarray([0.0, 1.0, 0.0]),
            atol=1.0e-6,
        )

    def test_processed_action_is_inverted_exactly(self) -> None:
        processed = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
        scale = np.linspace(0.5, 1.5, 29, dtype=np.float32)
        offset = np.linspace(-0.1, 0.1, 29, dtype=np.float32)
        raw = _raw_from_processed(processed, scale, offset, None, None)
        np.testing.assert_allclose(raw * scale + offset, processed, atol=1.0e-6)

    def test_initialization_payload_hash_is_deterministic(self) -> None:
        values = {
            "z": np.arange(5, dtype=np.float32),
            "a": np.asarray("identity"),
        }
        self.assertEqual(_payload_sha256(values), _payload_sha256(dict(reversed(list(values.items())))))
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.npz"
            second = Path(temporary) / "second.npz"
            first_hash, first_manifest = _write_exact_initialization(first, values)
            second_hash, second_manifest = _write_exact_initialization(second, values)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first_manifest["payload_sha256"], second_manifest["payload_sha256"])

    def test_source_hdf_identity_accepts_physics_v4_without_optional_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            subset = root / "subset"
            source_run = root / "source"
            source_hdf = source_run / "sonic_physics_sa_v3.hdf5"
            for path in (parent / "manifests", subset / "manifests", source_run):
                path.mkdir(parents=True, exist_ok=True)
            source_hdf.write_bytes(b"read-only-fixture")
            parent_manifest_path = parent / "manifests/dataset_manifest.json"
            parent_manifest_path.write_text(
                json.dumps({"format_version": "physics_v4_without_optional_anchor"}),
                encoding="utf-8",
            )
            recorded_hdf_sha = hashlib.sha256(source_hdf.read_bytes()).hexdigest()
            source_hashes = {
                "sources": [
                    {
                        "run_dir": str(source_run.resolve()),
                        "files": {
                            "dataset": {
                                "path": str(source_hdf.resolve()),
                                "size_bytes": source_hdf.stat().st_size,
                                "sha256": recorded_hdf_sha,
                            }
                        },
                    }
                ]
            }
            (parent / "manifests/source_hashes.json").write_text(
                json.dumps(source_hashes), encoding="utf-8"
            )
            subset_manifest = {
                "parent_dataset_run": str(parent.resolve()),
                "parent_dataset_manifest_sha256": hashlib.sha256(
                    parent_manifest_path.read_bytes()
                ).hexdigest(),
            }
            (subset / "manifests/dataset_manifest.json").write_text(
                json.dumps(subset_manifest), encoding="utf-8"
            )

            identity = _recorded_hdf_identity(
                subset,
                {
                    "hdf5_path": str(source_hdf.resolve()),
                    "source_run": str(source_run.resolve()),
                },
            )

            self.assertEqual(identity["sha256"], recorded_hdf_sha)
            self.assertIsNone(identity["source_hashes_manifest_parent_anchor"])

    def test_first_threshold_crossing_reports_frame_or_none(self) -> None:
        errors = {
            "joint_position_rmse_rad": np.asarray([0.0, 0.03]),
            "root_position_error_m": np.asarray([0.0, 0.01]),
            "root_orientation_error_deg": np.asarray([0.0, 6.0]),
            "body_mpjpe_m": np.asarray([0.06, 0.01]),
            "contact_agreement": np.asarray([1.0, 0.9]),
        }
        self.assertEqual(
            _first_threshold_crossings(errors),
            {
                "joint_position_rmse_rad": 1,
                "root_position_error_m": None,
                "root_orientation_error_deg": 1,
                "body_mpjpe_m": 0,
                "contact_agreement": 1,
            },
        )

    def test_shell_entry_keeps_exact_replay_separate_from_history(self) -> None:
        project = Path(__file__).resolve().parents[1]
        shell = (project / "cvae_repro.sh").read_text(encoding="utf-8")
        start = shell.index("posterior_h50_action_replay_exact_init()")
        stop = shell.index("validate_action_mask_replay()", start)
        function = shell[start:stop]
        self.assertIn("posterior_h50_action_replay_exact_init prepare", function)
        self.assertIn("render_h50a_exact_action_replays.py", function)
        self.assertIn("posterior_h50_action_replay_exact_init finalize", function)
        self.assertNotIn("CVAE_POSTERIOR_H50_REPLAY_POST_STEPS", function)
        self.assertIn("SONIC exact replay hook is missing", function)
        self.assertIn("apply_exact_replay_initialization", function)
        self.assertIn("posterior-h50-action-replay-exact-init", shell)

    def test_renderer_requires_three_independent_named_videos(self) -> None:
        renderer = (
            Path(__file__).resolve().parents[2]
            / "sonic-repro-kit/render_h50a_exact_action_replays.py"
        ).read_text(encoding="utf-8")
        for name in (
            "videos/recorded_hdf.mp4",
            "videos/original_action_exact_init.mp4",
            "videos/h50a_action_exact_init.mp4",
        ):
            self.assertIn(name, renderer)
        self.assertIn("for entry in entries", renderer)
        self.assertIn("if int(request.get(\"steps\", -1)) != 64", renderer)


if __name__ == "__main__":
    unittest.main()
