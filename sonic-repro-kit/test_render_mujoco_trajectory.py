from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from render_mujoco_trajectory import (
    G1_ISAACLAB_TO_MUJOCO_DOF,
    build_mujoco_qpos,
    prepare_runtime_xml,
)
from verify_minimal import path_is_within


class BuildQposTests(unittest.TestCase):
    def test_builds_36_value_qpos_in_mujoco_joint_order(self) -> None:
        trajectory = {
            "dof_pos": np.arange(58, dtype=np.float32).reshape(2, 29),
            "root_pos_w": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            "root_quat_w": np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
        }
        qpos = build_mujoco_qpos(trajectory)
        self.assertEqual(qpos.shape, (2, 36))
        np.testing.assert_array_equal(qpos[:, :3], trajectory["root_pos_w"])
        np.testing.assert_array_equal(
            qpos[:, 7:], trajectory["dof_pos"][:, G1_ISAACLAB_TO_MUJOCO_DOF]
        )

    def test_rejects_non_finite_data(self) -> None:
        trajectory = {
            "dof_pos": np.zeros((1, 29), dtype=np.float32),
            "root_pos_w": np.asarray([[0, 0, np.nan]], dtype=np.float32),
            "root_quat_w": np.asarray([[1, 0, 0, 0]], dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            build_mujoco_qpos(trajectory)

    def test_rejects_invalid_quaternion(self) -> None:
        trajectory = {
            "dof_pos": np.zeros((1, 29), dtype=np.float32),
            "root_pos_w": np.zeros((1, 3), dtype=np.float32),
            "root_quat_w": np.asarray([[2, 0, 0, 0]], dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "norm differs"):
            build_mujoco_qpos(trajectory)


class RuntimeXmlTests(unittest.TestCase):
    def test_creates_run_local_model_without_terrain_reference(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            (model_dir / "meshes").mkdir(parents=True)
            source = model_dir / "g1.xml"
            source_text = """<mujoco>
  <compiler meshdir="meshes" />
  <asset><mesh name="terrain_mesh" file="missing.stl" /></asset>
  <worldbody>
    <body name="terrain_body"><geom type="mesh" mesh="terrain_mesh" /></body>
  </worldbody>
</mujoco>
"""
            source.write_text(source_text, encoding="utf-8")
            destination = root / "run" / "manifests" / "g1_runtime.xml"

            details = prepare_runtime_xml(source, destination)

            self.assertEqual(source.read_text(encoding="utf-8"), source_text)
            runtime_root = ET.parse(destination).getroot()
            compiler = runtime_root.find("compiler")
            self.assertIsNotNone(compiler)
            self.assertEqual(Path(compiler.get("meshdir", "")), (model_dir / "meshes").resolve())
            self.assertIsNone(runtime_root.find(".//mesh[@name='terrain_mesh']"))
            self.assertIsNone(runtime_root.find(".//body[@name='terrain_body']"))
            self.assertEqual(details["removed_terrain_meshes"], 1)
            self.assertEqual(details["removed_terrain_bodies"], 1)

    def test_accepts_model_without_optional_terrain(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            (model_dir / "meshes").mkdir(parents=True)
            source = model_dir / "g1.xml"
            source.write_text(
                "<mujoco><compiler meshdir='meshes'/><worldbody/></mujoco>",
                encoding="utf-8",
            )
            details = prepare_runtime_xml(source, root / "run" / "g1.xml")
            self.assertEqual(details["removed_terrain_meshes"], 0)
            self.assertEqual(details["removed_terrain_geoms"], 0)
            self.assertEqual(details["removed_terrain_bodies"], 0)


class PathValidationTests(unittest.TestCase):
    def test_rejects_run_outside_runs_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            runs_root = root / "runs"
            self.assertTrue(path_is_within(runs_root / "offline_render_1", runs_root))
            self.assertFalse(path_is_within(root / "other" / "offline_render_1", runs_root))


if __name__ == "__main__":
    unittest.main()
