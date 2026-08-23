from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from render_mujoco_trajectory import (
    G1_ISAACLAB_TO_MUJOCO_DOF,
    RUNTIME_GROUND_GEOM_NAME,
    RUNTIME_GROUND_MATERIAL_NAME,
    RUNTIME_GROUND_TEXTURE_NAME,
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

            details = prepare_runtime_xml(source, destination, width=960, height=540)

            self.assertEqual(source.read_text(encoding="utf-8"), source_text)
            runtime_root = ET.parse(destination).getroot()
            compiler = runtime_root.find("compiler")
            self.assertIsNotNone(compiler)
            self.assertEqual(Path(compiler.get("meshdir", "")), (model_dir / "meshes").resolve())
            self.assertIsNone(runtime_root.find(".//mesh[@name='terrain_mesh']"))
            self.assertIsNone(runtime_root.find(".//body[@name='terrain_body']"))
            visual_global = runtime_root.find("./visual/global")
            self.assertIsNotNone(visual_global)
            self.assertEqual(visual_global.get("offwidth"), "960")
            self.assertEqual(visual_global.get("offheight"), "540")
            texture = runtime_root.find(f"./asset/texture[@name='{RUNTIME_GROUND_TEXTURE_NAME}']")
            material = runtime_root.find(
                f"./asset/material[@name='{RUNTIME_GROUND_MATERIAL_NAME}']"
            )
            ground = runtime_root.find(f"./worldbody/geom[@name='{RUNTIME_GROUND_GEOM_NAME}']")
            self.assertIsNotNone(texture)
            self.assertIsNotNone(material)
            self.assertIsNotNone(ground)
            self.assertEqual(ground.get("type"), "plane")
            self.assertEqual(ground.get("pos"), "0 0 0")
            self.assertEqual(details["removed_terrain_meshes"], 1)
            self.assertEqual(details["removed_terrain_bodies"], 1)
            self.assertEqual(details["framebuffer"], {"width": 960, "height": 540})
            self.assertEqual(details["ground"]["style"], "checker")

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
            destination = root / "run" / "g1.xml"
            details = prepare_runtime_xml(source, destination, width=1920, height=1080)
            self.assertEqual(details["removed_terrain_meshes"], 0)
            self.assertEqual(details["removed_terrain_geoms"], 0)
            self.assertEqual(details["removed_terrain_bodies"], 0)
            runtime_root = ET.parse(destination).getroot()
            self.assertIsNotNone(runtime_root.find("asset"))
            self.assertIsNotNone(runtime_root.find("worldbody"))
            visual_global = runtime_root.find("./visual/global")
            self.assertEqual(visual_global.get("offwidth"), "1920")
            self.assertEqual(visual_global.get("offheight"), "1080")

    def test_preserves_unrelated_nodes_and_replaces_runtime_elements(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            (model_dir / "meshes").mkdir(parents=True)
            source = model_dir / "g1.xml"
            source.write_text(
                f"""<mujoco>
  <compiler meshdir="meshes" />
  <visual><global shadowsize="2048" offwidth="320" offheight="240" /></visual>
  <asset>
    <texture name="keep_texture" type="2d" builtin="flat" width="8" height="8" />
    <texture name="{RUNTIME_GROUND_TEXTURE_NAME}" type="2d" builtin="flat" width="8" height="8" />
    <material name="{RUNTIME_GROUND_MATERIAL_NAME}" rgba="1 0 0 1" />
  </asset>
  <worldbody>
    <light name="keep_light" />
    <body name="keep_body" />
    <geom name="{RUNTIME_GROUND_GEOM_NAME}" type="sphere" size="1" />
  </worldbody>
</mujoco>
""",
                encoding="utf-8",
            )
            runtime_one = model_dir / "runtime_one.xml"
            runtime_two = model_dir / "runtime_two.xml"

            details = prepare_runtime_xml(source, runtime_one, width=960, height=540)
            prepare_runtime_xml(runtime_one, runtime_two, width=1920, height=1080)

            runtime_root = ET.parse(runtime_two).getroot()
            self.assertIsNotNone(runtime_root.find(".//texture[@name='keep_texture']"))
            self.assertIsNotNone(runtime_root.find(".//light[@name='keep_light']"))
            self.assertIsNotNone(runtime_root.find(".//body[@name='keep_body']"))
            self.assertEqual(
                len(runtime_root.findall(f".//texture[@name='{RUNTIME_GROUND_TEXTURE_NAME}']")),
                1,
            )
            self.assertEqual(
                len(runtime_root.findall(f".//material[@name='{RUNTIME_GROUND_MATERIAL_NAME}']")),
                1,
            )
            self.assertEqual(
                len(runtime_root.findall(f".//geom[@name='{RUNTIME_GROUND_GEOM_NAME}']")),
                1,
            )
            visual_global = runtime_root.find("./visual/global")
            self.assertEqual(visual_global.get("shadowsize"), "2048")
            self.assertEqual(visual_global.get("offwidth"), "1920")
            self.assertEqual(visual_global.get("offheight"), "1080")
            self.assertEqual(details["replaced_runtime_elements"]["textures"], 1)
            self.assertEqual(details["replaced_runtime_elements"]["materials"], 1)
            self.assertEqual(details["replaced_runtime_elements"]["ground_geoms"], 1)

    def test_rejects_non_positive_framebuffer_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            prepare_runtime_xml(Path("unused.xml"), Path("unused-runtime.xml"), 0, 540)


class PathValidationTests(unittest.TestCase):
    def test_rejects_run_outside_runs_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            runs_root = root / "runs"
            self.assertTrue(path_is_within(runs_root / "offline_render_1", runs_root))
            self.assertFalse(path_is_within(root / "other" / "offline_render_1", runs_root))


if __name__ == "__main__":
    unittest.main()
