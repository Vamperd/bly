from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import unittest

import prepare_g1_plus_locked29 as asset


ASSET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "sonic-repro"
    / "GR00T-WholeBodyControl"
    / "gear_sonic"
    / "data"
    / "assets"
    / "robot_description"
    / "urdf"
    / "g1_plus"
)


@unittest.skipUnless(ASSET_ROOT.is_dir(), "external G1+ asset bundle is not present")
class G1PlusAssetTests(unittest.TestCase):
    def test_transferred_bundle_has_expected_contract(self) -> None:
        manifest = asset.audit_asset(ASSET_ROOT)
        self.assertGreaterEqual(manifest["file_count"], 39)
        self.assertEqual(manifest["urdf"]["movable_joint_count"], 31)
        self.assertEqual(set(manifest["urdf"]["neck_joints"]), set(asset.NECK_JOINTS))
        self.assertEqual(manifest["urdf"]["missing_mesh_references"], [])
        self.assertEqual(len(manifest["urdf"]["mesh_references"]), 37)

    def test_locked_tree_removes_only_neck_from_movable_set(self) -> None:
        source = ASSET_ROOT / asset.URDF_NAME
        tree = ET.parse(source)
        original = asset.movable_joint_names(tree)
        locked = asset.lock_neck_tree(tree)
        self.assertEqual(locked["locked_joints"], list(asset.NECK_JOINTS))
        self.assertEqual(locked["movable_joint_count"], 29)
        self.assertEqual(
            set(original) - set(locked["movable_joint_names"]),
            set(asset.NECK_JOINTS),
        )
        for name in asset.NECK_JOINTS:
            joint = next(joint for joint in tree.getroot().findall("joint") if joint.get("name") == name)
            self.assertEqual(joint.get("type"), "fixed")
            self.assertIsNone(joint.find("axis"))
            self.assertIsNone(joint.find("limit"))

    def test_source_urdf_is_untouched_by_in_memory_lock(self) -> None:
        source = ASSET_ROOT / asset.URDF_NAME
        before = source.read_bytes()
        asset.lock_neck_tree(ET.parse(source))
        self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
