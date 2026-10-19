from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from prepare_g1_plus_unlocked31 import prepare


class Unlocked31PreparationTests(unittest.TestCase):
    def test_copies_source_without_locking_neck(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            asset = root / "asset"
            asset.mkdir()
            joints = []
            for index in range(29):
                joints.append(
                    f'<joint name="body_{index}" type="revolute">'
                    f'<parent link="base"/><child link="link_{index}"/>'
                    f"</joint>"
                )
            joints.extend([
                '<joint name="neck_pitch_joint" type="revolute"><parent link="base"/><child link="neck_pitch"/></joint>',
                '<joint name="neck_yaw_joint" type="revolute"><parent link="base"/><child link="neck_yaw"/></joint>',
            ])
            source_text = "<robot name=\"g1_plus\">" + "".join(joints) + "</robot>\n"
            (asset / "g1_pl.urdf").write_text(source_text, encoding="utf-8")
            (asset / "motor_info.md").write_text("test\n", encoding="utf-8")
            output = root / "run"

            result = prepare(asset, output)

            unlocked = output / "assets/g1_pl_unlocked31.urdf"
            self.assertTrue(unlocked.is_file())
            self.assertEqual(unlocked.read_text(encoding="utf-8"), source_text)
            self.assertEqual(result["movable_joint_count"], 31)
            self.assertEqual(result["neck_joint_types"]["neck_pitch_joint"], "revolute")
            self.assertEqual(result["neck_joint_types"]["neck_yaw_joint"], "revolute")
            manifest = json.loads(
                (output / "manifests/g1_plus_asset_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["unlocked31"]["movable_joint_count"], 31)
            parsed = ET.parse(unlocked).getroot()
            self.assertEqual(
                parsed.find("./joint[@name='neck_pitch_joint']").get("type"),
                "revolute",
            )


if __name__ == "__main__":
    unittest.main()
