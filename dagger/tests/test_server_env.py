import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import unittest


class ServerEnvironmentTests(unittest.TestCase):
    def test_separate_checkout_keeps_assets_and_code_roots(self):
        bash = shutil.which("bash")
        git_bash = Path("D:/Git/bin/bash.exe")
        if os.name == "nt" and git_bash.is_file():
            bash = str(git_bash)
        if not bash:
            self.skipTest("needs bash")
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        for name in ("SONIC_RUNS_ROOT", "SONIC_REPO_ROOT", "STATE_ACTION_CVAE_ROOT", "SONIC_TEACHER_CHECKPOINT",
                     "CVAE_DATASET_PATH_MAP", "CVAE_STUDENT_CHECKPOINT", "CVAE_KL2_DATASET_RUN",
                     "CVAE_KL2_MOTION_FILE", "DAGGER_MOTION_FILE", "DAGGER_MOTION_MANIFEST", "CVAE_MODEL_CONFIG"):
            environment.pop(name, None)
        environment.update(PYTHON=Path(sys.executable).as_posix(), BLY_PROJECT_ROOT="/tmp/server-assets",
                           BLY_MIGRATED_RUNS=root.as_posix())
        script = f"source {shlex.quote((root / 'scripts/server_env.sh').as_posix())} || exit\n"
        script += 'test "$SONIC_RUNS_ROOT" = /tmp/server-assets/runs || exit 10\n'
        script += 'case "$STATE_ACTION_CVAE_ROOT" in */state-action-cvae) ;; *) exit 11 ;; esac\n'
        script += 'test "$DAGGER_MOTION_MANIFEST" = "$(pwd)/configs/cvae_kl2_motion_manifest_v5.json"\n'
        result = subprocess.run([bash, "-s"], input=script, text=True, capture_output=True,
                                cwd=root, env=environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
