"""Replay-only recorder: stops before any mid-trajectory automatic reset."""
from pathlib import Path
from action_replay_recorder import ActionReplayTrajectoryRecorderTerm, ActionReplayTrajectoryRecorderCfg
from exact_replay_initializer import _atomic_json
from isaaclab.utils import configclass


class Replay65Recorder(ActionReplayTrajectoryRecorderTerm):
    def _save_audit(self):
        audit = getattr(self._env,"_replay65_audit",None)
        if audit is None:
            raise RuntimeError("replay65 exact initialization hook did not run")
        _atomic_json(Path(self.cfg.save_path)/f"{self.cfg.environment_id_offset:06d}.runtime.json",audit)

    def record_pre_step(self):
        self._save_audit()
        return super().record_pre_step()

    def record_post_step(self):
        result = super().record_post_step()
        self._env._replay65_audit["control_steps"] += 1
        self._save_audit()
        return result

    def record_pre_reset(self,env_ids):
        audit = getattr(self._env,"_replay65_audit",None)
        if audit is not None and (env_ids is None or len(env_ids)):
            audit["midrun_reset"] = True
            manager = self._env.termination_manager
            audit["termination_terms"] = {name:manager.get_term(name).detach().cpu().tolist() for name in manager.active_terms}
            self._save_audit()
            self.close("")
            raise RuntimeError("replay65 stopped before automatic reset; partial trajectory and termination reason preserved")
        return None,None


@configclass
class Replay65RecorderCfg(ActionReplayTrajectoryRecorderCfg):
    class_type = Replay65Recorder
