"""Trajectory recorder adapter that keeps only SONIC's 29 G1 body joints."""

from __future__ import annotations

import numpy as np

from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER, get_body_joint_indices
from gear_sonic.envs.manager_env.mdp.recorders import (
    TrajectoryRecorderCfg,
    TrajectoryRecorderTerm,
)
from isaaclab.utils import configclass


class BodyTrajectoryRecorderTerm(TrajectoryRecorderTerm):
    """Reuse SONIC recording behavior while removing dex-hand DOFs from saved frames."""

    cfg: BodyTrajectoryRecorderCfg

    def _initialize(self) -> None:
        super()._initialize()
        robot = self.env.scene["robot"]
        body_indices = get_body_joint_indices(robot)
        if len(body_indices) != 29:
            raise ValueError(f"Expected 29 G1 body joints, found {len(body_indices)}")
        self._body_joint_indices_np = body_indices.detach().cpu().numpy().astype(np.int64)
        resolved_names = [robot.joint_names[index] for index in self._body_joint_indices_np]
        if resolved_names != G1_ISAACLab_ORDER:
            raise ValueError(
                "Resolved G1 body joints do not match the canonical Isaac Lab ordering: "
                f"{resolved_names}"
            )

    def record_post_step(self):
        previous_lengths = {
            env_index: len(data["dof_pos"]) for env_index, data in self._frame_data.items()
        }
        result = super().record_post_step()
        for env_index, data in self._frame_data.items():
            if len(data["dof_pos"]) > previous_lengths.get(env_index, 0):
                raw_joint_pos = data["dof_pos"][-1]
                data["dof_pos"][-1] = raw_joint_pos[self._body_joint_indices_np]
        return result


@configclass
class BodyTrajectoryRecorderCfg(TrajectoryRecorderCfg):
    """Configuration for the 29-DOF offline-render trajectory adapter."""

    class_type = BodyTrajectoryRecorderTerm
