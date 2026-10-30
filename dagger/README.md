# SONIC teacher DAgger

`dagger/` 是独立的 DAgger 工程。SONIC nested repo 只作为冻结 teacher 和 Isaac 环境依赖；当前 CVAE 模型从 `state-action-cvae/src` 导入，不在 SONIC 或 CVAE 源码树新增 DAgger 文件。

完整配置、KL2 32-motion / intensive / v2 / stepwise / 渲染命令和结果见工作区根目录 `dagger.md`（10-history `sonic-causal-*` 入口见其末节）。

## 基本入口

```bash
cd ~/bly/dagger
export SONIC_REPO_ROOT=~/bly/sonic-repro/GR00T-WholeBodyControl
export SONIC_TEACHER_CHECKPOINT=$SONIC_REPO_ROOT/sonic_release/last.pt
export SONIC_RUNS_ROOT=~/bly/runs

# Plain MLP
bash scripts/run_dagger.sh smoke
bash scripts/run_dagger.sh pilot

# 4-motion plain MLP memorization
export DAGGER_MOTION_FILE=$SONIC_REPO_ROOT/data/motion_lib_bones_seed/robot_filtered
bash scripts/run_dagger.sh mlp-smoke
bash scripts/run_dagger.sh mlp-pilot

# 65-token CVAE student
export STATE_ACTION_CVAE_ROOT=~/bly/state-action-cvae
export CVAE_STUDENT_CHECKPOINT=/home/helloworld/bly/runs/<cvae_run>/checkpoints/best.pt
export CVAE_BASE_DATASET_RUN=/home/helloworld/bly/runs/<dataset_run>
bash scripts/run_dagger.sh cvae-smoke
bash scripts/run_dagger.sh cvae-pilot
```

- `mlp-smoke` / `mlp-pilot` 从零训练 plain MLP，输入 `actor_obs + command_multi_future_nonflat + motion_anchor_ori_b_mf_nonflat`；
  motion 由 `motion_seed` 从 `DAGGER_MOTION_FILE` 确定性选 4 个、每 motion 1 env。结果在 `~/bly/runs/dagger_mlp_smoke_*/`、
  `~/bly/runs/dagger_mlp_pilot_*/`。`memorization_pass` 只针对同一批 4 个 motion，execution 通过也不能解释为泛化或完整 SONIC 复现。
- CVAE student 使用 `65-token-hierarchical-standard-cvae-v1`，以 `anchor=32` 输出当前 Action；未来 State 在 rollout 中被 mask，
  只作仿真器产生的训练目标，SONIC Action 作为 anchor Action 的辅助监督。结果写入 `/home/helloworld/bly/runs/dagger_cvae_*/`。
- 两条路线都必须 smoke 成功后再运行 pilot。

## CVAE data contract

每个动态样本保存 `[65,70]` State、`[64,29]` Action、causal State/Action mask、teacher/student/executed raw 和 canonical Action、episode metadata 以及 latent seed。训练使用 50% 原始 CVAE 窗口和 50% DAgger 窗口，normalizer 保持原数据集版本并冻结。

当前无 reference 的 CVAE 版本是动态补全诊断。若 teacher action 在相似 student-visible 状态下仍有明显条件方差，manifest 会保留 `teacher_not_identifiable` 证据，不能把该结果解释为完整的 SONIC reference-conditioned imitation。
