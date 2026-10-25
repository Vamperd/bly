# SONIC teacher + CVAE DAgger 实施进度

最后更新：2026-10-26

## 当前架构

DAgger 的唯一实现目录是工作区根目录 `dagger/`。SONIC nested repo 只提供 release teacher、原始环境和 G1 reference；`state-action-cvae` 只提供现有 65-token CVAE 模型、数据集和 normalization contract。

当前新增的 CVAE 路径包括：

- `dagger/src/dagger/runtime/cvae_window.py`：维护 65-state/64-action rolling window，anchor 固定为 32，构造 causal mask 和 future-State loss mask；
- `dagger/src/dagger/runtime/sonic_env.py`：只读 SONIC recorder 的 Physics State，并实现 raw Action 与 29 维 canonical Action 的双向映射；
- `dagger/src/dagger/core/cvae_dataset.py`：保存和均衡采样各 DAgger round 的动态窗口；
- `dagger/src/dagger/core/cvae_loop.py`：Isaac 环境闭环收集 teacher/student/executed Action 和未来 State；
- `dagger/src/dagger/adapters/cvae_student.py`：加载 Stage-C 65-token CVAE，rollout 使用 condition + fixed standard-normal latent，训练使用 future-State loss、teacher-Action loss 和 KL；
- `dagger/configs/cvae_smoke.yaml`、`dagger/configs/cvae_pilot.yaml`：独立配置，不进入 SONIC Hydra defaults。

Plain MLP 的通用路径仍可通过 `smoke` 和 `pilot` 使用；4-motion 记忆实验通过
`mlp-smoke` 和 `mlp-pilot` 使用；CVAE 路径通过 `cvae-smoke` 和 `cvae-pilot` 使用。
Plain MLP 的 teacher/student mixture 已改为 **per-step**：每一步独立采样
`teacher_mask ~ Bernoulli(beta)`，teacher Action 始终计算并作为标签，环境执行 teacher 或
student 之一；`beta=0` 是纯 student rollout，`beta=1` 是纯 teacher。CVAE 路径仍使用
per-episode mixture。

## 术语和数据语义

- **DAgger**：学生自己进入环境，教师在学生实际访问的状态上重新给出动作标签，再聚合数据训练学生。
- **CVAE**：用隐变量补全被 Mask 的 State/Action 序列。训练时可用完整序列计算 posterior；部署时只使用 condition 和 standard-normal latent。
- **Mask**：告诉 condition encoder 哪些值被隐藏。rollout 中历史 State/Action 可见，未来 State/Action 被隐藏。
- **canonical Action**：仿真器处理后的关节目标减去 nominal joint position；SONIC raw policy Action 通过 action scale、offset、clip 转换到该坐标。
- **future State target**：student rollout 时不可见，但环境执行后可以获得，用作训练标签。

CVAE 的当前执行位置为：

```text
State[0:32]     history
State[32]       current state
State[33:64]    masked future state
Action[0:31]    executed action history
Action[32:63]   masked future action
output.action[32] = executable student action
```

未来 State 只进入训练 target，不进入 student rollout 输入。由于当前 CVAE 不接收 SONIC reference，这一版是动态误差补全诊断；manifest 中必须单独记录 `information_contract_pass`，不能直接解释为完整 reference-conditioned policy imitation。

## CVAE 固定训练合同

```text
round 0: beta=1.0
round 1: beta=0.5
round 2: beta=0.2
round 3: beta=0.0
```

`beta` 在 CVAE 路径按 episode 采样并在 episode 内保持不变，执行动作只能来自 teacher 或 student
其中之一。plain MLP 路径改为 per-step sampling，见上一节。训练混合比例为 50% 原始 CVAE
窗口和 50% DAgger 动态窗口，normalizer 从原 CVAE 数据集加载并冻结。

动态 loss 为：

```text
L_future_state
+ 0.1 * L_future_contact
+ 1.0 * L_teacher_action(anchor=32)
+ 0.001 * KL(q(z|x)||N(0,I))
```

优化器为 AdamW，学习率 `1e-5`，micro-batch `32`，gradient clip `1.0`，每轮 `2000` steps。smoke 将训练 steps 和环境步数缩小到 2 / 8。

## 已完成的静态验证

Windows 已完成：

- Python compile；
- CVAE window shape 和 causal Mask 测试；
- future State target 保留测试；
- round-balanced dataset 测试；
- canonical/raw action round-trip 测试；
- 原 plain MLP core tests；
- plain MLP 原始观测/单次归一化测试；
- 冻结归一化 round-trip 测试；
- checkpoint round-trip、action bounds clip、motion 选择测试；
- DAgger student-only teacher RMSE/per-motion 测试；
- per-step teacher/student mixture 比例测试；
- DAgger data shard round-trip 测试；
- same-motion memorization 门禁测试；
- SONIC reference term selection 和 motion key contract 测试。

当前共 21 个 Windows 单元测试通过。

当前测试命令：

```powershell
$env:PYTHONPATH="$PWD/dagger/src"
python -m compileall -q dagger/src dagger/tests
python -m unittest discover -s dagger/tests -v
```

## Ubuntu CVAE 执行入口

```bash
cd ~/bly/dagger
export SONIC_REPO_ROOT=~/bly/sonic-repro/GR00T-WholeBodyControl
export STATE_ACTION_CVAE_ROOT=~/bly/state-action-cvae
export SONIC_TEACHER_CHECKPOINT=$SONIC_REPO_ROOT/sonic_release/last.pt
export CVAE_STUDENT_CHECKPOINT=/home/helloworld/bly/runs/<cvae_run>/checkpoints/best.pt
export CVAE_BASE_DATASET_RUN=/home/helloworld/bly/runs/<dataset_run>
export SONIC_RUNS_ROOT=~/bly/runs

bash scripts/run_dagger.sh cvae-smoke
bash scripts/run_dagger.sh cvae-pilot
```

smoke marker：

```text
runs/dagger_cvae_smoke_*/markers/dagger_cvae_smoke.ok
```

pilot 结果必须包含 teacher checkpoint、student checkpoint、normalization、action mapping、SONIC/CVAE/DAgger commit、round sample count 和 `execution_pass / information_contract_pass / quality_pass`。

每轮 student-only 评测同时记录 `fixed_standard_normal`（正式部署路径）、`zero` 和
`posterior_mean`（当前 masked rollout 的诊断路径）；三者不合并成单一质量结论。

## Plain MLP 4-motion memorization 路线

2026-10-25 新增 `mlp-smoke` / `mlp-pilot`；2026-10-26 增加 `mlp-debug`、`mlp-scale`、
`mlp-repro`。目标不是在全部 motion 上复现 SONIC，而是在**固定 4 个 motion** 上证明 MLP
能通过 DAgger 部分拟合 release teacher。

观测第一版与 release teacher 的 G1 输入内容一致：

```text
actor_obs
+ command_multi_future_nonflat
+ motion_anchor_ori_b_mf_nonflat
```

三项都不包含 teacher 的 encoder 压缩特征；reference terms 从 release teacher 的 G1 encoder
配置动态读取，并写入 manifest。

motion 由固定 seed 从 motion 目录随机选 4 个，每个 motion 1 个环境。配置中的
`motion_count=4`、`motion_seed=20261024`、`max_unique_motions=4`、
`sample_unique_motions=true`；manifest 记录 requested/loaded keys 和逐 motion 样本数。

本轮修复了 plain MLP 的关键问题：

- `observe()` 返回原始拼接观测；归一化只在 `act()` 和 `train_on_dataset()` 中各应用一次，
  不再出现“dataset 存归一化值、训练再归一化”的双重归一化；MLP 配置在第一轮拟合后冻结
  mean/std；
- action clip 使用 env bounds；如果 bounds 是 ±Infinity，则回退到 `action_clip_value=20.0`；
- teacher load 使用兼容 unpickler，允许旧 `trl` trainer state 中的缺失类；
- plain DAgger mixture 改为 per-step：teacher 和 student Action 每一步都计算，teacher
  标签始终保存，执行动作按 `Bernoulli(beta)` 选择。

评估在 student 自己访问到的状态上查询 teacher，记录 teacher Action RMSE、逐 motion RMSE、
终止率、平均 episode 长度和动作饱和率。`memorization_pass` 只表示**同一批 4 个 motion**
上的记忆/拟合，不表示新 motion 泛化。

```bash
cd ~/bly/dagger
export SONIC_REPO_ROOT=~/bly/sonic-repro/GR00T-WholeBodyControl
export SONIC_TEACHER_CHECKPOINT=$SONIC_REPO_ROOT/sonic_release/last.pt
export SONIC_RUNS_ROOT=~/bly/runs
export DAGGER_MOTION_FILE=$SONIC_REPO_ROOT/data/motion_lib_bones_seed/robot_filtered

bash scripts/run_dagger.sh mlp-smoke
bash scripts/run_dagger.sh mlp-pilot
```

smoke marker 为 `markers/dagger_mlp_smoke.ok`，pilot execution marker 为
`markers/dagger_mlp_execution.ok`；只有满足记忆门禁时才额外写
`markers/dagger_mlp_memorization.ok`。

### 2 轮 repro 结果（2026-10-26）

run：

```text
/home/helloworld/bly/runs/dagger_mlp_repro_20261026_005350
```

合同为 4 env、4 motion、T1000、eval 100、2 轮、per-step mixture、beta `[1.0, 0.5]`。
round 0 为 teacher-only，round 1 的 `student_action_count` 为 `1977/4000`（49.4%），
说明 per-step mixture 生效；4 个 motion 在训练和评估中均有覆盖。

| 指标 | round 0 | round 1 |
|---|---:|---:|
| `student_only.teacher_action_rmse` | 0.8633 | 0.4023 |
| `termination_rate` | 0.0475 | 0.0100 |
| `action_saturation_ratio` | 0.0 | 0.0 |

manifest 为 `memorization_pass=true`，marker
`markers/dagger_mlp_memorization.ok` 存在。4 个 motion 中 3 个的 final RMSE 达到
`<=0.7 × round0`；`looking_by_binoculars_003__A100` 略高于该阈值。

证据边界：这是 **2 轮短 schedule** 的 same-motion 记忆结果，不是完整 6 轮 pilot，
也不是新 motion 泛化。完整 `mlp-pilot` 仍待执行。

### 2 轮分片续训

完整 pilot 可用 2 轮一个进程分片执行：

```bash
# 第一个 shard：rounds 0-1
export DAGGER_ROUNDS_PER_PROCESS=2
unset DAGGER_RESUME_RUN
bash scripts/run_dagger.sh mlp-pilot
RUN=$(ls -dt ~/bly/runs/dagger_mlp_pilot_* | head -1)

# 第二个 shard：rounds 2-3
export DAGGER_RESUME_RUN="$RUN"
bash scripts/run_dagger.sh mlp-pilot

# 第三个 shard：rounds 4-5，写最终 manifest 和质量 marker
bash scripts/run_dagger.sh mlp-pilot
```

恢复规则：

- 只从 `data/dagger/round_XX.pt` 与 `checkpoints/round_XX.pt` 同时存在的连续 round 继续；
- 加载最后一次 round checkpoint 的 model/optimizer/input mean/std；
- 加载已完成 round 的数据 shard 和 `metrics.partial.json`；
- `DAGGER_ROUNDS_PER_PROCESS=2` 限制本次进程只跑 2 轮；
- 中间 shard 只写 `dagger_mlp_manifest.partial.json`、`metrics.partial.json` 和
  `dagger_mlp_shard.ok`；全部 round 完成后才写最终 manifest、`dagger_mlp_execution.ok`
  和（若通过）`dagger_mlp_memorization.ok`；
- 同一时间只允许一个 runner（脚本有 `flock` 和 `pgrep` 检查）。

### 长时 192 步与自动分片

新增 `mlp-long`：10 轮、beta
`[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]`、T1000、每轮 1500 training
step、eval 192。评估现在额外报告：

- `max_episode_length`
- `survived_full_horizon`
- `survival_ratio`

这些是 192 步连续生存的诊断；`memorization_pass` 仍然只代表 same-motion 动作拟合，
不是长时 survival gate。

自动 2 轮分片、失败重试、自动 resume 的入口：

```bash
cd ~/bly/dagger
export SONIC_REPO_ROOT=~/bly/sonic-repro/GR00T-WholeBodyControl
export SONIC_TEACHER_CHECKPOINT=$SONIC_REPO_ROOT/sonic_release/last.pt
export SONIC_RUNS_ROOT=~/bly/runs
export DAGGER_MOTION_FILE=$SONIC_REPO_ROOT/data/motion_lib_bones_seed/robot_filtered
export PYTHON=~/bly/sonic-repro/.venv-sonic/bin/python
export PYTHONPATH="$PWD/src:$HOME/bly/state-action-cvae/src:$SONIC_REPO_ROOT"

export DAGGER_SHARD_ROUNDS=2
export DAGGER_MAX_RETRIES=3
bash scripts/run_dagger_auto.sh mlp-long
```

行为：

- 第一个 shard 创建新 run；后续 shard 自动设置 `DAGGER_RESUME_RUN`；
- 每个 shard 只跑 2 轮；
- 若 shard 崩溃但 completed round 增加，默认继续下一个 shard；
- 若连续 `DAGGER_MAX_RETRIES` 次没有新增 completed round，停止并保留 partial；
- `DAGGER_MAX_SHARDS` 是硬上限；
- 默认不因单次崩溃停止；设置 `DAGGER_STOP_ON_CRASH=1` 后，任何 non-zero shard
  退出都会立即停止；
- 全部 round 完成后自动退出并打印 `memorization_pass`。

历史 5/6 pilot partial（run `011239`）已完成 rounds 0–4；round 4 为
`beta=0`、student 4000/4000、RMSE `0.3705`、termination `0.0125`、
mean episode `38.2`。round 5 纯 student collect 触发 Isaac native SIGSEGV，
因此完整 6 轮未完成；5 轮 partial 的 `same_motion_memorization_pass` 为 true。

### MLP policy 视频回放

`mlp_render` 新增为独立入口：

- 创建 1 个 env、1 个训练 motion；
- 加载 `checkpoints/round_09.pt` 等 MLP checkpoint；
- 在 Isaac 中闭环执行 student policy；
- 记录 `dof_pos`、`root_pos_w`、`root_quat_w` 为
  `data/000000.trajectory.pkl`；
- 使用 `sonic-repro-kit/render_mujoco_trajectory.py` 生成
  `videos/mlp_policy.mp4`。

```bash
cd ~/bly/dagger

export SONIC_REPO_ROOT=~/bly/sonic-repro/GR00T-WholeBodyControl
export SONIC_TEACHER_CHECKPOINT=$SONIC_REPO_ROOT/sonic_release/last.pt
export SONIC_RUNS_ROOT=~/bly/runs
export DAGGER_MOTION_FILE=/home/helloworld/bly/runs/bones_seed_ingest_2048_20260825_132447/data/robot_filtered
export PYTHON=~/bly/sonic-repro/.venv-sonic/bin/python

CKPT=/home/helloworld/bly/runs/dagger_mlp_long_20261026_014151/checkpoints/round_09.pt
bash scripts/render_mlp_policy.sh "$CKPT" 192
```

第二个参数是 motion key，可省略；省略时使用训练 manifest 的第一个
`motion_keys_loaded`。每个 motion 单独渲染，例如：

```bash
bash scripts/render_mlp_policy.sh "$CKPT" fishing_standing_start_R_001__A502 192
```

视频是 **Isaac policy rollout 的 MuJoCo 运动学渲染**，不是 CVAE replay65，
也不是 MuJoCo torque 物理重放。若 policy 在 rollout 中终止，视频会记录到终止前的帧。

## 当前状态

| 项目 | 状态 |
|---|---|
| 独立 dagger 工程 | 已实现 |
| SONIC frozen teacher | 已实现，Ubuntu smoke 已通过 |
| plain MLP DAgger | 已实现，mixture 已改为 per-step |
| plain MLP 4-motion smoke | 已通过（execution） |
| plain MLP 4-motion pilot | 5/6 轮 partial 已通过 memorization；round 5 被 native SIGSEGV 阻塞 |
| plain MLP 2-round sharding/resume | 已实现；中间 shard 只写 partial，最终 shard 写完整结果 |
| plain MLP auto sharding | `run_dagger_auto.sh` 已实现；自动 resume/重试 |
| plain MLP 192-step survival | 已加入 `max_episode_length`/`survival_ratio` 诊断；尚无通过结论 |
| plain MLP policy video | `mlp_render` + `render_mlp_policy.sh` 已实现，Isaac rollout 后 MuJoCo 运动学渲染 |
| same-motion memorization 门禁 | 2 轮 repro 为 true；完整 pilot 仍待评估 |
| CVAE rolling window | 已实现并通过 Windows 单元测试 |
| CVAE 动态 shard | 已实现并通过 shape 测试 |
| CVAE Action mapping | 已实现并通过 CPU round-trip 测试 |
| CVAE DAgger smoke | 待 Ubuntu 执行 |
| CVAE DAgger pilot | smoke 成功后执行 |
| standard-normal 质量 | 待 pilot 评估 |
| teacher action information contract | 待多 motion/reference probe |
