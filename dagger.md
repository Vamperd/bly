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
- `dagger/src/dagger/core/cvae_objective.py`：teacher Action、future State、contact 和总损失的纯 tensor 组合；
- `dagger/src/dagger/core/motion_groups.py`：32 motion 的确定性分组和 pilot 选择；
- `dagger/configs/cvae_smoke.yaml`、`dagger/configs/cvae_pilot.yaml`：独立的通用 CVAE 配置；
- `dagger/configs/cvae_kl2_smoke.yaml`、`cvae_kl2_pilot.yaml`、`cvae_kl2_formal.yaml`：KL2 32-motion 适配配置；
- `dagger/configs/cvae_kl2_motion_manifest.json`：固定 32 keys、4×8 groups 和 pilot keys。

Plain MLP 的通用路径仍可通过 `smoke` 和 `pilot` 使用；4-motion 记忆实验通过
`mlp-smoke` 和 `mlp-pilot` 使用；CVAE 路径通过 `cvae-smoke` 和 `cvae-pilot` 使用。
Plain MLP 的 teacher/student mixture 已改为 **per-step**：每一步独立采样
`teacher_mask ~ Bernoulli(beta)`，teacher Action 始终计算并作为标签，环境执行 teacher 或
student 之一；`beta=0` 是纯 student rollout，`beta=1` 是纯 teacher。通用 CVAE 路径仍使用
per-episode mixture；KL2 32-motion 路径使用 per-step mixture。

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

新增 KL2 适配测试覆盖：

- 32-motion 分组与 pilot 选择；
- causal-prefix Mask；
- per-step teacher/student mixture；
- motion-weighted dataset sampling；
- action fallback clip；
- 双路径损失组合；
- 192 步指标聚合。

当前共 38 个 Windows 单元测试通过。

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

## CVAE KL2 32-motion 适配（2026-10-28）

### 一句话

在 KL2 300k 的 65-token CVAE 基础上，用冻结的 SONIC release teacher 在
**学生自己走到的状态**上现场给出 29 维 Action 标签，做同 32 motion 的
DAgger 动作模仿微调；不训练新模型，也不改 CVAE 架构。

### 数据从哪里来

```text
源 checkpoint:
  /home/helloworld/bly/runs/cvae_long_horizon_train_kl2_fp32_300k_20261016_005837_3380803/checkpoints/best.pt
  SHA256 = f0a254be2cfc04316d2565d1c747d4e60cfbfe16c2fecf873c8904c0982860a7

32-motion 基础集:
  /home/helloworld/bly/runs/cvae_overfit_subset_rebuilt_20260928_193357

long-horizon source:
  /home/helloworld/bly/runs/cvae_long_horizon_dataset_rebuilt_20261018_223710_254008
  groups_sha256  = 66a18e...
  windows_sha256 = 27d1e2...
  normalization  = d9d04c...
```

rebuilt long-horizon 与 KL2 checkpoint 记录的 groups/windows hash 完全一致，
所以 source mix 标记为 `source_mix_exact=true`。运动文件使用：

```text
/home/helloworld/bly/runs/bones_seed_ingest_2048_20260825_132447/data/robot_filtered
```

32 个 motion key、4 组 × 8、pilot 4 个 key 固定记录在：

```text
dagger/configs/cvae_kl2_motion_manifest.json
```

配置继承 KL2 的 `fixed` Mask、`kl_beta=0.002`、`beta_warmup=10000` 和
FP32 eager；不加载源 optimizer。

pilot/formal 使用 `source_cache_size=2048`，在训练前把 1504 个 source window
预加载到 CPU，训练循环不再反复读 HDF5；smoke 用 `1` 验证缓存路径。

训练按 `train_chunk_steps=500` 分块：每块保存 `round_XX_partial.pt` 和
`manifests/dagger_cvae_kl2_round_progress.json`；crash 后从最后一个 chunk
继续，而不是重跑整轮。auto 入口对 CVAE 默认 1 round/process、5 次无进展重试。

smoke 会对照 recorded schema/context 与 live env 的 Action mapping：
`scale` / `nominal` / `clip` 要求严格一致；`offset` 由 startup 的
默认关节随机化重新采样，允许其随机范围，但 canonical Action 仍相对
原始 nominal，语义不变。

### 一轮训练到底发生什么

```text
reset 8 个 env（formal）或 1/4 个 env（smoke/pilot）
↓
每步：
  1. 学生读最近 32 步 State/Action + 当前 State，组成 65-token 条件窗口
  2. 学生用 condition + 固定 standard-normal latent 输出当前 Action
  3. teacher 看同一个 observation，现场算 release Action
  4. 每个 env 独立采样 teacher_mask ~ Bernoulli(beta)
  5. 执行 teacher 或 student 的 Action
  6. 记录 executed / teacher / student Action、下一 State、done
↓
用 anchor=32 的窗口构造训练样本：
  - 未来 State 只作训练 target，不进入 rollout 输入
  - 只监督 anchor 处的 teacher Action
↓
50% batch 来自原始 long-horizon 重建窗口
50% batch 来自本轮 DAgger 动态窗口
↓
AdamW 更新 64M 参数
↓
192 步实时 Isaac 闭环评估
```

`beta` 是每步 teacher/student 混合比例：

```text
1.0 = 全部执行 teacher
0.0 = 全部执行 student
```

### 两条 latent 路径

```text
posterior mean:
  用完整序列的 posterior mean + condition，标签稳定，适合状态重建

standard normal:
  用 N(0,1) latent + condition，和真正部署路径一致

teacher Action loss:
  0.5 × standard-normal
+ 0.5 × posterior-mean
```

训练时 standard-normal epsilon 每个样本重新采样；训练 rollout 时每个 episode
固定一组 epsilon；评估时每个 env 在 192 步内固定同一组，保证 baseline/final 可比。

### 完整损失

```text
future State MSE
+ 0.1 × contact BCE
+ 0.5 × standard-normal teacher MSE
+ 0.5 × posterior-mean teacher MSE
+ 0.002 × KL
```

source batch 仍使用 KL2 原目标：

```text
full State/Action/contact reconstruction
+ 0.002 × KL
```

其中一部分 source batch 会主动使用 causal-prefix Mask，让模型提前适应
“未来 State/Action 全部隐藏”的 DAgger 条件分布。

### 从 4 个 motion 到 32 个 motion

```text
pilot:
  4 env
  4 keys = MLP 4 与 32 的交集 + seed=20261024 补齐
  2 轮：beta = [1.0, 0.5]
  500 rollout steps/轮
  1500 train steps/轮

formal:
  8 env
  每轮使用 G[round % 4] 的 8 个 key
  10 轮：beta = [1.0, 0.9, ..., 0.0]
  1000 rollout steps/轮
  1500 train steps/轮
  末尾单独 32-env × 192 步评估
```

MLP 的 4 个 key 不在 KL2 的 32 个 key 内，所以 pilot 实际使用：

```text
dance_vouge_butterfly_step_270_R_003__A324
cellphone_looking_for_service_walk_R_001__A427
lift_crate_walk_ff_stop_270_R_003__A202
big_heavy_one_hand_front_low_to_front_medium_R_001__A526
```

### 评估与通过标准

主目标是**同 32 motion 的 teacher Action 模仿**：

```text
final canonical teacher RMSE <= 0.5 × baseline
至少 24/32 motion <= 0.7 × baseline
termination_rate 不变差
action_saturation_ratio < 5%
```

pilot 把 24/32 降为 3/4。192 步实时 Isaac 闭环的 survival / episode length
只作诊断，不作为一票否决；它回答“动作更像 teacher 后是否更稳”，
不和 teacher RMSE 合并成一个分数。

固定 32-motion 评估会先跑 source baseline，再跑 final；如果 32-env OOM，
自动回退 4×8 并记录。

每轮保存 `round_XX.pt`；pilot/formal 结束后根据 teacher RMSE 和 survival
选择并复制 `best_teacher_rmse.pt` / `best_survival.pt`。

### 4-motion intensive 训练

`cvae_kl2_pilot_intensive.yaml` 使用当前 pilot 4 个 motion、4 env、12 rounds：
`2000 train steps/round`、`1000 rollout steps/round`、`500 chunk steps`、
`192 eval`。总量约 24,000 optimizer steps；它只是 calibration，不是完整 8–10h。
pilot 实测约 2.72 steps/s；8h 的 round1 实测约 2.1 steps/s，因此 calibration
约 3–4h，8–10h 约 72k–80k steps，即约 18–20 rounds × 4000 steps。放大前先看
calibration 的 teacher RMSE/termination 是否改善。

已提供候选 8h 配置 `cvae_kl2_pilot_intensive_8h.yaml`：
`rounds: 18`、`train_steps_per_round: 4000`、`beta_schedule_extension: hold_last`，
`beta_schedule: [1.0,1.0,0.8,0.6,0.4,0.2,0.1,0.0]`，共约 72k optimizer steps；
若实测偏慢可改为 17 rounds。它只在 calibration 显示改善后使用：

```bash
DAGGER_TEE=1 DAGGER_SHARD_ROUNDS=1 DAGGER_MAX_SHARDS=60 DAGGER_MAX_RETRIES=5 \
bash scripts/run_dagger_auto.sh cvae-kl2-pilot-intensive-8h
```

若实测吞吐不同，只调整 `dagger.rounds` 和 `student.train_steps_per_round`；
`hold_last` 会在 8 个 beta 值之后持续保持最后的 `0.0`，而不是重新回到 teacher-only。
前两轮 beta=1.0 保留给 teacher-only 数据；round 2 起快速降到 0.8/0.6/0.4，
使第 5 轮门禁时学生已访问自己的错误状态。

4m calibration 与 8h 配置都启用中途门禁：

```yaml
dagger:
  early_stop:
    enabled: true
    check_rounds: [4, 9]      # 第 5、10 轮
    baseline_round: 0
    min_teacher_improvement_ratio: 0.98
    termination_tolerance: 0.0
```

第 5/10 轮相对 round0：

```text
teacher_action_rmse <= 0.98 × round0
termination_rate 不变差
```

任一失败即写：

```text
manifests/dagger_cvae_kl2_early_stop.json
markers/dagger_cvae_kl2_early_stop.ok
```

并保留已完成 round/partial metrics；`run_dagger_auto.sh` 检测到该 marker 后退出，不再重试或继续后续 round。

```bash
DAGGER_TEE=1 DAGGER_SHARD_ROUNDS=1 DAGGER_MAX_RETRIES=5 \
bash scripts/run_dagger_auto.sh cvae-kl2-pilot-intensive
```

`source_cache.pt` 会在第一次 shard 写入 run 的 `data/`，后续 shard 直接加载，
避免重复读取 1504 个 HDF5 source window。

### Stepwise causal evaluator（渲染评估路线）

这条路线与训练门禁分开：它从 recorded 192-step group 的 exact initialization
开始，每步重建 causal window，只执行 Action[32]，重复 192 步。它不是
`recursive_state_feedback_b` 的 32-Action 开环块。

先准备 4 个 motion 的 start-at-zero source template：

```bash
export CVAE_ORIGINAL_CHECKPOINT=/home/helloworld/bly/runs/cvae_long_horizon_train_kl2_fp32_300k_20261016_005837_3380803/checkpoints/best.pt
export CVAE_KL2_DATASET_RUN=/home/helloworld/bly/runs/cvae_long_horizon_dataset_rebuilt_20261018_223710_254008
STEPWISE_PREP=/home/helloworld/bly/runs/cvae_kl2_4m_stepwise_prepare_$(date +%Y%m%d_%H%M%S)

bash scripts/run_stepwise_prepare.sh "$STEPWISE_PREP" \
  dance_vouge_butterfly_step_270_R_003__A324 \
  cellphone_looking_for_service_walk_R_001__A427 \
  lift_crate_walk_ff_stop_270_R_003__A202 \
  big_heavy_one_hand_front_low_to_front_medium_R_001__A526
```

再运行 original C 与 fine-tuned C。每次 `env.reset()` 后必须先把 motion command
的 `time_steps` 和 `motion_start_time_steps` 归零，再应用 exact initialization；
否则 reference anchor 可能落在随机初始帧，step 0 就触发 anchor/foot termination。
Isaac 在同一个 Kit 进程内销毁并重建 `SimulationContext` 会卡住，因此每个 motion
必须使用独立进程；`--case-index` 对应 cases manifest 的顺序：

```bash
export SONIC_KIT_ROOT=/home/helloworld/bly/sonic-repro-kit
export STEPWISE_CASES_MANIFEST="$STEPWISE_PREP/manifests/cases.json"
export CVAE_FINETUNED_CHECKPOINT=/home/helloworld/bly/runs/<4m_run>/checkpoints/best_teacher_rmse.pt

declare -a STEPWISE_OUTS
for index in 0 1 2 3; do
  STEPWISE_OUTS[$index]=/home/helloworld/bly/runs/cvae_kl2_4m_stepwise_case${index}_$(date +%Y%m%d_%H%M%S)
  bash scripts/run_stepwise_recursive_causal.sh \
    configs/stepwise_recursive_causal.yaml "${STEPWISE_OUTS[$index]}" \
    --case-index "$index"
done
```

每个 case 输出 source、original C、fine-tuned C 的 193 帧轨迹、fall frame、
State/Root/Action 指标，以及该 case 的 `manifests/stepwise_recursive_causal.json`。
source 是 recorded HDF，不是 live teacher。

### 三层 ghost 视频

对每个 case 分别渲染：

```bash
for index in 0 1 2 3; do
  python ../sonic-repro-kit/render_three_way_ghost.py \
    --manifest "${STEPWISE_OUTS[$index]}/manifests/stepwise_recursive_causal.json" \
    --model /home/helloworld/bly/sonic-repro/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml \
    --output-run "${STEPWISE_OUTS[$index]}/ghost"
done
```

渲染脚本固定为 ghost 布局，没有 `--layout` 参数；`--output-run` 必须是新且
空的目录。

每个 motion 生成：

```text
videos/ghost_source_original_finetuned_<motion>.mp4
videos/source_recorded_<motion>.mp4
videos/original_c_<motion>.mp4
videos/finetuned_c_<motion>.mp4
```

source opaque，original C 与 fine-tuned C 半透明；ghost 使用 source 相机，
三个单独视频各自跟随自己的 root 相机。渲染前会把 original/fine-tuned 的初始
root xy 对齐到 source（只影响可视化，不修改轨迹文件），并在
`raw_first_roots` 记录对齐前坐标；若 policy 在第一步终止，stepwise 必须先修好，
不能只靠渲染补帧。
视频固定 `libx264 + yuv420p` 并启用 `+faststart`；192 步在 50 Hz 下约 3.86 s，
“很短”是协议长度，不是截断。若某个 policy 提前终止，渲染器不再用三者最小帧数
截断，而是保留 source 的完整长度，并对短轨迹重复最后一帧且在视频上标记
“ended N, holding”；`three_way_ghost.json` 记录 `trajectory_frames` 和
`padded_frames`。如果 source 首帧渲染为空白，渲染器会直接报错而不是写出空视频。
已有 `ghost/` 目录时不要复用，改用新的 `ghost_fixed/` 输出目录。

### 8/16 步开环与 recursive B 的边界

当前微调只监督 Action[32]；未来 Action[33:63] 在 DAgger 窗口没有 teacher 标签。
因此：

- 不采用 8/16 步开环；
- 不使用 `recursive_state_feedback_b` 的 32 步块；
- 如果将来必须块执行，应另立 block-causal 微调，监督整个 block 的 Action。

### 从 MLP 路线复用的经验

| MLP 踩坑/有效方法 | CVAE 适配 |
|---|---|
| 双重归一化 | 只用 checkpoint normalization，禁止二次标准化 |
| action bounds ±Inf | fallback clip 20.0，记录 saturation |
| teacher 旧 trl 状态 | 复用兼容 unpickler |
| per-episode mixture 覆盖不足 | 改 per-step Bernoulli |
| 固定 4 motion | 32 keys 对齐；4-motion pilot |
| 跌倒 motion 加权 | formal 按 per-motion termination 加权采样 |
| RMSE 低但 termination 高 | 192 步 survival 单独报告 |
| 无分片/续训 | CVAE round shard + partial + auto resume |
| render 配置不匹配 | CVAE 用 replay65 exact-init，不套 MLP render |
| 动作拟合≠物理质量 | 视频/replay 只作定性，不作门禁 |

### 运行入口

```bash
cd ~/bly/dagger
export SONIC_REPO_ROOT=~/bly/sonic-repro/GR00T-WholeBodyControl
export SONIC_TEACHER_CHECKPOINT=$SONIC_REPO_ROOT/sonic_release/last.pt

bash scripts/run_dagger.sh cvae-kl2-smoke
bash scripts/run_dagger_auto.sh cvae-kl2-pilot
bash scripts/run_dagger_auto.sh cvae-kl2-formal
```

### 证据边界

- 只训练/评估 KL2 的同一批 32 个 motion，不是新 motion 泛化；
- 学生仍然不接收 reference，teacher 的 Action 标签在 partial history 上
  可能不是唯一可辨识的；
- `quality_pass` 只表示同 motion teacher Action 模仿门禁，不表示
  物理 replay、随机生成或长时生存一定通过；
- 所有结果必须写入 `~/bly/runs/dagger_cvae_kl2_*`，不得覆盖源 checkpoint。

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

### SONIC 与 MLP 同动作对比

`render_sonic_policy.sh` 用冻结 release teacher 在同一 motion 上闭环执行；
`render_policy_compare.sh` 会先渲染 SONIC、再渲染 MLP，并用 ffmpeg 生成
**左 SONIC、右 MLP** 的并排视频：

```bash
bash scripts/render_policy_compare.sh "$CKPT" <motion_key> 192
```

四个 motion 全部渲染：

```bash
RUN=/home/helloworld/bly/runs/dagger_mlp_long_20261026_014151
CKPT="$RUN/checkpoints/round_09.pt"
KEYS=$("$PYTHON" -c "import json; d=json.load(open('$RUN/manifests/dagger_mlp_manifest.json')); print('\n'.join(d['motion_keys_loaded']))")

while read -r KEY; do
  bash scripts/render_policy_compare.sh "$CKPT" "$KEY" 192
done <<< "$KEYS"
```

每个对比 run 输出：

```text
~/bly/runs/dagger_mlp_render_<timestamp>/
├── videos/mlp_policy.mp4
├── compare/sonic_left_mlp_right.mp4
├── compare/compare_manifest.json
└── markers/render.ok / render_sonic.ok / compare.ok
```

SONIC 与 MLP 是两次独立 Isaac rollout，各自记录自己的实际状态；对比视频只做
MuJoCo 运动学并排，不是同状态扭矩对比，也不是闭环物理一致性证明。

## 当前 DAgger 训练流程（通俗版）

### 一句话概括

当前 plain MLP 做的事情不是“学一个世界模型”，也不是“预测未来 State”，而是：

> 在 SONIC teacher 实际执行过的同一批 4 个 motion 上，让学生 MLP 在自己会走到的状态上，
> 尽量输出和 teacher 一样的 29 维 raw Action。

所以它本质上是 **在线查询 teacher 标签的行为克隆（DAgger）**。
它不接 CVAE latent，不接 65-token State/Action 序列，不使用 future-State loss。

### 1. 输入、输出和标签分别是什么

```text
MLP 输入:
  actor_obs                                  # SONIC policy 的当前观测/历史
  command_multi_future_nonflat               # G1 reference future command
  motion_anchor_ori_b_mf_nonflat             # G1 reference anchor orientation

MLP 输出:
  29 维 raw Action                            # 与 env.action_space 对齐

监督标签:
  teacher_action = SonicReleaseTeacher.act(observation, done)
```

关键点：

- `teacher_action` 是冻结的 release teacher 在当前 observation 上现场算出来的 Action；
- 它不是 HDF5 里回放的旧 Action，也不是学生实际执行的 Action；
- 即使当前环境执行的是学生 Action，标签仍然保存 teacher Action；
- plain MLP 没有 future-State 标签；future-State loss 只属于 CVAE 路径；
- 输入不包含 motion ID、package、outcome 或 teacher 的 encoder 压缩特征。

输入只做逐维标准化：第一轮用聚合数据拟合 mean/std，之后冻结并随 checkpoint 保存；
输出 Action 不做标准化，也不加 Action residual，直接回归 teacher 的 raw Action。
环境执行前再按 env action bounds 或 `action_clip_value=20.0` 裁剪。

### 2. 一轮 DAgger 到底发生什么

每一轮大致是：

```text
reset 4 个 env（固定 4 个 motion，每个 motion 1 个 env）
↓
for step in rollout_steps:
    1. 学生看 observation，算出 student_action
    2. teacher 看同一个 observation，算出 teacher_action（标签）
    3. 采样 teacher_mask ~ Bernoulli(beta)
    4. 环境执行 teacher_action 或 student_action
    5. 保存 (student_obs, teacher_action, student_action, executed_action,
             done, episode_id, timestep, motion_id, mixture_teacher)
↓
把本轮数据追加到所有历史 round 的数据集
↓
用聚合数据训练 MLP
↓
student-only 评测 192 步
↓
根据每个 motion 的终止率更新下一轮采样权重
↓
保存 round checkpoint
```

`beta` 的含义：

- `beta=1.0`：全部执行 teacher，数据分布接近 teacher；
- `beta=0.0`：全部执行 student，数据分布是学生自己的错误分布；
- 中间值：每一步独立决定执行 teacher 还是 student。

当前是 **per-step** mixture，不是整个 episode 固定。
这样学生能保留一部分 teacher 的稳定轨迹，同时暴露自己的错误状态。

### 3. 为什么标签会“从监督数据获取到参数更新”

完整链路是：

```text
SONIC release checkpoint
  → frozen teacher.act(observation, done)
  → teacher_action（在线标签）
  → 与 student_obs 配对存入 round
  → 所有 round 聚合成训练集
  → MLP forward
  → MSE(student_action, teacher_action)
  → backward
  → AdamW 更新 MLP 参数
```

teacher 从不更新，只有 MLP 参数更新。
这不是强化学习：没有 reward 最大化、没有 value function、没有 policy gradient。
它也不是“先离线存一堆数据再监督”：每一轮都会用当前学生重新访问状态，再查询 teacher。

### 4. 当前 MLP 和训练参数

当前 `mlp-long` 基线：

| 项目 | 当前基线 |
|---|---|
| motion 数 | 4 |
| 每 motion env 数 | 1，共 4 env |
| rollout | 每轮 1000 步 |
| beta | `[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.0]` |
| MLP hidden | `[2048,2048,1024,1024,512,512]` |
| 激活 | SiLU |
| LayerNorm | 无 |
| Dropout | 无 |
| 每轮训练 | 1500 step |
| batch | 256 |
| 学习率 | 固定 `1e-4` |
| weight decay | `1e-4` |
| gradient clip | `1.0` |
| 评估 | student-only 192 步 |
| checkpoint | 每轮 `round_XX.pt` |

当前最终 10 轮 run 为：

```text
/home/helloworld/bly/runs/dagger_mlp_long_20261026_014151
checkpoint: checkpoints/round_09.pt
memorization_pass=true
```

但 round 9 的 `survival_ratio=0.25`，`mean_episode_length=31.2`，所以它只证明动作拟合，
不证明 192 步连续生存。

### 5. 新的大规模方案：`mlp-intensive`

新增配置：

```text
dagger/configs/mlp_intensive.yaml
```

设计目标：**仍然固定 4 个 motion，预算约 1 小时，同时增加数据量、模型容量和训练技巧。**

| 项目 | `mlp-intensive` |
|---|---|
| motion 数 | 4 |
| env | 4，每 motion 1 个 |
| round | 16 |
| 每轮 rollout | 2000 步 |
| 每轮训练 | 2000 step |
| 总 transition | 16 × 4 × 2000 = 128,000 |
| 总 optimizer step | 16 × 2000 = 32,000 |
| 等效 batch 采样 | 32,000 × 512 = 16.4M 个样本 |
| MLP hidden | `[3072,3072,2048,1024,512,512]` |
| LayerNorm | 开启 |
| Dropout | 0.0（先不加；过拟合时再试 0.05） |
| batch | 512 |
| 学习率 | peak `1e-4`，cosine 衰减到 `1e-6` |
| warmup | 2000 step |
| weight decay | `1e-4` |
| gradient clip | `1.0` |
| beta | `[1.0,1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0,0.0]` |
| 评估 | 每轮 student-only 192 步 |

相对当前基线：

- MLP 参数量约增加 2 倍多（取决于实际输入维度）；
- 每轮 rollout 翻倍；
- 训练步数、batch、round 数都增加；
- 学习率从固定值改为 warmup + cosine；
- 增加 LayerNorm，改善宽 MLP 的优化稳定性；
- 保留 gradient clip 和 weight decay；
- 仍然不接 future State、不接 CVAE latent、不改 teacher。

时间预算不是硬编码保证。按当前 `mlp-long` 每轮约 100–120 秒、rollout 翻倍估算，
16 轮加 8 个两轮 shard 的启动时间，大约 **55–70 分钟**；如果首两个 shard 明显更慢，
优先减少 round 数，而不是降低 eval horizon。

### 6. 对“跌倒多”的 motion 增加训练

每轮 student-only 评估会得到每个 motion 的：

```text
teacher_action_rmse
terminations
termination_rate = terminations / steps
```

下一轮训练时，按终止率给 motion 加权：

```text
weight(motion) = 1 + 3 × termination_rate(motion) / max_termination_rate
```

效果：

- 全部不摔：权重全为 1；
- 某个 motion 摔得最多：它最多得到 4 倍采样权重；
- 权重作用于**训练样本采样**，不是额外增加该 motion 的 env 数量；
- 第 0 轮没有历史评估，使用均匀采样；
- 第 1 轮开始使用第 0 轮评估得到的权重；
- 权重和每轮样本数会写入 `metrics.partial.json`，便于检查。

如果要做“额外 rollout 只给跌倒 motion”，那是另一种更重的 motion-balanced 环境分配方案；
当前实现先采用更稳定的训练样本加权。

### 7. 实时监督训练效果

推荐用自动分片入口：

```bash
cd ~/bly/dagger
export SONIC_REPO_ROOT=~/bly/sonic-repro/GR00T-WholeBodyControl
export SONIC_TEACHER_CHECKPOINT=$SONIC_REPO_ROOT/sonic_release/last.pt
export SONIC_RUNS_ROOT=~/bly/runs
export DAGGER_MOTION_FILE=/home/helloworld/bly/runs/bones_seed_ingest_2048_20260825_132447/data/robot_filtered
export PYTHON=~/bly/sonic-repro/.venv-sonic/bin/python

export DAGGER_SHARD_ROUNDS=2
export DAGGER_MAX_RETRIES=3
export DAGGER_TEE=1
bash scripts/run_dagger_auto.sh mlp-intensive
```

另开一个终端看实时日志：

```bash
RUN=$(ls -dt ~/bly/runs/dagger_mlp_intensive_* | head -1)
tail -f "$(ls -dt ~/bly/runs/dagger_mlp_intensive_command_*.log | head -1)"
```

训练每 100 step 打印一次：

```text
[dagger] train round=... step=.../2000 loss=... lr=... grad=...
```

重点看：

- `loss` 是否总体下降；
- `lr` 是否按 warmup/cosine 变化；
- `grad` 是否突然爆炸；clip 后仍出现异常大值要停下检查；
- 是否出现 `non-finite`；
- 每轮 `eval_rmse` 是否下降；
- 逐 motion 的 `termination_rate` 是否下降；
- `survival_ratio` 是否改善；
- `action_saturation_ratio` 是否过高。

查看逐轮摘要：

```bash
python - <<'PY'
import json, pathlib
run = sorted(pathlib.Path.home().joinpath("bly/runs").glob("dagger_mlp_intensive_*"))[-1]
p = run / "manifests" / "dagger_mlp_metrics.partial.json"
data = json.loads(p.read_text())
for r in data:
    s = r["student_only"]
    print(
        f"round={r['round']:02d} beta={r['beta']:.2f} "
        f"rmse={s['teacher_action_rmse']:.4f} "
        f"term={s['termination_rate']:.4f} "
        f"mean_ep={s['mean_episode_length']:.1f} "
        f"max_ep={s.get('max_episode_length', -1):.0f} "
        f"survival={s.get('survival_ratio', -1):.2f}"
    )
    for key, item in s["teacher_action_rmse_by_motion"].items():
        print(
            f"    {key}: rmse={item['teacher_action_rmse']:.4f} "
            f"term={item['termination_rate']:.4f} "
            f"weight={r.get('next_motion_weights', {}).get(key, 1.0):.2f}"
        )
PY
```

也可以定时刷新：

```bash
watch -n 10 "python -c \"import json,pathlib; p=sorted(pathlib.Path.home().joinpath('bly/runs').glob('dagger_mlp_intensive_*'))[-1]/'manifests'/'dagger_mlp_metrics.partial.json'; print(len(json.loads(p.read_text())))\""
```

判断原则：

- **训练 loss 降、eval RMSE 也降**：方向正常；
- **训练 loss 降、eval RMSE 不降**：更像过拟合、数据覆盖不足或观测-标签不匹配，不要只加步数；
- **某个 motion 单独差**：先看加权是否生效，再看该 motion 的终止率和动作饱和；
- **RMSE 低但 termination 高**：动作拟合和闭环稳定性不是一回事，可能需要更多 DAgger 状态覆盖；
- **所有 motion 都高且 loss 不降**：检查学习率、归一化、输入维度、teacher 标签是否一致。

### 8. 新模型渲染

`mlp-intensive` 的 hidden 维度和 LayerNorm 与旧 `mlp-long` 不同，渲染必须使用匹配配置：

```bash
export DAGGER_RENDER_CONFIG=~/bly/dagger/configs/mlp_render_intensive.yaml
bash scripts/render_mlp_policy.sh "$CKPT" "$MOTION_KEY" 192
```

否则 checkpoint 的 `hidden_dims/layer_norm` 校验会拒绝加载。
旧的 `mlp-long` checkpoint 继续使用 `mlp_render.yaml`。

### 9. 边界

- 只训练和评估同一批 4 个 motion；不是新 motion 泛化；
- `memorization_pass` 是动作拟合门禁，不是 192 步生存门禁；
- 192 步 survival 只有 4 个 env 的粗粒度值；
- 当前 `mlp-intensive` 只是配置和代码方案，尚未在 Ubuntu 实际跑完；
- 视频是 Isaac rollout 后的 MuJoCo 运动学渲染，不是扭矩物理重放。

## 当前状态

| 项目 | 状态 |
|---|---|
| 独立 dagger 工程 | 已实现 |
| SONIC frozen teacher | 已实现，Ubuntu smoke 已通过 |
| plain MLP DAgger | 已实现，mixture 已改为 per-step |
| plain MLP 4-motion smoke | 已通过（execution） |
| plain MLP 4-motion pilot | 5/6 轮 partial 已通过 memorization；round 5 被 native SIGSEGV 阻塞 |
| plain MLP 4-motion long | `dagger_mlp_long_20261026_014151` 10/10 轮完成，`memorization_pass=true` |
| plain MLP 2-round sharding/resume | 已实现；中间 shard 只写 partial，最终 shard 写完整结果 |
| plain MLP auto sharding | `run_dagger_auto.sh` 已实现；自动 resume/重试 |
| plain MLP 192-step survival | 已加入 `max_episode_length`/`survival_ratio` 诊断；long run final survival=0.25，尚无通过结论 |
| `mlp-intensive` 16-round 1h 方案 | 配置/代码已实现，尚未 Ubuntu 执行 |
| 跌倒 motion 加权采样 | 已实现；按 `termination_rate` 最多 4× 训练采样 |
| cosine warmup / LayerNorm / 更大 MLP | 已实现；`mlp-intensive` 使用 |
| plain MLP policy video | `mlp_render` + `render_mlp_policy.sh` 已实现，Isaac rollout 后 MuJoCo 运动学渲染 |
| SONIC vs MLP compare video | `render_sonic_policy.sh` + `render_policy_compare.sh` 已实现，左 SONIC 右 MLP |
| same-motion memorization 门禁 | 10 轮 long run 为 true；2 轮/5 轮历史结果保留 |
| CVAE rolling window | 已实现并通过 Windows 单元测试 |
| CVAE 动态 shard | 已实现并通过 shape 测试 |
| CVAE Action mapping | 已实现并通过 CPU round-trip 测试 |
| CVAE DAgger smoke | 待 Ubuntu 执行 |
| CVAE DAgger pilot | smoke 成功后执行 |
| CVAE KL2 32-motion 适配 | 已实现；Windows 45 tests 通过 |
| CVAE KL2 motion manifest | 32 keys / 4×8 groups / pilot 4 keys 已固定 |
| CVAE KL2 source mix | `source_mix_exact=true`；rebuilt long-horizon hash 与 KL2 一致 |
| CVAE KL2 smoke | Ubuntu execution PASS；quality 为单轮 smoke 预期 false |
| CVAE KL2 pilot | round 0 step 800 native SIGSEGV；source cache + 500-step chunk resume + auto retry 已加，待重跑 |
| CVAE KL2 formal | 待 pilot 通过 |
| CVAE 4m intensive | `cvae_kl2_pilot_intensive` 配置/命令已实现；12×2000 候选，待 Ubuntu |
| CVAE source cache 持久化 | 已实现 `data/source_cache.pt`；待 Ubuntu 验证 |
| CVAE 4m early-stop 门禁 | 第 5/10 轮比较 teacher RMSE/termination；失败写 marker 并停止 auto |
| Stepwise causal prepare | `dagger.stepwise_prepare` + start-at-zero template 已实现；待 Ubuntu |
| Stepwise causal replay | `dagger.stepwise_recursive_causal` 已实现；每步 1 Action；待 Ubuntu |
| Three-way ghost renderer | `sonic-repro-kit/render_three_way_ghost.py` 已实现；待 Ubuntu |
| CVAE KL2 192-step survival | 实时 Isaac 闭环诊断；待 Ubuntu |
| standard-normal 质量 | 待 pilot 评估 |
| teacher action information contract | 待多 motion/reference probe |
