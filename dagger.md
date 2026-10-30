# SONIC teacher + CVAE DAgger 实施进度

最后更新：2026-10-26。本文件是 DAgger 命令与配置的主副本；Ubuntu 问题→修复链与 8h 结果见 process.md §23–24。

## 当前架构

唯一实现目录是工作区根目录 `dagger/`。SONIC nested repo 只提供 release teacher、原始环境和 G1 reference；`state-action-cvae` 只提供 65-token CVAE 模型、数据集和 normalization contract。

CVAE 相关文件：

- `dagger/src/dagger/runtime/cvae_window.py`：65-state/64-action rolling window，anchor 固定 32，causal mask 与 future-State loss mask；
- `dagger/src/dagger/runtime/sonic_env.py`：只读 SONIC recorder 的 Physics State，raw Action ↔ 29 维 canonical Action；
- `dagger/src/dagger/core/cvae_dataset.py`：保存并均衡采样各 round 动态窗口；
- `dagger/src/dagger/core/cvae_loop.py`：Isaac 闭环收集 teacher/student/executed Action 和未来 State；
- `dagger/src/dagger/adapters/cvae_student.py`：加载 Stage-C 65-token CVAE，rollout 用 condition + fixed standard-normal latent；训练用 future-State、teacher-Action 和 KL；
- `dagger/src/dagger/core/cvae_objective.py`：teacher Action、future State、contact 与总损失；
- `dagger/src/dagger/core/motion_groups.py`：32 motion 确定性分组与 pilot 选择；
- `dagger/configs/cvae_smoke.yaml`、`dagger/configs/cvae_pilot.yaml`：通用 CVAE 配置；
- `dagger/configs/cvae_kl2_smoke.yaml`、`cvae_kl2_pilot.yaml`、`cvae_kl2_formal.yaml`：KL2 32-motion 配置；
- `dagger/configs/cvae_kl2_motion_manifest.json`：固定 32 keys、4×8 groups 和 pilot keys。

入口：plain MLP 通用 `smoke`/`pilot`；4-motion 记忆 `mlp-smoke`/`mlp-pilot`；CVAE `cvae-smoke`/`cvae-pilot`。
mixture：plain MLP 与 KL2 32-motion 为 **per-step**（每步 `teacher_mask ~ Bernoulli(beta)`，teacher Action 始终计算并作标签，
`beta=0` 纯 student、`beta=1` 纯 teacher）；通用 CVAE 路径为 per-episode。

## 术语和数据语义

- **DAgger**：学生进入环境，教师在学生实际访问的状态上给动作标签，聚合数据训练学生。
- **CVAE**：用隐变量补全被 Mask 的 State/Action 序列；训练可用完整序列的 posterior，部署只用 condition + standard-normal latent。
- **Mask**：告诉 condition encoder 哪些值被隐藏；rollout 中历史 State/Action 可见，未来 State/Action 隐藏。
- **canonical Action**：仿真器处理后的关节目标减 nominal joint position；SONIC raw Action 经 action scale、offset、clip 转换到该坐标。
- **future State target**：rollout 时不可见、执行后可得，只作训练标签。

```text
State[0:32]     history
State[32]       current state
State[33:64]    masked future state
Action[0:31]    executed action history
Action[32:63]   masked future action
output.action[32] = executable student action
```

未来 State 只进入训练 target。当前 CVAE 不接收 SONIC reference，这一版是动态误差补全诊断；manifest 必须单独记录
`information_contract_pass`，不能直接解释为完整 reference-conditioned policy imitation。

## CVAE 固定训练合同（通用路径）

- beta：round 0–3 依次 `1.0 / 0.5 / 0.2 / 0.0`，按 episode 采样、episode 内不变，执行动作只来自 teacher 或 student 之一。
- 训练混合 50% 原始 CVAE 窗口 + 50% DAgger 动态窗口；normalizer 从原 CVAE 数据集加载并冻结。
- 动态 loss：`L_future_state + 0.1 * L_future_contact + 1.0 * L_teacher_action(anchor=32) + 0.001 * KL(q(z|x)||N(0,I))`。
- AdamW 学习率 `1e-5`，micro-batch `32`，gradient clip `1.0`，每轮 `2000` steps；smoke 将训练 steps / 环境步数缩为 2 / 8。

## 静态验证

Windows 单元测试覆盖：Python compile；CVAE window shape 与 causal Mask；future State target；round-balanced dataset；
canonical/raw action round-trip；plain MLP core、原始观测单次归一化、冻结归一化；checkpoint round-trip、action bounds clip、
motion 选择；student-only teacher RMSE/per-motion；per-step mixture 比例；data shard round-trip；same-motion memorization 门禁；
SONIC reference term 与 motion key contract。KL2 适配另覆盖 32-motion 分组与 pilot 选择、causal-prefix Mask、per-step mixture、
motion-weighted sampling、action fallback clip、双路径损失、192 步指标聚合。当前共 38 个 Windows 单元测试通过。

```powershell
$env:PYTHONPATH="$PWD/dagger/src"
python -m compileall -q dagger/src dagger/tests
python -m unittest discover -s dagger/tests -v
```

## Ubuntu CVAE 执行入口（通用路径）

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

- smoke marker：`runs/dagger_cvae_smoke_*/markers/dagger_cvae_smoke.ok`。
- pilot 结果须含 teacher/student checkpoint、normalization、action mapping、SONIC/CVAE/DAgger commit、round sample count 和
  `execution_pass / information_contract_pass / quality_pass`。
- 每轮 student-only 评测分别记录 `fixed_standard_normal`（正式部署路径）、`zero`、`posterior_mean`（诊断路径），不合并成单一质量结论。

## CVAE KL2 32-motion 适配（2026-10-28）

在 KL2 300k 的 65-token CVAE 上，用冻结 SONIC release teacher 在**学生自己走到的状态**上现场给 29 维 Action 标签，
做同 32 motion 的 DAgger 动作模仿微调；不训练新模型，不改 CVAE 架构。

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

运动文件:
  /home/helloworld/bly/runs/bones_seed_ingest_2048_20260825_132447/data/robot_filtered

32 keys / 4×8 groups / pilot 4 keys:
  dagger/configs/cvae_kl2_motion_manifest.json
```

- rebuilt long-horizon 与 KL2 checkpoint 记录的 groups/windows hash 完全一致，标记 `source_mix_exact=true`。
- 继承 KL2 的 `fixed` Mask、`kl_beta=0.002`、`beta_warmup=10000`、FP32 eager；不加载源 optimizer。
- pilot/formal `source_cache_size=2048`，训练前把 1504 个 source window 预加载到 CPU，不再反复读 HDF5；smoke 用 `1`。
- `train_chunk_steps=500` 分块保存 `round_XX_partial.pt` 与 `manifests/dagger_cvae_kl2_round_progress.json`，crash 后从最后
  chunk 继续；auto 入口对 CVAE 默认 1 round/process、5 次无进展重试。
- smoke 对照 recorded schema/context 与 live env 的 Action mapping：`scale`/`nominal`/`clip` 严格一致；`offset` 由 startup
  默认关节随机化重新采样，允许其随机范围，canonical Action 仍相对原始 nominal。

### 一轮训练

reset 8 个 env（formal）或 1/4 个 env（smoke/pilot）。每步：学生用最近 32 步 State/Action + 当前 State 组成 65-token 条件窗口，
以 condition + 固定 standard-normal latent 输出当前 Action；teacher 在同一 observation 上算 release Action；每个 env 独立采样
`teacher_mask ~ Bernoulli(beta)` 执行其一；记录 executed/teacher/student Action、下一 State、done。之后用 anchor=32 窗口构造样本
（未来 State 只作 target，只监督 anchor 处 teacher Action），50% batch 来自原始 long-horizon 重建窗口、50% 来自本轮动态窗口，
AdamW 更新 64M 参数，再做 192 步实时 Isaac 闭环评估。

```text
DAgger batch: future State MSE + 0.1 × contact BCE
              + 0.5 × standard-normal teacher MSE + 0.5 × posterior-mean teacher MSE + 0.002 × KL
source batch: full State/Action/contact reconstruction + 0.002 × KL（部分使用 causal-prefix Mask）
```

posterior mean 路径标签稳定、适合重建；standard-normal 路径与部署一致。epsilon：训练每样本重采样；训练 rollout 每 episode 固定；
评估每 env 在 192 步内固定，保证 baseline/final 可比。

### pilot / formal

```text
pilot:  4 env；4 keys；2 轮 beta=[1.0,0.5]；500 rollout steps/轮；1500 train steps/轮
formal: 8 env；每轮使用 G[round % 4] 的 8 个 key；10 轮 beta=[1.0,0.9,...,0.0]；1000 rollout / 1500 train steps/轮；
        末尾单独 32-env × 192 步评估
```

MLP 的 4 个 key 不在 KL2 的 32 个 key 内，pilot 取 MLP 4 与 32 的交集 + seed=20261024 补齐，实际为：

```text
dance_vouge_butterfly_step_270_R_003__A324
cellphone_looking_for_service_walk_R_001__A427
lift_crate_walk_ff_stop_270_R_003__A202
big_heavy_one_hand_front_low_to_front_medium_R_001__A526
```

通过标准（主目标是同 32 motion 的 teacher Action 模仿；pilot 把 24/32 降为 3/4）：

```text
final canonical teacher RMSE <= 0.5 × baseline
至少 24/32 motion <= 0.7 × baseline
termination_rate 不变差
action_saturation_ratio < 5%
```

192 步闭环 survival / episode length 只作诊断，不一票否决，也不与 teacher RMSE 合并成一个分数。固定 32-motion 评估先跑
source baseline 再跑 final；32-env OOM 时自动回退 4×8 并记录。每轮保存 `round_XX.pt`，结束后按 teacher RMSE 与 survival
复制 `best_teacher_rmse.pt` / `best_survival.pt`。

### 4-motion intensive 与 8h

- `cvae_kl2_pilot_intensive.yaml`：pilot 4 motion、4 env、12 rounds × 2000 train steps，1000 rollout、500 chunk、192 eval，
  约 24,000 optimizer steps；只作 calibration（约 3–4h），放大前先看 teacher RMSE/termination 是否改善。
- `cvae_kl2_pilot_intensive_8h.yaml`：`rounds: 18`、`train_steps_per_round: 4000`、`beta_schedule_extension: hold_last`、
  `beta_schedule: [1.0,1.0,0.8,0.6,0.4,0.2,0.1,0.0]`，约 72k optimizer steps；`hold_last` 在 8 个值之后保持 `0.0`。
  前两轮 teacher-only，round 2 起快速降到 0.8/0.6/0.4，使第 5 轮门禁时学生已访问自己的错误状态。实测 pilot ≈2.72 steps/s、
  8h round1 ≈2.1 steps/s；吞吐不同时只调 `dagger.rounds` 与 `student.train_steps_per_round`（偏慢可改 17 rounds）。
- intensive、8h、formal 启用 `post_fall_continuation: true`：训练时关闭 Isaac auto reset，done 不清 CVAE history，teacher 收到
  done=false；`error_anchor_pos`/`error_body_pos` 超过每个 motion 前 10 步 reference error baseline 3 倍并持续 50 步才停止该 env
  采样。没有绝对高度/竖直判断，爬行不会误判为摔倒。评估仍用标准 reset；已完成的历史 run 不受影响。
- `source_cache.pt` 在第一次 shard 写入 run 的 `data/`，后续 shard 直接加载，避免重复读取 1504 个 HDF5 source window。
- 8h 结果（18/18 完成、`quality_pass=false`、overall ratio 0.55424）见 process.md §24。

中途门禁（4m calibration 与 8h 均启用）：

```yaml
dagger:
  early_stop:
    enabled: true
    check_rounds: [4, 9]      # 第 5、10 轮
    baseline_round: 0
    min_teacher_improvement_ratio: 0.98
    termination_tolerance: 0.0
```

第 5/10 轮相对 round0 要求 `teacher_action_rmse <= 0.98 × round0` 且 termination_rate 不变差；任一失败写
`manifests/dagger_cvae_kl2_early_stop.json` 与 `markers/dagger_cvae_kl2_early_stop.ok`，保留已完成 round/partial metrics；
`run_dagger_auto.sh` 检测到该 marker 后退出，不再重试或继续后续 round。

```bash
DAGGER_TEE=1 DAGGER_SHARD_ROUNDS=1 DAGGER_MAX_RETRIES=5 \
bash scripts/run_dagger_auto.sh cvae-kl2-pilot-intensive

DAGGER_TEE=1 DAGGER_SHARD_ROUNDS=1 DAGGER_MAX_SHARDS=60 DAGGER_MAX_RETRIES=5 \
bash scripts/run_dagger_auto.sh cvae-kl2-pilot-intensive-8h
```

### Stepwise causal evaluator（渲染评估路线）

与训练门禁分开：从 recorded 192-step group 的 exact initialization 开始，每步重建 causal window、只执行 Action[32]，重复 192 步；
不是 `recursive_state_feedback_b` 的 32-Action 开环块。先准备 4 个 motion 的 start-at-zero source template：

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

再运行 original C 与 fine-tuned C。每次 `env.reset()` 后必须先把 motion command 的 `time_steps` 和 `motion_start_time_steps`
归零再应用 exact init（否则 step 0 就可能触发 anchor/foot termination）；同一 Kit 进程内重建 `SimulationContext` 会卡住，
所以每个 motion 独立进程，`--case-index` 对应 cases manifest 的顺序：

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

每个 case 输出 source、original C、fine-tuned C 的 193 帧轨迹、fall frame、State/Root/Action 指标和
`manifests/stepwise_recursive_causal.json`；source 是 recorded HDF，不是 live teacher。评估禁用 Isaac auto reset，首次 termination
只记录 fall_frame/terms 并继续到 192 步；exact init 后按 reference anchor 平移 root 并打印 `source_world_root_alignment`。

### 三层 ghost 视频

```bash
for index in 0 1 2 3; do
  python ../sonic-repro-kit/render_three_way_ghost.py \
    --manifest "${STEPWISE_OUTS[$index]}/manifests/stepwise_recursive_causal.json" \
    --model /home/helloworld/bly/sonic-repro/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml \
    --output-run "${STEPWISE_OUTS[$index]}/ghost"
done
```

- 固定 ghost 布局，没有 `--layout`；`--output-run` 必须是新且空的目录（已有 `ghost/` 时改用 `ghost_fixed/`）。
- 每个 motion 输出 `videos/ghost_source_original_finetuned_<motion>.mp4`、`videos/source_recorded_<motion>.mp4`、
  `videos/original_c_<motion>.mp4`、`videos/finetuned_c_<motion>.mp4`。
- source opaque，original/fine-tuned 半透明；ghost 用 source 相机，单独视频跟随各自 root 相机；初始 root xyz 对齐到 source
  （只影响可视化，不改轨迹文件，`raw_first_roots` 记录对齐前坐标）。默认黑色地面，checker 可选，只 tint robot body geom。
- `libx264 + yuv420p + +faststart`；192 步 / 50 Hz ≈3.86 s 是协议长度，不是截断。提前终止的轨迹保留 source 完整长度、
  重复末帧并标注 “ended N, holding”，`three_way_ghost.json` 记录 `trajectory_frames`/`padded_frames`；source 首帧空白直接报错。
  policy 在第一步终止时必须先修 stepwise，不能靠渲染补帧。

### 192-step recorded-state replay

`dagger.cvae_recorded_state_replay` 用于四路视频：initial 用 `s0` real + `s1..s64` expected、Action 全 mask，一次生成 64 Action
执行前 32；之后 `t=32..191` 用 `33 real + 32 expected` State、32 已执行 + 32 masked Action，预测并执行 `output.action[32]`，
窗口每步前进 1，最终 192 Action / 193 State。不判断 fall、不 reset（关闭 auto reset），输出 source/original/fine/ghost 四路视频。

```bash
python -m dagger.cvae_recorded_state_replay \
  --cases-manifest "$STEPWISE_PREP/manifests/cases.json" \
  --case-index 0 \
  --config configs/stepwise_recursive_causal.yaml \
  --original-checkpoint "$CVAE_ORIGINAL_CHECKPOINT" \
  --finetuned-checkpoint "$CVAE_FINETUNED_CHECKPOINT" \
  --output-run "$OUT"

python ../sonic-repro-kit/render_three_way_ghost.py \
  --manifest "$OUT/manifests/recorded_state_replay.json" \
  --model .../g1_29dof_rev_1_0.xml \
  --ground-style black \
  --output-run "$OUT/ghost"
```

### 8/16 步开环与 recursive B 的边界

当前微调只监督 Action[32]，未来 Action[33:63] 在 DAgger 窗口没有 teacher 标签。因此不采用 8/16 步开环，不使用
`recursive_state_feedback_b` 的 32 步块；若将来必须块执行，应另立 block-causal 微调，监督整个 block 的 Action（见下文 v2）。

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
- 学生仍不接收 reference，teacher 的 Action 标签在 partial history 上可能不是唯一可辨识的；
- `quality_pass` 只表示同 motion teacher Action 模仿门禁，不表示物理 replay、随机生成或长时生存一定通过；
- 所有结果必须写入 `~/bly/runs/dagger_cvae_kl2_*`，不得覆盖源 checkpoint。

## Plain MLP 4-motion memorization 路线

2026-10-25 新增 `mlp-smoke` / `mlp-pilot`；2026-10-26 增加 `mlp-debug`、`mlp-scale`、`mlp-repro`。目标是在**固定 4 个 motion**
上证明 MLP 能通过 DAgger 部分拟合 release teacher，不是在全部 motion 上复现 SONIC。本质是在线查询 teacher 标签的行为克隆：
不接 CVAE latent、不接 65-token 序列、无 future-State loss；MSE(student_action, teacher_action) + AdamW 只更新 MLP，teacher 冻结，
没有 reward/value/policy gradient。

- 输入：`actor_obs + command_multi_future_nonflat + motion_anchor_ori_b_mf_nonflat`，与 release teacher 的 G1 输入一致；不含
  teacher encoder 压缩特征、motion ID、package、outcome；reference terms 从 teacher 的 G1 encoder 配置动态读取并写入 manifest。
- 标签：`teacher_action = SonicReleaseTeacher.act(observation, done)` 现场计算，不是 HDF5 旧 Action，也不是执行的 Action。
- 输出：29 维 raw Action，不标准化、无 residual；执行前按 env action bounds 或 `action_clip_value=20.0` 裁剪。
- motion：固定 seed 选 4 个，每 motion 1 env；`motion_count=4`、`motion_seed=20261024`、`max_unique_motions=4`、
  `sample_unique_motions=true`；manifest 记录 requested/loaded keys 和逐 motion 样本数。
- 每轮：reset 4 env → 每步 student/teacher 各算 Action、per-step Bernoulli(beta) 执行其一，保存 `(student_obs, teacher_action,
  student_action, executed_action, done, episode_id, timestep, motion_id, mixture_teacher)` → 聚合所有 round 训练 → student-only
  192 步评测 → 按终止率更新下一轮采样权重 → 保存 round checkpoint。

问题→修复：
- 双重归一化 → `observe()` 返回原始拼接观测，归一化只在 `act()` 和 `train_on_dataset()` 各一次；第一轮拟合 mean/std 后冻结并随
  checkpoint 保存。
- env bounds 为 ±Infinity → 回退 `action_clip_value=20.0`。
- 旧 `trl` trainer state 缺失类 → teacher 用兼容 unpickler 加载。
- per-episode mixture 覆盖不足 → per-step，teacher 标签始终保存。

评估在 student 访问的状态上查询 teacher，记录 teacher Action RMSE、逐 motion RMSE、终止率、平均 episode 长度、动作饱和率。
`memorization_pass` 只表示**同一批 4 个 motion** 上的记忆/拟合，不表示新 motion 泛化。

```bash
cd ~/bly/dagger
export SONIC_REPO_ROOT=~/bly/sonic-repro/GR00T-WholeBodyControl
export SONIC_TEACHER_CHECKPOINT=$SONIC_REPO_ROOT/sonic_release/last.pt
export SONIC_RUNS_ROOT=~/bly/runs
export DAGGER_MOTION_FILE=$SONIC_REPO_ROOT/data/motion_lib_bones_seed/robot_filtered

bash scripts/run_dagger.sh mlp-smoke
bash scripts/run_dagger.sh mlp-pilot
```

markers：smoke `markers/dagger_mlp_smoke.ok`；pilot execution `markers/dagger_mlp_execution.ok`；只有满足记忆门禁时额外写
`markers/dagger_mlp_memorization.ok`。

2 轮 repro（2026-10-26）：run `/home/helloworld/bly/runs/dagger_mlp_repro_20261026_005350`，4 env、4 motion、T1000、eval 100、
beta `[1.0, 0.5]`；round 1 `student_action_count` 1977/4000（49.4%），per-step mixture 生效。`student_only.teacher_action_rmse`
0.8633→0.4023，`termination_rate` 0.0475→0.0100，`action_saturation_ratio` 0.0；`memorization_pass=true`，3/4 motion 达
`<=0.7 × round0`（`looking_by_binoculars_003__A100` 略高）。这只是 2 轮短 schedule 的 same-motion 记忆结果。

### 分片续训与自动分片

手动 2 轮一个进程：

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

恢复规则：只从 `data/dagger/round_XX.pt` 与 `checkpoints/round_XX.pt` 同时存在的连续 round 继续，加载最后 checkpoint 的
model/optimizer/input mean/std、已完成数据 shard 与 `metrics.partial.json`；中间 shard 只写 `dagger_mlp_manifest.partial.json`、
`metrics.partial.json` 和 `dagger_mlp_shard.ok`，全部完成后才写最终 manifest、`dagger_mlp_execution.ok` 和（若通过）
`dagger_mlp_memorization.ok`；同一时间只允许一个 runner（`flock` + `pgrep`）。

`mlp-long`：10 轮、beta `[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]`、T1000、每轮 1500 training step、eval 192；额外报告
`max_episode_length`、`survived_full_horizon`、`survival_ratio`（192 步生存诊断，不是 survival gate）。自动分片入口：

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

`run_dagger_auto.sh`：首个 shard 新建 run，后续自动设 `DAGGER_RESUME_RUN`；每 shard 跑 `DAGGER_SHARD_ROUNDS` 轮；崩溃但
completed round 增加则继续；连续 `DAGGER_MAX_RETRIES` 次无新增则停止并保留 partial；`DAGGER_MAX_SHARDS` 为硬上限；
`DAGGER_STOP_ON_CRASH=1` 时任何 non-zero shard 退出立即停止；全部完成后打印 `memorization_pass`。

结果：
- 历史 5/6 pilot partial（run `011239`）完成 rounds 0–4；round 4 `beta=0`、student 4000/4000、RMSE `0.3705`、termination
  `0.0125`、mean episode `38.2`；round 5 纯 student collect 触发 Isaac native SIGSEGV，5 轮 partial 的
  `same_motion_memorization_pass` 为 true。
- `mlp-long` 最终 run `/home/helloworld/bly/runs/dagger_mlp_long_20261026_014151`，`checkpoints/round_09.pt`，10/10 轮、
  `memorization_pass=true`；round 9 `survival_ratio=0.25`、`mean_episode_length=31.2`，只证明动作拟合，不证明 192 步连续生存。

### MLP 配置：`mlp-long` 与 `mlp-intensive`

`dagger/configs/mlp_intensive.yaml` 仍固定 4 个 motion，预算约 1h，增加数据量、模型容量和训练技巧；仍不接 future State、
不接 CVAE latent、不改 teacher。

| 项目 | `mlp-long` | `mlp-intensive` |
|---|---|---|
| env | 4（每 motion 1） | 4（每 motion 1） |
| round × rollout | 10 × 1000 | 16 × 2000（共 128,000 transition） |
| 每轮训练 | 1500 step | 2000 step（共 32,000 step；×512 ≈16.4M 样本） |
| MLP hidden | `[2048,2048,1024,1024,512,512]` | `[3072,3072,2048,1024,512,512]` |
| 激活 / LayerNorm / Dropout | SiLU / 无 / 无 | LayerNorm 开启；Dropout 0.0（过拟合时再试 0.05） |
| batch | 256 | 512 |
| 学习率 | 固定 `1e-4` | peak `1e-4`，warmup 2000 step，cosine 衰减到 `1e-6` |
| weight decay / gradient clip | `1e-4` / `1.0` | `1e-4` / `1.0` |
| beta | 见上 | `[1.0,1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0,0.0]` |
| 评估 / checkpoint | 每轮 student-only 192 步；`round_XX.pt` | 同左 |

`mlp-intensive` 时间按 `mlp-long` 每轮约 100–120 s、rollout 翻倍估算约 55–70 分钟（非硬保证）；首两个 shard 明显更慢时优先减少
round 数，不降低 eval horizon。

跌倒 motion 加权：每轮评估得到每个 motion 的 `teacher_action_rmse`、`terminations`、`termination_rate = terminations / steps`，
下一轮按 `weight(motion) = 1 + 3 × termination_rate(motion) / max_termination_rate` 加权训练样本采样（全不摔为 1，最多 4×；不增加
env 数）。第 0 轮均匀，第 1 轮起用上一轮评估；权重和每轮样本数写入 `metrics.partial.json`。额外 rollout 只给跌倒 motion 属于更重的
motion-balanced 环境分配方案，当前未采用。

实时监督：

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

# 另开终端看日志；训练每 100 step 打印 [dagger] train round=... step=.../2000 loss=... lr=... grad=...
RUN=$(ls -dt ~/bly/runs/dagger_mlp_intensive_* | head -1)
tail -f "$(ls -dt ~/bly/runs/dagger_mlp_intensive_command_*.log | head -1)"
```

逐轮摘要与定时刷新：

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

watch -n 10 "python -c \"import json,pathlib; p=sorted(pathlib.Path.home().joinpath('bly/runs').glob('dagger_mlp_intensive_*'))[-1]/'manifests'/'dagger_mlp_metrics.partial.json'; print(len(json.loads(p.read_text())))\""
```

重点看 `loss` 是否下降、`lr` 是否按 warmup/cosine 变化、`grad` 是否爆炸、是否出现 `non-finite`、每轮 `eval_rmse`、逐 motion
`termination_rate`、`survival_ratio`、`action_saturation_ratio`。判断：loss 降且 eval RMSE 降 → 正常；loss 降而 eval RMSE 不降 →
过拟合/覆盖不足/观测-标签不匹配，不要只加步数；单个 motion 差 → 先查加权是否生效、终止率与饱和；RMSE 低但 termination 高 →
需更多 DAgger 状态覆盖；全部高且 loss 不降 → 查学习率、归一化、输入维度、teacher 标签。

边界：只训练评估同一批 4 个 motion；`memorization_pass` 是动作拟合门禁，不是 192 步生存门禁；192 步 survival 只有 4 env 的粗粒度值；
`mlp-intensive` 尚未在 Ubuntu 实际跑完。

### MLP policy 视频与 SONIC 对比

`mlp_render`：1 env、1 个训练 motion，加载 MLP checkpoint（如 `checkpoints/round_09.pt`），在 Isaac 中闭环执行 student，记录
`dof_pos`、`root_pos_w`、`root_quat_w` 到 `data/000000.trajectory.pkl`，再用 `sonic-repro-kit/render_mujoco_trajectory.py` 生成
`videos/mlp_policy.mp4`。视频是 Isaac policy rollout 的 MuJoCo 运动学渲染，不是 CVAE replay65，也不是 MuJoCo torque 物理重放；
policy 终止时只记录到终止前的帧。

```bash
cd ~/bly/dagger
export SONIC_REPO_ROOT=~/bly/sonic-repro/GR00T-WholeBodyControl
export SONIC_TEACHER_CHECKPOINT=$SONIC_REPO_ROOT/sonic_release/last.pt
export SONIC_RUNS_ROOT=~/bly/runs
export DAGGER_MOTION_FILE=/home/helloworld/bly/runs/bones_seed_ingest_2048_20260825_132447/data/robot_filtered
export PYTHON=~/bly/sonic-repro/.venv-sonic/bin/python

CKPT=/home/helloworld/bly/runs/dagger_mlp_long_20261026_014151/checkpoints/round_09.pt
bash scripts/render_mlp_policy.sh "$CKPT" 192
# 第二个参数 motion key 可省略（默认训练 manifest 的第一个 motion_keys_loaded）；每个 motion 单独渲染
bash scripts/render_mlp_policy.sh "$CKPT" fishing_standing_start_R_001__A502 192

# mlp-intensive checkpoint 必须用匹配配置，否则 hidden_dims/layer_norm 校验拒绝加载；旧 mlp-long 继续用 mlp_render.yaml
export DAGGER_RENDER_CONFIG=~/bly/dagger/configs/mlp_render_intensive.yaml
bash scripts/render_mlp_policy.sh "$CKPT" "$MOTION_KEY" 192
```

`render_sonic_policy.sh` 用冻结 release teacher 在同一 motion 上闭环执行；`render_policy_compare.sh` 先渲染 SONIC 再渲染 MLP，
用 ffmpeg 生成**左 SONIC、右 MLP** 的并排视频。单个与四个 motion 全部渲染：

```bash
bash scripts/render_policy_compare.sh "$CKPT" <motion_key> 192

RUN=/home/helloworld/bly/runs/dagger_mlp_long_20261026_014151
CKPT="$RUN/checkpoints/round_09.pt"
KEYS=$("$PYTHON" -c "import json; d=json.load(open('$RUN/manifests/dagger_mlp_manifest.json')); print('\n'.join(d['motion_keys_loaded']))")

while read -r KEY; do
  bash scripts/render_policy_compare.sh "$CKPT" "$KEY" 192
done <<< "$KEYS"
```

每个对比 run 输出到 `~/bly/runs/dagger_mlp_render_<timestamp>/`：`videos/mlp_policy.mp4`、`compare/sonic_left_mlp_right.mp4`、
`compare/compare_manifest.json`、`markers/render.ok / render_sonic.ok / compare.ok`。两路是独立 Isaac rollout 的 MuJoCo 运动学并排，
不是同状态扭矩对比，也不是闭环物理一致性证明。

## CVAE DAgger v1 与 v2

### Legacy DAgger v1（已归档，不再作为主计划）

单步 receding DAgger：冻结 SONIC teacher，KL2 Stage-C CVAE 作 student；4 motion / 4 env，每轮 rollout 1000 步；每 env 最多 32 步
真实 history，当前 State 在 anchor 32，future State/Action hidden；输出 64 Action 只执行并监督 `Action[32]`；per-step beta；fall
后标准 reset（post-fall continuation 属于后续配置，8h 已完成 run 未使用）；source mix 0.5（source 中约 25% causal mask，其余
physical/full reconstruction）；18 轮 × 4000 optimizer steps；评估 192 步 student-only、`fixed_standard_normal`、标准 reset。

8h 代表结果：overall ratio = 0.55424，4/4 per-motion ratio <= 0.7，overall 0.5 门禁未通过（详见 process.md §24）。

优点：工程简单，与 anchor Action 训练一致；标签来自真实 student state；source mix 保留长序列重建。问题：64 个 Action 只监督 1 个；
T64 训练与 192 步部署间无多步序列监督；C 自己的 future state 未多步训练；每步重推理导致动作抖；窗口间 latent 不连续；fall 通常
reset，恢复数据少；视频里“一次性 64 Action 执行前 32”不是训练时的 anchor 路径，那 32 个初始 Action 未经 DAgger 逐步监督。

旧配置和入口保留；旧 run、旧 checkpoint、旧数据只读，不得为了 v2 修改后继续 resume：

```text
dagger/configs/cvae_kl2_pilot_intensive.yaml
dagger/configs/cvae_kl2_pilot_intensive_8h.yaml
dagger/configs/cvae_kl2_formal.yaml

bash scripts/run_dagger_auto.sh cvae-kl2-pilot-intensive-8h
bash scripts/run_dagger_auto.sh cvae-kl2-formal
```

### DAgger v2（episode mix + teacher prefix）

目标：让 teacher 在 C 真正会走到的状态上纠正 C，并一次执行、逐步监督 16 个连续 Action，而不是每次只看 anchor 32。

- Episode mix：4 env 时 1 teacher / 3 student（25%/75%），teacher env 每轮轮换；8 env formal 为 2 teacher / 6 student。
- Teacher episode：teacher 执行完整 192 步，C 在同一真实 state 上 shadow inference（不执行），每步记录 teacher Action 与 C 预测；
  用于稳定状态覆盖与对比，不能替代 student episode。
- Student episode：teacher 先执行 32 步填满 history，之后 C 每 16 步推理一次（输出 64 Action、执行前 16），每执行一步 teacher
  在真实 student state 上给纠正 Action；16 个执行步各形成一个 anchor 训练窗口（逐步 anchor 监督，还不是单窗口一次输出 16 Action 的
  block loss）；16 步后窗口整体前进；192 步结束或 done 后新 episode 重新 teacher prefix。
- 第一版 `teacher_prefix_steps = 32`、`student_block_steps = 16`；稳定后扩到 `student_block_steps = 32`。
- 初始动作：不再使用未训练的一次性 64 Action 初始块；第一个 C Action 来自 teacher prefix a0..a31 → 32 步真实 history → 首次
  block inference，与后续 block 同一 anchor/history 合同。视频里的 source 64 bootstrap 只作独立诊断，不参与 v2 训练。
- 每个执行步记录 executed Action、teacher Action、真实 State、done、motion id、teacher/student 来源标记。
- Source mix 从 0.5 降到约 0.4；source causal fraction 0.25；固定 source mask；KL 0.002，warmup 10000。
- 结束条件：第一版标准 episode reset；若启用 post-fall continuation，按 reference-relative error（前 10 步 baseline、
  `error_anchor_pos`/`error_body_pos` 超过 3×、连续 50 步）结束该 env，不用绝对高度或竖直姿态。
- 评估同 v1：标准 reset、192 步、`fixed_standard_normal`、student-only、per-motion termination/survival、teacher Action RMSE；
  建议另报 8/16/32/64/192 分段误差、多 latent sample、动作变化率、block 边界连续性。

v2 入口与配置：

```bash
bash scripts/run_dagger_auto.sh cvae-kl2-pilot-dagger-v2         # dagger/configs/cvae_kl2_pilot_dagger_v2.yaml
bash scripts/run_dagger_auto.sh cvae-kl2-pilot-dagger-v2-2m      # cvae_kl2_pilot_dagger_v2_2m.yaml，2-motion 约 3h
bash scripts/run_dagger_auto.sh cvae-kl2-pilot-dagger-v2-2m-2h   # cvae_kl2_pilot_dagger_v2_2m_2h.yaml，2-motion 约 2h
```

- `cvae_kl2_pilot_dagger_v2.yaml` 关键值：`episode_mix: true`、`teacher_prefix_steps: 32`、`student_block_steps: 16`、
  `teacher_episode_period: 4`、`post_fall_continuation: false`、`source_mix: 0.4`。
- `cvae_kl2_pilot_dagger_v2_2m.yaml`：2 env、12×2000、`teacher_episode_period=2`（每轮 1 teacher / 1 student）、prefix32/block16；
  单轮明显超过约 15 分钟时把 `rounds` 降到 9–10，明显更快可增加轮数。
- `cvae_kl2_pilot_dagger_v2_2m_2h.yaml`：2 env、8×2000 = 16k optimizer steps、`teacher_episode_period=2`、prefix32/block16、
  source mix 0.4。
- 后续方向（按优先级）：block16 稳定后扩到 block32；跨 block latent continuity；reference / expected-future-state condition；
  recovery / near-fall 数据；校准 standard-normal latent；最后再讨论更大模型。

| 项目 | v1 Legacy | v2 |
|---|---|---|
| 每步执行 | 1 Action | block16 内逐步执行 |
| teacher 标签 | 只监督 anchor32 | block16 内逐步监督 |
| episode 模式 | per-step beta | 25% teacher / 75% student |
| 初始动作 | 空 history anchor | teacher prefix 32 |
| 初始 64 block | 视频专用 | 不用于训练 |
| source mix | 0.5 | 0.4 |
| fall | 标准 reset | 第一版 reset；后续可 reference-relative continuation |
| 评估 | 192 standard-normal | 同左 + 分段/多 sample |
| 主要风险 | 长时漂移、动作抖 | teacher prefix 成本、block 误差累积 |

## 当前状态

| 项目 | 状态 |
|---|---|
| 独立 dagger 工程 / SONIC frozen teacher | 已实现；teacher Ubuntu smoke 已通过 |
| plain MLP DAgger | 已实现，per-step mixture；4-motion smoke 通过（execution） |
| plain MLP 4-motion pilot | 5/6 轮 partial 通过 memorization；round 5 被 native SIGSEGV 阻塞 |
| plain MLP 4-motion long | `dagger_mlp_long_20261026_014151` 10/10 轮，`memorization_pass=true`；final survival=0.25，survival 尚无通过结论 |
| plain MLP 分片 / auto | 2-round sharding/resume（中间 shard 只写 partial）与 `run_dagger_auto.sh` 自动 resume/重试已实现 |
| `mlp-intensive` 16-round 1h 方案 | 配置/代码、跌倒加权（最多 4×）、cosine warmup / LayerNorm / 更大 MLP 已实现；尚未 Ubuntu 执行 |
| MLP / SONIC 视频 | `mlp_render` + `render_mlp_policy.sh`；`render_sonic_policy.sh` + `render_policy_compare.sh`（左 SONIC 右 MLP） |
| CVAE rolling window / 动态 shard / Action mapping | 已实现并通过 Windows 单元测试、shape 测试、CPU round-trip 测试 |
| 通用 CVAE DAgger smoke / pilot | 待 Ubuntu 执行；pilot 在 smoke 成功后 |
| CVAE KL2 32-motion 适配 | 已实现；Windows 52 tests 通过；manifest 32 keys / 4×8 groups / pilot 4 keys 固定；`source_mix_exact=true` |
| CVAE KL2 smoke | Ubuntu execution PASS；quality 为单轮 smoke 预期 false |
| CVAE KL2 pilot | round 0 step 800 native SIGSEGV；source cache + 500-step chunk resume + auto retry 已加，待重跑 |
| CVAE KL2 formal | 待 pilot 通过 |
| CVAE 4m intensive / 8h | 配置、early-stop 门禁、`data/source_cache.pt` 已实现；8h run 18/18 完成、`quality_pass=false`（见 process.md §24） |
| Stepwise prepare / recursive causal / three-way ghost | `dagger.stepwise_prepare`、`dagger.stepwise_recursive_causal`、`sonic-repro-kit/render_three_way_ghost.py` 已实现；Ubuntu 问题→修复见 process.md §24 |
| CVAE KL2 192-step survival | 实时 Isaac 闭环诊断；待 Ubuntu |
| standard-normal 质量 | 待 pilot 评估 |
| teacher action information contract | 待多 motion/reference probe |

## sonic-10history-reference-cvae-v1：10-step SONIC 序列化 CVAE（新增）

该路线与旧 65-token/T64 CVAE、旧 KL2 DAgger 只读并行存在，不迁移旧 checkpoint。新模型合同为：

```text
state_history   [B,10,70]
action_history  [B,10,29]
reference       [B,10,64]  # joint_pos29 + joint_vel29 + root_orientation6
output          [B,29]     # canonical Action，运行时转 raw SONIC Action
```

逻辑输入序列为 `H[-9..0] + R[+0..9] + Q`，输出一个当前 Action；不再输出 64 个未来 Action 或未来 State。
Physics v5 reference-aware dataset 是唯一新数据源，C 阶段只 mask 当前目标 Action，训练 posterior Action、standard-normal Action 和 KL。

新增入口：

```bash
python -m cvae_sa.sonic_causal_train --config configs/sonic_causal_c_smoke.json
bash scripts/run_dagger.sh sonic-causal-smoke
bash scripts/run_dagger.sh sonic-causal-pilot
```

DAgger 新路线是单步 teacher/student mixture；评估范围是 same-motion teacher imitation，不能解释为 new-motion 泛化。
