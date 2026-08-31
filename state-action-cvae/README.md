# SONIC State–Action CVAE

这是独立于 SONIC/Isaac Lab 的只读训练项目。它既保留旧版
`sonic_minimal_sa.hdf5` 入口，也支持从两个 2048-motion Physics State–Action v3
采集 run 构建结构化 v4 索引，训练联合建模 State/Action 的 Physics Transformer
CVAE。源码树不保存数据、checkpoint 或日志；全部运行产物写入
`/home/helloworld/bly/runs/<run_id>/`。

## 数据合同

Indexer 只接受 `sonic_minimal_sa_v2`，只读取 `attempt_id=0` canonical episode，
但同时保留 completed 与 failed episode。跨 run 身份使用 `motion_key`，从不使用
可能重复的 `global_motion_id`。生产门禁要求精确得到 768 个 motion key、5120 个
canonical episode，其中 256 个动作有 4 个 variant，512 个动作有 8 个 variant。

模型使用以下输入：64 维 physical state、29 维 previous action、29 维 current
action 与 29 维静态 action scale。current/previous raw action 都按各 episode 的 runtime
schema 转换：

```text
action_rel = clip(raw_action * scale + offset) - default_joint_pos
```

offset、default 和 clip 只参与确定性转换，不进入网络。joint position、velocity、
base angular velocity、previous action 与 action 的统计量只从 train split 计算；gravity
保持单位向量，不做标准化。

每个窗口包含 128 个 transition、129 个 state，控制时长为 2.56 秒。训练随机裁剪，
validation/test 使用 stride 64；短 episode 右侧 padding，attention 与 loss 都忽略 padding。
同一 motion 的所有 variant 与窗口只能进入一个 split。

## Ubuntu 运行前检查

Windows 是唯一代码修改端。同步提交后，在 Ubuntu 只执行拉取和运行：

```bash
cd /home/helloworld/bly
git status --short --branch
git rev-parse HEAD
git -C sonic-repro/GR00T-WholeBodyControl status --short --branch
git -C sonic-repro/GR00T-WholeBodyControl rev-parse HEAD
git -C sonic-repro/IsaacLab status --short --branch
git -C sonic-repro/IsaacLab rev-parse HEAD
nvidia-smi

cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate
python -c 'import h5py, numpy, torch; print(h5py.__version__, numpy.__version__, torch.__version__)'
```

脚本不会安装、升级或降级任何依赖，也不会修改四个 collection run。

## 构建正式索引

默认输入已经固定为以下四个目录，无需再传路径：

```text
/home/helloworld/bly/runs/collect_state_action_20260823_210152
/home/helloworld/bly/runs/collect_state_action_20260823_235356
/home/helloworld/bly/runs/collect_state_action_20260823_224638
/home/helloworld/bly/runs/collect_state_action_20260824_005108
```

运行：

```bash
cd /home/helloworld/bly/state-action-cvae
bash ./cvae_repro.sh build-index

export CVAE_DATASET_RUN="$(cat /home/helloworld/bly/runs/latest_cvae_dataset_run_dir.txt)"
test -f "$CVAE_DATASET_RUN/markers/cvae_dataset.ok"
cat "$CVAE_DATASET_RUN/manifests/dataset_manifest.json"
```

Indexer 会完整 SHA256 四个 HDF5，因此总时长取决于磁盘顺序读取速度。成功目录包含
`episodes.jsonl`、split motion keys、source hashes、normalization、Action 特征审计和
`cvae_dataset.ok`。源 HDF5 只读打开，不复制。

## Smoke 与两种基线训练

先运行 200 optimizer step 的小模型 smoke：

```bash
CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
CVAE_MODEL_KIND=transformer \
CVAE_SEED=20260824 \
bash ./cvae_repro.sh smoke-train
```

正式训练 Transformer 和 TCN 时分别创建新 run，禁止复用或覆盖目录：

```bash
CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
CVAE_MODEL_KIND=transformer \
CVAE_SEED=20260824 \
bash ./cvae_repro.sh train

CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
CVAE_MODEL_KIND=tcn \
CVAE_SEED=20260824 \
bash ./cvae_repro.sh train
```

先比较两个 train run 的 validation `selection_score`，选定结构后再训练 Transformer 的
三个正式种子：

```bash
for seed in 20260824 20260825 20260826; do
  CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
  CVAE_MODEL_KIND=transformer \
  CVAE_SEED="$seed" \
  bash ./cvae_repro.sh train
done
```

默认优化配置为 BF16、micro batch 16、梯度累积 4、AdamW、100000 step、每 1000
step validation、12 次无改善 early stop。若显存门禁在实际 4090 上失败，只允许通过新
JSON override 降低 micro batch 并等比例增加梯度累积，不修改原默认配置或依赖。

## Physics State–Action v3 索引与训练

v3 不与旧数据混合。它直接读取每条 episode 的唯一连续序列
`states[T+1,70]` 和 `actions[T,29]`，并补充窗口开始前的 Action。新 token 顺序为
`A_before,S0,A0,S1,A1,...,S_T`；不构造重复的 `state_tp1`，也不在 State 中保存
`previous_action`。两个源 run 必须分别是同一2048动作集合的 startup variants 0–3
与 initial_state_mild variants 4–7，共16384条 canonical episode。

```bash
export RUN_STARTUP=/home/helloworld/bly/runs/collect_physics_state_action_20260825_133150
export RUN_MILD=/home/helloworld/bly/runs/collect_physics_state_action_20260825_172734

CVAE_SOURCE_RUNS="$RUN_STARTUP:$RUN_MILD" \
CVAE_EXPECTED_MOTIONS=2048 \
CVAE_EXPECTED_EPISODES=16384 \
CVAE_SPLIT_COUNTS=1638,205,205 \
CVAE_SEED=20260830 \
bash ./cvae_repro.sh build-physics-index

export CVAE_DATASET_RUN="$(cat /home/helloworld/bly/runs/latest_cvae_physics_dataset_run_dir.txt)"
test -f "$CVAE_DATASET_RUN/markers/cvae_physics_dataset.ok"
```

先运行结构相同的小模型 smoke：

```bash
CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
CVAE_CONFIG=configs/physics_v3_smoke.json \
CVAE_MODEL_KIND=physics_transformer \
CVAE_CONTEXT_MODE=hidden \
CVAE_SEED=20260830 \
bash ./cvae_repro.sh smoke-train
```

`physics_transformer` 使用 `[29,11]` joint robot information、29个 actuator type、
9维全局仿真信息和 `action_before_window[29]`。Action 已是 processed canonical target，
因此 scale/offset/clip 只服务重放，不进入模型。默认 `CVAE_CONTEXT_MODE=hidden` 隐藏
mass/inertia/COM/material；`oracle` 对照才显式读取648维 dynamics context。

```bash
CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
CVAE_CONFIG=configs/physics_v3.json \
CVAE_MODEL_KIND=physics_transformer \
CVAE_CONTEXT_MODE=hidden \
CVAE_SEED=20260830 \
bash ./cvae_repro.sh train

CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
CVAE_CONFIG=configs/physics_v3.json \
CVAE_MODEL_KIND=physics_transformer \
CVAE_CONTEXT_MODE=oracle \
CVAE_SEED=20260830 \
bash ./cvae_repro.sh train
```

### Action-focused fine-tune

在已训练的 PhysicsTransformer `best.pt` 上只热启动模型权重；optimizer、scheduler、
AMP scaler 与随机状态全部重新初始化。smoke 和正式训练分别写入独立 run，不覆盖 parent：

```bash
export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_physics_dataset_20260825_235244
export CVAE_INIT_CHECKPOINT=/home/helloworld/bly/runs/cvae_train_20260826_002252/checkpoints/best.pt

CVAE_CONFIG=configs/physics_v3_action_finetune_smoke.json \
CVAE_MODEL_KIND=physics_transformer CVAE_CONTEXT_MODE=hidden CVAE_SEED=20260831 \
bash ./cvae_repro.sh smoke-action-finetune

CVAE_CONFIG=configs/physics_v3_action_finetune.json \
CVAE_MODEL_KIND=physics_transformer CVAE_CONTEXT_MODE=hidden CVAE_SEED=20260831 \
bash ./cvae_repro.sh action-finetune
```

正式训练为40k local optimizer step。Mask curriculum依次把 Action 区间扩到
32/64/128步；每次validation同时比较parent固定基线，只有四项State指标均不超过
parent的105%才可写入`best.pt`。未通过State guard的最低Action分数仍保存在
`best_unguarded.pt`供诊断，但不会产生成功marker。

### 32-motion memorization benchmark

该实验只验证训练集记忆能力，不是 validation/test 泛化评测。子集从正式 Physics v4
索引的原 train split 中选择八个 package 各4个 motion，且每个 motion 的8个 variant
必须全部 completed；源 HDF5 只读引用，子集重新计算 normalization：

```bash
export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_physics_dataset_20260825_235244
CVAE_SEED=20260828 bash ./cvae_repro.sh build-overfit-subset
export CVAE_DATASET_RUN="$(cat /home/helloworld/bly/runs/latest_cvae_overfit_subset_run_dir.txt)"

# 先验证数据、模型、门禁和SVG链路，不代表模型质量通过。
CVAE_OVERFIT_SMOKE=true CVAE_SEED=20260828 bash ./cvae_repro.sh overfit-capacity
```

紧凑模型为15,065,048参数。容量阶段训练与成功门禁都使用posterior mean并关闭
KL、auxiliary、cycle；同一批次和Mask上的prior mean仅作为非门禁诊断，不能影响
`overfit_score`、checkpoint选择或成功marker。完整阶段从容量阶段best权重热启动，
重新初始化optimizer、scheduler和RNG，并恢复以deterministic prior mean作为成功门禁：

```bash
CVAE_OVERFIT_MODEL=compact CVAE_SEED=20260828 bash ./cvae_repro.sh overfit-capacity
export CAPACITY_RUN="$(cat /home/helloworld/bly/runs/latest_overfit_capacity_compact_run_dir.txt)"
CVAE_OVERFIT_MODEL=compact CVAE_SEED=20260828 \
CVAE_INIT_CHECKPOINT="$CAPACITY_RUN/checkpoints/best.pt" \
bash ./cvae_repro.sh overfit-full

# 2847万参数原模型对照；capacity/full都使用reference配置。
CVAE_OVERFIT_MODEL=reference CVAE_SEED=20260828 bash ./cvae_repro.sh overfit-capacity
```

紧凑模型seed 20260828两阶段通过后，再运行20260829与20260830。`joint_id_only`
需要分别运行capacity/full；`no_aux`只运行full，并从紧凑capacity checkpoint加载。
每个run的`videos/training_curves.svg`在validation后原子刷新；capacity额外生成
`videos/latent_mode_comparison.svg`，对比posterior gate与不参与门禁的prior diagnostic；
full run还生成`videos/input_sensitivity.svg`。最终把全部run用冒号传给汇总入口：

```bash
CVAE_OVERFIT_RUNS="/run/a:/run/b:/run/c" bash ./cvae_repro.sh summarize-overfit
```

只有紧凑模型具备三个完整seed pair、其中至少两个capacity/full均通过，并存在
seed 20260828的reference pair，才生成`cvae_overfit_suite.ok`。所有报告都明确禁止
泛化声明。汇总会拒绝缺少`gate_latent_mode`/`diagnostic_latent_modes`的新旧协议混合
run，因此旧的prior-gated capacity失败run只能保留作诊断，不能进入suite。

当前 compact/reference 联合实验不能再称为“严格记忆容量测试”。严格测试使用固定
窗口、固定 Mask、固定任务以及每任务相同的 `samples_seen_per_task`，五个任务分别运行：

```bash
export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
for task in forward_rollout inverse history_action arbitrary_state arbitrary_action; do
  CVAE_OVERFIT_MODEL=compact CVAE_OVERFIT_TASK="$task" CVAE_SEED=20260828 \
    bash ./cvae_repro.sh overfit-single-task
done

# 可选地使用失败 capacity 的 best.pt 同时计算梯度余弦和输入遮挡；不提供 checkpoint
# 时仍生成唯一 transition 近邻分散度和 RobotInfo 实际方差。
CVAE_ANALYSIS_CHECKPOINT=/run/compact-capacity/checkpoints/best.pt \
  bash ./cvae_repro.sh analyze-overfit

CVAE_OVERFIT_RUNS="/run/task-a:/run/task-b:/run/task-c:/run/task-d:/run/task-e" \
  bash ./cvae_repro.sh summarize-single-tasks
```

无 reference 的 deterministic inverse/history 只记录诊断值，不参与 compact 单任务
suite 成功判定。`identifiability_heatmap.svg` 中的近邻目标分散度是排除自身及同 episode
邻近帧后的经验歧义指标，不是数学误差下界；`input_sensitivity.svg` 只解释该训练集上的
遮挡敏感性。

单任务训练结束后，使用只读 exact-fixture 入口比较 `best.pt`、`last.pt` 在训练时原始
窗口/Mask 上的记忆指标与现有 unseen-Mask 门禁。该入口不会改变 checkpoint、训练门禁或
历史 marker；它的 `.ok` 只表示诊断执行和数据一致性通过：

```bash
export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
export CVAE_OVERFIT_RUNS="/run/forward:/run/inverse:/run/history:/run/state:/run/action"
export CVAE_FIXTURE_CHECKPOINTS=best,last
bash ./cvae_repro.sh diagnose-overfit-fixture

FIXTURE_RUN="$(
  cat /home/helloworld/bly/runs/latest_cvae_overfit_fixture_diagnostic_run_dir.txt
)"
cat "$FIXTURE_RUN/manifests/fixture_diagnostic.json"
cat "$FIXTURE_RUN/manifests/fixture_report_zh.md"
ls -lh "$FIXTURE_RUN/videos"
```

入口会对全部固定窗口重建 `episode_ref + window_start + fixed_mask_seed` 对应的训练 Mask，
并验证 step 1 与 checkpoint step 的 fixture hash 一致。质量结论保存在
`quality_summary`；即使 exact 指标失败，只要只读诊断完整，执行 marker 仍为 PASS。

## 最简 posterior capacity

`physics_posterior_transformer` 是独立的 posterior-only 记忆实验：只使用70维 State、29维
Action、逐特征 Mask 和位置/类型；共享双向 encoder 输出单个256维 global latent，单个双向
decoder 负责全部 reconstruction。它不包含 RobotInfo、reference 或 relation/rollout head，
`KL beta=0`，因此通过只证明 posterior capacity，不证明部署时条件推理。

```bash
export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
CVAE_POSTERIOR_MOTIONS=1 CVAE_POSTERIOR_WINDOW=8 \
  bash ./cvae_repro.sh posterior-capacity-smoke

CVAE_POSTERIOR_MOTIONS=1 CVAE_POSTERIOR_WINDOW=16 \
  bash ./cvae_repro.sh posterior-capacity

# 单窗口诊断：仍校验1个motion的8个variant，但只训练/验收索引中的第一个固定窗口
CVAE_POSTERIOR_MAX_WINDOWS=1 \
CVAE_POSTERIOR_MOTIONS=1 CVAE_POSTERIOR_WINDOW=16 \
  bash ./cvae_repro.sh posterior-capacity

CVAE_POSTERIOR_PHASE=generalization \
CVAE_INIT_CHECKPOINT=/run/fixed/checkpoints/best_exact.pt \
CVAE_POSTERIOR_MOTIONS=32 CVAE_POSTERIOR_WINDOW=16 \
  bash ./cvae_repro.sh posterior-capacity
```

固定阶段使用10类 deterministic Mask bank，并按 exact score 保存 `best_exact.pt`；只有全部窗口
连续3次满足 State/Action RMSE、max error、contact 和 zero-latent 门禁才写
`cvae_posterior_capacity.ok`。设置 `CVAE_POSTERIOR_MAX_WINDOWS=N` 会按既有固定索引顺序只取
前N个窗口，并在 summary 的 `selected_windows` 中记录身份。泛化阶段必须保持 checkpoint 的
motion 数、窗口长度与窗口子集一致，在每个已见窗口16个 held-out Mask 上通过后才写
`cvae_posterior_mask_generalization.ok`。

fixed 阶段的训练和 exact validation 必须复用同一 Mask seed，保证 element/time/feature/semantic
的具体坐标完全相同；summary 会记录 `training_mask_seed`、`validation_mask_seed` 和
`fixed_fixture_identity_match`。只有 generalization 阶段使用独立 validation seed。

`LeanSplit v1` 将确定性因果动力学、reference-conditioned Action 和双向 CVAE completion
拆开，生产配置为 6,204,665 参数。forward 只读取最近
`H=max(10, observed_max_delay+1)` 的 State/已知 Action，不读取 reference 或 CVAE latent；
Action 分支可读取采集时真实可见的 `10×64` runtime reference；CVAE latent 只服务任意
缺失补全。State 70维、Action 29维及 Isaac replay 合同保持不变。reference-aware 数据
必须是 Physics v5，manifest 会把输入标成 configured、measured、causally_estimated 或
oracle_only；deployment-only 模式拒绝 oracle context 和来源不明的 reference。

Physics v5 子集建立后，LeanSplit 固定单任务用法如下：

```bash
export CVAE_SOURCE_RUNS=/run/v5-startup:/run/v5-initial-state-mild
CVAE_EXPECTED_MOTIONS=32 CVAE_EXPECTED_EPISODES=256 \
CVAE_SPLIT_COUNTS=32,0,0 CVAE_SEED=20260828 \
  bash ./cvae_repro.sh build-physics-index

export CVAE_DATASET_RUN="$(
  cat /home/helloworld/bly/runs/latest_cvae_physics_dataset_run_dir.txt
)"
CVAE_SEED=20260828 bash ./cvae_repro.sh build-overfit-subset

export CVAE_DATASET_RUN="$(
  cat /home/helloworld/bly/runs/latest_cvae_overfit_subset_run_dir.txt
)"
CVAE_OVERFIT_MODEL=lean CVAE_OVERFIT_TASK=forward_rollout CVAE_SEED=20260828 \
  bash ./cvae_repro.sh overfit-single-task

# Action 信息增量实验通过 CVAE_CONFIG 依次选择：
# overfit_32_lean_action_history_only.json
# overfit_32_lean_action_history_queue.json
# overfit_32_lean_action_history_reference.json
# overfit_32_lean_action_history_reference_dynamics.json
CVAE_OVERFIT_MODEL=lean CVAE_OVERFIT_TASK=history_action \
CVAE_CONFIG=configs/overfit_32_lean_action_history_reference.json \
  bash ./cvae_repro.sh overfit-single-task
```

复合 Mask 以40% forward、35% Action inference、25% arbitrary S/A completion 采样。
模型分别报告一步/8步 forward、inverse Action、history-conditioned Action 和任意联合
补全；torque/impulse 只从 `(S_t,A_t)` 转移表示进行辅助监督，不能读取 `S_{t+1}`。

## Test 评测

测试集只在基于 validation 完成模型选择后运行一次：

```bash
export CVAE_CHECKPOINT=/home/helloworld/bly/runs/<selected_train_run>/checkpoints/best.pt

CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
CVAE_CHECKPOINT="$CVAE_CHECKPOINT" \
bash ./cvae_repro.sh evaluate
```

旧模型评测保持原指标。`physics_transformer` 分别报告一步/8步 forward、inverse Action、
history Action 与 arbitrary completion；必须同时满足打乱 Action 后 forward RMSE 恶化
至少10%、打乱下一 State 后 inverse RMSE 恶化至少10%，且修改不可见未来 State 不改变
history Action，才生成 `cvae_eval.ok`。

## Mask 补全采样

不提供输入时，默认读取 test split 的第一个窗口并采样 8 个 latent：

```bash
CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
CVAE_CHECKPOINT="$CVAE_CHECKPOINT" \
CVAE_SAMPLE_TASK=completion \
CVAE_SAMPLE_COMPLETION=step \
CVAE_LATENT_SAMPLES=8 \
bash ./cvae_repro.sh sample
```

也可令 `CVAE_SAMPLE_INPUT` 指向 NPZ，或指向已被 dataset index 收录的 HDF5。HDF5
输入还需设置 `CVAE_SAMPLE_EPISODE=demo_x` 和可选 `CVAE_SAMPLE_START`。旧版 NPZ 必须包含
`physical_state [129,64]`、`previous_action [129,29]`、`action_rel [128,29]`、
`action_scale [29]`；可选显式三个布尔 mask、valid mask、progress 和 `normalized`。
Physics v4 NPZ 改为 `physical_state [129,70]`、空的 `previous_action [129,0]`，并必须
附带 `action_before_window [29]`、`joint_robot_information [29,11]`、
`joint_actuator_type [29]`、`global_robot_information [9]`、兼容用
`robot_information [293]` 与 `dynamics_context [648]`；优先使用已索引 HDF5，避免
手工构造 context。
输出 `data/samples.npz` 保存物理单位下的 8 条样本、均值与方差。

## 通用 Action Mask 物理重放与视频

该入口自动识别旧 v1 数据或 `sonic_physics_state_action_cvae_dataset_v4`，只使用
validation（默认 Locomotion）动作，并自动选择完成且不少于 192 步的 motion。源 SONIC
轨迹只采集一次；Element、Step、随机 Feature、五个语义关节组及 Inverse Full 共用同一个
128 步窗口。Physics v4 批次为 `A_before,S0,A0,...,S128`，使用 70 维 State、空的
previous-action、canonical Action 与结构化 RobotInfo。普通补全使用 reconstruction Action
head；`inverse_full_128` 使用专用 inverse head，且是确定性单候选。未 Mask Action 保持
位级等于原序列；8 个随机 latent 只用于普通补全的不确定性，不用于挑选主结果。

先确保 README 中的 SONIC `0005` 外部 Action 补丁已经在 Ubuntu 嵌套仓库应用，再运行：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_physics_dataset_20260825_235244 \
CVAE_CHECKPOINT=/home/helloworld/bly/runs/cvae_train_20260826_002252/checkpoints/best.pt \
CVAE_REPLAY_SPLIT=validation \
CVAE_REPLAY_PACKAGE=Locomotion \
CVAE_REPLAY_MOTION_KEY=auto \
CVAE_MASK_PRESET=all_action_masks_v1 \
CVAE_REPLAY_LATENT_MODE=prior_mean \
CVAE_REPLAY_LATENT_SAMPLES=8 \
CVAE_REPLAY_RENDER=representatives \
CVAE_REPLAY_SEED=20260830 \
bash ./cvae_repro.sh validate-action-mask-replay
```

`CVAE_MASK_SCENARIOS=/绝对路径/custom_scenarios.jsonl` 可替代默认 preset；场景遵循
`sonic_action_mask_scenario_v1`，因此新增 Mask 不修改推理、Isaac replay 或视频后端。
`CVAE_REPLAY_RENDER=all` 为每个场景生成对比视频，`none` 只运行补全、物理重放和
指标。输出位于新的 `cvae_action_mask_eval_时间戳` run，包含 source/original、五个代表
场景、总览 grid、离线补全误差、16-step stride 全动作扫描、物理漂移和稳定性指标。
Physics v4 指标还包含 base 线/角速度、height、contact、RobotInfo/dynamics context 一致性。
工程门禁只要求坐标往返、源重放忠实度和 Mask 前一致；若 CVAE 弱于 hold-last 或线性
插值，run 仍诚实保存并在 summary 中令 `model_quality_pass=false`。

默认 `CVAE_REPLAY_LATENT_MODE=prior_mean` 使用条件 prior 的确定性均值；
`CVAE_REPLAY_LATENT_SAMPLES` 在该模式下只用于不确定性统计，不会把多个 Action 求平均。
若仅用于模型候选覆盖能力的 oracle 诊断，可设置
`CVAE_REPLAY_LATENT_MODE=oracle_best_of_n`：程序生成 N 条完整候选，并用被 Mask 的真实
Action 计算误差，只选整条误差最小的候选进行 Isaac 重放。该模式明确使用标签泄漏，不能
代表部署时可获得的性能；manifest 会保存每条候选误差、选中编号及
`oracle_uses_ground_truth_action=true`。

## State Mask 补全与三联视频

Physics v4 可直接从 validation HDF5 读取70维 State、canonical Action、RobotInfo 和保存的
root轨迹，不需要重新启动SONIC或Isaac。该入口独立于Action replay；主视频依次显示记录
轨迹、真值State积分重建和预测State积分重建：

```bash
CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_physics_dataset_20260825_235244 \
CVAE_CHECKPOINT=/home/helloworld/bly/runs/cvae_train_20260826_002252/checkpoints/best.pt \
CVAE_STATE_SPLIT=validation \
CVAE_STATE_PACKAGE=Locomotion \
CVAE_STATE_MOTION_KEY=auto \
CVAE_STATE_VARIANT=auto \
CVAE_STATE_MASK_PRESET=state_prediction_v1 \
CVAE_STATE_LATENT_MODE=prior_mean \
CVAE_STATE_LATENT_SAMPLES=8 \
CVAE_STATE_RENDER=representatives \
CVAE_STATE_ROOT_MODE=integrate_predicted \
CVAE_STATE_SEED=20260830 \
bash ./cvae_repro.sh validate-state-mask-video
```

1/2/4/8步forward rollout属于训练分布内指标；32步由四段8步预测连续推进并明确标记为
OOD。State-only场景不会Mask或改写Action，Mask外State也保持位级不变。8个latent只用于
completion不确定性，不能利用真值挑选结果。`CVAE_STATE_RENDER=none`仅生成NPZ和指标，
`representatives`使用32步连续State补全替代原8步代表场景，`all`为全部State场景生成三联视频。
输出使用独立的`cvae_state_mask_eval_*`目录和latest
指针，不覆盖任何Action评测产物。

## 测试

项目不创建新环境，也不主动安装 pytest。使用现有 Ubuntu 环境运行内置 unittest：

```bash
cd /home/helloworld/bly/state-action-cvae
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
```

测试覆盖 runtime Action 映射、per-environment 参数、跨 motion 划分、padding、三类 Mask、
Action/previous-action 防泄漏、通用 ActionMaskScenario、Transformer/TCN shape 和微型
数据过拟合，以及State-only Mask、32步分段rollout和root积分重建。
