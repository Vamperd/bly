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

## 测试

项目不创建新环境，也不主动安装 pytest。使用现有 Ubuntu 环境运行内置 unittest：

```bash
cd /home/helloworld/bly/state-action-cvae
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
```

测试覆盖 runtime Action 映射、per-environment 参数、跨 motion 划分、padding、三类 Mask、
Action/previous-action 防泄漏、通用 ActionMaskScenario、Transformer/TCN shape 和微型
数据过拟合。
