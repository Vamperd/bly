# SONIC State–Action CVAE

这是独立于 SONIC/Isaac Lab 的只读训练项目。它从四个已通过门禁的
`sonic_minimal_sa.hdf5` 构建索引，训练 Transformer-CVAE 或参数量相近的
TCN-CVAE。源码树不保存数据、checkpoint 或日志；全部运行产物写入
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

## Test 评测

测试集只在基于 validation 完成模型选择后运行一次：

```bash
export CVAE_CHECKPOINT=/home/helloworld/bly/runs/<selected_train_run>/checkpoints/best.pt

CVAE_DATASET_RUN="$CVAE_DATASET_RUN" \
CVAE_CHECKPOINT="$CVAE_CHECKPOINT" \
bash ./cvae_repro.sh evaluate
```

评测输出 masked normalized RMSE、物理单位 MAE/RMSE、gravity 角误差、一步与多步误差、
per-package 指标、latent active dimensions。只有随机打乱 Action 后 forward RMSE 至少
恶化 10% 才生成 `cvae_eval.ok`。

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
输入还需设置 `CVAE_SAMPLE_EPISODE=demo_x` 和可选 `CVAE_SAMPLE_START`。NPZ 必须包含
`physical_state [129,64]`、`previous_action [129,29]`、`action_rel [128,29]`、
`action_scale [29]`；可选显式三个布尔 mask、valid mask、progress 和 `normalized`。
输出 `data/samples.npz` 保存物理单位下的 8 条样本、均值与方差。

## 通用 Action Mask 物理重放与视频

该入口只使用 validation（默认 Locomotion）动作，自动选择完成且不少于 192 步的
motion。源 SONIC 轨迹只采集一次；Element、Step、随机 Feature、五个语义关节组及
Inverse Full 共用同一个 128 步窗口、同一组完整 physical state 和 prior mean。Mask
Action 时会同步隐藏下一 State 中的 previous-action 副本，未 Mask Action 保持逐元素
等于原序列。8 个随机 latent 只报告不确定性，不用于挑选主结果。

先确保 README 中的 SONIC `0005` 外部 Action 补丁已经在 Ubuntu 嵌套仓库应用，再运行：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_dataset_20260824_145739 \
CVAE_CHECKPOINT=/home/helloworld/bly/runs/cvae_train_20260824_150534/checkpoints/best.pt \
CVAE_REPLAY_SPLIT=validation \
CVAE_REPLAY_PACKAGE=Locomotion \
CVAE_REPLAY_MOTION_KEY=auto \
CVAE_MASK_PRESET=all_action_masks_v1 \
CVAE_REPLAY_LATENT_MODE=prior_mean \
CVAE_REPLAY_LATENT_SAMPLES=8 \
CVAE_REPLAY_RENDER=representatives \
CVAE_REPLAY_SEED=20260824 \
bash ./cvae_repro.sh validate-action-mask-replay
```

`CVAE_MASK_SCENARIOS=/绝对路径/custom_scenarios.jsonl` 可替代默认 preset；场景遵循
`sonic_action_mask_scenario_v1`，因此新增 Mask 不修改推理、Isaac replay 或视频后端。
`CVAE_REPLAY_RENDER=all` 为每个场景生成对比视频，`none` 只运行补全、物理重放和
指标。输出位于新的 `cvae_action_mask_eval_时间戳` run，包含 source/original、五个代表
场景、总览 grid、离线补全误差、16-step stride 全动作扫描、物理漂移和稳定性指标。
工程门禁只要求坐标往返、源重放忠实度和 Mask 前一致；若 CVAE 弱于 hold-last 或线性
插值，run 仍诚实保存并在 summary 中令 `model_quality_pass=false`。

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
