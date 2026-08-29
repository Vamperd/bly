# SONIC Physics State–Action 研究：当前代理交接说明

本文档是 `bly` 工作区的首要交接入口。后续会话开始任何工作前必须完整阅读；其中“已验证事实”“代码已实现但待执行”“历史兼容路径”不可混为一谈。若实际 Git、文件或 Ubuntu 日志与本文冲突，以只读检查得到的当前事实为准，并更新本文。

## 1. 当前研究目标与边界

本项目已经越过早期的 SONIC 环境复现与最简 S–G–A 采集阶段。当前主线是用 SONIC released policy 在 Isaac Lab 中产生可信的连续物理轨迹，训练联合建模 State 与 Action 的 PhysicsTransformer CVAE：

\[
p(S_{0:T},A_{0:T-1}\mid RobotInfo)
\]

目标能力包括：

- 前向动力学：由 `S_t,A_t` 预测 `S_{t+1}`，以及短期 autoregressive rollout。
- 逆动力学：由 `S_t,S_{t+1}` 推断产生该转移的 `A_t`。
- history Action：只使用 `S_≤t,A_<t` 预测当前 Action，不读取未来 State。
- 任意联合补全：对 State/Action 的元素、时间块、特征或语义组进行 Mask 后补全。
- Action 物理验证：把补全 Action 转回 raw Action，在 Isaac 中重放并生成 MP4。
- State 视频验证：由真实/预测 70 维 State 积分 root 轨迹，在 MuJoCo 中生成三联 MP4。

当前模型不接收 goal、motion ID、package、outcome 或 reference motion。SONIC policy 本身仍使用其原始 observation/history/future reference 来追踪动作；数据模型学习的是执行结果中的 State–Action 规律，二者不要混淆。

## 2. 强制协作、安全与写入规则

- **Windows 是唯一代码修改端**：`C:\Users\86136\Desktop\code\RL\bly`。Ubuntu 只允许同步、运行、采集、训练、评测和回传日志，不得在那里手工修源码。
- 所有修复先在 Windows 使用 `apply_patch` 完成并检查 Git 差异；嵌套 SONIC 修改通过外层仓库中的 patch 同步，不在 Ubuntu 重复编辑。
- 禁止自动更新或重装 NVIDIA driver、CUDA、Isaac Sim、Isaac Lab、PyTorch 及既有依赖；当前可运行环境优先于 `pip check` 的形式一致性。
- 不得删除嵌套仓库的 `.git`，不得 `git reset --hard`、擅自 checkout/恢复用户改动，且不得删除工作目录外内容；确需删除数据必须先报告绝对路径、大小、原因和可恢复性并取得许可。
- 所有 collection/train/eval/render 输出必须在 `/home/helloworld/bly/runs/<run_id>/`，不得写入 SONIC、IsaacLab 或 CVAE 源码树。
- 源 collection HDF5、数据索引和既有 checkpoint 始终只读；新实验创建新 run，不复用或覆盖历史目录。
- 若根目录出现 `.codegraph/`，理解源码时先使用 CodeGraph；当前检查时不存在，因此使用 `rg` 和只读文件检查。

## 3. 工作区、Git 与固定环境

| 用途 | Windows | Ubuntu |
|---|---|---|
| 外层仓库 | `C:\Users\86136\Desktop\code\RL\bly` | `/home/helloworld/bly` |
| SONIC 嵌套仓库 | `bly/sonic-repro/GR00T-WholeBodyControl` | `~/bly/sonic-repro/GR00T-WholeBodyControl` |
| IsaacLab 嵌套仓库 | `bly/sonic-repro/IsaacLab` | `~/bly/sonic-repro/IsaacLab` |
| 采集/重放工具 | `bly/sonic-repro-kit` | `~/bly/sonic-repro-kit` |
| CVAE 项目 | `bly/state-action-cvae` | `~/bly/state-action-cvae` |
| Python 环境 | Windows 仅做静态/轻量测试 | `~/bly/sonic-repro/.venv-sonic` |
| 全部运行产物 | `bly/runs`（通常不放大数据） | `~/bly/runs` |

截至 2026-08-27 的已观测状态：

- Windows 外层当前分支字面值为 `tiny-model`；Action-focused fine-tune 实现提交为 `197a1730fd829a512e153755f56e6e97d4b1329d`。Ubuntu 同步前仍须读取其实际分支，禁止为了匹配本文自动 reset。
- SONIC 当前分支为 `codex/minimal-state-action-recorder`，已观测 HEAD `fdd6dcf9c0c054ab0fbbde98a5bc8d4326ec00a6`，包含外层 `patches/0001`–`0007` 对应能力；`0008` reference recorder 已在 Windows 外层实现并通过 `git apply --check`，但尚未在 Ubuntu 应用或采集。
- IsaacLab 固定基线为 detached HEAD `37ddf626871758333d6ed89cf64ad702aef127d0`；Windows 有 9 个历史修改/元数据差异，当前任务不得触碰。
- Ubuntu 固定环境为 Ubuntu 24.04.3、RTX 4090 24 GB、driver 595.84、Isaac Sim 5.1、Python 3.11.14、PyTorch 2.7.0+cu128。
- released SONIC 原始基线是 `c374bae5b9039cd0ee71377e654d11ce1bc69e1d`，checkpoint 为 `sonic_release/last.pt`。
- 外层 Git 不会自动携带未提交的嵌套仓库差异；应用 SONIC patch 前必须先检查当前分支/HEAD，已应用的 patch 不得重复应用。

## 4. 当前主数据合同：Physics State–Action v3

每个 episode 只保存一份连续序列：

```text
S0, A0, S1, A1, ..., A(T-1), ST
states  = [T+1, 70]
actions = [T, 29]
```

严格语义是 `states[t] + actions[t] -> states[t+1]`。不再保存重复的 `state_tp1`，State 中也不再重复 `previous_action`；窗口开始前控制历史由 `action_before_window` 提供。

### 4.1 State：70维

| 字段 | 维度 | 物理含义 |
|---|---:|---|
| `joint_pos_canonical` | 29 | `q - nominal_default_joint_pos`，rad |
| `joint_vel` | 29 | 实际关节速度，rad/s |
| `base_lin_vel_robot` | 3 | 骨盆系基座线速度，m/s |
| `base_ang_vel_robot` | 3 | 骨盆系基座角速度，rad/s |
| `gravity_robot` | 3 | 世界向下方向在骨盆系的单位向量 |
| `base_height` | 1 | 骨盆相对平地高度，m |
| `foot_contact` | 2 | 左右脚接触二值标签，阈值 10 N |

该 State 利用平地世界 xy 平移与绝对 yaw 对称性，不把绝对 root xy/heading 输入模型；绝对 `root_pos_w/root_quat_w/body_pos_w` 单独保存用于重放和视频。

### 4.2 Action：29维

主模型 Action 是 ActionManager 处理后的关节目标在统一 nominal 坐标下的表示：

\[
A_t=q^{target}_t-q_{nominal}
\]

它与 State 的 canonical 关节角具有相同 29 关节顺序、零点和单位。`raw_policy_action`、`processed_joint_target_abs`、scale/offset/clip 仍保存在数据/schema 中用于精确重放，但不作为主模型输入。

### 4.3 RobotInfo 与 context

模型显式接收：

| 输入 | Shape | 内容 |
|---|---:|---|
| `joint_robot_information` | `[29,11]` | nominal、canonical 位置上下限、速度/力矩限制、Kp/Kd、armature、joint friction、delay 上下限 |
| `joint_actuator_type` | `[29]` | runtime actuator 类型词表 ID |
| `global_robot_information` | `[9]` | sim/control dt、decimation、gravity、solver position/velocity iterations、contact threshold |
| `action_before_window` | `[29]` | `A[start-1]`；episode 起点使用 initial processed target |

默认 `context_mode=hidden`，648维 mass/inertia/COM/material dynamics context 不输入模型，作为 CVAE latent 所代表的未观测动力学。`oracle/explicit` 仅用于上限对照。35维平均关节力矩与足端接触冲量只作 auxiliary supervision，不作为 State。

## 5. 已验证数据资产与来源

当前正式数据来自同一个 2048-motion BONES-SEED 子集，不使用已删除且错误的 `512_bones_seed_insight`：

```text
Motion subset:
/home/helloworld/bly/runs/bones_seed_ingest_2048_20260825_132447

Startup variants 0–3:
/home/helloworld/bly/runs/collect_physics_state_action_20260825_133150

Initial-state-mild variants 4–7:
/home/helloworld/bly/runs/collect_physics_state_action_20260825_172734

Physics v4 index:
/home/helloworld/bly/runs/cvae_physics_dataset_20260825_235244
```

已回传并通过的门禁：

| 数据 | 结果 |
|---|---|
| 2048 子集 | 八个 package 各 256，共 2048；本轮不要求与旧 256/512 子集零重叠 |
| startup | 8192 canonical；7986 completed；schema/action roundtrip/coverage 全 PASS |
| initial_state_mild | 8192 canonical；8072 completed；schema/action roundtrip/coverage 全 PASS |
| 合并索引 | 2048 motion、16384 canonical，按 motion key 划分 1638/205/205 |
| 控制时序 | `sim_dt=0.005`、`control_dt=0.02`、decimation=4，即 50 Hz |
| 数据隔离 | split 单位为 `motion_key`；同一动作全部 variant 与窗口只能位于一个 split |

原始 recorder 中间文件 `sonic_physics_sa_raw.hdf5` 不是训练输入，可能已被用户删除以释放磁盘；正式 index 只读取通过 marker 的 `sonic_physics_sa_v3.hdf5`。不得假设 raw 文件仍存在，也不得重新生成或复制大文件，除非用户明确要求。

旧 `sonic_minimal_sa_v2`、旧 768-motion v1 索引、Transformer/TCN checkpoint 仍作为历史兼容基线保留，但不得与 Physics v3/v4 数据混训。任何删除历史 run 的动作仍需用户授权。

## 6. 当前模型与训练实现

`state-action-cvae` 的主模型种类是 `physics_transformer`：

```text
d_model=384
encoder_layers=6
decoder_layers=8
heads=8
ffn_dim=1536
latent_dim=96
joint_width=128
dropout=0.1
```

token 布局为：

```text
A_before, S0, A0, S1, A1, ..., A127, S128
```

模型使用 joint-aware encoder/query decoder、RoPE 时间位置、State/Action 类型 embedding，并提供四条输出路径：任意 masked reconstruction、forward dynamics、inverse Action、history-conditioned Action。Forward head 只能读取 `S_t,A_t`；inverse head 只能读取 `S_t,S_{t+1}`；history head 使用 causal attention，不能读取未来 State。相关防泄漏测试已实现。

### 6.1 32-motion 诊断与 LeanSplit v1：代码已实现但待 Ubuntu 执行

基于 compact/reference 均未通过联合 overfit gate 的结果，Windows 代码新增了固定窗口、
固定 Mask 的五个单任务容量入口、唯一 transition 近邻目标分散度、RobotInfo 方差、共享
主干梯度余弦和输入遮挡分析。无 reference 的 deterministic inverse/history 只作诊断，
不参与 compact 单任务 suite 成功判定。

新增 `physics_lean_split` 为 6,204,665 参数：因果动力学分支使用至少10步历史与已发送
Action，不读取 reference/CVAE latent；Action 分支可读取 runtime command manager 直接记录的
`10×64` reference；双向 CVAE 仅负责 arbitrary completion。State/Action 仍为70/29维。
Physics v5 数据合同、`patches/0008` recorder、四类 Action 信息增量配置均已实现，但本文尚未
收到 Ubuntu smoke、v5 collection 或正式单任务训练日志，不得表述为已经验证成功。
`sonic-repro.sh prepare-overfit-reference-subset` 会从旧 overfit selection manifest 提取同一
32 个 motion，并在新 run 中建立经 hash 校验的只读绝对软链接；不得用另一批 motion 代替。

### 6.2 已完成 parent 训练

```text
Dataset:
/home/helloworld/bly/runs/cvae_physics_dataset_20260825_235244

Parent train run:
/home/helloworld/bly/runs/cvae_train_20260826_002252

Selected checkpoint:
/home/helloworld/bly/runs/cvae_train_20260826_002252/checkpoints/best.pt
```

训练完成 120000 optimizer step，最佳 validation selection score 为 `0.3173407225683331`。测试评测 run 为 `/home/helloworld/bly/runs/cvae_eval_20260827_104021`；该 test split 已在历史实验中使用，因此后续结果必须标注 `reused test`，不得声称全新 blind test。

关键 test 指标：

| 指标 | 结果 |
|---|---:|
| one-step forward normalized RMSE | 0.2634 |
| rollout-8 normalized RMSE | 0.4603 |
| forward joint position RMSE | 0.0179 rad |
| forward joint velocity RMSE | 0.2775 rad/s |
| inverse Action RMSE | 0.1321 rad |
| history Action RMSE | 0.1695 rad |

负对照全部通过：打乱 Action 后 forward RMSE 恶化 28.3%；打乱 `S_{t+1}` 后 inverse RMSE 恶化 152.3%；修改不可见未来 State 对 history Action 的最大影响为 0。模型确实使用了 S–A 关系，但长 Action 补全的物理重放仍不够好，这是当前 fine-tune 的直接动机。

## 7. Action-focused fine-tune：当前正在推进的任务

代码已经在外层提交 `197a173` 中实现，**但本文没有收到 Ubuntu smoke/正式训练完成日志，因此不得写成已训练完成**。新入口：

```bash
bash ./cvae_repro.sh smoke-action-finetune
bash ./cvae_repro.sh action-finetune
```

主要实现：

- 只从 parent 加载 model weights；optimizer、scheduler、AMP scaler 与 RNG 全部重新初始化，并严格校验 checkpoint v2、dataset manifest hash、模型结构及参数 shape。
- 任务比例改为 30% State forward、45% Action inference、25% arbitrary；Action inference 为 30% inverse + 15% history。
- arbitrary 内部为 20% State-only、40% Action-only、40% Both；State+Action 同时 Mask 时 Action reconstruction 权重为 1.25。
- Action curriculum 在 10k/25k/40k local step 把最大区间扩到 32/64/128；25k 后 inverse full-128 额外采样概率为 0.15。
- 训练窗口 50% 均匀裁剪，50% 从每条 episode Action derivative energy 最高 25% 的窗口中均匀选择。
- 学习率为 Action 头 `5e-5`、共享主干 `2e-5`、State/forward 头 `1e-5`；40k step，BF16，effective batch 64。
- 每 2k step 与固定 parent validation baseline 比较；四项 State guard 均不得超过 parent 的 105%，否则只能写 `best_unguarded.pt`，不能更新正式 `best.pt`。

正式 Action 改善门禁为 inverse local 至少 10%、inverse full-128 至少 15%、Action completion macro 至少 10%。任一失败时保留诊断产物但不生成 `cvae_action_finetune.ok`。

Ubuntu 先 smoke：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_physics_dataset_20260825_235244 \
CVAE_INIT_CHECKPOINT=/home/helloworld/bly/runs/cvae_train_20260826_002252/checkpoints/best.pt \
CVAE_CONFIG=configs/physics_v3_action_finetune_smoke.json \
CVAE_MODEL_KIND=physics_transformer CVAE_CONTEXT_MODE=hidden CVAE_SEED=20260831 \
bash ./cvae_repro.sh smoke-action-finetune
```

通过后正式训练：

```bash
CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_physics_dataset_20260825_235244 \
CVAE_INIT_CHECKPOINT=/home/helloworld/bly/runs/cvae_train_20260826_002252/checkpoints/best.pt \
CVAE_CONFIG=configs/physics_v3_action_finetune.json \
CVAE_MODEL_KIND=physics_transformer CVAE_CONTEXT_MODE=hidden CVAE_SEED=20260831 \
bash ./cvae_repro.sh action-finetune
```

新会话必须先寻找本次 fine-tune 的实际 run 和 marker/log，再判断当前处于“尚未运行、运行中、失败或完成”哪一种状态，不能仅凭代码存在推断训练成功。

## 8. 已实现评测与视频能力

### 8.1 Action Mask：物理重放

公共入口保持为：

```bash
bash ./cvae_repro.sh validate-action-mask-replay
```

它只使用 validation motion，默认 `prior_mean`，支持 element、step、random feature、腰/左右腿/左右臂语义组和 `inverse_full_128`。普通 completion 使用 reconstruction Action head；full inverse 使用专用 inverse head。补全 Action 会从 canonical 坐标严格转回 raw Action，并在隔离的单环境 Isaac 进程中逐场景重放；Mask 外 Action 必须位级不变。

源轨迹、original replay 和补全场景共享相同平面、seed、初始状态与 runtime mapping。物理重放成功门禁与模型质量分开：即使 CVAE 不如 hold-last/线性插值，仍保存 run，并令 `model_quality_pass=false`。已有回传表明 source replay、runtime mapping/context、planned/executed Action 与 pre-mask identity 均可 PASS；当前 parent 模型的 Action `model_quality_pass` 仍为 false。

代表视频固定包括 source、original replay、element-25、step-8、feature-25、left-leg、inverse-full-128 和总览 grid。`render=none` 仍会完整执行补全与 Isaac 重放，只跳过 MP4；改为 `representatives` 会重新执行整条评测，不会复用上一次 run。

### 8.2 State Mask：运动学三联视频

公共入口：

```bash
bash ./cvae_repro.sh validate-state-mask-video
```

该路径直接只读 Physics v4 HDF5，不启动 Isaac。左栏为记录的 root/joint 轨迹，中栏为由真实 70 维 State 积分重建，右栏为由预测 State 积分重建。Action 始终完整可见且不会被改写，因此与 Action replay 完全隔离。

1/2/4/8 步 forward rollout 属于训练分布内；32步预测由四段8步 rollout 连续推进并标记 OOD。代表视频已改为包含 32 步连续 State completion。已回传 `render=none` run `/home/helloworld/bly/runs/cvae_state_mask_eval_20260827_120853`，pipeline PASS；生成 MP4 时必须创建新的 `representatives` run。

## 9. 运行产物与 marker 协议

所有 run 使用：

```text
~/bly/runs/<run_id>/
├── data/
├── logs/
├── checkpoints/
├── videos/
├── manifests/
└── markers/
```

只有完整验收通过后才允许生成 `.ok` marker；Shell 非零退出会保留失败目录和 `cvae.failed`/日志，不覆盖旧 run。常用 marker：

| 阶段 | Marker |
|---|---|
| Physics 采集 | `collect_physics_state_action.ok` |
| Physics v4 索引 | `cvae_physics_dataset.ok` |
| 常规训练 | `cvae_train.ok` |
| Action fine-tune smoke | `cvae_action_finetune_smoke.ok` |
| Action fine-tune 正式 | `cvae_action_finetune.ok` |
| Action replay | `cvae_action_mask_replay.ok` |
| State 视频 | `cvae_state_mask_video.ok` |

`latest_*_run_dir.txt` 只在成功后更新，运行中的新目录不能依赖 latest 查找，应使用 `ls -dt ~/bly/runs/<prefix>_* | head -n1` 并核对创建时间。大 HDF5、checkpoint、MP4 和 BONES-SEED 归档不得未经体积检查提交 Git。

## 10. 下一步优先级

1. 在 Ubuntu 核对 `197a173` 已同步，运行 Action fine-tune smoke；回传 `training_summary.json`、parent baseline、最后 train/validation metric 和 marker。
2. smoke 通过后运行 40k 正式 fine-tune，监控三组学习率、curriculum stage、State guard 与 Action score；预计约 9 小时，但以实际 step rate 为准。
3. 只使用满足 State guard 且通过 Action 改善门禁的 `best.pt`；若无成功 marker，parent 仍是有效模型，不得用 `best_unguarded.pt` 冒充正式结果。
4. 对选中 checkpoint 运行 validation 离线指标、简单直行与两个非 Locomotion Action replay 视频，并用 State 视频回归确认 State 能力未退化。
5. 最终 test 可以复用既有 split，但报告必须明确写 `reused test`；优先比较 parent 与 fine-tune 的同一固定 validation fixture。
6. 若 Action 仍差，先按 Mask 类型/区间长度/关节组诊断，不立即扩大模型或重新采集；区分多解性、缺 goal、latent prior 和物理开环漂移。
7. oracle dynamics context 仅作为后续上限实验；不得将其结果与 hidden-context 主模型混为部署性能。

## 11. 后续代理启动检查

Windows 开始改动前：

```powershell
cd C:\Users\86136\Desktop\code\RL\bly
Get-Content AGENTS.md -Raw
git status --short --branch
git rev-parse HEAD
git -C sonic-repro/GR00T-WholeBodyControl status --short --branch
git -C sonic-repro/GR00T-WholeBodyControl rev-parse HEAD
git -C sonic-repro/IsaacLab status --short --branch
git -C sonic-repro/IsaacLab rev-parse HEAD
```

Ubuntu 执行前：

```bash
cd /home/helloworld/bly
git status --short --branch
git rev-parse HEAD
git -C sonic-repro/GR00T-WholeBodyControl status --short --branch
git -C sonic-repro/GR00T-WholeBodyControl rev-parse HEAD
git -C sonic-repro/IsaacLab status --short --branch
git -C sonic-repro/IsaacLab rev-parse HEAD
nvidia-smi
df -h /home/helloworld/bly/runs
```

不得因为本文记录了某个旧 HEAD 就自动 checkout/reset/pull。先保护实际工作树，再对实际分支执行 `git pull --ff-only`；若不满足 fast-forward 条件，停止并回传状态。

## 12. 代码入口索引

| 任务 | 主要文件 |
|---|---|
| Physics v3 采集/验证 | `sonic-repro-kit/sonic_repro.sh`、`consolidate_physics_state_action.py`、`verify_physics_state_action.py` |
| SONIC 嵌套改动 | `sonic-repro-kit/patches/0001`–`0008`（`0008`待 Ubuntu 应用） |
| Physics v4 index/dataset | `state-action-cvae/src/cvae_sa/physics_indexer.py`、`dataset.py`、`physics_schema.py` |
| 模型与损失 | `models.py`、`losses.py`、`masking.py` |
| 常规/fine-tune 训练 | `trainer.py`、`configs/physics_v3*.json`、`cvae_repro.sh` |
| Action completion/replay | `action_mask_eval.py`、`action_masks.py`、SONIC kit replay/render 脚本 |
| State completion/video | `state_mask_eval.py`、`state_masks.py`、`render_state_mask_comparison.py` |

修改前优先阅读对应入口及测试。当前测试命令为：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
```

Windows 可运行不依赖 Isaac/HDF5 的轻量测试；完整 HDF5、CUDA、Isaac replay 与视频门禁必须在 Ubuntu 固定环境执行。
