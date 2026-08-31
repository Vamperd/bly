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

截至 2026-08-30 的已观测状态：

- Windows 外层当前分支字面值为 `tiny-model`，已观测 HEAD 为
  `fcdb4f8861e539e3ea364e578d6bc96ce7ebd9b0`；它包含 Exact Training Fixture
  检测及零 Action-target 窗口修复。Action-focused fine-tune 的较早实现提交为
  `197a1730fd829a512e153755f56e6e97d4b1329d`。Ubuntu 同步前仍须读取其实际分支，
  禁止为了匹配本文自动 reset。
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

### 6.1 32-motion 数据与联合容量实验：已在 Ubuntu 完成

固定记忆数据集为：

```text
/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
```

它使用 seed `20260828`，从 Physics v4 原 train split 的八个 package 各选择 4 个
八 variant 全 completed motion，共 32 motion、256 episode。窗口为 128 transition、stride 64，
共有 1248 个固定 train 窗口；子集 HDF5 只读引用原数据并重新计算归一化统计。全部数据均标记
为 train，只能解释为 memorization benchmark，不能作为泛化结果。

容量阶段已修正为 posterior mean gate、prior mean 仅诊断；训练仍为 dropout/KL/auxiliary/cycle
权重 0、20k optimizer step、effective batch 64、500 step warmup 到 `2e-4` 后 cosine 降到
`2e-6`。修正后 compact/reference 仍未通过当时的固定 seed **unseen-Mask** 同集门禁：

| 模型 | 参数量 | 最佳 unseen score | step | forward | inverse | arbitrary State | arbitrary Action | history | rollout-8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 15,065,048 | 5.7831 | 20000 | 0.1835 | 0.2892 | 0.2826 | 0.0956 | 0.2002 | 0.3097 |
| reference | 28,475,448 | 5.4818 | 19500 | 0.1735 | 0.2741 | 0.2560 | 0.0819 | 0.1749 | 0.2679 |

reference 参数增加约 89%，各项仅改善约 5%–14%，因此现有证据不支持“继续扩大统一主干”是
主要解法。由于 capacity 未通过，compact/reference 的 Full CVAE 阶段、额外 compact seeds
均未启动；不得把尚未执行的 Full/prior 结果写成失败或成功。

### 6.2 五个单任务与 Exact Training Fixture：已在 Ubuntu 完成

五个 compact 单任务均从随机初始化训练 20k step，单个 run 各见 1,280,000 samples，约
1025.64 effective epoch；源 run 为：

```text
forward_rollout:
/home/helloworld/bly/runs/cvae_overfit_single_compact_forward_rollout_20260829_204847
inverse:
/home/helloworld/bly/runs/cvae_overfit_single_compact_inverse_20260830_025717
history_action:
/home/helloworld/bly/runs/cvae_overfit_single_compact_history_action_20260830_063605
arbitrary_state:
/home/helloworld/bly/runs/cvae_overfit_single_compact_arbitrary_state_20260830_095617
arbitrary_action:
/home/helloworld/bly/runs/cvae_overfit_single_compact_arbitrary_action_20260830_130446
```

`diagnose-overfit-fixture` 已在 Ubuntu 对上述五个 run 的 `best.pt`/`last.pt` 完成只读验收，
对每个 checkpoint 重建全部 1248 个训练窗口的相同 Mask，并与旧 unseen-Mask suite 分命名空间
比较。manifest 格式为 `sonic_overfit_exact_fixture_diagnostic_v1`，`execution_pass=true`；
`cvae_overfit_fixture_diagnostic.ok` **只表示检测完整和数据一致**。最佳/最后 checkpoint 均没有
使三个正式 compact 任务全部 exact 通过：

| 任务/指标 | last exact RMSE | 阈值 | last unseen RMSE | 结论 |
|---|---:|---:|---:|---|
| forward-one | 0.0542 | 0.05 | 0.2736 | exact 仅超阈值 8.3% |
| rollout-8 | 0.0578 | 0.10 | 0.3339 | exact PASS |
| inverse | 0.2182 | 0.05 | 0.2926 | 无 reference 诊断，远未通过 |
| history Action | 0.0880 | 0.08 | 0.3879 | 无 reference 诊断，exact 超阈值 9.9% |
| arbitrary State | 0.1254 | 0.05 | 0.4563 | exact 超阈值 150.9% |
| arbitrary Action | 0.0662 | 0.05 | 0.1995 | exact 超阈值 32.4% |

Exact 与 unseen 对照证明两个事实必须同时保留：旧门禁确实额外测量了较强的 Mask 组合迁移，
但模型对训练 fixture 本身也没有全部过拟合。`objective_metric_gap_flag` 对全部 task/checkpoint
均为 false，训练 loss proxy 与 exact RMSE 没有超过 2 倍的异常脱节，不能再把失败归因于
RMSE 实现错误。

旧 unseen-Mask checkpoint 选择还错过了更好的训练记忆点：forward 的 exact score 从
best step 6000 的 3.2422 降到 last step 20000 的 1.0833；history 从 step 15000 的 1.3851
降到 1.0994；arbitrary State 从 step 15500 的 3.1044 降到 2.5088。inverse 和 arbitrary
Action 的 best 已是 step 20000，与 last 相同。后续容量实验必须保存 `best_exact.pt`，unseen
Mask 只作诊断，不能继续用它选择记忆 checkpoint。

`arbitrary_action` 的 exact fixture 还发现 1248 个窗口中只有 1112 个含 Action target，136 个
窗口（10.9%）为零目标。这来自 Action-only 任务抽到 4 个 State-only semantic group；这些窗口
仍正确计入 fixture hash/逐窗口 JSONL，但不参与 Action RMSE。当前训练结果是真实协议的忠实
重放，下一轮则必须把 Action-only semantic 抽样限制到五个 Action 关节组，并要求每个固定
窗口至少有一个目标。

独立 `analyze-overfit` 入口也已在 Ubuntu 成功执行。已回传的定性结论为：该 32-motion 子集的
RobotInfo 数值方差接近零，任务间梯度余弦大多为正，输入敏感度以足接触、gravity、base
linear/angular velocity 等组更高。单任务 exact 仍失败说明多任务梯度冲突不是唯一原因；
inverse/history 的不可辨识或缺 reference、arbitrary completion 的编码/池化路径和训练协议
应优先于继续增加统一 Transformer 宽度。若需引用具体数值，必须重新读取实际 analysis
manifest，不得从本段定性描述反推数值。

### 6.3 LeanSplit v1 与 Physics v5：代码已实现但尚未运行

新增 `physics_lean_split` 为 6,204,665 参数：因果动力学分支使用至少10步历史与已发送
Action，不读取 reference/CVAE latent；Action 分支可读取 runtime command manager 直接记录的
`10×64` reference；双向 CVAE 仅负责 arbitrary completion。State/Action 仍为70/29维。
Physics v5 数据合同、`patches/0008` recorder、四类 Action 信息增量配置均已实现，但本文尚未
收到 Ubuntu smoke、v5 collection 或 LeanSplit 正式单任务训练日志，不得表述为已经验证成功。
`sonic-repro.sh prepare-overfit-reference-subset` 会从旧 overfit selection manifest 提取同一
32 个 motion，并在新 run 中建立经 hash 校验的只读绝对软链接；不得用另一批 motion 代替。

### 6.4 最简 posterior Transformer capacity：S0已通过、F1已失败、D1待运行

新增独立 `physics_posterior_transformer`：只读取归一化 State、Action、逐特征 Mask 与位置/类型，
使用共享双向 encoder、单个 global latent 和单个双向 decoder，不包含 RobotInfo、reference、
relation/rollout/auxiliary head。固定10类 Mask fixture 包括全 State、全 Action、全序列与部分
element/time/feature/semantic Mask；训练使用 posterior mean、`KL beta=0`，按 exact score 保存
`best_exact.pt`。代码、Windows 轻量测试和 Ubuntu S0 工程 smoke 已完成，但尚无正式容量
阶梯结果，不得写成已经实现完美拟合。入口为：

```bash
bash ./cvae_repro.sh posterior-capacity-smoke
bash ./cvae_repro.sh posterior-capacity
```

当前用户明确选择先执行该 posterior-only 容量实验；它只证明 posterior 无损记忆，不证明
部署时 `State→Action`、`Action→State` 或 conditional prior 能力。完整执行阶梯与结果台账位于
根目录 `plan.md`；每个 run 完成后必须用实际 summary/metrics/marker 更新该文件的状态、结论和
唯一下一步，再启动后续实验。

2026-08-31 已回传 S0 smoke run
`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t8_20260831_114746`：1 motion、window 8、
2 optimizer step、288 windows、2,880 fixtures，`cvae_posterior_capacity_smoke.ok` 已通过。这只验证
HDF5/CUDA/训练/evaluator/checkpoint/marker 工程链路；step-2 State RMSE 2.5195、Action RMSE
2.2020 等质量失败属于预期，不是容量结论。

随后正式 F1（1 motion、window 16、fixed）已运行结束，并由
`posterior_capacity.py` 抛出 `posterior capacity experiment did not satisfy every exact gate`。
失败 run 为 `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_20260831_114833`，
完成40k step；最佳/最后均在step 40000，score 752.0263，worst State/Action RMSE为
0.05842/0.02495、max abs 0.75203、contact 100%、zero-latent ratio 371.67。`full_state`、
`full_action`、`full_both` 已分别约0.00256、0.00239和0.00260/0.00247，但partial element/
feature/semantic Mask显著更差，尤其`element_both_50`达到0.05842/0.02495。39k–40k指标平台化；
这说明latent确实被使用且contact已拟合，主要瓶颈是部分Mask下可见token、Mask与global latent的
统一decoder融合，或144窗口规模下的容量/优化。下一步只做单固定window×10 Mask的D1诊断；
在D1结论前不得启动G0或扩大motion/window。

D1入口已在Windows实现：设置`CVAE_POSTERIOR_MAX_WINDOWS=1`后，代码仍先校验1个motion的8个
variant完整性，再按固定索引顺序只选择第一个window供训练和exact验收。summary会记录可用/实际
窗口数及所选window的motion、variant、episode和start；默认不设置该变量时保持原协议。当前只完成
轻量测试，尚未收到Ubuntu D1 smoke或formal结果，不得推断单窗口能够过拟合。

### 6.5 已完成 parent 训练

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

## 7. Action-focused fine-tune：代码已实现但尚未运行的独立分支

代码已经在外层提交 `197a173` 中实现，但当前 overfit 结果表明确定性容量、Mask 合同和
reference 条件仍需先处理；该 fine-tune 暂不列为当前最高优先级。本文仍没有收到 Ubuntu
smoke/正式训练完成日志，因此不得写成已训练完成。新入口：

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
| 32-motion 子集 | `cvae_overfit_subset.ok` |
| 只读数据分析 | `cvae_overfit_analysis.ok` |
| 单任务训练 | `cvae_overfit_single_task.ok`（只有严格 gate 通过才生成；当前五个 run 均未生成） |
| Exact fixture只读诊断 | `cvae_overfit_fixture_diagnostic.ok`（仅表示执行完整） |

`latest_*_run_dir.txt` 只在成功后更新，运行中的新目录不能依赖 latest 查找，应使用 `ls -dt ~/bly/runs/<prefix>_* | head -n1` 并核对创建时间。大 HDF5、checkpoint、MP4 和 BONES-SEED 归档不得未经体积检查提交 Git。

## 10. 下一步优先级

1. 只执行D1：`CVAE_POSTERIOR_MAX_WINDOWS=1`、1 motion、window 16，先smoke再formal；两次都
   从随机初始化开始，记录summary中的唯一`selected_windows`身份。在D1结论前禁止G0/F2。
2. 若D1通过，按相同结构逐级把窗口数扩为4、16、64、144，定位首次失败规模；每级独立随机
   初始化，一次只改变窗口数量，不改变Mask、阈值、学习率或模型。
3. 若D1仍呈现full mask好而partial mask差，优先做decoder融合单变量对照；不得直接增加motion、
   latent维度、Transformer层数或恢复KL/relation/rollout等复杂路径。
4. 应用并验证 `patches/0008` 后，只采集同一 32-motion 的 Physics v5 reference 子集；比较
   history、history+Action queue、history+runtime reference、再加 causal dynamics embedding。
   forward 分支严禁读取 reference，且 reference 扰动不得改变 forward 输出。
5. 在相同 fixed fixture、seed、学习率和 samples-per-task 下比较 compact 与 6,204,665 参数
   LeanSplit v1；inverse 使用 reference-conditioned deterministic 指标和概率覆盖率双报告。
6. 只有确定性 capacity 连续两次 exact 通过后，才恢复联合多任务与 Full CVAE prior gate；
   在此之前不要启动额外 compact seed、Full CVAE 或把 prior 失败解释成部署结论。
7. Action-focused fine-tune 保留为独立历史分支；若后续恢复，仍必须满足 parent State guard。
   motion ID、package/outcome、未来真实 State、真实随机 delay draw 和 oracle dynamics context
   不得进入部署模型；oracle 结果只能明确标注为上限实验。

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
| Exact fixture诊断 | `overfit_fixture_eval.py`、`cvae_repro.sh` |
| 最简 posterior capacity | `posterior_capacity.py`、`models.py`、`configs/posterior_capacity_minimal.json` |
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
