# 活动合同：H50-SCVAE 标准条件 CVAE

最后更新：2026-09-14

状态：正式M-F run `/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_h50_fixed_20260912_015547`已跑满60k，execution PASS、quality FAIL、best joint score `2.1918797`。H50-A exact-init正式run `/home/helloworld/bly/runs/cvae_posterior_h50a_action_replay_exact_init_20260914_112412`已通过execution、初始化身份和原Action基线，并支持该单个已见窗口的H50-A Action重放。当前已实现完整episode定性回放，可分别补充三个已见motion；这不升级为模型门禁，随后仍恢复M-F2。M-R和KL仍被质量marker阻断；CPD、D1与CRA均为历史`SUPERSEDED`。

### 0.1 当前唯一执行项：精确初始化重放

`posterior-h50-action-replay-exact-init`继续使用两个独立`num_envs=1` Isaac进程，两次reset后都加载同一份带SHA256的`exact_initialization.npz`。它恢复训练HDF的episode首帧root/关节状态、世界速度、初始Action历史与target，以及关节和刚体动力学参数；每次在第一条Action前生成独立readback。训练HDF完整episode直接渲染，不重新仿真；original重放整段原始Action，H50-A重放也覆盖整段episode，但只在所选T64窗口使用预测Action，其余Action与原记录逐位相同。视频底部进度条把该64步替换区间标红。SONIC端的可选reset hook由外层补丁`sonic-repro-kit/patches/0009-feat-restore-exact-external-replay-initialization.patch`提供，普通replay不触发该路径。

首次运行暴露出Isaac版本差异：配置声明的`{terrain}/physicsMaterial`并不保证存在。修复后不再硬编码该路径，而是发现实际Plane/Collision prim，创建`exactReplayPhysicsMaterial`并显式绑定，同时把collision/material路径与binding target写入readback。失败run不得续用；新run必须重新执行prepare与两次Isaac进程。

初始化身份通过要求两次readback最大差`≤1e-6`，各自与HDF State/context最大差`≤1e-5`，姿态误差`≤1e-4°`，Action历史/target误差`≤1e-6`。随后original基线要求joint RMSE`≤0.02 rad`、root RMSE`≤0.05 m`、姿态max`≤5°`、body MPJPE`≤0.05 m`、contact一致率`≥95%`。原Action基线不通过时不得评价H50-A，只能调查尚未记录的PhysX隐藏状态。

代码同步及补丁应用由用户预先完成后，Ubuntu只运行：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
export CVAE_POSTERIOR_H50_REPLAY_SOURCE_RUN=/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_continue15k_20260909_230256

unset CVAE_CONFIG CVAE_RUN_DIR CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_WARM_START
unset CVAE_POSTERIOR_H50_REPLAY_CHECKPOINT CVAE_POSTERIOR_H50_REPLAY_MOTION_KEY
unset CVAE_POSTERIOR_H50_REPLAY_VARIANT CVAE_POSTERIOR_H50_REPLAY_WINDOW_START

bash ./cvae_repro.sh posterior-h50-action-replay-exact-init
```

历史正式run已通过并生成三份65帧独立视频与三栏视频。完整episode扩展已在Windows实现并通过单元、编译与Shell检查，尚待Ubuntu对三个motion分别新建run；每个run都会生成三份完整视频和三栏视频，并在manifest记录完整帧数、红色窗口起止及窗口前/内/后的轨迹指标。

## 0. 已完成插入式诊断：H50-A已见窗口Action重放

这不是新的训练，也不改变SCVAE路线。默认读取H50-A续训run的step34000 `last.pt`，确定性选择按motion名称排序的第一个`variant 0 / start 0 / T64`训练窗口，并使用H50-A实际训练过的`full_both` Mask：posterior看完整窗口，decoder的State/Action condition全部被Mask。模型重建出的64步Action替换原Action后，在同一Isaac配置中与原Action各重放一次。

正式run为`/home/helloworld/bly/runs/cvae_posterior_h50a_seen_window_action_replay_20260913_230317`，源码`0f39e35424b3d616e52418c20ae8b13e324f81d8`。选择窗口为`baby_full_diaper_walk_ff_360_loop_R_001__A462 / variant 0 / start 0`。预测Action的normalized RMSE为`0.015191`，物理RMSE为`0.005441 rad`。中/右两栏的关节位置RMSE为`0.008448 rad`、root位置RMSE为`0.003229 m`、root姿态mean/max为`0.672°/2.010°`、body MPJPE为`0.004635 m`；两次mapping、runtime context及planned/executed Action均逐值一致。

训练记录与原Action重放的基线没有复现：关节位置RMSE`0.245825 rad`、root位置RMSE`0.251772 m`、root姿态max`106.918°`、body MPJPE`0.418972 m`。因此本实验支持“H50-A在这个已见窗口准确记忆Action，预测Action与原Action在相同重放条件下效果接近”，不能支持“H50-A复现了训练HDF5轨迹”。主视频位于run内`videos/h50a_seen_window_action_replay.mp4`，大小577 KiB。

Ubuntu命令不包含Git操作：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
export CVAE_POSTERIOR_H50_REPLAY_SOURCE_RUN=/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_continue15k_20260909_230256

unset CVAE_CONFIG CVAE_RUN_DIR CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_WARM_START
unset CVAE_POSTERIOR_H50_REPLAY_CHECKPOINT CVAE_POSTERIOR_H50_REPLAY_MOTION_KEY
unset CVAE_POSTERIOR_H50_REPLAY_VARIANT CVAE_POSTERIOR_H50_REPLAY_WINDOW_START

bash ./cvae_repro.sh posterior-h50-action-replay
```

以上旧命令仅保留用于复现历史run；本次`.ok`只表示推理、两次Isaac重放、指标和MP4均完整，不代表采集环境被精确恢复，也不代表conditional prior或随机Mask质量通过。

正式历史和数值见 [plan.md](plan.md)，通俗背景见 [explain.md](explain.md)，安全约束见 [AGENTS.md](AGENTS.md)。

## 1. 不可改变的数据流

```text
训练 posterior：完整序列 x + 当前 Mask c
                         ↓
                    q(z | x,c)
                         ↓
                  一份 zq ─────────┐
                                    ↓
Mask 后的可见序列 c → condition memory → 共享 D(zq,c) → 完整序列

训练/部署 prior：Mask 后的可见序列 c
                         ↓
                    p(z | c)
                         ↓
                  一份 zp ─────────┐
                                    ↓
Mask 后的可见序列 c → condition memory → 共享 D(zp,c) → 完整序列
```

posterior 与 prior 只共享同一个 decoder 的参数，不在一次推理中融合。每次 decoder 调用只能接收一组 `global + 16 local` latent。禁止拼接、求平均、attention 融合或 posterior fallback。最终部署固定为：

```text
c → p(z|c) → prior sample/mean → D(zp,c)
```

部署接口必须在 posterior 被禁用或抛错时仍可运行。隐藏真值只允许进入 posterior；prior 与 decoder condition 必须先把被 Mask 的完整 Token 置零。

## 2. 模型和初始化

新增模型种类 `physics_hierarchical_standard_cvae_transformer`：

| 组件 | 固定合同 |
|---|---|
| 主宽度 | 448；8 heads；FFN 1792 |
| posterior | 6层，读取完整真值和当前 Token Mask |
| conditional prior | 独立6层，只读取可见值和 Mask |
| condition encoder | 独立4层，读取真实 masked condition |
| latent | 1×256 global + 16×128 local |
| decoder | 8层 self-attention、condition+latent cross-attention、latent FiLM、FFN |
| 参数量 | 精确 `66,129,571`；其中 logvar 头 `344,832` |
| 正则 | dropout、weight decay均为0 |

初始化源固定为：

```text
/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_continue15k_20260909_230256/checkpoints/last.pt
```

H50-A step 34000 只初始化 posterior、condition encoder、decoder和均值头；不冻结、不保留旧 latent 坐标或旧输出。prior从posterior复制，新增Mask输入列置零。四个q/p global/local logvar头以权重0、bias `-4` 初始化，KL前冻结。optimizer、scheduler和RNG均不继承。

## 3. KL 前的均值训练

每条路径的重建损失为：

```text
R = 0.75 × 被Mask位置重建 + 0.25 × 完整有效序列重建
```

State MSE、Action MSE和contact BCE在各自存在时等权。均值阶段使用两次完全独立的decoder调用：

```text
Lmean = R(D(mu_p,c)) + 0.25 R(D(mu_q,c)) + lambda_z Lalign
```

`Lalign`只把prior均值拉向当前posterior均值，posterior目标侧stop-gradient；posterior仍由自己的重建项更新。H50-A初始latent标准差仅作量纲归一化，不作teacher目标。固定阶段 `lambda_z` 在0–10k为1，10k–30k线性降至0.1，之后为0.1；随机阶段为0.05。

M-F固定阶段：32 motion×8 variants、T64、stride64，使用原8类物理Mask，最多60k，step0及每2k完整评测，连续3次PASS提前结束。FP32、micro4×累积16、AdamW、clip1；encoder/condition/mean heads峰值LR `1e-4`，完整decoder `3e-5`，warmup1000后cosine到`1e-6`。

M-R随机阶段：从M-F `best_mean_fit.pt`只加载模型参数并重置训练状态。60%为原8类的动态版本，40%为2–3个互不重叠且中间至少保留一个完整transition的物理缺口组合。禁止散点、feature/element、full State和full both。最多40k；encoder峰值LR `5e-5`、decoder `1e-5`、warmup500。每2k检查固定8类，每4k同时检查独立seed的16个held-out物理Mask。

均值门禁要求posterior mean和prior mean分别满足：global State/Action RMSE≤`0.02`，每类worst-window State/Action≤`0.04`，p99 abs≤`0.08`，masked和完整contact均100%；q–p global/local标准化RMSE≤`0.25`且cosine≥`0.95`。max abs只报告。若所有高遮挡Mask的zero/cross-window/cross-motion latent替换ratio均≤`1.05`，视为latent被忽略并阻止进入KL。

## 4. K1 标准 KL 与三条独立路径

只有M-R连续三次同时保持held-out、固定Mask和posterior/prior均值门禁，才从其 `best_mean_fit.pt`进入KL：

```text
zq = mu_q + sigma_q * epsilon
L  = R(D(zq,c)) + beta KL(q(z|x,c) || p(z|c))
```

解冻logvar，范围裁剪到`[-8,4]`，free bits=0。beta前10k由0升至`1e-3`后固定；最多50k。q/p encoder峰值LR `5e-5`、logvar heads `1e-4`、decoder `1e-5`。每2k评测均值，每5k做8次采样完整评测。

三条路径必须分开运行：`D(mu_q,c)`、`D(mu_q+sigma_q epsilon,c)`、`D(mu_p+sigma_p epsilon,c)`。同一window、Mask和sample index下q/p共享同一epsilon，但latent不融合。记录mean/std/p50/p95/worst/best-of-8；best-of-8不控制PASS。

prior sample八次平均须满足fit门禁，跨采样p95 score≤`1.5`，每次contact 100%；posterior sample相对posterior mean退化≤25%，prior sample相对posterior sample退化≤50%。若posterior明显受损只允许从M-R以beta `1e-4`复核一次；若posterior良好但prior误差超过2倍，只允许beta `1e-2`复核一次。

## 5. Ubuntu 执行顺序

命令不包含Git操作；用户需预先完成同步。先执行smoke：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
export CVAE_POSTERIOR_STANDARD_CVAE_SOURCE_RUN=/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_continue15k_20260909_230256

unset CVAE_CONFIG CVAE_RUN_DIR CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_WARM_START
unset CVAE_POSTERIOR_STANDARD_CVAE_INIT_RUN

bash ./cvae_repro.sh posterior-hierarchical-standard-cvae-smoke
```

审核smoke后必须重新从H50-A启动正式M-F，不能从smoke续训：

```bash
unset CVAE_POSTERIOR_STANDARD_CVAE_INIT_RUN
bash ./cvae_repro.sh posterior-hierarchical-standard-cvae-fixed
```

只有M-F质量marker存在才执行M-R；只有M-R质量marker存在才执行KL：

```bash
export CVAE_POSTERIOR_STANDARD_CVAE_INIT_RUN=/home/helloworld/bly/runs/<正式M-F-run>
bash ./cvae_repro.sh posterior-hierarchical-standard-cvae-random-physical

export CVAE_POSTERIOR_STANDARD_CVAE_INIT_RUN=/home/helloworld/bly/runs/<正式M-R-run>
bash ./cvae_repro.sh posterior-hierarchical-standard-cvae-kl
```

每次回传 `manifests/standard_cvae_summary.json`、`source_commit.txt`、marker列表、checkpoint列表及最后三次evaluation。质量失败时停止，不越过marker强行启动下一阶段。

## 6. 产物、状态和结论边界

核心产物为 `standard_cvae_summary.json`、`physical_random_mask_bank.json`、KL阶段的`kl_three_path_comparison.json`、四张SVG、`best_mean_fit.pt`/`best_kl_fit.pt`及`last.pt`。smoke、execution、M-F质量、M-R质量、KL比较完整和KL质量使用独立marker。

截至本文更新，正式M-F在step60000的prior mean为global State/Action `0.043838/0.026704`、worst `0.074567/0.041620`、p99 `0.137523`，posterior mean为`0.038486/0.023520`、worst `0.058044/0.034188`、p99 `0.120053`；两者contact均100%。q-p global/local标准化RMSE `0.052575/0.080972`、cosine `0.999920/0.999921`，alignment通过且latent未被忽略。这表明当前先要解决联合训练中的重建底座，而非继续加强q-p对齐。即使后续全流程通过，也只能声明：模型在已见32-motion、T64窗口上，能对预注册的物理可推测完整Token Mask进行均值补全，并能从conditional prior采样得到稳定结果；不能声明未见motion泛化、任意无物理线索Mask或所有组合的数学完备性。
