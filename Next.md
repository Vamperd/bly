# 活动合同：H50-SCVAE 标准条件 CVAE

最后更新：2026-09-12

状态：Windows 已完成模型、固定/随机物理 Mask、均值训练、KL 三路径评测及 Shell 入口实现；尚未收到 Ubuntu smoke。H50-A 仅作初始化。CPD、D1 与 CRA 均为历史 `SUPERSEDED`，不得用于新路线初始化。

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

截至本文更新，代码静态/轻量测试已证明接口隔离和参数合同，但没有Ubuntu训练质量结果。即使全流程通过，也只能声明：模型在已见32-motion、T64窗口上，能对预注册的物理可推测完整Token Mask进行均值补全，并能从conditional prior采样得到稳定结果；不能声明未见motion泛化、任意无物理线索Mask或所有组合的数学完备性。
