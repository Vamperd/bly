# 65-token 层级 CVAE 模型合同

当前活动模型是 `physics_hierarchical_standard_cvae_transformer` 的
`65-token-hierarchical-standard-cvae-v1` 实现。它直接替换历史 H50-SCVAE 语义，首次训练随机初始化；旧 H50 checkpoint 不迁移、不部分加载。

## 输入与输出

数据合同仍是 `State=[T+1,70]`、`Action=[T,29]`，窗口为 64 个 transition。模型在内部补齐一个终点 Action：`Action_64=0` 且 `valid_action[64]=false`，因此终点只保留 `State_64`，Action 输出和 Action loss 仅覆盖 `t=0..63`。

Posterior 每个时间拼接 `[State_t(70), Action_t(29)]`，得到 `X:[B,65,99]`。它只读完整序列，完全不接收 mask。Condition 每个时间拼接 `[value_i, mask_i]`，得到 `C:[B,65,198]`；被 mask 的 value 先置零，mask bit 保留为 1。Condition 可见 value、mask bit 和有效 State token 进入唯一的 Condition Encoder。

## 两个 Encoder 与硬 chunk

Posterior 使用独立 `CLS_q`：`[CLS_q,X_0,...,X_64]`。`CLS_q` 经过 global mean/logvar head 得到 `q_global:[B,256]`；65 个 data hidden 按硬范围池化并经过 local mean/logvar head 得到 `q_local:[B,16,128]`。Posterior 的 logvar 只用于 Stage C 的标准正态 KL。

Condition 使用独立 `CLS_c`：`[CLS_c,C_0,...,C_64]`。`CLS_c` 只生成确定性 `condition_global:[B,256]`；65 个 data hidden 同样池化生成 `condition_local:[B,16,128]`，同时保留 `condition_memory:[B,65,d_model]`。`CLS_c` 不重复放入 decoder memory。两个 Encoder 的时间自注意力均为双向。

硬 chunk 映射为：`0..3, 4..7, ..., 56..59, 60..64`。最后一个 `local_15` 明确包含 `State_64`；padding chunk 使用可学习 fallback，不能改变 chunk 边界。

## Latent 融合、memory 与 FiLM

global 和 local 各有一套独立 gated-residual MLP：一套生成 decoder memory，另一套生成逐时间 FiLM 输入。local MLP 在 16 个 slot 间共享，并拼接 slot embedding。Condition 存在时，global 融合 `[q_global,condition_global]`，local 融合 `[q_local_i,condition_local_i,slot_i]`；Stage A 没有 Condition Encoder，condition 分量为零。

有条件时 decoder memory 为：65 个 `condition_memory` token、1 个 fused global token、16 个 fused local token，共 82 个 token。Stage A 为 posterior-only memory，共 17 个 latent token。FiLM 输入为 `U:[B,65,d_model]`：global 投影加对应 hard-chunk local 投影，`t=60..64` 均使用 `local_15`。每个 decoder layer 有独立的 FiLM head，weight/bias 零初始化；decoder block 顺序保持 self-attention → condition/latent cross-attention → FiLM → feed-forward。

## 三阶段课程

**Stage A（posterior，入口 `fixed`/`smoke`）**：完整 `X` → Posterior mean → latent-only decoder memory。Condition Encoder 不调用；训练 Posterior、latent fusion 和 Decoder，使用完整 State/Action 重建目标。

**Stage B（condition，入口 `random`）**：Posterior 以 mean 运行并冻结；不完整 `C` → Condition Encoder → condition memory/global/local；两路 latent 融合后重建。被 mask 真值在 Condition 输入中先置零，不能由 Condition 或 memory 泄漏。

**Stage C（KL，入口 `kl`）**：解冻全部模块，Posterior 使用 reparameterization sample；Condition latent 仍确定性。损失为重建加 `β_g KL(q_global||N(0,I)) + β_l KL(q_local||N(0,I))`，global/local 各自按元素归一化后等权；beta warm-up 由配置控制。不使用 `KL(q||p(z|C))`、q-p alignment 或 latent 对齐。

## 推理与隔离规则

最终推理只执行 `C → Condition Encoder → condition memory/global/local`，然后独立采样 `ε_global∼N(0,I_256)` 与 16 组 `ε_local∼N(0,I_128)`，再走两套融合、82-token memory 和 decoder；不得调用 Posterior Encoder。Posterior 对 mask 改变必须完全不变；Condition 只能依赖置零后的可见 value 与 mask bit。输出固定为 `State:[B,65,70]`、`Action:[B,64,29]`。

## 配置与 checkpoint

默认规模为 `d_model=448`、Posterior/Condition Encoder 各 6 层、Decoder 8 层、`heads=8`、`ffn_dim=1792`、`global=256`、`local=16×128`、`state_input_dim=99`、`condition_input_dim=198`。按当前实现构建的默认总参数量为 `64,377,959`（实际运行仍以构建结果为准）。参数量由构建后的模型计算并写入 summary、`model_signature.json` 和 checkpoint。checkpoint 必须包含 `architecture_version`、结构签名和参数量；加载旧 H50-A/H50-SCVAE 格式时显式报 `architecture signature mismatch`。

训练运行时可通过 `CVAE_POSTERIOR_MAX_STEPS`、`CVAE_POSTERIOR_LEARNING_RATE`、`CVAE_POSTERIOR_LR_SCHEDULE`（`constant/cosine/linear`）、`CVAE_POSTERIOR_WARMUP_STEPS` 和 `CVAE_POSTERIOR_MIN_LR_RATIO` 覆盖步数与学习率曲线；Python 入口对应 `--max-steps`、`--learning-rate`、`--lr-schedule`、`--warmup-steps`、`--min-lr-ratio`。这些只改训练协议，不改变架构签名。

Stage A 训练沿用旧 posterior capacity 的可审计记录方式：每个 optimizer step 立即 append+flush 到 `logs/metrics.jsonl`，记录 reconstruction/KL、学习率、梯度范数、耗时和 CUDA 峰值显存；`--validation-interval`（默认 1000）对全部 selected windows 做顺序完整评测，只有完整评测才比较并原子更新 `checkpoints/best.pt`。`--checkpoint-interval`（默认 250）滚动原子更新 `checkpoints/last.pt`；每次评测也更新 last。`plots/training_curves.svg` 和 `manifests/progress.json` 随评测/周期 checkpoint 刷新，异常或 Ctrl-C 会保存 last 并写 `markers/cvae.interrupted`。`--resume-run <run>` 恢复 model、optimizer、scheduler、RNG 与 optimizer step，且严格要求训练合同、数据身份和 architecture signature 一致；Shell 对应环境变量为 `CVAE_POSTERIOR_VALIDATION_INTERVAL`、`CVAE_POSTERIOR_CHECKPOINT_INTERVAL`、`CVAE_POSTERIOR_LOG_INTERVAL`、`CVAE_POSTERIOR_RESUME_RUN`。
