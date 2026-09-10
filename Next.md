# 活动合同：H50-CPD Conditional Prior 蒸馏

最后更新：2026-09-11

状态：Ubuntu smoke已完成工程验收，正式P0/P1/P2首次启动在step0全量评测前因PyTorch大张量`quantile()`限制停止，尚未发生optimizer更新，不形成模型结果。Windows已将全部p99计算替换为确定性CPU quantile并加入超过16,777,216元素的回归测试；待同步后须新建正式run重跑。smoke run为`/home/helloworld/bly/runs/cvae_posterior_hierarchical_prior_h50_cpd_train_smoke_20260911_022214`；其2-step质量与latent失败没有模型含义。H50-CRA已取消，不得再运行其入口。正式事实与历史结果见[plan.md](plan.md)，安全和交接规则见[AGENTS.md](AGENTS.md)。

## 1. 当前问题与正式数据流

H50-A已经证明：完整序列经过posterior encoder得到的层级latent，可以被H50 decoder重建到当前fit门槛。H50-B则表明：直接让decoder读取Mask条件会改变原解码路径并造成遗忘。CPD不再让condition绕过latent。

```text
Teacher：完整 State–Action
               ↓
         冻结H50-A posterior encoder
               ↓
       canonical global + 16 local latent

Student：Mask后的 State–Action + 完整Token Mask
               ↓
         新conditional prior encoder
               ↓
         预测global + 16 local latent
               ↓
         H50-A canonical decoder
               ↓
         完整 State–Action 预测
```

隔离合同：可见值和Mask只能进入conditional prior；decoder接口只接受latent及State/Action有效长度。decoder内部使用与样本内容无关的全Mask基线memory、固定时间和类型query，不能接收可见State、Action或查询Mask。full-both不参与确定性prior训练，只报告不可辨识性结果。

## 2. 模型与固定源

源run固定为：

```text
/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_continue15k_20260909_230256
```

必须读取其step34000 `checkpoints/last.pt`，并校验H50-A continuation、fit marker、数据/window hash、模型签名和checkpoint step。初始化只加载模型参数，不恢复optimizer、scheduler或RNG。

| 组件 | 合同 |
|---|---|
| H50-A teacher/base | 51,005,283参数；posterior和decoder来自step34000 |
| conditional prior | 独立6层、宽448、8 heads、FFN1792 |
| latent | `global [B,256] + local [B,16,128]`，每4个transition一个local |
| 初始化 | 复制H50 posterior的value投影、embedding、encoder、global/local heads；新Mask列置零 |
| 新增/总参数 | 14,779,456 / 65,784,739 |
| 当前概率部分 | 只输出确定性mean；`KL=0`，无logvar和采样 |

新增模型类型为`physics_hierarchical_conditional_prior_transformer`。严格decoder接口为：

```python
decode_from_canonical_latents(
    global_latent,
    local_latents,
    *,
    valid_state,
    valid_action,
)
```

## 3. 数据、Mask与损失

数据固定为原32 motion×8 variant、256 episode、T64、stride64、`random_crop=false`。Mask只允许遮挡完整70维State token或完整29维Action token，不允许element/feature Mask。

训练Mask使用seed `20260840`：55%独立State-only/Action-only/Both随机Token，遮挡率`U(0.05,0.95)`；25%动态物理Mask；10%稀疏1/2/4/8 Token；10% full State或full Action。评测seed `20260841`与训练隔离，每个window固定16个未训练Mask，覆盖三种域的10%/35%/65%/90%，另含single State、single Action、full State和full Action。每次还重评原8类固定物理Mask。

Teacher latent按全部训练window预缓存；缓存必须与在线H50-A输出一致。每个latent维度按teacher训练集标准差归一化，标准差下限`1e-3`：

```text
Lz = 0.5*MSE((z_global_hat-z_global)/sigma_global)
   + 0.5*MSE((z_local_hat-z_local)/sigma_local)

Lrec = 0.5*L_full + 0.5*L_masked
```

`L_full`覆盖完整有效序列，`L_masked`只覆盖被Mask token；二者内部均为State MSE、Action MSE和contact BCE的存在项等权平均。

## 4. P0/P1/P2：先冻结decoder

统一FP32、micro-batch4、累积16、effective batch64、AdamW、weight decay0、clip1.0。prior encoder峰值LR `1e-4`，输入/embedding/latent heads峰值LR `3e-4`，warmup1000后cosine到`1e-6`。三个阶段在同一run连续执行，不重置optimizer、scheduler或训练流。

| 阶段 | 绝对step | Decoder | 目标 |
|---|---:|---|---|
| P0 | 1–10k | 全冻结 | `10 Lz + 1 Lrec` |
| P1 | 10k–25k | 全冻结 | `Lz 10→2`、`Lrec 1→5`线性变化 |
| P2 | 25k–50k | 全冻结 | `1 Lz + 10 Lrec` |

step0及每2k完整评测。只有P2连续三次同时通过latent和重建门禁才提前停止。正式`best_latent.pt`和`best_prior_fit.pt`只能由P2评测产生，避免decoder适配误用早期checkpoint。

## 5. 门禁与固定决策

重建门禁：held-out随机Mask与固定物理Mask必须同时满足global State/Action RMSE各`≤0.02`，每类worst-window State/Action各`≤0.04`，masked continuous p99 abs`≤0.08`，contact 100%，student latent的zero/cross-window/cross-motion替换ratio各`≥10`；max abs只报告。评测同时记录完整有效序列上的State/Action RMSE、p99、max abs、contact和重建loss，但这些完整序列指标只作诊断，不能替代被Mask位置门禁。

latent门禁：global/local标准化RMSE各`≤0.25`，cosine各`≥0.95`，student-teacher误差相对cross-window及cross-motion donor误差的比例各`≤0.10`。

固定决策：

| P2结果 | 唯一动作 |
|---|---|
| latent FAIL | 停止；不能用decoder改动掩盖prior失败 |
| latent PASS、重建 PASS | 冻结为KL=0基线，进入KL三路径实现 |
| latent PASS、重建 FAIL | 从P2 `best_latent.pt`启动D1 |

full-both conditional prior只报告，不参与任何PASS。无信息输入对应一个确定性代表latent，不可能恢复每个window的唯一答案。

## 6. D1/D2：只在严格触发后适配decoder

D1最多12k，只解冻global/local memory projection、global/local FiLM projection、FiLM输出projection，以及8层cross-attention与其LayerNorm；conditional prior继续训练。prior LR `1e-5`，latent接口LR `5e-6`，warmup250，最低`1e-6`。目标为`Lz + 10Lrec + 20Lkeep`。

D1未通过时，只有latent仍PASS、teacher保持PASS、最后三次fit score中位数相对step0改善至少20%、且D1最终score`≤1.5`，才启动D2。D2最多8k，额外解冻decoder self-attention、FFN、query/type embedding、final norm和输出头；prior/接口/其他decoder LR分别为`5e-6/3e-6/1e-6`，目标为`Lz + 10Lrec + 30Lkeep`。

`Lkeep`同时约束teacher latent经过当前decoder仍重建真值，并与不可修改的H50-A reference输出接近。每次适配评测还必须满足：teacher-path各误差不超过`min(H50-A step34000×1.05, fit阈值)`，当前/reference输出State/Action RMSE各`≤2e-3`、p99差`≤1e-2`、contact完全一致，teacher latent replacement ratio各`≥10`。任一评测失败立即拒绝该适配run。D2后不再扩大解冻范围或延长训练。

## 7. Ubuntu执行与回传

命令不包含Git操作，默认代码已由用户预先同步：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
export CVAE_POSTERIOR_PRIOR_SOURCE_RUN=/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_continue15k_20260909_230256

unset CVAE_CONFIG CVAE_RUN_DIR CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_WARM_START
unset CVAE_POSTERIOR_PRIOR_INIT_RUN

bash ./cvae_repro.sh posterior-hierarchical-prior-smoke
# 审核smoke工程合同后，重新从H50-A启动正式训练：
bash ./cvae_repro.sh posterior-hierarchical-prior-train
```

只有正式训练summary明确输出`RUN_CONTROLLED_DECODER_ADAPTATION_D1`且存在execution、latent-alignment和`cvae.failed` marker时才执行：

```bash
export CVAE_POSTERIOR_PRIOR_INIT_RUN=/home/helloworld/bly/runs/<正式CPD-train-run>
bash ./cvae_repro.sh posterior-hierarchical-prior-decoder-adapt
```

每个run结束后回传：

```bash
RUN=/home/helloworld/bly/runs/<实际run>
jq '{mode,execution_pass,smoke,quality_pass,latent_alignment_pass,reconstruction_pass,
     completed_optimizer_steps,last_training_phase,best_optimizer_step,
     best_latent_optimizer_step,best_latent_score,unique_next_step,
     initial:.initial_evaluation,last_three:.last_three_evaluations,
     checkpoint_readback}' \
  "$RUN/manifests/posterior_conditional_prior_summary.json"
cat "$RUN/manifests/source_commit.txt"
find "$RUN/markers" -maxdepth 1 -type f -printf '%f\n' | sort
ls -lh "$RUN/checkpoints"
```

最终通过只能声明：在已见32-motion、T64窗口上，Mask后的完整Token条件可经确定性conditional prior预测层级latent，并在未训练随机Mask bank上补全。它仍不证明未见motion泛化、随机采样质量或全部Mask组合的数学完备性。
