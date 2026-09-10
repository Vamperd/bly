# State–Action Posterior 容量实验：精简计划与结果台账

最后更新：2026-09-11

本文只记录具有研究意义的正式训练、正式比较、只读诊断和当前唯一下一步。两步 smoke 统一视为“开机自检”，不作为模型质量证据。当前阶段的完整执行合同见 [Next.md](Next.md)，面向读者的通俗说明见 [explain.md](explain.md)。

## 1. 当前状态与唯一下一步

当前已确认：

- F4G direct-output smoke 已完成；两步误差没有实验意义。
- 正式 F4G 已在1,504个T64窗口、12,032个fixtures上跑满5k，但`fit`质量FAIL。所有主要指标持续单调改善，说明索引、Mask和loss确实产生有效梯度；失败不能归因于encoder、latent或decoder。
- 5k只让每个fixture平均被采样约106次。答案表从零初始化，而State存在绝对值约42的归一化目标；当前LR累计位移不足，最坏State仍主导门禁。因此该run更直接反映“独立查表参数的稀疏优化不足”，尚不能据此判定evaluator上限不可达。
- F4G-O已在相同1,504个T64窗口、12,032个fixtures上以0 optimizer step取得`quality_pass=true`、`best_fit_score=1.0`。这证明数据身份、Mask target、loss和evaluator存在解析可达解；原F4G失败被定位为稀疏独立查表优化不足。
- H38工程smoke已在前2个window上完成2步前向、反向、评测和checkpoint链路；`quality_pass=false`与score不构成容量结论。
- H38-A已在32 motion、T64上从随机初始化跑满30k，`quality_pass=false`、best fit score为`2.4948489`。最佳global State/Action RMSE为`0.024948/0.017838`，worst-window为`0.041763/0.025360`，p99/max abs为`0.075243/0.784172`，contact 100%，zero/cross-window/cross-motion latent ratio为`46.69/50.64/57.85`。
- 新fit门禁现固定为global State/Action≤`0.02`、每类worst-window State/Action≤`0.04`、p99≤`0.08`、max abs只报告、contact 100%、latent ratio≥10。历史summary与marker保持原样，仅作事后重算。
- H38-A按新门禁仍FAIL，score `1.24742`由global State控制；worst State也为阈值`1.044`倍，p99已通过。Action、contact和latent依赖通过。
- H38-B已从H38-A的`best_fit.pt`做model-only初始化，在1,504个T64窗口、12,032个固定物理Mask fixture上跑满60k；execution PASS但质量FAIL，最佳点仍为最后step60000，fit score为`2.4990567`。
- H38-B最佳global State/Action RMSE为`0.024991/0.016406`，worst-window为`0.047077/0.025066`，p99/max abs为`0.081572/1.611736`，contact 100%，zero/cross-window/cross-motion latent ratio为`18.83/21.42/24.47`。官方门禁由global State的2.499倍控制。
- H38-B按新门禁仍FAIL，score为`1.24953`并由global State控制；worst State和p99分别为新阈值的`1.177/1.020`倍。Action、contact和三种latent依赖通过。
- H38-B相对H38-A的global Action改善约8.0%，但global State几乎不变，worst State和p99分别恶化约12.7%和8.4%。A与B的Mask分布不同，因此该比例只作结构诊断，不能当作严格配对优劣。B中`state_rollout`最难，worst State `0.047077`、max abs `1.611736`。
- H50-A已从随机初始化训练30k，execution PASS但质量FAIL；run为`/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_20260909_120934`，源码`fd7928f26b1a12dfa6e01218c7defd9d5ffe2166`。最佳点是step30000，fit score `1.03707`；global State/Action为`0.020202/0.015074`，worst State/Action为`0.041483/0.022078`，p99/max为`0.060677/0.715894`，contact 100%，三种latent ratio为`56.32/62.06/70.89`。
- H50-A只差global State 1.01%和worst State 3.71%；28k/29k/30k score为`1.06244→1.04331→1.03707`，所有主要连续指标仍改善。相对同任务H38-A，H50把global State/Action和p99改善约19.0%/15.5%/19.4%，但worst State只改善0.67%；最差feature全部落在29维joint velocity区间。这支持了一次受控尾段续训，不证明加宽已经解决最坏窗口。
- H50-A尾段续训已在run `/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_continue15k_20260909_230256`完成，源码`bf6d14c3848ffc1c544a56195cf07d3a2188c863`。它从原H50-A `last.pt`恢复模型与AdamW，在绝对step 34000因step32000/33000/34000连续三次fit PASS提前结束；`quality_pass=true`，但`strict_memory_pass=false`且`legacy_exact_pass=false`。
- `best_fit.pt`停留在首次PASS的step32000：global State/Action `0.019671/0.014758`、worst State/Action `0.039847/0.021575`、p99/max abs `0.058849/0.693393`、contact 100%，zero/cross-window/cross-motion ratio `57.84/63.67/72.73`。step34000的对应连续指标进一步改善到`0.019080/0.014412`、`0.037716/0.021309`和p99 `0.057103`；但fit score通过后下限为1.0，保存条件只接受更低score，因此没有覆盖`best_fit.pt`。当前预注册B合同仍从step32000的`best_fit.pt`做model-only初始化。
- H50-B已从上述`best_fit.pt`做model-only初始化，在run `/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_fixed_20260910_012003`跑满60k，源码`bf6d14c3848ffc1c544a56195cf07d3a2188c863`。execution完成但`quality_pass=false`，best位于step60000，fit score `1.0192181`；唯一失败项是global State `0.020384`，比`0.02`高1.92%。global Action `0.013908`、worst State/Action `0.034900/0.021335`、p99 `0.066332`、contact 100%及latent ratio `19.48/22.66/25.88`均通过。
- B在52k→60k的score为`1.72360→1.21865→1.03544→1.02523→1.01922`，所有主连续指标在末段持续改善；但下降速度随学习率衰减而放缓。8类Mask中最难的是`state_rollout`，worst State `0.034900`，已经低于门槛。
- B没有保持A的full-both能力：同一checkpoint在B step0的full-both State/Action为`0.019671/0.014758`，到step60000变为`0.060442/0.016711`，max abs从`0.693393`恶化到`21.894020`，contact从100%降到`99.9519%`。52k→60k的full-both State仅`0.060971→0.060442`，处于明显平台。由于B训练bank不含full-both且重启高学习率，这构成condition适配伴随的A能力遗忘。
- H50-R路线已取消：H50-B没有质量PASS，且直接condition融合会破坏H50-A的canonical解码能力。
- H50-CRA在正式训练前被设计复审判定为错误方向并取消。它会让完整posterior latent始终承担答案、condition只做decoder侧修正，不能回答“Mask条件能否独立预测latent”。CRA没有Ubuntu正式结果，不形成模型结论。
- H50-CPD代码现已在Windows实现：Mask后的可见序列只进入新的conditional prior encoder，预测一个global和16个local latent；冻结的H50-A decoder只接收这些latent和有效长度，不直接读取可见值或Mask。当前仍为确定性mean、`KL=0`，尚无Ubuntu工程或质量结果。

H50-CPD固定读取H50-A续训run的step34000 `last.pt`。51,005,283参数的H50-A teacher/base与decoder先全部冻结；新增6层、宽448的conditional prior共14,779,456参数，总计65,784,739参数。teacher以完整序列产生canonical latent作为监督，student以Mask序列预测同拓扑latent，再由严格canonical decoder接口输出完整序列。

当前唯一下一步：同步代码后执行`posterior-hierarchical-prior-smoke`。smoke只验证H50-A源准入、teacher缓存、Mask真值隔离、strict decoder接口、CUDA、梯度和checkpoint；审核通过后才从H50-A重新运行50k上限的P0/P1/P2正式蒸馏。

F4G smoke报告语义修复的历史提交为`2feab9687ee8f91d48cb9425fb4c28ed697f8bde`。H50-CPD实现当前位于Windows工作树；正式Ubuntu run仍以用户同步后由run写入的`source_commit.txt`为准，不预填提交号。

固定推进顺序：

```text
H38-A 完整序列重建（已FAIL，仅作latent压力诊断）
→ H38-B 固定物理 Mask（已FAIL）
→ H50-A 30k参数规模复核（近门槛FAIL，末段仍改善）
→ H50-A受控续训（step 34000三连fit PASS）
→ H50-B固定Mask（60k；仅global State超1.92%，但full-both明显遗忘）
→ H50-CRA（CANCELLED/SUPERSEDED，未正式运行）
→ H50-CPD smoke（READY，待Ubuntu）
→ P0/P1/P2冻结decoder蒸馏（最多50k）
  ├─ latent FAIL：停止，调查conditional prior
  ├─ latent与重建PASS：冻结KL=0基线
  └─ latent PASS但重建FAIL：D1受控latent接口适配；严格满足条件才允许D2
→ posterior mean / posterior sample / conditional prior sample 三路径KL对照
```

H38-A不再是H38-B的硬门槛：A的全遮挡任务信息更少，只用于latent压力诊断。H50-A显示约32%的参数增长显著降低平均State与p99，但最差State窗口几乎未变。15k续训是针对“只差3.71%且末段仍单调改善”的一次预注册例外；不允许再追加第二次续训或H64阶梯。

## 2. 固定研究合同

### 2.1 我们正在验证什么

早期阶段验证posterior reconstruction capacity：posterior encoder读取完整真值并编码latent。当前H50-CPD首次单独验证conditional prior：student只能读取Mask后的可见序列，并必须先预测canonical latent，再通过不接收condition的decoder重建。

因此：

- 历史posterior通过只证明模型能记住并解码已见序列，不能证明只看Mask条件也能补全。
- CPD通过才支持“已见序列上的Mask条件能够预测确定性层级latent并完成补全”；teacher真值只用于训练监督，不进入student前向输出。
- 通过不能证明未见 motion 泛化。
- `KL=0` 时不要求 latent 接近某个可随机采样的分布，也不能把随机 latent 当作有效生成结果。
- 只有后续 conditional prior 在不读取被遮挡真值时通过，才可以讨论真实的条件生成能力。

### 2.2 数据与序列

训练数据固定来自：

```text
/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
```

该子集包含 32 个 motion，每个 motion 有 8 个 completed variant，共 256 个 episode；全部属于记忆基准，不是独立验证集。

序列关系固定为：

```text
S0, A0, S1, A1, ..., A(T-1), ST
```

其中 `State_t + Action_t → State_(t+1)`。State 为 70 维：68 个连续物理量和 2 个脚接触标签；Action 为 29 维关节目标。数据频率为 50 Hz，所以 T64 约覆盖 1.28 秒，T128 约覆盖 2.56 秒。

历史 T16/T128 实验使用各自固定窗口；当前 H38 路线固定为 T64、stride 64、`random_crop=false`。

### 2.3 模型代际

| 代际 | 核心结构 | 参数/记忆形式 | 用途 |
|---|---|---:|---|
| 最简 posterior Transformer | 共享双向 encoder + 单 global latent + 双向 decoder | 6.7M，latent 256 | 验证最小结构能否记忆小数据 |
| 25M posterior Transformer | encoder 6层、decoder 8层、宽度384 | 25,453,411，单 latent 256 | 检验扩大统一模型后的容量 |
| F4C gated decoder | 在原 decoder 每层重复加入 global latent | 25,456,483 | 检查 latent 只注入一次是否是问题 |
| F4E auto-decoder | 每个 window 直接学习一个共享256维 code，绕过 encoder | 80×256 code + 原 decoder | 区分 encoder 问题与 code/decoder 问题 |
| F4F-G8 | 每个 window 学8个256维 memory token | 每窗口2,048个 code 标量 | 检查多个全局 token 是否更易广播信息 |
| F4F-T129 | 每个 window 的每个时间位置学习16维 code | 每窗口2,064个 code 标量 | 检查时间局部注入是否更合适 |
| F4G / F4G-O direct output | 每个 window 对应完整 State/Action/contact 输出；F4G梯度学习，F4G-O直接复制真值 | 无 encoder、latent、decoder | 分开验证稀疏查表优化和解析 evaluator 上限 |
| H38 hierarchical | 独立 posterior/condition encoder + global/local latent + cross-attention/FiLM decoder | 37,574,883 | 历史T64层级容量实验 |
| H50-A | H38同结构，宽度扩大到448 | 51,005,283 | 已通过canonical posterior autoencoding fit |
| H50-CPD | H50-A teacher/base + 独立conditional prior；decoder无condition旁路 | 65,784,739 | 当前目标：Mask条件预测canonical层级latent |

H38 的 latent 为一个 256 维 global code 加 16 个 128 维 local code；每个 local code覆盖4个 transition。decoder 的每一层都能读取条件和17个 latent token，并再次接收 global/local FiLM 条件，避免所有时序细节只通过一个输入 token 传播。

### 2.4 Mask 合同

早期纯容量实验使用10类固定 Mask，包括完整 State、完整 Action、完整 State+Action以及 element/time/feature/semantic Mask。这些 Mask 只用于检验记忆和查询能力，不要求条件在物理上唯一可辨识。

H38/H50-B使用8类物理结构 Mask：

| Mask | 可见信息与目标 |
|---|---|
| State gap 4/16 | 遮挡连续 State，保留两侧边界和对应 Action |
| State rollout | 只保留 S0 和全部 Action，预测后续 State |
| Action gap 4/16 | 遮挡连续 Action，保留完整 State 和缺口前历史 |
| Full Action | 完整 State 可见，遮挡全部 Action |
| Joint gap 2/8 | 同时遮挡短 Action 段和内部 State，保留前后 State 边界 |

当前CPD复用这8类Mask作保持检查，同时以完整Token随机Mask为主：训练覆盖State-only、Action-only、Both的5%–95%随机比例、稀疏Token和full State/Action；独立评测bank覆盖10%/35%/65%/90%以及single/full域。它仍不是未见motion泛化实验，也不声称穷举全部Mask组合。

### 2.5 Loss、指标与门禁

训练 loss 固定为存在项等权平均：

```text
State continuous MSE + Action MSE + contact BCE
```

指标含义：

- RMSE：整体误差的平方均值开根号，越小越好。
- max abs：所有连续输出中的最大单点绝对误差。
- p99 abs：99%的连续误差都不超过的值，比单个最大值更稳定。
- contact accuracy：左右脚接触分类正确率。
- latent dependence：正确 latent 相比置零或换成其他窗口 latent 能改善多少倍；至少10倍才认为模型确实依赖 latent。

三套门禁互不替代：

| 门禁 | State/Action要求 | 尾部要求 | contact/latent | 用途 |
|---|---|---|---|---|
| `fit` | global RMSE≤`2e-2`；每类 worst-window RMSE≤`4e-2` | p99 abs≤`8e-2`；max只报告 | contact 100%；H38/H50 latent ratio≥10 | 当前规模推进 |
| `strict_memory` | worst RMSE≤`1e-2` | max abs≤`1e-2` | contact 100%；latent ratio≥10 | 更严格诊断 |
| `legacy_exact` | worst RMSE≤`1e-4` | max abs≤`1e-3` | contact 100%；latent ratio≥10 | 历史近无损标准 |

`progression` 或 `fit` PASS 只能表述为“达到推进精度”，不能称为完美拟合。execution marker只说明程序完整执行，不说明质量通过。

## 3. 正式实验结果

### 3.1 6.7M 最简模型：先证明小规模可记忆

| ID | 数据/训练 | 关键结果 | 状态与意义 | 正式 run |
|---|---|---|---|---|
| F1 | 1 motion、T16、全部144窗口、40k | State/Action `0.05842/0.02495`，max `0.75203` | `INVALID`：训练与评测的7类partial Mask坐标不同，只能视为新Mask诊断，不能证明fixed记忆失败 | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_20260831_114833` |
| D1 | 1 motion、T16、单窗口、10 Mask、34k | State/Action `4.719e-5/8.075e-5`，max `2.255e-4`，contact 100%，zero/swapped `10342.86/54.62` | `PASS legacy_exact`：证明单窗口可以被单global latent模型近乎无损记住 | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_w1_20260831_150654` |
| F1R | 1 motion、T16、144窗口、40k | State/Action `0.003641/0.002903`，max `0.017851` | `FAIL exact`：平均已很低，但最差单点仍未达到旧严格门禁 | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_20260831_202012` |
| W4 | 1 motion、T16、4窗口、40k | State/Action `1.610e-4/1.194e-4`，max `6.158e-4` | `FAIL exact / PASS progression诊断`：非常接近exact，说明暴露次数对小数据记忆很重要 | run路径未完整回传，保留为待补充 |
| P1 | 1 motion、T16、144窗口、最多100k | State/Action `0.002243/0.001460`，max `0.009938`，zero `610.99` | `PASS progression`：6.7M模型能在较长训练后达到推进精度，但不是完美拟合 | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_s100000_gprogression_20260901_105145` |

这一阶段证明：模型不是完全没有记忆能力；随着窗口数增加，要同时压低所有位置的误差会明显变难。

### 3.2 25.45M 单 global latent：扩大数据后的瓶颈

| ID | 数据/训练 | 关键结果 | 状态与意义 | 正式 run |
|---|---|---|---|---|
| L128 | 1 motion、T128、24窗口、240 fixtures、93.5k | State/Action `0.002302/0.001533`，max `0.009942`，zero `469.56` | `PASS progression`：较大模型能记住一个motion的长窗口 | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t128_25m_s100000_gprogression_20260901_173153` |
| F128 | 32 motion、T128、816窗口、200k | State/Action `0.261443/0.273665`，max `9.644091`，contact `99.9958%`，zero/swapped `12.06/7.91` | `FAIL`：从1个motion直接扩到32个后明显失效；不能归因于训练步数太少 | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m32_t128_25m_s200000_gprogression_20260902_235140` |
| F4D | 4 motion、T128、80窗口、100k | worst State/Action `0.023021/0.015263`，max `0.215620`，global约`0.009149/0.008118` | `FAIL`：平均误差已过`1e-2`，但大量局部位置仍不够准 | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425` |
| F4A | 只读重评 F4D 的80窗口×10 Mask | `21.077%`连续目标超过`1e-2`；全部fixtures都含超阈值元素 | `PASS execution`：支持“广泛的时序细节误差”，不是少数离群点造成 | `/home/helloworld/bly/runs/cvae_posterior_capacity_tail_diagnostic_f4a_20260905_200807` |

这一阶段支持：25M参数本身不足以保证长序列、多motion的均匀精确重建；主要问题出现在大量时序细节，而不是contact分类。

### 3.3 损失和注入方式对照：没有找到简单修补方案

所有F4B/F4C实验固定使用4 motion、T128、80窗口、800 fixtures，从同一个F4D checkpoint开始。

| 实验 | 改动 | worst/global State | worst/global Action | max / 超`1e-2`比例 | 结论 |
|---|---|---:|---:|---:|---|
| A | 继续普通MSE | `0.015999/0.008425` | `0.014238/0.007545` | `0.137220 / 18.456%` | 10k后仍FAIL |
| B | 每个域混合普通MSE与最差20%误差 | `0.016398/0.008444` | `0.014550/0.007629` | `0.130564 / 20.718%` | max略降，但整体没有受保护的显著改善 |
| C | decoder 8层逐层加入global latent gate | `0.019039/0.008196` | `0.013908/0.007330` | `0.171898 / 17.379%` | 部分平均项改善，最差误差恶化，仍FAIL |

正式run：

```text
A: /home/helloworld/bly/runs/cvae_posterior_capacity_ab_a_seed20260830_20260906_114634
B: /home/helloworld/bly/runs/cvae_posterior_capacity_ab_b_seed20260830_20260906_203844
C: /home/helloworld/bly/runs/cvae_posterior_capacity_ab_c_seed20260830_20260907_004301
A/B compare: /home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260906_235429
A/B/C compare: /home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260907_102553
```

最终比较的13项配对身份检查通过；B/C均未达到受保护的20%改善条件，正式决定为 `STOP_LOSS_LATENT_SEED_SEARCH`。这说明当前证据不支持继续搜索相似loss、gate或随机seed。

### 3.4 绕过 encoder 的容量诊断

#### F4E：每窗口一个256维code

F4E允许按window identity查表，但同一window的10种Mask必须共享一个code；这只是诊断，不是可部署模型。

| 阶段 | 训练参数 | 结果 |
|---|---|---|
| E1 | 只训练80个window code，5k | 未通过 |
| E2 | code与decoder联合适配，再15k | worst/global State `0.029776/0.008951`；Action `0.020474/0.007856`；max `0.209413`；`19.679%`元素超阈值 |

zero/cross-window/cross-motion code依赖为`95.65/94.85/119.93`，证明decoder确实在使用code；但E1/E2均未通过，所以尚无证据表明“单个256维共享code + 当前decoder”足以精确记住80个T128窗口。

正式run：`/home/helloworld/bly/runs/cvae_posterior_capacity_autodecoder_f4e_20260907_120414`

#### F4F：等code预算的全局与逐时间拓扑

| Arm | code布局 | worst/global State | worst/global Action | max / 超阈值比例 | code依赖 | 状态 |
|---|---|---:|---:|---:|---:|---|
| G8 | 8个全局memory token | `0.071671/0.018158` | `0.043046/0.013475` | `0.710758 / 39.456%` | `42.80/43.25/54.69` | FAIL |
| T129 | 129个逐时间16维code | `0.117395/0.040533` | `0.065137/0.025330` | `2.909683 / 53.122%` | `18.94/16.58/20.97` | FAIL |

两臂每window code预算只差0.775%，训练身份和18项比较合同全部一致。G8明显好于T129，但两者最后三个评测点都没有通过，因此不能选择G8继续推进，也不能声称“逐时间latent一定更好”。

正式run：

```text
G8: /home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_g8_20260907_184707
T129: /home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_t129_20260907_213123
compare: /home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_comparison_20260908_000708
```

正式结论为 `BOTH_FAIL_LATENT_TOPOLOGY_INSUFFICIENT`，因此停止继续延长F4F训练，转向F4G直接输出上限和新的T64层级模型。

### 3.5 当前 T64 路线

| ID | 数据与结构 | 预算/门禁 | 当前状态 | 通过后的唯一动作 |
|---|---|---|---|---|
| F4G | 32 motion、T64；1,504张独立答案表从零优化 | 5k，每250评测 | `FAIL quality`；best score `110.4097` | 不续训；执行F4G-O |
| F4G-O | 将1,504个window真值直接复制到共享答案表 | 0 optimizer step；完整bank重复评测3次 | `PASS fit`；best score `1.0` | H38 smoke |
| H38 smoke | 前2个window、层级37.57M模型 | 2 step，仅工程合同 | `PASS engineering` | H38-A |
| H38-A | 32 motion、T64、full-both posterior autoencoding | 30k，每1k评测 | `FAIL quality`；best global S/A `0.02495/0.01784`，worst S/A `0.04176/0.02536`，p99 `0.07524` | 无论质量结果均进入H38-B |
| H38-B | 从A的best checkpoint model-only初始化，8类固定物理Mask | 60k，每2k评测 | `FAIL quality`；best global S/A `0.02499/0.01641`，worst S/A `0.04708/0.02507`，p99 `0.08157` | 一次H50-A规模复核 |
| H38-R | 从同profile的B初始化，动态物理Mask训练，固定held-out Mask评测 | 30k，每2k评测 | BLOCKED；H38-B无fit marker | 仅B质量PASS后冻结KL=0基线 |
| H50-A复核 | H38同结构扩至51.01M；full-both、随机初始化 | 30k，每1k评测 | `FAIL quality`；score `1.03707`，仅global/worst State略超 | 受控续训15k |
| H50-A续训 | 恢复H50-A `last.pt`的模型和AdamW；低LR尾段重启 | 最多15k，每1k；三连PASS提前停 | `PASS fit`；best checkpoint step32000，step34000三连PASS后提前结束；strict/exact FAIL | H50-B |
| H50-B | 从续训`best_fit.pt` model-only初始化；8类固定物理Mask | 60k，每2k；三连PASS提前停 | `FAIL quality`；仅global State `0.020384`超1.92%，但full-both State退化到`0.060442` | 设计保留A能力的B修复，不进入R |
| H50-R | 从H50-B继续旧condition融合 | 原计划最多30k | `CANCELLED`；B会遗忘A且无fit marker | 不再执行 |
| H50-CRA | 冻结H50-A、decoder侧差分condition adapter | 未正式训练 | `CANCELLED/SUPERSEDED`；不能检验Mask条件能否预测latent | 删除训练入口，不形成结果 |
| H50-CPD | Mask序列→新conditional prior→global+16 local→H50-A canonical decoder | P0/P1/P2最多50k；必要时D1 12k、D2 8k | `CODE READY`，尚无Ubuntu结果 | smoke→正式蒸馏；按latent/重建双门禁决策 |

### 3.6 正式结果源码审计

source commit只用于复现实验，不用于替代run内的dataset、fixture和checkpoint hash。未回传项保持未知：

| 实验 | source commit |
|---|---|
| F1 | `fcdb4f8861e539e3ea364e578d6bc96ce7ebd9b0` |
| D1 | 未回传 |
| F1R / W4 | `b3aa63d9514cd8dd284e7f6091fc26877f57f021` |
| P1 / L128 | 未回传 |
| F128 / F4D | `6463b2ec960cda22c7ed70814a46a44e6804d4c0` |
| F4A | `2ff1ec95db72fed9db80d3b040cccc32b3f9703f` |
| F4B A/B及首次比较 | `c1ae5f79111bf61ddace32073df8122dbbefec95` |
| F4C正式 | `163be1f4c40c46bbd5c680b7ac1711e87f382e97` |
| A/B/C最终比较 | `8fbe327c487c66482ba4ece6912c8f76bab3720b` |
| F4E正式 | `cfb6735b54f56e49977665948397f80855987d74` |
| F4F G8正式 | `8012b972b5d842f3196586eb995c963fb6dda06d` |
| F4F T129及最终比较 | `6e7caed535721a5ee575b80eba138cb6e392152e` |
| F4G正式 | `2feab9687ee8f91d48cb9425fb4c28ed697f8bde` |
| F4G-O | 未回传 |
| H38 smoke | 未回传 |
| H38-A正式 | `a0a7f7e0efce25f1184fc522536c15eabe6e3b5b` |
| H38-B正式 | `c5932690f43b478fb05687ccf58e6834b0243a32` |
| H50-A正式 | `fd7928f26b1a12dfa6e01218c7defd9d5ffe2166` |
| H50-A续训 | `bf6d14c3848ffc1c544a56195cf07d3a2188c863` |
| H50-B | `bf6d14c3848ffc1c544a56195cf07d3a2188c863` |

## 4. 工程验收摘要

S0、S25、F4B A/B/C、F4E、F4F G8/T129、F4G及H38均执行过对应smoke。H38 smoke run为`/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h38_autoencode_smoke_20260908_025405`，完成2 step并返回execution PASS；`quality_pass=false`是预期语义。有效smoke只证明相关数据读取、真实CUDA前向/反向、短训练、完整评测、checkpoint读回和marker链路能够运行，不提供模型容量结论。

历史上发现并修复了三类协议/报告问题：F1的fixed训练与评测Mask seed不一致；F4E smoke误把“质量门禁不适用”报告成根因失败；F4F G8 smoke最初使用了错误的学习率配置键。它们均已修复或隔离，不得作为模型优劣证据。

H50续训首次启动在训练前被准入检查拦截：旧summary的末三点评测对象没有稳定携带step字段，step实际位于`metrics.jsonl`外层。现改为从源JSONL核对`28000/29000/30000`并校验其score与summary一致；该次没有执行optimizer step，不形成模型结论。

Windows代码READY或测试PASS只表示接口和静态合同通过，不写入正式实验结果表。CRA专项实现与入口已删除。CPD专项测试覆盖精确参数量、posterior复制和Mask列归零、canonical decoder隔离、masked真值隔离、完整Token约束、P0/P1/P2权重、D1/D2 allowlist与决策、随机Mask双seed、full-both报告、teacher latent donor及checkpoint读回；真实质量仍必须由Ubuntu正式run决定。

## 5. 当前执行与结果回填

### 5.0 H50-CPD当前入口

当前正式路线已由旧H50-R/CRA切换为H50-CPD。根本验收对象仍是完整State token、完整Action token或二者组合的随机Mask；不使用element/feature Mask。student可见信息只能先形成global+16 local latent，decoder没有condition旁路。

模型、Mask、P0/P1/P2、D1/D2、marker和Ubuntu命令的详细活动合同见[Next.md](Next.md)。目前仅允许执行CPD smoke；正式结果必须回填run路径、source commit、H50-A checkpoint hash、teacher cache hash、最后三次latent/重建门禁、full-both不可辨识性诊断、teacher保持、marker和唯一下一步。下文旧H50-B入口只保留为历史记录，不再执行。

### 5.1 H50-A续训结果与H50-B历史入口（已执行，不再使用）

H50-A尾段续训已经完成。run为：

```text
/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_continue15k_20260909_230256
```

它在绝对step34000因连续三次fit PASS提前结束，summary报告`quality_pass=true`；strict-memory和legacy-exact仍FAIL。历史H50-B从首次PASS的step32000 `best_fit.pt`初始化，没有修改A源run，并重置了优化器、调度器与训练随机序列。B没有产生`fixed_fit.ok`；旧H50-R从未启动，现已被CPD路线取代。CPD使用误差更低的step34000 `last.pt`，不再使用step32000 checkpoint。

历史H38/F4G正式结果已集中保留在第3节，不再重复执行命令或长日志。

### 5.2 后续 KL 三路径边界

只有H50-CPD的确定性conditional prior同时通过latent对齐、随机/固定Mask重建和teacher保持门禁后，才冻结KL=0基线并实现：

| 路径 | latent来源 | 是否读取被遮挡真值 | 作用 |
|---|---|---|---|
| Posterior mean | 完整序列encoder的均值 | 是 | 确定性最佳重建基线 |
| Posterior sample | posterior均值和方差重参数采样 | 是 | 检查采样噪声代价 |
| Conditional prior sample | Mask后序列产生均值/方差再采样 | 否 | 检查部署时真实条件生成能力 |

三条路径必须使用同一窗口、同一Mask和配对随机噪声。KL是否合适要同时看posterior mean是否保持、posterior sample退化和prior sample差距，不能只看KL数值。

### 5.3 每个正式run的精简回填模板

```markdown
### <日期> — <实验ID> <PASS|FAIL|INVALID|BLOCKED>

- Run/source：`<absolute run>`；`<source commit或未回传>`
- 合同：`<motion/window/Mask/model/初始化>`
- 执行：`<steps、是否提前停止、execution与quality marker>`
- 指标：`<global/worst State与Action、p99/max、contact、latent依赖>`
- 事实结论：`<证明、支持、尚无证据或不能证明的内容>`
- 唯一下一步：`<一个实验或一个工程修复>`
```

更新规则：

- smoke只更新第4节的一句话，不新增正式结果行。
- 正式训练、只读诊断和正式比较只在第3节对应表中更新一次，不再追加重复长日志。
- `RUNNING`只记录路径和最后step，不提前写质量结论。
- source commit或指标未回传时明确写“未回传”，不得推测。
- completed/execution marker和quality marker必须分开解释。
- 任何新结果都必须同时写清“能证明什么”和“不能证明什么”。
