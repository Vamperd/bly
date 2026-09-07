# State–Action Posterior 容量实验：精简计划与结果台账

最后更新：2026-09-08

本文只记录具有研究意义的正式训练、正式比较、只读诊断和当前唯一下一步。两步 smoke 统一视为“开机自检”，不作为模型质量证据。当前阶段的完整执行合同见 [Next.md](Next.md)，面向读者的通俗说明见 [explain.md](explain.md)。

## 1. 当前状态与唯一下一步

当前已确认：

- F4G direct-output smoke 已完成；两步误差没有实验意义。
- 正式 F4G 已在1,504个T64窗口、12,032个fixtures上跑满5k，但`fit`质量FAIL。所有主要指标持续单调改善，说明索引、Mask和loss确实产生有效梯度；失败不能归因于encoder、latent或decoder。
- 5k只让每个fixture平均被采样约106次。答案表从零初始化，而State存在绝对值约42的归一化目标；当前LR累计位移不足，最坏State仍主导门禁。因此该run更直接反映“独立查表参数的稀疏优化不足”，尚不能据此判定evaluator上限不可达。
- F4G-O已在相同1,504个T64窗口、12,032个fixtures上以0 optimizer step取得`quality_pass=true`、`best_fit_score=1.0`。这证明数据身份、Mask target、loss和evaluator存在解析可达解；原F4G失败被定位为稀疏独立查表优化不足。
- H38工程smoke已在前2个window上完成2步前向、反向、评测和checkpoint链路；`quality_pass=false`与score不构成容量结论。
- H38、H50、H38-A/B/R 均尚未产生质量结果；不能写成已通过或已失败。
- conditional prior、latent 随机采样和 KL 尚未实现到当前 H38 路线；不能声称项目已经进入完整 CVAE 阶段。

当前唯一下一步：从随机初始化正式运行H38-A 30k上限。它只使用full-both posterior autoencoding，每1k完整评测，连续3次fit PASS即可提前停止；不得加载smoke checkpoint。

当前Windows文档与F4G smoke报告语义修复提交为`2feab9687ee8f91d48cb9425fb4c28ed697f8bde`；正式Ubuntu run仍以其自身`source_commit.txt`为准。

固定推进顺序：

```text
H38-A 完整序列重建（30k上限）
→ H38-B 固定物理 Mask
→ H38-R 随机物理 Mask
→ 冻结 KL=0 基线
→ posterior mean / posterior sample / conditional prior sample 三路径 KL 对照
```

若 F4G-O 通过而 H38-A 失败，只允许一次 H50-A 复核；不再建立更多参数量阶梯。

## 2. 固定研究合同

### 2.1 我们正在验证什么

当前阶段只验证 posterior reconstruction capacity：posterior encoder 可以读取未遮挡的完整真值序列，并把信息编码成 latent；decoder 再结合 Mask 后的条件重建目标。

因此：

- 通过可以证明模型能记住并解码已见序列。
- 通过不能证明只看 Mask 后的序列也能补全，因为 posterior 仍读取了被遮挡真值。
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
| H38 hierarchical | 独立 posterior/condition encoder + global/local latent + cross-attention/FiLM decoder | 37,574,883 | 当前目标模型，处理32 motion、T64 |
| H50 fallback | H38 同结构，宽度扩大到448 | 51,005,283 | 仅在 F4G-O PASS、H38-A FAIL 时复核一次 |

H38 的 latent 为一个 256 维 global code 加 16 个 128 维 local code；每个 local code覆盖4个 transition。decoder 的每一层都能读取条件和17个 latent token，并再次接收 global/local FiLM 条件，避免所有时序细节只通过一个输入 token 传播。

### 2.4 Mask 合同

早期纯容量实验使用10类固定 Mask，包括完整 State、完整 Action、完整 State+Action以及 element/time/feature/semantic Mask。这些 Mask 只用于检验记忆和查询能力，不要求条件在物理上唯一可辨识。

当前 T64 路线只使用8类物理结构 Mask：

| Mask | 可见信息与目标 |
|---|---|
| State gap 4/16 | 遮挡连续 State，保留两侧边界和对应 Action |
| State rollout | 只保留 S0 和全部 Action，预测后续 State |
| Action gap 4/16 | 遮挡连续 Action，保留完整 State 和缺口前历史 |
| Full Action | 完整 State 可见，遮挡全部 Action |
| Joint gap 2/8 | 同时遮挡短 Action 段和内部 State，保留前后 State 边界 |

H38-R 使用相同物理语义动态训练，并在每个已见窗口的16个未参与训练的固定 Mask 上评测。它仍不是未见 motion 泛化实验。

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
| `fit` | global RMSE≤`1e-2`；每类 worst-window RMSE≤`2e-2` | p99 abs≤`5e-2` | contact 100%；H38 latent ratio≥10 | 当前规模推进 |
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
| H38-A | 32 motion、T64、full-both posterior autoencoding | 最多30k，每1k评测 | PENDING | PASS进H38-B；FAIL仅允许H50-A |
| H38-B | 从A的best_fit model-only初始化，8类固定物理Mask | 最多60k，每2k评测 | PENDING | H38-R |
| H38-R | 从B初始化，动态物理Mask训练，固定held-out Mask评测 | 30k，每2k评测 | PENDING | 冻结KL=0基线，设计KL三路径 |
| H50-A | 与H38同结构但约51M | 仅一次20k | 未授权 | PASS后以H50继续B/R；FAIL停止扩模 |

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
| H38正式 | 尚未运行 |

## 4. 工程验收摘要

S0、S25、F4B A/B/C、F4E、F4F G8/T129、F4G及H38均执行过对应smoke。H38 smoke run为`/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h38_autoencode_smoke_20260908_025405`，完成2 step并返回execution PASS；`quality_pass=false`是预期语义。有效smoke只证明相关数据读取、真实CUDA前向/反向、短训练、完整评测、checkpoint读回和marker链路能够运行，不提供模型容量结论。

历史上发现并修复了三类协议/报告问题：F1的fixed训练与评测Mask seed不一致；F4E smoke误把“质量门禁不适用”报告成根因失败；F4F G8 smoke最初使用了错误的学习率配置键。它们均已修复或隔离，不得作为模型优劣证据。

Windows代码READY或测试PASS只表示接口和静态合同通过，不写入正式实验结果表。F4G-O与posterior相关78项测试通过；全量131项中128项通过，另3项仍只是Windows缺少既有`h5py`的导入限制。当前H38专项测试覆盖参数量、T64 token对齐、16个local chunk、posterior跨Mask一致、condition真值隔离、cross-attention/FiLM梯度、donor替换及checkpoint读回；真实质量仍必须由Ubuntu正式run决定。

## 5. 当前执行与结果回填

### 5.1 H38-A 正式30k命令

命令不包含Git操作；代码同步由用户预先完成。

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
export CVAE_POSTERIOR_DIRECT_OUTPUT_RUN=/home/helloworld/bly/runs/cvae_posterior_direct_output_oracle_f4go_t64_20260908_023727
export CVAE_POSTERIOR_HIERARCHICAL_PROFILE=H38

unset CVAE_CONFIG CVAE_RUN_DIR CVAE_INIT_CHECKPOINT
unset CVAE_POSTERIOR_WARM_START
unset CVAE_POSTERIOR_HIERARCHICAL_INIT_CHECKPOINT

test -f "$CVAE_POSTERIOR_DIRECT_OUTPUT_RUN/markers/cvae_posterior_direct_output_oracle.ok"
test -f "$CVAE_POSTERIOR_DIRECT_OUTPUT_RUN/markers/cvae_posterior_direct_output_fit.ok"

bash ./cvae_repro.sh posterior-hierarchical-t64-autoencode
```

### 5.2 F4G结果与F4G-O固定决策

正式F4G与F4G-O run：

```text
F4G:   /home/helloworld/bly/runs/cvae_posterior_direct_output_f4g_t64_20260908_013241
F4G-O: /home/helloworld/bly/runs/cvae_posterior_direct_output_oracle_f4go_t64_20260908_023727
```

它使用1,504个window、12,032个fixtures和9,634,624个独立输出参数。step0→5000的global State RMSE从`0.9733`降到`0.1686`，Action从`0.9922`降到`0.07982`，p99从`3.3519`降到`0.35897`，contact达到100%；worst State仍为`2.2082`并控制`110.4097`分数。曲线持续改善而非发散，checkpoint读回通过。

这个结果证明梯度路径有效，但不能证明evaluator错误。按`5000×256/12032`计算，每个fixture平均只被采样约106.4次；独立答案表没有跨window参数共享，和H38每个batch都会更新共享权重的优化条件不同，因此禁止把F4G失败外推成H38表达能力失败。

F4G-O随后以0 optimizer step获得`quality_pass=true`和`best_fit_score=1.0`，正式证明相同数据、窗口、8类Mask、loss与evaluator存在解析可达解。其完整source commit、strict/exact字段和marker列表尚未回传，不得在台账中猜测；H38启动脚本会强制检查oracle与fit两个marker。

| F4G-O结果 | 事实结论 | 唯一下一步 |
|---|---|---|
| 三次解析评测均PASS且hash一致 | loss、Mask和evaluator在32 motion、T64上有零误差可达解；原F4G失败属于稀疏查表优化 | 设置F4G-O run路径并执行H38 smoke |
| 真值逐位复制后仍FAIL | evaluator、Mask target或window身份存在矛盾 | 停止H38并修复协议 |
| 工程失败 | 不能形成模型或目标函数结论 | 修复工程问题后重新smoke/正式run |

### 5.3 后续 KL 三路径边界

只有H38/H50-R正式通过并冻结KL=0基线后，才实现：

| 路径 | latent来源 | 是否读取被遮挡真值 | 作用 |
|---|---|---|---|
| Posterior mean | 完整序列encoder的均值 | 是 | 确定性最佳重建基线 |
| Posterior sample | posterior均值和方差重参数采样 | 是 | 检查采样噪声代价 |
| Conditional prior sample | Mask后序列产生均值/方差再采样 | 否 | 检查部署时真实条件生成能力 |

三条路径必须使用同一窗口、同一Mask和配对随机噪声。KL是否合适要同时看posterior mean是否保持、posterior sample退化和prior sample差距，不能只看KL数值。

### 5.4 每个正式run的精简回填模板

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
