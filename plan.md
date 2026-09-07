# 最简 Transformer CVAE Posterior 容量实验计划与结果台账

最后更新：2026-09-07
当前阶段：F4F显式比较已在run`/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_comparison_20260908_000708`完成，18项身份检查全PASS；G8/T129三个主比较点全部质量FAIL，固定决策为`BOTH_FAIL_LATENT_TOPOLOGY_INSUFFICIENT`，唯一下一步为`RUN_DIRECT_OUTPUT_MEMORY_CEILING_FOR_DECODER_AND_OBJECTIVE`。F4F只证明当前同预算拓扑、初始化、decoder与15k协议不足，不证明global/temporal latent理论上不可行。下一步先设计并实现直接输出记忆上限诊断，禁止追加F4F步数、32-motion、R128和KL。
本文是本轮 posterior-only 研究的概览、实验结果与后续决策的唯一台账；当前下一步的完整实施合同见[Next.md](Next.md)。每次实验结束后必须先更新本文，再启动下一项实验。

## 1. 研究问题、成功声明与边界

本轮只回答：单一 global latent 的纯 Transformer CVAE，在 posterior encoder 可以读取完整 State–Action 序列时，能否对小规模训练数据实现数值近零的 masked reconstruction。

模型目标为：

$$
q_\phi(z\mid X,M),\quad
p_\psi(z\mid X_{visible},M),\quad
p_\theta(X_{masked}\mid X_{visible},M,z)
$$

其中本轮训练和验收固定使用 `z = posterior_mean`，`KL beta=0`。即使全 State、全 Action或全序列被 Mask，posterior encoder 仍然读取完整真值。因此通过本实验只能声明：

- posterior encoder、global latent 与统一 decoder 具备有限训练集的无损记忆容量；
- 统一 reconstruction 路径可以处理固定和 held-out Mask；
- 不能声明 conditional prior 已学会补全；
- 不能声明模型真正依赖可见 State 推断 Action，或依赖可见 Action 推进 State；
- 不能声明对未见 motion、未见 episode 或部署数据具有泛化能力。

在 posterior capacity 全部通过之前，不恢复 KL、relation heads、rollout、RobotInfo、reference、auxiliary、cycle 或多任务损失。

本阶段的Mask是纯容量探针，不要求满足动力学可辨识性：允许任意element、feature、time、semantic、
全State、全Action或联合随机Mask。目标只是检验posterior读取完整序列后，global latent与decoder
能否无损记忆并按任意Mask查询缺失值；不能用这些Mask的物理不适定性解释capacity失败。

容量阶段通过后的KL实验回答另一个问题：在decoder始终接收同一masked condition时，posterior
均值、posterior重参数采样与只读取masked序列的conditional prior采样之间有多大质量差距，进而
判断当前KL权重是否在“保留重建能力”和“对齐posterior/prior”之间取得可用平衡。该阶段仍只在
已见32 motion上评测，不把结果表述为未见motion泛化。

### 1.1 2026-09-05独立审视：用受控改进实验检验原因

posterior mean、KL=0、固定数据和Mask的容量实验是合理的工程诊断。D1、P1、L128证明小规模记忆可行，
F128与F4D说明当前结构和优化组合扩展不足；这些结果尚不能确认某一个模块是根因，也不能证明最终
conditional CVAE的结构已经合理。当前不直接扩大统一主干、不增加多个latent、不同时改变多个模块。

T128有`129×68+128×29=12,484`个连续输出，经一个256维global latent重建。压缩并不在理论上阻止
有限样本记忆，但encoder汇总、latent组织与decoder使用latent的效率都值得检查。现有latent只作为
一个token进入decoder，后续优先用小规模、同checkpoint的条件注入对照检验优化通路。
[DiT原论文](https://arxiv.org/html/2212.09748v2)表明条件注入方式在其图像生成实验中影响结果；
这只提供设计动机，不是本项目有效性的证据，F4C也不等同于DiT或完整AdaLN。

平均MSE和最大误差具有相同的零误差最优解。平均RMSE达标而max abs失败，本身不能证明loss错误；
有限预算下困难元素的梯度权重可能影响收敛。旧F4B直接叠加CVaR还同时改变整体尺度、fixture权重和
连续量相对contact的权重，因此改为F4B-v2的归一化尾部混合。它减少简单尺度混杂，不保证梯度范数相同。

F4A中的`full_both/partial p95=1.851<3`仅表示未触发预设启发式判据，不能排除global latent问题。
joint velocity位于最差feature前列支持检查峰值衰减与时间偏移，尚不足以确认系统性平滑这一机制。
下一轮用固定案例的真值/预测/误差曲线补充证据，不提前下因果结论。

### 1.2 swapped-latent源码审计与历史解释修正

当前`evaluate_exact`通过`output.posterior_mean.flip(0)`生成swapped latent，而验证bank按每个window
连续排列10类Mask。T128的micro-batch为4，full-both位于slot 2：偶数window对应的donor为同窗口slot 1，
奇数window对应的donor为同窗口slot 5。故该指标实际测量同窗口、不同Mask的posterior latent替换，
不能据此评价跨窗口或跨motion的latent区分能力。历史6.7M实验还须按各自batch大小解释，单窗口D1
更不可能提供跨窗口置换证据。

全部历史swapped数值保留，统一标注为`legacy within-microbatch swap`。zero-latent依赖和正式门禁不变。
F4B-v2新增在全体80窗口相同full-both Mask下计算的跨窗口、跨motion donor诊断，记录可检查的身份映射；
不得覆盖旧指标。F128历史结论中由swapped ratio推断多窗口区分能力的部分，以本条解释为准。

## 2. 固定实验合同

### 2.1 数据与代码

| 项目 | 固定值 |
|---|---|
| Ubuntu 工作区 | `/home/helloworld/bly` |
| CVAE 项目 | `/home/helloworld/bly/state-action-cvae` |
| 数据集 | `/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506` |
| 数据规模 | 32 motion × 8 variant = 256 episode，全部 train-only |
| 数据读取 | HDF5、index、normalization 全部只读 |
| 模型 | `physics_posterior_transformer` |
| 当前主实验参数量 | 25,453,411（历史最简模型6,725,731保持兼容） |
| State / Action | 70 / 29维 |
| latent | 单个 global latent，256维 |
| 条件 | 仅 State、Action、逐特征 Mask、位置和 token type |
| 排除条件 | RobotInfo、reference、motion ID、action-before-window、dynamics context |
| seed | 历史入口沿用各run记录；F4B-v2/F4C/F4R固定fixture seed `20260830`，优化seed首轮`20260830`、复核`20260831` |
| 输出 | 仅 `/home/helloworld/bly/runs/<new_run_id>/` |

每个正式 run 前必须记录 Windows/Ubuntu 外层、SONIC、IsaacLab 的实际分支、HEAD 和状态。不得为了匹配本文自动 checkout、reset 或恢复历史 IsaacLab 修改。Ubuntu 只同步和执行，不手工修改源码。

### 2.2 KL=0容量阶段模型与优化

下表保留L128/F128/R128的历史规模合同。当前F4B-v2/F4C/F4R采用[Next.md](Next.md)的独立短程合同：
每支10k、峰值LR `3e-5`、250步warmup后cosine至`1e-6`、每1k验收，正式配对训练不提前成功停止。
不得把下表的`3e-4`、500步warmup或提前停止规则误用于本次A/B。

| 项目 | 固定值 |
|---|---:|
| `d_model` | 384 |
| encoder / decoder | 6 / 8层，共享 encoder |
| heads / FFN | 8 / 1536 |
| dropout / weight decay | 0 / 0 |
| posterior latent | `posterior_mean`，不采样 |
| KL beta / free bits | 0 / 0 |
| optimizer | AdamW |
| learning rate | `3e-4` |
| schedule | 500-step warmup，随后 cosine 到 `1e-6` |
| gradient clip | 1.0 |
| max optimizer steps | L128 100,000；F128 200,000；R128 50,000 |
| 完整验收 | L128每250 step；F128/R128每2,500 step |
| early success | 连续3次活动门禁完整验收全部通过 |

effective batch固定为64：窗口不超过32时为`16×4`，窗口64时为`8×8`，窗口128时为`4×16`。L128从随机初始化开始；F128与R128通过`CVAE_POSTERIOR_WARM_START`只加载前一级`last.pt`的模型权重，严格校验dataset manifest hash、模型结构、门禁与规模扩展方向，并重置optimizer、scheduler和RNG。历史`CVAE_INIT_CHECKPOINT` generalization接口保持兼容。

### 2.3 KL=0容量阶段的Mask、loss 与验收

固定 Mask bank 对每个窗口生成10个 fixture：`full_state`、`full_action`、`full_both`、10%和50% element Both、50%连续 State 时间块、50%连续 Action 时间块、50% State feature、50% Action feature、一个 State+Action joint semantic group。每个 fixture 必须至少有一个有效 target，padding 永远不作为 target。

fixed训练与exact validation必须使用同一个Mask seed，使每个窗口的element/time/feature/semantic
坐标逐位一致。`training_mask_seed == validation_mask_seed`且summary中的
`fixed_fixture_identity_match=true`是正式fixed run的启动断言。独立seed只允许用于
generalization阶段；旧F1违反此合同，因此只能作为同Mask类型、不同坐标的诊断，不能回答训练
fixture是否被完美记忆。

历史基线和当前A/C的训练loss仅包含被Mask坐标：State continuous MSE、Action MSE、contact BCE；当前batch中存在的三类loss等权平均。B只按第3.2节归一化加权连续元素，contact与有效分项的外层平均不变。全部分支仍用原始MSE/BCE报告可比重建指标。

每次validation同时计算两套互不替代的门禁：

| 指标 | exact诊断门禁 | progression规模推进门禁 |
|---|---:|---:|
| worst per-window State continuous normalized RMSE | `≤1e-4` | `≤1e-2` |
| worst per-window Action normalized RMSE | `≤1e-4` | `≤1e-2` |
| worst continuous/Action normalized absolute error | `≤1e-3` | `≤1e-2` |
| masked contact classification accuracy | `100%` | `100%` |
| `full_both` zero-latent RMSE / correct-latent RMSE | `≥10` | `≥10` |

两套score都取各自阈值比值的最大值。默认`CVAE_POSTERIOR_GATE=exact`保持历史协议；显式设为`progression`时，推进score控制checkpoint选择、提前停止、退出码和独立marker。两种门禁均要求连续3次PASS；分别保存`best_exact.pt`或`best_progression.pt`，`last.pt`始终保存。summary必须同时记录`exact_gate`与`progression_gate`，所以放宽推进门禁不会把严格失败改写成严格通过。swapped-latent仅作诊断。

上述默认入口不修改。F4B-v2独立入口保留两套阈值、checkpoint选择与连续3次的语义，但为了配对比较
固定跑完10k，质量通过以最后8000/9000/10000步为准；诊断完整与质量通过分开记录。相对改善可用于
选择下一项研究，不替代正式质量PASS，也不允许直接进入32-motion或R128。

### 2.4 过程loss与图表合同

`logs/metrics.jsonl`逐optimizer step记录训练batch的total/state/action/contact loss、learning rate与gradient norm。每次完整评测额外按全体masked元素的误差和/元素数计算State continuous MSE、Action MSE与contact BCE，再对存在的分项等权得到完整reconstruction loss；不得用验证batch均值造成最后一个batch偏置。

fixed图例必须写`Full fixed-fixture evaluation`，明确它是同一训练window和同一Mask的重评；R128写`Held-out-mask evaluation`，明确它仍使用已见序列。每次完整评测原子刷新`plots/training_curves.svg`、`plots/gate_curves.svg`和`plots/mask_breakdown.svg`。所有横轴注明`Optimizer step`；对数纵轴注明`log10 scale`并显示`10^-6`等实际刻度。原始训练loss最多均匀绘制2,000点，`EMA alpha=0.05`、评测点和JSONL不降采样；显示零值裁剪到`1e-12`但不改原始记录。

## 3. 最短必要执行链与停止规则

本轮不要求32 motion、T=128达到`1e-4` exact门禁，但要求它依次通过fixed与held-out Mask的`1e-2` progression门禁后再进入CVAE。这样用最短四级链检验25M posterior在目标规模上的可用记忆能力，同时保留exact曲线作为诊断。

| ID | 阶段 | 数据/初始化 | 上限 | 成功后下一步 |
|---|---|---|---:|---|
| S25 | 25M工程smoke | 1 motion、T=8、random init | 2 step | L128 |
| L128 | 单motion长窗口fixed | 1 motion、T=128、random init | 100k，每250 step验收 | F128 |
| F128 | 32-motion fixed规模门禁 | 32 motion、T=128，从L128 `last.pt` model-only warm-start | 200k，每2,500 step验收 | R128 |
| F4D | F128失败后的唯一规模边界诊断 | 4 motion、T=128，从L128 `last.pt` model-only warm-start；其余合同不变 | 100k，每1,000 step验收 | 已FAIL；执行F4A |
| F4A | F4D checkpoint只读尾部诊断 | 固定读取F4D `best_progression.pt`与同一80窗口×10 Mask | 已完成；execution PASS | F4B-v2 |
| F4B（旧） | 加法CVaR方案 | 未实现、未运行，保留历史提案 | SUPERSEDED | 由F4B-v2替代 |
| F4B-v2 | 归一化尾部加权A/B | 均从F4D `best_progression.pt`开始；A原损失，B只改连续元素权重 | 各10k，共20k | 按第3.2节选择复核或F4C |
| F4C | 小结构条件对照 | 同F4D起点、A原损失，只新增逐层零初始化latent门控 | 条件触发10k | 与同seed的A比较，选定复核方案或停止 |
| F4R | 训练顺序复核 | 仍从F4D起点，fixture seed不变；优化seed改为20260831 | A单独10k或A/胜出改动各10k | 先取得4-motion正式通过，再更新32-motion计划 |
| F4E | Auto-decoder根因诊断 | F4D原A decoder；每window共享一个可学习256维code，以十个q均值质心初始化 | E1 5k code-only；失败才E2最多15k code+decoder | 按encoder形成、协同优化或code+decoder容量三分结论 |
| F4F | 等code预算latent拓扑对照 | 均从F4D开始；G8=`8×256` global memory，T129=`129×16` per-time | 每臂固定15k，不早停；比较13k/14k/15k | 选择8-token posterior、层级CVAE或直接输出记忆上限诊断 |
| R128 | 动态随机Mask与16-slot held-out验收 | 32 motion、T=128，从F128 `last.pt` model-only warm-start | 50k，每2,500 step验收 | 冻结KL=0结果并开始K0代码实现 |
| K0 | KL三路径工程smoke | 仅在L128/F128/R128全部PASS后新增入口；从R128 `last.pt` model-only初始化 | 2 step、单个确定性窗口 | K1 |
| K1 | 32-motion KL三路径正式实验 | 32 motion、T=128，从R128 `last.pt` model-only初始化 | 50k，每2,500 step三路径验收 | 按KL判断表确定唯一下一步 |

初始快速链中任一级质量失败即停止，不直接启动后一级；T=256不在本轮关键路径。F128已经失败，
因此先完成F4D边界实验与F4A只读诊断，不恢复完整的4→8→16→32阶梯。F4B-v2/F4C比较已按规则
终止且不执行F4R；当前改为独立F4E根因诊断，正式预算至多20k step，各2-step工程smoke不计入预算。
F4E不修改数据、Mask、latent维度或A decoder结构，只将posterior输出替换为每window共享可学习code。
L128、F128、R128
三个`KL=0`正式阶段未全部获得对应PASS marker前，禁止新增或启用K0/K1模型接口、配置、训练器、
评测器、测试或Shell入口；只能继续更新本文的实际结果台账。R128通过并冻结其`last.pt`与基线指标
后，才接通posterior采样、conditional prior和KL。不读取目标真值的prior必须独立验收，
全State+Action同时遮挡只用于posterior latent依赖诊断，不作为conditional prior质量门禁。

marker保持可区分：exact fixed/generalization沿用`cvae_posterior_capacity.ok`与`cvae_posterior_mask_generalization.ok`；progression使用`cvae_posterior_capacity_progression.ok`与`cvae_posterior_mask_generalization_progression.ok`。progression PASS只允许声明“精度足以推进规模/机制实验”，不能称为完美拟合或物理单位无损。

### 3.1 F4A只读尾部诊断合同

F4A固定读取
`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425/checkpoints/best_progression.pt`，
重建与F4D逐位相同的80个window×10类fixed Mask。不得训练、更新checkpoint、改变Mask seed或
使用held-out Mask。输出必须位于新的`/home/helloworld/bly/runs/<f4a_run_id>/`，源HDF5与F4D
目录保持只读。

诊断必须记录每个fixture的motion、variant、window start、Mask名、State/Action RMSE、max abs，
并为最大误差记录State/Action类别、time index、feature index、target、prediction、absolute error及
对应normalization scale。聚合输出每类Mask的p50/p90/p95/p99/max、达到`1e-2` RMSE的fixture比例、
达到`1e-2` max abs的masked元素与fixture比例，以及最差20个window和最差20个feature。contact错误
单独列出，不与continuous尾部混合。

F4A结论按以下顺序确定：若partial Mask的p95 RMSE均`<=1e-2`且超阈值误差集中在不超过1%的
masked元素或5%的fixture，判定为`tail_objective_mismatch`，下一步只设计per-window均衡与尾部
惩罚对照；若partial Mask的p50或p90仍超过`1e-2`，判定为`broad_reconstruction_failure`，下一步
审计优化与表示容量；若`full_both`的p95 RMSE超过partial Mask宏平均p95的3倍，额外标记
`global_latent_bottleneck_suspected`。这些判断可以同时出现，但不得在F4A前预先修改loss或门禁。

### 3.2 F4B-v2 / F4C / F4R受控改进路线

状态：A/B/C正式训练及比较均已完成；manifest正式输出`STOP_LOSS_LATENT_SEED_SEARCH`，本受控改进路线结束，不执行F4R。详细合同、源路径、公式、接口和测试见[Next.md](Next.md)。
F4A的21.077%是跨全部fixed fixtures的masked元素出现次数比例，不是独立原始数据点的比例。
当前只检验有限预算下的优化与条件注入机制，不把平均误差达标直接归因为loss错误。

F4B-v2的A/B均从F4D `best_progression.pt` model-only开始，重置optimizer/scheduler/RNG；
4 motion、T128、80 windows、800 fixtures、micro-batch 4×累积16保持一致。每支10k，峰值LR `3e-5`，
250步warmup后cosine至`1e-6`，step0复现、随后每1k完整评测。fixture seed始终20260830；
第一轮优化seed为20260830，F4R只将优化seed改为20260831，不能改变Mask或窗口。

A保留原损失。B对每个fixture和State/Action连续域取`0.5×MSE+0.5×top-20% squared-error mean`，
再以各fixture该域的masked元素数加权，保持原micro-batch的元素数聚合口径。contact与有效分项外层
平均保持原样。`top_fraction=0.2`、`tail_mix=0.5`预先固定，不搜索；20%是启发式值，不声称由21.077%
严格推导出最优参数，也不同时增加per-fixture均衡或其他loss。

F4C已由正式比较manifest触发并在Windows实现。它保留A原损失与同一源起点，在每层decoder block前对有效
State/Action token加`g_l ⊙ P(z)`；`P`复用现有latent projection，`g_l`为384维零初始化向量。
latent token与padding不注入；只增加3072参数，总参数25,456,483，不修改encoder、RoPE或latent维度。
实现保持A/B结构开关默认关闭；C必须显式引用触发比较run，重新计算A/B summary哈希、配对与决定。
CPU测试已确认初始输出逐位一致、只注入有效数据token、固定latent不泄漏真值且门控梯度有限非零；
尚无真实HDF5/CUDA或质量效果证据。

主指标为最后8000/9000/10000步的超阈值元素比例和worst max abs，各自取三个对应点的中位数。
所有候选必须在三个点逐次满足contact 100%、zero-ratio至少10；global State/Action RMSE及worst
State/Action fixture RMSE分别相对同step的A与源F4D均不恶化超过10%。质量PASS要求最后连续3次
progression通过。保护条件、质量门禁、机制改善不能互相替代；原始MSE/BCE始终单独报告。

以下规则从上到下执行，避免“B已通过但改善不到50%仍启动C”的歧义：

| 阶段/观测 | 唯一决策 |
|---|---|
| A最后连续3次progression PASS | 跳过C，只以优化seed 20260831复核A；总正式训练30k |
| A未PASS，B满足全部保护条件且已PASS或两项主指标均改善至少50% | 跳过C，以新优化seed复核A/B；总40k |
| 上述两条均不满足 | 从F4D起点执行10k的C，与已有同seed、同step A比较 |
| C完成后，B/C中存在满足保护条件且两项主指标均改善至少20%的候选 | 选两项残余比例最大值较小者；相同优先B；复核A与胜出改动，总计不超过50k |
| 没有候选满足上一条 | 停止；记录本预算和两项干预未取得充分改善，不自动调loss、扩模型或续训 |

复核仍从F4D起点，不从第一轮胜者接着训练；只检验训练顺序稳健性，不称为从随机初始化的独立复现。
改动复核成功要求第二个优化seed同样取得至少20%的双指标改善并满足全部保护条件；强改善要求
两次均至少50%。A单独复核以最后3次正式progression为准。即使有稳健相对改善，若未取得4-motion
正式通过仍停留在该规模，后续预算和32-motion新run必须另行更新本台账后再执行。R128与KL冻结。

旧F4B提案（SUPERSEDED，未实现、未运行）：原定A/B各25k，B在原loss外增加权重1.0的per-fixture、
per-domain等权CVaR-20。该方案及“低于20%改善就只停止调loss”的旧规则由以上F4B-v2路线整体替代；
不作为当前实施合同，也不写成已完成的失败实验。

### 3.3 K1 KL训练合同（R128通过后才实现）

K1固定从R128的`last.pt`只加载模型参数，重新初始化optimizer、scheduler和RNG。训练数据为
32 motion、T=128；decoder只接收masked values、Mask和选定global latent。训练始终使用
posterior重参数采样：

$$
z_q=\mu_q+\exp(0.5\log\sigma_q^2)\epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$

$$
L=L_{reconstruction}+\beta D_{KL}\left(q(z\mid X_{full})\Vert
p(z\mid X_{visible},M)\right)
$$

`L_reconstruction`继续使用masked State MSE、Action MSE和contact BCE等权平均。`free_bits=0`；
`beta`在前10,000 optimizer step从0线性升至`1e-3`，之后保持；最多50,000 step，每2,500
step完整评测。保存`best_posterior_mean.pt`、`best_prior_sample.pt`和`last.pt`，三者不得互相覆盖。

K1只使用物理结构合理的Mask。State gap保留缺口前边界State、对应Action与覆盖控制延迟的近期
历史；Action gap保留缺口前Action历史及相邻State转移；State–Action联合gap只遮挡内部短片段，
保留前后边界并明确标注为双向inpainting。训练动态生成这些Mask；评测使用独立seed生成后固定的
held-out physical Mask bank。K1训练前先用未修改的R128 checkpoint在同一评测bank上运行一次，
作为`KL=0`基线。

### 3.4 三种latent注入路径与公平对照

只比较下列三种路径，不加入prior mean，也不把独立`N(0,I)`直接作为decoder latent：

| 路径 | latent | 可读取信息 | 解释 |
|---|---|---|---|
| `posterior_mean` | $z=\mu_q$ | 完整序列encoder + masked condition | encoder给出的确定性重建上限 |
| `posterior_sample` | $z=\mu_q+\sigma_q\epsilon$ | 完整序列encoder + masked condition | 检验posterior方差和采样噪声 |
| `conditional_prior_sample` | $z=\mu_p+\sigma_p\epsilon$ | 仅masked序列prior + masked condition | 不读取被Mask真值的实际CVAE生成路径 |

每个固定窗口/Mask上，`posterior_mean`运行1次，两个随机路径各运行8次。同一窗口、Mask和sample
index的posterior/prior采样必须复用同一个标准正态`epsilon`；seed由evaluation seed、motion、
variant、window start、Mask slot和sample index稳定派生。这样两条随机路径的差异主要来自
`q/p`分布而不是偶然噪声。任何使用posterior的结果都只能作为有真值上限，只有
`conditional_prior_sample`代表部署时不读取目标真值的路径。

三条路径分别记录masked State/Action normalized RMSE、continuous max abs和contact accuracy。
随机路径额外记录8次采样的mean、std、p50、p95、worst和best-of-8。定义同一评测bank上的宏观
连续误差$E$为State RMSE与Action RMSE的等权平均，并记录：

$$
R_{sample}=E_{posterior\_sample}/E_{posterior\_mean},\qquad
R_{prior}=E_{conditional\_prior\_sample}/E_{posterior\_sample}
$$

同时记录raw/weighted KL、当前beta、posterior/prior平均标准差、logvar范围，以及full-both
posterior mean的zero/swapped latent依赖。best-of-8只作oracle诊断，不参与checkpoint选择或质量
PASS；`best_prior_sample.pt`按8次采样均值对应的progression score选择。

### 3.5 KL权重判断与停止规则

以下判断是当前benchmark上的工程筛选，不声明`beta`在理论上最优。判断按表格从上到下执行；
若同时触发，posterior能力退化或latent collapse拥有最高优先级。

| 观测 | `kl_assessment` | 唯一下一步 |
|---|---|---|
| posterior mean保持progression；$R_{sample}\le1.25$；$R_{prior}\le1.50$；full-both zero-latent ratio仍`>=10` | `acceptable_beta_1e-3` | 进入conditional prior物理任务评测 |
| posterior mean相对同bank的R128基线恶化超过25%，或full-both zero-latent ratio低于10 | `kl_too_strong_or_collapse` | 从R128重新初始化，单独把beta改为`1e-4` |
| posterior保持progression且$R_{sample}\le1.25$，但$R_{prior}>2.0$ | `kl_too_weak` | 从R128重新初始化，单独把beta改为`1e-2` |
| $R_{sample}>1.25$但$R_{prior}\le1.50$ | `posterior_variance_unstable` | beta不变，先只读检查logvar、平均方差和逐sample误差 |
| 未落入以上区间，例如$1.50<R_{prior}\le2.0$ | `inconclusive` | 不自动改beta；先完成同checkpoint的方差与逐Mask诊断 |

progression绝对门禁仍要求worst per-window State/Action RMSE、continuous max abs均不超过`1e-2`，
contact为100%。`cvae_kl_latent_comparison.ok`只表示三路径评测完整、无真值泄漏且可复现；
KL是否可接受必须读取manifest中的`kl_assessment`，不得由marker名称推断。

K0/K1实现时必须生成`manifests/kl_latent_comparison.json`、
`plots/kl_training_curves.svg`与`plots/latent_three_path_comparison.svg`。训练图记录reconstruction、
raw KL、weighted KL、beta、learning rate与gradient norm；对照图分开显示三条路径及8次采样区间，
所有log轴继续使用明确的`10^n`刻度。manifest必须保存R128同bank基线、三路径完整统计、
`R_sample`、`R_prior`、latent依赖、`kl_assessment`和由判断表导出的唯一下一步。

延后实现时的测试范围固定为：三种latent公式；posterior可读完整真值而prior严格只能读取masked
序列；共享epsilon与稳定seed复现；8次采样聚合；beta线性预热与free-bits为0；三个checkpoint
互不覆盖；失败run仍保留manifest和SVG；comparison marker只表示执行完整。先通过CPU短合成测试，
再在Ubuntu执行K0；K0不承担任何质量结论。

## 4. Ubuntu 执行命令

### 4.1 每次运行前的只读检查

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

cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate
export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
```

实际 Git 状态若与 AGENTS.md 或本文冲突，先停止并回传状态；不得自动 reset。同步只能对干净且满足 fast-forward 的实际分支执行 `git pull --ff-only`。

### 4.2 Smoke、fixed 与 held-out

```bash
# S25：默认1 motion、T=8、2 step
unset CVAE_CONFIG CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_WARM_START CVAE_POSTERIOR_MAX_WINDOWS
unset CVAE_POSTERIOR_MAX_STEPS CVAE_POSTERIOR_VALIDATION_INTERVAL
CVAE_SEED=20260830 \
bash ./cvae_repro.sh posterior-capacity-25m-smoke

# L128：1 motion、T=128、随机初始化
unset CVAE_CONFIG CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_WARM_START CVAE_POSTERIOR_MAX_WINDOWS
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=fixed \
CVAE_POSTERIOR_GATE=progression \
CVAE_POSTERIOR_MAX_STEPS=100000 \
CVAE_POSTERIOR_VALIDATION_INTERVAL=250 \
CVAE_POSTERIOR_MOTIONS=1 \
CVAE_POSTERIOR_WINDOW=128 \
bash ./cvae_repro.sh posterior-capacity-25m

# F128：已于2026-09-04跑满200k并FAIL；以下命令仅保留历史复现，不立即重跑
unset CVAE_CONFIG CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_MAX_WINDOWS
test -f /home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t128_25m_s100000_gprogression_20260901_173153/checkpoints/last.pt
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=fixed \
CVAE_POSTERIOR_GATE=progression \
CVAE_POSTERIOR_WARM_START=/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t128_25m_s100000_gprogression_20260901_173153/checkpoints/last.pt \
CVAE_POSTERIOR_MAX_STEPS=200000 \
CVAE_POSTERIOR_VALIDATION_INTERVAL=2500 \
CVAE_POSTERIOR_MOTIONS=32 \
CVAE_POSTERIOR_WINDOW=128 \
bash ./cvae_repro.sh posterior-capacity-25m

# F4D：已于2026-09-05跑满100k并FAIL；以下命令仅保留历史复现
unset CVAE_CONFIG CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_MAX_WINDOWS
test -f /home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t128_25m_s100000_gprogression_20260901_173153/checkpoints/last.pt
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=fixed \
CVAE_POSTERIOR_GATE=progression \
CVAE_POSTERIOR_WARM_START=/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t128_25m_s100000_gprogression_20260901_173153/checkpoints/last.pt \
CVAE_POSTERIOR_MAX_STEPS=100000 \
CVAE_POSTERIOR_VALIDATION_INTERVAL=1000 \
CVAE_POSTERIOR_MOTIONS=4 \
CVAE_POSTERIOR_WINDOW=128 \
bash ./cvae_repro.sh posterior-capacity-25m

# R128：仅在F128 PASS后执行；动态训练Mask、16-slot held-out验收
unset CVAE_CONFIG CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_MAX_WINDOWS
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=generalization \
CVAE_POSTERIOR_GATE=progression \
CVAE_POSTERIOR_WARM_START=<F128_RUN>/checkpoints/last.pt \
CVAE_POSTERIOR_MAX_STEPS=50000 \
CVAE_POSTERIOR_VALIDATION_INTERVAL=2500 \
CVAE_POSTERIOR_MOTIONS=32 \
CVAE_POSTERIOR_WINDOW=128 \
bash ./cvae_repro.sh posterior-capacity-25m

# 不启动训练，只从已有metrics.jsonl重绘SVG
CVAE_RUN_DIR=<RUN> bash ./cvae_repro.sh posterior-capacity-plot
```

上述训练命令都必须创建新run，不得设置到已存在的`CVAE_RUN_DIR`或覆盖checkpoint；只有`posterior-capacity-plot`例外，它只读取指定run的JSONL并原子改写该run的三个SVG。运行中的目录使用`ls -dt /home/helloworld/bly/runs/cvae_posterior_capacity_* | head -n1`定位，不能依赖latest文件。

```bash
# F4A：已完成；以下只保留历史复现命令
export CVAE_POSTERIOR_DIAGNOSTIC_CHECKPOINT=/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425/checkpoints/best_progression.pt
unset CVAE_POSTERIOR_DIAGNOSTIC_BATCH_SIZE CVAE_POSTERIOR_DIAGNOSTIC_NUM_WORKERS
test -f "$CVAE_DATASET_RUN/markers/cvae_overfit_subset.ok"
test -f "$CVAE_POSTERIOR_DIAGNOSTIC_CHECKPOINT"
bash ./cvae_repro.sh posterior-capacity-tail-diagnostic
```

F4A必须创建新的`cvae_posterior_capacity_tail_diagnostic_f4a_*` run；源F4D目录、checkpoint和数据集
保持只读。`cvae_posterior_capacity_tail_diagnostic.ok`只表示800个fixture的诊断完整、source指标复现
一致且产物写全，不表示模型通过progression门禁。完成后读取
`manifests/posterior_tail_diagnostic.json`中的`tail_assessment`再选择唯一下一项。

F4B-v2 A/B入口现已实现。先固定只读来源并串行运行两支2-step smoke；不得设置`CVAE_RUN_DIR`，
不得从smoke checkpoint继续正式训练：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
export CVAE_POSTERIOR_AB_SOURCE_CHECKPOINT=/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425/checkpoints/best_progression.pt
export CVAE_POSTERIOR_F4A_RUN=/home/helloworld/bly/runs/cvae_posterior_capacity_tail_diagnostic_f4a_20260905_200807
export CVAE_POSTERIOR_OPTIMIZER_SEED=20260830
unset CVAE_CONFIG CVAE_RUN_DIR CVAE_POSTERIOR_AB_RUN_A CVAE_POSTERIOR_AB_RUN_B CVAE_POSTERIOR_AB_RUN_C CVAE_POSTERIOR_AB_INITIAL_COMPARISON

CVAE_POSTERIOR_AB_ARM=A bash ./cvae_repro.sh posterior-capacity-ab-smoke
CVAE_POSTERIOR_AB_ARM=B bash ./cvae_repro.sh posterior-capacity-ab-smoke
```

两支smoke已分别在
`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_a_seed20260830_smoke_20260906_112939`和
`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_b_seed20260830_smoke_20260906_113154`完成；
`source.step0_reproduction.passed=true`、checkpoint readback和`cvae_posterior_ab_smoke.ok`均成立，
且逐step训练身份SHA256完全一致。A/B正式10k与比较均已完成；以下命令保留为历史合同：

```bash
# A/B均已完成；不得重跑、续训或交叉加载checkpoint。
# CVAE_POSTERIOR_AB_ARM=A bash ./cvae_repro.sh posterior-capacity-ab
# CVAE_POSTERIOR_AB_ARM=B bash ./cvae_repro.sh posterior-capacity-ab

export CVAE_POSTERIOR_AB_RUN_A=<A_FORMAL_RUN>
export CVAE_POSTERIOR_AB_RUN_B=<B_FORMAL_RUN>
unset CVAE_POSTERIOR_AB_RUN_C CVAE_RUN_DIR
bash ./cvae_repro.sh posterior-capacity-ab-compare
```

正式比较run为
`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260906_235429`，其marker、全部配对检查和
`IMPLEMENT_F4C`决定、C smoke、C正式10k及A/B/C比较均已回传。以下命令全部保留为历史记录，当前没有获准的训练命令：

```bash
export CVAE_POSTERIOR_AB_TRIGGER_COMPARISON=/home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260906_235429
export CVAE_POSTERIOR_OPTIMIZER_SEED=20260830
unset CVAE_CONFIG CVAE_RUN_DIR CVAE_POSTERIOR_AB_INITIAL_COMPARISON

# 已完成：CVAE_POSTERIOR_AB_ARM=C bash ./cvae_repro.sh posterior-capacity-ab-smoke
# 已完成：CVAE_POSTERIOR_AB_ARM=C bash ./cvae_repro.sh posterior-capacity-ab

# 已完成：
# export CVAE_POSTERIOR_AB_RUN_A=/home/helloworld/bly/runs/cvae_posterior_capacity_ab_a_seed20260830_20260906_114634
# export CVAE_POSTERIOR_AB_RUN_B=/home/helloworld/bly/runs/cvae_posterior_capacity_ab_b_seed20260830_20260906_203844
# export CVAE_POSTERIOR_AB_RUN_C=/home/helloworld/bly/runs/cvae_posterior_capacity_ab_c_seed20260830_20260907_004301
# unset CVAE_POSTERIOR_AB_INITIAL_COMPARISON CVAE_RUN_DIR
# bash ./cvae_repro.sh posterior-capacity-ab-compare
```

C正式run已从同一F4D源重新初始化并跑满10k；gate与checkpoint协议全部通过。比较器随后重验A/B/C
逐step训练身份和固定合同，并正式输出停止决定。本路线不再提供追加训练命令。

正式run即使质量失败也应正常完成并同时保留`cvae_posterior_ab_execution.ok`与内容为
`QUALITY_FAIL`的`cvae.failed`；只有最后8k/9k/10k均通过progression才生成既有quality marker。
比较器核对逐step采样SHA256并只输出一个下一动作；C仅在缺失或无法通过哈希复核的触发comparison manifest时被训练入口拒绝。当前正式comparison已经触发`IMPLEMENT_F4C`。

K0/K1当前没有执行命令。只有L128、F128、R128全部通过并完成本文结果回填后，才允许在Windows
设计和实现新的KL配置、Python入口、Shell命令及测试；实现完成且Windows轻量验证通过后，再把
经实际代码确认的Ubuntu命令补入本节。不得提前用现有posterior-capacity入口伪装成KL实验。

### 4.3 完成后的结果采集

```bash
RUN=<absolute_run_dir>
cat "$RUN/manifests/posterior_capacity_summary.json"
tail -n 20 "$RUN/logs/metrics.jsonl"
cat "$RUN/manifests/source_commit.txt"
cat "$RUN/manifests/source_status.txt"
find "$RUN/markers" -maxdepth 1 -type f -printf '%f\n' | sort
ls -lh "$RUN/checkpoints"
```

回传时优先回传小型 manifest、日志尾部和 marker 列表，不复制 HDF5 或大 checkpoint。结果必须来自实际文件，不从终端片段猜测最佳指标。

F4B-v2使用独立summary；运行中可从最新run读取训练step，完成后回传最后三次正式评测、来源状态和marker：

```bash
RUN=<absolute_ab_run>
grep '"phase": "train"' "$RUN/logs/metrics.jsonl" | tail -n 1 | jq -c '{step:.optimizer_step,raw:.raw_reconstruction.total,opt:.optimization_objective.total,lr:.learning_rate,grad:.gradient_norm_before_clip,seconds:.step_seconds}'
jq '{execution_pass,quality_pass,arm,fixture_seed,optimizer_seed,checkpoint_readback,data_contract,final_three:[.evaluations[]|select(.optimizer_step==8000 or .optimizer_step==9000 or .optimizer_step==10000)|{step:.optimizer_step,pass:.exact.progression_gate.passed,state:.exact.worst_state_rmse,action:.exact.worst_action_rmse,max_abs:.exact.continuous_max_abs,exceed:.tail_global.threshold_exceed_fraction,contact:.exact.contact_accuracy,zero:.exact.latent_dependence.zero_ratio}]}' "$RUN/manifests/posterior_ab_summary.json"
cat "$RUN/manifests/source_commit.txt"
cat "$RUN/manifests/source_status.txt"
find "$RUN/markers" -maxdepth 1 -type f -printf '%f\n' | sort
```

## 5. 实验状态台账

实验状态只允许：`PENDING`、`RUNNING`、`PASS`、`FAIL`、`BLOCKED`；Windows代码准备项可写`READY`，被新计划明确替代且不再执行的旧项写`SUPERSEDED`。只有相应正式marker存在且summary一致时实验才能填`PASS`；进程正常退出但质量gate未通过仍为`FAIL`。

| ID | 状态 | run_dir | source HEAD | best step/score | State RMSE | Action RMSE | max abs | contact | zero ratio | marker | 结论/下一步 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| I0 Windows实现 | PASS | N/A | `fcdb4f8861e539e3ea364e578d6bc96ce7ebd9b0` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 27项组合测试、JSON、compile、Shell语法、diff check通过；执行S0 |
| S0 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t8_20260831_114746` | 待读取 `source_commit.txt` | step 2 / 25194.9549（仅smoke诊断） | 2.5195 | 2.2020 | 15.7025 | 52.47% | 0.9985 | `cvae_posterior_capacity_smoke.ok` | 工程链路通过；F1随后执行并失败 |
| F1 | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_20260831_114833` | `fcdb4f8861e539e3ea364e578d6bc96ce7ebd9b0` | step 40000 / 752.0263 | 0.05842 | 0.02495 | 0.75203 | 100% | 371.67 | `cvae.failed` | validation部分Mask坐标与训练不同；结果不构成fixed记忆失败证据 |
| D1 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_w1_20260831_150654` | 待读取`source_commit.txt` | step 34000 / 1.0 | 4.719e-5 | 8.075e-5 | 2.255e-4 | 100% | 10342.86 | `cvae_posterior_capacity.ok` | 单窗口10类Mask全部exact；执行F1R |
| F1R | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_20260831_202012` | `b3aa63d9514cd8dd284e7f6091fc26877f57f021` | step 39000 / 36.4095 | 0.003641 | 0.002903 | 0.017851 | 100% | 331.97 | `cvae.failed` | 全部Mask同量级且平台化；执行W4 |
| W4 | FAIL | `<latest>/cvae_posterior_capacity_fixed_m1_t16_w4_*` | `b3aa63d9514cd8dd284e7f6091fc26877f57f021` | step 39750 / exact 1.6103；progression 1.0 | 1.610e-4 | 1.194e-4 | 6.158e-4 | 100% | 8425.42 | `cvae.failed`（旧exact协议） | exact失败；最近5次均满足新progression gate，不重跑，直接P1 |
| P1 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_s100000_gprogression_20260901_105145` | 待读取`source_commit.txt` | step 93750 / progression 1.0；exact 22.4268 | 0.002243 | 0.001460 | 0.009938 | 100% | 610.99 | `cvae_posterior_capacity_progression.ok` | 历史6.7M证据；后续由25M快速阶梯接管 |
| P2 | SUPERSEDED | — | — | — | — | — | — | — | — | — | 由R128在32 motion、T=128上统一完成动态Mask验收 |
| I25 Windows实现 | PASS | N/A | `tiny-model@fa50f444d9a481b0f431edc1b1f974f0998f623a`+工作树 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Windows轻量验证及Ubuntu S25工程链路均通过 |
| S25 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t8_25m_gprogression_20260901_172054` | 待读取`source_commit.txt` | step 2，仅smoke诊断 | 未回传 | 未回传 | 未回传 | 未回传 | 未回传 | `cvae_posterior_capacity_smoke.ok`（由Shell成功返回确认） | 真实HDF5/CUDA、25M参数断言、checkpoint与三张SVG通过；执行L128 |
| L128 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t128_25m_s100000_gprogression_20260901_173153` | 待读取`source_commit.txt` | step 93500 / progression 1.0；exact 23.0210 | 0.002302 | 0.001533 | 0.009942 | 100% | 469.56 | `cvae_posterior_capacity_progression.ok`（由Shell成功返回确认） | 24窗口、240 fixed fixtures达到推进精度；执行F128 |
| F128 | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m32_t128_25m_s200000_gprogression_20260902_235140` | `6463b2ec960cda22c7ed70814a46a44e6804d4c0` | step 175000 / progression 964.4091 | 0.261443 | 0.273665 | 9.644091 | 99.9958% | 12.059 | `cvae.failed` | 200k末段平台化，直接1→32扩展大幅失败；执行F4D |
| F4D | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425` | `6463b2ec960cda22c7ed70814a46a44e6804d4c0` | step 100000 / progression 21.5620 | 0.023021 | 0.015263 | 0.215620 | 100% | 110.239 | `cvae.failed` | 平均RMSE已过1e-2但worst尾部失败；执行F4A |
| I-F4A Windows实现 | PASS | N/A | `tiny-model@6463b2ec960cda22c7ed70814a46a44e6804d4c0`+工作树 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 独立只读入口、严格F4D合同、逐fixture/feature/contact产物、SVG及execution-only marker已实现；执行F4A |
| F4A | PASS（execution） | `/home/helloworld/bly/runs/cvae_posterior_capacity_tail_diagnostic_f4a_20260905_200807` | `2ff1ec95db72fed9db80d3b040cccc32b3f9703f` | source step 100000；`mixed_tail_and_reconstruction_failure` | 0.009149 global / 0.023021 worst fixture | 0.008118 global / 0.015263 worst fixture | 0.215620 | 100% | N/A（只读复用F4D） | `cvae_posterior_capacity_tail_diagnostic.ok` | 21.08%元素及100% fixtures超max阈值；当前由F4B-v2接管 |
| F4B（旧） | SUPERSEDED | — | 未实现、未运行 | — | — | — | — | — | — | — | 加法CVaR与各25k提案由F4B-v2替代 |
| I-F4B-v2实现 | PASS | N/A | `c1ae5f79111bf61ddace32073df8122dbbefec95` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | A/B入口、严格step0、双seed、完整统计、真实donor、比较器与marker隔离已由两支Ubuntu smoke验证 |
| F4B-v2 A smoke | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_ab_a_seed20260830_smoke_20260906_112939` | `c1ae5f79111bf61ddace32073df8122dbbefec95` | step 2 / progression score 21.5601（仅smoke诊断） | 未回传 | 未回传 | 未回传 | 未回传 | 未回传 | `cvae_posterior_ab_smoke.ok` | execution-only：step0/legacy复现、2步训练、评测和checkpoint读回通过；无质量结论 |
| F4B-v2 B smoke | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_ab_b_seed20260830_smoke_20260906_113154` | `c1ae5f79111bf61ddace32073df8122dbbefec95` | step 2 / progression score 21.5606（仅smoke诊断） | 未回传 | 未回传 | 未回传 | 未回传 | 未回传 | `cvae_posterior_ab_smoke.ok` | execution-only：与A的窗口、fixture及两步训练身份逐位配对；step0与checkpoint读回通过 |
| F4B-v2 A正式 | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_ab_a_seed20260830_20260906_114634` | `c1ae5f79111bf61ddace32073df8122dbbefec95` | step 10000 / progression score 13.7220 | 0.015999 worst / 0.008425 global | 0.014238 worst / 0.007545 global | 0.137220 | 100% | 122.437 | `cvae_posterior_ab_execution.ok` + `cvae.failed` | 8k/9k/10k均FAIL；18.456%元素及800/800 fixtures仍超阈值；执行B正式10k |
| F4B-v2 B正式 | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_ab_b_seed20260830_20260906_203844` | `c1ae5f79111bf61ddace32073df8122dbbefec95` | step 10000 / progression score 13.0564 | 0.016398 worst / 0.008444 global | 0.014550 worst / 0.007629 global | 0.130564 | 100% | 120.443 | `cvae_posterior_ab_execution.ok` + `cvae.failed` | 8k/9k/10k均FAIL；20.718%元素仍超阈值；身份完全配对，运行比较器 |
| F4B-v2 A/B比较 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260906_235429` | `c1ae5f79111bf61ddace32073df8122dbbefec95` | `IMPLEMENT_F4C` | — | — | `R_p=1.11982` / `R_a=0.95902` | guards PASS | — | `cvae_posterior_ab_comparison.ok` | 13项配对检查全PASS；B不满足progression/50%规则，授权实现C |
| I-F4C Windows实现 | PASS | N/A | `c965487a5291ada3919db74b017900a3a04eb6ab` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 8层零初始化gate、触发授权、严格迁移、诊断与比较兼容已由Ubuntu smoke验证 |
| F4C smoke | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_ab_c_seed20260830_smoke_20260907_003414` | `443f4782c823fc7adaa8730e7513cd7671b630e6` | step 2 / progression score 21.560766（仅smoke诊断） | 0.023020 | 0.015263 | 0.215608 | 100% | 110.241 | `cvae_posterior_ab_smoke.ok` | execution-only：step0、触发、gate梯度/更新和checkpoint读回全部通过；执行C正式10k |
| F4C正式 | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_ab_c_seed20260830_20260907_004301` | `163be1f4c40c46bbd5c680b7ac1711e87f382e97` | step 10000 / progression score 17.1898 | 0.019039 worst / 0.008196 global | 0.013908 worst / 0.007330 global | 0.171898 | 100% | 123.268 | `cvae_posterior_ab_execution.ok` + `cvae.failed` | 8k/9k/10k均FAIL；17.379%元素超阈值；后续比较器已终止本路线 |
| F4B-v2 A/B/C比较 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260907_102553` | `8fbe327c487c66482ba4ece6912c8f76bab3720b` | `STOP_LOSS_LATENT_SEED_SEARCH` | — | — | B `1.11982/0.95902`；C `0.94117/1.25945` | B PASS；C FAIL | — | `cvae_posterior_ab_comparison.ok` | B/C均不满足受保护20%改善；终止当前loss/gate/seed路线 |
| F4R | SUPERSEDED | — | 未运行 | — | — | — | — | — | — | — | 正式比较未授权任何第二seed分支 |
| I-F4E Windows实现 | READY | N/A | `c54f0ce4c6da166cfe7b70422302bdc454d805e7` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 80个共享code、质心初始化、decoder-only API、E1/E2隔离训练、三类code依赖和独立marker已实现；53项相关测试PASS，全发现106项仅3个既有h5py导入限制；先执行Ubuntu smoke |
| F4E smoke v1 | REPORTING_FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_autodecoder_f4e_smoke_20260907_114405` | `5cf48cac8b4289b75d44c148b26513f1d2510c21` | 2 step，execution complete | 未回传 | 未回传 | 未回传 | 未回传 | 未回传 | `cvae_posterior_autodecoder_smoke.ok`（由Shell成功返回确认） | 工程链路通过，但根因字段误报正式失败；结果只作smoke，修复后重跑 |
| F4E smoke v2 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_autodecoder_f4e_smoke_20260907_115809` | `dffc0bf25aa0db21e06126e7e3dafba9689e4230` | E1 2 step | 工程诊断 | 工程诊断 | 工程诊断 | 工程诊断 | 无容量结论 | `cvae_posterior_autodecoder_smoke.ok` | 源复现、80/800身份、共享code、encoder隔离与checkpoint读回通过；允许正式F4E |
| F4E正式 | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_autodecoder_f4e_20260907_120414` | `cfb6735b54f56e49977665948397f80855987d74` | E1 5k + E2 15k；best step20k；score20.9413 | 0.029776 worst / 0.008951 global | 0.020474 worst / 0.007856 global | 0.209413 | 100% | zero/cross-window/cross-motion `95.65/94.85/119.93` | `cvae_posterior_autodecoder_execution.ok` + `cvae.failed` | E1/E2均FAIL；19.679%元素超阈值；单256维共享code+当前decoder容量未获证明 |
| I-F4F Windows实现 | READY | N/A | `302cb594c3e1c828256946110f6ba0aece36d8de` | G8/T129各2-step smoke入口与固定15k正式入口 | — | — | — | — | — | 尚无Ubuntu marker | 总参数25,620,323/25,625,059；60项posterior相关测试PASS，全发现116项PASS且3项仅缺h5py；先执行G8 smoke |
| F4F G8 smoke v1 | ENGINEERING_FAIL | 路径尚未回传 | 修复前`302cb594…` | 0 optimizer step；构造optimizer时报`KeyError: topology_learning_rate` | — | — | — | — | — | `cvae.failed`应由Shell生成，待路径核验 | 配置键不一致；无模型结论，使用`774c3ac…`重新smoke |
| F4F G8 smoke v2 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_g8_smoke_20260907_181608` | `478f479bb3345697ff5c7da14583c9c3e13be193` | 2 step；best step0/score2321.019 | 1.464988 | 1.779843 | 23.210190 | 94.725% | zero/cross-window/cross-motion `1.042/0.998/1.042` | `cvae_posterior_latent_topology_smoke.ok` | 工程合同全PASS；随机code两步质量无结论，执行T129 smoke |
| F4F T129 smoke | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_t129_smoke_20260907_182754` | `478f479bb3345697ff5c7da14583c9c3e13be193` | 2 step；随机code质量值不作容量判断 | 未回传；非smoke门禁 | 未回传；非smoke门禁 | 未回传；非smoke门禁 | 未回传；非smoke门禁 | 工程依赖检查PASS | `cvae_posterior_latent_topology_smoke.ok` | T129参数、身份、隔离及读回全PASS；执行G8正式15k |
| F4F G8 formal | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_g8_20260907_184707` | `8012b972b5d842f3196586eb995c963fb6dda06d` | 固定15k；best step15k/score71.0758 | 0.071671 worst / 0.018158 global | 0.043046 worst / 0.013475 global | 0.710758；39.456%超1e-2 | 100% | zero/cross-window/cross-motion `42.80/43.25/54.69` | `cvae_posterior_latent_topology_execution.ok` + `cvae.failed` | 有效质量失败；13k/14k/15k均FAIL，不追加步数，执行T129正式15k |
| F4F T129 formal | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_t129_20260907_213123` | `6e7caed535721a5ee575b80eba138cb6e392152e` | 固定15k；best step15k/score290.9683 | 0.117395 worst / 0.040533 global | 0.065137 worst / 0.025330 global | 2.909683；53.122%超1e-2 | 100% | zero/cross-window/cross-motion `18.94/16.58/20.97` | `cvae_posterior_latent_topology_execution.ok` + `cvae.failed` | 有效质量失败且全面差于G8；运行显式双臂比较器 |
| F4F topology compare | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_comparison_20260908_000708` | `6e7caed535721a5ee575b80eba138cb6e392152e` | 只读比较13k/14k/15k；18项身份检查PASS | G8中位0.073426；T129中位0.118866 | G8中位0.044032；T129中位0.065852 | G8中位0.729531；T129中位2.956917 | 两臂100% | 两臂均满足依赖≥10 | `cvae_posterior_latent_topology_comparison.ok` | `BOTH_FAIL_LATENT_TOPOLOGY_INSUFFICIENT`；停止latent拓扑扩展 |
| F4G direct-output ceiling | PENDING | — | 尚未设计/实现 | 先隔离objective/evaluator，再定位decoder上限 | — | — | — | — | — | — | 当前唯一方向；新合同明确前不得启动训练 |
| R128 | PENDING | — | — | — | — | — | — | — | — | — | 仅F128 PASS后训练动态Mask并验收held-out Mask；通过后才允许实现KL接口 |
| K0 | PENDING | — | 尚未实现；强制等待L128/F128/R128全部PASS | — | — | — | — | — | — | — | KL三路径单窗口2-step工程smoke |
| K1 | PENDING | — | 尚未实现；从R128 `last.pt` model-only初始化 | — | — | — | — | — | — | — | 32 motion、T128、beta线性预热与三路径正式对照 |
| K2 | PENDING | — | 不预先实现 | — | — | — | — | — | — | — | 仅当K1的`kl_assessment`要求调整beta或进入prior物理评测时确定 |

## 6. 每次实验完成后的强制总结

每次 run 结束后，在本节最上方追加一条记录，并同步更新第5节对应行。不得只写“通过/失败”；必须记录证据、解释边界和唯一下一步。

复制以下模板：

```markdown
### YYYY-MM-DD HH:MM — <ID> <PASS|FAIL|BLOCKED>

- Run：`<absolute_run_dir>`
- 代码：外层 `<branch>@<full_head>`；SONIC `<branch>@<full_head>`；IsaacLab `<head>`；工作树是否符合预期
- 配置：motion `<N>`，window `<T>`，seed `<seed>`，phase `<fixed|generalization>`，参数量 `<count>`
- 执行：开始/结束时间，optimizer step，samples seen，是否提前停止，marker 列表
- 最佳结果：exact score、State RMSE、Action RMSE、max abs、contact accuracy、zero/swapped latent ratio
- 分层结果：10类 fixed 或16类 held-out 中最差的三个 case及其具体指标
- 事实结论：本 run 能证明什么、不能证明什么；失败属于工程执行、优化、容量、Mask还是contact门禁
- 后续计划：只写一个下一实验 ID；若改变模型/优化，只允许改变一个变量并写明理由
```

K0/K1完成后除上述字段外，还必须追加以下KL专用字段：

```markdown
- R128同bank基线：posterior-mean State/Action RMSE、max abs、contact、zero/swapped ratio
- Posterior mean：State/Action RMSE、max abs、contact、相对R128退化比例
- Posterior sample（8次）：mean/std/p50/p95/worst/best-of-8、`R_sample`
- Conditional prior sample（8次）：mean/std/p50/p95/worst/best-of-8、`R_prior`
- 分布诊断：raw/weighted KL、beta、posterior/prior平均std、logvar范围、latent依赖
- KL判断：`kl_assessment`、触发的明确判据和唯一下一步
```

结果更新规则：

1. `RUNNING` 时只记录 run_dir、PID/进程状态和最后 step，不提前写质量结论。
2. 成功退出后读取 summary、metrics、marker 和 checkpoint 文件；四者矛盾时按失败处理并调查。
3. `cvae_posterior_capacity_smoke.ok` 只证明工程管线；不得填入正式质量指标结论。
4. 质量失败目录和 `cvae.failed` 必须保留，不删除、不复用；`best_exact.pt` 仍作为诊断资产。
5. 每次更新后同步修改本文“最后更新”和“当前阶段”，并在 AGENTS.md 记录新的已验证事实。

### 2026-09-08 00:07 — F4F显式比较 PASS（双臂质量FAIL）

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_comparison_20260908_000708`；源码`tiny-model@6e7caed535721a5ee575b80eba138cb6e392152e`，exit code 0，仅生成`cvae_posterior_latent_topology_comparison.ok`。
- 配对合同：dataset/source/F4E/fixture/window/motion、训练seed与完整15k训练身份、optimizer、比较step、初始化seed/分布、F4E全局保护、两臂shape/参数和code预算共18项检查全部PASS。G8/T129 summary SHA256为`845a3d5e...`/`92d2f632...`。
- 三点中位：G8/T129 score为72.953/295.692，worst State为0.073426/0.118866，worst Action为0.044032/0.065852，max abs为0.729531/2.956917，超阈值比例为39.849%/53.272%。两臂contact均100%，code依赖均超过10倍；G8全面优于T129但仍远未PASS。
- 预算公平性：每window code标量为2,048/2,064，差0.775%；总参数差0.0185%。因此结果不能归因于明显参数预算差异。
- 固定结论：`BOTH_FAIL_LATENT_TOPOLOGY_INSUFFICIENT`。在当前F4D初始化、A损失、decoder与15k协议下，增加global memory token或改为16维per-time code都没有解决已见80窗口的精确记忆；不允许选择相对更好的G8继续推进。
- 边界：该比较不证明global或temporal latent在其他初始化、更长训练或其他decoder中理论上不可行，也不涉及posterior/prior、随机Mask或未见motion；它只终止当前latent拓扑扩展路线。
- 唯一下一步：`RUN_DIRECT_OUTPUT_MEMORY_CEILING_FOR_DECODER_AND_OBJECTIVE`。先建立无需latent编码/广播的直接输出上限，区分objective/evaluator是否可达到门禁，再决定是否仍需诊断decoder；新合同明确前不启动训练。

### 2026-09-07 21:31 — F4F T129正式 FAIL（execution PASS）

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_t129_20260907_213123`；源码`tiny-model@6e7caed535721a5ee575b80eba138cb6e392152e`，完成固定15,000 step、exit code 0。F4E十一项授权检查全PASS；marker为`cvae_posterior_latent_topology_execution.ok`与内容`QUALITY_FAIL execution_complete=true arm=T129 last_three=false`的`cvae.failed`。
- 最佳及最后点：step15k、score290.968275；worst State/Action RMSE为0.117395/0.065137，continuous max abs为2.909683，global State/Action RMSE为0.040533/0.025330，分别超过0.009847/0.008642保护上限；contact 100%，53.122%连续元素超1e-2。
- 主比较三点：13k/14k/15k score为306.470/295.692/290.968，max abs为3.0647/2.9569/2.9097，超阈值比例为53.493%/53.272%/53.122%。趋势缓慢改善但距离门禁极远，不允许追加预算。
- Mask与依赖：`full_both`最差（State/Action 0.117395/0.065137、max abs 2.909683）；其余九类max abs也全部高于1e-2。zero/cross-window/cross-motion依赖为18.94/16.58/20.97，均通过10倍要求，说明time code被使用但重建精度不足。
- 公平与完整性：15,000个训练身份已记录，step1/15k SHA256为`ce6aa126...`/`6dfe514f...`，与G8完全一致。best/last checkpoint读回全PASS，SHA256为`4a9c51d0...`/`4d08038c...`，五张SVG均生成。
- 配对事实：T129相对G8的worst State/Action、max abs、global State/Action及超阈值比例全部更差；当前协议不支持“时间局部latent是必要结构”，也不能据此证明任何更长训练或其他decoder下的理论不可能性。
- 唯一下一步：只运行显式G8/T129比较器，核验全量身份并生成固定`BOTH_FAIL`结论；比较前后均不启动新训练。

### 2026-09-07 18:47 — F4F G8正式 FAIL（execution PASS）

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_g8_20260907_184707`；源码`tiny-model@8012b972b5d842f3196586eb995c963fb6dda06d`，完成固定15,000 step、exit code 0。marker为`cvae_posterior_latent_topology_execution.ok`与内容`QUALITY_FAIL execution_complete=true arm=G8 last_three=false`的`cvae.failed`。
- 最佳及最后点：step15k、score71.075785；worst State/Action RMSE为0.071671/0.043046，continuous max abs为0.710758，global State/Action RMSE为0.018158/0.013475，contact 100%，39.456%连续元素超1e-2。
- 主比较三点：13k/14k/15k score为75.341/72.953/71.076，max abs为0.7534/0.7295/0.7108，超阈值比例为40.374%/39.849%/39.456%。方向持续改善但远未接近门禁，不得以趋势追加预算。
- Mask与依赖：`full_both`最差（State/Action 0.071671/0.043046、max abs 0.710758）；其余九类max abs也全部高于1e-2。zero/cross-window/cross-motion依赖为42.80/43.25/54.69，证明code被强烈使用，失败不是code忽略；但这不证明G8拓扑在更长预算下理论上不可能拟合。
- 工程完整性：15,000个训练身份均已记录，step1/15k SHA256为`ce6aa126...`/`6dfe514f...`；best/last checkpoint读回全PASS，SHA256为`0a1838c9...`/`8e547c36...`，五张SVG均已生成。回传中global limit为`null`是查询字段顺序写反，manifest实际字段为`global_state_limit/global_action_limit`，不影响quality gate计算。
- 相对F4E：G8的全局、worst和max-abs指标均差于F4E正式E2结果；在当前固定初始化与15k协议下，多global token没有恢复已见80窗口的精确记忆能力。该结论仍不涉及posterior encoder、conditional prior、新Mask或未见motion。
- 唯一下一步：从同一F4D源、相同seed和训练顺序独立执行T129正式15k，不加载G8 checkpoint；T129结束后才运行显式比较器。

### 2026-09-07 18:16 — F4F G8 smoke v2 PASS（仅工程合同）

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_g8_smoke_20260907_181608`；源码`tiny-model@478f479bb3345697ff5c7da14583c9c3e13be193`，除两个预期嵌套仓库目录外status干净，exit code为0且只有smoke marker。
- 执行：G8完成2个optimizer step，入口打印`PASS (execution complete)`及run路径并返回提示符；依据Shell合同确认`cvae_posterior_latent_topology_smoke.ok`存在。
- 质量读数：`quality_pass=false`、best progression score2321.01898。smoke只跑2步且code随机初始化，这些数值不参与容量门禁，不能称为G8失败。
- 合同审计：F4D八项复现与F4E十项授权全PASS；4 motion、80 windows、800 fixtures和全部身份hash一致。G8 code为`[80,8,256]`共163,840参数，新增slot为3,072参数，总参数25,620,323；同window跨Mask共享且无per-fixture code。
- 初始化与隔离：seed/subseed为20260832/20260833，code/总初始化hash为`f760b14f...`/`1d8df28d...`；训练/fixture seed为20260831/20260830。encoder/posterior/prior调用均为0，79个冻结参数无梯度；两步sample identity分别为`ce6aa126...`和`85e87a2f...`。
- 读回：best/last checkpoint所有检查均PASS，SHA256为`299eda1f...`与`f3868860...`。训练拓扑组265,600参数包含code、slot及复用的98,688参数latent projection；它与model contract中“仅新增slot参数3,072”的统计口径不冲突。
- 唯一下一步：从同一F4D checkpoint独立运行T129 2-step smoke；不得加载G8 checkpoint。T129合同审计通过后，才按顺序启动G8正式15k。

### 2026-09-07 18:27 — F4F T129 smoke PASS（仅工程合同）

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_latent_topology_f4f_t129_smoke_20260907_182754`；源码`tiny-model@478f479bb3345697ff5c7da14583c9c3e13be193`，exit code为0且仅生成`cvae_posterior_latent_topology_smoke.ok`。
- 数据与模型：F4D八项复现检查全PASS；4 motion、T128、80 windows、800 fixtures，fixture/window hash分别为`2e97a990...`/`e740b5c1...`。code为`[80,129,16]`，共165,120参数、每window 2,064标量；总参数25,625,059，同window跨Mask共享且无per-fixture code。
- 初始化与公平性：初始化/拓扑subseed为20260832/20260833，code与总初始化hash为`2501bcf4...`/`dc344bc3...`；训练/fixture seed为20260831/20260830。step1/2采样身份为`ce6aa126...`/`85e87a2f...`，与G8完全相同。
- 隔离与读回：encoder/posterior/prior调用均为0，79个冻结参数无梯度；trainable topology/decoder/总参数为171,648/14,312,163/14,483,811。best/last checkpoint读回全PASS，SHA256为`6be0ca4a...`与`64a1b55a...`。
- 授权说明：回传命令错误查询`.source.f4e_authorization`因而显示`null`；实现中的真实字段为`.source.f4e_baseline.authorization_checks`，且任一授权检查失败都会抛错并阻断本run。成功完成2步、写出summary/checkpoint及marker证明该前置合同已通过。
- 质量边界：`quality_pass=false`不构成T129失败；smoke只验证随机code的step0与2步训练链路，未回传的质量数值不用于选择拓扑。
- 唯一下一步：从F4D `best_progression.pt`独立启动G8正式15k；不得加载G8/T129 smoke checkpoint。G8完成并回传后才启动T129正式15k。

### 2026-09-07 — F4F G8 smoke v1 ENGINEERING_FAIL

- Run与源码：Shell因非零退出未打印run路径；`source_commit.txt`、失败目录和marker列表尚未回传。输入固定为4 motion F4D checkpoint、正式F4E授权run与G8拓扑。
- 执行边界：完整step0评测已在optimizer构造之前发生，但随后在`_optimizer`读取学习率时抛出`KeyError: topology_learning_rate`；因此完成0个optimizer step，无G8容量、收敛或质量结论。
- 根因：`posterior_capacity_latent_topology.json`按合同定义`code_learning_rate/code_minimum_learning_rate`，训练器却错误读取不存在的`topology_learning_rate/topology_minimum_learning_rate`。这是纯工程键名不一致，与数据、显存、模型前向及门禁无关。
- 修复与验证：Windows提交`774c3ac9bcd285f73ef3336ec48d63eaa055d198`改为读取固定`code_*`键；回归测试现在对G8和T129都实际调用`configure_trainable_parameters`、构造两组AdamW参数组及双lambda scheduler。posterior相关60项测试重新全部PASS。
- 唯一下一步：同步修复后从F4D源新建G8 2-step smoke；不得复用失败run/checkpoint。回传新run的summary、source commit、marker、初始化与checkpoint读回后，再决定T129 smoke。

### 2026-09-07 — I-F4F Windows实现 READY

- 代码：实现提交为`tiny-model@302cb594c3e1c828256946110f6ba0aece36d8de`；新增固定配置、独立训练/比较模块及三个Shell入口，未修改F4D/F4E checkpoint或Ubuntu run。
- 模型：新增无参数`decode_from_latent_topology`。G8使用`[80,8,256]`共享window code、原256→384投影和8个slot embedding，总参数25,620,323；T129使用`[80,129,16]`纯时间code及16→384投影，总参数25,625,059。两者每window code标量差0.78%，总参数差0.019%。
- 训练：两臂都严格从F4D `best_progression.pt`开始，不读取F4E权重；code初始化seed20260832、loader seed20260831，micro4×累积16，A损失，拓扑/decoder LR分别`3e-4/3e-5`，固定15k且不早停。
- 评测与产物：step0及每1k重评800个fixed fixture，13k/14k/15k必须逐点通过；同时记录全局保护、10类Mask、97 feature、zero/cross-window/cross-motion整组code依赖、逐step采样hash、checkpoint读回及训练/gate/optimizer/Mask/code依赖SVG。execution、quality与comparison marker相互独立。
- Windows验证：posterior相关60项测试全部PASS；完整发现119项中116项PASS，3个既有模块仅因Windows缺`h5py`无法导入。全部config JSON、Python compile、CLI help、Shell语法和diff check通过；真实HDF5/CUDA尚未执行，因此只能标记READY。
- 唯一下一步：用户预先同步代码后，在Ubuntu只执行G8 2-step smoke；回传summary、marker、source commit、初始化/checkpoint hash及step0/step2指标并审核。G8 smoke通过后才执行T129 smoke，不并行启动正式训练。

### 2026-09-07 15:00 — F4E正式 FAIL（execution PASS）

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_autodecoder_f4e_20260907_120414`；源码`tiny-model@cfb6735b54f56e49977665948397f80855987d74`；完成20,000 optimizer step，最终stage为E2；marker为`cvae_posterior_autodecoder_execution.ok`与内容`QUALITY_FAIL execution_complete=true E1=false E2=false`的`cvae.failed`。
- 隔离合同：E1为5k code-only，E2为15k code+decoder；两个阶段的encoder/posterior/prior调用数均为0，79个冻结参数均无梯度。E1/E2 checkpoint读回和全部工程合同通过，失败可形成模型诊断。
- E1：step0→5k的score仅从2067.45降至2048.60；worst State/Action RMSE最终0.39079/0.32022、max abs 20.4860、contact 99.842%、超阈值元素35.325%。只移动80个共享质心code无法让冻结F4D decoder适应该Mask不变表示。
- E2：联合适配后best位于最后step20k，score20.9413；worst State/Action RMSE为0.029776/0.020474，continuous max abs 0.209413，contact 100%，超阈值元素19.679%。相对E1末尾，三项worst指标分别改善约92.4%、93.6%和99.0%，但仍距离progression阈值2.98×、2.05×和20.94×。
- 平均与依赖：全局State/Action RMSE由MSE开方为0.008951/0.007856，已低于1e-2；zero、cross-window及cross-motion错误相对正确code为95.65×、94.85×及119.93×，说明decoder强烈使用并区分window code。失败集中在均匀精确还原，而不是code被忽略。
- Mask结构：`full_both`最差，State/Action/max abs为0.029776/0.020474/0.209413；`full_state` State为0.010449。其余8类worst RMSE均低于1e-2，但10类的max abs均在0.03908–0.20941，且全局19.679%元素超阈值，所以不是单个异常点，不能只删除`full_both`或放宽一个max-abs门禁来宣称完全记忆。
- 与F4D源比较：E2全局State/Action RMSE约改善2%–3%，max abs改善约2.9%、超阈值比例改善约6.6%，但worst State/Action RMSE反而恶化约29%/34%。F4E只排除了“posterior encoder是唯一根因”，尚不能证明单256维global code理论上不足。
- 唯一下一步：实施F4F配对诊断。保持4 motion、80 windows、800 fixed Masks、A损失与15k预算不变，对比8×256 global memory tokens（每window 2,048 code标量）和129×16 per-time codes（每window2,064标量）；两者代码忆容量差0.78%。不再追加F4E训练，不改门禁，不进入32-motion/R128/KL。

### 2026-09-07 11:58 — F4E smoke v2 PASS（仅工程合同）

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_autodecoder_f4e_smoke_20260907_115809`；源码`tiny-model@dffc0bf25aa0db21e06126e7e3dafba9689e4230`；marker为`cvae_posterior_autodecoder_smoke.ok`。
- 源与数据：F4D八项指标复现全部PASS；固定4 motion、T128、80 windows和800 fixtures；fixture、window及identity合同hash均已记录。
- 模型：基础25,453,411参数，加80×256共享code后共25,473,891；`shared_code_per_window=true`、`per_fixture_code=false`、decoder路径绕过encoder，KL/dropout/weight decay均为0。
- 初始化：800个Mask-conditioned posterior编码hash为`701ebf6aa15479d647ec33582975cf005a7971b438c1af4be12042a0bc94d1f6`，80个质心hash为`f929e4b359b16974f886a3e0f1f08926e5cdc6c0d40729ff13c44ae83b2428a9`；同window十个posterior mean到质心的global RMS为1.31366、最大绝对差10.49436。这说明现有posterior编码明显依赖Mask，但这里只是待诊断现象，不提前判定根因。
- 执行与隔离：E1完成2个smoke step；E1 checkpoint readback的13项检查全部PASS。训练器在写`execution_pass=true`前强制要求encoder/prior/posterior调用计数为0且相关梯度为空，因此本次成功退出同时证明隔离合同通过。用户查询中的`training_contract`为null来自jq嵌套字段写法错误，不是manifest缺失。
- 语义：`quality_pass=false`和不生成E1/E2质量marker符合2-step smoke合同；根因字段正确为`SMOKE_EXECUTION_ONLY_NO_ROOT_CAUSE_ASSESSMENT`，不得据此判断256维code或decoder容量。
- 唯一下一步：从同一个F4D `best_progression.pt`全新启动正式F4E，不使用smoke checkpoint；E1固定5k，只有E1失败才进入E2最多15k。

### 2026-09-07 11:44 — F4E smoke v1 REPORTING_FAIL（工程执行完成）

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_autodecoder_f4e_smoke_20260907_114405`；源码`tiny-model@5cf48cac8b4289b75d44c148b26513f1d2510c21`。
- 已知执行事实：Shell报告`execution complete`、`smoke=true`、完成2 optimizer steps并成功返回run路径；因此其要求的`cvae_posterior_autodecoder_smoke.ok`已存在。`quality_pass=false`是2步smoke固定语义，不是质量失败。
- 报告缺陷：输出把`quality_pass=false`误映射为`E1_E2_FAIL_GLOBAL_CODE_DECODER_CAPACITY_UNPROVEN`。E1正式需要5k，E2在smoke中根本未执行，所以该root-cause字段逻辑上无效，不得写入模型结论。
- 修复：提交`6d7111d9f5cddd9aac7cd6950ffae00484b7fae4`将正式根因决策与smoke分支隔离；smoke固定写`SMOKE_EXECUTION_ONLY_NO_ROOT_CAUSE_ASSESSMENT`，并以单元测试覆盖。posterior/F4E相关54项测试全部PASS；全发现107项中104项PASS，另3项仅受既有Windows缺`h5py`限制。历史run只读保留，不修改其manifest。
- 尚待核验：完整summary中的F4D source reproduction、A/B/C STOP trigger重验、80/800身份、质心hash、encoder零调用/零梯度、checkpoint readback和实际marker列表尚未回传，因此不启动正式run。
- 唯一下一步：同步包含`6d7111d`的Windows HEAD，从同一F4D源重新运行2-step smoke；回传完整审计字段后更新本台账。

### 2026-09-07 — I-F4E Windows实现 READY

- 范围：实现提交为`c54f0ce4c6da166cfe7b70422302bdc454d805e7`；新增独立F4E训练器、固定配置和两个Shell入口；posterior模型只增加向后兼容的decoder-only方法，原`forward()`、F4D/F4A/F4B checkpoint与入口不变。
- 固定诊断：4 motion、T128、80 windows、800 fixed fixtures；从F4D加载原A结构，以每window十个Mask-conditioned posterior mean的质心初始化一个共享256维code，明确拒绝per-fixture code。
- 两阶段：E1仅训练20,480个code参数5k；失败才执行E2 code+decoder侧最多15k，并用不同seed重置loader、optimizer与scheduler；encoder及q/p头通过参数allowlist、调用计数和梯度状态三重隔离。
- 验收：保留原progression/exact、全局与尾部、97 feature、10 Mask、固定速度案例；新增zero/cross-window/cross-motion code依赖均至少10的质量门禁，execution/E1/E2 marker相互独立。
- 产物：保存800个原始q、80个质心、Mask间离散度及hash，逐step JSONL、四张SVG、donor身份和四类checkpoint；最终A/B/C STOP comparison会在训练前重算验证。
- 当前验证：posterior/F4E相关组合53项全部PASS；完整发现运行106项，其中103项PASS，另3个仍仅因既有Windows环境缺`h5py`而导入失败。Python compile、全部JSON、CLI help、Shell语法和diff check均通过；尚未读取真实HDF5或运行CUDA，因此只能标记READY。
- 唯一下一步：提交并同步到Ubuntu，运行`posterior-capacity-autodecoder-smoke`；先回填smoke summary、marker、source commit与checkpoint读回，再决定是否启动正式F4E。

### 2026-09-07 — F4B-v2 A/B/C正式比较 PASS（execution，路线停止）

- 运行：`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260907_102553`；源码`tiny-model@8fbe327c487c66482ba4ece6912c8f76bab3720b`，marker为`cvae_posterior_ab_comparison.ok`，`execution_pass=true`、`comparison_phase=initial`。
- 身份：A/B/C summary SHA256分别为`aaaa4173ff86137ea5bf463542c7446582ac236636268833ca00d455e951619c`、`deff24d26b2da39f906f5a9e549736921fc41434bcbc8fdcc900112923522db2`、`a5ea229cf5c6352c52c7177b25073f0bb27f7ccf1dd05148b1fdeaf219ad2f7a`；三支的fixture/optimizer seed、dataset、窗口、fixture、源checkpoint、F4A、逐step训练身份与决策点均配对通过，C结构和触发来源也通过。
- B结论：`R_p=1.1198187`、`R_a=0.9590181`，即超阈值比例恶化11.98%、max abs只改善4.10%；guards通过但progression、strong improvement和worth replicating均为false。
- C结论：`R_p=0.9411712`、`R_a=1.2594459`，即超阈值比例只改善5.88%、max abs恶化25.94%；8k/9k/10k的worst State均超过A的110%保护线，故guards、progression、strong improvement和worth replicating均为false。
- 正式决定：`STOP_LOSS_LATENT_SEED_SEARCH`，原因为B/C均未取得受保护的20%双指标改善；`next_optimizer_seed=null`且`replication_arms=[]`。因此F4R被取消，禁止追加C 20k或启动第二seed。
- 研究边界：该决定只终止本轮“B尾部损失、C逐层latent gate、重复seed”的局部搜索，不证明Transformer、单global latent或CVAE原则上无法记忆。32-motion fixed、R128与KL仍因4-motion未通过而冻结。
- 唯一下一步：暂停训练并做方向审查。优先候选是独立auto-decoder诊断：用每窗口可学习256维code替代posterior encoder、保留同一decoder/Mask，以一次实验区分“encoder无法形成可解码code”和“global code+decoder本身不能记忆”；用户确认前不实现。

### 2026-09-07 — F4C正式10k FAIL（quality）

- 运行：`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_c_seed20260830_20260907_004301`；源码`tiny-model@163be1f4c40c46bbd5c680b7ac1711e87f382e97`，完成10,000 steps；`execution_pass=true`、`quality_pass=false`，保留`cvae_posterior_ab_execution.ok`与`QUALITY_FAIL`。源码状态仅含两个既有嵌套仓库未跟踪项。
- 最佳/最后点：step 10000，score 17.189798；worst State/Action RMSE 0.0190393/0.0139081，max abs 0.171898，global State/Action RMSE 0.00819649/0.00732956，17.3789%连续目标超`1e-2`，contact 100%、zero ratio 123.268。
- 正式决策点：8k/9k/10k score依次17.4278/17.2959/17.1898，均FAIL；超阈值比例17.6342%/17.4665%/17.3789%，仍缓慢下降但远离完整门禁。所有gate梯度存在，3,072个参数全部非零，最终总L2为0.019783；故失败不是gate未更新。
- 完整性：checkpoint readback全部通过，last SHA256为`bb3bdc6b2a6686b89c0e39cfa8e9e598417b5aee183954cb90c658ebed8a1e2f`。
- 人工配对诊断：以三点中位数计算，C/A为`R_p≈0.941171`、`R_a≈1.259446`；超阈值比例仅改善5.88%，max abs反而恶化25.94%。worst State中位数为A的1.1912倍，超过110%保护线；因此按冻结规则预期不具备复核资格，但最终决定必须由比较manifest给出。
- 边界与唯一下一步：只运行显式A/B/C比较器，核验全部身份并生成唯一决定；不得把缓慢下降当作追加20k的授权。

### 2026-09-07 — F4C smoke PASS（execution）

- 运行：`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_c_seed20260830_smoke_20260907_003414`；源码`tiny-model@443f4782c823fc7adaa8730e7513cd7671b630e6`，marker为`cvae_posterior_ab_smoke.ok`；源码状态仅含两个既有嵌套仓库未跟踪项。
- 合同与触发：`execution_pass=true`、`smoke=true`、2 optimizer steps、25,456,483参数，其中8层gate共3,072参数；comparison manifest SHA256为`4eac018d95bb313e5126ff871ceafadd270910b61a2b205b57db0276e3f963f5`，其A/B summary哈希、配对和`IMPLEMENT_F4C`决定均重新验证通过。
- 初始化与等价：旧checkpoint只缺失预期8个gate key、无unexpected key；8层gate初始范数与非零数均为0。step0完整复现F4D：State/Action worst RMSE 0.0230213963/0.0152626950、max abs 0.2156203091、contact 100%、global State/Action RMSE 0.0091486114/0.0081180074。
- gate更新：全部gate梯度存在且观测到非零梯度；step2后3,072/3,072参数非零，总L2范数`1.38587e-5`，各层L2约`4.72e-6`至`5.04e-6`。checkpoint全部格式、来源、fixture、参数量、gate key/shape/有限性和optimizer/scheduler检查通过，last SHA256为`8c284021a7cd884d222e69fd2756b3e969cd6463e80aa95cdecefa10a65d73ac`。
- 2-step诊断：score 21.560766，worst State/Action RMSE 0.023019776/0.015262693、max abs 0.215607658、contact 100%、超阈值比例21.0771%、zero ratio 110.241。变化极小且`quality_pass=false`为smoke预期，不能解释为F4C有效或无效。
- 边界与唯一下一步：smoke仅证明真实工程链路和结构可训练。正式C必须从原F4D `best_progression.pt`重新初始化、使用同一trigger和optimizer seed 20260830跑满10k；不得加载smoke checkpoint或提前启动第二seed。

### 2026-09-07 — I-F4C Windows实现 PASS

- 触发：正式比较run `/home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260906_235429` 的`execution_pass=true`、全部配对检查、comparison marker与`decision=IMPLEMENT_F4C`已回传；实现入口要求显式提供该run并重新验证manifest、A/B summary哈希、数据/来源、配对及决定。
- 结构：只在`PosteriorCapacityTransformerCVAE`配置开启时创建8个384维零初始化gate；每个decoder block前向有效State/Action token加`gate × latent_projection(z)`，不作用于latent token或padding，不修改通用`TransformerStack`。A/B关闭时参数与state dict不变；C总参数25,456,483。
- 迁移与训练：C继续使用A原始MSE/BCE，从F4D `best_progression.pt` model-only初始化；加载仅允许缺失精确的8个gate key且验证全零，任何其他missing/unexpected key失败。每步记录逐层gate范数、max abs、裁剪前梯度和更新后非零数；评测、summary与checkpoint记录结构及触发来源。
- 防护：C只接受optimizer seed 20260830和有效的`IMPLEMENT_F4C`初始比较；触发参数传给A/B会拒绝。保存checkpoint回读验证8个gate key、shape、有限性、结构开关和25,456,483参数；A/C比较额外验证C结构与触发记录。
- Windows验证：posterior/model/tail/plot/AB组合42项通过，涵盖零gate输出逐位一致、有效token限定、真值隔离、gate梯度/更新、严格迁移、checkpoint gate读回、触发哈希及比较分支；参数断言A/B 25,453,411、C 25,456,483，compile、CLI、Shell语法和diff check通过。完整发现运行95项，只有3个既有模块因Windows缺`h5py`导入失败。
- 实现提交：`c965487a5291ada3919db74b017900a3a04eb6ab`。
- 验证结论：上述Ubuntu smoke已补齐真实HDF5/CUDA、step0等价、gate更新和checkpoint roundtrip，故实现项由READY转为PASS；模型质量仍未验收。
- 唯一下一步：从原F4D源独立执行C正式10k。

### 2026-09-07 — F4B-v2 A/B正式比较 PASS（execution）

- 运行：`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260906_235429`；源码`c1ae5f79111bf61ddace32073df8122dbbefec95`，marker为`cvae_posterior_ab_comparison.ok`，SVG为`plots/posterior_ab_comparison.svg`。
- 验证：`execution_pass=true`、`comparison_phase=initial`；fixture/optimizer seed、dataset、窗口、Mask、固定案例、source/F4A、25M模型、10,000 step样本身份和8k/9k/10k存在性等13项配对检查全部通过，guards PASS。
- 结果：B/A三点中位残余比例为`R_p=1.1198186676`、`R_a=0.9590180833`；B未通过progression，`strong_improvement=false`、`worth_replicating=false`。
- 决定：manifest正式输出`IMPLEMENT_F4C`，原因是B不满足受保护的progression/双指标50%改善规则；不启动第二seed，不把B checkpoint用于C。
- 唯一下一步：在Windows实现并轻量验证F4C，随后先运行独立C smoke。

### 2026-09-06 — F4B-v2 B正式10k FAIL（quality）

- 运行：`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_b_seed20260830_20260906_203844`；源码`c1ae5f79111bf61ddace32073df8122dbbefec95`，4 motion、T128、80 windows、800 fixtures，完成10,000 optimizer steps。
- 状态：`execution_pass=true`、`quality_pass=false`；checkpoint readback全部通过，保留`cvae_posterior_ab_execution.ok`与`QUALITY_FAIL`。源码状态只有两个既有嵌套目录未跟踪。A/B全部10,000 step训练身份SHA256及dataset/source/window/fixture合同diff均为空，满足配对比较前提。
- 最佳/最后点：step 10000，score 13.056414；worst State/Action RMSE为0.016398/0.014550，max abs为0.130564，contact 100%，zero ratio 120.443；global State/Action RMSE为0.008444/0.007629，连续元素超阈值比例20.718%。
- 正式决策点：8k/9k/10k全部FAIL。最后三点中位数相对A为`R_p=1.11982`和`R_a=0.95902`；即超阈值比例恶化11.98%，max abs仅改善4.10%。worst State/Action中位数也分别恶化2.41%/2.14%，global State/Action中位数恶化0.15%/1.05%，但仍处于10%保护范围，contact和zero-ratio保护成立。
- 初步判断：A未PASS，B也未PASS且两项残余比例远高于0.50，故按冻结决策表预期应触发`IMPLEMENT_F4C`；这只是人工复核，最终唯一动作必须由正式比较manifest给出。
- 唯一下一步：只运行显式A/B比较器；回填`posterior_ab_comparison.json`、comparison marker和decision后，若确为`IMPLEMENT_F4C`才回到Windows实现C。

### 2026-09-06 — F4B-v2 A正式10k FAIL（quality）

- 运行：`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_a_seed20260830_20260906_114634`；源码`c1ae5f79111bf61ddace32073df8122dbbefec95`，4 motion、T128、80 windows、800 fixtures，完成10,000 optimizer steps。
- 状态：`execution_pass=true`、`quality_pass=false`；checkpoint readback全部通过，保留`cvae_posterior_ab_execution.ok`与内容为`QUALITY_FAIL execution_complete=true progression_last_three=false`的`cvae.failed`。源码状态只有两个既有嵌套目录未跟踪，没有观测到CVAE源码差异。
- 最佳/最后点：step 10000，score 13.721974；worst State/Action RMSE为0.015999/0.014238，max abs为0.137220，contact 100%，zero ratio 122.437；global State/Action/combined RMSE为0.008425/0.007545/0.008173。
- 正式决策点：8k/9k/10k全部FAIL；其超`1e-2`元素比例为18.737%/18.558%/18.456%，max abs为0.138204/0.137329/0.137220，且每个点均为800/800 fixtures存在max-abs超阈值。
- 相对源F4D的描述性比较：最后三点中位数的超阈值比例下降11.95%，max abs下降36.31%，worst State/Action下降30.34%/6.51%；这表明A继续优化但不足以通过门禁。A/B主比较尚不能计算，必须等待B的同step配对结果。
- 唯一下一步：从相同F4D `best_progression.pt`、optimizer seed 20260830独立启动B正式10k；不得继承A checkpoint。B完成并回填后才运行比较器。

### 2026-09-06 — F4B-v2 A/B Ubuntu smoke PASS（execution）

- 运行：A为`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_a_seed20260830_smoke_20260906_112939`，B为`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_b_seed20260830_smoke_20260906_113154`；两支源码均为`c1ae5f79111bf61ddace32073df8122dbbefec95`。
- 固定合同：两支均为4 motion、T128、80 windows、800 fixtures、25,453,411参数、fixture/optimizer seed均为20260830；dataset、源checkpoint、窗口、Mask位图、固定速度案例和身份合同hash一致。
- 复现与执行：两支`execution_pass=true`、`smoke=true`；step0与legacy source reproduction的全部检查为true，2个optimizer step的训练身份SHA256逐步完全相同，checkpoint readback全部检查通过，marker均为`cvae_posterior_ab_smoke.ok`。
- 质量边界：A/B的step-2 progression score分别为21.5601/21.5606，`quality_pass=false`。smoke仅验证真实HDF5/CUDA、A/B loss路径、评测、诊断与落盘读回，2步结果不得用于判断A/B优劣或posterior容量。
- 唯一下一步：从同一F4D `best_progression.pt`重新初始化并完整运行A正式10k；回填A的summary、8k/9k/10k评测、source状态和marker后，再运行B正式10k。

### 2026-09-06 — I-F4B-v2 Windows实现 READY

- 范围：新增`posterior_capacity_ab.py`、固定A/B配置、三个Shell入口和独立测试；保留历史posterior/F4A入口与模型结构，不实现F4C、不启用KL/reference、不启动Ubuntu训练。
- 训练合同：A/B均严格加载同一F4D `best_progression.pt`，fixture seed固定20260830，optimizer seed独立；正式固定10k、每1k全评测、不早停。A完全调用原reconstruction loss；B仅按fixture/domain元素数加权混合MSE与top-20%平方误差。
- 复现与记录：更新权重前核对F4D step、80 window、800 fixture、Mask位图和全部源指标；每步写采样身份SHA256、原loss/实际目标、LR、裁剪、耗时与显存；每次评测写全局/worst/tail、97连续feature、contact、5个冻结速度案例和真实cross-window/cross-motion donor映射。
- 状态语义：2-step smoke、正式执行和比较各有execution-only marker；正式质量失败仍以0退出保留完整summary、execution marker及`QUALITY_FAIL`，quality PASS另写既有progression marker。比较器只使用8k/9k/10k中位数及逐点保护条件，拒绝身份不配对的run。
- Windows验证：新增A/B测试与既有posterior/tail/plot组合共37项通过；全发现中除3个既有模块仅因Windows未安装`h5py`导入失败外，其余均通过。25M参数断言为25,453,411，Python compile、CLI help、Shell语法及`git diff --check`通过。Windows未读真实HDF5/CUDA，故本项只能写READY，不能写实验PASS。
- 唯一下一步：同步到Ubuntu后串行执行optimizer seed20260830的A smoke与B smoke；两支step0复现、checkpoint readback及smoke marker均通过后，先回填本台账再启动正式A/B。

### 2026-09-05 — F4B-v2计划确定（仅文档；实验PENDING）

- 范围：更新本概览、新建Next.md并同步AGENTS.md；此次未修改训练源码、未执行Ubuntu训练。
- 科学判断：小规模posterior记忆可行；多motion失败尚不能单独归因于loss、global latent容量或decoder平滑。固定正式门禁，使用有界受控干预选择下一项研究。
- 协议修订：旧F4B加法CVaR/各25k标记SUPERSEDED；改为归一化尾部混合A/B各10k，可选C 10k，配对优化seed复核20k，最多50k正式训练步。
- 源码发现：batch=4的历史full-both swapped donor是同窗口不同Mask；旧数值不改，取消跨窗口区分解释；新实验增加真正跨窗口/跨motion donor。
- 实施状态：拟定入口、分离fixture/优化seed、比较器与C结构均未实现。零初始化门控只做过不落盘CPU合成可行性检查，不能表述为Ubuntu验证成功。
- 唯一下一步：按Next.md实现并轻量验证F4B-v2 A/B及其step0复现、完整统计与诊断身份；其后再执行独立Ubuntu smoke，更新台账后运行正式A/B。

### 2026-09-05 20:08 — F4A PASS（execution-only）

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_tail_diagnostic_f4a_20260905_200807`；源码`2ff1ec95db72fed9db80d3b040cccc32b3f9703f`；正式marker为`cvae_posterior_capacity_tail_diagnostic.ok`。先前一次`echo "$RUN"`指向F4D旧目录只是Shell变量选错，不能替代本条最终身份。
- 源复现：F4D step 100,000、800 fixtures、worst State/Action fixture RMSE 0.0230213969/0.0152626948、max abs 0.2156203091、contact 100%、global State/Action RMSE 0.0091486114/0.0081180074全部与源summary通过容差检查；诊断结果可信。开头的`^C`只是退出监控，不表示F4A计算失败。
- 全局分布：共3,775,101个continuous masked targets，其中795,690个绝对误差超过`1e-2`，比例21.077%；800/800 fixtures均至少有一个元素超过max-abs门禁。global combined RMSE为0.008854，说明“均方平均通过”和“逐元素全部通过”存在显著差距，但失败绝非少于1%的孤立异常点。
- 自动分类：`partial_p95_pass=true`且没有partial p50/p90 broad failure，但`tail_concentrated=false`，因此为`mixed_tail_and_reconstruction_failure`；`full_both` p95为0.015529，partial宏平均p95为0.008391，比值1.851，小于预设3倍，`global_latent_bottleneck_suspected=false`。
- 最差窗口：前5名中4个来自`big_heavy_one_hand_front_low_to_front_medium_R_001__A526`的variant 4/6/7与start 128/197，另一个为`body_stretch_4_002__A054` variant 3；最差窗口有30.19%的连续元素超阈值。困难与特定高动态动作/片段相关，而非均匀分布于全部motion。
- 最差feature：前5名全部是joint velocity——right ankle pitch、right shoulder roll、right knee、right hip yaw、left wrist yaw；各自约26.6%–30.0%的元素超阈值。最大物理误差分别可达0.236、0.138、0.299、0.107、0.086 rad/s。原记录据此提出速度峰值平滑假设；本轮解释修订为需由固定真值/预测曲线检验，尚不能确认机制。
- 后续计划（当时，已被F4B-v2替代）：F4B同起点目标A/B，原MSE与加法CVaR-20；未执行。当前下一步以第3.2节和Next.md为准，不启动R128。

### 2026-09-05 — I-F4A Windows实现 PASS

- 范围：只新增F4A只读诊断，不修改`PhysicsPosteriorTransformer`、训练损失、Mask生成、checkpoint内容、R128或KL接口。
- 合同：入口只接受4 motion、T=128、25,453,411参数、fixed/progression、KL=0、seed 20260830且源run失败的`best_progression.pt`；同时严格校验dataset hash、80个window身份、800个fixture、10类Mask和F4D summary。
- 产物：逐fixture JSONL、逐window聚合JSONL、97个continuous feature聚合JSONL、contact错误JSONL、最差20个fixture/window/feature、分位数与超阈值集中度、`posterior_tail_diagnostic.svg`以及总manifest；新run的`checkpoints/`保持为空。
- 复现门禁：诊断所得worst State/Action RMSE、max abs、contact、全局State/Action MSE必须与F4D summary匹配，否则失败且不生成marker；marker仅表示诊断执行完整，不代表模型质量PASS。
- 轻量验证：F4A与posterior相关组合测试19项全部通过；完整Windows discovery共72项，其中69项通过，3项仅因既有Windows Python缺少`h5py`而导入失败；Python compile、CLI help、Shell语法均通过，未安装或修改依赖。
- 后续计划：唯一下一项是在Ubuntu执行F4A并回传`posterior_tail_diagnostic.json`；根据`tail_objective_mismatch`、`broad_reconstruction_failure`与`global_latent_bottleneck_suspected`的实际分类，再决定下一次只改变一个因素的实验。

### 2026-09-05 — F4D FAIL

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425`；source HEAD为`6463b2ec960cda22c7ed70814a46a44e6804d4c0`，exit code 1，marker为`cvae.failed`。
- 初始化：严格从L128 step 94,000的`last.pt` model-only warm-start；source为1 motion、T=128 fixed，checkpoint SHA256为`35830ba06097b0f85c40b36c63c8b6287369c839ab688d05a96410c2c9d0d673`；optimizer、scheduler和RNG均未恢复。
- 配置与执行：4 motion、T=128、80个window、800个fixed fixtures、25,453,411参数、KL为0；跑满100,000 step，约见到6,400,000个fixture samples和8,000个fixture epoch。最佳点就是step 100,000；保存291MB `best_progression.pt`与291MB `last.pt`。
- 最佳结果：progression score 21.5620；worst State/Action RMSE为0.023021/0.015263，max abs为0.215620，contact 100%；zero/swapped latent ratio为110.24/38.59。全局masked-element State/Action MSE为8.370e-5/6.590e-5，对应聚合RMSE约0.00915/0.00812，二者已经低于`1e-2`。
- 分层结果：`full_both`最差，State/Action RMSE为0.02302/0.01526、max abs 0.21562；`full_state` State RMSE为0.01134；其余8类Mask的worst RMSE均不高于`1e-2`或仅接近阈值，但所有Mask的worst max abs仍高于`1e-2`。这说明RMSE失败集中于稠密遮挡，而max-abs失败遍布稀少尾部。
- 趋势与结论：step 91,000至100,000的score由24.03单调降至21.56，State/Action RMSE由0.02563/0.01615降至0.02302/0.01526；尚在缓慢改善，但距离max-abs门禁仍有21.6倍。latent依赖强，不能仅凭本run认定global latent失效；训练的全局平均MSE与验收的worst-window/max-element指标明显错位，直接续训或扩模型都不是信息量最高的动作。
- 后续计划：唯一下一项为F4A只读尾部诊断。固定读取F4D `best_progression.pt`，输出逐fixture/feature分位数、最大误差身份和超阈值集中度；完成前不启动新训练、不修改门禁、不执行R128或KL。

### 2026-09-04 — F128 FAIL

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m32_t128_25m_s200000_gprogression_20260902_235140`；source HEAD为`6463b2ec960cda22c7ed70814a46a44e6804d4c0`，exit code 1，只有`cvae.failed`。
- 配置：32 motion、256 episodes、T=128、816个固定window、8,160个fixed fixtures、25,453,411参数、KL为0；summary摘录未包含`initialization`字段，启动F4D前仍须从完整summary确认本run确由L128 `last.pt` model-only初始化。
- 执行：跑满200,000 optimizer step，按effective batch 64共见到12,800,000个fixture samples，约1,568.6个fixture epoch；保存291MB `best_progression.pt`与291MB `last.pt`。最佳活动score在step 175,000，训练没有提前停止。
- 最佳结果：progression score 964.4091；worst State/Action RMSE为0.261443/0.273665，max abs为9.644091，contact为99.9958%；correct/zero/swapped latent RMSE的比值为1/12.059/7.905。全局masked-element reconstruction total/state/action/contact为0.003114/0.006388/0.002829/0.0001267。
- 分层结果：`full_both`最差，State/Action RMSE为0.2614/0.2737、max abs 9.6441；`full_state`与`state_time_50`的max abs仍为1.9893/1.8989；即使较容易的`action_time_50`，Action RMSE与max abs也为0.04278/0.2790。失败不是单一Mask或单个contact项造成。
- 趋势：177,500至200,000 step的State/Action RMSE约稳定在0.239–0.260，max abs约9.72–11.52；最后step score 981.97。末段没有向门禁数量级收敛，不能用直接延长同一cosine schedule解释为“差一点”。
- 事实结论：当前25M模型、平均masked reconstruction目标和直接1→32 motion warm-start，在200k预算下不能完成32-motion fixed记忆；zero-latent ratio刚通过说明latent被使用，full-both灾难性失败值得检查latent与decoder通路。历史swapped ratio 7.91按第1.2节只能解释为batch内替换，不能用于判断跨窗口区分能力。部分Mask也明显失败，因此尚不能把问题仅归因于无可见条件的full-both；本run也不能单独证明25M参数容量在理论上不足。
- 后续计划：唯一下一项为F4D——保持代码、25M模型、T=128、10类fixed Mask、seed和progression门禁不变，从L128 `last.pt`扩到4 motion，100k上限、每1,000 step验收。F4D PASS才允许用其`last.pt`重跑F128；FAIL则停止训练并审计目标聚合与latent通道。R128和KL继续冻结。

### 2026-09-02 — L128 PASS

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t128_25m_s100000_gprogression_20260901_173153`。
- 代码：run的`source_commit.txt`与`source_status.txt`尚未回传，本记录不从Windows工作树反推Ubuntu HEAD。
- 配置：1 motion、8 variants、T=128、全部24个固定window、240个fixed fixtures、seed 20260830、25,453,411参数；random init、posterior mean、KL/dropout为0。
- 执行：上限100,000 step，实际在94,000 step因连续3次progression PASS提前停止；按effective batch 64计约见到6,016,000个fixture samples，约25,067个fixture epoch。`best_progression.pt`选择step 93,500；Shell打印PASS与run路径并返回提示符，因此按入口合同确认progression marker检查通过。
- 最佳结果：progression score 1.0；worst State/Action RMSE为0.00230210/0.00153279，max abs 0.009941995，contact 100%；correct/zero/swapped latent RMSE为0.00182971/0.859160/0.152794，zero/swapped ratio为469.56/83.51。完整评测reconstruction total/state/action/contact为1.339e-6/2.758e-6/1.199e-6/5.906e-8。
- 分层结果：`full_both`同时给出最差State RMSE、Action RMSE和max abs；`element_both_50`次之。其余element/time/feature/semantic Mask均处于同一可推进量级，10类fixture全部通过progression。
- 事实结论：25M模型能够在单motion、128-transition长窗口上记忆全部训练window和10类固定Mask，并强依赖posterior latent；exact score仍为23.0210，不能称为完美或数值无损，也没有验证新Mask、conditional prior或未见motion泛化。
- 后续计划：唯一下一项为F128。只加载本run的step 94,000 `last.pt`模型参数，重置optimizer、scheduler和RNG，直接扩到32 motion、T=128；R128通过前继续保持KL接口未实现。

### 2026-09-01 — I25 Windows入口 READY

- Run：N/A；尚未执行Ubuntu真实数据训练。
- 代码：基于外层`tiny-model@fa50f444d9a481b0f431edc1b1f974f0998f623a`的当前Windows工作树；未修改SONIC或IsaacLab。
- 模型：新增固定25,453,411参数配置，保持70维State、29维Action、256维单一global posterior mean、dropout/KL为0及纯Transformer输入合同。
- 训练协议：新增`posterior-capacity-25m[-smoke]`、model-only规模warm-start、max step/完整评测间隔覆盖；warm-start不恢复optimizer、scheduler或RNG。
- 指标与图：完整fixture按masked元素聚合State MSE、Action MSE和contact BCE；每次评测刷新三张SVG，明确log10轴刻度与fixed/held-out评测范围。
- Windows验证：35项posterior/model/isolation关键组合测试通过；全发现共65项通过，另3个既有模块仅因Windows缺`h5py`导入失败；32份config JSON、Python compile、Shell语法和`git diff --check`通过。
- 事实边界：只证明代码入口和轻量逻辑一致，尚未证明真实HDF5、CUDA显存、训练速度、loss收敛或任何质量门禁。
- 后续计划：唯一下一项为S25；只有其工程marker与25,453,411参数断言、三张SVG都存在后才执行L128。

### 2026-09-01 17:20 — S25 PASS

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t8_25m_gprogression_20260901_172054`。
- 代码：run的`source_commit.txt`与`source_status.txt`尚未回传；不得从Windows工作树反推Ubuntu HEAD。
- 配置：1 motion、T=8、25,453,411参数、fixed、progression、posterior mean、KL/dropout为0；smoke固定2 optimizer step。
- 执行：Python打印完整summary及run路径后返回Shell提示符；依据Shell合同，这同时确认`cvae_posterior_capacity_smoke.ok`存在。`best_progression.pt`与三张SVG路径均已写入summary。
- 结果：本次未回传State/Action RMSE、max abs、contact及latent ratio；smoke无论质量指标如何都只作工程诊断，不补写或猜测数值。
- 事实结论：Ubuntu真实HDF5/CUDA、25M参数范围断言、forward/backward、完整fixture evaluator、checkpoint、summary和SVG链路可执行；不证明任何拟合或门禁质量。
- 后续计划：唯一下一项为L128，从随机初始化训练1 motion、T=128，100k上限、每250 step完整progression验收。

### 2026-09-01 — Progression gate Windows入口 READY

- Run：N/A；Ubuntu尚未执行新门禁run。
- 实现：新增`CVAE_POSTERIOR_GATE=exact|progression`；progression阈值固定为State/Action normalized RMSE与continuous max abs均`1e-2`，contact 100%，zero-latent ratio至少10。
- 隔离：每次validation同时保存exact/progression两套score；活动门禁独立控制连续3次PASS、`best_<gate>.pt`、退出码和marker，旧exact默认行为不变。
- 验证：posterior/model/isolation组合32项通过；Python compile、全部config JSON、Shell语法、CLI help和`git diff --check`通过。Windows全发现测试另有3项因既有环境缺`h5py`无法导入，与本改动无关。
- 历史解释：W4最后5次validation均满足progression阈值，可作为进入P1的证据；它仍是exact FAIL，且因旧代码没有progression marker，不追记正式marker。
- 精简决策：取消W4-E80、W16、W64的预先执行；P1直接覆盖1 motion全部144窗口，P1失败时才恢复窗口边界诊断。
- 后续计划（当时）：唯一下一项为P1；该阶段已经完成，当前由I25记录后的S25计划接管。

### 2026-09-01 — P1 PASS

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_s100000_gprogression_20260901_105145`；源码HEAD尚未从`source_commit.txt`回传。
- 配置：1 motion、8 variants、全部144个window、window16、1,440 fixed fixtures、seed20260830、6,725,731参数；用户将训练上限由计划的80k提高到100k。
- 执行：完成96,250 optimizer step后提前停止，证明末段连续3次通过活动门禁；`best_progression.pt`选择step93,750，shell成功返回run路径，因此progression marker检查已通过。
- 最佳结果：progression score 1.0；State/Action RMSE为0.00224268/0.00145998，max abs 0.00993767，contact 100%；correct/zero/swapped RMSE为0.00145932/0.891624/0.856054，zero/swapped ratio为610.99/586.61。
- 分层结果：`element_both_10`给出最差State/Action RMSE；`full_both`给出最差max abs且仅比`1e-2`阈值低约0.62%；10类Mask均满足progression。
- 事实结论：模型已在同一motion全部fixed窗口上达到可推进的近似记忆精度并强依赖posterior latent；exact score仍为22.4268，不能称为完美拟合，也尚未证明新随机Mask或conditional prior。
- 后续计划（当时）：原定执行P2；该决定已被25M快速阶梯替代，旧P2不再单独运行。

### 2026-09-01 — W4 FAIL

- Run：绝对run路径尚未回传；命名模式为`cvae_posterior_capacity_fixed_m1_t16_w4_*`。
- 代码：`tiny-model@b3aa63d9514cd8dd284e7f6091fc26877f57f021`；training/validation Mask seed均为20260830，`fixed_fixture_identity_match=true`。
- 配置：前4个固定window、window 16、40 fixtures、参数量6,725,731、40k step fixed formal。
- 执行：跑满40,000 step并生成`cvae.failed`；最佳step 39750、score 1.6103；39k–40k score在1.61–1.67波动，当前cosine末段没有继续下降趋势。
- 最佳结果：State RMSE 1.610e-4、Action RMSE 1.194e-4、max abs 6.158e-4、contact 100%；correct/zero/swapped latent RMSE为1.094e-4/0.92137/0.92147，zero/swapped ratio为8425.42/8426.26。
- 分层结果：主要失败为`full_state` State 1.610e-4、`full_both` State 1.454e-4、`state_time_50` State 1.309e-4及`action_time_50` Action 1.194e-4；element/feature/semantic多数已显著低于阈值，max abs、contact和latent依赖全部通过。
- 事实结论：模型已经能高精度记忆4个窗口，失败集中在需要稠密输出的full/time Mask，而非逐元素Mask融合。40k时每fixture约64k次暴露，仅为D1至最佳点约136k次的一半；当前结果支持先检验训练暴露预算，不支持立即扩模型或放宽`1e-4`门禁。
- 后续计划（当时）：W4-E80。该决定已被上方新增的progression门禁记录取代；当前直接执行P1。

### 2026-09-01 — W4-E80 Windows入口 READY（已取消执行）

- Run：N/A；Ubuntu尚未执行。
- 实现：新增`CVAE_POSTERIOR_MAX_STEPS`与`--max-optimizer-steps`，只覆盖posterior capacity的训练上限；run名前缀增加`_sN`。
- 协议：设置80k时cosine按80k重算；默认不设置仍为40k，smoke无论覆盖值仍固定2 step。
- 可追溯性：summary新增`max_optimizer_steps`与`completed_optimizer_steps`。
- 验证：11项posterior测试、与模型/隔离组合共31项、Python compile、全部config JSON、Shell语法及CLI help通过。
- 后续计划（当时）：W4-E80。新增progression门禁后该严格精度追加训练不再位于关键路径，保留入口但不执行。

### 2026-09-01 — F1R FAIL

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_20260831_202012`。
- 代码：外层`tiny-model@b3aa63d9514cd8dd284e7f6091fc26877f57f021`；仅两个预期嵌套仓库目录为untracked。
- 配置：1 motion、8 variants、144 fixed windows、window 16、1,440 fixtures、seed 20260830、training/validation Mask seed一致、参数量6,725,731。
- 执行：跑满40,000 optimizer step；最佳step 39000，保存77MB `best_exact.pt`与77MB `last.pt`，生成`cvae.failed`。39k–40k score在36.41–36.81间平台化。
- 最佳结果：score 36.4095；State RMSE 0.003641、Action RMSE 0.002903、max abs 0.017851、contact 100%；correct/zero/swapped latent RMSE为0.002427/0.80568/0.85592，zero/swapped ratio为331.97/352.67。
- 分层结果：`state_feature_50`给出最差max abs 0.017851；`element_both_10`给出最差State/Action RMSE 0.003641/0.002903；`full_state/full_action/full_both`也约为0.0031/0.0027，并未显著优于partial Mask。
- 事实结论：协议正确、latent被使用且contact完全拟合，但144窗口在当前40k预算下未实现数值近零记忆。所有Mask同量级，证据不支持单一decoder Mask融合故障；末段平台说明沿当前cosine schedule直接小幅延长价值有限。D1每fixture至最佳点约重复136,000次，F1R每fixture仅约1,778次，训练暴露相差约76倍，因此尚不能把失败单独归因于参数容量。
- 后续计划：只执行W4——`max_windows=4`，其余模型、seed、Mask、阈值、40k schedule全部不变并从随机初始化开始；PASS后W16，FAIL则先诊断训练预算/目标聚合，不进入G0。

### 2026-08-31 — D1 PASS

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_w1_20260831_150654`。
- 代码：run的`source_commit.txt`尚未回传；协议字段确认training/validation Mask seed均为20260830，`fixed_fixture_identity_match=true`。
- 配置：1 motion的8 variants通过前缀校验；从144个可用window中确定性选择source window 0，即variant 0、`demo_217`、start 0；window 16、10 fixtures、参数量6,725,731、fixed formal。
- 执行：`best_metrics`位于step 34000；依据每250 step验收且需要连续3次PASS的代码合同，formal在后续两次PASS后提前结束并生成`cvae_posterior_capacity.ok`，未跑满40k。
- 最佳结果：score 1.0；State RMSE 4.719e-5、Action RMSE 8.075e-5、max abs 2.255e-4、contact 100%；correct/zero/swapped latent RMSE为4.707e-5/0.48682/0.002571，zero/swapped ratio为10342.86/54.62。
- 分层结果：State最差`state_time_50`为4.719e-5；Action最差`semantic_both`为8.075e-5；所有10类Mask均通过。score恒为1来自contact accuracy比值在100%时的定义，并非连续误差刚好压线。
- 事实结论：模型能够用单个global posterior mean，在同一窗口的全遮挡、element、time、feature与semantic固定Mask下实现严格数值近零重建；zero-latent负对照强烈失败，排除常量输出。该结果不证明新Mask、更多窗口、prior或条件方向能力。
- 后续计划：只执行F1R——取消`CVAE_POSTERIOR_MAX_WINDOWS`，同seed、同模型、同10类fixed fixture，从随机初始化训练全部144窗口；F1R前不改其他变量。

### 2026-08-31 13:51 — F1 FAIL

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_20260831_114833`
- 代码：外层 `tiny-model@fcdb4f8861e539e3ea364e578d6bc96ce7ebd9b0`；外层仅显示两个预期的嵌套仓库目录为 untracked。
- 配置：1 motion、8 variants、144个固定window、window 16、10类Mask共1,440 fixtures、seed 20260830、fixed formal、参数量6,725,731。
- 执行：完成40,000 optimizer step；最后五次validation在39k–40k基本平台化；保存77MB `best_exact.pt` 与77MB `last.pt`，生成 `cvae.failed`，无正式PASS marker。
- 最佳结果：step 40000，exact score 752.0263；worst State RMSE 0.05842、Action RMSE 0.02495、max abs 0.75203、contact 100%；correct/zero/swapped latent RMSE为0.002212/0.82213/0.85592，zero/swapped ratio为371.67/386.94。
- 分层结果：`full_state` State RMSE 0.00256，`full_action` Action RMSE 0.00239，`full_both` State/Action 0.00260/0.00247；最差partial为`element_both_50` State/Action 0.05842/0.02495、max abs 0.75203，其次是`element_both_10`与State/Action feature/semantic Mask。
- 事实结论（已修订）：global latent被模型强烈使用、contact已拟合、三个不依赖随机坐标的full Mask约为0.0025。事后代码审计发现训练使用`seed`，validation使用`seed+700001`，所以partial element/time/feature/semantic并不是训练过的具体Mask。该run测量了部分Mask坐标迁移，不能证明统一decoder无法记忆训练fixture，也不能作为fixed exact gate。
- 后续计划：修复协议后先执行D1；只有D1通过才重跑F1R，禁止直接进入G0/F2。

### 2026-08-31 — Fixed Mask seed协议修复 READY

- Run：N/A；尚未在Ubuntu运行修复后的真实数据实验。
- 根因：fixed训练调用`make_fixture_masks(batch, seed)`，旧validation却传入`seed+700001`；三个full Mask不受影响，七类partial Mask的具体坐标发生变化。
- 修复：fixed validation复用训练seed；generalization继续使用独立`seed+700001`。summary新增`training_mask_seed`、`validation_mask_seed`、`fixed_fixture_identity_match`。
- 验证：新增回归测试逐位比较fixed训练/validation State与Action Mask，并验证generalization seed保持独立；10项posterior测试、与模型/隔离组合共30项、Python compile、全部config JSON和Shell语法均通过。
- 结论边界：旧F1保留且不删除，但降级为Mask坐标迁移诊断；不能再用它支持decoder融合缺陷结论。
- 后续计划：执行corrected D1 smoke与formal；D1 PASS后执行F1R，D1 FAIL则停止扩展并检查decoder/逐fixture loss。

### 2026-08-31 — D1 Windows入口 READY

- Run：N/A；尚未执行Ubuntu真实数据实验，D1状态仍为`PENDING`。
- 代码：新增`CVAE_POSTERIOR_MAX_WINDOWS`/`--max-windows`，在完整motion/variant校验后确定性选择固定窗口前缀；D1取第一个窗口。
- 可追溯性：summary新增`available_window_count`、`max_windows`与`selected_windows`，后者记录source window index、motion、variant、episode与window start。
- 协议：训练与exact evaluator共享同一窗口子集；generalization checkpoint强制匹配motion、窗口长度和窗口子集；默认`max_windows=null`保持F1与原容量阶梯行为不变。
- 验证：9项posterior单元测试、与模型/隔离回归合计29项通过；Python compile、全部config JSON解析与Shell语法通过；真实HDF5/CUDA尚未验证。
- 后续计划：只执行D1 smoke，成功后执行D1 formal；在formal结论前禁止G0/F2。

### 2026-08-31 11:47 — S0 PASS

- Run：`/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t8_20260831_114746`
- 代码：外层 HEAD 尚未从 run 的 `source_commit.txt` 回传；根据已执行入口应包含 posterior capacity 实现，仍需实际文件确认。
- 配置：1 motion，window 8，seed 20260830，fixed smoke，参数量 6,725,731；8 episodes、288 windows、2,880 mask fixtures。
- 执行：完成2个 optimizer step并生成 `cvae_posterior_capacity_smoke.ok`；shell 成功返回 run_dir。
- 最佳结果：score 25194.9549，State RMSE 2.5195，Action RMSE 2.2020，max abs 15.7025，contact 52.47%，zero/swapped latent ratio 0.9985/0.9999。
- 分层结果：State最差为 `element_both_10` 2.5195；Action最差为 `semantic_both` 2.2020；最大绝对误差来自 `full_state` 15.7025。
- 事实结论：真实 HDF5、CUDA、forward/backward、exact evaluator、checkpoint、summary 与 smoke marker 工程链路可执行。仅训练2步，所有质量指标失败是预期现象，不能用于模型容量判断。
- 后续计划：F1 已执行并失败；读取其 summary/metrics 后再决定诊断。

### 2026-08-31 — I0 PASS

- Run：N/A，仅 Windows 实现与轻量测试。
- 代码：外层 `tiny-model@fcdb4f8861e539e3ea364e578d6bc96ce7ebd9b0`；SONIC 与 IsaacLab 未修改。
- 实现：新增6,725,731参数 posterior Transformer、10类 fixed Mask、16-slot held-out Mask、独立 loss/evaluator/checkpoint/marker、CLI/Shell入口与配置。
- 验证：新增7项 posterior 测试通过；与既有模型/隔离测试合计27项通过；全部 config JSON、Python compile、Shell语法与 `git diff --check` 通过。
- 边界：Windows Python 缺少 `h5py`，未执行真实 HDF5/CUDA smoke，不能声明真实数据训练可运行或已过拟合。
- 后续计划：执行 S0。

## 7. 失败诊断与后续研究门槛

下表保留早期阶段的诊断候选，不自动触发新训练；当前F4D之后的执行顺序、预算和停止规则由第3.2节
与Next.md接管。完成本轮前不得按下表同时开启latent扩维、loss权重搜索或额外续训。

| 主要失败模式 | 第一诊断 | 允许的首个单变量对照 |
|---|---|---|
| S0 工程失败 | traceback、HDF5 shape、CUDA OOM、checkpoint/marker | 只修工程错误，保持研究配置不变后重跑S0 |
| `full_both` 明显最差 | global latent或decoder容量不足 | latent 256→512；其余配置不变 |
| partial Mask差但`full_both`好 | Mask conditioning或fixture覆盖问题 | 固定同一数据，增加每窗口Mask重复/训练步数二选一，不同时改 |
| State好、Action差 | Action输出或尺度优化困难 | Action loss权重1→2；不增加专用Action head |
| Action好、State差 | State连续组或contact牵制 | 先分离continuous/contact曲线；只在证据支持时调整contact权重 |
| contact未达100%，连续量已通过 | BCE收敛/阈值问题 | 保持结构，仅延长到80k并检查logit margin |
| F1失败且合成过拟合仍通过 | 真实窗口数量或优化规模问题 | 增加“单固定窗口”诊断入口后先证实一窗口近零，不直接扩模型 |
| G0/G1失败但对应F通过 | Mask泛化而非记忆容量问题 | 保持checkpoint与数据，只扩大随机Mask训练覆盖 |

25M快速阶梯的R128通过后即可启动独立的“物理条件补全/CVAE”计划；无需让32 motion、T=128在
exact门禁下通过，但必须满足progression门禁。CVAE阶段逐级恢复posterior采样、conditional prior与非零KL，并分别评估posterior上限和
不读取目标真值的prior。该后续计划固定区分三类任务：

1. **State前向递推**：选择起点`u`和长度`h`，保留缺口前足以覆盖最大控制延迟的近期
   State–Action历史、`S_u`及`A_u...A_{u+h-1}`，Mask`S_{u+1}...S_{u+h}`；正式forward gate
   禁止读取`S_{u+h}`之后的未来State。
2. **Action逆推**：Mask`A_u...A_{u+h-1}`，保留缺口前的近期Action历史及逐转移两端的
   `S_u...S_{u+h}`；报告确定性RMSE，同时承认未观测delay queue、reference或一对多动作会使
   严格唯一恢复不成立。
3. **短片段联合修补**：只在序列内部Mask短区间的State与Action，保留缺口前后的State–Action
   context和两侧边界；这是双向smoothing/inpainting，不得表述为因果forward rollout。由于该任务
   通常一对多，开启KL与prior采样后用概率覆盖率/物理一致性评价，不能只要求单一真值exact RMSE。

三类物理Mask必须分别训练、分别验收；posterior真值latent只作上限，真正的条件能力必须使用
masked-input prior或不读取目标真值的确定性条件路径。
