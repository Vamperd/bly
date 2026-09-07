# 下一步实施合同：F4E Auto-Decoder 根因诊断

最后更新：2026-09-07

状态：首个Ubuntu smoke在`5cf48ca`上完成，但发现summary把2步smoke误报为正式E1/E2根因失败；该结论
无效。Windows已在提交`6d7111d9f5cddd9aac7cd6950ffae00484b7fae4`修复为专用的execution-only结论，当前唯一下一步是同步修复后重新运行
`posterior-capacity-autodecoder-smoke`；只有新smoke的源复现、80-code共享、encoder零调用/零梯度、
checkpoint读回、报告语义与execution marker全部通过，才从F4D源重新启动正式F4E。概览与实际结果仍以
[plan.md](plan.md)为唯一台账，安全规则见[AGENTS.md](AGENTS.md)。

## 1. 当前问题与固定诊断

F4E只区分两种根因：posterior encoder是否不能形成易解码的global code，以及单个256维global code
连同当前A结构decoder是否本身不足以记忆80个窗口。允许window identity查表仅是容量探针，不能进入
部署模型，也不证明conditional prior、新Mask、未见motion或真实State与Action之间的推理。

| 项目 | 固定合同 |
|---|---|
| 数据 | 原4 motion、32 episode、80 windows、T128 |
| Mask | seed 20260830的原10类fixed Mask，共800 fixtures |
| 来源 | F4D `best_progression.pt`、正式F4A、最终A/B/C STOP comparison |
| decoder | 原A结构，无F4C gate；State/Action输入投影、类型embedding与原8层decoder |
| code | `Embedding(80,256)`，同一window的10种Mask严格共享，禁止800个fixture code |
| 初始化 | 分别编码800个fixture的posterior mean，再取每个window十个向量的算术质心 |
| 损失与精度 | A的State MSE、Action MSE、contact BCE等权；KL/dropout/weight decay为0；FP32 |
| 批量 | micro-batch 4、累积16、effective batch 64 |

模型新增`decode_from_global_latent(batch,state_mask,action_mask,latent)`，只构建masked tokens并调用
latent projection和decoder，不执行共享encoder、posterior head或prior head。原`forward()`继续计算
q/p并复用该解码入口，旧checkpoint和历史接口不变。训练器通过forward调用计数和梯度状态同时证明
encoder隔离；被Mask真值改变时decoder-only输出必须逐位不变。

## 2. 前置身份与step-0门禁

训练前必须重验最终比较run
`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260907_102553`的marker、manifest和A/B/C
summary哈希，重新计算配对结果并得到`STOP_LOSS_LATENT_SEED_SEARCH`。随后严格加载F4D权重，在同一
80窗口×10 Mask上重新计算原posterior路径；F4D step、fixture数、worst State/Action RMSE、max abs、
contact与global RMSE须满足`rtol=1e-5、atol=1e-7`。任何失败都是工程失败，不得形成模型结论。

初始化产物同时保存800×256原始q编码、fixture→window/mask映射、80×256质心、确定性SHA256，以及
每个window十个q到质心的RMS/max离散度。step0另评测质心auto-decoder路径，但它不要求与原posterior
路径相等；差异正是诊断基线的一部分。

## 3. 两阶段20k协议

| 阶段 | 可训练参数 | 优化与验收 |
|---|---|---|
| E1 | 仅20,480个window-code参数 | 固定5k；seed 20260830；LR `3e-4`，warmup100，cosine至`1e-5`；每500步全评测；4k/4.5k/5k连续PASS则停止 |
| E2 | code加decoder侧，encoder与q/p头冻结 | 从E1 last继续但重置loader RNG、optimizer和scheduler；seed 20260831；code `3e-4→1e-5`、decoder `3e-5→1e-6`，warmup250；最多15k，每1k全评测，连续3次PASS早停 |

E2 decoder侧allowlist只包含State/Action输入投影、类型embedding、decoder latent token、latent projection、
8层decoder与State/Action/contact输出头。`encoder_cls`、共享Transformer encoder、posterior/prior分布头
始终冻结，训练和auto-decoder评测路径调用数必须为0。两阶段都只训练全部800个fixed fixtures，不使用
held-out Mask、动态Mask或裁剪。

正式progression门禁要求worst State/Action normalized RMSE和continuous max abs均`≤1e-2`、contact
100%，并且full-both的zero-code、cross-window donor与cross-motion donor误差相对正确code均`≥10×`。
三项依赖门禁统一使用State前68维与Action的continuous combined RMSE并排除contact；包含contact概率的
旧ratio继续以legacy字段保存。exact的`1e-4 RMSE + 1e-3 max abs`只作诊断。E1和E2分别有质量marker；
execution marker仅表示协议完整。

## 4. 结论与产物

| 结果 | 根因判断 | 唯一下一步 |
|---|---|---|
| E1 PASS | 原decoder与256维code足够，posterior code形成是主要瓶颈 | posterior-to-code蒸馏，或先auto-decoder后拟合encoder |
| E1 FAIL、E2 PASS | code+decoder容量足够，联合code形成/协同优化有问题 | 分阶段CVAE：先auto-decoder，再冻结decoder训练encoder |
| E1/E2均FAIL | 仍无证据支持当前256维共享code+decoder可达门禁 | 只比较更大global latent与per-time latent，或重审max-abs门禁 |
| 工程合同失败 | 无模型结论 | 修复后重跑smoke |

run输出summary、逐step JSONL、code初始化manifest/tensor、fixture/window/97-feature/contact明细、固定关节
速度曲线、donor身份映射、四张SVG与checkpoint。checkpoint固定为`best_code_only.pt`、
`code_only_last.pt`、条件存在的`best_coadapt.pt`及最终`last.pt`。marker为：
`cvae_posterior_autodecoder_smoke.ok`、`cvae_posterior_autodecoder_execution.ok`、
`cvae_posterior_autodecoder_code_only.ok`和`cvae_posterior_autodecoder_coadapt.ok`；质量失败额外保留
内容明确的`cvae.failed`。

## 5. Ubuntu固定执行命令

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
export CVAE_POSTERIOR_AUTODECODER_SOURCE_CHECKPOINT=/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425/checkpoints/best_progression.pt
export CVAE_POSTERIOR_F4A_RUN=/home/helloworld/bly/runs/cvae_posterior_capacity_tail_diagnostic_f4a_20260905_200807
export CVAE_POSTERIOR_AUTODECODER_TRIGGER_COMPARISON=/home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260907_102553

unset CVAE_CONFIG CVAE_RUN_DIR CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_WARM_START
bash ./cvae_repro.sh posterior-capacity-autodecoder-smoke

# 仅在smoke结果回填plan.md并确认工程PASS后，从F4D重新开始正式run：
# bash ./cvae_repro.sh posterior-capacity-autodecoder
```

每个run结束后先回传`posterior_autodecoder_summary.json`、marker列表、`source_commit.txt`、source status、
最后三个评测点和checkpoint列表，再更新台账。smoke checkpoint不得用于正式初始化。

## 附录：已结束的F4B-v2/F4C历史合同

以下内容只保留已执行实验的可审计合同；其训练命令不得重跑或用于绕过F4E。

## 1. 目标、证据边界与固定输入

当前只回答：同一已训练checkpoint在相同数据、Mask、训练顺序与短程预算下，调整连续误差权重是否改善困难元素；若收益不足，更直接地向decoder注入同一latent是否有效。暂不扩大网络宽度、增加多个latent、启用KL、reference或专用动力学头。平均MSE与max error具有相同零误差最优解，平均达标本身不是loss错误的证据。

F4A已完整重现F4D的800个fixed fixtures。21.077%表示跨这些fixtures的masked连续元素出现次数超阈值，并不表示独立原始样本比例。`full_both/partial p95=1.851`未触发旧3倍判据，但不能排除latent问题；速度峰值平滑仍是待检验假设。历史swapped-latent仅作batch内替换诊断，不能据此评价跨窗口区分能力。

### 1.1 只读资产与输出

| 资产 | 固定绝对路径 |
|---|---|
| Windows唯一代码修改端 | `C:\Users\86136\Desktop\code\RL\bly` |
| Ubuntu项目 | `/home/helloworld/bly/state-action-cvae` |
| 固定Python环境 | `/home/helloworld/bly/sonic-repro/.venv-sonic` |
| 数据集 | `/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506` |
| F4D源run | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425` |
| 所有分支的源checkpoint | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425/checkpoints/best_progression.pt` |
| F4A源诊断run | `/home/helloworld/bly/runs/cvae_posterior_capacity_tail_diagnostic_f4a_20260905_200807` |
| 新产物根目录 | `/home/helloworld/bly/runs/<new_run_id>/` |

从F4D读取`manifests/posterior_capacity_summary.json`，从F4A读取`manifests/posterior_tail_diagnostic.json`及其引用的逐fixture/window/feature记录。实际hash和数值来自文件，不用本文四舍五入的数值作为断言值。

检查数据集`cvae_overfit_subset.ok`、F4A execution-only marker、F4D失败身份和checkpoint元数据。每支创建全新的隔离run；禁止输出到数据集、F4D、F4A及其子目录，禁止覆盖已有run或checkpoint。不得删除源文件、复制大HDF5或安装/升级依赖。复核分支也从原F4D checkpoint开始，不从首轮胜者接着训练。

### 1.2 不变量

| 项目 | 合同 |
|---|---|
| 数据选择 | 与F4D summary逐项相同的4 motion、32 episodes、80 windows、T128；stride128与末端窗口规则保持原样 |
| 输入/输出 | State70、Action29；原normalization只读；State连续域为前68维、contact为后2维 |
| Mask | 原10类bank，共800 fixtures；逐位复用训练/验证Mask，valid/padding规则不变 |
| A/B模型 | `physics_posterior_transformer`，25,453,411参数；宽384、encoder6/decoder8、latent256 |
| latent及正则 | posterior mean；KL/free-bits/dropout/weight decay全部0；不采样 |
| 初始化 | model-only严格加载源权重；optimizer、scheduler与RNG重新初始化；记录checkpoint与dataset manifest SHA256 |
| 唯一允许差异 | B只改连续元素损失权重；触发后的C只改latent注入；F4R只改变优化seed |

完整窗口身份包含motion、variant、episode/source、window start。新实验生成固定排序的窗口/Mask身份及Mask位图hash，验证step0、训练、评测和A/B/C/复核间一致。输入既有metadata用于审计，不得加入模型输入。

## 2. 统一优化与A/B损失定义

### 2.1 优化合同与seed分离

| 项目 | 固定值 |
|---|---|
| 正式训练长度 | 每支10,000 optimizer step；不因暂时质量PASS提前停止 |
| micro-batch / accumulation | 4 / 16，effective batch64，保持shuffle与完整fixture覆盖 |
| optimizer | AdamW，betas=(0.9,0.999)、eps=1e-8、weight_decay=0 |
| LR | peak `3e-5`，250步线性warmup，再cosine至`1e-6`；复用历史scheduler的step计数约定并记录实际LR |
| gradient clip | global norm 1.0；记录裁剪前norm与`norm>1`比例 |
| 完整评测 | step0，随后1000、2000、…、10000；step0不得更新权重或优化器 |
| 精度/后端 | 保持现有posterior实现与固定环境；记录dtype、TF32/autocast等实际设置，不改后端或依赖 |
| fixture seed | 永远`20260830`，只负责固定Mask生成与身份 |
| optimizer seed | 首轮A/B/C `20260830`，F4R `20260831`；负责训练loader顺序与优化随机性 |

显式解耦两个seed：不能直接修改当前同时控制训练Mask的顶层seed来执行复核。先构建并严格加载源模型，再重置优化RNG、创建独立seed的loader generator；新增诊断不消耗训练RNG。所有分支使用同一数据顺序合同，并记录每步采样身份摘要以核对配对一致性。复核只说明训练顺序稳健性，不是随机初始化的独立训练，更不是held-out Mask实验。

### 2.2 A：保留原损失及梯度

A直接使用既有reconstruction loss：micro-batch内全部masked State连续元素MSE、全部masked Action元素MSE、masked contact BCE，对实际存在的分项等权平均。保留空域、无目标报错和padding处理。原代码的有效Mask由bank提供，新入口须检查无padding被选中，不借机改变A的reduction或其他数值行为。

### 2.3 B：归一化尾部混合，固定top_fraction=0.2、tail_mix=0.5

对micro-batch内fixture `i`、连续域 `d∈{State[:68],Action}`，取有效masked平方误差向量，元素数为`n_id`。只有`n_id>0`的域参与：

$$
m_{id}=\frac{1}{n_{id}}\sum_j e_{idj}^2,\qquad
k_{id}=\max(1,\lceil0.2n_{id}\rceil),\qquad
c_{id}=\frac{1}{k_{id}}\sum_{j\in\mathrm{TopK}(e_{id}^2,k_{id})}e_{idj}^2
$$

$$
L_d^B=\frac{\sum_i n_{id}(0.5m_{id}+0.5c_{id})}{\sum_i n_{id}},\qquad
L^B=\mathrm{mean}\left(\text{存在的 }L_{State}^B,L_{Action}^B,L_{contact}^{A}\right).
$$

contact、有效分项的外层平均、梯度累积和裁剪规则不变。不得将State与Action拼接后共同选top-k，不得跨micro-batch累计top-k，不得把所有fixture改为等权，也不得再加一个额外CVaR项。全域无目标时跳过，全部分项无目标时报错；padding永远不作为target。

该公式保留原有fixture元素数权重：每个域总权重归一。所有平方误差相同时B=A；单元素域B=A。它强调困难元素但不保证A/B梯度范数相同，所以仍需报告梯度裁剪情况。20%是预先固定的启发式值，21.077%的全局统计不能证明每fixture/每域top20%最优；本轮不搜索比例或混合系数。

每步同时记录`raw_reconstruction`（原total/state/action/contact MSE/BCE）与`optimization_objective`及其分项。A/B的优化目标不是同一个函数，比较学习曲线时以原始重建误差和统一评测指标为准。

## 3. 评测、真正的latent置换与产物

### 3.1 step0复现与正式质量

每支在任何optimizer step之前，使用原4大小validation batch和相同800 fixtures复现F4D。验证source step100000、窗口/Mask身份、worst State/Action RMSE、max abs、contact及global State/Action RMSE；复用已有容差`rtol=1e-5、atol=1e-7`，期望值读取源summary而非手写常量。失败即停止，不通过改Mask、batch或容差规避。

新增诊断与原eval结果分命名空间；step0为源指标一致性门槛，不要求原F4D质量通过。F4A的严格wrapper保留，只复用底层统计能力；C不得强行传入只接受原25M F4D的历史wrapper。

原始门禁如下，不能以分位数门禁、相对改善或诊断marker替代：

| 指标 | exact | progression |
|---|---:|---:|
| worst fixture State continuous normalized RMSE | ≤1e-4 | ≤1e-2 |
| worst fixture Action normalized RMSE | ≤1e-4 | ≤1e-2 |
| worst normalized continuous/Action max abs | ≤1e-3 | ≤1e-2 |
| masked contact accuracy | 100% | 100% |
| 原full-both zero-latent RMSE / correct RMSE | ≥10 | ≥10 |

每次评测同时保存两套score。活动门禁仍为progression，按既有选择规则保留`best_progression.pt`与`last.pt`；exact作为独立诊断记录，不将progression checkpoint声称为best exact。10k全部完成且最后8000/9000/10000三次均通过时才确认正式quality PASS，早期出现三次PASS不提前退出。

### 3.2 比较指标与固定曲线案例

每1k完整评测报告global State/Action RMSE、worst fixture State/Action RMSE、worst max abs、连续masked元素`abs(error)>1e-2`比例、97个连续feature的normalized/physical absolute-error p95、contact及latent负对照。各Mask统计沿用F4A的分位数与最大误差身份，contact错误单独存放。

从源F4A的top worst windows取固定前5名，身份不得按新模型重新选择。step0从源checkpoint重评这些窗口的10类Mask，在每个窗口的joint velocity维29..57中选masked最大绝对误差对应的Mask与feature作为曲线案例；同值按原Mask顺序、feature index、time index排序。将5个`window/Mask/feature`身份写成冻结manifest并校验hash，各分支与复核复用。

曲线显示同一案例的真值、masked预测、误差、可见/遮挡范围和控制时间；连续量同时保留normalized与物理单位。不得把Mask外未经监督的预测当成补全误差。曲线用于检查峰值衰减和时间偏移，不自动从feature排名得出平滑结论。

训练记录还包括每步裁剪前norm、是否裁剪、累计裁剪比例、LR、训练耗时与评测耗时、峰值显存。完整评测不以batch均值代替全体masked元素聚合。保留全部JSONL；SVG沿用明确的log10刻度和fixed-fixture图例，新增目标/原始loss对比及配对指标对比。

### 3.3 latent donor合同

旧`output.posterior_mean.flip(0)`指标保留为`legacy within-microbatch swap`。T128原bank每window连续10个slots、batch4时，full-both donor为同window的slot1或slot5。不得把这个数值改名成真正的跨窗口置换，也不重写历史数值。

每个评测点先在相同checkpoint下，为全部80个窗口生成完整有效区域被遮挡的full-both Mask并计算posterior mean；缓存只在本次评测使用，不跨训练step复用。按`(motion_key,variant_id,episode_ref,window_start,source_window_index)`稳定排序。

| 新诊断 | donor选择 |
|---|---|
| cross-window full-both swap | 排序后第i个窗口读取第`(i+1) mod 80`个窗口的full-both latent，禁止相同窗口身份；允许同motion，明确报告这一点 |
| cross-motion full-both swap | 按motion_key排序4组；组内按上述身份排序。第g组第r个窗口读取下一组`(g+1) mod 4`的第`r mod 组大小`个窗口；保证不同motion，donor允许重复，明确不是一一置换 |

decode始终使用目标窗口的full-both condition，只有latent来自donor。输出每一对target/donor身份、Mask名和指标；计算masked State/Action RMSE与与旧负对照一致聚合口径的combined RMSE/ratio，字段注明聚合是否含contact。原zero-ratio与门禁计算保持不变。置换生成不依赖训练RNG，禁止batch=1等边界意外退化为自己。

### 3.4 产物、marker及执行完整性

每个新run沿用`data/logs/manifests/checkpoints/plots/videos/markers`目录。保存resolved config、分支标签、两个seed、source checkpoint hash、dataset hash、fixture hash、每步采样身份摘要、step0复现结果、完整metrics JSONL、逐feature/window/fixture统计、donor映射、固定案例manifest、曲线及checkpoint。source_commit/status记录实际Windows/Ubuntu状态，不从旧文档推断。

配对比较在独立新比较run输出`manifests/posterior_ab_comparison.json`和SVG，保留全部参与run绝对路径/hash、最后三个匹配step的原始值、中位数、保护条件、改善比例、候选选择和唯一下一步。缺评测点或身份不一致不得生成有效比较结论；不能拿各支不同step的best checkpoint作主比较。

已实现的execution-only marker为`cvae_posterior_ab_smoke.ok`、`cvae_posterior_ab_execution.ok`和`cvae_posterior_ab_comparison.ok`，并已登记到AGENTS.md。它们分别表示工程smoke、完整10k诊断、合法配对比较，不表示质量PASS。正式质量marker仍按原exact/progression语义单独生成。执行完整但质量失败时保留失败质量状态及`cvae.failed`，不更新正式质量成功指针；比较器显式读取run路径及manifest，不能只依赖latest。

非有限loss、梯度或预测立即中止，保存错误信息、已完成步数、现有指标与诊断资产；不得生成execution完成marker。独立入口不能因为gate FAIL丢弃完整summary，也不能把gate FAIL伪装成PASS；阶段编排必须能区分工程失败与完整但质量失败。

## 4. 预算、配对比较与唯一下一步

### 4.1 固定比较定义

令`p_X(s)`为分支X在step s的连续超阈值比例，`a_X(s)`为其worst max abs，取`s∈{8000,9000,10000}`：

$$
R_p(X)=\frac{\mathrm{median}_s p_X(s)}{\mathrm{median}_s p_A(s)},\qquad
R_a(X)=\frac{\mathrm{median}_s a_X(s)}{\mathrm{median}_s a_A(s)}.
$$

改善比例分别为`1-R_p`与`1-R_a`，使用未四舍五入数值判断。若分母为0、分子也为0，残余比例记1（没有额外改善）；分母0而分子正则为无穷，不能通过相对改善。门禁PASS独立计算，不受这一诊断除零规则影响。

候选X必须在最后三个点逐次满足contact100%、原zero-ratio≥10。global State RMSE、global Action RMSE、worst State fixture RMSE、worst Action fixture RMSE四项均逐点同时满足`X(s)≤1.10×A(s)`与`X(s)≤1.10×F4D_source`。基线取原始完整summary；不能只以三点中位数掩盖某次保护条件失败。

定义strong为两项残余比例均≤0.50，worth_replicating为均≤0.80，且都必须满足上述保护条件。源F4D与A的对比用于描述重启优化后的改善，不把A变化归因于某个单独优化超参数。

### 4.2 从上到下执行的决策表

先顺序完成首轮A、B，各10k，禁止并发抢占同一GPU影响耗时对照。比较器每次只给出一个下一阶段：

| 优先级/已观测条件 | 决策 | 正式累计预算 |
|---|---|---:|
| A最后3次progression PASS | 跳过C，优化seed20260831仅复核A；不继续寻找更复杂方案 | 20k+10k=30k |
| A未PASS，B满足全部保护条件，且B最后3次PASS或strong | 跳过C，从源F4D重新启动新seed A/B配对复核 | 20k+20k=40k |
| 上两条不满足 | 此时才实现/验证/运行C 10k，与已有同seed A比较；不同时把B损失用于C | 30k |
| C后B/C至少一个worth_replicating | 在合格候选中选`max(R_p,R_a)`较小者；完全相等优先B，运行新seed A/胜出改动 | 30k+20k=50k |
| C后没有合格候选 | 停止，记录本预算和两项干预未取得充分改善 | 30k |

某分支质量PASS仍须如实记录；若未满足该阶段的机制改善/保护条件，不自动宣布根因已解决。若某branch发生工程失败，停止依赖它的比较并先修工程问题，不把未完成的run纳入质量判断或自动填FAIL根因。

### 4.3 F4R的含义与退出

复核固定fixture seed20260830，只把优化seed改为20260831，每支仍从原F4D checkpoint初始化并训练10k。A单独复核以最后3次正式progression为准。改动复核要求新seed仍双指标改善至少20%且全部保护条件成立；只有两轮都至少50%才写“本合同下强改善”。否则标记训练顺序不稳健，不声称loss或结构已被证明有效。

即使复核有效，未取得4-motion正式PASS仍停留在4-motion；只能在本轮结束并更新plan.md后另行确定该规模的后续训练预算。取得4-motion正式通过后，也须新建32-motion fixed run重新验收，不能沿用失败F128的身份或marker；32-motion fixed通过后才执行R128，R128通过并冻结后才实现KL阶段。

本轮正式新增训练最多50k，独立2-step工程smoke不计入该正式预算。没有候选或证据不足时可以提前停止；不得为了用完预算新增参数搜索、再次更换seed、拼接B+C、扩大latent或自动续训。

## 5. 条件触发的F4C结构规格

正式A/B比较已触发本节并授权实现C。目标是在原模型函数相同的起点，单独测试latent传递通路。

保留现有latent token、encoder、RoPE时间位置、所有输出头和256维global latent。令`P(z)`为现有latent projection的输出，每层decoder block之前，仅对有效State/Action token应用：

$$
h_{l,t}\leftarrow h_{l,t}+g_l\odot P(z),\qquad g_l\in\mathbb R^{384},\quad g_l=0\text{ at initialization}.
$$

P复用原参数（含原bias）；每层一个按通道的gate向量，所有该层数据token共享。latent token本身不加项，padding不注入；原block与最终LayerNorm保持原样。新增8×384=3072参数，总数25,456,483，不新建第二套projection、不改变RoPE位置或latent维度。A/B的结构开关默认关闭且state_dict仍兼容旧模型。

严格加载所有原模型参数；唯一可缺失项是明确列出的8层gate，全部置零。不能用无约束`strict=False`吞掉其他差异。新checkpoint记录结构开关与参数量；旧checkpoint默认原token路径。只为该诊断提供显式迁移校验，不放宽通用warm-start架构检查。

C使用A原损失、同样两个seed、优化参数和采样序列，训练10k。step0先证明与原F4D输出及源指标一致，再允许更新；额外记录gate范数、梯度和非零变化。固定latent后改变被Mask真值不得改变decode输出。新通路只接收实际选定latent，不得偷读posterior真值或完整encoder token。

源码实现及CPU测试现已证明gate=0时State/Action输出与原模型逐位一致、gate梯度有限非零、注入跳过latent/padding且固定latent不读取被遮挡真值；仍不证明真实数据训练有效。该设计借鉴逐层条件注入的研究动机，不等同于DiT/AdaLN；论文结果不代替本项目对照。[参考：DiT条件注入对照](https://arxiv.org/html/2212.09748v2)

## 6. 实施入口、验证顺序与交付验收

### 6.1 已实现A/B/C接口及兼容性

新增独立模块`src/cvae_sa/posterior_capacity_ab.py`和固定配置`configs/posterior_capacity_ab.json`；历史`posterior-capacity[-25m]`、其配置默认值、KL=0行为与strict F4A wrapper未改。Shell入口为`posterior-capacity-ab-smoke`、`posterior-capacity-ab`和`posterior-capacity-ab-compare`；通过`CVAE_POSTERIOR_AB_ARM=A|B|C`与`CVAE_POSTERIOR_OPTIMIZER_SEED`选择分支及优化seed。复用`CVAE_DATASET_RUN`指定固定数据集，并用`CVAE_POSTERIOR_AB_SOURCE_CHECKPOINT`、`CVAE_POSTERIOR_F4A_RUN`显式固定只读来源；fixture seed始终为20260830，不由优化seed覆盖。

`CVAE_POSTERIOR_AB_TRIGGER_COMPARISON`只允许用于C且必填；入口重算比较manifest引用的A/B summary哈希，并复验配对与`IMPLEMENT_F4C`决定。未经触发的C或把触发参数传给A/B均拒绝。resolved config、summary和checkpoint记录arm、fixture seed、optimizer seed、实际步数、结构开关与触发来源，不沿用旧F4B加法CVaR配置。

实现复用`posterior_capacity.py`的原始loss、fixed Mask与exact evaluator，以及`posterior_capacity_tail.py`的底层统计；独立逻辑处理seed、实际优化目标、真实donor、比较器和execution-only状态。比较器强制读取显式A/B/C run路径，核对source/dataset/window/fixture及逐step采样SHA256后才生成结构化结论。

### 6.2 Windows验证

| 测试组 | 必须覆盖的实质性行为 |
|---|---|
| A兼容 | 相同输入/Mask下原始loss与各参数梯度保持一致；包括缺State、缺Action、缺contact的有效分项组合 |
| B数学 | 手算不同fixture元素数的聚合；ceil top-k非整除、单元素、均匀误差、空域、全空报错、padding排除与contact不变；梯度有限并作用于预期masked元素 |
| 固定fixture与优化seed | 首轮A/B训练采样一致；复核改变采样顺序但Mask逐位/hash一致；额外诊断不推进训练RNG |
| 诊断身份与聚合 | step0源容差检查；跨窗口不自配对、跨motion不同motion、重复donor如实记录、结果不依赖batch分块；97连续feature及contact分开 |
| 比较器与状态 | 8000/9000/10000配对、中位数、逐点保护条件、除零、候选平局、50k预算路径、缺点/身份冲突、非有限失败；execution与quality marker不混淆 |
| 条件触发C后 | 零初始化输出一致、固定latent防泄漏、gate梯度与更新、padding、参数量断言、旧权重严格迁移和新checkpoint roundtrip |
| 历史回归 | 原exact/progression score、阈值、fixed/generalization seed及marker语义；JSON解析、Python compile、CLI help、Shell语法和git diff check |

先通过独立CPU合成测试，再执行相关现有posterior/model/isolation测试。Windows缺h5py的既有完整发现限制须如实记录，不安装依赖规避，也不能宣称已验证真实HDF5/CUDA。小改动不增加与实现简单镜像的无效测试。

### 6.3 Ubuntu与结果回填顺序

Windows改代码使用apply_patch并检查实际Git差异；记录外层、SONIC、IsaacLab分支/HEAD/工作树，保留用户改动。同步前再核对Ubuntu实际状态；只在可fast-forward时同步，不自动reset/checkout。Ubuntu仅执行，所有产物在新run，不手工改源码、不更新运行依赖。

Windows轻量验证通过且实际命令补入本文后，先对A/B分别创建独立2-step Ubuntu smoke：使用同一真实80窗口/800 fixtures，执行完整step0复现与末步评测、loss/梯度、保存与读回checkpoint、曲线、manifest和execution-only marker。smoke无质量结论，正式A/B必须重新从F4D源checkpoint起步，不能继承smoke权重。若触发C，同样先独立smoke再正式训练。

固定前置环境与A/B smoke命令如下；两支已在Ubuntu串行通过，且未设置`CVAE_RUN_DIR`：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
export CVAE_POSTERIOR_AB_SOURCE_CHECKPOINT=/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m4_t128_25m_s100000_gprogression_20260904_190425/checkpoints/best_progression.pt
export CVAE_POSTERIOR_F4A_RUN=/home/helloworld/bly/runs/cvae_posterior_capacity_tail_diagnostic_f4a_20260905_200807
export CVAE_POSTERIOR_OPTIMIZER_SEED=20260830
unset CVAE_CONFIG CVAE_RUN_DIR CVAE_POSTERIOR_AB_INITIAL_COMPARISON

CVAE_POSTERIOR_AB_ARM=A bash ./cvae_repro.sh posterior-capacity-ab-smoke
CVAE_POSTERIOR_AB_ARM=B bash ./cvae_repro.sh posterior-capacity-ab-smoke
```

两支smoke run分别为
`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_a_seed20260830_smoke_20260906_112939`与
`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_b_seed20260830_smoke_20260906_113154`，源码均为
`c1ae5f79111bf61ddace32073df8122dbbefec95`。二者均有`cvae_posterior_ab_smoke.ok`，step0/legacy复现、
checkpoint读回全部通过，窗口/fixture/逐step训练身份一致。以下是当时从原F4D checkpoint分别开始正式10k的执行合同；A/B现均已完成：

```bash
# A/B均已在独立run完成；不得重跑、续训或交叉使用checkpoint。
# CVAE_POSTERIOR_AB_ARM=A bash ./cvae_repro.sh posterior-capacity-ab
# CVAE_POSTERIOR_AB_ARM=B bash ./cvae_repro.sh posterior-capacity-ab

export CVAE_POSTERIOR_AB_RUN_A=<A_FORMAL_RUN>
export CVAE_POSTERIOR_AB_RUN_B=<B_FORMAL_RUN>
unset CVAE_POSTERIOR_AB_RUN_C CVAE_RUN_DIR
bash ./cvae_repro.sh posterior-capacity-ab-compare
```

比较器只接受相同optimizer seed的正式run。若decision为`REPLICATE_A_ONLY`或`REPLICATE_A_AND_B`，把optimizer seed改为20260831并只运行`replication_arms`列出的分支；完成后将首次比较run设为`CVAE_POSTERIOR_AB_INITIAL_COMPARISON`，把第二seed run重新设为`RUN_A/RUN_B`后再次调用同一比较入口。A-only复核必须unset B/C；A/B复核必须提供两支。复核比较器只在所选分支最后三次通过正式门禁时输出`NEW_32_MOTION_FIXED_RUN`。若首次decision为`IMPLEMENT_F4C`，停止Ubuntu训练并回到Windows实现C；若为`STOP_LOSS_LATENT_SEED_SEARCH`，停止本路线。不能人工跳过manifest决策。

比较run已合法输出`IMPLEMENT_F4C`，C smoke、正式10k及三支比较也已完成。以下命令仅保留为历史合同，不得重跑：

```bash
# export CVAE_POSTERIOR_AB_RUN_A=/home/helloworld/bly/runs/cvae_posterior_capacity_ab_a_seed20260830_20260906_114634
# export CVAE_POSTERIOR_AB_RUN_B=/home/helloworld/bly/runs/cvae_posterior_capacity_ab_b_seed20260830_20260906_203844
# export CVAE_POSTERIOR_AB_RUN_C=/home/helloworld/bly/runs/cvae_posterior_capacity_ab_c_seed20260830_20260907_004301
# unset CVAE_POSTERIOR_AB_INITIAL_COMPARISON CVAE_RUN_DIR
# bash ./cvae_repro.sh posterior-capacity-ab-compare
```

正式三支比较run为`/home/helloworld/bly/runs/cvae_posterior_capacity_ab_comparison_20260907_102553`，已输出`STOP_LOSS_LATENT_SEED_SEARCH`。不得追加20k、第二seed或绕过4-motion门禁进入R128/KL。

Windows与Ubuntu工程检查均已通过；停止原因是B/C未取得受保护改善，不是工程错误。若继续研究，必须另立能够区分根因的新诊断合同，不能继续微调当前loss/gate/seed。
