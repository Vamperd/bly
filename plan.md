# 最简 Transformer CVAE Posterior 容量实验计划与结果台账

最后更新：2026-09-05
当前阶段：F4D已在Ubuntu跑满100,000 step并未通过progression门禁。4 motion、T=128共有80个窗口和800个fixed fixtures；最佳点位于最后一步，worst State/Action RMSE为0.0230/0.0153、max abs为0.2156。与此同时，全体masked元素聚合的State/Action RMSE约为0.00915/0.00812，说明平均精度已过`1e-2`而少数window/feature尾部仍失败。F4A只读尾部诊断入口现已在Windows实现并通过轻量测试，唯一下一步是在Ubuntu固定环境读取F4D `best_progression.pt`执行该诊断；它不训练、不改checkpoint，完成前R128和全部KL代码继续冻结。
本文是本轮 posterior-only 研究的执行计划、实验结果与后续决策的唯一台账。每次实验结束后必须先更新本文，再启动下一项实验。

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
| seed | 主实验 `20260830`；复现实验 `20260831` |
| 输出 | 仅 `/home/helloworld/bly/runs/<new_run_id>/` |

每个正式 run 前必须记录 Windows/Ubuntu 外层、SONIC、IsaacLab 的实际分支、HEAD 和状态。不得为了匹配本文自动 checkout、reset 或恢复历史 IsaacLab 修改。Ubuntu 只同步和执行，不手工修改源码。

### 2.2 KL=0容量阶段模型与优化

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

训练 loss 仅包含被 Mask 坐标：State continuous MSE、Action MSE、contact BCE；当前 batch 中存在的三类 loss 等权平均。

每次validation同时计算两套互不替代的门禁：

| 指标 | exact诊断门禁 | progression规模推进门禁 |
|---|---:|---:|
| worst per-window State continuous normalized RMSE | `≤1e-4` | `≤1e-2` |
| worst per-window Action normalized RMSE | `≤1e-4` | `≤1e-2` |
| worst continuous/Action normalized absolute error | `≤1e-3` | `≤1e-2` |
| masked contact classification accuracy | `100%` | `100%` |
| `full_both` zero-latent RMSE / correct-latent RMSE | `≥10` | `≥10` |

两套score都取各自阈值比值的最大值。默认`CVAE_POSTERIOR_GATE=exact`保持历史协议；显式设为`progression`时，推进score控制checkpoint选择、提前停止、退出码和独立marker。两种门禁均要求连续3次PASS；分别保存`best_exact.pt`或`best_progression.pt`，`last.pt`始终保存。summary必须同时记录`exact_gate`与`progression_gate`，所以放宽推进门禁不会把严格失败改写成严格通过。swapped-latent仅作诊断。

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
| F4A | F4D checkpoint只读尾部诊断 | 固定读取F4D `best_progression.pt`与同一80窗口×10 Mask | 不训练；Windows入口READY | 根据误差分布决定目标对齐或latent诊断 |
| R128 | 动态随机Mask与16-slot held-out验收 | 32 motion、T=128，从F128 `last.pt` model-only warm-start | 50k，每2,500 step验收 | 冻结KL=0结果并开始K0代码实现 |
| K0 | KL三路径工程smoke | 仅在L128/F128/R128全部PASS后新增入口；从R128 `last.pt` model-only初始化 | 2 step、单个确定性窗口 | K1 |
| K1 | 32-motion KL三路径正式实验 | 32 motion、T=128，从R128 `last.pt` model-only初始化 | 50k，每2,500 step三路径验收 | 按KL判断表确定唯一下一步 |

初始快速链中任一级质量失败即停止，不直接启动后一级；T=256不在本轮关键路径。F128已经失败，
因此只新增F4D这一项最小边界诊断，不恢复完整的4→8→16→32繁琐阶梯。F4D已经失败，但全局
聚合RMSE已过`1e-2`且最佳点仍在最后一步，因此不直接续训，也不直接归因于global latent容量；
先执行F4A只读诊断，审计逐window/feature尾部误差与平均训练目标的错位。F4A完成前不增加step、
模型参数或新训练loss。L128、F128、R128
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

### 3.2 K1 KL训练合同（R128通过后才实现）

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

### 3.3 三种latent注入路径与公平对照

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

### 3.4 KL权重判断与停止规则

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
# F4A：当前唯一下一项；只读重评F4D的同一80窗口×10 fixed Mask
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
| F4A | READY | — | 读取F4D `best_progression.pt`，不写源run | — | — | — | — | — | — | 执行后才生成诊断marker | 唯一下一项：在Ubuntu定位逐window/feature尾部并自动分类 |
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
- 事实结论：当前25M模型、平均masked reconstruction目标和直接1→32 motion warm-start，在200k预算下不能完成32-motion fixed记忆；zero-latent ratio刚通过说明latent被使用，但swapped ratio仅7.91且full-both灾难性失败，提示global latent区分/解码多窗口是核心嫌疑。部分Mask也明显失败，因此尚不能把问题仅归因于无可见条件的full-both；本run也不能单独证明25M参数容量在理论上不足。
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

正式阶段失败后先按最差 case 分类，只运行一个单变量对照：

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
