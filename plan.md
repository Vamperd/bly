# 最简 Transformer CVAE Posterior 容量实验计划与结果台账

最后更新：2026-09-01
当前阶段：新增独立的规模推进门禁（State/Action RMSE与max abs均`≤1e-2`，contact 100%，zero-latent ratio `≥10`），原exact门禁保留为诊断。W4历史结果已连续满足推进阈值，因此取消W4-E80/W16/W64；唯一下一步是P1，直接在1 motion全部144个window上执行推进门禁，80k仅作上限并允许提前停止。
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
| 参数量 | 6,725,731 |
| State / Action | 70 / 29维 |
| latent | 单个 global latent，256维 |
| 条件 | 仅 State、Action、逐特征 Mask、位置和 token type |
| 排除条件 | RobotInfo、reference、motion ID、action-before-window、dynamics context |
| seed | 主实验 `20260830`；复现实验 `20260831` |
| 输出 | 仅 `/home/helloworld/bly/runs/<new_run_id>/` |

每个正式 run 前必须记录 Windows/Ubuntu 外层、SONIC、IsaacLab 的实际分支、HEAD 和状态。不得为了匹配本文自动 checkout、reset 或恢复历史 IsaacLab 修改。Ubuntu 只同步和执行，不手工修改源码。

### 2.2 模型与优化

| 项目 | 固定值 |
|---|---:|
| `d_model` | 256 |
| encoder / decoder | 4 / 4层，共享 encoder |
| heads / FFN | 8 / 1024 |
| dropout / weight decay | 0 / 0 |
| posterior latent | `posterior_mean`，不采样 |
| KL beta / free bits | 0 / 0 |
| optimizer | AdamW |
| learning rate | `3e-4` |
| schedule | 500-step warmup，随后 cosine 到 `1e-6` |
| gradient clip | 1.0 |
| max optimizer steps | 40,000 |
| exact validation | 每250 step |
| early success | 连续3次 exact validation 全部通过 |

effective batch 固定为64：窗口不超过32时为 `16×4`，窗口64时为 `8×8`，窗口128时为 `4×16`。所有 fixed 正式阶段均从随机初始化开始；held-out Mask阶段从对应fixed阶段的活动门禁checkpoint初始化（exact为`best_exact.pt`，progression为`best_progression.pt`）。

### 2.3 Mask、loss 与验收

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

## 3. 最短必要执行链与停止规则

本轮不再把“32 motion、128 transition、第二seed都达到`1e-4`”设为进入CVAE的前置条件；那会把容量精度、数据规模与条件生成三个问题绑在一起。进入CVAE前只要求同一motion的完整窗口和新随机Mask均通过progression gate，同时保留exact曲线作为诊断。

| ID | 阶段 | 数据/初始化 | 上限 | 成功后下一步 |
|---|---|---|---:|---|
| P1 | 全窗口fixed推进门禁 | 1 motion、144 window、T=16、random init | 80k，连续3次通过即停 | P2 |
| P2 | 动态随机Mask与16-slot held-out验收 | 同一144 window，从P1 `best_progression.pt`初始化 | 40k，连续3次通过即停 | C0 |
| C0 | 最小CVAE管线smoke | 1 motion、T=16、物理结构Mask；posterior采样、conditional prior与KL全部接通 | 2k | C1 |
| C1 | 小规模CVAE能力 | 4 motion、T=16、物理结构Mask；KL warmup，分别报告posterior与prior | 20k | C2 |
| C2 | motion规模扩展 | 32 motion、T=16；沿用C1唯一确定的结构/损失 | 40k | C3 |
| C3 | 长序列扩展 | 32 motion，先T=64再按需要T=128 | 每级40k | 正式条件/物理评测 |

P1失败时才回退W16/W64定位容量边界，不预先支付两次训练成本。P2失败说明问题是随机Mask迁移而不是fixed记忆，不增加motion。C0/C1必须使用不读取目标真值的prior路径验收；全State+Action同时遮挡不作为条件prior的确定性RMSE任务，因为此时条件为空。32 motion从C2开始，目标是检验已经可工作的CVAE机制能否扩展，而不是继续证明posterior能把数据背下来。

marker保持可区分：exact fixed/generalization沿用`cvae_posterior_capacity.ok`与`cvae_posterior_mask_generalization.ok`；progression使用`cvae_posterior_capacity_progression.ok`与`cvae_posterior_mask_generalization_progression.ok`。progression PASS只允许声明“精度足以推进规模/机制实验”，不能称为完美拟合或物理单位无损。

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
# S0
CVAE_SEED=20260830 \
CVAE_POSTERIOR_MOTIONS=1 \
CVAE_POSTERIOR_WINDOW=8 \
bash ./cvae_repro.sh posterior-capacity-smoke

# P1：直接训练1 motion的全部144个window；80k是上限，不是必须跑满
unset CVAE_POSTERIOR_MAX_WINDOWS CVAE_INIT_CHECKPOINT
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=fixed \
CVAE_POSTERIOR_GATE=progression \
CVAE_POSTERIOR_MAX_STEPS=80000 \
CVAE_POSTERIOR_MOTIONS=1 \
CVAE_POSTERIOR_WINDOW=16 \
bash ./cvae_repro.sh posterior-capacity

# P2：P1通过后训练动态随机Mask，并在每窗口16个固定held-out Mask上验收
unset CVAE_POSTERIOR_MAX_WINDOWS CVAE_POSTERIOR_MAX_STEPS
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=generalization \
CVAE_POSTERIOR_GATE=progression \
CVAE_INIT_CHECKPOINT=<P1_RUN>/checkpoints/best_progression.pt \
CVAE_POSTERIOR_MOTIONS=1 \
CVAE_POSTERIOR_WINDOW=16 \
bash ./cvae_repro.sh posterior-capacity
```

每个命令都必须创建新 run；不得设置到已存在的 `CVAE_RUN_DIR`，不得覆盖 checkpoint。运行中的目录使用 `ls -dt /home/helloworld/bly/runs/cvae_posterior_capacity_* | head -n1` 定位，不能依赖 latest 文件。

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

状态只允许：`PENDING`、`RUNNING`、`PASS`、`FAIL`、`BLOCKED`。只有相应正式 marker 存在且 summary 一致时才能填 `PASS`；进程正常退出但质量 gate 未通过仍为 `FAIL`。

| ID | 状态 | run_dir | source HEAD | best step/score | State RMSE | Action RMSE | max abs | contact | zero ratio | marker | 结论/下一步 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| I0 Windows实现 | PASS | N/A | `fcdb4f8861e539e3ea364e578d6bc96ce7ebd9b0` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 27项组合测试、JSON、compile、Shell语法、diff check通过；执行S0 |
| S0 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t8_20260831_114746` | 待读取 `source_commit.txt` | step 2 / 25194.9549（仅smoke诊断） | 2.5195 | 2.2020 | 15.7025 | 52.47% | 0.9985 | `cvae_posterior_capacity_smoke.ok` | 工程链路通过；F1随后执行并失败 |
| F1 | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_20260831_114833` | `fcdb4f8861e539e3ea364e578d6bc96ce7ebd9b0` | step 40000 / 752.0263 | 0.05842 | 0.02495 | 0.75203 | 100% | 371.67 | `cvae.failed` | validation部分Mask坐标与训练不同；结果不构成fixed记忆失败证据 |
| D1 | PASS | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_w1_20260831_150654` | 待读取`source_commit.txt` | step 34000 / 1.0 | 4.719e-5 | 8.075e-5 | 2.255e-4 | 100% | 10342.86 | `cvae_posterior_capacity.ok` | 单窗口10类Mask全部exact；执行F1R |
| F1R | FAIL | `/home/helloworld/bly/runs/cvae_posterior_capacity_fixed_m1_t16_20260831_202012` | `b3aa63d9514cd8dd284e7f6091fc26877f57f021` | step 39000 / 36.4095 | 0.003641 | 0.002903 | 0.017851 | 100% | 331.97 | `cvae.failed` | 全部Mask同量级且平台化；执行W4 |
| W4 | FAIL | `<latest>/cvae_posterior_capacity_fixed_m1_t16_w4_*` | `b3aa63d9514cd8dd284e7f6091fc26877f57f021` | step 39750 / exact 1.6103；progression 1.0 | 1.610e-4 | 1.194e-4 | 6.158e-4 | 100% | 8425.42 | `cvae.failed`（旧exact协议） | exact失败；最近5次均满足新progression gate，不重跑，直接P1 |
| P1 | PENDING | — | progression入口已实现 | — | — | — | — | — | — | — | 唯一下一实验：全144窗口fixed推进门禁 |
| P2 | PENDING | — | — | — | — | — | — | — | — | — | 等待P1的`best_progression.pt` |
| C0 | PENDING | — | 尚未实现CVAE最小训练协议 | — | — | — | — | — | — | — | P2通过后实现并smoke |
| C1 | PENDING | — | — | — | — | — | — | — | — | — | 4 motion物理Mask CVAE |
| C2 | PENDING | — | — | — | — | — | — | — | — | — | 32 motion从这里开始 |
| C3 | PENDING | — | — | — | — | — | — | — | — | — | 长序列扩展 |

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

结果更新规则：

1. `RUNNING` 时只记录 run_dir、PID/进程状态和最后 step，不提前写质量结论。
2. 成功退出后读取 summary、metrics、marker 和 checkpoint 文件；四者矛盾时按失败处理并调查。
3. `cvae_posterior_capacity_smoke.ok` 只证明工程管线；不得填入正式质量指标结论。
4. 质量失败目录和 `cvae.failed` 必须保留，不删除、不复用；`best_exact.pt` 仍作为诊断资产。
5. 每次更新后同步修改本文“最后更新”和“当前阶段”，并在 AGENTS.md 记录新的已验证事实。

### 2026-09-01 — Progression gate Windows入口 READY

- Run：N/A；Ubuntu尚未执行新门禁run。
- 实现：新增`CVAE_POSTERIOR_GATE=exact|progression`；progression阈值固定为State/Action normalized RMSE与continuous max abs均`1e-2`，contact 100%，zero-latent ratio至少10。
- 隔离：每次validation同时保存exact/progression两套score；活动门禁独立控制连续3次PASS、`best_<gate>.pt`、退出码和marker，旧exact默认行为不变。
- 验证：posterior/model/isolation组合32项通过；Python compile、全部config JSON、Shell语法、CLI help和`git diff --check`通过。Windows全发现测试另有3项因既有环境缺`h5py`无法导入，与本改动无关。
- 历史解释：W4最后5次validation均满足progression阈值，可作为进入P1的证据；它仍是exact FAIL，且因旧代码没有progression marker，不追记正式marker。
- 精简决策：取消W4-E80、W16、W64的预先执行；P1直接覆盖1 motion全部144窗口，P1失败时才恢复窗口边界诊断。
- 后续计划：唯一下一项为P1；通过后执行P2动态随机Mask，然后进入C0最小CVAE管线。

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

P1与P2通过后即可启动独立的“物理条件补全/CVAE”计划；无需先让32 motion、T=128在exact门禁下
通过。CVAE阶段逐级恢复posterior采样、conditional prior与非零KL，并分别评估posterior上限和
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
