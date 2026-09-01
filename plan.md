# 最简 Transformer CVAE Posterior 容量实验计划与结果台账

最后更新：2026-09-01
当前阶段：W4在40k step下接近但未通过：State/Action RMSE为`1.61e-4/1.19e-4`，其余门禁通过，末段在已衰减学习率下平台化。唯一下一步是W4-E80：同seed随机初始化，仅把总步数/对应cosine长度改为80k，使每fixture暴露接近D1通过水平。
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

effective batch 固定为64：窗口不超过32时为 `16×4`，窗口64时为 `8×8`，窗口128时为 `4×16`。所有正式阶段均从随机初始化开始；只有 held-out Mask 泛化阶段允许从对应 fixed 阶段的 `best_exact.pt` 初始化。

### 2.3 Mask、loss 与验收

固定 Mask bank 对每个窗口生成10个 fixture：`full_state`、`full_action`、`full_both`、10%和50% element Both、50%连续 State 时间块、50%连续 Action 时间块、50% State feature、50% Action feature、一个 State+Action joint semantic group。每个 fixture 必须至少有一个有效 target，padding 永远不作为 target。

fixed训练与exact validation必须使用同一个Mask seed，使每个窗口的element/time/feature/semantic
坐标逐位一致。`training_mask_seed == validation_mask_seed`且summary中的
`fixed_fixture_identity_match=true`是正式fixed run的启动断言。独立seed只允许用于
generalization阶段；旧F1违反此合同，因此只能作为同Mask类型、不同坐标的诊断，不能回答训练
fixture是否被完美记忆。

训练 loss 仅包含被 Mask 坐标：State continuous MSE、Action MSE、contact BCE；当前 batch 中存在的三类 loss 等权平均。

正式 exact gate 必须对每个窗口、每个 Mask 同时成立：

| 指标 | 阈值 |
|---|---:|
| worst per-window State continuous normalized RMSE | `≤1e-4` |
| worst per-window Action normalized RMSE | `≤1e-4` |
| worst continuous/Action absolute error | `≤1e-3` |
| masked contact classification accuracy | `100%` |
| `full_both` zero-latent RMSE / correct-latent RMSE | `≥10` |

总 exact score 为各阈值比值的最大值，`≤1` 才算一次 validation PASS。只有连续3次 PASS 才生成正式 marker。`best_exact.pt` 按最低 exact score 保存；`last.pt` 始终保存。swapped-latent 指标保留为诊断，不作为正式 gate，因为同 batch 可能含同一窗口的多个 Mask。

## 3. 执行阶梯与停止规则

所有阶段严格顺序执行。任何正式阶段失败时，不得继续增加 motion 数或窗口长度；必须先更新第5节台账和第6节完成记录，再按失败诊断矩阵决定唯一下一项实验。

| ID | 阶段 | motion | window | 初始化 | 成功后下一步 |
|---|---|---:|---:|---|---|
| S0 | Ubuntu 工程 smoke | 1 | 8 | random | F1 |
| F1 | 历史运行；validation seed不符合fixed合同 | 1 | 16 | random | 不作为容量gate，保留诊断资产 |
| D1 | 修复协议后的单窗口诊断 | 1个固定窗口 | 16 | random | PASS后F1R；FAIL则先修结构/目标 |
| F1R | 修复协议后的144窗口fixed重跑 | 1 | 16 | random | 已FAIL；转W4定位规模边界 |
| W4 | 窗口容量阶梯 | 前4个固定窗口 | 16 | random | 已FAIL；转W4-E80验证暴露预算 |
| W4-E80 | 训练暴露对照 | 前4个固定窗口 | 16 | random，80k step | PASS后W16；FAIL则检查目标/结构 |
| W16 | 窗口容量阶梯 | 前16个固定窗口 | 16 | random | PASS后W64 |
| W64 | 窗口容量阶梯 | 前64个固定窗口 | 16 | random | PASS后评估64→144边界 |
| G0 | 最小 held-out Mask sanity | 1 | 16 | 通过fixed规模的`best_exact.pt` | 后续另定 |
| F2 | motion 扩展 | 4 | 16 | random | F3 |
| F3 | motion 扩展 | 8 | 16 | random | F4 |
| F4 | motion 扩展 | 16 | 16 | random | F5 |
| F5 | motion 扩展 | 32 | 16 | random | F6 |
| F6 | 序列扩展 | 32 | 32 | random | F7 |
| F7 | 序列扩展 | 32 | 64 | random | F8 |
| F8 | 完整目标容量 | 32 | 128 | random | G1 |
| G1 | 完整 held-out Mask | 32 | 128 | F8 `best_exact.pt` | R1 |
| R1 | 第二 seed fixed 复现 | 32 | 128 | random，seed 20260831 | R2 |
| R2 | 第二 seed held-out 复现 | 32 | 128 | R1 `best_exact.pt` | posterior capacity 阶段完成 |

S0 只验证 HDF5、CUDA、forward/backward、checkpoint 和 manifest/marker 管线，生成 `cvae_posterior_capacity_smoke.ok`；它不是质量 PASS。F 系列生成 `cvae_posterior_capacity.ok`；G/R2 held-out 阶段生成 `cvae_posterior_mask_generalization.ok`。

固定 fixture 与 held-out Mask 均在两个 seed 的32-motion、128-transition配置通过后，才能声明“当前纯 posterior CVAE 在本 memorization benchmark 上实现了可复现的数值近零拟合”。

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

# 任一 F 阶段；替换 MOTIONS/WINDOW 为第3节表中的值
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=fixed \
CVAE_POSTERIOR_MOTIONS=<MOTIONS> \
CVAE_POSTERIOR_WINDOW=<WINDOW> \
bash ./cvae_repro.sh posterior-capacity

# D1工程 smoke：确定性选择索引中的第一个固定窗口，共10个fixture
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=fixed \
CVAE_POSTERIOR_MAX_WINDOWS=1 \
CVAE_POSTERIOR_MOTIONS=1 \
CVAE_POSTERIOR_WINDOW=16 \
bash ./cvae_repro.sh posterior-capacity-smoke

# D1正式训练：必须新建run并从随机初始化开始
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=fixed \
CVAE_POSTERIOR_MAX_WINDOWS=1 \
CVAE_POSTERIOR_MOTIONS=1 \
CVAE_POSTERIOR_WINDOW=16 \
bash ./cvae_repro.sh posterior-capacity

# F1R失败后的唯一下一实验W4：只把固定窗口数从1改为4
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=fixed \
CVAE_POSTERIOR_MAX_WINDOWS=4 \
CVAE_POSTERIOR_MOTIONS=1 \
CVAE_POSTERIOR_WINDOW=16 \
bash ./cvae_repro.sh posterior-capacity

# W4-E80：W4近阈值失败后，只把训练上限/cosine长度从40k改为80k
CVAE_SEED=20260830 \
CVAE_POSTERIOR_PHASE=fixed \
CVAE_POSTERIOR_MAX_WINDOWS=4 \
CVAE_POSTERIOR_MAX_STEPS=80000 \
CVAE_POSTERIOR_MOTIONS=1 \
CVAE_POSTERIOR_WINDOW=16 \
bash ./cvae_repro.sh posterior-capacity

# 任一 G/R2 阶段；motion和window必须与来源 checkpoint 完全一致
CVAE_SEED=<SEED> \
CVAE_POSTERIOR_PHASE=generalization \
CVAE_INIT_CHECKPOINT=<FIXED_RUN>/checkpoints/best_exact.pt \
CVAE_POSTERIOR_MOTIONS=<MOTIONS> \
CVAE_POSTERIOR_WINDOW=<WINDOW> \
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
| W4 | FAIL | `<latest>/cvae_posterior_capacity_fixed_m1_t16_w4_*` | `b3aa63d9514cd8dd284e7f6091fc26877f57f021` | step 39750 / 1.6103 | 1.610e-4 | 1.194e-4 | 6.158e-4 | 100% | 8425.42 | `cvae.failed` | dense full/time Mask近阈值；执行W4-E80 |
| W4-E80 | PENDING | — | Windows覆盖入口已实现 | — | — | — | — | — | — | — | 仅max step/cosine长度改为80k；下一唯一实验 |
| W16 | PENDING | — | — | — | — | — | — | — | — | — | 等待W4-E80 |
| W64 | PENDING | — | — | — | — | — | — | — | — | — | 等待W16 |
| G0 | PENDING | — | — | — | — | — | — | — | — | — | 等待fixed窗口规模结论 |
| F2 | PENDING | — | — | — | — | — | — | — | — | — | 等待G0 |
| F3 | PENDING | — | — | — | — | — | — | — | — | — | 等待F2 |
| F4 | PENDING | — | — | — | — | — | — | — | — | — | 等待F3 |
| F5 | PENDING | — | — | — | — | — | — | — | — | — | 等待F4 |
| F6 | PENDING | — | — | — | — | — | — | — | — | — | 等待F5 |
| F7 | PENDING | — | — | — | — | — | — | — | — | — | 等待F6 |
| F8 | PENDING | — | — | — | — | — | — | — | — | — | 等待F7 |
| G1 | PENDING | — | — | — | — | — | — | — | — | — | 等待F8 |
| R1 | PENDING | — | — | — | — | — | — | — | — | — | 等待G1 |
| R2 | PENDING | — | — | — | — | — | — | — | — | — | 等待R1 |

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

### 2026-09-01 — W4 FAIL

- Run：绝对run路径尚未回传；命名模式为`cvae_posterior_capacity_fixed_m1_t16_w4_*`。
- 代码：`tiny-model@b3aa63d9514cd8dd284e7f6091fc26877f57f021`；training/validation Mask seed均为20260830，`fixed_fixture_identity_match=true`。
- 配置：前4个固定window、window 16、40 fixtures、参数量6,725,731、40k step fixed formal。
- 执行：跑满40,000 step并生成`cvae.failed`；最佳step 39750、score 1.6103；39k–40k score在1.61–1.67波动，当前cosine末段没有继续下降趋势。
- 最佳结果：State RMSE 1.610e-4、Action RMSE 1.194e-4、max abs 6.158e-4、contact 100%；correct/zero/swapped latent RMSE为1.094e-4/0.92137/0.92147，zero/swapped ratio为8425.42/8426.26。
- 分层结果：主要失败为`full_state` State 1.610e-4、`full_both` State 1.454e-4、`state_time_50` State 1.309e-4及`action_time_50` Action 1.194e-4；element/feature/semantic多数已显著低于阈值，max abs、contact和latent依赖全部通过。
- 事实结论：模型已经能高精度记忆4个窗口，失败集中在需要稠密输出的full/time Mask，而非逐元素Mask融合。40k时每fixture约64k次暴露，仅为D1至最佳点约136k次的一半；当前结果支持先检验训练暴露预算，不支持立即扩模型或放宽`1e-4`门禁。
- 后续计划：只执行W4-E80——同seed重新随机初始化、同4窗口/Mask/阈值/优化器，仅把max step及随之对应的cosine长度从40k改为80k；每fixture约128k次暴露。通过后W16，失败则停止规模扩展。

### 2026-09-01 — W4-E80 Windows入口 READY

- Run：N/A；Ubuntu尚未执行。
- 实现：新增`CVAE_POSTERIOR_MAX_STEPS`与`--max-optimizer-steps`，只覆盖posterior capacity的训练上限；run名前缀增加`_sN`。
- 协议：设置80k时cosine按80k重算；默认不设置仍为40k，smoke无论覆盖值仍固定2 step。
- 可追溯性：summary新增`max_optimizer_steps`与`completed_optimizer_steps`。
- 验证：11项posterior测试、与模型/隔离组合共31项、Python compile、全部config JSON、Shell语法及CLI help通过。
- 后续计划：同步Windows提交后只运行W4-E80；不得复用W4 checkpoint、放宽阈值或同时改变其他变量。

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

只有 F8、G1、R1、R2 全部通过后，才启动独立的“物理条件补全”计划，逐级恢复非零KL并分别评估
posterior reconstruction与conditional prior；不得在本容量计划中临时混入物理Mask、reference或
relation head。该后续计划固定区分三类任务：

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
