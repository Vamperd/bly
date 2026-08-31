# 最简 Transformer CVAE Posterior 容量实验计划与结果台账

最后更新：2026-08-31  
当前阶段：已确认旧F1的fixed validation错误使用独立Mask seed，导致partial fixture并非训练坐标；旧F1不再作为fixed memorization gate。协议修复已在Windows完成，下一步先执行corrected D1；仅D1通过后重跑F1R。
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
| F1R | 修复协议后的最小fixed capacity重跑 | 1 | 16 | random | G0 |
| G0 | 最小 held-out Mask sanity | 1 | 16 | F1R `best_exact.pt` | F2 |
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
| D1 | PENDING | — | 协议修复待提交 | — | — | — | — | — | — | — | corrected `max_windows=1`；下一唯一实验 |
| F1R | PENDING | — | — | — | — | — | — | — | — | — | 仅D1通过后执行 |
| G0 | PENDING | — | — | — | — | — | — | — | — | — | 等待F1R |
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

只有 F8、G1、R1、R2 全部通过后，才规划下一研究阶段：逐级恢复非零 KL 并分别评估 posterior reconstruction 与 conditional prior。下一阶段必须新建独立计划，不得在本计划运行中临时加入 KL、reference 或 relation head。
