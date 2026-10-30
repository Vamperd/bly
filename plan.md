# 65-token CVAE：当前 A／B／C 实验计划与结果

最后更新：2026-10-30

只保留实验路径、概要结果及下一步；历史run、checkpoint、数据和源码均未删除。
结构合同见[model.md](model.md)，必要反常证据见[process.md](process.md)，DAgger见[dagger.md](dagger.md)，
交接入口为[AGENTS.md](AGENTS.md)。

## 新增路线：sonic-10history-reference-cvae-v1

独立新增10-step SONIC序列化CVAE：`State[10,70] + Action[10,29] + reference[10,64]`
输出一个canonical Action；Physics v5为唯一新数据源，C阶段只训练当前目标Action的
posterior/standard-normal loss与KL。旧65-token/T64与旧DAgger路线只读保留。
数据链路见第23节；训练与DAgger命令见dagger.md末节。

## 1. 当前状态与路线

架构 `65-token-hierarchical-standard-cvae-v1`，协议 `65-token-experiment-v2`。
参数64,377,959；99/198输入、16个hard chunk、latent、FiLM及原重建loss保持不变。

```text
A90k + model-only再训练60k：完成，封存为完整输入参考（第2节）
B-fixed：1 motion / 16窗口 / 128fixture / 20k，完成（第3、4节）
B-dynamic：同16窗口 / 20k，完成（第5节）
32-motion B-fixed / B-dynamic：各120k，完成（第9、10节）
C：32-motion全部T64窗口 / 1504窗口 / 动态Mask / 随机初始化 / 360k，完成（第11节）
KL3 C：fixed Mask / 200k，完成（第20节）
```

证据边界：16窗口均来自同一motion，不称新motion泛化或严格未见Mask；工程PASS不等于预注册质量PASS，
B与C的quality_pass均为null。32-motion B-fixed与C都随机初始化，不加载A或16窗口B权重，也不绕过数据身份校验。

## 2. A：完整序列 Posterior 重建基线

数据 `/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506`（已误删，重建见第12节）；
32 motion、256 episode、T64下1,504个已见训练窗口。Posterior mean → 层级latent → Decoder；
不使用Condition，KL=0。以下误差除特别标注外均在归一化连续特征空间。

| 项目 | A90k | 从A90k权重初始化再训练60k |
|---|---:|---:|
| optimizer step / best step | 90000 / 90000 | 60000 / 60000 |
| full reconstruction loss | 1.77765898e-5 | 6.98823541e-6 |
| State RMSE | 0.00588709 | 0.00373420 |
| Action RMSE | 0.00431750 | 0.00264767 |
| 联合continuous p99 | 0.01698315 | 0.01124605 |
| 联合continuous max_abs | 0.21955711 | 0.13111076 |

原始A：
`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_fixed_20260922_014417`。
再训练A：
`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_fixed_20260923_012104`。

两次batch均为32、warm-up 2k、cosine末端1e-6，峰值分别为1e-4和2e-5。后60k只加载模型权重，
重新初始化AdamW和调度器，**不是保留optimizer的无缝续训**。只读重评位于
`/home/helloworld/bly/runs/cvae_v2_reassessment_gweSqTsw`，数值复现，contact准确率100%；
历史身份缺失字段保留unknown，不声称精确恢复。

loss下降60.69%，max下降40.28%。原最大点 `confusion_103__A045 / variant7 / start384 / t46 / joint_vel_14`
（0.219557≈0.240674 rad/s），后最大点 `variant3 / start448 / t64 / joint_vel_28`（0.131111≈0.074355 rad/s），
不是同一坐标。局部速度变化拟合不足是候选机制，不能据此确认chunk边界或容量瓶颈；详见process.md。
A封存为参考，不要求尾部先归零才能推进B。

## 3. B-fixed：本次训练身份与工程核验

训练run：`/home/helloworld/bly/runs/cvae_v2_B_fixed_w16_s20000_0XqzpnZ5`。
回传目录：`C:/Users/86136/Desktop/replay/diagnostic_report`；回传索引1,390个文件hash全部匹配，
10个相关源码文件hash与Windows实现一致。源码commit `b64df6499fab832246117521427548069dbba501`。

| 合同 | 实际值 |
|---|---|
| 阶段／初始化 | B condition-only；随机初始化；无A/source checkpoint |
| 数据范围 | 1 motion `baby_full_diaper_walk_ff_360_loop_R_001__A462`、4个variant、16窗口 |
| Mask | 16窗口×8家族=128固定fixture |
| 预算 | 20000步，batch32，640000样本暴露 |
| 学习率 | 首步5e-8；2000步到1e-4；cosine至末步1e-6 |
| seed | initialization 20260921；training Mask 20260920 |
| 参数 | 总64,377,959；可训练49,501,351；posterior可训练参数0 |

posterior调用0、condition调用1，KL全为0，无NaN/Inf；`empty_local` 无梯度是未触发空块fallback的正常情况。
best=last=step20000，严格读回通过；两checkpoint记录SHA256均为
`e22931f3420cb7cf4a53a6bca500e4eaf2baf165f43722ac5a425e590daf6fe7`。

数据normalization SHA256：
`dad2270cb8466b5616cc6db63d49b065c0bebea65ac0440e60ccd82b6c918995`。
selected_windows SHA256：
`05e0e9932f93528f77bb0d0fe2e5e22b1b10152689c673e3c32108a858d00a39`。

证据边界：回传包未含 `data/fixed_mask_bank.json`、`data/heldout_mask_bank.json`，不能写成完整bank逐项重算一致
（B-dynamic回传已补齐，见第5节）。

## 4. B-fixed：结果与解释

| 归一化连续指标 | 已见fixed | held-out bank（含19个重合fixture） |
|---|---:|---:|
| family等权masked MSE | 5.293105e-7 | 1.695675e-4 |
| masked State / Action RMSE | 0.00098331 / 0.00070615 | 0.01348454 / 0.00778734 |
| masked State / Action max | 0.00796247 / 0.00489709 | 0.68288028 / 0.25016236 |
| contact准确率 | 100% | 100% |

held-out使用同一批已见窗口，只改变缺口位置及部分长度；不同坐标的109项masked State/Action RMSE为
0.01452174/0.01357753。15k→20k fixed改善63.81%，held-out仅改善2.66%（最小值在14500步），新Mask迁移已平台。
held-out最差为fixture55/window6（variant1、start64、joint_gap_8），masked State/Action RMSE 0.120357/0.074133，
贡献整个bank masked平方误差的51.73%/59.58%。step19500消融显示global/local/memory/FiLM路径均影响重建。

结论：condition-only链路能高度拟合16窗口128固定fixture；候选原因是每窗口每family只见一种固定Mask，
不优先支持容量不足。证据补齐前不换loss、latent、pooling或模型宽度。

## 5. B-dynamic（16窗口）：20k回传审核

run：`/home/helloworld/bly/runs/cvae_v2_B_dynamic_w16_s20000_HlVrXXXJ`。
回传：`C:/Users/86136/Desktop/replay/B dynamic 16`；诊断包1,361项hash匹配，两bank各128fixture内容hash通过。
固定bank hash为 `125ae17a3ac98989035dc19be7530fece8591f1b88357d096d64984aa9579fb8`，
另一bank为 `f755b53cce35a34b692381f7c2e8960f0f5973fcd6946b53456f5bba4b936635`。

从B-fixed step20000 best.pt做model-only初始化，重建AdamW/调度器；KL=0，读回通过。
配置micro_batch32，但**实际每步batch16**（dynamic sampler仅16个window且不跨epoch补齐），实际暴露320000。
因此改善不能严格归因于Mask变量单独作用。

| 指标（normalized masked，另一bank） | B-fixed末点 | B-dynamic末点 |
|---|---:|---:|
| State / Action RMSE | 0.01348454 / 0.00778734 | 0.00058543 / 0.00039055 |
| State / Action max | 0.68288028 / 0.25016236 | 0.00665534 / 0.00350285 |
| joint_gap_8 State / Action RMSE | 0.03964516 / 0.02486406 | 0.00083416 / 0.00057132 |

原fixed bank最终masked State/Action RMSE为0.00073464/0.00037982，无固定记忆退化证据。
best按fixed在19500步，另一bank最佳为20000步；末次 `masked_worsens_while_full_improves` 是小幅分歧告警，
不需追加训练。结论：小数据集条件链路与动态Mask适应获得支持，不证明多motion容量或最终随机生成。

## 6～8. 32-motion B 协议、回传与落地（已执行）

- 配置 `configs/posterior_hierarchical_standard_cvae_h50.json`（65-token架构，T64、stride64、max_windows=null、
  max_episodes256）；从随机初始化开始，不从16窗口checkpoint热启动，不用legacy开关绕过identity mismatch。
- 人工审核协议 `B32-fixed-review-v1`（normalized有效元素）：pooled masked State/Action RMSE≤0.01；各家族micro RMSE≤0.02；
  窗口p95≤0.03；pooled p99≤0.05；masked contact≥99.9%；最后连续3次评测达标。仅为研究推进阈值，
  不是代码门禁；不能修改marker伪装通过。
- 回传：完整评测每5000步；精简包保留manifests、最新plots、完整metrics/evaluations JSONL、normalization与两bank、
  每次summary，diagnostics只回传step0、best_step、末3次及最近一次消融；不打包HDF5、checkpoint、all_predictions。
- 落地：B训练新增 `best_fixed.pt` 与 `best_heldout.pt`；C checkpoint新增固定epsilon的standard-normal readback probe；
  修正 `best_heldout_step` 漏写。Windows全量discovery唯一已知失败为缺少 `h5py`。

## 9. 32-motion B-fixed 完成结论（2026-09-24）

run：`/home/helloworld/bly/runs/cvae_v2_B_fixed_m32_s120000_tWXXU9fj`。32 motion、256 episode、1504窗口、
12032 fixture、batch32、120000步、暴露3,840,000；`engineering_pass=true`。
`fixed_quality_pass=false`、`heldout_quality_pass=false`：fixed masked State/Action RMSE 0.020689/0.012962，
held-out 0.079144/0.035262；visible 0.009949/0.009520，瓶颈在隐藏元素补全。held-out约70k后平台，不做普通续训。

## 10. 32-motion B-dynamic 完成结论（2026-09-25）

run：`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_random_20260924_125341`。
从32-motion B-fixed `best.pt` model-only初始化，stage=B、mask_mode=dynamic，120000步，工程与读回通过。
fixed selection 0.000328276→0.000514385（+56.7%），`best_step=0`；held-out selection 0.007004692→0.000551900
（-92.1%），`best_heldout_step=120000`。最终fixed/held-out masked State/Action RMSE为0.023536/0.012274与
0.022965/0.013237，未达0.01；`quality_pass=null`，记录 `primary_regression_over_20_percent_for_3_evaluations`。
动态Mask缓解了fixed坐标过拟合，不证明新motion泛化；最后10k仅改善2.92%，转入第11节C。

## 11. C：独立随机初始化与标准正态部署完成结论（2026-09-29）

回传包：`C:/Users/86136/Desktop/replay/C train/cvae_C_return`。
Ubuntu run：`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_kl_20260925_211113`。
合同：`stage=C`、`mask_mode=dynamic`、32 motion、256 episode、1504个T64窗口、batch32、360000步、
11520000 sample exposure；`source_checkpoint=null`。生效seed为20260921／20260920；`command.txt` 的shell seed
20260824与effective config不一致，以effective config为准，后续启动命令应消除该歧义。

工程：360000条连续train记录、73次完整评测、顶层标量有限；best=last=step360000，严格读回通过，
checkpoint SHA256 `21a4545c50a9fcd7847c8f7d62b120d1a7be9abdc512935082269bf111cf5392`；
`cvae_posterior_standard_cvae_execution.ok` 存在，`execution_pass=true`、`quality_pass=null`。

| 路径（step360000，fixed bank） | energy | masked State / Action RMSE | full State / Action RMSE | State / Action max |
|---|---:|---:|---:|---:|
| posterior mean | 0.013585 | 0.016604 / 0.007912 | 0.008654 / 0.006309 | 0.7277 / 0.2511 |
| posterior sample | 0.012063 | 0.017468 / 0.008526 | 0.009039 / 0.006644 | 0.9293 / 0.2525 |
| standard normal（部署路径） | 0.027423 | 0.081051 / 0.040847 | 0.036627 / 0.026234 | 14.3284 / 2.4324 |
| zero latent | 0.032803 | 0.070393 / 0.033941 | 0.031667 / 0.021927 | 14.7097 / 2.6927 |

standard-normal相对posterior mean的masked RMSE放大约4.88／4.83倍，且比zero latent更差。held-out
standard-normal masked State/Action RMSE 0.078670/0.041141，与fixed无明显坐标迁移断崖；主问题是从posterior
切换到标准正态部署后的latent先验／条件融合失配。step345000消融：global latent几乎不敏感，Decoder主要依赖
condition memory与FiLM；末步KL global约1e-5、local约1.7e-2。

最终判定：**C工程执行通过；posterior路径可重建；standard-normal部署质量未通过，quality_pass保持null。**
不对同一run普通续训，也不同时改loss、latent、pooling或网络宽度。
回传包证据限制：`evaluations_summaries/step_000360000/` 含五条最终summary，但 `evaluations_detail/` 无step360000
逐元素diagnostics，末步最差坐标/window只能从Ubuntu源run只读导出。

## 12. Overfit subset recovery 与 window 844 回放（2026-09-28）

原始 `/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506` 已误删且无压缩备份；父数据集
`/home/helloworld/bly/runs/cvae_physics_dataset_20260825_235244` 完整（含 `cvae_dataset.ok` 与
`cvae_physics_dataset.ok`）。按 `motions_per_package=4`、`seed=20260828` 重新生成：

```text
/home/helloworld/bly/runs/cvae_overfit_subset_rebuilt_20260928_193357
```

新数据集为Physics v4，32 motion、256 episode、1504个T64 window；window 844仍对应 `jump_right_004__A029`、
variant 4、`demo_12025`。它是后续训练的正式新基准，但 `dataset_manifest_sha256` 可能与旧A/B/C checkpoint不同。

- `replay65 prepare --allow-recovered-dataset`：只放宽 `dataset_manifest_sha256`，仍严格要求 `episodes_index_sha256`、
  `normalization_sha256`、`selected_windows_sha256`、窗口数量与长度；manifest记录 `recovered_dataset=true`、
  `exact_identity_verified=false`。
- `--allow-recovered-identity`：用户明确接受诊断性结果时才用，允许身份显式漂移但不允许缺失字段；记录
  `recovered_identity_override=true`。不能用于训练或质量门禁。
- 新训练必须使用新数据集和新run；数据集元数据须单独压缩备份，不能只保留HDF5而丢失manifest、episodes、
  normalization和markers。

## 13. C latent 敏感性与五 seed full-action 回放

入口 `state-action-cvae/src/cvae_sa/latent_sweep65.py`，协议基础 `65-token-latent-sweep-v1`，当前实现
`65-token-latent-sweep-v4-t64-original`。不改训练/loss/checkpoint；标准正态sample只调用 `encode_condition` 与
`infer_from_condition`，posterior只生成 `full_truth_posterior_mean` 诊断参考。`prepare` 固定 `full_action`、
window 844，再按seed `20260928` 选两个不同motion；sample seed `20260923`--`20260927`、`sample_index=0`、
Isaac `simulation_seed=20260930`。输出目录统一为 `samples/seed_<seed>/simulations/full_action`；
render输出每seed `*_full_action_action.mp4`。重建数据集须显式 `--allow-recovered-dataset`；baseline无效时
物理质量为 `UNDETERMINED`。不是训练质量门禁。

Ubuntu smoke问题→修复：
- Ubuntu不存在 `/mnt/c/Users/...`，原同步命令实际没有传输文件。
- `prepare` 预创建空 `baseline/simulations` 被判为未完成child → 只预创建parent；render显式加入 `sonic-repro-kit` 到import path。
- 磁盘 `errno=28`；seed目录 `samples/20260924` 与render查找名不一致 → 统一命名，旧run不得复用，须新run重跑。

## 14. latent sweep 固定 64 帧回放

协议 `65-token-latent-sweep-v4-t64-original`：每个trajectory含64个Action、65个State frame；original baseline、
posterior reference和每个C sample只执行自己的T64 raw Action，不调用SONIC补后续Action。exact initialization
contract读回 `friction_combine_mode`、`restitution_combine_mode`、静/动摩擦、恢复系数、关节/刚体参数、sim_dt、
control_dt、decimation、gravity、solver iteration；`runtime_context_max_abs > 1e-5`、combine mode不一致或无法恢复
的actuator delay硬失败并保留 `exact_initialization_readback_*.json`。未记录的PhysX contact cache标为隐藏状态未确定。
MP4为65帧、50 Hz、叠加 `REPLAY 0-64`；旧129-frame run不得补marker冒充本协议。

## 15. v2 BF16、compile 与同步数据管线（2026-10-04）

- 保留逐特征z-score与canonical Action；训练precision默认BF16（参数、AdamW、loss、latent统计与评测FP32，不用GradScaler）。
- A/B/C与standard-normal inference各有compiled callable；strict readback保持eager；fallback/graph break/recompile写入
  runtime contract、checkpoint与summary。
- `cache_mode=auto` 按dataset、normalization与selected-window hash建立run内缓存，超出 `cache_max_bytes` 走未缓存路径。
- Ubuntu验收顺序：FP32 eager+uncached → BF16 eager+uncached → BF16 compiled+uncached → BF16 compiled+cached；
  吞吐达到训练1.25x、评测1.15x且数值门禁通过才设为活动配置，否则按BF16 eager、FP32 compiled、FP32 eager降级。

## 16. 十七组 posterior latent 分布诊断与 KL×2 对照（2026-10-04）

- `cvae_sa.latent_distribution65`：只读，只送完整State--Action进Posterior，报告 `global`、`local_00`--`local_15`
  的mean/logvar/std、sample统计与每维KL；输出 `plots/latent_group_distributions.svg`、`plots/latent_group_kl.svg`、
  `plots/global_dimension_kl.svg`、`plots/local_dimension_kl.svg`、`data/latent_arrays.npz`、`manifests/group_stats.json`。
- `cvae_sa.latent_normality65`：读取 `latent_arrays.npz`，计算KS距离、均值/标准差偏差、偏度、超额峰度、尾部偏差；
  阈值标签只是诊断提示。
- KL×2：只改 `kl_beta: 0.001 -> 0.002`，用 `CVAE_POSTERIOR_STANDARD_CVAE_KL_BETA=0.002` 新建随机初始化run，
  不从旧C恢复。

## 17～18. KL×2 smoke 的 compiled CUDA Graph 问题（2026-10-05／10-11）

问题→修复：`cvae_kl2_smoke_ReqoL9gV` 在step 0失败，PyTorch 2.7 compiled C forward的CUDA Graph输出buffer被覆盖；
加 `torch.compiler.cudagraph_mark_step_begin()` 后（提交 `18306c1`）新run
`/home/helloworld/bly/runs/cvae_kl2_smoke_fixed_YDFjPMkr` 仍失败；关闭compile/cache/pinned的
`/home/helloworld/bly/runs/cvae_kl2_smoke_eager_poHm7ke5` 完成2 steps（仅证明工程路径）。
最终修复：compiled core与BF16 autocast解耦，评测route输出加 `clone_before_eager_readback_v1`；运行时异常或
Dynamo recompile limit触发记录原因的eager fallback（不计为compiled pass）；默认mode改为
`max-autotune-no-cudagraphs`。compiled须先过2-step smoke与readback无fallback门禁；禁止从失败目录resume。

## 19. replay65 v2：3 s 回放、闭环重采集源与 A/B/C 对比

- 默认 `--horizon-steps 192`（3 chunk）；每chunk按65-token形状单独推理，相邻chunk丢弃后一块首帧，
  报告 `chunk_boundary_state_max_abs`、`window_start_shift`。
- `replay65 capture`：从HDF第0帧exact-init跑跟踪策略并记录raw Action，同seed开环重放两次（max-abs ≤1e-4），
  写 `capture_complete.json` 与 `markers/replay65_capture.status=DETERMINISTIC`。`prepare --capture-run` 以此为真值源，
  评分用绝对阈值 `FAIL_ABSOLUTE_THRESHOLDS`，前缀偏差报 `prefix_max_abs`；采集源 `training_seen=false`。
- `replay65 compare --runs RA RB RC`：要求同窗口/horizon/seed/源SHA；输出 `compare.json`、SVG、`compare_ghost.mp4`、
  `compare_grid.mp4`。视频是运动学回放，不代替物理指标。

## 20. KL3 C 200k eager 结果（2026-10-12）

run：
`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_kl3_eager_200k_eval50k_noeval0_zlqw641k`。
`stage=C`、`mask_mode=fixed`、32 motion、1504 T64窗口、12032 fixture、200000步、`kl_beta=0.003`、BF16 eager、
compile/cache关闭、跳过step 0评测；数据 `cvae_overfit_subset_rebuilt_20260928_193357`，seed 20260921/20260920，
`source_checkpoint=null`。三份checkpoint SHA256均为
`bc080d48a07e5861a2b8c2e0171cce980d35ee58eef7e921282c7597e3408a63`；`best_step=last_step=200000`。

| 路径 | fixed masked MSE / RMSE | fixed full State / Action RMSE | fixed energy | held-out energy |
|---|---:|---:|---:|---:|
| posterior mean | 1.69711e-4 / 0.013027 | 0.009673 / 0.007744 | — | — |
| posterior sample | 1.75595e-4 / 0.013251 | 0.009854 / 0.007899 | — | — |
| standard normal | 6.71202e-4 / 0.025908 | 0.020599 / 0.013384 | 0.013854 | 0.036871 |
| zero latent | 5.11567e-4 / 0.022618 | 0.018052 / 0.011364 | 0.016420 | — |

standard-normal fixed masked MSE约为posterior mean的3.95倍；held-out masked MSE `0.00359963`。主要尾部是State
`joint_vel`（fixed max `12.272463`，`crawl_ff_start_225_003__A133` 的 `state_rollout`）。global平均KL `8.7081e-6`，
local `0.002032`，其中 `local_15` 为 `0.016562`（约占50.9%）。结论：**工程通过；posterior重建较好；standard-normal
部署质量未通过；`quality_pass` 保持null。** 结果封存，下一步选一个单变量校准实验，不从该run续训。

### 20.1 与原 `kl_beta=0.001` C 的对照边界

原C为dynamic Mask、360000步、FP32 eager；KL3为fixed Mask、200000步、BF16 eager，属观测对照，非KL单变量因果实验；
`kl_beta=0.002` 只有2-step eager smoke。

| 指标 | 原 `0.001` C | KL3 `0.003` | 表面变化 |
|---|---:|---:|---:|
| fixed standard-normal masked State RMSE | 0.081051 | 0.045421 | 降低 44.0% |
| fixed standard-normal masked Action RMSE | 0.040847 | 0.022076 | 降低 46.0% |
| fixed standard-normal energy | 0.027423 | 0.013854 | 降低 49.5% |
| held-out standard-normal energy | 0.026876 | 0.036871 | 上升 37.2% |
| posterior mean fixed masked State / Action | 0.016604 / 0.007912 | 0.015256 / 0.009119 | State略好、Action变差 |

增大KL压低了local posterior偏离并改善fixed指标，但没有解决latent集中（`local_15`）或held-out Mask迁移。

## 21. 长时 T64 派生数据集（2026-10-15～10-18）

- `long_horizon_dataset.py`、`long_horizon_eval.py`、FP32 eager长时配置及 `cvae_repro.sh` 的build/smoke/train/eval/render入口；
  含episode分组、tail padding、transition coverage、源HDF5 hash和方法A/B的192-step离线预测。
- C评估提速：训练batch 32、评估batch 128；每5,000步summary evaluation（不写trace/SVG、不更新best），每50,000步
  和最终步full diagnostic evaluation并执行checkpoint选择。
- `recursive_state_feedback_b`：首窗完整Action Mask推理但只执行前32个Action，再用实际State反馈循环预测下一段32个
  Action到192步，生成193帧回放run。`tail32_context_next32` 只保留为recorded-context/OOD诊断，不代表闭环外推。
- `cvae_sa.long_horizon_ab_compare`：每个192-step group比较同一checkpoint的A与递归B（不画KL2/KL4横向曲线）；
  三动作批量输出只保留A/B replay、compare grid视频、compact JSON/SVG和报告，清理后更新prepared hash。

## 22. G1+ unlocked31 迁移与三动作视频（2026-10-20）

```text
source bundle: /home/helloworld/bly/sonic-repro/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/urdf/g1_plus/
final model:   /home/helloworld/bly/runs/g1plus_unlocked31_diag_20261019_160509_3309439/unlocked31_run/assets/g1_pl_unlocked31_mujoco_fixed6.xml
sha256:        abd1a3dcccdeb43e163ef0d6f21d9eec2e6d426c97fb255d4c3dfe452aa34455
```

结构门禁通过：`nq=38`、`nv=37`、`njnt=32`、`nu=31`，31个movable joints，两个neck hinge和actuator wiring可读回。
free-floating drift smoke未通过（测试把非neck torque ctrl设为零），不是C replay或实机质量结论。
转换细节（urdf2mjcf修复、trajectory路径）见model.md §12。

三动作ghost结果已封存：

```text
/home/helloworld/bly/runs/g1plus_unlocked31_action_01_source_static_moving_ghost_20261020_002708_157847/
/home/helloworld/bly/runs/g1plus_unlocked31_action_02_source_static_moving_ghost_20261020_002708_157847/
/home/helloworld/bly/runs/g1plus_unlocked31_action_03_source_static_moving_ghost_20261020_002708_157847/
```

每个结果是一条193帧ghost：source opaque、C recursive replay + neck static、C recursive replay + neck moving。
它是同相机运动学对比，不能写成MuJoCo torque rollout或sim-to-real通过。三动作的C source/replay依赖：

```text
/home/helloworld/bly/runs/cvae_long_horizon_ab_three_actions_20261019_012955_694718/kl4_200k/action_01/
/home/helloworld/bly/runs/cvae_long_horizon_ab_three_actions_20261019_012955_694718/kl4_200k/action_02/
/home/helloworld/bly/runs/cvae_long_horizon_ab_three_actions_20261019_012955_694718/kl4_200k/action_03/
```

该C多动作run、原始G1+ bundle、asset audit run和final fixed6 run在视频归档后仍需保留。
中间run（neck compare、zero-body-control dynamics compare、z-fight/prepare）只在manifest/失败证据归档后作为候选清理，
不自动删除；清单见process.md §22。

## 23. 10 历史 reference CVAE 的 Physics v5 数据链路（2026-10-30）

`sonic-10history-reference-cvae-v1` 的 C/DAgger 全部基于下列 run，按依赖顺序排列，
均需保留；任一被删除都必须从该步起整条链路重建（采集、索引、C、DAgger），不能混用：

```text
# 32-motion 选择（第 12 节重建，seed=20260828，8 package x 4）
/home/helloworld/bly/runs/cvae_overfit_subset_rebuilt_20260928_193357
# 2048-motion BONES ingest（PKL 真实来源，子集只读符号链接指向这里）
/home/helloworld/bly/runs/bones_seed_ingest_2048_20260825_132447
# 32-motion reference 采集输入（robot_filtered/*.pkl 为绝对符号链接）
/home/helloworld/bly/runs/prepare_overfit_reference_subset_20261030_174517
# Physics v5 采集 A：startup，variant 0-3，canonical 128，completion 126/128
/home/helloworld/bly/runs/collect_physics_state_action_reference_20261030_175230
# Physics v5 采集 B：initial_state_mild，variant 4-7，canonical 128，completion 126/128
/home/helloworld/bly/runs/collect_physics_state_action_reference_20261030_181411
# CVAE v5 索引：32 motion / 256 episode，split 32/0/0（无验证集），seed=20260828
/home/helloworld/bly/runs/cvae_physics_dataset_v5
# C smoke（2 step / 32 sample，仅链路验证）
/home/helloworld/bly/runs/sonic_causal_c_smoke
```

索引 `episodes_index_sha256=7a3cac2f41418f6e8ce5edee1e2299d9e99d806714a8760ea269ac6fca2d9a60`；
子集 `motion_manifest_sha256=5dbb426217bcee62348cfd4d5cea5104bef87009e93c775976533a6ff618e9b7`。
`cvae_physics_dataset_v5`、`sonic_causal_c`、`sonic_causal_c_8h`、`sonic_causal_c_smoke` 是固定
路径，重建前只能改名备份旧目录，不能原地覆盖。正式 C 采用 8 小时墙钟预算
（`configs/sonic_causal_cvae_c_8h.json`，`max_train_seconds=28800`，每 5000 步写
`checkpoints/latest.pt`，结束写 `checkpoints/best.pt`），输出
`/home/helloworld/bly/runs/sonic_causal_c_8h`；checkpoint 写入 `train_motion_keys`，DAgger 只从
这些 motion 中抽取。

本轮修正：`sonic_repro.sh` 统计 motion PKL 时改用 `find -L`，否则符号链接子集被计为 0 条；
`sonic-repro-kit/README.md` 的 `OVERFIT_SELECTION_RUN` 改指向重建子集。SONIC 分支中补丁
0006/0007 已被并入提交 `05248f3`，按标题查不到属预期，0008 为 `6a1166a`。
