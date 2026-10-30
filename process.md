# CVAE 必要异常与根因诊断

最后更新：2026-10-26。本文只记录会影响实验判断的反常结果、证据、候选原因和验证边界，
不复制常规训练流水或整体实验路线。路线与概要结果见 [plan.md](plan.md)，模型合同见 [model.md](model.md)，
DAgger命令见 [dagger.md](dagger.md)。10-history新路线见plan.md开头与§23；其质量证据必须与旧T64/CVAE结果分开记录，
同-motion DAgger imitation不等于new-motion泛化。

## 1. 本次证据与可比性

本地回传根目录：`C:/Users/86136/Desktop/replay/feedback`。
Ubuntu 重评根目录：`/home/helloworld/bly/runs/cvae_v2_reassessment_gweSqTsw`。
评测源码 commit：`b64df6499fab832246117521427548069dbba501`，协议 `65-token-experiment-v2`。
11 个回传包的 report_index 共342项文件哈希全部核对一致；这验证回传完整性，不等于独立验证数据真值。

关键证据索引（相对本地回传根目录）：

| 证据 | 用途 |
|---|---|
| `A_completed_training/logs/metrics.jsonl` | 60,000步训练、61次完整评测及学习率趋势 |
| `A_original_90k/evaluations/A/{summary,diagnostics}.json` | 原90k完整分布、argmax与曲线 |
| `A_continuation_best/evaluations/A/{summary,diagnostics}.json` | 续训最终分布、argmax与曲线 |
| `A_continuation_last/evaluations/A/diagnostics.json` | 与best逐字节相同的诊断内容；不声称checkpoint文件哈希相同 |
| `B_{best,last}_{posterior_condition,condition_zero_posterior,old_A_route}/evaluations/` | 同一checkpoint不同信息路径的受控比较 |
| `diagnostic_report/manifests/{standard_cvae_summary,audit_000000002}.json` | Ubuntu新B真实数据/CUDA工程smoke与梯度、路由审核 |

A原90k源run为 `cvae_posterior_hierarchical_standard_cvae_65_fixed_20260922_014417`；60k权重初始化训练run为
`cvae_posterior_hierarchical_standard_cvae_65_fixed_20260923_012104`，均在 `/home/helloworld/bly/runs/` 下。
旧checkpoint只记录局部step，重评中的 `cumulative_step=60000` 不代表权重只经历过60k训练。

各重评使用相同1,504个T64窗口、同一数据集、index与归一化：
normalization SHA256=`dad2270cb8466b5616cc6db63d49b065c0bebea65ac0440e60ccd82b6c918995`；
selected_windows SHA256=`5cb8002ce328bc750df3438dff495a1677f327980b9ed70f79a110f0769faca6`。
重评指标在浮点聚合误差范围内复现训练日志。历史checkpoint缺少index/normalization/window-table哈希，
`exact_identity_verified=false` 应保留。数据全部为已见训练窗口，不是泛化测试。

## 2. 异常A：平均误差很小，max仍明显较大

| 指标 | 原90k | 再训练60k后的最终点 |
|---|---:|---:|
| full reconstruction loss | 1.777659e-5 | 6.988235e-6 |
| State p99 / p99.9（重评） | 0.01800912 / 0.03589776 | 0.01183576 / 0.01947615 |
| State max / Action max | 0.21955711 / 0.06426299 | 0.13111076 / 0.04675962 |
| State abs error >0.05 / >0.1 / >0.2 的元素数 | 2,326 / 168 / 1 | 222 / 5 / 0 |
| State max >0.1 的窗口数 | 43 | 1 |

State连续元素共6,647,680；最终>0.1的5个元素均在同一窗口、同一feature。阈值仅为诊断用，不是质量门禁。
全部1,504个窗口State max仍>0.01，不能称逐元素无损重建。step46k联合max最低0.12337743，55k..60k在0.12494..0.13471
波动而loss继续下降：末段max非单调、降低很慢，不支持“训练失败”或“已达误差下限”。best按平均loss选择（best=last=60k）。

两代最大元素不是同一个点：

| 坐标 | 原90k最大点 | 最终60k最大点 |
|---|---|---|
| motion | confusion_103__A045 | confusion_103__A045 |
| episode | initial-state collection `demo_3022` | startup collection `demo_3020` |
| variant / window index / start | 7 / 277 / 384 | 3 / 242 / 448 |
| 相对t / 原始帧 | 46 / 430 | 64 / 512 |
| feature（零起始索引） | 43：joint_vel_14 | 57：joint_vel_28 |
| chunk / chunk内位置 | 11 / 2 | 15 / 4 |
| physical absolute error（rad/s） | 0.24067391 | 0.07435539 |

窗口277的State max降至0.07440391；最终最大的5个元素均来自window242的joint_vel_28（t=64/12/58/31/61），
不只发生在终点或chunk边界。最终速度域physical max约0.13566021 rad/s（joint_vel_10）；不能混合rad、rad/s、m。

## 3. A尾部的原因：证据强弱分开

- 已证实的局部现象：快速速度变化没有被逐帧充分还原。window277 t46孤立峰值幅值被低估；window242一阶差分标准差
  target/pred为0.07371/0.03953 rad/s。但预测峰值并非总偏小，不能归结为统一缩小幅度，也不证明标签有噪声。
- 最终State top100全是关节速度（53项joint_vel_28、29项joint_vel_27），86项来自confusion与neutral_looking_around；
  整体feature RMSE最差项却是joint_pos_23/24：平均误差与极端尾部重点不同。
- 最有依据的机制：训练优化均值MSE/BCE，不直接优化max；最大单元素平方误差只占State SSE约0.01854%。
  具体由Encoder、pooling或Decoder造成无法分离，不能宣称扩大模型即可消除。

| 已检查而不支持的假说 | 证据与边界 |
|---|---|
| 近零std放大 | 29个速度std为0.3170..1.9636 |
| 全局错一帧/两帧 | ±2帧诊断均为lag0最小；不能排除终点局部相位误差 |
| State_64丢失或local_15失效 | t64 RMSE 0.003593低于整体0.003734 |
| 硬chunk边界系统性断裂 | chunk内位置0..4 RMSE 0.003695/0.003755/0.003767/0.003728/0.003593 |
| Mask采样错误 | A不调用Condition或Mask，不适用 |
| KL或梯度爆炸 | A60k全部kl_beta=0，梯度范数最大约0.01205 |

重叠窗口同一原始帧的State预测最大跨度从0.176749降到0.108694，上下文依赖仍存在，但不能据此修改chunk。

## 4. 异常A：重启后先变差，最终才获得收益

`INIT_RUN` 加载原90k best模型并重置AdamW、scheduler和采样器；2k warm-up升至2e-5（原末端20倍）。
2k时loss升至7.449978e-5（约4.19倍），局部29k才首次优于基线，最终下降60.69%。高LR重启与AdamW重置同时改变，
不能独立归因。旧日志learning_rate是scheduler.step之后的下一次LR；v2分别记录actual/next LR。

## 5. 异常B：低训练loss与高评测loss来自不同信息路径

旧run `cvae_posterior_hierarchical_standard_cvae_65_random_20260922_152852` 上限150k，134,776步中断，best=134,000。
其B训练仍读取完整posterior，评测却固定走无condition的A路径。

| 同一旧B last checkpoint的路径 | full State / Action RMSE | masked State / Action RMSE | State max |
|---|---|---|---:|
| posterior mean + condition | 0.011087 / 0.007931 | 0.021973 / 0.011263 | 3.951270 |
| posterior latent=0 + condition | 0.655503 / 0.768258 | 1.223946 / 0.846521 | 39.442398 |
| 旧A评测路径，无condition | 0.199735 / 0.122331 | 不适用 | 22.512138 |

旧best的A路由重评max=22.5015106复现旧日志，确认此前高分来自错误评测路径；旧权重强依赖posterior输入。
零latent是训练分布外干预，不能证明独立重训Condition Encoder的能力上限。

## 6. 下一步如何区分原因（A尾部）

只读补诊应固定 window277/t46/feature43、window242/t64/feature57 及覆盖相同原始帧的重叠窗口，两代checkpoint都导出；
不重采数据、不改标签。A封存为已见序列重建基线，不原样追加60k，不把max降到某阈值作为启动B的硬门槛。
若需压低速度尾部，先做坐标配对和原始数据核查，再单变量比较尾部加权/采样、局部解码路径或pooling。
导数损失可能放大数据噪声，不默认启用；数据清洗、平滑、裁剪异常值需单独授权，不能用于掩盖max。

## 7. 原始Action回放漂移：已知异常与证据边界

历史H50单窗口未exact-init回放：记录→原Action的joint RMSE=0.245825rad、root RMSE=0.251772m、orientation max=106.918°，
而原Action→模型Action的joint RMSE仅0.008448rad；“原始与模型回放接近”不能证明复现了采集轨迹。
既有初始化恢复不含执行器延迟队列、历史delay draw和PhysX接触缓存。新回放先分别验证初始化、原始重复性、原始对记录复现；
两次原始回放同样偏离真值即判基线无效，此时模型物理质量不可判定。处理方式见第12节与第17节。

## 8. 32-motion B-fixed：固定Mask记忆与新Mask泛化分离（2026-09-24）

回传包：`C:/Users/86136/Desktop/replay/32motionfixB/compact_return_20260924_124152`。
Ubuntu run：`/home/helloworld/bly/runs/cvae_v2_B_fixed_m32_s120000_tWXXU9fj`。合同与末步RMSE见plan.md §9。

| 指标 | fixed bank | held-out bank |
|---|---:|---:|
| selection / masked MSE | 0.000328276 | 0.007004692 |
| masked State / Action p99 | 0.0640906 / 0.0379806 | 0.288760 / 0.131179 |
| State / Action max abs | 0.771466 / 0.157573 | 11.707875 / 4.282176 |

- fixed最大点：`confusion_103__A045`、variant 1、window_start 448、`state_gap_16`、相对帧30、`joint_vel_18`，
  物理误差-0.47590 rad/s，属稀疏尾部。
- held-out最大点：`crawl_ff_stop_225_R_003__A233`、variant 7、window_start 321、`joint_gap_8`、相对帧31、`joint_vel_8`，
  物理误差约3.71169 rad/s；masked State中17.4%元素>0.05、6.34%>0.1，不是单个离群点。
- 最难家族为 `joint_gap_2`/`joint_gap_8`。支持“缺口条件泛化不足和关节速度尖峰难补全”，不支持仅靠延长fixed训练，
  也无证据要求扩大主干。

## 9. 32-motion B-dynamic：held-out改善与fixed回退（2026-09-25）

回传包：`C:/Users/86136/Desktop/replay/32motionradomB/compact_return_20260925_195133`。
Ubuntu run：`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_random_20260924_125341`。
主结果见plan.md §10；`manifests/command.txt` 的shell seed 20260824与effective config（20260921/20260920）不一致，
以effective config为准。

- held-out State max 11.707875→1.139175；fixed selection上升56.7%，触发
  `primary_regression_over_20_percent_for_3_evaluations`，是清晰的fixed/held-out trade-off。
- held-out最大State点：`neutral_looking_around_R_001__A542`、variant 3、window_start 0、`joint_gap_2`、相对帧2、
  `joint_vel_8`，物理误差约0.361147 rad/s。已见/新坐标masked State/Action为0.021116/0.009718与0.023062/0.018286。
- `best_step=0`、`best_heldout_step=120000`：Dynamic质量checkpoint是 `best_heldout.pt` 或 `last.pt`，不是 `best.pt`。

## 10. C阶段：posterior重建好，standard-normal部署差（2026-09-29）

run与四路线结果见plan.md §11。已排除：中途崩溃、NaN、batch短缺、路由未执行或读回损坏。
候选机制：Decoder主要使用condition memory和FiLM，global通道可能低利用（末步global KL约1e-5、local约1.7e-2）。
但消融只有4个窗口且用posterior mean，不能证明global完全坍缩，也不能区分KL权重、latent尺度、融合门控和训练覆盖的贡献。
standard-normal energy 300k→360k仅降至0.027423，普通续训不是根因修复。补齐step360000尾部证据前不标记质量通过，
不启动新的结构性训练分支。

## 11. replay65 baseline：Isaac结束后marker目录缺失（2026-09-28）

问题→修复：`/home/helloworld/bly/runs/cvae_replay65_C_w0_lS0WeyvH` 中 `mark_stage` 写
`simulations/baseline/markers/action_mask_replay.ok.tmp.$$` 失败，因 `replay65.py::simulate_group` 只创建 `child/data`；
后续第二次simulate报 `incomplete/changed simulation is not overwritten`，其MP4不能作为执行验收。已修复为启动worker前创建
`data/manifests/logs/markers`，shell侧防御性创建目录。旧失败目录不复用、不手工补marker。

回放选窗规则：困难Action回放统一选 `joint_gap_8`（有State与Action缺口）；`first-suite` 按每个checkpoint八类Mask masked MSE
选最差窗口，并保留episode起点和首个非零起点作对照；三条路线共享同一最难motion/window。

## 12. 固定 64 帧原始 Action 回放合同（2026-09-28）

物理问题定义为open-loop：同一motion、初始化、仿真参数下，窗口内64个raw Action逐步送入Isaac；不补hold-final、
不读窗口后HDF5 Action。合同细节见plan.md §14。`exact_replay_initializer.py` 记录 `combine_modes_match`；
runtime audit记 `solver_contact_cache=unknown_not_recorded`。可见参数全部通过后仍发散，写成“隐藏模拟器状态未确定”。
Ubuntu新run须确认每个child的 `action_replay_request.json` 为 `post_action_mode=none`，再检查
`exact_initialization_readback_*.json`、`*.runtime.json` 和 `recorded_to_original_1_errors.npz` 的首个阈值帧。

## 13～16. Runtime、latent分组、KL×2 与 compiled 修复（2026-10-04～10-11）

- Runtime：compile、BF16、TF32、cache、pin-memory、non-blocking写入runtime contract；旧C的FP32 eager是历史基线。
  compile callable创建成功或缓存生成不等于吞吐提升或精度满足阈值。
- latent分组：没有实际 `group_stats.json` 与逐维热图前，不能断言“global已坍缩”；`near_standard_normal` 只是启发式标签。
- KL×2判定规则：standard-normal误差下降且local/global KL无单组过度收缩，才支持“KL权重不足”；若posterior重建变差而部署
  不改善，停止放大KL，转查posterior-to-prior校准、condition融合和decoder对latent的利用。
- CUDA Graph问题→修复：`/home/helloworld/bly/runs/cvae_kl2_smoke_ReqoL9gV` 在 `optimizer_step=0` 报
  `accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`（非BF16/磁盘/KL配置问题）；
  `cudagraph_mark_step_begin()` 不足，最终改为 `max-autotune-no-cudagraphs` + 输出clone + eager fallback
  （`compile_fallback_reason/phase`），详见plan.md §17～18。新run须检查 `runtime_contract.compile_mode`、
  `compile_observations`、`compile_info`、`failure.json/progress.json` 和execution marker。

## 17. 原始 Action 回放漂移的处理方式（2026-10-12）

不再补全不可观测内部状态，改让源来自同一仿真器（`replay65 capture`，见plan.md §19）。两次重放不满足≤1e-4时
`capture_determinism.json` 标记 `NON_DETERMINISTIC`，该窗口模型质量不可判定；采集与HDF的偏差只写入
`dataset_to_capture`，不参与判定。

## 18. 四配置 runtime smoke 的 route-hook 修复（2026-10-11）

问题→修复：`/home/helloworld/bly/runs/cvae_v2_runtime_smoke_20261011_fixed` 中FP32/BF16 eager完成2 steps；两个BF16 compiled
配置在首次update前因route isolation记录 `{'posterior': 0, 'condition': 0}` 失败，根因是eager preflight未先安装encoder forward
hooks。已修正 `cvae_training.py`：preflight先装hooks验证再移除，eager fallback时重装。失败目录只读，须用新run重做compiled smoke。

## 19. KL3 C 200k：工程通过，standard-normal 尾部仍未通过（2026-10-12）

回传目录 `C:/Users/86136/Desktop/replay/1012_KL3测评zip/`，对应 Ubuntu run
`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_kl3_eager_200k_eval50k_noeval0_zlqw641k`。
主结果与 `kl_beta=0.001` 对照见plan.md §20、§20.1。补充证据：

- held-out standard-normal full State/Action RMSE `0.028694/0.017950`；新坐标10400个fixture masked State/Action RMSE
  `0.066908/0.042441`，重合坐标1632个为 `0.050919/0.024446`：部署路径对Mask坐标迁移仍敏感。
- fixed最大State误差 `12.272463` 为 `joint_vel_26`；held-out最大State `11.898151`、Action `4.096210` 在 `joint_gap_8`。
  尾部是连续速度/动作误差，不是contact故障。
- 已排除：中途崩溃、NaN、checkpoint损坏、数据身份不一致、路由未执行和BF16不可用。不得把
  `cvae_posterior_standard_cvae_execution.ok` 写成质量通过，也不得同时修改多个模型变量。

## 20. 单窗口 A/B/C State-only 延伸回放（2026-10-12）

`--state-only-continuation`：选定T64 window只作mask、epsilon和fixture identity模板；后续每个64-transition块直接取同一
source capture的State，不调用 `chunk_indices()`。后续块Action target仍写入prediction artifact用于评分，但模型调用前置零，
`action_before_window` 也置零；manifest写 `input_contract=state_only` 与
`continuation_input_contract=state_only_after_first_chunk`。身份不一致时须显式 `--allow-recovered-identity`，
`exact_identity_verified` 按false解读。

## 21. 长时数据集实现的证据边界（2026-10-15）

派生dataset marker只证明manifest、source hash、chunk coverage和padding检查完整；long-horizon execution marker不证明
192-step Action或物理replay质量。方法B（前32步context加下一窗口前32个State的32-step Action gap）不在默认随机Mask分布内，
须单独报告warm-up、visible index、mask hash和OOD状态。递归B与 `tail32_context_next32` 的区别见plan.md §21。
多动作A/B诊断按checkpoint分组，最大State/Action误差须映射回具体关节；KL2与KL4不合并到同一质量结论。

## 22. G1+ unlocked31 证据边界与清理清单（2026-10-20）

模型路径、门禁和三动作ghost目录见plan.md §22。补充：G1+ bundle中的 `g1_pl.urdf`、`meshes/`、`motor_info.md` 是重建所需源资料。
每个ghost目录的 `ghost_source_static_moving.mp4` 与其manifest、model hash、source/replay hash一起归档；`quality_pass` 保持null，
`kinematic_only` 为true。body drift smoke失败是free-floating/no-hold测试设计边界，不能解释为neck actuator缺失。

不可清理：原始G1+ bundle、`g1plus_asset_audit_20261019_024703`、`g1plus_unlocked31_diag_20261019_160509_3309439`、
三动作C多动作run及三份最终ghost目录中的视频/manifest。

候选清理（当前未删除）：

```text
g1plus_unlocked31_C_action03_neck_compare_20261020_001423_157847
g1plus_unlocked31_action_dynamics_compare_20261020_000958_157847
g1plus_unlocked31_action03_neck_compare_20261019_235044_157847
g1plus_unlocked31_actuator_smoke_20261019_233013_157847
g1plus_unlocked31_mjcf_diag_20261019_232720_157847
g1plus_black192_zfight_fix_20261019_135444_3309439
g1plus_replay65_prepare_20261019_124743_1247522
```

清理前先把小型manifest/失败日志复制到归档位置，并确认最终视频已保存。
`latent_distribution_kl2_kl4_recovered_20261019_190949_4046727` 属于latent诊断，不在G1+清理范围。
zero-body-control dynamics compare中机器人下蹲，是因为29个身体torque ctrl全为零且无姿态保持；C canonical Action是关节目标，
不能直接塞入fixed6的torque actuator ctrl。

## 23. CVAE KL2 pilot native SIGSEGV（2026-10-26）

问题→修复：
- smoke记录initial State时CUDA/CPU混用，`torch.stack` device mismatch → `_new_record` 统一 `.cpu()`；smoke的
  `quality_pass=false` 是单轮expected。
- action mapping自检 `offset` max_abs=0.014，来自 `add_joint_default_pos` startup随机化 → offset容差0.03，其余严格1e-6。
- pilot `dagger_cvae_kl2_pilot_20261026_123821` 在round 0 step 800后native SIGSEGV（exit 139），dump位于h5py读取或
  `_source_loss`/backward，根因未定，该目录不能resume → `source_cache_size=2048`（smoke为1）预加载1504个source window；
  `train_chunk_steps=500` 分块保存 `checkpoints/round_XX_partial.pt` 与 `manifests/dagger_cvae_kl2_round_progress.json`
  （临时文件+atomic replace）；`run_dagger_auto.sh` 对CVAE默认 `DAGGER_SHARD_ROUNDS=1`、`DAGGER_MAX_RETRIES=5`。
  若仍复现，用 `source_mix=0` 独立诊断run区分HDF5与CVAE/CUDA/Isaac native原因，不直接进入formal。

## 24. 4-motion intensive 与 stepwise causal 渲染路线（2026-10-26）

配置与命令见dagger.md。`cvae_kl2_pilot_intensive`（12 rounds × 2000）作calibration；`cvae_kl2_pilot_intensive_8h` 为
18 rounds × 4000 ≈ 72k steps，beta `[1.0,1.0,0.8,0.6,0.4,0.2,0.1,0.0]` + hold_last；pilot实测约2.72 steps/s，8h round1约2.1 steps/s。
`source_cache.pt` 持久化在run的 `data/`。`early_stop`：第5/10轮（0-based 4/9）要求teacher RMSE ≤0.98×round0且termination不变差，
失败写 `manifests/dagger_cvae_kl2_early_stop.json` 与 `markers/dagger_cvae_kl2_early_stop.ok`，auto看到marker后退出。

8h run `dagger_cvae_kl2_pilot_intensive_8h_20261026_141123`：round0/1（beta=1.0，teacher-only rollout）teacher RMSE
0.2986→0.29996，只说明状态分布迁移；原schedule round4仍为beta=0.85，已改为上述schedule（round2内中断时partial先完成旧beta=0.95剩余chunk）。
最终18/18 rounds完成，`dagger_exit_code=0`、存在 `markers/dagger_cvae_kl2_execution.ok`；`quality_pass=false`：
overall ratio 0.55424未达 `<=0.5`，4个motion均满足 `<=0.7`，termination与saturation通过。teacher Action RMSE/ratio：
big_heavy `0.11050/0.39704`、lift_crate `0.14361/0.48511`、dance_vouge `0.21525/0.65443`、cellphone `0.20287/0.69711`。
不是执行失败，也不能写成质量通过；先渲染定性复核，再决定短续训或调整总体聚合门禁。

stepwise路线（`dagger.stepwise_prepare` + `dagger.stepwise_recursive_causal`）：从recorded 192-step group的exact init开始，每步重建
causal window、只执行Action[32]，重复192步；与env-reset live eval分开，也不是 `recursive_state_feedback_b` 的32-Action开环块。
`sonic-repro-kit/render_three_way_ghost.py` 输出source opaque、original C与fine-tuned C半透明的193帧、50 Hz运动学ghost。

Ubuntu stepwise/渲染问题→修复（均为工程问题，不构成数据或模型质量结论）：
- `replay65.prepare_window` 的 `args.masks` 缺字段 → wrapper补 `masks=[]`。
- manifest `created_at` 缺 `timezone` 导入（prepare与 `stepwise_recursive_causal.py` 各一次）→ 补 `datetime, timezone`。
- `STEPWISE_CASES_MANIFEST`、`CVAE_FINETUNED_CHECKPOINT` 默认值含 `<...>` 被OmegaConf拒绝 → 改为合法路径。
- stepwise配置缺顶层 `seed` → 补 `seed: 20261023`。
- replay只含 `physical_state` 无 `physical_state_v3` → `_load_source` 兼容两种key。
- 第二个case报 `Simulation context already exists`（`SonicDaggerEnv`/`ManagerEnvWrapper` 无 `close()`）→ 下钻到 `env.env.close`；
  同进程重建 `SimulationContext` 仍卡住 → 新增 `--case-index`，每个motion独立进程。
- ghost MP4无法播放/花屏 → `libx264 + yuv420p + +faststart`，另输出source、original C、fine-tuned C三个单独MP4。
- 短轨迹导致1帧视频 → 保留完整source帧数，短轨迹重复末帧并标注“ended N, holding”，记录 `trajectory_frames`/`padded_frames`。
  192步/50 Hz约3.86 s，不是截断。
- `env.reset()` 后reference锚定随机起始帧 → `reset_motion_to_first_frame()`（`time_steps`/`motion_start_time_steps` 清零）。
- 各单独视频用各自root相机，ghost按初始root xyz对齐，记录 `raw_first_roots`；默认黑色地面，只tint robot body geom。
- `set_motion_state` 把 `motion_num_steps` 设为Int32与Long buffer冲突 → wrapper cast为Int32，fall时打印termination terms。
- case0修复后original/fine-tuned到frame 36/41因 `foot_pos_xyz` 终止，属动作误差累积 → 关闭Isaac auto reset，首次fall只记录
  `fall_frame`/`fall_terms` 并继续到192帧。
- source HDF root xy在replay准备时被平移归零而reference仍是原世界坐标 → exact init后按reference anchor平移root，打印
  `source_world_root_alignment`。

训练侧 `post_fall_continuation`（新CVAE配置默认开启）：禁用Isaac auto reset，done不清CVAE history，teacher收到done=false；
按每个motion前10步reference error baseline，`error_anchor_pos`/`error_body_pos` 超过baseline 3倍并持续50步时停止该env采样。
评估仍用标准reset；已完成run只读。

`dagger.cvae_recorded_state_replay`：`s0 real + s1..s64 expected` 生成64 Action执行前32，之后每步用33 real + 32 expected State、
32 executed + 32 masked Action取 `output.action[32]` 至192 Action；输出source/original/fine/ghost四路，不判fall、不reset。

CVAE DAgger v2 `episode_mix`：4 env中1 teacher、3 student；student先执行32步teacher prefix，之后每16步block推理执行16个C Action，
每个执行步在真实student state上查询teacher标签。配置 `cvae_kl2_pilot_dagger_v2.yaml`（source mix 0.4）；2-motion约3h pilot为
`cvae_kl2_pilot_dagger_v2_2m.yaml` + `cvae_kl2_motion_manifest_2m.json`（2 env、12×2000）。旧v1配置和run只读。
问题→修复：round0后同进程再次 `create_manager_env` 挂起且auto只处理进程退出 → CVAE KL2在CLI层强制一轮一进程，
`run_dagger_auto.sh` 默认每shard 2400s超时、shard间等待5s，超时后自动重试/resume。
