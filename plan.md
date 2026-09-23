# 65-token CVAE：当前 A／B 实验计划与结果

最后更新：2026-09-23

只保留当前65-token A/B的实验路径、概要结果及下一步；历史条目仅从本文移除，
任何历史run、checkpoint、数据和源码均未删除。结构合同见[model.md](model.md)，
必要反常证据见[process.md](process.md)，交接入口为[AGENTS.md](AGENTS.md)。

## 1. 当前状态与路线

架构为 `65-token-hierarchical-standard-cvae-v1`，协议为 `65-token-experiment-v2`。
参数64,377,959；99/198输入、16个hard chunk、latent、FiLM及原重建loss保持不变。

```text
A90k + model-only再训练60k：完成，封存为完整输入参考
B-fixed：1 motion / 16窗口 / 128fixture / 20k，完成
B-dynamic：同16窗口 / 20k，完成，固定及另一Mask bank均显著改善
→ 建议下一步：32-motion全部T64窗口，随机初始化B-fixed，120k上限，分段审核
→ B-fixed审核后，再独立启动同规模B-dynamic
回放按用户选择暂缓；恢复时仍先原始Action双基线，物理结论不可跳过此门槛
```

本次支持扩大**数据范围**，不支持扩大模型或直接进入最终随机生成阶段。
16窗口均来自同一motion；动态训练后bank可能与训练采样重合，不称新motion泛化或严格未见Mask。
本次两轮summary的quality_pass仍为null，工程PASS不等于预注册质量PASS。
32-motion为新的独立condition-only容量实验，不加载A或16窗口B权重，不绕过数据身份校验。
以下第2～4节为A及B-fixed已完成记录；最新B-dynamic和下一步见第5～7节。

## 2. A：完整序列 Posterior 重建基线

数据为 `/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506`；
32 motion、256 episode，在T64配置下重建1,504个已见训练窗口。
Posterior mean → 层级latent → Decoder；不使用Condition，KL=0。
以下误差除特别标注外均在归一化连续特征空间。

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

两次batch均为32、warm-up 2k、cosine末端1e-6；峰值分别为1e-4和2e-5。
后60k只加载模型权重，重新初始化AdamW和调度器，**不是保留optimizer的无缝续训**。
最终best=last；只读重评位于
`/home/helloworld/bly/runs/cvae_v2_reassessment_gweSqTsw`，原始／best／last数值复现，
contact准确率100%。历史身份缺失字段保留unknown，不声称精确恢复。

loss下降60.69%，max下降40.28%，不能再描述为“max无法下降”。
原最大点为 `confusion_103__A045 / variant7 / start384 / t46 / joint_vel_14`，
归一化0.219557对应约0.240674 rad/s。
后最大点变为同motion的 `variant3 / start448 / t64 / joint_vel_28`，
归一化0.131111对应约0.074355 rad/s。两次全局最大值不是同一个坐标；
normalization未发现近零速度std。局部速度变化拟合不足是候选机制，不能据此确认chunk边界或容量瓶颈。
详细证据与缺失的同坐标对照见process.md。A封存为参考，不要求尾部先归零才能推进B。

## 3. B-fixed：本次训练身份与工程核验

训练run：
`/home/helloworld/bly/runs/cvae_v2_B_fixed_w16_s20000_0XqzpnZ5`。

回传目录：`C:/Users/86136/Desktop/replay/diagnostic_report`。
核对summary、provenance、selected_windows、progress、全部20,000条训练记录、
41次完整评测的摘要、最终分区／family／尾部及step19500消融。
回传索引1,390个文件hash全部匹配；10个相关源码文件hash与当前Windows实现一致。
记录的源码commit为 `b64df6499fab832246117521427548069dbba501`。

| 合同 | 实际值 |
|---|---|
| 阶段／初始化 | B condition-only；随机初始化；无A/source checkpoint |
| 实际数据范围 | 1 motion、4个variant/episode、16窗口；不是整个32-motion集合 |
| motion | baby_full_diaper_walk_ff_360_loop_R_001__A462 |
| 窗口分布 | variants 0/1/2各start 0、64、128、192、220；variant3仅start0 |
| Mask | 16窗口×8家族=128固定fixture；每窗口每家族只有一个固定坐标组合 |
| 训练预算 | 20000步，batch32，累计640000样本暴露，即每fixture平均5000次 |
| 学习率 | 首步5e-8；2000步到1e-4；cosine至末步1e-6 |
| 记录频率 | 每步JSONL；500步完整评测；250步恢复点；50步控制台刷新 |
| seed | initialization 20260921；training Mask 20260920 |
| 参数 | 总64,377,959；可训练49,501,351；posterior可训练参数0 |
| 损失／选择 | full State连续／Action／contact等权重建；best按八family等权masked域均值MSE |

20,000步逐步记录均为posterior调用0、condition调用1；KL及加权KL全为0；
step连续，顶层数值日志无NaN/Inf。主要Condition、融合、projection、FiLM、decoder、输出模块均记录到梯度。
`empty_local` 无梯度是全有效chunk下未触发空块fallback的正常情况，不是断路证据。
耗时约93分钟，执行完成，末次quality warnings为空，但告警为空不等价于已验证新Mask能力。

best=last=step20000，严格模型／optimizer／scheduler step／固定前向读回通过；
两checkpoint的记录SHA256均为
`e22931f3420cb7cf4a53a6bca500e4eaf2baf165f43722ac5a425e590daf6fe7`。

数据normalization SHA256：
`dad2270cb8466b5616cc6db63d49b065c0bebea65ac0440e60ccd82b6c918995`。
selected_windows SHA256：
`05e0e9932f93528f77bb0d0fe2e5e22b1b10152689c673e3c32108a858d00a39`。

证据边界：本次审核了回传结果与源码身份，没有在Windows重新执行源HDF5/checkpoint。
回传包未包含源run的 `data/fixed_mask_bank.json`、`data/heldout_mask_bank.json`；
当前源码export-report只加入data中的normalization，故不能把“包hash正确”写成“完整bank逐项重算一致”。
代表曲线含实际Mask、摘要有坐标重合统计，已足够定位本次主要问题；后续回传需额外附这两个bank文件。

## 4. B：结果、趋势与解释

### 4.1 已见fixed与另一Mask bank

| 归一化连续指标 | 已见fixed | held-out bank（包含19个重合fixture） |
|---|---:|---:|
| family等权masked MSE | 5.293105e-7 | 1.695675e-4 |
| full State RMSE | 0.00078724 | 0.01154769 |
| full Action RMSE | 0.00056938 | 0.00859879 |
| masked State RMSE | 0.00098331 | 0.01348454 |
| masked Action RMSE | 0.00070615 | 0.00778734 |
| masked State p99 | 0.00311462 | 0.04381449 |
| masked Action p99 | 0.00219759 | 0.02752260 |
| masked State max | 0.00796247 | 0.68288028 |
| masked Action max | 0.00489709 | 0.25016236 |
| full State / Action max | 0.01223171 / 0.00561768 | 0.76220381 / 0.37027586 |
| contact准确率 | 100% | 100% |

fixed完整loss为7.174677e-7；所有masked连续元素误差均小于0.01。
完整State仅2个元素超过0.01，均为可见元素的原始网络输出；正式补全保留可见真值，
所以不能把full max 0.012232当成隐藏区域最大误差。
该点为variant2/start128/t39的joint_vel_17，约0.0079895 rad/s。

held-out使用同一批已见窗口，不是新motion测试，并同时改变缺口位置及部分长度。
128项中19项坐标与fixed相同：16个full_action和3个state_rollout。
真正不同坐标的109项，masked State/Action RMSE为0.01452174/0.01357753。
full_action两bank结果相同是Mask相同的必然结果，不是全Action新条件泛化成功。

| Mask家族 | fixed masked State / Action RMSE | held-out masked State / Action RMSE |
|---|---:|---:|
| state_gap_4 | 0.000739 / — | 0.006718 / — |
| state_gap_16 | 0.000765 / — | 0.006976 / — |
| state_rollout | 0.001059 / — | 0.007834 / — |
| action_gap_4 | — / 0.000481 | — / 0.002939 |
| action_gap_16 | — / 0.000523 | — / 0.006461 |
| full_action | — / 0.000775 | — / 0.000775（坐标重合） |
| joint_gap_2 | 0.000711 / 0.000565 | 0.009210 / 0.005099 |
| joint_gap_8 | 0.000787 / 0.000554 | 0.039645 / 0.024864 |

“—”表示零目标，不能解释为零误差。state_rollout给定整段Action，不是严格因果逐步rollout。
full_action完整State可见，本次没有“State和Action全部隐藏”的训练任务；
因此本次不支持“全无条件补全太苛刻”的解释。

### 4.2 训练趋势与局部尾部

| step | fixed masked选择MSE | held-out masked选择MSE |
|---|---:|---:|
| 2000 | 7.580342e-4 | 2.793537e-3 |
| 5000 | 1.242308e-4 | 1.083487e-3 |
| 10000 | 3.192058e-5 | 4.100892e-4 |
| 15000 | 1.462790e-6 | 1.741985e-4 |
| 18000 | 7.242458e-7 | 1.705905e-4 |
| 20000 | 5.293105e-7 | 1.695675e-4 |

15k→20k fixed改善63.81%，held-out仅改善2.66%；
held-out最小值在14500步为1.683863e-4，最终约高0.70%。
因此固定记忆仍变好，但新Mask迁移已接近平台；不能把原样增加fixed步数视为已经有证据的主要解法。

held-out最差为fixture55/window6：variant1、start64、joint_gap_8家族。
实际隐藏Action t37..46（10步）、State t38..46（9帧），而fixed此家族为8步Action缺口。
该fixture masked State/Action RMSE为0.120357/0.074133，
贡献整个held-out bank各域masked平方误差的51.73%/59.58%。
masked State最大点为t38的joint_vel_14，0.682880归一化误差约对应0.748559 rad/s；
masked Action最大点为t37的action_17，0.250162归一化误差约对应0.0319165 rad。

该处同时移位且加长，当前证据不能拆分“新位置”“新长度”和局部运动难度各自的作用。
full max的两个最大元素反而可见，说明新Mask也能扰动网络在缺口外的原始输出；
正式回放必须保留可见值，但这并不能修复隐藏区域自身的0.682880/0.250162尾部。

step19500的依赖性消融使用variant0的4个不同窗口、相同state_gap_4家族：
基线full State/Action MSE为3.35e-7/1.83e-7；
global置换为3.11e-5/2.56e-5，local置换0.740/0.321，
condition memory置换0.00181/0.00158，FiLM置零0.637/0.485。
这些结果支持相关路径确实影响重建，不支持“latent完全没使用”；
消融仅测局部依赖性，不能排名模块重要性或证明latent对全部任务表达充分。

### 4.3 结论分级

已验证：独立condition-only链路能够高度拟合这16窗口的128固定fixture；
日志、分区指标、尾部定位和读回比只看full loss更充分。
fixed数据量小且只含1 motion，不能与A全32-motion误差直接比较并宣称B优于A。

候选原因：每窗口每family只见一种固定Mask，模型对缺口坐标／长度适应不足；
联合缺失State与Action比单域缺失更易暴露这种局限。
当前不优先支持参数容量不足，也不能把所有误差归因于训练步数不够。

尚缺证据：同16窗口动态训练后的迁移表现、多motion规模下的固定记忆、
同窗口同长度只移位对照、原始Action基线及模型物理回放。
上述证据补齐前不换loss、latent、pooling或模型宽度。

## 5. B-dynamic：20k回传审核

run：`/home/helloworld/bly/runs/cvae_v2_B_dynamic_w16_s20000_HlVrXXXJ`。
回传：`C:/Users/86136/Desktop/replay/B dynamic 16`。
诊断包1,361项hash匹配；10个相关源码hash与Windows一致；两bank各128fixture的内容hash均通过。
归一化与16窗口身份和B-fixed一致，step0精确复现B-fixed最后的两bank评测。
固定bank hash为 `125ae17a3ac98989035dc19be7530fece8591f1b88357d096d64984aa9579fb8`，
另一bank为 `f755b53cce35a34b692381f7c2e8960f0f5973fcd6946b53456f5bba4b936635`。
这也补齐了上次报告缺少bank坐标的证据。

从B-fixed step20000 best.pt做model-only初始化，重新建立AdamW/调度器；
新run局部step20000，source checkpoint另记20000，不能将当前cumulative_step字段当作全谱系训练总数。
KL=0，20000条记录posterior调用0、condition调用1，无非有限顶层数值，读回通过。
配置micro_batch32，但**实际每步batch16**：dynamic sampler仅16个window且不跨epoch补齐batch。
实际暴露320000，不是640000；此前启动说明对实际batch的预期需以此更正。
这不是梯度丢失，但固定→动态比较同时受额外训练、优化器重启、调度及实际batch变化影响，
因此只能确认本轮方案有效，不能将全部改善严格归因于Mask变量。

| 指标（normalized masked） | B-fixed末点另一bank | B-dynamic末点同bank |
|---|---:|---:|
| State RMSE | 0.01348454 | 0.00058543 |
| Action RMSE | 0.00778734 | 0.00039055 |
| State p99 | 0.04381449 | 0.00200471 |
| Action p99 | 0.02752260 | 0.00120443 |
| State max | 0.68288028 | 0.00665534 |
| Action max | 0.25016236 | 0.00350285 |
| masked contact accuracy | 100% | 100% |
| joint_gap_8家族 State / Action RMSE | 0.03964516 / 0.02486406 | 0.00083416 / 0.00057132 |

另一bank整体State/Action RMSE下降95.66%/94.98%，max下降99.03%/98.60%。
原异常fixture55的masked State/Action RMSE由0.120357/0.074133降至0.001134/0.000848；
其masked max为0.005328/0.003411。其余联合缺口也改善，不是仅修复一个点。
与原fixed坐标不同的109fixture最终masked State/Action为0.00056758/0.00039835；
“new”标签仅指与原fixed不同，不能证明未被dynamic抽到。

原fixed bank最终masked State/Action RMSE为0.00073464/0.00037982，max为0.00835478/0.00330624；
没有固定记忆持续退化的证据。前期warm-up附近有反弹，之后恢复。
best按fixed分数在19500步，另一bank最佳为最后20000步：
19500→20000 fixed选择MSE仅恶化0.385%，同时full loss和另一bank选择MSE改善。
因此末次 `masked_worsens_while_full_improves` 是小幅指标分歧告警，不是训练崩溃；
保留best/last，不需为消除这条告警而追加训练。

step19500的global/local/memory/FiLM消融仍显示重建依赖，范围仅4个窗口。
训练工程完整；小数据集条件链路与动态Mask适应能力获得支持，
不证明多motion容量、动作物理可行或最终CVAE随机生成。

## 6. 下一步：32-motion B-fixed预算与人工评估标准

建议从随机初始化开始，不从16窗口checkpoint热启动：当前训练初始化严格要求相同窗口身份，
直接加载会明确报identity mismatch。不要用legacy开关绕过已知不匹配。
当前默认配置 `configs/posterior_hierarchical_standard_cvae_h50.json` 已为65-token架构；
data为T64、stride64、max_windows=null、max_episodes256。
实际窗口由数据索引决定，motion_count字段本身不执行抽样；启动前核对32 motion、256 episode、
预计1504窗口，并由完整selected_windows和fixture表记录覆盖。

| 项目 | 本轮建议 |
|---|---|
| stage / Mask / 初始化 | B / fixed / 随机；不调用Posterior、不开KL |
| 数据 | 同32-motion数据集全部1504个T64窗口；每窗口8类fixture，共12032 |
| 预算 | 上限120000步，batch32；预计384万样本暴露，每fixture约319次 |
| LR | warm-up2000到1e-4，cosine最终1e-6；原loss、weight decay和结构不变 |
| 工程记录 | 每步JSONL不裁剪；50步控制台；500步last；5000步完整双bank评测 |
| 中间审核 | 30k / 60k / 90k / 120k；新run启动前及step0核对身份、fixture数和路由 |

窗口/fixture规模比小fixed增加94倍，预算只增加6倍；不能保证120k达到小实验约1e-3的精度。
本轮检验更宽松的规模推进标准，而不是要求复制小实验完美记忆。
无证据支持仅为追求等暴露直接训练约188万步；先用中间曲线判断是否需要受控延长。
全窗口采样为window均匀，不是motion严格等权；完整覆盖全部motion/variant，
后续若某motion显著支配误差，单独报告其窗口数与误差，不能悄悄改采样分布。

建议在启动前记录人工审核协议 `B32-fixed-review-v1`，以下均为normalized、有效元素统计：

| 检查 | 规模推进参考标准 |
|---|---|
| pooled masked连续误差 | State RMSE≤0.01，Action RMSE≤0.01 |
| 各Mask家族 | 每个有目标的State/Action域micro RMSE≤0.02；零目标不作零误差 |
| 窗口分布 | 各家族有目标window的masked RMSE p95≤0.03 |
| 元素尾部 | pooled masked State/Action p99≤0.05；max、超0.1比例、最差坐标与物理误差单独审核，不设max硬门禁 |
| contact | masked contact accuracy≥99.9%；保存计数，零目标家族排除 |
| 稳定性 | 最后连续3次完整fixed评测达到上述数值条件；另审核末段趋势与最差样本 |

这些是拟采用的研究推进阈值，不是已有生物力学/物理安全标准，也不是代码内自动门禁。
当前quality_pass仍会为null，best.pt仍按原family等权masked MSE选择；不能修改marker来伪装已通过。
另一bank只作Mask迁移诊断，不作为本轮fixed记忆失败的单独否决项；
动态训练阶段另作对比，不能把训练内固定能力与新Mask能力混为一谈。

若最后约20k主指标仍改善至少10%、尾部没有恶化且接近参考标准，可人工审核同协议保留AdamW的
30k～60k延长；否则先诊断family/window覆盖与可见／隐藏误差，不默认追加步数或扩大网络。
这个10%是预算决策参考，不替代质量阈值；边界情况与非单调波动必须看完整曲线。
工程错误停止，质量告警不自动停止或改LR。
B-fixed通过后再启动同32-motion的B-dynamic；回放继续独立待执行。

## 7. 记录与精简回传

本轮B-dynamic回传解压472,447,630字节，evaluations占444,017,344字节（约94%）；
主要是每次重复保存完整曲线和诊断，不能用删除源训练日志解决。
32-motion降低完整评测频率至5000步，共25次含step0；仍完整评测全部fixture，两bank不抽样。

本地保留全部日志、评测、checkpoint不变；精简回传包保留所有manifests、最新plots、
完整metrics/evaluations JSONL、normalization和两bank、每次summary。
详细diagnostics/曲线只回传step0、best_step、末3次评测（去重最多5次），
及最近一次消融；不打包其他中间诊断、console大日志、HDF5、checkpoint或all_predictions。
用zip压缩并输出所选文件hash索引；压缩体积以实际打印为准，不保证固定MB上限。
发现特定中间异常时再补传对应step文件，不删本地证据。

当前只给出新训练与精简打包命令，没有在Ubuntu启动训练；不修改训练/采样/evaluator代码。
真实Isaac/MuJoCo回放仍未收到结果。Windows文档与CLI静态检查不替代新32-motion工程/质量验收。
