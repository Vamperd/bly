# CVAE 必要异常与根因诊断

最后更新：2026-09-29。本文只记录会影响实验判断的反常结果、证据、候选原因和验证边界，
不复制常规训练流水或整体实验路线。路线与概要结果见 [plan.md](plan.md)，模型合同见 [model.md](model.md)。

## 1. 本次证据与可比性

本地回传根目录：`C:/Users/86136/Desktop/replay/feedback`。
Ubuntu 重评根目录：`/home/helloworld/bly/runs/cvae_v2_reassessment_gweSqTsw`。
评测源码 commit：`b64df6499fab832246117521427548069dbba501`，协议 `65-token-experiment-v2`。
11 个回传包的 report_index 共342项文件哈希全部核对一致；这验证回传完整性，不等于独立验证数据真值。

当前 C 源 run 为
`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_kl_20260925_211113`，
采用 `stage=C`、dynamic Mask、32 motion、1504 T64 windows、360000 steps。C 的最终 summary 已回传，
但 step360000 尚缺逐元素 diagnostics；因此末步最差坐标和逐窗口 trace 仍属于待补证据，不能从 summary
中的 max 值反推出具体根因。

关键证据索引（以下相对路径相对于本地回传根目录）：

| 证据 | 用途 |
|---|---|
| `A_completed_training/logs/metrics.jsonl` | 60,000步训练、61次完整评测及学习率趋势 |
| `A_original_90k/evaluations/A/{summary,diagnostics}.json` | 原90k完整分布、argmax与曲线 |
| `A_continuation_best/evaluations/A/{summary,diagnostics}.json` | 续训最终分布、argmax与曲线 |
| `A_continuation_last/evaluations/A/diagnostics.json` | 与best逐字节相同的诊断内容；不声称checkpoint文件哈希相同 |
| `B_{best,last}_{posterior_condition,condition_zero_posterior,old_A_route}/evaluations/` | 同一checkpoint不同信息路径的受控比较 |
| `diagnostic_report/manifests/{standard_cvae_summary,audit_000000002}.json` | Ubuntu新B真实数据/CUDA工程smoke与梯度、路由审核 |

A原90k源run为 `cvae_posterior_hierarchical_standard_cvae_65_fixed_20260922_014417`；
60k权重初始化训练run为 `cvae_posterior_hierarchical_standard_cvae_65_fixed_20260923_012104`。
二者均在 `/home/helloworld/bly/runs/` 下。本轮为90k权重基础上新训练60k，不是保留AdamW的精确续跑。
旧checkpoint只记录局部step，重评中的 `cumulative_step=60000` 不代表权重只经历过60k训练。

本次各重评使用相同1,504个T64窗口、同一数据集、index与归一化：
normalization SHA256=`dad2270cb8466b5616cc6db63d49b065c0bebea65ac0440e60ccd82b6c918995`；
selected_windows SHA256=`5cb8002ce328bc750df3438dff495a1677f327980b9ed70f79a110f0769faca6`。
原90k和最终60k的重评指标均在浮点聚合误差范围内复现训练日志，max完全相同。
历史checkpoint缺少index/normalization/window-table哈希，报告的 `exact_identity_verified=false`
应保留；不得把本次重评一致性改写为历史逐步精确恢复保证。数据全部为已见训练窗口，不是泛化测试。

## 2. 异常A：平均误差很小，max仍明显较大

### 2.1 先纠正“无法下降”

| 指标 | 原90k | 再训练60k后的最终点 |
|---|---:|---:|
| full reconstruction loss | 1.777659e-5 | 6.988235e-6 |
| State / Action RMSE | 0.00588709 / 0.00431750 | 0.00373420 / 0.00264767 |
| 联合continuous p99（训练日志） | 0.01698315 | 0.01124605 |
| State p99 / p99.9（重评） | 0.01800912 / 0.03589776 | 0.01183576 / 0.01947615 |
| State max / Action max | 0.21955711 / 0.06426299 | 0.13111076 / 0.04675962 |
| State abs error >0.05 的元素数 | 2,326 | 222 |
| State abs error >0.1 的元素数 | 168 | 5 |
| State abs error >0.2 的元素数 | 1 | 0 |
| State max >0.1 的窗口数 | 43 | 1 |

State连续元素总数为6,647,680；最终>0.1仅占0.0000752142%，5个元素均在同一窗口、同一feature。
这些计数为窗口内元素口径，重叠窗口不可当成独立原始样本。0.01/0.05/0.1/0.2是诊断阈值，不是新设质量门禁。
loss、State RMSE、Action RMSE、联合p99、max分别改善60.69%、36.57%、38.68%、33.78%、40.28%。
但仍有全部1,504个窗口的State max>0.01，故不能称为逐元素无损重建。

局部step46k的联合max最低为0.12337743，60k为0.13111076；55k..60k在0.12494..0.13471间波动，
同期loss从7.233659e-6继续降至6.988235e-6。这支持“末段最大值非单调、降低很慢”，
不支持“整体训练失败”或“已证明达到不可突破的误差下限”。best按平均重建loss选择，所以best=last=60k，
不保证max最优；不能事后把46k称为已保存可用checkpoint。

### 2.2 两代最大元素不是同一个点

| 坐标 | 原90k最大点 | 最终60k最大点 |
|---|---|---|
| motion | confusion_103__A045 | confusion_103__A045 |
| episode | initial-state collection `demo_3022` | startup collection `demo_3020` |
| variant / window index / start | 7 / 277 / 384 | 3 / 242 / 448 |
| 相对t / 原始帧 | 46 / 430 | 64 / 512 |
| feature（零起始索引） | 43：joint_vel_14 | 57：joint_vel_28 |
| chunk / chunk内位置 | 11 / 2 | 15 / 4 |
| normalized target / prediction | 0.54266077 / 0.32310367 | -0.24081212 / -0.10970137 |
| mean / std（rad/s） | 0.11093386 / 1.09617913 | -0.00662449 / 0.56711894 |
| physical target / prediction（rad/s） | 0.70578724 / 0.46511337 | -0.14319360 / -0.06883821 |
| physical absolute error（rad/s） | 0.24067391 | 0.07435539 |

原最大点所在窗口277的State max从0.219557降至0.07440391，因此旧异常至少已降到该上界以内；
最终未导出窗口277完整曲线，不能伪造同一个原始元素的精确末值。
最终最大的5个元素均来自window242的joint_vel_28：t=64/12/58/31/61，分别为
0.131111/0.118628/0.117837/0.108232/0.106339，并不只发生在终点或chunk边界。

物理误差只比较同单位feature：0.07436 rad/s是“normalized最大点”的物理误差，
不是全体关节速度物理误差的最大值。按完整feature表，最终速度域physical max约0.13566021 rad/s，
来自joint_vel_10；不能混合rad、rad/s、m作为一个物理max或RMSE。

## 3. A尾部的原因：证据强弱分开

### 3.1 已证实的局部现象：快速速度变化没有被逐帧充分还原

原90k window277、joint_vel_14在t45/46/47的真实速度为0.25769/0.70579/0.15158 rad/s，
预测为0.27007/0.46511/0.16549。主要是t46孤立峰值的幅值被低估，而不是峰值整体移动一帧。
其整窗速度标准差target/pred为0.07268/0.04662 rad/s，一阶差分标准差为0.09321/0.05130 rad/s。

最终window242、joint_vel_28在t60..64：

| t | target（rad/s） | prediction（rad/s） |
|---|---:|---:|
| 60 | -0.032426 | -0.080202 |
| 61 | -0.117347 | -0.057040 |
| 62 | -0.019014 | -0.043269 |
| 63 | -0.044155 | -0.081132 |
| 64 | -0.143194 | -0.068838 |

该65帧曲线的一阶差分标准差target/pred为0.07371/0.03953 rad/s，预测只有约54%的帧间变化幅度；
整窗标准差为0.04914/0.03312 rad/s。存在局部振荡幅值还原不足、过度平滑样的现象。
但其整窗绝对峰值target/pred为0.16684/0.17290 rad/s，预测峰值并非总是偏小；
不能将全部残差归结为“统一缩小速度幅度”。上述现象也不证明速度标签是错误或噪声，原始物理数据未复核。

最终State top100全是关节速度，53项joint_vel_28、29项joint_vel_27；
86项来自confusion与neutral_looking_around两个motion。窗口级top100只对应94个不同原始元素；
独立去重top100中这两种feature仍占54+28项，集中现象不是纯粹的重叠窗口重复造成。
整体feature RMSE最差项却已转为joint_pos_23/24等：平均误差重点与极端尾部重点不同。

### 3.2 最有依据的机制解释：均值目标与稀疏极值的优化优先级不同

现有训练优化有效State MSE、Action MSE、contact BCE三项均值，不直接优化max。
最终最大单元素的平方误差只占全体State SSE约0.01854%。这解释了为什么其变化不控制整体loss，
也解释了60k平均loss最优而46k的max更小；并不表示该元素没有梯度。
稀疏速度变化与共享编码/解码的局部拟合偏差是优先候选，具体由Encoder、pooling或Decoder哪部分造成，
现有前向结果不能分离。不能直接宣称扩大模型即可消除尾部，或MSE在理论上不能拟合这些点。

### 3.3 已检查而不支持的归因

| 假说 | 现有证据与边界 |
|---|---|
| 近零std放大 | 29个速度std为0.3170..1.9636；最大点对应std也正常。不支持数值奇异归一化；归一化仍会改变不同feature极值的排序 |
| 全局错一帧/两帧 | 两代各4条代表窗口、每条29速度的±2帧辅助诊断，均为lag0最小；没有系统性整帧错位证据。该诊断使用t2..62共同中心区，不能排除终点局部相位误差或更小时间偏移 |
| State_64丢失或local_15失效 | t64有102,272个连续目标；最终RMSE0.003593低于整体0.003734。原90k最大点在t46，最终其他>0.1点在内部；不支持普遍terminal缺失 |
| 硬chunk边界系统性断裂 | 最终chunk内位置0/1/2/3的RMSE为0.003695/0.003755/0.003767/0.003728，终点位置4为0.003593；无边界平均误差抬升证据。不排除个别窗口的上下文敏感性 |
| Mask采样错误导致A尾部 | A不调用Condition或Mask采样，不适用；旧B的Mask问题不可迁移成A根因 |
| KL或梯度爆炸 | A60k全部kl_beta=0，逐步标量有限，梯度范数最大约0.01205；不支持这两种解释 |

同一原始帧在重叠窗口中的State预测最大跨度从0.176749降到0.108694（归一化），
说明上下文/窗口位置依赖仍存在。但报告只保存全局跨度，没有argmax的配对坐标；
尚不能证明最终t64最大误差就是由窗口位置导致，也不能直接据此修改chunk。

## 4. 异常A：重启后先变差，最终才获得收益

`INIT_RUN`加载原90k best模型，但重置AdamW、scheduler和采样器。原末端LR=1e-6，
新段2k warm-up升至2e-5（原末端20倍）再cosine降至1e-6。
step0精确复现原基线；2k loss升至7.449978e-5（约4.19倍），直到局部29k的完整评测才首次优于基线。
最终60k loss下降60.69%，这次训练有效但前段效率低，不能再依据14k快照称其为最终失败。
高LR重启与AdamW重置共同解释反弹的候选机制；二者同时改变，不能独立归因。
若另做优化消融，应从同一checkpoint配对比较保留/重置optimizer或LR，不能同时换loss、采样和结构。
旧日志learning_rate记录在scheduler.step之后，是下一次更新的LR；终点目标约1e-6成立，
但旧日志不能当成精确的每步实际更新LR。v2分别记录actual/next LR。

## 5. 异常B：低训练loss与高评测loss来自不同信息路径

旧run `cvae_posterior_hierarchical_standard_cvae_65_random_20260922_152852`实际上限150k，
134,776步中断，best=134,000；不是正常完成的60k condition-only训练。
此前回传源码确认其B训练仍读取完整posterior，与condition融合；评测却固定走无condition的A。
新重评采用同一新固定bank（1,504窗口×8=12,032 fixture），不是精确回放旧batch位置依赖Mask。

| 同一旧B last checkpoint的路径 | full State / Action RMSE | masked State / Action RMSE | State max |
|---|---|---|---:|
| posterior mean + condition | 0.011087 / 0.007931 | 0.021973 / 0.011263 | 3.951270 |
| posterior latent=0 + condition | 0.655503 / 0.768258 | 1.223946 / 0.846521 | 39.442398 |
| 旧A评测路径，无condition | 0.199735 / 0.122331 | 不适用 | 22.512138 |

best checkpoint的对应full State/Action分别为0.011105/0.007941、0.655103/0.767954、0.199777/0.122421，
结论一致。旧best的A路由重评max=22.5015106复现旧日志，确认此前高分来自错误评测路径。
去掉posterior后masked RMSE约恶化55.7倍/75.2倍，visible也明显恶化；说明旧权重强依赖posterior输入，
但零latent是训练分布外干预，不足以证明独立重训的Condition Encoder能力上限。
原训练路径仍有尾部：last最大点是crouch_operating_cupboard_mid_in_R_003__A299、variant5、start0、t1、
joint_vel_14、state_rollout。masked State在joint_gap_2/8为0.04670/0.03991，full_action的Action为0.00848。
难度并不简单随Mask长度单调变化；缺口上下文、目标数和旧训练fixture覆盖均为混杂项，
不能由此称“full Mask太苛刻”，也不能据此扩大模型或再延长旧B。

## 6. 当前记录仍缺什么，下一步如何区分原因

固定trace只保留fixture0/1；当前最差trace随checkpoint变化，原最差277与新最差242未在两代同时导出。
因此可以确定分布改善与峰值转移，但不能完整重建同一极端坐标的训练轨迹。
必要的只读补诊应固定：window277/t46/feature43、window242/t64/feature57，以及覆盖相同原始帧的重叠窗口。
两代checkpoint都导出这些窗口，报告同帧不同窗口预测、邻帧值、同单位误差与真实记录；不重采数据、不改标签。
新的“异常固定窗口”应独立于每次动态最差窗口，不把旧窗口从top100消失当作误差归零。

A本轮结果封存为新的已见序列重建基线，不原样追加60k，不把max降到某任意阈值作为启动B的硬门槛。
若后续确需压低速度尾部，先做上述坐标配对和原始数据核查；确认仍为局部变化还原不足后，
再单变量比较尾部加权/采样、局部解码路径或pooling，而非同时换loss、latent、学习率和窗口分布。
导数损失会放大快速变化，也可能放大真实数据噪声，不能在数据来源未核查前默认启用；
数据清洗、平滑、裁剪异常值会改变任务，需单独授权，不能用于掩盖max。

整体实验推进与正常工程验收只记录在plan.md；本文件不将上述候选机制升级成已证明根因，
不以一次尾部消融替代独立condition-only能力检验。

## 7. 原始Action回放漂移：已知异常与证据边界

历史H50单窗口未exact-init回放中，记录→原Action的joint RMSE=0.245825rad、root RMSE=0.251772m、
orientation max=106.918°；但原Action→模型Action的joint RMSE仅0.008448rad。
这说明“原始与模型回放接近”不能证明复现了采集轨迹。随后一个exact-init已见T64窗口通过，
但完整episode实验仍出现初始化通过后原始Action长时漂移，因此随机化未恢复不是所有案例的已证唯一根因。

2026-09-23代码检查确认：既有初始化恢复暴露的状态、动力学参数和前一Action，不包含完整仿真快照。
执行器延迟队列、历史实际delay draw和PhysX接触缓存没有在该恢复包中记录；源码存在延迟执行器，
不等于当前采集必然启用非零延迟。需读source schema与runtime报告，不能据类定义认定异常原因。
新回放先分别验证初始化、原始重复性、原始对记录复现。两次原始回放同样偏离真值仍判基线无效；
基线无效时模型物理质量不可判定，视频可继续用于诊断。当前没有新的Ubuntu回放证据，
候选机制为参数/初态不一致、未恢复内部历史或接触状态、控制映射/时序与开环漂移；均不可写成已确诊。

## 8. 32-motion B-fixed：固定Mask记忆与新Mask泛化分离（2026-09-24）

回传包：`C:/Users/86136/Desktop/replay/32motionfixB/compact_return_20260924_124152`。
Ubuntu run：`/home/helloworld/bly/runs/cvae_v2_B_fixed_m32_s120000_tWXXU9fj`。
训练合同为`stage=B`、`mask_mode=fixed`、32 motion、256 episode、1504 window、12032 fixture、
batch 32、120000 optimizer step、3,840,000 exposure；参数量64,377,959。工程完成、有限值、
checkpoint readback和contact路径均正常，contact accuracy为100%。

### 8.1 末步分域指标

| 指标 | fixed bank | held-out bank |
|---|---:|---:|
| selection / masked MSE | 0.000328276 | 0.007004692 |
| full State RMSE | 0.0124872 | 0.0340117 |
| full Action RMSE | 0.0102388 | 0.0197627 |
| masked State RMSE | 0.0206887 | 0.0791436 |
| masked Action RMSE | 0.0129615 | 0.0352613 |
| masked State p99 | 0.0640906 | 0.288760 |
| masked Action p99 | 0.0379806 | 0.131179 |
| State max abs | 0.771466 | 11.707875 |
| Action max abs | 0.157573 | 4.282176 |

fixed visible State/Action RMSE为0.0099489/0.0095198；held-out visible为0.0126964/0.0138670。
因此误差主要由隐藏目标补全产生，而非Decoder对可见序列的整体损坏。

### 8.2 尾部坐标与根因边界

fixed最大点为`confusion_103__A045`、variant 1、window_start 448、`state_gap_16`，相对帧30、
State feature 47（`joint_vel_18`），normalized target/prediction为1.843302/1.071836，
误差-0.771466，对应物理误差-0.47590 rad/s。它是稀疏fixed尾部，不能代表整体崩溃。

held-out最大点为`crawl_ff_stop_225_R_003__A233`、variant 7、window_start 321、`joint_gap_8`，
相对帧31、`joint_vel_8`，normalized target/prediction为-13.48877/-1.78090，
误差11.70788，对应物理误差约3.71169 rad/s。后续最大点还集中在该motion及`joint_vel_26/27/28`，
且masked State中17.4%的元素超过0.05、6.34%超过0.1，故held-out失败不是单个max离群点。

fixed masked State最难家族为`joint_gap_2`（0.028645）和`joint_gap_8`（0.020668）；
held-out最差也主要是新坐标的`joint_gap_8`。这支持“Mask位置/缺口条件泛化不足和关节速度尖峰难补全”，
不支持仅靠延长同一fixed训练解决，也没有证据要求立即扩大主干。

### 8.3 趋势与决定

110k→120k fixed selection仅改善约3.4%，State max仅由0.774821降至0.771466；held-out在约70k后
平台。结论是：本run工程成功但fixed与held-out质量均未通过；不做同run resume或普通continue。
下一实验使用`best.pt`进行B-dynamic的model-only初始化，重新建立optimizer/scheduler和动态Mask，
固定其余架构、loss、batch和学习率。B-dynamic用于检验“稳定身份驱动的Mask变化训练能否改善新坐标泛化”；
其结果仍不能直接等同于新motion泛化。该条是B-fixed阶段的当时门禁；B-dynamic已在第9节完成审核，
不把fixed阶段的max尾部直接解释为模型容量缺陷。

## 9. 32-motion B-dynamic：held-out改善与fixed回退（2026-09-25）

回传包：`C:/Users/86136/Desktop/replay/32motionradomB/compact_return_20260925_195133`。
Ubuntu run：`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_random_20260924_125341`。
合同为`stage=B`、`mask_mode=dynamic`、32 motion、1504 window、batch32、120000 step、3,840,000 exposure；
参数量64,377,959。`execution_pass=true`，Posterior调用0、Condition调用1、KL=0，checkpoint readback通过。
`manifests/command.txt`中的shell seed为20260824，而`effective_config/training_contract`实际记录
`initialization_seed=20260921`、`training_mask_seed=20260920`；本次分析以effective config为训练身份。
这不构成数值训练失败，但后续启动命令应消除该身份差异。

### 9.1 与B-fixed源基线的配对结果

| 指标 | Dynamic step 0 / fixed源 | Dynamic step 120000 fixed | Dynamic step 120000 held-out |
|---|---:|---:|---:|
| selection MSE | 0.000328276 / 0.007004692 | 0.000514385 | 0.000551900 |
| full State RMSE | 0.0124872 / 0.0340117 | 0.0125561 | 0.0123264 |
| full Action RMSE | 0.0102388 / 0.0197627 | 0.00928084 | 0.00941772 |
| masked State RMSE | 0.0206887 / 0.0791436 | 0.0235355 | 0.0229650 |
| masked Action RMSE | 0.0129615 / 0.0352613 | 0.0122744 | 0.0132373 |
| masked State p99 | 0.0640906 / 0.288760 | 0.0811819 | 0.0816900 |
| masked Action p99 | 0.0379806 / 0.131179 | 0.0407310 | 0.0460907 |
| State max abs | 0.771466 / 11.707875 | 0.857141 | 1.139175 |
| Action max abs | 0.157573 / 4.282176 | 0.425431 | 0.596650 |

Dynamic把held-out masked State/Action RMSE从0.079144/0.035262降到0.022965/0.013237，
State max从11.707875降到1.139175；这是明显的Mask坐标泛化改善。fixed bank masked State从
0.020689升至0.023536，selection MSE上升56.7%，连续三次评测触发`primary_regression_over_20_percent_for_3_evaluations`。
因此Dynamic不是无条件质量通过，而是出现了清晰的fixed/held-out trade-off。

### 9.2 尾部和Mask family

最终fixed最难的State/Action family为`joint_gap_2`（0.040758/0.020884）和`joint_gap_8`
（0.036909/0.024499）；held-out为0.041275/0.023316和0.041529/0.028136。
held-out最大State点为`neutral_looking_around_R_001__A542`、variant 3、window_start 0、
`joint_gap_2`、相对帧2的`joint_vel_8`，normalized误差1.139175，物理误差约0.361147 rad/s。
最差window主要仍集中在`joint_gap_8`，但尾部已从B-fixed的11.7/4.28降到1.14/0.60。

held-out坐标分组中，已见坐标masked State/Action为0.021116/0.009718，新坐标为0.023062/0.018286；
新坐标仍较难，但不再出现B-fixed阶段的数量级差异。最终`best_step=0`，因为fixed primary从未超过
Dynamic step0源基线；`best_heldout_step=120000`，真正的Dynamic质量checkpoint是`best_heldout.pt`
或`last.pt`，不是`best.pt`。

### 9.3 阶段结论

已验证：Dynamic Mask训练显著改善同32-motion数据上的新Mask坐标泛化，并保持Condition-only路由、
完整重建和contact路径可用。未验证：masked RMSE达到0.01、新motion泛化、C的posterior/标准正态路径
和物理回放。最后20k held-out selection改善9.42%，最后10k仅改善2.92%，学习率已到1e-6，
继续同一Dynamic合同的预期收益有限。因此当时的下一步是进入C独立随机初始化训练；该C已在第10节完成，
仍不使用本B checkpoint，也不把本次`best.pt`误作为Dynamic最佳模型。

## 10. C阶段：训练执行有效，但标准正态部署出现latent路径失配（2026-09-29）

回传包：`C:/Users/86136/Desktop/replay/C train/cvae_C_return`。
Ubuntu run：`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_kl_20260925_211113`。
合同为`65-token-experiment-v2`、`stage=C`、`mask_mode=dynamic`，32 motion、256 episode、
1504个T64窗口、batch32、360000步、11520000 exposure，参数量64,377,959。
`source_checkpoint=null`，所以没有继承A/B权重。有效初始化／Mask seed为20260921／20260920；
`command.txt`另写seed20260824，属于启动元数据不一致，不能用它替换effective config的训练身份。

### 10.1 已排除的工程性故障

`logs/metrics.jsonl`有连续360000条训练记录、无step缺口；73次evaluation；每步batch均为32，
Posterior和Condition各调用一次；loss、reconstruction、KL、gradient norm全部有限。末步
loss/reconstruction/KL/gradient分别为`4.14769e-5`、`3.29129e-5`、`8.56401e-3`、
`1.10230e-3`，学习率到达`1e-6`。最终checkpoint严格读回、optimizer/scheduler读回、forward
comparison和SHA均通过，execution marker存在，summary为`execution_pass=true`、
`quality_pass=null`、warnings为空。因此这次不是中途崩溃、NaN、batch短缺、路由未执行或读回损坏。

### 10.2 主要反常：posterior重建好，部署先验差

末步fixed bank归一化指标如下；masked RMSE采用8次sample的单样本期望聚合：

| 路径 | energy | masked State / Action RMSE | full State / Action RMSE | State / Action max |
|---|---:|---:|---:|---:|
| posterior mean | 0.013585 | 0.016604 / 0.007912 | 0.008654 / 0.006309 | 0.7277 / 0.2511 |
| posterior sample | 0.012063 | 0.017468 / 0.008526 | 0.009039 / 0.006644 | 0.9293 / 0.2525 |
| standard normal | 0.027423 | 0.081051 / 0.040847 | 0.036627 / 0.026234 | 14.3284 / 2.4324 |
| zero latent | 0.032803 | 0.070393 / 0.033941 | 0.031667 / 0.021927 | 14.7097 / 2.6927 |

posterior mean到standard normal的masked State／Action RMSE约放大4.88／4.83倍；
zero latent的单样本masked误差反而小于standard normal，但energy较差。这表示Decoder在有真值
posterior时能拟合已见窗口，随机部署时latent分布／使用方式没有校准好；不能把这次结果简化为
“模型容量不足”，也不能把posterior均值的低误差写成C已具备生成能力。

held-out standard-normal energy为`0.026876`，full State／Action RMSE为`0.034439`／`0.025993`，
masked State／Action RMSE为`0.078670`／`0.041141`，max为`17.9064`／`1.8786`。
已见坐标与新坐标的masked micro RMSE分别为State `0.073941`／`0.077882`、Action
`0.042200`／`0.038218`；两者同属32-motion已见序列。这里没有B-fixed那种新坐标数量级断崖，
所以当前首要异常是posterior训练路径到标准正态部署路径的失配，而不是已证的Mask坐标过拟合。

### 10.3 候选机制与证据边界

step345000的4个跨motion窗口只测posterior mean依赖：global latent置零／置换几乎不改变
State／Action MSE（约`4.60e-5`／`2.67e-5`），local置零／置换明显变差；condition memory
置零／置换把MSE升到`0.398`／`0.495`或`1.454`／`1.789`，FiLM置零／置换升到
`0.0263`／`0.0347`或`0.0387`／`0.0436`。这支持Decoder主要使用condition memory和FiLM，
并提示global通道可能低利用；末步global KL约`1e-5`而local KL约`1.7e-2`与此方向一致。
但消融窗口只有4个、使用posterior mean，不能单独证明global latent完全坍缩、不能证明条件路径
足以独立生成，也不能区分KL权重、latent尺度、融合门控和训练覆盖的贡献。

standard-normal energy在300k→360k由`0.031322`降至`0.027423`，最后20k改善约2.17%、
最后10k约0.86%，且学习率已经是`1e-6`。这说明继续同一合同的收益正在平台化，不能把普通续训
当作已证根因修复。当前不同时更改loss、latent维度、pooling和网络宽度。

### 10.4 回传证据限制与下一项诊断

包内`evaluations_summaries/step_000360000/`有最终五路summary，但`evaluations_detail/`没有
step360000的逐元素diagnostics；只有step0完整diagnostics以及step20000至345000的消融文件。
因此本报告不能诚实给出末步最差window、最差coordinate或末步逐坐标trace。需要在Ubuntu源run
对step360000做一次只读、紧凑的尾部导出，补齐这些证据和posterior/global/local统计。

在补齐尾部证据之前，不把C标记为质量通过，也不启动新的结构性训练分支。

## 11. replay65 baseline：Isaac结束后marker目录缺失（2026-09-28）

失败run：`/home/helloworld/bly/runs/cvae_replay65_C_w0_lS0WeyvH`。
首个错误发生在原始Action双基线 worker 退出阶段：Isaac recorder 已打印
`Closed Isaac Lab recorder before process exit`，随后`sonic_repro.sh`的`mark_stage`尝试写
`simulations/baseline/markers/action_mask_replay.ok.tmp.$$`，但该子目录没有`markers`。
因此首先失败的是shell完成标记封存；这条日志本身不能把物理回放判成成功，也不能把模型判成失败。

根因是`replay65.py::simulate_group`原来只创建`child/data`，而`sonic_repro.sh`的
`phase_replay_action_mask`只验证路径、不调用通用`ensure_run_layout`。首个wrapper返回1后，
用户脚本仍继续执行`report`、第二次`simulate`、`render`和导出；第二次simulate发现已有但没有
`replay65_worker_complete.json`的baseline子目录，按保护逻辑报
`incomplete/changed simulation is not overwritten`。随后生成的MP4不能作为完整执行验收，
导出的报告也不能覆盖首次工程失败。

Windows已修复并通过静态／单元检查：Python在启动worker前创建`data/manifests/logs/markers`，
shell的`mark_stage`和Action-mask阶段也有防御性目录创建。`test_replay65.py`的12项通过，
`py_compile`、`bash -n`和`git diff --check`通过。Ubuntu必须同步这两个文件后使用新的run；
旧失败目录不复用、不手工补marker冒充完成。

本轮困难Action回放统一选择`joint_gap_8`：它同时有State和Action缺口，且Action目标存在，
会执行模型Action物理回放；`state_gap_16`是State-only，不会产生额外模型Action replay。
`first-suite`仍按每个checkpoint的八类Mask masked MSE选择最差窗口，同时保留episode起点和
首个非零起点作为对照；这比查看视频后手工挑选单个窗口可复现。A路由仍按合同执行full prediction，
`joint_gap_8`在A中是困难窗口的诊断标签，不会把A误写成condition-only或masked Action训练。
随后将三条路线改为共享同一个最难motion/window；A生成四个代表性State视频并只执行一次等价的
Action物理回放，B/C生成四个代表性State视频及各自的四个Action物理回放视频。

## 12. 固定 64 帧原始 Action 回放合同（2026-09-28）

用户已回退把后 64 步交给 SONIC policy 或用模型 desired-State 外推的改动。由于延长段
仍然出现跨进程物理差异，本轮不再延长视频。物理问题定义是 open-loop：同一 motion、同一
初始化、同一仿真参数下，source 窗口内的 64 个 raw Action 必须逐步送入 Isaac。
`latent_sweep65.py` 现在只写入和验证 T64 Action、65 帧 State，不补 hold-final，也不读取
窗口后的 HDF5 Action。这样可以把问题限定在窗口内；source HDF5 只需要提供同一窗口。

当前代码把 `terrain.friction_combine_mode` 和 `terrain.restitution_combine_mode` 写入
`replay65_contract`。`exact_replay_initializer.py` 在绑定专用 ground material 后记录期望值、
运行时配置值和 `combine_modes_match`，并对所有已记录的数值 runtime context 做读回比较。
`runtime_context_max_abs > 1e-5` 或 ground combine mode 不匹配会留下失败的 readback JSON 并
停止 worker；因此摩擦数值、恢复系数、combine mode、关节/刚体参数、gravity、solver、dt 等
可见差异不会再被视频掩盖；非零且无法恢复的 actuator delay queue 也会让 exact
initialization 失败。

这项修复不能保证跨独立 Isaac 进程 bitwise 相同：Physics State–Action v3 没有记录 PhysX
接触 warm-start/cache，也没有记录非零 actuator delay queue 的历史内容。runtime audit 会把
前者记为 `solver_contact_cache=unknown_not_recorded`；若配置出现非零 delay，exact worker
会停止而不是继续输出视频。若第一处 divergence 出现在可见参数全部通过之后，应将物理结论写成“隐藏模拟器状态未确定”，
而不是继续猜测地面摩擦或启动训练修改。

Windows 静态测试覆盖了 T64 source-window 回放和 combine-mode/读回合同检查。Ubuntu 下一次
必须建立全新 run，确认每个 child 的 `action_replay_request.json` 为
`post_action_mode=none`，再检查 `exact_initialization_readback_*.json`、`*.runtime.json`
和 `recorded_to_original_1_errors.npz` 的首个阈值帧。旧的 129-frame、hold-final 或
policy-continuation run 不得作为本轮原始 Action 一致性证据。

## 13. 运行时优化的证据边界（2026-10-04）

本轮实现针对 v2 runtime，不改变数据统计或 Action 语义。源码现在显式区分 FP32 保留区与
BF16 autocast 区域，并把 compile、BF16、TF32、cache、pin-memory 和 non-blocking 选项写进
runtime contract；旧 C 的 FP32 eager 结果仍是历史基线。标准化窗口缓存只写入当前新 run，缓存
identity 绑定 dataset manifest、episode index、normalization 和 selected-window hash，缓存失效
或超预算时回到同步 HDF5 读取。

Windows 已通过 Python compile 与现有 v2 posterior/protocol 测试，并新增缓存 identity/selected
index 回归测试。尚未在 Ubuntu RTX 4090 上执行四类 BF16/compile smoke、数值 parity 或稳态基准，
因此目前只能报告 execution implementation；不能把 compile callable 创建成功、缓存文件生成或
静态测试通过解释为训练吞吐提升、BF16 精度满足阈值或 standard-normal 质量改善。若 compile 在
运行时触发 graph break/异常，summary 必须保留 fallback 原因并按既定门槛选择 eager 路径。

## 14. C latent 分组诊断与 KL×2 实验边界（2026-10-04）

C 当前已知现象是 posterior mean/sample 重建较好，而部署 standard-normal 误差明显增大；全局
平均 KL 很小只能说明平均意义上的 `q(z|x)` 接近 `N(0,I)`，不能说明十七个 latent group 或
每个维度都同样接近。新增 `latent_distribution65.py` 后，诊断会对完整 State--Action window
逐批调用 Posterior Encoder，并按 `global` 与 `local_00`--`local_15` 分组输出：

- `kl_mean/median/p95/max` 与逐维 KL 分数，显示是否只有少数维度承载偏离；
- posterior mean 的跨窗口方差，区分有信息变化、均值坍缩和仅 logvar 承载变化；
- posterior sigma 的均值/分布，显示不确定性是否集中在某些 local chunk；
- 重参数 sample 的经验均值/标准差、直方图和相对标准正态的 gap。

因此在没有实际 `group_stats.json`、逐维热图和 standard-normal 对照前，不能断言“global 已
坍缩”或“local 只有一部分有效”；这些是待验证的机制假设。诊断只读旧 checkpoint 和重建数据集，
不改变 replay、训练和质量门禁。

KL×2 运行是单变量对照：仍从随机初始化开始，保留原 beta warmup 10000 steps、BF16/compile
配置和 360k C 步数，只把 effective `kl_beta` 改为 0.002。若该 run 的 standard-normal
误差下降，同时 local/global 的 KL 分布没有出现单组过度收缩，才支持“KL 权重不足”这一候选
机制；若 posterior 重建变差但部署路径不改善，应停止继续放大 KL，转查 posterior-to-prior
校准、condition 融合和 decoder 对 latent 的利用。此 run 的 `execution_pass` 与 `quality_pass`
必须分开记录。

## 15. KL×2 smoke 的 CUDA Graph 输出覆盖失败（2026-10-05）

失败 run：`/home/helloworld/bly/runs/cvae_kl2_smoke_ReqoL9gV`。该 run 已完成数据加载、BF16
能力检查、标准化 cache 建立和 checkpoint 写入，但在 `optimizer_step=0` 的第一次 compiled C
forward 失败。`failure.json` 的根因是 PyTorch 2.7 CUDA Graph 保留的输出 tensor 被后续 compiled
调用覆盖：

```text
accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run
```

因此这不是 BF16 不支持、磁盘不足或 KL beta 配置错误；summary 显示
`cuda_bf16_supported=true`、`compile_requested=true`、`cache_enabled=true`，但训练没有完成第一个
optimizer update。此前 Triton autotune 的 shared-memory `Ignoring this choice` 只是候选 kernel
被跳过，不能单独视为根因。

Windows 已在 `cvae_training.py::_compile_callable` 的 compiled wrapper 前调用
`torch.compiler.cudagraph_mark_step_begin()`，并在旧版 torch 没有该 API 时安全跳过。这个边界
同时覆盖 C 的 train callable 和四条 evaluation callable；eager 路径不改变。旧失败目录保留为
工程失败证据，不补 marker、不复用；修复后必须建立新的 smoke run，并检查 `failure.json`、
`progress.json`、`compile_info` 和最终 execution marker。
