# CVAE 必要异常与根因诊断

最后更新：2026-09-23。本文只记录会影响实验判断的反常结果、证据、候选原因和验证边界，
不复制常规训练流水或整体实验路线。路线与概要结果见 [plan.md](plan.md)，模型合同见 [model.md](model.md)。

## 1. 本次证据与可比性

本地回传根目录：`C:/Users/86136/Desktop/replay/feedback`。
Ubuntu 重评根目录：`/home/helloworld/bly/runs/cvae_v2_reassessment_gweSqTsw`。
评测源码 commit：`b64df6499fab832246117521427548069dbba501`，协议 `65-token-experiment-v2`。
11 个回传包的 report_index 共342项文件哈希全部核对一致；这验证回传完整性，不等于独立验证数据真值。

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
