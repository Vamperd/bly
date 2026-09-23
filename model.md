# 65-token 层级 CVAE：活动模型与实验合同

模型架构版本保持 `65-token-hierarchical-standard-cvae-v1`；实验协议为
`65-token-experiment-v2`，checkpoint 格式为
`sonic_65_token_hierarchical_standard_cvae_checkpoint_v2`。
本轮不改模型参数、融合、FiLM、数据或重建目标。历史 H50 checkpoint 不迁移。
工程 PASS 与质量判断严格分开；无预注册质量阈值时 `quality_pass=null`。

## 1. 数据、编码与解码

数据为 `states[T+1,70]`、`actions[T,29]`，语义为
`State_t + Action_t → State_(t+1)`。70 个 State 分量依次为关节位置29、关节速度29、
基座线速度3、角速度3、重力方向3、高度1、接触2；Action 为 canonical 关节目标29。
当前模型不使用 RobotInfo、reference、motion ID 等元数据；身份字段只供采样和诊断。

完整输入 `X[B,65,99]=[State_t(70),Action_t(29)]`。终点 `Action_64=0`、
有效性为 false，不输出或计入 Action loss；State_64 保留。
条件输入 `C[B,65,198]` 是每特征 `[value_i,mask_i]` 交错，True/1 表示隐藏，
隐藏值先置零。输出 State 为 `[B,65,70]`，Action 为 `[B,64,29]`。

Posterior 使用独立 CLS_q，从完整 X 得到 global mean/logvar `[B,256]`、
local mean/logvar `[B,16,128]`，完全不读取条件 Mask。
Condition 使用独立 CLS_c，从 C 得到确定性 global `[B,256]`、local `[B,16,128]`
及65个 condition memory token。CLS_c 不重复进入 decoder memory。

双 Encoder 自注意力均双向。硬 chunk pooling 范围为
`0..3,4..7,...,56..59,60..64`；最后 local_15 包含 State_64。
池化作用于 attention 后的 hidden token，不是直接平均原始物理轨迹。

两组参数独立的 gated-residual MLP 分别用于 memory 和 FiLM。
global 融合 posterior/global 与 condition/global；local 还拼接 slot embedding。
A 的 condition 半部为零；B 的 posterior 半部为零，没有 posterior forward。
有条件时 memory 为65 condition +1 global +16 local，共82 tokens；A 仅17 tokens。
FiLM 输入为 global 投影 + 每个 t 对应 chunk 的 local 投影，shape `[B,65,d_model]`。
Decoder query 使用时间 embedding；每层独立零初始化 FiLM head，
顺序为 self-attention → cross-attention → FiLM → feed-forward。

默认宽448、双 Encoder 各6层、Decoder 8层、heads8、FFN1792，
总参数 **64,377,959**，运行时重新计数写入签名；A/B/C 的可训练参数数目分别记录。

## 2. 实验路线与目标

```text
已完成 A + 正在运行的旧协议 B（让其完成，保留历史）
→ v2 工程核验 → 旧 A/B 只读重评
→ B-fixed 随机初始化 → B-dynamic 从修正版 B-fixed model-only 初始化
→ C 独立随机初始化
```

A：只调用 Posterior，使用 mean 和17-token memory；完整序列重建，无 KL。
B：只调用 Condition，使用 condition memory/global/local、fusion、FiLM、Decoder；
不执行 Posterior、不计算 KL。B-fixed 的每个窗口显式展开八类物理 Mask，
训练和 exact 评测相同；B-dynamic 保持八类语义，由稳定身份、seed、累计 step、
sample ordinal 显式采样，重置 optimizer/scheduler，使用独立固定评测 bank。

C：从第一步开始 posterior 重参数采样，Condition 确定性；完整模型随机初始化，
不拼接 A/B checkpoint。最终推理只用 Condition + 标准正态 global/local，
不读取完整真值、不调用 Posterior。训练为
`reconstruction + beta * 0.5 * (KL_global.mean + KL_local.mean)`，
KL 的两个域分别按各自全部元素求均值；不使用 conditional-prior KL 或 latent alignment。

三阶段重建目标均保持 State 连续 MSE、Action MSE、contact BCE 三项均值，
仅统计有效元素。评测的 masked 选择分数不改变训练 loss。
B-fixed/B-dynamic/C 默认各60k，batch32、峰值1e-4、warm-up2k、
cosine 最后一次实际更新为1e-6；A 保持40k默认，可 CLI 覆盖。
C beta 默认1e-3、warm-up10k。训练预算不是质量承诺，不自动进入下一阶段。

八类 Mask 为 state_gap_4/16、state_rollout、action_gap_4/16、full_action、joint_gap_2/8。
State gap 为双向补全；state_rollout 给定整段 Action，不能叫严格逐步因果 rollout；
full_action 可能多解。所有当前32-motion窗口均为 train，只支持已见序列实验，
held-out Mask 不等于 held-out motion。
held-out bank 中 full_action 等 fixture 可能与训练bank坐标相同；评测显式记录重合数，
并将真正新坐标与相同坐标分组统计，不能把整份bank一概称为未见Mask。

## 3. 身份、评测与尾部诊断

稳定窗口身份含 source/episode、motion、variant、start、有效 State/Action 长度，
禁止 batch 内序号回退。运行保存解析配置、CLI、commit、相关源码哈希、
dataset manifest/index/normalization/window-table 哈希及源 checkpoint 路径/step/hash。
固定/held-out bank 保存实际隐藏坐标与哈希；训练样本、Mask 摘要与暴露量逐步记录。

只读评测入口 `python -m cvae_sa.cvae_tools evaluate` 必须指定 checkpoint、dataset、
route、新 output run。历史65-token可读但未知身份显式标注；历史 B 原始评测不是
exact fixture，重评也不是原训练 Mask 的精确重放。源目录始终只读。

完整评测保存 full/masked/visible 分区，State/Action 分域 RMSE、MAE、
p95/p99/p99.9/max、全部97个连续 feature，阈值0.01/0.05/0.1/0.2超限统计，
contact BCE/accuracy/混淆、逐 family 微平均/窗口宏平均与目标数。
空域为 null，不参与平均。物理误差仅逐同单位特征汇总。

State/Action 各 top100 含窗口、原始元素身份、绝对/相对帧、chunk/位置、
feature/name/unit、pred/target/signed/absolute error、mean/std和物理值；
另存去重原始元素 top100，全部窗口和分域最差窗口。
报告每 t、chunk、chunk内位置（终点位置4）、重叠窗口同一帧的预测范围；
固定 fixture 0/1 与当前最差窗口分别保存完整连续/接触曲线和 Mask。
速度峰值、±2帧辅助对齐仅诊断；正式误差从不平移。
JSON 曲线同时给 normalized/physical，SVG 使用有符号线性轴。

C 每次评测同时保存 posterior_mean、posterior_sample、standard_normal、zero。
随机路由默认每 fixture 8 个可复现 epsilon，独立评测 RNG，不影响训练 RNG。
分布 quantile/极值/代表曲线明确来自固定 draw0；所有8个样本另外统计期望单样本
State/Action full/masked/visible MSE/RMSE/MAE、方差、masked energy score和补充 best-of-K。
Energy 定义为维度归一化欧氏范数的经验能量分数
`mean ||x-y||/sqrt(D) - 0.5 mean_ij ||x_i-x_j||/sqrt(D)`，包括 i=j；
按有目标 fixture 平均。不能用 best-of-K 代替部署结果，单参考不证明多模态覆盖。

每5次完整评测对固定不同窗口子集独立置零/置换 global、local、condition memory、
FiLM，记录 donor。C 的依赖性消融用 posterior mean 参考；A 的 condition memory消融不适用。
这些只证明依赖，B memory 可绕过 latent，重建成功不能证明 latent 表达充分。

## 4. 记录、审核与恢复

每步流式 JSONL：实际 LR、next LR、分项重建、KL_global/local、beta及加权贡献、
裁剪前后梯度、实际batch、sample/Mask摘要、累计暴露、局部/源/累计step、耗时、Encoder调用数。
默认50步刷新进度/控制台、500步 last、1000步全量评测，均可 CLI 调整。
summary 不复制逐步 records。loss/LR、尾部、masked-visible、family、C多路曲线随评测更新。

step0、首次完整评测、每5次评测及结束生成审核，分已验证事实/候选原因/缺失证据。
路由或非有限值等工程错误停止；质量告警不自动停止或调整训练：
5点评测改善不足1%、连续3点比此前best恶化20%、masked变差而full改善。
数值分母下限1e-12；零初始化FiLM首步上游梯度可为0，按多步观察审核，
全有效chunk时 empty_local 无梯度属正常。

A best 按 full reconstruction；B best 按八family等权、family内有效域均值 masked MSE；
C best 按实际 standard_normal masked energy；另存 best_reconstruction
（C 为 posterior_mean full 重建参考）。每次完整评测才更新 best。

| 模式 | 接口与语义 |
|---|---|
| 同run恢复 | `--resume-run RUN`，合同完全一致；恢复model/optimizer/scheduler/Python/NumPy/Torch/CUDA RNG/采样排列及游标 |
| 新run延长 | `--continue-checkpoint FILE --additional-steps N`，同stage保留AdamW；新LR段；v2保留采样状态；旧格式明确重启采样 |
| 新run初始化 | `--init-checkpoint FILE`，只加载模型；optimizer/scheduler/采样重置；兼容 `--init-run RUN` 只取best，不静默fallback |

旧checkpoint缺少归一化/窗口哈希时，只读重评允许并记录unknown；
训练默认拒绝，若确认来源后显式 `--allow-legacy-identity` 才允许延长/初始化，
仍不称为已验证的精确恢复。所有已知身份必须一致；旧H50拒绝。
v2严格resume仅支持未完成run，同一run有OS锁阻止双进程写入。
改变stage、Mask协议、batch或loss应建新初始化实验，不叫continuation。

SIGINT/SIGTERM转停止请求，完整更新边界持久化；非有限梯度在更新前阻止；
更新中异常绝不覆盖原有效last。强杀/断电只恢复到已持久化边界。
checkpoint写临时文件、flush/fsync、原子replace；先last后best，恢复可修复同step未落盘best。
严格readback加载模型、optimizer、scheduler、采样状态，检查step关系并比较固定前向。
最终报告与读回完成后才写完成标记；interrupted/failed/completed/qualitywarning分离。

## 5. CLI 与 Ubuntu 操作

先同步Windows源码；不改/安装既有依赖。全部产物必须放在
`/home/helloworld/bly/runs/` 下。以下命令显式使用虚拟环境Python，避免系统没有python别名。

必要真实工程smoke（不从旧A初始化；当前旧B可继续运行，资源不足时等待其结束）：

```bash
cd /home/helloworld/bly/state-action-cvae
export PYTHONPATH="$PWD/src"
PYTHON=/home/helloworld/bly/sonic-repro/.venv-sonic/bin/python
bash -n ./cvae_repro.sh
"$PYTHON" -m unittest discover -s tests -p test_cvae_protocol_v2.py -v
"$PYTHON" -m cvae_sa.posterior_hierarchical_standard_cvae \
  --dataset-run /home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506 \
  --output-run "/home/helloworld/bly/runs/cvae_v2_B_smoke_$(date +%Y%m%d_%H%M%S)" \
  --stage B --mask-mode fixed --smoke --max-steps 2 --warmup-steps 1 \
  --micro-batch 2 --validation-interval 1 --checkpoint-interval 1 --log-interval 1
```

A90k只读定位（不启动额外60k）：

```bash
cd /home/helloworld/bly/state-action-cvae
export PYTHONPATH="$PWD/src"
PYTHON=/home/helloworld/bly/sonic-repro/.venv-sonic/bin/python
EVAL_RUN="/home/helloworld/bly/runs/cvae_v2_A_tail_$(date +%Y%m%d_%H%M%S)"
"$PYTHON" -m cvae_sa.cvae_tools evaluate \
  --checkpoint /home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_fixed_20260922_014417/checkpoints/best.pt \
  --dataset-run /home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506 \
  --output-run "$EVAL_RUN" --route A --micro-batch 32
"$PYTHON" -m cvae_sa.cvae_tools export-report --run "$EVAL_RUN"
```

B旧checkpoint重评用同一入口 `--route B` 和独立新目录；它不是旧Mask精确重放。
后续正式B-fixed需审核后手动运行：

```bash
cd /home/helloworld/bly/state-action-cvae
export PYTHONPATH="$PWD/src"
PYTHON=/home/helloworld/bly/sonic-repro/.venv-sonic/bin/python
RUN_DIR="/home/helloworld/bly/runs/cvae_v2_B_fixed_$(date +%Y%m%d_%H%M%S)"
"$PYTHON" -m cvae_sa.posterior_hierarchical_standard_cvae \
  --dataset-run /home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506 \
  --output-run "$RUN_DIR" --stage B --mask-mode fixed \
  --max-steps 60000 --micro-batch 32 --learning-rate 1e-4 \
  --lr-schedule cosine --warmup-steps 2000 --min-lr-ratio 0.01 \
  --validation-interval 1000 --checkpoint-interval 500 --log-interval 50
```

独立终端监控和完成后回传（使用启动时打印的准确run，不猜最新目录）：

```bash
cd /home/helloworld/bly/state-action-cvae
export PYTHONPATH="$PWD/src"
PYTHON=/home/helloworld/bly/sonic-repro/.venv-sonic/bin/python
RUN_DIR=/home/helloworld/bly/runs/替换为准确的run目录
"$PYTHON" -m cvae_sa.cvae_tools monitor --run "$RUN_DIR" --interval 10
"$PYTHON" -m cvae_sa.cvae_tools export-report --run "$RUN_DIR"
```

回传 `diagnostic_report.zip`；包含配置/身份/normalization/趋势/审核/尾部/曲线/hash，
不包含HDF5或checkpoint；不覆盖已存在包，可显式 `--output` 指定新文件。
只读诊断可加 `--export-all` 保存全量预测，常规训练不保存全量。

B-dynamic 在正式B-fixed结果审核后用 `--stage B --mask-mode dynamic --init-checkpoint .../best.pt`
并创建新run；C独立用 `--stage C --kl-beta 1e-3 --beta-warmup-steps 10000 --eval-samples 8`。
A追加60k的接口是 `--stage A --continue-checkpoint .../last.pt --additional-steps 60000`，
学习率段仍用已有CLI设置；历史A需显式确认未知身份。当前不自动授权启动该续训。

Shell保留旧入口；新明确入口是 `...-condition-fixed`、`...-condition-dynamic`、
`...-condition-smoke`。`...-fixed` 仍指Stage A，`...-kl` 指C。
环境覆盖新增 `CVAE_POSTERIOR_INIT_CHECKPOINT`、`CONTINUE_CHECKPOINT`、
`ADDITIONAL_STEPS`、`MASK_MODE`、`EVAL_SAMPLES`、`BETA_WARMUP_STEPS`、
`ALLOW_LEGACY_IDENTITY`（这些短名均带 `CVAE_POSTERIOR_` 前缀）。
Python CLI 不受遗留Shell环境变量影响，优先用于可复现命令。

## 6. 结论边界与维护

A90k的联合max_abs=0.219557不能确定域、feature或帧；不能从RMSE前十表推断精确argmax。
normalization未见近零速度std；80k到90k max仍下降约9.15%，不称为已证明平台。
先回传只读诊断，再决定数据核查、受控续训、时间对齐或chunk消融，不同时改多个变量。

当前维护仅 AGENTS.md、model.md、plan.md；历史run和结果保留。
Windows测试不能替代Ubuntu HDF5/CUDA smoke，亦不能证明训练质量或物理可行性。
