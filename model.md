# 65-token 层级 CVAE：活动模型与实验合同

服务器迁移合同补充：`CVAE_DATASET_PATH_MAP` 为 JSON 绝对路径前缀映射，只在
StateActionWindowDataset、LongHorizonWindowDataset、ReferenceAnchorDataset 打开实际文件时使用。
源 manifest/index/anchor/normalization 文件不重写，`source_run`/`episode_ref` 和稳定采样 identity 不变；
源哈希及 checkpoint normalization 检查保持启用。该适配不改变模型、loss 或部署路线，也不赋予 DDP 能力。

最后更新：2026-10-30。C 已完成训练但标准正态部署质量未通过；实验结果见[plan.md](plan.md)，异常证据见[process.md](process.md)。

模型架构版本保持 `65-token-hierarchical-standard-cvae-v1`；实验协议为
`65-token-experiment-v2`，checkpoint 格式为
`sonic_65_token_hierarchical_standard_cvae_checkpoint_v2`。
本轮不改模型参数、融合、FiLM、数据或重建目标。历史 H50 checkpoint 不迁移。
工程 PASS 与质量判断严格分开。回放报告区分 `engineering_complete`、`baseline_valid`、
`model_physical_quality` 和 `quality_pass`；baseline 有效后，模型 Action 的 joint position、
root position、body MPJPE 相对同窗口原始回放不得超过 1.2 倍，否则标记
`FAIL_RELATIVE_TO_BASELINE`。

## 新增：sonic-10history-reference-cvae-v1

该模型是独立于65-token合同的新路线，不加载旧65-token checkpoint。输入为
`state_history[10,70]`、`action_history[10,29]`、SONIC `reference_future[10,64]`，
输出一个canonical Action[29]，运行时再通过SONIC Action contract转为raw Action。
逻辑序列为 `H[-9..0] + R[+0..9] + Q`；C阶段只mask当前目标Action，训练posterior Action、
standard-normal Action和KL，不再训练future State或64步Action序列。Physics v5是该路线唯一允许的
新数据源；episode 起点用 SONIC reset buffer 的重复观测并显式记录 history validity mask，旧65-token/T64路线保持只读兼容。
数据链路见plan.md §23，训练/DAgger命令见dagger.md末节。

## 1. 数据、编码与解码

数据为 `states[T+1,70]`、`actions[T,29]`，语义为
`State_t + Action_t → State_(t+1)`。70 个 State 分量依次为关节位置29、关节速度29、
基座线速度3、角速度3、重力方向3、高度1、接触2；Action 为 canonical 关节目标29。
旧65-token模型不使用 RobotInfo、reference、motion ID 等元数据；身份字段只供采样和诊断。

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

## 2. 实验阶段与目标

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
held-out Mask 不等于 held-out motion。held-out bank 中可能有与训练bank坐标相同的fixture；
评测显式记录重合数，并将新坐标与相同坐标分组统计。

## 3. 身份、评测与尾部诊断

稳定窗口身份含 source/episode、motion、variant、start、有效 State/Action 长度，
禁止 batch 内序号回退。运行保存解析配置、CLI、commit、相关源码哈希、
dataset manifest/index/normalization/window-table 哈希及源 checkpoint 路径/step/hash。
固定/held-out bank 保存实际隐藏坐标与哈希；训练样本、Mask 摘要与暴露量逐步记录。

只读评测入口 `python -m cvae_sa.cvae_tools evaluate` 必须指定 checkpoint、dataset、
route、新 output run。历史65-token可读但未知身份显式标注；源目录始终只读。

完整评测保存 full/masked/visible 分区，State/Action 分域 RMSE、MAE、
p95/p99/p99.9/max、全部97个连续 feature，阈值0.01/0.05/0.1/0.2超限统计，
contact BCE/accuracy/混淆、逐 family 微平均/窗口宏平均与目标数。
空域为 null，不参与平均。物理误差仅逐同单位特征汇总。
State/Action 各 top100（含窗口、原始元素身份、帧、chunk/位置、feature/unit、误差和物理值）另存去重版本；
报告每 t、chunk、chunk内位置、重叠窗口同一帧的预测范围；固定 fixture 0/1 与当前最差窗口保存完整曲线和 Mask。
速度峰值、±2帧辅助对齐仅诊断；正式误差从不平移。

C 每次评测同时保存 posterior_mean、posterior_sample、standard_normal、zero。
随机路由默认每 fixture 8 个可复现 epsilon，独立评测 RNG。8个样本统计期望单样本
full/masked/visible MSE/RMSE/MAE、方差、masked energy score和补充 best-of-K。
Energy 定义为 `mean ||x-y||/sqrt(D) - 0.5 mean_ij ||x_i-x_j||/sqrt(D)`（包括 i=j），按有目标 fixture 平均。
不能用 best-of-K 代替部署结果，单参考不证明多模态覆盖。

每5次完整评测对固定不同窗口子集独立置零/置换 global、local、condition memory、FiLM，记录 donor。
C 的依赖性消融用 posterior mean 参考。这些只证明依赖，重建成功不能证明 latent 表达充分。

## 4. 记录、审核与恢复

每步流式 JSONL：实际 LR、next LR、分项重建、KL_global/local、beta及加权贡献、
裁剪前后梯度、实际batch、sample/Mask摘要、累计暴露、局部/源/累计step、耗时、Encoder调用数。
默认50步刷新进度/控制台、500步 last、1000步全量评测，均可 CLI 调整。

step0、首次完整评测、每5次评测及结束生成审核，分已验证事实/候选原因/缺失证据。
路由或非有限值等工程错误停止；质量告警不自动停止或调整训练：
5点评测改善不足1%、连续3点比此前best恶化20%、masked变差而full改善。
零初始化FiLM首步上游梯度可为0；全有效chunk时 empty_local 无梯度属正常。

A best 按 full reconstruction；B best 按八family等权、family内有效域均值 masked MSE；
C best 按实际 standard_normal masked energy；另存 best_reconstruction
（C 为 posterior_mean full 重建参考）。每次完整评测才更新 best。
B 同时写出 `best_fixed.pt`（与 `best.pt` 同一固定bank选择）和 `best_heldout.pt`（held-out mask 选择），
并记录两者的 step、指标和严格读回；旧 run 缺少该别名时不得宣称已补齐历史选择。
C checkpoint 额外保存固定 epsilon 的 standard-normal readback probe，严格读回同时验证 posterior-mean 与部署路径。

| 模式 | 接口与语义 |
|---|---|
| 同run恢复 | `--resume-run RUN`，合同完全一致；恢复model/optimizer/scheduler/Python/NumPy/Torch/CUDA RNG/采样排列及游标 |
| 新run延长 | `--continue-checkpoint FILE --additional-steps N`，同stage保留AdamW；新LR段；v2保留采样状态；旧格式明确重启采样 |
| 新run初始化 | `--init-checkpoint FILE`，只加载模型；optimizer/scheduler/采样重置；兼容 `--init-run RUN` 只取best，不静默fallback |

旧checkpoint缺少归一化/窗口哈希时，只读重评允许并记录unknown；训练默认拒绝，确认来源后显式
`--allow-legacy-identity` 才允许延长/初始化，仍不称为精确恢复。旧H50拒绝。
v2严格resume仅支持未完成run，同一run有OS锁阻止双进程写入。改变stage、Mask协议、batch或loss应建新初始化实验。

SIGINT/SIGTERM转停止请求，完整更新边界持久化；非有限梯度在更新前阻止；更新中异常绝不覆盖原有效last。
checkpoint写临时文件、flush/fsync、原子replace；先last后best。严格readback加载模型、optimizer、scheduler、
采样状态并比较固定前向。最终报告与读回完成后才写完成标记；interrupted/failed/completed/qualitywarning分离。

## 5. CLI 与 Ubuntu 操作（A/B/C 历史命令）

全部产物必须放在 `/home/helloworld/bly/runs/` 下；显式使用虚拟环境Python。以下为当时的smoke、A只读定位、
B-fixed训练与监控/回传命令，保留作复现参考：

```bash
cd /home/helloworld/bly/state-action-cvae
export PYTHONPATH="$PWD/src"
PYTHON=/home/helloworld/bly/sonic-repro/.venv-sonic/bin/python
# smoke
"$PYTHON" -m cvae_sa.posterior_hierarchical_standard_cvae \
  --dataset-run /home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506 \
  --output-run "/home/helloworld/bly/runs/cvae_v2_B_smoke_$(date +%Y%m%d_%H%M%S)" \
  --stage B --mask-mode fixed --smoke --max-steps 2 --warmup-steps 1 \
  --micro-batch 2 --validation-interval 1 --checkpoint-interval 1 --log-interval 1
# A90k只读定位
EVAL_RUN="/home/helloworld/bly/runs/cvae_v2_A_tail_$(date +%Y%m%d_%H%M%S)"
"$PYTHON" -m cvae_sa.cvae_tools evaluate \
  --checkpoint /home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_fixed_20260922_014417/checkpoints/best.pt \
  --dataset-run /home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506 \
  --output-run "$EVAL_RUN" --route A --micro-batch 32
"$PYTHON" -m cvae_sa.cvae_tools export-report --run "$EVAL_RUN"
# B-fixed
RUN_DIR="/home/helloworld/bly/runs/cvae_v2_B_fixed_$(date +%Y%m%d_%H%M%S)"
"$PYTHON" -m cvae_sa.posterior_hierarchical_standard_cvae \
  --dataset-run /home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506 \
  --output-run "$RUN_DIR" --stage B --mask-mode fixed \
  --max-steps 60000 --micro-batch 32 --learning-rate 1e-4 \
  --lr-schedule cosine --warmup-steps 2000 --min-lr-ratio 0.01 \
  --validation-interval 1000 --checkpoint-interval 500 --log-interval 50
# 监控与回传（使用启动时打印的准确run）
RUN_DIR=/home/helloworld/bly/runs/替换为准确的run目录
"$PYTHON" -m cvae_sa.cvae_tools monitor --run "$RUN_DIR" --interval 10
"$PYTHON" -m cvae_sa.cvae_tools export-report --run "$RUN_DIR"
```

- 旧B重评用 `--route posterior_mean`（原信息路径）、`--route zero`（消融）、`--route A`（复核旧错误路径）；
  新 `--route B` 不是旧训练路径复现。
- `diagnostic_report.zip` 不含HDF5或checkpoint，不覆盖已存在包（可 `--output` 指定）；`--export-all` 仅用于只读诊断。
- B-dynamic：`--stage B --mask-mode dynamic --init-checkpoint .../best.pt`；C：`--stage C --mask-mode dynamic --kl-beta 1e-3
  --beta-warmup-steps 10000 --eval-samples 8`，不加载A/B权重。A追加：`--stage A --continue-checkpoint .../last.pt
  --additional-steps 60000`（不自动授权）。
- Shell入口：`...-condition-fixed`、`...-condition-dynamic`、`...-condition-smoke`；`...-fixed` 指Stage A，`...-kl` 指C。
  环境覆盖 `CVAE_POSTERIOR_INIT_CHECKPOINT`、`CONTINUE_CHECKPOINT`、`ADDITIONAL_STEPS`、`MASK_MODE`、`EVAL_SAMPLES`、
  `BETA_WARMUP_STEPS`、`ALLOW_LEGACY_IDENTITY`（短名均带 `CVAE_POSTERIOR_` 前缀）。Python CLI不受遗留Shell环境变量影响。

## 6. 结论边界与已封存 C run

A尾部定位与边界见process.md §2～6。维护分工：AGENTS.md（入口）、model.md（合同）、plan.md（计划与概要）、
process.md（异常诊断）、dagger.md（DAgger）。Windows测试不能替代Ubuntu HDF5/CUDA smoke。

- 原C（kl_beta=1e-3，已封存）：`/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_kl_20260925_211113`；
  四路线结果见plan.md §11。只允许从step360000做只读尾部诊断，不普通续训，不同时改loss、latent、pooling或宽度。
- KL3参考（已封存）：
  `/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_kl3_eager_200k_eval50k_noeval0_zlqw641k`，
  `skip_initial_eval=true`，compile/cache/pin-memory/non-blocking全关闭，checkpoint SHA256
  `bc080d48a07e5861a2b8c2e0171cce980d35ee58eef7e921282c7597e3408a63`；结果见plan.md §20。后续实验必须新建run。

## 7. 65-token State／Action回放合同（2026-09-23）

独立入口 `python -m cvae_sa.replay65 {prepare,simulate,render,report}`，协议 `65-token-replay-v1`（v2扩展见plan.md §19）。
不更改训练模型或loss，不复用旧H50入口。A接受历史65-token Stage-A checkpoint（缺失身份标unknown）；
B/C只接受对应阶段的v2 checkpoint，拒绝旧posterior参与的B。A完整输入posterior mean；B condition-only；
C固定epsilon标准正态部署。所有Mask均True=隐藏，B/C调用模型前再次置零隐藏真值。输出State65帧、Action64步。

### 7.1 准备、选择和原始基线

- prepare必须显式checkpoint/dataset/output和 `--window-index` 或 `--selection first-suite`（episode起点、首个非零起点、
  八家族masked MSE最差窗口，去重最多三个）。默认视频/物理代表为state_gap_16、action_gap_16、full_action、joint_gap_8。
- 各窗口独立子目录 `windows/wNNNNNN/`；再次prepare拒绝非空目录，simulate/render/report重验准备哈希。
- 每窗口原始Action在两个独立num_envs=1进程执行。B/C只替换隐藏Action；`--action-mode full-prediction` 是另一实验。
  反归一化后用记录nominal/scale/offset/clip转回raw，保留未裁剪预测与饱和计数。
- `--state-only-continuation`：manifest中 `single_window_continuation=true` 与 `input_contract=state_only` 为验收标记（详见process.md §20）。
- exact-init使用外层patch0009钩子：恢复采集后的物理参数，不重新抽取随机化；新interval事件替换为空操作并保存前后事件列表；
  源采集有未记录日程的interval事件时拒绝。非零延迟的历史draw和队列未存储时不伪造恢复；接触缓存未恢复，不承诺位级等价。
  回放中出现自动reset时在reset前停止，不拼接。
- 原始源复现与重复性阈值：joint RMSE≤0.02rad、root RMSE≤0.05m、orientation max≤5°、body MPJPE≤0.05m、
  contact accuracy≥95%。基线失败仍继续模型对照，但标 `BASELINE_INVALID / MODEL_QUALITY_UNDETERMINED`。
  缺失字段、映射错误、NaN/Inf属于工程失败。

### 7.2 State视频与报告

State三栏为HDF姿态、真实State积分重建、模型补全State积分重建，共享第0帧root锚点，标KINEMATIC ONLY。
Action视频为HDF、原Action、模型Action三栏；另有HDF/原Action1/原Action2基线三栏；65帧50Hz，固定HDF相机。
报告含分域误差、contact混淆/BCE、top100坐标、NPZ、逐帧误差及首次越界帧、初始化读回、runtime配置和seed；
物理误差按单位组统计。`replay65_execution.ok` 仅表示仿真和视频完整，不代表模型通过。

### 7.3 Ubuntu命令（不修改源码或安装依赖；先确认SONIC已应用patch0009）

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
python3 -m cvae_sa.replay65 --help
rg -n 'apply_exact_replay_initialization' ../sonic-repro/GR00T-WholeBodyControl/gear_sonic/eval_agent_trl.py
# B 窗口0原始双基线
read -r -p '请输入已完成的v2 B训练run绝对路径: ' B_RUN
test -s "$B_RUN/checkpoints/best.pt" || { echo 'checkpoint不存在'; exit 1; }
RUN_DIR=$(mktemp -d /home/helloworld/bly/runs/cvae_replay65_B_w0_XXXXXXXX)
python3 -m cvae_sa.replay65 prepare \
  --checkpoint "$B_RUN/checkpoints/best.pt" \
  --dataset-run /home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506 \
  --route B --window-index 0 --output-run "$RUN_DIR" && \
python3 -m cvae_sa.replay65 simulate --run "$RUN_DIR" --baseline-only && \
python3 -m cvae_sa.replay65 report --run "$RUN_DIR"
# C 窗口0
C_RUN=/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_kl_20260925_211113
RUN_DIR=$(mktemp -d /home/helloworld/bly/runs/cvae_replay65_C_w0_XXXXXXXX)
python3 -m cvae_sa.replay65 prepare \
  --checkpoint "$C_RUN/checkpoints/best.pt" \
  --dataset-run /home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506 \
  --route C --window-index 0 --output-run "$RUN_DIR" \
  --sample-seed 20260923 --sample-index 0 --simulation-seed 20260923
python3 -m cvae_sa.replay65 simulate --run "$RUN_DIR" --baseline-only
python3 -m cvae_sa.replay65 report --run "$RUN_DIR"
# 基线检查后：模型回放、三栏视频与导出
python3 -m cvae_sa.replay65 simulate --run "$RUN_DIR" && \
python3 -m cvae_sa.replay65 render --run "$RUN_DIR" \
  --model /home/helloworld/bly/sonic-repro/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml && \
python3 -m cvae_sa.replay65 report --run "$RUN_DIR" --export
# 另一终端监控
python3 -m cvae_sa.cvae_tools monitor --run "$RUN_DIR" --interval 10
```

- 工程错误时停止并建立新run，不手工补marker、不复用失败目录（含因 `markers/` 缺失而失败的旧C回放目录）。
- 回传 `$RUN_DIR/replay65_report.zip` 与 `$RUN_DIR/windows/w000000/videos/*.mp4`；ZIP不含HDF/checkpoint/MP4/pickle，
  已存在不覆盖，最终包用 `report --run "$RUN_DIR" --export --output "$RUN_DIR/replay65_report_final.zip"`。
- 首轮审核后再在新run用 `--selection first-suite`。A参考用 `--route A --window-index 0` 和
  `/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_fixed_20260923_012104/checkpoints/best.pt`；
  B-fixed用 `best_fixed.pt`，B-dynamic用 `best_heldout.pt`（dynamic的 `best.pt` 对应step0源权重）；C用
  `/home/helloworld/bly/runs/cvae_posterior_hierarchical_standard_cvae_65_kl_20260925_211113/checkpoints/best.pt`
  和 `--route C --sample-seed ... --sample-index ...`。

## 8. v2 运行时优化合同（2026-10-04）

逐特征标准化和canonical Action目标不变：`A_norm=(A-canonical_mean)/canonical_std`，无Action residual分支。
旧C run实际是FP32 eager；新设置只用于新run。BF16只覆盖CUDA密集前向；参数、AdamW状态、normalization、mask判断、
KL的exp/logvar/reduction、重建MSE、contact BCE、latent统计与评测指标保持FP32；不用GradScaler，不支持BF16直接失败。
runtime contract记录precision、autocast、dtype、TF32、compile、cache、pin-memory、non-blocking；精确resume要求一致，
改变运行时选项必须新建controlled initialization run。`data.cache_mode=auto` 写入当前run的 `data/normalized_windows.pt`，
identity绑定dataset manifest、episode index、normalization和selected-window hash；超出上限回到同步HDF5路径。
RecoverableSampler的permutation、cursor、epoch、exposure和RNG合同不变。

## 9. C latent 分组诊断（2026-10-04）

C的KL定义为 `0.5 * (global_kl + local_kl)`（global 256维均值、local `[16,128]` 均值）。`cvae_sa.latent_distribution65`
只调用Posterior Encoder，按 `global` 和十六个 `local` 分别保存逐维KL、posterior mean、sigma及固定seed sample；
posterior reference是统计估计，不是物理真值。`cvae_sa.latent_normality65` 对 `latent_arrays.npz` 做只读后处理，
`near_standard_normal` 只是启发式标签，不替代四条C路径的质量评测。输出文件见plan.md §16。

## 10. Compiled runtime 修复（2026-10-11）

默认compile mode为 `max-autotune-no-cudagraphs`；autocast context在compiled core外部，评测输出对physical State、Action和
contact logits执行detach/clone。运行时异常或Dynamo recompile limit只允许一次显式eager fallback，在 `compile_info`、
failure/progress和checkpoint runtime contract中记录requested/succeeded、fallback reason/phase、graph breaks、recompiles、
cold start与output ownership contract。旧 `bf16_compiled_uncached` 失败目录不能resume。

## 11. T64 连续动作分段与 192 步推理

派生合同 `sonic_long_horizon_state_action_dataset_v1`：episode按192 transition非重叠分组，每组最多三个连续T64窗口；
尾部group保留valid padding，所有有效Action恰好覆盖一次；manifest只保存group/window/source hash，normalization沿用源split。
模型仍只接收单个T64窗口。长时standard-normal路径：

- `full_action_t64`：每窗口隐藏全部Action。
- `tail32_context_next32`：前32步完整State/Action加后32步State，隐藏后32步Action；首块需32步warm-up，属结构化held-out/OOD诊断，
  不能当作递归闭环B或物理外推质量证据。
- `recursive_state_feedback_b`：先生成临时64步Action只执行前32步，再用真实执行State、上一段Action和记录未来State预测下一段32步；
  每个segment记录raw Action、exact initialization、执行State、mask identity和baseline replay。
- `cvae_sa.long_horizon_ab_compare`：同一192-step窗口内比较A与递归B，以recorded source和Origin Action replay为参考；
  KL2与KL4各自成组。

C评估提速：`micro_batch=32` 与 `eval_micro_batch=128` 分离，summary evaluation不参与 `best.pt` 选择（细节见plan.md §21）。

## 12. G1+ unlocked31 资产、模型与迁移状态（2026-10-20）

源bundle、final fixed6路径、sha256与三动作ghost目录见plan.md §22。run-local准备和转换产物位于：

```text
/home/helloworld/bly/runs/g1plus_unlocked31_diag_20261019_160509_3309439/unlocked31_run/
```

最终使用 `assets/g1_pl_unlocked31_mujoco_fixed6.xml`（sha256=abd1a3dcccdeb43e163ef0d6f21d9eec2e6d426c97fb255d4c3dfe452aa34455）。
转换器 `urdf2mjcf==0.0.4` 的两个缺陷已在run-local副本中修正：

1. 删除 `add_visual_geom_logic()` 产生的缺失 `size`/空 `type` 视觉复制geom；
2. 为converter synthetic `root` 保留freejoint，并写入原始pelvis的 `fullinertia`，而不是零质量root或虚构小质量。

fixed6是31-DoF G1+ runtime model；29-DoF C checkpoint仍只负责body joints，neck是额外渲染变量，不声称checkpoint已扩展为31-D policy。
每个ghost目录的 `videos/ghost_source_static_moving.mp4` 为193帧、50 Hz；manifest记录source、C recursive replay、neck static/moving
三层、模型hash、runtime XML hash和 `kinematic_only=true`。source路径是 `windows/w000000/data/source.trajectory.pkl`；C route路径是
`windows/w000000/simulations/recursive_state_feedback_b/data/replay/000000.trajectory.pkl`。
`simulate_g1_plus_action_compare.py` 的static-zero body control会让机器人在重力下蹲下，只说明无身体控制会失稳；
C canonical Action是关节目标，不能直接当作fixed6 MJCF的torque ctrl。

## 13. SONIC goal/state fixed-point 诊断边界（2026-11-07）

该诊断不改变65-token CVAE输入合同，也不把 SONIC goal、motion ID或reference输入加入序列模型。
SONIC actor实际使用的G1 goal tokenizer为10帧 `[joint_pos(29), joint_vel(29), relative_orientation_6d(6)]`；
Physics State为70维。实验只将State的joint position/velocity映射到goal，relative orientation设为identity
6D，并保留base velocity、gravity、height、contact为State侧不可映射字段。

32-motion、10轮、每轮32步的结果没有任何motion达到goal/update双门限；结果只作SONIC输入语义差异诊断，
不能解释为CVAE质量、DAgger质量或物理控制通过。固定点run、门限和失败证据见plan.md §26、
process.md §25。

## 14. 11-frame MLP Action completion（2026-11-08，独立试验）

该路线是独立于65-token Transformer/CVAE的新序列补全实验，不迁移旧Transformer权重，也不改变
65-token数据、checkpoint或部署合同。代码入口为：

```text
state-action-cvae/src/cvae_sa/mlp_action11.py
state-action-cvae/configs/mlp_action11.json
dagger/src/dagger/mlp_action11_recorded_state_replay.py
dagger/scripts/render_mlp_action11_ghost4.sh
```

### 14.1 数据与时间合同

- 数据为Physics v5的70维State与29维canonical Action；不读取SONIC goal，不把goal/state差异加入模型。
- 训练只使用train split中确定性选择的32个motion；不建立test split。v5迁移目录中的`status=failed`
  且`steps<10`的4个episode被读取层过滤，32个motion仍全部保留，其余252个completed episode可用。
- 每个样本是连续11帧State与10帧Action：

  ```text
  State :  S0 ... S10       [11,70]
  Action:  A0 ... A9        [10,29]
  ```

  `A10`不在合同中，因为它需要`S11`。主任务是已知`S0..S10`、已知`A0..A8`时补全`A9`。
- episode起点的`action_before_window`为零，并由`action_before_valid=0`显式标记；非起点使用真实前一Action
  且`action_before_valid=1`。这不是NaN或伪造Action。

### 14.2 MLP架构与完整维度

输入先按逐特征Mask置零，但Mask本身单独输入，避免零值与缺失混淆：

|输入|形状|含义|
|---|---:|---|
|State value|`[B,11,70]`|归一化Physics State|
|State visible|`[B,11,70]`|`1`为可见，`0`为隐藏|
|State valid|`[B,11]`|序列槽是否存在；渲染未知前缀为0|
|Action value|`[B,10,29]`|归一化canonical Action|
|Action visible|`[B,10,29]`|逐特征可见性|
|Action valid|`[B,10]`|Action槽是否有可用历史|
|target Action mask|`[B,10,29]`|只对这些缺口计算loss|
|Action before|`[B,29]`|窗口前一Action或零|
|Action before valid|`[B]`|窗口前一Action有效性|

Encoder flatten后的输入维度为：

```text
11*70       State value       =  770
11*70       State visible     =  770
10*29       Action value      =  290
10*29       Action visible    =  290
11          State valid       =   11
10          Action valid      =   10
10*29       target mask       =  290
29          Action before     =   29
1           before valid      =    1
                              -----
                               2461
```

实际配置：

```text
Encoder: 2461 -> 2048 -> 1024 -> 512 -> latent[256]
Decoder query:
  latent[256]
  + target_action[290]
  + action_visible[290]
  + state_visible[770]
  + state_valid[11]
  + action_valid[10]
  = 1627
Decoder: 1627 -> 1024 -> 2048 -> 512 -> 290
Output:  [B,10,29]
```

每个隐藏层使用`Linear -> LayerNorm -> GELU`；latent是确定性向量，不采样、不加KL、不提供
standard-normal部署路径。该配置实际参数量为`12,773,922`。模型kind为`mlp_action_completion_11`，
checkpoint只保存`best.pt`与`last.pt`。

### 14.3 训练前向与Mask采样

训练前向步骤：

1. 读取归一化`State[11,70]`、`Action[10,29]`。
2. 对隐藏的State/Action逐元素置零；把visible、valid、target mask和窗口前Action拼入Encoder。
3. Encoder输出单个`z[256]`。
4. Decoder接收`z`和查询Mask，输出所有`A0..A9`的29维预测。
5. 只在`target_action_mask=True`的Action元素上计算归一化MSE；已知Action不作为目标误差。

Mask比例固定为70%完整时间步/State-Action级任务、30%特征级任务：

- 时间步级任务：目标Action通常为连续1–3个Action槽；55%的样本优先选择末槽`A9`，其余在
  `A0..A9`中采样；目标之后的未来Action槽标记`valid=0`，不当作零Action。
- 时间步级任务的30%会再隐藏不触及目标转移端点的完整State间隔，保证目标Action所需的
  `S_t,S_{t+1}`仍可见。
- 特征级任务：目标Action随机隐藏4–15个特征；另外在非端点State上隐藏4–19个State特征，
  并可隐藏一个历史Action槽的少量特征。
- 所有目标至少包含一个Action特征；State缺口不会覆盖目标转移的两个端点，因此不采样
  “缺少必要边界却要求补全不可辨识Action”的组合。
- 训练只使用普通11/10窗口Mask；渲染时的递推输入不是训练样本伪造，而是独立的render-only
  right-aligned query，属于分布外闭环诊断并在manifest中单独标记。

### 14.4 逐Action递推渲染

渲染目标永远放在相对窗口的Action slot 9。对绝对时间`t`：

```text
State slots : S[t-9] ... S[t]  S[t+1(expected)]
Action slots: A[t-9] ... A[t-1] A[t(target)]
```

实际可用范围不足时，左侧槽填零并设`valid=0`。每一步：

1. 用当前Isaac真实State填充当前槽；用记录回放的期望`S[t+1]`填充最后一个State槽。
2. 用此前已经执行的canonical Action填充历史Action槽；未知历史使用`valid=0`。
3. MLP只调用一次，取输出的最后一行`A9`。
4. canonical Action经SONIC runtime mapping转为raw Action并执行。
5. 读取Isaac真实State，加入历史，进入`t+1`。

首步使用真实初始化后的`S0`与期望`S1`，更早的State槽是unknown；第十步开始窗口才逐渐
包含完整历史。192步输出193帧，允许摔倒但不因摔倒截断视频。

四动作对比入口：

```bash
DAGGER_MOTION_FILE=/path/to/robot_filtered \
bash dagger/scripts/render_mlp_action11_ghost4.sh \
  /data1/users/bailinying/FlexSeqWBC/bly/runs/<mlp_run>/checkpoints/best.pt \
  /data1/users/bailinying/FlexSeqWBC/bly/runs/<stepwise_prepare>/manifests/cases.json \
  /path/to/stepwise_recursive_causal.yaml \
  192
```

该入口为每个motion独立运行MLP Isaac递推；SONIC使用同motion的release teacher，随后由
`render_policy_ghost.py`输出MLP/SONIC两层193帧ghost。Isaac执行、轨迹、ghost和manifest都写入
新run；视频完整性与物理质量分开。

## 15. 11-frame MLP conditional CVAE（2026-11-08，训练与ghost完成）

该实验仍独立于65-token层级CVAE。它回归最小序列生成合同：`S0..S10`共11帧State、
`A0..A9`共10帧Action；不接收SONIC goal，不修正goal与Physics State的差异，也不使用
motion ID、reference或dynamics context。`A10`不在输入或输出中。训练只使用Physics v5
train split中确定性选择的32个motion，不建立test split。

代码入口：

```text
state-action-cvae/src/cvae_sa/mlp_cvae11.py
state-action-cvae/configs/mlp_cvae11.json
dagger/src/dagger/mlp_cvae11_recorded_state_replay.py
dagger/scripts/render_mlp_cvae11_ghost4.sh
```

### 15.1 输入与显式辅助编码

每个训练样本的原始张量为：

```text
State       [B,11,70]
Action      [B,10,29]
Action_before_window       [B,29]
Action_before_window_valid [B]
```

State/Action均使用行级`visible`、`valid`和`target`编码；不使用feature-level mask。MLP
condition向量的完整宽度为：

```text
11*70 + 11              # State value + State visible
+ 10*29 + 10            # Action value + Action visible
+ 11 + 10               # State/Action valid
+ 11 + 10               # State/Action target
+ 29 + 1                # action_before + valid
= 1153
```

### 15.2 q/p、latent与decoder

q posterior和p prior各自使用独立的：

```text
1153 -> 2816 -> 1792 -> 896 -> 512
```

每条路径输出`mean[384]`和`logvar[384]`。训练时：

```text
zq ~ q(z | complete sequence, mask)
zp ~ p(z | visible sequence, mask)
```

q接收完整target值，但仍接收visibility/valid/target标识；p只接收visibility置零后的
可见条件。q/p不把完整truth传给decoder。两条路径共享一个decoder：

```text
condition[1153] + z[384] = 1537
1537 -> 2048 -> 1536 -> 768 -> 1060
1060 = State[11,70] + Action[10,29]
```

实际参数量为`30,365,732`。每层为`Linear -> LayerNorm -> GELU`，latent为全局Gaussian。

### 15.3 mask与训练前向

任务族按样本采样：

```text
causal      40%
State-only  20%
Action-only 20%
Both        20%
```

causal任务随机选择`t∈[0,9]`，将目标`A_t`右对齐到Action slot 9，State slot 9为当前
`S_t`，State slot 10为`S[t+1]`。已知历史Action右对齐到slot 0..8；缺失左前缀为
`value=0, valid=0`。目标slot满足`valid=1, visible=0, target=1`。episode起点的
`action_before_window`由`valid=0`显式表示。

State-only、Action-only和Both只隐藏长度1–2的时间块；没有feature-level mask。训练不做
递推，使用teacher-forced窗口。masked State/Action只在对应target上计入loss：

```text
0.25*(q_State + q_Action + p_State + p_Action)
+ kl_beta * KL(q || p)
```

`kl_beta`从0线性升至`0.002`，前10000 step完成warmup。部署只允许p路径。

### 15.4 prior递推与ghost

`mlp_cvae11_recorded_state_replay.py`只用于render：

1. 以Isaac真实当前State、记录的期望下一State和已执行Action构造右对齐11/10 query；
2. 调用p encoder，采样`zp`；
3. decoder输出完整State/Action，取Action slot 9；
4. canonical Action转runtime/raw Action并执行一个Isaac step；
5. 读取真实State，重复192次。

该过程输出192个Action和193帧，再与SONIC release teacher渲染成两层193帧ghost。物理
摔倒不截断视频；视频完整性和物理质量分开记录。正式训练`2611-0439`完成300000 step，
`best.pt`选择step270000。最终严格prior-only replay使用显式`prior_forward()`，不计算
posterior q，服务器ghost run为：

```text
/data1/users/bailinying/FlexSeqWBC/bly/runs/mlp_cvae11_ghost4_20261109_133453/
```

四条视频已回传到：

```text
/home/vamper/Desktop/replay/mlp_cvae11_ghost4_20261109_133453/
```

同一`best.pt`、同一32-batch train-only monitor和同一mask下，prior-only配对结果为：

```text
sampled zp:       State 0.193812, Action 0.078920, causal Action 0.054099
deterministic μp: State 0.193561, Action 0.078413, causal Action 0.053153
```

采样/确定性RMSE ratio为`1.0013/1.0065/1.0178`。这证明当前离线误差差异主要来自
latent sampling，而不是posterior误用；它仍不等价于闭环物理质量通过。

## 16. 10-history reference-conditioned MLP-CVAE（2026-11-09，合同与实现）

该路线严格保留旧 `sonic-10history-reference-cvae-v1` 的单步合同，只把 Transformer
posterior/condition/decoder 替换为纯 MLP；不改变 SONIC goal、Physics State、reference
或 Action 的语义。训练和 DAgger 使用 8 个 C 训练过的 motion；ghost 只选 4 个代表 motion。

固定输入/输出：

```text
state_history    [B,10,70]
action_history   [B,10,29]
reference_future [B,10,64]
target_action    [B,29]
output action    [B,29]
```

reference 的 10×64 输入来自 Physics v5 的 reference future。State/Action history 的
validity 会显式编码；无效值先置零。condition 向量宽度为 `1738`：

```text
10*70 + 10*29 + 10*64       values
+ 10 + 10                   state/action valid
+ 10*2                      paired history validity
+ 10                        reference time offsets
+ 29 + 29                   target marker + zero target placeholder
= 1738
```

q posterior 额外接收真实 target Action，因此 posterior 输入为 `1767`；p condition
encoder 不接收 target。q/p 分别为：

```text
q: [1767,2560,1792,896,512] -> mean/logvar[256]
p: [1738,2560,1792,896,512] -> condition code[512]
decoder: [1738+256,2048,1536,768] -> Action[29]
```

Linear 层后为 LayerNorm+GELU；全局 latent 为 256 维 Gaussian。参数量为 `31,543,581`。
C 阶段独立计算 posterior-sample Action loss、standard-normal Action loss 和
`KL(q||p)`；部署及 DAgger 只走 `p(condition)` 加 standard-normal latent，不调用 q。
训练配置为 `kl_beta=0.002`，前 20k step warmup，配置 step 上限 245k；若达到 step
上限而未达到 8h，必须从 `last.pt` 连同 AdamW 状态续训至累计 8h。

DAgger 保持旧 10-history/reference 合同。每轮先以当前 Isaac State、Action history
和 SONIC reference 查询 student，再以 teacher Action 作标签；student-only 评估为
192-step same-motion teacher imitation。Isaac 长进程按 2 round 分片，保存
`round_XX.pt`、samples 和 partial metrics 后重启；分片恢复加载模型和 AdamW 状态。
该分片机制只改变进程生命周期，不改变训练数据、loss 或模型合同。

代码入口：

```text
state-action-cvae/src/cvae_sa/sonic_reference_mlp_model.py
state-action-cvae/src/cvae_sa/sonic_reference_mlp_train.py
state-action-cvae/configs/sonic_reference_mlp_cvae_c_8h.json
dagger/configs/sonic_causal_reference_mlp_4h.yaml
dagger/scripts/render_reference_mlp_threeway_ghost4.sh
```

服务器执行结果与ghost交付已在§17回填；工程marker不等价于质量通过。

## 17. 10-history reference MLP-CVAE：C、DAgger与训练motion ghost最终结果（2026-11-10）

本路线已完成工程训练与视频交付。C阶段run为：

```text
/data1/users/bailinying/FlexSeqWBC/bly/runs/sonic_reference_mlp_cvae_c_8h_20261109/
```

首段245k step达到配置step上限后，使用`last.pt`和AdamW状态继续；最终checkpoint为
`step=1325000`，best fixed-standard-normal metric为`0.0033501992`。续训没有重新初始化，
也没有改变数据、归一化或模型合同。

DAgger使用同一C训练过的8个motion和10-history/reference合同。最终run为：

```text
/data1/users/bailinying/FlexSeqWBC/bly/runs/dagger_sonic_causal_reference_mlp_4h_20261110_003455/
```

最终保留41个round checkpoint，最新已记录round 40的student-only teacher Action RMSE为
`0.1204419`。该指标仍是same-motion teacher imitation，不等价于新motion泛化或物理质量通过。

最终训练motion ghost均要求motion来自对应checkpoint的`train_motion_keys`，每组4个motion、
3层、193帧、50 Hz：

```text
A MLP-CVAE11：
/home/vamper/Desktop/replay/mlp_cvae11_trained_motion_ghost4_20261110_045551/

B 初始DAgger：
/home/vamper/Desktop/replay/reference_mlp_trained_motion_ghost4_20261110_043913/

B 二次续训后：
/home/vamper/Desktop/replay/reference_mlp_trained_motion_ghost4_20261110_073719/
```

A第四motion原Isaac rollout在140帧终止；ghost层使用末姿态补齐到193帧并在manifest中明确保留，
不是把摔倒或终止误报为模型执行失败。视频完整性与控制质量独立判断。
