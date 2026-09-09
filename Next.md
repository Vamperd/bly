# 活动合同：32-Motion、T64 层级 Latent Transformer 容量实验

最后更新：2026-09-09

状态：H38-A/B均质量FAIL；H50-A已随机初始化跑满30k且近门槛FAIL，best score `1.03707`。其global State与worst State只超阈值1.01%/3.71%，28k–30k仍单调改善，因此当前只允许从H50-A `last.pt`受控续训最多15k。H38-R仍因B没有fit marker而阻塞。历史run、源HDF5和checkpoint保持只读。事实结果仍以
[plan.md](plan.md)为唯一台账，安全规则见[AGENTS.md](AGENTS.md)。

## 1. 问题与顺序

本轮目标是在32 motion、256 episode的overfit subset上，把窗口降至64 transitions，先证明目标函数
和评测门禁可达，再检验层级posterior结构。固定顺序为：

```text
H38-A full-both autoencoding（已FAIL，仅作诊断）
→ H38-B fixed physical Masks（已FAIL）
→ H50-A full-both autoencoding（30k近门槛FAIL）
→ H50-A low-LR continuation（最多15k，当前唯一下一步）
→ 仅续训PASS时继续H50-B/H50-R
→ 冻结KL=0基线后再实现KL三路径
```

H50-A已表明加宽显著改善global State/Action与p99，但worst State改善很小。由于最终只差3.71%且末段仍改善，原“失败即停”规则被一次明确、有限的续训例外替代：最多再15k，不再允许第二次续训或H64。续训FAIL即停止扩模；PASS才以H50继续B/R。R之前不实现prior、logvar、采样或KL。

## 2. 固定数据与Mask合同

| 项目 | 合同 |
|---|---|
| 数据 | `/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506` |
| 选择 | 原train-only 32 motion，每个motion恰有variant 0..7，共256 episode |
| 窗口 | T64、stride64、`random_crop=false`；末端窗口规则沿用dataset实现 |
| State/Action | 70/29维归一化值；State前68维连续，后2维contact |
| 输入排除 | RobotInfo、reference、motion ID、per-window identity、action-before-window、dynamics context |
| 输出目录 | 只允许新建`/home/helloworld/bly/runs/<run_id>/` |

固定物理Mask bank每窗口8项：`state_gap_4`、`state_gap_16`、`state_rollout`、`action_gap_4`、
`action_gap_16`、`full_action`、`joint_gap_2`、`joint_gap_8`。State gap保留两端State及对应Action；
Action gap保留完整State和缺口外Action历史；joint gap同时隐藏区间Action与内部State、保留两端State。
`full_state/full_both`只作posterior诊断，不进入B/R物理fit gate。

R阶段按同一八类语义动态训练；评测使用独立seed `20260835`，每窗口16个确定性held-out Mask，每类
出现两次并使用独立start/length。禁止element、feature和semantic Mask。所有mask只覆盖valid位置，
位图和窗口身份均写SHA256。

## 3. F4G直接输出诊断与F4G-O解析上限

F4G为每个window学习一份完整`State[:68]`、contact logits和Action输出表；同一window的八种Mask严格共享该表。模型没有encoder、latent或decoder，Mask只决定loss中哪些坐标有效。

正式run `/home/helloworld/bly/runs/cvae_posterior_direct_output_f4g_t64_20260908_013241`已完成：1,504 windows、12,032 fixtures、9,634,624参数，5k步质量FAIL。step0到5k的global State/Action RMSE从`0.973343/0.992225`降到`0.168638/0.079822`，p99从`3.351939`降到`0.358968`，contact为100%；worst State `2.208194`控制score `110.409677`。checkpoint读回PASS，marker为execution加`cvae.failed`。

该结果的固定解释是：

- 所有主要指标持续改善，证明索引、Mask、loss和梯度路径有效；
- `5000×256/12032≈106.4`，每个fixture的平均更新暴露很少；
- 独立答案表没有跨window共享参数，和H38共享权重的训练条件不同；
- State初始max abs约42，5k后仍约38.4，符合从零开始累计更新位移不足；
- 禁止续训F4G或据此宣称H38表达能力失败。

F4G-O使用相同1,504 windows、12,032 fixtures、Mask seed、loss、evaluator和门禁，但把每个canonical window真值逐位复制到共享输出表，不执行optimizer：

- 连续State/Action必须逐位等于数据目标；contact写为符号正确的`±30` logits；
- 保存目标tensor、参数tensor及初始化SHA256，并要求二者一致；
- 完整评测重复三次，核心metric fingerprint必须完全一致；
- 三次均须通过fit、strict-memory与legacy-exact；
- 通过时生成oracle、execution、fit、strict和exact marker；失败只保留execution与`cvae.failed`。

F4G-O run `/home/helloworld/bly/runs/cvae_posterior_direct_output_oracle_f4go_t64_20260908_023727`已返回quality PASS、score 1.0和下一步`RUN_H38_ENGINEERING_SMOKE`。它表示loss/Mask/evaluator存在解析可达解，并把原F4G定位为稀疏查表优化不足；完整source commit、strict/exact字段和marker仍以run内文件为准。

H38授权必须同时检查`oracle_target_copy=true`、`cvae_posterior_direct_output_oracle.ok`和`cvae_posterior_direct_output_fit.ok`，不能用旧F4G训练run或手工补单个marker绕过。

原F4G训练合同仅作历史解释：

- FP32、AdamW、weight decay0、LR `1e-2`、最低`1e-4`、warmup50；
- micro-batch256、accumulation1、最多5k step、每250 step完整评测；
- smoke只使用前2个固定window，执行step0和2个optimizer step；
- 正式连续3次fit PASS才生成`cvae_posterior_direct_output_fit.ok`；
- 原F4G已经失败，不得重跑或续训。

两种答案表参数量都由实际window数决定，精确公式为`N_window × (65×70 + 64×29)`。F4G-O只是解析loss/evaluator上限，不是训练模型或可部署模型。

## 4. H38/H50模型合同

新kind为`physics_hierarchical_posterior_transformer`，与历史`physics_posterior_transformer`完全
分离。H38精确参数量锁定为37,574,883；唯一fallback H50为51,005,283。

| 组件 | H38 / H50 |
|---|---|
| width / heads / FFN | 384 / 8 / 1536；H50为448 / 8 / 1792 |
| posterior encoder | 6层双向；读取完整真值，输入Mask bit强制为0 |
| condition encoder | 独立4层；只读取masked value、Mask bit、物理时间和类型 |
| latent | global `[B,256]` + local `[B,16,128]`，每4 transitions一组 |
| decoder | 8层；每层query self-attention、condition+17 latent cross-attention、共享FiLM、FFN |
| query | 仅物理时间embedding、State/Action类型和可学习query base，不直接接masked token |
| output | 68维State连续、2维contact logits、29维Action |

posterior global来自CLS；local由对应物理时间组的有效State/Action编码作masked mean pooling，S64归入
第16组。空组使用显式learned fallback。canonical posterior对任意Mask逐位不变。decoder-only接口
`decode_from_hierarchical_latent`不调用posterior encoder，masked真值修改不得影响固定latent输出。

H38/H50均为dropout0、KL0、weight decay0、FP32。micro-batch4、accumulation16、clip1。posterior/
condition encoder与decoder self-attention/FFN峰值LR `1e-4`；latent head、cross-attention、FiLM、
query/input/output峰值`3e-4`；warmup1k后cosine到`1e-6`。每阶段只加载模型参数并重置optimizer、
scheduler、loader与RNG。

## 5. 阶段、门禁和结论

| 阶段 | 训练/评测 | 成功后的唯一动作 |
|---|---|---|
| H38 smoke | 前2 window、step0+2 step、full-both | 工程PASS；不形成质量结论 |
| H38-A | full-both，随机初始化，30k，每1k评测 | 已质量FAIL；仍进入H38-B |
| H38-B | 从A `best_fit.pt` model-only；8 fixed Mask，60k，每2k | 已质量FAIL；不进入H38-R |
| H38-R（BLOCKED） | 从B `best_fit.pt` model-only；动态Mask30k，每2k held-out评测 | 只有B fit后才冻结KL0基线 |
| H50-A复核 | A/B均FAIL后授权；full-both、随机初始化、30k，每1k | 已FAIL；score 1.03707且末段改善 |
| H50-A续训 | 从正式H50-A `last.pt`恢复模型/AdamW，低LR尾段重启，最多15k | PASS进H50-B；FAIL停止扩模 |

fit门禁必须连续三次完整评测同时满足：global State/Action RMSE各`≤2e-2`；每类Mask的worst-window
State/Action RMSE各`≤4e-2`；continuous p99 absolute error `≤8e-2`；contact accuracy精确100%；
zero、cross-window、cross-motion整组latent替换相对正确latent的continuous RMSE均`≥10×`。
continuous max abs只报告，不控制fit PASS；strict/legacy门禁仍保留各自max abs要求。

`strict_memory`仍诊断worst State/Action RMSE和max abs均`≤1e-2`；`legacy_exact`诊断`1e-4 RMSE +
1e-3 max abs`。只有fit控制推进，三种结果使用独立marker。分组donor还分别替换global或全部local，
但不替代整组latent主门禁。

## 6. 产物、marker与执行命令

每个run写summary、逐step JSONL、窗口/fixture hash、97连续feature的归一化与物理误差、full-state/
full-both诊断、整组/分组latent donor、checkpoint读回和五张SVG：training、gate、Mask、feature、latent。
execution marker只表示流程完整；质量失败保留`cvae.failed`且不伪造fit marker。

Ubuntu命令不包含任何Git操作，默认用户已经完成同步：

```bash
RUN=/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h38_autoencode_20260908_030633
jq '{execution_pass,quality_pass,completed_optimizer_steps,best_optimizer_step,best_fit_score,
     best:.best_evaluation,last_three:.last_three_evaluations,
     checkpoint_readback,unique_next_step}' \
  "$RUN/manifests/posterior_hierarchical_t64_summary.json"
cat "$RUN/manifests/source_commit.txt"
find "$RUN/markers" -maxdepth 1 -type f -printf '%f\n' | sort
ls -lh "$RUN/checkpoints"
```

H38-B和H50-A均已完成。当前只执行H50-A续训；源run保持只读：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

unset CVAE_CONFIG CVAE_RUN_DIR CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_WARM_START
unset CVAE_POSTERIOR_HIERARCHICAL_INIT_CHECKPOINT
unset CVAE_POSTERIOR_H38_FAILED_RUN
export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
export CVAE_POSTERIOR_DIRECT_OUTPUT_RUN=/home/helloworld/bly/runs/cvae_posterior_direct_output_oracle_f4go_t64_20260908_023727
export CVAE_POSTERIOR_HIERARCHICAL_PROFILE=H50
export CVAE_POSTERIOR_HIERARCHICAL_RESUME_RUN=/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_20260909_120934

test -f "$CVAE_POSTERIOR_HIERARCHICAL_RESUME_RUN/markers/cvae_posterior_hierarchical_t64_execution.ok"
test -f "$CVAE_POSTERIOR_HIERARCHICAL_RESUME_RUN/markers/cvae.failed"
test -f "$CVAE_POSTERIOR_HIERARCHICAL_RESUME_RUN/checkpoints/last.pt"
bash ./cvae_repro.sh posterior-hierarchical-t64-continue
```

续训恢复模型与AdamW动量，并验证源scheduler；新尾段scheduler在250步内从`1e-6`升至slow/fast `3e-6/1e-5`，之后余弦降回`1e-6`。每1k完整评测，连续3次fit PASS提前结束。v1 checkpoint没有保存DataLoader generator状态，因此loader顺序用独立seed `20260836`确定性重启；manifest必须写明这不是逐bit无缝续跑。

H50-A正式run为`/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h50_autoencode_20260909_120934`，源码`fd7928f26b1a12dfa6e01218c7defd9d5ffe2166`。best step30000：global S/A `0.020202/0.015074`、worst S/A `0.041483/0.022078`、p99/max `0.060677/0.715894`、contact 100%、latent整组ratio `56.32/62.06/70.89`。末三次score `1.06244/1.04331/1.03707`。

H38-B正式run为`/home/helloworld/bly/runs/cvae_posterior_hierarchical_t64_h38_fixed_20260908_133755`，源码`c5932690f43b478fb05687ccf58e6834b0243a32`。best step60000：global S/A `0.024991/0.016406`、worst S/A `0.047077/0.025066`、p99/max `0.081572/1.611736`、contact 100%、latent整组ratio `18.83/21.42/24.47`；原summary门禁FAIL，按新门禁重算仍FAIL、score `1.24953`。

每次run结束后先回传summary、最后三个evaluation、marker列表、checkpoint列表和`source_commit.txt`，
再把实际路径、hash、指标、结论与唯一下一步写回plan.md。smoke checkpoint不得用于正式初始化。

## 7. 声明边界

即使H38-R全部通过，也只证明已见32-motion、T64窗口上的deterministic posterior记忆，以及相同
已见序列上的新物理Mask查询能力。它不证明conditional prior、latent随机采样、KL、未见motion泛化、
因果rollout或真实部署时State↔Action推理。只有R通过后，才按既定三路径协议新增posterior mean、
posterior sample和conditional prior sample接口。
