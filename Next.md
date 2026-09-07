# 活动合同：32-Motion、T64 层级 Latent Transformer 容量实验

最后更新：2026-09-08

状态：正式F4G已质量FAIL，但F4G-O已在同一1,504 windows、12,032 fixtures上以0 optimizer step取得`quality_pass=true`和`best_fit_score=1.0`。这证明解析loss/Mask/evaluator上限可达，并把原F4G定位为稀疏查表优化不足。当前唯一下一步是以F4G-O run只读授权运行H38工程smoke；不得续训F4G或把smoke写成质量结论。历史F4D/F4E/F4F、源HDF5和checkpoint保持只读。事实结果仍以
[plan.md](plan.md)为唯一台账，安全规则见[AGENTS.md](AGENTS.md)。

## 1. 问题与顺序

本轮目标是在32 motion、256 episode的overfit subset上，把窗口降至64 transitions，先证明目标函数
和评测门禁可达，再检验层级posterior结构。固定顺序为：

```text
H38 smoke
→ H38-A full-both autoencoding
→ H38-B fixed physical Masks
→ H38-R held-out random physical Masks
→ 冻结KL=0基线后再实现KL三路径
```

只有F4G-O通过而H38-A失败时允许一次H50-A；不得继续建立参数阶梯。H38-A通过而B失败定位为condition
融合问题，B通过而R失败定位为随机Mask覆盖问题。R之前不实现prior、logvar、采样或KL。

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
| H38 smoke | 前2 window、step0+2 step、full-both | 审核summary后启动H38-A |
| H38-A | full-both，随机初始化，最多20k，每1k评测 | 进入H38-B |
| H38-B | 从A `best_fit.pt` model-only；8 fixed Mask，最多60k，每2k | 进入H38-R |
| H38-R | 从B `best_fit.pt` model-only；动态Mask30k，每2k held-out评测 | 冻结KL0基线 |
| H50-A | 仅F4G-O PASS且H38-A FAIL；随机20k | PASS后以H50继续B/R；FAIL停止扩模 |

fit门禁必须连续三次完整评测同时满足：global State/Action RMSE各`≤1e-2`；每类Mask的worst-window
State/Action RMSE各`≤2e-2`；continuous p99 absolute error `≤5e-2`；contact accuracy精确100%；
zero、cross-window、cross-motion整组latent替换相对正确latent的continuous RMSE均`≥10×`。

`strict_memory`仍诊断worst State/Action RMSE和max abs均`≤1e-2`；`legacy_exact`诊断`1e-4 RMSE +
1e-3 max abs`。只有fit控制推进，三种结果使用独立marker。分组donor还分别替换global或全部local，
但不替代整组latent主门禁。

## 6. 产物、marker与执行命令

每个run写summary、逐step JSONL、窗口/fixture hash、97连续feature的归一化与物理误差、full-state/
full-both诊断、整组/分组latent donor、checkpoint读回和五张SVG：training、gate、Mask、feature、latent。
execution marker只表示流程完整；质量失败保留`cvae.failed`且不伪造fit marker。

Ubuntu命令不包含任何Git操作，默认用户已经完成同步：

```bash
cd /home/helloworld/bly/state-action-cvae
source /home/helloworld/bly/sonic-repro/.venv-sonic/bin/activate

export CVAE_DATASET_RUN=/home/helloworld/bly/runs/cvae_overfit_subset_20260828_234506
unset CVAE_CONFIG CVAE_RUN_DIR CVAE_INIT_CHECKPOINT CVAE_POSTERIOR_WARM_START
unset CVAE_POSTERIOR_HIERARCHICAL_INIT_CHECKPOINT CVAE_POSTERIOR_HIERARCHICAL_PROFILE

# F4G-O已通过；当前只执行H38工程smoke：
export CVAE_POSTERIOR_DIRECT_OUTPUT_RUN=/home/helloworld/bly/runs/cvae_posterior_direct_output_oracle_f4go_t64_20260908_023727
export CVAE_POSTERIOR_HIERARCHICAL_PROFILE=H38
unset CVAE_POSTERIOR_HIERARCHICAL_INIT_CHECKPOINT
test -f "$CVAE_POSTERIOR_DIRECT_OUTPUT_RUN/markers/cvae_posterior_direct_output_oracle.ok"
test -f "$CVAE_POSTERIOR_DIRECT_OUTPUT_RUN/markers/cvae_posterior_direct_output_fit.ok"
bash ./cvae_repro.sh posterior-hierarchical-t64-smoke
```

H38 smoke审核通过后：

```bash
bash ./cvae_repro.sh posterior-hierarchical-t64-autoencode
```

H38-A通过后执行B，B通过后执行R：

```bash
export CVAE_POSTERIOR_HIERARCHICAL_INIT_CHECKPOINT=/home/helloworld/bly/runs/<h38_a_run>/checkpoints/best_fit.pt
bash ./cvae_repro.sh posterior-hierarchical-t64-fixed

export CVAE_POSTERIOR_HIERARCHICAL_INIT_CHECKPOINT=/home/helloworld/bly/runs/<h38_b_run>/checkpoints/best_fit.pt
bash ./cvae_repro.sh posterior-hierarchical-t64-random
```

若且仅若H38-A质量失败，清空初始化并执行唯一H50-A：

```bash
export CVAE_POSTERIOR_HIERARCHICAL_PROFILE=H50
export CVAE_POSTERIOR_H38_FAILED_RUN=/home/helloworld/bly/runs/<failed_h38_a_run>
unset CVAE_POSTERIOR_HIERARCHICAL_INIT_CHECKPOINT
bash ./cvae_repro.sh posterior-hierarchical-t64-autoencode
```

每次run结束后先回传summary、最后三个evaluation、marker列表、checkpoint列表和`source_commit.txt`，
再把实际路径、hash、指标、结论与唯一下一步写回plan.md。smoke checkpoint不得用于正式初始化。

## 7. 声明边界

即使H38-R全部通过，也只证明已见32-motion、T64窗口上的deterministic posterior记忆，以及相同
已见序列上的新物理Mask查询能力。它不证明conditional prior、latent随机采样、KL、未见motion泛化、
因果rollout或真实部署时State↔Action推理。只有R通过后，才按既定三路径协议新增posterior mean、
posterior sample和conditional prior sample接口。
