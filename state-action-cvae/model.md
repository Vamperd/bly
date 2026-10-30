# PhysicsTransformer CVAE：模型结构、物理语义与训练流程

本文记录 SONIC Physics State–Action 早期（2026-08）主线 **PhysicsTransformer CVAE**（T=128）的结构与训练合同；当前 65-token hierarchical CVAE 主线见工作区根目录 `model.md`。本文对应外层仓库提交 **197a1730fd829a512e153755f56e6e97d4b1329d**，事实来源是 [models.py](src/cvae_sa/models.py)、[dataset.py](src/cvae_sa/dataset.py)、[masking.py](src/cvae_sa/masking.py)、[losses.py](src/cvae_sa/losses.py) 与 [trainer.py](src/cvae_sa/trainer.py)（完整定位见文末）。

> 状态：120000 optimizer step 的 parent 模型已完成训练并有评测结果；Action-focused fine-tune 只有代码和配置，没有 Ubuntu smoke 或正式训练成功日志，不能写成已训练完成。

## 1. 目标与能力边界

模型学习 $p_\theta(S_{0:T},A_{0:T-1}\mid R)$（$S$ 为 State 序列，$A$ 为 Action 序列，$R$ 为机器人静态与仿真信息）。训练不是单一 ELBO，而是 CVAE 潜变量 + masked reconstruction + 前向/逆向动力学 + rollout + cycle + 辅助监督的多任务目标。

能力：
1. **任意联合补全**：在 State、Action 或二者中遮挡元素、时间块、特征或身体语义组并重建。
2. **前向动力学**：由可读 $S_t,A_t$ 预测 $S_{t+1}$，模型内 autoregressive rollout 最多 8 步。
3. **Action 推断**：inverse（$S_t,S_{t+1}\to A_t$）或 history（$S_{\leq t},A_{<t}\to A_t$）。
4. **物理验证**：预测 State 转运动学视频，或预测 Action 转回 raw policy Action 在 Isaac 重放。

边界：
- SONIC released policy 使用自己的 observation、历史和 future reference motion；本 CVAE 只读执行后的 State–Action 轨迹与 RobotInfo，**不接收** goal、reference motion、motion ID、package、episode outcome 或动作语义标签。
- history Action 学的是数据中的控制规律，不等价于恢复 SONIC 完整策略；同一物理历史可能对应多个合理 Action。
- 给定完整 Action 计划时可预测未来 State，但它不是不知道未来 Action 的在线控制器。
- 70 维 State 不含世界 root $x,y$、绝对 yaw 或 root quaternion（利用平地平移/heading 对称性）；State 视频需给定真实 root anchor 再重建 root 轨迹（§7）。

## 2. 维度、State/Action 定义与归一化

| 符号 | 值 | 定义 |
|---|---:|---|
| $T$ | 128 | 每窗口控制转移数（$\Delta t=0.02$ s，覆盖 2.56 s） |
| $J$ / $D_A$ | 29 | canonical 关节数 / Action 维数 |
| $D_S$ | 70 | State 维数 |
| $d$ / $d_J$ | 384 / 128 | 时序 token 宽度 / joint-aware 子网络宽度 |
| $H$ / $d_h$ | 8 / 48 | 时序 attention head 数 / 每 head 维度 |
| $D_Z$ | 96 | 全局 CVAE latent 维数 |
| $L_E$ / $L_D$ | 6 / 8 | 时序 Encoder / Decoder 层数 |

| batch 张量 | Shape | 是否进入 hidden 主模型 |
|---|---:|---|
| physical_state | $[B,129,70]$ | 是 |
| action | $[B,128,29]$ | 是 |
| action_before_window | $[B,29]$ | 是 |
| joint_robot_information | $[B,29,11]$ | 是 |
| joint_actuator_type | $[B,29]$ | 是 |
| global_robot_information | $[B,9]$ | 是 |
| dynamics_context | $[B,648]$ | hidden 模式否；oracle/explicit 模式是 |
| auxiliary_transition | $[B,128,35]$ | 只作监督目标 |
| progress | $[B,129]$ | 是 |
| valid_state / valid_action | $[B,129]$ / $[B,128]$ | attention 与 loss 门控 |

Physics v4 不再存储重复的 previous Action，previous_action 的 Shape 是 $[B,129,0]$，只为兼容 batch 接口；模型内没有 previous-Action tokenizer 或输出分支。

State $S_t=[q_t^{\mathrm{can}},\dot q_t,v_t^R,\omega_t^R,\gamma_t^R,h_t,c_t]\in\mathbb R^{70}$：

| 分量 | 索引 | 维数 | 坐标系与单位 |
|---|---:|---:|---|
| $q_t^{\mathrm{can}}$ | 0:29 | 29 | canonical joint position（$q^{\mathrm{abs}}-q^{\mathrm{nom}}$），rad |
| $\dot q_t$ | 29:58 | 29 | 实际 joint velocity，rad/s |
| $v_t^R$ | 58:61 | 3 | pelvis/robot frame 下的 base linear velocity，m/s |
| $\omega_t^R$ | 61:64 | 3 | pelvis/robot frame 下的 base angular velocity，rad/s |
| $\gamma_t^R$ | 64:67 | 3 | 世界向下方向在 pelvis frame 中的单位向量（$\lVert\gamma\rVert_2=1$） |
| $h_t$ | 67:68 | 1 | pelvis 相对平地高度，m |
| $c_t$ | 68:70 | 2 | 左/右足接触二值标签（足端接触力是否超过 10 N） |

Action：ActionManager 处理后的绝对关节目标减 nominal，$A_{t,j}=q_{t,j}^{\mathrm{target,processed,abs}}-q_j^{\mathrm{nom}}\in\mathbb R^{29}$，与 $q^{\mathrm{can}}$ 同顺序、同零点、rad 单位。raw action、scale、offset、clip 保存在 HDF5/schema 中用于精确反变换和 Isaac replay，不是模型输入。时间对齐 $S_t+A_t\to S_{t+1}$；窗口前一个命令 $A_{-1}^{\mathrm{win}}$ 在窗口起点 $s>0$ 时为 $A_{s-1}$，从 episode 起点开始时读 initial processed target，不人为填零。

RobotInfo 与辅助目标：
- 每关节 $r_j\in\mathbb R^{11}$：nominal、canonical 位置上下限、速度/力矩限制、$K_p/K_d$、armature、joint friction、actuator delay 上下限；actuator type ID $u_j$。
- 全局 $g\in\mathbb R^9$：simulation/control time step、decimation、世界 gravity、solver position/velocity iteration count、接触阈值。
- 随机化动力学 $\xi\in\mathbb R^{648}$（body mass、inertia、center of mass、body material、ground material）：默认 hidden 模式不输入网络，让未观测动力学差异由轨迹与 latent 表示；oracle/explicit 模式显式编码，只作上限对照。
- 辅助目标 $Y_t^{\mathrm{aux}}=[\bar\tau_t,I_t^L,I_t^R]\in\mathbb R^{35}$：一个 control interval 内平均 applied joint torque（29）与左右足接触冲量（3+3），不进入输入 State，只约束 $(S_t,A_t)$ 的 transition representation。

归一化：连续量用 train split 统计量 z-score（State、Action、RobotInfo、可见 dynamics context、auxiliary target 各自统计，validation/test 不参与）；gravity 与 contact 在 normalization 文件中强制 $\mu=0,\sigma=1$，保持原始单位向量和二值标签。训练、损失和 normalized RMSE 默认在标准化空间，反归一化后才能解释为 rad、rad/s、m/s 或 m。

窗口与 split：训练随机裁剪，validation/test 固定 stride 64；短 episode 右侧 padding 填零，由 valid mask 阻止进入 attention key 和 loss。按 motion_key 划分 split（同一 motion 的全部 variant 和窗口只属于一个 split）。parent 均匀裁剪；Action fine-tune 以 0.5 概率均匀裁剪、0.5 概率从每个 episode 中 Action derivative energy $E(s)=\frac{1}{T-1}\sum_{t=s}^{s+T-2}\lVert A_{t+1}-A_t\rVert_2^2$ 最高 25% 的窗口起点采样，不改变 validation 的固定窗口。

<!-- SACVAE_CONTINUE -->
