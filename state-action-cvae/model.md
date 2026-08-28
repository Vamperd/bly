# PhysicsTransformer CVAE：模型结构、物理语义与训练流程

本文面向希望从物理定义、数学原理一直理解到当前 PyTorch 实现的读者，完整说明 SONIC Physics State–Action 主线所使用的 **PhysicsTransformer CVAE**。本文对应外层仓库提交 **197a1730fd829a512e153755f56e6e97d4b1329d**，主要事实来源是 [models.py](src/cvae_sa/models.py)、[dataset.py](src/cvae_sa/dataset.py)、[masking.py](src/cvae_sa/masking.py)、[losses.py](src/cvae_sa/losses.py) 与 [trainer.py](src/cvae_sa/trainer.py)。

> 状态说明：120000 optimizer step 的 parent 模型已经完成训练并有评测结果；Action-focused fine-tune 的代码和配置已经实现，但本文所依据的工作区没有收到其 Ubuntu smoke 或正式训练成功日志，因此不能把 fine-tune 写成已经训练完成。

---

## 1. 研究目标、条件分布与能力边界

### 1.1 模型究竟在学习什么

对一段包含 $T$ 个控制转移的窗口，定义 State 序列为 $S_{0:T}$，Action 序列为 $A_{0:T-1}$，机器人静态与仿真信息统称为 $R$。当前模型希望学习的对象可以概括为：

$$p_\theta(S_{0:T},A_{0:T-1}\mid R).$$

这里的 $\theta$ 表示全部可训练网络参数。实际训练并不是直接最大化一个能够严格归一化的显式概率密度，而是使用 conditional VAE 潜变量、masked reconstruction、前向/逆向动力学、rollout、cycle 和辅助监督组成的多任务目标。因此更准确的说法是：

> 当前网络是一个以 CVAE 为生成骨架、以多种物理一致性目标共同训练的 State–Action 序列模型，而不是只有一个标准 ELBO 项的纯粹概率模型。

模型具备以下四类核心能力：

1. **任意联合补全**：在 State、Action 或二者中遮挡元素、时间块、特征或身体语义组，再重建被遮挡量。
2. **前向动力学**：由允许读取的 $S_t,A_t$ 预测 $S_{t+1}$，并执行最多 8 步的模型内 autoregressive rollout。
3. **Action 推断**：由 $S_t,S_{t+1}$ 做 inverse Action，或只由因果历史 $S_{\leq t},A_{<t}$ 做 history Action。
4. **物理验证**：将预测 State 转换为运动学视频，或将预测 Action 转回 raw policy Action 后在 Isaac 中重放。

### 1.2 它与 SONIC policy 的区别

SONIC released policy 是采集数据时的控制策略。它仍然使用 SONIC 自己的 observation、历史信息和 future reference motion 去追踪动作。PhysicsTransformer CVAE 只读取执行后采集到的 State–Action 轨迹与 RobotInfo。

当前 CVAE **不接收** goal、reference motion、motion ID、package、episode outcome 或动作语义标签。因此：

- history Action 预测是在学习数据中出现的控制规律，不等价于恢复 SONIC 的完整策略输入输出映射；
- 同一个物理历史可能对应多个合理 Action，Action 任务天然具有多解性；
- 模型可以在给定完整 Action 计划时预测未来 State，但它不是不知道未来 Action 的在线控制器。

### 1.3 70维 State 的对称性与缺失量

当前 State 不包含绝对世界坐标中的 root $x,y$，也不包含绝对 yaw 或 root quaternion。这利用了平地运动对世界平移和绝对 heading 的近似对称性，减少模型必须学习的无关自由度。

因此模型不会直接输出完整的 floating-base pose。State 视频需要额外给定一个真实 root anchor，再使用预测的 base velocity、base angular velocity、gravity 和 base height 重建 root 轨迹。这个后处理将在第 7 章说明。

---

## 2. 数学符号、物理量、张量 Shape 与归一化

### 2.1 基本索引与维度

| 符号 | 当前值 | 定义 |
|---|---:|---|
| $B$ | 运行时决定 | micro batch 中的窗口数 |
| $T$ | 128 | 每个窗口的控制转移数 |
| $J$ | 29 | canonical 关节数，同时也是 Action 维数 |
| $D_S$ | 70 | State 维数 |
| $D_A$ | 29 | Action 维数 |
| $d$ | 384 | 时序 Transformer 的 token 宽度 |
| $d_J$ | 128 | joint-aware 子网络的宽度 |
| $H$ | 8 | 时序 self-attention 的 head 数 |
| $d_h$ | 48 | 每个 attention head 的维度，$d_h=d/H$ |
| $D_Z$ | 96 | 全局 CVAE latent 维数 |
| $L_E$ | 6 | 时序 Encoder 层数 |
| $L_D$ | 8 | 时序 Decoder 层数 |

一个标准训练 batch 的主要张量为：

| 名称 | Shape | 是否进入 hidden 主模型 |
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
| valid_state | $[B,129]$ | attention 与 loss 门控 |
| valid_action | $[B,128]$ | attention 与 loss 门控 |

Physics v4 不再存储重复的 previous Action，所以 previous_action 的 Shape 是 $[B,129,0]$。它只为兼容统一 batch 接口而存在，PhysicsTransformer 内部没有 previous-Action tokenizer 或输出分支。

### 2.2 State 的70维物理定义

对时间 $t$，定义：

$$S_t=\left[q_t^{\mathrm{can}},\dot q_t,v_t^R,\omega_t^R,\gamma_t^R,h_t,c_t\right]\in\mathbb R^{70}.$$

各部分含义如下。

| 分量 | 索引 | 维数 | 坐标系与单位 |
|---|---:|---:|---|
| $q_t^{\mathrm{can}}$ | 0:29 | 29 | canonical joint position，rad |
| $\dot q_t$ | 29:58 | 29 | 实际 joint velocity，rad/s |
| $v_t^R$ | 58:61 | 3 | pelvis/robot frame 下的 base linear velocity，m/s |
| $\omega_t^R$ | 61:64 | 3 | pelvis/robot frame 下的 base angular velocity，rad/s |
| $\gamma_t^R$ | 64:67 | 3 | 世界向下方向在 pelvis frame 中的单位向量 |
| $h_t$ | 67:68 | 1 | pelvis 相对平地高度，m |
| $c_t$ | 68:70 | 2 | 左、右足是否接触的二值标签 |

canonical joint position 以每个环境的 nominal joint pose 为零点：

$$q_{t,j}^{\mathrm{can}}=q_{t,j}^{\mathrm{abs}}-q_j^{\mathrm{nom}},\qquad j=1,\ldots,J.$$

gravity 满足理想约束：

$$\lVert\gamma_t^R\rVert_2=1.$$

接触标签 $c_t=(c_t^L,c_t^R)$ 来自足端接触力是否超过 10 N，而不是需要连续回归的力值：

$$c_t^f\in\{0,1\},\qquad f\in\{L,R\}.$$

### 2.3 Action 的29维物理定义

ActionManager 已将 raw policy Action 处理成绝对关节目标。主模型 Action 再将其转换到与 State joint position 相同的 nominal 零点：

$$A_{t,j}=q_{t,j}^{\mathrm{target,processed,abs}}-q_j^{\mathrm{nom}}.$$

因此：

$$A_t\in\mathbb R^{29}.$$

$A_t$ 与 $q_t^{\mathrm{can}}$ 具有相同的关节顺序、零点和 rad 单位。raw action、scale、offset、clip 仍保存在 HDF5/schema 中用于精确反变换和 Isaac replay，但不是 PhysicsTransformer 的输入。

严格的时间对齐是：

$$S_t+A_t\longrightarrow S_{t+1}.$$

窗口开始前的一个控制命令记为 $A_{-1}^{\mathrm{win}}$。如果窗口从 episode 的第 $s>0$ 步开始，则：

$$A_{-1}^{\mathrm{win}}=A_{s-1}.$$

如果窗口从 episode 起点开始，则读取 initial processed target，而不是人为填零。

### 2.4 RobotInfo、隐藏动力学与辅助目标

第 $j$ 个关节的结构化机器人信息记为 $r_j\in\mathbb R^{11}$。它包括 nominal、canonical 位置上下限、速度/力矩限制、$K_p/K_d$、armature、joint friction 和 actuator delay 上下限。

执行器类型 ID 记为 $u_j$。全局仿真信息记为 $g\in\mathbb R^9$，包含 simulation/control time step、decimation、世界 gravity、solver position/velocity iteration count 和接触阈值。

随机化动力学上下文记为：

$$\xi\in\mathbb R^{648}.$$

$\xi$ 由 body mass、inertia、center of mass、body material 和 ground material 拼接得到。默认 hidden 模式不把 $\xi$ 输入网络，意图是让未观测动力学差异由轨迹与 latent 表示；oracle/explicit 模式将其显式编码，只用于上限对照。

辅助监督目标为：

$$Y_t^{\mathrm{aux}}=\left[\bar\tau_t,I_t^{L},I_t^{R}\right]\in\mathbb R^{35}.$$

其中 $\bar\tau_t\in\mathbb R^{29}$ 是一个 control interval 内的平均 applied joint torque，$I_t^L,I_t^R\in\mathbb R^3$ 是左右足接触冲量。它们不进入输入 State，只用于约束由 $(S_t,A_t)$ 形成的 transition representation。

### 2.5 训练空间与物理空间

除 actuator type ID、valid mask、gravity 和 contact 的特殊处理外，连续量使用 train split 统计量标准化。对任意连续特征 $x_i$：

$$\tilde{x}_i=\frac{x_i-\mu_i}{\sigma_i}.$$

反归一化为：

$$x_i=\tilde{x}_i\sigma_i+\mu_i.$$

State、Action、RobotInfo、可见的 dynamics context 与 auxiliary target 都分别使用各自统计量。统计量只由 train split 计算，validation/test 不参与。

gravity 与 contact 在 normalization 文件中被强制设置为：

$$\mu_{\gamma}=\mu_c=0,\qquad \sigma_{\gamma}=\sigma_c=1.$$

所以这两组量数值上保持原始单位向量和二值标签。模型训练、损失和 normalized RMSE 默认都在标准化空间计算；只有反归一化后，误差才能解释为 rad、rad/s、m/s 或 m。

### 2.6 窗口、padding、split 与采样

控制周期为：

$$\Delta t=0.02\ \mathrm{s}.$$

长度 $T=128$ 的窗口覆盖：

$$T\Delta t=2.56\ \mathrm{s}.$$

训练窗口随机裁剪；validation/test 使用固定 stride 64。短 episode 在右侧 padding，padding 张量填零，但 valid mask 会同时阻止它进入 attention key 和 loss。

数据按 motion_key 划分，而不是按窗口随机划分，因此同一 motion 的全部 variant 和窗口只能属于 train、validation、test 中的一个 split。

常规 parent 训练使用均匀窗口裁剪。Action fine-tune 以 0.5 概率使用均匀裁剪，以 0.5 概率从每个 episode 中 Action derivative energy 最高的 25% 窗口起点中均匀采样。长度为 $T$ 的候选窗口起点 $s$ 的能量分数是：

$$E(s)=\frac{1}{T-1}\sum_{t=s}^{s+T-2}\lVert A_{t+1}-A_t\rVert_2^2.$$

这一策略提高高动态 Action 片段在 fine-tune 中出现的频率，但不会改变 validation 的固定窗口。

---

## 3. Mask 构造、Robot Encoder、Token 化与 Transformer

### 3.1 Mask 约定

State 与 Action mask 分别记为：

$$M^S\in\{0,1\}^{(T+1)\times D_S},\qquad M^A\in\{0,1\}^{T\times D_A}.$$

当前代码的约定是 **1 表示隐藏/需要预测，0 表示可见**。可见输入通过零值替换得到：

$$\tilde S^{\mathrm{vis}}=(1-M^S)\odot\tilde S.$$

$$\tilde A^{\mathrm{vis}}=(1-M^A)\odot\tilde A.$$

仅仅把值设为零会与“标准化后恰好等于均值”的真实值混淆，因此 tokenizer 同时接收 mask bit。网络能够区分“真实零值”和“被遮挡后填零”。

MaskBatch 还分别保存 input mask 和 loss mask。二者多数情况下相同，但 history Action 中，未来 State 是为了保证因果性而隐藏，并不自动成为 State reconstruction target；forward/inverse 也会显式保证其物理关系所需的边界输入不被随机 overlay 破坏。

### 3.2 三大任务组

常规 parent 训练的任务组概率是：

$$P(\mathrm{forward},\mathrm{action\ inference},\mathrm{arbitrary})=(0.40,0.35,0.25).$$

Action fine-tune 改为：

$$P(\mathrm{forward},\mathrm{action\ inference},\mathrm{arbitrary})=(0.30,0.45,0.25).$$

其中 action inference 在 fine-tune 中按 $2/3$ inverse、$1/3$ history Action 划分，所以无条件总概率约为 30% inverse 与 15% history。

arbitrary 任务可以选择 State-only、Action-only 或 Both。parent 的比例是 $(0.25,0.25,0.50)$；fine-tune 改为：

$$P(\mathrm{State\ only},\mathrm{Action\ only},\mathrm{Both})=(0.20,0.40,0.40).$$

arbitrary 内部再随机选择 element、step、feature 或 semantic granularity。semantic joint group 按左右腿、腰和左右臂等 canonical 关节区间构造；State 还可以整组隐藏 base linear velocity、base angular velocity、gravity 或 contact。

### 3.3 Joint-aware Robot Encoder

令 $\phi_R$ 表示输入11维、输出 $d_J=128$ 的两层 GELU MLP；$e_j$ 是 joint ID embedding；$e_u$ 是 actuator type embedding。第 $j$ 个关节的初始表示为：

$$R_j^{(0)}=\phi_R(\tilde r_j)+e_j(j)+e_u(u_j).$$

29个关节表示组成：

$$R^{(0)}=\left[R_1^{(0)},\ldots,R_J^{(0)}\right]\in\mathbb R^{J\times d_J}.$$

它经过 2 层、4头、FFN 宽度 $4d_J=512$ 的 spatial TransformerEncoder：

$$R=\operatorname{RobotEncoder}(R^{(0)})\in\mathbb R^{J\times d_J}.$$

这里的 attention 发生在 29 个关节之间，而不是时间之间。它让膝、踝、髋、腰和手臂的控制属性相互交换信息。

对关节维求均值并映射到时序宽度，再加全局 RobotInfo：

$$\bar R=W_{\mathrm{pool}}\left(\frac{1}{J}\sum_{j=1}^{J}R_j\right)+\phi_G(\tilde g).$$

在 oracle/explicit 模式中还会加入：

$$\bar R_{\mathrm{oracle}}=\bar R+\phi_\xi(\tilde\xi).$$

默认 hidden 模式不存在 $\phi_\xi$，因此 648维随机动力学 context 不产生参数路径，也不会被误读。

### 3.4 State tokenizer

将前58维 State 按关节重排。第 $j$ 个关节在时刻 $t$ 的 tokenizer 输入是：

$$x_{t,j}^S=\left[\tilde q_{t,j}^{\mathrm{can}},\tilde{\dot q}_{t,j},M_{t,j}^{q},M_{t,j}^{\dot q},R_j\right]\in\mathbb R^{132}.$$

经 joint State MLP 后对 29 个关节平均，再投影到 $d=384$：

$$h_{t,\mathrm{joint}}^S=W_S^{\mathrm{pool}}\left(\frac{1}{J}\sum_{j=1}^{J}\phi_S^{J}(x_{t,j}^S)\right).$$

State 剩余12维记为 $b_t=[v_t^R,\omega_t^R,\gamma_t^R,h_t,c_t]$，同时拼接12维 mask 和 episode progress $\rho_t$：

$$x_t^{\mathrm{base}}=\left[\tilde b_t,M_t^{\mathrm{base}},\rho_t\right]\in\mathbb R^{25}.$$

$$h_{t,\mathrm{base}}^S=\phi_S^{\mathrm{base}}(x_t^{\mathrm{base}})\in\mathbb R^{384}.$$

最终 State token 是：

$$X_t^S=\phi_S^{\mathrm{fuse}}\left([h_{t,\mathrm{joint}}^S,h_{t,\mathrm{base}}^S]\right)\in\mathbb R^{384}.$$

### 3.5 Action tokenizer

第 $j$ 个 Action 输入包含标准化 Action、mask bit 和同一个 joint memory：

$$x_{t,j}^A=\left[\tilde A_{t,j},M_{t,j}^A,R_j\right]\in\mathbb R^{130}.$$

先逐关节编码、跨关节平均和投影，再拼接 progress：

$$h_{t,\mathrm{joint}}^A=W_A^{\mathrm{pool}}\left(\frac{1}{J}\sum_{j=1}^{J}\phi_A^J(x_{t,j}^A)\right).$$

$$X_t^A=\phi_A^{\mathrm{fuse}}\left([h_{t,\mathrm{joint}}^A,\rho_t]\right)\in\mathbb R^{384}.$$

$A_{-1}^{\mathrm{win}}$ 使用相同 Action tokenizer，但 progress 置零，再经过 before_fusion MLP。

### 3.6 交错 Token 序列

对 $T=128$，时序 token 顺序是：

$$X=\left[X_{-1}^A,X_0^S,X_0^A,X_1^S,X_1^A,\ldots,X_{T-1}^A,X_T^S\right].$$

总 token 数为：

$$L=2T+2=258.$$

type embedding 有三类：before Action、State、Action。任务组 embedding 也有三类：forward、action inference、arbitrary。tokenizer 输出之后，当前源码执行：

$$X_i\leftarrow X_i+e_{\mathrm{type}(i)}+\bar R+e_{\mathrm{task}}.$$

valid sequence 为：

$$V=\left[1,V_0^S,V_0^A,\ldots,V_{T-1}^A,V_T^S\right]\in\{0,1\}^{L}.$$

### 3.7 RoPE 多头自注意力

时序 Transformer 不使用 learned absolute position embedding，而是对 query 和 key 使用 rotary position embedding。每个 head 的维数为：

$$d_h=\frac{384}{8}=48.$$

对 position $p$ 和二维通道对索引 $i$，旋转角为：

$$\vartheta_{p,i}=p\exp\left(-\frac{2i}{d_h}\log 10000\right).$$

对任意二维分量 $(x_{2i},x_{2i+1})$：

$$\operatorname{RoPE}_p\begin{pmatrix}x_{2i}\\x_{2i+1}\end{pmatrix}=\begin{pmatrix}\cos\vartheta_{p,i}&-\sin\vartheta_{p,i}\\\sin\vartheta_{p,i}&\cos\vartheta_{p,i}\end{pmatrix}\begin{pmatrix}x_{2i}\\x_{2i+1}\end{pmatrix}.$$

第 $h$ 个 attention head 为：

$$Q_h=XW_h^Q,\qquad K_h=XW_h^K,\qquad U_h=XW_h^V.$$

$$\operatorname{Attn}_h(X)=\operatorname{softmax}\left(\frac{\operatorname{RoPE}(Q_h)\operatorname{RoPE}(K_h)^\top}{\sqrt{d_h}}+\mathcal B_{\mathrm{valid}}+\mathcal B_{\mathrm{causal}}\right)U_h.$$

$\mathcal B_{\mathrm{valid}}$ 将 padding key 置为 $-\infty$；需要因果性时，$\mathcal B_{\mathrm{causal}}$ 将所有未来 key 置为 $-\infty$。

Transformer block 使用 Pre-Norm 残差结构：

$$X' = X+\operatorname{MHA}(\operatorname{LN}(X)).$$

$$X''=X'+\operatorname{FFN}(\operatorname{LN}(X')).$$

FFN 为 $384\rightarrow1536\rightarrow384$，中间使用 GELU 和 dropout 0.1。Encoder 有6层，Decoder有8层，两个 stack 末尾都再做一次 LayerNorm。

forward 和 history Action 使用 causal attention；inverse 与 arbitrary completion 使用双向 attention。history Action 的未来 State 还会显式 mask，防止其内容经任何非因果路径进入预测。

---

## 4. Conditional VAE、Decoder 与各输出头

### 4.1 Visible prior 与 full posterior

令 $X^{\mathrm{vis}}$ 表示 mask 后的 token 序列，$X^{\mathrm{full}}$ 表示同一窗口的完整 token 序列。两者使用 **同一个** 6层 Encoder，不是两套独立 Encoder：

$$H^p=\operatorname{Encoder}(X^{\mathrm{vis}}).$$

$$H^q=\operatorname{Encoder}(X^{\mathrm{full}}).$$

对有效 token 做 masked mean pooling：

$$\bar H^p=\frac{\sum_{i=1}^{L}V_iH_i^p}{\max\left(\sum_{i=1}^{L}V_i,1\right)}.$$

$$\bar H^q=\frac{\sum_{i=1}^{L}V_iH_i^q}{\max\left(\sum_{i=1}^{L}V_i,1\right)}.$$

conditional prior 和 posterior 都是对角高斯：

$$p_\theta(z\mid X^{\mathrm{vis}},R)=\mathcal N\left(\mu_p,\operatorname{diag}(\sigma_p^2)\right).$$

$$q_\phi(z\mid X^{\mathrm{full}},R)=\mathcal N\left(\mu_q,\operatorname{diag}(\sigma_q^2)\right).$$

网络线性层直接输出 mean 与 log variance：

$$[\mu_p,\log\sigma_p^2]=W_p\bar H^p+b_p.$$

$$[\mu_q,\log\sigma_q^2]=W_q\bar H^q+b_q.$$

实现会将 log variance 截断到 $[-12,8]$，避免极端方差造成数值不稳定。

### 4.2 重参数化与训练/推理差异

训练的默认 forward 使用 posterior：

$$\epsilon\sim\mathcal N(0,I),\qquad z=\mu_q+\exp\left(\frac{1}{2}\log\sigma_q^2\right)\odot\epsilon.$$

推理时设置 sample_from_prior，改用：

$$z=\mu_p+\exp\left(\frac{1}{2}\log\sigma_p^2\right)\odot\epsilon.$$

deterministic 模式不采样，直接令 $z=\mu_p$ 或 $z=\mu_q$。这就是 prior_mean 推理。

完整未遮挡真值只用于训练 posterior。推理时条件来自 masked visible sequence，目标真值不会进入 prior。

### 4.3 Decoder condition 与一次重要实现细节

全局 latent 投影后与 Robot summary 组成 Decoder condition：

$$C^{\mathrm{global}}=\bar R+W_Zz.$$

Decoder 输入为：

$$D^{(0)}=X^{\mathrm{vis}}+C^{\mathrm{global}}.$$

需要特别注意：$X^{\mathrm{vis}}$ 在 token 构造时已经加入一次 $\bar R$，而 $C^{\mathrm{global}}$ 又包含一次 $\bar R$。所以按当前源码，Encoder token 含一次 Robot summary，Decoder 初始输入相对于 tokenizer 原始输出累计含两次 Robot summary：

$$D^{(0)}=X_{\mathrm{tokenizer}}+e_{\mathrm{type}}+e_{\mathrm{task}}+2\bar R+W_Zz.$$

这不是为了叙述方便而抽象出来的等价写法，而是当前实现的实际加法路径。

8层 Decoder 输出记为：

$$D=\operatorname{Decoder}(D^{(0)}).$$

State 位置与 Action 位置分别切片得到 $D_t^S$ 和 $D_t^A$。

### 4.4 主 State reconstruction head

对每个 State hidden $D_t^S$，复制到29个关节并与对应 joint memory 拼接：

$$Q_{t,j}^S=[D_t^S,R_j]\in\mathbb R^{512}.$$

经过 joint query decoder 后，每个关节输出两个量：

$$[\hat{\tilde q}_{t,j}^{\mathrm{can}},\hat{\tilde{\dot q}}_{t,j}]=W_S^{\mathrm{out}}\phi_S^{\mathrm{dec}}(Q_{t,j}^S).$$

另一个线性层输出10个连续 base 量：3维 linear velocity、3维 angular velocity、3维 gravity 和1维 height。gravity 的三个输出被显式单位化：

$$\hat\gamma_t^R=\frac{y_t^\gamma}{\max(\lVert y_t^\gamma\rVert_2,\varepsilon)},\qquad \varepsilon=10^{-6}.$$

接触 head 输出 logits $\ell_t^c\in\mathbb R^2$，ModelOutput 中的 State 接触分量是概率：

$$\hat c_t=\operatorname{sigmoid}(\ell_t^c).$$

最终：

$$\hat{\tilde S}_t=\left[\hat{\tilde q}_t^{\mathrm{can}},\hat{\tilde{\dot q}}_t,\hat{\tilde v}_t^R,\hat{\tilde\omega}_t^R,\hat\gamma_t^R,\hat{\tilde h}_t,\hat c_t\right].$$

### 4.5 主 Action reconstruction head

对每个 Action hidden $D_t^A$ 和每个 joint memory：

$$Q_{t,j}^A=[D_t^A,R_j].$$

每个关节独立输出一个标准化 Action：

$$\hat{\tilde A}_{t,j}=W_A^{\mathrm{out}}\phi_A^{\mathrm{dec}}(Q_{t,j}^A).$$

该 head 用于 arbitrary Action completion。它受到全局 latent $z$ 影响，所以可通过 prior sampling 生成多个候选 Action 窗口。

### 4.6 Local relation condition

专用 forward、inverse、history head 不直接使用全局 sampled latent。它们从允许读取的局部关系构造另一个 target-safe latent mean。

对局部关系向量 $r_t$：

$$[\mu_t^{\mathrm{rel}},\log(\sigma_t^{\mathrm{rel}})^2]=W_{\mathrm{rel}}r_t+b_{\mathrm{rel}}.$$

当前实现只使用 mean，不对局部 latent 采样：

$$C_t^{\mathrm{rel}}=\bar R+W_Z\mu_t^{\mathrm{rel}}.$$

因此专用 relation heads 在 eval 模式下对给定输入是确定性的；改变 global CVAE 的随机样本不会改变这些专用 head 的结果。

### 4.7 Forward dynamics head

从 visible Encoder 中取 $S_t$ 和 $A_t$ 的 hidden，形成：

$$r_t^F=\phi_F([H_t^S,H_t^A]).$$

$$h_t^F=r_t^F+C_t^{\mathrm{rel}}.$$

连续68维以 residual dynamics 形式预测：

$$\Delta\hat{\tilde S}_{t,0:68}=W_F^{\mathrm{cont}}h_t^F+b_F^{\mathrm{cont}}.$$

$$\hat{\tilde S}_{t+1,0:68}=\tilde S_{t,0:68}+\Delta\hat{\tilde S}_{t,0:68}.$$

更新后的 gravity 再次单位化。接触不做普通 delta 回归，而是单独预测 logits：

$$\ell_{t+1}^{F,c}=W_F^ch_t^F+b_F^c.$$

$$\hat c_{t+1}=\operatorname{sigmoid}(\ell_{t+1}^{F,c}).$$

ModelOutput.forward_delta 的最后2维为：

$$\Delta\hat c_t=\hat c_{t+1}-c_t.$$

所以对整个70维做 $\tilde S_t+\Delta\hat{\tilde S}_t$ 时会得到 next-State 接触概率。

forward relation 只读取可见的 $S_t,A_t$ 表示。Mask 生成器还会强制保证被监督 transition 的 $S_t$ 和 $A_t$ 不被 overlay mask。

### 4.8 Inverse Action head

inverse head 使用 mask 后的直接 State embedding，而不是 Decoder 的目标 token：

$$r_t^I=\phi_I([X_t^S,X_{t+1}^S]).$$

$$h_t^I=r_t^I+C_t^{\mathrm{rel}}.$$

$$\hat{\tilde A}_t^{I}=\operatorname{ActionDecoder}(h_t^I,R).$$

Mask 生成器保证 inverse transition 两端的 $S_t,S_{t+1}$ 可见，同时遮挡对应 $A_t$。因此该 head 实现的是：

$$S_t,S_{t+1}\longrightarrow \hat A_t^I.$$

### 4.9 History-conditioned Action head

history head 从 causal Encoder 的 State hidden 构造：

$$r_t^H=\phi_H(H_t^S).$$

$$\hat{\tilde A}_t^H=\operatorname{ActionDecoder}(r_t^H+C_t^{\mathrm{rel}},R).$$

由于 token 顺序和 causal mask，它只能使用：

$$S_{\leq t},A_{<t}\longrightarrow \hat A_t^H.$$

未来 State 会被显式隐藏，而且这些隐藏仅用于防泄漏，不加入 State loss。现有测试会人为大幅修改不可见未来 State，并验证 history Action 输出保持不变。

### 4.10 Rollout、cycle 与 auxiliary head

模型内 rollout 最多8步。给定起点 $\hat S_s=S_s$ 和可见 Action：

$$\hat S_{s+k+1}=F_\theta(\hat S_{s+k},A_{s+k},R,C_{s+k}^{\mathrm{rel}}),\qquad k=0,\ldots,H-1.$$

其中 $H\in\{2,4,8\}$，或者验证时固定为8。每一步的 relation condition 由 masked visible Encoder 预先计算；State 本身使用上一步预测递推。

inverse-to-forward cycle 为：

$$\hat A_t^I=I_\theta(S_t,S_{t+1},R),\qquad \hat S_{t+1}^{I\rightarrow F}=F_\theta(S_t,\hat A_t^I,R).$$

forward-to-inverse cycle 为：

$$\hat S_{t+1}^F=F_\theta(S_t,A_t,R),\qquad \hat A_t^{F\rightarrow I}=I_\theta(S_t,\hat S_{t+1}^F,R).$$

辅助 head 直接读取 forward transition representation：

$$\hat{\tilde Y}_t^{\mathrm{aux}}=W_{\mathrm{aux}}h_t^F+b_{\mathrm{aux}}.$$

### 4.11 ModelOutput 总览

| 输出 | Shape | 主要用途 |
|---|---:|---|
| physical_state | $[B,129,70]$ | masked State reconstruction |
| action | $[B,128,29]$ | masked Action reconstruction |
| forward_delta | $[B,128,70]$ | one-step forward 与 rollout |
| inverse_action | $[B,128,29]$ | inverse dynamics |
| history_action | $[B,128,29]$ | causal Action prediction |
| auxiliary_transition | $[B,128,35]$ | torque/impulse supervision |
| posterior_mean/logvar | $[B,96]$ | full posterior |
| prior_mean/logvar | $[B,96]$ | visible conditional prior |
| latent | $[B,96]$ | 本次全局 latent |
| state_contact_logits | $[B,129,2]$ | reconstruction contact BCE |
| forward_contact_logits | $[B,128,2]$ | forward contact BCE |
| rollout_state | $[B,0,70]$ 或 $[B,8,70]$ | autoregressive rollout |
| cycle_state | $[B,128,70]$ 或 None | inverse-to-forward cycle |
| cycle_action | $[B,128,29]$ 或 None | forward-to-inverse cycle |

---

## 5. 损失函数：从单项定义到总目标

### 5.1 Masked Huber loss

连续量主要使用 Huber loss，阈值固定为 $\delta=1$。对标量误差 $e=\hat x-x$：

$$\operatorname{Huber}(e)=\begin{cases}\frac{1}{2}e^2,&|e|\leq1,\\|e|-\frac{1}{2},&|e|>1.\end{cases}$$

对布尔 mask $M$：

$$\mathcal L_{\mathrm{Huber}}(\hat X,X;M)=\frac{\sum_iM_i\operatorname{Huber}(\hat X_i-X_i)}{\max(\sum_iM_i,1)}.$$

代码只在 mask 非空时计算；空目标返回与计算图相连的零值。

### 5.2 Contact BCE

接触目标使用 logits 上的 binary cross entropy：

$$\operatorname{BCEWithLogits}(\ell,c)=-c\log\sigma(\ell)-(1-c)\log(1-\sigma(\ell)).$$

rollout/cycle 路径只有接触概率而没有 logits，因此在 FP32 中对截断后的概率计算等价 Bernoulli negative log-likelihood：

$$\mathcal L_c=-c\log\hat c-(1-c)\log(1-\hat c).$$

### 5.3 七类语义平衡 State loss

State 被划分为七个语义组：

$$\mathcal G_S=\{q,\dot q,v^R,\omega^R,\gamma^R,h,c\}.$$

对一个 step-level State 目标，先在每组内部平均，再在实际存在的语义组之间等权平均：

$$\mathcal L_S=\frac{1}{|\mathcal G_{\mathrm{active}}|}\sum_{g\in\mathcal G_{\mathrm{active}}}\mathcal L_g.$$

前六个连续组使用 Huber，contact 使用 BCE。这个设计避免29维 joint position 和29维 joint velocity 仅凭维数支配3维 base motion、1维 height 与2维 contact。

### 5.4 Masked reconstruction loss

若一个 arbitrary 样本只遮挡 State：

$$\mathcal L_{\mathrm{masked}}=\mathcal L_S.$$

若只遮挡 Action：

$$\mathcal L_{\mathrm{masked}}=\mathcal L_A.$$

若 State 与 Action 同时被遮挡：

$$\mathcal L_{\mathrm{masked}}=\frac{\mathcal L_S+w_A^{\mathrm{mask}}\mathcal L_A}{1+w_A^{\mathrm{mask}}}.$$

parent 中 $w_A^{\mathrm{mask}}=1$。Action fine-tune 中 $w_A^{\mathrm{mask}}=1.25$，但它只在 State 与 Action 同时有目标时提高 Action 的相对权重；Action-only 样本仍然就是 $\mathcal L_A$。

### 5.5 Forward、inverse 与 history loss

对 forward transition 集合 $\Omega_F$：

$$\mathcal L_F=\frac{1}{|\Omega_F|}\sum_{t\in\Omega_F}\mathcal L_S(\hat S_{t+1}^F,S_{t+1}).$$

其中连续量通过预测 delta 得到 next State，contact 使用 forward contact logits。

inverse Action loss 为：

$$\mathcal L_I=\mathcal L_{\mathrm{Huber}}(\hat A^I,A;M^I).$$

history Action loss为：

$$\mathcal L_H=\mathcal L_{\mathrm{Huber}}(\hat A^H,A;M^H).$$

在 inverse/history 任务中，主 reconstruction Action head 还会受到 masked Action loss；专用 relation head 则分别受到 $\mathcal L_I$ 或 $\mathcal L_H$。这是两个不同输出头的联合监督，不是同一个张量被无意重复计算。

### 5.6 Conditional KL 与 free bits

对第 $k$ 个 latent dimension，对角高斯 posterior 到 conditional prior 的 KL 为：

$$\operatorname{KL}_k=\frac{1}{2}\left[\log\frac{\sigma_{p,k}^2}{\sigma_{q,k}^2}+\frac{\sigma_{q,k}^2+(\mu_{q,k}-\mu_{p,k})^2}{\sigma_{p,k}^2}-1\right].$$

实现对每一维使用下限为 $\lambda_{\mathrm{free}}=0.05$ 的 free bits：

$$\mathcal L_{\mathrm{KL}}=\frac{1}{B}\sum_{b=1}^{B}\sum_{k=1}^{D_Z}\max(\operatorname{KL}_{b,k},\lambda_{\mathrm{free}}).$$

这意味着即使两个高斯非常接近，日志中的未加权 KL 也可能接近 $D_Z\lambda_{\mathrm{free}}=4.8$，而不是严格趋近零。真正进入总损失的是 $\beta_k\mathcal L_{\mathrm{KL}}$。

### 5.7 Gravity、auxiliary、rollout 与 cycle

对 masked gravity step，单位模约束为：

$$\mathcal L_\gamma=\frac{1}{N_\gamma}\sum_t\left(\lVert\hat\gamma_t^R\rVert_2-1\right)^2.$$

辅助监督为：

$$\mathcal L_{\mathrm{aux}}=\mathcal L_{\mathrm{Huber}}(\hat Y^{\mathrm{aux}},Y^{\mathrm{aux}};M^F).$$

rollout loss 对每个有效预测步使用同一个七组平衡 State loss：

$$\mathcal L_{\mathrm{roll}}=\frac{1}{N_{\mathrm{roll}}}\sum_{k=1}^{H}\mathcal L_S(\hat S_{s+k},S_{s+k}).$$

cycle loss是两个可用方向的平均：

$$\mathcal L_{\mathrm{cycle}}=\operatorname{mean}\left(\mathcal L_S(\hat S^{I\rightarrow F},S_{\mathrm{next}}),\mathcal L_{\mathrm{Huber}}(\hat A^{F\rightarrow I},A)\right).$$

只有当前 mask 产生相应 transition 时，该项才非零。

### 5.8 总损失

统一写法为：

$$\mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{masked}}+w_F\mathcal L_F+w_I\mathcal L_I+w_H\mathcal L_H+w_R\mathcal L_{\mathrm{roll}}+w_C\mathcal L_{\mathrm{cycle}}+\beta_k\mathcal L_{\mathrm{KL}}+w_\gamma\mathcal L_\gamma+w_{\mathrm{aux}}\mathcal L_{\mathrm{aux}}.$$

parent 权重为：

| 权重 | 数值 |
|---|---:|
| $w_A^{\mathrm{mask}}$ | 1.0 |
| $w_F$ | 1.5 |
| $w_I$ | 1.0 |
| $w_H$ | 1.0 |
| $w_R$ | 1.0 |
| $w_C$ | 0.1 |
| $w_\gamma$ | 0.1 |
| $w_{\mathrm{aux}}$ | 0.1 |

Action fine-tune 只改变：

$$w_A^{\mathrm{mask}}=1.25,\qquad w_F=1.25,\qquad w_I=1.5.$$

其余权重不变。

---

## 6. 训练、迭代更新、验证与 checkpoint

### 6.1 Parent 模型配置与参数量

正式 hidden-context 模型配置是：

| 配置 | 数值 |
|---|---:|
| d_model | 384 |
| Encoder/Decoder layers | 6 / 8 |
| heads | 8 |
| FFN width | 1536 |
| latent dim | 96 |
| joint width | 128 |
| dropout | 0.1 |

按当前 H2 manifest 预期的两个 actuator vocabulary 项计算，总参数量为：

$$N_\theta=28,475,448.$$

actuator type 数量记为 $K_u$ 时，更一般的 hidden 模型参数公式是：

$$N_\theta=28,475,192+128K_u.$$

当前参数主要分布为：

| 子系统 | 参数量 |
|---|---:|
| 6层 Temporal Encoder | 10,647,552 |
| 8层 Temporal Decoder | 14,196,480 |
| Robot/joint encoder | 619,776 |
| State/Action tokenizer 与融合 | 1,360,896 |
| 其余 latent、relation 与输出 heads | 1,650,744 |

### 6.2 完成/失败 episode 的采样

PhysicsTransformer 训练保留 completed 与 failed canonical episode。如果两类同时存在，WeightedRandomSampler 将总采样质量分配为：

$$P(\mathrm{completed})=0.90,\qquad P(\mathrm{failed})=0.10.$$

每一类内部再按该类窗口数均分。这样模型能看到少量失败动力学，又不会让失败 episode 主导训练。

### 6.3 一个 optimizer step 的完整过程

parent 正式训练使用 micro batch 8、gradient accumulation 8：

$$B_{\mathrm{effective}}=8\times8=64.$$

设 optimizer step 为 $k$。MaskGenerator 首先接收当前 local step，生成该 micro batch 的任务与 mask。KL 权重线性 warmup：

$$\beta_k=\beta_{\max}\min\left(\frac{k}{K_{\mathrm{KL}}},1\right).$$

parent 使用 $\beta_{\max}=0.001,K_{\mathrm{KL}}=30000$；fine-tune 使用 $\beta_{\max}=0.001,K_{\mathrm{KL}}=10000$。

每个 micro batch 执行：

$$\mathrm{batch}\rightarrow\mathrm{mask}\rightarrow\mathrm{model\ forward}\rightarrow\mathcal L_{\mathrm{total}}\rightarrow\frac{\mathcal L_{\mathrm{total}}}{N_{\mathrm{acc}}}\rightarrow\mathrm{backward}.$$

累积后的梯度可以写为：

$$g_k=\frac{1}{N_{\mathrm{acc}}}\sum_{n=1}^{N_{\mathrm{acc}}}\nabla_\theta\mathcal L_n.$$

在 optimizer 更新前，对所有参数做 global norm clipping，阈值 $c=1$：

$$g_k^{\mathrm{clip}}=g_k\min\left(1,\frac{c}{\lVert g_k\rVert_2}\right).$$

然后依次执行 optimizer step、清空梯度和 scheduler step。每一步都会记录总损失、各子损失、学习率和裁剪前 gradient norm；非有限损失会立即终止训练。

### 6.4 AdamW 更新

当前代码没有覆盖 AdamW 的 $\beta_1,\beta_2,\epsilon$，所以使用 PyTorch 默认值：

$$\beta_1=0.9,\qquad\beta_2=0.999,\qquad\epsilon=10^{-8}.$$

给定裁剪后梯度 $g_k^{\mathrm{clip}}$：

$$m_k=\beta_1m_{k-1}+(1-\beta_1)g_k^{\mathrm{clip}}.$$

$$v_k=\beta_2v_{k-1}+(1-\beta_2)(g_k^{\mathrm{clip}})^2.$$

偏差修正为：

$$\hat m_k=\frac{m_k}{1-\beta_1^k},\qquad\hat v_k=\frac{v_k}{1-\beta_2^k}.$$

以 decoupled weight decay $\lambda=10^{-4}$ 表示一次概念化 AdamW 更新：

$$\theta_{k+1}=(1-\eta_k\lambda)\theta_k-\eta_k\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}.$$

这里所有乘方、平方根和除法对参数逐元素执行。

### 6.5 Warmup + cosine 学习率

设最大 optimizer step 为 $K$，warmup step 为 $K_w$，参数组基础学习率为 $\eta_0$。调度倍率为：

$$\alpha(k)=\begin{cases}\dfrac{\max(k,1)}{\max(K_w,1)},&k<K_w,\\\dfrac{1}{2}\left[1+\cos\left(\pi\min\left(\dfrac{k-K_w}{K-K_w},1\right)\right)\right],&k\geq K_w.\end{cases}$$

实际学习率为：

$$\eta_k=\eta_0\alpha(k).$$

parent 使用 $\eta_0=2\times10^{-4}$、$K_w=8000$、$K=120000$。

### 6.6 BF16 与 GradScaler

当 device 是 CUDA 且配置为 BF16 时，forward 与 loss 在 torch.autocast 下执行。代码只在 amp 等于 FP16 时启用 GradScaler，因此当前 BF16 配置满足：

$$\mathrm{GradScalerEnabled}=\mathrm{False}.$$

也就是说，BF16 使用自动混合精度，但不做动态 loss scaling。contact 概率形式的 BCE 会显式转到 FP32，以避免 CUDA autocast 对概率 BCE 的限制和数值问题。

### 6.7 Parent Mask curriculum

parent 的 forward 子任务比例为 one-step、rollout、cold：

$$P(F_1,F_{\mathrm{roll}},F_{\mathrm{cold}})=(0.25,0.50,0.25).$$

在 optimizer step 20000 之前，随机 forward 任务强制使用 one-step；较长 step mask 也限制在最多8步。20000之后允许 rollout/cold 和最长32步 mask，并从可行的 $\{2,4,8\}$ 中选择 rollout horizon。

### 6.8 Action fine-tune 初始化与参数组

fine-tune 只接受 checkpoint format v2，并严格核对 dataset manifest SHA256、模型结构配置和每个参数 Shape。加载方式是 weights-only：

$$\theta_{\mathrm{fine},0}\leftarrow\theta_{\mathrm{parent,best}}.$$

parent optimizer、scheduler、AMP scaler 与 RNG 均不恢复，而是重新初始化。

所有参数仍然可训练，没有冻结。它们被互斥划分为三组：

| 参数组 | 参数量 | 基础学习率 |
|---|---:|---:|
| Action embeddings/decoder/inverse/history | 1,200,001 | $5\times10^{-5}$ |
| shared robot/Transformer/CVAE 等 | 26,020,800 | $2\times10^{-5}$ |
| State/forward/auxiliary heads | 1,254,647 | $1\times10^{-5}$ |

三组之和满足：

$$1,200,001+26,020,800+1,254,647=28,475,448.$$

fine-tune 使用40,000 local optimizer step、warmup 2,000、micro batch 8、accumulation 8、BF16。

### 6.9 Action Mask curriculum

Action span curriculum 为：

| local step | 最大 Action span |
|---:|---:|
| 1–10,000 | 32 |
| 10,001–25,000 | 64 |
| 25,001–40,000 | 128 |

可用长度 bucket 为 1–8、9–32、33–64、65–128，基础采样权重为 $(0.20,0.30,0.25,0.25)$，只在当前最大长度允许的 bucket 间重新归一化。

从 step 25,000 开始，inverse 任务在有效窗口长度至少128时以额外 0.15 概率精确遮挡完整128步：

$$P(L_{\mathrm{inverse}}=128\mid k\geq25000)=0.15.$$

### 6.10 Parent validation 与模型选择

parent 固定验证包含 one-step forward、8步 rollout、inverse、history Action 与 arbitrary completion。选择分数为：

$$\mathrm{Score}_{\mathrm{parent}}=0.20R_{F1}+0.20R_{F8}+0.20R_I+0.15R_H+0.25R_{\mathrm{arb}}.$$

其中 $R$ 都是标准化空间 RMSE，越低越好。每2,000 step 验证一次，验证前使用固定 seed 和固定任务循环，使不同 checkpoint 面对相同分布的 mask。

若 Score 创新低，则原子写入 best.pt；每次验证都写 last.pt。连续15次验证无改善时 early stop。checkpoint 保存模型、optimizer、scheduler、scaler、配置、数据 hash、参数量和 Python/NumPy/Torch/CUDA RNG 状态。

### 6.11 Fine-tune validation、State guard 与质量门禁

fine-tune 的 Action 选择分数为：

$$\mathrm{Score}_{A}=0.30R_{I,\mathrm{local}}+0.25R_{I,128}+0.25R_{\mathrm{completion}}+0.10R_H+0.10R_{\mathrm{arbA}}.$$

Action completion macro 是 element、step-32、feature 和 semantic 四类 RMSE 的等权平均：

$$R_{\mathrm{completion}}=\frac{R_{\mathrm{element}}+R_{\mathrm{step32}}+R_{\mathrm{feature}}+R_{\mathrm{semantic}}}{4}.$$

fine-tune 开始前，使用相同固定 validation suite 计算 parent baseline。候选 checkpoint 的四个 State guard ratio 为：

$$G_j=\frac{R_{j,\mathrm{fine}}}{\max(R_{j,\mathrm{parent}},10^{-12})}.$$

四项分别是 one-step forward、rollout-8、arbitrary State、State step-32，必须全部满足：

$$G_j\leq1.05.$$

Score 更低但 State guard 失败的模型只能保存为 best_unguarded.pt；只有 guard 全部通过的模型才能成为正式 best.pt。

正式40k训练结束后，还要求相对 parent 的改善率：

$$I_m=1-\frac{R_{m,\mathrm{fine}}}{R_{m,\mathrm{parent}}}.$$

门槛是 inverse local 至少10%、inverse full-128 至少15%、Action completion macro 至少10%。任一失败时保留诊断产物，但不生成 cvae_action_finetune.ok。

### 6.12 已验证 parent 结果与 fine-tune 状态

已完成 parent run：

- 数据：/home/helloworld/bly/runs/cvae_physics_dataset_20260825_235244
- 训练：/home/helloworld/bly/runs/cvae_train_20260826_002252
- checkpoint：/home/helloworld/bly/runs/cvae_train_20260826_002252/checkpoints/best.pt
- 完成 optimizer step：120000
- 最佳 validation selection score：0.3173407225683331

历史 test 结果如下。该 test split 已经使用过，必须标为 **reused test**，不能称为全新 blind test。

| 指标 | 结果 |
|---|---:|
| one-step forward normalized RMSE | 0.2634 |
| rollout-8 normalized RMSE | 0.4603 |
| forward joint position RMSE | 0.0179 rad |
| forward joint velocity RMSE | 0.2775 rad/s |
| inverse Action RMSE | 0.1321 rad |
| history Action RMSE | 0.1695 rad |

负对照也已通过：打乱 Action 后 forward RMSE 恶化28.3%；打乱 $S_{t+1}$ 后 inverse RMSE 恶化152.3%；修改 history 任务中不可见的未来 State，对 history Action 的最大影响为0。

截至本文对应工作区状态，Action fine-tune 只有提交 197a173 中的代码与配置，没有可验证的 Ubuntu success marker、training summary 或正式结果，因此本文不报告 fine-tune 指标。

---

## 7. 推理、输出后处理、视频生成与理解边界

### 7.1 三种 Action 输出不要混用

普通 Action completion 使用主 Decoder 的 action 输出，它受到全局 CVAE latent 影响，可采用 prior mean 或多次 prior sample。

inverse_full_128 使用专用 inverse_action：

$$\hat A_t^I=I_\theta(S_t,S_{t+1},R).$$

history Action 使用 history_action：

$$\hat A_t^H=H_\theta(S_{\leq t},A_{<t},R).$$

三个输出虽然 Shape 都是 $[B,T,29]$，条件集合和训练目标不同，评测时不能互换。

### 7.2 从 masked sequence 做普通补全

普通补全流程是：

1. 将物理 State/Action 使用训练统计量标准化。
2. 构造 $M^S,M^A$，mask 外保持输入值，mask 内置零并输入 mask bit。
3. 用 visible conditional prior 得到 $\mu_p,\log\sigma_p^2$。
4. prior_mean 模式取 $z=\mu_p$；随机模式从 prior 采样多个 $z$。
5. Decoder 输出完整 $\hat{\tilde S},\hat{\tilde A}$。
6. 只把 mask 内预测写回，mask 外值必须逐元素保持原样。
7. 反归一化得到物理 State/Action。

如果生成 $N$ 个 latent 候选，oracle best-of-$N$ 只能用于离线分析，因为它需要真实 masked target 才能选最优样本；部署不能使用 ground truth 做选择。

### 7.3 给定初始 State 与 Action 序列预测未来 State

目标输入是初始 State $S_0$、Action 计划 $A_{0:H-1}$、RobotInfo 和窗口前 Action。先标准化，再隐藏 $S_{1:H}$，保持 $S_0$ 与全部输入 Action 可见。

模型内最大 rollout 为8：

$$\hat S_{t+1}=F_\theta(\hat S_t,A_t,R),\qquad t=0,\ldots,7.$$

1、2、4、8步属于训练实现覆盖的 rollout horizon。32步视频通过外部 segmented rollout 将四段8步首尾连接：

$$\hat S_{1:32}=\operatorname{Chain}\left(F_{1:8},F_{9:16},F_{17:24},F_{25:32}\right).$$

每一段完成后，将预测 State 写回 working sequence，再重新调用模型。32步结果明确标记为 OOD，因为训练时模型内 rollout 不超过8步。

### 7.4 State 反归一化与关节姿态

模型输出先反归一化：

$$\hat S_{t,i}^{\mathrm{phys}}=\hat{\tilde S}_{t,i}\sigma_i^S+\mu_i^S.$$

用于渲染的绝对 joint position 为：

$$\hat q_{t,j}^{\mathrm{abs}}=\hat q_{t,j}^{\mathrm{can}}+q_j^{\mathrm{nom}}.$$

需要注意：State joint position 可直接确定关节姿态，但70维 State 没有完整 root pose，因此还要进行 root trajectory reconstruction。

### 7.5 Root orientation 重建

给定 anchor 时刻的真实 root quaternion $Q_a$。对每一步，使用当前 State 的 robot-frame angular velocity $\omega_t^R$ 构造 axis-angle 增量：

$$\alpha_t=\lVert\omega_t^R\rVert_2\Delta t.$$

$$\Delta Q_t=\left[\cos\frac{\alpha_t}{2},\frac{\omega_t^R}{\lVert\omega_t^R\rVert_2}\sin\frac{\alpha_t}{2}\right].$$

$$Q_{t+1}^{\mathrm{int}}=Q_t\otimes\Delta Q_t.$$

然后使用下一时刻预测 gravity 恢复 roll 和 pitch，同时保留积分 quaternion 的 yaw：

$$\mathrm{pitch}_{t+1}=\arcsin(\operatorname{clip}(\gamma_{t+1,x}^R,-1,1)).$$

$$\mathrm{roll}_{t+1}=\operatorname{atan2}(-\gamma_{t+1,y}^R,-\gamma_{t+1,z}^R).$$

$$Q_{t+1}=\operatorname{QuatFromEuler}\left(\mathrm{roll}_{t+1},\mathrm{pitch}_{t+1},\operatorname{yaw}(Q_{t+1}^{\mathrm{int}})\right).$$

最后选择与上一 quaternion 点积为正的符号，避免 $Q$ 与 $-Q$ 的等价表示导致视频跳变。

### 7.6 Root position 重建

将 robot-frame linear velocity 用当前 quaternion 旋转到世界坐标：

$$v_t^W=\operatorname{Rotate}(Q_t,v_t^R).$$

世界平面位置用显式 Euler 积分：

$$p_{t+1,xy}^W=p_{t,xy}^W+v_{t,xy}^W\Delta t.$$

高度不积分垂直速度，而是使用 State 中的 base height 相对 anchor 校正：

$$p_{t+1,z}^W=p_{a,z}^W+h_{t+1}-h_a.$$

这说明视频轨迹是由 State 运动学重建得到的，不是模型直接预测的绝对 root trajectory。真实 State 也会经过同一重建器，作为重建误差下限；记录的真实 root 轨迹则作为最左侧 source。

### 7.7 三联视频的含义

State 视频的三栏分别是：

| 面板 | 内容 |
|---|---|
| Recorded source | HDF5 中记录的真实 root pose 与真实 joint pose |
| Truth State reconstruction | 用真实70维 State 经过上述积分/重建得到 |
| Predicted State reconstruction | 用模型补全/rollout State 经过相同重建得到 |

因此“Truth State reconstruction 与 Recorded source 的差异”是70维表示和积分器本身造成的 reconstruction floor；“Predicted 与 Truth reconstruction 的差异”才更接近模型 State 误差。

### 7.8 Action 反变换与 Isaac replay

预测 Action 先从标准化空间还原 canonical Action：

$$\hat A_t=\hat{\tilde A}_t\sigma_A+\mu_A.$$

再恢复 processed absolute target：

$$\hat q_t^{\mathrm{target,abs}}=\hat A_t+q^{\mathrm{nom}}.$$

最后按照采集 schema 的 scale、offset 和 clip 反解/映射为 raw Action，并在 Isaac 中执行。mask 外 Action 必须位级保持原轨迹，只有 mask 内元素允许被模型替换。

物理 replay 成功只说明数据映射、初始条件和执行链路正确；它不自动意味着 CVAE 优于 hold-last 或线性插值。模型质量门禁与 replay pipeline 门禁必须分开报告。

### 7.9 当前架构的正确理解

1. **它是序列条件模型，不是完整 policy。** 缺少 goal/reference 意味着 Action 多解性无法完全消除。
2. **global latent 主要服务主 reconstruction Decoder。** 专用 forward/inverse/history heads 使用 local relation mean，不因 global latent sampling 产生多样性。
3. **完整 Action 计划可以作为条件。** forward 场景允许所有输入 Action 可见，这符合“给定 $S_0$ 和 Action 序列预测 State”的验收目标。
4. **模型输出默认是标准化量。** 只有反归一化结果才具有物理单位。
5. **8步之后误差会累积。** 32步 chained rollout 是有价值的压力测试，但必须标记 OOD。
6. **参数量增加不保证物理效果自动提高。** 必须结合 normalized/physical RMSE、负对照、State guard、Action replay 和视频共同判断。

---

## 源码定位

| 内容 | 源码 |
|---|---|
| PhysicsTransformer、RoPE、Encoder/Decoder、各输出头 | [src/cvae_sa/models.py](src/cvae_sa/models.py) |
| State/Action/RobotInfo 读取、归一化、Action-energy crop | [src/cvae_sa/dataset.py](src/cvae_sa/dataset.py) |
| 70维 State 与结构化 RobotInfo schema | [src/cvae_sa/physics_schema.py](src/cvae_sa/physics_schema.py) |
| 任务概率、Mask、curriculum、因果性 | [src/cvae_sa/masking.py](src/cvae_sa/masking.py) |
| Huber/BCE/KL/rollout/cycle 总损失 | [src/cvae_sa/losses.py](src/cvae_sa/losses.py) |
| AdamW、调度、梯度累积、validation、checkpoint、fine-tune | [src/cvae_sa/trainer.py](src/cvae_sa/trainer.py) |
| State rollout、root 重建与视频数据准备 | [src/cvae_sa/state_mask_eval.py](src/cvae_sa/state_mask_eval.py) |

