# State–Action CVAE 实验通俗说明

最后更新：2026-09-08

这份说明面向大致了解神经网络或CVAE、但不熟悉本项目细节的读者。它解释我们想解决什么问题、数据长什么样、各代模型如何工作、每轮实验为什么做，以及目前究竟得出了什么结论。

精确指标、正式run路径和当前命令见 [plan.md](plan.md)；下一阶段的严格执行合同见 [Next.md](Next.md)。

## 1. 一句话理解这个项目

我们有一段机器人运动记录，其中交替保存了机器人的身体状态和发给关节的动作命令。我们希望把其中任意一段“挖掉”，再让模型根据剩余内容把缺失部分补回来。

可以把它想象成一段同时带有画面和控制指令的录像：

```text
身体状态 S0 → 动作 A0 → 身体状态 S1 → 动作 A1 → …… → 身体状态 ST
```

可能的任务包括：

- 已知动作，补出机器人随后如何运动。
- 已知前后身体状态，反推出中间用了什么动作。
- 同时缺少一小段状态和动作，根据前后内容补齐。
- 已知完整状态序列，补出整段动作。

最终希望得到的是一个真正的条件生成模型：它只看未被遮挡的内容，就能生成一个或多个合理的缺失片段。

但当前实验还没有走到最终阶段。我们正在先回答更基础的问题：当模型的posterior encoder可以偷看完整答案时，它是否至少有能力把这批训练轨迹记住并准确解码？如果连这一点都做不到，过早加入随机采样和KL只会让问题更难定位。

## 2. 三个容易混淆的能力层次

### 2.1 第一层：记住训练轨迹

模型在编码时可以读取完整序列，包括随后会被Mask的真值。它把整段序列压缩成latent，再让decoder还原缺失部分。

这像考试前允许学生看完整答案，并让他把答案压缩写到一张便签上。我们测试的是：这张便签和解题器是否足以把答案重新写出来。

当前绝大多数实验属于这一层，通常称为posterior reconstruction或posterior记忆实验。

### 2.2 第二层：只根据剩余内容推理

部署时不能再读取被遮挡真值。模型只能看剩下的State、Action和Mask，推测缺失内容。

这才对应conditional prior：根据可见条件产生latent，再由decoder补全。它比第一层难，因为输入中可能不存在唯一答案。

当前H38-R通过以前，我们不会实现或评价这一层。因此，posterior实验通过不能写成“模型已经学会State到Action推理”。

### 2.3 第三层：生成多种合理结果

有些缺失片段本来就不只有一个正确答案。例如只知道一段身体状态，可能有多种动作都能产生相近运动。CVAE希望通过随机采样生成这些不同但合理的可能性。

这一层需要合理的概率分布、方差和KL约束。当前实验把KL设为0，也不进行latent随机采样，所以目前不讨论生成多样性。

## 3. 数据到底包含什么

### 3.1 序列的严格时间关系

每个episode保存：

```text
S0, A0, S1, A1, ..., A(T-1), ST
```

这里：

- `S_t`是执行动作前的机器人状态。
- `A_t`是这一时刻发给29个关节的目标命令。
- `S_(t+1)`是执行该动作后的下一个状态。
- 所以严格关系是`S_t + A_t → S_(t+1)`。

如果窗口长度是T64，就有64个Action和65个State；模型内部交替处理129个State/Action token。T128则有128个Action和129个State。

数据频率为50 Hz，每一步约0.02秒。因此：

| 窗口 | 覆盖时间 | 含义 |
|---|---:|---|
| T16 | 约0.32秒 | 很短的局部动作片段 |
| T64 | 约1.28秒 | 当前主要实验长度 |
| T128 | 约2.56秒 | 历史长序列容量实验 |

这里的“32 motion”表示32种运动轨迹，不是Action有32维。Action始终是29维。

### 3.2 70维State

State由以下内容组成：

| 内容 | 维数 | 通俗含义 |
|---|---:|---|
| 关节位置 | 29 | 29个关节相对默认姿势转了多少 |
| 关节速度 | 29 | 每个关节转动得多快 |
| 身体线速度 | 3 | 骨盆向前后、左右、上下移动的速度 |
| 身体角速度 | 3 | 骨盆绕三个方向旋转的速度 |
| 重力方向 | 3 | 身体当前朝向和倾斜情况 |
| 身体高度 | 1 | 骨盆离地面的高度 |
| 左右脚接触 | 2 | 左脚和右脚是否踩到地面 |

前68维是连续数值，最后2维是0/1接触标签。连续State使用MSE训练；脚接触使用二分类BCE训练。

### 3.3 29维Action

Action是29个关节的目标位置，相对统一默认姿势表示。它和State中的29个关节位置使用相同的关节顺序和单位。

这比直接保存神经网络的原始输出更接近机器人真正收到的控制目标，也方便把预测动作重新送回仿真器验证。

### 3.4 当前记忆数据集

当前小数据集从正式Physics数据中选择32个motion，每个motion保留8个启动variant，共256个episode。

这些variant可理解为“同一种动作从稍微不同的初始状态开始执行”。同一motion的8个variant必须成组保留，避免实验悄悄改变数据身份。

所有数据都用于训练和记忆检查，因此结果只描述已见数据，不能称为验证集泛化或测试集表现。

## 4. Mask是怎样工作的

Mask就是一张“哪些答案被盖住”的布尔表。每个State和Action位置都有对应Mask；被Mask的位置用于计算重建loss，没有被Mask的值作为条件留给模型。

### 4.1 早期任意Mask

最初为了纯粹检查记忆容量，我们使用10类固定Mask：完整遮挡State、完整遮挡Action、两者全遮挡，以及随机元素、时间块、特征组和语义组。

这些Mask不一定都能根据物理条件唯一推导。例如State和Action全部遮挡后，decoder只能依靠posterior latent恢复。这在记忆实验中是允许的，因为目标就是检查latent是否携带完整答案。

### 4.2 当前物理结构Mask

进入T64路线后，主门禁只使用更符合物理关系的Mask：

| 类型 | 遮挡什么 | 为什么条件相对合理 |
|---|---|---|
| State gap 4/16 | 连续4或16步State | 两侧State边界和对应Action仍可见 |
| State rollout | S1到S64 | S0和全部Action可用于向前递推 |
| Action gap 4/16 | 连续4或16步Action | 完整State展示了动作前后的身体变化 |
| Full Action | 全部Action | 完整State轨迹提供逆动力学线索 |
| Joint gap 2/8 | 短State和Action片段同时缺失 | 缺口前后State边界仍保留，可做双向补洞 |

H38-B在固定Mask上训练和评测；H38-R训练时动态改变缺口位置，并在未参与训练的Mask上重评同一批已见轨迹。

## 5. 最简posterior模型如何工作

早期模型可以画成：

```text
完整 State–Action 序列
        │
        ▼
双向 Transformer encoder
        │
        ▼
一个 256维 global latent
        │
        ├──────────────┐
        │              │
Mask后的序列 ───────► Transformer decoder
                       │
                       ▼
              State / contact / Action预测
```

Transformer可以让序列中每个位置参考其他位置。双向表示它既能看过去，也能看未来，适合补洞任务，但不代表严格因果预测。

global latent像一张描述整段录像的“压缩便签”。decoder既看到这张便签，也看到未被遮挡的原始内容，然后填写空白。

最简6.7M模型使用宽度256、4层encoder和4层decoder。25.45M模型扩大到宽度384、6层encoder和8层decoder，但仍只有一个256维global latent。

## 6. Posterior、prior、采样和KL

### 6.1 Posterior mean

posterior是“看过完整答案后得到的latent分布”。encoder通常输出一个均值`mu_q`和一个方差。

当前容量实验直接使用`z = mu_q`。这条路径没有随机噪声，最适合检验模型的确定性重建上限。

### 6.2 Posterior sample

真正采样时会使用：

```text
z = mu_q + sigma_q × epsilon
```

其中`epsilon`来自标准正态分布。它相当于在posterior均值附近随机取一点，用于检查模型是否能容忍和利用posterior方差。

当前训练尚未启用这条路径。

### 6.3 Conditional prior sample

部署时没有完整答案，只能把Mask后的序列送给condition encoder，得到`mu_p`和`sigma_p`，然后采样：

```text
z = mu_p + sigma_p × epsilon
```

它不是把一个完全无条件的随机数直接塞给decoder，而是先由可见条件决定采样分布。这才是本项目计划中的常规条件生成路径。

### 6.4 KL散度

KL可以理解成一根把posterior分布和conditional prior分布拉近的橡皮筋。

- 拉得太松：posterior重建很好，但prior找不到posterior所在位置，部署时生成失败。
- 拉得太紧：所有序列的latent都被挤得太像，decoder不再使用latent，出现posterior collapse。
- 合适的强度：posterior仍能准确重建，prior也能从可见条件找到相近分布。

当前`KL beta=0`，等于暂时拿掉这根橡皮筋。这样做是为了先确定模型结构本身有足够容量。只有H38-R通过后，才会比较posterior mean、posterior sample和conditional prior sample。

## 7. 我们怎样判断模型是否成功

### 7.1 RMSE

RMSE把所有目标位置的平方误差取平均再开根号。它对较大误差比较敏感，但仍是整体平均指标。

例如global RMSE很低，只能说明总体不错，不能保证每个窗口、每类Mask或每个时间点都准确。

### 7.2 worst-window和最大误差

worst-window RMSE检查表现最差的窗口，防止模型只把大多数简单样本拟合好。

max absolute error检查整个评测中最大的单点错误。它非常严格，也容易被一个特殊位置控制。

### 7.3 p99误差

p99 absolute error表示99%的连续输出误差都小于这个数。它忽略最极端的1%，比max更能描述模型是否普遍准确。

当前`fit`门禁同时要求global、每类最差窗口和p99达标，因此既看平均，也看分布尾部。

### 7.4 Contact accuracy

contact accuracy检查左右脚是否接触地面。要求100%，因为这是离散且对动作物理含义很重要的标签。

### 7.5 Latent依赖检查

仅有低RMSE还不够：decoder可能绕过latent，只根据可见条件或常量模板输出。

因此在完整遮挡时，我们会：

- 把latent置零。
- 换成另一个window的latent。
- 换成另一个motion的latent。

如果错误至少变差10倍，说明正确latent确实携带关键序列信息。这个检查证明“latent被使用”，但不证明latent组织得适合随机采样。

## 8. 实验如何一步步演进

### 8.1 D1：先验证一条窗口能否精确记忆

模型：6.7M最简Transformer，一个256维global latent。

数据：1个motion中的1个T16窗口，对同一窗口使用10类Mask。

结果：State RMSE约`4.72e-5`，Action RMSE约`8.08e-5`，最大误差约`2.26e-4`，contact 100%，通过历史exact门禁。

它证明：最简结构至少能把一个很小的序列和多种Mask近乎无损记住；模型、loss和decoder并非从根本上完全不可用。

它不能证明：多个窗口、长序列、新Mask、conditional prior或真实State↔Action推理。

### 8.2 F1R、W4和P1：增加窗口数量与训练暴露

F1R把数据扩到同一motion的144个T16窗口，40k后平均误差已较低，但最大误差约`0.0179`，没有通过旧exact标准。

W4只使用4个窗口，40k后非常接近exact。P1再把144个窗口训练到约93.5k step，最终通过较宽松的progression门禁。

这些结果支持：

- 模型可以记住不止一个窗口。
- 数据增多后，每个fixture被反复看到的次数明显影响拟合程度。
- progression PASS不是数值意义上的完美重建。

更早的F1因为训练和评测用了不同的partial Mask坐标，不能作为fixed记忆失败证据，只能看成一次新Mask查询诊断。

### 8.3 L128：一个motion的长序列可以达到推进精度

模型扩大到25.45M，但仍使用单个256维global latent。

数据为1个motion、24个T128窗口、240个固定fixtures。训练约93.5k step后，State/Action RMSE约`0.00230/0.00153`，最大误差约`0.00994`，通过progression门禁。

它证明：单global latent模型能够在一个motion规模上记住约2.56秒长的序列，并达到`1e-2`推进精度。

它仍不能证明扩展到更多motion后容量足够。

### 8.4 F128：直接扩到32 motion明显失败

数据扩为32 motion、816个T128窗口、8,160个fixtures，训练200k step。

最佳State/Action RMSE约`0.261/0.274`，最大误差约`9.64`，远超门禁。后期指标已经平台化，所以继续少量追加step没有明确依据。

它证明不了“所有更大模型都会失败”，但明确说明当前25.45M单global latent结构和训练方式无法直接从1 motion扩展到32 motion、T128。

### 8.5 F4D/F4A：失败不是少数离群点

为了缩小规模，F4D只训练4 motion、80个T128窗口。

global State/Action RMSE已经约`0.00915/0.00812`，看上去低于`1e-2`；但最差窗口仍为`0.0230/0.0153`，最大误差`0.2156`。

F4A随后只读分析每个元素，发现约21.08%的连续目标仍超过`1e-2`，而且全部800个fixtures都至少包含一个超阈值位置。

这支持：问题不是几个偶然坏点，而是长序列中大量细节没有被均匀还原。contact已经100%，所以脚接触分类也不是主要瓶颈。

### 8.6 F4B/F4C：改loss或重复注入一个global latent仍不够

F4B做了严格配对的A/B实验：

- A继续使用普通MSE。
- B让每个State/Action域中误差最大的20%获得更高权重。

B让最大误差从约`0.1372`略降到`0.1306`，但超阈值元素比例反而从18.46%升到20.72%，没有达到预先规定的显著改善。

F4C在decoder每层都加入同一个global latent，希望避免latent信息只在输入处注入一次。它的部分平均指标略好，但最大误差升到约`0.1719`，仍未通过。

这些实验支持：当前问题不能靠相似的tail loss或简单重复广播同一global latent稳定解决。因此停止继续搜索类似loss、gate和随机seed。

### 8.7 F4E：直接学习每个窗口的code

F4E暂时绕过posterior encoder，为80个window各学习一个256维code。同一window的10种Mask必须共享同一个code，所以code只能代表窗口，不能直接记住每种Mask答案。

第一阶段只训练code；第二阶段让code和decoder共同调整。最终global State/Action RMSE约`0.00895/0.00786`，但最差窗口约`0.0298/0.0205`，最大误差`0.2094`，仍未通过。

把code置零或换成其他窗口后，误差恶化约95到120倍，证明decoder强烈使用了code。

最准确的结论是：encoder不是唯一疑点；即使显式给每个窗口一个可学习身份code，单个256维code和当前decoder也没有在预算内通过。它尚不能证明这种结构理论上绝对不可能，只能说容量仍未获证明。

### 8.8 F4F：8个全局token与逐时间code的等预算对照

F4F给每个window约2,000个可学习code标量，两臂预算几乎相同：

| 结构 | 类比 | 结果 |
|---|---|---|
| G8 | 把一张总便签换成8张总便签 | global State/Action约`0.0182/0.0135`，max约`0.711` |
| T129 | 给每个时间位置一张很小的便签 | global State/Action约`0.0405/0.0253`，max约`2.91` |

G8明显优于T129，但两者都失败，且T129更差。这说明“只要改成逐时间code就会更好”的直觉没有得到支持。

它也说明单纯改变latent的放置方式仍未解决问题，所以后续不继续给F4F追加训练，而是先验证更基础的直接输出上限。

### 8.9 F4G：直接输出表上限

F4G完全移除encoder、latent和decoder。每个T64窗口直接拥有一份可学习的完整State、contact和Action输出表，同一window面对不同Mask仍共享同一份答案。

它相当于不让学生理解题目，而是直接给每道题准备一张完整答案表。若这种最直接方法仍过不了门禁，问题更可能位于loss计算、Mask target、优化设置或evaluator，而不是latent结构。

正式F4G随后在1,504个窗口、12,032种“窗口×Mask”组合上训练5,000步。global State/Action RMSE从约`0.973/0.992`持续降到`0.169/0.0798`，contact达到100%，说明索引、Mask、loss和梯度路径确实在工作；但最坏State RMSE仍为`2.208`，没有通过。

这里不能直接得出“答案表也记不住”。5,000步、每步256个fixture，相当于每个fixture平均只出现约106次。1,504张答案表彼此独立，某个窗口未被抽到时，它的参数完全不会更新；而归一化State里存在绝对值接近42的目标，从零开始只靠约百次更新很难走到答案附近。评测曲线从头到尾都在下降，也更符合“更新量不足”，而不是loss完全接错。

因此新增F4G-O解析上限：不再用梯度慢慢学习，而是把每个窗口真值直接复制进对应答案表，然后对全部12,032个fixtures重复评测三次。它像直接把标准答案印进答案卡：

- 如果仍失败，Mask、窗口身份或evaluator确实存在矛盾。
- 如果得到零连续误差并通过，说明解析上限可达，原F4G只暴露了稀疏查表的优化问题；随后可以进入H38。

F4G-O当前已实现但尚未在Ubuntu运行，所以现在仍不能启动H38。

### 8.10 H38：缩短时序并使用层级latent

H38把目标从T128缩短到T64，并将模型提高到37,574,883参数。

它的结构为：

```text
完整序列 ─► 6层 posterior encoder
                  │
                  ├─► 1个256维 global latent
                  └─► 16个128维 local latent

Mask后序列 ─► 独立4层 condition encoder ─► condition memory

时间+类型query
    │
    ▼
8层decoder：self-attention
          + 读取condition和17个latent的cross-attention
          + 每层global/local FiLM调制
          + FFN
    │
    ▼
State连续值 / contact logits / Action
```

global latent像整段录像的总摘要；16个local latent像16个章节摘要，每个负责4个transition。condition encoder则专门整理没有被Mask的内容。

decoder不再把被Mask序列本身当作主要query，而是从时间位置和State/Action类型建立query，并在每一层主动读取条件和latent。这是针对“长序列信息难以从单个token均匀广播”设计的结构性改进。

H38目前尚未正式运行。只有F4G正式通过后，才按以下阶段执行：

1. H38 smoke：只检查工程链路。
2. H38-A：全序列遮挡，只检查层级latent能否重建完整答案。
3. H38-B：训练8类固定物理Mask，检查condition融合。
4. H38-R：训练随机物理Mask并评测未训练Mask，检查Mask查询能力。

如果F4G通过而H38-A失败，才允许把相同结构扩大到约51M的H50做一次复核。H50仍失败就停止扩模，而不是建立无止境的参数阶梯。

## 9. 完整实验时间线

| 阶段 | 核心问题 | 结果 | 由结果触发的动作 |
|---|---|---|---|
| D1 | 一个窗口能否近无损记忆 | exact PASS | 增加窗口数 |
| F1R/W4/P1 | 同一motion更多窗口能否记忆 | progression可达，exact较难 | 增加长度和模型规模 |
| L128 | 1 motion、T128能否达到推进精度 | PASS | 扩到32 motion |
| F128 | 32 motion、T128能否直接拟合 | 明显FAIL | 缩到4 motion诊断 |
| F4D/F4A | 失败是否由少数离群点造成 | 否，大量位置都不够准 | 做loss/注入对照 |
| F4B/F4C | tail loss或逐层global gate能否解决 | 均未解决 | 绕过encoder诊断 |
| F4E | 显式window code+原decoder是否足够 | 未通过 | 比较latent拓扑 |
| F4F | 多global token或逐时间code是否足够 | 两者均FAIL，G8相对较好 | 检查直接输出上限 |
| F4G | 从零优化独立答案表能否在5k内拟合 | 持续改善但质量FAIL | 用F4G-O拆分优化与评测问题 |
| F4G-O | 真值直接复制后loss/Mask/evaluator是否可达 | 已实现，Ubuntu待运行 | PASS后进入H38；FAIL则修复协议 |
| H38-A/B/R | 层级latent在32 motion、T64上是否有效 | 尚未运行 | R通过后进入KL三路径 |
| KL三路径 | posterior与conditional prior能否对齐 | 尚未实现 | 根据三路径差异调整KL |

## 10. 当前最客观的结论

目前证据证明：最简模型可以把单个短窗口近乎无损记住；在一个motion上，较长训练也能达到`1e-2`量级推进精度。多项zero/donor实验还证明模型确实使用latent，而不是完全忽略它。

目前证据支持：当序列变为T128并扩展到多个motion时，误差不是集中在极少数点，而是分布在大量时间和特征位置。简单增加统一Transformer参数、改变tail loss、重复注入单个global latent、给每个window学习一个code或改变code拓扑，都没有在既定预算内通过。

目前尚无证据证明：单个global latent在理论上绝对无法完成任务；F4E/F4F失败只说明被测试的结构、初始化和预算未通过。也尚无证据证明H38一定能成功，因为它还没有正式运行。

目前不能证明：模型已具备conditional prior能力、随机生成能力、新motion泛化能力或真正的State→Action/Action→State部署推理。posterior能看完整答案，这与只看剩余条件是两件事。

当前最合理的下一步不是继续盲目改latent，也不是简单给F4G追加步数，而是用F4G-O直接复制真值，证明目标函数和评测门禁在32 motion、T64上确实存在零误差解；随后再用H38判断“缩短时序 + global/local层级latent + 独立condition encoder + 每层cross-attention/FiLM”是否能把时序细节稳定传到所有输出位置。

如果H38-R最终通过，我们才有一个可信的KL=0重建基线。之后再加入概率分布和KL，并公平比较posterior mean、posterior sample与conditional prior sample，才能回答这个模型是否真正成为可用的条件CVAE。
