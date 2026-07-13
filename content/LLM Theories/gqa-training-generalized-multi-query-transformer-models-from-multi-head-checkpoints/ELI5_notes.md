# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints 通俗讲解

### 0. 整体创新点通俗解读

**痛点直击**

在自回归生成场景下，Transformer 模型每生成一个 Token，都要把之前的 **KV cache**（Key 和 Value 矩阵）重新加载到显存里。这导致了极其严重的 **Memory Bandwidth Overhead**（内存带宽开销）。
- **MHA (Multi-Head Attention)** 的做法是：每个 Query Head 都配一个专属的 Key/Value Head。模型越大，Head 越多，加载 KV cache 的负担就越重，推理速度慢得让人抓狂。
- **MQA (Multi-Query Attention)** 的极端做法是：所有 Query Head 共享唯一的一个 Key/Value Head。速度确实起飞了，但模型容量受损，质量明显掉档，而且从头训练一个 MQA 模型既费钱又容易遇到训练不稳定的问题。
- 痛点总结：**MHA 太慢太贵，MQA 太糙且重训成本高**。业界急需一个既不用从头训练，又能兼顾速度与质量的折中方案。

---

**通俗比方**

假设你有一个大公司（模型），里面有 8 个部门（Query Heads）。
- **MHA 的做法**：每个部门雇一个专属秘书（Key/Value Head）。查资料时8个秘书同时跑，效率高，但养人成本极高（显存带宽爆炸）。
- **MQA 的做法**：为了省钱，裁掉7个秘书，全公司 8 个部门共用 1 个秘书。省钱是省钱了，但这个秘书很快就会崩溃，业务质量直线下降（模型质量退化）。
- **GQA (Grouped-Query Attention) 的做法**：折中一下。把 8 个部门分成 2 组，每组 4 个部门共享 1 个秘书。这样总共只雇 2 个秘书，成本比 8 个秘书低得多，但每个秘书的压力又没有 1 个秘书那么大。
- **Uptraining 的做法**：你不需要重新招聘培训一个新公司。直接把原来 MHA 公司里的 8 个秘书叫到会议室，让他们把各自掌握的资料“合并汇总”（Mean-pooling）成 2 份标准档案。然后让员工们拿着这 2 份新档案稍微适应工作 5% 的时间，公司就能无缝切换到低成本运作模式。

![](gmq-architecture.png) *image*

---

**关键一招**

作者并没有重头训练一个新架构，而是巧妙地在中间插了一个**Checkpoint 转换与微调**的流程，并提出了 **GQA** 架构。

- **结构插值**：将原本 $H$ 个 KV Head 划分成 $G$ 个组。**GQA-1** 退化为 MQA，**GQA-H** 退化为 MHA。通过选择一个中间值（如 **GQA-8**），在带宽开销和模型容量之间找到最佳平衡点。
- **Mean-pooling 转换**：在把 MHA Checkpoint 转成 GQA 时，作者没有随机初始化，也没有简单粗暴地只选第一个 Head，而是把同一个 Group 内原始的多个 KV Head 进行**均值池化**，融合成一个新 Head。这最大程度保留了预训练模型的知识。
- **低成本 Uptraining**：转换完后，只需要用原始预训练数据继续训练原计算量的 **5%**，模型就能完全适应新的 GQA 结构。

![](recycling.png) *Overview of conversion from multi-head to multi-query attention. Key and value projection matrices from all heads are mean pooled into a single head.*

**核心收益对比**

| 模型架构 | KV Head 数量 | 推理速度 | 模型质量 | 转换成本 |
| :--- | :--- | :--- | :--- | :--- |
| **MHA-XXL** | $H$ (多) | 慢 (1.51s) | 最高 (47.2) | 无需转换 |
| **MQA-XXL** | 1 (单) | 最快 (0.24s) | 略降 (46.6) | 5% Uptraining |
| **GQA-8-XXL** | $G$ (中) | 极快 (0.28s) | 极高 (47.1) | 5% Uptraining |

通过这一招，**GQA** 几乎达到了 **MHA-XXL** 的质量，但推理速度却逼近极致压缩的 **MQA-XXL**。这是一场用 5% 的训练算力换取数倍推理加速的精妙买卖。

### 1. Grouped-query Attention (GQA)

**痛点直击**

在自回归生成时，模型每吐出一个Token，都要把之前所有的Key和Value（也就是**KV cache**）从显存搬进计算核心。对于**MHA**（Multi-Head Attention）来说，每个Query head都配有一套独立的KV head，导致KV cache极其庞大。这就像搬砖，砖头（KV cache）太多，搬运工（内存带宽）跑断腿，推理速度慢得让人抓狂。

为了提速，前辈搞出了**MQA**（Multi-Query Attention），让所有Query head共享唯一的一套KV head。砖头是少了，速度飞起，但模型质量严重掉水，甚至训练时不稳定。痛点很明确：**要么慢得像蜗牛但聪明，要么快得像闪电但弱智**。

---

**通俗比方**

把Attention机制想象成公司里的一群经理（Query heads）在做决策，他们需要查阅档案（Key和Value）。

- **MHA**：每个经理配一个专属秘书。资料最全，决策最准，但养秘书的成本极高（内存带宽爆炸）。
- **MQA**：全公司几十个经理共用一个秘书。成本极低，但秘书忙不过来，经常给错资料，导致决策频频出错（质量下降）。
- **GQA**：折中方案。把经理们分成几个大组（比如8个组），每个组配一个秘书。既不用养那么多秘书，又比全公司共用一个秘书靠谱得多。

![](gmq-architecture.png) *image*

---

**关键一招**

作者并没有从头训练一个新模型，而是巧妙地在已有模型和目标结构之间插了一个**结构转换**的桥梁。

- **分组共享**：把原来的多个Query heads硬生生划分成$G$个组，组内的heads共享一套KV head。
- **Mean Pooling转换**：面对已有的**MHA**模型，不是随机初始化新的KV，也不是随便挑一个KV，而是把同组内的原始KV heads做**Mean Pooling**（平均池化），揉成一个新的KV head。这最大程度保留了原模型的知识。
- **低成本Uptraining**：转换后，只用原模型预训练算力的**5%**继续预训练，让模型适应新结构。

这套组合拳下来，GQA在质量和速度之间找到了甜点：

| 模型类型 | KV head数量 | 推理速度 | 模型质量 |
| :--- | :--- | :--- | :--- |
| **MHA** | $H$ (等于Query heads) | 慢 | 最高 |
| **GQA** | $G$ (中间值，如8) | 接近MQA | 接近MHA |
| **MQA** | 1 | 最快 | 下降 |

### 2. MHA-to-MQA/GQA Uptraining

**痛点直击**

- 大模型在自回归解码时，**KV cache** 的显存带宽开销是致命瓶颈。
- **MQA** 虽然能大幅减少 KV head 数量来提速，但**从头训练 MQA 模型不仅质量下降，还容易训练崩溃**。
- 痛点在于：手里已经砸重金训好了高质量的 **MHA** 模型，为了推理快一点，直接扔掉重新训 **MQA** 太败家；不重训，又享受不到推理加速。

---

**通俗比方**

- 假设你有一个 8 人组成的顾问团（**MHA** 的 8 个 KV head），每次做决策都要汇总所有人的意见，沟通成本极高。
- 为了提效，你想精简成 1 个总顾问（**MQA**）。
- 如果你直接招个新人从头培养（从头训练 **MQA**），费时费力且能力存疑。
- 聪明的做法是：把这 8 个专家的经验**取个平均值**，融合成 1 个“超级顾问”（**Mean-pooling**），然后让他稍微**实习几天**适应一下新岗位（5% 预训练算力），就能直接高效上岗。

---

**关键一招**

- 作者并没有重头训练，而是巧妙地在中间插了一个**结构转换与微调**的流程。
- **核心操作分两步**：
  - **Checkpoint 转换**：把原 **MHA** 模型的多个 Key 和 Value 投影矩阵进行 **mean-pooling**，直接平均合并成 **MQA** 或 **GQA** 需要的少量 head。这比随机初始化或只挑一个 head 效果好得多，最大程度保留了预训练知识。
  - **少量预训练**：用原始预训练数据和配方，只跑原来 **5%** 的计算量，让模型适应新的注意力结构。

![](recycling.png) *Overview of conversion from multi-head to multi-query attention. Key and value projection matrices from all heads are mean pooled into a single head.*

- **效果对比**：

| 模型转换方式 | 预训练算力消耗 | 模型质量 | 推理速度 |
| :--- | :--- | :--- | :--- |
| 从头训练 MQA | 100% | 较差/不稳定 | 极快 |
| MHA 直接推理 | 0% (已有) | 优秀 | 慢 |
| **Uptrained MQA/GQA** | **5%** | **接近 MHA** | **接近 MQA** |

- 这种 **Uptraining** 策略完美实现了“旧瓶装新酒”，用极低的成本把存量 **MHA** 模型转化为高效的 **MQA/GQA** 模型。
