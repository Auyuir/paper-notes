# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism 通俗讲解

### 0. 整体创新点通俗解读

**痛点直击**

之前的深度学习训练，大家主要靠**Data Parallelism**（数据并行）来加速。但这有个致命前提：**整个模型必须能塞进单张GPU的显存里**。随着Transformer模型参数量飙升到几十亿甚至上百亿，单卡显存（哪怕当时顶级的32GB V100）根本装不下模型权重、梯度和优化器状态。

为了解决“装不下”的问题，之前有两条路，但都很别扭：
- **Pipeline Parallelism**（流水线并行）：把模型按层切成几段，分给不同GPU。但这就像工厂流水线，前一个工序没干完，后一个就得干等（产生**Pipeline Bubble**），效率大打折扣。
- **Mesh-TensorFlow**等框架：虽然能做更细粒度的切分，但要求你把模型重写一遍，还得依赖专门的编译器，门槛极高，普通PyTorch玩家根本玩不转。

---

**通俗比方**

想象你要组装一台极其复杂的巨型机器（大模型），但你的工作台（单张GPU）太小，放不下所有零件。

- **Pipeline Parallelism**的做法是：把工作台连成一条流水线。张三装底盘，装完推给李四装轮胎，李四再推给王五装外壳。问题是，张三装底盘时，李四和王五只能干瞪眼（流水线气泡）。
- **Megatron-LM**的做法（**Intra-layer Model Parallelism**）：不搞流水线，而是**把同一个工序的零件对半劈开**。张三和李四的工作台各放一半底盘零件，两人**同时**装底盘，装完把各自的一半拼起来，再一起进入下一道工序（装轮胎）。

关键在于怎么“劈”零件才不会让两人互相干扰（通信开销）。Megatron发现，只要顺着矩阵的“列”劈，两人各自算完非线性激活（GeLU）后再拼，中间根本不需要停下来对暗号。

---

**关键一招**

作者并没有去造新轮子写编译器，而是巧妙地在原生PyTorch里插了几个**All-Reduce**操作，把Transformer里的矩阵乘法（GEMM）给“解剖”了：

- **MLP块的切分**：第一个GEMM按**列**切，这样后面的GeLU激活函数可以各自独立算，不用同步；第二个GEMM按**行**切，直接吃前一步的输出。整个前向传播只需要一次All-Reduce，反向传播也只要一次。
- **Self-Attention块的切分**：天生适合切！把Query、Key、Value的矩阵按**列**切，每个GPU算几个**Attention Head**，天然独立，最后再All-Reduce。
- **Cross-entropy融合**：输出层如果直接切，通信量极大（要传整个词表的Logits）。作者把Cross-entropy损失函数和切分后的矩阵乘**融合**在一起，GPU之间只传标量Loss，不传Logits，通信量断崖式下降。

![](images/5986c0716d74fdfed6adee2df2a6ec8c2d716cdf0a2f2a709603ee7517b26c10.jpg)

除了并行切分，作者还顺手解决了一个BERT扩容的致命Bug：
- 原版BERT把**Layer Normalization**放在残差连接之后（**Post-LN**）。模型一大，训练就发散崩溃。
- 作者把Layer Normalization挪到了残差连接之前（**Pre-LN**结构），这一个小小的位置对调，让BERT模型顺利扩展到了**3.9B**参数，且越大规模效果越好。

---

**性能与效果**

这套组合拳打下来，效果极其炸裂：

| 模型类型 | 参数量 | GPU数量 | 核心指标 |
|---|---|---|---|
| **GPT-2** | 8.3B | 512 V100 | WikiText103 PPL **10.8** (原SOTA 15.8) |
| **BERT** | 3.9B | 512 V100 | RACE 准确率 **90.9%** (原SOTA 89.4%) |

在512张GPU上跑8.3B模型，实现了**15.1 PetaFLOPs**的算力，并行效率高达**76%**。这证明了只要切得巧，不需要花里胡哨的编译器，原生PyTorch也能训出超大模型。

### 1. Intra-layer Model Parallelism for Transformers

**痛点直击**

当 Transformer 模型的参数量飙升到数十亿级别时，单张 GPU 的显存直接被撑爆，连模型权重和 Optimizer State 都装不下。
面对这个问题，传统的 Pipeline Parallelism 采取的是“纵向切分”：把第 1-10 层放 GPU 1，第 11-20 层放 GPU 2。
这种做法在特定场景下非常难受：
- **流水线气泡**：GPU 1 算完往前传时，GPU 2 只能干等，反之亦然，硬件利用率极低。
- **框架绑架**：如果想做更细粒度的切分（比如把单层切开），以前只能求助于 Mesh-TensorFlow 这种需要重写代码、依赖自定义编译器的重型框架，开发门槛极高。

我们需要一种原生的 PyTorch 实现，不写一行 C++ 代码，就能把一个巨大的 Transformer 层“横向剖开”，分散到多个 GPU 上同时算。

---

**通俗比方**

想象一条流水线在组装一辆汽车。
- **Pipeline Parallelism** 就像是：工人 A 负责装车轮，工人 B 负责装引擎。如果工人 A 动作慢，工人 B 就得干等，这就是“气泡”。
- **Intra-layer Model Parallelism** 则是：我们要造一个极其庞大的引擎，单个工人造不完。于是我们把引擎的图纸一分为二，工人 A 和工人 B 同时开工造各自那一半，最后把两部分焊在一起。

在这个比方里，那个“焊在一起”的动作就是 GPU 之间的通信。Megatron-LM 的核心智慧就在于：**尽量推迟焊接的时间，并且减少焊接的次数**。

---

**关键一招**

作者并没有重头造轮子，而是巧妙地利用了矩阵乘法（GEMM）的数学特性，在原生 PyTorch 里插了几个通信原语。

**1. MLP 块的“列切行接”**

在标准的 MLP 中，流程是 $Y = \text{GeLU}(XA)$，然后再过一个矩阵 $B$。
如果按行切分矩阵 $A$，因为 GeLU 是非线性函数，算完必须立刻通信同步，极其低效。
作者的关键扭转在于：**按列切分矩阵 $A$**。
- $A$ 被切成 $[A_1, A_2]$，GPU 1 算 $XA_1$，GPU 2 算 $XA_2$。
- 因为是按列切，输出也自然分成了两半 $[Y_1, Y_2]$。
- 此时 GeLU 可以**各自独立**作用在 $Y_1$ 和 $Y_2$ 上，完全不需要通信！
- 接着，把第二个矩阵 $B$ 按行切分，直接吃掉 GeLU 的输出。
- 直到 MLP 块的最后，才做一次 all-reduce 把结果加起来。

![](images/5986c0716d74fdfed6adee2df2a6ec8c2d716cdf0a2f2a709603ee7517b26c10.jpg)

**2. Self-Attention 块的“头切”**

Self-Attention 天然适合并行，因为它有多个独立的 Attention Head。
- 把 Q、K、V 矩阵按列切分，每个 GPU 负责几个完整的 Head。
- 各算各的 Attention，完全不需要管对方。
- 最后通过一个按行切分的输出矩阵，直接接住结果，同样只在块末尾做一次 all-reduce。

![](images/b559b5df2340bcc3a19c9a3dae7bca78d0006f3b98d41d1683783240be7babbc.jpg) *(a) MLP (b) Self-Attention Figure 3. Blocks of Transformer with Model Parallelism. f and g are conjugate. $f$ is an identity operator in the forward pass and all reduce in the backward pass while g is an all reduce in the forward pass and identity in the backward pass.*

**3. 通信与计算的极致压缩**

通过上述巧妙的切分，一个完整的 Transformer 层被拆解后，通信开销被压到了最低。

| 模块 | 前向传播 | 反向传播 |
| :--- | :--- | :--- |
| MLP Block | 1 次 all-reduce | 1 次 all-reduce |
| Self-Attention Block | 1 次 all-reduce | 1 次 all-reduce |
| **总计 (单层)** | **2 次 all-reduce** | **2 次 all-reduce** |

![](images/7925c10de1b603c0e6d8c9d2d2065c9b8e487bbdb2bcad2342f541110dcf7a0e.jpg) *Figure 4. Communication operations in a transformer layer. There are 4 total communication operations in the forward and backward pass of a single model parallel transformer layer.*

此外，对于 Embedding 层，由于词表极大，直接通信会传输庞大的 Logits。作者把 Cross-entropy Loss 和并行 GEMM 融合，GPU 之间只通信标量 Loss，而不是整个词表的概率分布。

**总结**：这套组合拳打下来，不需要任何编译器层面的魔法，仅靠在 PyTorch 的前向/反向传播中精准插入 $f$ (identity in forward, all-reduce in backward) 和 $g$ (all-reduce in forward, identity in backward) 算子，就实现了十亿参数级模型的高效多卡训练。

### 2. Fused Cross-Entropy for Embedding Parallelism

**痛点直击**

在训练超大语言模型时，**Embedding** 层的权重矩阵非常庞大。它的维度是隐藏层大小（$H$）乘以词表大小（$v$）。对于现代语言模型，$v$ 通常在几万到几十万的量级（例如 GPT-2 的 50,257）。

当我们将模型切分到多块 GPU 上时，很自然的想法是按词表维度把这层矩阵切开，每块 GPU 各算一部分。但这带来了一个极其难受的通信瓶颈：
- 各个 GPU 算完自己那部分 **GEMM** 后，得到了局部的 **Logits**。
- 为了算出最终的 **Cross-Entropy Loss**，标准的 PyTorch 做法是必须把所有 GPU 上的 **Logits** 拼起来。
- 这意味着你需要执行一次 `all-gather`，通信的数据量是 $b \times s \times v$（Batch Size $\times$ Sequence Length $\times$ Vocab Size）。
- 因为 $v$ 极其庞大，这个通信量会直接把网络带宽塞满，GPU 算得再快也得停下来等数据传输，这就是典型的“顾头不顾尾”——省了显存，却毁了速度。

---

**通俗比方**

假设你们团队有 8 个人（8 块 GPU）共同批改一份有 5 万道单选题的试卷（词表大小 $v$）。

- **传统做法**：每个人只负责批改其中 6250 道题。为了算出学生的总分，每个人必须把自己批改的 6250 道题的详细得分表（巨大的 **Logits** 矩阵）打包，通过快递（网络通信）寄给主考官。主考官收齐 8 个大包裹后，拼成完整的 5 万道题的得分表，最后才算出总平均分。运送这些包裹的过程既慢又费钱。
- **融合做法**：主考官直接把计算总分的公式下发。每个人在本地批改完自己的 6250 道题后，直接在本地套用公式算出一个“局部平均分”（一个标量）。然后大家只需要在群里报一下自己的“局部平均分”（极小的数据量），主考官做个简单的加权平均就搞定了。你不再运送成千上万张试卷，只运送 8 个数字。

---

**关键一招**

作者并没有重写底层的通信库，而是巧妙地在计算流程中间插了一个“融合”操作，把通信的维度给彻底扭转了。

具体来说，他们改变了计算顺序：
- **没有**先做 `all-gather` 拼接巨大的 **Logits** 张量，再去算 **Cross-Entropy**。
- **而是**将 **Cross-Entropy** 的数学计算直接推到了每个 GPU 的本地 **Logits** 分片上。
- 每个 GPU 拿着自己那部分 $b \times s \times (v/N)$ 的输出，直接在本地计算出一个局部的 **Loss**。
- 随后，GPU 之间只需要对这个极小的局部结果执行 `all-reduce` 操作。

![](images/7925c10de1b603c0e6d8c9d2d2065c9b8e487bbdb2bcad2342f541110dcf7a0e.jpg) *Figure 4. Communication operations in a transformer layer. There are 4 total communication operations in the forward and backward pass of a single model parallel transformer layer.*

通过这一步“乾坤大挪移”，通信的数据量发生了质的飞跃：

| 对比维度 | 传统做法 | Megatron-LM 融合做法 |
|---|---|---|
| **通信内容** | 完整的 **Logits** 张量 | 标量 **Loss** 值 |
| **通信维度** | $b \times s \times v$ | $b \times s$ |
| **通信原语** | `all-gather` | `all-reduce` |
| **瓶颈消除** | 极易塞满网络带宽 | 几乎可忽略不计 |

这就叫“算子融合”：与其让 GPU 之间搬运庞大的中间结果，不如把计算逻辑分发下去，让数据就地在原地完成蜕变，最后只通信蜕变后的精华。

### 3. Rearranged Layer Normalization for BERT Scaling

**痛点直击**

- 之前的 BERT 架构在扩大规模时，会遇到严重的**模型退化**。
- 具体表现：当模型参数量超过 BERT-Large (336M) 后，继续增加层数或隐藏层维度，效果不升反降，训练过程极不稳定。
- 根本原因：深层网络带来的**梯度消失或爆炸**，导致优化器无法有效更新参数。之前的 ALBERT 试图用参数共享来缓解，但这限制了模型容量，属于治标不治本。

---

**通俗比方**

- 残差连接像是一条信息传递的**高速公路**，让梯度能一路畅通无阻地回传。
- Layer Normalization 像是高速公路上的**安检站**，负责把数据分布拉回正轨。
- 原来的 BERT 把安检站放在主干道汇合之后，所有车流汇合后再做一次大安检，车一多（模型变大）就容易堵死，导致梯度回传受阻。
- 现在把安检站放在车进入主干道之前的辅道上，主干道永远畅通无阻。

![](images/297aca6cd81f73e0657fcf206b71d4801322a761ec682fc981ad60976d183aee.jpg)

---

**关键一招**

- 作者并没有重头设计新架构，而是巧妙地调整了 Layer Normalization 的位置。
- 把 Layer Normalization 从残差连接之后移到之前，即从 **Post-LN** 变成 **Pre-LN**。
- 这样主干道上的梯度可以直接回传，不受 LN 的阻隔，大模型也能稳定训练。

| 架构 | Layer Normalization 位置 | 残差连接状态 | 大规模训练表现 |
| :--- | :--- | :--- | :--- |
| **原始 BERT (Post-LN)** | 残差连接之后 | 受 LN 阻隔 | 不稳定，易退化 |
| **Megatron BERT (Pre-LN)** | 残差连接之前 | 畅通无阻 | 稳定，效果提升 |
