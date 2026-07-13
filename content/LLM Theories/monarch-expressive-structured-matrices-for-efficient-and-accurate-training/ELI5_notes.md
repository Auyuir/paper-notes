# Monarch: Expressive Structured Matrices for Efficient and Accurate Training 通俗讲解

### 0. 整体创新点通俗解读

**痛点直击**

搞大模型的人都知道，权重矩阵太胖，训练和微调都极其烧钱。大家自然想到用稀疏或结构化矩阵（比如低秩、FFT）去替代稠密矩阵，但这招一直不好使，主要卡在两个极其难受的痛点上：
- **E2E训练的“硬件鸿沟”**：很多稀疏方法在纸面上FLOPs减少了，但在GPU上跑起来反而更慢。因为GPU是面向块计算的，细碎的稀疏分布会导致访存极差。而像FFT这类结构化矩阵，虽然数学上很美，但太死板，只能用特定的变换，无法通用表达卷积等操作。
- **D2S微调的“投影黑洞”**：拿到别人预训练好的稠密大模型，想把它“压缩”成结构化矩阵来加速微调。但对于稍微复杂一点的结构化矩阵，根本没有高效的算法去近似原矩阵。只能用启发式方法硬凑，不仅慢，还没理论保证。

---

**通俗比方**

你可以把GPU想象成一个极其高效的**流水线工厂**，它最喜欢处理规整的“标准箱”（Batch Matrix Multiply）。
- 之前的**Butterfly矩阵**就像是一堆形状奇特的小零件，虽然拼起来功能强大（能表示各种快速变换），但让流水线去处理这些零碎零件，频繁换模具，效率极低。
- **Monarch矩阵**的思路是：与其用零碎零件，不如把它们预先打包成几个“标准大模块”（两个块对角矩阵）。流水线处理这些大模块非常顺手，速度直接拉满；同时，这些大模块拼在一起，依然能拼出原来那些复杂零件的功能。
- 至于**投影难题**，就像你要把一幅名画（稠密矩阵）用乐高积木拼出来。Monarch的巧妙之处在于，它发现只要把画切分成网格，每个网格里只需要用最简单的长条形积木（秩为1的矩阵）去近似就行了。而且找这些积木的过程有标准答案，不用瞎猜。

---

**关键一招**

作者并没有去硬啃那些复杂的结构化矩阵优化问题，而是巧妙地改变了矩阵的参数化形式，并做了一个极其漂亮的**视角转换**：
- **结构重组**：把矩阵乘法拆解为 `置换 -> 块对角乘 -> 置换 -> 块对角乘`。这直接迎合了GPU的BMM指令，硬生生把访存效率提了上去。
- **张量重塑**：为了解决把稠密矩阵投影到Monarch的难题，作者没有去解复杂的非凸优化，而是把二维矩阵看作**四维张量**。
- **降维打击**：这一视角转换后，原本纠缠在一起的矩阵乘积近似问题，瞬间坍缩成了对一堆小矩阵的**独立秩1近似**问题。直接套用SVD，就能得到解析最优解。

![](monarch-main.png) *Monarch matrices unlock several ways to train sparse and dense models: end-to-end training a sparse (Monarch) model can be 2x faster than dense training thanks to its hardware efficiency; sparse-to-dense “reverse sparsification” can speed up training of large models such as GPT-2; and our dense-to-sparse Monarch projection algorithm can transfer knowledge from pretrained dense model to Monarch model and speed up BERT fine-tuning.*

这种结构带来的收益是全方位的，具体体现在三种训练范式上：
- **E2E训练**：直接用Monarch替代稠密权重，训练速度提升2倍，精度几乎不掉。
- **Sparse-to-Dense (反向稀疏化)**：先用Monarch训90%的时间，最后10%转回稠密。既享受了前期的速度，又保证了最终的稠密质量。
- **D2S微调**：用解析算法把预训练稠密模型直接投影成Monarch模型，微调速度提升1.7倍。

| 训练范式 | 模型/任务 | 核心收益 | 加速比 |
| :--- | :--- | :--- | :--- |
| **E2E训练** | ViT / GPT-2 | 训练速度翻倍，精度无损 | **2.0x** |
| **S2D训练** | GPT-2 / BERT | 前期稀疏加速，后期转稠密保质量 | **2.0x** / **1.23x** |
| **D2S微调** | BERT (GLUE) | 预训练模型直接投影，参数减半 | **1.7x** |

### 1. Monarch矩阵参数化

**痛点直击**

在训练庞大的 Transformer（如 ViT、GPT-2）时，最耗时的就是那些巨大的 Dense Weight Matrices（稠密权重矩阵）。为了省算力和内存，大家自然想到用 Structured Matrices（结构化矩阵，如 Sparse 或 Butterfly 矩阵）去替换它们。但之前的做法在 GPU 上简直是“顾头不顾尾”：
- **Sparse Matrices（稀疏矩阵）**：虽然理论上 FLOPs 少了，但在 GPU 上运行反而更慢。因为非零元素分布零散，无法利用 GPU 最擅长的 Block-oriented（块状）计算，导致严重的 Memory Fetching 开销。而且它无法表达 Fourier Transform 这类包含领域知识的常用变换。
- **Butterfly Matrices（蝴蝶矩阵）**：表达能力极强，能表示几乎所有快速变换。但它是由 $\log n$ 个极小的因子矩阵相乘而成。GPU 讨厌这种“碎活儿”，频繁启动微小的 Kernel 会导致极高的 Overhead，实际 Wall-clock Time 毫无优势。
- **Dense-to-Sparse 转换难**：如果你已经有一个预训练好的 Dense 模型，想把它转换成 Butterfly 矩阵来微调，这是一个 Nonconvex（非凸）问题，没有解析解，只能靠启发式搜索硬凑，极不tractable。

---

**通俗比方**

我们可以把矩阵乘法类比为**物流分拣系统**。假设你要把 $n$ 个包裹分发到 $n$ 个目的地。
- **Dense Matrix**：动用 $n \times n$ 的人工矩阵，一对一搬运。费时费力，但路线直接。
- **Sparse Matrix**：随机裁掉一部分工人。理论上人少了，但剩下的工人到处乱跑捡包裹，流水线全乱了，效率反而更低。
- **Butterfly Matrix**：设计了一套 $\log n$ 层的精密分拣传送带。逻辑完美，但每一层只处理极少量的包裹，GPU 这个“巨型分拣中心”根本吃不饱，大部分时间都在等传送带启动。
- **Monarch Matrix**：把 $\log n$ 层传送带直接压缩成**两道大工序**。
  - 第一道工序：把包裹打包成标准尺寸的货盘（对应 Block-diagonal Matrix $R$）。
  - 第二道工序：把货盘整体调换位置后，再用另一组标准货盘分拣（对应 Permuted Block-diagonal Matrix $L$）。
  - GPU 最喜欢处理这种“标准货盘”（Batch Matrix Multiply，BMM），吞吐量直接拉满。

![](monarch-1.png) *Monarch matrices are parametrized as products of two block-diagonal matrices up to permutation, allowing efficient multiplication algorithm that leverages batch matrix multiply.*

---

**关键一招**

作者并没有重头设计一套全新的数学变换，而是巧妙地在 Butterfly 矩阵的分解流程中插了一个**“化零为整”的聚合操作**。

- **参数化扭转**：作者将原本 $\log n$ 个碎片的 Butterfly 因子，强行打包成两个 Block-diagonal Matrices（$L$ 和 $R$），中间用一个固定的 Permutation Matrix（$P$）连接。公式变为 $M = P L P^T R$。
- **硬件效率**：$L$ 和 $R$ 都是块对角矩阵，这意味着在 GPU 上计算 $Mx$ 时，可以直接调用高度优化的 BMM（Batch Matrix Multiply）例程。$P$ 只是一个 Reshape 加 Transpose，几乎零成本。这直接带来了高达 **2x** 的 Dense Matrix 乘法加速。
- **解析投影算法**：这是最神来之笔。因为 $M = P L P^T R$ 的结构，当你把给定的 Dense Matrix $A$ Reshape 成一个 4D Tensor 时，寻找最优的 $L$ 和 $R$ 来逼近 $A$ 的问题，被完美解耦成了 $m \times m$ 个**独立的 Rank-1 Approximation 问题**。
- **逻辑转换**：作者不需要用梯度下降去硬搜非凸空间，而是直接对 $A$ 的各个切片做 **SVD（奇异值分解）**。这就像 Eckart-Young 定理在低秩近似中的神级地位一样，Monarch 给出了结构化矩阵投影的**解析最优解**。

| 特性 | Dense Matrix | Butterfly Matrix | **Monarch Matrix** |
| :--- | :--- | :--- | :--- |
| **GPU 硬件利用率** | 高 | 极低 | **极高 (BMM 友好)** |
| **表达能力** | 完全 | 极强 (可表 Fourier 等) | **强 (包含 Butterfly)** |
| **Dense 投影解析解** | N/A | 无 | **有 (基于 SVD)** |
| **实际训练加速** | 基准 | 往往变慢 | **2x 加速** |

这个设计不仅让 End-to-End 训练提速 2 倍，还解锁了“Reverse Sparsification”（先用 Monarch 快速预训练，最后 10% 转回 Dense 收尾）和直接微调预训练 Dense 模型的新玩法。

### 2. Monarch矩阵的解析投影与分解算法

**痛点直击**

当你手里攥着一个预训练好的庞大 Dense 模型（比如 BERT），想把它压缩成 Monarch 模型来加速下游任务的 Fine-tuning 时，你会撞上一堵硬墙：**矩阵投影**。
- 你需要找到一个 Monarch 矩阵，让它与原始的 Dense 权重矩阵尽可能接近。
- 以前的 Butterfly 矩阵也面临过这个问题，但它的数学结构太复杂，投影过程是一个**非凸优化问题**。
- 传统做法只能靠迭代法（如梯度下降、交替最小化）去硬凑，不仅慢得让人抓狂，还极容易陷入局部最优，导致投影后的模型“智商掉线”。
- 这就造成了“两头堵”的难受局面：直接训 Sparse 模型难收敛，想把 Dense 模型转成 Sparse 模型又找不到精准的转换算法。

---

**通俗比方**

想象你要把一幅极其复杂的名画（Dense 矩阵）用乐高积木（Monarch 矩阵）复刻出来。
- **传统迭代法**：像盲人摸象，先随便拼一块，看看不对再调整旁边的，反反复复试错。不仅耗时，最后拼出来的画可能面目全非（局部最优）。
- **Monarch 的解析投影**：你突然发现，只要把原画切分成 $m \times m$ 个小网格，并且规定每个网格只能用**单一颜色的渐变**（秩为1的矩阵）来填充。那你根本不需要试错，直接拿仪器测一下每个网格的主色调就行了，一步到位，绝不走眼。

---

**关键一招**

作者并没有去硬解那个非凸的矩阵乘积优化问题，而是通过**维度重塑**玩了一手漂亮的“偷梁换柱”。

- **维度变换**：把 $n \times n$ 的 Dense 矩阵 $\mathbf{A}$ 和 Monarch 矩阵 $\mathbf{M}$ 都 Reshape 成 $m \times m \times m \times m$ 的 4D 张量（其中 $n = m^2$）。
- **结构暴露**：在这个 4D 视角下，Monarch 矩阵 $\mathbf{M}$ 的结构底牌尽显——它变成了 $m \cdot m$ 个独立的、大小为 $m \times m$ 的子矩阵，且**每个子矩阵的秩必须为1**。
- **降维打击**：原本“求两个块对角矩阵乘积的最优逼近”这个非凸问题，瞬间坍缩成了 $m^2$ 个互相独立的“**求秩1近似**”问题。
- **解析求解**：求矩阵的秩1近似有现成的解析解——直接上 **SVD（奇异值分解）**。根据 Eckart-Young 定理，取最大奇异值对应的左右奇异向量，就是 Frobenius 范数下的最优解。

![](monarch-3.png) *With [alg:project] for our Monarch parameterization, we can convert a pretrained model into a model with Monarch weight matrices and speed up downstream fine-tuning.*

通过这一招，投影算法不仅有了**理论最优保证**，计算复杂度也降到了 $O(n^{5/2})$。

**传统投影 vs Monarch 投影对比**

| 特性 | 传统 Butterfly 投影 | Monarch 解析投影 |
| :--- | :--- | :--- |
| **求解方式** | 迭代优化 | SVD 解析求解 |
| **最优性** | 无保证，易陷局部最优 | **全局最优** |
| **计算复杂度** | 高，需多次迭代 | $O(n^{5/2})$，一步到位 |
| **应用场景** | 难以用于 D2S Fine-tuning | 完美适配预训练模型压缩 |

这招最巧妙的地方在于：**不是去攻克非凸优化，而是通过改变观察维度，把非凸问题直接变成了线性代数课本上的基础题。**

### 3. 反向稀疏化训练策略

**痛点直击**

在训练大模型（如 GPT-2 或 BERT）时，我们经常陷入一种**“既要又要”**的尴尬境地：
- **要效果好**：模型必须得“大”，得用 **Dense**（稠密）矩阵，因为语言建模需要海量参数来“死记硬背”文本规律。但 **Dense** 训练太慢，算力烧不起。
- **要速度快**：换成 **Sparse**（稀疏）矩阵训练，虽然 FLOPs 降了，但面临两个致命问题。第一，大多数稀疏结构在 GPU 上根本跑不快（访存不连续）；第二，就算用上了硬件友好的 **Monarch** 矩阵，由于总参数量变小了，模型到了后期会出现**“容量瓶颈”**，根本记不住那么多知识，精度铁定掉。

这就好比你想建一座摩天大楼，但预算有限。从头到尾都用最顶级的建材（**Dense**），太贵；全用轻钢龙骨（**Sparse**），楼建不高还会塌。以前的稀疏化训练（如剪枝）大多是“建好后拆”，但拆完往往精度受损，且训练阶段的钱一分没少花。

---

**通俗比方**

这其实就是一套**“先搭骨架，后精装修”**的工程学智慧。

想象你要画一幅超高清的《清明上河图**：
- **传统 Dense 训练**：从第一笔开始就用极细的工笔勾勒每一个人物的发丝。极其耗时，且前期你连整体构图都没定，画错一笔修改成本极高。
- **反向稀疏化（Reverse Sparsification）**：你先用大排刷（**Monarch** 矩阵）快速铺满 90% 的画布，把整体的构图、色调、大轮廓全打出来。这一步极快，因为笔触大（计算快）。等到了最后 10% 的收尾阶段，你突然换成极细的工笔（转回 **Dense** 矩阵），去精修人物表情、店铺招牌等细节。

作者并没有傻乎乎地从头到尾死磕，而是巧妙地利用了模型训练的**阶段性特征**：前期重在“学大结构”，后期重在“抠小细节”。

---

**关键一招**

作者在训练流程里插入了一个极其巧妙的**“中途换胎”**操作。

具体流程如下：
- **阶段一（0% - 90% 时间）**：用 **Monarch** 矩阵替换模型里的 **Dense** 权重进行训练。因为 **Monarch** 是两个块对角矩阵的乘积，完美契合 GPU 的批量矩阵乘法（BMM），所以这一阶段不仅参数少，而且**墙钟时间**跑得飞快。模型迅速收敛到一个“差不多”的状态。
- **阶段二（90% - 100% 时间）**：这是最神来之笔的一步。作者直接把训练好的 **Monarch** 矩阵的因子乘开（$M = P L P^\top R$），**硬生生展开还原成普通的 Dense 矩阵**。然后带着现有的权重，继续用传统的 Dense 方式训练剩下的 10%。

![](monarch-2.png) *With the “reverse sparsification” process, Monarch matrices can speed up GPT-2 training by 2x.*

**为什么这招能管用？**
- **前期省时间**：前 90% 的迭代里，模型主要在学习数据中全局的、粗粒度的模式，**Monarch** 的表达能力完全够用，且换来了 2 倍的加速。
- **后期破瓶颈**：最后 10% 换回 **Dense**，瞬间释放了模型的参数容量。这就好比赛车跑完了大半程，最后冲刺时把限速器拆了。模型利用这最后的迭代，把那些 **Monarch** 结构无法拟合的“高频细节”给补全了。

**实战效果**：
这套“反向稀疏化”在 GPT-2 和 BERT 的预训练上简直是无缝衔接。

| 模型与训练策略 | 达到目标 PPL 的时间 | 加速比 |
| :--- | :--- | :--- |
| GPT-2 (Dense) | 1.0x | - |
| GPT-2 (Reverse Sparsification) | 0.5x | **2.0x** |
| BERT (Nvidia MLPerf 1.1) | 30.2 小时 | - |
| BERT (Reverse Sparsification) | 23.8 小时 | **1.23x** |

你看，连 Nvidia 那些把算子优化到极致的工程师，也被这套“先糙后细”的算法策略在端到端时间上超越了。这就是跳出底层算子、在训练策略上做降维打击的威力。
