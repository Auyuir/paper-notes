# Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works. 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.

**发表期刊/会议 (Journal/Conference)**: NeurIPS

**发表年份 (Publication Year)**: 2017

**研究机构 (Affiliations)**: Google Brain, Google Research, University of Toronto

---

## 1. 摘要

**目的**

- 提出一种新的序列转导模型架构：**Transformer**。
- 核心目标：
  - 用**Attention mechanism**完全替代传统的**RNN**和**CNN**。
  - 解决 recurrent models 在长序列训练中的**串行计算瓶颈**。
  - 提升机器翻译任务中的**训练效率**、**并行化能力**和**建模长距离依赖的能力**。
- 研究动机：
  - 传统**RNN/LSTM/GRU**需要按 Token 顺序递推，训练时难以在单个样本内部并行。
  - **CNN-based sequence models**虽然可并行，但远距离位置之间的信息交互需要多层堆叠。
  - **Self-Attention**可在常数路径长度内连接任意两个位置，更适合捕获长距离依赖。

---

**方法**

- 提出的模型为**Transformer encoder-decoder architecture**，完全基于**Multi-Head Self-Attention**和**Position-wise Feed-Forward Networks**。

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

- **Encoder**设计：
  - 由**N=6**个相同层堆叠组成。
  - 每层包含：
    - **Multi-Head Self-Attention**
    - **Position-wise Feed-Forward Network**
  - 每个 sub-layer 外部使用：
    - **Residual Connection**
    - **Layer Normalization**
  - 模型维度为**d_model=512**。

- **Decoder**设计：
  - 同样由**N=6**个相同层堆叠组成。
  - 每层包含：
    - **Masked Multi-Head Self-Attention**
    - **Encoder-Decoder Attention**
    - **Position-wise Feed-Forward Network**
  - 通过 mask 阻止当前位置访问未来 Token，保持**auto-regressive**生成性质。

- **Scaled Dot-Product Attention**：
  - 使用 Query、Key、Value 计算注意力权重。
  - 公式为：**Attention(Q,K,V)=softmax(QKᵀ/√d_k)V**。
  - 使用**1/√d_k**缩放，避免高维 dot product 导致 softmax 饱和、梯度过小。

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

- **Multi-Head Attention**：
  - 将 Query、Key、Value 投影到多个子空间并行计算 Attention。
  - 本文 base model 使用：
    - **h=8**个 attention heads
    - **d_k=64**
    - **d_v=64**
  - 优势：
    - 不同 heads 可关注不同位置和不同语义子空间。
    - 缓解单头 attention 中信息被平均化的问题。

- **Position-wise Feed-Forward Network**：
  - 每个位置独立应用相同的两层全连接网络。
  - 使用**ReLU**激活。
  - 维度设置：
    - 输入/输出维度：**d_model=512**
    - 隐层维度：**d_ff=2048**

- **Positional Encoding**：
  - 因模型无 recurrence 和 convolution，需显式注入位置信息。
  - 使用正弦和余弦函数构造固定位置编码：
    - 偶数维使用**sin**
    - 奇数维使用**cos**
  - 设计目的：
    - 表示绝对位置。
    - 使模型更容易学习相对位置关系。
    - 可能泛化到训练时未见过的更长序列。

- **训练设置**：
  - 数据集：
    - **WMT 2014 English-German**
    - **WMT 2014 English-French**
  - Optimizer：
    - **Adam**
    - β₁=0.9，β₂=0.98，ε=10⁻⁹
  - Learning rate schedule：
    - 前**4000**步 warmup
    - 之后按 step number 的 inverse square root 衰减
  - Regularization：
    - **Residual Dropout**
    - **Embedding + Positional Encoding Dropout**
    - **Label Smoothing ε_ls=0.1**

---

**结果**

- **Transformer**在机器翻译任务上达到或超过当时 state-of-the-art。
- 相比 prior models，Transformer 在 BLEU 和训练成本上均表现突出。

| Model | EN-DE BLEU | EN-FR BLEU | EN-DE Training Cost |
|---|---:|---:|---:|
| ByteNet | 23.75 | — | — |
| GNMT + RL | 24.6 | 39.92 | 2.3×10¹⁹ FLOPs |
| ConvS2S | 25.16 | 40.46 | 9.6×10¹⁸ FLOPs |
| MoE | 26.03 | 40.56 | 2.0×10¹⁹ FLOPs |
| GNMT + RL Ensemble | 26.30 | 41.16 | 1.8×10²⁰ FLOPs |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7×10¹⁹ FLOPs |
| Transformer base | 27.3 | 38.1 | **3.3×10¹⁸ FLOPs** |
| Transformer big | **28.4** | **41.8** | 2.3×10¹⁹ FLOPs |

- **English-to-German**：
  - **Transformer big**达到**28.4 BLEU**。
  - 比此前最优 ensemble 结果高出超过**2 BLEU**。
  - 成为新的 state-of-the-art。

- **English-to-French**：
  - **Transformer big**达到**41.8 BLEU**。
  - 建立新的 single-model state-of-the-art。
  - 训练仅需**3.5天**，使用**8块P100 GPUs**。

- **训练效率**：
  - **Transformer base**训练约**12小时**。
  - **Transformer big**训练约**3.5天**。
  - 相比传统 recurrent 或 convolutional sequence models，训练成本显著降低。

- 不同层类型的复杂度对比显示，**Self-Attention**在并行性和路径长度方面具有明显优势。

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---:|---:|---:|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(logₖ(n)) |
| Restricted Self-Attention | O(r·n·d) | O(1) | O(n/r) |

- 模型变体实验表明：
  - **Multi-Head Attention**优于 single-head attention。
  - Attention heads 过少或过多都会影响质量。
  - 较大的模型通常取得更好性能。
  - **Dropout**对防止 overfitting 非常重要。
  - 固定 sinusoidal positional encoding 与 learned positional embedding 效果接近。

- 在**English constituency parsing**任务中，Transformer 也表现出良好的泛化能力。

| Model | Training Setting | WSJ 23 F1 |
|---|---|---:|
| Berkeley Parser | WSJ only | 90.4 |
| Dyer et al. | WSJ only | 91.7 |
| Transformer 4 layers | WSJ only | **91.3** |
| McClosky et al. | semi-supervised | 92.1 |
| Transformer 4 layers | semi-supervised | **92.1** |

- Attention 可解释性观察：
  - 部分 heads 学会捕获长距离依赖。
  - 部分 heads 关注 anaphora resolution。
  - 部分 heads 显示出与句法结构相关的模式。

![](57e3ad00e7c57fe0dc66b468b013f2fcf447ef78a7d2ee01be8b434fe6ef0669.jpg)

![](1678a839f4e6f07663c6d0c31aa1125d433138f2e6617aae82d039d1529967ea.jpg) *Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.*

![](b616696f63c30ad27de5f095b0cac5975742209d1231c70057510010187c1512.jpg)

---

**结论**

- **Transformer**证明了序列建模并不必须依赖**recurrence**或**convolution**。
- 完全基于**Attention**的架构可以：
  - 显著提升训练并行度。
  - 缩短长距离依赖的信息路径。
  - 在机器翻译任务上取得更高 BLEU。
  - 以更低训练成本达到或超过 previous state-of-the-art。
- **Multi-Head Self-Attention**是核心贡献：
  - 支持不同 heads 学习不同依赖关系。
  - 提升模型对语义、句法和长距离关系的表达能力。
- **Positional Encoding**有效弥补了无 recurrence/convolution 架构中的顺序信息缺失。
- 论文奠定了后续大规模语言模型和现代 foundation models 的关键架构基础。

---

## 2. 背景知识与核心贡献

**研究背景**

- 在本文发表前，主流的**sequence transduction**任务模型主要依赖复杂的**Recurrent Neural Network, RNN**或**Convolutional Neural Network, CNN**架构。
  - 典型任务包括：
    - **Machine Translation**
    - **Language Modeling**
    - **Sequence-to-Sequence Learning**
    - **Constituency Parsing**
  - 代表性模型包括：
    - **LSTM**
    - **GRU**
    - **Encoder-Decoder**
    - **GNMT**
    - **ConvS2S**
    - **ByteNet**

- 当时性能最强的模型通常采用**encoder-decoder**结构，并在两者之间加入**attention mechanism**。
  - **Attention**已经被证明能够有效建模输入与输出序列之间的依赖关系。
  - 但多数模型仍将**Attention**作为辅助模块，主体计算仍依赖**RNN**或**CNN**。

- **RNN**的核心限制在于计算具有天然的顺序依赖。
  - 每个时间步的隐藏状态依赖前一个时间步：
    - 当前状态需要基于$h_{t-1}$计算$h_t$。
  - 这导致训练阶段难以在单个序列内部进行充分并行化。
  - 序列越长，训练效率越受限。
  - 长距离依赖需要跨越较长路径，梯度传播和依赖建模更困难。

- **CNN-based sequence models**虽然比RNN更容易并行化，但仍存在路径长度问题。
  - 局部卷积需要多层堆叠才能连接远距离位置。
  - 对于长度为**n**的序列：
    - 普通卷积的远距离路径长度随距离增长。
    - Dilated Convolution可以降低路径长度，但仍不是常数级。
  - 这会增加学习远距离依赖的难度。

- 本文提出的关键背景判断是：
  - **Attention机制本身已经具备全局依赖建模能力**。
  - 如果完全去除**recurrence**和**convolution**，仅依赖**self-attention**，可能同时获得更强表达能力和更高并行效率。

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

---

**研究动机**

- 核心动机是设计一种更简单、更高效、更易并行的序列建模架构。
  - 摆脱**RNN**的顺序计算瓶颈。
  - 避免**CNN**在远距离依赖建模中需要多层传播的问题。
  - 保留并强化**Attention**对全局依赖的建模能力。

- 作者希望降低序列中任意两个位置之间的信息传递路径长度。
  - 在**RNN**中，两个远距离位置之间的信息传递路径长度为**O(n)**。
  - 在**CNN**中，路径长度依赖卷积核大小和层数。
  - 在**Self-Attention**中，任意两个位置可以在单层内直接交互，路径长度为**O(1)**。

- 作者希望提升训练并行性。
  - **RNN**需要按时间步顺序计算，最少顺序操作数为**O(n)**。
  - **Self-Attention**可以对所有位置同时计算，最少顺序操作数为**O(1)**。
  - 这使得模型更适合现代GPU/TPU上的矩阵并行计算。

- 作者希望建立一种统一的序列转导架构。
  - 同一套结构可用于：
    - **Encoder self-attention**
    - **Decoder masked self-attention**
    - **Encoder-decoder attention**
  - 不依赖任务特定的复杂结构设计。
  - 能够从机器翻译推广到句法分析等其他任务。

- 作者还希望增强模型可解释性。
  - **Attention heads**可以显示模型在不同位置之间建立的关联。
  - 一些head能够捕捉：
    - 长距离依赖
    - 指代关系
    - 句法结构
    - 语义关系

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

---

**核心贡献**

- 提出**Transformer**架构。
  - 这是首个完全基于**Attention**的序列转导模型。
  - 完全去除了：
    - **Recurrent layers**
    - **Convolutional layers**
  - 使用堆叠的：
    - **Multi-Head Self-Attention**
    - **Position-wise Feed-Forward Network**
    - **Residual Connection**
    - **Layer Normalization**
    - **Positional Encoding**

- 提出并系统化使用**Scaled Dot-Product Attention**。
  - 核心公式为：
    - **Attention(Q,K,V)=softmax(QK^T/√d_k)V**
  - 缩放因子**1/√d_k**用于缓解高维点积过大导致的softmax梯度过小问题。
  - 相比**additive attention**，该机制更适合高效矩阵乘法实现。

- 提出并验证**Multi-Head Attention**的重要性。
  - 将**Query、Key、Value**投影到多个子空间。
  - 多个head并行关注不同位置和不同表示子空间的信息。
  - 避免单一attention head对信息进行过度平均。
  - 在base模型中使用：
    - **h=8**
    - **d_model=512**
    - **d_k=64**
    - **d_v=64**

- 引入**Positional Encoding**解决无递归、无卷积结构中的顺序信息问题。
  - 使用不同频率的**sine**和**cosine**函数表示位置。
  - 使模型能够利用token的绝对位置和相对位置信息。
  - 作者认为正弦位置编码可能具备更好的长度外推能力。
  - 实验显示，**sinusoidal positional encoding**与**learned positional embedding**效果接近。

- 从理论角度比较**Self-Attention、RNN、CNN**的计算特性。
  - 证明**Self-Attention**在并行性和路径长度上具有明显优势。
  - 对于机器翻译中常见的句长场景，当**n<d**时，self-attention的复杂度通常比RNN更有优势。

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---:|---:|---:|
| **Self-Attention** | **O(n²·d)** | **O(1)** | **O(1)** |
| **Recurrent** | **O(n·d²)** | **O(n)** | **O(n)** |
| **Convolutional** | **O(k·n·d²)** | **O(1)** | **O(log_k(n))** |
| **Restricted Self-Attention** | **O(r·n·d)** | **O(1)** | **O(n/r)** |

- 在机器翻译任务上取得当时的**state-of-the-art**结果。
  - 在**WMT 2014 English-to-German**任务上：
    - **Transformer big**达到**28.4 BLEU**
    - 比此前最佳结果，包括ensemble，提升超过**2 BLEU**
  - 在**WMT 2014 English-to-French**任务上：
    - **Transformer big**达到**41.8 BLEU**
    - 成为新的single-model最佳结果之一
  - 训练成本显著低于此前强模型。

| Model | EN-DE BLEU | EN-FR BLEU | EN-DE Training Cost |
|---|---:|---:|---:|
| **GNMT + RL** | 24.6 | 39.92 | 2.3×10¹⁹ |
| **ConvS2S** | 25.16 | 40.46 | 9.6×10¹⁸ |
| **MoE** | 26.03 | 40.56 | 2.0×10¹⁹ |
| **ConvS2S Ensemble** | 26.36 | 41.29 | 7.7×10¹⁹ |
| **Transformer base** | **27.3** | 38.1 | **3.3×10¹⁸** |
| **Transformer big** | **28.4** | **41.8** | 2.3×10¹⁹ |

- 显著提升训练效率。
  - **base model**：
    - 训练**100,000 steps**
    - 使用**8 NVIDIA P100 GPUs**
    - 总训练时间约**12小时**
  - **big model**：
    - 训练**300,000 steps**
    - 使用**8 NVIDIA P100 GPUs**
    - 总训练时间约**3.5天**
  - 相比此前模型，在相同或更好效果下训练成本更低。

- 通过模型变体实验验证架构设计的关键性。
  - **Multi-Head Attention**优于单头attention。
  - 过少或过多head都会降低效果。
  - 增大模型容量通常提升BLEU。
  - **Dropout**对防止过拟合非常重要。
  - **Label Smoothing**虽然会提高困惑度，但能提升准确率和BLEU。
  - **learned positional embedding**与**sinusoidal positional encoding**效果接近。

- 验证Transformer在非翻译任务上的泛化能力。
  - 作者将Transformer用于**English Constituency Parsing**。
  - 在缺乏大量任务特定调参的情况下，取得有竞争力的结果。
  - 在**WSJ only**设置下达到**91.3 F1**。
  - 在**semi-supervised**设置下达到**92.1 F1**。
  - 表明Transformer并非只适用于机器翻译，而是一种通用序列建模架构。

| Task | Setting | Transformer Result |
|---|---|---:|
| **English Constituency Parsing** | WSJ only | **91.3 F1** |
| **English Constituency Parsing** | Semi-supervised | **92.1 F1** |

---

**核心思想总结**

- **Transformer**的根本创新在于：
  - 用**Self-Attention**替代序列建模中长期占主导地位的**Recurrence**和**Convolution**。
  - 将序列中任意位置之间的依赖建模变成直接交互。
  - 用**Multi-Head Attention**弥补单头attention表达能力不足的问题。
  - 用**Positional Encoding**补充序列顺序信息。
  - 用高度并行化的矩阵计算显著降低训练时间。

- 该论文的影响力来自三点：
  - **架构范式转移**：从RNN/CNN主导转向Attention-only。
  - **性能突破**：在WMT机器翻译任务上刷新state-of-the-art。
  - **工程效率突破**：更高并行性、更低训练成本、更快收敛。

- 本文奠定了后续大规模语言模型和基础模型的核心架构基础。
  - 后续的**BERT、GPT、T5、ViT**等模型都直接或间接建立在Transformer思想之上。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体架构概览**

- 本文提出的核心架构是**Transformer**，一种完全基于**Attention mechanism**的序列到序列模型。
- Transformer采用经典的**Encoder-Decoder**框架，但彻底移除了传统序列建模中的：
  - **Recurrent Neural Network, RNN**
  - **Long Short-Term Memory, LSTM**
  - **Gated Recurrent Unit, GRU**
  - **Convolutional Neural Network, CNN**
- 模型依赖三类核心组件完成序列建模：
  - **Multi-Head Self-Attention**
  - **Position-wise Feed-Forward Network**
  - **Positional Encoding**
- Transformer的关键设计目标：
  - 提升训练并行度
  - 缩短长距离依赖的路径长度
  - 降低序列建模中的顺序计算瓶颈
  - 在机器翻译任务中获得更高BLEU和更低训练成本

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

---

**Encoder-Decoder主干结构**

- Transformer整体仍遵循**sequence transduction**中的**Encoder-Decoder**范式：
  - **Encoder**将输入序列映射为连续表示序列。
  - **Decoder**基于Encoder输出和已生成Token，自回归地产生目标序列。
- 输入序列表示为：
  - **x₁, ..., xₙ**
- Encoder输出表示为：
  - **z₁, ..., zₙ**
- Decoder逐步生成输出序列：
  - **y₁, ..., yₘ**
- Decoder是**auto-regressive**的：
  - 当前位置预测只能依赖此前已经生成的Token。
  - 通过**masking**阻止当前位置访问未来位置。

---

**Encoder架构**

- Encoder由**N=6**个完全相同的Layer堆叠而成。
- 每个Encoder Layer包含两个Sub-layer：
  - **Multi-Head Self-Attention**
  - **Position-wise Feed-Forward Network**
- 每个Sub-layer外部都使用：
  - **Residual Connection**
  - **Layer Normalization**
- Encoder Layer的标准计算形式为：
  - **LayerNorm(x + Sublayer(x))**
- 所有Sub-layer和Embedding输出维度统一为：
  - **d_model=512**
- Encoder中的Self-Attention特性：
  - 每个输入位置都可以直接关注输入序列中的所有位置。
  - 任意两个Token之间的信息交互路径长度为**O(1)**。
  - 适合捕获长距离依赖。

---

**Decoder架构**

- Decoder同样由**N=6**个完全相同的Layer堆叠而成。
- 每个Decoder Layer包含三个Sub-layer：
  - **Masked Multi-Head Self-Attention**
  - **Encoder-Decoder Multi-Head Attention**
  - **Position-wise Feed-Forward Network**
- 每个Sub-layer同样使用：
  - **Residual Connection**
  - **Layer Normalization**
- Decoder中的Masked Self-Attention：
  - 只允许当前位置关注当前位置及其之前的位置。
  - 通过将非法连接对应的softmax输入设置为**−∞**实现。
  - 保证生成过程满足**auto-regressive**约束。
- Encoder-Decoder Attention：
  - Query来自Decoder上一层。
  - Key和Value来自Encoder输出。
  - 使Decoder每个位置都能关注输入序列的所有位置。

---

**Attention核心机制**

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

- Transformer中的Attention被定义为：
  - 输入包含**Query, Key, Value**
  - 输出是Value的加权和
  - 权重由Query与Key之间的兼容性决定
- 本文采用**Scaled Dot-Product Attention**：
  - 使用Query与Key的点积衡量相关性。
  - 使用**√d_k**进行缩放，避免高维点积导致softmax梯度过小。
- 计算公式为：

| 组件 | 公式 |
|---|---|
| **Scaled Dot-Product Attention** | **Attention(Q,K,V)=softmax(QKᵀ/√d_k)V** |

- 缩放因子的作用：
  - 当**d_k**较大时，点积幅度可能过大。
  - softmax容易进入梯度极小区域。
  - 使用**1/√d_k**可以稳定训练。

---

**Multi-Head Attention结构**

- **Multi-Head Attention**是Transformer的核心模块。
- 模型不是只执行一次Attention，而是将Query、Key、Value投影到多个子空间中并行计算Attention。
- 每个Head学习不同的表示子空间和位置关系。
- 多个Head的输出会被拼接后再次线性投影。
- 公式结构为：

| 组件 | 说明 |
|---|---|
| **head_i** | **Attention(QW_i^Q, KW_i^K, VW_i^V)** |
| **MultiHead(Q,K,V)** | **Concat(head₁,...,head_h)W^O** |

- 本文Base Transformer的Multi-Head配置：
  - **h=8**
  - **d_k=64**
  - **d_v=64**
  - **d_model=512**
- Multi-Head Attention的优势：
  - 能同时关注不同位置的信息。
  - 能在不同表示子空间中建模关系。
  - 缓解单一Attention Head对信息加权平均造成的表达能力不足。

---

**三类Attention用法**

| Attention类型 | Query来源 | Key/Value来源 | 作用 |
|---|---|---|---|
| **Encoder Self-Attention** | Encoder上一层输出 | Encoder上一层输出 | 建模输入序列内部Token关系 |
| **Masked Decoder Self-Attention** | Decoder上一层输出 | Decoder上一层输出 | 建模已生成目标序列关系，阻止访问未来Token |
| **Encoder-Decoder Attention** | Decoder上一层输出 | Encoder最终输出 | 建模目标Token对源序列Token的对齐与依赖 |

---

**Position-wise Feed-Forward Network**

- 每个Encoder Layer和Decoder Layer都包含**Position-wise Feed-Forward Network, FFN**。
- FFN对每个位置独立、相同地应用。
- 结构为两层Linear Transformation，中间使用**ReLU**激活。
- 公式为：
  - **FFN(x)=max(0,xW₁+b₁)W₂+b₂**
- Base模型中的维度配置：
  - 输入与输出维度：**d_model=512**
  - 中间层维度：**d_ff=2048**
- FFN也可以理解为：
  - 两个kernel size为**1**的卷积层
  - 在每个Token位置上独立进行非线性变换

---

**Embedding与Softmax层**

- Transformer使用可学习的**Embedding**将输入Token和输出Token映射到向量空间。
- Embedding维度为：
  - **d_model=512**
- Decoder输出经过：
  - Linear Transformation
  - Softmax
  - 得到下一个Token的概率分布
- 本文采用权重共享策略：
  - 输入Embedding
  - 输出Embedding
  - Pre-softmax Linear Transformation
- Embedding权重会乘以：
  - **√d_model**

---

**Positional Encoding位置编码**

- Transformer没有RNN和CNN，因此模型本身不具备天然的序列顺序感知能力。
- 为注入位置信息，模型在Encoder和Decoder底部加入**Positional Encoding**。
- Positional Encoding与Token Embedding维度相同：
  - **d_model=512**
- 两者通过逐元素相加融合：
  - **Input Representation = Token Embedding + Positional Encoding**
- 本文主要使用固定的正弦和余弦位置编码：
  - 偶数维使用**sin**
  - 奇数维使用**cos**
- 设计动机：
  - 支持模型学习相对位置关系。
  - 对训练时未见过的更长序列可能具备外推能力。
- 作者也测试了**learned positional embeddings**：
  - 结果与sinusoidal positional encoding几乎相同。
  - 最终选择sinusoidal版本，主要因为其潜在长度外推能力。

---

**标准Base Transformer参数规格**

| 参数 | 数值 | 含义 |
|---|---:|---|
| **N** | **6** | Encoder和Decoder Layer数量 |
| **d_model** | **512** | 主表示维度 |
| **d_ff** | **2048** | FFN中间层维度 |
| **h** | **8** | Attention Head数量 |
| **d_k** | **64** | 每个Head的Key维度 |
| **d_v** | **64** | 每个Head的Value维度 |
| **P_drop** | **0.1** | Dropout比例 |
| **ε_ls** | **0.1** | Label Smoothing系数 |
| **参数量** | **约65M** | Base模型参数规模 |

---

**并行化与复杂度设计**

- Transformer通过Self-Attention替代RNN的顺序递归计算。
- Self-Attention的优势：
  - 所有Token位置可以并行计算。
  - 任意两个位置之间路径长度为**O(1)**。
  - 长距离依赖更容易建模。
- 与RNN和CNN相比，Transformer在机器翻译常见序列长度下具有更优的并行训练效率。

| Layer Type | 每层复杂度 | 顺序操作数 | 最大路径长度 |
|---|---|---|---|
| **Self-Attention** | **O(n²·d)** | **O(1)** | **O(1)** |
| **Recurrent** | **O(n·d²)** | **O(n)** | **O(n)** |
| **Convolutional** | **O(k·n·d²)** | **O(1)** | **O(logₖ(n))** |
| **Restricted Self-Attention** | **O(r·n·d)** | **O(1)** | **O(n/r)** |

- 当序列长度**n**小于表示维度**d**时：
  - Self-Attention通常比Recurrent Layer更高效。
- 对于非常长的序列：
  - 完整Self-Attention的**O(n²)**复杂度可能成为瓶颈。
  - 作者提出未来可研究**restricted self-attention**降低复杂度。

---

**训练架构与优化策略**

- 优化器使用**Adam**：
  - **β₁=0.9**
  - **β₂=0.98**
  - **ε=10⁻⁹**
- 学习率采用带Warmup的动态调度：
  - 前**4000**步线性升高。
  - 之后按step number的逆平方根下降。
- 正则化策略包括：
  - **Residual Dropout**
  - **Embedding + Positional Encoding Dropout**
  - **Label Smoothing**
- Label Smoothing的效果：
  - 会提高perplexity。
  - 但可改善accuracy和BLEU。

---

**架构运行流程**

- 输入端流程：
  - Source Token序列输入模型。
  - Token被映射为**Embedding**。
  - 加入**Positional Encoding**。
  - 进入多层Encoder Stack。
  - 每层执行Self-Attention和FFN。
  - 得到Encoder Memory表示。
- 输出端流程：
  - Target Token右移一位后输入Decoder。
  - Token被映射为**Embedding**。
  - 加入**Positional Encoding**。
  - 进入多层Decoder Stack。
  - 每层先执行Masked Self-Attention。
  - 再通过Encoder-Decoder Attention读取Encoder输出。
  - 再经过FFN。
  - Decoder最终输出经Linear和Softmax生成下一个Token概率。
- 推理阶段：
  - 使用**beam search**。
  - 论文中主要使用beam size为**4**。
  - 使用length penalty，**α=0.6**。

---

**技术架构核心价值**

- **Transformer**的整体技术架构可以概括为：
  - 用**Self-Attention**替代序列递归。
  - 用**Multi-Head Attention**增强多子空间建模能力。
  - 用**Positional Encoding**补足序列顺序信息。
  - 用**Residual Connection + Layer Normalization**稳定深层训练。
  - 用**Position-wise FFN**增强每个Token位置上的非线性表达。
- 架构层面的主要贡献：
  - 彻底摆脱RNN/CNN依赖。
  - 显著提高训练并行度。
  - 缩短长距离依赖建模路径。
  - 在机器翻译任务中以更低训练成本达到或超过当时State-of-the-Art结果。

### 1. Scaled Dot-Product Attention

**Scaled Dot-Product Attention核心定义**

- **Scaled Dot-Product Attention**是Transformer中的基础Attention计算单元，用于根据**Query**与**Key**之间的相似度，对**Value**进行加权聚合。

- 其计算公式为：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- 公式中的核心含义：
  - **Q**：Query矩阵，表示“当前要查询什么信息”。
  - **K**：Key矩阵，表示“每个位置可以被匹配的索引特征”。
  - **V**：Value矩阵，表示“被聚合的实际内容信息”。
  - **$d_k$**：Key和Query的特征维度。
  - **$QK^T$**：计算Query与所有Key之间的点积相似度。
  - **$\sqrt{d_k}$**：缩放因子，用于控制点积数值规模。
  - **softmax**：将相似度转换为概率分布。
  - **乘以V**：根据Attention权重对Value进行加权求和。

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

---

**输入输出关系**

| 符号 | 含义 | 典型形状 | 作用 |
|---|---|---:|---|
| **Q** | Query矩阵 | $n_q \times d_k$ | 当前位置发出的查询向量 |
| **K** | Key矩阵 | $n_k \times d_k$ | 被查询位置的匹配向量 |
| **V** | Value矩阵 | $n_k \times d_v$ | 被聚合的内容向量 |
| **$QK^T$** | 相似度矩阵 | $n_q \times n_k$ | 每个Query对每个Key的匹配分数 |
| **softmax($QK^T/\sqrt{d_k}$)** | Attention权重矩阵 | $n_q \times n_k$ | 每个Query对所有Value的权重分布 |
| **输出** | 加权后的表示 | $n_q \times d_v$ | 每个Query位置融合上下文后的向量 |

- 输入是三组向量集合：
  - **Queries**：需要更新表示的位置。
  - **Keys**：用于被匹配的位置。
  - **Values**：真正被读取和聚合的信息。

- 输出是一个新的表示矩阵：
  - 每一行对应一个Query位置。
  - 每个输出向量都是对所有Value向量的加权和。
  - 权重由该Query与所有Key的相似度决定。

---

**算法流程**

- **步骤1：线性投影得到Q、K、V**
  - 在Transformer中，输入Hidden States通常先经过不同的线性变换，得到：
    - **Q = XWQ**
    - **K = XWK**
    - **V = XWV**
  - 在不同Attention场景中，Q、K、V来源可能不同：
    - **Encoder Self-Attention**：Q、K、V都来自Encoder上一层输出。
    - **Decoder Masked Self-Attention**：Q、K、V都来自Decoder上一层输出，但需要Mask未来位置。
    - **Encoder-Decoder Attention**：Q来自Decoder，K和V来自Encoder输出。

- **步骤2：计算点积相似度**
  - 对每个Query向量$q_i$和Key向量$k_j$计算点积：

$$
score_{ij}=q_i \cdot k_j
$$

  - 矩阵形式为：

$$
S=QK^T
$$

  - 含义：
    - 点积越大，表示Query与Key越相关。
    - 点积越小，表示相关性越弱。

- **步骤3：按$\sqrt{d_k}$缩放**
  - 对相似度矩阵进行缩放：

$$
\tilde{S}=\frac{QK^T}{\sqrt{d_k}}
$$

  - 缩放的目的：
    - 防止$d_k$较大时点积数值过大。
    - 避免softmax进入饱和区域。
    - 保持梯度稳定。

- **步骤4：应用softmax得到Attention权重**
  - 对每个Query对应的一行分数执行softmax：

$$
A=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

  - softmax后的每一行满足：
    - 所有权重非负。
    - 权重和为1。
    - 表示当前Query对所有Key/Value位置的关注分布。

- **步骤5：加权聚合Value**
  - 使用Attention权重矩阵乘以Value矩阵：

$$
O=AV
$$

  - 得到输出：

$$
O=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

  - 输出中的每个向量都是所有Value的加权组合。

---

**为什么需要$\sqrt{d_k}$缩放**

- **核心问题**
  - 当$d_k$较大时，Query和Key的点积会随着维度增大而变大。
  - 点积值过大会使softmax输出变得极端：
    - 最大值位置接近1。
    - 其他位置接近0。
    - 分布过于尖锐。

- **梯度问题**
  - softmax在输入绝对值很大时容易进入饱和区。
  - 饱和区的梯度非常小。
  - 梯度过小会导致训练变慢或不稳定。

- **缩放逻辑**
  - 假设$q$和$k$的各维度独立，均值为0，方差为1。
  - 点积为：

$$
q \cdot k=\sum_{i=1}^{d_k} q_i k_i
$$

  - 其方差大约随$d_k$线性增长。
  - 使用$\sqrt{d_k}$除以点积后，可以将方差重新稳定到合理范围。

- **直观解释**
  - 不缩放时：
    - 高维向量点积可能过大。
    - softmax过早做出“极端选择”。
    - 模型难以通过梯度调整Attention分布。
  - 缩放后：
    - 相似度数值更平滑。
    - softmax分布更稳定。
    - 训练更容易收敛。

---

**与Additive Attention和普通Dot-Product Attention的差异**

| Attention类型                      | 兼容性函数                        | 是否缩放 | 计算特点         | 论文中的判断             |
| -------------------------------- | ---------------------------- | ---: | ------------ | ------------------ |
| **Additive Attention**           | 使用Feed-Forward Network计算匹配分数 |    否 | 表达能力较强，但计算较慢 | 在大$d_k$时可能优于未缩放点积  |
| **Dot-Product Attention**        | $QK^T$                       |    否 | 可用矩阵乘法高效实现   | 大$d_k$时softmax容易饱和 |
| **Scaled Dot-Product Attention** | $QK^T/\sqrt{d_k}$            |    是 | 高效且训练稳定      | Transformer采用的核心形式 |

- **Scaled Dot-Product Attention**保留了Dot-Product Attention的高效性。
- 通过$\sqrt{d_k}$缩放，缓解了大维度下的梯度问题。
- 相比Additive Attention，它更适合GPU/TPU上的大规模矩阵并行计算。

---

**参数设置**

| 参数 | 论文Base Model设置 | 说明 |
|---|---:|---|
| **$d_{model}$** | 512 | Transformer中Hidden State和Embedding的主维度 |
| **h** | 8 | Multi-Head Attention中的Head数量 |
| **$d_k$** | 64 | 每个Head中Key和Query的维度 |
| **$d_v$** | 64 | 每个Head中Value的维度 |
| **缩放因子** | $\sqrt{64}=8$ | 每个Head中点积结果除以8 |
| **Attention类型** | Scaled Dot-Product Attention | 每个Head内部使用该机制 |

- Base Model中：
  - $d_{model}=512$
  - $h=8$
  - 每个Head的维度为：

$$
d_k=d_v=d_{model}/h=512/8=64
$$

  - 因此每个Head中的Attention计算为：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{8}\right)V
$$

- 这种设计使总计算量接近单个全维度Attention：
  - 单Head全维度Attention使用512维。
  - Multi-Head Attention把512维拆成8个64维子空间。
  - 多个Head并行计算后再拼接。

---

**在Transformer整体架构中的作用**

- **作为Multi-Head Attention的内部计算单元**
  - Transformer并不是只计算一次Scaled Dot-Product Attention。
  - 它会在多个Head中并行执行该计算。
  - 每个Head使用不同的线性投影矩阵，学习不同的关系模式。

- **在Encoder中的作用**
  - Encoder Self-Attention中：
    - 每个Token都可以关注输入序列中的所有Token。
    - 每个位置通过Attention直接融合全局上下文。
    - 长距离依赖路径长度为**O(1)**。

- **在Decoder中的作用**
  - Decoder Masked Self-Attention中：
    - 每个位置只能关注当前位置及之前位置。
    - 未来位置会被Mask掉。
    - 保证自回归生成过程合法。

- **在Encoder-Decoder Attention中的作用**
  - Query来自Decoder。
  - Key和Value来自Encoder输出。
  - Decoder每个生成位置可以关注源语言输入的所有位置。
  - 对机器翻译中的对齐关系非常关键。

---

**Mask机制**

- 在Decoder Self-Attention中，需要防止当前位置看到未来Token。
- 实现方式：
  - 对非法位置的Attention Score加上负无穷：

$$
score_{ij}=-\infty,\quad j>i
$$

  - softmax之后，这些位置的权重变为0。
  - 当前Token只能利用历史Token信息。

- Mask后的计算形式可写为：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T+M}{\sqrt{d_k}}\right)V
$$

- 其中：
  - **M**是Mask矩阵。
  - 合法位置为0。
  - 非法位置为$-\infty$。

---

**计算复杂度与并行性**

| 层类型 | 每层复杂度 | 顺序操作数 | 最大路径长度 |
|---|---:|---:|---:|
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| **Recurrent** | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| **Convolutional** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| **Restricted Self-Attention** | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

- **Self-Attention的优势**
  - 所有Token之间可以直接建立联系。
  - 不需要像RNN一样按时间步递归计算。
  - 训练时可以高度并行。
  - 长距离依赖的路径长度为常数级。

- **Self-Attention的代价**
  - 相似度矩阵大小为$n \times n$。
  - 序列长度$n$很大时，计算和显存开销会显著增加。
  - 论文中也提到可通过Restricted Self-Attention降低长序列成本。

---

**数值示例**

- 假设：
  - 序列长度$n=4$
  - 每个Head中$d_k=64$
  - 每个Head中$d_v=64$

- 输入形状：
  - $Q \in \mathbb{R}^{4 \times 64}$
  - $K \in \mathbb{R}^{4 \times 64}$
  - $V \in \mathbb{R}^{4 \times 64}$

- 计算过程：
  - $QK^T \in \mathbb{R}^{4 \times 4}$
  - 除以$\sqrt{64}=8$
  - softmax后得到Attention权重矩阵$A \in \mathbb{R}^{4 \times 4}$
  - $AV \in \mathbb{R}^{4 \times 64}$

- 输出含义：
  - 每个Token输出一个64维向量。
  - 每个向量融合了序列中所有Token的Value信息。
  - 融合权重由Query-Key相似度决定。

---

**实现层面的伪代码**

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1)
    scores = scores / sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores, dim=-1)
    output = weights @ V

    return output, weights
```

- 实现关键点：
  - **矩阵乘法**用于高效计算所有Query-Key对的相似度。
  - **缩放因子**必须在softmax之前应用。
  - **Mask**也必须在softmax之前加入。
  - **softmax维度**通常是最后一维，即对每个Query的所有Key归一化。
  - 输出包括：
    - **output**：上下文融合后的表示。
    - **weights**：Attention权重，可用于分析模型关注模式。

---

**在Multi-Head Attention中的组合方式**

- 单个Head执行：

$$
\mathrm{head_i}=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

- 多个Head并行执行后拼接：

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head_1},...,\mathrm{head_h})W^O
$$

- 关键作用：
  - 不同Head可以关注不同位置关系。
  - 不同Head可以学习不同表示子空间。
  - 避免单一Attention Head对信息做过度平均。

- 论文中的典型设置：
  - **h=8**
  - **$d_k=d_v=64$**
  - 拼接后维度为：

$$
8 \times 64=512
$$

  - 再通过$W^O$投影回$d_{model}=512$。

---

**核心价值**

- **Scaled Dot-Product Attention**解决了Transformer中的关键建模问题：
  - 如何让任意两个Token直接交互。
  - 如何并行计算全局依赖。
  - 如何在高维点积Attention中保持训练稳定。

- 它的技术贡献集中在三点：
  - **高效性**：基于矩阵乘法，适合GPU并行。
  - **稳定性**：通过$\sqrt{d_k}$缩放避免softmax饱和。
  - **表达性**：与Multi-Head Attention结合后，可同时建模多种依赖关系。

- 在Transformer整体中，它是：
  - **Encoder Self-Attention**的计算核心。
  - **Decoder Masked Self-Attention**的计算核心。
  - **Encoder-Decoder Attention**的计算核心。
  - 支撑Transformer摆脱RNN和CNN的基础机制。

### 2. Multi-Head Attention

**Multi-Head Attention核心定义**

- **Multi-Head Attention**是Transformer中的核心计算单元，用于让模型在多个**低维表示子空间**中并行执行**Scaled Dot-Product Attention**。
- 它不是只用一个Attention函数处理完整的**Query、Key、Value**，而是：
  - 将**Q、K、V**分别通过多组可学习线性映射投影到多个子空间。
  - 在每个子空间中独立计算Attention。
  - 将多个Attention head的结果拼接。
  - 再通过一个输出线性变换得到最终表示。
- 论文中的关键动机：
  - 单个Attention head会对所有被关注位置做加权平均，容易造成信息混合。
  - 多个head可以让模型同时关注：
    - 不同位置的Token。
    - 不同类型的依赖关系。
    - 不同表示子空间中的语义、句法或对齐信息。

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

---

**数学形式**

- Multi-Head Attention的整体公式为：

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O
$$

- 每个head的计算为：

$$
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

- 其中单个Attention使用**Scaled Dot-Product Attention**：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- 参数矩阵含义：
  - $W_i^Q$：第$i$个head的**Query投影矩阵**。
  - $W_i^K$：第$i$个head的**Key投影矩阵**。
  - $W_i^V$：第$i$个head的**Value投影矩阵**。
  - $W^O$：多头拼接后的**输出投影矩阵**。

---

**输入输出关系**

| 符号                | 含义                       |                   典型形状 | 作用                     |
| ----------------- | ------------------------ | ---------------------: | ---------------------- |
| **Q**             | Query矩阵                  | $n_q \times d_{model}$ | 表示当前位置“要查询什么”          |
| **K**             | Key矩阵                    | $n_k \times d_{model}$ | 表示每个位置“可被匹配的索引特征”      |
| **V**             | Value矩阵                  | $n_k \times d_{model}$ | 表示每个位置“真正被聚合的信息内容”     |
| **$W_i^Q$**       | Query线性投影                | $d_{model} \times d_k$ | 将Q映射到第$i$个head的子空间     |
| **$W_i^K$**       | Key线性投影                  | $d_{model} \times d_k$ | 将K映射到第$i$个head的子空间     |
| **$W_i^V$**       | Value线性投影                | $d_{model} \times d_v$ | 将V映射到第$i$个head的子空间     |
| **head$_i$**      | 单个Attention head输出       |       $n_q \times d_v$ | 一个子空间中的上下文表示           |
| **Concat(heads)** | 多head拼接结果                |     $n_q \times h d_v$ | 汇总所有head的信息            |
| **输出**            | Multi-Head Attention最终输出 | $n_q \times d_{model}$ | 输入到残差连接、LayerNorm或后续子层 |

- 对于Transformer base设置：
  - **$d_{model}=512$**。
  - **$h=8$**。
  - **$d_k=d_v=d_{model}/h=64$**。
  - 每个head只在64维子空间中计算Attention。
  - 8个head拼接后得到$8 \times 64=512$维，再通过$W^O$映射回512维。

---

**算法流程**

- 输入阶段：
  - 接收三个矩阵：**Q、K、V**。
  - 在不同应用场景中，Q、K、V来源可能不同：
    - Encoder self-attention：Q、K、V都来自Encoder上一层输出。
    - Decoder masked self-attention：Q、K、V都来自Decoder上一层输出，但加入未来位置mask。
    - Encoder-decoder attention：Q来自Decoder上一层，K和V来自Encoder输出。

- 多头线性投影：
  - 对每个head $i$，执行三组独立线性变换：
    - $Q_i=QW_i^Q$
    - $K_i=KW_i^K$
    - $V_i=VW_i^V$
  - 每个head拥有独立参数，因此可以学习不同的匹配模式。

- 并行Attention计算：
  - 对每个head计算：
    - $Q_iK_i^T$：得到Query与Key之间的相似度矩阵。
    - 除以$\sqrt{d_k}$：控制点积幅度，避免softmax进入梯度过小区域。
    - softmax：得到每个Query对所有Key位置的Attention权重。
    - 乘以$V_i$：按权重聚合Value信息。
  - 所有head可以并行计算，充分利用GPU矩阵乘法能力。

- 拼接与输出映射：
  - 将所有head输出沿特征维拼接：
    - $\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)$
  - 通过输出矩阵$W^O$做线性变换：
    - 将多头信息重新混合。
    - 恢复到$d_{model}$维度。
    - 形成可传递给后续子层的表示。

---

**参数设置**

| 配置项 | Transformer base取值 | 解释 |
|---|---:|---|
| **Encoder/Decoder层数N** | 6 | 每个Encoder和Decoder都堆叠6层 |
| **$d_{model}$** | 512 | Token表示、子层输出、残差连接的统一维度 |
| **head数量h** | 8 | 并行Attention head数量 |
| **$d_k$** | 64 | 每个head中Query和Key的维度 |
| **$d_v$** | 64 | 每个head中Value的维度 |
| **拼接后维度** | 512 | $h \times d_v = 8 \times 64$ |
| **输出投影矩阵$W^O$** | $512 \times 512$ | 将拼接后的多头结果映射回$d_{model}$ |
| **Dropout** | 0.1 | base model中用于正则化 |

- 参数设计的核心平衡：
  - 若使用单头Attention：
    - 每个head维度是512。
    - 只有一个Attention模式。
  - 若使用8头Attention：
    - 每个head维度是64。
    - 总计算成本与单头512维Attention相近。
    - 表达能力更强，因为有8组独立的注意力分布。

---

**为什么要缩放$\sqrt{d_k}$**

- Dot-Product Attention中，Query和Key的点积会随维度$d_k$增大而变大。
- 点积值过大时：
  - softmax输出会变得非常尖锐。
  - 大部分概率接近0，少数位置接近1。
  - 梯度容易变小，训练不稳定。
- 使用$\frac{1}{\sqrt{d_k}}$缩放：
  - 抑制点积幅度。
  - 保持softmax分布在更合适的范围。
  - 改善大维度Key/Query下的训练稳定性。

---

**在Transformer整体架构中的位置**

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

- 在**Encoder**中：
  - 每层包含一个**Multi-Head Self-Attention**子层。
  - 每个Token都可以直接关注输入序列中的所有Token。
  - 作用是构建上下文相关的输入表示。
  - 输出再进入**Position-wise Feed-Forward Network**。

- 在**Decoder**中：
  - 每层包含两个与Attention相关的子层：
    - **Masked Multi-Head Self-Attention**。
    - **Encoder-Decoder Multi-Head Attention**。
  - Masked self-attention保证自回归生成：
    - 当前位置只能关注当前位置及其之前的Token。
    - 不能看到未来Token。
  - Encoder-decoder attention用于跨序列对齐：
    - Decoder当前生成状态作为Query。
    - Encoder输出作为Key和Value。
    - 使每个输出Token可以关注输入句子的相关位置。

- 在每个Attention子层外部：
  - 使用**Residual Connection**。
  - 使用**Layer Normalization**。
  - 形式为：
    - $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$

---

**三种应用场景对比**

| 场景 | Q来源 | K来源 | V来源 | Mask | 主要作用 |
|---|---|---|---|---|---|
| **Encoder Self-Attention** | Encoder上一层 | Encoder上一层 | Encoder上一层 | 否 | 建模输入序列内部依赖 |
| **Decoder Masked Self-Attention** | Decoder上一层 | Decoder上一层 | Decoder上一层 | 是 | 建模已生成目标序列上下文，防止看到未来Token |
| **Encoder-Decoder Attention** | Decoder上一层 | Encoder输出 | Encoder输出 | 通常无未来mask | 建立目标端到源端的对齐关系 |

---

**Multi-Head的表达优势**

- **并行建模不同关系**
  - 一个head可以关注主谓关系。
  - 一个head可以关注代词指代。
  - 一个head可以关注短语边界。
  - 一个head可以关注远距离依赖。

- **缓解单头Attention的信息平均问题**
  - 单头Attention输出是对Value的加权平均。
  - 多个重要位置被混合后，细粒度信息可能被稀释。
  - 多头机制允许不同head保留不同聚合视角。

- **提升长距离依赖建模能力**
  - 任意两个位置之间只需一次Attention即可建立联系。
  - 最大路径长度是**O(1)**。
  - 相比RNN的**O(n)**路径更短，更利于梯度传播。

- **增强可解释性**
  - 论文附录中的Attention可视化显示：
    - 某些head学习到长距离依赖。
    - 某些head表现出指代消解能力。
    - 某些head关注句法结构。

---

**与单头Attention的计算对比**

| 方案 | head数量 | 每个head维度 | 总输出维度 | 表达能力 | 计算成本 |
|---|---:|---:|---:|---|---|
| **Single-Head Attention** | 1 | 512 | 512 | 单一注意力模式 | 与base多头接近 |
| **Multi-Head Attention** | 8 | 64 | 512 | 多个注意力模式并行 | 与单头全维度相近 |

- 关键点：
  - Multi-Head不是简单增加8倍计算量。
  - 每个head的维度被降低到$d_{model}/h$。
  - 总体计算量近似保持不变。
  - 表达空间明显更丰富。

---

**Table 3中的head数量实验结论**

| 设置         |   h | $d_k$ | $d_v$ |  PPL | BLEU | 结论       |
| ---------- | --: | ----: | ----: | ---: | ---: | -------- |
| **base**   |   8 |    64 |    64 | 4.92 | 25.8 | 默认最均衡    |
| **单头**     |   1 |   512 |   512 | 5.29 | 24.9 | BLEU下降明显 |
| **较少head** |   4 |   128 |   128 | 5.00 | 25.5 | 略差于base  |
| **更多head** |  16 |    32 |    32 | 4.91 | 25.8 | 与base接近  |
| **过多head** |  32 |    16 |    16 | 5.01 | 25.4 | 质量下降     |

- 实验说明：
  - **单头Attention比最佳设置低0.9 BLEU**。
  - head数量太少会限制模型关注多种关系的能力。
  - head数量太多会导致每个head维度过小，单个head表达能力下降。
  - **8个head**在论文base模型中是性能和计算的良好折中。

---

**Key/Value维度的影响**

- 论文在Table 3中还观察到：
  - 降低**$d_k$**会损害模型质量。
  - 说明计算Query-Key兼容性并不简单。
  - 过低的Key维度会削弱匹配能力。
- 原因：
  - Attention权重由$QK^T$决定。
  - $d_k$太小会压缩匹配特征。
  - Query与Key之间难以表达复杂语义关系。
- 启示：
  - Dot product虽然高效，但兼容性函数的表达能力仍受维度影响。
  - 更复杂的匹配函数可能进一步提升性能，但会牺牲计算效率。

---

**复杂度与并行性**

| 层类型 | 每层复杂度 | 顺序操作数 | 最大路径长度 |
|---|---:|---:|---:|
| **Self-Attention** | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| **Recurrent** | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| **Convolutional** | $O(knd^2)$ | $O(1)$ | $O(\log_k(n))$ |
| **Restricted Self-Attention** | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

- Multi-Head Attention继承Self-Attention的关键优势：
  - **高度并行**：所有位置可以同时计算。
  - **路径短**：任意两个Token之间可直接交互。
  - **适合机器翻译中的中等长度序列**：通常序列长度$n$小于表示维度$d$。

- 主要代价：
  - Self-Attention复杂度包含$n^2$，长序列成本较高。
  - 论文也提到可用**restricted self-attention**限制局部窗口，降低长序列开销。

---

**实现级伪流程**

- 输入：
  - $X_Q \in \mathbb{R}^{B \times n_q \times d_{model}}$
  - $X_K \in \mathbb{R}^{B \times n_k \times d_{model}}$
  - $X_V \in \mathbb{R}^{B \times n_k \times d_{model}}$

- 线性投影：
  - $Q=X_QW^Q$
  - $K=X_KW^K$
  - $V=X_VW^V$

- reshape为多头形式：
  - $Q \rightarrow B \times h \times n_q \times d_k$
  - $K \rightarrow B \times h \times n_k \times d_k$
  - $V \rightarrow B \times h \times n_k \times d_v$

- 计算Attention score：
  - $S=QK^T/\sqrt{d_k}$
  - $S \in \mathbb{R}^{B \times h \times n_q \times n_k}$

- 应用mask：
  - Decoder self-attention中，将未来位置score设为$-\infty$。
  - Padding位置也通常需要mask，避免模型关注无效Token。

- 归一化权重：
  - $A=\mathrm{softmax}(S)$

- 聚合Value：
  - $O=A V$
  - $O \in \mathbb{R}^{B \times h \times n_q \times d_v}$

- 拼接heads：
  - $O \rightarrow B \times n_q \times h d_v$

- 输出投影：
  - $Y=OW^O$
  - $Y \in \mathbb{R}^{B \times n_q \times d_{model}}$

---

**在模型语义层面的作用**

- **在Encoder中**
  - 将每个Token从孤立Embedding转化为上下文相关表示。
  - 例如句中某个动词可以直接关注远处的宾语、修饰语或从句成分。
  - 对机器翻译而言，这有助于理解源语言句子的整体结构。

- **在Decoder中**
  - 让目标端已生成Token之间建立上下文依赖。
  - 通过mask确保生成过程满足自回归约束。
  - 使当前位置生成只依赖历史输出。

- **在Encoder-Decoder交互中**
  - 负责源语言与目标语言之间的软对齐。
  - 每个目标端位置可以动态选择关注源端哪些Token。
  - 替代传统Seq2Seq中的RNN attention机制。

---

**关键技术价值**

- **去除RNN的顺序瓶颈**
  - 不再按Token逐步递归计算。
  - 训练时可对整个序列并行处理。

- **替代卷积的局部感受野限制**
  - 单层即可连接任意两个位置。
  - 不需要堆叠多层卷积来扩大感受野。

- **提升表示多样性**
  - 不同head学习不同语义或结构模式。
  - 多个子空间共同构成更强的上下文建模能力。

- **保持计算效率**
  - 每个head使用低维投影。
  - 总体计算成本接近单头全维Attention。
  - 适合GPU上的批量矩阵乘法实现。

- **成为Transformer的基础模块**
  - 后续几乎所有Transformer变体，包括BERT、GPT、T5、ViT等，均以Multi-Head Attention为核心组件。

### 3. Encoder-Decoder Stack without Recurrence or Convolution

**核心定位**

- **Encoder-Decoder Stack without Recurrence or Convolution**是 Transformer 的结构核心：
  - 使用**stacked self-attention**和**position-wise fully connected layers**构建 Encoder 与 Decoder。
  - 完全移除传统序列建模中的**RNN recurrence**和**CNN convolution**。
  - 序列中任意两个 Token 之间的信息交互不再依赖时间步递归或卷积感受野扩展，而是通过**Self-Attention**一次性建立全局连接。
  - 该设计使模型具备：
    - **更强并行性**
    - **更短长程依赖路径**
    - **更低训练时间成本**
    - **更灵活的全局上下文建模能力**

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

---

**整体架构关系**

- Transformer 仍然遵循经典的**Encoder-Decoder**范式：
  - Encoder 接收输入序列：
    - $(x_1, x_2, ..., x_n)$
  - Encoder 输出连续表示序列：
    - $\mathbf{z} = (z_1, z_2, ..., z_n)$
  - Decoder 基于 Encoder 输出和已生成目标 Token，自回归地产生输出序列：
    - $(y_1, y_2, ..., y_m)$

- 与传统 Encoder-Decoder 的关键区别：
  - 传统 Seq2Seq：
    - Encoder 通常由**RNN/LSTM/GRU**或**CNN**构成。
    - Decoder 通常逐步递归生成。
    - Attention 多作为 RNN/CNN 之上的辅助模块。
  - Transformer：
    - Encoder 内部完全由**Multi-Head Self-Attention + Feed-Forward Network**堆叠而成。
    - Decoder 内部由**Masked Multi-Head Self-Attention + Encoder-Decoder Attention + Feed-Forward Network**堆叠而成。
    - Attention 不再是辅助机制，而是主干计算机制。

---

**Encoder Stack 实现原理**

- Encoder 由 **N = 6** 个完全相同的 Layer 堆叠组成。
- 每个 Encoder Layer 包含两个核心 Sub-layer：
  - **Multi-Head Self-Attention**
  - **Position-wise Feed-Forward Network**

- 每个 Sub-layer 外部都包裹：
  - **Residual Connection**
  - **Layer Normalization**

- Encoder Layer 的标准计算形式为：
  - 对任意 Sub-layer：
    - $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$

- Encoder 的输入处理流程：
  - 输入 Token 被映射为 **Embedding**。
  - Embedding 加上 **Positional Encoding**。
  - 进入第 1 个 Encoder Layer。
  - 每层输出作为下一层输入。
  - 第 6 层输出作为整个 Encoder 的 memory，供 Decoder 的 Encoder-Decoder Attention 使用。

- Encoder 的每个位置都可以直接关注输入序列中的所有位置：
  - 对于输入位置 $i$，Self-Attention 可以访问位置 $1...n$。
  - 不存在递归顺序限制。
  - 不需要通过卷积层逐步扩大 receptive field。
  - 任意两个 Token 之间的最大路径长度为 **O(1)**。

---

**Decoder Stack 实现原理**

- Decoder 同样由 **N = 6** 个完全相同的 Layer 堆叠组成。
- 每个 Decoder Layer 包含三个核心 Sub-layer：
  - **Masked Multi-Head Self-Attention**
  - **Encoder-Decoder Multi-Head Attention**
  - **Position-wise Feed-Forward Network**

- 每个 Sub-layer 同样使用：
  - **Residual Connection**
  - **Layer Normalization**

- Decoder 的输入处理流程：
  - 目标端已生成 Token 右移一位后输入 Decoder。
  - Token 被映射为 **Output Embedding**。
  - Output Embedding 加上 **Positional Encoding**。
  - 进入第 1 个 Decoder Layer。
  - 每层先进行 Masked Self-Attention，再关注 Encoder 输出，最后通过 FFN。
  - 最终 Decoder 输出经过 Linear + Softmax 得到下一个 Token 的概率分布。

- Decoder 保持**auto-regressive**性质：
  - 预测位置 $i$ 时，只能使用位置 $< i$ 的目标端 Token。
  - 通过 Mask 将未来位置对应的 Attention logits 设为 $-\infty$。
  - Softmax 后未来位置权重变为 0。
  - 该机制替代了 RNN 中天然的时间顺序约束。

---

**Encoder 与 Decoder 的结构对比**

| 模块 | Encoder Layer | Decoder Layer |
|---|---:|---:|
| Layer 数量 | **6** | **6** |
| Self-Attention | **Multi-Head Self-Attention** | **Masked Multi-Head Self-Attention** |
| Encoder-Decoder Attention | 无 | **有** |
| Feed-Forward Network | **有** | **有** |
| Residual Connection | **有** | **有** |
| Layer Normalization | **有** | **有** |
| 是否可看见未来 Token | 输入端可全局可见 | 目标端不可看见未来 Token |
| 是否使用 recurrence | **否** | **否** |
| 是否使用 convolution | **否** | **否** |

---

**关键参数设置**

| 参数                   |       数值 | 作用                               |
| -------------------- | -------: | -------------------------------- |
| Encoder 层数 $N$       |    **6** | 堆叠多层抽象表示                         |
| Decoder 层数 $N$       |    **6** | 堆叠多层生成条件表示                       |
| $d_{\mathrm{model}}$ |  **512** | 所有 Sub-layer 与 Embedding 的统一表示维度 |
| Attention heads $h$  |    **8** | 并行关注不同表示子空间                      |
| $d_k$                |   **64** | 每个 head 的 Key 维度                 |
| $d_v$                |   **64** | 每个 head 的 Value 维度               |
| $d_{\mathrm{ff}}$    | **2048** | FFN 中间层维度                        |
| Dropout              |  **0.1** | Base model 正则化                   |
| Label smoothing      |  **0.1** | 提升泛化与 BLEU                       |
| Warmup steps         | **4000** | Learning rate warmup             |

---

**为什么必须统一 $d_{\mathrm{model}} = 512$**

- Residual Connection 要求输入与输出维度一致：
  - Sub-layer 输入为 $x \in \mathbb{R}^{512}$。
  - Sub-layer 输出也必须为 $\mathbb{R}^{512}$。
  - 才能执行 $x + \mathrm{Sublayer}(x)$。

- Encoder、Decoder、Embedding、Attention、FFN 的维度统一带来：
  - 简化模块接口。
  - 支持深层堆叠。
  - 避免频繁维度变换。
  - 保证 Encoder 输出可以直接作为 Decoder cross-attention 的 Key 和 Value 来源。

---

**Self-Attention 替代 Recurrence 的机制**

- RNN 的信息传播方式：
  - 位置 $t$ 的 hidden state 依赖 $h_{t-1}$。
  - 长距离依赖需要穿过多个时间步。
  - 并行性差，训练时序列内部必须顺序计算。

- Transformer Encoder 的信息传播方式：
  - 所有位置同时生成 Query、Key、Value。
  - 每个位置通过 Attention 权重直接聚合全序列信息。
  - 任意位置之间只需一次 Attention 计算即可交互。
  - 序列维度上可并行计算。

- 核心计算公式：
  - $$
    \mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$

- 对 Encoder 而言：
  - $Q$、$K$、$V$ 都来自同一输入序列。
  - 这是典型的 **self-attention**。

- 对 Decoder 的 Masked Self-Attention 而言：
  - $Q$、$K$、$V$ 都来自目标端历史 Token。
  - 加入 causal mask 防止访问未来 Token。

---

**Convolution 被移除后的影响**

- CNN 序列模型通过局部卷积建模上下文：
  - 单层卷积只能覆盖局部窗口。
  - 要连接远距离 Token，需要堆叠多层或使用 dilated convolution。
  - 最大路径长度随序列长度或卷积核结构增长。

- Transformer 不依赖卷积感受野：
  - Self-Attention 单层即可连接任意两个位置。
  - 不需要 kernel size。
  - 不需要 dilation。
  - 不需要池化或滑动窗口。
  - 对句子中远距离依赖更直接。

- 代价：
  - Self-Attention 的复杂度为 **O(n²·d)**。
  - 当序列很长时，Attention matrix 会带来较高显存和计算压力。
  - 原论文也提到未来可探索 **restricted self-attention** 来处理超长输入，如 image、audio、video。

---

**复杂度与路径长度对比**

| Layer Type                    |         每层复杂度 |    顺序操作数 |         最大路径长度 |
| ----------------------------- | ------------: | -------: | -------------: |
| **Self-Attention**            |   **O(n²·d)** | **O(1)** |       **O(1)** |
| **Recurrent**                 |   **O(n·d²)** | **O(n)** |       **O(n)** |
| **Convolutional**             | **O(k·n·d²)** | **O(1)** | **O(logₖ(n))** |
| **Restricted Self-Attention** |  **O(r·n·d)** | **O(1)** |     **O(n/r)** |

- Self-Attention 的优势：
  - 顺序操作数为 **O(1)**，训练并行性强。
  - 最大路径长度为 **O(1)**，远距离依赖更易学习。
  - 当序列长度 $n < d$ 时，Self-Attention 通常比 RNN 更高效。
  - 在机器翻译中，句子 Token 数通常小于表示维度 $d_{\mathrm{model}}$，因此适合使用全局 Self-Attention。

---

**Multi-Head Attention 在 Stack 中的作用**

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

- 单头 Attention 的问题：
  - 所有信息被压缩到一个 Attention 分布中。
  - 容易出现信息平均化。
  - 不同语义关系、句法关系、对齐关系难以同时表达。

- Multi-Head Attention 的解决方式：
  - 将 $Q$、$K$、$V$ 线性投影到多个子空间。
  - 每个 head 独立执行 Scaled Dot-Product Attention。
  - 所有 head 的输出拼接后再线性变换。
  - 允许模型同时关注：
    - 局部短语结构
    - 长距离依赖
    - 主谓关系
    - 指代关系
    - 目标端与源端的翻译对齐关系

- 在 base model 中：
  - $h = 8$
  - $d_k = d_v = 64$
  - $h \cdot d_v = 8 \times 64 = 512$
  - 拼接后维度仍与 $d_{\mathrm{model}}$ 一致。

---

**Position-wise Feed-Forward Network 的作用**

- 每个 Encoder Layer 和 Decoder Layer 都包含 FFN。
- FFN 对每个位置独立应用相同的两层 MLP：
  - $$
    \mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
    $$

- FFN 的功能：
  - Attention 负责 Token 间信息混合。
  - FFN 负责每个 Token 表示内部的非线性变换。
  - 类似对每个位置执行相同的 feature transformation。
  - 可视为 kernel size 为 1 的卷积，但不使用传统卷积结构。

- 维度变化：
  - 输入：$512$
  - 中间层：$2048$
  - 输出：$512$

| 阶段 | 维度 |
|---|---:|
| FFN 输入 | **512** |
| 第一层 Linear 后 | **2048** |
| ReLU 激活后 | **2048** |
| 第二层 Linear 后 | **512** |
| Residual 相加后 | **512** |

---

**Positional Encoding 的必要性**

- Transformer 移除了 recurrence 和 convolution 后，模型本身没有天然顺序感。
- Self-Attention 对输入集合近似是 permutation-invariant 的：
  - 如果没有位置编码，模型难以区分 Token 的顺序。
  - 例如 “dog bites man” 与 “man bites dog” 会缺乏顺序区分依据。

- 论文使用 sinusoidal positional encoding：
  - $$
    PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{\mathrm{model}}})
    $$
  - $$
    PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{\mathrm{model}}})
    $$

- Positional Encoding 的作用：
  - 注入绝对位置信息。
  - 使模型可学习相对位置关系。
  - 与 Embedding 维度相同，可直接相加。
  - 理论上有助于泛化到比训练时更长的序列。

- 输入到 Encoder 或 Decoder 底部的实际表示：
  - $$
    \mathrm{Input} = \mathrm{TokenEmbedding} + \mathrm{PositionalEncoding}
    $$

---

**Encoder 输入输出关系**

- Encoder 输入：
  - 源语言 Token 序列。
  - 经 Byte-Pair Encoding 或 WordPiece 处理后的 subword Token。
  - 每个 Token 被映射为 $d_{\mathrm{model}} = 512$ 的向量。
  - 加入 Positional Encoding 后形成输入矩阵：
    - $X \in \mathbb{R}^{n \times 512}$

- Encoder 输出：
  - 每个输入位置对应一个上下文化表示：
    - $Z \in \mathbb{R}^{n \times 512}$
  - $Z$ 同时包含：
    - Token 自身语义
    - 全局上下文信息
    - 位置关系
    - 多层 Attention 聚合后的抽象特征

- Encoder 输出在整体模型中的作用：
  - 作为 Decoder 的 memory。
  - 在 Encoder-Decoder Attention 中提供 Key 和 Value。
  - 帮助 Decoder 在生成每个目标 Token 时对源句进行动态对齐。

---

**Decoder 输入输出关系**

- Decoder 输入：
  - 目标语言已生成 Token 序列。
  - 训练时使用右移后的 ground-truth target sequence。
  - 推理时使用模型已生成的历史 Token。
  - 输入矩阵：
    - $Y_{\text{shifted}} \in \mathbb{R}^{m \times 512}$

- Decoder 内部信息来源：
  - Masked Self-Attention：
    - 来自目标端历史 Token。
  - Encoder-Decoder Attention：
    - Query 来自 Decoder。
    - Key 和 Value 来自 Encoder 输出 $Z$。
  - FFN：
    - 对每个目标位置进行非线性特征变换。

- Decoder 输出：
  - 每个目标位置输出一个 $512$ 维表示。
  - 经 Linear 映射到 vocabulary size。
  - 经 Softmax 得到下一个 Token 概率：
    - $P(y_i | y_{<i}, x)$

- Decoder 在整体模型中的作用：
  - 将 Encoder 的源语言表示转化为目标语言序列。
  - 通过 Mask 保持自回归生成。
  - 通过 Encoder-Decoder Attention 实现源端到目标端的动态对齐。

---

**完整算法流程**

- 输入处理：
  - 对源句和目标句进行 subword tokenization。
  - 将源端 Token 转换为 Input Embedding。
  - 将目标端右移 Token 转换为 Output Embedding。
  - 分别加入 Positional Encoding。

- Encoder 计算：
  - 对每个 Encoder Layer 重复执行：
    - Multi-Head Self-Attention：
      - 所有源端位置互相关注。
    - Add & LayerNorm：
      - 残差连接后归一化。
    - Position-wise FFN：
      - 对每个位置独立做非线性变换。
    - Add & LayerNorm：
      - 再次残差连接与归一化。
  - 得到 Encoder memory $Z$。

- Decoder 计算：
  - 对每个 Decoder Layer 重复执行：
    - Masked Multi-Head Self-Attention：
      - 目标端位置只能关注自身及历史位置。
    - Add & LayerNorm。
    - Encoder-Decoder Multi-Head Attention：
      - Decoder 表示作为 Query。
      - Encoder memory 作为 Key 和 Value。
    - Add & LayerNorm。
    - Position-wise FFN。
    - Add & LayerNorm。

- 输出预测：
  - Decoder 最后一层输出进入 Linear projection。
  - 使用 Softmax 得到 vocabulary 上的概率分布。
  - 训练时计算交叉熵并结合 label smoothing。
  - 推理时使用 beam search 逐步生成 Token。

---

**Encoder-Decoder Attention 的关键接口**

| Attention 类型 | Query 来源 | Key 来源 | Value 来源 | 作用 |
|---|---|---|---|---|
| Encoder Self-Attention | Encoder 前一层 | Encoder 前一层 | Encoder 前一层 | 源端全局上下文建模 |
| Decoder Masked Self-Attention | Decoder 前一层 | Decoder 前一层 | Decoder 前一层 | 目标端历史建模 |
| Encoder-Decoder Attention | Decoder 前一层 | Encoder 输出 | Encoder 输出 | 源端-目标端对齐 |

- Encoder-Decoder Attention 是连接 Encoder 与 Decoder 的核心桥梁：
  - Decoder 当前生成位置提出 Query。
  - Encoder 所有源端位置提供 Key 和 Value。
  - Attention 权重表示当前目标 Token 对源句不同位置的关注程度。
  - 该机制替代传统 Seq2Seq 中基于 RNN hidden states 的 attention alignment。

---

**没有 Recurrence 的收益**

- 训练并行性显著提升：
  - RNN 必须沿序列方向逐步计算 hidden states。
  - Transformer 可同时处理所有源端位置。
  - Decoder 训练时也可通过 mask 并行计算所有目标位置的损失。

- 长程依赖更容易建模：
  - RNN 中远距离依赖路径长度为 **O(n)**。
  - Transformer 中路径长度为 **O(1)**。
  - 梯度传播路径更短，有利于学习长距离关系。

- 更适合 GPU/TPU：
  - Attention 和 FFN 都可通过矩阵乘法高效实现。
  - 避免循环结构导致的硬件利用率不足。
  - 论文中 base model 训练约 **12 小时**，big model 训练约 **3.5 天**。

---

**没有 Convolution 的收益**

- 不受固定 kernel size 限制：
  - CNN 单层只能看局部窗口。
  - Transformer 单层即可看全局上下文。

- 不需要堆叠大量层扩大感受野：
  - CNN 要覆盖长距离依赖需要更多层或 dilation。
  - Transformer 直接通过 Attention matrix 建立全连接依赖。

- 对语言结构更灵活：
  - 句法依赖经常跨越较远距离。
  - Attention 可动态选择相关 Token，而不是固定窗口聚合。
  - 不同 head 可学习不同语言关系。

---

**图中 Attention 行为的证据**

![](57e3ad00e7c57fe0dc66b468b013f2fcf447ef78a7d2ee01be8b434fe6ef0669.jpg)

- Encoder Self-Attention 能捕捉长距离依赖：
  - 示例中多个 head 关注动词 “making” 与远处短语 “more difficult” 的关系。
  - 说明 Encoder 不需要 RNN 递归也能学习远距离句法和语义连接。

![](1678a839f4e6f07663c6d0c31aa1125d433138f2e6617aae82d039d1529967ea.jpg) *Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.*

- 不同 Attention head 可承担不同任务：
  - 某些 head 可能参与 anaphora resolution。
  - 对 “its” 等代词产生尖锐关注分布。
  - 说明 Multi-Head Self-Attention 不只是信息混合，也能形成可解释的结构化行为。

![](b616696f63c30ad27de5f095b0cac5975742209d1231c70057510010187c1512.jpg)

- 多个 head 显示出与句子结构相关的行为：
  - 不同 head 学到不同模式。
  - 这支持了“通过堆叠 Attention 层替代 recurrence/convolution”的可行性。

---

**在整体模型中的作用**

- Encoder Stack 的作用：
  - 将输入 Token 序列编码为全局上下文化表示。
  - 每个位置都融合源句全局信息。
  - 输出作为 Decoder 可检索的 memory。

- Decoder Stack 的作用：
  - 在目标端历史上下文基础上生成下一个 Token。
  - 通过 Encoder-Decoder Attention 读取源句信息。
  - 保持自回归生成约束。

- 去除 recurrence/convolution 后的整体效果：
  - 模型结构更统一。
  - 编码与解码都由 Attention 和 FFN 构成。
  - 训练效率显著提升。
  - 翻译质量超过当时的 RNN/CNN 模型和部分 ensemble 模型。

---

**关键技术结论**

- Transformer 的 Encoder-Decoder Stack 不是简单替换 RNN，而是重新定义了序列建模路径：
  - **Token 间交互**由 Self-Attention 完成。
  - **非线性变换**由 Position-wise FFN 完成。
  - **序列顺序信息**由 Positional Encoding 注入。
  - **深层稳定训练**由 Residual Connection 和 LayerNorm 保证。
  - **自回归约束**由 Masked Self-Attention 实现。
  - **源端-目标端对齐**由 Encoder-Decoder Attention 实现。

- 该设计的本质优势：
  - 用**全局可并行的动态连接**替代**顺序递归**。
  - 用**内容相关的 Attention 权重**替代**固定卷积窗口**。
  - 用**多头子空间建模**增强表达能力。
  - 用**堆叠层结构**逐步构建高阶语义表示。

### 4. Masked Decoder Self-Attention

**Masked Decoder Self-Attention核心定位**

- **Masked Decoder Self-Attention**是Transformer Decoder中的第一个Attention子层。
- 它的核心目标是保证Decoder满足**auto-regressive property**：
  - 生成第**i**个位置的Token时，只能依赖：
    - 已知的历史输出Token：位置**<i**
    - 当前输入位置在训练时对应的右移Token
  - 不能依赖：
    - 未来Token：位置**>i**
- 该机制通过在**Scaled Dot-Product Attention**的softmax之前加入**causal mask**实现。
- 在论文中，这一设计对应描述为：
  - Decoder self-attention子层会阻止当前位置attend到后续位置。
  - 结合**output embeddings are offset by one position**，确保位置**i**的预测只依赖位置**<i**的已知输出。

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

---

**在Transformer Decoder中的结构位置**

- Transformer Decoder由**N=6**个相同层堆叠组成。
- 每个Decoder layer包含三个主要子层：
  - **Masked Multi-Head Self-Attention**
    - 处理目标端已经生成的Token序列。
    - 使用mask禁止访问未来位置。
  - **Encoder-Decoder Multi-Head Attention**
    - Query来自Decoder上一子层输出。
    - Key和Value来自Encoder输出。
    - 允许Decoder每个位置attend到源序列所有位置。
  - **Position-wise Feed-Forward Network**
    - 对每个位置独立应用相同的两层MLP。
- 每个子层外部都包含：
  - **Residual Connection**
  - **Layer Normalization**
- 子层输出形式为：
  - **LayerNorm(x+Sublayer(x))**

| Decoder子层 | Query来源 | Key来源 | Value来源 | 是否Mask | 作用 |
|---|---|---|---|---|---|
| **Masked Decoder Self-Attention** | Decoder前一层输出 | Decoder前一层输出 | Decoder前一层输出 | 是 | 建模目标端历史依赖 |
| **Encoder-Decoder Attention** | Decoder当前表示 | Encoder输出 | Encoder输出 | 否 | 对齐源序列信息 |
| **Feed-Forward Network** | 当前Decoder表示 | 不适用 | 不适用 | 否 | 非线性特征变换 |

---

**输入输出关系**

- 输入：
  - 训练阶段：
    - Decoder输入是目标序列右移后的Token Embedding。
    - 例如目标序列为：
      - **y₁,y₂,y₃,y₄**
    - Decoder输入为：
      - **,y₁,y₂,y₃**
    - 模型预测目标为：
      - **y₁,y₂,y₃,y₄**
  - 推理阶段：
    - Decoder输入是已经生成的Token前缀。
    - 每一步只追加一个新Token。
- 输入表示：
  - Token经过**Embedding**映射为维度为**d_model=512**的向量。
  - 加上**Positional Encoding**。
  - 得到Decoder底部输入矩阵：
    - **X∈R^{m×d_model}**
    - **m**表示目标序列长度。
- 在Self-Attention中：
  - **Q=XW^Q**
  - **K=XW^K**
  - **V=XW^V**
- 输出：
  - 每个位置得到一个上下文相关表示。
  - 第**i**个位置的输出只融合位置**≤i**的信息。
  - 输出矩阵形状保持为：
    - **R^{m×d_model}**
- 该输出会进入后续的：
  - **Encoder-Decoder Attention**
  - **Feed-Forward Network**
  - 最终通过Linear+Softmax预测下一个Token。

---

**核心算法公式**

- 标准Scaled Dot-Product Attention为：

| 名称 | 公式 |
|---|---|
| **Scaled Dot-Product Attention** | **Attention(Q,K,V)=softmax(QKᵀ/√d_k)V** |

- Masked Decoder Self-Attention在softmax前加入mask矩阵：

| 名称 | 公式 |
|---|---|
| **Masked Attention** | **MaskedAttention(Q,K,V)=softmax((QKᵀ/√d_k)+M)V** |

- 其中：
  - **Q∈R^{m×d_k}**
  - **K∈R^{m×d_k}**
  - **V∈R^{m×d_v}**
  - **QKᵀ∈R^{m×m}**
  - **M∈R^{m×m}**是causal mask矩阵。
- Mask矩阵定义：
  - 当位置**j≤i**时：
    - **M_{i,j}=0**
    - 表示允许第**i**个位置attend到第**j**个位置。
  - 当位置**j>i**时：
    - **M_{i,j}=-∞**
    - 表示禁止第**i**个位置attend到未来位置**j**。
- softmax之后：
  - 被设置为**-∞**的位置概率变为**0**。
  - 未来Token不会贡献任何Value信息。

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

---

**Causal Mask矩阵示例**

- 假设目标序列长度为**5**。
- 行表示当前Query位置**i**。
- 列表示可被attend的Key/Value位置**j**。
- **0**表示允许访问。
- **-∞**表示禁止访问。

| Query位置 | Key 1 | Key 2 | Key 3 | Key 4 | Key 5 |
|---|---:|---:|---:|---:|---:|
| **1** | 0 | -∞ | -∞ | -∞ | -∞ |
| **2** | 0 | 0 | -∞ | -∞ | -∞ |
| **3** | 0 | 0 | 0 | -∞ | -∞ |
| **4** | 0 | 0 | 0 | 0 | -∞ |
| **5** | 0 | 0 | 0 | 0 | 0 |

- 该矩阵是一个**下三角mask**。
- 第**1**个位置只能看到自己。
- 第**3**个位置只能看到位置**1,2,3**。
- 第**5**个位置可以看到全部历史位置**1到5**。
- 如果训练时Decoder输入已经右移，则第**i**个预测实际对应目标Token **y_i**，它的可见输入只包含**y₁,...,y_{i-1}**。

---

**为什么必须Mask未来位置**

- 训练阶段通常采用**teacher forcing**：
  - 整个目标序列一次性输入Decoder。
  - 如果不mask，位置**i**可以直接attend到真实未来Token。
  - 模型会发生**information leakage**。
- 不加mask的后果：
  - 训练损失虚假降低。
  - 模型学到利用未来答案，而不是学习条件生成。
  - 推理阶段没有未来Token可用，训练和推理分布严重不一致。
- 加mask的效果：
  - 保证训练时每个位置只能使用合法上下文。
  - 让并行训练仍然等价于逐步auto-regressive生成。
  - 保持Decoder生成过程的因果性。

---

**算法流程**

- 输入目标序列Token：
  - 原始目标序列：**y₁,y₂,...,y_m**
  - Decoder输入序列：**y₁,...,y_{m-1}**
- 进行Embedding：
  - 查表得到Token Embedding。
  - 乘以**√d_model**。
- 加入Positional Encoding：
  - 为每个位置注入顺序信息。
  - 因为Transformer没有RNN或CNN，必须显式编码位置信息。
- 线性投影得到Q、K、V：
  - **Q=XW^Q**
  - **K=XW^K**
  - **V=XW^V**
- 计算注意力分数：
  - **S=QKᵀ/√d_k**
- 加入causal mask：
  - **S'=S+M**
  - 对未来位置设置为**-∞**
- softmax归一化：
  - **A=softmax(S')**
  - 每一行是当前位置对合法历史位置的attention distribution。
- 加权求和：
  - **O=AV**
- 多头并行：
  - 每个head独立执行上述流程。
  - 将所有head输出拼接。
  - 通过**W^O**投影回**d_model**维。
- 残差与归一化：
  - **LayerNorm(X+Dropout(O))**
- 输出进入下一子层：
  - **Encoder-Decoder Attention**

---

**Multi-Head版本中的Mask使用方式**

- Transformer使用**Multi-Head Attention**：
  - 论文base model中：
    - **h=8**
    - **d_model=512**
    - **d_k=64**
    - **d_v=64**
- 每个head都有独立的线性投影：
  - **W_i^Q**
  - **W_i^K**
  - **W_i^V**
- causal mask通常对所有heads共享：
  - mask形状可广播到：
    - **batch_size×h×m×m**
- 每个head都受到相同的因果约束：
  - 不同head可以关注不同历史位置。
  - 没有任何head可以访问未来Token。
- Multi-Head机制的价值：
  - 一个head可能关注局部短语。
  - 一个head可能关注长距离依赖。
  - 一个head可能关注句法或一致性信息。
  - mask只限制时间方向，不限制head学习不同模式。

| 参数 | Base Transformer设置 | 作用 |
|---|---:|---|
| **N** | 6 | Decoder堆叠层数 |
| **d_model** | 512 | 隐状态与Embedding维度 |
| **h** | 8 | Attention head数量 |
| **d_k** | 64 | 每个head的Key维度 |
| **d_v** | 64 | 每个head的Value维度 |
| **d_ff** | 2048 | Feed-Forward中间层维度 |
| **P_drop** | 0.1 | Dropout比例 |

---

**与Encoder Self-Attention的区别**

| 对比项 | **Encoder Self-Attention** | **Masked Decoder Self-Attention** |
|---|---|---|
| 输入来源 | 源序列Token表示 | 目标序列右移后的Token表示 |
| 是否允许看未来 | 允许 | 不允许 |
| Mask类型 | 通常无causal mask | 使用causal mask |
| Attention范围 | 任意源位置之间双向交互 | 只能访问当前位置及历史位置 |
| 目标 | 建模完整输入序列 | 建模已生成目标前缀 |
| 是否auto-regressive | 否 | 是 |

- Encoder处理的是完整源句。
  - 源句在编码时全部已知。
  - 任意位置可以attend到任意位置。
- Decoder生成目标句。
  - 未来Token在推理时不可得。
  - 必须通过mask保持因果生成约束。

---

**与Encoder-Decoder Attention的区别**

| 对比项 | **Masked Decoder Self-Attention** | **Encoder-Decoder Attention** |
|---|---|---|
| Query | Decoder历史表示 | Decoder当前表示 |
| Key | Decoder历史表示 | Encoder输出 |
| Value | Decoder历史表示 | Encoder输出 |
| 是否mask未来目标Token | 是 | 不适用 |
| 主要作用 | 建模目标端语言模型依赖 | 从源句中检索翻译相关信息 |
| 信息范围 | 已生成目标前缀 | 全部源序列 |

- Masked Decoder Self-Attention回答：
  - 当前目标端已经生成了什么？
  - 当前Token应如何依赖历史目标Token？
- Encoder-Decoder Attention回答：
  - 当前目标位置应该关注源句中的哪些Token？
  - 源端哪些语义对当前生成最相关？

---

**训练阶段的并行性**

- RNN Decoder在训练时通常需要按时间步顺序计算。
- Transformer Decoder虽然是auto-regressive模型，但训练阶段可以并行计算所有位置：
  - 所有目标位置的Q、K、V一次性矩阵化生成。
  - causal mask保证每个位置只使用合法上下文。
- 这带来两个关键优势：
  - 保持**auto-regressive correctness**。
  - 获得**parallel training efficiency**。
- 复杂度层面：
  - Self-Attention每层复杂度为**O(n²·d)**。
  - Sequential Operations为**O(1)**。
  - RNN的Sequential Operations为**O(n)**。

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---:|---:|---:|
| **Self-Attention** | **O(n²·d)** | **O(1)** | **O(1)** |
| **Recurrent** | **O(n·d²)** | **O(n)** | **O(n)** |
| **Convolutional** | **O(k·n·d²)** | **O(1)** | **O(log_k(n))** |

- Mask没有破坏矩阵并行计算。
- Mask只是对attention score矩阵做元素级加法。
- 因此Decoder训练仍可高效利用GPU矩阵乘法。

---

**推理阶段的行为**

- 推理时模型不能一次性知道完整目标序列。
- 生成过程仍然是逐Token进行：
  - 输入
  - 预测**y₁**
  - 输入**y₁**
  - 预测**y₂**
  - 输入**y₁,y₂**
  - 预测**y₃**
- Mask在推理阶段仍然可以使用：
  - 对当前前缀而言，未来位置通常不存在。
  - 如果采用固定长度缓存或批量解码，mask仍用于保证合法访问。
- 实际高效实现中常使用**KV cache**：
  - 历史Key和Value缓存下来。
  - 当前步只计算最新Token的Query。
  - 避免每一步重复计算全部历史表示。
- 论文原始重点在训练并行性，而现代Decoder-only LLM中KV cache成为推理加速关键。

---

**参数设置与实现细节**

- 论文base model中的相关参数：
  - **d_model=512**
  - **h=8**
  - **d_k=d_v=64**
  - **N=6**
  - **P_drop=0.1**
- mask本身通常不是可学习参数：
  - 它是根据序列长度构造的固定结构矩阵。
  - 每个batch根据目标长度生成或复用。
- mask数值实现：
  - 理论上使用**-∞**。
  - 工程中常用极小值：
    - **-1e9**
    - **-1e4**
    - dtype为float16时需避免数值溢出。
- mask加在softmax之前：
  - 正确位置：**attention logits**
  - 不应加在softmax之后。
- Dropout位置：
  - 论文中对每个子层输出使用Dropout。
  - Attention权重上也常见使用Attention Dropout，具体实现框架可能略有差异。
- 维度流转：

| 张量 | 形状示例 | 含义 |
|---|---|---|
| **X** | **batch×m×d_model** | Decoder输入表示 |
| **Q,K,V** | **batch×h×m×d_k** | 多头投影结果 |
| **Scores** | **batch×h×m×m** | Attention logits |
| **Mask** | **1或batch×1×m×m** | causal mask |
| **Attention Weights** | **batch×h×m×m** | softmax后权重 |
| **Head Output** | **batch×h×m×d_v** | 每个head输出 |
| **Output** | **batch×m×d_model** | 拼接并投影后的输出 |

---

**伪代码表达**

- Masked Decoder Self-Attention核心流程：

```text
Input: X ∈ R^{batch×m×d_model}

Q = X W_Q
K = X W_K
V = X W_V

Scores = Q K^T / sqrt(d_k)

Mask[i, j] = 0      if j <= i
Mask[i, j] = -inf   if j > i

Scores = Scores + Mask

A = softmax(Scores)

O = A V

Output = LayerNorm(X + Dropout(O))
```

- Multi-Head版本：

```text
for each head t:
    Q_t = X W_t^Q
    K_t = X W_t^K
    V_t = X W_t^V

    Scores_t = Q_t K_t^T / sqrt(d_k)
    Scores_t = Scores_t + CausalMask
    A_t = softmax(Scores_t)
    Head_t = A_t V_t

O = Concat(Head_1, ..., Head_h) W^O
Output = LayerNorm(X + Dropout(O))
```

---

**在整体模型中的作用**

- **目标端语言建模**
  - 捕获目标语言内部的词序、语法、搭配和长期依赖。
  - 类似一个条件语言模型中的历史上下文建模模块。
- **保证因果生成**
  - 防止Decoder看到未来答案。
  - 让训练目标与推理过程一致。
- **提供并行训练能力**
  - 相比RNN逐步展开，Transformer可一次性处理所有目标位置。
  - causal mask负责维持时序约束。
- **增强长距离依赖建模**
  - 任意历史位置到当前位置路径长度为**O(1)**。
  - 比RNN的**O(n)**路径更利于梯度传播。
- **为Encoder-Decoder Attention准备目标端上下文**
  - Decoder self-attention先形成目标端历史语义状态。
  - 后续Encoder-Decoder Attention再用该状态查询源端信息。
- **配合Positional Encoding引入顺序**
  - Attention本身对顺序不敏感。
  - mask只规定可见范围。
  - Positional Encoding提供位置信息。
  - 二者共同实现有序、因果的序列生成。

---

**关键结论**

- **Masked Decoder Self-Attention**是Transformer Decoder实现auto-regressive生成的核心机制。
- 它通过对attention logits加入**下三角causal mask**，禁止当前位置访问未来Token。
- 它既保持了生成任务所需的因果约束，又保留了Transformer的矩阵并行训练优势。
- 它的输出是目标端历史上下文表示，是后续**Encoder-Decoder Attention**和最终Token预测的基础。
- 没有该mask，Decoder在训练时会发生**future information leakage**，导致训练与推理不一致。

### 5. Position-wise Feed-Forward Network

**Position-wise Feed-Forward Network核心定义**

- **Position-wise Feed-Forward Network**是**Transformer**中每个**Encoder layer**和**Decoder layer**都包含的前馈子层。
- 它的核心特征是：
  - 对序列中每个**position/token**独立处理。
  - 所有位置共享同一组前馈网络参数。
  - 不在不同位置之间直接交换信息。
  - 由**两层线性变换**和中间的**ReLU activation**组成。
- 论文中的公式为：

$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

- 其中：
  - $x$：某个位置上的输入向量。
  - $W_1,b_1$：第一层线性变换的权重和偏置。
  - $W_2,b_2$：第二层线性变换的权重和偏置。
  - $\max(0,\cdot)$：**ReLU**非线性激活函数。

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

---

**在Transformer整体架构中的位置**

- 在**Encoder layer**中，每层包含两个主要子层：
  - **Multi-Head Self-Attention**
  - **Position-wise Feed-Forward Network**
- 在**Decoder layer**中，每层包含三个主要子层：
  - **Masked Multi-Head Self-Attention**
  - **Encoder-Decoder Attention**
  - **Position-wise Feed-Forward Network**
- **Position-wise Feed-Forward Network**位于每层的后半部分：
  - **Attention**负责跨位置的信息交互。
  - **FFN**负责对每个位置的表示进行非线性特征变换。
- 每个子层外部都使用：
  - **Residual Connection**
  - **Layer Normalization**
- 子层输出形式为：

$$
\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))
$$

- 对于**FFN sub-layer**，对应形式为：

$$
\mathrm{LayerNorm}(x+\mathrm{FFN}(x))
$$

---

**输入输出关系**

| 项目 | 说明 |
|---|---|
| 输入 | 序列表示矩阵，形状通常为 $n \times d_{\mathrm{model}}$ |
| 单个位置输入 | $x_i \in \mathbb{R}^{d_{\mathrm{model}}}$ |
| 输出 | 与输入长度相同的序列表示矩阵 |
| 单个位置输出 | $z_i \in \mathbb{R}^{d_{\mathrm{model}}}$ |
| 是否改变序列长度 | 不改变 |
| 是否改变最终表示维度 | 不改变 |
| 是否跨位置交互 | 不直接跨位置交互 |
| 参数是否位置共享 | 是 |

- 输入序列可表示为：

$$
X=(x_1,x_2,\ldots,x_n)
$$

- **FFN**对每个位置分别计算：

$$
z_i=\mathrm{FFN}(x_i)
$$

- 整体输出为：

$$
Z=(z_1,z_2,\ldots,z_n)
$$

- 关键点：
  - 每个$x_i$只依赖自身位置的输入。
  - 不依赖$x_j,j\neq i$。
  - 跨位置依赖已经由前面的**Self-Attention**完成。
  - **FFN**更像是对每个位置的局部特征进行深层映射。

---

**算法流程**

- 对每个位置的输入向量$x$执行第一层线性映射：
  - 将维度从$d_{\mathrm{model}}$扩展到$d_{\mathrm{ff}}$。
  - 计算形式为：

$$
h=xW_1+b_1
$$

- 对隐藏向量$h$施加**ReLU activation**：
  - 负值置零。
  - 正值保留。
  - 引入非线性表达能力。
  - 计算形式为：

$$
a=\max(0,h)
$$

- 对激活后的向量$a$执行第二层线性映射：
  - 将维度从$d_{\mathrm{ff}}$投影回$d_{\mathrm{model}}$。
  - 计算形式为：

$$
y=aW_2+b_2
$$

- 将输出$y$与子层输入$x$做**Residual Connection**：
  - 计算形式为：

$$
x+y
$$

- 对残差结果执行**Layer Normalization**：
  - 稳定训练。
  - 改善梯度传播。
  - 计算形式为：

$$
\mathrm{LayerNorm}(x+y)
$$

---

**参数设置**

| 参数 | 论文Base Model设置 | 含义 |
|---|---:|---|
| $d_{\mathrm{model}}$ | 512 | 输入和输出表示维度 |
| $d_{\mathrm{ff}}$ | 2048 | FFN中间隐藏层维度 |
| 激活函数 | ReLU | 两层线性变换之间的非线性函数 |
| 层数位置 | 每个Encoder/Decoder layer中都有 | 每层独立拥有自己的FFN参数 |
| 是否共享跨层参数 | 否 | 不同层使用不同FFN参数 |
| 是否共享跨位置参数 | 是 | 同一层内所有position使用同一FFN |
| 等价实现 | kernel size为1的两层卷积 | 对每个position独立变换 |

- **Base Transformer**中的典型配置：
  - $d_{\mathrm{model}}=512$
  - $d_{\mathrm{ff}}=2048$
  - 第一层线性变换：$512 \rightarrow 2048$
  - 第二层线性变换：$2048 \rightarrow 512$
- **Big Transformer**中配置增大：
  - $d_{\mathrm{model}}=1024$
  - $d_{\mathrm{ff}}=4096$
  - 第一层线性变换：$1024 \rightarrow 4096$
  - 第二层线性变换：$4096 \rightarrow 1024$

---

**参数矩阵形状**

| 符号 | 形状 | 作用 |
|---|---|---|
| $W_1$ | $d_{\mathrm{model}} \times d_{\mathrm{ff}}$ | 第一层线性投影 |
| $b_1$ | $d_{\mathrm{ff}}$ | 第一层偏置 |
| $W_2$ | $d_{\mathrm{ff}} \times d_{\mathrm{model}}$ | 第二层线性投影 |
| $b_2$ | $d_{\mathrm{model}}$ | 第二层偏置 |

- 在**Base Model**中：
  - $W_1$：$512 \times 2048$
  - $b_1$：$2048$
  - $W_2$：$2048 \times 512$
  - $b_2$：$512$
- 单个FFN子层参数量约为：
  - $512 \times 2048 + 2048 \times 512 + 2048 + 512$
  - 约**2.10M parameters**
- 每个**Encoder layer**和**Decoder layer**都有独立FFN：
  - Base模型共有6层Encoder和6层Decoder。
  - FFN在总参数量中占据重要比例。

---

**为什么称为Position-wise**

- **Position-wise**强调该网络按位置独立应用。
- 对序列矩阵$X \in \mathbb{R}^{n \times d_{\mathrm{model}}}$，FFN等价于：
  - 对每一行向量独立执行同一个MLP。
  - 不进行跨行混合。
- 与**Self-Attention**的差异：
  - **Self-Attention**在不同position之间建立依赖。
  - **FFN**只在每个position内部进行通道维度变换。
- 与**Convolution**的关系：
  - 论文指出它也可以被视为两层**kernel size=1 convolution**。
  - 这种卷积不会看相邻token。
  - 只对每个位置的特征维度做变换。

---

**与Self-Attention的功能分工**

| 模块 | 是否跨位置交互 | 主要作用 |
|---|---|---|
| **Multi-Head Self-Attention** | 是 | 建模不同token之间的依赖关系 |
| **Position-wise FFN** | 否 | 对每个token的表示做非线性特征变换 |
| **Residual Connection** | 否 | 保留原始信息，缓解深层训练问题 |
| **Layer Normalization** | 否 | 稳定训练分布 |

- **Attention**回答的问题：
  - 当前token应该从哪些其他token获取信息？
  - 哪些位置与当前位置相关？
  - 长距离依赖如何建模？
- **FFN**回答的问题：
  - 已聚合的信息如何进一步加工？
  - 每个token表示如何经过非线性变换提升表达能力？
  - 如何在特征维度上进行组合、筛选和重编码？

---

**计算复杂度与并行性**

- **FFN**对每个position独立执行，因此天然支持并行。
- 对长度为$n$的序列，每个位置执行两次矩阵乘法。
- 复杂度近似为：

$$
O(n \cdot d_{\mathrm{model}} \cdot d_{\mathrm{ff}})
$$

- 在Base配置中：
  - 每个position的主要计算量约为：
    - $512 \times 2048$
    - $2048 \times 512$
  - 两次矩阵乘法规模对称。
- 与**RNN**相比：
  - 不需要按时间步顺序递推。
  - 所有position可以同时计算。
- 与**Self-Attention**相比：
  - **Self-Attention**复杂度与$n^2$相关。
  - **FFN**复杂度与$n$线性相关。
  - 当序列很长时，Attention通常成为更明显的瓶颈。

---

**设计动机**

- **增加非线性表达能力**
  - 如果只有Attention中的线性投影和加权求和，模型表达能力受限。
  - **ReLU**使模型能够学习更复杂的特征组合。
- **对每个token进行特征重编码**
  - Attention聚合上下文后，每个position包含来自其他token的信息。
  - FFN进一步对该上下文表示做变换。
- **扩展再压缩的瓶颈结构**
  - $512 \rightarrow 2048 \rightarrow 512$形成“升维-非线性-降维”结构。
  - 中间层较宽，提升特征组合能力。
- **保持序列长度和接口一致**
  - 输入输出都是$d_{\mathrm{model}}$。
  - 便于与Residual Connection相加。
  - 保证Encoder和Decoder堆叠结构稳定。
- **提升硬件效率**
  - 所有位置共享参数。
  - 可通过大规模矩阵乘法高效实现。
  - 适合GPU/TPU并行计算。

---

**实现视角**

- 常见实现中，输入张量形状为：

$$
\mathrm{batch\_size} \times \mathrm{seq\_len} \times d_{\mathrm{model}}
$$

- FFN执行：
  - 第一层Linear：
    - $\mathrm{batch\_size} \times \mathrm{seq\_len} \times 512$
    - 变为$\mathrm{batch\_size} \times \mathrm{seq\_len} \times 2048$
  - ReLU：
    - 形状不变。
  - 第二层Linear：
    - $\mathrm{batch\_size} \times \mathrm{seq\_len} \times 2048$
    - 变回$\mathrm{batch\_size} \times \mathrm{seq\_len} \times 512$
- PyTorch风格伪代码可表达为：

```python
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

- 在完整Transformer层中通常还包括：

```python
x = x + dropout(ffn(x))
x = layer_norm(x)
```

---

**与Residual Connection和LayerNorm的耦合**

- 论文采用的结构是：
  - 子层输出先经过**dropout**。
  - 再与子层输入相加。
  - 最后执行**Layer Normalization**。
- 对FFN子层而言：

$$
\mathrm{Output}=\mathrm{LayerNorm}(x+\mathrm{Dropout}(\mathrm{FFN}(x)))
$$

- 这种结构的意义：
  - **Residual Connection**保留原始表示，避免深层网络退化。
  - **Dropout**抑制过拟合。
  - **LayerNorm**稳定每层输入分布。
  - 保证深层堆叠时梯度更容易传播。

---

**在Encoder中的作用**

- 在**Encoder layer**中，FFN接收**Self-Attention**后的上下文表示。
- 每个token已经通过Self-Attention聚合了全局上下文信息。
- FFN对每个token的上下文表示进行进一步加工：
  - 强化有用特征。
  - 抑制无用特征。
  - 引入非线性组合。
  - 将Attention聚合结果映射到更适合下一层处理的表示空间。
- Encoder中的典型流向：
  - Token Embedding + Positional Encoding
  - Multi-Head Self-Attention
  - Add & Norm
  - Position-wise FFN
  - Add & Norm
  - 输出到下一Encoder layer或Decoder cross-attention

---

**在Decoder中的作用**

- 在**Decoder layer**中，FFN位于两个Attention子层之后。
- FFN处理的是已经融合以下信息的目标端表示：
  - 已生成目标序列的历史信息。
  - 源序列Encoder输出的信息。
- Decoder中的典型流向：
  - Output Embedding + Positional Encoding
  - Masked Multi-Head Self-Attention
  - Add & Norm
  - Encoder-Decoder Attention
  - Add & Norm
  - Position-wise FFN
  - Add & Norm
  - 输出到下一Decoder layer或最终Softmax
- FFN在Decoder中的作用：
  - 对融合后的解码状态进行非线性转换。
  - 为预测下一个token提供更强的局部表达。
  - 保持自回归约束不被破坏，因为它不访问未来position。

---

**为什么FFN不需要跨位置交互**

- 跨位置依赖已经由**Self-Attention**负责。
- FFN的职责不是寻找token之间的关系，而是处理每个token内部的特征。
- 这种模块化分工带来清晰结构：
  - **Attention**负责信息路由。
  - **FFN**负责特征变换。
- 如果FFN也跨位置混合，会增加计算复杂度，并与Attention功能重叠。
- 独立位置处理使FFN保持：
  - 高并行性。
  - 简单实现。
  - 参数共享。
  - 稳定接口。

---

**与传统RNN/Convolution中的前馈变换对比**

| 机制 | 跨位置方式 | 非线性变换方式 | 并行性 |
|---|---|---|---|
| **RNN** | 通过隐藏状态递推 | 每个时间步内部非线性变换 | 受序列顺序限制 |
| **Convolution** | 通过局部kernel窗口 | 卷积后激活 | 可并行 |
| **Transformer FFN** | 不跨位置 | 每个position独立MLP | 高度并行 |
| **Self-Attention** | 全局加权聚合 | 注意力权重加权求和 | 高度并行 |

- **Transformer**将跨位置交互和非线性变换拆分为两个模块：
  - Self-Attention处理全局依赖。
  - FFN处理逐位置非线性映射。
- 这种分离是Transformer高效并行和强表达能力的重要来源。

---

**对模型能力的影响**

- **d_ff越大，模型容量越强**
  - 更宽的隐藏层提供更多特征组合空间。
  - Big模型使用$d_{\mathrm{ff}}=4096$，明显大于Base模型的2048。
- **FFN参数量较大**
  - 在Transformer中，FFN通常贡献大量参数。
  - 增大$d_{\mathrm{ff}}$会显著增加模型容量和计算量。
- **FFN是非线性来源之一**
  - Attention本质上主要是加权聚合。
  - FFN提供逐位置的非线性函数逼近能力。
- **FFN改善逐token表示质量**
  - 对翻译任务而言，每个目标token的预测依赖高质量隐藏表示。
  - FFN使每个位置的表示更适合后续Attention层或输出Softmax。

---

**Base与Big配置对比**

| 配置 | $N$ | $d_{\mathrm{model}}$ | $d_{\mathrm{ff}}$ | Attention heads | Dropout | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
| Transformer Base | 6 | 512 | 2048 | 8 | 0.1 | 约65M |
| Transformer Big | 6 | 1024 | 4096 | 16 | 0.3 | 约213M |

- **Big模型**提升了：
  - $d_{\mathrm{model}}$
  - $d_{\mathrm{ff}}$
  - Attention heads数量
  - 参数规模
- **FFN扩展**是Big模型容量增长的重要因素。
- 论文实验显示：
  - 更大的模型通常带来更好的BLEU。
  - 同时需要更强正则化，例如更高Dropout。

---

**关键技术结论**

- **Position-wise FFN**是Transformer中不可缺少的表示变换模块。
- 它通过**两层Linear + ReLU**实现逐位置非线性映射。
- 它不改变序列长度，也不改变最终表示维度。
- 它不直接建模token间依赖，而是处理每个token已经聚合后的上下文表示。
- 它与**Multi-Head Attention**形成互补：
  - **Attention**负责全局信息交互。
  - **FFN**负责局部特征加工。
- 它的升维-降维结构：
  - 提升模型表达能力。
  - 保持与Residual Connection兼容。
  - 保持Transformer层的输入输出接口统一。
- 它的逐位置独立计算方式：
  - 保证高度并行。
  - 适配GPU矩阵运算。
  - 是Transformer训练效率高的重要组成部分。

### 6. Sinusoidal Positional Encoding

**Sinusoidal Positional Encoding**

- **Sinusoidal Positional Encoding**是 Transformer 为解决**序列顺序建模**问题引入的位置注入机制。
- Transformer 完全移除了 **recurrence** 和 **convolution**：
  - 没有 **RNN** 的时间步递推结构。
  - 没有 **CNN** 的局部窗口与层级感受野。
  - 单纯的 **Self-Attention** 对输入 Token 集合本身是近似“无序”的：如果不注入位置信息，模型无法区分“我喜欢你”和“你喜欢我”这类 Token 相同但顺序不同的序列。
- 因此，论文在 **Encoder** 和 **Decoder** 底部，将 **Positional Encoding** 加到 **Token Embedding** 上，使模型同时获得：
  - **Token 语义信息**。
  - **Token 位置信息**。
  - **相对位置可学习性**。
  - 对更长序列的潜在 **extrapolation** 能力。

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

---

**核心动机**

- Transformer 的 **Self-Attention** 计算形式本身不包含序列方向或顺序：
  - Attention 关注的是 Query 与 Key 的相似度。
  - 如果输入只包含 Token Embedding，那么 Attention 只能知道“是什么词”，不能知道“这个词在第几个位置”。
- **Sinusoidal Positional Encoding**的目标是：
  - 为每个位置生成一个确定性的向量。
  - 向量维度与 **Token Embedding** 一致。
  - 通过加法融合到 Embedding 中。
  - 让后续的 **Multi-Head Attention** 和 **Feed-Forward Network** 能利用顺序信息。
- 论文选择固定的 **sin/cos** 函数，而不是只依赖 learned positional embedding，主要原因是：
  - 不增加额外可学习参数。
  - 位置编码可以计算任意长度位置。
  - 理论上更容易泛化到训练时未见过的更长序列。
  - 任意固定偏移量的相对位置关系可以通过线性变换表示。

---

**数学定义**

- 论文使用如下公式定义位置编码：

| 维度类型 | 公式 |
|---|---|
| 偶数维 | **PE(pos, 2i) = sin(pos / 10000^(2i / d_model))** |
| 奇数维 | **PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))** |

- 参数含义：
  - **pos**：Token 在序列中的绝对位置。
  - **i**：位置编码向量中的维度索引。
  - **d_model**：模型隐藏维度，论文 base model 中为 **512**。
  - **2i**：偶数维度使用 **sin**。
  - **2i+1**：奇数维度使用 **cos**。
  - **10000**：控制不同维度波长跨度的常数。
- 每一个位置都会得到一个长度为 **d_model** 的向量：
  - 输入 Embedding 维度是 **512**。
  - Positional Encoding 维度也是 **512**。
  - 两者可以逐元素相加。

---

**参数规格**

| 参数 | 论文设置 | 作用 |
|---|---:|---|
| **d_model** | **512** | Token Embedding 与 Positional Encoding 的维度 |
| **pos** | 从 0 或 1 开始的位置索引 | 表示 Token 在序列中的绝对位置 |
| **i** | 维度索引 | 控制不同频率的 sin/cos 分量 |
| **base** | **10000** | 控制波长呈几何级数变化 |
| **PE(pos, 2i)** | **sin** | 偶数维位置编码 |
| **PE(pos, 2i+1)** | **cos** | 奇数维位置编码 |

---

**算法流程**

- 输入：
  - 一个 Token 序列，例如长度为 **n**：
    - **x = [x₁, x₂, ..., xₙ]**
  - 每个 Token 经过 Embedding lookup 后得到：
    - **E = [e₁, e₂, ..., eₙ]**
  - 每个 **eᵢ ∈ R^d_model**。
- 生成位置编码：
  - 对序列中每个位置 **pos** 生成一个向量 **PE_pos**。
  - 对每个维度 **j**：
    - 如果 **j** 是偶数，使用 **sin**。
    - 如果 **j** 是奇数，使用 **cos**。
- 融合 Token 信息与位置信息：
  - 对每个位置执行逐元素相加：
    - **z_pos = Embedding(token_pos) + PE_pos**
- 输出：
  - 得到带有位置信息的输入表示：
    - **Z = [z₁, z₂, ..., zₙ]**
  - 该结果送入 **Encoder stack** 或 **Decoder stack** 的第一层。

---

**输入输出关系**

| 阶段 | 输入 | 输出 | 维度 |
|---|---|---|---|
| Tokenization | 原始文本 | Token 序列 | 长度为 **n** |
| Embedding | Token id | Token Embedding | **n × d_model** |
| Positional Encoding | 位置索引 **pos** | 位置向量 **PE_pos** | **n × d_model** |
| Add | Embedding + PE | 带顺序信息的表示 | **n × d_model** |
| Transformer Stack | 带位置的表示 | 上下文表示 | **n × d_model** |

- 对单个位置：
  - 输入：
    - **token_pos**
    - **pos**
  - 输出：
    - **x_pos = Embedding(token_pos) + PE(pos)**
- 对完整序列：
  - 输入矩阵：
    - **Embedding Matrix ∈ R^(n × d_model)**
    - **Positional Encoding Matrix ∈ R^(n × d_model)**
  - 输出矩阵：
    - **X ∈ R^(n × d_model)**

---

**为什么使用 sin 和 cos**

- **sin/cos**函数可以提供连续、平滑、周期性的位置信号。
- 不同维度使用不同频率，使一个位置被编码成多尺度信号：
  - 低维度变化快，捕捉局部位置差异。
  - 高维度变化慢，捕捉长距离位置趋势。
- 论文中波长从 **2π** 到 **10000·2π** 形成几何级数：
  - 让模型同时感知短距离和长距离位置关系。
  - 避免所有维度只表达同一种尺度的位置。
- 使用 **sin/cos 配对**还有一个关键性质：
  - 对任意固定偏移 **k**，**PE(pos+k)** 可以表示为 **PE(pos)** 的线性函数。
  - 这使模型更容易学习相对位置关系。

---

**相对位置可学习性的数学直觉**

- 对于某个频率 **ω**：
  - **sin(pos+k)ω = sin(posω)cos(kω) + cos(posω)sin(kω)**
  - **cos(pos+k)ω = cos(posω)cos(kω) - sin(posω)sin(kω)**
- 含义：
  - 如果模型知道 **sin(posω)** 和 **cos(posω)**，就可以通过线性组合得到 **pos+k** 的位置表示。
  - 这让模型在 **Self-Attention** 中更容易学习“当前位置前后若干距离”的模式。
- 对 Transformer 的意义：
  - Attention 不仅可以关注某个具体位置。
  - 还可以学习“相对偏移”模式，例如：
    - 当前词前一个词。
    - 当前词后两个词。
    - 与当前动词相隔较远的主语。
    - 长距离依赖中的对应成分。

---

**在整体架构中的位置**

- Positional Encoding 位于 Transformer 的最底层：
  - **Encoder**：
    - Source Token Embedding 与 Positional Encoding 相加。
    - 结果进入 Encoder 的第一个 Self-Attention layer。
  - **Decoder**：
    - Target Token Embedding 与 Positional Encoding 相加。
    - 结果进入 Decoder 的 masked Self-Attention layer。
- 论文还在 Embedding 与 Positional Encoding 的和上使用 **dropout**：
  - base model 中 **P_drop = 0.1**。
  - 用于降低过拟合。
- 进入后续层后，位置信息会被：
  - **Multi-Head Attention**利用。
  - **Position-wise Feed-Forward Network**变换。
  - **Residual Connection**保留。
  - **LayerNorm**稳定化。

---

**与 Token Embedding 的融合方式**

- 论文采用**加法融合**，而不是拼接：
  - **Input = Token Embedding + Positional Encoding**
- 选择加法的原因：
  - 保持维度不变，仍为 **d_model**。
  - 不增加后续层参数规模。
  - 让位置与语义在同一表示空间中交互。
  - 与 **Residual Connection** 和 **LayerNorm** 设计兼容。
- 加法融合的影响：
  - 每个 Token 的表示同时包含内容与位置。
  - Attention 计算 Query、Key、Value 时，都会间接包含位置信息。
  - 模型可以通过学习投影矩阵决定使用多少语义信息、多少位置信息。

---

**与 Self-Attention 的关系**

- **Self-Attention**计算：
  - **Attention(Q, K, V) = softmax(QKᵀ / √d_k)V**
- 如果没有 Positional Encoding：
  - Query、Key、Value 只来自 Token Embedding。
  - 相同 Token 在不同位置可能产生相同或高度相似的表示。
  - 模型难以区分排列顺序。
- 加入 Positional Encoding 后：
  - **Q = (Embedding + PE)W_Q**
  - **K = (Embedding + PE)W_K**
  - **V = (Embedding + PE)W_V**
- 结果：
  - Attention score 不仅由语义相似度决定。
  - 也会受到位置关系影响。
  - 不同 Head 可以学习不同类型的位置模式。

---

**与 Learned Positional Embedding 的对比**

| 方案 | 是否可学习 | 参数量 | 长度外推 | 论文结果 |
|---|---|---:|---|---|
| **Sinusoidal Positional Encoding** | 否 | 无额外位置参数 | 较强 | 论文默认选择 |
| **Learned Positional Embedding** | 是 | 需要位置表参数 | 受训练最大长度限制 | 与 sinusoidal 结果几乎一致 |

- 论文在 Table 3 row (E) 中比较了两种方式：
  - 将 sinusoidal positional encoding 替换为 learned positional embedding。
  - 结果几乎相同。
- 论文最终选择 sinusoidal 的原因：
  - 性能不差。
  - 更简单。
  - 无需学习位置表。
  - 可能更好地推广到训练长度之外的序列。

---

**在 Encoder 中的作用**

- Encoder 的输入是完整 source sequence。
- Positional Encoding 让 Encoder Self-Attention 能区分：
  - Token 的绝对位置。
  - Token 之间的顺序。
  - 远距离依赖结构。
- 对机器翻译尤其重要：
  - 主语、谓语、宾语顺序影响句义。
  - 修饰语位置影响语法关系。
  - 长距离依赖需要位置感知。
- Encoder 中每个位置都可以 attend 到所有其他位置：
  - Positional Encoding 提供“这些位置之间相隔多远”的基础信号。
  - 不同 Head 可以学习句法、语义或局部邻接模式。

---

**在 Decoder 中的作用**

- Decoder 使用 masked Self-Attention。
- Positional Encoding 在 Decoder 中帮助模型知道：
  - 当前生成到第几个位置。
  - 已生成 Token 的顺序。
  - 输出序列中的局部和长程结构。
- 与 mask 配合：
  - mask 防止当前位置看到未来 Token。
  - Positional Encoding 告诉模型历史 Token 的相对顺序。
- 对自回归生成的意义：
  - 预测第 **i** 个 Token 时，只能依赖位置小于 **i** 的输出。
  - 但模型仍需要知道这些历史 Token 分别处于什么位置。
  - Positional Encoding 提供这种顺序信号。

---

**实现细节伪代码**

- 典型实现流程：

```python
import math
import torch

def sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
```

- 关键实现点：
  - **max_len**：预先支持的最大序列长度。
  - **d_model**：必须与 Token Embedding 维度一致。
  - **0::2**：偶数维写入 **sin**。
  - **1::2**：奇数维写入 **cos**。
  - **div_term**：对应公式中的 **1 / 10000^(2i/d_model)**。
- 使用时：
  - 从位置编码矩阵中截取当前序列长度部分。
  - 与 Token Embedding 相加。
  - 送入 dropout。
  - 进入 Transformer stack。

---

**数值形态示例**

- 假设：
  - **d_model = 512**
  - 序列长度 **n = 10**
- 则：
  - Token Embedding shape：
    - **10 × 512**
  - Positional Encoding shape：
    - **10 × 512**
  - 相加后输入 Transformer 的 shape：
    - **10 × 512**
- 对 batch 输入：
  - 假设 batch size 为 **B**
  - Token Embedding shape：
    - **B × n × 512**
  - Positional Encoding shape：
    - **1 × n × 512** 或 **n × 512**
  - 广播相加后：
    - **B × n × 512**

---

**为什么不使用单一标量位置**

- 不能直接给每个 Token 加一个位置标量，原因包括：
  - 标量表达能力弱。
  - 不同尺度的距离关系难以表示。
  - 大位置数值可能造成训练不稳定。
  - Attention 投影矩阵难以从单一标量中提取丰富结构。
- Sinusoidal 向量的优势：
  - 每个位置由多个频率共同编码。
  - 可表达局部与全局位置信息。
  - 与神经网络线性层兼容。
  - 便于模型学习相对偏移。

---

**与模型超参数的关系**

| 组件 | 相关性 |
|---|---|
| **d_model** | Positional Encoding 维度必须等于 **d_model** |
| **sequence length** | 每个位置生成一个 PE 向量 |
| **Multi-Head Attention** | Q/K/V 投影会接收融合了 PE 的表示 |
| **dropout** | 论文对 Embedding + PE 的结果使用 dropout |
| **Encoder/Decoder depth** | PE 只在底部加入，后续通过层间传播 |
| **Vocabulary size** | 与 PE 无直接关系 |

---

**对训练与泛化的影响**

- 对训练：
  - 提供位置归纳偏置。
  - 帮助模型更快学习序列结构。
  - 减少模型仅靠数据自行推断顺序的负担。
- 对泛化：
  - 固定公式可以为任意位置计算编码。
  - 不受 learned position table 的最大索引限制。
  - 理论上可以处理比训练阶段更长的序列。
- 局限：
  - 虽然可以计算更长位置的编码，但不保证模型一定能可靠处理更长上下文。
  - 模型其他部分仍可能受训练分布、Attention 复杂度和长度泛化能力限制。

---

**关键结论**

- **Sinusoidal Positional Encoding**是 Transformer 在无 recurrence、无 convolution 条件下获得序列顺序信息的核心机制。
- 它通过固定的 **sin/cos 多频率函数**为每个位置生成长度为 **d_model** 的向量。
- 它与 **Token Embedding**直接相加，作为 **Encoder** 和 **Decoder** 的底层输入。
- 它使 **Self-Attention** 能够感知 Token 的绝对位置与相对位置关系。
- 与 **Learned Positional Embedding**相比，它性能相近，但更具长度外推潜力。
- 在原始 Transformer 中，它是支撑“完全基于 Attention 的序列建模”成立的关键设计之一。


---

## 4. 实验方法与实验结果

**实验设置**

- **任务设置**
  - 论文主要在三个任务上验证 **Transformer**：
    - **WMT 2014 English-to-German Translation**
      - 训练集约 **4.5M sentence pairs**
      - 使用 **Byte-Pair Encoding, BPE**
      - 源端与目标端共享词表，约 **37K tokens**
      - 测试集为 **newstest2014**
      - 开发集消融实验使用 **newstest2013**
    - **WMT 2014 English-to-French Translation**
      - 训练集约 **36M sentence pairs**
      - 使用 **WordPiece vocabulary**
      - 词表大小约 **32K tokens**
      - 测试集为 **newstest2014**
    - **English Constituency Parsing**
      - 使用 **Penn Treebank WSJ**
      - WSJ-only 设置约 **40K training sentences**
      - Semi-supervised 设置额外使用约 **17M sentences**
      - 测试集为 **WSJ Section 23**
      - 开发集为 **WSJ Section 22**

- **模型架构设置**
  - 基础架构为标准 **Encoder-Decoder Transformer**。
  - Encoder 与 Decoder 均堆叠 **N=6 layers**。
  - 每个 Encoder layer 包含：
    - **Multi-Head Self-Attention**
    - **Position-wise Feed-Forward Network**
    - **Residual Connection**
    - **Layer Normalization**
  - 每个 Decoder layer 包含：
    - **Masked Multi-Head Self-Attention**
    - **Encoder-Decoder Attention**
    - **Position-wise Feed-Forward Network**
    - **Residual Connection**
    - **Layer Normalization**
  - Decoder 使用 **masking** 防止当前位置访问未来 token，保证 **auto-regressive generation**。

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

- **Attention 机制设置**
  - 核心计算为 **Scaled Dot-Product Attention**：
    - 使用 **Q, K, V** 矩阵表示 queries、keys、values。
    - 使用缩放因子 **1/sqrt(d_k)** 缓解大维度 dot product 导致 softmax 梯度过小的问题。
  - 公式为：
    - **Attention(Q,K,V)=softmax(QKᵀ/sqrt(d_k))V**
  - **Multi-Head Attention** 默认设置：
    - **h=8 heads**
    - **d_model=512**
    - **d_k=64**
    - **d_v=64**
    - 每个 head 在不同表示子空间中并行建模依赖关系。

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

- **Base model 与 Big model 参数**
  - **Base Transformer**
    - **N=6**
    - **d_model=512**
    - **d_ff=2048**
    - **h=8**
    - **d_k=d_v=64**
    - 参数量约 **65M**
  - **Transformer Big**
    - **N=6**
    - **d_model=1024**
    - **d_ff=4096**
    - **h=16**
    - 参数量约 **213M**
    - 使用更高 dropout，English-German 中为 **P_drop=0.3**

| 模型 | N | d_model | d_ff | heads | d_k | d_v | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Transformer Base** | 6 | 512 | 2048 | 8 | 64 | 64 | 65M |
| **Transformer Big** | 6 | 1024 | 4096 | 16 | 64左右 | 64左右 | 213M |

- **Positional Encoding 设置**
  - 由于 Transformer 不使用 **RNN** 或 **CNN**，模型本身没有序列顺序归纳偏置。
  - 论文采用 **sinusoidal positional encoding**：
    - 偶数维使用 **sin**
    - 奇数维使用 **cos**
    - 不同维度对应不同频率
  - 设计动机：
    - 使模型更容易学习 **relative position**
    - 理论上可泛化到比训练时更长的序列
  - 论文也测试了 **learned positional embeddings**，性能与 sinusoidal 几乎相同。

- **训练硬件与批处理**
  - 使用单机 **8 NVIDIA P100 GPUs**。
  - 每个 batch 约包含：
    - **25K source tokens**
    - **25K target tokens**
  - 句子按近似长度分组，降低 padding 浪费。
  - **Base model**
    - 每步约 **0.4s**
    - 训练 **100K steps**
    - 总训练时间约 **12 hours**
  - **Big model**
    - 每步约 **1.0s**
    - 训练 **300K steps**
    - 总训练时间约 **3.5 days**

| 模型 | GPUs | Step Time | Train Steps | 总训练时间 |
|---|---:|---:|---:|---:|
| **Transformer Base** | 8 P100 | 0.4s | 100K | 约12小时 |
| **Transformer Big** | 8 P100 | 1.0s | 300K | 约3.5天 |

- **优化器与学习率**
  - 使用 **Adam optimizer**：
    - **β₁=0.9**
    - **β₂=0.98**
    - **ε=1e-9**
  - 使用特殊 learning rate schedule：
    - 前 **4000 warmup steps** 线性升高
    - 之后按 **inverse square root decay** 衰减
  - 该设计对训练稳定性非常关键，尤其适合深层 attention 模型。

- **正则化策略**
  - 使用三类正则化：
    - **Residual Dropout**
      - 对每个 sub-layer 输出施加 dropout
      - 再加入 residual connection 并做 LayerNorm
    - **Embedding + Positional Encoding Dropout**
      - 对 token embedding 与 positional encoding 的和进行 dropout
    - **Label Smoothing**
      - 使用 **ε_ls=0.1**
      - 牺牲部分 perplexity
      - 提升 accuracy 与 **BLEU**
  - Base model 默认：
    - **P_drop=0.1**
    - **ε_ls=0.1**

- **解码与评估设置**
  - 使用 **beam search**
    - beam size 为 **4**
    - length penalty **α=0.6**
  - 最大生成长度设为：
    - **input length + 50**
  - 对 checkpoint 做平均：
    - Base model 平均最后 **5 checkpoints**
    - Big model 平均最后 **20 checkpoints**
  - 翻译任务评价指标：
    - **BLEU**
  - Parsing 任务评价指标：
    - **WSJ Section 23 F1**

---

**主实验结果：Machine Translation**

- **English-to-German**
  - **Transformer Big** 达到 **28.4 BLEU**
  - 相比此前最强 ensemble 结果提升超过 **2 BLEU**
  - **Transformer Base** 达到 **27.3 BLEU**
  - Base model 已超过所有此前公开模型与 ensemble
  - 训练成本仅为许多竞争模型的一小部分

- **English-to-French**
  - **Transformer Big** 达到 **41.8 BLEU**
  - 创下当时 single-model state-of-the-art
  - 训练成本显著低于此前最佳模型
  - 论文正文中有一处描述为 **41.0 BLEU**，表格给出 **41.8 BLEU**；通常以表格结果为准。

| Model | EN-DE BLEU | EN-FR BLEU | EN-DE Training Cost | EN-FR Training Cost |
|---|---:|---:|---:|---:|
| **ByteNet** | 23.75 | - | - | - |
| **GNMT + RL** | 24.6 | 39.92 | 2.3e19 | 1.4e20 |
| **ConvS2S** | 25.16 | 40.46 | 9.6e18 | 1.5e20 |
| **MoE** | 26.03 | 40.56 | 2.0e19 | 1.2e20 |
| **GNMT + RL Ensemble** | 26.30 | 41.16 | 1.8e20 | 1.1e21 |
| **ConvS2S Ensemble** | 26.36 | 41.29 | 7.7e19 | 1.2e21 |
| **Transformer Base** | **27.3** | 38.1 | **3.3e18** | - |
| **Transformer Big** | **28.4** | **41.8** | 2.3e19 | - |

- **结果解读**
  - **质量优势明显**
    - Transformer Big 在 EN-DE 上超过此前最佳 ensemble 超过 **2 BLEU**。
    - Transformer Base 已经超过 RNN/CNN-based ensemble。
  - **训练成本优势突出**
    - Base model 训练成本 **3.3e18 FLOPs**，显著低于 ConvS2S、GNMT、MoE。
    - Big model 的 EN-DE 训练成本约 **2.3e19 FLOPs**，与 GNMT 单模型相近，但 BLEU 明显更高。
  - **并行化收益显著**
    - Transformer 摆脱 recurrent dependency。
    - Encoder 端 self-attention 可对所有 token 并行计算。
    - 相比 RNN 的 **O(n)** sequential operations，Self-Attention 为 **O(1)**。
  - **性能与成本的组合优势**
    - Transformer 不只是提高 BLEU，也大幅缩短训练时间。
    - 这说明其优势来自架构归纳偏置与硬件友好矩阵计算的结合。

---

**复杂度与架构效率分析**

- **Self-Attention 的优势**
  - 每层复杂度为 **O(n²·d)**。
  - 最小 sequential operations 为 **O(1)**。
  - 任意两个位置之间最大路径长度为 **O(1)**。
  - 对机器翻译常见句长而言，通常 **n < d**，因此 self-attention 计算成本可低于 recurrent layer。

- **Recurrent layer 的限制**
  - 每层复杂度为 **O(n·d²)**。
  - sequential operations 为 **O(n)**。
  - 最大路径长度为 **O(n)**。
  - 长距离依赖需要跨越多个时间步，梯度传播路径长。

- **Convolutional layer 的折中**
  - 每层复杂度为 **O(k·n·d²)**。
  - sequential operations 为 **O(1)**。
  - 最大路径长度为 **O(log_k n)**，假设使用 dilated convolution。
  - 需要堆叠多层才能覆盖长距离依赖。

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---:|---:|---:|
| **Self-Attention** | **O(n²·d)** | **O(1)** | **O(1)** |
| **Recurrent** | O(n·d²) | O(n) | O(n) |
| **Convolutional** | O(k·n·d²) | O(1) | O(log_k n) |
| **Restricted Self-Attention** | O(r·n·d) | O(1) | O(n/r) |

- **关键结论**
  - **Self-Attention** 的核心优势不是单纯低复杂度，而是：
    - 更短的 dependency path
    - 更高的并行度
    - 更适合 GPU/TPU 上的大矩阵乘法
  - 其主要短板是：
    - 对长序列存在 **O(n²)** attention matrix 成本
    - 论文也指出可通过 **restricted self-attention** 处理超长序列

---

**消融实验设置**

- 消融实验在 **WMT 2014 English-to-German development set newstest2013** 上进行。
- 使用 Base model 作为参照。
- 解码使用 beam search。
- 不使用 checkpoint averaging，以便更直接观察模型结构变化的影响。
- 主要考察因素：
  - **attention heads 数量**
  - **d_k 与 d_v**
  - **模型宽度与深度**
  - **dropout**
  - **label smoothing**
  - **positional encoding 类型**

---

**消融实验：Attention Heads 数量**

| 设置 | heads | d_k | d_v | PPL | BLEU |
|---|---:|---:|---:|---:|---:|
| **Base** | 8 | 64 | 64 | 4.92 | **25.8** |
| Single-head | 1 | 512 | 512 | 5.29 | 24.9 |
| Fewer heads | 4 | 128 | 128 | 5.00 | 25.5 |
| More heads | 16 | 32 | 32 | 4.91 | **25.8** |
| Too many heads | 32 | 16 | 16 | 5.01 | 25.4 |

- **现象**
  - **1 head** 明显变差，BLEU 从 **25.8** 降到 **24.9**。
  - **4 heads** 接近 base，但仍略低。
  - **16 heads** 与 base 持平。
  - **32 heads** 反而下降。
- **解释**
  - Multi-head 的价值在于让模型在多个子空间中同时建模不同依赖。
  - 单头 attention 容易被 weighted average 限制，无法同时捕获多类关系。
  - 过多 heads 会导致每个 head 的维度过小，例如 **d_k=16**，单个 head 表达能力不足。
- **结论**
  - **Multi-Head Attention 是核心组件**。
  - heads 数量存在最优区间，并非越多越好。
  - 原论文 base 中的 **8 heads** 是性能与计算的稳健折中。

---

**消融实验：Key/Value 维度**

| 设置 | d_k | d_v | PPL | BLEU | Params |
|---|---:|---:|---:|---:|---:|
| **Base** | 64 | 64 | 4.92 | **25.8** | 65M |
| Smaller key/value | 16 | 16 | 5.16 | 25.1 | 58M |
| Medium key/value | 32 | 32 | 5.01 | 25.4 | 60M |

- **现象**
  - 减小 **d_k** 与 **d_v** 会导致 BLEU 下降。
  - **d_k=d_v=16** 时，BLEU 降至 **25.1**。
  - **d_k=d_v=32** 时，BLEU 为 **25.4**，仍低于 base。
- **解释**
  - **d_k** 影响 query-key compatibility 的判别能力。
  - key 维度过小会限制 attention score 的表达能力。
  - value 维度过小会限制被聚合信息的容量。
- **结论**
  - Attention 不只是“选择位置”，也依赖足够维度表达复杂匹配关系。
  - 论文据此推测，更复杂的 compatibility function 可能进一步提升效果。

---

**消融实验：模型规模**

| 设置 | N | d_model | d_ff | PPL | BLEU | Params |
|---|---:|---:|---:|---:|---:|---:|
| Smaller depth/width | 2 | 512 | 2048 | 6.11 | 23.7 | 36M |
| Smaller d_model | 6 | 256 | 1024 | 5.19 | 25.3 | 50M |
| Larger d_model | 6 | 1024 | 4096 | 4.88 | 25.5 | 80M |
| Smaller d_ff | 6 | 512 | 1024 | 5.75 | 24.5 | 28M |
| Larger d_ff | 6 | 512 | 4096 | 4.66 | **26.0** | 168M |

- **现象**
  - 减少层数到 **N=2**，BLEU 大幅下降至 **23.7**。
  - 减小 **d_model** 到 256，BLEU 降至 **25.3**。
  - 增大 **d_ff** 到 4096，BLEU 提升到 **26.0**。
  - 更大模型通常带来更低 PPL 与更高 BLEU。
- **解释**
  - Transformer 的表示能力高度依赖：
    - self-attention 层堆叠深度
    - token representation 宽度
    - FFN 中间层容量
  - **Position-wise FFN** 不只是辅助模块，而是提供重要的非线性变换与通道混合能力。
- **结论**
  - **模型容量是性能提升的重要因素**。
  - 增大 **d_ff** 对 BLEU 有明显正面影响。
  - 但更大参数量也需要更强正则化，否则容易过拟合。

---

**消融实验：Dropout 与 Label Smoothing**

| 设置 | P_drop | ε_ls | PPL | BLEU |
|---|---:|---:|---:|---:|
| **Base** | 0.1 | 0.1 | 4.92 | **25.8** |
| No Dropout | 0.0 | 0.1 | 4.75 | 25.5 |
| Higher Dropout | 0.2 | 0.1 | 5.77 | 24.6 |
| No Label Smoothing | 0.1 | 0.0 | 4.95 | 25.3 |

- **Dropout 现象**
  - 不使用 dropout 时，PPL 反而更低，但 BLEU 下降。
  - dropout 过高到 **0.2** 时，PPL 与 BLEU 都明显变差。
- **Dropout 解释**
  - 无 dropout 可能导致模型对开发集 token likelihood 拟合更强，但泛化到 BLEU 不佳。
  - dropout 对避免 overfitting 有帮助，但过强会损害建模能力。
- **Label Smoothing 现象**
  - 去掉 label smoothing 后，BLEU 从 **25.8** 降至 **25.3**。
- **Label Smoothing 解释**
  - Label smoothing 会让模型不要过度自信。
  - 虽然可能损害 perplexity，但改善生成质量与 BLEU。
- **结论**
  - **正则化对 Transformer 至关重要**。
  - **PPL 与 BLEU 不完全一致**。
  - **Label Smoothing 是提升翻译质量的重要训练技巧**。

---

**消融实验：Positional Encoding**

| 设置 | Positional Encoding | PPL | BLEU |
|---|---|---:|---:|
| **Base** | Sinusoidal | 4.92 | **25.8** |
| Learned Position Embedding | Learned | 4.67 | 25.7 |

- **现象**
  - 使用 **learned positional embedding** 与 **sinusoidal positional encoding** 性能几乎相同。
  - BLEU 仅从 **25.8** 变为 **25.7**。
- **解释**
  - 对 WMT 翻译任务而言，两种位置编码都能有效注入顺序信息。
  - sinusoidal 的优势不主要体现在当前长度分布内，而在可能具备更好的长度外推能力。
- **结论**
  - Transformer 必须引入位置信息。
  - 但具体使用 fixed sinusoidal 还是 learned embedding，对该实验影响不大。

---

**Parsing 实验结果**

- **实验目标**
  - 验证 Transformer 是否只适用于机器翻译，还是能够迁移到其他 sequence transduction 任务。
  - English Constituency Parsing 具有额外挑战：
    - 输出序列通常长于输入序列
    - 输出受强结构约束
    - 小数据场景下 RNN seq2seq 表现通常不强

- **模型设置**
  - 使用 **4-layer Transformer**
  - **d_model=1024**
  - WSJ-only 设置词表为 **16K tokens**
  - Semi-supervised 设置词表为 **32K tokens**
  - 使用 beam size **21**
  - length penalty **α=0.3**
  - 最大输出长度设为 **input length + 300**

| Parser | Training | WSJ 23 F1 |
|---|---|---:|
| Vinyals & Kaiser et al. | WSJ only, discriminative | 88.3 |
| Petrov et al. | WSJ only, discriminative | 90.4 |
| Zhu et al. | WSJ only, discriminative | 90.4 |
| Dyer et al. | WSJ only, discriminative | **91.7** |
| **Transformer 4 layers** | WSJ only, discriminative | **91.3** |
| Zhu et al. | Semi-supervised | 91.3 |
| Huang & Harper | Semi-supervised | 91.3 |
| McClosky et al. | Semi-supervised | 92.1 |
| **Transformer 4 layers** | Semi-supervised | **92.1** |
| Luong et al. | Semi-supervised multi-task | 92.7 |
| Dyer et al. | Generative | 93.0 |

- **结果解读**
  - WSJ-only 中，Transformer 达到 **91.3 F1**。
  - 该结果超过 Berkeley Parser 等传统强基线。
  - Semi-supervised 中，Transformer 达到 **92.1 F1**。
  - 虽未超过最强 specialized parser，但在缺少大量 task-specific tuning 的情况下表现很强。
- **结论**
  - Transformer 不只是 translation-specific 架构。
  - Self-attention 对结构化输出任务同样有效。
  - 该实验为后续将 Transformer 扩展到更多 NLP 任务提供了证据。

---

**Attention 可解释性观察**

- 论文附录给出 attention visualization，用于说明不同 heads 学到了不同语言结构。
- 示例包括：
  - 长距离依赖建模
  - 指代消解
  - 句法结构相关模式

![](57e3ad00e7c57fe0dc66b468b013f2fcf447ef78a7d2ee01be8b434fe6ef0669.jpg)

- **长距离依赖**
  - 某些 head 能从 **making** 关注到远处与其构成短语关系的词。
  - 说明 self-attention 可以直接连接远距离 token。
  - 这与 RNN 需要跨多个时间步传播形成鲜明对比。

![](1678a839f4e6f07663c6d0c31aa1125d433138f2e6617aae82d039d1529967ea.jpg) *Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.*

- **指代消解**
  - 某些 head 对 **its** 等代词产生非常尖锐的 attention 分布。
  - 表明不同 attention head 可能自发学习到 anaphora resolution 相关功能。

![](b616696f63c30ad27de5f095b0cac5975742209d1231c70057510010187c1512.jpg)

- **句法结构**
  - 部分 head 呈现类似句法依赖的关注模式。
  - 这说明 Multi-Head Attention 不只是平均聚合上下文，而是具备结构分工能力。

---

**实验结论与关键判断**

- **Transformer 的主要实验证据**
  - 在 **WMT 2014 EN-DE** 上达到 **28.4 BLEU**，显著超过此前 state-of-the-art。
  - 在 **WMT 2014 EN-FR** 上达到 **41.8 BLEU**，刷新 single-model 表现。
  - 在训练成本上明显优于 RNN/CNN-based 竞争模型。
  - 在 parsing 上也取得强结果，验证跨任务泛化能力。

- **消融实验支撑的核心结论**
  - **Multi-Head Attention 必不可少**
    - 单头 attention 明显退化。
    - 过多 heads 也会因 head dimension 过小而下降。
  - **Attention 维度不能过小**
    - 降低 **d_k/d_v** 会损害匹配与信息聚合能力。
  - **模型容量直接影响性能**
    - 增大 **d_model**、**d_ff** 通常改善结果。
    - **d_ff** 对性能提升尤其明显。
  - **正则化非常关键**
    - Dropout 与 Label Smoothing 对 BLEU 有显著影响。
    - PPL 更低不一定代表 BLEU 更高。
  - **位置编码形式不是主要瓶颈**
    - learned 与 sinusoidal 表现接近。
    - sinusoidal 的潜在优势在长度外推。

- **整体评价**
  - 实验设计覆盖了：
    - 主任务性能对比
    - 训练效率对比
    - 架构复杂度理论分析
    - 关键模块消融
    - 跨任务泛化验证
    - attention 可解释性观察
  - 证据链较完整，清晰证明：
    - **完全基于 Attention 的架构可以替代 RNN/CNN**
    - **Self-Attention 在质量、并行性、长距离依赖建模上同时具备优势**
    - **Transformer 的成功来自架构设计、训练策略和正则化共同作用**

---

