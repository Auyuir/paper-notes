# Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works. 图表详解

### Figure 1: The Transformer - model architecture.

![f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg)

- **图像总体含义**
  - 该图展示了论文 **“Attention Is All You Need”** 中提出的 **Transformer model architecture**。
  - 整体结构是典型的 **Encoder-Decoder architecture**：
    - 左侧是 **Encoder stack**，负责将输入序列编码为上下文表示。
    - 右侧是 **Decoder stack**，负责基于已生成的目标序列和 Encoder 输出逐步生成目标 token。
  - 图中核心思想是：模型完全依赖 **Attention mechanisms**，尤其是 **Multi-Head Attention**，不使用 **RNN** 或 **CNN**。

- **整体结构概览**

| 区域 | 组件 | 作用 |
|---|---|---|
| 左侧 | **Encoder** | 对输入序列进行表示学习 |
| 右侧 | **Decoder** | 根据 Encoder 输出和历史目标 token 生成输出 |
| 底部 | **Input / Output Embedding** | 将 token 转换为向量表示 |
| 底部两侧 | **Positional Encoding** | 注入位置信息 |
| 中部 | **N× stacked layers** | Encoder 和 Decoder 都由多层重复模块堆叠而成 |
| 顶部 | **Linear + Softmax** | 将 Decoder 输出映射为词表概率分布 |

- **左侧 Encoder 分析**
  - Encoder 输入流程为：
    - **Inputs**
    - **Input Embedding**
    - **Positional Encoding**
    - **N× Encoder layers**
    - 输出给 Decoder 的 **encoder representations**
  - 图中左侧的 **N×** 表示 Encoder block 会重复堆叠多次。
  - 在论文默认配置中：
    - **N = 6**
    - **d_model = 512**
  - 每个 Encoder layer 包含两个核心子层：

| Encoder 子层 | 图中名称 | 功能 |
|---|---|---|
| 第 1 子层 | **Multi-Head Attention** | 对输入序列内部所有位置进行 self-attention |
| 第 2 子层 | **Feed Forward** | 对每个位置独立进行非线性变换 |
| 残差与归一化 | **Add & Norm** | 使用 residual connection 和 LayerNorm 稳定训练 |

- **Encoder 中 Multi-Head Attention 的作用**
  - 这里的 **Multi-Head Attention** 是 **self-attention**。
  - 对 Encoder 来说：
    - **Query = Key = Value = Encoder 当前层输入**
  - 每个输入 token 都可以直接关注输入序列中的所有其他 token。
  - 这使得模型能够以 **O(1) maximum path length** 建模长距离依赖。
  - 相比 RNN：
    - RNN 需要沿时间步逐步传播信息。
    - Transformer 中任意两个 token 可通过一次 self-attention 直接交互。

- **Encoder 中 Add & Norm 的含义**
  - 图中每个子层上方都有 **Add & Norm**。
  - 它表示：
    - **Add**：残差连接，即将子层输入直接加到子层输出上。
    - **Norm**：进行 **Layer Normalization**。
  - 对应公式为：
    - **LayerNorm(x + Sublayer(x))**
  - 作用：
    - 缓解深层网络训练困难。
    - 保留原始信息。
    - 提高梯度传播稳定性。

- **Encoder 中 Feed Forward 的作用**
  - **Feed Forward** 是位置前馈网络，即 **Position-wise Feed-Forward Network**。
  - 它对每个位置独立应用相同的两层全连接网络。
  - 论文中公式为：
    - **FFN(x) = max(0, xW₁ + b₁)W₂ + b₂**
  - 默认维度：
    - 输入 / 输出维度：**d_model = 512**
    - 中间层维度：**d_ff = 2048**
  - 它补充了 attention 的表示能力，引入非线性变换。

- **右侧 Decoder 分析**
  - Decoder 输入流程为：
    - **Outputs shifted right**
    - **Output Embedding**
    - **Positional Encoding**
    - **N× Decoder layers**
    - **Linear**
    - **Softmax**
    - **Output Probabilities**
  - Decoder 也是由 **N× stacked layers** 构成。
  - 论文默认配置中：
    - **N = 6**
  - 每个 Decoder layer 包含三个核心子层：

| Decoder 子层 | 图中名称 | 功能 |
|---|---|---|
| 第 1 子层 | **Masked Multi-Head Attention** | 对已生成目标 token 做 masked self-attention |
| 第 2 子层 | **Multi-Head Attention** | 对 Encoder 输出做 encoder-decoder attention |
| 第 3 子层 | **Feed Forward** | 对每个位置进行非线性变换 |
| 残差与归一化 | **Add & Norm** | 每个子层后都有 residual connection 和 LayerNorm |

- **Masked Multi-Head Attention 分析**
  - Decoder 底部的第一个 attention 模块是 **Masked Multi-Head Attention**。
  - 它是 Decoder 内部的 **self-attention**。
  - 与 Encoder self-attention 不同，Decoder 需要保持 **auto-regressive property**。
  - 因此，当前位置只能关注：
    - 当前 token
    - 之前已经生成的 token
  - 不能关注未来 token。
  - 图中使用 **Masked** 表明：
    - 在 softmax 前将非法的未来位置设置为 **−∞**。
    - 这样 softmax 后未来位置权重为 0。
  - 这保证了训练和推理时生成过程一致。

- **Outputs shifted right 的含义**
  - 图底部右侧写有 **Outputs (shifted right)**。
  - 表示 Decoder 的输入是目标序列右移一位后的结果。
  - 例如训练时目标序列为：
    - y₁, y₂, y₃, ...
  - Decoder 输入为：
    - <BOS>, y₁, y₂, ...
  - 这样模型在预测 yᵢ 时只能看到 y₁ 到 yᵢ₋₁。
  - 它与 **Masked Multi-Head Attention** 共同保证自回归生成。

- **Encoder-Decoder Multi-Head Attention 分析**
  - Decoder 中间的 **Multi-Head Attention** 是 **encoder-decoder attention**。
  - 它连接左侧 Encoder 输出和右侧 Decoder。
  - 图中可以看到从 Encoder stack 输出有一条箭头连接到 Decoder 中间 attention 层。
  - 在该层中：
    - **Query** 来自 Decoder 前一子层。
    - **Key** 和 **Value** 来自 Encoder 输出。
  - 作用是让 Decoder 在生成每个目标 token 时关注源语言输入序列的相关位置。
  - 这类似传统 seq2seq 模型中的 attention，但 Transformer 使用的是 **Multi-Head Attention**。

- **Decoder 中 Feed Forward 的作用**
  - Decoder 顶部的 **Feed Forward** 与 Encoder 中的 Feed Forward 结构相同。
  - 它对每个位置独立应用两层全连接网络。
  - 作用：
    - 增强每个 token 表示的非线性建模能力。
    - 与 attention 层互补。
    - attention 负责跨位置交互，FFN 负责逐位置特征变换。

- **Embedding 与 Positional Encoding**
  - 图底部显示输入和输出都先经过 Embedding：
    - 左侧：**Input Embedding**
    - 右侧：**Output Embedding**
  - 由于 Transformer 没有 recurrence 和 convolution，本身无法天然感知 token 顺序。
  - 因此图中在 Embedding 后加入 **Positional Encoding**。
  - 图中圆圈加号表示：
    - **Embedding vector + Positional Encoding vector**
  - 论文中使用 sinusoidal positional encoding：
    - 偶数维使用 **sin**
    - 奇数维使用 **cos**
  - 作用：
    - 注入绝对位置信息。
    - 帮助模型学习相对位置关系。
    - 支持比训练长度更长的序列外推。

- **顶部 Linear 与 Softmax**
  - Decoder stack 输出后进入：
    - **Linear**
    - **Softmax**
  - **Linear**：
    - 将 Decoder hidden states 映射到词表大小的 logits。
  - **Softmax**：
    - 将 logits 转换为每个词的概率。
  - 最终输出为：
    - **Output Probabilities**
  - 该概率分布用于预测下一个 token。

- **图中箭头含义**
  - 竖直箭头：
    - 表示数据流从下往上逐层传播。
  - 弯曲箭头：
    - 表示残差连接，即绕过子层后再与子层输出相加。
  - Encoder 到 Decoder 的横向箭头：
    - 表示 Encoder 输出被送入 Decoder 的 encoder-decoder attention。
  - Decoder 内部向上的箭头：
    - 表示目标序列逐层处理，最后生成输出概率。

- **核心模块关系表**

| 模块 | 所在位置 | 输入来源 | 主要作用 |
|---|---|---|---|
| **Input Embedding** | Encoder 底部 | Source tokens | token 向量化 |
| **Output Embedding** | Decoder 底部 | Shifted target tokens | 目标 token 向量化 |
| **Positional Encoding** | Encoder / Decoder 底部 | 位置信息 | 注入顺序信息 |
| **Multi-Head Self-Attention** | Encoder | Encoder 内部表示 | 建模源序列内部依赖 |
| **Masked Multi-Head Self-Attention** | Decoder | Decoder 历史输出 | 保证自回归生成 |
| **Encoder-Decoder Attention** | Decoder 中部 | Decoder query + Encoder key/value | 对源序列进行对齐和检索 |
| **Feed Forward** | Encoder / Decoder | 每个位置表示 | 非线性特征变换 |
| **Add & Norm** | 每个子层后 | 子层输入与输出 | 残差连接与归一化 |
| **Linear + Softmax** | Decoder 顶部 | Decoder final states | 输出词表概率 |

- **Multi-Head Attention 在图中的地位**
  - 图中最核心的模块是 **Multi-Head Attention**。
  - 它出现了三种形式：
    - Encoder self-attention
    - Decoder masked self-attention
    - Encoder-decoder attention
  - 多头机制的优势：
    - 不同 head 可以关注不同位置。
    - 不同 head 可以学习不同语义或句法关系。
    - 避免单一 attention head 的信息平均问题。
  - 论文默认设置：
    - **h = 8**
    - **d_k = 64**
    - **d_v = 64**
    - **d_model = 512**

- **Encoder 与 Decoder 的关键区别**

| 对比项 | Encoder | Decoder |
|---|---|---|
| 输入 | Source sequence | Shifted target sequence |
| 是否 masked | 否 | 是，第一层 self-attention masked |
| attention 类型 | Self-attention | Masked self-attention + encoder-decoder attention |
| 是否访问 Encoder 输出 | 不需要 | 需要 |
| 主要功能 | 理解输入 | 生成输出 |
| 信息流 | 双向可见整个输入序列 | 单向，只能看历史目标 token |

- **图像体现的 Transformer 设计理念**
  - **完全去除 recurrence**
    - 不再逐时间步处理序列。
    - 训练时可对序列位置并行计算。
  - **完全去除 convolution**
    - 不依赖局部卷积窗口传播信息。
    - 任意位置可通过 attention 直接交互。
  - **使用 self-attention 建模全局依赖**
    - 每个 token 都可以关注所有 token。
  - **使用 positional encoding 弥补顺序信息缺失**
    - 让模型区分不同位置。
  - **使用 residual connection + LayerNorm 保证深层稳定训练**
    - 支持多层堆叠。
  - **使用 encoder-decoder attention 完成源目标对齐**
    - 适用于 machine translation 等 sequence transduction task。

- **该架构相对 RNN / CNN 的优势**
  
| 维度 | Transformer | RNN | CNN |
|---|---|---|---|
| 并行性 | **高**，序列位置可并行 | 低，必须按时间步递推 | 较高 |
| 长距离依赖 | **直接建模** | 路径长，容易梯度衰减 | 依赖多层堆叠 |
| 最大路径长度 | **O(1)** | O(n) | O(logₖ n) 或 O(n/k) |
| 主要计算 | Attention matrix multiplication | Hidden state recurrence | Convolution |
| 顺序建模方式 | Positional Encoding | 时间递推天然包含顺序 | 卷积局部结构 |

- **图中 N× 的意义**
  - 图中 Encoder 和 Decoder 旁边都标有 **N×**。
  - 表示该模块不是单层，而是重复堆叠 N 次。
  - 在原论文 base model 中：
    - Encoder 有 **6 layers**
    - Decoder 有 **6 layers**
  - 堆叠层数越多，模型表达能力越强，但计算成本和参数量也增加。

- **从输入到输出的完整数据路径**
  - 输入侧：
    - **Inputs → Input Embedding → + Positional Encoding → Encoder stack**
  - 编码侧：
    - **Self-Attention → Add & Norm → Feed Forward → Add & Norm**
  - 解码侧：
    - **Outputs shifted right → Output Embedding → + Positional Encoding**
    - **Masked Self-Attention → Add & Norm**
    - **Encoder-Decoder Attention → Add & Norm**
    - **Feed Forward → Add & Norm**
  - 输出侧：
    - **Linear → Softmax → Output Probabilities**

- **图像的论文意义**
  - 该图是 Transformer 的核心架构图。
  - 它清晰表达了论文的主要贡献：
    - **用 attention 替代 recurrence 和 convolution**
    - **通过 multi-head self-attention 实现高效全局依赖建模**
    - **通过 stacked encoder-decoder 结构完成 sequence transduction**
  - 后续几乎所有现代大语言模型，如 **BERT**, **GPT**, **T5**, **ViT**，都可视为该架构思想的延伸或变体。

### Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.

![da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg)

- **图像对象**：该图是论文《Attention Is All You Need》中的 **Figure 2**，用于说明 Transformer 的两个核心模块：
  - 左侧：**Scaled Dot-Product Attention**
  - 右侧：**Multi-Head Attention**
  - 图像重点展示了从 **Q / K / V** 到注意力输出的计算流程，以及 Multi-Head 如何并行执行多个 attention head。

- **整体结构概览**

| 区域 | 模块 | 核心作用 |
|---|---|---|
| 左侧 | **Scaled Dot-Product Attention** | 计算单个 attention head 的输出 |
| 右侧 | **Multi-Head Attention** | 并行执行多个 Scaled Dot-Product Attention，并融合结果 |
| 输入符号 | **Q, K, V** | 分别表示 Query、Key、Value |
| 输出路径 | Attention 输出 | 经过 Linear 投影后进入后续网络层 |

- **左侧：Scaled Dot-Product Attention 流程分析**

| 步骤 | 图中模块 | 数学含义 | 作用 |
|---|---|---|---|
| 1 | **Q, K 输入 MatMul** | $QK^T$ | 计算 Query 与 Key 的相似度 |
| 2 | **Scale** | $\frac{QK^T}{\sqrt{d_k}}$ | 缩放点积，防止 softmax 梯度过小 |
| 3 | **Mask opt.** | mask illegal positions | 在 decoder 中屏蔽未来 token |
| 4 | **SoftMax** | $\text{softmax}(\cdot)$ | 将相似度转为 attention weights |
| 5 | **与 V 做 MatMul** | $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ | 对 Value 加权求和得到输出 |

- **Scaled Dot-Product Attention 的核心公式**

| 公式 | 含义 |
|---|---|
| **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V** | Transformer 中基础 attention 计算单元 |
| **QKᵀ** | 计算每个 Query 对所有 Key 的匹配程度 |
| **√dₖ** | 缩放因子，稳定训练 |
| **softmax** | 生成归一化注意力分布 |
| **V** | 被注意力权重加权汇聚的信息 |

- **图中 Q / K / V 的语义解释**

| 符号 | 英文 | 作用 | 直观理解 |
|---|---|---|---|
| **Q** | Query | 发起查询 | 当前 token 想找什么信息 |
| **K** | Key | 提供匹配索引 | 每个 token 可以被匹配的特征 |
| **V** | Value | 提供内容信息 | 真正被聚合的信息内容 |

- **Scale 模块的重要性**
  - **Scale** 是该图中区别普通 dot-product attention 的关键。
  - 当 $d_k$ 较大时，$QK^T$ 的数值可能变大。
  - 大数值输入 softmax 后容易导致分布过于尖锐。
  - 分布过尖会使梯度变小，训练不稳定。
  - 因此使用 **$\frac{1}{\sqrt{d_k}}$** 进行缩放。
  - 论文中默认：
    - **h = 8**
    - **d_model = 512**
    - **d_k = d_v = 64**
    - 因此每个 head 的缩放因子为 **√64 = 8**。

- **Mask opt. 模块分析**
  - **Mask opt.** 表示 optional mask。
  - 它主要用于 **decoder self-attention**。
  - 作用是防止当前位置看到未来位置的信息。
  - 在自回归生成中，第 $i$ 个位置只能依赖小于等于 $i$ 的位置。
  - 实现方式通常是：
    - 对非法位置加上 **−∞**
    - 再经过 **SoftMax**
    - 非法位置的 attention weight 变为 **0**

- **左侧流程的直观解释**
  - 每个位置先用 **Q** 去和所有位置的 **K** 做匹配。
  - 得到“我应该关注谁”的分数。
  - 分数经过 **Scale** 和 **SoftMax** 后变成权重。
  - 权重再作用到 **V** 上。
  - 最终得到当前位置融合全局上下文后的表示。

- **右侧：Multi-Head Attention 流程分析**

| 步骤 | 图中模块 | 操作 | 作用 |
|---|---|---|---|
| 1 | 输入 **V, K, Q** | 原始输入进入多个 Linear 层 | 生成不同子空间的 Q/K/V |
| 2 | 多组 **Linear** | $QW_i^Q, KW_i^K, VW_i^V$ | 为每个 head 学习独立投影 |
| 3 | 多个并行 attention | Scaled Dot-Product Attention | 每个 head 独立关注不同关系 |
| 4 | **Concat** | 拼接所有 head 输出 | 汇总多头信息 |
| 5 | 顶部 **Linear** | $W^O$ 投影 | 融合多头结果，恢复到 d_model 维度 |

- **Multi-Head Attention 公式对应图像**

| 图中部分 | 公式对应 |
|---|---|
| 多个 Linear | $QW_i^Q, KW_i^K, VW_i^V$ |
| 紫色 attention 模块 | $\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ |
| 多个并行 head | $\text{head}_1, ..., \text{head}_h$ |
| Concat | $\text{Concat}(\text{head}_1, ..., \text{head}_h)$ |
| 顶部 Linear | $\text{Concat}(\cdot)W^O$ |

- **Multi-Head Attention 的完整表达**

| 公式 | 说明 |
|---|---|
| **MultiHead(Q,K,V)=Concat(head₁,...,headₕ)Wᴼ** | 多头输出拼接后再线性变换 |
| **headᵢ=Attention(QWᵢQ, KWᵢK, VWᵢV)** | 每个 head 都是一次独立的 Scaled Dot-Product Attention |

- **图中并行 head 的视觉表达**
  - 右侧紫色模块后方有多个重叠的阴影框。
  - 这些重叠框表示多个 **Scaled Dot-Product Attention** 并行运行。
  - 右侧标注 **h**，表示 attention head 的数量。
  - 在论文基础模型中：
    - **h = 8**
    - 即 8 个 attention head 并行工作。

- **为什么需要 Multi-Head Attention**
  - 单个 attention head 容易把信息压缩成单一加权平均。
  - 多个 head 可以在不同表示子空间中学习不同关系。
  - 不同 head 可以关注：
    - **语法依赖**
    - **长距离依赖**
    - **指代关系**
    - **局部短语结构**
    - **源语言与目标语言对齐关系**
  - 论文附录中的可视化表明，不同 attention head 确实会学习不同功能。

- **Multi-Head 与单头 Attention 的对比**

| 对比项 | Single-Head Attention | Multi-Head Attention |
|---|---|---|
| 表示空间 | 单一空间 | 多个子空间 |
| 关注模式 | 单一注意力分布 | 多种注意力分布 |
| 表达能力 | 较弱 | 更强 |
| 信息融合 | 一次加权平均 | 多头结果拼接后融合 |
| 论文实验表现 | BLEU 较低 | Base 设置效果最佳 |

- **图像中的输入顺序细节**
  - 左侧显示输入为 **Q, K, V**。
  - 右侧底部从左到右显示为 **V, K, Q**。
  - 这只是图形布局选择，不影响数学定义。
  - 实际 Multi-Head Attention 中仍然同时使用 **Q, K, V** 三类输入。

- **该图与 Transformer 架构的关系**

| Transformer 位置 | 使用方式 | Q / K / V 来源 |
|---|---|---|
| **Encoder self-attention** | 编码器内部建模输入序列关系 | Q、K、V 都来自 encoder 上一层 |
| **Decoder masked self-attention** | 解码器内部建模已生成序列 | Q、K、V 都来自 decoder 上一层，并使用 mask |
| **Encoder-decoder attention** | 解码器关注输入序列 | Q 来自 decoder，K/V 来自 encoder 输出 |

- **左侧与右侧的层级关系**
  - **Scaled Dot-Product Attention** 是基础计算单元。
  - **Multi-Head Attention** 是由多个 Scaled Dot-Product Attention 并行组成的复合模块。
  - 因此右侧可以理解为：
    - 先复制多组 Q/K/V 投影；
    - 每组执行左侧流程；
    - 再拼接和线性映射。

- **计算复杂度分析**

| 模块 | 主要计算 | 复杂度特征 |
|---|---|---|
| Scaled Dot-Product Attention | $QK^T$ 和 attention weights 乘 V | 与序列长度平方相关 |
| Multi-Head Attention | h 个 head 并行计算 | 总计算量与单头 full dimension 接近 |
| Linear projections | Q/K/V 投影和输出投影 | 与 $d_{model}^2$ 相关 |

- **为什么多头总成本没有显著增加**
  - 虽然有多个 head，但每个 head 的维度被降低。
  - 论文中：
    - **d_model = 512**
    - **h = 8**
    - **d_k = d_v = 64**
  - 每个 head 只处理 64 维。
  - 8 个 head 拼接后回到 512 维。
  - 因此计算量大致接近一个 512 维单头 attention。

- **图像中的关键设计思想**
  - **并行化**：所有位置之间的 attention 可通过矩阵乘法并行计算。
  - **全局依赖**：任意两个 token 之间路径长度为 O(1)。
  - **子空间分解**：Multi-Head 让模型在不同语义空间中学习关系。
  - **训练稳定性**：Scale 避免 softmax 饱和。
  - **自回归约束**：Mask 保证 decoder 不泄露未来信息。

- **与 RNN / CNN 的差异**

| 模型结构 | 长距离依赖路径 | 并行能力 | 图中机制优势 |
|---|---|---|---|
| **RNN** | O(n) | 弱 | Attention 可直接连接任意位置 |
| **CNN** | O(log n) 或 O(n/k) | 强 | Attention 不依赖卷积堆叠扩大感受野 |
| **Self-Attention** | O(1) | 强 | 图中 QKᵀ 一步建立全局关系 |

- **图像传达的核心结论**
  - Transformer 的核心不是 recurrence，也不是 convolution，而是 **attention-based representation learning**。
  - 左侧说明了单个 attention 的数学计算。
  - 右侧说明了 Transformer 如何通过 **Multi-Head Attention** 增强表达能力。
  - 该模块是 Transformer 能够高效建模长距离依赖、实现高度并行训练的关键。

### 57e3ad00e7c57fe0dc66b468b013f2fcf447ef78a7d2ee01be8b434fe6ef0669.jpg

![57e3ad00e7c57fe0dc66b468b013f2fcf447ef78a7d2ee01be8b434fe6ef0669.jpg](57e3ad00e7c57fe0dc66b468b013f2fcf447ef78a7d2ee01be8b434fe6ef0669.jpg)

- **图像类型与来源**
  - 该图是 Transformer 论文《Attention Is All You Need》附录中的 **Figure 3**。
  - 展示的是 **encoder self-attention** 中某一层的注意力可视化。
  - 具体说明为：**第 5 层 / 共 6 层 encoder self-attention** 中，模型如何通过 attention 捕捉长距离依赖。
  - 图中重点分析的词是 **“making”**，展示多个 attention heads 从该词出发关注到句子中其他位置的情况。

- **图像核心信息概览**

| 项目 | 内容 |
|---|---|
| 模型 | **Transformer** |
| 模块 | **Encoder self-attention** |
| 层数 | **Layer 5 of 6** |
| 关注词 | **making** |
| 可视化对象 | 多个 **attention heads** 的注意力分布 |
| 主要现象 | “making” 关注到远距离词 **more**、**difficult** |
| 语言结构 | 捕捉短语 **“making ... more difficult”** |
| 论文意图 | 证明 self-attention 能建模 **long-distance dependencies** |

- **图像布局分析**
  - 图像上下各有一行相同或近似相同的英文 token 序列。
  - 上方 token 序列表示被 attention 指向的位置，即 **keys / values** 所在位置。
  - 下方 token 序列表示当前 query 所在位置，即从某个词发出 attention。
  - 中间的彩色线条表示 attention 权重连接。
  - 图中灰色竖向高亮区域标出下方与上方的 **“making”** 位置。
  - 多条彩色线从 **“making”** 出发，指向右侧远处的 **“more”**、**“difficult”** 等词。

- **句子内容分析**
  - 图中句子大致为：
    - **“It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult.”**
  - 该句包含一个典型的长距离依赖结构：
    - **making ... more difficult**
  - “making” 与 “more difficult” 之间隔着多个词：
    - **the registration or voting process**
  - 因此，这不是局部邻近词关系，而是典型的 **long-distance dependency**。

- **关键 token 及其作用**

| Token | 语法 / 语义作用 | 在图中的意义 |
|---|---|---|
| **making** | 现在分词，构成结果性表达 | attention 的出发点 |
| **the registration or voting process** | 被影响的对象 | 位于 making 与 difficult 之间 |
| **more** | 程度副词 | 与 difficult 组成比较级结构 |
| **difficult** | 形容词，结果状态 | making 的语义补足 |
| **.** | 句末标点 | 表示句子结束 |
| **\<EOS\>** | End of Sequence | 序列终止符 |
| **\<pad\>** | Padding token | 用于 batch 对齐，无实际语义 |

- **attention 连线分析**
  - 图中多条彩色线从 **“making”** 指向右侧远距离位置。
  - 最明显的目标包括：
    - **more**
    - **difficult**
    - 句末附近的符号或 padding 区域
  - 不同颜色表示不同的 **attention heads**。
  - 多个 head 同时关注同一远距离结构，说明模型在该层已经形成了较稳定的句法 / 语义关联。

- **多头注意力 Multi-Head Attention 的体现**

| 可视化元素 | 含义 |
|---|---|
| 不同颜色的线 | 不同 **attention heads** |
| 线条透明度 / 深浅 | attention 权重强弱 |
| 多条线从 making 发出 | 多个 head 同时以 making 为 query |
| 指向不同 token | 不同 head 学到不同依赖模式 |
| 指向 more / difficult | 捕捉语义补足关系 |

- **为什么该图重要**
  - 该图直观展示了 Transformer 的一个核心优势：
    - **self-attention 可以直接连接任意两个位置**
  - 对于 RNN 来说，“making” 到 “difficult” 的信息传递需要经过多个时间步。
  - 对于 CNN 来说，如果卷积核较小，也需要多层堆叠才能覆盖两者。
  - 对于 Transformer 的 self-attention：
    - **任意两个 token 之间的路径长度为 O(1)**。
  - 因此模型更容易捕捉远距离依赖。

- **与论文 Table 1 的对应关系**

| Layer Type | 最大路径长度 | 对该图的解释 |
|---|---:|---|
| **Self-Attention** | **O(1)** | making 可直接 attend 到 difficult |
| **Recurrent** | **O(n)** | 需沿序列逐步传递信息 |
| **Convolutional** | **O(logₖ n)** 或更长 | 需多层卷积扩大感受野 |
| **Restricted Self-Attention** | **O(n/r)** | 若限制窗口，可能无法直接看到 difficult |

- **语法关系分析**
  - “making” 是一个非谓语动词，引出结果：
    - **making the registration or voting process more difficult**
  - 其中：
    - 宾语是 **the registration or voting process**
    - 宾语补足语是 **more difficult**
  - 因此，“making” 与 “difficult” 存在强语法关系。
  - 图中 attention 正是捕捉到了这种非相邻但强相关的结构。

- **语义关系分析**
  - “making” 表示造成某种状态变化。
  - “more difficult” 表示变化后的状态。
  - 二者之间形成因果 / 结果语义链：
    - **laws → making → process → more difficult**
  - attention head 指向 “more difficult”，说明模型不仅关注邻近词，还能关注完成句义所需的远程补足成分。

- **视觉细节说明**
  - **灰色竖条**：
    - 突出显示当前分析的 token **making**。
  - **彩色半透明线**：
    - 表示 attention 从 “making” 指向其他 token。
  - **右下方彩色块**：
    - 表示不同 attention heads 在目标 token 附近的权重集中。
  - **线条集中指向 more / difficult**：
    - 表明这些词获得了较高 attention 权重。
  - **\<pad\> 区域仍出现弱连接**：
    - 可能是可视化或 padding mask 附近的残留显示，但主要语义关注点仍是有效 token。

- **该图体现的 Transformer 能力**
  - **长距离依赖建模**
    - 从 “making” 直接关注远处的 “more difficult”。
  - **句法结构感知**
    - attention head 能捕捉动词与补足语之间的关系。
  - **语义组合能力**
    - 将 “making” 与结果状态 “difficult” 联系起来。
  - **多头分工**
    - 不同 heads 学到不同关注模式。
  - **可解释性**
    - attention 可视化让模型内部行为具有一定可观察性。

- **与论文主张的关系**
  - 论文提出 Transformer 完全放弃 recurrence 和 convolution，仅依赖 attention。
  - 该图为这一主张提供了直观证据：
    - 即使没有 RNN 的顺序状态传递，模型仍能捕捉长距离语言结构。
  - 图中 “making → more difficult” 的连接说明：
    - **self-attention 不需要逐词传播即可建立远距离语义联系**。

- **从机器翻译角度理解**
  - 在机器翻译中，长距离依赖非常关键。
  - 如果模型误解 “making ... more difficult”，可能会翻译成局部片段，导致语义不完整。
  - 该 attention pattern 有助于模型正确理解：
    - 谁造成变化；
    - 什么对象被影响；
    - 变化结果是什么。
  - 因此这种机制直接支撑了 Transformer 在 WMT 翻译任务上的高 BLEU 表现。

- **与 Multi-Head Attention 设计的关系**
  - 单个 attention head 往往会把信息加权平均，可能难以同时关注多种关系。
  - Multi-Head Attention 允许模型并行学习不同关系：
    - 某些 head 关注局部短语；
    - 某些 head 关注动词-宾语；
    - 某些 head 关注长距离补足语；
    - 某些 head 关注句法边界或标点。
  - 图中多个颜色的线说明：
    - **不同 head 对 “making” 的解释并不完全相同**。
  - 这正是论文中强调的：
    - **“Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.”**

- **图像的技术含义**
  - 对于某个 query token “making”，attention 计算如下：
    - query 来自 “making”；
    - keys 来自同一句中所有 token；
    - values 也来自同一句中所有 token。
  - 模型通过 scaled dot-product attention 计算：
    - **Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V**
  - 图中的线条可以理解为 softmax 后的权重可视化。
  - 线越明显，说明该 token 对 “making” 的表示更新贡献越大。

- **注意力可解释性的边界**
  - 该图说明 attention 与语言结构存在明显相关性。
  - 但不能简单等同于：
    - attention 权重 = 完整因果解释。
  - 更准确地说：
    - 图像展示了模型在构造 “making” 表示时重点聚合了哪些 token 信息。
  - 因此它是模型行为的有用证据，但不是严格的因果证明。

- **综合评价**
  - 该图是 Transformer 论文中非常关键的可解释性示例。
  - 它展示了 encoder self-attention 在高层中学到的长距离依赖。
  - “making” 对 “more difficult” 的关注说明模型能够跨越多个中间 token 捕捉完整语义结构。
  - 图像有力支持了论文的核心观点：
    - **Attention alone is sufficient to model complex sequence dependencies.**
  - 该可视化也解释了为什么 Transformer 在机器翻译中优于传统 RNN / CNN 架构：
    - **路径更短、并行性更好、远距离依赖更易学习、注意力行为更可解释。**

### Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.

![1678a839f4e6f07663c6d0c31aa1125d433138f2e6617aae82d039d1529967ea.jpg](1678a839f4e6f07663c6d0c31aa1125d433138f2e6617aae82d039d1529967ea.jpg)

- **图像对象与上下文**
  - 该图来自论文 **Attention Is All You Need** 的附录可视化部分，对应 **Figure 4**。
  - 图像展示的是 **Transformer encoder self-attention** 中第 **5 层 / 共 6 层** 的注意力模式。
  - 论文说明指出：这两个 attention heads 似乎参与了 **anaphora resolution**，即**指代消解**。
  - 图中句子为：
    - **“The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion . <EOS> <pad>”**
  - 重点分析对象是代词 **“its”**，模型需要判断 **its** 指向什么先行词。

- **图像结构概览**

| 区域 | 内容 | 含义 |
|---|---|---|
| **上半部分** | Head 5 的完整 self-attention 可视化 | 展示句子中所有 token 之间的注意力连接 |
| **下半部分** | 仅隔离显示从 **“its”** 出发的注意力 | 对比 Head 5 与 Head 6 在代词 **its** 上的注意力行为 |
| **紫色线条** | attention 权重连接 | 颜色越深、线越明显，表示权重越高 |
| **灰色竖条** | 当前被关注或被隔离分析的 token 区域 | 用于突出 **its** 或相关目标词 |
| **上下两排 token** | query 与 key/value 的位置对应 | 表示某个位置对其他位置的关注分布 |

- **上半部分：Head 5 的完整注意力模式**
  - 上半部分展示了 **Head 5** 在整句话上的完整 attention 分布。
  - 大量浅紫色连线表示许多 token 之间存在弱注意力关系。
  - 少数深紫色连线表示 **Head 5** 对某些 token 对建立了强关联。
  - 图中可以观察到：
    - **“Law”** 与句子前部多个词之间存在较强连接。
    - **“its”** 附近存在较明显的注意力集中。
    - 后半句如 **“my opinion”**、**“missing”** 等也有局部强连接。
  - 这说明该 attention head 并非简单地关注相邻词，而是在句子内部建立较长距离依赖。

- **下半部分：从 “its” 出发的注意力**
  - 下半部分只保留了从 **“its”** 这个 token 出发的注意力连接。
  - 该部分是整张图的核心，因为 **“its”** 是一个指代性代词。
  - 图中显示：
    - **“its”** 对 **“Law”** 有一条非常强的紫色连接。
    - **“its”** 对 **“application”** 也存在明显连接，但颜色较偏棕/淡，权重相对较弱或属于另一个 head。
  - 这说明模型在处理 **“its application”** 时，可能学到了：
    - **its → Law**
    - 即 **“its application” = “the application of the Law”**

- **关键注意力关系分析**

| Query token | 被关注 token | 可能语义关系 | 说明 |
|---|---|---|---|
| **its** | **Law** | **指代消解 / anaphora resolution** | **its** 很可能指代前文的 **The Law** |
| **its** | **application** | 所属短语内部关系 | **its application** 构成名词短语，application 是被修饰中心词 |
| **Law** | **its / application** | 语义绑定 | 模型将法律与其应用建立关联 |
| **application** | **its** | 局部短语依赖 | 表明模型理解 possessive construction |

- **为什么该图体现 anaphora resolution**
  - **Anaphora resolution** 指模型识别代词、指示词等表达所对应的先行词。
  - 在该句中：
    - **“The Law”** 是先行词。
    - **“its”** 是代词。
    - **“application”** 是被拥有或被限定的对象。
  - 人类理解为：
    - “法律永远不会是完美的，但**它的应用**应该是公正的。”
  - 因此 **its** 的语义来源应是 **The Law**。
  - 图中 **its → Law** 的强连接说明 Transformer 的某些 attention heads 能够自动学习这种跨距离语义依赖。

- **Head 5 与 Head 6 的差异**
  - 图注指出下半部分展示的是 **attention heads 5 and 6** 中从 **“its”** 出发的注意力。
  - 从可视化上看：
    - 一个 head 更强地关注 **Law**。
    - 另一个 head 更明显地关注 **application**。
  - 这体现了 **Multi-Head Attention** 的核心机制：
    - 不同 heads 可以学习不同类型的关系。
    - 一个 head 负责捕捉 **指代关系**。
    - 另一个 head 可能负责捕捉 **短语结构关系** 或 **局部句法关系**。

| Attention Head | 主要关注点 | 可能功能 |
|---|---|---|
| **Head 5** | **its → Law** | 捕捉长距离指代关系 |
| **Head 6** | **its → application** 或短语局部结构 | 捕捉 possessive phrase 内部结构 |
| 二者合并 | Law / its / application 三者绑定 | 形成较完整的语义结构表示 |

- **“very sharp attention” 的含义**
  - 图注中特别强调：**“the attentions are very sharp for this word.”**
  - 这里的 **sharp** 指注意力分布非常集中。
  - 换言之，**its** 并没有平均关注很多词，而是将主要权重集中到少数关键 token 上。
  - 这与一般模糊注意力不同，说明模型在该位置做出了明确选择。
  - 对指代消解任务而言，这非常重要：
    - 如果注意力分散，说明模型不确定先行词。
    - 如果注意力尖锐集中到 **Law**，说明模型很可能识别出了正确指代对象。

- **从模型机制角度解释**
  - 在 Transformer encoder self-attention 中，每个 token 都会生成：
    - **Query**
    - **Key**
    - **Value**
  - 对于 token **its**：
    - 它的 Query 会与句中所有 token 的 Key 做相似度计算。
    - 如果 **Law** 的 Key 与 **its** 的 Query 高度匹配，则 attention 权重变大。
  - 因此图中的强连线可以理解为：
    - 模型认为 **Law** 对理解 **its** 的表示非常重要。
  - 最终 **its** 的表示会融合来自 **Law** 的 Value 信息，使其上下文表示更符合句意。

- **该图支持的论文观点**
  - 图像直接支持论文中关于 **Multi-Head Attention 可解释性** 的论述。
  - 论文认为 self-attention 不仅性能强，而且部分 attention heads 会学习到语言结构。
  - Figure 4 展示的正是这种现象：
    - 没有显式语法标注。
    - 没有专门设计指代消解模块。
    - 模型仍然在翻译训练中自发学到了类似 **anaphora resolution** 的行为。

- **与 RNN / CNN 模型的对比意义**
  - 对 RNN 来说，**its** 和 **Law** 之间的信息传递需要经过多个时间步。
  - 对 CNN 来说，如果卷积核较小，需要多层堆叠才能连接远距离词。
  - 对 Transformer 来说：
    - **its** 可以在一层 self-attention 中直接关注 **Law**。
    - 路径长度为 **O(1)**。
  - 这解释了为什么 Transformer 更擅长捕捉长距离依赖。

- **语言学角度分析**
  - 句子结构可拆解为：

| 片段 | 语法角色 | 说明 |
|---|---|---|
| **The Law** | 主语 / 名词短语 | 后文 **its** 的先行词 |
| **will never be perfect** | 谓语结构 | 描述 Law 的性质 |
| **but** | 转折连词 | 引出对比观点 |
| **its application** | 名词短语 | **its** 指代 Law，application 是中心词 |
| **should be just** | 谓语结构 | 描述 application 应具备的性质 |
| **this is what we are missing** | 解释性从句 | 指出缺失的是公正应用 |
| **in my opinion** | 插入性评价短语 | 表达说话人立场 |

  - 图中最重要的语言学关系是：
    - **The Law ↔ its**
    - **its ↔ application**
  - 这两个关系共同构成：
    - **The Law’s application**

- **视觉细节解读**
  - **左侧 “Law” 处的深紫色竖块**：
    - 表明 **Law** 是被强烈关注的目标。
  - **中部 “its” 处的灰色块**：
    - 表明分析重点是从 **its** 发出的 attention。
  - **从 its 指向 Law 的斜线**：
    - 表明存在远距离依赖。
  - **从 its 指向 application 的斜线**：
    - 表明存在局部短语依赖。
  - 上半部分大量浅线：
    - 表示其他 token 之间也存在注意力，但权重较弱。
  - 下半部分线条极少：
    - 用于突出 **its** 的关键依赖，降低干扰。

- **该图的核心结论**
  - **Transformer 的 encoder self-attention heads 能够捕捉代词与先行词之间的长距离依赖。**
  - **“its” 对 “Law” 的强注意力说明模型可能完成了指代消解。**
  - **不同 attention heads 分工不同，一个 head 偏向语义指代，另一个 head 偏向短语结构。**
  - **注意力分布非常 sharp，说明模型在该位置具有较明确的结构判断。**
  - **这为 Multi-Head Attention 的可解释性提供了直观证据。**

- **局限性说明**
  - 该图只能说明 attention pattern 与指代关系高度相关。
  - 不能严格证明模型“理解”了指代关系。
  - Attention 权重并不总是等价于因果解释。
  - 但在该例中，**its → Law** 的模式与人类语言分析高度一致，因此具有较强解释价值。

- **一句话总结**
  - **Figure 4 展示了 Transformer 第 5 层 encoder self-attention 中某些 heads 对代词 “its” 形成了高度集中的注意力，其中强烈指向 “Law”，说明 Multi-Head Attention 能够自发学习类似 anaphora resolution 的长距离语义依赖。**

### b616696f63c30ad27de5f095b0cac5975742209d1231c70057510010187c1512.jpg

![b616696f63c30ad27de5f095b0cac5975742209d1231c70057510010187c1512.jpg](b616696f63c30ad27de5f095b0cac5975742209d1231c70057510010187c1512.jpg)

- **图片定位**：该图是论文《Attention Is All You Need》附录中的 **Figure 5**，用于展示 **Transformer Encoder self-attention** 中不同 attention heads 学到的结构化行为。

- **图像核心含义**：  
  - 图片展示了两个不同的 **encoder self-attention heads** 的注意力连接模式。  
  - 上半部分为 **绿色 attention head**，下半部分为 **红色 attention head**。  
  - 两个 head 都来自 **encoder self-attention 的第 5 层，共 6 层**。  
  - 图中句子为：**“The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.”**  
  - 上下两排词分别表示同一句话中的 token 位置，连线表示某个 token 对另一个 token 的 attention 权重。  
  - 线越深、越粗，表示 **attention 权重越高**；浅色线表示弱注意力。

- **图像结构说明**：

| 区域 | 颜色 | 表示内容 | 主要观察 |
|---|---|---|---|
| 上半部分 | 绿色 | 一个 encoder self-attention head | 关注较多跨距离依赖和句法/语义相关词 |
| 下半部分 | 红色 | 另一个 encoder self-attention head | 更多呈现局部或对齐式结构关系 |
| 上下两行 token | 黑色文字 | 同一句子的不同位置表示 | attention 从一个位置指向另一个位置 |
| 深色连线 | 绿色/红色 | 高权重 attention | 表示模型重点关注的 token 关系 |
| 浅色连线 | 淡绿色/淡红色 | 低权重 attention | 表示背景性或弱关联注意力 |

- **句子 token 分布**：

| 序列片段 | Token |
|---|---|
| 主句部分 | The / Law / will / never / be / perfect |
| 转折部分 | , / but / its / application / should / be / just |
| 插入符号 | - |
| 从句部分 | this / is / what / we / are / missing |
| 结尾部分 | , / in / my / opinion / . / `<EOS>` / `<pad>` |

- **上半部分绿色 attention head 分析**：  
  - 绿色 head 展现出明显的 **长距离依赖捕获能力**。  
  - 例如多个 token 指向或关联到 **Law、application、missing、opinion** 等语义核心词。  
  - **Law** 与句首 **The**、以及后续语义成分之间存在较强连接，说明该 head 可能在捕获名词短语或主题中心。  
  - **application** 附近有多条强连接，表明该 head 可能关注 **“its application should be just”** 这一结构中的核心名词及其谓词关系。  
  - **what / we / are / missing** 一组 token 之间存在密集连接，说明该 head 对从句结构具有聚合行为。  
  - **opinion** 与 **in / my / opinion** 附近 token 有明显连接，体现出该 head 能捕获固定短语或介词短语结构。

- **绿色 head 的可能功能**：

| 观察现象 | 可能解释 |
|---|---|
| 多个远距离 token 被连接 | 捕获 **long-distance dependencies** |
| Law、application、missing 等词被重点关注 | 识别句子中的语义中心 |
| what / we / are / missing 之间连接密集 | 识别从句结构 |
| in / my / opinion 内部连接明显 | 捕获短语级结构 |
| `<EOS>` 与 `<pad>` 附近也有连接 | 模型处理序列边界和 padding 信息 |

- **下半部分红色 attention head 分析**：  
  - 红色 head 的注意力模式更加规则，很多连接接近垂直或短距离斜线。  
  - 这说明它可能更关注 **局部位置关系** 或 **相邻词对齐关系**。  
  - 多个 token 指向自身或邻近 token，例如 **Law → Law**、**never → never**、**perfect → perfect**、**just → just** 等，表现出类似 identity 或 positional tracking 的作用。  
  - 在 **what / we / are / missing** 附近，也可以看到集中连接，说明该 head 对局部从句结构仍有一定敏感性。  
  - 在 **in / my / opinion** 附近，红色 head 有较强连接，可能用于识别介词短语内部组合关系。

- **红色 head 的可能功能**：

| 观察现象 | 可能解释 |
|---|---|
| 大量近似垂直连接 | 可能进行 **self-position tracking** 或局部复制 |
| 相邻 token 间连接较多 | 捕获局部上下文 |
| what / missing 附近有较强注意力 | 关注从句局部结构 |
| in / my / opinion 内部连接明显 | 捕获固定短语 |
| 与绿色 head 相比远距离连接较少 | 更偏向局部句法功能 |

- **两个 attention heads 的对比**：

| 对比维度 | 绿色 head | 红色 head |
|---|---|---|
| 关注范围 | 更全局 | 更局部 |
| 长距离依赖 | 明显 | 较弱 |
| 句法结构 | 较复杂 | 较规则 |
| 语义聚合 | 更强 | 中等 |
| 位置跟踪 | 较弱 | 较强 |
| 可能功能 | 语义关系、短语结构、长距离依赖 | 局部依赖、位置对齐、短语边界 |

- **与 Transformer 机制的关系**：  
  - 该图体现了 **Multi-Head Attention** 的核心优势：不同 head 可以在不同表示子空间中学习不同类型的关系。  
  - 绿色 head 和红色 head 并没有学习完全相同的 attention pattern，而是呈现出功能分化。  
  - 这支持论文中关于 **multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions** 的论点。  
  - 如果只有单个 attention head，模型可能会把多种关系平均在一起，导致注意力模式不清晰；多头机制则允许某些 head 专注语义关系，另一些 head 专注局部结构或位置关系。

- **从句法角度看图像**：  
  - 句子中存在多个明显结构：  
    - **The Law**：名词短语。  
    - **will never be perfect**：谓语结构。  
    - **its application**：名词短语，其中 **its** 指代前文 **Law**。  
    - **should be just**：谓语结构。  
    - **what we are missing**：宾语从句或名词性从句。  
    - **in my opinion**：介词短语。  
  - 图中的 attention heads 似乎分别捕获了这些结构中的不同层级关系。

- **从语义角度看图像**：  
  - 句子的核心语义是：**法律不会完美，但它的应用应该公正，而这正是我们所缺失的。**  
  - 绿色 head 对 **Law、application、just、missing、opinion** 等关键词的关注较强，说明它可能在捕获句子的语义主干。  
  - 红色 head 则更像是在维护 token 的局部结构，使模型保持词序和短语组合信息。

- **重要现象：attention head 的功能分化**：  
  - 该图最重要的结论不是某一条具体连线，而是两个 head 的整体行为差异。  
  - **绿色 head 更像语义/结构 head**。  
  - **红色 head 更像位置/局部依赖 head**。  
  - 这说明 Transformer 并不是简单地“看所有词”，而是通过不同 attention heads 学习多种互补关系。

- **与 Figure 3 和 Figure 4 的联系**：  
  - Figure 3 展示 attention 能捕获长距离依赖，例如 **making ... more difficult**。  
  - Figure 4 展示 attention 可能参与 **anaphora resolution**，例如代词 **its** 的指代关系。  
  - Figure 5 则进一步说明：不同 heads 会学习到与 **sentence structure** 相关的不同行为。  
  - 三者共同支撑论文观点：**self-attention 不仅高效，也具有一定可解释性**。

- **技术意义**：  
  - 该图为 Transformer 的可解释性提供了视觉证据。  
  - 它表明 self-attention 可以自动学习：  
    - **局部依赖**  
    - **长距离依赖**  
    - **短语结构**  
    - **从句结构**  
    - **语义核心词关系**  
    - **序列边界信息**  
  - 这也是 Transformer 能够替代 RNN 和 CNN 的关键原因之一。

- **局限性说明**：  
  - 虽然图中 attention pattern 看起来与句法结构相关，但不能简单等同于严格的语言学句法树。  
  - attention 权重表示的是模型内部的信息路由或相关性，不一定完全等价于因果解释。  
  - 图像展示的是个例，不能单独证明所有 heads 都具备稳定的语言学功能。  
  - 但作为定性分析，它很好地说明了 **multi-head self-attention 的表达能力和可解释潜力**。

- **总结**：  
  - 这张图展示了 Transformer encoder 中两个不同 self-attention heads 的行为差异。  
  - **绿色 head** 更偏向捕获全局语义关系和长距离结构。  
  - **红色 head** 更偏向局部位置、邻近依赖和短语内部结构。  
  - 图像直观证明了 **Multi-Head Attention** 的价值：不同 heads 可以自动分工，分别学习句子中的不同结构关系。  
  - 该图也是论文中支持“attention heads can learn syntactic and semantic structure”的重要可视化证据。

