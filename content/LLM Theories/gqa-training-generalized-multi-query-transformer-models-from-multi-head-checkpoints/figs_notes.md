# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints 图表详解

### Overview of conversion from multi-head to multi-query attention. Key and value projection matrices from all heads are mean pooled into a single head.

![recycling.png](recycling.png)

- **图片核心含义**：该图展示了如何将已有 **Multi-Head Attention, MHA** checkpoint 中的多个 **Key projection matrices** 转换为 **Multi-Query Attention, MQA** 所需的单个 **Key projection matrix**。转换方式是对所有 attention heads 的 Key 投影矩阵进行 **Mean Pool**，即逐元素平均。

- **图中流程概览**：

| 图中部分 | 含义 | 维度标注 | 作用 |
|---|---|---:|---|
| 左侧多个矩形 | 原始 **MHA** 中每个 head 的 **Key Projection** | 每个为 **d_model × d_h** | 每个 query head 拥有独立 Key projection |
| “Key Projection K₁, K₂, …, K_H” | 第 1 到第 H 个 Key head 的投影矩阵 | H 个矩阵 | 表示 MHA 的多 Key head 结构 |
| 中间蓝色模块 | **Mean Pool** | 对 H 个矩阵求平均 | 将多个 Key projection 合并 |
| 右侧矩形 | 转换后的 **MQA Key Projection K_MQ** | **d_model × d_h** | 所有 query heads 共享的单个 Key projection |

- **数学表达**：

| 转换对象 | 公式 | 说明 |
|---|---|---|
| **Key projection** | **K_MQ = (1 / H) Σᵢ₌₁ᴴ Kᵢ** | 对 H 个 Key projection matrices 做逐元素平均 |
| **Value projection** | **V_MQ = (1 / H) Σᵢ₌₁ᴴ Vᵢ** | 论文说明 Value projection 也采用同样方式 |
| **Query projection** | **不合并** | MQA 仍保留多个 query heads |

- **维度解释**：

| 符号 | 含义 |
|---|---|
| **d_model** | Transformer hidden dimension，即模型主隐藏维度 |
| **d_h** | 单个 attention head 的维度 |
| **H** | 原始 MHA 中 attention heads 的数量 |
| **Kᵢ** | 第 i 个 head 的 Key projection matrix |
| **K_MQ** | 转换后 MQA 使用的共享 Key projection matrix |

- **视觉结构分析**：
  - 左侧的多个灰色长矩形表示 **MHA 中不同 head 的 Key projection matrices**。
  - 每个矩形高度对应 **d_h**，宽度对应 **d_model**，说明每个 head 都从完整 hidden representation 投影到一个 head-specific key space。
  - 中间的蓝色竖条 **Mean Pool** 表示合并操作，不是训练层，而是 checkpoint conversion 时的参数初始化策略。
  - 右侧只剩一个 **Key Projection K_MQ**，表示转换为 **MQA** 后所有 query heads 共享同一个 Key projection。
  - 图中只画了 **Key projection**，但论文明确说明 **Value projection matrices** 也以同样方式处理。

- **该图对应的 checkpoint conversion 过程**：

| 步骤 | 操作 | 目的 |
|---|---|---|
| 1 | 从已有 **MHA checkpoint** 中读取所有 Key heads 参数 | 利用已有预训练知识 |
| 2 | 对 **K₁, K₂, …, K_H** 做 Mean Pool | 得到单个共享 Key head |
| 3 | 对 **V₁, V₂, …, V_H** 做 Mean Pool | 得到单个共享 Value head |
| 4 | 保留 Query projections 的多头结构 | 维持多个 query heads 的表达能力 |
| 5 | 继续进行少量 **uptraining** | 让模型适应新的 attention 结构 |

- **为什么使用 Mean Pool**：
  - **Mean Pool 最大程度保留原始 checkpoint 信息**。
  - 相比选择单个 head，例如只用 **K₁**，平均多个 heads 可以整合所有 heads 中学到的统计结构。
  - 相比随机初始化，Mean Pool 不会完全破坏已有预训练参数，因此收敛更快、质量更好。
  - 论文 ablation 表明转换策略效果通常为：**Mean Pool > First Head > Random Initialization**。

- **从 MHA 到 MQA 的结构变化**：

| 属性 | MHA | MQA |
|---|---|---|
| Query heads | **H 个** | **H 个** |
| Key heads | **H 个** | **1 个** |
| Value heads | **H 个** | **1 个** |
| KV cache 大小 | 大 | 显著变小 |
| 解码内存带宽压力 | 高 | 低 |
| 推理速度 | 慢 | 快 |
| 潜在质量 | 高 | 可能下降 |

- **该转换的关键动机**：
  - Autoregressive decoding 时，每生成一个 token 都需要读取历史 **Key/Value cache**。
  - 在 **MHA** 中，每层每个 head 都保存独立 K/V cache，内存带宽开销很大。
  - **MQA** 通过共享单个 Key head 和 Value head，大幅减少 KV cache 体积。
  - 图中的 Mean Pool 操作是为了让已有 MHA 模型低成本转成 MQA，而不是从头训练一个 MQA 模型。

- **性能与工程意义**：
  - 该图体现的是一种 **checkpoint recycling / upcycling** 思路：复用已有 MHA checkpoint，而非重新训练。
  - 论文中使用约 **5% original pre-training compute** 进行 uptraining，即可获得接近原模型质量的 MQA/GQA 模型。
  - 对推理系统而言，该方法可以显著降低 **memory bandwidth bottleneck**，特别适合长序列生成和大 batch decoding。

- **该图与 GQA 的关系**：
  - 图中展示的是 **MHA → MQA**，即所有 heads 合并成一个共享 Key/Value head。
  - 对 **Grouped-Query Attention, GQA**，操作类似，但不是把所有 H 个 heads 平均成 1 个，而是按 group 分组平均。
  - 若有 **G** 个 groups，则每组内部 mean-pool 得到一个 Key head 和一个 Value head。
  - 因此：
    - **GQA-1 = MQA**
    - **GQA-H = MHA**
    - **GQA-G** 是二者之间的折中。

- **重要结论**：
  - recycling.png 展示了论文方法的核心初始化策略：**通过 Mean Pool 将 MHA 的多个 Key/Value heads 压缩为 MQA 的单个 Key/Value head**。
  - 该策略兼顾了 **参数继承**、**推理加速** 和 **训练稳定性**。
  - 它不是完整训练方法，而是 uptraining 前的 **checkpoint conversion step**。
  - 该转换为后续少量 pre-training 提供了良好初始化，是论文能够用低成本获得高质量 MQA/GQA 模型的关键。

### image

![gmq-architecture.png](gmq-architecture.png)

- **图片核心含义**：该图对比了三种 Transformer Attention 结构中 **Queries、Keys、Values** 的共享方式：
  - **Multi-head attention, MHA**：每个 query head 都有独立的 key head 和 value head。
  - **Grouped-query attention, GQA**：多个 query heads 被划分为若干组，每组共享一个 key head 和一个 value head。
  - **Multi-query attention, MQA**：所有 query heads 共享同一个 key head 和同一个 value head。

- **图中视觉元素对应关系**：

| 视觉元素 | 含义 | 颜色/样式 |
|---|---|---|
| **蓝色矩形** | **Queries / Query heads** | 位于底部 |
| **粉色矩形** | **Keys / Key heads** | 位于中部 |
| **橙色矩形** | **Values / Value heads** | 位于顶部 |
| **虚线连接** | Query heads 与其共享/对应的 Key head 的关系 | 表示注意力中 Q-K 匹配结构 |
| **三列布局** | 三种 Attention 机制对比 | 从左到右：MHA、GQA、MQA |

- **左侧 Multi-head 结构分析**：
  - 图中展示了 **8 个 query heads**。
  - 每个 query head 都对应一个独立的 **key head**。
  - 每个 key head 上方也对应一个独立的 **value head**。
  - 因此结构关系是 **1 Query head : 1 Key head : 1 Value head**。
  - 该结构表达能力最强，因为每个 attention head 都可以学习独立的 K/V 表示。
  - 但推理时需要缓存和读取所有 heads 的 **KV-cache**，内存带宽开销最大。

| 结构 | Query heads 数量 | Key heads 数量 | Value heads 数量 | KV-cache 开销 | 表达能力 |
|---|---:|---:|---:|---|---|
| **Multi-head** | 8 | 8 | 8 | **最高** | **最强** |

- **中间 Grouped-query 结构分析**：
  - 图中展示了 **8 个 query heads** 被分成 **4 组**。
  - 每组包含 **2 个 query heads**。
  - 每组共享一个 **key head** 和一个 **value head**。
  - 因此图中 GQA 的结构近似为 **GQA-4**，即 4 个 KV groups。
  - 它位于 **MHA 与 MQA 之间**，是一种折中设计。
  - 相比 MHA，GQA 减少了 Key/Value heads 数量，从而降低 **KV-cache** 大小和推理内存带宽。
  - 相比 MQA，GQA 保留多个 key/value heads，因此模型容量和质量损失更小。

| 结构 | Query heads 数量 | Groups 数量 | Key heads 数量 | Value heads 数量 | 共享方式 |
|---|---:|---:|---:|---:|---|
| **Grouped-query** | 8 | 4 | 4 | 4 | 每 2 个 query heads 共享 1 组 K/V |

- **右侧 Multi-query 结构分析**：
  - 图中同样有 **8 个 query heads**。
  - 但所有 query heads 只共享 **1 个 key head** 和 **1 个 value head**。
  - 这是最激进的 KV 压缩方式。
  - 它显著降低推理时的 **KV-cache** 读取量。
  - 但由于所有 heads 使用同一套 K/V 表示，模型表达能力下降，可能导致质量退化或训练不稳定。

| 结构 | Query heads 数量 | Key heads 数量 | Value heads 数量 | KV-cache 开销 | 潜在问题 |
|---|---:|---:|---:|---|---|
| **Multi-query** | 8 | 1 | 1 | **最低** | **质量下降、训练不稳定风险更高** |

- **三种结构的本质差异**：

| Attention 类型 | K/V 共享粒度 | 与 MHA 的关系 | 与 MQA 的关系 | 主要优势 | 主要代价 |
|---|---|---|---|---|---|
| **MHA** | 不共享，每个 query head 独立 K/V | 原始结构 | 最远 | **质量最好、容量最大** | **推理慢、KV-cache 大** |
| **GQA** | 组内共享 K/V | MHA 的压缩版 | MQA 的增强版 | **速度接近 MQA，质量接近 MHA** | 容量低于 MHA |
| **MQA** | 所有 query heads 共享同一组 K/V | MHA 的极限压缩 | 特例 GQA-1 | **推理最快、KV-cache 最小** | **质量和稳定性可能下降** |

- **从图中可以看出的关键设计思想**：
  - **Query heads 数量保持不变**：三种结构底部蓝色 query heads 数量相同，说明模型仍保留多 query 投影能力。
  - **压缩对象是 Key/Value heads**：从左到右，粉色和橙色矩形数量逐步减少。
  - **GQA 是连续谱上的中间点**：
    - **GQA-1 = MQA**
    - **GQA-H = MHA**
    - 其中 H 是 query heads 数量。
  - 图中的 GQA 展示的是一种中间配置：不是每个 query 独享 K/V，也不是所有 query 共享一套 K/V，而是 **按组共享**。

- **与论文方法的对应关系**：
  - 论文提出的 **Grouped-query attention** 正是图中中间部分。
  - 其目标是解决 **MQA 虽快但质量下降** 的问题。
  - 在从 MHA checkpoint 转换到 GQA checkpoint 时，论文采用：
    - 将原始 MHA 中同一组内的 **key projection matrices** 做 **mean pooling**。
    - 将同一组内的 **value projection matrices** 做 **mean pooling**。
  - 这样可以最大限度保留原始 MHA checkpoint 中的知识，而不是随机初始化或只选择某一个 head。

- **图中结构对推理性能的影响**：

| 指标 | MHA | GQA | MQA |
|---|---|---|---|
| **KV-cache 大小** | 最大 | 中等 | 最小 |
| **内存带宽压力** | 最大 | 显著降低 | 最低 |
| **decoder inference 速度** | 最慢 | 接近 MQA | 最快 |
| **模型质量** | 最好 | 接近 MHA | 可能下降 |
| **训练稳定性** | 通常稳定 | 较稳定 | 可能不稳定 |
| **适合场景** | 质量优先 | 质量-速度平衡 | 极致推理速度优先 |

- **图中隐含的数学关系**：
  - 设 query heads 数量为 **H**，GQA groups 数量为 **G**。
  - **MHA**：
    - Key heads = **H**
    - Value heads = **H**
  - **MQA**：
    - Key heads = **1**
    - Value heads = **1**
  - **GQA**：
    - Key heads = **G**
    - Value heads = **G**
    - 每组 query heads 数量约为 **H / G**
  - 因此 KV-cache 相对 MHA 的压缩比例约为：

| 结构 | Key/Value heads 数量 | 相对 MHA 的 KV-cache |
|---|---:|---:|
| **MHA** | H | 1 |
| **GQA** | G | G / H |
| **MQA** | 1 | 1 / H |

- **结合论文实验结果的解释**：
  - 论文中使用 **GQA-8-XXL**，即 8 个 key/value groups。
  - 实验显示：
    - **MHA-XXL** 推理时间为 **1.51s**，平均性能 **47.2**。
    - **MQA-XXL** 推理时间为 **0.24s**，平均性能 **46.6**。
    - **GQA-8-XXL** 推理时间为 **0.28s**，平均性能 **47.1**。
  - 这说明图中中间的 **Grouped-query** 结构几乎保留了 MHA 的质量，同时推理速度接近 MQA。

| 模型 | Attention 类型 | 推理时间 T_infer | 平均性能 | 结论 |
|---|---|---:|---:|---|
| **MHA-XXL** | Multi-head | 1.51s | 47.2 | 质量最高但最慢 |
| **MQA-XXL** | Multi-query | 0.24s | 46.6 | 最快但质量下降 |
| **GQA-8-XXL** | Grouped-query | 0.28s | 47.1 | **速度接近 MQA，质量接近 MHA** |

- **该图的核心结论**：
  - **MHA**：每个 query head 拥有独立 K/V，质量强但推理成本高。
  - **MQA**：所有 query heads 共享单个 K/V，推理快但压缩过强。
  - **GQA**：在多个 query head 之间分组共享 K/V，是更优的折中方案。
  - 图像直观说明了论文的主要贡献：**通过减少 Key/Value heads 而不是减少 Query heads，实现推理加速，同时尽量保留模型质量**。

