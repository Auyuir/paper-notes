# Improving Language Understanding by Generative Pre-Training 图表详解

### 2c69fed3c6ecd2c00e8a0cf6d4ca50400f8cce340bc2f13127851049bc9ba718.jpg

![2c69fed3c6ecd2c00e8a0cf6d4ca50400f8cce340bc2f13127851049bc9ba718.jpg](2c69fed3c6ecd2c00e8a0cf6d4ca50400f8cce340bc2f13127851049bc9ba718.jpg)

- **图片类型与核心含义**
  - 该图是论文 Figure 2 左图，用于分析 **“从预训练语言模型迁移多少层 Transformer layer 会影响下游任务性能”**。
  - 横轴是 **of layers transferred**，表示从无监督预训练的 **Transformer LM** 中迁移到监督任务的层数。
  - 图中同时展示两个任务：
    - **RACE**：阅读理解 / Question Answering 任务。
    - **MultiNLI**：Natural Language Inference 任务。
  - 总体结论非常明确：**迁移的预训练层数越多，下游任务性能越好**，说明预训练模型的每一层都学习到了可迁移的语言功能。

- **坐标轴与图例说明**

| 元素 | 含义 |
|---|---|
| 左纵轴 | **RACE Dev Accuracy**，RACE 开发集准确率 |
| 右纵轴 | **MultiNLI Matched Dev Accuracy**，MultiNLI matched 开发集准确率 |
| 实线蓝色 | **RACE Dev** |
| 虚线蓝色 | **RACE Train** |
| 实线橙色 | **MultiNLI Dev** |
| 虚线橙色 | **MultiNLI Train** |

- **主要视觉趋势**
  - **RACE Dev** 随迁移层数增加整体上升：
    - 从约 **40%** 左右逐步提升到约 **60%**。
    - 曲线并非完全线性，中间存在小幅波动，但总体趋势稳定上升。
  - **RACE Train** 上升更明显：
    - 从约 **53%** 左右提升到约 **92%**。
    - 训练集性能显著高于开发集，说明模型容量增强后对训练数据拟合能力明显提升。
  - **MultiNLI Dev** 也随迁移层数增加而提升：
    - 从约 **31%** 提升到约 **56%**。
    - 提升幅度非常大，说明 NLI 任务强烈受益于预训练层迁移。
  - **MultiNLI Train** 从约 **80%** 提升到约 **97%**：
    - 随层数增加持续上升，并在后期趋于饱和。
    - 表明更多预训练层能显著增强模型对自然语言推理训练样本的建模能力。

- **近似读数分析**

| 迁移层数 | RACE Dev 约值 | RACE Train 约值 | MultiNLI Dev 约值 | MultiNLI Train 约值 |
|---:|---:|---:|---:|---:|
| 0 | 40% | 53% | 31% | 80% |
| 1 | 42% | 52% | 38% | 83% |
| 2 | 41% | 52% | 37% | 84% |
| 3 | 47% | 59% | 44% | 89% |
| 4 | 49% | 67% | 45% | 90% |
| 5 | 50% | 72% | 45% | 91% |
| 6 | 53% | 76% | 49% | 94% |
| 7 | 54% | 79% | 51% | 95% |
| 8 | 54% | 79% | 53% | 95% |
| 9 | 55% | 84% | 53% | 96% |
| 10 | 56% | 85% | 54% | 96% |
| 11 | 58% | 92% | 55% | 97% |
| 12 | 59% | 92% | 56% | 97% |

- **对 RACE 的具体分析**
  - **RACE Dev** 从 0 层到 12 层约提升 **19 个百分点**。
  - 曲线在 0 到 2 层之间提升有限，甚至略有波动：
    - 这说明仅迁移 embedding 或少量底层参数，对复杂阅读理解任务帮助有限。
  - 3 层以后开始明显提升：
    - 表明中高层 Transformer 表征对阅读理解任务更关键。
  - 10 到 12 层仍有提升：
    - 说明完整迁移模型仍能带来额外收益。
  - RACE 是需要长文本理解、问题匹配和答案选择的任务，因此它明显受益于 **long-range dependency modeling**。

- **对 MultiNLI 的具体分析**
  - **MultiNLI Dev** 从约 31% 提升到约 56%，提升幅度约 **25 个百分点**。
  - 相比 RACE，MultiNLI Dev 的初始值更低，但随着层数增加提升稳定。
  - MultiNLI 需要判断 premise 与 hypothesis 之间的关系，包括：
    - **entailment**
    - **contradiction**
    - **neutral**
  - 这些判断依赖句法、语义组合、词汇关系和跨句推理，因此更深层的预训练表示非常有价值。
  - 该图支持论文中的观点：**每个 Transformer layer 都包含对下游任务有用的功能模块或语言知识**。

- **Train 与 Dev 的差异**
  - 两个任务中，**Train 曲线均显著高于 Dev 曲线**。
  - 这说明随着迁移层数增加，模型拟合训练集的能力快速增强。
  - 但 Dev 性能也同步提升，表明这不是单纯过拟合，而是预训练知识确实提高了泛化能力。
  - 不过，Train-Dev gap 在高层数时较大：
    - RACE Train 约 92%，RACE Dev 约 59%。
    - MultiNLI Train 约 97%，MultiNLI Dev 约 56%。
  - 这反映出复杂语言理解任务仍存在明显泛化挑战。

- **层数迁移带来的阶段性变化**

| 阶段 | 层数范围 | 现象 | 解释 |
|---|---:|---|---|
| 低层迁移 | 0–2 层 | Dev 提升有限 | 主要迁移词表征和浅层局部模式 |
| 中层迁移 | 3–6 层 | 性能明显提升 | 开始迁移短语、句法和局部语义结构 |
| 高层迁移 | 7–10 层 | 稳定继续提升 | 迁移更复杂的语义组合和跨句依赖能力 |
| 完整迁移 | 11–12 层 | 接近最佳性能 | 完整保留预训练 LM 的语言理解能力 |

- **与论文方法的关系**
  - 论文提出两阶段训练：
    - **Generative pre-training**：在 BooksCorpus 上训练 Transformer LM。
    - **Discriminative fine-tuning**：在具体任务上微调。
  - 该图验证了第一阶段预训练的重要性：
    - 如果只迁移少量层，收益有限。
    - 如果迁移完整 Transformer，性能最佳。
  - 这说明预训练不是只学到了 word embedding，而是学到了分布在多层中的 **hierarchical linguistic representations**。

- **为什么层数越多效果越好**
  - **底层 layer** 更可能捕捉：
    - token identity
    - 局部搭配
    - subword pattern
    - 基础句法线索
  - **中层 layer** 更可能捕捉：
    - phrase structure
    - syntactic dependency
    - entity relation
    - 局部语义组合
  - **高层 layer** 更可能捕捉：
    - discourse-level information
    - long-range dependency
    - sentence-level semantics
    - reasoning-relevant features
  - 因此，完整迁移 12 层能够让模型同时利用低层语言形式信息和高层语义推理信息。

- **对 Transformer 架构的支持**
  - 图中持续上升的趋势支持论文选择 **Transformer decoder** 而非传统 LSTM 的理由。
  - Transformer 的 **masked self-attention** 能够更有效地建模长距离依赖。
  - 对 RACE 这类长文本阅读理解任务尤其重要。
  - 这也呼应论文后续 ablation：
    - **LSTM w/ aux LM** 平均得分为 69.1。
    - **Transformer w/ aux LM** 平均得分为 74.7。
  - 因此，该图间接说明：**Transformer 预训练层中包含更强的可迁移结构化语言知识**。

- **关键结论**
  - **迁移更多预训练层会稳定提升 RACE 和 MultiNLI 的 Dev 性能。**
  - **完整迁移 12 层 Transformer 通常效果最好。**
  - **预训练收益不是只来自 embedding，而是来自整个深层 Transformer 表征。**
  - **每一层都对下游任务提供增益，说明语言模型预训练学习到了层级化、任务相关的语言知识。**
  - **该图是 GPT-style pre-training 有效性的核心证据之一。**

### 25b6444944c565e5b1eb049c975849f8e9b2426aeefee490f47a7c9290b2da1d.jpg

![25b6444944c565e5b1eb049c975849f8e9b2426aeefee490f47a7c9290b2da1d.jpg](25b6444944c565e5b1eb049c975849f8e9b2426aeefee490f47a7c9290b2da1d.jpg)

- **图像类型与出处**
  - 该图是论文《Improving Language Understanding by Generative Pre-Training》中 **Figure 2(right)**。
  - 展示的是：随着 **language model pre-training updates** 增加，模型在若干任务上的 **zero-shot performance** 如何变化。
  - 图中同时比较了 **Transformer** 与 **LSTM** 两种架构在预训练过程中的表现趋势。

- **核心含义**
  - 该图用于支持论文中的一个关键论点：**生成式语言模型预训练不仅学习下一个词预测，还会逐渐获得对下游任务有用的语言能力**。
  - 这些能力包括：
    - **sentiment analysis**
    - **winograd schema resolution**
    - **linguistic acceptability**
    - **question answering**
  - 图中可以看到，随着预训练步数增加，多数任务的相对性能稳步提升，说明 **pre-training 本身会诱导模型学习任务相关能力**。

- **坐标轴说明**

| 元素 | 含义 |
|---|---|
| 横轴刻度 | 对数尺度，约从 **10³** 到 **10⁶** |
| 纵轴 | **Relative Task Performance**，相对任务性能 |
| 纵轴范围 | **0.0 到 1.0** |
| 性能归一化方式 | 论文说明中表示性能被归一化到 **random guess baseline** 与当前 **single-model state-of-the-art** 之间 |

- **图例说明**

| 颜色 / 线型 | 任务或模型 |
|---|---|
| 蓝色实线 | **sentiment analysis** |
| 橙色实线 | **winograd schema resolution** |
| 绿色实线 | **linguistic acceptability** |
| 红色实线 | **question answering** |
| 实线 | **Transformer** |
| 虚线 | **LSTM** |

- **整体趋势**
  - **Transformer 实线整体高于 LSTM 虚线**。
  - **Transformer 的曲线更稳定、更持续上升**。
  - **LSTM 的曲线波动更明显**，特别是在部分任务上存在明显不稳定。
  - 这支持论文观点：**Transformer 的结构化 self-attention memory 更有利于迁移与 zero-shot 行为形成**。

- **各任务表现对比**

| 任务 | Transformer 趋势 | LSTM 趋势 | 观察结论 |
|---|---|---|---|
| **sentiment analysis** | 后期快速提升，最终约 **0.67–0.68** | 提升较慢，最终约 **0.48** | Transformer 优势明显，是提升最突出的任务 |
| **winograd schema resolution** | 稳定上升，最终约 **0.54** | 中期波动明显，最终低于 Transformer | Transformer 在常识与指代推理上更稳定 |
| **linguistic acceptability** | 早期快速提升，随后稳定在约 **0.43–0.45** | 稳定但较低，约 **0.35–0.38** | 语法可接受性能力较早出现 |
| **question answering** | 缓慢但持续上升，最终约 **0.25–0.26** | 基本接近低值，最终约 **0.02–0.04** | QA zero-shot 最难，但 Transformer 仍有明显学习信号 |

- **sentiment analysis 曲线分析**
  - 蓝色实线在 **10⁴ 到 10⁵** 附近出现显著跃升。
  - 从接近 **0** 快速增长到约 **0.6**。
  - 后期趋于平台，最终稳定在 **0.65+**。
  - 说明语言模型在持续预训练中逐渐学到与情感极性相关的统计与语义模式。
  - 论文中 zero-shot sentiment 的启发式方法是：
    - 给句子追加 token **“very”**
    - 限制输出词为 **positive / negative**
    - 根据语言模型概率选择预测结果
  - 因此该曲线说明：**语言模型的词预测分布中已经隐含了情感分类能力**。

- **winograd schema resolution 曲线分析**
  - 橙色实线整体从约 **0.05** 上升到约 **0.54**。
  - 该任务涉及代词消解与常识推理。
  - Transformer 曲线较平滑，说明随着预训练增加，模型对上下文指代关系的把握逐渐增强。
  - LSTM 虚线中期有明显波动，说明其 zero-shot 指代能力不稳定。
  - 这与论文论述一致：**Transformer 相比 LSTM 更适合捕捉长距离依赖与结构化上下文关系**。

- **linguistic acceptability 曲线分析**
  - 绿色实线在早期增长最快。
  - 在约 **10⁴** 附近已经达到接近 **0.4**。
  - 后续提升有限，基本维持在 **0.43–0.45**。
  - 这说明模型较早学到了句法、搭配、语法可接受性等信息。
  - 对应 CoLA 类任务时，语言模型可以通过句子平均 token log-probability 来判断句子是否自然。
  - 该现象说明：**语言建模目标天然会鼓励模型学习语法性判断能力**。

- **question answering 曲线分析**
  - 红色实线整体最低，但呈现缓慢持续上升。
  - 从接近 **0** 增长到约 **0.25**。
  - QA 是图中最困难的 zero-shot 任务，因为它需要：
    - 理解文档
    - 理解问题
    - 比较多个候选答案
    - 进行跨句推理
  - 即便如此，Transformer 仍然随预训练更新表现变好，说明 **LM pre-training 对阅读理解能力有正向贡献**。
  - LSTM 在该任务上的虚线几乎贴近底部，表明其 zero-shot QA 能力非常弱。

- **Transformer 与 LSTM 的关键差异**

| 对比点 | Transformer | LSTM |
|---|---|---|
| 曲线形态 | 多数任务稳定上升 | 波动更大 |
| 最终性能 | 多数任务更高 | 整体偏低 |
| 长距离依赖 | 更强，依赖 self-attention | 受序列递归限制 |
| zero-shot 稳定性 | 更好 | 较差 |
| 迁移潜力 | 更强 | 较弱 |

- **为什么 Transformer 表现更好**
  - **Self-attention** 可以直接建模任意 token 之间的关系。
  - 相比 LSTM 的逐步递归传播，Transformer 更容易捕捉：
    - 长距离依赖
    - 指代关系
    - 篇章结构
    - 问题与答案之间的匹配关系
  - 论文认为这种 **structured attentional memory** 是 Transformer 在 transfer learning 中优于 LSTM 的重要原因。

- **该图与论文主结论的关系**
  - 论文主张：**Generative pre-training + discriminative fine-tuning** 可以显著提升自然语言理解任务。
  - 该图进一步解释了为什么预训练有效：
    - 模型在没有监督 fine-tuning 的情况下，已经逐渐获得部分任务能力。
    - 这些能力在 supervised fine-tuning 时被进一步激活和适配。
  - 因此，fine-tuning 并不是从零学习任务，而是在已有语言能力基础上做任务映射。

- **图中最重要的观察**
  - **预训练步数越多，zero-shot 任务表现总体越好。**
  - **Transformer 的 zero-shot 能力明显强于 LSTM。**
  - **不同任务的能力出现速度不同：**
    - **linguistic acceptability** 较早出现。
    - **sentiment analysis** 中期快速提升。
    - **question answering** 最慢、最难。
    - **winograd schema resolution** 稳步提升，体现常识与指代能力增长。
  - **语言模型预训练会自然产生可迁移的语言理解能力。**

- **潜在局限**
  - 图中表现是 **heuristic zero-shot evaluation**，不是标准 supervised fine-tuning 结果。
  - 纵轴是归一化后的 **Relative Task Performance**，不是原始 accuracy / F1。
  - 不同任务之间的数值不能直接等价比较。
  - 曲线主要用于展示趋势，而非精确评估模型最终能力。
  - zero-shot 方法依赖人工设计的提示或打分方式，例如：
    - sentiment 中使用 **positive / negative**
    - QA 中比较候选答案平均 log-probability
    - CoLA 中使用平均 token log-probability 阈值

- **结论**
  - 该图清晰表明：**Transformer language model 在生成式预训练过程中，会逐渐学习到多种自然语言理解能力**。
  - 相比 **LSTM**，**Transformer** 在 zero-shot 行为上更稳定、性能更高。
  - 这为论文提出的 GPT 式范式提供了实验证据：**大规模 generative pre-training 是构建通用语言理解模型的有效路径**。

