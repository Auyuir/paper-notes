# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, et al.

**发表期刊/会议 (Journal/Conference)**: ArXiv

**发表年份 (Publication Year)**: 2023

**研究机构 (Affiliations)**: University of Southern California, Google Research

---

## 1. 摘要

**目的**

- 解决 Transformer 模型自回归解码推理阶段因加载 Decoder 权重和 Attention Keys/Values 导致的**内存带宽瓶颈**。
- 克服 Multi-query attention (MQA) 仅使用单组 Key/Value Head 带来的**质量下降**与**训练不稳定**问题。
- 避免为追求推理速度而从头单独训练新模型的高昂成本。

---

**方法**

- 提出 **Uptraining** 机制：
  - 将现有 Multi-head attention (MHA) Checkpoint 转换为 MQA 或 GQA 结构。
  - 转换策略：将所有原始 Key 和 Value Projection Matrices 进行 **mean-pooling**（均值池化），合并为单个或分组 Head。
  - 适应性训练：使用原始预训练设置和数据集，继续预训练原计算量 **5%** ($\alpha=0.05$) 的步数。
  - ![](recycling.png) *Overview of conversion from multi-head to multi-query attention. Key and value projection matrices from all heads are mean pooled into a single head.*
- 提出 **Grouped-query attention (GQA)** 架构：
  - 作为 MHA 与 MQA 的插值方案，将 Query Heads 划分为 $G$ 组。
  - 每组共享单个 Key Head 和 Value Head。
  - **GQA-1** 等同于 MQA，**GQA-H** 等同于 MHA。
  - ![](gmq-architecture.png) *image*
- 应用范围限制：仅应用于 Decoder 的 self-attention 和 cross-attention，不应用于 Encoder self-attention。

---

**结果**

- 性能与速度权衡：
  - Uptrained MQA-XXL 相比 MHA-Large 实现更高质量与更快推理速度。
  - Uptrained **GQA-8-XXL** 质量逼近 MHA-XXL，推理速度接近 MQA-XXL。
- 核心指标对比（基于 T5-XXL 架构）：

| Model | T_infer (s) | Average Score | CNN (R1) | TriviaQA (F1) |
| :--- | :---: | :---: | :---: | :---: |
| MHA-XXL | 1.51 | 47.2 | 43.8 | 81.9 |
| MQA-XXL | 0.24 | 46.6 | 43.0 | 81.3 |
| GQA-8-XXL | 0.28 | 47.1 | 43.5 | 81.6 |

- 消融实验发现：
  - Checkpoint 转换：**Mean-pooling** 保留信息最多，效果优于选择 First head 或 Random 初始化。
  - Uptraining 步数：GQA 转换后即具可用性，MQA 必须经过 Uptraining；5% Uptraining 效果显著，10% 出现边际收益递减。
  - Group 数量：从 1 (MQA) 增至 8 groups 仅带来轻微推理延迟，继续增加至 MHA 规模则成本显著上升，**8 groups** 为最佳平衡点。
  - 训练稳定性：MQA 在微调长输入任务时易出现训练不稳定（loss spikes），而 GQA 表现稳定。

---

**结论**

- **GQA** 成功在 MHA 的高质量与 MQA 的高速度之间取得最优折中。
- 通过仅需 **5%** 原始预训练算力的 Uptraining 方法，可低成本地将现有 MHA 模型转化为高效的 GQA 模型。
- 有效缓解了长序列生成中加载 Keys 和 Values 带来的内存带宽开销。

---

## 2. 背景知识与核心贡献

**研究背景**

- **Autoregressive decoder inference** 是 **Transformer** 模型的严重瓶颈。
- 瓶颈源于在每个 decoding step 加载 decoder weights 和所有 attention keys/values 时的 **memory bandwidth overhead**。
- **Multi-query attention (MQA)** 通过使用多个 query heads 但单个 key 和 value head 来大幅减少 **KV cache** 的加载量，从而加速推理。

---

**研究动机**

- **MQA** 会导致 **quality degradation** 和 **training instability**。
- 训练专门针对推理优化的独立模型成本高昂且通常不可行。
- 许多现有的语言模型（如 T5, LLaMA）使用的是 **Multi-Head Attention (MHA)**，而非 **MQA**。
- 需要一种方法，既能利用现有的 **MHA checkpoints** 获得快速推理能力，又能避免 **MQA** 的质量下降。

---

**核心贡献**

- 提出 **uptraining** 方法，将现有的 **MHA** 语言模型 checkpoints 转换为 **MQA** 模型，仅需原预训练计算量的 **5%**。
  - 转换过程通过将 key 和 value projection matrices 进行 mean-pooling 实现。
  - ![](recycling.png) *Overview of conversion from multi-head to multi-query attention. Key and value projection matrices from all heads are mean pooled into a single head.*
- 提出 **Grouped-query attention (GQA)**，一种介于 **MHA** 和 **MQA** 之间的插值方法。
  - 使用中间数量（多于1个，少于 query heads 数量）的 key-value heads。
  - **GQA-1** 等价于 **MQA**，**GQA-h** 等价于 **MHA**。
  - ![](gmq-architecture.png) *image*
- 证明 **uptrained GQA** 在质量上接近 **MHA**，在速度上接近 **MQA**。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**核心架构概述**
本文提出一种基于 Transformer 的注意力机制优化架构 **Grouped-Query Attention (GQA)**，通过 **Uptraining** 机制将现有的 **Multi-Head Attention (MHA)** 模型转换为 **GQA** 或 **Multi-Query Attention (MQA)** 模型，旨在解决自回归解码器推理阶段的内存带宽瓶颈，同时保持模型质量。

**Uptraining 转换流程**
- **Checkpoint 转换**：
  - 将原始 MHA 模型中的 Key 和 Value 投影矩阵进行 **mean-pooling**（平均池化），合并为目标结构所需的 head 数量。
  - 相比于选择单个 head 或随机初始化，mean-pooling 能最大程度保留预训练模型的信息。
- **额外预训练**：
  - 使用与原始预训练相同的设置和数据集。
  - 仅需原预训练计算量的 **5%** ($\alpha=0.05$) 即可完成适应性训练。
- **转换示意图**：
![](recycling.png) *Overview of conversion from multi-head to multi-query attention. Key and value projection matrices from all heads are mean pooled into a single head.*

**Grouped-Query Attention (GQA) 机制**
- **分组共享机制**：
  - 将 Query heads 划分为 $G$ 个 **groups**。
  - 每个 group 内部共享单一的 Key head 和 Value head。
- **架构边界**：
  - **GQA-1**：仅有一个 group，等同于 **MQA**。
  - **GQA-h**：groups 数量等于 heads 数量，等同于 **MHA**。
- **MHA 到 GQA 的转换**：
  - 在构建每个 group 的 Key 和 Value head 时，对该 group 内包含的所有原始 MHA heads 进行 **mean-pooling**。
- **架构对比图**：
![](gmq-architecture.png) *image*

**架构应用范围与优势**
- **应用层级限制**：
  - 仅应用于 **decoder self-attention** 和 **cross-attention**。
  - 不应用于 **encoder self-attention**（因其并行计算，内存带宽非主要瓶颈）。
- **扩展性与效率优势**：
  - 随着模型规模增大，GQA 能保持带宽和容量的同比例下降，避免 MQA 过于激进的裁剪。
  - 消除大模型标准分片中复制单一 Key/Value head 造成的资源浪费。
  - 在推理速度和模型质量之间提供最优的插值平衡点。

**性能与推理效率对比**
| Model | T_infer (s) | Average | CNN (R1) | arXiv (R1) | PubMed (R1) | MediaSum (R1) | MultiNews (R1) | WMT (BLEU) | TriviaQA (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MHA-Large | 0.37 | 46.0 | 42.9 | 44.6 | 46.2 | 35.5 | 46.6 | 27.7 | 78.2 |
| MHA-XXL | 1.51 | 47.2 | 43.8 | 45.6 | 47.5 | 36.4 | 46.9 | 28.4 | 81.9 |
| MQA-XXL | 0.24 | 46.6 | 43.0 | 45.0 | 46.9 | 36.1 | 46.5 | 28.5 | 81.3 |
| GQA-8-XXL | 0.28 | 47.1 | 43.5 | 45.4 | 47.7 | 36.3 | 47.2 | 28.4 | 81.6 |

### 1. Grouped-query Attention (GQA)

---
**核心概念与原理**

**Grouped-query Attention (GQA)** 是一种介于 **Multi-head Attention (MHA)** 与 **Multi-query Attention (MQA)** 之间的注意力机制泛化形式。
- 将 Query heads 划分为 $G$ 个 **group**。
- 每个 group 共享单一的 **key head** 和 **value head**。
- **极端情况转化**：
  - **GQA-1**：仅包含单个 group，等同于 **MQA**。
  - **GQA-h**：group 数量等于 Query heads 数量，等同于 **MHA**。
- 通过调整 group 数量 $G$，在模型质量与推理速度之间寻找最优插值点。

![](gmq-architecture.png) *image*

---
**算法流程与参数设置**

**Checkpoint 转换流程**
- 从现有的 **MHA** checkpoint 初始化。
- 将原始的 **key** 和 **value** projection matrices 划分至对应的 $G$ 个 group 中。
- 对每个 group 内的所有原始 heads 执行 **mean-pooling** 操作，生成该 group 共享的单一 key head 和 value head。
- 相比于随机初始化或直接选取第一个 head，**mean-pooling** 最大程度保留了预训练模型的信息，效果最优。

**Uptraining 阶段**
- 转换后的 checkpoint 需要进行额外的预训练以适应新结构。
- 预训练计算量设置为原模型预训练计算量的 $\alpha$ 比例。
- 实验表明，$\alpha = 0.05$（即 5% 的原始预训练步数）即可使模型收敛并达到优异性能，继续增加至 10% 收益递减。
- 使用与原模型相同的预训练数据集和超参数设置。

**参数配置策略**
- **group 数量 $G$**：实验选定 **GQA-8** 作为最优中间点。
  - 从 MQA (1 group) 增加到 8 groups，推理速度仅有轻微下降。
  - 继续增加 groups 数量逼近 MHA 时，推理开销显著上升。
- **应用层级限制**：仅应用于 **decoder self-attention** 和 **cross-attention**。
  - **encoder self-attention** 采用并行计算，内存带宽非主要瓶颈，不应用 GQA。

---
**输入输出关系与整体作用**

**输入输出关系**
- **输入**：与标准 Attention 一致，接收隐藏状态张量。
- **内部处理**：
  - Query 矩阵通过投影生成 $H$ 个独立的 Query heads。
  - Key 和 Value 矩阵通过投影仅生成 $G$ 个 heads（$G < H$）。
  - 每个 Key/Value head 被对应的 $H/G$ 个 Query heads 复用计算 Attention Score。
- **输出**：各 Query head 计算得出的上下文向量拼接后经线性投影输出，维度与 **MHA** 保持一致。

**在整体模型中的作用**
- **降低内存带宽开销**：自回归解码阶段，加载 **KV cache** 是核心瓶颈。GQA 将 KV cache 规模缩减为原来的 $1/G$，大幅减少每步解码的数据加载量。
- **优化大模型扩展性**：
  - 大模型通常增加 head 数量 $H$，若使用 MQA 会导致带宽和容量削减过于激进。
  - GQA 允许保持与模型规模成比例的带宽和容量缩减。
  - 消除大模型张量并行中单 KV head 被强制复制造成的算力与显存浪费。
- **平衡质量与速度**：

| **Model** | **T<sub>infer</sub> (s)** | **Average** |
|:---|:---:|:---:|
| MHA-XXL | 1.51 | 47.2 |
| MQA-XXL | 0.24 | 46.6 |
| GQA-8-XXL | 0.28 | 47.1 |

- **性能表现**：**GQA-8-XXL** 推理速度接近 **MQA-XXL**，质量逼近 **MHA-XXL**，提供最佳 Trade-off。
- **训练稳定性**：**MQA** 在微调长输入任务时易出现损失尖峰和发散。**GQA** 结构在 uptraining 和微调过程中表现出高度稳定性。

### 2. MHA-to-MQA/GQA Uptraining

**核心原理**

- 从现有的 Multi-Head Attention (MHA) 语言模型 checkpoint 出发，通过结构转换与少量继续预训练，将其转变为 Multi-Query Attention (MQA) 或 Grouped-Query Attention (GQA) 模型。
- 核心目标是以极低的计算成本（**5%** 的原始预训练计算量）获取具备高效推理能力的模型，同时避免从头训练 MQA 带来的质量退化与训练不稳定问题。

![](recycling.png) *Overview of conversion from multi-head to multi-query attention. Key and value projection matrices from all heads are mean pooled into a single head.*

**算法流程与参数设置**

- **Checkpoint 转换阶段**
  - 针对 MQA：将 MHA 中所有的 key 和 value projection matrices 进行 **mean pooling**，合并成单个 key head 和 value head。
  - 针对 GQA：将 query heads 划分为 $G$ 个 groups，对每个 group 内的原始 heads 进行 **mean pooling**，生成该 group 共享的 key head 和 value head。
  - 转换策略对比：**Mean pooling** 效果最优，优于选择单个 head (First) 或随机初始化，因其能最大程度保留预训练模型的信息分布。
- **继续预训练阶段**
  - 采用与原始模型完全相同的预训练设置与数据集。
  - 训练步数设置为原始预训练步数的 $\alpha$ 比例。
  - 关键参数：$\alpha = 0.05$。在 T5 XXL 规模模型上，此阶段约消耗 **600 TPUv3 chip-days**。
  - 收益边际效应：MQA 必须经过 uptraining 才能具备可用性；GQA 在转换后即有合理性能，在 $\alpha=0.05$ 时性能显著提升，但增至 $\alpha=0.1$ 时呈现收益递减趋势。

**输入输出关系与整体作用**

- **输入**：基于 MHA 架构的预训练语言模型 checkpoint（如 T5.1.1）。
- **输出**：具备 MQA 或 GQA 架构的高效推理语言模型 checkpoint。
- **整体作用**：
  - **突破内存带宽瓶颈**：大幅减少 autoregressive decoder 推理时的 memory bandwidth overhead，KV cache 的数据加载量急剧下降。
  - **优化质量与速度权衡**：Uptrained MQA 提供了优于 MHA-Large 的质量与速度折中；Uptrained GQA 在保持接近 MQA 推理速度的同时，质量逼近原始的 MHA-XXL。
  - **增强训练稳定性**：从头训练 MQA 易在 fine-tuning 阶段引发 loss spikes 甚至发散；Uptrained MQA 稳定性有所改善，而 Uptrained GQA 表现出完全稳定的训练动态。

| 模型变体 | 推理时间 (s) | 平均性能 | 核心特征 |
| :--- | :---: | :---: | :--- |
| **MHA-XXL** | 1.51 | 47.2 | 质量最高，但推理速度最慢 |
| **MQA-XXL** | 0.24 | 46.6 | 速度最快，但存在轻微质量退化 |
| **GQA-8-XXL** | 0.28 | 47.1 | 速度接近 MQA，质量逼近 MHA |


---

## 4. 实验方法与实验结果

**实验设置**

- **模型架构与基础**：基于 **T5.1.1** 架构，使用 **JAX**、**Flax** 和 **Flaxformer** 实现。主要实验对象为 **T5 Large** 和 **T5 XXL** (Multi-Head Attention, MHA)，以及经过 uptraining 的 **T5 XXL** (Multi-Query Attention, MQA 和 Grouped-Query Attention, GQA)。
- **优化器配置**：采用 **Adafactor** 优化器，超参数与学习率调度与原始 T5 保持一致。
- **注意力机制应用范围**：**MQA** 和 **GQA** 仅应用于 **decoder self-attention** 和 **cross-attention**，不应用于 **encoder self-attention**。
- **Uptraining 流程**：从公开 **T5.1.1 checkpoints** 初始化，将 key 和 value heads 进行 **mean-pooling** 以匹配目标结构，随后使用原始预训练设置和数据集进行额外训练，步数为原始预训练步数的 **$\alpha$** 比例（$\alpha=0.05$ 时约需 **600 TPUv3 chip-days**）。
- **评估数据集**：涵盖摘要（**CNN/Daily Mail**, **arXiv**, **PubMed**, **MediaSum**, **MultiNews**）、翻译（**WMT 2014 En-De**）和问答（**TriviaQA**）。未评估 **GLUE** 等分类基准。
- **Fine-tuning 设置**：所有任务采用恒定学习率 **0.001**，batch size **128**，dropout rate **0.1**。不同任务设置不同输入输出长度，训练至收敛并选择 dev 性能最高的 checkpoint，推理使用 **greedy decoding**。
- **Timing 测量**：报告每样本每 **TPUv4 chip** 的时间，使用 **xprof** 测量。使用 **8 TPUs**，最大 batch size 上限为 **32**，针对每个模型单独优化并行化策略。

---

**结果数据**

- **核心结论**：**Uptrained MQA** 相较于 **MHA-Large** 实现了更高质量与更快推理的权衡；**GQA** 在保持与 **MQA** 相似速度的同时，质量接近 **MHA-XXL**。
- **性能与推理时间对比**：

| **Model** | **T_infer (s)** | **Average** | **CNN (R1)** | **arXiv (R1)** | **PubMed (R1)** | **MediaSum (R1)** | **MultiNews (R1)** | **WMT (BLEU)** | **TriviaQA (F1)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MHA-Large** | 0.37 | 46.0 | 42.9 | 44.6 | 46.2 | 35.5 | 46.6 | 27.7 | 78.2 |
| **MHA-XXL** | 1.51 | 47.2 | 43.8 | 45.6 | 47.5 | 36.4 | 46.9 | 28.4 | 81.9 |
| **MQA-XXL** | 0.24 | 46.6 | 43.0 | 45.0 | 46.9 | 36.1 | 46.5 | 28.5 | 81.3 |
| **GQA-8-XXL** | 0.28 | 47.1 | 43.5 | 45.4 | 47.7 | 36.3 | 47.2 | 28.4 | 81.6 |

- **数据解读**：
  - **MHA-XXL** 质量最高（**Average 47.2**），但推理时间最长（**1.51s**）。
  - **MQA-XXL** 推理速度最快（**0.24s**），质量略低于 **MHA-XXL**，但优于 **MHA-Large**。
  - **GQA-8-XXL** 推理速度（**0.28s**）接近 **MQA-XXL**，质量（**Average 47.1**）几乎追平 **MHA-XXL**。

---

**消融实验**

- **Checkpoint 转换方法对比**：

![](recycling.png) *Overview of conversion from multi-head to multi-query attention. Key and value projection matrices from all heads are mean pooled into a single head.*

  - 评估 **T5-Large** uptraining 至 **MQA** ($\alpha=0.05$) 的三种转换方法。
  - **Mean pooling**（平均池化所有 key/value heads）效果最佳。
  - **First**（选择第一个 head）效果次之。
  - **Random**（从头随机初始化）效果最差。
  - 结论：保留预训练模型信息的程度直接决定转换后的性能。
- **Uptraining 步数影响**：
  - **GQA** 在仅进行 checkpoint 转换（无 uptraining）后即可达到合理性能，而 **MQA** 必须经过 uptraining 才具备可用性。
  - **5%** 的 uptraining ($\alpha=0.05$) 能为 **MQA** 和 **GQA** 带来显著收益。
  - **10%** 的 uptraining 呈现边际收益递减。
- **GQA 分组数量影响**：

![](gmq-architecture.png) *image*

  - 评估不同分组数（从 **1** 即 **MQA** 到接近 **MHA**）对推理速度的影响。
  - 从 **1 组** 增加到 **8 组**，推理时间增加幅度较小。
  - 继续增加组数接近 **MHA** 时，推理开销显著增大。
  - 结论：**8 组**（**GQA-8**）是推理速度与模型质量的最佳平衡点。

---

