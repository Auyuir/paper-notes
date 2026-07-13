# Improving Language Understanding by Generative Pre-Training 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Alec Radford, Karthik Narasimhan, Tim Salimans, et al.

**发表期刊/会议 (Journal/Conference)**: ArXiv

**发表年份 (Publication Year)**: 2018

**研究机构 (Affiliations)**: OpenAI

---

## 1. 摘要

**目的**

- 缓解自然语言理解任务对大规模人工标注数据的严重依赖。
- 利用丰富的无标签文本语料库学习通用语言表示，并通过最小化的架构改动将其迁移至多种下游任务。
- 探索无监督预训练结合有监督微调的半监督学习范式在自然语言理解中的有效性，旨在构建一个任务无关的通用模型。

---

**方法**

- 采用两阶段训练框架：无监督预训练与有监督微调。

- 无监督预训练：
  - 使用 BooksCorpus 数据集（包含超 7000 本未出版书籍，保留长距离连续文本结构）。
  - 模型架构采用 12 层 Transformer decoder（仅包含 masked self-attention heads）。
  - 优化目标为标准语言模型似然最大化。

- 有监督微调：
  - 将预训练参数适配至目标判别任务。
  - 在最终 Transformer block 的激活值后添加线性输出层进行预测。
  - 引入辅助语言建模目标（权重 $\lambda=0.5$）以提升泛化能力并加速收敛。

- 针对特定任务的输入变换：
  - 采用遍历式方法，将结构化输入转化为单一连续 Token 序列，避免对模型架构进行大量任务特定修改。
  - 文本蕴含：拼接前提 与假设，中间插入分隔 Token (`$`)。
  - 语义相似性：包含两种可能的句子顺序，独立处理后元素相加。
  - 问答与常识推理：拼接上下文、问题及候选答案，独立处理后通过 softmax 归一化。

![](2c30b602030ba602a6891860329f4ebe9dbe5eabfbe2e49fa9f7fc3469333c1d.jpg) *Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer.*

- 模型规格与超参数设置：

| 参数类别 | 具体配置 |
| :--- | :--- |
| **模型维度** | 768 维状态，12 个 Attention heads |
| **前馈网络** | 3072 维内部状态 |
| **优化器** | Adam，最大学习率 2.5e-4，余弦退火 |
| **正则化** | Dropout 概率 0.1，L2 正则化 (w=0.01) |
| **词表与位置** | BPE 词汇表 (40,000 merges)，学习式 Position Embeddings |
| **激活函数** | GELU |

---

**结果**

- 在 12 个评估数据集中的 9 个上取得了 SOTA 表现，超越了许多专门为特定任务设计的模型架构。

- 核心性能提升指标：

| 任务类别 | 数据集 | 性能提升 |
| :--- | :--- | :--- |
| **常识推理** | Story Cloze Test | 绝对提升 **8.9%** |
| **问答** | RACE | 绝对提升 **5.7%** |
| **文本蕴含** | MultiNLI | 绝对提升 **1.5%** |
| **多任务基准** | GLUE | 整体得分达到 **72.8** (提升 **5.5%**) |
| **分类** | CoLA | 得分 **45.4** (大幅超越之前的 35.0) |

![](2c69fed3c6ecd2c00e8a0cf6d4ca50400f8cce340bc2f13127851049bc9ba718.jpg)
![](25b6444944c565e5b1eb049c975849f8e9b2426aeefee490f47a7c9290b2da1d.jpg)

- 消融实验分析：
  - 移除预训练：平均得分下降 **14.8%**，证明预训练对跨任务迁移至关重要。
  - 替换架构：使用单层 2048 单元 LSTM 替代 Transformer，平均得分下降 **5.6%**，证明 Transformer 在捕获长程依赖上具有显著优势。
  - 移除辅助 LM 目标：对大规模数据集（如 NLI）有益，对小数据集效果不明显。

- 零样本行为：
  - 预训练过程中，模型逐步获得有用的语言知识，零样本性能随训练稳定提升。
  - 相比 LSTM，Transformer 架构的归纳偏置在迁移学习中表现出更低的方差。

---

**结论**

- 通过生成式预训练与判别式微调，单一任务无关模型能够实现强大的自然语言理解能力。
- 在包含长距离依赖的多样化连续文本语料上预训练，使模型获得了显著的世界知识与处理长程依赖的能力。
- Transformer 架构与长连续文本数据集是实现此方法显著性能提升的关键因素。
- 该研究验证了无监督学习在提升判别式任务性能方面的巨大潜力，为后续自然语言处理及其他领域的无监督学习研究奠定了基础。

---

## 2. 背景知识与核心贡献

**研究背景**

- 自然语言理解涵盖文本蕴含、问答、语义相似度评估和文档分类等多样化任务。
- 大规模无标签文本语料库极其丰富，但针对特定任务的有标签数据十分稀缺。
- 传统判别式训练模型在缺乏大量标注数据时表现受限。
- 早期半监督方法主要利用无标签数据计算词级别统计信息或训练 **Word Embeddings**，难以捕获高于词级别的语义信息。

---

**研究动机**

- 利用无标签文本学习高级语义表示并迁移至目标任务面临两大核心挑战：
  - 不确定性优化目标：语言建模、机器翻译、语篇连贯等不同优化目标在不同任务上表现不一，难以确定最有效的文本表示学习方案。
  - 缺乏统一迁移机制：现有技术需要对模型架构进行任务特定修改、使用复杂学习方案或添加辅助学习目标，难以开发通用的半监督学习框架。
- 亟需一种能够学习通用表示、仅需极少架构改动即可高效迁移至广泛任务的模型与方法。

---

**核心贡献**

- 提出结合 **无监督预训练** 和 **有监督微调** 的两阶段半监督学习框架。
- 采用 **Transformer** 架构作为基础模型，替代传统的循环网络（如 LSTM），提供更具结构化的记忆以处理文本中的长距离依赖。
- 设计任务感知的输入转换机制，将结构化文本输入转换为单一连续的 **Token** 序列，在微调期间实现有效迁移且最小化模型架构改动。

![](2c30b602030ba602a6891860329f4ebe9dbe5eabfbe2e49fa9f7fc3469333c1d.jpg) *Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer.*

- 构建了一个与任务无关的通用模型，在 12 个研究任务中的 9 个上显著超越了专门为各任务定制的判别式模型，达到 **SOTA**。
- 关键性能提升数据如下：

| 任务类型 | 数据集 | 绝对提升幅度 |
| --- | --- | --- |
| 常识推理 | Stories Cloze Test | **8.9%** |
| 问答 | RACE | **5.7%** |
| 文本蕴含 | MultiNLI | **1.5%** |
| 多任务基准 | GLUE | **5.5%** |

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**核心框架**

本文提出一种基于**无监督预训练**与**有监督微调**的两阶段半监督学习框架，旨在通过单一任务无关模型处理多种自然语言理解任务。

![](2c30b602030ba602a6891860329f4ebe9dbe5eabfbe2e49fa9f7fc3469333c1d.jpg) *Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer.*

---

**基础模型架构**

- 采用**多层 Transformer Decoder** 架构，利用 Masked Self-Attention 机制处理长距离依赖。
- 相比传统 LSTM，Transformer 提供更结构化的记忆，实现跨任务的稳健迁移。

**模型规格参数**

| 组件 | 参数配置 |
| --- | --- |
| 层数 | 12层 Transformer Decoder |
| 隐藏层维度 | 768维 |
| Attention Heads | 12个 |
| 前馈网络维度 | 3072维 |
| 词表 | 40,000 merges 的 BPE |
| 激活函数 | GELU |
| 位置编码 | Learned Position Embeddings |

---

**无监督预训练阶段**

- **训练数据**：BooksCorpus 数据集，包含超过7,000本未出版书籍，保留长连续文本结构以学习长程依赖。
- **优化目标**：标准语言模型目标，最大化给定上下文 Token 的条件概率。
  - 公式：$L_1(\mathcal{U}) = \sum_i \log P(u_i | u_{i-k}, ..., u_{i-1}; \Theta)$
- **前向传播流程**：
  - 输入 Token 向量与 Position Embedding 相加。
  - 经过多层 Transformer Block 提取特征。
  - 通过 Softmax 输出目标 Token 分布。

---

**有监督微调阶段**

- **优化目标**：在预训练参数基础上，使用带标签数据微调，最大化目标标签的条件概率。
  - 公式：$L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y|x^1, ..., x^m)$
- **辅助语言模型目标**：微调时加入预训练语言模型目标作为辅助任务，权重为 $\lambda$。
  - 作用：提升泛化能力，加速收敛。
  - 公式：$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda * L_1(\mathcal{C})$
- **任务特定输入变换**：采用 Traversal-style 方法，将结构化输入转换为连续 Token 序列，仅需添加线性输出层，避免大幅修改模型架构。
  - **Textual Entailment**：拼接前提和假设，中间插入 Delimiter Token ($)。
  - **Similarity**：将两种句子顺序分别输入模型，将输出表示逐元素相加后输入线性层。
  - **Question Answering & Commonsense Reasoning**：拼接文档、问题及候选答案，通过 Softmax 输出答案概率分布。

### 1. 无监督预训练

**核心原理与目标**
- 无监督预训练阶段的核心目标是学习神经网络的初始参数，通过标准语言建模目标最大化以下似然函数：
  $L_1(\mathcal{U}) = \sum_i \log P(u_i | u_{i-k}, ..., u_{i-1}; \Theta)$
- 其中 $k$ 是上下文窗口大小，条件概率 $P$ 由参数为 $\Theta$ 的神经网络建模，使用随机梯度下降进行优化。

**模型架构与计算流程**
- 采用多层 Transformer decoder 架构，通过多头自注意力机制处理长文本依赖。
- 计算流程分为三个阶段：
  - 输入表示：将上下文 Token 向量 $U = (u_{-k}, \dots, u_{-1})$ 与 Token Embedding 矩阵 $W_e$ 以及 Position Embedding 矩阵 $W_p$ 相加，得到初始隐藏状态 $h_0 = U W_e + W_p$。
  - 特征提取：经过 $n$ 层 Transformer block 处理，公式为 $h_l = \text{transformer\_block}(h_{l-1})$，提取深层上下文语义特征。
  - 输出预测：通过 Softmax 层将最终隐藏状态映射为目标 Token 的概率分布 $P(u) = \text{softmax}(h_n W_e^T)$。

![](2c30b602030ba602a6891860329f4ebe9dbe5eabfbe2e49fa9f7fc3469333c1d.jpg) *Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer.*

**预训练数据集特性**
- 使用 BooksCorpus 数据集，包含超过 7000 本不同类型的未出版书籍。
- 关键优势在于包含长连续文本片段，使生成模型能够学习基于长程信息的条件分布。
- 相比于在句子级别打乱的 1B Word Benchmark，该数据集保留了长程结构，为捕获长距离依赖提供了基础。

**核心参数与训练设置**
- 模型配置为 12 层 decoder-only Transformer，包含 768 维状态和 12 个 Attention heads。
- 具体训练参数如下表所示：

| 参数类别 | 具体配置 |
| --- | --- |
| 优化器 | Adam，最大学习率 **2.5e-4** |
| 学习率调度 | 前 2000 次更新线性增加，随后余弦退火至 0 |
| 训练规模 | **100 epochs**，Batch size **64**，序列长度 **512 tokens** |
| 权重初始化 | 正态分布 **N(0, 0.02)** |
| 词表 | Bytepair encoding (BPE)，**40,000 merges** |
| 正则化 | Dropout rate **0.1** (residual, embedding, attention)，L2 regularization (**w=0.01**) |
| 激活函数 | Gaussian Error Linear Unit (**GELU**) |
| 位置编码 | Learned position embeddings |
| 文本处理 | ftfy 库清洗，spaCy tokenizer |

**在整体框架中的作用**
- 作为两阶段训练流程的第一阶段，为后续的监督微调提供强大的初始参数。
- 通过处理长连续文本，模型获取了丰富的世界知识和处理长程依赖的能力。
- 消融实验表明，移除预训练会导致所有任务性能显著下降（平均下降 **14.8%**），证明无监督预训练是实现有效迁移学习和提升下游任务泛化能力的关键基础。

### 2. Transformer解码器架构

**架构概述与核心原理**

本文提出的模型采用**多层Transformer解码器**作为核心架构，这是一种基于自注意力机制的生成式语言模型。与传统的循环神经网络（如LSTM）相比，Transformer架构通过**多头自注意力机制**处理长距离依赖，提供了更具结构化的记忆，从而在跨任务迁移中表现出更强的鲁棒性。

![](2c30b602030ba602a6891860329f4ebe9dbe5eabfbe2e49fa9f7fc3469333c1d.jpg) *Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer.*

**算法流程与数学表达**

模型的计算流程分为输入表示、多层特征提取和输出生成三个核心阶段，具体数学表达如下：

- **输入表示阶段**：
  - 模型接收上下文Token序列 $U = (u_{-k}, \dots, u_{-1})$。
  - 通过**Token Embedding矩阵** $W_e$ 和**位置Embedding矩阵** $W_p$ 将输入序列映射为初始隐藏状态向量：$h_0 = U W_e + W_p$。
  - 采用**Learned position embeddings**（学习到的位置编码）替代原始Transformer中的正弦位置编码。

- **多层特征提取阶段**：
  - 初始状态 $h_0$ 依次通过 $n$ 层Transformer块进行处理：$h_l = \text{transformer\_block}(h_{l-1}) \forall i \in [1, n]$。
  - 每一层内部执行**Masked多头自注意力操作**，允许当前Token关注历史上下文，同时屏蔽未来信息以符合自回归语言模型的生成逻辑。
  - 注意力操作后接入**位置前馈层**，对特征进行非线性变换。

- **输出生成阶段**：
  - 最终层的激活值 $h_n$ 与转置的Token Embedding矩阵相乘，并通过**Softmax**函数归一化，生成目标Token的概率分布：$P(u) = \text{softmax}(h_n W_e^T)$。

**模型参数与配置规格**

在无监督预训练阶段，模型的具体超参数和优化配置如下表所示：

| 配置类别 | 参数名称 | 设定值 |
| :--- | :--- | :--- |
| **网络结构** | 层数 | 12层 Decoder-only Transformer |
| | 隐藏层维度 | 768 |
| | 注意力头数 | 12 |
| | 前馈层内部状态维度 | 3072 |
| **训练设置** | 优化器 | Adam |
| | 最大学习率 | 2.5e-4 |
| | 学习率调度 | 前2000步线性预热，随后余弦退火至0 |
| | Batch Size | 64 |
| | 序列长度 | 512 Tokens |
| | 训练轮数 | 100 Epochs |
| **正则化与初始化**| 权重初始化 | N(0, 0.02) |
| | Dropout | 0.1 (残差、Embedding、Attention) |
| | L2正则化权重 | 0.01 (非偏置及增益权重) |
| **其他组件** | 激活函数 | **GELU** (Gaussian Error Linear Unit) |
| | 词汇表 | **BPE** (Byte-Pair Encoding), 40,000 merges |
| | 归一化层 | **LayerNorm** |

**输入输出关系与整体作用**

- **输入**：长度为 $k$ 的上下文Token序列。在预训练阶段为无标签文本片段；在微调阶段根据具体任务被组装为包含特殊分隔符（如 `$`, `<start>`, `<end>`）的连续Token序列。
- **输出**：
  - 预训练阶段：输出下一个Token的概率分布。
  - 微调阶段：提取最终Transformer块的激活状态 $h_l^m$，送入额外的线性输出层 $W_y$，输出目标任务的标签概率 $P(y|x^1, \dots, x^m)$。
- **整体作用**：
  - **特征提取器**：通过大规模无监督预训练，学习文本的深层语法、语义及世界知识，构建通用的语言表征。
  - **迁移学习基座**：在微调阶段，预训练参数提供强大的初始化点，模型通过**Task-specific input transformations**适配多种NLP任务（如文本蕴含、问答、相似度计算），仅需引入极少量额外参数（如线性分类层 $W_y$）即可实现SOTA性能。

### 3. 监督微调

**核心机制与算法流程**

- 基于无监督预训练（等式1）获得的模型参数，进入监督微调阶段以适配特定判别任务。
- 输入处理：将带标签数据集 $\mathcal{C}$ 中的输入 token 序列 $x^1, \ldots, x^m$ 输入预训练模型。
- 特征提取：获取最终 Transformer block 的激活值 $h_l^m$。
- 预测输出：将 $h_l^m$ 送入新增的线性输出层（参数为 $W_y$），通过 softmax 函数计算标签 $y$ 的概率分布（等式3）。
- 优化目标：最大化对数似然（等式4），此过程仅引入 $W_y$ 和 delimiter token 的 embedding 作为额外参数。
- 辅助目标：引入语言建模作为辅助目标（等式5，权重 $\lambda$），通过联合优化提升泛化能力并加速收敛。

---

**任务特定的输入转换**

- 为处理结构化输入并保持预训练模型架构的完整性，采用 traversal-style 方法将结构化输入转换为连续 token 序列。
- 所有转换均添加随机初始化的 start 和 end tokens。
- Textual Entailment：拼接 premise 和 hypothesis，中间插入 delimiter token (\$)。
- Similarity：由于句子无固定顺序，将两种可能的顺序组合分别输入模型，将得到的表示进行 element-wise 相加后输入线性层。
- Question Answering 和 Commonsense Reasoning：将 context document、question 与每个候选答案拼接，独立处理后通过 softmax 层归一化以产生答案分布。

![](2c30b602030ba602a6891860329f4ebe9dbe5eabfbe2e49fa9f7fc3469333c1d.jpg) *Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer.*

---

**微调参数设置**

- 复用无监督预训练阶段的超参数设置。
- 具体参数配置如下：

| 参数名称 | 参数值 |
| --- | --- |
| Classifier Dropout | 0.1 |
| Learning Rate | 6.25e-5 |
| Batch Size | 32 |
| Epochs | 3 |
| Learning Rate Decay | Linear (0.2% warmup) |
| $\lambda$ (辅助目标权重) | 0.5 |

---

**在整体架构中的作用与影响**

- 核心作用：将无监督阶段学到的通用语言表示迁移至具体的下游任务，实现 task-agnostic 模型向 task-specific 模型的转化。
- 性能提升：在12个评估数据集中，9个达到 SOTA，证明微调策略的有效性。
- 辅助目标分析：辅助 LM 目标对大型数据集（如 NLI 任务、QQP）有益，但对小型数据集无显著提升。
- 消融实验对比：

| 模型变体 | Avg. Score | 性能变化 |
| --- | --- | --- |
| Transformer w/ aux LM (full) | 74.7 | 基准模型 |
| Transformer w/o pre-training | 59.9 | 缺乏预训练导致性能大幅下降 14.8% |
| Transformer w/o aux LM | 75.0 | 移除辅助目标后平均得分略升，但在部分任务上表现波动 |
| LSTM w/ aux LM | 69.1 | 替换为 LSTM 导致平均得分下降 5.6% |

### 4. 任务特定的输入变换

**核心思想与实现原理**

- **预训练模型**基于连续的文本序列进行训练，面对具有结构化输入的下游任务（如句子对、文档-问题-答案三元组）时，必须进行格式适配。
- 采用**遍历式方法**，将结构化输入转换为单一的连续 **Token** 序列，使预训练的 **Transformer** 模型无需修改核心架构即可直接处理。
- 此方法避免了为每个下游任务设计复杂的定制化网络层，实现了最小化的架构改动。
- 所有输入变换均包含随机初始化的起始和结束 **Token**（文中记为 hsi, hei）。

![](2c30b602030ba602a6891860329f4ebe9dbe5eabfbe2e49fa9f7fc3469333c1d.jpg) *Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer.*

---

**各类任务的序列转换流程**

- **文本蕴含**
  - 输入：前提 p 和假设 h。
  - 拼接方式：将 p 和 h 的 **Token** 序列直接拼接，中间插入分隔符 **$**。
  - 输出处理：拼接后的单一序列输入模型，取最后一个 **Transformer Block** 的激活值送入线性输出层。
- **语义相似度**
  - 输入：待比较的两个句子，无固有时间顺序。
  - 拼接方式：为反映无序性，构造两种可能的句子排列顺序，各自加入分隔符。
  - 输出处理：两个序列独立通过模型，获得各自的序列表示 $h_l^m$，随后进行逐元素相加，再送入线性输出层。
- **问答与常识推理**
  - 输入：上下文文档 z、问题 q、候选答案集合 $\{a_k\}$。
  - 拼接方式：将文档、问题与每一个候选答案分别拼接，格式为 [z; q; $; a_k]。
  - 输出处理：每个拼接序列独立输入模型，通过 **Softmax** 层进行归一化，生成候选答案的概率分布。

---

**参数设置与架构作用**

- **新增参数**：在 **Fine-tuning** 阶段，除线性输出层参数 $W_y$ 外，仅需额外学习分隔符和起止符的 **Embedding**。
- **初始化策略**：起止符和分隔符的 **Embedding** 采用随机初始化。
- **整体作用**：通过统一的序列化输入，使得单一的 **Task-agnostic** 模型能够无缝迁移至多种不同结构的 NLP 任务，极大提升了模型的通用性和迁移效率，证明了预训练学到的表征具有强大的泛化能力。

---

**输入输出关系映射**

| 任务类型 | 原始输入结构 | 序列拼接形式 | 输出层处理逻辑 |
| --- | --- | --- | --- |
| **文本分类** | 单一文本 x | [Start; x; End] | 直接取序列末尾激活值进行分类 |
| **文本蕴含** | 前提 p, 假设 h | [Start; p; $; h; End] | 直接取序列末尾激活值进行分类 |
| **语义相似度** | 句子 A, 句子 B | [Start; A; $; B; End] 与 [Start; B; $; A; End] | 两个序列表示逐元素相加后分类 |
| **问答/常识推理** | 文档 z, 问题 q, 答案 $a_k$ | [Start; z; q; $; $a_k$; End] (针对每个 $a_k$) | 各序列独立处理后 **Softmax** 归一化 |

### 5. 辅助语言建模目标

**核心机制与实现原理**

- 在 Supervised fine-tuning 阶段，模型不仅针对特定的 Target task（如 Textual entailment、Question answering）进行判别性学习，还同时保留 Unsupervised pre-training 阶段的 Language modeling 目标。
- 这种 Multi-task learning 策略将 Language modeling 作为 Auxiliary objective，旨在防止模型在微调过程中遗忘预训练阶段习得的通用语言知识，避免 Catastrophic forgetting。
- 核心收益体现为两点：改善监督模型的 Generalization 能力；加速模型收敛。

---

**算法流程与参数设置**

- 监督微调的基础目标是最大化条件概率，对应损失函数 $L_2(\mathcal{C})$，通过 Softmax 层输出标签 $y$ 的概率。
- 辅助目标沿用预训练阶段的 Language modeling 目标，即基于上下文预测当前 Token，对应损失函数 $L_1(\mathcal{C})$。
- 总优化目标 $L_3(\mathcal{C})$ 为两者加权和：$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda * L_1(\mathcal{C})$。
- 参数设置：
  - 辅助目标的权重 $\lambda$ 设定为 **0.5**。
  - 优化器沿用 Adam，学习率设为 **6.25e-5**。
  - Batch size 为 **32**。
  - 采用带 Warmup 的线性学习率衰减策略，Warmup 比例为训练步数的 **0.2%**。
  - 大多数任务仅需 **3 个 epochs** 即可完成微调。

---

**输入输出关系**

- 输入：带有标签的监督数据集 $\mathcal{C}$，包含输入 Token 序列 $x^1, ..., x^m$ 及标签 $y$。
- 中间状态：输入序列经过 Pre-trained Transformer 模型，提取最后一个 Transformer block 的激活值 $h_l^m$。
- 输出：
  - 监督分支：$h_l^m$ 经由新增的线性输出层（参数为 $W_y$）输出预测标签分布。
  - 辅助分支：基于隐藏状态输出 Token 的概率分布。
- 模型通过联合优化这两个输出分支的损失来更新整个网络的参数，其中微调阶段唯一新增的参数仅为 $W_y$ 及 Delimiter tokens 的 Embedding。

---

**在整体架构中的作用与消融分析**

- 作用：作为 Regularization scheme，防止过拟合；维持模型对长文本依赖和通用语义的表征能力。
- 消融实验表明，Auxiliary objective 的效果与数据集规模高度相关：
  - 在大型数据集（如 NLI 任务、QQP）上，辅助目标显著提升性能。
  - 在小型数据集上，辅助目标未带来明显收益，整体平均分甚至略有下降。

| 模型变体 | Avg. Score | MNLI (acc) | QNLI (acc) | QQP (F1) |
| :--- | :--- | :--- | :--- | :--- |
| Transformer w/ aux LM (full) | 74.7 | 81.8 | 88.1 | 70.3 |
| Transformer w/o aux LM | 75.0 | 81.1 | 86.9 | 69.8 |

![](2c30b602030ba602a6891860329f4ebe9dbe5eabfbe2e49fa9f7fc3469333c1d.jpg) *Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer.*


---

## 4. 实验方法与实验结果

**实验设置**

- **预训练数据**：采用 **BooksCorpus** 数据集，包含超过 **7,000** 本未出版书籍。该数据集包含长距离连续文本，使模型能够学习长程依赖信息。
- **模型架构**：使用 **12层 Transformer decoder**，隐藏层维度为 **768**，包含 **12** 个 self-attention heads。前馈网络内部状态维度为 **3072**。采用 **BPE (Byte-pair encoding)** 词汇表（40,000 merges）及可学习的位置嵌入。
- **优化与正则化配置**：
  - 预训练：使用 **Adam** 优化器，最大学习率 **2.5e-4**，采用 cosine schedule。训练 **100** epochs，batch size 为 **64**，序列长度 **512**。Dropout 比率设为 **0.1**，L2 正则化权重 **0.01**。激活函数采用 **GELU**。
  - 微调：大部分复用预训练超参数，学习率调整为 **6.25e-5**，batch size 为 **32**。通常训练 **3** epochs。辅助语言模型目标权重 **λ = 0.5**。

---

**结果数据分析**

模型在 **12** 个任务中的 **9** 个取得了 SOTA (State-of-the-art) 成绩，验证了任务无关架构的强大泛化能力。

- **自然语言推理 (NLI)**
  - 在 **MNLI**、**SNLI**、**SciTail** 和 **QNLI** 上均超越基线，绝对提升分别达 **1.5%**、**0.6%**、**5%** 和 **5.8%**。
  - 在小数据集 **RTE** 上表现略逊于多任务 BiLSTM 模型（**56.0%** vs **61.7%**）。

| Method | MNLI-m | SNLI | SciTail | QNLI | RTE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CAFE (5x) | 80.2 | 89.3 | - | - | - |
| Stochastic Answer Network (3x) | 80.6 | - | - | - | - |
| Multi-task BiLSTM + Attn | 72.2 | - | - | 82.1 | 61.7 |
| **Finetuned Transformer LM (ours)** | **82.1** | **89.9** | **88.3** | **88.1** | 56.0 |

- **问答与常识推理**
  - **Story Cloze** 测试取得 **86.5%** 准确率，绝对提升 **8.9%**。
  - **RACE** 数据集整体提升 **5.7%**，证明模型处理长距离上下文的能力。

| Method | Story Cloze | RACE |
| :--- | :--- | :--- |
| val-LS-skip | 76.5 | - |
| Dynamic Fusion Net (9x) | - | 51.2 |
| BiAttention MRU (9x) | - | 53.3 |
| **Finetuned Transformer LM (ours)** | **86.5** | **59.0** |

- **语义相似度与分类**
  - **CoLA** 任务取得 **45.4** 分，大幅超越之前的 **35.0** 分，展现出预训练模型习得的强语言直觉。
  - **GLUE** 基准总分达 **72.8**，显著优于之前的 **68.9**。

| Method | CoLA (mc) | SST2 (acc) | MRPC (F1) | STSB (pc) | QQP (F1) | GLUE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Single-task BiLSTM + ELMo + Attn | 35.0 | 90.2 | 80.2 | 55.5 | 66.1 | 64.8 |
| Multi-task BiLSTM + ELMo + Attn | 18.9 | 91.6 | 83.5 | 72.8 | 63.3 | 68.9 |
| **Finetuned Transformer LM (ours)** | **45.4** | 91.3 | 82.3 | **82.0** | **70.3** | **72.8** |

---

**消融实验与模型分析**

- **层数迁移影响**
  ![](2c69fed3c6ecd2c00e8a0cf6d4ca50400f8cce340bc2f13127851049bc9ba718.jpg)
  - 迁移更多 Transformer 层能持续提升下游任务性能。在 **MultiNLI** 上，完整迁移带来高达 **9%** 的性能提升，表明预训练模型的每一层都包含对目标任务有用的功能。
- **零样本行为**
  ![](25b6444944c565e5b1eb049c975849f8e9b2426aeefee490f47a7c9290b2da1d.jpg)
  - 随着预训练步数增加，模型在多个任务上的零样本性能稳定上升。
  - 相比 **LSTM**，**Transformer** 架构在零样本性能上表现出更低的方差，其结构化注意力机制有助于迁移。
- **消融研究**
  - **移除辅助 LM 目标**：对 NLI 和 QQP 等较大数据集有益，但小数据集无明显改善。整体平均分变化极小（**75.0** vs **74.7**）。
  - **替换为 LSTM 架构**：平均分下降 **5.6** 分，仅在 **MRPC** 上优于 Transformer，证明 Transformer 在捕获长距离依赖上的绝对优势。
  - **移除无监督预训练**：所有任务性能均大幅下降，平均分下降 **14.8%**，证明预训练是性能提升的核心来源。

| Method | Avg. Score | CoLA (mc) | SST2 (acc) | MRPC (F1) | STSB (pc) | QQP (F1) | MNLI (acc) | QNLI (acc) | RTE (acc) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformer w/ aux LM (full) | 74.7 | 45.4 | 91.3 | 82.3 | 82.0 | 70.3 | 81.8 | 88.1 | 56.0 |
| Transformer w/o pre-training | 59.9 | 18.9 | 84.0 | 79.4 | 30.9 | 65.5 | 75.7 | 71.2 | 53.8 |
| Transformer w/o aux LM | 75.0 | 47.9 | 92.0 | 84.9 | 83.2 | 69.8 | 81.1 | 86.9 | 54.4 |
| LSTM w/ aux LM | 69.1 | 30.3 | 90.5 | 83.2 | 71.8 | 68.1 | 73.7 | 81.1 | 54.6 |

---

