# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, et al.

**发表期刊/会议 (Journal/Conference)**: ArXiv

**发表年份 (Publication Year)**: 2019

**研究机构 (Affiliations)**: Stanford University, NVIDIA

---

## 1. 摘要

**目的**

- 训练超大型 Transformer 语言模型（数十亿参数级别），突破现代处理器单卡内存限制。
- 提出一种简单高效的模型并行方法，避免依赖自定义编译器或重写框架（如 Mesh-TensorFlow）。
- 验证模型规模扩大对下游自然语言处理任务准确率的提升作用。

---

**方法**

- **Intra-layer model parallelism（层内模型并行）**
  - 利用 Transformer 网络结构，在原生 PyTorch 中插入少量通信原语实现。
  - **MLP Block 切分**：将第一个 GEMM 按列切分，第二个 GEMM 按行切分，消除 GeLU 非线性层的同步点。前向传播仅需一次 all-reduce（g 算子），反向传播仅需一次 all-reduce（f 算子）。
  - **Self-Attention Block 切分**：按列切分 Q、K、V 矩阵，使每个 Attention Head 在本地 GPU 独立计算。输出线性层按行切分，直接接收并行 Attention 输出，无需额外通信。
  - **Embedding 层切分**：按词汇维度切分。输入 Embedding 后使用 all-reduce；输出 Embedding 与 Cross-entropy loss 融合，将通信量从 $b \times s \times v$ 降至 $b \times s$。
  - **计算复制策略**：Layer Normalization、Dropout 和 Residual connection 在各 GPU 上复制计算，保持计算密集型，避免额外通信。

![](images/5986c0716d74fdfed6adee2df2a6ec8c2d716cdf0a2f2a709603ee7517b26c10.jpg)

- **BERT 架构调整**
  - 重新排列 Layer Normalization 和 Residual connection 的顺序。
  - 解决原 BERT 架构在模型规模超越 BERT-Large 时出现的训练不稳定与性能退化问题。

---

**结果**

- **扩展性能**
  - 在 512 块 NVIDIA V100 GPU 上训练 8.3B 参数模型，实现 15.1 PetaFLOPs 算力。
  - 相较单 GPU 强基线（39 TeraFLOPs），8-way model parallelism 达到 77% 线性扩展效率。
  - 结合 64-way data parallelism，整体扩展效率达 74%。

![](images/7b50765568bf8dc48c09ab8674df6c763434877aa1f76fe303552ae8d9846776.jpg)

- **下游任务 SOTA 表现**
  - GPT-2 (8.3B) 与 BERT (3.9B) 模型在多个数据集上刷新 SOTA。
  - 验证了模型规模增大可单调提升下游任务准确率。

| 模型 | 数据集 | 核心指标 | Megatron-LM 结果 | 前期 SOTA |
| --- | --- | --- | --- | --- |
| **GPT-2 (8.3B)** | WikiText103 | Perplexity ↓ | **10.8** | 15.8 |
| **GPT-2 (8.3B)** | LAMBADA | Accuracy ↑ | **66.5%** | 63.2% |
| **BERT (3.9B)** | RACE | Accuracy ↑ | **90.9%** | 89.4% |

---

**结论**

- 仅需少量 PyTorch 修改即可实现高效模型并行，突破单 GPU 内存瓶颈。
- Layer Normalization 的位置调整对扩展 BERT 类模型规模至关重要。
- 模型规模扩大显著提升下游任务准确率，确立新的 SOTA 基线。
- 代码已开源，支持未来大规模预训练研究。

---

## 2. 背景知识与核心贡献

**研究背景与动机**

- NLP 领域的快速进展得益于计算能力和数据集规模的大幅增长，通过无监督预训练的大型 Transformer 模型在下游任务中持续取得 SOTA 成绩。
- 随着模型规模不断扩大，其参数量超出了现代处理器的内存限制。
- 广泛使用的优化算法（如 ADAM）需要为每个参数分配额外内存以存储动量等状态，进一步压缩了可有效训练的模型规模上限。
- 现有的模型并行方案（如 GPipe 和 Mesh-TensorFlow）存在局限性：
  - 需要重写模型代码。
  - 依赖仍在开发中的自定义编译器和框架。
- 迫切需要一种简单、高效且无需编译器或库更改的模型并行方法，以突破单 GPU 内存瓶颈，训练数十亿参数级别的语言模型。

---

**核心贡献**

- **简单高效的模型并行实现**
  - 利用 Transformer 网络的固有结构，实现了 intra-layer model parallel 方法。
  - 无需新的编译器或 C++ 代码，仅在原生 PyTorch 中插入少量通信原语（如 all-reduce）即可完成实现。
  - 该方法与基于流水线的模型并行正交且互补。

- **卓越的扩展性能与计算吞吐量**
  - 在 512 个 GPU 上成功训练了高达 8.3 billion 参数的 Transformer 模型。
  - 实现了 15.1 PetaFLOPs 的持续计算性能。
  - 相比于单 GPU 基线（39 TeraFLOPs），扩展效率达到 **76%**。

![](images/db87f162f03d23275e0aab6ee32c69909e432b9d6af0f4dc541052b88ceb6886.jpg)

- **BERT 架构优化的关键发现**
  - 发现 Layer Normalization 与残差连接的放置位置对 BERT 类模型规模扩大至关重要。
  - 通过调整 Transformer 层中的 Layer Normalization 和残差连接顺序，消除了大模型训练的不稳定性，实现了随模型规模增加而单调提升的准确率。

- **刷新多项下游任务 SOTA 记录**
  - 证明了扩大模型规模能显著提升准确率。
  - GPT-2 (8.3B) 和 BERT (3.9B) 模型在多个数据集上取得突破：

| 模型 | 数据集 | 评估指标 | Megatron-LM 结果 | 先前 SOTA |
| --- | --- | --- | --- | --- |
| GPT-2 (8.3B) | WikiText103 | Perplexity ↓ | **10.8** | 15.8 |
| GPT-2 (8.3B) | LAMBADA | Accuracy ↑ | **66.5%** | 63.2% |
| BERT (3.9B) | RACE | Accuracy ↑ | **90.9%** | 89.4% |

- **开源贡献**
  - 公开了模型代码、训练及评估流程，支持后续研究。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**核心架构概述**

Megatron-LM 提出了一种简单的 **Intra-layer Model Parallelism**（层内模型并行）方法，通过在原生 PyTorch 中插入少量通信原语，实现十亿参数级 Transformer 模型的高效训练。该架构无需自定义编译器或 C++ 代码，且与 Pipeline Model Parallelism 正交互补。

---

**Transformer 层内并行设计**

针对 Transformer 的核心组件进行切分，通过融合 GEMM 组消除中间同步点，保持 GPU 计算受限。

- **MLP 块并行**：
  - 第一个 GEMM 采用 **Column Parallel**（列并行），权重矩阵切分为 `[A1, A2]`，使 GeLU 非线性函数可独立应用于各分区输出，消除同步点。
  - 第二个 GEMM 采用 **Row Parallel**（行并行），直接接收 GeLU 层输出，无需通信。
  - 输出在传入 Dropout 层前进行一次跨 GPU 规约。
- **Self-Attention 块并行**：
  - 利用多头注意力机制的固有结构，将 Key (K)、Query (Q)、Value (V) 的 GEMM 进行列并行切分，确保每个 Attention head 的矩阵乘法在单个 GPU 本地完成。
  - 输出线性层采用行并行，直接接收并行注意力层的输出。

![](images/5986c0716d74fdfed6adee2df2a6ec8c2d716cdf0a2f2a709603ee7517b26c10.jpg)
![](images/b559b5df2340bcc3a19c9a3dae7bca78d0006f3b98d41d1683783240be7babbc.jpg) *(a) MLP (b) Self-Attention Figure 3. Blocks of Transformer with Model Parallelism. f and g are conjugate. $f$ is an identity operator in the forward pass and all reduce in the backward pass while g is an all reduce in the forward pass and identity in the backward pass.*

---

**通信原语与优化策略**

整个 Transformer 层仅需 4 次通信操作，通过极简的 `f` 与 `g` 算子实现前向与反向传播的协同。

| 算子 | 前向传播 | 反向传播 |
| --- | --- | --- |
| **f** | Identity | All-reduce |
| **g** | All-reduce | Identity |

- **计算冗余替代通信**：Dropout、Layer Normalization 和残差连接的计算在所有 GPU 上复制执行，避免广播中间结果，维持计算受限。
- **Embedding 层与交叉熵融合**：
  - 输入 Embedding 权重沿词汇表维度切分，前向传播需一次 all-reduce。
  - 输出 Embedding 与 cross-entropy loss 融合计算，将通信量从 `b × s × v`（logits）大幅降低至 `b × s`（标量损失）。

![](images/7925c10de1b603c0e6d8c9d2d2065c9b8e487bbdb2bcad2342f541110dcf7a0e.jpg) *Figure 4. Communication operations in a transformer layer. There are 4 total communication operations in the forward and backward pass of a single model parallel transformer layer.*

---

**混合并行与架构改进**

- **Hybrid Model & Data Parallelism**：模型并行组内的 GPU 分布模型，组间相同位置的 GPU 构成数据并行组进行梯度 all-reduce。总 GPU 需求为模型并行度与数据并行度的乘积。
- **BERT 架构调整**：
  - 将 Layer Normalization 和残差连接移至 Self-Attention 和 MLP 块的输入端（类似 Pre-LN 结构）。
  - 消除原始 BERT 架构在大规模下的训练不稳定性，实现训练损失单调下降。

### 1. Intra-layer Model Parallelism for Transformers

**核心原理与设计哲学**

- Megatron-LM 提出了一种高效的 **Intra-layer Model Parallelism** (层内模型并行) 方法，专门针对 Transformer 架构设计。
- 该方法无需引入新的编译器或重写底层库，仅通过在原生 PyTorch 中插入极少数的 **通信原语** 即可实现数十亿参数模型的训练。
- 核心设计哲学在于 **减少通信开销** 并保持 GPU 处于 **计算受限** 状态，通过巧妙划分矩阵乘法避免不必要的同步点。

---

**MLP Block 的并行化策略**

- Transformer 中的 MLP Block 包含两个 GEMM 操作，中间夹有 GeLU 非线性激活函数。
- 第一个 GEMM 采用 **列切分** 策略：将权重矩阵 A 沿列切分为 `[A1, A2]`。这使得 GeLU 非线性函数可以独立应用于每个分区的输出，即 `[Y1, Y2] = [GeLU(XA1), GeLU(XA2)]`。
- 该切分方式的关键优势在于 **消除了 GeLU 之前的同步点**，因为 `GeLU(X1A1 + X2A2) != GeLU(X1A1) + GeLU(X2A2)`，若按行切分则必须在 GeLU 前进行通信。
- 第二个 GEMM 采用 **行切分** 策略：直接接收 GeLU 层的输出作为输入，无需任何通信即可完成计算。
- 第二个 GEMM 的输出在进入 Dropout 层之前，通过 **all-reduce** 操作进行跨 GPU 聚合。
- 整个 MLP Block 在前向传播中仅需一次 all-reduce，反向传播中仅需一次 all-reduce。

![](images/5986c0716d74fdfed6adee2df2a6ec8c2d716cdf0a2f2a709603ee7517b26c10.jpg)

---

**Self-Attention Block 的并行化策略**

- 利用 Multi-head Attention 的固有结构，将 Key (K)、Query (Q)、Value (V) 的 GEMM 进行 **列切分**。
- 每个 Attention Head 的矩阵乘法完全在单个 GPU 上本地执行，无需立即通信即可完成 Self-Attention 计算。
- 输出线性层的 GEMM 采用 **行切分**，直接接收并行 Attention 层的输出，无需通信。
- 输出结果在进入后续操作前进行 **all-reduce** 聚合。
- 该策略将成对的 GEMM 融合，消除了中间的同步点，提升了扩展性。

![](images/b559b5df2340bcc3a19c9a3dae7bca78d0006f3b98d41d1683783240be7babbc.jpg) *(a) MLP (b) Self-Attention Figure 3. Blocks of Transformer with Model Parallelism. f and g are conjugate. $f$ is an identity operator in the forward pass and all reduce in the backward pass while g is an all reduce in the forward pass and identity in the backward pass.*

---

**通信原语与算子设计**

- 实现层内并行的核心在于两个共轭的通信算子：**f 算子** 和 **g 算子**。
- 这些算子可以用极少的 PyTorch 代码实现，例如 `f` 算子在 forward pass 中执行 identity 操作，在 backward pass 中执行 all-reduce。
- 算子行为定义如下表：

| 算子 | Forward Pass 行为 | Backward Pass 行为 |
|---|---|---|
| **f** | Identity (直接返回输入) | All-reduce (梯度聚合) |
| **g** | All-reduce (结果聚合) | Identity (直接传递梯度) |

- 在单个 Transformer Layer 的前向和反向传播中，总共仅需 **4 次通信操作**。

![](images/7925c10de1b603c0e6d8c9d2d2065c9b8e487bbdb2bcad2342f541110dcf7a0e.jpg) *Figure 4. Communication operations in a transformer layer. There are 4 total communication operations in the forward and backward pass of a single model parallel transformer layer.*

---

**Embedding 层与 Cross-entropy Loss 融合**

- Transformer 语言模型的输出 Embedding 维度为 Hidden-size (H) 乘以 Vocabulary-size (v)。由于 v 通常达到数万，必须对输出 Embedding 进行并行化。
- 输入 Embedding 权重矩阵沿 Vocabulary 维度进行 **列切分** (`E = [E1, E2]`)，每个分区只包含部分 Embedding 表，随后使用 **g 算子** 进行 all-reduce。
- 输出 Embedding 与输入 Embedding 共享权重。若直接对并行 GEMM 的输出 Logits 进行 all-gather，将产生 `b x s x v` (batch-size x sequence-length x vocabulary-size) 的巨大通信量。
- 为解决此瓶颈，将并行 GEMM 的输出与 **Cross-entropy Loss** 进行融合，将通信维度从 Logits 降至标量 Loss，通信量锐减至 `b x s`，极大提升了并行效率。

---

**计算冗余与参数优化**

- 为避免频繁的广播操作，对于 Dropout、Layer Normalization 和 Residual connection 等操作，采用 **跨 GPU 复制计算** 的策略。
- 在每个 GPU 上维护 Layer Normalization 参数的副本，直接在本地执行 Dropout 和 Residual connection。
- 每个 Model parallel worker 独立优化自己持有的参数子集。由于所有值要么是本地的，要么是复制的，无需在参数更新阶段进行通信。

---

**整体作用与输入输出关系**

- **输入**：来自上一层的激活张量 (如形状为 `[b, s, h]` 的 Hidden states)。
- **处理流程**：输入张量在进入 Model parallel 区域时，通过 `f` 算子保持不变；在 GEMM 计算中按列或行切分；在离开 Model parallel 区域时，通过 `g` 算子进行 all-reduce 聚合。
- **输出**：与单 GPU 计算在数学上完全等价的激活张量，但计算和内存压力被分散到多个 GPU 上。
- **整体作用**：打破了单 GPU 显存墙限制，使得训练包含数十亿参数的 GPT-2 和 BERT 模型成为可能。该技术与 Pipeline Model Parallelism 正交且互补，并能与 Data Parallelism 无缝结合，在 512 块 GPU 上实现了 15.1 PetaFLOPs 的持续算力。

### 2. Fused Cross-Entropy for Embedding Parallelism

**核心痛点与动机**

在 Transformer 语言模型中，输出 Embedding 层的维度由隐藏层维度（**Hidden Size**, $H$）与词汇表大小（**Vocabulary Size**, $v$）的乘积决定。由于现代语言模型的 $v$ 通常达到数万级别（例如 GPT-2 的 $v=50,257$），该层的矩阵乘法（**GEMM**）计算与内存开销巨大，必须进行并行化处理。然而，常规的并行化策略会引发极其庞大的通信开销，成为系统瓶颈。

---

**实现原理与算法流程**

Megatron-LM 采用列切分策略对输入与输出 Embedding 的权重矩阵 $E_{H \times v}$ 进行划分，即 $E = [E_1, E_2]$。在处理输出 Embedding 时，为消除通信瓶颈，设计了 **Fused Cross-Entropy** 机制。

- **常规并行策略及其缺陷**
  - 各 GPU 持有列切分的权重矩阵 $E_i$。
  - 前向传播时，各 GPU 并行计算部分 Logits：$[Y_1, Y_2] = [X E_1, X E_2]$。
  - 为获取完整的预测分布，需通过 **all-gather** 操作聚合所有 Logits：$Y = \text{all-gather}([Y_1, Y_2])$。
  - 随后将完整的 $Y$ 送入 **Cross-Entropy** 函数计算损失。
  - **缺陷**：此过程需通信 $b \times s \times v$ 个元素（$b$ 为 **batch-size**，$s$ 为 **sequence length**）。由于 $v$ 极大，通信量呈指数级膨胀，严重拖慢训练效率。

- **Fused Cross-Entropy 算法流程**
  - **取消聚合**：不再执行 **all-gather** 操作以收集完整 Logits $Y$。
  - **本地计算**：每个 GPU 利用本地计算出的部分 Logits $Y_i$，直接与真实标签计算部分 **Cross-Entropy** 损失。
  - **标量通信**：通过 **all-reduce** 操作跨 GPU 汇总各部分损失标量，得到最终的全局 Loss。
  - **反向传播**：基于本地 Logits 与全局 Loss 直接在本地计算梯度，无需重组完整 Logits 张量。

---

**参数与维度变化对比**

通过融合技术，通信张量的维度实现了从三维到二维的降维打击，极大降低了通信负载。

| 策略类型 | 通信操作 | 通信数据维度 | 通信量级评估 (以 $b=512, s=1024, v=50257$ 为例) |
| :--- | :--- | :--- | :--- |
| **常规策略** | **all-gather** | $b \times s \times v$ | 约 26 亿个元素 |
| **Fused 策略** | **all-reduce** | $b \times s$ | 约 52 万个元素 |

---

**输入输出关系**

- **输入**：
  - 上一层的隐藏状态张量 $X$（维度：$b \times s \times H$）。
  - 本地分片的权重矩阵 $E_i$（维度：$H \times v_i$，其中 $v_i = v / \text{Model Parallel Size}$）。
  - 真实标签索引（维度：$b \times s$）。
- **输出**：
  - 全局标量损失值（维度：$1$）。
  - 本地权重梯度 $\nabla E_i$（维度：$H \times v_i$）。

---

**在整体架构中的作用**

**Fused Cross-Entropy** 是 Megatron-LM 实现**计算与通信重叠**以及**保持 GPU 计算密集型**的关键设计之一。

- **消除通信瓶颈**：将输出层的通信量从 $O(b \cdot s \cdot v)$ 骤降至 $O(b \cdot s)$，使得模型并行不会因词汇表过大而失效。
- **维持计算密集度**：避免了大规模 Logits 张量在 GPU 间的物理拷贝，确保 GPU 资源用于 **GEMM** 计算而非数据搬运。
- **无缝协同**：该技术与 MLP 及 Self-Attention 块中的 $f$ 和 $g$ 通信算子无缝配合，共同构成了仅需少量 PyTorch 原语即可完成的**Intra-layer Model Parallelism** 基础设施。

![](images/7925c10de1b603c0e6d8c9d2d2065c9b8e487bbdb2bcad2342f541110dcf7a0e.jpg) *Figure 4. Communication operations in a transformer layer. There are 4 total communication operations in the forward and backward pass of a single model parallel transformer layer.*

### 3. Rearranged Layer Normalization for BERT Scaling

**核心观点**
- 重新排列 **Layer Normalization** 和 **Residual Connection** 的顺序是突破 **BERT-Large** 规模限制的关键技术。
- 原始 **BERT** 架构在模型规模超过 **336M** 参数时会出现性能退化与训练不稳定。
- 将 **Layer Normalization** 移至 **Multi-Head Attention** 和 **Feed Forward** 层的输入端（即 **Pre-LN** 结构），消除了训练不稳定性，并显著降低了训练损失。

**架构对比与实现原理**
- 原始 **BERT** 架构：
  - 采用 **Post-LN** 结构。
  - **Layer Normalization** 应用于 **Multi-Head Attention** 和 **Feed Forward** 层的输出之后。
  - **Residual Connection** 先于 **Layer Normalization** 执行。
  - 在深层网络中，反向传播的梯度在经过多个 **Layer Normalization** 层后容易消失或爆炸，导致大模型训练崩溃。
- 重新排列后的架构：
  - 采用 **Pre-LN** 结构。
  - **Layer Normalization** 应用于 **Multi-Head Attention** 和 **Feed Forward** 层的输入之前。
  - **Residual Connection** 直接跨越子层，与子层输出相加后再进入下一层。
  - 梯度可以通过 **Residual Connection** 直接回传，不受 **Layer Normalization** 的阻隔，极大提升了深层网络训练的稳定性。

![](images/297aca6cd81f73e0657fcf206b71d4801322a761ec682fc981ad60976d183aee.jpg)
![](images/2256c4b8c6b266b0d5b4a7db8ef9898f7b56227423634f1c21f028fdb2fc7b1b.jpg)

**算法流程与输入输出关系**
- 输入：上一层的输出张量 $X$。
- **Pre-LN** 流程：
  - 对输入 $X$ 进行 **Layer Normalization** 得到 $X'$。
  - 将 $X'$ 输入 **Multi-Head Attention** 层，计算注意力输出。
  - 将注意力输出与原始输入 $X$ 进行 **Residual Connection** 相加。
  - 对相加结果再次进行 **Layer Normalization**。
  - 将归一化后的结果输入 **Feed Forward** 层（包含 **GeLU** 非线性激活）。
  - 将 **Feed Forward** 输出与之前的相加结果进行 **Residual Connection**。
  - 输出：传递给下一层的张量。
- 作用：在整体模型中，该结构保证了信号在深层 **Transformer** 网络中的顺畅传播，使得模型参数可以扩展至 **3.9B** 甚至更大，而不会出现梯度消失或发散。

**参数设置与实验结果**
- 训练参数：
  - **Hidden size per attention head** 固定为 **64**。
  - 使用 **Adam** 优化器，**Weight decay** 设为 **0.01**。
  - **Dropout** 设为 **0.1**。
  - 采用 **Activation Checkpointing** 以减少内存占用。
  - **336M** 和 **1.3B** 模型训练 **2 million** iterations，**3.9B** 模型训练 **1.5 million** iterations。
- 性能对比：
  - 验证集 **Perplexity** 随模型规模增加而单调下降：**336M** 为 **1.58**，**1.3B** 为 **1.30**，**3.9B** 为 **1.16**。
  - 在下游任务（如 **RACE** 数据集）中，**3.9B** 模型取得了 **89.4%** 的准确率，超越了当时的 **SOTA**。

| 模型规模 | 验证集 Perplexity | RACE 准确率 |
| --- | --- | --- |
| 336M | 1.58 | 83.0% |
| 1.3B | 1.30 | 87.3% |
| 3.9B | 1.16 | 89.4% |

**在整体中的作用**
- 解决了 **BERT-style** 模型在扩展至 **BERT-Large** 以上规模时的性能退化问题。
- 提供了一种简单但极其有效的架构修改方案，无需引入复杂的参数共享机制即可实现模型扩展。
- 配合 **Megatron-LM** 的模型并行策略，使得在有限硬件资源下训练数十亿参数级别的双向 **Transformer** 模型成为可能。


---

## 4. 实验方法与实验结果

**实验设置**

- 硬件环境：使用最多 32 台 **DGX-2H** 服务器，总计 512 块 **Tesla V100 SXM3 32GB GPU**。服务器内部通过 **NVSwitch** 提供 300 GB/sec 带宽，服务器间通过 8 个 **InfiniBand** 适配器提供 100 GB/sec 带宽。
- 数据集构建：聚合 Wikipedia, CC-Stories, RealNews, OpenWebtext 数据集。使用 **Locality-Sensitive Hashing (LSH)** 去重，过滤 Jaccard Similarity 大于 0.7 的重复内容。移除短文档（小于 128 Token），最终获得 174GB 去重文本。GPT-2 训练排除 BooksCorpus 以避免与 LAMBADA 任务重叠。
- 优化器与超参数：
  - 采用 **Mixed Precision Training** 与 dynamic loss scaling。
  - 权重初始化 $W \sim \mathcal{N}(0, 0.02)$，残差层前缩放因子为 $1/\sqrt{2N}$。
  - 使用 **Adam** 优化器，**Weight Decay** 为 0.01，**Gradient Norm Clipping** 为 1.0，**Dropout** 为 0.1。
  - 引入 **Activation Checkpointing** 降低内存占用。
- GPT-2 训练配置：序列长度 1024，Batch Size 512，训练 300k iterations。Learning Rate 1.5e-4，包含 3k iterations warmup，随后进行 cosine decay，最低降至 1e-5。
- BERT 训练配置：Vocab Size 30,522。替换 Next Sentence Prediction 为 Sentence Order Prediction，采用 Whole Word N-gram Masking。Batch Size 1024，Learning Rate 1.0e-4，warmup 10k iterations 后线性衰减 2M iterations。
- 扩展性测试模型配置：保持 Hidden Size per Attention Head 为 96，通过调整层数和头数构建不同规模模型。Vocab Size 填充至 51,200 以确保 GEMM 效率。

| Hidden Size | Attention heads | Number of layers | Number of parameters (billions) | Model parallel GPUs | Model +data parallel GPUs |
|---|---|---|---|---|---|
| 1536 | 16 | 40 | 1.2 | 1 | 64 |
| 1920 | 20 | 54 | 2.5 | 2 | 128 |
| 2304 | 24 | 64 | 4.2 | 4 | 256 |
| 3072 | 32 | 72 | 8.3 | 8 | 512 |

---

**结果数据**

- 扩展性表现：
  - 单 GPU 基线（1.2B 参数）维持 39 TeraFLOPs，为理论峰值 30%。
  - 8-way Model Parallelism（8.3B 参数）实现 77% Linear Scaling。
  - 结合 Model + Data Parallelism（512 GPUs，8.3B 参数），达到 15.1 PetaFLOPs，Scaling Efficiency 为 74%。
  ![](images/db87f162f03d23275e0aab6ee32c69909e432b9d6af0f4dc541052b88ceb6886.jpg)
  ![](images/7b50765568bf8dc48c09ab8674df6c763434877aa1f76fe303552ae8d9846776.jpg)
- GPT-2 语言建模结果：
  - 模型规模扩大显著降低 Validation Perplexity，8.3B 模型达到 9.27。
  ![](images/b01f9400b7f333c09e1a5c2cf178a4ee8a85c3af29ead5fda2d6e9c38a032090.jpg)
  - Zero-shot 评估中，8.3B 模型在 WikiText103 上 Perplexity 为 10.81（SOTA 为 15.79），在 LAMBADA 上 Accuracy 为 66.51%（SOTA 为 63.24%）。

| Model | Wikitext103 Perplexity ↓ | LAMBADA Accuracy ↑ |
|---|---|---|
| 355M | 19.31 | 45.18% |
| 2.5B | 12.76 | 61.73% |
| 8.3B | 10.81 | 66.51% |
| Previous SOTA | 15.79 | 63.24% |

- BERT 下游任务结果：
  - 3.9B 模型在 RACE Test Set 达到 89.4% Accuracy，取得 SOTA。
  - 在 MNLI, QQP, SQuAD 1.1, SQuAD 2.0 Development Set 上均超越或匹敌 RoBERTa, ALBERT, XLNet。

| Model | MNLI m/mm accuracy | QQP accuracy | SQuAD 1.1 F1 / EM | SQuAD 2.0 F1 / EM | RACE m/h accuracy |
|---|---|---|---|---|---|
| RoBERTa | 90.2 / 90.2 | 92.2 | 94.6 / 88.9 | 89.4 / 86.5 | 83.2 |
| ALBERT | 90.8 | 92.2 | 94.8 / 89.3 | 90.2 / 87.4 | 86.5 |
| XLNet | 90.8 / 90.8 | 92.3 | 95.1 / 89.7 | 90.6 / 87.9 | 85.4 |
| Megatron-336M | 89.7 / 90.0 | 92.3 | 94.2 / 88.0 | 88.1 / 84.8 | 83.0 |
| Megatron-1.3B | 90.9 / 91.0 | 92.6 | 94.9 / 89.1 | 90.2 / 87.1 | 87.3 |
| Megatron-3.9B | 91.4 / 91.4 | 92.7 | 95.5 / 90.0 | 91.2 / 88.5 | 89.4 |

---

**消融实验**

- BERT 架构调整消融：
  - 原始 BERT 架构在模型规模超过 BERT-Large 时出现训练不稳定和性能退化。
  - 将 Layer Normalization 和 Residual Connection 重排至图 7(b) 所示位置，消除不稳定性并降低 Training Loss。
  ![](images/297aca6cd81f73e0657fcf206b71d4801322a761ec682fc981ad60976d183aee.jpg)
  ![](images/2256c4b8c6b266b0d5b4a7db8ef9898f7b56227423634f1c21f028fdb2fc7b1b.jpg)
- Attention Heads 数量对 Scaling 的影响：
  - 固定 8.3B 参数与 8-way Model Parallelism，增加 Attention Heads 数量会降低 Scaling Efficiency。
  - 原因：Attention Heads 增多导致 Self-Attention 层内部 GEMMs 尺寸缩小，Softmax 计算元素增加，计算密度下降。

| Attention heads | Hidden size per head | Scaling Efficiency |
|---|---|---|
| 16 | 192 | 82% |
| 24 | 128 | 80% |
| 32 | 96 | 77% |

- Strong Scaling 分析：
  - 固定 1.2B 参数模型与 Batch Size 8，增加 GPU 数量。
  - 2 GPUs 加速 1.64 倍，4 GPUs 加速 2.34 倍，8 GPUs 加速 2.98 倍。
  - 收益递减原因：单 GPU 计算量减少，Memory Bandwidth 与通信开销占比逐渐主导。

|---|---|---|---|---|
| Speedup | 1.0 | 1.64 | 2.34 | 2.98 |

---
