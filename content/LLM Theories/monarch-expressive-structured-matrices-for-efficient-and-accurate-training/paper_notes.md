# Monarch: Expressive Structured Matrices for Efficient and Accurate Training 论文解析

## 0. 论文基本信息

**作者 (Authors)**: unknown

**发表期刊/会议 (Journal/Conference)**: ArXiv

**发表年份 (Publication Year)**: 2021

**研究机构 (Affiliations)**: Stanford University

---

## 1. 摘要

**目的**

- 解决大型神经网络训练与微调过程中计算和内存成本高昂的问题。
- 克服现有结构化矩阵在端到端（E2E）训练中硬件效率与模型质量难以兼顾的劣势。
- 填补稠密到稀疏（D2S）微调场景中缺乏易处理算法以近似预训练稠密权重矩阵的长期空白。

---

**方法**

- 提出 **Monarch** 矩阵参数化方法：
  - 数学形式为 $\mathbf{M}= \mathbf{P}\mathbf{L}\mathbf{P}^\top \mathbf{R}$，其中 $\mathbf{L}$ 和 $\mathbf{R}$ 为块对角矩阵，$\mathbf{P}$ 为固定置换矩阵。
  - **硬件高效**：利用 GPU 优化的 Batch Matrix Multiply (BMM) 例程，相比稠密矩阵乘法实现最高 2x 加速。
  - **表达能力强**：包含 Butterfly 矩阵类，能够表示多种常用快速变换（如 Fourier、卷积、Hadamard 等）。
- 推导解析最优投影算法：
  - 解决将给定稠密矩阵投影至 Monarch 矩阵集合的非凸优化问题。
  - 通过将矩阵重塑为 4D Tensor 并对每个批次进行 SVD 秩 1 近似，时间复杂度为 $O(n^{5/2})$。
- 提出 $\mathcal{M}\mathcal{M}^*$ 矩阵分解算法：
  - 在假设矩阵可逆且对角块无零元素的条件下，利用同时对角化技术恢复 Monarch 因子。
- 定义三种应用范式：
  - **E2E 稀疏训练**：直接用 Monarch 矩阵替换 Attention 和 FFN 块中的稠密权重矩阵进行训练。
  - **S2D 训练（反向稀疏化）**：前 90% 迭代使用 Monarch 矩阵训练，后 10% 转换为稠密矩阵继续训练。
  - **D2S 微调**：利用投影算法将预训练稠密模型转换为 Monarch 模型，在下游任务进行微调。

![](monarch-main.png) *Monarch matrices unlock several ways to train sparse and dense models: end-to-end training a sparse (Monarch) model can be 2x faster than dense training thanks to its hardware efficiency; sparse-to-dense “reverse sparsification” can speed up training of large models such as GPT-2; and our dense-to-sparse Monarch projection algorithm can transfer knowledge from pretrained dense model to Monarch model and speed up BERT fine-tuning.*

---

**结果**

- E2E 训练性能提升：
  - ViT 和 GPT-2 训练速度提升 **2x**，保持同等 Accuracy / Perplexity。
  - 在 PDE 求解和 MRI 重建任务中，相比基于 Fourier 的领域专用方法，误差最多降低 **40%**。
- S2D 训练加速：
  - GPT-2 在 OpenWebText 上预训练速度提升 **2x**，无质量下降。
  - BERT 预训练速度比 Nvidia MLPerf 1.1 记录快 **23%**。
- D2S 微调优化：
  - BERT 在 GLUE 上微调速度提升 **1.7x**，参数减少 **2x**，且准确率与稠密模型相当。

| 应用场景 | 模型/任务 | 核心性能指标对比 |
| :--- | :--- | :--- |
| E2E 训练 | ViT / GPT-2 | 训练速度提升 **2x**，质量相当 |
| E2E 训练 | PDE / MRI | 误差最多降低 **40%** |
| S2D 训练 | GPT-2 (OpenWebText) | 预训练速度提升 **2x** |
| S2D 训练 | BERT | 比 Nvidia MLPerf 1.1 快 **23%** |
| D2S 微调 | BERT (GLUE) | 微调速度提升 **1.7x**，参数减少 **2x** |

---

**结论**

- **Monarch** 参数化成功结合了 Butterfly 矩阵的表达能力与 GPU 硬件效率，打破了结构化矩阵在深度学习中的落地瓶颈。
- 推导的解析投影算法为预训练稠密模型向结构化稀疏模型转换提供了理论保证与可行路径。
- 该方法解锁了大型神经网络在 E2E 训练、S2D 训练和 D2S 微调中的全新加速范式，在自然语言处理、计算机视觉及科学计算等领域展现出显著的效率与精度优势。

---

## 2. 背景知识与核心贡献

**研究背景与动机**

- 大型神经网络在诸多领域表现卓越，但**训练**与**微调**的计算和内存成本极高。
- 替代稠密权重矩阵的常见方法是使用**结构化矩阵**（如稀疏矩阵、低秩矩阵、Fourier transform）。
- 结构化矩阵未获广泛采用的两大核心痛点：
  - **End-to-End (E2E) training**：效率与质量权衡不佳。现有矩阵要么在 GPU 上**硬件效率低**，要么**表达能力不足**（无法表示卷积或 Fourier transform 等常用变换）。大多数稀疏训练方法在 wall-clock time 上反而**减慢**训练速度。
  - **Dense-to-Sparse (D2S) fine-tuning**：缺乏可处理算法将近似给定稠密权重矩阵投影到结构化矩阵，难以用于微调预训练模型。

---

**核心贡献**

- 提出 **Monarch** 矩阵：一种新型结构化矩阵参数化方法，兼具**硬件高效性**与**强表达能力**。
  - **硬件高效**：参数化为两个**块对角矩阵**的乘积（带置换），充分利用 GPU 优化的 **batch-matrix-multiply (BMM)** 例程。
  - **表达能力强**：包含 **butterfly matrices** 类，能表示多种快速变换（如 Fourier、sine/cosine、Chebyshev transforms、convolution）。
- 突破投影算法瓶颈：证明将近似稠密矩阵投影到 Monarch 矩阵的非凸问题存在**解析最优解**，时间复杂度为 $O(n^{5/2})$。
- 解锁三种全新训练与微调范式：
  - **E2E sparse training**：直接用 Monarch 矩阵替换稠密权重进行端到端训练。
  - **Sparse-to-Dense (S2D) training**（"reverse sparsification"）：前期使用 Monarch 矩阵快速训练，后期转换为稠密矩阵，加速大型模型预训练。
  - **D2S fine-tuning**：利用投影算法将预训练稠密模型转换为 Monarch 模型，加速下游任务微调。

![](monarch-main.png) *Monarch matrices unlock several ways to train sparse and dense models: end-to-end training a sparse (Monarch) model can be 2x faster than dense training thanks to its hardware efficiency; sparse-to-dense “reverse sparsification” can speed up training of large models such as GPT-2; and our dense-to-sparse Monarch projection algorithm can transfer knowledge from pretrained dense model to Monarch model and speed up BERT fine-tuning.*

---

**实验性能指标对比**

| 应用场景 | 模型/任务 | 核心指标 | 加速比/提升 |
| :--- | :--- | :--- | :--- |
| **E2E Training** | ViT / MLP-Mixer (ImageNet) | Top-1 Accuracy | 训练速度提升 **2x**，参数与 FLOPs 大幅降低 |
| **E2E Training** | GPT-2 (Wikitext-103) | Perplexity (PPL) | 训练速度提升 **2x**，PPL 相当 |
| **E2E Training** | PDE solving / MRI reconstruction | 重建误差 / pSNR | 误差降低 **40%** / pSNR 提升 1.5dB |
| **S2D Training** | GPT-2 (OpenWebText) | PPL / 下游分类准确率 | 预训练加速 **2x**，质量无损 |
| **S2D Training** | BERT (Wikipedia + BookCorpus) | Training time | 比 Nvidia MLPerf 1.1 记录快 **23%** |
| **D2S Fine-tuning** | BERT (GLUE) | GLUE Average | 参数减少 **2x**，微调速度提升 **1.7x**，精度相当 |

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**核心架构概述**

Monarch 是一种新型结构化矩阵参数化方法，旨在解决深度学习中 dense 权重矩阵计算与内存成本过高的问题。其核心架构通过将矩阵分解为两个块对角矩阵的乘积，兼顾了**硬件效率**与**表达能力**，并支持解析最优的投影算法。

![](monarch-main.png) *Monarch matrices unlock several ways to train sparse and dense models: end-to-end training a sparse (Monarch) model can be 2x faster than dense training thanks to its hardware efficiency; sparse-to-dense “reverse sparsification” can speed up training of large models such as GPT-2; and our dense-to-sparse Monarch projection algorithm can transfer knowledge from pretrained dense model to Monarch model and speed up BERT fine-tuning.*

---

**矩阵参数化与计算机制**

*   **数学定义**：对于 $n \times n$ 的方阵（设 $n=m^2$），Monarch 矩阵定义为 $M = P L P^T R$。
*   **组件说明**：
    *   $L$ 和 $R$：均为块对角矩阵，各包含 $m$ 个大小为 $m \times m$ 的 dense 块。
    *   $P$：固定的置换矩阵，将向量重塑为 $m \times m$ 矩阵并进行转置操作。
*   **计算流程**：输入向量 $x$ 首先被 $R$ 乘（batched matrix multiply），经过 $P$ 置换，接着被 $L$ 乘，最后再次置换。
*   **硬件效率**：该结构完美契合 GPU 的 **batch matrix multiply (BMM)** 底层优化，相比 dense 矩阵乘法实现最高 **2x** 加速。参数量为 $O(n\sqrt{n})$，FLOPs 为 $O(n\sqrt{n})$。

![](monarch-1.png) *Monarch matrices are parametrized as products of two block-diagonal matrices up to permutation, allowing efficient multiplication algorithm that leverages batch matrix multiply.*

---

**表达能力与表示范围**

*   **包含 Butterfly 矩阵**：Monarch 严格包含 butterfly 矩阵类（$\mathcal{M} \supset \mathcal{B}$），继承了其表示低深度算术电路的能力。
*   **支持复杂变换**：通过 Monarch 矩阵的乘积（如 $\mathcal{M}\mathcal{M}^*$ 或 $(\mathcal{M}\mathcal{M}^*)^2$），可表示多种常用快速变换：
    *   $\mathcal{M}\mathcal{M}^*$：卷积、Hadamard 变换、Toeplitz 矩阵。
    *   $(\mathcal{M}\mathcal{M}^*)^2$：Fourier 变换、离散正弦/余弦变换 (DST/DCT)。

---

**核心算法**

*   **投影算法**：
    *   目标：寻找最接近给定 dense 矩阵 $A$ 的 Monarch 矩阵。
    *   机制：将 $A$ 重塑为 4D tensor，问题被分解为 $m \times m$ 个独立的秩一近似问题。
    *   求解：对每个切片使用 **SVD** 求解，时间复杂度为 $O(n^{5/2})$，具有解析最优解。
*   **因式分解算法**：
    *   目标：恢复 $\mathcal{M}\mathcal{M}^*$ 类矩阵的 Monarch 因子。
    *   机制：在矩阵可逆且块无零元素的假设下，通过 **simultaneous diagonalization** 技术求解。

---

**应用训练范式**

*   **End-to-End (E2E) Sparse Training**：直接用 Monarch 矩阵替换 Transformer 中的 Attention 投影矩阵和 FFN 权重，随机初始化后正常训练。
*   **Sparse-to-Dense (S2D) Training ("reverse sparsification")**：
    *   前 90% 迭代使用 Monarch 权重训练以加速。
    *   后 10% 迭代将 Monarch 矩阵乘开转换为 dense 权重继续训练。
    *   适用于需要大量参数记忆的大型语言模型预训练。

![](monarch-2.png) *With the “reverse sparsification” process, Monarch matrices can speed up GPT-2 training by 2x.*

*   **Dense-to-Sparse (D2S) Fine-tuning**：利用投影算法将预训练 dense 模型（如 BERT）的权重转换为 Monarch 矩阵，然后在下游任务微调。

![](monarch-3.png) *With [alg:project] for our Monarch parameterization, we can convert a pretrained model into a model with Monarch weight matrices and speed up downstream fine-tuning.*

---

**性能指标对比**

| 模型 | 任务/数据集 | 精度/PPL | 加速比 | 参数量 |
| :--- | :--- | :--- | :--- | :--- |
| Monarch-ViT-B/16 | ImageNet | 78.9% | 2.0x | 33.0M |
| Monarch-GPT-2-Medium | WikiText-103 | 20.3 | 2.0x | 165M |
| Monarch-GPT-2m (S2D) | OpenWebText | 18.0 | 2.0x | - |
| Monarch-BERT-large (S2D) | Pretraining | - | 1.23x (vs MLPerf) | - |
| Monarch-BERT-large (D2S) | GLUE | 79.6 | 1.7x | 144M |

### 1. Monarch矩阵参数化

**核心定义与数学表达**

**Monarch矩阵**通过将密集权重矩阵参数化为两个块对角矩阵的乘积（带置换操作），实现硬件效率与表达能力的平衡。对于维度为 $n \times n$（其中 $n = m^2$）的方阵，其数学定义为：
- $\mathbf{M} = \mathbf{P}\mathbf{L}\mathbf{P}^\top \mathbf{R}$
- $\mathbf{L}$ 和 $\mathbf{R}$ 均为**块对角矩阵**，各自包含 $m$ 个大小为 $m \times m$ 的密集块。
- $\mathbf{P}$ 是一个**固定的置换矩阵**，其作用是将长度为 $n$ 的向量重塑为 $m \times m$ 的矩阵，转置后再还原为长度为 $n$ 的向量。$\mathbf{P}$ 不引入任何额外参数。

![](monarch-1.png) *Monarch matrices are parametrized as products of two block-diagonal matrices up to permutation, allowing efficient multiplication algorithm that leverages batch matrix multiply.*

---

**硬件高效性实现**

Monarch矩阵的设计直接针对现代GPU的块状内存访问特性优化，避免了传统稀疏矩阵导致的内存访问不规则问题。
- **计算流程**：输入向量 $\mathbf{x}$ 首先右乘块对角矩阵 $\mathbf{R}$，经过置换 $\mathbf{P}$，再右乘块对角矩阵 $\mathbf{L}$，最后逆向置换恢复原始维度。
- **底层优化**：块对角矩阵的乘法直接映射为 GPU 的 **batch matrix multiply (BMM)** 例程，实现极高的硬件利用率。
- **复杂度与实际速度**：
  - 理论 FLOPs 为 $O(n\sqrt{n})$，略高于 butterfly 矩阵的 $O(n \log n)$。
  - 实际运行速度极快，相比密集矩阵乘法可实现最高 **2倍** 的加速。

---

**表达能力分析**

Monarch矩阵在数学上严格包含 **butterfly matrices** 类（即 $\mathcal{M} \supset \mathcal{B}$），继承了其强大的表达能力。
- **基础变换支持**：$\mathcal{M}\mathcal{M}^*$ 类（两个 Monarch 矩阵的乘积）可精确表示卷积、Hadamard变换、Toeplitz矩阵等。
- **高级变换支持**：$(\mathcal{M}\mathcal{M}^*)^2$ 类可表示 **Fourier transform**、离散正余弦变换 (DST/DCT)、Fastfood 等复杂变换。
- **突破局限**：不同于 FFT 等固定变换，Monarch 矩阵是**可学习的**，能够在 PDE 求解和 MRI 重建等任务中替代固定的 Fourier 变换，自动学习最优的频率重加权。

---

**投影算法流程**

将预训练的密集矩阵转换为 Monarch 矩阵（即投影问题）通常是非凸优化问题，但 Monarch 参数化存在**解析最优解**。
- **目标函数**：$\mathop{\mathrm{argmin}}\limits_{\mathbf{M}\in \mathcal{M}} \left\|{\mathbf{A}- \mathbf{M}}\right\|^2_F$
- **算法步骤**：
  1. 将给定的 $n \times n$ 矩阵 $\mathbf{A}$ 重塑为 $m \times m \times m \times m$ 的 4D 张量。
  2. 将 Monarch 矩阵 $\mathbf{M}$ 视为 4D 张量，其结构可分解为 $m \cdot m$ 个独立的秩为 1 的矩阵块。
  3. 对 $\mathbf{A}$ 的每个 $m \times m$ 切片进行 **SVD 分解**，提取其最优的秩为 1 的近似。
  4. 将分解得到的向量重组为 $\mathbf{L}$ 和 $\mathbf{R}$ 的块对角结构。
- **复杂度**：算法时间复杂度为 $O(n^{5/2})$，类似于低秩近似的 Eckart-Young 定理，保证了全局最优性。

---

**参数设置与规格**

通过调整块大小，Monarch矩阵可灵活适配不同的参数预算和矩阵形状（包括矩形矩阵）。
- **参数量**：标准 $n \times n$ Monarch 矩阵包含 $2n\sqrt{n}$ 个参数（$\mathbf{L}$ 和 $\mathbf{R}$ 各 $n\sqrt{n}$）。
- **实践配置**：在 Transformer 模型中，块对角矩阵的块数通常设置为 **2 到 4** 之间。
- **参数压缩比**：实际应用中，Monarch 参数量通常设定为等效密集模型的 **25% 至 50%**。

| 矩阵类型 | 参数量 | 理论 FLOPs | GPU 友好度 |
| :--- | :--- | :--- | :--- |
| Dense | $n^2$ | $O(n^2)$ | 高 |
| Butterfly | $O(n \log n)$ | $O(n \log n)$ | 低 |
| **Monarch** | $2n\sqrt{n}$ | $O(n\sqrt{n})$ | **极高 (BMM)** |

---

**输入输出与系统作用**

- **输入输出关系**：接收维度为 $n$ 的特征向量，经过“块乘-置换-块乘-逆置换”四步运算，输出维度为 $n$ 的变换特征。整个过程等价于密集线性层，但受限于 Monarch 子空间。
- **在整体架构中的作用**：
  - **端到端稀疏训练 (E2E)**：直接替换 Transformer 中的 Attention 投影矩阵和 FFN 权重矩阵，从头训练实现 2倍 加速且无精度损失。
  - **稀疏到密集训练 (S2D / Reverse Sparsification)**：在 GPT-2 或 BERT 预训练的前 70%-90% 阶段使用 Monarch 矩阵快速训练，最后阶段转换为密集矩阵。利用 Monarch 作为快速中间表示，加速大模型收敛。
  - **密集到稀疏微调 (D2S)**：利用解析投影算法，将预训练的 BERT 权重直接投影为 Monarch 矩阵，在 GLUE 等下游任务中实现 1.7倍 微调加速且保持精度。

### 2. Monarch矩阵的解析投影与分解算法

**核心观点**

将预训练的 Dense 权重矩阵近似为 Monarch 矩阵的投影问题本质上是一个非凸优化问题。然而，通过将矩阵重塑为高维张量，该问题可被解耦为多个独立的秩1近似子问题，从而推导出基于奇异值分解（SVD）的解析最优解。这一突破使得对现有预训练模型进行高效压缩与加速成为可能。

---

**投影算法实现原理与流程**

投影算法旨在寻找一个 Monarch 矩阵 $\mathbf{M}$，使其与给定的 Dense 矩阵 $\mathbf{A}$ 之间的 Frobenius 范数误差最小。Monarch 矩阵的参数化形式为 $\mathbf{M} = \mathbf{P}\mathbf{L}\mathbf{P}^\top \mathbf{R}$，其中 $\mathbf{L}$ 和 $\mathbf{R}$ 为块对角矩阵，$\mathbf{P}$ 为固定置换矩阵。

- **张量重塑与降维**：
  - 将大小为 $n \times n$（其中 $n = m^2$）的输入矩阵 $\mathbf{A}$ 重塑为大小为 $m \times m \times m \times m$ 的 4D 张量 $\widetilde{\mathbf{A}}$。
  - 将 Monarch 矩阵 $\mathbf{M}$ 同样视为 4D 张量，其元素满足 $M_{\ell jki} = L_{j\ell k} R_{kji}$。
- **问题解耦**：
  - 优化目标 $\left\|{\mathbf{A}- \mathbf{M}}\right\|^2_F$ 在重塑后可分解为 $m \cdot m$ 个完全独立的子项。
  - 每个子项对应张量在维度 $k$ 和 $j$ 上的一个切片，大小为 $m \times m$。
  - 每个子项的优化目标变为寻找该切片的最佳秩1近似。
- **算法执行流程**：
  - 提取 $\widetilde{\mathbf{A}}$ 的每个切片 $\widetilde{\mathbf{M}}_{jk} = \widetilde{\mathbf{A}}_{:, j, k, :}$。
  - 对每个切片进行 SVD 分解，取最大奇异值对应的左右奇异向量构造秩1矩阵 $\mathbf{u}_{jk} \mathbf{v}_{jk}^\top$。
  - 将向量 $\mathbf{v}_{jk}$ 组装成 3D 张量 $\widetilde{\mathbf{R}}$，将向量 $\mathbf{u}_{jk}$ 组装成 3D 张量 $\widetilde{\mathbf{L}}$。
  - 将 $\widetilde{\mathbf{L}}$ 和 $\widetilde{\mathbf{R}}$ 重组为块对角矩阵 $\mathbf{L}$ 和 $\mathbf{R}$ 输出。
- **复杂度分析**：
  - 算法需执行 $m^2$ 次大小为 $m \times m$ 的 SVD 计算。
  - 总时间复杂度为 $O(m^5) = O(n^{5/2})$。

---

**$\mathcal{M}\mathcal{M}^*$ 矩阵分解算法**

对于更具表达力的矩阵类 $\mathcal{M}\mathcal{M}^*$（可表示为两个 Monarch 矩阵的乘积），直接分解面临挑战。在给定矩阵可逆且其对角块无零元素的假设下，可通过同时对角化技术求解。

- **矩阵结构展开**：
  - 目标矩阵 $\mathbf{M} \in \mathcal{M}\mathcal{M}^*$ 可展开为 $\widetilde{\mathbf{M}} = \mathbf{L}_1 (\mathbf{P}\mathbf{R}\mathbf{P}^\top) \mathbf{L}_2$。
  - 这构成了由对角块 $\mathbf{A}_i$、对角矩阵 $\mathbf{D}_{ij}$ 和对角块 $\mathbf{C}_j$ 组成的矩阵方程组：$\widetilde{\mathbf{M}}_{ij} = \mathbf{A}_i \mathbf{D}_{ij} \mathbf{C}_j$。
- **同时分解流程**：
  - 构造辅助矩阵 $\mathbf{F}(i, j) = \widetilde{\mathbf{M}}_{i1}^{-1} \widetilde{\mathbf{M}}_{ij} \widetilde{\mathbf{M}}_{1j}^{-1} \widetilde{\mathbf{M}}_{11}$。
  - 利用 $\mathbf{F}(i, j)$ 的特性，寻找一个矩阵 $\hat{\mathbf{C}}_1$ 对所有 $\mathbf{F}(i, j)$ 进行同时对角化。
  - 基于求得的 $\hat{\mathbf{C}}_1$，依次计算 $\hat{\mathbf{A}}_i = \widetilde{\mathbf{M}}_{i1} \hat{\mathbf{C}}_1^{-1}$ 和 $\hat{\mathbf{C}}_j = \hat{\mathbf{A}}_1^{-1} \widetilde{\mathbf{M}}_{1j}$。
  - 最后计算对角矩阵 $\hat{\mathbf{D}}_{ij} = \hat{\mathbf{A}}_i^{-1} \widetilde{\mathbf{M}}_{ij}\hat{\mathbf{C}}_j^{-1}$。
- **复杂度分析**：
  - 算法包含矩阵求逆与同时对角化操作。
  - 总时间复杂度为 $O(n^3/b)$，其中 $b$ 为块大小。

---

**输入输出关系与参数设置**

| 算法模块 | 输入 | 核心参数 | 输出 | 功能说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Projection on $\mathcal{M}$** | Dense 矩阵 $\mathbf{A}$ ($n \times n$) | $m = \sqrt{n}$ | 块对角矩阵 $\mathbf{L}, \mathbf{R}$ | 将任意稠密权重压缩为 Monarch 结构 |
| **Factorization of $\mathcal{M}\mathcal{M}^*$** | $\mathcal{M}\mathcal{M}^*$ 矩阵 $\mathbf{M}$ | 块大小 $b$ | 因子矩阵 $\mathbf{L}_1, \mathbf{R}, \mathbf{L}_2$ | 恢复复杂结构矩阵的底层 Monarch 因子 |

---

**在整体架构中的作用**

解析投影与分解算法直接打通了 Dense-to-Sparse (D2S) fine-tuning 的技术路径。

![](monarch-3.png) *With [alg:project] for our Monarch parameterization, we can convert a pretrained model into a model with Monarch weight matrices and speed up downstream fine-tuning.*

- **预训练模型加速**：
  - 现有的预训练模型（如 BERT）包含大量 Dense 权重矩阵。
  - 利用 Projection 算法，可将这些 Dense 权重无损地映射为 Monarch 权重矩阵。
- **Fine-tuning 效率提升**：
  - 转换后的 Monarch 模型在 Fine-tuning 阶段利用 BMM 加速，大幅降低计算与显存开销。
  - 实验表明，在 GLUE 基准上，Monarch-BERT 相比 Dense BERT 实现了 1.7$\times$ 的 Fine-tuning 速度提升，且参数量减少一半，精度保持相当。

### 3. 反向稀疏化训练策略

**核心思想与原理**
- **反向稀疏化** 是一种利用 **结构化矩阵** 加速 **Dense模型** 预训练的技术。
- 传统稀疏训练常面临 **表示困难** 或 **优化困难**，且在大型数据集上需要海量参数来记忆文本模式，此时Dense模型往往是必需的。
- 该策略将 **Monarch矩阵** 作为快速的中间表示，在训练前期利用其 **硬件高效性** 和 **高表达性** 快速提取特征，后期平滑过渡到Dense权重以突破稀疏模型的容量瓶颈。

---

**算法流程与参数设置**
- **阶段一：Monarch矩阵训练**
  - 将模型中的Dense权重矩阵替换为 **Monarch矩阵**。
  - 使用常规的一阶优化方法（如Adam）进行训练。
  - 占据总训练迭代步数的大部分。
- **阶段二：Dense矩阵过渡与微调**
  - 将Monarch矩阵的因子直接相乘并应用置换，转换为Dense矩阵。
  - 保持相同的总训练迭代步数，继续训练剩余的步数。
  - 此阶段模型恢复为全密集结构，进行最终的收敛优化。
- **关键参数设置**
  - **Block size**：通常设置为2到4，参数量约为Dense模型的25%至50%。
  - **训练步数分配**：根据模型架构和数据集规模动态调整。

| 模型架构 | 数据集 | Monarch训练阶段占比 | Dense训练阶段占比 | 优化器 |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-2** | OpenWebText | 90% | 10% | Adam |
| **BERT-large** | Wikipedia + BookCorpus | 70% | 30% | LAMB |

![](monarch-2.png) *With the “reverse sparsification” process, Monarch matrices can speed up GPT-2 training by 2x.*

---

**输入输出关系与整体作用**
- **输入**：随机初始化的 **Monarch权重矩阵** 及其对应的网络结构。
- **输出**：训练完成的 **Dense模型权重**，其性能与从头训练的Dense模型相当，但总训练时间大幅缩短。
- **整体作用**：
  - **加速预训练**：在GPT-2预训练中实现 **2倍** 加速，在BERT预训练中比Nvidia的MLPerf 1.1记录实现 **23%** 的速度提升。
  - **打破效率-质量权衡**：前期利用Monarch的BMM优化实现硬件加速，后期利用Dense矩阵保证模型容量和最终精度。
  - **解决稀疏训练痛点**：为需要大规模参数的语言建模任务提供了一种可行的稀疏到密集的训练范式，避免了纯稀疏训练在后期可能遇到的性能天花板。

![](rv-bar-temp.png) *Time required (in A100 GPU hours) to reach the same perplexity (18.0) for GPT-2-small on OpenWebText. With “reverse sparsification”, Monarch can speed up GPT-2 training by 2×.*


---

## 4. 实验方法与实验结果

---

**实验设置**

**End-to-End (E2E) Sparse Training**
- **模型替换**：将 ViT、MLP-Mixer、GPT-2 等 Transformer 模型中 Attention block 的 projection matrices 与 FFN block 的 linear layers 替换为 **Monarch** 矩阵。
- **参数配置**：Block-diagonal matrices 的 block 数量设为 **4**。由于模型体积变小，相应减少正则化强度（如 stochastic depth、dropout）。
- **硬件与指标**：在 V100 GPU 上测量 wall-clock training time，评估 Top-1 accuracy 或 Perplexity (PPL)。

**PDE Solving 与 MRI Reconstruction**
- **PDE 设置**：遵循 FNO 实验设定，求解 Navier-Stokes equation。测试不同粘度系数 ($v=10^{-3}, 10^{-4}, 10^{-5}$) 下的误差。
- **MRI 设置**：使用 SKM-TEA 数据集，对比 SENSE 与 U-Net。提出 **mSENSE**，用可学习的 **Monarch** 矩阵替换标准 IFFT 操作。评估指标为 **pSNR** 与 **SSIM**。

**Sparse-to-Dense (S2D) Training (Reverse Sparsification)**
- **GPT-2 Pretraining**：在 OpenWebText 数据集上，前 **90%** 的迭代使用 **Monarch** 权重训练，后 **10%** 转换为 dense 权重继续训练。
- **BERT Pretraining**：在 Wikipedia + BookCorpus 数据集上，前 **70%** 使用 **Monarch**，后 **30%** 转换为 dense。采用 LAMB optimizer，结合 DeepSpeed ZeRO-1 与 Nvidia MLPerf 1.1 的全部优化（如 FMHA kernel）。

**Dense-to-Sparse (D2S) Fine-tuning**
- **投影机制**：利用解析最优解算法，将预训练 BERT 的 dense 权重矩阵投影为 **Monarch** 矩阵。
- **微调任务**：在 GLUE 基准上进行微调，测量参数量与推理速度变化。

![](monarch-main.png) *Monarch matrices unlock several ways to train sparse and dense models: end-to-end training a sparse (Monarch) model can be 2x faster than dense training thanks to its hardware efficiency; sparse-to-dense “reverse sparsification” can speed up training of large models such as GPT-2; and our dense-to-sparse Monarch projection algorithm can transfer knowledge from pretrained dense model to Monarch model and speed up BERT fine-tuning.*

---

**结果数据分析**

**E2E Sparse Training 性能对比**
- **Monarch** 在保持模型质量（Accuracy / PPL）的前提下，实现了最高 **2.0x** 的训练加速，且大幅减少参数量与 FLOPs。

| Model | ImageNet acc. / PPL | Speedup | Params | FLOPs |
| :--- | :--- | :--- | :--- | :--- |
| ViT-B/16 | 78.5 | \- | 86.6M | 17.6G |
| **Monarch-ViT-B/16** | **78.9** | **2.0x** | **33.0M** | **5.9G** |
| GPT-2-Medium | 20.9 | \- | 355M | 361G |
| **Monarch-GPT-2-Medium** | **20.3** | **2.0x** | **165M** | **166G** |

**PDE 与 MRI 领域任务表现**
- 在 Navier-Stokes 方程求解中，**Monarch-NO** 在所有粘度系数下均取得最低误差，最高降低 **40%** 误差。
- 在 MRI 重建中，**mSENSE** 在 2x 加速下，pSNR 较 SENSE 提升 **1.5dB**，SSIM 提升 **2.5%**。

| Model | $v = 10^{-3}$ | $v = 10^{-4}$ | $v = 10^{-5}$ |
| :--- | :--- | :--- | :--- |
| FNO | 0.017 | 0.178 | 0.155 |
| **Monarch-NO** | **0.010** | **0.145** | **0.136** |

**S2D Training (Reverse Sparsification) 加速效果**
- **GPT-2** 预训练实现 **2x** 加速，且下游分类任务平均准确率（38.8）与 dense 模型（38.9）持平。
- **BERT** 预训练时间降至 **23.8 小时**，比 Nvidia MLPerf 1.1 记录快 **23%**。

| Implementation | Training time (h) |
| :--- | :--- |
| Nvidia MLPerf 1.1 | 30.2 |
| Nvidia MLPerf 1.1 + DeepSpeed | 29.3 |
| **Monarch (ours)** | **23.8** |

![](rv-bar-temp.png) *Time required (in A100 GPU hours) to reach the same perplexity (18.0) for GPT-2-small on OpenWebText. With “reverse sparsification”, Monarch can speed up GPT-2 training by 2×.*

**D2S Fine-tuning 效率**
- **Monarch-BERT-large** 在 GLUE 上的平均准确率（79.6）接近 dense 版本（80.4），但参数量减少一半，微调速度提升 **1.7x**。

| Model | GLUE (avg) | Speedup | Params |
| :--- | :--- | :--- | :--- |
| BERT-large | 80.4 | \- | 335M |
| **Monarch-BERT-large** | **79.6** | **1.7x** | **144M** |

---

**消融与对比分析**

**Expressive FFT 与 Data Efficiency (MRI)**
- **标准 IFFT 局限性**：零填充的 SENSE 使用标准 IFFT 无法去除混叠伪影。
- **mSENSE 优势**：保留 Fourier transform 结构的同时学习重新加权频率，有效抑制伪影。
- **Few-shot 性能**：在仅使用 1 个训练样本时，**mSENSE** 的 pSNR 达到 **33.8dB**，比 U-Net（29.4dB）高出 **4.4dB**，证明其具有极强的抗过拟合能力。

**Non-periodic Boundary Conditions (PDE)**
- 传统基于 Fourier transform 的谱方法在处理非周期边界条件时效果极差。
- **Monarch** 矩阵不局限于特定变换，能够自适应学习合适的 solution operator，在复杂边界条件下仍保持极高精度。

**Reverse Sparsification 机制分析**
- **训练阶段切换**：前 90% 时间利用 **Monarch** 的硬件效率快速降低 Loss，后 10% 转换为 dense 权重以突破稀疏表达的优化瓶颈。
- **无损过渡**：转换后的 dense 模型能够迅速收敛至与从头 dense 训练相同的 PPL 水平，证明 **Monarch** 可作为高质量的中间表示。

![](monarch-2.png) *With the “reverse sparsification” process, Monarch matrices can speed up GPT-2 training by 2x.*

**D2S Projection 算法验证**
- **投影可行性**：传统结构矩阵缺乏将 dense 权重投影到稀疏结构的解析算法。
- **Monarch 投影**：通过将矩阵重塑为 4D tensor 并利用 SVD 进行 rank-1 approximation，成功实现 BERT 权重的高效转换。
- **微调效果**：投影后的 **Monarch** 模型无需重新预训练，直接微调即可达到接近原模型的 GLUE 准确率，验证了投影算法的信息保留能力。

![](monarch-3.png) *With [alg:project] for our Monarch parameterization, we can convert a pretrained model into a model with Monarch weight matrices and speed up downstream fine-tuning.*

---

