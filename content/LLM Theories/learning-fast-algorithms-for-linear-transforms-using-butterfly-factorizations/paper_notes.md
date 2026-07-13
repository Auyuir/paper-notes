# Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Tri Dao, Albert Gu, Matthew Eichhorn, et al.

**发表期刊/会议 (Journal/Conference)**: NeurIPS

**发表年份 (Publication Year)**: 2019

**研究机构 (Affiliations)**: Stanford University, University at Buffalo

---

## 1. 摘要

**目的**

- 探索自动学习线性变换（如 **DFT**、**DCT**、卷积）快速算法的可行性，减少人工设计算法和针对不同平台定制实现的需求。
- 确定恢复结构化变换所需的最小先验知识。
- 将该自动学习方法作为轻量级替代品集成到机器学习流水线中，以学习高效且可压缩的潜在变换。

---

**方法**

- 基于矩阵可分解为稀疏矩阵乘积的特性，提出一种基于分治法的参数化方法。
- 引入 **Butterfly Factorizations**（蝴蝶因子分解）表示递归结构：
  - 定义 **BP** 参数化：$T_N = B^{(N)}P^{(N)}$。
  - 定义 **BPBP** 参数化：$T_N = B_2^{(N)}P_2^{(N)}B_1^{(N)}P_1^{(N)}$，以捕获更复杂的结构如卷积。
- **Permutation 学习**：
  - 将离散的排列选择转化为可微的连续优化问题。
  - 通过 3 个 logits 学习 categorical distribution，组合 8 种可能的排列矩阵。
- **初始化与优化**：
  - 将每个 butterfly factor 初始化为接近正交，防止梯度爆炸或消失。
  - 使用 **Adam** 优化器和 **Hyperband** 调参，最小化 Frobenius 范数误差。

![](images/butterfly-pattern-line.png) *Butterfly matrix for N = 16. From left to right: single copy of B16, blocks of B8, blocks of B4, blocks of B2.*

---

**结果**

- **算法恢复**：
  - 成功恢复 **DFT**、**DCT**、**DST**、**Hadamard** 和卷积等变换的快速算法。
  - 维度最高至 **N=1024**，误差达到机器精度（**RMSE < 1e-4**）。

![](images/heatmap.png) *image*

- **神经网络压缩**：
  - 在单隐层网络基准测试中，**BPBP** 方法在所有测试数据集上的分类准确率均超过全连接层。
  - 在 **CIFAR-10** 上，准确率超过无约束矩阵 **3.9** 个百分点，参数量减少 **40X**，推理速度提升 **4X**。

| Method | MNIST-bg-rot | MNIST-noise | CIFAR-10 | Compression factor |
|:---|:---|:---|:---|:---|
| Unstructured | 44.08 | 65.15 | 46.03 | 1 |
| BPBP (complex, fixed permutation) | **46.26** | 77.00 | **49.93** | 39.4 |
| LDR-TD | 45.81 | **78.45** | 45.33 | 56.9 |

- **ResNet 集成**：
  - 在 **ResNet18** 架构中添加 **BPBP** 层，准确率提升 **0.43** 个百分点，参数量仅增加 **0.07%**。
- **速度对比**：
  - 推理阶段：**BP** 快速乘法比 **GEMV** 快 1-2 个数量级，与 **FFT** 相差不到 5 倍。
  - 训练阶段：比 **GEMM** 快 **15%**，与 **FFT** 相差不到 **40%**。

![](images/speed-training-plot.png) *Training*
![](images/speed-plot.png) *Inference*

---

**结论**

- 提出的 **Butterfly Factorizations** 方法成功解决了自动学习线性变换快速算法的问题。
- 该方法不仅能精确恢复多种经典快速变换，还能作为端到端机器学习模型的有效组件。
- 在保证性能提升的同时，实现了显著的模型压缩和推理加速。

---

## 2. 背景知识与核心贡献

**研究背景**

- 快速线性变换（如 **DFT**, **DCT**, **Hadamard transform**, **Convolutions**）在机器学习各环节广泛应用，包括数据预处理、特征生成和模型压缩。
- 这些变换虽可表示为稠密矩阵向量乘法，但均依赖手工设计的专用次二次复杂度算法（如 **FFT**）。
- 现有方法高度依赖特定平台的专门实现（如 **FFTW**, **cuFFT**），缺乏跨平台通用性，且每种变换均需独立手工实现。

**研究动机**

- 探究手工设计这些算法及底层实现的必要性。
- 分析这些快速算法所编码的结构先验。
- 探索在给定结构变换下，自动学习其快速算法所需的最小先验知识。
- 受矩阵可分解为稀疏矩阵乘积这一特性的启发，特别是分治方案能带来快速的乘法算法。
- 旨在开发一种可微分、由基础线性代数原语组成的方法，使其能无缝集成至现代 ML 框架（如 **PyTorch**, **TensorFlow**）。

**核心贡献**

- 提出基于 **Butterfly Factorizations** 的分治方法参数化，通过特定块对角矩阵的乘积来表示一大类结构变换。
- 自动学习并恢复多种重要变换的快速算法，在维度 $N$ 高达 1024 时，以机器精度恢复 $O(N \log N)$ 的 **Cooley-Tukey FFT** 算法及其他变换（如 **DCT**, **Hadamard**, **Convolution**）。

![](images/heatmap.png) *image*

- 将该方法作为轻量级模块集成至端到端 ML 管道中，学习高效且可压缩的潜在变换。
- 在单隐层网络压缩基准任务中，在 **CIFAR-10** 数据集上分类准确率超越无约束矩阵 3.9 个点，同时实现 4 倍推理加速和 40 倍参数压缩。这是首次有结构化方法在此任务上超越无约束模型。

| **Method** | **CIFAR-10** | *Compression factor* |
|:---|:---|:---|
| Unstructured | 46.03 | *1* |
| **BPBP** (complex, fixed permutation) | **49.93** | *39.4* |
| **BPBP** (real, fixed permutation) | 48.69 | *56.9* |
| LDR-TD | 45.33 | *56.9* |

- 实现简单且高效，通用表示在速度上与特定变换的专门实现（如 **DFT**, **DCT**）相差仅 3-5 倍，同时具备学习更丰富通用变换类的能力。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体技术架构**

本文提出了一种基于 **Butterfly Factorization** 的参数化方法，旨在自动学习线性变换的快速算法。其核心思想是将具有快速矩阵-向量乘法的变换分解为稀疏矩阵的乘积，并通过分治策略构建可微的学习模块。

---

**核心组件**

- **Butterfly Matrix ($B$)**：由一系列具有固定稀疏模式的块对角矩阵（Butterfly Factors）相乘构成。每个因子是 $2 \times 2$ 的对角矩阵块。包含 $O(N)$ 个参数，提供 $O(N \log N)$ 的快速乘法复杂度。
- **Permutation Matrix ($P$)**：表示分治过程中的“分”步骤。将离散的排列选择转化为可微的概率分布。

![](images/butterfly-pattern-line.png) *Butterfly matrix for N = 16. From left to right: single copy of B16, blocks of B8, blocks of B4, blocks of B2.*

---

**参数化层级**

- **BP Parametrization**：$T_N = B^{(N)}P^{(N)}$。单层 Butterfly 和 Permutation 乘积。能够精确捕获 DFT 和 Hadamard Transform。
- **BPBP Parametrization**：$T_N = B_2^{(N)}P_2^{(N)}B_1^{(N)}P_1^{(N)}$。两层 BP 乘积。能够精确捕获 DCT、DST 和 Convolution。
- **BP Hierarchy**：定义了 $(BP)^k_r$ 类，通过堆叠 $k$ 个 BP 模块，理论上可以表示从高度结构化的线性参数矩阵到所有 $N \times N$ 方阵的完整谱系。

---

**Permutation 学习机制**

在递归的每一步，Permutation 包含 3 个二元选择（共 8 种组合）：
- 分离奇偶索引 ($P^a$)
- 反转前半部分 ($P^b$)
- 反转后半部分 ($P^c$)

![](images/permutation-matrices.png) *Three binary choices for constructing the permutation used at every step of the recursive process. One of 8 possible permutations can be constructed by multiplying a subset of these matrices in the presented order.*

- **可微化处理**：通过 3 个 logits 参数化，使用 sigmoid 函数计算概率 $p_s$。
- **公式表示**：$P_{N/2^k} = \prod_{s=c,b,a} (p_s P_{N/2^k}^s + (1-p_s)I)$。无需复杂的熵正则化即可收敛到接近 one-hot 的离散状态。

---

**优化与训练策略**

- **目标函数**：最小化目标变换矩阵 $T_N$ 与参数化矩阵之间的 Frobenius norm。
- **优化器**：Adam，结合 Hyperband 进行超参数调优（学习率、初始化、随机种子）。
- **初始化策略**：每个 Butterfly Factor 初始化为接近正交（如实数情况下使用 $\mathcal{N}(0, 1/2)$），以保持输入输出的幅度，防止梯度爆炸或消失。

---

**神经网络压缩性能对比**

在单隐藏层网络压缩任务中，BPBP 结构在大幅减少参数量的同时，分类准确率超过了无约束的全连接层：

| Method | MNIST-bg-rot | MNIST-noise | CIFAR-10 | Compression factor |
| :--- | :--- | :--- | :--- | :--- |
| Unstructured | 44.08 | 65.15 | 46.03 | 1 |
| **BPBP (complex)** | **46.26** | 77.00 | **49.93** | 39.4 |
| **BPBP (real)** | 46.16 | 75.00 | 48.69 | 56.9 |
| LDR-TD | 45.81 | **78.45** | 45.33 | 56.9 |
| Toeplitz-like | 42.67 | 75.75 | 41.78 | 56.9 |

---

**推理与训练速度**

- **训练速度**：在 GPU 上（$N=1024$, batch size 256），BP 的训练时间比密集矩阵乘法（GEMM）快 **15%**，且在 cuFFT 的 **40%** 范围内。
- **推理速度**：在 CPU 上，BP 快速乘法比密集矩阵-向量乘法（GEMV）快一到两个数量级，性能在 FFT 的 5 倍以内，DCT/DST 的 3 倍以内。

![](images/speed-training-plot.png) *Training*
![](images/speed-plot.png) *Inference*

### 1. Butterfly Factorization 参数化

**核心原理与动机**

- **理论基石**：快速矩阵向量乘法可等价为矩阵分解为一系列稀疏矩阵的乘积。传统手工设计的快速算法（如 FFT）均隐含此结构。
- **分治思想**：将维度为 $N$ 的线性变换 $\mathcal{T}_N$ 递归地分解为两个规模为 $N/2$ 的子变换，通过特定的 permutation 分离输入，再通过线性组合合并结果。
- **Butterfly Factorization**：提出一种基于特殊块对角矩阵序列的参数化方法，利用递归结构捕捉广泛的变换类，同时保持极低的参数量和计算复杂度。

---

**算法流程与数学表达**

- **递归分解公式**：对于输入向量 $x \in \mathbb{F}^N$，线性变换 $T_N$ 被分解为：
  - $T_N = \begin{bmatrix} D_1 & D_2 \\ D_3 & D_4 \end{bmatrix} \begin{bmatrix} T_{N/2} & 0 \\ 0 & T_{N/2} \end{bmatrix} P_{N}$
  - 其中 $P_N$ 为 permutation matrix，$D_1, \dots, D_4 \in \mathbb{F}^{N/2}$ 为对角矩阵。
- **Butterfly Factor 定义**：上述公式中的 $\begin{bmatrix} D_1 & D_2 \\ D_3 & D_4 \end{bmatrix}$ 被定义为 butterfly factor $B_N$。
- **展开递归**：将递归完全展开后，变换矩阵表示为 butterfly matrix $B^{(N)}$ 与 permutation matrix $P^{(N)}$ 的乘积：
  - $T_N = B^{(N)} P^{(N)}$ （称为 **BP 参数化**）
  - $T_N = B_2^{(N)} P_2^{(N)} B_1^{(N)} P_1^{(N)}$ （称为 **BPBP 参数化**，可捕获卷积等更复杂结构）
- **结构可视化**：Butterfly factors 呈现高度结构化的稀疏块对角模式，随着递归深度增加，块的大小减半，数量加倍。

![](images/butterfly-pattern-line.png) *Butterfly matrix for N = 16. From left to right: single copy of B16, blocks of B8, blocks of B4, blocks of B2.*

---

**参数化设置与学习机制**

- **Butterfly Matrix 参数**：
  - $B^{(N)}$ 由 $\log_2 N$ 个 butterfly factors 组成。
  - 每个 factor 的非零元素可直接通过梯度下降优化。
  - 总参数量严格限制为 **$4N$**。
- **Permutation 学习机制**：
  - 离散的 permutation 无法直接微分，论文将每一步的 permutation 参数化为 8 种可能选择的凸组合。
  - 每步包含 3 个独立的二元选择：分离奇偶索引 ($P^a$)、反转前半部分 ($P^b$)、反转后半部分 ($P^c$)。
  - 参数化公式：$P_{N/2^k} = \prod_{s=c,b,a} (p_s P_{N/2^k}^s + (1-p_s)I)$，其中 $p_s = \sigma(\ell_s)$ 通过 sigmoid 函数学习。
  - 总参数量仅为 **$3\log_2 N$**（若权重共享可降至 3）。

![](images/permutation-matrices.png) *Three binary choices for constructing the permutation used at every step of the recursive process. One of 8 possible permutations can be constructed by multiplying a subset of these matrices in the presented order.*

- **初始化策略**：
  - 为避免梯度消失或爆炸，每个 butterfly factor 初始化为接近正交。
  - 实数场景下，$B_k$ 的元素从 $\mathcal{N}(0, 1/2)$ 采样，保证 $\mathbb{E} B_k^* B_k = I_N$。

---

**输入输出关系与复杂度**

- **输入**：维度为 $N$ 的向量 $x$（实数或复数）。
- **输出**：经过特定线性变换后的维度为 $N$ 的向量 $X = T_N x$。
- **计算与存储复杂度**：

| 指标 | 传统 Dense Matrix | Butterfly Factorization |
| :--- | :--- | :--- |
| **参数量** | $O(N^2)$ | **$O(N)$** |
| **乘法复杂度** | $O(N^2)$ | **$O(N \log N)$** |
| **表达能力强** | 任意矩阵 | 覆盖 DFT, DCT, Hadamard, Convolution 等 |

---

**在整体架构中的作用**

- **自动算法发现**：作为通用替代件，从输入输出对中自动学习并恢复 DFT、DCT、Hadamard 等经典变换的 $O(N \log N)$ 快速算法，精度可达机器 epsilon。
- **神经网络压缩**：
  - 作为 drop-in replacement 替换全连接层。
  - 在单隐层网络压缩任务中，以超过 **56X** 的参数压缩率，实现高于无约束矩阵的分类准确率（如 CIFAR-10 提升 3.9 points）。
- **性能加速**：
  - 提供仅 5 行 Python 代码即可实现的快速乘法内核。
  - 在 GPU 训练中比 dense matrix multiply (GEMM) 快 15%；在 CPU 推理中比 GEMV 快 1-2 个数量级，且速度与高度优化的 FFT 内核相差不到 5 倍。

### 2. 可微的递归排列学习

**实现原理**

- **离散问题连续化**：传统的快速变换算法（如 FFT）依赖于特定的 index 排列（如 bit-reversal permutation）。直接搜索这些离散排列的空间复杂度为 $O(8^{\log_2 N})$，无法使用梯度下降优化。该方法将离散的排列选择松弛为**分类分布**的连续凸组合，使其变为可微操作。
- **递归分解**：将整体排列 $P^{(N)}$ 分解为 $\log_2 N$ 个递归步骤。在每一步 $k$ 中，排列 $P_{N/2^k}$ 决定如何将输入向量划分为两半。
- **独立二元选择**：每步递归包含3个独立的二元选择，共产生8种可能的排列组合：
  - **$P^a$**：分离奇偶索引（例如 $[0, 1, 2, 3] \to [0, 2, 1, 3]$）。
  - **$P^b$**：反转前半部分（例如 $[0, 1] \to [1, 0]$）。
  - **$P^c$**：反转后半部分（例如 $[2, 3] \to [3, 2]$）。

![](images/permutation-matrices.png) *Three binary choices for constructing the permutation used at every step of the recursive process. One of 8 possible permutations can be constructed by multiplying a subset of these matrices in the presented order.*

---

**算法流程与参数设置**

- **概率参数化**：假设3个二元选择相互独立，每步的排列矩阵表示为独立概率的乘积：
  - $P_{N/2^k} = \prod_{s=c,b,a} (p_s P_{N/2^k}^s + (1-p_s)I)$
- **参数设置**：每个二元选择由一个 logit $\ell_s$ 控制，通过 sigmoid 函数 $\sigma$ 映射为概率 $p_s = \sigma(\ell_s)$。
- **参数量控制**：

| 参数共享策略 | 可学习参数量 | 适用场景 |
| :--- | :--- | :--- |
| 无共享 | $3\log_2 N$ | 每层递归具有不同的划分逻辑 |
| 权重共享 | **3** | 算法具有自相似性（如标准 FFT） |

- **优化与收敛**：
  - 虽然可引入 entropy regularization 促使分布尖锐化，但实验表明这并非必需。
  - 训练完成后，网络会自然收敛至接近 one-hot 的状态，学习到的 transforms 通常将至少 **0.99** 的权重集中在单一确定的排列上。

---

**输入输出关系与整体作用**

- **输入输出关系**：
  - 输入：原始向量 $x$ 以及控制排列的 logits $(\ell_a, \ell_b, \ell_c)$。
  - 输出：经过递归概率排列后的向量，该向量随后进入 Butterfly factors 进行线性组合。
- **在整体架构中的作用**：
  - **打破组合爆炸瓶颈**：将不可微的离散搜索转化为基于梯度的连续优化，使得模型能够扩展至 $N=1024$ 甚至更大的现实维度。
  - **恢复经典算法**：能够自动发现并恢复 Cooley-Tukey FFT 算法中的 bit-reversal permutation，以及 DCT 等其他变换所需的特定排列。
  - **保持计算高效性**：尽管训练时引入了概率分布，但在推理阶段，可以直接使用 argmax 选出的固定排列，不增加额外的推理计算开销。


---

## 4. 实验方法与实验结果

**实验设置**

- **离散变换恢复**
  - 目标：通过最小化 Frobenius 范数误差，验证 Butterfly 参数化能否自动恢复 DFT、DCT、DST、Convolution、Hadamard 等变换的快速算法。
  - 优化器：Adam。
  - 超参数调优：使用 Hyperband 自动调整学习率和初始化种子。
  - 成功标准：平均每项误差 (RMSE) 低于 1e-4，视为达到机器精度。
  - 对比基线：Sparse、Low-rank、Sparse + low-rank (Robust PCA)，保持相同的总稀疏预算。

- **神经网络压缩**
  - 任务一：单隐层网络隐藏层压缩。
    - 数据集：MNIST-bg-rot、MNIST-noise、CIFAR-10。
    - 对比基线：Unstructured、LDR-TD、Toeplitz-like、Fastfood、Circulant、Low-rank。
  - 任务二：ResNet18 架构增强。
    - 数据集：CIFAR-10。
    - 操作：在最终 FC 层前插入额外的 FC 或 BPBP 层。

- **速度评估**
  - 训练速度：Tesla P100 GPU，batch size 为 256，对比 BP 快速乘法、GEMM (cuBLAS) 和 FFT (cuFFT)。
  - 推理速度：Intel Xeon CPU，单线程，对比 BP、GEMV、FFT、DCT、DST。

---

**结果数据分析**

- **离散变换恢复效果**
  - 核心结论：BP 和 BPBP 参数化成功恢复了多种重要变换的快速算法，最高支持维度 N=1024。
  - 具体表现：
    - DFT、DCT、DST、Hadamard、Hartley 在 N=1024 时 RMSE 均低于 1e-4。
    - Convolution 在 N=512 时 RMSE 为 6.3e-05，但在 N=1024 时误差突增至 1.9e-02。
    - Legendre 变换（已知仅有 O(N log^2 N) 算法）未能完美恢复，但优于基线。
    - 随机矩阵 作为无结构对照，误差最高。

| Transform   | N = 512  | N = 1024 |
| :---------- | :------- | :------- |
| DFT         | 8.0e-05  | 5.7e-05  |
| DCT         | 3.1e-05  | 7.3e-05  |
| Convolution | 6.3e-05  | 1.9e-02  |
| Hadamard    | 6.1e-05  | 3.6e-05  |
| Legendre    | 2.6e-03  | 1.6e-03  |
| Randn       | 4.4e-02  | 3.1e-02  |

- **单隐层网络压缩性能**
  - 核心结论：BPBP 参数化在大幅减少参数量的同时，分类准确率超越了无结构全连接层。
  - 具体表现：
    - 在 CIFAR-10 上，**BPBP (complex)** 达到 **49.93%** 准确率，比 Unstructured 提升约 **3.9** 个点，参数压缩 **39.4** 倍。
    - **BPBP (real)** 准确率为 48.69%，参数压缩 **56.9** 倍，优于其他所有结构化基线。

| Method                                 | MNIST-bg-rot | MNIST-noise | CIFAR-10 | Compression factor |
| :------------------------------------- | :----------- | :---------- | :------- | :----------------- |
| Unstructured                           | 44.08        | 65.15       | 46.03    | 1                  |
| **BPBP (complex, fixed permutation)** | **46.26**    | 77.00       | **49.93**| 39.4               |
| **BPBP (real, fixed permutation)**    | 46.16        | 75.00       | 48.69    | 56.9               |
| LDR-TD                                 | 45.81        | **78.45**   | 45.33    | 56.9               |
| Toeplitz-like                          | 42.67        | 75.75       | 41.78    | 56.9               |

- **ResNet 架构增强效果**
  - 核心结论：在 ResNet18 中插入 BPBP 层能以极小的参数代价提升模型精度。
  - 具体表现：插入 BPBP 层后准确率达到 **94.01%**，优于插入普通 FC 层 (93.89%) 和无插入基线 (93.58%)，参数量仅增加 0.07%。

- **训练与推理速度**
  - 核心结论：Butterfly 实现的通用快速算法在速度上极具竞争力，逼近高度优化的专用算法。
  - 具体表现：
    - 训练阶段：比 GEMM 快 **15%**，仅比 cuFFT 慢 **40%**。
    - 推理阶段：比 GEMV 快 1-2 个数量级，速度在 FFT 的 **5** 倍以内，在 DCT/DST 的 **3** 倍以内。

![](images/speed-training-plot.png) *Training*
![](images/speed-plot.png) *Inference*

---

**消融与变体分析**

论文通过对比不同参数化配置和层级组合，分析了 Butterfly 结构的表达能力边界与设计选择的影响。

- **BP vs BPBP 层级表达力**
  - **BP (单层 Butterfly-Permutation)**：足以完美表达 DFT 和 Hadamard 变换，因为这些变换具有直接的递归分治结构。
  - **BPBP (双层 Butterfly-Permutation)**：对于 DCT、DST 和 Convolution 是必需的。例如，Convolution 需要通过 FFT、点乘、逆 FFT 实现，单层结构无法捕获这种复合操作。

- **Permutation 学习策略**
  - **连续松弛**：将离散的排列组合选择转化为 3 个 logits 的 Sigmoid 概率组合，避免了组合爆炸。
  - **权重共享**：允许在 log2(N) 个递归层中共享 Permutation logits，反映了某些算法的自相似性，进一步减少参数量。
  - **收敛性**：实验表明，无需额外的熵正则化，学习到的 Permutation 概率通常会高度集中（权重 >0.99），自动逼近离散的最优排列。

- **实数 vs 复数参数化**
  - 在单隐层网络压缩任务中，**复数 BPBP** (Compression factor 39.4) 的分类准确率普遍高于**实数 BPBP** (Compression factor 56.9)。这表明复数空间提供了更强的表达能力，尽管参数量略有增加。

- **固定 vs 学习 Permutation**
  - 在单隐层网络压缩实验中，表格标注为 "fixed permutation"。这表明在端到端 ML 任务中，固定基础的 Permutation（如 bit-reversal）结合可学习的 Butterfly 因子，已足以提供强大的归纳偏置，简化训练过程。

---

