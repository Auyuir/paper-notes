# Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations 图表详解

### Butterfly matrix for N = 16. From left to right: single copy of B16, blocks of B8, blocks of B4, blocks of B2.

![butterfly-pattern-line.png](images/butterfly-pattern-line.png)

- **图像对象**：该图展示了当 **N = 16** 时，**Butterfly matrix / butterfly factorization** 中 4 个连续稀疏因子的非零结构；从左到右分别对应：
  
  | 位置 | 因子结构 | 论文说明 | 作用尺度 |
  |---|---|---|---|
  | 第 1 幅 | **single copy of B16** | 一个完整的 **B16 butterfly factor** | 全局 16 维混合 |
  | 第 2 幅 | **blocks of B8** | 两个块对角的 **B8** | 两个 8 维子问题 |
  | 第 3 幅 | **blocks of B4** | 四个块对角的 **B4** | 四个 4 维子问题 |
  | 第 4 幅 | **blocks of B2** | 八个块对角的 **B2** | 八个 2 维局部混合 |

- **核心视觉含义**：
  - 每个子图都是一个 **16 × 16 sparse matrix pattern**。
  - **红色方块表示非零参数 / learnable entries**。
  - **白色区域表示零元素**。
  - 整体展示了 butterfly factorization 中从 **大尺度交互到小尺度交互** 的递归稀疏结构。
  - 这些稀疏因子相乘后形成完整的 **butterfly matrix**，用于实现 **O(N log N)** 的快速矩阵-向量乘法。

- **第 1 幅：single copy of B16**
  - 对应最大尺度的 **B16**。
  - 结构可理解为：
    - 上半部分输出同时连接输入的前半部分和后半部分。
    - 下半部分输出也同时连接输入的前半部分和后半部分。
  - 在矩阵形式上接近论文中的：
    - **B_N = [[D1, D2], [D3, D4]]**
  - 其中 **D1, D2, D3, D4** 都是 diagonal matrices。
  - 因此每一行大约有 **2 个非零元素**，每一列也大约有 **2 个非零元素**。
  - 视觉上出现两组斜向红色轨迹，体现了跨越 **N/2 = 8** 间隔的配对交互。

- **第 2 幅：blocks of B8**
  - 表示两个 **B8** 以 block diagonal 方式排列：
    - 一个作用于前 8 个维度。
    - 一个作用于后 8 个维度。
  - 该层不再做 16 维全局混合，而是在两个 **8 维子空间** 内分别进行 butterfly mixing。
  - 红色非零点被限制在两个较小的对角块区域中。
  - 这对应递归展开中的：
    - **diag(B8, B8)**

- **第 3 幅：blocks of B4**
  - 表示四个 **B4** 的 block diagonal 结构。
  - 每个 **B4** 只作用于一个 4 维局部块。
  - 红色方块进一步聚集成四个小型局部模式。
  - 该图体现了 divide-and-conquer 中更深一层的递归：
    - 原始 16 维问题被拆成 **4 个 4 维子问题**。
  - 对应表达式：
    - **diag(B4, B4, B4, B4)**

- **第 4 幅：blocks of B2**
  - 表示八个 **B2** 小块。
  - 每个 **B2** 是最小的 butterfly mixing unit。
  - 视觉上呈现为沿主对角线排列的多个 **2 × 2 dense blocks**。
  - 每个小块内部有 4 个红色非零元素，表示两个输入之间的最基本线性组合。
  - 这是 butterfly recursion 的最底层。

- **递归结构总结**：

  | 层级 | 因子 | block 数量 | 每个 block 大小 | 交互范围 | 直观含义 |
  |---|---:|---:|---:|---:|---|
  | Layer 1 | **B16** | 1 | 16 × 16 | 最大 | 全局二分混合 |
  | Layer 2 | **B8** | 2 | 8 × 8 | 中等 | 两个半区内部混合 |
  | Layer 3 | **B4** | 4 | 4 × 4 | 较小 | 四个局部块混合 |
  | Layer 4 | **B2** | 8 | 2 × 2 | 最小 | 成对元素混合 |

- **数学结构对应关系**：
  - 对于 **N = 16 = 2⁴**，butterfly matrix 由 **log₂N = 4** 层 butterfly factors 组成。
  - 图中 4 个子图正好对应这 4 层。
  - 整体乘积可表示为：
    - **B^(16) = B16 · diag(B8, B8) · diag(B4, B4, B4, B4) · diag(B2, ..., B2)**
  - 这与论文中 FFT factorization 的展开形式一致。

- **稀疏性分析**：
  - 每一层 butterfly factor 都是高度稀疏的。
  - 对于 **N = 16**，每层大致有 **2N = 32** 个非零元素。
  - 总层数为 **log₂16 = 4**。
  - 因此总计算量约为：
    - **O(N log N) = O(16 × 4)**
  - 相比 dense matrix-vector multiplication 的 **O(N²) = O(256)**，该结构显著降低计算成本。

- **与 FFT 的关系**：
  - 图中的 pattern 与 **Cooley-Tukey FFT** 的计算图高度对应。
  - 在 FFT 中，每一层 butterfly factor 执行一组加权的两两组合，权重通常称为 **twiddle factors**。
  - 对于 DFT，具体参数由复数单位根决定。
  - 在本文方法中，这些非零位置固定，但对应数值是 **learnable parameters**，因此可以通过优化学习 DFT、DCT、Hadamard、convolution 等快速变换。

- **图像传达的关键思想**：
  - **不是学习任意 dense matrix**，而是学习一个由多层稀疏矩阵乘积构成的结构化矩阵。
  - **稀疏模式固定且递归**，避免了直接学习任意 sparsity pattern 的离散搜索困难。
  - **参数数量线性增长**，但通过多层组合获得较强表达能力。
  - **快速算法自动产生**：一旦学到这些红色位置上的参数，矩阵-向量乘法天然就是 butterfly-style fast multiplication。

- **与论文方法 BP / BPBP 的关系**：
  - 图中展示的是 **B part**，即 butterfly matrix 的稀疏因子结构。
  - 完整的 **BP parameterization** 还包括一个 permutation：
    - **T_N = B^(N) P^(N)**
  - 对于更复杂的变换，例如 convolution，论文使用：
    - **BPBP = B₂P₂B₁P₁**
  - 图像本身主要说明 **B^(N)** 的层级稀疏模式，而不是 permutation-learning 部分。

- **为什么该结构重要**：
  - **表达力**：可以精确表示 FFT、Hadamard；通过 BPBP 可表示 DCT、DST、convolution。
  - **效率**：矩阵-向量乘法复杂度从 **O(N²)** 降为 **O(N log N)**。
  - **可学习性**：只需优化红色非零位置的参数，而不是搜索任意稀疏结构。
  - **可部署性**：由基础 linear algebra 操作组成，适合 PyTorch / TensorFlow 等框架。
  - **压缩性**：参数数量远少于 dense layer，可用于 neural network compression。

- **视觉结论**：
  - 该图用 4 个稀疏矩阵模式清晰展示了 **Butterfly factorization 的递归分治本质**。
  - 从左到右，交互尺度逐层减半：
    - **16 → 8 → 4 → 2**
  - 每层只进行有限的局部或跨块连接，但多层相乘后可以形成复杂的全局线性变换。
  - 这正是论文能够“学习快速算法”的核心结构先验。

### Three binary choices for constructing the permutation used at every step of the recursive process. One of 8 possible permutations can be constructed by multiplying a subset of these matrices in the presented order.

![permutation-matrices.png](images/permutation-matrices.png)

- 这张图展示了在 Butterfly Factorization 中，递归 permutation 模块的 **三个基础二元选择**：**\(P_8^c\)**、**\(P_8^b\)**、**\(P_8^a\)**。每个矩阵都是一个 **\(8 \times 8\) permutation matrix**，红色方块表示该位置为 **1**，空白位置为 **0**。

- 图中三个 permutation matrix 的核心作用如下：

| 矩阵 | 图中模式 | 对输入向量 \(x=[x_0,x_1,\dots,x_7]\) 的作用 | 论文中的语义 |
|---|---|---|---|
| **\(P_8^c\)** | 前半部分保持对角线，后半部分反对角线 | \([x_0,x_1,x_2,x_3,x_7,x_6,x_5,x_4]\) | **reverse second half**，反转后半段 |
| **\(P_8^b\)** | 前半部分反对角线，后半部分保持对角线 | \([x_3,x_2,x_1,x_0,x_4,x_5,x_6,x_7]\) | **reverse first half**，反转前半段 |
| **\(P_8^a\)** | 偶数索引集中到前半，奇数索引集中到后半 | \([x_0,x_2,x_4,x_6,x_1,x_3,x_5,x_7]\) | **separate even and odd indices**，分离偶数/奇数索引 |

- 三个矩阵分别对应递归分解中的三个可选操作：

  - **\(P^a\)：even-odd split**
    - 将原始序列按索引奇偶划分。
    - 对 \(N=8\)：
      - 原序列：\([0,1,2,3,4,5,6,7]\)
      - 变换后：\([0,2,4,6,1,3,5,7]\)
    - 这是 **FFT / Cooley-Tukey** 中最典型的 divide step。

  - **\(P^b\)：reverse first half**
    - 只反转前 \(N/2\) 个元素。
    - 对 \(N=8\)：
      - 原序列：\([0,1,2,3,4,5,6,7]\)
      - 变换后：\([3,2,1,0,4,5,6,7]\)

  - **\(P^c\)：reverse second half**
    - 只反转后 \(N/2\) 个元素。
    - 对 \(N=8\)：
      - 原序列：\([0,1,2,3,4,5,6,7]\)
      - 变换后：\([0,1,2,3,7,6,5,4]\)

- 图像的视觉结构可以这样理解：

| 视觉特征 | 含义 |
|---|---|
| **红色方块** | permutation matrix 中的非零元素，数值为 **1** |
| **每行一个红块** | 每个输出位置只取一个输入元素 |
| **每列一个红块** | 每个输入元素只被使用一次 |
| **对角线结构** | 对应 identity 或保持顺序 |
| **反对角线结构** | 对应局部 reverse 操作 |
| **交错分布结构** | 对应 even-odd reordering |

- 这三个矩阵不是独立使用，而是按论文中指定的顺序组合：

  - **组合顺序**：
    - \[
      P_{N/2^k} = P^c P^b P^a
      \]
    - 但每个 \(P^a\)、\(P^b\)、\(P^c\) 都是 **binary choice**：可以选择使用，也可以选择不用。

  - 因此每一层递归 permutation 有：
    - \(P^a\)：用 / 不用
    - \(P^b\)：用 / 不用
    - \(P^c\)：用 / 不用

  - 总共有：
    - \[
      2^3 = 8
      \]
    - 种可能的 permutation。

- 对 \(N=8\)，这 8 种 permutation 可以由下表表示：

| 是否使用 \(P^a\) | 是否使用 \(P^b\) | 是否使用 \(P^c\) | 组合形式 |
|---|---|---|---|
| 否 | 否 | 否 | \(I\) |
| 是 | 否 | 否 | \(P^a\) |
| 否 | 是 | 否 | \(P^b\) |
| 否 | 否 | 是 | \(P^c\) |
| 是 | 是 | 否 | \(P^bP^a\) |
| 是 | 否 | 是 | \(P^cP^a\) |
| 否 | 是 | 是 | \(P^cP^b\) |
| 是 | 是 | 是 | \(P^cP^bP^a\) |

- 论文中的关键点是：作者没有在所有 permutation 上进行离散搜索，而是将这些选择变成 **可微分的概率化选择**。

- 原始离散搜索空间为：

  - 每一层有 **8** 种选择。
  - 总共有 \(\log_2 N\) 层递归。
  - 搜索空间大小：
    - \[
      8^{\log_2 N}
      \]
  - 这会导致组合爆炸。

- 作者的连续松弛方式是：

  - 对每个基础 permutation \(P^s\)，其中 \(s \in \{a,b,c\}\)，学习一个概率 \(p_s\)。
  - 用如下形式替代硬选择：
    - \[
      p_s P^s + (1-p_s)I
      \]
  - 整体 permutation 写成：
    - \[
      P_{N/2^k}
      =
      \prod_{s=c,b,a}
      \left(
      p_s P^s + (1-p_s)I
      \right)
      \]

- 这张图在论文方法中的作用非常关键：

| 作用 | 说明 |
|---|---|
| **定义可学习 permutation 空间** | 将复杂 permutation 限制到递归算法常见的结构 |
| **降低搜索难度** | 从指数级离散搜索转为少量连续参数优化 |
| **保留 FFT/DCT 等结构先验** | even-odd split 和 reverse 操作正是许多 fast transform 的核心 |
| **支持端到端训练** | permutation 概率可通过 gradient descent 学习 |
| **配合 Butterfly factor** | permutation 完成 divide，butterfly matrix 完成 combine |

- 这三个 permutation 与具体 fast transform 的关系：

| Transform | 相关 permutation 行为 |
|---|---|
| **DFT / FFT** | 主要依赖 **even-odd split \(P^a\)**，递归后形成 **bit-reversal permutation** |
| **DCT** | 需要 even-odd split，并常伴随后半部分 reverse |
| **DST** | 类似 DCT，也涉及 split、reverse 和 diagonal scaling |
| **Hadamard Transform** | 可用 identity permutation，即不一定需要这些重排 |
| **Convolution** | 通过 FFT 与 inverse FFT 表达，因此间接依赖这些 permutation |

- 从算法设计角度看，图中的三个矩阵体现了论文的核心思想：

  - **不是学习任意 permutation**。
  - **不是暴力搜索 fast algorithm**。
  - 而是利用 fast transform 中普遍存在的递归结构：
    - 分组
    - 反转
    - 奇偶拆分
    - 局部重排

- 这种设计的优势是：

  - **表达能力足够强**：可以覆盖 FFT、DCT、DST、Hadamard、Convolution 等重要变换。
  - **参数量极少**：每层 permutation 只需学习 3 个 logits。
  - **可微分**：适合 PyTorch / TensorFlow 等自动微分框架。
  - **易于收敛到硬 permutation**：论文指出实际训练中概率通常会接近 0 或 1，例如某个选择权重超过 **0.99**。

- 总结来说，这张图展示的是 Butterfly Factorization 中 permutation 学习模块的最小构件：

  - **\(P^a\)** 负责 **even-odd separation**。
  - **\(P^b\)** 负责 **first-half reversal**。
  - **\(P^c\)** 负责 **second-half reversal**。
  - 三者以固定顺序组合，形成每个递归层的 permutation。
  - 通过连续松弛，模型可以自动学习接近真实 fast transform algorithm 的重排结构。

### image

![heatmap.png](images/heatmap.png)

- **图像内容概述**
  - 该图是论文中用于评估不同矩阵压缩/分解方法重构经典线性变换能力的 **heatmap**。
  - 横向分为四个方法面板：**Butterfly**、**Sparse**、**Low rank**、**Sparse + Low rank**。
  - 纵轴为待重构的线性变换：**DFT、DCT、DST、Conv、Hadamard、Hartley、Legendre、Randn**。
  - 横轴为矩阵维度：**N = 8, 16, 32, 64, 128, 256, 512, 1024**。
  - 颜色表示重构误差 **RMSE**，采用对数色标：
  
| 颜色 | 误差量级 | 含义 |
|---|---:|---|
| **深蓝** | **≈ 1e-4** | 几乎精确重构，达到机器精度附近 |
| **浅蓝 / 白色** | **1e-3 到 1e-2** | 中等误差 |
| **粉红** | **1e-1** | 较大误差 |
| **红色** | **≈ 1e0** | 重构失败或非常差 |

- **核心视觉结论**
  - **Butterfly** 面板几乎在所有经典快速变换上呈现深蓝色，说明 **Butterfly factorization 能够以极低误差恢复这些 fast transforms**。
  - **Sparse、Low rank、Sparse + Low rank** 三个基线方法大多呈现粉红或红色，说明它们即使在相同计算/参数预算下，也难以有效表示这些结构化线性变换。
  - **Legendre** 和 **Randn** 是两个明显例外：
    - **Legendre** 在 Butterfly 中不是深蓝，而是从粉红逐渐过渡到蓝紫，说明 Butterfly 能部分近似但不能精确捕获。
    - **Randn** 在所有方法中误差都较高，说明随机高斯矩阵缺乏可利用的递归结构，不能被低复杂度结构有效压缩。

- **Butterfly 方法表现**
  
| Transform | Butterfly 视觉表现 | 解释 |
|---|---|---|
| **DFT** | **全深蓝** | 成功恢复 **FFT / Cooley-Tukey** 型分解 |
| **DCT** | **全深蓝** | 可由 FFT 与额外 permutation / scaling 表示，符合 **BPBP** 结构 |
| **DST** | **全深蓝** | 与 DCT 类似，可通过 FFT 结构表达 |
| **Conv** | **N ≤ 512 深蓝，N=1024 明显变浅** | 卷积可由 **FFT → diagonal → inverse FFT** 表示，即 **BPBP**；但 N=1024 训练未完全收敛或优化更困难 |
| **Hadamard** | **全深蓝** | 递归定义天然匹配 Butterfly |
| **Hartley** | **全深蓝** | 与 DFT 强相关，实数形式也可被 Butterfly 捕获 |
| **Legendre** | **粉红到蓝紫渐变** | 具有递归结构但不属于简单 **BP/BPBP** 可精确表达的类别 |
| **Randn** | **粉红** | 无结构随机矩阵，Butterfly 无法低误差表示 |

- **与论文附录数值对应**
  
| Transform | N=8 | N=64 | N=512 | N=1024 | 结论 |
|---|---:|---:|---:|---:|---|
| **DFT** | 3.1e-06 | 1.0e-05 | 8.0e-05 | 5.7e-05 | **机器精度级恢复** |
| **DCT** | 4.4e-06 | 1.2e-05 | 3.1e-05 | 7.3e-05 | **机器精度级恢复** |
| **DST** | 1.1e-06 | 5.1e-05 | 3.6e-05 | 4.6e-05 | **机器精度级恢复** |
| **Conv** | 4.0e-06 | 7.6e-05 | 6.3e-05 | 1.9e-02 | **N=1024 明显退化** |
| **Hadamard** | 8.8e-07 | 3.9e-05 | 6.1e-05 | 3.6e-05 | **机器精度级恢复** |
| **Hartley** | 3.4e-06 | 1.3e-05 | 4.5e-05 | 3.6e-05 | **机器精度级恢复** |
| **Legendre** | 3.4e-02 | 1.4e-02 | 2.6e-03 | 1.6e-03 | **可近似但不能精确恢复** |
| **Randn** | 1.4e-01 | 1.1e-01 | 4.4e-02 | 3.1e-02 | **随机矩阵难以结构化压缩** |

- **Sparse 方法表现**
  - **Sparse** 面板大部分为红色或粉红色，尤其是 **DFT、DCT、DST、Hadamard、Hartley** 等经典变换。
  - 这说明这些矩阵虽然有快速算法，但其显式矩阵通常是 **dense matrix**，并不稀疏。
  - 单纯保留最大元素的 sparse approximation 无法捕获其算法结构。
  - **Legendre** 在高维处略微变蓝，说明其某些条目分布或结构可能被稀疏近似部分利用，但仍远弱于 Butterfly。

- **Low rank 方法表现**
  - **Low rank** 面板同样整体偏红。
  - 经典正交/酉变换如 **DFT、DCT、Hadamard** 通常满秩，奇异值分布不适合低秩截断。
  - 因此，low-rank approximation 在相同参数预算下误差较大。
  - **Legendre** 随 N 增大颜色变浅蓝，说明其在该归一化和采样方式下可能存在一定可低秩近似性，但仍不能达到 Butterfly 在可表示变换上的机器精度。

- **Sparse + Low rank 方法表现**
  - **Sparse + Low rank** 比单独 Sparse 或 Low rank 略有改善，但仍无法接近 Butterfly。
  - 该方法能够同时利用局部大元素和低秩趋势，但没有显式建模 **recursive divide-and-conquer structure**。
  - 对 **DFT、DCT、DST、Hadamard、Hartley** 这类具有快速算法但矩阵本身 dense 且满秩的变换，仍然表现较差。
  - 对 **Legendre** 也有一定改善，但仍未达到精确恢复。

- **图中最重要的对比**
  
| 对比点 | Butterfly | Sparse / Low rank / Sparse+Low rank |
|---|---|---|
| 是否建模递归结构 | **是** | **否** |
| 是否能恢复 FFT-like transforms | **能** | **基本不能** |
| 参数复杂度 | **O(N)** 或少量 BP 组合 | 相同预算下表达力不足 |
| 乘法复杂度 | **O(N log N)** | 取决于稀疏度/秩，但误差高 |
| 对随机矩阵 | **不能精确表示** | 也不能 |
| 对 Legendre | **部分近似** | 部分近似但通常更弱 |

- **方法层面的解释**
  - **Butterfly factorization** 的优势不是普通稀疏性，而是 **structured sparsity across multiple factors**。
  - 一个 DFT 矩阵本身是 dense matrix，但可以分解为多个稀疏的 **butterfly factors** 与 permutation 的乘积。
  - 因此，Butterfly 可以用较少参数表达复杂全局混合，而 Sparse 只能表达单层稀疏矩阵，Low rank 只能表达低维子空间结构。
  - 图像直接验证了论文核心观点：**fast algorithms correspond to products of sparse structured matrices, not necessarily to sparse or low-rank matrices themselves**。

- **关于 Legendre 的细节**
  - **Legendre** 行在 Butterfly 中不是深蓝，说明 **Discrete Legendre Transform (DLT)** 不完全属于该论文简单 **BP/BPBP** 参数化能够精确表示的类别。
  - 论文指出 DLT 已知快速算法复杂度为 **O(N log² N)**，其递归结构比 DFT/DCT/Hadamard 更复杂。
  - 图中 Butterfly 对 Legendre 随 N 增大误差下降，说明它捕获了一部分递归规律，但表达能力仍不足以完全恢复。

- **关于 Randn 的意义**
  - **Randn** 是随机高斯矩阵，用作无结构基线。
  - 图中所有方法在 Randn 上误差都较高，说明 Butterfly 并不是万能矩阵近似器。
  - 这反而强化了实验可信度：Butterfly 的成功来自其对 **fast structured transforms** 的归纳偏置，而不是简单过拟合任意矩阵。

- **总体结论**
  - 该 heatmap 清晰表明：**Butterfly factorization 在恢复经典快速线性变换方面显著优于 Sparse、Low rank、Sparse + Low rank 基线**。
  - 对 **DFT、DCT、DST、Hadamard、Hartley**，Butterfly 基本达到 **1e-4 RMSE 或更低**，对应机器精度附近。
  - 对 **Convolution**，Butterfly 在 N≤512 时也达到机器精度，N=1024 时误差上升，主要体现优化或训练难度。
  - 对 **Legendre** 和 **Randn**，Butterfly 无法精确恢复，说明其表达能力有明确边界。
  - 该图是论文主张的关键证据：**通过学习 Butterfly sparse product structure，可以自动恢复多类经典 O(N log N) fast algorithms，而传统矩阵压缩方法无法做到。**

### Training

![speed-training-plot.png](images/speed-training-plot.png)

- **图片对象**：`speed-training-plot.png`，展示训练阶段不同线性变换实现相对 **dense GEMM** 的加速比。

- **实验语境**：
  - 该图对应论文 Section 4.3 的 **Training speed comparison**。
  - 比较对象包括：
    - **FFT**：红色曲线，调用高度优化的 **cuFFT**。
    - **Butterfly**：蓝色曲线，论文提出的 **Butterfly factorization CUDA implementation**。
    - **GEMM**：黑色水平线，作为基线，速度记为 **1×**。
  - 训练时间包含 **forward + backward**。
  - 文中说明实验设置为 **batch size = 256**，GPU 为 **Tesla P100 16GB**。

- **坐标轴含义**：

| 元素 | 含义 |
|---|---|
| 横轴 **N** | 输入/矩阵维度，取值为 **2⁶ 到 2¹³** |
| 纵轴 **Speedup over GEMM** | 相对 dense matrix-matrix multiply 的加速比 |
| 黑色水平线 | **GEMM baseline = 1×** |
| 红色曲线 | **FFT** 训练速度相对 GEMM 的加速 |
| 蓝色曲线 | **Butterfly** 训练速度相对 GEMM 的加速 |
| 纵轴尺度 | 近似 **log scale**，突出跨数量级变化 |

- **主要读数趋势**：

| N | FFT 加速比 | Butterfly 加速比 | 观察 |
|---:|---:|---:|---|
| **2⁶ = 64** | 约 **0.55×** | 约 **0.48×** | 两者均慢于 GEMM |
| **2⁷ = 128** | 约 **0.75×** | 约 **0.45×** | 小规模下 kernel overhead 明显 |
| **2⁸ = 256** | 约 **0.50×** | 约 **0.45×** | 仍低于 GEMM |
| **2⁹ = 512** | 约 **0.60×** | 约 **0.55×** | 接近但仍未超过 GEMM |
| **2¹⁰ = 1024** | 约 **1.5×** | 约 **1.1×** | 两者开始超过 GEMM |
| **2¹¹ = 2048** | 约 **3.5×** | 约 **2.5×** | Butterfly 明显加速 |
| **2¹² = 4096** | 约 **13×** | 约 **4×** | FFT 优势扩大 |
| **2¹³ = 8192** | 约 **30×** | 约 **7–8×** | 大规模下二者均显著快于 GEMM |

- **核心结论**：
  - **Butterfly 在 N ≥ 2¹⁰ 时训练速度超过 dense GEMM**。
  - 在 **N = 1024** 附近，Butterfly 约为 **1.1× GEMM**，与论文文字描述“训练时间比 dense matrix multiply 快约 **15%**”一致。
  - **FFT 始终快于 Butterfly**，尤其在大尺寸时差距扩大。
  - 但 Butterfly 是 **通用可学习结构**，不是针对某个固定 transform 的手工优化实现；因此能接近 FFT 的训练效率具有实际意义。

- **小尺寸性能解释**：
  - 当 **N ≤ 512** 时，FFT 和 Butterfly 都低于 **1×**。
  - 原因主要是：
    - **GPU kernel launch overhead** 占比较高。
    - 小矩阵下 GEMM/cuBLAS 已高度优化。
    - Butterfly 的分层稀疏计算在小规模下难以充分利用 GPU 并行度。
  - 因此，小尺寸时 **理论复杂度优势尚未转化为实际训练速度优势**。

- **大尺寸性能解释**：
  - 当 **N 增大** 后，dense GEMM 的复杂度为 **O(N²)**。
  - FFT 和 Butterfly 的复杂度接近 **O(N log N)**。
  - 随着 N 从 **2¹⁰ 增至 2¹³**，复杂度差异迅速放大，因此二者加速比持续提升。
  - **Butterfly 曲线稳步上升**，说明其 fast multiplication 在训练场景中确实能带来可扩展收益。

- **FFT 与 Butterfly 的差异**：
  - **FFT**：
    - 专门针对 Fourier transform。
    - 使用高度工程优化的 **cuFFT**。
    - 大尺寸下最高达到约 **30× GEMM**。
  - **Butterfly**：
    - 是可学习的通用结构化线性层。
    - 可表示/逼近 **DFT、DCT、DST、Hadamard、convolution** 等多类变换。
    - 大尺寸下达到约 **7–8× GEMM**。
    - 虽然慢于 FFT，但不依赖 transform-specific 手工实现。

- **论文论点支撑**：
  - 该图支撑论文的一个关键主张：**Butterfly factorization 不只是参数少，也能在训练阶段获得实际速度收益**。
  - 它说明：
    - 学到的结构化变换天然对应 **fast algorithm**。
    - 该 fast algorithm 可用于 **forward/backward training**。
    - 在实际 GPU 环境中，Butterfly 并非只有理论复杂度优势，而是能超过 dense GEMM。

- **对机器学习应用的含义**：
  - 对于大维度 fully-connected layer 或 structured linear layer，Butterfly 可作为 **drop-in replacement**。
  - 它同时提供：
    - **O(N log N)** 训练/推理复杂度。
    - **O(N)** 参数量。
    - 可学习的结构化归纳偏置。
  - 图中训练速度结果与论文中 CIFAR-10 单隐藏层实验的压缩结果相互补充：Butterfly 不仅能减少参数，还能提升实际计算效率。

- **总体评价**：
  - **Butterfly 在中大规模 N 上显著快于 GEMM，但仍慢于专用 FFT。**
  - 这是一种合理权衡：牺牲部分专用 kernel 的极致速度，换取 **可学习性、通用性、可压缩性和跨 transform 表达能力**。
  - 因此，该图的核心信息不是“Butterfly 比 FFT 快”，而是 **Butterfly 作为通用 learned fast transform，已经能接近专用 fast transform，并明显优于 dense GEMM**。

### Inference

![speed-plot.png](images/speed-plot.png)

- **图像类型与目的**
  - 该图为论文中 **Inference speed comparison** 的结果图，展示不同快速线性变换在推理阶段相对于 **dense matrix-vector multiplication / GEMV** 的加速比。
  - 对比对象包括：
    - **FFT**：红色曲线
    - **DCT**：橙色曲线
    - **DST**：绿色曲线
    - **BP / Butterfly Parameterization**：蓝色曲线
  - 横轴为输入维度 **N**，取值从 **2⁶ 到 2¹³**。
  - 纵轴为 **Speedup over GEMV**，即相对 GEMV 的加速倍数。
  - 黑色水平线位于 **1×**，表示与 GEMV 速度相当；高于该线表示比 GEMV 更快，低于该线表示比 GEMV 更慢。

- **坐标轴与尺度**
  - 横轴：**N**
    - 使用指数刻度，标注为 **2⁶, 2⁷, ..., 2¹³**。
    - 对应矩阵/向量规模从 **64 到 8192**。
  - 纵轴：**Speedup over GEMV**
    - 使用对数尺度。
    - 主要刻度包括 **10⁰ = 1×**, **10¹ = 10×**, **10² = 100×**。
  - 由于纵轴是对数尺度，曲线的上升代表加速比呈倍数级增长，而非线性增长。

- **主要视觉结论**
  - **所有方法在较大 N 时都显著快于 GEMV**。
  - **FFT 始终是最快方法**，尤其在大规模输入下优势明显。
  - **DCT 和 DST 性能非常接近**，二者曲线几乎重合，DST 略高于 DCT。
  - **BP 在小规模 N 时低于 GEMV 或接近 GEMV，但随着 N 增大快速超过 GEMV**。
  - 到 **N = 2¹³** 时，BP 仍能达到约 **150×** 左右的 GEMV 加速，说明 Butterfly 结构在推理中具有很强的实际效率。

- **近似数值读取**

| **N** | **FFT speedup** | **DCT speedup** | **DST speedup** | **BP speedup** |
|---:|---:|---:|---:|---:|
| **2⁶ = 64** | ~0.7× | ~0.45× | ~0.43× | ~0.18× |
| **2⁷ = 128** | ~1.0× | ~0.5× | ~0.48× | ~0.19× |
| **2⁸ = 256** | ~3× | ~1.6× | ~1.5× | ~0.6× |
| **2⁹ = 512** | ~9–10× | ~4× | ~4× | ~1.5× |
| **2¹⁰ = 1024** | ~22× | ~10× | ~11× | ~4.5× |
| **2¹¹ = 2048** | ~26–28× | ~22× | ~25× | ~10–11× |
| **2¹² = 4096** | ~210× | ~130× | ~130× | ~65× |
| **2¹³ = 8192** | ~450× | ~250× | ~280× | ~170× |

- **各方法趋势分析**
  - **FFT**
    - 红色曲线整体最高。
    - 在 **N = 2⁶** 时略低于 GEMV，约 **0.7×**。
    - 在 **N = 2⁷** 左右达到与 GEMV 持平。
    - 从 **N = 2⁸** 开始明显快于 GEMV。
    - 在 **N = 2¹² 到 2¹³** 区间出现显著跃升，最终达到数百倍加速。
    - 这反映了 **FFT 专用实现高度优化**，且其 **O(N log N)** 复杂度在大 N 下明显优于 GEMV 的 **O(N²)**。

  - **DCT**
    - 橙色曲线整体低于 FFT，但高于或接近 DST。
    - 小规模时低于 GEMV。
    - 在 **N = 2⁸** 左右超过 GEMV。
    - 到 **N = 2¹³** 时达到约 **250×** 加速。
    - 说明专用 **DCT** 实现同样受益于快速变换结构，在大规模推理时非常高效。

  - **DST**
    - 绿色曲线与 DCT 几乎重合。
    - 在中大规模时略高于 DCT。
    - 到 **N = 2¹³** 时约为 **280×**。
    - 表明 **DST** 的实现效率与 DCT 相当，且在部分规模下略优。

  - **BP / Butterfly Parameterization**
    - 蓝色曲线整体低于 FFT/DCT/DST。
    - 在 **N = 2⁶ 到 2⁸** 时低于 GEMV，说明小规模下 Butterfly 的结构化计算开销尚未被复杂度优势抵消。
    - 在 **N = 2⁹** 左右超过 GEMV。
    - 从 **N = 2¹⁰** 开始加速明显。
    - 到 **N = 2¹³** 时达到约 **170×** 加速。
    - 虽然不如手工优化的 FFT/DCT/DST，但作为一种**通用可学习结构**，BP 的推理效率已经非常强。

- **关键比较**

| **比较对象** | **观察结果** | **含义** |
|---|---|---|
| **FFT vs BP** | FFT 始终快于 BP，大规模下约为 BP 的 2–3 倍 | 专用 FFT kernel 仍有明显工程优化优势 |
| **DCT/DST vs BP** | DCT/DST 大多数情况下快于 BP，约为 BP 的 1.5–3 倍 | 专用 transform 实现仍更高效 |
| **BP vs GEMV** | BP 在 N ≥ 2⁹ 后明显快于 GEMV | Butterfly 的 **O(N log N)** 结构在大规模下发挥优势 |
| **FFT/DCT/DST vs GEMV** | 大规模下可达到数百倍加速 | 经典快速变换相比 dense GEMV 具备压倒性效率优势 |

- **与论文论点的关系**
  - 该图支撑论文中的核心观点：**Butterfly Parameterization 能以通用方式学习快速线性变换，并自动获得 O(N log N) 推理复杂度**。
  - 虽然 **BP** 没有针对某个特定 transform 做手工优化，但仍能在大规模输入上实现远超 **GEMV** 的加速。
  - 图中 BP 与 FFT/DCT/DST 的差距说明：
    - **专用实现仍然更快**；
    - 但 **BP 的优势在于通用性和可学习性**，它不依赖人工设计某个具体 transform 的算法。
  - 因此，这张图强调的是：**BP 牺牲少量常数级效率，换取了更强的表达能力、可学习性和跨平台实现便利性**。

- **复杂度解释**
  - **GEMV**：
    - dense matrix-vector multiplication。
    - 复杂度为 **O(N²)**。
    - 当 N 增大时，计算量快速上升。
  - **FFT / DCT / DST / BP**：
    - 均利用结构化稀疏因子分解或快速递归结构。
    - 复杂度约为 **O(N log N)**。
    - 因此随着 N 增大，相对 GEMV 的加速比会越来越高。
  - 图中曲线随 N 上升而整体上升，正是 **O(N log N)** 相对 **O(N²)** 的体现。

- **图中最重要的结论**
  - **BP 在推理阶段可以比 GEMV 快一个到两个数量级**。
  - **在 N = 8192 时，BP 约有 170× 加速**。
  - **FFT 是最快的基准方法，但 BP 已经接近专用快速变换实现的同一数量级**。
  - **Butterfly 方法不是为了替代高度优化的 FFT，而是为了提供一种可学习、通用、压缩且快速的线性层参数化方式**。
  - 该图证明了 BP 不仅理论上具有 **O(N log N)** 复杂度，而且在实际 CPU 推理中也能转化为显著加速。

