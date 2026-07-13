# Monarch: Expressive Structured Matrices for Efficient and Accurate Training 原文翻译

# Monarch：用于高效且准确训练的富有表达力的结构化矩阵

## 摘要

大型神经网络在许多领域表现出色，但它们的训练和微调成本很高。一种降低其计算/内存需求的流行方法是用结构化矩阵（例如，稀疏、低秩、傅里叶变换）替代稠密权重矩阵。这些方法尚未得到广泛采用，原因在于：(1) 在端到端训练中，其效率与质量的权衡不佳；(2) 在稠密到稀疏的微调中，缺乏可处理的算法来近似给定的稠密权重矩阵。为了解决这些问题，我们提出了一类矩阵，它们是*硬件高效的*（被参数化为两个块对角矩阵的乘积，以实现更好的硬件利用率）且*富有表达力的*（能够表示许多常用的变换）。令人惊讶的是，用 Monarch 矩阵近似稠密权重矩阵的问题虽然是非凸的，但具有解析最优解。Monarch 矩阵的这些特性为训练和微调稀疏及稠密模型解锁了新的方法。我们通过实验验证了 Monarch 能够在多个端到端稀疏训练应用中实现良好的准确率与效率权衡：在 ImageNet 分类和 Wikitext-103 语言建模上，以可比的模型质量将 ViT 和 GPT-2 的训练速度提升了 2$\times$，并将 PDE 求解和 MRI 重建任务上的误差降低了 40%。在稀疏到稠密的训练中，通过一种称为“反向稀疏化”的简单技术，Monarch 矩阵作为一种有用的中间表示，在 OpenWebText 上将 GPT-2 预训练速度提升了 2$\times$，且质量没有下降。同样的技术甚至比创下 MLPerf 1.1 纪录的 Nvidia 高度优化实现带来了 23% 的 BERT 预训练加速。在稠密到稀疏的微调中，作为概念验证，我们的 Monarch 近似算法在 GLUE 上将 BERT 微调速度提升了 1.7$\times$，且准确率相当。

---

# 引言

大型神经网络在许多领域表现出色，但其训练和微调需要大量的计算和内存 \[kaplan2020scaling\]。缓解这一成本的一种自然方法是用结构化矩阵（如稀疏和低秩矩阵以及傅里叶变换）来替换稠密权重矩阵。然而，结构化矩阵（可以看作是稀疏性的一般形式）至今尚未得到广泛采用，主要由于两个挑战。(1) 在**端到端** (E2E) 训练设置中，它们表现出了不利的效率-质量权衡。模型*效率*是指这些结构化矩阵在现代硬件（例如 GPU）上的运行效率。模型*质量*（在任务上的性能）取决于它们的表达能力（例如，它们能否表示常用的变换，如编码领域特定知识的卷积或傅里叶/余弦变换）。现有的结构化矩阵要么在硬件上效率不高，要么表达能力不足。(2) 在预训练模型的**稠密到稀疏** (D2S) 微调设置中，对于大多数类别的结构化矩阵来说，一个长期存在的问题是缺乏易于处理的算法来近似稠密的预训练权重矩阵 \[pan2012structured\]。

稀疏矩阵在训练深度学习模型方面取得了进展（例如剪枝 \[han2015deep\]、彩票假设 \[frankle2018lottery\]），但大多数关于（逐元素）稀疏化的工作集中在减少训练或推理的 FLOPs 上，这并不一定等同于现代硬件（例如 GPU）上的 E2E 训练时间。事实上，大多数稀疏训练方法在挂钟时间上反而*减慢*了训练速度 \[gale2019state, hooker2020hardware\]。此外，稀疏矩阵无法表示常用的变换，如卷积和傅里叶变换。另一类结构化矩阵，如傅里叶、正弦/余弦、切比雪夫矩阵，用于偏微分方程 (PDE) 求解 \[trefethen2000spectral\] 和医学成像 \[hsieh2003computed\] 等专门领域。然而，它们很难用于 E2E 训练，因为只有这些结构化矩阵的特定实例具有快速的 GPU 实现（例如 FFT）。此外，它们的应用需要领域专业知识来手动挑选合适的变换。这些变换的推广（例如类 Toeplitz \[sindhwani2015structured\]、正交多项式变换 \[driscoll1997fast\]、低位移秩 \[kailath1979displacement\]、拟可分 \[eidelman1999new\]），尽管是可学习的，通常也缺乏在 GPU 上进行 E2E 训练的高效实现 \[thomas2018learning\]。另外，它们没有已知的易于处理的算法来近似给定的稠密矩阵 \[pan2012structured\]，这使得它们难以用于 D2S 微调。

**E2E 训练。** 解决结构化矩阵效率-质量权衡的技术挑战在于找到一种参数化方法，使其既在面向块的硬件（例如 GPU）上高效，又具有表达能力（例如，能够表示许多常用的变换）。为了应对这一挑战，我们提出了一类称为 Monarch 的矩阵（它们以帝王蝶命名），其参数化为两个块对角矩阵的乘积（直至排列）。这种参数化利用了 GPU 上优化的批量矩阵乘法 (BMM) 例程，与稠密矩阵乘法相比，实现了高达 2$\times$ 的加速 (5.1.1)。我们证明了 Monarch 矩阵类包含蝴蝶矩阵类 \[parker1995random,dao2019learning\]，后者可以在接近最优的运行时间和参数大小下表示任何低深度算术电路 \[dao2020kaleidoscope\]。Monarch 矩阵继承了这种表达能力，因此可以表示许多快速变换（例如傅里叶、正弦/余弦/切比雪夫变换、卷积）(2)。

![Monarch matrices unlock several ways to train sparse and dense models: end-to-end training a sparse (Monarch) model can be 2x faster than dense training thanks to its hardware efficiency; sparse-to-dense “reverse sparsification” can speed up training of large models such as GPT-2; and our dense-to-sparse Monarch projection algorithm can transfer knowledge from pretrained dense model to Monarch model and speed up BERT fine-tuning.](monarch-main.png)

**稀疏到稠密 (S2D) 训练，又称“反向稀疏化”。** Monarch 矩阵的硬件效率和表达能力解锁了一种训练稠密模型的新方法：在大部分时间里使用 Monarch 权重矩阵进行训练，然后过渡到稠密权重矩阵 (3)。这种技术可用于稀疏训练面临表示或优化困难 \[evci2019difficulty\] 或必须使用稠密模型的情况。其中一个这样的应用是在大型数据集上进行语言建模，这需要大量参数 \[kaplan2020scaling\] 来记忆文本模式 \[geva2020transformer\]。Monarch 矩阵可以作为快速的中间表示，以加速稠密模型的训练过程。

**D2S 微调。** 虽然从稀疏矩阵过渡到稠密矩阵很容易，但反向操作却具有挑战性。主要的技术困难在于*投影*问题：在结构化矩阵类中找到最接近给定稠密矩阵的矩阵。只有少数特定类别的结构化矩阵具有易于处理的投影解决方案，例如逐元素稀疏矩阵（幅度剪枝 \[tewarson1973sparse\]）、低秩矩阵（Eckart-Young 定理 \[eckart1936approximation\]）和正交矩阵（正交 Procrustes 问题 \[schonemann1966generalized\]）。对于更具表达能力的结构化矩阵类，投影仍然是一个长期存在的问题 \[pan2012structured\]。例如，\[desa2018two\] 证明了所有结构化矩阵（以算术电路的形式）都可以写成稀疏矩阵的乘积，这可以表示为蝴蝶矩阵的乘积 \[dao2020kaleidoscope\]。已经提出了许多启发式方法来投影到蝴蝶矩阵集合或稀疏矩阵乘积上，这些方法基于迭代的一阶优化 \[le2016flexible, dao2019learning, khalitov2021sparse\] 或交替最小化 \[lin2021deformable\]。然而，它们缺乏理论保证。相比之下，我们为我们的 Monarch 参数化推导了一种投影算法，并证明它能找到最优解 (1)。我们还推导出了一种算法，用于对 Monarch 矩阵乘积的矩阵进行因式分解 (3.4)。这些新算法使我们能够轻松地将预训练模型微调为具有 Monarch 权重矩阵的模型 (5.3)。

我们在上述三种设置中实证验证了我们的方法，结果表明，与基线相比，我们的 Monarch 矩阵参数化在广泛的领域（文本、图像、PDE、MRI）上实现了有利的效率-准确率权衡。

- 在 **E2E 稀疏训练**设置 (5.1) 中，我们的 Monarch 矩阵模型比稠密模型训练快 2$\times$，同时在基准任务（ImageNet 分类上的 ViT，Wikitext-103 语言建模上的 GPT-2）上实现了相同的准确率/困惑度。在依赖手工制作的快速变换的科学和医学任务（PDE 求解、MRI 重建）上，与基于特定领域傅里叶的方法相比，Monarch 在相同的训练速度下将误差降低了高达 40%。

- 在 **S2D 训练**设置 (5.2) 中，我们使用 Monarch 矩阵的“反向稀疏化”过程在大型 OpenWebText 数据集上加速了 GPT-2 预训练，与 NVIDIA 的优化实现 \[shoeybi2019megatron\] 相比速度提升了 2$\times$，且具有可比的上游和下游（文本分类）质量。当应用于 BERT 预训练时，我们的方法比创下 MLPerf \[mattson2020mlperf\] 1.1 纪录的 Nvidia 实现快 23%。

- 在 **D2S fine-tuning** 设置 (5.3) 中，我们展示了我们的 Monarch 投影算法加速 BERT fine-tuning 的概念验证。我们将预训练的 BERT 模型投影到 Monarch 矩阵模型，并在 GLUE 上进行 fine-tuning，参数量减少了 2$\times$，fine-tuning 速度提高了 1.7$\times$，且平均 GLUE 准确率与稠密模型相似。（Monarch 代码可在 <https://github.com/HazyResearch/monarch> 获取）

# 相关工作与背景

## 相关工作

**稀疏训练。** 稀疏训练是一个活跃的研究课题。在压缩模型这一方向上已有启发性的工作，例如神经网络剪枝和 lottery tickets \[han2015deep,han2015learning, frankle2018lottery\]。剪枝方法通常通过迭代重训练 \[han2015deep,han2015learning,sanh2020movement\] 或在运行时 \[NIPS2017_a51fb975,dong2017learning\] 消除神经元和连接。尽管 Monarch 和剪枝方法都旨在生成稀疏模型，但我们的不同之处在于强调*整体*效率，而剪枝主要关注推理效率并忽略了寻找较小模型的成本。Lottery tickets \[frankle2018lottery,frankle2019stabilizing,frankle2020linear\] 是从较大的稠密网络中导出的一组小型子网络，在收敛速度和潜在的泛化能力上优于其父网络。Monarch 大致可以看作是一类手动构建的 lottery tickets。

**结构化矩阵。** 结构化矩阵是指具有次二次（对于 $n \times n$ 维为 $o(n^2)$）参数量和运行时间的矩阵。例子包括稀疏矩阵和低秩矩阵，以及快速变换（傅里叶、切比雪夫、正弦/余弦、正交多项式）。它们通常用于替换深度学习模型的稠密权重矩阵，从而减少参数量和训练/推理 FLOPs。大类的结构化矩阵（例如，类 Toeplitz \[sindhwani2015structured\]、低位移秩 \[kailath1979displacement\]、拟可分 \[eidelman1999new\]）已被证明能够表示许多常用的快速变换。例如，\[desa2018two\] 表明简单的分治方案可以为一大类结构化矩阵导出快速算法。我们的工作建立在蝴蝶矩阵 \[parker1995random, dao2019learning\] 之上，这些矩阵已被证明具有表达能力，但在硬件上效率低下。Pixelated butterfly \[chen2021pixelated\] 试图使蝴蝶矩阵对硬件更友好，但代价是表达能力降低。此外，目前尚不清楚是否可以在不重新训练的情况下，直接将稠密预训练模型分解为具有蝴蝶权重矩阵的模型。

## 蝴蝶矩阵

我们的工作建立在近期关于*蝴蝶矩阵*的研究之上。\[dao2019learning\] 引入了蝴蝶矩阵的概念，将其作为置换块对角矩阵的特定乘积，其灵感来自于 Cooley-Tukey 快速傅里叶变换算法 \[cooley1965algorithm\]。它们编码了许多快速乘法算法的分治结构。\[dao2020kaleidoscope\] 表明所有结构化矩阵都可以写成此类蝴蝶矩阵的乘积，并且这种表示在多对数因子范围内具有最优的内存和运行时间复杂度。我们现在回顾这些定义（遵循 \[dao2020kaleidoscope\]）。

大小为 $k$（其中 $k$ 为偶数）的**蝴蝶因子**是以下形式的矩阵 $\begin{bmatrix}
            \mathbf{D}_1 & \mathbf{D}_2 \\ \mathbf{D}_3 & \mathbf{D}_4
        \end{bmatrix}$，其中每个 $\mathbf{D}_i$ 是一个 $\frac{k}{2} \times \frac{k}{2}$ 的对角矩阵。我们将此类矩阵称为 $\mathcal{B}\mathcal{F}^{(k,k)}$。

大小为 $n$ 且块大小为 $k$ 的**蝴蝶因子矩阵**是由 $\frac{n}{k}$ 个大小为 $k$ 的蝴蝶因子组成的块对角矩阵：$$\mathrm{diag}\left(\mathbf{B}_1, \mathbf{B}_2, \hdots, \mathbf{B}_\frac{n}{k} \right),$$ 其中 $\mathbf{B}_i \in \mathcal{B}\mathcal{F}^{(k,k)}$。我们将此类矩阵称为 $\mathcal{B}\mathcal{F}^{(n,k)}$。

最后，大小为 $n = 2^s$ 的**蝴蝶矩阵**是一个矩阵 $\mathbf{M}$，可以表示为蝴蝶因子矩阵的乘积：$$\mathbf{M}= \mathbf{B}_n \mathbf{B}_{n/2} \hdots \mathbf{B}_2,$$ 其中每个 $\mathbf{B}_i \in \mathcal{B}\mathcal{F}^{(n, i)}$。我们用 $\mathcal{B}^{(n)}$ 表示大小为 $n$ 的蝴蝶矩阵的集合。等价地，$\mathbf{M}$ 可以写成以下形式：$$\mathbf{M}= \mathbf{B}_n \begin{bmatrix}
            \mathbf{M}_1 & 0 \\
            0 & \mathbf{M}_2
         \end{bmatrix},$$ 其中 $\mathbf{B}_n \in \mathcal{B}\mathcal{F}^{(n,n)}$ 且 $\mathbf{M}_1, \mathbf{M}_2 \in \mathcal{B}^{(\frac{n}{2})}$。

\[dao2020kaleidoscope\] 进一步引入了*万花筒矩阵层级*：类 $\mathcal{B}\mathcal{B}^{*(n)}$ 是形式为 $\mathbf{M}_1\mathbf{M}_2^*$ 的矩阵集合，其中 $\mathbf{M}_1,\mathbf{M}_2 \in \mathcal{B}^{(n)}$；而类 $(\mathcal{B}\mathcal{B}^{*(n)})^w_e$ 是所有形式为 $\left(\prod\limits_{i=1}^{w} \mathbf{M}_i \right)[1{:}n, 1{:}n]$ 的矩阵集合，其中每个 $\mathbf{M}_i \in \mathcal{B}\mathcal{B}^{*(e\cdot n)}$。（$\mathbf{A}^*$ 表示 $\mathbf{A}$ 的共轭转置。）当大小 $n$ 从上下文中明确时，我们将省略上标 $^{(n)}$（即只写 $\mathcal{B},\mathcal{B}\mathcal{B}^*$ 等）。如 \[dao2020kaleidoscope\] 的定理 1 所示，万花筒层级可以以近乎最优的参数和运行时间表示任何结构化矩阵：如果 $\mathbf{M}$ 是一个 $n \times n$ 矩阵，使得将任意向量 $v$ 乘以 $\mathbf{M}$ 可以表示为深度为 $d$ 且总门数为 $s$ 的线性算术电路，那么 $\mathbf{M}\in (\mathcal{B}\mathcal{B}^{*(n)})^{O(d)}_{O(s/n)}$。

# Monarch：定义与算法

在 3.1 中，我们介绍了 *Monarch 矩阵*，并描述了它们与蝴蝶矩阵的关系。在 3.2 中，我们证明了 Monarch 矩阵类至少与蝴蝶矩阵类具有同等的表达能力，同时允许实际高效的表示。特别地，许多快速变换（例如，傅里叶、卷积）可以表示为 Monarch 矩阵或两个或四个 Monarch 矩阵的乘积 (2)。在 3.3 中，我们展示了如何投影到 Monarch 矩阵集合上。这使我们能够用 Monarch 矩阵易于处理地逼近给定矩阵（例如，稠密预训练权重矩阵），从而解锁新的应用（参见 5）。在 3.4 中，我们展示了如何恢复两个 Monarch 矩阵乘积这一更大类的各个因子。

## 方阵的 Monarch 参数化

受4步FFT算法 \[bailey1990ffts\] 的启发，我们提出了 Monarch 矩阵类，每个矩阵被参数化为两个块对角矩阵在置换下的乘积：


**定义 1**。令 $n = m^2$。一个 $n \times n$ 的 *Monarch matrix* 具有以下形式：$$\mathbf{M}= \mathbf{P}\mathbf{L}\mathbf{P}^\top \mathbf{R},$$ 其中 $\mathbf{L}$ 和 $\mathbf{R}$ 是块对角矩阵，每个矩阵有 $m$ 个大小为 $m \times m$ 的块，而 $\mathbf{P}$ 是将 $[x_1, \dots, x_n]$ 映射到 $[x_1, x_{1+m}, \dots, x_{1+(m-1)m}, x_2, x_{2+m}, \dots, \newline x_{2+(m-1)m}, \dots, x_{m}, x_{2m}, \dots, x_n]$ 的置换。


我们将其称为 *Monarch parametrization*。我们将所有能写成这种形式的矩阵类记为 $\mathcal{M}^{(n)}$（在上下文清晰时省略上标）。2 说明了这种参数化。

![Monarch matrices are parametrized as products of two block-diagonal matrices up to permutation, allowing efficient multiplication algorithm that leverages batch matrix multiply.](monarch-1.png)

我们现在为这种参数化提供更多直觉，并将其与 butterfly 矩阵联系起来。为了便于说明，假设 $\mathbf{B}\in \mathcal{B}^{(n)}$ 其中 $n$ 是 4 的幂。然后令 $\mathbf{L}'$ 为 $\mathbf{B}$ 的 butterfly 分解中前 $\tfrac{\log_2 n}{2}$ 个 butterfly 因子矩阵相乘得到的矩阵，并令 $\mathbf{R}$ 为最后 $\tfrac{\log_2 n}{2}$ 个 butterfly 因子矩阵相乘得到的矩阵。（我们在 4 中更严谨地详细说明这一点。）

矩阵 $\mathbf{R}$ 是具有 $m = \sqrt{n}$ 个稠密块的块对角矩阵，每个块的大小为 $m \times m$：$\mathbf{R}= \mathrm{diag}(\mathbf{R}_1, \dots, \mathbf{R}_{m}).$

矩阵 $\mathbf{L}'$ 由 $m \times m$ 个大小为 $m \times m$ 的块组成，其中每个块都是对角矩阵：$$\mathbf{L}' =
  \begin{bmatrix}
    \mathbf{D}_{11} & \hdots & \mathbf{D}_{1m} \\
    \vdots & \ddots & \vdots \\
    \mathbf{D}_{m1} & \hdots & \mathbf{D}_{mm} \\
  \end{bmatrix}.$$

矩阵 $\mathbf{L}'$ 在对行和列进行置换后，也可以写成与 $\mathbf{R}$ 结构相同的块对角矩阵。具体来说，令 $\mathbf{P}$ 为定义 1 中的置换。我们可以这样解释 $\mathbf{P}$：它将大小为 $n$ 的向量 $x$ 重塑为大小为 $m \times m$ 的矩阵，转置该矩阵，然后再转换回大小为 $n$ 的向量。注意 $\mathbf{P}= \mathbf{P}^\top$。然后我们可以写出 $$\mathbf{L}= \mathbf{P}\mathbf{L}' \mathbf{P}^\top, \quad \text{where } \mathbf{L}= \mathrm{diag}(\mathbf{L}_1, \dots, \mathbf{L}_{m}).$$ 因此，在对行和列进行置换的意义下，$\mathbf{L}'$ 也是具有 $m$ 个稠密块的块对角矩阵，每个块的大小为 $m \times m$。

因此我们可以写出 $\mathbf{B}= \mathbf{P}\mathbf{L}\mathbf{P}^\top \mathbf{R}$，其中 $\mathbf{L}$、$\mathbf{R}$ 和 $\mathbf{P}$ 如 1 中所示。所以，$\mathbf{B}\in \mathcal{B}^{(n)}$ 意味着 $\mathbf{B}\in \mathcal{M}^{(n)}$。

**Monarch 矩阵的乘积。** 另一类重要的矩阵（由于其表达能力，参见 2）是 $\mathcal{M}\mathcal{M}^*$ 类：可以写成 $\mathbf{M}_1\mathbf{M}_2^*$ 的矩阵，其中 $\mathbf{M}_1, \mathbf{M}_2 \in \mathcal{M}$。此外，$(\mathcal{M}\mathcal{M}^*)^2$ 表示可以写成 $\mathbf{M}_1\mathbf{M}_2$ 的矩阵类，其中 $\mathbf{M}_1, \mathbf{M}_2 \in \mathcal{M}\mathcal{M}^*$。

**扩展到长方阵。** 在实践中，我们还希望有一种方法来参数化长方形权重矩阵，并增加 Monarch 矩阵的参数数量以适应不同的应用（类似于低秩矩阵中的秩参数和稀疏矩阵中的非零元素数量）。我们做出了简单的选择，即增加 Monarch 参数化中块对角矩阵的块大小，并允许长方形块。更多细节见 9。

## 表达能力与效率

我们讨论 Monarch 矩阵及其乘积的表达能力（表示许多结构化变换的能力），以及它们的计算和内存效率。

### 表达能力

如 3.1 节所述，任何矩阵 $\mathbf{B}\in \mathcal{B}^{(n)}$ 都可以写成 Monarch butterfly 表示，只需将总共 $\log_2 n$ 个因子压缩成两个矩阵即可。因此，Monarch butterfly 表示严格比原始 butterfly 表示更一般（因为也存在属于 $\mathcal{M}^{(n)}$ 但不属于 $\mathcal{B}^{(n)}$ 的矩阵）。换句话说，对于给定大小 $n$，$\mathcal{M}\supset \mathcal{B}$；类似地 $\mathcal{M}\mathcal{M}^*\supset \mathcal{B}\mathcal{B}^*$。特别是，\[dao2020kaleidoscope\] 表明以下矩阵类包含在 $\mathcal{B}\mathcal{B}^*$ 中，这意味着它们也包含在 $\mathcal{M}\mathcal{M}^*$ 中：


**命题 2**。*矩阵类 $\mathcal{M}\mathcal{M}^*$ 可以表示卷积、Hadamard 变换、Toeplitz 矩阵 \[gray2006toeplitz\] 和 AFDF 矩阵 \[moczulski2015acdc\]。矩阵类 $(\mathcal{M}\mathcal{M}^*)^2$ 可以表示傅里叶变换、离散正弦和余弦变换 (DST/DCT)、$(HD)^3$ \[yu2016orthogonal\] 类、Fastfood \[le2013fastfood\] 和 ACDC 矩阵 \[moczulski2015acdc\]。*

### 效率

**参数。** 一个 Monarch 矩阵 $\mathbf{M}= \mathbf{P}\mathbf{L}\mathbf{P}^\top \mathbf{R}$ 由 $2 n \sqrt{n}$ 个参数描述：$\mathbf{L}, \mathbf{R}$ 都有 $\sqrt{n}$ 个大小为 $\sqrt{n} \times \sqrt{n}$ 的稠密块，每个总参数量为 $n\sqrt{n}$。置换 $\mathbf{P}$ 是*固定的*，因此不增加任何参数。**速度。** 要与 $\mathbf{M}$ 相乘，我们需要与块对角矩阵 $\mathbf{R}$ 相乘，进行置换，与块对角矩阵 $\mathbf{L}$ 相乘，最后进行置换。所有这四个步骤都可以高效实现。总 FLOPs 数为 $O(n \sqrt{n})$，这比 butterfly 矩阵的 $O(n \log n)$ 要多。然而，由于我们可以利用高效的块对角乘法（例如，批量矩阵乘法），Monarch 乘法易于实现且在实践中速度很快（比稠密乘法快 2 倍，参见 5）。

## 在 Monarch 矩阵集合 $\mathcal{M}$ 上的投影

给定我们的结构化矩阵类，一个自然的问题是*投影*问题：寻找一个距离给定稠密矩阵最近的 Monarch 矩阵。我们证明该问题存在解析最优解，并展示了如何高效地计算它。这使我们能够将稠密模型投影到 Monarch 模型，从而实现 D2S 微调 (5.3)。

我们将该问题形式化：对于给定矩阵 $\mathbf{A}$，寻找 $$\label{eq:projection_objective}
  \mathop{\mathrm{argmin}}\limits_{\mathbf{M}\in \mathcal{M}} \left\|{\mathbf{A}- \mathbf{M}}\right\|^2_F.$$

尽管该问题是非凸的（因为 $\mathbf{M}$ 被参数化为两个矩阵的乘积），但在定理 1 中我们证明存在一个解析解（完整证明见 10）。这类似于 Eckart-Young 定理，该定理确立了最优低秩近似可以通过 SVD 获得 \[eckart1936approximation\]。

**定理 1**. *给定一个 $n \times n$ 矩阵 $\mathbf{A}$，存在一个 $O(n^{5/2})$ 时间的算法，能够最优地求解投影问题 [eq:projection_objective]，并返回 Monarch 因子 $\mathbf{L}$ 和 $\mathbf{R}$。*

我们现在通过考察 Monarch 矩阵 $\mathbf{M}$ 的结构来推导该算法 ([alg:project])。

我们首先重写 Monarch 矩阵-向量乘法（即计算 $\mathbf{M}\mathbf{x}$）的步骤。主要思想是将大小为 $n = m^2$ 的输入向量 $\mathbf{x}$ 视为大小为 $m \times m$ 的二维张量。然后，Monarch 参数化 $\mathbf{M}= \mathbf{P}\mathbf{L}\mathbf{P}^\top \mathbf{R}$ 中的两个矩阵 $\mathbf{L}$ 和 $\mathbf{R}$ 对应于沿 $\mathbf{x}$ 的一个维度进行批量矩阵乘法，随后沿 $\mathbf{x}$ 的另一个维度进行批量矩阵乘法。因此，我们将 $\mathbf{x}$ 视为大小为 $m \times m$ 的二维张量，并将 $\mathbf{L}$ 和 $\mathbf{R}$ 各自视为大小为 $m \times m \times m$ 的三维张量。

将 $\mathbf{x}$ 乘以 Monarch 矩阵 $\mathbf{M}= \mathbf{P}\mathbf{L}\mathbf{P}^\top \mathbf{R}$ 的步骤：

1.  将 $\mathbf{R}$ 乘以 $\mathbf{x}$：$y_{kj} = \sum_i R_{kji} x_{ki}$，得到一个大小为 $m \times m$ 的二维张量输出 $\mathbf{y}$。

2.  将 $\mathbf{P}\mathbf{L}\mathbf{P}^\top$ 乘以 $\mathbf{y}$：$z_{\ell j} = \sum_k L_{j\ell k} y_{kj}$，得到一个大小为 $m \times m$ 的二维张量输出。

3.  将 $\mathbf{z}$ 重塑回大小为 $n$ 的向量，并返回它。

因此，我们可以将输出 $\mathbf{z}$ 写为 $z_{\ell j} = \sum_{k, i} L_{j\ell k} R_{kji} x_{ki}$。由于 $\mathbf{M}= \mathbf{P}\mathbf{L}\mathbf{P}^\top\mathbf{R}$，我们可以写为： $$\label{eq:b_einsum}
  M_{\ell jki} = L_{j\ell k} R_{kji}.$$ 注意，这里我们将 $\mathbf{M}$ 视为大小为 $m \times m \times m \times m$ 的四维张量。

当被视为四维张量时，矩阵 $\mathbf{M}$ 的结构变得显而易见，投影问题的解也很容易看出。让我们考察 [eq:b_einsum]：$M_{\ell jki} = L_{j\ell k} R_{kji}$。我们看到，这个重塑后的张量版本的 $\mathbf{M}$ 仅仅是 $m \cdot m$ 批秩为1的矩阵：我们在维度 $k$ 和 $j$ 上进行批处理，每批仅仅是一个秩为1的矩阵 $(\mathbf{p}_{jk}) (\mathbf{q}_{jk})^\top$，其中 $\mathbf{p}_{jk}, \mathbf{q}_{jk}$ 是长度为 $m$ 的向量。

因此，投影目标 ([eq:projection_objective]) 可以被分解为 $m \cdot m$ 个独立项之和，每一项对应于 $\mathbf{A}$ 的一个大小为 $m \times m$ 的块。由于 Monarch 矩阵的结构迫使每个块如上所述具有秩1，投影问题的解变得显而易见：给定矩阵 $\mathbf{A}$，将其重塑为大小为 $m \times m \times m \times m$ 的四维张量，并使用 SVD 对每批进行秩1近似，这（在重塑之后）产生了所需矩阵 $\mathbf{M}\in \mathcal{M}$ 的因子 $\mathbf{L}, \mathbf{R}$。（注意，如果 $\mathbf{A}\in \mathcal{M}$ 本身，该算法将恢复因子使得 $\mathbf{A}= \mathbf{P}\mathbf{L}\mathbf{P}^\top\mathbf{R}$。）



矩阵 $\mathbf{A}\in \mathbb{R}^{n \times n}$，其中 $n = m^2$。将 $\mathbf{A}$ 重塑为大小为 $m \times m \times m \times m$ 的四维张量 $\widetilde{\mathbf{A}}$，其中 $\widetilde{\mathbf{A}}_{\ell jki} = \mathbf{A}_{(\ell - 1) m + j, (k-1)m + i}$，对于 $\ell, j, k, i = 1, \dots, m$。  
令 $\widetilde{\mathbf{M}}_{jk} = \widetilde{\mathbf{A}}_{:, j, k, :}$，大小为 $m \times m$。使用 $\widetilde{\mathbf{A}}$ 的 SVD 计算 $\widetilde{\mathbf{M}}_{jk}$ 的最佳秩1近似 $\mathbf{u}_{jk} \mathbf{v}_{jk}^\top$。令 $\widetilde{\mathbf{R}}$ 为 $m \times m \times m$ 的张量，其中 $\widetilde{\mathbf{R}}_{kji} = (\mathbf{v}_{jk})_i$。令 $\widetilde{\mathbf{L}}$ 为 $m \times m \times m$ 的张量，其中 $\widetilde{\mathbf{L}}_{j \ell k} = (\mathbf{u}_{jk})_\ell$。将 $\widetilde{\mathbf{L}}$, $\widetilde{\mathbf{R}}$ 作为块对角矩阵 $\mathbf{L}, \mathbf{R}$ 返回（其中 $\mathbf{L},\mathbf{R}$ 的第 $b^{th}$ 块分别为 $\widetilde{\mathbf{L}}_{b,:,:}$, $\widetilde{\mathbf{R}}_{b,:,:}$）

## $\mathcal{M}\mathcal{M}^*$ 矩阵的分解

在上一节中，我们看到了如何投影到集合 $\mathcal{M}$ 上。如定理 2 所示，更广泛的类别 $\mathcal{M}\mathcal{M}^*$ 也包含许多重要的线性变换。在本节中，我们在温和的假设下提出一种算法，以计算给定矩阵 $\mathbf{M}\in \mathcal{M}\mathcal{M}^*$ 的 Monarch 分解。这使我们能够高效地存储和应用 $\mathbf{M}$。

![With the “reverse sparsification” process, Monarch matrices can speed up GPT-2 training by 2x.](monarch-2.png)

具体来说，观察到如果 $\mathbf{M}\in \mathcal{M}\mathcal{M}^*$，我们可以将 $\mathbf{M}$ 写成 $\mathbf{M}= (\mathbf{P}\mathbf{L}\mathbf{P}^\top \mathbf{R}) (\mathbf{R}'^* \mathbf{P}\mathbf{L}'^* \mathbf{P}^\top) = (\mathbf{P}\mathbf{L}_1\mathbf{P}^\top) \mathbf{R}(\mathbf{P}\mathbf{L}_2\mathbf{P}^\top)$，其中 $\mathbf{L}_1,\mathbf{L}_2,\mathbf{R}$ 为块对角矩阵，$\mathbf{P}$ 为 1 的置换矩阵。然后，如 2 中所述，在假设 3 下，我们可以计算这种分解中的 $\mathbf{L}_1,\mathbf{L}_2,\mathbf{R}$。（注意该分解不是唯一的。）

**假设 3**。假设 (1) $\mathbf{M}\in \mathcal{M}\mathcal{M}^*$ 是可逆的，并且 (2) $\mathbf{M}$ 可以写成 $(\mathbf{P}\mathbf{L}_1\mathbf{P}^\top)\mathbf{R}(\mathbf{P}\mathbf{L}_2\mathbf{P}^\top)$，其中 $\mathbf{R}$ 的块没有零元素。

**定理 2**。*给定满足假设 3 的 $n \times n$ 矩阵 $\mathbf{M}\in \mathcal{M}\mathcal{M}^*$，存在一种 $O(n^{5/2})$ 时间的算法来寻找其 Monarch 因子 $\mathbf{L}_1, \mathbf{R}, \mathbf{L}_2$。*

为了理解如何做到这一点，定义 $\tilde{\mathbf{M}} = \mathbf{P}^\top \mathbf{M}\mathbf{P}$ 并观察到 $\tilde{\mathbf{M}}\, = \, \mathbf{L}_1 (\mathbf{P}\mathbf{R}\mathbf{P}^\top) \mathbf{L}_2 \, =$ ${}\hspace{-0.35em}\small \left(\begin{array}{ccccc} \mathbf{A}_1 \\ & \mathbf{A}_2 \\ & & \ddots \\ & & & \mathbf{A}_{m} \end{array}\right)
\left(\begin{array}{cccc} \mathbf{D}_{11} & \mathbf{D}_{12} & \dots & \mathbf{D}_{1m} \\ \mathbf{D}_{21} & \mathbf{D}_{22} & \dots & \mathbf{D}_{2m} \\ \ddots & \ddots & \ddots & \ddots  \\ \mathbf{D}_{m1} & \mathbf{D}_{m2} & \dots & \mathbf{D}_{mm} \end{array}\right)
\left(\begin{array}{ccccc} \mathbf{C}_1 \\ & \mathbf{C}_2 \\ & & \ddots \\ & & & \mathbf{C}_{m} \end{array}\right)$

其中 $m = \sqrt{n}$，$\mathbf{A}_i$’s 和 $\mathbf{C}_j$’s 分别表示 $\mathbf{L}_1,\mathbf{L}_2$ 的 $m \times m$ 对角块，每个 $\mathbf{D}_{ij}$ 是一个 $m \times m$ 对角矩阵。如果我们将 $\widetilde{\mathbf{M}}$ 写成一个包含 $m \times m$ 个大小为 $m \times m$ 的块的块矩阵，那么我们会看到块 $\widetilde{\mathbf{M}}_{ij}$ 等于 $\mathbf{A}_i \mathbf{D}_{ij} \mathbf{C}_j$。注意，$\mathbf{M}$ 可逆当且仅当所有的 $\mathbf{A}_i$’s 和 $\mathbf{C}_j$’s 都可逆（因为如果其中任何一个是奇异的，那么 $\mathbf{L}_1$ 或 $\mathbf{L}_2$ 就是奇异的）。

因此，我们的目标是找到矩阵 $\hat{\mathbf{A}}_1,\dots,\hat{\mathbf{A}}_m, \hat{\mathbf{C}}_1,\dots,\hat{\mathbf{C}}_m$ 和*对角*矩阵 $\hat{\mathbf{D}}_{11},\dots,\hat{\mathbf{D}}_{mm}$，使得对于所有的 $i,j$，都有 $\widetilde{\mathbf{M}}_{ij} = \hat{\mathbf{A}}_i \hat{\mathbf{D}}_{ij} \hat{\mathbf{C}}_j$；这表示 $\mathbf{M}$ 的一个有效的 Monarch 分解。

为了提供关于如何做到这一点的直觉，让我们分析一个简单的情况，即所有的 $\mathbf{D}_{ij}$’s 都是单位矩阵。然后我们得到方程组 $\mathbf{A}_i \mathbf{C}_j = \widetilde{\mathbf{M}}_{ij}$。再次假设 $\mathbf{A}_i$’s 和 $\mathbf{C}_j$’s 是可逆的，因此每个 $\widetilde{\mathbf{M}}_{ij}$ 也是可逆的。假设我们设 $\hat{\mathbf{C}}_1 = \mathbf{I}$（单位矩阵）。然后我们可以立即读出对于所有的 $i$，$\hat{\mathbf{A}}_i = \widetilde{\mathbf{M}}_{i1}$。接着我们可以设对于所有的 $j$，$\hat{\mathbf{C}}_j = \hat{\mathbf{A}}_1^{-1}\widetilde{\mathbf{M}}_{1j}$。现在让我们检查这个策略是否给出了一个有效的分解，即对于所有的 $i,j$，$\widetilde{\mathbf{M}}_{ij} = \hat{\mathbf{A}}_i \hat{\mathbf{C}}_j$。我们有 $\hat{\mathbf{A}}_i \hat{\mathbf{C}}_j = \widetilde{\mathbf{M}}_{i1} \widetilde{\mathbf{M}}_{11}^{-1} \widetilde{\mathbf{M}}_{1j}$。回想一下，在“真实”的分解中我们有 $\widetilde{\mathbf{M}}_{ij} = \mathbf{A}_i \mathbf{C}_j$，这等于 $(\mathbf{A}_i \mathbf{C}_1) (\mathbf{A}_1 \mathbf{C}_1)^{-1} (\mathbf{A}_1 \mathbf{C}_j) = \mathbf{A}_i \mathbf{C}_j$，正如所愿。

在一般情况下，我们还必须处理对角矩阵 $\mathbf{D}_{ij}$。我们将不再能够自由地设 $\hat{\mathbf{C}}_1 = \mathbf{I}$。然而，一旦我们找到了 $\hat{\mathbf{C}}_1$ 的合适选择，我们就可以用它来找到所有的 $\hat{\mathbf{A}}_i$’s 和 $\hat{\mathbf{C}}_j$’s。我们可以通过*同时对角化*的思想找到这样的 $\hat{\mathbf{C}}_1$；由于篇幅原因，我们将我们算法的完整描述及其分析推迟到附录 10。

![With [alg:project] for our Monarch parameterization, we can convert a pretrained model into a model with Monarch weight matrices and speed up downstream fine-tuning.](monarch-3.png)

# 在模型训练中使用 Monarch 矩阵

我们可以在几种设置中使用我们的 Monarch 矩阵类来参数化深度学习模型的权重矩阵。

- 在 **E2E 稀疏训练** 设置中，我们用相同维度的 Monarch 矩阵替换基线模型的稠密权重矩阵，随机初始化它们，并照常训练。我们的大多数基线模型是 Transformer，我们用 Monarch 矩阵替换 Attention 块中的投影矩阵，以及前馈网络 (FFN) 块的权重。Monarch 参数化是可微的，我们依靠自动微分来使用一阶方法（如 Adam \[kingma2014adam\]）进行训练。

- 在 **S2D 训练** 设置中，我们首先用 Monarch 矩阵替换基线模型的稠密权重矩阵，然后训练稀疏模型大约 90% 的常规迭代次数。然后我们将 Monarch 矩阵转换为稠密矩阵（通过简单地将因子 $L$ 和 $R$ 连同置换相乘），并继续训练剩余 10% 的迭代次数。与稠密的端到端训练相比，我们训练了相同数量的迭代次数，但由于 Monarch 矩阵的硬件效率，前 90% 的迭代速度更快。

- 在 **D2S 微调** 设置中，我们从一个稠密的预训练模型（例如 BERT）开始，并使用 3.3 中的算法将稠密权重矩阵（例如，在 Attention 块和 FFN 块中）投影到 Monarch 矩阵集合上。然后，我们使用一阶方法在下游任务（例如 GLUE）上微调生成的模型。

我们通常根据参数预算（稠密模型的 25% – 50%）将块对角矩阵中的块数设置为 2 到 4 之间。

# 实验

我们通过实验验证了我们的方法，结果表明，与基线相比，我们的 Monarch 矩阵参数化在广泛的领域（文本、图像、PDE、MRI）和三种设置（E2E 训练、S2D 训练和 D2S 微调）下实现了令人满意的效率-精度权衡：

- 在 5.1.1 节中，在图像分类和语言建模基准测试上，例如 ImageNet 上的 ViT / MLP Mixer 和 Wikitext-103 上的 GPT-2，Monarch 的训练速度比密集模型快 2$\times$，同时达到相同的精度 / 困惑度。在 5.1.2 节中，在常使用特殊变换的科学和医疗领域，Monarch 在 PDE 求解上优于基于 Fourier 变换的方法，误差最多降低 40%，在 MRI 重建上获得最多高出 15% 的 pSNR 和 3.8% 的 SSIM。

- 在 5.1.2 节中，我们展示了在大型 OpenWebText 数据集上，反向稀疏化（在大部分时间使用 Monarch 权重矩阵进行训练，然后过渡到密集权重矩阵）与密集模型相比，将 GPT-2 模型的预训练速度提高了 2$\times$，且上游或下游质量没有损失。此外，即使与创下 MLPerf \[mattson2020mlperf\] 1.1 纪录的 Nvidia 实现相比，反向稀疏化也将 BERT 预训练速度提高了 23%。

- 在 5.3 节中，作为概念验证，我们证明了我们的 Monarch 近似算法可以提高预训练模型的微调效率。我们展示了将 BERT 压缩为 Monarch 矩阵模型在 GLUE 上的表现与微调后的密集模型相当，但参数减少了 2$\times$，微调速度提高了 1.7$\times$。

## 端到端训练

### 基准测试任务：图像分类、语言建模

我们在 [table:pretrain,table:gpt_pretrain] 中展示了，在 ViT、MLP-Mixer 和 GPT-2 中用 Monarch 矩阵替换密集矩阵可以使训练速度提高最多 2$\times$，而不会牺牲模型质量。

**设置。** 我们使用流行的视觉基准测试 ImageNet \[deng2009imagenet\]。我们选择近期流行的 Vision Transformer \[dosovitskiy2020image\] 和 MLP-Mixer \[tolstikhin2021mlp\] 作为代表性的基础密集模型。对于语言建模，我们在 WikiText-103 \[merity2016pointer\] 上评估 GPT-2 \[radford2019language\]。


|                    |               |             |        |       |     |     |     |
|:------------------:|:-------------:|:-----------:|:------:|:-----:|:---:|:---:|:---:|
|       模型         | ImageNet 准确率 |   加速比    | 参数量 | FLOPs |     |     |     |
|     Mixer-S/16     |     74.0      |     \-      | 18.5M  | 3.8G  |     |     |     |
| Monarch-Mixer-S/16 |     73.7      | 1.7$\times$ |  7.0M  | 1.5G  |     |     |     |
|     Mixer-B/16     |     77.7      |     \-      | 59.9M  | 12.6G |     |     |     |
| Monarch-Mixer-B/16 |     77.8      | 1.9$\times$ | 20.9M  | 5.0G  |     |     |     |
|      ViT-S/16      |     79.4      |     \-      | 48.8M  | 9.9G  |     |     |     |
|  Monarch-ViT-S/16  |     79.1      | 1.9$\times$ | 19.6M  | 3.9G  |     |     |     |
|      ViT-B/16      |     78.5      |     \-      | 86.6M  | 17.6G |     |     |     |
|  Monarch-ViT-B/16  |     78.9      | 2.0$\times$ | 33.0M  | 5.9G  |     |     |     |
|                    |               |             |        |       |     |     |     |

Monarch 矩阵与 ViT / MLP-Mixer 在 ImageNet 上的性能，包括参数量和 FLOPs。我们测量了 Top-1 准确率以及与相应密集模型相比的训练时间加速比。



|                      |      |             |        |       |
|:--------------------:|:----:|:-----------:|:------:|:-----:|
|                      | PPL  |   加速比    | 参数量 | FLOPs |
|     GPT-2-Small      | 20.6 |     \-      |  124M  | 106G  |
| Monarch-GPT-2-Small  | 20.7 | 1.8$\times$ |  72M   |  51G  |
|     GPT-2-Medium     | 20.9 |     \-      |  355M  | 361G  |
| Monarch-GPT-2-Medium | 20.3 | 2.0$\times$ |  165M  | 166G  |
|                      |      |             |        |       |

Monarch 矩阵与 GPT-2-Small/Medium 在 WikiText-103 上的性能，包括参数量（\#）和 FLOPs。Monarch 实现了相似的困惑度，但速度快了 2.0$\times$。

### PDE 求解与多线圈 MRI 重建

许多科学或医学成像任务依赖于特殊变换，例如 Fourier 变换。我们展示了用更具表达能力的 Monarch 矩阵替换固定的 Fourier 变换，可以在模型速度相当的情况下产生更高的模型质量（更低的重建误差）。

**使用 Monarch 神经算子求解 PDE。** 我们遵循 FNO \[li2020fourier\] 中的实验设置，并将基于 Monarch 的神经算子应用于求解 Navier–Stokes PDE 的任务。与基线 U-Nets \[ronneberger2015u\]、TF-Nets \[wang2020towards\]、ResNets \[he2016deep\] 和 FNOs \[li2020fourier\] 相比，基于 Monarch 的神经算子在各空间分辨率下将求解精度提高了最多 $40\%$（表 3）。

#### 非周期边界条件。

基于 Fourier 变换的传统谱方法在周期边界条件和强制项下效果最好。然而，实际感兴趣的 PDE 通常表现出非周期甚至未知的边界条件。Monarch 算子不受限于 Fourier 变换，因此仍能以出色的精度学习解算子。


|            |               |               |               |
|:----------:|:-------------:|:-------------:|:-------------:|
|   模型     | $v = 10^{-3}$ | $v = 10^{-4}$ | $v = 10^{-5}$ |
|   U-Net    |     0.025     |     0.205     |     0.198     |
|   TF-Net   |     0.023     |     0.225     |     0.227     |
|   ResNet   |     0.070     |     0.287     |     0.275     |
|    FNO     |     0.017     |     0.178     |     0.155     |
| Monarch-NO |   **0.010**   |   **0.145**   |   **0.136**   |
|            |               |               |               |

Navier-Stokes 上的基准测试（训练和测试的分辨率均固定为 64 × 64）。降低粘度系数 $\nu$ 会使动力学更加混沌。


**加速 MRI 重建。** 我们表征了基于 Monarch 的 FFT 操作在加速 MRI 重建中的效用，这是一项需要兼具结构化 Fourier 算子和去混叠特性的方法来恢复高质量图像的任务。在临床采集的 3D MRI SKM-TEA 数据集 \[desai2021skm\] 上，与零填充 SENSE 相比，Monarch-SENSE (mSENSE) 将图像质量提高了超过 1.5dB pSNR 和 2.5% SSIM，并且在数据受限的情况下，与 U-Net 基线相比提高了高达 4.4dB 和 3.8% SSIM。设置详情见 11.5。

#### 表达性 FFT。

根据定义，零填充 SENSE 中的标准 IFFT 无法对信号进行去混叠，导致重建图像中出现伪影。mSENSE 用可学习的 Monarch 矩阵替换了标准 SENSE 中的逆 FFT (IFFT) 操作。因此，mSENSE 保留了 Fourier 变换的结构，同时学习重新加权频率以抑制混叠伪影。在多种加速比下，mSENSE 在峰值信噪比 和结构相似性 上分别实现了最高 +1.5dB 和 2.5% 的提升（表 4）。

#### 数据效率。

虽然CNN在MRI重建任务中展现出了潜力，但训练这些网络需要大量标注数据以避免过拟合。然而，在实践中获取大型数据语料库十分困难。mSENSE可以在有限的监督样本下进行高效训练。在小样本设置下，mSENSE在pSNR上比U-Net高出+4.4dB（$\approx$<!-- -->15%），在SSIM上高出3.8%（表5）。


|      |        |                          |                          |                             |                             |
|:----:|:------:|:------------------------:|:------------------------:|:---------------------------:|:---------------------------:|
|      |        |  pSNR (dB) ($\uparrow$)  |                          |      SSIM ($\uparrow$)      |                             |
| 加速比 | 模型  |            E1            |            E2            |             E1              |             E2              |
|      | SENSE  |   32.8$\pm$<!-- -->0.2   |   35.4$\pm$<!-- -->0.2   |   0.871$\pm$<!-- -->0.003   |   0.865$\pm$<!-- -->0.003   |
|      | mSENSE | **34.3$\pm$<!-- -->0.2** | **36.6$\pm$<!-- -->0.2** | **0.886$\pm$<!-- -->0.002** | **0.882$\pm$<!-- -->0.003** |
|      | SENSE  |   30.9$\pm$<!-- -->0.2   |   33.5$\pm$<!-- -->0.2   |   0.819$\pm$<!-- -->0.004   |   0.795$\pm$<!-- -->0.004   |
|      | mSENSE | **32.3$\pm$<!-- -->0.2** | **34.6$\pm$<!-- -->0.2** | **0.843$\pm$<!-- -->0.003** | **0.820$\pm$<!-- -->0.004** |
|      | SENSE  |   30.1$\pm$<!-- -->0.2   |   32.8$\pm$<!-- -->0.2   |   0.789$\pm$<!-- -->0.004   |   0.753$\pm$<!-- -->0.005   |
|      | mSENSE | **31.2$\pm$<!-- -->0.2** | **33.5$\pm$<!-- -->0.2** | **0.812$\pm$<!-- -->0.003** | **0.767$\pm$<!-- -->0.005** |
|      |        |                          |                          |                             |                             |

传统方法与Monarch-SENSE (mSENSE)在多个加速比（Acc.）下的双回波（E1, E2）MRI重建的均值 $\pm$ 标准误。



|     |        |                          |                          |                             |                             |
|:---:|:------:|:------------------------:|:------------------------:|:---------------------------:|:---------------------------:|
|     |        |  pSNR (dB) ($\uparrow$)  |                          |      SSIM ($\uparrow$)      |                             |
| $N$ | 模型  |            E1            |            E2            |             E1              |             E2              |
| N/A | SENSE  |   32.8$\pm$<!-- -->0.2   |   35.4$\pm$<!-- -->0.2   |   0.871$\pm$<!-- -->0.003   |   0.865$\pm$<!-- -->0.003   |
|     | U-Net  |   29.4$\pm$<!-- -->0.2   |   34.4$\pm$<!-- -->0.3   |   0.848$\pm$<!-- -->0.004   |   0.857$\pm$<!-- -->0.004   |
|     | mSENSE | **33.8$\pm$<!-- -->0.2** | **36.0$\pm$<!-- -->0.2** | **0.886$\pm$<!-- -->0.003** | **0.867$\pm$<!-- -->0.003** |
|     | U-Net  |   29.9$\pm$<!-- -->0.3   |   35.1$\pm$<!-- -->0.3   |   0.858$\pm$<!-- -->0.003   |   0.871$\pm$<!-- -->0.003   |
|     | mSENSE | **34.0$\pm$<!-- -->0.2** | **36.4$\pm$<!-- -->0.2** | **0.883$\pm$<!-- -->0.002** | **0.877$\pm$<!-- -->0.003** |
|     | U-Net  |   31.0$\pm$<!-- -->0.3   |   35.2$\pm$<!-- -->0.3   |   0.866$\pm$<!-- -->0.003   |   0.867$\pm$<!-- -->0.004   |
|     | mSENSE | **33.9$\pm$<!-- -->0.2** | **36.5$\pm$<!-- -->0.2** | **0.882$\pm$<!-- -->0.002** | **0.878$\pm$<!-- -->0.003** |
|     | U-Net  |   31.4$\pm$<!-- -->0.3   |   35.6$\pm$<!-- -->0.2   |   0.877$\pm$<!-- -->0.002   |   0.870$\pm$<!-- -->0.003   |
|     | mSENSE | **33.9$\pm$<!-- -->0.2** | **36.5$\pm$<!-- -->0.2** | **0.881$\pm$<!-- -->0.002** | **0.877$\pm$<!-- -->0.003** |
|     |        |                          |                          |                             |                             |

训练样本数量（$N$）对2倍加速下双回波MRI重建的影响。

## 稀疏到稠密训练（反向稀疏化）

#### GPT-2 预训练。

在大型 OpenWebtext 数据集 [Gokaslan2019OpenWeb] 上，我们使用 Monarch 权重矩阵对 GPT-2 模型进行 90% 的训练迭代，然后放宽对权重矩阵的约束，并在剩余 10% 的迭代中将它们作为稠密矩阵进行训练。我们将此技术称为“反向稀疏化”。以前的稀疏训练技术通常不会加速训练，而我们硬件高效的 Monarch 矩阵则可以。因此，我们可以将它们作为中间步骤，以减少 2$\times$ 的时间预训练大型语言模型 (GPT-2)。我们还评估了其在 [eval-harness] 的零样本生成和 [zhao2021calibrate] 的分类任务上的下游质量，取得了与稠密对应模型相当的性能 (6)。


|                |                   |           |                          |
|:--------------:|:-----------------:|:---------:|:------------------------:|
|     模型      | OpenWebText (ppl) |  加速比  | 分类 (平均准确率) |
|     GPT-2m     |       18.0        |    \-     |           38.9           |
| Monarch-GPT-2m |       18.0        | 2$\times$ |           38.8           |
|                |                   |           |                          |

使用 Monarch 反向稀疏化和传统稠密训练的 GPT-2-medium 在文本分类基准上的性能（准确率）。


在 5 中，我们展示了稠密 GPT-2 模型以及 Monarch GPT-2 模型的训练时间。在 90% 的时间训练 Monarch 模型后，在最后 10% 的训练步骤中，通过过渡到稠密权重矩阵，该模型能够达到另一个从头开始使用稠密权重矩阵训练的模型的相同性能。通过在 90% 的时间内使用 Monarch 矩阵进行训练，我们将总训练时间减少了 2$\times$。

#### BERT 预训练。

在 Wikipedia + BookCorpus 数据集 [zhu2015aligning] 上，我们使用 Monarch 权重矩阵对 BERT-large 模型进行 70% 的时间训练，并在剩余 30% 的时间过渡到稠密权重矩阵，这产生了与传统稠密训练相同的预训练损失。在 7 中，我们将总训练时间与几个基线实现进行了比较：广泛使用的 HuggingFace 实现 [wolf-etal-2020-transformers]，Megatron 的更优化实现 [shoeybi2019megatron]，以及我们所知的 Nvidia 用于创下 MLPerf 1.1 训练速度记录的最优化实现。我们的方法比 HuggingFace 快 3.5 倍，比 Nvidia 的 MLPerf 1.1 实现快 23%（我们的结果不是官方的 MLPerf 提交。我们根据标准的 BERT 训练方案 [devlin2018bert] 训练了 BERT 的第一阶段（序列长度 128）和第二阶段（序列长度 512），而 MLPerf 仅测量第二阶段的训练时间。）。实验细节见 11.4。


|        实现         | 训练时间 (h) |
|:-----------------------------:|:-----------------:|
|          HuggingFace          |       84.5        |
|           MegaTron            |       52.5        |
|       Nvidia MLPerf 1.1       |       30.2        |
| Nvidia MLPerf 1.1 + DeepSpeed |       29.3        |
|        Monarch (ours)         |     **23.8**      |

在 8 块 A100-40GB GPU (DGX A100) 上，使用 Monarch 反向稀疏化和传统稠密训练的 BERT-large 的总训练时间。训练包括两个阶段，第一阶段序列长度为 128，第二阶段序列长度为 512。Monarch 训练比 HuggingFace 快 3.5 倍，比 Nvidia 的 MLPerf 1.1 实现快 23%。

## 密集到稀疏的微调

![Time required (in A100 GPU hours) to reach the same perplexity (18.0) for GPT-2-small on OpenWebText. With “reverse sparsification”, Monarch can speed up GPT-2 training by 2×.](rv-bar-temp.png)

我们展示了我们的 Monarch 近似算法使我们能够高效地使用预训练模型，例如加速在 GLUE 上的 BERT 微调。

#### BERT 微调。

我们获取 BERT 预训练权重，用 Monarch 矩阵近似它们，并在 9 个 GLUE 任务上微调生成的模型。表 8 中的结果表明，我们获得了一个与密集 BERT 模型质量相似的 Monarch 微调模型，但微调速度快了 1.7$\times$。这作为一个概念验证，我们预计如果应用额外的模型压缩技术（例如，量化、核融合），将会获得进一步的加速。

|                    |            |             |        |       |     |     |     |
|:------------------:|:----------:|:-----------:|:------:|:-----:|:---:|:---:|:---:|
|       模型         | GLUE (平均)|    加速     | 参数量 | FLOPs |     |     |     |
|     BERT-base      |    78.6    |     \-      |  109M  | 11.2G |     |     |     |
| Monarch-BERT-base  |    78.3    | 1.5$\times$ |  55M   | 6.2G  |     |     |     |
|     BERT-large     |    80.4    |     \-      |  335M  | 39.5G |     |     |     |
| Monarch-BERT-large |    79.6    | 1.7$\times$ |  144M  | 14.6G |     |     |     |
|                    |            |             |        |       |     |     |     |

Monarch 矩阵在 GLUE 上微调 BERT 的性能。

# 结论

我们提出了 Monarch，一种新颖的矩阵参数化方法，它继承了蝴蝶矩阵的表达能力，因此可以表示许多快速变换。我们的参数化利用了 GPU 上优化的批量矩阵乘法例程，与密集矩阵乘法相比，实现了高达 2$\times$ 的加速。我们推导出了一种高效的算法，用于将任意密集矩阵投影到 Monarch 因子的集合上。我们的算法使我们能够轻松地将预训练模型微调为具有 Monarch 权重矩阵的模型。因此，Monarch 矩阵为大型神经网络的更快端到端训练、稀疏到密集的训练以及密集到稀疏的微调解锁了新途径。通过使结构化矩阵变得实用，我们的工作是在将稀疏模型应用于广泛的机器学习应用（包括科学和医学）方面解锁巨大性能提升的第一步。我们期待这项工作能够激发更多未来的研究，在计算资源有限的情况下推进用于跨学科研究的机器学习模型。

# 致谢

我们感谢 Laurel Orr、Xun Huang、Trevor Gale、Jian Zhang、Victor Bittorf、Sarah Hooper、Neel Guha 和 Michael Zhang 对本文早期草稿提供的有益讨论和反馈。

我们衷心感谢 NIH 在 No. U54EB020405 (Mobilize) 项目下、NSF 在 Nos. CCF1763315 (Beyond Sparsity)、CCF1563078 (Volume to Velocity) 和 1937301 (RTML) 项目下、ARL 在 No. W911NF-21-2-0251 (Interactive Human-AI Teaming) 项目下、ONR 在 No. N000141712266 (Unifying Weak Supervision) 项目下、ONR N00014-20-1-2480: Understanding and Applying Non-Euclidean Geometry in Machine Learning 以及 N000142012275 (NEPTUNE) 项目下提供的支持；同时感谢 NXP、Xilinx、LETI-CEA、Intel、IBM、Microsoft、NEC、Toshiba、TSMC、ARM、Hitachi、BASF、Accenture、Ericsson、Qualcomm、Analog Devices、Google Cloud、Salesforce、Total、HAI-GCP Cloud Credits for Research 项目、Stanford Data Science Initiative (SDSI) 以及 Stanford DAWN 项目的成员：Facebook、Google 和 VMWare 的支持。美国政府被授权出于政府目的复制和分发重印本，无论其上是否有任何版权标记。本材料中表达的任何意见、发现、结论或建议均属于作者，不一定反映 NIH、ONR 或美国政府的观点、政策或认可，无论是明示还是暗示的。

# 扩展的相关工作

在本节中，我们扩展了正文中引用的相关工作，并对其进行了详细讨论。

#### 稀疏训练。

我们的工作与神经网络剪枝松散相关。通过迭代地消除神经元和连接，剪枝在压缩复杂模型方面取得了巨大成功。[han2015deep,han2015learning] 提出了两种简单但有效的算法，将模型压缩高达 49 倍并保持相当的准确率。[li2016pruning] 采用滤波器剪枝将运行卷积模型的成本降低高达 38 $\%$，[NIPS2017_a51fb975] 在运行时对网络进行剪枝，从而保留了完整模型的灵活性。[dong2017learning] 以逐层的方式在局部对网络进行剪枝。[sanh2020movement] 使用确定性的一阶信息进行剪枝，这更能适应预训练模型权重。[lagunas2021block] 在微调期间使用块稀疏模式对 Transformer 模型进行剪枝，这导致了真实的硬件加速，同时保持了准确率。[zhu2017prune] 发现大型剪枝稀疏网络在相同的计算和内存占用下始终优于小型密集网络。尽管我们的方法和所有剪枝方法都旨在生成稀疏模型，但我们的不同之处在于强调整体效率，而剪枝主要关注推理效率，并忽略了寻找较小模型的成本。

最近有更多关于稀疏方法的工作集中在加速训练而不仅仅是推理上，例如 SNFS [dettmers2019sparse]、RigL [dettmers2019sparse] 和 Top-KAST [jayakumar2021top]。这些方法通常关注 FLOP 计数，这可能与现代硬件（例如，GPU）上的实际运行时间关联性不佳。块稀疏是另一种利用 GPU 面向块特性的方法 [gray2017gpu, child2019generating, guo2020accelerating]。稀疏模型也被发现可用于改进密集模型的训练过程。例如，稀疏性可用于正则化密集模型以提高准确率 [han2016dsd]，或者在稀疏和密集训练之间交替以简化部署 [peste2021ac]。我们的稀疏到密集的逆向稀疏化则侧重于加速密集训练，其中稀疏模型用于提高效率而不是正则化。

此外，我们工作中提出的模型可以粗略地看作是一类手动构建的彩票。彩票 [frankle2018lottery] 是从较大的密集网络中导出的一组小型子网络，它们在收敛速度和潜在的泛化能力上优于其父网络。大量研究从经验和理论上分析了这些彩票：[morcos2019one] 提出对所有视觉基准测试使用一张广义彩票，并获得了与专用彩票相当的结果；[frankle2019stabilizing] 通过迭代剪枝提高了彩票的稳定性；[frankle2020linear] 发现子网络只有在训练期间对 SGD 噪声稳定时才能达到完全准确率；[orseau2020logarithmic] 为最优子网络存在所需的参数数量提供了对数上限；[pensia2020optimal] 提出了一种通过解决子集和问题来构建彩票的方法，这是强彩票假设的构造性证明。此外，后续工作 [liu2020finding, wang2020picking, tanaka2020pruning] 表明我们可以在没有任何训练标签的情况下找到彩票。

#### 结构化矩阵与蝴蝶矩阵

结构化矩阵是指具有渐近快速矩阵-向量乘法算法（$o(n^2)$ 时间复杂度）且参数较少（$o(n^2)$ 空间复杂度）的矩阵。常见的例子包括稀疏矩阵与低秩矩阵，以及快速变换，如傅里叶变换、切比雪夫变换、勒让德变换，以及更一般的正交多项式变换。这些变换已被广泛应用于数据预处理（例如语音处理中的 DFT [jurafsky2014speech]）和核近似 [le2013fastfood,yu2016orthogonal]。这些变换的许多推广已被用于机器学习中以替代稠密权重矩阵 [sindhwani2015structured,thomas2018learning,gu2020hippo]。[desa2018two] 表明，任何结构化矩阵（以算术电路的形式）都可以写成稀疏矩阵的乘积，而 [dao2020kaleidoscope] 表明蝴蝶矩阵的乘积可以在运行时间和内存方面几乎最优地表示这些结构化矩阵。蝴蝶矩阵类 [parker1995random] 也已被用于核模型 [munkhoeva2018quadrature, choromanski2019unifying] 和深度学习模型 [vahid2020butterfly,lin2021deformable, ailon2021sparse]。

#### 用于偏微分方程的神经算子

深度学习已在微分方程和科学计算领域得到应用 [rackauckas2020universal]，针对预测和控制问题开发了相关方法 [kidger2020neural,massaroli2021differentiable]，以及数值格式的加速 [poli2020hypersolvers,jolicoeur2021gotta]。专门针对*偏微分方程*（PDEs）的方法旨在学习解算子 [raissi2019physics,fan2020solving,li2020fourier] 和混合求解器 [kochkov2021machine]，主要在经典流体力学上进行评估。

这些方法的前景在于，以初始训练过程为代价，提供比针对特定问题调优的适当数值方法更准确且更快的解，从而可用于实时预测或在更大的反馈回路中。尽管如此，神经算子的最优设计仍是一个开放问题，大多数方法依赖于快速傅里叶变换（FFT）或标准稠密神经架构。相反，基于 Monarch 的神经算子能够近似所有快速变换，从而允许在给定的 PDE 问题上自动优化以找到合适的变换。

#### MRI

加速多线圈 MRI 是减少长扫描时间并使某些扫描类型可行的关键机制。在多线圈 MRI 中，数据在空间傅里叶域（又称 *k-space*）中通过多个线圈（传感器）采集。为减少扫描时间，该数据的采样率低于恢复底层信号所需的速率（即奈奎斯特速率），从而导致信号混叠（见附录 11.5）。在这些设置中，直接应用逆快速傅里叶变换（FFT）无法抑制混叠伪影。

经典 MRI 重建方法通过利用多个线圈间的共享信息和强解析先验来正则化图像恢复目标，从而补充 FFT。基于 SENSE 的方法跨多个线圈联合去混叠图像，并根据每个线圈的空间灵敏度分布对最终图像进行重新加权 [pruessmann1999sense]。压缩感知在变换域（如傅里叶、小波）中促进图像稀疏性，同时强制重建图像的傅里叶变换与观测测量之间的数据一致性 [lustig2007sparse]。低秩方法在数据中缓慢变化的维度或局部块上强制低秩结构 [ong2016beyond,ravishankar2017low,haldar2013low]。此外，基于 GRAPPA 的技术优化核以直接插值缺失的 k-space 样本，从而促进傅里叶域中的平滑性 [griswold2002generalized]。尽管这些方法有效，但它们重建时间长，需要显式的解析先验，并且需要仔细的超参数微调。

CNN 已展现出作为经典 MRI 重建方法的快速推理、可学习替代方案的前景 [knoll2020deep]。在监督学习中，全卷积网络（如 U-Net [ronneberger2015u] 或展开网络 [sandino2020compressed,hammernik2018learning]）学习零填充图像与全采样真实图像之间的映射。然而，监督方法需要大量全采样（标注）数据集，并且对由于患者、硬件和序列异质性引起的分布漂移敏感 [darestani2021measuring]。为减少对标注数据的依赖，无监督方法使用了生成对抗网络 [cole2020unsupervised, mardani2018deep]、自监督学习 [yaman2020self]、字典学习 [lahiri2021blind] 和未训练网络 [darestani2021accelerated]。尽管这些方法具有标签效率，但其性能仍不及监督方法，且对分布偏移同样敏感。最近，一系列半监督重建方法展示了标签效率和对物理驱动扰动的鲁棒性，如信噪比变化或患者运动 [desai2021noise2recon, desai2021vortex]。然而，这些方法需要大量未标注数据，在少样本设置中难以收集。因此，尽管这些模型在受控环境中取得了成功，但其前瞻性临床部署仍受到阻碍 [chaudhari2020prospective]。

在我们的工作中，我们提出了一种具有单个 FFT 初始化的分解 Monarch 矩阵的模型。这样的矩阵可以同时提供像 FFT 这样的简单线性变换的好处，以及去除由欠采样 k-space 引起的混叠伪影的可学习机制。较小的可学习参数集可以在数据受限的设置中减少过拟合，同时保留傅里叶矩阵的变换结构。因此，我们的方法可以解释为解析约束的经典方法与数据依赖的 CNN 之间的混合。

# 符号说明

在本文中，我们使用小写字母表示标量（如 $k$），小写粗体表示向量（如 $\mathbf{v}$），大写粗体表示矩阵（如 $\mathbf{A}$）。

$\mathbf{I}$ 表示单位矩阵。我们使用 $\mathbf{A}^\top$ 表示矩阵的转置，使用 $\mathbf{A}^*$ 表示矩阵的共轭转置。本文中的所有结果适用于实数 $\mathbb{R}$ 或复数 $\mathbb{C}$ 上的矩阵；当所考虑的域可以是其中任意一个时，我们用 $\mathbb{F}$ 表示。

除特别说明外，本文全文使用 1 索引。

# 一般 Monarch 矩阵参数化

在第 9.1 节中，我们定义了不同"块大小"（即不一定是 $\sqrt{n}$）的方阵 Monarch 矩阵的参数化，并证明了它们的一些基本性质。在第 9.2 节中，我们进一步扩展以定义矩形 Monarch 矩阵，并证明它们的一些基本性质。

注意：在本节中，为了符号方便，我们使用 0 索引而非 1 索引。

## 一般方阵

### 参数化

在本节中，我们为方阵定义了一种更通用的 Monarch 参数化方法，允许使用不同的“块大小”。与 1 类似，该参数化涉及一个置换块对角矩阵与另一个块对角矩阵的乘积；区别在于我们现在允许矩阵 $\mathbf{L}$ 和 $\mathbf{R}$ 具有不同大小的对角块。因此，应用于 $\mathbf{L}$ 的置换（将其转换为每个块矩阵均为对角矩阵的块矩阵）也将相应地有所不同。

首先，在 4 中，我们为一类块对角矩阵定义了符号表示。


**定义 4**（类 $\mathcal{B}\mathcal{D}^{(b, n)}$）。设 $b \in (1, n)$ 为整除 $n$ 的整数。对于 $0\le i< \frac {n}{b}$，设 $\mathbf{R}_{i}\in\mathbb{F}^{b \times b }$ 为一个 $b \times b$ 的“块”矩阵。然后按如下方式定义具有*块大小* $b$ 的矩阵 $\mathbf{R}$： $$\label{eq:def-R}
  \mathbf{R}= \mathrm{diag}\left(\mathbf{R}_0, \dots, \mathbf{R}_{\frac {n}{b}-1}\right).$$


（注意 $\mathbf{R}$ 中可能的非零值数量为 $\frac {n}{b}\cdot b^2=nb$。）我们将所有可表示为这种形式的矩阵 $\mathbf{R}$ 构成的类记为 $\mathcal{B}\mathcal{D}^{(b, n)}$。注意，该类在（共轭）转置下是封闭的，并且包含单位矩阵。

接下来，在 5 中，我们为*块*为对角矩阵的一类块矩阵定义符号表示。


**定义 5**（类 $\mathcal{D}\mathcal{B}^{(b,n)}$）。设 $b \in (1, n)$ 为整除 $n$ 的整数。对于 $0 \le i, j < b$，设 $\mathbf{D}_{i,j}\in\mathbb{F}^{b\times b}$ 为一个 $b \times b$ 的对角矩阵。然后设 $\mathbf{L}$ 为具有以下形式的 $n \times n$ 矩阵： $$\label{eq:def-L}
    \mathbf{L}=
        \begin{bmatrix}
            \mathbf{D}_{0,0} & \dots & \mathbf{D}_{0,\frac{n}{b} -1} \\
            \vdots & \ddots & \vdots \\
            \mathbf{D}_{\frac{n}{b} -1,0} & \dots & \mathbf{D}_{\frac{n}{b} -1,\frac{n}{b} -1}
        \end{bmatrix}$$


（注意 $\mathbf{L}$ 中可能的非零值数量为 $\left( {\frac nb}\right)^2\cdot b=\frac{n^2}b$。）我们将所有可表示为这种形式的矩阵 $\mathbf{L}$ 构成的类记为 $\mathcal{D}\mathcal{B}^{(b, n)}$。注意，该类在（共轭）转置下是封闭的，并且包含单位矩阵。正如我们在 9.1.2 中所展示的，$\mathbf{L}$ 可以写成一个具有 $b$ 个大小为 $\tfrac{n}{b} \times \tfrac{n}{b}$ 块的块对角矩阵（即 $\mathcal{B}\mathcal{D}^{(\frac{n}{b}, \, n)}$ 中的矩阵），并在左侧和右侧乘以适当的置换矩阵。我们将所有可表示为这种形式的矩阵 $\mathbf{L}$ 构成的类记为 $\mathcal{D}\mathcal{B}^{(b, n)}$。注意，该类在（共轭）转置下是封闭的。正如我们在 9.1.2 中所展示的，$\mathbf{L}$ 可以写成一个具有 $b$ 个大小为 $\tfrac{n}{b} \times \tfrac{n}{b}$ 块的块对角矩阵（即 $\mathcal{B}\mathcal{D}^{(\frac{n}{b}, \, n)}$ 中的矩阵），并在左侧和右侧乘以适当的置换矩阵。

使用这两个定义，我们定义具有给定块大小的 Monarch 矩阵类。


**定义 6**（类 $\mathcal{M}^{(b,n)}$）。设 $b \in (1, n)$ 为整除 $n$ 的整数。大小为 $n \times n$ 且“块大小为 $b$”的 *Monarch 矩阵*是以下形式的矩阵： $$\label{eq:Monarch-general}
    \mathbf{M}= \mathbf{L}\mathbf{R}$$ 其中 $\mathbf{L}\in \mathcal{D}\mathcal{B}^{(b,n)}$ 且 $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b,n)}$。


我们将所有可表示为这种形式的矩阵 $\mathbf{M}$ 构成的类记为 $\mathcal{M}^{(b, n)}$。观察到当 $b = \sqrt{n}$ 时，这正是 1 中的矩阵类 $\mathcal{M}^{(n)}$。（换句话说，$\mathcal{M}^{(n)}$ 是 $\mathcal{M}^{(\sqrt{n}, n)}$ 的简写。）注意，$\mathcal{M}^{(b,n)}$ 中的矩阵由 $\frac{n^2}{b} + nb$ 个参数表示。

我们指出，对于所有整除 $n$ 的块大小 $b \in (1, n)$，都有 $\mathcal{M}^{(b,n)} \supset \mathcal{B}^{(n)}$。

基于 22，我们定义类 $\mathcal{M}\mathcal{M}^{*(b,n)}$ 和 $\mathcal{M}^*\mathcal{M}^{(b,n)}$：


**定义 7**（类 $\mathcal{M}\mathcal{M}^{*(b,n)}$、$\mathcal{M}^*\mathcal{M}^{(b,n)}$）。设 $b \in (1, n)$ 为整除 $n$ 的整数，并假设 $\mathbf{M}_1, \mathbf{M}_2 \in \mathcal{M}^{(b,n)}$。我们将 $\mathcal{M}\mathcal{M}^{*(b,n)}$ 定义为所有可表示为 $\mathbf{M}= \mathbf{M}_1 \mathbf{M}_2^*$ 形式的矩阵 $\mathbf{M}$ 构成的类。我们将 $\mathcal{M}^*\mathcal{M}^{(b,n)}$ 定义为所有可表示为 $\mathbf{M}= \mathbf{M}_1^* \mathbf{M}_2$ 形式的矩阵 $\mathbf{M}$ 构成的类。


观察到当 $b = \sqrt{n}$ 时，$\mathcal{M}\mathcal{M}^{*(b,n)}$ 正是 3 中定义的矩阵类 $\mathcal{M}\mathcal{M}^{*(n)}$。注意，$\mathcal{M}\mathcal{M}^{*(b,n)}$ 或 $\mathcal{M}^*\mathcal{M}^{(b,n)}$ 中的矩阵由 $2\frac{n^2}{b} + 2nb$ 个参数表示。

最后，基于 \[dao2020kaleidoscope\] 的万花筒层级结构，我们定义以下“Monarch 层级”：


**定义 8**（类 $(\mathcal{M}\mathcal{M}^{*(b,n)})^w_e$）。设 $b \in (1, n)$ 为整除 $n$ 的整数。我们将矩阵类 $(\mathcal{M}\mathcal{M}^{*(b,n)})^w_e$ 定义为所有可以表示为以下形式的矩阵 $\mathbf{M}$ 的集合： $$\label{eq:mm-hierarchy}
    \mathbf{M}= \left(\mathlarger{\prod\limits_{i=1}^{w}} \mathbf{M}_i\right)[1:n, 1:n]$$ 其中每个 $\mathbf{M}_i \in \mathcal{M}\mathcal{M}^{*(b,e\cdot n)}$。


注意，$(\mathcal{M}\mathcal{M}^{*(b,n)})^w_e$ 中的矩阵由 $2w\frac{e^2n^2}{b} + 2wenb$ 个参数表示。

### 属性

这里我们展示上面定义的矩阵类的一些属性。我们首先展示定义这些类的一些基本等价方法。然后我们展示 (3) $\mathcal{D}\mathcal{B}^{(b, n)}$ 中的矩阵是置换块对角矩阵；具体来说，它们可以通过应用适当的置换转换为 $\mathcal{B}\mathcal{D}^{(\frac{n}{b}, n)}$ 中的矩阵。最后，我们陈述一般“Monarch 层次结构”的表达性结果，该结果源自 \[dao2020kaleidoscope\] 的定理 1。

首先，我们定义一类置换。设 $1\le b\le n$ 为整数且 $b$ 整除 $n$。我们将需要以“块形式”表示每个索引 $0\le i<n$。更具体地说：

**定义 9**。设 $i \ge 0$, $b \ge 1$ 为整数。然后定义 $$i_0=i\text{ mod }{b},$$ 和 $$i_1=\left \lfloor \frac ib \right \rfloor.$$ 我们使用记号 $i\equiv\left( {{i_1},{i_0}}\right)_{{b}}$ 来表示上述表示。特别地，如果 $i\equiv(i_1,i_0)_{b}$，那么我们有 $$i = i_1 \cdot b + i_0$$

使用此记号，我们定义以下置换类：

**定义 10**。设 $b \in [1, n]$ 为整除 $n$ 的整数。设 $i\equiv\left( {{i_1},{i_0}}\right)_{{b}}$。定义 $$\label{eq:sigma_b-def}
        \sigma_{(b,n)}(i) = i_0\cdot\frac{n}{b} + i_1.$$ 也就是说，$\sigma_{(b,n)}(i)\equiv \left( {{i_0},{i_1}}\right)_{{\frac {n}{b}}}$。设 $\mathbf{P}_{(b,n)}$ 表示由置换 $\sigma_{(b,n)}$ 定义的 $n \times n$ 置换矩阵。

直观上，$\mathbf{P}_{(b,n)}$ 可以解释为将长度为 $n$ 的向量按行优先顺序重塑为 $b \times \tfrac{n}{b}$ 矩阵，将结果转置，然后（再次按行优先顺序）将其展平回向量。

现在，我们将 4 中的公式等价地重述为：

**命题 11**。* 矩阵 $\mathbf{R}$ 满足  （即 $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b,n)}$）当且仅当以下条件对任意 $0\le i,j< n$ 成立。设 $i\equiv\left( {{i_1},{i_0}}\right)_{{b}}$ 且 $j\equiv\left( {{j_1},{j_0}}\right)_{{b}}$。那么*

1.  * 如果 $i_1\ne j_1$，则 $\mathbf{R}[i,j]=0$。*

2.  *否则（即当 $i_1=j_1$ 时），$\mathbf{R}[i,j]=\mathbf{R}_{i_1}[i_0,j_0]$。*

我们将 5 中的公式等价地重述为：

**命题 12**。* 矩阵 $\mathbf{L}$ 满足  （即 $\mathbf{L}\in \mathcal{D}\mathcal{B}^{(b,n)}$）当且仅当以下条件对任意 $0\le i,j< n$ 成立。设 $i\equiv\left( {{i_1},{i_0}}\right)_{{b}}$ 且 $j\equiv\left( {{j_1},{j_0}}\right)_{{b}}$。那么*

1.  * 如果 $i_0\ne j_0$，则 $\mathbf{L}[i,j]=0$。*

2.  *否则（即当 $i_0=j_0$ 时），$\mathbf{L}[i,j]=\mathbf{D}_{i_1,j_1}[i_0,i_0]$。*

我们将论证以下内容：

**定理 3**。*设 $1\le b\le n$ 使得 $b$ 整除 $n$。回忆 $\mathbf{P}_{(b,n)}$ 是由置换 $\sigma_{(b,n)}$ 定义的置换矩阵。设 $\mathbf{L}$ 为 $\mathcal{D}\mathcal{B}^{(b, n)}$ 中的矩阵。那么我们有 $$\mathbf{R}'=\mathbf{P}_{(b,n)}\cdot\mathbf{L}\cdot\mathbf{P}_{(b,n)}^\top,$$ 其中 $\mathbf{R}' \in \mathcal{B}\mathcal{D}^{(\frac{n}{b},\, n)}$。*

*证明。* 我们首先注意到，在 $n\times n$ 矩阵右侧（和左侧）分别乘以 $\mathbf{P}_{(b,n)}^\top = \mathbf{P}_{(\frac nb,n)}$（和 $\mathbf{P}_{(b,n)}$）会根据 $\sigma_{(b,n)}$ 置换该矩阵的列（和行）。（这利用了 $\left( {\sigma_{(b,n)}}\right)^{-1}=\sigma_{(\frac nb,n)}$ 这一事实（这意味着 $P_{(\frac{n}{b}, n)} = P_{(b, n)}^\top$，因为置换矩阵的逆矩阵是其转置）。）这意味着对于任意 $0\le i,j<n$：$$\label{eq:L-permuted}
\mathbf{R}'[\sigma_{(b,n)}(i),\sigma_{(b,n)}(j)]=\mathbf{L}[i,j].$$ 为了完成证明，我们将论证 $\mathbf{R}'$ 满足  中的两个条件。

为此，设 $0\le i,j<n$ 为任意索引，并进一步定义 $i=\left( {{i_1},{i_0}}\right)_{{b}}$ 和 $j=\left( {{j_1},{j_0}}\right)_{{b}}$。然后注意 $\sigma_{(b,n)}(i)=\left( {{i_0},{i_1}}\right)_{{\frac nb}}$ 且 $\sigma_{(b,n)}(j)=\left( {{j_0},{j_1}}\right)_{{\frac nb}}$。

根据 ，我们有如果 $i_0\ne j_0$，则 $\mathbf{L}[i,j]=0$。注意 $i_0\ne j_0$ 满足  中项 [item:zero-loc-R] 中索引 $(\sigma_{(b,n)}(i),\sigma_{(b,n)}(j))$ 的基大小 $\frac nb$ 的前提条件。然后根据 [eq:L-permuted]，我们有 $\mathbf{R}'[\sigma_{(b,n)}(i),\sigma_{(b,n)}(j)]=0$，这满足了  中的项 [item:zero-loc-R]。

现在考虑 $i_0=j_0$ 的情况；那么根据  中的项 12，我们有 $\mathbf{L}[i,j]=\mathbf{D}_{i_1,j_1}[i_0,i_0]$。注意，如果我们如下定义 $\mathbf{R}'_{i_0}\in\mathbb{F}^{\frac nb\times\frac nb}$：$$\mathbf{R}'_{i_0}[i_1,j_1]=\mathbf{D}_{i_1,j_1}[i_0,i_0].$$ 那么 $i_0= j_0$ 满足  中项 11 中索引 $(\sigma_{(b,n)}(i),\sigma_{(b,n)}(j))$ 的基大小 $\frac nb$ 的前提条件。注意上述内容意味着 $$\mathbf{R}'=\mathrm{diag}\left( {\mathbf{R}'_0,\dots,\mathbf{R}'_{b-1}}\right),$$ 其中 $\mathbf{R}'_{\cdot}$ 的定义如上一段所述。这意味着 $\mathbf{R}' \in \mathcal{B}\mathcal{D}^{(\frac{n}{b}, n)}$，因为每个块 $\mathbf{R}_{i_0}'$ 都是大小为 $\frac{n}{b} \times \frac{n}{b}$ 的矩阵。 ◻

我们现在简要说明表示 $\mathcal{M}\mathcal{M}^{*(b,n)}$ 中矩阵的一些替代方法。

**命题 13**。*对于任意 $\mathbf{M}\in \mathcal{M}\mathcal{M}^{*(b,n)}$，我们可以写成 $\mathbf{M}= (\mathbf{P}_{(b,n)}^\top \mathbf{L}_1 \mathbf{P}_{(b,n)}) \mathbf{R}(\mathbf{P}_{(b,n)}^\top\mathbf{L}_2\mathbf{P}_{(b,n)})$，其中 $\mathbf{L}_1,\mathbf{L}_2 \in \mathcal{B}\mathcal{D}^{(\frac{n}{b},n)}$ 且 $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b,n)}$。*

*证明。* 根据定义（见 4 和 5），如果 $\mathbf{M}\in \mathcal{M}\mathcal{M}^{*(b,n)}$，我们可以写成 $\mathbf{M}= (\mathbf{L}_1' \mathbf{R}_1) (\mathbf{L}_2' \mathbf{R}_2)^* = \mathbf{L}_1' (\mathbf{R}_1^* \mathbf{R}_2) \mathbf{L}_2'^*$，其中 $\mathbf{L}_1',\mathbf{L}_2' \in \mathcal{D}\mathcal{B}^{(b,n)},\mathbf{R}_1,\mathbf{R}_2 \in \mathcal{B}\mathcal{D}^{(b,n)}$。

注意，由于 $\mathbf{R}_1^*, \mathbf{R}_2$ 都是具有相同结构的块对角矩阵（即两者都有大小为 $b \times b$ 的块），它们的乘积 $\mathbf{R}$ 也在 $\mathcal{B}\mathcal{D}^{(b,n)}$ 中。同样，根据 3 我们可以写成 $\mathbf{L}_1 = \mathbf{P}_{(b,n)} \mathbf{L}_1' \mathbf{P}_{(b,n)}^\top$, $\mathbf{L}_2 = \mathbf{P}_{(b,n)} \mathbf{L}_2' \mathbf{P}_{(b,n)}^\top$，其中 $\mathbf{L}_1,\mathbf{L}_2$ 都在 $\mathcal{B}\mathcal{D}^{(\frac{n}{b},n)}$ 中（即具有大小为 $\frac{n}{b} \times \frac{n}{b}$ 的块的块对角矩阵）。

因此，我们可以写成 $\mathbf{M}= (\mathbf{P}_{(b,n)}^\top \mathbf{L}_1 \mathbf{P}_{(b,n)}) \mathbf{R}(\mathbf{P}_{(b,n)}^\top\mathbf{L}_2\mathbf{P}_{(b,n)})$，其中 $\mathbf{L}_1,\mathbf{L}_2 \in \mathcal{B}\mathcal{D}^{(\frac{n}{b},n)}$ 且 $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b,n)}$。 ◻

我们使用上述内容来展示 $\mathcal{M}\mathcal{M}^{*(b,n)}$ 和 $\mathcal{M}^*\mathcal{M}^{(b,n)}$ 之间的简单关系。

**命题 14**。*如果 $\mathbf{M}\in \mathcal{M}\mathcal{M}^{*(b,n)}$，则 $\mathbf{P}_{(b,n)} \mathbf{M}\mathbf{P}_{(b,n)}^\top \in \mathcal{M}^*\mathcal{M}^{(\frac{n}{b},n)}$。反之，如果 $\mathbf{M}\in \mathcal{M}^*\mathcal{M}^{(b,n)}$，则 $\mathbf{P}_{(b,n)}^\top \mathbf{M}\mathbf{P}_{(b,n)} \in \mathcal{M}^*\mathcal{M}^{(\frac{n}{b},n)}$。*

*证明。* 假设 $\mathbf{M}\in \mathcal{M}\mathcal{M}^{*(b,n)}$。根据 13 我们可以写成 $\mathbf{M}= (\mathbf{P}_{(b,n)}^\top \mathbf{L}_1 \mathbf{P}_{(b,n)}) \mathbf{R}(\mathbf{P}_{(b,n)}^\top\mathbf{L}_2\mathbf{P}_{(b,n)})$，其中 $\mathbf{L}_1,\mathbf{L}_2 \in \mathcal{B}\mathcal{D}^{(\frac{n}{b},n)}$ 且 $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b,n)}$。因此 $\mathbf{P}_{(b,n)} \mathbf{M}\mathbf{P}_{(b,n)}^\top =
\mathbf{L}_1 (\mathbf{P}_{(b,n)} \mathbf{R}\mathbf{P}_{(b,n)}^\top) \mathbf{L}_2$。

令 $\mathbf{L}_1' = \mathbf{L}_1, \mathbf{L}_2' = \mathbf{L}_2^*, \mathbf{R}_1' = \mathbf{P}_{(b,n)} \mathbf{R}\mathbf{P}_{(b,n)}^\top$, 且 $\mathbf{R}_2' = \mathbf{I}$, 我们有 $\mathbf{L}_1', \mathbf{L}_2' \in \mathcal{B}\mathcal{D}^{(\frac{n}{b}, n)}$, $\mathbf{R}_1', \mathbf{R}_2' \in \mathcal{D}\mathcal{B}^{(\frac{n}{b}, n)}$, 并且 $\mathbf{L}_1 (\mathbf{P}_{(b,n)} \mathbf{R}\mathbf{P}_{(b,n)}^\top) \mathbf{L}_2 = 
\mathbf{L}_1' \mathbf{R}_1' \mathbf{R}_2'^* \mathbf{L}_2'^* = (\mathbf{L}_1' \mathbf{R}_1')(\mathbf{L}_2' \mathbf{R}_2')^* = \mathbf{M}_1'\mathbf{M}_2'^*$, 其中 $\mathbf{M}_1' = \mathbf{L}_1' \mathbf{R}_1', \mathbf{M}_2' = \mathbf{L}_2' \mathbf{R}_2'$, 因此 $\mathbf{M}_1', \mathbf{M}_2' \in \mathcal{M}^*\mathcal{M}^{(\frac{n}{b},n)}$。

现在假设 $\mathbf{M}\in \mathcal{M}^*\mathcal{M}^{(b,n)}$。因此对于某些 $\mathbf{R}_1, \mathbf{R}_2 \in \mathcal{B}\mathcal{D}^{(b,n)}$ 和 $\mathbf{L}_1, \mathbf{L}_2 \in \mathcal{D}\mathcal{B}^{(b,n)}$, 有 $\mathbf{M}= \mathbf{M}_1^* \mathbf{M}_2 = \mathbf{R}_1^* \mathbf{L}_1^* \mathbf{L}_2 \mathbf{R}_2$。因此，根据 3（以及 $\mathcal{B}\mathcal{D}^{(b,n)}$ 在共轭转置下封闭的事实），对于某些 $\mathbf{R}_1' \in \mathcal{D}\mathcal{B}^{(\frac{n}{b}, n)}$，我们可以将 $\mathbf{R}_1^*$ 写成 $\mathbf{R}_1^* = \mathbf{P}_{(\frac{n}{b},n)}^\top \mathbf{R}_1' \mathbf{P}_{(\frac{n}{b}, n)} = \mathbf{P}_{(b,n)} \mathbf{R}_1' \mathbf{P}_{(b, n)}^\top$，类似地，对于某些 $\mathbf{R}_2' \in \mathcal{D}\mathcal{B}^{(\frac{n}{b}, n)}$，可以将 $\mathbf{R}_2$ 写成 $\mathbf{R}_2 = \mathbf{P}_{(b,n)} \mathbf{R}_2' \mathbf{P}_{(b,n)}^\top$。

所以 $\mathbf{P}_{(b,n)}^\top \mathbf{M}\mathbf{P}_{(b,n)} = \mathbf{R}_1' (\mathbf{P}_{(b, n)})^\top \mathbf{L}_1^*)(\mathbf{L}_2 \mathbf{P}_{(b, n)})) \mathbf{R}_2' =
 \mathbf{R}_1' (\mathbf{P}_{(b, n)}^\top \mathbf{L}_1^* \mathbf{P}_{(b, n)})(\mathbf{P}_{(b, n)}^\top \mathbf{L}_2 \mathbf{P}_{(b, n)}) \mathbf{R}_2' = (\mathbf{R}_1' \mathbf{L}_1')(\mathbf{L}_2' \mathbf{R}_2')$, 其中 $\mathbf{L}_1' = \mathbf{P}_{(b, n)}^\top \mathbf{L}_1^* \mathbf{P}_{(b, n)}$, $\mathbf{L}_2' = \mathbf{P}_{(b, n)}^\top \mathbf{L}_2 \mathbf{P}_{(b, n)}$ 根据 3 属于 $\mathcal{B}\mathcal{D}^{(\frac{n}{b}, n)}$。因此，令 $\mathbf{M}_1' = \mathbf{R}_1'\mathbf{L}_1'$, $\mathbf{M}_2' = \mathbf{R}_2^*\mathbf{L}_2'^*$, 我们有 $\mathbf{M}= \mathbf{M}_1' \mathbf{M}_2'^*$ 且 $\mathbf{M}_1', \mathbf{M}_2' \in \mathcal{M}^{*(\frac{n}{b},n)}$。 ◻

我们现在证明类 $\mathcal{M}^{(b,n)}$ 严格包含 $n \times n$ butterfly 矩阵的类 $\mathcal{B}^{(n)}$（如 \[dao2020kaleidoscope\] 中所定义）。我们首先展示两个基本的“辅助”结果。

**命题 15**. *如果 $b,\, c \in (1, n)$ 满足 $b$ 整除 $c$ 且 $c$ 整除 $n$, 那么 $\mathcal{B}\mathcal{D}^{(b, n)} \subseteq \mathcal{B}\mathcal{D}^{(c, n)}$.*

*证明.* 假设 $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b, n)}$。那么根据 [prop:R-eqv-def], 只要 $\left\lfloor\frac{i}{b}\right\rfloor \ne \left\lfloor\frac{j}{b}\right\rfloor$ 就有 $\mathbf{R}[i, j] = 0$。因此，只要 $\left\lfloor\frac{i}{c}\right\rfloor \ne \left\lfloor\frac{j}{c}\right\rfloor$ 就有 $\mathbf{R}[i, j] = 0$, 因为根据 $b$ 整除 $c$ 的假设，$\left\lfloor\frac{i}{c}\right\rfloor \ne \left\lfloor\frac{j}{c}\right\rfloor$ 意味着 $\left\lfloor\frac{i}{b}\right\rfloor \ne \left\lfloor\frac{j}{b}\right\rfloor$。再次应用 [prop:R-eqv-def], 这意味着 $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(c,n)}$ 也成立。 ◻

**命题 16**. *如果 $b,\, c \in (1, n)$ 满足 $b$ 整除 $c$ 且 $c$ 整除 $n$, 那么 $\mathcal{D}\mathcal{B}^{(c, n)} \subseteq \mathcal{D}\mathcal{B}^{(b, n)}$.*

*证明.* 假设 $\mathbf{L}\in \mathcal{D}\mathcal{B}^{(c, n)}$。那么根据 [prop:L-eqv-def], 只要 $(i \text{ mod }c) \ne (j \text{ mod }c)$ 就有 $\mathbf{L}[i, j] = 0$。因此，只要 $(i \text{ mod }b) \ne (j \text{ mod }b)$ 就有 $\mathbf{L}[i, j] = 0$, 因为根据 $b$ 整除 $c$ 的假设，$(i \text{ mod }b) \ne (j \text{ mod }b)$ 意味着 $(i \text{ mod }c) \ne (j \text{ mod }c)$。再次应用 [prop:L-eqv-def], 这意味着 $\mathbf{L}\in \mathcal{D}\mathcal{B}^{(b,n)}$ 也成立。 ◻

**定理 4**. *令 $n \ge 4$ 为 2 的幂。矩阵类 $\mathcal{B}^{(n)}$ 是类 $\mathcal{M}^{(b, n)}$ 的子集, 对于所有整除 $n$ 的 $b \in (1, n)$。当 $n \ge 512$ 时, 它是一个严格子集。*

*证明.* 回顾 2.2 节，如果 $\mathbf{B}\in \mathcal{B}^{(n)}$, 它具有一个 *butterfly 分解* $\mathbf{B}= \mathbf{B}_n \mathbf{B}_{n/2} \hdots \mathbf{B}_2$, 其中每个 $\mathbf{B}_i \in \mathcal{B}\mathcal{F}^{(n, i)}$。

考虑将因子 $\mathbf{B}_b \mathbf{B}_{b/2} \dots \mathbf{B}_2$ 相乘（其中 $b \in (1, n)$ 整除 $n$）。由于 $\mathbf{B}_i \in \mathcal{B}\mathcal{F}^{(n,i)}$, 根据定义，它是具有大小为 $i \times i$ 的对角块的块对角矩阵；换句话说，$\mathbf{B}_i \in \mathcal{B}\mathcal{D}^{(i, n)}$。因此，矩阵 $\mathbf{B}_b, \mathbf{B}_{b/2}, \dots, \mathbf{B}_2$ 中的每一个都属于 $\mathcal{B}\mathcal{D}^{(b, n)}$（根据 15），即块大小为 $b \times b$ 的块对角矩阵。这意味着它们的乘积 $\mathbf{B}_b \mathbf{B}_{b/2} \dots \mathbf{B}_2$ 也是块大小为 $b \times b$ 的块对角矩阵，即它属于 $\mathcal{B}\mathcal{D}^{(b, n)}$。

现在，注意由于 $\mathbf{B}_i \in \mathcal{B}\mathcal{F}^{(n,i)}$, 根据定义，它是一个具有大小为 $i/2 \times i/2$ 块的块矩阵，其中每个块都是对角矩阵（注意其中一些块为零，除了 $\mathbf{B}_n$ 的情况）。换句话说，$\mathbf{B}_i \in \mathcal{D}\mathcal{B}^{(i/2, n)}$。因此，对于所有 $i \in \{n, n/2, \dots, 2b\}$, $\mathbf{B}_i \in \mathcal{D}\mathcal{B}^{((2b)/2, n)} = \mathcal{D}\mathcal{B}^{(b, n)}$（根据 16）。所以，它们的乘积 $\mathbf{B}_n \mathbf{B}_{n/2} \dots \mathbf{B}_{2b}$ 也属于 $\mathcal{D}\mathcal{B}^{(b, n)}$, 因为根据 3 我们可以写成 $\mathbf{B}_n \mathbf{B}_{n/2} \dots \mathbf{B}_{2b} = \mathbf{P}_{(b,n)}^\top (\mathbf{P}_{(b,n)} \mathbf{B}_n \mathbf{P}_{(b,n)}^\top) (\mathbf{P}_{(b,n)} \mathbf{B}_{n/2} \mathbf{P}_{(b,n)}^\top) \dots (\mathbf{P}_{(b,n)} \mathbf{B}_{2b} \mathbf{P}_{(b,n)}^\top) \mathbf{P}_{(b,n)}$ 并且前述表达式中的每个 $\mathbf{P}_{(b,n)} \mathbf{B}_i \mathbf{P}_{(b,n)}^\top$ 都属于 $\mathcal{B}\mathcal{D}^{(\frac{n}{b}, n)}$。

因此，如果我们令 $\mathbf{L}= \mathbf{B}_n \mathbf{B}_{n/2} \dots \mathbf{B}_{2b}$ 且 $\mathbf{R}= \mathbf{B}_b \mathbf{B}_{b/2} \dots \mathbf{B}_2$, 我们有 $\mathbf{B}= \mathbf{L}\mathbf{R}$ 且 $\mathbf{L}\in \mathcal{D}\mathcal{B}^{(b, n)}$, $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b, n)}$, 这意味着 $\mathbf{B}\in \mathcal{M}^{(b,n)}$ (22)。

为了展示包含关系是严格的，请注意任何 $\mathbf{M}\in \mathcal{M}^{(b,n)}$ 都是 $\mathbf{L}$ 和 $\mathbf{R}$ 的乘积，其中 $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b, n)}$ 且 $\mathbf{P}_{(b,n)}^\top \mathbf{L}\mathbf{P}_{(b,n)} \in \mathcal{B}\mathcal{D}^{(\frac{n}{b}, n)}$（根据 3）。注意，单位矩阵同时包含在 $\mathcal{B}\mathcal{D}^{(b,n)}$ 和 $\mathcal{D}\mathcal{B}^{(b,n)}$ 中。首先假设 $b \le \sqrt{n}$。那么即使我们将 $\mathbf{R}$ 设为单位矩阵，$\mathbf{M}$ 也至少有 $\frac{n^2}{b} \ge n^{3/2}$ 个自由参数（块对角矩阵 $\mathbf{P}_{(b,n)}^\top \mathbf{L}\mathbf{P}_{(b,n)}$ 块中的元素可以是任意的，并且有 $b$ 个这样的块，每个块的大小为 $\frac{n}{b}$）。类似地，在 $b > \sqrt{n}$ 的情况下，我们可以将 $\mathbf{L}$ 设为单位矩阵，并且 $\mathbf{M}$ 至少有 $nb \ge n^{3/2}$ 个自由参数（块对角矩阵 $\mathbf{R}$ 的元素可以是任意的，总共有 $nb$ 个这样的元素）。因此，唯一描述 $\mathcal{M}^{(b,n)}$ 中的任何矩阵至少需要 $n^{3/2}$ 个参数。然而，$\mathcal{B}^{(n)}$ 中的蝴蝶矩阵只有 $2n \log_2 n$ 个参数。对于 $n > 256$，$2n \log_2 n < n^{3/2}$。（注意，此分析并不紧致：更仔细的分析可以证明，即使对于较小的 $n$ 值，包含关系也是严格的。） ◻

我们以一个关于“君主层次结构”（君主矩阵的乘积）表达能力的定理来结束本节，该定理源自 \[dao2020kaleidoscope\] 的定理 1。

**定理 5**（君主层次结构的表达能力）。*设 $\mathbf{M}$ 为一个 $n \times n$ 矩阵，使得 $\mathbf{M}$ 与任意向量 $\mathbf{v}$ 的矩阵-向量乘法（即计算 $\mathbf{M}\mathbf{v}$）可以表示为一个深度为 $d$ 且总共有 $s$ 个门的线性算术电路。设 $b \in (1, n)$ 为 2 的幂且整除 $n$。那么，$\mathbf{M}\in (\mathcal{M}\mathcal{M}^{*(b, n)})^{O(d)}_{O(s/n)}$。*

*证明。* \[dao2020kaleidoscope\] 的定理 1 指出，如果 $n$ 是 2 的幂，且 $\mathbf{A}$ 是一个 $n \times n$ 矩阵，使得将任意向量 $v$ 乘以 $\mathbf{A}$ 可以表示为一个深度 $\le d$ 且总共有 $\le s$ 个门的线性算术电路，那么 $\mathbf{A}\in (\mathcal{B}\mathcal{B}^{*(n)})^{O(d)}_{O(s/n)}$（这是 $\mathbf{A}$ 的“万花筒表示”）。

回顾 4，对于任何 $b \in (1, n)$ 为 2 的幂且整除 $n$，$\mathcal{M}^{(b, n)} \supset \mathcal{B}^{(n)}$；因此，这意味着 $\mathcal{M}\mathcal{M}^{*(b,e\cdot n)} \supset \mathcal{B}\mathcal{B}^{*(e\cdot n)}$，进而 $(\mathcal{M}\mathcal{M}^{*(b,n)})^w_e \supset (\mathcal{B}\mathcal{B}^{*(n)})^w_e$。

由于 $\mathbf{A}\in  (\mathcal{B}\mathcal{B}^{*(n)})^{O(d)}_{O(s/n)}$，我们因此有 $\mathbf{A}\in  (\mathcal{M}\mathcal{M}^{*(b,n)})^{O(d)}_{O(s/n)}$。 ◻

根据 \[dao2020kaleidoscope\]，万花筒矩阵类 $(\mathcal{B}\mathcal{B}^{*(n)})^{O(d)}_{O(s/n)}$ 具有 $O(ds \log s)$ 个参数和运行时间，相比之下，电路的参数和运行时间为 $O(s)$。注意，在最坏的情况下，$s$ 为 $O(n^2)$。

定义 $f(n,s)$ 为 $\le \min\left\{\tfrac{n}{2}, \sqrt{s}\right\}$ 的最大 2 的幂。注意 $f(n,s) = O(\sqrt{s})$，并且由于 $s = O(n^2)$，$f(n,s) = \Omega(\sqrt{s})$，因此 $f(n,s) = \Theta(\sqrt{s})$。我们因此有 $\mathbf{A}\in (\mathcal{M}\mathcal{M}^{*(f(n,s), n)})^{O(d)}_{O(s/n)}$。类 $(\mathcal{M}\mathcal{M}^{*(f(n,s), n)})^{O(d)}_{O(s/n)}$ 具有 $O(d\frac{s^2}{f(n,s)} + dsf(n,s)) = O(ds^{3/2})$ 个参数。因此，与万花筒的 $O(d{}\,\log s)$ 相比，$\mathbf{A}$ 的君主表示最多差一个 $O(d\sqrt{s})$ 因子。

## 一般长方矩阵

在本节中，我们将 Monarch 参数化扩展至适用于*长方*矩阵，并证明相关矩阵类的一些基本性质。（注意，我们后续的理论结果 (10) 不依赖于本节，因为它们侧重于方形参数化。）

在本节的其余部分，我们将假设 $n_1, n_2, n_3, b_1, b_2 , b_3 \ge 1$ 是满足以下条件的整数：

- $b_i$ 整除 $n_i$，对于所有 $1\le i\le 3$，且

- $\frac{n_1}{b_1} = \frac{n_2}{b_2}$。

我们从以下长方块对角矩阵类的定义开始：

**定义 17**。对于 $0\le i< \frac{n}{b_1}$，令 $\mathbf{R}_{i}\in\mathbb{F}^{b_2 \times b_1}$ 为一个 $b_2 \times b_1$ 矩阵。然后按如下方式定义矩阵 $\mathbf{R}\in \mathbb{F}^{n_2\times n_1}$： $$\label{eq:def-rect-R}
  \mathbf{R}= \mathrm{diag}\left(\mathbf{R}_0, \dots, \mathbf{R}_{\frac {n_1}{b_1}-1}\right).$$

我们称 $\mathbf{R}$ 具有*块大小* $b_2 \times b_1$。回顾我们已假设 $\frac {n_1}{b_1}=\frac {n_2}{b_2}$，因此 [eq:def-rect-R] 是良定义的。（注意 $\mathbf{R}$ 中可能的非零值数量为 $\frac {n_1}{b_1}\cdot b_1 \times b_2 =n_1b_2$。）我们将所有能表示为这种形式的矩阵 $\mathbf{R}$ 的类记为 $\mathcal{B}\mathcal{D}^{(b_2 \times b_1, n_2 \times n_1)}$。注意该类仅在 $\frac {n_1}{b_1}=\frac {n_2}{n_2}$ 时有定义。

我们将上述定义等价地重述如下：

**命题 18**。* $\mathbf{R}\in\mathbb{F}^{n_2\times n_1}$ 属于 $\mathcal{B}\mathcal{D}^{(b_2 \times b_1, n_2 \times n_1)}$（其中 $\frac {n_1}{b_1}=\frac {n_2}{n_2}$）当且仅当以下条件对任意 $0\le i < n_2$ 和 $0\le j< n_1$ 成立。令 $i\equiv\left( {{i_1},{i_0}}\right)_{{b_2}}$ 且 $j\equiv\left( {{j_1},{j_0}}\right)_{{b_1}}$（回顾 9 中的此记法。那么*

1.  * 若 $i_1\ne j_1$，则 $\mathbf{R}[i,j]=0$。*

2.  *否则（即当 $i_1=j_1$ 时），$\mathbf{R}[i,j]=\mathbf{R}_{i_1}[i_0,j_0]$。*

在定义长方矩阵 $\mathbf{L}$ 之前，我们首先需要定义“缠绕对角”（wrapped diagonal）矩阵的概念：

**定义 19**。*缠绕对角*矩阵 $\mathbf{S} \in\mathbb{F}^{b_3\times b_2}$ 定义如下。首先假设 $b_2\le b_3$。那么对于任意 $0\le i<b_3$ 和 $0\le j<b_2$，我们有以下结论。若 $i\text{ mod }{b_2}\ne j$，则 $\mathbf{S}[i,j]=0$。（若 $b_2>b_3$，则将前述定义应用于 $\mathbf{S}^{\top}$。）

我们现在定义以下块矩阵类，其中每个块都是一个*缠绕对角*矩阵。

**定义 20**。令 $\mathbf{L}\in\mathbb{F}^{n_3\times n_2}$ 具有以下形式： $$\label{eq:rect-def-L}
    \mathbf{L}=
        \begin{bmatrix}
            \mathbf{S}_{0,0} & \dots & \mathbf{S}_{0,\frac{n_2}{b_2} -1} \\
            \vdots & \ddots & \vdots \\
            \mathbf{S}_{\frac{n_3}{b_3} -1,0} & \dots & \mathbf{S}_{\frac{n_3}{b_3} -1,\frac{n_2}{b_2} -1}
        \end{bmatrix},$$ 其中每个 $\mathbf{S}_{\cdot,\cdot}$ 都是 $\mathbb{F}^{b_3 \times b_2}$ 中的缠绕对角矩阵。

我们称 $\mathbf{L}$ 具有*块大小* $b_3 \times b_2$。（注意 $\mathbf{L}$ 中可能的非零值数量为 $\left( {\frac {n_2}{b_2}\cdot \frac{n_3}{b_3}}\right) \max(b_2,b_3)=\frac{n_2 \cdot n_3}{\min(b_2,b_3)}$。）我们将所有能表示为这种形式的矩阵 $\mathbf{L}$ 的类记为 $\mathcal{D}\mathcal{B}^{(b_3 \times b_2, n_3 \times n_2)}$。

我们将上述定义等价地重述如下：

**命题 21**。* $\mathbf{L}\in\mathbb{F}^{n_3\times n_2}$ 属于 $\mathcal{D}\mathcal{B}^{(b_3 \times b_2, n_3 \times n_2)}$ 当且仅当以下条件对任意 $0\le i < n_3$ 和 $0 \le j< n_2$ 成立。令 $i\equiv\left( {{i_1},{i_0}}\right)_{{b_3}}$ 且 $j\equiv\left( {{j_1},{j_0}}\right)_{{b_2}}$。假设 $b_2 \le b_3$，我们有：*

1.  * 若 $i_0\text{ mod }{b_2}\ne j_0$，则 $\mathbf{L}[i,j]=0$。*

2.  *否则（即当 $i_0\text{ mod }{b_2}=j_0$ 时），$\mathbf{L}[i,j]=\mathbf{S}_{i_1,j_1}[i_0,j_0]$。*

*若 $b_2>b_3$，则在上述内容中，条件“$i_0\text{ mod }{b_2}\ne j_0$”将被替换为“$j_0\text{ mod }{b_2}\ne i_0$”。*

使用上述定义，我们现在定义长方 Monarch 矩阵类。

**定义 22**（长方 Monarch 矩阵）。令 $\mathbf{M}\in \mathbb{F}^{n_3 \times n_1}$ 为具有以下形式的矩阵： $$\label{eq:Monarch-general}
    \mathbf{M}= \mathbf{L}\mathbf{R}$$ 其中 $\mathbf{L}\in \mathcal{D}\mathcal{B}^{(b_3 \times b_2, n_3 \times n_2)}$ 且 $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b_2 \times b_1, n_2 \times n_1)}$。

（如前所述，我们假设对于 $i = 1,2,3$，$b_i$ 整除 $n_i$，且 $n_1/b_1 = n_2/b_2$。）我们将所有能表示为这种形式的矩阵 $\mathbf{M}$ 的类记为 $\mathcal{M}^{((b_1,b_2,b_3), (n_1,n_2,n_3))}$。注意到当 $b_1 = b_2 = b_3 = b$ 且 $n_1 = n_2 = n_3 = n$ 时，这恰好是 22 中的矩阵类 $\mathcal{M}^{(b, n)}$。

我们现在准备好证明本节的主要结果，该结果基本上源于以下观察：如果我们置换 $\mathbf{L}$ 的行和列，使得 $\mathbf{L}$ 中的行/列块大小成为置换后矩阵中行/列块的数量（反之亦然），那么置换后的矩阵具有 $\mathbf{R}$ 的形式。

**定理 6**。*设 $1\le b,n_2,n_3$ 满足 $b$ 整除 $n_2$ 和 $n_3$。假设 $\mathbf{L}\in\mathbb{F}^{n_3\times n_2} \in \mathcal{D}\mathcal{B}^{(b \times b, n_3 \times n_2)}$。那么如果我们定义 $$\mathbf{R}'=\mathbf{P}_{(b,n_3)}\cdot\mathbf{L}\cdot\mathbf{P}_{(b,n_2)}^\top,$$ 我们有 $\mathbf{R}' \in \mathcal{B}\mathcal{D}^{(\frac{n_3}{b_3} \times \frac{n_2}{b_2}, n_3 \times n_2)}$。*

*证明。*我们回顾，将一个 $m\times n$ 矩阵在右（和左）侧分别乘以 $\mathbf{P}_{(b,n)}^\top = \mathbf{P}_{(\frac nb,n)}$（和 $\mathbf{P}_{(b,m)}$）会分别根据 $\sigma_{(b,n)}$（和 $\sigma_{(b,m)}$）置换该矩阵的列（和行）。（这利用了 $\left( {\sigma_{(b,n)}}\right)^{-1}=\sigma_{(\frac nb,n)}$ 这一事实。）这意味着对于任意 $0\le i,j<n$： $$\label{eq:rect-L-permuted}
\mathbf{R}'[\sigma_{(b,n_3)}(i),\sigma_{(b,n_2)}(j)]=\mathbf{L}[i,j].$$

回顾在 20 的记法中我们有 $b_2=b_3=b$，所以我们处于 $b_2 \le b_3$ 的情况。为了完成证明，我们将论证 $\mathbf{R}'$ 满足 中的两个条件。（注意我们还需要行/列长度与行/列块大小的比率相同；即在我们的情况中，我们需要 $\frac{n_3}{n_3 / b_3}=\frac{n_2}{n_2 / b_2}$，这是成立的，因为 $b_2=b_3=b$。）

为此，令 $0\le i,j<n$ 为任意索引，并进一步定义 $i=\left( {{i_1},{i_0}}\right)_{{b}}$ 和 $j=\left( {{j_1},{j_0}}\right)_{{b}}$。那么注意到 $\sigma_{(b,n_3)}(i)=\left( {{i_0},{i_1}}\right)_{{\frac {n_3}{b}}}$ 且 $\sigma_{(b,n_2)}(j)=\left( {{j_0},{j_1}}\right)_{{\frac {n_2}{b}}}$。

根据 ，我们有如果 $i_0\text{ mod }{b}\ne j_0$，则 $\mathbf{L}[i,j]=0$。注意由于根据定义 $i_0,j_0<b$，条件 $i_0\text{ mod }{b}\ne j_0$ 等价于 $i_0\ne j_0$。注意 $i_0\ne j_0$ 满足 中项 [item:rect-zero-loc-R] 里对于基大小 $\frac {n_3}{b}\times \frac {n_2}{b}$ 和索引 $(\sigma_{(b,n_3)}(i),\sigma_{(b,n_2)}(j))$ 的前提条件。然后根据 [eq:rect-L-permuted]，我们有 $\mathbf{R}'[\sigma_{(b,n_3)}(i),\sigma_{(b,n_2)}(j)]=0$，这满足了 中的项 [item:rect-zero-loc-R]。

现在考虑 $i_0=j\text{ mod }b$ 的情况，根据上一段的观察，这等同于 $i_0=j_0$。然后根据 中的第 21 项，我们有 $\mathbf{L}[i,j]=\mathbf{S}_{i_1,j_1}[i_0,j_0]$。注意，$i_0= j_0$ 满足 中的第 18 项中索引 $(\sigma_{(b,n_3)}(i),\sigma_{(b,n_2)}(j))$ 的基础大小为 $\frac{n_3}{b}\times \frac{n_2}{b}$ 的前置条件，如果我们将 $\mathbf{R}'_{i_0}\in\mathbb{F}^{\frac{n_3}{b}\times\frac{n_2}{b}}$ 定义如下： $$\mathbf{R}'_{i_0}[i_1,j_1]=\mathbf{S}_{i_1,j_1}[i_0,j_0].$$

注意，上述内容意味着 $$\mathbf{R}'=\mathrm{diag}\left( {\mathbf{R}'_0,\dots,\mathbf{R}'_{b-1}}\right),$$ 其中 $\mathbf{R}'_{\cdot}$ 的定义如上一段所述。这意味着 $\mathbf{R}' \in \mathcal{B}\mathcal{D}^{(\frac{n_3}{b} \times \frac{n_2}{b}, n_3 \times n_2)}$，因为 $\mathbf{R}'$ 的大小为 $n_3 \times n_2$，且每个块 $\mathbf{R}_{i_0}'$ 是大小为 $\frac{n_3}{b} \times \frac{n_2}{b}$ 的矩阵。 ◻

# 理论

## $\mathcal{M}$ 的表达能力


*2的证明。* 正如 [dao2020kaleidoscope] 所展示的，矩阵类 $\mathcal{B}\mathcal{B}^*$ 可以表示卷积、Hadamard 变换、Toeplitz 矩阵和 AFDF。由于 Monarch 类 $\mathcal{M}\mathcal{M}^*$ 包含蝴蝶类 $\mathcal{B}\mathcal{B}^*$（由 4 得出），因此 $\mathcal{M}\mathcal{M}^*$ 也可以表示这些变换/矩阵。

注意，Hadamard 变换实际上属于 $\mathcal{B}$ [dao2020kaleidoscope]，因此它也属于 $\mathcal{M}$。

[dao2020kaleidoscope] 还表明，矩阵类 $(\mathcal{B}\mathcal{B}^*)^2$ 可以表示傅里叶变换、离散正弦/余弦变换、$(HD)^3$ 类、Fastfood 和 ACDC 矩阵。基于同样的论点，由于 Monarch 类 $(\mathcal{M}\mathcal{M}^*)^2$ 包含蝴蝶类 $(\mathcal{B}\mathcal{B}^*)^2$，因此 $(\mathcal{M}\mathcal{M}^*)^2$ 也可以表示这些变换/矩阵。 ◻

## 投影到 $\mathcal{M}$

在 [alg:project] 中，我们提供了 3.3 中概述的算法的伪代码。我们现在证明 1。注意，通过将方形块替换为矩形块，矩形矩阵的情况可以从方形矩阵的情况自然推广。


*1的证明。* 如 3.3 所示，在将 Monarch 矩阵 $\mathbf{M}$ 重塑为 4D 张量 $M_{\ell jki}$ 并将两个块对角矩阵 $\mathbf{L}$ 和 $\mathbf{R}$ 写为 3D 张量 $L_{j \ell k}$ 和 $R_{k j i}$ 后，我们得到： $$M_{\ell j k i} = L_{j \ell k} R_{k j i}, \quad \text{for } \ell, j, k, i = 1, \dots, m.$$ 我们可以类似地将给定矩阵 $A$ 重塑为大小为 $m \times m \times m \times m$ 的 4D 张量 $A_{\ell j k i}$。

由于平方 Frobenius 范数目标 $\left\|{A - M}\right\|_F^2$ ([eq:projection_objective]) 仅依赖于 $A$ 和 $M$ 的条目而不依赖于它们的形状，我们可以在重塑后重写目标： $$\begin{aligned}
    \left\|{A - M}\right\|_F^2
    &= \sum_{\ell  j k i} \left(A_{\ell  j k i} - M_{\ell  j k i}\right)^2 \\
    &= \sum_{\ell  j k i} \left( A_{\ell  j k i} - L_{j \ell  k} R_{k j i} \right)^2 \\
    &= \sum_{j k} \sum_{\ell i} \left( A_{\ell  j k i} - L_{j \ell  k} R_{k j i} \right)^2.
  
\end{aligned}$$ 我们看到目标分解为 $m \times m$ 个独立项（由 $j$ 和 $k$ 索引）。对于 $j$ 和 $k$ 的每个值，目标恰好是对应切片 $\mathbf{A}_{:, j, k, :}$ 的秩为 1 的近似目标。

设 $\mathbf{u}_{jk} \mathbf{v}_{jk}^\top$ 为 $\mathbf{A}_{:, j, k, :}$ 的最佳秩为 1 的近似（我们可以使用 SVD，根据针对 Frobenius 范数的 Eckart–Young 定理 [eckart1936approximation] 来计算）。设 $\mathbf{R}$ 为大小为 $m \times m \times m$ 的 3D 张量，其中 $\mathbf{R}_{kji} = (\mathbf{v}_{jk})_i$，并设 $\mathbf{L}$ 为大小为 $m \times m \times m$ 的 3D 张量，其中 $\mathbf{L}_{j \ell k} = (\mathbf{u}_{jk})_\ell$。然后，目标中的每一项都被最小化，因此整体目标也被最小化。

我们看到该算法需要 $m \cdot m$ 次 SVD，每次大小为 $m \times m$。每次 SVD 耗时 $O(m^3)$ [trefethen2000spectral]，因此总体时间复杂度为 $O(m^5) = O(n^{5/2})$。 ◻

## $\mathcal{M}\mathcal{M}^*$ 中矩阵的 Monarch 分解

在本节中，我们描述了之前在 3.4 ([alg:mm_recovery]) 中概述的对 $\mathcal{M}\mathcal{M}^*$ 中矩阵进行分解的算法。同样，[alg:mm_recovery] 处理 $L$ 和 $R$ 的块大小可以不同的一般情况。然后我们证明 7，其直接推论为 2。

因此，我们的目标是计算 $\mathbf{M}$ 分解中的矩阵 $\mathbf{L}_1,\mathbf{R},\mathbf{L}_2$。为了计算该分解，我们对 $\mathbf{M}$ 提出以下假设：

**Assumption 23**. 假设 (1) $\mathbf{M}\in \mathcal{M}\mathcal{M}^{*(b,n)}$ 是可逆的，并且 (2) $\mathbf{M}$ 可以写成 $(\mathbf{P}_{(b,n)}^\top \mathbf{L}_1 \mathbf{P}_{(b,n)}) \mathbf{R}(\mathbf{P}_{(b,n)}^\top\mathbf{L}_2\mathbf{P}_{(b,n)})$，其中 $\mathbf{L}_1,\mathbf{L}_2 \in \mathcal{B}\mathcal{D}^{(\frac{n}{b},n)},\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b,n)}$，且 $\mathbf{R}$ 的对角块中没有非零项。（注意，根据 13，我们可以将任何 $\mathbf{M}\in \mathcal{M}\mathcal{M}^{*(b,n)}$ 写成 $(\mathbf{P}_{(b,n)}^\top \mathbf{L}_1 \mathbf{P}_{(b,n)}) \mathbf{R}(\mathbf{P}_{(b,n)}^\top\mathbf{L}_2\mathbf{P}_{(b,n)})$；因此，(2) 仅仅是假设 $\mathbf{R}$ 的块中没有零项。）

这类似于 3，但适用于更一般的块大小 $b$。我们现在提出 [alg:mm_recovery] 来寻找满足 23 的矩阵的因子 $\mathbf{L}_1,\mathbf{R},\mathbf{L}_2$。

首先，观察到如果我们定义 $\widetilde{\mathbf{M}}= \mathbf{P}_{(b,n)} \mathbf{M}\mathbf{P}_{(b,n)}^\top$，我们有 $\widetilde{\mathbf{M}}= \mathbf{L}_1 (\mathbf{P}_{(b,n)} \mathbf{R}\mathbf{P}_{(b,n)}^\top) \mathbf{L}_2$。根据 3，矩阵 $\mathbf{P}_{(b,n)} \mathbf{R}\mathbf{P}_{(b,n)}^\top$ 属于 $\mathcal{D}\mathcal{B}^{(\frac{n}{b}, n)}$，即是一个块大小为 $\frac{n}{b} \times \frac{n}{b}$ 的块矩阵，其中每个块都是对角矩阵。因此，我们可以写成：

$$\left(\begin{array}{cccc} \widetilde{\mathbf{M}}_{11} & \widetilde{\mathbf{M}}_{12} & \dots & \widetilde{\mathbf{M}}_{1b} \\ \widetilde{\mathbf{M}}_{21} & \widetilde{\mathbf{M}}_{22} & \dots & \widetilde{\mathbf{M}}_{2b} \\ \ddots & \ddots & \ddots & \ddots  \\ \widetilde{\mathbf{M}}_{b1} & \widetilde{\mathbf{M}}_{b2} & \dots & \widetilde{\mathbf{M}}_{bb} \end{array}\right)
= \left(\begin{array}{ccccc} \mathbf{A}_1 \\ & \mathbf{A}_2 \\ & & \ddots \\ & & & \mathbf{A}_{b} \end{array}\right)
\left(\begin{array}{cccc} \mathbf{D}_{11} & \mathbf{D}_{12} & \dots & \mathbf{D}_{1b} \\ \mathbf{D}_{21} & \mathbf{D}_{22} & \dots & \mathbf{D}_{2b} \\ \ddots & \ddots & \ddots & \ddots  \\ \mathbf{D}_{b1} & \mathbf{D}_{b2} & \dots & \mathbf{D}_{bb} \end{array}\right)
\left(\begin{array}{ccccc} \mathbf{C}_1 \\ & \mathbf{C}_2 \\ & & \ddots \\ & & & \mathbf{C}_{b} \end{array}\right),$$

其中 $\mathbf{A}_1,\dots,\mathbf{A}_b$ 是 $\frac{n}{b} \times \frac{n}{b}$ 矩阵，为 $\mathbf{L}_1$ 的对角块；$\mathbf{C}_1,\dots,\mathbf{C}_b$ 是 $\frac{n}{b} \times \frac{n}{b}$ 矩阵，为 $\mathbf{L}_2$ 的对角块；$\mathbf{D}_{11},\dots,\mathbf{D}_{1b},\mathbf{D}_{21},\dots,\mathbf{D}_{2b},\dots,\mathbf{D}_{b1},\dots,\mathbf{D}_{bb}$ 是 $\frac{n}{b} \times \frac{n}{b}$ 的*对角*矩阵，为 $\mathbf{P}_{(b,n)} \mathbf{R}\mathbf{P}_{(b,n)}^\top$ 的块；且 $\widetilde{\mathbf{M}}_{11},\dots,\widetilde{\mathbf{M}}_{1b},\widetilde{\mathbf{M}}_{21},\dots,\widetilde{\mathbf{M}}_{2b},\dots,\widetilde{\mathbf{M}}_{b1},\dots,\widetilde{\mathbf{M}}_{bb}$ 是 $\frac{n}{b} \times \frac{n}{b}$ 矩阵，为 $\widetilde{\mathbf{M}}= \mathbf{P}_{(b,n)} \mathbf{M}\mathbf{P}_{(b,n)}^\top$ 的块。

因此，我们得到矩阵方程组 $\mathbf{A}_i \mathbf{D}_{ij} \mathbf{C}_j = \widetilde{\mathbf{M}}_{ij}$，其中 $1 \le i, j \le b$。注意，假设 $\mathbf{R}$ 的块中没有非零项 (23) 等价于假设任何矩阵 $\mathbf{D}_{ij}$ 的对角项均不为零。此外，假设 $\mathbf{M}$ 是可逆的意味着 $\mathbf{L}_1, \mathbf{L}_2$ 是可逆的（因为奇异方阵的乘积是奇异的），这反过来意味着每个块矩阵 $\mathbf{A}_i$ 和每个块矩阵 $\mathbf{C}_j$ 都是可逆的（因为如果一个方块对角矩阵的其中一个块是奇异的，那么该矩阵本身也是奇异的）。综上所述，这意味着每个矩阵 $\widetilde{\mathbf{M}}_{ij}$ 都是可逆的，因为 $\widetilde{\mathbf{M}}_{ij} = \mathbf{A}_i \mathbf{D}_{ij} \mathbf{C}_j$ 且等式右侧的每个矩阵都是可逆的。

观察到，给定方程组 $\mathbf{A}_i \mathbf{D}_{ij} \mathbf{C}_j = \widetilde{\mathbf{M}}_{ij}$ 的一个解，如果我们适当地重新缩放和排列矩阵 $\mathbf{A}_i, \mathbf{D}_{ij}, \mathbf{C}_j$，结果仍然是该方程组的解。具体来说，令 $\mathbf{P}$ 为任意置换矩阵，且 $\{\mathbf{S}_i\}_{i=1}^b, \{\mathbf{S}_j'\}_{j=1}^b$ 为任意可逆对角矩阵（即对角线上没有任何零的对角矩阵）。对于所有 $i, j$，定义 $\mathbf{D}_{ij}' = \mathbf{S}_i \mathbf{P}^\top {\mathbf{D}}_{ij} \mathbf{P}\mathbf{S}_j'$。注意，$\mathbf{P}^\top \mathbf{D}_{ij} \mathbf{P}= \mathbf{P}^{-1} \mathbf{D}_{ij} \mathbf{P}$ 是对角矩阵，因为 $\mathbf{D}_{ij}$ 是对角矩阵。因此，$\mathbf{D}_{ij}'$ 是对角的（且可逆的），因为对角矩阵的乘积是对角矩阵。对于所有 $i, j$，定义 $\mathbf{A}_i' = \mathbf{A}_i \mathbf{P}\mathbf{S}_i^{-1}$ 和 $\mathbf{C}_j' = \mathbf{P}^\top \mathbf{S}_j'^{-1} \mathbf{C}_j$。因此，对于所有 $i, j$，我们有 $\widetilde{\mathbf{M}}_{ij} = \mathbf{A}_i \mathbf{D}_{ij} \mathbf{C}_j = (\mathbf{A}_i \mathbf{P}\mathbf{S}_i^{-1} ) \mathbf{D}_{ij}' (\mathbf{P}^\top \mathbf{S}_j'^{-1} \mathbf{C}_j) = {\mathbf{A}}_i' \mathbf{D}_{ij}' \mathbf{C}_j'$：换句话说，我们可以在右侧用任意可逆对角矩阵缩放 $\mathbf{A}_i$，在左侧用任意可逆对角矩阵缩放 $\mathbf{C}_j$，对 $\mathbf{C}_j$ 的行和 $\mathbf{A}_i$ 的列应用匹配的置换，并对 $\mathbf{D}_{ij}$ 应用匹配的变换，结果将仍然是有效的分解。这意味着，只要我们恢复出一个“正确”的 $\hat{\mathbf{C}}_1$（至多相差其行的置换和缩放），我们就可以将 $\hat{\mathbf{D}}_{i1}$ 和 $\hat{\mathbf{D}}_{1j}$ 设置为单位矩阵，然后通过方程 $\hat{\mathbf{A}}_i = \widetilde{\mathbf{M}}_{i1}\hat{\mathbf{C}}_1^{-1}$ 和 $\hat{\mathbf{C}}_j = \hat{\mathbf{A}}_1^{-1}\widetilde{\mathbf{M}}_{1j}$ 计算剩余的 $\hat{\mathbf{A}}_i$ 和 $\hat{\mathbf{C}}_j$。

为了理解我们如何计算这样一个矩阵 $\hat{\mathbf{C}}_1$，定义 $\mathbf{F}(i, j) = \widetilde{\mathbf{M}}_{i1}^{-1} \widetilde{\mathbf{M}}_{ij} \widetilde{\mathbf{M}}_{1j}^{-1} \widetilde{\mathbf{M}}_{11}$ 并观察到 $$\begin{aligned}
\mathbf{F}(i, j) &= \widetilde{\mathbf{M}}_{i1}^{-1} \widetilde{\mathbf{M}}_{ij} \widetilde{\mathbf{M}}_{1j}^{-1} \widetilde{\mathbf{M}}_{11} \\ &=
(\mathbf{C}_1^{-1} \mathbf{D}_{i1}^{-1} \mathbf{A}_i^{-1}) (\mathbf{A}_i \mathbf{D}_{ij} \mathbf{C}_j) (\mathbf{C}_j^{-1} \mathbf{D}_{1j}^{-1} \mathbf{A}_1^{-1}) (\mathbf{A}_1 \mathbf{D}_{11} \mathbf{C}_1) \\
&= \mathbf{C}_1^{-1} (\mathbf{D}_{i1}^{-1} \mathbf{D}_{ij} \mathbf{D}_{1j}^{-1} \mathbf{D}_{11}) \mathbf{C}_1
\end{aligned}$$ 对于所有 $1 \le i, j \le b$ 成立。注意 $\mathbf{D}_{i1}^{-1} \mathbf{D}_{ij} \mathbf{D}_{1j}^{-1} \mathbf{D}_{11}$ 是一个对角矩阵；因此，$\mathbf{C}_1 \mathbf{F}(i, j) \mathbf{C}_1^{-1}$ 对于所有 $i, j$ 都是对角矩阵，即 $\mathbf{C}_1$ 同时对角化所有矩阵 $\mathbf{F}(i, j)$。（注：在本文中，我们说一个矩阵 $\mathbf{Q}$ “同时对角化”一组矩阵 $\mathbf{G}_1, \dots, \mathbf{G}_k$，如果 $\mathbf{Q}\mathbf{G}_i \mathbf{Q}^{-1}$ 对于所有 $1 \le i \le k$ 都是对角矩阵。注意，文献中有时使用相反的约定\[即 $\mathbf{Q}^{-1} \mathbf{G}_i \mathbf{Q}$ 必须是对角矩阵\]；为了符号上的方便，我们采用前者。）实际上，如果*任何*矩阵同时对角化了所有这些矩阵，那么它就会导出一个有效的分解，我们在 7 的证明中展示了这一点。因此，我们计算某个同时对角化所有这些矩阵的矩阵，并将 $\hat{\mathbf{C}}_1$ 设为该矩阵。

这些思想构成了 [alg:mm_recovery] 的基础，其形式化表述如下。[alg:mm_recovery] 将同时对角化作为一个子程序；我们在下面讨论如何求解同时对角化问题。

块大小 $b$；满足 23 的矩阵 $\mathbf{M}\in \mathcal{M}\mathcal{M}^{*(b,n)}$

将 $\widetilde{\mathbf{M}}_{ij}$（大小为 $\frac{n}{b} \times \frac{n}{b}$）定义为 $\mathbf{P}_{(b,n)} \mathbf{M}\mathbf{P}_{(b,n)}^\top$ 的 $i,j$ 块

计算 $\mathbf{F}(i,j) := \widetilde{\mathbf{M}}_{i1}^{-1}\widetilde{\mathbf{M}}_{ij}\widetilde{\mathbf{M}}_{1j}^{-1}\widetilde{\mathbf{M}}_{11}$

$\hat{\mathbf{C}}_1 \leftarrow \ \textsc{SIMULTANEOUS\_DIAG}\left(\{\mathbf{F}(i,j)\}_{i,j=1,1}^{b,b}\right)$

$\hat{\mathbf{A}}_i \leftarrow \widetilde{\mathbf{M}}_{i1} \hat{\mathbf{C}}_1^{-1}$

$\hat{\mathbf{C}}_j \leftarrow \hat{\mathbf{A}}_1^{-1} \widetilde{\mathbf{M}}_{1j}$

$\hat{\mathbf{D}}_{ij} \leftarrow \hat{\mathbf{A}}_i^{-1} \widetilde{\mathbf{M}}_{ij}\hat{\mathbf{C}}_j^{-1}$

**定理 7**. *给定满足假设 3 的 $n \times n$ 矩阵 $\mathbf{M}\in \mathcal{M}\mathcal{M}^{*(b,n)}$，[alg:mm_recovery] 在 $O\left(\frac{n^3}{b} \right)$ 时间内找到其 Monarch 因子 $\mathbf{L}_1, \mathbf{R}, \mathbf{L}_2$。*

注意，通过设置 $b = \sqrt{n}$，我们立即恢复 2。还要注意，根据 14，7 意味着给定一个 $\mathbf{M}\in \mathcal{M}^*\mathcal{M}^{(\frac{n}{b},n)}$，我们同样可以在 $O(\frac{n^3}{b})$ 时间内找到其 Monarch 分解（例如，简单地将其置换为 $\mathcal{M}\mathcal{M}^{*(b,n)}$ 中的矩阵，然后运行 [alg:mm_recovery]）。我们现在证明 7。

*证明.* 我们首先证明 [alg:mm_recovery] 返回的分解是有效的，这归结为证明如上所述，对于所有 $1 \le i, j \le b$，(1) $\widetilde{\mathbf{M}}_{ij} = \hat{\mathbf{A}}_i \hat{\mathbf{D}}_{ij} \hat{\mathbf{C}}_j$ 且 (2) $\hat{\mathbf{D}}_{ij}$ 是对角矩阵。

如上所述，由于 $\widetilde{\mathbf{M}}$ 满足 23，则存在一个矩阵（$\mathbf{C}_1$）同时对角化所有的 $\mathbf{F}(i,j)$。因此，我们总是可以计算某个同时对角化这些矩阵的矩阵（即，[alg:mm_recovery] 的第 2 行将始终返回一个有效的解）；我们在下面讨论如何实际执行此操作。根据同时对角化的定义，这个矩阵（我们将其设为 $\hat{\mathbf{C}}_1$）是可逆的。

因此，$\hat{\mathbf{A}}_i = \widetilde{\mathbf{M}}_{i1}\hat{\mathbf{C}}_1^{-1}$ 对于所有 $i$ 都是可逆的。因此 $\hat{\mathbf{C}}_j = \hat{\mathbf{A}}_1^{-1} \widetilde{\mathbf{M}}_{1j}$ 对于所有 $j$ 也是可逆的。（注意，等式 $\hat{\mathbf{C}}_j = \hat{\mathbf{A}}_1^{-1} \widetilde{\mathbf{M}}_{1j}$ 对于 $j \ge 2$ 由 $\hat{\mathbf{C}}_j$ 的构造成立，而对于 $j = 1$ 由 $\hat{\mathbf{A}}_1$ 的构造成立。）由于根据定义 $\hat{\mathbf{D}}_{ij} = \hat{\mathbf{A}}_i^{-1} \widetilde{\mathbf{M}}_{ij}\hat{\mathbf{C}}_j^{-1}$，因此我们有对于所有 $i, j$，$\widetilde{\mathbf{M}}_{ij} = \hat{\mathbf{A}}_i \hat{\mathbf{D}}_{ij} \hat{\mathbf{C}}_j$。

剩下需要证明的是 $\hat{\mathbf{D}}_{ij}$ 是对角矩阵。 $$\begin{aligned}
\hat{\mathbf{D}}_{ij} &= \hat{\mathbf{A}}_i^{-1} \widetilde{\mathbf{M}}_{ij}\hat{\mathbf{C}}_j^{-1} \\
&= (\widetilde{\mathbf{M}}_{i1} \hat{\mathbf{C}}_1^{-1})^{-1} \widetilde{\mathbf{M}}_{ij} (\hat{\mathbf{A}}_1^{-1} \widetilde{\mathbf{M}}_{1j})^{-1} \\
&= \hat{\mathbf{C}}_1 \widetilde{\mathbf{M}}_{i1}^{-1} \widetilde{\mathbf{M}}_{ij} \widetilde{\mathbf{M}}_{1j}^{-1} \hat{\mathbf{A}}_1  \\
&= \hat{\mathbf{C}}_1 (\widetilde{\mathbf{M}}_{i1}^{-1} \widetilde{\mathbf{M}}_{ij} \widetilde{\mathbf{M}}_{1j}^{-1} \widetilde{\mathbf{M}}_{11}) \hat{\mathbf{C}}_1^{-1} \\
&= \hat{\mathbf{C}}_1 \mathbf{F}(i, j) \hat{\mathbf{C}}_1^{-1}%
\end{aligned}$$

但是根据 $\hat{\mathbf{C}}_1$ 作为同时对角化 $\mathbf{F}(i,j)$ 的矩阵的*定义*，$\hat{\mathbf{C}}_1 \mathbf{F}(i, j) \hat{\mathbf{C}}_1^{-1}$ 对于所有 $i,j$ 都是对角矩阵。

至于 $\mathbf{L}_1,\mathbf{R},\mathbf{L}_2$，回想一下，我们可以简单地设 $\mathbf{L}_1 = \mathrm{diag}(\hat{\mathbf{A}}_1, \dots, \hat{\mathbf{A}}_b)$，$\mathbf{L}_2 = \mathrm{diag}(\hat{\mathbf{C}}_1, \dots, \hat{\mathbf{C}}_b)$，以及 $\mathbf{R}= \mathbf{P}_{(b,n)}^\top \left(\begin{array}{cccc} \hat{\mathbf{D}}_{11} & \hat{\mathbf{D}}_{12} & \dots & \hat{\mathbf{D}}_{1b} \\ \hat{\mathbf{D}}_{21} & \hat{\mathbf{D}}_{22} & \dots & \hat{\mathbf{D}}_{2b} \\ \ddots & \ddots & \ddots & \ddots  \\ \hat{\mathbf{D}}_{b1} & \hat{\mathbf{D}}_{b2} & \dots & \hat{\mathbf{D}}_{bb} \end{array}\right)
\mathbf{P}_{(b,n)}$，并且如上所述，我们有 $\mathbf{M}= (\mathbf{P}_{(b,n)}^\top \mathbf{L}_1 \mathbf{P}_{(b,n)}) \mathbf{R}(\mathbf{P}_{(b,n)}^\top\mathbf{L}_2\mathbf{P}_{(b,n)})$，其中 $\mathbf{L}_1, \mathbf{L}_2 \in \mathcal{B}\mathcal{D}^{(\frac{n}{b}, n)}$ 且 $\mathbf{R}\in \mathcal{B}\mathcal{D}^{(b,n)}$。这就完成了正确性的证明。

现在，我们分析运行时间。有 $b^2$ 个矩阵 $\mathbb{F}(i,j)$ 需要计算，计算每个矩阵需要 $O(\frac{n^3}{b^3})$ 时间。一旦我们找到了 $\hat{\mathbf{C}}_1$，就有 $b$ 个矩阵 $\hat{\mathbf{A}}_i$ 需要计算，每个耗时 $O(\frac{n^3}{b^3})$；有 $b-1$ 个矩阵 $\hat{\mathbf{C}}_j$（对于 $j \ge 2$）需要计算，每个耗时 $O(\frac{n^3}{b^3})$；然后有 $b^2$ 个矩阵 $\hat{\mathbf{D}}_{ij}$ 需要计算，每个耗时 $O(\frac{n^3}{b^3})$。（注意，我们可以使用快速矩阵乘法/求逆更快地计算其中的每一个；然而，事实证明这并不重要，因为同时对角化是瓶颈。）

最后，我们分析同时对角化的运行时间。一组矩阵 $\{\mathbf{G}_1, \dots, \mathbf{G}_k\}$ 的同时对角化等价于为这些矩阵寻找一个共同的特征基，因为如果 $\mathbf{D}_i$ 是一个对角矩阵且 $\mathbf{Q}\mathbf{G}_i \mathbf{Q}^{-1} = \mathbf{D}_i$，那么 $\mathbf{Q}$ 的第 $j^{th}$ 列就是 $\mathbf{G}_i$ 的特征向量，其特征值等于 $\mathbf{D}_i$ 的第 $j^{th}$ 个元素。

假设一组矩阵实际上是可同时对角化的（这意味着每个矩阵单独也是可对角化的），对其进行同时对角化的一个简单算法如下（例如参见 \[Conrad_theminimal, gerstner1993numerical\]）：首先，设 $i = 1$ 并对角化第一个矩阵 $\mathbf{G}_i = \mathbf{G}_1$（即找到一组特征基），并设 $\mathbf{Q}$ 为对角化矩阵（即特征向量矩阵）。因此，$\mathbf{Q}\mathbf{G}_1 \mathbf{Q}^{-1}$ 是对角矩阵。根据矩阵实际上是可同时对角化的这一假设，对于所有 $j \ne i$，$\mathbf{Q}\mathbf{G}_j \mathbf{Q}^{-1}$ 也将是置换块对角矩阵：每个块的大小对应于 $\mathbf{G}_1$ 相应特征值的重数。（注意，如果 $\mathbf{G}_1$ 具有唯一的特征值，那么特征基是唯一的（至多相差排列和非零缩放），因此在这种情况下，$\mathbf{G}_1$ 唯一地确定同时对角化矩阵，至多相差行的任意排列和非零缩放。换句话说，在这种情况下块大小将为 1，意味着对于所有 $j$，$\mathbf{Q}\mathbf{G}_j \mathbf{Q}^{-1}$ 将是对角矩阵，我们就完成了。）

现在，我们对直到 $k$ 的所有 $i$ 重复以下操作。递增 $i$ 并计算 $\mathbf{Q}\mathbf{G}_i \mathbf{Q}^{-1}$。如果它已经是对角矩阵，则继续下一步。否则，首先置换 $\mathbf{Q}\leftarrow \mathbf{P}\mathbf{Q}\mathbf{P}^\top$ 使其成为块对角矩阵（注意到这维持了对于所有 $j < i$，$\mathbf{Q}\mathbf{G}_j \mathbf{Q}^{-1}$ 是对角矩阵的性质，因为对于任何置换矩阵 $\mathbf{P}$ 和对角矩阵 $\mathbf{D}$，$\mathbf{P}\mathbf{D}\mathbf{P}^\top$ 都是对角矩阵）。然后对于每个大小 $> 1$ 的块，计算一个对该块进行对角化的矩阵；将块的数量（包括大小为 1 的块）记为 $b$，令 $\mathbf{Q}_1', \dots, \mathbf{Q}_b'$ 表示相应的对角化变换，当块大小为 1 时则记为标量 1。最后设 $\mathbf{Q}' \leftarrow \mathrm{diag}(\mathbf{Q}_1', \dots, \mathbf{Q}_b')$ 且 $\mathbf{Q}\leftarrow \mathbf{Q}'^{-1} \mathbf{Q}\mathbf{Q}'$。根据构造，$\mathbf{Q}\mathbf{G}_i \mathbf{Q}^{-1}$ 现在将是对角矩阵；同时，对于所有 $j < i$，$\mathbf{Q}\mathbf{G}_j \mathbf{Q}^{-1}$ 仍然是对角矩阵，因为可对角化矩阵对应于重特征值 $\lambda$ 的一组特征向量的任何线性组合本身就是该矩阵具有特征值 $\lambda$ 的特征向量。

因此，一旦我们处理了所有 $k$ 个 $\mathbf{G}_i$，$\mathbf{Q}$ 就是一个同时对角化所有这些矩阵的矩阵。在每一步 $i$，我们计算大小总和为 $n$ 的方块矩阵的对角化变换。由于对于 $n \times n$ 矩阵，特征分解（对于固定的期望精度）需要 $O(n^3)$ 的时间，这意味着步骤 $i$ 的总运行时间为 $O\left(\sum_{j=1}^{k} s_i^3 \right)\le O(n^3)$。因此，整个同时对角化过程的总运行时间为 $O(kn^3)$，其中 $k$ 是矩阵的数量。（注意，同时也存在同时对角化的迭代方法 \[gerstner1993numerical,akema2020approximate\]，在实践中可用于加速此步骤。）

将此应用于我们的问题，我们有 $b^2$ 个矩阵需要同时对角化，每个矩阵的大小为 $\frac{n}{b} \times \frac{n}{b}$。这导致整个同时对角化过程的总运行时间为 $O\left(b^2 \cdot (\frac{n}{b})^3\right)= O\left(\frac{n^3}{b}\right)$，因此 [alg:mm_recovery] 的运行时间也是 $O\left(\frac{n^3}{b}\right)$，符合预期。

（注意：从上述分析可以看出，我们实际上不需要 $\mathbf{M}$ 本身可逆——我们只需要它的所有块 $\widetilde{\mathbf{M}}_{ij}$ 可逆，从而所有的 $\mathbf{A}_i$ 和 $\mathbf{C}_j$ 可逆即可，鉴于我们已经由于对 $\mathbf{R}$ 的块的非零假设而假设 $\mathbf{D}_{ij}$ 是可逆的，这是一个比 $\mathbf{M}$ 的可逆性更弱的假设。） ◻

# 实验细节

## 模型配置与超参数

我们在下面总结了复现我们的实验所需的细节。

### 图像分类

**基线模型：** 对于稠密模型，我们使用来自 `timm` 库和 T2T-ViT 代码库 \[yuan2021tokens\] 的 ViT \[dosovitskiy2020image\]、MLP-Mixertolstikhin2021mlp 的标准实现。

这些模型的 Monarch 版本简单地将 attention 块（投影矩阵）和 FFN 块（线性层）中的稠密权重矩阵替换为 Monarch 矩阵。我们将块对角矩阵中的块数设置为 4。我们还减少了正则化（随机深度）的量，因为我们的 Monarch 模型比稠密模型更小。

我们采用了 \[yuan2021tokens\] 中的超参数（优化器、学习率、学习率调度器）。详细信息见 \[table:imagenet_hparams\]。

我们在 V100 GPU 上测量实际训练时间。

我们遵循 Vision Transformer 论文和 MLP-Mixer 论文中的命名约定。具体而言，ViT-S 和 ViT-B 分别指代小型和基础 ViT 模型，16 指代 16x16 的 patch 大小。MLP-Mixer 模型遵循相同的约定。

### 语言建模

对于稠密模型，我们使用来自 Huggingface `transformers` 库和 Nvidia 的 Megatron-LM 仓库的 GPT-2 \[radford2019language\] 的标准实现。我们遵循 Megatron-LM 仓库的训练方案。

这些模型的 Monarch 版本简单地将 attention 块（投影矩阵）和 FFN 块（线性层）中的稠密权重矩阵替换为 Monarch 矩阵。我们将块对角矩阵中的块数设置为 4。由于我们的模型更小，我们也降低了正则化强度。

我们在 \[table:wt103\] 和 \[table:owt\] 中报告了所使用的超参数。我们使用 512 的有效 batch size，并使用梯度累积以适应可用的 GPU 内存。

我们在 V100 GPU 上测量实际训练时间。

## PDE 求解细节

我们采用了 FNO \[li2020fourier\] 中 Navier-Stokes 方程的实验设置和数据生成。它考虑了在单位环面上的涡量形式的粘性不可压缩流体的二维 Navier-Stokes 方程： $$\begin{aligned}
    \partial_{t} w(x, t) + u(x, t) \cdot \nabla w(x, t) & = v \Delta w(x, t) + f(x), & x \in (0, 1)^2, t \in (0, T] \\
    \nabla w(x, t) & = 0, & x \in (0, 1)^2, t \in (0, T] \\
    w(x, 0) & = w_0(x), & x \in (0, 1)^2 \\
\end{aligned}$$ 其中对于任意 $r>0$，$u \in C([, T0])$;$H_{per}((0, 1)^2; \mathbb{R}^2))$ 是速度场，$w=\nabla \times u$ 是涡量，$w_0 \in L^2_{per}((0, 1)^2; \mathbb{R})$ 是初始涡量，$v \in \mathbb{R_{+}}$ 是粘性系数，$f \in L_{per}^2((0, 1)^2; \mathbb{R})$ 是强迫函数。$T$ 表示时间间隔，因为它是时间依赖方程。$v$ 表示粘度。N 表示训练对或数据的数量。3 分别展示了粘度 $v=1e-3, 1e-4, 1e-5$，$T=50, 30, 20$ 的结果，并使用 $N=1000$。

## GPT-2 下游任务细节

我们在更大规模的数据集 OpenWebText 上训练 Pixelfly-GPT2-small，并在来自 \[zhao2021calibrate\] 的零样本生成和分类任务上评估下游质量，取得了与稠密模型相当甚至更好的性能。具体而言，这些数据集包含五个流行的分类任务：SST2、Trec、CB、Agnews 和 Dbpedia。我们还调整了来自 \[zhao2021calibrate\] 的校准指标进行评估。每个单独任务的结果如 \[table:gpt_finetune_full\] 所示。

## BERT 预训练的详细信息

我们遵循 Nvidia Deep Learning examples 参考实现（<https://github.com/NVIDIA/DeepLearningExamples>）的训练流程和超参数。具体而言，我们使用 LAMB 优化器，学习率为 4e-3。我们使用尽可能大的、仍能容纳在 GPU 内存（A100-40GB）中的小批量大小，并使用梯度累积在阶段 1（最大序列长度 128）达到 64k 序列的有效批量大小，在阶段 2（最大序列长度 512）达到 32k 的有效批量大小。我们以混合精度（fp16 和 fp32）进行训练。

我们使用了 Nvidia 在 MLPerf 1.1 中的 BERT 实现的所有优化：

1.  仅计算 masked tokens 的预测分数（最后一层），因为其他 tokens 的输出不用于计算掩码语言建模损失。

2.  移除 padding tokens，仅计算非 padding tokens 的 attention。

3.  使用融合的 CUDA kernel (FMHA)，将 4 个步骤合并为一个 kernel：计算 $Q K^T$，取 softmax，应用 dropout，乘以 $V$，其中 $Q, K, V$ 分别是 query、key 和 value。

4.  在前馈网络 (FFN) 层中，将矩阵乘法和加偏置融合到一个 CUDA kernel 中。偏置的梯度也在反向传播中与矩阵乘法融合。

5.  在 attention 输出投影中，将矩阵乘法和加偏置融合到一个 CUDA kernel 中。

6.  在 attention 和 FFN 块末尾的残差连接中，融合 dropout 和加残差。

我们使用 DeepSpeed \[rasley2020deepspeed\] ZeRO 优化器的 stage 1 进行训练以分片优化器状态，从而减少 GPU 内存使用并允许我们使用更大的批量大小。对于 Nvidia MLPerf 实现，我们报告了 Apex 的自动混合精度 (AMP) 级别 O2（如原始实现中所示）和 DeepSpeed ZeRO 优化器的速度。

## 加速多线圈 MRI 重建

### 背景

在多线圈 MRI 中，多个接收线圈（即传感器）在空间频率（又称 *k-space*）域中获取复数值测量结果。这些测量结果受空间变化的灵敏度图调制，该图表征了每个线圈对成像目标的灵敏度。在加速 MRI 中，通过减少在 k-space 中获取的样本数量来缩短扫描时间。由于数据的采样率低于奈奎斯特率，重建底层图像是一个不适定问题。

加速多线圈 MRI 的正向问题可以写成矩阵方程 $$y = \Omega\boldsymbol{F}\boldsymbol{S}x + \epsilon$$ 其中 $\Omega$ 是在 k-space 中索引获取样本的二元欠采样掩码，$y$ 是 k-space 中的向量化测量信号，$\boldsymbol{F}$ 是离散傅里叶变换矩阵，$\boldsymbol{S}$ 是接收线圈灵敏度图，$x$ 是图像空间中的真实信号，$\epsilon$ 是加性复高斯噪声。加速因子由 $R = \frac{\sum_i^{|N|} \Omega_i}{|\Omega|}$ 给出。

### 实验细节

#### 数据集。

我们在 SKM-TEA Raw Data Track 上对我们的方法进行基准测试，该数据集由双回波 3D MRI 扫描组成 \[desai2021skm\]。使用随数据集分发的泊松盘欠采样掩码对扫描进行加速。在训练期间，生成、缓存泊松盘掩码，并将其应用于掩蔽 k-space 数据以模拟加速扫描。

#### 矩阵形状。

与所有矩阵一样，Monarch 矩阵具有显式的形状约束，这是这些矩阵在 MRI 重建任务中的局限性。因此，对 SKM-TEA 数据集进行了过滤，以包含形状为 $512 \times 512 \times 160$ 的扫描，这是最常出现的扫描形状。从数据集最初的 155 次扫描中总共去除了 3 次扫描。我们的方法和所有基线均在此过滤后的数据集上进行训练。

#### 基线。

我们将我们的方法与两个基线 SENSE 和 U-Net 进行比较。参数数量和超参数可在表 [table:skmtea-config] 中找到。

- *SENSE*：SENSE 对在每个线圈上获取的图像执行线性组合 \[pruessmann1999sense\]。在此，对每个线圈的获取 k-space 应用快速傅里叶逆变换 (IFFT)。生成的图像通过用相应的线圈灵敏度图对每个线圈图像进行加权，组合成单个复数图像。在加速 MRI 中，未采样的频率值为零；因此，SENSE 生成*零填充图像*。请注意，SENSE 不需要任何训练。

- *U-Net*：U-Net 是用于 MRI 重建的流行的全卷积神经网络基线 \[ronneberger2015u\]。我们使用 \[desai2021skm\] 用于 SKM-TEA 数据集基准测试的默认实现和超参数。在这种方法中，SENSE 重建的零填充图像被映射到 SENSE 重建的真实图像。

#### Monarch-SENSE (mSENSE)：

我们提出了一种对 SENSE 方法的修改，其中 (IFFT) 由分解的 Monarch 矩阵参数化。该矩阵被初始化为 IFFT，但与 SENSE 不同的是，它是可学习的。虽然 mSENSE 是可训练的，但它的可训练参数比 U-Net 少 137 倍。

#### 指标：

我们分别使用峰值信噪比 (pSNR) 和结构相似性 (SSIM) 对两个回波（echo1 - E1，echo2 - E2）的重建性能进行评估。这两个指标都是在每个回波的 3D 体积上计算的。

#### 扩展结果。

我们在数据受限环境下提供了 SENSE、mSENSE 和 U-Net 在第一个（图 6）和第二个（图 7）回波的样本重建结果。SENSE 和 U-Net 重建的图像都有混叠伪影。由于随机的泊松盘欠采样模式，这些伪影是不相干的，导致它们在精细结构和边缘周围表现为模糊。相比之下，mSENSE 可以以更高的保真度恢复这些结构。即使在信噪比 (SNR) 低于第一个回波的第二个回波中，mSENSE 也不会对图像造成过度模糊。

使用 SENSE、Monarch-SENSE (mSENSE) 和 U-Net 在 SKM-TEA 数据集中对第一个回波进行 2 倍加速的样本重建。mSENSE 和 U-Net 均使用 1 次训练扫描进行训练。SENSE 是一种未经训练的方法。

使用 SENSE、Monarch SENSE (mSENSE) 和 U-Net 在 SKM-TEA 数据集中对第二个回波进行 2 倍加速的样本重建。mSENSE 和 U-Net 均使用 1 次训练扫描进行训练。SENSE 是一种未经训练的方法。

## References

- Ailon, N., Leibovitch, O., and Nair, V. Sparse linear networks with a fixed butterfly structure: theory and practice. In Uncertainty in Artificial Intelligence, pp.\ 1174--1184. PMLR, 2021.
- Akema, R., Yamagishi, M., and Yamada, I. Approximate simultaneous diagonalization of matrices via structured low-rank approximation. arXiv preprint arXiv:2010.06305, 2020.
- Bailey, D. H. {FFT}s in external or hierarchical memory. The journal of Supercomputing, 4\penalty0 (1):\penalty0 23--35, 1990.
- Bunse-Gerstner, A., Byers, R., and Mehrmann, V. Numerical methods for simultaneous diagonalization. SIAM Journal on Matrix Analysis and Applications, 1993.
- Chaudhari, A. S., Sandino, C. M., Cole, E. K., Larson, D. B., Gold, G. E., Vasanawala, S. S., Lungren, M. P., Hargreaves, B. A., and Langlotz, C. P. Prospective deployment of deep learning in {MRI}: A framework for important considerations, challenges, and recommendations for best practices. Journal of Magnetic Resonance Imaging, 2020.
- Chen, B., Dao, T., Liang, K., Yang, J., Song, Z., Rudra, A., and R{\'e}, C. Pixelated butterfly: Simple and efficient sparse training for neural network models. In International Conference on Learning Representations (ICLR), 2022.
- Child, R., Gray, S., Radford, A., and Sutskever, I. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.
- Choromanski, K., Rowland, M., Chen, W., and Weller, A. Unifying orthogonal {M}onte {C}arlo methods. In International Conference on Machine Learning, pp.\ 1203--1212, 2019.
- Cole, E. K., Pauly, J. M., Vasanawala, S. S., and Ong, F. Unsupervised {MRI} reconstruction with generative adversarial networks. arXiv preprint arXiv:2008.13065, 2020.
- Conrad, K. The minimal polynomial and some applications.
- Cooley, J. W. and Tukey, J. W. An algorithm for the machine calculation of complex fourier series. Mathematics of computation, 19\penalty0 (90):\penalty0 297--301, 1965.
- Dao, T., Gu, A., Eichhorn, M., Rudra, A., and R{\'e}, C. Learning fast algorithms for linear transforms using butterfly factorizations. In International Conference on Machine Learning (ICML), 2019.
- Dao, T., Sohoni, N., Gu, A., Eichhorn, M., Blonder, A., Leszczynski, M., Rudra, A., and R{\'e}, C. Kaleidoscope: An efficient, learnable representation for all structured linear maps. In International Conference on Learning Representations (ICLR), 2020.
- Darestani, M. Z. and Heckel, R. Accelerated {MRI} with un-trained neural networks. IEEE Transactions on Computational Imaging, 7:\penalty0 724--733, 2021.
- Darestani, M. Z., Chaudhari, A., and Heckel, R. Measuring robustness in deep learning based compressive sensing. arXiv preprint arXiv:2102.06103, 2021.
- De Sa, C., Gu, A., Puttagunta, R., R{\'e}, C., and Rudra, A. A two-pronged progress in structured dense matrix vector multiplication. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms, pp.\ 1060--1079. SIAM, 2018.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp.\ 248--255. Ieee, 2009.
- Desai, A. D., Gunel, B., Ozturkler, B. M., Beg, H., Vasanawala, S., Hargreaves, B. A., R{\'e}, C., Pauly, J. M., and Chaudhari, A. S. Vortex: Physics-driven data augmentations for consistency training for robust accelerated {MRI} reconstruction. arXiv preprint arXiv:2111.02549, 2021{ {a}}.
- Desai, A. D., Ozturkler, B. M., Sandino, C. M., Vasanawala, S., Hargreaves, B. A., Re, C. M., Pauly, J. M., and Chaudhari, A. S. Noise2recon: A semi-supervised framework for joint {MRI} reconstruction and denoising. arXiv preprint arXiv:2110.00075, 2021{ {b}}.
- Desai, A. D., Schmidt, A. M., Rubin, E. B., Sandino, C. M., Black, M. S., Mazzoli, V., Stevens, K. J., Boutin, R., Re, C., Gold, G. E., et al. {SKM-TEA}: A dataset for accelerated {MRI} reconstruction with dense image labels for quantitative clinical evaluation. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021{ {c}}.
- Dettmers, T. and Zettlemoyer, L. Sparse networks from scratch: Faster training without losing performance. arXiv preprint arXiv:1907.04840, 2019.
- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
- Dong, X., Chen, S., and Pan, S. J. Learning to prune deep neural networks via layer-wise optimal brain surgeon. arXiv preprint arXiv:1705.07565, 2017.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- Driscoll, J. R., Healy Jr, D. M., and Rockmore, D. N. Fast discrete polynomial transforms with applications to data analysis for distance transitive graphs. SIAM Journal on Computing, 26\penalty0 (4):\penalty0 1066--1099, 1997.
- Eckart, C. and Young, G. The approximation of one matrix by another of lower rank. Psychometrika, 1\penalty0 (3):\penalty0 211--218, 1936.
- Eidelman, Y. and Gohberg, I. On a new class of structured matrices. Integral Equations and Operator Theory, 34\penalty0 (3):\penalty0 293--324, 1999.
- Evci, U., Pedregosa, F., Gomez, A., and Elsen, E. The difficulty of training sparse neural networks. arXiv preprint arXiv:1906.10732, 2019.
- Fan, T., Xu, K., Pathak, J., and Darve, E. Solving inverse problems in steady-state navier-stokes equations using deep neural networks. arXiv preprint arXiv:2008.13074, 2020.
- Frankle, J. and Carbin, M. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635, 2018.
- Frankle, J., Dziugaite, G. K., Roy, D. M., and Carbin, M. Stabilizing the lottery ticket hypothesis. arXiv preprint arXiv:1903.01611, 2019.
- Frankle, J., Dziugaite, G. K., Roy, D., and Carbin, M. Linear mode connectivity and the lottery ticket hypothesis. In International Conference on Machine Learning, pp.\ 3259--3269. PMLR, 2020.
- Gale, T., Elsen, E., and Hooker, S. The state of sparsity in deep neural networks. arXiv preprint arXiv:1902.09574, 2019.
- Gao, L., Tow, J., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., McDonell, K., Muennighoff, N., Phang, J., Reynolds, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for few-shot language model evaluation, September 2021. URL https://doi.org/10.5281/zenodo.5371628.
- Geva, M., Schuster, R., Berant, J., and Levy, O. Transformer feed-forward layers are key-value memories. arXiv preprint arXiv:2012.14913, 2020.
- Gokaslan, A., Cohen, V., Ellie, P., and Tellex, S. Openwebtext corpus, 2019.
- Gray, R. M. Toeplitz and circulant matrices: A review. {Foundations and Trends{ } in Communications and Information Theory}, 2\penalty0 (3):\penalty0 155--239, 2006.
- Gray, S., Radford, A., and Kingma, D. P. {GPU} kernels for block-sparse weights. arXiv preprint arXiv:1711.09224, 3, 2017.
- Griswold, M. A., Jakob, P. M., Heidemann, R. M., Nittka, M., Jellus, V., Wang, J., Kiefer, B., and Haase, A. Generalized autocalibrating partially parallel acquisitions (grappa). Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 47\penalty0 (6):\penalty0 1202--1210, 2002.
- Gu, A., Dao, T., Ermon, S., Rudra, A., and R{\'e}, C. Hippo: Recurrent memory with optimal polynomial projections. In Advances in neural information processing systems (NeurIPS), 2020.
- Guo, C., Hsueh, B. Y., Leng, J., Qiu, Y., Guan, Y., Wang, Z., Jia, X., Li, X., Guo, M., and Zhu, Y. Accelerating sparse dnn models without hardware-support via tile-wise sparsity. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp.\ 1--15. IEEE, 2020.
- Haldar, J. P. Low-rank modeling of local $ k $-space neighborhoods (loraks) for constrained {MRI}. IEEE transactions on medical imaging, 33\penalty0 (3):\penalty0 668--681, 2013.
- Hammernik, K., Klatzer, T., Kobler, E., Recht, M. P., Sodickson, D. K., Pock, T., and Knoll, F. Learning a variational network for reconstruction of accelerated {MRI} data. Magnetic resonance in medicine, 79\penalty0 (6):\penalty0 3055--3071, 2018.
- Han, S., Mao, H., and Dally, W. J. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149, 2015{ {a}}.
- Han, S., Pool, J., Tran, J., and Dally, W. J. Learning both weights and connections for efficient neural networks. arXiv preprint arXiv:1506.02626, 2015{ {b}}.
- Han, S., Pool, J., Narang, S., Mao, H., Gong, E., Tang, S., Elsen, E., Vajda, P., Paluri, M., Tran, J., et al. Dsd: Dense-sparse-dense training for deep neural networks. arXiv preprint arXiv:1607.04381, 2016.
- He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp.\ 770--778, 2016.
- Hooker, S. The hardware lottery. arXiv preprint arXiv:2009.06489, 2020.
- Hsieh, J. Computed tomography: principles, design, artifacts, and recent advances, volume 114. SPIE press, 2003.
- Jayakumar, S. M., Pascanu, R., Rae, J. W., Osindero, S., and Elsen, E. Top-{KAST}: Top-{K} always sparse training. arXiv preprint arXiv:2106.03517, 2021.
- Jolicoeur-Martineau, A., Li, K., Pich{\'e}-Taillefer, R., Kachman, T., and Mitliagkas, I. Gotta go fast when generating data with score-based models. arXiv preprint arXiv:2105.14080, 2021.
- Jurafsky, D. and Martin, J. H. Speech and language processing, volume 3. Pearson London, 2014.
- Kailath, T., Kung, S.-Y., and Morf, M. Displacement ranks of matrices and linear equations. Journal of Mathematical Analysis and Applications, 68\penalty0 (2):\penalty0 395--407, 1979.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- Khalitov, R., Yu, T., Cheng, L., and Yang, Z. Sparse factorization of large square matrices. arXiv preprint arXiv:2109.08184, 2021.
- Kidger, P., Morrill, J., Foster, J., and Lyons, T. Neural controlled differential equations for irregular time series. arXiv preprint arXiv:2005.08926, 2020.
- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.
- Knoll, F., Hammernik, K., Zhang, C., Moeller, S., Pock, T., Sodickson, D. K., and Akcakaya, M. Deep-learning methods for parallel magnetic resonance imaging reconstruction: A survey of the current approaches, trends, and issues. IEEE signal processing magazine, 37\penalty0 (1):\penalty0 128--140, 2020.
- Kochkov, D., Smith, J. A., Alieva, A., Wang, Q., Brenner, M. P., and Hoyer, S. Machine learning--accelerated computational fluid dynamics. Proceedings of the National Academy of Sciences, 118\penalty0 (21), 2021.
- Lagunas, F., Charlaix, E., Sanh, V., and Rush, A. M. Block pruning for faster transformers. arXiv preprint arXiv:2109.04838, 2021.
- Lahiri, A., Wang, G., Ravishankar, S., and Fessler, J. A. Blind primed supervised (blips) learning for mr image reconstruction. arXiv preprint arXiv:2104.05028, 2021.
- Le, Q., Sarl{\'o}s, T., and Smola, A. Fastfood-computing hilbert space expansions in loglinear time. In International Conference on Machine Learning, pp.\ 244--252, 2013.
- Le Magoarou, L. and Gribonval, R. Flexible multilayer sparse approximations of matrices and applications. IEEE Journal of Selected Topics in Signal Processing, 10\penalty0 (4):\penalty0 688--700, 2016.
- Li, H., Kadav, A., Durdanovic, I., Samet, H., and Graf, H. P. Pruning filters for efficient convnets. arXiv preprint arXiv:1608.08710, 2016.
- Li, Z., Kovachki, N. B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A., et al. Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations, 2020.
- Lin, J., Rao, Y., Lu, J., and Zhou, J. Runtime neural pruning. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.
- Lin, R., Ran, J., Chiu, K. H., Chesi, G., and Wong, N. Deformable butterfly: A highly structured and sparse linear transform. Advances in Neural Information Processing Systems, 34, 2021.
- Liu, T. and Zenke, F. Finding trainable sparse networks through neural tangent transfer. In International Conference on Machine Learning, pp.\ 6336--6347. PMLR, 2020.
- Lustig, M., Donoho, D., and Pauly, J. M. Sparse {MRI}: The application of compressed sensing for rapid mr imaging. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 58\penalty0 (6):\penalty0 1182--1195, 2007.
- Mardani, M., Gong, E., Cheng, J. Y., Vasanawala, S. S., Zaharchuk, G., Xing, L., and Pauly, J. M. Deep generative adversarial neural networks for compressive sensing {MRI}. IEEE transactions on medical imaging, 38\penalty0 (1):\penalty0 167--179, 2018.
- Massaroli, S., Poli, M., Sonoda, S., Suzuki, T., Park, J., Yamashita, A., and Asama, H. Differentiable multiple shooting layers. arXiv preprint arXiv:2106.03885, 2021.
- Mattson, P., Cheng, C., Diamos, G., Coleman, C., Micikevicius, P., Patterson, D., Tang, H., Wei, G.-Y., Bailis, P., Bittorf, V., et al. Mlperf training benchmark. Proceedings of Machine Learning and Systems, 2:\penalty0 336--349, 2020.
- Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.
- Moczulski, M., Denil, M., Appleyard, J., and de Freitas, N. {ACDC: a structured efficient linear layer}. In International Conference on Learning Representations, 2016.
- Morcos, A. S., Yu, H., Paganini, M., and Tian, Y. One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers. arXiv preprint arXiv:1906.02773, 2019.
- Munkhoeva, M., Kapushev, Y., Burnaev, E., and Oseledets, I. Quadrature-based features for kernel approximation. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 31, pp.\ 9165--9174. Curran Associates, Inc., 2018.
- Ong, F. and Lustig, M. Beyond low rank+ sparse: Multiscale low rank matrix decomposition. IEEE journal of selected topics in signal processing, 10\penalty0 (4):\penalty0 672--687, 2016.
- Orseau, L., Hutter, M., and Rivasplata, O. Logarithmic pruning is all you need. Advances in Neural Information Processing Systems, 33, 2020.
- Pan, V. Y. Structured matrices and polynomials: unified superfast algorithms. Springer Science & Business Media, 2012.
- Parker, D. S. Random butterfly transformations with applications in computational linear algebra. 1995.
- Pensia, A., Rajput, S., Nagle, A., Vishwakarma, H., and Papailiopoulos, D. Optimal lottery tickets via subsetsum: Logarithmic over-parameterization is sufficient. arXiv preprint arXiv:2006.07990, 2020.
- Peste, A., Iofinova, E., Vladu, A., and Alistarh, D. Ac/dc: Alternating compressed/decompressed training of deep neural networks. Advances in Neural Information Processing Systems, 34, 2021.
- Poli, M., Massaroli, S., Yamashita, A., Asama, H., Park, J., et al. Hypersolvers: Toward fast continuous-depth models. Advances in Neural Information Processing Systems, 33, 2020.
- Pruessmann, K. P., Weiger, M., Scheidegger, M. B., and Boesiger, P. Sense: sensitivity encoding for fast {MRI}. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 42\penalty0 (5):\penalty0 952--962, 1999.
- Rackauckas, C., Ma, Y., Martensen, J., Warner, C., Zubov, K., Supekar, R., Skinner, D., Ramadhan, A., and Edelman, A. Universal differential equations for scientific machine learning. arXiv preprint arXiv:2001.04385, 2020.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1\penalty0 (8):\penalty0 9, 2019.
- Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:\penalty0 686--707, 2019.
- Rasley, J., Rajbhandari, S., Ruwase, O., and He, Y. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp.\ 3505--3506, 2020.
- Ravishankar, S., Moore, B. E., Nadakuditi, R. R., and Fessler, J. A. Low-rank and adaptive sparse signal (lassi) models for highly accelerated dynamic imaging. IEEE transactions on medical imaging, 36\penalty0 (5):\penalty0 1116--1128, 2017.
- Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pp.\ 234--241. Springer, 2015.
- Sandino, C. M., Cheng, J. Y., Chen, F., Mardani, M., Pauly, J. M., and Vasanawala, S. S. Compressed sensing: From research to clinical practice with deep neural networks: Shortening scan times for magnetic resonance imaging. IEEE signal processing magazine, 37\penalty0 (1):\penalty0 117--127, 2020.
- Sanh, V., Wolf, T., and Rush, A. M. Movement pruning: Adaptive sparsity by fine-tuning. arXiv preprint arXiv:2005.07683, 2020.
- Sch{\"o}nemann, P. H. A generalized solution of the orthogonal procrustes problem. Psychometrika, 31\penalty0 (1):\penalty0 1--10, 1966.
- Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., and Catanzaro, B. Megatron-{LM}: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.
- Sindhwani, V., Sainath, T., and Kumar, S. Structured transforms for small-footprint deep learning. In Advances in Neural Information Processing Systems, pp.\ 3088--3096, 2015.
- Tanaka, H., Kunin, D., Yamins, D. L., and Ganguli, S. Pruning neural networks without any data by iteratively conserving synaptic flow. arXiv preprint arXiv:2006.05467, 2020.
- Tewarson, R. P. Sparse matrices, volume 69. Academic Press New York, 1973.
- Thomas, A., Gu, A., Dao, T., Rudra, A., and R{\'e}, C. Learning compressed transforms with low displacement rank. In Advances in neural information processing systems, pp.\ 9052--9060, 2018.
- Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., Yung, J., Keysers, D., Uszkoreit, J., Lucic, M., et al. Mlp-{M}ixer: An all-mlp architecture for vision. arXiv preprint arXiv:2105.01601, 2021.
- Trefethen, L. N. Spectral methods in MATLAB. SIAM, 2000.
- Vahid, K. A., Prabhu, A., Farhadi, A., and Rastegari, M. Butterfly transform: An efficient fft based neural architecture design. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp.\ 12021--12030. IEEE, 2020.
- Wang, C., Zhang, G., and Grosse, R. Picking winning tickets before training by preserving gradient flow. arXiv preprint arXiv:2002.07376, 2020{ {a}}.
- Wang, R., Kashinath, K., Mustafa, M., Albert, A., and Yu, R. Towards physics-informed deep learning for turbulent flow prediction. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp.\ 1457--1466, 2020{ {b}}.
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Scao, T. L., Gugger, S., Drame, M., Lhoest, Q., and Rush, A. M. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp.\ 38--45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.emnlp-demos.6.
- Yaman, B., Hosseini, S. A. H., Moeller, S., Ellermann, J., U{ {g}}urbil, K., and Ak{ {c}}akaya, M. Self-supervised physics-based deep learning {MRI} reconstruction without fully-sampled data. In 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), pp.\ 921--925. IEEE, 2020.
- Yu, F. X., Suresh, A. T., Choromanski, K. M., Holtmann-Rice, D. N., and Kumar, S. Orthogonal random features. In Lee, D. D., Sugiyama, M., Luxburg, U. V., Guyon, I., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 29, pp.\ 1975--1983. Curran Associates, Inc., 2016.
- Yuan, L., Chen, Y., Wang, T., Yu, W., Shi, Y., Tay, F. E., Feng, J., and Yan, S. Tokens-to-token {V}i{T}: Training vision transformers from scratch on imagenet. arXiv preprint arXiv:2101.11986, 2021.
- Zhao, T. Z., Wallace, E., Feng, S., Klein, D., and Singh, S. Calibrate before use: Improving few-shot performance of language models. arXiv preprint arXiv:2102.09690, 2021.
- Zhu, M. and Gupta, S. To prune, or not to prune: exploring the efficacy of pruning for model compression. arXiv preprint arXiv:1710.01878, 2017.
- Zhu, Y., Kiros, R., Zemel, R., Salakhutdinov, R., Urtasun, R., Torralba, A., and Fidler, S. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pp.\ 19--27, 2015.
