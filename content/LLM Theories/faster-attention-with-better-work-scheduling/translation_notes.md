# : Faster Attention with Better Work Scheduling 原文翻译

# : 通过更好的工作调度实现更快的 Attention

# 背景

我们提供一些关于现代硬件（GPU）上常见深度学习操作性能特征的背景知识。我们还将描述 attention 的标准实现，以及 FlashAttention。

## 硬件性能

我们在此关注 GPU。其他硬件加速器上的性能类似 (Jouppi et al. 2017; Jia et al. 2019)。

**GPU 内存层次结构。** GPU 内存层次结构 ([fig:banner] left) 包含多种不同大小和速度的内存形式，较小的内存速度更快。例如，A100 GPU 具有 40-80GB 的高带宽内存 (HBM)，带宽为 1.5-2.0TB/s，以及 108 个流多处理器每个拥有 192KB 的片上 SRAM，带宽估计约为 19TB/s (Jia et al. 2018; Jia and Van Sandt 2021)。片上 SRAM 比 HBM 快一个数量级，但在大小上小几个数量级。随着计算速度相对于内存速度变得更快 (NVIDIA 2017, 2020, 2022)，操作越来越受到内存 (HBM) 访问的瓶颈限制。因此，利用快速的 SRAM 变得更加重要。

**执行模型。** GPU 拥有大量线程来执行一个操作（称为 kernel）。每个 kernel 从 HBM 加载输入到寄存器和 SRAM，进行计算，然后将输出写入 HBM。

**性能特征。** 根据计算和内存访问的平衡情况，操作可以分为计算受限或内存受限。这通常通过*算术强度* (Williams, Waterman, and Patterson 2009) 来衡量，即每次内存访问字节数对应的算术运算次数。

1.  计算受限：操作所花费的时间取决于有多少算术运算，而访问 HBM 的时间要小得多。典型的例子是具有较大内维度的矩阵乘法，以及具有大量通道的卷积。

2.  内存受限：操作所花费的时间取决于内存访问的次数，而花费在计算上的时间要小得多。例子包括大多数其他操作：逐元素操作（例如，激活、dropout）和归约操作（例如，求和、softmax、batch norm、layer norm）。

**Kernel 融合。** 加速内存受限操作最常见的方法是 kernel 融合：如果有多个操作应用于同一个输入，则可以从 HBM 加载一次输入，而不是为每个操作加载多次。编译器可以自动融合许多逐元素操作 (Li et al. 2020; Paszke et al. 2019; Sabne 2020)。然而，在模型训练的上下文中，中间值仍然需要写入 HBM 以便为反向传播保存，这降低了朴素 kernel 融合的有效性。

## 标准 Attention 实现

给定输入序列 $\mathbf{Q}, \mathbf{K}, \mathbf{V}\in \mathbb{R}^{N \times d}$，其中 $N$ 是序列长度，$d$ 是头维度，我们要计算 attention 的输出 $\mathbf{O}\in \mathbb{R}^{N \times d}$： $$\mathbf{S}= \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{N \times N}, \quad \mathbf{P}= \mathrm{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \quad \mathbf{O}= \mathbf{P}\mathbf{V}\in \mathbb{R}^{N \times d},$$ 其中 $\mathrm{softmax}$ 是按行应用的。（为了阐述清晰，我们省略了 $\mathbf{Q}\mathbf{K}^\top$ 的缩放（通常为 $1/\mathrm{d}$），以及可选的对 $\mathbf{S}$ 的逐元素掩码和/或应用于 $\mathbf{P}$ 的 dropout）对于多头注意力 (MHA)，同样的计算在多个头之间并行执行，并在批次维度（一个批次中输入序列的数量）上并行执行。

attention 的反向传播过程如下。设 $\mathbf{dO}\in \mathbb{R}^{N \times d}$ 为 $\mathbf{O}$ 关于某个损失函数的梯度。然后根据链式法则（即反向传播）： $$\begin{aligned}
  \mathbf{dV}&= \mathbf{P}^\top \mathbf{dO}\in \mathbb{R}^{N \times d} \\
  \mathbf{dP}&= \mathbf{dO}\mathbf{V}^\top \in \mathbb{R}^{N \times N} \\
\end{aligned}$$

标准的 attention 实现将矩阵 $\mathbf{S}$ 和 $\mathbf{P}$ 物化到 HBM 中，这需要 $O(N^2)$ 的内存。通常 $N \gg d$（通常 $N$ 在 1k–8k 的数量级，而 d 在 64–128 左右）。标准的 attention 实现 (1) 调用矩阵乘法 (GEMM) 子程序来计算 $\mathbf{S}= \mathbf{Q}\mathbf{K}^\top$，将结果写入 HBM，然后 (2) 从 HBM 加载 $\S$ 以计算 softmax 并将结果 $\mathbf{P}$ 写入 HBM，最后 (3) 调用矩阵乘法得到 $\mathbf{O}= \mathbf{P}\mathbf{V}$。由于大多数操作受限于内存带宽，大量的内存访问转化为缓慢的挂钟时间。此外，由于必须存储，所需的内存为 $O(N^2)$

## FlashAttention

为了加速 attention

## 参考文献


Jia, Zhe, Marco Maggioni, Benjamin Staiger, and Daniele P Scarpazza. 2018. “Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking.” *arXiv Preprint arXiv:1804.06826*.



Jia, Zhe, Blake Tillman, Marco Maggioni, and Daniele Paolo Scarpazza. 2019. “Dissecting the Graphcore IPU Architecture via Microbenchmarking.” *arXiv Preprint arXiv:1912.03413*.



Jia, Zhe, and Peter Van Sandt. 2021. “Dissecting the Ampere GPU Architecture via Microbenchmarking.”



Jouppi, Norman P, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, et al. 2017. “In-Datacenter Performance Analysis of a Tensor Processing Unit.” In *Proceedings of the 44th Annual International Symposium on Computer Architecture*, 1–12.



Li, Mingzhen, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, and Depei Qian. 2020. “The Deep Learning Compiler: A Comprehensive Survey.” *IEEE Transactions on Parallel and Distributed Systems* 32 (3): 708–27.



NVIDIA. 2017. “Nvidia Tesla V100 GPU Architecture.” Aug.



———. 2020. “Nvidia A100 Tensor Core GPU Architecture.”



———. 2022. “Nvidia H100 Tensor Core GPU Architecture.”



Paszke, Adam, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, et al. 2019. “Pytorch: An Imperative Style, High-Performance Deep Learning Library.” *Advances in Neural Information Processing Systems* 32.



Sabne, Amit. 2020. “XLA: Compiling Machine Learning for Peak Performance.”



Williams, Samuel, Andrew Waterman, and David Patterson. 2009. “Roofline: An Insightful Visual Performance Model for Multicore Architectures.” *Communications of the ACM* 52 (4): 65–76.