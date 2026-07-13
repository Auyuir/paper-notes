# MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures 原文翻译

# MLX：面向空间架构上结构化LLM工作负载加速的多层执行

Haibin Wu<sup>1,2</sup>, Wenming Li<sup>1,2,∗</sup>, Zhihua Fan<sup>1,2</sup>, Zirui Ma<sup>1,2</sup>, Yuqun Liu<sup>1,2</sup>, Tengfei Xia<sup>1,2</sup>, Yanhuan Liu<sup>1,2,3</sup> , Kunming Zhang<sup>1,2,3</sup>, Xiaochun Ye<sup>1,2</sup>, Dongrui Fan<sup>1,2</sup>, Jian Weng<sup>4</sup>

<sup>1</sup>中国科学院计算技术研究所处理器芯片全国重点实验室，北京，中国

<sup>2</sup>中国科学院大学，北京，中国 <sup>3</sup>Ricore IC Technologies Ltd.，中国

<sup>4</sup>阿卜杜拉国王科技大学，图沃，沙特阿拉伯

{wuhaibin, liwenming, yexiaochun, fandr}@ict.ac.cn, {liuyanhuan, zhangkunming}@ri-core.cn, jian.weng@kaust.edu.sa

摘要——结构化稀疏是扩展大语言模型（LLM）推理的一种极具前景的方法，但由于深度的阶段依赖和有限的批量并行性，现有的形式（如蝶形结构稀疏投影和变换）在映射到GPU时往往效率低下。本文提出了MLX，一种用于结构化LLM推理的算法-架构协同设计。MLX将语义感知的FFT压缩和分层稀疏投影与空间数据流执行相结合，使分阶段的结构化算子能够在紧凑阵列上高效运行。MLX定义了闭依赖组件来捕获确定性的仅前向数据流区域，这些区域可以跨层折叠并在紧凑阵列上进行流水线处理。随后，它通过一种多层执行架构来实现CDCs，该架构具有有限跳数的跳跃路由、基于标签的调度以及解耦的计算/传输流水线，以重叠跨深度算子的通信和计算。我们在12nm工艺下对MLX进行了原型验证，结果显示，与Jetson Xavier相比，它实现了3.2倍的硬件加速和3.1倍的节能。针对Transformer的专用精简设计进一步实现了比现有稀疏加速器高达5.7倍的加速。MLX在扩展到8×8网格时也表现出近乎线性的可扩展性，并且在1K到4K的长序列上仍然有效，这证明了结构化算子语义可以转化为稀疏LLM的高效空间执行。

关键词——数据流架构，空间加速器，结构化稀疏，大语言模型

## I. 引言

Transformer 模型已成为现代人工智能在自然语言处理（NLP）[1]、计算机视觉（CV）[2]和多模态任务中的主导基础。尽管它们具有很强的推理能力，但这些模型严重依赖矩阵乘法（如图 1(a) 所示），这带来了扩展成本： self-attention 产生 $O ( n ^ { 2 } d )$ 的计算量和大量数据流量， 线性投影产生 $O ( n d ^ { 2 } )$ 的计算量以及 $O ( d ^ { 2 } )$ 的参数存储量。随着 n 的增长，二次方的 attention 项及其相关的内存移动越来越主导端到端的延迟和能耗。

先前的工作通过保持规律性同时降低计算量的结构化近似来降低这种成本。一个方向是将 butterfly 分解应用于线性投影 [3, 4, 5, 6]，用结构化矩阵替换稠密权重（图 1(b)），并将计算量降低至 butterfly 稀疏矩阵乘法（BSMM）。这降低了投影层的成本，但 Q、K 和 V 仍然是稠密的，因此对于长上下文，后续的 attention 过程仍然是主要瓶颈。第二个方向（图 1(c)）通过稀疏 attention [7, 8] 或傅里叶变换 [9] 来修改或替换 token 混合，从而降低二次方 attention 的成本。激进的傅里叶式混合极大地降低了 token 交互复杂度，导致 FLOPs 大幅减少。

尽管前景广阔，但将这些方法扩展到现代 LLM 暴露了两个实际挑战。首先，现有的 butterfly 稀疏性分解应用于整个投影矩阵；在 d 较大时，分解问题的复杂度增加，变得更难收敛，并且可能产生更大的近似误差。其次，用 FFT 式 token 混合完全替换依赖内容的 attention 移除了显式的 token 间交互，这可能会损害准确性，并且不易适用于标准 LLM 流水线。我们的关键观察有两方面：LLM 层沿序列维度表现出语义频率局部性，我们利用这一点为基于 FFT 的 token 混合选择性地保留信息丰富的频率分量，而块结构将 butterfly 稀疏性局部化到较小的子矩阵，使分解更容易收敛且精度损失更小。这两点见解共同将傅里叶运算和分解统一在单一的结构化 butterfly 稀疏性之下。然而，在批量并行架构上将这些算术节省转化为有效加速仍然很困难，这促使我们采用一种协同设计的方法，以更好地利用结构化和可预测的数据重用。

我们在图 2 中的分析结果揭示了这种脱节。尽管 FFT attention 在理论上可以将算术运算量减少 10 倍以上，但实际实现的端到端加速往往小得多。例如，在 batch size 为 64 的 NVIDIA AGX Orin 上，基于 FFT 的结构化 Transformer 块在序列长度为 8K 和 512 时，分别仅比稠密基线实现了 3.77 倍和 2.56 倍的加速。为了解释为什么 FLOP 的减少在 GPU 上未能充分实现，我们使用 roofline 分析来分离计算受限和带宽受限的区间。Orin 暴露了边缘情况的症状，而 H100 提供了现代参考范围。图 3 绘制了使用优化后的 cuFFT 的 H100 roofline。在 CUDA 核心上运行的 FFT 和 butterfly 的操作强度（OI）远低于 TensorCore 单元（TCU）上的稠密 GEMM，使它们处于带宽主导的区间。然而，它们仍然远远低于 CUDA 带宽 roofline，表明存在内存受限之外的低效性。我们将这种差距主要归因于破坏局部性的多阶段数据重排和执行单元不匹配，详见第 II-B 节。

![](images/0b6609dcb510de942c77cbe591310e55db6034c7c638876830867bc85eed315d.jpg)  
图 1：不同 Transformer 块实现之间的权衡。操作强度（OI）衡量为每字节片外 DRAM 流量的有效 FLOPs，仅计算投影和 attention 阶段。

![](images/1c4c5ddbedce684569d1db898198872c93251ca17472223737092581a035be2a.jpg)

![](images/af9e728e8a7cb0e4e994b6080bb7ff3f8bb9db22e7cd517803808e2cce01d8e7.jpg)  
图 2：NVIDIA AGX Orin 上的性能分析结果。阴影部分为在 attention 和投影上应用 FFT 和 BSMM 的基于 FFT 的内核。

为了解决现有基于 butterfly 方法的局限性及其硬件不匹配问题，我们提出了 MLX，一种结构化的 LLM 协同设计。如图 7 所示，MLX 将沿序列维度的语义感知压缩与沿隐藏维度的分层 butterfly 稀疏性相结合。这些算子共同将计算分解为具有严格前向层对齐依赖的有界闭集，这在批量同步执行下效率低下，但自然映射到空间数据流 [10, 11, 12, 13, 14]。这一见解引出了 Multi-Layer Execution（MLX），这是一种折叠抽象，能够在紧凑的数据流阵列上实现有序的数据重用和跨层流水线。我们的贡献如下：

• 用于 LLM 加速的 Butterfly 数据流：我们将层感知频谱截断以缩短 token 序列，与分层 butterfly 分解以降低投影复杂度相结合。它们共同降低了主流 LLM 中的计算和内存成本，同时保留了结构化稀疏性以实现高效的数据流执行。

• Multi-Layer Execution：一种通用的多层执行模型，将跨层依赖折叠到保持局部性的流水线中，从而能够在空间阵列上高效执行深度堆叠的结构化算子。

• 层折叠空间基底：我们构建了一个空间基底，可实现层间数据流路由、解耦的计算-传输流水线以及灵活的阵列映射。这种设计允许稠密和稀疏算子在紧凑阵列上被折叠和深度重叠，从而在 LLM 工作负载中保持高利用率。

![](images/506a0298dbfedfa610559e9e7760a1f48b3c6d130065d8ebb7ea7f5cd1e7b6598.jpg)  
图 3：NVIDIA H100 GPU 上 LLaMA2-7B (FP16) 在 prefill 阶段（N = 512, 8K）的 Roofline 模型和 CUDA 利用率。

• 流片芯片：这项工作受益于先前的实际硬件开发，所提出的加速器是源自通用数据流设计的简化和缩减变体。实际的流片为设计可行性和合理性带来了更高的信心。

评估表明，我们改进的 Transformer 块将 FLOPs 减少到形状匹配的稠密 Transformer 块的 30%，且准确率下降不到 1.8%。与先前基于 FFT 的 Transformer 相比，它在使用更少 FLOPs 的同时将准确率提高了 1.9%。我们提出的加速器相较于先前的 SOTA 稀疏加速器，实现了高达 5.8 倍的加速和 2.6 倍的节能。流片设计在相同技术节点下，且峰值 FLOP/s 与 NVIDIA Jetson Xavier 相似，在提出的稀疏化 Transformer 模型上实现了 3.2 倍的加速和 3.1 倍的节能。

## II. 背景与动机

## A. Transformer 中的先验结构化稀疏

稀疏性通过移除不必要的权重、激活或交互来减少有效计算和数据移动。结构化通过将稀疏模式约束为规则块、分阶段变换或确定性混合路径，而非任意不规则的非零元素，使这种减少变得可预测。两者结合，提供了算法压缩和架构规律性，暴露出可重用的数据流、有界依赖和可特化的执行调度。先前的工作 [6, 15, 16, 17] 利用这种结构化算子来降低 Transformer 块的复杂度。接下来，我们将讨论两种先前稀疏方法的原理和权衡：块对角矩阵（butterfly 因子分解）和傅里叶变换。

先前 Butterfly 稀疏的局限性。图 4(a) 展示了 butterfly 稀疏因子分解。该方法允许用结构化稀疏矩阵替换投影和前馈阶段的密集权重矩阵。密集矩阵可以近似为块对角矩阵的乘积 [4, 6]。总共有 $\log _ { 2 } n$ 个稀疏因子矩阵，每个矩阵在固定距离上执行结构化的成对混合。我们将第 k 个因子表示为 $B _ { n } ^ { ( k ) } \in \mathbb { R } ^ { n \times n }$ ，其中 n 是 2 的幂，阶段 k 混合距离为 $2 ^ { k }$ 的索引对。在 2 参数 $2 \times 2$ 混合参数化下，每个因子包含 2n 个参数；因此总参数量为 2n log n，相对于密集的 n×n 矩阵，压缩比为 $2 \log _ { 2 } n / n$ 。与这些 butterfly 因子矩阵相乘 (BSMM) 仅需 $O ( n ^ { 2 } \log { n } )$ 的复杂度——与密集投影 $O ( n ^ { 3 } )$ 的成本相比降低了一个数量级。然而，对任意大型密集矩阵进行分解本身会产生巨大的计算开销，并且在离线阶段变得更难求解 [6, 18]，使得该方法在 LLM 的参数规模下不太实用。

基于 FFT 的 Attention 的局限性。先前的一种替代方案是用基于傅里叶变换的 token 混合来替换 attention（图 4(b)）。该方法使用 2-D FFT [9, 19] 通过固定的傅里叶基全局混合 token 和隐藏维度。它移除了 attention 中依赖于内容的成对加权，但也失去了将交互适应于特定输入的局部或语义依赖的能力。尽管其成本是次二次的（例如 O(N D log N)），但在依赖细粒度 token 交互的任务上，它的性能可能不如 attention，并且与基于缓存的增量更新至关重要的 prefill/decode 流水线兼容性较差 [20, 21, 22]。

## B. GPU 上的结构化执行不匹配

图 3 显示，attention 阶段的 butterfly 内核具有较低的 OI，因此受带宽限制，但其实际性能仍低于 CUDA 带宽屋顶线。这表明除了内存限制之外还存在差距，其根源在于 butterfly 数据流与 GPU 执行之间的不匹配。具体而言，FFT 流水线涉及多阶段的跨步和混洗访问，这破坏了局部性并阻碍了带宽利用率，这与图 2 中高缓存未命中率相一致。与能有效映射到 TCU 的密集基线不同，butterfly 阶段主要作为 CUDA 的向量原语执行，伴随频繁的内存访问和数据重排，尽管减少了 FLOPs，但限制了端到端的加速。

在更深层面上，这种不匹配源于 BSMM 和 FFT 的阶段性依赖结构（图 8）。随着工作集在各个阶段扩展，它们需要有序的数据交换，这与 GPU 的批量同步、分块规则执行相冲突，而 TCU 上的 2:4 结构化稀疏则保留了分块局部的、类似密集访问的特性。因此，在分块内实现 butterfly 排列会导致昂贵的数据混洗。尽管最近基于 TCU 的 FFT 设计 [19, 23] 优于 CUDA 核心，但由于块分解增加了超出理想 O(N log N) 成本的额外工作，并且仅部分填充了 TCU 流水线，它们仍然留有利用率提升的空间。

![](images/81cc490529b56fcd60ba1596a2ea91f81833739a30c4b08074a1e27e1c5c1511.jpg)  
图 4：使用结构化稀疏改进 transformer 块。

总的来说，这些局限性促使我们进一步研究 LLM 工作负载中的 butterfly 稀疏性。接下来，我们将表征其结构并确定我们协同设计的机会。

挑战：算子异构性。基于 butterfly 的加速涵盖 BSMM、FFT 和密集投影，它们表现出截然不同的数据流模式。这种异构性使特化变得复杂，似乎需要不同的硬件支持。

机会 1：统一的依赖结构。尽管存在差异，FFT 和 BSMM 共享固定的、分层的 butterfly 阶段（图 4(c)），并且密集投影也可以表示为分块的生产者-消费者流。它们共同遵循共享的阶段性依赖表示，从而在这些内核之间实现统一的执行流水线。

挑战：GPU 上有限的细粒度并行性。BSMM 引入了长距离跨步混合和加载-同步周期，这阻碍了 GPU 在单个矩阵-向量变换中利用细粒度重用。

机会 2：可预测的跨层数据流。同时，BSMM 保留了显式的分阶段结构。每个阶段产生的输出会馈送到下一阶段中少量预定的消费者集合，如图 8 所示。这种结构化依赖允许将部分结果直接路由到后续阶段，形成无需全局内存往返的细粒度多层流水线。

机会 3：正交维度并行性。基于 FFT 的序列压缩和 BSMM 投影暴露出与多层流水线正交的丰富的隐藏层和 token 级并行性。可以通过空间数据流上的向量化或时间多迭代执行来利用它，在保持分阶段调度的同时确保吞吐量。

总结：BSMM 和 FFT 都归结为相同的行为：持续产生可预测的生产者-消费者依赖的分阶段结构化线性变换。

![](images/cb1a84963a6d5afe7ca4bb83b4a4e1cc14791549ff4da2cf93cda7f75f2f413a.jpg)  
图 5：Llama2-7B 各层中 QKV 的主频。

![](images/47bfa25f0cd9fc52e6aaf6d25342038818cf3abac4804acb0eb99aaa3b6125e0.jpg)

![](images/ab1c9177b52530c8755a955f39a38050318469db2dfaaa406cebb3a2214b63.jpg)  
图 6：Llama2 第 1/16 层中 token 序列的频域能量。

## C. 空间数据流：基础设计

结构化算子暴露出可预测的跨阶段依赖，但 GPU 通常将它们作为分离的阶段来执行，伴随频繁的同步、跨步/置换交换和较差的局部性。相比之下，空间数据流架构 [11, 24, 25] 可以通过显式的操作数路由和确定性网格流来实现这种有序依赖，通过流水线时间执行维持并行性。先前的工作 [10, 26] 已经利用了类似的数据流特性进行稀疏计算。Butterfly 算子提供了更强的结构：它们的依赖是稀疏的、固定的、有界的，并且在各个阶段严格前向。这不仅降低了路由复杂性，还允许部分结果在连续阶段之间保持进行中状态，形成深度流水线数据流，而不是孤立的逐阶段执行。

这些特性促使了多层执行（Multi-Layer Execution）的提出，它将有序的跨阶段依赖折叠到紧凑的空间阵列上，以最大化数据重用，将通信与计算窗口重叠，并维持高 PE/FU 利用率。

## III. 算法创新

我们首先激发我们的算法创新，即混合 Transformer 块，它结合了压缩和稀疏技术，以加速现代规模下的 LLM。

## A. 语义感知的傅里叶压缩

在本节中，我们解释如何将 FFT 压缩应用于上下文序列。Transformer 层表现出不同的语义行为：浅层倾向于关注局部的、细粒度的 token 细节，而深层则编码更广泛的上下文信息。从信号处理的角度来看，这表现为序列长度 N 上不同的频率分布：细粒度模式映射到高频分量，而上下文抽象将能量转移到低频。我们通过对 Llama2-7B [27] 各 Transformer 层的 $Q , K , V$ 向量应用 FFT 来验证这一点（图 5）。图 6 中的 K 频谱显示，第 1 层由右侧的高频内容主导，而第 16 层更加平滑，以低频为主。尽管 $\overset { \cdot } { Q } / \kappa / V$ 是中间表示，其频率分布反映了每一层如何沿序列维度聚合语义信息。受观察到的频谱特征启发，我们将每层的块长度 $L _ { l }$ 定义为与第 l 层最短显著变化尺度匹配的序列间隔。令 $\tilde { f } _ { H }$ 表示能量超过相对阈值（例如峰值能量的固定比例）的最高频率频谱峰值。我们定义标称尺度 $\tilde { L } = N / \tilde { f } _ { H }$，并将其量化为 2 的幂以实现硬件友好的对齐：

![](images/939e965eafbc5b61d248a4f2d99e10cbdad37e7a2bbedd3892546ca0a074cad7.jpg)  
图 7：我们的方法：结构化稀疏与 FFT 的混合（解压缩为对称操作，此处省略）。

$$
\tilde { L } = N / \tilde { f } _ { H } , \qquad L = \mathrm { P o w 2 R o u n d } ( \tilde { L } ) .\tag{1}
$$

核心思想：从 N 进行分块将 FFT 长度固定为 L，并在语义感知的间隔内实现 token 混合的局部化，同时实现高效的、流式友好的傅里叶压缩，以极小的信息内容损失去除占比较低的高频分量（图 7(b)）。具体而言，对于投影后的每个矩阵 $Q , K , V \in \mathbb { R } ^ { N \times D }$：（1）重塑为 $N / L$ 个块，并对每个特征维度执行 $N / L$ 次独立的 L 点 FFT，以获得逐块频谱；（2）沿每个 L 维度截断最后 $( 1 - s )$ 比例的高频系数，保留前导的 $s L$ 个信息分量 [28]；（3）对每个块保留的系数应用 sL 点 iFFT，在低频子空间中重新生成缩短的 token 表示。

该过程丢弃低能量的高频分量，并通过 s 实现可调的计算-精度权衡。我们在第 VII 节评估代表性的工作点（例如 s=0.5、0.75）。缩短后的序列将 prefill 开销降低至 $O ( s ^ { 2 } N ^ { 2 } D )$，同时缩小了 attention 矩阵，缓解了缓冲压力和内存流量。由于该二次项仍然主导 attention 流水线，额外的分块 FFT 开销 $O ( N D \log L )$ 相对较小，使得基于 FFT 的压缩具有成本效益。

表 I：LLM 中基于 Butterfly 的 Kernel 对比。
<table><tr><td>Butterfly Kernel</td><td>适用范围</td><td>Prefill</td><td>Decode</td><td colspan="2">精度可调</td></tr><tr><td>2D-FFT</td><td>Attn.</td><td>√</td><td>×</td><td>×</td><td>已有方法</td></tr><tr><td>BSMM</td><td>QKV / FFN</td><td></td><td>√</td><td>X</td><td rowspan="2"></td></tr><tr><td>FFT 压缩</td><td>Attn. / KV Cache</td><td></td><td>√</td><td> $\checkmark ( s )$ </td></tr><tr><td>分层 BSMM</td><td>QKV / FFN</td><td>√</td><td>√</td><td> $\checkmark ( B )$ </td><td>本文</td></tr></table>

在 prefill 阶段，语义 FFT 以固定大小的 L-token 块应用于提示。在 decode 阶段，尽管 N 不断增长，我们保持 L 固定，并避免对完整前缀进行重新变换。已完成的块复用缓存的压缩块，而新 token 在本地缓冲区中累积。一旦缓冲区达到 $L ,$，我们触发 FFT 压缩并追加一个新块。这产生了一个仅追加的、块粒度的缓存，并将 FFT 开销分摊到 L 个 token 上，保持与 KV-cache 解码过程兼容。

## B. 分层 Butterfly 分解

传统 BSMM 将 butterfly 分解应用于整个权重矩阵，这在理论上代价高昂且对于大型 LLM 模型不切实际 [4, 6]。我们转而采用一种分层变体，将 butterfly 结构限制在局部块内。权重矩阵 W 被划分为 $( D / B ) \times ( D / B )$ 个大小为 $B \times B$ 的块，并仅在每个块内应用 butterfly 因子。因此，与全局分解的 O(D log D) 相比，总的 butterfly 参数计算量变为 $( D / B ) ^ { 2 } \cdot O ( B \log B ) = O ( \frac { D ^ { 2 } } { B } \log B )$：

$$
\begin{array} { r } { O ( \frac { \log D } { D } ) \Rightarrow O ( \frac { \log B } { B } ) } \end{array}\tag{2}
$$

因此，块大小 B 提供了另一个可调的精度-效率旋钮：在固定的 butterfly 分解下，增大 B 施加更强的结构化稀疏性（更低的复杂度比 $O ( \log B / B )$），从而降低计算开销，但倾向于增加近似误差。该结构自然映射到分层的、逐块执行：块间计算遵循粗粒度的 blocked-GEMM 数据流，而块内 BSMM 实现细粒度的结构化 butterfly 数据流。混合 Butterfly Kernel。如表 I 所列，先前基于 butterfly 的应用（例如 FFT attention 变体和全局 BSMM [6, 9, 29]）大多在不直接针对现代 LLM 规模的 prefill-decode 推理场景中研究。因此，我们将语义感知的分块 FFT（序列维度 N）与分层 BSMM（隐藏维度 D）耦合。二者在正交维度上暴露并行性，并产生互补的数据流。这激发了接下来介绍的空间执行模型，该模型协同设计硬件和映射支持，以将这些算法优势转化为统一的空间加速。

## C. 算子抽象：多层执行（MLX）

分块 FFT 和分层 BSMM 可以表示为具有层对齐、仅前向依赖的阶段序列。更广泛地说，其他具有分阶段、步长规则依赖的结构化稀疏算子也符合此抽象。由于每个阶段具有有界的阵列驻留占用，执行可以随时间折叠：一次只有一部分阶段驻留在阵列上，而其他阶段按依赖顺序进行时分复用。我们将这种执行抽象称为 MLX，它将逻辑阶段深度与物理阵列大小解耦，通过在紧凑的 PE 阵列上进行折叠执行来实现深度流水线。MLX 的详细形式化定义在第 V-B 节中提供。

(a) 应用于向量的连续 BPMM（下半部分省略）  
![](images/a1f9b3cfd9fbb0f157fefc6dcfa298fed316e791e99dc88adc3992b5db65b420.jpg)  
图 8：跨多个 butterfly 稀疏矩阵乘法（BSMM）的流水线计算。

## IV. MLX 架构

本节描述如何在硬件中实现 MLX 范式。如图 9 所示，该架构由一个主机控制器、暂存存储器以及通过跳数编码网络连接的处理单元（PE）阵列组成。

## A. BSMM 作为 MLX 设计的动机案例

在我们的混合模型算子中，BSMM 最清晰地说明了为什么需要 MLX。如图 8(b) 所示，每个 BSMM 层消耗前一层的即时输出，形成具有完全可预测流水线依赖的深度且严格分层的数据流 [30, 31, 32]。原则上，连续的 BSMM 层可以在空间阵列上重叠，以暴露出大量但细粒度的并行性。然而，完整的 BSMM 数据流图太大且层数太深，无法一次性映射到固定大小的网格上。一旦计算单元在多个 BSMM 层之间共享，加速器就必须引入额外的专用化设计以维持高吞吐量：(1) 调度指令使得不同的 BSMM 层能够在功能单元 (FU) 之间以交错且高利用率的方式执行，以及 (2) 通过短且可预测的路径将中间结果路由到其明确的下游。

我们的目标是构建这样一个加速器，它专为具有中等算术强度的大型结构化数据流图而设计，能够协调重叠的层执行和显式的低延迟数据传输，如图 9(a) 所示。

## B. 用于层折叠执行的跳步 NoC 拓扑

层折叠将跨层依赖转化为有界的、规则的通信模式。在 BSMM 和 FFT 中，每个折叠层访问确定性的步长为 $- 2 ^ { k }$ 的邻居，这种访问模式不适合全局内存流量，但自然匹配拓扑感知的网格 NoC。因此，MLX 采用了跳步网格 (Fig. 9(b))，在每个 PE 上除了本地邻居转发之外，还扩展了固定距离的链路。这些链路直接跨越折叠依赖半径，并将大多数跨层传输减少到一跳或两跳。

为了以最小的硬件状态实现这些传输，MLX 使用了一种跳步编码的数据移动原语。每条 xfer 指令仅携带剩余跳数、路由方向和目标寄存器。路由器是无状态的：当跳数达到零时，数据在本地写入。否则，路由器消耗最大允许的步长——单位步长或跳步——并转发数据包。这将结构化的 MLX 依赖转化为确定性的有界跳数传输，避免了路由表、虚拟通道和动态路由计算。同一原语自然涵盖了蝶形步长、FFT 配对、密集矩阵乘法的脉动运动以及有界窗口交互 (Sec. V-C)，为折叠执行提供了统一的空间基底。

## C. 空间 PE：实现层折叠执行

微架构。MLX 的折叠层执行要求每个 PE 并发推进多个层：为未来层加载输入、计算当前层，以及转发前一层 的输出。MLX 没有通过细粒度的指令级冒险来跟踪这些交互，而是将每个 PE 解耦为四个独立的流水线，分别用于内存移动、数据流传输和异构算术，如图 9(c) 所示。这种解耦对于 MLX 的混合算子至关重要，这些算子将实数/复数算术与激活和归一化函数混合在一起，如图 9(d) 所示。用单一的指令级调度器管理这些异构单元将涉及大量的面积和控制复杂度。相反，MLX 将它们分离为并行流水线，允许每个 PE 以层粒度进行调度，同时自然地重叠各层的阶段。因此，多个层有效地分时共享相同的 PE 资源，在活动层窗口内维持高利用率，同时保持 PE 控制的轻量化。

层编码指令存储。执行许多 MLX 层将需要一个大型的指令缓冲区来保存所有逐层操作 $( \Theta ( K \cdot I _ { \mathrm { l a y e r } } ) )$ 。相反，MLX 利用了一个关键的结构特性：每一层形成一个固定的、可重用的指令模板，其内部顺序永不改变。因此，我们将每个逻辑层编码为一个紧凑的标记块，即一个短的静态指令序列加上一个循环计数，它捕获了该层的精确计算足迹。随着折叠执行窗口向前滑动，这些标记块被重复访问，从而将指令存储大小与总层数解耦。在任何时候，只有一小组活动块需要驻留在 PE 中，从而允许恒定占用的指令层次结构，而与算子深度无关。

(1) 层对齐调度。标记块将硬件调度单元与 MLX 层边界对齐。每个块携带一个标签、一个缓冲的指令序列和一个循环计数 n，允许 PE 以层粒度跟踪就绪状态、进度和完成情况。这避免了逐指令的簿记、大型依赖表和细粒度的冒险元数据，同时保留了折叠层执行所需的语义边界。

![](images/2e77063b30e8dddd282a5e3c91afd30379fb4a3b0fe9cbe5732817e1b9347294.jpg)  
Fig. 9: The spatial accelerator design of MLX architecture.

(2) 活动窗口流水线重叠。标记块作为解耦 PE 流水线中活动层窗口内的可调度条目。在此窗口内，不同的折叠层可以同时占据不同的流水线阶段——一个加载输入，另一个计算，第三个转发结果。标签标识每个块属于哪个活动层，允许 PE 将多个层作为结构化流水线推进。这暴露了跨层并行性，同时保持控制粗粒度、可预测且低成本。

混合调度。MLX 提供了一种既非完全动态也非完全静态的调度机制。完全动态的调度器 [12] 必须跟踪 FFT/BSMM 层内操作之间的细粒度依赖，而完全静态的调度则要求编译器联合推理所有折叠层的路由、周期时序和资源冲突。相反，MLX 将层内确定性与跨层弹性分离。在每个折叠层内，依赖是本地且有序的，允许编译器发出静态指令序列。在跨层方面，通信被简化为一小部分拓扑对齐的标签事件，例如传输和转发，由硬件弹性仲裁 [33, 34]。因此，软件固定确定性的本地调度，而硬件仅管理标签级的跨层协调，从而减少调度状态，而无需依赖细粒度的动态唤醒或全局周期级规划。

在图 9(c) 中，每个折叠层是一个由标签标识的粗粒度执行窗口。仲裁器仅看到层的前沿指令 inst\_i 并解决解耦流水线之间的争用。标签 ID 编码了层之间的有效偏序。优先处理较小的标签可保持依赖正确性，同时允许层重叠并隐藏延迟。当一个层的 LD 完成时，它会在标签条目中设置一个就绪位，以启用下一阶段（计算和/或 xfer）；每个流水线在资源可用的情况下在就绪标签中进行选择。如果多个层就绪，仲裁器可以采用轮询方式，使用标签 ID 作为决胜依据。图 9(d) 展示了这种粗粒度仲裁的工作方式。当 tag1 的 add 和 tag2 的 mul 争用计算流水线时，仲裁器授予较低的标签并暂停另一个块，以块/标签粒度做出决策。

## D. 设计参数

设计参数由 MLX 算子的特性和混合网络模型决定：

原则 1 – SIMD 宽度。对于 GEMM，一层的计算块在 $n \times n$ 的密集 tile 上运行。一个 tile 执行 $n ^ { 3 }$ 次 MAC 运算，同时读取 $2 n ^ { 2 }$ 个输入操作数，给出基准计算-流量比为 $n / 2$，以及维持有效复用的必要条件 $n \geq 4$。BSMM tile 施加了一个正交约束：butterfly 稀疏性必须有意义。对于具有 $k = \log _ { 2 } n$ 级的完整 n 点 butterfly，非零密度 $\mathrm { { i s } } \ \frac { 2 \log _ { 2 } n } { n }$，这表明稀疏性要有效，最小 2 的幂次宽度为 $n \geq 8$。因此，8 路 SIMD 是我们的精简设计所遵循的必要下界，而我们的完整设计采用 32 路 SIMD 以更好地利用并行性并在 MLX 下维持更高的吞吐量。

原则 2 - Mesh 规模与指令存储协同扩展。随着 mesh 规模扩展，MLX 的依赖半径增大，增加了层间加载和 butterfly 交换的物理跳距离。这增加了以周期为单位的通信延迟，包括 $T _ { \mathrm { l o a d } }$ 和 $T _ { \mathrm { x f e r } } .$。为了维持利用率，每个活跃层块必须提供足够的计算来隐藏主导的通信延迟：

$$
T _ { \mathrm { c o m p u t e } } ( { \mathsf { b l o c k } } ) \ \geq \ \operatorname* { m a x } ( T _ { \mathrm { l o a d } } , \ T _ { \mathrm { x f e r } } ) .
$$

因此，更大的 mesh 需要更大的活跃层窗口和成比例增加的片上指令存储来保持流水线满载。由于 MLX 以标记块的粒度调度执行，每个块可以利用结构化的块内复用（例如，butterfly 原语中的算术复用或 MM 中的 tile 内复用），从而提高其有效计算强度，这与指令级数据流中操作数通常在节点内仅消耗一次形成对比。随着 mesh 规模扩展，这种更高的块级强度有助于隐藏因步幅膨胀的通信延迟 $T _ { \mathrm { l o a d } }$ 和 $T _ { \mathrm { x f e r } } ,$。因此，可以通过增大块的计算预算 C 或并发活跃标签数 $B _ { T }$ 来维持覆盖率

$$
B _ { T } \cdot C \geq T _ { \mathrm { l o a d } } + T _ { \mathrm { x f e r } } .
$$

因此，指令存储容量不是由每个 kernel 的指令数决定的，而是由在大规模下摊销加载/传输延迟所需的覆盖窗口决定的。基于规模与延迟之间的这一权衡，我们选择了一个紧凑的设计点：4 × 4 mesh，每个 PE 32 条指令，这足以满足覆盖条件。

原则 3 - 精度支持与非线性。我们的 attention 模型支持高达 8192 点的全序列 FFT，需要 4096 个不同的旋转因子。由于低于 FP16 的精度会使 butterfly 累加不稳定，MLX 使用 FP16 作为最低稳定精度。超越函数单元被集成到每个 PE 中以支持非线性运算，但它们仅以 SIMD 宽度的四分之一运行。

![](images/a59823916bc8cf6515650c13717393e266456071a4f10add91667580c363de7b.jpg)  
图 10：为 BSMM 分配计算资源。（为清晰起见，省略了基于 batch 的 SIMD 以及 stride = 4、8 的垂直跳。）

Host Controller：在图 9(a) 中，加速器由一个小型 RISC-V 主机核心编排，该核心处理外层循环控制，同时向空间阵列发出紧凑命令。只需要最少的 ISA 扩展即可加载 MLX 配置并协调内存搬运，使控制平面与现有 RISC-V 软件兼容。

## V. 在 MLX 下映射 LLM 算子

我们描述了如何将三个核心结构化算子映射到 MLX 上，并给出了其形式化和抽象。

## A. 混合模型的结构化算子映射

以 BSMM 为例。图 10(a) 抽象了一个蝶形稀疏矩阵-向量乘积。我们将其表示为三个嵌套循环。如图 10(b) 所示，最内层循环 i2 在 $4 \times 4$ 网格上完全展开。中间循环 i1 在每个 PE 内部本地运行，随后结果在后续的蝶形层执行中流经整个阵列。而外层循环 $i _ { 0 } ,$ 连同与蝶形计算正交的独立维度 $i _ { \perp }$ ，作为由片上定序器驱动的数据流图迭代向前推进。这种分解产生了一个包含 64 个输出元素的闭集样本，这些元素可以在整个阵列上并发计算（为清晰起见，我们在此省略向量化细节，重点关注依赖结构）。该闭集代表了生产者-消费者依赖完全保留在 PE 阵列内部的最大占用空间，使我们能够对 MLX 进行流水线处理而不会溢出中间值。

步长对齐的数据路由。蝶形层表现出确定性的步长模式（±2, ±4, ±8, ...），这直接映射到我们的跳跃网格上的跳数距离。如图 10(c) 所示，每个 $\mathrm { P E _ { x } }$ 将其部分和路由到消费者 $\mathrm { P E } _ { \mathrm { x + s } }$ ，偏移量等于该层的步长。更远的步长 stride=4, 8 将被垂直转换为 1/2 跳 $( \mathrm { y } { = } \pm 1 , \pm 2 )$ ，为清晰起见，在图 10(c) 中省略了这一点。这种对齐允许几个 BSMM 层并发执行而不会产生路由争用，从而形成严格分层的片上流水线。

通过标记块实现 PE 内流水线。每个标记块具有固定的指令布局，分为几组：开头的加载、中间的计算和结尾的传输（图 10(d)）。这种规则的布局允许 PE 的解耦内存、计算和传输流水线重叠来自不同层的块。尽管内存流水线在中间层可能会偶尔空闲，但占主导地位的计算流水线仍保持持续占用。因此，MLX 用轻量级的标记块编排取代了细粒度的操作调度，在有限的硬件复杂度下维持了高利用率。利用率结果报告于第七-C节。

优化数据布局。在图 11(a) 中，暂存 SRAM 使用 SIMD 条带行并支持两种访问模式。列访问将 SIMD 通道与 BSMM 的序列轴 N 对齐，而行访问则沿块 FFT 的隐藏轴 D 流式传输连续元素。在行宽为 V（这里 $V { = } 8 )$ 的情况下，这种通道条带布局使 FFT 和 BSMM 的正交 SIMD 模式对齐到相同的 SRAM 组织。因此，它避免了算子之间的全阵列转置，允许中间数据保持原位并驻留在阵列中，以实现连续的 FFT-BSMM 数据流。

闭集局部性。在分块 FFT 中，每个 L 点的段形成一个封闭的依赖集。k 层 BSMM 具有相同的属性：在块大小 $B = 2 ^ { k }$ 的情况下，n 元素向量被划分为 $\frac { n } { B }$ 个大小为 B 的不相交闭集，并且所有蝶形交互都保留在每个集合内。然而，随着 L 或 B 的增长，默认的索引/布局排序将层交换转变为长步长洗牌（例如，当 $B = n / 2 )$ 时的半阵列步长），这破坏了空间局部性并迫使中间值在网格上跨越长距离（图 11(b)）。我们的关键观察是，蝶形依赖图在代数上是可划分的：FFT 和 BSMM 可以被重新排序以严格遵循其闭集。这使得将长蝶形流水线分解为可重用的数据流阶段成为可能。在阶段之间，I/O 洗牌使用图 11(a) 中的访问原语对暂存值进行重新索引。如图 11(c) 所示，洗牌后的阶段 2 值 $B _ { 8 } ^ { \prime } ( 0 )$ 在逻辑上继承自阶段 1 的 $B _ { 8 } ( 2 )$ ，但被重新映射到与 $B _ { 8 } ( 0 )$ 相同的空间占用和执行模板。因此，长距离蝶形依赖被转换为重复的紧凑局部数据流加上有限数量的阶段间交换。

## B. 从闭集局部性泛化 MLX

闭集局部性的一个关键洞察是，许多结构化算子（例如 FFT、BSMM、块 MM 和结构化 attention）共享一种通用的执行形式：它们的数据流图在具有有限接口的可重复局部组件上遵循一种仅前向的分层依赖结构。MLX 利用这一结构来 (i) 将每个组件编译成一个简短的带标签指令块，以及 (ii) 在具有有限在途状态的固定网格上将各层作为深度流水线执行。闭依赖组件（Closed Dependency Components, CDCs）。给定算子数据流图 $G = ( V , E )$ ，CDC 是一个对传入依赖封闭的子图 $C \subseteq V$ ：

(a) 沿 N/D 进行向量化布局优化

![](images/35e26c2cb95190e37c70ab6a4af8ecd64a2a572ddfdc1ddead3a31f1197c994b.jpg)

![](images/bedef6ddd48cec0edf40ddb5256e83520bf9b29bd9ae48f9e92428e25f406d84.jpg)  
图 11：(a) 针对友好 SIMD 封装优化数据布局；(b) BSMM 的数据占用空间；(c) 针对更小占用空间闭集的洗牌操作。

$$
\forall v \in C : \ ( v ) \subseteq C \ \cup \ \operatorname { I n } ( C ) ,
$$

其中 In(C) 表示 C 的外部输入。CDC 形成了一个具有有限局部性的自包含局部更新区域。与任意分块不同，CDC 是由算子的闭依赖模式定义的，而不是由启发式分块选择定义的。每个 CDC 都有一个固定的输入/输出接口：其交换值仅由模板参数（如蝶形宽度或 MM/CONV 块形状）决定，且不随整体问题规模增长。因此，结构化算子包含许多具有相同接口的重复 CDC 实例，允许 MLX 在它们之间重用相同的带标签块模板。

仅前向分层。许多结构化算子可以表示为 CDC 层 $\big \{ C _ { 0 } , \dots , C _ { K } \big \}$ ，其中每条边都是层内或指向下一层：

$$
( u \to v ) \in E \Rightarrow \ell ( v ) = \ell ( u ) { \mathrm { ~ o r ~ } } \ell ( v ) = \ell ( u ) + 1 ,\tag{3}
$$

其中 ℓ(·) 为层索引。层内的 CDC 是并行的，且层间依赖严格为仅前向，形成了一个没有长距离或循环依赖的流水线。随着深度增加，各层通常将交互范围从局部扩展到更全局，同时保持相邻层约束。分层路由编码。MLX 为每个 CDC 分配一个轻量级索引 ℓ 以表示其流水线类别。根据公式 3，CDC 仅在 ℓ 内部或向 $_ { \ell + 1 }$ 通信，因此 ℓ 直接选择下一级路由类别，而端点 PE 由 CDC 到 PE 的静态放置决定。对于结构化算子，传输通常属于一小组仿射偏移，因此路由可以通过 $( \Delta x , \Delta y )$ 紧凑地参数化，但关键属性是层间诱导的有限路由类别。每个 CDC 由循环驱动的带标签块执行，并由基于标签的依赖触发：

![](images/b11ebb5eb949f6b671b1f94efc9d71de8fd43d3094880817d3de08d9ce791172.jpg)  
图 12：滑动窗口 attention 的折叠 MLX 数据流：在同一个 2D 阵列上重叠 FMA/FMAX/FEXP 阶段。

$$
C _ { i } \ \mapsto \ \mathrm { l o o p } ( k ) \mathrm { t a g g e d \_ k e r n e l } _ { i } ( k ) ,
$$

其中 PE 在各个 CDC 实例间重放一个简短的带标签指令块，从而分摊解码和操作调度成本。原理与意义。任何能够表示为在闭工作集 $S _ { 0 } \subseteq \cdots \subseteq S _ { K }$ 上具有严格前向依赖（公式 3）的 CDC 层 $( C _ { 0 } , \dots , C _ { K } )$ 的结构化算子，都是 MLX 可执行的。CDC 的层间边形成了一个确定性的前向流水线，消除了全局调度。这种结构还支持空间折叠，将许多逻辑 CDC 层叠加到固定网格上，并将逻辑深度与物理阵列规模解耦。在实践中，折叠不需要保持所有逻辑层都处于活动状态。一个小的在途窗口足以维持 FU 利用率，同时对片上缓冲进行限制。相同的原理也扩展到了蝶形内核之外的其他分层结构化算子。

## C. 蝶形算子之外的结构化内核

为了证明 MLX 不局限于 FFT/蝶形风格同构内核，我们引入了来自 transformer 推理的第二个结构化工作负载示例：一个滑动窗口 attention（SWA）切片。尽管其计算混合了不同的原语（矩阵累加、规约、指数运算和归一化），其数据流仍然可以表示为具有严格分阶段依赖的一小段 CDC 层序列，这直接映射到同一 2D 阵列上的 MLX 折叠，如图 12 所示： 窗口化得分累加 $( Q K ^ { \top }$ ，FMA 为主)， 逐行最大值规约， 指数运算和归一化统计（FEXP + 求和/广播），以及 加权累加和归一化（SV ，FDIV，FMA）。

这些 CDC 层形成了一个相邻依赖链，其中每一层仅消耗紧接前一层的 CDC 边界输出（加上切片局部状态）。通过将逻辑层折叠到同一 PE 阵列上的紧凑片上矩阵流水线中，来自不同层的 CDC 批次可以部分处于在途状态，而所有层间通信仅通过显式的 CDC 边界 xfer 操作进行，使得传输可检查且有界。由于不同层侧重于不同的 FU 原语（FMA、FMAX 和 FEXP），带标签块执行可以轻松利用这种异构性。在层粒度具有足够延迟窗口覆盖的情况下，MLX 实现了稳态重叠，同时保持并发活动层的数量有界。

![](images/2a8606f93af71a3d4f1a06cf5cc3a3a39c039bfcc29e00d945d9debc332ac906.jpg)  
图 13：在多层数据流中将密集 MM 映射到 MLX。

MM 内核也适用于 MLX。在图 13 中，每个 PE 计算一个 8 × 8 的 SIMD 对齐切片，并在 $4 \times 4$ 网格上仅前向操作数传播下累加 psums。然后，我们将一系列切片折叠到这个紧凑的网格上，并将它们作为 MLX 层发出，使用固定的 load–comp–xfer 模板来错开各阶段并重叠工作。当每个切片的计算时间很短（例如，较小的 K）或切片是部分的（在 attention 中很常见）时，这是有益的，此时折叠分摊了填充/排空和边界开销，从而提高了利用率。

## VI. 方法论

## A. 软硬件实现

我们的设计继承自一个真实流片设计的实践经验。提出的 MLX 架构是一个基于性能分析的专用子集，源自一个通用数据流设计，该设计使用 Verilog RTL 实现，并使用 Synopsys DC 在 12nm @1GHz 下进行综合。对 FFT、BSMM 和密集 LLM 内核的性能分析表明，通用设计的许多特性对于结构化算子是不必要的，并且阻碍了多层执行模型。这使得我们能够构建一个专门为混合 LLM 工作负载量身定制的精简架构。

功耗与面积：完整流片设计的版图（图 14）作为投影面积和功耗的参考。在 Sec. IV-D 中参数分析的指导下，我们将 SIMD 宽度从 32 减小到 8，并移除了未使用的单元，例如向量混洗、除法和高精度浮点流水线。由此产生的精简设计仅占原始芯片面积的 10% 和功耗的 8%（表 II）。完整设计的功耗是在流片后测量的，而精简设计的功耗是根据综合后报告估算的。性能：我们使用（i）架构探索期间使用的周期精确 MLX 模拟器和（ii）流片硬件的测量结果来报告性能。这两个数字都将被报告，以便与具有相同峰值性能的同类加速器进行比较，如表 IV 所示。

软件部署：RISC-V CPU 作为我们加速器的主机控制器。为了将空间加速器比特流嵌入到 C 程序中，开发者编写数据流风格的汇编来指定每个 PE 的操作，或者使用基于 LLVM 的 C 编译器 [35] 进行编程。然后，一个轻量级的“空间汇编器”将这种文本格式编译为二进制文件，并将其导出为头文件，以便在 MLX 上进行配置。

![](images/986605da585a40f8cb9332e0a9838562c80fe2883a06b0bcf60d83b4195116f0.jpg)

<table><tr><td></td><td>面积-mm²</td><td>功耗-mW</td></tr><tr><td>配置网络</td><td>0.018</td><td>11.3</td></tr><tr><td>数据网络</td><td>0.092</td><td>56.2</td></tr><tr><td>控制逻辑</td><td>0.011</td><td>7.5</td></tr><tr><td>Tag 缓冲区</td><td>0.019</td><td>9.3</td></tr><tr><td>寄存器堆</td><td>0.044</td><td>28.7</td></tr><tr><td>FU (SIMD32)</td><td>0.298</td><td>252.4 (70%)</td></tr><tr><td>PE (跳步开销)</td><td>0.482 (6.2%)</td><td>365.4</td></tr><tr><td>PE 阵列</td><td>7.712</td><td>5846.4</td></tr><tr><td>精简版 (SIMD8)</td><td>0.772</td><td>433.8</td></tr></table>

图 14：MLX 版图。表 II：面积与功耗。

## B. 基准测试模型与硬件基线

(1) 从算法角度来看，我们使用 BERT、VIT 以及两个 LLM - Llama2-7B 和 InternLM2-7B 的代表性模型，评估了我们的混合稀疏方法（FFT 压缩和 BSMM）在准确性和计算量减少方面的效果，详见表 III，其缩写标记在括号中。我们还在 H100 上对 Llama2-7B 的两种 attention 实现进行了加速比比较。

(2) 从架构角度来看，MLX 为结构化算子提供了统一的执行模式。为了评估这如何转化为 LLM 工作负载上的硬件效率，我们使用一组具有代表性的硬件基线来评估 MLX，如表 IV 所总结。

为了确保公平和全面的比较，我们采用了双管齐下的评估策略：真实的流片设计（1 TOp/s）与 NVIDIA GPU 进行比较，而在我们的模拟器 [36] 中调整了精简的 256 GOp/s 版本，以匹配几种先前具有相同峰值吞吐量的算法-加速器协同设计 [26, 29, 37, 38, 39, 40]。这些基线的性能数据直接引用自其原始论文。为了将 MLX 的架构优势与算法（ALGO）节省区分开来，我们还在表 IV 的最后一行列出了每项先前工作在 T 上实现的 FLOP 减少。选择 Jetson Xavier 是因为其具有可比的峰值性能（1.7 TFLOP/s 对比我们的 1 TFLOp/s）和相同的 12 nm 技术节点。最后，我们还与另外两款更先进的 GPU（AGX Orin 和 RTX-3090）进行了比较，以证明我们效率提升的通用性。

## VII. 评估

## A. 验证算法改进

如图 15 所示，我们在多个模型上评估了 FFT-CMP 和分层 BSMM，以确认所提出的结构化算子在典型的 Transformer 中提供了可预测的准确度与计算权衡。

ViT 模型 [2, 42] 被纳入其中，因为其适中的规模允许从头开始完整训练，从而能够对我们蝴蝶稀疏方法进行干净、面向理论的验证。用基于块的分解（“bd.\*”）替换密集投影可将 FLOPs 减少 45-55%，但仅有轻微的准确度损失。使用像 FNet（“fnet.fft”）[9] 中的 2D-FFT token 混合可以实现类似的计算减少，但会遭受 2-3% 的准确度下降。相比之下，我们在 s=0.5 时的 FFT-CMP 实现了 65% 的 FLOP 减少，相对于密集基线仅有 1.6% 的准确度下降，在效率和准确度上都优于现有的基于 FFT 的 transformer [9]。

表 III：单层 Transformer (T) 和 5 个模型 (V, F, B, I, L)。
<table><tr><td rowspan=1 colspan=1>基准</td><td rowspan=1 colspan=2>Trans.  VIT(T) [41]](V) [42]]</td><td rowspan=1 colspan=1>|FABNet||(F)[29] |</td><td rowspan=1 colspan=1>BERT(B) [1]|</td><td rowspan=1 colspan=1>BERT(B0)[43]</td><td rowspan=1 colspan=1>InternLM2 ||-7B (I) [44]</td><td rowspan=1 colspan=1>Llama2-7B(L) [27]</td></tr><tr><td rowspan=2 colspan=1>ND</td><td rowspan=1 colspan=1>1K</td><td rowspan=1 colspan=1>196</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>8K</td><td rowspan=2 colspan=1>5121K</td><td rowspan=2 colspan=1>1K to 4K4K</td><td rowspan=2 colspan=1>128 to 2K4K</td></tr><tr><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>1K</td><td rowspan=1 colspan=1>768</td><td rowspan=1 colspan=1>1K</td></tr></table>

表 IV：基线加速器与目标工作负载。
<table><tr><td>硬件</td><td>频率 (GHz)</td><td>峰值性能 (Op/s)</td><td>工艺节点 (归一化比率6)</td><td>功耗 (W)</td><td>基准</td><td>T上的算法FLOP节省</td></tr><tr><td>MLX</td><td>1.0</td><td> $\begin{array} { c } { { 1 \mathrm { T } ^ { 1 } \ \mathrm { ( F P 1 6 ) } } } \\ { { 2 5 6 \mathrm { G } ^ { 2 } } } \end{array}$ </td><td>12 nm</td><td> $\begin{array} { l } { 5 . 8 5 ^ { 1 } + 0 . 6 ^ { 3 } } \\ { 0 . 4 1 ^ { 2 } + 0 . 1 1 ^ { 3 } } \end{array}$ </td><td>All</td><td>4.1 (s=0.75) 6.1 (s=0.5)</td></tr><tr><td>Jetson Xavier</td><td>1.0</td><td>1.7T4 (6T5)</td><td>12 nm</td><td>15</td><td>L</td><td>1</td></tr><tr><td>FABNet[29]</td><td>0.2</td><td rowspan="6">256G</td><td>16nm (FPGA)</td><td>11.35</td><td>T, F</td><td>13.5</td></tr><tr><td>SpAtten[37]</td><td>1.0</td><td>40 nm (5×)</td><td>1.06</td><td>T</td><td>3.0</td></tr><tr><td>DOTA[26]</td><td>1.0</td><td>22 nm (2×)</td><td>0.86</td><td>T</td><td>5.0</td></tr><tr><td>Sanger[38]</td><td>1.0</td><td>55 nm (7×)</td><td>0.80</td><td>T</td><td>5.9</td></tr><tr><td>ViTALity[39]</td><td>0.5</td><td>28 nm (3×)</td><td>1.46</td><td>T</td><td>5.9</td></tr><tr><td>BitVert[40]</td><td>0.8</td><td>28 nm</td><td>0.17 (int8)</td><td>T</td><td>4.0</td></tr></table>

<sup>1</sup> 完整设计；<sup>2</sup> 精简设计；<sup>3</sup> 内存功耗；<sup>4</sup> CUDA 峰值性能；<sup>5</sup> TCU 峰值性能；<sup>6</sup> 使用 $P \propto \hat { C } \cdot V _ { d d } ^ { 2 } \cdot f$ 模型归一化至 12 nm 节点。

BERT [1] 也足够小以进行重训练，这允许我们使用语义区间长度 L（公式 1）结合分层 BSMM 稀疏性（(32)）来应用逐层 FFT 压缩。图 15(b) 展示了将我们的混合方法<sup>¯</sup>应用于最后 k 层且 (s=0.5) 的五种情况。随着 k 的增加，计算量可预测地下降，而准确度仅适度下降。替换所有 12 层实现了 69% 的 FLOP 减少，仅伴随 1.75% 的 EM 和 1.3% 的 F1 损失。

图 15(c,d) 评估了 LLM 中 attention 阶段的 FFT 压缩和 QKV 投影中基于块的 BSMM，并使用 LoRA 微调 [45] 来精炼压缩层。我们在 Winogrande-xl [46] (N=512)、Wikitext-2/103 [47] (1K/2K) 和 Ada-LEval [48] (1K/2K/4K) 上进行了测试。我们逐步将结构化算子应用于超过 60% 的 transformer 层。在分别统一设置 s=0.75 和 s=0.5 的情况下，我们在修改层内减少了 57%–64% 和 67%–72% 的 QKV+Attention 计算，在所有变体中整体准确度下降均低于 1.45%。尽管某些层可以容忍更激进的压缩，但我们使用统一的 s 来清楚地展示对压缩强度的敏感性，并避免逐层调整。（例如，ada-2k/4k 和 wiki-103），具有 GQA [49] 的 InternLM2-7B 在 s=0.5 时由于降低了 KV 投影成本而产生了更大的节省。在自回归文本生成中，我们观察到压缩后的模型可以在更少的 epoch 内收敛，并产生略低的困惑度。在图 16 中，我们还评估了分层 BSMM 对块大小 B ∈ {16, 32, 64} 的敏感性。较大的 B 实现了更大的线性层 FLOP 减少，但通常会导致更大的准确度损失。在我们评估的长上下文设置中，B = 32 提供了最佳的权衡，同时 B 可以进一步与 FFT 压缩 s 共同调整，以达到不同的准确度-效率点。

H100 上的性能：为了评估现代 GPU 处理蝴蝶稀疏性的能力，我们在两个基准下将混合压缩的 Llama2-7B 部署在 H100 上：eager attention 和 FlashAttention2 (FA) [50, 51]，使用保守的稀疏性设置 - (s=0.5, B=32)。图 17 显示了相对于原始模型的加速。在计算密集型的 prefill 阶段，对于长序列，FFT-CMP 相对于 eager 实现了高达 2.72× 的加速，相对于 FA 实现了 1.64× 的加速，而对短序列则几乎没有收益。在 H100 上的增益较为温和，因为 FFT-CMP 在 PyTorch 层面运行，未与 FA 融合，并且 TensorCores 对蝴蝶稀疏性的支持有限，导致执行回退到 CUDA 核心。在 decode 阶段，FFT-CMP 减少了 KV-cache 流量，并与 block-BSMM 一起产生了 1.4–1.9× 的端到端加速。

![](images/23c0207fadcf399a58898c813b50c1ae20ff60ac4af7754ceb95db40de70b682.jpg)

![](images/cfca7e2c10efe0a1db994bf8952056b616ca9de2dffa49fd33919115d1f88421.jpg)  
图 15：FFT cmp. 和 BSMM 下的准确度与效率敏感性（计算减少量在 Llama2 和 InternLM2 的代表性稀疏化层 (>60%) 中的 QKV proj. 和 attn. 上测量）。

![](images/b92c4b6511b8381b0e1e7e17b81b062b3b6a786e991278562482e4546986883d.jpg)  
图 16：在固定 s=0.75 下三个模型对块大小 B 的准确度与困惑度敏感性。

![](images/fbec223fdf4ec6650198208a4553bc827a40822ccefc2a1cd45d80ba630a16e1.jpg)  
<sup>Sequence</sup> <sup>length</sup>图 17：Llama2-7B 上的 H100 加速 (s=0.5, B=32)。

![](images/16481a3c71159be6996ba566202d25cfbc80345a5852027962a20e24c26b2156.jpg)

![](images/887dd23dbc4da8ad39b25f830057ee5dcaf709a7f87546d07bb9437207b13bcc.jpg)

![](images/93034da551cb7ab7326940ed490d6536ebece036b4ba0f98421c9713a2a70729.jpg)

## B. MLX 性能

先前的稀疏加速器：图18将MLX与五个具有代表性的稀疏加速器进行了比较，使用了从原始论文 [26, 29, 37, 38, 39, 40] 中引用的并经过工艺归一化的能耗数据，以及表IV中的工艺缩放因子。SpAtten 是基线（“1.0”）。在 $s { = } 0 . 7 5 / 0 . 5$ 的两种设置下，得益于在 attention 和 projection 中统一的蝴蝶加速，MLX 在动态稀疏下相比前三个加速器实现了 2.93–4.10× 和 4.14– 5.8× 的加速。与针对低秩视觉 Transformer 的 ViTALiTy 相比，MLX 在具有可比的 FLOP 减少量的情况下，实现了 1.28× 和 1.81× 的加速。MLX 的性能也超过了 BitVert 2.3×；BitVert 报告了更高的节能效果（2×），这主要归因于其 INT8 精度，而 MLX 运行在 FP16。图18(c) 进一步报告了软硬件亲和度（通过 FLOP 节省量归一化的加速比）。MLX 获得了持续的高亲和度，因为 BSMM/FFT 是强 FMA 主导的：大部分周期用于 PE 中的常规 MAC 操作，只有适度的控制和簿记开销，这与需要数据相关索引或选择的不规则稀疏内核形成对比。这也突显了部署蝴蝶稀疏性的实际便利性。

图18：在单个 Transformer 块上与先前稀疏加速器的比较（N = 1024, D = 512）。
<table><tr><td rowspan=1 colspan=1>Ours FABNet Ratio</td></tr><tr><td rowspan=1 colspan=1>LUT 410k 358k   1.14</td></tr><tr><td rowspan=1 colspan=1>FF 620k 536k   1.15</td></tr><tr><td rowspan=1 colspan=1>DSP 512640    0.80</td></tr><tr><td rowspan=1 colspan=1>TABLE V: FPGARe-</td></tr></table>

source 使用量比较。

![](images/1a2df41028f15b861a5281cbfdb8e349177be4ce09fd706c3b27651dab012712.jpg)  
图19：相对 FABNet-Large 的加速比。

真实的蝴蝶加速器：FABNet [29] 是最接近的先前设计，提出了一种基于 FPGA 的蝴蝶-加速器协同设计，对 attention 使用 2D-FFT，对 FFN 使用全局 BSMM，不包括指数运算符。我们在 MLX 上重新实现了相同的模型和参数设置。图19 显示，在不同上下文长度下，MLX 实现了 1.19×–1.30× 的端到端加速，在此工作负载设置下具有 1.14× 的 LUT 开销（表V）；在 FABNet 式部署中，LUT 是限制性的 FPGA 资源 [52]。分解这些增益，2D-FFT attention 提升了 1.11×–1.23×，而 BSMM-FFN 提升了 1.21×–1.31×。FFT 侧较小的增益与 FABNet 对复数值蝴蝶操作的更强专业化相一致，这缩小了 MLX 的 FFT 提升空间。在 512 处的峰值加速出现在工作负载恰好适应 MLX 最大单级 BSMM 占用空间时，从而避免了阶段转换及相关的 SPM 往返。

NVIDIA Xavier GPU：图20 比较了在 Jetson Xavier 和 MLX 上，Llama2-7B 针对短（256）和长（8K） Token 输入的八个内核。在图20(a) 中，与 Xavier 的密集 TensorCore 内核相比，MLX 的蝴蝶稀疏内核实现了 3.1× 的加速和 3.2× 的节能。图20(b)

![](images/272a60750ae51fe2a5f3fe518eec0dc5380de833dca6e91d98097afced48b3c6.jpg)  
图20：完整设计相对 NVIDIA Jetson Xavier 的加速比。

![](images/0a89efd6159a8a316c4e2627bc132a23e6d2acac43b20df1da69ddf49dccdfca.jpg)

![](images/a2ba8abf272c801cab46526f2e977bef5c2d0ecff6d3fe54b3c04c707c9e138b.jpg)  
图21： 在不同上下文长度下，Llama2-7B 相对 Jetson Xavier 的端到端加速比。 内存使用量 (GB)。

进一步表明，相比稀疏化的 CUDA 执行，平均实现了 3.2× 的加速和 3.1× 的节能。在 GPU 上，密集内核通常使用 Tensor Cores，而蝴蝶和结构化稀疏内核通常在 CUDA cores 上运行。这压缩了稀疏性带来的相对增益，同时也突显了 MLX 在结构化加速方面专业化的价值。

图21 展示了 MLX 上的稀疏化 Llama2-7B 与 Xavier 上的密集模型之间的端到端比较。所有推理算子，包括 RMSNorm 和位置编码，均由 MLX 通过指令驱动的可编程性以及我们完整设计中所需的计算单元（向量洗牌和超越函数支持）来支持。由于其 16 GB 的内存容量，Xavier 无法维持超过 512 Token 的上下文，而 MLX 可以处理长达 2048 的序列。尽管当密集线性层占主导地位时加速比会下降，但通过 BSMM 和 FFT-CMP 稀疏化，MLX 在长上下文设置中保持了稳健的优势。

## C. 资源利用率和可扩展性

图22 总结了 BSMM 和 FFT-CMP 内核上的 PE 利用率。对于小尺寸，内核启动开销约为 17%，但随着内核尺寸增大，它降至 12% 以下。我们将加载/存储/传输单元分组为统一的数据供给流水线，这表现出一致的延迟行为。BSMM 和 FFT 显示出相似的趋势，因为两者都映射到多级蝴蝶算子，仅存在由实数与复数算术引起的微小差异。总体计算利用率达到约 90%，表明我们的指令调度有效地隐藏了数据移动延迟和流水线空闲。

我们使用尺寸为 $N { = } \{ 5 1 2 , 1 \mathrm { K } , 2 \mathrm { K } , 4 \mathrm { K } , 8 \mathrm { K } \}$ 且 D=512 的 Transformer 块评估可扩展性，批量大小为 8，以为所有设计提供足够的并行度。通过将 8 路 和 32 路 SIMD 与 4×4 和 8×8 网格组合，测试了四种配置，每种配置提供 4× 的峰值计算缩放。如图23 所示，两个维度均接近线性缩放，产生 3.9× (SIMD) 和 3.6× (网格) 的平均加速，联合缩放时高达 14×。SIMD 直接受益于 Token 级别的并行性，但由于多端口寄存器文件的成本和有限的每层并行性，无法无限增长。

![](images/33ab3b8666bb09a7c83c663642c347f0ad3f7d63a5174b805242386b274a53ed.jpg)

图22：PE 资源利用率分解。  
![](images/c43674147d46e622e5fcb96785eca2a1d5a89efff4c5d2ed03ca58c953272814.jpg)  
图23：SIMD 程度和网格尺寸的可扩展性。

网格缩放通过利用层间流水线提供了一条更可持续的路径。轻量级的跳步互连减少了多跳延迟，并为 8×8 网格实现了接近线性的缩放，在 12 nm 工艺下即使频率为 1 GHz，也仅有近 6.2% 的面积开销和适度的时序开销。

## D. 结构化 LLM 工作负载的敏感性分析

图24在批大小为32的条件下，跨多个模型和序列设置，将MLX与更强的AGX Orin和RTX-3090在我们的结构化工作负载套件上进行了比较。尽管峰值算力和带宽大幅低于对手，MLX在部分蝶形算子上仍然优于Orin。在若干小型FFT/BSMM案例中，与RTX的差距也缩小了，这部分归因于MLX紧凑的互连结构和更低的启动开销。从FFT-CMP到具有增大块尺寸的BSMM和SWA，计算模式逐渐变得更粗粒度且更具分块规则性。这种增加的规则性暴露出更多批量并行性，并更自然地映射到GPU的密集执行上，因此MLX的速度优势相应减弱。即便如此，MLX在两个SWA案例上仍保持3.6倍和2.3倍的平均归一化加速比（W：窗口宽度，Q：块大小）。

为排除峰值资源差异的影响，图25报告了roofline利用率，即在计算和带宽约束下，实际达到的性能相对于roofline极限的归一化值。蝶形结构化算子在MLX上实现了52%–84%的利用率，而在Orin上为12%–29%，在RTX上为8.2%–31%，表明MLX在执行深度分阶段依赖方面更为高效。对于SWA，MLX的重叠流水线维持了43%–75%的FMA利用率。剩余差距主要源于窗口化KV流量的带宽损失，但MLX仍超过GPU基线（10.8%–31%和8.9%–28%）。总体而言，这些结果表明MLX超越了蝶形稀疏性，能够高效支持更广泛的结构化算子。

![](images/6185bd2c07f6a281ad95292a2bf1f65cc03950484af1a0674c76e22415751d93.jpg)

图24：在Orin和RTX-3090上的结构化算子扫描。  
![](images/932ca116af19c31a1dbdd63728a9a9755ba060012e6c6c2dbd78a568ab286fcb.jpg)  
图25：四个代表性模型和序列案例下FMA操作的利用率 $( P _ { \mathrm { a c h i e v e } } /$ min $( P _ { \mathrm { p e a k } } ,$ , OI · BW ))。

## E. 泛化性与灵活性讨论

MLX通过将语义感知FFT和层次化BSMM分解为可参数化的CDC块 $( L , B )$ 来处理多样形状和长序列，确保效率随块数量扩展的同时保持占用面积和局部性。MLX主要依赖结构化、可预测的数据流，该数据流被预编译到CDC边界并通过标记块执行。轻量级辅助运行时通过调度这些预定义的CDC序列来提供必要的灵活性，处理粗粒度不规则性（例如，分桶MoE）。对于更不规则的模式，MLX通过基于信用的流控维持功能正确性，尽管极端不平衡可能引入气泡并降低利用率。由于当前设计点保持传输紧凑高效，支持细粒度动态模式将需要谓词传输（掩码/段编码）和额外的控制状态，这反映了一个明确的灵活性-效率权衡，留待未来工作解决。

## VIII. 相关工作

我们讨论稀疏Transformer加速和空间数据流方面的相关工作，以将MLX置于先前研究的背景中。稀疏加速器设计：最近的加速器[24, 26, 37, 53, 54, 55]主要针对动态或非结构化稀疏，或利用图工作负载中的类稀疏模式[56, 57, 58]。结构化稀疏在实践中仍较少被探索。虽然先前的分析框架研究了其潜在收益[59, 60]，但它们并未扩展到端到端的实际硬件。相比之下，MLX提供了实用的调节旋钮来调整attention块中的结构化稀疏粒度[61, 62]，从而实现跨模型变体的自适应权衡。FABNet [29]和EIE [63]等协同设计系统与我们的目标最为接近，而这些面向蝶形的设计[64, 65, 66]缺乏对现代AI工作负载的通用性和适应性。

LLM加速：其他LLM系统主要关注数值优化，如量化[67]和比特级稀疏[40, 68, 69]，以及在线-离线混合KV缓存量化[22, 70]。MLX则通过结构化序列/隐藏压缩探索了一个正交维度，这与低位算术互补，并可能实现进一步减少内存流量的混合优化。

空间数据流范式：空间数据流架构提供依赖驱动的执行和ISA暴露的资源分配，实现激进的软件流水线[30, 71]。先前的设计通过地址生成[72, 73]、缓冲优化[11]和稀疏感知执行[10, 74]增强稀疏支持。空间阵列中的执行由PE和互连[75, 76, 77]共同塑造，并在时序和资源维度上探索异构性[78]。MLX与传统数据流设计的不同之处在于，它将可预测依赖的规则性作为一等映射抽象加以利用。这使得大型、规则化的数据流图可以折叠到紧凑的空间阵列上，有效地将逻辑图规模与物理阵列大小解耦——这是先前设计未明确解决的能力。

## IX. 结论

本工作提出了MLX，一种用于结构化LLM加速的统一算法-架构协同设计。通过将语义FFT压缩与层次化蝶形分解相结合，MLX暴露出可预测、精度可调的稀疏性。我们识别出FFT、BSMM及相关结构化算子中共同的分阶段依赖模式，并构建多层执行（Multi-Layer Execution）将这些深度算子折叠到紧凑的空间数据流中。MLX集成了跳跃路由、基于标记的调度和解耦流水线，以在分阶段执行下维持高利用率。在Llama2-7B和InternLM2-7B上的实验表明，计算量减少57%–72%，精度损失极小。我们的12 nm原型展示了相对于边缘GPU和先前稀疏加速器的竞争优势，同时可扩展到更大的网格和更长的序列。总体而言，这些结果表明MLX超越了蝶形稀疏性，为高效加速更广泛类别的结构化算子提供了通用基础。

## 致谢

本工作受国家重点研发计划（项目编号2023YFB4503500）、江苏省前沿技术研发计划（项目编号BF2024029）、中国科学院基础研究青年科学家项目（项目编号YSBR-029）、国家自然科学基金（项目编号62502498）以及北京市自然科学基金（项目编号L234078）资助。

[1] J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep bidirectional transformers for language understanding,” in NAACL-HLT 2019.

[2] A. Dosovitskiy, L. Beyer et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in 9th International Conference on Learning Representations, ICLR 2021.

[3] T. Dao, A. Gu, M. Eichhorn, A. Rudra, and C. Re,´ “Learning fast algorithms for linear transforms using butterfly factorizations,” in Proceedings of the 36th International Conference on Machine Learning, ICML 2019.

[4] T. Dao, B. Chen, N. S. Sohoni, A. D. Desai et al., “Monarch: Expressive structured matrices for efficient and accurate training,” in International Conference on Machine Learning, ICML 2022.

[5] B. Li, S. Pandey, H. Fang et al., “FTRANS: energyefficient acceleration of transformers using FPGA,” in ISLPED, 2020.

[6] T. Dao, A. Gu, M. Eichhorn, A. Rudra, and C. Re,´ “Learning fast algorithms for linear transforms using butterfly factorizations,” in Proceedings of the 36th International Conference on Machine Learning, ICML 2019.

[7] M. Zaheer, G. Guruganesh, A. Dubey, J. Ainslie, C. Alberti, S. Ontan˜on, P. Pham, A. Ravula, Q. Wang, L. Yang,´ and A. Ahmed, “Big bird: Transformers for longer sequences,” CoRR, vol. abs/2007.14062, 2020.

[8] P. Zhang, X. Dai, J. Yang, B. Xiao, L. Yuan, L. Zhang, and J. Gao, “Multi-scale vision longformer: A new vision transformer for high-resolution image encoding,” in International Conference on Computer Vision, ICCV 2021.

[9] J. Lee-Thorp, J. Ainslie, I. Eckstein, and S. Ontan˜on,´ “Fnet: Mixing tokens with fourier transforms,” in NAACL, 2021.

[10] V. Dadu, J. Weng, S. Liu, and T. Nowatzki, “Towards general purpose acceleration by exploiting common datadependence forms,” in Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, MICRO 2019.

[11] H. Kwon, A. Samajdar, and T. Krishna, “MAERI: enabling flexible dataflow mapping over DNN accelerators via reconfigurable interconnects,” in ASPLOS 2018.

[12] A. Parashar, M. Pellauer, M. Adler, B. Ahsan, N. Crago, D. Lustig, V. Pavlov, A. Zhai et al., “Triggered instructions: a control paradigm for spatially-programmed architectures,” in Proceedings of the 40th Annual International Symposium on Computer Architecture, 2013.

[13] Z. Chen, Z. Qu, Y. Quan, L. Liu, Y. Ding, and Y. Xie, “Dynamic N:M fine-grained structured sparse attention mechanism,” in Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming, PPoPP 2023.

[14] R. Prabhakar, R. Sivaramakrishnan, D. Gandhi, Y. Du et al., “Sambanova sn40l: Scaling the ai memory wall
with dataflow and composition of experts,” in 57th IEEE/ACM International Symposium on Microarchitecture (MICRO), 2024.

[15] G. Shen, J. Zhao, Q. Chen, J. Leng, C. Li, and M. Guo, “SALO: an efficient spatial accelerator enabling hybrid sparse attention mechanisms for long sequences,” in 59th ACM/IEEE Design Automation Conference, 2022.

[16] J. Zhao, P. Zeng, G. Shen, Q. Chen, and M. Guo, “Hardware–software co-design enabling static and dynamic sparse attention mechanisms,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2024.

[17] Y. Qin, Y. Wang, D. Deng, Z. Zhao, X. Yang, L. Liu, S. Wei, Y. Hu, and S. Yin, “Fact: Ffn-attention cooptimized transformer architecture with eager correlation prediction,” in Proceedings of the 50th Annual International Symposium on Computer Architecture, 2023.

[18] L. Zheng, E. Riccietti, and R. Gribonval, “Efficient identification of butterfly sparse matrix factorizations,” SIAM Journal on Mathematics of Data Science, vol. 5, no. 1, pp. 22–49, 2023.

[19] D. Y. Fu, H. Kumbong, E. Nguyen, and C. Re, “Flashfft-´ conv: Efficient convolutions for long sequences with tensor cores,” in The Twelfth International Conference on Learning Representations (ICLR), 2024.

[20] A. K. Kamath, R. Prabhu, J. Mohan, S. Peter, R. Ramjee, and A. Panwar, “Pod-attention: Unlocking full prefilldecode overlap for faster LLM inference,” in ASPLOS 2025.

[21] Y. Zhao, D. Wu, and J. Wang, “ALISA: accelerating large language model inference via sparsity-aware KV caching,” in 51st ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2024.

[22] M. Kim, S. Hong, R. Ko, S. Choi, H. Lee, J. Kim, J. Kim, and J. Park, “Oaken: Fast and efficient LLM serving with online-offline hybrid KV cache quantization,” in Proceedings of the 52nd Annual International Symposium on Computer Architecture, ISCA 2025.

[23] B. Li, S. Cheng, and J. Lin, “tcfft: A fast half-precision fft library for nvidia tensor cores,” in 2021 IEEE International Conference on Cluster Computing (CLUSTER), 2021.

[24] Y.-H. Chen, T.-J. Yang, J. Emer, and V. Sze, “Eyeriss v2: A flexible accelerator for emerging deep neural networks on mobile devices,” IEEE Journal on Emerging and Selected Topics in Circuits and Systems, 2019.

[25] T. Liu, Z. Fan, W. Li, Z. Wang, Y. Qiu, S. Tang, H. Wu, Y. Liu, X. Ye, and D. Fan, “Dfgas: Exploring the balance of hw-sw scheduling through the dfg-aware scheme,” ACM Trans. Archit. Code Optim., 2025.

[26] Z. Qu, L. Liu, F. Tu, Z. Chen, Y. Ding, and Y. Xie, “DOTA: detect and omit weak attentions for scalable transformer acceleration,” in ASPLOS ’22.

[27] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi et al., “Llama 2: Open foundation and fine-tuned chat models,” CoRR, vol. abs/2307.09288, 2023.

[28] M. H. Rasheed, O. M. Salih, M. M. Siddeq, and M. A. Rodrigues, “Image compression based on 2d discrete fourier transform and matrix minimization algorithm,” Array, vol. 6, p. 100024, 2020.

[29] H. Fan, T. Chau, S. I. Venieris, R. Lee et al., “Adaptable butterfly accelerator for attention-based nns via hardware and algorithm co-design,” in 55th IEEE/ACM International Symposium on Microarchitecture, MICRO, 2022.

[30] T. Nowatzki, V. Gangadhar, N. Ardalani, and K. Sankaralingam, “Stream-dataflow acceleration,” in Proceedings of the 44th Annual International Symposium on Computer Architecture, ISCA 2017.

[31] T. Plano and J. Buhler, “Scheduling irregular dataflow pipelines on SIMD architectures,” in WPMVP@PPoPP ’20: Sixth Workshop on Programming Models for SIMD/Vector Processing, 2020.

[32] T. J. Repetti, J. P. Cerqueira, M. A. Kim, and M. Seok, “Pipelining a triggered processing element,” in Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture, MICRO 2017.

[33] O. Ragheb, R. Beidas, and J. Anderson, “Statically scheduled vs. elastic cgra architectures: Impact on mapping feasibility,” in 2023 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), 2023, pp. 468–475.

[34] A. Shukla and Y. Simmhan, “Toward reliable and rapid elasticity for streaming dataflows on clouds,” in 38th IEEE International Conference on Distributed Computing Systems, ICDCS 2018, Vienna, Austria, July 2-6, 2018.

[35] H. Zhao, Y. Xiang, Y. Liu, X. Ye, D. Zeng, J. Yang, W. Cui, Q. Chen, J. Leng, and M. Guo, “DACO: unlocking latent dataflow opportunities in edge-side SIMT accelerators,” in Advanced Parallel Processing Technologies - 16th International Symposium, APPT 2025.

[36] X. Ye, D. Fan, N. Sun, S. Tang, M. Zhang, and H. Zhang, “Simict: A fast and flexible framework for performance and power evaluation of large-scale architecture,” in International Symposium on Low Power Electronics and Design (ISLPED), 2013.

[37] H. Wang, Z. Zhang, 和 S. Han, “Spatten: Efficient sparse attention architecture with cascade token and head pruning,” 载于 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2021, pp. 97–110.

[38] L. Lu, Y. Jin, H. Bi, Z. Luo, P. Li, T. Wang, 和 Y. Liang, “Sanger: A co-design framework for enabling sparse attention using reconfigurable architecture,” 载于 MICRO ’21: 54th Annual IEEE/ACM International Symposium on Microarchitecture.

[39] J. Dass, S. Wu, H. Shi, C. Li, Z. Ye, Z. Wang, 和 Y. Lin, “Vitality: Unifying low-rank and sparse approximation for vision transformer acceleration with a linear taylor attention,” 载于 IEEE International Symposium on High-Performance Computer Architecture, HPCA 2023.

[40] Y. Chen, J. Meng, J. Seo, 和 M. S. Abdelfattah, “BBS: bi-directional bit-level sparsity for deep learning acceleration,” 载于 57th IEEE/ACM International Symposium on Microarchitecture, MICRO 2024.

[41] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit 等, “Attention is all you need,” 载于 Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017.

[42] Y. Di, Z. Jiang, 和 H. Zhang, “A public dataset for fine-grained ship classification in optical remote sensing images,” Remote. Sens., 卷 13, 期 4, p. 747, 2021.

[43] P. Rajpurkar, J. Zhang, K. Lopyrev, 和 P. Liang, “Squad: 100, 000+ questions for machine comprehension of text,” 载于 Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016.

[44] Z. Cai, M. Cao, H. Chen, K. Chen, K. Chen, X. Chen, X. Chen 等, “Internlm2 technical report,” 2024. [在线]. 可用: https://arxiv.org/abs/2403.17297

[45] Y. Xu, L. Xie, X. Gu, X. Chen, H. Chang, H. Zhang, Z. Chen, X. Zhang, 和 Q. Tian, “Qa-lora: Quantizationaware low-rank adaptation of large language models,” 载于 The Twelfth International Conference on Learning Representations, ICLR 2024.

[46] K. Sakaguchi, R. L. Bras, C. Bhagavatula, 和 Y. Choi, “Winogrande: An adversarial winograd schema challenge at scale,” 载于 The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020.

[47] S. Merity, C. Xiong, J. Bradbury, 和 R. Socher, “Pointer sentinel mixture models,” 2016. [在线]. 可用: https://arxiv.org/abs/1609.07843

[48] C. Wang, H. Duan, S. Zhang, D. Lin, 和 K. Chen, “Ada-leval: Evaluating long-context llms with lengthadaptable benchmarks,” 2024. [在线]. 可用: https://arxiv.org/abs/2404.06480

[49] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron, 和 S. Sanghai, “GQA: training general-´ ized multi-query transformer models from multi-head checkpoints,” 载于 Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023.

[50] M. Pagliardini, D. Paliotta, M. Jaggi, 和 F. Fleuret, “Fast attention over long sequences with dynamic sparse flash attention,” 载于 Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems NeurIPS 2023.

[51] T. Dao, “Flashattention-2: Faster attention with better parallelism and work partitioning,” 载于 The International Conference on Learning Representations, ICLR 2024.

[52] S. Liu, J. Weng, D. Kupsh, A. Sohrabizadeh, Z. Wang 等, “Overgen: Improving fpga usability through domain-specific overlay generation,” 载于 Proceedings of the 55th Annual IEEE/ACM International Symposium on Microarchitecture, 2023.

[53] U. Bakhtiar, H. Hosseini, 和 B. Asgari, “Acamar: A dynamically reconfigurable scientific computing accelerator for robust convergence and minimal resource underutilization,” 载于 2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO).

[54] H. Wang, J. Fang, X. Tang, Z. Yue, J. Li, Y. Qin, S. Guan, Q. Yang, Y. Wang, C. Li, Y. Hu, 和 S. Yin, “SOFA: A compute-memory optimized sparsity accelerator via cross-stage coordinated tiling,” 载于 57th IEEE/ACM International Symposium on Microarchitecture, MICRO 2024.

[55] Z. Fan, W. Li, Z. Wang, T. Liu, H. Wu, Y. Liu, M. Wu, X. Wu 等, “Accelerating convolutional neural networks by exploiting the sparsity of output activation,” IEEE Trans. Parallel Distributed Syst., 2023.

[56] M. C. Jeffrey, S. Subramanian, C. Yan, J. Emer, 和 D. Sanchez, “A scalable architecture for ordered parallelism,” 载于 Proceedings of the 48th International Symposium on Microarchitecture, 2015.

[57] V. Dadu 和 T. Nowatzki, “Taskstream: accelerating taskparallel workloads by recovering program structure,” 载于 ASPLOS ’22, 2022.

[58] V. Dadu, S. Liu, 和 T. Nowatzki, “Polygraph: exposing the value of flexibility for graph processing accelerators,” 载于 Proceedings of the 48th Annual International Symposium on Computer Architecture, 2021.

[59] Y. N. Wu, J. S. Emer, 和 V. Sze, “Accelergy: An architecture-level energy estimation methodology for accelerator designs,” 载于 2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), 2019.

[60] Y. N. Wu, P.-A. Tsai, A. Parashar, V. Sze, 和 J. S. Emer, “Sparseloop: An analytical approach to sparse tensor accelerator modeling,” 载于 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO).

[61] J. Liu, S. Zeng, J. Zhao, L. Ding, Z. Wang, J. Li, Z. Zhu, X. Ning, C. Zhang, Y. Wang, 和 G. Dai, “TB-STC: transposable block-wise N:M structured sparse tensor core,” 载于 IEEE International Symposium on High Performance Computer Architecture, HPCA 2025.

[62] X. Xiong, Z. Chen, Y. Liang, M. Tian, J. Shang, J. Zhong, 和 D. Liu, “Dynax: Sparse attention acceleration with dynamic X: M fine-grained structured pruning,” 载于 ASP-LOS 2025.

[63] S. Han, X. Liu, H. Mao, J. Pu, A. Pedram, M. A. Horowitz, 和 W. J. Dally, “Eie: efficient inference engine on compressed deep neural network,” 载于 Proceedings of the 43rd International Symposium on Computer Architecture, 2016.

[64] D. Wang, X. Du, L. Yin, C. Lin, H. Ma, W. Ren, H. Wang, X. Wang 等, “Mapu: A novel mathematical computing architecture,” 载于 2016 IEEE International Symposium on High Performance Computer Architecture (HPCA).

[65] S. Fan, Z. Wang, W. Xu, R. Hou, D. Meng, 和 M. Zhang, “Tensorfhe: Achieving practical computation on encrypted data using gpgpu,” 载于 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2023.

[66] M. Garrido, “A survey on pipelined fft hardware architectures,” Journal of Signal Processing Systems, 2021.

[67] A. H. Zadeh, M. Mahmoud, A. Abdelhadi, 和 A. Moshovos, “Mokey: enabling narrow fixed-point inference for out-of-the-box floating-point transformer models,” 载于 ISCA ’22: The 49th Annual International Symposium on Computer Architecture, 2022.

[68] Y. Liu, W. Li, K. Zhang, Y. Liu, S. Wen, L. Wang, T. Liu, H. Wu, Z. Fan, X. Ye, D. Fan, 和 X. An, “Bitred: Taming non-uniform bit-level sparsity with a programmable RISC-V ISA for DNN acceleration,” 载于 Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS 2026.

[69] H. Lu, L. Chang, C. Li, Z. Zhu, S. Lu, Y. Liu, 和 M. Zhang, “Distilling bit-level sparsity parallelism for general purpose deep learning acceleration,” 载于 MICRO-54, 2021.

[70] H. Wang, Y. Li, H. Xu, Y. Wang, L. Liu, J. Yang, 和 Y. Han, “LAD: efficient accelerator for generative inference of LLM with locality aware decoding,” 载于 IEEE International Symposium on High Performance Computer Architecture, HPCA 2025.

[71] T. Nowatzki, N. Ardalani, K. Sankaralingam, 和 J. Weng, “Hybrid optimization/heuristic instruction scheduling for programmable accelerator codesign,” 系列 PACT ’18, 2018.

[72] Z. Li, P. Dangi, C. Yin, T. K. Bandara 等, “Enhancing CGRA efficiency through aligned compute and communication provisioning,” 载于 ASPLOS 2025.

[73] J. Weng, S. Liu, V. Dadu, Z. Wang, P. Shah, 和 T. Nowatzki, “Dsagen: Synthesizing programmable spatial accelerators,” 载于 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA).

[74] L. Wu, A. Lottarini, T. K. Paine, M. A. Kim, 和 K. A. Ross, “Q100: 数据库处理单元的架构与设计,” 载于 Proceedings of the 19th International Conference on Architectural Support for Programming Languages and Operating Systems, 2014.

[75] W. Li, Z. Fan, T. Liu, Z. Wang, H. Wu, M. Wu, K. Zhang, Y. Liu, N. Sun, X. Ye, 和 D. Fan, “DFU-E: 面向边缘 DSP 和 AI 应用的数据流架构,” IEEE Trans. Parallel Distributed Syst., 2025.

[76] R. Prabhakar, Y. Zhang, D. Koeplinger 等, “Plasticine: 面向并行模式的可重构架构,” 载于 Proceedings of the 44th Annual International Symposium on Computer Architecture, ser. ISCA ’17.

[77] H. Wu, W. Li, Z. Fan, Z. Wang, T. Liu 等, “缓解 DSP 应用数据流加速器中的传输延迟,” 载于 41st IEEE International Conference on Computer Design, ICCD 2023.

[78] J. Weng, S. Liu, Z. Wang, V. Dadu, 和 T. Nowatzki, “面向归纳矩阵算法的混合脉动-数据流架构,” 载于 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA).