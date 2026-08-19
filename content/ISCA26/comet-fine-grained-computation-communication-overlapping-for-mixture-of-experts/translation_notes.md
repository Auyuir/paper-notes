# {Comet}{}: Fine-grained Computation-communication Overlapping for Mixture-of-Experts 原文翻译

# {Comet}{}：面向混合专家模型的细粒度计算-通信重叠

# 引言

大语言模型的最新进展已经彻底改变了多个领域，包括自然语言处理 (Vaswani 2017; Touvron et al. 2023)、计算机视觉 (Z. Liu et al. 2021) 和多模态感知 (H. Liu et al. 2024; Cao et al. 2023)。这些成就表明，扩大模型规模可以显著增强模型能力。然而，模型参数的增长为这类巨型模型的部署带来了巨大挑战，因为计算资源日益成为制约模型能力的瓶颈 (Sharir, Peleg, and Shoham 2020)。

为此，混合专家模型 (Mixture-of-Experts, MoE) (Shazeer et al. 2017) 引入了一种稀疏结构，在该结构中仅激活部分参数。与在稠密模型中与所有参数进行交互不同，MoE 模型允许每个输入仅与少数专家进行交互。例如，Mixtral-8x7B 模型 (Jiang et al. 2024) 总共包含 450 亿个参数，而在运行时仅有 140 亿个参数处于激活状态。如今，MoE 已成为将模型扩展至万亿级以上参数的关键架构。

MoE 模型中参数规模的增加允许集成更多信息，但也给专家放置带来了挑战。一种典型的方法是将专家分布到不同的 GPU 上，因为单个 GPU 无法存储所有专家 (Lepikhin et al. 2020)。因此，在执行 MoE 层时，GPU 之间存在密集的数据交换需求。如 [fig:overall_breakdown](a) 所示，在几种流行的 MoE 模型的正向传播中，设备间的通信平均占总执行时间的 $47\%$。

MoE 执行过程分析。(a) 使用 Megatron-LM 在 8 个 H800 GPU 上执行 MoE 模型的时间分解。(b) 通过将专家计算 kernel 分为两部分来实现通信-计算重叠的示意图。

在分布式环境中，执行 MoE 层涉及数据接收、专家计算和数据传输，如 [fig:overall_breakdown](b) 所示。为了减少通信开销，一种有效的策略是对该过程进行流水线处理，使通信与专家计算重叠 (Hwang et al. 2023; He et al. 2022; Shi et al. 2023, 2024)。这种方法涉及将输入数据划分为更小的数据块，允许分解后的通信和计算阶段重叠。在 [fig:overall_breakdown](b) 的示例中，接收到的输入数据被分为两个数据块，这种粗粒度的重叠与非流水线执行相比减少了总体执行时间。

现有机制中的重叠仍然不够理想，主要存在两个低效问题。首先，随着分配给每个专家的数据块变小，分区专家的效率会下降，可能导致 GPU 计算资源利用不足（例如，分区后专家的总计算时间 $t_1+t_2$ 超过了原始时间 $t$）。粗粒度分区导致在初始和最终通信阶段（例如接收数据块 1 和发送数据块 2 时）出现不可避免的 GPU 空闲时间，这些阶段无法与计算重叠。因此，在保持计算效率的同时最小化这些阶段中的非重叠时间至关重要。这极具挑战性，因为通信和计算之间的数据依赖关系复杂，难以在细粒度上实现高效重叠。其次，由于 MoE 的动态特性，专家的输入形状在运行时各不相同，从而给 GPU 带来不同的通信和计算负担。像几乎所有先前的研究那样，将通信和计算任务封装在不同流上的独立 kernel 中，限制了对硬件资源的控制，并导致 kernel 性能不确定，从而阻碍了无缝重叠（例如，数据块 1 的计算与数据块 2 的接收未对齐）。因此，第二个挑战是在运行时动态确保在计算和通信工作负载之间精确分配硬件资源。

MoE 中复杂的数据依赖以及动态的计算和通信工作负载阻碍了现有系统实现高效的通信-计算重叠。因此我们提出了 Comet，一个能够实现细粒度通信-计算重叠以高效执行 MoE 的系统。Comet 引入了两个关键设计：1）一种依赖解析方法，用于识别 MoE 中通信和计算操作之间的复杂数据依赖，从而实现优化的计算-通信流水线结构。2）一种自适应工作负载分配方法，动态地将 GPU 线程块分配给 kernel 内的不同工作负载，平衡通信和计算以提高延迟隐藏。

Comet 通过分析通信和计算操作之间共享的数据缓冲区（称为*共享张量*）来促进 MoE 中的细粒度重叠。通过沿特定维度分解共享张量并重组张量数据以及算子内执行顺序，Comet 消除了通信和计算之间的粒度不匹配，从而实现细粒度重叠。为了确保精确的资源分配和有效的延迟隐藏，Comet 将通信和计算任务集成在融合的 GPU kernel 中。通过线程块专业化，Comet 隔离了通信对计算性能的影响，保持了高计算效率。通过调整分配给每个工作负载的线程块数量，Comet 有效地平衡了通信和计算延迟，并减少了重叠中的气泡。

我们已将 Comet 集成到 Megatron-LM (Shoeybi et al. 2019) 中，并通过各种并行策略验证了 Comet 的能力。我们在 Nvidia H800 和 L20 集群上的大量实验表明，与 SOTA MoE 系统相比，Comet 为典型的 MoE 层提供了 $1.96\times$ 的加速，为端到端 MoE 模型执行（Mixtral-8x7B (Jiang et al. 2024)、Qwen2-MoE (Bai et al. 2023)、Phi3.5-MoE (Abdin et al. 2024)）平均提供了 $1.71\times$ 的加速。Comet 已部署于加速由超过一万台 GPU 组成的生产集群中的大型 MoE 模型的训练和推理，节省了数百万 GPU 小时。Comet 引入了一种用于计算和通信的细粒度流水线编程模型。**我们将开源 COMET，旨在启发进一步的优化**，例如使用 Triton (OpenAI 2022) 或 TVM (T. Chen et al. 2018) 等编译器实现 Comet 中的编程模型。

# 背景与动机

## MoE 结构

跨两个GPU的MoE层示例，其中两个专家驻留在GPU0上，两个驻留在GPU1上。MoE层由两个前馈层组成。在此示例中，对于输入缓冲区中的每个token，它被分发到layer0中的三个专家（topk = 3），然后结果在layer1中合并。专家的形状在layer0中为N × K，在layer1中为K × N。

| 符号 | 描述                                                                |
|:-------|:---------------------------------------------------------------------------|
| $L$    | Transformer层数                                               |
| $E$    | 专家总数                                                    |
| $topk$ | 每个token被路由到的专家数量                             |
| *TP*   | Tensor并行大小                                                       |
| *EP*   | 专家并行大小                                                       |
| $W$    | 总并行世界大小 (*TP*$\times$ *EP*)                              |
| $M$    |  输入token长度 $\times$ Batch size  |
| $N$    | token的Embedding大小                                                  |
| $K$    | 专家中前馈层的隐藏大小                           |

符号说明。

混合专家对于高效扩展模型至关重要。通过实现参数的稀疏激活，MoE允许在不增加执行成本的情况下集成更多参数，从而提升性能。MoE的核心思想在于它由多个小模型（即*专家*）组成，并且token仅被路由到部分专家进行计算。[fig:background]展示了MoE层的典型执行流程，[tab:description]解释了描述MoE模型执行的符号。

每个输入token被分配给一个或多个专家进行计算，分配由各种算法决定 (Zuo et al. 2021; Y. Zhou et al. 2022; R. Liu et al. 2022)。一种常见的方法涉及一个门控网络 (Shazeer et al. 2017)，它为每个token选择 $topk$ 个专家，如 [fig:background] 所示，其中token A被路由到Expert0、Expert1和Expert3。经过两层通用矩阵乘法（GEMM）前馈层后，$topk$ 个输出被收集并归约以产生最终结果。

MoE layer0中的操作包括跨GPU的token通信（dispatch）和第一层专家计算（GEMM操作），从而建立通信-计算流水线。MoE的layer1包括第二层专家计算、token反调度和topk归约（combine），形成计算-通信流水线。

MoE采用两种主要的并行化策略：**专家并行** (Lepikhin et al. 2020) 和 **Tensor并行** (Shoeybi et al. 2019)。在专家并行中，不同专家的权重分布在不同的GPU上，每个专家的权重是完全完整的。Token被路由到其各自专家的对应设备。[fig:background] 展示了专家并行的一个案例，Expert0和Expert1驻留在GPU0上，其他的驻留在GPU1上。相反，Tensor并行沿着隐藏维度对所有专家的权重进行切分，每个GPU托管所有专家的一部分权重。专家并行和Tensor并行对于MoE的高效执行都至关重要。在MoE模型的实际部署中，通常会应用结合专家并行和Tensor并行的混合并行方法。

## 计算与通信重叠

随着MoE架构变得更大且更稀疏，MoE模型中通信所花费的时间比例变得越来越显著，如 [fig:overall_breakdown](a) 所示。如 [sec:intro] 所述，粗粒度的计算与通信重叠提供的优化潜力有限，且内核级调度对于动态工作负载效率不高。因此，在细粒度级别（例如token级别）执行重叠，并将计算和通信工作负载集成到融合的GPU内核中更为高效。采用这种更细粒度的重叠可以极大地释放进一步的优化机会。然而，在MoE中实现这种细粒度的重叠并非易事，根据我们的观察，存在两个主要障碍。

<embed src="figures/overview_v3.pdf" />

### 计算与通信之间的粒度不匹配

在MoE系统中，token作为数据移动的基本单位，如 [fig:background] 中Token A的移动所示。为了最大化GPU计算效率，高性能GEMM（GroupGEMM）内核通常将行组织成块进行处理。[fig:background] 中的紫色块代表了GEMM内核中的一个计算块，以128x128的块为例。因此，与单个专家相关的GEMM计算可能需要分布在多个GPU上的128个token。在细粒度上融合计算和通信时，token级数据传输与块级计算之间的差异带来了相当大的挑战：复杂的数据依赖性对重叠效率产生了不利影响，促使使用细粒度通信，而在融合内核中将细粒度通信与计算集成也具有挑战性。

**复杂的数据依赖。** 每个计算块所需的token由MoE的门控在运行时决定，随机分布在多个设备上。在所有必需的token可用之前，块的计算无法开始。如 [fig:background] 所示，Expert0的块在接收到Token A和Token B之前不会启动处理。因此，在粗粒度数据通信下，由于这种不规则且复杂的数据依赖，每个计算块的数据准备时间可能会延长。为了缓解这一问题，我们应采用细粒度通信，其中每个计算块仅通过统一虚拟地址 (Nvidia 2017) 直接读取或写入其所需的数据，并利用数据重组和重调度来有效地将其与计算隐藏。

**细粒度通信。** 将token级通信与块级计算集成以实现重叠并非易事。GPU之间的远程I/O操作表现出比本地GPU内存访问高得多的延迟。因此，在计算线程块内对远程数据token执行大量细粒度的读写操作会阻塞后续计算任务，导致内核效率显著下降。这一挑战在Hopper架构中尤为明显，其中计算内核利用Tensor Memory Accelerator (TMA) 硬件指令 (Nvidia 2022b) 来建立异步计算流水线。在这些异步流水线中集成长延迟的远程I/O操作会大大延长整体执行时间，对性能产生不利影响。因此，限制细粒度通信对计算内核的影响至关重要。 

我们的第一个洞察是，解决MoE模型中计算与通信之间的粒度不匹配是实现这两个过程高效重叠的关键。

### 计算与通信的多样化负载

MoE 的另一个特征是将 token 动态路由到不同的专家，导致在运行时专家的输入形状发生变化（例如，如 [fig:background] 所示，Expert0 和 Expert1 接收到的 token 数量不同）。这种可变性对 GPU 提出了不同的通信和计算需求。此外，硬件环境也可能具有不同的计算架构或网络拓扑，提供不同的计算能力和通信带宽。因此，实现计算与通信之间的无缝重叠需要动态调整 GPU 资源在不同负载上的分配，而这很难通过将负载封装为独立的 kernel 来实现。

我们的第二个洞察是，资源分配应在运行时于 kernel 内部进行自适应调整，以进一步实现通信与计算的无缝重叠。

# Comet 的设计

在本节中，我们介绍 Comet 的核心设计，这是一个 Mixture of Experts (MoE) 系统，通过流水线执行和通信与计算的细粒度重叠来优化 MoE 层的高效执行。我们的分析表明，MoE 架构具有两个不同的生产者-消费者流水线：通信-计算流水线和计算-通信流水线，如 [fig:overview] 所示。Token 如图所示遍历流水线，每个流水线内的操作通过一个共享缓冲区连接，称为 **shared tensor**，同时作为生产者的输出缓冲区和消费者的输入缓冲区。为了最小化整体延迟并提升流水线性能，Comet 引入了两个关键机制，旨在有效地重叠计算和通信负载。

1. 基于 shared tensor 的依赖解析：如前所述，通信与计算之间复杂的数据依赖对实现这些操作之间的无缝重叠构成了挑战。为了解决这一问题，我们通过分析 shared tensor 来检查数据依赖。我们的分析表明，shared tensor 可以被分解，相关的计算可以被重新调度以更有效地与通信重叠。因此，依赖解析过程在 shared tensor 上采用了两个关键优化策略，如 [fig:overview] 所示：沿特定维度分解 shared tensor 以打破粗粒度的数据依赖，以及重新调度计算以提高效率，同时确保有效的重叠。

2. 自适应负载分配：在通过依赖解析进行流水线优化之后，通信-计算重叠的模式变得更加一致和规律。为了有效隐藏细粒度的通信延迟，必须为通信和计算负载分配适当的硬件资源。鉴于这些负载根据输入形状、模型配置和硬件环境表现出不同的性能特征，自适应负载分配方案动态地平衡计算和通信。该方法为 MoE 系统生成高效的横向融合 kernel，从而优化延迟隐藏。

如 [fig:overview] 所示，Comet 首先利用基于 shared tensor 的依赖解析方法，通过分解和重新调度 shared tensor 来优化 MoE 结构中的流水线。根据重构后的流水线，Comet 随后通过自适应负载分配机制提供高效的融合 kernel。

## 基于 Shared Tensor 的依赖解析

我们现在介绍如何解决 MoE 中计算与通信之间的复杂数据依赖。其目标是通过分解和重新调度 shared tensor 来弥合通信和计算操作之间的粒度差异，以维持高效率。

MoE 层的 layer0（左）和 layer1（右）的生产者-消费者建模。对于 layer0 和 layer1，shared tensor 的全局大小均为 (M×topk,N)。

### 如何分解 shared tensor？

Shared tensor 作为生产者算子和消费者算子之间的桥梁，是实现重叠的关键。值得注意的是，重叠只有在生产者和消费者对 shared tensor 中的独立数据进行操作时才能发生，如 [fig:shared_tensor] 所示。因此，我们分析算子对 shared tensor 的访问模式，并沿消费者算子数据保持独立的特定维度对其进行分解。

例如，在 layer0 的通信-计算流水线中，消费者算子是 GEMM，shared tensor 作为其输入矩阵。在这种情况下，token 沿着 $M$（token）维度彼此独立，允许沿 $M$ 分解 shared tensor。然而，由于 GEMM tile 的计算涉及沿 token embedding 维度的乘法和规约以产生最终输出，因此沿该维度分解 shared tensor 是不可行的。

至于 layer1 的计算-通信流水线，消费者算子包含一个 top-K 规约，沿 $M$ 维度对 token 进行规约，导致该维度上 token 之间存在显著的相互依赖。因此，shared tensor 只能沿元素独立的 $N$ 维度进行分解。

### 如何重新调度分解后的共享张量？

在最细粒度上，共享张量可以被拆分为单独的行或列，使得消费者在接收到单行或单列后即可开始计算。然而，这种粒度会导致计算效率低下，尤其是在涉及计算密集型 GEMM 的流水线中，这些流水线通常以 tile 为单位进行组织和处理以实现高利用率。因此，在沿特定维度分解共享张量后，生成的子张量必须重新组织并重新调度为 tile 以进行计算。共享张量的重新调度遵循两个原则：重新调度的子张量应与原始计算 tile 粒度对齐，以保证计算效率。调度策略应优先处理消费者可以立即使用的生产者部分，允许消费者尽早开始执行。

分解并重新调度 MoE layer0 中的共享张量。在此图中，三个 expert 位于 Rank 0 上，每个 expert 都需要本地和远程数据进行计算。

Comet 利用 GroupGEMM 执行当前 rank 上所有 expert 的计算。在通信-计算流水线（MoE layer0）中，被 GroupGEMM 消费的共享张量沿 $M$ 维度分解。为了使 expert 能够尽早开始计算，根据 token 的源 rank 对其进行排序，如 [fig:layer0] 所示。随后设计 GroupGEMM 中 tile 的计算序列以最小化对远程数据的依赖，从包含本地 token 的 tile 开始计算，同时并发地进行其他远程 token 的传输。

在计算-通信流水线（MoE layer1）中，共享张量在经过 expert 的 GroupGEMM 处理后进行 top-k reduction。如前所述，共享张量沿 $N$ 维度分解。调整 tile 计算序列（[fig:layer1]）以使消费者算子能够在 expert 计算完全完成之前开始处理。GroupGEMM 操作不再按顺序计算每个 expert，而是按列执行。这种方法允许在计算出共享张量的前 $T_N$ 列后立即进行 reduction 和通信操作。如果不进行重新调度，token 只能在所有 expert 完成计算后才能被 reduce。

MoE layer1 重新调度后的计算序列（E = 3 且 topk = 3）。GroupGEMM 的执行顺序由颜色指示（黄色 → 绿色 → 蓝色 → 灰色）。这里，TN 表示 GroupGEMM 沿 N 维度的 tile 大小。

## 自适应负载分配

随着共享张量的分解和重新调度，MoE 中的流水线现在可以实现细粒度的重叠。为了确保有效的延迟隐藏，细粒度通信和计算的持续时间必须紧密对齐，以最小化流水线气泡。实现这一点需要为计算和通信进行自适应资源分配，并针对所涉及的具体任务进行定制。

Hopper 架构上 MoE layer1 的 Kernel 设计。每个 SM 仅容纳一个 thread block。红色箭头指示数据移动的路径。

### Thread block 专用化

在混合专家模型中实现通信与计算重叠的一种直接方法是，将整个流水线封装在同质的 thread block 中，将通信 I/O 集成到计算的序言或尾声中，这种策略在此称为垂直融合。通过垂直融合，thread block 并发执行，但重叠的发生是不规则的，导致通信和计算的延迟不确定，使得平衡它们的持续时间以实现延迟隐藏变得具有挑战性。此外，MoE 中 token 级别的细粒度 I/O 会显著降低底层 kernel 的计算效率，特别是在 Hopper 等先进架构上。为了解决这个问题，我们在通信和计算负载之间实现了 thread block 级别的隔离。这种隔离能够精确控制每个负载的硬件资源分配，促进计算和通信之间的平衡分配，从而最大化延迟隐藏。

[fig:cta_division] 描述了 Hopper 上 thread block 专用化 kernel 的细节，关键数据路径以红色标出。由于通信和计算之间的隔离，Comet 中的 GEMM thread block 使用了与融合前默认 GEMM 相同的实现。在 [fig:cta_division] 所描绘的场景中，其中 GEMM 是在 Hopper 架构上使用 CUTLASS 编译的，GEMM 的执行分布在不同的 warp 上。具体来说，生产者 warp 使用异步 TMA 指令将数据从全局内存加载到共享内存缓冲区，而消费者 warp 启动 tensor core MMA 操作 (Nvidia 2024a)。随后，通信 thread block 从全局内存中读取消费者 warp 生成的结果。在 top-K reduction 之后，通信 block 内的 warp 要么将 token 写入本地全局内存，要么将其传输到远程目的地。这种 thread block 专用化的编程模型很容易移植到其他架构，例如 Ampere 和 Volta，只需替换相应的计算 thread block 实现即可。

**硬件资源限制。** 所提出的 thread block 专用化 kernel 旨在最小化数据移动成本。然而，这种设计也必须应对硬件资源的限制。例如，从理论上讲，将通信 warp 与计算 warp 集成在同一个 thread block 中以消除冗余的全局内存访问是可行的。然而，warp 的线程数量限制限制了通信算子充分利用通信带宽。从另一个角度来看，用于通信的 warp 也会干扰同一 thread block 内的计算 warp。

### 自适应 thread block 分配

假设融合 kernel 有 $n$ 个 thread block，其中 $n_p$ 个 block 在流水线中作为生产者，$n_c$ 个 block 作为消费者。确定最佳划分点 $n_p/n_c$ 对于最大化整体效率至关重要。我们证明了最佳划分点受 MoE 层中输入的形状和特定模型配置的影响。为了研究这一点，我们在不同的输入序列长度和并行化策略下测量了 MoE layer1 的持续时间，如 [fig:cta_num] 所示。观察到在不同的配置下存在一个最佳划分点。

当输入 token 长度变化时，尽管通信和计算操作处理的数据大小都随输入长度缩放，但各自资源需求的可扩展性不同。因此，最佳划分点会随着输入长度的变化而偏移。例如，当 $\textit{TP}=8$ 时，当 $M$ 从 4096 变为 16384 时，最佳的 $n_c$ 从 18 变为 26。当修改模型配置（并行策略）时，最佳划分点会发生显著改变。例如，当 $\textit{TP}$ 从 8 调整为 4 时，在 $M=16384$ 的情况下，最佳的 $n_c$ 从 26 变为 46。

Comet 的库包含多个预编译的 kernel，每个 kernel 都有一个不同的划分点。在部署之前，对每种设置的最佳配置进行性能分析并存储为元数据。在运行期间，Comet 利用这些元数据来选择最佳的 kernel 执行。

具有不同分配给通信的 thread block 数量的 MoE layer1 kernel 的持续时间。thread block 的总数与 Hopper(132) 上的 SM 数量相同。该图显示了具有不同并行度的四种情况。

# 实现

Comet 由约 12k 行 C++ 和 CUDA 代码以及 2k 行 Python 代码组成。Comet 提供了一套用户友好的 Python API，开发者可以将其无缝集成到各自的框架中。在生产环境中，Comet 已在 Megatron-LM 中实现，用于大规模 MoE 训练。源代码将在 GitHub 上开源。

**针对 MoE 优化的 GEMM 内核。** Comet 广泛利用 CUTLASS 提供的编程模板来生成高效的 GEMM 内核。此外，它还融入了多种优化手段以最小化数据搬运开销。例如，在 MoE layer 0 中，GEMM 操作输入矩阵的行索引在每次 K 迭代时都需要从全局内存中读取。通过将这些行索引缓存到寄存器中，Comet 显著降低了全局内存访问开销。

**NVSHMEM 作为通信库。** 我们在内核中使用 NVSHMEM (Nvidia 2024d) 来支持细粒度通信。NVSHMEM 是一个专为 NVIDIA GPU 设计的通信库。它创建了一个跨越多个 GPU 内存的全局地址空间，可以通过细粒度的 GPU 发起操作和 CPU 发起操作进行访问。与面向高层通信操作的 NCCL (Nvidia 2024c) 不同，NVSHMEM 提供了更具组合性的低层 API，便于在内核内实现更细粒度的数据访问。

# 评估



|                |  L  |  E  | topk |  N   |   K   |
|:---------------|:---:|:---:|:----:|:----:|:-----:|
| Mixtral 8x7B   | 32  |  8  |  2   | 4096 | 14336 |
| Qwen2-MoE-2.7B | 24  | 64  |  4   | 2048 | 1408  |
| Phi-3.5-MoE    | 32  | 16  |  2   | 4096 | 6400  |

实验中使用的 MoE 模型配置。这些模型均在 Hugging Face (Huggingface 2022) 上开源。符号的含义在 [tab:description] 中解释。

## 实验设置

**测试环境。** 我们在一台配备 8 块 Nvidia H800 GPU（每块 80 GB 内存）的服务器上评估 Comet。这些 GPU 通过 NVLink 互联。我们的软件环境包括 CUDA 12.3、NVSHMEM 2.11、Pytorch 2.4.0 和 Megatron-LM（git-hash 6dbe4c）。

**对比目标。** 我们随后将 Comet 与多个基线进行对比。所有基线均在 Megatron-LM 上实现，Megatron-LM 是一个广泛采用的高性能模型执行框架，集成了混合并行策略。

基线包括： **Megatron-Cutlass**：使用 CUTLASS grouped GEMM (Nvidia 2024b) 实现 MoE 专家的 Megatron。 **Megatron-TE**：使用 Transformer Engine (Nvidia 2024e) 实现专家的 Megatron。Transformer Engine 是 Nvidia 的库，用于在 NVIDIA GPU 上加速 Transformer 模型。 **FasterMoE** (He et al. 2021, 2022)：FasterMoE 是一个 MoE 系统，它定制 All-to-All 通信以重叠专家的通信和计算操作。 **Tutel** (Hwang et al. 2023)：Tutel 提供了多种优化技术以实现高效且自适应的 MoE，包括自适应并行、二维分层 All-to-All 算法以及 GPU 上的稀疏计算快速编码/解码。

## 整体性能

我们在多个大型 MoE 模型上评估 Comet 的端到端性能，包括 Mixtral 8x7B (Jiang et al. 2024)、Qwen2-MoE (Bai et al. 2023) 和 Phi3.5-MoE (Abdin et al. 2024)。这些模型的配置如 [tab:configuration] 所示。实验在不同的输入 Token 长度和多样的混合并行策略下进行。实验细节和结果如 [fig:e2e] 所示。需要注意的是，当 $\textit{TP}<W$ 时，Megatron-LM 为非 MoE 层启用数据并行以提高整体吞吐量，数据并行大小为 $W/\textit{TP}$。不同机制下使用 Megatron-LM 的 Attention 层计算是相同的，仅 MoE 层的实现方式因不同机制而异。

可以观察到，与 Megatron-Cutlass、Megatron-TE、FasterMoE 和 Tutel 相比，Comet 将各基准测试的端到端延迟分别降低了 $34.1\%$、$42.6\%$、$44.4\%$ 和 $31.8\%$。在分离相同 Attention 计算的情况下，性能提升更为显著。Comet 在所有配置中均优于其他基线，因为它实现了充分的重叠，并且高性能融合内核内部的调度极大地减少了 CPU 侧的开销。

此外，我们还可以观察到 Megatron-Cutlass 和 Megatron-TE 的表现相似。这是因为除了 GEMM/GroupGEMM 的实现不同外，它们在其他方面完全相同。两者都不支持重叠，而 Megatron-TE 在某些情况下表现更差，原因是 Transformer Engine API 调用存在额外开销。Tutel 的表现优于其他基线，因为它通过精心的调度和自适应并行将通信融入专家的计算中。尽管通信和计算实现了部分重叠，但当专家数量较大时（Qwen2），Tutel 的优势因调度开销过大而减弱。FasterMoE 仅支持专家并行（$\textit{EP}=W$），且它在 Qwen2 上表现也不佳，因为 Qwen2 中的专家规模较小，专家的内核调用时间主导了 MoE 层的开销。

## 单个 MoE 层的详细评估

随后，我们对单个 MoE 层进行了深入检查，以进行详细分析。

**处理不同的输入 token 长度。** 具有不同输入 token 长度的单个 MoE 层的延迟如 [fig:single_tp1] 所示。随着输入 token 数量的变化，与基线相比，Comet 经历了更短的持续时间，且改进是稳定的。与基线相比，Comet 平均实现了 $1.28\times$ 到 $2.37\times$ 的加速。值得注意的是，Comet 的优势在 $M$ 较小时尤为突出。这是因为当 $M$ 较小时，主机端的调度时间在整体持续时间中占主导地位，而 Comet 通过在融合 kernel 内进行 kernel 调度减少了此类开销。对于具有 kernel 级调度的机制（FasterMoE 和 Tutel），调度开销随着 $topk$ 和 $E$ 的增加而增加，因为要管理的专家变得更加复杂，导致需要调度更多的 kernel。

具有专家并行（EP = 8）的单个 MoE 层持续时间。x 轴表示总输入 token 长度 M。在 token 分发之前，每个设备拥有 M/W 个 token。专家的形状与 Mixtral 8x7B 相同。

**MoE 层的时间分解分析。** 特定 MoE 层的时间分解如 [fig:breakdown] 所示。注意，通信部分仅包含 GPU 到 GPU 的通信时间，而本地设备上的 token 索引、分发和合并操作被视为计算部分。如图所示，Megatron-TE 和 Megatron-cutlass 在通信和计算之间没有重叠。FasterMoE 通过定制的 Scatter 和 Gather 算子减少了通信延迟，而引入的本地索引延长了计算时间。Tutel 通过优化的 all-to-all 原语设计减少了通信开销。然而，其优化的 all-to-all 也加剧了本地计算的负担。Megatron-TE 没有通信重叠。Comet 平均隐藏了 $86.5\%$ 的通信延迟，并且专家的计算效率不受影响，而 FasterMoE 和 Tutel 仅分别隐藏了 $29.2\%$ 和 $68.6\%$。

具有专家并行的 MoE 层的时间分解。(EP = 8, TP = 1, E = 8, topk = 2 且 M = 16384)。

**MoE 层内的并行。** 由于专家并行的引入，MoE 层内的并行策略可以不同于模型的整体并行策略。[fig:single_para] 展示了应用不同并行策略的方法的性能。在所有基线中，遗憾的是 FasterMoE 不支持张量并行。对于其他基线（Megatron-TE、Megatron-Cutlass 和 Tutel），当 *TP* 增加时，MoE 层延迟也随之增加。这是因为张量并行将每个专家拆分到多个设备上，触发了更多针对专家的碎片化小型 GEMM，导致计算效率下降。然而，Comet 在不同的并行中保持了低延迟，因为共享张量被重新调度以保持计算效率，并且权重切换开销被消除。

在 E = 8, topk = 2, M = 8192, EP × TP = 8 的各种并行策略下的单个 MoE 层持续时间。

## 对不同配置的适应性

我们进一步探究了 Comet 在适应不同模型配置、运行时工作负载和系统环境时的性能。

**具有各种 MoE 参数的性能。** 我们调整了专家数量 $E$ 以及 $topk$，以评估 Comet 在各种 MoE 结构中的性能。结果如 [fig:experts] 所示。随着 $topk$ 的增加，MoE 层的持续时间增加，因为运行时的计算量被放大。Comet 在不同的 $topk$ 和 $E$ 值上始终表现出卓越的性能，与基线实现相比，产生了 $1.16\times$ 到 $1.83\times$ 范围内的加速。

具有不同专家数量 E 和 topk 的单个 MoE 层持续时间 (M = 16384, EP = 8, TP = 1)。

**具有不同 token 分布的性能。** 当使用专家并行时，路由到不同设备的 token 数量会有所不同。我们在 token 分布不平衡的场景中评估了 Comet 的性能。不同专家间 token 分布的标准差记为 $std$。如 [fig:scaling] 左图所示，8192 个 token 以不同的分布方式分发到各个专家。当 $std=0$ 时，token 均匀分布，每个专家接收 $M\times topk / E = 2048$ 个 token。在 $std=0.05$ 时，负载最轻的专家仅被分配几百个 token。在生产环境的典型训练任务中，平均 $std$ 为 $0.032$。当负载不平衡问题加剧时，所有系统中 MoE 层的延迟都会延长。Comet 始终优于其他 MoE 系统。

扩展到不同场景时 MoE 层的性能。左：具有专家并行的各种 token 分布下的持续时间 (E = 8, topk = 2, M = 8192, TP = 1, EP = 8)。右：具有不同并行的 L20 集群上的持续时间 (E = 8, topk = 4, M = 8192, EP × TP = 8)。

**扩展到不同集群。** 我们在另一个具有不同网络环境的不同集群上进行了实验。该集群配备了 8 块 Nvidia L20 GPU（46 GB 内存），GPU 之间通过 PCIe 桥接器连接。经测试，GPU 间的带宽约为 25 GB/s，远低于 H800 集群。在 L20 集群上的实验代表了带宽受限的环境。如 [fig:scaling] 右图所示，与其他基线相比，Comet 的平均加速比为 $1.19\times$ 到 $1.46\times$。结果表明了 Comet 在不同集群环境下的优越性。

| Mem(MB) | Mixtral 8x7B | Qwen2-MoE | Phi3.5-MoE |
|:-------:|:------------:|:---------:|:----------:|
| M=4096  |      32      |    16     |     32     |
| M=8192  |      64      |    32     |     64     |

NVSHMEM 所需的设备内存大小。

## 开销分析

Comet 利用 NVSHMEM 在每个设备上为通信分配一个共享内存缓冲区。缓冲区大小取决于模型配置，等于 $MN$，其中 $M$ 是输入序列长度，$N$ 是模型隐藏层大小。对于 BF16 或 FP16 数据类型，分配的内存大小为 $2MN$。通信缓冲区对于整个模型的执行是全局的，这意味着它在各层和专家之间共享。我们在 [tab:memory] 中列出了 Comet 的设备内存消耗，与当前 GPU 上的大容量设备内存相比，这可以忽略不计。

# 相关工作

随着 MoE 在大规模分布式训练和推理中的成功应用，有大量工作致力于系统级优化，以减少 MoE 结构中继承的通信开销。

#### 通信优化。

为了减少 MoE 执行中的通信开销，一种直接的方法是利用高效的通信算法 (Nvidia 2022a; Shen et al. 2022) 来实现更快的数据传输。最近的工作 (Hwang et al. 2023; Rajbhandari et al. 2022; Nie et al. 2022) 还提出了 2D 分层 all-to-all 算法，以更好地利用节点内带宽并加速 MoE 通信。其他一些工作提出通过数据压缩来减少通信量。例如，ScheMoE (Shi et al. 2024) 和 Zhou 等人， (Q. Zhou et al. 2022) 提出应用数据压缩技术来减少 all-to-all 通信量，同时保持模型收敛。

#### 计算与通信重叠。

针对稠密模型的计算与通信重叠技术已被广泛应用于分布式训练和推理中 (C. Chen et al. 2024; Jangda et al. 2022; Song et al. 2023; S. Wang et al. 2022; Y. Wang et al. 2023; Chang et al. 2024)。对于 MoE 结构，最近的研究也试图寻找 all-to-all 操作的通信任务与 GEMMs 计算任务的流水线机会。FasterMoE (He et al. 2022) 允许流水线深度为 2，以将专家计算和 all-to-all 通信进行流水线化。Tutel (Hwang et al. 2023) 支持手动设置流水线深度，或在有限的搜索空间内进行启发式搜索，这可能不是最优的。PipeMoE (Shi et al. 2023) 和 ScheMoE (Shi et al. 2024) 旨在调度 MoE 算子，以更好地利用内部和互连带宽。这些解决方案通过内核级调度实现重叠，但并未完全解决 MoE 中的细粒度数据依赖问题。

# 结论

在本文中，我们提出了 Comet，这是一个旨在为 MoE 实现细粒度通信与计算重叠的 MoE 系统。Comet 具有两个关键设计，以在不影响计算效率的情况下实现无缝重叠：基于共享张量的依赖解析机制，实现了细粒度重叠，同时消除了由细粒度通信 I/O 引起的瓶颈；工作负载分配机制，确保算子的精确和自适应重叠，从而实现最大程度的延迟隐藏。与现有文献相比，Comet 在单个 MoE 层中实现了 $1.96\times$ 的加速，在 MoE 模型的端到端执行中实现了 $1.71\times$ 的加速。

## References


Abdin, Marah, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, et al. 2024. “Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone.” *arXiv Preprint arXiv:2404.14219*.



Bai, Jinze, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. “Qwen-Vl: A Frontier Large Vision-Language Model with Versatile Abilities.” *arXiv Preprint arXiv:2308.12966*.



Cao, Bing, Yiming Sun, Pengfei Zhu, and Qinghua Hu. 2023. “Multi-Modal Gated Mixture of Local-to-Global Experts for Dynamic Image Fusion.” In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 23555–64.



Chang, Liwen, Wenlei Bao, Qi Hou, Chengquan Jiang, Ningxin Zheng, Yinmin Zhong, Xuanrun Zhang, et al. 2024. “FLUX: Fast Software-Based Communication Overlap on GPUs Through Kernel Fusion.” *arXiv Preprint arXiv:2406.06858*.



Chen, Chang, Xiuhong Li, Qianchao Zhu, Jiangfei Duan, Peng Sun, Xingcheng Zhang, and Chao Yang. 2024. “Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning.” In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, 178–91.



Chen, Tianqi, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, et al. 2018. “$\{$TVM$\}$: An Automated $\{$End-to-End$\}$ Optimizing Compiler for Deep Learning.” In *13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)*, 578–94.



He, Jiaao, Jiezhong Qiu, Aohan Zeng, Zhilin Yang, Jidong Zhai, and Jie Tang. 2021. “Fastmoe: A Fast Mixture-of-Expert Training System.” *arXiv Preprint arXiv:2103.13262*.



He, Jiaao, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li. 2022. “Fastermoe: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models.” In *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, 120–34.



Huggingface. 2022. “Hugging Face.” <https://huggingface.co/>.



Hwang, Changho, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, et al. 2023. “Tutel: Adaptive Mixture-of-Experts at Scale.” *Proceedings of Machine Learning and Systems* 5: 269–87.



Jangda, Abhinav, Jun Huang, Guodong Liu, Amir Hossein Nodehi Sabet, Saeed Maleki, Youshan Miao, Madanlal Musuvathi, Todd Mytkowicz, and Olli Saarikivi. 2022. “Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads.” In *Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems*, 402–16.



Jiang, Albert Q, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, et al. 2024. “Mixtral of Experts.” *arXiv Preprint arXiv:2401.04088*.



Lepikhin, Dmitry, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2020. “Gshard: Scaling Giant Models with Conditional Computation and Automatic Sharding.” *arXiv Preprint arXiv:2006.16668*.



Liu, Haotian, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. “Visual Instruction Tuning.” *Advances in Neural Information Processing Systems* 36.



Liu, Rui, Young Jin Kim, Alexandre Muzio, and Hany Hassan. 2022. “Gating Dropout: Communication-Efficient Regularization for Sparsely Activated Transformers.” In *International Conference on Machine Learning*, 13782–92. PMLR.



Liu, Ze, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. 2021. “Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows.” In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 10012–22.



Nie, Xiaonan, Pinxue Zhao, Xupeng Miao, Tong Zhao, and Bin Cui. 2022. “HetuMoE: An Efficient Trillion-Scale Mixture-of-Expert Distributed Training System.” *arXiv Preprint arXiv:2203.14685*.



Nvidia. 2017. “Unified Memory for CUDA Beginners.” <https://developer.nvidia.com/blog/unified-memory-cuda-beginners/>.



———. 2022a. “Doubling All2all Performance with NVIDIA Collective Communication Library 2.12.” <https://developer.nvidia.com/blog/doubling-all2all-performance-with/nvidia-collective-communication/library-2-12/>.



———. 2022b. “Nvidia Hopper Architecture in-Depth.” <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>.



———. 2024a. “CUTLASS.” <https://github.com/NVIDIA/cutlass>.



———. 2024b. “Grouped GEMM for MoE.” <https://github.com/fanshiqing/grouped_gemm>.



———. 2024c. “NCCL.” <https://developer.nvidia.com/nccl>.



———. 2024d. “NVSHMEM.” <https://developer.nvidia.com/nvshmem>.



———. 2024e. “Transformer Engine.” <https://github.com/NVIDIA/TransformerEngine>.



OpenAI. 2022. “Introducing Triton: Open-Source GPU Programming for Neural Networks.” <https://openai.com/index/triton/>.



Rajbhandari, Samyam, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. 2022. “Deepspeed-Moe: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation Ai Scale.” In *International Conference on Machine Learning*, 18332–46. PMLR.



Sharir, Or, Barak Peleg, and Yoav Shoham. 2020. “The Cost of Training Nlp Models: A Concise Overview.” *arXiv Preprint arXiv:2004.08900*.



Shazeer, Noam, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.” *arXiv Preprint arXiv:1701.06538*.



Shen, Liang, Zhihua Wu, WeiBao Gong, Hongxiang Hao, Yangfan Bai, HuaChao Wu, Xinxuan Wu, et al. 2022. “Se-Moe: A Scalable and Efficient Mixture-of-Experts Distributed Training and Inference System.” *arXiv Preprint arXiv:2205.10034*.



Shi, Shaohuai, Xinglin Pan, Xiaowen Chu, and Bo Li. 2023. “PipeMoE: Accelerating Mixture-of-Experts Through Adaptive Pipelining.” In *IEEE INFOCOM 2023-IEEE Conference on Computer Communications*, 1–10. IEEE.



Shi, Shaohuai, Xinglin Pan, Qiang Wang, Chengjian Liu, Xiaozhe Ren, Zhongzhe Hu, Yu Yang, Bo Li, and Xiaowen Chu. 2024. “ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling.” In *Proceedings of the Nineteenth European Conference on Computer Systems*, 236–49.



Shoeybi, Mohammad, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. “Megatron-Lm: Training Multi-Billion Parameter Language Models Using Model Parallelism.” *arXiv Preprint arXiv:1909.08053*.



Song, Jaeyong, Jinkyu Yim, Jaewon Jung, Hongsun Jang, Hyung-Jin Kim, Youngsok Kim, and Jinho Lee. 2023. “Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression.” In *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, 560–73.



Touvron, Hugo, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, et al. 2023. “Llama 2: Open Foundation and Fine-Tuned Chat Models.” *arXiv Preprint arXiv:2307.09288*.



Vaswani, A. 2017. “Attention Is All You Need.” *Advances in Neural Information Processing Systems*.



Wang, Shibo, Jinliang Wei, Amit Sabne, Andy Davis, Berkin Ilbeyi, Blake Hechtman, Dehao Chen, et al. 2022. “Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models.” In *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1*, 93–106.



Wang, Yuke, Boyuan Feng, Zheng Wang, Tong Geng, Kevin Barker, Ang Li, and Yufei Ding. 2023. “$\{$MGG$\}$: Accelerating Graph Neural Networks with $\{$Fine-Grained$\}$$\{$intra-Kernel$\}$$\{$communication-Computation$\}$ Pipelining on $\{$Multi-GPU$\}$ Platforms.” In *17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23)*, 779–95.



Zhou, Qinghua, Pouya Kousha, Quentin Anthony, Kawthar Shafie Khorassani, Aamir Shafi, Hari Subramoni, and Dhabaleswar K Panda. 2022. “Accelerating Mpi All-to-All Communication with Online Compression on Modern Gpu Clusters.” In *International Conference on High Performance Computing*, 3–25. Springer.



Zhou, Yanqi, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. 2022. “Mixture-of-Experts with Expert Choice Routing.” *Advances in Neural Information Processing Systems* 35: 7103–14.



Zuo, Simiao, Xiaodong Liu, Jian Jiao, Young Jin Kim, Hany Hassan, Ruofei Zhang, Tuo Zhao, and Jianfeng Gao. 2021. “Taming Sparsely Activated Transformer with Stochastic Experts.” *CoRR* abs/2110.04260. <https://arxiv.org/abs/2110.04260>.
