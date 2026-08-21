# Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference 原文翻译

# 突破内存墙：面向长上下文 Agentic LLM 推理的优化路径

Haoran Wu<sup>\*</sup>, Can Xiao<sup>†</sup>, Jiayi Nie<sup>\*</sup>, Xuan Guo<sup>†</sup>, Binglei Lou<sup>†</sup>, Jeffrey T.H. Wong<sup>†</sup>, Zhiwen Mo<sup>†</sup>, Cheng Zhang<sup>†</sup>, Przemyslaw Forys<sup>†</sup>, Chengyang Ai<sup>‡</sup>, Timi Adeniran<sup>\*</sup>, Wayne Luk<sup>†</sup>, Hongxiang Fan<sup>†</sup>, Jianyi Cheng<sup>‡</sup>, Timothy M. Jones<sup>\*</sup>, Rika Antonova<sup>\*</sup>, Robert Mullins<sup>\*</sup>, Aaron Zhao <sup>\*</sup>剑桥大学, <sup>†</sup>帝国理工学院, <sup>‡</sup>爱丁堡大学

摘要——大型语言模型 (LLM) 作为 AI 智能体的核心组件，广泛应用于包括企业工作流自动化、软件工程、网页自动化、计算机使用和科学研究在内的众多领域。这些 Agentic LLM 推理任务与传统以聊天机器人为中心的推理有着根本的不同——它们通常具有更长的上下文长度，以捕获复杂、持久的输入，例如整个网页 DOM 或复杂的工具调用轨迹。这反过来会在推理阶段为硬件产生大量的片外内存流量，并导致工作负载受到两道内存墙的限制，即带宽墙和容量墙，从而阻碍计算单元实现高利用率。

在本文中，我们介绍了 PLENA，这是一个采用三种核心优化路径的软硬件协同设计系统。PLENA 具有一种新颖的扁平化脉动阵列架构（路径 1）以及支持非对称量化方案的高效计算和内存单元（路径 2）。它还提供了对 FlashAttention 的原生支持（路径 3）。此外，PLENA 的开发配备了完整的软硬件栈，包括定制的 ISA、编译器、事务级模拟器以及自动化设计空间探索流程。实验结果表明，在 LLaMA Agentic 推理期间，在相同的乘法器数量和内存配置下，PLENA 的吞吐量分别比 A100 GPU 和 TPU v6e 高出 2.23 倍和 4.70 倍。PLENA 的能效也比 A100 GPU 高出 4.04 倍。

关键词——LLM 加速器，Agentic 推理，脉动阵列，FlashAttention，量化

## I. 引言

Transformer 已经彻底改变了人工智能在语言、视觉和科学等各个领域的发展 [35], [63], [67]。基于 Transformer 的自回归大型语言模型（LLM），如 GPT [47] 和 LLaMA [61]，现已广泛应用于许多场景，例如聊天机器人 [73]、代码生成 [33] 和计算机使用工作流 [46]。

LLM 智能体能力的快速崛起，例如计算机使用 [40]、网络自动化 [28], [45] 和命令行智能体 [2]，严重依赖于其处理和推理超大上下文窗口的能力。例如，命令行智能体必须理解并生成大规模代码库 [31], [51], [71]，而工具和计算机使用智能体工作流必须跟踪跨越长输入的多条信息（例如完整的网页 DOM），通常需要非常长的上下文 [11], [16], [36]。图 1(a) 显示，与聊天机器人工作负载相比，智能体工作负载每次推理消耗的 Token 多达 100 倍。作为回应，现代 LLM 有意扩大了其上下文窗口：最初的 GPT-3 [8] 支持大约 2k 个 Token，而 GPT-4 [47] 达到 32k 个 Token，LLAMA-4-Maverick [4] 甚至达到 1M 个 Token。

![](images/4350dc7220f6521d1422a27bb94852e9ccc07ce4a2cbf5af4083475a56cc2b26.jpg)  
(a) 标准聊天机器人 [73], [76]、编程 [10], [17] 以及智能体工作负载（包括计算机使用智能体 (CUA) [3], [69] 和网络使用智能体 (WUA) [9], [69]）之间的 Token 使用量比较。尽管在智能体任务中 decode 使用的 Token 远少于 prefill，但由于要在整个上下文上进行顺序处理，它仍然对延迟有显著贡献。

![](images/8ebb7251ee890aaf3689268dad7465c8befd9a0189675bf61f4da89dfdf38c0e.jpg)

![](images/fd8b092b7b8cfb7f1b319a0eb0a364c88b59d0dd75ef76c0865ba202fbd9746e.jpg)  
(b) 随着上下文长度的增加，计算强度从 FFN 转移到 attention 块。  
(c) KV cache 随上下文长度增加而扩展，最终主导内存使用。  
图 1：智能体推理工作负载的示意图，展示了它们通常在单次推理运行中生成更多的 Token (a)，包含 FFN 计算密集型和 attention 计算密集型阶段 (b)，并在单次推理运行中包含权重-内存-容量主导和 KV 主导的阶段 (c)。

为了阐明智能体工作负载的计算特征，图 1(b) 分析了长上下文 LLAMA-3-70B 模型。在较小的上下文长度下，推理主要由 Feed-Forward Networks (FFN) 主导，其占据了大部分 FLOPs。随着上下文的增长（主要由智能体任务中的大型 prefill 序列驱动），计算配置文件从 FFN 密集型转变为 attention 密集型，最终 attention 主导了整体的 FLOP 数量。

![](images/dcec76007a09d13435026891701539a23d51f2e891155f770416f1fa74d7a354.jpg)

智能体 LLM 推理还消耗大量的 HBM 资源。图 1(c) 指出了内存方面的两个主要限制因素。首先，必须读取的大量 KV 值和权重，连同写回的那部分 KV 值，施加了巨大的内存带宽需求。其次，随着上下文长度的增加，KV-cache 的需求呈线性增长，迅速增加内存使用量，并且通常超过模型权重的大小，使得 HBM 容量成为主要的限制因素。例如，在 LLAMA-3-70B 中，在 128k 上下文 [44] 下，单个 batch 的 FP16 KV cache 约为 39 GB，这限制了可以在芯片上保留多少 batch [23]。基于这一观察，我们提出在片外内存方面存在两个主要挑战，即 有限内存带宽和 受限内存容量。我们将这些统称为内存墙。它们共同阻碍了设备在推理时达到峰值性能，这与先前工作 [15], [23], [74] 中的观察结果一致。

内存墙现象导致了硬件（例如 TPU 和 GPU）上计算资源的利用率不足。这种效应在通用矩阵-矩阵乘法 (GEMM) 操作 $( \mathbb { R } ^ { M \times K } \times \mathbb { R } ^ { K \times N } $ $\mathbb { R } ^ { M \times \mathbf { \dot { N } } }$ )（记为 $( M , K ) \times ( K , N )$）中尤为明显，该操作构成了 LLM 推理期间的核心计算工作负载 [26]。在微架构层面，大多数硬件都采用正方形脉动阵列或矩阵乘法单元构建，通常设计为 M 和 N 维度的大小接近 K。例如，TPU v3 [24] 具有一个 128×128 的脉动阵列，支持 $M \ = \ K \ = \ N \ = \ 1 2 8$ 的 GEMM 操作。然而，在长上下文智能体模型中，如图 1(c) 所示，内存通常成为推理 batch size 的主要约束。这导致了一个胖 GEMM 操作，其中与 batch 相关的维度（通常是 $( M , K ) \times ( K , N )$ 中的 M）远小于其他操作维度。这本质上产生了一个不均匀的矩阵形状<sup>1</sup>。这种不平衡阻碍了脉动阵列和 Tensor Core 实现高计算资源利用率 [29]。

为此，我们提出了可编程长上下文高效神经加速器 (PLENA)，这是一种高效的 Transformer 模型加速器系统，旨在在所有推理阶段（prefilling 和 decoding）保持 GEMM 单元的高利用率，特别是对于具有大上下文的智能体 LLM 推理任务。PLENA 通过在硬件和软件设计空间中探索三种优化路径，实现了长上下文推理的高效率。

首先，我们新颖的扁平化脉动阵列（路径 1）解决了推理中使用的典型正方形 GEMM 造成的架构不匹配问题，实现了更高的计算利用率，如图 2(a) 和 2(b) 所示。其次，我们应用了带有 Post-Training

![](images/45560d485999d69f738b406b0a90ef975f342cef4248da4eea26b370b2cf4ba0.jpg)  
(a) PLENA 在使用相同资源的情况下，比标准正方形脉动阵列实现了更高的利用率。  
(b) PLENA 的优化路径——(1) 扁平化脉动阵列和 (2) 非对称量化——共同实现了更高的有效内存带宽利用率，并有助于减少内存容量限制。  
图 2：在使用相同数量乘法器<sup>2</sup>的情况下，正方形脉动阵列（例如在 TPU 中）与 PLENA 的可达 FLOPs 比较。

Quantization (PTQ) 优化的非对称量化策略（路径 2），其中权重 (W) / 激活值 (A) / KV Cache (KV) 可以设置为不同的精度，以缓解内存带宽和容量墙。通过更激进的量化，我们在 HBM 中释放了更多空间用于数据扩展（例如支持更大的 batch size）。图 2 展示了与未进行任何优化的传统正方形脉动阵列加速器相比，这些路径如何共同提高利用率。最后，由于图 1(b) 显示 attention 在较长上下文长度下主导计算，我们设计了 PLENA 的定制 ISA 和新颖架构以有效支持 FlashAttention（路径 3）——这是一种 IO 感知、融合的 attention 算法，可大幅减少片外内存流量 [13]。这降低了 attention 操作使内存带宽饱和的可能性，从而减弱了内存墙效应。

这些优化路径共同产生了比传统正方形脉动阵列加速器高得多的利用率。主要贡献如下：

• 我们分析性地刻画了智能体 LLM 推理中的带宽和容量内存墙，并表明现有的脉动阵列加速器在运行智能体工作负载时通常严重未充分利用。

• 我们引入了三条优化路径，共同解决由内存墙导致的利用率不足问题： 一种扁平化脉动阵列架构； 一种非对称量化方案，并深入探讨了微缩放算术与旋转及范数引导的迭代优化等优化技术的兼容性； 对 FlashAttention 的原生支持。这些优化共同构成了一种整体方法，通过集成硬件级和算法级优化来解决带宽和容量限制。

• 我们提出了 PLENA，一个完整的软硬件系统，实现了上述优化。PLENA 集成了： 用于大型 Transformer 推理的自定义指令集（PLENA ISA）； PyTorch 到 PLENA ISA 的编译器； 支持 HBM 的事务式模拟器； 自动化且精度感知的设计空间探索（DSE）

![](images/58747ef0a90b3f1e484e43388f0994fccf4af425750768d606567f8f6cf21dee.jpg)  
图 3：可配置 MX 数据格式设计的示意图，由可调配置参数化。每个元素块共享一个二次幂缩放因子。

流程；以及 完整的 RTL 实现。我们证明 PLENA 支持不同的 SOTA Transformer 模型变体（例如 GQA 和 MLA [42]、Dense 和 MoE [6]）。我们还表明，PLENA 在智能体 LLM 推理中实现了卓越的效率。在 LLaMA 智能体推理期间，在相同的乘法器数量和内存配置下，PLENA 的吞吐量分别比 A100 GPU 和 TPU v6e 高达 2.23 倍和 4.70 倍，并且能效（Token / J）比 A100 高达 4.04 倍。整个 PLENA 系统已公开发布<sup>3</sup>。

## II. 背景与相关工作

## A. 微缩放数据格式

块数据表示的概念被引入，用于使用共享的缩放因子来共同表示一组值 [14]。在此想法基础上，Rouhani 等人 [54] 提出了微缩放（MX）数据格式，作为块数据格式的一种特定变体，其中每个元素块共享一个以 E8M0 二次幂格式编码的公共缩放因子。此后，MX 格式已被开放计算项目 [52] 标准化。最近的扩展探索了多级缩放，其中缩放因子跨粒度分层应用。MicroScopiQ [50] 采用了一种两级缩放方案，具有粗粒度的块级缩放和更细粒度的微块级缩放，而 NVFP4 [1] 采用了类似的层次结构，使用张量级 E8M23 缩放和块级 E4M3 缩放。为了平衡硬件复杂性和软件性能，我们在可配置的 MX 数据格式中采用单级缩放方案，MXFP 和 MXINT 的可调参数分别为 (M, E, S, B) 和 (M, S, B)，如图 3 所示。

## B. PTQ 与微缩放数据格式的协同设计

现有的现成训练后量化（PTQ）方法针对整数数据格式已有充分研究 [7], [19]。然而，我们发现这些方法在 MX 数据格式方面探索较少——在某些情况下，甚至不直接适用。

GPTQ [19] 最初是为整数量化开发的。我们探索了其在我们参数化 MX 数据格式上的适配，并提出了一种能更好地适配 MX 格式的变体方法。详情推迟至第 IV-B 节。基于旋转的 PTQ 方法是缓解激活异常值最有效的技术之一。QuaRot [7] 证明了 Hadamard 变换的应用可以有效抑制此类异常值。然而，我们通过实验经验发现，如果不加谨慎处理，直接应用这些方法可能会导致 MX 数据格式的模型性能显著下降。详情推迟至第 IV-C 节。

## C. FlashAttention

FlashAttention 优化了标准 Attention 层中的内存 I/O [13]。在标准 Attention 层中，计算 QK<sup>⊤</sup> 会产生一个极其庞大的方阵，通常大小为数千乘以数千。由于片上内存无法容纳这一中间结果，它必须被写入片外内存，随后在后续的 softmax 和 P V 步骤中重新加载，这会显著降低性能。FlashAttention 通过对 Attention 计算（GEMM–Softmax–GEMM）进行分块和融合，避免了这种往返操作，从而使所有中间结果都能容纳在片上。

大多数现有的基于脉动阵列的加速器并不原生支持 FlashAttention。SystolicAttention [39] 是最早将 FlashAttention 集成到脉动架构中的工作之一。相比之下，PLENA 采用了更灵活的方法，实现了激进的内存预取重叠，并利用支持混合精度的扁平化脉动阵列和头级分解，以实现更高的计算利用率和效率。

## D. 加速器及其量化支持

最近的 LLM 加速器 [22], [25], [27], [30], [32], [37], [49], [50], [72] 在计算组织、内核专用化和系统集成方面探索了多样的架构权衡。然而，这些设计大多专注于加速特定内核（例如 GEMM 或 Attention），而不是支持完整的 Transformer 推理流水线，通常需要将不支持的操作卸载到外部处理器。这种部分覆盖可能会引入额外的数据移动，并限制在长上下文推理工作负载下的持续利用率。相反，PLENA 的目标是在加速器结构上直接进行完整的 Transformer 推理。

先前的工作也探索了硬件和量化协同设计 [25], [27], [30], [50], [70]。MicroScopiQ [50] 采用 GPTQ 进行两级 MX 量化。ANT 和 MANT [27], [30] 提出了在运行时适应输入分布的混合数据格式。OliVe [25] 通过将异常值与相邻的低幅度权重配对来处理它们。然而，这些工作大多专注于权重和激活量化，而没有共同解决长上下文推理场景下的 KV 缓存量化问题。相比之下，PLENA 是第一个原生支持可调 MX 格式并同时支持硬件友好的 QuaRot [7] 和 GPTQ [19] 的工作，且原生针对长上下文工作负载。

先前的工作，如 SCALE-Sim [56]，支持用于 DNN 推理的扁平化脉动阵列模拟，而 SARA [55] 探索了可重构阵列形状以优化 DNN 工作负载。然而，这些方法并未明确考虑自回归 Transformer 推理的特性。相反，PLENA 采用了工作负载驱动的设计，重塑了脉动组织，以解决内存受限批处理下 FlashAttention 和 FFN 计算之间的不平衡问题，如第 III-B 节所述。

![](images/5f6878f7b16e75497c8fc15ffe1da3f648c6ec80ab70171ef99fff713e53935b.jpg)  
图 4：PLENA 加速器架构概览。执行由解码器的系统流水线控制器控制，该控制器从解码指令中派生控制信号并监控内存依赖关系。例如，当从仍在被向量或矩阵单元更新的 Vector SRAM 行读取数据时，控制器会插入一个停顿以确保正确性。

## III. PLENA 硬件系统

PLENA 的整体配置如图 4 所示。它采用指令级流水线，主要由三个计算单元组成：矩阵单元、向量单元和标量单元。所有单元都高度可配置，支持多种数据类型和精度（表 III），从而能够将不同的量化方法应用于加速器。

PLENA 还包括两个主要的片上 SRAM 块。Vector SRAM 作为计算的暂存器，存储频繁使用的数据（如激活值），这些数据不需要写回 HBM，从而减少了内存访问开销。定制的 Matrix SRAM 专用于加载权重和 KV 张量，并支持以转置或非转置访问模式读取数据，且仅需极少的额外资源开销和访问开销。

## A. 非对称算术数据通路

为了支持非对称量化策略，PLENA 原生支持跨其计算和存储单元的多种数值格式——涵盖不同的数据类型和精度。这种创新的非对称数据处理配置具有以下特性。

 激活值对量化误差比 KV 或权重更敏感，因此以高精度浮点 (FP) 格式存储在片上 Vector SRAM 中。 由于 KV 和权重对精度不那么敏感，可以使用较低精度的 MX 格式（MXFP 或 MXINT）进行更激进的量化，并暂存于 Matrix SRAM 中。 可选的片上旋转步骤可以在量化前抑制异常值以保持精度。

此外，在 attention 期间向 HBM 中的 KV cache 追加新的 K 和 V 向量时，我们选择性地应用基于 Hadamard 的旋转（算法详见第 IV-C 节），在将其量化为 MX 数据类型并存储到 HBM 之前抑制异常值。由于 K 和 V 仅供 attention GEMM 消耗，它们被直接加载到 Matrix SRAM 中，并在使用前应用逆 Hadamard 变换。这些旋转/逆旋转阶段可以按张量选择性地应用；例如，加载到矩阵单元的权重会绕过逆变换。

## B. 扁平脉动阵列

如图 2(b) 所示，长上下文工作负载在 feed-forward (FFN) 计算过程中经常涉及胖 GEMM，其中与 batch 相关的维度（通常是 (M, K) × (K, N) 中的 M）远小于其他维度，导致矩阵形状不均（图 5），而归约维度 K 往往非常长，例如，权重-激活 GEMM 在模型的隐藏层大小上进行归约（例如，LLAMA-3-8B 为 4,096，LLAMA-3-70B 为 8,192）。

此外，在 FlashAttention 阶段，需要逐 head 的胖 GEMM 操作。head 维度通常较小（例如 LLAMA-3-70B 为 128），而 Grouped Query Attention (GQA) 范式要求每个 key head 同时与多个 query head 相乘。这导致在 FlashAttention 中执行逐 head GEMM 时，大规模脉动阵列的利用率很低，因为计算维度变得相对较小。

为了提高这两个计算最密集层的硬件效率，我们提出了一种扁平脉动阵列架构，该架构对两者都能实现高利用率。对于 FFN 层，每个处理单元执行 (BLEN, MLEN) × (MLEN, BLEN) GEMM，产生形状为 (BLEN, BLEN) 的输出。对于 FlashAttention 模块，脉动阵列被划分为多个更小的扁平阵列核心，以支持逐 head GEMM 计算，其中每个核心在 (MLEN/HLEN) 个 head 上并行执行 (BLEN, HLEN)×(HLEN, BLEN) GEMM。

为了保持高利用率，这种扁平脉动阵列专为输出驻留数据流而设计。如图 5 所示，操作数沿着大的归约维度 K 流动，而部分和在 PE 中保持驻留。然后该阵列进行全流水线化，消除连续 GEMM 块之间的空闲气泡。扁平脉动阵列的微架构如图 6 所示。它由一系列小型方形脉动阵列组成，每个阵列由处理元素 网格构成。每个 PE 重复执行乘加操作，并将数据传递给阵列中其下方和右侧的相邻 PE。如第 III-A 节所述，该脉动阵列设计为原生接收 MX 格式的数据。

![](images/ee20e4180c087206c605b0575faec8ab33f2ae18629c149e83003895034cdb72.jpg)  
图 5：权重-激活输出驻留 GEMM 的处理流程。由于内存容量限制了 batch size，M 维度保持较小。在扁平脉动阵列上设置 BLEN = M 可获得高利用率。

![](images/dbc24c800aeaa4db7a40fa192c83d8dbb0be2c664a85685ae77163b16a575890.jpg)  
图 6：在每个周期，扁平脉动阵列获取两个 MLEN 宽的输入：一个来自 Matrix SRAM（顶部），另一个来自 Vector SRAM（左侧）。输入被缓冲并重新排序，然后被划分为 MLEN/BLEN 个子向量，每个子向量的宽度为 BLEN。每个子向量从顶部和左侧方向转发到相应的子阵列。缩放因子和元素分别流式传输到每个子阵列。为了提高资源效率，每个 PE 消耗 MX 格式的输入并以 INT 精度执行累加。累加结果在写回 Vector SRAM 之前被转换为目标激活精度。

然而，仅由子阵列组成的矩阵单元不足以完成 (BLEN, MLEN) × (MLEN, BLEN) GEMM。每个阵列仅累积结果片段的部分和；要产生完整的 (BLEN, BLEN) 输出，需要进行跨阵列归约，将平铺行中 PE 所持有的部分和相加。为了解决这个问题，我们集成了一个结果加法树（见图 6），它可以高效地执行跨阵列求和。该单元通过专用指令 M SUM 调用，因为沿大归约维度计算 GEMM 时只需要一次跨阵列求和。这防止了气泡并提高了计算效率。

![](images/6634593a23e73382f3753155fa97edecc275e8802c1818555b6db463e3110970.jpg)  
图 7：内存系统的数据布局和数据通路。具有不同 MX 精度和数据类型的数据按照统一的 HBM 存储模式进行存储。当数据进入 Vector SRAM 时会执行到 FP16 的转换，该 SRAM 作为向量单元的暂存器；向量单元以高精度 FP16 运行。对于 Matrix SRAM，从 HBM 加载的 MX 格式数据可以直接存储，无需额外转换。

## C. 非对称内存平衡

我们的内存系统具有两个关键特性：1) 支持非对称精度、可变长度内存传输以及到 HBM 的跨步加载/存储；2) 通过与主执行并行操作的内存加载单元来隐藏 HBM 访问的延迟。

为了更有效地利用 HBM 容量，如第 III-A 节所述，所有存储在 HBM 中的数据都保持为 MX 格式。由于将每个数据块与其逐块缩放因子拼接起来很少能产生与 2 的幂次方内存边界对齐的组合大小，我们改为分别存储每个张量的块及其相应的缩放因子，以确保两者都与内存边界正确对齐。这种布局提高了内存效率，同时保持了数据局部性，如图 7 所示。

内存加载单元对于充分利用 HBM 带宽至关重要。硬件预取引擎被集成到 Matrix 和 Vector SRAM 中，能够从 HBM 进行后台获取并将数据流式传输到每个 SRAM 中，同时 PLENA 的其余部分继续执行其他指令。这维持了计算单元的充分利用，并避免了因 HBM 延迟而导致的停顿。

## D. PLENA ISA

我们定制的 ISA（表 I）旨在涵盖 Transformer 推理所需的所有操作。每条指令（32 位）在结构上平衡了效率与灵活性，能够支持多种基于 Transformer 的模型和计算优化。除了 FlashAttention 之外，该 ISA 还支持各种 Transformer 变体，包括 MHA、MLA [42] 和 MoE [6]。

表 I：PLENA ISA 概览。
<table><tr><td>类型</td><td>描述</td><td>指令数</td></tr><tr><td>矩阵(M)</td><td>控制 GEMM 和 GEMV 操作，支持矩阵转置或不转置</td><td>12</td></tr><tr><td>向量(V)</td><td>执行逐元素和归约操作，以及用于量化的旋转</td><td>12</td></tr><tr><td>标量(S)</td><td>执行标量 INT 和 FP 运算</td><td>17</td></tr><tr><td>HBM(H)</td><td>处理 HBM 与矩阵/向量 SRAM 之间的数据传输</td><td>3</td></tr><tr><td>控制(C)</td><td>定义操作设置，如 HBM 地址、嵌套循环配置和其他执行参数</td><td>8</td></tr></table>

![](images/0a1e89cbb8843ff9d2ac243ec093ac3cc7bd086c50ac4f21c4d15772b0964846.jpg)  
图 8：单批次单头 Attention 算法如何映射到 PLENA 定制 ISA 的示例。指令前缀表示单元类型（例如，M 表示矩阵指令）。

为了实现效率与灵活性的平衡，该 ISA 旨在最小化开销的同时最大化计算和内存资源的利用率。这是通过诸如 tile 级调度等特性实现的，该特性能够在 tile 粒度上对计算和内存指令进行细粒度控制。

## E. 矩阵 SRAM

矩阵 SRAM 旨在支持转置和非转置访问，且不引入额外的延迟或数据移动开销。该设计专门针对在低硬件开销下优化 FlashAttention 中的转置矩阵乘法 $( Q K ^ { \top } )$（见图 8）。

在自回归推理中，在 (QK<sup>⊤</sup>) 计算期间显式转置大型 tile 会引入显著的面积、能耗和延迟开销。将 $( K ^ { \top } )$ 直接存储在 HBM 中也是不切实际的，因为在解码期间必须将新生成的 K 向量附加到 KV cache 中。因此，必须即时执行转置，这促使我们采用一种支持高效行和列访问而无需显式数据重排的 SRAM 组织方式。

如图 9 所示，矩阵 SRAM 将每个逻辑行分布到多个子 SRAM bank 中，将同一行的元素存储在不同 bank 的不同地址处。这种布局确保行和列访问映射到不同的 bank，允许转置和非转置访问并行进行而不会产生 bank 冲突，从而保留了带宽并避免了显式数据移动。

![](images/9716793725b3dd382ce08b4f94f316afb23addc4a899f8c4049c1f4b313f3953.jpg)  
图 9：可转置矩阵 SRAM 设计确保对于非转置和转置访问，每个子 SRAM 每周期最多被访问一个元素。因此，不会引入额外的访问周期。

我们将所提出的设计与具有相同读取宽度和内存深度为 32 的传统转置缓冲加 SRAM 基线进行了对比评估。使用第 V-A 节中的综合工具，我们的设计在保持读取吞吐量的同时，将面积减少了 65.17%。对于转置读取，它仅需两个周期——一个用于子 SRAM 访问，另一个用于数据重排——而基线必须读取 32 行才能产生转置输出。

## F. 支持 FlashAttention

大多数现有的基于脉动阵列的加速器原生不支持 FlashAttention，原因在于以下四个关键要素：

1) 它们不支持片外内存预取与计算的 tile 级重叠，导致额外的延迟开销，因为执行必须等待数据从片外内存加载。

2) 它们缺乏内存布局支持，例如读取时转置和高效的跨步/分块流式传输。

3) 它们仅暴露 GEMM 原语，缺乏在线 softmax 所需的内联逐行归约和非线性操作（max/sum、exp、div）。

4) 它们的 ISA 强制执行固定调度和粗粒度内核边界，这限制了细粒度的逐 tile 执行并阻碍了融合计算模式。

在 PLENA 中，我们通过提出的矩阵 SRAM（见第 III-E 节）解决了问题 (1) 和 (2)，该 SRAM 实现了对内存预取的指令级控制，并以低开销支持读取时转置。挑战 (3) 由实现归约和逐元素操作的向量与标量单元解决。向量宽度（VLEN）是可配置的，以匹配 FlashAttention 使用的 tile 维度。计算精度也是可配置的，但通常设置为更高的精度（例如 FP12），以在 softmax 计算期间保持数值精度。对于问题 (4)，我们定制的 ISA 提供了可组合的细粒度控制，实现了融合 Attention 流水线的持久化逐 tile 调度。这使得 FlashAttention 的每个阶段都可以在 tile 粒度上单独编排。这些机制共同使 PLENA 能够原生且高效地支持 FlashAttention。

## G. PLENA 编译与仿真栈

PLENA 提供了一个全面的设计与评估框架，能够快速适应新模型或新硬件加速器并对其进行优化（图 10）。

由于 Transformer 计算高度重复且结构统一，PLENA 编译器被刻意保持轻量级：它从模型配置文件中解析配置元数据，并将其映射到预定义的 PLENA 定制 ISA 汇编模板上。

![](images/8072276b8d6cadfe082fdf94940d21b7a6d326b7ab287683d3e62d522bc0d6e4.jpg)  
图 10：协同设计框架由具有不同保真度的分层结构（实际硬件、事务级仿真器和分析型仿真器）组成。事务级仿真器提供了良好的保真度（周期精确），同时与 RTL 仿真相比实现了超过 200 倍的加速。

为了评估架构权衡，我们用 Rust 开发了一个事务级（周期近似）仿真器，以事件驱动的方式执行生成的机器代码。该仿真器在周期粒度上对计算执行、指令调度和内存事务进行建模。它与 Ramulator [41] 和 DRAMSys [59] 集成，以提供详细的片外内存时序和带宽建模，包括 bank 级行为。这使得能够对内存与计算的交互进行定量分析，这对于分析长上下文 LLM 推理中的内存墙至关重要。

该仿真器支持完整的 PLENA 架构设计空间，包括非对称混合精度算术（第 III-A 节）。通过连接分析建模和 RTL 仿真，它能够对架构机制（如扁平化脉动映射和片上 FlashAttention）进行准确评估，同时保持比 RTL 仿真快得多的速度。我们针对我们的完整 RTL 实现验证了该仿真器：它在执行延迟和数值精度上都与 RTL 综合结果紧密匹配，同时提供了大约 200 倍的加速，如表 II 所示。

## IV. 量化

我们的工作与先前使用微缩放数据格式 [50], [53] 的研究密切相关。尽管如此，我们在工作中强调，虽然现有的 SoTA PTQ 优化（如旋转 [7] 和范数引导优化 [19]）对整数量化有益，但它们与微缩放格式并不十分契合。我们指出了将 PTQ 优化技术应用于微缩放算术时的这些注意事项：

表 II：针对 LLAMA-3-70B 模型的单个 Transformer block，在不同仿真级别下五次试验的平均误差率，并与 RTL 和综合结果进行对比。
<table><tr><td>评估器 / 模型</td><td>延迟</td><td>面积</td><td>功耗</td><td>执行时间</td></tr><tr><td>解析仿真器</td><td>11.32%</td><td>4.79%</td><td>23.81%</td><td>8ms</td></tr><tr><td>事务模拟器</td><td>4.17%</td><td>不支持</td><td>不支持</td><td>4.3mins</td></tr><tr><td>RTL 仿真 / 综合</td><td>参考</td><td>参考</td><td>参考</td><td>14hrs</td></tr></table>

1) 对于权重量化，MXFP 通常与这些 PTQ 优化不兼容。MXINT 表现出兼容性，但朴素地应用它会导致性能下降。我们引入了一种新颖的块级截断优化，该优化与基于块的算术（如 MXINT）自然互补（第 IV-B 节）。

2) 对于激活量化，诸如 QuaRot 等旋转方案在朴素应用时，会导致 MXINT 和 MXFP 的性能下降。只有当它们被选择性地应用于激活时，才能实现性能提升（第 IV-C 节）。

总而言之，我们指出带有 PTQ 优化的 MXINT 是权重量化的事实上的方法。同时，激活量化可以利用 MXINT 或 MXFP，但旋转应仅选择性地应用。本节的其余部分详细阐述了这些优化策略以及不兼容的根本原因，并在第 IV-D 节详细说明了如何将这些量化集成到 PLENA 系统中，以促进软硬件协同设计。

## A. 预备知识

我们首先使用三个元素：MX 数据格式（τ）、缩放因子（s）和零点（z），来形式化单级缩放方案下的 MX 量化。MX 数据格式由元组 $\tau = ( d , b , B )$ 定义，其中 d 表示数据类型，b 是其位宽，B 是微缩放块大小。例如，τ = (INT, 4, 16) 对应于块大小为 $B ~ = ~ 1 6$ 的 MXINT4 格式，而 $\tau =$ (minifloat, 4, 16) 对应于具有相同块大小的 MXFP4 格式。在这两种情况下，一个块内的所有值共享一个单一的块级缩放因子 s 和零点 z。

对于任何数据格式 τ，可表示的值的集合被限制在一个有限区间内，我们将其表示为

$$
\Omega ( \tau ) = \{ x \in \mathbb { R } \mid \operatorname* { m i n } _ { ( d , b ) } \leq x \leq \operatorname* { m a x } _ { ( d , b ) } \} .\tag{1}
$$

整数 MX 格式（即 d = INT）的可表示范围 $[ \operatorname* { m i n } _ { \tau } , \operatorname* { m a x } _ { \tau } ]$ 由下式给出：

$$
\operatorname* { m i n } _ { \tau } = - ( 2 ^ { b - 1 } - 1 ) , \qquad \operatorname* { m a x } _ { \tau } = \ 2 ^ { b - 1 } - 1 .\tag{2}
$$

我们将高精度张量 W 划分为大小为 B 的块 $w \in \mathbb { R } ^ { B }$。对于每个块 $w ,$ 缩放因子为

$$
s = \frac { \operatorname* { m a x } | w | } { \operatorname* { m a x } _ { \tau } } .\tag{3}
$$

零点 z 移动范围以进行对齐；我们在全文采用对称量化 $( z = 0 )$ 并在后续表达式中省略它。然后，量化将 w 映射到目标格式 w，如下所示：

$$
w _ { \tau } = \mathrm { c l i p } \left( \mathrm { R T N } \left( \frac { w } { s } \right) , \ : \mathrm { m i n } , \ : \mathrm { m a x } \right) ,\tag{4}
$$

其中 RTN(·) 表示最近舍入投影。相应的反量化算子重构原始块的近似值

$$
Q ( w ; s , \tau ) = s \cdot w _ { \tau } .\tag{5}
$$

## B. 针对权重量化的微缩放裁剪优化

现有的微缩放算术实现采用静态裁剪策略，通常对每个块使用固定值作为裁剪阈值（见公式 (3)）。然而，采用更小块的一个优势是能够对数值进行更细粒度的控制。因此，我们引入了微缩放逐块裁剪，这是一种在裁剪溢出误差和内点下溢误差之间提供有意识平衡的技术。

对于以格式 $\tau ,$ 表示的同一切片块 w，其可表示范围为 $[ \operatorname* { m i n } _ { \tau } , \operatorname* { m a x } _ { \tau } ]$，经验范围为 $[ \operatorname* { m i n } _ { w } , \operatorname* { m a x } _ { w } ]$，我们引入一个裁剪参数 $p \in \mathcal { P } \subset$ $[ 0 . 5 , 0 . 9 9 ]$ 。该参数将有效范围缩小至 $[ p \operatorname* { m i n } _ { w } , p \operatorname* { m a x } _ { w } ]$

通过遍历离散集合 ${ \mathcal P } ,$ ，我们可以获得给定块的最优裁剪 $p ^ { \star }$

$$
p ^ { \star } = \arg \operatorname* { m i n } _ { p \in \mathcal { P } } \big \| w - \mathbf { Q } ( w ; p , \tau ) \big \| _ { 2 } ^ { 2 } .\tag{6}
$$

此处 ∥ · ∥<sup>2</sup> 表示欧几里得范数的平方。

对经验范围进行裁剪引入了裁剪误差与下溢误差之间的权衡。这个问题对于基于微缩放的算术尤为关键，因为与张量维度相比，块大小相对较小。选择最优的裁剪范围可以显著影响性能；在我们的实验中，在仅 4 位权重量化设置下，优化后的裁剪在 LLAMA-3-8B 上将困惑度改善了 5.5%。

在此，我们将裁剪优化直接整合到 GPTQ 的迭代误差传播流程中，并引入了一种新的输出范数引导的逐块裁剪搜索，该搜索最小化的是输出块的量化误差，而不是权重块的量化误差。形式上，设 $\mathbf { X } \in \mathsf { \tilde { \mathbb { R } } } ^ { M \times K }$ 为输入，$\mathbf { W } \in \mathbb { R } ^ { N \times K }$ 为权重。给定线性层 $\mathbf { Y } = \mathbf { X } \mathbf { W } ^ { \top }$，我们沿 K 维度以块大小 B（例如 MX 数据格式 τ 中的 MLEN）对权重进行切片，得到待量化的块切片 $\mathbf { \bar { W } } _ { b } \in \mathbb { R } ^ { N \times B }$，类似地，我们可以得到沿 K 维度的激活 $\mathbf { X } _ { b } \in \mathbb { R } ^ { \tilde { M } \times B }$ 。设 P 表示允许的裁剪百分位集合，并设 $\mathbf { Q } ( \cdot ; P , \tau )$ 表示在数据格式 τ 中的逐行量化，其中 $P = ( p _ { 1 } , \ldots , p _ { N } ) \bar { \in } \mathcal { P } ^ { N }$ 是逐行裁剪百分位的集合，我们的优化采用带有 Hessian 信息 $\mathbf { H } _ { F }$ 的外循环优化，以迭代校准权重值 $( W _ { b } + = \delta _ { F }$ ，改编自 GPTQ)

$$
\begin{array} { r } { \delta _ { F } = - \Big ( \mathbf { W } _ { b } - \mathbf { Q } \big ( \mathbf { W } _ { b } ; P _ { b } ^ { \star } , \tau \big ) \Big ) \Big ( [ \mathbf { H } _ { F } ^ { - 1 } ] _ { b b } \Big ) ^ { - 1 } \big ( \mathbf { H } _ { F } ^ { - 1 } \big ) _ { : , b } , } \end{array}\tag{7}
$$

其中 $\mathbf { H } _ { F } = 2 \mathbf { X } _ { F } \mathbf { X } _ { F } ^ { \top }$ 。这与一种新颖的内循环优化相结合，该优化由输出范数引导

$$
P _ { b } ^ { \star } = \arg \operatorname* { m i n } _ { P _ { b } \in \mathcal { P } ^ { N } } \left\| \mathbf { X } _ { b } \Big ( \mathbf { W } _ { b } - \mathbf { Q } ( \mathbf { W } _ { b } ; P _ { b } , \tau ) \Big ) ^ { \top } \right\| _ { 2 } ^ { 2 } ,\tag{8}
$$

C. 用于激活和 KV 量化的选择性旋转微缩放数据格式

基于旋转的优化，例如 QuaRot [7]，试图通过引入旋转矩阵来平滑数值异常值，其中 X、W、H 分别代表激活、权重和 Hadamard 矩阵

$$
l _ { r o t } ( \mathbf { X } ) = \mathbf { Q } ( \mathbf { X H } ) \cdot \mathbf { Q } ( \mathbf { H } ^ { - 1 } \mathbf { W } ) .\tag{9}
$$

令人惊讶的是，我们注意到将旋转应用于更细粒度的权重量化（例如具有小块大小的 MXINT）实际上增加了困惑度。直观上，与激活相比，权重具有更小的动态范围。旋转可能是不必要的，因为大多数权重异常值已经被共享指数捕获。

然后，我们提出一种用于激活量化的选择性旋转策略

$$
\begin{array} { r } { S = \arg \underset { s \in \mathcal { M } } { \operatorname* { m i n } } \sum _ { s \in \mathcal { M } } \Delta _ { p p l } ( l _ { r o t } ^ { * } ) , } \\ { l _ { r o t } ^ { * } ( \mathbf { X } ) = \ Q ( \mathbf { X } \mathbf { H } ) \cdot \mathbf { H } ^ { - 1 } \cdot \mathbf { Q } ( \mathbf { W } ) . } \end{array}\tag{10}
$$

现在 S 是一个由 $\mathcal { M } ,$ 中的层组成的集合，而 $\Delta _ { p p l } ( l _ { r o t } ^ { * } )$ 反映了每层 l 由于旋转带来的性能提升。目标是最小化 M 中所有层的性能损失之和，以选择要包含在 S 中的子集。另一个关键区别是，当这种旋转应用于激活时，我们必须在运行时执行与 ${ \bf H } ^ { - 1 }$ 的乘法，而 PLENA 为此操作提供了原生硬件支持。

## D. 非对称量化与硬件协同设计

如前所述，MXINT 是事实上的权重量化方式，而我们现在在第 IV-C 节中暴露了一个使用 MXINT 或 MXFP 的搜索空间。此外，我们还必须考虑各种精度设置和硬件设计参数（例如 tile 大小、加载/写入大小）。我们建立了一个协同设计框架，在 PLENA 的多保真度模拟器的支持下进行此类探索，如图 10 所示。我们的协同设计可以在不同的保真度下运行，如图 10 所示，但除非另有说明，否则我们选择在事务级别运行，以兼顾合理的速度和良好的保真度。表 III 显示了搜索空间及其相关约束。我们的搜索空间考虑了 A/KV 的各种算术类型，包括 MXINT 和 MXFP，以及不同的精度配置。搜索完成后，结果可以提供一个非对称量化的 PLENA 加速器设计。

为了自动化寻找最优的硬件设计和量化参数，我们建议采用主动学习进行设计空间探索（DSE）。我们还提供了研究优化不同目标之间权衡的能力。为此，我们在 BOTorch 中采用了多目标贝叶斯优化（BO），允许以主动的方式探索 Pareto 前沿。在我们的案例中，目标函数有三个组成部分：精度、延迟和芯片面积：$\textbf { f } = \ \big [ f _ { \mathrm { a c c u r a c y } } ( \cdot ) , f _ { \mathrm { l a t e n c y } } ( \cdot ) , f _ { \mathrm { a r e a } } ( \cdot ) \big ]$ 。该探索方法还通过应用拒绝采样来丢弃无效或不可行的候选者，从而考虑了约束条件。这避免了不必要的、昂贵的目标评估，并加速了搜索的收敛。我们首先在 LLAMA3.2-1B 上进行实验以实现快速迭代，然后将评估扩展到 LLAMA-3-8B。结果在第 IV-D 节中描述。

表 III：选定的硬件和量化参数协同设计搜索空间。示例约束包括：(1) 内存带宽约束 MLEN · KV\_WIDTH ≤ MemBandwidth；(2) MLEN mod BLEN = 0；(3) MLEN ≥ HLEN ≥ BLEN。
<table><tr><td>参数</td><td>描述</td><td>搜索范围</td></tr><tr><td>BLEN</td><td>块单元的 Tile 大小</td><td>[2, 4, ..., 64]</td></tr><tr><td>MLEN</td><td>矩阵单元的 Tile 大小</td><td>[2, 4, ..., 1024]</td></tr><tr><td>VLEN</td><td>向量单元的 Tile 大小</td><td>[2, 4, .., 1024]</td></tr><tr><td>M_LOAD</td><td>从 HBM 加载的矩阵 SRAM 数量（每条指令加载的矩阵数量） 从 HBM 加载的向量 SRAM 数量</td><td>[2, 4, ..., 256]</td></tr><tr><td>V_LOAD</td><td>（每次迭代加载的向量数量）</td><td>[2, 4, ..., 256]</td></tr><tr><td>V_WRITE</td><td>写入 HBM 的向量 SRAM 数量（每次迭代写入的向量数量）</td><td>[2, 4, …., 256]</td></tr><tr><td>ACT_WIDTH</td><td>激活精度</td><td>MXINT†, MXFP†</td></tr><tr><td>KV_WIDTH</td><td>Key/Value 精度</td><td>MXINT†, MXFP†</td></tr><tr><td></td><td>FP_SETTING 浮点精度</td><td>FP†</td></tr></table>

## V. 评估

## A. 实验设置

a) 模型和数据集：我们在流行的开源 LLM 上评估我们的量化框架，即 LLaMA-2 [62] 和 LLaMA-3 [44]，以及 MoE [6]（例如 GPT-OSS）和 Qwen3 模型 [60]。量化性能以 WikiText-2 [43] 上的困惑度、通过 lmevaluation-harness [20] 在六个下游任务上的零样本准确率，以及长上下文和智能体工作负载来衡量：代码生成（HumanEval [10]）、数学推理（GSM8K-Platinum [64]）和工具使用（BFCL-Web Search Base [48]）。所有量化实验均在 NVIDIA B200 GPU 180GB 上运行，使用 PyTorch 2.11.0、CUDA 12.8、Transformers 5.5.0 和 lm-evaluation-harness 0.4.11。

硬件实验使用来自标准和智能体基准测试的 token 使用轨迹进行，并使用 LLAMA-3.3-70B 进行评估，如表 IV 所示。

b) 量化基线：我们与几种最先进（SoTA）的量化方法进行比较，包括针对 GPU 的基于软件的方法，如 GPTQ [19]、OmniQuant [58] 和 QuaRoT [7]，以及用于硬件加速器的方法，如 Atom [75] 和 MicroscopiQ [50]。

表 IV：跨基准测试的 Token 使用量（预填充/输出）：GSM8K [73]、BFCL-Web Search Base (BFCL-W) [48]、OS-World LibreOffice (OSWorld-L) [69]。
<table><tr><td></td><td>GSM8K</td><td>BFCL-W</td><td>OSWorld-L</td></tr><tr><td>预填充</td><td>1.4k</td><td>114k</td><td>90k</td></tr><tr><td>输出</td><td>0.2k</td><td>5k</td><td>8k</td></tr></table>

c) 加速器实现：PLENA 采用 SystemVerilog RTL 实现。我们使用 Synopsys Design Compiler 和 7 nm OpenROAD 预测性 PDK [12] 进行综合。这有助于我们在 1 GHz 时钟频率下生成面积和功耗估算。

d) 加速器基线：由于我们的基线 MicroscopiQ [50]、FIGNA [32]、SystolicAttention [39] 和 Olive [25] 要么没有完全开源，要么无法在一致的技术节点和工具链下进行评估，我们重新实现了它们的核心组件，并将它们集成到 PLENA 系统中，以进行公平的推理性能比较。此外，DeepScale [57] 用于整体系统性能估算，将所有设计缩放到 7 nm 工艺。核心单元的详细面积和功耗使用我们自己的实现进行评估。

e) 推理过程：我们不仅与先前的加速器设计进行比较，还将 PLENA 与高性能商业计算平台进行评估，包括 GPU（A100 80GB 和 H100 80GB）和 TPU（v6e-8），以提供公平且实用的比较。GPU 实验在 Ubuntu 22.04、CUDA 12.8、Python 3.11、PyTorch 2.8.0 和 vLLM 0.10 V1 环境中进行。TPU 实验在 v2-alpha-tpuv6e 软件环境中进行。

## B. 通过协同设计平衡面积、延迟和困惑度

本节展示了我们的设计空间探索实验结果。图 11 展示了使用 LLAMA3.2-1B 和 LLAMA-3-8B 进行优化时所得 Pareto 前沿的经验达成曲面 (Empirical Attainment Surfaces, EAS)。EAS 是一种可视化方法，非常适合用于传达不同随机种子下多次运行所得 Pareto 前沿的不确定性 [18], [34]。现有工具支持对两个目标进行可视化分析 [66]，因此我们首先绘制了准确率和延迟的 EAS。图 11 表明，使用 BoTorch 采样器的主动学习在延迟和困惑度之间取得了比朴素随机采样显著更好的权衡。在使用 LLAMA3.2-1B 进行优化时，Tree-Structured Parzen Estimator (TPE) 相比使用 BoTorch 采样器展现出更温和的收益，因此我们在 LLAMA-3-8B 的实验中重点关注后者。

在表 V 中，我们展示了由多目标优化运行生成的协同设计结果。这些运行可以产生沿 Pareto 前沿具有不同权衡的设计，其中一些自然地结合了多精度和多算术元素。PLENA 系统促进了这种探索，这得益于其对这些算术类型和精度级别的全面仿真和 RTL 支持。

表 V：LLAMA-3-8B 上 BoTorch 运行配置的多目标搜索结果。我们展示了 Pareto 前沿上四个具有不同困惑度 (↓)、延迟（秒 ↓）、面积 $( \mu \mathrm { { m } ^ { 2 } }$ ↓) 权衡的代表性设计点。多目标搜索的完整经验达成曲面见图 11。最佳结果已高亮。
<table><tr><td colspan="10">参数</td><td colspan="3">指标</td></tr><tr><td>BLEN</td><td>MLEN</td><td>VLEN</td><td> $\mathrm { M \_ L O A D }$ </td><td> $\mathtt { V \_ I O A D }$ </td><td> $\mathrm { v \_ w R I T E }$ </td><td>ACT_WIDTH</td><td>KV_WIDTH</td><td>FP_SETTING</td><td>困惑度 ↓</td><td>延迟 (s) ↓</td><td> $\mathbf { A r e a } \ ( \mathbf { m m ^ { 2 } } ) \downarrow$ </td></tr><tr><td>32</td><td>512</td><td>128</td><td>128</td><td>64</td><td>256</td><td>MXFP_E4M3</td><td>MXFP_E3M4</td><td>FP_E4M7</td><td>6.70</td><td>0.137</td><td>137.6</td></tr><tr><td>32</td><td>1024</td><td>1024</td><td>256</td><td>256</td><td>128</td><td>MXINT_8</td><td>MXINT_4</td><td>FP_E3M2</td><td>6.76</td><td>0.116</td><td>203.4</td></tr><tr><td>8</td><td>128</td><td>32</td><td>128</td><td>8</td><td>256</td><td>MXFP_E3M4</td><td>MXFP_E3M4</td><td>FP_E5M6</td><td>6.54</td><td>0.166</td><td>26.45</td></tr><tr><td>16</td><td>128</td><td>16</td><td>4</td><td>16</td><td>64</td><td>MXINT_8</td><td>MXFP_E4M3</td><td>FP_E3M2</td><td>6.60</td><td>0.174</td><td>23.64</td></tr></table>

![](images/7ce55307e20f9c72d1468aa988f884af1a983f4f18e7317e64acaf12bb9d6a5b.jpg)

![](images/ed03d6292ca87b38e77e0c732f462540b049ffe3ff322a17b7d374c57c593043.jpg)

![](images/76f66dc54e25f78cfff4f944a631906256c927c3a65fc37d90102fb3f5fa572c.jpg)  
图 11：在表 III 所示的协同设计空间中，使用 LLAMA3.2-1B 和 LLAMA-3-8B 评估的跨多个种子的延迟 (↓) 和困惑度 (↓) 目标的经验达成曲面。对于 1B 模型，我们运行了 9 个种子和 50 次试验，将 BoTorch 和 TPE 方法与随机采样进行了比较。对于 8B 模型，我们运行了 5 个种子和 50 次试验，将 BoTorch 与随机采样进行了比较。阴影区域显示了跨种子的 25% 和 75% 达成带。

## C. 通过裁剪与旋转保持 MX 困惑度

a) 主要结果：我们将我们的量化方法与相关工作进行了评估对比；结果总结在表 VI 中。为了公平比较，我们首先通过仅量化解码器中的线性 GEMM 来匹配先前的设置。我们的方法在所有三种量化精度设置下（W4A16KV16、W4A4KV16 和 W4A4KV4）均匹配并优于所有相关工作。我们还在表 VIII 中报告了下游零样本结果，在 W4A4KV4 设置下，我们的方法在所有任务上均优于 QuaRoT [7]。我们还评估了使用量化向量核心的结果。我们发现将剩余算子量化为 MiniFloat E6M5 格式在困惑度上实际上是无效的，同时相对于 FP16 减少了 25% 的内存占用。这一性能提升的关键贡献来自两个方面：1) 输出范数引导的逐块裁剪搜索：通过将输出范数引导的逐块裁剪集成到迭代权重量化中，我们验证了输出重建误差与最终任务性能强相关；因此，我们的方法显著减少了困惑度退化。2) 选择性旋转：我们的方法为每个模型搜索最佳的逐层旋转组合。与 QuaRoT [7] 将旋转合并到权重中不同，我们仅对特定层应用在线旋转。

b) 消融研究：为了进一步评估我们关键贡献的影响，我们在 LLAMA-3-8B 上进行了消融研究，以验证 1) 输出范数引导的逐块裁剪搜索和 2) 选择性旋转的有效性。消融研究分为三个阶段： 仅有权重量化， 在量化权重基础上的激活和 KV-cache 量化，以及 量化所有 MX 感知算子的全系统仿真。我们在表 VII 中展示了它们。

表 VI：LLAMA 在仅 GEMM 仿真（非线性算子为全精度）下的 WikiText-2 困惑度 (↓)。W/A/KV 表示权重、激活和 KV cache 的位宽。标记为 <sup>∗</sup> 的结果由发布的代码复现。
<table><tr><td></td><td></td><td colspan="3">LLaMA-2 [62]</td><td colspan="2">LLaMA-3 [44]</td></tr><tr><td>方法</td><td>W/A/KV</td><td>7B</td><td>13B</td><td>70B</td><td>8B</td><td>70B</td></tr><tr><td>基线</td><td>16/16/16</td><td>5.47</td><td>4.83</td><td>3.31</td><td>6.13</td><td>2.85</td></tr><tr><td>GPTQ [19]</td><td>4/16/16</td><td>6.23</td><td>5.58</td><td>4.28</td><td>8.12</td><td>3.75</td></tr><tr><td>AWQ [38]</td><td>4/16/16</td><td>5.82</td><td>5.19</td><td>4.08</td><td>7.96</td><td>3.58</td></tr><tr><td>OmniQuant [58]</td><td>4/16/16</td><td>5.74</td><td>5.02</td><td>3.47</td><td>7.09</td><td>3.46</td></tr><tr><td>MicroScopiQ [50]</td><td>4/16/16</td><td>5.65</td><td>5.02</td><td>3.42</td><td>6.89</td><td>3.25</td></tr><tr><td>QuaRot [7]</td><td>4/16/16</td><td>5.60</td><td>5.00</td><td>3.41</td><td>6.52*</td><td>3.53*</td></tr><tr><td>PLENA (MXFP)</td><td>4/16/16</td><td>7.09</td><td>5.91</td><td></td><td>11.95</td><td></td></tr><tr><td>PLENA (本文)</td><td>4/16/16</td><td>5.59</td><td>4.98</td><td>3.40</td><td>6.45</td><td>3.25</td></tr><tr><td>OmniQuant [58]</td><td>4/4/16</td><td>11.47</td><td>8.32</td><td>5.41</td><td>10.21</td><td>5.30</td></tr><tr><td>SmoothQuant [68]</td><td>4/4/16</td><td>20.47</td><td>15.63</td><td>17.62</td><td>29.54</td><td>19.32</td></tr><tr><td>Atom [75]</td><td>4/4/16</td><td>6.16</td><td>6.12</td><td>5.20</td><td>8.12</td><td>4.69</td></tr><tr><td>MicroScopiQ [50]</td><td>4/4/16</td><td>6.11</td><td>5.57</td><td>4.48</td><td>8.12</td><td>4.65</td></tr><tr><td>QuaRot [7]</td><td>4/4/16</td><td>6.02*</td><td>5.36*</td><td>3.78</td><td>8.00*</td><td>6.33*</td></tr><tr><td>M-ANT [30]</td><td>4/4/16</td><td>5.92</td><td>5.24</td><td></td><td></td><td></td></tr><tr><td>PLENA (MXFP)</td><td>4/4/16</td><td>15.89</td><td>10.30</td><td></td><td>91.71</td><td></td></tr><tr><td>PLENA (本文)</td><td>4/4/16</td><td>5.82</td><td>5.14</td><td>3.56</td><td>6.99</td><td>3.87</td></tr><tr><td>QuaRot [7]</td><td>4/4/4</td><td>6.10</td><td>5.40</td><td>3.79</td><td>8.16</td><td>6.66</td></tr><tr><td>QuaRot-128G [7]</td><td>4/4/4</td><td>5.93</td><td>5.26</td><td>3.61</td><td>7.36</td><td>5.51</td></tr><tr><td>PLENA (MXFP)</td><td>4/4/4</td><td>67.35</td><td>27.44</td><td></td><td>256.22</td><td></td></tr><tr><td>PLENA (本文)</td><td>4/4/4</td><td>5.87</td><td>5.18</td><td>3.58</td><td>7.17</td><td>4.09</td></tr></table>

表 VII：量化技术及其对 LLAMA-3-8B 中微缩放数据格式影响的消融研究，其中全系统设置量化了所有 9 个 GEMM。结果报告为 WikiText-2 困惑度。GPTQ 用于裁剪：$\mathrm { E r r } _ { y }$ 表示输出范数裁剪；$\mathrm { E r r } _ { w }$ 表示权重范数裁剪。
<table><tr><td>方法</td><td>PPL↓</td><td>方法</td><td>PPL↓</td></tr><tr><td>基线 FP16</td><td>6.13</td><td>仅 ACT 和 KV</td><td></td></tr><tr><td>仅权重</td><td></td><td>MXFP4</td><td>29.75</td></tr><tr><td> $\mathrm { M X } \mathrm { \overset { \smile } { I N T } } + \mathrm { \ ` { R T N } }$ </td><td>6.83</td><td>MXINT4</td><td>7.24</td></tr><tr><td> $\mathrm { M X F P + R T N }$ </td><td>11.94</td><td>MXFP4 + 选择性旋转</td><td>14.50</td></tr><tr><td>MXINT4 + 旋转</td><td>6.98</td><td>MXINT4 + 选择性旋转</td><td>7.05</td></tr><tr><td> $\mathrm { M X F P 4 } + \mathrm { R o t a t i o n }$ </td><td>13.71</td><td> $\mathrm { M X I N T }$ 全系统</td><td></td></tr><tr><td> $\mathrm { M X I N T 4 } + \mathrm { E r r } _ { w } \ \mathrm { C l i p }$ </td><td>6.53</td><td>RTN</td><td>9.32</td></tr><tr><td> $\mathsf { M X I N T 4 } + \mathrm { E r r } _ { y } \mathrm { \thinspace C l i p }$ </td><td>6.45</td><td>Erry 裁剪</td><td>8.41</td></tr><tr><td></td><td></td><td>Erry 裁剪 + 选择性旋转</td><td>7.43</td></tr></table>

首先，MXFP4 在所有设置中始终不如 MXINT4。受此启发，我们在所有后续评估中采用 MXINT 作为默认数据类型。其次，对于仅有权重量化，我们表明旋转通常会损害性能——这简直与微缩放算术不兼容。此外，我们证明我们的输出范数引导的逐块裁剪 $( \mathrm { E r r } _ { y } )$ 与权重误差引导的逐块裁剪 $( \mathrm { E r r } _ { w } )$ 相比取得了更好的性能。第三，选择性旋转有效地增强了 MXFP4 和 MXINT4 的激活和 KV 量化。这与我们在权重量化中的观察不同，在权重量化中旋转对困惑度产生负面影响。我们假设这是由于激活和 KV 值中存在更宽的数值范围，这得益于旋转缓解异常值存在的能力。最后，我们的全系统结果证实，逐块裁剪搜索和选择性激活旋转都提高了整体性能。

TABLE VIII: LLaMA-3 模型在 4-bit 权重、激活和 KV 量化下的零样本下游任务准确率，在 PIQA (PQ)、WinoGrande (WG)、HellaSwag (HS)、Arc-Easy (A-e)、Arc-Challenge (A-c) 和 LAMBADA (LA) 上评估。
<table><tr><td>方法</td><td>PQ</td><td>WG</td><td>HS</td><td>A-e</td><td>A-c</td><td>LA</td><td>平均</td></tr><tr><td>LLaMA-3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FP16</td><td>80.74</td><td>72.77</td><td>79.06</td><td>77.82</td><td>53.33</td><td>75.63</td><td>73.22</td></tr><tr><td>QuaRot [7]</td><td>75.14</td><td>65.82</td><td>72.94</td><td>68.01</td><td>43.34</td><td>65.81</td><td>65.18</td></tr><tr><td>PLENA (本文)</td><td>76.99</td><td>69.85</td><td>75.91</td><td>76.73</td><td>48.72</td><td>72.39</td><td>70.10</td></tr><tr><td>LLaMA-3-70B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FP16</td><td>84.66</td><td>80.51</td><td>84.89</td><td>85.86</td><td>64.25</td><td>79.47</td><td>79.94</td></tr><tr><td>QuaRot</td><td>78.07</td><td>69.30</td><td>77.33</td><td>73.44</td><td>47.53</td><td>69.57</td><td>69.21</td></tr><tr><td>PLENA (本文)</td><td>81.61</td><td>77.98</td><td>84.12</td><td>82.66</td><td>58.62</td><td>79.24</td><td>77.37</td></tr></table>

TABLE IX: 在 Qwen3-32B 上对长上下文和智能体工作负载的评估，涵盖代码生成 (HumanEval [10])、数学推理 (GSM8K-Platinum [64]) 和函数调用 (BFCL-Web Search Base [48]) 基准测试。W/A/KV 表示权重、激活和 KV cache 的位宽。
<table><tr><td>方法</td><td>W/A/KV</td><td>HumanEval pass@1 ↑</td><td>GSM8K-PLA EM↑</td><td>BFCL-W Acc ↑</td></tr><tr><td>基线</td><td>16/16/16</td><td>89.6</td><td>97.85</td><td>27.0</td></tr><tr><td>本文</td><td>4/4/4</td><td>84.1</td><td>97.85</td><td>24.0</td></tr></table>

c) 长上下文和智能体工作负载：我们进一步验证了 PLENA 在强调长上下文和智能体设置的三种工作负载上的量化有效性，在 Table IX 中展示了 4-bit 权重、激活和 KV 量化结果，并在 Table X 中展示了额外的算法消融。

## D. 通过扁平化脉动阵列提高利用率

PLENA 的扁平化脉动阵列对 LLAMA-3-8B 模型的 FFN 和 FlashAttention (FA) 层的利用率分析总结在 Figure 13 中。预填充阶段的结果被省略，因为 FFN 和 FA 在此阶段均以接近最大利用率运行。对于 FlashAttention，它被排除是因为其计算与 batch size 无关，而 FFN 被省略是因为其重要性随着生成的 token 长度增加而降低。

TABLE X: 在 Qwen3 模型上跨 HumanEval、GSM8K-Platinum 和 BFCL-Web Search Base 的量化技术消融研究。$\mathrm { E r r } _ { y }$ 表示输出范数裁剪。Qwen3-8B 用于 HumanEval 和 GSM8K-Platinum，Qwen3- 32B 用于 BFCL-Web Search Base。HumanEval 和 GSM8K-Platinum 的结果使用 Qwen3-8B；BFCL-Web Search Base 的结果使用 Qwen3-32B，因为其更高的基线准确率能更好地隔离量化的影响。
<table><tr><td>配置</td><td>HumanEval pass@1↑</td><td>GSM8K-PLA EM↑</td><td>BFCL-W Acc ↑</td></tr><tr><td>基线 (FP16)</td><td>84.8</td><td>90.9</td><td>27</td></tr><tr><td>仅 W INT4 (RTN)</td><td>82.9</td><td>88.7</td><td>22</td></tr><tr><td> $+ ~ \mathrm { A C T } ~ \& ~ \mathrm { K V } ~ \mathrm { I N T } 4$ </td><td>72.0</td><td>74.4</td><td>15</td></tr><tr><td>+ GPTQ</td><td>73.2</td><td>87.7</td><td>24</td></tr><tr><td> $+ \ \mathrm { E r r } _ { y } \ \mathrm { C l i p }$ </td><td>74.4</td><td>88.6</td><td>24</td></tr><tr><td>+ 选择性旋转</td><td>78.7</td><td>88.8</td><td>24</td></tr></table>

TABLE XI: 在 Table IV 中 batch size $B \ = \ 8 .$ 下，OSWorld-L 工作负载（90k 预填充，8k 输出 token）对 LLAMA-3.3-70B 内存占用和带宽的量化配置影响。W/A/KV 分别表示权重、激活和 KV cache 的位精度。
<table><tr><td>W/A/KV (bits)</td><td>16/16/16</td><td>4/16/16</td><td>4/4/16</td><td>4/4/4</td></tr><tr><td>峰值带宽 (GB/s)</td><td>8192</td><td>8192</td><td>5120</td><td>2048</td></tr><tr><td>KV Cache 占用 (GB)</td><td>239.26</td><td>239.26</td><td>239.26</td><td>59.81</td></tr><tr><td>权重存储 (GB)</td><td>129.46</td><td>32.36</td><td>32.36</td><td>32.36</td></tr></table>

![](images/a52f5afbce62d68b990028cd7e29d4b5746343549d84ee5262f6b994d2a562a6.jpg)  
Fig. 12: 不同脉动阵列形状的矩阵单元的功耗和面积比较。尽管扁平化脉动阵列会产生略高的面积和功耗，但其更高的利用率使得在智能体任务 OSWorld-L 中的 FFN 和 attention 工作负载的有效能耗显著降低。

DC 综合结果报告在 Table XIII 中。这些结果表明，与先前的加速器相比，扁平化脉动阵列在 FFN 和 FlashAttention 层均实现了更高的计算资源利用率。此外，Figure 12 表明，尽管与传统的方形阵列相比存在一些功耗和面积开销，但扁平化脉动组织提供了更高的能效。

脉动阵列优化的整体消融研究呈现在 Figure 14 中。结果表明，扁平化脉动阵列结合原生的 FlashAttention 支持，显著减少了预填充和解码阶段中 attention 和 FFN 组件的执行时间，尤其是对于长上下文推理。

## E. 系统性能分析

系统级性能比较如表 XII 所示，评估了基于 GQA 的小型和大型 LLaMA 模型，以及基于 MoE 的 GPT-OSS 模型和 Qwen3-32B，并支持长上下文输入。PLENA 和 MicroScopiQ 的性能结果是通过我们的分析模拟器获得的。为了公平起见，我们与 4×A100 SXM GPU 系统（每个 GPU 80 GB HBM 和 1.99 TB/s 带宽）、4×H100 SXM GPU 系统（每个 GPU 80 GB HBM 和 3.35 TB/s 带宽）以及 16×TPU v6e 系统（每个设备 32 GB HBM 和 1.56 TB/s 带宽）进行了系统级比较。PLENA 和 MicroScopiQ 均被建模为 16 加速器系统，其聚合 HBM 容量和带宽与 A100 GPU 系统相当。

![](images/6269938e2b2a9ec88b0f3d184d3993efb221cf947cafaefd5143520c26b71af2.jpg)

![](images/fb40e67f01181ca61de0fce42e8d13d6dca8ae922e992b0e260a1778f27ae373.jpg)

![](images/1a53b37827b0ce9bfb4b503b2a7135a40ffaaa0fac6d4cbc54eed7e3fead12fd.jpg)

![](images/b3d443e20eb0ea5dfeee201191d56d0ad2956e51430a7bcc2dcdcbce75503f86.jpg)

Fig. 13: 脉动阵列在 FFN 层中当其块长度（BLEN）与 batch size 对齐时达到最佳利用率。FA = Flash Attention。SA = 脉动阵列。对于 FA，展平阵列通过允许并行处理多个 attention head 来提高利用率，这对于具有较小有效 batch size 的长上下文推理特别高效。  
![](images/f1332d04b6176986c4ea4cc64aaab80c72e6248f0b288a5b9dc9a1134c5dfd67.jpg)  
Fig. 14: 此图展示了在 batch size 为 16 的 LLAMA-3.3-70B 模型中，PLENA 在 prefill 和 decode 阶段的时序性能分解。该分解包括计算活动时间（Comp）、内存活动时间（Mem）、脉动阵列（SA）利用率以及整个推理流程中的内存带宽利用率。借助片上 FlashAttention 支持，大型中间激活被保留在片上而不是写入片外内存，从而大幅减少了内存流量，同时内存预取隐藏了大部分数据访问延迟。此外，展平的脉动阵列配置在 prefill 和 decode 阶段均保持了高利用率。对于 attention 工作负载，展平阵列通过实现并行多头执行和 head 预加载，实现了高计算和内存利用率。

考虑到 GPU 的非计算组件，鉴于制造节点的差异，我们通过近似匹配乘法器数量而非硅片面积来确定设备数量。此外，对于 H100，我们对齐内存容量而不是乘法器数量，因为它提供了比其他平台高得多的计算资源。协同设计选择的 PLENA 配置（BLEN = 32, MLEN = 2048, VLEN = 2048, Precision W/A/KV = 4/4/4）在所有评估的工作负载中均展现出提升的性能。

如图所示，在相同的 HBM 设置和乘法器数量下，PLENA 实现了比 A100 和 TPU v6e 更高的 TPS，对于 agentic 工作负载，最高达到 A100 的 2.23 倍和 TPU v6e 的 4.70 倍。在 PLENA 中观察到的更高 TTFT 是由于它能够使用我们的量化方案在相同的 HBM 容量中存储更多的 batch。随着 batch size 的增加，由于额外的内存访问和计算，prefill 阶段变得更长。

表 XII：表 IV 中各工作负载的系统级比较。性能评估在充分利用 HBM 容量的情况下进行，将每个工作负载-硬件对的批大小 (BS) 设置为可容纳的最大值。注：我们复现了 MicroScopiQ [50] 并将其计算单元部署在 PLENA 平台上进行测试。对于 GPT-OSS 20B (MoE) [6] 和 Qwen3-32B [60]，由于不支持这些配置 [65]，未包含其余加速器和 TPU。
<table><tr><td rowspan="2"></td><td colspan="10"></td><td colspan="7"></td></tr><tr><td colspan="4">(1.4k, 0.2k)</td><td colspan="4">(114k, 5k)</td><td colspan="4">(90k, 8k)</td><td colspan="4">(90k, 8k) 相同批次</td><td></td></tr><tr><td>系统</td><td>TTFT (s)</td><td>TPS (×A100)</td><td>Tok/J</td><td>BS</td><td>TTFT (s)</td><td>TPS (×A100)</td><td>Tok/J</td><td>BS</td><td>TTFT (s)</td><td>TPS (×A100)</td><td></td><td>Tok/J BS</td><td>TTFT (s)</td><td></td><td>TPS (×A100)</td><td>Tok/J</td><td>BS</td></tr>
<tr><td></td><td></td><td></td><td>1.00x</td><td></td><td>7.40</td><td></td><td></td><td></td><td></td><td>1.00x</td><td></td><td>1.00x</td><td>16</td><td>5.00</td><td>1.00x</td><td></td><td>16</td></tr>
<tr><td>A100</td><td>0.68 0.73</td><td>1.00x 1.12x</td><td>1.12x</td><td>2048 4096</td><td>8.63</td><td>1.00x 1.10x</td><td>1.00x 1.10x</td><td>16 32</td><td>5.00 5.97</td><td>1.14x</td><td></td><td>1.14x</td><td>32</td><td>4.79</td><td>1.08x</td><td>1.00x 1.08x</td><td>16</td></tr>
<tr><td>A100 QuaRot [7] H100</td><td>2.42</td><td>1.65x</td><td>0.94x</td><td>2048</td><td>2.66</td><td>2.50x</td><td>1.43x</td><td>16</td><td>1.83</td><td>2.48x</td><td></td><td>1.41x</td><td>16</td><td>1.83</td><td>2.48x</td><td>1.41x</td><td>16</td></tr>
<tr><td>H100 QuaRot [7]</td><td>2.51</td><td>1.77x</td><td>1.01x</td><td>4096</td><td>2.97</td><td>2.57x</td><td>1.47x</td><td>32</td><td>2.01</td><td>2.55x</td><td></td><td>1.46x</td><td>32</td><td>1.77</td><td>2.51x</td><td>1.43x</td><td>16</td></tr>
<tr><td>TPU v6e</td><td>5.61</td><td>0.88x</td><td>N/A</td><td>2048</td><td>7.58</td><td>0.51x</td><td>N/A</td><td>16</td><td>7.23</td><td>0.53x</td><td></td><td>N/A</td><td>16</td><td>7.23</td><td>0.53x</td><td>N/A</td><td>16</td></tr>
<tr><td>MicroScopiQ [50]</td><td>3.47</td><td>0.83x</td><td>1.67x</td><td>8192</td><td>21.28</td><td>0.37x</td><td>0.74x</td><td>64</td><td>19.13</td><td>0.39x</td><td></td><td>0.78x</td><td>64</td><td>4.93</td><td>0.27x</td><td>0.54x</td><td>16</td></tr>
<tr><td>PLENA</td><td>3.41</td><td>1.91x</td><td>3.50x</td><td>8192</td><td>20.13</td><td>1.45x</td><td>2.66x</td><td>64</td><td>18.87</td><td>1.45x</td><td></td><td>2.65x</td><td>64</td><td>4.68</td><td>1.17x</td><td>2.10x</td><td>16</td></tr>
<tr><td></td><td colspan="10"></td><td colspan="3"></td><td colspan="4"></td></tr>
<tr><td></td><td colspan="4"></td><td colspan="4">LLAMA-3.3-70B</td><td colspan="4"></td><td colspan="4"></td><td></td></tr>
<tr><td></td><td></td><td>(1.4k, 0.2k)</td><td></td><td>BS</td><td></td><td>(114k, 5k)</td><td></td><td></td><td></td><td>(90k, 8k) TPS (×A100)</td><td></td><td>BS</td><td>TTFT (s)</td><td></td><td>(90k, 8k) 相同批次</td><td></td><td>BS</td></tr>
<tr><td>系统</td><td>TTFT (s)</td><td>TPS (×A100)</td><td>Tok/J</td><td></td><td>TTFT (s)</td><td>TPS (×A100)</td><td>Tok/J</td><td>BS</td><td>TTFT (s)</td><td></td><td></td><td>Tok/J 1.00x</td><td>4</td><td></td><td>TPS (×A100)</td><td>Tok/J</td><td></td></tr>
<tr><td>A100</td><td>0.78</td><td>1.00x</td><td>1.00x</td><td>256 512</td><td>43.18 42.89</td><td>1.00x</td><td>1.00x</td><td>4</td><td>29.67</td><td>1.00x</td><td></td><td>1.13x</td><td>29.67 27.69</td><td></td><td>1.00x 1.11x</td><td>1.00x</td><td>4 4</td></tr>
<tr><td>A100 QuaRot [7]</td><td>1.17 0.34</td><td>1.08x 2.34x</td><td>1.08x 1.34x</td><td>256</td><td>14.30</td><td>1.13x 2.13x</td><td>1.13x 1.21x</td><td>8 4</td><td>32.17 10.10</td><td>1.13x 2.04x</td><td>1.22x</td><td>8 4</td><td>10.10</td><td></td><td>2.04x</td><td>1.11x 1.22x</td><td>4</td></tr>
<tr><td>H100 H100 QuaRot [7]</td><td>0.44</td><td>2.36x</td><td>1.35x</td><td>512</td><td>16.12</td><td>2.19x</td><td>1.25x</td><td>8</td><td>11.37</td><td>2.14x</td><td>1.22x</td><td>8</td><td>9.88</td><td></td><td>2.08x</td><td>1.18x</td><td>4</td></tr>
<tr><td>TPU v6e</td><td>11.7</td><td>0.85x</td><td>N/A</td><td>256</td><td>41.96</td><td>0.46x</td><td>N/A</td><td>4</td><td>37.61</td><td>0.47x</td><td>N/A</td><td>4</td><td>37.61</td><td></td><td>0.47x</td><td>N/A</td><td>4</td></tr>
<tr><td>MicroScopiQ [50]</td><td>8.32 7.58</td><td>0.79 1.82x</td><td>1.59x</td><td>1024</td><td>73.28</td><td>0.20x</td><td>0.41x</td><td>16</td><td>49</td><td>0.17x</td><td></td><td>0.35x 16</td><td>23.93</td><td></td><td>0.11x</td><td>0.23x</td><td>4</td></tr>
<tr><td>PLENA</td><td></td><td></td><td>3.32x</td><td>1024</td><td>69.10</td><td>2.23x</td><td>4.07x</td><td>16</td><td>43.43</td><td>2.21x</td><td></td><td>4.04x 16</td><td></td><td>21.68</td><td>1.34x</td><td>2.45x</td><td>4</td></tr>
<tr><td></td><td colspan="14">GPT-OSS 20B (MoE)</td><td rowspan="2"></td><td colspan="4"></td></tr>
<tr><td></td><td>(1.4k, 0.2k)</td><td></td><td></td><td></td><td></td><td>(114k, 5k)</td><td></td><td></td><td></td><td>(90k, 8k)</td><td></td><td></td><td></td><td>(90k, 8k) 相同批次</td><td></td><td></td></tr>
<tr><td>系统</td><td colspan="2">TTFT (s) TPS (×A100)</td><td></td><td>Tok/J BS</td><td>TTFT (s)</td><td>TPS (×A100)</td><td>Tok/J</td><td>BS</td><td>TTFT (s)</td><td>TPS (×A100)</td><td></td><td>Tok/J</td><td>BS</td><td>TTFT (s)</td><td>TPS (×A100)</td><td>Tok/J</td><td>BS</td></tr>
<tr><td>A100</td><td>1.46</td><td>1.00x</td><td>1.00x</td><td>1024</td><td>11.81</td><td>1.00x</td><td>1.00x</td><td>8</td><td>8.05</td><td>1.00x</td><td>1.00x</td><td>8</td><td>8.05</td><td></td><td>1.00x</td><td>1.00x</td><td>8</td></tr>
<tr><td>H100</td><td>4.03</td><td>0.89x</td><td>0.51x</td><td>1024</td><td>1.85</td><td>3.10x</td><td>1.78x</td><td>8</td><td>1.38</td><td>2.90x</td><td></td><td>1.66x 64</td><td>9.77</td><td>1.38</td><td>2.90x 0.99x</td><td>1.66x</td><td>88</td></tr>
<tr><td>PLENA</td><td>13.41 1.15x</td><td></td><td>2.10x</td><td>4096</td><td>47.63</td><td>1.96x</td><td>3.58x</td><td>64</td><td>41.08</td><td>1.93x</td><td></td><td>3.52x</td></table>

表 XIII：不同脉动阵列设计的计算面积、利用率和可达 FLOPs。基线使用 64 × 64 阵列，而 PLENA 使用扁平化的 4 × 1024 阵列。S.A.T 表示在 GSM8K 上的标准可达 TOPs，A.A.T 表示在表 IV 的 OSWorld-L 工作负载上的智能体可达 TOPs，均在整个推理流程中测量。SystolicAttn 具有更大的计算面积，因为它使用未量化的 FP16，但通过在阵列内融合完整的 Attention 计算实现了高利用率。
<table><tr><td>设计</td><td>计算面积 (mm2)</td><td>TOPs/mm²</td><td>S.A.T/mm2*</td><td>A.A.T/mm2*</td></tr>
<tr><td>MicroscopiQ [50]</td><td>0.1378</td><td>59.45</td><td>26.36</td><td>5.83</td></tr>
<tr><td>Olive [25]</td><td>0.319</td><td>25.66</td><td>13.76</td><td>2.40</td></tr>
<tr><td>FIGNA [32]</td><td>0.471</td><td>17.39</td><td>7.51</td><td>1.83</td></tr>
<tr><td>SystolicAttn [39]</td><td>1.17</td><td>14.00</td><td>7.14</td><td>4.76</td></tr>
<tr><td>PLENA</td><td>0.237</td><td>34.49</td><td>29.31</td><td>12.81</td></tr></table>

## VI. 结论

我们提出了 PLENA，一种利用扁平化脉动阵列实现高效智能体推理加速的新型加速器设计系统。我们识别了智能体模型推理中由内存带宽与容量墙所导致的资源利用率不足挑战。为解决这些问题，我们提出了一种面向 MX 硬件加速的非对称量化方案，以及对 FlashAttention 的原生架构支持。在硬件之外，PLENA 还提供了一个完整的系统探索框架，包括新的 ISA 支持、自动化代码生成、多级模拟器以及协同设计探索引擎。这提供了一个超越特定加速器架构实现的探索平台，使未来研究能够为新兴 Transformer 模型进行原型设计与优化探索，类似于 Berkeley 面向 DNN 的 Gemmini 框架 [21]。

我们未来的工作将聚焦于将 PLENA 与 GPU 系统集成，以实现异构 LLM 加速，充分发挥两种架构的优势。此外，随着多轮和多模态智能体工作负载日益普及，我们计划扩展 PLENA 以更好地支持和优化这些新兴工作负载。

## 致谢

本工作得到了高级研究与发明局（Advanced Research and Invention Agency）在扩展计算计划（Scaling Compute Programme）下的支持。

[1] F. Abecassis, A. Agrusa, D. Ahn, J. Alben, S. Alborghetti, M. Andersch, S. Arayandi, A. Bjorlin, A. Blakeman, E. Briones et al., “Pretraining Large Language Models with NVFP4,” arXiv preprint arXiv:2509.25149, 2025.

[2] M. Agarwal, J. J. Barroso, T. Chakraborti, E. M. Dow, K. Fadnis, B. Godoy, M. Pallan, and K. Talamadupula, “Project CLAI: Instrumenting the Command Line as a New Environment for AI Agents,” 2020. [Online]. Available: https://arxiv.org/abs/2002.00762

[3] S. Agashe, K. Wong, V. Tu, J. Yang, A. Li, and X. E. Wang, “Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents,” 2025. [Online]. Available: https://arxiv.org/abs/2504.00906

[4] M. AI. (2025, Apr.) The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation. Accessed: 2025-08-16. [Online]. Available: https://ai.meta.com/blog/llama-4-multimodal-intelligence

[5] R. Y. Aminabadi, S. Rajbhandari, M. Zhang, A. A. Awan, C. Li, D. Li, E. Zheng, J. Rasley, S. Smith, O. Ruwase, and Y. He, “DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale,” 2022. [Online]. Available: https://arxiv.org/abs/2207.00032

[6] M. Artetxe, S. Bhosale, N. Goyal, T. Mihaylov, M. Ott, S. Shleifer, X. V. Lin, J. Du, S. Iyer, R. Pasunuru, G. Anantharaman, X. Li, S. Chen, H. Akin, M. Baines, L. Martin, X. Zhou, P. S. Koura, B. O’Horo, J. Wang, L. Zettlemoyer, M. Diab, Z. Kozareva, and V. Stoyanov, “Efficient Large Scale Language Modeling with Mixtures of Experts,” 2022. [Online]. Available: https://arxiv.org/abs/2112.10684

[7] S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, P. Cameron, M. Jaggi, D. Alistarh, T. Hoefler, and J. Hensman, “QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs,” 2024. [Online]. Available: https://arxiv.org/abs/2404.00456

[8] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, “Language Models are Few-Shot Learners,” 2020. [Online]. Available: https://arxiv.org/abs/2005.14165

[9] H. Chae, N. Kim, K. T. iunn Ong, M. Gwak, G. Song, J. Kim, S. Kim, D. Lee, and J. Yeo, “Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation,” 2025. [Online]. Available: https://arxiv.org/abs/2410.13232

[10] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba, “Evaluating Large Language Models Trained on Code,” 2021.

[11] T. L. S. D. Chezelles, M. Gasse, A. Drouin, M. Caccia, L. Boisvert, M. Thakkar, T. Marty, R. Assouel, S. O. Shayegan, L. K. Jang, X. H. Lu, O. Yoran, D. Kong, F. F. Xu, S. Reddy, Q. Cappart,\` G. Neubig, R. Salakhutdinov, N. Chapados, and A. Lacoste, “The BrowserGym Ecosystem for Web Agent Research,” 2025. [Online]. Available: https://arxiv.org/abs/2412.05467

[12] L. T. Clark, V. Vashishtha, L. Shifren, A. Gujja, S. Sinha, B. Cline, C. Ramamurthy, and G. Yeric, “ASAP: A 7-nm finFET predictive process design kit,” Microelectronics Journal, vol. 53, pp. 105–115, Jul. 2016.

[13] T. Dao, “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning,” 2023. [Online]. Available: https://arxiv.org/abs 2307.08691

[14] B. Darvish Rouhani, R. Zhao, V. Elango, R. Shafipour, M. Hall, M. Mesmakhosroshahi, A. More, L. Melnick, M. Golub, G. Varatkar et al., “With shared microexponents, a little shifting goes a long way,” in Proceedings of the 50th Annual International Symposium on Computer Architecture, 2023, pp. 1–13.

[15] M. Davies, N. Crago, K. Sankaralingam, and C. Kozyrakis, “Efficient LLM Inference: Bandwidth, Compute, Synchronization,

and Capacity are all you need,” 2025. [Online]. Available: https: //arxiv.org/abs/2507.14397

[16] A. Drouin, M. Gasse, M. Caccia, I. H. Laradji, M. Del Verme, T. Marty, D. Vazquez, N. Chapados, and A. Lacoste, “WorkArena: How Capable are Web Agents at Solving Common Knowledge Work Tasks?” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, Eds., vol. 235. PMLR, 21–27 Jul 2024, pp. 11 642–11 662. [Online]. Available: https://proceedings.mlr.press/v235/drouin24a.htm

[17] Z. Fan, K. Vasilevski, D. Lin, B. Chen, Y. Chen, Z. Zhong, J. M. Zhang, P. He, and A. E. Hassan, “SWE-Effi: Re-Evaluating Software AI Agent System Effectiveness Under Resource Constraints,” 2025. [Online]. Available: https://arxiv.org/abs/2509.09853

[18] C. M. Fonseca, A. P. Guerreiro, M. Lopez-Ib´ anez, and L. Paquete, “On´ the computation of the empirical attainment function,” in International Conference on Evolutionary Multi-criterion Optimization. Springer, 2011, pp. 106–120.

[19] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “Gptq: Accurate post-training quantization for generative pre-trained transformers,” arXiv preprint arXiv:2210.17323, 2022.

[20] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou, “The language model evaluation harness,” 07 2024. [Online]. Available: https://zenodo.org/records/12608602

[21] H. Genc, S. Kim, A. Amid, A. Haj-Ali, V. Iyer, P. Prakash, J. Zhao, D. Grubb, H. Liew, H. Mao, A. Ou, C. Schmidt, S. Steffl, J. Wright, I. Stoica, J. Ragan-Kelley, K. Asanovic, B. Nikolic, and Y. S. Shao, “Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration,” in Proceedings of the 58th Annual Design Automation Conference (DAC), 2021.

[22] S. Ghodrati, S. Kinzer, H. Xu, R. Mahapatra, Y. Kim, B. H. Ahn, D. K. Wang, L. Karthikeyan, A. Yazdanbakhsh, J. Park, N. S. Kim, and H. Esmaeilzadeh, “Tandem Processor: Grappling with Emerging Operators in Neural Networks,” in Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ser. ASPLOS ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 1165–1182. [Online]. Available: https://doi.org/10.1145/3620665. 3640365

[23] A. Gholami, Z. Yao, S. Kim, C. Hooper, M. W. Mahoney, and K. Keutzer, “AI and Memory Wall,” IEEE Micro, vol. 44, no. 3, p. 33–39, May 2024. [Online]. Available: https://doi.org/10.1109/MM. 2024.3373763

[24] Google, “System Architecture: TPU VM,” https://cloud.google.com/ tpu/docs/system-architecture-tpu-vm, Google Cloud, Technical Report, 2025, last updated August 1, 2025.

[25] C. Guo, J. Tang, W. Hu, J. Leng, C. Zhang, F. Yang, Y. Liu, M. Guo, and Y. Zhu, “Olive: Accelerating large language models via hardwarefriendly outlier-victim pair quantization,” in Proceedings of the 50th Annual International Symposium on Computer Architecture, 2023, pp. 1–15.

[26] C. Guo, C. Wei, J. Tang, B. Duan, S. Han, H. Li, and Y. Chen, “Transitive Array: An Efficient GEMM Accelerator with Result Reuse,” 2025. [Online]. Available: https://arxiv.org/abs/2504.16339

[27] C. Guo, C. Zhang, J. Leng, Z. Liu, F. Yang, Y. Liu, M. Guo, and Y. Zhu, “Ant: Exploiting adaptive numerical data type for low-bit deep neural network quantization,” in 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE, 2022, pp. 1414– 1433.

[28] H. He, W. Yao, K. Ma, W. Yu, Y. Dai, H. Zhang, Z. Lan, and D. Yu, “WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models,” 2024. [Online]. Available: https: //arxiv.org/abs/2401.13919

[29] K. Hong, G. Dai, J. Xu, Q. Mao, X. Li, J. Liu, K. Chen, Y. Dong, and Y. Wang, “FlashDecoding++: Faster Large Language Model Inference on GPUs,” 2024. [Online]. Available: https://arxiv.org/abs/2311.01282

[30] W. Hu, H. Zhang, C. Guo, Y. Feng, R. Guan, Z. Hua, Z. Liu, Y. Guan, M. Guo, and J. Leng, “M-ANT: Efficient Low-bit Group Quantization for LLMs via Mathematically Adaptive Numerical Type,” in 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA). IEEE, 2025, pp. 1112–1126.

[31] Y. Ishibashi and Y. Nishimura, “Self-Organized Agents: A LLM Multi-Agent Framework toward Ultra Large-Scale Code Generation and Optimization,” 2024. [Online]. Available: https://arxiv.org/abs/2404. 02183

[32] J. Jang, Y. Kim, J. Lee, and J.-J. Kim, “FIGNA: Integer Unit-Based Accelerator Design for FP-INT GEMM Preserving Numerical Accuracy,” in 2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2024, pp. 760–773.

[33] J. Jiang, F. Wang, J. Shen, S. Kim, and S. Kim, “A Survey on Large Language Models for Code Generation,” 2024. [Online]. Available: https://arxiv.org/abs/2406.00515

[34] J. Knowles, “A summary-attainment-surface plotting method for visualizing the performance of stochastic multiobjective optimizers,” in 5th International Conference on Intelligent Systems Design and Applications (ISDA’05). IEEE, 2005, pp. 552–557.

[35] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, “Large Language Models are Zero-Shot Reasoners,” 2023. [Online]. Available: https://arxiv.org/abs/2205.11916

[36] D. Lee, J. Lee, K. Kim, J. Tack, J. Shin, Y. W. Teh, and K. Lee, “Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents,” 2025. [Online]. Available: https://arxiv.org/abs/2503.10689

[37] J. Lee, W. Lee, and J. Sim, “Tender: Accelerating Large Language Models via Tensor Decomposition and Runtime Requantization,” in 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA), 2024, pp. 1048–1062.

[38] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration,” Proceedings of Machine Learning and Systems, vol. 6, pp. 87–100, 2024.

[39] J. Lin, G. Chen, Y. Li, and T. Bourgeat, “SystolicAttention: Fusing FlashAttention within a Single Systolic Array,” 2025. [Online]. Available: https://arxiv.org/abs/2507.11331

[40] Y. Lu, J. Yang, Y. Shen, and A. Awadallah, “OmniParser for Pure Vision Based GUI Agent,” 2024. [Online]. Available: https: //arxiv.org/abs/2408.00203

[43] S. Merity, C. Xiong, J. Bradbury, and R. Socher, “Pointer sentinel mixture models,” arXiv preprint arXiv:1609.07843, 2016.

[42] F. Meng, P. Tang, X. Tang, Z. Yao, X. Sun, and M. Zhang, “TransMLA: Multi-Head Latent Attention Is All You Need,” 2025. [Online]. Available: https://arxiv.org/abs/2502.07864

[41] H. Luo, Y. C. Tugrul, F. N. Bostancı, A. Olgun, A. G. Ya ˘ glıkc¸ı, and˘ O. Mutlu, “Ramulator 2.0: A Modern, Modular, and Extensible DRAM Simulator,” 2023. [Online]. Available: https://arxiv.org/abs/2308.11030

[44] A. Meta, “Introducing meta llama 3: The most capable openly available llm to date,” Meta AI, 2024.

[45] M. Muller and G. ¨ Zuni <sup>ˇ</sup> c, “Browser Use: Enable AI to control yourˇ browser,” https://github.com/browser-use/browser-use, 2024, gitHub repository.

[46] B. Niu, Y. Song, K. Lian, Y. Shen, Y. Yao, K. Zhang, and T. Liu, “Flow: Modularized Agentic Workflow Automation,” 2025. [Online]. Available: https://arxiv.org/abs/2501.07834

[47] OpenAI, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, R. Avila, I. Babuschkin, S. Balaji, V. Balcom, P. Baltescu, H. Bao, M. Bavarian, J. Belgum, I. Bello, J. Berdine, G. Bernadett-Shapiro, C. Berner, L. Bogdonoff, O. Boiko, M. Boyd, A.-L. Brakman, G. Brockman, T. Brooks, M. Brundage, K. Button, T. Cai, R. Campbell, A. Cann, B. Carey, C. Carlson, R. Carmichael, B. Chan, C. Chang, F. Chantzis, D. Chen, S. Chen, R. Chen, J. Chen, M. Chen, B. Chess, C. Cho, C. Chu, H. W. Chung, D. Cummings, J. Currier, Y. Dai, C. Decareaux, T. Degry, N. Deutsch, D. Deville, A. Dhar, D. Dohan, S. Dowling, S. Dunning, A. Ecoffet, A. Eleti, T. Eloundou, D. Farhi, L. Fedus, N. Felix, S. P. Fishman, J. Forte, I. Fulford, L. Gao, E. Georges, C. Gibson, V. Goel, T. Gogineni, G. Goh, R. Gontijo-Lopes, J. Gordon, M. Grafstein, S. Gray, R. Greene, J. Gross, S. S. Gu, Y. Guo, C. Hallacy, J. Han, J. Harris, Y. He, M. Heaton, J. Heidecke, C. Hesse, A. Hickey, W. Hickey, P. Hoeschele, B. Houghton, K. Hsu, S. Hu, X. Hu, J. Huizinga, S. Jain, S. Jain, J. Jang, A. Jiang, R. Jiang, H. Jin, D. Jin, S. Jomoto, B. Jonn, H. Jun, T. Kaftan, Łukasz Kaiser, A. Kamali, I. Kanitscheider, N. S. Keskar, T. Khan, L. Kilpatrick, J. W. Kim, C. Kim, Y. Kim, J. H. Kirchner, J. Kiros, M. Knight, D. Kokotajlo, Łukasz Kondraciuk, A. Kondrich, A. Konstantinidis, K. Kosic, G. Krueger, V. Kuo, M. Lampe, I. Lan,

T. Lee, J. Leike, J. Leung, D. Levy, C. M. Li, R. Lim, M. Lin, S. Lin, M. Litwin, T. Lopez, R. Lowe, P. Lue, A. Makanju, K. Malfacini, S. Manning, T. Markov, Y. Markovski, B. Martin, K. Mayer, A. Mayne, B. McGrew, S. M. McKinney, C. McLeavey, P. McMillan, J. McNeil, D. Medina, A. Mehta, J. Menick, L. Metz, A. Mishchenko, P. Mishkin, V. Monaco, E. Morikawa, D. Mossing, T. Mu, M. Murati, O. Murk, D. Mely, A. Nair, R. Nakano, R. Nayak, A. Neelakantan, R. Ngo,´ H. Noh, L. Ouyang, C. O’Keefe, J. Pachocki, A. Paino, J. Palermo, A. Pantuliano, G. Parascandolo, J. Parish, E. Parparita, A. Passos, M. Pavlov, A. Peng, A. Perelman, F. de Avila Belbute Peres, M. Petrov, H. P. de Oliveira Pinto, Michael, Pokorny, M. Pokrass, V. H. Pong, T. Powell, A. Power, B. Power, E. Proehl, R. Puri, A. Radford, J. Rae, A. Ramesh, C. Raymond, F. Real, K. Rimbach, C. Ross, B. Rotsted, H. Roussez, N. Ryder, M. Saltarelli, T. Sanders, S. Santurkar, G. Sastry, H. Schmidt, D. Schnurr, J. Schulman, D. Selsam, K. Sheppard, T. Sherbakov, J. Shieh, S. Shoker, P. Shyam, S. Sidor, E. Sigler, M. Simens, J. Sitkin, K. Slama, I. Sohl, B. Sokolowsky, Y. Song, N. Staudacher, F. P. Such, N. Summers, I. Sutskever, J. Tang, N. Tezak, M. B. Thompson, P. Tillet, A. Tootoonchian, E. Tseng, P. Tuggle, N. Turley, J. Tworek, J. F. C. Uribe, A. Vallone, A. Vijayvergiya, C. Voss, C. Wainwright, J. J. Wang, A. Wang, B. Wang, J. Ward, J. Wei, C. Weinmann, A. Welihinda, P. Welinder, J. Weng, L. Weng, M. Wiethoff, D. Willner, C. Winter, S. Wolrich, H. Wong, L. Workman, S. Wu, J. Wu, M. Wu, K. Xiao, T. Xu, S. Yoo, K. Yu, Q. Yuan, W. Zaremba, R. Zellers, C. Zhang, M. Zhang, S. Zhao, T. Zheng, J. Zhuang, W. Zhuk, and B. Zoph, “GPT-4 Technical Report,” 2024. [Online]. Available: https://arxiv.org/abs/2303.08774

[48] S. G. Patil, H. Mao, F. Yan, C. C.-J. Ji, V. Suresh, I. Stoica, and J. E. Gonzalez, “The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models,” in Forty-second International Conference on Machine Learning, 2025.

[49] J. Qin, T. Xia, C. Tan, J. Zhang, and S. Q. Zhang, “PICACHU: Plug-In CGRA Handling Upcoming Nonlinear Operations in LLMs,” in Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ser. ASPLOS ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 845–861. [Online]. Available: https://doi.org/10.1145/3676641.3716013

[50] A. Ramachandran, S. Kundu, and T. Krishna, “Microscopiq: Accelerating foundational models through outlier-aware microscaling quantization,” in Proceedings of the 52nd Annual International Symposium on Computer Architecture, 2025, pp. 1193–1209.

[51] S. Rando, L. Romani, A. Sampieri, L. Franco, J. Yang, Y. Kyuragi, F. Galasso, and T. Hashimoto, “LongCodeBench: Evaluating Coding LLMs at 1M Context Windows,” 2025. [Online]. Available: https: //arxiv.org/abs/2505.07897

[52] B. Rouhani, N. Garegrat, T. Savell, R. Zhao, and A. More, “OCP Microscaling Formats (MX) Specification,” Open Compute Project, 2023.

[53] B. D. Rouhani, R. Zhao, A. More, M. Hall, A. Khodamoradi, S. Deng, D. Choudhary, M. Cornea, E. Dellinger, K. Denolf, S. Dusan, V. Elango, M. Golub, A. Heinecke, P. James-Roxby, D. Jani, G. Kolhe, M. Langhammer, A. Li, L. Melnick, M. Mesmakhosroshahi, A. Rodriguez, M. Schulte, R. Shafipour, L. Shao, M. Siu, P. Dubey, P. Micikevicius, M. Naumov, C. Verrilli, R. Wittig, D. Burger, and E. Chung, “Microscaling Data Formats for Deep Learning,” 2023. [Online]. Available: https://arxiv.org/abs/2310.10537

[54] B. D. Rouhani, R. Zhao, A. More, M. Hall, A. Khodamoradi, S. Deng, D. Choudhary, M. Cornea, E. Dellinger, K. Denolf et al., “Microscaling data formats for deep learning,” arXiv preprint arXiv:2310.10537, 2023.

[55] A. Samajdar, E. Qin, M. Pellauer, and T. Krishna, “Self adaptive reconfigurable arrays (SARA): learning flexible GEMM accelerator configuration and mapping-space using ML,” in Proceedings of the 59th ACM/IEEE Design Automation Conference, ser. DAC ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 583–588. [Online]. Available: https://doi.org/10.1145/3489517.3530506

[56] A. Samajdar, Y. Zhu, P. Whatmough, M. Mattina, and T. Krishna, “SCALE-Sim: Systolic CNN Accelerator Simulator,” arXiv preprint arXiv:1811.02883, 2018.

[57] S. Sarangi and B. Baas, “DeepScaleTool: A Tool for the Accurate Estimation of Technology Scaling in the Deep-Submicron Era,” in 2021 IEEE International Symposium on Circuits and Systems (ISCAS), 2021, pp. 1–5.

[58] W. Shao, M. Chen, Z. Zhang, P. Xu, L. Zhao, Z. Li, K. Zhang, P. Gao, Y. Qiao, and P. Luo, “Omniquant: Omnidirectionally calibrated quantization for large language models,” arXiv:2308.13137, 2023.

[59] L. Steiner, M. Jung, F. S. Prado, K. Bykov, and N. Wehn, “DRAMSys4.0: A Fast and Cycle-Accurate SystemC/TLM-Based DRAM Simulator,” in Embedded Computer Systems: Architectures, Modeling, and Simulation: 20th International Conference, SAMOS 2020, Samos, Greece, July 5–9, 2020, Proceedings. Berlin, Heidelberg: Springer-Verlag, 2020, p. 110–126. [Online]. Available: https://doi.org/ 10.1007/978-3-030-60939-9 8

[60] Q. Team, “Qwen3 Technical Report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

[61] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez,\` A. Joulin, E. Grave, and G. Lample, “LLaMA: Open and Efficient Foundation Language Models,” 2023. [Online]. Available: https: //arxiv.org/abs/2302.13971

[62] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.

[63] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention Is All You Need,” 2023. [Online]. Available: https://arxiv.org/abs/1706.03762

[64] J. Vendrow, E. Vendrow, S. Beery, and A. Madry, “Do large language model benchmarks test reliability?” arXiv preprint arXiv:2502.03461, 2025.

[65] vLLM Project, “Hardware Supported Models – TPU (Text-Only Language Models),” https://docs.vllm.ai/en/v0.9.2/models/hardware supported models/tpu.html#text-only-language-models, 2025, accessed: 2025-11-12.

[66] S. Watanabe, “Python tool for visualizing variability of Pareto fronts over multiple runs,” arXiv preprint arXiv:2305.08852, 2023.

[67] J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, E. H. Chi, T. Hashimoto, O. Vinyals, P. Liang, J. Dean, and W. Fedus, “Emergent Abilities of Large Language Models,” 2022. [Online]. Available: https://arxiv.org/abs/2206.07682

[68] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “Smoothquant: Accurate and efficient post-training quantization for large language models,” in International Conference on Machine Learning. PMLR, 2023, pp. 38 087–38 099.

[69] T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. J. Hua, Z. Cheng, D. Shin, F. Lei, Y. Liu, Y. Xu, S. Zhou, S. Savarese, C. Xiong, V. Zhong, and T. Yu, “OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments,” 2024. [Online]. Available: https://arxiv.org/abs/2404.07972

[70] A. H. Zadeh, I. Edo, O. M. Awad, and A. Moshovos, “Gobo: Quantizing attention-based nlp models for low latency and energy efficient inference,” in 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE, 2020, pp. 811–824.

[71] D. Zan, Z. Huang, W. Liu, H. Chen, L. Zhang, S. Xin, L. Chen, Q. Liu, X. Zhong, A. Li, S. Liu, Y. Xiao, L. Chen, Y. Zhang, J. Su, T. Liu, R. Long, K. Shen, and L. Xiang, “Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving,” 2025. [Online]. Available: https://arxiv.org/abs/2504.02605

[72] S. Zeng, J. Liu, G. Dai, X. Yang, T. Fu, H. Wang, W. Ma, H. Sun, S. Li, Z. Huang, Y. Dai, J. Li, Z. Wang, R. Zhang, K. Wen, X. Ning, and Y. Wang, “FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs,” in Proceedings of the 2024 ACM/SIGDA International Symposium on Field Programmable Gate Arrays, ser. FPGA ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 223–234. [Online]. Available: https://doi.org/10.1145/3626202.3637562

[73] Z. Zeng, P. Chen, S. Liu, H. Jiang, and J. Jia, “MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=br4H61LOoI

[74] H. Zhang, A. Ning, R. B. Prabhakar, and D. Wentzlaff, “LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference,” in 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA), 2024, pp. 1080–1096.

[75] Y. Zhao, C.-Y. Lin, K. Zhu, Z. Ye, L. Chen, S. Zheng, L. Ceze, A. Krishnamurthy, T. Chen, and B. Kasikci, “Atom: Low-bit quantization for

efficient and accurate llm serving,” Proceedings of Machine Learning and Systems, vol. 6, pp. 196–209, 2024.

[76] J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, and L. Hou, “Instruction-Following Evaluation for Large Language Models,” 2023. [Online]. Available: https://arxiv.org/abs/2311.0791