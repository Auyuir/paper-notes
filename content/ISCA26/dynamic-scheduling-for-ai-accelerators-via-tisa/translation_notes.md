# Dynamic Scheduling for AI Accelerators via TISA 原文翻译

# 基于 TISA 的 AI 加速器动态调度

Guanghui Song<sup>1,∗</sup>, Xiaoqiang Dan<sup>2,∗</sup>, Chengke Wang<sup>2</sup>, Fei Liu<sup>2</sup>, Wenyuan Lv<sup>2</sup>, Zhongzhou Jiang<sup>2</sup>, Jianjian Guan<sup>2</sup>, Teng Lu<sup>2</sup>, Lin Tao<sup>2</sup>, Cheng Li<sup>2</sup>, Weixing Pan<sup>2</sup>, Wei Huang<sup>2</sup>, Zirong Shen<sup>2</sup>, Yi Yang<sup>2</sup>, Hui Liu<sup>2</sup>, Jie Zhao<sup>1,†</sup>

<sup>1</sup>湖南大学，中国 <sup>2</sup>EVAS Intelligence，中国

<sup>∗</sup>同等贡献。 <sup>†</sup>通讯作者：jiezhao@hnu.edu.cn

摘要——现代 AI 加速器存在利用率低的问题，因为静态的编译时调度无法适应运行时的可变性，也无法有效地协调异构单元。本文提出了一个语义感知的动态分块调度框架，该框架恢复了自适应执行所需的缺失的运行时语义。它协同设计了三个组件，包括一个在下传过程中保持算子边界和依赖类型的语义保持编译器，一个编码类型化依赖、资源意图和分块级内存范围的分块级指令集（TISA），以及一个利用这些语义动态重排分块、解决争用，并在张量、向量和 DMA 单元之间重叠执行的冲突感知运行时调度器。该设计统一了软件语义和硬件调度，实现了超越静态方法的跨算子和跨迭代并行。在 ResNet50、BERT、GPT-J 和 LLaMA2 上，我们的工作比基线实现了 1.52–1.92 倍的加速，比强静态分块级流水线调度带来了 1.14–1.63 倍的额外提升；在 FlashAttention-3（头维度 128）上，它与最先进的 H100 实现相比，将硬件利用率提高了 26.4%。消融研究进一步表明，仅语义保持就能带来 1.2 倍的收益，证实了将调度信息恢复到运行时的独立价值。

关键词——动态调度，分块级指令集，深度学习加速器，异构单元

## I. 引言与动机

深度学习的快速发展推动了日益复杂的加速器的发展 [11], [35]。代表性平台包括 GPU（包括 NVIDIA 和 AMD [26]）、TPU [20]、Ascend [25] 和 Tenstorrent [37], [38]。这些系统采用异构单元架构，集成 Tensor、Vector 和 DMA 引擎以利用大规模并行性，如 Table I 所示。

TABLE I: 各厂商的异构执行单元。我们将厂商特定术语统一为“Tensor Units”和“Vector ALUs”。“DMA Engines”表示 DMA / 拷贝引擎 / 片上数据包搬运器。映射遵循厂商文档和先前的工作 [20], [25], [26], [37], [38]。
<table><tr><td>厂商</td><td>Tensor Units</td><td>Vector ALUs</td><td>DMA Engines</td></tr><tr><td>NVIDIA</td><td>Tensor</td><td>CUDA cores</td><td>TMA</td></tr><tr><td>AMD</td><td>Matrix</td><td>SIMD</td><td>SDMA</td></tr><tr><td>TPU</td><td>MXU</td><td>V-ALU</td><td>async DMA</td></tr><tr><td>Ascend</td><td>Cube</td><td>VU</td><td>on-chip DMA</td></tr><tr><td>Tenstorrent</td><td>SFPU</td><td>FPU</td><td>on-chip DMA</td></tr></table>

当前的 AI 加速器主要依赖编译时编排的流水线模板来协调异构单元。尽管现代 GPU 在 warp 发射、thread-block 分配和异步内存执行方面采用了动态调度，但这些机制运行在 tile 级别的算子协调之下，并且不会基于语义就绪状态动态重排跨单元执行。这种方法仍然普遍存在，因为它符合长期的软硬件契约：编译器发出确定的指令流，硬件以最小的控制开销可预测地执行它们。该模型简化了验证和工具链设计，并与批量同步编程范式（如 CUDA [27]、XLA [36]、TensorRT [28]）保持一致。

然而，随着模型和硬件复杂性的增加，静态 tile 级别的流水线调度暴露出一些瓶颈。尽管重叠 Tensor、Vector 和 DMA 单元有望实现指令级并行（ILP）[8], [22], [35], [39], [41]，但编译时决策无法响应运行时变异性，如 DMA 背压、cache-bank 冲突或热降频。因此，执行偏离了编译时假设：tile 无法跨单元重新定时或重新平衡，留下空闲气泡和未充分利用的硬件。Tile 在这里自然出现，因为现代编译器已经将算子划分为 tile，以匹配片上内存和带宽约束。因此，tile 代表了计算、内存和 DMA 共享的最细粒度的语义粒度，使其成为自适应调度的理想单元。

尽管无处不在，静态 tile 级别的流水线调度存在四个限制可扩展性和效率的基本局限性：

• 编译时复杂性和工程负担。协调跨单元依赖关系并探索重排机会带来了与最小完工时间和装箱问题 [18] 相关的优化挑战，这两者都是 NP-hard 问题。因此，开发者通常求助于特定架构的启发式方法和手动调优，这在系统复杂性增加时扩展性很差。

• 抽象粒度不足。CUDA 流和图级运行时以粗粒度管理 kernel，掩盖了 tile 边界和数据依赖。相反，指令级 调度 [46] 在非常细的粒度上运行，语义在很大程度上丢失。Tile 级调度实现了有效的平衡：它保留了算子上下文并暴露了跨单元重叠，同时保持硬件可调度 [43]。

• 无法适应运行时变异性。静态 tile 级流水线调度假设固定的延迟和带宽，但实际系统表现出动态波动，如带宽争用、Cache/SPM 冲突以及由于热量或操作系统效应导致的单元不同步。这些变化改变了真实的关键路径，然而静态方法无法相应地重新定时或重新重叠 tile。

• 历史先例：静态方法在不确定性下失败。超标量 CPU [7], [15] 通过动态调度实现高 ILP，而静态 VLIW/IA-64 [5], [13] 架构未能在不同工作负载和微架构变化中泛化。这一历史表明，将有限且具有语义感知的调度部分移至运行时既健壮又可扩展。

总的来说，这些局限性突出了对运行时可见语义层的需求，该层能够在适当的粒度下实现自适应重排。

动态调度提供了一条前进的道路，但将其应用于任意指令粒度对于深度学习加速器来说是不切实际的。Tile 抽象提供了最佳平衡点：每个 tile 封装了具有明确定义的数据范围和资源需求的语义连贯计算。这种粒度使运行时能够 (1) 精确检测依赖关系，(2) 安全仲裁异构资源，以及 (3) 响应运行时条件调整执行。因此，Tile 粒度调度将动态系统的适应性与编译时推理的语义安全性相结合，构成了我们设计的基础。

我们提出了一种 tile 粒度的语义感知动态调度框架，通过三个协同组件的协同设计统一了软件语义和硬件调度：一个保持语义的编译器，在编译流水线中保持算子标识、类型化依赖和资源亲和性，防止限制当前运行时的语义侵蚀；一个 tile 级别的 ISA (TISA)，作为现有每单元执行 ISA 之上的正交调度-语义层，编码每个 tile 的算子类型、依赖描述符、资源意图和内存范围，为运行时提供足够的信息来推理合法性、就绪状态和重叠，而无需昂贵的静态分析；以及一个语义引导的运行时调度器，它消耗 TISA 语义以在 Tensor、Vector 和 DMA 单元之间动态重排 tile，解决争用并在运行时变异性下利用跨算子和跨迭代并行性。

该工作流直接解决了静态 tile 级流水线调度的弱点：(1) 语义保持消除了组合式编译时重排的需要；(2) tile 级粒度平衡了调度可见性和硬件可处理性；(3) 动态决策使执行适应运行时漂移，同时确保正确性。

我们在多个加速器（包括 Epoch、H100 和 A100 GPU）上，跨代表性工作负载（包括 ResNet50 [17]、BERT [10]、GPT-J [31]、LLaMA2 [42] 和 DeepSeek-R1 [16]）评估了我们的动态调度框架。我们的工作实现了比基线高 1.52–1.92× 的加速，比强静态 tile 级流水线调度提供了 1.14–1.63× 的额外改进。对于主流 head-dim-128 配置中的 FlashAttention 算子，它比最先进的 H100 FlashAttention-3 [33] 提供了约 26.4% 的更高硬件利用率。消融实验进一步表明，仅语义保持就带来了 1.2× 的收益，突出了恢复运行时可见语义的独立价值。总而言之，这项工作的主要贡献是：

• 我们设计了一个语义感知的动态 tile 调度框架，为异构 AI 加速器桥接了静态编译和运行时适应能力。

• 我们引入了 TISA，一个 tile 级别的指令集，暴露了类型化依赖、算子语义和资源需求，以实现安全的动态重排。

• 我们构建了一个保持语义的编译器，该编译器在从高级框架到硬件可调度 TISA 指令的过程中保持算子上下文。

• 我们展示了全面的实验验证，表明在跨加速器和工作负载的情况下，利用率、适应性和可移植性均有所提高。

本文的组织结构如下。第二节指出了具体的调度挑战。第三节分析了根本性差距并提出了总体架构。第四节对 TISA 语义进行了形式化，随后分别在第五节和第六节介绍了我们的动态调度器和语义保持编译器。第七节介绍了我们框架的实现，而第八节提供了全面的实验验证。最后，第九节讨论了相关工作，第十节进行总结。

## II. 瓦片调度挑战

本节详细阐述了上述挑战。动态调度分块计算必须考虑两种相互交织的冲突形式，即结构性争用和数据依赖，它们的相互作用决定了瓦片何时可以安全地并行执行。我们使用 FlashAttention-3 [33] 中的融合瓦片来说明这一点，如图 1 所示。

<table><tr><td rowspan=1 colspan=1>M00</td><td rowspan=1 colspan=1>M01</td><td rowspan=1 colspan=1>MO2</td><td rowspan=1 colspan=1>M03</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>M0N-3</td><td rowspan=1 colspan=1>M0N-2</td><td rowspan=1 colspan=1>MON-1</td></tr><tr><td rowspan=3 colspan=4>So    S1    S2    S3</td><td rowspan=1 colspan=1></td><td rowspan=3 colspan=3>SN-3   SN-2   SN-1</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>M10</td><td rowspan=1 colspan=1>M11</td><td rowspan=1 colspan=1>Ml2</td><td rowspan=1 colspan=1>M13</td><td rowspan=1 colspan=1>…</td><td rowspan=1 colspan=1>M1N-3</td><td rowspan=1 colspan=1>M1N-2</td><td rowspan=1 colspan=1>M1N-1</td></tr></table>

图 1：FlashAttention-3 中的融合与分块计算。分块后，Attention 模式涉及的三个操作被融合，并沿水平方向迭代执行。每次迭代（以不同颜色显示）包含三个瓦片：实线框表示张量单元执行，虚线框表示向量单元 softmax 计算。垂直箭头表示迭代内的数据依赖。

Attention 机制被分解为三种异构的瓦片类型：M0、S 和 M1，分别对应 GEMM0 $( Q K ^ { \top } )$ 、Softmax 和 GEMM1 (Attention · V) 的分块计算。每个垂直的瓦片序列代表一次迭代，包含三个操作：两个张量单元 GEMM（M0 和 M1）和一个向量单元 Softmax (S)。箭头表示迭代内的数据依赖。

结构性争用。张量和向量单元是不同但互补的资源。两个 GEMM 阶段独占地占用张量单元，而 Softmax 阶段使用向量单元。编译器生成的内核和像 FlashAttention-3 这样手工调优的内核所使用的静态瓦片级流水线调度器，都利用了这种迭代内的异构性：当向量单元执行 $S _ { i }$ 时，张量单元可以并发地开始同一次迭代中的下一个 GEMM（例如，将 $S _ { i }$ 与 $M 1 _ { i }$ 重叠）。这种迭代内并行性已经被静态流水线充分利用。

数据依赖。然而，这三个分块操作通过生产者-消费者边相连。$S _ { i }$ 依赖于 $M 0 _ { i }$ 的输出，而 $M 1 _ { i }$ 依赖于 $S _ { i }$ 的输出。静态编译通过有序融合或循环携带的同步来正确执行这些依赖关系。然而，这种执行在迭代之间引入了隐式屏障：直到迭代 i 的所有依赖结果完成并同步后，下一次迭代 (i+1) 才能开始。

影响：隐式同步与错失紧凑的迭代间并发。尽管结构异构性允许张量和向量单元并行运行，但跨迭代的隐式同步阻碍了它们实现更紧凑的流水线并行。例如，$S _ { i }$ 和 $M 0 _ { i + 1 }$ 是数据无关的且使用不相交的资源，但静态调度通过全局排序将它们串行化。结果，张量和向量单元都会周期性空闲：张量单元在 $S _ { i }$ 期间停顿，而向量单元在 $M 0 _ { i + 1 }$ 期间停顿，尽管真正的依赖关系并不要求这样做。图 2 说明了这种隐式同步。图 2(a) 展示了图 1 中融合瓦片的顺序执行，其中异构单元的资源利用率不足显而易见。

图 2(b) 展示了一种静态流水线方法，它将三个分块操作划分为两个阶段：第一阶段是 $M 0 _ { i }$ 和 $S _ { i }$ ，第二阶段是 $M 1 _ { v }$ 。这种双阶段结构提高了利用率，消除了一些空闲时间 $( E _ { 0 } )$ 。然而，瓦片启动序列是预计算并在编译时固定的，通过显式同步屏障（例如，GPU 内核中的 PTX 级 bar.sync 或 NPU 内核中编译器放置的栅栏）来强制执行。这种调度假设了确定性的执行时间；任何运行时的变化（如缓存未命中、DMA 背压或去同步化的线程束）都会破坏对齐并降低利用率。

图 2(d) 中展示了一种更激进的静态流水线，它通过将每个分块操作视为一个独立的阶段来增加阶段数。这种三阶段设计重叠了相邻操作，并实现了更大的节省（E）。然而，它对编译时排序的依赖仍然在迭代之间强制执行同步；固定的顺序阻止了 $S _ { i }$ 与 $M 0 _ { i + 1 }$ 的重叠，尽管它们是相互独立的。

虽然高级的软件流水线（例如，模调度 [23], [32]）可以通过利用可预测的延迟和控制流谓词来实现迭代间重叠，但现代 AI 加速器和 GPU 面临着根本不同的约束：(1) 由于 DMA 背压、共享内存 bank 冲突和热降频导致的非确定性执行延迟；(2) 缺乏用于重量级张量和向量单元的硬件谓词；(3) 超出经典模调度范围的异构多单元协调需求。此外，这种高级软件流水线主要重叠细粒度指令，而加速器内核操作的是跨多个专用单元协调的粗粒度瓦片。因此，指令级软件流水线充其量只能在图 2(b,d) 所示的静态阶段结构内略微平滑流水线气泡，但并不能从根本上改变阶段有序的执行模板。因此，诸如 FlashAttention-3 之类的最先进的 GPU 内核依赖于固定的同步屏障，将执行锁定在静态重叠模板中，并排除了基于就绪状态的动态跨迭代重排序。

![](images/05cf9efabbd8f0b944e16a47971964207912f2e2cdee9abb672aaf4eef1e2480.jpg)  
图 2：图 1 中融合瓦片的不同执行方式。阴影区域 $( E _ { x } )$ 表示通过调度节省的延迟。垂直虚线表示静态调度施加的迭代间同步屏障。$\begin{array} { r } { E _ { 0 } + E _ { 2 } = E _ { 1 } + E _ { 3 } } \end{array}$ 说明了双阶段或三阶段动态调度可实现的等效延迟节省。

这些静态调度暴露了一个核心低效问题：编译时假设抹去了关于算子类型、资源亲和性和依赖方向的语义信息，迫使采用保守的同步来保证正确性，但牺牲了性能。相比之下，语义感知的运行时保留了这些信息，并根据瞬时硬件状态做出调度决策。如

图 2(c) 和 所示，运行时调度器会观察哪些单元空闲以及哪些瓦片的输入已就绪，然后动态发出下一个符合条件的瓦片。这允许更紧凑的迭代间重叠，例如并发执行 $S _ { i }$ 与 $M 0 _ { i + 1 }$ 或 M1<sub>i</sub> 与 $S _ { i + 1 }$ ，并自然适应运行时变化，在不依赖刚性阶段定义的情况下实现 $E _ { 0 } + E _ { 2 }$ 或 $E _ { 1 } + E _ { 3 }$ 的累积延迟降低。

关键洞察。根本区别在于信息保持。当前加速器编译流程中的静态 tile 级流水线调度在 tiling 和融合将程序降低为不透明的指令流后，便丢失了算子语义；因此，硬件只能看到有序的指令序列，无法区分真实依赖与人为依赖。我们的语义感知方法在运行时保持这些关系，使硬件能够区分真正的资源冲突（两个单元均繁忙）和可恢复的停顿（一个单元在等待）。通过基于就绪状态和资源可用性动态重排 tile，它消除了不必要的同步，并恢复了静态流水线加速器内核（例如图 2(b,d) 中的 FlashAttention）必须保守放弃的空闲时间。

## III. 设计概述

在识别出由于在不恰当的层级进行调度而导致的语义丢失，以及由于保守的编译时假设可能强制同步的问题后，我们现在介绍我们解决方案的设计。我们的目标是恢复运行时可见的语义，即硬件安全且自适应地调度 tile 所需的最小信息集，同时保持与现有编译流程的兼容性。

一方面，深度学习工作负载的编译本质上是一个层次化分解过程，将网络从高层算子降低为硬件可执行指令。在此过程中，算子和数据粒度逐步细化以对齐目标硬件 ISA。虽然这种层次化降低对效率至关重要，但它不可避免地丢弃了对动态调度不可或缺的关键语义信息（如算子边界、资源亲和性和依赖类型）。另一方面，现有 ISA 并非围绕 tile 级执行设计，即使编译器尝试保持或表达此类语义也难以实现。

这种语义丢失在软件和硬件之间造成了关键鸿沟：一旦算子被分解为扁平的指令流，运行时便不再知道哪些指令属于哪个算子，或者它们应占用哪些功能单元。超标量 CPU 通过基于寄存器的依赖跟踪来解决这一问题的一个更简单版本，但 AI 加速器需要更丰富的语义接口来描述算子级依赖、异构资源绑定和 tile 级内存区域。弥合这一语义鸿沟需要一个中间抽象层：它比 kernel 流更细粒度，但比原始指令更粗粒度，这正是 tile 级 ISA 的角色。

基于这些观察，我们设计了 TISA，一种保持语义的抽象，在软件分解和硬件调度之间架起桥梁。这一抽象使我们能够开发一个动态 tile 调度器，同时保持与现有编译流程的兼容性。图 3 展示了我们方法的整体架构。

![](images/5922dd03371347d5c15a0cba08b5218ab9af0684e9ff047eb72ebe4fbcccddf8.jpg)
图 3：我们集成框架的整体架构。

与传统 CPU ISA 同时编码执行语义（如 ADD 计算什么）和调度语义（通过寄存器名称实现基于记分板的依赖跟踪）不同，现有领域专用 ISA（如 Cambricon [9]、TPU [20]、Graphcore IPU [19]）主要定义执行语义——张量操作计算什么以及在哪个单元上——但省略了调度语义。这导致它们通常通过显式屏障或排序约束来编码单元间协调，而非暴露结构化的、硬件可消费的调度语义。TISA 通过充当硬件可消费的调度语义层来弥合这一鸿沟：其 OpType、UnitMap 和 TileMem 字段类似于 CPU ISA 中的寄存器名称，但是在异构单元间的 tile 粒度上，这与 Task Superscalar [12] 的多核 CPU 任务级动态调度有所不同。

值得注意的是，这一功能无法在软件中实现。在 AI 核心 tile 级别，调度预算严格在纳秒范围内：我们的 RTL 综合测量结果显示每个 tile 调度为 7 ∼ 9 个周期（在 1 GHz 频率下）。在控制处理器上执行的软件运行时需要指令取指、解码、分支评估和内存访问来进行依赖检查，通常产生微秒级开销（慢 100−1000 倍）。这将抵消 tile 级调度的优势，而 tile 执行本身通常在 10<sup>3</sup> 到 10<sup>5</sup> 个周期范围内。因此，硬件 ISA 接口对于纳秒级的机会性重叠是必不可少的。

作为比传统指令级 ISA 更高层级的 ISA，TISA 集成了三个协同设计的组件，共同将语义信息恢复到运行时：

• TISA 抽象定义了一种 tile 级指令格式，编码算子类型、依赖语义和资源需求。它在编译器和运行时调度器之间建立了语义契约。

• 语义感知运行时调度器解释 TISA 元数据，在异构执行单元间动态调度 tile，实现运行时的自适应重叠、负载均衡和冲突解决。

• 保持语义的编译器在整个编译流程中保持算子语义，并输出 TISA 指令作为目标中间表示。

这些组件共同弥合了编译与执行之间的语义鸿沟。TISA 层确保算子意图和依赖语义在降低过程中得以保留，使运行时调度器能够做出当前固定模板静态调度方法在运行时未能利用的、知情的自适应决策。我们现在依次详细介绍每个组件。

## IV. TISA：一种语义保持的 Tile 级 ISA

为了支持动态调度，TISA 必须在轻量级的指令格式中编码所有与调度相关的语义（计算什么、需要哪些资源，以及其数据如何与其他 tile 交互）。这种设计产生了两种互补的数据结构：操作数结构，用于指定数据属性和内存作用域；以及 TISA 指令结构，用于捕获每个 tile 的语义和资源级元数据。这两种结构共同构成了编译器和运行时调度器之间完整的语义契约。这两种结构的详细定义分别如表 II 和表 III 所示。

表 II：操作数结构定义。
<table><tr><td>字段</td><td>类型</td><td>描述</td></tr><tr><td colspan="3">Operand = (TileShape, TileMem, AccessType)</td></tr><tr><td>TileShape TileMem AccessType</td><td>Symbolic/Parametric (base, scope) {R, W, RW}</td><td>计算边界 内存规范 访问模式</td></tr><tr><td colspan="3">TileMem = (base, scope)</td></tr><tr><td>base scope</td><td>Address {Private,Local,Shared}</td><td>符号/常量地址 内存层级</td></tr></table>

表 III：TISA 指令结构定义。
<table><tr><td>字段</td><td>类型/约束</td><td>描述</td></tr><tr><td></td><td>TISA_Inst = (OpType, Operands, Attributes, UnitMap)</td><td></td></tr><tr><td>OpType Operands</td><td>{GEMM, SOFTMAX, ...} [op1, op2, ...]</td><td>语义标识符 Operand 数组</td></tr><tr><td>Attributes</td><td> $| o u t s | \leq 3 , | i n s | \leq 7$  Op Attrs,Schedule Params</td><td>硬件约束 重排约束，</td></tr><tr><td>UnitMap</td><td>(unit, quantity, affinity)</td><td>同步要求 资源规范</td></tr></table>

每条 TISA 指令保留了静态指令流通常会抹除的三类语义：

• 计算语义，由 OpType 表示，用于标识算子及其预期的执行类别（例如，张量、向量或标量单元）。这使调度器能够将指令与兼容的硬件单元相关联。

• 数据语义，由 Operands 和 TileMem 捕获，定义了数据使用的空间和时间范围，允许跨 tile 进行细粒度的冲突检测。

• 调度语义，编码在 Attributes 和 UnitMap 中，用于约束重排并指定资源亲和性，以确保在异构环境下的正确性。

这些字段共同使运行时能够做出经过合法性检查的明智调度决策，而这些决策以前被锁定在编译时。更详细的调度机制将在第五节中描述，但在此做一个简要总结是有用的：OpType 指导结构映射，Deps 和 TileMem 确保数据正确性，而 UnitMap 实现分布式的每单元仲裁。

TISA 指令之间的依赖关系表示为 Deps = {(src, type, condition)}，其中 type ∈ {RAW, WAR, WAW} 表示写后读、读后写和写后写关系。这些依赖关系是通过对每个操作数的 TileMem 字段进行基于区间的重叠分析来自动推导的。例如，如果两个 tile 引用了重叠的地址范围，并且一个写而另一个读，则会创建 RAW 依赖并强制执行，直到写入者提交。condition 字段表示部分或有条件的就绪状态（例如，部分 tile 可用性），允许调度器在所需子区域变为有效时提前发射依赖的 tile。该模型允许跨迭代的细粒度依赖解析和重叠，同时保证内存安全。

我们当前的设计采用基于区间的重叠分析，使用连续的 [start\_addr, end\_addr] 范围。对于非连续或跨步访问（例如，列向量），该模型是保守的：它可能会对跨步内的间隙标记错误的冲突，但绝不会遗漏真正的冲突。规则的非连续访问通常在更高的编译级别通过转置等操作转换为连续访问。使用显式的跨步元数据扩展 TileMem 描述符仍然是未来的工作。

动态调度器消耗每条 TISA 指令的语义字段，以协调跨异构单元的执行。它首先使用 Deps 和 TileMem 执行 RAW/WAR/WAW 验证以检测冲突。然后，在 UnitMap 和 OpType 的指导下，它将就绪的 tile 分配给张量、向量或 DMA 引擎的单元本地队列，动态解决结构性争用。最后，它评估依赖条件，以便在数据就绪时机会主义地发射指令，而不是在固定的编译时同步点。此过程在保持正确性的同时，最大化了跨算子和迭代的重叠。

因此，TISA 抽象通过将计算什么（算子意图）与何时何地执行（调度）清晰地分离开来，弥合了软硬件语义鸿沟。与不透明的内核流或扁平的指令序列不同，TISA 向运行时公开了类型化的依赖关系、显式的资源映射（UnitMap）和 tile 级内存区域（TileMem），从而实现了基于规则的合法性检查和自适应调度，这将在第五节 A 中介绍。

正如我们在第三节中介绍的，TISA 定义了一个硬件消耗的调度契约，类似于 CPU ISA 中的寄存器名称如何构成基于记分板的调度的架构契约。硬件调度器直接读取 TISA 字段（OpType、TileMem、UnitMap）以做出分发决策——这些字段不由软件解释。虽然违反或省略 TISA 语义会导致保守（而非不正确）的执行——类似于绕过 CPU 记分板并回退到顺序发射——但调度语义是架构上可见且由硬件消耗的，满足了 ISA 级抽象的本质属性。特别是，TISA 在 Epoch（将在第八节中评估的 AI 加速器）上具有具体的二进制编码。由于篇幅限制，编码细节被省略。最好将其理解为一种调度语义 ISA 扩展，用于补充现有的每单元执行 ISA。

对于上游编译器，TISA 提供了一个语义目标 IR：框架可以发射 TISA 指令，同时保留算子标识、依赖关系和资源需求。然后，运行时利用这些信息，通过类型化依赖就绪状态、基于区间的冲突测试和动态的每单元仲裁来确保正确性，从而消除了进行昂贵的不透明流分析的需要。对于下游硬件，TISA 对异构执行后端进行了抽象：通用 GPU 和特定领域的 NPU 都可以在同一契约下得到支持，其中 OpType 可以对应于软件级算子或粗粒度硬件指令。这种统一的抽象实现了在多样硬件上的语义感知调度，而无需重新架构运行时。

总体而言，TISA 提供了缺失的语义接口，允许动态调度对算子意图和硬件资源状态进行推理，构成了我们语义感知运行时框架的基础。

## V. 动态 Tile 调度

借助语义上下文（OpType）、资源映射（UnitMap）和内存范围描述符（TileMem），动态调度器可以在运行时安全地利用跨算子重叠和自适应资源分配。我们现在将详细说明类型化依赖分析和异构资源管理如何协同工作以实现这一能力。

## A. 类型化依赖分析

传统的指令调度器依赖基于寄存器或保守的基于内存的依赖跟踪，这要么无法捕获更高层的语义，要么保守地阻碍了并行性。相比之下，我们的动态分块调度器执行类型化依赖分析，显式利用 TISA 指令中的语义标注。这种方法消除了人为依赖，同时确保了正确性，实现了跨算子和迭代的安全重排与并行执行。每条 TISA 指令通过 TileMem 字段暴露其数据接口，这些字段描述了地址范围、内存作用域（例如，L1-private、L2-local 或 HBM channel）以及访问类型（READ/WRITE）。对不相交内存作用域或不重叠范围的访问被视为独立的，允许调度器并发发射指令而不会有数据冒险的风险。

为了跟踪运行时状态，每个执行单元 u 维护一个在途语义表：

$$
\mathcal { F } _ { u } = \left\{ \begin{array} { c l } & { ( i d x , s t a r t \_ a d d r , e n d \_ a d d r , } \\ & { a c c e s s \_ t y p e , u n i t , i n s t \_ p t r ) } \end{array} \right\}
$$

这里，idx 索引指令的操作数；start\_addr/end\_addr 表示该操作数访问的地址区间；access\_type 是访问模式（READ/WRITE）；unit 是目标执行单元；而 inst\_ptr 是指向该指令的指针。该表允许调度器推断部分完成情况（例如，当子分块或内存区域准备就绪时），并在其所需数据可用时立即唤醒依赖的分块。与跟踪寄存器标签的传统记分牌不同，$\mathcal { F } _ { u }$ 编码了语义和空间上下文，从而实现细粒度、作用域感知的依赖解析。

在评估候选指令 I 是否可以发射到单元 u 时，我们的动态调度器执行基于规则的冒险检测：

$$
\mathrm { H a z a r d } ( I , \mathcal { F } _ { u } ) = \exists r \in \mathcal { F } _ { u } : \mathsf { S e m a n t i c C o n f l i c t } ( I , r )
$$

其中 SemanticConflict 通过以下类型化规则来解决：

• 数据依赖（RAW/WAR/WAW）：对 TileMem 使用区间重叠测试以识别真实冲突；不相交的范围被视为独立的。

• 内存作用域隔离：针对不同内存层级或存储体的操作（例如，L1 与 L2，或独立的 HBM 通道）是非别名化的，可以安全地并行进行。

• 语义兼容性：具有兼容 OpTypes（例如，GEMM 与 Softmax）的指令如果其数据范围不冲突且其单元类别不同，则可以重叠。

• 资源可行性：如果当前单元容量（例如，本地缓冲区大小或带宽）无法容纳 I，调度器会推迟其发射，直到释放了足够的资源。

这些规则检查通过将数据、结构和语义维度结合在一个统一的框架中，扩展了传统的冒险模型。它们允许调度器区分影响正确性的真实冒险与可被利用于并行执行的无害重叠。目前，TISA 的 TileMem 缺乏原生的跨步数组访问，要求编译器将列转置为连续地址，或者发出产生错误依赖的保守边界。然而，由于密集的 LLM 和 CNN 本质上在大型连续块上操作，精度损失仍然可以忽略不计。

## B. 异构资源管理

我们的动态分块调度器在分布式的一组异构执行单元上运行，每个单元由一个独立的、语义感知的队列对管理。调度机制被组织为四个协作步骤（图 4），它们共同形成了一个去中心化的仲裁机制。

步骤 1：语义路由。传入的 TISA 指令被解析其 OpType、UnitMap 和依赖元数据，然后路由到每个目标单元的相应等待队列（WQ）。每个 WQ 保留算子语义，并提供每种单元类型的就绪候选者的局部视图。

步骤 2：依赖解析。调度器定期从每个 WQ 中选择一个就绪窗口 W，并针对该单元的在途表 $\mathcal { F } _ { u }$ 检查语义冒险。只有通过依赖和资源检查的指令才会被提升至发射队列（IQ）。此步骤在确保正确性的同时，实现了跨单元的乱序准入。

![](images/b0ba14e5b1d1e6031d46dddfcd50257c1f8ce7ada784a28e41f7abd53211f034.jpg)  
图 4：每单元语义调度。去中心化队列将依赖检查局部化，并防止异构单元之间无关的阻塞。WQ：等待队列；IQ：发射队列。

步骤 3：自适应发射。一旦 IQ 条目的依赖被清除，它们就会被发射到硬件执行流水线。

步骤 4：反馈。完成后，相应的 $\mathcal { F } _ { u }$ 条目被退役，依赖指令被通知，并且根据观察到的争用或延迟自适应地更新每单元的调度优先级。这种反馈机制在运行时持续调整重叠和单元利用率。

这些步骤自然形成了一个五级微架构：(1) 接收（指令）解码；(2) 路由到每单元的 WQ；(3) 匹配窗口的依赖检查；(4) 将无冲突指令从 WQ 发射到 $\mathrm { I Q } ;$ 以及 (5) 将 IQ 指令分派到各单元。

发射的 TISA 指令以运行至完成、非抢占的方式执行，调度决策仅在分块边界做出。由于这些边界是粗粒度的（通常超过 $1 0 ^ { 3 }$ 次操作），控制开销保持较低，即根据我们的 RTL 综合测量，每次分派约为 7∼9 个周期。

## C. 动态调度算法

算法 1 形式化了这一调度过程。在宏观层面上，调度器持续接收带有语义注释的 TISA 指令，将其路由到适当的队列，执行依赖和资源检查，乱序发射就绪的 tiles，并在完成时更新运行时状态。算法 2 详细描述了支撑这一过程的语义冲突检测例程。通过这种机制，调度器实现了图 2(c,e) 中所示的执行模式。

调度周期复杂度为 $O ( U \cdot W \cdot | \mathcal { F } | _ { \operatorname* { m a x } } )$ ，其中 U 是执行单元的数量，W 是窗口大小， $| \mathcal { F } | _ { \operatorname* { m a x } }$ 是每个单元中在途条目的最大数量。在典型设置 $( W ~ \leq ~ 8 , ~ | \mathcal { F } _ { u } | ~ \leq ~ 1 6 )$ 下，有效复杂度在每个周期接近 $O ( U )$ 且常数极小。冲突检测子程序对每个候选者运行时间为 $O ( | \mathcal { F } _ { u } | )$ ，并使用常数空间的重叠检查。这种设计的扩展性优于需要 $O ( N ^ { 2 } )$ 全局比较的集中式 ILP 调度器（例如，Tomasulo [41]），同时实现了更细粒度的、语义驱动的并行性。

我们综合的 RTL 实现将每个加速器核心集成一个调度器（表 IV）。在 $W = 8 ,$ 时，调度器需要 1.5M 门（0.25 mm<sup>2</sup>，每个核心面积的 1.5%，100 mW）。扩展到 $W = 2 5 6$ 时，面积呈次平方级增长至 6.8M 门（32 倍条目对应 4.5 倍增长），且具有有限的 9 周期发射延迟，这得益于对数 CAM 结构和 $W \geq 3 2$ 的流水线仲裁。在 $W \ : = \ : 2 5 6$ 时，功耗保持在核心功耗的 <0.3%，因为发射是稀疏的（约 5% slots/cycle）。8 条目基线足以满足大多数算子的需求；更大的窗口仅有利于访存受限的 kernel，其延迟、门数、面积和功耗在表 IV 中有所增长。

```prolog
Algorithm 1: Dynamic Tile Scheduling
Input: Stream of semantically-annotated TISA instructions
Output: Scheduled instruction sequences across
heterogeneous units
Initialize semantic tracking structures for all units u;
while system running do
// 1: Semantic Routing
if Reception Buffer ̸= empty then
I ← pop(Reception Buffer) ;
extract semantic context(I) ;
// Analyze OpType, dependencies, resource needs
u ← adaptive unit selection(I) ;
// Consider load balancing
enqueue with priority(WQ[u], I) ;
foreach u in Units do
C ← select ready window(WQ[u]);
// Semantic-aware selection
foreach I ∈ C (by adaptive priority) do
// 2: Dependency Resolution(call Algorithm 2)
if !semantic conflict detection( $I , \mathcal { F } _ { u } )$ and
resources available(u) then
allocate resources adaptively(I, u);
update semantic tracking( $\mathcal { F } _ { u } , I ) ;$
// 3: Adaptive Issue
issue out of order(I, u);
foreach u in Units do
foreach completed inst J from Exec[u] do
// 4: Feedback
update semantic state $( \mathcal { F } _ { u } , J ) ;$
trigger dependent instructions(WQ[u], J);
adapt scheduling policy(u);
```

Algorithm 2: Semantic Conflict Detection   
Input: Instruction I with semantic annotations, in-flight   
semantic table $\mathcal { F } _ { u }$   
foreach $r \in \mathcal { F } _ { u }$ do   
if not same scope(I, r) then   
continue ;   
if semantic compatibility(I.OpType, r.OpType) then   
// Semantically compatible operations can overlap   
continue ;   
if memory range overlap(I, r) and true dependency(I,   
r) then   
if cannot reorder safely(I, r) then   
// True semantic conflict detected   
return true ;   
// Safe to execute in parallel   
return false ;

表 IV：调度器随窗口大小的扩展。
<table><tr><td>W</td><td>延迟</td><td>门数</td><td>面积 (mm²)</td><td>功耗 (mW)</td></tr><tr><td>8</td><td>7 周期</td><td>1.5M</td><td>0.25</td><td>100</td></tr><tr><td>16</td><td>7 周期</td><td>2.0M</td><td>0.33</td><td>120</td></tr><tr><td>32</td><td>8 周期</td><td>2.8M</td><td>0.46</td><td>150</td></tr><tr><td>64</td><td>8 周期</td><td>3.9M</td><td>0.65</td><td>180</td></tr><tr><td>128</td><td>9 周期</td><td>5.2M</td><td>0.87</td><td>240</td></tr><tr><td>256</td><td>9 周期</td><td>6.8M</td><td>1.13</td><td>300</td></tr></table>

如 图 2(c,e) 所示，最终的执行模式展示了比静态方法所能实现的更紧凑的跨算子和跨迭代并发。静态流水线依赖保守的同步假设——显式屏障和固定延迟（图 5）——来确保正确性，从根本上限制了重叠。虽然此类流水线实现了一定程度的并发，但 TISA 的语义感知消除了这些刚性约束：调度器动态解决依赖关系并在没有程序员插入的同步的情况下重排 tiles，从而在保持正确性的同时产生更紧密的重叠。

## VI. 语义保持编译器

TISA 抽象不仅在运行时支持动态分块调度，还提供了一种统一的中间表示，使编译器能够将语义信息保留至硬件接口。我们的编译器栈反映了深度学习编译传统的层次化分解流程，同时保持算子上下文、依赖关系和资源语义直到生成 TISA。端到端流程由框架桥接器、图编译器、融合编译器、TISA 生成器和特定于后端的代码生成组成。这些组件共同将模型从高级算子图逐步降低至硬件可执行的 TISA 二进制文件，同时保留动态调度器使用的语义元数据。

a) 框架桥接器：桥接层使用 torchxla [36] 前端从 PyTorch [3]、JAX [6] 和 TensorFlow [1] 导入模型，将框架图导出为 XLA 或 StableHLO 方言。通过将 TISA 的 OpType 分类法与 StableHLO 算子抽象对齐，我们确保了跨框架的算子语义映射一致性，简化了后续的依赖和资源分析。

b) 图编译器 (GC)：我们基于 MLIR [24] 的 GC 消费 StableHLO IR 并执行架构感知优化，包括融合、分块和局部性驱动的重排序。它生成一个软件调度的分块图（例如，图 2），该图揭示了跨异构单元（张量、向量、DMA）的合法重叠机会，同时最小化片外通信。与传统的图优化器不同，GC 在自定义的 MLIR 方言中显式保留算子边界和类型化依赖边，形成语义丰富的中间表示，作为融合编译器的输入。

分块维度在 SRAM 容量约束（例如，Epoch 256 KB 暂存缓冲区的 64×64）下最大化算术强度。不可整除的张量边界触发边缘分块。TISA 不依赖填充开销，而是直接编码这些边界的精确 Shape 和 TileMem 范围，生成定制指令，无缝调度更小的边缘分块而不会产生未对齐惩罚。对于包含数千个并发分块的工作负载，层次化调度确保了可扩展性：每个核在本地管理 256 个分块，编译器通过静态分块到核的分配执行全局协调。

c) 融合编译器 (FC)：FC 将 GC 生成的融合子图特化为 TISA 兼容的算子。FC 构建于 MLIR 之上，定义了自定义的 TISA 方言，其操作（例如，tisa.gemm、tisa.softmax）通过 OpType 和依赖描述符编码算子语义，通过 $\mathtt { U n i t M a p } .$ 编码资源意图（到执行单元类的映射），并通过符号化 TileMem 范围和作用域编码内存访问模式。通过该方言，编译器将软件调度的分块图转换为保留算子身份、数据依赖和资源亲和性的 TISA 指令流。这些属性构成了运行时调度器后续消费的语义契约。

传统编译器将算子展平为循环嵌套，丢弃张量级边界。TISA 编译器在分块粒度截断降低——它不需要降低到细粒度 ISA 指令，因为硬件调度器直接消费分块级语义。这简化了优化：例如，乒乓缓冲只需分配两个缓冲区并发出交替的 TISA 分块；不需要循环展开或指令重排，因为运行时调度器处理重叠。语义三元组从高级图组件被同样编码到二进制输出中，无缝地将上下文过渡到硬件。

d) TISA 生成器和后端：TISA 生成器提供虚拟分块级指令集，统一多个硬件后端。其操作语义反映 StableHLO 算子，而其数据语义定义在适合 L1/L2 或共享 SRAM 容量的分块上。OpType 静态绑定到加速器单元类（张量、向量、DMA），这允许运行时执行合法性检查并启用跨单元重叠。

目前，实现了两个后端：(1) TISA-NPU 后端，针对我们的 Epoch 硬件，提供完整的动态调度支持。它使用自定义的基于 LLVM 的降低路径，将 TISA 元数据嵌入到最终二进制文件中，供硬件调度器消费。(2) TISA-CPU 后端，发出优化的 CPU 内核用于功能验证和参考执行。在 CPU 上，重叠的分块串行执行，但该后端保留了相同的 TISA 语义，实现了端到端验证。两个后端都保留相同的语义描述符，以保证跨平台一致的调度行为。

e) 运行时接口：在执行期间，编译后的二进制文件发出每个分块的描述符，这些描述符编码了所有必需的调度属性，包括 OpType、UnitMap 和 TileMem。这些描述符形成就绪集，填充运行时的等待队列 (WQs) 和发射队列 (IQs)，在第五节描述的动态分块调度器下进行仲裁和分发。每个分块以运行至完成、非抢占的方式执行，调度决策在分块边界做出，以平衡自适应性与低硬件开销。

f) 讨论：这种编译器-运行时协同设计闭合了语义保持与动态执行之间的循环。通过将算子上下文和依赖元数据向下传递至 TISA，编译器使运行时能够直接基于语义而不是不透明的指令流做出合法性和重叠决策。反过来，动态调度器将这些语义转化为运行时性能，在不牺牲正确性的情况下实现自适应性。

## VII. 实现

## A. 在 Epoch 上的实现

我们在 CPU 后端和 AI 加速器 Epoch 上均实现了我们的框架。CPU 实现通过提供 TISA 语义算子库作为功能和精度的参考，而 Epoch 后端则通过原生 ISA 支持充分利用了 TISA 的动态调度和异构执行能力。

a) 硬件概述：Epoch 是一款面向吞吐量的 AI 加速器，通过高带宽互连从主机 CPU 卸载计算密集型内核，并共享 48 GB DDR 内存。该芯片已成功在 1 GHz 下流片，目前正在商业化。本文中展示的所有 Epoch 性能结果均是在这块已流片的物理硅片上以 W = 8 测量的。它被组织为通过 32 个核心暴露丰富的 tile 级并行性。每个核心集成了三个专用引擎：用于张量算术的 Matrix Engine (ME)、用于逐元素和归约操作的 Vector Engine (VE)，以及用于 DMA 和异步数据移动的 Data Engine (DE)。这种异构结构与表 I 中总结的架构高度相似，因此，将此框架移植到其他加速器需要向其添加硬件调度器和 TISA。

b) 内存层次结构：每个核心提供 1.5 MB 的本地内存，核心通过片上共享 SRAM 进行通信，从而实现核心间的 tile 重用。片上 NoC 连接整个系统，参数和激活值通过 48 GB 的全局 DDR 内存与主机进行交换。

c) TISA 集成：TISA 指令作为 Epoch 上的软硬件契约，每个核心集成了一个硬件调度器（第五节），该调度器消耗 TISA 描述符并编排异构执行，而无需显式的软件屏障。在 ME 上，我们引入了用于块矩阵/张量算术的自定义张量指令。在 VE 上，我们通过 tile 友好型操作扩展了向量 ISA。在 DE 上，我们公开了 DMA 风格的描述符以支持异步、非阻塞传输。所有组件均遵循 TISA 接口，该接口向调度器传递语义上下文 (OpType)、资源亲和性 (UnitMap) 和 tile 内存描述符 (TileMem)，以进行合法性检查和动态重叠。

d) 内核库与编译器集成：在这些硬件扩展的基础上，我们实现了一个高性能算子库，其中内核以 tile 粒度表示，并通过双缓冲流水线跨

```c
1 // CUDA fa3 pseudocode 1 // TISA fa3 pseudocode
// Load Q K data 2 // Load Q K data
3 tma_load_q ( s_Q ,Q) ; 3 tisa :: load <de >( s_Q ,Q);
4 tma_load_k_transpose (s_K ,K); 4 tisa :: load_transpose <de >( s_K ,K);
warpgroup_fence_producer (); 5 // Matrix multiply (P=Q*K)
6 // Matrix multiply (P=Q*K) 6 tisa :: gemm <me >( s_P ,s_Q ,s_K);
7 wgmma :: mma_sync (s_P ,s_Q , s_K ); 7 // Softmax compute (S= softmax (P))
8 // Softmax compute (S= softmax (P)) 8 tisa :: softmax <ve >( s_S ,s_P , state );
9 wgmma :: wait (); // Wait s_P 9 for ( int j = 0; j < Tc ; j ++) {
10 softmax_warpgroup (s_S ,s_P , state ); 10 if (j < Tc - 1) {
11 for ( int j = 0; j < Tc; j ++) { 11 // Load K data ( next tile )
12 if (j < Tc - 1) { 12 tisa :: load_transpose <de >(
13 // Load K data ( next tile ) s_K_next , K_next );
14 tma_load_k_transpose ( s_K_next , 13 // Matrix multiply ( next tile )
K_next ) ; 14 tisa :: gemm <me >( s_S_next , s_Q ,
15 warpgroup_barrier_arrive (); s_K_next );
16 // Matrix multiply ( next tile ) 15 }
17 wgmma :: mma_async ( s_S_next , s_Q , 16 // Load V data
s_K_next ); 17 tisa :: load <de >( s_V ,V);
18 } 18 // Matrix multiply (R=S*V)
19 // Load V data 19 tisa :: gemm <me >( s_R ,s_S , s_V );
20 tma_load_v (s_V ,V); 20 if (j < Tc - 1) {
warpgroup_barrier_wait (); 21 // Softmax compute ( next tile )
22 // Matrix multiply (R=S*V) 22 tisa :: softmax <ve >( s_S_next ,
23 wgmma :: mma_sync (s_R ,s_S , s_V ); s_S_next , state_next );
24 if (j < Tc - 1) { 23 }
25 // Softmax compute ( next tile ) 24 // Rescale R data (O= Rescale (R))
26 wgmma :: wait (); // Wait s_S_next 25 tisa :: rescale <ve >( s_O ,s_R ,state ,
27 softmax_warpgroup ( s_S_next , state_next );
s_S_next , state_next ); 26 // Update next index
28 } 27 update_next_index ();
29 // Rescale R data (O= Rescale (R)) 28 }
30 wgmma :: wait (); // Wait s_R 29 // Store O data
31 rescale_warpgroup (s_O ,s_R ,state , 30 tisa :: store <de >(O, s_O);
state_next );
32 warpgroup_commit_batch () ;
33 // Update next index
34 update_carousel_index ();
35 }
36 //Store 0 data
37 tma_store_o (O,s_O);
38 warpgroup_epilogue ();
```  
图 5：FlashAttention-3 CUDA 与 TISA 伪代码对比。CUDA 显式管理同步（第 5、9、15、21、26、30、32 和 38 行），而 TISA 通过语义感知的依赖解析消除了所有屏障。

ME/VE/DE。内核是形状参数化的，并直接映射到 OpType 指示的单元，而上游编译器（第六节）自动生成相应的 TISA 指令。这种分离将 TISA 语义与特定于硬件的实现解耦，使得运行时调度可以保持与硬件无关，同时仍然利用每种算子类型的优化内核。

e) 多核执行：对于多核执行，编译器采用空间划分：独立的 tile 组（例如 attention heads、batch dimensions）被静态分配给各个核心。每个核心的本地 TISA 调度器通过其进行中的语义表独立运行。核心间同步使用由共享 SRAM bank 更新触发的轻量级 NoC 信号。运行时负载均衡在内核调用之间的软件层面发生，而不是在 tile 调度器内部。

## B. 案例研究：FlashAttention-3 伪代码

为了展示 TISA 的优势，我们将使用 FlashAttention-3 内核作为说明性示例。图 5 对比了 CUDA 伪代码（左）与我们生成的 TISA Epoch 伪代码（右）。

CUDA 内核将 QK<sup>⊤</sup> 乘法、缩放/掩码、softmax 和 V 投影融合到一个单一内核中。它依赖于手动线程块分解、共享内存暂存、warp 级聚合、显式屏障和手动调优的预取来形成静态流水线。

相比之下，Epoch TISA 内核由编译器自动生成，并完全以 TISA 指令表达。每条指令都标注了 OpType、依赖描述符和资源映射。在运行时，每核心调度器通过评估指令就绪状态来动态编排 ME/VE/DE 的执行，隐式管理核心本地内存层次结构内的同步、双缓冲和异构重叠。

本案例研究突显了从命令式、以屏障为中心的 GPU 编程到声明式、保留语义的 TISA 执行的范式转变。在实践中，GPU 内核对应于图 2(b,d) 中的静态执行模式，而 TISA 内核实现了图 2(c,e) 中的动态重叠模式，从而获得了更高的指令级并行性。定量地说，TISA 内核将代码大小减少了 30%，同步频率降低了 50%，并实现了在手调基准 5% 以内的性能，且所有这些都是由编译器生成的。通过将同步语义直接嵌入到 ISA 中，TISA 消除了手动编排，并实现了由编译器驱动、独立于架构的性能可移植性。

## VIII. 实验验证

我们在四个代表性的深度学习工作负载上评估了我们框架的有效性，包括 ResNet50、BERT-Base、GPT-J-6B、LLaMA2-13B 和 DeepSeek-R1-16B。我们实现了两个后端（CPU 和 Epoch），但在四个硬件平台上进行实验：Epoch(Silicon)、NVIDIA H100、NVIDIA A100 和 Intel Xeon。表 V 总结了平台规格。对于 NVIDIA GPU，我们使用驱动程序 550.127.0、CUDA 12.1、cuDNN 9.1.0、TensorRT 10.3.0（用于 ResNet50/BERT-Base）和 TensorRT-LLM 0.12.0（用于 GPT-J/LLaMA2/DeepSeek-R1）。

表 V：实验平台规格。
<table><tr><td>平台</td><td>Epoch</td><td>H100</td><td>A100</td><td>Xeon 6348</td></tr><tr><td>架构</td><td>X</td><td>Hopper</td><td>Ampere</td><td>x86_64</td></tr><tr><td>核心</td><td>32 Cores</td><td>132 SMs</td><td>108 SMs</td><td>56 Cores</td></tr><tr><td>算力</td><td>256T FP16</td><td>989T FP16</td><td>312T FP16</td><td>9.2T FP32</td></tr><tr><td>内存</td><td>48GB GDDR6</td><td>80GB HBM3</td><td>40GB HBM2</td><td>512GB DDR4</td></tr><tr><td>带宽</td><td>1TB/s</td><td>3.35TB/s</td><td>1.55TB/s</td><td>204.8GB/s</td></tr></table>

## A. Epoch 平台上的结果

我们首先报告在 Epoch 加速器上的结果，其中 TISA 作为标准的软硬件接口。我们比较了三种配置，以分离语义保持和动态调度的效果：（1）Naive TISA，仅将 TISA 作为软硬件接口，并依赖显式 fence 指令手动强制执行单元间的依赖关系，不进行指令重排；（2）TISA + Static，在编译时静态调度 TISA 指令，如同传统编译器流程，同样采用基于 fence 的手动依赖管理，但通过指令重排来改善执行单元间的并行性；以及（3）TISA + Dynamic，由我们的语义感知动态调度器（第五节）调度 TISA 指令。

Naive 采用基本优化，没有硬件介导的跨单元调度。Static 执行多阶段软件流水线（图 2(b,d)），代表针对 Epoch 优化的最佳编译时策略。Dynamic 应用相同的编译器优化，但将执行跟踪委托给硬件调度器，严格隔离了延迟容忍运行时发射的收益。所有配置共享相同的编译器级优化（包括指令级流水线），并使用相同的基础算子库以确保公平比较，仅隔离调度策略的贡献。表 VI 报告了总执行周期数。

表 VI：Epoch 上的执行周期（越小越好）。M：百万周期；N：Naive；S：Static；D：Dynamic。
<table><tr><td>模型</td><td>Naive</td><td>Static</td><td>Dynamic</td><td>S vsN</td><td>D vsN</td><td>D vs S</td></tr><tr><td>ResNet50</td><td>8.98M</td><td>8.72M</td><td>5.92M</td><td>1.03×</td><td>1.52×</td><td>1.47×</td></tr><tr><td>BERT</td><td>13.16M</td><td>11.94M</td><td>7.37M</td><td>1.10×</td><td>1.79×</td><td>1.62×</td></tr><tr><td>GPTJ(oneblk)</td><td>14.44M</td><td>13.54M</td><td>8.30M</td><td>1.07×</td><td>1.74×</td><td>1.63×</td></tr><tr><td>LLaMA2(oneblk)</td><td>25.85M</td><td>15.29M</td><td>13.47M</td><td>1.69×</td><td>1.92×</td><td>1.14×</td></tr></table>

Naive 基线确认了我们的框架能够正确地将端到端工作负载降低为可执行的 TISA 二进制文件，确立了功能正确性。集成静态调度器（TISA+Static）使执行周期减少了 1.03–1.69 倍，这表明 TISA 保留的语义信息已经可以辅助编译时优化。

然而，当与我们的动态调度器（TISA+Dynamic）结合时，收益显著放大，在静态版本的基础上额外产生了 1.14–1.63 倍的加速。总体而言，具有动态调度的 TISA 相比于 naive 基线实现了 1.52–1.92 倍的端到端性能提升。这些结果证实，语义保持在编译时和运行时均能实现优化，并且动态调度可以进一步利用超越静态的运行时可变性。

此外，表 VII 和表 VIII 中的累积重叠分数通过将成对类别（DM、DV、MV、DMV）之间的重叠相加来量化并发密度；由于同时进行的三方单元激活，其有效地超过了总执行周期数。多核结果报告了所有核心的算术平均值。这些表显示，我们的动态调度器始终比静态流水线调度实现更多的跨单元重叠，在运行时根据实际数据依赖性、内存可用性和执行就绪情况调整决策。

虽然静态流水线方法在理论上规划了这些偏移量，但确定最佳边界需要穷举多参数编译时探索，或者通过启发式方法权衡搜索开销，代价是次优性。相比之下，动态调度利用运行时语义来机会性地利用安全并行性，提供了对异构计算单元的卓越利用率。

## B. 与 GPU 基准的比较

1) 端到端执行延迟：虽然第八节A部分中的 Epoch 结果展示了 TISA 和动态调度的内部贡献，但它们并未直接量化相对于最先进 GPU 系统的优势。同时，尽管我们的动态调度器在概念上可以泛化到任何异构加速器，但它无法直接部署在商用 GPU 上。当代 GPU 仅通过固定功能的 warp 调度器暴露粗粒度的 kernel 和 stream 调度，缺乏对算子或 tile 级语义的可见性。它们的同步和依赖机制（例如 CUDA stream 和 barrier）是静态定义的，在运行时无法访问，从而阻碍了动态依赖解析或每单元仲裁。相比之下，我们的 Epoch 平台在 tile 粒度上提供了细粒度控制，并暴露了可编程的发射队列，使其适合实现我们语义感知的运行时调度器。

表VII：引擎上的累计重叠分数（越大越好）。N：Naive；S：Static；D：Dynamic；DM：数据-矩阵引擎重叠；DV：数据-向量引擎重叠；MV：矩阵-向量引擎重叠；DMV：数据-矩阵-向量重叠。由于多个单元同时处于活动状态的单个物理周期会贡献给多个重叠类别，因此累计分数可能超过表VI中报告的总执行周期。
<table><tr><td rowspan="2">模型</td><td colspan="4">Naive</td><td colspan="4">Static</td><td colspan="4">Dynamic</td></tr><tr><td>DM</td><td>DV</td><td>MV</td><td>DMV</td><td>DM</td><td>DV</td><td>MV</td><td>DMV</td><td>DM</td><td>DV</td><td>MV</td><td>DMV</td></tr><tr><td>ResNet50</td><td>723707</td><td>76864</td><td>291879</td><td>0</td><td>1320131</td><td>87160</td><td>321727</td><td>80877</td><td>1727358</td><td>306680</td><td>671217</td><td>130234</td></tr><tr><td>BERT</td><td>24133</td><td>142857</td><td>243664</td><td>0</td><td>959895</td><td>198540</td><td>259273</td><td>33370</td><td>3033524</td><td>1469712</td><td>1867914</td><td>1129011</td></tr><tr><td>GPTJ(oneblk)</td><td>45468</td><td>5581</td><td>84332</td><td>0</td><td>858258</td><td>67693</td><td>84115</td><td>0</td><td>4160114</td><td>190218</td><td>149977</td><td>67171</td></tr><tr><td>LLaMA2(oneblk)</td><td>111659</td><td>187473</td><td>233095</td><td>0</td><td>8472512</td><td>706636</td><td>636877</td><td>450134</td><td>9173782</td><td>811941</td><td>750138</td><td>566249</td></tr></table>

表VIII：Epoch上的累计重叠分数（越大越好）。M：百万周期；N：Naive；S：Static；D：Dynamic。值代表表VII中的DM+DV+MV+DMV。
<table><tr><td>模型</td><td>Naive</td><td>Static</td><td>Dynamic</td><td>S vsN</td><td>D vsN</td><td>D vs S</td></tr><tr><td>ResNet50</td><td>1.09M</td><td>1.81M</td><td>2.84M</td><td>1.66×</td><td>2.60×</td><td>1.57×</td></tr><tr><td>BERT</td><td>0.41M</td><td>1.45M</td><td>7.50M</td><td>3.54×</td><td>18.29×</td><td>5.17×</td></tr><tr><td>GPTJ(oneblk)</td><td>0.14M</td><td>1.01M</td><td>4.57M</td><td>7.21×</td><td>32.64×</td><td>4.52×</td></tr><tr><td>LLaMA2(oneblk)</td><td>0.53M</td><td>10.27M</td><td>11.30M</td><td>19.38×</td><td>21.32×</td><td>1.10×</td></tr></table>

为了建立公平的性能基准，我们将我们在 Epoch 上的 TISA 实现的执行延迟与运行优化 TensorRT 配置的 NVIDIA A100 和 H100 GPU 进行了比较。表IX总结了结果。

表IX：执行延迟比较（越低越好）。
<table><tr><td>模型</td><td>配置</td><td>Epoch</td><td>A100</td><td>加速比</td></tr><tr><td>ResNet50</td><td>FP16, batch=128, 224×224</td><td>6.2ms</td><td>9.3ms</td><td>1.50×</td></tr><tr><td>BERT-Base</td><td>FP16, batch=64, seq=128</td><td>7.5ms</td><td>9.8ms</td><td>1.31×</td></tr><tr><td>GPT-J-6B</td><td>FP16, batch=1, seq=512, prefill</td><td>29.9ms</td><td>37.3ms</td><td>1.25×</td></tr><tr><td>LLaMA2-13B</td><td>FP16, batch=1, seq=512, prefill</td><td>54.0ms</td><td>77.1ms</td><td>1.43×</td></tr><tr><td>DeepSeek-R1-16B</td><td>BF16, batch=50, seq=100, prefill</td><td>213.5ms</td><td>412.3ms</td><td>1.93×</td></tr><tr><td>DeepSeek-R1-16B</td><td>BF16, batch=50, seq=700, decode</td><td>51.2ms</td><td>69.0ms</td><td>1.35×</td></tr></table>

如表V所示，NVIDIA A100 提供的峰值计算吞吐量远高于 Epoch。因此，传统观点认为，A100，尤其是在 TensorRT 的高级静态调度器下，应该优于较小的加速器。尽管存在这种硬件劣势，我们在 Epoch 上的 TISA 框架与 A100 上的 TensorRT 相比，实现了平均 1.46 倍的延迟降低。

这一结果突出了一个关键洞察：动态调度可以解锁甚至最先进的静态 GPU 调度器也无法暴露的运行时并行性。通过响应瞬时硬件条件和类型化依赖就绪状态，我们的运行时更有效地重叠了张量、向量和 DMA tile，从而在峰值 FLOPs 较低的情况下转化为更短的端到端延迟。

2) FlashAttention-3 性能分析：我们进一步在 FlashAttention-3 [33] 上评估了该框架，这是一个手工优化的 GPU kernel，代表了 Transformer Attention 工作负载的最先进水平。该基准测试尤为相关，因为它对 GEMM、softmax 和内存操作之间的跨单元协调提出了严格要求，是评估语义感知动态调度优势的理想测试平台。

我们比较了 Epoch（基于 TISA 的实现）和 NVIDIA H100 上 FlashAttention-3 的持续 BF16 吞吐量，序列长度从 512 到 16K token，包括有因果掩码和无因果掩码的情况。Epoch 在多种向量与矩阵密集计算比率（1:8、1:16、1:32）下进行评估，而 H100 以其原生 1:8 比率运行。结果（硬件利用率 = 峰值 GFLOPs）如图6所示。

![](images/5a3100e384f07c9a9839631370208a96fb4f271542b72d3beb4ba9817517d9d8.jpg)  
图6：Epoch 和 H100 上 FlashAttention-3 的性能比较。硬件利用率是在不同的序列长度和头维度下测量的。左：非因果 Attention；右：因果 Attention。H100 结果来自 FlashAttention-3 [33]，这是一个高度优化的实现。

在匹配的 1:8 BF16 比率下，Epoch 上的 TISA 实现在所有评估的序列长度上都实现了 10% 以上的更高硬件利用率，在主流配置（头维度 128）下利用率高出 26.4%。即使计算比率降至 1:16，Epoch 仍保持 15.7% 的优势，而在 1:32 时，在几种配置下利用率仍与 H100 相当。

尽管 Epoch 的内存带宽显著较低（1.0 TB/s 对比 H100 上的 3.35 TB/s），但仍取得了这些增益，这表明是调度效率的提升而非原始硬件能力推动了性能提升。TISA 运行时有效地跨迭代重叠了 GEMM–Softmax–GEMM tile，而 FlashAttention-3 的静态融合流水线强制执行严格的逐迭代同步。这证实了 TISA 的语义描述符和动态调度器实现了图2(e) 所示的执行模式，实现了当前静态流水线融合 GPU kernel（图2(d)）无法达到的并行性。

虽然 H100 FA3 采用了先进的机制（TMA、WGMMA、bar.sync）来维持 warp 专用流水线 [33]，但其同步仍然是静态固定的。如果 TMA 操作原生停顿，依赖的消费者序列将无条件阻塞。相比之下，TISA 使序列发射适应精确的就绪轨迹，从而在相对于 H100 存在 3.35 倍内存带宽架构劣势的情况下，仍能提供卓越的归一化利用率。TISA 调度原理适用于其他加速器。

这些结果共同表明，TISA 的语义感知动态调度弥合了静态编译器融合与真正的运行时适应性之间的差距。即使运行在具有较低理论 FLOPs 和内存带宽的硬件上，Epoch 与高端 GPU 相比也实现了具有竞争力或更优的性能，这验证了语义引导的运行时调度（而非暴力的计算密度）是实现大规模持续利用率的关键。

## C. 可移植性

基于第八节 A 部分的基础，本节评估了 TISA 在 CPU 上的适用性，在 CPU 上无法进行动态调度。本研究将语义引导编译的益处与运行时调度分离开来，证实 TISA 的设计原则保持架构无关性。即使在同构 CPU 环境中，保留算子语义也能为编译时优化提供信息，从而产生可衡量的性能提升。

我们比较了两种 CPU 实现：（1）Torch-Manual，它使用标准的 PyTorch 算子组合，没有语义引导，代表传统的基于框架的执行；（2）Triton-TISA，它引入了 TISA 的语义抽象层，为内核融合、循环排序和内存局部性等编译时优化提供信息。这两种实现都依赖于相同的 PyTorch 算子库，确保了公平的比较，从而分离出语义引导的贡献。我们在层和全模型粒度上测量了三个代表性工作负载（ResNet50、BERT 和 LLaMA2）的执行时间。对于仅解码器的 Transformer 架构，我们选择 LLaMA2 作为代表性工作负载，因为其架构优化（例如 RMSNorm、SwiGLU 激活和旋转位置 Embedding）使其比 GPT-J 更高效，同时保持了相当的模型复杂度。图 7 报告了这两种实现的执行时间。

![](images/14dd75c5b7b39e2d5713cbe0a8f7079c80608fdf3b067de3e4e45b44a5b9524e.jpg)  
图 7：Triton-TISA 和 Torch-Manual 之间的执行时间（以毫秒为单位）比较。

在所有工作负载中，TISA 引导的编译实现了一致的改进：ResNet50 比 Torch-Manual 获得了 1.13–1.19× 的加速，BERT 获得了 1.02–1.20× 的加速，LLaMA2 获得了 1.14–1.18× 的加速。这些结果证实，即使在没有运行时调度的情况下，TISA 的语义信息也能有效地引导编译时优化。

这一结果与 GPU 端的分块编程框架（如 Triton [40]、TileLang [43] 和 ThunderKittens [34]）产生了共鸣，这些框架使用分块抽象来提高效率和可编程性。在 CPU 上，TISA 发挥了类似的作用，利用语义为编译器的优化阶段提供信息，从而通过结构化的、分块级别的推理而非专门的硬件特性来提高性能。

## IX. 相关工作

本节将我们的框架置于AI加速器研究的更广阔背景中。我们分析三个关键领域：指令级并行、动态调度方法和语义感知编译框架，特别关注区分TISA与先前工作的语义粒度和运行时调度能力。

经典的ILP调度器，如Tomasulo [41]和记分板（scoreboarding）[39]，在CPU中实现动态调度，但依赖寄存器级依赖跟踪，对AI加速器而言不够充分。近期针对AI加速器的ILP方法，如Mosaic [44]、LLVA [2]、SISA [4]、TPP [14]和Cambricon [9]，通过可重构架构和手动优化实现了有限的改进，但本质上仍是静态的。我们的框架不同之处在于，通过语义感知的指令格式实现动态tile调度，保留算子上下文、类型化依赖和资源需求，以供运行时决策。

现有的动态调度方法存在根本性局限：GPU硬件调度（warp/CTA）[21]在线程级运行，跨单元语义可见性有限，而基于任务的张量系统[45]使用编译时静态方案来弥补硬件无法动态执行跨单元调度的不足。这种静态方法本质上限制了运行时适应性。

模调度（Modulo scheduling）[32]在可预测的CPU架构上通过硬件谓词[5]优雅地重叠固定循环组件。TISA应对的是现代多引擎AI加速器固有的非确定性延迟约束，在这种场景下纯静态方法不可避免地会失效。

Task Superscalar [12]在同构CPU核心上实现粗粒度、依赖驱动的执行，作为硬件乱序任务流水线。TISA面向不同的目标和场景：AI加速器，其中tile级ISA让硬件在依赖就绪时自动调度异构张量、向量和DMA引擎间的并发工作。我们的工作通过协同设计的指令格式、语义接口和硬件调度器实现运行时跨单元协调，在指令粒度上运行并保留算子级语义上下文。

我们的TISA抽象在抽象粒度上与NVIDIA PTX [29]等虚拟ISA存在根本差异。PTX在细粒度指令级运行（如add.f32、ld.global），通过驱动程序翻译到原生ISA来抽象指令级以下的硬件差异。相比之下，TISA在tile级运行，每条指令代表语义丰富的计算tile（如`tisa::gemm<me>`、`tisa::softmax<ve>`），保留算子语义、依赖和资源需求。这使得我们的工作能够保留细粒度ISA必然抹除的调度关键语义，从而使动态跨算子调度成为可能。

领域特定ISA（Cambricon [9]、TPU [20]、IPU [19]）指定什么在哪里运行；跨tile排序通过栅栏或BSP式屏障强制执行。cuTile [30]同样是编译时固定的：屏障（如`bar.sync`）固定发射顺序，因此硬件无法根据就绪状态重排。TISA增加了调度器消费的调度语义，在依赖清除时发射tile，而不仅在静态屏障位置。

作为总结，表X在影响调度能力的维度上将TISA与现有方法进行比较。

表X：TISA在ILP调度领域中的定位。

| 方法 | 语义位于 | 调度位于 | 运行时 | 跨单元 | 硬件需求 |
|---|---|---|---|---|---|
| CPU | 寄存器 | 指令 | 是 | 单单元 | 复杂逻辑 |
| Triton/TileLang | Tile | Warp/CTA | 否 | 否 | 最小 |
| ThunderKittens | Tile | Warp/CTA | 否 | 否 | 最小 |
| TensorIR/TVM | 算子 | 线程 | 否 | 否 | 最小 |
| MLIR HLO | 算子 | Kernel | 否 | 否 | 最小 |
| GPU Sched | 线程 | Warp/CTA | 是 | 有限 | 硬件调度器 |
| TISA | Tile | 指令 | 是 | 是 | 硬件调度器 |

我们将现有方法分为两类：（1）粗粒度编程方法如Triton [40]、Tile-Lang [43]和ThunderKittens [34]保留语义但缺乏运行时适应能力，（2）细粒度硬件方法（CPU或GPU warp调度器）提供运行时适应能力但跨单元协调的语义上下文不足。我们的框架独特地将tile级语义保留与指令粒度运行时调度相结合，实现了现有方法因语义局限或粒度不匹配而无法达成的跨异构单元协调。

此外，我们框架的语义接口和调度器可适配到其他架构。在NVIDIA GPU上，语义感知协调器可以位于warp/CTA调度器之上，管理跨单元依赖而无需改变warp调度。对于领域特定加速器，TISA可以作为原生ISA之上的薄硬件封装：原生每单元指令保持不变运行，而TISA为动态调度提供语义。

## X. 结论

我们通过协同设计三个暴露运行时可见调度语义的组件来解决根源于静态调度的利用率不足问题：tile级指令集（TISA，第IV节）、语义感知动态tile调度器（第V节）和语义保留编译器（第VI节）。这使得算子/tile边界、带就绪状态的类型化依赖、资源意图和tile内存范围在运行时可被消费，用于跨异构单元的重叠和重排。

在Epoch、NVIDIA H100和A100 GPU上，该方法在ResNet50、BERT、GPT-J、LLaMA2和DeepSeek-R1上相比基线实现了1.52–1.92×的性能提升；在FlashAttention-3上，在head dim为128时实现了约26.4%的矩阵单元利用率提升。这些结果表明，恢复调度语义能够在运行时实现超越固定模板静态方法的跨算子和跨迭代重叠（第IX节），并且将受限的调度片段移至运行时可提高可用性：运行时可见的语义消除了大部分手动屏障和跨单元流水线，而调度器吸收硬件/工作负载漂移，从而简化跨加速器的重调和移植。

我们的方法引入了适度的每核心硬件开销：专用调度器占用少量硅片面积，这反映了调度逻辑与额外计算单元之间的刻意权衡，以换取更高的利用率和更简单的软件。端到端性能还取决于底层tile级算子库的质量，这可能限制动态调度的实际收益。尽管如此，鉴于所展示的显著利用率提升和编程简便性，这些权衡是可接受的。

我们的框架有两个局限。首先，tile级粒度对于需要子tile控制的操作（如不规则稀疏模式）可能不够充分，在这种情况下执行可能回退到原生每单元指令。其次，跨单元重叠极少的纯内存受限工作负载获益有限。我们计划扩展TileMem对步幅访问的支持以及调度器中的依赖分析，扩大工作负载覆盖范围，并将该工作扩展到训练和分布式场景，以及与新兴编译器栈和自动调优框架的更紧密集成。

## XI. 致谢

我们感谢审稿人提出的建设性反馈，以及合著者所做的贡献和讨论。本工作部分得到了国家自然科学基金项目编号 T2422007 和 U24A20235 的资助。

[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. J. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz,´ L. Kaiser, M. Kudlur, J. Levenberg, D. Mane, R. Monga, S. Moore,´ D. G. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. A. Tucker, V. Vanhoucke, V. Vasudevan, F. B. Viegas,´ O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng, “Tensorflow: Large-scale machine learning on heterogeneous distributed systems,” CoRR, vol. abs/1603.04467, 2016. [Online]. Available: http://arxiv.org/abs/1603.04467

[2] V. S. Adve, C. Lattner, M. Brukman, A. Shukla, and B. Gaeke, “LLVA: A low-level virtual instruction set architecture,” in Proceedings of the 36th Annual International Symposium on Microarchitecture, San Diego, CA, USA, December 3-5, 2003. IEEE Computer Society, 2003, pp. 205– 216. [Online]. Available: https://doi.org/10.1109/MICRO.2003.1253196

[3] J. Ansel, E. Z. Yang, H. He, N. Gimelshein, A. Jain, M. Voznesensky, B. Bao, P. Bell, D. Berard, E. Burovski, G. Chauhan, A. Chourdia, W. Constable, A. Desmaison, Z. DeVito, E. Ellison, W. Feng, J. Gong, M. Gschwind, B. Hirsh, S. Huang, K. Kalambarkar, L. Kirsch, M. Lazos, M. Lezcano, Y. Liang, J. Liang, Y. Lu, C. K. Luk, B. Maher, Y. Pan, C. Puhrsch, M. Reso, M. Saroufim, M. Y. Siraichi, H. Suk, S. Zhang, M. Suo, P. Tillet, X. Zhao, E. Wang, K. Zhou, R. Zou, X. Wang, A. Mathews, W. Wen, G. Chanan, P. Wu, and S. Chintala, “Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation,” in Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2024, La Jolla, CA, USA, 27 April 2024- 1 May 2024, R. Gupta, N. B. Abu-Ghazaleh, M. Musuvathi, and D. Tsafrir, Eds. ACM, 2024, pp. 929–947. [Online]. Available: https://doi.org/10.1145/3620665.3640366

[4] M. Besta, R. Kanakagiri, G. Kwasniewski, R. Ausavarungnirun, J. Beranek, K. Kanellopoulos, K. Janda, Z. Vonarburg-Shmaria,´ L. Gianinazzi, I. Stefan, J. Gomez-Luna, J. Golinowski, M. Copik,´ L. Kapp-Schwoerer, S. D. Girolamo, N. Blach, M. Konieczny, O. Mutlu, and T. Hoefler, “SISA: set-centric instruction set architecture for graph mining on processing-in-memory systems,” in MICRO ’21: 54th Annual IEEE/ACM International Symposium on Microarchitecture Virtual Event, Greece, October 18-22, 2021. ACM, 2021, pp. 282–297. [Online]. Available: https://doi.org/10.1145/3466752.3480133

[5] J. Bharadwaj, W. Y. Chen, W. Chuang, G. Hoflehner, K. N. Menezes, K. Muthukumar, and J. Pierce, “The intel IA-64 compiler code generator,” IEEE Micro, vol. 20, no. 5, pp. 44–53, 2000. [Online]. Available: https://doi.org/10.1109/40.877949

[6] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang, “JAX: composable transformations of Python+NumPy programs,” 2018. [Online]. Available: http://github.com/jax-ml/jax

[7] C. Chen, X. Xiang, C. Liu, Y. Shang, R. Guo, D. Liu, Y. Lu, Z. Hao, J. Luo, Z. Chen, C. Li, Y. Pu, J. Meng, X. Yan, Y. Xie, and X. Qi, “Xuantie-910: Innovating cloud and edge computing by RISC-V,” in IEEE Hot Chips 32 Symposium, HCS 2020, Palo Alto, CA, USA, August 16-18, 2020. IEEE, 2020, pp. 1–19. [Online]. Available: https://doi.org/10.1109/HCS49909.2020.9220630

[8] Y. Chen, J. S. Emer, and V. Sze, “Eyeriss: A spatial architecture for energy-efficient dataflow for convolutional neural networks,” in 43rd ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2016, Seoul, South Korea, June 18-22, 2016. IEEE Computer Society, 2016, pp. 367–379. [Online]. Available: https://doi.org/10.1109/ISCA.2016.40

[9] Y. Chen, H. Lan, Z. Du, S. Liu, J. Tao, D. Han, T. Luo, Q. Guo, L. Li, Y. Xie, and T. Chen, “An instruction set architecture for machine learning,” ACM Trans. Comput. Syst., vol. 36, no. 3, pp. 9:1–9:35, 2018. [Online]. Available: https://doi.org/10.1145/3331469

[10] J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), J. Burstein, C. Doran, and T. Solorio, Eds. Association for Computational Linguistics, 2019, pp. 4171–4186. [Online]. Available: https://doi.org/10.18653/v1/n19-1423

[11] M. Elgamal, D. Carmean, E. Ansari, O. Zed, R. Peri, S. Manne, U. Gupta, G. Wei, D. Brooks, G. Hills, and C. Wu, “Carbon-efficient design optimization for computing systems,” in Proceedings of the 2nd Workshop on Sustainable Computer Systems, HotCarbon 2023, Boston, MA, USA, 9 July 2023, G. Porter, T. Anderson, A. A. Chien, T. Eilam, C. Josephson, and J. Park, Eds. ACM, 2023, pp. 16:1–16:7. [Online]. Available: https://doi.org/10.1145/3604930.3605712

[12] Y. Etsion, F. Cabarcas, A. Rico, A. Ram´ırez, R. M. Badia, E. Ayguade,´ J. Labarta, and M. Valero, “Task superscalar: An out-of-order task pipeline,” in 43rd Annual IEEE/ACM International Symposium on Microarchitecture, MICRO 2010, 4-8 December 2010, Atlanta, Georgia, USA. IEEE Computer Society, 2010, pp. 89–100. [Online]. Available: https://doi.org/10.1109/MICRO.2010.13

[13] J. A. Fisher, “The VLIW machine: A multiprocessor for compiling scientific code,” Computer, vol. 17, no. 7, pp. 45–53, 1984. [Online]. Available: https://doi.org/10.1109/MC.1984.1659185

[14] E. Georganas, D. D. Kalamkar, S. Avancha, M. Adelman, C. Anderson, A. Breuer, J. Bruestle, N. Chaudhary, A. Kundu, D. Kutnick, F. Laub, M. Vasimuddin, S. Misra, R. Mohanty, H. Pabst, B. Ziv, and A. Heinecke, “Tensor processing primitives: a programming abstraction for efficiency and portability in deep learning workloads,” in International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2021, St. Louis, Missouri, USA, November 14-19, 2021, B. R. de Supinski, M. W. Hall, and T. Gamblin, Eds. ACM, 2021, p. 14. [Online]. Available: https://doi.org/10.1145/3458817.3476206

[15] R. Ghanbari, H. Kao, J. P. L. de Carvalho, E. Amiri, and J. N. Amaral, “Scalar interpolation: A better balance between vector and scalar execution for superscalar architectures,” in Proceedings of the 23rd ACM/IEEE International Symposium on Code Generation and Optimization, CGO 2025, Las Vegas, NV, USA, March 1-5, 2025, J. Doerfert, T. Grosser, H. Leather, and P. Sadayappan, Eds. ACM, 2025, pp. 77–89. [Online]. Available: https://doi.org/10.1145/3696443. 3708950

[16] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Xu, H. Ding, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Chen, J. Yuan, J. Tu, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. You, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Zhou, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang, “Deepseek-r1 incentivizes reasoning in llms through reinforcement learning,” Nature, vol. 645, no. 8081, p. 633–638, Sep. 2025. [Online]. Available: http://dx.doi.org/10.1038/s41586-025-09422-z

[17] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. IEEE Computer Society, 2016, pp. 770–778. [Online]. Available: https://doi.org/10.1109/CVPR.2016.90

[18] Z. Hua, F. Qi, G. Liu, and S. Yang, “Learning to schedule DAG tasks,” CoRR, vol. abs/2103.03412, 2021. [Online]. Available: https://arxiv.org/abs/2103.03412

[19] Z. Jia, B. Tillman, M. Maggioni, and D. P. Scarpazza, “Dissecting the Graphcore IPU architecture via microbenchmarking,” CoRR, vol. abs/1912.03413, 2019. [Online]. Available: https://arxiv.org/abs/1912. 03413

[20] N. P. Jouppi, C. Young, N. Patil, D. A. Patterson, G. Agrawal, R. Bajwa, S. Bates, S. Bhatia, N. Boden, A. Borchers, R. Boyle, P. Cantin, C. Chao, C. Clark, J. Coriell, M. Daley, M. Dau, J. Dean, B. Gelb, T. V. Ghaemmaghami, R. Gottipati, W. Gulland, R. Hagmann, C. R. Ho, D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz, A. Jaffey, A. Jaworski, A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar, S. Lacy, J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin, G. MacKean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan, R. Narayanaswami, R. Ni, K. Nix, T. Norrie, M. Omernick, N. Penukonda, A. Phelps, J. Ross, M. Ross, A. Salek, E. Samadiani, C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Steinberg, A. Swing, M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan, R. Walter, W. Wang, E. Wilcox, and D. H. Yoon, “In-datacenter performance analysis of a tensor processing unit,” in Proceedings of the 44th Annual International Symposium on Computer Architecture, ISCA 2017, Toronto, ON, Canada, June 24-28, 2017. ACM, 2017, pp. 1–12. [Online]. Available: https://doi.org/10.1145/3079856.3080246

[21] M. Khairy, A. G. Wassal, and M. Zahran, “A survey of architectural approaches for improving GPGPU performance, programmability and heterogeneity,” J. Parallel Distributed Comput., vol. 127, pp. 65–88, 2019. [Online]. Available: https://doi.org/10.1016/j.jpdc.2018.11.012

[22] H. Kwon, A. Samajdar, and T. Krishna, “MAERI: enabling flexible dataflow mapping over DNN accelerators via reconfigurable interconnects,” in Proceedings of the Twenty-Third International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS 2018, Williamsburg, VA, USA, March 24-28, 2018, X. Shen, J. Tuck, R. Bianchini, and V. Sarkar, Eds. ACM, 2018, pp. 461–475. [Online]. Available: https://doi.org/10.1145/ 3173162.3173176

[23] M. S. Lam, “Software pipelining: An effective scheduling technique for VLIW machines,” in PLDI. ACM, 1988, pp. 318–328.

[24] C. Lattner, M. Amini, U. Bondhugula, A. Cohen, A. Davis, J. A. Pienaar, R. Riddle, T. Shpeisman, N. Vasilache, and O. Zinenko, “MLIR: scaling compiler infrastructure for domain specific computation,” in IEEE/ACM International Symposium on Code Generation and Optimization, CGO 2021, Seoul, South Korea, February 27 - March 3, 2021, J. W. Lee, M. L. Soffa, and A. Zaks, Eds. IEEE, 2021, pp. 2–14. [Online]. Available: https://doi.org/10.1109/CGO51591.2021.9370308

[25] H. Liao, J. Tu, J. Xia, and X. Zhou, “Davinci: A scalable architecture for neural network computing,” in 2019 IEEE Hot Chips 31 Symposium (HCS), Cupertino, CA, USA, August 18-20, 2019. IEEE, 2019, pp. 1–44. [Online]. Available: https://doi.org/10.1109/HOTCHIPS.2019.8875654

[26] M. Naumov, D. Mudigere, H. M. Shi, J. Huang, N. Sundaraman, J. Park, X. Wang, U. Gupta, C. Wu, A. G. Azzolini, D. Dzhulgakov, A. Mallevich, I. Cherniavskii, Y. Lu, R. Krishnamoorthi, A. Yu, V. Kondratenko, S. Pereira, X. Chen, W. Chen, V. Rao, B. Jia, L. Xiong, and M. Smelyanskiy, “Deep learning recommendation model for personalization and recommendation systems,” CoRR, vol. abs/1906.00091, 2019. [Online]. Available: http://arxiv.org/abs/1906. 00091

[27] NVIDIA, NVIDIA CUDA C++ Programming Guide, NVIDIA Corporation, 2007. [Online]. Available: https://docs.nvidia.com/cuda/ cuda-c-programming-guide/index.html

[28] ——, NVIDIA TensorRT Documentation, NVIDIA Corporation, 2021. [Online]. Available: https://docs.nvidia.com/deeplearning/tensorrt/latest/ index.html

[29] NVIDIA Parallel Thread Execution ISA, NVIDIA Corporation, 2025. [Online]. Available: https://docs.nvidia.com/cuda/ parallel-thread-execution/index.html

[30] ——, NVIDIA CUDA Tile, NVIDIA Corporation, 2026. [Online]. Available: https://developer.nvidia.com/cuda/tile

[31] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 2023, pp. 28 492–28 518. [Online]. Available: https://proceedings.mlr. press/v202/radford23a.htm

[32] B. R. Rau, “Iterative modulo scheduling: An algorithm for software pipelining loops,” pp. 63–74, 1994.

[33] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao, “Flashattention-3: Fast and accurate attention with asynchrony and low-precision,” in Advances in Neural Information

Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, A. Globersons, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. M. Tomczak, and C. Zhang, Eds., 2024. [Online]. Available: http://papers.nips.cc/paper files/paper/2024/hash/ 7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html

[34] B. Spector, S. Arora, A. Singhal, D. Y. Fu, and C. Re, “Thunderkittens:´ Simple, fast, and adorable AI kernels,” CoRR, vol. abs/2410.20399, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2410.20399

[35] V. Sze, Y. Chen, T. Yang, and J. S. Emer, “Efficient processing of deep neural networks: A tutorial and survey,” Proc. IEEE, vol. 105, no. 12, pp. 2295–2329, 2017. [Online]. Available: https://doi.org/10.1109/JPROC.2017.2761740

[36] TensorFlow Team, “Xla: Optimizing compiler for machine learning,” https://www.tensorflow.org/xla, 2017, accessed: 2025-08-01.

[37] Tenstorrent Inc., “Grayskull high performance ai processor,” 2020, aI processor launched in 2020, technical details available at company website. [Online]. Available: https://tenstorrent.com/

[38] ——, “Wormhole: Next-generation ai processor architecture,” 2022, tenstorrent Wormhole AI processor architecture. [Online]. Available: https://tenstorrent.com/hardware/wormhole

[39] J. THORNTON, “Design of a computer-the control data 6600,” Glenview, IL: Scott, Foresman, 1970.

[40] P. Tillet, H. Kung, and D. D. Cox, “Triton: an intermediate language and compiler for tiled neural network computations,” in Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, MAPL@PLDI 2019, Phoenix, AZ, USA, June 22, 2019, T. Mattson, A. Muzahid, and A. Solar-Lezama, Eds. ACM, 2019, pp. 10–19. [Online]. Available: https://doi.org/10.1145/3315508.3329973

[41] R. M. Tomasulo, “An efficient algorithm for exploiting multiple arithmetic units,” IBM Journal of research and Development, vol. 11, no. 1, pp. 25–33, 1967.

[42] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom, “Llama 2: Open foundation and fine-tuned chat models,” CoRR, vol. abs/2307.09288, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2307.09288

[43] L. Wang, Y. Cheng, Y. Shi, Z. Tang, Z. Mo, W. Xie, L. Ma, Y. Xia, J. Xue, F. Yang, and Z. Yang, “Tilelang: A composable tiled programming model for AI systems,” CoRR, vol. abs/2504.17577, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2504.17577

[44] J. Xu, Y. Wen, Z. Liu, R. Xu, T. Ruan, J. Bi, R. Zhang, D. Huang, X. Song, Y. Hao, X. Hu, Z. Du, C. Zhao, J. Jie, and Q. Guo, “Mosaic: Exploiting instruction-level parallelism on deep learning accelerators with iTex tessellation,” in Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2025, Rotterdam, Netherlands, 30 March 2025 - 3 April 2025, L. Eeckhout, G. Smaragdakis, K. Liang, A. Sampson, M. A. Kim, and C. J. Rossbach, Eds. ACM, 2025, pp. 672–688. [Online]. Available: https://doi.org/10.1145/3676641.3716262

[45] R. Yadav, M. Garland, A. Aiken, and M. Bauer, “Task-based tensor computations on modern gpus,” pp. 396–420, 2025. [Online]. Available: https://doi.org/10.1145/3729262

[46] Y. Yu, W. Xiao, X. He, H. Guo, Y. Wang, and X. Chen, “A stall-aware warp scheduling for dynamically optimizing thread-level parallelism in gpgpus,” in Proceedings of the 29th ACM on International Conference on Supercomputing, ICS’15, Newport Beach/Irvine, CA, USA, June 08 - 11, 2015, L. N. Bhuyan, F. Chong, and V. Sarkar, Eds. ACM, 2015, pp. 15–24. [Online]. Available: https://doi.org/10.1145/2751205.2751234