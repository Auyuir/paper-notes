# PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction 原文翻译

# PIPEWEAVE：协同解析模型与学习模型以实现统一的GPU性能预测

Kaixuan Zhang<sup>1,3</sup>, Yunfan Cui<sup>1</sup>, Shuhao Zhang<sup>1</sup>, Chutong Ding<sup>1</sup>, Shiyou Qian<sup>1,\*</sup>, Luping Wang<sup>2,\*</sup>, Jian Cao<sup>1</sup>, Guangtao Xue<sup>1</sup>, Cheng Huang<sup>2</sup>, Guodong Yang<sup>2</sup>, and Liping Zhang<sup>2</sup>

<sup>1</sup>上海交通大学，中国 <sup>2</sup>阿里巴巴集团，中国 <sup>\*</sup>通讯作者

<sup>3</sup>在阿里巴巴集团实习期间完成的工作

{zks1anx, cuiyunfan, zhang-shuhao, qshiyou}@sjtu.edu.cn

摘要——基于Transformer的大语言模型的快速发展极大地增加了对高性能GPU的需求。因此，对快速、准确且具有广泛泛化能力的GPU性能模型的需求日益增长，以支持下一代硬件选型和系统级探索。然而，当前的数据驱动方法存在局限性，在不同硬件间的泛化能力较差，且对现代推理栈中常见的复杂生产级内核建模不足。为解决这些问题，我们提出了PIPEWEAVE，一个统一的GPU建模框架。该方法首先采用解析模型来量化给定内核对GPU异构指令流水线的需求。然后将这些解析特征输入到机器学习（ML）模型中，以捕获复杂的跨流水线交互和资源依赖关系，从而实现高保真的性能预测。我们在两个广泛使用的服务系统上对来自四大主要架构代际的11种GPU类型进行了评估，结果表明PIPEWEAVE具有高保真度和强泛化能力。它实现了准确的预测，内核级平均误差仅为6.1%，端到端推理为8.5%——分别将最先进方法的误差降低了6.7倍和4.4倍。我们还展示了PIPEWEAVE"超越仿真"的价值，利用其性能上限来诊断实现缺陷并指导一个生产级融合MoE Triton内核的优化，实现了高达1.7倍的加速。代码可在 https://github.com/zksainx/pipeweave 获取。

## I. 引言

基于Transformer [70]的大语言模型（LLM）的出现从根本上重塑了人工智能的格局。如今，大量的LLM——从专有旗舰模型如Gemini [16]到开源系列如Llama [67]和Qwen [3]——被部署以支持从编程助手到文档摘要等各类服务。在异构硬件平台上高效地服务这些多样化工作负载，需要大规模的高性能节点集群。

性能需求的激增伴随着主要厂商如NVIDIA和AMD以快速创新的硬件迭代来满足，它们频繁发布具有重大性能和特性更新的新GPU架构。例如，自Ampere架构推出以来，NVIDIA已发布了四代不同的架构以及数十种面向不同市场细分的非消费级型号变体[39]。模型与硬件的这种快速协同演进给系统设计师和架构师带来了关键挑战。海量的硬件配置使得穷举测试变得不切实际，而无法获取每一种可能的配置——甚至无法访问未发布的下一代硬件——进一步加剧了这一问题。因此，为支持大规模系统设计、硬件选型和下一代系统的开发，对快速、准确且具泛化能力的GPU性能模型的需求前所未有地迫切。

从历史上看，GPU性能建模主要遵循三种范式，每种范式在保真度、速度和通用性之间展现出不同的权衡。第一，周期精确模拟器[4], [22], [66]通过详细模拟微架构行为提供最高保真度，但其仿真速度计算代价高昂，且缺乏可移植性使其难以泛化到新的或未公开的硬件。第二，解析模型[6], [19], [25]通过依赖如区间分析等性能公式提供更快的估算，但其准确性往往受限，且对硬件特定微基准测试的依赖限制了其对未见架构的泛化能力。第三，数据驱动方法[26], [76]通过从测量中学习分块级延迟实现高速度，但其预测准确性可能不稳定，且其高层建模假设——将分块视为原子单元、假设SM行为均匀、未能充分捕获融合内核耦合（如FlashAttention [11]）——可能影响跨工作负载和硬件代际的泛化能力。

为弥合这一差距，我们提出了PIPEWEAVE，一个通过结合解析-ML设计将流水线级分析编织为准确预测，从而实现高保真、快速和广泛泛化能力的GPU性能建模框架。该框架首先将给定内核分解为一组基本任务，每个任务代表一个Streaming Multiprocessor（SM）可调度的工作单元。然后模拟这些任务如何根据内核的执行范式映射到SM上，产生真实的任务分布。基于此分布，PIPEWEAVE解析地推导每个任务对SM异构指令流水线的流水线需求及相关的理论周期数，并将它们聚合成一个紧凑的多级特征集。最后，一个轻量级MLP消费这些特征来预测内核的执行时长。

我们进行了广泛的评估来验证我们的框架。我们的实验测试平台涵盖4个硬件代际，包含11种不同的GPU类型（6种用于训练，5种用于未见测试），以及vLLM [72]和SGLang [59]等框架常用的5类关键内核（如GEMM、Attention），精度涵盖FP8、BF16/FP16和FP32。在内核级别，PIPEWEAVE在已见GPU上实现了6.1%的低平均MAPE，在未见GPU上为11.4%，大幅超越最先进（SOTA）基线方法Neusight [26]，误差分别降低了6.7倍和3.8倍。我们进一步在复杂推理工作负载上验证了我们的模型，使用了三个大模型（Qwen2.5-14B、Qwen3-32B、Llama3.1-70B）以及各种Tensor和Pipeline Parallel（TP/PP）配置。在这些端到端场景中，PIPEWEAVE保持了高保真度，在已见GPU上平均误差为8.5%，在未见GPU上为10.7%——将Neusight的预测误差分别降低了4.4倍和3.1倍。最后，我们展示了PIPEWEAVE"超越仿真"的价值。通过利用该模型建立潜在的性能上限，我们识别了一个生产级融合MoE Triton内核中特定于硬件的实现低效问题，并指导了有针对性的优化，实现了高达1.7倍的加速。

总结而言，本文做出以下贡献。

• 统一建模框架：我们提出了PipeWeave，一个协同解析建模与机器学习的统一框架，以准确捕获复杂的流水线交互，实现高保真预测。

• 卓越的泛化能力：在跨越四代架构的11种GPU上得到验证，PIPEWEAVE在未见架构上实现了SOTA准确性，预测误差较先前方法降低高达6.7倍。

• 优化指导：我们通过建立性能上限来诊断实现低效问题并指导生产级内核的有针对性优化，展示了"超越仿真"的实用价值。

## II. 背景

## A. 大型语言模型

现代 LLM 主要基于 Transformer 架构 [70]。典型的 Transformer 模块包含两个关键组件：多头注意力机制和位置式前馈网络（FFN）。LLM 推理通常分为两个阶段：预填充和解码 [2], [59], [72]。在预填充阶段，并行处理输入提示，并为每个 Token 计算和存储键/值对到 KV 缓存中。解码阶段则以自回归方式生成输出 Token。

表 I 展示了在带有张量并行（TP=4）的 4×A100 集群上（批次大小为 8，序列长度为 8192）进行 Qwen2.5-32B 推理的内核（GPU 可执行函数）运行时分布。这些类别对应于当今主流分布式 LLM 中使用的 Transformer 架构的核心计算构建组件和通信原语：GEMM 内核 [2] 占据了主要工作负载，这源于注意力层和前馈层中的线性投影；Attention 内核计算 Token 之间的关系；RMSNorm 内核 [77] 在注意力和前馈计算之前稳定激活值；诸如 SiLU&Mul 等操作实现激活函数和逐元素计算，这是许多 LLM 中使用的 SwiGLU FFN 的核心 [61]；而 All-Reduce 内核处理跨 GPU 的必要集合通信。这些主要内核合计占据了总运行时的绝大部分。由于它们的优势在各种模型、软件栈和硬件代际中持续存在 [2], [11], [64]，我们的分析侧重于准确建模这些内核。

表 I
在带有张量并行（TP=4）的 4×A100 集群上 QWEN2.5-32B 推理的运行时分解。
<table><tr><td>阶段</td><td>GEMM</td><td>Attention</td><td>RMSNorm</td><td>SiLU&amp;Mul</td><td>All-Reduce</td><td>其他</td></tr><tr><td>预填充</td><td>72.70%</td><td>8.22%</td><td>3.85%</td><td>2.26%</td><td>12.10%</td><td>0.87%</td></tr><tr><td>解码</td><td>65.05%</td><td>17.78%</td><td>3.19%</td><td>1.50%</td><td>5.76%</td><td>6.72%</td></tr></table>

此外，LLM 优化的最新进展促使了专用内核的采用。低精度数据类型的使用，特别是用于 W8A8（8位权重和激活）推理的 FP8 量化 [24], [63]，使缩放矩阵乘法内核流行起来 [40], [41], [45]。同时，混合专家架构 [9], [13], [62] 利用了融合 MoE 内核。这些内核在 Token 路由完成后，跨专家子网络高效执行批处理 GEMM 操作。

## B. 从内核到 GPU 架构

不失一般性，我们的讨论在 NVIDIA GPU 架构的背景下进行，主要关注 LLM 推理场景。为了分析此上下文中的性能，我们区分了两个紧密相关的视角：软件视角，涉及如何定义和启动内核；以及硬件视角，描述如何将内核映射到底层微架构并执行，如图 1 所示。

内核执行涉及两个概念阶段。编译阶段以 SASS 指令的形式生成 GPU 可执行代码，这是 NVIDIA SM 唯一可以原生执行的表示。在实践中，编译工具链可能会发出一个胖二进制文件，其中包含特定于架构的 SASS [33] 和虚拟指令集架构（ISA）（PTX [54]）；CUDA 驱动程序在必要时可能会进一步将 PTX JIT 编译为 SASS。无论内核是源自 Triton、CUDA C++，还是诸如 cuBLAS 等预编译库，其最终执行始终解析为与目标 GPU 微架构兼容的 SASS 指令 [48], [68]。

运行时阶段负责启动已编译的内核。内核启动指定了 CUDA 执行配置（Grid 及其协作线程阵列）[49]。

![](images/5f1447795b74a3a76b7f2a834537561b3e4a0c4664cee9e861fbf2c3f3db4e89.jpg)
图 1. 软件层次结构与物理 GPU 硬件层次结构之间映射的示意图。

GPU 硬件工作分配逻辑将 CTA 调度到 SM 上，以 CTA 为单位动态分派工作，而不是一次性映射整个 Grid [34], [47]。在 SM 级别，编译生成的 SASS 指令通过指令缓存层次结构获取，并由 Warp 调度器发出到 SM 的各种执行流水线。

Ampere [36] 及后续架构中的 SM 通常被划分为四个 SM 子分区和 一个内存 I/O (MIO) 单元。每个 SMSP 主要包括一个用于逐周期指令分派的 Warp 调度器、一个寄存器堆以及一组专用数学流水线。典型的例子包括固定延迟数学流水线，如 FMA（处理大多数 FP32 算术运算，如 FMUL、FADD）和 Tensor（执行 MMA 指令，如 HMMA），以及可变延迟数学流水线，如 XU（用于特殊函数，如以 2 为底的指数 MUFU.EX2）。MIO 单元专用于管理数据移动。它包含片上内存缓存——L1 缓存和共享内存——以及加载存储单元，后者执行内存指令（例如 LDGSTS、STS）以访问全局、局部或共享内存 [52], [53]。

这种跨 Ampere 及后续架构的一致的高层 SM 组织提供了一个稳定的微架构抽象。PIPEWEAVE 建立在此基础之上，实现了跨内核和硬件平台的通用性。

## III. 研究动机

LLM 和 GPU 的快速协同演化要求性能建模工具具备快速、准确且能跨多种 LLM、服务框架和硬件架构泛化的能力。当前的方法，如周期精确模拟 [4], [22]、解析建模 [6], [19], [25] 和数据驱动方法 [1], [73]，各自提供了部分解决方案，但没有一种能完全满足这些要求。尽管周期精确模拟器提供了高保真度，但它们速度很慢。解析模型耗费大量人力且缺乏泛化能力。

在这种背景下，数据驱动策略——特别是将解析建模与机器学习技术集成的“灰盒”方法 [26], [76]——代表了一种独特的范式。这些方法旨在通过利用诸如 Tile 级分解等策略并结合硬件规格，提供快速预测并改善泛化能力。然而，尽管取得了显著进展，即使是这些先进技术也会遇到限制其精度的建模约束，预测误差超过 40%（Section VI-C）。

核心问题源于微架构保真度不足。具体而言，虽然像 Neusight [26] 这样最先进的模拟器结合了工作负载级特征工程（例如，Tile 分解），但它们在三个关键方面仍未达到真正的微架构建模水平。

粒度不匹配：尽管此类方法将工作负载分解为线程块 Tile，但其硬件表示仍然是粗粒度的。它们主要依赖 Tile 级描述符作为 ML 模型的输入，而没有明确建模 Tile 的执行如何转化为对 SM 异构指令流水线的特定需求和竞争。例如，单个 Tile 的执行本质上涉及多个硬件单元的并发活动——如 Tensor、FMA 和内存流水线。因此，以 Tile 为中心的抽象将异构流水线活动合并为单一的聚合工作负载表示，实际上将 SM 视为单一的黑盒，而不是反映其真实的执行行为。

无法对融合内核进行建模：此类方法主要在计算图中的单个深度学习算子级别（例如，GEMM、Softmax）对性能进行建模。这种抽象假设内核执行可以近似为标准算子的组合。然而，现代高性能实现越来越依赖融合内核（例如，FlashAttention [11]），其中多个算子被紧密集成到单个 GPU 内核中。在此类内核中，性能由算子融合引入的紧密耦合执行模式决定，其中多个计算在共享的执行结构中执行，并且中间数据在各步骤间重用。因此，有效的计算和数据移动行为不再与单个算子的边界对齐。这些执行特征无法通过算子级分解来准确捕获，从而导致显著的建模局限性。

静态 Wave 建模：虽然当前的基线方法试图考虑 Wave 量化（线程块调度的尾部效应），但它们通常依赖于一个静态假设，即 Wave 内的 Tile 表现出统一的执行延迟。在实践中，动态工作负载——例如处理可变长度 Token 的因果注意力或具有提前退出条件的内核——引入了显著的 Tile 间延迟变化。如果没有细粒度的调度模型，这些方法难以捕获经常降低实际性能的跨 SM 负载不平衡和尾部效应。

这些局限性凸显了对一种新的建模方法的需求，这种方法能更忠实地反映 GPU 执行。一个准确且可泛化的模型必须将高层工作负载结构与内核在现代 GPU 上执行的微架构现实联系起来。特别是，它应该明确表征工作负载分解和调度如何转化为对异构指令流水线的具体需求，而不是将执行视为抽象的工作负载描述。同时，纯解析建模不足以捕获实际内核中出现的复杂交互和资源耦合，并且此类模型通常针对特定算子和硬件假设定制，使得它们在出现新内核或 GPU 架构时难以在没有大量重新推导的情况下进行泛化。因此，需要一种混合设计：将模型建立在执行感知的解析结构基础上，同时利用基于学习的组件来捕获高阶性能效应。这种设计理念是 PIPEWEAVE 的基础。

## IV. PIPEWEAVE 设计

实现准确且可泛化的 GPU 性能预测，需要全面理解软件内核与底层硬件架构之间错综复杂的相互作用。一种稳健的建模方法必须同时考虑确定性的一阶效应和复杂的动态交互。因此，我们提出了 PIPEWEAVE，这是一个建立在由知识与数据双重原则指导的方法论之上的框架。

知识驱动组件是一个层次化的解析模型，它利用 GPU 并行执行模型的深度领域特定知识，系统地分解内核复杂的执行流程。这种自顶向下的分解从整个内核推进到一组基本任务，并进一步推进到对特定指令流水线的基本需求。这种分解为互补的数据驱动组件产生了一个可解释的特征集：一个轻量级的 MLP，旨在捕获难以通过解析方法表征的复杂非线性交互和资源竞争。正是这种知识驱动分解与用于高阶效应的数据驱动建模的整合，使得 PIPEWEAVE 能够实现高保真性能预测。

PIPEWEAVE 包含四个核心模块，如 Figure 2 所示：(1) Kernel Decomposer，将内核的整体执行分解为一组基本任务 (§IV-A)；(2) Scheduling Simulator，对任务如何分配到 GPU 的 SM 进行建模，并产生最终的任务分布 (§IV-B)；(3) Feature Analyzer，将任务分布转换为捕获指令流水线需求和相关理论周期的多级特征集 (§IV-C)；以及 (4) Performance Estimator，使用轻量级 MLP 合成这些特征以进行最终预测，从而对复杂的高阶交互进行建模 (§IV-D)。

这种多阶段设计是 PIPEWEAVE 泛化能力的基础。前两个模块通过将任何内核转换为与其来源无关的统一任务分布，确保了内核泛化能力。第三个模块随后通过一个表示目标 GPU 架构参数的紧凑向量将此分布映射到特征集，从而实现硬件泛化能力。一旦针对给定内核的 MLP 在各种硬件配置上进行了训练，预测任何新输入或 GPU（甚至是未见过的架构）的性能就变得非常高效。该过程仅涉及运行快速的解析步骤以生成相应的特征向量，然后执行一次 MLP 的前向传播，从而实现实时预测。

![](images/fd259f026f4195e249e9e3bad09565565c6b0c207df0cc2411f38a6d8b7cea59.jpg)  
Fig. 2. PIPEWEAVE 建模框架概述，详细说明了从内核分解到最终性能预测的流程。

虽然我们的评估在 NVIDIA GPU 架构上得到了验证 (Table VI)，但将内核分解为对异构指令流水线需求的原则从根本上说是通用的。这可以很容易地扩展到其他现代加速器，例如 AMD GPU。

## A. 内核分解器

为了准确捕捉第 II-B 节中描述的现代 GPU 的并行执行过程，PIPEWEAVE 将内核的工作负载分解为一组较小的任务。这种分解是我们方法的核心，因为它以与 GPU 并行性 [47] 相一致的方式对内核进行建模。尽管先前的研究 [26], [76] 已经探索过将内核划分为多个 tile，但它们通常依赖于从性能分析数据中推断简化的 tiling 逻辑。相比之下，PIPEWEAVE 强调对可用源代码进行确定性分析。这使得分解过程更加准确和可验证，能够捕捉现代内核中复杂且多样的任务结构。

任务的精确定义可能因 GPU 架构和内核实现而异。在传统的 GPU 执行模型 [47]（例如 FlashAttention-2 [10]）中，任务通常对应于一个 Cooperative Thread Array (CTA)，也称为线程块。内核启动会生成一个 CTA 网格，硬件调度器在其执行期间将每个 CTA 分配给一个可用的 SM。然而，在现代高性能 GPU 范式中，例如 Ping-Pong GEMM [27], [50] 等模式中使用的 persistent kernel，这种一对一映射不再成立。在这种执行模型下，一个长生命周期的 CTA 驻留在 SM 上并充当持久化工作线程。因此，基本的可调度单元——即我们的任务——不是 CTA 本身，而是驻留 CTA 从全局工作队列中获取的较小计算包。

尽管基本的分解方法是一致的，但具体实现因内核而异。为了表征任务的执行属性，我们的框架识别出定义其范围和规模的维度参数 (d<sub>i</sub>)。虽然这些维度参数以及由此产生的计算工作负载在内核中的所有任务之间通常是统一的（例如，每个 GEMM 任务通常由相同的 tile 维度 (tile M, tile N, tile K) 定义），但情况并非总是如此。一个关键的例外发生在应用了 causal masking 的 FlashAttention [10], [11], [60], [75] 中。由于 causal 约束，处理较早 query token 的任务比处理较晚 token 的任务关注的 key/value token 更少。因此，即使名义上的任务维度看起来是统一的，每个任务的实际工作负载也可能存在显著差异。

表 II  
PIPEWEAVE 所需的硬件规格。
<table><tr><td>参数</td><td>取值范围</td><td>单位</td></tr><tr><td>计算能力</td><td>8.0 - 12.0</td><td></td></tr><tr><td>SM 数量</td><td>78 - 188</td><td></td></tr><tr><td>SM 时钟频率</td><td>1410- 2520</td><td>MHz</td></tr><tr><td>Tensor Pipe 吞吐量</td><td>512 - 4096</td><td>ops/cycle/SM</td></tr><tr><td>FMA Pipe 吞吐量</td><td>64 - 128</td><td>ops/cycle/SM</td></tr><tr><td>XU Pipe 吞吐量</td><td>16</td><td>ops/cycle/SM</td></tr><tr><td>全局内存带宽</td><td>696-4916</td><td>GB/s</td></tr><tr><td>L2 缓存带宽</td><td>2430 - 10400</td><td>GB/s</td></tr><tr><td>每个 SM 的共享内存带宽</td><td>128</td><td>Byte/cycle</td></tr><tr><td>每个 SM 的共享内存大小</td><td>100- 228</td><td>KB</td></tr><tr><td>每个 SM 的寄存器文件大小</td><td>256</td><td>KB</td></tr></table>

我们通过映射函数 F 形式化了推导这些任务及其参数的过程。对于给定的内核，$\mathcal { F }$ 将输入参数 X 和硬件的架构规格 S（表 II）映射到完整的任务集合 $\mathcal { T } = \{ \tau _ { 1 } , \tau _ { 2 } , . . . , \tau _ { t } \}$

$$
\{ \tau _ { 1 } , \tau _ { 2 } , . . . , \tau _ { t } \} = \mathscr { F } ( \mathbf { X } , \mathbf { S } )\tag{1}
$$

每个任务 $\tau _ { i }$ 封装了内核工作负载的特定部分，由其维度参数向量 $\mathbf { d } _ { i }$ 表征。这些参数构成了分析推导任务执行属性（如计算和内存需求）的基础，详见后续章节（§IV-C）。

推导分解函数 $\mathcal { F }$ 的方法取决于内核的可访问性。对于开源库（例如 FlashInfer [75]），$\mathcal { F }$ 是通过直接从源代码中提取并行化策略和线程块映射逻辑来推导的。然而，这种方法不适用于诸如 NVIDIA 的 cuBLAS [51] 等闭源库。为了处理这种情况，我们通过经验推断映射函数。例如，为了识别在 BF16 精度下运行的 cuBLAS GEMM 内核的分解逻辑，我们使用 PyTorch Profiler [56] 等工具对其在不同输入矩阵维度 (M, N, K) 下的执行进行性能分析。通过分析性能分析数据，特别是内核名称、CTA 数量与输入大小之间的相关性，我们逆向工程了内核的隐式任务划分策略。这种经验方法使我们能够构建一个代理映射函数 $\mathcal { F }$，该函数紧密近似专有的分解逻辑。

## B. 调度模拟器

内核的性能不仅取决于其总工作负载，还取决于这些工作如何在 GPU 的并行资源之间分配。在将内核分解为抽象任务集之后，我们框架的下一个关键组件是模拟这些任务在 SM 上的调度。这种调度分析将任务集转换为具体的任务分布，提供任务到特定 SM 的精确映射。这种映射至关重要，因为它能够对内核行为进行准确的每 SM 表征，并有助于识别由工作负载不平衡导致的性能瓶颈——这是先前研究 [26], [29], [74], [76] 中被忽略的一个关键方面。它们通常完全依赖于聚合的内核级指标，并假设一个过度简化的调度模型，其中所有任务都被统一处理。PIPEWEAVE 专为通用性而设计，支持现代 GPU 应用程序中使用的两种主要调度范式。

硬件实现的调度器。对于常规内核，任务调度由 GPU 的硬件调度器（称为 GigaThread Engine [28], [47]）处理。由于该硬件组件的确切行为未公开记录，其默认调度策略通常从经验研究中推断为轮询（RR）[18], [20], [21], [28], [30], [31], [35], [65], [79]。该策略首先为每个 SM 分配至少一个任务（即一个 CTA）。如果一个 SM 仍有足够的资源（例如寄存器、共享内存、warp 插槽等）来支持额外的任务，则执行第二轮分配。这种循环分配过程持续进行，直到所有 SM 因资源限制或硬件限制而饱和。之后，当现有任务完成并从 SM 退出时，新任务才会被分配给该 SM。

软件实现的调度器。对于持久内核，硬件调度器在分发 CTA 中的作用退居次要，因为每个 CTA 仅启动一次并在执行期间驻留在 SM 上。关键调度逻辑由软件处理。在此设置中，长生命周期的 CTA 重复处理取自全局列表的细粒度工作单元。在类 GEMM 内核中，这些单元通常实现为块，代表我们任务的具体形式。块分配由块调度器 [50], [71] 管理，这是一个具有特定于内核逻辑的软件组件。

通过模拟这些调度机制，PIPEWEAVE 准确地推导出现实的任务分布。我们将这种分布形式化为总任务集 $\boldsymbol { \mathcal { T } } =$ $\{ \tau _ { 1 } , \tau _ { 2 } , \ldots , \tau _ { t } \}$ 在可用 SM 上的一个划分。该划分包含集合 $\{ \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 } , \ldots , \mathcal { T } _ { N _ { S M } } \}$ ，其中 $N _ { S M }$ 表示 SM 数量，每个集合 $\tau _ { j }$ 包含分配给第 $j -$ 个 SM 的所有任务。我们的调度模拟器由映射函数 M 表示，按如下方式生成此划分：

$$
\{ \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 } , . . . , \mathcal { T } _ { N _ { S M } } \} = \mathcal { M } ( \mathcal { T } , \mathbf { S } )\tag{2}
$$

集合 $\{ \mathcal { T } _ { j } \}$ 构成 $\tau .$ 的一个划分，使得 $\begin{array} { r } { \bigcup _ { j = 1 } ^ { N _ { S M } } \mathcal { T } _ { j } = \mathcal { T } } \end{array}$ 且对于 $i \neq j$，$\mathcal { T } _ { i } \cap \mathcal { T } _ { j } = \emptyset$

## C. 特征分析器

特征工程在概念上受 Roofline 性能模型 [74] 的原理指导。这一经典模型通过确定 kernel 是受限于硬件的峰值计算吞吐量还是内存带宽，提供了强大的第一阶分析。然而，其对现代 GPU 的预测准确性仍然有限。这是因为其高层次的、二维的计算与内存视图无法捕捉到复杂的现代 kernel 在异构硬件上执行时所产生的错综复杂的资源争用和动态交互。

![](images/b7f5bf4866d00aa73054befc1434b7aa73a344f932133a81546872ff15ca8bb5.jpg)

![](images/9c095713a8f370dffda01b2c98bab04b912c2c90dbec44abd3e9a97d28c0b3c8.jpg)  
图 3. PIPEWEAVE 对 A100 上 FlashAttention-2 的多维分析示例。随着需求增加，两种不同配置的实测性能趋近于理论“屋顶”并趋于平缓。

为了克服这一局限性，PIPEWEAVE 将 Roofline 模型扩展为多维分析。我们的模型不再使用单一的计算屋顶和单一的内存屋顶，而是为每个关键指令流水线计算单独的理论性能极限。这就需要沿着两个基本维度刻画 kernel 执行：(1) 需求，衡量施加于每个流水线的总工作负载（例如操作数或字节数）；(2) 理论周期，由需求推导而来，表示如果仅该流水线成为瓶颈时的理想执行时间。这类似于特定流水线的“屋顶”。图 3 展示了一个具体示例。它绘制了执行效率（即理论周期与实测延迟的比值）相对于绝对流水线需求的变化。与标准 roofline 不同，流水线被解耦到独立的图表中，每个图表都显示出可预测且独立的饱和趋势。

此外，我们不为复杂的指令级并发（例如 Tensor 和 FMA 流水线的并行执行）或架构特定机制（例如 Hopper 的 Tensor Memory Accelerator (TMA)）构建僵化的分析模型。精确建模此类微架构细节将需要针对特定代际的逆向工程，这会损害跨代际的泛化能力并显著增加建模复杂性。相反，PIPEWEAVE 采用了经过深思熟虑的抽象策略。我们将多种内存访问机制——从传统的 LSU 指令到先进的异步拷贝——统一为广义的内存流水线需求。通过将这些基础流水线需求作为独立的原始特征暴露出来，我们允许模型在随后的 MLP 阶段自动学习它们之间复杂的非线性交互。根据经验，我们发现这种抽象足以捕捉跨架构的主要性能行为，同时保持强大的泛化能力。

这些特征的生成遵循跨越三个层级的自底向上过程。首先，在任务层级，我们刻画 Math 流水线和 MIO 流水线的独立需求，推导它们对应的每任务理论周期。接下来，这些每任务特征被聚合到 SM 层级，为每个 SM 创建详细的配置文件，并能够识别利用率最高的 SM 的特征。最后，第二次聚合产生一个全 GPU 配置文件，包含所有主要流水线的需求和理论周期指标。

表 III  
关键 MATH 流水线执行的主要操作。
<table><tr><td>Math 流水线</td><td>主要操作</td></tr><tr><td>Tensor</td><td>各种精度的 MMA 指令（例如 FP8、FP16、BF16）。</td></tr><tr><td>FMA</td><td>FP32 浮点加法、乘法和融合乘加。</td></tr><tr><td>XU</td><td>FP32 近似浮点特殊函数（例如倒数、倒数平方根、以 2 为底的对数、以 2 为底的指数、正弦、余弦）。</td></tr></table>

1) Math 流水线：对于每个任务 $\tau _ { i } \in \mathcal { T } .$ ，我们通过其执行的操作数来定义每个 math 流水线的计算需求。这些流水线主要处理两种操作类型：在 Tensor 流水线上执行的矩阵乘累加 (MMA) 操作，以及由 FMA 或 XU 流水线等单元处理的逐元素 (EW) 操作。每个 math 流水线的关键操作 [46], [47], [52], [53] 概述于表 III 中。

对于 $\tau _ { i } .$ 中的 MMA 操作，操作计数 $( N _ { \mathrm { o p s , T e n s o r } } )$ 直接从任务维度向量 $\mathbf { d } _ { i } .$ 推导而来，该向量包含分块几何 {tile M, tile N, tile K}。总操作计数为：

$$
N _ { \mathrm { o p s , T e n s o r } } = \alpha \cdot t i l e \_ M \cdot t i l e \_ N \cdot t i l e \_ K\tag{3}
$$

这里，系数 $\alpha$ 表示在 MMA 计算期间每个输出元素的基本乘加操作总数。在标准 GEMM kernel [42], [43] 中，一次矩阵乘法给出 $\alpha \ : = \ : 2$ ，而 FlashAttention kernel 每个任务执行两次顺序矩阵乘法 [11]，导致 $\alpha = 4$

对于任务 $\tau _ { i } ,$ 中的 EW 操作，我们的分析直接计算每个 math 流水线的总操作数 $( \mathrm { e . g . , } N _ { \mathrm { o p s , F M A } } , N _ { \mathrm { o p s , X U } } )$ 。这需要通过分析 kernel 的算术表达式和循环迭代空间，推导特定硬件流水线（表 III）的聚合操作计数。

最后，对于每个流水线 $p ,$ 执行这些操作所需的理论周期 $C _ { p }$ 是通过将总操作计数 $N _ { \mathrm { o p s } , p }$ 除以其对应的吞吐量 $T h _ { p } ,$ （一个来自硬件规格 S 的参数）来确定的：

$$
C _ { p } = \frac { N _ { \mathrm { o p s } , p } } { T h _ { p } }\tag{4}
$$

在获得每任务需求特征后，我们使用自底向上的方法将任务分布 $\{ \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 }$

$\dots , \mathcal { T } _ { N _ { S M } } \Big )$ 聚合为 SM 级和 GPU 级特征。从 SM 级开始，对于每个流水线 $p ,$ 我们结合分配给 $\mathbf { S } \mathbf { M } _ { j }$ 的所有任务的需求，以计算每 SM 总操作数 $N _ { \mathrm { o p s , \it 1 } } ^ { \mathrm { S M _ { \it j } } }$ <sub>p</sub> 和理论周期 $C _ { p } ^ { \mathrm { { S M _ { \it 3 } } } }$ 。这些每 SM 的值被求和以获得总体 GPU 操作数 $\bar { N } _ { \mathrm { o p s , p } } ^ { \mathrm { G P U } }$ 。相应的 GPU 级理论周期由该总工作负载和流水线 $p \mathrm { : }$ 的合并吞吐量推导得出：

表 IV  
作为输入提供给 MLP 的分析特征向量。
<table><tr><td>流水线</td><td>粒度</td><td>特征</td></tr><tr><td rowspan="3">Math</td><td>GPU</td><td>总操作数 总理论周期</td></tr><tr><td></td><td>最大 SM 操作数</td></tr><tr><td>SM</td><td>最大 SM 理论周期</td></tr><tr><td rowspan="3">MIO</td><td>GPU</td><td>总内存需求 理论周期 (Global, L2)</td></tr><tr><td></td><td></td></tr><tr><td>SM</td><td>最大 SM 内存需求 理论周期 (Global, L2, Shared)</td></tr></table>

$$
C _ { p } ^ { \mathrm { G P U } } = \frac { N _ { \mathrm { o p s , } p } ^ { \mathrm { G P U } } } { N _ { S M } \cdot T h _ { p } }\tag{5}
$$

2) MIO流水线：对于MIO流水线，我们在三个层级上以字节为单位衡量总需求。首先，对于每个任务 $\tau _ { i } ,$ 我们通过累加其从内存层次结构中加载的所有数据来计算每个任务的总内存需求 $B _ { i }$。采用这种方法是因为在大多数kernel中，加载操作通常位于关键执行路径上。数据停顿会直接影响消费者延迟（math流水线）[53]。利用任务分布，将这些逐任务值累加得到集合 $\tau _ { j }$ 中任务的每SM内存需求 $B ^ { \mathrm { S M } _ { j } }$。最后，将所有每SM值 $B ^ { \mathrm { S M } _ { j } }$ 相加得到全局内存需求 $B ^ { \mathrm { G P U } }$

从这些聚合的字节计数中，我们推导出若干理论周期特征。理论周期 $C _ { \mathrm { m e m } }$ 通过将给定层级的总字节数除以特定内存子系统的理论带宽来计算，表示为 $C _ { \mathrm { m e m } } = B / B W _ { \mathrm { m e m } }$。在GPU层级，我们使用 $B ^ { \mathrm { G P U } }$ 以及L2 Cache和Global Memory带宽来应用此公式。在SM层级，$B ^ { \mathrm { S \bar { M } } _ { j } }$ 与Shared Memory、L2 Cache和Global Memory的每SM带宽一起使用。

## D. 性能估计器

PIPEWEAVE的最后一个组件是一个轻量级机器学习模型，用于预测整体kernel执行时长。MLP使用单一特征向量作为输入，该向量是先前阶段所有分析特征的拼接（Section IV-C）。该向量包含来自MIO流水线的特征（Table IV），加上基于kernel特定操作的一个或多个Math流水线的特征。

我们采用逐kernel建模方法，为每个kernel类别训练一个单独的MLP。每个MLP的训练数据集通过在各种GPU架构和输入参数下对相应kernel的执行进行性能分析来构建。对于每个样本，我们记录物理硬件上的实际执行延迟作为真实值。

## V. 实现细节

## A. 分析模型

为确保我们的性能模型准确反映真实世界的LLM推理工作负载，我们直接从流行的高性能服务框架（如SGLang [59]和vLLM [72]）的后端中选择了一组具有代表性的关键kernel。这些kernel的关键特征总结在Table V中。注意，对于GEMM和Attention等类别，通常存在多种实现。对于cuBLAS GEMM kernel，我们观察到特定实现在不同硬件架构上有所差异。对于FlashInfer Attention kernel，我们的分析包括FlashAttention-2 (FA2)和FlashAttention-3 (FA3)变体，涵盖paged和ragged KV cache布局的实现 [15]。

TABLE V  
所选kernel的关键特征。
<table><tr><td>Category</td><td>Source</td><td>Language</td><td>Precision</td><td>Scheduler</td><td>Math Pipe</td></tr><tr><td>GEMM</td><td>cuBLAS</td><td>预编译</td><td>BF16/FP16</td><td>HW/SW</td><td>Tensor</td></tr><tr><td>Scaled MM</td><td>vLLM</td><td>CUDA C++</td><td>FP8</td><td>HW/SW</td><td>Tensor</td></tr><tr><td>Attention</td><td>FlashInfer</td><td>CUDA C++</td><td>BF16/FP16</td><td>HW/SW</td><td>Tensor, XU</td></tr><tr><td>RMSNorm</td><td>FlashInfer</td><td>CUDA C++</td><td>FP32</td><td>HW</td><td>FMA, XU</td></tr><tr><td>SiLU&amp;Mul</td><td>FlashInfer</td><td>CUDA C++</td><td>FP32</td><td>HW</td><td>FMA, XU</td></tr><tr><td>Fused MoE</td><td>SGLang</td><td>Triton</td><td>BF16/FP16</td><td>HW</td><td>Tensor</td></tr></table>

对于每个kernel类别，Kernel Decomposer的实现非常简洁，仅需10-50行代码。除了cuBLAS GEMM的分解直接取自性能分析数据外，其他kernel的Equation (1)中的映射函数 $\mathcal { F }$ 均来自其源代码。由于cuBLAS GEMM是闭源的，且其实现在不同硬件架构上有所不同，因此在新GPU上的分解行为是未知的。因此，对于缺乏闭源kernel性能分析数据的未见GPU，我们使用性能分析数据集中架构最相似的GPU的分解逻辑。

在kernel分解之后，Scheduling Simulator将任务分配到各SM上。对于所分析的大多数kernel（Table V），它们使用基于硬件的调度器，我们模拟了Section IV-B中描述的广泛推断的RR策略。对于Hopper架构上的cuBLAS GEMM和FlashInfer FA3 kernel [75]，两者均采用persistent kernel设计，我们对其各自的基于软件的调度器进行建模。以FlashInfer FA3为例，我们在模拟器中精确复现了其基于MinHeap的调度器逻辑，约用40行代码。

Feature Analyzer将任务分布转换为全面的特征集。对于math流水线，我们的实现聚焦于对LLM工作负载最为关键的三种指令流水线：Tensor、FMA和XU流水线。我们发现这三者共同覆盖了目标kernel中的大部分计算需求。其他流水线，如处理逻辑操作的ALU [53]，由于在kernel中利用率通常较低且难以分析计数其操作而被排除在外。

## B. 数据集构建

为了训练和评估 PIPEWEAVE，我们通过对选定的 kernel（表 V）在各种 NVIDIA GPU 架构上进行性能分析，构建了一个全面的数据集。该数据集涵盖了 11 种不同的 GPU 模型 [36]–[38], [44]，代表了多种架构和市场细分。如表 VI 所示，这些模型被分为两组：第一组用于训练，而第二组仅保留用于测试，以评估 PIPEWEAVE 对未见硬件的泛化能力。

表 VI  
被评估的 NVIDIA GPU 的关键规格。
<table><tr><td>GPU</td><td>架构</td><td>SMs</td><td>内存带宽 (GB/s)</td><td>Tensor BF16 (ops/clk/SM)</td><td>频率 (MHz)</td></tr><tr><td>A40</td><td>Ampere</td><td>84</td><td>696</td><td>1024</td><td>1740</td></tr><tr><td>A100</td><td>Ampere</td><td>108</td><td>2039</td><td>2048</td><td>1410</td></tr><tr><td>RTX 6000 Ada</td><td>Ada</td><td>142</td><td>960</td><td>1024</td><td>2505</td></tr><tr><td>L20</td><td>Ada</td><td>92</td><td>864</td><td>516</td><td>2520</td></tr><tr><td>H20</td><td>Hopper</td><td>78</td><td>4023</td><td>1024</td><td>1830</td></tr><tr><td>H800</td><td>Hopper</td><td>132</td><td>3352</td><td>4096</td><td>1830</td></tr><tr><td>RTX A6000</td><td>Ampere</td><td>84</td><td>768</td><td>1024</td><td>1800</td></tr><tr><td>L40</td><td>Ada</td><td>142</td><td>864</td><td>512</td><td>2490</td></tr><tr><td>H100</td><td>Hopper</td><td>132</td><td>3352</td><td>4096</td><td>1830</td></tr><tr><td>H200</td><td>Hopper</td><td>132</td><td>4917</td><td>4096</td><td>1830</td></tr><tr><td>RTX PRO 6000 S</td><td>Blackwell</td><td>188</td><td>1792</td><td>1024</td><td>2340</td></tr></table>

性能分析是在一致的软件环境中使用 PyTorch 2.8.0、CUDA Toolkit 12.8、FlashInfer 0.4.1、SGLang 0.5.4、vllm 0.11.0 和 Triton 3.4.0 进行的。对于 kernel、输入参数、服务框架和 GPU 硬件的每种组合，我们使用 PyTorch Profiler 测量执行延迟。我们进行了 5 次预热运行，随后进行了 10 次测量运行，并使用它们的平均值作为真实值。性能分析数据集包含 6 个关键 kernel，作为 vLLM [72] 和 SGLang [59] 的核心计算后端：

• Attention：104,958 个样本（71,969 个训练和 32,989 个测试）。bs $\in ~ [ 1 , 1 6 ]$ , nh $\in \ [ 2 , 1 2 8 ]$ , nkv $\in \ [ 1 , 8 ]$ , hd ∈ {64, 128}, qlen ∈ [1, 20097], kvlen $\in \ [ 4 , 2 0 4 8 1 ]$ 。Query 和 KV 的长度在每个 batch 内随机变化，以模拟真实的变长序列模式。

• GEMM：613,263 个样本（494,463 个训练和 118,800 个测试）。$\begin{array} { l l l } { M } & { \in } & { [ 2 } \end{array}$ , 131072], $\begin{array} { r l r } { N } & { { } \in } & { [ 3 8 4 , 1 5 2 0 6 4 ] } \end{array}$ $K \in$ [256, 53248]。

• RMSNorm：65,036 个样本（44,592 个训练和 20,444 个测试）。seq ∈ [2, 131072], dim ∈ [128, 16384]。

• SiLU&Mul：104,834 个样本（71,868 个训练和 32,966 个测试）。seq ∈ [2, 131072], dim ∈ [768, 106496]。

• Scaled MM：25,228 个样本（16,818 个训练和 8410 个测试）。$M \in [ 2 , 1 3 1 0 7 2 ] , N \in [ 3 8 4 , 8 1 9 2 ] , K \in [ 2 5 6 , 8 1 9 2 ] .$

• Fused MoE：33,264 个样本。$M \in [ 2 , 8 1 9 2 ] , E \in [ 8 , 1 2 8 ]$ topk ∈ [2, 8], H ∈ [1024, 4096], $N \in \ [ 5 1 2 , 3 0 7 2 ]$ 。该 kernel 在第七节中作为我们优化方法的详细案例研究。

## C. MLP 模型训练

如第 IV-D 节所述，使用导出的解析特征为每种 kernel 类型训练一个单独的 MLP。该 MLP 具有浅层架构，包含 3 个隐藏层（256、128 和 64 个单元），采用 ReLU 激活函数，随后使用 Batch Normalization 和 Dropout（比率 0.1）进行正则化。输出层利用 Sigmoid 激活函数将预测值限制在 [0, 1] 范围内，表示 kernel 的执行效率（定义为理论执行时间与实际延迟的比率）。最终的延迟预测是通过将理论执行时间除以该估计效率来获得的。

训练使用第 V-B 节中描述的数据集。应用 AdamW 优化器 [32]，初始学习率为 0.001，并使用权重衰减。平均绝对百分比误差（MAPE）作为损失函数，用于最小化相对预测误差。采用早停法，通过监控验证损失来防止过拟合。

## D. 端到端性能预测

除了预测单个 kernel 的性能外，我们还验证了我们的框架在建模端到端 LLM 推理延迟方面的准确性。我们基于 SGLang [59] 和 vLLM [72] 的模型定义和 kernel 调用逻辑构建了一个 Workload Generator。给定模型配置和输入参数，该生成器会创建一个代表真实推理场景的 kernel 调用序列。遵循先前的工作 [26], [76], [80]，我们假设 kernel 顺序执行而没有重叠。对于序列中的每个 kernel，我们使用 PIPEWEAVE 根据其类型和输入维度预测其运行时间。单 GPU 推理的总端到端延迟是通过将所有预测的 kernel 持续时间相加来计算的。

预测分布式环境中的端到端性能需要对多 GPU 并行所需 [1], [26], [73], [78] 的计算 kernel 和通信 kernel 进行建模。根据所采用的并行策略，这包括诸如用于 Tensor Parallelism (TP) 的 All-Reduce 或用于 Pipeline Parallelism (PP) 的 Send/Recv 原语等 kernel。为了对这些通信 kernel 进行建模，我们使用了一种简化的方法。我们在不同的网络拓扑和通信量下分析它们的性能，以构建基线性能数据库。利用这些数据，我们应用数据驱动的回归技术（例如 Random Forest）来估计通信 kernel 的延迟。然后将该预测与计算 kernel 的估计相结合，以预测分布式推理的总端到端延迟。

## VI. 评估

## A. 基线

为了全面评估 PIPEWEAVE，我们通过将其预测准确性与四个主要基线进行比较来进行初步评估：(1) 经典的解析 Roofline 模型 [74]；(2) 基于 Linear 回归的模型 [29]；(3) Habitat [76]；以及 (4) Neusight [26]，一种最先进的数据驱动方法。为了确保这些主要基线之间的公平比较，我们对它们进行了调整，以纳入我们的解析组件。Linear 模型遵循原论文 [29] 中的方法，使用来自我们的 Feature Analyzer（第 IV-C 节）的两个主要特征进行训练：用于聚合计算和内存需求的理论周期。类似地，我们为 Habitat 和 Neusight 提供了来自我们的 Kernel Decomposer（第 IV-A 节）的精确任务定义。

此外，为了突出我们的解析-ML 混合设计在预测准确性和模拟效率方面的优势，我们引入了代表高度详细建模范式的第二组基线：AMALI [6]，一种基于指令跟踪的解析模型；以及 LLMCompass [78]，一个集成了解析模型和周期精确脉动阵列建模的混合框架。由于这些详细的模拟器对各种现代 kernel 的支持有限，并且对于端到端 LLM 工作负载会产生大量的运行时开销，我们将此比较限制在独立的 GEMM 上。

表 VII  
PIPEWEAVE 解析操作计数的 MAPE (%)
<table><tr><td>指标</td><td>gemm8</td><td>gemm9</td><td>FA2</td><td>FA3</td></tr><tr><td>最大 SM 操作数 (%)</td><td>0.07</td><td>0.04</td><td>6.34</td><td>0.45</td></tr><tr><td>总操作数 (%)</td><td>0.01</td><td>0.14</td><td>0.50</td><td>0.00</td></tr></table>

## B. 分析组件的验证

我们首先验证 PIPEWEAVE 的核心分析组件：Kernel Decomposer、Scheduling Simulator 和 Feature Analyzer。这一步至关重要，因为这些部分按顺序工作。任何错误都可能会传播并降低最终的特征质量。

我们首先验证 Kernel Decomposer 的正确性。具体来说，我们将我们分解过程中的 CTA 数量与数据集中多个 kernel 的真实配置进行比较。结果完全一致，证实了分解的准确性。

接下来，我们评估 Scheduling Simulator 和 Feature Analyzer 的准确性。我们的方法将分析推导出的数学流水线操作计数（包括总操作数（kernel 范围内）和每个 SM 的最大操作数）与来自 NVIDIA Nsight Compute (NCU) 工具 [53] 的真实测量值进行比较。由于较高的分析开销和受限的硬件访问权限，我们在两款旗舰设备上执行此验证：A100 和 H100。评估涵盖了四个关键的 kernel 实现：cuBLAS GEMM（A100 上的 gemm8 和 H100 上的 gemm9）、FA2 和 FA3。每个实现包含约 500 个从第 V-B 节定义的工作负载配置范围中随机采样的测试样本。如表 VII 所示，我们的模型在总操作数上的最大误差为 0.5%，在每 SM 最大操作数上的最大误差为 6.3%。FA2 (6.34%) 的误差高于 FA3 (0.45%)，这主要是由于它们不同的调度机制：FA3 使用具有确定性任务调度的 persistent-kernel 设计，可以被显式模拟，而 FA2 依赖于动态硬件调度，这在预测每 SM 峰值工作负载时引入了额外的不确定性。

最后，我们使用 GEMM 和 Attention kernel 的完整数据集（第 V-B 节）对它们进行消融研究，以突出我们核心组件的贡献。我们将完整的 PIPEWEAVE 模型与三个消融变体进行比较：(1) w/o MIO（无 MIO 特征），(2) w/o Math，和 (3) w/o MLP（用基于 Roofline 的预测器替换 MLP）。如图 4 所示，每个组件对准确的性能预测都至关重要。对于 Attention kernel，完整模型比 w/o MIO、w/o Math 和 w/o MLP 分别实现了 1.1×、1.8× 和 2.9× 更高的准确度。这种影响对于 GEMM 更为显著，我们的完整模型分别提高了 3.2×（w/o MIO）、2.7×（w/o Math）和 3.5×（w/o MLP）的准确度。虽然两个 kernel 都从我们的建模框架中显著受益，但 Attention kernel 的最终预测误差 (15.54%) 仍然高于 GEMM kernel (8.39%)。如前面表 VII 所示，这种差距不是由分析操作计数的不准确引起的，因为两者的分析操作计数都保持在相对较低的水平。相反，它源于 Attention 机制固有的不均匀工作负载分布和动态执行特征。与 GEMM 中任务由跨 tile 的统一维度参数定义不同，Attention 工作负载表现出显著的方差。这种方差主要源于 causal masking——处理较早 query token 的任务比处理较晚 token 的任务关注更少的 key/value token——以及 batch 内随机变化的序列长度。此外，Attention 引入了更复杂的内存行为和具有不同计算-内存特征的异构执行阶段，这进一步增加了运行时的变异性。这些因素使得执行延迟对硬件调度动态更加敏感，并导致更大的块间完成时间方差。因此，对于 MLP 而言，Attention 延迟本质上比 GEMM 工作负载中观察到的稳定且统一的执行模式更难建模。

表 VIII  
在已见和未见 GPU 上的预测误差。
<table><tr><td>硬件</td><td>Roofline</td><td>Linear</td><td>Habitat</td><td>Neusight</td><td>PIPEWEAVE</td></tr><tr><td>已见</td><td>72.22%</td><td>59.50%</td><td>28.92%</td><td>43.49%</td><td>6.77%</td></tr><tr><td>未见</td><td>79.61%</td><td>70.28%</td><td>85.96%</td><td>46.70%</td><td>13.14%</td></tr></table>

我们在包含来自 11 种不同 GPU 的约 100 万个样本的数据集上评估了 PIPEWEAVE（第 V-B 节）。该数据集包含了来自现代推理框架（如 vLLM [72] 和 SGLang [59]）的基础 kernel，涵盖 FP32、BF16/FP16 和 FP8 精度。PIPEWEAVE 实现了最先进的预测精度，并显著超越了先前的工作。在已见硬件上，它达到了 6.0% 的平均 MAPE，优于次优的 Neusight（42.6%）。在未见硬件上，我们的框架展现了卓越的泛化能力，平均 MAPE 为 11.5%——与 Neusight（45.1%）相比提升了 3.9 倍。

图 5 展示了在 BF16 LLM 推理场景下四个典型 kernel 的预测精度（MAPE）。相应地，表 VIII 总结了这四个 kernel 在已见和未见硬件上的平均 MAPE。Linear 和 Roofline 模型在已见和未见硬件上的误差都显著高于 PIPEWEAVE，峰值 MAPE 分别达到 215.6% 和 263.5%。尽管 SOTA 基线 Neusight 优于其他基线，但其最高误差 75.7% 仍远高于 PIPEWEAVE 的 23.4%。此外，分析方法的预测误差，即 Linear 和 Roofline 模型，高度依赖于硬件。例如，图 5(b) 突出显示了 Roofline 模型在 H20 (11%) 和 H800 (127%) 上对 GEMM kernel 的 MAPE 形成了鲜明对比。这种差异源于两款 GPU 不同的计算内存比。具体而言，虽然 H20 保留了 H800 约 120% 的内存带宽，但其峰值计算能力被限制在 H800 的约 15%。在这种极低的计算内存比下，H20 上的计算单元很容易饱和。丰富的内存带宽确保了执行流水线不断得到数据供给，使得 GEMM 能够维持非常接近理论峰值的吞吐量；因此，Roofline 的估计保持准确。相反，H800 具有庞大的计算能力，在大多数实际场景中极难完全饱和，因为达到理论峰值需要近乎完美的指令级并发和不间断的数据交付。在实践中，不可避免的微架构摩擦阻碍了 kernel 接近这种理想化的 Roofline 峰值，从而导致显著的高估。与这类模型不同，PIPEWEAVE 的 MLP 自然地学习了这些特定于硬件的低效性，从而实现了显著更低的预测误差。

![](images/e510c30ecee74c3142a467101f8beaf5c0bfeb8e8e40599c61c2c76364d3b18f.jpg)  
Fig. 4. 关于 MIO 和 Math Pipeline 特征对 GEMM 和 Attention kernel 影响的消融研究。

Qwen2.5-14B arxiv\_8  
(d) SiLU&Mul  
(c) RMSNorm  
![](images/e9b89eb8858bb8f933bd22f83e3fa264bfc79a3acf311e7c8d5b120a9247e05c.jpg)

![](images/f3ce7edd7239be23bc905ea5b088ca8d5ab0f46be4f0deae3cf27d15ad3d0b46.jpg)

![](images/236ba39764f6ed7df64114fea45024f7477993bdb34fbaf2b16a6832fc777af0.jpg)

![](images/6c69659831d99e297eadc20ea1fe9f8aad016992acc738208b9d52d07135b331.jpg)

Fig. 5. PIPEWEAVE 和基线模型的 Kernel 级预测精度（MAPE）。未见硬件平台以灰色背景标识。  
![](images/bc8a92cc102f09c68549f370d34d9506b3827de140585b26ad5a5abd4dbe13cf.jpg)

![](images/459518b8d76b7fead3591e27adbb57828e7223396ba61adbea11e845ec94aa36.jpg)  
Qwen2.5-14B splitwise\_32  
Fig. 6. 使用 SGLang 进行单 GPU Qwen2.5-14B 推理时，PIPEWEAVE 和基线模型的端到端推理预测精度（MAPE）。未见硬件平台以灰色背景标识。

除了 BF16 LLM 推理场景中常见的四个 kernel 外，我们还在 Hopper 架构上训练并测试了用于 FP8 推理的 scaled mm kernel（块级量化），实现了较高的预测精度。在已见硬件（H20、H800）上，PIPEWEAVE 的 MAPE 分别为 1.9% 和 4.1%，而在未见硬件（H100、H200）上，MAPE 分别为 4.2% 和 5.2%。这突显了该框架对 FP8 精度 kernel 的适应性，与 Roofline、Linear、Habitat 和 Neusight 相比，平均精度分别提升了 10.8 倍、9.5 倍、5.5 倍和 7.8 倍。

最后，为了评估 PIPEWEAVE 的预测精度和模拟效率，我们在 A100 GPU 上与 AMALI 和 LLMCompass 进行了针对性的比较。正如我们的基线方法论（第 VI-A 节）所述，由于这些详细模拟器的高计算开销，此比较仅限于 GEMM。使用从我们的数据集（第 V-B 节）中随机抽取的 540 个具有不同维度的 GEMM 样本，我们测量了预测误差和每个 GEMM 的模拟开销。图 7 展示了比较结果，其中预测误差以带符号的相对误差报告，以捕捉高估和低估的情况。总体而言，PIPEWEAVE 在保持更高预测精度的同时，实现了显著更低的模拟开销。平均而言，它获得了 6.4% 的 MAPE，而 AMALI 和 LLMCompass 分别为 28.3% 和 29.7%，同时将预测时间减少了 3 到 7 个数量级。这些结果表明，灰盒设计——将流水线需求分析建模与 ML 相结合——能够有效捕捉主导性能因素，而无需昂贵的低级模拟。

![](images/7652f8d100393c89b6efe09d3f526cce70aa109fe2d48b82672d87aeea946fc9.jpg)  
Fig. 7. A100 GPU 上 GEMM 工作负载的模拟开销与相对预测误差比较。

## D. E2E 推理精度

除了内核级别的验证，我们还通过将其模拟结果与 SGLang [59] 和 vLLM [72] 的实际服务延迟进行比较，来评估 PIPEWEAVE 的端到端预测精度。遵循先前的工作 [1]，我们使用了两个具有代表性的数据集——Arxiv Summarization [8] 和 Splitwise [55]——并在单 GPU (TP=1) 和分布式 (TP, PP) 推理设置下测试了三个典型的 LLM (Qwen2.5-14B、Qwen3-32B 和 Llama3.1-70B)。

这些数据集的工作负载是通过随机采样请求以创建不同大小的批次生成的，例如 arxiv_8 和 splitwise_64。arxiv_<sub>*</sub>（其中 <sub>*</sub> 表示批次大小）工作负载的平均输入长度为 2,630 个 Token，而 splitwise_<sub>*</sub> 工作负载平均为 982 个 Token。输出长度从 5 到 4,056 个 Token 不等。

对于单 GPU (TP=1) 评估，我们在所有 11 个 GPU 上测试了 Qwen2.5-14B（Figure 6）。PIPEWEAVE 实现了 11.3% 的平均 MAPE，显著优于最佳基线 Neusight 的 34.5%。此外，PIPEWEAVE 在未见过的 GPU 上保持了高精度，MAPE 为 12.5%——比 Neusight 的 34.4% 提升了显著的 2.8 倍。

这种鲁棒性延伸到了分布式推理。如 Table IX 所示，PIPEWEAVE 在多样化的端到端推理场景中提供了一致的精度。它跨越了两个推理框架（SGLang 和 vLLM）、多个模型（Qwen3-32B 和 Llama3.1-70B）以及各种并行策略（TP=2、TP=4、TP=8 和 TP=4&PP=2）。PIPEWEAVE 始终保持较低的 MAPE 平均值：8.4%（SGLang, Qwen3-32B, TP=2）、4.3%（SGLang, Llama3.1-70B, TP=4）、7.7%（SGLang, Llama3.1-70B, TP=8）以及出色的 4.0%（vLLM, Llama3.1-70B, TP=4&PP=2）。这一性能显著超越了最佳基线 Neusight。在所有 20 个测试配置中，PIPEWEAVE 实现了 6.6% 的总体平均 MAPE，而 Neusight 为 34.7%，显示出 5.3 倍的平均精度提升。有趣的是，我们的分析表明，在一些 E2E 推理场景中，诸如 Neusight 之类的基线尽管内核级预测精度较差，但可以表现出非常低的 E2E 误差（例如 0.5%）。我们指出了造成这种现象的两个主要原因。首先，E2E 延迟聚合了许多内核的执行时间，这导致了系统性的误差抵消：对某些内核的高估抵消了对其他内核的低估，从而降低了整体 E2E 误差。其次，与全面的内核级评估（Section V-B）中涵盖的工作负载维度相比，E2E 推理通常涉及更窄、更受限的工作负载维度集合；因此，这些工作负载通常位于基线预测的“最佳点”附近。

总而言之，PIPEWEAVE 为 GPU 性能建模提供了高保真度、快速预测和广泛的泛化能力。

## VII. 超越模拟

在先前章节中，我们验证了 PIPEWEAVE 的鲁棒性。在 MAPE 损失下训练，我们的框架在预测各种高度优化的内核在多样硬件平台上的性能方面展现出了强大的精度。在本节中，我们从一般预测过渡到更具挑战性的任务：优化指导。我们的目标是跨硬件平台提升 Fused MoE Triton 内核——SGLang [59] 中默认的 MoE 后端——的性能。

主要挑战在于性能潜力是不透明的。对于任何给定的输入形状和硬件平台，可达到的性能上限是未知的。因此，我们无法先验地确定当前执行是接近最优还是次优。例如，在 A40 上达到 roofline [74] 的 50% 可能很差，如果真实的上限是 70%；而在 A100 上达到 20% 可能是接近最优的，如果上限只有 21%。缺乏这一真实基准，就不可能系统地量化性能差距或确定系统级优化工作应指向何处。

因此，我们超越了模拟。我们的目标不是预测平均性能，而是提供实用的优化指导。我们旨在解决以下问题：

(1) 建模能否帮助确立内核真正的“潜在性能上限”，以区别于次优配置的噪声？

(2) 这个估计的“上限”能否作为参考，以识别系统性的利用不足并指导优化工作？

## A. 通过分位数损失定义潜在上限

为了解决这个问题，我们采用了分位数回归 [23] 的原理。我们使用 Section V-C 中描述的相同特征集和目标（执行效率）训练了一个 MLP 模型，但采用 Quantile Loss 作为训练目标。我们专门将模型配置为预测第 80 百分位数（P80）。这种方法提供了对性能上限的统计上鲁棒的估计，与 P90 等较高分位数相比，它对极端异常值或测量噪声不太敏感。

通过以 P80 为目标，该模型被有效地训练以拟合前 20% 的性能数据点，捕获高性能配置的特征，同时系统地过滤掉后 80% 的次优结果。因此，得到的预测 $\hat { y } _ { p 8 0 }$ 并不代表典型的平均值。相反，它作为一个统计定义的潜在

表 IX
使用 SGLANG 和 VLLM 进行多 GPU 推理时 PIPEWEAVE 及基线的端到端性能预测 MAPE (%)。
<table><tr><td>框架</td><td>模型</td><td>数据集</td><td>硬件</td><td>Roofline</td><td>Linear</td><td>Habitat</td><td>Neusight</td><td>PIPEWEAVE</td></tr><tr><td rowspan="8">SGLang</td><td rowspan="8"></td><td></td><td>A100</td><td>48.6 59.2</td><td>42.7 43.3</td><td>47.3 44.9</td><td>45.0 30.4</td><td>2.4 9.1</td></tr><tr><td>arxiv_12</td><td>6000Ada H100</td><td>73.5</td><td>77.1</td><td>34.9</td><td>31.1</td><td>7.5</td></tr><tr><td></td><td>PRO6000</td><td>46.5</td><td>15.2</td><td>39.6</td><td>56.6</td><td>9.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>A100</td><td>49.0</td><td>44.6</td><td>35.5</td><td>45.9</td><td>3.9</td></tr><tr><td>splitwise_48</td><td>6000Ada</td><td>53.2</td><td>51.5 49.6</td><td>35.1 33.3</td><td>38.8 60.2</td><td>7.9</td></tr><tr><td></td><td>H100 PRO6000</td><td>62.4 47.1</td><td>29.8</td><td>36.9</td><td>18.5</td><td>16.5 10.9</td></tr><tr><td></td><td></td><td>45.2</td><td></td><td>50.1</td><td></td><td></td></tr><tr><td rowspan="6">SGLang</td><td rowspan="6">Llama3.1-70B (TP=4)</td><td>arxiv_16</td><td>A100 H100</td><td>78.6</td><td>30.3 69.4</td><td>45.6</td><td>76.5 34.5</td><td>2.6 5.4</td></tr><tr><td></td><td>A100</td><td></td><td></td><td>57.5</td><td>55.6</td><td>2.1</td></tr><tr><td>splitwise_64</td><td>H100</td><td>46.0 82.2</td><td>26.2 64.8</td><td>56.1</td><td>47.2</td><td>7.0</td></tr><tr><td></td><td>H20</td><td></td><td>70.5</td><td>54.4</td><td></td><td></td></tr><tr><td>arxiv_16</td><td>H800</td><td>90.1 66.7</td><td>46.7</td><td>25.9</td><td>27.1 17.2</td><td>4.0 12.3</td></tr><tr><td></td><td>H20</td><td>91.8</td><td>74.3</td><td>62.7</td><td>20.4</td><td>3.7</td></tr><tr><td rowspan="6"></td><td rowspan="6"></td><td>splitwise_64</td><td>H800</td><td>69.8</td><td>51.5</td><td>29.1</td><td>26.1</td><td>10.7</td></tr><tr><td>arxiv_16</td><td>H20</td><td>69.1</td><td>45.2</td><td>54.6</td><td>0.5</td><td>3.0</td></tr><tr><td></td><td>H800</td><td>25.7</td><td>60.8</td><td>9.0</td><td>16.9</td><td>0.7</td></tr><tr><td>splitwise_64</td><td>H20</td><td>76.7</td><td>64.7</td><td>72.6</td><td>19.1</td><td>2.3</td></tr><tr><td></td><td>H800</td><td>49.5</td><td>67.1</td><td>38.6</td><td>23.7</td><td>9.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

性能上限，代表了内核实现的一个高但现实可达到的目标。

## B. 诊断性能差距

我们首先验证 P80 模型作为诊断工具的有效性。训练好的模型用于预测 P80 性能上限 yˆ<sub>p80</sub>，并将其应用于整个 Fused MoE 数据集（Section V-B）。然后，我们通过计算预测上限与实际性能之间的差异来衡量性能差距：

$$
\mathrm { p e r f \_ g a p } = \hat { y } _ { p 8 0 } - y _ { \mathrm { a c t u a l } }
$$

这里，y<sub>actual</sub> 表示执行效率（Section V-C）。

图 8 展示了对这些差距的综合分析。每个垂直柱状图代表一个硬件平台，柱状图的高度表示在该平台上识别出的性能不佳点的总数。折线图展示了在所有评估的硬件平台上汇总的性能差距的累积分布函数（CDF）。该图揭示了两个关键发现。首先，CDF 折线确认了一种“长尾”模式。我们观察到，虽然绝大多数配置的性能接近其潜力，但大约 80% 的点的性能差距低于 0.1。基于这一观察，我们将“性能不佳点”定义为性能差距 > 0.1 的任何配置。其次，柱状图指出了这些性能不佳点发生的位置，揭示了显著的低效问题具有硬件特定性。例如，A40 GPU 表现出最大的差异，占据了绝大多数的低效问题，共有 921 个不同的性能不佳点（占所有 A40 样本的 30.4%）。这清楚地表明，该 kernel 当前的配置逻辑不适用于这种特定的硬件架构。形成鲜明对比的是，H20 取得了接近最优的结果，表现出零个此类点。

![](images/77a2447b10020214b88fcf0ccc86662df320ce35de4a16d9e452dabb3362180a.jpg)  
图 8. 性能差距分析。差距分布的 CDF（折线）和各硬件的“性能不佳点”（差距 > 0.1）数量（柱状图）。

## C. 通过调优参数缩小性能差距

在 Section VII-B 中，我们应用我们的 P80 模型成功识别了“性能不佳点”。我们现在验证这些差距是可操作的，并表明了系统性的优化潜力。为每个 GPU：A40、L20、A100 和 H800 选择了大约 70 个独特的“性能不佳点”配置。对于这些目标案例，通过在三个参数上进行暴力自动调优来进行优化：BLOCK\_SIZE、num\_stages 和 num\_warps。

为了明确验证我们的统计诊断方法与实际优化结果的关系，Table X 展示了性能不佳点的系统密度与所实现的调优收益之间的关系。观察到明显的正相关性（Pearson 相关系数为 0.86）：具有较多性能不佳点的硬件平台在调优后获得了更大的几何平均加速比。这一结果证实，我们的统计诊断有效地反映了真实的优化潜力，并能指导调优工作转向具有最大预期收益的配置。

此外，Figure 9 展示了这些被诊断出的性能不佳点的实际影响。在应用暴力自动调优后，平均性能差距显著缩小，特别是在最初表现出较大低效问题的硬件上。例如，A40 上的平均差距从 0.187 降至 0.083，L20 上的平均差距从 0.274 降至 0.215。相比之下，A100 和 H800 上的改进较为有限，因为它们的基线配置已经接近估计的性能上限。尽管有这些改进，通常仍会存在残余差距。这表明某些低效问题无法仅通过参数调优来完全消除，而是可能源于更深层次的因素，例如 kernel 的结构设计或 Triton 编程模型 [12], [58] 的固有限制。

TABLE X  
各 GPU 上的加速比与性能不佳点对比。
<table><tr><td>GPU</td><td>性能不佳点</td><td>几何平均加速比</td></tr><tr><td>A40</td><td>921</td><td>1.61×</td></tr><tr><td>L20</td><td>728</td><td>1.12×</td></tr><tr><td>A100</td><td>488</td><td>1.06×</td></tr><tr><td>H800</td><td>340</td><td>1.03×</td></tr></table>

![](images/b293323980823ddfdcff3ce71e29d2d0e95d9b40660761f852df965d21972af5.jpg)  
图 9. 四个 GPU 平台上模型引导优化前后的性能差距分布。

## VIII. 相关工作

## A. GPU 性能建模

关于 GPU 性能建模的研究大致分为三类：周期精确模拟器 [4], [22], [66]、分析模型 [6], [19], [25], [78] 和数据驱动方法 [26], [76]。尽管这些方法很有用，但它们存在固有的权衡。高保真度的周期精确模拟器计算成本高昂，且难以在新硬件上泛化。更快的替代方案——分析和数据驱动模型——通常面临精度有限、硬件特定约束以及粗粒度假设的问题，这些假设忽略了诸如融合 kernel 耦合等复杂行为，限制了它们的泛化能力。PIPEWEAVE 旨在通过将原则性分析建模与数据驱动技术的速度和灵活性相结合来解决这些限制，从而实现高保真度和广泛的泛化能力。

Table XI 总结了具有代表性的 GPU 性能模型。与以前依赖经验黑盒学习或粗粒度分析抽象（例如，Tile 级吞吐量和静态波调度）的方法不同，PIPEWEAVE 通过微架构感知的、流水线级别的公式化推进了灰盒范式。通过显式捕获异构流水线需求和动态调度，它实现了高精度和跨架构的可移植性。

TABLE XI  
微架构建模能力对比。
<table><tr><td>维度</td><td>Habitat</td><td>Neusight</td><td>PIPEWEAVE (本文)</td></tr><tr><td>建模策略</td><td>黑盒</td><td>宏观灰盒</td><td>微架构灰盒</td></tr><tr><td>粒度</td><td>Kernel 级</td><td>Tile 级</td><td>流水线级</td></tr><tr><td>硬件保真度</td><td>GPU</td><td>SM</td><td>流水线</td></tr><tr><td>调度语义</td><td>N/A</td><td>静态波假设</td><td>动态 SM 调度</td></tr><tr><td>Kernel 类型</td><td>基础 Kernel</td><td>基础 Kernel</td><td>融合与基础 Kernel</td></tr><tr><td>跨架构通用性</td><td>低</td><td>中</td><td>高</td></tr><tr><td>预测精度</td><td>低</td><td>中</td><td>高</td></tr></table>

## B. 网络模拟

随着计算在多节点集群上的扩展，数据中心互连的精确建模变得越来越重要。通用网络模拟器 [17], [69] 提供细粒度的数据包级控制，以评估拥塞、路由行为和协议交互。较新的专注于 AI 的网络模拟框架 [57], [73] 针对 All-Reduce 和 All-Gather 等通信模式，以及大规模、通信密集型分布式工作负载的性能特征。

## C. 系统级模拟

除了单个组件之外，大量研究集中在端到端 DNN 系统性能模拟上。这些工具对大型模型的计算、通信和调度策略之间的复杂交互进行建模。在分布式训练中，模拟器 [5], [57], [73] 评估各种并行策略，如数据并行、流水线并行、张量并行。在推理中，特别是在 LLM 服务中，工具 [1], [7], [14] 模拟动态批处理和调度策略。我们的 PIPEWEAVE 框架不仅包含用于推理的系统级模拟器，还提供了以前系统级工具所需的高保真度、可插拔的 GPU 计算模型。

## IX. 结论

我们提出了 PIPEWEAVE，这是一个统一框架，将知识引导的分析建模与数据驱动的学习相融合，以实现高保真的 GPU 性能预测。通过将 kernel 分解为基本的流水线需求，并通过 MLP 捕获复杂的运行时交互，PIPEWEAVE 在多样化的 kernel、工作负载和硬件代际间展现出了最先进的准确性和泛化能力。除了预测之外，我们还验证了其在诊断硬件特定低效问题和指导针对性优化方面的实用价值。

未来工作将聚焦于两个主要方向。首先，我们将把 PIPEWEAVE 扩展到复杂的分布式环境，纳入对多节点集群和高级并行策略（如 Expert Parallelism (EP)）的支持。其次，我们计划通过开发自动化工具来拓宽我们的模型引导优化方法，以检测性能瓶颈并增强更多生产 kernel 的配置逻辑。

## X. 致谢

我们感谢匿名审稿人提出的建设性反馈和宝贵建议，这些意见改进了本工作。

我们也感谢同事们的有益讨论。本工作使用了 LLM 进行文本润色和代码生成。本工作得到了阿里巴巴集团技术基础设施与可靠性工程（TRE）通过阿里巴巴创新研究计划和阿里巴巴研究实习生计划的支持。

## REFERENCES

[1] A. Agrawal, N. Kedia, J. Mohan, A. Panwar, N. Kwatra, B. S. Gulavani, R. Ramjee, and A. Tumanov, “Vidur: A large-scale simulation framework for llm inference,” in Proceedings of the 2024 Conference on Machine Learning and Systems (MLSys ’24), 2024, also available at arXiv:2405.05465. [Online]. Available: https://arxiv.org/abs/2405.05465

[2] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, A. Tumanov, and R. Ramjee, “Taming throughput-latency tradeoff in llm inference with sarathi-serve,” in Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’24). Santa Clara, CA, USA: USENIX Association, 2024, also available on arXiv:2403.02310. [Online]. Available: https://arxiv.org/abs/2403.02310

[3] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, B. Hui, L. Ji, M. Li, J. Lin, R. Lin, D. Liu, G. Liu, C. Lu, K. Lu, J. Ma, R. Men, X. Ren, X. Ren, C. Tan, S. Tan, J. Tu, P. Wang, S. Wang, W. Wang, S. Wu, B. Xu, J. Xu, A. Yang, H. Yang, J. Yang, S. Yang, Y. Yao, B. Yu, H. Yuan, Z. Yuan, J. Zhang, X. Zhang, Y. Zhang, Z. Zhang, C. Zhou, J. Zhou, X. Zhou, and T. Zhu, “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023, arXiv:2309.16609.

[4] A. Bakhoda, G. L. Yuan, W. W. L. Fung, H. Wong, and T. M. Aamodt, “Analyzing cuda workloads using a detailed gpu simulator,” in IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). IEEE, 2009, pp. 163–174.

[5] J. Bang, Y. Choi, M. Kim, Y. Kim, and M. Rhu, “vTrain: A simulation framework for evaluating cost-effective and computeoptimal large language model training,” in Proceedings of the 57th IEEE/ACM International Symposium on Microarchitecture (MICRO 2024). IEEE / ACM, 2024, pp. 153–167. [Online]. Available: https://arxiv.org/abs/2312.12391

[6] S. Cao, J. Wu, J. Chen, H. An, and Z. Yu, “Amali: An analytical model for accurately modeling llm inference on modern gpus,” in Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA ’25). ACM, 2025, pp. 1495–1508.

[7] J. Cho, M. Kim, H. Choi, G. Heo, and J. Park, “Llmservingsim: A hw/sw co-simulation infrastructure for llm inference serving at scale,” in 2024 IEEE International Symposium on Workload Characterization (IISWC), 2024, pp. 1–12.

[8] A. Cohan, F. Dernoncourt, D. S. Kim, T. Bui, S. Kim, W. Chang, and N. Goharian, “A discourse-aware attention model for abstractive summarization of long documents,” in Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Volume 2 (Short Papers). New Orleans, Louisiana: Association for Computational Linguistics, Jun. 2018, pp. 615–621. [Online]. Available: https://aclanthology.org/N18-2097/

[9] D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, Z. Xie, Y. K. Li, P. Huang, F. Luo, C. Ruan, Z. Sui, and W. Liang, “Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Bangkok, Thailand: Association for Computational Linguistics, 2024, pp. 1280–1297. [Online]. Available: https://aclanthology.org/2024.acl-long.70

[10] T. Dao, “FlashAttention-2: Faster attention with better parallelism and work partitioning,” in International Conference on Learning Representations (ICLR), 2024.

[11] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, “FlashAttention: Fast´ and memory-efficient exact attention with IO-awareness,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[12] J. H. Davis et al., “Taking gpu programming models to task for performance: an empirical study,” in Proceedings of ICS 2025, 2025, demonstrates that abstraction and language-level limitations cause persistent, architecture-dependent performance gaps. [Online]. Available: https://hpcrl.github.io/ICS2025- webpage/program/Proceedings ICS25/ics25-63.pdf

[13] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal of Machine Learning Research, vol. 23, pp. 1–39, 2022.

[14] Y. Feng, X. Tan, K. H. Sew, Y. Jiang, Y. Zhu, and H. Xu, “Simulating the next generation of llm inference systems,” in Proceedings of the 4th Workshop on Practical Adoption Challenges of ML for Systems (PACMI ’25). ACM, 2025.

[15] FlashInfer Team, “Kv cache layout tutorial,” https://docs.flashinfer.ai/ tutorials/kv layout.html, 2025, accessed: 2025-10-27.

[16] Google DeepMind, “Gemini 2.5: Expanding the Capabilities of Multimodal AI Models,” https://blog.google/technology/google-deepmind/ gemini-model-thinking-updates-march-2025/, 2025, accessed: Nov. 2025.

[17] T. R. Henderson, M. Lacage, G. F. Riley, C. Dowell, and J. Kopena, “Network simulations with the ns-3 simulator,” in SIGCOMM Demonstration, 2008. [Online]. Available: https://www.nsnam.org/

[18] S. Hong and H. Kim, “An analytical model for a gpu architecture with memory-level and thread-level parallelism awareness,” in Proceedings of the 36th Annual International Symposium on Computer Architecture (ISCA ’09). ACM, 2009, pp. 152–163.

[19] J.-C. Huang, J. H. Lee, H. Kim, and H.-H. S. Lee, “Gpumech: Gpu performance modeling technique based on interval analysis,” in 2014 47th Annual IEEE/ACM International Symposium on Microarchitecture, 2014, pp. 268–279.

[20] Y. Ji, W. Li, X. Shen, and X. Shen, “Dynamic thread block scheduling for gpu-based computing,” in Proceedings of the 22nd International Conference on Parallel Architectures and Compilation Techniques (PACT ’13). IEEE, 2013, pp. 375–386.

[21] A. Jog, P. Nadkarni, O. Kayiran, R. Das, M. Kandemir, O. Mutlu, V. Narayanan, and C. R. Das, “Owl: Cooperative thread array aware scheduling techniques for improving gpgpu performance,” in Proceedings of the 43rd Annual International Symposium on Computer Architecture (ISCA ’16). IEEE, 2016, pp. 395–406.

[22] M. Khairy, Z. Shen, T. M. Aamodt, and T. G. Rogers, “Accel-sim: An extensible simulation framework for validated gpu modeling,” in 47th Annual International Symposium on Computer Architecture (ISCA). IEEE/ACM, 2020, pp. 473–486.

[23] R. Koenker and G. Bassett, “Regression quantiles,” Econometrica, vol. 46, no. 1, pp. 33–50, 1978.

[24] A. Kuzmin, M. van Baalen, Y. Ren, M. Nagel, J. Peters, and T. Blankevoort, “Fp8 quantization: The power of the exponent,” in Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2022/hash/ 5e07476b6bd2497e1fbd11b8f0b2de3c-Abstract-Conference.html

[25] J. Lee, Y. Ha, S. Lee, J. Woo, J. Lee, H. Jang, and Y. Kim, “Gcom: a detailed gpu core model for accurate analytical modeling of modern gpus,” in Proceedings of the 49th Annual International Symposium on Computer Architecture (ISCA ’22). Association for Computing Machinery, 2022, pp. 424–436.

[26] S. Lee, A. Phanishayee, and D. Mahajan, “Forecasting gpu performance for deep learning training and inference,” in Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, ser. ASPLOS ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 493–508. [Online]. Available: https://doi.org/10. 1145/3669940.3707265

[27] A. H. Less Wright, “Deep Dive on CUTLASS Ping-Pong GEMM Kernel,” https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/, November 2024, accessed: 2025-10-18.

[28] A. Li, S. L. Song, W. Liu, X. Liu, A. Kumar, and H. Corporaal, “Locality-aware cta clustering for modern gpus,” in Proceedings of the 22nd International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS ’17). Xi’an, China: ACM, 2017, pp. 297–311. [Online]. Available: https://doi.org/10.1145/3037697.3037709

[29] Y. Li, Y. Sun, and A. Jog, “Path forward beyond simulators: Fast and accurate gpu execution time prediction for dnn workloads,” in Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture, ser. MICRO ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 380–394. [Online]. Available: https://doi.org/10.1145/3613424.3614277

[30] A. Liu, S. L. Song, W. Liu, A. Kumar, and H. Corporaal, “Greedy dual-size thread block scheduling for gpus,” in Proceedings of the 42nd

International Conference on Parallel Processing (ICPP ’13). IEEE, 2013, pp. 320–329.

[31] X. Liu, A. Li, J. Yang, A. Nukada, B. Ren, and W.-m. W. Hwu, “Locality analysis for gpgpu programs,” in Proceedings of the International Symposium on Microarchitecture (MICRO ’12). IEEE, 2012, pp. 63–74.

[32] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations (ICLR), 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7

[33] Modal Labs, “Streaming Assembler (SASS) — GPU Glossary,” https: //modal.com/gpu-glossary/device-software/streaming-assembler, 2025, accessed: 2025-10-20.

[34] J. Nickolls, “Gpu parallel computing architecture and cuda programming model,” in 2007 IEEE Hot Chips 19 Symposium (HCS), 2007, pp. 1–12.

[35] NVIDIA Corporation, NVIDIA CUDA C Programming Guide, 2009, version 2.3. [Online]. Available: https://docs.nvidia.com/cuda/cuda-cprogramming-guide/

[36] NVIDIA Ampere Architecture Whitepaper (GA10x/A100), 2020, “NVIDIA A100 Tensor Core GPU Architecture In-Depth” and “NVIDIA Ampere GA102 GPU Architecture” Whitepapers. [Online]. Available: https://www.nvidia.com/content/PDF/ nvidia-ampere-architecture-whitepaper.pdf

[37] ——, NVIDIA Ada GPU Architecture Whitepaper (Ada Lovelace), 2022, “NVIDIA Ada GPU Architecture” V2.02. [Online]. Available: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidiaada-gpu-architecture.pdf

[38] ——, NVIDIA Hopper GPU Architecture Whitepaper (H100 Tensor Core GPU), 2022, “NVIDIA H100 Tensor Core GPU Architecture” Whitepaper V1.01. [Online]. Available: https://advancedclustering.com/ wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf

[39] ——, “Cuda gpus,” 2024. [Online]. Available: https://developer.nvidia. com/cuda-gpus

[40] ——, CUTLASS: CUDA Templates for Linear Algebra Subroutines – Scaled Matrix Multiplication, 2024, version 3.5, Persistent and ScaledMM kernels. [Online]. Available: https://github.com/NVIDIA/ cutlass

[41] ——, DeepGEMM: High-Performance FP8 GEMM Kernels for Transformer Inference, 2024, fP8 GEMM library for Hopper and Ada architectures. [Online]. Available: https://github.com/NVIDIA/ DeepGEMM

[42] ——, “Efficient gemm in cutlass,” https://docs.nvidia.com/cutlass/media/ docs/cpp/efficient gemm.html, oct 2024, accessed: 2025-10-27. CUT-LASS Documentation.

[43] ——, “Matrix multiplication,” https://docs.nvidia.com/deeplearning/ performance/dl-performance-matrix-multiplication/index.html, 2024, accessed: 2025-10-27. Part of the NVIDIA Deep Learning Performance Guide.

[44] ——, NVIDIA Blackwell Architecture Whitepaper (RTX/AI Data-Center), 2024, “NVIDIA RTX Blackwell GPU Architecture” Whitepaper V1.1. [Online]. Available: https://images.nvidia.com/aem-dam/ Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf

[45] ——, Transformer Engine: FP8 Training and Inference, 2024, version 1.6, Apache License 2.0. [Online]. Available: https://github.com/ NVIDIA/TransformerEngine

[46] ——, CUDA C++ Best Practices Guide, 2025, version 13.0. [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

[47] ——, CUDA C Programming Guide, 2025, version 13.0. [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[48] ——, “CUDA Compiler Driver NVCC Documentation,” https://docs. nvidia.com/cuda/cuda-compiler-driver-nvcc/, 2025, accessed: 2025-10- 20.

[49] ——, CUDA Driver API Documentation, NVIDIA Corporation, 2025, cUDA Toolkit v13.0.97; last updated Oct 2, 2025. [Online]. Available: https://docs.nvidia.com/cuda/cuda-driver-api/

[50] ——, “CUTLASS Documentation,” https://docs.nvidia.com/cutlass/ index.html, 2025, accessed: 2025-10-18.

[51] ——, “NVIDIA cuBLAS Library Documentation,” https://docs.nvidia. com/cuda/cublas/, 2025, accessed: 2025-10-18.

[52] ——, “NVIDIA Developer Forums,” https://forums.developer.nvidia. com, 2025, accessed: 2025-10-20.

[53] ——, “NVIDIA Nsight Compute Documentation,” https://docs.nvidia. com/nsight-compute, 2025, accessed: 2025-10-20.

[54] ——, “Parallel Thread Execution ISA Version 9.0 Documentation,” https://docs.nvidia.com/cuda/parallel-thread-execution/, 2025, accessed: 2025-10-20.

[55] P. Patel, E. Choukse, C. Zhang, A. Shah, <sup>´</sup>I. Goiri, S. Maleki, and R. Bianchini, “Splitwise: Efficient generative llm inference using phase splitting,” in Proceedings ofthe 51st Annual International Symposium on Computer Architecture (ISCA), Buenos Aires, Argentina, 2024. [Online]. Available: https://dl.acm.org/doi/10.1109/ISCA59077.2024.00019

[56] PyTorch Team, “Pytorch profiler: Performance analysis tool for deep learning,” https://pytorch.org/docs/stable/profiler.html, 2024, accessed: 2025-11-04.

[57] S. Rashidi, S. Sridharan, S. Srinivasan, and T. Krishna, “Astra-sim: Enabling sw/hw co-design exploration for distributed deep learning training platforms,” in 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). IEEE, 2020, pp. 81–92.

[58] B. Ringlein, T. Parnell, and R. Stoica, “Gpu performance portability needs autotuning,” arXiv preprint, 2025, shows that residual performance gaps often stem from fundamental kernel design limits rather than parameter tuning alone. [Online]. Available: https://arxiv.org/abs/2505. 03780

[59] SGLang Project, “SGLang: Fast Serving Framework for Large Language Models and Vision-Language Models,” https://github.com/sgl-project/ sglang, 2024, version 0.5.3, Apache License 2.0.

[60] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao, “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision,” https://arxiv.org/abs/2407.08608, July 2024, arXiv:2407.08608 [cs.LG].

[61] N. Shazeer, “Glu variants improve transformer,” arXiv preprint arXiv:2002.05202, 2020. [Online]. Available: https://arxiv.org/abs/2002. 05202

[62] N. Shazeer et al., “Outrageously large neural networks: The sparselygated mixture-of-experts layer,” in International Conference on Learning Representations (ICLR), 2017.

[63] H. Shen, N. Mellempudi, X. He, Q. Gao, C. Wang, and M. Wang, “Efficient post-training quantization with fp8 formats,” in Proceedings of the 6th Conference on Machine Learning and Systems (MLSys 2024), 2024, arXiv preprint arXiv:2309.14592v2. [Online]. Available: https://proceedings.mlsys.org/paper files/paper/2024/ hash/dea9b4b6f55ae611c54065d6fc750755-Abstract-Conference.html

[64] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, B. Chen, P. Liang, C. Re, I. Stoica, and C. Zhang, “Flexgen: High-throughput generative´ inference of large language models with a single gpu,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 31 094–31 116. [Online]. Available: https://proceedings.mlr.press/v202/sheng23a.html

[65] S. L. Song, A. Li, X. Liu, A. Kumar, and H. Corporaal, “Understanding the impact of cta scheduling on gpu performance,” IEEE Transactions on Parallel and Distributed Systems, vol. 27, no. 6, pp. 1738–1751, 2016.

[66] Y. Sun, T. Baruah, S. A. Mojumder, S. Dong, X. Gong, S. Treadway, Y. Bao, S. Hance, C. McCardwell, V. Zhao, and et al., “Mgpusim: Enabling multi-gpu performance modeling and optimization,” in Proceedings of the 46th Annual International Symposium on Computer Architecture (ISCA). ACM, 2019, pp. 197–209.

[67] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez,\` A. Joulin, Edouard Grave, and G. Lample, “Llama: Open and efficient<sup>´</sup> foundation language models,” arXiv preprint arXiv:2302.13971, 2023, arXiv:2302.13971.

[68] Triton Team, “Triton Language Documentation,” https://triton-lang.org/ main/index.html, 2025, accessed: 2025-10-20.

[69] A. Varga and R. Hornig, “An Overview of the OMNeT++ Simulation Environment,” in Proceedings of the 1st International Conference on Simulation Tools and Techniques for Communications, Networks and Systems, ser. SIMUTOOLS ’08. ICST, 2008, pp. 1–10.

[70] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, ser. NIPS’17. Red Hook, NY, USA: Curran Associates Inc., 2017, p. 6000–6010.

[71] A. Vladimirov, “CUTLASS Tutorial: Persistent Kernels and Stream-K,” https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-andstream-k/, 2024, accessed: 2025-10-18.

[72] vLLM Project, “vLLM: A High-Throughput and Memory-Efficient Inference and Serving Engine for Large Language Models,” https:

//github.com/vllm-project/vllm, 2025, version 0.11.0 (latest Oct 2 2025), Apache License 2.0.

[73] X. Wang, Q. Li, Y. Xu, G. Lu, D. Li, L. Chen, H. Zhou, L. Zheng, S. Zhang, Y. Zhu, Y. Liu, P. Zhang, K. Qian, K. He, J. Gao, E. Zhai, D. Cai, and B. Fu, “Simai: Unifying architecture design and performance tuning for large-scale large language model training with scalability and precision,” in Proceedings of the 22nd USENIX Symposium on Networked Systems Design and Implementation (NSDI ’25). Philadelphia, PA, USA: USENIX Association, 2025, pp. 541–558. [Online]. Available: https://www.usenix.org/conference/ nsdi25/presentation/wang-xizheng-simai

[74] S. Williams, A. Waterman, and D. Patterson, “Roofline: an insightful visual performance model for multicore architectures,” Commun. ACM, vol. 52, no. 4, p. 65–76, Apr. 2009. [Online]. Available: https://doi.org/10.1145/1498765.1498785

[75] Z. Ye, L. Chen, R. Lai, W. Lin, Y. Zhang, S. Wang, T. Chen, B. Kasikci, V. Grover, A. Krishnamurthy, and L. Ceze, “Flashinfer: Efficient and customizable attention engine for llm inference serving,” arXiv preprint arXiv:2501.01005, 2025. [Online]. Available: https: //arxiv.org/abs/2501.01005

[76] G. X. Yu, Y. Gao, P. Golikov, and G. Pekhimenko, “Habitat: A runtimebased computational performance predictor for deep neural network training,” in USENIX Annual Technical Conference, 2021. [Online]. Available: https://api.semanticscholar.org/CorpusID:236992542

[77] B. Zhang and R. Sennrich, “Root mean square layer normalization,” CoRR, vol. abs/1910.07467, 2019. [Online]. Available: http://arxiv.org/ abs/1910.07467

[78] H. Zhang, A. Ning, R. B. Prabhakar, and D. Wentzlaff, “Llmcompass: Enabling efficient hardware design for large language model inference,” in Proceedings ofthe 51st Annual International Symposium on Computer Architecture, ser. ISCA ’24. IEEE Press, 2025, p. 1080–1096. [Online]. Available: https://doi.org/10.1109/ISCA59077.2024.00082

[79] J. Zhang and A. Jog, “Tlp-aware cooperative scheduling for efficient gpu memory system utilization,” in Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA ’17). ACM, 2017, pp. 93–104.

[80] H. Zhu, A. Phanishayee, and G. Pekhimenko, “Daydream: Accurately estimating the efficacy of optimizations for dnn training,” in Proceedings of the 2020 USENIX Annual Technical Conference (USENIX ATC). USENIX Association, 2020, pp. 337–352. [Online]. Available: https://www.usenix.org/conference/atc20/presentation/zhu-hongyu