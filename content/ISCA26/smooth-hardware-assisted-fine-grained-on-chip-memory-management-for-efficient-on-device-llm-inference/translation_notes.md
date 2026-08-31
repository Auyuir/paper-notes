# SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference 原文翻译

![](images/3464bb9da07bf92faabdd7e7b32d34d53520f81e600252e90da99597a0c5a87a.jpg)

# SMOOTH：面向高效端侧 LLM 推理的硬件辅助细粒度片上内存管理

Seulki Kim   
DGIST   
韩国大邱   
skkim@dgist.ac.kr   
Hwanjun Lee   
DGIST   
韩国大邱   
lee.hwanjun@dgist.ac.kr

Bokyeong Kim 三星研究院 韩国首尔 bokyeong.kim@samsung.com

Sungju Kim   
延世大学   
韩国首尔   
sungju.kim@yonsei.ac.kr   
Kyeonghyeon Ryu   
DGIST   
韩国大邱   
khryu@dgist.ac.kr   
Yunhyeong Jeon   
DGIST   
韩国大邱   
yhjeon@dgist.ac.kr   
Yeji Jung   
DGIST   
韩国大邱   
jung.yeji@dgist.ac.kr   
Daehoon Kim<sup>\*</sup>   
延世大学   
韩国首尔   
daehoonkim@yonsei.ac.kr

摘要——在移动设备上直接运行大型语言模型（LLM）的需求日益增长，加剧了在严格的内存和带宽约束下实现高效端侧推理的需求。尽管编译器级别的优化（如内存分块和基于生命周期的分配）能够提升片上 SRAM 的利用率，但它们在应对自回归解码阶段中计算密集型与 I/O 密集型交替所产生的突发内存流量和内存碎片问题方面仍然效果有限。本文提出 SMOOTH，一种硬件辅助的片上内存管理框架，可在运行时动态优化便笺式存储器的使用。首先，一种细粒度的、基于块的分配与预加载方案提升了有效的 SRAM 利用率，并充分利用了空闲的内存带宽。其次，一种硬件驱动的早期回收机制利用缓冲区级别的信号及时释放未使用的内存块，从而实现更激进、更及时的预加载。我们使用 Verilog 实现了 SMOOTH，并将其集成到 LLMCompass（ScaleSim 的 LLM 优化扩展）中，以进行周期精确的评估。实验结果表明，与先前在内存受限的移动 SoC 上的基线方法相比，SMOOTH 将首 Token 延迟（TTFT）降低了高达 59.2%，将末 Token 延迟（TTLT）降低了高达 73.0%，与最先进的基线方法相比平均能耗降低了高达 51.2%。

关键词——大型语言模型，端侧推理，SoC，便笺式存储器管理。

## I. 引言

大型语言模型（LLM）已成为从自然语言助手 [1]– [9] 到个性化推荐 [10]–[13] 等应用中不可或缺的一部分。最近，这些模型的端侧部署开始兴起，以实现实时响应并保护用户隐私 [14]– [16]。然而，尽管服务器级系统在 LLM 推理期间也会遇到内存瓶颈，但在移动硬件上这种限制要严重得多。LLM 推理主要由受限于 I/O 的通用矩阵-向量乘法（GEMV）操作主导，而移动 SoC 有限的 SRAM（2–8 MB）和低带宽的 LPDDR5（13–34 GB/s）加剧了这一瓶颈。因此，内存带宽迅速饱和，严重降低了端侧推理性能。LLM 固有的架构特征进一步加剧了这些问题。基于 Transformer 的模型在自回归解码期间交替进行非线性和线性操作 [17]–[19]。这导致了高度突发的内存流量：在计算阶段，内存带宽基本处于空闲状态，但当 GEMV 层加载大型权重时，带宽会被完全占满。这种阶段交替的行为导致移动硬件上的资源利用率长期低下，留下了大量未开发的余量。然而，这种高度突发的模式与现代编译器使用的静态、分块级内存规划根本不匹配：带宽余量出现在短暂且不可预测的窗口中，无法被固定的编译时调度所捕获。

现代深度学习编译器，包括 XLA [20] 和 TVM [21]，试图通过编译时优化（如内存分块、算子融合和基于生命周期的分配）来缓解内存瓶颈 [22]–[24]。虽然这些技术减少了内存占用，但它们本质上是静态的，无法适应依赖于运行时的行为，例如不断变化的 Key/Value (KV) 缓存大小或波动的执行时间。此外，移动 SoC 中的统一内存架构由于并发运行的 CPU 和 GPU 工作负载的竞争，会导致可用带宽动态波动。此外，实现最佳执行在很大程度上取决于运行时条件，因为不同用户请求的输入和输出 Token 长度差异很大。由于这些动态因素在编译时未知，静态编译器必须保守地固定分块大小。这经常与运行时条件不匹配，导致推理延迟严重退化超过 2 倍。从历史上看，暂存内存（SPM）避免了块级分配，因为每块的地址转换需要非平凡的元数据和访问开销，而且传统的 CNN/DNN 工作负载根本无法从中受益。由于 CNN 的分块形状规则且重用率高，内存碎片化极小，使得基于连续偏移量的 SPM 既足够又最优。因此，标准的 SPM 管理采用了粗粒度的方法，在张量或分块级别分配内存——通常为几十到几百千字节 [25], [26]。然而，这些假设对于 LLM 来说并不成立。融合的、特定于层的分块模式和长生命周期的中间缓冲区造成了严重的碎片化，而数据重用率低且不同操作间的分块形状各异。因此，粗粒度、连续的 SPM 根本无法适应解码器端 LLM 不规则、突发的内存行为。

为了弥补这一点，现代编译器依赖于日益复杂的生命周期分析和分配启发式方法。然而，这些机制仍然依赖于静态估计的生命周期，这些生命周期经常与实际的运行时行为发生偏离，并导致低效的重用。算子融合进一步延长了缓冲区的生命周期并加剧了碎片化，而软件管理的预加载由于缺乏对执行进度的可见性，无法利用短暂、瞬时的可用带宽窗口。为了量化这些限制的实际影响，我们在代表性的移动 SoC 上对 LLM 推理进行了性能分析，然后使用周期精确的模拟器重现了观察到的行为。作为基线，我们评估了一个编译器理想的 SPM，它假设具有完美的生命周期知识并最优地预加载所有非重叠的分块，但仍然缺乏运行时反馈并受到连续分配的约束。尽管有这种乐观的假设，编译器理想的 SPM 在 4K Token 时仍然由于碎片化导致的 SRAM 利用不足而增加了 32.7% 的停顿周期，远低于能够以字节级粒度放置数据的最优策略。因此，现有的编译器驱动的 SPM 技术从根本上说过于静态和粗粒度，无法满足端侧 LLM 推理对细粒度、时间敏感的内存需求。

为了解决这些限制，我们提出了 SMOOTH（SMOothing I/O Traffic with Hardware support，利用硬件支持平滑 I/O 流量），这是一个在运行时动态编排片上 SRAM 使用的硬件辅助 SPM 管理框架。与之前仅依赖静态编译器决策的 SPM 方法不同，SMOOTH 引入了现有 SPM 系统从根本上缺乏的两种能力。首先，SMOOTH 在块级粒度上虚拟化 SPM，同时结合了一种轻量级机制，可以绕过连续区域的地址转换。这产生了一种双模式混合设计：SMOOTH 在碎片化情况下使用细粒度的块虚拟化模式，但当块保持连续时，会自动切换到绕过转换的快速路径。这使得 SMOOTH 能够保留块级分配的灵活性（这对于 LLM 引起的碎片化至关重要），同时在常见的连续访问情况下提供与传统 SPM 相同的零开销行为。据我们所知，以前没有任何 SPM 架构支持块级虚拟化，也没有提供将细粒度块分配与零开销连续快速路径相结合的双模式机制——这些能力对于逐层执行反复引起碎片化并暴露短暂带宽余量的 LLM 工作负载来说是必不可少的。其次，SMOOTH 集成了一种硬件驱动的早期回收机制，该机制基于缓冲区级别的运行时信号而不是编译器估计的生命周期来释放块。这使得 SMOOTH 能够利用瞬时的带宽余量并支持细粒度预取，这些能力在静态 SPM 系统中是根本不具备的。这些能力共同实现了首个执行细粒度块放置和运行时驱动回收的 SPM 架构，这两个属性是 LLM 根本需要的，但在之前所有的 SPM 设计中都是缺失的。

我们使用 LLMCompass [27]（ScaleSim [28] 的 LLM 优化扩展）实现了 SMOOTH，并在 Verilog 中综合了其硬件逻辑，包括集成 DMC 的块表和基于位图的地址转换机制。实验结果表明，SMOOTH 显著改善了端侧 LLM 的延迟和效率：与具有代表性的基于编译器和硬件的基线相比，它将 Time-to-First-Token (TTFT) 降低了高达 59.2%，将 Time-to-Last-Token (TTLT) 降低了高达 73.0%。此外，SMOOTH 显著降低了模型推理能耗，与最先进的加速器相比，平均能耗降低高达 51.2%。

总而言之，本工作做出了以下贡献：

• 揭示移动 LLM 推理中内存效率低下的根本原因：我们量化了突发的内存需求和粗粒度的编译器决策（分块、生命周期分析、融合）是如何导致大量带宽余量未被利用并引起碎片化，从而导致严重停顿的。此外，我们描述了动态运行时因素——特别是统一内存架构中波动的可用带宽和变化的用户序列长度——是如何使静态分块大小变得严重次优的，有时会导致延迟退化高达 2.9 倍。

• 揭示静态 SPM 管理的根本局限性：通过分析和编译器理想实验，我们表明即使是假设完美的静态 SPM 分配器也无法适应动态运行时变化，并因碎片化而遭受高达 32.7% 的额外停顿周期。

• 设计面向 LLM 的硬件辅助、运行时感知的暂存存储器架构：SMOOTH 提供块级 SPM 虚拟化和硬件驱动的早期回收机制，实现细粒度预取并更有效地利用瞬时带宽空闲，最终改善移动 SoC 上的延迟和吞吐量。

## II. 背景

## A. 移动 SoC 中 LLM 推理的特征

基于 Decoder 的 LLM 推理在两个不同阶段运行。在 prompt 阶段，模型使用全自注意力机制对所有 Token 对同时处理整个输入序列。该阶段以通用矩阵-矩阵乘法（GEMM）运算为主，表现出高操作强度（High-OI），从根本上受计算能力限制。相反，在 Token 生成阶段，模型利用键值（KV）缓存将输入从 $d \times l$ 矩阵缩减为 d × 1 向量。因此，注意力矩阵从 $l \times l$ 缩小为 $l \ \times 1$，将工作负载转移到重复的通用矩阵-向量乘法（GEMV）。在整个生成阶段，执行在线性操作（如 QKV 和 W0 投影）和非线性操作（如 softmax 和激活函数）之间持续交替。线性操作需要移动大量 $d \times d$ 权重矩阵，而计算量相对较少，导致低操作强度（Low-OI），使其严重受 I/O 带宽限制。相比之下，非线性操作主要依赖向量算术吞吐量，导致存储带宽严重利用不足。随着序列长度 l 增长，这种迭代式 GEMV 模式急剧增加存储流量，在移动片上系统（SoC）环境的严格硬件约束下造成严重的带宽瓶颈。这种突发流量模式在我们对配备 8 MB 移动 NPU 和 LPDDR5 内存的 Qualcomm Hexagon V73 处理器的仿真中清晰可见（详见 § VI）。如图 1 所示，执行在计算阶段和 I/O 阶段之间剧烈交替，在 Low-OI 阶段由于频繁的片外 DRAM 访问而产生大量停顿周期。因此，计算受限和 I/O 受限操作之间的这种长期失衡严重降低了资源利用率，凸显了在边缘设备上实现低延迟 LLM 推理对高效片上存储管理的迫切需求。

![](images/d4839a8d864addbb72559590153c8ea316bcb08f7498eae9c72ba857010ba634.jpg)  
图 1. 移动 SoC 上 Transformer Decoder 的执行流程，其中高 OI 操作和低 OI 操作交替出现，导致突发存储流量和片外 DRAM 瓶颈。

![](images/a37861c52940bb0e5575e6239327bc1bb50f9529878fb046adb004076bf619db.jpg)  
图 2. 深度学习编译流程。

## B. 面向 LLM 的编译器驱动片上存储管理

基于 Transformer 的模型需要大量设备端存储，使 LLM 推理从根本上受存储带宽限制。现代加速器因此依赖分层存储系统，将快速但容量有限的片上 SRAM 与大容量片外 DRAM 相结合。SRAM 通常作为硬件 Cache 或软件 Scratchpad 进行管理 [29], [30]。硬件管理的 Cache 依赖空间和时间局部性，但 Transformer 工作负载表现出不规则的、低重用的访问模式。投影权重大，在解码步骤间很少被重用，且以编译器定义的 Tile 进行访问，这些 Tile 因层而异，而 KV-cache 访问随序列长度增长。因此，Cache 行表现出极小的时间局部性，且在每个 Tile 内仅有有限的空间重用，在紧张的片上容量下导致 Cache 效率低下。此外，与基于 SPM 的设计不同，Cache 缺乏对未来数据流和缓冲区生命周期的编译器可见性，无法在带宽密集阶段进行主动数据准备。相比之下，SPM 向软件或编译器暴露显式地址空间，实现对数据放置和重用的确定性控制。虽然这增加了软件复杂性和运行时开销，但与利用静态模型结构和数据流的编译器引导优化配合时，可实现更高的利用率 [31]–[33]。因此，基于 SPM 的架构在深度学习加速器中被广泛采用，其中编译器管理的数据编排缓解了片上容量限制。图 2 总结了深度学习编译工作流程。编译器首先将模型降级为基本操作并应用前端优化（如融合、Tiling、生命周期分析），然后构建捕获操作和 Tensor 元数据的中间表示（IR）。后端阶段利用此 IR 进行硬件感知优化，如存储分配和调度 [22]。在后端，存储分配对于将 Tensor 映射到有限的片上缓冲区至关重要 [34], [35]，在确保正确执行的同时最大化数据重用并减少传输。

传统后端分配器通常采用启发式或基于求解器的策略。启发式分配器，如 TFLite [36]、XLA [24] 和 TVM [25] 中使用的，提供快速编译，但在粗粒度上操作：存储在 Tensor 或 Tile 级别进行分配和重用，其中 Tile 通常跨越数十到数百千字节。然而，在 LLM 中，Tile 形状在不同层之间差异显著（如投影、前馈块和注意力头），这些不匹配经常在 SPM 中产生锯齿状间隙，导致大量内部碎片。此外，算子融合虽然提高了计算局部性，但延长了中间 Tensor 的生命周期，减少了在操作之间回收空间的机会，加剧了碎片化效应。基于求解器的方法，如 ILP 或基于约束的公式 [37], [38]，可以实现更高的利用率，但仍然继承了相同的 Tile 粒度限制，并产生显著的编译时开销。最终，现有的编译器驱动 SPM 方案从根本上是静态和粗粒度的：它们在 Tensor 或 Tile 级别分配和回收存储，缺乏对细粒度运行时进度的可见性，无法快速响应由突发 LLM 执行产生的短空闲窗口。这种粗粒度、静态分配与细粒度、时间敏感的带宽行为之间的差距，催生了一种新的片上存储管理方法，该方法能够以更小的单元虚拟化 SPM 容量，并将分配和回收与运行时执行协调。

Jetson S24 EdgeTPU  
![](images/7528f8a88cccd0d5891831a15a7ec9551dba4f96f7f1f3baeed1980a10b71180.jpg)  
图 3. CPU 和 GPU 并发工作负载下 NPU 的空闲存储带宽。

![](images/d2c33289a75a98b40e5d6eadf5ba70391c7ceb53322186baf3ffe39c874692c8.jpg)  
(a) 计算和存储带宽利用率，显示模型越大带宽需求越高。

![](images/459469c7045d1d73bbbc348b4a489074c4800984ac1313fa6f229c3f4be2e90c.jpg)  
(b) 解码层推理期间随时间变化的存储带宽利用率。  
图 4. LLaMA-3 在 Constrained-SoC 上的计算和存储带宽利用率。

## III. 研究动机

与高性能服务器或桌面级平台相比，移动 SoC 在更为紧张的内存和功耗限制下运行。它们仅提供一小部分片外 DRAM 带宽和片上缓冲区容量，这使得维持 LLM 推理的吞吐量变得极其困难。此外，端侧推理通常以 batch size 为 1 运行，导致计算单元利用率极低，使其特别容易受到内存停顿的影响。这种不平衡在 decoder 侧推理期间进一步加剧，此时加载投影权重和键值 (KV) 缓存等内存密集型操作会产生突发访问模式。我们首先对一款商用 SoC 进行了性能分析，并识别出 LLM 推理期间存在内存带宽瓶颈。为了进一步研究观察到的带宽利用率不足的根本原因，我们进行了一系列基于模拟器的实验。

## A. 动态运行时条件下静态编译器的低效性

移动环境的动态特性以及 LLM 的计算特性对静态编译器驱动的优化提出了重大挑战。在移动 SoC 中，硬件资源并非专用于单一应用，而是在多个并发运行的工作负载之间共享，导致资源可用性高度可变。这种环境多变性，加上 LLM 推理波动的需求，从根本上限制了离线静态优化策略的有效性。

![](images/24ec1652a02ec254f8d057bbc04ecbf0aa8d30df44d90216881a4040eb15714f.jpg)

![](images/8534adddac24a1e4217e5ff633116281f3866728aa45b3fea37a246888cb5960.jpg)

![](images/dd962c1e310b755371f2e2f193e3b482c2fa48e72f966c981fbe2e2c9cc6db73.jpg)  
<sub>(a) Operation intensity of</sub> (b) TinyLLaMA 的 Latency breakdown。 (c) GPT-3 在 batch size 为 1 时的 Latency breakdown。

Fig. 5. 移动平台上的 Operation intensity 和端到端 Latency breakdown。  
![](images/87b5cd7b0cc6f4ca1bcf3522f3217828bfd912df1565596b10dc14185152d001.jpg)

![](images/391e46124447737ccb266ad44f4a8f3f5ca5c40da536880da3dfa8b45b9564e0.jpg)  
(b) LLaMA2 (w4a8)  
Fig. 6. 通过模拟器模拟静态编译器以 N × K 大小的分块对每个权重进行分块时，Gemma-2 和 LLaMA2 随各 tile size 变化的 Latency variation。

系统引起的带宽可变性。移动 SoC 采用统一内存架构，其中 CPU、GPU 和 NPU 共享一个带宽受限的公共系统内存。在多程序执行下，由于并发运行的工作负载的竞争，每个处理单元可用的有效内存带宽会动态波动。图 3 通过表征商用移动平台（搭载 Snapdragon 8 Elite SoC 的 Samsung Galaxy S25+）上 NPU 观察到的空闲内存带宽，说明了这种可变性。结果表明，NPU 可用的空闲带宽根据并发 CPU 和 GPU 活动的存在与类型而有大幅变化。为了模拟真实的移动使用场景，我们采用了 Geekbench 6 [39]，并在执行两个代表性 CPU 工作负载和四个代表性 GPU 工作负载时测量了 NPU 的空闲内存带宽，从而捕捉实际多应用环境中内存带宽固有的可变性。这种可变性使得静态编译器调度难以持续选择与可用运行时带宽相匹配的 tile size。

工作负载引起的带宽可变性。不可预测性不仅限于空闲时段，还延伸到 LLM 推理期间的活跃内存带宽使用。我们在三个平台——Samsung Galaxy S24 Ultra、Google Edge TPU 和 NVIDIA Jetson AGX Orin——上进行了实验，以研究 LLM 的片外内存流量模式。由于商用移动设备上的硬件可见性有限，我们仅在 Jetson AGX Orin 上直接分析了计算和内存带宽，该平台配备 8 核 Cortex-A78AE CPU、2048 核 Ampere GPU 和 64 GB LPDDR5。为了近似移动 SoC 的性能特征，我们配置了一个特定环境，在此称为 Constrained-SoC。在 Constrained-SoC 中，我们通过限制内存控制器 (EMC) 频-

(b) 融合时的 SRAM 碎片化。

率至 512 MHz（对应于约 32 GB/s 的峰值带宽），并将 GPU 频率限制在 714 MHz（产生约 5.5 TFLOPS 的 FP16 吞吐量）来限制硬件资源。

图 4a 显示了三种不同模型大小的 LLaMA-3 在解码阶段的计算和内存带宽利用率。虽然计算利用率几乎保持不变，但随着模型大小的增加，吞吐量会下降，这表明内存带宽成为主要的性能瓶颈。图 4b 对 LLaMA 8B 上单个 Transformer decoder 层随时间变化的内存带宽利用率进行了更详细的分析。性能分析结果表明，内存带宽使用波动显著。Operation Intensity (OI) 定义为计算与内存流量的比率，有助于解释带宽波动 [40]。线性操作具有低 OI 并使内存带宽饱和，而非线性操作（例如 Softmax 和 GELU）具有高 OI 并使内存系统利用率不足。图 5b 和 5c 展示了使用 QNN [41]、edgetpu compiler [42] 和 NVCC [43] 编译的模型在 LLM 解码层分别在 S24、EdgeTPU 和 Jetson 等移动级设备上执行时的运行时分解。归因于高 OI 非线性操作的端到端执行时间比例在三个平台上保持一致。这种一致性验证了受限的 Jetson 配置准确反映了移动级设备的计算-内存行为。此外，结果表明 OI 特性在推理过程中动态演变，并且高 OI 操作构成了整体 Latency 的很大一部分，从而促使对它们进行预加载。

序列长度和可用内存带宽的可变性使得静态编译器驱动的优化难以持续选择与运行时条件相匹配的 tile size。首先，输入和输出 Token 的长度在不同用户请求之间差异很大。其次，如图 6 所示，即使在受片上内存容量限制的可行 tile size 范围内，模型推理 Latency 也会根据静态编译器确定的 tile size 发生显著变化，最高增加 2.9 倍。因此，实现最佳性能需要动态调整 tile size，以匹配运行时不断变化的序列长度和波动的可用内存带宽。然而，试图针对每种可能的序列长度和硬件条件静态优化或重新编译执行图是极其昂贵的。最近的研究强调，在移动边缘处理器上，针对单一变化的 prompt 大小优化执行图可能需要长达 11.5 秒 [44]。因此，离线编译器级优化无法有效地扩展以覆盖运行时遇到的所有动态序列长度和带宽波动。

## B. LLM 推理中编译器管理的片上内存的局限性

为了进一步调查观察到的带宽利用不足的潜在原因，我们进行了基于模拟的分析，以模拟编译器管理的 SPM 的行为。具体而言，我们 (1) 分析了由融合操作创建的 tile 的生命周期，以了解它们在推理过程中如何占用和碎片化片上内存，以及 (2) 评估了编译器的静态预取策略能在多大程度上减少由内存带宽饱和引起的计算停顿。详细的方法和实现见第 VI 节。

![](images/e7f0a4143af5d6464918c5476b59846d280e430b07a823161f4f94eb03722fe3.jpg)  
(a) Transformer 层内的参数生命周期。

![](images/35d5917739c722f4d26f46caa36149586e450531684e90781b1e23fbb2d5e078.jpg)

![](images/b555865bb52aaac54c39c7d8bbbf4652931efd71a3378980e7d00ac2953d9ee6.jpg)  
(c) LLM 推理期间的计算停顿周期。  
图 7. 编译器管理的片上内存的局限性。

内存碎片化。现代深度学习编译器通常将每个张量分配到片上 SRAM 的连续区域中。随着层复杂度的增加，大型中间张量不均匀的生命周期自然会导致内存碎片化。算子融合的广泛使用进一步加剧了这一问题。为了减少运行时开销并提高内存局部性，编译器通常采用融合技术将多个操作合并为单个 kernel。常见的例子包括 QKV 投影融合、FlashAttention（融合了 $Q \times K ^ { T }$、Softmax 和 $S \times V ,$）以及 FFN 融合（合并了 W1 投影、GELU 和 W2 投影）[23], [33]。

图 7a 展示了在模拟编译器应用三种代表性优化（QKV 投影、Flash attention 和 FFN 融合）的模拟器中，基于数据复用的每个操作的生命周期。例如，QKV 投影融合强制 Q、K 和 V 激活在同一个 kernel 内同时被计算和保持，从而阻止了它们的提前释放。图 7b 是在 2 MB SRAM 上应用所有三种融合时，地址空间中随时间使用的内存占用的一部分。这使得在分配新内存时难以重用碎片化的内存地址空间。虽然这种融合提高了数据复用并减少了 kernel 启动开销，但它导致了输入和输出之间内存生命周期的重叠。这些相互依赖性阻碍了内存的提前回收，留下了碎片化且不可用的 SRAM 部分。即使是最佳适配等高级启发式分配器也无法完全缓解这种影响，导致 SRAM 利用率次优，并在某些情况下导致片外内存溢出。至关重要的是，这种碎片化不仅仅是启发式分配的假象：它源于 tile 必须以连续方式和固定 tile 粒度映射的要求，这阻止了编译器将较小的活动区域打包到这些间隙中。

![](images/cdaa870cfd7d1d2725b7299acbbe664d5fc1dc482a42ca7274a6e49900a40fe3.jpg)  
(a) Tile 大小粒度的暂存器内存管理。  
(b) 具有提前回收机制的细粒度内存管理。  
图 8. 通过片上内存管理缓解 I/O 突发。

预取局限性。图 7c 展示了解码层中计算停顿周期数与 token 生成长度的关系，比较了实际实现与理论上限。第一种策略 Compiler-Ideal 基于全图活跃性分析模拟了真实的 XLA 行为。虽然它最小化了峰值内存使用量，但它坚持连续内存分配的严格约束，迫使数据作为连续块加载到 SPM 中。相反，第二种策略 Optimal 通过放宽这种连续性约束来表示理论上限。它假设数据可以以字节级粒度进行预取，从而有效利用整个 SRAM 容量而没有碎片化开销。Compiler-Ideal 和 Optimal 之间的计算停顿差距（阴影部分）随着生成长度逐渐增加，在 4K 时达到 32.7% 的峰值。该差距主要源于静态 SPM 系统中粗粒度的内存管理和预取约束，反映的是一种基础架构的局限性，而不是任何特定编译器的缺陷。

这种差距源于允许静态编译器发出预取请求的条件不足。在当前编译器驱动的 SPM 系统中，只有当以下条件同时满足时才会触发预取：(1) 当前有可用的内存带宽，(2) 有足够的时间获取整个连续的内存 tile，以及 (3) 存在足够大的连续空闲片上内存区域来容纳数据。内存连续性的要求与内存碎片化相结合，显著限制了编译器主动预取即将到来的数据的能力。因此，可用的内存带宽往往未被充分利用，增加了由于数据可用性延迟而导致计算停顿的概率。与 Optimal 相比，Compiler-Ideal 中增加的计算停顿周期定量地证明了当前预取策略中连续性约束带来的低效性。

## IV. SMOOTH 概述

我们提出 SMOOTH，即 SMOothing I/O Traffic with Hardware support，一种硬件辅助的片上内存管理框架，旨在最大化设备端 LLM 推理的内存带宽利用率。现有的基于 Scratchpad 的架构通常依赖粗粒度的 Tile 分配，这需要连续的物理空间，并将内存复用推迟到完整计算完成之后。这种刚性导致碎片化，并阻碍了在计算密集型周期内利用内存带宽余量。为解决这些限制，SMOOTH 引入了一种细粒度的基于块的内存系统，将逻辑 Tensor 组织与物理 SRAM 布局解耦，从而实现激进的预取和持续的吞吐。

SMOOTH 基于硬件实现的动态内存控制器（DMC）构建，该控制器通过三个关键设计原则管理内存操作：第一，细粒度块分配。内存不使用可变大小的 Tile，而是以与硬件处理单元对齐的固定大小块进行管理。这种方法消除了可变大小分配中常见的外部碎片和内存空洞，简化了空闲空间追踪的硬件逻辑。第二，低开销地址转换。DMC 采用直接映射的块表和基于位图的空闲列表，将编译器可见的逻辑地址转换为物理 SRAM 地址。为最小化延迟，该转换机制包含一个地址检查模块，允许顺序映射区域直接访问，在空间局部性保持时跳过表查找。第三，硬件驱动的早期回收。与等待显式软件释放信号的传统设计不同，SMOOTH 通过硬件管理的 `use_cnt` 自主追踪 Tile 使用情况，使得内存块在其数据被消费后即可立即回收。

SMOOTH 将复杂内存调度的负担从编译器转移到运行时硬件。编译器执行静态生命周期分析以标注使用计数，而 DMC 负责动态分配和释放。这种协同设计使 SMOOTH 能够通过为逻辑上相邻的 Tensor 使用非连续物理内存，显著放宽分配约束。因此，计算受限阶段的可用带宽被有效利用，以比传统流水线方案更早地预取即将使用的数据（例如后续层的权重）。如图 8 所示，SMOOTH 的细粒度管理与早期回收机制积极利用这些空闲时间进行预取，最大化计算与 I/O 的重叠，并缓解内存受限 SoC 中的带宽瓶颈。基于块的内存分配通过解决现有方法的粒度限制，显著提升了 Attention 阶段的预取效率。如图 9 所示，我们比较了四种片上内存数据管理和未来数据预取策略，分别针对非碎片化和碎片化场景： 硬件管理的 Cache，其中硬件以最细粒度预取数据； 由编译时静态分析驱动的尽力而为预取，以实现最小的片上内存占用，这一方式被现代深度学习编译器广泛采用； 硬件驱动的基于块内存分配与编译器驱动的预取； 基于块内存分配与通过快速回收已用块的激进预取。首先，在连续内存中，硬件管理的预取器 盲目地在当前 Tensor 范围内操作，无法提前查看 Vcache（记为 $V$）。相比之下， 依赖编译器引导的生命周期分析来确定内存分配。激进预取策略 试图通过尽早且紧凑地分配 Tile 来最大化片上内存利用率。然而，其有效性从根本上受到粗粒度 Tile 边界的制约：如果某个 Tile 无法放入剩余的连续区域，预取就无法进行。基于块的分配 通过将逻辑 Tile 与物理布局分离消除了这一限制，尽管它引入了内部碎片。这有效提高了 SRAM 利用率，因为 Tile 的部分内容（如 V − cache 块）可以在少量空闲空间可用时立即被预取。

![](images/77f8dbf606051b8639fcb71fda9bec48a5ebf77c903b559375b89cd53cd02eda.jpg)  
图 9. 连续和非连续内存场景下的片上内存管理策略。

这一优势在碎片化场景中更为关键。随着 S × V 计算的推进，一些已使用的 V 投影和 Attention 输出被释放，以允许未来 Tile 的预取。当释放导致内存碎片化时，硬件 Cache 可以通过细粒度 Cache 行分配利用碎片化空间，但它缺乏对未来 Tile 访问的认知，因此无法主动为下一操作预取数据。标准编译器 无法有效利用碎片化空间，并预取了大的连续权重 Tensor $W 0 _ { 0 }$ 和 $W 0 _ { 1 }$ ，导致外部碎片化。然而，基于块的分配 有效识别了这些分散的可用块，并用额外的权重（W1）进行预取，以在物理碎片化的情况下维持高片上利用率。在 中，基于块分配与早期块回收机制抢先回收了已被消费的 $V _ { 3 }$ 和 $S _ { 3 }$ 块。释放的块随后被用于预取下一个 Tile $( W 1 _ { 1 } )$

## V. SMOOTH 架构

## A. 基于块的片上内存管理

为简便起见，SMOOTH 使用固定大小的块来管理片上内存，类似于虚拟内存系统中的分页机制。然而，与操作系统的分页不同，其虚拟地址空间并不比物理 SRAM 大多少。传统的虚拟内存抽象出一个宽敞的每进程地址空间，而 SMOOTH 则提供了一个块虚拟化的分配接口，以协助 DL 编译器协调 SPM 分配。尽管放宽物理连续性要求减少了碎片，但所有活跃数据仍必须容纳在物理 SRAM 内；超过此限制将迫使代价高昂的片外访问。因此，编译器可见的虚拟空间实际上受限于 SRAM 容量，从而实现了一个高效的直接映射转换表。图 12a 展示了所需的微架构。每个块表项存储物理块地址（p blk）、分配的连续块数量以及编译器推导的剩余使用计数，同时使用一个位图来跟踪所有物理块的分配状态，以实现快速的空闲空间搜索和回收。缓冲区中的地址检查模块决定一次访问是否需要转换。DMC 中的四个轻量级模块实现了低开销转换和高效的块管理：find zero 识别最长的空闲区域，alloc 预取并分配块，free 回收过期的块，而 block table lookup 解析映射关系。

![](images/fa0cf980d310f6f301f46dfc71344d6174599630ee2d7c735bc3d2650774d290.jpg)  
Fig. 10. 基于块的片上内存分配。

![](images/a5b168e126d365125ea685d83b4a27ac9f9ddd2ff0e5aac2aa79c55a5149a027.jpg)  
Fig. 11. Q 投影期间从缓冲区请求的内存访问。

图 10 描述了一个内存分配场景：在虚拟地址 0x05 处请求 4 MB，use cnt 为 2。分配策略取决于 SRAM 中连续空闲块的可用性，这由分配位图跟踪。在情况 ⃝1 中，存在至少 4 MB 的连续空闲区域，分配器识别出一个覆盖物理块 0x02 到 0x08 的空闲跨度。位图被更新以反映此分配，并且直接映射块表记录了虚拟地址与分配的物理块之间的映射。对于对应于 virt=0x05 的每个虚拟块，p blk 字段存储分配的物理块地址，cont 字段存储剩余的连续块数量，而所有表项的 use cnt 字段均设置为 2。在情况 ⃝2 中，如果碎片化导致无法找到足够大的连续区域来满足请求，DMC 会分配多个不相交的物理块区域。DMC 首先通过 find zero 模块获取最长连续区域的起始块地址和大小，并从该起始地址开始顺序分配块。如果请求的分配大小超过该区域的大小，分配器会重复搜索以寻找下一个最长的连续区域（分配 0x09–0x0C，然后是 0x01–0x03）。位图相应地更新，块表在 p blk 表项中记录分配的块索引。cont 字段记录每个分配段内的连续长度，而 use cnt 字段与连续分配情况中保持一致。

![](images/c0779fd3fc7b947cf2b283589f328c6640b96c90d1923247b208787438fcea33.jpg)  
Fig. 12. SMOOTH 的设计组件。 带地址转换的访问。 无需块表查找的直接访问。 带 end cmd 的访问以实现早期回收。 回收确保数据完整性的块。 利用空闲带宽将数据预加载到回收的块中。

## B. 快速高效的地址转换

为了提高 LLM 推理的片上内存管理效率，我们提出了一种快速且轻量级的地址转换机制，该机制利用空间局部性来最小化查找开销。该机制建立在先前引入的基于块的内存管理方案之上，并采用直接映射块表在编译器生成的虚拟地址和物理 SRAM 地址之间进行快速转换。核心思想是利用深度学习工作负载中常见的连续分配模式来减少转换查找，特别是在 Q/K/V 投影等矩阵乘法操作期间。图 8 展示了输入向量与权重 Q 矩阵乘法操作期间从缓冲区请求的内存访问场景。

在没有物理地址的情况下访问数据 a 和 A 需要进行块表查找。然而，在地址转换之后，连续地址内的数据访问可以直接使用物理地址进行访问。如图 12b 所示，当缓冲区请求带有查找标志 1 的数据 a 时，DMC 通过块表执行地址转换，并将数据连同相应的物理地址和连续块的数量（p blk=0x2400, cont=4）传输到缓冲区。转换完成后，缓冲区会缓存连续范围信息以供后续访问使用，从而允许在该连续内存内直接通过物理地址访问数据，而无需额外的块表查找。为了有效管理这些直接访问，缓冲区逻辑动态监控与架构块大小定义的块边界相对应的地址位字段（例如，对于 1 KB 的块大小，跟踪第 10 位地址位）。在 ISA 操作执行期间，缓冲区使用此位级信息来检测块索引何时发生变化，表明访问已移动到一个新块。图 12c 中的数据 b 位于先前接收的连续空间中，因此可以直接在物理地址（0x2500）处访问它，从而降低访问延迟。但是，如果跨越了块边界并且缓存的 cont 信息指示下一个块在物理上不连续，缓冲区会重新置位查找标志以发起新的地址转换请求。此外，缓冲区逻辑了解正在执行的 ISA 操作的内存访问模式和输入大小，使其能够确定何时完成了对缓冲区的所有访问。

通过跟踪 ISA 执行的进度，缓冲区可以识别何时发生对一个块的最后一次访问。在为此最后一次访问发出内存加载请求时，缓冲区会置位 end cmd 标志。例如，在图 12d 中，缓冲区在请求块 0x2400–0x27FF 的最后一个数据元素 d 时设置 end cmd=1，指示该块将不再用于当前操作。随后，DMC 递减块表中关联的 use cnt 表项，允许该块被提前回收并重新用于后续分配。

## C. 数据预加载

为了缓解 LLM 推理期间的突发内存流量，SMOOTH 执行一种硬件辅助的回收与预加载机制，以从未使用的数据中回收内存，并在数据被请求之前将其预加载到释放的空间中。在没有挂起内存请求的空闲周期期间，DMC 内部周期性地识别 `use cnt` 已达到零的已分配块（图 12e）。如图 12f 所示，早期回收遵循严格的顺序，以确保安全地重用回收的区域。DMC 首先更新 `block table` 状态，将相应的块标记为不再使用，然后清除关联的 `bitmap` 条目。由于分配决策依赖 `bitmap` 来识别空闲空间，这种顺序防止了新分配在回收过程完全完成之前覆盖数据。回收内存后，DMC 立即开始预加载，以利用原本空闲的内存带宽。当检测到空闲带宽周期时，DMC 按顺序预加载数据。在每次预加载机会期间，要加载的块数由公式 (1) 确定：

$$
N _ { \mathrm { p r e l o a d } } = \lfloor ( U \times B W ) / B l o c k _ { - } s i z e \rfloor\tag{1}
$$

其中 $N _ { \mathrm { p r e l o a d } }$ 是要预加载的块数，U 代表可用的空闲计算周期，BW 是可用的内存带宽，由硬件在执行期间动态测量。随着块被分配用于预加载，`bitmap` 和 `block table` 中的相应条目被更新以反映其新状态。

预加载持续进行，直到空闲预算耗尽或没有剩余空闲区域。DMC 以细粒度的块级别将后续数据块从主存预加载到 SRAM 中，并将最后检索的块的索引存储在寄存器中。当从缓冲区访问数据时，DMC 查询此寄存器以确定相应的数据是否已完全加载到片上内存中。如果加载完成，则直接从 SRAM 读取数据；否则，从主存中获取，以确保数据传输的无缝继续。通过将早期回收与带宽感知预加载相结合，SMOOTH 实现了细粒度、主动的 SRAM 管理，减少了突发性，掩盖了碎片化损失，并在有限的片外带宽下维持了高吞吐量的数据流。

## D. 开销

为了评估所提出的硬件模块的面积、时序和功耗开销，我们使用开源综合工具 Yosys [45] 综合了五个关键功能。鉴于 Snapdragon 8 Gen3 采用 TSMC 的 4 nm 工艺 [46] 制造，我们使用了公开可用的 ASAP7 7 nm 标准单元库 [47]，据我们所知，这是目前可用的最精确的开源技术节点。由于确切的 NPU 裸片面积未公开，我们保守地假设其占总 SoC 面积的 10%，并使用此估计值作为计算相对面积开销的基准。综合是在表 III 中的硬件配置下执行的，目标为移动级 NPU 和表 IV 中描述的 GPT-Neo-Quant。表 I 报告了在 1 KB 块大小下的面积估计值，其他块大小的结果提供在 §VI 中。NPU 和 SRAM 行对应于假设的基准组件，而计算和内存（SRAM）条目显示了来自我们模块的综合开销。计算逻辑仅增加了 0.0023%，内存控制逻辑仅增加了 0.095%，相对于估计的

表 I  
所提出模块的面积开销。
<table><tr><td></td><td>NPU</td><td>SRAM</td><td>计算</td><td>内存 (SRAM)</td></tr><tr><td> $\mathrm { A r e a } ( \mu \mathrm { m } ^ { 2 } )$</td><td>13,730,000</td><td>1,811,939</td><td>314</td><td>13,050</td></tr><tr><td>比例 (%)</td><td></td><td>13.2</td><td>0.0023</td><td>0.095</td></tr></table>

NPU 面积，证实了整体硬件占用面积可以忽略不计。表 II 总结了每个硬件模块的延迟（以皮秒为单位）和功耗（以皮瓦为单位）。相对于观察到的延迟减少，SMOOTH 引入的时序开销极小，而其功耗保持在亚纳瓦范围内，表明对整体系统效率的影响可以忽略不计。具体而言，在表 IV 描述的硬件配置下，在输入长度为 1024 且输出长度为 2048 的所有实验中，控制开销保持在总延迟的 0.1% 以下。此开销被纳入 § VI 中呈现的评估结果，以确保准确测量端到端执行时间。

表 II

每个硬件模块的延迟和功耗。
<table><tr><td>指标</td><td>find_zero</td><td>alloc</td><td>addr_check</td><td>bt_lookup</td><td>free</td></tr><tr><td>时间</td><td>364.4</td><td>1508.2</td><td>83.7</td><td>615.2</td><td>654.6</td></tr><tr><td>功耗</td><td> $1 . 4 \times 1 0 ^ { - 1 }$</td><td> $5 . 5 \times 1 0 ^ { - 1 }$</td><td> $3 . 0 \times 1 0 ^ { - 2 }$</td><td> $2 . 3 \times 1 0 ^ { - 1 }$</td><td> $2 . 8 \times 1 0 ^ { - 1 }$</td></tr></table>

## VI. 评估

## A. 实验设置

我们使用 LLMCompass [27]（一个用于基于 Transformer 的 LLM 推理的周期精确模拟器）来评估 SMOOTH。LLMCompass 基于 ScaleSim [28] 构建，模拟 Transformer 模型的生成阶段。我们在模拟器中集成了一个端到端的 SRAM 管理器，以支持基于地址的分配并实现整个执行过程中的数据预加载。所有实验均在反映移动 NPU 架构的硬件配置下进行，这些架构具有严格的 SRAM 限制、低内存带宽以及矩阵和向量单元等固定功能计算引擎。详细的系统配置见表 III，该配置是考虑了 Qualcomm Hexagon V73 处理器（HMX、HVX [48]）和移动 DDR 内存（LPDDR5）[49]–[51] 进行设置的。

在 § III 所述的实验中，对于 TinyLLaMA，非线性操作在 Jetson AGX Orin、Galaxy S24 Ultra 和 Edge TPU 上分别占总执行时间的 20.4%、17.0% 和 14.1%，而对于 GPT-2.7B 则分别占 17.1%、12.5% 和 10.3%（图 5b）。相比之下，模拟器报告 TinyLLaMA 和 GPT-2.7B 的比例较小，分别为 9.4% 和 5.7%（图 13）。这些结果表明，我们的模拟环境对非线性操作所花费的执行时间提供了保守的估计。

基线。我们比较了五种片上内存管理策略。Compiler-Ideal：一种理想化的基于编译器的策略，假设最大程度的内存预加载和最佳适配内存分配，利用生命周期分析和重用非重叠内存缓冲区来提高内存效率 [22], [25], [52]。此外，对于每一层和操作，它通过模拟评估从 512 B 到 4 MB 的分块大小，并选择产生最小延迟的配置。Capuchin [53]：一种硬件管理策略，将片上内存视为 64 字节缓存，基于运行时访问模式以缓存行粒度动态预取张量，以提高数据局部性并减少停顿。Gemmini [54]：一个全栈 DNN 加速框架。它采用流水线化的片上内存分配策略，通过重叠输入/输出分块，实现细粒度的字节级预加载。SMOOTH-Base：一种块粒度的内存分配器，通过实现紧凑的数据放置来减少 SPM 内的碎片并提高内存带宽利用率。SMOOTH-ER：带有额外未使用内存块早期回收机制的 SMOOTH-Base。该回收机制增加了内存重用，并允许及时预加载未来的数据，从而支持连续且高效的数据流。

表 IV 模型配置详情。  
表 III  
移动 NPU 的模拟环境。
<table><tr><td>参数</td><td>移动 NPU</td></tr><tr><td>核心频率</td><td>940 MHz</td></tr><tr><td>核心数</td><td>1</td></tr><tr><td>矩阵引擎 (ME)</td><td>32×32</td></tr><tr><td>向量引擎 (VE) (32 ALUs/lane)</td><td>32 lanes</td></tr><tr><td>SRAM 大小</td><td>2 /  8  /  32MB</td></tr><tr><td>DRAM 带宽</td><td>16 /  32 / 64 /  128 GB/s</td></tr></table>

![](images/0be07a4603b10f816ad29a5a148f6419aefa583cfcdc707decb3f647786e3878.jpg)  
图 13. Compiler-Ideal（基线）上线性和非线性操作所花费的端到端延迟分解。

## B. 模型

为了反映真实的移动部署场景，我们选择了适合在资源受限的移动 NPU 上执行的基于 Transformer 的 LLM。鉴于移动 NPU 上对具有大量参数的 LLM 推理的需求不断增长，我们还评估了大型模型，例如 GPT-3 13B。所选模型在架构规模和量化格式上各不相同，从而能够对一系列计算和内存需求进行全面评估。表 IV 总结了它们的配置。所有模型都采用了 § III 中描述的三种操作融合（图 7a），这些融合在现代深度学习编译器中被广泛采用。为了与设备端助手和聊天应用等移动用例保持一致，我们在所有实验中均使用批次大小为 1，如 [50], [55] 中所述。

<table><tr><td>模型</td><td>#参数</td><td>#层数</td><td>#头数</td><td>dmodel</td><td>量化</td></tr><tr><td>TinyLLaMA [56]</td><td>1.1B</td><td>22</td><td>32</td><td>2048</td><td>w4a8/int8</td></tr><tr><td>GPT-Neo [57]</td><td>1.3B</td><td>24</td><td>16</td><td>2048</td><td>w4a8/int8</td></tr><tr><td>GPT-3 XL [58]</td><td>1.3B</td><td>24</td><td>24</td><td>2048</td><td>w4a8/int8</td></tr><tr><td>Gemma-2 [59]</td><td>2.0B</td><td>18</td><td>8</td><td>2048</td><td>w4a8/int8</td></tr><tr><td>GPT-3 2.7B [58]</td><td>2.7B</td><td>32</td><td>32</td><td>2560</td><td>w4a8/int8</td></tr><tr><td>LLaMA2 [60]</td><td>7.0B</td><td>32</td><td>32</td><td>4096</td><td>w4a8/int8</td></tr><tr><td>Bloom [61]</td><td>7.1B</td><td>30</td><td>32</td><td>4096</td><td>w4a8/int8</td></tr><tr><td>GPT-3 13B [58]</td><td>13.0B</td><td>40</td><td>40</td><td>5140</td><td>w4a8/int8</td></tr></table>

![](images/6024f447c8180c5a116b6284e68be8bd8c75f049d7f029ac723b781a090b4ab5.jpg)  
图 14. 归一化到 Compiler-Ideal 的 TTFT。

TTFT. 图 14 展示了在 8 MB SRAM 下，五种分配策略的归一化首 Token 响应时间（TTFT），以 Compiler-Ideal 为基准进行归一化。首 Token 推理不需要 KV cache，因此 8 MB 是足够的。因此，将 SRAM 增加到 32 MB 最多只能将 TTFT 降低 1.0%。由于 TTFT 的计算强度较高，像 Gemmini 那样简单地对下一个 tile 进行流水线处理可以提升性能，但 Compiler-Ideal 由于预加载时间不足而表现不佳。然而，得益于细粒度的 block 级别预加载，SMOOTH-ER 相比 Compiler-Ideal 实现了平均 41.4%、最高 59.2% 的 TTFT 降低。Capuchin 在 GPT 模型上表现出 TTFT 的降低，但在其他模型上与 Compiler-Ideal 相似。这种延迟是由于硬件 cache 无法预取因 SRAM 尺寸增加而由 FlashAttention 增加的 attention tile，因为它缺乏编译器提供的关于 tensor 数据生命周期的信息。

TTLT. 图 16a 展示了在输入长度为 512 个 token 和 8 MB SRAM 容量下，端到端生成延迟（称为 Time-to-Last-Token，TTLT）的扩展性。TTLT 衡量从 prompt 输入到生成最终 token 的延迟，是评估用户感知响应速度的综合指标。附带的柱状图描绘了所提出的 SMOOTH-ER 内存管理方案相对于两个基线（Compiler-Ideal 和 Gemmini）所实现的相对改进。SMOOTH-ER 相比 Compiler-Ideal 显示出 43.2% 的总体平均性能提升，相比 Gemmini 显示出 49.1% 的总体平均性能提升，分别实现了 60.0% 和 73.0% 的最大性能提升。此外，SMOOTH-ER 相比基线 SMOOTH-Base 实现了高达 24.0% 的平均提升。另外，柱状图中的阴影区域表示 prompt 阶段贡献的性能增益比例。对于短输出长度，大部分增益来自 prompt 阶段。然而，随着输出 token 长度的增加，生成阶段占据了整体性能提升的大部分。对于短输出长度，attention 和非线性操作时间较短，留下的空闲周期不足以进行预加载。这导致与对下一个 tile 进行流水线处理的 Gemmini 相比，改进较小。然而，随着输出 token 长度的增加，预加载更多数据显著提升了性能。相反，随着输出长度的增加，Compiler-Ideal 也通过预加载更多数据来提升性能，但由于连续的 SPM 地址分配，这会导致内存碎片化。SMOOTH-ER 通过解决内存碎片问题，在所有输出长度下相比 Compiler-Ideal 显著改善了延迟。图 15 展示了改进对 SRAM 大小的敏感性。当 SRAM 大小减小到 2 MB 或增加到 32 MB 时，改进趋于减小。使用流水线预加载下一个 tile 的 Gemmini 对 SRAM 几乎不敏感。在片上内存较小的情况下，用于预加载的物理内存容量有限。然而，在片上内存较大的情况下，可以通过更大的 tile 尺寸和连续地址分配来实现延迟改进，从而进一步减小了改进幅度。特别是，随着片上内存大小的增加，SMOOTH-ER 相比 Compiler-Ideal 的性能增益显著下降。这是因为，在片上内存充足的情况下，Compiler-Ideal 遭受的内存碎片化较少，允许其预加载足够数量的数据。

![](images/347cef71e0cd79a6cdb0019f0af265bdbaa6ce3387c39f65dbe072a4997f9185.jpg)  
--)'-\$"% !&!

![](images/dc8daae47bac74c60122a800f339be78207be694f261b6ec3023908054de0ee0.jpg)  
--)'-"" #  
图 15. 相对于 8 MB 基线，在 2 MB 和 32 MB 下增益的 SRAM 大小敏感性。

片上内存占用率. 图 16b 和图 16c 展示了每 N 个 token 的每 token 生成延迟以及 token 生成期间所有层的平均 SRAM 占用率，比较了有无操作融合的情况。在没有融合的情况下，随着输出序列长度的增加，由于 KV cache，需要加载到片上内存的数据量增加。然而，由于每个操作在未经优化的情况下顺序执行，内存带宽变得饱和，包括基线 Compiler-Ideal 在内的所有策略的性能提升都受到限制。相反，操作融合偶尔会缓解内存带宽饱和，从而实现激进的预加载，这显著降低了推理延迟。具体来说，图 17 展示了 attention 操作结束时的 SRAM 占用率。对于 Capuchin，端到端层占用率与其他策略相当，但在 attention 阶段结束时占用率急剧下降。在没有融合的情况下，每个操作独立执行，阻止了即使是强大的预取器预测后续操作，从而限制了性能。然而，有了融合，多个操作被组合成一个单一的 tensor 级执行单元，从而实现更高效的预取并降低延迟。图 18 展示了随着输出长度增加，在 GPT-Neo 和 LLaMA2 中为每个操作加载 tile 时可以从 buffer 提供服务的 tile 比例（命中 tile）。尽管 block 被快速回收，但像 LLaMA2 这样的大型模型每个操作仍然需要许多 tile，而一次只有一小部分可以驻留在片上内存中。因此，即使 SRAM 占用率很高，SMOOTH-ER 相比 SMOOTH-Base 带来的额外延迟降低仍然有限。此外，SMOOTH 利用编译器计算的操作生命周期信息来预加载未来 tensor 的数据，否则这些数据在硬件层面很难预测，从而实现进一步的延迟降低。

内存带宽敏感性. 图 19a 评估了在不同内存带宽（16、32、64、128 GB/s）下以及在 64 GB/s 最大带宽下的 Geekbench 并发工作负载干扰下，SMOOTH-ER 对 GPT-Neo 的 token 间延迟（ITL）的影响，显示了 SMOOTH-ER 相比基线策略的性能提升。Geekbench 使用了两个 CPU 工作负载和四个 GPU 工作负载进行测试，如 § III 所述。结果表明，随着内存带宽的降低，系统越来越受限于内存，从而导致 SMOOTH-ER 获得更显著的性能提升。在评估的配置中，SMOOTH-ER 相比 Capuchin 实现了 30.5% 的平均延迟降低，相比 Compiler-Ideal 实现了 40.0% 的平均延迟降低。与 SMOOTH-Base 相比，SMOOTH-ER 提供了 11.1% 的平均提升（高达 47.0%）。当内存带宽足够大时，内存容量和传输限制得到缓解，导致 SMOOTH-ER 和 SMOOTH-Base 之间的性能差距缩小。相反，在带宽受限的条件下，早期回收提供了更大的性能优势。此外，尽管由于 CPU/GPU 工作负载干扰导致空闲带宽动态变化，ITL 相比 Compiler-Ideal 实现了 42.7% 的平均增益，相比 SMOOTH-Base 实现了 5.0% 的平均增益。

输入序列长度敏感性。图 19b 展示了在固定输出生成长度为 1024 个 Token 的情况下，归一化 ITL 随输入序列长度的变化。最近，即使在移动环境中，对长上下文推理的需求也在迅速增加。随着输入序列长度的增加，KV cache 的内存占用成比例增加，这严重降低了生成阶段的延迟。在这些内存密集型条件下，所提出架构的有效性变得非常显著。SMOOTH-ER 相比 Gemmini 实现了高达 73.0% 的性能提升，并且相比 SMOOTH-Base 进一步实现了高达 26.4% 的提升。按序列长度的详细划分揭示了 SMOOTH-ER 相对优势的明显上升趋势。例如，相比 Gemmini 的平均增益从 2K 序列长度时的 50.1% 稳步扩大到 32K 时的 66.8%，证实了 SMOOTH-ER 有效缓解了与处理长输入序列相关的不断攀升的内存开销。

能耗分析。图 20 展示了根据块大小生成第 N 个 Token 的能耗，因为所提出架构的开销随每个输出序列长度的块大小而变化。随着生成长度的增加，基线架构中频繁的内存访问和低效的缓存利用率导致能耗显著激增。在这些严重受限于内存的条件下，所提出架构的能效变得非常显著。总体而言，假设每个序列长度采用最佳块大小，SMOOTH-ER 相比 Compiler-Ideal、Gemmini 和 Capuchin 分别实现了 44.0%、51.2% 和 39.9% 的平均能耗降低。按生成长度的详细划分揭示了 SMOOTH-ER 相对节能效果的明显上升趋势。例如，相比 Gemmini 的能耗降低从 1K 序列长度时的 28.1% 稳步扩大到 32K 时显著的 70.7%。类似地，相比 Compiler-Ideal 的节能从 30.7% 增长到 56.7%。此外，实验数据证实 SMOOTH-ER 引入的硬件模块开销极其微小——对于 32K 序列仅达到 15 纳焦耳的峰值。SMOOTH-ER 有效缓解了与长上下文生成相关的增加的能耗需求，同时几乎不产生额外的架构开销。

(a) 第 1K 个 Token  
![](images/6433459e0c0da56a4b5d7c740a9117efff56b32f0ea392528f82feca341a6cf1.jpg)

![](images/4cd5650c8bf2131120faa9852757312680116f9af3077eceb3a4a429226dc6cd.jpg)  
(a)

![](images/8f19682366d5f73ff73a0fb07cf01faaf9df3c6e8cc36e5a62eac8e7d3091dac.jpg)

(b) 无操作融合。  
![](images/ddaf2f8415184e3dc341e1cfc4164a5fa476b4f9b95fd9957e7e77193e77770c.jpg)  
(c) 有操作融合。

图 16. (a) TTLT 以及 SMOOTH-ER 相比 Capuchin、Compiler-Ideal 和 Gemmini 的增益，(b-c) 在 Compiler-Ideal 下归一化到输出长度为 1 的情况下的单 Token 生成延迟和 SRAM 占用率，(b) 无融合，(c) 有融合。  
图 17. Attention 结束时的 SRAM 占用率，归一化到 Compiler-Ideal。  
![](images/23d7e7d7b191b2094eb636c7d4beafe00f5bda6e57e7f97fe48799beeab05f15.jpg)

![](images/56a4a282f604283607de765dfccb7700c0dea1b833e0040c0977622a401573c2.jpg)  
图 18. 每次操作中命中 tile 相对于未命中 tile 的缓冲内存占用率。

(a) ITL 的提升取决于 16-128 GB/s 的内存带宽以及 64 GB/s 时的协同运行工作负载干扰  
![](images/2d999b685c359106cfaf8b05fff949bb13c453a78d21541adfd1b03f017e4512.jpg)  
(b) 取决于输入序列长度的归一化 ITL。  
图 19. 动态运行时因素对 inter-token 延迟的影响。

![](images/567b392c7beec4d5cb2f01742b390a8f0215edddaf67c72e7b5c646d0822a24f.jpg)

![](images/d0edfcc93aeac1070d525eb9e82c2764a8b895af50b3c8b7751ba0d577f2ed78.jpg)  
(b) 第 8K 个 Token

![](images/52f526b30e9c879f7ecde4980ee81ae176339cd18d726ac66425a12db275cfb1.jpg)  
图 20. 不同块大小下生成第 N 个 Token 的能耗。

块大小敏感性。图 21a 展示了三个代表性模型——GPT-Neo、LLaMA2 和 GPT-3 13B——在 SMOOTH-ER 不同块大小下的端到端延迟，评估条件为输入长度 1024 和输出长度 2048，SRAM 容量为 8 MB。延迟值归一化到块大小为 1024 字节时的基线延迟。每个柱状图上方的数字注释表示在每个块大小下产生的相对控制开销。较小的块大小通过细粒度预加载和改善内存重用来降低延迟；然而，这会增加块表查找开销。虽然有限的 SRAM 会因碎片化引起的 find zero 操作而加剧延迟，但 SMOOTH 的专用硬件设计确保了控制开销可以忽略不计。开销图量化了连续区域高效地址转换带来的延迟降低。虽然较小的块通常会增加块表查找开销，但我们的 lookup flg 机制避免了对连续地址的冗余转换。更大的 SRAM 容量也增加了找到连续空闲区域的可能性，从而减少了 find zero 和 alloc 开销。与基线相比，连续地址转换产生了 0.2% 的延迟降低。SMOOTH-Base 和 SMOOTH-ER 通常将块大小设置为模型维度。然而，如果块大小与 tile 大小不对齐，内部碎片化可使延迟增加高达 9.9%（图 21b）。

![](images/d511cf0ec9766f0f80c4d7f37537d94b0ef7a8bf3a060a2e48cf89cb9341ffd7.jpg)

(a) 不同块大小下的归一化端到端延迟和相对控制开销。  
![](images/5c10e8cdfcc43ff30d70918c9080de65b7da62190888a5dd91d5a97880284c16.jpg)  
(b) 块大小未对齐时由内部碎片化引起的延迟退化  
图 21. SMOOTH-ER 的块大小敏感性。

## VII. 相关工作

模型级内存占用缩减。为了解决 LLM 庞大的内存需求，各种模型级技术已被广泛采用，包括权重/激活量化 [62], [63]、剪枝 [64] 以及 KV cache 优化 [65]–[68]。虽然这些方法有效降低了内存容量需求，但它们通常涉及复杂的权衡，例如潜在的精度下降或不规则的计算模式。与这些模型修改无关，SMOOTH 纯粹关注微架构效率，而不改变模型表示。因此，SMOOTH 不会引入精度下降，并且可以与现有的压缩技术正交结合，以实现进一步的推理加速。

静态内存分配。基于软件的方法（例如，XLA [24]、TVM [25]、FlashAttention [33]）应用分块和融合来提高计算局部性。然而，这些方法依赖于静态生命周期分析。这种刚性使得它们无法适应运行时的变化，例如波动的移动内存带宽或变化的 LLM 推理长度，这通常会导致严重的片上碎片化，并且无法在突发 LLM 解码阶段利用瞬态带宽松弛。为了减轻数据移动的开销，现代 GPU 架构引入了硬件加速的拷贝引擎。例如，NVIDIA 的 Tensor Memory Accelerator (TMA) [69] 卸载了异步数据传输，在硬件中处理地址生成以降低寄存器压力。

然而，虽然 TMA 作为一个高效的数据移动引擎，但它不提供内存虚拟化能力；它严格要求物理连续或跨步的地址模式。因此，它缺乏利用在 LLM 推理执行阶段不可避免地产生的分散内存碎片的灵活性。相比之下，SMOOTH 采用块级虚拟化将逻辑张量与物理地址解耦，使硬件能够利用静态编译器和固定功能拷贝引擎都无法利用的非连续空闲空间。

动态内存虚拟化。为了克服静态分配的刚性，硬件辅助的动态内存管理已被积极探索。基础工作如 SPMVisor [70] 引入了硬件/软件虚拟化层 (vSPMs) 以透明地分配分布式片上内存，而 HaVOC [71] 将此扩展到混合 SRAM/NVM 架构。最近，提出了诸如 Amoeba-Cache [72] 之类的自适应缓存架构，通过基于空间局部性动态调整缓存块大小来减少存储浪费。然而，这种纯硬件方法本质上是反应性的：它们依赖过去的访问模式，而不了解未来的数据生命周期。这一限制使得它们无法执行处理 LLM 突发 I/O 流量所需的主动内存回收和预取。相比之下，我们的工作通过将细粒度块分配与编译器驱动的主动管理相结合，解决了片上 SRAM 利用率不足的问题，使 SMOOTH 能够通过静态活跃性分析提前回收内存并预取数据。

## VIII. 结论

我们展示了一种新方法，用于解决限制移动 SoC 上基于 Transformer 的 LLM 推理的突发片外内存流量。我们的硬件辅助、块粒度 SRAM 管理——由运行时数据活跃性跟踪和早期回收驱动——在不修改模型参数或损害精度的情况下，随时间分散加载/存储请求。该设计以 Verilog 实现并集成到 LLMCompass 中，在现实的 SRAM 和 DRAM 带宽约束下，将 TTLT 降低了高达 73.0%。其收益随着片上容量的增加而增长，对 Compiler-Ideal、Capuchin 和 Gemmini 基线形成了补充。未来的工作包括用于联合生命周期分析的更紧密的编译器-硬件协同调度，向异构加速器池的扩展，以及面向多租户或流式场景的感知争用策略。

## 致谢

本研究得到了韩国政府 (MSIT) 资助的信息通信技术规划评价院 (IITP) 拨款（编号：RS-2024-00396013、RS-2024-00459797、RS-2025-02263869 和 RS-2025-09942968），以及韩国政府 (MSIT) 资助的韩国国家研究基金会 (NRF) 拨款（编号：RS-2026-25490694）的支持。

[1] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al., “Palm 2 technical report,” arXiv preprint arXiv:2305.10403, 2023.

[2] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[3] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar,\` et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.

[4] A. T. Neumann, Y. Yin, S. Sowe, S. Decker, and M. Jarke, “An llm-driven chatbot in higher education for databases and information systems,” IEEE Transactions on Education, 2024.

[5] J. K. Kim, M. Chua, M. Rickard, and A. Lorenzo, “ChatGPT and large language model (LLM) chatbots: The current state of acceptability and a proposal for guidelines on utilization in academic medicine,” Journal of Pediatric Urology, vol. 19, no. 5, pp. 598–604, 2023.

[6] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, et al., “Judging llm-as-a-judge with mt-bench and chatbot arena,” Advances in neural information processing systems, vol. 36, pp. 46595–46623, 2023.

[7] Z. Yang, X. Xu, B. Yao, E. Rogers, S. Zhang, S. Intille, N. Shara, G. G. Gao, and D. Wang, “Talk2care: An llm-based voice assistant for communication between healthcare providers and older adults,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 8, no. 2, pp. 1–35, 2024.

[8] A. Mahmood, J. Wang, B. Yao, D. Wang, and C.-M. Huang, “Llm-powered conversational voice assistants: Interaction patterns, opportunities, challenges, and design guidelines,” arXiv preprint arXiv:2309.13879, 2023.

[9] S. Huang, X. Zhao, D. Wei, X. Song, and Y. Sun, “Chatbot and fatigued driver: Exploring the use of LLM-based voice assistants for driving fatigue,” in Extended Abstracts ofthe CHI Conference on Human Factors in Computing Systems, 2024, pp. 1–8.

[10] J. Xu, Z. Li, W. Chen, Q. Wang, X. Gao, Q. Cai, and Z. Ling, “On-device language models: A comprehensive review,” arXiv preprint arXiv:2409.00088, 2024.

[11] Meta, “What’s New Across Our AI Experiences,” 2023. [Online]. Available: https://about.fb.com/news/2023/12/meta-ai-updates/

[12] Meta, “Meta AI is Now Multilingual, More Creative and Smarter,” 2024. [Online]. Available: https://about.fb.com/news/2024/07/meta-ai-is-now multilingual-more-creative-and-smarter/

[13] Meta Quest Blog, “Smart(er) Glasses: Introducing New Ray-Ban — Meta Styles + Expanding Access to Meta AI with Vision,” 2024. [Online]. Available: https://www.meta.com/blog/ray-ban-metasmart-glasses-new-styles-multimodal-ai-ferrari/

[14] Apple, “Apple Intelligence,” 2024. [Online]. Available: https://www.apple.com/apple-intelligence/

[15] Samsung, “Galaxy AI,” 2024. [Online]. Available: https://www.samsung.com/us/galaxy-ai/

[16] L. Yang, K. Sreedhar, H. Liu, and E. Beigne, “Enabling On-Device Large Language Models with 3D-Stacked Memory,” in NeurIPS 2024 Workshop Machine Learning with new Compute Paradigms, 2024.

[17] G. Heo, S. Lee, J. Cho, H. Choi, S. Lee, H. Ham, G. Kim, D. Mahajan, and J. Park, “Neupims: Npu-pim heterogeneous acceleration for batched llm inferencing,” in Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, 2024, pp. 722–737.

[18] J. Park, J. Choi, K. Kyung, M. J. Kim, Y. Kwon, N. S. Kim, and J. H. Ahn, “Attacc! unleashing the power of pim for batched transformerbased generative model inference,” in Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, 2024, pp. 103–119.

[19] M. Zhou, W. Xu, J. Kang, and T. Rosing, “Transpim: A memorybased acceleration via software-hardware co-design for transformer,” in 2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2022, pp. 1071–1085.

[20] OpenXLA, “xla: A machine learning compiler for GPUs, CPUs, and ML accelerators,” 2025. [Online]. Available: https://github.com/openxla/xla

[21] Apache TVM, “Apache TVM: An End-to-End Machine Learning Compiler Framework for CPUs, GPUs and Accelerators,” 2024. [Online]. Available: https://tvm.apache.org/

[22] M. Li, Y. Liu, X. Liu, Q. Sun, X. You, H. Yang, Z. Luan, L. Gan, G. Yang, and D. Qian, “The deep learning compiler: A comprehensive survey,” IEEE Transactions on Parallel and Distributed Systems, vol. 32, no. 3, pp. 708–727, 2020.

[23] “Optimize Large Language Model TVM How-To Tutorial,” 2025. [Online]. Available: https://tvm.apache.org/docs/how to/tutorials/optimize llm.html

[24] TensorFlow, “XLA (Accelerated Linear Algebra),” 2024. [Online]. Available: https://www.tensorflow.org/xla?hl=ko

[25] T. Chen, T. Moreau, Z. Jiang, L. Zheng, E. Yan, H. Shen, M. Cowan, L. Wang, Y. Hu, L. Ceze, et al., “TVM: An automated End-to-End optimizing compiler for deep learning,” in 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18), 2018, pp. 578–594.

[26] Y. Shi, Z. Yang, J. Xue, L. Ma, Y. Xia, Z. Miao, Y. Guo, F. Yang, and L. Zhou, “Welder: Scheduling deep learning memory access via tilegraph,” in 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23), 2023, pp. 701–718.

[27] H. Zhang, A. Ning, R. B. Prabhakar, and D. Wentzlaff, “Llmcompass: Enabling efficient hardware design for large language model inference,” in 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA), 2024, pp. 1080–1096.

[28] A. Samajdar, Y. Zhu, P. Whatmough, M. Mattina, and T. Krishna, “Scale-sim: Systolic cnn accelerator simulator,” arXiv preprint arXiv:1811.02883, 2018.

[29] N. P. Jouppi, C. Young, N. Patil, D. Patterson, G. Agrawal, R. Bajwa, S. Bates, S. Bhatia, N. Boden, A. Borchers, et al., “In-datacenter performance analysis of a tensor processing unit,” in Proceedings of the 44th annual international symposium on computer architecture, 2017, pp. 1–12.

[30] R. Krashinsky, O. Giroux, S. Jones, N. Stam, and S. Ramaswamy, “NVIDIA A100 Tensor Core GPU Architecture: Ampere Archi tecture Whitepaper,” Technical White Paper, NVIDIA Corporation, May 2020. [Online]. Available: https://images.nvidia.com/aem-dam/enzz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

[31] Y.-H. Chen, T. Krishna, J. S. Emer, and V. Sze, “Eyeriss: An energyefficient reconfigurable accelerator for deep convolutional neural networks,” IEEE journal of solid-state circuits, vol. 52, no. 1, pp. 127–138, 2016.

[32] S. Zouzoula, M. A. Maleki, M. W. Azhar, and P. Trancoso, “Scratchpad Memory Management for Deep Learning Accelerators,” in Proceedings of the 53rd International Conference on Parallel Processing, 2024, pp. 629–639.

[33] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Re, “FlashAttention: Fast´ and Memory-Efficient Exact Attention with IO-Awareness,” Advances in Neural Information Processing Systems, vol. 35, pp. 16344–16359, 2022.

[34] L. Zheng, C. Jia, M. Sun, Z. Wu, C. H. Yu, A. Haj-Ali, Y. Wang, J. Yang, D. Zhuo, K. Sen, et al., “Ansor: Generating {High-Performance} tensor programs for deep learning,” in 14th USENIX symposium on operating systems design and implementation (OSDI 20), 2020, pp. 863–879.

[35] H. Zhu, R. Wu, Y. Diao, S. Ke, H. Li, C. Zhang, J. Xue, L. Ma, Y. Xia, W. Cui, et al., “{ROLLER}: Fast and efficient tensor compilation for deep learning,” in 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), 2022, pp. 233–248.

[36] TensorFlow Team, “TensorFlow Lite,” 2025. [Online]. Available: https://www.tensorflow.org/lite

[37] P. Jain, A. Jain, A. Nrusimha, A. Gholami, P. Abbeel, J. Gonzalez, K. Keutzer, and I. Stoica, “Checkmate: Breaking the memory wall with optimal tensor rematerialization,” Proceedings of Machine Learning and Systems, vol. 2, pp. 497–511, 2020.

[38] M. Maas, U. Beaugnon, A. Chauhan, and B. Ilbeyi, “Telamalloc: Efficient on-chip memory allocation for production machine learning accelerators,” in Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, 2022, pp. 123–137.

[39] “Geekbench: Cross-Platform Benchmark,” 2026. [Online]. Available: https://www.geekbench.com/

[40] S.-C. Kao, S. Subramanian, G. Agrawal, A. Yazdanbakhsh, and T. Krishna, “Flat: An optimized dataflow for mitigating attention bottlenecks,” in Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, 2023, pp. 295–310.

[41] Qualcomm Technologies, Inc., “Qualcomm AI Engine Direct SDK,” 2025. [Online]. Available: https://www.qualcomm.com/developer/software/qualcomm-ai-enginedirect-sdk

[42] Coral, “Edge TPU Compiler,” Google LLC, 2025. [Online]. Available: https://www.coral.ai/docs/edgetpu/compiler

[43] NVIDIA Corporation, “NVIDIA CUDA Compiler Driver NVCC,” 2025. [Online]. Available: https://docs.nvidia.com/cuda/cuda-compiler-drivernvcc/

[44] D. Xu, H. Zhang, L. Yang, R. Liu, G. Huang, M. Xu, and X. Liu, “Fast on-device llm inference with npus,” in Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, 2025, pp. 445–462.

[45] C. Wolf, “Yosys Open SYnthesis Suite,” 2023. [Online]. Available: https://yosyshq.net/yosys/

[46] J. Yuan, J. Deng, V. Lin, Y. Chen, J. Chiu, M. Lin, J. Chen, D. Zhang, Y. Chen, D. Liu, et al., “High performance 5G mobile SOC productization with 4nm EUV Fin-FET technology,” in 2023 IEEE Symposium on VLSI Technology and Circuits (VLSI Technology and Circuits), 2023, pp. 1–2.

[47] “ASAP7 PDK,” [Online]. Available: https://github.com/The-OpenROAD-Project/asap7

[48] “Qualcomm Hexagon V73 Technical Reference,” 2025. [Online]. Available: https://docs.qualcomm.com/bundle/publicresource/80-N2040- 54.pdf

[49] “Snapdragon 8 Gen 3 Mobile Platform Product Brief,” 2025. [Online]. Available: https://docs.qualcomm.com/bundle/publicresource/87-71408- 1 REV C Snapdragon 8 gen 3 Mobile Platform Product Brief.pdf

[50] Z. Xue, Y. Song, Z. Mi, X. Zheng, Y. Xia, and H. Chen, “Powerinfer-2: Fast large language model inference on a smartphone,” arXiv preprint arXiv:2406.06282, 2024.

[51] L. Chen, D. Feng, E. Feng, Y. Wang, R. Zhao, Y. Xia, P. Xu, and H. Chen, “Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference,” in Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles, 2025, pp. 359–374.

[52] Z. Zheng, P. Zhao, G. Long, F. Zhu, K. Zhu, W. Zhao, L. Diao, J. Yang, and W. Lin, “Fusionstitching: boosting memory intensive computations for deep learning workloads,” arXiv preprint arXiv:2009.10924, 2020.

[53] X. Peng, X. Shi, H. Dai, H. Jin, W. Ma, Q. Xiong, F. Yang, and X. Qian, “Capuchin: Tensor-based gpu memory management for deep learning,” in Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems, 2020, pp. 891–905.

[54] H. Genc, S. Kim, A. Amid, A. Haj-Ali, V. Iyer, P. Prakash, J. Zhao, D. Grubb, H. Liew, H. Mao, et al., “Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration,” in Proceedings of the 58th Annual Design Automation Conference (DAC), 2021.

[55] Z. Yu, S. Liang, T. Ma, Y. Cai, Z. Nan, D. Huang, X. Song, Y. Hao, J. Zhang, T. Zhi, et al., “Cambricon-llm: A chiplet-based hybrid architecture for on-device inference of 70b llm,” in 2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO), 2024, pp. 1474–1488.

[56] TinyLlama, “TinyLlama-1.1B-Chat-v1.0 (Hugging Face model),” 2024. [Online]. Available: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

[57] EleutherAI, “GPT-Neo-1.3B (Hugging Face model),” 2024. [Online]. Available: https://huggingface.co/EleutherAI/gpt-neo-1.3B

[58] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al., “Language models are few-shot learners,” Advances in neural information processing systems, vol. 33, pp. 1877–1901, 2020.

[59] Google, “Gemma-2-2B-IT (Hugging Face model),” 2024. [Online]. Available: https://huggingface.co/google/gemma-2-2b-it

[60] Meta, “Llama-2-7b (Hugging Face model),” 2023. [Online]. Available: https://huggingface.co/meta-llama/Llama-2-7b

[61] BigScience, “BLOOM-7B1 (Hugging Face model),” 2022. [Online]. Available: https://huggingface.co/bigscience/bloom-7b1

[62] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “Gptq: Accurate post-training quantization for generative pre-trained transformers,” arXiv preprint arXiv:2210.17323, 2022.

[63] S. Li, X. Ning, K. Hong, T. Liu, L. Wang, X. Li, K. Zhong, G. Dai, H. Yang, and Y. Wang, “Llm-mq: Mixed-precision quantization for efficient llm deployment,” in The Efficient Natural Language and Speech Processing Workshop with NeurIPS, vol. 9, 2023, p. 3.

[64] M. Zhu and S. Gupta, “To prune, or not to prune: exploring the efficacy of pruning for model compression,” arXiv preprint arXiv:1710.01878, 2017.

[65] I. Beltagy, M. E. Peters, and A. Cohan, “Longformer: The longdocument transformer,” arXiv preprint arXiv:2004.05150, 2020.

[66] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang, et al., “Big bird: Transformers for longer sequences,” Advances in neural information processing systems, vol. 33, pp. 17283–17297, 2020.

[67] E. Voita, D. Talbot, F. Moiseev, R. Sennrich, and I. Titov, “Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned,” arXiv preprint arXiv:1905.09418, 2019.

[68] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. Re, C. Barrett,´ et al., “H2o: Heavy-hitter oracle for efficient generative inference of large language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 34661–34710, 2023.

[69] NVIDIA Corporation, “Tensor Memory Accelerator (TMA) CUDA Core Compute Libraries (CCCL),” 2026. [Online]. Available: https://nvidia.github.io/cccl/unstable/cccl/tma.html

[70] L. A. D. Bathen, N. D. Dutt, D. Shin, and S.-S. Lim, “SPMVisor: dynamic scratchpad memory virtualization for secure, low power, and high performance distributed on-chip memories,” in Proceedings of the seventh IEEE/ACM/IFIP international conference on Hardware/software codesign and system synthesis, 2011, pp. 79–88.

[71] L. A. Bathen and N. Dutt, “HaVOC: A hybrid memory-aware virtualization layer for on-chip distributed scratchpad and non-volatile memories,” in Proceedings ofthe 49th Annual Design Automation Conference, 2012, pp. 447–452.

[72] S. Kumar, H. Zhao, A. Shriraman, E. Matthews, S. Dwarkadas, and L. Shannon, “Amoeba-cache: Adaptive blocks for eliminating waste in the memory hierarchy,” in 2012 45th Annual IEEE/ACM International Symposium on Microarchitecture, 2012, pp. 376–388.



## 附录

## A. 摘要

本工件包含了 SMOOTH（一种硬件辅助的细粒度片上内存管理框架）的实现，以及用于对比的基线解决方案（Compiler-Ideal、Capuchin、Gemmini）。该工件将我们定制的片上内存管理机制集成到了开源的 LLMCompass 周期精确模拟器中，以评估推理延迟和能耗。此外，它还包含了所提出的硬件模块（例如，动态内存控制器、早期回收逻辑）的 Verilog RTL 代码，这些代码使用 Yosys 和 ASAP7 预测性 7 nm 标准单元库进行综合，以评估面积、功耗和时序开销。所有基础实验均通过 shell 和 Python 脚本执行，允许复现每个模型的执行指标并生成图 14、16 和 20，以及表 1 和表 2。由于模拟器使用结构元数据而不是执行实际的模型权重，因此该工件的内存效率极高。论文中观察到的一般性能趋势和架构开销在不同的主机上仍然有效。

## B. 工件清单（元信息）

• 算法：硬件辅助的基于块的内存分配和早期回收

• 程序：LLMCompass（基于 Python 的模拟器），SMOOTH RTL（Verilog）

• 编译：Yosys（用于 Verilog RTL 综合），OpenSTA（用于静态时序分析）

• 模型：TinyLLaMA、GPT-Neo、Gemma-2、LLaMA2、Bloom、GPT-3（结构元数据）

• 数据集：工作负载和元数据配置（包含在仓库中）

• 运行环境：Docker（推荐）或带有 Conda 的 Linux（Python 3.9）

• 硬件：标准 x86 多核 CPU，8–16 GB RAM

• 执行：Bash shell 脚本和 Python 脚本

• 指标：Time-to-First-Token (TTFT)、Time-to-Last-Token (TTLT)、能耗、硬件面积、功耗、时序

• 输出：图表（EPS/PNG 格式），用于表格的原始数据日志

• 实验：延迟/能耗模拟以及用于硬件开销的 RTL 综合

• 需要多少磁盘空间（大约）？：∼10 GB（用于模拟日志和综合输出）

• 准备工作流需要多少时间（大约）？：15–30 分钟

• 完成实验需要多少时间（大约）？：在 48 核 CPU 上约 20 小时（取决于主机 CPU 性能）

• 是否公开可用？：是

• 使用的工件流自动化框架？：Docker、Bash/Shell 脚本

• 是否归档（提供 DOI）？：是 (https://doi.org/10.5281/zenodo.20020344)

## C. 描述

1) 如何访问：所有源代码、脚本和配置文件均可在我们的 GitHub 仓库中找到：https: //github.com/skkim-caslab/SMOOTH。

2) 硬件依赖：配备 x86 CPU 和至少 8–16 GB 主存的标准工作站或笔记本电脑。不需要专用硬件（GPU、FPGA、NPU），因为该工件依赖于基于软件的周期精确模拟和逻辑综合。

3) 软件依赖：我们强烈建议使用 Docker，因为它会自动解决旧版硬件综合工具所需的所有系统级依赖（例如 glibc、libreadline）。提供的 Docker 镜像基于 Ubuntu 22.04。如果在主机上原生运行，所需软件包括 Linux 操作系统、Conda (Python 3.9)、PyTorch (v2.0.0)、scalesim==2.0.2（严格执行以防止配置错误）、matplotlib、pandas 和 seaborn。此外，必须在主机上原生安装 Yosys 和 OpenSTA。ASAP7 预测性 PDK 已包含在仓库中。

4) 数据集：该工件评估了多个大型语言模型。由于模拟器使用结构元数据（例如，层、维度、头）对执行进行建模，而不是加载实际的参数权重，因此不需要外部数 GB 大小的数据集。所有工作负载配置元数据均在仓库中本地提供。

5) 模型：模拟轨迹代表了 TinyLLaMA、GPT-Neo、Gemma-2、LLaMA2、Bloom 和 GPT-3 的执行。

## D. 安装

评估者可以选择我们推荐的基于 Docker 的设置或基于本地 Conda 的设置。

选项 1：Docker 环境（强烈推荐）

1) 克隆仓库：

git clone <repository\_url> SMOOTH

2) 从根目录构建 Docker 镜像：cd SMOOTH && docker build -t isca2026\_smooth\_ae .

3) 运行容器并挂载仓库：docker run -it --rm --name smooth\_ae\_env -v \$(pwd):/workspace/SMOOTH isca2026\_smooth\_ae（环境变量 \$SMOOTH\_HOME 会在容器内自动设置。）

## 选项 2：Conda 环境（备选）

1) 克隆仓库并设置环境变量：export SMOOTH\_HOME=/path/to/your/SMOOTH

2) 设置 Python 环境：

```shell
conda create -n smooth_ae python=3.9
&& conda activate smooth_ae
pip install scalesim==2.0.2
matplotlib pandas seaborn
conda install pytorch==2.0.0 -c
pytorch
```

3) 在您的主机系统上安装 Yosys 和 OpenSTA（例如，通过 apt 或从源码构建）。

```shell
cd $SMOOTH_HOME/src/ae/figure20 &&
python plot_energy.py
```

## E. 实验工作流

从 \$SMOOTH\_HOME 目录（在 Docker 容器内或在您的 Conda 环境中）执行以下指令以复现结果：

## 1) 生成基线和 SMOOTH 策略数据：

## 2) 复现延迟图表（图 14 & 16）：

&& python plot\_latency.py

## 3) 复现能耗图表（图 20）：

## 4) 综合硬件模块：

```shell
cd $SMOOTH_HOME/src/verilog/ && bash
run_all.sh
```

## 5) 复现开销表格（表 1 & 2）：

```shell
cd $SMOOTH_HOME/src/ae/table1 &&
python get_area.py
cd $SMOOTH_HOME/src/ae/table2 &&
python get_power.py
```

## F. 评估与预期结果

与图 14、16 和 20 以及表 1 和 2 对应的关键实验结果将由上述脚本在各自的目录中生成。

• 图 14 (TTFT)：展示了 SMOOTH 相比基线实现的归一化 Time-to-First-Token 减少。

• 图 16 (TTLT)：说明了不同 token 长度下整体生成延迟的减少。

• 图 20（能耗）：验证了 SMOOTH 为第 N 个 token 生成提供的能效收益。

• 表 1 & 2（硬件开销）：详细说明 5 个综合硬件模块（address\_check、alloc、bt\_lookup、find\_zero、free）面积、功耗和时序开销的原始输出。结果将确认相对于性能提升，这些开销可以忽略不计。

## G. 实验定制

评估者可以通过修改提供的 shell 脚本中的模拟参数来定制实验。例如，更改输入/输出序列长度或调整目标 SRAM 容量（例如，从 8 MB 更改为 2 MB 或 32 MB）将允许测试 SMOOTH 对不同内存约束的敏感性，这与论文中讨论的敏感性分析相匹配。

## H. 注意事项

由于 LLMCompass 作为确定性周期精确模拟器运行，因此无论主机的绝对计算速度如何，报告的延迟周期都将是一致的。模拟器本身的执行时间可能会因主机 CPU 而异，但最终评估的硬件指标将保持稳定。