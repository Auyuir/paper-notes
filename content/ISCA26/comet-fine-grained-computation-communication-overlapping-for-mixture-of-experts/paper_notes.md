# {Comet}{}: Fine-grained Computation-communication Overlapping for Mixture-of-Experts 论文解析

## 0. 论文基本信息

**作者 (Authors)**: (作者提取失败，待补充)

**发表期刊/会议 (Journal/Conference)**: ArXiv

**发表年份 (Publication Year)**: 2024

**研究机构 (Affiliations)**: (机构提取失败，待补充)

---

## 1. 摘要

**目的**

- 解决 Mixture-of-Experts (MoE) 模型分布式执行中通信开销过大问题（通信占比达 **47%**）。
- 克服现有粗粒度重叠方法的缺陷：
  - 数据分块过小导致 GPU 计算资源利用率低及首尾通信阶段的空闲。
  - MoE 动态路由导致输入形状多变，独立 Kernel 调度无法精确控制硬件资源分配，阻碍无缝重叠。
- 实现细粒度的计算与通信重叠，提升 MoE 层执行效率。

---

**方法**

提出 **Comet** 系统，核心包含两大关键设计：

- **Shared tensor based dependency resolving**（基于共享张量的依赖解析）：
  - 分析通信与计算操作间的 **Shared tensor**，沿数据独立的维度进行分解。
    - Layer0：沿 **M**（Token）维度分解，按源 Rank 排序 Token，优先计算本地数据。
    - Layer1：沿 **N**（Hidden size）维度分解，按列执行 GroupGEMM，提前启动 Top-K reduction。
  - 重新调度子张量至计算 Tile 粒度，消除 Token 级通信与 Tile 级计算的粒度不匹配。
- **Adaptive workload assignment**（自适应工作负载分配）：
  - 采用 **Thread block specialization**，在融合 Kernel 内隔离通信与计算 Thread block，避免通信 I/O 阻塞计算（如 Hopper 架构的 TMA 流水线）。
  - 根据输入形状和模型配置，动态调整分配给生产者（$n_p$）和消费者（$n_c$）的 Thread block 数量，最大化延迟隐藏。
- 系统实现：
  - 基于 **CUTLASS** 模板生成高效 GEMM Kernel，缓存行索引至寄存器减少访存开销。
  - 集成 **NVSHMEM** 库支持 Kernel 内细粒通信，创建跨 GPU 的全局地址空间。

![](figures/overview_v3.pdf)

---

**结果**

在 H800 和 L20 集群上对比 SOTA 系统（Megatron-Cutlass, Megatron-TE, FasterMoE, Tutel）进行评估：

- 单个 MoE 层加速比达 **1.96x**。
- 端到端模型执行平均加速 **1.71x**。
- 通信延迟平均隐藏 **86.5%**（远超 FasterMoE 的 **29.2%** 和 Tutel 的 **68.6%**）。
- 在不同 Token 长度、并行策略（EP/TP）、负载不均衡场景及低带宽环境（L20 集群）下均保持稳定性能优势。

端到端延迟降低幅度对比：

| 对比基线 | Megatron-Cutlass | Megatron-TE | FasterMoE | Tutel |
| :--- | :--- | :--- | :--- | :--- |
| 延迟降低比例 | **34.1%** | **42.6%** | **44.4%** | **31.8%** |

![](figures/f1_overall_v2.pdf)

---

**结论**

- **Comet** 通过细粒度流水线编程模型成功实现了 MoE 架构中计算与通信的无缝重叠。
- 依赖解析机制打破了粗粒度数据依赖，自适应工作负载分配确保了精确的资源调度与延迟隐藏。
- 系统已部署于超万卡生产集群，节省数百万 GPU 小时，并将开源以激发基于 Triton 或 TVM 的进一步优化。

---

## 2. 背景知识与核心贡献

**研究背景与动机**

- 大语言模型通过扩大参数规模提升性能，但面临计算资源受限的瓶颈。
- **Mixture-of-Experts (MoE)** 引入稀疏结构，仅激活部分参数，使模型能扩展至万亿级参数规模。
- 由于单 GPU 无法存储所有 experts，MoE 需将 experts 分布在不同 GPU 上，导致执行 MoE 层时产生大量跨设备数据交换，通信开销平均占总执行时间的 **47%**。
- 现有通信与计算重叠机制存在两大低效瓶颈：
  - **粒度不匹配**：粗粒度分区引发 GPU 空闲；token 级数据传输与 tile 级 GEMM 计算间存在复杂的数据依赖，难以高效融合。
  - **动态负载差异**：MoE 动态路由导致运行时 experts 输入形状各异，将通信与计算封装在不同 stream 的独立 kernel 中，无法在运行时精确分配硬件资源，阻碍无缝重叠。

![](images/overview_v3.pdf)

---

**核心贡献**

- 提出 **Comet** 系统，针对 MoE 架构实现细粒度的计算与通信重叠。
- **Shared tensor based dependency resolving**：
  - 分析通信与计算操作间的共享张量。
  - 沿特定维度分解共享张量，打破粗粒度数据依赖。
  - 重排张量数据与 intra-operator 执行顺序，消除 token 级通信与 tile 级计算的粒度不匹配。
- **Adaptive workload assignment**：
  - 将通信与计算任务整合至融合的 GPU kernel 中。
  - 通过 thread block specialization 隔离通信对计算性能的影响。
  - 动态调整分配给不同负载的 thread blocks 数量，平衡通信与计算延迟，减少重叠气泡。
- **性能表现**：
  - 集成至 **Megatron-LM**，在 Nvidia H800 和 L20 集群上验证。
  - 相比 SOTA MoE 系统，单 MoE 层实现 **1.96x** 加速，端到端 MoE 模型执行实现 **1.71x** 加速。
  - 已部署于超万卡生产集群，节省数百万 GPU 小时，并将完全开源。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体技术架构概述**

Comet 是一个针对 Mixture-of-Experts (MoE) 模型的高效执行系统，其核心架构旨在通过**细粒度的计算与通信重叠**来消除 GPU 空闲时间并提升整体吞吐量。该架构主要包含两大核心设计模块：**基于共享张量的依赖解析** 与 **自适应工作负载分配**。

---

**核心模块一：Shared Tensor Based Dependency Resolving**

该模块通过分析并重组通信与计算操作之间的**共享张量**，解决两者之间的粒度不匹配问题，从而实现细粒度的流水线重叠。
- **张量分解**：沿特定维度将共享张量分解为独立的数据块，打破粗粒度依赖。
- **计算重调度**：将分解后的子张量重新组织为计算块，确保消费者操作能尽早启动。
- **Layer0 (通信-计算管道)**：
  - 消费者为 GEMM，Token 间相互独立。
  - 沿 $M$ (Token) 维度分解共享张量。
  - 按 Token 的源 Rank 排序，优先计算本地 Token，同时并行传输远程 Token。
- **Layer1 (计算-通信管道)**：
  - 消费者包含 top-K reduction，Token 间存在依赖。
  - 沿 $N$ (Hidden size) 维度分解共享张量。
  - 调整 GroupGEMM 执行顺序为列优先，使得前 $T_N$ 列计算完成后即可启动 reduction 与通信，无需等待所有 Expert 计算结束。

---

**核心模块二：Adaptive Workload Assignment**

在依赖解析的基础上，该模块将通信与计算任务融合进单一的 GPU Kernel 中，通过动态资源分配实现无缝的延迟隐藏。
- **Thread block specialization (线程块特化)**：
  - 在 Kernel 内部隔离通信与计算线程块，避免细粒度 I/O 拖累计算效率。
  - 计算线程块复用标准 CUTLASS GEMM 实现，利用 TMA 与 MMA 指令。
  - 通信线程块负责读取计算结果并进行 top-K reduction 及远程传输。
- **Adaptive thread block assignment (自适应线程块分配)**：
  - 动态调整通信线程块 ($n_p$) 与计算线程块 ($n_c$) 的比例。
  - 根据输入 Token 长度、并行策略 (TP/EP) 及硬件环境寻找最优分配点。
  - 预编译多种分配比例的 Kernel，运行时通过元数据 选择最优配置。

---

**底层实现与集成**

- **计算库**：深度集成 CUTLASS 模板生成高效 GEMM kernels，并将 MoE Layer0 的行索引缓存至寄存器以减少全局内存访问。
- **通信库**：采用 NVSHMEM 替代 NCCL，提供跨 GPU 的全局地址空间和细粒度 GPU 发起的通信操作。
- **系统兼容性**：已集成至 Megatron-LM，支持 Hopper、Ampere、Volta 等多种架构，并兼容 Tensor Parallelism 与 Expert Parallelism 混合策略。

---

**架构数据对比**

| 模块 | 优化目标 | 核心机制 | 适用阶段 |
| :--- | :--- | :--- | :--- |
| Dependency Resolving | 消除粒度不匹配 | 张量分解与重调度 | Layer0 & Layer1 |
| Workload Assignment | 隐藏通信延迟 | 线程块特化与自适应分配 | Fused Kernel 内部 |

![](figures/overview_v3.pdf)

### 1. Shared Tensor Decomposition and Rescheduling

**核心概念与问题背景**

- **Shared Tensor 定义**：在 MoE 架构的 Producer-Consumer 管道中，Shared Tensor 作为通信操作与计算操作之间的共享数据缓冲区，既是前者的输出缓冲，也是后者的输入缓冲。
- **粒度不匹配挑战**：
  - MoE 系统中数据传输以 Token 为基本单位。
  - 高性能 GEMM 计算通常以 Tile（如 128x128）为单位组织数据。
  - 单个计算 Tile 所需的 Token 随机分布在多个 GPU 上，导致计算必须等待所有远程 Token 到达后才能启动，延长了数据准备时间。
- **解决思路**：通过分析 Shared Tensor 的访问模式，将其沿特定维度分解，并重新调度计算任务，从而打破粗粒度数据依赖，实现细粒度的通信与计算重叠。

---

**Shared Tensor Decomposition 原理**

分解操作的核心在于寻找消费者算子中数据保持独立的维度进行切分，确保重叠仅在独立数据块间发生。

- **Layer 0 (通信-计算管道) 分解策略**：
  - 消费者算子：GEMM。
  - Shared Tensor 角色：GEMM 的输入矩阵。
  - 维度分析：Token 沿 $M$ (Token) 维度相互独立；沿 $N$ (Embedding) 维度存在乘加归约依赖。
  - 分解操作：沿 $M$ 维度分解 Shared Tensor。
- **Layer 1 (计算-通信管道) 分解策略**：
  - 消费者算子：包含 Top-K reduction。
  - Shared Tensor 角色：GEMM 的输出矩阵。
  - 维度分析：Top-K reduction 沿 $M$ 维度进行归约，导致 Token 间存在强依赖；沿 $N$ 维度元素相互独立。
  - 分解操作：沿 $N$ 维度分解 Shared Tensor。

---

**Rescheduling 算法流程**

分解后的子张量若直接以单行或单列送入计算，会破坏 GEMM 的 Tile 计算效率。因此必须进行重调度。

- **重调度原则**：
  - **对齐 Tile 粒度**：重调度的子张量必须重新组织成 Tile 大小，以维持 GPU 计算效率。
  - **消费者优先**：调度策略优先处理消费者能立即使用的生产者数据，使消费者尽早启动执行。
- **Layer 0 重调度流程**：
  - 基于 Token 的源 Rank 对 Token 进行排序。
  - 设计 GroupGEMM 的 Tile 计算序列，最小化对远程数据的依赖。
  - 优先计算包含本地 Token 的 Tile，同时并行传输其他远程 Token。
- **Layer 1 重调度流程**：
  - 调整 Tile 计算序列，打破按 Expert 顺序执行的模式。
  - 采用列向执行策略，按列处理 GroupGEMM。
  - 一旦计算出 Shared Tensor 的前 $T_N$ 列，立即触发 Top-K reduction 与通信操作，无需等待所有 Expert 计算完成。

![](images/overview_v3.pdf)

---

**输入输出关系与系统作用**

- **输入**：MoE 层原始的通信与计算操作序列、运行时动态路由的 Token 分布、Shared Tensor 的维度信息。
- **输出**：消除粒度不匹配的优化管道结构，以及重新组织后的 Tile 计算序列。
- **在整体中的作用**：
  - 消除通信与计算之间的复杂数据依赖壁垒。
  - 为后续的 Adaptive Workload Assignment 提供规则且一致的重叠模式基础。
  - 在不损害 GPU 计算效率的前提下，最大化细粒度通信延迟隐藏。

### 2. Thread Block Specialized Fused Kernel

**核心概念**

Thread Block Specialized Fused Kernel 是 Comet 系统中实现计算与通信精细重叠的核心底层机制。通过在 GPU 硬件层面实施 Thread block 级别的物理隔离，该机制将通信 I/O 任务与计算任务彻底解耦，从而在实现细粒度流水线重叠的同时，**维持高计算效率**并**消除非确定性延迟**。

---

**实现原理与架构设计**

- **传统垂直融合的缺陷**：传统方法将通信 I/O 嵌入 GEMM 的 prologue 或 epilogue 中，导致重叠不规则，引发非确定性延迟。在 Hopper 架构中，细粒度的远程 I/O 操作会严重破坏基于 Tensor Memory Accelerator (TMA) 的异步计算流水线，导致 kernel 效率大幅下降。
- **Thread block 隔离机制**：Comet 将通信与计算工作负载分配给独立的 Thread block。这种隔离确保了 GEMM Thread block 的实现与融合前的默认 GEMM 完全一致，不受通信任务干扰。
- **硬件资源限制考量**：理论上可将通信 warp 与计算 warp 集成在同一 Thread block 内以减少 global memory 访问，但受限于 warp 数量约束，通信操作无法充分利用带宽；同时，通信 warp 会干扰同块内的计算 warp，因此必须采用物理隔离。

![](images/overview_v3.pdf)

---

**算法流程与执行路径**

以 Hopper 架构为例，每个 SM 仅容纳一个 Thread block，关键数据路径如下：

- **GEMM Thread block 执行流程**：
  - 基于 CUTLASS 模板编译实现。
  - **Producer warp**：利用 async TMA 指令将数据从 global memory 异步加载至 shared memory buffer。
  - **Consumer warp**：发起 tensor core MMA 操作执行矩阵乘法，并将结果写回 global memory。
- **Communication Thread block 执行流程**：
  - 从 global memory 读取 GEMM Consumer warp 产生的计算结果。
  - 执行 top-K reduction 操作。
  - 块内的 warp 将处理后的 token 写入本地 global memory，或通过 NVSHMEM 传输至远程 GPU。

---

**参数设置与自适应分配**

为实现无缝延迟隐藏，Comet 引入自适应 Thread block 分配策略，动态平衡计算与通信资源。

- **核心参数**：设总 Thread block 数量为 $n$，其中 $n_p$ 个作为 producer，$n_c$ 个作为 consumer。
- **最优划分点 $n_p/n_c$**：该比例受输入形状和模型配置（如并行策略）显著影响。
- **运行时调度**：Comet 库包含多个预编译 kernel，每个具有不同的划分点。部署前通过 profiling 提取最优配置并存储为 metadata。运行时根据动态负载直接调用最优 kernel。

| 配置场景 | 参数变化 | 最优 $n_c$ 变化 |
| :--- | :--- | :--- |
| 输入长度变化 | TP=8, M 从 4096 增至 16384 | 18 增至 26 |
| 并行策略变化 | M=16384, TP 从 8 降至 4 | 26 增至 46 |

---

**输入输出关系与整体作用**

- **输入**：经过 Shared Tensor 分解与重排后的细粒度数据块。
- **输出**：完成 GEMM 计算及 top-K reduction 的本地结果，以及已发起远程传输的 token。
- **整体作用**：在 Comet 的整体架构中，Dependency Resolving 机制优化了 MoE 的流水线结构并消除了粒度不匹配。Thread Block Specialized Fused Kernel 则作为最终的执行单元，通过精确的硬件资源控制，确保细粒度通信延迟被有效隐藏，同时保障计算 kernel 的高效运行，是实现端到端 1.71x 加速的关键底层支撑。

### 3. Adaptive Thread Block Assignment

**核心观点**

**Adaptive Thread Block Assignment** 旨在动态分配 GPU thread blocks 给通信和计算任务，以平衡两者的延迟，最大化隐藏延迟并减少 pipeline bubbles。该机制是 Comet 实现细粒度计算通信重叠的硬件层执行保障。

---

**实现原理与算法流程**

- **Thread block specialization**：Comet 摒弃了将通信 I/O 塞入计算 kernel 前后置的垂直融合 方案，转而采用 thread block 级别的隔离。
  - 将通信和计算工作负载分配给独立的 thread blocks，实现硬件资源的精准控制。
  - 避免了 token 级细粒度 I/O 拖累计算 kernel 效率，尤其在 Hopper 架构上保护了异步计算 pipeline 不被长延迟远程 I/O 破坏。
- **算法执行流程**：
  - 设定总 thread blocks 数量为 $n$，其中 $n_p$ 个作为 producer，$n_c$ 个作为 consumer。
  - 寻找最优分割点 $n_p/n_c$，该点受输入形状和模型配置（如并行策略）动态影响。
  - 部署前：对每种配置进行 profile，将最优配置存储为 metadata。
  - 运行时：根据输入和并行策略，读取 metadata，选择并启动最优的 pre-compiled kernel。

---

**参数设置与动态调整机制**

- **核心参数**：总 thread blocks 数量 $n$（通常等于 GPU 的 SM 数量，如 Hopper 架构的 132）、计算 thread blocks 数量 $n_c$、通信 thread blocks 数量 $n_p$。
- **动态调整依据**：
  - **输入 token 长度 ($M$) 变化**：数据量增加时，计算和通信资源需求扩展性不同，导致最优分割点偏移。
  - **并行策略 ($TP$) 变化**：Tensor parallelism 会切分专家权重，改变计算与通信负载比例，显著影响最优分割点。
- **配置变化下的最优 $n_c$ 演变**：
  
| 并行策略 ($TP$) | 输入长度 ($M$) | 最优计算 Thread Blocks ($n_c$) |
| :---: | :---: | :---: |
| 8 | 4096 | 18 |
| 8 | 16384 | 26 |
| 4 | 16384 | 46 |

---

**硬件资源限制与 Kernel 设计**

- **Hopper 架构上的 Kernel 设计**：
  - 每个 SM 仅容纳一个 thread block。
  - **计算 thread blocks**：使用 CUTLASS 编译。Producer warp 利用 async TMA 指令从 global memory 加载数据至 shared memory buffer；Consumer warp 发起 tensor core MMA 操作。
  - **通信 thread blocks**：从 global memory 读取 consumer warp 的计算结果，执行 top-K reduction，随后将 token 写入本地或通过 NVSHMEM 传输至远程 GPU。
- **硬件限制考量**：
  - 理论上可将通信与计算 warps 集成在同一 thread block 以消除冗余的 global memory 访问。
  - 实际受限于 warp 的线程数限制，通信 operator 无法充分利用带宽。
  - 通信 warps 会干扰同一 thread block 内的计算 warps，因此必须进行物理隔离。
- **跨架构移植性**：该 thread block 专用的编程模型可移植至 Ampere 和 Volta 架构，仅需替换对应的计算 thread block 实现。

---

**输入输出关系与系统作用**

- **输入**：经分解和重排后的 shared tensors、运行时输入 token 长度 $M$、并行策略 $TP$。
- **输出**：高度优化的 horizontally-fused kernels，实现计算与通信延迟的精准匹配与无缝重叠。
- **在整体中的作用**：
  - 承接 **Shared Tensor Based Dependency Resolving**，在数据依赖被解除且 pipeline 规律化后，提供硬件层面的资源平衡。
  - 解决 MoE 动态路由导致的计算和通信负载不均问题。
  - 消除 CPU 端调度开销，通过 fused kernel 内部的 thread block 调度维持高计算效率。

![](images/overview_v3.pdf)


---

## 4. 实验方法与实验结果

**实验设置**

- **硬件环境**：测试集群配备 **8张 Nvidia H800 GPUs** (80 GB显存)，GPU间通过 **NVLink** 互连。软件栈包括 **CUDA 12.3**、**NVSHMEM 2.11**、**Pytorch 2.4.0** 以及 **Megatron-LM**。
- **对比基线**：所有基线均基于 **Megatron-LM** 实现，包括：
  - **Megatron-Cutlass**：使用 **CUTLASS grouped GEMM** 实现专家计算。
  - **Megatron-TE**：使用 **Transformer Engine** 库加速。
  - **FasterMoE**：定制化 **All-to-All** 通信以实现重叠。
  - **Tutel**：提供自适应并行、2D分层 **All-to-All** 及快速编解码。
- **评估模型**：实验选取了三种主流开源 **MoE** 模型，配置如下：

| Model | L | E | topk | N | K |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Mixtral 8x7B** | 32 | 8 | 2 | 4096 | 14336 |
| **Qwen2-MoE-2.7B** | 24 | 64 | 4 | 2048 | 1408 |
| **Phi-3.5-MoE** | 32 | 16 | 2 | 4096 | 6400 |

---

**整体性能结果**

![](images/f1_overall_v2.pdf)

- **端到端延迟降低**：相较于基线系统，**Comet** 在端到端执行中显著降低了延迟，具体数据如下：

| Baseline | Latency Reduction |
|:---|:---:|
| **Megatron-Cutlass** | 34.1% |
| **Megatron-TE** | 42.6% |
| **FasterMoE** | 44.4% |
| **Tutel** | 31.8% |

- **性能优势原因**：
  - 实现了充分的计算与通信重叠。
  - 融合内核内的调度大幅减少了 **CPU** 端的开销。
- **基线表现分析**：
  - **Megatron-Cutlass** 与 **Megatron-TE** 性能相近，因均不支持重叠，且 **TE** 存在额外 **API** 调用开销。
  - **Tutel** 通过调度实现部分重叠，但在专家数量多（如 **Qwen2**）时调度开销过大。
  - **FasterMoE** 仅支持专家并行，在专家较小的模型上内核调用时间成为瓶颈。

---

**单层 MoE 详细评估**

- **处理可变输入长度**：
  - 随着输入 **Token** 长度变化，**Comet** 始终保持最短执行时间。
  - 相比基线，平均实现 **1.28x** 至 **2.37x** 的加速。
  - 在 **M** 较小时优势尤为明显，因为融合内核有效消除了主导整体耗时的主机端调度开销。
- **时间分解分析**：
  - **Comet** 平均隐藏了 **86.5%** 的通信延迟，且不影响专家的计算效率。
  - 对比之下，**FasterMoE** 仅隐藏 **29.2%**，**Tutel** 隐藏 **68.6%**。
  - **FasterMoE** 引入的本地索引延长了计算时间；**Tutel** 优化的 **all-to-all** 加剧了本地计算负担。
- **层内并行策略适应性**：
  - 当 **TP** 增大时，基线系统因专家被切分导致小碎片 **GEMM** 增加，计算效率下降。
  - **Comet** 通过共享张量重排维持了高计算效率，消除了权重切换开销，在各并行策略下均保持低延迟。

---

**消融实验与适应性分析**

- **不同 MoE 参数下的性能**：
  - 调整专家数 **E** 和 **topk** 值，**Comet** 均表现优异。
  - 随 **topk** 增加，运行时计算量上升导致耗时增加，但 **Comet** 仍保持 **1.16x** 至 **1.83x** 的加速。
- **负载不均衡场景**：
  - 使用标准差 **std** 衡量 **Token** 分布不均程度（生产环境平均 **std** 为 **0.032**）。
  - 当 **std=0.05** 时，负载不均衡加剧，所有系统耗时均上升，但 **Comet** 依然优于其他系统。
- **不同集群环境扩展性**：
  - 在配备 **8张 L20 GPUs** (46 GB显存)、通过 **PCIe** 桥接互连（带宽约 **25 GB/s**）的带宽受限集群上测试。
  - **Comet** 相较基线实现了 **1.19x** 至 **1.46x** 的平均加速，证明了其在不同硬件环境下的优越性。

---

**开销分析**

- **内存开销**：
  - **Comet** 利用 **NVSHMEM** 分配全局通信缓冲区，大小为 **2MN**（对应 **BF16/FP16**）。
  - 该缓冲区在所有层和专家间共享，所需显存极小，与当前 **GPU** 的大容量显存相比可忽略不计。

| Mem(MB) | Mixtral 8x7B | Qwen2-MoE | Phi3.5-MoE |
|:---|:---:|:---:|:---:|
| M=4096 | 32 | 16 | 32 |
| M=8192 | 64 | 32 | 64 |

---

