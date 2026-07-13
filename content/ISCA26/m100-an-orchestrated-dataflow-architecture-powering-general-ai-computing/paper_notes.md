# M100: An Orchestrated Dataflow Architecture Powering General AI Computing 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Yan Xie, Changkui Mao, Changsong Wu, et al.

**发表期刊/会议 (Journal/Conference)**: ArXiv

**发表年份 (Publication Year)**: 2025

**研究机构 (Affiliations)**: Li Auto

---

## 1. 摘要

**目的**

- 解决现有 GPGPU 架构在 AI 推理中效率低、成本高，以及 Domain-Specific Architectures (DSAs) 灵活性差、难以适应快速演进算法的问题。
- 设计兼具高性能与成本效益的通用 AI 计算架构，满足自动驾驶 (AD)、大语言模型 (LLM) 及智能座舱交互的推理需求。
- 降低硬件复杂度与 BOM 成本，同时保持对未来 Vision-Language-Action (VLA) 等模型的适应性。

**方法**

- **Orchestrated Dataflow Architecture**：采用数据驱动并行执行模型，通过编译器与架构协同设计，在时空维度编排计算与数据移动。
- **Memory Hierarchy**：消除多级缓存，采用软件管理的显式数据流，通过可编程 DMA 引擎控制 TPB 本地内存与 SRAM/DDR 间的数据传输。
- **Operation Granularity**：以 **Tensor** 作为基本数据元素和指令粒度，实现流式处理，消除寄存器文件和显式 load/store 指令。
- **Hardware Design**：
  - 集成 14 个 Tensor Processing Block (TPB) 集群与 1 个 Central Control Block (CCB)。
  - TPB 内部包含 Tensor Computing Unit (TCU)、Configurable Vector Unit (CVU)、High Bandwidth Shared Memory (HBSM) 及 Tensor Walker Unit (TWU)。
  - 采用 2D Mesh Bus 与 Data Ring Bus (DRB) 提供高带宽点对点通信与确定性广播路径。
- **Synchronization**：基于硬件 Synchronization Counters (SCs) 实现高效的双向生产者-消费者同步，支持 barrier、broadcast 等模式。
- **Software Stack**：包含空间-时间调度器、图编译器和后端编译器，实现子图到 TPB 的空间映射与 Tensor 流式处理。

![](images/a71ef2bca7e4fd633b370fe10c5488cbebf3067cf9143d0000ec8286bd8aea87.jpg) *Fig. 5. The high level architecture of M100 NPU.*

**结果**

- **硬件配置对比**：

| Metric | Thor-U | M100 |
|---|---|---|
| DDR Memory Bandwidth | 273 GB/s | 273 GB/s |
| Die Size | 415 mm2 | 399.8 mm2 |
| Process | TSMC N4 | TSMC N5A |

- **UniAD 性能**：在仅启用 8 个集群的情况下，M100 达到 **30 FPS**，相比 Thor-U 的 **7.9 FPS** 实现 **3.8x** 帧率提升，各核心模块加速比达 **1.2x 至 6.3x**。
- **LLaMA2-7B 性能**：
  - Decode 阶段 (W4A16)：延迟 **21.34ms**，与 Thor-U (**20ms**) 持平。
  - Prefill 阶段 (W8A8)：延迟 **79ms**，相比 Thor-U (**154ms**) 实现 **1.95x** 加速。
- **MindVLA (LLM Part) 性能**：Decode 阶段加速 **3x**，Prefill 阶段加速 **2.1x**。
- **硬件利用率**：Profiling 追踪显示 TCU、CVU、DMA 等单元保持持续活跃，任务执行高度重叠，硬件利用率极高。

![](images/07650e5d0ed868809ae18948c761e219c56c4bf30c7e07cb864f166cd64d891e.jpg) *Fig. 15. UniAD Framework.*

**结论**

- M100 通过编译器与运行时软件编排计算与数据移动，成功降低了传统数据流架构的硬件设计复杂度。
- 在不牺牲灵活性的前提下，M100 在关键 AD 工作负载中显著超越主流 GPGPU 平台。
- 验证了软硬件复杂度平衡的编排式数据流架构是应对现代 AI 计算需求演进的有效方向。

---

## 2. 背景知识与核心贡献

**研究背景**

- 深度学习AI技术全面渗透，智能汽车领域对**通用AI计算架构**的需求急剧增长。
- 自动驾驶（AD）、大语言模型（LLM）及智能座舱交互成为当前汽车平台的核心竞争维度。
- 现有主流计算架构在应对AI推理任务时面临瓶颈：
  - **GPGPU架构**：具备出色的通用性和成熟的软件生态，但基于Cache的内存层级导致效率低下、成本高昂，且存在优化不可预测性。
  - **特定领域架构（DSA）**：针对特定AI任务硬连线化，效率极高，但难以适应快速演进的AI算法（如端到端VLA模型），生命周期短且重构成本高。

**研究动机**

- 理想汽车需开发自研AD推理加速芯片，以在提供卓越AD体验的同时严格控制**BOM成本**。
- 现成GPGPU平台（如NVIDIA Orin/Thor）在峰值性能、能效、定制化及**TCO**上存在局限，且未针对特定AD软件栈优化。
- 行业急需一种兼顾**效率**与**灵活性**的中间路线架构，既能高效处理规则可预测的Tensor计算，又能适应未来算法的快速迭代。

**核心贡献**

- 提出**M100 SoC**及**M100 NPU**，创新性地采用**Orchestrated Dataflow Architecture**（编排数据流架构）。
- 通过编译器与架构协同设计，在时空维度编排计算与数据移动，大幅降低硬件复杂度与成本。
- 架构设计实现关键突破：
  - 摒弃多级Cache，采用软件管理的数据流与可编程DMA，实现计算与数据传输的高度重叠。
  - 确立**Tensor**为基本调度与执行粒度，消除寄存器文件与显式load/store指令，实现流式计算。
  - 引入基于硬件**Synchronization Counters (SCs)**的Producer-Consumer同步模型，以极低开销协调大规模并行执行单元。
- 实验验证卓越性能，在AD及LLM推理任务中全面超越主流GPGPU平台。

![](images/a71ef2bca7e4fd633b370fe10c5488cbebf3067cf9143d0000ec8286bd8aea87.jpg) *Fig. 5. The high level architecture of M100 NPU.*

**硬件配置对比**

| Metric | Thor-U | M100 |
| :--- | :--- | :--- |
| **DDR Memory Bandwidth** | 273 GB/s | 273 GB/s |
| **Die Size** | 415 mm² | 399.8 mm² |
| **Process** | TSMC N4 | TSMC N5A |

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体架构概述**

M100是理想汽车推出的面向自动驾驶（AD）、大语言模型（LLM）等通用AI推理的SoC。其核心是自研的NPU，采用**Orchestrated Dataflow Architecture**（编排数据流架构），通过编译器与硬件协同设计，在时间和空间上编排计算与数据移动，摒弃传统多级缓存，以Tensor为基本操作粒度，实现高效率与高扩展性。

**核心设计哲学**

- **计算单元融合**：集成Tensor、Vector、Scalar处理单元，形成统一的计算块。
- **无缓存内存层次**：摒弃多级Cache，采用软件管理的显式数据流和DMA，计算与数据传输重叠。
- **Tensor级操作粒度**：指令定义在Tensor级别，操作数直接流向内存，消除寄存器文件和显式load/store指令。
- **高效数据流同步**：基于硬件**Synchronization Counters (SCs)** 的生产者-消费者模型，支持barrier、broadcast、reduction等模式。
- **集中式指令派发**：指令按序派发，跨执行单元可乱序完成，依赖关系由软件管理。

**SoC与NPU顶层架构**

M100 SoC集成应用CPU、多媒体IP、安全岛及自研NPU。

![](images/6002664168c6717a305e5e30a0c2baf61d900024c258f8c2ff10ede60d5600a7.jpg)

| 模块 | 规格/特性 |
| --- | --- |
| 内存子系统 | 8个LPDDR5X，64GB容量，**273 GB/s**峰值带宽 |
| CPU集群 | 24个ARM Cortex-A78AE核心 |
| 视频输入 | 支持最多11个摄像头，集成ISP |
| 安全与功耗 | Functional Safety Island (FSI)，Power Management Unit (PMU) |

NPU作为SoC的核心推理引擎，包含1个**Central Control Block (CCB)** 和14个**TPB Cluster**。

![](images/a71ef2bca7e4fd633b370fe10c5488cbebf3067cf9143d0000ec8286bd8aea87.jpg) *Fig. 5. The high level architecture of M100 NPU.*

- **互连网络**：
  - **2D Mesh Bus**：提供节点间可扩展、高带宽的点对点通信（单节点对最高256 GB/s）。
  - **Data Ring Bus (DRB)**：提供确定性、高效率的广播路径（聚合带宽256 GB/s），适合多播。
  - **Instruction Chain Bus (ICB)**：连接CCB与TPB Cluster的菊花链总线，用于广播Tensor操作指令。

**NPU核心组件解析**

**Central Control Block (CCB)**

- 运行4核SiFive X280 RISC-V CPU，配备自定义Vector引擎，支持4个并发推理任务。
- 包含32MB片上SRAM（4个8MB bank，4KB交织）。
- 集成DMA引擎，管理DDR与CCB SRAM间的数据传输，并可通过DRB向TPB广播权重。

![](images/ef456eaa300ae8fb3d6c704736168c7118b1b5d9daa9c3493a5867e095c55e37.jpg) *Fig. 6. Architecture of the CCB.*

**TPB Cluster**

- 包含4个**Tensor Processing Block (TPB)**，共享指令缓冲、ICB/DRB节点和1个RISC-V CPU。
- 引入Cluster层级以提升计算密度，实现低延迟、高带宽的近邻通信。

![](images/57963d624bbe626d3b965799a60b7a95ee284c30ad57b03229b2f098be5dd414.jpg)

**Tensor Processing Block (TPB)**

TPB是执行Tensor计算与转换的核心单元。

![](images/e673c06906a90944ab4c037f48918012d8890ec45d64d2968b1662a9146a47c.jpg)

- **High Bandwidth Shared Memory (HBSM)**：2MB SRAM，采用32 bank架构，作为功能单元间的数据交换枢纽，统一数据移动与同步。
- **Tensor Computing Unit (TCU)**：8x64 MAC阵列，支持卷积和矩阵乘法，包含非线性激活流水线。
- **Configurable Vector Unit (CVU)**：模块化向量算术单元，可重构为自定义流水线（如Softmax）。
- **Tensor Walker Unit (TWU)**：生成复杂的非线性地址序列，支持双缓冲，无需专用数据路径即可实现复杂通信。
- **Synchronization Unit (SU)**：管理硬件同步计数器，实现基于状态的更新与监控。
- **DMA Units**：包含**DTDU**（数据转换与搬移）和**GSDU**（由CPU控制的非连续地址Gather/Scatter）。
- **CPU Starter Unit (CSU)**：触发Cluster CPU执行标量/向量处理或控制GSDU。

**软件与编译栈**

M100采用垂直集成的软件栈，核心在于编译器对数据流的编排。

![](images/f43f781de1f0234658de4c53cef8f4007c9b6b9f8d4266bd665b80cd1d8ea223.jpg)

- **Space-time scheduler**：将神经网络子图映射到NPU硬件，在空间上分配TPB，在时间上调度Tensor流。
- **Graph compiler**：执行算子融合、死代码消除、动态内存分配等图优化。
- **Back-end compiler**：生成利用M100硬件特性（Tensor计算、数据移动、同步）的内在指令。
- **Runtime & Firmware**：运行在ARM核心上的推理运行时和驱动，以及运行在RISC-V核心上的固件（采用JIT技术动态生成TPB指令）。

### 1. Orchestrated Dataflow Architecture (编排式数据流架构)

**核心设计哲学**

M100 NPU摒弃了传统CPU和GPU的指令序列执行模型，采用**数据驱动并行执行模型**。在该模型中，张量操作指令被分发至大量执行单元，数据在单元间流动并触发指令执行。为避免传统数据流架构固有的硬件设计复杂性与开销，M100通过编译器在更高的**Tensor粒度**上编排任务执行，将同步与调度的复杂性转移至软件层，从而在硬件极简性与软件可控性之间取得平衡，形成**编排式数据流架构**。

---

**实现原理与架构组件**

- **操作粒度与计算元素**
  - 架构选择**Tensor**作为基础数据元素和指令粒度，构建流式架构。
  - 操作数和结果直接在内存与执行单元间流动，消除传统寄存器文件和显式load/store指令。
  - 集成Tensor、Vector、Scalar处理单元于统一的计算块中，共享本地内存与同步机制。
  - 大张量有效摊销内存延迟，流水线执行最大化吞吐量。

- **内存层次与数据流**
  - 大幅消除多级缓存，每个**Tensor Processing Block (TPB)**配备高带宽本地内存。
  - 数据传输由可编程DMA单元显式控制，实现软件管理的数据移动。
  - 计算与数据传输高度重叠，最大化系统吞吐量。

- **数据流同步机制**
  - 采用基于硬件**Synchronization Counters (SCs)**的**生产者-消费者**模型。
  - 生产者写入数据后更新SC，消费者监控SC并在数据就绪后启动处理；消费者释放缓冲区后更新另一SC通知生产者。
  - 同步操作由专用硬件处理，开销极低，且同步粒度由软件灵活控制。
  - 支持扩展至多代理同步网络，实现barrier、broadcast和reduction等模式。

![](images/8d040780487db0aa5c9be62b3eb60f0c32a6ad265ad9bfc2ce44caa8d16a3760.jpg) *Fig. 3. Two-way producer/consumer synchronization scheme for concurrent processing engines.*

- **指令分发机制**
  - 采用集中式指令分发器，通过**Instruction Chain Bus (ICB)**以菊花链方式广播张量操作指令。
  - 单个处理元素内的指令严格按分发顺序执行，跨不同元素的指令允许乱序完成。
  - 依赖关系的同步管理完全交由软件负责，简化硬件设计。

---

**编译器与运行时编排流程**

- **空间时间调度**
  - 编译器的空间时间调度器将神经网络子图映射到M100 NPU硬件上。
  - 大张量被切分为**mini-tensors**。
  - 空间维度：子图计算算子分配至多个TPBs并行处理。
  - 时间维度：mini-tensors按调度阶段流式传输通过分配的TPBs。

![](images/871fe25f00279863cfc45fb585cc91eb34d87e445126eea4b436c376dda147c1.jpg) *Fig. 14. Space-time scheduler subgraph mapping and tensor streaming on M100.*

- **指令生成与执行**
  - 图编译器执行算子融合、死代码消除及动态内存分配。
  - 后端编译器生成调用硬件特性的内联指令。
  - 运行时固件采用**Just-In-Time (JIT)**编译技术，动态生成优化的TPB指令，并即时计算张量形状与内存地址。

---

**输入输出关系与系统作用**

- **输入输出关系**
  - **输入**：神经网络子图及大规模输入张量。
  - **输出**：经过多层算子处理与张量变换后的推理结果张量。

- **在整体架构中的作用**
  - 提供高硬件利用率和极致的并行执行能力。
  - 以极低同步开销实现计算与数据移动的深度重叠。
  - 支撑Autonomous Driving (AD)和Large Language Models (LLMs)等多样化AI推理工作负载的高效执行，在保持灵活性的同时显著优于传统GPGPU架构。

### 2. 基于硬件同步计数器的无缓存流式内存系统

**核心架构设计理念**
M100 NPU 摒弃了传统的多级缓存架构，采用基于硬件同步计数器的无缓存流式内存系统。该系统通过软件显式控制数据搬移，结合高效的硬件同步机制，实现计算与数据传输的深度重叠，最大化流式处理吞吐量。

![](images/dd339399be078c47916aeb503075edde33ae4d3d4d9eeb675abca063d885f689.jpg) *Fig. 2. Architecture of the M100 NPU memory system without multi-level caches.*

---

**无缓存流式内存系统实现原理**
- **摒弃多级缓存**：M100 NPU 大规模移除了多级缓存，避免缓存一致性带来的扩展性瓶颈和流式性能损耗。
- **HBSM (High Bandwidth Shared Memory)**：每个 TPB 内部集成 2MB 的 HBSM，作为数据存储与通信枢纽。功能单元直接向 HBSM 流式读写数据，无需专用数据路径。
- **分体式内存设计**：HBSM 采用 32 个内存 bank 的设计，每个 bank 支持 32 bytes/cycle 带宽。地址空间以 32-byte 为粒度进行交错，支持多请求者并发访问。
- **显式数据搬移**：数据在 HBSM、SRAM 和 DDR 之间的传输由可编程 DMA 单元显式控制，软件全权管理数据流动。

| 组件 | 参数规格 | 功能描述 |
| :--- | :--- | :--- |
| **HBSM** | 2MB 容量 | TPB 内部数据存储与通信枢纽 |
| **Memory Banks** | 32 个 | 降低端口冲突，维持高带宽 |
| **Bank 带宽** | 32 bytes/cycle | 单 bank 数据吞吐量 |
| **Requester Ports** | 8 个 | 支持多功能单元并发访问 |
| **地址交错粒度** | 32-byte | 实现跨 bank 的均匀分布 |

![](images/558759e6c16b26b415a61b6b6edb33a17f6b907474412f67d764aafe73bb264d.jpg)

---

**Tensor Walker Unit (TWU) 地址生成与参数设置**
- **功能定位**：TWU 负责为 TPB 功能单元生成特定的张量访问地址序列，支持卷积等非线性嵌套循环寻址。
- **参数配置**：
  - **嵌套循环层级**：支持配置多级嵌套循环。
  - **循环变量**：每级循环配置 Initial（初始值）、Step（步长）、Final（终止值）。
  - **地址生成逻辑**：输出地址为所有循环层级 Value 计数器的总和。内层循环每周期无条件递增，外层循环在内层循环达到 Final 时触发递增。
- **双缓冲机制**：通过在外层循环指定带有 buffer offset 的 Step 值，实现两个缓冲区的无缝交替。

---

**硬件同步计数器机制**
- **Synchronization Unit (SU)**：TPB 内部的 SU 管理硬件计数器，跟踪和协调各功能单元的执行状态。
- **Producer-Consumer 模型**：
  - **状态更新**：代理在执行任务时更新自身的执行状态计数器。
  - **状态监视**：另一个代理监视该计数器，判断是否满足依赖关系以继续执行。
  - **双向协同**：Producer 更新生产状态并监视消费状态；Consumer 更新消费状态并监视生产状态，形成高效流水线。
- **同步操作流程**：
  - **Update 请求**：功能单元发出更新请求时，SU 将分配的计数器加 1。
  - **Monitor 请求**：包含一个 expected value（预期值）。仅当计数器达到或超过该值时，SU 才会响应；否则请求单元暂停执行。
- **内存访问与同步绑定**：同步动作与内存访问绑定。当请求者赢得仲裁时，同步动作被触发。此后该访问被视为全局可见，后续请求无法超越。

![](images/8d040780487db0aa5c9be62b3eb60f0c32a6ad265ad9bfc2ce44caa8d16a3760.jpg) *Fig. 3. Two-way producer/consumer synchronization scheme for concurrent processing engines.*

---

**输入输出关系与整体作用**
- **输入**：来自 DDR 或其他 TPB 的 Tensor 数据流，经由 DMA 传输至 HBSM。
- **输出**：功能单元计算后的结果数据流，写回 HBSM 并最终经由 DMA 传回外部存储。
- **整体作用**：
  - **消除缓存开销**：避免缓存一致性问题，降低硬件复杂度。
  - **极致并行度**：通过硬件计数器实现极低开销的 Producer-Consumer 同步，使计算单元与 DMA 保持持续活跃。
  - **Dataflow 执行核心**：HBSM 与 SU 结合，将数据流动与同步控制解耦于传统指令流，构成 M100 Orchestrated Dataflow Architecture 的基础，支撑高硬件利用率的 AI 推理。

### 3. Configurable Vector Unit (CVU) 的多阶段流水线重构

**核心机制与设计理念**

- **Configurable Vector Unit (CVU)** 是 M100 NPU 中 **Tensor Processing Block (TPB)** 的核心功能单元，专门处理非张量收缩类的向量操作。
- CVU 采用模块化设计，内部集成了多个 **single-function vector arithmetic operators**（单功能向量算术单元）。
- 每个底层算子接受一个或两个输入向量流，并产生单一输出流。
- 核心灵活性在于：通过 **TPB instructions** 的配置，CVU 能够动态重构数据通路，既可将输入直接路由通过单个算子，也可串联多个算子构建带有 **intermediate buffers** 的 **multi-stage pipelines**（多阶段流水线）。

---

**多阶段流水线重构原理**

- **指令驱动配置**：**TPB instructions** 充当硬件配置媒介，定义算子间的路由拓扑、流水线深度以及各算子的具体运算类型。
- **Intermediate Buffers 机制**：在构建多阶段流水线时，前级算子的输出并不写回 **High Bandwidth Shared Memory (HBSM)**，而是直接流入级间的 **intermediate buffers**。后级算子从该缓存读取数据，形成紧密耦合的硬件级流水线。
- **数据流执行模型**：CVU 的流水线重构完全契合 M100 的数据流驱动架构。数据在算子间的流动由硬件同步机制保障，一旦前级产生数据并填入缓存，后级即可消费，实现算子级的数据流并行。

---

**典型算法流程：Softmax 计算**

- Softmax 是 Transformer 模型中频繁使用的复合算子，包含指数运算、累加求和、除法归一化等多个步骤。
- 在 CVU 中，Softmax 被拆解并映射到配置好的多阶段流水线上，整个过程由单条 **TPB instruction** 驱动完成。

| 流水线阶段 | 参与运算的向量算子 | 输入数据流 | 输出数据流向 |
| :--- | :--- | :--- | :--- |
| 阶段 1 | 指数运算算子 | 原始输入向量 | 写入 **Intermediate Buffer** |
| 阶段 2 | 累加算子 | 读取阶段 1 缓存数据 | 产生累加和标量/向量 |
| 阶段 3 | 除法算子 | 读取阶段 1 缓存与阶段 2 结果 | 输出最终归一化结果 |

![](images/cf32b7aefeff11c79f048188938ed3a8a00b229c223c298a802583e9b98ef89.jpg)

---

**复杂操作的降级处理**

- 当遇到无法在单一流水线中完全执行的极复杂向量操作时，CVU 支持跨指令的多阶段处理模式。
- **多指令拆分**：复杂操作被编译器分解为多个子任务，每个子任务由一条独立的 **TPB instruction** 控制，分批次配置 CVU 硬件。
- **性能权衡**：这种跨指令处理需要将中间数据写回 **HBSM**，可能因访存延迟略微降低吞吐量。但得益于 CVU 庞大的配置空间和底层高效的向量算子，其性能依然与传统向量核心相当或更优。

---

**在整体架构中的作用**

- **硬件级算子融合**：CVU 的多阶段流水线重构本质上是一种硬件级算子融合机制，针对 Pooling、Softmax、Layer Normalization 等 AI 推理常见操作进行深度优化。
- **缓解内存瓶颈**：通过 **intermediate buffers** 将中间数据保留在算子流水线内部，大幅减轻了 **HBSM** 的带宽压力和端口冲突，使 **Tensor Computing Unit (TCU)** 等重度计算单元能获得更多内存带宽资源。
- **适应算法演进**：模块化和可重构特性使 CVU 能够灵活适应快速演进的 AI 算法（如各类变体 Attention 机制），无需固定硬件逻辑，延长了架构的生命周期。


---

## 4. 实验方法与实验结果

**实验设置**

- **硬件平台对比**：选取 **M100** 与 **NVIDIA Thor-U** 进行基准测试。两者在关键物理指标上高度对齐，确保对比公平性。
  - **DDR Memory Bandwidth**：均为 **273 GB/s**。
  - **Die Size**：**Thor-U** 为 **415 mm2**，**M100** 为 **399.8 mm2**。
  - **Process**：**Thor-U** 采用 **TSMC N4**，**M100** 采用 **TSMC N5A**。
  - **Power Budget**：在相同的功率预算下收集性能数据。

- **Benchmark 选择**：覆盖自动驾驶与智能座舱两大核心场景。
  - **UniAD**：修改版端到端自动驾驶算法，将 **ResNet-101** 替换为 **RegNet**，包含基于 **CNN** 的骨干网络和基于 **Transformer** 的感知/预测模块。
  - **LLaMA2-7B**：标准 **decoder-only transformer** 架构大语言模型，包含 **prefill** 与 **decode** 两个推理阶段。
  - **MindVLA**：理想汽车自研下一代自动驾驶算法，集成了 **LLM** 组件与 **Mixture-of-Experts (MoE)** 架构，评估时采用 **431M** 参数配置的 **LLM** 部分。

- **资源分配与量化策略**：
  - **M100 良率设计**：芯片设计包含 14 个 **clusters**，实验中仅激活 12 个，预留 2 个冗余以提升芯片良率。
  - **多域隔离**：在 **UniAD** 测试中，仅分配 8 个 **clusters** 给自动驾驶任务，其余 6 个保留给座舱域功能，验证多工作负载并发与性能隔离能力。
  - **量化方案**：**LLaMA2-7B** 的 **decode** 阶段采用 **W4A16**，**prefill** 阶段采用 **W8A8**。

---

**结果数据**

- **UniAD 性能对比**：**M100** 在仅使用 8 个 **clusters** 的情况下，全面超越 **Thor-U**。

| Module | M100 (8 clusters active) | Thor-U | M100 Speedup |
| :--- | :--- | :--- | :--- |
| **RegNet** | 13.1 ms | 57.4 ms | **4.4x** |
| **FPN** | 4.23 ms | 5.1 ms | **1.2x** |
| **BEVFormer** | 7.92 ms | 32.83 ms | **4.1x** |
| **TempFusion** | 4.47 ms | 17 ms | **3.8x** |
| **TrackFormer** | 1.27 ms | 7.95 ms | **6.3x** |
| **MapFormer** | 1.46 ms | 6.14 ms | **4.2x** |
| **Frame rate** | **30 FPS** | 7.9 FPS | **3.8x** |

- **LLaMA2-7B 性能对比**：使用 12 个 **clusters**，在计算密集型阶段展现显著优势。

| Phase | M100 (12 clusters active) | Thor-U | M100 Speedup |
| :--- | :--- | :--- | :--- |
| **decode** (W4A16) | 21.34 ms | 20 ms | **0.94x** |
| **prefill** (W8A8) | 79 ms | 154 ms | **1.95x** |

- **MindVLA (LLM Part) 性能对比**：在自研生产级模型上，**M100** 延迟优势进一步扩大。

| Phase | M100 (12 clusters active) | Thor-U | M100 Speedup |
| :--- | :--- | :--- | :--- |
| **decode** | 0.1 ms | 0.3 ms | **3x** |
| **prefill** | 0.84 ms | 1.74 ms | **2.1x** |

---

**配置分析与执行追踪**

论文未提供传统意义上的架构组件消融实验，但通过资源分配策略与执行追踪验证了其设计的有效性。

- **资源分配策略验证**：
  - **UniAD** 测试中，**M100** 以 57% 的算力资源（8/14 clusters）实现了 **30 FPS** 的实时感知帧率，满足高速自动驾驶需求。
  - 剩余算力可并行处理智能座舱任务，证明了 **M100** 架构在多域工作负载下的弹性与资源隔离能力。
  - **12/14 clusters** 的可用性设计证明了其在提升芯片良率的同时，仍能保持极高的性能输出。

- **执行时间线分析**：
  - 通过内部性能分析工具追踪了 **M100** 的指令执行时间线。
  - 追踪结果显示，**CCB** 中的 **DMAs** 以及 **TPB** 内部的 **TCU**、**CVU**、**CSU**、**GSDU** 在大部分采样窗口内保持持续活跃。
  - 各单元的任务执行存在大量重叠，验证了 **Orchestrated Dataflow Architecture** 在计算与数据移动之间实现了极高程度的并行与硬件利用率。

![](images/07650e5d0ed868809ae18948c761e219c56c4bf30c7e07cb864f166cd64d891e.jpg) *Fig. 15. UniAD Framework.*

- **性能瓶颈与优势分析**：
  - 在 **LLaMA2-7B** 的 **decode** 阶段，**M100** 略逊于 **Thor-U** (**0.94x**)。该阶段为内存带宽受限操作，两者 **DDR** 带宽相同，且 **NVIDIA** 针对开源模型有深度优化。
  - 在计算密集的 **prefill** 阶段及自研 **MindVLA** 模型中，**M100** 凭借高效的 **Tensor Computing Unit (TCU)** 与 **dataflow** 同步机制，实现了 **1.95x** 至 **3x** 的显著加速。

---

