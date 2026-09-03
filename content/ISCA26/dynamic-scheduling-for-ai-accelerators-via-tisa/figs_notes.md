# Dynamic Scheduling for AI Accelerators via TISA 图表详解

### Fig. 2: Different execution manners of the fused tiles shown in Figure 1. Shaded regions $( E _ { x } )$ represent latency saved through scheduling. Vertical dashed lines denote synchronization barriers between iterations imposed by static scheduling. $\begin{array} { r } { E _ { 0 } + E _ { 2 } = E _ { 1 } + E _ { 3 } } \end{array}$ illustrates the equivalent latency savings achievable by dual-stage or triple-stage dynamic scheduling.

![05cf9efabbd8f0b944e16a47971964207912f2e2cdee9abb672aaf4eef1e2480.jpg](images/05cf9efabbd8f0b944e16a47971964207912f2e2cdee9abb672aaf4eef1e2480.jpg)

本图直观展示了融合算子（如FlashAttention中的GEMM与Softmax）在不同调度策略下的**执行时序与资源占用情况**，核心论证了**静态同步屏障（synchronization barriers）** 对硬件利用率的限制，以及**语义感知动态调度（semantics-aware dynamic scheduling）** 在消除隐式依赖、压缩执行延迟方面的显著优势。

* **(a) Sequential execution**：展示**顺序执行**模式。Tensor单元（M0, M1）与Vector单元（S）串行工作，无重叠，导致严重的**硬件空闲气泡（idle bubbles）**，执行延迟最高。
* **(b) Static dual-staged pipelined execution**：展示**静态双阶段流水线**。编译器将M0与S重叠，M1置于下一阶段。垂直虚线代表**强制同步屏障（barriers）**，限制了跨迭代的进一步重叠，仅节省延迟 $E_0$。
* **(c) Dynamic dual-staged pipelined execution**：展示**动态双阶段流水线**。移除静态屏障，运行时根据**数据就绪状态（readiness）** 动态调度。例如 $S_i$ 与 $M0_{i+1}$ 实现跨迭代重叠，额外节省延迟 $E_2$，总节省 $E_0 + E_2$。
* **(d) Static triple-staged pipelined execution**：展示**静态三阶段流水线**。M0、S、M1各自独立成阶段，重叠度提升，但依然受限于**垂直虚线同步屏障**，仅节省延迟 $E_1$。
* **(e) Dynamic triple-staged pipelined execution**：展示**动态三阶段流水线**。完全消除固定屏障，实现**最紧凑的跨算子与跨迭代并行（cross-operator and cross-iteration parallelism）**，额外节省延迟 $E_3$，总节省 $E_1 + E_3$。

| 执行模式 | 阶段划分 | 同步屏障 (Barriers) | 跨迭代重叠能力 | 延迟节省量 | 硬件利用率 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| (a) Sequential | 无 | 无 | 无 | 0 | 极低 |
| (b) Static Dual | 2阶段 | 强制 (垂直虚线) | 受限 | $E_0$ | 中等 |
| (c) Dynamic Dual | 2阶段 | 无 (语义驱动) | 强 | $E_0 + E_2$ | 高 |
| (d) Static Triple | 3阶段 | 强制 (垂直虚线) | 受限 | $E_1$ | 较高 |
| (e) Dynamic Triple | 3阶段 | 无 (语义驱动) | 极强 | $E_1 + E_3$ | 极高 |

* **揭示静态调度瓶颈**：证明即使采用高级软件流水线，在面临**非确定性运行时延迟（如DMA backpressure, cache conflicts）** 时，固定屏障仍会导致性能退化与资源闲置。
* **验证TISA语义价值**：动态调度依赖TISA提供的**类型化依赖（typed dependencies）** 与**资源意图（resource intents）**，使硬件能精准区分**真实冲突（true hazards）** 与**可恢复停顿（recoverable stalls）**。
* **确立设计动机**：为论文提出的**TISA指令集**与**动态Tile调度器**提供了坚实的可视化支撑，证明将部分调度权移交运行时是提升异构加速器（Tensor/Vector/DMA）利用率的必然路径。

### Fig. 3: The overall architecture of our integrated framework.

![5922dd03371347d5c15a0cba08b5218ab9af0684e9ff047eb72ebe4fbcccddf8.jpg](images/5922dd03371347d5c15a0cba08b5218ab9af0684e9ff047eb72ebe4fbcccddf8.jpg)

- **整体架构概述**：该图展示了集成框架的三层架构设计，核心在于通过 **TISA Interface Layer** 弥合软件编译与硬件执行之间的语义鸿沟，实现跨异构单元的自适应调度。

- **层级组件与数据流分析**：
  - **Software Layer**：负责模型的高层抽象与语义保留编译。
    - **ML Frameworks**：作为输入源，提供深度学习模型。
    - **Semantic Compiler**：接收框架输入，执行语义保留的编译优化。
    - **TISA Instructions**：编译器的最终输出，作为软硬件契约的中间表示。
  - **TISA Interface Layer**：作为软硬件之间的语义桥梁，提取并传递调度所需的关键元数据。
    - **Semantic Context**：提供算子类型与计算语义。
    - **Scheduling Info**：提供依赖关系与重排序约束。
    - **Resource Hints**：提供资源映射与内存范围提示。
  - **Hardware Layer**：负责实际的动态调度与异构执行。
    - **Dynamic Tile Scheduler**：核心硬件调度器，消费接口层信息以进行动态瓦片调度。
    - **Heterogeneous Exec Units**：包含 Tensor、Vector 和 DMA 等异构执行单元，接收调度指令并执行。

- **架构层级与组件功能映射**：

| 架构层级 | 核心组件 | 功能与数据流向 |
| :--- | :--- | :--- |
| **Software Layer** | **ML Frameworks** | 提供深度学习模型输入，启动编译流程。 |
| | **Semantic Compiler** | 执行语义保留编译，将高层算子降级为硬件可执行的指令。 |
| | **TISA Instructions** | 输出包含丰富语义的瓦片级指令集，作为软硬件契约。 |
| **TISA Interface Layer** | **Semantic Context** | 提取算子类型与计算语义，指导硬件单元匹配。 |
| | **Scheduling Info** | 解析依赖关系与重排序约束，支持动态乱序执行。 |
| | **Resource Hints** | 提供资源映射与内存范围，辅助冲突检测与资源分配。 |
| **Hardware Layer** | **Dynamic Tile Scheduler** | 消费接口层元数据，进行纳秒级动态瓦片调度与冲突解决。 |
| | **Heterogeneous Exec Units** | 接收调度指令，执行 Tensor、Vector 和 DMA 等异构计算任务。 |

- **设计核心洞察**：
  - **语义解耦**：**TISA Interface Layer** 将“计算什么”（**Semantic Context**）与“何时何地执行”（**Scheduling Info** 和 **Resource Hints**）解耦，使硬件调度器无需解析底层指令流即可做出合法性检查。
  - **硬件级调度**：**Dynamic Tile Scheduler** 直接消费 **TISA Interface Layer** 的元数据，在纳秒级周期内完成依赖解析与资源仲裁，避免了软件运行时的微秒级开销。
  - **跨层协同**：从 **ML Frameworks** 到 **Heterogeneous Exec Units** 的全链路语义保留，消除了传统静态流水线中的隐式同步屏障，实现了跨算子与跨迭代的紧凑并行。

### Fig. 4: Per-unit semantic scheduling. Decentralized queues localize dependency checking and prevent unrelated blocking across heterogeneous units. WQ: waiting queue; IQ: issue queue.

![b0ba14e5b1d1e6031d46dddfcd50257c1f8ce7ada784a28e41f7abd53211f034.jpg](images/b0ba14e5b1d1e6031d46dddfcd50257c1f8ce7ada784a28e41f7abd53211f034.jpg)

该图片展示了基于 **TISA** 的**每单元语义调度（Per-unit semantic scheduling）** 架构，核心通过**去中心化队列（Decentralized queues）** 实现异构执行单元的独立仲裁与高效协同。

- **架构核心组件**

| 组件名称 | 英文全称 | 功能描述 |
| :--- | :--- | :--- |
| **接收缓冲区** | **Reception Buffer** | 接收并解析带有语义注释的 **TISA** 指令，提取 **OpType** 与 **UnitMap**。 |
| **等待队列** | **WQ (Waiting Queue)** | 按 **Tensor**、**Vector**、**DMA** 物理隔离，缓存等待依赖解决的指令。 |
| **发射队列** | **IQ (Issue Queue)** | 存储已通过语义冲突检测、准备发射的指令。 |
| **执行单元** | **Exec (Execution Unit)** | 实际执行硬件引擎，包含张量、向量和数据搬运单元。 |

- **四阶段调度流程**

| 步骤编号 | 阶段名称 | 核心动作与机制 |
| :--- | :--- | :--- |
| **1** | **语义路由 (Semantic Routing)** | 指令从 **Reception Buffer** 路由至对应异构单元的 **WQ**，实现指令分类与排队。 |
| **2** | **依赖解决 (Dependency Resolution)** | 从 **WQ** 选取就绪窗口，结合 **in-flight semantic table** 进行冲突检测，合法指令提升至 **IQ**。 |
| **3** | **自适应发射 (Adaptive Issue)** | **IQ** 中的指令在依赖清除后，无序（out-of-order）发射至 **Exec** 单元执行。 |
| **4** | **状态反馈 (Feedback)** | **Exec** 完成后更新语义表，唤醒 **WQ** 中的依赖指令，并自适应调整调度优先级。 |

- **架构设计优势分析**
  - **局部化依赖检查**：将依赖追踪限制在单一执行单元内部，避免了全局状态扫描带来的性能瓶颈。
  - **消除无关阻塞**：异构单元（如 **Tensor** 与 **DMA**）调度相互独立，防止单一单元延迟阻塞其他无关单元。
  - **细粒度并行**：支持跨算子（cross-operator）和跨迭代（cross-iteration）的指令级重叠，突破静态编译调度的同步壁垒。

### 5a3100e384f07c9a9839631370208a96fb4f271542b72d3beb4ba9817517d9d8.jpg

![5a3100e384f07c9a9839631370208a96fb4f271542b72d3beb4ba9817517d9d8.jpg](images/5a3100e384f07c9a9839631370208a96fb4f271542b72d3beb4ba9817517d9d8.jpg)

- **图表基本信息**
  - 该图表展示了 **FlashAttention-3** 在 **Epoch** 加速器与 **NVIDIA H100** GPU 上的 **Hardware Utilization** 对比。
  - 图表采用 3 行 2 列的网格布局，分别对应不同的 **Head dim**（64、128、256）和注意力掩码模式（**w/o causal** 与 **w/ causal**）。
  - X 轴表示 **Sequence Length**（从 512 到 16k），Y 轴表示 **Hardware Utilization**。
  - 图例包含四条曲线：**H100**（基准）、**Epoch (1:8)**、**Epoch (1:16)** 和 **Epoch (1:32)**，代表不同的向量与矩阵计算比例。

- **核心数据对比（以主流配置 Head dim 128 为例）**
  - 以下表格提取了 **Head dim 128 (w/o causal)** 在 **Sequence Length 16k** 时的近似硬件利用率数据：

| 平台与配置 | Hardware Utilization (近似值) | 相对 H100 提升幅度 |
| :--- | :--- | :--- |
| **H100** | 0.65 | 基准 |
| **Epoch (1:8)** | 0.85 | **+30.8%** |
| **Epoch (1:16)** | 0.78 | **+20.0%** |
| **Epoch (1:32)** | 0.60 | -7.7% |

- **趋势与规律分析**
  - **计算比例影响**：**Epoch (1:8)** 配置在所有 **Head dim** 和序列长度下均展现出最高的硬件利用率，显著超越 **H100**。随着比例向 1:16 和 1:32 降低，利用率呈阶梯式下降。
  - **序列长度影响**：随着 **Sequence Length** 的增加，各配置的硬件利用率普遍呈上升趋势并逐渐趋于平稳，表明长序列更能发挥 **TISA** 动态调度的跨迭代重叠优势。
  - **掩码模式影响**：对比 **w/o causal** 和 **w/ causal**，引入因果掩码后整体硬件利用率有所下降，但 **Epoch (1:8)** 依然保持对 **H100** 的明显领先优势。
  - **Head dim 影响**：在 **Head dim 128** 时，**Epoch** 与 **H100** 的性能差距最为显著；而在 **Head dim 256** 时，两者差距缩小，**H100** 在长序列下表现接近 **Epoch (1:16)**。

- **论文结论印证**
  - 图表数据直接印证了论文的核心主张：**TISA** 的语义感知动态调度能够有效重叠 **GEMM-Softmax-GEMM** 计算块，消除静态流水线中的隐式同步屏障。
  - 尽管 **Epoch** 的内存带宽（1.0 TB/s）远低于 **H100**（3.35 TB/s），但凭借卓越的调度效率，**Epoch (1:8)** 在主流配置（**Head dim 128**）下实现了约 **26.4%** 的硬件利用率提升。
  - 这证明了打破静态编译流水线限制、利用运行时语义进行动态重排序，是提升异构加速器实际效能、克服硬件物理参数劣势的关键路径。

### 14dd75c5b7b39e2d5713cbe0a8f7079c80608fdf3b067de3e4e45b44a5b9524e.jpg

![14dd75c5b7b39e2d5713cbe0a8f7079c80608fdf3b067de3e4e45b44a5b9524e.jpg](images/14dd75c5b7b39e2d5713cbe0a8f7079c80608fdf3b067de3e4e45b44a5b9524e.jpg)

- **图表基本信息**
  - **图表类型**：分组柱状图（Grouped Bar Chart）。
  - **Y轴**：执行时间（Time），单位为毫秒（ms），采用对数刻度（log scale）。
  - **X轴**：不同深度学习模型及其评估粒度，包括 **ResNet50 Layer**、**ResNet50 Full**、**BERT Layer**、**BERT Full**、**LLaMA2 Layer**、**LLaMA2 Full**。
  - **图例**：蓝色柱代表 **Triton-TISA**，粉色柱代表 **Torch-Manual**。

- **详细数据对比**
  | 模型与粒度 | Triton-TISA (ms) | Torch-Manual (ms) | 性能提升 (Speedup) |
  | :--- | :--- | :--- | :--- |
  | **ResNet50 Layer** | 8.5 | 9.6 | 1.13× |
  | **ResNet50 Full** | 34.8 | 41.5 | 1.19× |
  | **BERT Layer** | 124 | 127 | 1.02× |
  | **BERT Full** | 1464 | 1754 | 1.20× |
  | **LLaMA2 Layer** | 194 | 221 | 1.14× |
  | **LLaMA2 Full** | 1988 | 2316 | 1.17× |

- **核心结论与分析**
  - **一致性性能提升**：在所有测试的工作负载（**ResNet50**、**BERT**、**LLaMA2**）和粒度（**Layer**、**Full**）下，**Triton-TISA** 的执行时间均严格低于 **Torch-Manual**。
  - **语义指导编译的优势**：即使在缺乏硬件动态调度支持的 **CPU** 环境中，**TISA** 的语义抽象层仍能有效指导编译时优化（如算子融合、循环排序和内存局部性），从而带来显著的性能收益。
  - **架构无关性验证**：**ResNet50** 获得 1.13–1.19× 加速，**BERT** 获得 1.02–1.20× 加速，**LLaMA2** 获得 1.14–1.18× 加速。这证实了 **TISA** 的设计原则具有架构无关性（architecture-agnostic），其语义信息能有效赋能传统编译器。
  - **全局优化收益更显著**：在 **Full** 模型粒度下，**Triton-TISA** 相比 **Torch-Manual** 的绝对时间节省和相对加速比通常更为明显，表明语义指导在复杂全局优化中更具价值。

