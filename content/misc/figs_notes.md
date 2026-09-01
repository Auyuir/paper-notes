# Redwood: A Frontier AI Accelerator Designed, Verified, and Deployed from Scratch in 2 Weeks by AI 图表详解

### Fig. 1: Redwood SoC architecture.

![35949b40b7c9d3ad3d4648a4a226110af095e96c2c42bc5fdfca24e827795e75.jpg](images/35949b40b7c9d3ad3d4648a4a226110af095e96c2c42bc5fdfca24e827795e75.jpg)

- **Redwood SoC** 整体架构采用模块化与层次化设计，核心目标是实现高效的 **空间数据流 (Spatial-Dataflow)** 计算与低延迟的数据调度。
- **外部存储与接口层**：
  - 架构外围分布着 **West L2 LLC**、**North L2 LLC** 和 **East L2 LLC**，作为片上最后一级 SRAM 缓存，通过双向箭头与内部互联，提供高带宽数据暂存。
  - 底部连接 **DDR4 Controller**，通过双向箭头与全局控制单元交互，负责与外部 DRAM 进行大容量数据吞吐。
- **内部互联网络**：
  - 采用 **AXI4 NoC (Network-on-Chip)** 作为核心数据总线，支持高带宽的批量数据传输与低延迟的控制信号分发。
  - 所有外部存储与内部计算单元均通过该 NoC 进行数据路由，实现解耦的内存访问。
- **计算 Fabric 区域**：
  - 核心计算区由 **Tile 阵列**（图中红蓝相间的网格）构成，每个 Tile 包含独立的计算引擎与控制核心，执行矩阵与向量运算。
  - 阵列四周环绕着 **North DMA Engine**、**West DMA Engine** 和 **East DMA Engine**，负责数据在 NoC 与 Tile 阵列之间的高效搬运。
  - 包含 **AI/ML Core** 与 **Compute Memory**，用于特定 AI 任务的加速与本地数据暂存，减少全局内存访问延迟。
- **全局控制单元 (Global Control)**：
  - 位于架构底部，包含 **MCU (Microcontroller Unit)**，作为全局控制核心，负责内核启动与生命周期管理。
  - 集成 **Queuing and Scheduling** 模块，负责任务的排队、分发与时间围栏调度。
  - 包含 **Global DMA Fabric**，协调全局内存到内存的批量传输，与边缘 DMA 引擎协同工作。

| 架构模块 | 核心组件 | 主要功能描述 |
| --- | --- | --- |
| 外部接口 | West/North/East L2 LLC | 片上最后一级 SRAM 缓存，提供高带宽数据暂存 |
| 外部接口 | DDR4 Controller | 管理外部 DRAM 访问，处理大容量数据吞吐 |
| 互联网络 | AXI4 NoC | 核心数据总线，支持批量数据传输与控制信号路由 |
| 计算单元 | Tile 阵列 | 空间数据流计算核心，执行矩阵与向量运算 |
| 数据搬运 | Edge DMA Engines | 边缘 DMA 引擎，负责 NoC 与 Tile 间的数据调度 |
| 全局控制 | Global Control (MCU) | 全局任务管理、内核启动、排队与调度 |

### Fig. 2: Redwood tile architecture.

![2b7554909144192f9e8a138da31274700c440f1b178755bcc3c2ca0120681c38.jpg](images/2b7554909144192f9e8a138da31274700c440f1b178755bcc3c2ca0120681c38.jpg)

- **Redwood Tile Architecture** 整体划分为 **Frontend (FE)** 与 **Backend (BE)** 两大模块，实现控制逻辑与数据计算的物理及逻辑隔离，以支持异构时钟域与激进功耗管理。
- **Frontend (前端)** 专注于控制与编程，运行于较低时钟域：
  - **Control** 接口接收外部全局控制信号。
  - **CDTC (Core Debug Trace & Control)** 负责核心调试、追踪与控制，引导 **CRV** 启动。
  - **CRV (Tile Control Core)** 为基于 RISC-V 的极简控制核心，负责执行内核软件并下发任务。
  - **ITCM (Instruction Tightly Coupled Memory)** 与 **DTCM (Data Tightly Coupled Memory)** 分别为指令与数据紧耦合内存，供 CRV 专属使用。
- **Backend (后端)** 专注于高带宽数据搬运与并行计算：
  - **Message In/Out** 与 **From/To Mesh** 接口处理瓦片间及片上网络 (NoC) 的通信与消息传递。
  - **CTM (Task Manager)** 作为核心任务管理器，桥接 CRV 与后端功能单元，支持乱序任务调度、硬件追踪与日志记录。
  - **Trace Buffer** 用于缓存硬件追踪数据，通过中断向软件发送通知。
  - **DMA 引擎** 包含 **Ingress DMA**、**Egress DMA** 和 **Internal DMA**，负责数据在 **CMEM** 与外部/内部网络间的高效搬运。
  - **计算引擎** 包含 **GEMM** (矩阵乘法)、**SIMD** (单指令多数据流)、**Transpose** (转置) 等专用硬件单元，针对 Transformer 推理工作负载深度优化。
  - **CMEM (Core Memory)** 为 512-KB 的本地共享暂存内存，供计算单元和 CRV 共享访问。
- **数据与控制流机制**：
  - **控制流**：外部 **Control** 经 **CDTC** 传递给 **CRV**，**CRV** 通过 MMIO 将任务列表下发给 **CTM**，随后 **CRV** 可进入空闲状态以降低功耗。
  - **数据流**：**DMA** 引擎从 Mesh 或外部获取数据写入 **CMEM**，计算引擎从 **CMEM** 读取数据进行处理，结果写回 **CMEM** 并通过 **Egress DMA** 传出。
  - **消息流**：**CTM** 通过 **Message In/Out** 直接与其他瓦片的 CTM 通信，实现无需 CRV 介入的硬件级同步、预取与双缓冲控制。

| 模块 | 组件名称 | 英文全称/缩写 | 核心功能 |
|---|---|---|---|
| Frontend | 控制核心 | **CRV** | 执行内核软件，下发任务至 CTM，支持极简 RISC-V 指令集 |
| Frontend | 调试与控制 | **CDTC** | 引导 CRV 启动，处理调试、追踪与外部控制信号 |
| Frontend | 紧耦合内存 | **ITCM / DTCM** | 存储内核指令与操作数据，隔离控制与计算内存访问 |
| Backend | 任务管理器 | **CTM** | 桥接 CRV 与计算单元，管理任务队列、乱序执行与硬件同步 |
| Backend | 数据搬运 | **Ingress/Egress/Internal DMA** | 负责 CMEM 与外部/网络间的数据流转，支持高带宽突发传输 |
| Backend | 计算单元 | **GEMM / SIMD / Transpose** | 执行矩阵运算、向量运算及数据重排，针对 AI 推理优化 |
| Backend | 核心内存 | **CMEM** | 512-KB 本地共享暂存器，供计算单元与 CRV 共享访问 |

### Fig. 3: Standard Redwood tile BE units.

![869f38be789a14a00eb3812aa11425f5c5d4c232b6e2796988fc7c84b75fd031.jpg](images/869f38be789a14a00eb3812aa11425f5c5d4c232b6e2796988fc7c84b75fd031.jpg)

该图详细展示了 **Redwood tile BE (Back End)** 的标准计算单元架构，这些硬件单元专为现代 Transformer 推理工作负载（包括 on-device prefill 和 decode）进行了深度协同设计。

- **MAC (Multiply-Accumulate) 单元**：作为最基础的计算模块，接收 **INT8** 格式的输入 **A** 和 **B**，通过乘法器和加法器进行运算，并由累加器 **acc** 累积，最终输出 **INT32** 结果。
- **GEMM (General Matrix Multiply) 引擎**：用于大规模矩阵乘法运算。采用 **16x16 MAC array** 结构，处理 **A[0:15]** 和 **B[0:15]** 的数据块，实现高吞吐量的矩阵计算。
- **GEMV (General Matrix-Vector Multiply) 引擎**：专为矩阵向量乘法优化。采用 **1x64 MAC array** 结构，处理 **A[0:63]** 和向量 **B[i]**，高度契合 Transformer 中的注意力机制和线性投影需求。
- **FPU (Floating-Point Unit)**：处理浮点运算。支持 **BF16** 数据格式，采用 **pipelined**（流水线）设计，包含 **F0** 至 **F3** 四个处理阶段，确保高频率下的数据吞吐。
- **SIMD (Single Instruction, Multiple Data) 引擎**：用于元素级操作和归约计算。构建于 **16-lane Vector Engine** 之上，包含 **FPU[0]** 到 **FPU[15]**。支持 **coeff.**（系数）、**RegFile**（寄存器文件）和 **LUTs**（查找表）输入，并具备 **folding for reduction tree ops**（归约树操作折叠）能力，以高效处理 Softmax 等复杂算子。

| 计算单元 | 核心结构/阵列 | 数据格式/输入 | 关键特性与功能 |
| :--- | :--- | :--- | :--- |
| **MAC** | 基础乘加器 | **INT8** 输入, **INT32** 输出 | 包含乘法、加法及 **acc** 累加器 |
| **GEMM** | **16x16 MAC array** | **A[0:15]**, **B[0:15]** | 高吞吐量矩阵乘法，支持脉动数据流 |
| **GEMV** | **1x64 MAC array** | **A[0:63]**, **B[i]** | 矩阵向量乘法，优化注意力与投影计算 |
| **FPU** | 4级流水线 (**F0-F3**) | **BF16** 输入与输出 | **pipelined** 设计，支持高精度浮点运算 |
| **SIMD** | **16-lane Vector Engine** | **BF16**, **LUTs**, **RegFile** | 支持 **folding for reduction tree ops**，处理元素级与归约操作 |

- **架构优势**：通过将整数 MAC 阵列（用于 GEMM/GEMV）与多通道 FPU/SIMD 引擎（用于元素级和归约操作）结合，Redwood tile BE 能够直接映射 Transformer 中的核心算子（如 Attention、GEMM、Normalization 和 Activations），从而在硬件层面实现极高的计算效率和能效比。

### Fig. 4: Redwood FE-BE orchestration.

![bb7e96ceb83a233f8bb2b5c78e005f252e41a26c4964136e424c21beefbac84e.jpg](images/bb7e96ceb83a233f8bb2b5c78e005f252e41a26c4964136e424c21beefbac84e.jpg)

该图展示了 **Redwood** 架构中 **Tile Frontend (FE)** 与 **Tile Backend (BE)** 的协同编排机制，体现了控制流与数据流的深度解耦设计。

核心组件与功能映射：
| 区域 | 组件 | 功能描述 | 触发/交互时机 |
|---|---|---|---|
| **Tile Frontend (FE)** | **ITCM** | 存储 Kernel 代码（如 `run_gemv`） | **Boot time** 加载 |
| | **CRV** | Tile 控制核心，执行 Kernel 逻辑 | 运行时解析指令 |
| | **DTCM** | 存储 Function calls 及操作数 | **Runtime** 写入 |
| **Tile Backend (BE)** | **CTM** | 核心任务管理器，接收并调度任务描述符 | 接收 CRV 指令 |
| | **BE units** | 实际执行计算的功能单元（如 GEMV, SIMD） | 接收 CTM 分发 |

工作流程解析：
- **启动阶段 (Boot time)**：
  - **Kernels**（如 `run_gemv`, `run_fused_multiply_add`）被加载至 FE 的 **ITCM** 中。
  - **CRV** 被引导启动，等待后续指令。
- **运行阶段 (Runtime)**：
  - 外部通过 **DTCM** 传入 **Function calls**（包含 `n_tile`, `N`, `k_tile`, `K`, `addr` 等参数）。
  - **CRV** 读取 **ITCM** 中的 Kernel 代码与 **DTCM** 中的调用参数，执行逻辑展开。
  - **CRV** 将计算任务转化为任务描述符（**Descriptors**），通过控制总线发送至 **CTM**。
  - **CTM** 根据描述符类型（如 `gemv`, `simd`）将任务精准分发至对应的 **BE units** 执行。

架构设计优势：
- **控制与计算解耦**：**FE** 处理稀疏控制，**BE** 处理高带宽数据，允许 **FE** 在低时钟域运行或在 Kernel 执行时关闭以**降低功耗**。
- **CRV 极简设计**：**CRV** 仅需实现最小化 **RISC-V** 规范，**Decoder** 无需随 **BE** 功能单元的增删而修改。
- **高效任务调度**：**CRV** 可将任务列表 handed off 给 **CTM** 后进入**空闲状态 (Idle)**，由 **CTM** 独立处理乱序完成、硬件追踪及任务循环，提升整体执行效率。

### Fig. 5: Redwood CTM messaging network.

![359895db840f525dd3c84f9602125a394282f9a4ce4f4d0e0778201b803ae200.jpg](images/359895db840f525dd3c84f9602125a394282f9a4ce4f4d0e0778201b803ae200.jpg)

- **Redwood CTM 消息网络**展示了系统内不同控制单元之间的通信机制，旨在实现无需 **CRV** 或 **MCU** 介入的控制流排序。
- 架构主要由三个核心区域构成：
  - **Redwood Fabric**：包含 **Edge DMA**、**Edge Task Manager** 以及 **North/West/East L2 SRAM**。
  - **AI/ML Core**：包含 **N Cores**，每个核心内部集成 **Tile CRV**、**Tile CTM** 和 **Core DMAs**。
  - **Global Control**：包含 **Global MCU**、**Global Task Manager** 和 **Global DMA**。
- 消息传递路径与通信机制如下表所示：

| 通信源 (Source) | 通信目标 (Destination) | 消息类型/作用 | 视觉标识 |
| :--- | :--- | :--- | :--- |
| **Tile CTM** | **Edge Task Manager** | 协调数据搬运与任务调度 | 粉色箭头 |
| **Edge Task Manager** | **Tile CTM** | 下游数据发送信号 (Go-ahead) | 粉色箭头 |
| **Global Task Manager** | **Edge Task Manager** | 全局任务分发与内存拷贝控制 | 蓝色箭头 |
| **Edge Task Manager** | **Global Task Manager** | 状态反馈与完成中断 | 蓝色箭头 |
| **Global Task Manager** | **Tile CTM** | 核心任务下发与同步 | 绿色箭头 |
| **Tile CTM** | **Global Task Manager** | 核心状态上报与中断请求 | 绿色箭头 |
| **Global MCU** | **Global Task Manager** | 全局控制指令下发 | 黑色箭头 |
| **Tile CRV** | **Tile CTM** | 核心级任务队列提交 | 黑色箭头 |

- **核心优势与机制**：
  - **解耦控制**：通过显式的消息传递，**CTM** 能够独立处理任务排序和同步，大幅降低 **CRV** 和 **MCU** 的干预频率。
  - **灵活配置**：消息机制支持 **fire-and-forget**（即发即弃）和 **acknowledgment-based**（基于确认）两种模式。
  - **高效调度**：编译器利用此消息网络协调 **prefetching**（预取）、**double-buffering**（双缓冲）和 **out-of-order computation**（乱序计算），减少网格内的复杂仲裁需求，将调度逻辑转移至软件栈。

### Fig. 6: Redwood CTM messaging flow example.

![d293c3217fcb3abeb533ac889fc079ba3d66e728cf4dfcc3c3f48dc9c15d0a19.jpg](images/d293c3217fcb3abeb533ac889fc079ba3d66e728cf4dfcc3c3f48dc9c15d0a19.jpg)

该图展示了 **Redwood CTM (Core Task Manager) 消息传递流** 的具体示例，说明了 **Tile CTM** 与 **DMA Engine CTM** 之间如何通过消息机制协同控制数据搬运与计算，从而在不依赖 **CRV** 或 **MCU** 的情况下实现硬件级调度。

- **架构组件与连接关系**
  - **West/East DMA Engine**：分别连接 **West/East LLC**，内部包含 **DMA** 模块与 **DMA Engine CTM**。
  - **TILE_0_0 Backend**：包含 **CMEM**、**Ingress DMA** 与 **Tile CTM**。
  - **数据通路（蓝色箭头）**：**West DMA Engine** 与 **East DMA Engine** 通过 **Ingress DMA** 进行双向数据交互。
  - **消息通路（粉色箭头）**：**DMA Engine CTM** 与 **Tile CTM** 之间建立直接的消息通信链路。

- **消息与数据流协同机制**
  - **控制流解耦**：**Tile CTM** 向 **West/East DMA Engine CTM** 发送交错消息（interleaved messages），控制数据下发节奏。
  - **任务同步**：**DMA Engine CTM** 在接收到“go-ahead”消息前处于 **wait_msg** 状态，收到消息后才释放并执行下一个 **transfer** 任务。
  - **双向通信**：消息流支持反向传递，**DMA CTM** 可通知 **Tile CTM** 将数据向下游发送。

- **时序与操作调度分析**

| 组件 | 操作序列 | 时间轴数据块 |
|---|---|---|
| **West DMA Engine CTM** | `wait_msg(1)` → `transfer(TILE_0_0, 0x0000, 100)` → `wait_msg(2)` → `transfer(TILE_0_0, 0x2000, 100)` | 粉色块、青色块 |
| **Tile CTM** | `send_msg(west, 1)` → `ingress(100)` → `send_msg(east, 1)` → `ingress(100)` → `send_msg(west, 2)` → `ingress(100)` → `send_msg(east, 2)` → `ingress(100)` | 触发各阶段 Ingress 与 Transfer |
| **East DMA Engine CTM** | `wait_msg(1)` → `transfer(TILE_0_0, 0x0000, 100)` → `wait_msg(2)` → `transfer(TILE_0_0, 0x2000, 100)` | 紫色块、绿色块 |
| **Ingress DMA** | 连续执行四次 `ingress(100)` 操作 | 粉色、紫色、青色、绿色数据块 |

- **核心设计优势**
  - **降低仲裁复杂度**：通过显式的消息流量控制（Explicit traffic control），减少了 **mesh** 网络中复杂的硬件仲裁逻辑。
  - **调度软件化**：将调度逻辑转移至软件栈（**software stack**），由编译器利用消息机制协调 **prefetching**、**double-buffering** 和 **out-of-order computation**。
  - **灵活配置**：消息传递机制支持 **fire-and-forget** 或 **acknowledgment-based** 模式，适应不同场景需求。

### Fig. 7: Redwood dispatch and kernel execution flow.

![324527900adfb64ea479fb724a4811d0329167bd16abc57fa11e172e66cc258b.jpg](images/324527900adfb64ea479fb724a4811d0329167bd16abc57fa11e172e66cc258b.jpg)

该图展示了 **Redwood** 加速器的 **dispatch and kernel execution flow**（调度与内核执行流程），详细描绘了从主机处理器到外部内存，再到全局控制核心与计算瓦片阵列的数据与控制交互机制。

- **核心组件架构**
  - **Host Processor**：发起计算请求，向全局控制核心发送 **Dispatch ID + Args**。
  - **External Memory**：存储 **Global Dispatch Set A**（包含多种 Dispatch Type）、**Tile Kernel Program Set A**（包含多个 Kernel）、**Input Data** 与 **Output Data**，并预留 **Reserve Memory Region**。
  - **Redwood Global**：包含 **Global RISC-V**（即 **MCU**），配备 **DTCM** 与 **ITCM**，以及负责全局调度的 **Global Task Manager**。
  - **Redwood Tile Array**：由多个计算瓦片（Tile）组成，每个瓦片包含 **Tile RISC-V**（配备 **ITCM** 与 **DTCM**）和 **Tile BE**（后端计算单元）。
  - **DMA Engines**：包含 **West DMA Engine** 与 **East DMA Engine**，分别配备 **West DMA Task Manager** 与 **East DMA Task Manager**，负责数据搬运。

- **数据与控制流向**
  | 流向路径 | 数据类型/控制信号 | 视觉标识 |
  |---|---|---|
  | Host Processor → Global RISC-V (DTCM) | **Dispatch ID + Args** | 绿色细箭头 |
  | External Memory → Global RISC-V (ITCM) | **Global Dispatch Set A** | 蓝色箭头 |
  | External Memory → Tile RISC-V (ITCM) | **Tile Kernel Program Set A** | 红色箭头 |
  | External Memory → West DMA → Tile Array | **Input Data** | 绿色粗箭头 |
  | Tile Array → East DMA → External Memory | **Output Data** | 紫色粗箭头 |
  | Global Task Manager → DMA Task Managers | **DMA 任务配置与调度** | 橙色细箭头 |
  | Global RISC-V (DTCM) → Tile RISC-V (DTCM) | **Kernel ID + Operands** | 橙色细箭头 |
  | Tile RISC-V (DTCM) ↔ Tile BE | **任务分发与状态反馈** | 橙色双向箭头 |

- **执行流程步骤**
  - **预加载阶段**：主机将 **DP set** 加载至 **MCU ITCM**，将 **kernel set** 加载至所有 **Tile ITCMs**。
  - **触发调度**：主机将 **Dispatch ID** 和 **Operands** 写入 **MCU DTCM** 并触发 **MCU**。
  - **全局配置**：**MCU** 执行 **Dispatch Program**，配置内部路由表、**Global Task Manager**（用于预取和内存拷贝）以及 **DMA Task Managers**。
  - **内核启动**：**MCU** 将 **Kernel ID** 和 **Operands** 写入目标 **Tile DTCM**，唤醒 **Tile RISC-V** 执行内核。
  - **数据流转**：**West DMA** 将输入数据搬运至瓦片阵列，计算完成后，**East DMA** 将结果写回外部内存的 **Output Data** 区域。
  - **完成中断**：**DP** 执行完毕后，**MCU** 向主机发送中断，通知执行完成并释放缓冲区。

### Fig. 8: Redwood Nano FPGA placement and instantiation hierarchy.

![76805ede9df1aa74b859b43b3b4f05ff86a12a7db39c4fc29325f81c63d2353f.jpg](images/76805ede9df1aa74b859b43b3b4f05ff86a12a7db39c4fc29325f81c63d2353f.jpg)

- **图片整体概述**：该图展示了 **Redwood Nano** 在 **AMD Versal VPK180 FPGA** 上的物理布局布线（左图）与实例化层次结构（右图），直观反映了 **2 × 2 tile array** 及 **DMA engines** 的硬件映射关系。
- **左侧物理布局分析**：
  - 展示了 FPGA 内部的 **Placement and Routing** 结果。
  - 不同颜色区块（如蓝色、粉色、橙色）代表不同的逻辑模块在硅片上的物理分布。
  - 密集的布线网络体现了 **250 MHz** 时钟频率下的信号互联与资源占用情况。
- **右侧层次与逻辑划分分析**：
  - 展示了 **Instantiation Hierarchy**，清晰划分了 **u_npu**、**u_compute** 和 **u_fabric** 等核心层级。
  - **u_fabric** 内部包含四个计算 Tile（对应 **2 × 2 tile array**），分别以粉色、青色、橙色和红色区块标识。
  - 每个 Tile 内部进一步划分为 **u_be**（后端计算与数据通路）和 **u_crm**（控制与路由管理）。
  - 边缘区域部署了四个 **DMA engines**（**u_west_dma**, **u_north_dma**, **u_east_dma**, **u_south_dma**），负责与外部存储的数据交互。
- **模块与接口映射**：

| 模块层级 | 实例名称 | 功能描述 | 物理/逻辑特征 |
| --- | --- | --- | --- |
| 顶层封装 | **u_npu** | 神经网络处理器顶层 | 包含计算与 DMA 子系统 |
| 计算子系统 | **u_compute** | 核心计算单元 | 包含 Fabric 与边缘 DMA |
| 计算阵列 | **u_fabric** | 2 × 2 Tile 阵列 | 包含四个独立 Tile 实例 |
| 边缘接口 | **u_west_dma** / **u_north_dma** | 西侧/北侧 DMA | 优先保障 **Ingress read bandwidth** |
| 边缘接口 | **u_east_dma** | 东侧 DMA | 优先保障 **Egress write bandwidth** |
| 边缘接口 | **u_south_dma** | 南侧 DMA | 辅助数据搬运与同步 |

### Fig. 9: Representative life cycle of a traditional ASIC program.

![a31331d5614b68be6487ab55fce6acfe18499385c3679e0885204c5a3c9207f9.jpg](images/a31331d5614b68be6487ab55fce6acfe18499385c3679e0885204c5a3c9207f9.jpg)

- **图片核心主旨**：图示揭示了**传统 ASIC 项目**高度串行、长周期的生命周期，以及为维持发布节奏而采用的多世代流水线开发模式。

- **单一世代（Gen. N）开发流程**：
  - **架构定义**：从 **Architectural Spec.** 到 **u-Architectural Spec.**，依次经历 **Architecture Freeze** 与 **u-Architecture Freeze**（雪花图标标识）。
  - **前端实现**：**RTL Development** 与 **Design Verification** 并行开展，最终锁定 **RTL Freeze**。
  - **后端流片**：进入 **Backend** 物理设计阶段，最终完成 **Tape-out**（红色文字标识）。

- **多世代流水线重叠机制**：
  - 传统硬件团队通过**流水线开发（pipelining development）** 掩盖单世代长延迟，以维持 **9-12 个月**的硬件发布周期。
  - 各世代开发状态与阶段映射如下：

| 世代节点 | 核心执行阶段 | 关键里程碑/状态 |
| :--- | :--- | :--- |
| **Gen. N-1** | **Backend** 物理设计 | 推进至 **Tape-out** |
| **Gen. N** | **RTL Development** 与 **Design Verification** | 达成 **RTL Freeze** |
| **Gen. N+1** | **Architectural Spec.** 与 **u-Architectural Spec.** | 达成 **Architecture Freeze** 与 **u-Architecture Freeze** |

- **传统流程的结构性缺陷**（结合论文上下文分析）：
  - **变更成本极高**：流程中存在严格的**冻结（Freeze）** 机制， collateral 一旦冻结，修改只能推迟至下一代。
  - **软硬件协同失效**：漫长的串行周期导致硬件定型时，目标 AI 工作负载可能已发生偏移，无法实现真正的 **HW-SW co-design**。
  - **被迫过度设计**：为对冲未来负载的不确定性，设计者被迫增加通用特性（generality added as a hedge），导致**性能、功耗与面积（PPA）** 妥协。

### Fig. 10: Architect Labs Platform and end-to-end chip design flow.

![e797f87415806ad4f07079554627d79b9469cbf288259090402c61930aa06870.jpg](images/e797f87415806ad4f07079554627d79b9469cbf288259090402c61930aa06870.jpg)

该图展示了 **Architect Labs Platform (ALP)** 驱动的端到端芯片设计流程，实现了从架构规范到 **Tape-out** 的全自动化与并行化。

- **核心输入与专家交互**
  - **Architectural Spec.** 与 **u-Architectural Spec.** 作为顶层输入，定义芯片的高层架构与微架构规范。
  - **Human Experts** 通过 **Architect Labs Platform** 介入设计，负责调整设计意图并接收系统反馈，无需干预底层实现。

- **自动化前端执行 (Automated Frontend Execution)**
  - **RTL Design**：自动生成寄存器传输级代码。
  - **Functional Verification (UVM)**：基于通用验证方法学自动生成测试平台与用例。
  - **Formal Verification (SVA)**：利用系统验证断言进行形式化验证。
  - **Firmware & Software Stack**：协同生成固件与底层软件栈。
  - 上述模块的输出统一汇入 **Coverage Closure, Performance Validation, Formal Proving** 阶段，确保代码与功能覆盖率达到标准。

- **后端执行与流片流程**
  - **Synthesis, DFT, and Early Timing**：完成逻辑综合、可测试性设计 (DFT) 及早期时序分析。
  - **Backend Execution**：执行物理设计、布局布线等后端流程。
  - **Tape-out**：最终交付流片。

- **多维反馈与优化机制**

| 反馈类型 | 来源阶段 | 目标阶段 | 核心作用 |
| --- | --- | --- | --- |
| **Software Feedback** | Firmware & Software Stack | Human Experts / Platform | 指导架构与规范调整 |
| **Performance Feedback** | Synthesis / Backend | Automated Frontend Execution | 优化前端设计与性能 |
| **Shallow PD Feedback** | Backend Execution | Automated Frontend Execution | 提供浅层物理设计约束 |
| **Deep PD Feedback** | Backend Execution | Automated Frontend Execution | 提供深层物理设计约束 |

### 0b9ac1812176dd866eff4e1034614dd5e6eef3f862ac7a392caefd1b27703ce8.jpg

![0b9ac1812176dd866eff4e1034614dd5e6eef3f862ac7a392caefd1b27703ce8.jpg](images/0b9ac1812176dd866eff4e1034614dd5e6eef3f862ac7a392caefd1b27703ce8.jpg)

**图片总体概述**
该图片为 **Fig. 11: Redwood design process history**，以甘特图形式展示了 Redwood 芯片从架构定义到 FPGA 部署的完整设计历史，时间跨度为 **22天**。图表分为 **Architecture**（架构设计）和 **Automated Frontend Execution**（自动化前端执行）两大核心阶段，并通过四种颜色区分任务类型：**Specification Development**（深蓝色）、**Closure and Optimization**（浅蓝色）、**RTL Design, Verification, Synthesis, Timing Closure**（紫色）以及 **Firmware, Kernels, Live-testing**（橙色）。

**Architecture 阶段分析**
- **SoC Architecture**：贯穿整个架构阶段，从 **Spec v1** 迭代至 **Spec Final**。
- **Task Manager**：经历 **Spec v1** 到 **Spec v4** 的快速迭代，最终收敛于 **Spec Final**。
- **General Infrastructure**：直接由 **Spec v2** 演进至 **Spec Final**。
- **Matrix Engine**：完成 **Spec v1**、**Spec v2** 至 **Spec Final** 的三次迭代。
- **SIMD Engine**：经历 **Spec v1**、**Spec v2**、**Spec v3** 至 **Spec Final** 的四次迭代。
- **Local Memory**：从 **Spec v2** 迭代至 **Spec Final**。
- **DMA v1 / DMA v2**：分别完成 **Spec v1** 和 **Spec v2** 的初始定义，随后统一收敛至 **Spec Final**。
- **Memory Interfaces**：经历 **Spec v1**、**Spec v2**、**Spec v3** 至 **Spec Final** 的完整迭代。

**Automated Frontend Execution 阶段分析**
- **General Infrastructure**：从 **RTL** 开发到 **Generated RTL**，再到 **Closure** 和 **Integrated RTL**，最终进入 **Selected baseline** 和 **Optimization**。
- **Task Manager**：完成 **Generated RTL** 和 **Closure**，随后进行 **Workload mapped**。
- **Matrix Engine**：经历 **Generated RTL**、**Sim**、**DSP**，完成 **GEMV closure** 和 **Performance alignment**，最终实现 **Workload mapped**。
- **SIMD Engine**：从 **Generated RTL** 到 **Sim**、**DSP**，完成 **Performance alignment** 和 **Perf alignment**，经过 **Perf** 优化和 **DSP updates**，最终达到 **Selected baseline**。
- **Local Memory**：完成 **Generated RTL** 和 **Timing closure**，进入 **RTL rendering** 和 **Selected baseline**。
- **DMA v1 / DMA v2**：分别完成 **Generated RTL** 和 **Functional RTL**，最终达到 **Selected baseline** 和 **Optimized RTL**。
- **Memory Interfaces**：从 **RTL development** 到 **AXI4 integration**，完成 **e2e interface**，达到 **Hybrid baseline** 和 **Throughput closure**。
- **SoC Integration**：完成 **RTL** 开发，达到 **Selected baseline** 和 **Hybrid baseline**，最终实现 **Platform qualified**。
- **Workload Mapping**：完成 **e2e RTL/RTL/DPs** 和 **Qwen** 部署，生成 **Inference RTL**。
- **FPGA Bring-up**：在后期完成最终的 **FPGA bring-up**。

**关键模块设计周期与阶段映射**
| 模块名称 | 架构规范迭代 (Spec) | 前端执行核心阶段 (Frontend Execution) | 最终状态 |
| --- | --- | --- | --- |
| **SoC Architecture** | v1 至 Final | - | Spec Final |
| **Task Manager** | v1 至 v4 | Generated RTL, Closure, Workload mapped | Workload mapped |
| **Matrix Engine** | v1 至 Final | Sim, DSP, GEMV closure, Performance alignment | Workload mapped |
| **SIMD Engine** | v1 至 Final | Sim, DSP, Perf alignment, DSP updates | Selected baseline |
| **Local Memory** | v2 至 Final | Timing closure, RTL rendering | Selected baseline |
| **DMA v1/v2** | v1/v2 至 Final | Functional RTL, RTL rendering | Optimized RTL |
| **Memory Interfaces** | v1 至 Final | AXI4 integration, e2e interface | Throughput closure |
| **SoC Integration** | - | RTL, e2e interface | Platform qualified |
| **Workload Mapping** | - | e2e RTL/DPs, Qwen | Inference RTL |

**设计流程特征总结**
- **高度并行化**：架构规范（Spec）与前端执行（RTL/Verification）在时间轴上存在显著重叠，打破了传统串行流。
- **快速迭代收敛**：各模块在 **Day 15** 前基本完成 **Spec Final**，随后全面转入自动化 RTL 生成与优化。
- **软硬协同验证**：后期（**Day 17-22**）集中进行 **Workload Mapping** 和 **FPGA Bring-up**，验证 **Qwen** 模型推理，实现 **Inference RTL** 闭环。
- **AI 驱动效率**：整个流程在 **22天** 内完成从架构定义到 FPGA 部署，印证了 AI 系统在硬件设计中的端到端加速能力。

### 22bdd0f093e790372b924e619a9638f98e98d805f0ddd829d68aa0256ad360ec.jpg

![22bdd0f093e790372b924e619a9638f98e98d805f0ddd829d68aa0256ad360ec.jpg](images/22bdd0f093e790372b924e619a9638f98e98d805f0ddd829d68aa0256ad360ec.jpg)

- **图表概述**：该图展示了 **Redwood** 项目生命周期内 AI 系统的 **merge-commit** 历史，直观反映了从架构设计到 **FPGA** 部署各阶段的代码提交活跃度。
- **坐标轴定义**：
  - **X轴**：**Day**（项目天数），跨度为 1 至 22 天。
  - **Y轴**：**Commits per day**（每日提交数），范围从 0 到 120。
- **阶段趋势分析**：
  - **Architecture（架构设计）**：在第 1 至 15 天持续活跃，初期与 **RTL / DV** 并行，后期维持中低频波动。
  - **RTL / DV（RTL设计与验证）**：在第 3 至 4 天达到初期峰值，随后与架构设计交替推进，整体呈下降趋势。
  - **Closure / optimization（收敛与优化）**：在第 7 天左右出现局部小高峰，随后在 10 至 15 天期间保持低频稳定提交。
  - **Workload mapping / FPGA（工作负载映射与FPGA部署）**：从第 16 天开始呈指数级攀升，在第 20 天达到全局最高峰，随后在第 22 天迅速回落至基线。
- **关键数据洞察**：
  - **初期峰值**：**RTL / DV** 阶段在第 3-4 天达到约 **33 commits/day**。
  - **全局峰值**：**Workload mapping / FPGA** 阶段在第 20 天达到约 **108 commits/day**（论文正文提及单日最高达 **115 merge commits**），标志着 **firmware** 和 **kernels** 的密集优化。
  - **流程特征**：前 15 天为硬件设计与验证期，后 7 天为软件映射与 **FPGA** 验证期，体现了高度并行的 **AI-native** 开发流。

| 阶段 (Phase) | 活跃周期 (Day) | 峰值提交数 (Commits/day) | 峰值出现时间 (Day) |
| :--- | :--- | :--- | :--- |
| **Architecture** | 1 - 15 | ~22 | 10, 13 |
| **RTL / DV** | 1 - 15 | ~33 | 3, 4 |
| **Closure / optimization** | 1 - 15 | ~13 | 7 |
| **Workload mapping / FPGA** | 16 - 21 | ~108 (正文称115) | 20 |

### Fig. 13: Autonomous SIMD engine exploration path.

![f796c5e81b11a5d06a34e22542c26c77811582fffdc33543aa6d53f8fa9d898d.jpg](images/f796c5e81b11a5d06a34e22542c26c77811582fffdc33543aa6d53f8fa9d898d.jpg)

- 该图展示了 **Redwood** 项目中 **Autonomous SIMD engine exploration path**（自主 SIMD 引擎探索路径），描绘了从 **Specification Drop** 到 **Final Design** 的自动化迭代流程。整个探索过程被划分为三个核心区域（**Regime**），并行或串行推进以优化 **PPA**（Power, Performance, Area）和验证覆盖率。

- **Alignment area regime**（性能对齐、时序与面积优化）：
  - 起点为 **Timing closure**（对齐率 0.84%，时序 0.00）。
  - 分支一：优化 **Lane latency 8 to 7**（对齐率 0.702），随后进行 **Deferred operation 2 to 1**（对齐率 0.704），最终达成 **Balanced layout**（面积影响 0.0%）。
  - 分支二：优化 **Lane latency 7 to 6**（对齐率 0.707）。

- **Physical-only regime**（仅时序与面积边界约束）：
  - 起点为 **Physical seed**（仅包含时序和面积数据）。
  - 分支一：执行 **GDFix mapping**（对齐影响 0.0%），随后进行 **Area and timing check**（0.004，0.0% 资源），最终归档至 **Physical archive**。
  - 分支二：执行 **Retiming**（0.0% 资源），随后进行 **Area and timing check**，最终归档至 **Physical archive**。

- **Verification regime**（代码覆盖率与功能有效性验证）：
  - 起点为 **Coverage seed**（**RTL coverage** 87.00%）。
  - 分支一：通过 **PR #124**（**RTL coverage** 92.00%）提升至 **PR #200**（**RTL coverage** 95.00%），最终达成 **Verification signoff**（127 of 127 tests pass）。
  - 分支二：通过 **Directed test closure**（Tier 1-3 of 128 tests pass）提升至 **PR #200**，最终达成 **Verification signoff**。

- 核心数据指标汇总：

| 探索阶段 (Regime) | 关键节点/操作 | 核心指标/状态 |
| :--- | :--- | :--- |
| **Alignment area** | Timing closure | Alignment 0.84%, timing 0.00 |
| **Alignment area** | Lane latency 8 to 7 | Alignment 0.702 |
| **Alignment area** | Deferred operation 2 to 1 | Alignment 0.704 |
| **Alignment area** | Balanced layout | Area impact 0.0% |
| **Physical-only** | GDFix mapping | Alignment impact 0.0% |
| **Physical-only** | Area and timing check | 0.004, 0.0% resource |
| **Verification** | Coverage seed | RTL coverage 87.00% |
| **Verification** | PR #124 | RTL coverage 92.00% |
| **Verification** | PR #200 | RTL coverage 95.00% |
| **Verification** | Verification signoff | 127 of 127 tests pass |

- 图表意义分析：
  - 该流程图直观展示了 **Architect Labs AI system** 如何在无需人工干预的情况下，自主遍历 **performance-area-timing** 搜索空间。
  - 通过多分支并行探索，系统能够生成具有不同控制路径、数据路径和状态机的 **RTL** 候选方案，突破了传统人工设计的认知局限。
  - 验证阶段从 87.00% 的初始覆盖率自动提升至 95.00% 并实现 100% 测试通过，证明了自动化 **UVM** 和形式验证方法的有效性。

### d32cc2b4355fef4e9a2f242fe02b388ac6301203fc231218889c0eb72e6f8df6.jpg

![d32cc2b4355fef4e9a2f242fe02b388ac6301203fc231218889c0eb72e6f8df6.jpg](images/d32cc2b4355fef4e9a2f242fe02b388ac6301203fc231218889c0eb72e6f8df6.jpg)

- **图片整体概述**：该图展示了 AI 系统对 **SIMD engine** 进行微架构探索与优化的全过程，涵盖 **performance**、**area**、**timing** 以及 **code coverage** 四个维度的自动化迭代。
- **左上子图：3D 搜索空间探索 (Search walk)**：
  - 展示了 AI 系统在三维空间中的搜索路径，坐标轴分别为 **Performance alignment**、**Area efficiency** 和 **Timing margin, WNS (ns)**。
  - 路径从 **Start** 点出发，经过多次迭代到达 **Merged point**，最终收敛于蓝色的 **Optimal region**。
  - 图例明确区分了 **Search walk**、**Physical-only candidate** 和 **Optimal region**。
- **右上子图：性能与时序优化 (Performance and timing)**：
  - 横轴为 **Performance alignment**，纵轴为 **Timing margin, WNS (ns)**。
  - 曲线显示随着性能对齐度的提升，时序裕量逐步改善，最终进入右上角的 **Optimal region**，表明设计在满足高性能的同时实现了时序收敛。
- **右中子图：性能与面积优化 (Performance and area)**：
  - 横轴为 **Performance alignment**，纵轴为 **Area efficiency**。
  - 曲线表明在提升性能对齐度的过程中，**Area efficiency** 呈现稳步上升趋势，最终达到最优区域，实现了性能与面积的高效平衡。
- **右下子图：RTL 代码覆盖率演进 (RTL code coverage progression)**：
  - 横轴为 **Coverage-directed iteration**，纵轴为 **Coverage (%)**。
  - 展示了从 **RTL Start** 到 **Signoff** 的覆盖率自动化提升过程，包含 **Intermediate iteration** 和 **Recruited checkpoint**。
  - 具体数据点如下表所示：

| 迭代阶段 (Iteration Stage) | 覆盖率 (Coverage %) |
| :--- | :--- |
| RTL Start | 0.00% |
| IR #114 | 67.60% |
| IR #164 | 92.08% |
| IR #205 | 93.55% |
| Signoff | 95.00% |

- **核心技术与业务意义**：
  - 证明了 AI 系统能够自主遍历 **performance-area-timing** 搜索空间，生成具有不同控制路径、数据路径和状态机的 **RTL candidates**。
  - 在探索过程中，系统突破了人类设计师的认知局限，同时严格维持了 **verification rigor**。
  - 最终所有模块均达到 **95% code and functional coverage**，验证了 AI 驱动的微架构探索在 **PPA (Power, Performance, Area)** 优化上的巨大潜力与自动化闭环能力。

### Fig. 15: Requirements for recursive self-improvement.

![9b81894c049c6c387f721c5496734dad30dcea7a59e13acc9a1e5b9841d9ec4b.jpg](images/9b81894c049c6c387f721c5496734dad30dcea7a59e13acc9a1e5b9841d9ec4b.jpg)

图15直观地展示了实现**递归自我改进（recursive self-improvement）** 的核心机制与当前面临的**部署错位（Deployment Mismatch）** 挑战。

- **AI system capable of designing frontier HW**：代表能够自主设计前沿硬件（frontier HW）的AI系统能力边界，在图中表现为外层虚线圆环。
- **AI models that can deploy on designed frontier HW**：代表能够实际部署在该前沿硬件上的AI模型能力边界，在图中表现为内层实线圆环。
- **Deployment Mismatch**：红色高亮缺口区域，表示上述两种能力之间的**差距（gap）**。当前AI系统设计的硬件复杂度超出了当前可部署AI模型的实际运行能力。
- **循环箭头**：黑色双向箭头表示理想的**递归闭环**。当错位被消除时，AI模型在硬件上运行并优化下一代硬件设计，形成持续的正向反馈循环。

| 视觉元素 | 对应概念 | 论文上下文含义 |
| --- | --- | --- |
| 外层虚线圆 | AI system capable of designing frontier HW | AI系统自主设计复杂硬件的能力上限 |
| 内层实线圆 | AI models that can deploy on designed frontier HW | 当前AI模型在目标硬件上的实际部署与运行能力 |
| 红色缺口 | Deployment Mismatch | 硬件设计能力与模型部署能力之间的差距 |
| 黑色循环箭头 | Recursive self-improvement loop | 模型优化硬件、硬件加速模型的闭环迭代 |

- **核心挑战**：当前存在显著的**Deployment Mismatch**，即AI系统自主设计硬件的能力与AI模型实际部署在该硬件上的能力不匹配。
- **演进目标**：随着AI系统向更复杂的硬件设计扩展，必须缩小这一差距以闭合循环。
- **最终愿景**：消除错位后，**recursive self-improvement** 将成为推动现代硬件与AI协同进化的最大驱动力，实现智能的指数级增长。

