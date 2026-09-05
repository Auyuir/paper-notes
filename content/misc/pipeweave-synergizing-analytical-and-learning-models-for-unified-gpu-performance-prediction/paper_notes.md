# PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Kaixuan Zhang, Yunfan Cui, Shuhao Zhang, et al.

**发表期刊/会议 (Journal/Conference)**: unknown

**发表年份 (Publication Year)**: 2026

**研究机构 (Affiliations)**: Shanghai Jiao Tong University, Alibaba Group

---

## 1. 摘要

**目的**
- 解决现有数据驱动 GPU 性能模型在跨硬件泛化能力差及对复杂生产级 kernel（如 FlashAttention）建模不足的问题。
- 提供快速、准确且具有广泛泛化能力的 GPU 性能预测模型，以支持下一代硬件选型与系统级探索。

---

**方法**
- 提出 PIPEWEAVE 框架，采用解析模型与机器学习（ML）相结合的混合设计。
- **Kernel Decomposer**：将给定 kernel 分解为基础任务集，支持开源代码提取与闭源库（如 cuBLAS）逆向工程推断。
- **Scheduling Simulator**：模拟硬件（Round-Robin）与软件（Persistent kernel）调度机制，将任务映射至 SM，生成真实的任务分布。
- **Feature Analyzer**：基于 Roofline 模型扩展，将任务分布转化为多级特征集，量化对异构指令流水线（Tensor, FMA, XU, MIO）的需求与理论周期。
- **Performance Estimator**：利用轻量级 MLP 捕获复杂的跨流水线交互与资源依赖，预测最终执行时长。

![](images/fd259f026f4195e249e9e3bad09565565c6b0c207df0cc2411f38a6d8b7cea59.jpg) *Fig. 2. Overview of the PIPEWEAVE modeling framework, detailing the flow from kernel decomposition to the final performance prediction.*

---

**结果**
- 在 11 种 GPU（涵盖 4 代主流架构）和 5 类关键 kernel 上进行评估。
- **Kernel 级别预测精度**：

| 硬件类型 | PIPEWEAVE MAPE | SOTA (Neusight) MAPE | 误差降低倍数 |
| :--- | :--- | :--- | :--- |
| 已见 GPU | **6.1%** | 43.49% | **6.7x** |
| 未见 GPU | **11.4%** | 46.70% | **3.8x** |

- **E2E 推理预测精度**：

| 硬件类型 | PIPEWEAVE MAPE | SOTA (Neusight) MAPE | 误差降低倍数 |
| :--- | :--- | :--- | :--- |
| 已见 GPU | **8.5%** | - | **4.4x** |
| 未见 GPU | **10.7%** | - | **3.1x** |

- **超越仿真的优化指导**：利用 Quantile Regression 预测性能上限，诊断生产级 Fused MoE Triton kernel 的实现缺陷，指导参数调优后实现最高 **1.7x** 加速。

---

**结论**
- PIPEWEAVE 通过将知识驱动的流水线级解析分解与数据驱动的 MLP 相结合，成功弥合了高保真与泛化能力之间的鸿沟。
- 在多种 LLM 推理框架（vLLM, SGLang）和硬件代际上实现了 SOTA 预测精度。
- 验证了模型不仅可用于仿真预测，还能作为诊断工具识别硬件特定的实现低效并指导生产级 kernel 的针对性优化。

---

## 2. 背景知识与核心贡献

**研究背景**

- Transformer-based LLMs（如Gemini, Llama, Qwen）的快速演进大幅增加了对高性能GPU的需求。
- 硬件厂商（NVIDIA, AMD）频繁发布新架构，导致硬件配置组合爆炸，系统设计者难以穷举测试或获取下一代未发布硬件。
- 亟需快速、准确且具备广泛泛化性的GPU性能模型，以支持系统级探索和硬件选型。

---

**研究动机**

现有GPU性能建模三大范式均存在明显权衡，无法完全满足需求：

- Cycle-accurate simulators：保真度最高，但仿真速度极慢且缺乏跨架构可移植性。
- Analytical models：预测速度快，但精度受限且依赖特定硬件微基准测试，难以泛化。
- Data-driven approaches：通过学习tile级延迟实现高速预测，但高层假设（将tile视为原子、假设SM行为均匀、忽略fused-kernel耦合）导致跨硬件和负载泛化能力差。

现有SOTA数据驱动方法（如Neusight）存在微架构保真度不足的核心问题：

- 粒度不匹配：将异构pipeline活动折叠为单一聚合负载，将SM视为黑盒，未反映实际执行行为。
- 无法建模Fused Kernels：基于独立算子分解，无法捕捉FlashAttention等融合算子中算子紧耦合的执行模式与数据复用。
- 静态Wave建模：假设wave内tile执行延迟均匀，无法捕捉causal attention等动态负载导致的跨SM负载不均与尾效应。

---

**核心贡献**

- 统一建模框架：提出PIPEWEAVE，将kernel分解为基本任务并模拟其在SM上的调度，通过解析模型量化异构指令pipeline需求，再输入轻量级MLP捕获复杂非线性和资源竞争，实现高保真预测。

![](images/fd259f026f4195e249e9e3bad09565565c6b0c207df0cc2411f38a6d8b7cea59.jpg) *Fig. 2. Overview of the PIPEWEAVE modeling framework, detailing the flow from kernel decomposition to the final performance prediction.*

- 卓越的泛化能力：在4代11种GPU上验证，在未见架构上实现SOTA精度，将预测误差较SOTA方法降低最高6.7倍。

| 硬件类型 | Roofline | Linear | Habitat | Neusight | PIPEWEAVE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Seen | 72.22% | 59.50% | 28.92% | 43.49% | **6.77%** |
| Unseen | 79.61% | 70.28% | 85.96% | 46.70% | **13.14%** |

- 优化指导：超越单纯仿真，利用Quantile Regression建立性能天花板，诊断生产级Fused MoE Triton kernel的硬件特定实现低效，指导针对性优化，实现最高1.7倍加速。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体技术架构**

PIPEWEAVE采用**知识驱动与数据驱动相结合**的混合设计，通过将底层的微架构分析特征输入到机器学习模型中，实现高保真、高泛化性的GPU性能预测。其整体架构由四个核心模块构成：

- **Kernel Decomposer (内核分解器)**
  - 将内核的执行流分解为一组基本任务，每个任务代表Streaming Multiprocessor (SM)的一个可调度工作单元。
  - 支持传统硬件调度模型和Persistent Kernels的软件调度模型。
  - 通过映射函数 $\mathcal{F}$ 结合输入参数和硬件规格生成任务集 $\mathcal{T}$。

- **Scheduling Simulator (调度模拟器)**
  - 模拟任务在GPU SMs上的分配过程，生成真实的任务分布。
  - 支持两种调度范式：
    - **Hardware-Implemented Scheduler**：基于GigaThread Engine的Round-Robin (RR)策略。
    - **Software-Implemented Scheduler**：针对Persistent Kernels的软件Tile调度逻辑。

- **Feature Analyzer (特征分析器)**
  - 将任务分布转化为多级特征集，量化对异构指令流水线的需求。
  - 基于多维Roofline模型，计算各关键流水线的Demand和Theoretical Cycles。
  - 特征生成分为三个层级：
    - **Task Level**：计算单任务的Math pipelines (Tensor, FMA, XU)和MIO pipelines需求。
    - **SM Level**：聚合任务特征，识别负载最重的SM特征。
    - **GPU Level**：聚合全GPU特征，生成最终输入向量。

- **Performance Estimator (性能估计器)**
  - 使用轻量级MLP模型，输入为Feature Analyzer生成的特征向量。
  - 捕获复杂的非线性交互和资源竞争，预测内核的执行效率。
  - 最终延迟预测通过理论执行时间除以估计效率得出。

![](images/fd259f026f4195e249e9e3bad09565565c6b0c207df0cc2411f38a6d8b7cea59.jpg) *Fig. 2. Overview of the PIPEWEAVE modeling framework, detailing the flow from kernel decomposition to the final performance prediction.*

### 1. Pipeline-level Analytical Feature Extraction

**核心观点**

PIPEWEAVE 的特征分析模块通过将经典 Roofline 模型扩展为多维度的流水线级分析，显式量化任务对 GPU 异构指令流水线的需求与理论周期。该模块采用自底向上的聚合策略，将孤立的 Task 级特征转化为紧凑的多级特征集，为后续机器学习模型提供高保真、可解释的输入，同时避免了复杂的指令级并发建模，保障了跨架构的泛化能力。

---

**实现原理与抽象策略**

- **多维 Roofline 扩展**：突破传统 Roofline 模型仅关注计算与内存两个维度的局限，为 SM 内的每一个关键指令流水线（如 Tensor, FMA, XU, MIO）分别计算独立的理论性能上限。
- **双维度特征刻画**：沿两个基础维度表征 Kernel 执行：
  - **Demand**：测量施加在每个流水线上的总工作负载（如操作数或字节数）。
  - **Theoretical Cycles**：由 Demand 推导，表示如果仅该流水线成为瓶颈时的理想执行周期。
- **抽象与解耦策略**：不构建复杂的指令级并发或架构特定机制（如 Hopper 的 TMA）的刚性分析模型。将多样的内存访问机制统一为广义的内存流水线需求，将这些基本需求作为原始特征暴露给 MLP，由 MLP 自动学习复杂的非线性交互与资源争用。

![](images/b7f5bf4866d00aa73054befc1434b7aa73a344f932133a81546872ff15ca8bb5.jpg)
![](images/9c095713a8f370dffda01b2c98bab04b912c2c90dbec44abd3e9a97d28c0b3c8.jpg)

---

**算法流程：自底向上三级聚合**

特征生成遵循自底向上的过程，跨越 Task、SM、GPU 三个层级：

- **Task Level（任务级）**
  - **Math Pipelines**：针对每个 Task $\tau_i$，通过维度参数向量 $\mathbf{d}_i$ 计算各流水线的操作数。
    - **Tensor pipeline**：计算 MMA 操作数。公式为 $N_{\mathrm{ops, Tensor}} = \alpha \cdot tile\_M \cdot tile\_N \cdot tile\_K$。标准 GEMM 中 $\alpha = 2$，FlashAttention 执行两次矩阵乘法，$\alpha = 4$。
    - **FMA/XU pipelines**：通过分析 Kernel 的算术表达式和循环迭代空间，推导元素级（EW）操作的总数。
    - **理论周期计算**：将操作数除以硬件规格中的吞吐量 $Th_p$，得到 $C_p = N_{\mathrm{ops, p}} / Th_p$。
  - **MIO Pipelines**：计算每个 Task 从内存层次结构加载的总字节数 $B_i$，重点关注 Load 操作因其常位于关键执行路径上。
- **SM Level（SM级）**
  - 结合 Scheduling Simulator 生成的任务分布 $\{ \mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_{N_{SM}} \}$，将 Task 级特征聚合到各个 SM 上。
  - 计算 per-SM 的总操作数 $N_{\mathrm{ops, p}}^{\mathrm{SM_j}}$ 和总内存需求 $B^{\mathrm{SM_j}}$。
  - 识别负载最重、利用率最高的 SM 的特征（如 Max SM Operations, Max SM Theoretical Cycles），用于捕捉跨 SM 的负载不均衡和尾效应。
- **GPU Level（GPU级）**
  - 将所有 SM 的特征再次聚合，生成全局特征。
  - 计算 GPU 级别的总操作数 $N_{\mathrm{ops, p}}^{\mathrm{GPU}}$ 和总内存需求 $B^{\mathrm{GPU}}$。
  - 结合全局带宽和 SM 数量推导 GPU 级理论周期，例如计算流水线的 GPU 级周期为 $C_p^{\mathrm{GPU}} = N_{\mathrm{ops, p}}^{\mathrm{GPU}} / (N_{SM} \cdot Th_p)$。

---

**参数设置与特征向量表**

特征分析模块依赖目标 GPU 的硬件规格参数 $\mathbf{S}$ 进行计算，关键参数包括：

| Parameter | Value Range | Unit |
| :--- | :--- | :--- |
| Tensor Pipe Throughput | 512 - 4096 | ops/cycle/SM |
| FMA Pipe Throughput | 64 - 128 | ops/cycle/SM |
| XU Pipe Throughput | 16 | ops/cycle/SM |
| Global Memory Bandwidth | 696-4916 | GB/s |
| L2 Cache Bandwidth | 2430 - 10400 | GB/s |
| Shared Memory Bandwidth per SM | 128 | Byte/cycle |

最终生成的紧凑多级特征向量结构如下表所示，该向量直接作为后续 MLP 的输入：

| Pipeline | Granularity | Features |
| :--- | :--- | :--- |
| **Math** | GPU | Total Operations, Total Theoretical Cycles |
| **Math** | SM | Max SM Operations, Max SM Theoretical Cycles |
| **MIO** | GPU | Total Memory Demand, Theoretical Cycles (Global, L2) |
| **MIO** | SM | Max SM Memory Demand, Theoretical Cycles (Global, L2, Shared) |

---

**输入输出关系与整体作用**

- **输入**：
  - Task 分布集合 $\{ \mathcal{T}_1, \mathcal{T}_2, \dots, \mathcal{T}_{N_{SM}} \}$（来自 Scheduling Simulator）。
  - 硬件架构规格参数 $\mathbf{S}$（如各流水线吞吐量、各级内存带宽）。
- **输出**：
  - 包含 Math 与 MIO 流水线、跨越 Task/SM/GPU 三个层级的 Demand 与 Theoretical Cycles 的紧凑特征向量。
- **在整体中的作用**：
  - **桥梁作用**：作为 PIPEWEAVE 框架的第三模块，连接了知识驱动的确定性分析与数据驱动的非线性预测。将高阶的、复杂的 Kernel 执行行为转化为 MLP 可理解的数值特征。
  - **泛化性基础**：通过将硬件参数抽象为紧凑向量，使得一旦 MLP 在多种硬件配置下训练完成，面对新输入或未见的 GPU 架构时，仅需运行快速分析步骤生成特征向量即可进行实时预测，无需重新训练复杂的微观架构模型。
  - **优化诊断基准**：在 "Beyond Simulation" 阶段，该特征向量对应的执行效率被用作 Quantile Regression 的基础，通过预测 P80 性能天花板，诊断生产环境 Kernel 的实现缺陷。

### 2. Dynamic Task Scheduling Simulation

**核心机制概述**

- **Scheduling Simulator** 是 PIPEWEAVE 框架的第二个核心模块。
- 负责将 **Kernel Decomposer** 分解出的抽象任务集映射到 GPU 的物理 **Streaming Multiprocessors (SMs)** 上。
- 核心目标是生成精确的 **Task Distribution**，从而刻画每个 SM 的实际工作负载，识别因 **Workload Imbalance** 导致的性能瓶颈。

![](images/fd259f026f4195e249e9e3bad09565565c6b0c207df0cc2411f38a6d8b7cea59.jpg) *Fig. 2. Overview of the PIPEWEAVE modeling framework, detailing the flow from kernel decomposition to the final performance prediction.*

---

**调度范式与算法流程**

- 支持现代 GPU 应用的两种主要调度范式：

| 调度范式 | 适用场景 | 调度主体 | 核心机制 |
| :--- | :--- | :--- | :--- |
| **Hardware-Implemented Scheduler** | 传统 Kernel | GPU 硬件调度器 | **GigaThread Engine**，基于 **Round-Robin (RR)** 策略 |
| **Software-Implemented Scheduler** | **Persistent Kernels** | 软件组件 | 常驻 **CTA** 从全局队列获取细粒度工作单元，由 **Tile Scheduler** 管理 |

- **Hardware-Implemented Scheduler 算法流程**
  - 第一轮：为每个 SM 分配至少一个任务（即 **CTA**）。
  - 后续轮次：若 SM 仍有足够资源（如寄存器、Shared Memory、warp-slots），则继续分配任务。
  - 持续分配直到所有 SM 因资源限制或硬件限制而饱和。
  - 动态补充：当某个 SM 上的任务完成并退出时，分配新任务给该 SM。

- **Software-Implemented Scheduler 算法流程**
  - 硬件调度器作用弱化，**CTA** 启动后常驻 SM 作为 persistent worker。
  - 常驻 **CTA** 从全局工作队列中重复获取细粒度工作单元（如 GEMM 中的 tiles）。
  - 任务分配由特定的软件逻辑管理，例如 Hopper 架构上的 **FlashInfer FA3** 使用 **MinHeap-based scheduler**。
  - 实现成本：模拟器仅需约 40 行代码即可精确复制此调度逻辑。

---

**形式化定义与参数设置**

- 输入参数：
  - 总任务集 $\mathcal{T} = \{\tau_1, \tau_2, \ldots, \tau_t\}$。
  - 硬件架构规格 $\mathbf{S}$（包含 SM 数量等参数）。
- 映射函数：$\mathcal{M}(\mathcal{T}, \mathbf{S})$。
- 输出结果：跨 SM 的任务集合划分 $\{\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_{N_{SM}}\}$。
- 约束条件：集合构成 $\mathcal{T}$ 的严格划分，满足 $\bigcup_{j=1}^{N_{SM}} \mathcal{T}_j = \mathcal{T}$ 且 $\mathcal{T}_i \cap \mathcal{T}_j = \emptyset$。

---

**输入输出关系与整体作用**

- 在 PIPEWEAVE 流水线中处于承上启下的关键位置：
  - 承上：接收 **Kernel Decomposer** 输出的抽象任务集。
  - 启下：将具体的任务分布传递给 **Feature Analyzer**，用于计算 per-SM 和 GPU 级别的理论周期与需求特征。
- 整体作用：
  - 打破了以往方法（如 Neusight）依赖聚合指标和 **Static Wave Modeling** 的局限。
  - 能够精确捕捉 **Cross-SM Load Imbalance** 和 **Tail Effect**。
  - 解决了动态工作负载（如 **Causal Attention** 中处理不同长度 token 的任务，或带有 early-exit 条件的 Kernel）导致的 tile-to-tile latency variation 建模难题。

### 3. Quantile Regression for Performance Ceiling

**核心原理与算法流程**

- 传统性能预测模型通常拟合平均性能，无法回答“当前实现是否已达到硬件极限”这一关键问题。
- PIPEWEAVE 引入 **Quantile Regression** 机制，通过改变损失函数，将预测目标从“平均执行效率”转变为“潜在性能天花板”。
- 算法流程如下：
  - 复用 Section V-C 中定义的 MLP 架构与输入特征集（包含 MIO pipeline 与 Math pipeline 的需求及理论周期特征）。
  - 保持预测目标为 **execution efficiency**（理论执行时间与实际延迟的比值）。
  - 将训练损失函数从 MAPE 替换为 **Quantile Loss**。
  - 将分位数参数设定为 **80th percentile (P80)**。
  - 模型拟合数据集中表现最优的前 20% 配置点，自动过滤掉后 80% 的次优或异常配置。

---

**参数设置与输入输出关系**

- **输入特征**：与标准 PIPEWEAVE 模型完全一致，即由 Feature Analyzer 生成的多级分析特征向量，涵盖 GPU 级别与 SM 级别的 Tensor、FMA、XU 算力需求及 Global、L2、Shared Memory 带宽需求。
- **模型参数**：沿用轻量级 MLP（隐藏层为 256, 128, 64），激活函数为 ReLU，配合 Batch Normalization 与 Dropout。
- **输出结果**：模型输出 $\hat{y}_{p80}$，代表在给定输入特征下，该 kernel 在特定硬件上**统计学定义的潜在性能天花板**。
- **P80 参数选择依据**：
  - 相比 P90 或更高分位数，P80 对极端离群点和测量噪声具有更强的鲁棒性。
  - 它代表一个“高但现实可达”的性能目标，避免了过高估计导致的优化误导。

---

**在整体系统中的作用：性能诊断与优化指导**

- 该机制的核心价值在于“超越仿真”，将预测模型转化为优化工具。
- **性能差距计算**：通过公式 $\text{perf\_gap} = \hat{y}_{p80} - y_{\text{actual}}$ 量化当前实现与理论天花板之间的差距。
- **识别低效点**：定义 **Underperforming Points** 为 $\text{perf\_gap} > 0.1$ 的配置点。这直接定位了存在巨大优化潜力的硬件与输入组合。

![](images/77a2447b10020214b88fcf0ccc86662df320ce35de4a16d9e452dabb3362180a.jpg)

- **指导参数调优**：针对识别出的 Underperforming Points，系统进行暴力自动调优，搜索空间包括 `BLOCK_SIZE`, `num_stages`, `num_warps`。
- **验证与闭环**：调优后，性能差距显著缩小，证明了 P80 天花板设定的有效性。

| GPU 平台 | Underperforming Points 数量 | 调优后几何平均加速比 |
|---|---|---|
| A40 | 921 | 1.61× |
| L20 | 728 | 1.12× |
| A100 | 488 | 1.06× |
| H800 | 340 | 1.03× |

![](images/b293323980823ddfdcff3ce71e29d2d0e95d9b40b60761f852df965d21972af5.jpg)

- **深层限制揭示**：即使经过参数调优，部分平台仍残留性能差距。这表明瓶颈可能源于 kernel 的结构性设计或 Triton 编程模型的固有局限，需引导开发者进行更深层次的代码重构。


---

## 4. 实验方法与实验结果

**实验设置**

- **硬件环境**：涵盖 **4 个主要架构代际**（Ampere, Ada, Hopper, Blackwell）的 **11 种 NVIDIA GPU 类型**。其中 **6 种**用于训练，**5 种**用于未见测试，以验证跨架构泛化能力。
  
| GPU | 架构 | SMs | Mem BW (GB/s) | Tensor BF16 (ops/clk/SM) | Freq (MHz) |
|---|---|---|---|---|---|
| A100 | Ampere | 108 | 2039 | 2048 | 1410 |
| H20 | Hopper | 78 | 4023 | 1024 | 1830 |
| H800 | Hopper | 132 | 3352 | 4096 | 1830 |
| RTX PRO 6000 S | Blackwell | 188 | 1792 | 1024 | 2340 |

- **软件与框架**：统一使用 PyTorch 2.8.0, CUDA Toolkit 12.8, FlashInfer 0.4.1, SGLang 0.5.4, vLLM 0.11.0, Triton 3.4.0。
- **数据集与 Kernels**：包含约 **1M 样本**，覆盖 **6 类关键 Kernels**（GEMM, Scaled MM, Attention, RMSNorm, SiLU&Mul, Fused MoE），精度涵盖 **FP8, BF16/FP16, FP32**。
  - **GEMM**：613,263 样本，维度范围广泛。
  - **Attention**：104,958 样本，模拟真实可变长序列模式。
  - **E2E 推理**：使用 Qwen2.5-14B, Qwen3-32B, Llama3.1-70B，结合 Arxiv Summarization 与 Splitwise 数据集，测试 TP/PP 并行策略。
- **Baselines**：
  - **主要基线**：Roofline, Linear regression, Habitat, Neusight (SOTA)。
  - **次级基线**：AMALI, LLMCompass（仅限 GEMM 对比，因运行开销过大）。

![](images/fd259f026f4195e249e3bad09565565c6b0c207df0f0cc2411f38a6d8b7cea59.jpg)

---

**结果数据分析**

- **Kernel 级别预测精度**
  - **PIPEWEAVE** 在 Seen GPU 上平均 **MAPE** 为 **6.0%**，在 Unseen GPU 上为 **11.5%**。
  - 相较 SOTA 基线 Neusight，误差分别降低 **6.7×** 和 **3.9×**。

| 硬件类型 | Roofline | Linear | Habitat | Neusight | PIPEWEAVE |
|---|---|---|---|---|---|
| Seen | 72.22% | 59.50% | 28.92% | 43.49% | **6.77%** |
| Unseen | 79.61% | 70.28% | 85.96% | 46.70% | **13.14%** |

- **FP8 精度适应性**：在 Hopper 架构上测试 Scaled MM Kernel，Seen GPU (H20, H800) 的 MAPE 为 **1.9%** 和 **4.1%**，Unseen GPU (H100, H200) 为 **4.2%** 和 **5.2%**。
- **端到端 推理精度**
  - 单 GPU (TP=1) 场景：平均 MAPE 为 **11.3%**，较 Neusight 提升 **2.8×**。
  - 分布式多 GPU 场景：覆盖 TP=2/4/8 及 TP=4&PP=2，整体平均 MAPE 为 **6.6%**，较 Neusight 提升 **5.3×**。
- **模拟开销对比**

![](images/7652f8d100393c89b6efe09d3f526cce70aa109fe2d48b82672d87aeea946fc9.jpg)

  - 与 AMALI 和 LLMCompass 在 A100 上对比 GEMM 工作负载。
  - **PIPEWEAVE** MAPE 为 **6.4%**，优于 AMALI (28.3%) 和 LLMCompass (29.7%)。
  - 预测时间开销减少 **3 至 7 个数量级**。

---

**消融实验与组件验证**

- **分析组件准确性验证**
  - **Kernel Decomposer**：分解产生的 CTA 数量与 Ground-truth 完全一致，准确率 **100%**。
  - **Scheduling Simulator 与 Feature Analyzer**：通过 NCU 工具验证操作数计数。
    - 总操作数最大误差仅 **0.5%**。
    - 单 SM 最大操作数误差为 **6.3%**。
    - FA2 误差 (6.34%) 高于 FA3 (0.45%)，因 FA2 依赖动态硬件调度，而 FA3 使用确定性的 Persistent Kernel 设计。

- **核心模块贡献度分析**

![](images/e510c30ecee74c3142a467101f8beaf5c0bfeb8e8e40599c61c2c76364d3b18f.jpg)

  - 移除 MIO 特征、Math 特征 或 MLP 模块 均导致精度显著下降。
  - **Attention Kernel**：完整模型精度分别比 w/o MIO 提升 **1.1×**，比 w/o Math 提升 **1.8×**，比 w/o MLP 提升 **2.9×**。
  - **GEMM Kernel**：完整模型精度分别比 w/o MIO 提升 **3.2×**，比 w/o Math 提升 **2.7×**，比 w/o MLP 提升 **3.5×**。
  - Attention 的最终误差 (15.54%) 高于 GEMM (8.39%)，主因是 Causal Masking 与可变序列长度导致的负载不均及动态执行特性。

- **超越模拟的性能优化指导**
  - 采用 **Quantile Loss (P80)** 定义 Potential Performance Ceiling，识别 Underperforming Points (Gap > 0.1)。

![](images/77a2447b10020214b88fcf0ccc86662df320ce35de4a16d9e452dabb3362180a.jpg)

  - 识别出硬件特定的低效问题：A40 有 **921** 个低效点，H20 则为 **0**。

![](images/b293323980823ddfdcff3ce71e29d2d0e95d9b40660761f852df965d21972af5.jpg)

  - 针对低效点进行参数 Autotuning (BLOCK_SIZE, num_stages, num_warps)。

| GPU | Underperforming Points | Geo-mean Speedup |
|---|---|---|
| A40 | 921 | **1.61×** |
| L20 | 728 | **1.12×** |
| A100 | 488 | **1.06×** |
| H800 | 340 | **1.03×** |

  - 低效点密度与调优收益呈强正相关（Pearson 相关系数 **0.86**），验证了统计诊断方法的有效性。

---

