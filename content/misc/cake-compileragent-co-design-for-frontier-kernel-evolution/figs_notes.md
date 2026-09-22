# CAKE: Compiler–Agent Co-Design for Frontier Kernel Evolution 图表详解

### Figure 1:Overview of Cake. Kernel evolution consumes structured compiler evidence; compiler evolution is the outer loop.

![x1.png](images/x1.png)

- **图片整体概述**：该图展示了 **Cake** 系统的核心架构理念，直观对比了传统的 **Kernel-only evolution**（仅内核进化）与 Cake 提出的 **Kernel and compiler-harness co-evolution**（内核与编译器工具链协同进化）机制。

- **左侧：Existing Approach (Kernel-only evolution)**
  - **Optimization agent**：负责生成和修改内核代码。
  - **Kernel candidates**：智能体生成的候选内核代码。
  - **Agent harness**：作为固定黑盒，接收候选代码并执行。
  - **Environment benchmark**：提供最终的基准测试反馈（**benchmark feedback**），形成单一的内核优化闭环。
  - **核心局限**：编译器环境被视为固定黑盒，仅返回最终结果，无法提供细粒度的诊断信息，也无法自我进化以弥补能力缺口。

- **右侧：Cake (Kernel and compiler-harness co-evolution)**
  - **Kernel Evolution（蓝色内环）**：
    - **Optimization agent** 执行搜索与修复（**search and repair**）。
    - 生成 **Kernel candidates** 并输入到 **Evaluation harness**。
    - **Evaluation harness** 包含静态验证（**Static validation**）、性能模型（**Performance model**）和代码生成与GPU运行时（**Codegen + GPU runtime**）。
    - 提供 **structured feedback**（结构化反馈）给智能体，指导其进行精准修复。
  - **Compiler-Harness Evolution（红色外环）**：
    - 当 **Evaluation harness** 遇到能力缺口（**capability gap**）时，触发 **Compiler evolution** 来弥补缺口。
    - 进行 **Harness update**（涵盖 **IR**、**analysis**、**lowering** 的更新）。
    - 更新后的工具链重新赋能 **Evaluation harness**，实现编译器环境的自我进化。
  - **底层基准环境**：**PyTorch reference + target GPU** 提供正确性验证与延迟测量（**correctness + measured latency**）。

- **核心机制对比**

| 维度 | Existing Approach | Cake |
| :--- | :--- | :--- |
| **进化对象** | 仅内核代码 (**Kernel-only**) | 内核与编译器工具链协同 (**Co-evolution**) |
| **反馈机制** | 单一基准测试反馈 (**Benchmark feedback**) | 结构化反馈 (**Structured feedback**) + 能力缺口识别 (**Capability gap**) |
| **工具链角色** | 固定黑盒 (**Fixed black box**) | 动态进化目标 (**Evolution target**) |
| **核心组件** | Agent harness | Evaluation harness (**Static validation**, **Performance model**, **Codegen**) |

- **总结与意义**：该图揭示了 **Cake** 的核心创新点，即将编译器工具链从“固定黑盒”转变为“协同进化者”。通过引入 **Compiler-Harness Evolution** 外环，系统能够将重复出现的失败转化为新的验证规则、**IR** 原语或成本模型校准，从而从根本上提升 **Optimization agent** 探索前沿内核（**Frontier kernels**）的能力与效率。

### Figure 4:Evidence-driven compiler evolution. Corpus and runtime evidence drive validated compiler changes.

![x2.png](images/x2.png)

该图展示了 **Cake** 系统中**证据驱动的编译器演进 (Evidence-driven compiler evolution)** 机制，揭示了**语料库 (Corpus)** 与**运行时证据 (Runtime evidence)** 如何共同驱动经过验证的编译器变更。整个流程分为输入源与双轨演进循环。

输入源分类与特征：
| 输入类别 | 具体来源 | 作用与特征 |
| --- | --- | --- |
| **Knowledge Context** | **Production Kernels** | 提供真实生产环境中的内核代码作为基础语料。 |
| | **Documentation** | 提供硬件与架构的官方规范与设计约束。 |
| **Dynamic feedbacks** | **Compute Sanitizer** | 捕获内核演进过程中的内存、同步等底层运行时错误。 |
| | **Program failure cases** | 记录程序失败案例，为编译器分析提供反例与边界条件。 |

双轨演进循环机制：
- **日常 IR 演进 (DAILY IR EVOLUTION)**
  - **IR update proposal**：基于生产内核与文档，提出中间表示 (**IR**) 的更新提案。
  - **IR sanity check**：结合 **IR Design Principles** 对提案进行健全性与合规性检查，确保不违背核心设计原则。
  - **IR Implementation**：通过检查后，执行具体的 **IR** 实现与代码生成。
- **证据驱动的编译器工具链演进 (EVIDENCE-DRIVEN COMPILER HARNESS EVOLUTION)**
  - **Compiler analysis and pass creation**：接收来自 **IR Implementation** 的变更，并结合 **Dynamic feedbacks**（如 **Compute Sanitizer** 日志与失败案例），创建新的编译器分析逻辑或编译 **Pass**。
  - **Updated compiler harness**：将新创建的分析与 **Pass** 集成，输出更新后的编译器工具链 (**compiler harness**)，从而形成闭环，提升后续内核演进的验证能力。

机制核心价值：
- **双向驱动**：将静态的**知识上下文**与动态的**运行时反馈**深度融合，避免编译器设计与实际硬件执行脱节。
- **闭环验证**：通过 **IR sanity check** 与 **compiler harness** 的持续更新，确保每次 **IR** 变更都能被更强大的静态分析与验证工具所覆盖。
- **自动化演进**：使编译器工具链本身成为**内核演进 (kernel evolution)** 的受益者与迭代目标，而非固定不变的黑盒。

### Figure 5:Flash-KMeans fixed-shape clean-start attainment on B200. At each 5-million-token budget from 10M through 80M, every run contributes its best validated speedup so far. Curves show three-run means, bands show run minimum–maximum, and the horizontal line is the tuned FlashML K-means baseline.

![x3.png](images/x3.png)

- **图表概述**：该图展示了在 **B200** GPU 上，**Flash-KMeans** 固定形状 clean-start 任务中，不同 token 预算下的性能演进轨迹。
- **坐标轴与图例**：
  - **X轴**：**Cumulative tokens (millions)**，范围从 10M 到 80M，步长为 5M。
  - **Y轴**：**Performance / baseline (x)**，表示相对于基线的性能加速比。
  - **图例**：蓝色曲线代表 **CAKE IR mean**（含 min-max 阴影带），橙色曲线代表 **CUDA/PTX mean**（含 min-max 阴影带），黑色虚线代表 **tuned FlashML Kmeans Baseline**（y=1.0）。
- **趋势分析**：
  - **CAKE IR**：性能随 token 消耗稳步提升，在约 55M tokens 时突破基线（1.0x），最终在 80M 时达到约 **1.14x**。阴影带较窄，表明三次运行结果**稳定性高**。
  - **CUDA/PTX**：性能提升缓慢，在 80M token 预算耗尽时**仍未达到基线**（约 0.93x）。阴影带较宽，表明运行结果**波动较大**。
- **关键数据对比**：

| Token 预算 (Millions) | CAKE IR 性能倍数 (x) | CUDA/PTX 性能倍数 (x) | 基线达标状态 |
| :--- | :--- | :--- | :--- |
| 10 | ~0.35 | ~0.15 | 均未达标 |
| 30 | ~0.75 | ~0.35 | 均未达标 |
| 55 | ~1.05 | ~0.65 | 仅 **CAKE IR** 达标 |
| 80 | ~1.14 | ~0.93 | 仅 **CAKE IR** 达标 |

- **核心结论**：在相同的 80M token 预算下，**CAKE IR** 能够引导 Agent 发现超越专家调优基线的物理调度，而直接编写 **CUDA/PTX** 则无法在预算内达到基线水平，证明了 **Cake** 编译器协同设计在 Kernel 进化中的显著优势。

### Figure 6:KDA prefill evolution on B200. Orange is fixed $H{=}96$, $S{=}8192$ bring-up; blue is six-shape geometric-mean speedup over official FlashKDA. All points pass correctness.

![x4.png](images/x4.png)

- **图表核心主题**：展示 **KDA prefill** 在 **B200** 硬件上的性能演化轨迹，横轴为 **Cumulative tokens (input + output, millions)**，纵轴为相对基线的 **Performance / baseline (x)**。
- **基线与验证**：黑色虚线代表 **official FlashKDA** 基线（值为 **1.00**）。所有演化节点均通过正确性验证（**All points pass correctness**）。
- **橙色曲线 (H96 fixed trend)**：对应固定形状 **$H=96, S=8192$** 的 bring-up 阶段。在约 **100M tokens** 处迅速突破基线，随后在 **150M 至 550M tokens** 区间内稳定维持在 **1.10x** 左右的加速比。
- **蓝色曲线 (six-shape trend)**：对应六个形状的几何平均加速比。从约 **600M tokens** 接续演化，初始值约 **1.05x**，随后呈阶梯状攀升，在 **1000M tokens** 处达到峰值 **2.05x**，最终收敛于 **2.00x** 左右。
- **图例与视觉元素**：实线代表 **EMA trend**（指数移动平均趋势），散点代表 **raw samples**（原始采样数据），直观反映了搜索过程中的性能波动与收敛状态。

| 演化阶段 (Trend) | 颜色标识 | 关键起始点 (M tokens) | 突破基线 (1.00x) | 峰值性能 (x) | 结束点 (M tokens) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **H96 fixed trend** | 橙色 | ~0 | ~100 | ~1.10 | ~550 |
| **six-shape trend** | 蓝色 | ~600 | ~600 (初始>1.0) | ~2.05 | ~1150 |

### Figure 7:TinyGEMM evolution on B200. Orange tracks $N{=}8$, $M{=}2048$, $K{=}2048$; blue is the geometric mean over four recurring shapes, including orange. Dots are valid checkpoints, staircases are best-so-far, and the dotted line begins the follow-up.

![x5.png](images/x5.png)

- **图表概述**：该图展示了 **TinyGEMM** 在 **B200** GPU 上的演化轨迹，以 **FlashInfer** 为性能基线。X轴表示累积消耗的 token 数量（百万），Y轴表示相对性能倍数。
- **数据曲线解析**：
  - **橙色曲线**：追踪特定小尺寸 shape（**$N=8, M=2048, K=2048$**）的性能演化。
  - **蓝色曲线**：追踪包含上述 shape 在内的四个 recurring shapes 的几何平均性能（**geometric mean**）。
  - **散点与阶梯线**：散点代表有效的验证检查点（**valid checkpoints**），阶梯线代表历史最佳性能（**best-so-far**）。
- **演化阶段与关键指标**：
  - 在约 60M tokens 处，系统启动了针对小尺寸 shape 的后续优化（**small-shape follow-up begins**）。
  - 优化过程分为两个主要阶段，具体性能跃升数据如下表所示：

| 演化阶段 | 优化目标 (Target) | 达成指标 (Metric) | 达成时 Token 消耗 (约) |
| :--- | :--- | :--- | :--- |
| **第一阶段** (初始优化) | 单 shape 目标 **0.940x** | 四 shape 几何平均 **1.274x** | ~55M |
| **第二阶段** (后续跟进) | 单 shape 目标 **1.020x** | 四 shape 几何平均 **1.334x** | ~95M |

- **核心结论**：通过 **Cake IR** 的演化机制，**TinyGEMM** 不仅在特定小尺寸 shape 上突破了 **FlashInfer** 基线（达到 **1.020x**），还在更广泛的 shape 组合上实现了显著的性能提升（几何平均达 **1.334x**），验证了编译器协同设计在复杂 kernel 优化中的有效性。

### Figure 8:Alpha-MoE W8A8 Hopper-to-Blackwell rewrite on B200. Gray shows per-shape CUPTI medians; orange is the five-shape geometric mean (GM); blue is the best GM; shading is pre-checkpoint bring-up.

![x6.png](images/x6.png)

* **图表基本信息**
  * **横轴**：**Cumulative tokens (input + output, millions)**，表示累计消耗的Token数量（百万级）。
  * **纵轴**：**Performance (initial CAKE = 1)**，表示相对于初始CAKE版本的性能倍数。

* **数据系列与图例解析**
  * **灰色散点**：**40 per-shape measurements**，代表40个不同shape的独立CUPTI中位数测量值。
  * **橙色散点**：**five-shape GM checkpoint**，代表5个shape的几何平均值（Geometric Mean）检查点。
  * **蓝色实线**：**best GM so far**，记录演进过程中的历史最佳几何平均值。
  * **灰色阴影**：**pre-checkpoint bring-up**，表示检查点前的模型启动与探索阶段。

* **关键演进节点与性能突破**
  * **初始基线**：在约44M tokens处确立初始基线，标记为 **initial CAKE = 1.000x**。
  * **初步优化**：在约50M tokens处引入 **TMA gather4**，性能维持在1.0x左右。
  * **性能回退**：在约58M tokens处尝试 **persistent schedule**，虽然结果正确（correct），但导致性能下降（slower），跌至约0.9x。
  * **性能跃升**：在约60M tokens处应用 **warp-broadcast metadata + scales** 策略，性能实现大幅跃升，突破1.1x。
  * **最终收敛**：在约70M tokens处达到最终最佳性能，标记为 **1.137x vs. initial CAKE**。

* **演进节点数据汇总**

| 演进阶段 (Tokens) | 关键策略/事件 | 性能表现 (Relative to Initial) | 状态评估 |
| :--- | :--- | :--- | :--- |
| ~44M | 初始基线确立 | 1.000x | Baseline |
| ~50M | TMA gather4 | ~1.000x | 平稳 |
| ~58M | persistent schedule | ~0.900x | 正确但变慢 (correct, slower) |
| ~60M | warp-broadcast metadata + scales | ~1.130x | 显著跃升 |
| ~70M | 最终收敛 | 1.137x | 最佳几何平均值 (best GM) |

* **核心结论**
  * 图表展示了 **Alpha-MoE W8A8 Hopper-to-Blackwell rewrite** 在B200上的Agent演进轨迹。
  * 演进过程并非单调递增，存在如 **persistent schedule** 导致的性能回退，但Agent能够通过持续探索找到 **warp-broadcast** 等更优策略。
  * 最终实现了 **1.137x** 的五shape几何平均性能提升，验证了Cake IR在跨架构重写中的有效性。

