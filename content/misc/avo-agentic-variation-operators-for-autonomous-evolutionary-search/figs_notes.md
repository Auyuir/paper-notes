# AVO: Agentic Variation Operators for Autonomous Evolutionary Search 图表详解

### Figure 1:EVO vs AVO: Comparison between prior evolutionary search frameworks (e.g. FunSearch, AlphaEvolve, and related LLM-augmented evolutionary approaches) and the proposed Agentic Variation Operator. Left: Prior approaches follow a fixed pipeline where the LLM is confined to a single-turn generation step or a predefined workflow, with sampling and evaluation controlled by the framework. Right: AVO replaces this pipeline with an autonomous AI agent that iteratively plans, implements, tests, and debugs across long-running sessions, with direct access to previous solutions, evaluation utilities, tools, and persistent memory.

![x1.png](images/x1.png)

- **图片整体结构**：该图分为左右两个面板，直观对比了经典进化搜索框架（**EVO**）与提出的智能体变异算子（**AVO**）在架构和工作流上的根本差异。
- **左侧面板（EVO）**：展示了传统基于大语言模型（**LLM**）的进化搜索框架（如 **FunSearch**、**AlphaEvolve**）。其核心是一个**固定流水线（Fixed Pipeline）**，包含 **Previous Solutions**（历史解）、**Sampling**（采样）、**LLM**（单轮生成或预定义工作流）和 **Evaluation**（评估）四个单向或简单循环的模块。
- **右侧面板（AVO）**：展示了提出的 **Agentic Variation Operators**。其核心是一个**自主智能体循环（Agent Loop）**，包含 **Planning**（规划）、**Implementing**（实现）、**Testing**（测试）和 **Debugging**（调试）。**AI Agent** 处于中心位置，与 **Previous Solutions** 和 **Evaluation Utility** 进行双向交互，并底层依赖 **Tools**（工具）和 **Memory**（记忆）。

| 特征维度 | EVO (Classical Evolutionary Search) | AVO (Agentic Variation Operators) |
| :--- | :--- | :--- |
| **核心机制** | 固定流水线 (Fixed Pipeline) | 自主智能体循环 (Agent Loop) |
| **LLM 角色** | 单轮生成器 (Single-Turn Generator) | 自主变异算子 (Autonomous Variation Operator) |
| **工作流控制** | 框架控制采样与评估 (Framework-controlled) | 智能体自主规划、实现、测试与调试 |
| **上下文与记忆** | 仅依赖当前采样的父代解 | 拥有持久记忆 (Memory) 和工具 (Tools) 访问权 |
| **交互模式** | 单向或简单反馈循环 | 与历史解和评估工具的双向深度交互 |

- **核心结论**：图片强调了 **AVO** 将 **LLM** 从受限的候选生成器提升为具备长期自主探索能力的**变异算子**，打破了传统框架中由启发式规则主导的僵化流程，实现了跨长时间会话的迭代优化。

### Figure 2:Illustration of the Agentic Variation Operator (AVO).

![x2.png](images/x2.png)

- **Input to AVO** 模块为系统提供初始上下文与评估标准，包含三个核心子模块：
  - **Population $\mathcal{P}_t$**：存储历史候选解及其对应分数，如 $(x_1, f(x_1))$ 至 $(x_t, f(x_t))$。
  - **Knowledge Base $\mathcal{K}$**：提供领域专业知识，包括 **CUDA & PTX Documents** 和 **Reference GPU Kernel Codebases**。
  - **Scoring Function $f$**：定义评估指标，涵盖 **Correctness Check**（正确性检查）与 **Throughput Measurement**（吞吐量测量）。

- **AVO Main Agent Loop** 是系统的核心执行引擎，由具备 **Tools** 和 **Memory** 的 **AI Agent** 驱动，通过 **Reasoning** 形成闭环迭代：
  - **Planning**：基于 $\mathcal{K}$ 和 $\mathcal{P}_t$ 提出代码修改建议（**Propose Edits**）。
  - **Implementation**：执行具体的代码修改（**Apply Code Modifications**）。
  - **Evaluation**：使用评分函数 $f$ 评估当前代码性能（**Evaluate Performance**）。
  - **Bug-Fixing**：若评估未达标，则进行诊断、修复并调整计划（**Diagnose, Repair, and Adapt Plan**）。

- **AVO Supervisor Agent** 作为外部监控机制，负责 **Monitor Stagnation**（监控停滞状态），并在必要时触发 **Conditional Intervention**（条件干预）以引导主循环跳出局部最优。

- **Output of AVO** 模块输出经过验证和优化的新一代候选解及其分数，即 $(x_{t+1}, f(x_{t+1}))$。

| 模块区域 | 核心组件与功能 | 关键数据流向 |
| :--- | :--- | :--- |
| **Input to AVO** | **Population $\mathcal{P}_t$**, **Knowledge Base $\mathcal{K}$**, **Scoring Function $f$** | 提供历史解、领域文档、评估标准至主循环 |
| **AVO Main Agent Loop** | **AI Agent** (Planning, Implementation, Evaluation, Bug-Fixing) | 接收 $\mathcal{P}_t, \mathcal{K}, f$，执行自主迭代优化 |
| **AVO Supervisor Agent** | **Monitor Stagnation**, **Conditional Intervention** | 监控主循环状态，必要时介入干预 |
| **Output of AVO** | 新一代优化解生成 | 输出 $(x_{t+1}, f(x_{t+1}))$ 并更新种群 |

### Figure 3:Multi-head attention forward-pass prefilling throughput (TFLOPS) on NVIDIA B200 with head dimension 128, 16 heads, and BF16 precision. Batch size and sequence length are varied with a fixed total of 32k tokens.

![x3.png](images/x3.png)

- **图片概述**：该图展示了在 **NVIDIA B200** GPU 上，**Multi-head attention (MHA)** 前向传播预填充吞吐量（**TFLOPS**）的对比结果。实验配置为 **head dimension 128**、**16 heads**、**BF16 precision**，总 token 数固定为 **32k**。图表分为左侧 **causal=False**（Non-causal）和右侧 **causal=True**（Causal）两部分，对比了 **cuDNN**、**FA4** 和 **AVO** 三种实现。

- **吞吐量数据对比**：

| 序列长度 (Batch Size) | cuDNN (Non-causal) | FA4 (Non-causal) | AVO (Non-causal) | cuDNN (Causal) | FA4 (Causal) | AVO (Causal) |
|---|---|---|---|---|---|---|
| 4K (bs=8) | 1573 | 1578 | 1573 | 1344 | 1259 | 1392 |
| 8K (bs=4) | 1609 | 1614 | 1615 | 1477 | 1412 | 1482 |
| 16K (bs=2) | 1626 | 1637 | 1664 | 1551 | 1502 | 1582 |
| 32K (bs=1) | 1638 | 1651 | 1668 | 1590 | 1550 | 1637 |

- **关键分析**：
- **Non-causal 场景表现**：在 **causal=False** 场景下，**AVO** 在长序列（16K 和 32K）中展现出明显优势，分别达到 **1664 TFLOPS** 和 **1668 TFLOPS**，超越了 **cuDNN** 和 **FA4**。在短序列（4K）中，三者性能基本持平，处于测量误差范围内。
- **Causal 场景表现**：在 **causal=True** 场景下，**AVO** 在所有测试配置中均全面领先。特别是在 4K 序列下，**AVO** 达到 **1392 TFLOPS**，显著优于 **FA4** 的 **1259 TFLOPS** 和 **cuDNN** 的 **1344 TFLOPS**。
- **性能提升幅度**：随着序列长度增加，**AVO** 在 **Causal** 下的绝对吞吐量优势持续扩大，在 32K 时达到 **1637 TFLOPS**，比 **cuDNN** 高出 **47 TFLOPS**，比 **FA4** 高出 **87 TFLOPS**。
- **基线对比差异**：**FA4** 在 **Causal** 的短序列（4K）下性能出现明显下降（**1259 TFLOPS**），而 **AVO** 成功克服了这一瓶颈，证明了其在复杂掩码场景下的优化有效性。

### Figure 4:Grouped-query attention forward-pass prefilling throughput (TFLOPS) on NVIDIA B200 with 32 query heads, head dimension 128 and BF16 precision. Results are shown for two GQA configurations (group sizes 8 and 4) under both causal and non-causal masking. The GQA kernel was produced by prompting the AVO agent to adapt the evolved MHA kernel, requiring approximately 30 minutes of autonomous effort.

![x4.png](images/x4.png)

* **图片概述**：该图展示了 **Grouped-Query Attention (GQA)** 在 NVIDIA B200 GPU 上的前向传播预填充吞吐量（Forward TFLOPS）对比结果。实验验证了 **Agentic Variation Operators (AVO)** 从 Multi-Head Attention (MHA) 迁移到 GQA 的泛化能力与性能优势。
* **实验配置**：
  * **硬件与精度**：NVIDIA B200 GPU，BF16 精度，Head dimension 128，32 query heads。
  * **GQA 配置**：Group size 8（对应 Qwen3-30B-A3B）与 Group size 4（对应 Qwen3-8B）。
  * **掩码类型**：Causal（因果掩码）与 Non-causal（非因果掩码）。
  * **序列长度与 Batch Size**：总 token 数固定为 32768，序列长度分别为 4K (bs=8)、8K (bs=4)、16K (bs=2)、32K (bs=1)。
  * **对比基线**：NVIDIA 闭源 **cuDNN** (v9.19.1) 与开源 **FlashAttention-4 (FA4)**。

* **性能数据对比**：

| 配置 (Group, Masking) | 序列长度 (Batch Size) | cuDNN (TFLOPS) | FA4 (TFLOPS) | AVO (TFLOPS) |
| :--- | :--- | :--- | :--- | :--- |
| **Group=8, Non-causal** | 4K (bs=8) | 1590 | 1602 | **1646** |
| | 8K (bs=4) | 1620 | 1633 | **1677** |
| | 16K (bs=2) | 1607 | 1643 | **1663** |
| | 32K (bs=1) | 1441 | 1472 | **1528** |
| **Group=8, Causal** | 4K (bs=8) | 1377 | 1342 | **1467** |
| | 8K (bs=4) | 1503 | 1471 | **1511** |
| | 16K (bs=2) | 1571 | 1541 | **1618** |
| | 32K (bs=1) | 1603 | 1530 | **1625** |
| **Group=4, Non-causal** | 4K (bs=8) | 1586 | 1601 | **1633** |
| | 8K (bs=4) | 1617 | 1627 | **1679** |
| | 16K (bs=2) | 1626 | 1624 | **1679** |
| | 32K (bs=1) | 1477 | 1460 | **1526** |
| **Group=4, Causal** | 4K (bs=8) | 1368 | 1344 | **1464** |
| | 8K (bs=4) | 1497 | 1470 | **1506** |
| | 16K (bs=2) | 1568 | 1536 | **1603** |
| | 32K (bs=1) | 1601 | 1517 | **1647** |

* **核心发现与分析**：
  * **全面超越基线**：在所有测试的 GQA 配置（Group 8/4，Causal/Non-causal）和序列长度下，**AVO 均实现了最高的 TFLOPS**，全面击败 cuDNN 和 FA4。
  * **Causal 场景优势显著**：在 Causal 掩码下，尤其是短序列（4K, bs=8）场景中，AVO 的性能提升幅度最大。例如在 Group=8 时，AVO 比 cuDNN 提升约 **6.5%**，比 FA4 提升约 **9.3%**。
  * **高效的跨任务迁移**：GQA 内核并非从头训练，而是通过提示 AVO agent 对已演化的 MHA 内核进行自主适配。该过程**仅需约 30 分钟**，证明了 AVO 发现的底层微架构优化具有极强的**泛化能力**，能够无缝适应 GQA 不同的计算与内存访问模式。
  * **长序列性能稳健**：在 32K 长序列（bs=1）下，尽管绝对吞吐量因计算特性有所下降，但 AVO 依然保持领先，特别是在 Causal 配置下，相比 FA4 展现出显著的性能冗余优势。

### Figure 5:Evolution trajectory of AVO across 40 kernel versions over 7 days on causal MHA. The solid green line tracks the running-best geometric mean throughput across all configurations; green circles mark versions that set a new best. Dashed colored lines show per-configuration throughput (seq_len = 4k, 8k, 16k, 32k). Horizontal dashed lines indicate the geometric mean throughput of cuDNN and FA4.

![x5.png](images/x5.png)

- 本图展示了 **AVO** 在 7 天内针对 **Causal BF16 Multi-head Attention** 生成的 40 个 **Kernel Version** 的自主演进轨迹。
- **坐标轴与图例说明**：
  - **X轴**：代表 **Kernel Version**，从 v1 到 v40。
  - **Y轴**：代表吞吐量，单位为 **TFLOPS**，范围从 1100 到 1600。
  - **绿色实线与圆点**：分别表示 **Running best (Geomean)** 和 **New best (Geomean)**，即所有配置下的几何平均吞吐量及其最佳更新节点。
  - **彩色虚线**：代表不同序列长度（**seq_len=4k, 8k, 16k, 32k**）的独立吞吐量表现。
  - **水平虚线**：代表基线性能，**cuDNN** 为 1488 TFLOPS（蓝色），**FA4** 为 1426 TFLOPS（红色）。
- **基线对比与最终成果**：
  - **AVO v40** 最终达到 **1520 TFLOPS**，显著超越了 **cuDNN (1488 TFLOPS)** 和 **FA4 (1426 TFLOPS)** 的基线水平。
- **关键演进节点分析**：
  - 演进过程呈现明显的**阶梯式跃升**特征，而非平滑的线性增长。
  - 早期版本（v1-v7）处于探索期，性能在 1050 TFLOPS 左右徘徊。
  - v8 至 v18 期间实现快速突破，在 v18 首次超越 **FA4** 基线。
  - v29 至 v40 进入微调期，逐步突破 **cuDNN** 基线并持续刷新最佳记录。

| Kernel Version | 关键事件 / 性能表现 | 估算 TFLOPS (Geomean) | 对比基线状态 |
| :--- | :--- | :--- | :--- |
| **v1 - v7** | 初始探索与基础优化 | ~1050 | 远低于 **FA4** 与 **cuDNN** |
| **v8** | 首次显著性能跃升 | ~1110 | 仍低于基线 |
| **v13** | 架构调整带来大幅提升 | ~1280 | 差距显著缩小 |
| **v18** | 突破首个基线 | ~1430 | **超越 FA4 (1426)** |
| **v20** | 引入无分支累加器重缩放 | ~1460 | 逼近 **cuDNN** |
| **v29** | 流水线重叠优化 | ~1480 | 接近 **cuDNN (1488)** |
| **v40** | 最终演进版本 | **1520** | **全面超越 cuDNN 与 FA4** |

- **演进特征总结**：
  - **离散跃升**：性能提升集中在少数关键版本（如 v8, v13, v18, v20, v29），对应重大的架构或微架构优化。
  - **平台期与微调**：在重大跃升之间存在平台期，后续版本通过寄存器重平衡等细粒度调整持续榨取性能余量。
  - **长尾累积**：后期版本（v21-v40）虽然单次提升幅度减小，但通过持续累积最终实现了对顶尖专家手工优化内核的全面超越。

### Figure 6:Evolution trajectory of AVO across 40 kernel versions over 7 days on non-causal MHA. The solid green line tracks the running-best geometric mean throughput across all configurations; green circles mark versions that set a new best. Dashed colored lines show per-configuration throughput (seq_len = 4k, 8k, 16k, 32k). Horizontal dashed lines indicate the geometric mean throughput of cuDNN and FA4.

![x6.png](images/x6.png)

- **图表概述**：该图展示了 **AVO** 在 **Non-Causal BF16 Multi-head Attention** 任务中，历经 7 天、40 个 **Kernel version** 的自主进化轨迹。
- **坐标轴与视觉元素**：
  - **Y轴**：计算吞吐量，单位为 **TFLOPS**（刻度范围 1300 至 1700）。
  - **X轴**：内核版本迭代，从 **v1** 至 **v40**。
  - **核心曲线**：绿色实线代表 **Running best**（历史最佳几何平均吞吐量），绿色圆点标记 **New best**（刷新历史最佳的版本）。
  - **配置曲线**：彩色虚线分别代表不同序列长度（**seq_len=4k, 8k, 16k, 32k**）的独立吞吐量表现。
  - **基线参考**：蓝色水平虚线为 **cuDNN (1611)**，红色水平虚线为 **FA4 (1620)**。
- **核心性能数据**：
  | 评估对象 | 吞吐量 (TFLOPS) |
  | :--- | :--- |
  | **cuDNN** 基线 | 1611 |
  | **FA4** 基线 | 1620 |
  | **AVO v40** 最终版本 | 1630 |
- **进化阶段分析**：
  - **初期探索阶段 (v1-v11)**：性能在 1300 至 1400 TFLOPS 区间震荡，尚未触及基线水平。
  - **首次突破阶段 (v12-v13)**：实现第一次阶梯式跃升，**Running best** 从约 1330 TFLOPS 跃升至 1480 TFLOPS。
  - **核心跃升阶段 (v20-v21)**：迎来进化过程中最显著的性能飞跃，**Running best** 从 1480 TFLOPS 直接攀升至 1590 TFLOPS，大幅缩小与基线的差距。
  - **后期微调与超越阶段 (v27-v40)**：进入精细化调优期，通过 **v28**、**v36** 和 **v40** 的持续迭代，最终在 **v40** 达到 **1630 TFLOPS**，成功超越 **cuDNN** 和 **FA4** 两大基线。
  - **长序列优势**：在进化后期，**seq_len=16k** 和 **seq_len=32k** 的配置展现出极高的性能上限，峰值吞吐量突破 1650 TFLOPS，显著拉高了整体几何平均值。

### Figure 7:Multi-head attention forward-pass throughput (TFLOPS) on NVIDIA B200, comparing AVO (measured on our hardware) against cuDNN and FA4 baseline numbers as reported in the FA4 paper [24]. Head dimension 128, 16 heads, BF16. Left: non-causal. Right: causal.

![x7.png](images/x7.png)

- **图片概述**：该图展示了在 **NVIDIA B200** GPU 上，**Multi-head attention** 前向传播的吞吐量（**TFLOPS**）对比。对比对象包括 **cuDNN**、**FA4**（基于 FA4 论文报告的数据）以及本文提出的 **AVO**（在作者硬件上实测）。实验配置为 **Head dimension 128**、**16 heads**、**BF16** 精度。
- **图表结构**：包含左右两个子图，左侧为 **non-causal (causal=False)** 注意力机制，右侧为 **causal (causal=True)** 注意力机制。X 轴表示不同的序列长度（**4K, 8K, 16K, 32K**）及对应的 batch size（**bs=8, 4, 2, 1**），Y 轴表示吞吐量（**TFLOPS**）。

- **详细数据对比**：

| 序列长度 (Batch Size) | cuDNN (Non-causal) | FA4 (Non-causal) | AVO (Non-causal) | cuDNN (Causal) | FA4 (Causal) | AVO (Causal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **4K (bs=8)** | 1552 | 1532 | **1573** | 1295 | 1279 | **1392** |
| **8K (bs=4)** | 1585 | 1579 | **1615** | 1430 | 1426 | **1482** |
| **16K (bs=2)** | 1609 | 1601 | **1664** | 1509 | 1526 | **1582** |
| **32K (bs=1)** | 1613 | 1613 | **1668** | 1540 | 1576 | **1637** |

- **核心发现与分析**：
  - **全面领先**：在所有测试的序列长度和注意力掩码配置下，**AVO** 的吞吐量均**超越**了 **cuDNN** 和 **FA4** 的基线数据。
  - **Non-causal 性能提升**：在 **non-causal** 场景下，**AVO** 的吞吐量范围在 **1573 至 1668 TFLOPS** 之间，相较于 **FA4** 基线提升了约 **2.3% 至 3.9%**。
  - **Causal 性能提升显著**：在 **causal** 场景下，**AVO** 的优势更为突出，吞吐量范围在 **1392 至 1637 TFLOPS** 之间，相较于 **FA4** 基线提升了约 **3.7% 至 8.8%**，尤其在短序列（**4K**）下提升幅度最大。
  - **基线对比严谨性**：该图特意将 **AVO** 的实测数据与 **FA4 论文中报告的基线数据**进行对比，旨在排除系统级差异（如驱动版本、热状态、时钟频率）对绝对 **TFLOPS** 数值的影响，进一步验证了 **AVO** 的鲁棒性和优越性。

