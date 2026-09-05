# Beyond Prefill-Decode Disaggregation: Dissecting LLM Inference for Heterogeneous Platforms via Dynamic Operator Scheduling 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Jiaqi Yang, Jiayi Li, Yihan Fu, et al.

**发表期刊/会议 (Journal/Conference)**: unknown

**发表年份 (Publication Year)**: 2026

**研究机构 (Affiliations)**: Peking University

---

## 1. 摘要

**目的**

- 解决异构 NPU-PIM 系统上 LLM 推理的算子放置问题。
- 超越传统的 Prefill-Decode Disaggregation (PD) 和基于 Roofline 的静态分配策略。
- 联合优化算子调度与权重布局，减少 relayout 开销并避免内存重复，从而最大化性能或效率。

---

**方法**

提出 DOPS (Dynamic Operator Scheduling) 框架，构建从输入到部署验证的闭环优化流程。

![](images/f2626a5b17f6a1a83a1c63253d89be55853a794eedb64df457d58947d17d3554.jpg) *Figure 4: Overall workflow of DOPS for dissecting LLM inference on a heterogeneous NPU–PIM system. E2E: end-to-end.*

- **Computation Graph Builder**：构建 stage-aware DAG，显式表示算子放置与权重布局的耦合关系，包含 Prefill 和 Decode 阶段的算子序列及数据依赖。
- **Bifocal Scheduler**：动态列表调度器，基于 Bifocal Score 进行算子到设备的映射。
  - **Near-focus**：结合 Earliest Finish Time (EFT) 和 DAG-window lookahead，评估短期完成时间及对后续算子链的扰动影响。
  - **Far-focus**：包含 phase-reuse bias 和 token-amortization bias，捕获跨阶段权重复用潜力和自回归 Decode 的长期摊销成本。
- **Weight Layout Arbiter (WLA)**：两阶段块坐标搜索策略，优化主存中的权重格式。
  - **Outer Stage**：基于上一次调度提取的块级 reload pressure，进行粗粒度格式分配。
  - **Inner Stage**：通过局部坐标搜索进行针对性细化，仅在预测 makespan 下降时接受格式翻转。

---

**结果**

在华为 Ascend 910B NPU 与 SK Hynix AiM GDDR6-PIM 组成的异构平台上进行了广泛评估。

| 硬件组件 | NPU | PIM |
| :--- | :--- | :--- |
| **设备型号** | Huawei Ascend910B | SK Hynix AiM |
| **峰值算力** | 280 TFLOPS @FP16 | 16 TFLOPS @FP16 |
| **内存容量** | 16 GB | 16 GB/device |
| **峰值带宽** | 0.8 TB/s | 16 TB/s/device |

- **调度性能**：Bifocal 相比 PD baseline 实现了 **1.20×** 到 **2.23×** 的几何平均加速。
- **布局优化**：WLA 在 Bifocal/Linear 基础上额外提供了 **1.28×** 到 **1.33×** 的几何平均加速，且几乎匹配双副本基准的性能而不翻倍内存占用。
- **模型验证**：模拟与实际执行之间的误差极小，加速比差距在 **-4%** 到 **+6%** 之间。
- **硬件扩展性**：增加 PIM 容量的边际收益非普遍存在，受 prefill length、decode length、batch size 和模型大小共同决定。

![](images/a7d159be5575147fd780535774a1db1684d892147db591139dbd0a8afb418f6b.jpg)

---

**结论**

- DOPS 成功将异构 NPU-PIM 系统上的 LLM 推理形式化为耦合的调度与布局优化问题。
- 证明了在边缘导向系统中，超越粗粒度 PD 分离和静态 Roofline 规则的必要性。
- 为未来 LLM serving 的软硬件协同设计提供了系统级优化层与开源工具支持。

---

## 2. 背景知识与核心贡献

**研究背景**

- LLM 推理在严格的 SLO 和内存预算下面临高昂的延迟与服务成本，在边缘平台尤为突出。
- LLM 请求包含两个执行特征截然不同的阶段：**Prefill** 阶段具有高算术强度，而 **Decode** 阶段受限于带宽且需不断读取增长的 KV cache。
- 传统的 **Prefill-Decode Disaggregation (PD)** 策略将两阶段解耦，但在异构系统中显得过于粗粒度。
- **NPU-PIM** 异构系统结合了计算密集型的 NPU 和访存密集型的 PIM，为边缘 LLM 部署提供了互补的硬件底座。

![](images/839e6c252159be098ed484ab262b356d392acf59a7ea1c8fa8f8f170a6af758f.jpg) *Figure 1: (a) Challenges for LLM inference on NPU–PIM systems. (b) The best static mapping policy for minimizing latency varies across workload configurations, parameterized by {prefill length (x), decode length (y), and batch size (z)}. PD, AF, PD+Linear, PD+FFN, and PD+Attn are diferent policies. See Table 1 for details.*

---

**研究动机**

- **静态调度与 Roofline 模型的局限性**：
  - 实际内存带宽受设备间传输和缓存驻留影响，并非恒定值。
  - 峰值算力往往高估实际性能，尤其在非线性算子和短 Decode 核心上。
  - 静态放置规则缺乏时间维度考量，易导致设备争用和排队，局部最优的设备选择可能延长全局端到端延迟。
- **权重布局不匹配问题**：
  - NPU 偏好分块布局，PIM 偏好 Bank/Channel 感知布局。
  - 维护单一设备无关的副本会导致运行时频繁的 Relayout 开销。
  - 维护双副本虽消除转换开销，但会使内存占用翻倍，违反边缘设备的内存预算。
- **缺乏量化框架**：需要一种机制来预测 NPU-PIM 异构系统对特定 LLM 工作负载是否有益，并联合优化算子放置与权重布局。

---

**核心贡献**

- 提出 **DOPS** (Dynamic Operator Scheduling)，一个硬件感知的闭环优化框架，将 LLM 推理形式化为耦合的调度与数据存储优化问题。
- 设计 **Bifocal** 动态调度器：
  - 结合严格的完成时间估计与下游前瞻及重用感知偏差。
  - 包含 Near-focus（局部完成时间与短窗口前瞻）和 Far-focus（跨阶段权重重用与 Token 摊销）评分机制。
- 开发 **Weight Layout Arbiter (WLA)**：
  - 采用两阶段块坐标搜索策略。
  - 在严格内存约束下选择分块权重布局，平衡 Relayout 开销与内存重复成本。
- 构建闭环工作流：从用户输入、仿真、部署到验证，支持系统瓶颈分析与硬件可扩展性探索。

![](images/f2626a5b17f6a1a83a1c63253d89be55853a794eedb64df457d58947d17d3554.jpg) *Figure 4: Overall workflow of DOPS for dissecting LLM inference on a heterogeneous NPU–PIM system. E2E: end-to-end.*

---

**性能表现**

- 在结合 Huawei Ascend 910B NPU 与 SK Hynix AiM GDDR6-PIM 的代表性异构系统上评估。
- **Bifocal** 调度器相较于 PD 基线实现几何平均加速比：

| 模型 | 硬件配置 | 几何平均加速比 |
| :--- | :--- | :--- |
| Qwen-1.8B | HP32 | 2.23x |
| Qwen-7B | HP32 | 1.36x |
| Qwen-14B | HP32 | 1.44x |
| Llama-7B | HP64 | 1.20x |
| Llama-13B | HP64 | 1.25x |
| Mixtral-8x7B | HP128 | 1.72x |
| Llama-70B | HP128 | 1.20x |

- **WLA** 在 Bifocal/Linear 基础上额外提供 **1.28x** 至 **1.33x** 的几何平均加速比。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体架构概述**

本文提出了 **DOPS (Dynamic Operator Scheduling)** 框架，这是一个硬件感知的闭环优化系统，专门针对异构 **NPU-PIM** 平台上的 **LLM Inference** 场景。该框架将算子调度与权重布局优化形式化为一个耦合问题，通过构建阶段感知的 **DAG**，联合优化执行时间与内存开销。

![](images/649f067c3884e9266bc1f22584535da3b24071ed87c0a0e4161c875d79747d0b.jpg)

---

**输入层**

DOPS 接收三类核心输入以驱动整个优化流程：

- **LLM Model Card with Optimization**：定义目标架构与算子形状，支持建模 **Quantization** 与 **Sparsification**，以反映其对内存占用与通信流量的影响。
- **Hardware Abstraction**：描述目标 **NPU-PIM** 系统，包含设备类型、数量、内存资源与互连拓扑。性能输入源自四个抽象层级：**Roofline models**、**Operator-level measurements**、**Program snippets** 与 **End-to-end measurements**。
- **Workload Configuration**：描述服务形态与搜索空间，包括 **Prefill length**、**Decode length**、**Batch size**、优化目标（如 **TTFT**、**TPOT**）以及内存容量等约束。

---

**核心功能组件**

基于输入，DOPS 构建以下基础组件以支撑调度决策：

- **Computation Graph Builder**：构建阶段感知的 **DAG**。每个顶点代表可调度任务，记录张量形状、精度与算术工作量；有向边编码数据依赖与数据移动（包括激活传输、**KV cache** 更新、权重加载与 **Relayout**）。
- **Performance Model**：将标注后的 **DAG** 转换为延迟基元，估算每个算子在不同设备上的计算时间、内存服务时间、权重加载开销以及缓存未命中的惩罚。
- **Communication Topology**：指定互连结构与带宽约束，支持全连接拓扑与星型拓扑。通过将路由映射到共享通信管道，建模运行时争用与传输重叠。

---

**核心优化引擎**

DOPS 的优化循环由两个核心引擎驱动：

- **Bifocal Scheduler**：一种列表调度器，基于 **HEFT** 算法扩展，通过计算 **Bifocal score** 动态决定算子在 **NPU** 与 **PIM** 间的放置策略。
  - **Near-focus score**：结合 **EFT (Earliest Finish Time)** 与局部 **DAG-window lookahead** 估算，评估当前设备可用性与短期下游影响。
  - **Far-focus score**：包含 **Phase-reuse bias**（捕获跨阶段权重-设备一致性）与 **Token-amortization bias**（捕获解码阶段自回归特性，允许当前步承担更高成本以摊销未来开销）。

- **Weight Layout Arbiter (WLA)**：在严格内存约束下，为内存中的持久化权重块选择最高效的物理布局。
  - 采用两阶段块坐标搜索策略。
  - **Outer Stage (Dominance Assignment)**：基于上一次调度产生的块级加载压力，进行粗粒度格式重分配。
  - **Inner Stage (Targeted Refinement)**：通过局部坐标搜索，对特定候选块进行格式翻转评估，仅当预测 makespan 显著下降时才接受更新。

---

**输出与验证闭环**

- **Outputs**：生成具体的算子到设备映射方案（含起止时间）、优化的块级权重存储计划（**Linear**, **NPU_OPT**, **PIM_OPT** 等）以及模拟的端到端延迟。
- **Verify & Deploy Loop**：将生成的调度与布局部署到目标 **NPU-PIM** 系统，测量真实执行延迟。通过对比经验延迟与模拟预测，评估模型保真度与优化有效性，形成闭环反馈。

### 1. Bifocal 动态算子调度机制

**核心概念与定位**

Bifocal 是 DOPS 框架中的核心动态调度器，基于列表调度算法在异构 NPU-PIM 系统上执行算子放置。它通过构建增量式的执行时间线，结合严格的完成时间估计与下游前瞻能力，解决静态规则无法适应动态工作负载的问题。

---

**输入与输出关系**

Bifocal 接收 DAG 图、合法设备集和初始系统状态，输出具体的算子映射和时间表。

| 类别 | 具体内容 |
| --- | --- |
| **输入** | 执行 DAG $G=(V,E)$；合法设备集 $\{\mathcal{K}_v\}$；权重格式 $\phi$；初始系统状态 $s_0$ |
| **输出** | 算子到设备的映射 $\sigma: V \to \mathcal{K}$；开始/完成时间表 $\tau = \{(\tau_{vs}, \tau_{vc})\}$；更新后的系统状态 $s$ |

---

**Bifocal Score 评分机制**

Bifocal 的核心在于其双焦评分模型，通过综合近焦和远焦指标来评估算子放置的优劣，分数越低越优先。公式为：$Score(v, k) = C_{near}(v, k) + C_{far}(v, k)$。

**近焦指标**
关注当前调度状态下的局部最优性。
- **最早完成时间 $EFT(v, k)$**：结合前驱节点到达时间、设备可用时间 $avail(k)$、设备执行时间 $p_v(k)$ 以及缓存引起的权重重载惩罚 $t_{reload}$，计算局部完成时间估计。
- **DAG 窗口前瞻项 $\widehat{T}_H(v, k)$**：通过递归追踪最大负载的依赖边，探索最多 $H$ 个顶点的后继链。在此窗口内枚举合法设备分配，选择模拟完成时间最小的方案，评估当前提交对下游的扰动。

**远焦指标**
捕获跨阶段和跨 Token 的长期成本效益。
- **阶段重用偏差 $B_{phase}(v, k)$**：维护权重-设备提示图 $\theta$。若候选设备与历史记录匹配则无惩罚，不匹配则产生正惩罚。在 Prefill 阶段，若设备容量能容纳全部权重，则给予负偏差以鼓励驻留。
- **Token 摊销偏差 $B_{token}(v, k)$**：针对 Decode 阶段的自回归特性。计算当前 Token 成本与剩余解码步数 $R_i$ 内的平均成本差异。当剩余 Token 较多时，调度器可接受当前较高的迁移成本以摊销未来开销。

---

**调度算法流程**

- 初始化所有算子的映射与时间表为空，设置系统状态 $s$ 与就绪任务集 $\mathcal{R}$。
- 当 $\mathcal{R}$ 非空时循环：
  - 遍历就绪集 $\mathcal{R}$ 中的每个任务 $v$ 及其合法设备 $k$。
  - 调用 Evaluate 函数计算 Bifocal Score 及其关联的提示图。
  - 将评估结果压入以分数为键的最小堆 $\mathcal{Q}$。
  - 弹出堆顶任务 $(v^*, k^*)$，提交至当前调度计划。
  - 更新系统状态：设备可用性、缓存状态 $c$、近未来提示图 $H$ 及权重-设备提示图 $\theta$。
  - 将 $v^*$ 移出就绪集，插入新的就绪任务。

---

**关键参数设置**

- $\gamma \in [0,1]$：调度器超参数，用于平衡 $EFT$ 与 $\widehat{T}_H$ 的权重。
- $H$：DAG 窗口前瞻的最大顶点数。
- $\alpha \geq 0$：Token 摊销缩放系数，控制长期成本优化的影响力度。
- $\epsilon$：早停容忍度，用于 WLA 阶段判断是否接受格式翻转。

---

**在 DOPS 框架中的作用**

Bifocal 调度器是 DOPS 闭环优化流程的执行引擎。
- **驱动 WLA 优化**：Weight Layout Arbiter (WLA) 在每次迭代中调用 Bifocal 来评估特定权重布局下的 makespan，并提取调度衍生的加载统计数据。
- **生成部署方案**：将抽象的计算图转化为具体的设备映射与时间线，直接用于目标 NPU-PIM 系统的部署与验证。
- **提供性能基准**：通过模拟预测端到端延迟，为不同调度策略和硬件配置提供统一的比较基准。

---

**可视化展示**

![](images/649f067c3889c9266bc1f22584535da3b24071ed87c3a0e4161c875d79747d0b.jpg)

### 2. 权重布局仲裁器 (WLA)

**核心概念与背景**

在异构 NPU-PIM 系统中，NPU 倾向于分块布局以高效馈送 Tensor 引擎，而 PIM 倾向于 Bank 或 Channel 感知布局以利用内部并行性。维持单一设备无关的副本会导致运行时重复重排，而维护双设备特定副本会膨胀内存占用并违反边缘设备内存预算。**Weight Layout Arbiter (WLA)** 旨在解决此耦合问题，通过在严格内存约束下，为内存中的持久化权重选择块级布局，以最小化预测的端到端延迟。

![](images/8a84979cc4fb470c677b6a483f9d6a8ce74bdf7dcc865e24517351f8f97e3ce6.jpg) *Figure 6: Two-stage block-coordinate search for optimizing the blockwise weight-format map. The left panel summarizes initialization; the upper-right and lower-right panels specify the outerand inner-stage update rules, respectively.*

**输入与输出关系**

- **输入**:
  - 共享的执行 DAG $G = (V, E)$、设备集 $\mathcal{K}$ 及成本原语（与 Bifocal 调度器相同）。
  - 权重标识符集合 $W$ 及其划分的稳定块集合 $B$。
  - 支持的存储格式集合 $F$（包含 **Linear**, **NPU_OPT**, **PIM_OPT**）。
- **输出**:
  - 最优的块级格式映射 $\phi_{best}: B \to F$。
  - 预测的最小 makespan $T_{best}$。

**两阶段块坐标搜索算法流程**

WLA 采用 **Two-Stage Block Coordinate Search** 策略，在离散空间 $F^{|B|}$ 中搜索，避免全量精确坐标扫描的高昂成本。

- **初始化**:
  - 将所有权重块分配为 **Linear** 格式。
  - 调用 Bifocal 调度器评估一次，获取初始 makespan $T(\phi_{ini})$ 及调度衍生的每权重加载统计。
  - 将每权重统计聚合至块粒度并归一化，得到 NPU 与 PIM 的加载压力指标 $\tilde{\ell}_{b, NPU}$ 和 $\tilde{\ell}_{b, PIM}$。
- **外部阶段：主导分配**:
  - 基于上一轮调度观察到的块级重载压力执行粗略的块级重分配。
  - 比较 $\tilde{\ell}_{b, NPU}$ 与 $\tilde{\ell}_{b, PIM}$ 的主导关系，为每个块 $b$ 决定一个有潜力的格式。
- **内部阶段：定向细化**:
  - 构建候选集：包含当前格式与刷新后调度中观察到的加载行为不一致的块。
  - 对每个候选块 $b$，固定其他块分配，显式评估邻近映射 $\phi^{(bf)}$（其中 $f \in F \setminus \{\phi(b)\}$）。
  - 仅当预测 makespan 减少超过 $\epsilon$ 时接受翻转。
  - 每次接受翻转后，再次调用 Bifocal 更新调度状态，再处理下一候选块。
- **终止条件**:
  - 外部阶段不再产生格式变更。
  - 达到迭代预算。
  - 激活早停容差 $\epsilon$。

**参数设置与核心指标**

| 参数/指标 | 说明 |
| --- | --- |
| **格式集合 $F$** | Linear, NPU_OPT, PIM_OPT |
| **早停容差 $\epsilon$** | 接受格式翻转的最小 makespan 降幅阈值 |
| **加载压力 $\tilde{\ell}_{b, c}$** | 归一化的块级权重在特定设备类 $c$ 上的重载压力 |
| **评估引擎** | 调用 Bifocal 获取 makespan 及加载统计 |

**在整体架构中的作用**

- **调度与布局耦合优化**: WLA 与 **Bifocal scheduler** 形成闭环。WLA 的每次格式评估均依赖 Bifocal 提供的 makespan 预测与加载统计；而 Bifocal 的调度决策又受 WLA 选定的权重格式影响（因格式影响加载成本与重排开销）。
- **内存与性能平衡**: 通过动态选择混合权重布局，WLA 避免了双副本带来的内存翻倍，同时最大程度减少运行时重排开销。
- **性能提升**: 实验表明，在 HP32 平台上，**Bifocal/WLA** 相比 **Bifocal/Linear** 几何平均加速比达 **1.28× 至 1.33×**，且在短序列场景下收益更大。其性能几乎媲美理想化的双副本基线 **Bifocal/Dual**。

### 3. 调度与布局联合优化的闭环框架 (DOPS)

**DOPS框架核心定位与作用**

DOPS（Dynamic Operator Scheduling）是一个硬件感知的闭环优化框架，将NPU-PIM异构系统上的LLM推理形式化为**调度与数据存储联合优化问题**。该框架突破传统Prefill-Decode Disaggregation (PD) 粗粒度分离和静态Roofline模型的局限，通过联合导航算子放置与权重布局的设计空间，最大化系统性能或效率。

---

**输入与输出关系**

DOPS构建从用户输入到仿真部署的完整闭环，其输入输出映射关系如下：

| 组件类型 | 输入要素 | 输出结果 |
| :--- | :--- | :--- |
| **模型与硬件** | LLM Model Card (含量化/稀疏配置)<br>Hardware Abstraction (NPU/PIM数量、内存、互连拓扑) | **Operator Placement & Scheduling**<br>算子到设备的映射及起止时间 |
| **工作负载** | Workload Configuration (Prefill/Decode长度、Batch Size、SLO约束) | **Optimized Weight Layout**<br>主存中持久化权重的块级存储计划 |
| **性能评估** | Roofline模型、算子级测量数据、程序片段 | **Simulated Performance**<br>端到端延迟预测及候选策略对比 |

---

**核心功能组件与执行流**

DOPS的优化循环由以下核心组件驱动：

- **Computation Graph Builder**：构建阶段感知的DAG（有向无环图）。每个顶点代表可调度任务，有向边编码数据依赖与数据移动（如激活传输、KV cache更新、权重加载、Relayout操作）。该表示使得算子放置与权重布局的耦合关系在单一DAG中显式化。
- **Performance Model**：将带注释的DAG转换为延迟原语。针对每个算子-设备对，估算计算时间、内存服务时间、权重加载成本、Relayout成本及Cache-miss处理成本。
- **Communication Topology**：指定互连结构与带宽约束。支持全连接拓扑与星型拓扑，通过将路由映射到共享通信管道来建模运行时争用，评估数据传输延迟。

![](images/f2626a5b17f6a1a83a1c63253d89be55853a794eedb64df457d58947d17d3554.jpg) *Figure 4: Overall workflow of DOPS for dissecting LLM inference on a heterogeneous NPU–PIM system. E2E: end-to-end.*

---

**优化引擎一：Bifocal Scheduler**

Bifocal是一个列表调度器，在异构NPU-PIM系统上动态决定算子放置。其核心在于引入**Bifocal Score**，综合评估当前局部完成时间与下游全局影响。

**Bifocal Score 计算公式**：
`Score(v, k) = C_near(v, k) + C_far(v, k)`

| 评分维度 | 组成部分 | 功能与原理 |
| :--- | :--- | :--- |
| **Near-focus** | **EFT(v, k)** | 最早完成时间。结合前驱到达时间、设备可用性、设备执行时间及权重重载惩罚，估算局部最优。 |
| **Near-focus** | **$\widehat{T}_H(v, k)$** | DAG-window lookahead。对包含最多H个顶点的后继链进行轻量级前瞻仿真，评估提交当前任务对下游的扰动。 |
| **Far-focus** | **$B_{phase}(v, k)$** | Phase-reuse bias。捕获调度局部的权重-设备一致性。若候选设备与历史权重放置设备不一致，则引入正惩罚；Prefill阶段对能容纳全模型权重的设备给予负偏置。 |
| **Far-focus** | **$B_{token}(v, k)$** | Token-amortization bias。捕获Decode阶段自回归特性。当剩余Decode horizon较长时，调度器可接受当前较高的迁移成本，以降低未来Token的平均成本。 |

**算法流程**：
- 初始化系统状态，包含ready set、设备可用性、Cache状态及hint maps。
- 循环评估ready set中的每个任务在所有合法设备上的Bifocal Score。
- 将候选任务推入min-heap，键值为Bifocal Score。
- 提交Score最小的任务，更新调度时间线、Cache状态及hint maps，直至DAG所有任务调度完毕。

![](images/649f067c3884c9266bc1f22584535da3b24071ed87a3a0e4161c875d79747d0b.jpg)

---

**优化引擎二：Weight Layout Arbiter (WLA)**

WLA在严格内存约束下，为主存中的持久化权重块选择硬件高效的存储格式（Linear, NPU_OPT, PIM_OPT），最小化预测的端到端makespan。

**问题形式化**：
- 权重集合W被划分为稳定块B。
- 目标：寻找块级映射 $\phi: B \to F$，使得由Bifocal预测的makespan $T(\phi)$ 最小。

**两阶段块坐标搜索策略**：

- **初始化**：所有块分配为Linear格式，调用Bifocal评估获取初始makespan及每权重加载统计。将统计聚合至块粒度并归一化，得到 $\tilde{\ell}_{b, NPU}$ 和 $\tilde{\ell}_{b, PIM}$。
- **Outer Stage (Dominance Assignment)**：基于上一轮调度的块级重载压力进行粗粒度重分配。比较 $\tilde{\ell}_{b, NPU}$ 与 $\tilde{\ell}_{b, PIM}$ 的支配关系，为每个块决定有潜力的格式。
- **Inner Stage (Targeted Refinement)**：通过局部坐标搜索细化映射。构建候选集（当前格式与观测加载行为不一致的块）。固定其他块分配，评估邻近映射 $\phi^{(bf)}$，仅当预测makespan下降超过阈值 $\epsilon$ 时接受翻转。每次接受翻转后重新调用Bifocal。
- **终止条件**：外层无格式变更、达到迭代预算或触发早停容差 $\epsilon$。

![](images/8a84979cc4fb470c677b6a483f9d6a8ce74bdf7dcc865e24517351f8f97e3ce6.jpg) *Figure 6: Two-stage block-coordinate search for optimizing the blockwise weight-format map. The left panel summarizes initialization; the upper-right and lower-right panels specify the outerand inner-stage update rules, respectively.*

---

**闭环验证与部署**

- **Verify & Deploy Loop**：将生成的DAG调度与优化后的权重布局部署至目标NPU-PIM系统。
- **测量与反馈**：通过程序片段或端到端测量获取真实执行延迟。对比实测延迟与仿真预测，验证目标是否满足，并直接评估性能模型的保真度与优化引擎的有效性。


---

## 4. 实验方法与实验结果

**实验设置**

- Benchmarks与Workloads
  - 模型选择：覆盖Llama-7B/13B/70B、Qwen-1.8B/7B/14B以及Mixtral-8x7B，包含Dense、GQA与MoE架构。
  - 量化策略：Llama-70B与Mixtral-8x7B采用INT8量化，其余模型使用FP16。
  - 参数扫描：Prefill长度{128, 512, 1024, 2048}，Decode长度{128, 256, 512, 1024}，Batch size{1, 4, 8, 16}。
- Hardware Settings
  - 物理设备：Huawei Ascend 910B NPU搭配SK Hynix AiM GDDR6-PIM。
  - 平台变体：构建HP0、HP32、HP64、HP128四种配置，后缀数字代表PIM总容量（GB）。
  - 互联与内存：PCIe Gen4 x16全连接拓扑；KV cache按头粒度存于PIM；权重块采用LRU策略驱逐；计算与权重加载允许最大程度重叠。

| 设备类型 | NPU | PIM |
|---|---|---|
| 型号 | Huawei Ascend 910B | SK Hynix AiM |
| Peak compute throughput | 280 TFLOPS @FP16 | 16 TFLOPS @FP16 |
| Memory | 16 GB | 16 GB/device |
| Peak memory bw. | 0.8 TB/s | 16 TB/s/device |

- Performance Models与Baselines
  - NPU调度输入：采用校准后的roofline model，结合利用率拟合函数与CANN内核启动开销。
  - PIM调度输入：采用operator model。
  - Relayout成本估算：内存侧使用Ramulator2，NPU侧使用Huawei CANN转换内核。
  - 对比基线：静态策略PD、AF、PD+Linear、PD+Attn、PD+FFN；布局策略PD/Linear、PD/Dual、Bifocal/Linear、Bifocal/WLA、Bifocal/Dual。

---

**结果数据分析**

- 调度性能与加速比
  - 绝对延迟与加速：在HP32平台上，Llama-7B模拟加速比达1.10×至1.48×（验证最高1.47×）；Qwen-1.8B模拟加速比达1.89×至2.43×（验证最高2.41×）。

![](images/0ba1a5019bf56ee283b569dec986a49912e23146c4fd42c57c57212412ed88f5.jpg)

  - 几何平均加速比：Bifocal调度器在不同硬件配置下相对PD基线取得显著提升。

| 模型 | 平台配置 | 几何平均加速比 |
|---|---|---|
| Qwen-1.8B | HP32 | 2.23× |
| Qwen-7B | HP32 | 1.36× |
| Qwen-14B | HP32 | 1.44× |
| Llama-7B | HP64 | 1.20× |
| Llama-13B | HP64 | 1.25× |
| Mixtral-8x7B | HP128 | 1.72× |
| Llama-70B | HP128 | 1.20× |

- 模拟与验证误差
  - 模拟与实际验证加速比之间的Signed Gap维持在**-4%至+6%**区间，证明DOPS性能模型具备高保真度。

![](images/737626ae51e220563e9a7de1223ba5574542a13081126cf2d7e5b96f0070134.jpg)

- 设备利用率与协同计算
  - DOPS未孤立最大化NPU或PIM利用率，而是实现最高的**CoUtil**（时间平均重叠利用率）。
  - 动态分配策略选择最匹配当前全局调度的设备，而非仅看单机执行时间，有效避免关键路径上的排队拥塞。

![](images/e57212316c60490bd3bf025b50894e47bc3b0f5c991044b981f097365b669784.jpg)

- 硬件扩展性与边际收益
  - 增加PIM容量总体有益，但边际收益受Workload配置显著影响。

![](images/bcece7ebf1e566decd6a5b288b45788fcb29c74ee016757629a8deeb79f5f4e.jpg)

  - **Insight 1**：更长的Prefill长度将最高收益点推向更大PIM容量（因KV cache流量与attention内存成本增加）。
  - **Insight 2**：更长的Decode长度同样导致收益点右移（因内存访问重复性高，大PIM预算更易均摊成本）。
  - **Insight 3**：更大的Batch size会延迟PIM容量的收益兑现（因大Batch提升了NPU基线利用率）。
  - **Insight 4**：更大的Model size将最优边际收益点推向更大PIM容量（因权重增大易触发LRU替换）。

- WLA有效性验证
  - 延迟降低：相比Bifocal/Linear，Bifocal/WLA延迟降低**8.3%至40.4%**，在短序列下因布局开销占主导而收益最大。
  - 加速表现：在Qwen-1.8B上，相比PD/Linear几何平均加速**2.12×**，相比Bifocal/Linear加速**1.32×**。
  - 收敛特性：从全Linear布局出发，经过粗粒度块移动与有限细化，通常在3次外循环内快速收敛至低延迟混合布局。

![](images/840a04a34e4529ba8827cd8c2f696c4fd7f209077130599510e30ae34b93d26.jpg)

---

**消融实验分析**

- 实验目的：隔离Bifocal调度器各评分组件的贡献，验证其对端到端延迟的具体影响。
- 消融组件设置：
  - **EFT-only**：仅保留最早完成时间评估。
  - **Bifocal w/o LA**：移除DAG-window lookahead。
  - **Bifocal w/o Phase**：移除phase-reuse bias。
  - **Bifocal w/o Token**：移除token-amortization bias。
  - **Bifocal**：完整模型。

![](images/00f5fddfb4234b2d66248d70a9132e3b8c863123adba5e300d3c83f9aa8d5f20.jpg)

- 核心发现：
  - **EFT-only**捕获了大部分性能增益，因其综合考量了前驱完成时间、通信延迟、设备可用性及缓存重载惩罚。
  - 移除**Phase-reuse**导致延迟增加**7.0%至10.0%**，证明跨Prefill-Decode边界与连续Decode步骤的weight-device亲和性至关重要。
  - 移除**Token-amortization**导致延迟增加**16.0%至25.6%**，证明其对长Decode阶段不可或缺，能有效均摊一次性迁移或重载成本。
  - 移除**DAG-window lookahead**导致延迟增加**6.4%至11.8%**，表明前瞻模拟能有效避免下游资源争用。

---

