# SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Hengrui Zhang, August Ning, Pratyush Patel, et al.

**发表期刊/会议 (Journal/Conference)**: ArXiv

**发表年份 (Publication Year)**: 2024

**研究机构 (Affiliations)**: Princeton University, University of Washington

---

## 1. 摘要

---
**目的**

- 解决现有 GPU/TPU 在 disaggregated LLM inference 中因“more-is-better”设计哲学导致的硬件资源利用率低下问题。
- Prefill 阶段属于 compute-bound，导致昂贵的 HBM memory bandwidth 利用不足。
- Decode 阶段属于 memory-bound，导致庞大的 compute capacity 利用不足。
- 旨在通过硬件定制化设计，在维持相同性能的前提下，显著降低 LLM serving 的硬件成本与 TDP。

---
**方法**

- 提出 **SPAD** (Specialized Prefill and Decode hardware)，采用 **less-is-more** 设计哲学，针对推理双阶段特性定制专用芯片。
- **Prefill Chip 设计**：
  - 增大 **Systolic Array** 尺寸至 32×32，提升 compute-bound 阶段的 tensor 运算性能。
  - 采用成本更低的 **GDDR7** 替代 HBM3，将 memory bandwidth 降至 2048 GB/s。
  - 减小 vector unit 宽度与 L2 cache 大小，节省芯片面积与成本。
- **Decode Chip 设计**：
  - 保留 **HBM3** 以满足 memory-bound 阶段对高 bandwidth 和大 capacity 的需求。
  - 缩小 **Systolic Array** 尺寸至 16×16，降低 compute capacity 以节省面积和功耗。
  - 减小 L1/L2 cache 容量，因为 decode 阶段对 cache 复用率较低。
- **Adaptive Reallocation**：设计时保留两种芯片运行另一阶段的能力，以应对未来 models 和 workloads 变化时的集群重分配。

---
**结果**

- 芯片级仿真对比（基于 LLMCompass 与 modeled H100）：

| Specifications | Prefill Chip | Decode Chip | H100 |
| :--- | :--- | :--- | :--- |
| Memory Protocol | GDDR7 | HBM3 | HBM3 |
| Memory Bandwidth | 2048 GB/s | 3352 GB/s | 3352 GB/s |
| Est. Die Area | 784 mm² | 520 mm² | 814 mm² |
| Est. Norm. HW Cost | 0.48 | 0.88 | 1 |
| Est. TDP | 596 W | 507 W | 700 W |
| Norm. Prefill Perf. | 1.08 | 0.69 | 1 |
| Norm. Decode Perf. | 0.80 | 0.97 | 1 |

![](images/55214a0d189e14452641e603d05d8c0aaf2b0155758754373efccec00c558086.jpg) *Figure 4. Proposed SPAD Cluster and Chips Overview. Die area is estimated and will be further explained in Section 6.1.*

- **性能与成本表现**：
  - **Prefill Chip** 平均提升 8% prefill 性能，硬件成本降低 52%。
  - **Decode Chip** 达到 97% decode 性能，TDP 降低 28%。
- **端到端集群仿真**：
  - 在生产级 traces 测试中，SPAD 集群相比 baseline 降低 19%-41% 硬件成本与 2%-17% TDP。
  - 当 models 和 workloads 发生变化时，通过 adaptive reallocation，SPAD 仍可实现 11%-43% 的硬件成本节省，验证了设计的 longevity。

---
**结论**

- **SPAD** 成功验证了针对 LLM 推理双阶段特性采用异构专用硬件的可行性。
- 通过 **less-is-more** 方法论，摒弃传统 GPU 最大化所有资源的策略，实现了 disaggregated LLM serving 的成本效益最优化。
- 该设计在显著降低 TCO 和功耗的同时，保持了与现有高端 GPU 相当的服务性能，并具备应对未来模型演进的长期适应能力。

---

## 2. 背景知识与核心贡献

**研究背景**

- LLM 推理需求急剧上升，服务成本极其高昂。
- LLM 推理具有显著的双阶段特性：
  - **Prefill 阶段**：并行处理输入 Prompt 生成首个 Token 与 KV Cache，属于计算密集型。
  - **Decode 阶段**：依赖历史 KV Cache 逐个生成后续 Token，属于访存密集型。
- 为提升硬件效率，业界采用 **Prefill-Decode Disaggregation** 调度策略，将两阶段分离至不同硬件执行，以实现阶段特定的资源管理。

---

**研究动机**

- 现有数据中心 GPU/TPU 遵循“多多益善”的设计哲学，在有限的晶圆面积内最大化计算与内存资源。
- 硬件规格与 Disaggregated LLM Inference 的负载需求严重错配：
  - **Prefill 阶段**：高算术强度导致昂贵的 HBM 带宽利用率严重不足。
  - **Decode 阶段**：低算术强度导致庞大的计算能力闲置。
- 资源利用率低下直接转化为高昂的服务成本，亟需针对双阶段特性进行硬件“适度配置”。

![](images/e27b6f5d5b75ce57e1a1c75575f50f7ca4df6a13e8d9d404e864c707b7ee0206.jpg)

---

**核心贡献**

- 提出 **SPAD (Specialized Prefill and Decode hardware)** 异构系统，采用“少即是多”设计方法论，针对不同阶段定制专用芯片，同时保留运行另一阶段的灵活性。
- 设计专用 **Prefill Chip**：
  - 扩大 **Systolic Arrays** 以加速计算密集的 Tensor 操作。
  - 采用高性价比的 **GDDR7** 内存替代 HBM，削减非核心的 Vector Units 与 L2 Cache。
- 设计专用 **Decode Chip**：
  - 保留高带宽 HBM3 以满足 KV Cache 的访存需求。
  - 缩减 Systolic Arrays、Vector Units 及 Cache 面积，以降低 TDP 与芯片面积。
- 核心性能与成本优势（对比 modeled H100）：

| 芯片类型 | 核心设计变更 | 性能表现 | 成本/TDP 变化 |
| --- | --- | --- | --- |
| **Prefill Chip** | 更大 Systolic Arrays, GDDR7 | 平均性能提升 **8%** | 硬件成本降低 **52%** |
| **Decode Chip** | 削减计算与 Cache, 保留 HBM3 | 性能达 **97%** | TDP 降低 **28%** |

- 集群级端到端仿真验证：
  - 在生产级 Trace 下，SPAD 集群在保持同等性能时，硬件成本降低 **19%-41%**，TDP 降低 **2%-17%**。
  - 具备自适应重分配能力，当模型或负载变化时，仍可实现 **11%-43%** 的硬件成本节省，证明了设计的长效性。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体技术架构概述**

SPAD是一种异构硬件系统，专为基于分离的LLM推理服务设计。其核心架构摒弃了传统GPU“多即是好”的设计理念，转而采用“少即是多”的方法论，针对LLM推理的Prefill和Decode两阶段截然不同的计算特性，定制了专用的硬件芯片。

**集群级架构**

SPAD集群由异构的Prefill Machines和Decode Machines组成，替代了传统的同构GPU集群。
- **硬件组成**：每台机器包含8颗定制芯片。Prefill Machines搭载Prefill Chips，Decode Machines搭载Decode Chips。
- **互联网络**：机器内部采用高速scale-up互联（如NVLink，900 GB/s带宽）；机器之间采用scale-out互联（如Infiniband，50 GB/s带宽），用于传输KV cache。
- **调度机制**：采用基于分离的调度。请求首先在Prefill Machines上处理生成KV cache，随后通过scale-out互联传输至Decode Machines完成后续Token生成。
- **配置与自适应**：根据目标工作负载动态配置两类机器的比例。当模型或工作负载发生变化时，支持自适应重新分配，允许任一类型的芯片运行另一阶段的任务以保持集群效率。

**芯片级架构**

SPAD基于H100 GPU作为参考设计，通过成本感知的架构设计空间探索，裁剪非必要组件以降低成本和TDP，同时保留运行另一阶段的灵活性。
- **Prefill Chip设计**：
  - **计算单元**：扩大Systolic Array尺寸（32x32）以提升Tensor计算能力；缩减Vector Unit宽度（16）和L2 Cache（32MB），因为非张量操作在Prefill阶段影响较小。
  - **内存系统**：采用GDDR7替代昂贵的HBM3。带宽降至2048 GB/s，容量64 GB。由于Prefill阶段是计算密集型，对内存带宽不敏感，且KV cache仅作临时存储，GDDR足以满足需求且大幅降低成本。
- **Decode Chip设计**：
  - **计算单元**：缩小Systolic Array尺寸（16x16）和Vector Unit宽度（8），因为Decode阶段算术强度低，大计算单元无法被充分利用。
  - **内存系统**：保留HBM3以提供高带宽（3352 GB/s）和大容量（80 GB），满足Decode阶段不断增长的KV cache存储需求和高内存访问要求。削减L1/L2 Cache大小，因为流式内存访问对Cache复用率低。

**核心硬件规格对比**

| 规格参数 | Prefill Chip | Decode Chip | H100 (参考) |
| --- | --- | --- | --- |
| Systolic Array | 32x32 | 16x16 | Eq. to 16x32 |
| Vector Width | 16 | 8 | Eq. to 32 |
| Memory Protocol | GDDR7 | HBM3 | HBM3 |
| Memory Bandwidth | 2048 GB/s | 3352 GB/s | 3352 GB/s |
| Memory Capacity | 64 GB | 80 GB | 80 GB |
| FP16/BF16 Tensor PFLOPs | 1.92 | 0.54 | 0.99 |
| Est. Die Area (4nm) | 784 mm² | 520 mm² | 814 mm² |
| Est. Norm. Total HW Cost | 0.48 | 0.88 | 1 |
| Est. TDP | 596 W | 507 W | 700 W |

**架构图示与设计探索**

![](images/55214a0d189e14452641e603d05d8c0aaf2b0155758754373efccec00c558086.jpg) *Figure 4. Proposed SPAD Cluster and Chips Overview. Die area is estimated and will be further explained in Section 6.1.*
![](images/096907e9c4cc5b3038ce0f6b336623fd333b7ff3b377bd646e1cbc3e5073bcb5.jpg)
![](images/1ecab74d3c44ed05b594d544ef471596ec45a6ee4e0afd937aa6e2baa38b8c7f.jpg)

### 1. Less-is-More 专用芯片设计方法论

---
**核心设计理念**
**Less-is-More**方法论摒弃了传统GPU**More-is-Better**的设计哲学，将**Cost**与**TDP**作为一等公民进行架构设计空间探索。该方法论针对LLM推理的**Prefill**（计算受限）与**Decode**（内存受限）阶段的算术强度差异，对硬件资源进行精准的**Rightsizing**（适度裁剪），在保障目标性能的前提下最大限度降低制造成本与功耗，同时保留运行另一阶段的灵活性以应对Workload变化。

---
**实现原理与设计流程**
- **资源利用率分析**：通过**LLMCompass**模拟器量化分析各阶段对硬件资源的敏感度。例如，将H100的Memory Bandwidth降低40%，Prefill延迟仅增加17%；将计算核心减半，Decode延迟仅增加22%。
- **成本感知架构探索**：在满足SLO的前提下，评估各组件（Systolic Array, Vector Unit, L1/L2 Cache, Memory）对性能与成本/TDP的边际效益。
- **跨阶段灵活性约束**：在针对某一阶段进行硬件裁剪时，必须确保该芯片仍能以可接受的效率运行另一阶段，避免Workload改变时产生死锁或严重性能倒退。

---
**Prefill Chip 专用设计**
针对计算密集型的**Prefill**阶段，设计目标是最大化Tensor计算能力并降低昂贵的Memory成本。

![](images/096907e9c4cc5b3038ce0f6b336623fd333b7ff3b377bd646e1cbc3e5073bcb5.jpg)

- **Memory子系统**：
  - 采用**GDDR7**替代昂贵的**HBM3**，提供**2048 GB/s**带宽与**64 GB**容量。
  - 原理：Prefill阶段计算受限，对带宽需求较低；且KV Cache仅需短暂存储后即可转移至Decode节点，对容量要求较小。
  - 效果：Memory成本降低3倍，整体硬件成本降低**52%**。
- **Compute子系统**：
  - 增大**Systolic Array**至**32x32**以加速计算受限的**Matmul**操作。
  - 减半**Vector Width**至16，因为非Tensor操作（如Layer Normalization）是内存受限的，对算力要求低。
  - 减小**L2 Cache**至32MB，因为LLM推理中L2收益递减，30MB已足够。
  - 效果：在降低成本的同时，平均Prefill性能提升**8%**。

---
**Decode Chip 专用设计**
针对内存带宽受限的**Decode**阶段，设计目标是保留高带宽与大容量Memory，削减冗余的Compute资源。

![](images/1ecab74d3c44ed05b594d544ef471596ec45a6ee4e0afd937aa6e2baa38b8c7f.jpg)

- **Memory子系统**：
  - 保留**HBM3**以提供**3352 GB/s**带宽与**80 GB**容量。
  - 原理：Decode阶段需不断加载模型权重与持续增长的**KV Cache**，对带宽与容量极度敏感。
- **Compute子系统**：
  - 缩小**Systolic Array**至**16x16**，**Vector Width**至8。
  - 原理：Decode阶段算术强度极低，大型计算单元无法被充分利用。
  - 减小**L1 Cache**至128KB，**L2 Cache**至30MB。
  - 原理：Decode阶段的内存访问多为流式读取，缺乏数据复用，大Cache收益甚微。
  - 效果：Die Area缩小**36%**，TDP降低**28%**，同时维持**97%**的Decode性能。

---
**参数设置与规格对比**
| Specifications | Prefill Chip | Decode Chip | H100 |
| :--- | :--- | :--- | :--- |
| **Systolic Array** | 32x32 | 16x16 | Eq. to 16x32 |
| **Vector Width** | 16 | 8 | Eq. to 32 |
| **L1/L2 Cache** | 320KB / 32MB | 128KB / 30MB | 256KB / 50MB |
| **Memory Protocol** | GDDR7 | HBM3 | HBM3 |
| **Memory Bandwidth** | 2048 GB/s | 3352 GB/s | 3352 GB/s |
| **Memory Capacity** | 64 GB | 80 GB | 80 GB |
| **FP16 Tensor PFLOPs** | 1.92 | 0.54 | 0.99 |
| **Est. Die Area (4nm)** | 784 mm² | 520 mm² | 814 mm² |
| **Est. Norm. HW Cost** | 0.48 | 0.88 | 1.0 |
| **Est. TDP** | 596 W | 507 W | 700 W |

---
**输入输出关系与整体作用**
- **输入**：
  - **Prefill Chip**：接收用户的Input Prompt Token序列。
  - **Decode Chip**：接收Prefill阶段生成的KV Cache与先前生成的Token。
- **输出**：
  - **Prefill Chip**：输出首个Token与完整的KV Cache。
  - **Decode Chip**：逐个输出后续生成的Token。
- **在整体中的作用**：
  - SPAD系统通过部署异构的Prefill与Decode Machine组成 disaggregated cluster。
  - 在正常Workload下，相比同构H100集群，SPAD可降低**19%-41%**硬件成本与**2%-17%** TDP。
  - 当模型或Workload发生偏移时（如从Coding转向Conversation），SPAD支持**Adaptive Reallocation**，将Prefill Machine逻辑切换为Decode运行（或反之），凭借跨阶段运行能力，依然能维持**11%-43%**的成本优势，保障了硬件设计的Longevity。

### 2. 跨阶段自适应硬件重分配机制

**核心观点**

**跨阶段自适应硬件重分配机制**是 SPAD 系统保证集群在多生命周期内具备**Longevity**的核心策略。该机制允许在模型架构演进或负载特征变化时，通过逻辑层面的重新调度，将原本用于 **Prefill** 的芯片调配去运行 **Decode**，反之亦然，从而在无需物理更换硬件的前提下，持续维持**11%-43%**的硬件成本优势。

---

**实现原理与硬件基础**

该机制的可行性建立在 SPAD 独特的 **Less-is-more** 芯片设计理念之上，即专用芯片在极致优化某一阶段的同时，仍保留运行另一阶段的基本能力：

- **Prefill Chip 的反向兼容能力**：采用 **GDDR7** 替代 **HBM3** 以大幅削减成本。尽管其内存带宽较低导致运行 **Decode** 阶段时硬件效率有所下降，但由于 **GDDR** 相较于 **HBM** 具有显著的成本优势，即便降效运行，其整体成本效益依然优于传统同构 **H100** 集群。
- **Decode Chip 的反向兼容能力**：削减了 **Systolic Array** 规模与 **L1/L2 Cache** 容积以降低 **TDP** 和 Die 面积。但在设计时保留了适度的 Tensor 计算能力，确保在接管 **Prefill** 阶段时，性能不会发生断崖式下跌，避免成为系统瓶颈。
- **架构相似性**：两种芯片均由同款基础 GPU 架构裁剪而来，共享底层指令集与内存管理模型，避免了异构硬件常带来的软件栈割裂问题。

---

**算法流程与参数设置**

重分配机制并非底层硬件的自动路由，而是由集群编排器主导的逻辑调度过程，其核心流程如下：

- **初始配置**：基于目标 **Workload**（如 Coding 或 Conversation）和 **Model**（如 BLOOM-176B）的特征，通过设计空间探索确定最佳的 **Prefill** 与 **Decode** 机器比例（如 18P+7D）。
- **状态监测与预测**：集群运行期间，结合 **Load predictor**（如 **ARIMA** 或 **Meta Prophet**）实时预测未来的 **Prefill** 与 **Decode** 负载需求分布。
- **逻辑重分配触发**：当负载特征改变（如从 Coding 转向 Conversation，输出变长导致 Decode 需求激增）或模型演进（如从 MHA 转向 GQA）时，调度器动态调整机器角色。
- **参数调优**：重分配后，调度器针对新角色调整批处理大小和并行度策略。例如，当 **Decode Chip** 被用于运行 **Prefill** 时，由于计算能力较弱，系统会自动规避极大的 Batch Size 或超长序列输入，以防 Softmax 操作成为瓶颈。

---

**输入输出关系与系统作用**

- **输入**：
  - 变化的负载特征（如输入输出 Token 比例变化）。
  - 更迭的模型架构（如从 BLOOM-176B 切换至 Llama3-70B 或 DeepSeek-V2）。
  - 初始部署的 SPAD 异构集群物理拓扑。
- **输出**：
  - 重新匹配后的逻辑集群配置（新的 P+D 机器比例）。
  - 在新条件下仍能满足 **SLOs**（如 P90 TTFT, P99 TBT）的最大支持吞吐量。
  - 相较于重新采购同构 H100 集群，依然保持极低的硬件成本与 **TDP**。
- **系统作用**：打破了专用硬件“一次部署、固定场景”的局限。在模型迭代速度极快的 LLM 时代，使得数据中心已部署的硬件能够跨越单次模型生命周期，抵御负载漂移带来的效率衰减。

---

**性能表现与数据支撑**

通过端到端仿真，重分配机制在多种场景下展现了强大的适应力与成本优势。

![](images/a2412fafac6ac6f93c89ccc840b801e4e13d076eef0ee6ef4a4e6179ed3bd158.jpg)
![](images/fbf919691246dd7ea6620a51fd8c6e8646daa697b222a262a96ddd672b6aab8c.jpg)

**负载变化场景下的重分配结果**

| 初始配置集群 | 重分配后负载 | 重分配后吞吐量 | 等效 Splitwise 最小硬件需求 | (HW, TDP) 节省比例 |
| :--- | :--- | :--- | :--- | :--- |
| 18P + 7D (Coding) | Conversation | 55 req/s | 19 H100 | (23%, -7%) |
| 8P + 17D (Conversation) | Coding | 60 req/s | 21 H100 | (11%, 9%) |

**模型演进场景下的重分配结果**

| 初始配置集群 | 重分配后模型 | 重分配后吞吐量 | 等效 Splitwise 最小硬件需求 | (HW, TDP) 节省比例 |
| :--- | :--- | :--- | :--- | :--- |
| 18P + 7D (Coding) | Llama3-70B | 188 req/s | 26 H100 | (43%, 22%) |
| 8P + 17D (Conversation) | Llama3-70B | 171 req/s | 27 H100 | (31%, 29%) |
| 18P + 7D (Coding) | DeepSeek-V2 | 103 req/s | 23 H100 | (36%, 11%) |
| 8P + 17D (Conversation) | DeepSeek-V2 | 183 req/s | 24 H100 | (22%, 20%) |

- **GQA 架构优势**：Llama3-70B 采用 **Grouped-Query Attention**，提升了算术强度，更利于 **Prefill Chip** 发挥算力优势，因此成本节省比例最高可达**43%**。
- **MoE 架构挑战**：DeepSeek-V2 采用 **MoE** 架构，单 Token 激活的参数减少，降低了权重复用率与算术强度，削弱了专用芯片的极致性能，但依然维持了**22%-36%**的硬件成本优势。

---

**扩展适应策略**

为进一步增强在极端负载波动下的鲁棒性，SPAD 提出了两项辅助策略：

- **Buffer Pool 机制**：在 SPAD 异构集群旁挂载由平衡型硬件（如 H100）组成的缓冲池。当负载发生不可预知的剧烈波动时，动态划拨缓冲池资源以弥补专用芯片的效能缺口。
- **Runtime Load Predictor**：在 Orchestrator 层引入时间序列预测算法，提前感知负载变化趋势，平滑执行硬件角色的转换，避免实时调度引发的系统震荡。


---

## 4. 实验方法与实验结果

**实验设置**

**成本与功耗建模**
- **Die Area 估算**：基于 LLMCompass 模型修改，参考 H100 die photo，假设 10% 面积开销（包含空白与缺陷组件），采用 TSMC 4nm 工艺，晶圆成本设为 $20,000/300mm。
- **Memory 成本**：GDDR7 估算为 $3/GB；HBM3 基于 1:3 的 GDDR7:HBM3 成本比例，设为 $9/GB。
- **TDP 估算**：以 H100 的 700W 为基准，假设 10% TDP 开销用于 VRM 转换损耗与外围设备，单个 HBM 封装功耗 30W，GDDR7 功耗基于 4.5 pJ/bit 估算。

| 组件 | 参数假设 | 成本/功耗估算 |
|---|---|---|
| Wafer (TSMC 4nm) | $20,000 / 300mm | - |
| GDDR7 | $3 / GB | - |
| HBM3 | $9 / GB | - |
| H100 Die TDP | 700W × 90% - 30W × 5 | 480W |
| GDDR7 功耗 | 4.5 pJ/bit | - |

**端到端仿真流程**
- **调度器集成**：扩展 SplitwiseSim（实现 Disaggregation 调度）与 Vidur（实现 Co-location 调度），将 LLMCompass 作为统一的底层架构性能模型，确保不同调度器与硬件对比的公平性。
- **仿真层级**：调度器进行迭代级请求批处理并分配给机器，LLMCompass 估算每台机器完成单次迭代的时间。所有结果均为模拟而非物理实测。

![](images/e46d87129363c3cea68b520f5f6b72c7e5977551904b0e22b5c2f380c0326fea8.jpg)

**模型与工作负载**
- **Models**：
  - **BLOOM-176B**：Multi-Head Attention，FP16，Tensor Parallelism=8。
  - **Llama3-70B**：Grouped-Query Attention，FP16，Tensor Parallelism=4。
  - **DeepSeek-V2-236B**：Multi-head Latent Attention 与 MoE 架构，FP8，Expert Parallelism=8。
- **Workloads**：采用 Microsoft 开源请求 traces，覆盖两类场景：
  - **Coding**：长输入（中位数 1500 tokens），短输出（中位数 13 tokens）。
  - **Conversation**：短输入（中位数 1020 tokens），长输出（中位数 129 tokens）。
- **SLOs**：基于无批处理 H100 延迟定义相对 SLO，涵盖 P90/P99 的 TTFT 与 TBT，分为 Loose/Normal/Tight 三档。

**Baselines**
- **Sarathi**：基于同构 H100 的 Co-location 调度。
- **Splitwise-homo**：基于同构 H100 的 Disaggregation 调度。
- **Splitwise-pcap**：Prefill 使用 H100，Decode 使用 450W TDP 的降频 H100。
- **Splitwise-hetero**：Prefill 使用 H100，Decode 使用 A100。

---

**结果数据分析**

**集群配置**
- **Coding 工作负载**：SPAD 需 18 台 Prefill 与 7 台 Decode 机器。相比最佳基线，**硬件成本降低 41%**，**TDP 降低 13%**。Sarathi 因 Prefill-Decode 干扰需 36 台 H100，不适用于低延迟场景。
- **Conversation 工作负载**：SPAD 需 8 台 Prefill 与 17 台 Decode 机器。相比 Splitwise-homo，**硬件成本降低 19%**，**TDP 降低 17%**；相比 Splitwise-pcap，**硬件成本降低 31%**，**TDP 降低 2%**。

| 工作负载 | 系统配置 | 硬件需求 (8-chip machines) | Norm. HW Cost | Norm. TDP |
|---|---|---|---|---|
| Coding (70 req/s) | Sarathi (H100) | 36 | 36 | 36 |
| Coding (70 req/s) | Splitwise-homo | 25 | 25 | 25 |
| Coding (70 req/s) | SPAD (P+D) | 18 P + 7 D | 14.7 | 20.4 |
| Conversation (70 req/s)| Splitwise-homo | 23 | 23 | 23 |
| Conversation (70 req/s)| SPAD (P+D) | 8 P + 17 D | 18.7 | 19.1 |

![](images/527509ee03d7a2dfc6aed36babab036f2408783e3ee501a2408ff79667b782c7.jpg)

**不同 SLO 下的表现**
- 在 Loose 到 Tight 的 SLO 变化中，SPAD 始终保持显著优势。
- Coding 场景下，硬件节省幅度稳定在 **40%-42%**。
- Conversation 场景下，硬件节省幅度在 **15%-32%** 之间，TDP 节省幅度最高达 **21%**。

**集群重分配**
- **工作负载变化**：为 Coding 配置的集群重分配给 Conversation 时，8 台 Prefill 机器转为 Decode，仍可实现 55 req/s 吞吐，相比基线**节省 23% 硬件成本**。反之，为 Conversation 配置的集群转为 Coding 时，14 台 Decode 机器转为 Prefill，实现 60 req/s 吞吐，**节省 11% 硬件成本**与 **9% TDP**。
- **模型变化**：当模型从 BLOOM-176B 切换至 Llama3-70B 或 DeepSeek-V2 时，SPAD 依然有效。
  - 切换至 Llama3-70B：由于 GQA 提高了 arithmetic intensity，更利于 Prefill Chip，**硬件成本节省 31%-43%**。
  - 切换至 DeepSeek-V2：由于 MoE 架构降低了 per-expert 权重复用，节省幅度相对较小，但仍实现 **22%-36% 硬件成本节省**。

![](images/a2412fafac6ac6f93c89ccc840b801e4e13d076eef0ee6ef4a4e6179ed3bd158.jpg)

---

**消融与敏感性分析**

**芯片设计空间探索**
- **Prefill Chip**：
  - 增大 Systolic Array 显著提升 Prefill 性能。
  - 减小 Vector Unit 对性能影响极小，因为非 Tensor 操作受限于 Memory Bandwidth。
  - L2 Cache 缩减至 32MB 足矣，继续增大边际收益递减。
- **Decode Chip**：
  - 减半 Core Count 仅导致 2% 延迟增加，证实 Decode 阶段计算资源利用率低。
  - 采用 16x16 Systolic Array 与 Width 8 Vector Unit 即可高效运行，进一步减小会严重影响运行 Prefill 的灵活性。
  - L1/L2 Cache 缩减对 Memory-bound 的 Decode 性能无显著负面影响。

**性能边界与瓶颈转移**
- **Prefill 阶段**：
  - 极短序列（≤256 tokens）或极长序列（≥12288 tokens）时，Prefill Chip 可能慢于 H100。极短序列缺乏权重复用，极长序列导致 Softmax 二次复杂度凸显，受限于较低的 Memory Bandwidth。
- **Decode 阶段**：
  - 极大 Batch Size（≥256）时，arithmetic intensity 增加，Decode Chip 性能可能不及 H100，但生产环境中受限于 HBM 容量与延迟要求，此情况较少见。

![](images/707e7eb70cb87080e13e0c018ee602a995e68c31862d53c26b9ba928aea63de5.jpg)

![](images/74a74cffb27171f912b4fdc8e09bde4bc714c47e640e5c51cd2054ee4c0afc2c.jpg)

**并行策略敏感性**
- 在不同 Tensor Parallelism (TP) 与 Pipeline Parallelism (PP) 组合下，SPAD 提出的芯片性能表现一致，未因并行策略改变而出现显著性能退化。

**HBM 成本敏感性**
- 即使将 HBM 成本假设从 $6/GB 提升至 $12/GB，Decode Chip 的总成本（$1147）仍低于 H100（$1275），证明 SPAD 的成本优势在 HBM 价格波动下依然稳健。

| HBM Cost Assumptions | $6/GB | $9/GB | $12/GB |
|---|---|---|---|
| Estimated Decode Chip Cost | $667 | $907 | $1147 |
| Estimated H100 Cost | $795 | $1035 | $1275 |

---

