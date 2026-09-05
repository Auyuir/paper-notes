# SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference 图表详解

### e27b6f5d5b75ce57e1a1c75575f50f7ca4df6a13e8d9d404e864c707b7ee0206.jpg

![e27b6f5d5b75ce57e1a1c75575f50f7ca4df6a13e8d9d404e864c707b7ee0206.jpg](images/e27b6f5d5b75ce57e1a1c75575f50f7ca4df6a13e8d9d404e864c707b7ee0206.jpg)

- **图表核心主题**：该图表直观对比了主流 LLM Serving 硬件的 **Tensor Performance** 与 **Memory Bandwidth**，并揭示了 **Prefill** 与 **Decode** 阶段截然不同的计算特性需求，从而论证了 SPAD 定制化芯片设计的合理性。

- **坐标轴与参考线分析**：
  - **X轴 (Memory Bandwidth)**：衡量硬件的内存带宽吞吐能力，单位为 TB/s。
  - **Y轴 (Tensor Performance)**：衡量硬件的张量计算峰值性能，单位为 PFLOPs。
  - **Prefill Arithmetic Intensity (红色点线)**：斜率陡峭，表明 Prefill 阶段具有极高的计算强度，属于典型的 **Compute-bound**（计算受限）任务，对 **Tensor Performance** 需求极高。
  - **Decode Arithmetic Intensity (红色长划线)**：斜率平缓，表明 Decode 阶段计算强度极低，属于典型的 **Memory-bound**（内存受限）任务，对 **Memory Bandwidth** 需求极高。

- **现有硬件分布特征**：
  - 现有主流数据中心硬件（如 **H100**, **B200**, **MI300X**, **TPUv6e**）普遍遵循“越多越好”（**more-is-better**）的设计理念，试图在单一芯片上同时堆叠极高的计算与内存资源。
  - 这种通用设计导致严重的**资源错配**：在 Prefill 阶段，昂贵的 **Memory Bandwidth** 被严重闲置；在 Decode 阶段，庞大的 **Tensor Performance** 无法被充分利用。
  - **Groq** 采用极端设计，提供高达 80 TB/s 的 **Memory Bandwidth**，但 **Tensor Performance** 极低，仅适用于极小批量的 Decode 场景。

- **SPAD 定制化芯片定位**：
  - **Prefill Chip (粉色方块)**：精准落在 **Prefill Arithmetic Intensity** 虚线附近。通过**大幅削减 Memory Bandwidth**（降至约 2 TB/s）并**成倍提升 Tensor Performance**（升至约 1.9 PFLOPs），完美契合 Prefill 阶段的计算密集型特征。
  - **Decode Chip (灰色三角形)**：精准落在 **Decode Arithmetic Intensity** 虚线附近。通过**维持高 Memory Bandwidth**（保持约 3.35 TB/s）并**显著降低 Tensor Performance**（降至约 0.5 PFLOPs），完美契合 Decode 阶段的内存密集型特征。

- **硬件参数与特性对比**：

| 硬件型号 | Memory Bandwidth (TB/s) | Tensor Performance (PFLOPs) | 设计哲学与阶段匹配度 |
| :--- | :---: | :---: | :--- |
| **A100** | ~2.0 | ~0.3 | 通用设计，资源利用率低 |
| **TPUv5p** | ~2.5 | ~0.4 | 通用设计，资源利用率低 |
| **TPUv6e** | ~1.5 | ~0.9 | 偏向计算，但仍无法精准匹配 |
| **H100** | ~3.35 | ~1.0 | 通用旗舰，Prefill/Decode 均存在资源闲置 |
| **MI300X** | ~5.3 | ~1.3 | 堆料设计，成本高昂且利用率不足 |
| **B200** | ~8.0 | ~2.5 | 极致堆料，进一步加剧资源错配与成本浪费 |
| **Groq** | ~80.0 | ~0.2 | 极端内存带宽，牺牲计算能力 |
| **SPAD Prefill Chip** | **~2.0** | **~1.9** | **Less-is-more，精准匹配 Prefill 计算需求** |
| **SPAD Decode Chip** | **~3.35** | **~0.5** | **Less-is-more，精准匹配 Decode 内存需求** |

- **核心结论**：
  - 现有通用 GPU/TPU 无法同时高效满足 LLM 推理两个阶段的差异化需求，导致**硬件成本与功耗的严重浪费**。
  - SPAD 采用 **less-is-more** 方法论，通过解耦并定制化设计 **Prefill Chip** 与 **Decode Chip**，实现了硬件资源与 **Arithmetic Intensity** 的精准对齐，从而在**不牺牲性能的前提下大幅降低硬件成本与 TDP**。

### ad5f0fb2e9545af19fe7b1025b5d0727de124cbd24adc5db622e00f9328f0fab.jpg

![ad5f0fb2e9545af19fe7b1025b5d0727de124cbd24adc5db622e00f9328f0fab.jpg](images/ad5f0fb2e9545af19fe7b1025b5d0727de124cbd24adc5db622e00f9328f0fab.jpg)

* **图片基本信息**
  * **图表主题**：Simulated Prefill Latency Under Varying Memory Bandwidths（不同内存带宽下的模拟 Prefill 延迟）。
  * **X轴指标**：Memory Bandwidth (GB/s)，测试节点包含 1000、1500、2000、2500、3352（**H100** 基准）、4000。
  * **Y轴指标**：Prefill Latency (s)，数值区间为 0.0 至 0.4。
  * **图例构成 (Latency Breakdown)**：Matmul、Softmax、LayerNorm、Activation、AllReduce、Others。

* **核心数据对比**
  * 整体 Prefill Latency 随 Memory Bandwidth 提升而下降，但边际收益递减。
  * 各带宽节点下的相对延迟表现如下：

| Memory Bandwidth (GB/s) | 相对 H100 延迟倍数 | 绝对延迟估值 (s) |
| :--- | :--- | :--- |
| 1000 | **1.59x** | ~0.33 |
| 1500 | **1.32x** | ~0.28 |
| 2000 | **1.17x** | ~0.24 |
| 2500 | **1.08x** | ~0.22 |
| 3352 (H100) | **1.00x** | ~0.21 |
| 4000 | **0.97x** | ~0.20 |

* **延迟细分 (Latency Breakdown) 剖析**
  * **Matmul 绝对主导**：**Matmul**（蓝色区块）占据总延迟的 60% 以上，且其绝对耗时在不同带宽下几乎保持水平，证明其对 Memory Bandwidth 极不敏感。
  * **非 Tensor 操作敏感**：**LayerNorm** 与 **Softmax**（绿色与橙色区块）在低带宽下膨胀明显，随带宽增加迅速收缩，属于典型的 memory-bound 操作。
  * **通信开销稳定**：**AllReduce**（紫色区块）高度基本恒定，不受单卡内存带宽调整的直接干扰。

* **核心结论与硬件启示**
  * **Compute-bound 本质**：Prefill 阶段的核心瓶颈在于计算而非内存吞吐，呈现强烈的 **compute-bound** 特征。
  * **HBM 带宽冗余**：将 **H100** 的内存带宽削减 25%（至 2500 GB/s），延迟仅恶化 8%；削减 40%（至 2000 GB/s），延迟仅增加 17%。
  * **Less-is-More 设计依据**：为 Prefill 阶段配置超高带宽的昂贵 HBM 会导致严重的资源闲置。采用低成本内存（如 GDDR）并扩大计算阵列（Systolic Arrays）是提升成本效益的正确路径。

### 115d3f191374a72c58b81ef6929c44298b1a836f0ad0b03efab085a81222d179.jpg

![115d3f191374a72c58b81ef6929c44298b1a836f0ad0b03efab085a81222d179.jpg](images/115d3f191374a72c58b81ef6929c44298b1a836f0ad0b03efab085a81222d179.jpg)

- **图表基本信息**
  - **Y轴**：**Decode Latency (s)**，表示大模型推理中 Decode 阶段的延迟时间。
  - **X轴**：**Core Count**，表示计算核心数量（对应 NVIDIA GPU 的 SM count），取值范围从 44 到 160。
  - **图例 (Latency Breakdown)**：延迟时间被分解为 **Matmul**、Softmax、LayerNorm、Activation、AllReduce 和 Others 六个部分。
  - **实验配置**：使用 **LLMCompass** 进行模拟，目标模型为 BLOOM-176B (FP16)，batch size 为 64，sequence length 为 1024，tensor parallelism 为 8。

- **核心数据提取**
  | Core Count | 相对延迟倍数 | 绝对延迟估值 (s) | 状态说明 |
  | :--- | :--- | :--- | :--- |
  | 44 | 1.40x | ~0.058 | 核心数最少，延迟最高 |
  | 66 | 1.22x | ~0.051 | 延迟随核心数增加明显下降 |
  | 108 | 1.02x | ~0.043 | 核心数比 H100 少约 20% |
  | 132 | 1.00x | ~0.042 | **H100 基准配置** (红色箭头标注) |
  | 144 | 0.99x | ~0.041 | 性能提升进入瓶颈期 |
  | 160 | 0.99x | ~0.041 | 核心数最多，延迟最低 |

- **趋势与结论分析**
  - **延迟主导因素**：**Matmul** 操作占据了 Decode Latency 的绝对主导地位，其他非张量操作（如 Softmax、LayerNorm）占比较小。
  - **亚线性扩展 (Sub-linear Scaling)**：Decode 性能随 Core Count 的增加呈现**亚线性扩展**。当核心数从 132 (H100) 提升至 160 时，延迟几乎停滞在 0.99x，表明增加计算核心无法带来相应的性能收益。
  - **计算资源严重冗余**：对比 108 核心与 132 核心 (H100)，计算核心削减近 **20%**，但 Decode 延迟仅增加约 **2%** (1.02x vs 1.00x)。这直接证明了在 Decode 阶段，传统 GPU (如 H100) 庞大的计算容量处于**低效利用 (Underutilized)** 状态。
  - **架构设计启示**：该图表为论文提出的 **less-is-more** 设计理念提供了核心数据支撑。它表明在 Decode 阶段，可以通过大幅削减计算容量（如采用更小的 systolic arrays），在几乎不牺牲性能的前提下，显著降低硬件制造成本与 **TDP**。

### Figure 4. Proposed SPAD Cluster and Chips Overview. Die area is estimated and will be further explained in Section 6.1.

![55214a0d189e14452641e603d05d8c0aaf2b0155758754373efccec00c558086.jpg](images/55214a0d189e14452641e603d05d8c0aaf2b0155758754373efccec00c558086.jpg)

- **SPAD Cluster** 整体架构由异构的 **Prefill Machine** 和 **Decode Machine** 组成，旨在通过硬件 specialization 优化 LLM 推理的 prefill 和 decode 阶段。
- 集群内部采用两级互联机制：
  - **Scale-Up Interconnect**：用于连接同一 Machine 内的多个 Chip（如 NVLink），提供高带宽片间通信。
  - **Scale-Out Interconnect**：用于连接不同的 Machine（如 Infiniband），支持跨节点的 KV cache 传输。

- **Prefill Chip** 专为 compute-bound 的 prefill 阶段设计，核心策略是增大 tensor compute 并采用 cost-effective 的 **GDDR7** 内存。
  - **Core 架构**：包含 128 个 **Prefill Core**，每个 Core 配备 16-wide **Vector Unit**、32x32 **Systolic Array** (x4) 以及 320KB **L1 Cache**。
  - **内存子系统**：集成 64GB **GDDR7**，提供 512-bit 总线宽度和 2TB/s 内存带宽。
  - **Die Area 估算**：

| 组件名称 | 规格/描述 | 估算面积 |
| :--- | :--- | :--- |
| **Prefill Cores** | 128 Cores in Total | 422 mm² |
| **L2 Cache & Crossbar** | 32MB | 111 mm² |
| **GDDR7 Interface** | 32-bit×16 in Total | 106 mm² |
| **PCIe Interface** | x32 | 39 mm² |
| **Interconnect Interface** | 900GB/s | 28 mm² |

- **Decode Chip** 专为 memory-bound 的 decode 阶段设计，核心策略是保留高内存带宽并削减冗余 compute 资源以降低 TDP 和成本。
  - **Core 架构**：包含 144 个 **Decode Core**，每个 Core 配备 8-wide **Vector Unit**、16x16 **Systolic Array** (x4) 以及 128KB **L1 Cache**。
  - **内存子系统**：集成 80GB **HBM3**，提供 5120-bit 总线宽度和 3.3TB/s 内存带宽。
  - **Die Area 估算**：

| 组件名称 | 规格/描述 | 估算面积 |
| :--- | :--- | :--- |
| **Decode Cores** | 144 Cores in Total | 206 mm² |
| **L2 Cache & Crossbar** | 30MB | 107 mm² |
| **HBM3 Interface** | 1024-bit×5 in Total | 88 mm² |
| **PCIe Interface** | x32 | 39 mm² |
| **Interconnect Interface** | 900GB/s | 14 mm² |

- **设计对比总结**：
  - **Prefill Chip** 牺牲了部分内存带宽（**GDDR7** vs **HBM3**），换取了更大的 **Systolic Array** 和更低的内存成本。
  - **Decode Chip** 保留了 **HBM3** 的高带宽，但大幅缩小了 **Systolic Array** 尺寸和 **L1/L2 Cache** 容量，以优化 die area 和功耗。

### 096907e9c4cc5b3038ce0f6b336623fd333b7ff3b377bd646e1cbc3e5073bcb5.jpg

![096907e9c4cc5b3038ce0f6b336623fd333b7ff3b377bd646e1cbc3e5073bcb5.jpg](images/096907e9c4cc5b3038ce0f6b336623fd333b7ff3b377bd646e1cbc3e5073bcb5.jpg)

- **图表概述**：该散点图展示了 **Prefill Chip** 设计空间探索中，不同硬件架构配置下的 **Die Area** 与 **Prefill Latency** 的权衡关系。
- **坐标轴与图例映射**：
  - **X轴**：**Die Area (mm²)**，代表芯片物理面积与制造成本。
  - **Y轴**：**Prefill Latency (s)**，代表预填充阶段延迟，数值越低性能越好。
  - **形状图例**：代表 **Systolic Array** 尺寸，包含 16x32（圆形）、24x32（三角形）和 32x32（正方形）。
  - **颜色图例**：代表 **L1 Size (KB)**，色阶从 192 KB（黄色）渐变至 512 KB（深紫色）。
- **关键配置对比**：

| 硬件节点 | Die Area (mm²) | Prefill Latency (s) | Systolic Array | 设计特征 |
| :--- | :--- | :--- | :--- | :--- |
| **H100** (基准) | ~814 | ~0.21 | 16x32 (等效) | 面积较大，延迟较高 |
| **Prefill Chip** (提出) | ~784 | ~0.195 | 32x32 | 面积更小，延迟更低 |

- **核心结论分析**：
  - **Systolic Array 主导性能**：图中 32x32 配置（正方形）在同等 **Die Area** 下普遍具备更低的 **Prefill Latency**，证实增大张量计算单元可显著提升预填充性能。
  - **L1 Cache 协同优化**：深色数据点（大 **L1 Size**）多集中于低延迟区域，表明增大 L1 缓存能有效配合大尺寸 **Systolic Array** 提升数据复用率。
  - **Less-is-More 方法论验证**：**Prefill Chip** 通过裁剪非核心计算单元，将 **Die Area** 压缩至 **784 mm²**（优于 H100 的 814 mm²），同时凭借 32x32 **Systolic Array** 将延迟降至 **0.195s**，实现了面积与性能的双重优化。

### ce83eaf0291afccb814134aaaa9f82dc13b2daf2a112ce2ca613732ee06cc481.jpg

![ce83eaf0291afccb814134aaaa9f82dc13b2daf2a112ce2ca613732ee06cc481.jpg](images/ce83eaf0291afccb814134aaaa9f82dc13b2daf2a112ce2ca613732ee06cc481.jpg)

- **图表概述**：该散点图展示了 **Prefill Chip** 的设计空间探索结果，横轴为 **Die Area (mm²)**，纵轴为 **Prefill Latency (s)**。
- **图例参数**：
  - **Vector Width** 通过形状区分：圆圈代表 8，三角形代表 16，正方形代表 32。
  - **L2 Size (MB)** 通过颜色区分：从黄色（24 MB）渐变至深紫色（72 MB）。
- **核心数据对比**：

| 硬件配置 | Die Area (mm²) | Prefill Latency (s) | 视觉标记 |
| :--- | :--- | :--- | :--- |
| **H100** | ~814 | ~0.215 | 橙色五角星 |
| **Prefill Chip** | ~784 | ~0.195 | 橙色三角形 |

- **趋势与洞察**：
  - 整体趋势表明，增加 **Die Area** 通常可降低 **Prefill Latency**，但存在边际效益递减现象。
  - 较大的 **Vector Width** 和 **L2 Size** 配置通常对应更大的 **Die Area**。
  - **Prefill Chip** 成功打破了常规面积与延迟的权衡曲线，在 **Die Area** 比 **H100** 减少约 3.7% 的前提下，实现了约 9.3% 的 **Prefill Latency** 降低。
- **设计验证**：
  - 图表直观验证了 **less-is-more** 设计方法论的有效性。
  - 通过精准裁剪非核心计算资源（如调整 **Vector Width** 和 **L2 Size**），**Prefill Chip** 在缩减芯片面积、降低硬件成本的同时，反而提升了 **Prefill** 阶段的执行效率。

### 1ecab74d3c44ed05b594d544ef471596ec45a6ee4e0afd937aa6e2baa38b8c7f.jpg

![1ecab74d3c44ed05b594d544ef471596ec45a6ee4e0afd937aa6e2baa38b8c7f.jpg](images/1ecab74d3c44ed05b594d544ef471596ec45a6ee4e0afd937aa6e2baa38b8c7f.jpg)

- **图表基本信息**
  - **X轴**：**Die Area (mm²)**，衡量芯片的物理面积与潜在制造成本，范围覆盖 400 至 800+ mm²。
  - **Y轴**：**Decode Latency (s)**，衡量解码阶段的延迟性能，范围集中在 0.044 s 至 0.050 s。
  - **图例参数**：
    - **Systolic Array**：通过形状区分，包含 16x8（圆形）、16x16（三角形）、16x32（正方形）。
    - **L1 Size (KB)**：通过颜色渐变区分，范围从 64 KB（黄色）至 256 KB（紫色）。

- **数据分布与趋势分析**
  - 数据点广泛散布，表明不同的 **Systolic Array** 和 **L1 Size** 组合对 **Die Area** 影响显著。
  - 增大 **Systolic Array** 或 **L1 Size** 会显著增加 **Die Area**，但对降低 **Decode Latency** 的收益呈现明显的边际递减效应。
  - 多数配置在 0.045 s 至 0.050 s 的延迟区间内，说明 decode 阶段受限于 memory bandwidth，单纯堆叠 compute 资源无法线性提升性能。

- **关键设计点对比**

| 设计点 | Systolic Array | L1 Size (约) | Die Area (mm²) | Decode Latency (s) |
| :--- | :--- | :--- | :--- | :--- |
| **Decode Chip** | 16x16 | 128 KB | ~520 | ~0.045 |
| **H100** | 16x32 (等效) | 256 KB | ~814 | ~0.0445 |

- **设计空间探索结论**
  - **Decode Chip** 的设计点（橙色三角形）在仅牺牲极小 **Decode Latency**（约 0.0005 s）的前提下，将 **Die Area** 从 **H100** 的 814 mm² 大幅缩减至约 520 mm²。
  - 该结果直接验证了论文的 **less-is-more** 设计理念：针对 memory-bound 的 decode 阶段，削减冗余的 compute capacity 和 cache size 能够显著降低硬件成本，同时维持极具竞争力的性能表现。

### a30f5a80d4f3a5242a315d98cba91a49970e0b700b9ee28e8ea5684a207638d6.jpg

![a30f5a80d4f3a5242a315d98cba91a49970e0b700b9ee28e8ea5684a207638d6.jpg](images/a30f5a80d4f3a5242a315d98cba91a49970e0b700b9ee28e8ea5684a207638d6.jpg)

- **图表概述**：该图展示了 **Decode Chip** 的设计空间探索（Design Space Exploration），旨在评估不同硬件架构配置下 **Decode Latency** 与 **Die Area** 之间的权衡关系，验证“less-is-more”设计理念在解码阶段的有效性。

- **坐标轴与图例解析**：
  - **X轴（Die Area）**：表示芯片的物理面积，范围从 400 mm² 到 800 mm² 以上。
  - **Y轴（Decode Latency）**：表示解码阶段的延迟时间，范围在 0.044s 到 0.050s 之间。
  - **形状图例（Vector Width）**：通过不同形状区分向量宽度，圆圈代表 8，三角形代表 16，正方形代表 32。
  - **颜色图例（L2 Size）**：通过颜色渐变表示 L2 缓存大小，从 24 MB（黄色）到 48 MB（紫色）。

- **关键设计点对比**：
  - **H100（基线）**：位于图表右下角（橙色五角星），拥有最大的 **Die Area**（约 814 mm²）和最低的 **Decode Latency**（约 0.0445s），代表了传统“more-is-better”的设计哲学。
  - **Decode Chip（优化设计）**：位于图表中左侧（绿色大圆圈），**Die Area** 大幅缩减至约 520 mm²，而 **Decode Latency** 仅微增至约 0.045s，实现了极高的面积效率。

- **设计空间探索结论**：
  - **性能与面积的边际效应**：当 **Die Area** 从 800 mm² 缩减至 500 mm² 区间时，**Decode Latency** 的增加极其平缓，表明解码阶段对计算和缓存资源的利用率存在瓶颈，过度堆砌硬件资源无法带来显著的性能提升。
  - **最优配置特征**：在较小面积（500-600 mm²）下，采用较小的 **Vector Width**（如 8）和适中的 **L2 Size**（如 24-32 MB）能够维持极低的延迟，这为 **Decode Chip** 削减非核心计算单元和缓存提供了数据支撑。

- **核心数据对比表**：

| 硬件设计 | Die Area (mm²) | Decode Latency (s) | Vector Width | L2 Size (MB) | 设计理念 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H100** | ~814 | ~0.0445 | 32 (推测) | 48 | more-is-better |
| **Decode Chip** | ~520 | ~0.0450 | 8 | ~30 | less-is-more |

### 707e7eb70cb87080e13e0c018ee602a995e68c31862d53c26b9ba928aea63de5.jpg

![707e7eb70cb87080e13e0c018ee602a995e68c31862d53c26b9ba928aea63de5.jpg](images/707e7eb70cb87080e13e0c018ee602a995e68c31862d53c26b9ba928aea63de5.jpg)

- **图片基本信息**：该图为论文中的 **Figure 7(a)**，展示了所提出的 **Prefill Chip** 在不同 **Batch Size** 和 **Sequence Length** 下的 **Prefill Latency**（归一化至模拟的 **H100** 延迟）。
- **坐标轴与图例**：X轴代表 **Sequence Length**（64至16384），Y轴代表 **Batch Size**（1至256）。右侧颜色条表示归一化延迟值（0.85至1.20），**数值越低（偏黄）代表性能越优于 H100，数值越高（偏紫）代表性能越差**。
- **核心性能趋势**：
  - **整体性能优势**：在绝大多数中等规模的 **Batch Size** 和 **Sequence Length** 组合下，热力图呈现黄绿色，延迟比值稳定在 **0.9** 左右。这表明得益于更大的 **systolic arrays**，**Prefill Chip** 的张量操作更快，平均 **Prefill** 性能比 **H100** 提升 **8%**。
  - **左下角性能瓶颈（短序列/小Batch）**：当 **Batch Size** 极小（1-2）且 **Sequence Length** 极短（64-256）时，颜色变为深紫色，延迟比值高达 **1.2**。这是因为极短序列的权重复用率低、计算强度低，**Prefill Chip** 采用的 **GDDR** 内存带宽劣势被放大。
  - **右下角性能瓶颈（长序列）**：当 **Sequence Length** 达到 **12288** 及以上时，颜色转为蓝绿色，延迟比值升至 **1.0 - 1.1**。这是因为注意力机制中的 **Softmax** 操作复杂度随序列长度呈二次方增长，**Prefill Chip** 削减的非张量计算能力和内存带宽使其成为新的瓶颈。
- **关键区域数据总结**：

| 区域特征 | Batch Size | Sequence Length | 归一化延迟 | 性能表现与底层原因 |
| :--- | :--- | :--- | :--- | :--- |
| **主流工作负载** | 中等 (如 8-64) | 中等 (如 512-8192) | **~0.9** | **优于 H100**。大尺寸 **systolic arrays** 加速了计算密集型矩阵乘法。 |
| **极短序列/小Batch** | 极小 (1-2) | 极短 (64-256) | **1.2** | **劣于 H100**。低计算强度下，**GDDR** 内存带宽不足成为瓶颈。 |
| **极长序列** | 任意 | 极长 (≥12288) | **1.0 - 1.1** | **持平或略劣**。**Softmax** 二次复杂度导致非张量计算和内存带宽受限。 |

- **架构优化建议**：针对极长序列导致的 **Softmax** 瓶颈，论文建议在实际部署中采用 **chunking long prefills**（分块预填充）或 **sequence parallelism**（序列并行）技术来缓解性能下降。

### 9ef9faa2d23f79593418c9978a526c473be966249a3baae2a0e4a6703b927339.jpg

![9ef9faa2d23f79593418c9978a526c473be966249a3baae2a0e4a6703b927339.jpg](images/9ef9faa2d23f79593418c9978a526c473be966249a3baae2a0e4a6703b927339.jpg)

该图片为论文中的 **Figure 7(b)**，展示了 **Prefill Chip** 在执行 **Decode** 阶段时的延迟与模拟 **H100** 延迟的归一化比值（**Decode Latency of the Prefill Chip (Norm. to Simulated H100)**）。

* **图表类型**：二维热力图（Heatmap）。
* **横轴（X-axis）**：**Sequence Length**（序列长度），取值范围从 64 到 16384。
* **纵轴（Y-axis）**：**Batch Size**（批处理大小），取值范围从 1 到 128。
* **颜色映射（Colorbar）**：表示归一化延迟比值，数值范围从 **1.20**（黄色）到 **1.28 以上**（深紫色）。

**数据分布与趋势分析**：
* **整体表现**：图中所有单元格的数值均介于 **1.2** 到 **1.3** 之间，表明 **Prefill Chip** 在运行 **Decode** 任务时，其延迟比 **H100** 高出 **20% 到 30%**。
* **低负载区域**：在 **Batch Size** 较小（如 1-4）且 **Sequence Length** 较短的区域，延迟比值稳定在 **1.2** 左右。
* **高负载区域**：随着 **Batch Size** 和 **Sequence Length** 的增加（向图表右上角移动），延迟比值逐渐上升至 **1.3**，并在 **Batch Size** 较大（如 32-128）且 **Sequence Length** 较长（如 512-16384）的区域达到峰值（深紫色，**>1.28**）。

**关键数据分布表**：

| Batch Size 范围 | Sequence Length 范围 | 归一化延迟比值 (Norm. Latency) | 视觉颜色指示 |
| :--- | :--- | :--- | :--- |
| 1 - 4 | 64 - 16384 | 1.2 | 黄色 / 浅绿 |
| 8 - 16 | 64 - 512 | 1.2 | 绿色 |
| 8 - 16 | 1024 - 16384 | 1.3 | 青色 / 蓝色 |
| 32 - 128 | 64 - 256 | 1.2 | 绿色 |
| 32 - 128 | 512 - 16384 | 1.3 | 深蓝 / 深紫 |

**核心结论与论文关联**：
* **内存带宽瓶颈**：**Prefill Chip** 采用了成本更低的 **GDDR7** 内存（带宽 2048 GB/s），而 **Decode** 阶段是典型的 **memory-bound**（内存带宽受限）任务。因此，**Prefill Chip** 在执行 **Decode** 时无法达到 **H100**（带宽 3352 GB/s）的性能水平。
* **设计权衡**：尽管 **Prefill Chip** 运行 **Decode** 时存在性能折损，但其硬件成本大幅降低（降低 52%）。在 **SPAD** 集群的自适应重分配（**Adaptive Reallocation**）策略中，这种性能牺牲是可接受的，因为其主要设计目标仍是优化 **Prefill** 阶段，同时保留了运行 **Decode** 的基础能力以应对负载变化。

### 8d3e607f5f0ca6eb14493aee4029397ae4f3282121b9438cf82445bed69e68f5.jpg

![8d3e607f5f0ca6eb14493aee4029397ae4f3282121b9438cf82445bed69e68f5.jpg](images/8d3e607f5f0ca6eb14493aee4029397ae4f3282121b9438cf82445bed69e68f5.jpg)

- **图表概述**：该热力图展示了 **Decode Chip** 在执行 **Prefill** 阶段任务时，其延迟相对于基准 **H100** 的归一化表现（对应论文 Figure 7(c)）。横轴为 **Sequence Length**（64 至 16384），纵轴为 **Batch Size**（1 至 128），颜色映射代表归一化延迟倍数（1.0 至 1.5）。
- **数据趋势分析**：
  - **低负载性能持平**：在极小的 **Batch Size**（如 1）和较短的 **Sequence Length**（64、128）下，归一化延迟为 **1.0**，表明此时 **Decode Chip** 与 **H100** 性能一致。
  - **高负载延迟攀升**：随着 **Batch Size** 和 **Sequence Length** 的增加，归一化延迟迅速上升并稳定在 **1.5**，意味着在计算密集型场景下，**Decode Chip** 的延迟比 **H100** 高出 **50%**。
  - **瓶颈快速显现**：当 **Sequence Length** 超过 256 且 **Batch Size** 大于 1 时，延迟即达到 **1.3** 至 **1.4**，说明 **Decode Chip** 的计算资源很快成为 **Prefill** 任务的瓶颈。
- **核心结论**：
  - **计算资源削减的代价**：**Decode Chip** 采用了更小的 **Systolic Array**（16×16）和 **Vector Width**（8），导致其在处理高算术强度的 **Prefill** 任务时性能显著下降。
  - **设计权衡与灵活性**：尽管运行 **Prefill** 时延迟增加，但设计者并未进一步削减张量计算能力，以避免在 **Adaptive Reallocation**（自适应重分配）时造成不可接受的性能损失，从而保障了集群应对负载变化的灵活性。
- **关键数据提取**：

| Batch Size | Sequence Length | 归一化延迟 (Norm. Latency) | 性能表现评估 |
| :--- | :--- | :--- | :--- |
| 1 | 64, 128 | **1.0** | 与 H100 持平，无性能损失 |
| 1 | 256 | **1.3** | 延迟开始增加，计算瓶颈初显 |
| 1 | 12288, 16384 | **1.4** | 长序列导致延迟显著上升 |
| 128 | 64 至 512 | **1.5** | 大 Batch Size 导致延迟达到峰值 |
| 64 | 1024 | **1.5** | 典型生产负载下延迟增加 50% |

### 557bc47e1e7cdf5abe479700c2d9f847d0a3556f9263aaf64b45354ea495e2ee.jpg

![557bc47e1e7cdf5abe479700c2d9f847d0a3556f9263aaf64b45354ea495e2ee.jpg](images/557bc47e1e7cdf5abe479700c2d9f847d0a3556f9263aaf64b45354ea495e2ee.jpg)

* **图表主题**：该热力图展示了 **Decode Chip** 在执行 **Prefill** 阶段时的归一化延迟（Normalized Prefill Latency），性能基准为模拟的 **H100** 芯片（对应论文 Figure 7c）。
* **坐标轴与映射**：
  * **X轴**：**Sequence Length**，跨度从 64 至 16384。
  * **Y轴**：**Batch Size**，跨度从 1 至 256。
  * **颜色映射**：右侧色条代表归一化延迟倍数，区间为 **1.00**（黄/绿色）至 **1.25**（紫色），数值越接近 1.00 代表性能越优。
* **核心数据表现**：
  * **主流场景**：在绝大多数 **Batch Size** 与 **Sequence Length** 组合下，归一化延迟稳定在 **1.0**，证明 **Decode Chip** 能够以与 **H100** 相当的性能处理常规 **Prefill** 任务。
  * **性能瓶颈区**：当 **Batch Size ≥ 256** 且 **Sequence Length ≤ 256** 时，延迟显著攀升至 **1.2** 至 **1.3**。此现象源于 **Decode Chip** 削减了计算单元（如 **Systolic Array** 和 **Vector Units**），在极高并发短序列下暴露出计算瓶颈。
  * **长序列边缘区**：当 **Batch Size** 处于 4 至 8 且 **Sequence Length** 达到 8192 至 12288 时，延迟轻微上升至 **1.1**。
  * **容量截断区**：数据分布呈三角形，右下角（超长 **Sequence Length**）因超出 **HBM** 内存容量限制而未显示数据。
* **关键数据点提取**：

| Batch Size | Sequence Length | 归一化延迟 (Norm. Latency) | 性能状态评估 |
| :--- | :--- | :--- | :--- |
| **256** | **64** | **1.3** | 显著下降 (计算瓶颈凸显) |
| **256** | **128, 256** | **1.2** | 轻微下降 |
| **4 - 8** | **8192 - 12288** | **1.1** | 边缘轻微下降 |
| **1 - 128** | **常规长度** | **1.0** | 与 H100 性能持平 |

* **架构设计启示**：
  * **Decode Chip** 贯彻 **Less-is-More** 设计理念，通过降低计算能力换取更优的 **TDP** 与硬件成本。
  * 尽管在极端大 **Batch Size** 下 **Prefill** 性能受损，但实际生产中受限于 **HBM** 容量和延迟 **SLO**，**Batch Size ≥ 256** 的场景极为罕见，验证了该芯片在真实部署中的灵活性与可行性。

### Figure 8. End-to-End Simulation Setup

![e46d87129363c3cea68b5205f6b72c7e5977551904b0e22b5c2f380c0326fea8.jpg](images/e46d87129363c3cea68b5205f6b72c7e5977551904b0e22b5c2f380c0326fea8.jpg)

- **输入数据与配置模块**
  - **Workload Trace**：提供仿真所需的请求特征，包括到达时间（Arrival time）以及每个请求的输入/输出长度（input/output length）。
  - **Cluster Config**：定义底层集群的硬件规格与拓扑结构。
  - **Scheduler Choice**：指定调度策略，分为 **Co-location**（共置调度）和 **Disaggregation**（分离调度）两条路径。

- **核心调度仿真引擎**
  - **Cluster Scheduling Simulation**：作为中枢控制模块，根据调度策略路由请求：
    - **Vidur (Sarathi)**：专门处理 **Co-location** 模式下的请求批处理与调度。
    - **SplitwiseSim**：专门处理 **Disaggregation** 模式下的 Prefill 与 Decode 阶段分离调度。

- **硬件仿真与闭环反馈**
  - **Scheduling Decision**：调度引擎按迭代（Per Iteration）输出决策，明确指定哪些请求分配至哪些机器运行。
  - **HW Simulation**：核心为 **LLMCompass** 架构仿真器，接收 **HW Config** 与调度决策，计算单次迭代的执行时间。
  - **Iteration Latency**：硬件仿真器将计算出的迭代延迟反馈回 **Cluster Scheduling Simulation**，形成闭环控制。

- **性能评估输出**
  - **TTFT & TBT**：仿真流程的最终输出指标，即首词延迟（Time To First Token）和词间延迟（Time Between Tokens），用于评估系统服务质量。

| 仿真组件 | 输入/触发条件 | 核心动作/输出 |
| :--- | :--- | :--- |
| **Workload Trace** | 外部请求日志 | 提供请求到达时间与序列长度 |
| **Scheduler Choice** | 策略配置 | 路由至 **Vidur** 或 **SplitwiseSim** |
| **Cluster Scheduling Simulation** | 工作负载与集群配置 | 生成 **Scheduling Decision** |
| **LLMCompass** | **HW Config** 与调度决策 | 计算并返回 **Iteration Latency** |
| **Cluster Scheduling Simulation** | 接收 **Iteration Latency** | 最终输出 **TTFT & TBT** 指标 |

### 527509ee03d7a2dfc6aed36babab036f2408783e3ee501a2408ff79667b782c7.jpg

![527509ee03d7a2dfc6aed36babab036f2408783e3ee501a2408ff79667b782c7.jpg](images/527509ee03d7a2dfc6aed36babab036f2408783e3ee501a2408ff79667b782c7.jpg)

- **图表类型与核心功能**：该图是一张**二维热力图**，直观展示了 SPAD 集群在处理 **Coding Trace** 负载时，不同 **Prefill Machine Count** 与 **Decode Machine Count** 组合下的归一化性能指标（如延迟）分布情况。
- **坐标轴与数据映射**：
  - **X轴**：代表 **Prefill Machine Count**，数值范围约为 0 至 28。
  - **Y轴**：代表 **Decode Machine Count**，数值范围约为 0 至 28。
  - **Colorbar（颜色条）**：右侧色标显示数值区间为 **1.0 至 4.0**。深紫色代表较低的性能开销（最优），黄绿色代表较高的性能开销（最差）。
- **关键视觉元素**：
  - **SLO 边界线**：图中白色虚线明确标注为 **SLO=3**，作为满足服务等级目标（Service-Level Objective）的临界阈值。虚线左下方的深紫色区域表示满足 SLO 要求的可行配置空间。
  - **最优设计点**：紫色圆点标记为 **Optimal Design**，精准定位在满足 SLO 约束的边界内侧。
- **核心数据与配置结论**：

| 关键指标 | 具体数值 / 描述 |
| :--- | :--- |
| **Optimal Design 坐标** | Prefill Machine Count = **18**, Decode Machine Count = **7** |
| **SLO 阈值** | **3.0** (归一化延迟/性能指标) |
| **总机器数量** | **25** 台 (18 + 7) |
| **性能表现** | 位于 **SLO=3** 边界内侧，确保满足严格的延迟约束 |

- **工程与业务意义**：
  - 该图验证了 SPAD **less-is-more** 设计理念的有效性。通过异构芯片的灵活组合，系统能够在满足 **SLO=3** 的前提下，找到资源消耗最小的**最优配置（18 Prefill + 7 Decode）**。
  - 尽管总机器数量（25台）与使用同构 H100 的 **Splitwise-homo** 基线相当，但由于 Prefill 和 Decode 专用芯片的硬件成本大幅降低，该配置在保持同等性能的同时实现了**硬件成本的显著节约**。

### cd94778d93a44883f0aab5af9995fdf21e1404fcc7477da6e5e09f1ece392de7.jpg

![cd94778d93a44883f0aab5af9995fdf21e1404fcc7477da6e5e09f1ece392de7.jpg](images/cd94778d93a44883f0aab5af9995fdf21e1404fcc7477da6e5e09f1ece392de7.jpg)

- **图表概述**：该图是一张二维热力图，展示了在特定工作负载下，**Prefill Machine Count**（预填充机器数量）与 **Decode Machine Count**（解码机器数量）的不同组合对系统性能指标的影响，并标出了满足 **SLO**（服务等级目标）约束的最优资源配置。
- **坐标轴与图例解析**：
  - **X轴**：**Prefill Machine Count**，刻度范围从 4 到 28，代表集群中用于处理计算密集型预填充阶段的专用机器数量。
  - **Y轴**：**Decode Machine Count**，刻度范围从 4 到 28，代表集群中用于处理内存带宽密集型解码阶段的专用机器数量。
  - **颜色条（Colorbar）**：右侧色标范围从 1 到 8，颜色由黄色（低值）向深紫色（高值）渐变，代表某种归一化性能指标、延迟或资源消耗程度。
- **关键数据点与标记**：
  - **SLO=6 虚线**：图中一条对角虚线标注为 **SLO=6**，作为系统性能的约束边界。虚线左下方区域颜色较深（紫色），表明该区域的机器配置无法满足 SLO 要求或性能较差。
  - **Optimal Design（最优设计）**：左上角的红色圆点（坐标约为 Prefill=10, Decode=26）被明确标注为 **Optimal Design**，代表在满足 SLO 约束下的最佳资源配比。
  - **对比配置点**：右下角的红色圆点（坐标约为 Prefill=18, Decode=8）代表另一种紧贴 SLO 边界的配置方案，对应不同算术强度的工作负载。
- **数据分布与趋势分析**：
  - 热力图呈现出明显的对角线分布特征，表明 **Prefill** 和 **Decode** 机器数量之间存在显著的权衡（Trade-off）关系。
  - 增加任一阶段的机器数量均可改善系统性能（颜色变浅），但为了最小化硬件成本，最优配置必须紧贴 **SLO=6** 的边界线，避免资源冗余。
- **配置方案对比表**：

| 配置点位置 | Prefill Machine Count | Decode Machine Count | 标注/含义 | 适用场景推测 |
| :--- | :---: | :---: | :--- | :--- |
| 左上角红点 | ~10 | ~26 | **Optimal Design** | 内存/带宽密集型（如长文本对话 Conversation） |
| 右下角红点 | ~18 | ~8 | 对比配置点 | 计算密集型（如代码生成 Coding） |

- **结论与论文关联**：
  - 该图直观验证了论文中提出的 **SPAD** 异构集群设计理念：通过解耦 **Prefill** 和 **Decode** 阶段，可以根据工作负载的算术强度（Arithmetic Intensity）动态调整两类机器的比例。
  - 紧贴 **SLO** 边界的 **Optimal Design** 证明了 **less-is-more** 设计方法论的有效性，即在保证延迟约束的前提下，通过精准的资源调配（Right-sizing）实现硬件成本与功耗的最小化。

### dc5ff0b447493898ec06add094310a1724084c237b802eec41ba9b36c1db5e47.jpg

![dc5ff0b447493898ec06add094310a1724084c237b802eec41ba9b36c1db5e47.jpg](images/dc5ff0b447493898ec06add094310a1724084c237b802eec41ba9b36c1db5e47.jpg)

- **图表类型**：二维热力图（Heatmap），用于可视化 SPAD 集群在不同机器配比下的延迟表现。
- **坐标轴定义**：
  - **X轴**：**Prefill Machine Count**（预填充机器数量），刻度范围约为 0 至 28。
  - **Y轴**：**Decode Machine Count**（解码机器数量），刻度范围约为 0 至 28。
- **颜色映射（Colorbar）**：
  - 代表 **Normalized P99 TTFT**（归一化 P99 首字延迟）。
  - 数值区间为 **1.0**（深紫色，延迟最低）至 **2.8**（亮黄色，延迟最高）。颜色越深代表性能越优。
- **关键视觉标记**：
  - **Optimal Design**（紫色圆点）：定位在坐标 **(18, 7)**，代表满足所有延迟约束且硬件成本最低的最优集群配置。
  - **SLO=2**（白色虚线）：表示 P99 TTFT 延迟等于 2 的临界边界。虚线左下方的深色区域为满足 SLO 的可行配置空间，右上方浅色区域为违反 SLO 的无效配置。

| 视觉元素 | 含义与数据解析 |
| :--- | :--- |
| **X轴 / Y轴** | 探索 **Prefill** 与 **Decode** 机器的异构配比空间 |
| **颜色梯度** | 深紫 (1.0) 至 亮黄 (2.8)，反映 **P99 TTFT** 延迟劣化程度 |
| **Optimal Design** | 坐标 **(18, 7)**，即 18 台 Prefill 机器 + 7 台 Decode 机器 |
| **SLO=2 边界** | 划分性能达标区与未达标区，**Optimal Design** 紧贴该边界以实现成本最小化 |

- **实验上下文**：该图对应 **Coding Trace**（编码工作负载，70 req/s，BLOOM-176B 模型）的集群配置探索（Figure 9b）。
- **核心结论**：
  - 增加 **Prefill Machine Count** 或 **Decode Machine Count** 均可降低 **P99 TTFT** 延迟。
  - **Optimal Design** 点精准落在 **SLO=2** 边界上，证明 SPAD 的 **less-is-more** 设计方法论能够在严格满足延迟 SLO 的前提下，最大化硬件资源利用率并最小化集群总成本（相比基线显著减少机器数量）。

### 04046cb656890910f865aa3b47d4dec9d3eb3c52695691b4cfc4ed7c5a11a668.jpg

![04046cb656890910f865aa3b47d4dec9d3eb3c52695691b4cfc4ed7c5a11a668.jpg](images/04046cb656890910f865aa3b47d4dec9d3eb3c52695691b4cfc4ed7c5a11a668.jpg)

该图片为论文中的 **Figure 9(c)**，展示了 **SPAD** 集群在 **Coding Trace** 工作负载下的 **Normalized P90 TBT**（归一化 P90 词间延迟）热力图。

- **坐标轴与映射**：
  - **X轴**：**Prefill Machine Count**（预填充机器数量），刻度范围约为 0 至 28。
  - **Y轴**：**Decode Machine Count**（解码机器数量），刻度范围约为 0 至 28。
  - **颜色映射**：右侧 **Colorbar** 表示 **Normalized P90 TBT** 数值，范围从 **1**（深紫色，低延迟）到 **7**（亮黄色，高延迟）。颜色越深代表延迟越低，性能越好。

- **关键视觉元素**：
  - **SLO=5 边界线**：图中白色虚线标记了 **SLO=5** 的等值线。虚线左下方的深紫色区域表示满足该延迟服务等级目标（**SLO**）的配置，右上方区域则不满足。
  - **Optimal Design**：紫红色圆点标记了 **最优设计点**，坐标位于 **Prefill Machine Count = 18** 和 **Decode Machine Count = 7** 处。

- **图表分析与结论**：
  - 该图直观反映了集群资源配置对 **Decode** 阶段延迟的影响。增加 **Prefill** 或 **Decode** 机器数量均可降低 **TBT**，但存在边际效益递减。
  - **最优设计点**（18 Prefill, 7 Decode）精准落在 **SLO=5** 边界线的内侧边缘。这表明该配置在严格满足 **P90 TBT** 延迟要求的前提下，实现了硬件资源（机器数量）的最小化，从而验证了 **SPAD** 的 **less-is-more** 设计理念与成本优化目标。

- **关键参数总结**：

| 参数/元素 | 描述/数值 |
| :--- | :--- |
| **图表类型** | 二维热力图 (Heatmap) |
| **X轴变量** | **Prefill Machine Count** (0 - 28) |
| **Y轴变量** | **Decode Machine Count** (0 - 28) |
| **颜色映射指标** | **Normalized P90 TBT** (1 - 7) |
| **约束边界** | **SLO = 5** (白色虚线) |
| **最优配置点** | **18 Prefill Machines**, **7 Decode Machines** |
| **工作负载** | **Coding Trace** (BLOOM-176B) |

### a2412fafac6ac6f93c89ccc840b801e4e13d076eef0ee6ef4a4e6179ed3bd158.jpg

![a2412fafac6ac6f93c89ccc840b801e4e13d076eef0ee6ef4a4e6179ed3bd158.jpg](images/a2412fafac6ac6f93c89ccc840b801e4e13d076eef0ee6ef4a4e6179ed3bd158.jpg)

- **图表主题**：该图展示了初始为 **Coding** 负载优化的 SPAD 集群，在重新分配（Reallocation）后运行 **Conversation** 负载（**BLOOM-176B** 模型）时的延迟表现与请求率的关系。
- **坐标轴定义**：
  - **X轴**：请求率（**Request Rate**），单位为 req/s，范围从 50 到 70。
  - **Y轴**：归一化延迟（**Normalized Latency**），范围从 1 到 8。
- **性能指标曲线**：
  - **P99 TTFT**（蓝色实线）：随请求率增加呈阶梯状上升，在 55 req/s 附近急剧攀升。
  - **P99 TBT**（黄色实线）：在低请求率下波动，在 55 req/s 后显著上升。
  - **P90 TTFT**（绿色实线）与 **P90 TBT**（棕色虚线）：在测试范围内保持相对平稳，延迟较低。
- **SLO 阈值限制**：图中通过水平虚线标定了各项指标的 **Service-Level Objectives (SLO)** 边界。
- **最大吞吐量瓶颈**：红色垂直虚线标示出该集群在此场景下的 **Maximal Throughput** 为 **55 req/s**。在此请求率下，**P99 TTFT** 和 **P99 TBT** 触及或逼近其 SLO 限制，成为系统吞吐量的瓶颈。

| 指标 (Metric) | SLO 阈值 (Normalized Latency) | 达到 SLO 瓶颈时的请求率 (req/s) |
| :--- | :--- | :--- |
| **P99 TTFT** | ~6.2 | **55** |
| **P99 TBT** | ~5.0 | **55** |
| **P90 TTFT** | ~2.0 | > 65 |
| **P90 TBT** | ~1.8 | > 65 |

- **核心结论**：尽管集群最初是为 **Coding** 负载配置的，但通过逻辑重新分配，它仍能以 **55 req/s** 的吞吐量有效支持 **Conversation** 负载，证明了 **SPAD** 架构在不同负载类型间的**适应性（Adaptability）**与**长寿性（Longevity）**。

### fbf919691246dd7ea6620a51fd8c6e8646daa697b222a262a96ddd672b6aab8c.jpg

![fbf919691246dd7ea6620a51fd8c6e8646daa697b222a262a96ddd672b6aab8c.jpg](images/fbf919691246dd7ea6620a51fd8c6e8646daa697b222a262a96ddd672b6aab8c.jpg)

- **图片基本信息与实验背景**
  - 该图展示了 **SPAD** 集群在**自适应重新分配（Adaptive Reallocation）** 场景下的端到端延迟表现。
  - 具体实验场景为：一个最初为 **Conversation** 工作负载优化的集群（包含 8 个 Prefill 机器和 17 个 Decode 机器），被重新分配用于运行 **Coding** 工作负载（模型为 **BLOOM-176B**）。

- **图表坐标轴与图例说明**
  - **X轴**：**Request Rate (req/s)**，表示每秒请求数，测试范围从 50 到 70。
  - **Y轴**：**Normalized Latency**，表示相对于无批处理基准的归一化延迟。
  - **图例指标**：包含 **P99 TTFT**、**P99 TBT**、**P90 TTFT** 和 **P90 TBT** 四项延迟指标。
  - **SLO 限制线**：图中用水平虚线标出了各项指标的 **Service-Level Objectives (SLOs)** 阈值。

- **关键数据与性能表现**
  - 随着请求率的增加，各项延迟指标均呈上升趋势，其中 **P99 TTFT** 对负载最为敏感。
  - 当请求率达到 **60 req/s** 时，**P99 TTFT** 的归一化延迟达到 5.0，图中将其标记为该集群在此工作负载下的 **Maximal Throughput**（最大吞吐量）。
  - 在极限吞吐量（60 req/s）下，其他指标（**P99 TBT**、**P90 TTFT**、**P90 TBT**）均远低于各自的 SLO 阈值，表现良好。
  - 当请求率继续增加至 70 req/s 时，**P99 TTFT** 严重超标（>8.0），而 **P90 TTFT** 也逼近其 SLO 阈值。

- **数据对比表格**
| 指标 (Metric) | SLO 阈值 (Threshold) | 60 req/s 时延迟表现 | 70 req/s 时延迟表现 |
| :--- | :--- | :--- | :--- |
| **P99 TTFT** | 6.0 | 5.0 (标记为瓶颈) | >8.0 (严重超标) |
| **P99 TBT** | 5.0 | ~1.5 | ~4.0 |
| **P90 TTFT** | 3.0 | ~1.2 | ~2.8 |
| **P90 TBT** | 2.0 | ~1.1 | ~1.5 |

- **核心结论与意义**
  - 该图证明了 **SPAD** 设计的**灵活性（Flexibility）** 和**长寿性（Longevity）**。
  - 即使集群最初是为长输出序列的 **Conversation** 场景配置的，通过逻辑重新分配，它依然能够以 **60 req/s** 的吞吐量高效支持长输入序列的 **Coding** 场景。
  - 结合论文正文，这种重新分配方案相比使用通用 **H100** 集群（需 21 台机器达到同等吞吐量），仍能实现 **11% 的硬件成本降低**和 **9% 的 TDP 降低**，验证了“少即是多（less-is-more）”设计理念的有效性。

### bebad7f32bf537a893961b648709b5ea4073940ffecda3f57af8895fe17996d6.jpg

![bebad7f32bf537a893961b648709b5ea4073940ffecda3f57af8895fe17996d6.jpg](images/bebad7f32bf537a893961b648709b5ea4073940ffecda3f57af8895fe17996d6.jpg)

该图片为论文中的 **Figure 10(c)**，展示了最初为 **BLOOM-176B** 优化的 **SPAD** 集群（18 Prefill + 7 Decode）在重新分配以运行 **Llama3-70B** 模型（**Coding** 工作负载）时的端到端延迟与吞吐量表现。

- **坐标轴定义**：横轴为 **Request Rate (req/s)**，范围从 120 到 200；纵轴为 **Normalized Latency**，范围从 1 到 8。
- **性能曲线**：图中包含四条延迟曲线，分别代表 **P99 TTFT**、**P99 TBT**、**P90 TTFT** 和 **P90 TBT**。
- **SLO 阈值**：水平虚线标示了各项指标的 **Service-Level Objectives (SLO)** 上限。
- **最大吞吐量**：红色垂直虚线标示了满足所有 SLO 约束下的 **Maximal Throughput**，即 **188 req/s**。

| 指标 (Metric) | SLO 阈值 (Normalized Latency) | 在 188 req/s 时的实际延迟 | 状态 |
| :--- | :--- | :--- | :--- |
| **P99 TTFT** | 6.0 | ~5.3 | 触及瓶颈 |
| **P99 TBT** | 5.0 | ~1.5 | 远低于 SLO |
| **P90 TTFT** | 3.0 | ~2.0 | 满足 SLO |
| **P90 TBT** | 2.0 | ~1.5 | 满足 SLO |

- **瓶颈分析**：**P99 TTFT** 是限制集群最大吞吐量的主要瓶颈。随着 **Request Rate** 增加，**P99 TTFT** 延迟呈非线性急剧上升，并在 **188 req/s** 时触及 **P99 TTFT SLO** 边界。
- **资源利用率**：在最大吞吐量下，**TBT** 和 **P90 TTFT** 的延迟远低于其对应的 **SLO** 阈值，表明 **Decode** 阶段和常规 **Prefill** 阶段的资源仍有冗余。
- **设计 longevity 验证**：该图验证了 **SPAD** 架构的适应性。即使模型从 **BLOOM-176B** 切换为采用 **Grouped-Query Attention (GQA)** 的 **Llama3-70B**，重新分配的集群仍能实现 **188 req/s** 的高吞吐量，相比基线 **Splitwise** 节省 **43%** 的硬件成本与 **22%** 的 **TDP**。

### 1176f0a68c558973c5603b50474a1b99c7b5794ac2d69f171f9f6605904fd4b8.jpg

![1176f0a68c558973c5603b50474a1b99c7b5794ac2d69f171f9f6605904fd4b8.jpg](images/1176f0a68c558973c5603b50474a1b99c7b5794ac2d69f171f9f6605904fd4b8.jpg)

- **图表主题**：该图展示了 **SPAD** 集群在模型变更后（从 **BLOOM-176B** 重新分配以运行 **Llama3-70B** 的 **Conversation** 工作负载）的端到端性能表现，具体对应论文中的 Figure 10(d)。
- **坐标轴与图例**：
  - **X轴**：请求率（**Request Rate**），单位为 req/s，范围覆盖 110 至 190。
  - **Y轴**：归一化延迟（**Normalized Latency**），范围从 1 到 8。
  - **图例指标**：包含 **P99 TTFT**（蓝色圆点）、**P99 TBT**（橙色菱形）、**P90 TTFT**（绿色方块）和 **P90 TBT**（棕色三角形）。
- **SLO 限制与最大吞吐量**：图中通过水平虚线标定了各项延迟指标的服务级别目标（**SLO**），并通过垂直红色虚线标定了集群的最大吞吐量。

| 延迟指标 | SLO 限制 (归一化延迟) | 触及 SLO 时的请求率 |
| :--- | :--- | :--- |
| **P99 TTFT** | ~6.0 | **171 req/s** (系统瓶颈) |
| **P99 TBT** | ~5.0 | > 171 req/s |
| **P90 TTFT** | ~3.0 | > 171 req/s |
| **P90 TBT** | ~2.0 | ~171 req/s |

- **性能趋势分析**：
  - 随着请求率的增加，所有延迟指标均呈现非线性上升趋势。
  - 在请求率达到 **171 req/s** 之前，所有延迟指标均严格保持在各自的 **SLO** 限制范围内。
  - 当请求率逼近 **171 req/s** 时，**P99 TTFT** 率先触及 **SLO** 限制（蓝色虚线），成为限制集群进一步扩展吞吐量的**核心瓶颈**。
- **结论意义**：该图验证了 **SPAD** 集群在面临模型架构变更（如从 MHA 转向 GQA）时，通过自适应重分配（**Adaptive Reallocation**）依然能够维持高吞吐量（**171 req/s**）并满足严格的延迟 **SLO**，证明了其硬件设计的**长效性（Longevity）**与灵活性。

### 8597347f9f27bf0307b134ce491ae354a574987b4f3d6a37b49b9a2a0d4db3e1.jpg

![8597347f9f27bf0307b134ce491ae354a574987b4f3d6a37b49b9a2a0d4db3e1.jpg](images/8597347f9f27bf0307b134ce491ae354a574987b4f3d6a37b49b9a2a0d4db3e1.jpg)

* 图片核心主题：评估在不同张量并行（**Tensor Parallelism, TP**）和流水线并行（**Pipeline Parallelism, PP**）配置下，**Prefill Chip** 与基准 **H100** 的 **Prefill Latency** 表现。
* 实验环境设置：
  * 仿真工具：**LLMCompass**。
  * 目标模型：**FP16 BLOOM-176B**。
  * 关键参数：序列长度（Sequence Length）为 1024，Prefill 阶段批处理大小（Batch Size）为 2。
* 坐标轴与图例说明：
  * 横轴：并行度组合 **(TP, PP)**，包含 (1, 8)、(2, 4)、(4, 2)、(8, 1) 四种配置。
  * 纵轴：**Prefill Latency (s)**，数值越低代表性能越好。
  * 图例：蓝色折线代表 **H100**，橙色折线代表 **Prefill Chip**。
* 性能数据对比：

| 并行配置 (TP, PP) | H100 延迟 (s) | Prefill Chip 延迟 (s) | 性能对比结果 |
| :--- | :--- | :--- | :--- |
| (1, 8) | ~1.10 | ~0.95 | **Prefill Chip 显著优于 H100** |
| (2, 4) | ~0.60 | ~0.55 | **Prefill Chip 优于 H100** |
| (4, 2) | ~0.35 | ~0.32 | **Prefill Chip 略优于 H100** |
| (8, 1) | ~0.20 | ~0.20 | **两者性能基本持平** |

* 核心趋势与结论分析：
  * **并行度影响**：随着 **TP** 增大和 **PP** 减小，两种硬件的 **Prefill Latency** 均大幅下降，表明增加张量并行能有效加速 Prefill 阶段。
  * **低 TP 优势**：在低 **TP** 配置（如 (1, 8)）下，**Prefill Chip** 凭借更大的 Systolic Array 和优化的计算架构，展现出比 **H100** 更低的延迟。
  * **高 TP 趋同**：在最高 **TP** 配置（(8, 1)）下，计算资源已高度饱和，导致两者延迟表现趋于一致。
  * **设计鲁棒性**：数据证明 **Prefill Chip** 在多种模型并行策略下均能保持**一致且高效**的性能，验证了其架构设计的灵活性与泛化能力。

### dff8e4c6a17f17d4d372ed1eceaf6bc5279b07e16f097a46327bb7dd8da10dac.jpg

![dff8e4c6a17f17d4d372ed1eceaf6bc5279b07e16f097a46327bb7dd8da10dac.jpg](images/dff8e4c6a17f17d4d372ed1eceaf6bc5279b07e16f097a46327bb7dd8da10dac.jpg)

- **图表基本信息**
  - **图表类型**：折线图，用于展示不同并行配置下的解码阶段延迟表现。
  - **X轴**：并行策略组合 **(TP, PP)**，即 **Tensor Parallelism** 与 **Pipeline Parallelism** 的具体配置，包含 **(1,8)**、**(2,4)**、**(4,2)** 和 **(8,1)** 四个节点。
  - **Y轴**：**Decode Latency (s)**，即解码延迟，单位为秒。
  - **对比对象**：基准硬件 **H100**（蓝色圆点线）与本文提出的专用硬件 **Decode Chip**（橙色方块线）。

- **数据趋势与对比分析**
  - **并行度影响**：随着 **TP** 的增加和 **PP** 的减少（从 (1,8) 演进至 (8,1)），**Decode Latency** 呈现显著的**非线性下降趋势**。这表明在解码阶段，提高张量并行度能更有效地降低单步推理延迟。
  - **性能差距**：**Decode Chip** 的延迟曲线与 **H100** 高度贴合。在全部测试配置中，**Decode Chip** 的延迟仅略高于或等同于 **H100**，性能损耗被控制在极小范围内。
  - **数据视觉估算**：
    | (TP, PP) 配置 | H100 延迟 (s) | Decode Chip 延迟 (s) | 性能差异评估 |
    | :--- | :--- | :--- | :--- |
    | (1, 8) | ~0.22 | ~0.23 | 差距极微 |
    | (2, 4) | ~0.12 | ~0.13 | 差距极微 |
    | (4, 2) | ~0.07 | ~0.075 | 差距极微 |
    | (8, 1) | ~0.045 | ~0.048 | 差距极微 |

- **图表核心结论**
  - **架构一致性**：该图（对应论文 **Figure 11**）证实了 **SPAD** 系统中的 **Decode Chip** 在不同的模型并行策略下，均能保持与基准 **H100** 高度一致的性能表现。
  - **设计有效性**：验证了 **Decode Chip** 在采用“少即是多”（**less-is-more**）理念削减计算资源（如缩小 **systolic arrays**）后，并未在复杂的分布式并行场景下引入额外的性能瓶颈，具备优异的**架构鲁棒性**与**部署灵活性**。

### 0351366be633af9f1d8b0384ba2c6981235af03d113804c53d3dd7d0e6e16362.jpg

![0351366be633af9f1d8b0384ba2c6981235af03d113804c53d3dd7d0e6e16362.jpg](images/0351366be633af9f1d8b0384ba2c6981235af03d113804c53d3dd7d0e6e16362.jpg)

该图片展示了在特定工作负载下，同构 **H100** 集群中 **Prefill** 和 **Decode** 阶段机器数量分配对 **Normalized P90 TTFT** 的影响热力图。

- **坐标轴定义**：
  - **X轴**：**H100 Machine Count for Prefill**，表示分配给预填充阶段的 **H100** 机器数量，刻度范围从 4 到 28。
  - **Y轴**：**H100 Machine Count for Decode**，表示分配给解码阶段的 **H100** 机器数量，刻度范围从 4 到 28。
- **颜色映射 (Colorbar)**：
  - 代表 **Normalized P90 TTFT**（归一化 P90 首 Token 延迟）。
  - 颜色从深紫色（数值 1.0，延迟最低）渐变至亮黄色（数值 4.0，延迟最高）。
- **关键视觉标记**：
  - **SLO 边界线**：图中虚线（标注为 **SLO=3**）划分了满足与不满足服务等级目标 (**SLO**) 的配置区域。左上方深色区域为可行域，右下方浅色区域为违规域。
  - **Optimal Design (最优设计点)**：图中紫色圆点标记了满足 **SLO** 的 **Pareto-optimal**（帕累托最优）配置。根据论文图注，核心最优配置点位于 **(18, 7)**，即 18 台 **Prefill** 机器和 7 台 **Decode** 机器。
- **数据与配置分析**：

| 配置特征 | 机器数量分配 (Prefill, Decode) | 性能表现 (Normalized P90 TTFT) | 区域状态 |
| :--- | :--- | :--- | :--- |
| **核心最优配置** | (18, 7) | 贴近 SLO 阈值 | 边界边缘，资源利用率最高 |
| **高 Decode 配置** | (8, 20) 附近 | 远低于 SLO 阈值 (深紫色) | 性能冗余，资源浪费 |
| **低 Decode 配置** | (20, 4) 附近 | 远超 SLO 阈值 (亮黄色) | 严重违反 SLO，存在解码瓶颈 |

- **核心结论**：
  - 在 **Splitwise-homo**（同构 **H100** 集群）和 **Coding** 工作负载（70 req/s）下，**Decode** 阶段对机器数量更为敏感。
  - 减少 **Decode** 机器数量会导致 **P90 TTFT** 急剧上升（颜色迅速变黄）。
  - 维持至少 25 台总机器（如 18+7）是满足严格 **SLO** 的最低硬件要求，证明了同构集群在资源分配上的局限性与成本劣势。

### 7f9f95378206861c7c2307698a300fdaca949a1524496502968378d3b509dabc.jpg

![7f9f95378206861c7c2307698a300fdaca949a1524496502968378d3b509dabc.jpg](images/7f9f95378206861c7c2307698a300fdaca949a1524496502968378d3b509dabc.jpg)

- **图表主题**：该图展示了在 **Splitwise-homo** 架构下，使用 **BLOOM-176B** 模型处理 **Coding Trace (70 req/s)** 工作负载时，不同 **Prefill** 和 **Decode** 机器数量组合对 **Normalized P90 TTFT** 延迟的影响。
- **图表元素解析**：

| 元素 | 描述 |
| --- | --- |
| **X轴** | 用于 **Prefill** 阶段的 **H100 Machine Count**（范围约 0-28） |
| **Y轴** | 用于 **Decode** 阶段的 **H100 Machine Count**（范围约 0-28） |
| **颜色映射** | 代表 **Normalized P90 TTFT** 延迟值（1至8）。深色（紫/蓝）表示低延迟，浅色（黄/绿）表示高延迟 |
| **白色虚线** | 标记为 **SLO=6**，划分满足与不满足该 **SLO** 约束的边界区域 |
| **紫色圆点** | 标记为 **Optimal Design**，位于坐标 **(18, 7)** 处 |

- **核心分析**：
  - **资源分配权衡**：图表直观反映了 **Prefill** 与 **Decode** 资源分配的博弈。增加 **Prefill** 机器数量可显著降低 **TTFT**（颜色向深色过渡），但会挤压 **Decode** 机器的可用数量。
  - **最优配置识别**：在满足 **SLO=6** 约束（虚线左下方区域）的前提下，**(18, 7)** 的组合（总计 25 台机器）被识别为 **Optimal Design**。这与论文正文中“至少需要 25 台 modeled 8-H100 机器”的结论完全吻合。
  - **性能瓶颈边界**：当 **Prefill** 机器数量不足（如 X轴 < 12）时，无论分配多少 **Decode** 机器，**TTFT** 均无法满足 **SLO=6** 的要求，表明 **Prefill** 阶段的计算资源是降低 **TTFT** 的决定性因素。

### 75df246c626f919926b95e5878a707fbb07d16c04e2c969f08284c7bc15ee2e6.jpg

![75df246c626f919926b95e5878a707fbb07d16c04e2c969f08284c7bc15ee2e6.jpg](images/75df246c626f919926b95e5878a707fbb07d16c04e2c969f08284c7bc15ee2e6.jpg)

* **图表基本信息**
  * **图表来源**：Figure 12 (c)，展示 **Splitwise-homo** 架构在 **Coding Trace (70 req/s)** 和 **BLOOM-176B** 模型下的 **Normalized P90 TBT** 配置结果。
  * **图表类型**：二维热力图（Heatmap）。
  * **X轴**：**H100 Machine Count for Prefill**（分配给 Prefill 阶段的 H100 机器数量）。
  * **Y轴**：**H100 Machine Count for Decode**（分配给 Decode 阶段的 H100 机器数量）。

* **视觉元素与数据解析**
  * **颜色映射（Colorbar）**：代表 **Normalized P90 TBT** 数值，范围从 **1.0**（深紫色，延迟最低）到 **2.8**（亮黄色，延迟最高）。
  * **SLO 边界线**：图中白色虚线标注为 **SLO=2.0**，作为性能达标与否的分界线。虚线左下方（深紫色区域）表示满足 SLO 要求，右上方（黄绿色区域）表示违反 SLO。
  * **最优设计点（Optimal Design）**：紫色圆点标记了帕累托最优配置，位于坐标 **(18, 7)**，即 **18 台 Prefill 机器** 和 **7 台 Decode 机器**。

* **核心结论**
  * **最小硬件需求**：在 **Splitwise-homo** 基线配置下，至少需要 **25 台** 8-H100 机器（18+7）才能满足严格的延迟 SLO 要求。
  * **资源分配比例**：Prefill 阶段需要更多的计算资源（18台），而 Decode 阶段由于内存带宽瓶颈，仅需较少机器（7台）即可满足 70 req/s 的吞吐量需求。

* **图表数据摘要**
| 参数/指标 | 数值/描述 |
| --- | --- |
| **评估架构** | Splitwise-homo (同构 H100 集群) |
| **工作负载** | Coding Trace (70 req/s), BLOOM-176B |
| **评估指标** | Normalized P90 TBT |
| **SLO 阈值** | 2.0 |
| **最优配置 (Optimal Design)** | 18 Prefill 机器 + 7 Decode 机器 |
| **总机器数量** | 25 台 (8-H100 机器) |

* **研究意义**
  * 该图确立了同构 GPU 集群在 **Disaggregated LLM Serving** 中的硬件成本基线。
  * 通过展示 **Splitwise-homo** 需要 25 台 H100 机器，为后文证明 **SPAD** 架构（使用专用芯片）能够以更低的硬件成本和 TDP 达到相同性能提供了直接的对比依据。

### a33ff23db73702046c9557cd2b2d84b45c20d0b2884ce2d1a3b55de928fcdefb.jpg

![a33ff23db73702046c9557cd2b2d84b45c20d0b2884ce2d1a3b55de928fcdefb.jpg](images/a33ff23db73702046c9557cd2b2d84b45c20d0b2884ce2d1a3b55de928fcdefb.jpg)

* **图表概述**：该图是一张**热力图**，展示了在 **Splitwise-homo** 调度策略下，针对 **Coding Trace** (70 req/s) 和 **BLOOM-176B** 模型，不同 **Prefill** 和 **Decode** 机器数量组合对 **Normalized TBT** 性能的影响。
* **坐标轴与映射**：
  * **X轴**：**H100 Machine Count for Prefill**，表示用于 Prefill 阶段的 H100 机器数量。
  * **Y轴**：**H100 Machine Count for Decode**，表示用于 Decode 阶段的 H100 机器数量。
  * **颜色条 (Colorbar)**：数值范围从 **1 到 7**，颜色由深紫（低延迟/高性能）向亮黄（高延迟/低性能）过渡，代表 **Normalized TBT** 的倍数。
* **关键标记与区域**：
  * **SLO=5 虚线**：划分了满足与不满足 **SLO (Service-Level Objective)** 阈值的区域。虚线左下方的深紫色区域表示配置能够满足严格的延迟要求。
  * **Optimal Design 紫点**：标记了 **Pareto-optimal** 设计点，其具体坐标为 **Prefill=18** 和 **Decode=7**。该点位于满足 SLO 的边界内，代表了在满足性能要求的前提下，硬件资源分配最优的配置。
* **数据与结论总结**：

| 指标/特征 | 详细描述 |
| :--- | :--- |
| **评估模型** | **BLOOM-176B** (FP16, TP=8) |
| **工作负载** | **Coding Trace** (70 req/s) |
| **调度策略** | **Splitwise-homo** (同构 H100 集群) |
| **最优配置 (Optimal Design)** | **18** 台 Prefill 机器 + **7** 台 Decode 机器 (总计 25 台) |
| **SLO 阈值** | **SLO=5** (Normalized TBT 边界) |
| **核心结论** | 至少需要 **25** 台 modeled 8-H100 机器才能满足所有 **SLOs**；该图直观证明了同构集群在资源分配上的权衡与硬件成本瓶颈。 |

### 0e6f97d3771504c34404f6501f48dc5dadd877fa2b8041faaa99420d505d47db.jpg

![0e6f97d3771504c34404f6501f48dc5dadd877fa2b8041faaa99420d505d47db.jpg](images/0e6f97d3771504c34404f6501f48dc5dadd877fa2b8041faaa99420d505d47db.jpg)

- **图表主题**：该图展示了使用 **Sarathi** 调度器在 **Coding** 工作负载（70 req/s）下，运行 **BLOOM-176B** 模型时的集群配置结果（Figure 13a）。
- **坐标轴定义**：
  - **X轴**：**H100 Machine Count**（H100 机器数量），范围从 20 到 50。
  - **Y轴**：**Normalized Latency**（归一化延迟），范围从 1 到 8。
- **曲线与阈值**：
  - 图中包含四条延迟曲线：**P99 TTFT**、**P99 TBT**、**P90 TTFT** 和 **P90 TBT**。
  - 对应的水平虚线表示各项指标的 **SLO**（Service-Level Objectives）阈值。
- **关键数据点**：
  - **P99 TTFT** 是限制集群规模的最严格指标。当机器数量为 **36** 时，P99 TTFT 曲线与 SLO 阈值相交（图中蓝色圆点及红色垂直虚线标记）。
  - **P90 TTFT** 在机器数量约为 **33** 时满足 SLO 要求（图中绿色圆点）。
  - **TBT** 指标（P90 和 P99）在机器数量约为 **20** 时即可满足 SLO（图中橙色和棕色方块）。
- **核心结论**：在 **Sarathi** 架构下，为了满足所有 **SLO** 要求，至少需要配置 **36** 台 modeled 8-H100 机器。

| 指标 (Metric) | 满足 SLO 所需最小 H100 机器数 | 对应图中标记 |
| :--- | :--- | :--- |
| **P99 TTFT** | **36** | 蓝色圆点 (Blue Circle) |
| **P90 TTFT** | **~33** | 绿色圆点 (Green Circle) |
| **P99 TBT** | **~20** | 橙色方块 (Orange Square) |
| **P90 TBT** | **~20** | 棕色方块 (Brown Square) |

### 4c5078d938f544db6a12f2d044db11e1d7de53363810ddc00878eeb31e398796.jpg

![4c5078d938f544db6a12f2d044db11e1d7de53363810ddc00878eeb31e398796.jpg](images/4c5078d938f544db6a12f2d044db11e1d7de53363810ddc00878eeb31e398796.jpg)

- **图片主题**：展示 **Sarathi** 系统在 **Conversation (70 req/s)** 工作负载下，运行 **BLOOM-176B** 模型时的集群配置结果（对应 Figure 13(b)）。
- **坐标轴与图例**：
  - **X轴**：**H100 Machine Count**，表示配置的 8-H100 机器数量，范围从 20 到 50。
  - **Y轴**：**Normalized Latency**，表示归一化延迟，范围从 1 到 8。
  - **图例指标**：包含四个延迟指标，分别为 **P99 TTFT**、**P99 TBT**、**P90 TTFT** 和 **P90 TBT**。
- **SLO 阈值与曲线映射**：
| 延迟指标 | SLO 阈值 (Normalized Latency) | 曲线颜色与样式 |
| --- | --- | --- |
| **P99 TTFT** | 6 | 蓝色虚线圆点 |
| **P99 TBT** | 5 | 橙色虚线方块 |
| **P90 TTFT** | 3 | 绿色虚线圆点 |
| **P90 TBT** | 2 | 棕色虚线方块 |
- **关键数据点**：
  - 图中红色垂直虚线标记了 **34**，代表满足所有 **SLO** 约束所需的最小 **H100 Machine Count**。
  - 当机器数量为 34 时，**P99 TTFT** 曲线刚好降至其 **SLO** 阈值（6）以下，成为限制集群规模的核心瓶颈。
- **趋势分析**：
  - 随着 **H100 Machine Count** 的增加，所有延迟指标均呈显著下降趋势。
  - **P99 TTFT** 对机器数量最为敏感，下降斜率最大；而 **TBT** 指标在机器数量较少时已接近其 **SLO** 阈值。
- **核心结论**：
  - 在 **Conversation** 场景下，**Sarathi** 系统至少需要 **34 台** 8-H100 机器才能满足所有延迟 **SLO** 要求。
  - 这证明了 **co-location-based scheduling**（如 **Sarathi**）在满足严格延迟约束时，需要消耗大量的硬件资源，从而凸显了 **SPAD** 异构集群设计的成本优势。

### b87f9ac519c2f64b8ee0e067d1e71d169e4fe6e13d9b08aa83115903cbb7c606.jpg

![b87f9ac519c2f64b8ee0e067d1e71d169e4fe6e13d9b08aa83115903cbb7c606.jpg](images/b87f9ac519c2f64b8ee0e067d1e71d169e4fe6e13d9b08aa83115903cbb7c606.jpg)

- **图表类型**：二维热力图（Heatmap），用于可视化不同硬件配置下的延迟性能分布。
- **坐标轴定义**：
  - **X轴**：**Prefill Machine Count**（Prefill阶段机器数量），刻度范围约为0至28。
  - **Y轴**：**Decode Machine Count**（Decode阶段机器数量），刻度范围约为0至28。
- **颜色映射（Colorbar）**：
  - 右侧色条表示 **Normalized P90 TTFT**（归一化P90首Token延迟）的数值，范围从 **1.0** 到 **4.0**。
  - **深紫色/黑色**区域代表较低的延迟（性能更优），**黄绿色**区域代表较高的延迟（性能较差）。

| 参数/指标 | 描述/数值 |
| :--- | :--- |
| **X轴** | **Prefill Machine Count** (0 - 28) |
| **Y轴** | **Decode Machine Count** (0 - 28) |
| **颜色映射** | **Normalized P90 TTFT** (1.0 - 4.0) |
| **SLO边界** | **SLO-3** (白色虚线) |
| **最优配置点** | **(8, 17)** (紫色实心圆点) |

- **关键标记与边界**：
  - **白色虚线**：标记为 **SLO-3**，代表满足特定服务等级目标（Service Level Objective）的性能边界线。虚线左下方深色区域为满足SLO的有效配置区。
  - **紫色实心圆点**：图例标注为 **Optimal Design**（最优设计），精确位于坐标 **(8, 17)** 处。
  - **红色空心圆圈**：位于最优设计点附近，代表次优或边界对比配置点。
- **图表结论**：
  - 该图展示了基于 **Conversation Trace**（对话工作负载）的 **SPAD** 集群配置结果。
  - 在满足SLO约束的前提下，系统通过权衡Prefill与Decode机器的数量，找到了成本与性能的最佳平衡点。
  - 最终确定的 **Optimal Design** 为 **8台Prefill机器** 和 **17台Decode机器**，该配置在满足延迟要求的同时实现了硬件成本的最优化。

### 0b96fdfb5599cae6bf365f9330648239c1dd3d66635c618e6818bb132b10cd12.jpg

![0b96fdfb5599cae6bf365f9330648239c1dd3d66635c618e6818bb132b10cd12.jpg](images/0b96fdfb5599cae6bf365f9330648239c1dd3d66635c618e6818bb132b10cd12.jpg)

- **图表主题**：该图展示了在 **Conversation Trace** 工作负载下，**SPAD** 集群的 **Provisioning Results**，核心评估指标为 **Normalized P90 TBT**。
- **坐标轴定义**：
  - **X轴**：**Prefill Machine Count**，表示预填充阶段的机器数量，刻度范围约为 0 至 28。
  - **Y轴**：**Decode Machine Count**，表示解码阶段的机器数量，刻度范围约为 0 至 28。
- **颜色映射（Colorbar）**：
  - 颜色代表 **Normalized P90 TBT** 的数值，范围从 **1.0**（黄色，延迟最低）到 **2.8**（深紫色，延迟最高）。
  - 左下角深紫色区域表示机器数量严重不足，导致 TBT 远超限制；右上角黄绿色区域表示资源充足，TBT 接近基准值 1.0。
- **关键标记**：
  - **白色虚线**：标记为 **SLO=2**，代表满足 P90 TBT 服务级别目标（即延迟不超过基准 2 倍）的临界边界。
  - **红圈标记**：标记为 **Optimal Design**，位于坐标 **(8, 17)** 处，即配置 8 台 **Prefill Machine** 和 17 台 **Decode Machine**。该点位于 SLO 边界内侧，是满足性能要求且硬件成本最低的最优解。
- **核心数据与配置总结**：

| 元素 | 描述 / 数值 |
| --- | --- |
| **工作负载** | **Conversation Trace** |
| **评估指标** | **Normalized P90 TBT** |
| **SLO 边界** | **SLO=2** (白色虚线) |
| **最优设计 (Optimal Design)** | **8** Prefill Machines + **17** Decode Machines |
| **颜色极值** | 1.0 (黄色, 最优) 至 2.8 (深紫色, 最差) |

### 74a74cffb27171f912b4fdc8e09bde4bc714c47e640e5c51cd2054ee4c0afc2f.jpg

![74a74cffb27171f912b4fdc8e09bde4bc714c47e640e5c51cd2054ee4c0afc2f.jpg](images/74a74cffb27171f912b4fdc8e09bde4bc714c47e640e5c51cd2054ee4c0afc2f.jpg)

- **图表基本信息**
  - **图表类型**：折线图，展示 **Normalized Prefill Latency** 随 **Memory Bandwidth** 变化的趋势。
  - **X轴**：**Memory Bandwidth (GB/s)**，范围从 1000 到 4000。
  - **Y轴**：**Norm. Prefill Latency**，归一化延迟，范围从 1.0 到 2.2 左右。
  - **基准点**：**H100**（紫色五角星），位于 **Memory Bandwidth** 约 3352 GB/s 处，归一化延迟为 1.0。
  - **数据系列**：固定 **Batch Size (BS)** 为 2，包含四种 **Sequence Length (Seq)**：64、1024、4096、16384。

- **数据趋势分析**
  - **整体趋势**：随着 **Memory Bandwidth** 的降低，所有序列长度的 **Norm. Prefill Latency** 均呈上升趋势，表明 **Prefill** 阶段对内存带宽具有依赖性。
  - **长序列敏感度**：当 **Seq** 为 16384（红色曲线）时，在低带宽（如 1000 GB/s）下延迟最高（约 2.1），表明长序列的 **Prefill** 对 **Memory Bandwidth** 极度敏感。
  - **短序列表现**：当 **Seq** 为 64（蓝色曲线）时，在低带宽下延迟同样显著增加（约 2.05），说明极短序列因数据复用率低，也会向 **memory-bandwidth-bound** 偏移。
  - **中等序列**：**Seq** 为 1024 和 4096 的曲线相对平缓，在带宽降至 2000 GB/s 时，延迟增幅相对较小。

- **核心结论**
  - **Bottleneck Shifting**：图表验证了 **Prefill** 阶段的计算瓶颈会发生动态偏移。
  - **极短序列**：由于有限的 **data reuse**，**Prefill** 会向 **memory-bandwidth-bound** 偏移。
  - **极长序列**：由于 **Attention** 机制的二次复杂度，内存压力增大，同样向 **memory-bound** 偏移，且 **KV cache** 容量也会成为瓶颈。

- **关键数据估算表**
  | Memory Bandwidth (GB/s) | Seq: 64 (BS: 2) | Seq: 1024 (BS: 2) | Seq: 4096 (BS: 2) | Seq: 16384 (BS: 2) |
  | :--- | :--- | :--- | :--- | :--- |
  | 1000 | ~2.05 | ~1.60 | ~1.85 | ~2.10 |
  | 1500 | ~1.55 | ~1.35 | ~1.45 | ~1.60 |
  | 2000 | ~1.30 | ~1.20 | ~1.25 | ~1.35 |
  | 3000 | ~1.10 | ~1.05 | ~1.10 | ~1.10 |
  | 3352 (H100) | 1.00 | 1.00 | 1.00 | 1.00 |

### ccc86f72d2acb298b4a33d60f59ababe8985220d3f8d878326f0a72b08dab762.jpg

![ccc86f72d2acb298b4a33d60f59ababe8985220d3f8d878326f0a72b08dab762.jpg](images/ccc86f72d2acb298b4a33d60f59ababe8985220d3f8d878326f0a72b08dab762.jpg)

- **图表基本信息**

| 属性 | 描述 |
| :--- | :--- |
| **图表类型** | 折线图（Line Chart） |
| **横轴（X轴）** | **Core Count**（核心数量，范围 50 - 150） |
| **纵轴（Y轴）** | **Norm. Decode Latency**（归一化解码延迟，基准 1.0） |
| **基准参考点** | **H100** 模型（Core Count ≈ 132，延迟 = 1.0） |
| **控制变量** | **Sequence Length** = 256 |
| **自变量** | **Batch Size**（BS: 32, 64, 128, 256） |

- **数据趋势分析**

  - **小 Batch Size 场景（BS: 32, 64）**：曲线走势平缓。当 **Core Count** 从 132 降至 50 时，**Norm. Decode Latency** 仅轻微上升（约 1.05 至 1.15），表明在常规负载下，解码阶段对计算资源不敏感，主要受限于内存带宽（**memory-bound**）。
  - **大 Batch Size 场景（BS: 128, 256）**：曲线走势陡峭。随着 **Core Count** 减少，**Norm. Decode Latency** 急剧上升。例如，当 BS 为 256 且 Core Count 降至 50 时，延迟飙升至 1.8 以上。
  - **瓶颈偏移现象**：随着 **Batch Size** 增大，计算密度（arithmetic intensity）显著提升，解码阶段的性能瓶颈逐渐从内存带宽向计算能力（**compute-bound**）偏移。

- **核心结论**

  - **硬件资源冗余验证**：在常规或较小 **Batch Size** 下，大幅削减 **Core Count** 对 **Decode Latency** 影响甚微，直接证明了通用 GPU（如 **H100**）在解码阶段存在严重的计算资源闲置。
  - **架构设计支撑**：该数据有力支撑了论文中 **Decode Chip** 采用较小计算阵列（smaller systolic arrays）的“少即是多（**less-is-more**）”设计理念，能够在几乎不牺牲性能的前提下降低硬件成本与 TDP。
  - **边界条件提示**：在极高并发（如 BS ≥ 256）的极端场景下，计算资源将成为瓶颈，提示在实际集群调度中需结合 **Batch Size** 动态评估硬件适配性。

