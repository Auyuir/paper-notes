# Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Haoran Wu, Can Xiao, Jiayi Nie, et al.

**发表期刊/会议 (Journal/Conference)**: ArXiv

**发表年份 (Publication Year)**: 2026

**研究机构 (Affiliations)**: University of Cambridge, Imperial College London, University of Edinburgh

---

## 1. 摘要

**目的**

- 解决 agentic LLM inference 中因 long context（如 128k tokens）引发的 **memory walls**（bandwidth 和 capacity 限制）问题。
- 消除传统 square-shaped systolic arrays 在处理 fat GEMM（batch size 远小于 reduction dimension）时导致的计算资源严重 under-utilization 现象。

![](images/4350dc7220f6521d1422a27bb94852e9ccc07ce4a2cbf5af4083475a56cc2b26.jpg)

---

**方法**

提出 **PLENA**，一个 hardware-software co-designed system，包含三条核心优化 pathways：

- **Pathway 1: Flattened Systolic Array**
  - 采用 output-stationary dataflow，将大尺度 square array 重构为 flattened 结构。
  - 在 FFN 层执行 (BLEN, MLEN) × (MLEN, BLEN) GEMM，在 FlashAttention 层支持 per-head GEMM 并行计算。
  - 集成 result adder tree 执行 cross-array summation，消除计算 bubbles。
- **Pathway 2: Asymmetric Quantization Scheme**
  - 支持 Weights (W) / Activations (A) / KV Cache (KV) 采用不同精度的 Microscaling (MX) 数据格式（MXINT 或 MXFP）。
  - 提出 **output-norm guided blockwise clipping**，将其集成至 GPTQ 迭代流程中，优化 weight quantization。
  - 提出 **selective rotation** 策略，仅对特定层的 activation 和 KV cache 应用 Hadamard 变换以抑制 outliers。
- **Pathway 3: Native FlashAttention Support**
  - 设计 custom ISA，支持 fine-grained, tile-level scheduling，实现 fused GEMM-Softmax-GEMM pipeline。
  - 开发 **transposable Matrix SRAM**，支持 transpose-on-read 和 strided/blocked streaming，避免 bank conflicts。
  - 集成 hardware prefetch engines，实现 off-chip memory 访问与计算的 overlap。
- 提供完整 software-hardware stack：
  - 包含 custom ISA、PyTorch-to-PLENA compiler、transaction-level simulator（比 RTL 快 **200×**）。
  - 采用 multi-objective Bayesian optimization 进行 automated design-space exploration (DSE)。

![](images/dbc24c800aeaa4db7a40fa192c83d8dbb0be2c664a85685ae77163b16a575890.jpg) *Fig. 6: At each cycle, the flattened systolic array fetches two MLEN-wide inputs: one from the Matrix SRAM (top) and one from the Vector SRAM (left). The inputs are buffered and reordered, then partitioned into MLEN/BLEN subvectors, each of width BLEN. Each subvector is forwarded to a corresponding sub-array from the top and left directions. The scales and elements are streamed separately to each subarray. For improved resource efficiency, each PE consumes MX format inputs and performs accumulation in INT precision. The accumulated results are converted to the target activation precision before being written back to the Vector SRAM.*

---

**结果**

- 系统性能对比（基于 LLaMA-3.3-70B agentic workload）：

| System | TPS (×A100) | Tok/J (×A100) | BS |
| :--- | :--- | :--- | :--- |
| A100 | 1.00x | 1.00x | 4 |
| TPU v6e | 0.46x | N/A | 4 |
| MicroScopiQ | 0.20x | 0.41x | 16 |
| **PLENA** | **2.23x** | **4.07x** | 16 |

- 量化精度对比（LLaMA-3-8B, W4A4KV4 设置）：
  - WikiText-2 perplexity 为 **7.17**，优于 QuaRot 的 **8.16**。
  - Zero-shot downstream tasks 平均准确率为 **70.10%**，优于 QuaRot 的 **65.18%**。
- 硬件利用率：
  - Flattened systolic array 在 FFN 和 FlashAttention 层均维持高 utilization。
  - 相比 MicroscopiQ、Olive 等基线，PLENA 在 agentic workload 下的 attainable TOPs/mm² 达到 **12.81**。

![](images/45560d485999d69f738b406b0a90ef975f342cef4248da4eea26b370b2cf4ba0.jpg) *(a) PLENA achieves higher utilization than the standard square systolic array with the same resources. (b) PLENA’s optimization pathways—(1) a flattened systolic array and (2) asymmetric quantization—together achieve improved effective memory bandwidth utilization and help reduce memory capacity limitations. Fig. 2: A comparison of attainable FLOPs between a squareshaped systolic array (e.g. in a TPU) and PLENA’s when using the same number of multipliers<sup>2</sup>.*

---

**结论**

- PLENA 成功通过 flattened systolic array、asymmetric quantization 和 native FlashAttention support 突破了 agentic LLM inference 的 memory walls。
- 构建了完整的系统探索框架，为未来 Transformer 模型优化提供了原型平台。
- 未来工作将聚焦于与 GPU 系统的 heterogeneous 集成，以及扩展对 multi-turn 和 multi-modal agentic workloads 的支持。

---

## 2. 背景知识与核心贡献

**研究背景与动机**
- LLM作为AI agents的核心，需处理长上下文输入（如完整网页DOM、复杂工具调用轨迹），与传统chatbot任务存在根本差异。
- Agentic workloads单次推理消耗的Token数量可达传统任务的**100倍**，导致显著的off-chip memory traffic。
- 长上下文推理面临两大**Memory Walls**限制：
  - **Bandwidth Wall**：大量KV值和权重的读取及写回产生巨大内存带宽需求。
  - **Capacity Wall**：KV cache随context length线性增长，迅速主导内存使用并限制batch size。
- 计算特征发生转移：随着context增长，计算重心从FFN-intensive转向**Attention-intensive**。
- 现有硬件架构不匹配：传统TPU和GPU采用square-shaped systolic arrays，在处理长上下文受限batch size导致的**fat GEMM**（M维度远小于K和N维度）时，计算资源利用率严重低下。

![](images/4350dc7220f6521d1422a27bb94852e9ccc07ce4a2cbf5af4083475a56cc2b26.jpg)

![](images/8ebb7251ee890aaf3689268dad7465c8befd9a0189675bf61f4da89dfdf38c0e.jpg)

![](images/fd8b092b7b8cfb7f1b319a0eb0a364c88b59d0dd75ef76c0865ba202fbd9746e.jpg)

---

**核心贡献**
- 提出**PLENA**：一种软硬件协同设计的Transformer加速器系统，旨在打破Memory Walls，维持长上下文Agentic inference全阶段的高计算利用率。
- **Pathway 1：Flattened Systolic Array架构**
  - 解决传统square-shaped GEMM硬件的架构不匹配问题。
  - 在FFN和FlashAttention阶段实现高计算利用率。
- **Pathway 2：Asymmetric Quantization Scheme**
  - 支持Weights (W)、Activations (A)和KV Cache (KV)采用不同精度。
  - 深入探索microscaling arithmetic与rotation、norm-guided iterative optimization等PTQ优化技术的兼容性。
  - 提出block-wise clipping和selective rotation策略，缓解带宽与容量限制。
- **Pathway 3：原生FlashAttention支持**
  - 通过custom ISA和可转置Matrix SRAM，实现tile-level内存预取重叠和低开销转置读取。
  - 支持在线softmax所需的行内归约和非线性操作，减少off-chip memory traffic。
- **完整软硬件栈**
  - 包含custom ISA、PyTorch-to-PLENA ISA编译器、HBM-enabled transactional emulator和自动化DSE流程。
- **性能表现**
  - 在相同乘法器数量和内存配置下，LLaMA agentic inference吞吐量显著提升。

![](images/45560d485999d69f738b406b0a90ef975f342cef4248da4eea26b370b2cf4ba0.jpg) *(a) PLENA achieves higher utilization than the standard square systolic array with the same resources. (b) PLENA’s optimization pathways—(1) a flattened systolic array and (2) asymmetric quantization—together achieve improved effective memory bandwidth utilization and help reduce memory capacity limitations. Fig. 2: A comparison of attainable FLOPs between a squareshaped systolic array (e.g. in a TPU) and PLENA’s when using the same number of multipliers<sup>2</sup>.*

| 对比平台 | 吞吐量提升倍数 | 能效提升倍数 |
| :--- | :--- | :--- |
| **A100 GPU** | **2.23×** | **4.04×** |
| **TPU v6e** | **4.70×** | N/A |

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**系统概述**

PLENA 是一种针对长上下文 Agentic LLM Inference 的软硬件协同设计系统，旨在打破 Memory Walls（带宽与容量限制）。其整体架构围绕三大核心优化路径构建：
- **Pathway 1: Flattened Systolic Array**：打破传统方形阵列在 Fat GEMM 中的利用率瓶颈。
- **Pathway 2: Asymmetric Quantization Scheme**：结合 Microscaling (MX) 数据格式，缓解带宽与容量压力。
- **Pathway 3: Native FlashAttention Support**：通过定制 ISA 与架构，原生支持 IO-aware 的融合注意力计算。

![](images/45560d485999d69f738b406b0a90ef975f342cef4248da4eea26b370b2cf4ba0.jpg) *(a) PLENA achieves higher utilization than the standard square systolic array with the same resources. (b) PLENA’s optimization pathways—(1) a flattened systolic array and (2) asymmetric quantization—together achieve improved effective memory bandwidth utilization and help reduce memory capacity limitations. Fig. 2: A comparison of attainable FLOPs between a squareshaped systolic array (e.g. in a TPU) and PLENA’s when using the same number of multipliers<sup>2</sup>.*

---

**硬件架构**

PLENA 硬件采用指令级流水线设计，主要由以下计算与存储单元构成：
- **计算单元**：
  - **Matrix Unit**：基于 Flattened Systolic Array 架构，由多个小型方形 sub-arrays 组成，支持 Output-stationary 数据流，并集成 Result Adder Tree 进行跨阵列求和。
  - **Vector Unit**：执行 elementwise 和 reduction 操作，支持在线 Softmax 所需的非线性运算。
  - **Scalar Unit**：执行标量 INT 和 FP 算术运算。
- **存储单元**：
  - **Vector SRAM**：作为暂存器，存储高精度 FP 格式的 Activations，避免频繁读写 HBM。
  - **Matrix SRAM**：专用于加载 Weights 和 KV tensors，支持低开销的 Transpose-on-read 访问模式，优化 FlashAttention 中的 $QK^\top$ 计算。
  - **Memory Load Unit**：集成硬件预取引擎，实现 HBM 访问与计算指令的并行，隐藏内存延迟。

![](images/5f6878f7b16e75497c8fc15ffe1da3f648c6ec80ab70171ef99fff713e53935b.jpg) *Fig. 4: PLENA accelerator architecture overview. Execution is controlled by the decoder’s system-pipeline controller, which derives control signals from decoded instructions and monitors memory dependencies. For example, when reading from a Vector SRAM row that is still being updated by the vector or matrix unit, the controller inserts a stall to ensure correctness.*

---

**量化与数据格式**

PLENA 采用非对称量化策略，针对不同数据的敏感度配置不同精度：
- **Activations**：存储于片上 Vector SRAM，保持高精度 Floating-point (FP) 格式。
- **Weights 与 KV Cache**：采用低精度 Microscaling (MX) 格式（MXINT 或 MXFP），存储于 Matrix SRAM 与 HBM。
- **Selective Rotation**：在量化前对特定层应用 Hadamard-based rotation 抑制 Outliers，硬件原生支持运行时逆变换。
- **Block-wise Clipping**：在 GPTQ 迭代流程中引入 Output-norm guided 裁剪搜索，优化 MXINT 算术的表示范围。

![](images/58747ef0a90b3f1e484e43388f0994fccf4af425750768d606567f8f6cf21dee.jpg) *Fig. 3: Illustration of the configurable MX data format design, parameterized with tunable configs. Each block of elements shares a power-of-two scaling factor.*

---

**软件栈与协同设计**

PLENA 提供完整的软硬件协同设计框架，支持多保真度评估与自动化探索：
- **PLENA ISA**：32 位定制指令集，支持 Tile-level scheduling，覆盖 Matrix、Vector、Scalar、HBM 及 Control 操作。
- **Compiler**：轻量级编译器，解析模型配置并映射至预定义的 PLENA ISA 汇编模板。
- **Simulator**：基于 Rust 的 Transaction-level 仿真器，集成 Ramulator 和 DRAMSys，提供 Cycle-accurate 的时序与带宽建模。
- **Design-Space Exploration (DSE)**：利用多目标 Bayesian optimization (BoTorch) 自动探索 Pareto frontier，平衡 Accuracy、Latency 与 Area。

![](images/8072276b8d6cadfe082fdf94940d21b7a6d326b7ab287683d3e62d522bc0d6e4.jpg) *Fig. 10: The co-design framework consists of hierarchical layers (actual hardware, transactional emulator, and analytic simulator) with different fidelities. The transaction-level simulator offers good fidelity (cycle-accurate) while achieving an over 200× speedup compared to RTL simulation.*

---

**关键参数对比**

PLENA 与传统架构在计算面积与利用率上的对比（基于 Table XIII 数据）：

| Design | Comp Area (mm²) | TOPs/mm² | S.A.T/mm²* | A.A.T/mm²* |
|---|---|---|---|---|
| MicroscopiQ | 0.1378 | 59.45 | 26.36 | 5.83 |
| Olive | 0.319 | 25.66 | 13.76 | 2.40 |
| FIGNA | 0.471 | 17.39 | 7.51 | 1.83 |
| SystolicAttn | 1.17 | 14.00 | 7.14 | 4.76 |
| **PLENA** | **0.237** | **34.49** | **29.31** | **12.81** |

### 1. Flattened Systolic Array Architecture

**核心背景与设计动机**

PLENA 提出 **Flattened Systolic Array** 架构，旨在解决长上下文 Agentic LLM Inference 中因 **Memory Walls**（Bandwidth 与 Capacity 限制）导致的硬件算力严重闲置问题。

![](images/45560d485999d69f738b406b0a90ef975f342cef4248da4eea26b370b2cf4ba0.jpg) *(a) PLENA achieves higher utilization than the standard square systolic array with the same resources. (b) PLENA’s optimization pathways—(1) a flattened systolic array and (2) asymmetric quantization—together achieve improved effective memory bandwidth utilization and help reduce memory capacity limitations. Fig. 2: A comparison of attainable FLOPs between a squareshaped systolic array (e.g. in a TPU) and PLENA’s when using the same number of multipliers<sup>2</sup>.*

- **Memory Walls 约束**：长上下文任务导致 KV Cache 体积线性膨胀，迅速耗尽 HBM Capacity，迫使硬件只能运行极小的 Batch Size。
- **Fat GEMM 现象**：在 GEMM 运算 $(M, K) \times (K, N)$ 中，受限于极小 Batch Size，与 Batch 相关的维度 $M$ 远小于 Reduction 维度 $K$ 和输出维度 $N$，形成矩阵形状极不均衡的 **Fat GEMM**。
- **传统架构缺陷**：传统 TPU 或 GPU 采用 Square-shaped Systolic Array（如 128×128），设计假设 $M \approx K \approx N$。在处理 **Fat GEMM** 时，大量 Processing Elements (PEs) 处于空闲状态，Compute Utilization 极低。

---

**实现原理与微架构设计**

Flattened Systolic Array 通过重塑阵列形状与数据流，适配长上下文推理的矩阵特征。

![](images/dbc24c800aeaa4db7a40fa192c83d8dbb0be2c664a85685ae77163b16a575890.jpg) *Fig. 6: At each cycle, the flattened systolic array fetches two MLEN-wide inputs: one from the Matrix SRAM (top) and one from the Vector SRAM (left). The inputs are buffered and reordered, then partitioned into MLEN/BLEN subvectors, each of width BLEN. Each subvector is forwarded to a corresponding sub-array from the top and left directions. The scales and elements are streamed separately to each subarray. For improved resource efficiency, each PE consumes MX format inputs and performs accumulation in INT precision. The accumulated results are converted to the target activation precision before being written back to the Vector SRAM.*

- **Output-stationary Dataflow**：部分和保持静止在 PEs 中，Operands 沿着长 Reduction 维度 $K$ 流动，避免频繁的数据搬移。
- **Sub-array 组合架构**：整个阵列由一系列小型 Square-shaped Systolic Arrays（sub-arrs）构建。每个 PE 执行 Multiply–Accumulate (MAC) 操作，并向下、向右传递数据。
- **Cross-array Reduction**：单个 sub-array 仅产生部分和。为完成完整的 GEMM Tile，集成 **Result Adder Tree** 执行跨阵列求和。
- **Pipeline 优化**：阵列完全流水线化，消除连续 GEMM Tiles 之间的 Idling Bubbles。
- **Native MX Format Support**：PE 原生接收 **MX format** 数据输入，并在 INT 精度下执行累加，最终转换为 Target Activation Precision 后写回 Vector SRAM。

---

**算法流程与参数设置**

针对 Transformer 推理中计算最密集的 FFN 与 FlashAttention 两个模块，Flattened Systolic Array 采用了差异化的映射策略。

![](images/ee20e4180c087206c605b0575faec8ab33f2ae18629c149e83003895034cdb72.jpg) *Fig. 5: Processing flow for the weight–activation output stationary GEMM. Because memory capacity constrains batch size, the M dimension remains small. Setting BLEN = M on the flattened systolic array yields high utilization.*

- **FFN 层映射**：
  - 执行 $(BLEN, MLEN) \times (MLEN, BLEN)$ GEMM，输出形状为 $(BLEN, BLEN)$。
  - 令 $BLEN = M$（Batch Size），使阵列形状完美匹配极小的 Batch 维度，实现高 Utilization。
- **FlashAttention 模块映射**：
  - 阵列被划分为多个更小的 Flattened Array Cores，以支持 Per-head GEMM 并行计算。
  - 每个 Core 执行 $(BLEN, HLEN) \times (HLEN, BLEN)$ GEMM，并在 $(MLEN/HLEN)$ 个 Heads 上并行处理。
  - 解决了 Grouped Query Attention (GQA) 范式下 Head Dimension 较小导致的大规模阵列利用率低下的问题。
- **关键指令与参数**：
  - 跨阵列求和仅通过专用指令 **M_SUM** 触发，在沿大 Reduction 维度计算 GEMM 时仅需一次 Summation。
  - 核心设计参数与约束如下表所示：

| 参数 | 描述 | 搜索范围/约束 |
| :--- | :--- | :--- |
| **BLEN** | Block Unit 的 Tile Size | $[2, 4, \dots, 64]$ |
| **MLEN** | Matrix Unit 的 Tile Size | $[2, 4, \dots, 1024]$ |
| **HLEN** | Attention Head Dimension | $MLEN \ge HLEN \ge BLEN$ |
| **对齐约束** | 阵列划分对齐要求 | $MLEN \pmod{BLEN} = 0$ |

---

**输入输出关系与整体系统作用**

- **输入流**：
  - Matrix SRAM 提供宽度为 $MLEN$ 的权重或 KV Cache 数据流（MX format）。
  - Vector SRAM 提供宽度为 $MLEN$ 的 Activation 数据流。
  - 数据在进入 PE 前被 Buffer 重排，并划分为 $MLEN/BLEN$ 个宽度为 $BLEN$ 的 Sub-vectors，分别送入对应的 Sub-array。
- **输出流**：
  - 各 Sub-array 内 PE 产生的部分和，经 **Result Adder Tree** 跨阵列累加。
  - 最终结果转换为高精度浮点格式（如 FP16），写回 Vector SRAM。
- **在整体系统中的作用**：
  - 作为 PLENA 系统的 Pathway 1，Flattened Systolic Array 从微架构层面消除了长上下文 Agentic 任务中矩阵形状不均衡引发的算力浪费。
  - 配合 Pathway 2（Asymmetric Quantization 缓解 Capacity 限制）与 Pathway 3（Native FlashAttention 支持减少 Bandwidth 压力），共同突破了 Memory Walls 限制。
  - 在相同 Multiplier 数量和 Memory 配置下，使得 PLENA 在 Agentic Inference 中的算力利用率远超传统 A100 GPU 与 TPU v6e。

### 2. Asymmetric Quantization with Microscaling (MX) Co-design

**核心观点**

PLENA 提出了一种软硬件协同设计的**非对称量化策略**，通过将 Weights (W)、Activations (A) 和 KV Cache (KV) 设置为不同的精度，并结合 Post-Training Quantization (PTQ) 优化与 Microscaling (MX) 数据格式，有效缓解了长上下文 Agentic LLM 推理中的 memory bandwidth 和 capacity walls。

---

**Microscaling (MX) 数据格式与参数化**

PLENA 采用单级缩放的 Microscaling (MX) 数据格式，以平衡硬件复杂度与软件性能。每个数据块共享一个以 E8M0 格式编码的 power-of-two 缩放因子。

![](images/5f6878f7b16e75497c8fc15ffe1da3f648c6ec80ab70171ef99fff713e53935b.jpg) *Fig. 4: PLENA accelerator architecture overview. Execution is controlled by the decoder’s system-pipeline controller, which derives control signals from decoded instructions and monitors memory dependencies. For example, when reading from a Vector SRAM row that is still being updated by the vector or matrix unit, the controller inserts a stall to ensure correctness.*

- **参数化配置**：MX 格式由元组 $\tau = (d, b, B)$ 定义，其中 $d$ 为数据类型，$b$ 为位宽，$B$ 为微缩放块大小。
- **格式分类**：
  - **MXFP**：参数化为 $(M, E, S, B)$，支持 minifloat 数据类型。
  - **MXINT**：参数化为 $(M, S, B)$，支持整数数据类型。
- **量化映射**：对于高精度张量中的块 $w$，计算缩放因子 $s = \max|w| / \max_\tau$，并通过 $w_\tau = \text{clip}(\text{RTN}(w/s), \text{min}, \text{max})$ 进行量化映射。

---

**硬件层面的非对称数据通路**

为了支持非对称量化，PLENA 在硬件架构上对计算单元和存储单元进行了差异化设计。

![](images/6634593a23e73382f3753155fa97edecc275e8802c1818555b6db463e3110970.jpg) *Fig. 7: Data layouts and data paths for the memory system. Data with different MX precisions and datatypes are stored following a unified HBM storage pattern. A conversion to FP16 is performed as the data enter the Vector SRAM, which serves as the scratchpad for the vector unit; the vector unit operates in high-precision FP16. For the Matrix SRAM, MXformatted data loaded from HBM can be stored directly without additional conversion.*

- **Activations (A)**：对量化误差最敏感，以高精度 Floating-Point (FP) 格式存储在片上 Vector SRAM 中，并在 Vector Unit 中进行高精度计算。
- **Weights (W) 与 KV Cache (KV)**：对精度不敏感，采用激进的低精度 MX 格式存储在 Matrix SRAM 中，直接从 HBM 加载无需额外转换。
- **Outlier 抑制机制**：在将新的 K 和 V 向量追加到 HBM 的 KV cache 前，可选择性地应用基于 Hadamard 的 rotation 步骤抑制异常值，量化后存储；使用时在 Matrix SRAM 中应用逆 Hadamard 变换。

---

**PTQ 优化算法与流程**

PLENA 深入探索了 PTQ 优化技术与 MX 格式的兼容性，并提出了针对性的改进算法。

**1. Weight Quantization: Block-wise Clipping 与 Output-norm Guided GPTQ**
- **兼容性发现**：MXFP 与 PTQ 优化不兼容；MXINT 兼容，但直接应用会导致性能下降。
- **Microscaling Block-wise Clipping**：引入参数 $p \in [0.5, 0.99]$ 缩小经验范围，平衡 clipping overflow error 与 underflow error。
- **Output-norm Guided 搜索**：结合 GPTQ 框架，外层使用 Hessian 信息 $\mathbf{H}_F$ 迭代校准权重，内层优化以最小化输出块的量化误差为目标，而非仅最小化权重块误差。
  - 外层校准：$\delta_F = -(\mathbf{W}_b - \mathbf{Q}(\mathbf{W}_b; P_b^\star, \tau))([\mathbf{H}_F^{-1}]_{bb})^{-1}(\mathbf{H}_F^{-1})_{:,b}$
  - 内层搜索：$P_b^\star = \arg\min_{P_b \in \mathcal{P}^N} \|\mathbf{X}_b(\mathbf{W}_b - \mathbf{Q}(\mathbf{W}_b; P_b, \tau))^\top\|_2^2$

**2. Activation & KV Quantization: Selective Rotation**
- **兼容性发现**：直接将 QuaRot 的 Hadamard 变换应用于细粒度权重量化会增加 perplexity；但应用于 Activations 和 KV 时，因其数值范围更广，能有效缓解异常值。
- **Selective Rotation 策略**：遍历模型层集合 $\mathcal{M}$，仅选择能使性能提升的层子集 $S$ 应用旋转。
- **运行时支持**：PLENA 硬件原生支持运行时的 $\mathbf{H}^{-1}$ 乘法操作。

---

**软硬件协同设计搜索空间**

PLENA 构建了多保真度仿真框架，并使用多目标贝叶斯优化 进行设计空间探索。

| Parameter | Description | Search Range |
| --- | --- | --- |
| BLEN | Tile size of block unit | [2, 4, ..., 64] |
| MLEN | Tile size of Matrix Unit | [2, 4, ..., 1024] |
| VLEN | Tile size of Vector Unit | [2, 4, .., 1024] |
| M_LOAD | Matrix SRAM load amount from HBM | [2, 4, ..., 256] |
| V_LOAD | Vector SRAM load amount from HBM | [2, 4, ..., 256] |
| V_WRITE | Vector SRAM write amount to HBM | [2, 4, …., 256] |
| ACT_WIDTH | Activation precision | MXINT, MXFP |
| KV_WIDTH | Key/Value precision | MXINT, MXFP |
| FP_SETTING | Floating-point precision | FP |

- **优化目标**：同时优化 accuracy、latency 和 chip area，寻找 Pareto frontier。
- **典型配置输出**：
  - 高吞吐配置：ACT=MXINT_8, KV=MXINT_4, Lat=0.116s, Area=203.4 mm²
  - 低功耗配置：ACT=MXFP_E3M4, KV=MXFP_E3M4, Lat=0.166s, Area=26.45 mm²

---

**在整体架构中的作用与性能表现**

**输入输出关系与作用**：
- **输入**：高精度的 FP16 模型权重、Activations 以及长上下文产生的庞大 KV Cache。
- **处理**：通过协同设计框架，将 W/A/KV 映射为非对称的 MX 格式，并在 Flattened Systolic Array 上执行混合精度计算。
- **输出**：保持高精度的推理结果，同时大幅降低 HBM 的带宽压力和容量占用。

**性能表现**：
- **Memory Footprint 降低**：在 LLAMA-3.3-70B 的 OSWorld-L 工作负载下，将 16/16/16 量化降至 4/4/4，KV Cache 占用从 239.26 GB 降至 59.81 GB，Weight Storage 从 129.46 GB 降至 32.36 GB。
- **精度保持**：在 LLAMA-3-8B 的 W4A4KV4 设置下，PLENA 的量化策略实现了 7.17 的 WikiText-2 Perplexity，显著优于 QuaRot 的 8.16 和直接应用 MXFP 的 256.22。
- **系统级收益**：通过释放 HBM 容量，支持了更大的 batch size，结合 Flattened Systolic Array，在 LLaMA agentic inference 中实现了比 A100 GPU 高 2.23 倍的吞吐量和 4.04 倍的能效。

### 3. Native Hardware Support for FlashAttention

**核心观点**
PLENA 通过定制化指令集架构（ISA）与底层微架构的协同设计，原生支持 FlashAttention 算法。此设计打破了传统基于 Systolic Array 的加速器在处理长上下文时面临的 Memory Bandwidth 与 Capacity 限制，通过 Tile 级别的计算与访存重叠、片上融合计算，大幅减少了 Off-chip Memory Traffic。

---

**FlashAttention 算法背景与瓶颈**
- 标准 Attention 机制在计算 QK^T 时会产生巨大的方阵，其规模通常达到数千乘数千。
- 由于 On-chip Memory 容量受限，中间结果必须写回 Off-chip Memory，并在后续 Softmax 与 PV 计算阶段重新加载，引发严重的读写瓶颈。
- FlashAttention 采用 IO-aware 策略，通过 Tiling 与 Fusing 将 GEMM-Softmax-GEMM 流水线融合，使中间结果完全保留在片上，从而规避昂贵的 Off-chip 往返。

---

**原生硬件支持的核心架构设计**
传统加速器无法原生支持 FlashAttention，主要受限于四个维度的缺失。PLENA 针对这些挑战提出了针对性的硬件与指令集优化：

- **挑战 1：缺乏 Tile 级别的 Memory Prefetching 重叠**
  - **PLENA 方案**：集成硬件 Prefetch Engine 于 Matrix SRAM 与 Vector SRAM 中。
  - **执行逻辑**：在计算单元执行当前 Tile 计算的同时，背景预取下一 Tile 的数据，实现 Instruction-level 的访存与计算重叠，隐藏 HBM Latency。

- **挑战 2：缺乏 Memory Layout 支持（如 Transpose-on-read）**
  - **PLENA 方案**：设计定制化的 Transposable Matrix SRAM。
  - **执行逻辑**：将逻辑行分布至多个 sub-SRAM banks 中，使得行访问与列访问映射至独立的 banks。无需显式的数据重排，即可在无 Bank Conflict 的情况下并行执行转置与非转置访问。
  - **性能指标**：相比传统 Transpose-buffer-plus-SRAM 基线，面积减少 **65.17%**，且仅需 **2 个周期** 完成 Transpose Read。

![](images/9716793725b3dd382ce08b4f94f316afb23addc4a899c8c4049c1f4b313f3953.jpg)

- **挑战 3：缺乏 In-line Reductions 与非线性操作支持**
  - **PLENA 方案**：配置 Vector Unit 与 Scalar Unit 执行 Softmax 所需的 max/sum, exp, div 等操作。
  - **参数设置**：Vector width (VLEN) 可配置，以精准匹配 FlashAttention 使用的 Tile dimensions。计算精度可配置，通常设为高精度（如 **FP12**）以保留 Softmax 计算期间的数值精度。

- **挑战 4：固定调度与粗粒度 Kernel 边界限制**
  - **PLENA 方案**：设计 Custom ISA 提供可组合的细粒度控制。
  - **执行逻辑**：支持 Persistent, tile-by-tile scheduling 的融合 Attention 流水线，允许 FlashAttention 的每个阶段在 Tile 粒度上被独立编排。

![](images/0a1e89cbb8843ff9d2ac243ec093ac3cc7bd086c504f4f21c4d15772b0964846.jpg)

---

**Flattened Systolic Array 在 FlashAttention 中的作用**
- 在长上下文场景中，FlashAttention 需要执行 Per-head Fat GEMMs。由于 Head Dimension 通常较小（如 LLAMA-3-70B 的 128），且 Grouped Query Attention (GQA) 范式要求每个 Key Head 被多个 Query Head 乘以，传统大规模方形阵列利用率极低。
- PLENA 将 Flattened Systolic Array 分区为多个更小的 Flattened Array Cores。
- 每个 Core 执行 (BLEN, HLEN) × (HLEN, BLEN) GEMM，并跨 (MLEN/HLEN) Heads 并行处理，从而在长上下文且有效 Batch Size 较小的情况下维持极高的计算资源利用率。

---

**输入输出关系与系统级作用**
- **输入数据**：从 HBM 加载的 MX 格式化的 Query (Q), Key (K), Value (V) 张量。
- **片上流转**：K 与 V 直接加载至 Matrix SRAM，若应用了 Hadamard Rotation 抑制异常值，则在 Matrix SRAM 内执行 Inverse Hadamard Transform。Q 存储于 Vector SRAM。
- **计算输出**：Attention Score 直接在片上完成 GEMM-Softmax-GEMM 全流程，最终输出写回 Vector SRAM，无需将中间激活写回 HBM。
- **整体作用**：在 Agentic LLM Inference 中，随着 Context Length 增加，Attention 计算逐渐主导 FLOPs。原生 FlashAttention 支持避免了中间结果对 HBM 带宽的挤占，配合 Memory Prefetching，使得计算单元在 Prefill 与 Decode 阶段均保持高利用率。

![](images/b3d443e20eb0ea5dfeee201191d56d0ad2956e51430a7bcc2dcdcbce75503f86.jpg)

---

**架构性能对比**
PLENA 的原生支持在面积效率与可达算力上显著优于现有重构方案：

| Design | Comp Area (mm²) | TOPs/mm² | S.A.T/mm²* (Standard) | A.A.T/mm²* (Agentic) |
| :--- | :--- | :--- | :--- | :--- |
| MicroscopiQ | 0.1378 | 59.45 | 26.36 | 5.83 |
| SystolicAttn | 1.17 | 14.00 | 7.14 | 4.76 |
| **PLENA** | **0.237** | **34.49** | **29.31** | **12.81** |
*(注：S.A.T 为标准任务可达算力，A.A.T 为 Agentic 任务可达算力)*


---

## 4. 实验方法与实验结果

**实验设置**

实验评估涵盖软件量化效果与硬件系统性能两个维度，具体设置如下：

*   **模型与数据集**
    *   **模型**：LLaMA-2, LLaMA-3, GPT-OSS (MoE), Qwen3。
    *   **评估指标**：WikiText-2 的 perplexity、6 个下游任务的 zero-shot accuracy、长上下文与 Agentic 工作负载（HumanEval 代码生成、GSM8K-Platinum 数学推理、BFCL-Web Search Base 工具调用）。
*   **量化基线**
    *   对比对象包括 GPTQ, OmniQuant, QuaRot, Atom, MicroScopiQ 等 SoTA 方法。
*   **硬件实现与基线**
    *   **PLENA 实现**：使用 SystemVerilog RTL 实现，基于 7nm OpenROAD PDK 在 1 GHz 频率下进行综合。
    *   **硬件基线**：重新实现 MicroScopiQ, FIGNA, SystolicAttention, Olive 核心组件并集成至 PLENA 平台。商业平台对比包括 A100 (80GB), H100 (80GB) 以及 TPU v6e-8。

---

**量化结果与消融实验**

PLENA 提出基于 Microscaling (MX) 数据格式的非对称量化方案，通过消融实验验证了 **Block-wise Clipping** 与 **Selective Rotation** 的有效性。

*   **主实验结果**
    *   在 W4A16KV16, W4A4KV16, W4A4KV4 三种精度设置下，PLENA 在 LLaMA 系列模型上的 WikiText-2 perplexity 均匹配或超越现有 SoTA 方法。
    *   在 Zero-shot 下游任务中，W4A4KV4 配置下的 PLENA 在所有任务上的准确率均显著优于 QuaRot。

<table>
    <tr>
        <td>Method</td>
        <td>W/A/KV</td>
        <td>LLaMA-3-8B PPL↓</td>
        <td>LLaMA-3-70B PPL↓</td>
    </tr>
    <tr>
        <td>Baseline</td>
        <td>16/16/16</td>
        <td>6.13</td>
        <td>2.85</td>
    </tr>
    <tr>
        <td>QuaRot</td>
        <td>4/4/4</td>
        <td>8.16</td>
        <td>6.66</td>
    </tr>
    <tr>
        <td>PLENA (ours)</td>
        <td>4/4/4</td>
        <td>7.17</td>
        <td>4.09</td>
    </tr>
</table>

*   **消融实验分析**
    *   **数据类型选择**：**MXINT4** 在所有设置下性能均优于 **MXFP4**，因此被确立为权重和后续评估的默认数据类型。
    *   **Rotation 策略**：将 Rotation 应用于权重会显著增加 perplexity，因为权重的动态范围较小，共享指数已足够捕获异常值。而 Selective Rotation 应用于 Activation 和 KV Cache 时，能有效抑制异常值，提升量化精度。
    *   **Clipping 优化**：基于 Output-norm guided 的 block-wise clipping (Err_y) 相比 Weight-norm guided clipping (Err_w) 取得了更低的 perplexity，证明了以输出重建误差为优化目标的优越性。

![](images/7ce55307e20f9c72d1468aa988f884af1a983f4f18e7317e64acaf12bb9d6a5b.jpg)

*   **Agentic 工作负载验证**
    *   在 Qwen3-32B 上的 W4A4KV4 量化测试中，PLENA 在 HumanEval, GSM8K-Platinum, BFCL-W 等长上下文与 Agentic 任务上保持了极高的性能保留率。

---

**硬件架构与系统性能分析**

PLENA 硬件系统的核心在于 **Flattened Systolic Array** 与原生 **FlashAttention** 支持，以打破 Memory Wall 限制。

*   **计算利用率与面积效率**
    *   传统方形 Systolic Array 在处理 Agentic 推理中的 Fat GEMM 时利用率低下。PLENA 的 Flattened Systolic Array 在 FFN 和 FlashAttention 阶段均实现了极高的计算资源利用率。
    *   在同等乘法器资源下，PLENA 的 Agentic Attainable TOPs (A.A.T/mm2) 达到 12.81，远超 MicroScopiQ (5.83) 和 SystolicAttention (4.76)。

![](images/6269938e2b2a9ec88b0f3d184d3993efb221cf947cafaefd5143520c26b71af2.jpg)

*   **系统级性能对比**
    *   在 LLaMA-3.3-70B 的 Agentic 推理中，PLENA 实现了高达 **2.23x** 于 A100 和 **4.70x** 于 TPU v6e 的吞吐量。
    *   能效比方面，PLENA 达到 **4.04x** 于 A100 的 Token/J。这主要归功于 4-bit 非对称量化大幅降低了 HBM 带宽压力与 KV Cache 容量占用，以及 Flattened Systolic Array 带来的高计算利用率。

<table>
    <tr>
        <td>System</td>
        <td>Workload (Prefill, Output)</td>
        <td>TPS (×A100)</td>
        <td>Tok/J (×A100)</td>
        <td>BS</td>
    </tr>
    <tr>
        <td>A100</td>
        <td>(90k, 8k)</td>
        <td>1.00x</td>
        <td>1.00x</td>
        <td>4</td>
    </tr>
    <tr>
        <td>H100</td>
        <td>(90k, 8k)</td>
        <td>2.04x</td>
        <td>1.22x</td>
        <td>8</td>
    </tr>
    <tr>
        <td>TPU v6e</td>
        <td>(90k, 8k)</td>
        <td>0.47x</td>
        <td>N/A</td>
        <td>4</td>
    </tr>
    <tr>
        <td>PLENA</td>
        <td>(90k, 8k)</td>
        <td>2.21x</td>
        <td>4.04x</td>
        <td>16</td>
    </tr>
</table>

*   **时序分解与优化效果**
    *   时序分析表明，PLENA 的硬件架构在 Prefill 和 Decode 阶段均维持了高利用率和内存带宽利用率。
    *   原生 **FlashAttention** 支持将大型中间 Activation 保留在片上，避免了昂贵的 Off-chip Memory 往返。结合内存预取机制，有效隐藏了数据访问延迟。

![](images/f1332d04b6176986c4ea4cc64aaab80c72e6248f0b288a5b9dc9a1134c5dfd67.jpg)

---

