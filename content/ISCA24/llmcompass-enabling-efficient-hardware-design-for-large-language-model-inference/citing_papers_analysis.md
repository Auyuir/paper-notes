# LLMCompass 施引文献调研与过滤

> 调研对象：LLMCompass (ISCA'24, Zhang et al., Princeton) 的被引文献
> 调研目标：从施引文献中筛选出**实质性利用 LLMCompass 做建模/仿真**、且**有影响力（顶会发表）**的工作，了解它们如何克服或绕过 LLMCompass 的局限性（测试集规模小、单测例覆盖度不佳）。

## 方法

1. **数据源**：Semantic Scholar Academic Graph API，`/paper/{id}/citations`，一次性拉取全部 **91** 篇施引文献的完整元数据（citation contexts、intents、venue、abstract、isInfluential、citationCount、externalIds）。
2. **过滤**：5 路并行 subagent 逐篇判定，分类维度 = `extends / uses / baseline-compares / background-mention` × `venue tier / influential`。
3. **标定验证**：
   - ✅ `Combating the Memory Walls` (ISCA'26, 实质仿真) → 正确保留
   - ✅ `WSC-LLM` (ISCA'25, related-work 轻提) → 正确排除

## ⚠️ 覆盖缺口（重要）

Semantic Scholar 的 91 篇 **少于** Google Scholar 的 129 篇。已发现一处遗漏：**SMOOTH (ISCA'26)** 在本仓库中明确"集成进 LLMCompass（ScaleSim 的 LLM 扩展）"做仿真，但**不在 S2 的 91 篇内**。因此本清单是**下限而非全集**——如需完备，应以 GS/SerpAPI 或 arXiv 全文检索交叉补漏。

## 筛选结果概览

- **保留 (KEEP)**：16 篇（实质利用 LLMCompass）
- **边界 (MAYBE)**：~12 篇（证据不足或仅部分相关）
- **排除 (EXCLUDE)**：~63 篇（related-work 仅提及 / 领域不匹配 / 水平低）

---

## 一、扩展 LLMCompass 作为核心组件（extends）

这类工作直接修改/集成 LLMCompass 作为自身仿真后端，是"实质性继承"最强的一档。

### [52] SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference
- **发表**：arXiv 预印本 (2025) | [arXiv:2510.08544](https://arxiv.org/abs/2510.08544) | cites=9, influential
- **如何用**："We extended LLMCompass to support H100 modeling and new models"——所有 chip 延迟与设计空间探索跑在扩展版 LLMCompass 上。
- **克服局限**：扩展了硬件覆盖（加 H100）与模型覆盖；并加入 production-trace 端到端集群评估。
- **摘要**：less-is-more 思路为 prefill/decode 分别定制专用芯片。

### [57] ReaLLM: A Trace-Driven Framework for Rapid Simulation of Large-Scale LLM Inference
- **发表**：ASAP'25 | [DOI:10.1109/ASAP65064.2025.00022](https://doi.org/10.1109/ASAP65064.2025.00022) | cites=4, influential
- **如何用**：ReaLLM 的 kernel simulator 直接构建在 LLMCompass 之上。
- **克服局限**：**直接攻 LLMCompass 的"慢"**——"simulating each Matmul still takes a minute"；用预计算 kernel 库实现 6×/164× 加速，并叠加 trace 驱动的系统级执行分析（覆盖 batching/scheduling 行为，LLMCompass 所缺）。
- **摘要**：在 LLMCompass 微架构 kernel 建模之上叠加系统级评估的 trace 驱动模拟器。

### [41] RACAM: Enhancing DRAM with Reuse-Aware Computation and Automated Mapping for ML Inference
- **发表**：arXiv (2025) | [arXiv:2512.09304](https://arxiv.org/abs/2512.09304) | cites=0, influential
- **如何用**："Our LLM Parser is built on top of LLMCompass"；并用其评估 H100 系统延迟。
- **克服局限**：未直接扩展测试集。
- **摘要**：首个 in-DRAM 比特串行架构 + 自动映射，LLM 推理较 GPU 提升 9–102×。

### [66] MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing
- **发表**：arXiv (2025) | [arXiv:2506.07366](https://arxiv.org/abs/2506.07366) | cites=5, influential
- **如何用**："Performance is simulated with an augmented version of LLMCompass"。
- **克服局限**：将 LLMCompass 扩展到 MoE expert-duplication 的运行时权衡建模，拓宽工作负载覆盖。
- **摘要**：指导 MoE 预测器选择（Mixtral 8x7B 上 +23%）。

### [60] BlockPIM: Optimizing Memory Management for PIM-enabled Long-Context LLM Inference
- **发表**：DAC'25 | [DOI:10.1109/DAC63849.2025.11133193](https://doi.org/10.1109/DAC63849.2025.11133193) | cites=3
- **如何用**："We extended the supported operators of LLMCompass to simulate this overhead"。
- **克服局限**：扩展算子覆盖以仿真 PIM 开销。
- **摘要**：PIM 长上下文 LLM 推理的跨通道块内存布局与 attention 方案。

### [7] DOPS: Beyond Prefill-Decode Disaggregation (Dynamic Operator Scheduling)
- **发表**：arXiv (2026) | [arXiv:2607.25498](https://arxiv.org/abs/2607.25498) | cites=0, influential
- **如何用**："DOPS supports LLMCompass and our in-house Ascend 910B simulator"——作为仿真后端集成。
- **克服局限**：未直接触及测试集；聚焦调度/布局。
- **摘要**：异构 NPU/PIM 上硬件感知的闭环算子调度 + 权重布局仲裁器。

---

## 二、使用 LLMCompass 做仿真/评估（uses）

这类工作直接运行 LLMCompass 做实验，但不改其代码。

### [54] Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference (PLENA)  ★标定样例
- **发表**：ISCA'26 | [DOI:10.1109/ISCA66397.2026.00023](https://doi.org/10.1109/ISCA66397.2026.00023) / [arXiv:2509.09505](https://arxiv.org/abs/2509.09505) | cites=13
- **如何用**：基于 LLMCompass 仿真做 PLENA 的评估/协同设计。
- **克服局限**：未扩展 LLMCompass 测试集——PLENA 自带 transaction-level 模拟器 + 自动 DSE 流程。
- **摘要**：长上下文 agentic LLM 推理的硬软件协同设计加速器（扁平脉动阵列、非对称量化、FlashAttention）。本仓库已收录。

### [62] Chip Architectures Under Advanced Computing Sanctions
- **发表**：ISCA'25 | [DOI:10.1145/3695053.3731012](https://doi.org/10.1145/3695053.3731012) | cites=3, influential
- **如何用**："we use LLMCompass to explore how different hardware designs affect LLM inference"——作为架构设计空间探索框架。
- **克服局限**：探索大范围合规芯片配置（多种 array/cache/bandwidth 组合）而非小固定集，**部分回应"测试集小"**。
- **摘要**：首个计算出口管制对 LLM 推理芯片的架构/经济性研究。

### [61] LLMShare: Optimizing LLM Inference Serving with Hardware Architecture Exploration
- **发表**：DAC'25 | [DOI:10.1109/DAC63849.2025.11132534](https://doi.org/10.1109/DAC63849.2025.11132534) | cites=3
- **如何用**："We use LLMCompass for cost and latency simulation"，并借鉴其通用设备模板描述 H100/MI210/TPUv3。
- **克服局限**：跨 prefill/decode 硬件配置的设计空间探索（部分拓宽覆盖）。
- **摘要**：对齐硬件到 prefill/decode 需求的模拟器+DSE 框架，省 13% 成本、>4× 吞吐。

### [75] NVR: Vector Runahead on NPUs for Sparse Memory Access
- **发表**：DAC'25 | [DOI:10.1109/DAC63849.2025.11132724](https://doi.org/10.1109/DAC63849.2025.11132724) / [arXiv:2502.13873](https://arxiv.org/abs/2502.13873) | cites=2
- **如何用**："we leverage LLMCompass, a specialised simulator designed to evaluate hardware optimisations in LLM inference"——作为评估模拟器。
- **克服局限**：未扩展测试集。
- **摘要**：NPU 向量 runahead 预取机制，降低稀疏 DNN 工作负载的 cache miss。

### [38] Enabling Cost-Efficient LLM Inference on Mid-Tier GPUs With NMP DIMMs
- **发表**：IEEE Computer Architecture Letters (2026) | [DOI:10.1109/LCA.2025.3646622](https://doi.org/10.1109/LCA.2025.3646622) | influential
- **如何用**："We use the LLMCompass simulator for performance characterization and sensitivity studies"；按 LLMCompass 的 A100 实测数据 scale 到 A10。
- **克服局限**：未扩展测试集。
- **摘要**：为中端 GPU（A10/A100）加 NMP DIMM，提升 LLM 推理性价比。

### [39] Prefill vs. Decode Bottlenecks: SRAM-Frequency Tradeoffs and the Memory-Bandwidth Ceiling
- **发表**：arXiv (2025) | [arXiv:2512.22066](https://arxiv.org/abs/2512.22066) | cites=1, influential
- **如何用**："We use LLMCompass to simulate the latency of the matrix multiplications of an LLM runtime, for each frequency"，配合 OpenRAM/ScaleSIM。
- **克服局限**：扩展参数扫描（SRAM 16KB–256KB、频率、buffer），但**工作负载仍窄**（GPT-3 单层、batch 8）——作者自己也未解决覆盖度问题。
- **摘要**：仿真 SRAM 大小/频率权衡，定位 prefill vs decode 的能效甜点。

### [69] MLDSE: Scaling Design Space Exploration Infrastructure for Multi-Level Hardware
- **发表**：arXiv (2025) | [arXiv:2503.21297](https://arxiv.org/abs/2503.21297) | cites=1, influential
- **如何用**："Area evaluation used LLMCompass and CACTI"——LLMCompass 作为面积/延迟模型喂入 MLDSE 的多级 DSE。
- **克服局限**：引入递归硬件 IR + 通用模拟器生成，探索 LLMCompass 预定义模板之外的多级空间层次，**拓宽设计空间覆盖**。
- **摘要**：多级 DSE 基础设施（硬件 IR、时空映射 IR、事件驱动模拟器生成）。

---

## 三、作为基线/对比（baseline / compares）

### [37] PipeWeave: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
- **发表**：ISCA'26 | [DOI:10.1109/ISCA66397.2026.00126](https://doi.org/10.1109/ISCA66397.2026.00126) / [arXiv:2601.14910](https://arxiv.org/abs/2601.14910) | cites=2
- **如何用**：将 LLMCompass 作为"高度详细的建模范式"基线之一对比。
- **克服局限**：跨 11 款 GPU / 4 代架构泛化（拓宽硬件覆盖）。
- **摘要**：分析+ML 混合 GPU 性能预测器，在 kernel/端到端误差上击败 SOTA。

### [28] DeepStack: Co-Design Exploration of 3D DRAM-Stacked Accelerators for Distributed LLM Inference
- **发表**：arXiv (2026) | [arXiv:2604.04750](https://arxiv.org/abs/2604.04750) | influential
- **如何用**：在对比表中与 LLMCompass 并列，并批评其"assume idealized bank interleaving… overestimate achievable bandwidth"。
- **克服局限**：针对 LLMCompass 的带宽建模假设（理想化 interleaving）而非测试集问题。
- **摘要**：3D-DRAM 堆叠加速器协同设计，带真实 bank 级带宽建模。

### [42] LaMoSys3.5D（3.5D-IC LLM-serving 架构）
- **发表**：arXiv (2025) | cites=2
- **如何用**：将 LLMCompass 提议的设计（LC-L、LC-T）作为对比平台，并引用其通信模型校准。
- **克服局限**：未直接扩展测试集。
- **摘要**：首个可扩展 3.5D-IC LLM serving 架构（dataflow、并行映射、热感知 DSE）。

---

## 四、绕过 LLMCompass 局限（analytical alternative）

### [65] AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs
- **发表**：ISCA'25 | [DOI:10.1145/3695053.3731064](https://doi.org/10.1145/3695053.3731064) | cites=11
- **如何用**：**不运行** LLMCompass，但**直接以其局限为动机**——"Zhang et al. introduced LLMCompass… but it still faces the challenge of long simulation time"。
- **克服局限**：用解析模型**绕过**仿真慢的问题（与 ReaLLM 的"加速仿真"路线互补）。
- **摘要**：现代 GPU 上 LLM 推理的精确解析建模，定位为 LLMCompass 仿真路线的替代。
- **判定**：MAYBE→保留为"绕过局限"的重要对照案例。

---

## 五、边界候选（MAYBE，未深读，供人工二次判定）

| idx | 标题 | 会议 | 不确定原因 |
|----|------|------|-----------|
| 33 | LUMINA: LLM-Guided GPU Architecture Exploration | arXiv | 领域对，contexts 空，疑似基线未证实 |
| 72 | ADOR | ISPASS'25 | 同 DSE 领域，contexts 空 |
| 79 | DFModel | arXiv | 点名 LLMCompass"无法建模片内 dataflow 映射"，但仅一句对比 |
| 89 | LLM Inference on Chiplet-based Architectures | 未索引 | 明确"leverages LLMCompass"但无会议/摘要/cite |
| 19 | CCL-Bench 1.0 | arXiv | 直击"小测试集"痛点，但仅 list 引用 LLMCompass |
| 53 | SnipSnap | ASP-DAC | 引用 LLMCompass 的评测协议，主体为 list-mention |

---

## 六、局限性对策小结

| LLMCompass 局限 | 施引文献的应对 |
|----------------|---------------|
| 测试集小、单测例覆盖差 | 多数 extends 类工作拓宽了**算子/硬件/模型**覆盖（SPAD 加 H100、BlockPIM 加算子、MoE-GPS 加 MoE、MLDSE 加多级 IR），但**真正扩充 benchmark 工作负载集**的很少；[39] 甚至自承工作负载仍窄 |
| 仿真慢 | ReaLLM（预计算 kernel 库，6×/164× 加速）、AMALI（改用解析模型绕过） |
| 带宽建模理想化 | DeepStack（bank 级真实带宽）、DFModel（片内 dataflow 映射，MAYBE） |
| 无网络建模 | 未见施引文献补该缺口（LLMCompass 自身建议集成 ASTRA-sim） |

**总体判断**：LLMCompass 被作为"快速 DSE 后端"广泛复用（extends/use 两类共 13 篇），其"小测试集"局限**基本未被正面解决**——这恰是你这篇 LLMCompass 综述/批判可切入的空白。
