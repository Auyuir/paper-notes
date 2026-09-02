# LLMCompass 施引文献调研与过滤

> 调研对象：LLMCompass (ISCA'24, Zhang et al., Princeton) 的被引文献
> 调研目标：从施引文献中筛选出**实质性利用 LLMCompass 做建模/仿真**、且**有影响力（顶会发表）**的工作，了解它们如何克服或绕过 LLMCompass 的局限性（测试集规模小、单测例覆盖度不佳）。

## 方法

1. **数据源（双源并集）**：
   - Semantic Scholar：`/paper/{id}/citations`，拉取 **91** 篇施引文献（含 citation contexts、intents、venue、abstract、isInfluential、externalIds）。
   - OpenAlex：`filter=cites:W4401211642`，拉取 **56** 篇；与 S2 去重后得 **18 篇 delta**（S2 漏掉的引文边，如 SMOOTH）。delta 无引文上下文，用 S2 by-DOI 补 abstract/tldr 后判定。
   - 合计去重后约 **109** 篇唯一施引文献。
2. **过滤**：S2 91 篇分 5 路、OpenAlex 18 篇 delta 分 2 路并行 subagent 逐篇判定。
3. **判据（已拧紧）**：KEEP 仅限**明确运行/扩展 LLMCompass 做仿真**（摘要或上下文出现 "we use/extend/integrate LLMCompass to simulate/evaluate/model..."）；**仅引用其结论/数据**（即使不在 related work 段）= 排除。
4. **标定验证**：
   - ✅ `SMOOTH` (ISCA'26, 集成 LLMCompass 做周期精确评估) → 保留（OpenAlex delta 发现，S2 漏）
   - ✅ `WSC-LLM` (ISCA'25, related-work 轻提) → 排除
   - ⚠️ `Combating the Memory Walls` (ISCA'26) → **初判误留**（误把"引用结论"当"实质仿真"），人工复核重读全文后**改判排除**。此案例暴露"仅凭引文上下文一句话"区分"用工具"与"引结论"不可靠，已据此拧紧判据。

## ⚠️ 覆盖缺口（重要）

S2 的 91 篇 < GS 的 129 篇，且 S2 漏掉了 OpenAlex 有的 **18 条引文边**（含 SMOOTH）。根因经查证：S2 建引文边依赖参考消歧，**有 arXiv 预印本 + OA PDF** 的论文（如 Combating）被 S2ORC 全文解析干净建边；**无 arXiv、PDF 对 S2 CLOSED** 的论文（如 SMOOTH）只能靠 Crossref 存款，消歧失败率高、易漏边。OpenAlex 对 IEEE 存款的参考解析更鲁棒，故能补上 SMOOTH 等边。**结论：单靠 S2 会漏，必须 S2+OpenAlex 并集**。本清单仍为下限，GS/arXiv 全文检索可再补。

## 筛选结果概览

- **保留 (KEEP)**：17 篇（实质利用 16 + 绕过对照 1）
- **边界 (MAYBE)**：~21 篇（S2 侧 12 + OpenAlex delta 9，多数因无引文上下文、仅凭摘要无法确认是否真用了 LLMCompass）
- **排除 (EXCLUDE)**：~71 篇（related-work 仅提及 / 引用结论 / 领域不匹配 / 水平低）

---

## 一、扩展 LLMCompass 作为核心组件（extends）

这类工作直接修改/集成 LLMCompass 作为自身仿真后端，是"实质性继承"最强的一档。

### [SMOOTH] SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference  ★标定样例
- **发表**：ISCA'26 | [DOI:10.1109/ISCA66397.2026.00057](https://doi.org/10.1109/ISCA66397.2026.00057) | cites=0（新）
- **如何用**：摘要明示——"We implement SMOOTH in Verilog and **integrate it into LLMCompass**, an LLM-optimized extension of ScaleSim, for cycle-accurate evaluation."
- **克服局限**：在 LLMCompass 之上做周期精确评估，验证 SMOOTH 的运行时 scratchpad 管理缓解自回归解码的突发流量/碎片（TTFT −59.2%、TTLT −73.0%、能耗 −51.2%）。
- **数据源说明**：S2 漏了这条引文边（无 arXiv 版、PDF 对 S2 CLOSED，靠 Crossref 存款消歧失败），由 OpenAlex 补回。本仓库已收录。
- **摘要**：面向端侧 LLM 推理的硬件辅助细粒度片上内存管理框架。

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

### ~~[54] Combating the Memory Walls (PLENA)~~ — ⚠️ 改判排除
- **发表**：ISCA'26 | [DOI:10.1109/ISCA66397.2026.00023](https://doi.org/10.1109/ISCA66397.2026.00023) / [arXiv:2509.09505](https://arxiv.org/abs/2509.09505) | cites=13
- **改判原因**：经人工重读全文，该文**仅引用 LLMCompass 的结论**（关于 prefill/decode 硬件需求的启示），**并未用 LLMCompass 做仿真**——PLENA 自带 transaction-level 模拟器 + 自动 DSE 流程。初判因引文上下文含 PLENA 描述而误判为 "uses"，已据此教训拧紧判据。归入排除（结论引用型）。本仓库已收录该文笔记，仅供对照。

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

S2 侧（有引文上下文，但仍难定夺）：

| idx | 标题 | 会议 | 不确定原因 |
|----|------|------|-----------|
| 33 | LUMINA: LLM-Guided GPU Architecture Exploration | arXiv | 领域对，contexts 空，疑似基线未证实 |
| 72 | ADOR | ISPASS'25 | 同 DSE 领域，contexts 空 |
| 79 | DFModel | arXiv | 点名 LLMCompass"无法建模片内 dataflow 映射"，但仅一句对比 |
| 89 | LLM Inference on Chiplet-based Architectures | 未索引 | 明确"leverages LLMCompass"但无会议/摘要/cite |
| 19 | CCL-Bench 1.0 | arXiv | 直击"小测试集"痛点，但仅 list 引用 LLMCompass |
| 53 | SnipSnap | ASP-DAC | 引用 LLMCompass 的评测协议，主体为 list-mention |

OpenAlex delta 侧（S2 漏的边，无引文上下文，仅凭摘要，更不确定）：

| 来源 | 标题 | 会议 | 不确定原因 |
|----|------|------|-----------|
| OA-d3 | LP-Spec: LPDDR PIM for LLM Mobile Speculative Inference | ICCAD'25 | 顶会+领域对，摘要未点名 LLMCompass，需全文核验是仿真后端还是 related-work |
| OA-d4 | Optically Connected Multi-Stack HBM Modules | IEEE CAL'25 | HBM for LLM，自称"A100 modeled baseline"，需核验是否基于 LLMCompass |
| OA-d5 | SuperMesh: Collective Communications for Accelerators | SC/ASPLOS 系'25 | 加速器集合通信，非 LLM 专向，疑似 workload/related 提及 |
| OA-d8 | EONSim: NPU Simulator for On-Chip Memory | IEEE CAL'26 | 自建 NPU 模拟器，疑似把 LLMCompass 列为 prior simulator |
| OA-d11 | Wafer-Scale GPU Memory Pool w/ In-Package Optics | IEEE CAL'26 | 自建"HBM-pool"分析框架，未点名 LLMCompass |
| OA-d15 | HydraPIM: Heterogeneous PIM for Attention | IEEE TC'26 | PIM 长上下文 LLM，对比 PIM 基线，需核验 LLMCompass 角色 |
| OA-d16 | BlockPIM: Sparse LLM Inference on Dense PIM | IEEE CAL'26 | 与 DAC'25 [60] 同组疑似续作，需核验是否复用 LLMCompass |

> **去重提示**：OA-d8 EONSim 与 S2 侧 [46] 疑为同篇（标题一致）但 DOI/年份形式差异导致自动去重未命中——人工合并时留意，勿重复计数。这暴露按 DOI+norm(title)+year 去重在版本/年份不一致时仍会漏。

---

## 六、局限性对策小结

| LLMCompass 局限 | 施引文献的应对 |
|----------------|---------------|
| 测试集小、单测例覆盖差 | 多数 extends 类工作拓宽了**算子/硬件/模型**覆盖（SPAD 加 H100、BlockPIM 加算子、MoE-GPS 加 MoE、MLDSE 加多级 IR），但**真正扩充 benchmark 工作负载集**的很少；[39] 甚至自承工作负载仍窄 |
| 仿真慢 | ReaLLM（预计算 kernel 库，6×/164× 加速）、AMALI（改用解析模型绕过） |
| 带宽建模理想化 | DeepStack（bank 级真实带宽）、DFModel（片内 dataflow 映射，MAYBE） |
| 无网络建模 | 未见施引文献补该缺口（LLMCompass 自身建议集成 ASTRA-sim） |

**总体判断**：LLMCompass 被作为"快速 DSE 后端"广泛复用（extends/use 两类共 13 篇，含 SMOOTH 这类在 LLMCompass 之上做周期精确扩展的 ISCA 工作），其"小测试集"局限**基本未被正面解决**——这恰是你这篇 LLMCompass 综述/批判可切入的空白。注意 Combating the Memory Walls 一类"引用结论而非用工具"的论文会被误判，须以全文复核兜底。
