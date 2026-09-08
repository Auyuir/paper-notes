# LLMCompass 被引文献：Vector/Matrix 重叠仿真调研（阶段 1）

> **状态**：候选筛选和已取得材料的复核已完成；以下不是“已经证明存在目标工作的最终结论”。A/B 类候选仍需全文、配置或代码交叉验证。全文仅使用已有本地材料和公开免费入口，不使用付费下载或 OpenAlex content/PDF 配额。

## 1. 调研问题与严格命中标准

目标是从 LLMCompass 的被引文献中找出同时满足下列条件的仿真器工作：

1. **确实运行或扩展了仿真器**，而不是只在 related work 中引用 LLMCompass；
2. **实际运行 Attention/Transformer workload**，最好包括 FlashAttention 或分 tile attention；
3. **把 Vector Unit 与 Matrix/Systolic/Tensor Unit 作为独立资源**；
4. **显式建立并发/流水线/event timeline**，例如 `Vector(tile i) || Matrix(tile i+1)`、独立资源日程或 `max(T_vector,T_matrix)`；
5. 这种重叠实际进入 Attention 的 latency/throughput/makespan 计算。

以下不算目标命中：

- 只把 `T_vector + T_matrix` 相加；
- 只做 memory/DMA ↔ aggregate compute overlap；
- 只比较 Vector width、systolic-array size 等硬件参数；
- 只在文本中声称“支持并行”，没有 Attention 测试配置或执行路径。

## 2. 数据源、检索与成本控制

### 2.1 数据集基线

复用 [`citing_papers_analysis.md`](./citing_papers_analysis.md) 已完成的双源并集：

- Semantic Scholar：91 篇 citing papers；
- OpenAlex：56 篇，补回 S2 漏掉的 18 条引文边；
- 去重后约 109 篇唯一论文，已有 KEEP 17 篇、MAYBE 约 21 篇。

本轮不重新遍历全图，而是从已有 KEEP/MAYBE 中优先筛选 `attention/flash/systolic/vector/tensor/overlap/concurrent/pipeline/event/timeline` 相关候选，再检查本地 BlaBlaPaper 笔记、arXiv/作者主页/GitHub 等免费来源。

### 2.2 API 与下载策略

| 来源 | 本轮采用的策略 | 成本/限流注意 |
|---|---|---|
| Semantic Scholar | 复用已有引文集；只对新增候选做 singleton 查询，使用精简 `fields`、缓存和退避 | 官方页面未给出公开按请求价格；未认证请求共享约 1000 RPS，API key 初始约 1 RPS，不能假设政策永久不变 |
| OpenAlex | 复用已有 56 篇结果；不重新跑全量 filter | 当前说明：无 key 约 `$0.10/day`，免费 key `$1/day`；list/filter 约 `$0.10/1000 calls`，search 约 `$1/1000 calls`；PDF/content 下载约 `$10/1000 PDFs`，本轮不使用 |
| 全文 | 优先本地笔记、arXiv、ar5iv、GitHub、作者 OA PDF；必要时记录 DOI/公开入口 | 无法免费取得的文章只记录为“有希望但文本未获取”，留给人工补全文本 |

### 2.3 证据分级

- **E0**：只有标题/引文上下文；不能证明用了仿真器；
- **E1**：摘要明确使用/扩展仿真器，但缺少执行细节；
- **E2**：全文有 Attention 配置、资源模型或时间线描述；
- **E3**：全文加公开代码/工件可定位到执行入口或结果生成脚本。

## 3. 暂定评分规则

| 项目 | 分值 |
|---|---:|
| 明确使用/扩展仿真器 | +2 |
| 实际运行 Attention/Transformer | +3 |
| 独立建模 Vector 与 Matrix/Systolic/Tensor 单元 | +3 |
| 明确的 overlap/concurrent/pipeline/event timeline | +4 |
| 重叠进入 Attention latency/throughput/makespan | +4 |
| 有公开代码或 artifact 可定位执行路径 | +2 |
| 只有 related-work/引用结论 | −4 |
| 只有 matmul/softmax 微基准、无 Attention | −3 |
| overlap 仅是 memory/DMA ↔ aggregate compute | −3 |
| 只有论文文字，没有配置或入口 | −2 |

分级：A（≥12，值得重点获取全文/代码）、B（8–11，强近候选）、C（4–7，补充或反例）、D（≤3，排除）。但有一个硬门槛：第 4、5 项均为 0 时，即使总分较高，也只能称为“近命中”，不能声称解决 Vector/Matrix 重叠问题。

## 4. 已有四篇工作的复核

| 工作            | 已确认事实                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 对目标问题的判断                                                                                                                  | 暂定分/证据                                              |                              |                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------- | ---------------------------------------- |
| **DOPS**      | 支持 LLMCompass 和内部 Ascend 910B simulator；DAG 中有 Attention、Q/K/V、Softmax、SV；模型区分 Ascend 的 Cube/Vector engine；用 Bifocal scheduler、时间线和 CoUtil 计算 NPU/PIM 协同。见 [`translation_notes.md`](../../misc/beyond-prefill-decode-disaggregation-dissecting-llm-inference-for-heterogeneous-platforms-via-dynamic-operator-scheduling/translation_notes.md) §4–§5。                                                                                                            | **高分但语义不完全相同**：它明确仿真了 Attention 中跨 NPU/PIM、跨算子/设备的重叠；目前没有证据证明同一芯片内 `Vector(tile i)                                        |                                                     | Cube(tile i+1)` 的 tile 级时间线。 | 18（A−，跨设备/算子重叠）；E2，TriForm 回放入口已在论文笔记中登记 |
| **PIPEWEAVE** | 实际采集并建模约 104,958 个 Attention 样本，拆分 Tensor、FMA、XU、MIO pipeline；覆盖 FlashInfer FA2/FA3 和变长 causal attention。其原文明确“不为 Tensor 与 FMA 的指令级并发建立僵化分析模型”，而是让 MLP 学习交互。见 [`paper_notes.md`](../../misc/pipeweave-synergizing-analytical-and-learning-models-for-unified-gpu-performance-prediction/paper_notes.md) 与 [`translation_notes.md`](../../misc/pipeweave-synergizing-analytical-and-learning-models-for-unified-gpu-performance-prediction/translation_notes.md)。 | **最有价值的近命中/反例**：有独立矩阵与向量类 pipeline、真实 Attention，但没有显式 Vector/Matrix overlap timeline；预测的是端到端 kernel latency，不是可解释的单元并发仿真。 | 8（B−，第 4/5 项为 0）；E2，全文笔记已在仓内；本地未找到代码                |                              |                                          |
| **SPAD**      | 扩展 LLMCompass，运行 BLOOM-176B、Llama3-70B、DeepSeek-V2（MHA/GQA/MLA/MoE）；分别扫描 Vector width、Systolic Array、cache 等参数。笔记明确说明每次迭代由 LLMCompass 返回整机 iteration latency。见 [`translation_notes.md`](../../misc/spad-specialized-prefill-and-decode-hardware-for-disaggregated-llm-inference/translation_notes.md) §5–§6。                                                                                                                                                     | **不命中**：真实运行了 Attention，也区分 Vector/Systolic，但实验是资源规格 DSE；没有同一 Attention tile 的独立资源时间线。其“重叠”主要是 prefill/decode 请求或集群调度层次。  | 8（B−近候选，严格目标为 C）；E2，arXiv 免费入口已在主清单登记               |                              |                                          |
| **ReaLLM**    | kernel simulator 直接构建在 LLMCompass 上，加入 MQA/MLA、SiLU、逐元素乘法，并用 kernel library + trace-driven system simulation 加速系统级评估；硬件模板包含 Vector Unit 与 Systolic Array。见 [`paper_notes.md`](../../misc/reallm-a-trace-driven-framework-for-rapid-simulation-of-large-scale-llm-inference/paper_notes.md) §3–§5。                                                                                                                                                                | **不命中**：解决的是 LLMCompass 仿真慢和系统级 batching/scheduling 缺失，不是 Vector/Matrix 资源竞争；没有独立的双资源 timeline。                           | 10（B−近候选，严格目标为 C）；E3，本地已有 `04-code/reallm.md` 与仓库记录 |                              |                                          |
| **SMOOTH**    | 集成 LLMCompass 做周期精确评估，运行 Transformer/FlashAttention；硬件含矩阵和向量固定功能单元。核心是 block-level SRAM 管理、早期回收和预取，目标是最大化 compute 与 I/O overlap。见 BlaBlaPaper 的 [`translation_notes.md`](../../../../../../workspace/BlaBlaPaper/outputs/smooth-hardware-assisted-fine-grained-on-chip-memory-management-for-efficient-on-device-llm-inference/translation_notes.md) §4–§6。                                                                                                      | **不命中**：论文中的 overlap 是 SRAM/DRAM 预取或 DMA 与计算的重叠；没有 Vector Unit 与 Matrix/Systolic Unit 的独立并发计算时间线。                         | 5（C，扣除 memory-only overlap）；E2，工件说明在本地翻译笔记中         |                              |                                          |

**复核结论**：用户已人工阅读的判断成立。四篇都不能作为“已证明 LLMCompass 缺少 Vector/Matrix overlap、且已有工作修复它”的最终证据；DOPS 和 PIPEWEAVE 是最值得继续深挖的两条线，但前者偏跨设备/算子，后者反而明确回避指令级并发。

## 5. 当前高优先级候选

下表中的“高”表示值得先拿全文或代码，不表示已经命中严格标准。

| 优先级 | 候选                                                          | 为什么值得查                                                                                                 | 当前缺口与获取状态                                                                                                                   |
| --- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| 1   | **DOPS: Beyond Prefill-Decode Disaggregation**              | 已有 E2 全文笔记；同时出现 Attention、Cube/Vector、DAG、hybrid execution、CoUtil 和 makespan，是现有列表中唯一明确给出重叠时间线的候选      | 需要确认 timeline 粒度是否进入同一 attention tile；TriForm 回放链接已记录，但本地未取得完整 simulator 源码                                                 |
| 2   | **PIPEWEAVE**                                               | Attention 数据集大、FlashInfer/FlashAttention 真实 kernel、Tensor/FMA/XU 独立 pipeline，适合与 LLMCompass 的算子串行模型做对照 | 原文明确不建模指令级 Tensor–FMA 并发，可能只能作为“近命中反例”；本地有全文翻译，未定位公开执行代码                                                                    |
| 3   | **DFModel**                                                 | 被引清单中明确批评 LLMCompass 无法建模片内 dataflow mapping，可能涉及比 LLMCompass 更细的资源/调度抽象                               | 当前只有引文上下文/摘要，尚未取得全文；无法确认是否运行 Attention、是否有 Vector/Matrix timeline。**有希望但文本未获取**                                             |
| 4   | **LLM Inference on Chiplet-based Architectures**            | 清单摘要称“leverages LLMCompass”，且主题涉及异构计算资源/互连调度                                                           | 当前只有题名和引文级信息；无法确认 workload、资源拆分和 overlap。**有希望但文本未获取**                                                                      |
| 5   | **MLDSE**                                                   | 事件驱动模拟器、硬件 IR、多级 DSE，可能提供可组合的资源并发抽象；已知使用 LLMCompass/CACTI 做部分评估                                        | 目前只确认 LLMCompass 被作为面积/延迟模型调用，未确认 Attention 或 Vector/Matrix overlap。arXiv 公开入口：<https://arxiv.org/abs/2503.21297>；本轮未取得本地全文 |
| 6   | **HydraPIM: Heterogeneous PIM for Attention**               | 主题直接包含 Attention、异构 PIM 与计算单元协同，可能比一般 DSE 工作更接近目标                                                      | OpenAlex 侧候选，尚未确认是否真的使用 LLMCompass；当前无全文。**有希望但文本未获取**                                                                      |
| 7   | **EONSim: NPU Simulator for On-Chip Memory**                | NPU simulator 可能有 Vector/Matrix 资源模型；且与 LLMCompass 存在版本/去重疑点                                           | 只知道其作为 OpenAlex delta/候选出现，未确认 LLMCompass 角色或 Attention 测试。**有希望但文本未获取**                                                    |
| 8   | **LP-Spec: LPDDR PIM for LLM Mobile Speculative Inference** | ICCAD、LLM mobile attention/PIM，可能包含异构计算调度                                                              | 摘要未点名 LLMCompass，尚未证明是后端仿真而非 related-work。**有希望但文本未获取**                                                                     |

### 暂不优先

RACAM、MoE-GPS、BlockPIM、LLMShare、NVR、Chip Architectures Under Advanced Computing Sanctions、Prefill vs. Decode Bottlenecks、DeepStack、AMALI 等已有证据更偏内存、MoE、成本 DSE、系统调度、解析模型或带宽假设；除非全文出现 Vector/Matrix timeline，否则不优先投入下载和代码分析。

## 6. 免费全文/代码获取登记

| 候选 | 已获得材料 | 下一步 | 状态 |
|---|---|---|---|
| DOPS | 仓内完整翻译、图表和 paper notes；TriForm 公开回放入口 | 获取 TriForm/模拟器源码，搜索 `Cube/Vector` 资源事件和 Attention tile DAG | 可继续核验 |
| PIPEWEAVE | 仓内完整翻译、paper notes、figs notes | 查公开 artifact；重点搜索是否有独立 resource timeline，而不是只看 MLP 特征 | 可继续核验 |
| SPAD | 仓内完整翻译和图表；arXiv 入口在主清单 | 无需再下载；若需要，检查扩展版 LLMCompass 是否公开 | 已足够判定“不命中” |
| ReaLLM | 仓内论文笔记、`04-code/reallm.md`、本地仓库登记 | 检查 kernel simulator 是否有资源事件接口；现有材料已显示没有目标 timeline | 已足够判定“不命中” |
| SMOOTH | BlaBlaPaper 完整翻译和 artifact 说明 | 若需要，人工提供工件压缩包以检查 SRAM manager 与 compute engine 的接口 | 已足够判定“仅 I/O overlap” |
| DFModel / Chiplet / MLDSE / HydraPIM / EONSim / LP-Spec | 仅摘要、题名或引文上下文 | 优先 arXiv、作者主页、GitHub；若仍无公开文本，等待人工提供 | **尚未获取全文** |

## 7. 阶段性核心结论

1. **在已取得全文的 LLMCompass 施引文献中，尚未发现严格命中者。** SPAD、ReaLLM、SMOOTH 复用或扩展了 LLMCompass，也运行了 Transformer/Attention，但没有把 Vector 与 Matrix 单元的跨 tile 并发纳入 latency 计算。
2. **DOPS 是当前最高分候选，但它验证的是 NPU/PIM、跨算子/跨设备的重叠。** 它证明“LLM 推理仿真可以用 DAG + 独立资源时间线计算 overlap”，但尚不能证明其模型覆盖 LLMCompass 所缺的同一 Attention tile 内 Vector/Systolic 并发。
3. **PIPEWEAVE 是最有价值的对照反例。** 它在真实 Attention kernel 上区分 Tensor、FMA、XU、MIO pipeline，说明资源分解确实能改善预测；但作者明确把 Tensor–FMA 指令级并发留给学习模型，没有提供我们需要的可解释 overlap simulator。
4. **现有被引文献更常见的“重叠”是三类不同问题：** (a) memory/DMA ↔ compute（SMOOTH）；(b) prefill/decode 或请求/集群级 overlap（SPAD）；(c) NPU ↔ PIM、算子级 overlap（DOPS）。这些不能直接当作 Vector ↔ Matrix overlap。
5. **最值得人工补全文本的未知候选是 DFModel、Chiplet-based Architectures、HydraPIM、EONSim 和 LP-Spec。** 本轮不对它们作“没有”的结论，只保留为待核验高优先级；如果用户提供任一全文或 artifact，可按第 1、3 节标准快速复核。

## 8. 下一轮核验清单

对每个高分候选只需查五件事：

1. Attention 测试是否真的执行了 FlashAttention/分 tile attention，而非只列为 future work；
2. 是否有 `Vector/Cube/Tensor/Systolic` 独立 resource class 或 event queue；
3. 是否存在跨 tile/跨算子依赖和资源可用时间 `ready_time`；
4. makespan 是否取资源时间线的并集/最大完成时间，而非延迟简单相加；
5. 是否能从代码、配置或图表复现一个 `Vector(i) || Matrix(i+1)` 的最小例子。

只要第 5 项无法定位，论文最多标为“资源分解/系统级 overlap 近似”，不作为已解决 LLMCompass 缺口的证据。
