# SGLang: Efficient Execution of Structured Language Model Programs 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al.

**发表期刊/会议 (Journal/Conference)**: ArXiv

**发表年份 (Publication Year)**: 2024

**研究机构 (Affiliations)**: Stanford University, UC Berkeley, Shanghai Jiao Tong University

---

## 1. 摘要

**目的**
- 解决大型语言模型程序在编程与执行过程中的低效问题。
- 消除多调用结构中的冗余计算与内存浪费。
- 简化复杂工作流（如 Agent 控制、逻辑推理、多轮对话）的编程难度。

---

**方法**
- 提出 **SGLang** 系统，包含前端语言与后端运行时。
- 前端语言：
  - 嵌入 Python 的领域特定语言 (DSL)。
  - 提供原语：`gen` (生成)、`select` (选择)、`fork` (并行分支)、`join` (合并)。
  - 支持异步流执行与计算图编译。
- 后端运行时优化：
  - **RadixAttention**：
    - 利用 Radix tree 管理 KV cache，实现自动、系统的跨调用复用。
    - 采用 LRU eviction policy 与 cache-aware scheduling (最长共享前缀优先)。
    - 兼容 continuous batching、paged attention 与 tensor parallelism。
  - **Compressed Finite State Machine (FSM)**：
    - 分析正则表达式约束，将相邻的 singular transition edges 压缩为单边。
    - 允许单次 forward pass 解码多个 Token，加速结构化输出 (如 JSON)。
  - **API Speculative Execution**：
    - 针对黑盒 API 模型 (如 GPT-4)，通过忽略停止条件继续生成，复用后续原语结果，降低延迟与输入 Token 成本。

---

**结果**
- 性能提升对比：

| 指标 | 提升幅度 |
| :--- | :--- |
| 吞吐量 | 最高提升 **6.4倍** |
| 延迟 | 最高降低 **3.7倍** |
| 多模态吞吐量 | 最高提升 **6倍** |
| JSON 解码吞吐量 | 提升 **1.6倍** |

- Cache 表现：
  - Cache hit rate 达到 **50% 至 99%**。
  - Cache-aware scheduling 平均达到最优命中率的 **96%**。
  - RadixAttention 数据结构维护开销仅 **0.3%**。
- 实际部署：在 Chatbot Arena 中，Vicuna-33B 的 Cache hit rate 达 **74.1%**，首 Token 延迟平均降低 **1.7倍**。

---

**结论**
- SGLang 通过前后端协同设计，显著提升了复杂 LM 程序的执行效率。
- RadixAttention 与 Compressed FSM 等创新运行时优化有效解决了 KV cache 冗余与结构化解码缓慢的问题。
- 系统广泛兼容各类开源权重模型、多模态模型及 API 模型，是开发高级提示技术与 Agent 工作流的有力工具。

---

## 2. 背景知识与核心贡献

**研究背景与动机**

- LLMs能力提升，应用场景从简单聊天转向多轮规划、推理及Agent交互，形成包含多次LLM调用、控制流及结构化输入输出的**Language Model Programs** (LM Programs)。
- 现有系统在表达和执行LM Programs时存在两大痛点：
  - **编程困难**：LLM的非确定性导致开发过程充斥大量字符串操作、Prompt调试、脆弱的输出解析及复杂的并行控制，代码可读性极差。
  - **执行低效**：现有推理引擎缺乏对工作负载的感知，导致冗余计算与内存浪费。典型问题包括无法跨调用重用**KV Cache**，以及在受限解码（如JSON模式）时只能逐**Token**处理，速度极慢。

---

**核心贡献**

- 提出**SGLang**系统，通过前后端协同设计实现LM Programs的高效编程与执行。
- **前端语言设计**：作为Python嵌入式DSL，提供`gen`、`select`、`fork`、`join`等原语，原生兼容Python控制流，大幅简化多调用工作流的编程复杂度。
- **后端Runtime优化**：提出三项核心技术加速执行：
  - **RadixAttention**：利用Radix tree管理**KV Cache**的LRU缓存，实现跨请求、跨调用的自动前缀匹配与重用，配合Cache-aware scheduling提升命中率。
  - **Compressed Finite State Machine (FSM)**：分析正则表达式约束并构建压缩状态机，将多Token路径压缩为单步路径，支持单次前向传播解码多个Token，大幅加速结构化输出。
  - **API Speculative Execution**：针对黑盒API模型（如GPT-4），通过推测执行忽略停止条件，提前生成后续内容并复用，降低多调用程序的延迟与输入Token成本。
- **性能提升**：在Agent控制、逻辑推理、JSON解码等广泛任务上，相比vLLM、Guidance等系统，实现最高达**6.4倍**的吞吐量提升与**3.7倍**的延迟降低。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体技术架构**

SGLang 的整体技术架构由 **前端语言** 和 **后端运行时** 两部分组成，旨在通过前后端协同设计 系统性地利用 LM Programs 中的多调用结构以实现高效执行。

---

**前端语言**

前端是一个嵌入在 Python 中的领域特定语言 (DSL)，旨在简化 LM Programs 的编程复杂度。
- **核心原语**：
  - `gen`：调用模型进行生成，支持 `regex` 参数约束输出格式。
  - `select`：让模型从选项列表中选择概率最高的选项。
  - `extend` (`+=`)：向 prompt 状态追加字符串。
  - `fork` / `join`：创建并行分支与合并分支，实现程序内并行。
  - `image` / `video`：处理多模态输入。
- **执行模式**：
  - **解释器模式**：将 prompt 视为异步流，原语提交至流中异步执行，支持后台线程的流执行器以实现程序内并行。
  - **编译器模式**：将程序编译为计算图，通过图执行器运行，支持图重写等静态优化。

---

**后端运行时**

后端运行时负责加速 SGLang 程序的执行，包含三大核心优化技术：
- **RadixAttention**：
  - 利用 **Radix Tree** 管理 KV cache，实现自动且系统的跨请求、跨调用复用。
  - 采用 **LRU eviction policy**，优先驱逐最近最少使用的叶子节点。
  - 引入 **Cache-aware scheduling**，基于最长共享前缀优先 排序请求，最大化缓存命中率。
- **Compressed Finite State Machine (FSM)**：
  - 针对结构化输出 (如 JSON)，将正则表达式转换为 FSM。
  - 压缩相邻的单一转换边，将多 token 路径压缩为单步路径。
  - 允许在一次前向传播 中解码多个 token，大幅加速 constrained decoding。
- **API Speculative Execution**：
  - 针对 API-only 模型 (如 GPT-4)，在首次调用时忽略停止条件继续生成。
  - 解释器保留额外生成内容，与后续原语匹配复用，减少 API 调用延迟与输入 token 成本。

---

**前后端协同设计**

- 前端解释器将完整 prompt 发送至后端进行前缀匹配。
- 在执行 `fork` 原语时，前端先发送前缀作为 **hint**，确保其正确插入 Radix Tree，简化后端调度与匹配。
- 前端与后端紧密配合，使得运行时能更好地处理动态树结构等复杂复用模式。

### 1. RadixAttention

**核心概念与背景**
- **RadixAttention** 是一种在 runtime 自动且系统地复用 **KV cache** 的新技术。
- 在 LLM 推理中，**KV cache** 存储了前向传播的中间张量，其计算仅依赖于前缀 **Token**。
- 具有相同 prompt 前缀的请求可以复用 **KV cache**，从而减少冗余计算和内存使用。
- 现有系统在请求完成后丢弃 **KV cache**，导致跨调用复用困难。**RadixAttention** 将 **KV cache** 作为传统 cache 管理，使用 **基数树** 进行高效的匹配、插入和驱逐。

**数据结构与存储机制**
- **基数树**：作为经典字典树的压缩变体，其边可以标记为变长元素序列，显著提升匹配效率。
- **映射关系**：系统利用基数树管理 **Token** 序列到其对应 **KV cache** 张量的映射。
- **分页布局**：**KV cache** 张量以非连续的分页布局存储，每页大小等同于一个 **Token**。
- **内存共享**：缓存 Token 和当前运行中的请求共享同一内存池，系统动态分配内存。

**算法流程与内存管理**
- **LRU 驱逐策略**：
  - 引入 **LRU** (Least Recently Used) 驱逐策略，优先驱逐最近最少使用的叶子节点。
  - 驱逐叶子节点后，其公共祖先节点可被复用，直到祖先变为叶子并被驱逐。
  - 当等待运行的请求足够多时，系统会驱逐所有缓存 Token 以换取更大的 **batch size**。
- **引用计数器**：
  - 每个节点维护一个引用计数器，指示有多少运行中的请求正在使用它。
  - 引用计数为零的节点才可被驱逐，确保 continuous batching 设置下正在运行的 batch 不会被干扰。
- **前端与后端协同**：
  - 前端解释器将完整 prompt 发送到 runtime，runtime 执行前缀匹配和复用。
  - 在执行 `fork` 原语时，前端先发送前缀作为 **hint**，确保前缀正确插入树中，再发送剩余 prompt。这简化了 runtime 调度和匹配。

**Cache-aware Scheduling 策略**
- **定义**：**cache hit rate** = 缓存的 prompt **Token** 数 / prompt **Token** 总数。
- **调度算法**：在 batch-processing 设置中，按匹配前缀长度对请求排序，优先处理具有更长匹配前缀的请求，而非先到先服务。
- **最优性证明**：在离线情况下，最长共享前缀优先顺序等价于深度优先搜索 (**DFS**) 顺序，可证明达到最优 **cache hit rate**。
- **在线场景**：在线情况下，**DFS** 顺序会被打乱，但该调度策略仍在增强的基数树上近似 **DFS** 行为。

**分布式场景扩展**
- **Tensor parallelism**：每个 GPU 维护分片的 **KV cache**，无需额外同步，因为树操作相同。
- **Data parallelism**：
  - 每个 worker 维护自己的子树，router 监督一个 meta-tree。
  - **Meta-tree** 追踪所有子树及关联设备，基于请求的 affinity（共享前缀长度）进行调度。
  - 采用弱一致性分布式 cache 设计，worker 发生驱逐时提交到队列，router 在低活动期处理更新。

**输入输出关系与整体作用**
- **输入**：前端提交的完整 prompt 序列、`fork` 操作产生的 **hint**、等待队列中的请求。
- **输出**：复用或新计算的 **KV cache** 张量，供后续 **Token** 解码使用。
- **整体作用**：
  - 自动处理各种 **KV cache** 复用模式（如多级树结构共享），无需手动配置。
  - 减少 prefill 阶段的冗余计算，降低首个 **Token** 延迟。
  - 通过共享 **KV cache** 减少总内存使用，允许更大的 **batch size**，从而提升最大吞吐量。
  - 兼容 continuous batching、paged attention 和 tensor parallelism 等技术，且在无 cache 命中时引入的内存和时间开销可忽略不计（低于 0.3%）。

---

**RadixAttention 与现有系统对比**

| 特性 | 现有系统 | RadixAttention |
|---|---|---|
| **KV cache** 生命周期 | 请求完成后丢弃 | 维护在基数树中作为 LRU cache |
| 复用模式 | 简单前缀共享，需手动配置 | 自动处理多级树结构等所有复用模式 |
| 调度策略 | 先到先服务 (FCFS) | Cache-aware scheduling (最长共享前缀优先) |
| 内存管理 | 固定大小内存池 | 缓存与运行请求共享动态内存池 |
| 分布式支持 | 有限 | 支持 Tensor parallelism 与 Data parallelism |

### 2. Compressed Finite State Machine

**核心概念与系统作用**

**Compressed Finite State Machine** (压缩有限状态机) 是 SGLang 运行时用于加速结构化输出解码的核心技术。在 LM programs 中，开发者常使用 `regex` 参数约束 LLM 输出特定格式（如 JSON schema）。该技术通过分析约束条件，将多 Token 路径压缩为单步路径，实现一次前向传播解码多个 Token，从而大幅提升解码速度。其在系统中的作用是解决现有推理引擎逐 Token 解码导致的性能瓶颈，特别是在处理常量字符串前缀时。

---

**现有系统瓶颈**

- 现有系统将正则表达式转换为 **Finite State Machine (FSM)** 指导解码。
- 解码时维护当前 FSM 状态，获取下一状态允许的 Token，将无效 Token 概率置零。
- 这种机制强制模型逐 Token 解码。
- 当遇到常量序列（如 `{"summary": "`）时，即使存在唯一有效下一 Token，系统仍需多次执行 forward pass，导致计算资源浪费与解码延迟。

---

**实现原理与算法流程**

- **基础构建**：基于字符/字符串构建原始 FSM，而非直接基于 Token，以规避复杂的 Token 映射问题。
- **定义核心结构**：
  - **Singular transition edge** (单一转换边)：满足两个条件的边：1) 源节点仅有一个后继节点；2) 边上仅有一个可接受的字符或字符串。
  - **Compressed edge** (压缩边)：由连续相邻的边 $(e_0, e_1, \ldots, e_k)$ 压缩而成，当且仅当 $e_1, \ldots, e_k$ 均为单一转换边。压缩边的文本为各边文本的拼接。
- **压缩流程**：
  - 从基于字符的原始 FSM 出发。
  - 递归地将单一转换边合并至其前驱边。
  - 持续合并直至无法进一步压缩，最终生成 **Compressed FSM**。
- **批量预处理与复用**：
  - 系统对状态机进行预处理。
  - 预处理结果在批量请求中复用，避免为每个请求重复构建 FSM。

---

**Tokenization 处理与加速机制**

- **Jump Forward (跳转前进) 机制**：
  - 当生成新 Token 时，系统获取其字符串并搜索当前状态的所有出边。
  - 找到以该字符串开头的边后，系统向前移动。
  - 若该边为包含长字符串的压缩边，系统可预期后续多轮的解码字符串。
  - 多个 Token 在压缩转换边上可在一个 forward pass 中完成解码，实现加速。
- **Retokenization (重新分词) 处理**：
  - **问题**：压缩边包含的长字符串需转换为 Token 供下一解码阶段使用。由于 LLM 特定的预训练和分词方法，直接切分字符串可能改变语义（如 `{"summary": "` 应被分词为 `{"`, `summary`, `":` 和 `_"`，而非随机切分）。
  - **解决方案**：使用原始 tokenizer 对之前的所有文本及压缩边文本进行重新分词，确保与 LLM 原始输入格式对齐。
  - **开销**：仅带来轻微的 retokenization overhead。

---

**性能数据与参数设置**

| 优化组件 | 性能指标 | 提升幅度 |
| :--- | :--- | :--- |
| **Compressed FSM** | JSON decoding 吞吐量 | **1.6x** 提升 |
| 预处理复用 (避免重复构建) | JSON decoding 吞吐量 | 避免 **2.4x** 下降 |
| **Compressed FSM** | 整体解码速度 | 多 Token 单步解码加速 |

---

**局限性与未来方向**

- **Distorted Probability (概率扭曲)**：
  - 字符串与 Token 间的映射导致概率分布扭曲。
  - 例如，对于 regex `Excellent|Above Average|Fair|Below Average`，运行时可能因 `Above Average` 位于压缩转换边上，错误地将 `A` 映射至 `Above Average`。
  - 原因：LLM 未识别具体选项范围，导致不恰当的 Token 序列。
- **当前变通方案**：将选项或 regex 直接包含在 prefill prompt 中，引导 LLM 感知选项并输出正确的 Token 序列。
- **未来研究方向**：需进一步研究以改进压缩 FSM 的准确性，解决底层概率扭曲问题。

### 3. API Speculative Execution

**核心概念与背景**
- 针对 **API-only models**（如 OpenAI GPT-4, GPT-3.5）的专属优化技术。
- 解决 LM Programs 中多链式 `gen` 调用导致的 **Latency** 过高与 **Cost** 浪费问题。
- 传统执行模式下，多个 `gen` 原语对应多次独立 API 调用，导致相同的 **Context** 被重复输入并计费。

---

**实现原理与算法流程**
- **核心机制**：在首次 API 调用时启用 **Speculative Execution**，主动忽略首个 `gen` 原语的 `stop` 停止条件。
- **执行流程**：
  - 模型在生成首个目标字段后不停止，继续生成后续若干 **Token**。
  - **Interpreter** 捕获并保留这些额外的生成输出。
  - **Interpreter** 将保留的输出与后续 `gen` 原语的预期结果进行模式匹配。
  - 匹配成功则直接复用这些 **Token**，跳过后续的 API 调用。

---

**输入输出关系与参数设置**
- **输入**：包含多个链式 `gen` 调用的 SGLang 程序（如 `s += context + "name:" + gen("name", stop="\n") + "job:" + gen("job", stop="\n")`）。
- **输出**：从单次 API 调用的扩展生成结果中提取并拼接的完整结构化数据。
- **关键参数与设置**：
  - `stop`：原语中的停止条件（如 `\n`），在首次调用中被运行时主动忽略。
  - **Prompt Engineering**：需精心设计提示词模板，确保模型能以高准确率自然匹配后续原语的格式要求。

| 对比维度 | 传统执行模式 | API Speculative Execution 模式 |
| :--- | :--- | :--- |
| **API 调用次数** | N 次（对应 N 个 `gen` 原语） | 1 次（首次调用后忽略停止条件） |
| **Context 计费** | 重复计算 N 次 | 仅计算 1 次 |
| **Latency** | 高（多次网络往返） | 低（单次网络往返） |

---

**在整体架构中的作用与性能表现**
- **系统作用**：作为 SGLang 运行时的第三大优化技术，填补了黑盒 API 模型在多调用程序优化上的空白，与针对开源模型的 RadixAttention 和 Compressed FSM 形成互补。
- **性能指标**：
  - 在 GPT-3.5 提取 Wikipedia 页面三个字段的测试中，通过 few-shot prompting 保证高准确率。
  - **Cost 降低**：输入 **Token** 成本降低约 **3 倍**。


---

## 4. 实验方法与实验结果

**实验设置**

- **模型配置**：
  - Dense 模型：Llama-2 (7B/70B)
  - Sparse MoE 模型：Mixtral-8x7B
  - Multi-modal 模型：LLaVA-v1.5-7B (image), LLaVA-NeXT-34B (video)
  - API 模型：OpenAI GPT-3.5
  - 精度设置：float16
- **硬件环境**：
  - 主要采用 AWS EC2 G5 实例，配备 NVIDIA A10G GPU (24GB)
  - 部分实验在 A100G (80GB) 上运行
  - 7B 模型单卡运行，大模型采用 Tensor Parallelism 跨多卡运行
- **基线对比**：
  - Guidance (v0.1.8, llama.cpp backend)
  - vLLM (v0.2.5)
  - LMQL (v0.7.3, Hugging Face Transformers backend)
- **工作负载**：
  - Benchmark：5-shot MMLU, 20-shot HellaSwag
  - Agent：ReAct agent, Generative agents
  - Prompting 技术：Tree-of-thought, Skeleton-of-thought, Branch-solve-merge
  - 结构化输出：JSON decoding
  - 对话与检索：Multi-turn chat (short/long), DSPy RAG pipeline
- **评估指标**：
  - 吞吐量：每秒程序数
  - 延迟：单程序平均执行延迟

---

**结果数据**

- **开源权重模型表现**：
  - Llama-7B 吞吐量提升最高达 **6.4x**，延迟降低最高达 **3.7x**
  - MMLU：通过 RadixAttention 复用 5-shot examples 的 KV cache
  - HellaSwag：实现 few-shot examples 与 common question prefix 的二级 KV cache 共享
  - ReAct/Generative agents：复用 agent template 与 previous calls 的 KV cache
  - Tree-of-thought/Skeleton-of-thought：并行化生成调用并复用 KV cache
  - JSON decoding：通过 Compressed FSM 实现多 token 一次性解码
  - Multi-turn chat：复用 chat history 的 KV cache，短输出提速明显，长输出受限于 decoding 时间提速有限
  - DSPy RAG：复用 common context example 的 KV cache，cache hit rate 达 50%-99%
- **大模型与多模态表现**：
  - Mixtral-8x7B 与 Llama-70B 在 Tensor Parallelism 下展现相似加速趋势
  - LLaVA 系列模型吞吐量提升最高达 **6x**，通过对输入图像计算 hash 作为 radix tree 的 key，复用同一图像的 image tokens 的 KV cache
- **生产环境部署**：
  - 部署于 Chatbot Arena
  - LLaVA-Next-34B 的 RadixAttention cache hit rate 为 **52.4%**
  - Vicuna-33B 的 cache hit rate 为 **74.1%**
  - Vicuna-33B 的首 token 延迟平均降低 **1.7x**
- **API 模型优化**：
  - 针对 OpenAI GPT-3.5 使用 API speculative execution
  - 输入 token 成本降低约 **3x**

| 工作负载 | 优化技术 | 加速效果/提升 |
| :--- | :--- | :--- |
| Llama-7B 综合任务 | RadixAttention, 并行化, Compressed FSM | 吞吐量提升 **6.4x**，延迟降低 **3.7x** |
| LLaVA 多模态任务 | 图像 hash 映射至 radix tree | 吞吐量提升 **6x** |
| Vicuna-33B 生产部署 | 复用系统消息与多轮历史 | 首 token 延迟降低 **1.7x** |
| GPT-3.5 API 调用 | API speculative execution | 输入 token 成本降低 **3x** |
| JSON decoding | Compressed FSM 预处理与复用 | 吞吐量提升 **1.6x** |

---

**消融实验**

- **Cache hit rate 与性能关系**：
  - 更高的 cache hit rate 直接带来更大的 batch size、更高的吞吐量以及更低的延迟
- **RadixAttention 组件有效性**：
  - 对比项包含 No Cache, No Tree-Structure, FCFS Schedule, Random Schedule, No Frontend Parallelism, No Frontend Hint
  - 所有组件均为最佳性能所必需，前端并行性与 hint 对运行时性能至关重要，验证了 frontend-runtime co-design 的必要性
- **RadixAttention 开销**：
  - 在无 KV cache 复用机会的 ShareGPT 数据集上测试
  - 管理 RadixAttention 数据结构耗时仅 0.2s，总耗时 74.3s，开销低于 **0.3%**，可默认开启
- **Compressed FSM 有效性**：
  - 在 JSON decoding 基准测试中，吞吐量提升 **1.6x**
  - 需对状态机进行预处理并在 batch 请求中复用，否则吞吐量下降 **2.4x**

---

