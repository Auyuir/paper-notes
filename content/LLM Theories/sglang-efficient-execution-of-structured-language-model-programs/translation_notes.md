# SGLang: Efficient Execution of Structured Language Model Programs 原文翻译

# SGLang：结构化语言模型程序的高效执行

*Lianmin Zheng$^2$ {同等贡献。} $ $ Liangsheng Yin$^3$ $ $ Zhiqiang Xie$^1$ $ $ Chuyue Sun$^1$ $ $ Jeff Huang$^4$ Cody Hao Yu$^5$ $ $ Shiyi Cao$^2$ $ $ Christos Kozyrakis$^1$ $ $ Ion Stoica$^2$ $ $ Joseph E. Gonzalez$^2$ Clark Barrett$^1$ $ $ Ying Sheng$^{1*}$ $^1$ 斯坦福大学 $^2$ 加州大学伯克利分校 $^3$ 上海交通大学 $^4$ 德克萨斯农工大学 $^5$ 独立研究员*

## 摘要

大型语言模型（LLM）越来越多地被用于需要多次生成调用、高级提示技术、控制流和结构化输入/输出的复杂任务。然而，目前缺乏用于编程和执行这些应用的高效系统。我们介绍了 ，一个用于高效执行复杂语言模型程序的系统。 由前端语言和运行时组成。前端通过提供生成和并行控制的原语来简化编程。运行时通过 RadixAttention（用于 KV 缓存重用）和压缩有限状态机（用于加速结构化输出解码）等新颖优化来加速执行。实验表明，在包括智能体控制、逻辑推理、少样本学习基准、JSON 解码、检索增强生成流水线和多轮对话等任务上，与最先进的推理系统相比， 在各种大型语言和多模态模型上实现了高达 $6.4\times$ 的吞吐量提升。代码已在 <https://github.com/sgl-project/sglang> 公开。

# 引言

大型语言模型（LLM）能力的近期提升拓宽了其应用范围，使其能够处理更广泛的通用任务并充当自主代理 (OpenAI 2023; Bubeck et al. 2023; Park et al. 2023; G. Wang et al. 2023; Sumers et al. 2023)。在这些应用中，LLM 进行多轮规划、推理以及与外部环境的交互。这通过工具使用 (Schick et al. 2023; Patil et al. 2023)、多种输入模态 (Team et al. 2023; Alayrac et al. 2022) 以及各种提示技术 (X. Liu et al. 2023) 来实现，例如少样本学习 (Brown et al. 2020)、自一致性 (X. Wang et al. 2022)、思维骨架 (Ning et al. 2024) 和思维树 (Yao et al. 2023)。所有这些新用例都需要多次、通常相互依赖的 LLM 生成调用，呈现出使用多调用结构来完成复杂任务的趋势 (Yao et al. 2022; Kim et al. 2023)。

这些模式的出现标志着我们与 LLM 交互方式的转变，从简单的聊天转向更复杂的 LLM 编程式使用，即使用程序来调度和控制 LLM 的生成过程。我们将这些程序称为"语言模型程序"（LM Programs） (Beurer-Kellner, Fischer, and Vechev 2023; Khattab et al. 2023)。上述高级提示技术和代理工作流都属于 LM 程序的范畴。LM 程序有两个共同特性：（1）LM 程序通常包含多个穿插着控制流的 LLM 调用。这是完成复杂任务和提高整体质量所必需的。（2）LM 程序接收结构化输入并产生结构化输出。这是实现 LM 程序的组合以及将 LM 程序集成到现有软件系统中所必需的。

尽管 LM 程序被广泛使用，但当前用于表达和执行它们的系统仍然效率低下。我们识别出与 LM 程序高效使用相关的两个主要挑战：*首先，由于 LLM 的非确定性，编写 LM 程序既繁琐又困难。* 开发 LM 程序通常需要大量的字符串操作、提示的实验性调优、脆弱的输出解析、处理多种输入模态以及实现并行机制。这种复杂性显著降低了即使是简单程序的可读性 ([sec:programming_model])。

*其次且同样重要的是，由于冗余计算和内存使用，执行 LM 程序效率低下。* 最先进的推理引擎（例如 vLLM (Kwon et al. 2023)、TGI (Hugging Face, n.d.) 和 TensorRT-LLM (NVIDIA, n.d.)）已经过优化以降低延迟并提高吞吐量，但无需直接了解工作负载。这使得这些系统具有通用性和鲁棒性，但也导致在任何给定工作负载下存在显著的效率低下。一个突出的例子是 Key-Value（KV）缓存的重用 ([sec:radix_attention])。KV 缓存由可重用的中间张量组成，这些张量对生成式推理至关重要。在 LM 程序的典型批量执行中，存在大量跨多个共享公共前缀的不同 LLM 调用重用 KV 缓存的机会。然而，当前系统缺乏有效的机制来促进这种重用，导致不必要的计算和内存浪费。另一个例子是结构化输出的约束解码（例如 JSON 模式），其中 LLM 的输出被限制为遵循由正则表达式定义的特定语法规则 ([sec:compressed-fsm])。在这些约束下，通常可以一次解码多个 Token。然而，现有系统一次只能解码一个 Token，导致解码速度不理想。

系统架构：一个解释器通过优化的运行时执行语言原语。

为了应对这些挑战，我们提出了 SGLang，一种用于 LLM 的<u>结</u>构化<u>生</u>成<u>语</u>言。其核心思想是系统地利用 LM 程序中的多调用结构以实现高效执行。如 [fig:architecture] 所示，它由两部分组成：前端语言和后端运行时。前端简化了 LM 程序的编写，运行时加速了其执行。两部分可以协同工作以获得更好的性能，也可以独立运行。

我们将 SGLang 作为嵌入在 Python 中的领域特定语言引入。它提供了生成原语（例如 `extend`、`gen`、`select`）和并行控制原语（例如 `fork`、`join`）。SGLang 兼容 Python 的控制流和库，因此用户可以使用原生 Python 语法轻松开发高级提示工作流。我们提供了 SGLang 的解释器和编译器。解释器将提示状态作为流进行管理，并将原语操作提交到流中异步执行，确保对同步和程序内并行的适当控制。此外，SGLang 程序可以被追踪和编译以进行更多优化。

在运行时方面，我们提出了几种新颖的优化技术来加速 SGLang 程序的执行。*第一种技术*，RadixAttention，实现了跨多次生成调用自动重用 KV 缓存。在现有推理引擎中，请求的 KV 缓存在处理完成后即被丢弃，阻止了 KV 缓存在多次调用间被重用，从而显著降低了执行速度。相反，我们的系统在基数树中为所有请求维护 KV 缓存的 LRU 缓存。这种方法将 KV 缓存作为传统缓存进行管理，并使用基数树进行高效的匹配、插入和驱逐。它允许运行时通过缓存感知的调度策略高效地处理各种重用模式。*第二种技术*是压缩有限状态机，它实现了更快的结构化输出约束解码。现有系统仅通过屏蔽不允许的 Token 的概率来约束下一个 Token，使其每次只能解码一个 Token。相反，我们的系统分析约束并构建压缩有限状态机来表示约束。这种方法在可能的情况下将多 Token 路径压缩为单步路径，允许一次解码多个 Token 以实现更快的解码速度。最后，SGLang 还支持仅 API 模型，如 OpenAI 的 GPT-4，我们引入了*第三种技术*，API 推测执行，以优化仅 API 模型的多调用程序。

使用 SGLang，我们实现了各种 LLM 应用，包括代理控制、逻辑推理、少样本学习基准、JSON 解码、检索增强生成流水线、多轮对话和多模态处理。我们在包括 Llama-7B/70B (Touvron et al. 2023)、Mistral-8x7B (Jiang et al. 2024)、LLaVA-v1.5-7B（图像） (H. Liu et al. 2024) 和 LLaVA-NeXT-34B（视频） (Zhang et al. 2024) 的模型上，在 NVIDIA A10G 和 A100 GPU 上测试了性能。实验结果表明，与现有编程和推理系统（包括 Guidance (Guidance AI, n.d.)、vLLM (Kwon et al. 2023) 和 LMQL (Beurer-Kellner, Fischer, and Vechev 2023)）相比，SGLang 在广泛的工作负载、模型和硬件配置下实现了高达 $6.4\times$ 的吞吐量提升。

# 编程模型

 本节通过一个运行示例介绍 SGLang 编程模型，描述其语言原语和执行模式，并概述了运行时优化机会。该编程模型通过提供灵活且可组合的原语，可以简化多调用工作流中的繁琐操作（例如，字符串操作、API 调用、约束规范、并行性）。

SGLang 中多维度作文评判器的实现利用了 branch-solve-merge 提示技术 (Saha et al. 2023)。SGLang 提供的原语以红色显示。

**一个运行示例。** 该语言是嵌入在 Python 中的领域特定语言。[fig:example_first_program] 展示了一个使用 branch-solve-merge 提示方法 (Saha et al. 2023) 评估关于图像的作文的程序。函数 `multi_dimensional_judge` 接受三个参数：`s`、`path` 和 `essay`。`s` 管理提示状态，`path` 是图像文件路径，`essay` 是作文文本。可以使用 `+=` 运算符将新字符串和 SGLang 原语附加到状态 `s` 中以供执行。首先，该函数将图像和作文添加到提示中。然后，它使用 `select` 检查作文是否与图像相关，并将结果存储在 `s["related"]` 中。如果相关，则将提示分叉为三个副本，以便从不同维度进行并行评估，使用 `gen` 将结果存储在 `f["judgment"]` 中。接下来，它合并评判结果，生成摘要，并分配字母等级。最后，它以 JSON 格式返回结果，遵循由正则表达式约束 `regex` 定义的模式。SGLang 极大地简化了这个程序，因为使用类似 OpenAI API 接口的等效程序将需要 $2.1\times$ 的代码行数，这是由于手动字符串操作和并行性控制造成的。

**语言原语。** SGLang 提供了用于控制提示状态、生成和并行性的原语。它们可以与 Python 语法和库一起使用。以下是这些原语：“`gen`”调用模型进行生成，并将结果存储在以其第一个参数指定名称命名的变量中。它支持“`regex`”参数，以约束输出遵循由正则表达式定义的语法（例如，JSON 模式）。“`select`”调用模型从列表中选择概率最高的选项。运算符“`+=`”或“`extend`”将字符串附加到提示中。运算符“`[variable_name]`”获取生成的结果。“`fork`”创建提示状态的并行分叉。“`join`”重新合并提示状态。“`image`”和“`video`”接收图像和视频输入。

**执行模式。** 执行 SGLang 程序最简单的方法是通过解释器，其中提示被视为异步流。像 `extend`、`gen` 和 `select` 这样的原语被提交到流中以进行异步执行。这些非阻塞调用允许 Python 代码继续运行，而无需等待生成完成。这类似于异步启动 CUDA 内核。每个提示由后台线程中的流执行器管理，从而实现程序内并行性。获取生成结果将阻塞直到它们准备就绪，从而确保正确的同步。或者，SGLang 程序可以编译为计算图，并使用图执行器执行，从而允许进行更多优化。本文默认使用解释器模式，并在 [sec:compiler_mode] 中讨论编译器模式的结果。SGLang 使用其自己的 SGLang Runtime (SRT) 支持开放权重模型，以及诸如 OpenAI 和 Anthropic 模型之类的 API 模型。

**对比。** 用于 LLM 的编程系统可以分为高级（例如 LangChain、DSPy）和低级（例如 LMQL、Guidance、SGLang）。高级系统提供预定义或自动生成的提示，例如 DSPy 的提示优化器。低级系统通常不改变提示，但允许直接操作提示和原语。SGLang 是一个类似于 LMQL 和 Guidance 的低级系统。[tab:diff_lang] 比较了它们的特性。SGLang 更关注运行时效率，并带有其自身协同设计的运行时，允许引入稍后介绍的新型优化。高级语言（例如 DSPy）可以编译为低级语言（例如 SGLang）。我们在 [sec:eval] 中展示了将 SGLang 作为 DSPy 后端集成以获得更好运行时效率的方法。

**运行时优化。** [fig:example_first_program] 展示了三个运行时优化机会：KV cache 复用、快速约束解码、API 推测执行。我们将在以下章节中讨论它们。

SGLang 程序可以通过 "`fork`" 原语链接多个生成调用并创建并行副本。此外，不同的程序实例通常共享一些共同部分（例如，系统提示词）。这些场景在执行过程中创建了许多共享的提示词前缀，从而为重用 KV cache 提供了大量机会。在 LLM 推理期间，KV cache 存储来自前向传播的中间张量，用于解码未来的 Token。它们以 self-attention 机制中的键值对命名 (Vaswani et al. 2017)。KV cache 的计算仅依赖于前缀 Token。因此，具有相同提示词前缀的请求可以重用 KV cache，从而减少冗余计算和内存使用。更多背景和一些示例在 [sec:additional_radix_attention] 中提供。

鉴于 KV cache 重用的机会，优化 SGLang 程序的一个关键挑战是在多个调用和实例之间重用 KV cache。虽然一些系统探索了某些 KV cache 重用案例 (Kwon et al. 2023; L. Ye et al. 2024; Juravsky et al. 2024; Gim et al. 2023)，但它们通常需要手动配置，并且无法处理所有重用模式（例如，动态树结构）。因此，大多数最先进的推理系统会为每个请求重新计算 KV cache。我们将在 [sec:related-work] 中讨论它们的局限性以及我们的不同之处。

本节介绍 RadixAttention，这是一种在运行期间自动且系统地重用 KV cache 的新技术。与现有的在生成请求完成后丢弃 KV cache 的系统不同，我们的系统将提示词和生成结果的 cache 保留在基数树中，从而实现高效的前缀搜索、重用、插入和驱逐。我们实现了 LRU 驱逐策略和缓存感知调度策略，以提高缓存命中率。RadixAttention 与 continuous batching (Yu et al. 2022)、paged attention (Kwon et al. 2023) 和 tensor parallelism (Shoeybi et al. 2019) 等技术兼容。此外，在没有缓存命中时，它引入的内存和时间开销可以忽略不计。

**RadixAttention。** 基数树是一种数据结构，可作为经典字典树（前缀树）的空间高效替代方案。与典型的树不同，基数树的边不仅可以标记单个元素，还可以标记不同长度的元素序列，从而显著提高效率。在我们的系统中，我们利用基数树来管理 Token 序列及其相应 KV cache 张量之间的映射。这些 KV cache 张量以非连续的分页布局存储，其中每个页面的大小相当于一个 Token。由于 GPU 内存很快被 KV cache 填满，我们引入了一个简单的 LRU 驱逐策略，该策略优先驱逐最近最少使用的叶子节点。通过优先驱逐叶子节点，我们使其公共祖先节点能够被重用，直到这些祖先节点成为叶子节点并被驱逐。

在 continuous batching 设置中，我们无法驱逐当前运行批次正在使用的节点。因此，每个节点都维护一个引用计数器，指示有多少正在运行的请求正在使用它。如果节点的引用计数器为零，则该节点是可驱逐的。请注意，我们不会预先分配固定大小的内存池作为缓存。相反，我们让缓存的 Token 和当前运行的请求共享同一个内存池。因此，系统会为缓存和运行中的请求动态分配内存。当足够多的等待请求运行时，系统将驱逐所有缓存的 Token，以支持更大的批次大小。[fig:example_radix_attn] 展示了如何为几个传入请求维护基数树。前端解释器将完整的提示词发送到运行时，运行时执行前缀匹配和重用。树结构存储在 CPU 上，维护开销可以忽略不计。在执行 `fork` 原语期间，前端首先将前缀作为提示发送，确保前缀被正确插入到树中。然后它发送剩余的提示词。这种“前端提示”简化了运行时调度和匹配，体现了前端与运行时协同设计的好处。

带有 LRU 驱逐策略的 RadixAttention 操作示例，通过九个时间点进行说明。该图展示了基数树响应各种请求的动态演变。这些请求包括两个聊天会话、一批少样本学习查询以及一次自一致性采样。每条树边都带有一个表示子字符串或 Token 序列的标签。节点采用颜色编码以反映不同的状态：绿色表示新添加的节点，蓝色表示在该时间点被访问的缓存节点，红色表示已被驱逐的节点。在步骤 (1) 中，基数树初始为空。在步骤 (2) 中，服务器处理传入的用户消息“Hello”并以 LLM 输出“Hi”作为响应。系统提示词“You are a helpful assistant”、用户消息“Hello!”和 LLM 回复“Hi!”被合并到树中，作为链接到新节点的一条边。在步骤 (3) 中，一个新提示词到达，服务器在基数树中找到该提示词的前缀（即对话的第一轮）并重用其 KV cache。新的一轮作为新节点附加到树中。在步骤 (4) 中，一个新的聊天会话开始。(3) 中的节点“b”被拆分为两个节点，以允许两个聊天会话共享系统提示词。在步骤 (5) 中，第二个聊天会话继续。然而，由于内存限制，必须驱逐 (4) 中的节点“c”。新的一轮被附加到 (4) 中节点“d”的后面。在步骤 (6) 中，服务器接收到一个少样本学习查询，对其进行处理，并将其插入到树中。根节点被拆分，因为新查询不与现有节点共享任何前缀。在步骤 (7) 中，服务器接收到一批额外的少样本学习查询。这些查询共享同一组少样本示例，因此我们拆分 (6) 中的节点“e”以实现共享。在步骤 (8) 中，服务器接收到来自第一个聊天会话的新消息。它驱逐了第二个聊天会话的所有节点（节点“g”和“h”），因为它们是最近最少使用的。在步骤 (9) 中，服务器接收到一个请求，要求对 (8) 中节点“j”的问题进行更多答案的采样，这可能是用于自一致性提示。为了为这些请求腾出空间，我们驱逐了 (8) 中的节点“i”、“k”和“l”。

**缓存感知调度。** 我们将缓存命中率定义为 $\frac{\text{number of cached prompt tokens}}{\text{number of prompt tokens}}$。当等待队列中有许多请求时，它们的执行顺序会显著影响缓存命中率。例如，如果请求调度器频繁在不同的、不相关的请求之间切换，可能会导致缓存抖动和较低的命中率。我们设计了一种缓存感知调度算法来提高缓存命中率。在批处理设置中，我们根据匹配的前缀长度对请求进行排序，并优先处理具有较长匹配前缀的请求，而不是使用先到先得的调度策略。[alg:cache_aware_scheduling] (Appendix) 展示了具有连续批处理的缓存感知调度的伪代码。该算法使用最长共享前缀优先的顺序。在对延迟更敏感的设置中，我们可能仍然能够容忍有限的批次重排序以改善缓存重用。此外，我们证明了以下关于离线情况下最优调度的定理。（在实践中，计算与 [thm:optimal] 证明中描述的不同，因为不可预测的输出 Token 数量可能会导致 KV cache 的重新计算。）

theoremoptimal  对于一批请求，我们可以通过以深度优先搜索顺序访问请求的基数树来实现最佳缓存命中率，其中缓存大小 $\geq$ 最大请求长度。最长共享前缀优先顺序等同于深度优先搜索顺序。

证明在 [subsec:schedule_proof]（附录）中。在在线情况下，DFS 顺序会被打乱，但我们的调度仍然近似于完整基数树（radix tree）扩充部分上的 DFS 行为，如 [subsec:schedule_proof] 所述。虽然贪婪的缓存感知调度可以实现高吞吐量，但它可能导致饥饿。我们将其与其他公平调度方法 (Sheng, Cao, et al. 2023) 的集成留作未来工作。

**分布式情况。** RadixAttention 可以扩展到多个 GPU。对于张量并行，每个 GPU 维护一个分片的 KV cache。由于树操作相同，因此不需要额外的同步。带有多个 worker 的数据并行在 [subsec:distributed_radix_attention]（附录）中讨论。

# 使用压缩有限状态机的高效约束解码

在 LM 程序中，用户通常希望约束模型的输出以遵循特定格式，例如 JSON schema。这可以提高可控性和鲁棒性，并使输出更容易解析。SGLang 提供了一个 `regex` 参数来使用正则表达式强制执行此类约束，这对于许多实际场景来说已经足够具有表现力。现有系统通过将正则表达式转换为有限状态机（FSM）(Willard and Louf 2023) 来支持这一点。在解码期间，它们维护当前的 FSM 状态，从下一个状态中检索允许的 token，并将无效 token 的概率设置为零，逐个 token 进行解码。然而，当有机会一次解码多个 token 时，这种逐个 token 的方法效率低下。例如，[fig:example_first_program] 中的常量序列 `{"summary": "` 在正常解码过程中跨越多个 token，如 [fig:example_compressed_fsm] (c) 所示，需要多个解码阶段，即使解码时只有一个有效的下一个 token。因此，整个序列可以在单步（即前向传递）中解码。然而，现有系统一次只能解码一个 token，因为现有系统中 FSM 和模型运行器之间缺乏集成，阻碍了多 token 处理，导致解码缓慢。

正常和压缩 FSM 的解码过程（下划线 _ 表示空格）。

SGLang 通过创建一个带有压缩 FSM 的快速约束解码运行时来克服这一限制。该运行时分析 FSM 并将 FSM 中相邻的单一转换边压缩为单条边，如 [fig:example_compressed_fsm] (b) 所示，使其能够识别何时可以一起解码多个 token。在 [fig:example_compressed_fsm] (d) 中，压缩转换边上的多个 token 可以在一次前向传递中解码，这极大地加速了解码过程。它也具有通用性，适用于所有正则表达式。关于背景和实现的更多细节在 [sec:additional_fsm] 中。

# 使用 API 推测执行的高效端点调用

前面的章节介绍了对开放权重模型的优化，这需要修改模型推理过程。此外，SGLang 适用于仅限 API 访问的模型，例如 OpenAI 的 GPT-4。然而，对于这些模型，我们只能调用黑盒 API 端点。

本节介绍了一种针对黑盒 API 模型的新优化，该优化使用推测执行加速执行并降低多调用 SGLang 程序的 API 成本。例如，程序可能要求模型使用多调用模式生成角色的描述：`s += context + "name:" + gen("name", stop="\n") + "job:" + gen("job", stop="\n")`。简单来说，这两个 `gen` 原语对应于两次 API 调用，这意味着用户需要为 `context` 上的输入 token 费用支付两次。在 SGLang 中，我们可以在第一次调用时启用推测执行，并通过忽略停止条件让它继续生成更多 token。解释器保留额外的生成输出，并将它们与后续原语进行匹配和重用。在某些情况下，通过仔细的提示工程，模型可以以高精度正确匹配模板，从而为我们节省一次 API 调用的延迟和输入成本。

# 评估

我们评估了 SGLang 在不同 LLM 工作负载下的性能。随后，我们进行了消融研究和案例研究，以证明特定组件的有效性。SGLang 使用 PyTorch (Paszke et al. 2019) 实现，并带有来自 FlashInfer (Z. Ye et al. 2024) 和 Triton (Tillet, Kung, and Cox 2019) 的自定义 CUDA 内核。

## 设置

**模型。** 我们测试了密集型 Llama-2 模型 (Touvron et al. 2023)、稀疏混合专家 Mixtral 模型 (Jiang et al. 2024)、多模态 LLaVA 图像 (H. Liu et al. 2023) 和视频模型 (Zhang et al. 2024)，以及 API 模型 OpenAI 的 GPT-3.5。对于开放权重模型，参数数量从 70 亿到 700 亿不等，我们使用 float16 精度。

**硬件。** 我们在 AWS EC2 G5 实例上运行大部分实验，这些实例配备了 NVIDIA A10G GPU (24GB)。我们在单个 A10G GPU 上运行 7B 模型，并使用张量并行 (Shoeybi et al. 2019) 在多个 A10G GPU 上运行更大的模型。我们在 A100G (80GB) GPU 上运行了一些额外的实验。

**基线。** 我们将 SGLang 与具有各自语言和默认运行时的高级编程系统，以及具有标准类 OpenAI Completion API 的低级推理引擎进行比较。除非另有说明，否则我们不会开启会改变计算结果的优化，以便所有系统计算出相同的结果。基线包括：

- Guidance(Guidance AI, n.d.)，一种用于控制 LLM 的语言。我们使用 Guidance v0.1.8 和 llama.cpp 后端。

- vLLM (Kwon et al. 2023)，一种高吞吐量推理引擎。我们使用 vLLM v0.2.5 及其默认 API 服务器（RadixAttention 已作为可选实验功能部分集成到最新版本的 vLLM 中；因此，我们使用了早期版本进行比较。）。

- LMQL (Beurer-Kellner, Fischer, and Vechev 2023)，一种查询语言。我们使用 LMQL v0.7.3 和 Hugging Face Transformers 后端。

**工作负载。** 我们测试以下内容：5-shot MMLU (Hendrycks et al. 2020) 和 20-shot HellaSwag (Zellers et al. 2019) 基准测试。我们为 MMLU 解码一个 token，并使用原语 `select` 为 HellaSwag 选择概率最高的答案。对于 ReAct agent (Yao et al. 2022) 和生成式 agent (Park et al. 2023)，我们从原论文中提取轨迹并重放它们。我们使用 Tree-of-thought (Yao et al. 2023) 解决 GSM-8K 问题，并使用 Skeleton-of-thought (Ning et al. 2024) 生成提示。我们使用带有 branch-solve-merge (Saha et al. 2023) 技术的 LLM 评判器；使用正则表达式指定 schema 的 JSON 解码；4 轮多轮对话，其中每轮的输入在 256-512 个 token 之间随机采样。多轮对话（短）意味着短输出（4-8 个 token），多轮对话（长）意味着长输出（256-512 个 token）；DSPy 检索增强生成 (RAG) 流水线 (Khattab et al. 2023) 及其官方示例。

**指标。** 我们报告两个性能指标：吞吐量和延迟。对于吞吐量，我们运行足够大的程序实例批次来计算最大吞吐量，比较每秒执行的程序实例数（programs per second, p/s）。对于延迟，我们一次执行单个程序而不进行批处理，并报告多个实例的平均延迟。

Llama-7B 模型上的归一化吞吐量。越高越好。

Llama-7B 模型上的归一化延迟。越低越好。

## 端到端性能

**开源权重模型上的结果。** 延迟和吞吐量结果如[fig:exp_llama_7b_throughput]和[fig:exp_llama_7b_latency]所示。SGLang将吞吐量提高了最高$6.4\times$，并将延迟降低了最高$3.7\times$。这些提升得益于 KV cache 复用、利用单程序内的并行性以及更快的约束解码。接下来，我们解释每个基准测试中加速的原因。

在 MMLU 上，SGLang 可以利用 RadixAttention 复用 5-shot 示例的 KV cache。RadixAttention 同时有利于吞吐量和延迟。RadixAttention 通过共享 KV cache 来减少总内存使用量，从而允许更大的 batch size 以提高最大吞吐量。RadixAttention 还减少了 prefill 的计算量，从而降低了首个 token 的延迟。在 HellaSwag 上，SGLang 复用了 few-shot 示例和多项选择的公共问题前缀的 KV cache，实现了两级共享。对于 ReAct 和生成式智能体，SGLang 复用了智能体模板和先前调用的 KV cache。在 Tree-of-thought 和 Skeleton-of-thought 上，SGLang 并行化了单个程序内的生成调用，并尽可能复用 KV cache。在 JSON 解码中，SGLang 通过使用压缩的有限状态机一次解码多个 token 来加速解码。在多轮对话中，SGLang 复用了对话历史的 KV cache。对于短输出，加速效果更明显，因为 KV cache 复用主要有助于减少前缀时间。对于长输出，由于不同对话会话之间没有太多共享，且解码时间占主导地位，因此几乎没有加速。在 DSPy RAG 流水线中，SGLang 复用了公共上下文示例的 KV cache。在这些基准测试中，缓存命中率从 50% 到 99% 不等。[fig:exp_cache_hit_rate]（附录）列出了所有测试的已实现和最佳缓存命中率，表明我们的缓存感知调度平均达到了最佳命中率的 96%。

由于性能缓慢和功能缺失，我们将 LMQL 和 Guidance 从最后五个基准测试中排除。LMQL 的问题源于缓慢的 token 级处理和未优化的后端，而 Guidance 缺乏批处理和并行支持。

使用张量并行的 Mixtral-8x7B 模型上的归一化吞吐量。越高越好。

**使用张量并行的更大模型上的结果。** 我们在相同的基准测试集上使用张量并行运行更大的模型 Mixtral-8x7B 和 Llama-70B，并在[fig:exp_mixtral_8x7b_throughput]和[fig:exp_llama_70b_throughput]（附录）中报告结果。在更大模型上的加速显示出与在较小模型上观察到的相似趋势，表明我们的优化能很好地推广到更大的模型。我们在这里省略了 Guidance 和 LMQL，因为它们缺乏张量并行的高效实现。

**多模态模型上的结果。** SGLang 原生支持带有 `image` 和 `video` 原语的多模态模型。本文中的优化与多模态模型兼容。对于 RadixAttention，我们计算输入图像的哈希值并将其用作基数树中的键，使我们能够复用来自同一图像的图像 token 的 KV cache。我们在 llava-bench-in-the-wild 上运行 LLaVA-v1.5-7B（图像），并在 ActivityNet 上运行 LLaVA-NeXT-34B（视频）。由于这些模型未被其他基线系统很好地支持，我们使用模型作者在 Hugging Face Transformers 中的原始实现作为基线。如[tab:e2e_multi_modal]所示，SGLang 在这些基准测试上的吞吐量提高了最高$6\times$。在 llava-bench-in-the-wild 中，对同一张图像有多个问题，SGLang 运行时在这种情况下会复用 KV cache。

**生产部署。** SGLang 已部署在 Chatbot Arena (Chiang et al. 2024) 中以服务开源权重模型。由于某些模型的流量较低，每个模型仅由一个 SGLang worker 服务。一个月后，我们观察到 LLaVA-Next-34B (H. Liu et al. 2024) 的 RadixAttention 缓存命中率为 52.4%，Vicuna-33B (Chiang et al. 2023) 为 74.1%。缓存命中来自于公共系统消息、频繁复用的示例图像以及多轮对话历史。这使得 Vicuna-33B 的首个 token 延迟平均降低了$1.7\times$。

**API 模型上的结果。** 我们测试了一个使用 OpenAI 的 GPT-3.5 模型从维基百科页面提取三个字段的提示。通过使用 few-shot 提示，API 推测执行的准确率很高，并且由于提取了三个字段，它将输入 token 的成本降低了约三倍。

(a)(b) 缓存命中率消融研究。(c) RadixAttention 消融研究。

## 消融研究

**缓存命中率与延迟/吞吐量的对比。** [fig:exp_hit_rate_vs_perf](a)(b) 展示了在 tree-of-thought 基准测试上缓存命中率与性能指标（首个 token 延迟、总延迟、batch size 和吞吐量）之间的关系。该图是通过在运行时部分禁用匹配的 token 获得的。它表明更高的缓存命中率会带来更大的 batch size、更高的吞吐量和更低的延迟。

**RadixAttention 的有效性。** 我们在几个具有代表性的基准测试上测试了 RadixAttention 及其组件的有效性。如[fig:exp_hit_rate_vs_perf](c)所示，“No Cache”表示不使用任何缓存，“No Tree-Structure”表示使用简单的基于表的缓存而不是树状结构的缓存，“FCFS Schedule”表示使用先来先服务策略而不是我们的缓存感知调度，“Random Schedule”表示使用随机顺序调度请求，“No Frontend Parallelism”表示禁用解释器中的并行性，“No Frontend Hint”表示禁用从解释器发送 fork 提示，而“Full optimizations”表示我们开启所有优化。实验结果表明，要实现最佳性能，这些组件都是必需的。禁用前端解释器的并行性和提示也会导致运行时性能不佳，这突出了协同设计前端语言和运行时的重要性。

**RadixAttention 的开销。** 我们在没有 KV cache 复用机会的基准测试上测试了 RadixAttention 的开销。该基准测试测量在 ShareGPT 数据集上的吞吐量。运行 100 个请求需要 74.3 秒；然而，用于管理 RadixAttention 数据结构的时间仅为 0.2 秒，这是一个不到 0.3% 的可忽略的开销。这是因为树操作的复杂度是线性的且很小。因此，我们可以默认开启 RadixAttention。

**压缩有限状态机的有效性。** 我们在 JSON 解码基准测试上测试了压缩有限状态机及其组件的有效性。实验结果表明，压缩有限状态机将吞吐量提高了$1.6\times$，因为它可以一次解码多个 token。此外，我们需要对状态机进行预处理，并将其用于一批请求。否则，对每个请求重新进行预处理会使吞吐量降低$2.4\times$。

# 相关工作

许多研究已经探索了 KV cache 的重用，其中许多与我们的工作并行。独特的是，我们的 RadixAttention 首次提出将 KV cache 视为基于树的 LRU 缓存。它是第一个支持多级共享、缓存感知调度、前端-运行时协同调度以及分布式情况的解决方案。vLLM (Kwon et al. 2023) 和 ChunkedAttention (L. Ye et al. 2024) 探索了一些简单的重用案例（例如，系统提示共享），但没有涵盖多级树结构共享或 LRU 缓存。PromptCache (Gim et al. 2023) 提出了前缀之外的 KV cache 模块化重用，但可能导致高达 43% 的精度下降。HydraGen (Juravsky et al. 2024)、FlashInfer (Z. Ye et al. 2024) 和 ChunkedAttention (L. Ye et al. 2024) 专注于 CUDA 内核优化，不包含 LRU 缓存的概念。API Serve (Abhyankar et al. 2024) 和 LLM-SQL (S. Liu et al. 2024) 研究了特定应用的 KV cache 重用，例如与外部 API 调用和关系数据库的交错，但它们没有我们的基数树或缓存感知调度。

目前存在多种 LLM 编程和智能体框架，例如 Guidance (Guidance AI, n.d.)、LMQL (Beurer-Kellner, Fischer, and Vechev 2023)、DSPy (Khattab et al. 2023)、LangChain (LangChain AI, n.d.)、AutoGen (Wu et al. 2023) 和 LLM Compiler (Kim et al. 2023)。Guidance 和 LMQL 与 SGLang 最为相似，我们在 [sec:programming_model] 中对它们进行了比较。我们的创新在于通过新颖的运行时优化来加速所提出的编程模型。SGLang 与其他框架兼容，并且可以加速它们（例如，我们评估中的 DSPy 示例）。此外，SGLang 兼容许多其他常见的推理优化 (Yu et al. 2022; Pope et al. 2023; Aminabadi et al. 2022; Kwon et al. 2023; Z. Ye et al. 2024; Dao et al. 2022; Lin et al. 2023; Hooper et al. 2024; Kang et al. 2024; Zirui Liu et al. 2024; Zichang Liu et al. 2024; Ge et al. 2023)。

# 未来方向与结论 

**未来方向。** 尽管 SGLang 取得了进展，但仍存在一些局限性，为未来的研究揭示了有前景的方向。这些方向包括扩展 SGLang 以支持额外的输出模态，使 RadixAttention 能够跨内存层次的多个层级（例如，DRAM、磁盘）运行 (Sheng, Zheng, et al. 2023)，在 RadixAttention 中启用模糊语义匹配，在 SGLang 之上提供更高级的原语，修复缓存感知调度中的饥饿问题 (Sheng, Cao, et al. 2023)，以及增强 SGLang 编译器以执行高级静态优化（如调度和内存规划）。

**结论。** 我们介绍了 SGLang，这是一个用于高效编程和执行结构化语言模型程序的框架。SGLang 通过 RadixAttention、压缩有限状态机和语言解释器等新颖优化，显著提高了复杂 LM 程序的吞吐量和延迟。它是开发高级提示技术和智能体工作流的有价值工具。源代码已公开。

# 致谢

本项目得到了斯坦福自动推理中心以及来自 Astronomer、Google、IBM、Intel、Lacework、Microsoft、穆罕默德·本·扎耶德人工智能大学、Nexla、Samsung SDS、Uber 和 VMware 的赠款支持。Lianmin Zheng 获得了 Meta 博士奖学金的支持。我们感谢 Yuanhan Zhang 和 Bo Li 对 LLaVA-NeXT (video) 的支持。

# RadixAttention 的更多细节

## KV Cache 背景

如今使用的大多数 LLM，如 GPT-3 (Brown et al. 2020)、PaLM (Chowdhery et al. 2022) 和 LLaMA (Touvron et al. 2023)，都基于自回归 Transformer 架构 (Vaswani et al. 2017)。这些模型基于前序 token 来预测序列中下一个 token 的概率。在推理期间，模型首先通过前向传播处理输入 token 序列（此过程称为“预填充”）。然后，它顺序解码输出 token，每个 token 都依赖于先前的 token（此过程称为“解码”）。我们将接收输入 token 序列并生成输出 token 序列的过程称为单次生成调用。在整个过程中，每个 token 都会生成一些中间张量，用于解码后续 token。这些中间张量被称为“KV Cache”，以自注意力机制中的键值对命名。在讨论本文中的优化时，一个重要的观察是，KV cache 的计算仅依赖于所有先前的 token，因此具有相同前缀的不同序列可以重用前缀 token 的 KV cache，从而避免冗余计算。

通常在 LLM 程序中，多个文本段和生成调用会被附加到单个提示中。在多个链式调用中为先前 token 缓存已计算的 KV cache 可以减少冗余计算。然而，这种优化既不是免费的，也不是微不足道的，因为它需要额外的存储和更复杂的内存管理。此外，在 LLM 程序中，从单个提示生成多个输出或从当前状态分叉出新提示是很常见的 (Li et al. 2022)。基础前缀共享已在 vLLM (Kwon et al. 2023) 中得到研究。也可以采用更高级的共享模式，如不规则树结构共享。[fig:example_sharing] 展示了跨多次调用的四种典型 KV cache 共享模式；现有系统中没有一个能自动处理所有这些模式。相反，我们在 [sec:radix_attention] 中的 RadixAttention 可以在运行时自动处理所有这些模式。

KV cache 共享示例。蓝色框代表可共享的提示部分，绿色框表示不可共享的部分，黄色框标记不可共享的模型输出。可共享元素包括少样本学习示例、自一致性中的问题 (X. Wang et al. 2022)、多轮对话中的聊天历史以及思维树中的搜索历史 (Yao et al. 2023)。

基数树 $T$，内存池 $P$，当前运行的批次 $B$，等待队列 $Q$。完成的请求和更新的系统状态。

$requests \leftarrow Q.\text{get\_all\_requests}()$

$req.prefix\_node, req.prefix\_len \leftarrow$ $T.\text{match\_prefix}(req.input\_tokens)$

$requests.\text{sort}()$

$available\_size \leftarrow T.\text{evictable\_size}() + P.\text{available\_size}()$ $current\_size \leftarrow 0$ $new\_batch \leftarrow []$

$new\_batch.\text{append}(req)$ $delta \leftarrow T.\text{increase\_ref\_counter}(req.prefix\_node)$ $available\_size \leftarrow available\_size + delta$ $Q.\text{remove\_requests}(new\_batch)$

$B.\text{merge}(new\_batch)$

$needed\_size \leftarrow B.\text{needed\_size}()$ $success, buffer \leftarrow P.\text{alloc}(needed\_size)$ $T.\text{evict}(needed\_size)$ $success, buffer \leftarrow P.\text{alloc}(needed\_size)$

$B.\text{run}(buffer)$

$finished\_requests \leftarrow B.\text{drop\_finished\_requests}()$ $T.\text{decrease\_ref\_counter}(req.prefix\_node)$ $T.\text{insert}(req)$

**return** $finished\_requests$

## 缓存感知调度的伪代码

[alg:cache_aware_scheduling] 展示了带有连续批处理的 RadixAttention 缓存感知调度的伪代码。

## [thm:optimal] 的证明 

*证明.* 首先，我们证明深度优先搜索 (DFS) 顺序能够实现最佳的缓存命中率。设 $R$ 表示批次中的请求集合，$T$ 表示由 $R$ 构建的基数树。对于 $T$ 的每条边 $e$，与 $e$ 关联的 KV cache 至少需要计算一次。设 $|e|$ 表示与 $e$ 关联的 KV cache 的大小。设 $C$ 表示 $R$ 的 KV cache 的计算复杂度。我们得到下界 $$C \geq \sum_{e \in \text{edges}(T)} |e|.$$

考虑我们以 DFS 顺序遍历基数树 $T$。对于 $T$ 的每条边 $e$，当我们第一次计算与 $e$ 关联的 KV cache 时，我们将接着计算 $e$ 的整个子树。在计算 $e$ 的子树期间，边 $e$ 将被连续命中，因此不会发生额外的计算。在完成以 $e$ 为根的子树的计算后，边 $e$ 将不会被再次访问。注意，在缓存大小 $\geq$ 最大请求长度（等于基数树 $T$ 中的最长路径）的情况下，边 $e$ 在其子树的计算过程中不会被驱逐，因为包含 $e$ 的子树的公共前缀将被连续命中。因此，与每条边 $e$ 关联的 KV cache 将只被计算一次。于是，我们达到了下界 $$C = \sum_{e \in \text{edges}(T)} |e|.$$

缓存命中率定义为 $$\frac{\sum_{r\in R}\text{number of cached prefill tokens in $r$}}{\sum_{r\in R}\text{number of prefill tokens in $r$}},$$ 等于 $1 - \frac{C}{\sum_{r\in R}\text{number of prefill tokens}}$，达到其上界，从而实现了最优。

接下来，我们通过归纳法证明最长共享前缀优先顺序等价于 DFS 顺序。

- （基例）在开始时，由于没有任何缓存，将处理一个与 $T$ 中节点 $x$ 对应的随机请求。所有与从根到 $x$ 路径上的节点 $\{v_1, ..., v_n\}$ 对应的请求都不需要重新计算。与节点 $\{v_1,..., v_n, x\}$ 对应的请求的计算复杂度与有效的 DFS 相一致。从根到 $x$ 的路径被缓存。

- （归纳）假设我们刚刚访问了 $T$ 中的节点 $y$，并且访问过的节点符合 DFS 顺序。设 $P$ 表示从根到 $y$ 的路径。那么每个未被访问的节点与 $P$ 上的已访问节点都有一个最低公共祖先。由于 $P$ 上的节点已被缓存，一个在 $P$ 上具有最低公共祖先且未被访问的节点 $z$ 将具有最长的共享前缀。最长共享前缀优先顺序将选择 $z$，这是一个有效的 DFS 顺序。从根到 $z$ 的路径将被缓存，因为它是最近被访问的。

 ◻

在在线情况下，DFS 顺序将被打乱，但最长共享前缀调度仍然近似于隐含基数树在新增部分上的 DFS 行为。我们通过考虑添加新批次请求的步骤来证明这一点。

设 $T$ 表示迄今为止已访问的基数树部分，$T'$ 表示添加新批次请求后的整个新基数树。设 $C$ 表示 $T$ 中已缓存节点的集合。设 $\text{longest}(C)$ 表示 $C$ 中从根出发路径最长且在 $T'$ 中具有未被完全访问的子树的节点。

最长共享前缀调度随后将以 DFS 顺序处理 $T'$ 中以 $\text{longest}(C)$ 为根的子树。在此过程中，可能会发生驱逐，$C$ 中剩余的缓存节点变为 $C^{(1)} \subseteq C$。随后，将对 $T'$ 中以 $\text{longest}(C^{(1)})$ 为根的子树进行 DFS。

类似地，我们将得到 $C^{(2)}, \ldots, C^{(k)}$，直到 $C^{(k)}$ 仅包含 $C^{(k)}$ 中一个在 $T'$ 中子树未被完全访问的叶节点。此时，我们已达到有效的 DFS 状态。$T'$ 的剩余部分将按照 [thm:optimal] 证明中所述的 DFS 顺序进行访问。

## 数据并行分布式 RadixAttention

为了使 RadixAttention 适应具有多个副本工作节点的分布式环境（即数据并行），我们开发了一种机制，其中每个工作节点维护自己的子树，而路由器负责管理一个元树。这个元树充当一个 trie，用于跟踪所有子树及其关联的设备。当新的请求批次到达路由器时，会在元树上执行前缀匹配。我们根据每个请求的亲和力（通过与其他特定工作节点及同一组中其他请求的共享前缀长度来衡量）实现了各种策略，以做出高效的调度决策，从而最小化冗余计算。每次处理新请求时，路由器和工作节点都会独立更新各自的树。如果工作节点发生驱逐，它会将此驱逐提交到一个队列中，路由器随后在低活动期间处理该队列以更新元树。我们使用四个工作节点和 MMLU 数据集对这种分布式配置进行了基准测试，观察到它实现了线性扩展和最佳的缓存命中率，且这种弱一致性的分布式缓存设计带来的开销极小。在最大化数据局部性和并行处理效率之间存在权衡。探索高级调度策略以优化这种权衡被指定为未来的研究方向。此外，来自 Preble (Srivatsa et al. 2024) 的并发工作研究了基于 SGLang 早期版本的数据并行调度。

# 压缩有限状态机的额外细节

本节讨论用于更快约束解码的压缩有限状态机的背景和实现细节。我们旨在让 LLM 遵循正则表达式，这提供了更强的表达能力，可用于表示常见的格式，如 JSON schemas。为了实现这一点，我们将正则表达式转换为有限状态机 (FSM)，以在解码过程中指导生成过程 (Willard and Louf 2023)。FSM 本质上是一个包含节点（状态）和边（带有字符串/字符的转移）的图。从初始状态开始，每次转移都会在边上附加字符串以移动到下一个状态，并使用一组最终状态来结束该过程。此机制指导 LLM 的解码，根据 FSM 当前状态的转移过滤掉无效的 token，如 [fig:fsm_demo] 所示。解码可能涉及 FSM 中的多次转移，直到达到最终状态。

正则表达式如何转换为 FSM 以及 FSM 如何指导解码过程的示例。

约束解码的挑战源于这样一个事实：约束通常以自然语言格式表达，即正则表达式由字符/字符串描述，而 LLM 被设计为将这些解释并处理为 token。这创造了一个复杂的场景，因为字符串和 token 之间的映射错综复杂且缺乏一一对应关系 (Kuchnik, Smith, and Amvrosiadis 2023)。

本节源自较早的一篇博客文章 (<https://lmsys.org/blog/2024-02-05-compressed-fsm/>)。也鼓励读者阅读该博客文章以获取更多背景知识并便于理解。

## 压缩有限状态机的实现细节

为了简化压缩有限状态机（Compressed FSM）的构建，我们在字符/字符串而不是 Token 上构建原始 FSM。我们正式定义单一转移边和压缩边的概念如下：

- 单一转移边：如果一条边满足 1) 其源节点只有一个后继节点，且 2) 该边上只有一个可接受的字符/字符串，则该边为单一转移边。

- 压缩边：一条边由若干连续相邻的边 $(e_0, e_1, \ldots, e_k)$ 压缩而成，当且仅当 $e_1, \ldots, e_k$ 均为单一转移边。压缩边的文本是 $e_0, e_1, \ldots, e_k$ 文本的拼接。

从基于字符的 FSM 出发，我们递归地将单一转移边合并到其前继边中，直到无法进一步压缩，从而得到压缩 FSM。这种方法加速了解码过程，正如 SGLang 在压缩 FSM 上的运行时效率所展示的那样，见 [fig:compressed_fsm]。

使用压缩 FSM 与普通 FSM 解码的对比：左子图描绘了每次前向传播的解码过程，而右子图解释了各结果组件的来源。

## 通过重新分词处理分词伪影

当生成一个新 Token 时，我们获取该 Token 的字符串，并搜索当前状态的所有出边，找到以刚刚解码的字符串开头的那一条，然后向前移动。然而，当转移边是一条被充分压缩且包含很长字符串的边时，我们也可以预期接下来几轮解码出的字符串。这就是加速发生的地方，我们将此过程称为 *Jump Forward*。然而，我们仍然需要将字符串转换为 Token 以进行下一阶段的解码，由于 LLM 特定的预训练和分词方法，这并非易事；直接切分可能会改变预期的含义 (Tran-Thien 2024)。例如，[fig:example_first_program] 正则表达式中的压缩文本是 `{"summary": "`，根据分词器，它只能被分词为 `{"`、`summary`、`":` 和 `_"`，而不是像 `{"`、`summa`、`ry` 和 `":_"` 这样随机切分。为了解决这个问题，我们使用原始分词器对所有先前的文本以及压缩边的文本进行重新分词，以确保与 LLM 的原始输入格式对齐。并且这只会带来微小的重新分词开销。

## 未来扩展：解决概率失真问题

字符串与 Token 之间差距引起的挑战也带来了概率分布偏斜的问题 (Tran-Thien 2024)。例如，在 [fig:example_first_program] 中，正则表达式 `"[ABCD][+-]?"` 表示从 `A+` 到 `D-` 的等级，但如果替换为更宽泛的术语如 `Excellent|Above Average|Fair|Below Average`，运行时可能会因为 `Above Average` 术语处于压缩转移上而错误地将 `A` 映射到 `Above Average`，从而歪曲了等级层次。发生这种情况是因为 LLM 没有识别出特定的选择范围，导致了不合适的 Token 序列。计算每个选项的准确概率需要对导致每个选项的所有 Token 序列的概率进行求和，这使得解码复杂化并增加了开销。一种权宜之计是将选项或正则表达式直接包含在预填充提示中，引导 LLM 意识到其选项并以正确的 Token 序列输出决策。然而，这种方法并没有解决概率失真的根本问题，凸显了进一步研究以提高压缩 FSM 准确性的必要性。

# 附加实验设置与结果

**附加实验设置。** [fig:exp_llama_7b_throughput] 和 [fig:exp_llama_7b_latency] 是通过在单张 A10G (24GB) GPU 上运行 Llama-7B 获得的。[fig:exp_mixtral_8x7b_throughput] 是通过在 8 张 A10G (24GB) GPU 上使用张量并行运行 Mixtral-8x7B 获得的。[fig:exp_hit_rate_vs_perf](c) 是通过在单张 A10G (24GB) GPU 上运行 Llama-7B 获得的。[fig:exp_llama_70b_throughput] 是通过在 4 张 A100G (80GB) GPU 上使用张量并行运行 Llama-70B 获得的。[tab:e2e_multi_modal] 是通过在单张 A10G (24GB) GPU 上运行 LLaVA-v1.5-7B 并在单张 A100 (80GB) GPU 上运行 LLaVA-Next-34B 获得的。基准测试图中的每个柱状图需要几分钟到一小时的时间运行。

**附加实验结果。** [fig:exp_cache_hit_rate] 展示了在 [fig:exp_llama_7b_throughput] 列出的基准测试上达到的缓存命中率和最佳缓存命中率。[fig:exp_llama_70b_throughput] 展示了使用张量并行的 Llama-2-70B 的吞吐量。

使用张量并行的 Llama-2-70B 模型上的归一化吞吐量。越高越好。

在各种基准测试上达到的缓存命中率和最佳缓存命中率。

带有思维骨架提示的并行提示建议的 SGLang 程序。
[fig:example_tip_suggestion] 中程序的计算图。这三个流对应于三个函数调用。
一个 SGLang 程序及其对应的数据流图。

# 编译器模式

除了论文主体部分使用的解释器模式外，运行 SGLang 程序的另一种方式是将它们编译为计算图，并使用图执行器执行。这为更多的编译优化提供了机会，因为我们可以重写图并执行更多的静态规划。

## 设计与实现

我们为 SGLang 设计了一种中间表示（IR），它将 SGLang 程序结构和操作表示为计算图。该图包含代表基本算子的节点和代表依赖关系的边。对应于 [fig:example_tip_suggestion] 中程序的计算图见 [fig:example_dataflow_graph]。在程序中，每次对被装饰函数或 fork 的调用都会创建一个新的提示状态或流。

节点有多种类型。SGLang 程序中算子 `+=` 和 `+` 的每个操作数都由一个 IR 节点表示。这些节点包括 `ConstantText`、`Argument`、`Gen`、`Select`、`Variable`、`Fork`、`GetForkItem` 和 `Join`。依赖关系有两种类型：流内依赖，即使用 `+=` 提交到流中的操作必须在该流中所有前序操作之后执行；以及流间依赖，即当一个流需要从另一个流获取变量值时发生，需要进行同步。像 fork 这样的操作会操纵多个流，因此会引入流间依赖。

为了生成图，我们使用追踪技术，用抽象参数运行程序并动态构建图。此方法仅限于没有数据依赖控制流的程序，这是我们计划在未来工作中解决的局限性。一旦构建完成，图就可以通过图执行器执行，从而无需重新解释原始 Python 程序。这带来了诸如图重写优化、减少运行时开销和程序序列化等好处。在执行时，会为每个数据流启动流执行器，并按拓扑顺序将 IR 节点分发给各个流。

## 编译器优化案例研究：用于改进前缀共享的代码移动

我们探索了一种针对 SGLang IR 的编译优化：用于改进前缀共享的代码移动。我们预计更多经典的编译器技术也可以被应用，例如自动调优和指令选择。

该优化旨在通过重排序图中的节点来增加常量前缀的长度，从而改进前缀共享。它并不严格保持原始计算，因此被归类为激进优化。例如，将提示从 "Here is a question + {question}. Please act as a math expert and solve the given question." 改为 "Please act as a math expert and solve the given question. Here is a question + {question}." 可以产生更长的可共享前缀。这一优化很有趣，因为由于 SGLang 中存在自然语言指令，传统的程序分析无法实现它。相反，我们通过提示 GPT-4 来重排序图节点。我们编写了一个包含若干示例的提示来教授 GPT-4 SGLang IR 的概念，我们发现 GPT-4 能够成功地对一些简单的 SGLang 程序应用此优化。

我们评估了该优化的有效性。我们从互联网上收集了 20 个提示模板，并在 SGLang 中实现了它们。我们将其中的 5 个模板用作 few-shot 训练示例，其余 15 个作为测试用例。我们的评估表明，在 15 个模板中有 12 个，GPT-4 成功地重排序了图节点且未改变语义，这一点已通过对修改后的提示进行人工检查得到确认。平均而言，该优化使可共享前缀长度增加了 60 个 token，展示了 GPT-4 的有效性。创建优化提示顺序的失败源于对图节点背后语义含义的错误理解。它过于激进，即使这种排序会改变原始语义，也会将所有常量提前。本案例研究旨在探索 GPT-4 在编译器优化中的应用。未来需要更多工作来使这类优化变得可靠。

## References


Abhyankar, Reyna, Zijian He, Vikranth Srivatsa, Hao Zhang, and Yiying Zhang. 2024. “APIServe: Efficient API Support for Large-Language Model Inferencing.” *arXiv Preprint arXiv:2402.01869*.



Alayrac, Jean-Baptiste, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, et al. 2022. “Flamingo: A Visual Language Model for Few-Shot Learning.” *Advances in Neural Information Processing Systems* 35: 23716–36.



Aminabadi, Reza Yazdani, Samyam Rajbhandari, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Olatunji Ruwase, et al. 2022. “DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale.” In *SC22: International Conference for High Performance Computing, Networking, Storage and Analysis*, 1–15. IEEE.



Beurer-Kellner, Luca, Marc Fischer, and Martin Vechev. 2023. “Prompting Is Programming: A Query Language for Large Language Models.” *Proceedings of the ACM on Programming Languages* 7 (PLDI): 1946–69.



Brown, Tom, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, et al. 2020. “Language Models Are Few-Shot Learners.” *Advances in Neural Information Processing Systems* 33: 1877–1901.



Bubeck, Sébastien, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, et al. 2023. “Sparks of Artificial General Intelligence: Early Experiments with Gpt-4.” *arXiv Preprint arXiv:2303.12712*.



Chiang, Wei-Lin, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, et al. 2023. “Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%\* ChatGPT Quality.” <https://lmsys.org/blog/2023-03-30-vicuna/>.



Chiang, Wei-Lin, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, et al. 2024. “Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.” *arXiv Preprint arXiv:2403.04132*.



Chowdhery, Aakanksha, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, et al. 2022. “Palm: Scaling Language Modeling with Pathways.” *arXiv Preprint arXiv:2204.02311*.



Dao, Tri, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. “Flashattention: Fast and Memory-Efficient Exact Attention with Io-Awareness.” *Advances in Neural Information Processing Systems* 35: 16344–59.



Ge, Suyu, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2023. “Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs.” In *The Twelfth International Conference on Learning Representations*.



Gim, In, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. 2023. “Prompt Cache: Modular Attention Reuse for Low-Latency Inference.” *arXiv Preprint arXiv:2311.04934*.



Guidance AI. n.d. “A Guidance Language for Controlling Large Language Models.” <https://github.com/guidance-ai/guidance>.



Hendrycks, Dan, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. “Measuring Massive Multitask Language Understanding.” In *International Conference on Learning Representations*.



Hooper, Coleman, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. 2024. “KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.” *arXiv Preprint arXiv:2401.18079*.



Hugging Face. n.d. “Text Generation Inference.” <https://github.com/huggingface/text-generation-inference>.



Jiang, Albert Q, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, et al. 2024. “Mixtral of Experts.” *arXiv Preprint arXiv:2401.04088*.



Juravsky, Jordan, Bradley Brown, Ryan Ehrlich, Daniel Y Fu, Christopher Ré, and Azalia Mirhoseini. 2024. “Hydragen: High-Throughput LLM Inference with Shared Prefixes.” *arXiv Preprint arXiv:2402.05099*.



Kang, Hao, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, and Tuo Zhao. 2024. “Gear: An Efficient Kv Cache Compression Recipefor Near-Lossless Generative Inference of Llm.” *arXiv Preprint arXiv:2403.05527*.



Khattab, Omar, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, et al. 2023. “DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.” *arXiv Preprint arXiv:2310.03714*.



Kim, Sehoon, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. 2023. “An LLM Compiler for Parallel Function Calling.” *arXiv Preprint arXiv:2312.04511*.



Kuchnik, Michael, Virginia Smith, and George Amvrosiadis. 2023. “Validating Large Language Models with Relm.” *Proceedings of Machine Learning and Systems* 5.



Kwon, Woosuk, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. “Efficient Memory Management for Large Language Model Serving with Pagedattention.” In *Proceedings of the 29th Symposium on Operating Systems Principles*, 611–26.



LangChain AI. n.d. “LangChain.” <https://github.com/langchain-ai/langchain>.



Li, Yujia, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, et al. 2022. “Competition-Level Code Generation with Alphacode.” *Science* 378 (6624): 1092–97.



Lin, Ji, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. 2023. “AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration.” *arXiv Preprint arXiv:2306.00978*.



Liu, Haotian, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023. “Improved Baselines with Visual Instruction Tuning.” *arXiv Preprint arXiv:2310.03744*.



Liu, Haotian, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. “LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge.” <https://llava-vl.github.io/blog/2024-01-30-llava-next/>.



Liu, Shu, Asim Biswal, Audrey Cheng, Xiangxi Mo, Shiyi Cao, Joseph E Gonzalez, Ion Stoica, and Matei Zaharia. 2024. “Optimizing LLM Queries in Relational Workloads.” *arXiv Preprint arXiv:2403.05821*.



Liu, Xiaoxia, Jingyi Wang, Jun Sun, Xiaohan Yuan, Guoliang Dong, Peng Di, Wenhai Wang, and Dongxia Wang. 2023. “Prompting Frameworks for Large Language Models: A Survey.” *arXiv Preprint arXiv:2311.12785*.



Liu, Zichang, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2024. “Scissorhands: Exploiting the Persistence of Importance Hypothesis for Llm Kv Cache Compression at Test Time.” *Advances in Neural Information Processing Systems* 36.



Liu, Zirui, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. 2024. “KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.” *arXiv Preprint arXiv:2402.02750*.



Ning, Xuefei, Zinan Lin, Zixuan Zhou, Zifu Wang, Huazhong Yang, and Yu Wang. 2024. “Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation.” In *The Twelfth International Conference on Learning Representations*. <https://openreview.net/forum?id=mqVgBbNCm9>.



NVIDIA. n.d. “TensorRT-LLM.” <https://github.com/NVIDIA/TensorRT-LLM>.



OpenAI. 2023. “GPT-4 Technical Report.” <https://arxiv.org/abs/2303.08774>.



Park, Joon Sung, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. “Generative Agents: Interactive Simulacra of Human Behavior.” In *In the 36th Annual ACM Symposium on User Interface Software and Technology (UIST ’23)*. UIST ’23. New York, NY, USA: Association for Computing Machinery.



Paszke, Adam, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, et al. 2019. “Pytorch: An Imperative Style, High-Performance Deep Learning Library.” *Advances in Neural Information Processing Systems* 32.



Patil, Shishir G., Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. 2023. “Gorilla: Large Language Model Connected with Massive APIs.” *arXiv Preprint arXiv:2305.15334*.



Pope, Reiner, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2023. “Efficiently Scaling Transformer Inference.” *Proceedings of Machine Learning and Systems* 5.



Saha, Swarnadeep, Omer Levy, Asli Celikyilmaz, Mohit Bansal, Jason Weston, and Xian Li. 2023. “Branch-Solve-Merge Improves Large Language Model Evaluation and Generation.” *arXiv Preprint arXiv:2310.15123*.



Schick, Timo, Jane Dwivedi-Yu, Roberto Dessı̀, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. “Toolformer: Language Models Can Teach Themselves to Use Tools.” *arXiv Preprint arXiv:2302.04761*.



Sheng, Ying, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E Gonzalez, and Ion Stoica. 2023. “Fairness in Serving Large Language Models.” *arXiv Preprint arXiv:2401.00588*.



Sheng, Ying, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. 2023. “FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU.” In *International Conference on Machine Learning*, 31094–116. PMLR.



Shoeybi, Mohammad, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. “Megatron-Lm: Training Multi-Billion Parameter Language Models Using Model Parallelism.” *arXiv Preprint arXiv:1909.08053*.



Srivatsa, Vikranth, Zijian He, Reyna Abhyankar, Dongming Li, and Yiying Zhang. 2024. “Preble: Efficient Distributed Prompt Scheduling for LLM Serving.”



Sumers, Theodore, Shunyu Yao, Karthik Narasimhan, and Thomas Griffiths. 2023. “Cognitive Architectures for Language Agents.” *Transactions on Machine Learning Research*.



Team, Gemini, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, et al. 2023. “Gemini: A Family of Highly Capable Multimodal Models.” *arXiv Preprint arXiv:2312.11805*.



Tillet, Philippe, Hsiang-Tsung Kung, and David Cox. 2019. “Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.” In *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*, 10–19.



Touvron, Hugo, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, et al. 2023. “Llama 2: Open Foundation and Fine-Tuned Chat Models.” *arXiv Preprint arXiv:2307.09288*.



Tran-Thien, Vivien. 2024. “Fast, High-Fidelity LLM Decoding with Regex Constraints.” *Unsupervised Thoughts (Blog)*. <https://vivien000.github.io/blog/journal/llm-decoding-with-regex-constraints.html>.



Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. “Attention Is All You Need.” *Advances in Neural Information Processing Systems* 30.



Wang, Guanzhi, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. “Voyager: An Open-Ended Embodied Agent with Large Language Models.” *arXiv Preprint arXiv: Arxiv-2305.16291*.



Wang, Xuezhi, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. “Self-Consistency Improves Chain of Thought Reasoning in Language Models.” In *The Eleventh International Conference on Learning Representations*.



Willard, Brandon T, and Rémi Louf. 2023. “Efficient Guided Generation for Large Language Models.” *arXiv e-Prints*, arXiv–2307.



Wu, Qingyun, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. 2023. “Autogen: Enabling Next-Gen Llm Applications via Multi-Agent Conversation Framework.” *arXiv Preprint arXiv:2308.08155*.



Yao, Shunyu, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. “Tree of Thoughts: Deliberate Problem Solving with Large Language Models.” *arXiv Preprint arXiv:2305.10601*.



Yao, Shunyu, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2022. “ReAct: Synergizing Reasoning and Acting in Language Models.” In *The Eleventh International Conference on Learning Representations*.



Ye, Lu, Ze Tao, Yong Huang, and Yang Li. 2024. “ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition.” *arXiv Preprint arXiv:2402.15220*.



Ye, Zihao, Lequn Chen, Ruihang Lai, Yilong Zhao, Size Zheng, Junru Shao, Bohan Hou, et al. 2024. “Accelerating Self-Attentions for LLM Serving with FlashInfer.” <https://flashinfer.ai/2024/02/02/introduce-flashinfer.html>.



Yu, Gyeong-In, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. “Orca: A Distributed Serving System for $\{$Transformer-Based$\}$ Generative Models.” In *16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)*, 521–38.



Zellers, Rowan, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. “HellaSwag: Can a Machine Really Finish Your Sentence?” In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 4791–4800.



Zhang, Yuanhan, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. 2024. “LLaVA-NeXT: A Strong Zero-Shot Video Understanding Model.” <https://llava-vl.github.io/blog/2024-04-30-llava-next-video/>.
