# Orca: A Distributed Serving System for Transformer-Based Generative Models 图表详解

### Figure 1: Illustrations for GPT’s inference procedure, Transformer layer, and internal state usage.

![29af86d2cac3229be2028a4c5e39953ebc1273f54f3a92b385f30916c5abab2c.jpg](29af86d2cac3229be2028a4c5e39953ebc1273f54f3a92b385f30916c5abab2c.jpg)

- **图像整体含义**：该图是 ORCA 论文中的 Figure 1，用于解释 **GPT 类 Transformer-based generative model 的推理过程、单个 Transformer layer 的内部结构，以及 Transformer 与 LSTM 在 internal state 使用方式上的差异**。整张图由三个子图组成：
  - **(a)** GPT 自回归生成的计算图。
  - **(b)** GPT 中使用的 Transformer layer 结构。
  - **(c)** Transformer 与 LSTM 的 internal state usage 对比。

| 子图 | 核心内容 | 主要说明 |
|---|---|---|
| **(a)** | GPT inference procedure | 展示 GPT 如何通过多次 iteration 自回归生成 token |
| **(b)** | Transformer layer | 展示 GPT decoder-only Transformer layer 的主要算子 |
| **(c)** | Internal state usage | 对比 Transformer Attention K/V cache 与 LSTM hidden state 的状态传递方式 |

- **(a) GPT 推理过程计算图分析**
  - 左侧子图展示了一个简化的 **3-layer GPT model** 的推理过程。
  - 图中的圆角矩形或圆形节点表示 **Transformer layers**，节点内的数字表示执行顺序。
  - 相同灰度的节点表示使用同一组模型参数，即同一个 Transformer layer 在不同 iteration 中被重复调用。
  - 输入文本为 **“I think this”**，模型逐步生成：
    - 第 1 次 iteration 生成 **“is”**
    - 第 2 次 iteration 生成 **“great”**
    - 第 3 次 iteration 生成 **“<EOS>”**
  - **<EOS>** 表示 end-of-sequence token，生成该 token 后推理终止。

| Iteration | 输入 | 执行内容 | 输出 token | 阶段 |
|---|---|---|---|---|
| **iter 1** | “I think this” | 一次性处理全部 input tokens | “is” | **Initiation phase** |
| **iter 2** | “is” | 使用上一步生成 token 继续推理 | “great” | **Increment phase** |
| **iter 3** | “great” | 继续生成下一个 token | “<EOS>” | **Increment phase / termination** |

- **(a) 中体现的关键机制**
  - GPT 是 **autoregressive language model**，每次只生成一个 next token。
  - 当前生成的 token 会被反馈给模型，作为下一次 iteration 的输入。
  - 因此，一个请求不是单次模型执行完成，而是需要执行 **multiple iterations**。
  - 第一次 iteration 通常处理完整 prompt，即 **initiation phase**。
  - 后续每次 iteration 只处理上一步生成的一个 token，即 **increment phase**。
  - 这正是 ORCA 论文提出 **iteration-level scheduling** 的背景：传统 request-level scheduling 无法高效处理这种多轮生成过程。

- **(b) GPT 中 Transformer layer 结构分析**
  - 中间子图展示了 GPT 使用的 **decoder-only Transformer layer**。
  - 该结构主要由以下部分组成：
    - **LayerNorm**
    - **QKV Linear**
    - **Attention**
    - **Attn Out Linear**
    - **Add residual connection**
    - 第二个 **LayerNorm**
    - **MLP**
    - 第二个 **Add residual connection**
  - 输入首先经过 **LayerNorm**，再进入 **QKV Linear**。
  - **QKV Linear** 会生成三个张量：
    - **Query**
    - **Key**
    - **Value**
  - Query、Key、Value 输入到 **Attention** 中完成注意力计算。
  - Attention 输出再经过 **Attn Out Linear**。
  - 随后通过 **Add** 与残差连接相加。
  - 之后进入第二个 **LayerNorm** 和 **MLP**。
  - MLP 内部包括：
    - **Linear**
    - **GeLU**
    - **Linear**
  - 最后通过另一个 **Add** 输出该 Transformer layer 的结果。

| 模块 | 作用 | 是否含模型参数 | 与 batching 的关系 |
|---|---|---|---|
| **LayerNorm** | 归一化 hidden state | 部分参数 | 可对 token 展平后批处理 |
| **QKV Linear** | 生成 Query/Key/Value | **有大量参数** | 非常适合 batching |
| **Attention** | 根据 Query-Key 权重聚合 Value | **无模型参数** | 受 sequence length / K/V state 影响，难以普通 batching |
| **Attn Out Linear** | Attention 输出线性变换 | **有参数** | 适合 batching |
| **Add** | Residual connection | 无参数 | 易批处理 |
| **MLP** | 非线性前馈网络 | **有大量参数** | 非常适合 batching |
| **GeLU** | 激活函数 | 无参数 | 易批处理 |

- **(b) 与 ORCA selective batching 的关系**
  - 图中最关键的是 **Attention** 与其他算子的差异。
  - **Linear、LayerNorm、Add、GeLU、MLP** 等操作通常可以把多个 token 展平成大矩阵后一起处理。
  - 但 **Attention** 需要区分不同 request 的上下文，且每个 request 的历史 token 数可能不同，因此其输入 shape 不规则。
  - ORCA 的 **selective batching** 正是基于这一观察：
    - 对 **non-Attention operations** 执行 batching。
    - 对 **Attention operation** 按 request 分开执行。
  - 由于 Attention 本身不包含大规模模型参数，不 batch Attention 对整体效率影响较小。
  - 反而对 Linear/MLP 这类参数密集型操作 batching，能显著减少 GPU memory read，提高吞吐。

- **(c) Transformer internal state usage 分析**
  - 右侧子图对比了 **Transformer layer** 和 **LSTM layer** 在连续 token 处理过程中的状态传递方式。
  - 上半部分是 Transformer，下半部分是 LSTM。
  - 图中符号含义：
    - **h**：layer input/output hidden state
    - **k**：Attention key
    - **v**：Attention value
    - **c**：LSTM internal memory
    - **l**：layer index
    - **t**：token index

| 符号 | 含义 |
|---|---|
| **h_l,t** | 第 l 层在 token t 位置的 hidden state |
| **k_l,t** | 第 l 层 token t 的 Attention key |
| **v_l,t** | 第 l 层 token t 的 Attention value |
| **c_l,t** | LSTM 第 l 层 token t 的 internal memory |
| **t** | token index |
| **l** | layer index |

- **Transformer 状态传递方式**
  - 在处理第 t 个 token 时，Transformer layer 不仅需要当前 token 的 hidden state，还需要之前所有 token 的 **Attention keys 和 values**。
  - 图中标注为：
    - 输入历史状态：**k_l,1:t-1、v_l,1:t-1**
    - 输出更新状态：**k_l,1:t、v_l,1:t**
  - 也就是说，每处理一个新 token，Transformer 都会把当前 token 的 key/value 追加到已有 K/V cache 中。
  - 因此 Transformer 的 internal state 随 token index t 增长而增长。
  - 这也是 GPT 推理需要维护 **K/V cache** 的原因。

- **LSTM 状态传递方式**
  - LSTM 每个时间步只传递固定大小的状态：
    - **c_l,t**
    - **h_l,t**
  - 处理下一个 token 时，LSTM 只需要前一个时间步的 **c_l,t-1 和 h_l,t-1**。
  - 状态大小不会随着序列长度增长。
  - 因此 LSTM 的 internal state 是 **constant-size state**。

| 模型 | Internal state | 状态大小是否随 token 数增长 | 对 serving 的影响 |
|---|---|---|---|
| **Transformer / GPT** | **Attention K/V cache** | **会增长** | 需要复杂的 K/V memory management |
| **LSTM** | **hidden state h + memory c** | 不增长 | 状态管理较简单 |

- **(c) 揭示的 Transformer serving 难点**
  - 对于 Transformer，每个 request 的 K/V cache 长度取决于它已经处理了多少 token。
  - 不同 request 可能处于不同 token index。
  - 因此不同 request 的 Attention 输入 shape 可能不同。
  - 这会导致传统 batching 很难直接应用。
  - 例如：
    - request A 已生成 10 个 token，需要访问 **k/v 1:10**。
    - request B 已生成 100 个 token，需要访问 **k/v 1:100**。
    - 两者 Attention 的 K/V tensor shape 不一致，无法简单拼成统一 batch tensor。
  - ORCA 的 **selective batching** 就是为了解决这一问题。

- **图像对 ORCA 论文核心问题的支撑**
  - 该图从模型结构层面解释了为什么 GPT serving 不同于传统 DNN serving。
  - 传统模型如 ResNet、BERT 通常一个 request 只需一次 forward。
  - GPT 类生成模型需要多次 forward，每次生成一个 token。
  - 每个 request 的生成长度可能不同，因此会出现：
    - **early-finished requests**
    - **late-joining requests**
    - **variable-length K/V cache**
    - **irregular Attention tensor shapes**
  - 这些特征共同导致传统 request-level batching 效率低。

- **图像与 ORCA 两个核心技术的对应关系**
  - **Iteration-level scheduling**
    - 来源于 (a) 中 GPT 多 iteration 的生成过程。
    - ORCA 不再等整个 request 完成，而是在每个 iteration 后重新调度 batch。
  - **Selective batching**
    - 来源于 (b) 和 (c) 中 Attention 与其他算子的差异。
    - ORCA 对非 Attention 算子 batch，对 Attention 单独处理。
  - **K/V cache management**
    - 来源于 (c) 中 Transformer 的状态累积特性。
    - ORCA 需要为每个 request 管理可增长的 Attention key/value state。

| 图中现象 | 系统挑战 | ORCA 解决方案 |
|---|---|---|
| GPT 每次只生成一个 token | request 需要多轮执行 | **Iteration-level scheduling** |
| 不同请求生成长度不同 | 早完成请求被阻塞 | 每轮 iteration 后检查完成状态 |
| 新请求到达时当前 batch 未完成 | late-joining request 等待过长 | 新请求可在下一轮 iteration 加入 |
| Attention K/V cache 长度不同 | 难以普通 batching | **Selective batching** |
| Linear/MLP 参数量大 | 需要提高 GPU 利用率 | 对 non-Attention ops 执行 batching |
| K/V cache 随 token 增长 | GPU memory 管理复杂 | Attention K/V manager |

- **关键结论**
  - 该图清晰说明了 GPT 推理的本质：**多轮 autoregressive iteration + 可增长 Attention K/V state**。
  - Transformer 的 Attention 机制使得每个 request 的状态 shape 与已处理 token 数强相关。
  - 因此，传统 serving 系统中的 **request-level scheduling + 全算子 batching** 并不适合 GPT 类生成模型。
  - ORCA 的设计正是围绕该图揭示的两个事实展开：
    - **调度粒度应从 request 降到 iteration。**
    - **batching 应从全算子 batching 改为 selective batching。**

### Figure 2: Overall workflow of serving a generative language model with existing systems.

![d6fabf0757f910f5740f07d6a24617d223c4ee028a6fe3f79eaa579d2a4fbd50.jpg](d6fabf0757f910f5740f07d6a24617d223c4ee028a6fe3f79eaa579d2a4fbd50.jpg)

- **图像主题**：该图展示了使用现有推理服务系统 serving Transformer-based generative language model 的整体工作流，典型组合是 **Triton Inference Server + FasterTransformer**。
- **核心含义**：现有系统采用 **request-level scheduling**，即 Serving System 将多个请求组成一个 batch 后交给 Execution Engine，Execution Engine 负责完整生成所有请求的输出后才统一返回结果。
- **图中关键组件如下：**

| 组件 | 图中位置 | 作用 |
|---|---:|---|
| **Endpoint** | 左侧 Serving System 内部 | 接收客户端 **request**，返回 **response** |
| **Request Queue** | Serving System 底部 | 缓存尚未被调度的请求 |
| **Scheduler** | Serving System 中部 | 从队列中取请求、组成 batch、调用 Execution Engine |
| **Execution Engine** | 右侧独立模块 | 执行模型推理，例如 **FasterTransformer** |
| **Serving System** | 左侧大框 | 负责请求管理、排队、调度和响应 |
| **Client request / response** | 最左侧箭头 | 用户请求进入系统，最终响应返回用户 |

- **图中编号流程解析：**

| 编号 | 动作 | 说明 |
|---:|---|---|
| **①** | Scheduler 从 **Request Queue** 中取请求 | Scheduler 查询队列，将多个请求组织成 batch |
| **②** | Scheduler 将 batch 发送给 **Execution Engine** | 示例 batch 包含两个请求：**x₁: “I think”** 和 **x₂: “I love”** |
| **③** | Execution Engine 执行生成式推理 | Engine 对整个 batch 执行多轮 autoregressive decoding |
| **④** | Execution Engine 返回完整结果 | 返回 **x₁: “this is great”**，**x₂: “you”**，然后 Serving System 再响应客户端 |

- **请求示例含义：**
  - **x₁: “I think”** 是一个输入 prompt。
  - **x₂: “I love”** 是另一个输入 prompt。
  - Execution Engine 对两个请求进行 batched inference。
  - 最终生成：
    - **x₁ → “this is great”**
    - **x₂ → “you”**

- **该图体现的传统 serving 模式：**
  - Serving System 并不直接执行 Transformer 计算。
  - Serving System 只负责 **batching、scheduling、request/response handling**。
  - Execution Engine 负责实际的 GPU 计算和 autoregressive generation。
  - Scheduler 与 Execution Engine 的交互粒度是 **request-level**，不是 **iteration-level**。

- **该工作流的关键问题：**
  - **batch 一旦提交给 Execution Engine，就固定不变**。
  - 新请求到达后只能留在 **Request Queue** 中等待。
  - 已经提前生成完成的请求不能立即返回。
  - Execution Engine 只有在整个 batch 全部完成后才把结果返回给 Scheduler。
  - 因此会产生 **queueing delay** 和 **head-of-line blocking**。

- **为什么该模式不适合 generative language model：**
  - Transformer-based generative model 使用 **autoregressive decoding**。
  - 一个请求通常需要执行多次模型 iteration，每次生成一个 token。
  - 不同请求的生成长度不同，例如图中：
    - **x₁** 生成 “this is great”，需要更多 token。
    - **x₂** 只生成 “you”，更早完成。
  - 但在现有系统中，**x₂ 即使早完成，也必须等待 x₁ 完成**。

- **图中隐含的性能瓶颈：**

| 瓶颈 | 原因 | 后果 |
|---|---|---|
| **早完成请求无法提前返回** | Engine 统一返回整个 batch 结果 | 增加 tail latency 和 average latency |
| **新请求无法中途加入 batch** | batch 在 request-level 固定 | 增加 queueing time |
| **无效计算** | 已完成请求仍可能占据 batch 位置 | GPU 利用率下降 |
| **调度粒度过粗** | Scheduler 只在 batch 开始/结束时控制 Engine | 无法动态调整 batch |
| **对生成长度敏感** | 不同请求 output length 不同 | batch 内部负载不均衡 |

- **该图与 ORCA 的对比意义：**
  - Figure 2 展示的是 **existing systems** 的传统架构。
  - ORCA 后续提出的核心改进是：
    - **iteration-level scheduling**
    - **selective batching**
  - 与图中模式不同，ORCA 让 Scheduler 在每个 generation iteration 后重新决定 batch。
  - 因此 ORCA 可以：
    - 让完成请求立即返回。
    - 让新请求尽快加入执行。
    - 减少无效计算。
    - 提升吞吐并降低延迟。

- **从系统接口角度看，该图暴露的问题是：**
  - Serving System 与 Execution Engine 的接口过于粗粒度。
  - Scheduler 调用 Engine 时，是让 Engine 完成整个 request 的完整生成过程。
  - Engine 内部进行了多轮 decoding，但这些 iteration 对 Scheduler 不可见。
  - Scheduler 无法在 token 级别或 iteration 级别介入。

- **简要总结：**
  - 该图说明了传统 generative model serving 的基本流程：**request → queue → scheduler → execution engine → full generated response**。
  - 其核心限制是 **request-level scheduling**。
  - 对于 autoregressive Transformer generation，这种设计会导致 **早完成请求等待、晚到请求排队、batch 固定、GPU 资源浪费**。
  - ORCA 正是针对这一问题提出 **iteration-level scheduling**，将调度粒度从完整请求降低到每次 token generation iteration。

### Figure 3: An illustration for a case where the requests have the same input length but some requests finish earlier than others. Shaded tokens represent input tokens. “-” denotes inputs and outputs of extra computation imposed by the scheduling.

![b3265c6d41183072bb80fc0b68bca2d365929f9140ea282d5e57c2e0b51679fc.jpg](b3265c6d41183072bb80fc0b68bca2d365929f9140ea282d5e57c2e0b51679fc.jpg)

- **图片核心含义**
  - 该图展示了在现有 **request-level scheduling** 与固定 batch 执行模式下，两个生成式请求虽然具有相同输入长度，但由于生成结束时间不同，会导致 **early-finished request** 无法提前返回，并产生 **extra computation**。
  - 图中有两个请求：**x₁** 和 **x₂**。
  - 两个请求初始输入长度相同，都是 2 个 token：
    - **x₁**：`I think`
    - **x₂**：`I love`
  - 阴影 token 表示客户端输入的原始 token。
  - `-` 表示由于调度机制限制而产生的无效输入或无效输出，即 **额外计算**。

- **图中请求生成过程**

| Iteration | x₁ 当前输入 | x₁ 输出 | x₂ 当前输入 | x₂ 输出 | 说明 |
|---|---|---|---|---|---|
| **iter 1** | `I think` | `this` | `I love` | `you` | 两个请求一起进入 batch，正常执行 |
| **iter 2** | `this` | `is` | `you` | `<EOS>` | **x₂ 已经生成结束符 `<EOS>`，应当完成** |
| **iter 3** | `is` | `great` | `-` | `-` | x₂ 已完成，但仍被迫留在 batch 中，占用计算 |
| **iter 4** | `great` | `<EOS>` | `-` | `-` | x₁ 结束，整个 batch 才返回 |

- **x₁ 的完整生成结果**
  - 输入：**`I think`**
  - 生成：**`this is great <EOS>`**
  - 实际需要执行：**4 次 iteration**
  - 该请求是 batch 中较慢完成的请求。

- **x₂ 的完整生成结果**
  - 输入：**`I love`**
  - 生成：**`you <EOS>`**
  - 实际只需要执行：**2 次 iteration**
  - 但由于与 x₁ 固定在同一个 batch 中，x₂ 仍然被迫等待到 **iter 4** 才能返回。
  - 因此 x₂ 遭遇了明显的 **tail waiting latency**。

- **图中暴露的系统问题**

| 问题 | 图中表现 | 后果 |
|---|---|---|
| **Early-finished request cannot return early** | x₂ 在 iter 2 已生成 `<EOS>`，但不能立即返回 | 增加 x₂ 的端到端延迟 |
| **Fixed batch membership** | batch 在执行期间始终包含 x₁ 和 x₂ | 已完成请求无法移出 |
| **Extra computation** | iter 3 和 iter 4 中 x₂ 位置为 `-` | GPU 对无效请求继续执行，占用算力 |
| **Request-level scheduling 粒度过粗** | serving system 只能等整个 batch 完成后再交互 | 无法动态加入新请求或移除完成请求 |
| **Throughput degradation** | 无效 token 仍参与 batch 计算 | 有效计算比例下降 |

- **为什么会产生额外计算**
  - 在传统 serving 系统中，例如 **Triton Inference Server + FasterTransformer**，调度器通常以 **request** 为单位提交 batch。
  - 一旦 batch 被提交给 execution engine，batch 中的请求集合在整个生成过程中保持不变。
  - 对于自回归生成模型，每个请求需要多次执行模型，每次生成一个 token。
  - 如果 batch 中某些请求提前生成 `<EOS>`，这些请求理论上已经完成，但系统仍然必须等待 batch 中最慢的请求结束。
  - 因此，x₂ 在 **iter 3** 和 **iter 4** 中产生了 `-`，表示没有真实 token 需要处理，却仍然被调度机制强行占位。

- **延迟影响分析**

| 请求 | 实际完成 iteration | 实际可返回时间 | 传统 batch 返回时间 | 额外等待 |
|---|---:|---:|---:|---:|
| **x₁** | iter 4 | iter 4 | iter 4 | 0 |
| **x₂** | iter 2 | iter 2 | iter 4 | 2 个 iteration |

- **计算浪费分析**

| Iteration | 有效请求数 | 无效请求数 | 说明 |
|---|---:|---:|---|
| **iter 1** | 2 | 0 | x₁、x₂ 都有效 |
| **iter 2** | 2 | 0 | x₂ 生成 `<EOS>`，但本轮仍有效 |
| **iter 3** | 1 | 1 | x₂ 已完成，产生无效计算 |
| **iter 4** | 1 | 1 | x₂ 继续占位，产生无效计算 |

- **关键矛盾**
  - **batching** 可以提升 GPU 利用率，但固定 batch 会导致早结束请求无法退出。
  - **autoregressive generation** 中不同请求的输出长度天然不同。
  - 因此，传统 **request-level batching** 与生成式模型的 **multi-iteration characteristic** 存在结构性冲突。
  - 图中 x₂ 的 `-` 正是这种冲突的直观体现。

- **与 ORCA 的关系**
  - 该图是 ORCA 提出 **iteration-level scheduling** 的直接动机之一。
  - ORCA 的思路是：
    - 每次只调度模型执行 **one iteration**。
    - 每轮结束后检查哪些请求已经完成。
    - 已完成请求立即返回客户端。
    - 新到达请求可以在下一轮 iteration 中加入 batch。
  - 如果使用 ORCA，x₂ 在 **iter 2** 生成 `<EOS>` 后即可返回，不需要等待 x₁ 完成 iter 3 和 iter 4。
  - 同时，iter 3 和 iter 4 的 batch slot 可以被其他新请求填补，避免 `-` 对应的无效计算。

- **图示总结**
  - 该图用一个极简例子说明：即使请求输入长度相同，只要输出长度不同，传统固定 batch 执行仍会造成 **latency inflation** 和 **GPU computation waste**。
  - 核心问题不是 Transformer 本身的计算错误，而是 serving 系统的调度粒度过粗。
  - ORCA 通过 **iteration-level scheduling** 解决早完成请求无法提前返回的问题，并通过 **selective batching** 支持更灵活的 batch 组合。

### Figure 5: An illustration of ORCA execution engine running a Transformer layer on a batch of requests with selective batching. We only depict the QKV Linear, Attention, and Attention Out Linear operations for simplicity.

![ad0b0963b5ec1aad7ea60ce20ebc70b0a2d0741ed61c64f672c493bec1ab8f6f.jpg](ad0b0963b5ec1aad7ea60ce20ebc70b0a2d0741ed61c64f672c493bec1ab8f6f.jpg)

- **图像核心含义**
  - 该图展示了 **ORCA execution engine** 在一个 **Transformer layer** 内如何对一批异构请求执行 **selective batching**。
  - 图中只画出了简化路径：**Layer Input → QKV Linear → Split → Attention → Merge → Attn Out Linear**。
  - 重点在于：**非 Attention 操作采用批处理，Attention 操作按请求拆分单独执行**。
  - 这样可以让不同阶段、不同 token 长度、不同历史上下文长度的请求仍然能被放入同一个 batch 中执行。

- **输入 batch 的组成**

  | 请求 | 当前输入 token | 当前阶段 | 当前输入长度 | 是否已有历史 K/V |
  |---|---:|---|---:|---|
  | **x₁** | x₁₄ | increment phase | 1 | 有，历史为 x₁₁, x₁₂, x₁₃ |
  | **x₂** | x₂₂ | increment phase | 1 | 有，历史为 x₂₁ |
  | **x₃** | x₃₁, x₃₂ | initiation phase | 2 | 无或刚开始建立 |
  | **x₄** | x₄₁, x₄₂, x₄₃ | initiation phase | 3 | 无或刚开始建立 |

  - 该 batch 一共包含 **7 个当前待处理 token**：
    - x₁：1 个
    - x₂：1 个
    - x₃：2 个
    - x₄：3 个
  - 因此图中 **Layer Input** 的总形状为 **[7, H]**。
  - 这里的 **H** 表示 Transformer hidden size。

- **为什么这是一个“异构 batch”**
  - 该 batch 中请求并不规整：
    - **x₁ 和 x₂** 处于 **increment phase**，每个请求只处理 1 个新 token。
    - **x₃ 和 x₄** 处于 **initiation phase**，分别处理 2 个和 3 个输入 token。
    - 不同请求的历史上下文长度不同：
      - x₁ 已有 3 个历史 token 的 K/V。
      - x₂ 只有 1 个历史 token 的 K/V。
  - 在传统 batching 中，这种 batch 很难直接合并，因为 Attention 需要保留请求边界，并且不同请求的 sequence length 不同。
  - ORCA 的关键做法是：**不强行对 Attention 做统一 batch，而是只对适合 batch 的操作进行 batch**。

- **QKV Linear 阶段：采用 token-wise batching**

  | 操作 | 输入形状 | 输出形状 | 是否 batch | 原因 |
  |---|---:|---:|---|---|
  | **QKV Linear** | [7, H] | [7, 3H] | 是 | Linear 操作对每个 token 独立变换，不需要知道 token 属于哪个请求 |

  - **QKV Linear** 将每个 token 的 hidden vector 映射为 **Query、Key、Value**。
  - 每个 token 的输出维度从 **H** 变为 **3H**。
  - 由于这个操作本质上是矩阵乘法，所有 token 可以展平成一个大矩阵一起处理。
  - 因此即使请求长度不同，也可以形成统一输入 **[7, H]** 并高效执行。

- **Split 阶段：恢复请求边界**

  | 请求 | Split 后的 QKV tensor 形状 |
  |---|---:|
  | **x₃** | [2, 3H] |
  | **x₄** | [3, 3H] |
  | **x₁** | [1, 3H] |
  | **x₂** | [1, 3H] |

  - **Split** 的作用是把 QKV Linear 产生的统一 tensor **[7, 3H]** 按请求切开。
  - 切分后，每个请求拥有自己的 QKV tensor。
  - 这是执行 Attention 前必须做的步骤，因为 **Attention 不能跨请求计算**。
  - 例如：
    - x₃ 的 token 只能 attend 到 x₃ 内部 token。
    - x₄ 的 token 只能 attend 到 x₄ 内部 token。
    - x₁ 当前 token 需要 attend 到 x₁ 的历史 token，而不是 x₂、x₃ 或 x₄。

- **Attention K/V Manager 的作用**

  | 请求 | 已缓存 Key/Value | 当前 Attention 需要使用 |
  |---|---|---|
  | **x₁** | x₁₁, x₁₂, x₁₃ | 历史 K/V + 当前 token x₁₄ 的 K/V |
  | **x₂** | x₂₁ | 历史 K/V + 当前 token x₂₂ 的 K/V |
  | **x₃** | 无历史或正在初始化 | 当前输入 x₃₁, x₃₂ 的 K/V |
  | **x₄** | 无历史或正在初始化 | 当前输入 x₄₁, x₄₂, x₄₃ 的 K/V |

  - **Attention K/V Manager** 维护每个请求的历史 **Key/Value cache**。
  - 对于 **increment phase** 请求：
    - 当前只输入一个新 token。
    - Attention 仍然需要访问该请求过去所有 token 的 K/V。
    - 因此 x₁ 的 Attention 要使用 x₁₁、x₁₂、x₁₃ 的历史 K/V。
    - x₂ 的 Attention 要使用 x₂₁ 的历史 K/V。
  - 对于 **initiation phase** 请求：
    - x₃ 和 x₄ 正在处理初始输入序列。
    - 它们的 K/V 会在本轮计算中生成并存入 manager，供后续 increment phase 使用。

- **Attention 阶段：不做整体 batch，而是 per-request 执行**

  | Attention 单元 | 输入形状 | 输出形状 | 说明 |
  |---|---:|---:|---|
  | **Attn x₃** | [2, 3H] | [2, H] | 处理 x₃ 的 2 个输入 token |
  | **Attn x₄** | [3, 3H] | [3, H] | 处理 x₄ 的 3 个输入 token |
  | **Attn x₁** | [1, 3H] + 历史 K/V | [1, H] | 处理 x₁ 当前 token，并读取历史 K/V |
  | **Attn x₂** | [1, 3H] + 历史 K/V | [1, H] | 处理 x₂ 当前 token，并读取历史 K/V |

  - 图中每个请求都有单独的 **Attn xᵢ** 模块。
  - 这是 **selective batching** 的核心：
    - **Attention 不进行统一 batch**。
    - 每个请求单独执行 Attention。
  - 原因是 Attention 依赖请求内部的序列结构：
    - 不同请求的 sequence length 不同。
    - 不同请求的历史 K/V 长度不同。
    - Attention 必须避免跨请求 token 混合。
  - 因此传统形如 **[B, L, H]** 的统一 Attention batch 不适用。

- **为什么单独执行 Attention 的代价可接受**
  - Attention 操作本身 **不包含大规模模型参数**。
  - Transformer 中主要参数集中在：
    - **QKV Linear**
    - **Attn Out Linear**
    - **MLP**
  - 对这些参数密集型操作做 batching，可以复用从 GPU memory 读取的模型参数。
  - Attention 不含模型权重，因此不 batch Attention 带来的参数复用损失较小。
  - 论文的关键判断是：**牺牲 Attention batching，换取任意请求组合 batching，是值得的**。

- **Merge 阶段：重新合并 Attention 输出**

  | 来源 | Attention 输出形状 |
  |---|---:|
  | **Attn x₃** | [2, H] |
  | **Attn x₄** | [3, H] |
  | **Attn x₁** | [1, H] |
  | **Attn x₂** | [1, H] |
  | **Merge 后总输出** | [7, H] |

  - 各个请求的 Attention 输出会通过 **Merge** 合并回一个统一 tensor。
  - 合并后的形状重新变为 **[7, H]**。
  - 这使得后续非 Attention 操作可以继续使用高效 batch 执行。

- **Attn Out Linear 阶段：恢复 batch 执行**

  | 操作 | 输入形状 | 输出形状 | 是否 batch |
  |---|---:|---:|---|
  | **Attn Out Linear** | [7, H] | [7, H] | 是 |

  - **Attn Out Linear** 是 Attention 后的线性投影。
  - 它是参数密集型操作，适合 batch。
  - ORCA 在 Merge 后重新把所有 token 合并，继续以 **[7, H]** 形式执行。
  - 图中的省略号表示后续还会继续执行 Transformer layer 中的其他操作，如：
    - **Add**
    - **LayerNorm**
    - **MLP**
    - **GeLU**
    - residual connection 等。

- **图中张量形状流动总结**

  | 阶段 | 张量形状 | 处理方式 |
  |---|---:|---|
  | **Layer Input** | [7, H] | 所有请求 token 展平合并 |
  | **QKV Linear** | [7, H] → [7, 3H] | batched execution |
  | **Split** | [7, 3H] → [2,3H], [3,3H], [1,3H], [1,3H] | 按请求拆分 |
  | **Attention** | 每个请求单独处理 | per-request execution |
  | **Merge** | [2,H], [3,H], [1,H], [1,H] → [7,H] | 合并回统一 tensor |
  | **Attn Out Linear** | [7, H] → [7, H] | batched execution |

- **该设计解决的问题**
  - 传统 serving 系统的问题：
    - batch 内请求必须形状一致。
    - 新请求不能中途加入。
    - 已完成请求不能及时离开。
    - 不同生成长度导致无效计算。
  - ORCA 通过该图所示机制实现：
    - **任意请求可以组成 batch**。
    - **initiation phase 和 increment phase 请求可以混合执行**。
    - **不同输入长度请求可以混合执行**。
    - **不同历史上下文长度请求可以混合执行**。
    - **scheduler 可以每个 iteration 动态调整 batch**。

- **与 iteration-level scheduling 的关系**
  - 该图是 **iteration-level scheduling** 能够高效落地的关键。
  - Scheduler 每次只调度一个 iteration。
  - 每次 iteration 中，batch 可能包含：
    - 新来的请求。
    - 已经生成若干 token 的请求。
    - 即将完成的请求。
  - 如果没有 **selective batching**，这种动态 batch 很难高效执行。
  - 因此：
    - **iteration-level scheduling** 解决调度粒度问题。
    - **selective batching** 解决执行引擎中的异构 batch 问题。

- **图中的关键创新点**
  - **按 token 展平非 Attention 操作**
    - 不再要求所有请求有相同 sequence length。
    - 使用 **[ΣL, H]** 形式代替传统 **[B, L, H]**。
  - **Attention 保留请求级语义**
    - 通过 Split 和 per-request Attention 避免跨请求错误计算。
  - **K/V cache 独立管理**
    - 每个请求拥有独立的历史 Key/Value。
    - 支持增量解码和动态 batch。
  - **Merge 后继续批处理**
    - 最大化参数密集型操作的 GPU 利用率。
    - 减少模型参数重复读取。

- **一句话总结**
  - 该图说明 ORCA 的 **selective batching** 如何在 Transformer layer 内部实现：**Linear、MLP 等参数密集型操作统一 batch，Attention 按请求拆分执行，再合并回 batch，从而支持异构请求的高效 iteration-level serving**。

### Figure 7: An illustration of the distributed architecture of ORCA’s execution engine using the parallelization configuration shown in Figure 6. For example, the first inter-layer partition (Layer1 and Layer2) in Figure 6 is assigned to Worker1, while the second partition is assigned to Worker2.

![e26cdf4e4361d4662eebb26fffdbb4b80c7a96cb3d8be2a0302a733054b7f7b4.jpg](e26cdf4e4361d4662eebb26fffdbb4b80c7a96cb3d8be2a0302a733054b7f7b4.jpg)

- **图像核心含义**
  - 该图展示了 **ORCA execution engine 的分布式架构**，重点说明在大模型推理中，ORCA 如何结合 **inter-layer parallelism** 与 **intra-layer parallelism**，并将控制信息与张量数据通信分离。
  - 图中采用 Figure 6 的并行配置思想：
    - **Worker 1** 负责前一组 Transformer layers，例如 Layer 1 和 Layer 2。
    - **Worker 2** 负责后一组 Transformer layers，例如 Layer 3 和 Layer 4。
    - 每个 Worker 内部有多个 **GPU**，对应 **intra-layer partitions**。
  - 整体结构体现了 ORCA 的关键设计：**Scheduler 只与 Engine Master 交互，Engine Master 再协调多个 Worker 进行流水线式分布式推理**。

- **图中主要组件**

| 组件 | 图中位置 | 作用 |
|---|---|---|
| **Scheduler** | 最左侧 | 按 iteration-level scheduling 选择请求 batch，并向执行引擎发起调度 |
| **Execution Engine** | 大框整体 | ORCA 的分布式模型执行层 |
| **Engine Master** | Execution Engine 左侧 | 接收 Scheduler 的调度请求，并向 Worker 转发 tokens 与 control message |
| **Worker 1** | 中间大框 | 负责第一个 inter-layer partition，例如模型前半部分 layers |
| **Worker 2** | 右侧大框 | 负责第二个 inter-layer partition，例如模型后半部分 layers |
| **Controller** | 每个 Worker 上方 | Worker 的 CPU 控制逻辑，解析控制消息并驱动 GPU kernel |
| **GPU** | 每个 Worker 内部多个圆角框 | 执行 intra-layer partition 的张量计算 |
| **Control Plane** | Worker 上半部分 | 传递 control message 与 tokens，不经过 GPU |
| **Data Plane** | Worker 下半部分 | 通过 GPU/NCCL 传递中间 tensor 数据 |

- **整体执行流程**

| 阶段 | 数据/控制流 | 说明 |
|---|---|---|
| 1 | **Scheduler → Engine Master** | Scheduler 将一个 iteration 的 batch 调度给 Engine Master |
| 2 | **Engine Master → Worker 1 Controller** | Engine Master 向第一个 Worker 发送 **tokens** 与 **control message** |
| 3 | **Worker 1 Controller → Worker 1 GPUs** | Controller 解析请求信息，向本地多个 GPU 发起 kernel |
| 4 | **Worker 1 GPUs → Worker 2 GPUs** | Worker 1 完成其 inter-layer partition 后，将中间 tensor 通过 Data Plane 发给 Worker 2 |
| 5 | **Worker 1 Controller → Worker 2 Controller** | 控制消息沿 Control Plane 继续转发给下一个 Worker |
| 6 | **Worker 2 Controller → Worker 2 GPUs** | Worker 2 根据控制消息继续执行后续 layers |
| 7 | **Worker 2 → Engine Master → Scheduler** | 最后一个 Worker 生成 output tokens，经 Engine Master 返回给 Scheduler |

- **图中箭头含义**

| 箭头类型 | 表示内容 | 技术含义 |
|---|---|---|
| **实线箭头 schedule** | Scheduler 到 Engine Master | 表示调度一个 iteration 的 batch |
| **实线箭头 tokens** | Engine Master 与 Worker/Controller 之间 | 表示输入 tokens 或输出 tokens 的传递 |
| **实线箭头 control message** | Controller 之间 | 表示请求 ID、token index、input length 等控制信息 |
| **虚线箭头** | GPU 之间 | 表示中间 tensor 数据传输，通常使用 **NCCL** |
| **回传箭头 tokens** | Engine Master 到 Scheduler | 表示该 iteration 生成的新 token 返回给 Scheduler |

- **Control Plane 与 Data Plane 的分离**

| 平面 | 承载内容 | 通信对象 | 典型技术 | 设计目的 |
|---|---|---|---|---|
| **Control Plane** | control message、tokens | CPU Controller、Engine Master | **gRPC** 等 CPU 通信机制 | 避免 CPU 控制信息走 GPU/NCCL，减少同步开销 |
| **Data Plane** | intermediate tensors | GPU 与 GPU | **NCCL** | 高效传输大规模张量数据 |

- **Control Plane 的具体作用**
  - **Control Plane** 传递的是 CPU 需要解析的信息，例如：
    - **request id**
    - **current token index**
    - **input token length**
    - 当前请求处于 **initiation phase** 还是 **increment phase**
    - batch 中包含哪些请求
  - 这些信息用于指导 GPU kernel 的发起，例如：
    - Attention kernel 需要根据 **request id** 找到对应请求的 **Attention K/V cache**。
    - 根据 token index 决定当前 Attention 需要读取多少历史 keys/values。
  - ORCA 将这些控制信息通过 CPU 通信路径传递，而不是通过 NCCL，从而避免频繁的 **CPU-GPU synchronization**。

- **Data Plane 的具体作用**
  - **Data Plane** 专门处理 GPU 之间的大规模 tensor 数据传输。
  - 图中 Worker 内部多个 GPU 之间的虚线表示 **intra-layer parallelism** 所需的 GPU 通信。
  - Worker 1 到 Worker 2 的跨 Worker 虚线表示 **inter-layer partition** 之间传递 hidden states。
  - 这类数据量大、由 GPU 生产并由 GPU 消费，因此使用 **NCCL** 更合适。

- **Worker 内部结构分析**
  - 每个 Worker 包含一个 **Controller** 和多个 **GPU**。
  - **Controller** 负责 CPU 端调度：
    - 接收 Engine Master 或前一个 Worker 的控制消息。
    - 将控制消息分发给本 Worker 内的 GPU 控制线程。
    - 发起对应 CUDA kernels。
    - 将 control message 转发给下一个 Worker。
  - 多个 **GPU** 对应同一个 inter-layer partition 内的 **intra-layer partitions**。
  - 这意味着每个 Transformer layer 的矩阵乘法、Attention 等操作可以被切分到多个 GPU 上并行执行。

- **Worker 之间关系**
  - **Worker 1** 与 **Worker 2** 不是复制关系，而是负责模型不同层段。
  - Worker 1 执行模型前半部分 layers，Worker 2 执行模型后半部分 layers。
  - 因此请求在 Worker 间流动类似流水线：
    - 输入 tokens 先进入 Worker 1。
    - Worker 1 产生中间 hidden states。
    - hidden states 被发送到 Worker 2。
    - Worker 2 继续计算并最终产生 output token。
  - 这种设计对应 **inter-layer parallelism**，也就是 pipeline/model parallelism 的一种形式。

- **Engine Master 的作用**
  - **Engine Master** 是 Scheduler 与分布式 Worker 集群之间的桥接层。
  - 它不会直接执行模型计算，而是负责：
    - 接收 Scheduler 发来的 batch 调度。
    - 将 tokens 和 control message 发送给第一个 Worker。
    - 从最后一个 Worker 收集生成的 output tokens。
    - 将 output tokens 返回给 Scheduler。
  - 该设计使 Scheduler 不需要感知底层 Worker 拓扑和 GPU 分布细节。

- **Scheduler 与 Engine 的交互特点**
  - 图中 Scheduler 与 Engine Master 之间的交互发生在 **每个 iteration**。
  - 这与传统 serving system 不同：
    - 传统系统通常一次调度整个 request。
    - ORCA 一次只调度一个 **model iteration**。
  - 因此 ORCA 可以在每个 iteration 后：
    - 插入新到达请求。
    - 移除已经完成请求。
    - 动态调整 batch 组成。
  - 这正是 **iteration-level scheduling** 的系统基础。

- **与 FasterTransformer 等系统的关键差异**
  - 传统系统中，控制信息常通过 GPU 通信路径，例如 **NCCL**，间接传递。
  - 这会导致频繁 **CPU-GPU synchronization**：
    - CPU 需要知道控制信息才能发 kernel。
    - 但控制信息若经过 GPU/NCCL，则 CPU 需要等待 GPU 通信完成。
  - ORCA 的改进是：
    - **Control message 走 CPU Control Plane**。
    - **Tensor data 走 GPU Data Plane**。
  - 该分离减少了每个 iteration 的同步成本，尤其适合 autoregressive generation 中大量重复 iteration 的场景。

- **图中体现的并行策略**

| 并行方式 | 图中体现 | 作用 |
|---|---|---|
| **Inter-layer parallelism** | Worker 1 与 Worker 2 分别负责不同 layers | 支持超大模型跨机器/跨 GPU 部署 |
| **Intra-layer parallelism** | 每个 Worker 内部多个 GPU | 将单层内部矩阵计算切分到多个 GPU |
| **Pipeline parallelism** | batch 在 Worker 间顺序流动 | 提高多 Worker 利用率，减少空闲 |
| **Iteration-level scheduling** | Scheduler 每次 schedule 一个 iteration | 支持动态 batch、早完成返回、晚到请求加入 |
| **Control/Data plane separation** | 上方 Control Plane 与下方 Data Plane | 降低同步开销，提高分布式推理效率 |

- **为什么需要这种架构**
  - GPT-3 175B、341B 等模型无法放入单个 GPU。
  - 必须将模型参数和计算拆分到多个 GPU 甚至多台机器。
  - Autoregressive generation 每生成一个 token 都要执行一次完整模型 iteration。
  - 每个 iteration 都伴随：
    - batch 调度。
    - K/V cache 访问。
    - GPU kernel launch。
    - Worker 间通信。
  - 如果每次 iteration 都产生较高控制开销，整体吞吐会严重下降。
  - 因此 ORCA 采用该架构来同时解决：
    - **大模型容量问题**
    - **多 GPU 并行问题**
    - **iteration-level 调度问题**
    - **控制通信开销问题**

- **对 Attention K/V cache 的影响**
  - 虽然图中未显式画出 **Attention K/V manager**，但 control message 中的 request id 和 token index 会用于定位 K/V cache。
  - 每个 Worker/GPU 需要保存其负责 layers 的 K/V cache。
  - 当某个请求进入 increment phase 时：
    - 当前 token 的 query/key/value 在本 iteration 产生。
    - 历史 keys/values 从 K/V cache 中读取。
    - Attention kernel 根据 control message 找到对应缓存区域。
  - 因此 Control Plane 的低延迟传递对于 K/V cache 管理非常关键。

- **图像传达的系统设计重点**
  - **Scheduler 不直接操作 GPU**，而是通过 Engine Master 调用分布式执行引擎。
  - **Engine Master 不负责细粒度 CUDA 执行**，而是作为协调入口。
  - **Worker Controller 负责 CPU 端 kernel orchestration**。
  - **GPU 负责高吞吐 tensor computation**。
  - **Control Plane 与 Data Plane 分离** 是 ORCA 相比常规分布式推理系统的重要优化。
  - **多 Worker + 多 GPU** 的层级结构让 ORCA 能够扩展到百亿、千亿参数级 Transformer generative models。

- **总结**
  - 该图是 ORCA 分布式执行引擎的核心架构图。
  - 它展示了 ORCA 如何将一个大 Transformer 模型拆成多个 **inter-layer partitions**，由多个 Worker 顺序执行。
  - 每个 Worker 内部又通过多个 GPU 实现 **intra-layer parallelism**。
  - ORCA 的关键创新之一是将 **control message/tokens** 与 **intermediate tensors** 分别放到 **Control Plane** 和 **Data Plane** 中传输。
  - 这种设计减少了 CPU-GPU 同步开销，使 ORCA 更适合高频 iteration 的 autoregressive Transformer serving。

### 1bcb134e9996cb4c0568dcef5e3969cd1779242b2a656541ee509a8f9aa9f1cf.jpg

![1bcb134e9996cb4c0568dcef5e3969cd1779242b2a656541ee509a8f9aa9f1cf.jpg](1bcb134e9996cb4c0568dcef5e3969cd1779242b2a656541ee509a8f9aa9f1cf.jpg)

- **图片类型与来源**
  - 该图是 ORCA 论文 Figure 9(b) 的子图，展示 **101B GPT 模型在 8 GPUs 上的 engine microbenchmark 结果**。
  - 横轴是 **Batch Size**，取值为 **1、2、4、8、16、32**。
  - 纵轴是 **Execution Time (ms)**，表示处理一个 batch 的总执行时间。
  - 对比对象包括：
    - **ft(32)**：FasterTransformer，输入长度 32 tokens。
    - **ft(128)**：FasterTransformer，输入长度 128 tokens。
    - **orca(32)**：ORCA engine，输入长度 32 tokens。
    - **orca(128)**：ORCA engine，输入长度 128 tokens。
  - 所有请求生成长度相同，实验中每个请求生成 **32 output tokens**，因此该图主要比较 **execution engine 本身效率**，不体现 ORCA iteration-level scheduling 的收益。

- **图中核心观察**
  - **Batch Size 从 1 增加到 8 时，FasterTransformer 与 ORCA 的执行时间都缓慢上升**。
  - **ORCA 与 FasterTransformer 在 Batch Size ≤ 8 时性能接近**，说明 ORCA 的 selective batching 并没有显著损害 engine 层执行效率。
  - **FasterTransformer 在 Batch Size ≥ 16 时缺失数据**，表示发生 **Out of Memory (OOM)**。
  - **ORCA 可以继续扩展到 Batch Size 16 和 32**，说明 ORCA 在 Attention K/V cache 内存管理上更灵活、更节省。
  - 输入长度从 **32 增加到 128** 时，执行时间略有增加，但增加幅度不大，说明在该配置下主要开销仍来自大模型参数读写与矩阵计算，而非仅由输入 token 数决定。

- **近似数据读取**

| Batch Size | ft(32) | ft(128) | orca(32) | orca(128) | 主要现象 |
|---:|---:|---:|---:|---:|---|
| 1 | 约 1000 ms | 约 1000 ms | 约 1000 ms | 约 1050 ms | 四者几乎相同 |
| 2 | 约 1020 ms | 约 1050 ms | 约 1060 ms | 约 1130 ms | ORCA 略慢 |
| 4 | 约 1080 ms | 约 1120 ms | 约 1150 ms | 约 1210 ms | ORCA 略慢但接近 |
| 8 | 约 1130 ms | 约 1200 ms | 约 1230 ms | 约 1320 ms | ORCA 仍接近 FT |
| 16 | OOM / 缺失 | OOM / 缺失 | 约 1260 ms | 约 1500 ms | ORCA 可运行，FT 不可运行 |
| 32 | OOM / 缺失 | OOM / 缺失 | 约 1420 ms | 约 1950 ms | ORCA 继续扩展，但时间上升明显 |

- **性能趋势分析**
  - **FasterTransformer**
    - 在 Batch Size 1 到 8 范围内，执行时间从约 **1000 ms** 增至约 **1200 ms**。
    - 增长幅度较小，说明 batching 有效摊薄了模型参数读取和 GPU kernel 启动等固定开销。
    - 但在 Batch Size 16 和 32 下没有结果，论文说明这是因为 FasterTransformer 为每个 request 的 Attention K/V cache 按最大序列长度 **2048 tokens** 固定预分配内存，导致大 batch 下 GPU memory 不足。
  - **ORCA**
    - 在 Batch Size 1 到 8 时，执行时间与 FasterTransformer 相近，略高。
    - 这种略高主要来自 **selective batching**：ORCA 不对 **Attention operation** 做 request-wise batching，而是将 batch 拆分后分别执行 Attention。
    - 但 ORCA 对 **Linear、LayerNorm、MLP、GeLU、Add** 等非 Attention 操作仍进行 token-wise batching，因此整体性能损失较小。
    - 在 Batch Size 16 和 32 时，ORCA 仍可运行，体现其 **K/V cache 按 request 的 max_tokens 动态管理** 的优势。
  - **输入长度影响**
    - **orca(128)** 通常慢于 **orca(32)**。
    - **ft(128)** 通常慢于 **ft(32)**。
    - 这是因为 initiation phase 需要处理更多 input tokens，Attention 也需要处理更长上下文。
    - 但在 Batch Size 较小时差距不大，说明 101B 模型的主要瓶颈仍是大规模 Transformer 层计算和参数访问。

- **ORCA 与 FasterTransformer 的关键差异**

| 维度 | FasterTransformer | ORCA |
|---|---|---|
| Batching 策略 | 对所有操作统一 batching | **Selective batching**，Attention 单独处理，非 Attention 操作 batching |
| K/V cache 分配 | 通常按最大 sequence length 固定预分配 | **按 request 的 max_tokens 预留 slots** |
| 大 Batch Size 支持 | Batch Size ≥ 16 出现 OOM | **支持 Batch Size 16 和 32** |
| Engine 层性能 | 小 batch 下略优或相近 | 小 batch 下略慢，但差距小 |
| 内存灵活性 | 较差 | **更好** |
| 对后续 scheduling 的支持 | 适合 request-level scheduling | 支持 **iteration-level scheduling** |

- **为什么 ORCA 略慢但仍有价值**
  - 图中在 Batch Size ≤ 8 时，ORCA 执行时间比 FasterTransformer 略高。
  - 原因是 ORCA 的 **Attention operation 不做完整 batch 化**，这会损失一部分 GPU 执行效率。
  - 但论文强调，Attention 本身没有模型参数，不能像 Linear/MLP 那样通过 batching 显著复用参数读取，因此不 batch Attention 的代价有限。
  - ORCA 保留了非 Attention 操作的 batching，因此大部分计算仍能利用 GPU 并行性。
  - 更重要的是，ORCA 获得了 **任意请求组合成 batch 的能力**，这为后续的 **iteration-level scheduling** 提供基础。

- **图中最重要结论**
  - **ORCA engine 在 101B 模型、8 GPUs 场景下，与 FasterTransformer 的单 batch 执行效率基本相当。**
  - **ORCA 的 selective batching 没有造成严重性能损失。**
  - **ORCA 在内存管理上明显优于 FasterTransformer，可支持更大的 Batch Size。**
  - 该图主要证明：ORCA 的 engine 层并不是靠牺牲底层执行效率换取调度灵活性；它在保持接近 FasterTransformer 性能的同时，为后续端到端吞吐提升打下基础。

- **与论文整体结论的关系**
  - 这张图不是 ORCA 最大优势的体现。
  - 在该 microbenchmark 中，请求长度相同、生成长度相同，且不运行 ORCA scheduler，因此不会体现：
    - **early-finished requests 提前返回**
    - **late-joining requests 插入当前执行**
    - **iteration-level scheduling**
    - **pipeline across workers**
  - ORCA 真正的大幅收益出现在 Figure 10 的端到端实验中。
  - 但 Figure 9(b) 证明了一个关键前提：**即使使用 selective batching，ORCA engine 本身也足够高效，性能接近 FasterTransformer。**

### 99b764806eaf71042a3bcd00de9f694aeae56d4ecdf3beaacee53e0e858d5d40.jpg

![99b764806eaf71042a3bcd00de9f694aeae56d4ecdf3beaacee53e0e858d5d40.jpg](99b764806eaf71042a3bcd00de9f694aeae56d4ecdf3beaacee53e0e858d5d40.jpg)

- **图片定位**
  - 该图是论文 ORCA Figure 10 的子图 **(a) 101B model, 8 GPU**。
  - 横轴是 **Throughput，单位 req/s**，表示系统每秒完成的请求数。
  - 纵轴是 **Normalized Latency，单位 ms/token**，表示端到端延迟按生成 token 数归一化后的中位数延迟。
  - 纵轴采用 **对数刻度**，范围大致从几十 ms/token 到超过 **10³ ms/token**。
  - 图中比较了 **ORCA** 与 **FasterTransformer, ft** 在不同批大小配置下的端到端服务性能。

- **图例含义**

| 曲线 | 系统 | 配置含义 | 说明 |
|---|---|---|---|
| **ft(1, 1)** | FasterTransformer | max batch size = 1，microbatch size = 1 | 单请求执行，无批处理收益 |
| **ft(8, 8)** | FasterTransformer | max batch size = 8，microbatch size = 8 | 请求级批处理，批大小为 8 |
| **orca(1)** | ORCA | max batch size = 1 | iteration-level scheduling，但无批处理收益 |
| **orca(8)** | ORCA | max batch size = 8 | ORCA 中等批大小 |
| **orca(16)** | ORCA | max batch size = 16 | ORCA 较大批大小 |
| **orca(32)** | ORCA | max batch size = 32 | ORCA 最大批大小配置之一 |

- **核心结论**
  - **ORCA 在高负载下显著优于 FasterTransformer**。
  - **FasterTransformer 的吞吐很快达到瓶颈**，大约在 **0.4–0.6 req/s** 后延迟急剧升高。
  - **ORCA 可以在延迟较低的情况下扩展到更高吞吐**：
    - **orca(8)** 可达到约 **3 req/s**；
    - **orca(16)** 可达到约 **4 req/s**；
    - **orca(32)** 可达到约 **6 req/s 以上**。
  - 在低负载区域，**ORCA 与 FasterTransformer 差距不大**，甚至 FasterTransformer 可能接近 ORCA，因为请求不足时批处理和调度优势尚未充分发挥。
  - 随着负载升高，**ORCA 的 iteration-level scheduling 和 selective batching 优势迅速体现**。

- **主要曲线趋势**

| 配置 | 吞吐范围趋势 | 延迟趋势 | 观察 |
|---|---:|---:|---|
| **ft(1, 1)** | 约 0–0.5 req/s | 从几十 ms/token 快速升至 >10³ ms/token | 单请求处理能力有限，高负载下排队严重 |
| **ft(8, 8)** | 约 0–0.6 req/s | 低负载较低，接近饱和后急剧上升 | 批处理带来一定收益，但请求级调度限制明显 |
| **orca(1)** | 约 0–0.5 req/s | 与 ft(1,1) 类似，后期急剧升高 | max batch size=1 时 ORCA 退化，优势有限 |
| **orca(8)** | 约 0–3 req/s | 约 50–100 ms/token 区间缓慢上升，末端急剧上升 | 明显优于 ft，吞吐扩展能力更好 |
| **orca(16)** | 约 0–4 req/s | 中低负载稳定，高负载后上升 | 比 orca(8) 更高吞吐 |
| **orca(32)** | 约 0–6+ req/s | 延迟长期维持在约 80–150 ms/token，末端上升 | 图中最佳吞吐表现 |

- **FasterTransformer 的瓶颈表现**
  - **ft(1,1)** 和 **ft(8,8)** 曲线都集中在图左侧。
  - 即使使用 **ft(8,8)**，吞吐也没有明显突破 **1 req/s**。
  - 延迟在接近系统饱和时呈现接近垂直上升，说明：
    - 请求排队时间迅速增加；
    - 系统无法及时吸收新到达请求；
    - request-level scheduling 导致当前 batch 未结束前，新请求无法加入；
    - batch 内较早完成的请求不能提前返回。
  - 这正对应论文提出的两个问题：
    - **early-finished requests**：早完成请求被迫等待；
    - **late-joining requests**：新请求无法加入正在执行的 batch。

- **ORCA 的优势表现**
  - **orca(8)、orca(16)、orca(32)** 曲线明显向右扩展，说明 ORCA 能支持更高吞吐。
  - 在较长区间内，ORCA 的延迟增长较慢，说明：
    - 每个 iteration 后调度器都可以重新组织 batch；
    - 新请求最多等待一个 iteration 即可加入；
    - 已完成请求可以及时返回；
    - 不同 token 位置、不同输入长度、不同生成长度的请求仍可通过 **selective batching** 一起执行。
  - **orca(32)** 最突出：
    - 吞吐超过 **6 req/s**；
    - 在较大吞吐范围内，延迟仍保持在约 **10² ms/token** 量级；
    - 相比 FasterTransformer，吞吐提升接近一个数量级。

- **不同 ORCA batch size 的影响**
  - **orca(1)**：
    - 批大小为 1，无法利用批处理；
    - 性能与 FasterTransformer 单请求配置接近；
    - 说明 ORCA 的主要优势不是单次 kernel 更快，而是调度与批处理机制更适合生成式模型。
  - **orca(8)**：
    - 明显提升吞吐；
    - 延迟在较高吞吐前保持稳定；
    - 是较均衡配置。
  - **orca(16)**：
    - 相比 orca(8)，吞吐进一步提升；
    - 延迟仍保持较好。
  - **orca(32)**：
    - 图中吞吐最高；
    - 在高负载时仍保持较低 normalized latency；
    - 说明对该 101B 模型和测试负载而言，更大 max batch size 能有效提升 GPU 利用率。

- **为什么 ORCA 的曲线更平缓**
  - ORCA 使用 **iteration-level scheduling**：
    - 传统系统一次调度一个完整 request 或 batch；
    - ORCA 一次只调度一个模型 iteration；
    - 每生成一个 token 后，scheduler 就能重新决定 batch 成员。
  - ORCA 使用 **selective batching**：
    - 对 Linear、LayerNorm、MLP 等非 Attention 操作进行 token-wise batching；
    - 对 Attention 操作按 request 分开处理；
    - 因此不同长度、不同阶段的请求也能组合到同一个 batch 中。
  - 结果是：
    - **减少排队时间**；
    - **减少无效计算**；
    - **提升 GPU 利用率**；
    - **降低 tail congestion 对中位延迟的影响**。

- **图中最重要的系统含义**
  - 对于 **Transformer-based generative models**，传统 request-level batching 并不适合。
  - 生成任务具有 **multi-iteration autoregressive decoding** 特征：
    - 每个请求生成 token 数不同；
    - 输入长度不同；
    - 到达时间不同；
    - 每个 iteration 的状态长度不同。
  - FasterTransformer 虽然是高性能 Transformer 推理引擎，但其与传统 serving scheduler 结合时，无法灵活处理这些动态性。
  - ORCA 把调度粒度从 request 降到 iteration，使 serving system 与 execution engine 更紧密协同，因此在端到端场景中优势明显。

- **与论文文字结论的对应**
  - 论文指出，在 **101B model, 8 GPUs** 的低负载情况下，ORCA 与 FasterTransformer 差距不大。
  - 图中左侧低吞吐区域确实显示：
    - 多条曲线延迟都处于几十 ms/token；
    - 请求数量不足时，batching 与 scheduling 的优势有限。
  - 但负载升高后：
    - FasterTransformer 很快饱和；
    - ORCA 继续扩展到数倍吞吐；
    - 这验证了 ORCA 对动态生成式负载的适配能力。

- **可读出的近似性能对比**

| 系统配置 | 近似最高稳定吞吐 | 稳定区间延迟量级 | 饱和特征 |
|---|---:|---:|---|
| **ft(1,1)** | < 0.5 req/s | 数十到数百 ms/token | 很快垂直上升 |
| **ft(8,8)** | < 0.7 req/s | 数十到数百 ms/token | 批处理后仍很快饱和 |
| **orca(1)** | < 0.6 req/s | 数十到数百 ms/token | 无批处理优势，表现有限 |
| **orca(8)** | ≈ 3 req/s | 约 60–100 ms/token | 高负载后上升 |
| **orca(16)** | ≈ 4 req/s | 约 80–300 ms/token | 高负载后上升 |
| **orca(32)** | > 6 req/s | 约 80–150 ms/token 为主 | 最右端开始上升 |

- **总体评价**
  - 该图清楚展示了 **ORCA 在 101B GPT 模型、8 GPU 环境下的端到端服务优势**。
  - **核心优势不是单个 kernel 更快，而是调度粒度和批处理方式更适合 autoregressive generation**。
  - **FasterTransformer 受限于 request-level scheduling**，在请求长度和生成长度变化较大的真实服务负载下吞吐很低。
  - **ORCA 通过 iteration-level scheduling + selective batching，使高吞吐和低延迟可以同时实现**。
  - 图中最佳配置是 **orca(32)**，在保持约 **10² ms/token** 延迟量级的同时达到最高吞吐。

