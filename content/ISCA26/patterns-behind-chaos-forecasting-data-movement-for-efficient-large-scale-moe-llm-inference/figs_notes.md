# Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference 图表详解

### 87bd6df3f0e12ac68c809b7c0d0077a580d1a4f8d37accb6a11093834db633e6.jpg

![87bd6df3f0e12ac68c809b7c0d0077a580d1a4f8d37accb6a11093834db633e6.jpg](images/87bd6df3f0e12ac68c809b7c0d0077a580d1a4f8d37accb6a11093834db633e6.jpg)

- 图片展示 **ACM Artifact Review and Badging** 体系中的三枚圆形徽章，横向排列，分辨率约为 **387 × 130 px**。整体采用“齿轮/放射状奖章边缘 + 双层圆环 + ACM 标志”的统一视觉语言，强调研究工件（artifact）的开放性、可运行性与结果可复现性。

| 位置 | 徽章文字 | 主色调 | 图像元素 | 表达的认证含义 |
|---|---|---:|---|---|
| 左侧 | **Artifacts Available** | 绿色 | 绿色齿状外圈、深绿色内环、白色 “acm” 标识 | 论文的代码、数据、脚本或其他工件已对外公开，读者可以获取。 |
| 中间 | **Artifacts Evaluated – Functional** | 红/粉色 | 红色齿状外圈、红色内环、白色 “acm” 标识 | 审核者已评估该工件，确认其具备基本功能、能够安装或执行。 |
| 右侧 | **Results Reproduced** | 深蓝色 | 蓝色齿状外圈、深蓝内环、白色 “acm” 标识 | 审核者利用工件成功复现了论文所报告的关键实验结果。 |

- 三枚徽章体现出由低到高的可信度链条：
  - **Available**：工件“可获得”；
  - **Functional**：工件“可运行”；
  - **Results Reproduced**：工件不仅可运行，而且“能够支持复现实验结论”。

- 视觉编码具有明确区分：
  - **绿色**通常关联开放、可访问与可用性；
  - **红色**突出评估、验证或功能性检查；
  - **蓝色**传递正式、可靠和高可信度，对应最强的复现主张。
  - 三者中央均放置 **ACM** logo，表示该认证框架由 ACM artifact evaluation 体系提供或采用。

- 与论文内容的对应关系非常直接。论文的 Artifact Appendix 明确说明：
  - 提供了两个公开仓库及其 **Zenodo DOI**；
  - 包含代码、专家选择 traces、脚本和画图工具；
  - 提供 `main_ae.py` 以自动下载 traces、运行实验并生成 Figure 12 与 Figure 17；
  - 对 Case Study 1 给出 CPU 可运行的 wafer-scale GPU simulator，对 Case Study 2 给出真实 **8×H100** 实验复现路径；
  - 说明了软硬件依赖、磁盘空间、运行时间与预期结果。

- 因而，该图若用于这篇论文的投稿或展示材料，传达的核心信息是：论文并非仅报告性能数字，而是将研究结果配套为可审查的实验工件。特别是 **Results Reproduced** 徽章意味着评审流程中应已基于所提交的 artifact 成功再现关键趋势或图表，而不只是确认代码链接存在。

- 从科研可信度角度看，三枚徽章共同降低了该类系统论文的验证风险：

| 可信度维度 | 图片对应徽章 | 对本文的价值 |
|---|---|---|
| 可访问性 | **Artifacts Available** | 读者可获得超过 150 GB 的 expert-selection traces、模拟器及实验代码。 |
| 可执行性 | **Artifacts Evaluated – Functional** | 验证模拟器和实验工作流能够按文档运行，而非不可部署的原型。 |
| 结果可靠性 | **Results Reproduced** | 支撑其核心主张，例如 wafer-scale 场景的平均 **6.6×** 吞吐改进，以及真实 GPU 集群中最高 **1.25×** 的 MoE computation 加速。 |

- 图片本身没有展示实验数据、模型结构或算法流程；它属于一种 **research reproducibility certification badge**。其信息重点不是“方法如何工作”，而是“论文的实验资产是否公开、是否可用、结果是否可复现”。

### Figure 1. MoE LLM models sizes and release dates. Bubble size indicates the number of experts in each layer. Prior studies [13], [15]–[17] provide limited analysis of smaller models from narrow perspectives, while our work presents the first comprehensive analysis of multiple unstudied SOTA models.

![eae2ec48d9fa3ef19211fef8d192565e24b9451d80d2d6d9014f72335ba80b85.jpg](images/eae2ec48d9fa3ef19211fef8d192565e24b9451d80d2d6d9014f72335ba80b85.jpg)

- **图表定位**：该图以时间和模型规模为双轴，展示 MoE LLM 的演进轨迹，并用颜色区分既有研究覆盖范围与本文 profiling 覆盖范围。核心论点是：**MoE 模型在 2025 年快速进入 200B–1000B+、百级专家的规模，而既有数据移动分析主要停留在较小模型。**

- **坐标与视觉编码**：

  | 视觉元素 | 含义 | 图中信息 |
  |---|---|---|
  | 横轴 | 发布年份 | 从 **2023** 延伸至 **2026** |
  | 纵轴 | Parameter Size (B) | 参数规模，刻度约为 **8B、40B、200B、1000B**；刻度间距表明其采用或近似采用**对数尺度** |
  | 气泡位置 | 模型的发布时间与参数量 | 越靠右表示发布时间越晚，越靠上表示参数量越大 |
  | 气泡面积 | 每层的专家数量 | 气泡越大，意味着该模型每层拥有更多 experts |
  | 蓝色 | LLMs with no study | 未被相关工作研究的模型 |
  | 绿色 | LLMs with limited prior study | 仅有有限既有研究的模型 |
  | 红色 | SOTA LLMs first profiled in our work | 本文首次系统 profiling 的 SOTA 模型 |
  | 黄色 | LLMs Released after our work | 在本文工作之后发布的模型 |

- **图中的模型分布可概括如下**：

  | 时间段 | 代表模型 | 大致规模 | 研究覆盖状态 | 图传达的意义 |
  |---|---|---:|---|---|
  | 2023 | JetMoE | 约 **8B** | 蓝色 | 早期、小型 MoE，尚缺少相关研究 |
  | 2023–2024 | Qwen1.5-MoE、Mixtral8x7B、Openmoe-34B、Phi-3.5-MoE | 约 **20B–40B** | 主要绿色 | 既有研究主要聚焦此类小模型 |
  | 2024 | DBRX、DeepSeekCoder-V2、Grok1、Jamba、Mixtral8x22B | 约 **40B–300B** | 蓝色/绿色 | 模型规模开始上升，但系统性数据移动分析仍不充分 |
  | 2025 | **Qwen3-235B** | **235B** | 红色 | 本文 profiling 对象之一 |
  | 2025 | **Llama4-Maverick** | **402B** | 红色 | 本文 profiling 对象之一 |
  | 2025 | **DeepSeek V3** | **671B** | 红色 | 本文 profiling 对象之一 |
  | 2025 | **Kimi K2** | **1000B** | 红色 | 本文 profiling 对象之一，也是图中最大规模的已分析模型 |
  | 2026 | GLM-5、DeepSeek V4 | 约 **200B–1000B+** | 黄色 | 表明 MoE 前沿仍在向更大参数与更多专家持续扩张 |

- **最重要的规模跃迁**：
  - 2023–2024 年的代表模型大多集中于 **8B–50B** 区间，少数达到约 **200B–300B**。
  - 到 2025 年，前沿模型迅速跨越至 **235B、402B、671B、1000B**。
  - 图中的红色气泡不仅整体更靠上，也通常更大，意味着新一代模型同时扩大了：
    - **总参数量**；
    - **每层 expert 数量**；
    - 因而也扩大了 expert routing 的组合空间与潜在数据移动复杂度。
  - 以论文正文中的 DeepSeek V3 为例，其每层从 **256 个 experts 中选择 8 个**；潜在组合数量为  
    **C(256, 8) = 4,426,165,368**。  
    这说明即使单 token 只激活少数 experts，调度、缓存、迁移和跨设备通信仍会面对极大的动态选择空间。

- **气泡大小揭示的结构性趋势**：
  - 小模型如 JetMoE、Qwen1.5-MoE、Mixtral8x7B 的气泡较小，反映其每层 expert 数相对有限。
  - **DeepSeek V3、Kimi K2、DeepSeek V4** 等模型的气泡显著增大，体现了前沿 MoE 从“稀疏激活的小规模扩展”转向“**超大参数量 + 大专家池**”的架构范式。
  - 这直接强化了论文的问题设定：当每层 experts 更多时，expert placement、All-to-All、remote-memory access 与负载失衡不再是次要开销，而会成为 serving 的主导瓶颈。

- **颜色分布支持的论文动机**：
  - 绿色模型集中在左下区域，表示先前工作虽涉及 MoE 分析，但对象主要是 **Mixtral8x7B、Openmoe-34B、Phi-3.5-MoE** 等较小模型。
  - 蓝色模型说明即使一些中大型 MoE 已发布，也仍然缺乏专门的 data-movement-centric study。
  - 红色模型集中于 2025 年、参数量 **200B–1000B** 区间，直观表达本文覆盖了此前未被系统分析的 SOTA MoE。
  - 因此，该图并不是为了比较模型性能，而是用于建立研究缺口：**已有结论未必能外推到当前超大规模 MoE serving 场景。**

- **粉色阴影区域的含义**：
  - 图中从 2024 年约 1000B 附近向 2026 年约 10B 附近倾斜的粉色带状区域，视觉上强调了模型发布序列中“**越新的模型总体越大、专家规模也越高**”的趋势。
  - 它不是严格的拟合线或置信区间；更合理的解读是作者用其突出 frontier MoE 的快速规模化方向。
  - 该趋势解释了为什么过去针对小模型所得的缓存、路由和 placement 经验可能不足：系统瓶颈会随模型规模、expert 数量、跨节点部署规模发生质变。

- **与全文方法的对应关系**：
  - 红色的四个核心模型对应论文实际 profiling 对象：
  
    | 本文模型 | 参数规模 | 在图中的角色 |
    |---|---:|---|
    | Qwen3-235B | **235B** | 超大规模 MoE 的较低端基线 |
    | Llama4-Maverick | **402B** | 中高参数、不同 routing 架构代表 |
    | DeepSeek V3 | **671B** | 高 expert-count、大规模 serving 的关键对象 |
    | Kimi K2 | **1000B** | 千亿至万亿参数级 MoE 的极端代表 |

  - 通过这四种模型，论文避免将某一模型的 expert-selection 特征误认为普遍规律，并据此提炼了跨模型的 temporal 与 spatial insights。

- **该图对论文贡献的支撑**：
  - 它为“**first comprehensive analysis**”这一主张提供了范围证据：本文不是只对一个模型、一个 workload 或一种硬件环境做局部观察，而是覆盖多个此前未研究的超大规模 SOTA MoE。
  - 图中的红色集群明确位于当前模型演进前沿，因此论文后续对：
    - prefill-to-decode correlation；
    - cross-layer / cross-token expert correlation；
    - hot-expert skewness；
    - task/language-aware routing；
    - co-activated expert-pair affinity；
    
    的分析，具有比小模型研究更强的现实部署意义。

- **需要谨慎解读之处**：
  - **气泡面积只表示每层 expert 数量**，不表示 activated parameters、吞吐量、精度、训练成本或推理延迟。
  - 参数规模并不完全等价于实际 MoE inference cost；真正影响成本的还包括 top-k、expert hidden size、MoE layer 数、batch size、expert placement 和互连拓扑。
  - 图中黄色的 DeepSeek V4 与 GLM-5 属于“本文之后发布”的模型，它们用于说明趋势延续，**不是本文 profiling 与实验评估对象**。
  - 图的论证重点是研究覆盖空白与规模演进，不能单独据此得出某一个模型的实际通信量一定更高；这一结论仍需结合论文后续 profiling 和 latency breakdown 结果。

### a5717c77c8d18fc9c589075293004fa16adbb237ada6d4924f62c001921af639.jpg

![a5717c77c8d18fc9c589075293004fa16adbb237ada6d4924f62c001921af639.jpg](images/a5717c77c8d18fc9c589075293004fa16adbb237ada6d4924f62c001921af639.jpg)

- 该图为论文 **Figure 2**，比较 DeepSeek-V3 在 **4K sequence** 下、三种 serving configuration 的归一化端到端延迟构成。横轴是 **Total Time**，每一条堆叠柱均归一化为 100%。

| 配置 | MoE All-to-All | MoE Weight | Attn All Reduce | Attn KV Cache | Attn Weight | 核心瓶颈 |
|---|---:|---:|---:|---:|---:|---|
| SGLang 16×H20 | 约 60% | 约 3% | 接近 0% | 约 8% | 约 29% | MoE 通信与 Attention weight 读取并重 |
| SGLang 72×H100 | 约 85% | 约 3% | 接近 0% | 约 5% | 约 7% | **MoE All-to-All 绝对主导** |
| Default 256×H800 | 约 76% | 约 2% | 约 18% | 约 3% | 约 2% | MoE 通信主导，同时 Attention collective 成本显著 |

- 图例对应五类开销：
  - **MoE All-to-All**：token 经 router 分发至不同 expert、以及 expert 输出回传时产生的跨 GPU 通信。
  - **MoE Weight**：expert weight 的获取、加载或迁移开销，常见于 offload / 分层存储情形。
  - **Attn All Reduce**：Attention 的 tensor/sequence parallel collective communication。
  - **Attn KV Cache**：KV cache 的访问与传输成本。
  - **Attn Weight**：Attention layer 权重读取或计算相关的时间。

- 最突出的视觉结论是：三种配置中，粉红色的 **MoE All-to-All** 始终是最大区段，占比约为 **60%–85%**。这直接支撑论文的中心论点：对于大规模 MoE serving，系统的主要矛盾已从纯计算转向 **expert routing 所导致的数据移动**。

- 配置规模从 **16×H20** 增至 **72×H100** 时，MoE All-to-All 的占比由约 60% 上升到约 85%。这说明：
  - 更多 GPU 虽提供更强 aggregate compute；
  - 但 Expert Parallelism 扩大后，token 的跨设备分发范围也扩大；
  - 因而计算加速并不能自动转化为端到端加速，反而可能被 **all-to-all communication** 吞噬。

- **SGLang 16×H20** 中，灰色的 Attn Weight 仍接近 30%，表明较小规模或较低带宽配置下，Attention weight 的访问也不可忽略。但即便在这一情况下，MoE 相关开销（MoE All-to-All + MoE Weight）仍已超过约 **60%**。

- **SGLang 72×H100** 的柱状图最具代表性：几乎整个执行时间被 MoE All-to-All 占据。这意味着在高性能 GPU 集群中，单卡计算能力提升后，MoE 的动态 token exchange 成为更尖锐的瓶颈；优化 GEMM 本身的边际收益会显著低于优化通信、placement 和 routing locality。

- **Default 256×H800** 中，除了约 78% 的 MoE 相关开销，还出现约 18% 的 **Attn All Reduce**。这反映超大规模部署会暴露第二类通信问题：
  - MoE 中的 token-to-expert 路由产生 All-to-All；
  - Attention 并行化产生 All-Reduce；
  - 两类 collective communication 叠加，使网络拓扑、带宽、拥塞控制和并行策略成为系统性能的决定因素。

- 该图的设计价值在于，它并非仅说明“通信很慢”，而是揭示了优化优先级：
  1. 首先优化 **MoE All-to-All**，例如 expert placement、token/task allocation、expert replication、co-activation-aware placement。
  2. 其次降低 **MoE Weight** 的远端访问，例如 prefetch、cache、migration 与分层内存管理。
  3. 在超大规模集群中，再联合处理 **Attention All Reduce**，避免只优化 MoE 后瓶颈转移至 Attention collective。

- 图中结果也为后续六项 insight 提供了问题背景。尤其是论文提出的：
  - **Prefill-data-driven prediction**：利用 prefill expert trace 预测 decode 的热门 expert；
  - **Cross-hierarchy memory management**：利用 token/layer temporal locality 做 prefetch 与 caching；
  - **Expert-placement-aware workload distribution**：根据 expert 所在位置和负载分配任务；
  - **Popular expert decentralization**：复制或分散高频 expert；
  
  它们的共同目标都是减少图中占比最大的粉红色区段，即 **MoE All-to-All data movement**。

- 需要注意，图展示的是**时间占比**而非绝对 latency。因此它适合比较“瓶颈结构”，不能直接据此判断某一配置的绝对响应时间更低。即使某系统的 MoE All-to-All 占比更高，也可能因为总时间更短而拥有更低的绝对延迟。

### f78706f02fbc47e399b362d2600964bafcdbee653a218f232f3d61e2989faeeb.jpg

![f78706f02fbc47e399b362d2600964bafcdbee653a218f232f3d61e2989faeeb.jpg](images/f78706f02fbc47e399b362d2600964bafcdbee653a218f232f3d61e2989faeeb.jpg)

- **图片定位**：该图对应论文 Figure 4，分析 MoE 中**相邻层（Cross-Layer）专家选择的条件相关性**，即已知当前层选择 Expert \(e_i\) 后，下一层选择 Expert \(e_j\) 的概率 \(P(e_j\mid e_i)\)。

- 图由三部分构成：

| 子图 | 对象 | 横轴 | 纵轴 | 核心作用 |
|---|---|---|---|---|
| (a) | DeepSeek V3 | Next Layer Expert ID，0–256 | Previous Layer Expert ID，0–256 | 展示 256 个 experts 下相邻层的联合路由模式 |
| (b) | Qwen3 | Next Layer Expert ID，0–128 | Previous Layer Expert ID，0–128 | 展示 128 个 experts 下相邻层的联合路由模式 |
| (c) | DeepSeek V3、Qwen3、Llama 4、Kimi K2 | Top-\(K\) Experts (% of Pool) | Cumulative \(P(e_j\mid e_i)\) | 量化不同模型中跨层预测的集中程度 |

- **热力图的语义**：
  - 每一个像素代表一个 expert 对 \((e_i,e_j)\)。
  - 横轴是**下一层 expert ID**，纵轴是**前一层 expert ID**。
  - 颜色表示条件概率 \(P(e_j\mid e_i)\)：颜色越亮，意味着当前层激活 \(e_i\) 时，下一层激活 \(e_j\) 的概率越高。
  - 色条范围约为 **0 到 0.8**。大多数区域颜色较暗，说明并非所有 expert 对都有明显关联；少数亮点、亮线则表明存在高度偏好的跨层 expert 转移关系。

- **DeepSeek V3 的观察**：
  - 子图 (a) 分别比较了 **Layer 3 → Layer 4** 与 **Layer 30 → Layer 31**。
  - 两个层对都不是均匀噪声，而是可观察到局部较亮的点状或条带状区域，表明 expert routing 存在跨层统计依赖。
  - 但整体热图相对较暗、分散，说明 DeepSeek V3 的跨层相关性在四个模型中相对较弱；这与子图 (c) 中其曲线最靠右的结果一致。
  - 低层与高层的热力图纹理并不完全相同，意味着**跨层关联具有 layer-dependent 特征**：不能简单用全模型统一的 expert transition table 替代逐层建模。

- **Qwen3 的观察**：
  - 子图 (b) 展示相同的两个层对：**Layer 3 → Layer 4** 和 **Layer 30 → Layer 31**。
  - 相比 DeepSeek V3，Qwen3 热图中可见的高概率局部结构更突出，表明其路由行为具有更强的条件集中性。
  - 这意味着：给定当前层选择的 experts，下一层的候选 experts 可以被更有效地缩小。
  - 从系统角度，Qwen3 更适合采用基于跨层历史的 **expert prefetching**、**LLC caching** 或局部 expert migration。

- **CDF 曲线的量化结论**：
  
| 模型 | Top 20% 下一层 experts 覆盖的条件概率质量 | 跨层相关性强度 | 对预测/缓存的意义 |
|---|---:|---|---|
| DeepSeek V3 | **50%** | 四者中最弱 | 仍有可利用局部性，但候选池需更大 |
| Qwen3 | **65%** | 较强 | 可较有效地筛选下一层 expert 候选 |
| Llama 4 | **77%** | **最强** | 最适合激进的跨层预取与缓存 |
| Kimi K2 | **56%** | 中等 | 存在稳定收益空间，但不如 Qwen3/Llama 4 集中 |

- 子图 (c) 的读取方式：
  - 横轴表示保留下一层 expert 候选集合的比例，例如 **20%** 代表只保留全部 experts 中概率最高的前 20%。
  - 纵轴表示这些候选所累计覆盖的真实条件概率质量。
  - 曲线越靠近**左上角**，表示只需少量候选 experts 就可覆盖大部分下一层选择概率，因而可预测性越强。
  - 曲线排序为：**Llama 4 > Qwen3 > Kimi K2 > DeepSeek V3**。
  - 即使是相关性最弱的 DeepSeek V3，前 20% 候选仍覆盖约一半概率质量，说明“大规模 MoE routing 完全随机”的假设并不成立。

- **关键论文结论**：
  - 图中证明了相邻 MoE layers 之间存在可利用的**Cross-Layer expert correlation**。
  - 这种相关性不是模型无关的固定规律，而是同时受**模型架构、具体 layer 位置、expert 数量与训练后路由行为**影响。
  - 因此，系统应使用**per-model、per-layer 的条件概率表或 heatmap**，而非采用统一静态策略。

- **直接对应的系统设计启示**：
  - 对于短 reuse distance 的相邻层，可将当前层的路由结果作为下一层预测器输入。
  - 可在当前层计算期间，提前将下一层高概率 experts 的权重加载至更快存储层，例如 **LLC、local HBM 或 local DRAM**。
  - 预测候选数应按模型自适应配置：
  
| 模型特征 | 建议策略 |
|---|---|
| 高相关性，如 Llama 4、Qwen3 | 选择较小 Top-\(K\)，进行积极 prefetch，降低无效搬运 |
| 中低相关性，如 Kimi K2、DeepSeek V3 | 扩大候选集合，或仅预取权重较小、迁移成本较低的 experts |
| 不同层相关性差异明显 | 使用 layer-specific predictor，而非全局 predictor |

- **图的局限性**：
  - 热图展示的是聚合后的统计条件概率，不能保证单个 token 的 expert routing 一定可精确预测。
  - 图仅覆盖**相邻层**，不能直接推出远距离层之间也存在同等强度的相关性。
  - 高概率预测会带来缓存/预取收益，但也会产生额外容量占用和错误预取流量；实际系统需要结合 expert weight 大小、HBM 容量、D2D 带宽与替换策略做成本决策。
  - 因而，该图支持的是**概率性、统计性优化**，而不是确定性的 routing 替代。

### (a) MoE-LLM inference process and temporal relations (b) MoE operation and spatial Relation Figure 3. Inference process of MoE LLMs and the categorization method for our proposed data-centric profiling approach.

![6a9284cd6179cc7a026cc7f52ce48afea9d425e060ba08581e7e5ae3ec5740c2.jpg](images/6a9284cd6179cc7a026cc7f52ce48afea9d425e060ba08581e7e5ae3ec5740c2.jpg)

- **图3的核心作用**：该图不是性能结果图，而是论文的**分析框架图**。它将 MoE-LLM 中看似随机的 expert routing 划分为两类可研究、可优化的数据运动规律：
  - **Temporal Relation（时间关系）**：同一请求在不同推理时间点的 expert selection 是否存在相关性。
  - **Spatial Relation（空间关系）**：同一时刻/同一 MoE layer 内，expert 请求如何在 experts 与计算单元之间分布。

- **整体数据流**可概括为：用户请求 → Prefill Stage → 多次 Decode Stage iteration → 每层中的 Attention 与 MoE operation → Gating 将 token/request 路由至 experts。图中左半部分解释“未来会选谁”，右半部分解释“当前哪些 expert 被选得多、哪些 expert 经常一起被选”。

- 左侧的 **Temporal Relation** 展示了自回归推理过程：
  - 用户输入示例为 “Tell a story / Who are you / Is Apple red / …”，这些输入 token 首先进入 **Prefill Stage**。
  - 在 **LLM Iter 0** 中，输入序列的多个 token 可并行经过 Layer 0、Layer 1 到 Layer N，以计算首个输出 token。
  - 随后进入 **Decode Stage**。每一个 LLM iteration 生成一个 token，例如图中的 “Once”、“upon”、“a”、“…”；新 token 被拼接到上下文中，再触发下一轮 iteration。
  - 该流程决定了 MoE routing 的三个时间尺度：**跨层、跨 token、跨阶段**。

- 图中红色箭头定义了三项 temporal observations，其含义如下：

| 图中标记 | 分析粒度 | 比较对象 | 图示位置 | 可支持的系统优化 |
|---|---:|---|---|---|
| **Ob1: Layer Level** | 相邻 layer | 当前 token 在 Layer \(l\) 与 Layer \(l+1\) 的 expert selection | Decode Iter 2 内部的相邻竖条 | 短重用距离的 **LLC cache、layer-ahead prefetch** |
| **Ob2: Token Level** | 相邻 token | 同一 layer 在 token \(t\) 与 token \(t+1\) 的 expert selection | Decode Iter 1 与 Iter 2 之间 | 较长重用距离的 **local DRAM/HBM cache、next-token prediction** |
| **Ob3: Prefill-decode Level** | 两个推理阶段 | Prefill 的 routing trace 与早期 Decode 的 routing behavior | Prefill Iter 0 指向 Decode Iter 1 | **prefill-guided placement、decode cold-start prediction** |

- **Ob1: Layer-Level correlation** 的箭头从同一 Decode iteration 内的一个 MoE layer 指向下一 MoE layer。它表达的不是“相邻 layer 使用同一个 expert”，而是：已知上一层激活 expert 后，下一层被选 expert 的概率分布会收缩到较小候选集合。  
  - 这类关系的时间间隔最短。
  - 因而适合放在访问延迟最低、容量较小的存储层级中利用，如 **LLC**。
  - 对于 remote expert weights，系统可在当前 layer 执行时预取下一 layer 的高概率 experts。

- **Ob2: Token-Level correlation** 的红色弧线连接相邻 Decode iteration。它刻画相邻生成 token 的 routing 相关性：
  - token \(t\) 经过某个 layer 选择的 experts，会影响 token \(t+1\) 在该 layer 的选择分布。
  - 图中以 “Once” 到 “upon” 的生成推进来直观表示这一关系。
  - 它的重用距离比 Ob1 长：必须等待 token 穿过剩余 Transformer layers 并完成下一轮 decode。
  - 因此论文将其对应到容量更大但延迟更高的 **local HBM/DRAM**，用于跨 token 的 expert caching 或 migration。

- **Ob3: Prefill-decode correlation** 跨越了推理的两个阶段：
  - Prefill 已经处理了完整 prompt，因此天然积累了丰富的 expert activation trace。
  - Decode 刚开始时历史输出 token 很少，通常缺乏在线统计数据；但 prefill trace 可作为初始预测信号。
  - 这对于 **Prefill-Decode disaggregation** 尤为重要：即使 Prefill 与 Decode 运行于不同 GPU/机器，也可传递一份轻量级 expert-frequency 或 routing summary，指导 decode 端的初始 expert placement。
  - 图的箭头从 Prefill 直接连接到 Decode Iter 1，准确强调该方法主要解决 **decode cold start**。

- 右侧的 **Spatial Relation** 放大了一个典型 Transformer layer 的内部结构：
  - 每层由 **Attention** 和 **MoE** 两部分组成。
  - Attention 之后的多个 requests/tokens，以 A、B、C、D 表示，进入 **Gating**。
  - Gating 为每个 request 选择若干 experts；蓝色连线代表 token/request 到 selected expert 的动态路由。
  - Expert 0–5 表示一层中可供选择的 experts。实际模型中 expert 数量可达 128、256 或更多，图中仅做示意。

- 图中 requests 的矩阵和右侧 **Req Num** 柱状分布共同表达了 **single-expert activation imbalance**：
  - 左边 A、B、C、D 的颜色深浅可理解为不同 request/token 的 routing 请求。
  - 右侧每个 expert 对应的柱/楔形宽度表示其收到的请求数量，即 **Req Num**。
  - 图形明显不均匀：例如 Expert 3、Expert 4 的请求量高于其他 experts，而 Expert 0、Expert 5 较低。
  - 这就是 **Ob4: Single Expert**：expert activation 不是均匀随机分布，而具有热点与长尾。

| 空间观察 | 图中视觉编码 | 反映的现象 | 主要系统风险 | 对应优化方向 |
|---|---|---|---|---|
| **Ob4: Single Expert** | 右侧不同高度/宽度的 Req Num | 热门 expert 的 activation frequency 显著更高 | 个别 GPU/die 过载、其他单元空闲 | **popular expert duplication、decentralization、load-aware placement** |
| **Ob5: Expert Pair** | 一个 request 同时连向多个 experts | 特定 expert pairs 有更高 co-activation probability | 共激活 experts 共置时形成热点 | **expert-pair separation**，以并行化处理 |

- **Ob4 的系统含义**很直接：如果一个热门 expert 只部署在单个 GPU/die，那么大量 token 会汇聚到该单元，导致：
  - 该单元的 GEMM 和 HBM bandwidth 饱和；
  - 其他单元即使空闲，也需等待热点单元完成；
  - 来自远程单元的 all-to-all 或 weight access 增多；
  - 最终造成吞吐受最慢 expert shard 限制。  
  因此，论文后续提出 **popular expert decentralization**：复制热门 experts，或避免把多个热门 experts 放在同一个计算单元。

- **Ob5: Expert Pair co-activation affinity** 由同一 request 指向多个 experts 的蓝色连线表示。对于 top-\(k\) MoE，一个 token 不只选择一个 expert；因此需要研究“哪些 expert 经常同时出现”：
  - 若频繁共激活的一对 experts 被放在同一 die，它们的计算和 memory access 将叠加，产生局部拥塞。
  - 将这类 pairs 分置于不同单元，能让它们并行执行，提高资源利用率。
  - 但这种分离也可能增加跨单元通信。因此图隐含的设计原则是：**在 parallelism gain 与 communication cost 之间做 topology-aware trade-off**，而非机械地拆散所有 pairs。

- 中间从 Decode iteration 指向右侧 layer/MoE 的灰色虚线，强调两类关系的连接方式：
  - **Temporal Relation** 追踪同一请求随推理进展的 routing 演化；
  - **Spatial Relation** 则在某一指定 layer、指定时刻截取 routing 快照，统计所有 requests 对 experts 的负载分布及共激活关系。
  - 前者更适合 **fine-grained、动态**的 prefetch/cache/migration；后者更适合 **coarse-grained、全局**的 placement、replication 和 workload distribution。

- 图3与论文六项 Insights 的映射关系清晰：

| 图中 observation | 论文对应 Insight | 核心用途 |
|---|---|---|
| **Ob3: Prefill-decode Level** | **Insight 1: Prefill-data-driven prediction** | 用 prefill trace 预测 decode routing 与热点 expert |
| **Ob1 + Ob2** | **Insight 2: Cross-hierarchy memory management** | 分别驱动短距离与长距离的分层 cache/prefetch |
| **Ob4 + Ob5** | **Insight 3: Expert-placement-aware workload distribution** | 依据 placement、hotness、co-activation 分配任务 |
| **Ob4** | **Insight 4: Popular expert decentralization** | 复制或分散热点 expert |
| **Ob5** | **Insight 5: Expert-pair separation** | 将高频 co-activated experts 分离以提高并行性 |
| **Ob4 与 workload metadata** | **Insight 6: Workload-aware serving system** | 按任务类型、语言预先部署相应 experts |

- 该图最重要的论证是：MoE routing 虽然在单次 gating 前不可见，却并非统计意义上的完全随机。它同时具有：
  - **时间可预测性**：前 layer、前 token、Prefill 都对后续 routing 有信息价值；
  - **空间非均匀性**：expert 热度存在偏斜，expert pairs 存在结构化共激活；
  - **工作负载依赖性**：不同 prompt、任务类型和语言可诱导不同 expert distribution。  

- 因此，图3为全篇奠定了方法论：从“routing 不可预知，只能被动 all-to-all”转向“利用 routing pattern 主动进行 **prediction、caching、placement、replication 与 scheduling**”。

### 6fd17182387a5a1ecdd39e1c97b3f95a6aa09e116bc57a0a9d75f948e2889854.jpg

![6fd17182387a5a1ecdd39e1c97b3f95a6aa09e116bc57a0a9d75f948e2889854.jpg](images/6fd17182387a5a1ecdd39e1c97b3f95a6aa09e116bc57a0a9d75f948e2889854.jpg)

- 该图展示 **Cross-layer expert correlation（跨层专家相关性）**：比较相邻 MoE 层中，前一层专家被激活时，下一层各专家被激活的条件分布。图中每个像素可理解为：
  \[
  P(e^{l+1}_j \mid e^l_i)
  \]
  即“当前层选择 Expert \(i\) 后，下一层选择 Expert \(j\) 的概率”。

- 图由两组模型、四个相邻层对组成：

| 子图 | 模型 | Expert 数量 | 对比层 | 坐标范围 |
|---|---:|---:|---|---|
| 左上 | DeepSeek V3 | 256 | Layer 3 → Layer 4 | 0–256 |
| 右上 | DeepSeek V3 | 256 | Layer 30 → Layer 31 | 0–256 |
| 左下 | Qwen3 | 128 | Layer 3 → Layer 4 | 0–128 |
| 右下 | Qwen3 | 128 | Layer 30 → Layer 31 | 0–128 |

- 坐标含义明确：
  - **纵轴 Previous Layer Expert ID**：前一层被选择的专家编号。
  - **横轴 Next Layer Expert ID**：下一层被选择的专家编号。
  - 颜色由深红至浅色表示条件概率由低到高；图右侧 colorbar 表示归一化强度。**浅色点、竖向亮线越明显，说明下一层的选择越具有确定性或偏好性。**

- DeepSeek V3 的两个热图整体呈现较均匀的深红背景，仅夹杂零散亮点：
  - 这意味着其跨层路由虽然并非完全随机，但整体条件相关性相对弱、分布更分散。
  - 左上 Layer 3→4 中，局部亮点较为稀疏，说明低层相邻 MoE layers 的专家组合存在有限但明确的偏好。
  - 右上 Layer 30→31 中，部分竖向区域略亮，表明某些 **Next Layer experts** 会在多种前序专家条件下被频繁选中，即存在“全局热门专家”。
  - 相比于强对角线结构，该图并没有突出的对角线，因此不能简单解释为“专家 \(i\) 在下一层仍倾向选择编号相同的专家 \(i\)”；其主要模式是**特定跨层 expert pairs 的关联**。

- Qwen3 的相关性结构显著更强：
  - 左下和右下均有更多高亮散点与清晰的亮色纵向条带。
  - **纵向亮线**尤其关键：它表示无论前一层激活哪个专家，下一层某些固定 Expert ID 都有较高概率被激活。这反映出 Qwen3 某些专家具有更强的通用性或路由吸引力。
  - 右下 Layer 30→31 的亮竖线更突出、聚集更明显，说明在该高层相邻层对中，下一层专家选择存在更强的集中性。
  - 这也与论文文字结论一致：**Qwen3 的跨层相关性强于 DeepSeek V3**，不同层间的模式仍有明显差异。

- 从“层位置”看，图传达了一个重要结论：**跨层相关性是 layer-pair-dependent，而非整个模型共享同一个固定路由规律。**
  - DeepSeek V3 的 Layer 3→4 与 Layer 30→31 的亮点位置、密度和竖向集中区域不同。
  - Qwen3 的两个层对同样呈现不同的高概率专家区域。
  - 因而，预测器不能只维护一个模型级全局 expert popularity 排名；更合理的做法是维护 **per-layer transition heatmap**，即针对每一对相邻层单独建模 \(P(e^{l+1}\mid e^l)\)。

- 该图对系统设计的直接意义如下：

| 观察到的视觉模式 | 所对应的系统含义 | 可采用的优化 |
|---|---|---|
| 零散高亮点 | 存在高概率的 expert-pair transition | 依据当前层路由结果预取下一层专家 |
| 明显亮竖线 | 某些下一层专家跨多种条件均热门 | 对热门专家进行缓存、复制或优先放入本地 HBM/DRAM |
| 层对之间模式不同 | 相关性具有 layer specificity | 为每层维护独立预测表，而非全局静态表 |
| Qwen3 的亮线/亮点更强 | 可预测性更高 | Qwen3 更适合激进的 prefetching 与 cache placement |
| DeepSeek V3 分布更平坦 | 预测收益可能较小且需更谨慎 | 使用 Top-k 候选预取，避免过度复制造成容量和带宽浪费 |

- 图支撑论文中的 **Insight 2: Cross-hierarchy memory management**。其逻辑是：
  - 在当前 MoE layer 完成 gating 后，可由已知的 Previous Layer Expert IDs 查询对应热图行。
  - 从该行选取概率最高的若干 Next Layer experts。
  - 将这些候选专家提前放入更快的缓存层级，例如 **LLC**，以隐藏紧邻下一层执行时的远端访存延迟。
  - 由于相邻层间 reuse distance 很短，跨层关系特别适用于快速但容量有限的近端缓存；相比之下，跨 token 关系更适合管理 local DRAM、CXL memory 或其他较慢但容量更大的层级。

- 对图中结果应保持两个边界认识：
  - **热图证明的是统计相关性，不是确定性路由规则。** 即使亮点较明显，仍应使用 Top-k 或概率阈值预测，而非只预测一个专家。
  - 图中展示的是 expert ID 层面的条件分布；实际预取收益还取决于 expert weight 大小、batch size、cache capacity、D2D bandwidth、远端访问延迟以及错误预取代价。

- 总体而言，该图的核心信息是：**MoE expert routing 表面上动态且稀疏，但相邻层之间存在可利用的结构化条件相关性；Qwen3 的该规律比 DeepSeek V3 更显著，且规律随层深而变化。** 这为按层构建轻量级 expert predictor、执行跨层预取和分层缓存管理提供了实证依据。

### de4db1a7213b16b2993303249699a1ada002e0bb15da833304cbc697fcfed04a.jpg

![de4db1a7213b16b2993303249699a1ada002e0bb15da833304cbc697fcfed04a.jpg](images/de4db1a7213b16b2993303249699a1ada002e0bb15da833304cbc697fcfed04a.jpg)

- 图片展示 **Figure 4(c)**：不同 MoE 模型中，给定当前层已选择的 top-1 expert 后，下一层 expert 选择的条件累计分布，即 **Conditional CDF of next-layer expert selection**。

- 横轴为 **Top-K Experts (% of Pool)**：
  - 表示按条件概率从高到低排序后，取下一层 expert 池中前 K% 的候选 expert。
  - 横轴越靠左，代表只保留越少的候选 expert。

- 纵轴为 **Cumulative \(P(e_j \mid e_i)\)**：
  - 表示这些 top-K 候选 expert 覆盖的下一层条件选择概率总和。
  - 曲线越陡、越靠左上，说明当前层 expert \(e_i\) 对下一层 expert \(e_j\) 的预测能力越强，即跨层 routing 越有规律。

| 模型 | 曲线特征 | Top 20% 候选覆盖的条件概率质量 | 跨层相关性强度 |
|---|---:|---:|---|
| **Llama 4** | 最靠左上、上升最快 | **约 77%** | **最强** |
| **Qwen3** | 次陡，明显优于 DeepSeek V3/Kimi K2 | **约 65%** | 强 |
| **Kimi K2** | 居中，前段上升较快但后段趋缓 | **约 56%** | 中等 |
| **DeepSeek V3** | 曲线最靠右、最平缓 | **约 50%** | 四者中最弱，但仍存在显著相关性 |

- 图中最关键的定量结论是：
  - 对 **Llama 4**，仅依据当前层已激活 expert，保留下一层 **20%** 的 expert 候选，就能覆盖约 **77%** 的实际条件概率。
  - 对 **Qwen3**，相同候选规模覆盖约 **65%**。
  - 即便相关性最弱的 **DeepSeek V3**，top 20% 候选仍可覆盖约 **50%** 概率，而完全随机选择下，20% 候选理论上只应覆盖约 **20%** 概率。
  - 因此，MoE 的跨层 expert routing **并非独立随机事件**；其条件分布显著集中在少量 expert 上。

- 图中的浅灰对角线可视作近似的 **随机/均匀选择基线**：
  - 若下一层 expert 与当前层 expert 无关，选择 top-K% expert 应大致只获得 K% 概率质量。
  - 四条曲线均显著高于该基线，证明跨层存在可利用的 temporal locality。
  - **Llama 4 与随机基线的差距最大**，表示其 expert routing 具有最强的结构化规律。

- 从系统设计角度，该图直接支持论文的 **Insight 2: Cross-hierarchy memory management**：
  - 当前 MoE layer 完成 gating 后，可根据 \(P(e_j \mid e_i)\) 预测相邻下一层可能使用的 experts。
  - 对预测概率最高的一小部分 experts，可提前执行 **prefetching**，或放入更快的缓存层级，如 LLC、local HBM。
  - 因为相邻 layer 的复用距离很短，跨层预测尤其适合管理容量较小、延迟较低的存储层，而非只依赖传统 LRU 等被动缓存策略。

- 该图也揭示模型间优化收益的差异：
  - **Llama 4** 更适合 aggressive prefetch：少量候选即可获得高覆盖率，因此预测错误率较低，预取性价比高。
  - **Qwen3** 同样适合基于跨层关联的缓存与迁移，但预取范围可能需要略大于 Llama 4。
  - **DeepSeek V3** 的跨层预测不应过度激进；若盲目预取过多 experts，可能挤占缓存空间并引入无效数据搬运。更适合采用概率阈值、带宽感知或与 token-level predictor 联合的策略。
  - **Kimi K2** 介于两者之间，可通过模型层级、batch size 与缓存容量自适应设定 top-K。

- 需要注意的限制是：
  - 图反映的是平均条件分布，**不同 layer 的相关性可能不同**；论文的 heatmap 已指出低层和高层、不同模型层之间存在明显差异。
  - CDF 高并不等于单个 expert 的精确预测准确率高；它衡量的是“候选集合覆盖概率”，因此更适合 **top-K prefetch / cache candidate selection**，而非只预测唯一一个 expert。
  - 实际部署还需权衡预取带来的收益与代价，包括 remote-memory bandwidth、cache capacity、expert weight size、D2D congestion，以及错误预取造成的缓存污染。

- 总体而言，这张图的核心信息是：**看似随机的 MoE expert routing 在相邻层之间具有显著可预测性，而且这一可预测性高度依赖模型架构。** 这为基于 expert affinity 的跨层缓存、预取、数据迁移和本地化 placement 提供了实证依据。

### 8c938f5ac86d9342c3516959c39ae3acae5029b1f81c13e51cf6066787fe7a4a.jpg

![8c938f5ac86d9342c3516959c39ae3acae5029b1f81c13e51cf6066787fe7a4a.jpg](images/8c938f5ac86d9342c3516959c39ae3acae5029b1f81c13e51cf6066787fe7a4a.jpg)

- **图片定位**：该图对应论文 Figure 6，核心验证 **Prefill 阶段与 Decode 阶段的 expert selection pattern 高度一致**。对象包括 Qwen3 的两类局部热图，以及四个 MoE 模型跨层统计的 Spearman’s Ratio。

| 子图 | 分析对象 | 阶段/模型 | 主要视觉证据 | 结论 |
|---|---|---|---|---|
| (a) | Cross-layer heatmap | Qwen3，Prefill，Layer 3→4 | 红色高亮点、纵向亮条纹及局部热点 | 相邻层 expert 选择存在条件相关性 |
| (b) | Cross-layer heatmap | Qwen3，Decode，Layer 3→4 | 与 (a) 的热点位置、密集区域高度接近 | Decode 保留了 Prefill 的跨层路由结构 |
| (c) | Cross-token heatmap | Qwen3，Prefill，Layer 13 | 蓝色对角线、竖直亮带明显 | 相邻 token 倾向复用相同或相关 expert |
| (d) | Cross-token heatmap | Qwen3，Decode，Layer 13 | 与 (c) 的对角线和亮带近似重合 | Decode 的 token 级 temporal locality 可由 Prefill 预测 |
| (e) | Cross-layer similarity | DS、Llama、Qwen、Kimi | 各层 Spearman’s Ratio 多数较高 | 跨层相关结构跨阶段稳定 |
| (f) | Cross-token similarity | DS、Llama、Qwen、Kimi | 各层相关性整体较高，且 Kimi/Qwen 更稳定 | token 级路由模式跨阶段可迁移 |

- **热图编码含义**：
  - 横轴和纵轴均为 **Expert ID**，范围为 0–128，说明图中分析的是 Qwen3 某个含 128 个 experts 的 MoE layer。
  - 色条范围是 **0–0.8**；越亮表示某一 expert 对的条件共激活概率越高。
  - (a)(b) 使用红色系，表达 **cross-layer**：已知 Layer 3 选择某 expert 后，Layer 4 选择各 expert 的条件分布。
  - (c)(d) 使用蓝色系，表达 **cross-token**：已知当前 token 在 Layer 13 的 expert 后，下一个 token 选择各 expert 的条件分布。
  - 红色/蓝色虚线框及右侧放大图用于展示原热图中局部区域；其目的不是强调单个特定 expert，而是让热点结构在 Prefill 与 Decode 间可直接对照。

- **(a) 与 (b)：Qwen3 的 Cross-layer pattern 基本保持不变**：
  - 两张图均展示 **Layer 3 and Layer 4** 的 expert-pair conditional heatmap。
  - 两者都可见稀疏但清晰的高概率亮点，表明某些 “上一层 expert → 下一层 expert” 的组合远比其他组合更常出现。
  - 放大区域中，Prefill 与 Decode 都出现相近位置的横向亮块、点状热点与局部条纹；这意味着 Decode 并不是重新随机地产生 expert pairing，而是在延续输入上下文已经形成的路由偏好。
  - 图中的若干纵向亮带表示：**部分下一层 experts 具有较高的全局受欢迎度**，即它们不完全依赖于前一层选择哪个 expert，仍会频繁被激活。
  - 该现象支持在 Decode 开始前，从 Prefill trace 预估后续需要的 expert；尤其适合指导 expert prefetch、HBM/DRAM cache warm-up 与初始 expert placement。

- **(c) 与 (d)：Qwen3 的 Cross-token pattern 同样跨阶段稳定**：
  - 该组聚焦 Qwen3 **Layer 13** 中相邻 token 的 expert 路由关系。
  - 最显著的结构是从左上到右下的亮色对角线，即 **same-expert persistence**：当前 token 激活 expert \(i\) 后，下一个 token 再次激活 expert \(i\) 的概率明显较高。
  - 对角线之外还存在竖直亮带，说明除“复用原 expert”外，一些 experts 对不同前序选择也持续具有吸引力，属于稳定的 hot experts。
  - Decode 图 (d) 继承了 Prefill 图 (c) 的对角线、亮带和局部热点布局。因此，Prefill 阶段不仅能预测 Decode 的总体 expert frequency，也能预测 **相邻 token 的 transition pattern**。
  - 这一规律有直接的存储层次意义：下一 token 的 expert weights 可在当前 token 执行期间提前迁入本地 HBM、DRAM 或 LLC，以降低 remote-memory/D2D access。

- **(e)：Cross-layer heatmap similarity 的定量解读**：
  - 纵轴为 **Spearman’s Ratio**，衡量同一 layer pair 在 Prefill 与 Decode 两阶段热图排序结构的单调相关性；越接近 1，两个阶段越相似。
  - 横轴为 **Layer ID**，覆盖约 90 个 layer，反映该结论不是单个 layer 的偶发现象。
  - **Kimi（蓝灰色）**总体最高，约集中于 **0.75–0.85**，且随 layer 变化较平稳，显示最强的跨阶段结构稳定性。
  - **Qwen（绿色）**通常约为 **0.70–0.80**，后部 layers 可接近或超过 0.8；这是图中 Qwen3 局部热图高度相似的统计支撑。
  - **Llama（红色）**大致落在 **0.55–0.70**，存在一定波动，属于中等到较强的一致性。
  - **DeepSeek / DS（橙色）**相对较低，约 **0.43–0.62**，中间 layers 有明显下探；即使如此仍显示非随机关联，但预测策略应更保守。
  - 模型间差异说明：**Prefill-guided prediction 是普适方向，但预测阈值、cache 容量和复制策略必须 model-aware，而不能采用固定参数。**

- **(f)：Cross-token heatmap similarity 的定量解读**：
  - 与 (e) 相比，Cross-token 的 Spearman’s Ratio 呈现类似的模型排序：**Kimi 最高，Qwen 次之，Llama 居中，DS 最低**。
  - Kimi 大多位于 **0.70–0.85**，Qwen 多在 **0.67–0.78**；这意味着其 Decode token-level routing 可以较可靠地由 Prefill 建模。
  - DS 在多个 layers 接近 **0.4–0.55**，而不是稳定超过 0.7，说明其 token-to-token expert transition 的阶段迁移性弱于 Qwen/Kimi。
  - 相关性并非所有 layers 都单调或恒定：早期、中期、后期 layers 存在波动。这表明实际系统应采用 **per-layer prediction table**，而非为整个模型建立单一全局 expert popularity ranking。

- **该图支撑的论文核心观察是 Ob3**：
  - **Prefill-to-Decode-Level Correlation**：Prefill 和 Decode 的 expert pair distribution、expert transition distribution 以及总体路由排序结构高度相似。
  - 这直接导出论文的 **Insight 1: Prefill-data-driven prediction**。
  - 在 Decode 初始阶段，历史 decode token 很少，传统在线负载均衡器尚无法积累足够统计信息；而 Prefill 已经一次性处理完整 prompt，可立即提供有价值的 routing prior。

- **面向系统设计的具体含义**：

| 机制 | 图中依据 | 可采取的动作 | 预期收益 |
|---|---|---|---|
| Decode 初始 expert placement | (a)(b)(e) 的 cross-layer 一致性 | 用 Prefill expert frequency 重排或复制 hot experts | 缓解初始 decode workload imbalance |
| Next-token prefetch | (c)(d)(f) 的对角线与 transition 稳定性 | 当前 token 执行时预取下 token 高概率 experts | 隐藏远程权重访问延迟 |
| Local HBM/DRAM caching | Prefill/Decode 热点重叠 | 将 Prefill 判定的热门 remote experts 缓存至本地 | 降低 D2D、CXL 或 CPU-GPU 数据移动 |
| Per-layer resource policy | (e)(f) 中 layer 间相关性有波动 | 高相关 layer 激进预取；低相关 layer 保守缓存 | 避免错误预测挤占有限容量 |
| Model-aware tuning | DS、Llama、Qwen、Kimi 的相关性不同 | 按模型设置 top-k、复制数、置信度阈值 | 提升策略鲁棒性 |

- **与论文后续 case study 的联系**：
  - 图 6 是 Section VI 中 **Prefill-Guided Expert Placement** 的关键实证基础。
  - 论文据此使用 Prefill traces 估计每层 expert frequency，并提出 **Remap-based placement** 与 **Duplication-based placement**。
  - 在 8×H100、Qwen3-235B 上，二者相对 Default 分别取得 **15.5%** 和 **12.5%** 的 MoE computation speedup；该效果表明图中观察到的“相似性”具有实际调度价值，而非仅是统计现象。

- **图的边界与应注意的问题**：
  - Spearman’s Ratio 衡量的是**排序/单调关系**，并不保证实际激活概率完全相同；因此，Prefill 适合作为 prior，而不应被当作 Decode 的精确路由结果。
  - 图中仅展示 Qwen3 的局部热图，跨模型普适性主要由右侧散点图支撑；不同模型的稳定程度差异明显。
  - 低频 experts 的绝对概率较小但可能变化较大，系统应优先预测和缓存高频 expert，而不是追求覆盖所有 experts。
  - 对于 DS 这类跨阶段相关性相对较弱的模型，过度复制可能造成额外内存占用和迁移成本；应以置信度、访问代价和 cache budget 联合决策。

### ed9820887ef91e9f685915cbd4fa2e2d05f3532708fe09a789d9bd3a26bc4ca0.jpg

![ed9820887ef91e9f685915cbd4fa2e2d05f3532708fe09a789d9bd3a26bc4ca0.jpg](images/ed9820887ef91e9f685915cbd4fa2e2d05f3532708fe09a789d9bd3a26bc4ca0.jpg)

- 该图展示 **Cross-token expert correlation**：在同一 MoE layer 中，给定前一 token 激活的 expert，下一 token 激活各 expert 的条件概率分布。图中包含：
  
  | 子图 | 模型 | Expert 数 | Layer |
  |---|---:|---:|---|
  | (a) | DeepSeek-V3 | 256 | 3、17、43 |
  | (b) | Llama 4 | 128 | 1、17、43 |
  | (c) | Qwen3 | 128 | 1、17、43 |

- 坐标与颜色含义如下：

  | 视觉元素 | 含义 |
  |---|---|
  | 横轴 `Next Token Expert ID` | 下一 token 被路由到的 expert 编号 |
  | 纵轴 `Previous Token Expert ID` | 前一 token 被路由到的 expert 编号 |
  | 单元格亮度 | 条件激活概率，近似表示 \(P(e_{t+1}=j\mid e_t=i)\) |
  | 深蓝色 | 条件概率接近 0 |
  | 浅蓝/白色 | 条件概率较高；右侧色条范围约为 0–0.8 |
  | 主对角线 | 同一 expert 在相邻 token 间持续被选择，即 \(e_t=e_{t+1}\) |
  | 竖直亮线/亮点 | 某些 next-token experts 具有全局热度，较少依赖前一 token 的 expert |
  | 红框和放大图 | 对 Layer 43 局部细粒度路由结构的强调 |

- **最核心结论是：Expert routing 并非随机；相邻 token 的 expert selection 具有显著条件相关性，且该相关性在高层更强。**

- 从层深角度看，三个模型都呈现出相似演化：

  | 层位置 | 可见模式 | 含义 |
  |---|---|---|
  | 低层，如 DeepSeek Layer 3、Llama/Qwen Layer 1 | 热图整体较暗、亮点稀疏、对角线弱或不连续 | 相邻 token 在底层的 expert 路由相对分散，前一 token 的预测价值有限 |
  | 中层，如 Layer 17 | 出现更明显的斜向结构、局部亮点与热 expert | expert choice 开始体现 token 连续性与语义上下文延续 |
  | 高层，如 Layer 43 | 对角线显著增强，局部出现连续亮带或阶梯状路径 | 高层 token representation 更稳定；同一或相邻 expert 在连续 token 中被重复使用的概率较高 |

- DeepSeek-V3 的图像特征最明显地表现为 **高层的 expert persistence**：
  
  | Layer | 观察 | 解读 |
  |---|---|---|
  | Layer 3 | 几乎全深蓝，仅有少量离散亮点 | 底层选择较弱相关 |
  | Layer 17 | 可见较淡、分段式的对角趋势 | 连续 token 已产生一定 expert reuse |
  | Layer 43 | 主对角线明显；放大区显示连续斜向亮带及局部热点 | 前一 token 选择的 expert 对下一 token 有较强预测能力 |
  
  - DeepSeek 使用 **256 experts**，组合空间更大，因此热图总体比 Llama 更稀疏。
  - 即使 expert 空间很大，Layer 43 仍显现清晰对角结构，说明这种时间局部性不是简单由 expert 数少造成的。

- Llama 4 的相关性最集中、最清晰：
  
  | Layer | 观察 | 解读 |
  |---|---|---|
  | Layer 1 | 亮点有限 | 底层缺乏强 token-to-token routing 规律 |
  | Layer 17 | 出现跨较大 expert-ID 范围的较清晰对角线 | 相邻 token 常沿相同或邻近的 expert 路径演化 |
  | Layer 43 | 对角线最亮、最连续；放大图呈明显阶梯状/块状轨迹 | 具有较强的条件可预测性和稳定的高层 expert reuse |
  
  - 论文的整体统计也支持这一视觉结论：Llama 4 的 top 20% next-token expert candidates 覆盖约 **80%** 条件概率质量，是四个模型中最强的 token-level correlation。
  - 这意味着对 Llama 4，基于上一 token 的轻量级预测器具有更高命中潜力。

- Qwen3 的图像呈现 **对角相关性与全局热门 expert 并存** 的模式：
  
  | Layer | 观察 | 解读 |
  |---|---|---|
  | Layer 1 | 已存在较多纵向亮线和点状活跃区域 | 一部分 experts 对不同前序状态都较热门 |
  | Layer 17 | 对角线增强，同时纵向亮带密集 | 同时具有局部 temporal dependence 与全局 popularity skew |
  | Layer 43 | 放大图中可见强对角线、亮块和多条纵向热点 | routing 受前序 expert 影响，但也有少量持续热门的 experts |
  
  - Qwen3 的热图比 DeepSeek 更“密”，表明其 routing probability 分布更集中。
  - **对角线**适合做 next-token reuse prediction；**竖线**则适合做 hot-expert replication 或长期缓存。

- 红色放大框尤其说明：高层的对角线不是由单一模糊趋势构成，而是由若干 **局部稳定的 expert-transition path** 组成。
  
  - 在 Llama 4 和 Qwen3 的放大区中，亮度沿对角方向呈离散块或阶梯状分布。
  - 这表示“前一 token 选择 expert \(i\)”不会均匀地导向任意 next-token expert，而是倾向于导向少数特定 expert，尤其包括 \(i\) 自身或其邻近/关联 expert。
  - 这种局部结构正是静态随机模型无法捕捉、但预测式 cache/prefetch policy 可以利用的信息。

- 该图支持论文的 **Observation 2 / Insight 2**：
  
  | 论文洞见 | 图中证据 | 系统含义 |
  |---|---|---|
  | Token-level correlation | 高层明显主对角线、局部亮块、稳定热点 | 当前 token 的 expert trace 可预测下一 token 的候选 experts |
  | Cross-hierarchy memory management | token-level correlation 的 reuse distance 较长 | 更适合用于 local DRAM、HBM、CXL memory 或 SSD tier 的 expert caching/prefetching |
  | Popular-expert management | Qwen3 等图中的竖向热点 | 应复制、预取或常驻高频 expert |
  | Layer-aware policy | 底层与高层相关性显著不同 | predictor 不应对所有 layers 使用同一阈值或同一 cache budget |

- 对 serving system 的直接设计启示包括：

  | 优化策略 | 使用的图中模式 | 预期收益 |
  |---|---|---|
  | Next-token expert prefetch | 对角线与局部转移亮块 | 在下一 decode step 前预取高概率 expert weights |
  | Local HBM/DRAM caching | 跨 token 的重复 expert activation | 将刚访问或预测将访问的 remote experts 缓存到 local memory |
  | Layer-specific predictor | 高层强、低层弱的相关性差异 | 避免在低层进行低命中率预取，降低无效搬运 |
  | Hot expert replication | 竖直亮线、亮度集中的列 | 减少热门 expert 导致的远程访存和单元拥塞 |
  | Candidate-set routing | 每行仅少数亮点 | 将预测范围缩小至 top-\(k\) next experts，控制 predictor 和缓存开销 |

- 图也揭示了一个重要权衡：**不能把对角线解释为“永远只预取同一个 expert”。**
  
  - 高层虽有明显 diagonal reuse，但放大图仍有非对角亮块，说明下一 token 也可能转移到相关 expert。
  - 因而合理策略应是：以当前激活 expert 为索引，从 transition heatmap 中选择 **top-\(n\) conditional candidates**，而不是只缓存当前 expert。
  - 这正对应论文在 wafer-scale case study 中的 `Data-Driven Predictor`：查询当前 expert 对应的 heatmap row，并选择高概率 next-token experts 进行本地副本管理。

- 图的局限性也应明确：
  
  - 热图说明的是 **统计相关性**，并不保证单个请求、单个 token 的 routing 可被完全准确预测。
  - 不同 layer、模型、任务类型和语言会改变热点结构；因此 transition table 应按 **model、layer，必要时按 workload category** 维护。
  - 缓存或复制 expert 会占用 HBM/DRAM 容量；只有当预测概率、expert weight 大小、跨 die 访问代价和 reuse 次数共同满足阈值时，预取才有净收益。
  - 对于低层热图，相关性较弱，激进预取可能引入额外 D2D traffic 和 cache pollution。

- 总结而言，该图以可视化证据证明了：**MoE 的相邻 token expert routing 具有结构化 temporal locality；这种 locality 在高层尤其显著，并同时包含“同 expert 重用”“特定 expert 转移”与“全局热门 expert”三类模式。** 这为 decode-stage 的 expert prefetch、缓存、迁移与副本放置提供了直接的数据基础。

### 4eed7c6a04629f0a4e90bea2ea1e0d46283e4a1b3a4dbd9e877e889abbc23dc0.jpg

![4eed7c6a04629f0a4e90bea2ea1e0d46283e4a1b3a4dbd9e877e889abbc23dc0.jpg](images/4eed7c6a04629f0a4e90bea2ea1e0d46283e4a1b3a4dbd9e877e889abbc23dc0.jpg)

- 图片展示 **Cross-Token expert correlation（跨 Token 专家选择相关性）** 的条件累积分布函数（Conditional CDF）。
  - 横轴：**Top-K Experts（% of Pool）**，即为预测下一个 token 所激活的 expert，保留候选 expert 池中排名前 K 的比例。
  - 纵轴：**Cumulative \(P(e_j \mid e_i)\)**，即已知当前 token 激活 expert \(e_i\) 后，下一个 token 选择候选 expert \(e_j\) 的累计条件概率。
  - 曲线越靠近左上角，意味着只需保留更少的候选 expert，便可覆盖更多下一个 token 的路由概率；因此其**跨 token 可预测性更强**。

| 模型 | 曲线颜色 | Top 20% 候选 expert 覆盖的条件概率质量 | 跨 Token 相关性 | 系统含义 |
|---|---:|---:|---|---|
| Llama 4 | 蓝色 | **80%** | 最强 | 最适合基于上一 token 做 expert prefetch / cache |
| Qwen3 | 绿色 | **62%** | 较强 | 可用较小候选集获得有效预测 |
| Kimi K2 | 橙色 | **53%** | 中等 | 存在可利用的局部性，但预测收益较有限 |
| DeepSeek V3 | 红色 | **47%** | 四者中最弱 | 仍非完全随机，但需要更宽的候选集合 |

- 核心观察是：**四条曲线均明显高于随机选择基线**。
  - 若下一个 token 的 expert 选择完全随机，则累计概率应近似沿灰色对角线增长：保留 \(x\%\) expert 只能覆盖约 \(x\%\) 概率。
  - 图中所有模型的曲线都向左上方凸起，说明当前 token 的 expert 选择对下一 token 有信息量。
  - 因此，MoE routing 虽然在执行时由 gate 动态决定，但统计上并非不可预测的“完全随机”过程。

- 在 **20% expert pool** 这一关键工作点，模型间差异尤其显著：
  - **Llama 4：80%**。仅预取或缓存 20% 的候选 expert，就可命中约 80% 的下一 token 路由概率质量，压缩比约为 **4:1**。
  - **Qwen3：62%**。20% 候选覆盖接近三分之二概率，说明轻量预测器具有实际价值。
  - **Kimi K2：53%** 与 **DeepSeek V3：47%**。虽然弱于前两者，但仍远优于随机基线的 20%，意味着预测策略仍可能减少远端访存或跨设备数据移动。

- 图中曲线的相对位置表达了明确的模型依赖性：
  - **Llama 4 始终最陡、最靠左上**：其 next-token routing 的条件分布最集中，专家复用和连续性最强。
  - **Qwen3 位于中上部**：具有稳定而可观的预测空间。
  - **Kimi K2 与 DeepSeek V3 更接近**随机线：其路由分布相对更分散，不能只缓存极少数 expert，否则覆盖率不足。
  - 当候选池比例逐步扩大至约 **50%–70%** 时，所有模型都能覆盖大部分概率质量；这表明即便较难预测的模型，也可通过“有限宽度候选集”而非全量 expert 实现高命中率。

- 该图支撑论文的 **Observation 2（Ob2）** 与 **Insight 2**：
  - **Ob2：相邻 token 的 expert selection 存在显著相关性。**
  - 在较高层 Transformer layer 中，论文观察到 co-activation heatmap 的亮对角线更明显，即 expert 往往会在相邻 token 上被重复选择。
  - 这意味着当前 token 已访问或激活的 expert，是预测下一 token 所需 expert 的重要信号。

- 对 serving system 的直接设计启示如下：

| 优化机制 | 可利用的图中信息 | 预期收益 |
|---|---|---|
| Expert prefetching | 依据当前 token 的 active experts 预测下一 token Top-K experts | 提前搬运权重，隐藏远端访问延迟 |
| Local HBM caching | 将高条件概率的 next-token experts 缓存在 local DRAM/HBM | 降低 remote HBM read 与 D2D traffic |
| LLC management | 对短 reuse-distance 的候选 expert 放入 LLC | 利用更快缓存服务相邻层或紧邻 token 请求 |
| Multi-tier memory placement | 根据预测置信度决定 expert 位于 LLC、local HBM、remote memory 或 CXL memory | 在容量与命中率间平衡 |
| Adaptive prediction width | Llama 4 使用较小 K；DeepSeek V3 使用较大 K | 避免“一刀切”缓存策略带来的空间浪费或低命中率 |

- 对 wafer-scale GPU 案例而言，该图是其 **Data-Driven Predictor** 的统计基础：
  - Global Command Processor 根据当前 MoE kernel 中的 expert selection，从 cross-token heatmap 中取对应行。
  - 对每个已激活 expert，提取其下一个 token 的 **Top-n** 条件候选 expert。
  - Prediction Unit（PDU）据此控制 remote expert 是否复制到 **local HBM**。
  - 由于图中 CDF 明确表明 Top-K 候选可以覆盖远高于随机水平的概率质量，这种硬件预测与缓存并非盲目复制，而是由模型路由规律驱动。

- 需要正确理解该图的边界：
  - 它说明的是**概率质量集中性**，不是“每个下一 token 都能被完全准确预测”。
  - CDF 曲线来自大量请求、层和 token 的统计平均；不同 layer、任务类型、语言和 batch composition 下，相关性强度会变化。
  - 对 DeepSeek V3 这类曲线较平缓的模型，若 Top-K 设置太小，可能导致低缓存命中率；预测系统应采用**按模型、按层、甚至按任务自适应的 K 值**。
  - 高预测覆盖率不自动等价于等比例端到端加速，因为实际收益还受 HBM 带宽、D2D 拓扑、缓存容量、复制开销和 workload imbalance 共同限制。

- 总结而言，该图最重要的结论是：**MoE 的相邻 token expert routing 具有显著但模型相关的时间局部性。**  
  - **Llama 4** 最适合激进的 next-token expert prediction；
  - **Qwen3** 适合中等规模的预测缓存；
  - **Kimi K2 与 DeepSeek V3** 仍可受益，但应使用更宽候选集合或结合其他信号；
  - 这为论文提出的跨层级 memory management、expert prefetching 与 local-HBM duplication 提供了直接实证依据。

### 320301bd96dce6d431e6171072ab10989157715ea2d405c891f4635043ef93dc.jpg

![320301bd96dce6d431e6171072ab10989157715ea2d405c891f4635043ef93dc.jpg](images/320301bd96dce6d431e6171072ab10989157715ea2d405c891f4635043ef93dc.jpg)

- 该图对应 Figure 6(e)，展示四个 MoE 模型在 **Prefill 与 Decode 阶段的 cross-layer expert co-activation heatmap 相似度**。纵轴是 **Spearman’s Ratio（ρ）**，横轴是 **Layer ID**；每个点代表一个 MoE 层。

| 图例模型 | 颜色 | ρ 的大致范围 | 主要观察 |
|---|---:|---:|---|
| DeepSeek-V3（DS） | 橙色 | **0.43–0.67** | 整体为中等到较强相关；中间层出现明显低谷。 |
| Llama4 | 红色 | **0.62–0.76** | 相对稳定，绝大多数层保持较强正相关。 |
| Qwen3 | 绿色 | **0.72–0.90** | 四者中相关性最强；高层接近或超过 0.85。 |
| Kimi K2 | 蓝色 | **0.59–0.70** | 较稳定的强/中强相关，波动小于 DS。 |

- **ρ 的含义**：Spearman correlation 衡量两个 heatmap 中 expert-pair activation pattern 的秩一致性，而非绝对数值是否完全相同。
  - **ρ > 0.7**：通常可视为强相关；
  - **0.4 < ρ ≤ 0.7**：中等相关；
  - 图中全部数据均显著大于 0.4，说明 Prefill 和 Decode 的跨层 expert 路由结构并非独立或随机。

- **最关键结论是跨阶段模式稳定性强**：
  - Qwen3 大部分层位于 **0.8 左右甚至更高**，表明其 Prefill 期间观察到的“某 expert 被激活后，下一层哪些 expert 更可能共同激活”的结构，能够高度迁移到 Decode。
  - Llama4 和 Kimi K2 虽低于 Qwen3，但大部分层仍接近或达到强相关阈值。
  - DeepSeek-V3 的相关性最弱且波动最大，但最低点仍约为 **0.43–0.45**，仍保留可利用的统计预测信号。

- **层间差异说明 expert routing 的可预测性具有模型依赖性**：
  - **Qwen3** 的绿色点从前层到约 Layer 90 整体上升，后部层的 Prefill–Decode 一致性尤其突出。这意味着其高层语义生成阶段可能呈现更稳定的 expert routing regime。
  - **DeepSeek-V3** 在中部 Layer ID 附近下降到约 0.45，表明部分层的 Decode 路由相对更依赖生成历史或当前 token 状态，不能使用单一、静态的预测置信度。
  - **Llama4/Kimi K2** 基本围绕较窄区间波动，说明它们更适合采用统一的 Prefill-guided placement 或缓存策略。

- 图中横轴最大到约 **90**，不同颜色点覆盖长度不同，反映模型的 MoE 层数量不同；因此，横向位置不能简单理解为四个模型完全相同的网络深度，而应理解为各模型自身的 layer index。

- 该图直接支撑论文的 **Insight 1: Prefill-data-driven prediction**：
  - 在 Decode 开始时，系统尚未积累足够的 decode token routing history；
  - 但 Prefill 已处理完整 prompt，能够得到每层的 expert selection trace 和 expert-pair co-activation heatmap；
  - 由于两阶段 heatmap 的 ρ 普遍较高，系统可用 Prefill trace 预测 Decode 初期的热点 experts、专家对以及潜在的跨设备访问。

- 从系统设计角度，这一结果可用于以下优化：

| 优化动作 | 图中依据 | 预期收益 |
|---|---|---|
| **Prefill-guided expert placement** | Prefill 与 Decode heatmap 高相关 | 在 Decode 前重映射或复制热点 experts，缓解 GPU 间负载不均。 |
| **Expert prefetching** | 下一阶段可能激活的 expert-pair 可由 Prefill 预估 | 提前把远端 expert weights 拉取到 local HBM/DRAM。 |
| **Local HBM caching** | 高 ρ 表示热点关系可延续 | 优先缓存 Prefill 中频繁关联的 experts，减少 D2D/NVLink/CXL 通信。 |
| **分层预测策略** | DS 波动较大、Qwen 更稳定 | 针对不同模型、不同层设置不同预测阈值和缓存配额。 |

- 该图也解释了为什么论文在真实 **8×H100** 集群上可实现 Prefill-guided placement：
  - 对 Qwen3-235B，Prefill information 是 Decode routing 的有效代理；
  - 因而即使没有 oracle decode trace，也可通过 Remap-based 或 Duplication-based placement 改善 Decode 阶段的 expert compute balance；
  - 论文报告该方法相对 Default placement 获得 **12.5%–15.5%** 的 MoE computation speedup，并距使用 oracle Decode 信息得到的 Best placement 不超过约 **10%**。

- 需要注意，图中高相关不代表每一个 token 都会路由到相同 experts，也不意味着 Prefill trace 可以无误预测 Decode。它证明的是 **聚合后的 expert-pair 排序和结构分布具有稳定性**。因此，合理策略应是“基于置信度的预测、缓存和复制”，而不是完全依赖确定性预取。

- 总体而言，该图揭示了一个核心事实：MoE expert selection 表面上由 gate 动态决定、似乎混乱，但在 **Prefill→Decode 的跨阶段尺度上存在稳定的统计规律**。其中 **Qwen3 的规律最强、DeepSeek-V3 的异质性最大**；这种差异要求 serving system 采用模型感知、层感知的动态数据移动策略。

### 9f7741f5ef410cfc84a1ab39d0a5fee5336e63246b043ec9c07d28a50d3c32e6.jpg

![9f7741f5ef410cfc84a1ab39d0a5fee5336e63246b043ec9c07d28a50d3c32e6.jpg](images/9f7741f5ef410cfc84a1ab39d0a5fee5336e63246b043ec9c07d28a50d3c32e6.jpg)

- 该图为 Figure 6(e) 的子图，展示四个 MoE 模型中，各层 **prefill 与 decode 阶段 cross-layer expert-activation heatmap 的 Spearman’s Ratio（ρ）**。

| 视觉元素 | 含义 |
|---|---|
| 横轴 `Layer ID` | 模型层编号，范围约为 0–90 |
| 纵轴 `Spearman's Ratio` | prefill 与 decode 路由模式的秩相关性，范围约 0.2–1.0 |
| 蓝色 `DS` | DeepSeek-V3 |
| 红色 `Llama` | Llama4 Maverick |
| 绿色 `Qwen` | Qwen3-235B |
| 灰色 `Kimi` | Kimi K2 |
| 每个散点 | 一个模型层中，两阶段 expert-pair 激活模式的相关程度 |

- 图的核心结论是：**绝大多数层的 ρ 位于 0.6–0.9，且大量点高于 0.7**。按论文采用的解释标准，`|ρ| > 0.7` 表示强相关，因此 prefill 阶段观察到的 expert routing pattern 通常能够较可靠地迁移到 decode 阶段。

- 各模型的趋势具有明显差异：

| 模型 | 图中趋势 | 解释 |
|---|---|---|
| **Qwen3** | 整体相关性较高；前部层约为 0.7–0.85，中段略降，后部层逐渐升至约 0.85–0.9 | **prefill-decode 一致性最稳定、后层预测价值尤其强** |
| **Llama4** | 多数点集中在约 0.65–0.8，波动相对温和 | 跨阶段相关性较强，但弱于 Qwen3 的后层峰值 |
| **DeepSeek-V3** | 呈现较明显的“先降后升”：低层约 0.7，约第 25–45 层出现 0.4–0.6 的低谷，后续回升 | 中间层 routing behavior 的阶段漂移更明显，不能对所有层采用同一激进预测策略 |
| **Kimi K2** | 点主要处于中高相关区间，整体与其他模型一致 | 说明跨阶段一致性不是单一模型的偶然现象，而是大规模 MoE 的普遍特征 |

- 图中最值得注意的结构性现象是 **layer heterogeneity**：
  - 同一个模型的不同层相关性并不恒定。
  - DeepSeek-V3 中部层的相关性最低，约可接近 **0.4**，属于中等相关区间。
  - Qwen3 后部层达到接近 **0.9**，说明其 decode routing 很大程度上可由 prefill routing 预估。
  - 因此，预测器不应只使用“全模型统一阈值”，而应维护 **per-layer confidence** 或 **per-layer top-k prediction budget**。

- 从系统设计角度，该图直接支撑论文的 **Insight 1: Prefill-data-driven prediction**：
  - 在 decode 刚开始时，历史 decode token 很少，在线统计尚不足以稳定识别 hot experts。
  - prefill 一次性处理完整 prompt，能够快速积累足够的 expert selection trace。
  - 若 prefill 与 decode 的 heatmap 保持高 Spearman correlation，则系统可在 decode 开始前预先：
    - **迁移或复制可能成为热点的 experts**；
    - **预热 local HBM / DRAM / LLC cache**；
    - **调整 expert-to-GPU placement**；
    - **预分配计算资源，规避热门 expert 所在 GPU 成为 straggler**。

- 图所表达的并不是“decode expert ID 可以被逐 token 精确预测”，而是更稳健的统计结论：**expert-pair 的相对激活排序在 prefill 和 decode 之间保持相似**。这意味着它特别适合做：
  - 热门 expert 集合预测；
  - top-k expert candidate 预取；
  - 负载均衡导向的复制与放置；
  - 概率性缓存，而非要求零误差的 exact routing prediction。

- 该结果也隐含了一个重要边界条件：
  - 对于 ρ 较低的层，尤其是 DeepSeek-V3 的中部层，基于 prefill 的 placement 可能失配。
  - 实际系统应采用 **confidence-aware 策略**：高相关层积极预取、复制和重映射；中等相关层保留更多 cache capacity 或采用在线 decode trace 逐步修正。
  - 这比静态地把所有 prefill-hot experts 全量复制更节省 HBM 容量，也能降低无效的数据复制流量。

- 结合论文后续真实集群实验，这一图的统计发现被转化为 `Remap-based` 与 `Duplication-based` 两类 **prefill-guided expert placement**。在 8×H100 上，它们相对默认 placement 分别实现约 **15.5%** 与 **12.5%** 的 MoE computation 加速，并且与使用 oracle decode trace 的最佳 placement 相差不足约 10%。这说明图中的相关性具有实际可操作价值，而非仅是离线统计现象。

### 1e977b934a47e2a372ef999d341cc55f626f2f80496675a3f75b6c6f440d55b2.jpg

![1e977b934a47e2a372ef999d341cc55f626f2f80496675a3f75b6c6f440d55b2.jpg](images/1e977b934a47e2a372ef999d341cc55f626f2f80496675a3f75b6c6f440d55b2.jpg)

- 该图验证论文的核心时间相关性结论：**Prefill 阶段的 expert selection 能够有效预测 Decode 阶段的 expert 热度分布**。图中以 Qwen3 为具体案例，并扩展到 DeepSeek（DS）、Llama、Kimi 等模型。

- 图由三个子图构成，分别从**单层频率分布**、**Top-K 热门专家集合重叠**和**跨层统计相关性**三个角度证明 Prefill–Decode similarity。

| 子图 | 分析对象 | 主要指标 | 核心结论 |
|---|---|---|---|
| (a) | Qwen3 Layer 66 | 每个 expert 的激活频率 | Prefill 与 Decode 的总体频率形状高度一致，且均表现出明显长尾与热点偏斜 |
| (b) | Qwen3 全部 MoE layers | Prefill/Decode 的 Top-K expert overlap | Prefill 的热门 experts 对 Decode 热门 experts 具有较强覆盖能力 |
| (c) | DS、Llama、Qwen、Kimi 各层 | Spearman’s Ratio | 大多数层的相关性达到或接近强相关区间，规律跨模型成立 |

- 子图 (a) 展示 Qwen3 的 **Layer 66 Prefill-Decode Expert Selection Frequency**：
  - 横轴是 expert ID，纵轴是每个 expert 的选择频率；蓝色柱表示 **Prefill**，橙色柱表示 **Decode**。
  - experts 大体按照激活频率由高到低排列，左侧少数 experts 被高频选择，右侧大量 experts 使用频率低，形成显著的**长尾分布（long-tail distribution）**。
  - Prefill 与 Decode 的柱状轮廓高度重合：高频 experts 在两个阶段通常都保持高频，低频 experts 也大多保持低频。
  - Decode 频率存在一些更尖锐的局部峰值，说明自回归生成出的 token 会带来短时波动；但这类波动没有改变总体热点排序结构。
  - 该子图揭示两个重要事实：
    - **MoE routing 并非均匀随机**：不同 experts 的负载差异很大。
    - **阶段切换不会破坏热点结构**：Prefill 已经包含对后续 Decode workload 的有效先验。

- 子图 (b) 展示 Qwen3 中 Prefill（PF）与 Decode（DC）热门 experts 的集合重叠率，纵轴为 Overlap Rate，横轴为 layer。
  - 四条曲线分别表示不同的预测与目标集合组合：

| 曲线 | 含义 | 图中大致范围 | 解读 |
|---|---|---:|---|
| PF top-5 → DC top-5 | 用 Prefill 前 5 热门 experts 预测 Decode 前 5 | 约 45%–75% | 最严格条件下仍保留明显重叠 |
| PF top-10 → DC top-5 | 用更大的 Prefill 候选集合覆盖 Decode top-5 | 约 65%–95% | Prefill top-10 通常已覆盖大部分 Decode top-5 |
| PF top-10 → DC top-10 | 同规模 Top-10 集合比较 | 约 70%–95% | 热门 expert 排序具有较稳定延续性 |
| PF top-20 → DC top-10 | 用 Prefill top-20 覆盖 Decode top-10 | 多数约 85%–100% | 扩大候选池后，Decode 热点几乎均能被覆盖 |

  - 红色的 **PF top-5 → DC top-5** 曲线最低，但仍普遍远高于随机或无关基线，说明即使只保留极少数 hot experts，Prefill 信息仍有实际预测价值。
  - 绿色的 **PF top-20 → DC top-10** 曲线最接近 100%，表明系统若在 Prefill 后将约 20 个高频 experts 作为 Decode 的候选热点集合，就可以覆盖绝大多数 Decode 的 top-10 experts。
  - 不同 layer 的曲线存在波动，说明 expert specialization 与阶段一致性具有层间差异；系统不应假设一个统一阈值适用于全部层，而应采用**per-layer placement / caching policy**。
  - 图中的虚线“PD fully irrelevant Baselines”接近 0，强调观测到的重叠率并非偶然，而是明显超过 Prefill 和 Decode 完全无关时的预期。

- 子图 (c) 使用 **Spearman’s Ratio** 量化不同模型中、不同层内 Prefill 与 Decode expert-frequency 排名的一致性：
  - 横轴为 layer，纵轴为 Spearman’s Ratio；该指标越接近 1，代表两个阶段对 expert 热度的单调排序越一致。
  - 四类散点对应 DS、Llama、Qwen、Kimi。绝大多数点位于 **0.7 以上**，不少接近 0.9 或更高。
  - 依据论文采用的解释标准，\(\rho > 0.7\) 可视为强相关；因此图中结果表明：**Prefill 与 Decode 的 expert frequency 在绝大多数层强相关**。
  - Qwen、Llama 和 Kimi 的多个层表现出接近 0.9 的相关性，意味着其 Decode 阶段的热点 experts 可由 Prefill 较可靠地推断。
  - DS 虽然部分层的值较低、离散度更大，但多数仍处于中高相关范围，表明这种规律不是单一模型的偶然特性，而是大规模 MoE 的跨模型现象。
  - 少数低于 0.7 的层说明预测并不完美。这些层可能对生成 token 的局部语义、上下文演化或 router 行为更敏感，因此需要保留动态校正机制，而不能完全静态地锁定 expert placement。

- 从系统设计角度，该图直接支持论文的 **Insight 1: Prefill-data-driven prediction**：
  - 在 Decode 开始前，系统已经拥有完整的 Prefill expert trace；因此可立即估计每层的 Decode hot experts。
  - 对于 **Prefill–Decode disaggregation** 系统，Prefill machine 可将 expert frequency、Top-K experts 或 layer-wise heatmap 传给 Decode machine，用于冷启动 placement。
  - 对于 GPU memory offloading、CXL memory、remote HBM 或 multi-chiplet GPU，系统可提前：
    - **prefetch** 预测热点 experts；
    - 将热点 expert weights **cache** 到 local DRAM / HBM；
    - 对高频 experts 做 **replication**；
    - 将潜在高负载 experts **remap** 到不同 GPUs 或 dies；
    - 在 Decode 初期避免等待在线 profiling 数据积累。

- 图对应论文第 VI 节的真实 GPU case study：作者用 Prefill trace 指导 Qwen3-235B 在 8×H100 上的 Decode expert placement。
  - **Remap-based placement**：不增加显存占用，重新分配 experts，使各 GPU 的预测 workload 更均衡。
  - **Duplication-based placement**：为每张 GPU 预留额外 expert slots，复制预测的 hot experts，并用动态 dispatch 将 tokens 分发到副本。
  - 实验结果显示，Remap 和 Dup 相对默认 contiguous placement 分别实现约 **15.5%** 和 **12.5%** 的 MoE computation speedup，并接近使用 oracle Decode information 的最优 placement。

- 该图的关键价值不只是“Prefill 和 Decode 相似”，而是给出了一个可操作的精度—资源权衡：
  - 若 local memory 极其有限，可采用 **PF top-5**，但覆盖率较低且层间波动更大。
  - 若希望较好地覆盖 Decode 的最热 experts，**PF top-10** 是较实用的候选集合。
  - 若系统可承担更多 cache/replica 空间，采用 **PF top-20** 能以很高概率覆盖 Decode top-10，是更稳健的 placement 与 prefetch 策略。
  - 因而，实际系统可按层、按显存余量、按通信代价自适应选择 Top-K，而非使用固定复制数量。

- 图的边界与注意事项：
  - 该图证明的是**频率排序和热门集合的统计相关性**，并不意味着每个 Decode token 的 expert routing 都可以被逐 token 精确预测。
  - 图中 Qwen3 Layer 66 的规律不能自动等价于所有模型、所有层和所有工作负载；子图 (c) 已显示存在模型间与层间变化。
  - 频率相关性适合指导粗粒度的 caching、replication 与 placement；若要做细粒度 token-level prefetch，还应结合论文中的 cross-token heatmap 和实时 decode trace。
  - 对短输出请求，该方法尤其有价值：传统基于 Decode 历史的动态 load balancer 尚未收集足够样本时，Prefill trace 已可作为即时先验。

### 1308dea61ec4fc505a6dae0791e9dbf9e266f37bcc1e873748563fca78e3d230.jpg

![1308dea61ec4fc505a6dae0791e9dbf9e266f37bcc1e873748563fca78e3d230.jpg](images/1308dea61ec4fc505a6dae0791e9dbf9e266f37bcc1e873748563fca78e3d230.jpg)

- 该图展示 **Llama4 layer 7** 在 **MMLU English** 工作负载下，各个 **Expert** 被路由/激活的归一化次数（**Norm Count**）分布，用于刻画单个 Expert 的空间负载不均衡。

| 视觉元素 | 含义 | 图中观察 |
|---|---|---|
| 横轴 **Expert ID** | 专家编号 | 图像分辨率未展示可辨识的具体 ID，但覆盖该层全部 Experts。 |
| 纵轴 **Norm Count** | 相对平均激活次数的归一化计数 | 刻度约为 0、5、10、15、20。 |
| 青绿色柱状分布 | 每个 Expert 的实际激活热度 | 左侧少数 Experts 形成尖锐高峰，右侧绝大多数接近零。 |
| 红色水平线 **Average Value** | 所有 Experts 的平均激活水平 | 平均值位于接近底部的位置，凸显峰值远高于均值。 |

- 图中最显著的模式是 **极强的长尾（long-tail）与偏斜（skewness）分布**：
  - 少量头部 Experts 的激活次数显著高于平均值。
  - 最高柱约为 **16–17 倍平均值**；这与论文正文“部分 Experts 被激活次数超过平均值 16 倍”的结论一致。
  - 高热度区域集中在少数相邻的 Expert ID 附近，随后快速衰减。
  - 大量尾部 Experts 的 Norm Count 接近 0，说明它们在该特定任务/语言组合中几乎不参与计算。

- 从系统视角看，该图否定了“Expert selection 完全均匀随机”的假设：
  - 若所有 Experts 被均匀选择，柱状高度应围绕红色 **Average Value** 相对平坦地分布。
  - 实际分布却呈现“一小部分极热、绝大部分极冷”的结构，意味着 MoE 的路由行为具有可利用的统计规律。
  - 因而，MoE 推理瓶颈不仅是总计算量，还来自 **hot Experts** 所在设备的排队、HBM 访问争用和跨设备通信集中。

- 图所揭示的直接性能问题如下：

| 问题 | 形成机制 | 后果 |
|---|---|---|
| **Workload imbalance** | 热门 Expert 接收远多于平均水平的 tokens | 承载热 Expert 的 GPU/die 成为关键路径，其他设备闲置。 |
| **Memory bandwidth contention** | 大量请求重复读取同一热 Expert 权重 | 热 Expert 本地 HBM、cache 或远程链路发生拥塞。 |
| **Inter-unit communication concentration** | 热 Expert 不在 token 当前执行单元本地时，需要集中 remote fetch | All-to-All、D2D、NVLink 或 CXL 流量集中，尾延迟上升。 |
| **低效静态均分** | 按 Expert 数量而非按激活频率进行均匀放置 | “每卡 Expert 数相同”不等于“每卡工作量相同”。 |

- 该图支撑论文的 **Insight 4: Popular expert decentralization**：
  - 对频繁被激活的 **hot Experts** 进行复制（replication），让多个 compute units 分担其 tokens。
  - 避免将多个热 Experts 共置于同一 GPU/die，否则其负载会叠加。
  - 在多 chiplet 或 wafer-scale GPU 中，可将热 Experts 分散到拓扑上不同的位置，以降低局部 HBM 压力和 D2D 拥塞。
  - 在内存受限系统中，复制策略需优先分配给热度最高、预计能最大幅度降低瓶颈负载的 Experts。

- 该结果也为任务感知策略提供了基础，但其适用范围应被准确限定：
  - 此图对应的是 **MMLU English**，即特定模型、特定层、特定任务与特定语言下的统计结果。
  - 它证明了该工作负载中的 Expert 热度具有明显偏斜，**不意味着所有任务共享完全相同的热 Expert 集合**。
  - 论文后续的 MMLU Chinese 对比表明，语言变化会显著改变热门 Expert 的构成。因此，部署时应依据任务类型、语言和在线请求构成更新 placement，而不能永久固化一套热度排序。

- 综合而言，这张图的核心信息是：**MoE Expert 激活并非均匀随机，而是具有高度集中的热点结构。** 这一结构使得面向热 Expert 的复制、去中心化放置、负载感知调度以及任务/语言感知迁移成为合理且必要的系统优化手段。

### e28e21f0dbc22424b6c7637404ae562fee6dfe55f91ed402a293673b537a7511.jpg

![e28e21f0dbc22424b6c7637404ae562fee6dfe55f91ed402a293673b537a7511.jpg](images/e28e21f0dbc22424b6c7637404ae562fee6dfe55f91ed402a293673b537a7511.jpg)

- 图像由两个并列热图构成，对应 **Llama4 layer 7** 在 57 个 MMLU subjects 上的 Top-10 expert 选择结果：左图为 **MMLU English**，右图为内容相同但语言切换后的 **MMLU Chinese**。  
  - 横轴：**Subject**，约覆盖 57 个学科。  
  - 纵轴：**Expert ID**，范围约为 0–127。  
  - 颜色：**Expert Rank**；深紫表示该 expert 未进入该 subject 的 Top-10，颜色越亮表示其在该 subject 中的排名越靠前/更显著。  

- 图像最核心的视觉结论是：专家路由呈现出同时存在的 **跨任务稳定性** 与 **任务/语言特异性**，并非均匀随机分布。

| 视觉现象 | 左图：MMLU English | 右图：MMLU Chinese | 含义 |
|---|---|---|---|
| 长水平亮线 | 在约 Expert ID 70–80、95–105 等区域存在跨大量 subjects 的连续亮线 | 稳定亮线的位置明显改变，约 Expert ID 35–45、70 附近更突出 | 少数 expert 在许多任务中反复进入 Top-10，即存在 **globally popular experts** |
| 零散亮点/短线 | 分布在多个 Expert ID 和特定 subjects 上 | 同样存在，但空间位置与 English 显著不同 | 部分 expert 只对某些领域或题型更活跃，体现 **task specialization** |
| 两图亮线重合度 | 有少量共同的活跃区域 | 整体主热点发生迁移 | 仅改变语言即可改变 expert preference，说明语言本身是关键 routing factor |
| 背景大面积深色 | 绝大多数 expert 很少或从未进入 Top-10 | 同左图 | expert activation 呈明显 **long-tail / skewed distribution** |

- 左图中的多条亮色水平带尤其值得注意：  
  - 它们几乎贯穿全部或大部分 subjects，表示对应 expert 不依赖于某一个具体学科，而是具有较强的通用激活倾向。  
  - 这类 expert 会形成系统中的持续热点；若多个请求并发到来，极易成为某些 GPU、die 或 memory tier 的负载和访问瓶颈。  
  - 图中其他只在局部 subject 区间出现的亮点，则表明不同学科会调用不同的 specialized experts。

- 左右图的对照比单图更有价值：  
  - English 与 Chinese 使用相同或高度相近的知识任务，但热门 expert 的主要水平带明显不一致。  
  - 这直接支持论文的结论：**expert selection strongly correlates with task type, and language is a decisive task characteristic**。  
  - 从图像上看，两种语言并非只是对同一组热门 expert 进行轻微重排序，而是出现了较明显的热点 expert 集合迁移。论文据此指出：两种语言中虽然仍有约 5–6 个普遍活跃 expert，但它们的主要热门 expert 重叠有限。

- 该图验证的是论文的 **Observation 4 (Ob4): Single Expert Activation Imbalance**：  
  - expert 使用频率高度不均匀；  
  - 热点 expert 与请求的 task/domain/language 有关；  
  - 因而静态、均匀地将 experts 映射到设备，并不能保证实际执行负载均衡。  

- 对 MoE serving system 的直接设计启示如下：

| 对应 Insight | 图像支撑证据 | 推荐机制 |
|---|---|---|
| **Insight 4: Popular Expert Decentralization** | 少数 expert 在大量 subjects 中持续高亮 | 对热点 expert 做 replication，或将其分布到多个 GPU/die，避免单点拥塞 |
| **Insight 6: Workload-Aware Serving System** | English 与 Chinese 的热点带显著不同 | 根据 language、task type、tenant workload 等元数据，在服务前迁移或预复制相应 experts |
| **Insight 3: Expert-Placement-Aware Workload Distribution** | 任务特异 expert 会使不同设备承受不同请求量 | 调度请求时同时考虑 expert placement、热点程度和设备当前 load，而非只追求本地执行 |

- 对部署实践而言，图像传递的关键策略是：  
  - 当 workload 以 English 请求为主时，应优先识别并复制左图中跨 subject 持续高亮的 experts。  
  - 当 workload 切换为 Chinese 时，不能直接沿用 English 的 expert placement；应改为针对右图热点重新布局。  
  - 若系统无法完全复制热点 experts，则至少应避免把多个高频 expert colocate 在同一 GPU/die 上。  
  - 这种优化只需对每个模型和典型 workload 进行一次离线 profiling，之后可作为运行时 placement policy 的先验知识复用。  

- 图像也有边界：它展示的是 **Top-10 rank** 而非完整的绝对 token frequency，因此最适合用于识别“哪些 experts 经常成为候选热点”及其跨任务稳定性；若要精确确定 replication 数量，还需结合每个 expert 的实际 token 数、batch size、expert capacity、通信拓扑与显存余量。

### 6e7efca2e242f3899ba2a26abd6ae773ba6131f225788284524784ac7a54b458.jpg

![6e7efca2e242f3899ba2a26abd6ae773ba6131f225788284524784ac7a54b458.jpg](images/6e7efca2e242f3899ba2a26abd6ae773ba6131f225788284524784ac7a54b458.jpg)

- 该图是 Figure 9(a) 的局部，即 **DeepSeek-V3（DS）在 Layer 17 的 expert-pair co-activation heatmap**。标题中的 **“8/256”** 表示：每个 MoE layer 有 **256 个 experts**，每个 token 路由到 **8 个 experts**。

- 图的核心目的，是展示任意两个 experts 在同一 token 上被同时激活（co-activated）的频率，而非单个 expert 的独立访问频率。

| 视觉元素 | 含义 | 图中观察 |
|---|---|---|
| 横轴、纵轴 | Expert ID，范围约为 0–255 | 完整矩阵为 256×256 的 expert pair 组合 |
| 色彩亮度 | Expert pair 的共激活频率，相对随机路由理论概率归一化 | 越亮表示该 expert pair 被同时选择得越频繁 |
| 右侧色条 | 归一化后的 co-activation 强度 | 可见刻度约为 **0 到 20**；论文正文指出部分 pair 可达随机期望的 **20–40×** |
| 左侧大图 | 全部 256 个 experts 的 pairwise 共激活结构 | 呈现大量稀疏亮点及明显的块状/对角模式 |
| 青绿色方框与连线 | 局部放大区域 | 放大 expert ID 约 **64–128** 范围内的局部共激活关系 |
| 右侧放大图 | 方框区域的细粒度结构 | 可清晰看到高频 pair 沿斜对角方向形成离散亮点链 |

- **最显著的现象是共激活高度非均匀。**
  - 若 expert routing 完全随机，热力图应接近均匀的低强度噪声。
  - 实际图中存在大量局部高亮点和高亮带，说明少量 expert pairs 的联合激活概率远高于随机期望。
  - 这证明 MoE routing 虽然对单次请求而言动态变化，但在统计意义上存在稳定、可利用的结构。

- **矩阵具有中心对称性。**
  - Pair \((e_i,e_j)\) 与 \((e_j,e_i)\) 表示同一对 experts，因此其共激活频率应相同。
  - 图中主对角线两侧的亮点分布基本镜像对应，符合 co-activation matrix 的理论性质。
  - 该对称性也说明图记录的是“是否共同被选中”的无序 expert pair，而不是 expert 调用顺序。

- **放大区域显示出明显的斜对角带状结构。**
  - 在 expert 约 64–128 的子区域内，亮点沿着与主对角线平行的方向分布，而不是均匀散落。
  - 这意味着某些 ID 相近、或属于相同 router/node group 的 experts 更容易共同被选中。
  - 图中红色虚线/红色高亮轨迹用于强调这类结构化高亲和 pair；它们并非简单的单点热点，而是一组具有相对规律位置关系的高频配对。

- 从论文上下文看，这种块状与斜线结构和 **DeepSeek 的 routing restriction** 有关：
  - DeepSeek 会限制 token 优先路由到相邻节点中的 experts，以降低通信开销。
  - 因而，部署上邻近或分组相关的 experts 会形成更高的 co-activation affinity。
  - 这解释了为什么图中高亮 pair 并非随机遍布整个 256×256 空间，而是集中于若干受约束区域。

| 结论 | 图像证据 | 系统含义 |
|---|---|---|
| Co-activation 有强偏斜 | 少数亮点远亮于背景 | 可优先优化少量高频 expert pairs，而非均等对待全部组合 |
| Pair affinity 与 expert 分组有关 | 放大图中有局部斜对角/块状高亮 | Expert placement 应考虑 router group 或 node locality |
| 同机放置高亲和 pair 会造成计算集中 | 高频 pair 同时请求同一设备上的两个 experts | 可能导致单个 die/GPU 的 MoE workload 峰值更高 |
| 跨设备拆分高亲和 pair 可提升并行度 | 高频 pair 可被分配到不同 compute units 并行执行 | 对应论文的 **Insight 5: Expert-pair separation** |
| 分离也有通信代价 | 同一 token 访问分属不同设备的 experts | 必须在 workload balancing 与 All-to-All/D2D communication 间权衡 |

- 该图直接支撑论文的 **Observation 5 (Ob5)**：**expert-pair co-activation affinity**。论文进一步量化指出，跨层统计中 **top 10% 的 expert pairs 可贡献约 60%–80% 的总共激活量**。因此，优化对象并不是全部约 \(256 \times 255 / 2 = 32,640\) 个 pair，而是其中很小的一部分热点 pair。

- 对 MoE serving 的具体启示如下：
  - **Expert-pair separation**：将经常共同激活的 experts 放到不同 GPU、chiplet 或 die，避免其计算同时堆积到一个单元。
  - **Topology-aware placement**：若必须拆分高亲和 pair，应优先置于低延迟、高带宽互连的邻近单元，降低跨单元数据移动代价。
  - **Batch-aware decision**：大 batch 下并行化收益通常更显著；小 batch 下，拆分 pair 带来的通信成本可能超过负载均衡收益。
  - **Task-aware adaptation**：由于不同任务、语言会改变 expert 热度分布，pair affinity 也应通过离线 profiling 或周期性在线统计更新，而不宜假设一个静态 placement 对全部 workload 都最优。

- 总体而言，这张图反驳了“MoE expert 组合完全随机”的直觉：**DeepSeek-V3 的 expert 共同激活呈现稀疏、成组、可预测的统计结构**。这为跨 GPU/chiplet 的 expert placement、负载均衡和通信调度提供了可操作的依据。

### c5cff5acbbb857668d59a474821b8afa4a162a84b462629460c42db871899e0c.jpg

![c5cff5acbbb857668d59a474821b8afa4a162a84b462629460c42db871899e0c.jpg](images/c5cff5acbbb857668d59a474821b8afa4a162a84b462629460c42db871899e0c.jpg)

- 该图对应 Figure 9(c)，展示**同一 MoE layer 内 expert-pair co-activation 的累计分布（CDF）**，用于验证不同 expert pair 的共同激活并非均匀随机，而是呈现强烈的长尾与热点特征。

| 图表元素 | 含义 |
|---|---|
| 横轴：**Percent of Top Pairs** | 按共同激活频率从高到低排序后，取排名靠前的 expert pairs 的比例 |
| 纵轴：**Percent of Total Count** | 这些 top expert pairs 累计贡献的全部共同激活次数比例 |
| 红色 | **DeepSeek V3** |
| 绿色 | **Qwen3** |
| 黄色 | **Kimi K2** |
| “Same Layer” | 统计限定在**同一 MoE layer**内被同时选中的 expert pairs，而非跨层 pair |

- 曲线均在左侧快速上升、随后逐渐饱和，表明 co-activation 高度集中：
  - 仅取**前 10% 左右的 expert pairs**，即可覆盖约 **60%–80%** 的总共同激活量。
  - 到前 **20%–30%** pair 时，累计覆盖率已经接近或超过 **80%–90%**。
  - 约在前 **40%–50%** pair 范围内，三个模型的累计激活量已接近 **100%**；剩余大量 pair 几乎很少共同出现。

- 不同模型的集中程度存在明显差别：

| 模型 | 曲线特征 | 解读 |
|---|---|---|
| **Qwen3** | 绿色曲线整体最高、最陡 | co-activation 最集中；少量 pair 承担了最大比例的共同激活，pair-level affinity 最强 |
| **Kimi K2** | 黄色曲线居中 | 存在显著热点 pairs，但集中程度低于 Qwen3 |
| **DeepSeek V3** | 红色曲线整体最低、初段更平缓 | 共同激活较分散；但仍远非均匀随机分布，热点 pair 依旧覆盖多数激活 |

- 若 expert pair 的共激活完全均匀，则 CDF 应大致接近对角线关系：前 \(x\%\) 的 pairs 仅解释 \(x\%\) 的激活次数。图中的三条曲线均显著高于这一均匀基线，因而直接支持论文的 **Ob5: Expert Pair Co-activation Affinity**：
  - 某些 expert pairs 被共同选择的概率远高于随机期望；
  - expert routing 虽然表面上由 token 动态决定，但在 pair 粒度上存在稳定、可利用的统计结构；
  - 这种结构跨 DeepSeek V3、Qwen3、Kimi K2 三个大规模 MoE 模型均成立，具有较好的跨模型普适性。

- 图中曲线在接近 100% 时较早饱和，还揭示一个重要事实：**大量理论上可能的 expert pairs 在实际 workload 中几乎从不共激活**。因此，系统无需对全部 \(O(E^2)\) pair 空间进行等价建模；只维护热门 pair 的稀疏统计即可获得大部分收益。这降低了 placement、scheduling 和 profiling 的状态维护成本。

- 对系统设计的直接启示是论文提出的 **Insight 5: Expert-pair separation**：
  - 应将高频共同激活的 experts 尽量部署到**不同 compute units / GPU dies**；
  - 这样可以把原本集中在单一设备上的并行 expert computation 分散到多个设备执行；
  - 其目标是降低单设备 hotspot、缩短尾部执行时间，并提升总体 hardware utilization。

| 优化策略 | 图中依据 | 潜在收益 | 主要代价 |
|---|---|---|---|
| 高频 pair 分离部署 | top 10% pair 覆盖 60%–80% 激活 | 缓解同一设备上的计算拥塞，提升 parallelism | 增加 cross-unit all-to-all / remote access |
| 热门 pair 感知的 task scheduling | co-activation 可被少量 pair 描述 | 提前避开将相关 token block 派给同一资源 | 需要维护在线或离线 pair heatmap |
| pair-aware replication | 某些 pair 长期共同热门 | 为热点 experts 增加副本、降低排队 | 占用额外 HBM/DRAM 容量 |
| workload-aware placement | 不同模型、任务可能有不同热点 pair | 针对具体请求分布改善 placement | workload 漂移时需重新估计 |

- 该图也限定了 “pair separation” 的适用边界：
  - **不能机械地分离所有共同激活 pair**。如果设备间互连较慢、batch 较小，通信开销可能超过并行化收益。
  - 对于 DeepSeek V3，其曲线较 Qwen3 平缓，说明 pair affinity 相对弱，placement 决策应更加谨慎。
  - 对于 Qwen3，热点更集中，pair-aware placement 的优先级更高，因为少量部署决策即可影响大比例请求。
  - 最优目标应是联合优化：**负载均衡、expert replication、设备拓扑距离、remote-memory 代价与通信带宽**，而非只最小化 co-location。

- 总体而言，这张图以 CDF 形式将复杂的 pairwise routing 行为压缩为一个清晰结论：**MoE expert pair 的共同激活具有显著稀疏性和热点性；少量高频 pairs 主导了大部分执行负载。** 这为面向 multi-GPU、multi-chiplet GPU 和 wafer-scale GPU 的静态 expert placement 与动态 workload distribution 提供了可操作的统计依据。

### Figure 10. (a) Wafer-scale multi-chiplet GPU architecture with additional units highlighted in orange. (b) SoW (System-on-Wafer) technology structure. (c) Data format in the Global Command Processor for our proposed task distribution strategy.

![b4e95e3f028d600afe5407f559f62ae7a17814d96fc30b618e6dc56ceb5df9f0.jpg](images/b4e95e3f028d600afe5407f559f62ae7a17814d96fc30b618e6dc56ceb5df9f0.jpg)

- 图10展示了论文提出的 **wafer-scale multi-chiplet GPU** 设计，以及为 MoE serving 增加的轻量级硬件控制单元。其核心思想是：在维持 **single-GPU-like programming model** 的同时，由硬件感知 expert placement、访问局部性和预测结果，降低 **die-to-die（D2D）** 数据移动并缓解负载不均衡。

| 子图 | 展示对象 | 核心信息 |
|---|---|---|
| (a) | Wafer-Scale GPU Architecture | 两级 Command Processor、Local HBM 缓存/复制、ATU 与 PDU 驱动的远程 expert 数据访问优化 |
| (b) | System-on-Wafer (SoW) Technology | GPU Die、HBM Die 与互连结构，说明 local / remote HBM 访问代价差异的物理根源 |
| (c) | Data in Global CP | Global CP 使用的 expert distribution 与 cross-token correlation 数据结构 |

- 子图 (a) 左侧给出整体 **Wafer-Scale Multi-Chiplet GPU** 的抽象视图：
  - 整个 wafer 由大量重复的计算单元构成；每个单元包含一个 **GPU Die** 和相邻的 **HBM**。
  - 图中以粉橙色突出显示新增或增强的控制组件，包括 **Global CP**、每个 die 内的 **Local CP**、以及 D2D Controller 中的 **ATU** 和 **PDU**。
  - 原始 GPU 的问题在于：虽将 wafer 暴露为一个统一 GPU，但传统调度器将全部 SM 视作同质资源，**不理解 expert 位于哪个 die，也不理解请求的 expert 热度和访问相关性**。
  - 因而，一个 MoE kernel 可能在远离 expert weights 的 die 上执行，造成多跳 remote HBM access；同时，热门 expert 的计算会集中在少数 die，使其他 die 空闲。

- 图 (a) 中的控制路径用虚线标示，体现 **two-level Command Processor**：
  - **① Global CP 接收 MoE Kernel**。此时 Global CP 获得当前 batch 的 expert requests，即每个 expert 需要处理多少 token/request block。
  - **② Global CP 向各 Local CP 分发 sub-kernel 和预测信息**。Global CP 根据 expert 的实际分布、相邻 die 的负载和通信成本，决定某个 expert 的计算应放在哪些 die 上执行。
  - **③ Local CP 将 sub-kernel 映射给本地 SM**。这使得全局的 topology-aware 决策可以转换成 die 内常规的 SM task scheduling。
  - **④ Local CP 收集 expert duplication / placement 状态并反馈给 Global CP**。Global CP 据此更新全局 expert 分布，形成闭环调度。

- 该层级化结构的功能分工如下：

| 组件 | 所处层级 | 主要职责 | 解决的问题 |
|---|---:|---|---|
| **Global CP** | Wafer 全局 | 分配 MoE sub-kernel；维护 expert placement；运行 predictor | 避免无视数据位置的全局任务分配 |
| **Local CP** | 单 GPU Die | 将收到的 sub-kernel 调度至本地 SM；配置本地预测状态 | 保持 die 内调度轻量、快速 |
| **D2D Ctrl** | 单 GPU Die | 处理跨 die 读写与远程数据返回 | 支持 remote HBM 数据传输 |
| **ATU** | D2D Ctrl 内 | 将远程 expert 地址重定向为本地副本地址 | 避免已复制数据仍发生远程访问 |
| **PDU** | D2D Ctrl 内 | 根据 prediction table 判断是否复制远程 expert | 将未来可能使用的 remote data 留在 local HBM |

- 图 (a) 下方和右侧的数据路径区分了两种访问模式：
  - **绿色路径：Non-duplicated remote read。**
    - 当 SM 请求的数据仅存在于 remote HBM 时，请求经 **D2D Ctrl** 发出，穿越 D2D 网络后访问远端 HBM。
    - 数据返回本 die 后，PDU 查表决定该数据是否值得复制到本地。
    - 无论是否复制，数据都会先返回给发起请求的 SM，保证计算继续执行。
    - 若 PDU 决定复制，则该 expert weights 会被写入 **LLC** 和 **local HBM**；同时 ATU 更新地址映射，PDU 将 `is_local` 置为 1。

  - **蓝色路径：Duplicated local read。**
    - 对于已经缓存/复制到 local HBM 的远程 expert，SM 仍可使用原始 remote address 发起请求。
    - **ATU** 自动将该 remote address 翻译为 local address。
    - 请求被重定向至本地 LLC / Memory Controller / local HBM，而无须再经过 D2D 网络。
    - 这对软件透明：软件无需显式管理 expert migration 或修改指针，但硬件可将远程访问转化为低延迟、本地高带宽访问。

- 图中的数据流编号与论文工作流相对应：

| 编号 | 位置 | 含义 |
|---:|---|---|
| ① | Global CP / SM / D2D access | Kernel 到达 Global CP；SM 发起远程数据请求 |
| ② | Global CP→Local CP、D2D network | 分发任务与预测；远程请求通过 D2D 网络传输 |
| ③ | Local CP→SM、remote data return | 本地执行调度；远程数据从网络返回 D2D Ctrl |
| ④ | Local CP→D2D Ctrl、PDU→LLC | 配置预测表；判定并启动本地复制 |
| ⑤ | LLC→Mem Ctrl / local HBM | 将被预测的高价值 expert 写入 local HBM |

- 图 (b) 解释了该架构为何必须重视数据位置：
  - 每个 **GPU Die** 直接连接附近多个 HBM die，因此访问本地 HBM 的路径较短、带宽更高、时延更低。
  - 相邻 GPU Die 的连接包括：
    - **LSI（Local Silicon Interconnect）**：用于垂直方向邻接连接；
    - **SerDes**：用于水平方向互连；
    - **UCIe**：图中以双向蓝色箭头标记的 die-to-die 互连协议/接口语义。
  - **TIV**、**RDL** 和 **LSI** 代表 SoW 封装中的互连层与重分布层，承载 GPU Die、HBM Die 之间的高密度连接。
  - 虽然该互连可提供 terabit 级带宽，但 remote HBM access 仍需跨越一个或多个 GPU Die；当大量请求同时访问远端热门 expert 时，会出现链路争用、拥塞和多跳时延累积。
  - 因此，该图直接支撑论文的两个优化方向：
    - **Task allocation**：尽量在靠近 expert 的 die 上执行对应计算；
    - **Predictive local HBM duplication**：将未来高概率访问的 remote expert 提前转为 local expert。

- 图 (c) 展示 Global CP 的两类关键状态。其中，**Expert Distribution Table** 是空间信息，**Cross-token Heatmap** 是时间相关性信息。

| 数据结构 | 字段/形式 | 含义 | 用途 |
|---|---|---|---|
| **Expert Distribution Table** | `Expert` | expert ID | 标识 expert |
|  | `Init Die` | 初始放置的 die ID | 保存静态初始 placement |
|  | `Distribution` | n-bit bitmap | 每个 bit 表示该 expert 是否已在对应 die 上存在副本 |
| **Cross-token Heatmap** | expert×expert 矩阵 | 当前 token expert 与下一 token expert 的条件共激活关系 | 预测下一个 token 高概率使用的 experts |

- **Expert Distribution Table** 中的 `Distribution` 位图是动态调度的基础：
  - 例如，expert 0 的初始位置为 die 0，若其 distribution 表示 `1100...0`，意味着该 expert 除初始 die 外，已经在另一个 die 上存在副本。
  - 对于 expert 255，`Init Die = 24`、`Distribution = 000...1` 表示它仅存在于 die 24。
  - Global CP 使用这一表判断：某项 expert computation 能否本地执行、是否应分配到邻近 die、以及是否需要远程读取或复制。
  - 该设计使 placement 不再是固定的 **Expert Parallelism (EP)** 静态映射，而成为可随访问行为演化的动态分布。

- **Cross-token Heatmap** 是论文中 temporal relation 的硬件化表示：
  - 行可理解为当前 token 已激活的 expert，列表示下一 token 的候选 expert。
  - 深色单元表示较高的条件激活概率，即在当前 expert 出现时，某些下一 token experts 更可能被路由器选中。
  - Global CP 根据当前 MoE kernel 已选择的 experts，读取对应 heatmap 行，提取每行的 top-\(n\) 候选，再将结果写入各 die 的 **Prediction Table**。
  - 每个 die 的 Prediction Table 包含：
    - `cp_en`：该 expert 是否应被预测性缓存/复制；
    - `is_local`：该 expert 的副本当前是否已经位于 local HBM。
  - 因此，预测机制并不试图精确预测完整路由结果，而是以较低硬件复杂度识别“**值得提前本地化的热门远程 expert**”。

- 该图体现的端到端优化逻辑可概括为：
  - **先分配**：Global CP 基于 expert placement 和 die load，将 MoE workload 划分为 locality-aware sub-kernels。
  - **再预测**：从 cross-token expert correlation 中预测下一 token 的热点 expert。
  - **后缓存/复制**：PDU 在远程数据首次返回时，按预测结果将高价值 weights 复制到 local HBM。
  - **最后重定向**：ATU 在后续访问中自动将 remote address 重定向至 local copy。
  - 其本质是把 MoE routing 看似随机的行为，转化为可利用的 **temporal locality** 与 **spatial placement** 信息。

- 从架构设计角度看，图10的关键价值在于：
  - **不要求修改上层 MoE 模型或应用程序**：软件仍将整个 wafer 当作单一 GPU 使用。
  - **不要求程序员显式管理 D2D communication**：数据 placement、复制和地址重定向均由硬件控制面完成。
  - **将全局粗粒度决策与局部细粒度执行分离**：Global CP 负责跨 die 优化，Local CP 负责本地调度。
  - **同时优化通信与负载均衡**：仅做 local placement 会导致热门 expert 所在 die 过载；仅做负载迁移又会增加 remote access。该设计通过邻近候选 die、预测缓存和动态副本同时权衡二者。
  - 这也是论文后续获得显著吞吐提升的机制基础：**Allo** 主要减少不必要的跨 die 执行并平衡任务，**Pred** 主要减少未来 token 的 remote HBM reads，二者联合构成 **Allo+Pred**。

### Figure 11. Proposed task allocation algorithm and data-driven predictor.

![18bbec3c008db03eaaa38c66274a4471663b92f10b171928dd36cdab16f68073.jpg](images/18bbec3c008db03eaaa38c66274a4471663b92f10b171928dd36cdab16f68073.jpg)

- 图 11 将论文的两项核心机制并列展示：**(a) task allocation algorithm** 解决“计算应分配到哪个 Die”，目标是在负载均衡与跨 Die 数据访问之间折中；**(b) data-driven predictor** 解决“哪些 remote experts 应提前复制到 local HBM”，目标是降低下一 token 的 D2D 数据移动。

- | 子图 | 输入信息 | 核心操作 | 输出/作用 | 对应论文洞见 |
  |---|---|---|---|---|
  | **(a) Task Allocation Algorithm** | expert request 数量、expert 的当前 Die 分布、各 Die 负载、mesh 拓扑 | 为每个 expert 生成有限的候选 Die，并按 cost model 分配 request blocks | 每个 Die 的 sub-kernel/task allocation plan | **Insight 3：Expert-placement-aware workload distribution** |
  | **(b) Data-Driven Predictor** | 当前 token 激活 experts、Cross-Token Heatmap | 查询当前 experts 对应行，提取高概率 next-token experts | local HBM 的 expert duplication/caching 决策 | **Insight 1、2：跨 token 预测与层级 memory management** |

- 在图 11(a) 中，整个 **Wafer-Scale GPU** 被抽象为二维 Die 网格：
  - **绿色方块（Local Die）**表示本地存有目标 expert 权重的 Die，例如图中的 expert 2、3、0 所在位置。
  - **红色方块（Remote Die）**表示存有其他 expert 的远端 Die，例如 expert 4、8、7、5、1、6、9。
  - 空白方块表示当前未承载相关 expert 权重、但可被纳入候选范围的 Die。
  - 图的重点不是固定某个 expert 的具体编号，而是说明：**task 不应只局限在 expert 原始存储 Die 上执行，也不能无约束地分散到整块 wafer。**

- 图 11(a) 的候选生成逻辑对应 Algorithm 1 中的 `GenCandidateList(expert id, dis=1)`：
  - 首先选取保存该 expert 的 **local die list**。
  - 随后通过 `FindNearDies` 加入距 local Die 不超过预设距离 `dis` 的邻近 Die，即图中蓝色框和箭头所指的 **Final Candidate Dies**。
  - 候选范围被限制在 local Die 及其拓扑邻居，而不是所有 Die，体现了一个关键设计原则：**优先利用数据局部性，再用有限的远端扩展能力消除热点负载。**
  - 这种候选裁剪显著压缩了搜索空间。若在 24/25 个 Die 上全局搜索，调度开销和远程通信风险都会增大；仅考虑相邻 Die，则可在较低 hop 成本下实现负载转移。

- 图 11(a) 所表达的调度权衡可概括为：

  | 分配选择 | 优点 | 代价/风险 | 算法态度 |
  |---|---|---|---|
  | 在 expert 原始 **Local Die** 执行 | 无 remote HBM 读取，D2D 流量最低 | 热门 expert 可能压垮少数 Die | 优先选择，但不绝对限定 |
  | 在邻近 **Remote Die** 执行 | 可分摊热门 expert 的计算压力 | 需访问远端权重，产生 D2D hop | 当 local Die 过载时启用 |
  | 在远距离 Die 执行 | 理论上负载更均衡 | 多跳通信、拥塞和访问延迟明显增加 | 候选机制主动避免 |

- 该子图解释了为何论文的 **Allo Only** 策略可以大幅降低 hop count：任务分配以 expert placement 为先验，尽量让计算靠近权重；但它又不同于传统 **Expert Parallelism (EP)** 的“expert 计算只能在本地执行”，因为它允许把极热 expert 的 token request 分块转移到相邻 Die。论文实验显示，这一机制在大 batch 下相对 EP 具有优势，因为此时热门 expert 的并行拆分可以抵消有限 D2D 访问的成本。

- 图 11(b) 展示 **Data-Driven Predictor** 如何将 Cross-Token Correlation 转化为可执行的缓存策略：
  - 横轴为 **Next Token Expert**，纵轴为 **Previous Token Expert**。
  - 每个热图单元表示条件概率：已知当前/前一 token 激活 expert \(e_i\)，下一 token 激活 expert \(e_j\) 的概率。
  - 深色单元表示较高的条件激活概率；这说明 MoE routing 并非完全随机，而是存在可利用的跨 token 相关性。
  - 例子中当前 token 激活的 expert 集合是 **(1, 4)**，图中以红色标记指出应查询的相关 heatmap rows。

- 图 11(b) 的预测过程可拆为三个明确步骤：

  | 步骤 | 图中标记 | 操作 | 示例结果 |
  |---|---:|---|---|
  | 查询条件行 | **①** | 根据当前 token 的 selected experts，读取 heatmap 对应行 | 读取 expert 1、4 对应的行 |
  | 提取高概率候选 | **②** | 在每个已选行中选取 top-\(n\) 高概率 next experts | 获得多个高概率列候选 |
  | 合并预测集合 | **③** | 对候选去重、合并，得到即将热门的 experts | **(2, 4, 6)** |

- 图中的红色圆角框表示各查询行中被选出的 **Top-\(n\) Experts**。多个当前 expert 的 top-\(n\) 预测结果取并集后，形成绿色框所示的预测集合：**Popular Experts to Duplicate Locally: (2, 4, 6)**。

- 一个容易忽略但很关键的细节是：预测集合 **(2, 4, 6)** 不意味着立即把三者都复制到 local HBM。
  - 当前 Die 在此轮实际读取的是 expert **1** 和 **4**。
  - 因而，当前真正从 remote HBM 返回、且同时被预测为未来仍会使用的 expert 是 **4**。
  - 系统会在返回路径中将 expert 4 复制到 local HBM；expert 2、6 虽被预测为热门，但若本轮尚未发生对它们的 remote read，则不必立即主动搬运。
  - 这是一种**预测驱动但按需触发（predictive yet demand-triggered）**的复制机制：它避免纯预取造成大量无效搬运和 local HBM 容量浪费。

- 该机制与论文的硬件实现直接对应：
  - **Global Command Processor (Global CP)** 保存并更新 cross-token heatmap，基于当前 MoE kernel 的 expert selection 生成 `cp_en`（copy-enable）指导信息。
  - 每个 Die 的 **Prediction Unit (PDU)** 保存 prediction table，判断 remote expert 返回时是否应在 local HBM 中保留副本。
  - **Address Translation Unit (ATU)** 在 expert 已复制后，将原本的 remote address 映射到 local address，使后续 SM 请求直接命中 local HBM/LLC。
  - 因此，图 11(b) 描绘的不只是统计预测，还闭合了“**选择历史 → 预测未来 → remote read 返回 → local duplication → address redirection**”的数据路径。

- 两个子图形成互补关系，而非替代关系：
  - **Task allocation** 主要减少“本轮”计算对远端权重的依赖，并缓解 Die 间 workload imbalance。
  - **Predictor** 主要利用 token 间时间局部性，减少“后续轮次”重复的 remote HBM 访问。
  - 前者属于**计算位置优化**，后者属于**数据位置优化**。
  - 前者优先解决 expert hotspot 导致的并行度不足，后者优先解决相同或相关 expert 在连续 token 中被反复远程读取的问题。

- | 策略组合 | 主要减少的瓶颈 | 图 11 中的对应部分 | 论文实验结论 |
  |---|---|---|---|
  | **Pred Only** | remote HBM read、D2D traffic | (b) | hop count 降低约 **4.5×**，性能提升约 **3.0×** |
  | **Allo Only** | 不合理 task placement、Die 负载失衡 | (a) | hop count 降低约 **142×**，性能提升约 **6.3×** |
  | **Allo + Pred** | 同时降低远端访问与负载失衡 | (a)+(b) | hop count 降低超过 **213×**，平均吞吐提升约 **6.6×** |

- 图 11 还隐含揭示了性能瓶颈的迁移：
  - 在 baseline 中，大量 task 被分配到不持有相应 expert 权重的 Die，**remote access 与多跳 D2D 通信**占主导。
  - 使用 allocation 后，多数计算已靠近 expert placement，D2D 流量骤降；此时性能更受 **workload distribution、HBM bandwidth 与 compute capacity**约束。
  - 再叠加 predictor 后，remote read 进一步减少，但相对 Allo Only 的额外收益较小，原因是 allocation 已消除了大部分不必要远程访问。换言之，**预测的边际收益取决于调度后仍残留多少 remote access。**

- 从系统设计视角看，图 11 的核心价值在于把此前看似随机的 MoE routing 转化为两个可操作信号：
  - **空间信号**：expert 当前位于哪里、哪些 Die 负载较低、哪些邻居可承担计算。
  - **时间信号**：当前 token 所选 experts 会提高哪些 next-token experts 的激活概率。
  - 这使 wafer-scale GPU 可以在维持 **single-GPU-like programming model** 的同时，由硬件透明地进行 topology-aware task placement 与 local-HBM-aware data caching，而无需应用程序显式管理 Die 间通信。

### bbfd327cdd0dd2e239c291641160caee5c0b4b34313a13a9a9dd0e9b9b53e7f0.jpg

![bbfd327cdd0dd2e239c291641160caee5c0b4b34313a13a9a9dd0e9b9b53e7f0.jpg](images/bbfd327cdd0dd2e239c291641160caee5c0b4b34313a13a9a9dd0e9b9b53e7f0.jpg)

- 图中包含两类归一化指标，均以 **Base (w/o proposed HW)=1×** 为基准：
  - 上半部分：**MoE decode throughput**，越高越好，线性纵轴。
  - 下半部分：**Hop Reduction**，表示跨 die 通信 hop 数的降低倍数，越高越好；纵轴为 **log scale**。
  - 横向四列分别为 **DeepSeek-V3、Kimi-K2、Llama-4-Maverick、Qwen3-235B**。
  - 每列上、下两行分别对应 **5×5 Wafer（25 dies）** 与 **8×3 Wafer（24 dies）**。
  - 每组有三个 batch size：**4096、8192、16384**。

- 策略及其含义如下：

| 策略 | 是否使用 proposed HW | 核心机制 | 图中预期作用 |
|---|---:|---|---|
| **Base** | 否 | 均匀放置 experts、忽略 task/data locality 的传统分配 | 性能和 hop 基线 |
| **EP** | 是 | 将 expert computation 固定安排在 expert 所在 die | 消除/减少远程访问，但可能造成负载不均 |
| **Pred Only** | 是 | 基于 cross-token correlation 预测下一 token 所需 experts，并缓存至 local HBM | 降低 remote HBM reads 与 D2D traffic |
| **Allo Only** | 是 | data-placement-aware task allocation，将计算分配到存有或邻近 expert 的 dies | 同时减少通信并平衡负载 |
| **Allo+Pred** | 是 | 结合 allocation 与 prediction/caching | 综合收益最高 |

- **总体结论非常明确：Allo Only 是主要性能来源，Pred Only 是重要补充，Allo+Pred 在绝大多数设置中最佳。**
  - **Pred Only** 的 throughput 通常为 **约 1.4×–4.0×**，说明仅靠预测与 local HBM duplication 已能显著缓解远程访问。
  - **Allo Only** 通常达到 **约 2×–10×** throughput，表明针对 expert placement 的任务分配比单纯缓存更关键。
  - **Allo+Pred** 通常达到 **约 2×–10.5×**，在 Allo Only 基础上进一步消除剩余热点 expert 的远程读取。
  - 论文正文汇总称，组合策略平均带来约 **6.6× throughput speedup**。

- 代表性 throughput 结果如下；数值均为相对 Base 的近似/图中标注倍数：

| Model | Wafer | Batch | Pred Only | Allo Only | Allo+Pred | 主要观察 |
|---|---|---:|---:|---:|---:|---|
| **DeepSeek-V3** | 5×5 | 4096 | 3.4× | 7.9× | **8.1×** | allocation 已接近最优，prediction 提供小幅补强 |
| **DeepSeek-V3** | 8×3 | 4096 | 3.7× | 10.2× | **10.3×** | 图中最强的一组收益之一 |
| **Kimi-K2** | 5×5 | 4096 | 3.7× | 8.2× | **8.3×** | expert 数多，受益明显 |
| **Kimi-K2** | 8×3 | 4096 | 4.0× | 10.4× | **10.5×** | 对跨 die locality 优化高度敏感 |
| **Llama-4-Maverick** | 5×5 | 4096 | 2.8× | 6.5× | **6.7×** | 稳定获益，但幅度低于 DeepSeek/Kimi |
| **Llama-4-Maverick** | 8×3 | 4096 | 3.1× | 8.4× | **8.5×** | 长距离拓扑下 allocation 价值更高 |
| **Qwen3-235B** | 5×5 | 4096 | 2.7× | 4.8× | **5.4×** | 四模型中相对收益最小 |
| **Qwen3-235B** | 8×3 | 4096 | 3.0× | 6.2× | **6.9×** | 仍显著优于 Base 和 EP |

- 模型之间存在明显差异：

| 模型 | 图中表现 | 原因解释 |
|---|---|---|
| **DeepSeek-V3** | 高收益，尤其在 8×3、batch=4096 时超过 10× | experts 更多、routing 更复杂、原始跨 die movement 更严重，因此优化空间更大 |
| **Kimi-K2** | 与 DeepSeek 类似，组合方案可达约 10.5× | 大规模 expert configuration 对 placement 与 locality 更敏感 |
| **Llama-4-Maverick** | throughput 改善稳定，但不及前两者 | 图中显示其通信优化空间大，但 compute/load 等因素更早成为限制 |
| **Qwen3-235B** | 收益最低，约 2×–7× | expert 数量较少，默认 placement 的通信与负载问题相对温和 |

- **Wafer topology 会放大优化价值。**
  - 在多数模型和 batch size 下，**8×3 Wafer** 的收益高于 **5×5 Wafer**。
  - 例如 DeepSeek-V3 的 batch=4096：
    - 5×5 上 **Allo+Pred = 8.1×**；
    - 8×3 上 **Allo+Pred = 10.3×**。
  - 原因是 8×3 mesh 更狭长，平均 Manhattan distance 更大；传统 Base 的远程访问需要经过更多 hops。因此，**data-placement-aware allocation 对不规则、长距离拓扑更有价值**。

- **batch size 越大，throughput 加速通常越低，但通信 hop 的减少仍然极其显著。**
  - 以 DeepSeek-V3 的 5×5 Wafer 为例，Allo+Pred 大致从：
    - batch=4096 的 **8.1×**，
    - 降至 batch=8192 的约 **6×–7×**，
    - 再降至 batch=16384 的约 **3×–4×**。
  - 这不代表优化失效，而是说明瓶颈发生迁移：
    - 小 batch 时，remote HBM access 和 D2D communication 主导，优化可大幅提升吞吐。
    - 大 batch 时，更多 expert requests 被分摊，且 computation、local DRAM bandwidth、负载分配本身开始成为主要瓶颈。
  - 因此，**通信减少倍数远大于最终 throughput 增幅**是合理现象。

- 下半图的 **Hop Reduction** 最能揭示该论文方案的机制性价值：

| 策略 | Hop Reduction 特征 | 含义 |
|---|---|---|
| **Pred Only** | 大多约 **3.8×–5.2×** | prediction 将部分未来远程 expert access 转化为 local HBM access |
| **Allo Only** | 大多约 **42×–410×** | 将请求优先调度给持有相关 experts 的 local/nearby dies，显著消除远程 D2D requests |
| **Allo+Pred** | 大多约 **75×–480×** | allocation 先解决主体通信，prediction 再处理少量无法避免的热点远程访问 |

- 典型 hop-reduction 数据表明，**Allo Only 的效果远超 Pred Only**：
  - DeepSeek-V3，5×5，batch=4096：
    - Pred Only：**4.4×**；
    - Allo Only：**186.8×**；
    - Allo+Pred：**189.3×**。
  - Llama-4-Maverick，8×3，batch=4096：
    - Pred Only：**4.2×**；
    - Allo Only：**410.9×**；
    - Allo+Pred：**479.9×**。
  - Kimi-K2，5×5，batch=4096：
    - Pred Only：**4.7×**；
    - Allo Only：**226.6×**；
    - Allo+Pred：**227.5×**。

- 图中一个重要的非线性现象是：
  - **Allo+Pred 的 hop reduction 可达数百倍，但吞吐只提升约 6×–10×。**
  - 例如 Llama-4-Maverick 在 8×3、batch=4096 下，Allo+Pred 的 hop reduction 达 **479.9×**，但 throughput 为 **8.5×**。
  - 这说明提出的 allocation 机制已经将 D2D traffic 从主导瓶颈降为次要因素；之后性能受制于：
    - **expert GEMM computation**；
    - **local HBM bandwidth**；
    - **热门 expert 的剩余 load imbalance**；
    - kernel launch、LLC、调度等固定开销。
  - 换言之，图不仅证明方案减少通信，也证明其成功改变了系统的主导瓶颈。

- **EP 的定位值得特别注意。**
  - EP 通过让 computation 靠近 expert weights，避免了部分远程访问。
  - 但其 throughput 通常弱于 **Allo Only** 和 **Allo+Pred**，原因是 EP 过于刚性：
    - hot experts 所在 die 可能拥塞；
    - cold-expert dies 可能闲置；
    - 无法利用邻近 die 的计算资源来分摊热门 experts。
  - **Allo Only** 在“保持 locality”和“动态负载均衡”之间做了更优折中：优先选择 local die，同时允许必要时在 nearby dies 间分割请求。

- 图像直接验证了论文的三项关键设计判断：
  - **Insight 1 / Insight 2：expert selection 存在 temporal predictability。**
    - Pred Only 持续获得约 3× 左右吞吐收益和约 4×–5× hop reduction，证实 cross-token heatmap 可以有效指导 expert caching。
  - **Insight 3：expert-placement-aware workload distribution 是核心。**
    - Allo Only 带来数十到数百倍 hop reduction，且吞吐通常超过纯预测策略两倍以上。
  - **Prediction 与 allocation 互补，而非替代。**
    - allocation 解决“大多数请求应在哪里执行”的问题；
    - prediction 解决“少数仍需远程访问的热门 experts 是否应提前复制”的问题；
    - 因而 **Allo+Pred** 几乎在所有配置下取得最高或并列最高表现。

- 对图中结果的严谨解读还应注意：
  - **Base 不使用 proposed HW，而 EP/Pred/Allo/Allo+Pred 使用 proposed HW**。因此，Base 到其他方案的差距反映的是“硬件扩展 + 调度/预测策略”的联合收益。
  - **Allo Only 与 Allo+Pred 的比较**更适合衡量 predictor 的增量价值。
  - **EP 与 Allo Only 的比较**更能衡量 placement-aware dynamic allocation 相对静态 expert-local execution 的价值。
  - 该图评测的是 **decode-stage MoE layer throughput**，不等同于端到端 LLM latency；attention、All-to-All、top-k、prefill 等开销不在该指标内。

- 综合而言，该图的核心信息是：MoE expert routing 并非完全随机。通过 **expert-placement-aware allocation** 消除大部分无效 D2D movement，再通过 **data-driven prediction** 将残余热点 expert 缓存在 local HBM，系统可在 24/25-die wafer-scale GPU 上获得显著而一致的 MoE decode throughput 提升；其中，**跨 die communication 越昂贵、模型 expert 数越多、拓扑越不规则，收益越大**。

### 6a17a43122360612904c20f526a5f4790710ad64012b153ae7bb5f7afc62819e.jpg

![6a17a43122360612904c20f526a5f4790710ad64012b153ae7bb5f7afc62819e.jpg](images/6a17a43122360612904c20f526a5f4790710ad64012b153ae7bb5f7afc62819e.jpg)

- 图片对应论文 **Figure 13**，核心目的为验证自研的 event-driven multi-chiplet GPU simulator 是否能准确复现真实 **8×H100 DGX** 上的两类关键行为：
  - **单 GPU 上 MoE expert 的计算时间**；
  - **两 GPU 间 P2P 数据迁移的通信带宽**。
  - 图例表明：**蓝色**为真实 H100 DGX 测量值（Measurements），**粉色**为模拟器输出（Reported by Our Simulator）。

- 图由上下两个子图构成，验证范围覆盖 MoE serving 的两个主要成本来源。

| 子图 | 验证对象 | 横轴 | 纵轴 | 主要结论 |
|---|---|---|---|---|
| 上图 | MoE expert computation | batch size / token 数，16–8192 | MoE Time（ms） | 模拟器准确复现 DeepSeek V3 与 Qwen 3 的计算延迟增长趋势 |
| 下图 | GPU P2P communication | Data Block Size，4 KB–4 GB | Actual BW（GB/s） | 模拟器准确复现小消息低带宽、消息增大后饱和的通信规律 |

- 上图的横轴从 **16、32、64** 一直到 **8192**，表示处理同一个 expert 的 token/batch 规模逐步扩大；纵轴为单个 MoE expert 所包含的三次 GEMM 的耗时，范围约为 **0–1.5 ms**。

- 上图中比较了两个模型，其计算规模和延迟特征明显不同：

| 模型 | 图中标记/走势 | 约略延迟范围 | 含义 |
|---|---|---:|---|
| **DeepSeek V3** | 方形点，曲线上升更快 | batch 16 时接近 0；8192 时约 **1.0–1.1 ms** | 单 expert 的计算量更高，随着 batch 增大，GEMM 计算时间显著提升 |
| **Qwen 3** | 圆形点，增长更平缓 | batch 16 时接近 0；8192 时约 **0.5 ms** | 每层 expert 计算相对较轻，因此大 batch 下时延约为 DeepSeek V3 的一半 |

- **DeepSeek V3 的曲线**在 batch size 约 **1024 之后**增长明显：约从 0.1 ms 增至 2048 时的约 0.25 ms、4096 时的约 0.5 ms，以及 8192 时的约 1 ms。这表明在较大 batch 下，expert GEMM 逐步从启动开销主导转向吞吐/算力主导。

- **Qwen 3 的曲线**保持更低的斜率：3072、4096、5120、6144、7168、8192 等点大致呈持续平缓增长。这一差异也解释论文后续的观察：Qwen3 有更多 MoE layers（94 vs. 58）但单层计算较短，因此一些固定调度或 CPU–GPU 往返开销会占据更大的相对比例。

- 对上图最关键的观察是：两种模型中，粉色模拟曲线几乎覆盖蓝色真实测量曲线。尤其在中高 batch 区间，模拟器不但拟合了总体上升趋势，也保留了两模型之间的**绝对时延差异**与**增长斜率差异**。这说明模拟器对以下因素的抽象是有效的：
  - expert GEMM 的计算成本；
  - batch size 对计算效率和时延的影响；
  - 不同模型 expert 形状/规模导致的计算差异。

- 下图验证 GPU-to-GPU 的 **P2P data migration**。横轴采用对数式离散尺度，数据块从 **4 KB、16 KB、64 KB、256 KB、1 MB、4 MB、16 MB、64 MB、256 MB、1 GB 到 4 GB**；纵轴为实际带宽，最高约 **400 GB/s**。

- 下图呈现典型的通信带宽曲线：

| 消息规模区间 | 实测带宽特征 | 系统含义 |
|---|---|---|
| **4 KB–64 KB** | 带宽接近 0 GB/s | 固定通信启动延迟占主导，小粒度迁移效率很低 |
| **256 KB–4 MB** | 带宽快速爬升 | payload 增大后，启动开销被摊薄，链路利用率提升 |
| **16 MB–64 MB** | 带宽达到约 280–350 GB/s | 链路进入高利用率区间 |
| **256 MB–4 GB** | 饱和在约 380–400 GB/s | 通信受 H100 DGX P2P/NVLink 路径的稳态带宽限制 |

- 下图中蓝、粉两条曲线从小数据块到大数据块都高度重合。模拟器不仅复现了大数据传输时接近 **400 GB/s** 的饱和带宽，也复现了中小数据块下的带宽爬升阶段。这很重要，因为 MoE inference 的跨设备流量并不总是大块连续传输：expert dispatch、weight migration、remote HBM access 都可能产生不同粒度的数据移动。

- 该图支撑了论文模拟方法的可信度。论文明确指出，在所有测试点上，模拟误差控制在 **5% 以内**。从图形上看，少量可见偏差主要集中在：
  - 上图的中等 batch 区域，例如 DeepSeek V3 在约 4096–6144 的区间；
  - 下图的带宽爬升区间，例如 1–16 MB；
  - 这些位置本身通常更易受到 kernel scheduling、缓存状态、通信协议分段以及测量噪声影响。

- 这张图对后续 **wafer-scale GPU** 结论的作用是基础性的。论文后续报告的 **Allo Only、Pred Only、Allo+Pred** 吞吐收益均来自模拟器，因此先证明以下两点是必要前提：
  - **计算模型正确**：能反映 MoE expert 在不同 batch size 下的实际执行时间；
  - **通信模型正确**：能反映不同数据迁移粒度下的 P2P 带宽行为。

- 与论文主张的关系可概括如下：

| 论文问题 | Figure 13 提供的证据 | 后续设计含义 |
|---|---|---|
| MoE 的瓶颈是否可被模拟？ | MoE Time 与实测高度一致 | 可评估 task allocation 对计算负载均衡的影响 |
| D2D/P2P 流量是否可被模拟？ | Actual BW 与实测高度一致 | 可评估 expert placement、remote access、HBM duplication 的收益 |
| 为什么降低 hop count 可能提高性能？ | 通信带宽存在启动区与饱和区 | 减少跨 die 传输可避免带宽竞争与远程访问时延 |
| 为什么需硬件内的 Global/Local CP？ | MoE 计算可能很短，尤其是 Qwen3 | CPU 参与调度时，固定往返开销会侵蚀优化收益 |

- 图片也反映出一个值得注意的边界：该验证主要覆盖 **单 expert GEMM** 与 **双 GPU P2P transfer**，而非对完整 24/25-die wafer 的端到端逐周期验证。因此，图能够强力证明 simulator 的核心计算和基础通信参数已校准，但对于超大规模 mesh 中的：
  - 多源并发远程 HBM 访问；
  - 网络拥塞；
  - 多跳 XY routing；
  - 跨 die 的负载不均衡；
  
  仍主要依赖事件驱动模型的资源竞争抽象。这也是论文采用 “validated simulator” 而不是宣称完整 cycle-accurate wafer prototype 的合理之处。

- 总体而言，这是一张**方法学可信度验证图**，而非性能优化结果图。其最重要结论是：自研模拟器在 **MoE compute latency** 和 **P2P communication bandwidth** 两个决定 MoE data movement 性能的基础维度上，与真实 **8×H100 DGX** 结果高度一致；因此，论文以该模拟器推演未来 wafer-scale GPU 上的 **6.6× 平均加速**具有较好的实验依据。

### d0a933b4c4cd3c7d82cec4508ce588e5d7bac934075ad4fe3a04ebfde801be34.jpg

![d0a933b4c4cd3c7d82cec4508ce588e5d7bac934075ad4fe3a04ebfde801be34.jpg](images/d0a933b4c4cd3c7d82cec4508ce588e5d7bac934075ad4fe3a04ebfde801be34.jpg)

- 图展示 **Qwen3、TSMC-SoW、batch size = 4096** 下，不同策略的 **DRAM Access（GB）** 构成；横轴为策略，纵轴为累计 DRAM 访问量。

- 堆叠柱的三类访问含义如下：

| 颜色 | 指标 | 含义 |
|---|---:|---|
| 深绿色 | **Local Rd** | 从执行计算的 die 的本地 HBM 读取 expert weights |
| 中绿色 | **Rmt Rd** | 从远端 die 的 HBM 读取，需经过 D2D 通信 |
| 浅绿色 | **Local Wt** | 将远端 expert 副本写入本地 HBM，属于预测缓存带来的额外写流量 |

- 各策略的视觉估算与变化趋势如下；图中没有标注精确数字，因此数值为近似读图结果。

| 策略 | Local Rd | Rmt Rd | Local Wt | 总 DRAM 访问量 | 核心特征 |
|---|---:|---:|---:|---:|---|
| **Base** | 接近 0 | 约 17–18 TB | 0 | 约 17–18 TB | 几乎完全依赖远端读取 |
| **Pred Only** | 约 7–8 TB | 约 9–10 TB | 约 8–9 TB | 约 26 TB | 预测驱动复制，将部分远端读转为本地读，但有明显写入成本 |
| **Allo Only** | 约 12 TB | 约 6 TB | 0 | 约 18–19 TB | 任务分配优先在 expert 所在 die 上执行，显著减少远端读取 |
| **Allo+Pred** | 约 17–18 TB | 约 0.5–1 TB | 约 1–2 TB | 约 19 TB | 分配与预测结合，远端读取几乎被消除 |

- **Base 的问题最突出**：
  - 其访问量几乎全部是 **Rmt Rd**。
  - 这意味着每次 expert weight 获取都频繁穿越 die-to-die 网络。
  - 在 wafer-scale mesh 中，远端 HBM 访问不仅有更高延迟，也会带来链路拥塞、竞争和多跳传输，因此是 MoE 性能瓶颈。

- **Pred Only 的作用与代价**：
  - 它借助 temporal correlation 预测下一 token 可能使用的 experts，并提前复制到本地 HBM。
  - 因而 **Rmt Rd 从约 17–18 TB 降至约 9–10 TB**，大量访问转化为 **Local Rd**。
  - 但预测式复制也引入较大的 **Local Wt**；总 DRAM 流量反而增至约 26 TB。
  - 这说明该机制优化的并不是“总字节数最小化”，而是更关键的 **远端访问与 D2D traffic 最小化**。本地 HBM 写入虽有代价，但远低于反复远端读取的系统级代价。

- **Allo Only 是最有效的流量源头治理手段**：
  - 通过 data-placement-aware task allocation，把某 expert 的 token 计算尽量调度到保存该 expert weights 的本地 die。
  - 相比 Base，**Local Rd 大幅增加**，而 **Rmt Rd 降至约三分之一**。
  - 它不需要复制权重，因此没有 **Local Wt**。
  - 这一结果直接验证论文的 **Insight 3: Expert-placement-aware workload distribution**：已知 expert placement 后，任务调度本身就能显著降低跨 die 数据移动。

- **Allo+Pred 达到最优的远端访问抑制**：
  - **Allo** 先消除大多数“本不该发生”的远端计算与远端读取。
  - **Pred** 再处理仍不可避免的少量远端热门 expert：将其缓存/复制至当前 die 的 local HBM。
  - 因此，图中 **Rmt Rd 几乎消失**，只剩很小的一段。
  - 虽然出现了少量 **Local Wt**，且总 DRAM 访问量可能略高于 Allo Only，但访问性质已从昂贵的远端通信转换为高带宽、低延迟的本地 HBM 访问。

- 右侧放大框的目的，是强调 **Allo Only 与 Allo+Pred 的总柱高接近**：
  - 两者总 DRAM 流量都在约 **18–19.5 TB** 的窄区间。
  - 但 Allo+Pred 的组成明显更优：它用少量本地写入替代了相当一部分远端读取。
  - 因此，性能收益不能仅由总 DRAM access 判断；更应关注 **Rmt Rd 占比、D2D hop、链路拥塞和访问局部性**。

- 图所支持的系统结论是：

| 结论 | 图中证据 | 对应论文洞察 |
|---|---|---|
| **远端 HBM 读取是关键瓶颈** | Base 主要由 Rmt Rd 构成 | MoE data movement 主导延迟 |
| **任务调度比盲目缓存更基础** | Allo Only 已显著压缩 Rmt Rd，且不产生复制写入 | **Insight 3** |
| **预测缓存适合补足调度无法解决的热点访问** | Allo+Pred 进一步将 Rmt Rd 压至接近零 | **Insight 1、Insight 2** |
| **应优化通信位置而非只优化总流量** | Pred Only 总访问量更高但远端读更少；Allo+Pred 用 Local Wt 替代 Rmt Rd | Locality-aware architecture |
| **硬件需要支持本地 HBM 副本与地址重定向** | Allo+Pred 的 Local Wt/Local Rd 转换依赖复制与复用 | PDU、ATU、hardware-managed HBM |

- 从架构角度看，该图是对论文软硬件协同设计的直接验证：
  - **Global CP / Local CP**：负责根据 expert placement 和请求量执行 **Allo**。
  - **Prediction Unit (PDU)**：依据历史 expert-selection heatmap 决定哪些远端 experts 应复制。
  - **Address Translation Unit (ATU)**：在副本已经存在时，将远端地址请求重定向至 local HBM。
  - 三者协同后，MoE weight access 从“跨 die 拉取”为主，转变成“本地读取”为主。

### b26d10d5d8f2a8567ffbf8f91c80a6250ec5491e2f59fe3fdaca6408b64a6f91.jpg

![b26d10d5d8f2a8567ffbf8f91c80a6250ec5491e2f59fe3fdaca6408b64a6f91.jpg](images/b26d10d5d8f2a8567ffbf8f91c80a6250ec5491e2f59fe3fdaca6408b64a6f91.jpg)

- 图像为论文 **Figure 15: “Host CPU implementation overhead under varying models and batch sizes”**。其核心目的是比较：若将 MoE task allocation algorithm 放在 **Host CPU** 而非 GPU 内部的 **Global Command Processor** 执行，会给端到端执行带来多大的额外开销。

- 横轴包含两种模型、两种 batch size：
  
  | 模型与 batch size | Dojo | Dojo-Enhanced |
  |---|---:|---:|
  | DS V3 (256) | 约 **6%** | 约 **24%** |
  | DS V3 (8192) | 约 **5%** | 约 **19%** |
  | Qwen3 (256) | 约 **14%** | 约 **51%** |
  | Qwen3 (8192) | 约 **10%** | 约 **42%** |

- 图例含义：
  - 浅粉色柱：原始 **Dojo** 配置。
  - 深橙红色柱：计算能力和内存带宽更强的 **Dojo-Enhanced** 配置。
  - 纵轴为 **Overhead**，范围约为 0%–50%；柱越高表示 Host CPU 方案侵蚀性能越严重。

- 从模型维度看，**Qwen3 的 CPU-side allocation 开销显著高于 DeepSeek V3（DS V3）**：
  - 在 Dojo 上，Qwen3 的开销约为 **10%–14%**，而 DS V3 约为 **5%–6%**。
  - 在 Dojo-Enhanced 上，Qwen3 达到约 **42%–51%**，DS V3 为约 **19%–24%**。
  - 这与正文解释一致：Qwen3 有 **94 个 MoE layers**，多于 DeepSeek V3 的 **58 个 MoE layers**。Host CPU 每层均需与 GPU 交换 Expert Distribution Table 和 allocation plan，因此 layer 数更多意味着更频繁的 **PCIe transfer** 与同步。

- 从硬件维度看，**Dojo-Enhanced 的 CPU 实现开销远大于 Dojo**：
  
  | 对比项 | DS V3 | Qwen3 |
  |---|---:|---:|
  | Dojo overhead | 约 5%–6% | 约 10%–14% |
  | Dojo-Enhanced overhead | 约 19%–24% | 约 42%–51% |
  | 增幅趋势 | 约 3–4 倍 | 约 3.6–4.2 倍 |

  - Dojo-Enhanced 的单 die 计算性能为 **4500 TFLOPS FP16**，明显高于 Dojo 的约 **989 TFLOPS FP16**。
  - GPU 计算与本地 HBM 访问变快后，CPU-GPU 间相对固定的 PCIe 往返成本无法同步缩短，因而其占比急剧上升。
  - 该图直观体现了一个关键系统规律：**计算硬件越快，控制路径中的跨设备通信和同步越容易成为瓶颈。**

- batch size 的影响表现为：从 256 增至 8192 后，所有配置的 overhead 都有所下降。
  - DS V3：Dojo 从约 6% 降至 5%，Dojo-Enhanced 从约 24% 降至 19%。
  - Qwen3：Dojo 从约 14% 降至 10%，Dojo-Enhanced 从约 51% 降至 42%。
  - 原因是大 batch 提供了更多 MoE computation，可部分摊薄每层固定的 CPU-GPU 控制、数据传输和同步成本。
  - 但下降幅度有限，尤其在 Dojo-Enhanced + Qwen3 情形下仍超过 **40%**，说明仅靠扩大 batch 不能从根本上解决控制路径瓶颈。

- 图中最严重的点是 **Qwen3 (256) + Dojo-Enhanced**，开销约 **51%**：
  - 这意味着 Host CPU 调度近乎额外消耗了原任务执行时间的一半。
  - 对低 batch 或低延迟服务场景，这种开销尤其不可接受，因为调度本身可能抵消大部分硬件性能升级收益。

- 图中最轻微的点是 **DS V3 (8192) + Dojo**，开销约 **5%**：
  - 在较慢的基础硬件和较大 batch 下，Host CPU 实现仍可能是可容忍的工程折中。
  - 但该结论不应推广到未来高性能 wafer-scale GPU，因为图中 Dojo-Enhanced 已显示这种折中会快速失效。

- 该图直接支撑论文的架构结论：应将 task allocation algorithm 部署在 GPU 内部的 **Global Command Processor**，而不是 Host CPU。
  - 这样可消除或显著缩减每个 MoE layer 的 PCIe 往返。
  - 这也解释了论文提出两级 Command Processor 的必要性：**Global CP** 持有全局 expert placement 和 routing 信息，**Local CP** 负责本地执行分派，从而避免控制决策跨越 Host–GPU 边界。
  - 结合 Table II，论文估计这一硬件扩展的面积和功耗开销均约为整块 25-die wafer 的 **0.04%**；相对于图中最高约 **51%** 的性能损失，体现出较高的成本效益。

- 图的主要结论可概括为：**Host CPU 调度在当前较慢硬件上尚可接受，但在未来高算力 wafer-scale GPU 上会成为严重瓶颈；模型 MoE layer 越多、batch 越小、GPU 越快，越需要将 allocation logic 下沉到片上 Command Processor。**

### Figure 16. Demonstration of expert placement strategies.

![c6147e5c0a9ea6edd0e67214ad09512bb65b98c4fc8c2ba8a4546e216cb999cb.jpg](images/c6147e5c0a9ea6edd0e67214ad09512bb65b98c4fc8c2ba8a4546e216cb999cb.jpg)

- **图16展示了两类基于 Prefill trace 的 expert placement 策略**，用于缓解 MoE decode 阶段因 Expert Popularity 不均导致的 GPU workload imbalance。示例包含 **2 个 GPU、8 个 experts（E0–E7）**。

- 图例中的颜色表示 expert 热度，而非 expert 类型：

| 视觉元素 | 含义 | 图中示例 |
|---|---|---|
| 深红色 | 激活频率最高、最热门的 expert | **E7** |
| 浅红色 | 次热门 expert | **E6** |
| 蓝色 | 中等热度 expert | **E1** |
| 灰色 | 较低热度 expert | E0、E2、E3、E4、E5 |
| 绿色虚线/箭头 | `Extra-Slots` 与复制后的动态 token 分流路径 | Dup-Based 中的跨 GPU 副本 |

- **Default placement** 是常规 contiguous placement：按 Expert ID 连续、均匀地将 experts 切分到 GPU。

| GPU | Default experts | 问题 |
|---|---|---|
| GPU 0 | E0, E1, E2, E3 | 仅有中等热度的 E1，其余较冷 |
| GPU 1 | E4, E5, E6, E7 | 同时包含热门 E7 与次热门 E6 |

- Default 的核心缺陷是：**存储容量均衡，不等于计算负载均衡**。虽然两张 GPU 均保存 4 个 experts，但 GPU 1 汇集了 E7 和 E6，因而承担更多 routed tokens；GPU 0 则可能较早完成计算并空闲。MoE layer 的整体 completion time 由 GPU 1 决定，形成 straggler。

- **Remap-Based placement** 通过迁移、交换 experts 来实现负载再平衡，且**不增加模型副本、不改变每张 GPU 的 expert 容量**。

| GPU | Default | Remap-Based | 变化与目的 |
|---|---|---|---|
| GPU 0 | E0, E1, E2, E3 | E0, **E7**, E2, E3 | 将最热 E7 移入原本较空闲的 GPU 0 |
| GPU 1 | E4, E5, E6, E7 | E4, E5, E6, **E1** | 将较冷的 E1 换入 GPU 1 |
| 每 GPU 容量 | 4 | 4 | **容量不变** |

- 图中的红色虚线和绿色箭头表示一次 **E7 ↔ E1 的交换**。其原则是：根据 Prefill 阶段统计的 expert frequency，按预计计算成本排序，将高成本 experts 贪心地分散给当前负载最低的 GPU。

- Remap-Based 的优缺点如下：

| 维度 | 分析 |
|---|---|
| 优势 | **无需额外 HBM expert slots**；可直接降低最忙 GPU 的 MoE GEMM 负担 |
| 适用场景 | GPU memory 紧张、无法额外复制 expert weights 的 Expert Parallelism 系统 |
| 代价 | 需要改变初始 expert-to-GPU mapping；若 placement 重配频繁，可能产生权重迁移和服务扰动 |
| 本质 | **“移动 expert”来均衡负载** |

- **Dup-Based placement** 保留 Default 的原始 placement，但利用每张 GPU 的 `Extra-Slots` **复制非本地的热门或有价值 experts**。图中 GPU 0 额外保存 E7，GPU 1 额外保存 E1。

| GPU | 原始 experts | Extra-Slot 副本 | 最终可执行 experts |
|---|---|---|---|
| GPU 0 | E0, E1, E2, E3 | **E7** | E0, E1, E2, E3, E7 |
| GPU 1 | E4, E5, E6, E7 | **E1** | E4, E5, E6, E7, E1 |

- Dup-Based 中，**E7 在 GPU 0 和 GPU 1 上各有一个副本**。因此，原先全部发往 GPU 1 的 E7 tokens 可以被动态切分并分发到两张 GPU，直接削弱 E7 导致的 hotspot。绿色箭头表示这种基于副本的 token redistribution。

- 图中也将 E1 放入 GPU 1 的 Extra-Slot。这表明 Dup-Based 不仅是“复制全局最热 expert”，而是考虑**复制后各 GPU 的最大预计负载是否下降**。当 GPU 0 承接一部分 E7 工作时，复制 E1 到 GPU 1 可为 E1 token 的调度提供更多选择，从而进一步降低最终 bottleneck load。

- Dup-Based 的优缺点如下：

| 维度 | 分析 |
|---|---|
| 优势 | 不必破坏默认 contiguous layout；通过副本直接提供额外计算并行度 |
| 适用场景 | 有可用 HBM 余量，且 serving backend 支持 replicated experts 的动态 token dispatch |
| 代价 | 消耗额外显存；副本数量受 `Extra-Slots` 限制 |
| 关键机制 | 对某 expert 的 tokens 在其多个副本间**均匀分流** |
| 本质 | **“复制 expert”来扩展热点的服务能力** |

- 两种策略的核心对比如下：

| 属性 | Remap-Based | Dup-Based |
|---|---|---|
| 是否改变默认 expert placement | 是 | 否，保留原 placement |
| 是否增加 expert 副本 | 否 | 是 |
| 单 GPU expert 数量 | 固定为 \(E/G\) | 可增加至 \(E/G+R\) |
| 显存额外需求 | 基本无 | 需要 `Extra-Slots` 对应的 HBM |
| 热点缓解方式 | 将热点 expert 分散到不同 GPU | 为热点 expert 增加多个服务副本 |
| 主要约束 | 重映射/迁移成本 | 显存容量与副本调度能力 |

- 该图直接对应论文的 **Insight 1: Prefill-data-driven prediction**：Prefill 与 Decode 的 expert frequency 高度相关，因此系统可以在 Decode 初始阶段尚无足够历史统计时，使用 Prefill trace 预先判断 E7、E6、E1 等热点 experts，并完成 remap 或 duplication。

- 从系统角度看，图16强调的不是减少 All-to-All 的通信本身，而是降低 **MoE computation 的最慢 GPU 执行时间**。论文随后在 8×H100、Qwen3-235B 上验证：Prefill-guided 的 **Remap** 和 **Dup** 相比 Default 分别带来 **15.5%** 与 **12.5%** 的 MoE computation speedup，且与使用 oracle Decode 信息的 Best placement 相差不足约 10%。

### 84d1b3da13153389b341c254c391ffe8dc728a28fd7fa5cbb18deadc54ee2b99.jpg

![84d1b3da13153389b341c254c391ffe8dc728a28fd7fa5cbb18deadc54ee2b99.jpg](images/84d1b3da13153389b341c254c391ffe8dc728a28fd7fa5cbb18deadc54ee2b99.jpg)

- 图中评估 **Prefill-Guided Expert Placement** 在真实 **8×H100、Qwen3-235B** 集群上的效果；纵轴为归一化的 **MoE Speedup**，以默认专家布局 **Default = 1.00×** 为基准，横轴为 Batch Size（4K、8K、16K、24K）。

- 图例中各方案含义如下：

| 方案 | 含义 | 是否实际可用 |
|---|---|---|
| **Default** | Qwen/SGLang 默认连续专家布局，每 GPU 放置连续编号的 experts | 是 |
| **Worst** | 利用 oracle decode trace 构造的理论最差布局 | 否，仅作下界 |
| **Best** | 利用 oracle decode trace 构造的理论最优布局 | 否，仅作上界 |
| **Remap (Ours)** | 基于 Prefill trace，重新映射 experts 至各 GPU，保持每 GPU 专家数量不变 | 是 |
| **Dup (Ours)** | 基于 Prefill trace，对热点 experts 进行复制；每 GPU 额外保留一个 slot | 是 |

- 各 Batch Size 的定量结果为：

| Batch Size | Default | Worst | Best | Remap (Ours) | Dup (Ours) |
|---:|---:|---:|---:|---:|---:|
| **4K** | 1.00× | 0.80× | 1.33× | **1.25×** | **1.22×** |
| **8K** | 1.00× | 0.56× | 1.20× | **1.14×** | **1.04×** |
| **16K** | 1.00× | 0.51× | 1.25× | **1.11×** | **1.04×** |
| **24K** | 1.00× | 0.57× | 1.28× | **1.12×** | **1.20×** |

- 图的核心结论是：**Prefill 阶段收集的 expert-selection 信息足以有效预测 Decode 阶段的热点 experts**。即便没有访问真实 Decode routing trace，两种 proposed 方法都稳定优于 Default，说明论文的 **Insight 1: Prefill-data-driven prediction** 在实际 GPU 系统上成立。

- **Remap (Ours)** 的表现最稳定：
  - 在四个 batch 下均获得正收益，范围为 **1.11×–1.25×**。
  - 平均加速约为 **1.155×**，即约 **15.5%**，与正文报告一致。
  - 4K 时达到 **1.25×**，接近 oracle 的 Best（1.33×）；其与 Best 的差距仅约 **6%**。
  - 它不需要新增 expert replicas，因此更适用于 **显存容量紧张**、不希望改变每 GPU expert capacity 的部署环境。

- **Dup (Ours)** 的平均加速约为 **1.125×**，即约 **12.5%**：
  - 在 4K 和尤其 24K batch 时效果突出，分别达到 **1.22×** 与 **1.20×**。
  - 24K 下，Dup 与 Best 的 **1.28×** 很接近，表明复制热点 experts 能够有效降低最忙 GPU 的计算拥塞。
  - 8K 和 16K 时仅为 **1.04×**，增益较弱，说明复制策略高度依赖 batch 内实际的热点强度、expert 频率偏斜程度及 replica 分流效果。
  - 该方法以额外显存为代价：实验中每层从 **128 个 expert slots** 扩展至 **136 个**，即 8 个 GPU 各增加一个可放置 replica 的 slot。

- **Best/Worst** 两组 oracle 基线说明 expert placement 对 MoE performance 的潜在影响很大：
  - Worst 在 8K、16K 时只有 **0.56×、0.51×**，意味着糟糕的布局可使 MoE 计算时间接近翻倍。
  - Best 可达 **1.20×–1.33×**，表示仅通过更优的 placement，理论上可带来约 **20%–33%** 性能提升。
  - Remap 和 Dup 通常距离 Best 不超过约 **10%–17%** 的相对速度差；论文正文概括为“within 10% of Best”，图中最接近的情形是 4K Remap 与 24K Dup。

- Default 在所有 batch 中固定为 1.00×，而 Worst 随 batch 增大显著下降，反映出：**较大的 batch 会放大 expert activation skewness 导致的 GPU workload imbalance**。当热点 experts 被集中部署在少数 GPU 时，其他 GPU 会等待最忙 GPU 完成 expert GEMM，形成 MoE computation 的 tail-latency bottleneck。

- Remap 与 Dup 的差异揭示两类系统设计取舍：
  - **Remap**：直接把热点 experts 分散到不同 GPU，优先实现负载均衡；不额外占显存，性能更稳定。
  - **Dup**：保留原始默认布局并增加热点副本，通过 replica 分流降低热点 GPU 压力；避免大范围迁移，但需要额外显存，且对 workload 分布更敏感。
  - 因此，**显存受限时优先 Remap；有冗余显存且热点在大 batch 下明显集中时，Dup 具有竞争力。**

- 需要注意该图的指标边界：
  - 测量的是 **MoE computation time**，即三个 expert linear layers 的执行时间。
  - **不包括** attention、top-k routing、All-to-All communication 等时间。
  - 因而图中加速不能直接等价为完整 LLM end-to-end latency 或 token throughput 的同等比例提升；它直接证明的是 placement 对 **expert-side compute load balance** 的改善。

- 图中没有显示 error bars 或置信区间，因此无法仅凭该图判断不同 batch 下小幅差异（如 1.04× 对 1.00×）的统计显著性。不过结合论文的真实 GPU 实验设置及其报告的约 ±5% timing variation，**Remap 的 11%–25% 增益更具稳健性；Dup 在 8K/16K 的 4% 增益则应谨慎解释。**

