# TileBench: A Controlled Benchmark for Performance Evaluation and Bottleneck Diagnosis of Tile-Based Programming Models 通俗讲解

### 0. 整体创新点通俗解读

**一句话抓住这篇论文**

- **TileBench**不是在发明一个新的 GPU kernel DSL，也不是说 **Triton**或**cuTile**谁“绝对更强”。
- 它真正解决的问题是：
  - 过去大家比较 **tile-based programming models**时，经常是“拿各自最擅长的例子来打擂台”。
  - 结果看起来热闹，但很难回答一个工程上真正重要的问题：
    - **同一个算子、同一台 GPU、相近实现方式、相近调参预算下，Triton 和 cuTile 到底差在哪里？**
- 这篇论文的核心贡献，是搭了一个**受控实验场**：
  - 把 **45 个 AI kernel**放进同一个评测框架；
  - 给每个算子配齐 **PyTorch reference、Triton 实现、cuTile 实现、dtype sweep、size sweep、autotune、roofline 指标、profiling 诊断**；
  - 不只看“谁跑得快”，还看“为什么快、为什么慢、LLM 写起来哪个更省 token”。

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

---

**痛点直击：之前最难受的地方是什么**

- GPU kernel DSL 的比较，过去很容易变成**不公平比较**。
  - **Triton**有成熟生态、很多生产系统案例、pointer-level 表达能力强。
  - **cuTile**更贴近 NVIDIA 新硬件路径，尤其是 **Blackwell B200**上的 **TMA、Tensor Core、tcgen05**。
  - 如果只拿 GEMM 或 attention 这种规则 tile 算子比，cuTile 可能很亮眼。
  - 如果拿 irregular indexing、packing、streaming kernel 比，Triton 又可能明显占优。
  - 问题是：这些结论混在一起，很难分清是**编程模型本身强**，还是**选的算子刚好适合它**。

- 另一个难受点是：只报 latency 没法指导优化。
  - 一个 kernel 慢，可能是：
    - tile size 没调好；
    - compiler lowering 走了错误路径；
    - shared memory bank conflict；
    - Tensor Core 没用上；
    - TMA latency 没被计算覆盖；
    - register pressure 太大；
    - irregular load 被迫走了很重的 gather-mask-stage 链。
  - 只说“慢 1.5×”没有意义。
  - 工程师真正想知道的是：
    - **慢在哪里？该改 tile？改 memory layout？还是等 compiler 改？**

- 还有一个新痛点：LLM 写 kernel 越来越常见，但 DSL 的“可生成性”没人认真量化。
  - 现在很多 benchmark 问的是：
    - LLM 能不能写 CUDA？
    - LLM 能不能写 Triton？
  - 但这篇论文问得更细：
    - **同一个 LLM，在 Triton 和 cuTile 两个目标上，哪个更容易写出正确且快的 kernel？**
    - **哪个更费 token？**
  - 这其实是在评估一个 DSL 的“人类可用性”和“LLM 可用性”。

---

**通俗比方：TileBench 像什么**

- 可以把 **Triton**和**cuTile**想成两种“开车方式”。
  - **Triton**像一辆手感成熟的手动挡车：
    - 你可以比较自由地控制路线；
    - 遇到小路、绕行、临时变道，也能处理；
    - 不一定总能吃满新硬件的自动驾驶能力，但鲁棒性强。
  - **cuTile**像一辆为最新高速公路设计的自动挡赛车：
    - 如果路很直、车道规则、能稳定巡航，它能很好利用新硬件；
    - 但如果路况碎、弯多、临时变道多，它的抽象反而可能别扭。

- **TileBench**做的事情，就是修了一个“标准赛车场”。
  - 以前大家各自找路测：
    - cuTile 在高速公路上测；
    - Triton 在城市道路上测；
    - PyTorch 在普通路况上测。
  - 这样比出来的结果当然说不清。
  - TileBench 把它们拉到同一个赛道：
    - 同一批路线；
    - 同样天气；
    - 同样计时器；
    - 同样规则；
    - 还在车上装了诊断仪，看发动机、刹车、油耗、换挡逻辑。

- 更准确的算法类比：
  - TileBench 不是一个 optimizer，而是一个**controlled ablation framework**。
  - 它像机器学习里的“消融实验平台”：
    - 固定数据；
    - 固定任务；
    - 固定评价指标；
    - 只改变一个关键因素：**programming model**。
  - 这样才能把“模型差异”从“实验噪声”和“实现差异”里剥离出来。

---

**关键一招：作者到底把流程里哪一步换掉了**

- 作者没有提出新的 kernel 编译器。
- 作者也没有给 Triton 或 cuTile 做一个统一中间表示。
- 它最巧妙的地方是：
  - **把“零散 benchmark 比性能”替换成“受控 benchmark 诊断 programming model”**。

- 原来的流程大概是：
  - 找几个热门算子；
  - 各自写一个实现；
  - 跑 latency；
  - 得出“某某 DSL 更快”的结论。

- TileBench 把这个流程扭转成：
  - 每个算子先有一个**PyTorch semantic reference**，定义正确性；
  - 再写**匹配语义、结构尽量可比**的 Triton 和 cuTile 实现；
  - 每个算子都跑多种 **dtype**和多种 **input size**；
  - 同时提供：
    - **default configuration**；
    - **autotuned configuration**；
    - **roofline utilization**；
    - **Nsight Compute/PTX/SASS 诊断**；
    - **LLM-generated kernel track**。
  - 最后回答的不只是“谁快”，而是：
    - **在哪类算子快？**
    - **调参能救多少？**
    - **剩下的性能头寸卡在哪里？**
    - **LLM 更容易写哪个 DSL？**

---

**核心结果：这篇论文真正告诉我们什么**

| 问题 | 直觉结论 | 关键数据 |
|---|---|---|
| **Triton vs PyTorch** | Triton 通常明显优于 PyTorch | **36/45**算子快于 PyTorch，geomean speedup **2.7×** |
| **cuTile vs PyTorch** | cuTile 也通常优于 PyTorch | **34/45**算子快于 PyTorch，geomean speedup **2.2×** |
| **Triton vs cuTile** | Triton 更稳，cuTile 在特定硬件友好算子上强 | cuTile 只在 **11/45**算子上快于 Triton |
| **Autotuning** | 有帮助，但不是万能药 | roofline utilization gain：Triton **1.20×**，cuTile **1.15×** |
| **Roofline headroom** | 大量 kernel 不是 tile size 没调好，而是 compiler/memory path 没走好 | autotune 后达到 **80% roofline**：Triton **8/45**，cuTile **4/45** |
| **LLM 生成 kernel** | Triton 对 LLM 更友好、更省 token | GPT-5.5 TokenEfficiency：Triton **20.9**，cuTile **13.4**；Claude：Triton **17.1**，cuTile **8.3** |

![](c46cba47f4d618a2e72eb29c059a5c2144d83ca4d63ed90b4f4c0c5a37a4b113.jpg)

---

**对 Triton 和 cuTile 的最直观判断**

- **Triton 的强项**
  - 更适合：
    - irregular indexing；
    - runtime-computed pointer；
    - sparse layout；
    - low-bit packing；
    - streaming bandwidth-bound kernels；
    - boundary-heavy control。
  - 原因很直接：
    - Triton 的 **pointer-based model**更像“你直接告诉我每个元素去哪读”。
    - 对不规则访问来说，这种表达很自然。
  - 所以在：
    - **block_sparse_attention**
    - **flash_decode**
    - **linear_self_attention**
    - **weight_dequant**
  - Triton 更鲁棒。

- **cuTile 的强项**
  - 更适合：
    - dense matmul；
    - attention prefill；
    - stencil/conv；
    - tile reuse 明显的算子；
    - 能自然映射到 **TMA + Tensor Core + TMEM**的场景。
  - 原因也直接：
    - cuTile 的抽象更接近 Blackwell 硬件的原生 tile path。
    - 如果问题本身就是规则 tile 搬运、规则 tile 计算，它就像“直接走高速专用车道”。

- **核心 trade-off**
  - Triton：
    - **表达自由度更高**；
    - irregular kernel 更舒服；
    - 但不一定自动走最新 Blackwell native path。
  - cuTile：
    - **硬件贴合度更高**；
    - 规则 tile reuse 场景更有潜力；
    - 但 irregular access 可能变成很重的 gather/mask/stage 链。

---

**为什么 autotuning 救不了大多数性能问题**

![](95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg)

- 这篇论文一个很重要的判断是：
  - **很多 kernel 慢，不是因为 tile size 没调到最优。**
  - **而是 backend lowering、memory staging、instruction selection 本身有问题。**

- 这点对研究生特别重要。
  - 很多人看到性能差，第一反应是：
    - 调 BLOCK_SIZE；
    - 调 num_warps；
    - 调 num_stages；
    - 调 occupancy。
  - 但 TileBench 的数据说明：
    - 这些参数当然有用；
    - 但它们只能在当前 compiler lowering path 里微调；
    - 如果底层已经走错路，比如：
      - 没用上 **TMA**；
      - 没用上 **tcgen05**；
      - 被迫做大量 shared-memory staging；
      - register spill；
      - bank conflict；
    - 那调参只是“在错误路线里选一条稍微不堵的路”。

- 所以这篇论文的诊断价值在这里：
  - 它提醒我们，tile-based DSL 的性能瓶颈经常不是 DSL 代码表面那几行。
  - 真正关键的是：
    - **DSL 抽象如何被 compiler 降到 PTX/SASS；**
    - **memory movement 是否匹配硬件路径；**
    - **operand tile 是否以硬件喜欢的形态出现。**

---

**LLM 代码生成这部分的直觉意义**

![](336730bb607696cb1e402d8e5305e120d7e5b75eee5b3cbb91f30035d728ffb6.jpg)

- 这部分不是论文的主线性能评测，而是一个很有意思的“可用性实验”。
- 它问的是：
  - 如果未来 kernel 越来越多由 LLM 帮忙写，那么 DSL 的设计会不会影响 LLM 的效率？
- 结果很明确：
  - **Triton 更 token-efficient。**
  - cuTile 需要更多迭代、更多 token，且速度收益更低。

| Model + Backend | TokenCost@10 | TokenEfficiency@10 | 直觉解释 |
|---|---:|---:|---|
| **GPT-5.5 + Triton** | **0.26M** | **20.9** | LLM 熟悉 Triton 语法和常见模式 |
| **GPT-5.5 + cuTile** | **0.32M** | **13.4** | cuTile 更新、更少公开代码，模型先验弱 |
| **Claude Opus 4.7 + Triton** | **0.28M** | **17.1** | Triton 仍更稳定 |
| **Claude Opus 4.7 + cuTile** | **0.42M** | **8.3** | cuTile 生成更不稳定，验证失败更多 |

- 这里的重点不是说 cuTile 设计不好。
- 更准确的理解是：
  - **Triton 已经进入 LLM 的“经验库”。**
  - cuTile 太新，公开代码少，LLM 没有足够模式可模仿。
  - 所以 cuTile 对 LLM 来说像一门刚发布的新 API：
    - 文档里有；
    - 逻辑上能推；
    - 但少了大量“见过的写法”。

---

**这篇论文的整体贡献可以拆成四个层次**

- **Benchmark 贡献**
  - 提供 **45 个算子**，覆盖：
    - point-wise；
    - reduction/normalization；
    - matrix multiplication/attention；
    - stencil/convolution；
    - data layout。
  - 每个算子都有：
    - **PyTorch reference**；
    - **Triton implementation**；
    - **cuTile implementation**；
    - **config.yaml**；
    - dtype 和 size sweep；
    - correctness checker。

| Category | #Ops | 代表算子 |
|---|---:|---|
| **Point-wise** | **12** | vector_add, relu, swiglu, weight_dequant |
| **Reduction/Normalization** | **11** | softmax, layernorm, argmax, moe_topk_gating |
| **Matrix Multiplication/Attention** | **8** | matmul, matmul_int8, flash_attention, linear_self_attention |
| **Stencil/Convolution** | **6** | 1d_conv, 3d_conv, gaussian_blur, jacobi_stencil_2d |
| **Data Layout** | **8** | matrix_copy, matrix_transpose, bitonic_sort, radix_sort |
| **Total** | **45** | — |

- **评价方法贡献**
  - 不只看 latency。
  - 同时看：
    - **speedup over PyTorch**；
    - **TFLOPS**；
    - **effective bandwidth**；
    - **arithmetic intensity**；
    - **roofline utilization**。
  - 这让结果更像硬件-aware analysis，而不是单纯跑分。

- **诊断方法贡献**
  - 对大性能差距的 case 做：
    - **Nsight Compute profiling**；
    - **PTX/SASS inspection**；
    - TMA/tcgen05 usage 分析；
    - bank conflict 分析；
    - register pressure 分析。
  - 它把“谁快”进一步推进到“为什么快”。

- **LLM usability 贡献**
  - 同一个 operator set；
  - 同样 10 次 refinement；
  - 同样 correctness gate；
  - 比较 LLM 生成 Triton 和 cuTile kernel 的：
    - **BestSpeedup@10**；
    - **TokenCost@10**；
    - **TokenEfficiency@10**。
  - 这实际上把 DSL 的“可学习性”和“可生成性”纳入了系统评测。

---

**最值得记住的洞察**

- **没有一个 tile-based DSL 对所有 AI kernel 都通吃。**
  - 规则 tile reuse 强的任务，cuTile 很有优势。
  - irregular、streaming、pointer-heavy 的任务，Triton 更稳。

- **Autotuning 不是性能优化的终点。**
  - 它能修 tile shape 和 occupancy。
  - 但修不了 compiler lowering 走错路。
  - 很多性能头寸卡在：
    - instruction selection；
    - operand staging；
    - shared memory layout；
    - TMA/TMEM pipeline；
    - register pressure。

- **编程模型的抽象会影响 compiler，也会影响 LLM。**
  - 一个 DSL 不只是给人写的。
  - 现在它也越来越像是给 LLM 写的。
  - Triton 的成熟生态让 LLM 更容易生成正确高效代码。
  - cuTile 的硬件贴合度高，但当前对 LLM 和 irregular access 还不够友好。

- **TileBench 的价值不在于宣布冠军，而在于提供显微镜。**
  - 它让你能看清：
    - 什么 workload 适合什么 DSL；
    - 为什么某个 backend 慢；
    - 调参是否还有意义；
    - 性能差距到底是代码问题、抽象问题，还是 compiler lowering 问题。

### 1. Controlled Tile-Based Benchmark Suite

**痛点直击：为什么需要“Controlled Tile-Based Benchmark Suite”**

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

- **TileBench**要解决的不是“有没有 benchmark”这个问题，而是“现有 benchmark 很难公平比较 **Triton** 和 **cuTile**”这个问题。

- 之前比较 **tile-based programming models** 时，最难受的地方在于：

  - **语义不一致**
    - 一个 Triton kernel 可能实现的是某个略微融合过的版本。
    - 一个 cuTile kernel 可能用了不同的数据布局、不同边界处理、不同精度路径。
    - 最后看起来是在比 **Triton vs cuTile**，其实混进了很多“算法差异”和“实现差异”。

  - **输入规模不一致**
    - 有些 benchmark 只测一个大 case。
    - 有些只测几个固定 shape。
    - 但 GPU kernel 的性能非常依赖 shape：
      - 小 shape 可能被 launch overhead 主导。
      - 大 shape 可能进入 bandwidth-bound。
      - GEMM/Attention 类 kernel 还可能在某些尺寸下才真正触发 Tensor Core 优势。

  - **dtype 不一致**
    - FP32、FP16、BF16、FP8、INT8 对应的硬件路径完全不同。
    - 如果一个后端在 FP16 上强，另一个在 FP32 上强，只测单一 dtype 就很容易得出偏结论。

  - **实现结构不一致**
    - 一个后端可能用了成熟优化路径。
    - 另一个后端可能只是“能跑”的 naive 版本。
    - 这种比较只能说明“这两个代码谁写得更好”，不能说明“这两个 programming model 谁更适合这个 operator”。

  - **性能原因说不清**
    - 只给 speedup 数字没用。
    - 研究者真正想知道的是：
      - 是 **TMA** 没用上？
      - 是 **Tensor Core** 路径没走对？
      - 是 shared memory bank conflict？
      - 是 register spilling？
      - 是 irregular memory access 不适合某个 DSL？

- 所以 **Controlled Tile-Based Benchmark Suite** 的核心痛点是：

  - **把所有干扰项尽量按住，只让 Triton 和 cuTile 这两个 programming model 本身暴露差异。**

---

**通俗比方：这不是普通考试，而是“同题、同卷、同评分标准”的实验室测评**

- 可以把普通 GPU benchmark 想成“各学校自己出题、自己判卷”：

  - Triton 交的是一套题。
  - cuTile 交的是另一套题。
  - PyTorch baseline 又是第三套题。
  - 最后拿分数比较，当然会有争议。

- **TileBench**更像一个严格的高考考场：

  - **同一道题**
    - 每个 operator 都有明确的 **PyTorch reference**，作为语义标准答案。

  - **同一套输入**
    - 每个 operator 都跑统一声明的 **dtype/input-size sweeps**。
    - 不是挑一个对自己有利的 shape。

  - **同样的判卷方式**
    - Triton 和 cuTile 的输出都要和 PyTorch reference 做 correctness check。
    - 浮点用统一 tolerance，整数要求 exact match。

  - **同样的计时方式**
    - 使用相同 timing protocol：
      - warmup
      - repeat
      - CUDA graph capture-and-replay
      - L2 cache flushing
      - mean latency

  - **同样的分析口径**
    - 统一计算：
      - **latency**
      - **speedup over PyTorch**
      - **TFLOPS**
      - **effective bandwidth**
      - **arithmetic intensity**
      - **roofline utilization**

- 更贴切一点说：

  - **TileBench 不是在问“谁跑得最快”这么粗糙的问题。**
  - 它是在问：
    - “在同样题目、同样输入、同样评分标准下，哪个 tile-based DSL 在什么类型 workload 上更自然、更稳定、更接近硬件上限？”

---

**关键一招：把 benchmark 从“结果比较”改造成“受控实验”**

- 作者最巧妙的地方，不是发明了一个新 kernel，也不是提出了一个新 compiler pass。

- 关键一招是：

  - **把 Triton 和 cuTile 的比较流程标准化成一个 controlled experiment。**

- 原来的流程通常是：

  - 找一些 Triton kernel。
  - 找一些 cuTile kernel。
  - 跑一跑性能。
  - 看谁快。
  - 再尝试解释原因。

- TileBench 把这件事扭转成：

  - **先固定语义**
    - 每个 operator 都有 **PyTorch reference**。
    - Triton 和 cuTile 都必须实现同一个数学定义。

  - **再固定覆盖范围**
    - 45 个 AI operators。
    - 覆盖：
      - **Point-wise**
      - **Reduction/Normalization**
      - **Matrix Multiplication/Attention**
      - **Stencil/Convolution**
      - **Data Layout**

  - **再固定输入空间**
    - 每个 operator 在 `config.yaml` 里声明：
      - supported dtypes
      - input-size sweep
      - verification tolerance
      - FLOP formula
      - HBM-byte formula

  - **再固定实现比较方式**
    - 每个任务都有：
      - `impl_torch.py`
      - `impl_triton.py`
      - `impl_cutile.py`
      - `config.yaml`
    - Triton 和 cuTile 尽量使用**matched semantics**和**comparable implementation structures**。

  - **最后才比较性能**
    - 比较的不只是 raw latency。
    - 还看：
      - speedup over PyTorch
      - roofline utilization
      - autotune gain
      - profiling-guided diagnosis

| 控制项 | 以前容易混入的干扰 | TileBench 的处理方式 |
|---|---|---|
| **Operator semantics** | 两边实现的数学含义不完全一样 | 用 **PyTorch reference** 作为统一 correctness oracle |
| **dtype** | 只测某个后端擅长的 dtype | 每个 operator 标准化声明 supported dtypes |
| **input size** | shape 选择带偏向 | 每个 dtype 下做 standardized input-size sweeps |
| **implementation structure** | 一个实现深度优化，另一个只是 naive | 手写并验证 Triton/cuTile，保持 comparable structure |
| **timing protocol** | warmup、repeat、cache 状态不同 | 统一使用 Proton、CUDA graph、L2 flush |
| **performance interpretation** | 只有 speedup，无法定位瓶颈 | 加入 roofline、NCU、PTX/SASS diagnosis |

- 这背后的思想很简单：

  - **不要让 benchmark 变成“代码写作比赛”。**
  - **要让 benchmark 变成“programming model 性格测试”。**

- 这样才看得清：

  - **cuTile**在什么情况下强：
    - regular tile movement
    - Tensor Core-friendly
    - TMA-friendly
    - static tile abstraction 适配良好

  - **Triton**在什么情况下强：
    - irregular indexing
    - masked pointer loads
    - streaming bandwidth-bound kernels
    - lightweight operators
    - runtime-computed addresses

- 所以 **Controlled Tile-Based Benchmark Suite** 的本质是：

  - **作者没有直接宣称 Triton 或 cuTile 谁更好，而是先搭了一个足够干净的实验笼子，让两只“鸟”的飞行方式自然暴露出来。**

- 这也是这篇论文的价值所在：

  - 它不是单纯贡献了 45 个 operator。
  - 它贡献的是一套**公平比较 tile-based DSL 的方法论**。
  - 之后别人要评估 Tile-Lang、ThunderKittens、Tilus，甚至新的 GPU DSL，也可以沿着这个受控 benchmark 思路扩展。

### 2. Unified Evaluation Harness

**痛点直击：为什么需要Unified Evaluation Harness**

- **Unified Evaluation Harness**解决的不是“怎么把某个kernel写快”，而是更基础的问题：**怎么判断谁真的快**。
- 在比较**Triton**和**cuTile**这类GPU DSL时，最难受的地方是：
  - **实现细节会污染结论**
    - 一个backend用了更好的计时方式，另一个backend用了普通计时。
    - 一个backend提前热身充分，另一个backend第一次运行还带着编译、cache冷启动、allocator扰动。
    - 结果看似是“语言A比语言B快”，实际可能只是**测试方法不公平**。
  - **PyTorch baseline容易变成不稳定参照物**
    - 如果每个operator自己写一套验证和计时代码，PyTorch参考结果、容差、输入生成、dtype处理都可能不一致。
    - 这样得到的speedup很难横向比较。
  - **GPU测量天然很容易抖**
    - CUDA kernel launch有开销。
    - L2 cache命中状态会影响带宽型kernel。
    - 第一次运行可能包含JIT compilation、runtime初始化、memory pool行为。
    - 小kernel尤其容易被这些“杂音”淹没。
  - **benchmark最怕“看起来系统，实际各测各的”**
    - 45个operator、多个dtype、20组输入规模，如果没有统一流程，最后不是benchmark，而是一堆松散实验。

- 这篇论文的关键判断是：
  - 要比较**programming model**，就必须把其他变量尽量按住。
  - **Unified Evaluation Harness**就是那个“按住变量”的实验夹具。
  - 它让Triton和cuTile在同一个裁判、同一把尺、同一套赛道上跑。

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

---

**通俗比方：它像一个标准化风洞，而不是普通跑道**

- 可以把每个GPU kernel想成一辆赛车：
  - **Triton kernel**是一辆车。
  - **cuTile kernel**也是一辆车。
  - **PyTorch reference**是官方基准车。
- 如果你让它们在不同天气、不同轮胎、不同路面、不同计时器下比赛，结论一定不可靠。
- **Unified Evaluation Harness**像一个**标准化风洞测试间**：
  - 所有车都进同一个房间。
  - 风速固定。
  - 温度固定。
  - 计时器固定。
  - 测试前都先跑几圈热身。
  - 每次测试前把“路面残留”清掉。
  - 最后取多次稳定测量的平均值。

- 这个比方里，各组件对应关系很清楚：

| 现实测试 | TileBench里的对应物 | 作用 |
|---|---|---|
| 官方标准答案 | **PyTorch reference** | 判断结果对不对 |
| 同一裁判 | **shared correctness checker** | 避免各operator自己定规则 |
| 热身圈 | **20 warmup runs** | 去掉冷启动和初始化扰动 |
| 正式计时圈 | **100 timed runs** | 降低随机波动 |
| 固定起跑流程 | **CUDA graph capture-and-replay** | 减少launch开销和Python调度噪声 |
| 清理赛道残留 | **L2-cache flushing** | 避免某个kernel吃到cache红利 |
| 同一计时系统 | **Proton profiler** | 统一记录latency |

- 直觉上，它不是在帮某个backend作弊，而是在防止任何backend“无意中占便宜”。
- 这点很重要：
  - 如果没有这个harness，论文里的性能差异可能来自**测量流程差异**。
  - 有了它，性能差异才更可能来自**Triton/cuTile本身的表达能力、compiler lowering和memory behavior**。

---

**关键一招：把“每个operator自己测”替换成“所有operator进同一条流水线”**

- 作者最巧妙的地方，不是发明了新的计时公式，而是把原来分散的实验流程**收束成一个统一执行协议**。
- 原来的隐含流程大概是：
  - 每个operator自己写输入。
  - 自己跑PyTorch。
  - 自己验证。
  - 自己计时。
  - 自己算speedup。
  - 自己处理warmup和cache。
- 这样的问题是：
  - 每一步都可能引入偏差。
  - operator之间不好比较。
  - backend之间也不好比较。

- TileBench把这套流程改成：
  - 每个operator只声明：
    - **PyTorch reference**
    - **Triton implementation**
    - **cuTile implementation**
    - **config.yaml**
      - dtype列表
      - case grid
      - verify tolerance
      - FLOP公式
      - HBM byte公式
  - 然后统一交给**shared harness**执行：
    - 生成同样输入。
    - 用PyTorch产出标准答案。
    - 分别运行Triton和cuTile。
    - 用同一套容差检查正确性。
    - 通过同样的warmup、cache flushing、CUDA graph replay和timing流程测latency。
    - 用同样公式计算speedup、bandwidth、throughput和roofline utilization。

- 这个“替换”非常关键：
  - 作者并没有只说“我们手动保证公平”。
  - 作者是把公平性写进了**benchmark machinery**里。
  - 换句话说，公平不是靠实验者自觉，而是靠harness强制执行。

---

**这个技术点的核心逻辑**

- **Correctness先于Performance**
  - 每个backend的输出都要对齐**PyTorch reference**。
  - 浮点输出用dtype/operator-specific tolerance。
  - 整数输出要求exact match。
  - 没过正确性，性能数据就没有意义。

- **测量要压低非kernel因素**
  - **CUDA graph capture-and-replay**减少Python launch路径和调度噪声。
  - **20 warmup runs**让JIT、cache、runtime状态进入稳定区间。
  - **100 timed runs**用重复采样压低偶然波动。
  - **L2-cache flushing**防止某些kernel因为刚好命中缓存而显得更快。

- **指标计算要统一**
  - latency只是底层观测。
  - speedup、TFLOPS、effective bandwidth、roofline utilization都来自同一套公式。
  - 这样才能比较不同operator、不同dtype、不同backend。

| Harness环节 | 解决的偏差 | 对结论的价值 |
|---|---|---|
| **PyTorch correctness oracle** | 结果错但跑得快 | 保证性能比较有意义 |
| **CUDA graph replay** | Python和launch overhead | 让小kernel计时更稳定 |
| **L2-cache flushing** | cache状态不一致 | 让带宽型kernel比较更公平 |
| **20 warmup runs** | 冷启动、JIT、runtime初始化 | 避免第一次运行污染数据 |
| **100 timed runs** | 单次计时抖动 | 得到更可靠mean latency |
| **统一config.yaml** | case/dtype/metric不一致 | 支持跨operator聚合分析 |

---

**一句话抓住直觉**

- **Unified Evaluation Harness**就是TileBench里的“实验法官”：
  - 它不负责让Triton或cuTile变快。
  - 它负责确保当论文说“Triton更强”或“cuTile在某类kernel更好”时，这个结论不是由计时脚本、cache状态、warmup次数或验证规则造成的。
- 它的核心价值是把GPU性能评测从“各测各的经验活”变成“同一流水线下的可比实验”。

### 3. Config-Driven Metrics and Roofline Analysis

**痛点直击：为什么需要Config-Driven Metrics and Roofline Analysis**

- 这篇论文要比较 **Triton** 和 **cuTile**，最怕的不是“谁跑得快”，而是“快得有没有意义”。
  - 如果只看 **latency**：
    - 一个 kernel 快，可能只是因为它算得少。
    - 一个 kernel 慢，可能是因为它处理的数据量更大。
    - 不同 dtype、不同 shape、不同 operator 放在一起比，很容易变成“苹果和橘子硬比”。
  - 如果只看相对 **PyTorch speedup**：
    - PyTorch reference 本身可能很弱。
    - 某个 kernel 比 PyTorch 快 100×，不代表它接近硬件极限。
    - 某个 kernel 只快 2×，也可能已经接近 **HBM bandwidth** 或 **Tensor Core peak**。

- 之前做 benchmark 很容易陷入一个尴尬：
  - **性能数字很多，但解释力很弱**。
  - 你知道 Triton 比 cuTile 快，但不知道是因为：
    - 算力用得更好？
    - 内存带宽吃得更满？
    - dtype 影响？
    - shape 影响？
    - PyTorch baseline 太差？
    - 还是 operator 本身就不可能接近 peak？

- **TileBench** 的难受场景尤其明显：
  - 它有 **45个operator**。
  - 每个 operator 有不同的 **dtype**。
  - 每个 dtype 又扫 **20个input configurations**。
  - operator 类型横跨 **Point-wise、Reduction、Attention、Convolution、Data Layout**。
  - 如果没有统一的指标来源，每个实验都可能变成“各说各话”。

- 所以作者要解决的核心问题不是单纯测速，而是：
  - **把每个kernel的性能放回它自己的理论上限里看**。
  - 也就是问一句更有诊断价值的话：
    - “这个 kernel 跑到现在这个速度，是已经接近硬件天花板了，还是还差很远？”

![](95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg)

---

**通俗比方：这不是排行榜，而是体检报告**

- 你可以把普通 benchmark 想成学校运动会排名：
  - 谁100米跑得快，谁第一。
  - 但这个排名很粗糙。
  - 因为有人跑的是100米，有人跑的是400米，有人背着沙袋跑，有人顺风跑。

- **Config-driven metrics** 更像给每个运动员做一张“个性化体检表”：
  - 这个项目理论上最多能跑多快？
  - 这个人现在跑到了自己极限的多少百分比？
  - 他慢是因为心肺不行，还是肌肉不行，还是鞋不合适？

- 放到 GPU kernel 里：
  - **FLOP** 就像“要完成多少计算动作”。
  - **HBM bytes** 就像“要搬多少箱货”。
  - **Arithmetic Intensity** 就像“每搬一箱货，能做多少计算”。
  - **Roofline utilization** 就像“你现在离理论天花板还有多远”。

- 最有顿悟感的点是：
  - 作者不是只问“谁更快”。
  - 作者问的是：
    - **这个operator理论上应该受限于算力，还是受限于内存？**
    - **当前实现到底吃满了哪个资源？**
    - **如果没吃满，问题可能在编译器lowering、memory staging、instruction selection，而不是tile size没调好。**

- 这就像评估餐厅厨房效率：
  - 只看出菜速度不够。
  - 还要知道：
    - 这道菜需要切多少菜？
    - 需要炒多久？
    - 厨师最多每分钟能炒多少？
    - 传菜口最多每分钟能送多少？
  - 如果一道菜慢，不一定是厨师慢，也可能是传菜口堵了。
  - **Roofline Analysis** 做的就是这件事：判断瓶颈到底像“厨师算力不足”，还是“传菜带宽不足”。

---

**关键一招：把指标定义从代码里抽出来，放进config.yaml**

- 作者最巧妙的地方，是没有让每个 benchmark 脚本自己随手算指标。
  - 那样会很危险：
    - A operator 用一种 FLOP 公式。
    - B operator 用另一种 bytes 估计。
    - Triton 和 cuTile 可能各自统计口径不同。
    - 最后 roofline utilization 失去可比性。

- 作者把这些东西统一写进每个 operator 的 **config.yaml**：
  - **dtype lists**
  - **case grids**
  - **verification tolerances**
  - **flops_expr**
  - **bytes_expr**
  - timing options
  - plotting metrics

- 这个动作的本质是：
  - **把benchmark的“语义定义”和“性能解释规则”绑定在一起**。
  - 每个 operator 不只是提供代码，还提供一份“实验说明书”。

| 配置项 | 直觉作用 | 为什么重要 |
|---|---|---|
| **dtype list** | 声明这个operator测哪些数据类型 | FP16、BF16、FP32、INT8 的硬件peak不同，不能混着解释 |
| **case grid** | 声明输入规模怎么扫 | 避免只挑一个对自己有利的shape |
| **verification tolerance** | 声明正确性误差范围 | reduction、mixed precision 会有合理数值误差 |
| **flops_expr** | 声明理论计算量 | 用来算 **TFLOPS** |
| **bytes_expr** | 声明理论HBM访问量 | 用来算 **effective bandwidth** 和 **Arithmetic Intensity** |
| **B200.json** | 声明硬件peak | 用来归一化成 **Roofline utilization** |

- 关键逻辑转换是：
  - 原来流程是：
    - “跑kernel → 得到latency → 比谁快”
  - 作者改成：
    - “跑kernel → 结合config里的FLOP/byte公式 → 算throughput、bandwidth、AI → 再除以硬件roofline → 判断离理论极限多远”

- 这一步非常关键，因为它把性能分析从“感性比较”变成了“可诊断比较”。
  - **latency** 告诉你“快不快”。
  - **speedup over PyTorch** 告诉你“比baseline强多少”。
  - **TFLOPS / bandwidth** 告诉你“资源用了多少”。
  - **Roofline utilization** 告诉你“离硬件上限还有多少空间”。

- 论文里的一个核心结论就是靠这个机制支撑的：
  - autotuning 后，Triton 和 cuTile 都有提升。
  - 但大多数 kernel 仍然离 roofline 很远。
  - 这说明问题不只是 **BLOCK_SIZE、num_warps、occupancy** 没调好。
  - 更深的问题在：
    - compiler lowering
    - memory staging
    - Tensor Core 使用路径
    - TMA 使用方式
    - shared memory layout
    - bank conflict
    - instruction selection

---

**这项技术点真正的价值**

- **Config-Driven Metrics** 让每个 operator 的评估口径固定下来。
  - 不靠论文作者临时解释。
  - 不靠每个脚本随意统计。
  - 每个 case、每个 dtype、每个 backend 都走同一套公式。

- **Roofline Analysis** 让性能数字有了“物理意义”。
  - 不是只说 Triton 2.7×、cuTile 2.2×。
  - 而是进一步问：
    - 这个 2.7× 是否接近 B200 的能力？
    - 这个 kernel 是 compute-bound 还是 bandwidth-bound？
    - autotuning 之后还有没有硬件headroom？
    - backend gap 是调参问题，还是编译器生成代码的问题？

- 这也是为什么 Figure 3 很重要：
  - 它不是简单展示 autotuning 让 latency 降了多少。
  - 它展示的是：
    - autotuning 把 kernel 往 roofline 推近了一点。
    - 但多数箭头仍然停在远低于 **80% roofline utilization** 的区域。
  - 这直接支持论文判断：
    - **参数搜索只能解决表层问题，真正瓶颈经常在lowering和memory behavior。**

---

**一句话抓住本质**

- **Config-Driven Metrics and Roofline Analysis** 的核心不是“多算几个指标”，而是给每个 kernel 配一把统一的尺子：
  - 先用 **config.yaml** 固定“这个任务到底算了多少、搬了多少、允许多少误差”；
  - 再用 **Roofline** 判断“它离B200理论极限还有多远”；
  - 最后把性能比较从“谁快”升级成“为什么快、哪里没吃满、下一步该查什么”。

### 4. Comparable Default and Autotuned Execution Paths

**痛点直击：为什么要有“Comparable Default and Autotuned Execution Paths”**

- 这件事要解决的不是单纯“跑得更快”，而是一个更麻烦的问题：**怎么公平比较 Triton 和 cuTile**。
- 如果直接拿两个 backend 的最快结果比，很容易变成“调参能力比赛”，而不是“编程模型能力比较”。
  - Triton 有自己的调参旋钮：**BLOCK_SIZE、num_warps、num_stages**。
  - cuTile 有自己的调参旋钮：**TILE_SIZE、occupancy**。
  - 这些旋钮不是一一对应的，不能说 Triton 的 **num_warps=4** 就等价于 cuTile 的 **occupancy=4**。
- 之前最“难受”的地方在于：
  - 如果只比默认配置，可能会冤枉某个 backend。
    - 一个 backend 的默认参数选得保守，就显得慢。
    - 另一个 backend 的默认参数刚好撞上甜点区，就显得强。
  - 如果只比 autotuned 最优配置，又可能不公平。
    - 某个 backend 的搜索空间更大、调参接口更成熟，就占便宜。
    - 某个 backend 暴露的 knob 少，不代表模型差，可能只是调参自由度不同。
  - 如果每个 backend 都随便调到满意为止，那就不可复现，也不像 benchmark，更像手工优化竞赛。
- 所以作者真正想回答的是：
  - **在合理默认配置下，两个模型开箱即用表现如何？**
  - **在相近调参预算下，两个模型各自能挤出多少性能？**
  - **调参之后还慢，是参数没选好，还是 backend lowering / memory staging / instruction selection 本身有问题？**

![](95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg)

---

**通俗比方：这不是让两个人穿同一双鞋，而是给他们同等训练时间**

- 可以把 Triton 和 cuTile 想成两个短跑运动员。
  - Triton 擅长穿“钉鞋”：**BLOCK_SIZE、num_warps、num_stages** 是它调节步幅、频率、起跑姿势的方式。
  - cuTile 擅长穿“专业跑鞋”：**TILE_SIZE、occupancy** 是它调节步幅和体能分配的方式。
- 公平比赛不是要求两个人穿同一双鞋。
  - 因为鞋型不同，硬塞同一套参数反而不公平。
  - Triton 的 **num_stages** 和 cuTile 的 **occupancy** 本来就不是同一种东西。
- 真正公平的做法是：
  - 给两个人一个**合理默认姿势**，先测“开箱表现”。
  - 再给两个人**差不多的训练时间和搜索预算**，让他们用各自熟悉的方法优化。
  - 最后比较：
    - 谁默认就跑得稳。
    - 谁经过训练提升大。
    - 谁训练完还离硬件极限很远。
- 这个思维模型很重要：
  - **Comparable** 不等于 **identical**。
  - 不是“参数完全一样”，而是“调参机会大致公平”。
  - 这就是论文里 **backend-native autotuning under comparable search budgets** 的核心含义。

| 比较方式 | 看起来公平吗 | 实际问题 |
|---|---:|---|
| 只比默认配置 | 部分公平 | 容易受人工默认参数影响 |
| 只比各自最优配置 | 不一定公平 | 搜索空间和调参成本可能不同 |
| 强行使用同一套参数 | 不公平 | Triton 和 cuTile knob 语义不同 |
| **默认路径 + 可比预算 autotune** | **更公平** | 既看开箱表现，也看合理调参后的上限 |

---

**关键一招：作者没有强行统一参数，而是统一“比较协议”**

- 作者最聪明的地方在于：没有试图把 Triton 和 cuTile 的参数机械对齐。
- 他们做的是把执行路径分成两条：
  - **Default execution path**
    - 每个 operator 给一个人工选择的固定配置。
    - 用来衡量“一个有经验的人写了一个合理版本后，backend 默认能跑到什么水平”。
  - **Autotuned execution path**
    - 每个 backend 在自己的 native knob 上搜索。
    - Triton 搜 **BLOCK_SIZE、num_warps、num_stages**。
    - cuTile 搜 **TILE_SIZE、occupancy**。
    - 搜索预算和粒度保持可比，而不是参数逐项相等。
- 这一步的逻辑转换是：
  - 以前的问题是：**我们到底是在比较 backend，还是在比较调参运气？**
  - 作者把它改成：**先固定一个合理起点，再给双方相近的调参机会，看调参能不能弥补差距。**
- 这就把性能差距拆成了三层：
  - **默认实现差距**
    - 谁在常规写法下更稳。
  - **调参可恢复差距**
    - 谁通过 tile shape、warps、stages、occupancy 能追回性能。
  - **调参无法解决的差距**
    - 谁的问题来自 compiler lowering、Tensor Core 路径、TMA 使用、shared memory staging、bank conflict 等更底层因素。
- 论文中的结果正好说明这一点：
  - Autotuning 确实有效。
    - Triton 的几何平均 roofline utilization gain 是 **1.20×**。
    - cuTile 是 **1.15×**。
  - 但调参不是万能的。
    - Autotune 后，只有 **8/45** 个 Triton 实现达到至少 **80% roofline utilization**。
    - cuTile 只有 **4/45** 个达到这个水平。
  - 这说明很多瓶颈不是“BLOCK_SIZE 没选好”，而是 backend 生成代码的路径本身不够理想。

| Backend | Default path | Autotuned path | 主要调参 knob | Autotune gain |
|---|---|---|---|---:|
| **Triton** | 手工固定配置 | backend-native 搜索 | **BLOCK_SIZE、num_warps、num_stages** | **1.20×** |
| **cuTile** | 手工固定配置 | backend-native 搜索 | **TILE_SIZE、occupancy** | **1.15×** |

- 一句话抓住这个技术点：
  - **作者不是让 Triton 和 cuTile 使用同一套参数，而是让它们在各自语言最自然的调参空间里，用相近预算比赛。**
- 这让 TileBench 的比较更可信：
  - 如果 default 下 Triton 强，说明 Triton 开箱更稳。
  - 如果 autotune 后 cuTile 追上，说明 cuTile 只是需要更合适的 tile/occupancy。
  - 如果 autotune 后仍然差很多，说明问题大概率不在参数，而在 **compiler lowering** 或 **memory behavior**。

### 5. Profiling-Guided Bottleneck Diagnosis

**痛点直击：为什么需要Profiling-Guided Bottleneck Diagnosis**

- **TileBench要解决的不是“谁快谁慢”这么简单的问题，而是“为什么快、为什么慢”。**
  - 如果只看 latency，结论很容易停在表面：
    - **Triton快**：可能是因为它的 pointer-based load 更自然。
    - **cuTile快**：可能是因为它走到了 Blackwell-native 的 **TMA/tcgen05/Tensor Core** 路径。
    - 但这些都只是猜测。
  - 真正难受的地方在于：
    - 同一个 operator，Triton 和 cuTile 的源码结构可能很像，但编译器最后生成的 **PTX/SASS** 完全不同。
    - 同样写了 tile-level computation，后端可能一个走 **TMA + tcgen05**，另一个退化成 **cp.async + ldmatrix + register-fragment**。
    - 同样是 `matmul` 或 `conv`，瓶颈可能不是 arithmetic，而是 **operand materialization**、**shared-memory staging**、**bank conflicts**、**register spilling** 或 **async pipeline wait**。

- **传统benchmark的问题是：只能告诉你“红绿灯”，不能告诉你“发动机哪里坏了”。**
  - 只报告 speedup：
    - 像是在说“这辆车比那辆车慢30%”。
    - 但你不知道是轮胎打滑、油路堵了、发动机没吃满，还是司机一直踩刹车。
  - TileBench想进一步回答：
    - **慢在memory movement？**
    - **慢在Tensor Core没有用上？**
    - **慢在shared memory bank conflict？**
    - **慢在compiler lowering选错路径？**
    - **慢在TMA latency没有被compute覆盖？**

- **这个痛点在tile-based DSL里尤其尖锐。**
  - Triton 和 cuTile 都把程序员从 CUDA 细节里解放出来。
  - 但代价是：
    - 你写的是高层 tile 操作。
    - 真正执行的是编译器生成的底层指令。
    - 性能问题往往藏在“你看不见的lowering结果”里。
  - 所以作者必须用 **Nsight Compute + PTX/SASS inspection** 把这层黑盒拆开。

![](f9e643fecb6869d7899b6b7a50bbcaf63d4acc756e36ecba362971843a1ab893.jpg)

---

**通俗比方：这不是跑分，而是GPU内科体检**

- 可以把 **Profiling-Guided Bottleneck Diagnosis** 理解成：
  - 普通benchmark像是给病人量体温。
  - TileBench的diagnosis像是做一套完整检查：
    - **验血**：看 instruction mix。
    - **CT**：看 PTX/SASS 里实际用了哪些指令。
    - **心电图**：看 warp stall、scoreboard、async wait。
    - **血管造影**：看 memory path、TMA、shared memory bank conflict。
    - **病理切片**：看某一行源码最终被lower成了什么硬件行为。

- 这个比方的关键点是：
  - latency只是“发烧”。
  - 真正的病因可能完全不同。
  - 两个kernel都慢，但一个是：
    - **Tensor Core没吃满**。
  - 另一个是：
    - **shared memory bank conflict严重**。
  - 再另一个是：
    - **compiler没有把tile操作lower到Blackwell-native路径**。

- 更贴近系统研究的类比：
  - 这有点像做 **compiler optimization debugging**。
  - 源码层面你以为自己写的是：
    - “加载tile → 做mma → 存回去”
  - 但硬件层面可能变成：
    - “一堆index计算 → gather → mask → shared/local staging → register spilling → 再喂给mma”
  - TileBench的diagnosis就是把“你以为的执行路径”和“GPU实际走的执行路径”对齐。

---

**关键一招：作者把性能比较从“结果对比”替换成“执行路径对比”**

- 作者没有只做一个大表格说：
  - Triton在多少个operator上赢。
  - cuTile在多少个operator上赢。
  - autotune提升了多少。
- 更巧妙的地方是：
  - 作者把性能差异拆成了几个可观察的硬件层症状。
  - 然后用 **Nsight Compute** 和 **PTX/SASS inspection** 去定位症状背后的编译器行为。

| 诊断对象 | 看什么 | 直觉含义 |
|---|---|---|
| **Instruction mix** | SASS里有哪些指令占大头 | 算力花在真正计算上，还是花在搬运、重排、解包上 |
| **Memory behavior** | global/shared/local memory traffic | 是HBM瓶颈，还是shared/local staging太重 |
| **TMA/tcgen05 usage** | 是否出现Blackwell-native指令 | tile abstraction有没有真正映射到新硬件路径 |
| **Bank conflicts** | shared-memory conflict counters | shared memory是不是看似快、实际在排队 |
| **Warp stalls** | stall_long_scoreboard等 | warp是在等memory、等async phase，还是等依赖 |
| **PTX/SASS lowering** | 高层DSL操作最终变成什么 | 编译器有没有选对执行策略 |

- 这个逻辑转换很重要：
  - 原来的问题是：
    - “Triton和cuTile谁更快？”
  - 作者把它改成：
    - “这个operator的瓶颈是哪一种硬件症状？”
    - “这种症状是由哪一种DSL表达方式或compiler lowering造成的？”
  - 这就让benchmark从排行榜变成了诊断工具。

---

**拿论文里的case看，这一招为什么有用**

- **2d_conv/fp32** 这个例子很典型。
  - 表面上看：
    - cuTile比Triton慢。
  - 如果只看operator类别，你可能会误以为：
    - convolution天然适合tile。
    - cuTile应该更强。
  - 但profiling发现：
    - 瓶颈不在FP32 arithmetic本身。
    - 问题出在 **virtual-im2col operands** 的materialization。
  - cuTile这里需要通过：
    - **linearized per-element indices**
    - **mask**
    - **ct.gather**
    - **ct.where**
    - 以及可能的 shared/local staging
  - Triton则更直接：
    - 构造 pointer tile。
    - 把 boundary predicate 直接挂到 `tl.load` 上。
  - 直觉上说：
    - cuTile像是先把一堆零散货物重新打包成标准集装箱，再运输。
    - Triton像是直接拿着地址清单去取货，边界条件顺手处理。
  - 所以这里Triton更自然。

- **streamk_matmul/bf16** 的case也很有意思。
  - NCU显示主要stall落在一个 predicated `BRA` 指令上。
  - 如果只看源码，你可能会以为：
    - “分支跳转慢了？”
  - 但作者继续追SASS里的predicate来源，发现：
    - predicate来自 **SYNCS.PHASECHK.TRANS64.TRYWAIT**。
  - 这说明真正瓶颈不是branch，而是：
    - **async MMA/TMEM phase没有完成**
    - warp在等 Tensor Core/TMEM pipeline
  - 关键诊断方法是：
    - 不要只看stall落在哪条指令。
    - 要看这条指令依赖的producer是谁。
  - 这就是profiling-guided diagnosis比普通profiling更强的地方：
    - 它不止看“哪里堵”。
    - 它追问“为什么堵”。

---

**更深一层：它诊断的是“DSL表达能力”和“compiler lowering”的错配**

- TileBench的核心洞察是：
  - Triton和cuTile不是单纯的两种写法。
  - 它们背后代表两种不同的表达风格。
- **cuTile** 的强项：
  - 规则tile。
  - 静态tile shape。
  - 可复用tile movement。
  - 容易映射到 **TMA**、**Tensor Core**、**tcgen05**。
- **Triton** 的强项：
  - pointer arithmetic灵活。
  - masked load自然。
  - irregular address表达方便。
  - 对 sparse、decode、dequant、layout kernel 更稳。

| 场景 | 更容易占优的backend | 原因 |
|---|---|---|
| 规则GEMM / attention | **cuTile** | tile movement和Tensor Core路径更贴近Blackwell硬件 |
| irregular indexing | **Triton** | pointer-based model表达任意地址更自然 |
| streaming pointwise | **Triton** | plain `tl.load` 避免不必要TMA/shared staging |
| low-bit packed GEMM | **cuTile** | 更可能走原生 Blackwell `UTMALDG.2D` / `UTCIMMA` |
| gather-mask-heavy conv | **Triton** | predicate直接附着在load上，producer chain更短 |

- 所以profiling-guided diagnosis的价值不是“给每个kernel找一个锅”。
- 它真正揭示的是：
  - 某个DSL抽象在某类workload上是否“贴硬件”。
  - 编译器有没有把高层tile语义翻译成正确的低层执行路径。
  - autotune为什么救不了某些kernel。

---

**为什么autotune不够，必须做diagnosis**

- 论文里一个很关键的结果是：
  - autotune确实有提升。
  - 但提升有限。
  - 很多kernel离roofline还很远。

| 指标 | Triton | cuTile |
|---|---:|---:|
| default到autotuned的几何平均roofline提升 | **1.20×** | **1.15×** |
| autotuned后达到80% roofline的operator数 | **8/45** | **4/45** |

- 这说明：
  - tile size、num_warps、num_stages、occupancy这些参数当然重要。
  - 但它们只能调“方向盘和油门”。
  - 如果底层路线选错了，比如：
    - 没走TMA。
    - 没走tcgen05。
    - shared memory layout冲突严重。
    - register spilling严重。
    - async pipeline没有overlap起来。
  - 那autotune再怎么调，也只是把一辆路线错误的车开得稍微快一点。

![](95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg)

---

**一句话抓住这个技术点**

- **Profiling-Guided Bottleneck Diagnosis** 的本质是：
  - 不满足于说“这个DSL快”或“那个DSL慢”。
  - 而是把性能差异一路追到 **SASS指令、memory path、Tensor Core路径、shared-memory conflict、compiler lowering**。
- 它最巧妙的地方在于：
  - 作者把benchmark从“打分系统”改造成了“验尸报告”。
  - 每个性能gap都尽量对应到一个具体机制：
    - 是 **TMA没用上**。
    - 是 **tcgen05路径没走通**。
    - 是 **ct.gather/ct.where造成operand materialization过重**。
    - 是 **shared memory bank conflict**。
    - 是 **async phase wait**。
    - 是 **register pressure/spilling**。
- 对研究生来说，最该带走的直觉是：
  - **tile-based DSL的性能不是源码决定的，而是源码经compiler lowering后在硬件上实际走的路径决定的。**
  - TileBench的diagnosis就是把这条隐藏路径照出来。

### 6. Iterative LLM Kernel Generation Track

**痛点直击：为什么要单独设计Iterative LLM Kernel Generation Track**

- 这里真正要解决的痛点，不是简单问一句：**LLM能不能写GPU kernel**。
- 更难受的问题是：
  - **写得对**和**写得快**不是一回事。
    - LLM很可能生成一个能编译、能跑、结果也对的 Triton/cuTile kernel。
    - 但这个 kernel 可能比 PyTorch 还慢，甚至只是“形式上用了DSL”，没有真正利用GPU。
  - **一次生成结果不公平**。
    - LLM写代码通常不是一锤子买卖。
    - 真实使用时，用户会看报错、看性能、改tile size、改memory access，再让模型迭代。
    - 如果 benchmark 只测 iteration 0，就像只看学生第一次草稿，不能反映模型“调试和优化”的能力。
  - **只看最终速度也不公平**。
    - 一个模型可能最后写出了很快的 kernel，但花了巨量 prompt、反馈、代码重写。
    - 另一个模型可能速度略低，但很快收敛、token 成本低。
    - 对实际工程来说，**TokenCost**就是钱、时间和API预算。
  - **Triton和cuTile的可生成性不同**。
    - Triton有更多公开样例，LLM训练语料里见得多。
    - cuTile更新、更少见，API约束也更硬，比如 tile shape、ct.load/ct.store、occupancy、padding。
    - 所以论文不只是在比较两个backend跑得快不快，也在比较它们是否“适合被LLM写出来”。

- 直白讲，作者想评估的是：
  - **LLM作为kernel工程师时，在10轮调试预算内，能不能写出正确且高性能的Triton/cuTile kernel？**
  - **为了达到这个性能，它烧掉了多少token？**
  - **哪个DSL更适合LLM自动生成？**

---

**通俗比方：这不是考试，是十轮代码面试加性能调参**

- 可以把这个 track 想成一个**GPU kernel编程面试**：
  - 面试官给你：
    - operator 描述
    - PyTorch reference
    - config.yaml
    - Triton/cuTile API说明
    - 硬件约束
  - 你要交：
    - `impl_triton.py`
    - `impl_cutile.py`
  - 面试官每轮告诉你：
    - 编译过没过
    - correctness 过没过
    - 比 PyTorch 快多少
    - roofline utilization 多高
    - 哪些 case 最差
  - 你最多有**10轮修改机会**。

- 关键是，这个面试不是只看“最后答案漂不漂亮”。
  - 如果你前9轮都编译失败，第10轮终于写对，也会被记录：
    - 因为前9轮消耗的**token**是真实成本。
  - 如果你写了一个很快但结果错的 kernel：
    - 分数直接归零。
    - 因为错的高性能 kernel 在系统里没有价值。
  - 如果你偷偷调用 PyTorch、cuBLAS、cuDNN：
    - 会被判作弊。
    - 因为 benchmark 要测的是LLM写DSL kernel的能力，不是套库能力。

- 更形象一点：
  - **BestSpeedup@10**像是“10轮内你最好的一次成绩”。
  - **TokenCost@10**像是“你为了考出这个成绩花了多少草稿纸和辅导费”。
  - **TokenEfficiency@10**像是“每花100万token，换来了多少有效加速”。

| 指标 | 直觉含义 | 为什么重要 |
|---|---|---|
| **Correctness-gated speedup** | 只有结果正确，速度才算数 | 防止“错得很快”的kernel刷分 |
| **BestSpeedup@10** | 10轮里最好的正确实现速度 | 衡量LLM最终能达到的性能上限 |
| **TokenCost@10** | 10轮总token消耗 | 衡量生成和调参成本 |
| **TokenEfficiency@10** | 单位token换来的加速 | 衡量哪个backend更“好写”、更省钱 |

![](336730bb607696cb1e402d8e5305e120d7e5b75eee5b3cbb91f30035d728ffb6.jpg)

---

**关键一招：作者把“静态代码生成评测”扭转成“闭环优化过程评测”**

- 最巧妙的地方在于：
  - 作者没有只让LLM生成一次 kernel。
  - 也没有只拿最终运行时间做排名。
  - 而是把LLM生成kernel这件事，建模成一个**带反馈的迭代优化过程**。

- 原来的常见流程大概是：
  - 给LLM一个operator描述。
  - 让它输出代码。
  - 编译、验证、测速。
  - 得到一个结果。
  - 结束。

- TileBench把这一步替换成：
  - **生成 → 验证 → 测速 → 反馈 → 再生成**，连续做10轮。
  - 每一轮都重新构造 prompt，而不是简单接着聊天。
  - prompt 里放入：
    - 历史 iteration 结果
    - 上一轮源码
    - correctness 状态
    - speedup over PyTorch
    - roofline utilization
    - 最差 case 信息
    - 如果上一轮退化，还给出历史最佳 verify-clean 代码作为回退参考

- 这个设计的直觉很重要：
  - LLM不是被当作“代码补全器”。
  - LLM被当作一个**会读profiling反馈的初级kernel工程师**。
  - benchmark测的不是单次灵感，而是：
    - 能不能根据失败修 bug
    - 能不能根据性能反馈调参数
    - 能不能避免退化
    - 能不能在有限 token 预算内收敛

- **Correctness-gated speedup**是这里的“闸门”：
  - 如果某一轮代码不编译：
    - speedup = 0
  - 如果编译但结果错：
    - speedup = 0
  - 如果结果正确：
    - 才计算相对 PyTorch 的加速
  - 这就把“正确性”放在性能前面，避免LLM靠非法捷径或错误实现拿高分。

- **TokenEfficiency@10**是另一个关键扭转：
  - 传统benchmark通常问：
    - 谁最快？
  - 这里还问：
    - 谁是花最少token达到较好速度？
  - 这对LLM kernel generation很现实：
    - API调用有成本。
    - 迭代调试有时间。
    - 长prompt和复杂API会拖慢搜索。
    - DSL越难写，模型越容易反复犯错，token效率越低。

- 论文最后得到的结论很清楚：
  - **Triton比cuTile更适合当前LLM生成**。
  - 不是因为Triton在所有kernel上都绝对更快。
  - 而是因为：
    - LLM更熟悉Triton语法和惯用模式。
    - Triton的pointer-based表达更接近常见GPU编程思维。
    - cuTile较新，公开代码少，模型先验弱。
    - cuTile的tile abstraction和API约束让模型更容易在正确性和性能之间摇摆。

| 组合 | TokenCost@10 | TokenEfficiency@10 | 直觉解读 |
|---|---:|---:|---|
| **GPT-5.5 + Triton** | **0.26M** | **20.9** | 最省token且效率最高 |
| **GPT-5.5 + cuTile** | **0.32M** | **13.4** | 能写，但更费token |
| **Claude Opus 4.7 + Triton** | **0.28M** | **17.1** | Triton仍然稳定 |
| **Claude Opus 4.7 + cuTile** | **0.42M** | **8.3** | cuTile生成难度明显更高 |

- 一句话抓住这个技术点：
  - **Iterative LLM Kernel Generation Track不是在测“LLM会不会写kernel”，而是在测“LLM作为一个有10轮反馈机会的kernel工程师，能用多少token写出多快且正确的Triton/cuTile kernel”。**
  - 这让benchmark从单纯性能比较，升级成了对**DSL可生成性**、**调试友好性**和**token经济性**的评估。
