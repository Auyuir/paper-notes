# TileBench: A Controlled Benchmark for Performance Evaluation and Bottleneck Diagnosis of Tile-Based Programming Models 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Anonymous

**发表期刊/会议 (Journal/Conference)**: ACL

**发表年份 (Publication Year)**: 2026

**研究机构 (Affiliations)**: unknown

---

## 1. 摘要

**目的**

- **TileBench**旨在构建一个受控、可复现的 benchmark，用于系统评估两类 tile-based GPU programming models：
  - **Triton**
  - **cuTile**
- 论文关注的核心问题包括：
  - 在相同 operator 语义和相近实现结构下，**Triton 与 cuTile 的性能差异**。
  - **autotuning** 能带来多少收益，以及距离 B200 GPU 的 **roofline limit** 还有多少差距。
  - 性能差异背后的 **compiler lowering、memory access、Tensor Core/TMA 使用、bank conflict** 等底层原因。
  - 在 **LLM-generated kernels** 场景下，哪种 programming model 更容易生成正确且高性能的代码，并具有更高 **Token Efficiency**。

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

---

**方法**

- **Benchmark 构建**
  - TileBench 包含 **45 个 AI operator**。
  - 来源：
    - **26 个**来自 **TritonBench**
    - **19 个**来自 **LeetGPU**
  - 覆盖典型 AI kernel 模式：
    - **Point-wise**
    - **Reduction/Normalization**
    - **Matrix Multiplication/Attention**
    - **Stencil/Convolution**
    - **Data Layout**

| 类别 | Operator 数量 | 代表 operator |
|---|---:|---|
| **Point-wise** | 12 | vector_add, relu, swiglu, weight_dequant |
| **Reduction/Normalization** | 11 | softmax, layernorm, argmax, moe_topk_gating |
| **Matrix Multiplication/Attention** | 8 | matmul, matmul_int8, flash_attention, linear_self_attention |
| **Stencil/Convolution** | 6 | 1d_conv, 3d_conv, gaussian_blur, jacobi_stencil_2d |
| **Data Layout** | 8 | matrix_copy, matrix_transpose, bitonic_sort, radix_sort |
| **总计** | **45** | — |

- **标准化实现**
  - 每个 operator 提供：
    - **PyTorch reference**：语义基准与 correctness oracle。
    - **Triton implementation**：手写并验证。
    - **cuTile implementation**：手写并验证。
    - **config.yaml**：定义 dtype、输入尺寸、tolerance、FLOP/byte 公式、timing 配置。
  - 每个 operator 针对支持的 dtype 运行 **20 个 input configurations**。
  - Triton 与 cuTile 在：
    - operator 语义
    - dtype 设置
    - 输入规模 sweep
    - correctness check
    - timing protocol
    - autotuning budget  
    上保持一致。

- **实验平台**
  - GPU：**NVIDIA B200**
  - HBM：**180 GB HBM3e**
  - 实测峰值带宽：**6539.4 GB/s**
  - 软件环境：
    - **PyTorch 2.10**
    - **CUDA 13.0**
    - **Triton 3.6.0**
    - **cuda-tile 1.3.0**

| 指标 | 数值 |
|---|---:|
| GPU | **NVIDIA B200** |
| HBM bandwidth | **6539.4 GB/s** |
| FP32/TF32 peak | **1100 TFLOPS** |
| FP16/BF16 peak | **2250 TFLOPS** |
| FP8 peak | **4500 TFLOPS** |
| INT8 peak | **4500 TFLOPS** |

- **性能测量**
  - 使用 **Proton profiler**。
  - 启用：
    - **CUDA graph capture-and-replay**
    - **L2 cache flushing**
    - **20 warmup runs**
    - **100 timed runs**
  - 关键指标：
    - **Latency**
    - **Speedup over PyTorch**
    - **TFLOPS**
    - **Effective bandwidth**
    - **Arithmetic intensity**
    - **Roofline utilization**

- **Autotuning 设计**
  - 每个 backend 均包含：
    - **default path**：人工选择的固定配置。
    - **autotuned path**：在预设候选空间中搜索最优配置。
  - Triton 调节：
    - **BLOCK_SIZE**
    - **num_warps**
    - **num_stages**
  - cuTile 调节：
    - **TILE_SIZE**
    - **occupancy**
  - 该设计比较的是各 backend 在自身 native tuning interface 下的最佳表现，而不是一一对应参数映射。

- **诊断分析**
  - 对性能差距大的 case 使用：
    - **Nsight Compute**
    - **PTX/SASS inspection**
  - 关注：
    - **instruction mix**
    - **TMA/tcgen05 usage**
    - **Tensor Core path**
    - **shared-memory bank conflict**
    - **register pressure**
    - **stall_long_scoreboard**
    - **MMA/TMEM pipeline wait**

- **LLM code-generation track**
  - 评估两个 reasoning models：
    - **OpenAI GPT-5.5**
    - **Anthropic Claude Opus 4.7**
  - 每个 operator、每个 backend 运行 **10 次迭代 refinement**。
  - LLM 输入包括：
    - framework guide
    - API reference
    - operator description
    - config.yaml
    - PyTorch reference
    - 前一轮 verification/performance feedback
  - 不允许：
    - 调用 PyTorch/cuBLAS/cuDNN 完成计算
    - 输出缓存作弊
    - 使用 backend autotuner
  - 评估指标：
    - **BestSpeedup@10**
    - **TokenCost@10**
    - **TokenEfficiency@10**

---

**结果**

- **RQ1：Triton 与 cuTile 的总体性能**
  - 两者都明显快于 PyTorch。
  - **Triton 整体更强**：
    - Triton 几何平均 speedup：**2.7×**
    - cuTile 几何平均 speedup：**2.2×**
    - Triton 超过 PyTorch：**36/45 operators**
    - cuTile 超过 PyTorch：**34/45 operators**
    - Triton median speedup：**3.1×**
    - cuTile median speedup：**2.7×**

![](c46cba47f4d618a2e72eb29c059a5c2144d83ca4d63ed90b4f4c0c5a37a4b113.jpg)

| 指标 | Triton | cuTile |
|---|---:|---:|
| Geomean speedup over PyTorch | **2.7×** | **2.2×** |
| Median speedup | **3.1×** | **2.7×** |
| 超过 PyTorch 的 operator 数 | **36/45** | **34/45** |
| 快于对方的 operator 数 | **34/45** | **11/45** |

- **cuTile 的优势集中在规则 tile-reuse workload**
  - cuTile 在 **11/45 operators** 上快于 Triton。
  - 优势场景包括：
    - **dense matrix multiplication**
    - **attention**
    - **stencil/convolution**
    - **Tensor Core/TMA-friendly kernels**
  - 代表 operator：
    - **matmul_fp32_fp16_fp8**
    - **1d_conv**
    - **2d_conv**
    - **flash_attention**
    - **matmul_int8**

- **Triton 对不规则和轻量 workload 更稳健**
  - Triton 更适合：
    - runtime-computed indices
    - sparse layout
    - low-bit packing
    - masked load
    - boundary-heavy control
    - bandwidth-bound streaming kernels
  - 代表 operator：
    - **block_sparse_attention**
    - **flash_decode**
    - **linear_self_attention**
    - **weight_dequant**

- **类别级 roofline utilization**
  - **Point-wise** 类别利用率最高：
    - Triton：**0.73**
    - cuTile：**0.64**
  - **Reduction/Normalization** 与 **Data Layout** 居中。
  - **Stencil/Convolution** 与 **Matrix Multiplication/Attention** 平均利用率较低，主要受 flash_decode、linear_self_attention、小 filter convolution 等 case 拖累。
  - operator 内部结构对性能上限的影响通常大于 backend 平均差异。

![](56a81d119d5f0e2475b0ac4c118839dcfc5eb04af7067a25c7639e684180dcf7.jpg)

- **RQ2：Autotuning 与 roofline headroom**
  - Autotuning 带来稳定但有限的收益。
  - 几何平均 roofline utilization gain：
    - Triton：**1.20×**
    - cuTile：**1.15×**
  - Median gain：
    - Triton：**1.07×**
    - cuTile：**1.04×**
  - Triton 在 **argmax** 上达到最高 autotune gain：**2.60×**
  - cuTile 在 **2d_conv** 上达到最高 autotune gain：**2.90×**

![](95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg)

![](6e1f4d31e8492995f3941738cd3ec827bbdd3969b7866160a9a882e8d2f8f2f7.jpg)

| 指标 | Triton | cuTile |
|---|---:|---:|
| Geomean autotune gain | **1.20×** | **1.15×** |
| Median autotune gain | **1.07×** | **1.04×** |
| ≥80% roofline utilization 的 operator 数 | **8/45** | **4/45** |
| 最高 autotune gain | **2.60× argmax** | **2.90× 2d_conv** |

- **多数 kernel 仍远低于 hardware limit**
  - autotuning 后仍只有少数实现达到 **≥80% roofline utilization**。
  - 说明瓶颈不只是 tile size、num_warps、occupancy 等参数。
  - 更关键的因素是：
    - **compiler lowering path**
    - **instruction selection**
    - **Tensor Core usage**
    - **TMA usage**
    - **shared-memory layout**
    - **register spilling**
    - **operand staging**

- **RQ3：性能差距的底层原因**
  - Triton 与 cuTile 的主要差异来自 **memory access expression** 与 **compiler lowering behavior**。
  - cuTile 在规则 tile 上更贴近 Blackwell native path：
    - **ct.load**
    - **ct.mma**
    - **ct.store**
    - **TMA**
    - **tcgen05**
    - **TMEM accumulator**
  - Triton 的 pointer-based model 更适合：
    - 任意地址表达
    - masked load
    - irregular gather
    - runtime indexing

![](f9e643fecb6869d7899b6b7a50bbcaf63d4acc756e36ecba362971843a1ab893.jpg)

- **2d_conv/fp32 case**
  - 两个 backend 都使用 true-FP32 path，而非 Tensor Core。
  - cuTile 的问题在于 virtual-im2col operand materialization：
    - index computation
    - indirect load
    - mask selection
    - shared/local memory staging
    - register spilling
  - NCU 显示 cuTile 在 **ct.mma** 附近出现大量 **stall_long_scoreboard**。
  - 根因不是 FP32 arithmetic，而是 gather-mask-stage chain 导致 operand 准备路径过重。
  - Triton 直接构造 pointer tile，并将 predicate 附着到 **tl.load**，避免额外 select 和 tile materialization。

- **streamk_matmul/bf16 case**
  - cuTile 较慢的原因是 K-loop 中暴露了显式 **Tensor Core/TMEM pipeline wait**。
  - NCU 将 stall 归因到 predicated **BRA**，但其 predicate 来自：
    - **SYNCS.PHASECHK.TRANS64.TRYWAIT**
  - 表明 branch 实际等待的是 asynchronous MMA/TMEM phase，而不是 scalar loop increment。
  - 优化方向是提高 **MMA/TMEM pipelining**，减少 async phase wait。

- **matmul_int8 case**
  - cuTile 明显快于 Triton：
    - cuTile：**346.88 µs**
    - Triton：**477.44 µs**
    - cuTile 快 **1.38×**
  - SASS 指令数量：
    - cuTile：**78.8M**
    - Triton：**187.0M**
    - cuTile 少 **2.37×**
  - Triton 未使用 **TMA** 或 **tcgen05**，走 legacy：
    - cp.async
    - ldmatrix
    - register-fragment
    - IMMA
  - cuTile 使用 Blackwell native path：
    - **UTMALDG.2D**
    - **UTCIMMA**
    - **TMEM accumulator**
  - 结论：
    - Triton 存在 packed low-bit aware MMA lowering limitation。
    - cuTile 仍存在 shared-memory swizzling/padding 优化空间。

- **flash_attention case**
  - cuTile 使用 **TMA** 与 **TC-Gen05**。
  - cuTile latency：**17.70 ms**
  - Triton latency：**22.77 ms**
  - cuTile 更快，但仍未达到理想性能。
  - cuTile 的主要问题：
    - Tensor Core utilization 仅 **43.2%**
    - register usage 高达 **185**
    - TMA load latency 未被 compute 完全覆盖
    - mbarrier synchronization 引发大量 **stall_long_scoreboard**

- **Bank conflict 分析**
  - 高 bank conflict 并非只属于某一个 backend。
  - top-20 conflict score case 中：
    - cuTile：**6 个**
    - Triton：**14 个**
  - 冲突主要集中在 tile-heavy kernels。
  - 常见来源：
    - shared-memory stores
    - shared-memory loads
    - global-to-shared copies
    - MMA operand staging

![](eecac1f3f846e36c1c4ebbc4ff6729c5eda72ef2563106488aac0344e4ef83b5.jpg)

![](98c4463baedfff6877d1206ce66cafa179292649e329275fea02456eac9f6331.jpg)

- **Triton tensor descriptor/TMA ablation**
  - tensor descriptor 并非普适优化。
  - 对少数 tile-reuse-heavy kernels 有帮助：
    - **flash_attention**
    - **matmul_int8**
    - **streamk_matmul**
  - 对多数 one-pass streaming kernels 会退化：
    - pointwise
    - layout
    - normalization
    - reduction
  - 实用规则：
    - tile movement 可复用或可与 compute overlap 时使用 descriptor/TMA。
    - streaming kernel 优先使用 pointer load。

![](8280c07c994efbcaeb79af11fbfd5ac7be69c82490fbbab5d2e2b7956adea84c.jpg)

- **RQ4：LLM 生成代码的可用性与 Token Efficiency**
  - 四种 model-backend 组合多数都能生成快于 PyTorch 的 kernels：
    - GPT-5.5 + Triton：**39/45**
    - GPT-5.5 + cuTile：**38/45**
    - Claude Opus 4.7 + Triton：**39/45**
    - Claude Opus 4.7 + cuTile：**35/45**
  - 但 **Triton 始终更 token-efficient**。
  - cuTile 消耗更多 tokens，且 speedup per token 更低。

![](336730bb607696cb1e402d8e5305e120d7e5b75eee5b3cbb91f30035d728ffb6.jpg)

| Model + Backend | TokenCost@10 | TokenEfficiency@10 |
|---|---:|---:|
| **GPT-5.5 + Triton** | **0.26M** | **20.9** |
| **GPT-5.5 + cuTile** | **0.32M** | **13.4** |
| **Claude Opus 4.7 + Triton** | **0.28M** | **17.1** |
| **Claude Opus 4.7 + cuTile** | **0.42M** | **8.3** |

- **backend gap 大于 model gap**
  - 同一 model 下，Triton 均比 cuTile 更省 tokens。
  - 同一 backend 下，GPT-5.5 与 Claude 的差距小于 Triton 与 cuTile 的差距。
  - 原因主要来自：
    - Triton 生态更成熟。
    - Triton 在公开代码和训练语料中更常见。
    - cuTile 较新，LLM 先验知识不足。
    - cuTile API 与 tile abstraction 对 LLM 更不友好。

![](e9e45d81864f1e99ab5bd4542c44ef7ee151bd4710d6eae1fd3a21ac42153072.jpg)

---

**结论**

- **TileBench**提供了一个面向 tile-based programming models 的受控 benchmark，覆盖 **45 个 AI kernels**，支持 correctness、performance、roofline、autotuning 和 profiling diagnosis 的统一评估。
- **Triton 与 cuTile 均显著优于 PyTorch**，但不存在绝对优势 backend：
  - **Triton**更适合 irregular、streaming、bandwidth-bound、runtime-indexed kernels。
  - **cuTile**更适合 regular tile reuse、Tensor Core、TMA、Blackwell-native memory movement 友好的 workloads。
- **Autotuning 有收益但不足以接近硬件极限**：
  - Triton geomean gain：**1.20×**
  - cuTile geomean gain：**1.15×**
  - 多数 kernel 仍存在明显 roofline headroom。
- 性能瓶颈更多来自 backend compiler 与 low-level code generation：
  - **lowering path**
  - **instruction selection**
  - **TMA/tcgen05 使用**
  - **shared-memory staging**
  - **bank conflict**
  - **register pressure**
  - **async pipeline overlap**
- **LLM-generated kernel** 评估显示 programming model 会影响代码生成效率：
  - Triton 更容易被 LLM 正确、高效地产生。
  - cuTile 当前 token cost 更高、iteration variance 更大、verification failure 更多。
- TileBench 的主要价值在于：
  - 为 tile-centric DSL 提供公平比较基准。
  - 揭示不同 backend 的适用边界。
  - 支持 compiler 与 DSL 设计的瓶颈定位。
  - 为 LLM-based kernel generation 提供可量化的 backend usability 指标。

---

## 2. 背景知识与核心贡献

**研究背景**

- 现代 **AI workloads** 对自定义 GPU kernel 的依赖快速增加：
  - **LLM**、diffusion models、多模态模型等系统中，常见计算已不再局限于标准 GEMM 或 convolution。
  - 大量关键算子涉及：
    - **Attention variants**
    - **Normalization**
    - **Mixture-of-Experts routing**
    - **Quantization / Dequantization**
    - **Fused point-wise operations**
    - 稀疏访问、动态索引、数据重排等非规则模式

- 传统高性能 GPU kernel 开发仍然门槛很高：
  - **CUDA** 提供对 memory hierarchy、synchronization、Tensor Cores 等底层机制的精细控制。
  - 这种控制能力带来较高工程成本，开发者需要处理：
    - thread/block 映射
    - shared memory layout
    - register pressure
    - bank conflict
    - Tensor Core / TMA 使用
    - instruction scheduling

- **Tile-based programming models** 试图降低这一复杂度：
  - 代表系统包括 **Triton** 和 **cuTile**。
  - 二者都让开发者以 **tile-level abstraction** 编写 kernel，而不是直接管理每个 thread 的 SIMT 细节。
  - 程序员通过 tile 级操作表达：
    - tile load / store
    - tile-level compute
    - matrix multiply accumulation
    - masked memory access
    - Tensor Core 相关计算
  - 编译器和 runtime 负责将这些高级 tile 操作映射到底层 GPU 指令。

- **Triton** 与 **cuTile** 的抽象风格存在差异：
  - **Triton**：
    - Python-embedded DSL。
    - 以 block program、pointer arithmetic、masked load/store 为核心。
    - 更自然地表达不规则地址、runtime-computed indices 和轻量 streaming kernel。
  - **cuTile**：
    - CUDA ecosystem 内的 tile abstraction。
    - 使用 `ct.load`、`ct.mma`、`ct.store` 等 tile primitive。
    - 更贴近 NVIDIA Blackwell 上的 **Tensor Core**、**TMA**、**tcgen05** 等硬件路径。

---

**研究动机**

- 社区缺乏对 tile-centric DSL 的系统性、公平性评估：
  - 现有研究或 benchmark 往往关注单一 backend、单类算子或 LLM 代码生成能力。
  - 很少有工作在相同硬件、相同算子语义、相近实现结构下比较 **Triton** 与 **cuTile**。
  - 不同 benchmark 的输入规模、dtype、计时方法、正确性检查和调优策略不统一，导致结论难以横向比较。

- 实际开发中，开发者不仅关心“谁更快”，还需要知道：
  - 哪类 workload 更适合 **Triton**。
  - 哪类 workload 更适合 **cuTile**。
  - 性能差异来自 tile size、occupancy，还是 compiler lowering。
  - bottleneck 是 memory bandwidth、Tensor Core utilization、TMA latency、bank conflict，还是 register spilling。
  - LLM 是否更容易生成某种 DSL 的正确高性能 kernel。

- **cuTile** 作为较新的 NVIDIA tile-based programming model，需要更全面的实践评估：
  - 既要验证其在 Blackwell B200 上使用 **TMA / Tensor Core / tcgen05** 的潜力。
  - 也要识别其在不规则访问、动态索引、轻量 bandwidth-bound kernel 上的局限。

- **Triton** 虽已广泛用于生产系统，但性能高度依赖：
  - compiler version
  - autotuning space
  - implementation details
  - tensor descriptor / pointer-based load 选择
  - kernel fusion 方式

- 因此，论文提出 **TileBench**，目标是构建一个受控 benchmark，用于：
  - 公平比较 **Triton** 与 **cuTile**。
  - 衡量 default configuration 与 autotuned configuration 的差距。
  - 使用 roofline 和 profiling 诊断性能瓶颈。
  - 分析 LLM 生成不同 DSL kernel 的难易度和 token efficiency。

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

---

**核心贡献**

- 提出 **TileBench**：
  - 一个面向 tile-based GPU programming models 的受控 benchmark。
  - 在单张 **NVIDIA B200 GPU** 上系统比较 **Triton** 与 **cuTile**。
  - 覆盖 **45 个 AI operator**。
  - 每个 operator 均包含：
    - **PyTorch reference implementation**
    - 手写且验证正确的 **Triton implementation**
    - 手写且验证正确的 **cuTile implementation**
    - 标准化 `config.yaml`
    - dtype sweep
    - input-size sweep
    - default configuration
    - autotuned configuration
    - FLOP / HBM byte analytical formula
    - roofline-based metrics

- 构建覆盖广泛的算子集合：
  - 算子来源包括 **TritonBench** 与 **LeetGPU**。
  - 覆盖主流 AI kernel 模式：
    - point-wise
    - reduction / normalization
    - matrix multiplication / attention
    - stencil / convolution
    - data layout / indexing / sorting

| 类别 | 算子数量 | 代表算子 |
|---|---:|---|
| **Point-wise** | 12 | `vector_add`, `relu`, `swiglu`, `weight_dequant` |
| **Reduction/Normalization** | 11 | `softmax`, `layernorm`, `argmax`, `moe_topk_gating` |
| **Matrix Multiplication/Attention** | 8 | `matmul`, `matmul_int8`, `flash_attention`, `linear_self_attention` |
| **Stencil/Convolution** | 6 | `1d_conv`, `3d_conv`, `gaussian_blur`, `jacobi_stencil_2d` |
| **Data Layout** | 8 | `matrix_copy`, `matrix_transpose`, `bitonic_sort`, `radix_sort` |
| **Total** | **45** | — |

- 建立统一评估流程：
  - 使用 **PyTorch** 作为语义参考、正确性 oracle 和 baseline。
  - 所有 backend 使用相同输入 case、dtype、计时协议和正确性检查。
  - 计时采用：
    - **Proton profiler**
    - CUDA graph capture-and-replay
    - L2 cache flushing
    - 20 次 warmup
    - 100 次 timed runs
  - 性能指标包括：
    - latency
    - speedup over PyTorch
    - TFLOPS
    - effective bandwidth
    - arithmetic intensity
    - roofline utilization

- 系统评估 default 与 autotuned 配置：
  - 对 **Triton** 与 **cuTile** 分别提供 default path 和 autotuned path。
  - 通过可比的 tuning budget 比较两者在各自 native tuning knobs 下的最佳表现。
  - 发现 autotuning 有收益，但不能解决大多数性能瓶颈：
    - **Triton** roofline utilization 几何平均提升 **1.20×**。
    - **cuTile** roofline utilization 几何平均提升 **1.15×**。
    - autotuned 后仅 **8/45** 个 Triton kernel 和 **4/45** 个 cuTile kernel 达到至少 **80% roofline utilization**。

| 指标 | **Triton** | **cuTile** |
|---|---:|---:|
| Default 几何平均 speedup over PyTorch | **2.7×** | **2.2×** |
| Default 中快于 PyTorch 的算子数 | **36/45** | **34/45** |
| Median speedup over PyTorch | **3.1×** | **2.7×** |
| Autotune roofline gain | **1.20×** | **1.15×** |
| Autotuned 后 ≥80% roofline utilization 的算子数 | **8/45** | **4/45** |

- 给出 workload-dependent 的性能结论：
  - **Triton** 整体更稳健，尤其适合：
    - irregular access
    - streaming kernel
    - bandwidth-bound operator
    - sparse layout
    - runtime-computed indices
    - low-bit packing / unpacking
  - **cuTile** 在小部分硬件友好 workload 上更强，尤其适合：
    - dense GEMM
    - attention
    - stencil / convolution
    - Tensor Core-friendly computation
    - TMA-friendly reusable tile movement
    - 静态 affine tile-space access

- 引入 profiling-guided bottleneck diagnosis：
  - 使用 **Nsight Compute**、PTX/SASS inspection 分析大性能差距案例。
  - 诊断维度包括：
    - TMA / tcgen05 使用
    - Tensor Core path
    - shared memory bank conflict
    - register pressure
    - local/shared memory staging
    - instruction mix
    - async MMA / TMEM pipeline stall
  - 典型发现：
    - cuTile 在规则 tile 访问下可自然映射到 Blackwell-native TMA / Tensor Core 路径。
    - Triton 在 pointer-based masked load 上更适合表达不规则访问。
    - cuTile 的 `ct.gather`、`ct.where` 等路径在不规则场景可能带来更重的 producer chain。
    - Triton tensor descriptor / TMA 并非总是有益，只适合 tile reuse 或可与计算重叠的数据搬运场景。

![](f9e643fecb6869d7899b6b7a50bbcaf63d4acc756e36ecba362971843a1ab893.jpg)

- 增加 **LLM code-generation track**：
  - 使用相同 operator set 评估 LLM 生成 **Triton** 与 **cuTile** kernel 的能力。
  - 每个 backend 进行 10 次 iterative refinement。
  - 评估指标不仅包括正确性和性能，还包括：
    - **TokenCost@10**
    - **BestSpeedup@10**
    - **TokenEfficiency@10**
  - 发现 **Triton** 对 LLM 更友好、更 token-efficient：
    - GPT-5.5 + Triton 平均 token cost 为 **0.26M**。
    - GPT-5.5 + cuTile 增至 **0.32M**。
    - Claude Opus 4.7 + Triton 为 **0.28M**。
    - Claude Opus 4.7 + cuTile 增至 **0.42M**。
    - Triton 的 token efficiency 明显高于 cuTile。

| Model + Backend | Mean TokenCost@10 | Mean TokenEfficiency@10 |
|---|---:|---:|
| **GPT-5.5 + Triton** | **0.26M** | **20.9** |
| **GPT-5.5 + cuTile** | **0.32M** | **13.4** |
| **Claude Opus 4.7 + Triton** | **0.28M** | **17.1** |
| **Claude Opus 4.7 + cuTile** | **0.42M** | **8.3** |

![](336730bb607696cb1e402d8e5305e120d7e5b75eee5b3cbb91f30035d728ffb6.jpg)

---

**核心结论**

- **TileBench** 的主要价值在于提供了一个受控、可复现、跨 DSL 的评估框架。
- **Triton** 与 **cuTile** 都能显著快于 PyTorch，但不存在绝对优胜者。
- **Triton** 更适合不规则、轻量、streaming、bandwidth-bound 场景。
- **cuTile** 更适合静态 tile 结构清晰、可复用数据搬运、Tensor Core / TMA 友好的场景。
- **Autotuning** 能带来增益，但大多数 kernel 的性能瓶颈来自 compiler lowering、memory staging、instruction selection 和 shared memory behavior，而非 tile 参数本身。
- 对 **LLM-generated kernels** 而言，**Triton** 当前比 **cuTile** 更容易生成正确且高性能的实现，也更节省 token。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体技术架构**

- **TileBench**是一个面向**tile-based GPU programming models**的受控评测框架，核心目标是在同一硬件、同一算子语义、相近实现结构和统一测量协议下，对**Triton**与**cuTile**进行性能比较、瓶颈诊断和LLM代码生成可用性评估。
- 系统整体由三条主线组成：
  - **Benchmark Construction**：构建统一算子任务、参考实现、Triton/cuTile实现与配置文件。
  - **Benchmark Execution**：在统一harness中完成正确性验证、编译、运行、计时、autotuning。
  - **Comparative Analysis**：汇总性能指标、roofline指标，并结合**Nsight Compute**、**PTX/SASS**进行瓶颈诊断。
- 论文的整体架构可以理解为一个“**统一任务规范 + 双后端实现 + 标准化测量 + 硬件感知分析 + LLM生成评估**”的闭环系统。

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

---

**架构分层**

| 层级 | 主要组件 | 技术作用 |
|---|---|---|
| **任务定义层** | 45个AI算子、PyTorch reference、config.yaml | 统一算子语义、输入规模、dtype、误差容忍度和分析公式 |
| **实现后端层** | **Triton implementation**、**cuTile implementation** | 在两种tile-centric DSL中实现相同算子逻辑 |
| **执行评测层** | TileBench harness、Proton profiler、CUDA graph、L2 flush | 统一编译、验证、计时和性能采样流程 |
| **调优层** | Default configuration、Autotuned configuration | 比较手工默认配置与后端可暴露参数搜索后的性能 |
| **分析诊断层** | Roofline model、NCU、PTX/SASS inspection | 解释性能差异来源，如TMA、Tensor Core、bank conflict、register pressure |
| **LLM生成层** | GPT-5.5、Claude Opus 4.7、10轮迭代生成 | 评估Triton/cuTile作为LLM kernel生成目标的性能与Token效率 |

---

**Benchmark Construction：任务构建阶段**

- TileBench从两个已有benchmark中选取算子语义：
  - **TritonBench**：26个算子。
  - **LeetGPU**：19个算子。
- 所有任务被重构为统一格式，而不是直接沿用原benchmark的脚本：
  - 每个算子包含：
    - **impl_torch.py**：PyTorch语义参考实现。
    - **impl_triton.py**：人工编写并验证的Triton实现。
    - **impl_cutile.py**：人工编写并验证的cuTile实现。
    - **config.yaml**：声明dtype、case grid、验证容差、计时参数、FLOP/byte公式。
- 构建阶段强调**matched semantics**和**comparable implementation structures**：
  - 保证Triton和cuTile实现相同数学语义。
  - 保证输入形状、dtype sweep、验证逻辑和性能统计一致。
  - 避免因benchmark协议差异导致不公平比较。

---

**算子覆盖结构**

| 类别 | 算子数量 | 代表算子 | 主要覆盖模式 |
|---|---:|---|---|
| **Point-wise** | 12 | vector_add、relu、swiglu、weight_dequant | elementwise、fused pointwise、量化/反量化 |
| **Reduction/Normalization** | 11 | softmax、layernorm、argmax、moe_topk_gating | reduction、normalization、loss、top-k gating |
| **Matrix Multiplication/Attention** | 8 | matmul、matmul_int8、flash_attention、linear_self_attention | GEMM、Attention、Tensor Core路径 |
| **Stencil/Convolution** | 6 | 1d_conv、3d_conv、gaussian_blur、jacobi_stencil_2d | stencil、convolution、局部邻域访问 |
| **Data Layout** | 8 | matrix_copy、matrix_transpose、bitonic_sort、radix_sort | copy、transpose、sort、scatter/gather |
| **Total** | 45 | — | 多种AI kernel访问与计算模式 |

---

**Benchmark Execution：统一执行阶段**

- 执行阶段由统一的TileBench harness控制，避免不同后端采用不同测量逻辑。
- 每个算子的执行流程包括：
  - 使用**PyTorch reference**生成正确输出与baseline latency。
  - 编译并运行**Triton kernel**。
  - 编译并运行**cuTile kernel**。
  - 将两个后端输出与PyTorch结果比较：
    - 浮点输出使用**torch.testing.assert_close**。
    - 整数输出使用精确比较。
  - 对通过验证的实现进行标准化计时。
- 计时协议包括：
  - **Proton profiler**。
  - **CUDA graph capture-and-replay**。
  - **L2-cache flushing**。
  - **20 warmup runs**。
  - **100 timed runs**。
  - 统计**mean kernel latency**。
- 执行结果用于计算：
  - **Latency**。
  - **Speedup over PyTorch**。
  - **Achieved TFLOPS**。
  - **Effective bandwidth**。
  - **Arithmetic intensity**。
  - **Roofline utilization**。

---

**配置与标准化机制**

- **config.yaml**是TileBench架构中的关键控制文件，承担任务元数据和分析公式的统一声明。
- 每个算子独立配置，而不是使用全局固定配置：
  - 不同算子具有不同shape敏感性。
  - matmul需要覆盖Tensor Core有效区间。
  - reduction需要覆盖occupancy饱和区间。
  - elementwise需要覆盖memory-bound到large-scale streaming场景。
- config.yaml通常包含：
  - **dtype list**。
  - **case grid**。
  - **verification tolerance**。
  - **benchmark timing options**。
  - **flops_expr**。
  - **bytes_expr**。
- 该设计使所有后端在相同输入和相同分析公式下评估，提升可复现性与公平性。

---

**双后端实现架构：Triton与cuTile**

| 后端 | 编程抽象 | 优势场景 | 局限场景 |
|---|---|---|---|
| **Triton** | Python-embedded block program、pointer arithmetic、tl.load/tl.store/tl.dot | irregular access、masked load、streaming kernel、bandwidth-bound算子 | 对Blackwell原生TMA/tcgen05路径利用不总是充分 |
| **cuTile** | CUDA生态中的tile operations，如ct.load、ct.mma、ct.store | regular tile、Tensor Core、TMA-friendly、tile reuse-heavy workload | 对runtime-computed indices、ct.gather、ct.where等不规则模式表达较重 |

- 两个后端都采用tile-centric编程模型：
  - 用户以tile为基本单位加载、计算和存储。
  - 编译器负责将tile操作lower到GPU执行路径。
- 架构设计并不追求证明某一后端绝对更优，而是识别：
  - 哪类算子适合**Triton pointer-level flexibility**。
  - 哪类算子适合**cuTile static tile abstraction**。
  - 性能差异是否来自编程模型、compiler lowering、memory staging或硬件指令路径。

---

**Autotuning架构**

- TileBench为人工实现提供两种执行路径：
  - **Default path**：
    - 使用单个手工选择配置。
    - 代表合理开发者初始实现。
  - **Autotuned path**：
    - 在预定义搜索空间中寻找最佳配置。
    - 用于衡量参数搜索带来的增益。
- 两个后端的搜索参数不同，但搜索预算与粒度尽量保持可比：
  - Triton调优参数：
    - **BLOCK_SIZE**。
    - **num_warps**。
    - **num_stages**。
  - cuTile调优参数：
    - **TILE_SIZE**。
    - **occupancy**。
- 该层的作用不是搜索完全不同算法，而是评估：
  - tile shape选择是否影响性能。
  - occupancy和pipeline参数能否缩小roofline差距。
  - 剩余瓶颈是否超出autotuning可控范围。

---

**Roofline与硬件感知分析架构**

- TileBench使用**roofline model**对性能进行硬件归一化。
- 每个case由config.yaml中的公式计算：
  - **FLOP count**。
  - **HBM byte count**。
  - **Arithmetic intensity**。
- 使用B200硬件参数计算理论上限：
  - **peak bandwidth**：6539.4 GB/s。
  - dtype-specific **peak TFLOPS**。
- 核心指标包括：
  - **Achieved TFLOPS**：
    - 衡量实际计算吞吐。
  - **Effective bandwidth**：
    - 衡量实际内存带宽利用。
  - **Roofline utilization**：
    - 衡量相对硬件理论上限的利用率。
- Roofline层帮助区分：
  - 算子是**compute-bound**还是**memory-bound**。
  - 性能差距来自tile参数不足，还是compiler lowering、memory staging、shared memory layout等更底层因素。

---

**B200硬件规格抽象**

| 参数 | 数值 |
|---|---:|
| **GPU** | NVIDIA B200 |
| **HBM3e容量** | 180 GB |
| **Measured peak bandwidth** | 6539.4 GB/s |
| **FP32/TF32 peak** | 1100 TFLOPS |
| **FP16/BF16 peak** | 2250 TFLOPS |
| **FP8 peak** | 4500 TFLOPS |
| **INT8 peak** | 4500 TFLOPS |

---

**Profiling-guided Diagnosis：瓶颈诊断架构**

- 对性能差距较大的case，TileBench进入深入诊断流程。
- 诊断工具包括：
  - **Nsight Compute**：
    - 分析warp stall、bank conflict、memory traffic、occupancy、register usage。
  - **PTX/SASS inspection**：
    - 检查实际生成指令。
    - 观察是否使用TMA、tcgen05、MMA、cp.async、ldmatrix等路径。
- 诊断关注点包括：
  - **Instruction selection**。
  - **Tensor Core usage**。
  - **TMA usage**。
  - **TMEM pipeline wait**。
  - **shared-memory bank conflict**。
  - **register pressure**。
  - **register spilling**。
  - **local/shared memory staging**。
- 该层将性能结果转化为可操作结论：
  - cuTile适合规则tile与TMA路径。
  - Triton适合不规则地址和masked load。
  - autotuning无法解决compiler lowering路径不佳的问题。
  - bank conflict和operand staging可能成为主要瓶颈。

---

**LLM Code-generation Track架构**

- 除人工实现主评测外，TileBench还设计了独立的**iterative LLM code-generation track**。
- 目标是评估Triton与cuTile作为LLM生成目标时的：
  - **Correctness**。
  - **Runtime performance**。
  - **Token cost**。
  - **Token efficiency**。
- 每个算子执行10轮迭代生成：
  - Iteration 0输入：
    - TileBench framework guide。
    - Triton API reference。
    - cuTile API reference。
    - operator description。
    - config.yaml。
    - PyTorch reference。
    - output-format instructions。
  - Iteration N输入：
    - 之前轨迹。
    - 编译/验证反馈。
    - speedup。
    - roofline utilization。
    - worst-first性能摘要。
    - 上一轮源码。
    - 必要时提供best verify-clean源码作为回退参考。
- LLM track与主benchmark track隔离：
  - LLM看不到人工实现。
  - LLM生成代码不使用backend autotuning。
  - 每轮必须提交单一配置。
- 系统会拒绝reward hacking：
  - 禁止调用PyTorch、cuBLAS、cuDNN或reference实现代替kernel。
  - 禁止基于tensor identity、data pointer等缓存输出。
  - 禁止导入autotuner绕过迭代搜索。

---

**LLM评估指标结构**

| 指标 | 含义 |
|---|---|
| **BestSpeedup@10** | 10轮内通过正确性验证的最佳speedup |
| **TokenCost@10** | 10轮累计消耗Token数 |
| **TokenEfficiency@10** | 每百万Token获得的最佳正确性门控speedup |
| **Correctness-gated speedup** | 未编译或验证失败的迭代记为0 speedup，但仍计入Token成本 |

- 该指标设计强调：
  - 错误代码也消耗搜索成本。
  - 性能必须以正确性为前提。
  - 编程模型本身会影响LLM生成效率。

---

**数据流与控制流概括**

- TileBench整体数据流如下：
  - **Operator source selection**
    - 从TritonBench和LeetGPU选取算子语义。
  - **Task normalization**
    - 统一为PyTorch reference、Triton实现、cuTile实现和config.yaml。
  - **Input generation**
    - 根据case grid和dtype生成多组输入。
  - **Correctness checking**
    - 以PyTorch输出为oracle。
  - **Kernel timing**
    - 使用统一profiling协议测量mean latency。
  - **Metric computation**
    - 计算speedup、TFLOPS、bandwidth、AI、roofline utilization。
  - **Autotuning**
    - 在各后端原生参数空间中搜索最佳配置。
  - **Profiling diagnosis**
    - 对大gap case进行NCU和SASS分析。
  - **LLM generation evaluation**
    - 评估LLM在Triton/cuTile上的生成表现和Token效率。

---

**核心设计取舍**

- **公平性**
  - 通过统一语义、统一输入、统一计时、统一metric公式降低benchmark偏差。
- **可诊断性**
  - 不只报告latency，而是结合roofline、NCU和SASS解释性能差异。
- **后端原生性**
  - 不强行让Triton和cuTile使用完全相同参数，而是在相近预算下使用各自自然的调优接口。
- **硬件关联性**
  - 评测绑定B200，重点观察Blackwell上的TMA、Tensor Core、tcgen05、TMEM行为。
- **生成式可用性**
  - 将LLM代码生成纳入架构，衡量编程模型对自动kernel开发的友好程度。

---

**架构总结**

- **TileBench**不是单纯的算子性能排行榜，而是一个面向tile-based DSL的系统化评测与诊断平台。
- 其整体架构以**45个标准化算子任务**为基础，以**PyTorch reference**保证语义正确，以**Triton/cuTile双后端**进行受控对比，以**Proton + CUDA graph**进行统一计时，以**roofline + NCU + SASS**解释性能瓶颈，并额外通过**LLM iterative generation**评估编程模型的自动生成友好性。
- 架构核心价值在于同时回答三个问题：
  - **哪个后端更快**。
  - **为什么更快或更慢**。
  - **哪个后端更容易被LLM生成出正确且高性能的kernel**。

### 1. Controlled Tile-Based Benchmark Suite

**核心定位**

- **Controlled Tile-Based Benchmark Suite**是 TileBench 的基础设计：用同一套受控任务、输入、验证、计时和指标体系，对比 **Triton** 与 **cuTile** 两类 tile-based GPU programming models。
- 该套件的核心目标不是简单跑分，而是隔离**编程模型差异**：
  - 避免不同 operator 语义不一致导致的性能误判。
  - 避免不同输入规模、dtype、计时协议导致的不可比。
  - 避免只比较 GEMM 或 Attention 等少数“友好 workload”带来的偏置。
  - 支持从**性能评估**进一步走向**bottleneck diagnosis**。
- TileBench 包含 **45 个 AI operators**：
  - 每个 operator 都有 **PyTorch reference** 作为语义基准。
  - 每个 operator 都有人工编写并验证的 **Triton implementation** 与 **cuTile implementation**。
  - 每个 operator 都有独立的 **config.yaml**，声明 dtype、input-size sweep、验证容差、计时参数、FLOP/HBM-byte 公式。
  - 每个 backend 都支持 **default configuration** 与 **autotuned configuration** 两条路径。

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

---

**设计动机**

- Tile-based DSL 的性能很难直接比较，原因包括：
  - **Triton** 与 **cuTile** 暴露的编程抽象不同。
    - Triton 更偏向 **pointer-level block program**。
    - cuTile 更偏向 **static tile abstraction**，通过 `ct.load`、`ct.mma`、`ct.store` 表达 tile 操作。
  - 不同 backend 的 tuning knob 不完全对应。
    - Triton 常调 `BLOCK_SIZE`、`num_warps`、`num_stages`。
    - cuTile 常调 `TILE_SIZE`、`occupancy`。
  - operator 类型差异极大。
    - Dense GEMM 与 Attention 可能受益于 Tensor Core、TMA。
    - Point-wise、Reduction、Data Layout 更容易受 memory bandwidth、mask、indexing、control flow 影响。
  - 只用 PyTorch baseline 无法说明 backend 间差异来自哪里。
    - PyTorch 可能调用高性能库，也可能执行较弱 reference path。
    - TileBench 因此把 PyTorch 定位为**语义基准与软件 baseline**，而不是唯一性能真值。

---

**Benchmark Suite 组成**

| 维度 | TileBench 设计 |
|---|---|
| Operator 数量 | **45 个 AI operators** |
| 来源 | **26 个来自 TritonBench**，**19 个来自 LeetGPU** |
| 主要 backend | **Triton**、**cuTile** |
| 语义基准 | **PyTorch reference implementation** |
| 每个任务文件 | `impl_torch.py`、`impl_triton.py`、`impl_cutile.py`、`config.yaml` |
| 输入覆盖 | 每个 supported dtype 下 sweep **20 个 input configurations** |
| 计时方式 | Proton profiler、CUDA graph capture-and-replay、L2 flush、20 warmup、100 timed runs |
| 指标 | Latency、Speedup、TFLOPS、Effective Bandwidth、Arithmetic Intensity、Roofline Utilization |
| 调优路径 | Default path 与 Autotuned path |
| 诊断工具 | Nsight Compute、PTX/SASS inspection |

---

**Operator 覆盖范围**

| 类别 | 数量 | 代表 operator | 主要压力来源 |
|---|---:|---|---|
| **Point-wise** | 12 | `vector_add`、`relu`、`swiglu`、`weight_dequant` | HBM bandwidth、coalescing、low-bit packing |
| **Reduction/Normalization** | 11 | `softmax`、`layernorm`、`argmax`、`moe_topk_gating` | reduction tree、warp-level parallelism、shared memory |
| **Matrix Multiplication/Attention** | 8 | `matmul`、`matmul_int8`、`flash_attention`、`linear_self_attention` | Tensor Core、TMA、tile reuse、K-loop pipeline |
| **Stencil/Convolution** | 6 | `1d_conv`、`3d_conv`、`gaussian_blur`、`jacobi_stencil_2d` | halo access、boundary predicate、operand staging |
| **Data Layout** | 8 | `matrix_copy`、`matrix_transpose`、`bitonic_sort`、`radix_sort` | gather/scatter、transpose、irregular memory access |

- 这 5 类 operator 覆盖了 AI kernel 中常见的计算与访存模式：
  - **高算术强度**：GEMM、Attention。
  - **低算术强度**：Point-wise、copy、transpose。
  - **不规则访问**：sparse attention、sort、top-k、weight dequant。
  - **边界敏感访问**：convolution、stencil、pooling。
  - **归约密集访问**：softmax、layernorm、argmax。

---

**输入输出关系**

- 每个 operator 的输入输出关系由 **PyTorch reference** 定义。
  - `impl_torch.py` 是 correctness oracle。
  - Triton/cuTile 实现必须与 PyTorch reference 在语义上匹配。
  - Floating-point 输出使用 `torch.testing.assert_close`。
  - Integer 输出使用 exact comparison。
- 每个 operator 的输入规模由 **config.yaml** 控制。
  - dtype 列表是 operator-specific。
  - case grid 是 operator-specific。
  - 每个 supported dtype 会评估 **20 个 input configurations**。
- 每个 backend 的输出必须满足：
  - 输出 tensor shape 与 PyTorch reference 一致。
  - 输出 dtype 与语义要求一致。
  - 数值误差满足 config 中的 tolerance。
  - 不允许调用 PyTorch、cuBLAS、cuDNN 或 reference implementation 代替 kernel computation。

---

**单个 Operator 的标准结构**

| 文件 | 作用 |
|---|---|
| `impl_torch.py` | 定义 operator 的语义、输入输出 contract、baseline latency |
| `impl_triton.py` | Triton kernel 与 Python `run()` wrapper |
| `impl_cutile.py` | cuTile kernel 与 Python `run()` wrapper |
| `config.yaml` | dtype、case grid、verify tolerance、timing options、FLOP/byte 公式 |

- `impl_torch.py`
  - 负责提供 reference output。
  - 负责作为 speedup denominator。
  - 不要求是最高性能实现，而是语义基准。
- `impl_triton.py`
  - 使用 `@triton.jit` 编写 kernel。
  - 通过 block-level pointer arithmetic 和 mask 表达 tile access。
  - 常见参数包括 `BLOCK_SIZE`、`BLOCK_M`、`BLOCK_N`、`BLOCK_K`、`num_warps`、`num_stages`。
- `impl_cutile.py`
  - 使用 `@ct.kernel` 编写 kernel。
  - 通过 `ct.load`、`ct.mma`、`ct.store` 表达 tile-level 操作。
  - 常见参数包括 `TILE`、tile shape、`occupancy`。
- `config.yaml`
  - 是受控 benchmark 的关键。
  - 它将不同 operator 的实验设置显式化，避免隐藏 benchmark policy。

---

**算法流程**

- TileBench 的 pipeline 分为三个阶段：

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| **Benchmark Construction** | TritonBench、LeetGPU operator semantics | 统一 task format，重写 PyTorch/Triton/cuTile 实现 | 45 个标准化 operator tasks |
| **Benchmark Execution** | Operator task、dtype、case grid | 编译、运行、验证、计时 default/autotuned kernel | latency、correctness、config、runtime statistics |
| **Comparative Analysis** | Timing result、FLOP/byte formula、hardware peak | 计算 speedup、TFLOPS、bandwidth、roofline utilization，选择 case 做 NCU/SASS 分析 | backend comparison 与 bottleneck diagnosis |

- 单个 case 的执行流程：
  - 根据 `config.yaml` 生成输入 tensor。
  - 执行 `impl_torch.run()`：
    - 得到 reference output。
    - 得到 PyTorch baseline latency。
  - 执行 `impl_triton.run()`：
    - 编译 Triton kernel。
    - 运行一次 correctness verification。
    - 若通过验证，则进入 timed runs。
  - 执行 `impl_cutile.run()`：
    - 编译 cuTile kernel。
    - 运行 correctness verification。
    - 若通过验证，则进入 timed runs。
  - 根据 `flops_expr` 与 `bytes_expr` 计算：
    - **Achieved TFLOPS**
    - **Effective Bandwidth**
    - **Arithmetic Intensity**
    - **Roofline Utilization**
  - 汇总到 operator-level 与 suite-level 结果。

---

**参数设置机制**

- TileBench 的参数设置遵循两个原则：
  - **语义完全匹配**：
    - Triton 与 cuTile 的输出必须对齐 PyTorch reference。
    - dtype 与 input shape 由同一份 config 控制。
  - **实现结构可比**：
    - 两个 backend 尽量采用相同算法结构。
    - tuning budget 与 search granularity 尽量保持可比。
    - 不强制一一映射 tuning knob，因为 Triton 与 cuTile 的抽象不同。

| 参数类别 | Triton 示例 | cuTile 示例 | 作用 |
|---|---|---|---|
| Tile size | `BLOCK_SIZE`、`BLOCK_M`、`BLOCK_N`、`BLOCK_K` | `TILE`、`tm`、`tn`、`tk` | 控制每个 program/CTA 处理的数据块 |
| 并行度 | `num_warps` | `occupancy` | 控制 CTA 内执行资源与 occupancy |
| Pipeline | `num_stages` | backend-specific scheduling hints | 控制 memory/computation overlap |
| Boundary handling | `mask` in `tl.load/tl.store` | `padding_mode`、OOB store drop | 处理非整除 shape 与边界条件 |
| Tensor Core path | `tl.dot` | `ct.mma` | 触发 MMA/Tensor Core |
| TMA path | `tl.make_tensor_descriptor` | `ct.load(..., allow_tma=True)` | 利用 Tensor Memory Accelerator |

---

**Default 与 Autotuned 路径**

- **Default path**
  - 每个 operator 使用一个人工选择的固定配置。
  - 不是随机 baseline，也不是故意弱配置。
  - 目标是代表有经验开发者给出的合理初始 kernel。
- **Autotuned path**
  - 在人工定义的候选配置空间中搜索。
  - Triton 与 cuTile 使用各自 native tuning interface。
  - 搜索空间保持“可比”，但不要求参数完全一一对应。
- 关键限制：
  - Autotuning 只改变 tile shape、parallelism、occupancy 等参数。
  - Autotuning 不改变算法结构。
  - Autotuning 不改变 compiler lowering path。
  - 因此它能揭示 tuning headroom，但不能解决所有 backend codegen 问题。

| Backend | Default 常见参数 | Autotune 常见搜索项 |
|---|---|---|
| **Triton** | `BLOCK_SIZE`、`num_warps`、`num_stages` | tile size、warp 数、pipeline stage |
| **cuTile** | `TILE_SIZE`、`occupancy` | tile shape、occupancy |
| **共同目标** | 合理初始性能 | 搜索 backend 暴露的参数空间上限 |

---

**指标计算逻辑**

- TileBench 使用统一公式计算硬件相关指标，避免不同 backend 自定义指标造成偏差。
- 对每个 operator `o`、backend `b`、dtype `d`、case `c`，记录平均 kernel latency：
  - **Tₒ,b,d,c**
- Speedup over PyTorch：
  - **PyTorch latency / backend latency**
- Achieved throughput：
  - **TFLOPS = FLOPs / latency**
- Arithmetic intensity：
  - **AI = FLOPs / HBM bytes**
- Roofline utilization：
  - **R = Achieved TFLOPS / min(dtype peak TFLOPS, AI × peak HBM bandwidth)**

| 指标 | 含义 | 用途 |
|---|---|---|
| **Latency** | kernel 平均运行时间 | 直接性能对比 |
| **Speedup over PyTorch** | 相对 PyTorch 的加速比 | 判断 custom kernel 是否有价值 |
| **TFLOPS** | 实际计算吞吐 | 衡量 compute utilization |
| **Effective Bandwidth** | 实际访存带宽 | 衡量 memory-bound kernel |
| **Arithmetic Intensity** | FLOPs/byte | 判断 kernel 偏 compute-bound 还是 memory-bound |
| **Roofline Utilization** | 相对硬件上限的利用率 | 判断距离 speed-of-light 的差距 |

---

**B200 硬件参数**

| 项目 | 数值 |
|---|---:|
| GPU | **NVIDIA B200** |
| HBM | **180 GB HBM3e** |
| Measured peak bandwidth | **6539.4 GB/s** |
| FP32/TF32 peak | **1100 TFLOPS** |
| FP16/BF16 peak | **2250 TFLOPS** |
| FP8 peak | **4500 TFLOPS** |
| INT8 peak | **4500 TFLOPS** |

- TileBench 在 roofline 计算中使用 dtype-specific peak。
- FP32 dot/MMA workload 使用 **TF32 Tensor Core dense ceiling**，因为 Triton `tl.dot` 默认会对 FP32 输入使用 TF32。
- HBM bandwidth 使用实测值 **6539.4 GB/s**，不是直接引用 datasheet。

---

**标准化 dtype 与 input-size sweep**

- 每个 operator 的 dtype 支持是独立声明的。
  - GEMM/Attention 可能覆盖 `fp16`、`bf16`、`fp8`、`int8`。
  - Point-wise 与 Reduction 可能覆盖 `fp16`、`fp32` 等。
  - Sort、indexing、histogram 等可能涉及 integer dtype。
- 每个 dtype 下 sweep 20 个 input cases。
  - 小规模 case 测试 launch overhead 与 under-occupancy。
  - 中等规模 case 测试 cache 与 occupancy 转折点。
  - 大规模 case 测试 HBM bandwidth、Tensor Core utilization、roofline behavior。
- 这种设计使 benchmark 能覆盖：
  - **small-to-large scaling**
  - **B200 roofline behavior**
  - **LLM-like regimes**
  - **operator-specific shape sensitivity**

---

**正确性控制**

- Correctness checking 是 Controlled Benchmark Suite 的核心约束。
- 检查策略：
  - Floating-point 输出：
    - 使用 `torch.testing.assert_close`。
    - tolerance 由 dtype 与 operator 决定。
    - Reduction 类 operator 可放宽 tolerance，以吸收不同 reduction order 的误差。
  - Integer 输出：
    - 使用 exact comparison。
    - `atol = 0`，`rtol = 0`。
- 设计意义：
  - 防止 backend 通过近似错误结果换取速度。
  - 防止不同 backend 实现了不同语义。
  - 防止 LLM-generated kernel reward hacking。

---

**计时协议**

- TileBench 使用统一计时协议，降低 measurement noise。
- 核心设置：
  - **Proton profiler**
  - **CUDA graph capture-and-replay**
  - **L2-cache flushing**
  - **20 warmup runs**
  - **100 timed runs**
  - 记录 mean kernel latency
- 设计原因：
  - CUDA graph 降低 launch overhead 的波动。
  - L2 flush 避免 cache residency 导致的非公平收益。
  - warmup 避免首次编译、lazy initialization、clock ramp 影响。
  - 多次 timed run 提高统计稳定性。

---

**在整体系统中的作用**

- Controlled Tile-Based Benchmark Suite 是 TileBench 的主干模块，承担四类作用：
  - **语义锚点**
    - PyTorch reference 统一定义 operator 行为。
    - Triton/cuTile 输出必须对齐 reference。
  - **公平对比平台**
    - 相同 dtype、shape、timing、metric。
    - 同类算法结构与可比 tuning budget。
  - **性能画像工具**
    - 从 latency 扩展到 TFLOPS、bandwidth、roofline utilization。
    - 能识别 operator 是 memory-bound、compute-bound 还是 codegen-bound。
  - **诊断入口**
    - 对性能差距大的 case 使用 NCU、PTX/SASS 分析。
    - 关联 source-level tile abstraction 与 hardware-level bottleneck。

---

**与 RQ1/RQ2/RQ3/RQ4 的关系**

| Research Question | Benchmark Suite 的作用 |
|---|---|
| **RQ1: Programming-model Performance** | 用 45 个 matched operators 比较 Triton/cuTile/PyTorch |
| **RQ2: Autotuning and Headroom** | 通过 default 与 autotuned path 测量 tuning gain 与 roofline gap |
| **RQ3: Code-level Diagnosis** | 选取 large-gap cases，用 NCU/SASS 解释 backend 差异 |
| **RQ4: LLM Usability** | 复用同一 operator suite，评估 LLM 生成 Triton/cuTile kernel 的 correctness、speedup、token efficiency |

---

**关键实现原则**

- **Matched semantics**
  - 所有 backend 必须实现同一 PyTorch reference 语义。
  - 不允许 backend-specific 语义简化。
- **Comparable implementation structures**
  - Triton 与 cuTile 尽量使用同一算法骨架。
  - 避免一个 backend 用高级库，另一个 backend 手写 kernel。
- **Standardized sweeps**
  - dtype 与 input sizes 由 `config.yaml` 统一控制。
  - 不继承原始 TritonBench 或 LeetGPU 的不一致 case 设计。
- **Shared timing harness**
  - 同一 profiling path。
  - 同一 warmup/repeat 设置。
  - 同一 L2 flush 与 CUDA graph 策略。
- **Unified hardware-aware metrics**
  - FLOP 与 HBM byte 公式由 config 声明。
  - 所有 backend 用同一公式计算 roofline utilization。
- **Post-hoc diagnosis**
  - Benchmark 不止给出谁快。
  - 通过 NCU、PTX、SASS 解释为什么快或慢。

---

**典型数据结果**

| 指标 | Triton | cuTile |
|---|---:|---:|
| Geometric-mean speedup over PyTorch | **2.7×** | **2.2×** |
| Median speedup over PyTorch | **3.1×** | **2.7×** |
| Faster than PyTorch 的 operator 数量 | **36/45** | **34/45** |
| cuTile 快于 Triton 的 operator 数量 | \- | **11/45** |
| Autotune roofline gain | **1.20×** | **1.15×** |
| Autotuned 后达到 ≥80% roofline 的 operator 数量 | **8/45** | **4/45** |

- 数据说明：
  - 两个 backend 都显著优于 PyTorch。
  - Triton 在整体上更稳健。
  - cuTile 的优势集中在 Tensor Core/TMA-friendly workload。
  - Autotuning 有收益，但大多数 kernel 仍离 roofline 上限较远。
  - 剩余性能差距更多来自 compiler lowering、memory staging、instruction selection、shared-memory layout，而非单纯 tile size。

![](c46cba47f4d618a2e72eb29c059a5c2144d83ca4d63ed90b4f4c0c5a37a4b113.jpg)

---

**为什么需要 45 个 Operator**

- 少数 operator 无法代表 tile-based DSL 的真实适用范围。
- Dense GEMM/Attention 往往偏向规则 tile reuse，容易让 cuTile 的静态 tile abstraction 受益。
- Sparse、indexing、low-bit packing、boundary-heavy operator 更能暴露：
  - pointer arithmetic 能力。
  - mask 表达能力。
  - gather/scatter 成本。
  - local/shared-memory staging 成本。
  - compiler lowering 限制。
- 45 个 operator 的设计使 benchmark 能同时观察：
  - backend 的最佳表现。
  - backend 的失败模式。
  - operator structure 对 performance ceiling 的影响。
  - tuning 与 compiler codegen 的相对重要性。

---

**Triton 与 cuTile 在 Suite 中的可比性处理**

- Triton 与 cuTile 并不是完全同构的 DSL，因此 TileBench 使用“可比”而非“完全相同”的实现策略。
- 可比性来自：
  - 同一 operator 语义。
  - 同一 input tensor。
  - 同一 dtype sweep。
  - 同一 correctness checker。
  - 同一 timing harness。
  - 同一 FLOP/byte 公式。
  - 类似算法结构。
  - 类似 tuning budget。
- 不强制相同的原因：
  - Triton 的 `tl.load` + pointer tile 与 cuTile 的 `ct.load` tile-space access 表达能力不同。
  - Triton 的 `num_warps/num_stages` 与 cuTile 的 `occupancy` 不存在严格一一映射。
  - cuTile tile shape 要求 power-of-two，而 Triton 可通过 mask 灵活处理非整除。
  - TMA、tcgen05、TMEM 等 Blackwell-native path 在两个 backend 中暴露程度不同。

---

**Benchmark Suite 的诊断价值**

- TileBench 不仅报告平均性能，还能定位性能差异的底层原因。
- 典型诊断路径：
  - 找出 Triton/cuTile latency gap 最大的 operator-dtype-case。
  - 使用 NCU 采集 stall、memory、shared-memory、instruction metrics。
  - 检查 PTX/SASS 是否使用：
    - **TMA**
    - **Tensor Core**
    - **tcgen05**
    - **TMEM**
    - **cp.async**
    - **ldmatrix**
    - **LDS/STL**
  - 将 stall source 回溯到 source-level DSL 语句。
- 示例结论：
  - cuTile 在规则 tile reuse 上容易映射到 Blackwell-native path。
  - Triton 在 irregular pointer/mask load 上更自然。
  - cuTile 的 `ct.gather`、`ct.where` 对某些动态索引场景可能引入更重的 producer chain。
  - Triton tensor descriptor/TMA 不是所有 workload 的通用优化，streaming kernel 可能因额外 staging 反而退化。

![](f9e643fecb6869d7899b6b7a50bbcaf63d4acc756e36ecba362971843a1ab893.jpg)

---

**与传统 Benchmark 的差异**

| 维度 | 普通 GPU Kernel Benchmark | TileBench Controlled Suite |
|---|---|---|
| 关注点 | 单个 kernel 或少量 workload 的性能 | 编程模型层面的系统比较 |
| 语义控制 | 可能来自不同实现 | PyTorch reference 统一定义 |
| backend 对比 | 可能实现结构不一致 | Triton/cuTile matched implementation |
| 输入规模 | 常固定或不统一 | 每 dtype 20 case sweep |
| 指标 | latency 或 speedup 为主 | latency、TFLOPS、bandwidth、roofline |
| 调优 | 不一定公平 | default 与 comparable autotuning |
| 诊断 | 通常缺少 | NCU/PTX/SASS guided diagnosis |
| LLM 生成评估 | 可选 | 作为独立 track 复用同一 suite |

---

**局限性**

- 当前 suite 只在 **NVIDIA B200** 上评估。
  - 结论对 Hopper、Ada、Ampere 等架构不一定直接成立。
- 当前比较只覆盖 **Triton** 与 **cuTile**。
  - 未纳入 TileLang、ThunderKittens、Tilus、NKI 等其他系统。
- Roofline 使用 analytical FLOP/HBM-byte formula。
  - 优点是标准化、可复现。
  - 缺点是不能完全替代 NCU counter-based analysis。
- Matched implementation 无法完全消除 backend 风格差异。
  - 某些 operator 天然更适合 pointer-based Triton。
  - 某些 operator 天然更适合 static tile cuTile。
- Autotuning 不搜索算法变体。
  - 它只能探索 exposed tuning knobs。
  - 无法自动修复 compiler lowering 或 memory layout 问题。

---

**关键结论**

- **Controlled Tile-Based Benchmark Suite**的本质是一个受控实验平台，用于回答“tile-based programming model 在真实 AI operator 上表现如何、为什么表现如此”。
- 其技术核心是：
  - **45 个 operator 的广覆盖**
  - **PyTorch reference 的语义锚定**
  - **Triton/cuTile matched implementations**
  - **标准化 dtype 与 input-size sweeps**
  - **统一 timing 与 roofline metrics**
  - **default/autotuned 双路径**
  - **profiling-guided bottleneck diagnosis**
- 该设计使 TileBench 能把性能差异拆解为：
  - operator structure 差异。
  - tile abstraction 表达能力差异。
  - compiler lowering 差异。
  - Tensor Core/TMA 使用差异。
  - memory staging 与 shared-memory layout 差异。
  - autotuning parameter search 的收益与边界。

### 2. Unified Evaluation Harness

**Unified Evaluation Harness的核心定位**

- **Unified Evaluation Harness**是TileBench保证**公平比较**与**可复现测量**的执行层。
- 它将每个operator的**PyTorch reference**、**Triton implementation**、**cuTile implementation**放入同一套流程中运行，避免不同backend因为计时方式、输入生成、正确性检查或缓存状态不同而产生偏差。
- 它在TileBench整体pipeline中承担三个关键职责：
  - **语义对齐**：所有backend输出都与**PyTorch reference**比较。
  - **测量标准化**：所有backend使用一致的warmup、timing、CUDA graph、L2-cache flush策略。
  - **指标归一化**：所有结果通过同一套config.yaml中的**FLOP公式**、**HBM-byte公式**和**dtype信息**计算speedup、throughput、bandwidth、roofline utilization。

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

---

**输入与输出关系**

- Harness的输入来自每个operator目录中的标准化artifact：
  - **impl_torch.py**
    - 定义operator的语义参考实现。
    - 作为correctness oracle。
    - 同时作为PyTorch baseline latency来源。
  - **impl_triton.py**
    - 人工实现或LLM生成的Triton kernel。
    - 必须暴露统一接口：`run(*args, **kwargs)`和`get_last_config()`。
  - **impl_cutile.py**
    - 人工实现或LLM生成的cuTile kernel。
    - 同样必须暴露`run(*args, **kwargs)`和`get_last_config()`。
  - **config.yaml**
    - 声明dtype列表、case grid、验证容差、timing参数、FLOP公式、byte公式。
  - **B200.json**
    - 提供NVIDIA B200的硬件peak信息，包括**peak bandwidth**和不同dtype的**peak TFLOPS**。

| 输入组件 | 作用 | 对公平性的贡献 |
|---|---:|---:|
| **impl_torch.py** | 语义参考、baseline latency | 所有backend共享同一correctness oracle |
| **impl_triton.py** | Triton kernel实现 | 与cuTile接受相同输入case和dtype |
| **impl_cutile.py** | cuTile kernel实现 | 与Triton接受相同验证和计时协议 |
| **config.yaml** | case grid、dtype、容差、FLOP/byte公式 | 统一输入规模、指标计算和验证标准 |
| **B200.json** | peak TFLOPS、peak HBM bandwidth | 统一roofline bound |
| **data/tensors.py** | 生成输入tensor | 确保每个backend面对相同输入分布 |

- Harness的输出包括：
  - **correctness结果**
    - 是否compile成功。
    - 是否通过PyTorch reference验证。
    - 若失败，返回失败case和误差信息。
  - **latency**
    - 以mean kernel latency为主。
    - 通过20次warmup和100次timed runs统计。
  - **speedup over PyTorch**
    - 定义为：`PyTorch latency / backend latency`。
  - **achieved throughput**
    - 基于config.yaml中的FLOP公式和实测latency计算。
  - **effective bandwidth**
    - 基于config.yaml中的HBM-byte公式和实测latency计算。
  - **arithmetic intensity**
    - 定义为：`FLOPs / HBM bytes`。
  - **roofline utilization**
    - 以B200 dtype-specific peak和measured peak bandwidth归一化。
  - **backend configuration**
    - 由`get_last_config()`返回。
    - 用于autotuning分析和LLM iterative refinement反馈。

---

**执行流程**

- Harness对每个operator、每个dtype、每个input case执行相同流程：
  - 生成输入tensor。
  - 调用**PyTorch reference**得到reference output。
  - 测量PyTorch baseline latency。
  - 编译并运行Triton implementation。
  - 将Triton输出与PyTorch reference output比较。
  - 若Triton验证通过，执行标准timing。
  - 编译并运行cuTile implementation。
  - 将cuTile输出与PyTorch reference output比较。
  - 若cuTile验证通过，执行标准timing。
  - 记录backend配置、latency和派生指标。
  - 聚合case级、dtype级、operator级结果。

- 典型执行逻辑可抽象为：
  - 对operator `o`：
    - 对dtype `d`：
      - 对case `c`：
        - `inputs = generate_inputs(o, d, c)`
        - `ref = impl_torch.run(*inputs)`
        - `T_torch = time(impl_torch.run)`
        - 对backend `b ∈ {Triton, cuTile}`：
          - `out = impl_b.run(*inputs)`
          - `verify(out, ref)`
          - 若verify成功：
            - `T_b = time_with_cuda_graph_and_l2_flush(impl_b.run)`
            - `cfg_b = impl_b.get_last_config()`
            - 计算`speedup`、`TFLOPS`、`bandwidth`、`roofline utilization`

---

**正确性检查机制**

- Harness以**PyTorch reference**作为唯一语义标准。
- 对浮点输出使用：
  - **torch.testing.assert_close**
  - 容差来自operator-specific config。
  - 支持dtype-specific tolerance。
- 对整数输出使用：
  - **exact comparison**
  - `atol = 0`
  - `rtol = 0`
- 该机制保证：
  - Triton和cuTile不需要彼此对齐，只需要同时对齐PyTorch reference。
  - backend之间的性能比较不会被错误实现污染。
  - 非结合性reduction带来的浮点误差可以通过operator-specific tolerance处理。

| 输出类型 | 检查方式 | 容差策略 |
|---|---:|---:|
| **floating-point tensor** | `torch.testing.assert_close` | dtype/operator-specific `atol`、`rtol` |
| **integer tensor** | exact comparison | `atol = rtol = 0` |
| **tuple outputs** | 逐项比较 | 继承对应dtype策略 |
| **compile/runtime failure** | 标记失败 | 不进入timing |

---

**计时协议**

- Harness使用统一timing协议测量所有backend。
- 关键参数包括：
  - **CUDA graph capture-and-replay**
  - **L2-cache flushing**
  - **20 warmup runs**
  - **100 timed runs**
  - **mean kernel latency**
  - **Proton profiler**
- 这些设置的目的：
  - 降低Python launch overhead对kernel latency的影响。
  - 减少偶然冷启动、JIT编译、cache残留造成的波动。
  - 保证PyTorch、Triton、cuTile在相同采样条件下比较。

| 参数 | TileBench设置 | 作用 |
|---|---:|---:|
| **warmup** | 20 runs | 排除冷启动、JIT、初始cache效应 |
| **repeat** | 100 timed runs | 获取稳定mean latency |
| **CUDA graph** | enabled | 降低launch overhead和host-side噪声 |
| **L2-cache flushing** | enabled | 减少跨run cache reuse偏差 |
| **profiler** | Proton | 统一采集kernel timing |
| **post-hoc profiling** | Nsight Compute | 只用于选定case的深入诊断 |

---

**CUDA graph capture-and-replay的作用**

- **CUDA graph capture-and-replay**用于稳定计时过程。
- 对GPU kernel benchmarking而言，普通launch会包含host侧调度开销、Python runtime开销和CUDA API开销。
- CUDA graph将一段固定执行路径capture成graph，再重复replay：
  - 减少CPU launch jitter。
  - 提高timing repeatability。
  - 让测量更接近kernel本身开销。
- 在TileBench中，这对轻量级operator尤其重要：
  - pointwise operator
  - reduction operator
  - data-layout operator
  - small tensor cases
- 若不使用CUDA graph，轻量kernel的真实GPU执行时间可能被host overhead掩盖，导致backend比较失真。

---

**L2-cache flushing的作用**

- **L2-cache flushing**用于降低缓存命中残留对测量的影响。
- GPU benchmark中，如果连续运行相同输入：
  - 后续run可能复用L2 cache中的数据。
  - bandwidth-bound kernel可能得到不真实的高性能。
  - 不同backend因为访问模式不同，cache收益可能不同。
- TileBench通过flush L2使每次timed run更接近统一初始状态。
- 该策略尤其影响：
  - **streaming memory-bound kernels**
  - **matrix-copy**
  - **vector-addition**
  - **weight-dequant**
  - **matrix-transpose**
  - **layout/indexing kernels**
- 该设计使effective bandwidth和roofline utilization更可信。

---

**Warmup与Timed Runs**

- **20 warmup runs**用于稳定执行状态：
  - 避免首次kernel launch开销。
  - 避免lazy initialization影响。
  - 避免JIT compilation或runtime setup进入计时。
  - 让GPU进入相对稳定的运行状态。
- **100 timed runs**用于统计mean latency：
  - 减少偶然调度波动。
  - 平滑硬件噪声。
  - 对短kernel提供更稳定的均值。
- TileBench报告的是**mean kernel latency**，而不是best latency：
  - mean更适合描述真实可复现表现。
  - best latency容易被偶然cache状态或系统噪声影响。

---

**config.yaml驱动的统一参数体系**

- 每个operator通过config.yaml声明benchmark行为。
- Harness不会硬编码operator-specific规则，而是读取config.yaml进行标准化执行。
- config.yaml中的核心字段包括：
  - **dtype list**
  - **case grid**
  - **verification tolerance**
  - **warmup/repeat**
  - **CUDA graph开关**
  - **L2 flush开关**
  - **autotune开关**
  - **flops_expr**
  - **bytes_expr**
  - **plots/metrics**

```yaml
benchmark:
  warmup: 20
  repeat: 100
  use_cuda_graph: true
  flush_l2: true
  autotune: false
case_defaults:
  M: 2048
case_grid:
  N: [20480]
  dtype: ["fp16", "fp32"]
metrics:
  flops_expr: "M * N"
  bytes_expr: "M * N * dtype_size"
plots:
  - latency_ms
  - bandwidth_GBs
  - speedup
```

- 该设计使不同operator可以拥有不同shape sweep，同时保持执行协议一致。
- 例如：
  - matmul类operator可选择触发Tensor Core路径的规模。
  - reduction类operator可选择覆盖occupancy变化明显的reduce dimension。
  - pointwise类operator可覆盖从small到large的memory-bound区间。

---

**指标计算方式**

- Harness使用统一公式计算性能指标。
- 对operator `o`、backend `b`、dtype `d`、case `c`，记录mean latency：
  - **Tₒ,b,d,c**
- **Speedup over PyTorch**：
  - `T_o,torch,d,c / T_o,b,d,c`
- **Achieved TFLOPS**：
  - `F_o,d,c / T_o,b,d,c × 10^-12`
- **Arithmetic Intensity**：
  - `F_o,d,c / B_o,d,c`
- **Roofline Utilization**：
  - `achieved TFLOPS / min(dtype_peak_TFLOPS, arithmetic_intensity × peak_bandwidth)`

| 指标 | 输入来源 | 含义 |
|---|---:|---:|
| **latency** | Proton timing | 实际kernel平均耗时 |
| **speedup** | PyTorch latency与backend latency | 软件baseline加速比 |
| **TFLOPS** | FLOP公式与latency | 计算吞吐 |
| **effective bandwidth** | HBM-byte公式与latency | 有效内存带宽 |
| **arithmetic intensity** | FLOP/byte | 判断compute-bound或memory-bound倾向 |
| **roofline utilization** | B200 peak与实测性能 | 相对硬件上限的利用率 |

---

**在主benchmark track中的作用**

- 在human-written implementation评估中，Unified Evaluation Harness用于回答：
  - Triton和cuTile相比PyTorch是否加速。
  - Triton和cuTile彼此谁更快。
  - default configuration与autotuned configuration差距多大。
  - 哪些operator仍有roofline headroom。
- 由于所有operator都经过同一流程：
  - Figure 2中的speedup scatter具有可比性。
  - Figure 3中的default-to-autotuned roofline utilization具有可比性。
  - Figure 4中的top gap analysis可以基于同一latency定义排序。

![](c46cba47f4d618a2e72eb29c059a5c2144d83ca4d63ed90b4f4c0c5a37a4b113.jpg)

![](95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg)

---

**在autotuning评估中的作用**

- Harness支持两种执行路径：
  - **default path**
    - 使用人工选择的固定配置。
  - **autotuned path**
    - 在人工定义的candidate configuration集合中搜索。
- 对Triton和cuTile，autotuning knobs不同：
  - Triton：
    - **BLOCK_SIZE**
    - **num_warps**
    - **num_stages**
  - cuTile：
    - **TILE_SIZE**
    - **occupancy**
- Harness的作用不是强行建立一一对应参数，而是保证：
  - search budget相近。
  - case grid相同。
  - timing协议相同。
  - 指标计算相同。
- 这使论文能够得出：
  - Triton autotuning几何平均提升约**1.20×**。
  - cuTile autotuning几何平均提升约**1.15×**。
  - 但大多数operator仍有明显roofline headroom。

---

**在LLM code-generation track中的作用**

- Harness也用于评估LLM生成的Triton和cuTile代码。
- 与主benchmark track不同：
  - LLM-generated code不使用backend autotuning。
  - 每轮iteration必须提交单一固定配置。
  - 每个backend独立追踪10轮refinement。
- Harness向下一轮prompt返回结构化反馈：
  - compile status
  - verification status
  - latency
  - speedup over PyTorch
  - roofline utilization
  - worst-first performance table
  - `get_last_config()`返回的配置
- Harness还负责拒绝reward hacking：
  - 禁止调用PyTorch、cuBLAS、cuDNN替代真实kernel。
  - 禁止缓存输出。
  - 禁止调用backend autotuner。
  - 使用static scan、two-pass anti-cache verification和rotating-input timing检测作弊。

---

**为什么Unified Evaluation Harness是必要的**

- Tile-based DSL性能差异往往很小或高度case-dependent。
- 若没有统一harness，以下因素会污染结论：
  - 不同输入规模。
  - 不同dtype覆盖。
  - 不同warmup策略。
  - 不同cache状态。
  - 不同timing API。
  - 不同correctness tolerance。
  - 不同PyTorch baseline。
  - 不同FLOP/byte计算口径。
- Unified Evaluation Harness将这些变量固定，使观察到的差异更可能来自：
  - Triton与cuTile的programming model差异。
  - compiler lowering差异。
  - memory staging差异。
  - Tensor Core/TMA使用差异。
  - shared memory bank conflict差异。

---

**关键设计取舍**

- **PyTorch作为reference而非性能目标**
  - PyTorch定义语义和baseline。
  - 但TileBench关注Triton/cuTile之间的programming-model比较。
- **mean latency而非minimum latency**
  - 更稳定。
  - 更接近实际重复执行表现。
- **analytical FLOP/HBM-byte公式**
  - 保证跨backend一致。
  - 但不是NCU counter的替代品。
- **Proton用于常规测量，NCU用于诊断**
  - Proton适合大规模自动benchmark。
  - NCU开销更高，保留给large-gap case和post-hoc analysis。
- **CUDA graph与L2 flush同时使用**
  - CUDA graph降低host-side噪声。
  - L2 flush降低cache-state偏差。
  - 两者共同提高latency和bandwidth测量可信度。

---

**整体作用总结**

- **Unified Evaluation Harness**是TileBench从“多个kernel集合”变成“controlled benchmark”的核心机制。
- 它把operator语义、输入生成、正确性验证、计时协议、配置记录和指标计算统一起来。
- 它使论文中的核心结论具备可解释性：
  - Triton整体更稳健。
  - cuTile在Tensor Core/TMA-friendly workload上更有优势。
  - autotuning有收益但不能解决所有roofline headroom。
  - LLM生成Triton代码比生成cuTile代码更token-efficient。
- 没有该harness，TileBench无法区分性能差距究竟来自**programming model本身**，还是来自**benchmark流程不一致**。

### 3. Config-Driven Metrics and Roofline Analysis

**核心定位**

- **Config-Driven Metrics and Roofline Analysis**是 TileBench 的度量基础设施设计：
  - 每个 operator 都通过独立的 **config.yaml** 声明评测所需的元信息。
  - Benchmark harness 不依赖硬编码规则，而是从 config 中读取：
    - **dtype list**
    - **case grid**
    - **verification tolerances**
    - **FLOP analytical formula**
    - **HBM-byte analytical formula**
    - **timing options**
  - 这些配置被统一用于 PyTorch、Triton、cuTile 三类实现，使不同 backend 的性能指标具有可比性。
- 该机制的核心价值：
  - 将**算子语义**、**输入规模**、**正确性标准**、**硬件感知指标公式**从 kernel 代码中解耦。
  - 保证每个 operator 在相同 dtype、相同 shape、相同 FLOP/byte 定义下比较。
  - 支撑 TileBench 的主要分析指标：
    - **latency**
    - **speedup over PyTorch**
    - **achieved throughput**
    - **effective bandwidth**
    - **arithmetic intensity**
    - **roofline utilization**

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

---

**config.yaml 的作用边界**

- **config.yaml**不是简单的运行参数文件，而是每个 operator 的完整评测契约。
- 它定义了 benchmark harness 如何生成输入、如何验证输出、如何计时、如何计算硬件利用率。
- 每个 operator 独立配置，而不是全局共享配置，原因包括：
  - 不同 operator 的 shape sensitivity 不同：
    - Matmul 需要覆盖 Tensor Core 路径能够触发的规模。
    - Reduction/Norm 需要覆盖 occupancy 饱和前后的规模。
    - Point-wise 需要覆盖从小规模 launch overhead 到大规模 HBM bandwidth-bound 的区间。
    - Attention 需要覆盖 LLM-like sequence length。
  - 不同 operator 的数值误差特性不同：
    - Elementwise 通常容差较小。
    - Reduction 类算子由于 floating-point reassociation，可能需要更宽容差。
    - Low-bit quantization/dequantization 可能涉及 dtype-specific tolerance。
  - 不同 operator 的 FLOP 与 HBM byte 计算方式不同：
    - Matmul 的 FLOP 通常与 M、N、K 成比例。
    - Point-wise 的 FLOP 通常与元素数成比例。
    - Data Layout 可能 FLOP 很低，但 byte movement 很高。
    - Sorting、Top-K、Attention 的公式更依赖算法语义和近似建模。

---

**核心配置字段**

| 配置字段 | 作用 | 对应输出指标或流程 |
|---|---|---|
| **dtype list** | 声明 operator 支持的 dtype，如 fp16、bf16、fp32、int8、fp8 | 决定输入生成、峰值算力选择、验证容差 |
| **case grid** | 声明输入规模扫描参数 | 生成多组 benchmark cases |
| **verification tolerances** | 声明正确性检查中的 atol、rtol | 控制 Triton/cuTile 与 PyTorch reference 的一致性判定 |
| **benchmark options** | warmup、repeat、CUDA graph、L2 flush 等 | 控制计时协议 |
| **flops_expr** | 分析性 FLOP 公式 | 计算 TFLOPS、Roofline utilization |
| **bytes_expr** | 分析性 HBM byte 公式 | 计算 effective bandwidth、arithmetic intensity |
| **plots/metrics** | 声明需要导出的指标 | 统一报告 latency、bandwidth、speedup 等 |

---

**配置示例解析**

- 文档中的 argmax 配置示例展示了 config-driven 机制的最小结构：

```yaml
benchmark:
  warmup: 20
  repeat: 100
  use_cuda_graph: true
  flush_l2: true
  autotune: false
case_defaults:
  M: 2048
case_grid:
  N: [20480]
  dtype: ["fp16", "fp32"]
metrics:
  flops_expr: "M * N"
  bytes_expr: "M * N * dtype_size"
plots:
  - latency_ms
  - bandwidth_GBs
  - speedup
```

- 该配置表达的语义：
  - **benchmark.warmup: 20**
    - 每个 case 正式计时前执行 20 次 warmup。
    - 用于消除首次 kernel launch、cache 状态、JIT compilation 后效应。
  - **benchmark.repeat: 100**
    - 每个 case 采集 100 次 timed runs。
    - 报告 mean kernel latency。
  - **use_cuda_graph: true**
    - 使用 CUDA graph capture-and-replay。
    - 降低 Python launch overhead 对短 kernel 的干扰。
  - **flush_l2: true**
    - 每轮计时前刷新 L2 cache。
    - 避免缓存命中导致 bandwidth-bound kernel 被高估。
  - **autotune: false**
    - 在 LLM code-generation track 中禁用 backend autotuner。
    - 在主 benchmark track 中则存在 default path 与 autotuned path。
  - **case_defaults.M: 2048**
    - 为未在 case_grid 中枚举的参数提供默认值。
  - **case_grid.N: [20480]**
    - 对 N 维度指定输入规模。
  - **case_grid.dtype: ["fp16", "fp32"]**
    - 同一 operator 在 fp16 和 fp32 下分别评测。
  - **flops_expr: "M * N"**
    - 对每个 case 计算理论操作数。
  - **bytes_expr: "M * N * dtype_size"**
    - 对每个 case 计算理论 HBM 访问字节数。
  - **plots**
    - 指定输出图表或汇总字段。

---

**输入到输出的关系**

- 输入侧：
  - **config.yaml**提供 operator-level 配置。
  - **B200.json**提供 hardware-level 配置。
  - **impl_torch.py**提供语义 reference。
  - **impl_triton.py**和**impl_cutile.py**提供待测 kernel。
  - Benchmark harness 生成 `(operator, backend, dtype, case)` 组合。
- 输出侧：
  - 对每个组合记录 mean kernel latency：
    - `T_o,b,d,c`
  - 计算 PyTorch baseline speedup：
    - `T_o,torch,d,c / T_o,b,d,c`
  - 根据 `flops_expr` 计算 FLOP：
    - `F_o,d,c`
  - 根据 `bytes_expr` 计算 HBM byte：
    - `B_o,d,c`
  - 进一步计算：
    - **TFLOPS**
    - **Effective bandwidth**
    - **Arithmetic intensity**
    - **Roofline utilization**

| 输入来源 | 关键内容 | 被用于 |
|---|---|---|
| **config.yaml** | dtype、shape grid、FLOP/byte 公式、容差 | case 生成、正确性验证、指标计算 |
| **B200.json** | peak bandwidth、dtype peak TFLOPS | Roofline 上界计算 |
| **impl_torch.py** | PyTorch reference output 与 latency | 正确性 oracle、speedup baseline |
| **impl_triton.py** | Triton kernel latency | backend 性能评估 |
| **impl_cutile.py** | cuTile kernel latency | backend 性能评估 |
| **Proton profiler** | mean latency | 性能统计 |
| **Nsight Compute** | selected case 的硬件 counter | 诊断分析，不替代标准化 roofline 指标 |

---

**B200.json 的硬件参数作用**

- **B200.json**将硬件峰值能力显式配置化。
- TileBench 使用这些常数计算 dtype-specific roofline bound。
- 文档中的 B200 示例包含：
  - GPU 型号：**NVIDIA B200**
  - measured peak bandwidth：**6539.4 GB/s**
  - dtype peak throughput：
    - fp64: 37 TFLOPS
    - fp32: 1100 TFLOPS
    - tf32: 1100 TFLOPS
    - fp16: 2250 TFLOPS
    - bf16: 2250 TFLOPS
    - fp8: 4500 TFLOPS
    - int8: 4500 TFLOPS

| dtype | Peak TFLOPS | 说明 |
|---|---:|---|
| **fp64** | 37 | 双精度上界 |
| **fp32** | 1100 | 文中按 TF32 Tensor Core dense ceiling 处理 FP32 dot/MMA |
| **tf32** | 1100 | Triton `tl.dot` 对 FP32 默认使用 TF32 precision |
| **fp16** | 2250 | Tensor Core 主力 dtype |
| **bf16** | 2250 | LLM 常用 dtype |
| **fp8_e4m3fn** | 4500 | Blackwell 上高吞吐低精度 |
| **fp8_e5m2** | 4500 | Blackwell 上高吞吐低精度 |
| **int8** | 4500 | 量化 GEMM 相关上界 |

- 关键设计点：
  - **peak bandwidth**不是直接采用 datasheet，而是在 B200 系统上测得，为 **6539.4 GB/s**。
  - **FP32 dot/MMA**使用 TF32 Tensor Core dense ceiling，因为 Triton `tl.dot` 默认会将 FP32 input precision 走 TF32 路径。
  - 不包含 2:4 sparsity 加速上界，避免高估 dense kernel 的可达性能。

![](95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg)

---

**算法流程**

- **Step 1：读取 operator 配置**
  - Benchmark harness 读取当前 operator 的 **config.yaml**。
  - 获得：
    - dtype list
    - case grid
    - case defaults
    - tolerance
    - benchmark timing options
    - analytical formulas
- **Step 2：展开 case grid**
  - 将 `case_defaults` 与 `case_grid` 做组合展开。
  - 对每个 dtype 生成多个 input cases。
  - TileBench 主评测中，每个 operator 对每个 supported dtype 评测 **20 input configurations**。
- **Step 3：生成输入张量**
  - 根据 operator-specific generator 生成输入。
  - 输入 shape 由 config 中的参数决定。
  - dtype 由 config 中的 dtype list 决定。
- **Step 4：运行 PyTorch reference**
  - 调用 `impl_torch.run(*inputs)`。
  - 产生 reference output。
  - 记录 PyTorch latency：
    - `T_o,torch,d,c`
- **Step 5：运行 Triton 与 cuTile**
  - 调用 `impl_triton.run(*inputs)`。
  - 调用 `impl_cutile.run(*inputs)`。
  - 与 PyTorch output 进行 correctness check。
  - 通过验证后采集 latency：
    - `T_o,triton,d,c`
    - `T_o,cutile,d,c`
- **Step 6：计算 analytical FLOPs 和 bytes**
  - 将当前 case 参数代入 `flops_expr`。
  - 将当前 case 参数、dtype_size 代入 `bytes_expr`。
  - 得到：
    - `F_o,d,c`
    - `B_o,d,c`
- **Step 7：计算派生指标**
  - 计算 speedup。
  - 计算 achieved TFLOPS。
  - 计算 effective bandwidth。
  - 计算 arithmetic intensity。
  - 计算 roofline utilization。
- **Step 8：聚合**
  - case 内求 arithmetic mean。
  - operator 间通常用 geometric mean。
  - 输出跨 operator、跨 category、跨 backend 的对比结果。

---

**指标计算细节**

- **Mean kernel latency**
  - 对 operator `o`、backend `b`、dtype `d`、case `c`：
    - 记为 `T_o,b,d,c`
  - 单位通常为 ms 或 s。
  - 是后续所有性能指标的基础。
- **Speedup over PyTorch**
  - 定义：
    - `Speedup = T_o,torch,d,c / T_o,b,d,c`
  - 含义：
    - 大于 1 表示 backend 快于 PyTorch。
    - 小于 1 表示 backend 慢于 PyTorch。
- **Achieved throughput**
  - 定义：
    - `TFLOPS_o,b,d,c = F_o,d,c / T_o,b,d,c * 10^-12`
  - 含义：
    - kernel 实际达到的计算吞吐。
    - 对 compute-bound kernel 尤其重要。
- **Effective bandwidth**
  - 通常可由：
    - `Bandwidth = B_o,d,c / T_o,b,d,c`
  - 含义：
    - kernel 按 analytical HBM byte 公式估算出的有效带宽。
    - 对 point-wise、copy、transpose、dequant、layout 类 kernel 尤其重要。
- **Arithmetic intensity**
  - 定义：
    - `AI_o,d,c = F_o,d,c / B_o,d,c`
  - 单位：
    - FLOP/byte
  - 含义：
    - 每访问 1 byte HBM 数据能产生多少 FLOP。
    - 决定 kernel 在 roofline 模型中更接近 memory-bound 还是 compute-bound。
- **Roofline bound**
  - 定义：
    - `Bound = min(P_d^peak, AI_o,d,c * BW^peak)`
  - 含义：
    - 如果 `AI * BW_peak` 小于 dtype peak compute，则 kernel 理论上 memory-bound。
    - 如果 dtype peak compute 更小，则 kernel 理论上 compute-bound。
- **Roofline utilization**
  - 定义：
    - `R_o,b,d,c = TFLOPS_o,b,d,c / Bound`
  - 含义：
    - 衡量 kernel 达到硬件理论上界的比例。
    - 比单纯 speedup 更能反映 hardware efficiency。

---

**Roofline Analysis 的解释逻辑**

- **Roofline Analysis**不是只看 kernel 快不快，而是判断 kernel 距离硬件可达上界有多远。
- TileBench 的 roofline 使用两个上界：
  - **compute peak**
    - dtype-specific peak TFLOPS。
  - **memory bandwidth peak**
    - `AI * peak_BW`。
- 二者取最小值：
  - 如果 arithmetic intensity 很低：
    - `AI * peak_BW`成为瓶颈。
    - kernel 属于 bandwidth-bound。
  - 如果 arithmetic intensity 很高：
    - dtype peak TFLOPS 成为瓶颈。
    - kernel 属于 compute-bound。
- **Roofline utilization R**越高，说明 kernel 越接近对应理论瓶颈：
  - `R >= 0.8`
    - 通常说明实现已经接近硬件上限。
  - `R < 0.2`
    - 通常说明存在明显 headroom。
    - 可能原因包括 compiler lowering、instruction selection、memory staging、bank conflict、register spilling、occupancy 不足等。

---

**参数设置与标准化策略**

| 参数类别 | TileBench 设置 | 技术目的 |
|---|---|---|
| **warmup** | 20 runs | 稳定 GPU 状态，避免首次执行扰动 |
| **timed repeat** | 100 runs | 获得稳定 mean latency |
| **CUDA graph** | enabled | 减少 launch overhead |
| **L2 flush** | enabled | 避免缓存复用虚高 |
| **input cases** | 每 dtype 20 个 configurations | 覆盖小到大规模变化 |
| **dtype** | operator-specific | 匹配 operator 语义与硬件路径 |
| **FLOP formula** | config-defined analytical expression | 统一 throughput 计算 |
| **HBM byte formula** | config-defined analytical expression | 统一 bandwidth 与 AI 计算 |
| **hardware peak** | B200.json | 统一 roofline bound |
| **aggregation** | case 内 arithmetic mean，operator 间 geometric mean | 减少 outlier 支配整体结果 |

---

**正确性验证与 metrics 的关系**

- TileBench 的所有性能指标都是**正确性门控**的。
- 对每个 backend：
  - 若 output 未通过 PyTorch reference 校验，则该 case 不进入有效性能统计。
  - 浮点输出使用 `torch.testing.assert_close`。
  - 整数输出使用 exact comparison：
    - `atol = 0`
    - `rtol = 0`
- 容差来自 config：
  - dtype-specific tolerance
  - operator-specific tolerance
- 该设计避免 backend 通过不等价计算获得虚假性能优势。
- 对 reduction 类 operator，容差配置尤其重要：
  - 不同 backend 可能采用不同 reduction order。
  - floating-point 非结合性会导致微小误差。
  - config 中的 tolerance 允许合理误差，但仍约束语义一致性。

---

**在整体 TileBench Pipeline 中的作用**

- 在 benchmark construction 阶段：
  - config.yaml 将来自 TritonBench 和 LeetGPU 的 heterogeneous task 规格统一成 TileBench 格式。
  - 每个 operator 的 shape sweep、dtype sweep、验证策略被标准化。
- 在 benchmark execution 阶段：
  - harness 根据 config 驱动 PyTorch、Triton、cuTile 的执行。
  - 所有 backend 使用同一输入 case 与同一 correctness oracle。
- 在 comparative analysis 阶段：
  - config 中的 FLOP/byte 公式提供硬件感知指标。
  - B200.json 提供硬件峰值上界。
  - 最终生成 speedup、throughput、bandwidth、roofline utilization 等跨 backend 可比结果。
- 在 profiling-guided diagnosis 阶段：
  - Roofline utilization 用来筛选低效 kernel 或大差距 case。
  - 对 selected cases 再使用 Nsight Compute 和 PTX/SASS 进行深入分析。
  - 因此 config-driven roofline 是诊断入口，而 NCU 是后续定位工具。

---

**为什么使用 analytical FLOP/HBM-byte formulas**

- 优点：
  - **统一性**
    - 所有 backend 使用完全相同的 FLOP 和 byte 定义。
    - 避免不同 compiler 或 profiler counter 对同一操作解释不同。
  - **可复现性**
    - 指标由 config 和 case 参数确定。
    - 不依赖不稳定的 runtime counter。
  - **跨 operator 可比较**
    - 可以把 point-wise、reduction、matmul、attention、layout 放到统一 roofline 框架。
  - **轻量**
    - 不需要对每个 case 都运行 Nsight Compute。
    - 适合 45 operators、多 dtype、多 shape sweep 的大规模评测。
- 局限：
  - **不等价于真实硬件访问量**
    - compiler 可能引入额外 shared-memory staging、local memory spill、重复 global load。
    - analytical bytes 只表示语义或理想算法层面的 HBM traffic。
  - **不能直接反映 bank conflict**
    - bank conflict、shared-memory replay、uncoalesced access 需要 NCU counter。
  - **不能揭示 instruction mix**
    - 是否使用 TMA、tcgen05、ldmatrix、cp.async、IMMA 需要 SASS/PTX inspection。
  - **复杂 operator 的 FLOP 建模可能近似**
    - Sorting、Top-K、sparse attention、flash decode 的实际 work 可能与输入分布或控制流有关。

---

**与 autotuning 的关系**

- config-driven metrics 也用于评估 autotuning 前后的变化。
- 对每个 operator：
  - default path 使用手动选择的固定配置。
  - autotuned path 搜索 backend-specific tuning space。
- 评估中使用相同的 roofline utilization：
  - `R_default`
  - `R_autotune`
  - `R_autotune / R_default`
- 文中报告：
  - Triton 的 geometric-mean autotune gain 为 **1.20×**。
  - cuTile 的 geometric-mean autotune gain 为 **1.15×**。
  - autotuned 后：
    - Triton 仅 **8/45** 个实现达到至少 **80% roofline utilization**。
    - cuTile 仅 **4/45** 个实现达到至少 **80% roofline utilization**。
- 结论：
  - autotuning 可以改善 tile shape、num_warps、num_stages、occupancy 等参数。
  - 但大量 headroom 来自 backend lowering、memory behavior、instruction selection，而非单纯参数搜索。

| Backend | Geomean autotune gain | ≥80% roofline utilization 的 operator 数 | 解释 |
|---|---:|---:|---|
| **Triton** | **1.20×** | **8/45** | 参数搜索有收益，但仍受 compiler lowering 与 memory staging 限制 |
| **cuTile** | **1.15×** | **4/45** | 对 TMA/TC-friendly workload 有优势，但不规则场景 headroom 更大 |

---

**指标如何支撑 RQ1 到 RQ4**

- **RQ1：Programming-model Performance**
  - 使用 speedup over PyTorch 比较 Triton 与 cuTile 的总体性能。
  - 使用 per-operator speedup scatter 判断 backend 优势分布。
  - 结果：
    - Triton geomean speedup：**2.7×**
    - cuTile geomean speedup：**2.2×**
    - Triton beats PyTorch：**36/45**
    - cuTile beats PyTorch：**34/45**
- **RQ2：Autotuning and Performance Headroom**
  - 使用 roofline utilization 衡量 default 与 autotuned 配置距离硬件上界的差距。
  - 说明 tuning knobs 不能完全解决 backend lowering 问题。
- **RQ3：Code-level Performance Diagnosis**
  - 使用 roofline 和 latency gap 筛选异常 case。
  - 再通过 NCU、PTX/SASS 分析具体瓶颈：
    - TMA 是否使用
    - Tensor Core 是否触发
    - shared-memory bank conflict
    - register spilling
    - local/shared-memory staging
    - async pipeline wait
- **RQ4：LLM Usability and Token Efficiency**
  - LLM track 中仍使用 config 的 dtype、case、verification、roofline 信息作为 feedback。
  - token efficiency 依赖 correctness-gated speedup。
  - 错误实现不计性能收益，但 token cost 仍累计。

---

**典型指标解释示例**

- 对一个 fp16 matmul case：
  - config 声明：
    - `M`
    - `N`
    - `K`
    - `dtype = fp16`
    - `flops_expr = 2 * M * N * K`
    - `bytes_expr = (M*K + K*N + M*N) * dtype_size`
  - harness 运行 Triton kernel 得到 latency。
  - 计算 achieved TFLOPS。
  - 从 B200.json 读取 fp16 peak：
    - `P_fp16^peak = 2250 TFLOPS`
  - 读取 peak bandwidth：
    - `BW^peak = 6539.4 GB/s`
  - 计算：
    - `AI = FLOPs / bytes`
    - `roofline_bound = min(2250, AI * 6539.4)`
    - `R = achieved_TFLOPS / roofline_bound`
- 对一个 point-wise vector add case：
  - FLOP 很低，byte 很高。
  - arithmetic intensity 低。
  - roofline bound 通常由 memory bandwidth 决定。
  - effective bandwidth 与 roofline utilization 更有解释力，TFLOPS 本身意义较弱。
- 对一个 flash_attention case：
  - FLOP 量大，但实际瓶颈可能是 TMA latency、register pressure、softmax/reduction、shared-memory staging。
  - analytical roofline 可以显示 headroom。
  - 具体原因仍需 NCU 和 SASS 分析。

---

**设计优势**

- **公平性**
  - Triton 与 cuTile 不各自定义性能公式。
  - 所有 backend 使用相同 case、dtype、FLOP、byte、tolerance。
- **可扩展性**
  - 新增 operator 时，只需提供：
    - PyTorch reference
    - Triton implementation
    - cuTile implementation
    - config.yaml
  - harness 自动完成生成、验证、计时、指标计算。
- **诊断友好**
  - Roofline utilization 可快速识别：
    - 已接近硬件上限的 kernel
    - 明显存在 headroom 的 kernel
    - memory-bound 与 compute-bound 倾向
- **适合跨类别 benchmark**
  - TileBench 包含 point-wise、reduction、matmul/attention、stencil/conv、data layout。
  - config-driven metrics 让这些异构 operator 能被放在同一评测框架内。
- **适合 LLM code-generation**
  - LLM 每轮可获得结构化 feedback：
    - verify status
    - speedup
    - roofline utilization
    - worst-first performance summary
  - 这些 feedback 均来自同一 config-driven evaluation pipeline。

---

**关键限制**

- **analytical bytes 不一定等于真实 HBM traffic**
  - Compiler 可能产生额外 load/store。
  - Cache hit、reuse、prefetch、spill 不在公式中直接体现。
- **analytical FLOPs 不一定等于真实 instruction work**
  - Low-bit unpack、index calculation、masking、predicate branch 可能增加大量非 FLOP 指令。
  - 例如 matmul_int8 中，Triton 产生大量 PRMT 和 LDS 指令，这些不会反映在语义 FLOP 中。
- **Roofline utilization 是 summary metric，不是 root-cause**
  - 低 R 只能说明存在 headroom。
  - 不能单独判断原因是 memory coalescing、bank conflict、register pressure、occupancy、TMA latency 还是 Tensor Core 未触发。
- **复杂控制流 operator 的公式可能是近似**
  - Sparse、sort、top-k、decode 类 operator 的实际工作量可能依赖数据分布或边界条件。
- **单 GPU 硬件绑定**
  - 当前 TileBench 使用单张 NVIDIA B200。
  - B200.json 中的 peak constants 使指标对该平台有效，但跨 GPU 需要重新定义 hardware profile。

---

**实现原理总结**

- **config.yaml**将每个 operator 的评测语义参数化。
- Benchmark harness 将 config 展开为 `(dtype, case)` 组合。
- PyTorch reference 给出 correctness oracle 和 latency baseline。
- Triton/cuTile kernel 在相同输入上运行并通过相同容差验证。
- `flops_expr`与`bytes_expr`被代入 case 参数后生成标准化 analytical workload size。
- B200.json 提供 dtype peak compute 与 measured peak bandwidth。
- Roofline 模型通过 `min(compute peak, AI * bandwidth peak)`给出理论上界。
- **roofline utilization**衡量 kernel 距离该理论上界的比例。
- 该机制贯穿 TileBench 的主评测、autotuning 分析、profiling case 选择和 LLM feedback。

### 4. Comparable Default and Autotuned Execution Paths

**Comparable Default and Autotuned Execution Paths 的核心定位**

- **Comparable Default and Autotuned Execution Paths** 是 TileBench 用来公平比较 **Triton** 与 **cuTile** 的关键实验设计。
- 该机制解决的问题是：
  - **Triton** 与 **cuTile** 暴露的调优参数并不完全一致。
  - 直接要求两者使用相同参数组合并不合理，因为两种 DSL 的执行模型、编译器接口、tile 抽象和 launch hint 不同。
  - TileBench 因此采用 **backend-native autotuning**：
    - Triton 使用 Triton 原生的 tile/block 参数与调度 hint。
    - cuTile 使用 cuTile 原生的 tile shape 与 occupancy hint。
  - 同时保证两者具有 **comparable search budgets**：
    - 搜索空间规模相近。
    - 调优粒度相近。
    - 每个 backend 都在自身合理的参数范围内寻找较优配置。
- 该设计的核心目标不是让 Triton 和 cuTile 参数一一对应，而是让它们在各自自然表达方式下接受**公平、可复现、受控**的性能评估。

---

**默认路径与自动调优路径的区别**

- TileBench 为每个 Triton/cuTile 实现提供两条执行路径：
  - **Default execution path**
    - 使用人工选择的单一固定配置。
    - 代表开发者在理解算子结构后给出的合理初始版本。
    - 不是随机配置，也不是刻意弱化的 baseline。
  - **Autotuned execution path**
    - 在人工定义的候选配置集合中搜索。
    - 每个 backend 使用自身支持的调优参数。
    - 记录最优配置，并用于后续性能分析。
- 两条路径共同回答两个问题：
  - **Default path** 衡量 DSL 在常规开发条件下的性能表现。
  - **Autotuned path** 衡量在合理搜索预算内，参数调优还能释放多少性能潜力。

| 执行路径 | 配置来源 | 作用 | 代表含义 |
|---|---:|---|---|
| **Default** | 人工固定配置 | 提供稳定 baseline | 合理手写内核的默认性能 |
| **Autotuned** | 候选配置搜索 | 寻找更优 tile/occupancy 组合 | 参数级调优后的性能上限 |
| **PyTorch reference** | PyTorch 实现 | 正确性 oracle 与软件 baseline | 语义基准与速度对照 |

---

**Triton 的默认与调优参数**

- Triton 的核心调优维度包括：
  - **BLOCK_SIZE**
    - 控制每个 program instance 处理的数据规模。
    - 对 pointwise、reduction、layout 类 kernel 影响显著。
  - **num_warps**
    - 控制每个 Triton program 使用的 warp 数。
    - 影响并行度、occupancy、寄存器压力和 reduction 效率。
  - **num_stages**
    - 控制 software pipelining 的 stage 数。
    - 对 memory latency hiding、shared-memory staging 和 Tensor Core pipeline 有影响。
- 对 GEMM/Attention 类 kernel，Triton 通常还会扩展到更结构化的 block 参数：
  - **BLOCK_M**
  - **BLOCK_N**
  - **BLOCK_K**
  - **GROUP_SIZE_M**
  - **num_warps**
  - **num_stages**
- 对 1D pointwise kernel，典型默认配置类似：
  - **BLOCK_SIZE=2048**
  - **num_warps=4**
  - **num_stages=2**
- 对 fp16/bf16 matmul，推荐起点类似：
  - **BLOCK_M=128**
  - **BLOCK_N=128**
  - **BLOCK_K=64**
  - **num_warps=8**
  - **num_stages=3**
- 对 fp32 matmul，配置通常更保守：
  - **BLOCK_M=128**
  - **BLOCK_N=64**
  - **BLOCK_K=32**
  - **num_warps=4**
  - **num_stages=2**
- 参数作用可以概括为：
  - **BLOCK_SIZE / BLOCK_M/N/K** 决定单个 CTA 或 program 的工作集大小。
  - **num_warps** 决定并行执行资源分配。
  - **num_stages** 决定内存访问与计算之间的流水深度。

---

**cuTile 的默认与调优参数**

- cuTile 的核心调优维度包括：
  - **TILE_SIZE**
    - 控制 ct.load、ct.mma、ct.store 操作中的 tile shape。
    - cuTile 的 tile shape 要求通常更静态，且维度需要是 **power-of-2**。
  - **occupancy**
    - 作为 kernel launch hint，影响每个 SM 上并发 CTA 数或资源分配倾向。
    - 与寄存器使用、shared memory 使用、TMA pipeline、Tensor Core pipeline 相关。
- 对 1D pointwise kernel，典型默认配置类似：
  - **TILE=2048**
  - **occupancy=8**
- 对 fp16/bf16 matmul，典型默认配置类似：
  - **tm=128**
  - **tn=128**
  - **tk=64**
  - **occupancy=8**
- cuTile 参数的作用可以概括为：
  - **TILE_SIZE** 决定静态 tile 抽象的粒度。
  - **occupancy** 决定 launch 层面的并发策略。
  - **ct.load / ct.mma / ct.store** 的组合决定是否能自然映射到 **TMA**、**Tensor Core**、**tcgen05** 等 Blackwell-native 路径。
- cuTile 的调优空间较 Triton 更偏向：
  - 静态 tile shape。
  - 规则内存访问。
  - Tensor Core/TMA 友好路径。
  - occupancy hint 与资源约束之间的平衡。

---

**两类 backend 的参数对比**

| 维度 | **Triton** | **cuTile** | 对性能的主要影响 |
|---|---|---|---|
| Tile 粒度 | **BLOCK_SIZE**, **BLOCK_M/N/K** | **TILE_SIZE**, **tm/tn/tk** | 单个 program/CTA 的工作集大小 |
| 并行度 | **num_warps** | **occupancy** | SM occupancy、warp 利用率、调度效率 |
| 流水深度 | **num_stages** | 主要由 tile primitive 与编译器调度决定 | memory latency hiding、pipeline overlap |
| 内存表达 | pointer arithmetic、masked load | **ct.load**, **ct.gather**, **ct.store**, **ct.scatter** | 规则/不规则访问效率 |
| Tensor Core 路径 | **tl.dot** | **ct.mma** | MMA 指令选择与数据 staging |
| TMA 支持 | tensor descriptor 路径可触发 | 静态 tile path 更自然 | 大 tile 数据搬运与复用 |
| 约束特征 | 参数灵活，pointer-level 表达强 | tile shape 更静态，power-of-2 约束强 | irregular kernel 中 Triton 更自然，regular tile kernel 中 cuTile 更直接 |

---

**实现原理：为什么不能做一一对应调参**

- Triton 与 cuTile 的抽象层不同：
  - Triton 更接近 **block program + pointer arithmetic**。
  - cuTile 更接近 **tile primitive + static tile-space indexing**。
- 即使两个配置看似相同，例如都使用 128×128×64 tile：
  - Triton 可能生成 pointer load、shared-memory staging、ldmatrix、MMA 或 fallback path。
  - cuTile 可能生成 ct.load、TMA、TMEM、tcgen05 或 ct.mma path。
- 同名或近似参数并不保证底层行为一致：
  - Triton 的 **num_stages** 是显式 software pipeline hint。
  - cuTile 的 **occupancy** 更像 launch/resource hint。
  - Triton 的 **BLOCK_SIZE** 可用于任意 pointer mask。
  - cuTile 的 **TILE_SIZE** 受到静态 tile shape 和 power-of-2 约束。
- TileBench 因此采用 **comparable tuning ranges**，而不是 **identical tuning knobs**。
- 这种设计避免两个问题：
  - 用 Triton 的参数体系强行约束 cuTile，导致 cuTile 不自然。
  - 用 cuTile 的静态 tile 体系限制 Triton，导致 Triton pointer-level 优势被抹平。

---

**算法流程：Default Path**

- 每个算子都有三份实现：
  - **impl_torch.py**
  - **impl_triton.py**
  - **impl_cutile.py**
- Default path 的执行流程为：
  - 读取 **config.yaml** 中声明的 dtype、case grid、tolerance、FLOP/HBM-byte 公式。
  - 为当前算子选择人工固定配置。
  - 对每个 dtype 和 input case：
    - PyTorch 运行 reference，生成正确性输出。
    - Triton 使用默认配置 launch kernel。
    - cuTile 使用默认配置 launch kernel。
    - 输出与 PyTorch reference 对齐验证。
    - 通过验证后统计 kernel latency。
  - 根据 latency 计算：
    - **speedup over PyTorch**
    - **TFLOPS**
    - **effective bandwidth**
    - **arithmetic intensity**
    - **roofline utilization**
- Default path 的输入输出关系：
  - 输入：
    - 算子输入 tensor。
    - dtype。
    - shape/case 参数。
    - backend 默认配置。
  - 输出：
    - kernel 输出 tensor。
    - latency。
    - correctness status。
    - backend config。
    - 派生性能指标。

---

**算法流程：Autotuned Path**

- Autotuned path 在 Default path 基础上加入候选配置搜索。
- 执行流程为：
  - 为每个 backend 定义候选配置集合。
  - 对 Triton：
    - 遍历不同 **BLOCK_SIZE / BLOCK_M/N/K**。
    - 遍历不同 **num_warps**。
    - 遍历不同 **num_stages**。
  - 对 cuTile：
    - 遍历不同 **TILE_SIZE / tile shape**。
    - 遍历不同 **occupancy**。
  - 对每个候选配置：
    - 编译或实例化 kernel。
    - 执行正确性验证。
    - 通过验证后计时。
    - 计算性能指标。
  - 选择最优配置：
    - 通常以 latency 最低或 roofline utilization 更高为准。
    - 记录 winning configuration。
  - 聚合每个算子的 autotuned performance。
- Autotuned path 的输入输出关系：
  - 输入：
    - 算子输入 tensor。
    - dtype 与 shape sweep。
    - backend 候选调优空间。
  - 输出：
    - 最优 kernel 输出。
    - best latency。
    - best config。
    - autotuned roofline utilization。
    - default-to-autotuned gain。

---

**与整体 TileBench pipeline 的关系**

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

- Comparable Default and Autotuned Execution Paths 位于 TileBench 的 **benchmark-execution stage**。
- 它连接了三个部分：
  - **PyTorch reference**
    - 定义语义。
    - 提供 correctness oracle。
    - 提供 speedup baseline。
  - **Triton/cuTile implementations**
    - 提供手写 kernel。
    - 提供 default 与 autotuned 两种执行路径。
  - **comparative analysis**
    - 聚合 latency、speedup、throughput、bandwidth、roofline utilization。
    - 对大性能差距案例做 Nsight Compute 与 PTX/SASS 诊断。
- 在整体中，该机制的作用是：
  - 将“语言/编程模型差异”与“参数选择差异”尽量分离。
  - 判断性能差距来自：
    - 参数没有调好。
    - backend lowering 不理想。
    - memory staging 不佳。
    - Tensor Core/TMA 路径未触发。
    - shared-memory layout 或 bank conflict 问题。
  - 为 RQ1、RQ2、RQ3 提供一致实验基础。

---

**关键性能数据**

| 指标 | **Triton** | **cuTile** | 解释 |
|---|---:|---:|---|
| Default 几何平均 speedup over PyTorch | **2.7×** | **2.2×** | Triton 默认整体更强 |
| Default 击败 PyTorch 的算子数 | **36/45** | **34/45** | 两者多数情况下优于 PyTorch |
| Default median speedup | **3.1×** | **2.7×** | 中位数同样显示 Triton 更稳 |
| cuTile 快于 Triton 的算子数 | － | **11/45** | 主要集中于 Tensor Core/TMA 友好 kernel |
| Autotune roofline gain 几何平均 | **1.20×** | **1.15×** | 调参有收益，但不是决定性因素 |
| Autotuned 后达到 ≥80% roofline 的算子数 | **8/45** | **4/45** | 大多数 kernel 仍有明显 roofline headroom |
| Autotune gain 中位数 | **1.07×** | **1.04×** | 多数算子收益温和 |
| Autotune regression 数量 | **5** | **8** | 部分默认配置已接近搜索空间最优 |

---

**Default-to-Autotuned 的性能含义**

![](95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg)

- 图中的箭头表示：
  - 起点是 default roofline utilization。
  - 终点是 autotuned roofline utilization。
- 主要结论：
  - Autotuning 对两者都有收益。
  - Triton 的平均收益略高于 cuTile。
  - 但收益幅度有限，说明默认配置已经较合理。
  - 调优只能修正 tile size、warps、stages、occupancy 等参数。
  - 调优无法改变 backend compiler 的 lowering path。
- 更关键的发现：
  - 即使 autotuned 后，大量 kernel 仍远低于 roofline。
  - 这说明瓶颈常常不在参数，而在：
    - 指令选择。
    - Tensor Core/TMA 是否触发。
    - shared-memory staging。
    - register pressure。
    - local memory spill。
    - bank conflict。
    - irregular memory access 表达方式。

---

**参数设置背后的硬件约束**

- TileBench 的参数范围不是任意设定，而是受 **NVIDIA B200** 硬件约束影响。
- B200 相关关键约束包括：
  - **180GB HBM3e**
  - measured peak bandwidth：**6539.4 GB/s**
  - shared memory：约 **228KB/CTA**
  - register file 资源限制。
  - Tensor Core 支持 fp16、bf16、fp8、int8、tf32 等路径。
- 对 Triton：
  - BLOCK 过小：
    - launch overhead 占比高。
    - memory coalescing 与 occupancy 可能不足。
  - BLOCK 过大：
    - register pressure 上升。
    - shared memory 占用上升。
    - occupancy 下降。
  - num_stages 过低：
    - memory latency hiding 不足。
  - num_stages 过高：
    - shared memory 与 register 使用膨胀。
- 对 cuTile：
  - TILE_SIZE 过小：
    - tile reuse 不足。
    - TMA/Tensor Core 路径收益不明显。
  - TILE_SIZE 过大：
    - power-of-2 padding 浪费增加。
    - register pressure 或 shared memory pressure 增加。
  - occupancy 过低：
    - SM 并发不足。
  - occupancy 过高：
    - 每个 CTA 可用资源减少，可能恶化 spilling 或 pipeline 调度。

---

**不同算子类型中的调优行为**

| 算子类别 | Triton 调优重点 | cuTile 调优重点 | 常见瓶颈 |
|---|---|---|---|
| **Point-wise** | BLOCK_SIZE、num_warps | TILE_SIZE、occupancy | HBM bandwidth、tail mask |
| **Reduction/Normalization** | BLOCK_N、num_warps、reduction layout | TILE_SIZE、padding mode、occupancy | reduction tree、shared memory、warp utilization |
| **Matrix Multiplication/Attention** | BLOCK_M/N/K、num_warps、num_stages | tm/tn/tk、occupancy、ct.mma | Tensor Core utilization、TMA overlap、register pressure |
| **Stencil/Convolution** | pointer tile、mask、BLOCK spatial size | static tile staging、ct.load/gather | boundary handling、operand materialization |
| **Data Layout** | pointer arithmetic、coalescing、mask | ct.store/scatter、tile layout | irregular indexing、scatter/gather、bank conflict |

---

**为什么 Autotuning 后仍有大量 roofline headroom**

- 参数搜索只覆盖 **configuration-level optimization**。
- 它无法自动改变以下底层行为：
  - 是否使用 TMA。
  - 是否使用 tcgen05。
  - 是否走 Tensor Core native path。
  - 是否产生额外 LDS/STL。
  - 是否引入 local memory spill。
  - shared-memory layout 是否有 bank conflict。
  - packed low-bit 数据是否能直接进入合适 MMA pipeline。
- 论文中的案例说明：
  - **matmul_int8**
    - cuTile 能使用 Blackwell-native path。
    - Triton fallback 到 legacy cp.async → ldmatrix → register-fragment → IMMA pipeline。
    - 这不是简单调 tile size 可以解决的问题。
  - **2d_conv/fp32**
    - cuTile 在 irregular virtual-im2col materialization 中产生更重的 gather-mask-stage 链。
    - Triton 的 pointer tile + masked tl.load 表达更自然。
  - **streamk_matmul/bf16**
    - cuTile 的瓶颈来自 Tensor Core/TMEM pipeline 上的等待。
    - NCU 显示 stalled predicated branch 实际依赖 async phase-check。
- 这些问题都属于 **compiler lowering / memory behavior / instruction scheduling** 层面，不属于单纯参数调优层面。

---

**输入输出关系的形式化理解**

- 对任意算子 **o**、backend **b**、dtype **d**、case **c**，TileBench 记录：
  - **T_o,b,d,c**
    - 平均 kernel latency。
  - **F_o,d,c**
    - config.yaml 声明的 analytical FLOP count。
  - **B_o,d,c**
    - config.yaml 声明的 analytical HBM-byte count。
- Default path 与 Autotuned path 的区别体现在 backend 配置：
  - Default：
    - 使用固定 config。
  - Autotuned：
    - 在候选 config 集合中选择 best config。
- 输出指标包括：
  - **Speedup over PyTorch**
    - PyTorch latency / backend latency。
  - **Achieved throughput**
    - FLOP / latency。
  - **Arithmetic intensity**
    - FLOP / HBM bytes。
  - **Roofline utilization**
    - achieved throughput / roofline bound。
- 该机制使得不同 backend 的输出可以在同一坐标系下比较：
  - 同一算子语义。
  - 同一 dtype。
  - 同一 input shape。
  - 同一 correctness checker。
  - 同一 timing protocol。
  - 同一 FLOP/byte 公式。
  - 同一 roofline 常量。

---

**计时与验证协议**

- Default path 与 Autotuned path 共享统一 measurement protocol：
  - 使用 **Proton profiler**。
  - 使用 **CUDA graph capture-and-replay**。
  - 执行 **L2-cache flushing**。
  - **20 warmup runs**。
  - **100 timed runs**。
  - 报告 mean kernel latency。
- 正确性验证规则：
  - 浮点输出使用 **torch.testing.assert_close**。
  - tolerance 由 dtype 和 operator 决定。
  - 整数输出要求 exact comparison。
- 该协议避免以下偏差：
  - PyTorch、Triton、cuTile 计时方式不一致。
  - cache 状态差异影响结果。
  - 小 kernel launch overhead 波动过大。
  - 不正确 kernel 被纳入性能统计。

---

**与 LLM-generated track 的区别**

- 主 benchmark track：
  - 使用人工编写并验证的 Triton/cuTile kernel。
  - 同时评估 Default path 与 Autotuned path。
  - 重点是 **programming-model performance** 与 **bottleneck diagnosis**。
- LLM-generated track：
  - 不使用 backend autotuning。
  - 每轮 LLM 必须提交单一固定配置。
  - 10 次 refinement iteration 本身就是搜索过程。
  - 禁止使用 **triton.autotune**、**CutileAutotuner** 或其他自动搜索机制。
- 因此：
  - 主 track 的 autotuning 是系统化参数搜索。
  - LLM track 的“调优”是模型通过反馈手动修改配置与代码。
  - 两者服务于不同研究问题，不应混淆。

---

**设计价值与技术判断**

- **Comparable Default and Autotuned Execution Paths** 的价值在于：
  - 保持 backend 自然表达方式。
  - 避免不合理的一一参数映射。
  - 控制调优预算，提升比较公平性。
  - 区分参数问题与编译器/codegen 问题。
  - 为后续 NCU、PTX/SASS 诊断提供可追踪配置。
- 该设计支撑了论文的关键判断：
  - Triton 在整体上更稳健，尤其适合 irregular、streaming、bandwidth-bound operator。
  - cuTile 在 static tile、Tensor Core、TMA-friendly workload 上有优势。
  - Autotuning 有收益，但不能消除多数 roofline headroom。
  - 真正的大性能差距往往来自 backend lowering、memory staging、instruction selection，而不是简单 tile 参数选择。

### 5. Profiling-Guided Bottleneck Diagnosis

**核心定位**

- **Profiling-Guided Bottleneck Diagnosis**是 TileBench 中用于解释 **Triton** 与 **cuTile** 性能差异的诊断流程。
- 它不只比较 latency 或 speedup，而是进一步追踪：
  - **生成代码差异**
  - **compiler lowering 差异**
  - **PTX/SASS 指令路径**
  - **memory staging 行为**
  - **TMA/tcgen05 使用情况**
  - **shared-memory bank conflicts**
  - **stall_long_scoreboard 等硬件级 stall 来源**
- 该流程的目标是把“哪个 backend 更快”转化为“为什么更快、瓶颈在哪里、下一步应该优化什么”。

---

**整体输入输出关系**

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| **TileBench harness** | operator、dtype、case、backend 实现 | latency、speedup、throughput、bandwidth、roofline utilization | 找出性能差异显著的候选 case |
| **Nsight Compute** | 选定 kernel profile | stall、memory、bank conflict、instruction、occupancy 等硬件指标 | 定位硬件瓶颈 |
| **PTX/SASS inspection** | backend 编译产物 | 实际指令路径、TMA/tcgen05/LDGSTS/LDS/PRMT 等指令证据 | 判断 compiler lowering 是否走到理想硬件路径 |
| **Source correlation** | NCU Source view、kernel 源码 | bottleneck source line | 将硬件事件映射回 DSL 代码 |
| **Diagnosis report** | 指标与代码证据 | backend 差异解释、优化建议 | 支撑 RQ3 的 case study 与设计结论 |

---

**诊断流程**

- **性能差异筛选**
  - TileBench 先在 45 个 operator 上运行默认配置与 autotuned 配置。
  - 对每个 operator–dtype–case 记录：
    - **mean kernel latency**
    - **speedup over PyTorch**
    - **TFLOPS**
    - **effective bandwidth**
    - **arithmetic intensity**
    - **roofline utilization**
  - 选取 **Triton–cuTile latency gap 最大**的 case 进入深入诊断。
  - Figure 4 用 slower-backend latency / faster-backend latency 展示最大差距。

![](f9e643fecb6869d7899b6b7a50bbcaf63d4acc756e36ecba362971843a1ab893.jpg)

- **硬件 profile 采集**
  - 对选定 case 使用 **Nsight Compute** 采集 kernel profile。
  - 关注指标包括：
    - **stall_long_scoreboard**
    - **shared-memory bank conflicts**
    - **register pressure**
    - **local memory spill**
    - **global/shared memory transaction**
    - **MMA/Tensor Core utilization**
    - **TMA activity**
    - **branch predicate dependency**
    - **issued/executed IPC**
  - NCU 不用于全量计时，而用于 selected post-hoc case studies。

- **PTX/SASS 检查**
  - 通过 PTX/SASS inspection 检查 compiler 实际生成的底层路径。
  - 重点判断是否出现：
    - **TMA 指令**，如 **UTMALDG.2D**
    - **tcgen05 / Blackwell Tensor Core path**
    - **UTCIMMA**
    - **TMEM accumulator**
    - **LDGSTS.E**
    - **LDS/STL**
    - **PRMT**
    - **cp.async**
    - **ldmatrix**
    - **IMMA**
    - **SYNCS.PHASECHK.TRANS64.TRYWAIT**
    - **STTM**
    - **FENCE.VIEW.ASYNC.T**
    - **SYNCS.ARRIVE**

- **Source-level attribution**
  - 使用 NCU Source view 将 stall 或 conflict 归因到源码行。
  - 需要避免简单地把 NCU 标出的行当成根因。
  - 例如：
    - NCU 可能把 stall_long_scoreboard 标到 **ct.mma** 行。
    - 但 SASS 可能显示真正等待的是前面的 **gather–mask–stage chain**。
    - 因此诊断要追踪 producer–consumer dependency，而不是只看 source line 标签。

- **形成瓶颈解释**
  - 把性能差距归因到具体机制：
    - **memory access expression**
    - **tile materialization**
    - **TMA 是否被触发**
    - **Tensor Core/TMEM pipeline 是否被充分 overlap**
    - **shared-memory layout 是否产生 bank conflict**
    - **compiler lowering 是否退化到 legacy pipeline**
    - **register spilling 是否压低 occupancy 或增加 local memory traffic**

---

**关键参数与测量设置**

| 参数 | TileBench 设置 |
|---|---|
| GPU | **NVIDIA B200** |
| HBM | **180 GB HBM3e** |
| measured peak bandwidth | **6539.4 GB/s** |
| PyTorch | **2.10 + CUDA 13.0** |
| Triton | **3.6.0** |
| cuda-tile | **1.3.0** |
| timing profiler | **Proton** |
| diagnosis profiler | **Nsight Compute** |
| warmup | **20 runs** |
| timed runs | **100 runs** |
| timing mode | **CUDA graph capture-and-replay** |
| cache control | **L2-cache flushing** |
| diagnosis scope | selected large-gap cases |

---

**诊断指标体系**

- **latency**
  - 直接衡量 backend 的端到端 kernel 时间。
  - 用于筛选差距 case。
- **roofline utilization**
  - 衡量实现距离硬件上限的比例。
  - 公式核心是：
    - achieved TFLOPS / min(dtype peak compute, arithmetic intensity × peak bandwidth)
  - 用于区分：
    - **tile 参数不足**
    - **compiler lowering 不佳**
    - **memory behavior 限制**
- **instruction mix**
  - 判断 kernel 主要时间花在计算、搬运、重排还是同步。
  - 例如 matmul_int8 中：
    - Triton 出现大量 **PRMT** 和 **LDS**。
    - cuTile 使用 **UTMALDG.2D**、**UTCIMMA**、**TMEM**。
- **stall_long_scoreboard**
  - 表示 warp 等待长延迟依赖。
  - 可能来自：
    - global memory load
    - shared/local memory dependency
    - async pipeline phase wait
    - TMA/mbarrier synchronization
- **shared-memory bank conflict score**
  - 用于量化 shared-memory bank conflict 的相对严重程度。
  - TileBench 定义：
    - **C_LSU = 100 × B_LSU / W_LSU**
  - 其中：
    - **B_LSU**是 LSU shared-memory path 的 bank-conflict event 数。
    - **W_LSU**是服务 LSU shared-memory traffic 的 L1TEX wavefront 数。
  - 该指标是 normalized severity，不是精确 source-level speedup 估计。

---

**Bank Conflict 诊断方法**

- TileBench 使用 NCU profile 采集 204 个 kernel 的 shared-memory conflict 相关指标。
- 主要 NCU counters：
  - **l1tex_data_bank_conflicts_pipe_lsu_mem_shared.sum**
  - **l1tex_data_pipe_lsu_wavefronts_mem_shared.sum**
- 归一化 conflict score：
  - 消除 kernel 规模差异。
  - 让不同 operator、dtype、backend 更可比。
- 诊断分类包括：
  - **confirmed direct LSU conflict evidence**
  - **weaker source-derived evidence**
  - **divergence-confounded attribution**
  - **no direct LSU conflict evidence**

![](eecac1f3f846e36c1c4ebbc4ff6729c5eda72ef2563106488aac0344e4ef83b5.jpg)

![](f40e30cb49a6e88710c31c7a1255f22ad48c646bcd4612828aa878bf469b88af.jpg)

- 关键发现：
  - 严重 bank conflict 不是某一个 backend 独有。
  - top-20 conflict cases 中：
    - **6 个是 cuTile**
    - **14 个是 Triton**
  - 高 conflict score 集中在少数 tile-heavy kernels，而不是全 suite 均匀出现。
  - conflict 站点主要来自：
    - shared-memory stores
    - shared-memory loads
    - global-to-shared copies
    - MMA operand staging

![](e055de22c98b3a5a7cd8160a32a8750a8815cc4733f2537d3189b67514330c8a.jpg)

---

**compiler lowering 诊断重点**

- **Triton**
  - 优势：
    - pointer-level programming model 更自然表达 arbitrary address。
    - **tl.load + mask** 可以把 boundary predicate 直接绑定到 load。
    - 对 irregular access、runtime-computed indices、streaming kernels 更稳健。
  - 风险：
    - 对某些 Blackwell-native path 触发不充分。
    - 未必自动生成 **TMA/tcgen05**。
    - packed low-bit + runtime unpack 的场景可能退化到 legacy pipeline。
- **cuTile**
  - 优势：
    - **ct.load / ct.mma / ct.store** 对 regular tile 更贴近 Blackwell native path。
    - 更容易触发 **TMA**、**tcgen05**、**TMEM accumulator**。
    - 对 dense GEMM、attention、stencil/conv 中可复用 tile movement 的 workload 更有优势。
  - 风险：
    - **ct.load** 更适合 affine tile-space access。
    - runtime-computed gather、mask、where 可能导致重的 producer chain。
    - **ct.gather / ct.where / indirect materialization** 可能增加 register pressure、shared/local traffic 与 spilling。

---

**Case Study：2d_conv/fp32**

- 问题背景：
  - 该 case 使用 true-FP32 arithmetic path，不是 Tensor Core 主导。
  - Triton 与 cuTile 都需要构造 virtual-im2col operands。
- cuTile 瓶颈：
  - NCU 将大量 **stall_long_scoreboard** 归因到 **ct.mma** 行。
  - SASS 显示周围存在大量：
    - **LDS**
    - **STL**
    - local/shared-memory dependency
  - 说明真正瓶颈不是 FP32 arithmetic，而是 operand materialization。
- 根因：
  - cuTile 的 regular **ct.load** 更适合规则 tile。
  - 该实现中 input/weight operands 通过 linearized per-element indices 和 masks 构造。
  - 形成较重链路：
    - index computation
    - indirect load
    - mask selection
    - register/shared/local memory staging
- Triton 优势：
  - Triton 直接构造 pointer tile。
  - boundary predicate 直接附着到 **tl.load**。
  - 避免先 gather 成 tile 再 select。
- 诊断结论：
  - 对 irregular virtual tile，**pointer + predicate load** 通常优于 **gather + where + staging**。
  - 该 case 暴露的是 programming model 对逻辑 operand tile 的表达成本差异。

---

**Case Study：streamk_matmul/bf16**

- 问题背景：
  - stream-k matmul 依赖 K-loop 与 Tensor Core/TMEM pipeline。
  - 性能关键不是单次 MMA，而是 async pipeline overlap。
- cuTile 瓶颈：
  - NCU 把 dominant **stall_long_scoreboard** 标到 predicated **BRA** loop branch。
  - 进一步追踪 SASS predicate producer 发现：
    - branch 依赖 predicate **P0**
    - P0 来自 **SYNCS.PHASECHK.TRANS64.TRYWAIT**
  - 周围 SASS 包含：
    - **STTM**
    - **FENCE.VIEW.ASYNC.T**
    - **SYNCS.ARRIVE**
- 根因：
  - branch 本身不是瓶颈。
  - branch 正在等待 asynchronous MMA/TMEM phase 完成。
  - 本质是 **Tensor Core/TMEM pipeline overlap 不足**。
- 诊断规则：
  - 看到 predicated branch stall 时，不应直接归因到 control flow。
  - 应追踪 predicate producer。
  - 如果 producer 是 async phase-check instruction，瓶颈是 **async pipeline wait**。

---

**Case Study：matmul_int8**

- 问题背景：
  - operator 是 INT8 GEMM，B 矩阵使用 2-bit quantized weight。
  - 每个 kernel 需要在 register 中 unpack B，然后执行 MMA。
  - Triton 和 cuTile 算法结构相近，autotuned tile shape 相同：
    - **128×128×64**
  - grid size 相同：
    - **256 CTAs**
  - 两者都 register-limited：
    - **255 registers/thread**
    - **12.5% theoretical occupancy**

| 指标 | Triton | cuTile | 解释 |
|---|---:|---:|---|
| latency | **477.44 µs** | **346.88 µs** | cuTile 快 **1.38×** |
| SASS instructions | **187.0M** | **78.8M** | cuTile 指令数少 **2.37×** |
| main path | cp.async → ldmatrix → register-fragment → IMMA | UTMALDG.2D + UTCIMMA + TMEM | cuTile 更贴近 Blackwell native path |
| LDS | **42.0M** | **0.17M** | Triton shared-memory staging 更重 |
| PRMT | 高 | 降低 **3.7×** | Triton unpack/reorder 成本更高 |
| TMA/tcgen05 | 无 | 有 | Triton lowering 未触发 native path |

- Triton 瓶颈：
  - 未生成 TMA 或 tcgen05。
  - packed B unpack 后变成 runtime-computed register tile。
  - compiler 未能把该模式 lower 到 Blackwell-native path。
  - 退化为 legacy **cp.async → ldmatrix → register-fragment → IMMA**。
- cuTile 优势：
  - 使用 **UTMALDG.2D**、**UTCIMMA**、**TMEM accumulator**。
  - 大幅减少 shared-memory round trip 与 operand staging。
- cuTile 剩余问题：
  - unpacked-B scratch tile 仍有 shared-memory layout inefficiency。
  - NCU 显示：
    - **4.4-way store conflicts**
    - **77% conflicted shared-store wavefronts**
    - **74% uncoalesced MMA reads**
- 诊断结论：
  - Triton 的主要问题是 **packed-low-bit-aware MMA lowering 不足**。
  - cuTile 的主要问题是 **intermediate tile swizzling/padding 不足**。

---

**Case Study：flash_attention**

- 问题背景：
  - FP16 non-GQA flash_attention。
  - 参数：
    - batch_size = **4**
    - n_heads = **32**
    - head_dim = **128**
    - sequence_length = **20480**
- 性能数据：

| Backend | latency | 特征 |
|---|---:|---|
| Triton | **22.77 ms** | 未使用 tensor_descriptor，因此未走 TMA |
| cuTile | **17.70 ms** | 使用 TMA 与 TC-Gen05 |
| cuTile compute throughput utilization | **43.2%** | Tensor Core under-utilized |

- cuTile 虽然更快，但仍未达到理想效率。
- 诊断发现两个问题：
  - **register utilization 极高**
    - live register count 达到 **185**
    - 说明 register pressure 大
    - 可能来自 TMEM/SMEM/register file 之间不必要的数据移动
  - **TMA load latency 未被 computation 覆盖**
    - warp stall 主要来自 mbarrier synchronization 的 **stall_long_scoreboard**
    - 表示 KV tensors 的 TMA load 等待时间没有被计算隐藏
- 诊断结论：
  - 对 flash_attention，cuTile 的 TMA/tcgen05 路径有效。
  - 但性能瓶颈转移到：
    - **register allocation**
    - **TMA/computation overlap**
    - **instruction scheduling**
    - **mbarrier synchronization latency**

---

**Triton Tensor Descriptor / TMA Ablation**

- TileBench 额外评估 Triton tensor descriptor / TMA path。
- 结论不是“descriptor 总是更好”。
- 结果显示：
  - descriptor 只帮助少量 tile-reuse-heavy kernels：
    - **flash_attention**
    - **matmul_int8**
    - **streamk_matmul**
  - 对多数 one-pass kernels 反而回退：
    - pointwise
    - data layout
    - normalization
    - reduction
- 实用规则：
  - 当 tile movement 能被复用或与 compute overlap，使用 descriptor/TMA。
  - 当 kernel 是 streaming one-pass，plain pointer load 更合适。
  - 对 1D pointwise，不应为了 TMA 引入额外 shared-memory staging。

![](8280c07c994efbcaeb79af11fbfd5ac7be69c82490fbbab5d2e2b7956adea84c.jpg)

---

**诊断中最重要的因果链**

- **latency gap**
  - 只是现象。
- **roofline utilization**
  - 判断距离硬件上限有多远。
- **NCU stall / memory / conflict metrics**
  - 找到硬件层面的等待或低效。
- **PTX/SASS instruction mix**
  - 判断 backend 是否生成理想硬件路径。
- **Source correlation**
  - 将底层瓶颈映射回 Triton 或 cuTile 代码结构。
- **compiler-lowering explanation**
  - 解释为什么同一算法在两个 DSL 下产生不同机器码。
- **optimization rule**
  - 形成可迁移经验，而不是只解释单个 case。

---

**典型瓶颈模式与优化方向**

| 瓶颈模式 | 诊断证据 | 常见 backend | 优化方向 |
|---|---|---|---|
| **irregular gather materialization** | LDS/STL、多级 index/mask/select、stall_long_scoreboard | cuTile 更常见 | 避免 ct.gather/ct.where 链；改写为规则 tile 或 predicate load |
| **legacy MMA pipeline** | cp.async、ldmatrix、PRMT、LDS 多；无 TMA/tcgen05 | Triton 某些 low-bit GEMM | 改用 descriptor/TMA；改善 packed-low-bit lowering |
| **async MMA/TMEM wait** | BRA stall，predicate 来自 SYNCS.PHASECHK | cuTile | 增加 pipeline depth；改善 producer-consumer overlap |
| **TMA latency uncovered** | mbarrier stall_long_scoreboard | cuTile attention | 改善 scheduling；增加 double buffering；降低 register pressure |
| **shared-memory bank conflict** | high C_LSU、conflicted wavefronts | 两者都有 | swizzling、padding、调整 tile layout |
| **register spilling** | high register usage、local memory traffic | 两者都有 | 降低 live range；拆分 tile；减少中间 tile materialization |
| **descriptor overuse** | descriptor path regressions | Triton | streaming kernel 使用 pointer load |

---

**在 TileBench 整体中的作用**

- **支撑 RQ3**
  - RQ1/RQ2 给出性能现象。
  - Profiling-guided diagnosis 解释这些现象背后的代码生成与硬件原因。
- **区分 programming model 能力边界**
  - Triton 更适合：
    - irregular access
    - masked pointer load
    - streaming bandwidth-bound kernels
  - cuTile 更适合：
    - regular tile
    - reusable tile movement
    - Tensor Core/TMA-friendly kernels
- **解释 autotuning 的局限**
  - Autotuning 只能搜索 tile size、warps、stages、occupancy。
  - 它不能改变 compiler lowering path。
  - 很多 roofline headroom 来自：
    - instruction selection
    - shared-memory layout
    - TMA usage
    - Tensor Core pipeline
    - register pressure
- **提供 actionable optimization patterns**
  - 不是只报告“cuTile 慢”或“Triton 慢”。
  - 而是指出：
    - 哪条 SASS path 不理想
    - 哪个 source expression 触发重 staging
    - 哪种 tile layout 产生 bank conflict
    - 哪类 workload 应使用 TMA
    - 哪类 workload 应避免 descriptor staging

---

**关键结论**

- **Profiling-Guided Bottleneck Diagnosis**的核心价值在于把 benchmark 从性能排名推进到性能解释。
- TileBench 的诊断逻辑强调 **cross-layer evidence**：
  - DSL source
  - compiler lowering
  - PTX/SASS
  - NCU hardware counters
  - roofline utilization
- 主要发现是：
  - cuTile 在 **regular tile + TMA + Tensor Core/tcgen05** 场景中更容易生成 Blackwell-native path。
  - Triton 在 **irregular pointer arithmetic + masked load + streaming** 场景中更稳健。
  - bank conflict、register pressure、async wait、descriptor staging 都可能成为实际瓶颈。
  - 参数 autotuning 只能解决一部分问题，真正的大差距往往来自 **compiler-lowering choices** 和 **memory behavior**。

### 6. Iterative LLM Kernel Generation Track

**核心定位**

- **Iterative LLM Kernel Generation Track**是 TileBench 中独立于人工手写 kernel 主评测的一条实验轨道，用于评估 LLM 面向 **Triton** 与 **cuTile** 生成 GPU kernel 的能力。
- 该轨道关注的不只是最终 kernel 的速度，还同时衡量：
  - **Correctness**：生成代码是否能编译并通过 PyTorch reference 校验。
  - **Performance**：正确 kernel 相对 PyTorch 的加速比。
  - **Token Cost**：10 轮迭代中消耗的总 Token 数。
  - **Token Efficiency**：单位 Token 带来的有效性能收益。
- 其核心问题是：
  - 同样的 operator specification、同样的反馈机制、同样的迭代预算下，LLM 更容易为 **Triton** 还是 **cuTile** 生成正确且高性能的 kernel。
  - 编程模型本身是否影响 LLM 的代码生成效率。

---

**整体流程**

- 每个 operator 都被转换为一个 LLM kernel generation 任务。
- LLM 需要基于自然语言描述、PyTorch reference、config.yaml、框架约束和 API reference，生成两个文件：
  - `impl_triton.py`
  - `impl_cutile.py`
- 每个 backend 独立评估：
  - Triton 生成结果只计入 Triton backend。
  - cuTile 生成结果只计入 cuTile backend。
- 每个 operator 每个 backend 运行 **10 个 refinement iterations**。
- 每一轮都会经历：
  - Prompt 构造。
  - LLM 生成代码。
  - 静态规则检查。
  - 编译与加载。
  - 正确性校验。
  - 性能测量。
  - 结构化反馈进入下一轮。
- 最终结果不是最后一轮，而是：
  - 从 10 轮中选择 **best verify-clean iteration**。
  - 即所有 evaluated dtypes 都正确，并且 correctness-gated speedup 最高的一轮。

---

**输入与输出关系**

| 类型 | 输入内容 | 作用 |
|---|---|---|
| Operator description | 自然语言问题定义 | 给出数学语义、输入输出契约、边界条件 |
| `impl_torch.py` | PyTorch reference implementation | 作为正确性 oracle 和 PyTorch baseline |
| `config.yaml` | dtype、case grid、tolerance、benchmark options、FLOP/byte 公式 | 规定测试空间和指标计算方式 |
| Framework guide | 文件结构、run contract、禁止行为、硬件约束 | 限制 LLM 输出满足 TileBench harness |
| Triton API reference | Triton 3.6.0 可用 API | 防止使用不兼容 API |
| cuTile API reference | cuda-tile 1.3.0 可用 API | 约束 cuTile 生成语法 |
| Previous trajectory | 历史配置、验证状态、速度、roofline feedback | 指导下一轮优化 |
| Previous source files | 上一轮生成代码 | 允许 LLM 局部修改或回退 |
| Best verify-clean code | 当前最优正确代码 | 防止回归后丢失可用实现 |

- LLM 输出固定为两个代码文件：
  - **Triton kernel + Python wrapper**
  - **cuTile kernel + Python wrapper**
- 每个文件必须导出：
  - `run(*args, **kwargs)`
  - `get_last_config() -> dict | None`
- `run()` 的签名必须与 PyTorch reference 完全一致。
- 输出 tensor 必须与 PyTorch reference 在 `config.yaml` 指定 tolerance 下匹配。
- `get_last_config()` 用于返回当前 iteration 的配置，进入下一轮 feedback。

---

**迭代算法流程**

- **Iteration 0**
  - Prompt 从零构造，不包含历史轨迹。
  - 输入包括：
    - TileBench framework guide。
    - Triton API reference。
    - cuTile API reference。
    - Operator description。
    - `config.yaml`。
    - PyTorch reference。
    - Strict output-format instructions。
  - LLM 必须同时生成 `impl_triton.py` 和 `impl_cutile.py`。
  - 生成代码随后被 harness 加载、校验、计时。

- **Iteration N > 0**
  - Prompt 每轮重新构造，而不是在长上下文中继续 append。
  - 新 prompt 会加入：
    - 历史 trajectory table。
    - 上一轮每个 backend 的 source file。
    - 编译错误、verification failure、timeout、performance regression 等反馈。
    - 每个 backend 的 worst-first performance summary。
    - 每个 dtype/case 的 roofline utilization。
    - 当前 best verify-clean source file。
  - 如果上一轮某 backend 出现回归或 verification failure：
    - Prompt 会提示 LLM 回到 best verify-clean 实现。
    - 再尝试不同优化方向。
  - 这样设计可以减少错误累积，避免模型在失败分支上持续漂移。

---

**评测执行路径**

- 对每个 operator、backend、iteration：
  - 生成代码先经过静态扫描。
  - 若触发禁止模式，直接失败，不进入 timing。
  - 若通过静态检查，则进入编译加载。
  - 编译失败记为该 iteration 不正确。
  - 编译成功后，对 representative large case per supported dtype 做 correctness check。
  - 所有 dtype 都通过时，该 iteration 才被视为 **correct**。
  - 正确后才测量 latency。
- 生成代码不使用 backend autotuning。
  - 每轮只能提交一个 hard-coded configuration。
  - 10 轮 refinement 本身就是搜索机制。
- 与主 benchmark track 的区别：
  - 主轨道使用人工写的 Triton/cuTile kernel。
  - 主轨道有 default path 和 autotuned path。
  - LLM 轨道只测 LLM 生成的单配置 kernel。
  - LLM 轨道额外计入 Token 消耗。

---

**关键参数设置**

| 参数 | 设置 | 含义 |
|---|---:|---|
| Iteration budget | 10 | 每个 operator/backend 最多 10 轮 refinement |
| Backend | Triton、cuTile | 两个 tile-based DSL 独立追踪 |
| Models | GPT-5.5、Claude Opus 4.7 | 通过 API 调用的 reasoning models |
| Autotuning | 禁用 | 不允许 `triton.autotune` 或 cuTile autotuner |
| Case selection | 每个 supported dtype 选 1 个 representative large case | 降低 LLM 轨道成本，同时覆盖 dtype 差异 |
| Correctness gate | 所有 evaluated dtypes 必须通过 | 任一 dtype 失败则 speedup 记为 0 |
| Final selection | Best verify-clean iteration | 不是取最后一轮 |
| Timing | 与 TileBench harness 一致 | 使用 PyTorch baseline 与 generated kernel latency 对比 |

---

**禁止行为与反作弊机制**

- LLM 生成代码在 timing 前会被拒绝以下行为：
  - 调用 PyTorch、cuBLAS、cuDNN 或 reference implementation 完成实际计算。
  - 使用 `torch.matmul`、`torch.mm`、`torch.softmax`、`torch.sort`、`torch.topk` 等直接替代目标 operator。
  - 导入 `impl_torch.run`。
  - 定义 decoy kernel，但实际计算交给 PyTorch。
  - 基于 tensor identity、data pointer、version key 等缓存输出。
  - 使用 backend autotuner 绕过 10 轮手动 refinement。
- 检测方式包括：
  - 静态扫描 forbidden imports 和 forbidden APIs。
  - 两轮 anti-cache verification。
  - rotating-input timing。
  - 检查异常高于 roofline 的 suspicious throughput。
- 这些失败会作为 structured feedback 返回给下一轮 prompt。
- 错误 iteration 仍然计入 Token Cost。

---

**Correctness-Gated Speedup**

- LLM 轨道使用 **correctness-gated speedup** 衡量“有用性能”。
- 对 operator \(o\)、backend \(b\)、iteration \(i\)、dtype 集合 \(\mathcal{D}_o\)：
  - 若该 iteration 对所有 dtype 都正确：
    - 计算每个 dtype 上 PyTorch latency 与 generated kernel latency 的比值。
    - 对所有 dtype 求平均。
  - 若编译失败、验证失败或任一 dtype 不正确：
    - 该 iteration 的 speedup 直接记为 **0**。
- 公式语义：
  - \(G_{o,b,i}\) 是第 \(i\) 轮的 correctness-gated speedup。
  - \(T_{o,\mathrm{torch},d}\) 是 PyTorch reference latency。
  - \(T_{o,b,i,d}\) 是 LLM-generated implementation latency。
- 该设计避免把错误但很快的 kernel 计入收益。
- 也避免 LLM 通过近似计算、漏算边界、忽略 dtype 来获得虚假速度。

---

**BestSpeedup@10**

- **BestSpeedup@10** 表示 10 轮中最好的 correctness-gated speedup。
- 定义为：
  - \(\max_{0 \le i < 10} G_{o,b,i}\)
- 若某 backend 10 轮全部失败：
  - BestSpeedup@10 为 0。
- 使用 best 而非 final 的原因：
  - LLM refinement 可能出现 regression。
  - 某些优化尝试可能破坏 correctness。
  - 取 best verify-clean 更符合自动搜索过程的最终可用产物。

---

**TokenCost@10**

- **TokenCost@10** 是某 operator/backend 在 10 轮中消耗的总 Token。
- 计入范围包括：
  - 成功 iteration。
  - 编译失败 iteration。
  - correctness failure iteration。
  - 性能回退 iteration。
- 失败也计入 Token 的原因：
  - 编译错误和错误实现是 LLM 搜索过程的一部分。
  - 若不计失败成本，会高估不稳定 backend 的可用性。
- 该指标反映生成某种 DSL kernel 的交互成本和上下文复杂度。

---

**TokenEfficiency@10**

- **TokenEfficiency@10** 衡量单位 Token 产生的有效加速收益。
- 定义为：
  - \(\mathrm{BestSpeedup@10} / (\mathrm{TokenCost@10} / 10^6)\)
- 单位可理解为：
  - **speedup per million tokens**
- 该指标同时惩罚：
  - 低性能 kernel。
  - 频繁失败导致的高 Token 消耗。
  - 需要大量上下文才能修复的 DSL。
- 它是 RQ4 的核心指标，因为 LLM kernel generation 的实际成本不仅来自 GPU timing，也来自 API Token 预算。

---

**Prompt 设计机制**

- Prompt 不是简单让 LLM“写一个 kernel”，而是把 TileBench 环境完整暴露给模型。
- Prompt 由多个模块拼接而成：
  - **Framework conventions**
    - 文件命名。
    - `run()` 签名。
    - `_LAST_CFG` 使用规则。
    - 禁止 autotune。
    - 禁止 PyTorch delegation。
    - B200 tile-size bounds。
  - **Backend API references**
    - 限定可用 Triton/cuTile API。
    - 防止模型使用训练数据中较新或不存在的 API。
  - **Operator specification**
    - 数学定义。
    - 输入输出形状。
    - 示例。
  - **PyTorch reference**
    - 作为精确语义定义。
  - **Config**
    - dtype。
    - case grid。
    - tolerance。
    - metric formulas。
  - **Feedback**
    - 上一轮编译错误。
    - verification status。
    - speedup_vs_torch。
    - roofline_pct。
    - worst-case dtype/case。
- Iteration N 的 prompt 会包含上一轮代码，使模型能够做局部优化。
- 若上一轮失败，会附带 best verify-clean code，形成显式回退机制。

---

**配置搜索方式**

- LLM track 禁止 `triton.autotune` 与 cuTile autotuner。
- 搜索空间不是由程序自动枚举，而是由 LLM 在 10 轮中手动探索。
- 每轮必须选择单一配置：
  - Triton 常见参数：
    - `BLOCK_SIZE`
    - `BLOCK_M`
    - `BLOCK_N`
    - `BLOCK_K`
    - `num_warps`
    - `num_stages`
    - group size 等 launch scheduling 参数
  - cuTile 常见参数：
    - `TILE`
    - tile shape
    - `occupancy`
    - `ct.load`/`ct.store` tile dimensions
    - `ct.mma` tile dimensions
- `get_last_config()` 会把实际配置返回给 harness。
- 下一轮 prompt 使用这些配置和性能反馈，引导 LLM 调整 tile size、parallelism、occupancy 或 memory access strategy。

---

**与主 Benchmark Track 的差异**

| 维度 | Main Benchmark Track | Iterative LLM Kernel Generation Track |
|---|---|---|
| Kernel 来源 | 人工手写并验证 | LLM 生成 |
| Backend | Triton、cuTile | Triton、cuTile |
| Operator 数量 | 45 | 同一组 45 |
| 配置方式 | Default + autotuned | 每轮单一 hard-coded config |
| Autotuning | 启用 | 禁用 |
| 迭代次数 | 不适用 | 10 |
| 正确性 | 人工实现已验证 | 每轮 correctness gate |
| 主要指标 | latency、speedup、roofline utilization | BestSpeedup@10、TokenCost@10、TokenEfficiency@10 |
| 关注点 | 编程模型性能上限与瓶颈 | LLM 生成难度、稳定性、Token 成本 |

---

**实验结果概览**

![](336730bb607696cb1e402d8e5305e120d7e5b75eee5b3cbb91f30035d728ffb6.jpg)

| Model + Backend | Faster-than-PyTorch operators | Mean TokenCost@10 | TokenEfficiency@10 |
|---|---:|---:|---:|
| GPT-5.5 + Triton | 39/45 | 0.26M | 20.9 |
| GPT-5.5 + cuTile | 38/45 | 0.32M | 13.4 |
| Claude Opus 4.7 + Triton | 39/45 | 0.28M | 17.1 |
| Claude Opus 4.7 + cuTile | 35/45 | 0.42M | 8.3 |

- **Triton 在两个模型上都更 Token-efficient**。
- **cuTile 消耗更多 Token，但单位 Token 的加速收益更低**。
- Backend 差异大于 model 差异：
  - GPT-5.5 与 Claude 的差别存在。
  - 但 Triton 与 cuTile 的差别更稳定、更显著。
- 主要原因：
  - Triton 更成熟，公开代码更多，LLM pretraining corpus 中覆盖更充分。
  - cuTile 发布时间更新，公开样例少，API 与编程习惯更不稳定。
  - cuTile 的 tile abstraction 对规则 tensor tile 友好，但对 LLM 来说更容易在 index、padding、shape power-of-two、ct.store/static index 等细节上出错。

---

**Softmax 迭代轨迹案例**

![](e9e45d81864f1e99ab5bd4542c44ef7ee151bd4710d6eae1fd3a21ac42153072.jpg)

- Softmax 的 10 轮 refinement 展示了 backend 差异：
  - **GPT-5.5 + Triton**
    - iteration 0 就超过 PyTorch。
    - 后续保持验证通过。
    - iteration 8 达到 2.64×。
  - **Claude Opus 4.7 + Triton**
    - iteration 0 超过 PyTorch。
    - iteration 9 达到 1.73×。
  - **GPT-5.5 + cuTile**
    - iteration 0 verification failure。
    - iteration 4 曾跌到 0.76×。
    - iteration 9 恢复到 2.45×。
  - **Claude Opus 4.7 + cuTile**
    - 10 轮内未超过 PyTorch parity。
    - 大约停留在 0.43×。
- 该案例说明：
  - Triton 生成轨迹更稳定。
  - cuTile 轨迹波动更大。
  - cuTile 更容易出现 verification failure 或性能 regression。
  - 更多 iteration 不一定解决根因，因为瓶颈主要来自模型对 DSL 的先验知识不足。

---

**Triton 更适合 LLM 生成的技术原因**

- **Pointer-based programming model 更贴近 LLM 训练语料**
  - Triton 的 `tl.load(ptr + offsets, mask=mask)` 模式在公开 kernel 中非常常见。
  - LLM 更容易学会 offset、mask、block size 的组合。
- **边界处理表达更直接**
  - Triton 通过 `mask` 附着在 `tl.load` 和 `tl.store` 上。
  - 对非整除 shape、tail block、runtime index 更自然。
- **配置参数更常见**
  - `BLOCK_SIZE`、`num_warps`、`num_stages` 是 Triton kernel 优化中的常见模式。
  - LLM 更容易从历史代码中迁移。
- **错误反馈更容易映射到修改动作**
  - 若 OOB 错误，通常改 mask。
  - 若性能低，通常调 block size、warps、stages。
  - 若 reduction 慢，通常调 reduce dimension 或分层 reduction。
- **API 成熟度更高**
  - Triton 已被 vLLM、Liger-Kernel、Unsloth 等系统采用。
  - 公开样例覆盖 attention、normalization、quantization、pointwise fusion 等大量场景。

---

**cuTile 对 LLM 更困难的技术原因**

- **API 新且样例少**
  - cuTile 随 CUDA 13.1 在 2026 年发布。
  - LLM pretraining corpus 中相关代码少。
- **tile shape 限制更严格**
  - cuTile tile dimensions 必须是 power-of-two。
  - 非 power-of-two shape 需要 `padding_mode`。
- **static index 与 runtime scatter/gather 区分更复杂**
  - `ct.store()` 只接受 static indices。
  - runtime-computed scatter 需要 `ct.scatter()`。
  - LLM 容易混用导致编译失败或语义错误。
- **regular tile abstraction 对不规则访问不友好**
  - 对 affine tile-space access 很自然。
  - 对 gather、mask、boundary-dependent control、low-bit packing 更容易生成冗长或低效代码。
- **性能调优空间更依赖硬件语义**
  - TMA、Tensor Core、TMEM、occupancy、tile staging 之间关系复杂。
  - LLM 即使能写出正确代码，也未必能触发 Blackwell-native fast path。
- **错误恢复更困难**
  - 编译错误可能来自 API 约束、shape 约束、padding 语义或 launch contract。
  - 单轮反馈不一定足以让模型定位根因。

---

**在 TileBench 整体中的作用**

- **补充人工 kernel 性能评测**
  - 主轨道回答“哪个 backend 的手写 kernel 更快”。
  - LLM 轨道回答“哪个 backend 更容易被 LLM 写好”。
- **衡量编程模型的可生成性**
  - 一个 DSL 即使人工可写出高性能 kernel，也不一定适合 LLM 自动生成。
  - TokenEfficiency 把“易用性”量化为可比较指标。
- **揭示 DSL 成熟度影响**
  - Triton 的优势不仅来自性能，也来自生态和语料积累。
  - cuTile 的劣势部分来自新 DSL 的低样本环境。
- **为未来 AI-assisted kernel optimization 提供指标**
  - 仅看 final speedup 不够。
  - 需要同时看 correctness、search stability、iteration cost、Token cost。
- **支撑 RQ4 结论**
  - Triton 是当前更适合作为 LLM kernel generation target 的 DSL。
  - cuTile 在 Tensor Core/TMA-friendly workload 上潜力较高，但 LLM 使用成本更高。

---

**关键结论**

- **Iterative LLM Kernel Generation Track**将 kernel 生成建模为一个受控的 10 轮搜索过程。
- **Correctness-gated speedup**确保只有正确 kernel 才能贡献性能收益。
- **TokenCost@10**把失败尝试也计入成本，反映真实 LLM 开发代价。
- **TokenEfficiency@10**综合性能与 Token 成本，是衡量 LLM-friendly DSL 的核心指标。
- 实验显示：
  - **Triton 更稳定、更省 Token、更容易达到 PyTorch parity 以上**。
  - **cuTile 更容易出现 verification failure、性能波动和更高 Token 消耗**。
  - Backend 选择对 LLM kernel generation 的影响大于模型选择本身。
- 该轨道说明，未来 GPU DSL 的设计不仅要考虑 human expert performance，也要考虑 **LLM generability**、**debuggability** 和 **token-efficient optimization**。


---

## 4. 实验方法与实验结果

**实验设置**

- **评估目标**
  - TileBench 设计为一个受控基准，用于比较两类 **tile-based GPU programming models**：
    - **Triton 3.6.0**
    - **cuda-tile/cuTile 1.3.0**
  - 核心问题覆盖：
    - **RQ1：Programming-model Performance**
    - **RQ2：Autotuning and Performance Headroom**
    - **RQ3：Code-level Performance Diagnosis**
    - **RQ4：LLM Usability and Token Efficiency**

- **硬件与软件环境**

| 项目 | 配置 |
|---|---|
| GPU | **NVIDIA B200** |
| 显存 | **180 GB HBM3e** |
| 实测 HBM 带宽 | **6539.4 GB/s** |
| PyTorch | **2.10** |
| CUDA | **13.0** |
| Triton | **3.6.0** |
| cuda-tile/cuTile | **1.3.0** |
| Profiler | **Proton**，选定案例使用 **Nsight Compute** |
| LLM | **GPT-5.5**，**Claude Opus 4.7** |

- **B200 roofline 参数**
  - 论文使用 **B200.json** 中的 dtype-specific peak throughput 与实测带宽做 roofline 归一化。
  - FP32 使用 **TF32 Tensor Core dense ceiling**，因为 Triton 的 `tl.dot` 默认对 FP32 输入采用 TF32。

| dtype | Peak TFLOPS |
|---|---:|
| fp64 | 37 |
| fp32 / tf32 | 1100 |
| fp16 | 2250 |
| bf16 | 2250 |
| fp8_e4m3fn | 4500 |
| fp8_e5m2 | 4500 |
| int8 | 4500 |

![](b858acd1f160da5bef3c3c98baf9cd3354b5b2284a733c50e28c624929e2e8e4.jpg) *Figure 1: TileBench Overview.*

- **Benchmark 组成**
  - 总计 **45 个 operator**：
    - **26 个**来自 TritonBench
    - **19 个**来自 LeetGPU
  - 每个 operator 包含：
    - `impl_torch.py`：PyTorch reference
    - `impl_triton.py`：人工实现 Triton kernel
    - `impl_cutile.py`：人工实现 cuTile kernel
    - `config.yaml`：dtype、shape sweep、verify tolerance、FLOP/byte 公式、timing 配置
  - 每个 operator 按支持 dtype 执行 **20 个输入配置**。
  - PyTorch 作为：
    - **语义参考**
    - **正确性 oracle**
    - **软件性能 baseline**

| 类别 | 数量 | 代表 operator |
|---|---:|---|
| Point-wise | 12 | vector_add, relu, swiglu, weight_dequant |
| Reduction/Normalization | 11 | softmax, layernorm, argmax, moe_topk_gating |
| Matrix Multiplication/Attention | 8 | matmul, matmul_int8, flash_attention, linear_self_attention |
| Stencil/Convolution | 6 | 1d_conv, 3d_conv, gaussian_blur, jacobi_stencil_2d |
| Data Layout | 8 | matrix_copy, matrix_transpose, bitonic_sort, radix_sort |
| Total | 45 | - |

- **测量协议**
  - 使用 **Proton profiler**。
  - 启用 **CUDA graph capture-and-replay**。
  - 启用 **L2 cache flushing**。
  - 每个 case：
    - **20 次 warmup**
    - **100 次 timed runs**
  - 报告 **mean kernel latency**。
  - 聚合方式：
    - operator 内按 case 做 **arithmetic mean**
    - operator 间使用 **geometric mean**

- **性能指标**
  - **Latency**：核心原始指标。
  - **Speedup over PyTorch**：
    - `T_torch / T_backend`
  - **Achieved TFLOPS**：
    - `FLOPs / latency`
  - **Effective bandwidth**：
    - 由 `bytes_expr / latency` 得到。
  - **Arithmetic intensity**
    - `AI = FLOPs / HBM bytes`
  - **Roofline utilization**
    - 将 achieved TFLOPS 除以 dtype-specific roofline bound：
    - `min(Peak_TFLOPS_dtype, AI × Peak_BW)`

---

**主实验结果：Triton 与 cuTile 默认配置性能**

![](c46cba47f4d618a2e72eb29c059a5c2144d83ca4d63ed90b4f4c0c5a37a4b113.jpg)

- **总体结论**
  - **Triton 和 cuTile 均显著快于 PyTorch**。
  - **Triton 整体更强**，但 **cuTile 在少数 Tensor Core/TMA-friendly workloads 上具有优势**。
  - 性能差异高度依赖 workload，而不是某个 backend 绝对优越。

| 指标 | Triton | cuTile |
|---|---:|---:|
| Geomean speedup over PyTorch | **2.7×** | **2.2×** |
| Median speedup over PyTorch | **3.1×** | **2.7×** |
| 快于 PyTorch 的 operator 数 | **36/45** | **34/45** |
| 慢于 PyTorch 的 operator 数 | **9/45** | **11/45** |
| 快于对方的 operator 数 | **34/45** | **11/45** |

- **Triton 优势场景**
  - 更适合 **irregular access patterns**。
  - 更适合 **runtime-computed indices**。
  - 更适合 **masked loads / boundary-dependent control**。
  - 更适合 **streaming / bandwidth-bound kernels**。
  - 代表 operator：
    - **block_sparse_attention**
    - **flash_decode**
    - **linear_self_attention**
    - **weight_dequant**
    - sparse layout、low-bit packing、动态索引相关 kernel

- **cuTile 优势场景**
  - 更适合 **static tile abstraction**。
  - 更适合 **regular reusable tiles**。
  - 更适合 **Tensor Core computation**。
  - 更适合 **TMA-backed tile movement**。
  - 代表 operator：
    - **matmul_fp32_fp16_fp8**
    - **1d_conv**
    - **2d_conv**
    - dense GEMM、attention、部分 stencil/convolution

- **解释**
  - Triton 的 **pointer-level programming model** 对任意地址计算、mask、边界处理表达更自然。
  - cuTile 的 `ct.load`、`ct.mma`、`ct.store` 更贴近 Blackwell 原生 tile/TMA 路径，但前提是数据访问模式规则。
  - 当 operator 需要 `ct.gather`、`ct.where`、per-element index、mask selection 时，cuTile 的表达和 lowering 容易变重。

---

**按类别的 roofline utilization 分布**

![](56a81d119d5f0e2475b0ac4c118839dcfc5eb04af7067a25c7639e684180dcf7.jpg)

- **Speedup distribution**
  - 两个 backend 的分布高度重叠。
  - Triton 的 tail 超过 **100×**。
  - cuTile 的 tail 超过 **80×**。
  - 长尾主要来自 PyTorch reference 较弱的 operator。

![](fb9b61904f598daa037da72176e1ff88850595260aed8af1924d698112584fca.jpg)

- **按类别 roofline utilization**
  - **Point-wise** 类别 utilization 最高：
    - Triton mean R：**0.73**
    - cuTile mean R：**0.64**
  - **Reduction/Normalization** 和 **Data Layout** 居中：
    - mean R 大约在 **0.34–0.42**
  - **Stencil/Convolution** 和 **Matrix Multiplication/Attention** 最低：
    - mean R 不超过 **0.15**
  - 低 utilization 的原因包括：
    - tile reuse 不充分
    - 小 filter convolution 的数据复用效率有限
    - flash_decode、linear_self_attention 这类 attention 变体 memory/computation behavior 不规则

| 类别 | 性能特征 | 主要瓶颈 |
|---|---|---|
| Point-wise | roofline utilization 最高 | HBM bandwidth 接近饱和 |
| Reduction/Normalization | 中等，内部差异大 | reduction order、shared memory、occupancy |
| Data Layout | 中等到偏低 | irregular access、scatter/gather |
| Matrix Multiplication/Attention | 平均较低 | Tensor Core utilization、TMA overlap、layout |
| Stencil/Convolution | 平均较低 | boundary、virtual im2col、operand staging |

- **关键判断**
  - operator 内部结构带来的性能差异，经常大于 Triton 与 cuTile 的 backend 差异。
  - benchmark 的设计价值在于揭示 **operator-specific bottlenecks**，而不是简单给出 backend 排名。

---

**Autotuning 实验与性能余量**

![](95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg)

- **Autotuning 设置**
  - Triton autotuning 搜索：
    - `BLOCK_SIZE`
    - `num_warps`
    - `num_stages`
  - cuTile autotuning 搜索：
    - `TILE_SIZE`
    - `occupancy`
  - 两者并非一一对应参数，而是使用 **comparable tuning ranges**。
  - 默认配置不是随机弱 baseline，而是人工挑选的合理配置。

| 指标 | Triton | cuTile |
|---|---:|---:|
| Geomean autotune gain in R | **1.20×** | **1.15×** |
| Median autotune gain | **1.07×** | **1.04×** |
| Autotuned 后 R ≥ 80% 的 operator | **8/45** | **4/45** |
| 出现轻微 regression 的 operator | **5/45** | **8/45** |

- **Autotuning 的实际效果**
  - 能恢复部分 tile shape、parallelism、occupancy 选择带来的性能。
  - 对已经接近最优的手工默认配置，提升有限。
  - 少数 operator 存在明显提升：
    - Triton 在 **argmax** 上最高达到 **2.60×**
    - cuTile 在 **2d_conv** 上最高达到 **2.90×**

![](6e1f4d31e8492995f3941738cd3ec827bbdd3969b7866160a9a882e8d2f8f2f7.jpg)

- **性能余量分析**
  - Autotuning 后仍有大量 kernel 远低于 roofline。
  - 说明主要瓶颈并不总是 tile 参数。
  - 更深层因素包括：
    - **compiler lowering path**
    - **instruction selection**
    - **Tensor Core usage**
    - **TMA usage**
    - **shared-memory layout**
    - **register pressure**
    - **bank conflicts**
    - **operand staging**
    - **async pipeline overlap**

- **关键结论**
  - Autotuning 是必要但不充分的优化手段。
  - TileBench 的结果支持一个明确判断：
    - **tile-size search 只能优化表层参数**
    - **backend compiler lowering 与 memory behavior 决定更大的性能上限**

---

**消融实验：Triton Tensor Descriptor / TMA Ablation**

![](8280c07c994efbcaeb79af11fbfd5ac7be69c82490fbbab5d2e2b7956adea84c.jpg)

- **实验目的**
  - 检验 Triton 的 **tensor descriptor / TMA-style path** 是否能普遍替代 pointer-based load。
  - 对比 descriptor 版本相对于 pointer-based baseline 的 speedup。

- **主要观察**
  - Tensor descriptor 只在少数 tile-reuse-heavy kernels 上有效：
    - **flash_attention**
    - **matmul_int8**
    - **streamk_matmul**
  - 多数 one-pass operator 出现 regression：
    - pointwise
    - data layout
    - normalization
    - reduction

- **原因**
  - TMA 适合大块、规则、可复用的数据搬运。
  - 对 streaming kernel，TMA 可能引入额外 shared-memory staging。
  - 如果数据只使用一次，descriptor/TMA 的 setup 与 staging 成本可能超过收益。

- **实践规则**
  - 对 **tile reuse / compute overlap** 明显的 workload 使用 descriptor/TMA。
  - 对 **one-pass streaming / bandwidth-bound** workload 优先使用 plain pointer loads。
  - Triton 的 pointer model 在不规则场景更稳健，不应机械替换为 descriptor。

---

**性能差距案例诊断**

![](f9e643fecb6869d7899b6b7a50bbcaf63d4acc756e36ecba362971843a1ab893.jpg)

- **Top Triton–cuTile gap 的总体模式**
  - cuTile 胜出的案例多与 Blackwell 原生路径相关：
    - **TMA**
    - **Tensor Core**
    - **tcgen05**
    - **TMEM**
  - Triton 胜出的案例多与灵活地址表达相关：
    - arbitrary pointer tile
    - masked `tl.load`
    - runtime index
    - sparse / low-bit / boundary-heavy operators

- **2d_conv/fp32 案例**
  - 两个 backend 都使用 true FP32 path，而不是 Tensor Core。
  - 差异主要来自 **virtual-im2col operands materialization**。
  - cuTile 路径：
    - 使用 linearized per-element indices
    - 需要 gather、mask、select、staging
    - NCU 显示 `stall_long_sb` 多集中在 `ct.mma` 附近
    - SASS 存在大量 LDS/STL、本地/共享内存依赖
    - register file utilization 高，存在 register spilling
  - Triton 路径：
    - 直接构造 pointer tile
    - 将 boundary predicate 绑定到 `tl.load`
    - 避免先 materialize gathered tile 再做 select
  - 诊断结论：
    - 问题不是 convolution 本身，而是 cuTile 对不规则 operand tile 的表达成本更高。

- **streamk_matmul/bf16 案例**
  - cuTile 较慢的关键原因是 Tensor Core/TMEM pipeline 上的显式等待。
  - NCU 将主要 `stall_long_sb` 归因到 predicated `BRA`。
  - 进一步追踪发现：
    - branch predicate `P0` 由 `SYNCS.PHASECHK.TRANS64.TRYWAIT` 产生
    - 周围 SASS 包含 `STTM`、`FENCE.VIEW.ASYNC.T`、`SYNCS.ARRIVE`
  - 诊断结论：
    - 表面是 branch stall，实质是 async MMA/TMEM phase 未完成。
    - 优化方向是改善 MMA/TMEM pipelining，而不是简单调整 loop branch。

---

**Bank Conflict 诊断实验**

![](eecac1f3f846e36c1c4ebbc4ff6729c5eda72ef2563106488aac0344e4ef83b5.jpg)

- **实验设计**
  - 使用 NCU profile **204 个 kernels**。
  - 定义 normalized conflict score：
    - `C_LSU = 100 × B_LSU / W_LSU`
  - 其中：
    - `B_LSU` 是 shared-memory bank conflict events
    - `W_LSU` 是 LSU shared-memory wavefronts
  - 该指标用于衡量相对严重程度，不是精确 source-level speedup estimator。

- **Top-20 bank-conflict cases**
  - cuTile：**6 个**
  - Triton：**14 个**
  - 说明严重 bank conflict 并非某个 backend 独有。

- **Top-10 source-level 归因**
  - shared-memory stores：**4 个**
  - shared-memory loads 或 global-to-shared copies：**4 个**
  - MMA source lines：**2 个**
    - 实质通常是 operand staging，而不是 Tensor Core arithmetic 本身。

- **典型案例**
  - **streamk_matmul/Triton/fp32**
    - 最高 conflict score
    - 对应 B-tile load：
      - `b = tl.load(B_ptrs, ...)`
    - SASS 对应 `LDGSTS.E` global-to-shared copies
    - 瓶颈指向 compiler-selected staging pattern

![](f40e30cb49a6e88710c31c7a1255f22ad48c646bcd4612828aa878bf469b88af.jpg)

- **诊断分布**
  - NCU 结果被分为：
    - confirmed direct LSU conflict evidence
    - weaker source-derived evidence
    - divergence-confounded attribution
    - no direct LSU conflict evidence
  - 分布说明：
    - 不能仅凭单个 bank conflict counter 判断性能瓶颈。
    - branch divergence 可能干扰 attribution。

![](e055de22c98b3a5a7cd8160a32a8750a8815cc4733f2537d3189b67514330c8a.jpg)

![](98c4463baedfff6877d1206ce66cafa179292649e329275fea02456eac9f6331.jpg)

![](658915d4c4dc89964f3708c8d937a87d133acabceb646e3365b9e7470215528e.jpg)

![](98dcb8a46260be12cbf9061048965ee05ee7785ee754a725ee8a32a6184688d8.jpg)

- **关键结论**
  - 高 bank conflict 集中在少量 tile-heavy kernels。
  - Triton 与 cuTile 都可能出现严重 shared-memory layout 问题。
  - IPC gap 与 conflict severity 不强相关，因此不能用 IPC gap 替代 bank conflict analysis。
  - 真实优化应检查：
    - load/store tiling
    - operand layout
    - shared-memory swizzling
    - padding
    - MMA operand staging

---

**扩展案例：matmul_int8**

- **任务特征**
  - INT8 GEMM。
  - 权重矩阵 B 使用 **2-bit quantized storage**。
  - 每个 byte 包含 4 个 2-bit fields。
  - kernel 需要在 register 中 unpack B，再执行 MMA。

- **性能对比**

| 指标 | Triton | cuTile |
|---|---:|---:|
| Latency | **477.44 μs** | **346.88 μs** |
| cuTile speedup over Triton | - | **1.38×** |
| SASS instructions | **187.0M** | **78.8M** |
| 指令数差异 | - | **2.37× fewer** |
| Register/thread | **255** | **255** |
| Theoretical occupancy | **12.5%** | **12.5%** |

- **Triton 瓶颈**
  - 未使用 TMA 或 tcgen05。
  - lowering 到 legacy pipeline：
    - `cp.async`
    - `ldmatrix`
    - register-fragment
    - IMMA
  - 大量指令用于 operand staging，而非 compute：
    - PRMT：**60.3M**
    - LDS：**42.0M**
    - operand staging 相关指令约占 **72%**

- **cuTile 优势**
  - 使用 Blackwell native path：
    - `UTMALDG.2D`
    - `UTCIMMA`
    - TMEM accumulator
  - `LDSM` 降为 0。
  - LDS 从 **42.0M** 降到 **0.17M**。
  - PRMT 减少 **3.7×**。

- **cuTile 仍存在的问题**
  - unpacked-B scratch tile 有 shared-memory layout inefficiency：
    - **4.4-way store conflicts**
    - **77% conflicted shared-store wavefronts**
    - **74% uncoalesced MMA reads**
  - 优化方向：
    - swizzling
    - padding
    - unpacked intermediate tile layout 改善

- **诊断结论**
  - Triton 的主要问题是 **packed-low-bit-aware MMA lowering 不足**。
  - cuTile 的主要问题是 **intermediate tile shared-memory layout 不佳**。

---

**扩展案例：flash_attention**

- **任务配置**
  - FP16 non-GQA flash_attention。
  - batch size：**4**
  - heads：**32**
  - head_dim：**128**
  - sequence length：**20480**

| 指标 | Triton | cuTile |
|---|---:|---:|
| Latency | **22.77 ms** | **17.70 ms** |
| cuTile speedup over Triton | - | **1.29×** |
| cuTile compute throughput utilization | - | **43.2%** |

- **cuTile 优势**
  - 使用：
    - TMA
    - TC-Gen05 instructions
  - 对长序列 dense attention，regular tile movement 能获得收益。

- **cuTile 未达最优的原因**
  - **register utilization 极高**
    - live register count 达到 **185**
    - 可能来自不合理 register allocation
    - 也可能来自 TMEM/SMEM/register file 之间不必要的数据移动
  - **TMA load latency 未被 compute 完全覆盖**
    - warp stall 主要来自 mbarrier synchronization 的 `stall_long_scoreboard`
    - 说明 KV tensors 的 TMA load 与 compute overlap 不充分
  - 优化方向：
    - 改善 instruction scheduling
    - 降低 register pressure
    - 增强 TMA/computation pipelining

---

**LLM 代码生成实验**

![](336730bb607696cb1e402d8e5305e120d7e5b75eee5b3cbb91f30035d728ffb6.jpg)

- **实验设置**
  - 对同一 45 个 operator 进行 iterative LLM code-generation。
  - 每个 backend 独立跟踪：
    - Triton
    - cuTile
  - 每个 operator、backend 执行 **10 次 refinement iterations**。
  - 每次 iteration 只能提交单一配置。
  - 不允许 backend autotuning。
  - 最终结果选择 10 次中 **best verify-clean iteration**。

- **Prompt 输入**
  - TileBench framework guide
  - Triton/cuTile API reference
  - operator natural-language description
  - `config.yaml`
  - PyTorch reference implementation
  - previous trajectory feedback
  - worst-first roofline/performance summary
  - previous generated code
  - regression 时提供 best verify-clean code 作为 rebuild reference

- **反作弊机制**
  - 禁止委托给：
    - PyTorch
    - cuBLAS
    - cuDNN
    - reference implementation
  - 禁止输出缓存：
    - tensor identity
    - data pointer
    - version-dependent keys
  - 禁止导入 backend autotuners。
  - 检查方式：
    - static scans
    - two-pass anti-cache verification
    - rotating-input timing

| 组合 | 快于 PyTorch 的 operator 数 | Mean TokenCost@10 | TokenEfficiency@10 |
|---|---:|---:|---:|
| GPT-5.5 + Triton | **39/45** | **0.26M** | **20.9** |
| GPT-5.5 + cuTile | **38/45** | **0.32M** | **13.4** |
| Claude Opus 4.7 + Triton | **39/45** | **0.28M** | **17.1** |
| Claude Opus 4.7 + cuTile | **35/45** | **0.42M** | **8.3** |

- **主要结论**
  - Triton 对 LLM 更友好。
  - cuTile 需要更多 token，且 speedup per token 更低。
  - backend 差异大于 model 差异。
  - 原因包括：
    - Triton 生态更成熟
    - public code 更多
    - LLM pretraining corpus 中 Triton 样例更多
    - cuTile 更新，公开样例稀缺
    - cuTile API 与 tile abstraction 对 LLM 更陌生

![](e9e45d81864f1e99ab5bd4542c44ef7ee151bd4710d6eae1fd3a21ac42153072.jpg)

- **softmax refinement trajectory**
  - Triton：
    - GPT-5.5 和 Claude 在 iteration 0 即超过 PyTorch parity。
    - 验证通过后的 performance 更稳定。
    - GPT-5.5 在 iteration 8 达到 **2.64×**。
    - Claude 在 iteration 9 达到 **1.73×**。
  - cuTile：
    - GPT-5.5 iteration 0 出现 verification failure。
    - iteration 4 降到 **0.76×**，后续恢复到 **2.45×**。
    - Claude cuTile 10 次内未超过 parity，约 **0.43×**。
  - 说明：
    - Triton convergence 更快。
    - cuTile variance 更大。
    - cuTile verification failure 更多。
    - LLM 对 cuTile 的 prior knowledge 明显不足。

---

**实验可信度与局限**

- **可信度优势**
  - operator semantics 由 PyTorch reference 统一定义。
  - Triton 与 cuTile 实现手工编写并验证。
  - dtype、shape sweep、timing protocol、roofline formula 统一。
  - 默认配置与 autotuned 配置均报告。
  - 对大性能差距 case 使用 NCU、PTX/SASS 做深入诊断。
  - LLM track 与人工实现 main track 分离，避免混淆人工优化与生成式代码能力。

- **主要局限**
  - 仅评估 **单张 NVIDIA B200 GPU**。
  - 仅比较 **Triton** 与 **cuTile**，未覆盖 TileLang、ThunderKittens、Tilus 等系统。
  - roofline 使用 analytical FLOP/HBM-byte formulas。
    - 适合标准化比较。
    - 不能替代完整 NCU counter analysis。
  - autotuning 只搜索 backend 暴露的参数。
    - 不改变算法。
    - 不改变 lowering path。
    - 不搜索更复杂的 memory layout 或 pipeline scheduling。
  - cuTile 较新，LLM 生成结果可能受到训练语料不足影响，不完全代表长期可用性。

---

**核心判断**

- **Triton 的优势是稳健性**
  - 在 irregular、streaming、bandwidth-bound、runtime-indexed workloads 中更可靠。
  - pointer-level abstraction 更适合复杂地址表达和 mask 处理。

- **cuTile 的优势是硬件贴合度**
  - 在 regular tile reuse、Tensor Core、TMA、Blackwell native path 充分发挥时更强。
  - 但一旦 workload 偏离静态规则 tile，`ct.gather`、`ct.where`、staging、register pressure 可能迅速放大成本。

- **Autotuning 不能解决核心 compiler bottleneck**
  - 默认到 autotuned 的提升存在但有限。
  - 大量 kernel 仍远低于 roofline。
  - 更关键的是 backend lowering、instruction selection、TMA/TMEM pipeline、shared-memory layout。

- **LLM 代码生成更偏向 Triton**
  - Triton token cost 更低。
  - Triton token efficiency 更高。
  - cuTile 的可生成性瓶颈主要来自生态和语料成熟度，而不是单纯 refinement iteration 数不足。

---

