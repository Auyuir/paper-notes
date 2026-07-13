# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Tri Dao, Daniel Y. Fu, Stefano Ermon, et al.

**发表期刊/会议 (Journal/Conference)**: NeurIPS

**发表年份 (Publication Year)**: 2022

**研究机构 (Affiliations)**: Stanford University, University at Buffalo, SUNY

---

## 1. 摘要

**目的**

- 论文旨在解决 **Transformer Attention 在长序列场景下速度慢、显存占用高**的问题。
- 核心目标不是改变 Attention 的数学定义，而是在保持 **exact attention** 的前提下，通过 **IO-aware algorithm** 降低 GPU 内存层级之间的数据读写开销。
- 论文指出：
  - 传统高效 Attention 方法多关注降低 **FLOPs**，但实际 wall-clock speedup 往往有限。
  - 现代 GPU 中 **HBM 带宽**相对计算能力成为瓶颈，许多 Transformer 操作受限于 **memory access** 而非算力。
  - 标准 Attention 会显式物化 **N×N attention matrix**，导致大量 HBM 读写和二次方显存占用。

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

---

**方法**

- 论文提出 **FlashAttention**：
  - 一种 **IO-aware exact attention algorithm**。
  - 使用 **tiling** 将 Q、K、V 分块加载到 GPU on-chip **SRAM**。
  - 避免将完整的 **S=QKᵀ** 和 **P=softmax(S)** 写入 HBM。
  - 将 Attention 的矩阵乘、mask、softmax、dropout、输出矩阵乘等步骤融合到单个 CUDA kernel 中。

- 核心技术包括：
  - **Tiling**
    - 将 Q、K、V 按 block 切分。
    - 外层循环遍历 K/V block，内层循环遍历 Q block。
    - 每次只在 SRAM 中计算局部 Attention block。
  - **Online softmax**
    - 通过维护每一行的 **row max m** 和 **normalization factor ℓ**，实现分块 softmax 的精确合并。
    - 避免必须一次性访问完整 Attention row。
  - **Recomputation**
    - Forward pass 不保存完整 attention matrix。
    - Backward pass 只保存 **O、m、ℓ**，并在 SRAM 中重新计算局部 attention block。
    - 虽然增加部分 FLOPs，但大幅减少 HBM 访问，因此实际运行更快。
  - **Kernel fusion**
    - 将多个 memory-bound 操作融合，减少中间张量在 HBM 中的反复读写。
  - **Block-sparse FlashAttention**
    - 将 FlashAttention 扩展到 block-sparse attention。
    - 仅计算 sparsity mask 中非零 block。
    - 在近似 Attention 场景下进一步降低 IO 和运行时间。

- 理论复杂度对比：

| 方法 | 是否 exact | HBM access complexity | 额外显存复杂度 | 主要特点 |
|---|---:|---:|---:|---|
| Standard Attention | 是 | **Θ(Nd+N²)** | **O(N²)** | 显式存储 S 和 P |
| FlashAttention | 是 | **Θ(N²d²/M)** | **O(N)** | 不物化 N×N attention matrix |
| Block-sparse FlashAttention | 否 | **Θ(Nd+N²d²s/M)** | **O(N)** | 按 sparsity ratio s 降低 IO |

- 其中：
  - **N** 表示 sequence length。
  - **d** 表示 head dimension。
  - **M** 表示 SRAM size。
  - **s** 表示 block sparsity mask 中非零 block 的比例。

![](f2d4c917e30135292cb1109238fa9a694368cd1928cd6e9ebe3bada6de6be14e.jpg)

---

**结果**

- 运行速度：
  - FlashAttention 在 Attention 计算上最高实现 **7.6× speedup**。
  - 在常见 sequence length 128–2K 上，相比 PyTorch standard attention 最高约 **3× faster**。
  - 在 BERT-large、GPT-2、Long-Range Arena 等真实训练任务中带来显著 end-to-end speedup。

- 显存效率：
  - FlashAttention 的 attention memory footprint 随 sequence length **线性增长**。
  - 相比 exact attention baseline，显存占用最高降低约 **20×**。
  - 支持扩展到 **64K sequence length**。

![](816f269e79b71dde2a9df554b824ddbc83a97eec30f481dc2e340516d6ac0fcb.jpg)

- 主要训练加速结果：

| 任务 / 模型 | Baseline | FlashAttention 结果 | 关键收益 |
|---|---:|---:|---|
| BERT-large | Nvidia MLPerf 1.1: **20.0±1.5 min** | **17.4±1.4 min** | **15% faster** |
| GPT-2 small | HuggingFace: **9.5 days** | **2.7 days** | **3.5× speedup** |
| GPT-2 medium | HuggingFace: **21.0 days** | **6.9 days** | **3.0× speedup** |
| LRA | Standard Transformer | FlashAttention | **2.4× speedup** |
| LRA | Approximate Attention baselines | Block-sparse FlashAttention | **2.8× speedup** |

- GPT-2 训练结果：

| Model implementation | OpenWebText ppl | Training time | Speedup |
|---|---:|---:|---:|
| GPT-2 small - HuggingFace | 18.2 | 9.5 days | 1.0× |
| GPT-2 small - Megatron-LM | 18.2 | 4.7 days | 2.0× |
| GPT-2 small - FlashAttention | 18.2 | 2.7 days | **3.5×** |
| GPT-2 medium - HuggingFace | 14.2 | 21.0 days | 1.0× |
| GPT-2 medium - Megatron-LM | 14.3 | 11.5 days | 1.8× |
| GPT-2 medium - FlashAttention | 14.3 | 6.9 days | **3.0×** |

- 长上下文建模效果：
  - FlashAttention 允许 GPT-2 使用更长 context length。
  - GPT-2 small 使用 **4K context length** 时，仍比 Megatron-LM 的 **1K context length** 更快，同时 perplexity 更低。
  - OpenWebText 上 perplexity 从 **18.2** 降至 **17.5**，提升 **0.7 ppl**。

| Model implementation | Context length | OpenWebText ppl | Training time |
|---|---:|---:|---:|
| GPT-2 small - Megatron-LM | 1K | 18.2 | 4.7 days |
| GPT-2 small - FlashAttention | 1K | 18.2 | 2.7 days |
| GPT-2 small - FlashAttention | 2K | 17.6 | 3.0 days |
| GPT-2 small - FlashAttention | 4K | **17.5** | **3.6 days** |

- Long document classification：
  - 更长 sequence length 提升 MIMIC-III 和 ECtHR 表现。
  - MIMIC-III 从 length 512 的 **52.8** 提升到 length 16K 的 **57.1**。
  - ECtHR 从 length 512 的 **72.2** 提升到 length 8K 的 **80.7**。

| Dataset | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|---:|
| MIMIC-III | 52.8 | 50.7 | 51.7 | 54.6 | 56.4 | **57.1** |
| ECtHR | 72.2 | 74.3 | 77.1 | 78.6 | **80.7** | 79.2 |

- Path-X 和 Path-256：
  - FlashAttention 使 Transformer 首次在 **Path-X** 上达到高于随机的性能。
  - Block-sparse FlashAttention 进一步扩展到 **Path-256 / 64K sequence length**。

| Model | Path-X | Path-256 |
|---|---:|---:|
| Standard Transformer / Linformer / Performer / Reformer 等 | X | X |
| FlashAttention | **61.4** | X |
| Block-sparse FlashAttention | 56.0 | **63.1** |

---

**结论**

- **FlashAttention** 的核心贡献在于将 Attention 优化从单纯降低 FLOPs 转向降低 **HBM-SRAM IO cost**。
- 在不改变模型数学定义的情况下，FlashAttention 实现了：
  - **exact attention**。
  - **linear memory footprint**。
  - 显著更少的 HBM 访问。
  - 更快的真实训练速度。
  - 对长上下文任务更强的可扩展性。

- 论文的关键结论：
  - 对现代 GPU 而言，Attention 的主要瓶颈往往是 **memory access**，不是纯计算量。
  - 避免物化 **N×N attention matrix** 是提升速度和显存效率的关键。
  - **Recomputation** 虽然增加 FLOPs，但在 memory-bound 场景中可以提升 wall-clock performance。
  - **IO-aware design** 对深度学习系统优化具有普遍意义，可扩展到 sparse MLP、kernel methods、multi-GPU attention 等方向。

- 局限性：
  - 当前实现依赖手写 CUDA kernel。
  - 新 Attention 变体需要额外底层工程工作。
  - 不同 GPU 架构上的性能收益受 SRAM size、HBM bandwidth、kernel 实现细节影响。

- 总体评价：
  - FlashAttention 是一种兼具理论分析和系统工程价值的 Attention 实现范式。
  - 它证明了在 Transformer 优化中，**IO-awareness** 可以比单纯降低理论 FLOPs 更直接地转化为实际速度和显存收益。

---

## 2. 背景知识与核心贡献

**研究背景**

- **Transformer**已成为NLP、视觉等任务中的主流架构，但其核心模块**Self-Attention**在长序列场景下面临明显瓶颈：
  - 时间复杂度为**O(N²)**。
  - 显存复杂度为**O(N²)**。
  - 当序列长度**N**增大时，Attention矩阵**S=QKᵀ**和Softmax后的矩阵**P**都会达到**N×N**规模，导致训练和推理成本急剧上升。

- 现有大量**Approximate Attention**方法试图缓解该问题：
  - **Sparse Attention**：如Longformer、BigBird、Reformer。
  - **Low-rank Attention**：如Linformer。
  - **Kernel/Linear Attention**：如Performer、Linear Attention。
  - 混合方法：如Scatterbrain、Long-short Transformer。

- 这些方法通常关注降低**FLOPs**，将理论计算复杂度从二次降为线性或近线性，但实际问题是：
  - FLOPs减少并不必然带来**wall-clock speedup**。
  - 很多方法在真实GPU上没有显著加速，甚至由于实现复杂、访存不友好而变慢。
  - 近似方法还可能牺牲模型质量，尤其在需要精确全局依赖建模的任务中表现不稳定。

- 论文指出，现代GPU上的真正瓶颈往往不是算力，而是**IO / memory access**：
  - GPU计算速度增长快于内存带宽增长。
  - 许多Transformer操作属于**memory-bound**，运行时间主要受HBM读写限制。
  - Attention中的Softmax、mask、dropout等操作频繁读写巨大中间矩阵，是典型的访存瓶颈。

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

**研究动机**

- 论文的核心动机是重新审视Attention加速问题：
  - 不只优化**计算复杂度**。
  - 更要优化**IO complexity**，即GPU不同层级内存之间的数据搬运量。
  - 重点减少**HBM**与片上**SRAM**之间的读写。

- 标准Attention实现的问题在于会显式物化大规模中间矩阵：
  - 计算**S=QKᵀ**后，将**S∈Rᴺ×ᴺ**写入HBM。
  - 从HBM读取**S**，计算**P=softmax(S)**，再将**P∈Rᴺ×ᴺ**写入HBM。
  - 再读取**P**和**V**，计算输出**O=PV**。
  - 训练时还需要保存**P**等中间结果用于backward pass。

- 这种实现造成两个主要开销：
  - **显存占用高**：需要存储**N×N** Attention矩阵。
  - **访存量高**：反复读写**S**和**P**，导致实际速度慢。

- 论文提出的关键观察：
  - Attention的总FLOPs仍然是**O(N²d)**，但如果避免将**N×N**矩阵写入HBM，就能显著提升实际速度。
  - 在GPU上，重新计算部分中间结果可能比从HBM读取它们更快。
  - 因此，可以用**tiling**和**recomputation**交换一部分额外计算，换取大幅减少HBM访问。

**核心方法**

- 论文提出**FlashAttention**：
  - 一种**IO-aware**的精确Attention算法。
  - 不改变Attention数学定义。
  - 不近似Softmax。
  - 不牺牲模型质量。
  - 通过优化GPU内存访问实现更快、更省显存的Attention。

- FlashAttention的关键技术包括：
  - **Tiling**
    - 将**Q、K、V**切分为块。
    - 将小块加载到高速片上**SRAM**。
    - 在SRAM中完成局部矩阵乘、Softmax统计量更新和输出累积。
    - 避免在HBM中物化完整**N×N** Attention矩阵。
  - **Online Softmax / Softmax Decomposition**
    - 对每一行维护Softmax所需的统计量：
      - 行最大值**m**。
      - 归一化因子**ℓ**。
    - 分块处理时动态更新这些统计量，保证结果与完整Softmax完全一致。
  - **Recomputation**
    - forward pass不保存完整Attention矩阵**P**。
    - backward pass中利用保存的**O、m、ℓ**重新在SRAM中计算需要的Attention块。
    - 避免读取巨大中间矩阵，降低HBM访问。
  - **Kernel Fusion**
    - 将矩阵乘、mask、Softmax、dropout、再矩阵乘融合到单个CUDA kernel中。
    - 避免多个kernel之间反复读写HBM。

**核心贡献**

- **提出IO-aware Attention设计原则**
  - 论文强调Attention性能瓶颈不仅来自FLOPs，更来自**HBM memory access**。
  - 将经典的**IO complexity**分析引入Transformer Attention优化。
  - 证明在现代GPU上，减少数据搬运比单纯减少算术操作更关键。

- **提出FlashAttention：精确且内存高效的Attention算法**
  - 计算结果与标准Attention完全一致。
  - 不需要近似。
  - 不显式存储**N×N** Attention矩阵。
  - 额外显存从**O(N²)**降为**O(N)**。
  - FLOPs仍为**O(N²d)**，但实际运行更快。

- **给出IO复杂度理论分析**
  - 标准Attention的HBM访问复杂度为：
    - **Θ(Nd+N²)**
  - FlashAttention的HBM访问复杂度为：
    - **Θ(N²d²/M)**
  - 其中：
    - **N**为序列长度。
    - **d**为head dimension。
    - **M**为SRAM大小。
  - 当典型**d=64/128**且SRAM足够容纳合理tile时，FlashAttention显著减少HBM访问。
  - 论文还给出下界，说明在一定SRAM范围内，FlashAttention的IO复杂度具有近似最优性。

![](f2d4c917e30135292cb1109238fa9a694368cd1928cd6e9ebe3bada6de6be14e.jpg)

- **扩展到Block-sparse FlashAttention**
  - 在FlashAttention基础上支持块稀疏Attention。
  - 跳过稀疏mask中的零块，只计算非零块。
  - IO复杂度进一步降低为：
    - **Θ(Nd+N²d²s/M)**
  - 其中**s**为非零块比例。
  - 在长序列任务中可扩展到**64K**序列长度。
  - 实验中比FlashAttention进一步快**2–4×**。

- **显著提升真实训练速度**
  - 在多个模型和任务上获得端到端加速。
  - 重点是实际wall-clock时间，而非理论FLOPs。

| 场景 | 对比对象 | 结果 |
|---|---:|---:|
| **BERT-large** | MLPerf 1.1 Nvidia记录 | **15%**端到端加速 |
| **GPT-2 small** | HuggingFace | **3.5×**加速 |
| **GPT-2 medium** | HuggingFace | **3.0×**加速 |
| **GPT-2 medium** | Megatron-LM | **1.7–1.8×**加速 |
| **Long Range Arena** | 标准Attention | **2.4×**加速 |
| **Block-sparse FlashAttention on LRA** | 标准Attention | **2.8×**加速 |

- **支持更长上下文并提升模型质量**
  - FlashAttention使标准Transformer可以在更长序列上训练。
  - 长上下文带来质量提升，而不是仅仅加速。

| 任务 | 方法 / 设置 | 结果 |
|---|---|---:|
| **GPT-2 OpenWebText** | context length从1K扩展到4K | perplexity提升约**0.7** |
| **MIMIC-III长文档分类** | sequence length 512 → 16K | micro-F1从**52.8**提升到**57.1** |
| **ECtHR长文档分类** | sequence length 512 → 8K | micro-F1从**72.2**提升到**80.7** |
| **Path-X** | FlashAttention，sequence length 16K | **61.4%**accuracy |
| **Path-256** | Block-sparse FlashAttention，sequence length 64K | **63.1%**accuracy |

- **实现层面贡献**
  - 使用CUDA实现细粒度内存控制。
  - 支持mask、dropout、backward pass。
  - 相比Apex FMHA：
    - 支持更长序列。
    - 支持更多head dimension。
    - 支持更广GPU类型。
    - 显存占用明显更低。

**与现有工作的关键区别**

| 维度 | 标准Attention | Approximate Attention | FlashAttention |
|---|---|---|---|
| 是否精确 | **是** | 通常**否** | **是** |
| 是否物化N×N矩阵 | **是** | 视方法而定 | **否** |
| 主要优化目标 | 简单实现 / 通用矩阵运算 | 降低FLOPs | 降低**HBM访问** |
| 显存复杂度 | **O(N²)** | 通常低于O(N²) | **O(N)**额外显存 |
| 质量风险 | 无近似误差 | 可能有近似误差 | 无近似误差 |
| 实际速度 | 长序列慢 | 不一定快 | 显著加速 |

**核心结论**

- FlashAttention的核心价值不在于改变Attention公式，而在于改变Attention的**计算调度方式**。
- 它通过**IO-aware tiling**避免读写巨大Attention矩阵，使标准精确Attention在GPU上变得更快、更省显存。
- 论文的重要启示是：
  - 深度学习系统优化不能只看FLOPs。
  - 对于现代GPU，**memory hierarchy**和**data movement**往往决定真实性能。
  - 通过算法与硬件特性协同设计，可以在不牺牲模型质量的前提下获得显著加速。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

**总体框架**

- 本文的整体技术架构围绕一个核心目标展开：在保持 **exact attention** 数学结果不变的前提下，尽量减少 **GPU HBM** 与 **on-chip SRAM** 之间的数据搬运。
- 方案分成三层：
  - **算法层**：用 **tiling** 把原本一次性计算的 `QK^T` 拆成块级计算。
  - **内存层**：只在 SRAM 中暂存当前需要的 `Q/K/V` 子块和少量统计量，避免显式 materialize 巨大的 `N×N` attention matrix。
  - **实现层**：把矩阵乘、softmax、mask、dropout、再乘 `V` 融合进一个 **CUDA kernel**，减少中间结果读写。

**核心数据流**

- 输入是 **Q, K, V ∈ R^{N×d}**。
- 前向过程中：
  - 按块加载 **K/V** 到 SRAM。
  - 对每个 **Q** 块，计算局部 `S = QK^T`。
  - 在块内做 **row-wise softmax**，同时维护两类统计量：
    - **m**：每行的最大值，用于数值稳定性。
    - **ℓ**：每行的归一化和，用于跨块累积 softmax。
  - 用局部概率块与 `V` 相乘，并把输出 **O** 逐步更新回 HBM。
- 反向过程中：
  - 不保存完整的 **attention matrix P**。
  - 只保留前向输出 **O**、统计量 **m/ℓ** 和 **RNG state**。
  - 通过 **recomputation** 在 SRAM 中重建需要的局部注意力块，再计算 **dQ/dK/dV**。

**前向架构模块**

- **Block partitioning**
  - 将 **Q** 分成 `T_r` 个行块，将 **K/V** 分成 `T_c` 个列块。
  - 块大小由 SRAM 容量决定，使每次只加载能放入片上内存的数据。
- **Streaming softmax**
  - 利用 softmax 的可分解性质，逐块更新 `m` 和 `ℓ`。
  - 这样不需要先算完整的 `N×N` 矩阵再做 softmax。
- **Output accumulation**
  - 每处理一个 `K/V` 块，就对对应的输出块 **O_i** 做增量更新。
  - 最终得到与标准 attention 完全一致的结果。

**反向架构模块**

- **Recompute-based backward**
  - 反向不依赖 `P` 的完整缓存，而是在块内重算 `S` 和 `P`。
- **Saved state**
  - 保存 **O、m、ℓ、RNG state**，支持 dropout 的确定性重建。
- **Gradient flow**
  - 在块级别直接计算：
    - **dV**：由 `P^T dO` 得到
    - **dP/dS**：通过 softmax Jacobian 推导
    - **dQ/dK**：由局部 `dS` 与 `Q/K` 乘积得到
- 反向和前向使用同一套分块策略，因此内存访问模式一致，便于融合实现。

**IO-aware 设计思想**

- 不是单纯减少 FLOPs，而是以 **HBM access** 为优化目标。
- 架构上把 attention 重新设计为：
  - **HBM** 只负责大张量的顺序流入流出；
  - **SRAM** 负责局部矩阵乘和 softmax；
  - **统计量** 负责连接各个块的局部结果。
- 这使得实际速度提升来自 **更少的数据搬运**，而不只是理论计算量下降。

**扩展架构：Block-Sparse FlashAttention**

- 在基础 FlashAttention 之上再加一层 **block-sparsity mask**。
- 只计算非零块：
  - 跳过无效的 `QK^T` 子块
  - 跳过对应的 `P V` 子块
- 架构仍然保持与 FlashAttention 一致的块式流水线，只是计算图更稀疏。
- 结果是：
  - 更低的 IO complexity
  - 更长序列的可扩展性
  - 在部分任务上优于许多 approximate attention 方法

**整体模块关系**

| 模块 | 作用 | 关键机制 |
|---|---|---|
| **Tiling Engine** | 分块处理 `Q/K/V` | block-wise matrix multiply |
| **Streaming Softmax** | 保持 exact softmax | `m` / `ℓ` 增量更新 |
| **Output Accumulator** | 累积最终 `O` | 块级写回 HBM |
| **Recomputation Unit** | 反向重建局部注意力 | 省去 `P` 存储 |
| **CUDA Fusion** | 减少 kernel launch 和中间写回 | 单 kernel 融合 |
| **Block-Sparse Adapter** | 支持稀疏注意力 | 跳过零块 |

**一句话概括**

- 这篇论文的整体架构可以概括为：把传统 attention 从“先算完再存起来”改造成“**按块流式计算、按需重算、全程 IO-aware**”的 GPU 原生流水线，并进一步扩展到 **block-sparse exact/approximate attention**。

### 1. IO-aware Tiling for Exact Attention

**核心观点**

- **IO-aware Tiling** 的目标不是减少 attention 的数学结果，而是减少 **GPU HBM** 和 **on-chip SRAM** 之间的数据搬运。
- FlashAttention 计算的是 **exact attention**，即输出仍然是 $softmax(QK^T)V$，没有近似误差。
- 关键做法是：
  - 将 **Q、K、V** 切成 block；
  - 把 block 从慢速 **HBM** 搬到快速 **SRAM**；
  - 在 SRAM 内完成局部矩阵乘、softmax 归一化和输出累加；
  - **不在 HBM 中 materialize N×N attention matrix**。
- 这使得 attention 的瓶颈从“存中间大矩阵”转为“分块流式计算”，显著降低内存访问次数。

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

**实现原理**

- 标准 attention 的流程是：
  - `S = QK^T`
  - `P = softmax(S)`
  - `O = PV`
- 问题在于：
  - `S` 和 `P` 都是 `N×N`；
  - 需要写回 HBM；
  - 后续 softmax 和 `PV` 又要重新从 HBM 读出；
  - 造成大量 **HBM read/write**。
- FlashAttention 的核心思想是：
  - 对每个 query block `Q_i`，按顺序扫描 key/value block `K_j, V_j`；
  - 在 SRAM 中只计算当前块对应的局部 attention；
  - 用 **online softmax** 累积全局归一化量，而不是先算完整 `S` 再整体 softmax。
- 在线 softmax 的关键统计量是：
  - `m_i`：当前行的累计最大值
  - `l_i`：当前行的累计归一化分母
- 当新的 block 到来时，利用数值稳定的重标定公式更新：
  - `m_new = max(m_old, m_block)`
  - `l_new = exp(m_old - m_new) * l_old + exp(m_block - m_new) * l_block`
  - `O_new` 也同步按相同尺度重加权更新
- 这样可以保证：
  - 结果与标准 softmax 完全一致；
  - 但不需要保存 `N×N` 中间矩阵。

**算法流程**

- 输入：
  - `Q ∈ R^(N×d)`
  - `K ∈ R^(N×d)`
  - `V ∈ R^(N×d)`
- 输出：
  - `O ∈ R^(N×d)`
- 附加保存：
  - `l ∈ R^N`
  - `m ∈ R^N`
- 处理方式：
  - 将 `Q` 切成 `T_r = ceil(N / B_r)` 个 block；
  - 将 `K、V` 切成 `T_c = ceil(N / B_c)` 个 block；
  - 外层遍历 `K_j, V_j`；
  - 内层遍历 `Q_i`，逐块更新输出 `O_i`。
- 每个 `(i, j)` block 对执行：
  - 计算局部分数 `S_ij = Q_i K_j^T`
  - 做 rowmax 得到 `m~_ij`
  - 计算 `exp(S_ij - m~_ij)` 得到局部未归一化概率 `P~_ij`
  - 求行和 `l~_ij`
  - 更新全局 `m_i` 与 `l_i`
  - 用重标定后的系数更新 `O_i`
- 该过程本质上是：
  - 用多个小块的局部 softmax 逐步拼出全局 softmax；
  - 每一步都保持数值稳定。

**参数设置**

| 参数 | 含义 | 作用 |
|---|---|---|
| `N` | sequence length | 决定 attention 矩阵规模 |
| `d` | head dimension | 决定每个 token 的特征维度 |
| `M` | SRAM size | 决定 block 大小上限 |
| `B_c` | K/V block size | 控制一次载入多少 key/value |
| `B_r` | Q/O block size | 控制一次载入多少 query/output |
| `m` | row-wise max | 用于 softmax 数值稳定 |
| `l` | row-wise normalization sum | 用于恢复全局 softmax |
| `τ` | softmax scaling | 通常是 `1/√d` |
| `p_drop` | dropout probability | 可选，用于训练正则化 |

- 论文中的典型块大小设置为：
  - `B_c = ceil(M / 4d)`
  - `B_r = min(ceil(M / 4d), d)`
- 这组设置的目的：
  - 让 `Q_i`、`K_j`、`V_j`、`O_i`、局部 `S_ij` 都能放进 SRAM；
  - 在可行的 SRAM 约束下尽量增大 block，减少 HBM 往返次数。
- 设计约束是：
  - `B_c * d = O(M)`
  - `B_r * d = O(M)`
  - `B_r * B_c = O(M)`
- 这意味着：
  - block 太小会增加 loop 次数；
  - block 太大又放不进 SRAM；
  - FlashAttention 实际上是在做一个硬件约束下的最优折中。

**输入输出关系**

- 输入 `Q、K、V`：
  - 都保留在 HBM 中；
  - 进入 kernel 后按 block 搬到 SRAM。
- 中间输出 `O`：
  - 不是一次性算完；
  - 而是对每个 `Q_i` 进行多轮 block 累积更新。
- 统计量 `m、l`：
  - 是 forward 的辅助状态；
  - 用来在 backward 中重建 softmax 概率；
  - 避免保存整个 `P` 或 `S`。
- 最终输出 `O`：
  - 仍然是标准 attention 的精确结果；
  - 但写回 HBM 的只有 `O`、`m`、`l`，没有 `N×N` 中间矩阵。

**为什么它能快**

- GPU 上很多时候不是算力不够，而是 **HBM 带宽不够**。
- 标准 attention 的主要成本在：
  - 写 `S`
  - 读 `S`
  - 写 `P`
  - 读 `P`
- FlashAttention 的优化点在于：
  - `S` 和 `P` 从未落到 HBM；
  - 只在 SRAM 中短暂存在；
  - 中间结果边算边消费。
- 结果是：
  - **HBM accesses** 显著下降；
  - 即使 FLOPs 略有增加，整体 wall-clock 仍然更快。
- 论文给出的结论是：
  - 标准 attention 的 HBM 访问量约为 `Θ(Nd + N^2)`
  - FlashAttention 的 HBM 访问量约为 `Θ(N^2 d^2 / M)`
- 在常见设置下，`d^2 << M`，所以速度提升明显。

**与标准 attention 的差异**

| 项目 | 标准 attention | FlashAttention |
|---|---|---|
| 中间矩阵 `S/P` | 写入 HBM | 仅在 SRAM 中临时计算 |
| 内存复杂度 | `O(N^2)` | `O(N)` 额外内存 |
| 数据搬运 | 多次读写大矩阵 | 分块流式读写 |
| 数值稳定性 | 直接 softmax | online softmax |
| backward | 依赖保存的 `P` | 只保存 `m、l` 并重算 |
| 结果 | exact | exact |

**在整体系统中的作用**

- FlashAttention 不是简单的 attention 小优化，而是 Transformer 长上下文能力的基础组件。
- 它的作用体现在：
  - 降低训练显存占用；
  - 提升训练吞吐；
  - 让更长的 sequence length 成为可能；
  - 支持后续的 block-sparse FlashAttention；
  - 让 Path-X、Path-256 这类超长序列任务可训练。
- 从系统视角看，它把 attention 从“显存受限算子”变成“可高效流式执行的算子”。

**关键实现细节**

- **Kernel fusion**
  - 将 matmul、softmax、mask、dropout、matmul(V) 融合到一个 CUDA kernel 中。
  - 减少 kernel launch 和 HBM 往返。
- **Recomputation**
  - backward 不保存 `N×N` attention matrix；
  - 通过 `m、l` 和原始 `Q、K、V` 在 SRAM 中重算局部 attention；
  - 以少量额外 FLOPs 换取巨大内存节省。
- **在线归一化**
  - 通过 `m、l` 维护全局 softmax；
  - 解决“不能一次看到全部输入”的问题；
  - 保证 exactness 和稳定性同时成立。

**一句话总结**

- **IO-aware Tiling for Exact Attention** 的本质是：用 **block-wise streaming + online softmax** 替代 `N×N` 中间矩阵 materialization，在不改变 attention 数学定义的前提下，把瓶颈从 **HBM bandwidth** 转移到更高效的 **SRAM 计算**，从而同时实现 **exactness、低显存和更快速度**。

### 2. Online Softmax Normalization

**核心定位**

- **Online Softmax Normalization**是FlashAttention能够在不物化完整$N \times N$ Attention矩阵的前提下，仍然计算**精确Softmax Attention**的关键机制。
- 标准Attention需要一次性得到整行分数：
  - $S = QK^\top$
  - $P = \operatorname{softmax}(S)$
  - $O = PV$
- 问题在于：
  - 每一行Softmax的归一化分母依赖该行的**所有Key位置**。
  - 如果按Block计算，只看到局部$S_{ij}$，看不到完整行，不能直接做标准Softmax。
- Online Softmax的解决方式：
  - 为每一行维护两个统计量：
    - **$m$**：当前已处理Block中的行最大值，用于数值稳定。
    - **$\ell$**：当前已处理Block中的指数和，即Softmax归一化因子。
  - 每处理一个新的Key/Value Block，就增量更新$m$、$\ell$和输出$O$。
  - 最终结果与一次性计算完整$\operatorname{softmax}(QK^\top)V$**数学等价**。

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

---

**标准Softmax的数值稳定形式**

- 对一个向量$x \in \mathbb{R}^{B}$，标准Softmax通常不会直接计算$e^{x_i}$，而是使用最大值平移：
  - $m(x)=\max_i x_i$
  - $f(x)_i=e^{x_i-m(x)}$
  - $\ell(x)=\sum_i e^{x_i-m(x)}$
  - $\operatorname{softmax}(x)_i=\frac{e^{x_i-m(x)}}{\ell(x)}$
- 这样做的原因：
  - 避免$x_i$较大时$e^{x_i}$溢出。
  - 避免$x_i$较小时指数下溢过快。
  - 保证Softmax在FP16/BF16等低精度训练中更稳定。
- FlashAttention沿用这一思想，但将其扩展为**分块增量版本**。

---

**Online Softmax的数学原理**

- 假设一行Attention分数$x$被分成两个Block：
  - $x = [x^{(1)}, x^{(2)}]$
- 对每个Block分别计算局部统计量：
  - $m_1=\max(x^{(1)})$
  - $\ell_1=\sum e^{x^{(1)}-m_1}$
  - $m_2=\max(x^{(2)})$
  - $\ell_2=\sum e^{x^{(2)}-m_2}$
- 合并两个Block时，全局最大值为：
  - $m=\max(m_1,m_2)$
- 全局归一化因子不能简单相加，因为两个Block使用了不同的最大值平移。
- 需要把两个Block的指数和重新缩放到同一个参考最大值$m$：
  - $\ell=e^{m_1-m}\ell_1+e^{m_2-m}\ell_2$
- 该公式是Online Softmax的核心：
  - 每个Block可以独立计算局部Softmax统计量。
  - 合并时只需保留$m$和$\ell$。
  - 不需要保存完整Attention行。

---

**从两块扩展到多块**

- 对第$i$行Query，FlashAttention按Key/Value Block逐步处理：
  - 当前已处理Block的统计量：
    - $m_i$
    - $\ell_i$
    - $O_i$
  - 新Block产生的局部统计量：
    - $\tilde{m}_{ij}$
    - $\tilde{\ell}_{ij}$
    - $\tilde{P}_{ij}V_j$
- 新的最大值：
  - $m_i^{new}=\max(m_i,\tilde{m}_{ij})$
- 新的归一化因子：
  - $\ell_i^{new}=e^{m_i-m_i^{new}}\ell_i+e^{\tilde{m}_{ij}-m_i^{new}}\tilde{\ell}_{ij}$
- 新的输出也需要同步重标定：
  - $O_i^{new}=\frac{e^{m_i-m_i^{new}}\ell_i O_i+e^{\tilde{m}_{ij}-m_i^{new}}\tilde{P}_{ij}V_j}{\ell_i^{new}}$
- 关键点：
  - 旧输出$O_i$原本是基于旧归一化因子$\ell_i$得到的。
  - 新Block加入后，Softmax分母变化，旧输出贡献必须重新缩放。
  - 该缩放保证最终输出等价于一次性处理所有Key。

---

**核心变量含义**

| 符号 | 维度 | 含义 | 存储位置 | 作用 |
|---|---:|---|---|---|
| $Q_i$ | $B_r \times d$ | 当前Query Block | SRAM | 与当前Key Block计算局部分数 |
| $K_j$ | $B_c \times d$ | 当前Key Block | SRAM | 参与$Q_iK_j^\top$ |
| $V_j$ | $B_c \times d$ | 当前Value Block | SRAM | 参与局部输出累积 |
| $S_{ij}$ | $B_r \times B_c$ | 局部Attention Score | SRAM | 不写入HBM |
| $\tilde{m}_{ij}$ | $B_r$ | 当前Block每行最大值 | SRAM | 更新全局$m_i$ |
| $\tilde{\ell}_{ij}$ | $B_r$ | 当前Block每行指数和 | SRAM | 更新全局$\ell_i$ |
| $m_i$ | $B_r$ | 已处理Block的每行最大值 | HBM/SRAM | 数值稳定统计量 |
| $\ell_i$ | $B_r$ | 已处理Block的每行归一化因子 | HBM/SRAM | Softmax分母 |
| $O_i$ | $B_r \times d$ | 当前累计输出 | HBM/SRAM | 最终Attention输出 |

---

**算法流程**

- 输入：
  - $Q,K,V \in \mathbb{R}^{N \times d}$
  - Sequence length：**$N$**
  - Head dimension：**$d$**
  - SRAM大小：**$M$**
- 初始化：
  - $O = 0$
  - $\ell = 0$
  - $m = -\infty$
- 分块：
  - 将$Q$划分为$T_r$个Block，每个大小为$B_r \times d$。
  - 将$K,V$划分为$T_c$个Block，每个大小为$B_c \times d$。
- 外层循环遍历Key/Value Block：
  - 加载$K_j,V_j$到SRAM。
- 内层循环遍历Query Block：
  - 加载$Q_i,O_i,\ell_i,m_i$到SRAM。
  - 计算局部分数：
    - $S_{ij}=Q_iK_j^\top$
  - 计算局部最大值：
    - $\tilde{m}_{ij}=\operatorname{rowmax}(S_{ij})$
  - 计算局部指数矩阵：
    - $\tilde{P}_{ij}=e^{S_{ij}-\tilde{m}_{ij}}$
  - 计算局部指数和：
    - $\tilde{\ell}_{ij}=\operatorname{rowsum}(\tilde{P}_{ij})$
  - 更新全局最大值：
    - $m_i^{new}=\max(m_i,\tilde{m}_{ij})$
  - 更新全局归一化因子：
    - $\ell_i^{new}=e^{m_i-m_i^{new}}\ell_i+e^{\tilde{m}_{ij}-m_i^{new}}\tilde{\ell}_{ij}$
  - 更新输出：
    - $O_i^{new}=\frac{e^{m_i-m_i^{new}}\ell_iO_i+e^{\tilde{m}_{ij}-m_i^{new}}\tilde{P}_{ij}V_j}{\ell_i^{new}}$
  - 将更新后的$O_i,\ell_i,m_i$写回HBM。
- 输出：
  - $O=\operatorname{softmax}(QK^\top)V$

---

**参数设置**

- FlashAttention根据SRAM大小$M$设置Block大小：
  - $B_c=\left\lceil \frac{M}{4d}\right\rceil$
  - $B_r=\min\left(\left\lceil \frac{M}{4d}\right\rceil,d\right)$
- 该设置背后的约束：
  - $K_j$和$V_j$需要同时放入SRAM。
  - $Q_i$和$O_i$需要放入SRAM。
  - 局部分数矩阵$S_{ij}$也需要放入SRAM。
- 需要满足：
  - $B_c d = O(M)$
  - $B_r d = O(M)$
  - $B_rB_c = O(M)$
- 设计目标：
  - 尽可能增大$B_c$，减少遍历$Q$的次数。
  - 保证局部矩阵计算和Softmax统计量都在SRAM内完成。
  - 避免将$S$和$P$写入HBM。

---

**输入输出关系**

- 输入：
  - **$Q$**：Query矩阵，形状$N \times d$。
  - **$K$**：Key矩阵，形状$N \times d$。
  - **$V$**：Value矩阵，形状$N \times d$。
- 中间逻辑输出：
  - **$m$**：每个Query位置对应的Softmax行最大值，形状$N$。
  - **$\ell$**：每个Query位置对应的Softmax归一化因子，形状$N$。
- 最终输出：
  - **$O$**：Attention输出，形状$N \times d$。
- 关系：
  - FlashAttention不会输出完整$S$或$P$。
  - $S_{ij}$和$\tilde{P}_{ij}$只在SRAM中临时存在。
  - $m$和$\ell$作为Forward Pass的必要统计量，会保留给Backward Pass使用。

---

**在FlashAttention整体中的作用**

- **避免物化Attention矩阵**
  - 标准Attention会将$S=QK^\top$和$P=\operatorname{softmax}(S)$写入HBM。
  - Online Softmax让FlashAttention只保存$O,m,\ell$。
  - 额外显存从$O(N^2)$降低到$O(N)$。
- **支持Tiling**
  - Softmax本身跨整行耦合，天然不利于分块。
  - Online Softmax打破了这一限制，使每个Block可以独立处理并正确合并。
- **保证精确性**
  - FlashAttention不是近似Attention。
  - 其输出与标准Attention数学等价。
  - 差异主要来自浮点运算顺序，而不是算法近似。
- **保证数值稳定**
  - 每次合并都维护当前全局最大值$m$。
  - 指数项始终以当前最大值为参考缩放。
  - 避免直接计算大指数。
- **降低HBM访问**
  - 大型中间矩阵不写入HBM。
  - 主要数据搬运变为Block级别的$Q,K,V,O,m,\ell$读写。
  - Attention计算更多发生在SRAM中。

---

**与标准Attention的关键差异**

| 维度 | 标准Attention | FlashAttention + Online Softmax |
|---|---|---|
| Softmax计算方式 | 对完整$N \times N$矩阵逐行计算 | 按Block增量计算 |
| 是否保存$S$ | 保存到HBM | 不保存 |
| 是否保存$P$ | 保存到HBM | 不保存 |
| 额外显存 | $O(N^2)$ | $O(N)$ |
| 输出精度 | 精确Attention | 精确Attention |
| 数值稳定 | 使用max-shift Softmax | 使用Online max-shift Softmax |
| 主要瓶颈 | HBM读写 | 更接近计算和SRAM访问 |

---

**Backward Pass中的作用**

- Forward Pass保存：
  - $O$
  - $m$
  - $\ell$
  - Dropout随机数状态，若启用Dropout
- Backward Pass不读取完整$P$：
  - 根据$Q_i,K_j,m_i,\ell_i$在SRAM中重新计算局部$P_{ij}$。
- 局部Softmax概率重构：
  - $P_{ij}=\operatorname{diag}(\ell_i)^{-1}\exp(S_{ij}-m_i)$
- 关键收益：
  - 避免保存Forward中的$N \times N$ Attention矩阵。
  - Backward虽然增加重计算FLOPs，但显著减少HBM访问。
  - 在GPU上通常更快，因为Attention常是**memory-bound**而非纯**compute-bound**。

---

**为什么只保存$m$和$\ell$就足够**

- 对任意Block的Attention分数$S_{ij}$：
  - Backward或后续合并时可以通过$Q_iK_j^\top$重新得到。
- 要恢复对应的Softmax概率，需要知道完整行的稳定归一化信息：
  - 行最大值$m_i$
  - 行指数和$\ell_i$
- 因此：
  - $P_{ij}=\frac{e^{S_{ij}-m_i}}{\ell_i}$
- 保存$m,\ell$即可将任意局部Block转化为与完整Softmax一致的概率块。
- 不需要保存完整$P$。

---

**数值稳定性的细节**

- 如果新Block的最大值小于旧最大值：
  - $m_i^{new}=m_i$
  - 旧统计量基本保持主导。
  - 新Block贡献乘以$e^{\tilde{m}_{ij}-m_i}$，数值较小但稳定。
- 如果新Block的最大值大于旧最大值：
  - $m_i^{new}=\tilde{m}_{ij}$
  - 旧统计量需要乘以$e^{m_i-\tilde{m}_{ij}}$重新缩放。
  - 避免旧指数和在新参考系下失真。
- 这种机制保证：
  - 任意Block处理顺序下，统计量都能被校准到当前全局最大值。
  - 不需要对完整行重新扫描即可得到正确归一化因子。

---

**简单示例**

- 假设一行分数分为两个Block：
  - $x^{(1)}=[1,2]$
  - $x^{(2)}=[3,0]$
- 第一个Block：
  - $m_1=2$
  - $\ell_1=e^{1-2}+e^{2-2}=e^{-1}+1$
- 第二个Block：
  - $m_2=3$
  - $\ell_2=e^{3-3}+e^{0-3}=1+e^{-3}$
- 合并：
  - $m=\max(2,3)=3$
  - $\ell=e^{2-3}(e^{-1}+1)+e^{3-3}(1+e^{-3})$
  - $\ell=e^{-1}(e^{-1}+1)+1+e^{-3}$
  - $\ell=e^{-2}+e^{-1}+1+e^{-3}$
- 这等价于直接对完整向量$[1,2,3,0]$使用最大值$3$：
  - $\ell=e^{1-3}+e^{2-3}+e^{3-3}+e^{0-3}$
  - $\ell=e^{-2}+e^{-1}+1+e^{-3}$

---

**工程实现要点**

- **Kernel Fusion**
  - $QK^\top$、mask、Softmax统计、Dropout、$PV$被融合在单个CUDA Kernel中。
  - 避免多个Kernel之间反复读写HBM。
- **SRAM驻留**
  - $S_{ij}$和$\tilde{P}_{ij}$只在SRAM中存在。
  - 计算完当前Block后立即用于更新$O_i$。
- **HBM写回内容极小**
  - 每个Block只写回：
    - $O_i$
    - $\ell_i$
    - $m_i$
  - 不写回$S_{ij}$或$P_{ij}$。
- **Recomputation**
  - Backward中重新计算局部$S_{ij}$和$P_{ij}$。
  - 用更多FLOPs换取更少HBM IO。
  - 对现代GPU更划算。

---

**复杂度影响**

| 项目 | 标准Attention | FlashAttention |
|---|---:|---:|
| FLOPs | $O(N^2d)$ | $O(N^2d)$ |
| Forward额外显存 | $O(N^2)$ | $O(N)$ |
| 是否精确 | 是 | 是 |
| HBM访问 | $\Theta(Nd+N^2)$ | $\Theta(N^2d^2/M)$ |
| 核心优化来源 | 无 | Online Softmax + Tiling + Recomputation |

- Online Softmax本身不降低Attention的理论FLOPs。
- 它降低的是：
  - 中间矩阵存储。
  - HBM读写。
  - Kernel启动和中间结果搬运开销。
- 这正是FlashAttention在实际GPU上获得显著速度提升的原因。

---

**与Approximate Attention的区别**

- Online Softmax不改变Attention定义。
- 它不会引入：
  - Low-rank近似。
  - Sparse近似。
  - Hashing近似。
  - Kernel feature近似。
- 它只是改变计算组织方式：
  - 从“完整矩阵计算”变为“Block流式计算”。
- 因此FlashAttention输出仍是：
  - $O=\operatorname{softmax}(QK^\top)V$
- 这是其区别于Linformer、Performer、Reformer等方法的核心点。

---

**关键结论**

- **Online Softmax Normalization**使Softmax可以在Block级别增量计算。
- 维护每行的**$m$**和**$\ell$**即可精确合并局部Softmax结果。
- 输出$O$必须随$m$和$\ell$变化同步重缩放。
- 该机制让FlashAttention：
  - 不保存$N \times N$ Attention矩阵。
  - 保持与标准Attention精确等价。
  - 显著减少HBM访问。
  - 将额外显存从$O(N^2)$降为$O(N)$。
  - 支撑长序列Transformer训练和推理。

### 3. Backward Recomputation Instead of Attention Matrix Storage

**核心观点**

- **Backward recomputation** 的核心不是“省掉反向传播的计算”，而是用**少量保存的中间统计量**替代 **N×N attention matrix** 的存储。
- FlashAttention 在 forward 只保留：
  - **O**：attention 输出
  - **ℓ**：每行 softmax 归一化因子
  - **m**：每行 row-max，用于数值稳定
  - **R**：dropout 的 pseudo-random state，用于 backward 重新生成 dropout mask
- backward 时不读取 HBM 中的巨大 **attention matrix P**，而是：
  - 从 HBM 读取 **Q, K, V, O, dO, ℓ, m**
  - 在 on-chip SRAM 中按 block 重算 **S = QK^T**
  - 重新得到该 block 的 **P**
  - 直接计算 **dV, dQ, dK**
- 这个设计把瓶颈从 **HBM 访问** 转成了更多的 **FLOPs**，但因为 GPU 上 attention 通常是 **memory-bound**，所以整体反而更快。

![](f2d4c917e30135292cb1109238fa9a694368cd1928cd6e9ebe3bada6de6be14e.jpg)

**问题背景：为什么不能直接存 P**

- 标准 attention 的 forward 会生成：
  - **S = QK^T ∈ R^{N×N}**
  - **P = softmax(S) ∈ R^{N×N}**
  - **O = PV ∈ R^{N×d}**
- backward 常规做法需要：
  - 读取 **P**
  - 计算 **dP = dO V^T**
  - 再通过 softmax Jacobian 求 **dS**
  - 最后得到 **dQ = dS K**、**dK = dS^T Q**、**dV = P^T dO**
- 问题在于：
  - **P** 的大小是 **N×N**
  - 对长序列，写入和读出 **P** 到 HBM 的代价极高
  - backward 中真正贵的往往不是数学运算，而是**HBM 带宽**

**实现原理**

- 关键观察 1：
  - softmax 的结果不一定要把整个矩阵一次性保存
  - 只要保存每一行的：
    - **m_i = max_j S_{ij}**
    - **ℓ_i = Σ_j exp(S_{ij} - m_i)**
  - 就能在任意 block 上重建局部 softmax
- 关键观察 2：
  - backward 真正需要的是局部的 **P_ij**
  - 只要能从 **Q_i, K_j** 重算出 **S_ij**
  - 再结合 **m_i, ℓ_i**，就能得到：
    - **P_ij = exp(S_ij - m_i) / ℓ_i**
- 关键观察 3：
  - dropout 不需要保存完整 mask
  - 只需保存 **R**，在 backward 用同一个随机状态重建 mask
- 结果：
  - 不再存储 **N×N** 的 **P**
  - backward 变成“**流式重算 + 局部梯度聚合**”

**backward 的输入输出关系**

- 输入：
  - **Q, K, V**
  - forward 输出 **O**
  - 上游梯度 **dO**
  - forward 保存的 **ℓ, m**
  - dropout 随机状态 **R**
- 输出：
  - **dQ**
  - **dK**
  - **dV**
- 关系可以概括为：
  - **O** 提供 forward 最终输出
  - **ℓ, m** 提供 softmax 的数值稳定恢复信息
  - **Q, K, V** 负责重新生成局部 attention block
  - **dO** 作为链式法则起点，传回各输入梯度
  - **R** 确保 dropout 前后一致

**算法流程**

- backward 按 block 执行，外层遍历 **K/V blocks**，内层遍历 **Q blocks**
- 每次处理一个 block 对 **(i, j)**：
  - 从 HBM 读入 **Q_i, O_i, dO_i, ℓ_i, m_i**
  - 读入 **K_j, V_j**
  - 在 SRAM 中重算：
    - **S_ij = τ Q_i K_j^T**
    - 加上 mask
    - 恢复 **P_ij**
    - 重新生成 dropout mask
- 该 block 的局部梯度计算：
  - **dV_j += P_ij^T dO_i**
  - **dP_ij = dO_i V_j^T**
  - **D_i = rowsum(dO_i ∘ O_i)**
  - **dS_ij = P_ij ∘ (dP_ij - D_i)**
  - **dQ_i += τ dS_ij K_j**
  - **dK_j += τ dS_ij^T Q_i**
- 其中：
  - **D_i = dO_i^T O_i** 的等价写法是关键优化
  - 它避免了显式构造和访问完整 **P_i:** 或 **dP_i:**
- 最终：
  - 每个 **dQ_i** 在遍历所有 **j** 后写回 HBM
  - 每个 **dK_j**、**dV_j** 在聚合完所有 **i** 后写回 HBM

**梯度公式拆解**

- **dV**
  - 标准形式：**dV = P^T dO**
  - block 形式：**dV_j += P_ij^T dO_i**
  - 含义：每个 value 向量接收来自所有 query 的加权梯度
- **dP**
  - **dP_ij = dO_i V_j^T**
  - 含义：输出梯度通过 value 投影回 attention 权重
- **softmax 反传**
  - 对每行有：
    - **dS_i: = (diag(P_i:) - P_i: P_i:^T) dP_i:**
  - 可写成：
    - **dS_ij = P_ij (dP_ij - D_i)**
  - 其中：
    - **D_i = P_i:^T dP_i: = dO_i^T O_i**
- **dQ, dK**
  - **dQ_i = Σ_j dS_ij K_j**
  - **dK_j = Σ_i dS_ij^T Q_i**
- 这套推导的关键点在于：
  - 不需要保存完整的 **P**
  - 只要能局部重建 **P_ij**
  - 就能完成所有梯度传播

**参数设置与块大小**

- backward 使用与 forward 类似的 tiling 逻辑
- block size 由 SRAM 容量 **M** 和 head dimension **d** 决定：
  - **B_c = ceil(M / (4d))**
  - **B_r = min(ceil(M / (4d)), d)**
- 设计意图：
  - **K_j, V_j** block 必须能放进 SRAM
  - **Q_i, O_i, dO_i, dQ_i** 也要能放进 SRAM
  - 中间矩阵 **S_ij** 必须能临时容纳
- 含义：
  - **M 越大**，可用 block 越大，HBM 往返次数越少
  - **d 越大**，每个 block 变宽，能容纳的 block 数会下降
- 文中结论：
  - backward 的 IO 复杂度仍然是 **Θ(N²d²/M)**，与 forward 同阶
  - 但相比标准 attention 的 **Θ(Nd + N²)**，在典型硬件上显著更少

**为什么这会更快**

- 直觉上，recomputation 增加了 FLOPs，但减少了更慢的 HBM 访问
- GPU 上很多 attention 场景是 **memory-bound** 而非 **compute-bound**
- 所以：
  - 少写一次 **N×N** 的 **P**
  - 少读一次 **N×N** 的 **P**
  - 往往比多做一些矩阵乘法更重要
- 论文实验也验证了这一点：
  - FlashAttention backward 虽然重算了 attention
  - 但整体 runtime 仍优于标准实现

**和“存储 attention matrix”的对比**

| 方案 | forward 存储 | backward 读取 | 额外内存 | HBM 访问特征 | 速度表现 |
|---|---:|---:|---:|---|---|
| 标准 attention | **P ∈ R^{N×N}** | 直接读取 **P** | **O(N²)** | 大量读写 **N×N** | 慢 |
| FlashAttention | **O, ℓ, m, R** | 现场重算 **P_ij** | **O(N)** | 主要读写 **Q,K,V,O,dO** | 快 |

**在整体中的作用**

- 这是 FlashAttention 能成立的第二个支柱
- 第一个支柱是 forward 的 **blockwise softmax**
- 第二个支柱就是 backward 的 **recomputation**
- 两者共同实现：
  - **exact attention**
  - **linear extra memory**
  - **显著减少 HBM accesses**
  - **实际 wall-clock speedup**
- 如果只有 forward tiling、没有 backward recomputation：
  - 训练阶段仍可能被存储/读取 **P** 拖慢
- 如果只有 backward recomputation、没有 forward 的 blockwise 计算：
  - forward 仍要 materialize 巨大的中间矩阵
- 所以这不是一个单点优化，而是**训练全链路优化**

**实现上的工程意义**

- 需要把 forward 和 backward 写成**单个 fused CUDA kernel family**
- 需要保证：
  - 每个 block 都能在 SRAM 内完成完整计算
  - dropout 随机性可重现
  - softmax 数值稳定
  - 反向梯度累积顺序正确
- 其本质是把深度学习算子从“PyTorch 风格的张量级中间结果”改造成“**streaming + recompute**”的硬件友好执行方式

**一句话总结**

- **Backward Recomputation** 的本质是：用 **Q/K/V + O + (m,ℓ) + R** 在 SRAM 中按 block 重新构造 attention，替代在 HBM 中保存和读取巨大的 **N×N attention matrix**，从而把训练阶段的主要瓶颈从 **显存容量** 和 **带宽** 转移到更高效的 on-chip 计算。

### 4. CUDA Kernel Fusion

**CUDA Kernel Fusion 的核心作用**

- **FlashAttention** 的实现关键不是减少 Attention 的理论 FLOPs，而是减少 **HBM reads/writes**。
- 标准 Attention 通常把以下步骤拆成多个 CUDA kernels：
  - **QKᵀ matrix multiplication**
  - **masking**
  - **softmax**
  - **dropout**
  - **PV value aggregation**
- 每个 kernel 之间都需要把中间结果写回 **HBM**，再由下一个 kernel 读出。
- FlashAttention 将这些操作融合到一个 CUDA kernel 中：
  - 中间矩阵 **S=QKᵀ** 不写入 HBM。
  - softmax 后的概率矩阵 **P** 不写入 HBM。
  - dropout mask 不完整存储到 HBM。
  - 只把最终输出 **O**、softmax 统计量 **m** 和 **ℓ** 写回 HBM。
- 这种设计直接避免了最大瓶颈：
  - 标准实现会 materialize 大小为 **N×N** 的 Attention matrix。
  - FlashAttention 只在 **SRAM/on-chip memory** 中短暂生成 block-level attention scores。

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

---

**为什么 Kernel Fusion 对 Attention 特别重要**

- Attention 的计算链条中包含大量 **memory-bound operations**：
  - **masking** 是 elementwise operation。
  - **softmax** 是 reduction operation。
  - **dropout** 是 elementwise random masking。
- 这些操作本身 FLOPs 不高，但会反复访问大矩阵。
- 标准 Attention 的典型数据流是：
  - 写入 **S∈R^{N×N}** 到 HBM。
  - 读取 **S**，执行 softmax，再写入 **P∈R^{N×N}**。
  - 读取 **P** 和 **V**，计算输出 **O**。
- 当 **N≫d** 时，**N×N** 矩阵远大于输入矩阵 **Q,K,V∈R^{N×d}**。
  - 例如 GPT-2 中常见 **N=1024, d=64**。
  - **S/P** 的元素数量是 **1,048,576**。
  - 单个 **Q/K/V** 的元素数量只有 **65,536**。
- 因此，避免 **S/P** 的 HBM materialization 是性能提升的主要来源。

| 实现方式 | 中间矩阵 S | 中间矩阵 P | kernel 数量倾向 | HBM 访问特征 |
|---|---:|---:|---:|---|
| 标准 Attention | 写入 HBM | 写入 HBM | 多个 kernel | 反复读写 **N×N** |
| 部分 fused Attention | 可能减少 masking/softmax 读写 | 通常仍保存 P | 较少 kernel | 仍存在较大 HBM 压力 |
| **FlashAttention** | **不写入 HBM** | **不写入 HBM** | **单个 fused kernel** | 主要读写 **Q,K,V,O,m,ℓ** |

---

**融合的具体计算单元**

- FlashAttention 在一个 CUDA kernel 内融合以下逻辑：
  - **Q block × K blockᵀ**
    - 计算局部 score block：
      - **Sᵢⱼ=QᵢKⱼᵀ**
    - 该矩阵只存在于 **SRAM/registers** 中。
  - **masking**
    - 对 **Sᵢⱼ** 应用 causal mask、padding mask 或其他 mask。
    - 被 mask 的位置直接设为 **−∞**。
  - **row-wise max**
    - 计算当前 block 的：
      - **m̃ᵢⱼ=rowmax(Sᵢⱼ)**
    - 用于 numerically stable softmax。
  - **exponentiation**
    - 计算：
      - **P̃ᵢⱼ=exp(Sᵢⱼ−m̃ᵢⱼ)**
  - **row-wise sum**
    - 计算：
      - **ℓ̃ᵢⱼ=rowsum(P̃ᵢⱼ)**
  - **online softmax update**
    - 将当前 block 的 softmax 统计量与历史统计量合并：
      - **mᵢ^{new}=max(mᵢ,m̃ᵢⱼ)**
      - **ℓᵢ^{new}=exp(mᵢ−mᵢ^{new})ℓᵢ+exp(m̃ᵢⱼ−mᵢ^{new})ℓ̃ᵢⱼ**
  - **dropout**
    - 在 SRAM 内对局部概率块应用 dropout。
    - forward pass 保存 pseudo-random number generator state。
    - backward pass 重新生成 dropout mask，避免存储完整 **N×N dropout mask**。
  - **value aggregation**
    - 直接计算局部 contribution：
      - **P̃ᵢⱼVⱼ**
    - 将其累加到输出 block **Oᵢ**。
  - **output rescaling**
    - 因为 softmax normalization 会随着新 block 更新，旧的 **Oᵢ** 需要按新的 **m/ℓ** 重新缩放。
    - 这是 FlashAttention 能够逐 block 精确计算 softmax 的关键。

---

**单个 fused CUDA kernel 内的数据流**

- 输入位于 **HBM**：
  - **Q∈R^{N×d}**
  - **K∈R^{N×d}**
  - **V∈R^{N×d}**
- kernel 内部循环按 block 处理：
  - 外层遍历 **K,V blocks**。
  - 内层遍历 **Q blocks**。
- 每次处理一个 block pair：
  - 从 HBM 读取：
    - **Kⱼ**
    - **Vⱼ**
    - **Qᵢ**
    - 当前 **Oᵢ**
    - 当前 **mᵢ**
    - 当前 **ℓᵢ**
  - 在 SRAM/registers 中执行：
    - **matmul**
    - **masking**
    - **softmax statistics**
    - **dropout**
    - **P̃V aggregation**
    - **O update**
  - 写回 HBM：
    - 更新后的 **Oᵢ**
    - 更新后的 **mᵢ**
    - 更新后的 **ℓᵢ**
- 不写回 HBM：
  - **Sᵢⱼ**
  - **P̃ᵢⱼ**
  - block-level dropout mask
  - 完整 **N×N attention matrix**

---

**算法流程**

- 初始化：
  - **O=0**
  - **ℓ=0**
  - **m=−∞**
  - 这些张量存储在 HBM 中。
- 设置 block sizes：
  - **B_c=⌈M/(4d)⌉**
  - **B_r=min(⌈M/(4d)⌉,d)**
  - 其中：
    - **M** 是可用 on-chip SRAM 大小。
    - **d** 是 head dimension。
- 切分输入：
  - **Q** 被切成 **T_r=⌈N/B_r⌉** 个 row blocks。
  - **K,V** 被切成 **T_c=⌈N/B_c⌉** 个 column blocks。
- 外层循环：
  - 遍历 **Kⱼ,Vⱼ**。
  - 将当前 **Kⱼ,Vⱼ** 从 HBM load 到 SRAM。
- 内层循环：
  - 遍历 **Qᵢ**。
  - 将 **Qᵢ,Oᵢ,mᵢ,ℓᵢ** load 到 SRAM。
  - 在 SRAM 中计算局部 Attention。
  - 更新 **Oᵢ,mᵢ,ℓᵢ**。
  - 写回 HBM。
- 结束后：
  - HBM 中的 **O** 即为完整结果：
    - **O=softmax(QKᵀ)V**

---

**参数设置与设计原因**

| 参数 | 含义 | 设置方式 | 作用 |
|---|---|---|---|
| **N** | sequence length | 输入决定 | 决定 Attention matrix 的边长 |
| **d** | head dimension | 模型决定，常见 64/128 | 决定 Q/K/V block 宽度 |
| **M** | SRAM capacity | GPU 硬件决定 | 决定能放入 SRAM 的 block 大小 |
| **B_c** | K/V block size | **⌈M/(4d)⌉** | 控制每次加载多少 K/V token |
| **B_r** | Q/O block size | **min(⌈M/(4d)⌉,d)** | 控制每次处理多少 query rows |
| **T_c** | K/V block 数 | **⌈N/B_c⌉** | 决定对 Q/O 的扫描次数 |
| **T_r** | Q block 数 | **⌈N/B_r⌉** | 决定每个 K/V block 内的 inner loop 次数 |

- **B_c** 不能太大：
  - **Kⱼ,Vⱼ** 必须同时放入 SRAM。
  - 局部 score matrix **Sᵢⱼ∈R^{B_r×B_c}** 也需要放入 SRAM。
- **B_r** 不能太大：
  - **Qᵢ,Oᵢ,mᵢ,ℓᵢ** 需要放入 SRAM。
  - softmax 的 row-wise reduction 也要在 on-chip memory 内完成。
- block size 增大通常减少 HBM access：
  - 更大的 **B_c** 意味着更少的 **T_c**。
  - 更少的 **T_c** 意味着对 **Q/O** 的重复扫描次数减少。
- block size 过大后收益会下降：
  - SRAM 装不下。
  - occupancy 降低。
  - register pressure 增加。
  - 计算而非 IO 成为瓶颈。

---

**输入输出关系**

- Forward pass 输入：
  - **Q**
  - **K**
  - **V**
  - 可选 **mask**
  - 可选 **dropout probability p_drop**
  - softmax scaling factor **τ**，通常为 **1/√d**
- Forward pass 输出：
  - **O**
  - softmax row max statistics **m**
  - softmax row sum statistics **ℓ**
  - pseudo-random number generator state **R**
- **O** 是模型后续层需要的 Attention output。
- **m** 和 **ℓ** 是 backward pass 的关键 checkpoint。
- **R** 用于 backward pass 重新生成 dropout mask。
- 不输出：
  - 完整 **S**
  - 完整 **P**
  - 完整 dropout mask

| 张量 | Shape | 存储位置 | 是否 materialize 到 HBM | 说明 |
|---|---:|---|---|---|
| **Q** | **N×d** | HBM | 是 | 输入 |
| **K** | **N×d** | HBM | 是 | 输入 |
| **V** | **N×d** | HBM | 是 | 输入 |
| **S=QKᵀ** | **N×N** | SRAM block-level | **否** | 只生成 **Sᵢⱼ** |
| **P=softmax(S)** | **N×N** | SRAM block-level | **否** | 只生成 **Pᵢⱼ** |
| **O=PV** | **N×d** | HBM | 是 | 输出 |
| **m** | **N** | HBM | 是 | softmax max statistics |
| **ℓ** | **N** | HBM | 是 | softmax normalization statistics |
| **dropout mask** | **N×N** | 重新生成 | **否** | 通过 RNG state 复现 |

---

**在整体 FlashAttention 中的作用**

- **CUDA Kernel Fusion** 是 FlashAttention 获得 wall-clock speedup 的工程核心。
- **Tiling** 解决的是：
  - 如何把 **N×N Attention** 拆成能放入 SRAM 的 block。
- **Online Softmax** 解决的是：
  - 如何在没有完整 row 的情况下计算精确 softmax。
- **Recomputation** 解决的是：
  - backward pass 不保存 **P**，而是在 SRAM 中重新计算。
- **Kernel Fusion** 解决的是：
  - 如何把所有 block 内操作放在一个 CUDA kernel 中，避免中间结果反复落到 HBM。
- 四者共同实现：
  - **exact attention**
  - **O(N) extra memory**
  - **显著降低 HBM access**
  - **端到端训练加速**

---

**与标准 Attention 的 HBM 访问差异**

| 阶段 | 标准 Attention | FlashAttention fused kernel |
|---|---|---|
| QKᵀ | 写 **S∈N×N** 到 HBM | **Sᵢⱼ** 仅在 SRAM 中存在 |
| Mask | 读写 **S** 或 masked S | 在 SRAM 中直接修改 **Sᵢⱼ** |
| Softmax | 读 **S**，写 **P∈N×N** | 在 SRAM 中计算 **P̃ᵢⱼ** |
| Dropout | 读写 **P/dropout(P)** | 在 SRAM 中应用，保存 RNG state |
| PV | 读 **P** 和 **V** | 直接用局部 **P̃ᵢⱼVⱼ** 更新 **Oᵢ** |
| Backward | 依赖保存的 **P** | 用 **Q,K,V,m,ℓ** 重新计算 |

---

**性能影响**

- 标准 Attention 的 IO complexity：
  - **Θ(Nd+N²)**
- FlashAttention 的 IO complexity：
  - **Θ(N²d²/M)**
- 当 **M≫d²** 时：
  - FlashAttention 的 HBM access 明显少于标准实现。
- 论文实验显示：
  - GPT-2 attention computation 最高达到 **7.6× speedup**。
  - 常见 sequence length 下整体 Attention 可达到 **up to 3× speedup**。
  - memory footprint 从 quadratic scaling 降为近似 linear scaling。

![](f2d4c917e30135292cb1109238fa9a694368cd1928cd6e9ebe3bada6de6be14e.jpg)

---

**Backward pass 中的 Fusion 与 Recomputation**

- Forward pass 不保存 **P**，因此 backward pass 需要重新计算局部 **Sᵢⱼ** 和 **Pᵢⱼ**。
- backward fused kernel 内部执行：
  - load **Qᵢ,Kⱼ,Vⱼ,Oᵢ,dOᵢ,mᵢ,ℓᵢ**
  - recompute **Sᵢⱼ=QᵢKⱼᵀ**
  - apply mask
  - recompute **Pᵢⱼ**
  - regenerate dropout mask
  - compute **dV**
  - compute **dP**
  - compute softmax gradient **dS**
  - accumulate **dQ,dK**
- 关键优化：
  - 不需要读取 **P∈N×N**。
  - 不需要读取 dropout mask。
  - recomputation 增加 FLOPs，但减少 HBM traffic。
- 在现代 GPU 上：
  - Attention 中很多步骤是 **memory-bound**。
  - 少量额外 FLOPs 的成本小于大量 HBM reads/writes 的成本。
  - 因此 backward pass 也能加速。

---

**为什么不是普通框架级 Fusion**

- PyTorch、TensorFlow、XLA 等高层框架可以自动 fusion 一些 elementwise operations。
- 但 FlashAttention 需要更细粒度控制：
  - block-level matrix multiplication
  - on-chip softmax reduction
  - SRAM residency
  - register allocation
  - warp/thread-level scheduling
  - dropout RNG replay
  - backward recomputation
- 普通 compiler fusion 很难保证：
  - **Sᵢⱼ** 不落到 HBM。
  - **Pᵢⱼ** 不落到 HBM。
  - softmax statistics 在 block 间正确增量更新。
- 因此论文采用手写 **CUDA kernel**。
- 代价是：
  - 工程复杂度高。
  - 对 GPU architecture 敏感。
  - 新 Attention variant 通常需要新 kernel。

---

**Masking 与 Dropout 的 Fusion 细节**

- **Masking**
  - 在 **Sᵢⱼ** 计算后立即执行。
  - 对 padding 或 causal 不可见位置设为 **−∞**。
  - 后续 softmax 自动得到概率 0。
  - 避免单独 kernel 读取和写入 **S**。
- **Dropout**
  - 在局部 softmax probability block 上执行。
  - forward pass 不保存完整 mask。
  - 只保存 RNG state。
  - backward pass 按相同 RNG state 重新生成 mask。
  - 避免 **O(N²)** dropout mask memory。

---

**Kernel Fusion 的限制**

- Fusion 后 kernel 更复杂：
  - register usage 更高。
  - SRAM 使用更紧张。
  - block size 选择更敏感。
- 对不同 head dimension 的适配不同：
  - **d=64** 通常更高效。
  - **d=128** 会降低可用 block size，speedup 可能下降。
- 对硬件依赖强：
  - SRAM 越大，FlashAttention 越容易使用大 block。
  - HBM bandwidth 越低，减少 HBM access 的收益越明显。
- 对长序列特别有效：
  - 因为标准 Attention 的 **N×N** HBM traffic 随 **N²** 增长。
- 对极短序列收益有限：
  - kernel launch overhead、occupancy、计算开销可能占比更高。

---

**核心结论**

- **CUDA Kernel Fusion** 是 FlashAttention 将理论 IO-aware 设计转化为真实 GPU speedup 的关键。
- 它将 **matmul → masking → softmax → dropout → value aggregation** 放入同一个 CUDA kernel。
- 它的核心目标是：
  - **不 materialize S**
  - **不 materialize P**
  - **不保存 N×N dropout mask**
  - **最大化 SRAM reuse**
  - **最小化 HBM traffic**
- 该机制使 FlashAttention 在保持 **exact attention** 的同时，实现：
  - **更低 memory footprint**
  - **更少 HBM access**
  - **更快 forward/backward**
  - **支持更长 context length**

### 5. IO Complexity Optimization

**核心观点**

- 这篇论文的关键不在于把 Attention 的 **FLOPs** 再压低一点，而是把 GPU 上最慢、最贵的部分——**HBM 读写**——降下来。
- 标准 Attention 会显式生成并读写 **N×N** 的 **attention matrix**，导致 **Θ(N²d + N²)** 级别的 HBM 访问。
- **FlashAttention** 通过 **tiling + online softmax + recomputation**，把中间大矩阵留在 **on-chip SRAM** 里处理，只在 HBM 中保留必要的输入、输出和少量统计量，从而把 HBM 访问降到 **Θ(N²d²/M)**。
- 在现代 GPU 上，很多 Attention 场景是 **memory-bound** 而不是 **compute-bound**，所以这类优化会直接转化为 wall-clock speedup。

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

**输入输出关系**

- 输入：
  - **Q, K, V ∈ R^(N×d)**，分别表示 query、key、value。
  - 额外输入包括：
    - **mask**：padding mask 或 causal mask
    - **dropout probability p_drop**
    - **softmax scaling τ**，通常是 **1/√d**
    - **SRAM 容量 M**
- 输出：
  - **O ∈ R^(N×d)**，即 attention 结果
- 训练阶段还会保存：
  - **m ∈ R^N**：每行 softmax 的最大值统计
  - **ℓ ∈ R^N**：每行 softmax 的归一化统计
  - **R**：随机数状态，用于 backward 时重建 dropout mask
- 在整体 Transformer 中，FlashAttention 替换的是标准 self-attention kernel。
  - 它不改变模型定义，不改变 attention 公式本身
  - 改变的是 **实现方式**，因此是 **exact attention**，不是近似 attention

**标准 Attention 为什么慢**

- 标准实现的主路径是：
  - 计算 **S = QKᵀ**
  - 计算 **P = softmax(S)**
  - 计算 **O = PV**
- 问题在于：
  - **S** 和 **P** 都是 **N×N**
  - 必须写回 HBM，再从 HBM 读出给下一步用
  - 这两个矩阵的规模随序列长度平方增长
- 结果就是：
  - 计算量虽然是 **O(N²d)**
  - 但更致命的是大量 **HBM traffic**
  - 对长序列而言，内存带宽往往先成为瓶颈

| 方法 | 中间矩阵 | HBM 访问量 | 复杂度特征 |
|---|---:|---:|---|
| Standard Attention | 显式存 **S、P** | **Θ(Nd + N²)** | 受 **N²** 级 memory traffic 限制 |
| FlashAttention | 不存 **S、P** | **Θ(N²d²/M)** | 通过 **SRAM tiling** 降低读写 |

**实现原理**

- 核心思想是把原本一次性处理的 **N×N** attention 分解成小块：
  - 将 **K、V** 划分成列块
  - 将 **Q** 划分成行块
- 每次只把一小块 **Q/K/V** 搬入 **SRAM**，在片上完成：
  - **QKᵀ**
  - **masked softmax**
  - **PV**
- 通过维护每一行的在线统计量：
  - **m_i**：当前见过的最大值
  - **ℓ_i**：当前累积归一化因子
- 即使 softmax 分多块计算，也能保证数值稳定且结果精确一致。

**在线 softmax 的关键递推**

- 对一个行向量的分块输入，FlashAttention 用下面的递推来保证 softmax 可流式计算：
  - 新的行最大值：
    - **m_new = max(m_old, m_block)**
  - 新的归一化项：
    - **ℓ_new = exp(m_old - m_new)·ℓ_old + exp(m_block - m_new)·ℓ_block**
  - 输出也同步重标定更新
- 这意味着：
  - 不需要把整行 **S** 或 **P** 存到 HBM
  - 每处理一个 block，就把贡献直接融合进最终输出

**算法流程**

- Forward 过程：
  - 将 **Q** 切成 **T_r = ceil(N/B_r)** 个 row blocks
  - 将 **K, V** 切成 **T_c = ceil(N/B_c)** 个 column blocks
  - 外层循环遍历 **K/V block**
  - 内层循环遍历所有 **Q block**
  - 每个 block 内做：
    - 载入 **Q_i, K_j, V_j**
    - 计算 **S_ij = τ Q_i K_jᵀ**
    - 做 **mask**
    - 计算 block 内 **row max** 和 **row sum**
    - 更新 **m_i, ℓ_i**
    - 更新输出 **O_i**
- Backward 过程：
  - 只保存 **O, m, ℓ, RNG state**
  - 反向时在片上重建所需 attention 分块
  - 重新计算 softmax 和 dropout mask
  - 直接求 **dQ, dK, dV**
- 这相当于一种 **selective gradient checkpointing**
  - 不是存下巨大中间张量
  - 而是用少量统计量换取可重算性

**参数设置**

- 块大小由 SRAM 容量 **M** 和 head dimension **d** 决定：
  - **B_c = ceil(M / 4d)**
  - **B_r = min(ceil(M / 4d), d)**
- 设计逻辑：
  - **K_j, V_j** 必须能放进 SRAM
  - **Q_i, O_i, ℓ_i, m_i** 也必须能放进 SRAM
  - 中间块 **S_ij** 也必须能放进 SRAM
- 因此块大小不是任意设定，而是由硬件资源约束出来的。
- 论文里的分析假设：
  - **d ≤ M ≤ Nd**
  - 这覆盖了实际 GPU 上常见的 SRAM/HBM 层次结构

**为什么 HBM 访问复杂度会降**

- 标准 attention：
  - 需要写出并读回 **N×N** 的 **S/P**
  - 访问规模被平方项主导
- FlashAttention：
  - 每个 **K/V block** 只加载一次
  - 对每个 **K/V block**，需要遍历所有 **Q blocks**
  - 于是总访问量近似是：
    - **N d × T_c**
    - 而 **T_c = Θ(Nd/M)**
  - 所以得到：
    - **Θ(N²d²/M)**
- 直观解释：
  - **M** 越大，单次能装下的 block 越大，重复搬运次数越少
  - 所以 SRAM 越充足，FlashAttention 越接近理想情况
  - 当 **M ≈ Nd** 时，理论上接近只需读写输入输出的下界

**为何说优化目标是 memory traffic 而不是 FLOPs**

- FlashAttention 会做额外计算：
  - 重新计算部分中间值
  - 在 backward 中重建 attention
- 但这些额外 FLOPs 的代价，通常小于减少 HBM 访问带来的收益。
- 在 GPU 上：
  - HBM 带宽远低于 on-chip SRAM 带宽
  - 很多 attention kernel 的瓶颈不是算力，而是数据搬运
- 所以论文的核心判断是：
  - **减少 FLOPs 不一定快**
  - **减少 HBM reads/writes 才真正快**

**整体作用**

- 在模型层面：
  - 允许训练更长上下文
  - 显著降低显存占用
  - 提高吞吐和 wall-clock 性能
- 在系统层面：
  - 把 Attention 从“算术主导”重新改写成“IO-aware kernel”
  - 为后续高效 attention 变体提供了一个底层 primitive
- 在应用层面：
  - 支撑 GPT-2、BERT、LRA、Path-X、Path-256 等长序列任务
  - 使以前因显存和带宽限制无法训练的长上下文模型成为可能

**与标准 Attention 的对比**

| 维度 | Standard Attention | FlashAttention |
|---|---|---|
| Attention 类型 | Exact | Exact |
| 中间矩阵存储 | **S、P** 写回 HBM | 不存大矩阵 |
| 主要瓶颈 | HBM traffic | 片上计算与块搬运 |
| Forward HBM | **Θ(Nd + N²)** | **Θ(N²d²/M)** |
| Backward HBM | **Θ(Nd + N²)** | **Θ(N²d²/M)** |
| 额外内存 | **O(N²)** | **O(N)** |
| 数值结果 | 一致 | 一致 |

**结论**

- FlashAttention 的本质是把 Attention 的优化目标从 **FLOPs 最小化**转向 **IO 最小化**。
- 它通过 **block tiling**、**online softmax** 和 **recomputation**，避免将 **N×N** 中间结果落到 HBM。
- 结果是：
  - **更少的内存访问**
  - **更低的显存占用**
  - **更高的实际训练速度**
  - **更长序列的可训练性**

如果你需要，我可以继续把这一部分拆成：
- **算法 1 的逐行解析**
- **forward/backward 的数学推导**
- **Θ(N²d²/M) 的 IO 复杂度证明**
- **和 Rabe & Staats [66] 的差异对比**

### 6. Block-Sparse FlashAttention

**核心定位**

- **Block-Sparse FlashAttention**是对**FlashAttention**的稀疏化扩展：
  - **FlashAttention**解决的是**exact dense attention**的IO瓶颈，通过**tiling**避免显式物化完整的N×N attention matrix。
  - **Block-Sparse FlashAttention**进一步引入**block sparsity mask**，只计算被保留的attention block，跳过mask中为0的block。
  - 它不是单纯减少FLOPs，而是同时减少：
    - **HBM读写次数**
    - **SRAM中的中间块计算**
    - **softmax相关的块级归一化更新**
    - **P·V乘法中的无效块计算**

- 核心复杂度从FlashAttention的：

  - **Θ(N²d²/M)**

  降低为：

  - **Θ(Nd+N²d²s/M)**

  其中：
  - **N**：sequence length
  - **d**：head dimension
  - **M**：on-chip SRAM大小
  - **s**：block sparsity mask中非零block的比例，即**fraction of nonzero blocks**
  - **Nd**项来自必须读取输入、写出输出的线性IO下界
  - **N²d²s/M**项来自只访问和计算非零attention blocks

---

**整体思想**

- 标准dense attention需要计算：

  - **S=QKᵀ**
  - **P=softmax(S)**
  - **O=PV**

- 对于长序列，S和P都是**N×N矩阵**：
  - 显式存储会造成**O(N²)**显存占用。
  - 读写S、P会造成大量**HBM access**。
  - softmax、dropout、mask等操作通常是**memory-bound**，不是compute-bound。

- **FlashAttention**通过tiling避免物化S和P：
  - 将Q按row block切分。
  - 将K、V按column block切分。
  - 每次只在SRAM中计算一个小块Sᵢⱼ。
  - 使用在线softmax统计量**m**和**ℓ**维护跨block归一化。
  - 只把最终O和少量统计量写回HBM。

- **Block-Sparse FlashAttention**在此基础上加入block级mask：
  - 如果block mask **Mᵢⱼ=0**：
    - 不加载对应的Qᵢ/Oᵢ统计块用于该block计算。
    - 不计算Sᵢⱼ。
    - 不计算softmax局部统计量。
    - 不更新Oᵢ。
  - 如果block mask **Mᵢⱼ=1**：
    - 执行与FlashAttention完全相同的tile计算。
    - 将该block纳入softmax归一化和输出累积。

---

**与Dense FlashAttention的关系**

| 维度 | FlashAttention | Block-Sparse FlashAttention |
|---|---:|---:|
| Attention类型 | **Exact dense attention** | **Approximate sparse attention** |
| 是否计算所有Q-K block | 是 | 否，只计算mask为1的block |
| 是否物化N×N attention matrix | 否 | 否 |
| 是否使用tiling | 是 | 是 |
| 是否使用online softmax | 是 | 是 |
| 是否使用SRAM复用K/V block | 是 | 是 |
| 额外显存复杂度 | **O(N)** | **O(N)** |
| IO复杂度 | **Θ(N²d²/M)** | **Θ(Nd+N²d²s/M)** |
| 加速来源 | 避免N×N矩阵HBM读写 | 避免N×N矩阵读写，并跳过零block |

---

**稀疏结构：Block Sparsity Mask**

- Block-Sparse FlashAttention要求attention mask具有**block form**：
  - 原始token级mask是N×N。
  - 算法按block粒度组织稀疏性，而不是单个元素粒度。
  - 给定block sizes：
    - **Bᵣ**：Q方向的row block size
    - **B꜀**：K/V方向的column block size
  - block mask为：

    - **M∈{0,1}^(N/Bᵣ × N/B꜀)**

- 对任意token位置k、l：
  - 如果k属于第i个Q block，l属于第j个K/V block：
    - token级mask值由block mask **Mᵢⱼ**决定。
  - 若 **Mᵢⱼ=1**：
    - block内所有token pair允许attention。
  - 若 **Mᵢⱼ=0**：
    - block内所有token pair被视为masked out，即对应logit为**−∞**。

- 这种block级设计的原因：
  - GPU更适合处理规则矩阵块，而不是非结构化稀疏点。
  - block级稀疏可以保持较高的SIMT/SIMD利用率。
  - SRAM tile可以直接复用。
  - 与FlashAttention的tiling结构天然兼容。

---

**输入输出关系**

| 项目 | 符号 | 形状 | 存储位置 | 作用 |
|---|---:|---:|---|---|
| Query | **Q** | N×d | HBM | 提供每个token的query表示 |
| Key | **K** | N×d | HBM | 提供被匹配的key表示 |
| Value | **V** | N×d | HBM | 提供被加权汇聚的value表示 |
| Block sparsity mask | **M** | N/Bᵣ × N/B꜀ | HBM或常量/索引结构 | 决定哪些attention block被计算 |
| SRAM size | **M** | scalar | 硬件参数 | 决定tile大小 |
| Output | **O** | N×d | HBM | sparse attention输出 |
| Softmax row max | **m** | N | HBM | online softmax数值稳定统计量 |
| Softmax row sum | **ℓ** | N | HBM | online softmax归一化统计量 |
| RNG state | **R** | 实现相关 | HBM | dropout backward中重建dropout mask |

- 输出满足稀疏attention定义：

  - **S=QKᵀ**
  - **S_maskedᵢⱼ=Sᵢⱼ**，当对应block mask为1
  - **S_maskedᵢⱼ=−∞**，当对应block mask为0
  - **P=softmax(S_masked)**
  - **O=PV**

- 与dense attention相比：
  - 每一行softmax只在允许的block集合上归一化。
  - 被跳过的block等价于softmax前被置为**−∞**。
  - 输出O仍然是完整的N×d矩阵，供Transformer后续层使用。

---

**参数设置**

- Block-Sparse FlashAttention沿用FlashAttention的核心tile参数：

| 参数 | 典型设置 | 含义 |
|---|---:|---|
| **B꜀** | **⌈M/(4d)⌉** | K/V方向block size |
| **Bᵣ** | **min(⌈M/(4d)⌉,d)** | Q/O方向block size |
| **T꜀** | **⌈N/B꜀⌉** | K/V block数量 |
| **Tᵣ** | **⌈N/Bᵣ⌉** | Q block数量 |
| **s** | 非零block比例 | 控制稀疏程度 |
| **τ** | 通常为**1/√d** | softmax scaling |
| **p_drop** | 如0.1 | dropout probability |

- 这些设置背后的约束：
  - **Kⱼ、Vⱼ**需要同时放入SRAM：
    - B꜀·d=O(M)
  - **Qᵢ、Oᵢ、mᵢ、ℓᵢ**需要放入SRAM：
    - Bᵣ·d=O(M)
  - 局部attention score block **Sᵢⱼ**需要放入SRAM：
    - Bᵣ·B꜀=O(M)
  - 因此选择B꜀≈M/d，Bᵣ受M/d和d共同限制。

- **s的影响非常直接**：
  - s=1时，退化为dense FlashAttention。
  - s越小，非零block越少。
  - 主导IO项按s线性降低。
  - runtime通常也随s近似线性下降，直到受到固定开销、kernel launch、内存带宽或负载不均衡影响。

---

**算法流程**

- 初始化阶段：
  - 在HBM中初始化输出：
    - **O=0**
  - 初始化softmax统计量：
    - **ℓ=0**
    - **m=−∞**
  - 保存dropout所需的pseudo-random number generator state：
    - **R**
  - 将Q、K、V切分为block：
    - Q切成Tᵣ个**Bᵣ×d**块
    - K、V切成T꜀个**B꜀×d**块
  - 将O、m、ℓ按Q block方向切分。

- 外层循环遍历K/V block：
  - 对每个 **j∈[1,T꜀]**：
    - 将**Kⱼ、Vⱼ**从HBM加载到SRAM。
    - 该设计复用K/V block，减少重复读取。

- 内层循环遍历Q block：
  - 对每个 **i∈[1,Tᵣ]**：
    - 检查block sparsity mask：
      - 如果 **Mᵢⱼ=0**：
        - 跳过该block。
        - 不计算QᵢKⱼᵀ。
        - 不更新mᵢ、ℓᵢ、Oᵢ。
      - 如果 **Mᵢⱼ=1**：
        - 加载**Qᵢ、Oᵢ、mᵢ、ℓᵢ**到SRAM。
        - 在SRAM中计算局部score：
          - **Sᵢⱼ=τQᵢKⱼᵀ**
        - 应用masking函数：
          - 例如causal mask、padding mask。
        - 计算局部softmax统计：
          - **m̃ᵢⱼ=rowmax(Sᵢⱼ)**
          - **P̃ᵢⱼ=exp(Sᵢⱼ−m̃ᵢⱼ)**
          - **ℓ̃ᵢⱼ=rowsum(P̃ᵢⱼ)**
        - 更新全局online softmax统计：
          - **mᵢ_new=max(mᵢ,m̃ᵢⱼ)**
          - **ℓᵢ_new=e^(mᵢ−mᵢ_new)ℓᵢ+e^(m̃ᵢⱼ−mᵢ_new)ℓ̃ᵢⱼ**
        - 可选执行dropout：
          - **P̃ᵢⱼ_dropped=dropout(P̃ᵢⱼ,p_drop)**
        - 更新输出block：
          - 将旧Oᵢ按新的softmax normalizer重新缩放。
          - 将当前block贡献**P̃ᵢⱼVⱼ**累加进去。
        - 将更新后的**Oᵢ、mᵢ、ℓᵢ**写回HBM。

---

**关键伪代码逻辑**

- Block-Sparse FlashAttention的核心差异可以压缩为一条判断：

```text
for j in K/V blocks:
    load K_j, V_j to SRAM
    for i in Q blocks:
        if block_mask[i, j] == 1:
            load Q_i, O_i, m_i, l_i to SRAM
            compute S_ij = Q_i K_j^T
            compute local softmax statistics
            update global m_i, l_i
            update O_i
            write O_i, m_i, l_i to HBM
        else:
            skip
```

- 与dense FlashAttention相比：
  - dense版本内层循环对所有i,j都执行。
  - block-sparse版本只对**nonzero blocks**执行。
  - 这使FLOPs和HBM access都乘上近似的**s**因子。

---

**Online Softmax如何适配稀疏block**

- dense FlashAttention中，每一行softmax需要覆盖所有K block。
- block-sparse版本中，每一行softmax只覆盖mask允许的K block。
- 在线更新仍然成立，因为softmax可以按block增量合并：
  - 对已经处理过的非零block维护：
    - 当前最大值**mᵢ**
    - 当前归一化和**ℓᵢ**
    - 当前输出**Oᵢ**
  - 新block到来时：
    - 重新计算新的最大值。
    - 对旧贡献和新贡献按指数缩放。
    - 得到与一次性对所有非零block做softmax相同的结果。

- 被跳过的零block等价于：
  - 它们的logit为**−∞**。
  - exp(−∞)=0。
  - 对m、ℓ、O无贡献。
  - 因此跳过不会破坏softmax正确性。

---

**IO复杂度分析**

- FlashAttention的dense IO复杂度：

  - **Θ(N²d²/M)**

- Block-Sparse FlashAttention中：
  - 非零block比例为**s**。
  - 只处理s比例的Q-K/V block pair。
  - 主导的block访问和计算减少为s倍。
  - 但输出O仍然必须完整写回。
  - 输入Q、K、V和输出O至少带来线性IO项。

- 因此IO复杂度为：

  - **Θ(Nd+N²d²s/M)**

| 项 | 来源 | 是否受s影响 |
|---|---|---:|
| **Nd** | 读取输入、写出输出、必要统计量访问 | 否 |
| **N²d²s/M** | 非零attention block的tile访问与计算 | 是 |
| **s=1** | 等价dense FlashAttention | 无稀疏收益 |
| **s≪1** | 长序列稀疏attention | 显著降低IO |

- 当N较大时：
  - 主导项通常是**N²d²s/M**。
  - 若使用s≈N^(-1/2)的稀疏率：
    - IO约为**Θ(N^(3/2)d²/M)**量级。
  - 若使用s≈logN/N的稀疏率：
    - IO可接近**Θ(Nd+Nd²logN/M)**。
  - 论文中提到某些稀疏模式可得到接近**Θ(Nd)**或**Θ(NlogN)**的有效扩展趋势，取决于具体s设置和d、M关系。

---

**运行时随稀疏率变化**

- 论文实验显示：
  - Block-Sparse FlashAttention的runtime会随着s降低而近似线性下降。
  - 对sequence length 4K的实验中，稀疏度越高，速度越快。
  - 加速比例与非零block比例强相关。

![](f2d4c917e30135292cb1109238fa9a694368cd1928cd6e9ebe3bada6de6be14e.jpg)

- 该结果强调：
  - FlashAttention本身已经减少了HBM access。
  - Block-Sparse FlashAttention进一步减少了需要访问和计算的block数量。
  - 稀疏结构只有在与IO-aware tiling结合时，才能更稳定地转化为wall-clock speedup。

---

**性能数据**

| 方法 | LRA平均准确率 | Speedup |
|---|---:|---:|
| Transformer | 59.3 | 1× |
| **FlashAttention** | **59.8** | **2.4×** |
| **Block-sparse FlashAttention** | **59.6** | **2.8×** |
| Linformer | 54.9 | 2.5× |
| Linear Attention | 59.6 | 2.3× |
| Performer | 58.9 | 1.8× |
| Local Attention | 56.0 | 1.7× |
| Reformer | 57.6 | 1.3× |
| Smyrf | 57.9 | 1.7× |

- 关键结论：
  - **Block-sparse FlashAttention**在LRA上准确率接近dense Transformer和FlashAttention。
  - Speedup达到**2.8×**，高于dense FlashAttention的**2.4×**。
  - 相比许多approximate attention方法，它在速度和准确率之间取得更好的折中。

---

**长序列扩展能力**

| 任务 | 方法 | Sequence Length | 结果 |
|---|---|---:|---:|
| Path-X | FlashAttention | 16K | 61.4 |
| Path-X | Block-sparse FlashAttention | 16K | 56.0 |
| Path-256 | Block-sparse FlashAttention | 64K | 63.1 |

- Block-Sparse FlashAttention的重要作用：
  - dense FlashAttention已经能支持比标准attention更长的序列。
  - 但当sequence length扩展到**64K**时，dense exact attention的计算量仍然极高。
  - Block-sparse版本通过稀疏mask进一步降低复杂度，使Transformer能处理**Path-256**这种极长序列任务。
  - 论文报告其为首个在Path-256上达到better-than-chance performance的Transformer类模型结果之一。

---

**Benchmark中的runtime表现**

| Sequence Length | FlashAttention Forward+Backward | Block-Sparse FlashAttention Forward+Backward |
|---:|---:|---:|
| 1024 | 2.55 ms | 0.89 ms |
| 2048 | 9.56 ms | 1.95 ms |
| 4096 | 37.49 ms | 4.12 ms |
| 8192 | 147.75 ms | 7.64 ms |
| 16384 | 586.61 ms | 16.60 ms |
| 32768 | 2339.11 ms | 32.73 ms |
| 65536 | 9341.30 ms | 64.11 ms |

- 数据来自带dropout和masking的forward+backward benchmark。
- 关键观察：
  - dense FlashAttention仍随N近似二次增长。
  - Block-Sparse FlashAttention表现出更接近线性的增长趋势。
  - 在64K长度上，Block-Sparse FlashAttention约为**64.11 ms**，而dense FlashAttention约为**9341.30 ms**。
  - 稀疏mask带来的收益在长序列上被显著放大。

---

**Memory Footprint**

| Sequence Length | FlashAttention Memory | Block-Sparse FlashAttention Memory |
|---:|---:|---:|
| 1024 | 209 MB | 209 MB |
| 2048 | 418 MB | 418 MB |
| 4096 | 836 MB | 836 MB |
| 8192 | 1672 MB | 1672 MB |
| 16384 | 3344 MB | 3344 MB |
| 32768 | 6688 MB | 6690 MB |
| 65536 | 13376 MB | 13384 MB |

- Block-Sparse FlashAttention与FlashAttention的memory footprint几乎相同：
  - 二者都不存储N×N attention matrix。
  - 主要显存来自Q、K、V、O、gradient和线性统计量。
  - 稀疏版本多出的mask/索引结构相对很小。
- 与标准attention不同：
  - 标准attention需要存储P或S等N×N中间矩阵。
  - FlashAttention系列显存随N近似线性增长。

![](816f269e79b71dde2a9df554b824ddbc83a97eec30f481dc2e340516d6ac0fcb.jpg)

---

**在整体Transformer中的作用**

- Block-Sparse FlashAttention替换的是Transformer block中的**self-attention kernel**：
  - 输入仍然是经过linear projection得到的Q、K、V。
  - 输出仍然是attention output O。
  - 后续仍接：
    - output projection
    - residual connection
    - layer norm
    - MLP/FFN
  - 因此它可以作为drop-in replacement接入Transformer架构。

- 它改变的是attention内部计算方式：
  - 不改变Q/K/V的语义。
  - 不改变输出shape。
  - 改变attention connectivity：
    - dense attention：每个token attend所有token。
    - block-sparse attention：每个token只attend mask允许的block。
  - 因此它是**approximate attention**，而不是exact dense attention。

- 主要价值：
  - 支持更长context。
  - 降低attention runtime。
  - 降低长序列训练显存压力。
  - 让block-sparse pattern的理论FLOP优势更真实地转化为GPU wall-clock speedup。

---

**Backward Pass机制**

- Block-Sparse FlashAttention的backward同样沿用FlashAttention的思路：
  - forward不保存完整P。
  - backward中按block重新计算Sᵢⱼ和Pᵢⱼ。
  - 只对mask为1的block执行反向计算。
  - 对mask为0的block跳过gradient贡献。

- backward涉及的梯度：
  - **dV=PᵀdO**
  - **dP=dOVᵀ**
  - **dS=P∘(dP−D)**
  - **dQ=dSK**
  - **dK=dSᵀQ**

- 稀疏mask的影响：
  - 对于零block：
    - Pᵢⱼ=0
    - dSᵢⱼ=0
    - 对dQ、dK、dV无贡献
  - 因此可安全跳过。
  - dropout mask不需要存储N×N，而是通过保存RNG state在backward中重建。

---

**为什么必须是Block Sparse而不是任意Sparse**

- 任意非结构化稀疏attention在GPU上通常难以高效：
  - irregular memory access严重。
  - warp divergence增加。
  - 难以使用Tensor Core。
  - 稀疏索引开销可能抵消FLOP减少。
  - 小粒度稀疏矩阵乘法通常不如dense block matmul高效。

- Block Sparse更适合GPU：
  - 每个非零block内部仍是dense matmul。
  - 可复用FlashAttention的tile kernel。
  - SRAM布局规则。
  - HBM访问更连续。
  - 更容易利用shared memory和register。
  - 更容易和masking、softmax、dropout融合到一个kernel中。

---

**稀疏模式选择**

- 论文实验中使用了**fixed butterfly sparsity pattern**：
  - 该模式来自butterfly structured matrices相关工作。
  - 具有较好的表达能力和硬件友好性。
  - 能以较少block连接覆盖长距离信息传播。
  - 适合长序列任务。

- 选择block sparsity pattern时需要平衡：
  - **s越小**：
    - runtime越低。
    - IO越少。
    - 但attention表达能力可能下降。
  - **s越大**：
    - 更接近dense attention。
    - 精度更稳。
    - 但速度收益下降。
  - 良好的pattern应同时满足：
    - 覆盖local信息。
    - 支持global或long-range communication。
    - block布局规则。
    - GPU kernel可高效遍历。

---

**实现层面的关键点**

- 需要自定义CUDA kernel：
  - PyTorch原生算子难以控制HBM和SRAM之间的数据移动。
  - 需要把matmul、mask、softmax、dropout、PV融合在一个kernel内。
  - 需要显式管理shared memory、register和block-level scheduling。

- 需要高效表示block mask：
  - 不能简单存一个巨大N/Bᵣ×N/B꜀ dense mask后逐项低效判断。
  - 更优做法通常是存储每个row block对应的nonzero column block列表。
  - 这样内层循环可以只遍历非零block，而不是遍历所有block再判断。

- 需要处理负载均衡：
  - 如果不同row block的非零block数量差异很大，会造成GPU workload imbalance。
  - 固定规则pattern通常更容易负载均衡。
  - 动态稀疏或内容相关稀疏需要额外调度设计。

- 需要保证数值稳定：
  - 稀疏softmax仍使用row-wise max trick。
  - 每行只在非零block集合上计算m和ℓ。
  - block处理顺序不应改变最终数学结果，只会带来浮点舍入差异。

---

**优势**

- **IO复杂度按s缩放**：
  - 主导项从N²d²/M降为N²d²s/M。
- **显存线性增长**：
  - 不保存N×N矩阵。
- **长序列能力强**：
  - 支持16K、64K级别sequence length。
- **速度优于多数approximate attention实现**：
  - 原因不是只看FLOPs，而是实际减少HBM traffic。
- **与FlashAttention兼容性高**：
  - 复用tiling、online softmax、recomputation思想。
- **适合作为高性能稀疏attention primitive**：
  - 可服务Longformer/BigBird式稀疏思想，也可服务butterfly pattern等结构化稀疏。

---

**局限**

- **不是exact dense attention**：
  - block mask会改变attention connectivity。
  - 质量依赖sparsity pattern。
- **实现复杂度高**：
  - 需要CUDA级优化。
  - 不同GPU架构可能需要重新调参。
- **稀疏pattern选择敏感**：
  - 不合适的mask会损害长距离建模。
  - 过度稀疏可能导致信息瓶颈。
- **小序列收益有限**：
  - 当N较小时，kernel overhead和固定IO项占比更高。
  - 稀疏调度开销可能抵消部分收益。
- **动态稀疏更难优化**：
  - 固定block mask最容易高效实现。
  - 内容相关稀疏需要额外索引生成和调度，可能引入新的瓶颈。

---

**关键结论**

- **Block-Sparse FlashAttention=FlashAttention的IO-aware tiling框架+block-level sparse mask**。
- 它通过跳过mask为0的attention blocks，将主导IO复杂度降为**Θ(Nd+N²d²s/M)**。
- 它保留了FlashAttention的核心优势：
  - 不物化N×N attention matrix。
  - 使用SRAM tile计算。
  - 使用online softmax。
  - backward中重计算attention block。
- 它进一步将稀疏attention的理论优势转化为实际GPU加速。
- 在长序列场景中，它比dense FlashAttention更具扩展性，是论文中实现**64K sequence length Transformer**能力的关键组件。


---

## 4. 实验方法与实验结果

None

---

