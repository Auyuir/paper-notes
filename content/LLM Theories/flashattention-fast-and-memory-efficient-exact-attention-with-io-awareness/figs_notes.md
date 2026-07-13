# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness 图表详解

### Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.

![0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg)

- **图像整体结构**
  - 该图由三部分组成，围绕 **FlashAttention 的 IO-aware 设计思想** 展开：
    - 左侧：展示 **GPU memory hierarchy**，强调 **SRAM、HBM、CPU DRAM** 的带宽和容量差异。
    - 中间：展示 **FlashAttention 的 tiling/blocking 计算流程**，说明如何避免将完整的 **N × N attention matrix** 写入 HBM。
    - 右侧：对比 **PyTorch attention** 与 **FlashAttention** 在 GPT-2 attention 计算上的运行时间，突出 **kernel fusion** 和减少 HBM IO 带来的速度提升。

| 图中区域 | 核心内容 | 主要信息 |
|---|---|---|
| 左侧 | **Memory Hierarchy** | SRAM 快但小，HBM 较慢但大，CPU DRAM 更慢 |
| 中间 | **FlashAttention tiling** | 分块加载 Q/K/V 到 SRAM，在片上计算 attention block |
| 右侧 | **Runtime comparison** | FlashAttention 显著快于 PyTorch attention，约 **7.6× speedup** |

- **左侧：GPU Memory Hierarchy 的含义**
  - 图中用金字塔表示现代 GPU 的多级存储层次：
    - 顶层是 **GPU SRAM**：
      - 带宽约 **19 TB/s**。
      - 容量约 **20 MB**，这里通常指所有 SM 上 SRAM 的总量。
      - 速度最快，但容量极小。
    - 中层是 **GPU HBM**：
      - 带宽约 **1.5 TB/s**。
      - 容量约 **40 GB**。
      - 是 GPU 主显存，容量大但比 SRAM 慢一个数量级。
    - 底层是 **Main Memory / CPU DRAM**：
      - 带宽约 **12.8 GB/s**。
      - 容量通常大于 **1 TB**。
      - 容量最大，但速度远低于 GPU 内部存储。

| 存储层级 | 图中带宽 | 图中容量 | 特点 | 对 attention 的影响 |
|---|---:|---:|---|---|
| **GPU SRAM** | **19 TB/s** | **20 MB** | 极快、极小 | 适合存放 Q/K/V block 和局部 attention block |
| **GPU HBM** | **1.5 TB/s** | **40 GB** | 较快、较大 | 标准 attention 会频繁读写 S/P 矩阵，成为瓶颈 |
| **CPU DRAM** | **12.8 GB/s** | **>1 TB** | 慢、大 | 通常不参与单个 GPU kernel 内 attention 主计算 |

- **核心观察**
  - 现代 GPU 上，很多 Transformer 操作不是单纯受 FLOPs 限制，而是受 **HBM memory bandwidth** 限制。
  - 标准 attention 会显式生成并存储：
    - **S = QKᵀ**
    - **P = softmax(S)**
  - 这两个矩阵都是 **N × N**，当序列长度 N 较大时，会造成巨大的 HBM 读写压力。
  - FlashAttention 的关键不是减少理论 FLOPs，而是减少 **HBM accesses**。

- **中间：FlashAttention 计算流程**
  - 中间部分展示 FlashAttention 的核心算法：**tiling + SRAM block computation**。
  - 输入矩阵：
    - **Q: N × d**
    - **Kᵀ: d × N**
    - **V: N × d**
  - 输出矩阵：
    - **sm(QKᵀ)V: N × d**
  - 虚线框表示传统 attention 中可能被物化的巨大 **N × N attention matrix**。
  - FlashAttention 的目标是：**不在 HBM 中 materialize N × N attention matrix**。

| 矩阵 | 尺寸 | 在图中的角色 |
|---|---:|---|
| **Q** | **N × d** | 按行 block 加载到 SRAM |
| **Kᵀ** | **d × N** | 按列 block 加载到 SRAM |
| **V** | **N × d** | 与 K 对应分块加载到 SRAM |
| **QKᵀ** | **N × N** | 只在 SRAM 中局部计算，不完整写入 HBM |
| **Output** | **N × d** | 分块更新后写回 HBM |

- **Outer Loop 的作用**
  - 图中红色箭头表示 **Outer Loop**。
  - FlashAttention 在外层循环中遍历 **K 和 V 的 block**。
  - 每次将一个 K block 和对应的 V block 从 **HBM copy 到 SRAM**。
  - 这样可以让当前 K/V block 被多个 Q block 复用，减少重复从 HBM 读取 K/V 的次数。
  - 图中 Kᵀ 顶部和 V 右侧都标出了红色 **Outer Loop**，说明 K/V block 是外层遍历对象。

- **Inner Loop 的作用**
  - 图中蓝色箭头表示 **Inner Loop**。
  - 在固定一个 K/V block 后，FlashAttention 遍历所有 **Q block**。
  - 每次将一个 Q block 从 HBM 加载到 SRAM。
  - 在 SRAM 中计算当前局部 block：
    - **Sᵢⱼ = QᵢKⱼᵀ**
    - 局部 softmax 统计量
    - 局部输出贡献
  - 然后将更新后的输出 block 写回 HBM。

| 循环层级 | 遍历对象 | 加载到 SRAM 的内容 | 目的 |
|---|---|---|---|
| **Outer Loop** | K/V blocks | **Kⱼ, Vⱼ** | 复用 K/V block |
| **Inner Loop** | Q blocks | **Qᵢ, Oᵢ, softmax stats** | 计算局部 attention 并更新输出 |

- **虚线 N × N 框的意义**
  - 中间的虚线大框标注为 **QKᵀ: N × N**。
  - 它代表标准 attention 中会被显式生成的 attention score matrix。
  - FlashAttention 不将完整 **QKᵀ** 或 **softmax(QKᵀ)** 写入 HBM。
  - 只在 SRAM 中生成一个小块：
    - **Sᵢⱼ = QᵢKⱼᵀ**
  - 处理完当前 block 后，局部矩阵即可丢弃。
  - 因此避免了 **O(N²)** 的中间显存占用。

- **FlashAttention 如何正确计算 softmax**
  - attention 的难点在于 softmax 是按行归一化的，理论上每一行需要看到所有 key。
  - FlashAttention 使用 **online softmax / incremental softmax normalization**。
  - 每个 Q block 会维护：
    - 当前行最大值 **m**
    - 当前归一化因子 **ℓ**
    - 当前输出累积值 **O**
  - 每处理一个新的 K/V block，就更新这些统计量。
  - 最终结果与标准 attention 完全一致，因此 FlashAttention 是 **exact attention**，不是近似 attention。

| 统计量 | 含义 | 作用 |
|---|---|---|
| **m** | row-wise maximum | 保证 softmax 数值稳定 |
| **ℓ** | row-wise normalization factor | 累积 softmax denominator |
| **O** | output accumulator | 累积 attention output |

- **中间图传达的核心优化**
  - 标准 attention 的计算路径通常是：
    - 计算 **QKᵀ**
    - 写入 HBM
    - 从 HBM 读取 **QKᵀ**
    - 计算 softmax
    - 写入 HBM
    - 再读取 softmax 结果
    - 与 V 相乘
  - FlashAttention 的路径是：
    - 将 Q/K/V block 加载到 **SRAM**
    - 在 SRAM 中完成 **Matmul + Mask + Softmax + Dropout + Matmul**
    - 只将最终 **O block** 写回 HBM
  - 因此 FlashAttention 的关键优势是：**减少中间结果在 HBM 中的读写，而不是改变 attention 数学定义**。

- **右侧：PyTorch 与 FlashAttention 运行时间对比**
  - 右侧柱状图标题为 **Attention on GPT-2**。
  - 纵轴是运行时间，单位为 **milliseconds (ms)**。
  - 左侧柱为 **PyTorch** attention。
  - 右侧柱为 **FlashAttention**。
  - PyTorch attention 的总时间约为 **16–17 ms**。
  - FlashAttention 的时间约为 **2 ms** 左右。
  - 图注说明获得约 **7.6× speedup**。

| 方法 | 图中时间量级 | 特点 |
|---|---:|---|
| **PyTorch Attention** | 约 **16–17 ms** | 多个 kernel，频繁读写 HBM |
| **FlashAttention** | 约 **2 ms** | fused kernel，避免 N × N attention matrix 写入 HBM |
| **Speedup** | 约 **7.6×** | 主要来自 IO 减少 |

- **PyTorch 柱状图中的分段含义**
  - PyTorch attention 被拆成多个阶段：
    - **Matmul**
    - **Mask**
    - **Softmax**
    - **Dropout**
    - **Matmul**
  - 每个阶段往往对应不同 kernel 或至少需要中间结果落到 HBM。
  - 这些阶段之间的大量中间矩阵读写导致运行时间显著增加。
  - 特别是 **Mask、Softmax、Dropout** 等操作本身 FLOPs 不高，但对 **N × N matrix** 进行读写，非常 memory-bound。

| PyTorch 阶段 | 操作 | 是否涉及 N × N 中间矩阵 | 性能瓶颈 |
|---|---|---|---|
| **Matmul** | QKᵀ | 是 | 计算 + 写 HBM |
| **Mask** | attention mask | 是 | HBM 读写 |
| **Softmax** | row-wise softmax | 是 | HBM 读写 + reduction |
| **Dropout** | dropout on P | 是 | HBM 读写 |
| **Matmul** | PV | 是 | 读取 P 和 V |

- **FlashAttention 柱状图中的 Fused Kernel**
  - FlashAttention 右侧柱旁标注 **Fused Kernel**。
  - 表示多个 attention 子操作被融合到一个 CUDA kernel 中：
    - **QKᵀ block matmul**
    - **Mask**
    - **Softmax**
    - **Dropout**
    - **PV block matmul**
  - 这些操作在 SRAM 中完成，避免将中间的 **S/P matrix** 写入 HBM。
  - 这正是图中运行时间大幅下降的直接原因。

- **为什么减少 HBM 读写比减少 FLOPs 更关键**
  - 图像左侧给出的带宽对比说明：
    - SRAM 带宽 **19 TB/s**
    - HBM 带宽 **1.5 TB/s**
  - SRAM 约比 HBM 快 **一个数量级以上**。
  - 标准 attention 中，**N × N attention matrix** 对 HBM 的压力极大。
  - FlashAttention 即使在 backward 中需要 recomputation，增加了一些 FLOPs，也可能更快。
  - 因为实际瓶颈往往是 **data movement**，而不是单纯算术计算。

- **图中体现的 IO-aware 思想**
  - FlashAttention 是典型的 **IO-aware algorithm**：
    - 显式考虑 GPU 不同 memory level 的带宽差异。
    - 通过 tiling 提高 SRAM reuse。
    - 通过 kernel fusion 减少 HBM round trips。
    - 通过 recomputation 避免保存大型中间激活。
  - 其优化目标不是只看 **FLOPs complexity**，而是看 **HBM access complexity**。

| 设计点 | 标准 attention | FlashAttention |
|---|---|---|
| 是否物化 **S = QKᵀ** | 是 | 否 |
| 是否物化 **P = softmax(S)** | 是 | 否 |
| 中间矩阵存储位置 | HBM | SRAM 局部 block |
| softmax 方式 | 全矩阵 softmax | online/block softmax |
| kernel 数量 | 多个 | fused kernel |
| 显存复杂度 | **O(N²)** | **O(N)** additional memory |
| attention 类型 | exact | exact |

- **与论文算法的对应关系**
  - 图中中间流程对应论文中的 **Algorithm 1 FlashAttention**：
    - 第 6 行：加载 **Kⱼ, Vⱼ** 到 SRAM。
    - 第 8 行：加载 **Qᵢ, Oᵢ, ℓᵢ, mᵢ** 到 SRAM。
    - 第 9 行：计算 **Sᵢⱼ = QᵢKⱼᵀ**。
    - 第 10–11 行：计算局部 softmax 统计量。
    - 第 12 行：更新输出 **Oᵢ**。
    - 第 13 行：写回 **ℓᵢ, mᵢ**。
  - 图中“Output to HBM”对应每个 block 输出累积更新后写回。

- **图像传达的核心结论**
  - FlashAttention 的主要贡献不是提出新的 attention 近似形式，而是实现了：
    - **exact attention**
    - **faster wall-clock runtime**
    - **lower memory footprint**
    - **IO-aware GPU execution**
  - 通过避免写入和读取巨大的 **N × N attention matrix**，FlashAttention 将 attention 计算从 HBM-heavy 流程变为 SRAM-friendly 流程。
  - 右图中的 **7.6× speedup** 说明，在 GPT-2 attention 场景下，IO 优化可以带来远超普通 kernel-level 优化的收益。

- **一句话总结**
  - 这张图的核心信息是：**FlashAttention 通过 tiling 将 Q/K/V 分块搬入高速 SRAM，在片上完成 attention block 计算，并用 fused kernel 避免 N × N attention matrix 在 HBM 中物化，从而显著降低 IO 开销并实现约 7.6× 的 GPT-2 attention 加速。**

### f2d4c917e30135292cb1109238fa9a694368cd1928cd6e9ebe3bada6de6be14e.jpg

![f2d4c917e30135292cb1109238fa9a694368cd1928cd6e9ebe3bada6de6be14e.jpg](f2d4c917e30135292cb1109238fa9a694368cd1928cd6e9ebe3bada6de6be14e.jpg)

- **图像内容概述**
  - 该图展示 FlashAttention 中 **Block Size** 对两个指标的影响：
    - **HBM Accesses**：绿色曲线，对应左侧纵轴，单位为 **GB**。
    - **Forward Runtime**：蓝色曲线，对应右侧纵轴，单位为 **ms**。
  - 横轴为 **Block Size**，取值约为 **64、128、256、512**。
  - 图像对应论文 Figure 2 的中间子图，用于验证：**更大的 tile/block 能减少 HBM 访问，从而降低运行时间，但收益会逐渐饱和**。

- **坐标轴与曲线含义**

| 元素 | 含义 | 观察 |
|---|---|---|
| 横轴 **Block Size** | FlashAttention 中 tile/block 的大小 | 从 64 增加到 512 |
| 左纵轴 **HBM Accesses (GB)** | GPU HBM 读写量 | 随 Block Size 增大显著下降 |
| 右纵轴 **Fwd Runtime (ms)** | 前向传播运行时间 | 随 HBM 访问减少而下降，但后期趋于平缓 |
| 绿色曲线 | **HBM Accesses** | 下降趋势明显 |
| 蓝色曲线 | **Runtime** | 先快速下降，后进入平台期 |

- **近似数据读取**

| Block Size | HBM Accesses 约值 | Forward Runtime 约值 | 主要现象 |
|---:|---:|---:|---|
| 64 | **6–7 GB** | **6–7 ms** | HBM 访问最多，运行最慢 |
| 128 | **约 3 GB** | **约 3–4 ms** | HBM 访问大幅下降，runtime 明显降低 |
| 256 | **约 1.5–2 GB** | **约 2.5–3 ms** | 继续下降，但收益减弱 |
| 512 | **约 1 GB** | **约 2.5 ms 左右** | HBM 继续下降，但 runtime 基本饱和 |

- **核心结论**
  - **Block Size 增大 → HBM Accesses 减少**。
  - **HBM Accesses 减少 → Forward Runtime 降低**。
  - 但当 Block Size 增大到约 **256** 之后，runtime 降低不再明显，说明此时计算不再主要受 HBM IO 限制，而可能转向：
    - **算术计算开销**
    - **SRAM 容量限制**
    - **寄存器压力**
    - **线程并行度 / occupancy**
    - **kernel scheduling overhead**

- **与 FlashAttention 机制的关系**
  - FlashAttention 的关键思想是用 **tiling** 将 Q、K、V 分块加载到 GPU on-chip **SRAM** 中。
  - 标准 attention 会显式 materialize 大小为 **N × N** 的 attention matrix，并频繁读写 HBM。
  - FlashAttention 避免将完整 attention matrix 写入 HBM，而是在 SRAM 内完成：
    - **QKᵀ block computation**
    - **online softmax**
    - **softmax normalization update**
    - **PV block accumulation**
  - 因此，Block Size 越大，每次能处理的 K/V 块越大，需要重复扫描 Q/O 的次数越少，HBM 访问量越低。

- **理论解释**
  - 论文中给出的 FlashAttention IO complexity 为：
    - **Θ(N²d² / M)**
  - 其中：
    - **N** 是 sequence length
    - **d** 是 head dimension
    - **M** 是 SRAM size
  - 增大 Block Size 等价于更充分利用 SRAM，即有效减少外层 block 数量。
  - 当 block 更大时，访问 HBM 的次数减少，因此绿色曲线下降。
  - 但 Block Size 不能无限增大，因为中间矩阵块如 **Sᵢⱼ ∈ R^{Bᵣ × B꜀}** 必须放入 SRAM。

- **为什么 runtime 后期趋于平缓**
  - 图中从 **64 → 128**，runtime 下降最明显，说明此阶段主要瓶颈是 **HBM bandwidth**。
  - 从 **256 → 512**，HBM Accesses 仍然下降，但 runtime 几乎不再下降，说明：
    - **IO bottleneck 已被部分缓解**
    - 剩余时间更多由 **matrix multiplication、softmax、exp、reduction** 等计算决定
    - 更大的 block 可能降低并行度或增加片上资源压力
  - 因此 FlashAttention 的最优 Block Size 不是越大越好，而是需要在 **IO 减少** 与 **硬件资源利用率** 之间权衡。

- **图像支撑的论文论点**
  - 该图直接支持论文的核心观点：**attention 的实际 wall-clock speed 不只由 FLOPs 决定，HBM IO 是关键因素**。
  - 即使 FlashAttention 可能因为 recomputation 增加部分 FLOPs，只要显著减少 HBM 读写，整体仍然更快。
  - 图中绿色曲线和蓝色曲线高度相关，说明：
    - **HBM access 是 FlashAttention forward runtime 的主要决定因素之一**
    - **IO-aware algorithm design 对 Transformer 性能优化非常关键**

- **工程启示**
  - 在实现 FlashAttention 或类似 fused attention kernel 时，Block Size 是核心调优参数。
  - 较小 Block Size：
    - SRAM 压力小
    - 但需要更多 HBM 访问
    - runtime 较高
  - 较大 Block Size：
    - HBM 访问少
    - 但可能受 SRAM、register、occupancy 限制
  - 实际部署中应根据 GPU 架构选择合适 block：
    - **A100** SRAM 较大，可使用更大 block
    - **T4 / 较小 GPU** SRAM 较小，block 需要更小，speedup 也会下降

- **一句话总结**
  - 该图说明：**FlashAttention 通过增大 Block Size 更充分利用 SRAM，显著减少 HBM Accesses，从而降低 forward runtime；但当 Block Size 足够大后，性能收益进入平台期，瓶颈从 IO 转向计算和硬件资源限制。**

### 581d9c655d98724b3a4a319d39161be45a2bdc250eb82557da2871ade1203941.jpg

![581d9c655d98724b3a4a319d39161be45a2bdc250eb82557da2871ade1203941.jpg](581d9c655d98724b3a4a319d39161be45a2bdc250eb82557da2871ade1203941.jpg)

- **图片核心信息**
  - 该图展示了 **Block-Sparse FlashAttention** 在不同稀疏度下的运行时间变化，并与 **Dense FlashAttention** 进行对比。
  - 横轴是 **% Non-Zero Blocks**，表示 block-sparse attention mask 中非零块的比例。
  - 纵轴是 **Fwd + Bwd runtime，单位 ms**，表示 forward pass 与 backward pass 的总运行时间。
  - 图中红色虚线表示 **Dense FlashAttention** 的运行时间基线。
  - 蓝色曲线表示 **Block-Sparse FlashAttention** 的运行时间，随着非零块比例增加而上升。

- **图中元素解析**

| 图中元素 | 含义 | 观察结果 |
|---|---|---|
| **横轴：% Non-Zero Blocks** | block-sparse mask 中实际参与计算的 block 比例 | 从约 **10%** 增加到约 **80%** |
| **纵轴：Fwd + Bwd runtime (ms)** | 前向与反向传播总耗时 | 范围约 **25 ms 到 180 ms** |
| **红色虚线 Dense FlashAttention** | 稠密 FlashAttention 的耗时 | 基本恒定，约 **170–180 ms** |
| **蓝色曲线 Block-Sparse FlashAttention** | 块稀疏 FlashAttention 的耗时 | 随非零块比例增加而近似单调上升 |
| **两者差距** | 稀疏化带来的加速收益 | 非零块越少，加速越明显 |

- **主要趋势**
  - **Block-Sparse FlashAttention 的运行时间与非零块比例正相关**。
  - 当 **% Non-Zero Blocks 较低** 时，Block-Sparse FlashAttention 显著快于 Dense FlashAttention。
  - 当非零块比例接近较高水平时，Block-Sparse FlashAttention 的耗时逐渐接近 Dense FlashAttention。
  - 图中蓝色曲线并非完全线性，但整体趋势接近论文中所述：**运行时间提升与 sparsity ratio 成比例**。

- **定量近似解读**

| % Non-Zero Blocks | Block-Sparse FlashAttention 近似耗时 | 相对 Dense FlashAttention 的效果 |
|---:|---:|---|
| **约 10%** | **约 25–30 ms** | 极大加速，约 **6×** 左右 |
| **约 20%** | **约 40 ms** | 明显加速，约 **4×** 左右 |
| **约 40%** | **约 80–90 ms** | 约 **2×** 加速 |
| **约 60%** | **约 125–140 ms** | 加速幅度减小 |
| **约 80%** | **约 170–180 ms** | 接近 Dense FlashAttention |

- **与论文理论分析的对应关系**
  - 论文 Proposition 4 给出 Block-Sparse FlashAttention 的 IO complexity：
    - **Θ(Nd + N²d²M⁻¹s)**
  - 其中：
    - **N**：sequence length
    - **d**：head dimension
    - **M**：SRAM size
    - **s**：non-zero blocks fraction，即图中横轴对应的比例
  - 因此，当 **s 变小** 时，主导项 **N²d²M⁻¹s** 会按比例下降。
  - 图中的蓝色曲线验证了这一点：**非零块比例越低，HBM access 越少，forward + backward runtime 越低**。

- **为什么 Block-Sparse FlashAttention 能更快**
  - 普通 **Dense FlashAttention** 虽然避免了显式 materialize 完整 attention matrix，但仍然要计算所有 Q-K block pair。
  - **Block-Sparse FlashAttention** 只计算 mask 中非零的 block。
  - 被 mask 掉的 block：
    - **不加载对应 Q/K/V block**
    - **不计算 QKᵀ**
    - **不执行 softmax block**
    - **不参与 PV 乘法**
    - **减少 backward recomputation**
  - 因此，它不仅减少 FLOPs，也减少关键瓶颈：**HBM reads/writes**。

- **与 Dense FlashAttention 的对比**
  - **Dense FlashAttention** 的耗时基本不随稀疏比例变化，因为它没有稀疏 mask，所有 block 都计算。
  - **Block-Sparse FlashAttention** 的耗时随非零 block 数量变化。
  - 当 block mask 很稀疏时，Block-Sparse FlashAttention 显著快于 Dense FlashAttention。
  - 当 mask 接近 dense 时，Block-Sparse FlashAttention 的优势减弱，最终接近 Dense FlashAttention。

- **图中隐含的系统性能结论**
  - 该图强调：在 attention 中，实际 wall-clock speedup 不只取决于理论 FLOPs，还取决于 **IO-aware implementation**。
  - 如果只是使用 sparse pattern，但实现仍然频繁访问 HBM 或产生额外 kernel overhead，可能无法获得真实加速。
  - Block-Sparse FlashAttention 的关键优势是将 **sparsity** 与 **tiling、kernel fusion、SRAM reuse、recomputation** 结合起来。

- **与 Figure 2 右图说明的一致性**
  - 论文原文指出：  
    **“The runtime of block-sparse FlashAttention is faster than FlashAttention by a factor proportional to the sparsity.”**
  - 该图正是这一结论的实验证据。
  - 红色虚线给出 Dense FlashAttention 的固定运行时间。
  - 蓝色曲线展示随着非零块比例增加，Block-Sparse FlashAttention 逐渐失去稀疏加速优势。

- **对长序列任务的意义**
  - 对于长序列，dense attention 的计算量与 IO 仍然随 **N²** 增长。
  - Block-sparse pattern 可以将有效计算块数量大幅减少。
  - 因此在 **Path-X、Path-256、Long-Range Arena** 等长上下文任务中，Block-Sparse FlashAttention 可以支持更长 sequence length。
  - 论文中报告 Block-Sparse FlashAttention 可扩展到 **64K sequence length**，并在 **Path-256** 上达到 **63.1% accuracy**。

- **关键结论**
  - **Block-Sparse FlashAttention 的运行时间随非零 block 比例增加而增加。**
  - **低 non-zero block ratio 带来显著 speedup。**
  - **Dense FlashAttention 是强基线，但在高稀疏场景下 Block-Sparse FlashAttention 更快。**
  - **图像实验证明了论文的 IO complexity 分析：稀疏度 s 会近似线性影响主要 IO 成本与实际运行时间。**
  - **该结果说明 FlashAttention 不只是一个 exact attention kernel，也可以作为高效 approximate/sparse attention 的底层 primitive。**

### 816f269e79b71dde2a9df554b824ddbc83a97eec30f481dc2e340516d6ac0fcb.jpg

![816f269e79b71dde2a9df554b824ddbc83a97eec30f481dc2e340516d6ac0fcb.jpg](816f269e79b71dde2a9df554b824ddbc83a97eec30f481dc2e340516d6ac0fcb.jpg)

- **图片内容概览**
  - 该图来自 **FlashAttention** 论文中的实验部分，对比不同 attention 实现随 **sequence length** 增长时的：
    - **左图：Attention Runtime（forward pass + backward pass）**
    - **右图：Attention Memory Usage**
  - 主要比较对象包括：
    - **FlashAttention**
    - **Block-Sparse FlashAttention**
    - **PyTorch Attention**
    - **Megatron Attention**
    - **Linformer Attention**
    - **OpenAI Sparse Attention**

- **图例对应关系**

| 曲线 | 方法 | 类型 | 主要特征 |
|---|---|---|---|
| 黑色点线 | **FlashAttention** | exact attention | 精确 attention，减少 HBM IO |
| 绿色虚线 | **Block-Sparse FlashAttention** | sparse / approximate attention | 块稀疏版本，速度更快 |
| 蓝色实线 | **PyTorch Attention** | standard exact attention | 标准实现，显式物化 attention matrix |
| 橙色实线 | **Megatron Attention** | optimized exact attention | 融合优化版本，但仍受内存访问限制 |
| 棕色实线 | **Linformer Attention** | approximate attention | 低秩近似，内存线性增长 |
| 青色虚线 | **OpenAI Sparse Attention** | sparse attention | 稀疏 attention 实现 |

- **左图：Attention Runtime 分析**
  - 横轴是 **Sequence Length**，范围大约从 **128 到 8192**。
  - 纵轴是 **Runtime(ms)**，使用 **log scale**，说明不同方法之间存在数量级差异。
  - 运行时间统计的是 **forward pass + backward pass**，即训练阶段完整 attention 计算开销。

- **左图核心趋势**

| 方法 | 运行时间趋势 | 关键观察 |
|---|---|---|
| **PyTorch Attention** | 随序列长度快速上升 | 标准 attention 显式读写 $N \times N$ attention matrix，长序列下很快变慢 |
| **Megatron Attention** | 比 PyTorch 更快，但仍近似二次增长 | 通过 kernel fusion 优化，但仍无法完全避免 attention matrix 的 IO 开销 |
| **FlashAttention** | 仍呈二次计算趋势，但常数显著更低 | 不物化完整 attention matrix，减少 HBM 访问，因此明显快于 PyTorch / Megatron |
| **Block-Sparse FlashAttention** | 增长最慢 | 利用 block sparsity，计算与 IO 均减少 |
| **Linformer Attention** | 长序列增长较平缓 | 低秩近似降低复杂度，长序列后可能超过 FlashAttention |
| **OpenAI Sparse Attention** | 中短序列开销较高 | 稀疏理论复杂度低，但实际实现开销明显 |

- **左图中的 Crossover Points**
  - 图中用红圈和红色箭头标出 **Crossover Points**。
  - 含义是：
    - 在短序列和中等序列长度下，**FlashAttention 比许多 approximate attention 更快**。
    - 当序列长度继续增大后，某些理论复杂度更低的近似方法，如 **Linformer Attention**，可能开始在 runtime 上超过 dense **FlashAttention**。
  - 这说明论文的一个重要观点：
    - **FLOPs 少不等于实际更快**。
    - 对 GPU 而言，**HBM IO cost** 往往比理论 FLOPs 更决定 wall-clock runtime。

- **左图的主要结论**
  - **FlashAttention 在常见序列长度下显著快于 PyTorch Attention 和 Megatron Attention**。
  - **Block-Sparse FlashAttention 是图中整体最快的方法**。
  - 近似 attention 方法在很长序列时有优势，但在短中序列上，由于实现和 IO 开销，不一定快。
  - 该图支撑论文观点：**IO-aware exact attention 可以在实际速度上击败许多 approximate attention 方法**。

- **右图：Attention Memory Usage 分析**
  - 横轴是 **Sequence Length**，范围从约 **256 到 64K**。
  - 纵轴是 **Memory Footprint(GB)**。
  - 右图重点展示 attention 实现的显存占用随序列长度增长的趋势。

- **右图核心趋势**

| 方法 | 显存增长趋势 | 关键观察 |
|---|---|---|
| **PyTorch Attention** | 急剧增长，很快超出图中范围 | 因为需要存储 $N \times N$ attention matrix，显存复杂度近似 $O(N^2)$ |
| **FlashAttention** | 线性增长 | 不保存完整 attention matrix，只保存输出和 softmax statistics |
| **Block-Sparse FlashAttention** | 线性增长，几乎与 FlashAttention 重合 | 稀疏计算降低 runtime，但 memory footprint 与 FlashAttention 接近 |
| **Linformer Attention** | 线性增长，但常数较大 | 虽然是 approximate linear attention，但实际显存占用高于 FlashAttention |
| **OpenAI Sparse Attention** | 只在较短范围显示 | 长序列下受限于实现或内存 |

- **右图中的 20x 与 2x 标注**
  - 左侧 **20x**：
    - 表示在较短到中等序列长度附近，**FlashAttention 相比标准 exact attention 可节省最高约 20× memory**。
    - 主要原因是 FlashAttention 避免存储完整 $N \times N$ attention matrix。
  - 右侧 **2x**：
    - 表示在 **64K sequence length** 附近，**FlashAttention 仍比 Linformer Attention 更省约 2× memory**。
    - 这很关键，因为 Linformer 是近似线性 attention，而 FlashAttention 是 **exact attention**。

- **右图的主要结论**
  - **FlashAttention 的显存占用随 sequence length 线性增长**。
  - **标准 PyTorch Attention 的显存占用随 sequence length 二次增长**，长序列下不可扩展。
  - **FlashAttention 在保持 exact attention 的同时，比一些 approximate attention 更省显存**。
  - **Block-Sparse FlashAttention 继承了 FlashAttention 的线性显存优势，同时进一步提升速度**。

- **两幅图合并解读**

| 维度 | 标准 Attention | FlashAttention | Block-Sparse FlashAttention |
|---|---|---|---|
| 是否 exact | **是** | **是** | 否，block-sparse approximate |
| 是否物化 $N \times N$ attention matrix | **是** | **否** | **否** |
| Runtime | 长序列下迅速恶化 | 显著降低 | 最快 |
| Memory | $O(N^2)$ | **近似 $O(N)$** | **近似 $O(N)$** |
| 核心优化 | 常规矩阵乘法 / kernel fusion | **tiling + recomputation + IO-awareness** | **tiling + sparsity + IO-awareness** |
| 适用场景 | 短序列 | 中长序列 exact attention | 超长序列或可接受稀疏近似的任务 |

- **与论文方法的对应关系**
  - 该图直接验证了论文中的两个核心设计：
    - **Tiling**
      - 将 Q、K、V 分块加载到 GPU on-chip SRAM。
      - 避免把完整 $N \times N$ attention matrix 写入 HBM。
    - **Recomputation**
      - backward pass 不读取保存好的 attention matrix。
      - 只保存 softmax normalization statistics，例如 $m$ 和 $\ell$。
      - 反向传播时在 SRAM 中重新计算局部 attention block。
  - 因此 FlashAttention 虽然可能增加部分 FLOPs，但大幅减少 **HBM reads/writes**，实际运行更快。

- **图中体现的关键论文观点**
  - **Attention 的瓶颈不只是 FLOPs，而是 IO。**
  - **标准 attention 慢的关键原因是反复读写巨大 attention matrix。**
  - **FlashAttention 通过 IO-aware algorithm，把计算留在 SRAM 中完成，避免 HBM 中间结果读写。**
  - **实际 wall-clock speedup 来自减少内存访问，而不是降低理论计算复杂度。**

- **实验意义**
  - 这张图说明 FlashAttention 不是单纯的理论优化，而是针对现代 GPU 架构的实际性能优化。
  - 对 A100 等 GPU 来说：
    - HBM 带宽远低于 SRAM 带宽。
    - 许多 attention 操作是 **memory-bound**。
    - 减少 HBM IO 会直接带来 runtime 和 memory footprint 改善。
  - 因此，FlashAttention 能在不改变模型数学定义的情况下：
    - **加速训练**
    - **降低显存占用**
    - **支持更长上下文**
    - **保持 exact attention 精度**

- **最终结论**
  - 该图清楚展示了 **FlashAttention 的核心优势：在保持 exact attention 的同时，同时实现更快 runtime 和更低 memory usage**。
  - **Block-Sparse FlashAttention** 进一步说明，将 IO-aware 设计与稀疏结构结合，可以在超长序列任务中获得更强扩展性。
  - 该图也是论文论证“**IO-awareness 是高效 attention 的关键原则**”的核心实验证据之一。

### d35a1d471ce092ea04d538ae3cd9cddb63d49efed5c090144b2995332511f530.jpg

![d35a1d471ce092ea04d538ae3cd9cddb63d49efed5c090144b2995332511f530.jpg](d35a1d471ce092ea04d538ae3cd9cddb63d49efed5c090144b2995332511f530.jpg)

- **图像内容概览**
  - 该图展示了 **GPT-2 small** 与 **GPT-2 medium** 在 OpenWebText 验证集上的 **validation perplexity** 随 **training steps** 变化的曲线。
  - 对比对象包括：
    - **GPT-2-small HuggingFace**
    - **GPT-2-small FlashAttention**
    - **GPT-2-medium HuggingFace**
    - **GPT-2-medium FlashAttention**
  - 横轴为 **Training steps**，范围约为 0 到 400k。
  - 纵轴为 **Val perplexity**，范围约为 10 到 30。
  - 图例位于图像上方偏右区域。

- **核心结论**
  - **FlashAttention 与 HuggingFace 标准实现的验证困惑度曲线几乎完全重合**。
  - 这说明 FlashAttention 在替换标准 attention 实现后，**不会改变模型训练动态或最终模型质量**。
  - FlashAttention 的收益主要来自 **更快的 wall-clock training time** 和 **更低的 memory footprint**，而不是改变模型结构或优化目标。
  - 对 GPT-2 small 和 GPT-2 medium 均可观察到：**同一模型规模下，FlashAttention 与 HuggingFace 达到几乎相同的 validation perplexity**。

- **曲线对应关系**

| 曲线 | 颜色 | 模型 | Attention 实现 | 观察结果 |
|---|---:|---|---|---|
| GPT-2-small HuggingFace | 蓝色 | GPT-2 small | HuggingFace 标准实现 | 与 FlashAttention small 曲线几乎重合 |
| GPT-2-small FlashAttention | 橙色 | GPT-2 small | FlashAttention | 与 HuggingFace small 曲线几乎重合 |
| GPT-2-medium HuggingFace | 绿色 | GPT-2 medium | HuggingFace 标准实现 | 与 FlashAttention medium 曲线几乎重合 |
| GPT-2-medium FlashAttention | 红色 | GPT-2 medium | FlashAttention | 与 HuggingFace medium 曲线几乎重合 |

- **GPT-2 small 曲线分析**
  - **GPT-2 small** 的两条曲线为蓝色和橙色。
  - 两者从训练初期较高的 perplexity 快速下降，随后进入缓慢下降阶段。
  - 在大约 **100k steps** 附近，validation perplexity 约为 **20.5–21**。
  - 在接近训练末期时，validation perplexity 约为 **18.2** 左右。
  - 蓝色与橙色曲线几乎完全重叠，说明：
    - **FlashAttention 不影响 GPT-2 small 的收敛轨迹**。
    - **数值稳定性与 HuggingFace 标准实现一致**。
    - 最终 perplexity 与论文表 2 中 GPT-2 small 的结果一致：约 **18.2**。

- **GPT-2 medium 曲线分析**
  - **GPT-2 medium** 的两条曲线为绿色和红色。
  - 曲线下降速度明显快于 GPT-2 small，最终 perplexity 也更低。
  - 在大约 **100k steps** 附近，validation perplexity 约为 **16–16.5**。
  - 在训练末期，validation perplexity 约为 **14.2–14.3**。
  - 绿色与红色曲线也高度重合，说明：
    - **FlashAttention 不改变 GPT-2 medium 的训练行为**。
    - **模型质量与 HuggingFace 标准实现一致**。
    - 与论文表 2 中 GPT-2 medium 的最终 perplexity 结果一致：约 **14.2–14.3**。

- **模型规模对 perplexity 的影响**

| 模型 | 训练末期 validation perplexity | 相对表现 |
|---|---:|---|
| GPT-2 small | 约 **18.2** | 较高 perplexity，模型容量较小 |
| GPT-2 medium | 约 **14.2–14.3** | 更低 perplexity，模型容量更强 |

- **关键观察**
  - **GPT-2 medium 明显优于 GPT-2 small**：
    - medium 曲线整体位于 small 曲线下方。
    - 说明更大的模型容量带来更好的语言建模能力。
  - **FlashAttention 与 HuggingFace 曲线重合**：
    - small 模型中蓝色与橙色几乎重合。
    - medium 模型中绿色与红色几乎重合。
    - 这验证了 FlashAttention 是一种 **exact attention implementation**，不是近似 attention。
  - **训练早期下降最快**：
    - 所有曲线在初期快速下降。
    - 之后进入长尾阶段，perplexity 缓慢改善。
  - **没有观察到 FlashAttention 导致训练不稳定**：
    - 曲线平滑。
    - 无明显震荡、发散或系统性偏移。
    - 说明其 tiling、online softmax 和 backward recomputation 没有破坏数值行为。

- **与论文主张的关系**
  - 该图主要支撑论文中的一个重要实验结论：**FlashAttention 只改变 attention 的实现方式，不改变模型定义，因此可以保持相同的模型质量**。
  - 在论文实验中，FlashAttention 相比 HuggingFace 和 Megatron-LM 显著缩短训练时间：
  
| 模型 | HuggingFace 训练时间 | Megatron-LM 训练时间 | FlashAttention 训练时间 | 最终 perplexity |
|---|---:|---:|---:|---:|
| GPT-2 small | **9.5 days** | **4.7 days** | **2.7 days** | **18.2** |
| GPT-2 medium | **21.0 days** | **11.5 days** | **6.9 days** | **14.3** |

- **图像传达的实验意义**
  - 该图并不是为了证明 FlashAttention 降低 perplexity，而是为了证明：
    - **FlashAttention 加速训练但不牺牲精度**。
    - **FlashAttention 与标准 attention 在数值上等价或近似完全一致**。
    - **训练曲线一致性说明其 backward recomputation 是可靠的**。
  - 这对系统优化类论文非常关键，因为如果加速方法改变了收敛曲线，就可能意味着其并非严格等价实现。

- **技术解释**
  - FlashAttention 使用 **tiling** 避免显式存储完整的 $N \times N$ attention matrix。
  - 使用 **online softmax normalization** 分块计算 softmax，同时保持数值稳定。
  - backward pass 中通过保存 softmax 统计量并重新计算 attention block，避免从 HBM 读取巨大中间矩阵。
  - 因为它计算的是 **exact attention**，所以理论上应与标准 attention 产生相同训练行为。
  - 图中的曲线重合正是这一点的经验验证。

- **对数值稳定性的启示**
  - Softmax 是 attention 中最容易出现数值误差的部分。
  - FlashAttention 使用 row-wise max 和 normalization statistics，即 **m** 与 **ℓ**，保证分块 softmax 与全量 softmax 等价。
  - 图中没有出现曲线偏移，说明：
    - **online softmax 的数值稳定性良好**。
    - **recomputation 不会累积明显误差**。
    - **mixed precision training 下仍可稳定运行**。

- **综合评价**
  - 该图清楚表明：**FlashAttention 在 GPT-2 small 和 GPT-2 medium 上保持了与 HuggingFace 实现几乎完全一致的 validation perplexity 曲线**。
  - 因此，FlashAttention 的优势不是通过近似或牺牲模型质量获得的，而是通过 **IO-aware algorithm design**、**kernel fusion**、**SRAM tiling** 和 **HBM access reduction** 获得的。
  - 该图是论文中证明 FlashAttention “**fast but exact**” 的关键证据之一。

### 37d17f02fd68ec902f4ed41bfb64ef4fbc9d2c0f2ef8ca18013296e679204c5a.jpg

![37d17f02fd68ec902f4ed41bfb64ef4fbc9d2c0f2ef8ca18013296e679204c5a.jpg](37d17f02fd68ec902f4ed41bfb64ef4fbc9d2c0f2ef8ca18013296e679204c5a.jpg)

- **图像主题**：该图展示了 **FlashAttention 在 A100 GPU 上相对于标准 PyTorch attention 的加速比**，横轴为 **Sequence Length**，纵轴为 **Speedup（X times faster）**。

- **实验设置解读**：
  - 硬件：**NVIDIA A100**
  - 对比基线：**standard PyTorch attention**
  - 任务：不同序列长度下的 attention 前向+反向整体加速表现
  - 配置变量：
    - **Dropout + Masking**
    - **Masking Only**
    - **No Masking, No Dropout**

- **图中主要数据趋势**：

| Sequence Length | Dropout + Masking | Masking Only | No Masking, No Dropout |
|---:|---:|---:|---:|
| 128 | 约 **2.25×** | 约 **2.6×** | 约 **2.05×** |
| 256 | 约 **2.2×** | 约 **2.7×** | 约 **2.15×** |
| 512 | 约 **4.05×** | 约 **3.9×** | 约 **2.5×** |
| 1024 | 约 **4.05×** | 约 **3.8×** | 约 **2.2×** |
| 2048 | 约 **4.2×** | 约 **4.1×** | 约 **2.05×** |
| 4096 | 约 **4.3×** | 约 **4.25×** | 约 **2.05×** |

- **核心观察**：
  - **FlashAttention 在所有序列长度上均显著快于 PyTorch attention**。
  - 当存在 **Masking** 或 **Dropout + Masking** 时，加速比明显更高，尤其在 **Sequence Length ≥ 512** 后稳定达到 **约 4×**。
  - 在 **No Masking, No Dropout** 条件下，加速比相对较低，大多维持在 **约 2×–2.5×**。
  - 说明 FlashAttention 的优势不仅来自 attention 主计算本身，还来自对 **masking、dropout、softmax、matrix multiplication 的 kernel fusion** 和 **HBM IO 减少**。

- **不同配置对比**：

| 配置 | 加速表现 | 原因分析 |
|---|---|---|
| **Dropout + Masking** | 最稳定，长序列约 **4×+** | FlashAttention 将 dropout、masking、softmax、矩阵乘融合，减少多次 HBM 读写 |
| **Masking Only** | 与 Dropout + Masking 接近，部分短序列更高 | masking 在 PyTorch 中通常引入额外 memory-bound 操作，FlashAttention 融合后收益明显 |
| **No Masking, No Dropout** | 约 **2×** 左右 | 少了可融合的额外 elementwise 操作，主要收益来自避免 materialize attention matrix |

- **序列长度影响**：
  - **128–256**：
    - 加速比约 **2×–2.7×**。
    - 短序列下 attention matrix 较小，PyTorch 的 HBM IO 压力还不极端，因此 FlashAttention 优势有限。
  - **512**：
    - 出现明显跃升，**Dropout + Masking** 达到约 **4×**。
    - 这是因为标准 attention 开始受到 **N×N attention matrix** 的显著内存访问开销影响。
  - **1024–4096**：
    - FlashAttention 在带 Masking 的场景下保持 **约 4×–4.3×**。
    - 表明随着序列增长，FlashAttention 的 **IO-aware tiling** 优势更加稳定。

- **为什么 Masking / Dropout 场景加速更明显**：
  - 标准 PyTorch attention 通常会产生多个中间矩阵：
    - **S = QKᵀ**
    - **masked S**
    - **P = softmax(S)**
    - **dropout(P)**
    - **O = PV**
  - 这些中间结果需要反复读写 **HBM**。
  - FlashAttention 在一个 CUDA kernel 内完成：
    - **QKᵀ block computation**
    - **masking**
    - **softmax**
    - **dropout**
    - **PV**
  - 因此避免了大量 **N² 级别 HBM traffic**。

- **与论文核心论点的关系**：
  - 该图直接验证了论文主张：**attention 的实际瓶颈不仅是 FLOPs，而是 HBM memory access**。
  - FlashAttention 没有改变 exact attention 的数学结果，但通过 **tiling** 和 **recomputation** 避免显式存储 attention matrix。
  - 因此即使 FLOPs 没有降低，甚至 backward 中有 recomputation，整体 wall-clock runtime 仍显著下降。

- **从 IO 复杂度角度解释**：

| 方法 | HBM 访问复杂度 | 实际含义 |
|---|---|---|
| Standard Attention | **Θ(Nd + N²)** | 需要写入/读取完整 attention matrix |
| FlashAttention | **Θ(N²d² / M)** | 利用 SRAM tiling，减少 HBM 访问 |
| 图中体现 | 长序列加速更明显 | N 增大时，避免 N² HBM 访问收益扩大 |

- **图像传达的工程结论**：
  - **FlashAttention 对中长序列尤其有效**。
  - 如果模型中包含 **causal mask、padding mask、dropout**，FlashAttention 的收益更大。
  - 对于 A100 这类高性能 GPU，即便 HBM 带宽很高，attention 仍然明显受限于 memory IO。
  - FlashAttention 通过更好地使用 **on-chip SRAM**，将原本 memory-bound 的 attention 计算变得更高效。

- **局限性解读**：
  - 在 **No Masking, No Dropout** 场景下，加速比低于带 masking/dropout 的场景，说明 kernel fusion 的附加收益减少。
  - 序列长度较短时，PyTorch baseline 的绝对开销较小，FlashAttention 的优势不如长序列显著。
  - 图中最高只展示到 **4096**，未体现更长序列下 memory footprint 的巨大优势；但论文其他实验显示 FlashAttention 可扩展到更长上下文。

- **总体结论**：
  - 该图清晰表明：在 **A100 GPU** 上，FlashAttention 相比标准 PyTorch attention 通常可获得 **约 2×–4.3×** 的加速。
  - 对含 **Masking / Dropout** 的真实 Transformer 训练场景，FlashAttention 的加速最明显，长序列下稳定达到 **约 4×**。
  - 图像有力支持论文的核心观点：**IO-aware algorithm design 是提升 Transformer attention 实际速度的关键，而不仅仅是减少 FLOPs**。

### 6ae80dc33e87470816bbc495deb6ec5d91634f70e06b2d4a8136a6333fab22c7.jpg

![6ae80dc33e87470816bbc495deb6ec5d91634f70e06b2d4a8136a6333fab22c7.jpg](6ae80dc33e87470816bbc495deb6ec5d91634f70e06b2d4a8136a6333fab22c7.jpg)

- 这张图展示的是 **FlashAttention 在 A100 GPU、head dimension = 128** 时，相对 **标准 PyTorch attention** 的加速比，横轴是 **sequence length**，纵轴是 **Speedup（越大越快）**。
- 图中有 4 组条件：
  - **Dropout + Padding Masking**：蓝色
  - **Padding Masking Only**：橙色
  - **Causal Mask**：红色
  - **No Masking, No Dropout**：绿色
- 总体上，这张图说明了一个很重要的结论：**FlashAttention 的收益不仅取决于 sequence length，也强烈依赖 attention 的算子形态**。  
  - 对 **causal mask** 的收益最大
  - 对 **无 mask、无 dropout** 的收益最小
  - 在 **head dim = 128** 这种更大的维度下，FlashAttention 的优势比 d=64 时更受限制

- 先给出图中大致读数（近似值）：

| Sequence Length | Dropout + Padding Masking | Padding Masking Only | Causal Mask | No Masking, No Dropout |
|---|---:|---:|---:|---:|
| 128 | **2.15x** | **2.38x** | **2.52x** | **2.06x** |
| 256 | **2.18x** | **2.13x** | **2.43x** | **1.85x** |
| 512 | **1.88x** | **1.94x** | **2.50x** | **1.32x** |
| 1024 | **1.86x** | **1.95x** | **2.90x** | **1.15x** |
| 2048 | **1.88x** | **2.00x** | **3.20x** | **1.03x** |

- 从趋势看，可以分成 4 个结论：
  - **Causal Mask 最强，且随长度增长继续变强**
    - 从约 **2.5x** 增长到 **3.2x**
    - 说明在 causal 场景下，FlashAttention 更能发挥其 **tiling + IO reduction** 的优势
    - 也意味着标准 attention 在 causal 场景下的内存访问和 kernel 效率更差，FlashAttention 的相对优势更明显
  - **Padding Masking Only 和 Dropout + Padding Masking 维持在 2x 左右**
    - 整体比较稳定
    - 说明 FlashAttention 对这些常见训练算子有很好的 **kernel fusion** 效果
    - 它不仅减少 HBM 读写，还把 mask / dropout / softmax / matmul 这些步骤融合进单个 kernel
  - **No Masking, No Dropout 的收益最弱，并且随着长度增长快速下降**
    - 从约 **2.1x** 下降到接近 **1.0x**
    - 表明在这个配置下，标准 PyTorch attention 已经相对更接近高效实现，FlashAttention 的额外 IO 优势被削弱
    - 也说明 **FlashAttention 的加速不是“无条件”的**，而是对具体算子路径高度敏感
  - **head dimension = 128 会压缩加速比**
    - 论文正文也提到：维度变大后，每个 block 更占 SRAM，需要更小 block size
    - 这会带来更多 block 轮转和更多开销
    - 所以这张图里的加速比整体比 d=64 情况更保守

- 这张图最值得注意的地方是 **红色柱子（Causal Mask）明显最高**，这是非常符合 FlashAttention 设计逻辑的：
  - causal attention 有天然的结构稀疏性
  - FlashAttention 可以在 block 层面更好地跳过无效计算/减少访存
  - 标准实现往往仍然会付出较多的中间矩阵读写代价
  - 所以在长序列下，**FlashAttention 的 IO-aware 优势被进一步放大**

- 另一个关键信号是 **绿色柱子在长序列上接近 1x**：
  - 这表示当没有 mask、也没有 dropout 时，FlashAttention 在这个设置下对标准 PyTorch attention 的优势不再明显
  - 这从侧面说明：
    - **FlashAttention 的核心价值主要在于减少 HBM 访问**
    - 如果 baseline 本身已经较高效，或者额外融合收益有限，那么 speedup 就会收缩
  - 这也解释了论文为何强调：**“HBM access 才是主导 runtime 的关键因素”**

- 从论文主张角度看，这张图是一个很强的实证支持：
  - **不是单纯减少 FLOPs 才能提速**
  - 真正决定 wall-clock time 的是 **memory hierarchy / IO behavior**
  - FlashAttention 通过把 attention 变成 **block-wise streamed computation**，实现了：
    - 更少 HBM 访问
    - 更少中间张量 materialization
    - 更好的 kernel fusion
    - 更稳定的实际训练加速

- 如果用一句话概括这张图：
  - **FlashAttention 在 A100 上对常见 attention 变体普遍有 1.8x–3.2x 左右加速，其中 causal mask 场景收益最大，而无 mask、无 dropout 场景收益最弱。**
- 如果你愿意，我还可以进一步把这张图和论文里的 **Figure 5/7/8** 做横向对比，解释为什么 **d=128** 的曲线和 **d=64** 不同。

### 7f94258e436b99b1a20a26393f436d4f131ea85237ecb55f5b7d241830254bf7.jpg

![7f94258e436b99b1a20a26393f436d4f131ea85237ecb55f5b7d241830254bf7.jpg](7f94258e436b99b1a20a26393f436d4f131ea85237ecb55f5b7d241830254bf7.jpg)

- **图片类型**：分组柱状图，用于展示 **FlashAttention 相对标准 PyTorch attention 的加速比**。

- **图中主题**：不同 **Sequence Length** 下，FlashAttention 在 **GTX/RTX 3090** GPU 上的速度提升情况。

- **坐标轴含义**：

  | 元素 | 含义 |
  |---|---|
  | 横轴 | **Sequence Length**，序列长度，取值为 **128、256、512、1024、2048** |
  | 纵轴 | **Speedup (X times faster)**，相对基线的加速倍数 |
  | 基线 | 标准 **PyTorch Attention** 实现 |
  | 对比方法 | **FlashAttention** |
  | 硬件 | 图题写作 **GTX 3090**，论文正文称为 **RTX 3090**，应视为同一类消费级 NVIDIA GPU 环境 |

- **图例说明**：

  | 颜色 | 配置 | 含义 |
  |---|---|---|
  | 蓝色 | **Dropout + Masking** | 同时启用 dropout 和 masking |
  | 橙色 | **Masking Only** | 仅启用 masking |
  | 红色 | **No Masking, No Dropout** | 不启用 masking，也不启用 dropout |

- **近似读数**：

  | Sequence Length | Dropout + Masking | Masking Only | No Masking, No Dropout |
  |---:|---:|---:|---:|
  | 128 | **约 2.9×** | **约 3.5×** | **约 2.7×** |
  | 256 | **约 3.6×** | **约 3.4×** | **约 2.3×** |
  | 512 | **约 4.0×** | **约 3.6×** | **约 2.4×** |
  | 1024 | **约 4.3×** | **约 4.0×** | **约 2.3×** |
  | 2048 | **约 4.6×** | **约 4.3×** | **约 2.5×** |

- **主要趋势**：

  | 观察点 | 分析 |
  |---|---|
  | **加速比随序列长度增大整体上升** | 从 128 到 2048，FlashAttention 的优势更明显，尤其在启用 masking/dropout 时 |
  | **Dropout + Masking 通常最快** | 在 256 及以上长度时，蓝色柱普遍最高，最高达到约 **4.6×** |
  | **Masking Only 表现也很强** | 橙色柱基本维持在 **3.4×–4.3×**，说明 masking 场景下 kernel fusion 和 IO 优化收益明显 |
  | **No Masking, No Dropout 加速较低** | 红色柱稳定在约 **2.3×–2.7×**，说明在最简单 attention 场景中，FlashAttention 仍有优势，但收益小于复杂操作场景 |
  | **长序列收益更突出** | 2048 长度时，FlashAttention 在带 dropout/masking 的场景达到最高加速 |

- **核心结论**：

  | 结论 | 解释 |
  |---|---|
  | **FlashAttention 在 RTX/GTX 3090 上显著快于 PyTorch Attention** | 所有配置、所有序列长度均超过 **2×** 加速 |
  | **复杂 attention 操作越多，FlashAttention 优势越大** | dropout、masking 等操作在标准实现中会引入额外 HBM 读写；FlashAttention 通过 kernel fusion 减少 IO |
  | **加速主要来自 IO-aware 设计，而非减少 FLOPs** | FlashAttention 仍计算 exact attention，但避免显式 materialize 大型 attention matrix |
  | **消费级 GPU 上收益更明显** | RTX 3090 的 HBM/GDDR 带宽低于 A100，因此减少 memory access 带来的相对收益更高 |

- **与论文主旨的关系**：

  | 论文观点 | 图中证据 |
  |---|---|
  | **Attention 性能瓶颈主要来自 HBM access** | FlashAttention 减少 HBM 读写后，在 3090 上获得 **2.3×–4.6×** 加速 |
  | **Kernel fusion 能显著优化 dropout/masking 场景** | 蓝色和橙色柱明显高于红色柱 |
  | **FlashAttention 对长序列更有效** | Sequence Length 从 128 增至 2048，加速比整体提升 |
  | **IO-aware algorithm 比单纯 FLOPs 优化更关键** | 即使计算量没有降为线性，实际 wall-clock speedup 仍然显著 |

- **为什么 Dropout + Masking / Masking Only 更快**：

  | 原因 | 说明 |
  |---|---|
  | **标准 PyTorch Attention 需要多次读写中间矩阵** | 包括 QKᵀ、mask 后结果、softmax 输出、dropout mask 等 |
  | **FlashAttention 避免写出 N × N attention matrix** | 中间结果主要留在 SRAM 中处理 |
  | **多个操作被融合到一个 CUDA kernel 中** | 减少 kernel launch overhead 和 HBM traffic |
  | **Masking/dropout 越多，标准实现越吃亏** | 因此 FlashAttention 的相对加速更高 |

- **为什么 No Masking, No Dropout 加速较低**：

  | 原因 | 说明 |
  |---|---|
  | **标准实现中的额外操作更少** | 没有 mask/dropout 时，PyTorch baseline 的冗余 HBM 访问相对少一些 |
  | **FlashAttention 的 recomputation 有额外 FLOPs** | 后向传播中会重算部分 attention block，虽然减少 IO，但也带来计算开销 |
  | **因此加速稳定但不极端** | 红色柱维持在约 **2.3×–2.7×** |

- **序列长度维度分析**：

  | Sequence Length | 现象 | 解读 |
  |---:|---|---|
  | **128** | 加速已明显，但不同配置差异较大 | 短序列下 kernel overhead 和固定开销影响较明显 |
  | **256** | Dropout + Masking 提升到约 3.6× | IO 优化开始发挥更大作用 |
  | **512** | 蓝色达到约 4× | attention matrix 规模变大，避免 N² HBM 访问收益增加 |
  | **1024** | 蓝色约 4.3×，橙色约 4.0× | FlashAttention 对中长序列优势明显 |
  | **2048** | 蓝色约 4.6×，橙色约 4.3× | 长序列下标准 attention 的 HBM bottleneck 更严重 |

- **图像中的细节问题**：

  | 问题 | 分析 |
  |---|---|
  | **标题写作 GTX 3090** | NVIDIA 常见型号是 **RTX 3090**，论文正文也写 **RTX 3090**，图题可能存在标注误差 |
  | **未显示误差线** | 图中展示的是平均或单次 benchmark 结果，无法判断方差 |
  | **只展示到 2048** | 该图主要说明中等序列长度下的加速，不覆盖 4096 及以上长序列 |
  | **基线未在图中直接标出** | 加速比默认相对标准 PyTorch attention，1× 表示无加速 |

- **整体评价**：

  | 维度 | 评价 |
  |---|---|
  | **性能收益** | 非常显著，最高约 **4.6×** |
  | **趋势一致性** | 清晰，序列越长、操作越复杂，加速越明显 |
  | **对论文论点支持力度** | 很强，直接证明 FlashAttention 的 **IO-aware** 优化在实际 GPU 上有效 |
  | **工程意义** | 对使用 RTX 3090 等消费级 GPU 训练 Transformer 的场景尤其有价值 |
  | **算法意义** | 说明 exact attention 也可以通过内存访问优化获得大幅 wall-clock speedup，无需牺牲模型精度 |

### 30a604bc17a343f8a531a854ce68df57c89b545def7f7dd04796df5ad669247a.jpg

![30a604bc17a343f8a531a854ce68df57c89b545def7f7dd04796df5ad669247a.jpg](30a604bc17a343f8a531a854ce68df57c89b545def7f7dd04796df5ad669247a.jpg)

- **图像内容概览**
  - 该图展示了 **FlashAttention 在 NVIDIA T4 GPU 上相对标准 PyTorch Attention 的加速比**。
  - 横轴是 **Sequence Length**，包含 **128、256、512、1024、2048**。
  - 纵轴是 **Speedup (X times faster)**，表示 FlashAttention 比 PyTorch Attention 快多少倍。
  - 每个序列长度下有三组柱状结果：
    - **Dropout + Masking**
    - **Masking Only**
    - **No Masking, No Dropout**

- **近似数据读取**

| Sequence Length | Dropout + Masking | Masking Only | No Masking, No Dropout |
|---:|---:|---:|---:|
| 128 | **约 3.75×** | **约 3.40×** | **约 2.70×** |
| 256 | **约 3.20×** | **约 2.90×** | **约 2.25×** |
| 512 | **约 3.18×** | **约 2.88×** | **约 1.85×** |
| 1024 | **约 3.30×** | **约 2.90×** | **约 1.60×** |
| 2048 | **约 3.15×** | **约 2.85×** | **约 1.50×** |

- **核心观察**
  - **FlashAttention 在 T4 上始终获得显著加速**，所有设置下均快于标准 PyTorch Attention。
  - **Dropout + Masking 场景加速最高**，大多保持在 **3.1×–3.8×**。
  - **Masking Only 场景次之**，大致稳定在 **2.8×–3.4×**。
  - **No Masking, No Dropout 场景加速最低**，并且随着序列长度增长，加速比明显下降，从约 **2.7×** 降至约 **1.5×**。

- **为什么 Dropout + Masking 加速最高**
  - 标准 PyTorch Attention 通常将计算拆成多个 kernel，例如：
    - **QKᵀ matrix multiplication**
    - **masking**
    - **softmax**
    - **dropout**
    - **PV matrix multiplication**
  - 这些步骤会频繁读写 GPU HBM，尤其是中间的 **N × N attention matrix**。
  - FlashAttention 通过 **kernel fusion** 将这些操作融合到一个 CUDA kernel 中，并使用 **tiling** 在 SRAM 中完成局部计算。
  - 因此，当存在 **Masking** 和 **Dropout** 这类额外 elementwise 操作时，FlashAttention 能避免更多 HBM 往返读写，收益更大。
  - 所以图中 **Dropout + Masking > Masking Only > No Masking, No Dropout** 的加速顺序非常符合论文的 IO-aware 设计逻辑。

- **随序列长度变化的趋势**
  - **Dropout + Masking**
    - 加速比整体较稳定，约在 **3×以上**。
    - 说明 FlashAttention 在包含复杂 attention 后处理操作时，对 T4 仍具有稳定优势。
  - **Masking Only**
    - 加速比也比较稳定，约 **2.85×–3.4×**。
    - 相比 Dropout + Masking 少了一项可融合操作，因此加速略低。
  - **No Masking, No Dropout**
    - 加速比随序列长度增加明显下降。
    - 从 **128 token 的约 2.7×** 降至 **2048 token 的约 1.5×**。
    - 说明在没有额外 elementwise 操作可融合时，FlashAttention 的优势主要来自减少 attention matrix 的 HBM IO；但在 T4 上，较小 SRAM 限制了 tile size，使长序列下收益受限。

- **与硬件特性的关系**
  - T4 GPU 的 **on-chip SRAM 较小**，相比 A100 能容纳的 tile 更小。
  - FlashAttention 的 IO complexity 近似为：

| 方法 | HBM 访问复杂度 |
|---|---|
| Standard Attention | **Θ(Nd + N²)** |
| FlashAttention | **Θ(N²d² / M)** |

  - 其中：
    - **N** 是 sequence length
    - **d** 是 head dimension
    - **M** 是 SRAM size
  - 当 **M 较小**时，FlashAttention 需要更多 block passes，HBM 访问减少幅度不如 A100 明显。
  - 因此论文也指出：**T4 上的加速低于 A100 或 RTX 3090**，这与图中结果一致。

- **与论文主张的对应关系**
  - 图像验证了论文的一个重要观点：**Attention 的实际瓶颈不只是 FLOPs，而是 HBM IO**。
  - FlashAttention 并没有改变 exact attention 的理论计算复杂度，仍然是 **O(N²d)** FLOPs。
  - 但它通过避免显式 materialize **N × N attention matrix**，显著减少 HBM 读写。
  - 在包含 **masking/dropout** 的真实训练场景下，IO savings 与 kernel fusion 共同带来更高 wall-clock speedup。

- **对不同场景的解释**
  - **训练场景**
    - 通常包含 **dropout** 和 **masking**。
    - 因此图中最相关的是 **Dropout + Masking** 曲线。
    - 该场景下 FlashAttention 在 T4 上约 **3×以上加速**，说明即使在中端推理/训练 GPU 上也有明显收益。
  - **推理场景**
    - 推理一般不使用 dropout，但可能使用 **causal mask** 或 padding mask。
    - 因此 **Masking Only** 更接近常见推理设置。
    - 图中显示仍有约 **2.8×–3.4×** 加速。
  - **纯 attention benchmark**
    - 如果不使用 masking/dropout，FlashAttention 仍然更快，但优势随序列长度变弱。
    - 这说明 benchmark 是否包含真实模型中的 mask/dropout 会显著影响性能结论。

- **关键结论**
  - **FlashAttention 在 T4 上对标准 PyTorch Attention 有稳定加速。**
  - **包含 Dropout 和 Masking 时加速最大，最高接近 3.8×。**
  - **Masking Only 场景下加速也稳定接近 3×。**
  - **无 Masking、无 Dropout 时，加速随序列长度增加下降，2048 时约为 1.5×。**
  - **该图支持论文核心观点：IO-aware 设计与 kernel fusion 是 FlashAttention 获得实际 wall-clock speedup 的关键。**

### b6db8aaa18cd6fa136ae36fc6f601365e8b393a3a23bc15cae5d4eed19bf34af.jpg

![b6db8aaa18cd6fa136ae36fc6f601365e8b393a3a23bc15cae5d4eed19bf34af.jpg](b6db8aaa18cd6fa136ae36fc6f601365e8b393a3a23bc15cae5d4eed19bf34af.jpg)

- 这是一张 **FlashAttention 在 T4 GPU 上的 forward-only 加速比** 对比图，纵轴是 **Speedup（越高越快）**，横轴是 **Sequence Length**。
- 图中三组设置分别是：
  - **Dropout + Masking**（蓝色）
  - **Masking Only**（橙色）
  - **No Masking, No Dropout**（红色）
- 总体结论很明确：**FlashAttention 在所有序列长度下都能带来稳定加速**，且在 **T4 这种 SRAM 更小、HBM 带宽更有限的 GPU** 上，速度优势仍然明显。

- 近似读数如下（根据柱状图估算）：

| Sequence Length | Dropout + Masking | Masking Only | No Masking, No Dropout |
|---|---:|---:|---:|
| 128  | 2.9× | 3.5× | 2.7× |
| 256  | 3.6× | 3.4× | 2.3× |
| 512  | 4.0× | 3.6× | 2.4× |
| 1024 | 4.3× | 4.0× | 2.3× |
| 2048 | 4.6× | 4.4× | 2.5× |

- 主要趋势分析：
  - **序列长度越长，加速越明显**，尤其是 **Dropout + Masking** 和 **Masking Only** 两组。
  - **No Masking, No Dropout** 的加速比相对更低，基本稳定在 **2.3×–2.6×**，说明当原始 attention 的附加元素操作减少时，FlashAttention 的优势主要来自 **减少 HBM 读写**，但额外收益会变小。
  - **Dropout + Masking** 通常是最强的一组，长序列下接近 **4.6×**，说明 **kernel fusion + IO 减少** 在真实训练场景里叠加效果更好。
  - **Masking Only** 也很强，在长序列处接近 **4.4×**，表明 **masking 相关的中间张量处理** 是传统 attention 的一大开销点。

- 这张图反映出的核心技术含义：
  - **FlashAttention 的收益不是单纯靠减少 FLOPs，而是靠减少 HBM traffic。**
  - 在 **T4** 上，由于硬件更偏 memory-bound，**IO-aware 设计** 的优势依然成立。
  - 带 **masking / dropout** 的场景里，传统实现通常需要处理更多中间结果，而 FlashAttention 把这些步骤融合进一个 kernel，因此加速更显著。

- 从工程角度看，这张图说明：
  - **FlashAttention 对训练场景比纯推理更有价值**，因为训练中经常同时存在 **masking 和 dropout**。
  - **长序列收益更大**，因此它特别适合长上下文 Transformer、长文档任务和大模型训练。
  - 即使在 **T4 这种并非顶级训练卡** 上，也能看到明显收益，说明方法具有较强的硬件适应性。

- 结合论文主线，这张图支持了两个关键论点：
  - **IO-awareness 比单纯 FLOP reduction 更重要。**
  - **FlashAttention 在不同硬件上都能稳定提升 wall-clock performance**，而且在“真实训练设置”下收益更突出。

- 一句话总结：**这张图证明了 FlashAttention 在 T4 上也能稳定提速，且随着序列变长、masking/dropout 更复杂，优势越明显，体现了其强 IO-aware 和 kernel fusion 能力。**

