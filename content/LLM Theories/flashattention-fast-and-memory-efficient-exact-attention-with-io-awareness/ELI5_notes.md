# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness 通俗讲解

### 0. 整体创新点通俗解读

**痛点直击：这篇论文真正想解决的不是“Attention算得太多”，而是“Attention搬数据搬得太蠢”**

- 过去大家看 Transformer 长序列慢，第一反应通常是：
  - **Attention是O(N²)**，所以要减少 FLOPs。
  - 于是出现大量 approximate attention：
    - sparse attention
    - low-rank attention
    - hashing attention
    - linear attention

- 但这篇论文的核心判断非常尖锐：
  - 很多方法在纸面上 FLOPs 降了，**真实GPU上并没有明显变快**。
  - 原因是它们只盯着“算多少”，没盯着“数据从哪里读、写到哪里”。

- GPU上有一个很现实的矛盾：
  - **HBM很大但慢**，比如几十 GB。
  - **SRAM很小但快**，比如每个 SM 只有约百 KB 量级。
  - 现代 GPU 的算力增长很快，但内存带宽没跟上。
  - 所以很多 Transformer 操作不是“算不动”，而是“数据搬来搬去拖死了”。

- 标准 Attention 最难受的地方在于：
  - 它会显式生成并保存一个巨大的 **N×N attention matrix**。
  - 计算流程大致是：
    - 读 **Q、K**，算出 **S=QKᵀ**，把 **S** 写回 HBM。
    - 再从 HBM 读 **S**，做 **softmax**，把 **P** 写回 HBM。
    - 再从 HBM 读 **P、V**，算 **O=PV**。
  - 问题不是只在于 **N×N** 大，而是这个大矩阵还要被**反复读写到慢内存**。

- 这就像你做饭：
  - 菜其实不复杂。
  - 但你每切一根葱，都跑到楼下仓库放一次，再跑回来取一次。
  - 最后不是厨艺慢，是**仓库往返把时间耗光了**。

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

---

**通俗比方：FlashAttention像“边翻书边做摘要”，而不是“先复印整本书再读”**

- 标准 Attention 的做法像这样：
  - 你要回答一本书里的问题。
  - 你先把整本书所有页和所有页之间的关系，做成一张巨大表格。
  - 然后把这张表存起来。
  - 再反复拿出来查。
  - 表格非常大，存取非常贵。

- FlashAttention 的做法更像：
  - 不复印整本书。
  - 每次只拿一小摞页到桌面上。
  - 当场算这一小摞页对答案的贡献。
  - 算完就更新手头的“摘要状态”。
  - 不保存那张巨大关系表。

- 这里的“桌面”就是 **SRAM**。
  - 小，但非常快。
  - 关键是让真正热的数据尽量留在桌面上处理。
  - 不要动不动把中间结果扔回 **HBM**。

- 更准确地说：
  - 标准 Attention 是：
    - **先完整造出attention matrix，再做softmax，再乘V**。
  - FlashAttention 是：
    - **把attention matrix切成小块，在SRAM里算完一块就消化一块**。
  - 它不是近似。
  - 它算出来的仍然是**exact attention**。

- 这个思路有点像经典算法里的 **blocked matrix multiplication / tiling**：
  - 不是改变数学问题。
  - 是改变执行顺序。
  - 让数据局部性变好。
  - 用更少的慢内存访问换取更快的真实运行时间。

---

**关键一招：作者没有改Attention公式，而是改了Attention的“执行顺序”和“保存策略”**

- 这篇论文最巧妙的地方是：
  - 它没有说“我们不要标准 Attention 了”。
  - 它也没有牺牲精度去近似。
  - 它说：
    - **Attention的数学定义不变。**
    - **但不要把N×N矩阵真的落到HBM里。**

- 作者具体扭转了两件事。

- 第一件事：把“一次性softmax”改成“分块累积softmax”
  - softmax通常看起来必须看到整行所有元素才能归一化。
  - 这也是为什么标准实现会倾向于先把整张 **S=QKᵀ** 算出来。
  - FlashAttention 利用一个简单但关键的事实：
    - softmax的最大值和归一化因子可以**分块维护**。
    - 每看到一块新的 K/V，就更新当前行的最大值、归一化因子和输出。
  - 换句话说：
    - 不需要把整行 attention score 都存下来。
    - 只要维护少量“统计量”，就能保证最后结果和完整 softmax 一模一样。

- 第二件事：把“保存巨大中间矩阵用于反向传播”改成“反向时在SRAM里重算”
  - 标准训练里，forward 会保存 **P=softmax(S)**，backward 时再读出来。
  - 但 **P** 是 **N×N**，非常占显存，也带来大量 HBM 访问。
  - FlashAttention 选择只保存：
    - 输出 **O**
    - softmax 的归一化统计量
    - 必要的随机状态，比如 dropout seed
  - backward 时：
    - 重新加载小块 **Q、K、V**
    - 在 SRAM 中重新算出当前块的 attention
    - 直接完成梯度计算
  - 这看似多算了一点 FLOPs。
  - 但在现代 GPU 上，**多算一点比多搬很多数据便宜**。

- 这就是论文的核心 trade-off：
  - 用少量 recomputation 换取大量减少 HBM IO。
  - 牺牲一点理论计算量。
  - 换来真实 GPU 上更快、更省显存。

---

**一句话抓住核心贡献**

- **FlashAttention不是一种新的Attention机制，而是一种新的Attention执行方式。**
- 它的核心贡献是：
  - 把标准 Attention 中最昂贵的 **N×N中间矩阵读写**消掉。
  - 用 **tiling** 让小块数据在 **SRAM** 中完成计算。
  - 用 **online softmax** 保证分块计算仍然精确。
  - 用 **recomputation** 避免 backward 保存巨大 attention matrix。
  - 最终得到 **exact attention**，但显存从近似 quadratic 降到近似 linear，真实训练速度显著提升。

---

**核心对比：标准Attention vs FlashAttention**

| 维度 | 标准Attention | FlashAttention |
|---|---|---|
| 数学结果 | exact attention | exact attention |
| 是否保存N×N attention matrix | 保存 | 不保存 |
| 主要瓶颈 | HBM读写巨大中间矩阵 | 更充分利用SRAM |
| 显存占用 | **O(N²)** 中间量 | **O(N)** 额外中间量 |
| FLOPs | 标准 | backward中有少量重算 |
| 真实GPU表现 | 长序列下慢且吃显存 | 更快、更省显存 |
| 核心思想 | 先算完整矩阵 | 分块计算、边算边归一化 |

---

**为什么这件事重要：它改变了“高效Transformer”的路线判断**

- 过去很多 efficient Transformer 的逻辑是：
  - **O(N²)太贵，所以必须近似。**
  - 例如把 attention 做 sparse、low-rank、kernelized。

- FlashAttention 给出的反驳是：
  - **在很多实际长度下，O(N²)不是唯一问题。**
  - 标准实现之所以慢，很大一部分是因为 IO 不友好。
  - 如果把 exact attention 的 IO 做好，它可以比很多 approximate attention 更快。

- 这点很有启发性：
  - 算法复杂度不是只看 FLOPs。
  - 在 GPU 上，还必须看：
    - 数据是否反复进出 HBM
    - kernel 是否融合
    - SRAM 是否充分利用
    - 中间结果是否 materialize

- 所以这篇论文真正推动的是一种观念：
  - **AI算法设计不能只IO-blind地写数学公式。**
  - 要把硬件内存层级纳入算法设计。
  - 这就是论文标题里的 **IO-Awareness**。

---

**实验结果的直观含义**

| 任务/场景 | 结果含义 |
|---|---|
| BERT-large | 相比 MLPerf 1.1 记录快约 **15%** |
| GPT-2 | 相比 HuggingFace/Megatron 实现最高约 **3×** 端到端训练加速 |
| LRA benchmark | 约 **2.4×** 训练加速 |
| Attention benchmark | 常见长度下比 PyTorch attention 最高约 **3×** |
| 显存 | 相比标准 exact attention 最高约 **20×** 更省 |
| 长上下文GPT-2 | context length 从 **1K** 扩到 **4K**，还比 Megatron 1K 更快，perplexity 更好 |
| Path-X / Path-256 | 让 Transformer 首次在这些超长序列任务上超过随机表现 |

- 注意这些结果的重点不是“某个 benchmark 赢了”。
- 重点是：
  - **exact attention本身还有巨大工程-算法联合优化空间。**
  - 不一定非要先牺牲模型质量去做近似。
  - 先把数据搬运方式改对，收益就很大。

---

**Block-Sparse FlashAttention：顺手把近似Attention也做快了**

- 作者还进一步做了 **block-sparse FlashAttention**。
- 直觉很简单：
  - FlashAttention 已经能高效处理一个个 attention block。
  - 如果某些 block 本来就是 mask 掉的，那就直接跳过。
  - 这样既保留了 SRAM-friendly 的实现，又享受 sparsity 带来的计算减少。

- 这说明 FlashAttention 不只是一个单点优化。
- 它更像是一个高效 attention primitive：
  - dense exact attention 可以用。
  - block-sparse approximate attention 也可以用。
  - 后续很多 attention 变体都可以沿着这个思路重新实现。

---

**最值得记住的直觉**

- 标准 Attention 的低效，不只是因为“算了太多 pairwise score”。
- 更致命的是：
  - 它把巨大的 **attention matrix** 当成中间产物反复写入/读出 HBM。
- FlashAttention 的顿悟点是：
  - **这个矩阵在数学上存在，但在硬件上没必要真的存下来。**
- 它像是在说：
  - “我们可以承认有一张N×N的表，但不要真的把表打印出来。”
  - “每次只看一小块，现场算、现场更新、现场忘掉。”
- 这就是它又快又省显存，同时还保持 exact 的根本原因。

### 1. IO-aware Tiling for Exact Attention

**痛点直击：真正卡住的不是算不动，而是搬不动**

- **标准 Attention**最“难受”的地方，不只是有一个**N×N**的 Attention matrix，而是它会真的把这个大矩阵写到 **HBM** 里，再从 **HBM** 里读回来。
  - 计算流程大概是：
    - 算 **S=QKᵀ**
    - 把 **S** 写入 **HBM**
    - 从 **HBM** 读 **S**，做 **softmax**
    - 把 **P=softmax(S)** 写入 **HBM**
    - 再读 **P** 和 **V**，算输出 **O=PV**
  - 问题在于：
    - **S** 是 **N×N**
    - **P** 也是 **N×N**
    - 当 sequence length **N** 变大时，这两个矩阵会迅速变成显存和带宽黑洞

- 现代 GPU 的核心矛盾是：
  - **算力很强**
  - **HBM 很大但相对慢**
  - **SRAM 很快但很小**
  - Attention 里很多操作，尤其是 **softmax、mask、dropout、写回中间矩阵**，本质上更像是**memory-bound**，不是纯粹的 **compute-bound**

- 所以 FlashAttention 抓住的痛点非常准：
  - 以前大家盯着 **FLOPs**，想少算一点
  - 但真实训练里，很多时间花在了**把中间结果搬来搬去**
  - 标准 Attention 像是在 GPU 里反复搬一个巨大的临时仓库，计算单元反而经常等数据

- 这就是 **IO-aware** 的含义：
  - 不只是问“要算多少次乘加”
  - 更要问“数据在 **HBM** 和 **SRAM** 之间搬了多少次”
  - 在长序列 Attention 里，**少搬数据**往往比**少算一点**更重要

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

---

**通俗比方：不要把整本账本复印出来，只在柜台上逐页结算**

- 可以把 **Attention** 想成查账：
  - **Q** 是一批查询人
  - **K** 是账本索引
  - **V** 是账本内容
  - 标准 Attention 的做法是：
    - 先把每个查询人和每条账本索引的匹配分数全部算出来
    - 得到一张巨大的 **N×N** 对账表
    - 把这张表复印出来放进仓库 **HBM**
    - 之后再拿出来归一化、再复印、再拿出来算结果

- 这很浪费，因为你其实不需要永久保存那张完整的 **N×N** 对账表。
  - 你真正要的是最终结果：
    - 每个查询人应该从哪些账本内容里取信息
    - 加权之后得到一个输出向量
  - 中间那张巨大的分数表只是“过程产物”
  - 它像草稿纸，不该被装订成册、搬进仓库、再搬出来

- FlashAttention 的 Mental Model 是：
  - 不复印整本账本
  - 每次只拿一小叠 **K/V** 到柜台，也就是放进很快的 **SRAM**
  - 再让一小批 **Q** 来和这叠 **K/V** 对账
  - 算完这部分贡献后，立刻更新最终答案
  - 然后换下一叠 **K/V**
  - 全程不把完整 **N×N Attention matrix** 写进 **HBM**

- 这就像经典算法里的**分块矩阵乘法**或**cache blocking**：
  - 不是改变数学问题
  - 是改变数据访问顺序
  - 让热数据尽量留在快缓存里
  - 避免大规模中间结果反复进出慢内存

- 关键的顿悟点是：
  - **FlashAttention 不是 approximate attention**
  - 它没有少看某些 token
  - 它没有低秩近似
  - 它没有改 Attention 定义
  - 它只是把原来“先完整生成中间矩阵”的流程，改成了“边算边归并结果”

---

**关键一招：把“先造完整 Attention matrix”扭转为“在线累积 softmax 输出”**

- 标准流程的问题在于它默认必须先有完整的 **S=QKᵀ**，才能做完整的 **softmax**。
  - 这看起来很自然：
    - softmax 要看一整行所有分数
    - 不看完整一行，怎么知道归一化分母？
  - 这也是为什么很多实现会 materialize **N×N** 矩阵

- FlashAttention 最巧妙的地方是：
  - 作者没有改变 **softmax**
  - 而是把 **softmax 的归一化过程**改成了可以**分块在线更新**
  - 每处理一个 block，就更新当前行的：
    - **最大值 m**
    - **归一化因子 ℓ**
    - **当前输出 O**
  - 这样即使没有一次性看到整行，也能保证最后得到的结果和完整 softmax 一模一样

- 直观理解这个在线 softmax：
  - 你分批看到一组分数
  - 每批都有自己的局部最大值和局部加权和
  - 当新一批来了，如果发现更大的最大值，就把旧结果按比例缩放
  - 再把新批次的贡献加进去
  - 最后得到的归一化结果，等价于你一开始就拿到了所有分数

- 这就是那个关键逻辑转换：
  - 原来：
    - **先生成完整 S**
    - **写入 HBM**
    - **再读出 S 做 softmax**
    - **再写入 P**
    - **再读出 P 乘 V**
  - 现在：
    - **每次只生成一个 S block**
    - **在 SRAM 里立刻做 mask、softmax 局部统计、乘 V**
    - **只把最终 O 和少量统计量写回 HBM**
    - **从不把完整 S 或 P 写入 HBM**

- 可以把作者的操作概括成一句话：
  - 作者并没有发明新的 Attention，而是巧妙地把原来必须落盘的 **N×N 中间矩阵**，变成了在 **SRAM** 里一块一块用完即丢的临时草稿。

---

**为什么这招有效：它用更多一点计算，换来少很多 IO**

- FlashAttention 在 backward 里会**重算部分 Attention block**。
  - 表面看，这是增加了 FLOPs
  - 但它避免了从 **HBM** 读写巨大的 **P matrix**
  - 在现代 GPU 上，这笔账通常是赚的

- 这背后的硬件现实是：
  - 从 **SRAM** 读写很快
  - 从 **HBM** 读写慢得多
  - 大量小 kernel 之间来回写中间结果，会严重拖慢训练

- 所以 FlashAttention 的选择很“工程理性”：
  - 宁可在快的地方重算
  - 不愿在慢的地方反复搬运
  - 这就是典型的 **recompute vs memory traffic** 权衡

| 做法 | 中间 Attention matrix | 额外显存 | 核心瓶颈 | 是否 exact |
|---|---:|---:|---|---|
| 标准 Attention | 写入 **HBM** | **O(N²)** | **HBM IO** | 是 |
| Approximate Attention | 通常减少计算或稀疏化 | 视方法而定 | 可能有额外 overhead | 否或近似 |
| **FlashAttention** | **不 materialize** | **O(N)** | 更接近计算/SRAM 利用 | 是 |

---

**一句话抓住本质**

- **IO-aware Tiling for Exact Attention** 的本质是：
  - 把 Attention 从“先造一张巨大的 **N×N** 表再处理”，改成“把 **Q/K/V** 切成小块，在 **SRAM** 里边算边归并，绝不把大表写进 **HBM**”。

- 这篇工作的漂亮之处在于：
  - 它没有靠牺牲模型表达能力换速度
  - 它没有改变 Transformer 的数学定义
  - 它只是重新安排了数据流
  - 但这个重新安排正好击中了现代 GPU 的真实瓶颈：**IO 比 FLOPs 更贵**。

### 2. Online Softmax Normalization

**痛点直击：为什么需要Online Softmax Normalization**

- 标准Attention里最“难受”的地方，不是矩阵乘法本身，而是**Softmax必须看完整一整行**。
  - 对某个Query来说，它要和所有Key算分数，得到一整行长度为**N**的score。
  - Softmax要做两件事：
    - 找这一整行的最大值，用来做**numerical stability**。
    - 对整行指数值求和，得到归一化分母。
  - 问题是：如果你把整行score都算出来，就会自然地产生一个巨大的**N×N attention matrix**。

- 这在长序列上非常痛苦。
  - **N×N矩阵太大**，必须写到GPU HBM里。
  - HBM虽然容量大，但比on-chip SRAM慢得多。
  - Attention里很多操作，例如**mask、softmax、dropout**，本质上是memory-bound，不是compute-bound。
  - 也就是说，GPU不是算不动，而是被“搬数据”拖慢了。

- FlashAttention真正想避免的是：
  - 不要把完整的**S=QKᵀ**写到HBM。
  - 不要把完整的**P=softmax(S)**写到HBM。
  - 不要为了Softmax的归一化，被迫 materialize 整个attention matrix。

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

- 但这里有一个看似矛盾的要求：
  - Softmax需要整行信息。
  - FlashAttention又不想一次性存下整行。
  - **Online Softmax Normalization**就是解决这个矛盾的关键。

---

**通俗比方：分批统计全班成绩，但最后排名必须完全正确**

- 想象你要给一个班级的成绩做归一化。
  - 标准做法是：
    - 把全班所有人的分数摊在桌上。
    - 找最高分。
    - 每个人减去最高分。
    - 算指数、求和、归一化。
  - 这当然正确，但桌子要足够大。

- FlashAttention面对的问题像是：
  - 桌子很小，只能一次放一小批学生的成绩。
  - 但你最后还必须得到和“全班成绩一次性摊开”完全一样的Softmax结果。
  - 不能近似，不能偷懒，不能改变答案。

- Online Softmax的思路是：
  - 每看一批成绩，就记录两个东西：
    - 当前见过的最高分：**m**
    - 基于这个最高分计算出来的指数和：**ℓ**
  - 新来一批成绩时：
    - 如果新批次里出现了更高的分数，就把旧统计量“换尺子”重新缩放。
    - 如果没有更高分，就直接把新批次贡献加进去。
  - 这样你不需要记住所有学生的原始成绩，只需要维护这两个小账本：**m和ℓ**。

- 关键顿悟是：
  - Softmax不是非得“等全行都在手里”才能算。
  - 只要你维护好**最大值m**和**归一化分母ℓ**，你就可以像流式处理一样，一块一块地吃进去。
  - 最后得到的结果，和一次性处理整行**完全等价**。

---

**关键一招：把“全局Softmax”改成“可合并的块级Softmax”**

- 作者没有近似Softmax，也没有改Attention定义。
  - 它仍然计算**exact attention**。
  - 巧妙之处在于：把Softmax的归一化过程改造成了一个可以**incrementally combine**的过程。

- 原来的流程是：
  - 算完整的**QKᵀ**。
  - 得到完整的每一行score。
  - 对每一行做Softmax。
  - 再乘以**V**。
  - 这会强迫系统保存巨大的**N×N中间矩阵**。

- FlashAttention替换掉的是中间这一步：
  - 不再等整行score全部算完。
  - 而是每次只算一个小块：某一批Query对某一批Key的score。
  - 对这个小块，临时算出：
    - 块内最大值：**m̃**
    - 块内指数和：**ℓ̃**
    - 块内未最终归一化的Softmax贡献。
  - 然后把这批结果合并进当前这行的全局统计：**m和ℓ**。

- 这里最巧的一点是“换基准”。
  - Softmax为了稳定性，会减去最大值。
  - 但最大值是会随着新块到来而变化的。
  - 如果新块里有更大的score，旧的指数和并不是作废，而是按新的最大值重新缩放。
  - 这就像你原来用“90分是最高分”作为基准统计，后来发现有人考了95分，你不用重看所有试卷，只要把旧账按新基准折算一下。

- 同时，FlashAttention不只是维护Softmax分母，还同步维护输出**O**。
  - 每处理一个Key/Value块，就把它对最终Attention输出的贡献加进去。
  - 如果新的最大值改变了，旧的输出贡献也会一起被重新缩放。
  - 所以最后的**O**就是完整Attention输出，而中间从没把完整**P矩阵**写到HBM。

---

**一句话抓住本质**

- **Online Softmax Normalization**的本质是：
  - 把Softmax从一个“必须一次看到整行”的操作，改造成一个“只要维护m和ℓ，就能分块合并”的操作。
  - 它让FlashAttention可以在小SRAM里一块一块算Attention，同时保持：
    - **数值稳定**
    - **完全精确**
    - **不物化N×N attention matrix**
    - **显著减少HBM读写**

---

**和普通Softmax的直观对比**

| 做法 | 需要完整score行吗 | 是否存N×N矩阵 | 是否精确 | 核心代价 |
|---|---:|---:|---:|---|
| Standard Attention Softmax | 需要 | 需要 | 是 | 大量HBM读写 |
| Online Softmax Normalization | 不需要，一块一块合并 | 不需要 | 是 | 多维护**m、ℓ**并做缩放 |
| Approximate Attention | 通常不需要完整矩阵 | 通常不需要 | 否 | 可能损失模型质量 |

---

**为什么这招对FlashAttention特别关键**

- 如果没有Online Softmax：
  - Tiling只能帮你分块做矩阵乘法。
  - 但Softmax仍然会逼你把整行score凑齐。
  - 最后还是绕不开**N×N attention matrix**。

- 有了Online Softmax：
  - 每个block都可以在SRAM里完成：
    - score计算
    - mask
    - max统计
    - exp
    - denominator更新
    - output累积
  - HBM只需要保存：
    - 输入**Q、K、V**
    - 输出**O**
    - 每行的统计量**m、ℓ**
  - 这就是FlashAttention能做到**memory-efficient exact attention**的根本原因。

- 更直白地说：
  - **Online Softmax是FlashAttention避免写出attention matrix的数学许可证。**
  - 没有它，工程上的tiling就不够；有了它，Attention才真正能被改造成IO-aware算法。

### 3. Backward Recomputation Instead of Attention Matrix Storage

**痛点直击**
- 传统 attention 的 backward 最难受的地方，不是“算不过来”，而是“**拿什么来算**”。
- 因为 forward 如果把整个 **N×N attention matrix** 存下来，显存里立刻被这个大块头吃掉；序列一长，**HBM 读写**就爆炸，训练速度和可扩展性一起掉。
- 更要命的是，backward 真正用到这个大矩阵时，很多信息其实只是为了“回看一眼”，但你却得把它完整地写回、再读出来，属于典型的**为了省事而付出巨额搬运成本**。
- FlashAttention 抓住的就是这个痛点：与其把中间结果老老实实堆在 HBM 里，不如**只保留少量关键信息**，其他的在反向时现场重算。

![](f2d4c917e30135292cb1109238fa9a694368cd1928cd6e9ebe3bada6de6be14e.jpg)

**通俗比方**
- 这就像你在做一道超长计算题。
- 传统做法是：每一步推导都拍照存档，最后检查时再翻相册。结果相册太大、翻得太慢，真正耗时间的不是思考，而是**找照片**。
- FlashAttention 的做法是：你不存整本相册，只记下最后答案和几个“路标”——比如每段推导的归一化信息。等需要回推时，直接在桌面上把那一小段重新算一遍。
- 这有点像数据库里的 **materialize vs recompute**：
  - **materialize**：把中间结果全存下来，省计算，费 IO；
  - **recompute**：少存一点，必要时重算，省 IO，通常更快。
- 在 GPU 上，这个选择尤其关键，因为 **HBM 很慢，SRAM 很快**。很多时候，**少搬一次大矩阵**，比少做几次乘法更值钱。

**关键一招**
- 作者并没有改 attention 的数学定义，也没有偷偷做近似。
- 他们只是把 backward 里的“默认动作”扭转了：  
  - 传统 backward：先把 forward 的 **attention matrix** 存起来，反向时直接读它。
  - FlashAttention backward：**不存整个 attention matrix**，只存 **output O** 和 softmax 的少量统计量 **m、l**，反向时按 block 在 on-chip SRAM 里把需要的 attention 片段重新算出来。
- 这一步的妙处在于：
  - forward 已经留下了足够的信息，能保证 backward **精确复原**；
  - backward 只重算“当前这块用得到的那部分”，不用把整个 **N×N** 矩阵搬来搬去；
  - 于是瓶颈从 **HBM storage/readback** 变成了更多一点的局部计算，但整体反而更快。
- 可以把它理解成：作者不是在“保存计算过程”，而是在“保存足够的索引”，让 backward 在 **SRAM 里临时复原现场**，避免对 HBM 形成灾难性压力。
- 这也是为什么它叫 **Backward Recomputation**，核心不是“多算”，而是“**别把最贵的中间结果存到最慢的地方**”。

- 一句话抓住本质：**用一点重新计算，换掉一大坨显存读写。**
- 对 FlashAttention 来说，这个交换非常划算，因为 attention 的真实瓶颈常常不是 FLOPs，而是 **memory traffic**。

### 4. CUDA Kernel Fusion

**痛点直击：为什么需要CUDA Kernel Fusion**

- 标准Attention的难受点，不只是**FLOPs多**，而是**搬数据太频繁**。
  - 在普通PyTorch实现里，Attention通常被拆成几步：
    - **QKᵀ矩阵乘法**：算出Attention score矩阵**S**
    - **masking**：把不该看的位置设成负无穷
    - **softmax**：归一化成概率矩阵**P**
    - **dropout**：对P做随机丢弃
    - **PV矩阵乘法**：把概率加权到Value上，得到输出**O**
  - 每一步通常是一个单独的CUDA kernel。
  - 每个kernel之间都要把中间结果写回**HBM**，下一步再从**HBM**读回来。

- 问题在于，**HBM很大但相对慢，SRAM很小但快得多**。
  - A100上，HBM带宽大约是**1.5–2.0TB/s**。
  - 片上SRAM带宽可到约**19TB/s**量级。
  - 所以很多Attention操作不是“算不动”，而是“数据搬来搬去把时间耗光了”。

- 最痛的中间结果是**N×N Attention matrix**。
  - 当sequence length是**N**时，S和P都是**N×N**。
  - 这两个矩阵非常大。
  - 标准实现会反复把它们写入、读出HBM。
  - 这就像你每做一道菜的一个小步骤，都把半成品端到楼下仓库，再端回来继续做。

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

---

**通俗比方：Kernel Fusion像“厨房流水线合并”**

- 想象你在厨房做一盘菜。
  - 普通实现的做法是：
    - 切菜后，把菜端进仓库。
    - 炒之前，再从仓库拿出来。
    - 加调料后，又端回仓库。
    - 勾芡前，再拿出来。
    - 装盘前，又来回一趟。
  - 你真正累的不是切菜、炒菜，而是**一直往返仓库**。

- **CUDA Kernel Fusion**的思路是：
  - 不要每个小步骤都单独开工。
  - 把“切菜、调味、翻炒、勾芡、装盘”尽量安排在同一个灶台上连续完成。
  - 数据一旦从**HBM**搬到**SRAM/registers**，就尽量在片上把后续步骤做完。
  - 最后只把最终结果写回HBM。

- 放到FlashAttention里，这个“厨房灶台”就是一个融合后的**CUDA kernel**。
  - 它不是算完**S**就存起来。
  - 也不是算完**P**再存起来。
  - 而是在片上连续做：
    - **matrix multiplication**
    - **masking**
    - **softmax**
    - **dropout**
    - **value aggregation**
  - 中间的S、P尽量只作为片上临时变量存在，不落到HBM。

---

**关键一招：把“分步落盘”改成“片上直通”**

- 作者最巧妙的地方不是发明了一个新的Attention公式。
  - FlashAttention仍然计算**exact attention**。
  - 它没有把softmax近似掉。
  - 它没有改模型结构。
  - 它改的是**执行方式**。

- 原来的流程是“每一步都生成一个大中间结果”：
  - **QKᵀ**生成S，写回HBM。
  - **mask**读S，写回masked S。
  - **softmax**读masked S，写回P。
  - **dropout**读P，写回dropped P。
  - **PV**读dropped P和V，写回O。

- FlashAttention把这条链条扭转成：
  - 只把一小块**Q/K/V block**搬进SRAM。
  - 在SRAM里直接完成这一小块上的所有操作。
  - 得到这块对输出O的贡献。
  - 用在线softmax统计量更新最终输出。
  - 不把完整的**N×N S/P矩阵**写入HBM。

- 这就是**CUDA Kernel Fusion**在这里的价值：
  - 不是单纯“少启动几个kernel”。
  - 更关键的是避免了跨kernel之间的**HBM round-trip**。
  - 中间结果不再变成HBM里的庞大临时文件。
  - 数据路径从“算一步、存一步、读一步”变成“搬进来、一口气算完、写结果”。

---

**直觉版总结**

- 标准Attention慢，很多时候不是因为GPU算力不够，而是因为它一直在搬运巨大的**Attention matrix**。
- **CUDA Kernel Fusion**就是把多个原本分散的GPU操作合成一个kernel，让数据在**SRAM/registers**里连续流过完整计算链。
- 在FlashAttention中，它和**tiling**配合使用：
  - **tiling**解决“SRAM太小，装不下整个Attention matrix”的问题。
  - **kernel fusion**解决“每一步都要回HBM中转”的问题。
- 一句话说：
  - **作者没有换Attention，而是把Attention的执行流程从“仓库中转制”改成了“片上流水线制”。**

### 5. IO Complexity Optimization

**痛点直击：为什么要把优化目标从FLOPs扭到IO Complexity**

- 标准 Attention 的“难受点”不只是**算得多**，而是**搬得太多**。
  - 传统实现会显式生成并保存两个巨大的中间矩阵：
    - **S=QKᵀ**，大小是 **N×N**
    - **P=softmax(S)**，大小也是 **N×N**
  - 这些矩阵会被反复从 **HBM** 读出、写回。
  - 当序列长度 **N** 变大时，**N×N attention matrix** 很快成为内存流量黑洞。

- 现代 GPU 上，很多 Transformer 操作不是被计算单元卡住，而是被**显存带宽**卡住。
  - **SRAM** 很快，但很小。
  - **HBM** 很大，但慢一个数量级。
  - 标准 Attention 的问题是：
    - 算完一块东西，马上写进 HBM。
    - 下一步又从 HBM 读回来。
    - GPU 算力很强，但大量时间花在“搬箱子”上。

- 所以这篇论文的关键视角是：
  - 不要只问：**这个算法有多少FLOPs？**
  - 更要问：**它从HBM搬了多少数据？**
  - FlashAttention 的核心贡献就是把 Attention 从一个“算术复杂度问题”，重新定义成一个**IO Complexity Optimization** 问题。

![](0cdd6983342c6c52df0fd4f96eeb69fd392bf4cbc985ff75b5a1ec5255708e66.jpg) *Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large ?? × ?? attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.*

---

**通俗比方：不是少做题，而是少跑仓库**

- 可以把 GPU 想象成一个厨房：
  - **SRAM** 是厨师手边的小案板。
  - **HBM** 是很远的大仓库。
  - **FLOPs** 是切菜、炒菜的动作。
  - **IO Complexity** 是厨师跑去仓库拿菜、放菜的次数。

- 标准 Attention 的做法像这样：
  - 先把所有半成品菜都做好。
  - 每做一步就把半成品端回仓库。
  - 下一步再从仓库搬回来继续加工。
  - 菜本身没那么难做，真正累的是厨师一直来回跑。

- FlashAttention 的做法像这样：
  - 每次只从仓库拿一小批菜到案板。
  - 在案板上把能做的步骤一次性做完：
    - 乘法
    - mask
    - softmax
    - dropout
    - 乘 V
  - 只把最终需要的结果写回仓库。
  - 中间的大盘子 **N×N attention matrix** 根本不端回仓库。

- 这就是“顿悟点”：
  - FlashAttention 并不是神奇地少算了 Attention。
  - 它是避免把最大、最占地方、最频繁访问的中间结果写进慢内存。
  - 它宁愿在 SRAM 里**局部重算一点点**，也不愿意反复读写巨大的 **N×N matrix**。

---

**关键一招：把“全矩阵落地”替换成“分块流式累积”**

- 标准 Attention 的流程是：
  - 计算完整的 **QKᵀ**
  - 写入 **HBM**
  - 从 **HBM** 读出做 **softmax**
  - 再写入 **HBM**
  - 再读出和 **V** 相乘

- FlashAttention 把这条流程扭转了：
  - 不再把 **S** 和 **P** 作为完整矩阵存下来。
  - 改成每次处理一个 **tile/block**。
  - 在 **SRAM** 里完成这个 block 对输出的贡献。
  - 用一组很小的统计量维护 softmax 的全局一致性：
    - 每行当前最大值 **m**
    - 每行归一化因子 **ℓ**
  - 最后逐块累积出正确的 **O=softmax(QKᵀ)V**。

- 最巧妙的逻辑转换是：
  - softmax 看起来必须看到整行所有元素才能算。
  - 作者发现可以把 softmax 的“全局归一化”拆成可增量更新的状态。
  - 所以每处理一个 K/V block，就更新一次当前行的：
    - 最大值
    - 归一化常数
    - 输出向量
  - 这使得 Attention 可以像 streaming algorithm 一样工作。

- backward pass 也用了同一个思想：
  - 标准实现为了反向传播，会保存巨大的 **P matrix**。
  - FlashAttention 不保存它。
  - 只保存小的 softmax statistics。
  - 反向时在 SRAM 中重新算需要的 attention block。
  - 这看似增加了一点 FLOPs，但大幅减少 HBM traffic，反而更快。

---

**核心复杂度对比**

| 方法 | HBM Accesses | 直观含义 |
|---|---:|---|
| **Standard Attention** | **Θ(Nd+N²)** | 必须反复读写完整 **N×N attention matrix** |
| **FlashAttention** | **Θ(N²d²/M)** | 利用 **SRAM size M** 做 tiling，减少 HBM 往返 |
| **Block-sparse FlashAttention** | **Θ(Nd+N²d²s/M)** | 再乘上 sparsity ratio **s**，只处理非零 block |

- 这里的关键不是公式本身，而是公式背后的含义：
  - 标准 Attention 的瓶颈里有一个硬邦邦的 **N² HBM traffic**。
  - FlashAttention 让 HBM traffic 受 **SRAM size M** 影响。
  - **M** 越大，每次能在 SRAM 里处理的 block 越大，需要往返 HBM 的次数越少。
  - 所以它把硬件结构纳入算法设计，而不是写完算法再交给 GPU 去“碰运气”。

---

**一句话抓住本质**

- **IO Complexity Optimization** 在这篇论文里不是“顺手优化内存”，而是核心算法思想：
  - 作者没有改变 Attention 的数学定义。
  - 也没有用近似来减少计算。
  - 而是把原本“生成巨大中间矩阵再处理”的流程，改成“在 SRAM 中分块计算、在线归一化、只写最终结果”。
  - 结果是：**FLOPs 可能没少，甚至 backward 还多了一点重算，但 HBM traffic 大幅下降，所以真实 wall-clock time 更快。**

- 这也是 FlashAttention 最重要的启发：
  - 在现代 GPU 上，快算法不一定是 FLOPs 最少的算法。
  - 真正快的算法，往往是最少打扰 **HBM** 的算法。

### 6. Block-Sparse FlashAttention

**痛点直击：为什么需要Block-Sparse FlashAttention**

- **FlashAttention解决的是“别把完整Attention矩阵搬来搬去”**：
  - 标准Attention最难受的地方不是只在于算了**N×N**个分数，而是还把这个巨大的**Attention matrix**反复写入、读出**HBM**。
  - FlashAttention用**tiling**把计算搬进**SRAM**，避免物化完整矩阵，所以它已经比标准Attention快很多、也省很多显存。

- 但FlashAttention仍然有一个根本负担：
  - 它虽然不存完整**N×N**矩阵，但仍然默认“每个Query block都要看每个Key block”。
  - 换句话说，它减少了**搬运成本**，但没有减少“哪些block需要被算”的范围。
  - 当序列特别长，比如**16K、64K**时，哪怕IO已经优化，完整dense attention仍然太重。

- **Block-Sparse FlashAttention的痛点更具体**：
  - 很多长序列任务并不真的需要每个Token都全局互看。
  - 但传统Sparse Attention经常“理论上省FLOPs，实际不快”：
    - 稀疏模式带来额外索引开销。
    - 小块稀疏计算不容易吃满GPU。
    - 仍然可能有糟糕的HBM访问模式。
  - 结果就是：**省了计算量，却没省到墙钟时间**。

- 这篇论文的关键判断是：
  - 如果要做Sparse Attention，不能只问“少算了多少FLOPs”。
  - 更要问：**少搬了多少数据？这些数据是不是以GPU喜欢的block形式被搬？**

---

**通俗比方：它像“只查需要查的货架”，但每次整箱搬**

- 把Attention想象成一个巨大仓库：
  - 每个**Query block**是一张采购清单。
  - 每个**Key/Value block**是一排货架。
  - Dense Attention的做法是：每张清单都去检查所有货架。
  - FlashAttention的进步是：不把全仓库账本复印出来，而是一次搬几排货架到前台处理，处理完再换下一批。

- **Block-Sparse FlashAttention再进一步**：
  - 它提前拿到一张“货架访问地图”，也就是**block sparsity mask**。
  - 地图上标0的货架，说明这张清单根本不用看。
  - 作者不是逐个商品跳着查，而是按**block**整箱整箱地查：
    - 该看的block，完整搬进**SRAM**高效计算。
    - 不该看的block，直接跳过，连搬都不搬，连算都不算。

- 这个比方的重点不是“稀疏”本身，而是**块状稀疏**：
  - 如果你一个商品一个商品跳着拿，GPU会很痛苦。
  - 如果你一整箱一整箱搬，GPU可以保持高吞吐。
  - 所以它不是随便Sparse，而是**GPU友好的Sparse**。

![](581d9c655d98724b3a4a319d39161be45a2bdc250eb82557da2871ade1203941.jpg)

---

**关键一招：在FlashAttention的tiling循环里，加一个“跳过空block”的门**

- 作者没有重新设计Attention，也没有把Softmax换成近似核函数。
- 它保留了FlashAttention最核心的机制：
  - **tiling**
  - **on-chip SRAM计算**
  - **online softmax统计量更新**
  - **不物化完整Attention matrix**
  - **backward中重算Attention block**

- 真正巧妙的一招是：
  - 原来的FlashAttention会遍历所有**Q block × K/V block**组合。
  - Block-Sparse FlashAttention在这个双重循环中插入一个判断：
    - 如果**mask[i,j]=1**，就照常加载这个**K/V block**，计算对应的Attention block。
    - 如果**mask[i,j]=0**，直接跳过这个block。
  - 这一步非常朴素，但威力很大：**跳过的不只是FLOPs，而是整块HBM读写和SRAM计算。**

- 可以把它理解成对FlashAttention流程的一个“扭转”：
  - FlashAttention：**所有block都算，但每个block算得很省IO。**
  - Block-Sparse FlashAttention：**只算必要block，并且每个必要block仍然算得很省IO。**

| 方法 | 是否精确Dense Attention | 是否避免N×N矩阵物化 | 是否跳过无用block | 主要收益 |
|---|---:|---:|---:|---|
| Standard Attention | 是 | 否 | 否 | 实现简单，但HBM压力大 |
| FlashAttention | 是 | 是 | 否 | 大幅减少HBM访问 |
| Block-Sparse FlashAttention | 否，按mask近似 | 是 | 是 | 同时减少HBM访问和计算量 |

- 它的复杂度变化很直观：
  - FlashAttention的主要IO成本来自遍历所有block。
  - 如果只有比例为**s**的block是非零的，那主要IO成本就大约乘上**s**。
  - 所以IO复杂度变成：**Θ(Nd+N²d²s/M)**。
  - 这里的**s**越小，跳过的block越多，速度越接近线性扩展。

| 符号 | 直觉含义 |
|---|---|
| **N** | 序列长度 |
| **d** | head dimension |
| **M** | SRAM容量 |
| **s** | block sparsity mask中非零block比例 |
| **Θ(Nd)** | 至少要读输入、写输出，逃不掉 |
| **N²d²s/M** | 真正的block attention计算与IO成本，被稀疏比例**s**缩小 |

- 最关键的直觉结论：
  - **Block-Sparse FlashAttention不是“Sparse Attention + FlashAttention”的简单拼接。**
  - 它真正做对的是：让Sparse Attention也遵守FlashAttention的IO-aware原则。
  - 传统Sparse Attention常常只是在数学上少算；它是在GPU内存层级上也真的少搬、少写、少读。

- 这也是为什么它在长序列上特别强：
  - 对短序列，dense FlashAttention已经很快，稀疏化收益有限。
  - 对超长序列，dense的block数量爆炸，跳过大部分block后收益巨大。
  - 论文中它能把Transformer推到**64K sequence length**，核心靠的就是这一点。

- 一句话记住：
  - **FlashAttention是在“全都要算”的前提下，把每次计算搬得更聪明。**
  - **Block-Sparse FlashAttention是在“只算该算的”的前提下，继续用最聪明的搬法。**
