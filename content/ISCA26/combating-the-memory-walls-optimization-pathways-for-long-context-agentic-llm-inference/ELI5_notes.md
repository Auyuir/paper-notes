# Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference 通俗讲解

### 0. 整体创新点通俗解读

**痛点直击**

之前的GPU和TPU在处理Chatbot时如鱼得水，但在面对Agentic LLM任务时却极其难受。这并非因为算力不够，而是因为“形状不对”和“仓库太小”。
- Agentic任务（如读取整个网页DOM或大型代码库）需要极长的Context，导致Prefill阶段的Token数量暴增。
- 由于HBM容量有限，无法容纳巨大的KV Cache，硬件只能被迫缩小Batch Size。
- 这导致核心的GEMM运算变成了“瘦长”的Fat GEMM（M维极小，K维极长）。
- 传统硬件的Square Systolic Array是为方正的矩阵设计的，处理这种瘦长矩阵时，大量计算单元只能闲置围观。
- 同时，巨大的KV Cache不仅撑爆了HBM容量，其频繁的读写也瞬间打满了Memory Bandwidth，导致计算单元无米下锅。

![](images/4350dc7220f6521d1422a27bb94852e9ccc07ce4a2cbf5af4083475a56cc2b26.jpg)

---

**通俗比方**

想象你雇了一支100x100的方阵军队去搬砖。
- 在Chatbot时代，砖块也是方方正正的，军队刚好能一拥而上，效率拉满。
- 但在Agentic时代，砖块变成了极细极长的一条（Fat GEMM）。你让100x100的方阵去搬这根长条，其实只有第一排的几个人能碰到砖，后面99排士兵全在原地罚站。
- 更糟的是，仓库（HBM）太小，这根超长砖块（KV Cache）根本塞不进去几根，搬运工为了运这点砖跑断了腿，把路（Bandwidth）全堵死了。

![](images/45560d485999d69f738b406b0a90ef975f342cef4248da4eea26b370b2cf4ba0.jpg) *(a) PLENA achieves higher utilization than the standard square systolic array with the same resources. (b) PLENA’s optimization pathways—(1) a flattened systolic array and (2) asymmetric quantization—together achieve improved effective memory bandwidth utilization and help reduce memory capacity limitations. Fig. 2: A comparison of attainable FLOPs between a squareshaped systolic array (e.g. in a TPU) and PLENA’s when using the same number of multipliers<sup>2</sup>.*

---

**关键一招**

作者并没有选择硬刚算力，而是巧妙地重塑了硬件的“阵型”和“物流系统”，提出了PLENA架构，核心逻辑如下：
- **把方阵拆成长条**：抛弃传统的Square Systolic Array，设计出Flattened Systolic Array。这就好比把100x100的方阵拆成4x2500的长条阵，刚好对齐那根细长的砖块，让每个士兵都有砖搬，直接拉满计算利用率。
- **砖块压缩，核心不妥协**：采用Asymmetric Quantization。把不敏感的Weights和KV Cache用低精度的MX格式狠狠压缩，省出仓库空间；但对敏感的Activations保留高精度。甚至加入了Selective Rotation，在压缩前先把刺头的异常值磨平。
- **中间结果不回库**：原生支持FlashAttention。以前算Attention要把巨大的中间结果搬回仓库再取出来，现在直接在芯片内部用Tile-level调度算完，彻底砍断了最吃带宽的物流往返。

| 优化维度 | 传统架构痛点 | PLENA的扭转策略 |
| :--- | :--- | :--- |
| **计算阵列形状** | Square Array处理Fat GEMM大量闲置 | **Flattened Systolic Array**对齐长边 |
| **存储容量与带宽** | KV Cache过大撑爆HBM，读写打满带宽 | **Asymmetric Quantization** (MX格式)压缩KV与权重 |
| **Attention计算** | 中间矩阵过大导致频繁HBM读写 | **Native FlashAttention**支持，片内融合计算 |

通过这三招，PLENA在同等乘法器和HBM配置下，把Agentic推理的吞吐量拉到了A100的2.23倍，TPU v6e的4.70倍。

### 1. Flattened Systolic Array Architecture

**痛点直击**

- 传统加速器（如 TPU）通常采用 **Square-shaped Systolic Array**（如 128×128），其底层硬件设计假设 GEMM 的三个维度（M, K, N）大小相近。
- 在 **Agentic LLM Inference** 中，极长的 Context 会吃掉大量的 HBM 容量，导致硬件能容纳的 **Batch Size** 极小。
- 这就产生了 **Fat GEMM** 现象：与 Batch 相关的维度（通常是 M）极短，而与 Context/Hidden size 相关的归约维度（K）极长。
- 把一个极短的 M 维度塞进宽阔的方形阵列中，会导致绝大多数 **Processing Elements (PEs)** 无数据可算，直接处于 **Idle** 状态，算力利用率暴跌。

**通俗比方**

- 想象你有一个 **128×128 的方阵工人**（方形阵列），他们负责把传送带上的零件（K 维度）组装成产品（M 维度）。
- 常规情况下，传送带送来 128 个产品份的零件，每个工人都在满负荷干活。
- 但在 Agentic 场景下，由于内存限制，你只能同时组装 **4 个产品**，但传送带上的零件却长达 10000 米。
- 如果你硬把他们塞在 128×128 的方阵里，只有最边上的 4 列工人在疯狂加班，其余 124 列工人全程摸鱼。
- PLENA 的做法是：把方阵打散，排成一条 **4×1024 的长蛇阵**，让所有工人沿着超长的传送带一字排开，全员并行接活。

**关键一招**

- 作者并没有增加硬件乘法器的总数，而是直接 **重塑了阵列的物理形状**。
- PLENA 将方形阵列拆解为多个细长的 **Sub-arrays**，让数据沿着超长的 K 维度流动，而部分和（**Partial Sums**）则固定在 PE 中。
- 为了汇总这些散布在长条阵列中的部分和，作者在硬件中插入了一个 **Result Adder Tree**。
- 这个加法树通过一条专门的 `M_SUM` 指令触发，在沿着 K 维度计算完毕后，高效完成跨阵列的求和。
- 这一招彻底消除了 Fat GEMM 在方形阵列上的映射空洞，让 FFN 和 FlashAttention 阶段的计算单元利用率直接拉满。

![](images/ee20e4180c087206c605b0575faec8ab33f2ae18629c149e83003895034cdb72.jpg) *Fig. 5: Processing flow for the weight–activation output stationary GEMM. Because memory capacity constrains batch size, the M dimension remains small. Setting BLEN = M on the flattened systolic array yields high utilization.*

| 维度 | 传统 Square Array | PLENA Flattened Array |
|---|---|---|
| 阵列形状 | 128x128 (正方形) | 4x1024 (扁平长条) |
| 匹配的 GEMM | M ≈ K ≈ N (常规推理) | M << K (长上下文 Agentic 推理) |
| PE 利用率 | 极低 (大量 PE 空转) | 极高 (全员并行) |

### 2. Asymmetric Quantization with Microscaling (MX) Co-design

**痛点直击**

在 Agentic LLM 推理场景下，硬件面临极其恶略的“Memory Walls”（带宽与容量墙）：
- Agentic 任务动辄处理 100k+ tokens 的长上下文（如整个网页 DOM 或代码库），KV Cache 随上下文长度线性疯长，直接撑爆 HBM 容量。
- 为了省显存，传统量化方案往往“一刀切”，把 Weights、Activations、KV Cache 统一压到低比特（如 INT4），但这会导致严重的精度暴跌。
- 更隐蔽的痛点在于：现有的 PTQ 优化算法（如 QuaRot 的旋转、GPTQ 的迭代补偿）原本是为标准整数格式设计的。直接套用到 Microscaling (MX) 格式上，不仅不兼容，反而会引发更严重的性能退化。

---

**通俗比方**

这就像是搬家打包，难点在于“既要装得多，又不能把东西弄坏”：
- **Activations (A)** 就像易碎的瓷器，对量化误差极度敏感，必须用厚泡沫（高精度 FP）包好，放在随手拿到的随身包（Vector SRAM）里。
- **Weights (W) 和 KV Cache** 就像厚重的书本和衣服，不怕压，可以用真空袋抽干空气（低精度 MX 格式），死死塞进大卡车（HBM）里。
- **Microscaling (MX)** 就像是“共享托盘”，一小堆物品共用一个标签（共享缩放因子），既省空间又好管理。
- 传统做法是所有东西都用同一种箱子，要么太浪费空间，要么把瓷器压碎了。而作者发现，盲目把所有书本和瓷器都扔进搅拌机（全局旋转 PTQ），反而会让书本更难码放。

---

**关键一招**

作者并没有重头训练，也没有强行套用现成的量化算法，而是巧妙地在软硬件栈中间插入了**非对称分配与定制化 PTQ** 的组合拳：

- **非对称精度分配**：
  - 硬件层面原生支持混合精度数据通路。Activations 在片上以高精度 FP 存于 Vector SRAM；Weights 和 KV Cache 以低精度 MX 格式存于 Matrix SRAM 和 HBM。
  - 这种设计直接打破了容量墙：以 LLAMA-3.3-70B 为例，在 4/4/4 配置下，KV Cache 占用从 239.26 GB 暴降至 59.81 GB，权重从 129.46 GB 降至 32.36 GB。

![](images/6664593a23e73382f3753155fa97edecc27e8802c1818555b6db46e3e3110970.jpg)

- **修复 PTQ 与 MX 的兼容性（核心逻辑转换）**：
  - *针对 Weights*：作者没有直接用 GPTQ 的权重误差最小化目标，而是将其扭转成 **output-norm guided block-wise clipping**。他们不追求权重本身量化误差最小，而是追求“输出结果的误差最小”。这完美契合了 MX 格式的 block 级运算特性。
  - *针对 Activations/KV*：作者发现 QuaRot 的全局旋转对 MX 格式有害。于是他们把“全局旋转”改成了 **Selective Rotation**。只对特定层在运行时进行 Hadamard 变换，并且让硬件在数据加载进 Matrix SRAM 时原生支持逆变换操作，实现零额外延迟的异常值抑制。

- **软硬件协同搜索**：
  - 通过多目标贝叶斯优化，在精度、延迟、面积构成的 Pareto 前沿上，自动找出 W/A/KV 的最佳精度组合与硬件微调参数（如 Tile size）。

| W/A/KV (bits) | Peak Bandwidth (GB/s) | KV Cache Footprint (GB) | Weight Storage (GB) |
| :--- | :--- | :--- | :--- |
| 16/16/16 | 8192 | 239.26 | 129.46 |
| 4/4/4 | 2048 | 59.81 | 32.36 |

### 3. Native Hardware Support for FlashAttention

**痛点直击**

在长上下文 Agentic LLM 推理中，Attention 计算的算力需求会随上下文长度呈平方级爆炸。传统的 Attention 计算流程存在一个极其致命的“顾头不顾尾”问题：
- 计算中间矩阵 $QK^\top$ 时，会产生一个庞大的二维矩阵。
- 芯片上的 **SRAM** 根本装不下这个巨无霸，只能将其写入 **HBM**（显存）。
- 随后的 **Softmax** 和 $PV$ 计算又必须把这个矩阵从 **HBM** 读回来。
这种“写了又读”的往返操作，直接把 **HBM** 的带宽吃干抹净，导致计算单元一直在等数据，这就是典型的 **Memory Wall**（带宽墙）。

虽然软件层面提出了 **FlashAttention** 来解决这个 IO 瓶颈，但传统的 **Systolic Array**（脉动阵列）硬件根本没法高效跑这个算法：
- 传统硬件只暴露粗粒度的 **GEMM** 原语，无法做细粒度的分块融合。
- 缺乏在线 **Softmax** 所需的行级归约和非线性计算支持。
- 必须显式转置大矩阵，带来巨大的延迟和面积开销。

---

**通俗比方**

想象一家快餐店厨房（芯片）处理一份超大订单（长上下文）。
- **传统做法**：切菜工（**GEMM** 单元）切好一堆土豆，装盆送到地下室冷库（**HBM**）；炒菜工（**Softmax** 单元）再跑去地下室把土豆搬上来调味，然后再送回地下室；最后出锅工（$PV$ 计算）再去地下室搬一次。电梯（内存带宽）堵死，大家都在干等。
- **软件版 FlashAttention**：切菜工切一小批土豆，直接递给旁边的炒菜工调味，再直接递给出锅工。不跑地下室了。但这要求厨师们（软件）自己极其小心地协调节奏，非常别扭。
- **PLENA 的原生硬件支持**：PLENA 不是让厨师去适应旧厨房，而是直接把厨房改造成“流水线案板”。案板自带切菜、调味、翻炒一体化功能，且案板内部有暗格（片上 **SRAM**）直接传递半成品。厨师只需要按特定口令（**ISA**）操作，中间产物根本不需要离开案板。

![](images/0a1e89cbb8843ff9d2ac243ec093ac3cc7bd086c50ac4f21c4d15772b0964846.jpg) *Fig. 8: Example of how the single batch single head attention algorithm maps onto PLENA’s custom ISA. Instruction prefixes denote the unit type (e.g., M for Matrix instructions).*

---

**关键一招**

作者并没有修改 **FlashAttention** 的数学逻辑，而是巧妙地在硬件架构的三个关键环节进行了“扭转”和“替换”，让硬件原生贴合算法的流式分块逻辑：

- **替换显式转置为“读取时转置”**：
  - 传统架构要显式搬运并转置 $K$ 矩阵，开销巨大。
  - PLENA 设计了特殊的 **Matrix SRAM**，将逻辑行分散存储在不同 **Bank** 中。读取时通过跨 **Bank** 并行访问，零额外周期直接输出转置后的数据。

![](images/9716793725b3dd382ce084f94f316afb23addc4a8998c4049c1f4b313f3953.jpg)

- **插入细粒度的自定义 ISA 控制**：
  - 传统硬件指令粗粒度，只能调度整个大矩阵运算。
  - PLENA 设计了专门的 **PLENA ISA**，支持 **Tile-level scheduling**（分块级调度）。它把 **FlashAttention** 的 GEMM-Softmax-GEMM 流水线拆解为可单独编排的指令序列，实现持久化的、逐块计算的融合流水线。

- **打通内联计算的数据通路**：
  - 传统架构中，非线性操作（如 max、sum、exp）需要把数据踢给独立的 Vector 单元处理，再存回。
  - PLENA 将 **Vector Unit** 和 **Scalar Unit** 紧密耦合在计算单元旁边，且 **VLEN**（向量宽度）可配置以匹配 **FlashAttention** 的分块大小。中间的归约和指数运算直接在片上高精度完成，彻底切断了与 **HBM** 的中间数据交互。
