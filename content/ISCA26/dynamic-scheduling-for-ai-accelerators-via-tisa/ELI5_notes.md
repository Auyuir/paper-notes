# Dynamic Scheduling for AI Accelerators via TISA 通俗讲解

### 0. 整体创新点通俗解读

**痛点直击**

现代 AI 加速器（如 GPU、TPU）内部其实是“异构多引擎”结构，包含 Tensor Units（算力）、Vector ALUs（处理 Softmax 等）和 DMA Engines（搬数据）。为了协调这些引擎，传统做法依赖**Static Scheduling**（编译时静态调度）。

这种模式在真实运行时非常“难受”，核心痛点在于：
- **语义丢失导致过度同步**：编译器在把高层算子（如 GEMM、Softmax）降级成底层指令时，为了追求极致的底层优化，把算子边界、数据依赖等“语义信息”全抹平了。硬件只看到一串无意义的指令流，为了保底，只能在迭代之间插入死板的**Synchronization Barriers**（同步屏障）。
- **无法应对运行时波动**：静态调度假设每条指令的执行时间是确定的。但真实硬件会遭遇 DMA 背压、Cache 冲突、降频等波动。一旦某个引擎卡顿，整个流水线就会因为死板的屏障而停顿等待，其他引擎只能干瞪眼，硬件利用率极低。
- **跨迭代并行被锁死**：比如 FlashAttention 里的 $S_i$（Softmax，用 Vector 单元）和 $M0_{i+1}$（下一轮的矩阵乘，用 Tensor 单元）本可以并行，但因为静态调度强制按轮次同步，硬生生被串行化。

![](images/05cf9efabbd8f0b944e16a47971964207912f2e2cdee9abb672aaf4eef1e2480.jpg) *Fig. 2: Different execution manners of the fused tiles shown in Figure 1. Shaded regions $( E _ { x } )$ represent latency saved through scheduling. Vertical dashed lines denote synchronization barriers between iterations imposed by static scheduling. $\begin{array} { r } { E _ { 0 } + E _ { 2 } = E _ { 1 } + E _ { 3 } } \end{array}$ illustrates the equivalent latency savings achievable by dual-stage or triple-stage dynamic scheduling.*

---

**通俗比方**

这就好比一家拥有**烤肉区**（Tensor）、**切菜区**（Vector）和**传菜区**（DMA）的流水线餐厅。

- **传统做法（Static Scheduling）**：餐厅经理提前写死了一张时间表。“烤肉区做完第1单，切菜区马上切第1单的葱花，传菜区再端走。在切菜区切完之前，烤肉区绝对不许动第2单的肉！”结果呢？切菜区偶尔刀钝了慢了一秒，烤肉区明明可以提前腌第2单的肉，却因为经理的死命令，只能干等。经理为了防止出错，把每一步的依赖都锁死了。
- **TISA 的做法（Dynamic Scheduling）**：经理不写死时间表了。他给每盘食材贴了个电子标签：“这是烤肉任务，需要烤肉区，依赖切菜区的葱花”。餐厅大堂经理（硬件调度器）盯着大屏幕，只要看到烤肉区闲了，且第2单的肉和调料齐了（不依赖葱花），立刻放行让烤肉区开干。这就叫** readiness-driven**（就绪驱动），彻底打破了死板的轮次屏障。

---

**关键一招**

作者并没有去重造一个极其复杂的硬件乱序调度器，而是巧妙地在“编译器”和“硬件”之间，插了一个极其轻量但语义完整的**Tile-level ISA (TISA)** 抽象层。

这个巧妙的逻辑转换在于：**把“调度决策”的时机，从编译时推迟到了运行时，且依赖的是保留住的 Tile 语义，而非原始指令。**

具体扭转的流程如下：
- **编译端扭转**：传统编译器会把算子彻底打碎成底层指令。TISA 编译器则“适可而止”，它只把算子降级到 **Tile** 粒度，不再往下拆。并且把每个 Tile 的 `OpType`（算子类型）、`TileMem`（内存区间）、`UnitMap`（资源意图）打包成 TISA 指令直接喂给硬件。
- **硬件端扭转**：硬件不再无脑执行指令流，而是内置了一个**Semantics-Aware Runtime Scheduler**。它拿到 TISA 指令后，利用 `RAW/WAR/WAW`（读后写/写后读/写后写）规则做区间冲突检测。
- **动态发射**：只要当前 Tile 的数据区间没冲突，且对应的异构单元空闲，调度器就立刻**Out-of-order**（乱序发射）这个 Tile。这就天然实现了跨算子、跨迭代的重叠执行，完全不需要程序员手写 `bar.sync`。

![](images/5922dd03371347d5c15a0cba08b5218ab9af0684e9ff047eb72ebe4fbcccddf8.jpg) *Fig. 3: The overall architecture of our integrated framework.*

通过这一招，TISA 实现了在硬件资源较弱的情况下，反超算力更强的 GPU。以下是在 FlashAttention-3 上的硬件利用率对比，TISA 凭借动态调度在 head dim 128 配置下比 H100 的 SOTA 实现高出 **26.4%**：

| 硬件平台 | 核心机制 | 相对利用率表现 (head dim 128) |
| :--- | :--- | :--- |
| **NVIDIA H100** | Static Pipeline (TMA/WGMMA + bar.sync) | Baseline |
| **Epoch (TISA)** | Dynamic Scheduling (1:8 ratio) | **+26.4%** |
| **Epoch (TISA)** | Dynamic Scheduling (1:16 ratio) | **+15.7%** |

**总结一句话**：TISA 的核心贡献就是证明了**“只要把语义信息保留到运行时，硬件自己比编译器更懂怎么排流水线”**。通过剥离死板的同步屏障，让异构引擎基于数据就绪状态自由重叠，从而榨干硬件的每一滴算力。

### 1. Tile-Level Instruction Set Architecture (TISA)

**痛点直击**

之前的做法在异构加速器上“很难受”，根本原因是**软件和硬件之间存在语义断层**。
- 编译器在 lowering 阶段，把高层的算子打碎成了底层扁平的指令流。
- 硬件只看到一堆“盲盒”指令，根本不知道哪条指令属于哪个算子，也不知道它们之间到底有没有真实的数据依赖。
- 为了保证不出错，编译器只能采取最保守的策略：**插入死板的同步屏障**。
- 结果就是：即使下一个迭代的 GEMM 和当前迭代的 Softmax 毫无数据冲突，硬件也不敢并行执行，必须卡在 barrier 上干等。一旦遇到 DMA 延迟或 Cache 冲突，整个流水线直接停摆。

---

**通俗比方**

这就像一个大型餐厅的后厨流水线。
- 传统做法是下发一张**死板的步骤单**：“第一步切菜，第二步炒菜，第三步装盘”。并且规定，必须等所有菜切完，才能开火炒。哪怕切菜工闲得抠脚，炒菜工也得干等。
- **TISA** 的做法是给每个任务发一张**带语义的智能工牌**。工牌上写着：“我是切土豆丝，我需要案板，我切的是 3号筐的土豆”。
- 切菜工只要切完 3号筐的一半，立刻广播：“3号筐前半部分已就绪”。炒菜工听到后，立刻拿去炒，完全不需要等整个流程走完。
- 这样一来，切菜和炒菜就能根据**实时就绪状态**无缝重叠，彻底消灭了干等的时间。

---

**关键一招**

作者并没有让编译器去死磕静态调度，也没有让硬件去猜指令依赖，而是**在中间插了一个 TISA 语义层**。
- 编译器在 lowering 时，不再把算子彻底打碎，而是停在 **Tile 粒度**。
- 保留三大核心语义：
  - **OpType**：算子类型（告诉硬件该派给 Tensor 还是 Vector 单元）。
  - **TileMem**：内存范围（告诉硬件数据有没有重叠，能不能并行）。
  - **UnitMap**：资源意图（告诉硬件需要哪些计算/搬运资源）。
- 硬件调度器直接读取这些元数据，通过简单的区间重叠测试，就能动态判断“现在能不能发车”。
- 这一招直接**把显式的同步屏障给干掉了**，让硬件能根据数据就绪情况，自动、合法地乱序执行。

![](images/5922dd03371347d5c15a0cba08b5218ab9af0684e9ff047eb72ebe4fbcccddf8.jpg) *Fig. 3: The overall architecture of our integrated framework.*

| 特性 | 传统 ISA | TISA |
| --- | --- | --- |
| **语义保留** | 丢失，降维成扁平指令 | 保留，停在 Tile 粒度 |
| **依赖管理** | 静态插入 Barrier | 硬件动态检查 RAW/WAR/WAW |
| **执行模式** | 保守的顺序流水 | 基于就绪状态的动态重叠 |

### 2. Semantics-Preserving Compiler Stack

**痛点直击**

之前的深度学习编译器在把高层算子（比如GEMM和Softmax）翻译成底层机器码时，会经历层层降级。在这个降级的过程中，算子原本的边界、谁依赖谁、需要什么硬件资源这些**高层语义信息全被抹平了**。

硬件最终看到的只是一长串平铺的、不透明的指令流。它不知道哪条指令属于哪个算子，也不知道哪些指令之间是真正的数据依赖，哪些只是编译器排出来的先后顺序。

为了不出错，硬件只能采取最保守的策略：用**显式的同步屏障**把指令死死卡住。这就导致硬件在遇到运行时的波动（比如Cache miss或DMA延迟）时，无法灵活调整执行顺序，只能干等，造成大量闲置气泡。

---

**通俗比方**

就像你是一个工厂调度员（硬件调度器），手下有三个车间：加工车间、包装车间和搬运车间。

以前的编译器就像一个极其死板的跟单员，他把客户的订单拆解成几百张工序单，但**不告诉你这些单子属于哪个产品，也不告诉你包装和搬运能不能同时进行**。他只给你一个严格的执行顺序表，并在关键节点画上红线：“必须等前面这步彻底做完，才能做下一步”。

结果就是，加工车间在等搬运车的时候，包装车间明明闲着，也不能去帮忙处理下一个订单的半成品。因为跟单员没告诉你它们之间没冲突，你也不敢冒这个险。

而这篇论文的“语义保留编译器”，就像换了一个聪明的跟单员。他在每一张工序单上不仅写清了要做什么，还特别标注了：**这是哪个产品的哪一步、它依赖哪个半成品、它需要哪个车间、它的数据存在哪个货架**。这样你作为调度员，一看就知道“哦，包装车间现在闲着，而且下一个订单的半成品已经搬到货架上了，没有冲突，可以直接开工！”

![](images/5922dd03371347d5c15a0cba08b5218ab9af0684e9ff047eb72ebe4fbcccddf8.jpg) *Fig. 3: The overall architecture of our integrated framework.*

---

**关键一招**

作者并没有重头设计一套全新的硬件架构，而是巧妙地在传统编译流程和硬件之间**插了一个“语义备忘录”**。

具体来说，他们设计了一个**Tile-level Instruction Set (TISA)**。在编译器逐层降级的过程中，不再把算子彻底打碎成无意义的指令流，而是**停留在Tile粒度**。

编译器把算子的身份、依赖类型（RAW/WAW/WAR）、需要的硬件单元和内存范围，像打包行李一样，**原封不动地贴在TISA指令上**，一路带到硬件runtime。

这样一来，硬件调度器拿到的不再是盲盒指令，而是带说明书的任务包。它可以根据运行时的实际情况，动态判断哪些Tile的数据已经ready，哪些硬件单元空闲，从而**跨越operator边界和iteration边界**，实现真正的动态乱序执行，彻底干掉了那些保守的同步屏障。

### 3. Decentralized Semantics-Aware Hardware Scheduler

**痛点直击**

之前的静态调度在应对**运行时变数**时极其难受。编译器为了保底，只能在代码里插入大量保守的同步屏障。
- 这导致硬件单元经常出现**结构性闲置**：明明 Tensor 单元闲着，但因为 Vector 单元在算 Softmax，且编译器设定了必须等这一轮迭代结束才能开始下一轮，Tensor 单元只能干等。
- 一旦运行时发生 DMA 背压或 Cache 冲突，执行时间发生偏移，静态排好的流水线就会“对不上拍”，导致流水线气泡被放大，硬件利用率暴跌。
- 根本原因在于：指令降级到底层后，**丢失了高层语义**，硬件只看到一堆无意义的指令流，为了不出错，只能死板地按顺序执行。

---

**通俗比方**

这就像一个后厨的流水线：切菜工、炒菜工、传菜工。
- 传统静态调度是死板的“工序手册”：切完这批土豆，必须等炒菜工炒完，切菜工才能切下一锅。哪怕炒菜工去上厕所，切菜工也只能干等。
- **Decentralized Semantics-Aware Hardware Scheduler** 就像给每个工位配了一个**智能对讲机**。
- 这个对讲机不仅去中心化，而且懂语义。它不喊“我要用刀”，而是喊“我要切土豆”。
- 只要切菜工听到“我要的土豆没被别人占着，且我的案板空着”，他根本不管手册怎么写，直接开干下一批。这就把跨迭代的并行度给“挤”出来了。

![](images/05cf9efabbd8f0b944e16a47971964207912f2e2cdee9abb672aaf4eef1e2480.jpg) *Fig. 2: Different execution manners of the fused tiles shown in Figure 1. Shaded regions $( E _ { x } )$ represent latency saved through scheduling. Vertical dashed lines denote synchronization barriers between iterations imposed by static scheduling. $\begin{array} { r } { E _ { 0 } + E _ { 2 } = E _ { 1 } + E _ { 3 } } \end{array}$ illustrates the equivalent latency savings achievable by dual-stage or triple-stage dynamic scheduling.*

---

**关键一招**

作者没有去死磕编译器的排布算法，而是巧妙地在编译器和底层硬件之间插入了 TISA，把**排班权下放给硬件**。
- 编译器不再负责死板地排定执行顺序，而是通过 TISA 指令保留**语义信息**（算子类型、数据内存范围、资源需求）。
- 硬件调度器在运行时直接读取这些语义，进行**类型化依赖分析**。
- 只要数据没冲突、单元有空闲，调度器就立刻乱序发射 Tile，把原本被静态屏障挡住的跨迭代并行度给释放了。

| 调度策略 | 决策时机 | 语义保留 | 跨迭代并行 | 容错性 |
| :--- | :--- | :--- | :--- | :--- |
| 静态流水线 | 编译时 | 丢失 | 受限 | 极差 |
| **动态语义调度** | **运行时** | **完整** | **自由** | **极强** |
