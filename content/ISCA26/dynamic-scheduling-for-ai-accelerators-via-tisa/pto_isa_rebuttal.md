# 对 PTO-ISA Novelty 主张的技术反驳

## 审稿结论

该投稿把 PTO 的三个不同层次混为一谈：

1. Tile 寄存器操作数是数据命名与合法性接口；
2. Cell 是 Tile 的物理布局和有效性粒度；
3. 动态调度器则需要独立的就绪跟踪、资源仲裁、依赖消解和发射机制。

PTO 规范确实覆盖前两者，但这并不推出第三者。因而，“Tile register dataflow + Cell lifetime”并不能构成对 TISA 核心创新的实质超越，也不能证明 PTO 已经实现了面向异构单元的动态调度。

## 1. Tile register 不是动态调度语义

投稿把显式 Tile register 操作数直接等同于 CPU 风格的 scoreboard/Tomasulo 接口，这是概念跳跃。

PTO 的 `TileInstructionOperands` 只是统一的解码操作数载体，包含 `destination0..2` 和 `source0..8` 等索引；`TMATMUL` 也只是规定 `destination0`、`source0` 和 `source1` 的操作角色。它说明“该操作读写哪些架构 Tile”，没有说明：

- 多条在途指令如何在不同执行引擎间选择和发射；
- 一个 Tile register 是否存在多个可同时在途的 value version；
- register reuse 如何进行物理重命名；
- wakeup/select 如何连接 VEC、TLSU、CUBE 和 SFU；
- 何时可以越过程序顺序而执行后继操作。

因此，register naming 只能提供潜在的 RAW 关系。它不能自动提供 TISA 所定义的完整调度契约：算子类型、执行单元亲和性、内存范围、资源需求、重排约束和运行时反馈。

更关键的是，PTO 的规范明确把“physical Tile allocation、backend intrinsics、latency、throughput 和 target scheduling”排除在可移植 ISA 契约之外（`pto-spec/README.md`）。既然 target scheduling 不属于 PTO ISA，本投稿就不能仅凭 PTO 的寄存器操作数声称其 ISA 已提供 Tomasulo 风格异构动态调度。

## 2. “可重命名”不是 PTO 规范中的已证明机制

投稿声称 PTO 可把同一架构 Tile register 映射到不同的物理 Tile register/value version，以消除 WAR/WAW。但 PTO 的架构状态不是这样定义的。

PTO 的 Tile 状态是 64 个扁平的 T/U/M/N Tile registers，并带有 Local/Shared ownership、descriptor、allocation、valid region 和 definedness。`TMATMUL` 的规范描述的是向一个 newly published destination Tile 写入结果；它没有定义通用的 rename table、physical Tile-register namespace、版本选择规则或基于版本的消费者重绑定。

PTO 中出现的 “renamed destination” 也不能被解释为 CPU 式寄存器重命名。它主要是 bundle 提交时为新 Tile 分配架构可见目的地，解决 Tile allocation 和 publication 的原子性。例如，Cell rearrangement schema 在解析目的地时寻找未分配 Tile，并将结果写入 `destination`；这属于目的地分配，不是允许同一逻辑定义存在多个可独立调度版本的通用物理重命名机制。

没有这些机制，投稿关于“后续迭代同名 Tile register 可无 WAR/WAW 地并发”的结论没有规范依据。反过来，若实现自行加入 rename table 和 value-version tracking，那是 PTO 之外新增的微架构和调度方案，不能作为 PTO-ISA 本身的 novelty。

## 3. Cell 几何不等于 Cell 级执行或生命周期调度

PTO 的 `CUBE Cell` 定义了 128-byte Cell、布局相关的行列几何、repeat 数量、Cell 数量、payload index 和容量计算。它解决的是：

- Tile 如何布局；
- 逻辑坐标如何映射到 payload；
- valid region 与 padding 如何解释；
- 需要多少存储空间。

这些都是数据表示和合法性问题，不是运行时调度问题。一个 Cell 可寻址或可计算，并不意味着硬件已经拥有按 Cell 粒度的 producer-consumer wakeup，也不意味着 Cell storage 可以在不同 generation 之间安全提前复用。

投稿声称 PTO 的 `B.ASSEMBLE` 已经形成“生产者声明 Cell 区间、消费者记录 Cell 集合、全部就绪后唤醒”的两级数据流模型，必须区分规范中实际存在的机制：

- `covered_cells` 和 `ready_cells` 是 generation 状态中的覆盖/完成记录；
- `BundleConsumerDependency` 是 B.ASSEMBLE 相关的 portable readiness carrier；
- `CompleteBundleLocalGenerationWriterEvent` 是完成事件，用于更新 ready 状态，并在 LAST 和完整 readiness 满足后进行一次性 publication。

这是一套针对 assembled generation 的提交与可见性协议，不是通用的所有 Tile 指令的 Cell 级动态调度器。尤其不能从它推出跨 CUBE、VEC、TLSU、SFU 的通用 Cell 级 wakeup/select、资源仲裁或乱序发射。

此外，PTO 对 Shared Tile 还明确维护 `whole_parent_ready` 和 `published`，Shared consumer 必须等待 whole-parent readiness 与 publication。这与投稿所宣称的“局部 Cell 到达即可普遍唤醒消费者”并不一致：在至少一类关键 Shared 数据路径上，规范选择的是 parent-level ready，而不是任意局部 Cell 的通用早发射语义。

## 4. PTO 没有解决 TISA 所针对的异构资源调度问题

TISA 的核心不是单纯表达数据流，而是把运行时需要的调度语义交给硬件消费：

- `OpType` 表达操作语义和执行类别；
- `UnitMap` 表达资源亲和性、数量和目标单元；
- `TileMem` 表达作用域和访问范围；
- 类型化依赖区分 RAW/WAR/WAW；
- 每单元 WQ/IQ 与 in-flight semantic table 支持局部仲裁；
- 反馈机制根据实际争用和延迟调整优先级。

PTO 的规范则定义四类执行引擎和大量 Tile 操作的分类，例如矩阵操作使用 CUBE、内存/传输使用 TLSU、逐元素操作使用 VEC、复杂专用操作使用 SFU。但“某条操作属于某个 engine”是静态分类，不等于存在运行时的跨 engine scheduler。PTO 的 `README` 甚至明确说 target scheduling 不在 portable contract 内。

因此，即使 PTO 的 Tile register 依赖可以用于构建单引擎或单流水线 scoreboard，它仍没有回答 TISA 的关键问题：当 CUBE tile 因内存或资源原因暂时停顿时，调度器如何识别并发就绪的 VEC/TLSU tile，如何跨操作和跨迭代发射，如何避免一个单元的队头阻塞另一个无关单元。

## 5. Cell 粒度并不能自动优于 Tile 粒度

投稿把更细的 Cell 粒度直接当成更高的调度质量，但粒度越细并不意味着语义越完整。

Cell 是布局依赖的物理组织单位。一个 Cell 的含义随 `CUBE_M16`、`CUBE_M32`、`CUBE_N8`、dtype 和 layout 改变；普通 element-wise 操作又以 logical coordinate 为语义，只有特定 rearrangement 操作才暴露 raw Cell order。这说明 Cell 并不是所有算子共享的稳定语义边界。

TISA 选择 Tile 作为调度粒度，原因正是 Tile 同时保留算子上下文、数据范围和资源意图，并能作为 Tensor、Vector、DMA 共同理解的调度单位。若把调度语义下降到 Cell，系统还必须额外定义：

- Cell 到算子语义的映射；
- 跨 layout 和 dtype 的别名规则；
- Cell 子集与资源占用的关系；
- partial result 的正确性与提交规则；
- 不规则/跨步访问的依赖表示。

PTO 当前规范没有将这些问题统一为跨异构单元的调度契约。因此 Cell 级状态最多是潜在的实现基础，不能被包装成已经完成的动态调度创新。

## 6. 与 TISA 的实质差异应如何界定

公平的技术定位应是：PTO 提供了丰富的 Tile 数据表示、操作编码、架构状态以及部分 Cell-range generation/readiness 协议；TISA 提供的是面向异构加速器运行时的语义调度层。

二者并非简单的“whole Tile vs Cell”竞争关系：

- PTO 的 register operands 可以作为 TISA `Operands` 的一种实现载体；
- PTO 的 Cell geometry 可以帮助实现更精确的 `TileMem` 或部分就绪描述；
- PTO 的 CUBE/VEC/TLSU/SFU 分类可以作为 `UnitMap` 的后端映射；
- 但 PTO 仍需要额外的 TISA-like runtime scheduler，才能获得跨单元、跨迭代、冲突感知的动态发射能力。

换言之，PTO 的这些机制是 TISA 可以利用的下层执行接口或增强实现，而不是对 TISA 核心创新的替代。投稿若不能提出独立的跨异构运行时调度协议，就不能以“register-based Tile dataflow”或“Cell-grained lifetime”证明其超越 TISA。

## 最终意见

该投稿最有价值的部分是指出了 Tile register versioning 和局部 Cell readiness 可能增强动态调度器。但其当前论证把“规范中存在操作数、Cell 和 generation 状态”错误地升级为“规范已经定义了可重命名的异构乱序调度 ISA”。

在不额外引入 PTO 规范之外的 rename、wakeup/select、跨 engine arbitration 和 runtime feedback 机制的前提下，其核心创新仍停留在：

1. 用寄存器名表达 Tile 数据流；
2. 用 Cell 描述 Tile 的布局和部分完成状态。

这两点都不能否定 TISA 首先提出的运行时可见 Tile 语义、跨异构单元动态调度以及语义保持编译器协同设计。更准确的结论是：PTO 可以作为 TISA 的一个潜在后端或底层载体，但目前没有技术依据声称 PTO-ISA 已经取代或实质超越 TISA。
