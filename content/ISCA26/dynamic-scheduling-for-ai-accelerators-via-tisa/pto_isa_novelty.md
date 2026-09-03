# PTO-ISA 相对 TISA 的 Novelty

TISA 已经说明：将 Tile 语义暴露给硬件，可以在 CUBE/Tensor、VEC/Vector 和 TLSU/DMA 之间实现运行时动态调度。因此，PTO-ISA 不应把“Tile 级动态调度”本身作为主要创新，而应强调一个更具体的 ISA 差异：**PTO 将 Tile 定义为可进行寄存器数据流执行的架构值，并进一步把 Tile 内部 Cell 纳入依赖和生命周期管理。**

## 1. Register-Based Tile Dataflow

PTO 的 Tile 操作通过显式 Tile registers 传递生产者和消费者关系，而不是主要依赖 TileMem 地址区间推断依赖。`pto-spec/docs/tile/model/state/types.md` 中的 `TileInstructionOperands` 将多个 source/destination Tile register 纳入统一操作数模型；`pto-spec/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL.md` 明确表达 `destination0 <- source0 × source1` 的数据流。

因此，PTO 硬件可以围绕 Tile register 构建真正的 scoreboard、wakeup/select 和 Tomasulo 风格的乱序执行，并通过寄存器重命名消除 Tile register 复用造成的 WAR/WAW 伪依赖。尤其在循环分块中，编译器通常会反复使用有限的 Tile register 名称；若没有重命名，后一轮对同一寄存器名的写入会被错误地串行化，形成与真实数据流无关的人工依赖。PTO 可以把这些写入映射到不同的物理 Tile register/value version，仅保留真正的 RAW 依赖。TISA 解决的是“哪些 Tile 可以调度”，PTO 进一步提供了“哪个架构值版本驱动哪个消费者”的数据流基础，使成熟 CPU ILP 机制能够自然迁移到异构 Tile 单元。

## 2. Cell-Grained Register Lifetime

PTO 的 Tile register 不是不可分割的黑盒 Tile。其物理组织以固定大小的 Cell 表达：`pto-spec/docs/tile/model/shape/cube-cell.md` 定义了由 Tile shape、layout 和 dtype 推导的 Cell 映射。在 `B.ASSEMBLE` 数据流中，生产者可以声明连续 Cell 区间，消费者则记录所需的 Cell 集合；消费者只有在所需 Cell 全部 ready 后才具备执行资格（`pto-spec/docs/block/model/state/types.md`、`pto-spec/docs/block/model/operands/portable-carriers.md`）。

这形成了 **Tile Register → Cell Register** 的两级依赖模型：指令级用连续 Cell-register 集合紧凑表达 producer 关系，硬件内部按 Cell 粒度 wake up consumer，并管理不同 Tile value generation 的生命周期。相比 whole-Tile readiness 或静态 barrier，PTO 可以让消费者在所需局部结果到达后立即唤醒，减少等待延迟，并允许 Cell storage 在 generation 之间更早复用，从而同时改善处理延迟和寄存器空间利用率。

## 总结

PTO-ISA 的核心主张不是“也能动态调度异构 Tile”，而是：**PTO 把动态调度建立在可重命名的 Tile-register 数据流和 Cell-grained value lifetime 之上。** 前者使能 scoreboard/Tomasulo、伪依赖消除和跨 CUBE/VEC/TLSU 的乱序发射；后者使能周期级消费者唤醒与精细的寄存器空间复用。由此，PTO 将 TISA 的语义感知调度推进为一种面向异构 AI 加速器的两级寄存器数据流 ISA。
