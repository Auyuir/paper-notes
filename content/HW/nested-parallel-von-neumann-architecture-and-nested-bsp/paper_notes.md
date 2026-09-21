# Nested Parallel von Neumann Architecture and Nested BSP 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Liao Heng

**发表期刊/会议 (Journal/Conference)**: unknown

**发表年份 (Publication Year)**: 2026

**研究机构 (Affiliations)**: Huawei Technologies Co., Ltd.

---

## 1. 摘要

**目的**

- 回答一个核心问题：当处理器规模达到**十万乃至百万级**时，如何让它们仍然构成**一台计算机**，而非松散机器的拼凑
- 将经典 **BSP** 递归扩展为 **Nested BSP**，将冯·诺依曼单机架构扩展为**嵌套并行冯·诺依曼架构**（Nested Parallel von Neumann Architecture）
- 破解传统互连的“悬崖”问题：跨机箱一步，带宽骤降一个数量级、延迟成倍上升
- 以华为 **SuperNode** 集群为载体，提出 **Unified Bus** 作为端到端落地互连技术

---

**方法**

理论框架——Nested BSP：

- 每层执行经典 BSP 四步循环：**并行计算 → barrier → 交换与聚合 → 进入下一 phase**
- 硬约束：各层单元皆为**对等体**，无独占 barrier 的 master、无仅能服从的 slave；**并行性嵌套，控制权不嵌套**
- 软件嵌套（外→内）：DP → PP → EP → FSDP → CP → TP，每层内含多重子并行单元，构成并行性来源
- 硬件嵌套（内→外）：CORE → chiplet → 封装 → 板 → 机架 → SuperNode → 数据厅 → 自治域 → 数据中心 → 数据中心间
- 软件嵌套与硬件嵌套**逐层对齐**，任一层错位则并行溃散；**τ Scaling law** 在每层折叠时间常数，六层**乘性折叠**

![](images/x1.png) *Figure 1:Software nesting dolls: Nested BSP. From outermost to innermost, the levels are data parallel (DP), pipeline parallel (PP), expert parallel (EP), fully sharded data parallel (FSDP), context parallel (CP), and tensor parallel (TP).*

![](images/x2.png) *Figure 2:Hardware nesting dolls: the body of the machine, from chip package to autonomous zone. One autonomous zone holds many data halls, one hall holds many SuperNodes, and one SuperNode holds many racks, all the way down to the package.*

Unified Bus 六项硬件设计原则：

- **单一协议**：从封装到自治域端到端统一，消除跨边界协议转换开销（传统集群中 80%+ 能耗用于数据搬运，其中相当部分损耗于协议转换）
- **内存语义**：原生 load/store + 硬件一致性，取代 TCP/IP 软件栈与 RDMA 的“发送-等待”消息语义
- **对等架构**：CPU、NPU、内存、存储、NIC 同总线对等，任一节点可发起、可响应，消除中心拥塞
- **铜近光远**：近处铜保延迟、远处光保规模，介质切换协议不变；光侧选择 **NPO**（near-package optics）
- **物理疏散、逻辑紧致**：不追求 megawatt 机架，以光将疏散部署的节点绑成一台机器
- **每代只攻坚少数硬仗**：五个 90% 成功率目标可交付，二十个的联合成功率趋近于零

工程实践三要点：

- 交换芯片**高 radix**：同规模下减少中间层级，节省延迟与功耗
- 计算芯片同样**高 radix**：单跳覆盖 = 交换端口数 × 芯片端口数，二者均为乘数，直接放大 one-hop SuperNode 规模
- 高密度 NPO 将电光转换边界从 **1 m 内移至 10 mm**（封装边缘），铜段仅剩最后几厘米

![](images/x3.png) *Figure 3:Design freedom from NPO: package at ten millimeters, board at ten centimeters, rack at one meter, SuperNode at ten meters, data hall at a hundred meters, and data center at a kilometer. Moving the electro-optical boundary inward from 1 m to 10 mm relaxes every outer layer—physically sparse, logically tight.*

---

**结果**

关键性能指标：

| 指标 | 数值 |
|---|---|
| 通信往返延迟 | 从数十 μs 降至 **约 100 ns**（约 **500 倍**提升） |
| 单芯片 I/O 带宽 | **7.2 Tbps** 级 |
| SuperNode 节点规模 | **>8000 节点** |
| 聚合内存带宽 | **6.7 PB/s** |
| 全互连带宽 | **400 Tbps** |
| 全系统 barrier | **<10 μs** |

NPO 与 CPO 方案对比：

| 维度 | NPO | CPO |
|---|---|---|
| 损耗消除（全电路径 >20 dB） | 一步消除 **9–11 dB** | 在此基础上仅多约 3 dB |
| 延迟 | 十纳秒级 | — |
| 成本 | 比 CPO **低 40%+** | 较高 |
| 可维护性 | 光引擎可独立构建、测试、更换 | — |
| 生态 | 已立项 **OIF** 项目，数十家伙伴参与 | — |

电光边界内移两个数量级后的尺度放宽：

| 物理层级 | 尺度 |
|---|---|
| 封装 | 10 mm |
| 板 | 10 cm |
| 机架 | 1 m |
| SuperNode | 10 m |
| 数据厅 | 100 m |
| 数据中心 | 1 km |

- 各外层尺度逐级放宽十倍，散热、供电、可靠性无需同时对抗物理极限
- **256K 节点级**单机 Super AI 计算机系统部署中，为“一台机器”而非二十万台机器的粘合
- **Unified Bus Protocol** 规范已开放发布

---

**结论**

- **Nested BSP + 嵌套并行冯·诺依曼架构 + Unified Bus** 三者结合，实现**多处理器仍是一台计算机**、每个节点均为总线上的对等体
- τ Scaling law 在每层折叠时间：单层折叠有限，六层乘性叠加，整体任务时间真正下降
- **“physically sparse, logically tight”**（物理疏散、逻辑紧致）确立为新设计范式：空间密度不再是必攻极限
- 用户价值：加芯片即加算力，芯片算力不再消耗于数据搬运，**单 token 成本下降**
- 开放标准定位大于任何单一公司：接入者获得同等延迟与同等线性扩展性

---

## 2. 背景知识与核心贡献

**研究背景**

- **冯·诺依曼架构**定义了“如何造一台计算机”，八十年来处理器沿此路径不断增强单点性能，但其原初设计只针对**单机**场景，从未回答“百万处理器如何协同为一台机器”的问题
- 大规模 AI 计算的竞争逻辑已发生根本转变：不再是“一颗更强的处理器”之争，而是**十万乃至百万级处理器在一个指挥体系下能否仍是一台计算机**之争
- HPC 领域早前已触及该瓶颈，Leslie Valiant 与 Bill McColl 提出的 **BSP（Bulk Synchronous Parallel）模型**将计算切分为阶段，缓解了部分问题，但经典 BSP 缺乏递归嵌套能力
- 传统互连技术存在致命的**“悬崖”效应**：跨出机箱边界后，带宽骤降一个数量级、延迟数倍上升；机箱内是总线协议，机箱外是网络协议，每次边界穿越都需“拆包—检查—重打包”

**研究动机**

- 核心问题只有一个：当处理器数量达到十万、百万量级时，**体系结构设计方法是否依然成立**
- 大规模集群中 **80% 以上的能耗消耗于数据搬运**，其中很大比例损耗在协议转换边界上；TCP/IP 软件栈往返延迟达数十微秒，Barrier 与 Reduce 操作随规模扩大而急剧恶化
- 传统设计中的**主从架构**（CPU 为主、加速器为从；主机为主、设备为从）是规模化的结构性障碍：中心拥塞导致“一加一小于二”
- **铜互连已触物理墙**：单线速率提升导致线径变粗、传输距离缩短；千根铜缆捆扎后无法施工安装
- 软件侧的六层并行策略（DP、PP、EP、FSDP、CP、TP）与硬件层次（封装至自治域）若不能**逐层对齐**，任何一层的错位都会导致并行度无法兑现

**核心贡献一：Nested BSP（软件侧理论扩展）**

- 将经典 BSP 递归嵌套：最外层是全军 BSP，内部每一次“并行推进”本身又是一支更小的部队，在其自身范围内重演**并行计算 → Barrier → 交换与聚合 → 下一阶段**四步，层层递归
- 相对经典 BSP 增加一条硬约束：**每一层的单元皆为对等体**——不存在必须掌控所有 Barrier 的主机，也不存在只能服从的从机，即“并行可嵌套，权力不嵌套”

![](images/x1.png) *Figure 1:Software nesting dolls: Nested BSP. From outermost to innermost, the levels are data parallel (DP), pipeline parallel (PP), expert parallel (EP), fully sharded data parallel (FSDP), context parallel (CP), and tensor parallel (TP).*

- 六层嵌套自外向内依次为 **Data Parallel（DP）→ Pipeline Parallel（PP）→ Expert Parallel（EP）→ Fully Sharded Data Parallel（FSDP）→ Context Parallel（CP）→ Tensor Parallel（TP）**
- 关键洞察：每层大娃娃内部包含**多个**而非一个小娃娃——一个 DP 持有多个 PP，一个 PP 持有多个 EP——这种**多重性正是并行度的来源**

**核心贡献二：Nested Parallel von Neumann Architecture（硬件侧架构扩展）**

- 将冯·诺依曼“单机”定义扩展为“一个处理器军团如何仍是一台计算机”：**软件嵌套与硬件嵌套逐层对齐、端到端内存语义、总线上人人对等、物理稀疏而逻辑紧致**
- 硬件嵌套自内向外为 **CORE → Chiplet → 芯片封装 → 板 → 机柜 → SuperNode → 数据大厅 → 自治域 → 数据中心 → 数据中心间**

![](images/x2.png) *Figure 2:Hardware nesting dolls: the body of the machine, from chip package to autonomous zone. One autonomous zone holds many data halls, one hall holds many SuperNodes, and one SuperNode holds many racks, all the way down to the package.*

- 与 **τ Scaling Law** 形成配套：每一层的 Nested BSP 与每一个物理尺度都在“折叠时间”——单一层折叠有限，**六层嵌套折叠呈乘性叠加**，整体时间开销才能真正下降
- 嵌套遵循分形逻辑：越外层规模越大、距离越长、带宽越低、延迟越高，这是被接受的物理规律；关键是将“悬崖”铺成“缓坡”，使指令逐层下传时**不缺一层**

**核心贡献三：Unified Bus 及六项硬件设计原则**

| 设计原则 | 对治的旧问题 | 关键收益 |
|---|---|---|
| **单一协议**（封装至自治域端到端） | 机箱内总线 / 机箱外网络的双协议转换税 | 全程无协议转换 |
| **内存语义**（原生 load/store，硬件维护一致性） | TCP/IP 数十微秒软件往返、RDMA 仍是消息模型 | 通信往返降至**约 100 纳秒**，约**五百倍**提升；单芯片 I/O 带宽达 **7.2 Tbps** 级 |
| **对等架构**（CPU/NPU/内存/存储/网卡同总线） | 主从模式中心拥塞 | 每层 Barrier/聚合无需回中心，**无单点瓶颈** |
| **铜近光远**（NPO 近封装光学） | 铜互连物理墙 | 单步消除 **9–11 dB** 损耗；十纳秒级延迟；成本较 CPO **低 40%+**；已成 **OIF 标准**项目 |
| **不挤单帐篷** | 追求兆瓦级机柜导致散热、供电、占地失控 | 物理稀疏、逻辑紧致（stand sparsely, compute tightly） |
| **每代只打少数硬仗** | 同时挑战 20 项 90% 成功率的技术，联合成功率趋近于零 | 选五项关键突破即可交付 |

**核心贡献四：工程落地的三项实践要点**

- **交换芯片必须高 radix**：junction 出口越多，层级越少，延迟与功耗节省是实打实的
- **计算芯片也必须高 radix**——这是常被忽视的一项：芯片本身应成为 junction，且**单跳覆盖规模 = 交换芯片端口数 × 芯片端口数**，两侧皆为乘数；芯片 radix 翻倍则单跳 SuperNode 规模翻倍，而最内层、最高频的 Nested BSP 层恰落在该覆盖范围内
- **高密度 NPO 将光电转换边界从约 1 米内移至封装边缘 10 毫米**：铜只走最后几厘米，其余交给光；边界内移后，**每一外层均获十倍尺度放松**，板级不再对抗插损，机柜不必堆叠到兆瓦，冷却、供电、可靠性三道坎无需同时对抗物理极限

![](images/x3.png) *Figure 3:Design freedom from NPO: package at ten millimeters, board at ten centimeters, rack at one meter, SuperNode at ten meters, data hall at a hundred meters, and data center at a kilometer. Moving the electro-optical boundary inward from 1 m to 10 mm relaxes every outer layer—physically sparse, logically tight.*

**核心指标与落地成果**

| 指标 | 数值 |
|---|---|
| SuperNode 节点规模 | **8000+ 节点** |
| 聚合内存带宽 | **6.7 PB/s** |
| 全互连带宽 | **400 Tbps** |
| 全军 Barrier 延迟 | **< 10 微秒** |
| 正在部署的系统 | **256K 节点级** Super AI Computer（一台机器，而非二十万台机器的黏合网络） |
| 协议开放性 | **Unified Bus Protocol 规范**已开放发布，任何接入者获得同等延迟与线性度 |

**总结**

- 本文沿两条主线完成体系结构理论的递进扩展：**BSP → Nested BSP**（软件战场计划）、**冯·诺依曼架构 → Nested Parallel von Neumann Architecture**（硬件军队之躯），二者由 **Unified Bus** 端到端落地连接
- 最终命题的答案是：**多数处理器仍是一台计算机，且总线上每一个节点都是对等体**

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**核心命题**

本文的整体技术架构可概括为“**双巢对齐、一线贯通**”：软件侧的 **Nested BSP** 与硬件侧的 **Nested Parallel von Neumann Architecture** 逐层嵌套并一一对应，由 **Unified Bus (UB)** 从 package 到 autonomous zone 端到端粘合，各层时间折叠由 **τ Scaling law** 支配并呈**乘积效应**，最终使十万乃至百万级处理器在逻辑上仍是**一台计算机**——而非二十万台机器的粘合网络。

---

**软件侧架构：Nested BSP**

- 将 Valiant/McColl 的经典 **BSP (Bulk Synchronous Parallel)** 模型递归嵌套化
- 经典 BSP 的四步循环：
  - 并行计算
  - Barrier 同步
  - 交换与聚合
  - 进入下一阶段
- 嵌套规则：外层 BSP 的每一次“并行推进”本身又是一支更小的部队，在其作用域内重演同样的四步，逐层递归直至最内层
- 关键硬约束：**各层单元均为对等体**——不存在必须独占 Barrier 的 master，也不存在只能服从的 slave；**并行性嵌套，控制权不嵌套**
- 并行度的来源是**每层内的多重性**：一个 DP 内含多个 PP，一个 PP 内含多个 EP，一个 EP 内含多个 FSDP 分片

![](images/x1.png) *Figure 1:Software nesting dolls: Nested BSP. From outermost to innermost, the levels are data parallel (DP), pipeline parallel (PP), expert parallel (EP), fully sharded data parallel (FSDP), context parallel (CP), and tensor parallel (TP).*

---

**硬件侧架构：Nested Parallel von Neumann Architecture**

- 硬件本体同为嵌套结构，由内向外：CORE → chiplet → chip package → board → rack → SuperNode → data hall → autonomous zone → data center → inter data center
- 本文设计焦点为中间层段：**chip package 至 autonomous zone**（更内层是单兵训练，更外层是跨园区通信）
- 物理分形规律：
  - 越靠外：尺度越大、距离越长、带宽越低、延迟越高
  - 越靠内：尺度越小、距离越短、带宽越高、延迟越低
- 每一硬件层均部署更多**对等并发单元**，且软件巢与硬件巢必须逐层贴合、无间隙——任何一层错位都会导致并行单元失效

![](images/x2.png) *Figure 2:Hardware nesting dolls: the body of the machine, from chip package to autonomous zone. One autonomous zone holds many data halls, one hall holds many SuperNodes, and one SuperNode holds many racks, all the way down to the package.*

---

**连接层：Unified Bus 的六项硬件准则**

定位：把过去的“带宽悬崖”铺成“缓坡”，使命令逐层下传而不在任一边界断裂。

- **单一协议端到端**：从 package 到 autonomous zone 一套协议，消除机箱内总线与机箱外网络的双语言转换（传统方案中集群 80% 以上能耗用于数据搬运，其中大量损耗于协议转换点）
- **内存语义**：原生 load/store，一致性由硬件维护，取代 TCP/IP/RDMA 的“发消息-等回复”报文语义；通信往返从数十微秒降至约 **100 ns**（约 **500 倍**提升），单芯片 I/O 带宽达 **7.2 Tbps** 量级
- **拆除 master–slave**：CPU、NPU、内存、存储、NIC 同总线对等，任意节点可发起、可响应；各 Nested BSP 层的 Barrier 与聚合无需向中心回报，系统扩展无单一咽喉
- **铜近光远、语言不变**：近距用铜保延迟，远距用光保规模，介质切换时协议保持不变
- **NPO 路线**：选用 near-package optics 而非 CPO，具体权衡如下：

| 维度 | NPO | CPO |
|---|---|---|
| 单步消除插损 | 9–11 dB（全电路径损耗 20 dB 以上） | 仅在 NPO 基础上再降约 3 dB |
| 工程形态 | 光引擎可独立制造、测试、替换 | 集成度高但可维护性差 |
| 延迟 | 十纳秒量级 | — |
| 成本 | 较 CPO 低 40% 以上 | — |

- **不追求极限堆叠**：算力随面积增长，而 I/O 带宽与功耗仅随周长增长，失衡随堆叠加剧；不追逐 megawatt rack，采取“**站得稀疏、算得紧密**”策略，用光互连将稀疏部署的单元绑成一台机器
- **每代只打少数硬仗**：同时挑战约二十个成功率各 90% 的物理极限，联合成功率趋近于零；每代精选约五个可控风险交付

---

**工程落地三要务**

- **交换芯片高 radix**：交点出口越多，跨军层的手交越少；同等规模下中间层级更少，时延与功耗节省真实可量化
- **计算芯片同样高 radix**（常被忽视的要点）：
  - 提升总出口带宽
  - 提供路径冗余——万卡级集群中链路与模块失效是常态而非意外
  - 最关键收益：**单跳覆盖 = 交换端口数 × 芯片端口数**，两侧皆为乘数；单跳范围越大，最内层、最高频的 Nested BSP 层越能落在其中
  - 两侧 radix 乘积决定每一嵌套硬件层的并行规模
- **高密度 NPO 贴至芯片边缘**：将电-光转换边界从传统的约 **1 m** 内移两个数量级至封装边缘 **10 mm**，残余铜段仅剩数厘米，外向各层随之全部解放

![](images/x3.png) *Figure 3:Design freedom from NPO: package at ten millimeters, board at ten centimeters, rack at one meter, SuperNode at ten meters, data hall at a hundred meters, and data center at a kilometer. Moving the electro-optical boundary inward from 1 m to 10 mm relaxes every outer layer—physically sparse, logically tight.*

各物理层尺度（由内向外跨越五个数量级）：

| 层级 | 尺度 |
|---|---|
| Package | 10 mm |
| Board | 10 cm |
| Rack | 1 m |
| SuperNode | 10 m |
| Data hall | 100 m |
| Data center | 1 km |

边界内移的直接效果：board 不再对抗数十条铜缆的插损，rack 无需满装或堆至兆瓦级；每外扩一层尺度可放大十倍，组件可更稀疏排布，而全系统在逻辑上始终保持一台计算机——散热、供电、可靠性三大障碍无需同时对抗物理极限。

---

**τ Scaling law 在架构中的位置**

- 在系统层级，τ 定律映射到软件与硬件两个嵌套体
- 每层的共同剧本：将串行工作摊到该层可同时工作的多个单元上，该层的**时间常数被折叠得更短**
- 单层折叠有限，**六层折叠相乘**，整场“战役”的总时间才真正降下来

---

**量化指标**

| 指标 | 数值 |
|---|---|
| SuperNode 节点规模 | 8000+ |
| 聚合内存带宽 | 6.7 PB/s |
| 全互连带宽 | 400 Tbps |
| 全军 Barrier 时延 | < 10 μs |
| 单芯片 I/O 带宽 | 7.2 Tbps 量级 |
| 单次通信往返（内存语义） | ~100 ns |
| 在建单系统规模 | 256K 节点级 |

---

**架构哲学**

- 延续 von Neumann 原理不变，改变的是计算机的**规模、嵌套层数与单条指令的可达范围**；远端内存须如同本地内存般自然
- Turing 机是单带串行的计算图景；百万处理器的战役需要同样四步、层层嵌套的对等 BSP 结构
- 软件巢（Nested BSP）与硬件巢互为镜像，Unified Bus 是二者的对齐机制，τ Scaling law 是时间维度上的支配律
- Unified Bus 协议规范开放发布，标准设计大于任何单一公司，接入者获得相同的时延与相同的线性度

### 1. Nested BSP（递归嵌套的批量同步并行模型）

**核心定位**

- **Nested BSP**（递归嵌套批量同步并行模型）是本文提出的核心理论扩展：将 Valiant 与 McColl 的经典 **BSP**（Bulk Synchronous Parallel）模型**递归嵌套**，使并行结构在软件侧呈现“套娃”式层层自相似。
- 论文原文的一句话定义：**Nested BSP extends classic BSP layer within layer under peer equality**——层内套层地扩展经典 BSP，且施加一条硬约束：**每一层的单元都是对等的**。
- 它在整体架构中扮演**“作战计划 + 命令链”**的角色：软件侧的 Nested BSP 与硬件侧的 **Nested Parallel von Neumann Architecture** 逐层对齐，二者由 **Unified Bus** 端到端粘合，最终回答论文开篇之问——百万处理器如何仍是一台计算机。

---

**一、原理溯源：经典 BSP 的四步循环**

- 经典 BSP（Leslie Valiant、Bill McColl）将一次大规模计算切割为若干 **superstep（阶段）**，每个阶段严格执行四步：
  - ① **并行推进**：各单元独立计算，互不等待；
  - ② **Barrier（全军队屏障对齐）**：所有单元到达标记线后统一同步；
  - ③ **交换与归约**：该交换的交换、该聚合的聚合；
  - ④ **进入下一阶段**：交接干净后，全体再次并行推进。
- 经典 BSP 的成本模型可表述为每个 superstep 的耗时：`T = max(w_i) + g·h + l`，其中 `w_i` 为本地计算量、`h` 为最大通信量、`g` 为单位通信代价、`l` 为 barrier 代价。
- 经典模型的局限：**单一全局 barrier** 在处理器数量达到十万、百万量级时，`l` 项本身成为不可逾越的瓶颈——这正是 Nested BSP 要解决的问题。

---

**二、递归结构定义与算法流程**

- 核心思想一句话：**最外层是全军的 BSP，而每个“并行推进”的单元自身又是一支更小的军队，在其作用域内重演同样四步；层层向内，递归到底。**
- 递归算法骨架：

```
NestedBSP(layer ℓ, task P):
        return partial_result
        NestedBSP(ℓ+1, P_i)
```

- **关键结构特征——Multiplicity（多重性）**：每层大套娃内部不是一个小套娃，而是**多个**：
  - 一个 **DP** 持有多个 **PP**；
  - 一个 **PP** 持有多个 **EP**；
  - 一个 **EP** 持有多个 **FSDP** 分片。
- 论文明确指出：**这种多重性正是并行度的来源**。若每层只含单一子单元，递归退化为串行分解，不产生任何并行增益。
- 递归终止条件：抵达最内层（张量并行 TP，切分到单算子权重矩阵），并行粒度收敛到算子内部的 partial sum 归约。

![](images/x1.png) *Figure 1:Software nesting dolls: Nested BSP. From outermost to innermost, the levels are data parallel (DP), pipeline parallel (PP), expert parallel (EP), fully sharded data parallel (FSDP), context parallel (CP), and tensor parallel (TP).*

---

**三、六层软件嵌套的逐层语义**

- 论文给出的软件嵌套顺序（外→内）：**DP → PP → EP → FSDP → CP → TP**，每一层都运行同一套“**compute in parallel, then barrier and reduce**”——即 **map 和 reduce** 的 playbook。
- 各层切分对象、归约原语与同步频率存在严格的梯度关系：

| 层级 | 全称 | 切分对象 | 典型交换/归约原语 | 同步频率 | 典型匹配硬件尺度 |
|---|---|---|---|---|---|
| **DP** | Data Parallel | 全局 batch（按样本） | 梯度 **AllReduce** | 每 step 一次 | 数据中心 / 自治域 |
| **PP** | Pipeline Parallel | 模型层段（按深度） | 微批次 P2P 传递 + 调度 barrier | 每微批次 | 数据厅 / SuperNode |
| **EP** | Expert Parallel | MoE 专家分布 | token **All-to-All** 分发/回收 | 每 MoE 层两次 | SuperNode / 机柜 |
| **FSDP** | Fully Sharded Data Parallel | 参数/优化器状态/梯度分片 | **AllGather** + **ReduceScatter** | 每层参数使用前后 | 机柜 / 板卡 |
| **CP** | Context Parallel | 序列维度 | 跨段 KV/激活交换 + 归约 | 每 Attention 层 | 板卡 / 封装 |
| **TP** | Tensor Parallel | 单算子权重矩阵 | partial sum 归约 | 每算子（最高频） | 封装内 / 片间 |

- 隐藏的设计规律：**越靠内的层，barrier 频率越高、对时延越敏感；越靠外的层，频率越低、可容忍时延越长**。这一梯度与物理世界的分形规律（越远尺度越大、距离越长、带宽越低、时延越高）天然吻合——这是软件嵌套能与硬件嵌套逐层对齐的物理前提。

---

**四、硬约束：Peer Equality（对等性）**

- Nested BSP 超越经典 BSP 的**唯一硬约束**：**每一层的单元都是对等的——不存在必须独占每次 barrier 的 master，也不存在只能服从的 slave。**
- 论文的凝练表述：**Parallelism is nested, authority is not（并行是嵌套的，权力不是）。**
- 反面推演（若无对等性）：
  - 传统计算机的默认模式即 master–slave：CPU 为主、加速器为从；host 为主、device 为从；控制面独占发起权，其余单元被动等待；
  - 军队越大，中心越堵；**加士兵不增加战力——一加一小于二**；
  - 嵌套作战计划会**坍缩回 master 处的一条排队队列**，递归结构名存实亡。
- 正面收益：每一层 Nested BSP 的 barrier 与归约**无需向中心上报**，Nested Parallel von Neumann Architecture 得以**无单点咽喉地扩展**。

---

**五、软硬件双嵌套的逐层对齐**

- Nested BSP（软件嵌套）必须与硬件嵌套体**逐层咬合**，硬件层级由内向外为：**CORE → chiplet → chip package → board → rack → SuperNode → data hall → autonomous zone → data center → inter data center**。

![](images/x2.png) *Figure 2:Hardware nesting dolls: the body of the machine, from chip package to autonomous zone. One autonomous zone holds many data halls, one hall holds many SuperNodes, and one SuperNode holds many racks, all the way down to the package.*

- 对齐失效的后果（论文警告）：任何一层错位——无论在协议中、软件中、还是机柜间网络中——**队伍当场散架**；此时无论放置多少并行单元，都无法兑现算力。
- 论文研究范围聚焦中间地带：**从 chip package 到填满一个数据中心的 cluster**——更内是“如何训练单个士兵”，更外是“campus 之间如何对话”。
- 硬件侧必须提供的四项属性，全部服务于让嵌套结构在物理世界立得住：
  - **One protocol end to end**：Unified Bus 从 package 到 autonomous zone 单一协议，无转换税（大集群中 **>80% 能耗花在搬数据**，其中相当比例损耗在协议边界）；
  - **Memory semantics**：原生 load/store，一致性由硬件保证，单次通信往返从数十微秒降至 **~100 ns**（约 **500×** 提升）；
  - **Full peer equality**：CPU、NPU、内存、存储、NIC 同总线对等，任何人可发起、任何人可响应；
  - **Copper near, optics far**：铜护时延、光护规模，介质切换时协议不变。

---

**六、执行样例：一次 LLM 训练 step 的嵌套展开**

- 以一个完整训练步的调用栈为例，展示递归的输入输出传递：
  - **DP 层（外）**：全局 batch 按样本切分为 k 份 → 各副本递归进入；
  - **PP 层**：每副本的模型按深度切为多个 stage，微批次在 stage 间流水，stage 边界做调度级 barrier；
  - **EP 层**：每个 MoE 层触发两次 token **All-to-All**（分发/回收），专家路由归约后对齐；
  - **FSDP 层**：参数分片在前向/反向前 **AllGather**，使用后释放，梯度以 **ReduceScatter** 聚回分片；
  - **CP 层**：长序列切分为段，Attention 计算中跨段交换 KV/中间激活并归约；
  - **TP 层（内）**：单算子权重按行/列切分，矩阵乘后立即 partial sum 归约——**频率最高、路径最短**；
  - **回卷阶段**：梯度沿层级向上逐层归约（每层自带 barrier），最终在 DP 层完成梯度 **AllReduce**，全军队对齐后进入下一 step。
- 逐层输入输出关系：
  - **层输入**：上一阶段对齐后的全局状态 + 本层切分出的子任务集合；
  - **层输出**：各子单元的 partial result → 经 **barrier** 确认全员齐备 → 经 **exchange/aggregate** 变换为对下一层/下一阶段有效的一致状态。
- 系统级输入输出：
  - **系统输入**：超大规模训练/推理任务 + 十万至百万级处理器池；
  - **系统输出**：以“一台计算机”语义完成的同步训练步；对外表现为**加芯片即加算力、芯片算力不再消耗于搬数据、token 成本下降**。

---

**七、关键参数与 τ Scaling Law 的时间折叠**

- **τ Scaling law** 与双嵌套的映射关系：在每一层 Nested BSP 层级和每一物理尺度上，把“本应一件接一件串行完成的工作摊到该层可同时工作的多个单元上”，该层的**时间常数折叠变短**。
- 折叠的乘性本质：
  - **One layer alone folds only so much, six layers fold multiplicatively**——单层折叠有限，六层**乘性折叠**，整个 campaign 的时间才真正降下来；
  - 形式化表述：`τ_total ≈ Π(ℓ=1..6) τ_ℓ / fold_ℓ`，即各层折叠因子连乘。
- 支撑该折叠的实践参数：

| 指标 | 数值 | 对 Nested BSP 的意义 |
|---|---|---|
| SuperNode 节点规模 | **> 8000 节点** | 内层高频 barrier 的“一台计算机”边界 |
| 聚合内存带宽 | **6.7 PB/s** | 每层归约的数据通路保障 |
| 全互联带宽 | **400 Tbps** | exchange 阶段无瓶颈 |
| 全军 barrier 时延 | **< 10 μs** | 外层 barrier 的可实现性证明 |
| 单芯片 I/O 带宽 | **7.2 Tbps 级** | 芯片自身作为 junction 的前提 |
| 内存语义往返时延 | **~100 ns**（较 TCP/IP 数十 μs 提升 ~500×） | 最内层高频同步的前提 |
| 当前部署规模 | **256K-node 级** | Nested BSP 战场计划的实际落地 |
- 一跳覆盖与最内层的对应关系：
  - **One-hop coverage = switch radix × chip radix**，两侧均为乘数；芯片 radix 翻倍，一跳 SuperNode 规模翻倍；
  - 一跳覆盖越大，barrier 与归约路径越短——**恰好最内、最高频的 Nested BSP 层级（TP/CP）落在这个覆盖范围内**；
  - 两 radix 之积决定每个嵌套硬件层可设置的并行度上限。

---

**八、与经典 BSP 的系统性对比**

| 维度 | 经典 BSP | Nested BSP |
|---|---|---|
| 同步范围 | 单一全局 barrier | 每层局部 barrier，层层嵌套分治 |
| 层级数 | 1 层 | 递归任意层（论文落地为 6 层软件 + 多层硬件） |
| 节点关系 | 未强制约束 | **Peer Equality**：无 master/slave，任何单元可发起/响应 |
| 通信模型 | superstep 末集中通信 | 每层 superstep 末 exchange + aggregate，递归复合 |
| 拓扑关系 | 均一处理器集合，无物理对应 | 与物理嵌套层级**逐层对齐**（分形：越远带宽越低时延越高） |
| 扩展瓶颈 | 全局 barrier 与中心控制 | barrier 按层分治；无单点咽喉 |
| 成本模型 | `T = max(w_i) + g·h + l` | 递归复合 + **τ Scaling law** 六层乘性折叠 |
| 理论渊源 | Valiant/McColl 单层并行模型 | 延续 von Neumann“一台计算机”定义的层次化扩展 |

---

**九、在整体体系中的作用与结论**

- 三位一体的架构关系：
  - **软件嵌套 = Nested BSP**：作战计划与命令链；
  - **硬件嵌套 = Nested Parallel von Neumann Architecture**：军队的身体，每层容纳更多对等并发单元；
  - **Unified Bus = 落地的互连**：让每一层的 split 与 join 对齐，**命令不在任何一层边界断裂**。
- 理论定位：Turing machine 是单带、串行的计算图景；百万处理器的 campaign 需要同样的四步——**并行推进、barrier、交换与归约、下一阶段**——层内套层地执行。Nested BSP 即此图景的形式化。
- 最终效果声明：**Many processors, still one computer**——SuperNode 以一套 Nested BSP 战斗计划和一套 Nested BSP / Unified Bus 命令集贯穿到底，256K 节点级系统是**一台机器**，而非二十几万台机器粘成的网络。

### 2. Nested Parallel von Neumann Architecture（嵌套并行冯·诺依曼架构）

**核心定位：架构要解决的问题**

- 传统 **von Neumann 架构** 回答的是“如何造一台更强的计算机”，其八十年演进路径是不断提升单处理器性能
- 当处理器规模达到 **十万级、百万级** 时，设计问题发生质变：不再是“谁的拳头更硬”，而是“百万大军能否保持队形、仍是一台计算机”
- **Nested Parallel von Neumann Architecture** 的核心命题：将 von Neumann 对“一台计算机”的定义，扩展为“一个处理器军团如何仍然是一台计算机”

---

**总体设计：双巢咬合结构**

该架构由两个相互对齐的嵌套体系构成，二者**逐层对应**，缺一不可：

- **软件巢**：即 **Nested BSP**，负责计算任务的递归分解
- **硬件巢**：即 **Nested Parallel von Neumann Architecture** 本体，负责提供逐层的物理承载
- 对齐失败后果：任何一层软件与硬件不匹配，“步子就散了”——散在协议里、软件里、机架间网络里，无论放置多少并行单元都无法兑现算力
- 递归分形规律：**越外层，尺度越大、距离越长、带宽越低、延迟越高；越内层，尺度越小、距离越短、带宽越高、延迟越低**——这与物理世界的分层规律一致

![](images/x1.png) *Figure 1:Software nesting dolls: Nested BSP. From outermost to innermost, the levels are data parallel (DP), pipeline parallel (PP), expert parallel (EP), fully sharded data parallel (FSDP), context parallel (CP), and tensor parallel (TP).*

---

**软件巢原理：Nested BSP 算法流程**

**经典 BSP 基础**（Valiant 与 McColl 提出）：

- 每个 **superstep** 包含四个步骤：
  - **并行计算**：各单元独立推进，互不等待
  - **Barrier**：全军在同一战线汇合
  - **交换与聚合**：同步数据、完成 reduction
  - **进入下一阶段**：整体并行推进
- 简言之即 **map and reduce** 模式

**Nested BSP 扩展机制**：

- **递归嵌套**：最外层是全军的 BSP；外层 BSP 中每一个“并行推进”的单元，其内部再运行完整四步循环——层层递归，直至最内层
- **乘法性并行来源**：每个大 doll 内部不是一个小 doll，而是**多个**——一个 DP 容纳多个 PP，一个 PP 容纳多个 EP，一个 EP 容纳多个 FSDP shard——这种**多重性**是并行度的来源
- **硬约束——对等性**：每一层的单元都是 **peers**，不存在必须独占 barrier 的 master，也不存在只能服从的 slave；**并行可以嵌套，权力不能嵌套**。这是 Nested BSP 区别于经典 BSP 的关键扩展

**软件巢六层（由外到内）**：

| 层级 | 并行策略 | 分解对象 |
|---|---|---|
| **DP** (Data Parallel) | 数据并行 | 不同数据批次 |
| **PP** (Pipeline Parallel) | 流水线并行 | 模型层段 |
| **EP** (Expert Parallel) | 专家并行 | MoE 专家分布 |
| **FSDP** (Fully Sharded Data Parallel) | 全分片数据并行 | 参数/优化器状态分片 |
| **CP** (Context Parallel) | 上下文并行 | 序列上下文切分 |
| **TP** (Tensor Parallel) | 张量并行 | 单个张量/矩阵运算切分 |

- 每一层运行同一套 playbook：**并行计算 → barrier → 交换与聚合 → 下一阶段**
- 指令逐层下达，军队才能保持队形

---

**硬件巢原理：物理层级结构**

![](images/x2.png) *Figure 2:Hardware nesting dolls: the body of the machine, from chip package to autonomous zone. One autonomous zone holds many data halls, one hall holds many SuperNodes, and one SuperNode holds many racks, all the way down to the package.*

**硬件层级（由内向外，共十层）**：

- CORE → **chiplet** → **chip package** → **board** → **rack** → **SuperNode** → **data hall** → **autonomous zone** → **data center** → **inter data center**
- 论文聚焦**中段**：**package 到 autonomous zone**——这是“十万或百万变成一支军队”的主战场
  - 更内层（CORE/chiplet）：属于“如何训练一个士兵”
  - 更外层（data center 间）：属于“园区之间如何对话”
- 每一层的关键属性：**更多并发单元、对等地位、共享一条总线**
- 一个 autonomous zone 容纳多个 data hall，一个 hall 容纳多个 SuperNode，一个 SuperNode 容纳多个 rack，逐层向下直至 package

**软硬对齐逻辑**：

- 按**分形带宽/延迟梯度**推断对应关系：最内层的 **TP**（通信最频繁、延迟最敏感）落在 **package/board** 级（纳秒级、最高带宽）；最外层的 **DP** 落在 **data hall/autonomous zone** 级
- 对应关系一旦断裂，barrier 和 aggregation 的路径变长，最内层、最高频的 Nested BSP 层首先受害

---

**τ Scaling Law：时间折叠机制**

- **τ Scaling law** 在系统层面的作用：对两个巢逐层**折叠时间**
- 折叠机制：将原本串行完成的工作，摊到该层的多个并发单元上同时执行，该层的**时间常数被折叠变短**
- 折叠发生位置：**package 内一次、board 上一次、rack 内一次、SuperNode 内一次、data hall 内再一次**
- **乘法效应**：单层折叠幅度有限，**六层折叠是乘法叠加**——这是整体战役时间真正缩短的来源
- 输入输出关系：
  - 输入：按 Nested BSP 组织的并行计算任务
  - 输出：经逐层时间折叠后的聚合结果
  - 作用：τ 是时间折叠定律，Nested Parallel von Neumann Architecture 是承载该定律的扩展对等架构

---

**硬件实现：Unified Bus 六大关键技术**

Unified Bus 是让每一层的“拆分与汇合”对齐、让命令链不因边界而断裂的互连底座。

**第一：端到端单一协议**

- 过去的双协议格局：
  - 机箱内总线：纳秒级、快，但**天生短距，出不了机箱**
  - 机箱外网络：可达远，但**每个边界都是收费站**——拆包、检查、重打包
- 代价：大型集群中 **超过 80% 的能耗用于数据搬运**，其中很大比例损失在协议转换处
- 方案：**Unified Bus 将“总线”与“网络”合并为单一协议**，从 package 到 autonomous zone 端到端**零转换**

**第二：内存语义**

- 演进对比：

| 技术 | 通信模型 | 往返延迟 | 特征 |
|---|---|---|---|
| **TCP/IP** | 消息传递，逐层爬升至应用层再跨协议转换 | **数十微秒**，主要耗时在软件栈 | barrier/reduction 全部卡死于此 |
| **RDMA** | 仍是“发消息、等回复” | 显著改善 | 本质未变，规模越大卡点越严重 |
| **内存语义** | 原生 **load/store**，硬件保证一致性 | **约 100 纳秒** | 提升 **约 500 倍** |

- 配套指标：**单芯片 I/O 带宽达到 7.2 Tbps 量级**

**第三：拆除主从结构——对等性基石**

- 主从模式是传统计算机设计的默认模式：**CPU 为主、加速器为从；host 为主、device 为从；控制面独占发起权**
- 瓶颈机理：军队越大，中心越堵；**加士兵不涨战斗力，一加一小于二**
- 方案：**CPU、NPU、内存、存储、NIC 全部以 peer 身份上同一条总线**——任何节点可发起、任何节点可响应
- 关键收益：每一层 Nested BSP 的 barrier 与聚合**无需向中心回报**；没有对等性，嵌套作战计划会塌缩为 master 处的排队，**嵌套架构必须无单点咽喉才能扩展**

**第四：铜近光远，协议不变**

- 铜的物理墙：单线速率越高，线越粗、距离越短；数千根铜缆捆扎后**粗到无法安装**
- 方案：**铜保近距延迟，光保远距规模，介质切换时协议不变**
- 光侧选型：**NPO (near-package optics)**，而非 CPO

| 指标 | **NPO** | **CPO** |
|---|---|---|
| 损耗削减 | 一步移除 **9–11 dB**（全电通路总损耗 >20 dB） | 在此基础上仅能再移除 **约 3 dB** |
| 光引擎形态 | **可独立构建、测试、更换的单模块** | 与芯片深度耦合 |
| 延迟 | **十纳秒级** | — |
| 成本 | **低于 CPO 40% 以上** | 基准 |
| 生态 | 已成为 **OIF 项目**，数十家伙伴参与 | — |

**第五：不把士兵塞进一个帐篷**

- 旧思路的数学困境：**算力随面积增长，I/O 带宽与功耗只随周长增长**——越塞满，失配越严重
- 散热的具体约束：**MW 级机架**可能需要数百平方米冷却面积；**GW 级机房**仅容纳千级机架却占地一平方公里，水管需延伸一公里
- 经济逻辑：**机房地面便宜，机架里的芯片贵**
- 方案：不追 MW 机架——**物理上稀疏、逻辑上紧耦合**：稀疏站立、紧密计算，用光互连把散开的军队绑成一台机器

**第六：每代只打少数硬仗**

- 概率逻辑：同时挑 **20 个物理极限**、每个成功率 90%，联合成功率趋近于零；只挑 **5 个**，仍可交付
- **取舍本身是设计的一部分**——选择何时攻坚、何时推迟到下一代形态

---

**工程落地三要素**

**要素一：交换芯片必须高 radix**

- 交叉路口出口越多，全军转交次数越少
- 同等规模下，**高 radix → 更少的中间层级 → 延迟与功耗的实际节省**

**要素二：计算芯片也必须高 radix（常被忽视）**

- 旧观念：芯片只管计算，一两条出链足够，其余交给网络——在百万军团中等于“整个营的调度都挤一个门”
- 芯片本身必须成为交叉路口，多出口带来三重收益：
  - **更高的总出口带宽**
  - **更多路径冗余**：万卡集群中链路和模块故障是**常态而非意外**，断一条另有路可走
  - **一跳覆盖范围 = 交换芯片端口数 × 计算芯片端口数**——**双方都是乘数**，计算芯片 radix 翻倍，一跳 SuperNode 规模翻倍
- 战略意义：一跳覆盖范围越大，**barrier 与 aggregation 的路径越短**——恰恰是最内层、最高频的 Nested BSP 层落在这个范围内
- **两层 radix 的乘积决定每一层嵌套硬件的并行度**

**要素三：高密度 NPO 把光推到计算芯片边缘**

- 电光转换位置越靠近芯片边缘，残余铜段越短

![](images/x3.png) *Figure 3:Design freedom from NPO: package at ten millimeters, board at ten centimeters, rack at one meter, SuperNode at ten meters, data hall at a hundred meters, and data center at a kilometer. Moving the electro-optical boundary inward from 1 m to 10 mm relaxes every outer layer—physically sparse, logically tight.*

**空间尺度阶梯（五个数量级跨度）**：

| 层级 | 典型尺度 |
|---|---|
| Package | **10 mm** |
| Board | **10 cm** |
| Rack | **1 m** |
| SuperNode | **10 m** |
| Data hall | **100 m** |
| Data center | **1 km** |

- 历史症结：电光边界卡在 **1 米**（出机箱才有光），**package → board → rack 整段由铜独扛**：board 对抗 insertion loss，rack 对抗密度，每一步都撞物理极限
- NPO 的动作：**将边界向内推进两个数量级，从 1 m 移到 package 边缘的 10 mm**
- 连锁放松效应：
  - 铜只走最后几厘米，其余全部交给光
  - Board 不再为几十条铜缆对抗 insertion loss
  - Rack 无需塞满、无需堆到 MW 级
  - **每向外一层可扩大十倍尺度**，组件可更稀疏摆放
  - 光不在乎距离，机架直接拉开即可
- **物理稀疏、逻辑紧耦合**的落地逻辑：逻辑紧 = 一跳可达 + 内存语义 + SuperNode 仍是一台计算机；物理稀疏 = 芯片不必挤进一个铁盒子
- 关键收益：**冷却、供电、可靠性三道门槛不再需要同时对抗物理极限**——不必挑战空间密度极限

---

**量化规格与部署现状**

| 指标 | 数值 |
|---|---|
| SuperNode 节点规模 | **超过 8000 节点** |
| 聚合内存带宽 | **6.7 PB/s** |
| 全互连带宽 | **400 Tbps** |
| 全军 barrier 延迟 | **< 10 μs** |
| 单次通信往返延迟 | TCP/IP 数十 μs → **约 100 ns**（**约 500 倍**提升） |
| 单芯片 I/O 带宽 | **7.2 Tbps 量级** |
| 部署规模 | **256K 节点级** Super AI 计算机系统正在部署 |

---

**输入输出关系与整体作用**

**输入侧**：

- 按 **Nested BSP** 组织的并行计算任务：从最外层 DP 的批次划分，到最内层 TP 的张量切分，每一层携带自己的并行单元数、barrier 点与聚合需求
- 硬件输入：逐层递归的物理结构——package 内的 die、board 上的 package、rack 内的 board、SuperNode 内的 rack

**处理机制**：

- **Unified Bus** 作为唯一命令链：单一协议贯穿 package 到 autonomous zone，内存语义端到端，全节点 peer 对等
- 每层执行同一四步循环，时间常数被逐层折叠，六层乘法叠加

**输出侧**：

- 对用户：**加芯片即加算力**，芯片算力不再消耗在搬数据上，**单 token 成本下降**
- 对系统：仍然是一台计算机——不是二十万台机器的网络拼接，而是**一个 Nested BSP 作战计划 + 一套 Unified Bus 命令集贯穿到底**的单一机器

**在整个体系中的作用**：

- **Turing 机**是单带串行的计算图景；**经典 BSP** 把大规模并行切成相位；**Nested BSP** 把相位循环递归嵌套到每一层并在对等约束下运行；**Nested Parallel von Neumann Architecture** 则为这套递归作战计划提供物理躯体
- **Unified Bus Protocol 规范已开放**，标准设计目标大于任何单一公司：任何接入者获得相同的延迟与相同的线性度
- 一句话总结架构本质：**延伸 BSP 至 Nested BSP，延伸 von Neumann 单机架构至嵌套并行架构，由 Unified Bus 端到端连接——多处理器仍是一台计算机，且每个节点都是 peer**

### 3. Unified Bus：端到端统一协议与内存语义互连

**核心定位**

- Unified Bus 是 Nested Parallel von Neumann Architecture 的**物理承载层与命令链**，定位为从 **chip package 到 autonomous zone 的端到端单一协议互连**，同时消灭协议层割裂与消息语义开销。
- 论文原句点明设计意图："Unified Bus merges 'bus' and 'unified' into a single protocol from the package all the way to the autonomous zone, end to end, with no transfers."
- 在“百万大军”隐喻中，它的作用是保证命令逐层下达、**无一层断裂**（command does not break at the boundaries）。

---

**问题定义：从 Cliff 到 Slope**

- 被接受的物理规律：越远距离越低带宽、越高时延——这是 **slope**（斜坡），设计目标是让它平缓可爬。
- 历史失败点：跨出 chassis 一步，带宽骤降一个数量级、时延数倍上升——这是 **cliff**（断崖）。rack 内像一台机器，rack 外各自为战。
- 断崖的两个根因：
  - **双协议割裂**：机内 bus 快但出不了机箱；机间 network 能达远但每个边界都是 toll booth（unpack – inspect – re-pack）。
  - **消息语义拖累**：通信必须穿越软件栈，barrier 与 reduction 全部卡在此处，**规模越大卡得越死**。
- 能耗佐证：大型集群中**超过 80% 的能耗用于数据搬运**，其中相当比例损耗在协议转换处。

---

**统一协议的实现原理**

- **协议覆盖范围**：package → board → rack → SuperNode → data hall → autonomous zone，全程一种语言、**零协议转换**。
- **介质自适应而语义不变**：copper 承载近距以保时延，optics 承载远距以保规模，介质切换时**协议不换**。
- 该设计与硬件 nest 的 fractal 逻辑严格对齐：越内层距离越短、带宽越高、时延越低，但六层尺度的命令语言完全一致。

![](images/x2.png) *Figure 2:Hardware nesting dolls: the body of the machine, from chip package to autonomous zone. One autonomous zone holds many data halls, one hall holds many SuperNodes, and one SuperNode holds many racks, all the way down to the package.*

---

**内存语义：实现原理与时延分解**

这是 Unified Bus 最核心的一处技术跃迁，需从语义模型对比来理解。

- 旧路径（**message semantics**）的数据流：
  - 数据需逐层上爬至应用层，再跨协议转换，路径长且经过软件。
  - TCP/IP 一个 round trip 代价 **tens of microseconds**，**大头耗在软件栈**——协议栈处理、中断、上下文切换、多次内存拷贝。
  - RDMA 砍掉了拷贝与内核绕行，但本质仍是 **“send a message, wait for a reply”**：发起方必须显式构造消息、经 queue pair 提交、等待完成事件——**同步原语仍活在软件里**。
- 新路径（**memory semantics**）的机制：
  - 远端内存以 **native load/store** 直接访问，指令形式与本地内存访问无异——即论文的架构信条 **“distant memory must feel as natural as memory at hand”**，这是 von Neumann 单一地址空间模型向“大军规模”的直接延伸。
  - **consistency 由硬件全托管**：一致性维护（可理解为 cache coherence 机制向集群尺度的外推）下沉到总线硬件，软件彻底退出关键路径。
  - 编址视角：SuperNode 内形成平坦可寻址空间，远端地址与本地地址统一编址，barrier 之后所有 peer 看到一致状态。
- 量化收益与归因：
  - 单次通信 round trip 从 tens of microseconds 降至**约 100 纳秒量级，约 500 倍改善**。
  - 数量级跨越的来源：μs 级的软件栈耗时被整体移除，剩余时延由物理传播与硬件协议处理（ns 级）主导。
  - 单芯片 I/O 带宽达 **7.2 Tbps 级**，为内存语义提供充足的物理通道容量。

---

**对等架构：Peer Equality 的机制意义**

- 传统默认模式：**CPU 为 master、加速器为 slave；host 为 master、device 为 slave；控制面独占 initiation**——这不是某产品线的偶然，而是传统计算机设计的默认范式。
- 规模化恶果：中心拥塞，**“adding soldiers does not add strength——one plus one is less than two”**。
- Unified Bus 的做法：**CPU、NPU、memory、storage、NIC 在同一总线上完全对等，anyone can initiate，anyone can respond**。
- 对 Nested BSP 的支撑关系：
  - Nested BSP 的硬约束是 **“Parallelism is nested, authority is not”**——每层单元必须是 peers，无 master 独占 barrier、无 slave 只能听命。
  - 由此，每层的 barrier 与 aggregation **无需向中心节点回报**即可在对等单元间完成；若无此对等性，“nested battle plan collapses back into a queue at the master”，嵌套结构退化为主节点前的排队。

---

**算法流程：Unified Bus 如何服务 Nested BSP 四步循环**

Nested BSP 每一层执行同一 playbook：**parallel work → barrier → exchange & aggregate → next phase**（即 map–reduce）。Unified Bus 在各阶段的角色：

- **parallel work 阶段**：各 peer 单元独立计算，互连仅承载各自局部流量，互不等待。
- **barrier 阶段**：硬件级同步原语完成全军会合，SuperNode 规模下**full-army barrier 低于 10 微秒**。
- **exchange & aggregate 阶段**：以 load/store 直接读写远端 buffer，reduction 在对等单元间直连完成，不经中心、不经软件。
- **next phase 阶段**：handoff 干净后齐步进入下一相，循环嵌套递归至更内层。

![](images/x1.png) *Figure 1:Software nesting dolls: Nested BSP. From outermost to innermost, the levels are data parallel (DP), pipeline parallel (PP), expert parallel (EP), fully sharded data parallel (FSDP), context parallel (CP), and tensor parallel (TP).*

软件 nest 的六层——**DP、PP、EP、FSDP、CP、TP**——中，**TP 是最内层、执行频率最高的 Nested BSP 层**，其 barrier 与 gradient aggregation 流量恰好落在 one-hop 覆盖范围内，这正是内存语义与时延收益兑现最密集的位置。

---

**物理层：Copper Near、Optics Far 与 NPO**

- **copper 的物理墙**：单线速率推高则线更粗、reach 更短；数千根铜缆集束后厚到无法安装。
- **损耗账本**：全电通道 insertion loss **超过 20 dB**；**NPO（near-package optics）一步消除 9–11 dB**；**CPO 相比 NPO 仅能再多约 3 dB**，却牺牲工程可行性。
- **NPO 的工程收益**：
  - optical engine 作为**单一模块**可独立制造、测试、更换。
  - 时延**十纳秒级**。
  - 整体成本比 CPO **低 40% 以上**。
  - 已立项为 **OIF 项目**，数十家伙伴共同推进——标准意图“大于任何单一公司”。
- **电光边界内移**：从传统 **1 m 内移至 10 mm 的 package 边缘**（两个数量级），copper 只走最后几厘米，其余全部交给光。
- **尺度链**（见下图）：package 10 mm → board 10 cm → rack 1 m → SuperNode 10 m → data hall 100 m → data center 1 km，跨越五六个数量级，**逻辑上始终是一台计算机**——**physically sparse, logically tight（站得稀疏，算得紧密）**。

![](images/x3.png) *Figure 3:Design freedom from NPO: package at ten millimeters, board at ten centimeters, rack at one meter, SuperNode at ten meters, data hall at a hundred meters, and data center at a kilometer. Moving the electro-optical boundary inward from 1 m to 10 mm relaxes every outer layer—physically sparse, logically tight.*

---

**关键性能指标汇总**

| 指标 | 数值 | 对比基准 |
| --- | --- | --- |
| 单次通信 round trip | **~100 ns** | 消息语义的 tens of μs，约 **500×** |
| 单芯片 I/O 带宽 | **7.2 Tbps 级** | — |
| SuperNode 节点规模 | **8000+ 节点** | — |
| 聚合内存带宽 | **6.7 PB/s** | — |
| 全互连带宽 | **400 Tbps** | — |
| full-army barrier | **<10 μs** | — |
| NPO 光引擎时延 | **十纳秒级** | 成本比 CPO 低 **40%+** |
| 电光边界位置 | **10 mm（package 边缘）** | 旧边界 1 m，内移两个数量级 |
| 部署规模 | **256K 节点级** Super AI computer | 一台机器，而非二十万台机器的拼接 |

**与既有方案横向对比**

| 维度 | TCP/IP | RDMA | Unified Bus |
| --- | --- | --- | --- |
| 通信语义 | message | message（send–wait） | **memory（native load/store）** |
| 一致性处理 | 软件栈 | 部分卸载 | **硬件全托管** |
| 协议边界 | 每层转换 | 减少但仍跨 PCIe/以太网 | **package 到 autonomous zone 零转换** |
| round trip | tens of μs | μs 级 | **~100 ns** |
| 发起权 | host 中心 | 受限于 queue pair 配置 | **anyone can initiate** |
| 角色关系 | master–slave | host 中心残留 | **full peer equality** |

---

**输入输出关系与在整体架构中的作用**

- **输入侧**：
  - 来自 Nested BSP 各层的**同步与聚合需求**：barrier、allreduce 类归约、参数/激活/梯度交换。
  - 来自计算芯片的 native load/store 请求流，以硬件一致性协议为约束。
- **输出侧**：
  - barrier 后的**数据可见性保证**——所有 peer 见到一致状态。
  - 归约完成信号，触发 next phase 启动，驱动四步循环滚动。
- **在整体中的四重作用**：
  - **两个 nest 的对齐机构**：软件 nest（Nested BSP）与硬件 nest 逐层对齐、无隙咬合的保障；任一层错位则“step scatters”，堆再多并行单元也无效。
  - **τ Scaling law 的物理载体**：每层时间常数的折叠依赖该层 barrier/exchange 不成为瓶颈；**单层折叠有限，六层乘性折叠**才使整场战役的总时间真正下降——Unified Bus 让每一次折叠都不被通信卡断。
  - **成本结构重塑**：芯片算力不再消耗于搬数据，**cost per token 下降**，加芯片即加算力。
  - **开放生态锚点**：Unified Bus Protocol 规范已开放，任何接入者获得**相同的时延与相同的线性度**。

---

**Radix 设计与 One-hop 覆盖的乘法逻辑**

- 核心公式：**one-hop 覆盖 = switch ports × chip ports**，两侧均为乘数。
- **计算芯片高 radix 的三重收益**：
  - 更高的总出口带宽。
  - 更多路径冗余：万卡级集群中**链路与 module 失效是常态而非意外**，断一条仍有多条可走。
  - one-hop 覆盖翻倍——**最内层、最高频的 Nested BSP 层（TP 及其 barrier/aggregation）恰好落入该覆盖**，同步路径最短。
- 设计推论：**两级 radix 的乘积决定每一层嵌套硬件的并行度上限**；传统“芯片只管算、一两条出链即可”的旧观念，在百万大军中等同于让整个营的调度挤过一扇门。
- 硬件落点：高 radix 交换芯片 + 高 radix 计算芯片 + **高密度 NPO 将光推到芯片边缘**，三者共同支撑 NPO 带来的尺度松弛——冷却、供电、可靠性三道难关无需同时对抗物理极限。

---

**工程节奏：每代只打少数硬仗**

- 风险算术：同时挑战 20 个物理极限、各自 90% 成功率，**联合成功率趋近于零**；每代收敛到 **5 个**以内关键战役，仍可按期交付。
- 选什么、缓什么本身就是设计的一部分——这解释了 Unified Bus 路线图的迭代逻辑：先立**统一协议 + 内存语义 + 对等架构**为不变量，再逐代推进 NPO 边界内移、radice 提升、one-hop 覆盖扩张。
- 终局形态：**many processors, still one computer**——从 package 到 autonomous zone，一套 Nested BSP 战役计划、一套 Unified Bus 指令集贯彻到底，每个节点都是总线上的 peer。

### 4. NPO 近封装光学：电光边界内移与"物理稀疏、逻辑紧致"

**核心命题：一次边界内移，六层全链松绑**

NPO的全部技术动作可以压缩为一句话：把**电光转换边界**从 **~1 m**（机框出口）内移两个数量级至 **10 mm**（package 边缘）。铜只承担最后几厘米，其余距离全部交给光。这一个动作同时是Unified Bus“**铜近光远、语言不变**”策略的物理落点，也是论文口号 **physically sparse, logically tight**（站得稀、算得紧）得以成立的支点。

---

**物理背景：铜的三堵墙**

- **损耗墙**（速率-距离-线径的三角约束）
  - 单lane速率提升 → 趋肤效应与介质损耗加剧 → 插入损耗上升
  - 补偿手段只有加粗导体或加重均衡 → 线变粗、可达距离变短
  - 论文量化锚点：全电通路（package→board→rack，约1 m跨度）损耗 **>20 dB**
- **密度墙**
  - 捆束数千根铜缆后“厚到无法安装”
  - board层对抗几十条铜缆的插入损耗，rack层对抗布线密度
- **几何失配墙**
  - 算力随**面积**增长，I/O带宽与供电只随**周长**增长——封装越密、机架越满，失配越严重
  - **单芯片 7.2 Tbps 级 I/O** 在1 m铜尺度上无法物理兑现
- **光的解法**
  - 光纤衰减在百米至公里尺度几乎可忽略，带宽-距离积远超铜
  - 光不关心距离 → 外层物理尺度可以自由拉开，且不违反嵌套系统的分形物理（越远带宽越低、延迟越高——这是被接受的坡道，不是要消灭的断崖）

---

**路线抉择：损耗-模块化的权衡算术**

设计决策流程为一条清晰的排除链：

- **可插拔光模块**（边界 ≈ 1 m）：package、board、rack三层全靠铜 → 三堵墙同时压在最贵、最密的内圈
- **CPO**（边界进入封装，die旁毫米级）：相对NPO只能再多移除 **~3 dB**，代价是光引擎与计算die强耦合——不可独立构建、测试、更换，良率与散热互相拖累
- **NPO**（边界 10 mm，package边缘）：一步移除 **9–11 dB**，光引擎是独立模块，延迟**十纳秒级**，综合成本比CPO**低40%以上** → 当前代的正确选择

| 维度 | 可插拔（边界 ≈ 1 m） | **NPO（边界 ≈ 10 mm）** | CPO（边界进封装） |
|---|---|---|---|
| 铜承担段落 | package→board→rack全程 | **仅最后厘米级** | 仅die旁毫米级 |
| 移除的电段损耗 | 基线（全电通路 **>20 dB**） | **一步移除 9–11 dB** | 在NPO之上仅多 **~3 dB** |
| 光引擎形态 | 独立模块 | **独立模块：可构建/可测试/可更换** | 与计算die封装耦合 |
| 延迟 | 受制于前置1 m铜段 | **十纳秒级** | 十纳秒级（略低，同量级） |
| 综合成本 | — | **比CPO低 >40%** | 最高 |
| 生态 | 成熟 | **OIF项目，数十家伙伴** | 早期 |

损耗预算算术的结论：**边际收益递减**（20 → 移除11 → 再+3 dB），而模块化代价陡增——这正是论文第六条“每代只打几场硬仗”在光互连选型上的兑现：CPO是被主动defer的后续form。

---

**实现机制：信号通路与参数设置**

- **输入**：计算芯片SerDes输出的高速差分电信号
- **通路流程**（五步）：
  1. die → package边缘：~10 mm铜走线，损耗受控的最后一段电段
  2. 近封装光引擎完成**电光转换**
  3. 光纤承载，依次跨越 board（10 cm）→ rack（1 m）→ SuperNode（10 m）→ data hall（100 m）→ data center（1 km），五个数量级距离由同一介质覆盖
  4. 目的端光引擎完成**光电转换**，还原电信号
  5. 进入对端package，交由Unified Bus语义层处理
- **输出**：对端芯片可直接消费的电信号——**load/store内存语义与协议报文在介质切换前后完全不变**，软件零感知
- **关键参数设置**：
  - 边界位置：**10 mm**（内移两个数量级）
  - 引擎延迟：**十纳秒级**——嵌在 ~100 ns 通信往返预算内
  - 支撑的I/O密度：**单芯片 7.2 Tbps级**

---

**尺度阶梯：六层逐级释放**

![](images/x3.png) *Figure 3:Design freedom from NPO: package at ten millimeters, board at ten centimeters, rack at one meter, SuperNode at ten meters, data hall at a hundred meters, and data center at a kilometer. Moving the electro-optical boundary inward from 1 m to 10 mm relaxes every outer layer—physically sparse, logically tight.*

| 嵌套层 | 尺度 | NPO前状态 | NPO后状态 |
|---|---|---|---|
| Package | **10 mm** | 电段起点 | **电光转换发生地**，铜只剩最后几厘米 |
| Board | **10 cm** | 几十条铜缆对抗插入损耗 | 光纤直连，板级布线压力消失 |
| Rack | **1 m** | 旧电光边界所在，对抗密度 | 机架可拉开，无需堆满、无需冲兆瓦 |
| SuperNode | **10 m** | 铜不可达 | 光互连，一跳覆盖的目标尺度 |
| Data hall | **100 m** | — | 组件稀疏摆放，冷却/供电解耦 |
| Data center | **1 km** | — | 逻辑上仍是一台计算机 |

- **关键环在内圈**：过去边界停在~1 m，意味着package、board、rack三层全部由铜硬扛
- 边界内移后：**每层向外可放宽十倍尺度**，组件摆放更稀疏；全程扩张五个数量级（10 mm → 1 km），逻辑上仍是**一台计算机**

![](images/x2.png) *Figure 2:Hardware nesting dolls: the body of the machine, from chip package to autonomous zone. One autonomous zone holds many data halls, one hall holds many SuperNodes, and one SuperNode holds many racks, all the way down to the package.*

---

**物理稀疏：三个解耦**

- **散热解耦**：兆瓦机架需数百平方米冷却面积；吉瓦机房约千个机架却占地一平方公里、水管拉一公里
- **供电/密度解耦**：不追兆瓦机架，让军队“站开”，再用光把散开的单元绑成一台机器
- **可靠性解耦**：万卡集群中link与module失效是常态——NPO引擎作为可更换单模块，维护面收敛
- **经济逻辑**：**机房地面便宜，机架里芯片贵**——把稀缺资源放稀疏，把廉价资源放开
- 三道坎（散热、供电、可靠性）不再同时顶物理极限，全部落在这个内移动作上

---

**逻辑紧致：三个不变量**

- **一跳覆盖**
  - 一跳覆盖 = **交换芯片radix × 计算芯片radix**，两侧均为乘数
  - 计算芯片自身必须是枢纽，而高密度NPO是其前提：芯片出口变成光纤，才可能容纳大量高速lane
  - chip radix翻倍 → 一跳SuperNode规模翻倍
- **内存语义**：原生load/store、硬件一致性，一次通信往返 **~100 ns**（相比TCP/IP数十微秒约**500倍**提升）
- **协议端到端**：Unified Bus单协议从package贯通至autonomous zone，无转换、无“收费站”

| 指标 | 数值 |
|---|---|
| SuperNode节点规模 | **>8000** |
| 聚合内存带宽 | **6.7 PB/s** |
| 全互连带宽 | **400 Tbps** |
| 全军barrier | **<10 μs** |
| 单芯片I/O | **7.2 Tbps级** |

- 与软件巢的映射关系：**barrier/reduce最频繁的最内层（TP、CP）必须落在一跳覆盖之内**——NPO撑起的radix乘积正是这条底线

![](images/x1.png) *Figure 1:Software nesting dolls: Nested BSP. From outermost to innermost, the levels are data parallel (DP), pipeline parallel (PP), expert parallel (EP), fully sharded data parallel (FSDP), context parallel (CP), and tensor parallel (TP).*

---

**系统级作用：从悬崖到坡道**

- 过去的失败不是坡道而是**悬崖**：跨机框一步，带宽掉一个数量级、延迟翻数倍
- NPO + Unified Bus的工作是把悬崖**填成坡**——越远带宽越低、延迟越高的分形物理被接受，层间断崖被消除
- 完整依赖链：
  - NPO内移边界 → 芯片获得大量光出口 → chip radix提升 → radix乘积扩大、一跳覆盖增长 → 最内层Nested BSP层进入一跳 → barrier/reduce路径变短 → 全军barrier压进10 μs → **τ Scaling在六层各自折叠时间、六层相乘**
- 反面约束：任何一层边界失配（协议、软件、rack间网络），step scatters——**并行单元再多也无法兑现**
- 角色定位：NPO是Nested Parallel von Neumann Architecture的**物理基座**，是硬件巢各层并发、对等单元之间的黏合剂

---

**工程边界与风险对冲**

- 收益递减曲线：**>20 dB → NPO移除9–11 dB → CPO仅再+3 dB**，越靠近die，每毫米铜的“赎回价”越低
- 模块化是NPO的护城河：可构建/可测试/可更换单模块，直接命中大规模制造与万卡运维的失效常态
- 标准化对冲：OIF多方项目 + Unified Bus协议规范开放——“标准大于任何一家公司”，接入者获得同等延迟与同等线性度

---

**总结**

- NPO用**一次边界内移（1 m → 10 mm）**，把铜压缩到最后几厘米、把光铺满五个数量级的外层空间：**物理上每层可站开十倍，逻辑上从package到SuperNode仍是load/store可达的一台计算机**——这是Unified Bus“铜近光远、语言不变”的物理支点，也是Nested BSP最内层高频barrier/reduce压进10 μs的前提。

### 5. 交换芯片与计算芯片双高基数的一跳互联拓扑

**核心论点：双基数乘法是一跳覆盖的第一性公式**

- 论文的关键论断：**one-hop coverage = switch ports × chip ports**，即一跳可达规模 **N₁ ≈ R_sw × R_chip**（交换芯片 radix × 计算芯片上行 radix）
- 两侧**互为乘数**：任一侧 radix 翻倍，一跳覆盖规模等比放大——这是“加士兵”到“成倍扩军”的质变
- 更深一层的论断：**"Raising the product of these two radices sets the parallelism at every nested hardware layer"**——同一乘积公式在 package → board → rack → SuperNode 每个硬件嵌套层**递归成立**，逐层设定该层能容纳的并行单元数
- 落地锚点：SuperNode **>8000 节点**（8192 = 2¹³ 量级）、单芯片 I/O **7.2 Tbps class**、全军 **barrier < 10 μs**

---

**一、问题定义：为什么“一跳”是 Nested BSP 的生死线**

Nested BSP 每层执行同一四步循环：parallel work → barrier → exchange/aggregate → next phase。一跳拓扑就是这四步中后两步的物理载体。

- 最内层并行（**TP / CP / FSDP**）的 barrier 与 reduce 频率最高：TP 每 forward/backward 层都触发 all-reduce，CP 需高频交换 KV——这些通信若跨多跳、跨协议边界，同步开销直接吞噬算力
- 多跳代价是**乘性叠加**：每加一跳 = 一次交换转发 + 一次排队 + 一次（可能的）协议处理；传统 2–3 层 fat-tree 把路径延迟推到 μs 级
- 论文定义的失败模式是“**cliff（断崖）**”：跨机架带宽掉一个数量级、延迟翻数倍；一跳拓扑的使命是把断崖铺成“**slope（斜坡）**”
- **peer equality** 要求任意节点可主动发起通信：一跳 + 对等总线让每层的 barrier/reduce **无需回中心排队**，否则嵌套战法坍缩为 master 前的一条队

---

**二、拓扑数学：二部图模型与覆盖条件**

把 SuperNode 抽象为**计算芯片—交换芯片二部图**：

- 变量：N 个计算芯片、S 台交换芯片、每计算芯片 **R_chip** 条上行链路（接 R_chip 台不同交换机）、每台交换机 **R_sw** 个端口
- **一跳邻居上界**：任一计算芯片经其 R_chip 台交换机可达 R_chip × (R_sw − 1) 个对端，即 **N₁ ≈ R_sw × R_chip**
- **全一跳覆盖条件**：N ≤ R_sw × R_chip（任意两芯片至少共享一台交换机）
- **链路守恒**：N × R_chip = S × R_sw（计算侧上行口总数 = 交换侧端口消耗总数）
- **共享交换机期望**：布线随机时两芯片平均共享 R_chip²/S 台交换机——决定任意点对的**单跳可用带宽**与负载均衡空间；确定性全覆盖需组合设计保证每点对 ≥ 1

参数化推演（论文未公布具体 radix 数值，以下为与文中指标**自洽的示意分解**）：

| 参数 | 示例取值 | 推得结果 | 与论文指标对齐 |
| --- | --- | --- | --- |
| R_sw | 1024 | 单台交换容量 1024 × 400G ≈ **409.6 Tbps** | "full interconnect of **400 Tbps**" |
| R_chip | 8 | 单芯片上行 8 × ~900G ≈ **7.2 Tbps** | "per-chip I/O **7.2 Tbps class**" |
| N₁ = R_sw × R_chip | 1024 × 8 | **8192** | "**>8000 nodes**" |
| S = N × R_chip / R_sw | 8192 × 8 / 1024 | **64 台交换芯片** | 单层二部图即可覆盖 |
| 总链路数 | N × R_chip | **65536 条** | — |

自洽性校验（一种合理读法）：

- **8192 × 7.2 Tbps ≈ 58.9 Pbps ≈ 7.37 PB/s** 原始注入带宽 → 扣除编码/协议开销后 ≈ **6.7 PB/s** 聚合内存带宽（效率 ~91%），与论文数字吻合
- 反算：6.7 PB/s ÷ 8192 ≈ **818 GB/s/节点**（≈ 6.5 Tbps 有效带宽/节点）

---

**三、交换芯片高基数：把层级压掉**

- 同等规模下 radix 越高 → **中间层级越少**：8192 端点用低基数交换需 2–3 层 fat-tree（数百台交换机、多跳路径）；高基数单层二部图**一次交换穿越**直达任意对端
- 单跳路径的延迟预算构成：
  - 铜尾巴：仅最后几厘米（NPO 把电光边界从 1 m 内推到 **10 mm package 边缘**）
  - NPO 光引擎：**10 ns 量级**
  - 光传播：SuperNode 为 10 m 尺度，光纤约 5 ns/m，单向 ~50 ns
  - 合计往返落在**百 ns 量级**——对应论文"one communication round trip ~ **100 ns**"（较 μs 级改善约 **500×**）
- 功耗与损耗：NPO 一步消除全电路径 >20 dB 插损中的 **9–11 dB**（CPO 只能再消 ~3 dB）；每省一跳即省一组 SERDES/re-drive
- 交换侧高基数是行业共识（论文原话 "Everyone agrees on this"），真正的杠杆在另一侧

---

**四、计算芯片高基数：被忽视的乘数**

论文点名这是 "often overlooked" 的一侧——旧观点认为芯片只管算、留一两根出线交给网络，等价于“一个营的调令挤一扇门”。芯片自身必须是 junction，三重收益：

- **收益一：总出口带宽**——R_chip 条上行直乘总 I/O；**7.2 Tbps class** 是 TP/EP 大流量 all-reduce / all-to-all 的带宽前提
- **收益二：路径冗余与容错**——万卡级集群中**链路与光模块失效是常态而非意外**；R_chip = k 条独立路径意味着单链路故障后流量绕行其余 k−1 条，芯片被完全隔离的概率降至 p^k（p 为单链路失效率）；k 条路径同时提供跨交换机的**负载分担**
- **收益三（论文标记 most important）：一跳覆盖乘数**——芯片 radix 与交换 radix **互为乘数**，芯片 radix 翻倍即一跳 SuperNode 规模翻倍
- 深层张力：**算力随面积增长、I/O 带宽只随周长增长**——封装塞得越满，I/O 短板越恶化。这正是芯片侧高基数昂贵、且必须靠 **NPO** 把光引擎贴到 package 边缘来解锁周长约束的原因
- 递归含义：同一公式在每一硬件层重演——package 内 chiplet 出口数 × 片内 fabric radix、board 内芯片 radix × 板级交换 radix、SuperNode 内芯片 radix × SuperNode 交换 radix = **8192**——**每一层的双基数之积，就是该层的并行度上限**

---

**五、NPO：让双高基数在物理上成立**

![](images/x3.png) *Figure 3:Design freedom from NPO: package at ten millimeters, board at ten centimeters, rack at one meter, SuperNode at ten meters, data hall at a hundred meters, and data center at a kilometer. Moving the electro-optical boundary inward from 1 m to 10 mm relaxes every outer layer—physically sparse, logically tight.*

- 电光边界从 **1 m（机箱外）** 内移两个数量级至 **10 mm（package 边缘）**；铜只走最后几厘米，其余全部交给光
- 边界内移后**每一外层解耦**：板级不再对抗几十根铜缆的 insertion loss，机架不必堆到 megawatt，每向外一层尺度放宽十倍——即 "**physically sparse, logically tight**"
- 尺度阶梯：package **10 mm** → board **10 cm** → rack **1 m** → SuperNode **10 m** → data hall **100 m** → data center **1 km**
- NPO vs CPO：光引擎可独立构建、测试、更换的**模块化设计**；延迟 **10 ns 量级**；总成本比 CPO **低 40%+**；已立项为 **OIF** 项目，数十家伙伴参与
- 对双基数的意义：光不惧距离 → 芯片可“站得稀”地铺开 R_chip 条光出线，不必在封装周长上硬拼铜的物理极限

---

**六、软硬件双层对齐：一跳域必须罩住最内层 BSP**

![](images/x1.png) *Figure 1:Software nesting dolls: Nested BSP. From outermost to innermost, the levels are data parallel (DP), pipeline parallel (PP), expert parallel (EP), fully sharded data parallel (FSDP), context parallel (CP), and tensor parallel (TP).*

![](images/x2.png) *Figure 2:Hardware nesting dolls: the body of the machine, from chip package to autonomous zone. One autonomous zone holds many data halls, one hall holds many SuperNodes, and one SuperNode holds many racks, all the way down to the package.*

- 软件巢：**DP → PP → EP → FSDP → CP → TP**，每层同一剧本——并行计算、barrier、reduce
- 硬件巢：**CORE → chiplet → package → board → rack → SuperNode → data hall → autonomous zone**，每层并发、对等
- 对齐规则：**最内、最高频的 BSP 层（TP/CP/FSDP 的 barrier 与 all-reduce）必须整体落在一跳 SuperNode 域内**；外层（DP 每步一次的梯度同步、PP 微批间激活传递）延迟容忍度高，可落在 data hall / autonomous zone 尺度
- **τ Scaling law** 的映射：每个嵌套层各自折叠该层时间常数 τ，**单层折叠有限、六层相乘**——而每层能折多少，正由该层“双基数之积”设定的并行度决定
- 一跳域执行结果：**8192 节点全军 barrier < 10 μs**（树深 log₂8192 = 13 级，每级百 ns 级同步加原子操作竞争，总时间仍压在 10 μs 内）

---

**七、算法流程：一跳二部图上的 collective 执行**

以 SuperNode 内 TP all-reduce 为例：

- **步骤 1（切分）**：待聚合梯度按 R_chip 条上行链路切成 R_chip 份并行流出，同时发往 R_chip 台不同交换机——上行带宽全部吃满
- **步骤 2（一跳传输）**：以**内存语义**执行——对目的芯片地址空间直接 native **load/store**，一致性由硬件维护；同步原语用远端原子操作（fetch-and-add / compare-and-swap）落在共享同步字上，无需任何协议装卸
- **步骤 3（reduce-scatter）**：每台交换机所连的 R_sw 个芯片的对应分片在域内归约，各芯片获得互不重叠的聚合分片
- **步骤 4（all-gather）**：分片沿一跳路径反向广播，全网拼回完整梯度
- **步骤 5（barrier 收口）**：全军对齐，进入下一 Nested BSP phase
- **失效处理**：任一上行链路或光模块故障 → 流量重路由至其余 k−1 条路径，collective 以降带宽**继续执行而非中断**

输入输出关系与整体定位：

- **输入**：并行策略（DP/PP/EP/FSDP/CP/TP）分解出的通信原语流——梯度 all-reduce、KV 交换、expert token 路由、参数分片 all-gather
- **变换**：Unified Bus 将原语映射为对等节点间的 load/store + 原子操作，经“芯片 R_chip × 交换 R_sw”一跳路径完成
- **输出**：**<10 μs 全军 barrier**、聚合一致的梯度/参数视图、**token 成本下降**（芯片算力不再消耗在搬数据上）
- **整体作用**：它是 Nested BSP 每层 "barrier + exchange/aggregate" 两步的**物理执行体**，也是 Nested Parallel von Neumann Architecture 中 "**every node a peer on one bus**" 的物化——没有它，软件巢的每一层 reduce 都要退化为跨协议、多跳、经中心的慢路径

---

**八、关键指标汇总**

| 指标 | 数值 | 层级归属 |
| --- | --- | --- |
| 单芯片 I/O 带宽 | **7.2 Tbps class** | package 边缘（R_chip 路径合计） |
| 通信往返延迟 | **~100 ns**（较 μs 级改善 ~500×） | 内存语义一跳往返 |
| SuperNode 节点数 | **>8000（8192 class）** | 一跳覆盖 R_sw × R_chip |
| 聚合内存带宽 | **6.7 PB/s** | SuperNode 全域 |
| 全互联带宽 | **400 Tbps** | SuperNode 交换侧 |
| 全军 barrier | **<10 μs** | SuperNode 全域 |
| NPO 损耗消除 | **9–11 dB**（CPO 仅再消 ~3 dB） | 电光转换 |
| NPO 延迟 / 成本 | **10 ns 量级 / 低于 CPO 40%+** | 光引擎 |
| 电光边界内移 | **1 m → 10 mm**（两个数量级） | package 边缘 |
| 部署规模 | **256K-node class 单机系统** | autonomous zone 尺度 |

---

**九、与传统方案的对比**

| 维度 | 传统互联栈 | Unified Bus 双高一跳 |
| --- | --- | --- |
| 协议 | 机箱内总线 + 机箱间网络，**两套语言**，每边界一次装卸转换 | **单一协议**从 package 到 autonomous zone 端到端 |
| 通信语义 | 消息传递（TCP/IP 数十 μs；RDMA 仍是 send/reply） | **native load/store** + 硬件一致性 |
| 拓扑 | 2–3 层 fat-tree、多跳、层级递进 | 单层二部图**一跳直达** |
| 角色模型 | CPU/host 为 **master**，设备为 slave | **全对等**，任意节点可发起/响应 |
| 跨机架行为 | **断崖**：带宽掉一个量级、延迟数倍 | **斜坡**：仅受“远则带宽低、延迟高”的物理规律约束 |
| 芯片 I/O | 一两根出线，其余交给网络 | **7.2 Tbps class**，芯片即 junction |
| 失效表现 | 单链路故障易隔离节点 | k 条路径绕行，**降级不中断** |

---

**十、设计权衡与代际纪律**

- radix 不能无限推高：受封装周长/球数、SERDES 每 lane 功耗、光引擎数量与成本约束——对应论文“**算力随 area 增长、I/O 只随 perimeter 增长**”的失配规律
- 每代只打**少数几场硬仗**（≤5 个物理极限，联合成功率才可保住）：当前代押注 **NPO 边界内移 + 双侧 radix 提升**，megawatt 机架等留给后续形态
- 不追空间密度：**"stand sparsely, compute tightly"**——冷却、供电、可靠性三道坎不再同时顶物理极限；机架可以拉开，因为光不介意距离
- 开放性：**Unified Bus Protocol 规范开放发布**，任何接入者获得相同延迟与相同线性度——标准大于任何单一公司，这是一跳拓扑从单点方案走向生态的最后一环


---

## 4. 实验方法与实验结果

**论文证据结构判定**

- 本文属于**架构定位型论文**，而非实证研究论文：全文无受控实验、无训练吞吐 benchmark、无端到端精度/收敛性对比。
- 论文的“实验—结果—消融”三要素以**替代形态**存在：
  - **实验设置** → 系统设计目标、物理约束与部署形态（SuperNode、256K-node 级单系统）。
  - **结果数据** → 散布于第 2–5 节的量化指标（延迟、带宽、规模、损耗）。
  - **消融实验** → 第 3 节“六件事”与第 4 节“三件事”，每项设计决策均附带**移除该设计后的退化模式**，构成**隐式反事实消融**。
- 以下按此替代框架逐项分析。

---

**一、实验设置：系统设计目标与部署形态**

**问题规模设定**

- 目标场景：**十万至百万级处理器**协同，即“百万大军仍是一台计算机”。
- 核心假设：系统瓶颈从单处理器强度转移到**跨层通信与命令传递**——大集群中 **>80% 能量消耗于数据搬运**，其中相当比例损失在协议边界转换上。

**研究范围界定**

- 聚焦硬件中间层：**chip package → board → rack → SuperNode → data hall → autonomous zone**。
- 明确排除两端：更内层（单 soldier 训练）与更外层（campus 间互联）不在本文范围。

**软件侧设置：Nested BSP**

![](images/x1.png) *Figure 1:Software nesting dolls: Nested BSP. From outermost to innermost, the levels are data parallel (DP), pipeline parallel (PP), expert parallel (EP), fully sharded data parallel (FSDP), context parallel (CP), and tensor parallel (TP).*

- 六层并行嵌套，由外到内：**DP → PP → EP → FSDP → CP → TP**，与主流大模型训练并行策略的常见层次一致（TP 最内、最贴近芯片，DP 最外）。
- 每层执行同一套四步循环：**并行计算 → barrier → exchange/aggregate → 下一 phase**，即 map-reduce 语义的递归嵌套。
- 关键约束：**每层单元均为 peer，无 master/slave**——这是区别于经典 BSP 的唯一硬性扩展。
- 并行性来源：**层间乘法关系**——一个 DP 含多个 PP，一个 PP 含多个 EP，一个 EP 含多个 FSDP shard。

**硬件侧设置：嵌套并行 von Neumann 架构**

![](images/x2.png) *Figure 2:Hardware nesting dolls: the body of the machine, from chip package to autonomous zone. One autonomous zone holds many data halls, one hall holds many SuperNodes, and one SuperNode holds many racks, all the way down to the package.*

- 十级物理嵌套：**CORE → chiplet → package → board → rack → SuperNode → data hall → autonomous zone → data center → inter data center**。
- 分形规律：越外层，尺度越大、距离越长、带宽越低、延迟越高（物理规律，被接受为前提）。
- 设计目标不是消除该梯度，而是将历史遗留的“**悬崖**”（跨机箱带宽骤降一个数量级、延迟翻数倍）铺成“**斜坡**”。

**部署载体**

- 单 **SuperNode**：>8000 节点。
- 单 **Super AI computer system**：正在部署的 **256K-node 级**，主张其为“一台计算机”而非二十几万台机器的网络拼接。
- 开放性变量：**Unified Bus Protocol** 规范公开 [8]，构成可复现性/生态变量。

---

**二、结果数据：量化指标汇总**

| 指标 | 数值 | 对照基线 / 推导 |
|---|---|---|
| 通信往返延迟 | **~100 ns** | TCP/IP 软件栈 **数十 μs** → 约 **500×** 改善 |
| 单芯片 I/O 带宽 | **7.2 Tbps** 级 | — |
| SuperNode 节点数 | **>8000** | — |
| SuperNode 聚合内存带宽 | **6.7 PB/s** | — |
| SuperNode 全互联容量 | **400 Tbps** | 维度未指明（见下文质疑） |
| 全军 barrier 延迟 | **<10 μs** | 跨 8000+ 节点 |
| NPO 光损耗消除 | **9–11 dB** | 全电路径总损耗 >20 dB；CPO 在此之上仅再降 ~3 dB |
| NPO 延迟 | **十纳秒级** | — |
| NPO 成本 | 比 CPO **低 >40%** | — |
| 电-光转换边界迁移 | **1 m → 10 mm** | 两个数量级内移 |
| 物理尺度跨度 | **五个数量级** | 10 mm → 1 km（Fig. 3） |

![](images/x3.png) *Figure 3:Design freedom from NPO: package at ten millimeters, board at ten centimeters, rack at one meter, SuperNode at ten meters, data hall at a hundred meters, and data center at a kilometer. Moving the electro-optical boundary inward from 1 m to 10 mm relaxes every outer layer—physically sparse, logically tight.*

**内部一致性核算**

- **6.7 PB/s ÷ 8000 节点 ≈ 840 GB/s/节点**，与 HBM 级本地内存带宽量级吻合——支持“远端内存如本地内存”的主张在算术上自洽。
- **500× 倍数反推**：100 ns × 500 = 50 μs，即隐含假设 TCP/IP 往返约 50 μs，处于合理区间但**未披露测量条件**（消息大小、拓扑、软件栈版本）。
- **>80% 数据搬运能耗**与 **8000 节点 <10 μs barrier** 是 Nested BSP 可行性的两个核心支柱：前者论证改造成本的正当性，后者直接对应外层 BSP 的 barrier 步骤代价。

**关键数据解读**

- **<10 μs 全军 barrier** 是全文最重要的结果：Nested BSP 每层都需要 barrier，若外层 barrier 停留在消息语义（数十 μs 级），六层嵌套的 barrier 链将不可承受。
- **one-hop 覆盖 = switch radix × chip radix**（双乘数关系）：这是第 4 节唯一给出的“性能模型”，芯片 radix 翻倍即 one-hop SuperNode 规模翻倍，决定了最内层、最高频的 Nested BSP 层（barrier/reduce 最密集处）能否落在 one-hop 之内。
- **NPO 的 dB 经济学**：从 >20 dB 全电路径损耗中一次性消除 9–11 dB，而 CPO 相对 NPO 仅再获得 ~3 dB——边际收益递减，换取的却是**可独立制造/测试/更换的光引擎模块**、十纳秒级延迟和 >40% 成本优势。该论证已进入 **OIF** 标准项目 [7]，有产业验证背书。

---

**三、消融分析：六项设计决策的反事实论证**

第 3 节的“六件事”每项均可重构为标准消融形式——**移除设计 X → 观察退化模式 Y**：

|---|---|---|---|
| 1 | **单一协议**（package → autonomous zone 端到端） | 机箱内/外两种语言，每个边界均为“收费站”（unpack-inspect-repack） | >80% 能耗在数据搬运，相当比例损失于边界转换 |
| 2 | **内存语义**（原生 load/store，硬件一致性） | 消息语义“发送-等待”，barrier/reduce 全部卡死，规模越大越糟 | 往返延迟数十 μs → ~100 ns（~500×） |
| 3 | **peer 对等**（CPU/NPU/内存/存储/NIC 同一总线） | master-slave 中心拥塞，"1+1<2"，嵌套作战计划坍缩为 master 处的排队 | 定性论证，无量化数据 |
| 4 | **铜近光远**（NPO） | 铜线速率↑ → 线径↑ → 距离↓，数千根铜缆无法安装 | NPO 消除 9–11 dB；CPO 仅 +3 dB；成本 -40% |
| 5 | **物理稀疏布局** | 追求 MW 级机架：冷却需数百 m²；GW 级机房仅容 ~1000 机架却占地 1 km²，水管延展 1 km | 冷却/供电/可靠性三道物理墙同时对抗 |
| 6 | **每代只打少数硬仗** | 同时攻坚 20 项 90% 成功率的物理极限 | 联合成功率 **0.9²⁰ ≈ 12%** vs 5 项时 **0.9⁵ ≈ 59%** |

第 4 节“三件事”是落地层的**工程消融**：

- **高 radix 交换芯片**：同等规模下减少中间层级，直接节省 latency 与功耗（定性，无数值）。
- **高 radix 计算芯片**：这是论文自称“最常被忽视”的一项，传统观点认为芯片只需 1–2 条出口链路。消融其收益为三条：
  - 总出口带宽提升；
  - **路径冗余**——万卡集群中链路/模块失效是常态而非意外；
  - one-hop 覆盖乘数（最重要）：**switch ports × chip ports**，直接决定 SuperNode one-hop 规模与最内层 Nested BSP 的 barrier/reduce 路径长度。
- **NPO 推至芯片边缘**：电-光边界从 **1 m 内移至 10 mm**，铜只剩最后几厘米，此后每一外层尺度可**×10 扩展**（board 10 cm → rack 1 m → SuperNode 10 m → hall 100 m → data center 1 km），同时“逻辑上仍是一台计算机”——**physically sparse, logically tight** 落脚于这一步。

**消融逻辑的一个量化亮点**

- 第 6 项（0.9²⁰ vs 0.9⁵）是全文唯一显式的概率推理，将“设计范围裁剪”本身纳入设计变量，等同于对**工程野心的消融**。

---

**四、证据强度与缺失评估**

**证据缺口**

- **无端到端训练结果**：主张“加芯片即加算力、单 token 成本下降”，但全文无 LLM 训练吞吐、**MFU**、loss curve、token 成本测量。
- **无命名 baseline 对比**：与 InfiniBand、RoCE、Ethernet 集群以及业界同类 NVLink/NVSwitch 域方案**零对比数据**；~500×、<10 μs 等数字均为单边陈述，无同条件对照。
- **τ Scaling law 引而未证**：文献 [6] 被反复引用为“时间折叠定律”，但本文既未给出定义、公式，也未用本文数据拟合验证；“六层相乘折叠”是**断言**而非测量结论。
- **指标维度缺失**：400 Tbps 未说明是聚合、bisection 还是单平面容量——若按 8000 节点 × 7.2 Tbps = 57.6 Pbps 聚合口径，400 Tbps 无法对齐，量纲归属存疑。
- **可靠性主张无数据**：万卡集群“链路失效是常态”无 MTBF/失效恢复时间支撑。
- **部署状态为进行时**：256K-node 系统描述为 "currently being deployed"，无验收或实测数据。

**证据强度分级**

- 强（有量纲自洽或产业背书）：NPO dB/成本数据（OIF 项目）、6.7 PB/s 与节点数的一致性、尺度阶梯。
- 中（单边量化、无测量条件）：~500× 延迟、7.2 Tbps、<10 μs barrier。
- 弱（纯定性）：peer 对等的“无单一咽喉”论证、τ Scaling law 六层乘法、one-hop 覆盖模型（仅有乘数关系式，无参数值）。

**可复现性**

- 唯一开放变量是 **Unified Bus Protocol 规范** [8]；硬件指标（NPO 光学、radix 设计）均无第三方复现路径。

---

**总结**

- 本文的“实验”是**一套嵌套架构的工程实现**，“结果”是 SuperNode 的规格表（>8000 节点、6.7 PB/s、<10 μs barrier），“消融”是六项设计决策各自的反事实退化论证。
- 论证链条最有力的部分：**NPO 边界内移 1 m → 10 mm** 带来的五数量级尺度松弛，以及 **one-hop 覆盖 = 双 radix 乘积**这一简洁的规模模型。
- 最薄弱的部分：所有关键指标均**缺少 baseline 对照与测量方法披露**，τ Scaling law 与“六层乘法折叠”停留在断言层面，端到端训练有效性（加芯片是否真加算力、token 成本是否真降）尚待 256K-node 系统的实测数据补全。

---

