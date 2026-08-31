# LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference 图表详解

### Fig. 1: An Overview of LLMCompass. LLMCompass can aid the hardware design process as a versatile evaluation tool.

![0c4c3fe5fa1e2b50092a08942a11e0422bfc70af21ba2f7d77a63ed74bdbc3d9.jpg](images/0c4c3fe5fa1e2b50092a08942a11e0422bfc70af21ba2f7d77a63ed74bdbc3d9.jpg)

- **LLMCompass** 是一个专为大语言模型（LLM）推理工作负载设计的硬件评估框架，其系统架构与数据流向如 **Figure 1** 所示。
- 框架通过模块化的设计，将模型计算图与硬件描述转化为性能与面积报告，核心流程包含以下关键组件：
  - **输入层**：
    - **LLM Computational Graph**：提供大语言模型的计算图结构与算子定义。
    - **Hardware Description (Sec. III-A)**：提供目标硬件的架构参数与层级描述。
  - **核心处理层 (LLMCompass Sec. III)**：
    - **Performance Model (Sec. III-B)**：负责执行性能仿真，内部包含 **Mapper**、**Mapping & Scheduling** 以及 **Architecture Simulator**。
    - **Parameter Search**：通过红色闭环箭头标识，在 **Architecture Simulator** 与 **Mapper** 之间进行自动化参数搜索，以寻找性能最优的映射与调度方案。
    - **Area Model (Sec. III-D)**：接收硬件描述，负责估算芯片的物理面积与制造成本。
  - **输出层**：
    - **Performance Report**：由 **Performance Model** 输出的延迟、吞吐量等性能指标报告。
    - **Area Report**：由 **Area Model** 输出的面积与成本分析报告。

- 框架各模块的功能定位与数据流向对应关系如下表所示：

| 阶段 | 模块名称 | 核心功能 | 关联章节 |
| :--- | :--- | :--- | :--- |
| **输入** | LLM Computational Graph | 定义模型计算逻辑与算子依赖 | - |
| **输入** | Hardware Description | 定义硬件架构参数与内存层级 | Sec. III-A |
| **处理** | Performance Model | 执行高层级性能仿真与评估 | Sec. III-B |
| **处理** | Mapper & Scheduling | 管理内存层级、数据复用与任务调度 | Sec. III-B |
| **处理** | Architecture Simulator | 模拟硬件底层执行周期与流水线 | Sec. III-B |
| **处理** | Area Model | 估算芯片物理面积与供应链成本 | Sec. III-D |
| **输出** | Performance Report | 输出系统级与算子级性能指标 | - |
| **输出** | Area Report | 输出硬件设计的面积与成本分析 | - |

- 该框架通过内置的 **Parameter Search** 机制，消除了手动调优的瓶颈，确保在评估不同硬件设计时能够充分挖掘其性能潜力，为架构师提供快速、准确且具备成本感知能力的硬件设计空间探索工具。

### Fig. 2: A Decoder-Only Transformer Layer with Tensor Parallelism. GPT-3 175B [8] consists of a stack of 96 such layers.

![2d9734a222f8912bd70564006ee3e72ce08267487541067b561f67010b658c82.jpg](images/2d9734a222f8912bd70564006ee3e72ce08267487541067b561f67010b658c82.jpg)

- **图片整体概述**：该图展示了带有 **Tensor Parallelism** 的 **Decoder-Only Transformer Layer** 架构。根据图注，**GPT-3 175B** 模型由 96 个这样的层堆叠而成。
- **主图数据流（左侧）**：
  - 数据自下而上流动，依次经过 **LayerNorm - MHA** 和 **Multi Head Attention (MHA)** 模块。
  - MHA 模块后接 **AllReduce MHA** 操作，用于张量并行下的跨设备通信与同步。
  - 随后数据进入 **LayerNorm - FFN** 和 **Feed Forward Network (FFN)** 模块。
  - FFN 模块后同样接 **AllReduce FFN** 操作，完成该层的计算与同步。
- **子图细节展开（右侧）**：
  - **MHA 内部结构（右下）**：包含四个核心矩阵乘法（**Matmul**）和一个 **Softmax** 操作。具体流程为：输入经过 **Matmul: Q_K_V** 生成查询、键、值；随后进行 **Matmul: Q_mul_K** 计算注意力分数，并通过 **Softmax** 归一化；归一化后的权重与值进行 **Matmul: A_mul_V** 计算；最后通过 **Matmul: Wo_proj** 进行输出投影。
  - **FFN 内部结构（右上）**：包含两个矩阵乘法和激活函数。输入首先经过 **Matmul: W1_proj** 进行特征变换，接着通过 **GELU** 激活函数，最后经过 **Matmul: W2_proj** 进行输出投影。
- **核心组件与操作映射表**：

| 模块/层级 | 包含的核心操作/组件 | 功能/作用 |
| --- | --- | --- |
| **MHA (Multi Head Attention)** | **Matmul: Q_K_V**, **Matmul: Q_mul_K**, **Softmax**, **Matmul: A_mul_V**, **Matmul: Wo_proj** | 计算自注意力机制，提取序列上下文特征 |
| **FFN (Feed Forward Network)** | **Matmul: W1_proj**, **GELU**, **Matmul: W2_proj** | 进行非线性特征变换与映射 |
| **Normalization** | **LayerNorm - MHA**, **LayerNorm - FFN** | 稳定计算过程，加速模型收敛 |
| **Communication** | **AllReduce MHA**, **AllReduce FFN** | 在 **Tensor Parallelism** 下聚合跨设备计算结果 |

### Fig. 3: LLMCompass' Hardware Description Template. In this example, each device has 2 cores and each core has 2 lanes.

![aa633e0bfcc516e4a3d0d9ee8b0e59352f9aedf0f653f06070a4f63012dc0e98.jpg](images/aa633e0bfcc516e4a3d0d9ee8b0e59352f9aedf0f653f06070a4f63012dc0e98.jpg)

该图片展示了 **LLMCompass** 的 **Hardware Description Template**（硬件描述模板），以层级化的方式抽象了主流机器学习硬件平台的架构。图中示例配置为每个 **Device** 包含 2 个 **Core**，每个 **Core** 包含 2 个 **Lane**。

- **架构层级结构**
  - **System 级**：由多个 **Device** 通过 **device-device interconnect** 连接组成。
  - **Device 级**：包含多个 **Core**、共享的 **Global Buffer** 以及片外 **Main Memory**。
  - **Core 级**：包含多个 **Lane** 以及共享的 **Local Buffer**。
  - **Lane 级**：独立执行单元，包含 **Vector Unit** 和 **Systolic Array**。

- **硬件组件与互联关系**
  | 组件/层级 | 包含子组件 | 互联方式 (Interconnect) | 功能描述 |
  | :--- | :--- | :--- | :--- |
  | **Device** | **Core**, **Global Buffer**, **Mem PHY** | **off-chip interconnect** (连接 **Main Memory**) | 封装计算核心与全局缓存，通过 **Mem PHY** 访问主存 |
  | **Core** | **Lane**, **Local Buffer** | **on-chip interconnect** (连接 **Global Buffer**) | 封装执行通道与本地缓存，通过片上网络与全局缓存通信 |
  | **Lane** | **Vector Unit**, **Systolic Array** | **on-chip interconnect** (连接 **Local Buffer**) | 独立计算单元，负责向量运算与矩阵乘加操作 |

- **通信与数据流机制**
  - **片外互联 (off-chip interconnect)**：图中红色双向箭头表示，用于 **Device** 与 **Main Memory** 之间，以及不同 **Device** 之间的数据传输。
  - **片上互联 (on-chip interconnect)**：图中黑色双向箭头表示，用于 **Lane** 与 **Local Buffer**、**Local Buffer** 与 **Global Buffer**、以及 **Global Buffer** 与 **Mem PHY/Device PHY** 之间的数据交换。
  - **内存管理**：**Local Buffer** 和 **Global Buffer** 通常由片上 SRAM 构成，由 **mapper** 显式管理，不区分 cache 与 scratchpad，以最大化数据复用和计算效率。

### Fig. 4: Visualization of a Matrix Multiplication in LLMCompass as in Section III-B1.

![c2f4196c7201abf995fb400f697deb2f7e185e5f4720d22ad4d9c43ec56a8446.jpg](images/c2f4196c7201abf995fb400f697deb2f7e185e5f4720d22ad4d9c43ec56a8446.jpg)

*   该图片直观展示了 **LLMCompass** 框架中**矩阵乘法 (Matrix Multiplication)** 的多级内存层次结构、数据分块（Tiling）机制以及多核调度策略，整体划分为 **Hardware Domain**、**Software Domain** 和 **Math Domain**。

*   **多级内存与数据划分机制**
    *   **Main Memory**：存储完整的原始矩阵 Matrix A、Matrix B 和 Matrix C。
    *   **Global Buffer**：采用 **Tile-by-tile** 策略从主存加载数据，缓存 $A\_tile_{m,k}$、$B\_tile_{k,n}$ 和 $C\_tile_{m,n}$ 以最大化数据复用。
    *   **Local Buffer (Cores)**：Global Buffer 中的 Tiles 被进一步细分为 **Sub-tiles**，分发至 Core 0 和 Core 1 的本地缓冲区进行并行计算。

*   **数学域与软件域映射**
    *   底层数学逻辑为 $C += AB$。
    *   软件域展示了矩阵按维度 $M, K, N$ 进行层级分块的过程，对应中间层级的 $C\_tile_{m,n} += A\_tile_{m,k} B\_tile_{k,n}$ 以及最细粒度的 Subtile 累加求和公式。

*   **多核调度方案 (Schedule Schemes) 对比**
    *   图片右侧详细对比了两种将 Sub-tiles 映射到多核计算单元的调度策略。

| 特性维度 | Schedule Scheme 1 | Schedule Scheme 2 |
| :--- | :--- | :--- |
| **任务分配逻辑** | 不同核心处理同一列的**不同** $C\_subtile$ | 不同核心处理**同一个** $C\_subtile$ |
| **计算与写回** | 核心独立计算并直接写回最终结果 | 核心计算部分结果 (Partial results) |
| **后续聚合操作** | 无额外聚合操作 | 必须执行 **Reduction** (累加) 操作 |
| **内存访问优化** | 触发 **Merge** (合并读取相同的 $B\_subtile$) | 无特定内存合并展示 |
| **依赖处理机制** | 自动处理 **Read-After-Write** 依赖 | 依赖 Reduction Wave 进行数据同步 |

*   **关键硬件模拟优化机制**
    *   **内存访问合并 (Merge)**：在 Schedule Scheme 1 的 Wave 0 阶段，Core 0 和 Core 1 均需读取相同的 $B\_subtile_{0,0}$，模拟器自动识别并合并该 Global Buffer 访问请求，降低带宽压力。
    *   **读写依赖处理 (Read-After-Write)**：在 Schedule Scheme 1 的 Wave 1 阶段，Core 0 需要读取其在 Wave 0 写入的 $C\_subtile_{0,0}$。模拟器自动拦截此操作，避免将数据写回 Global Buffer 后再重新读取，从而优化延迟。

### 13c9cea6a318c2290e721c0a10341878cdf8c0e8816a5cb729acf84a75f0f24c.jpg

![13c9cea6a318c2290e721c0a10341878cdf8c0e8816a5cb729acf84a75f0f24c.jpg](images/13c9cea6a318c2290e721c0a10341878cdf8c0e8816a5cb729acf84a75f0f24c.jpg)

- **图片标识**：Figure 5(a)，子图标题为 **Matmul with M = 8192**，用于验证 **LLMCompass** 在矩阵乘法算子上的性能预测准确性。
- **坐标轴定义**：
  - **X轴**：矩阵维度 **N = K**，采用对数刻度，范围从 $2^7$ 到 $2^{15}$。
  - **Y轴**：计算吞吐量 **TFLOPS**，范围从 0 到 300。
- **数据系列**：图中展示了三组不同硬件平台的性能曲线，每组包含真实硬件实测值与 **LLMCompass** 模拟预测值。
- **性能趋势分析**：
  - 当 **N = K** 较小时，**TFLOPS** 处于低位，主要受限于硬件启动开销与计算单元利用率。
  - 随着 **N = K** 维度增大，**TFLOPS** 迅速攀升并逐渐趋于平缓，最终逼近硬件的 **Roofline** 性能上限。
  - **LLMCompass** 的模拟曲线与实测曲线高度重合，证明了其在 **Matmul** 算子上的高保真度（平均误差率 9.0%）。

| 曲线层级 | 峰值 TFLOPS (约) | 趋势特征 |
| --- | --- | --- |
| **最高性能曲线** (绿色) | ~300 | 斜率陡峭，迅速达到饱和，计算利用率极高 |
| **中等性能曲线** (红色) | ~120 | 随维度增加稳步上升，在 $2^{11}$ 后趋于饱和 |
| **最低性能曲线** (蓝色) | ~60 | 增长相对平缓，受限于底层架构或内存带宽 |

- **核心结论**：在固定 **M = 8192** 的 **Matmul** 场景下，**LLMCompass** 能够精准捕捉不同硬件平台随矩阵维度变化的性能拐点与饱和趋势，为大规模 **LLM** 推理的硬件设计空间探索提供了可靠的性能评估基准。

### 46729a704e2d900f14724557eb29d42f09c692e9c69e540889d457106e370e05.jpg

![46729a704e2d900f14724557eb29d42f09c692e9c69e540889d457106e370e05.jpg](images/46729a704e2d900f14724557eb29d42f09c692e9c69e540889d457106e370e05.jpg)

- **图表标识**：该图为 Fig. 5(b)，图注为 **Matmul with N = K = 12288**，隶属于 **Performance Model Validations**（性能模型验证）章节。
- **实验配置**：在单设备（Single GPU/TPU device）环境下测试 **Matmul**（矩阵乘法）算子。维度 **N = K = 12288** 严格对齐 **GPT-3** 的模型隐藏层维度，**M** 为动态变化的矩阵行数。
- **坐标轴解析**：
  - **X轴**：**M**（矩阵行数），采用对数坐标，跨度从 $2^7$ 至 $2^{15}$。
  - **Y轴**：**TFLOPS**（每秒万亿次浮点运算），采用线性坐标，量程覆盖 0 至 300+。
- **曲线映射**：图中多组曲线（实线与虚线结合不同几何标记）分别映射 **NVIDIA A100**、**AMD MI210** 与 **Google TPUv3** 的 **LLMCompass** 模拟值与真实硬件实测值。
- **性能演进趋势**：
  - **冷启动阶段**（$M < 2^9$）：矩阵规模受限导致计算阵列利用率不足，**TFLOPS** 处于低位并呈陡峭上升态势。
  - **饱和阶段**（$M \ge 2^{11}$）：计算负载足以掩盖内存与调度开销，硬件算力被充分榨取，**TFLOPS** 曲线趋于平缓并触及物理峰值（最高组曲线逼近 300 TFLOPS）。
- **验证结论**：**LLMCompass** 的预测曲线与真实硬件轨迹高度重合。论文指出 **Matmul** 算子的平均误差率控制在 **9.0%**，证实了框架在处理大规模密集计算时的卓越准确性。

| 硬件平台 | 数据精度 | 峰值 TFLOPS 表现 | 模型拟合度 |
| :--- | :--- | :--- | :--- |
| **NVIDIA A100** | FP16 | 逼近 300 TFLOPS | 极高，曲线几乎重叠 |
| **AMD MI210** | FP16 | 约 150 TFLOPS | 高，极小 M 值处略有偏差 |
| **Google TPUv3** | BF16 | 约 50-60 TFLOPS | 高，整体趋势完全一致 |

### aaa7ea84985a47d73b78eb3b57d9425a2c977bd14fce4087a340c19c04cd1707.jpg

![aaa7ea84985a47d73b78eb3b57d9425a2c977bd14fce4087a340c19c04cd1707.jpg](images/aaa7ea84985a47d73b78eb3b57d9425a2c977bd14fce4087a340c19c04cd1707.jpg)

- **图片主题**：图5(c)展示了 **Softmax** 算子在固定维度 **N = 4096** 下的性能模型验证结果，用于评估 **LLMCompass** 框架的预测准确性。
- **坐标轴解析**：
  - **X轴**：矩阵行数 **M**，采用对数刻度，范围从 $2^7$ 至 $2^{15}$。
  - **Y轴**：计算吞吐量 **G Elements/s**，线性刻度，范围从 0 至 400。
- **数据趋势表现**：
  - **低负载区间**：当 **M** 较小（$2^7$ 至 $2^9$）时，吞吐量极低，主要受限于内核启动开销（kernel launch overhead）及硬件利用率不足。
  - **高负载区间**：随着 **M** 增大至 $2^{11}$ 及以上，吞吐量呈陡峭上升趋势，峰值逼近 **400 G Elements/s**，表明硬件计算单元被充分激活。
  - **模型拟合度**：图中实线（真实硬件）与虚线（**LLMCompass** 预测）走势高度吻合，验证了框架的可靠性。
- **核心验证结论**：
  - 该图表证实了 **LLMCompass** 在处理包含归约（reduction）操作的 **Softmax** 算子时，能够实现 **12.0%** 的平均误差率。
  - 尽管 **Softmax** 和 **LayerNorm** 因涉及归约操作导致误差略高于元素级算子（如 **GELU**），但框架仍能准确捕捉到不同输入规模下的性能变化趋势。

| 评估维度 | 详细参数与表现 |
| :--- | :--- |
| **目标算子** | **Softmax** (归一化维度 N = 4096) |
| **变量维度** | **M** ($2^7 \rightarrow 2^{15}$) |
| **峰值吞吐量** | 约 **400 G Elements/s** (当 M = $2^{15}$) |
| **平均误差率** | **12.0%** (针对 Softmax 算子) |
| **测试环境** | 单设备测试 (**NVIDIA A100**, **AMD MI210**, **Google TPUv3**) |

### 73bebbc85ca6dcde99e7c208da482ca2d506d72db6bf9400753dec3a40090521.jpg

![73bebbc85ca6dcde99e7c208da482ca2d506d72db6bf9400753dec3a40090521.jpg](images/73bebbc85ca6dcde99e7c208da482ca2d506d72db6bf9400753dec3a40090521.jpg)

- **图表基本信息**：该图为 Figure 5 的子图 (e)，用于验证 **LLMCompass** 性能模型在 **LayerNorm** 操作上的准确性。
- **坐标轴定义**：
  - **X轴**：归一化维度 **N**（对数尺度，范围从 $2^7$ 到 $2^{15}$），固定 **M = 4096**。
  - **Y轴**：计算吞吐量，单位为 **G Elements/s**（范围 0 至 400）。
- **核心趋势分析**：
  - 随着维度 **N** 的增加，吞吐量呈现**先上升后下降**的非线性趋势。
  - 当 **N** 增加到极端值时，由于 **reduction** 操作成本的急剧增加，吞吐量出现明显回落。
- **模型验证结论**：
  - **LLMCompass** 成功捕捉到了这一由 **reduction** 成本导致的性能下降趋势。
  - 证明了该框架比传统的 **Roofline model** 更具准确性，能够反映真实硬件的复杂行为。
- **误差来源说明**：
  - **LayerNorm** 涉及跨维度的 **reduction** 计算，导致其平均误差率（13.8%）高于简单的 element-wise 操作（如 GELU 的 5.0%）。

| 特征维度 | 详细说明 |
| --- | --- |
| **测试算子** | LayerNorm (M = 4096, 在 N 维度归一化) |
| **X轴变量** | 归一化维度 N ($2^7$ 至 $2^{15}$) |
| **Y轴指标** | 吞吐量 (G Elements/s) |
| **性能拐点** | 当 N 极大时，吞吐量因 reduction 成本增加而下降 |
| **模型表现** | LLMCompass 准确拟合了真实硬件的下降趋势 |

### ca35d6a33bdfc7e7295c954f61f2eb1c01d4ebca9da7476d40bc2fabd7b8a658.jpg

![ca35d6a33bdfc7e7295c954f61f2eb1c01d4ebca9da7476d40bc2fabd7b8a658.jpg](images/ca35d6a33bdfc7e7295c954f61f2eb1c01d4ebca9da7476d40bc2fabd7b8a658.jpg)

该图片为论文中的 **Figure 5(h)**，展示了 **All-reduce** 通信原语在不同硬件平台上的性能验证结果，旨在评估 **LLMCompass** 框架在模拟设备间通信时的准确性。

* **坐标轴定义**
  * **X轴 (M)**：表示 **All-reduce** 操作的数据量大小（元素数量），范围从 $2^7$ 到 $2^{15}$。
  * **Y轴 (G Elements/s)**：表示通信吞吐量，单位为十亿元素每秒。

* **数据系列对比**
  图表对比了三种主流 AI 加速器的 **Roofline** 理论上限、**Real**（真实硬件测试）与 **Simulated**（LLMCompass 模拟）性能。

| 硬件平台 | 图例颜色 | Roofline 表现 | Real 表现 | Simulated 表现 | 峰值吞吐量 (约) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA A100** | 绿色 | 最高 (虚线) | 随数据量显著上升 (圆点) | 高度贴合 Real (叉号) | ~280 G Elements/s |
| **AMD MI210** | 红色 | 居中 (虚线) | 随数据量平缓上升 (圆点) | 高度贴合 Real (叉号) | ~110 G Elements/s |
| **Google TPUv3** | 蓝色 | 极低 (虚线) | 基本保持低位 (圆点) | 高度贴合 Real (叉号) | ~15 G Elements/s |

* **核心分析与发现**
  * **模拟准确性高**：**LLMCompass** 的 **Simulated** 曲线（叉号）与 **Real** 曲线（圆点）在三种硬件上均高度重合，证明了其在通信原语建模上的高保真度。
  * **Roofline 模型的局限性**：所有硬件的 **Real** 和 **Simulated** 性能均显著低于 **Roofline** 理论上限（虚线），尤其是在小数据量（$2^7$ 至 $2^9$）阶段。这表明 **All-reduce** 操作受限于通信延迟和协议开销，**Roofline** 模型无法准确捕捉这些非理想硬件特性。
  * **硬件互联差异**：**NVIDIA A100** 凭借 **NVLink** 的高带宽优势，在 **All-reduce** 吞吐量上大幅领先 **AMD MI210** 和 **Google TPUv3**，凸显了设备间互联带宽对张量并行（Tensor Parallelism）性能的决定性作用。

### 64a94f3ef867da41203ee8a5cf32c7f843e899c8599365f2a9b0cac984394a9b.jpg

![64a94f3ef867da41203ee8a5cf32c7f843e899c8599365f2a9b0cac984394a9b.jpg](images/64a94f3ef867da41203ee8a5cf32c7f843e899c8599365f2a9b0cac984394a9b.jpg)

- **图表主题**：本图对应论文中的 **Figure 5(e)**，展示了在矩阵行数 $M=4096$ 时，**LayerNorm** 算子在不同硬件平台上的吞吐量（**G Elements/s**）随归一化维度 $N$ 变化的性能验证结果。
- **坐标轴定义**：X轴为归一化维度 $N$（对数尺度，范围 $2^7$ 至 $2^{15}$），Y轴为吞吐量（**G Elements/s**）。
- **对比维度**：图表同时对比了三种主流 AI 硬件（**NVIDIA A100**、**AMD MI210**、**Google TPUv3**）的 **Roofline** 理论上限、**Real**（真实硬件测试）与 **Simulated**（**LLMCompass** 模拟）数据。

- **模型准确性验证**：**LLMCompass** 的 **Simulated** 曲线与 **Real** 曲线在所有硬件平台上均高度重合，证明了其在 **LayerNorm** 算子上的模拟精度极高，能够精准反映真实硬件行为。
- **硬件性能分化**：
  - **NVIDIA A100** 表现最优，峰值吞吐量接近 **270 G Elements/s**。
  - **AMD MI210** 居中，峰值吞吐量约为 **90 G Elements/s**。
  - **Google TPUv3** 表现最弱，吞吐量始终低于 **20 G Elements/s**，表明其在非矩阵乘法算子上的架构适配性较弱。
- **Reduction 维度瓶颈洞察**：当 $N$ 增大至 $2^{13}$ 及以上时，**Real** 与 **Simulated** 曲线均出现明显的性能拐点或下降趋势。这揭示了 **LayerNorm** 在极端 **Reduction** 维度下的计算与同步开销。
- **Roofline 模型局限性**：传统的 **Roofline** 模型（虚线）无法预测上述性能下降，依然保持线性或平缓增长，凸显了 **LLMCompass** 在捕捉复杂算子微架构行为方面的优越性。

| 硬件平台 | 峰值吞吐量 (G Elements/s) | 模拟与真实吻合度 | 大维度 N 趋势特征 |
| :--- | :--- | :--- | :--- |
| **NVIDIA A100** | ~270 | 极高，曲线几乎重合 | 达到峰值后出现回落 |
| **AMD MI210** | ~90 | 极高，曲线几乎重合 | 达到峰值后出现回落 |
| **Google TPUv3** | ~15 | 极高，曲线几乎重合 | 整体平缓，无明显峰值 |

- **核心结论**：该图不仅验证了 **LLMCompass** 对 **LayerNorm** 等包含 **Reduction** 操作算子的高保真模拟能力，还深刻揭示了硬件在处理高维归一化时的性能瓶颈，弥补了传统分析模型的盲区，为后续硬件设计中的计算单元与 **Buffer** 分配提供了关键依据。

### 7b3c126ed5442d3d6eee1112d25b82935097378000793834ee3f323ead2e1942.jpg

![7b3c126ed5442d3d6eee1112d25b82935097378000793834ee3f323ead2e1942.jpg](images/7b3c126ed5442d3d6eee1112d25b82935097378000793834ee3f323ead2e1942.jpg)

- **图表主题**：该图表展示了 **LLMCompass** 性能模型在特定算子（如 Softmax、LayerNorm 或 All-reduce）上的**吞吐量（Throughput）验证结果**，对比了模拟数据与真实硬件的表现。
- **坐标轴定义**：
  - **X轴**：`Elements`（元素数量），采用对数刻度，范围从 $2^{12}$ 跨越至 $2^{27}$ 以上。
  - **Y轴**：`G Elements/s`（每秒十亿元素处理量），线性刻度，范围从 0 到 500。
- **数据趋势分析**：图表包含三组不同性能层级的数据曲线，每组均由**实线（真实硬件）**与**虚线（LLMCompass 模拟）** 组成，具体表现如下表所示：

| 曲线特征 | 峰值吞吐量 (G Elements/s) | 增长趋势与硬件表现 |
| :--- | :--- | :--- |
| **绿色 (圆形标记)** | **~400** | 在 $2^{21}$ 元素后呈指数级急剧上升，代表**最高性能**配置或算子。 |
| **紫色 (方形标记)** | **~150** | 在 $2^{18}$ 元素后开始稳步爬升，代表**中等性能**层级。 |
| **蓝色 (三角形标记)** | **~50** | 整体增长极为平缓，代表**受限性能**层级或高开销算子。 |

- **模型准确性验证**：在所有元素规模下，**LLMCompass 的虚线预测轨迹与真实硬件的实线轨迹高度重合**，证明了该框架在捕捉算子吞吐量随输入规模变化趋势时的**极高准确性**。
- **上下文标注纠错**：尽管文档正文中将该图片标记为 `(1) TPU Decoding Latency`，但根据 Y 轴单位 `G Elements/s` 及曲线形态判断，此图实际反映的是**算子吞吐量（Throughput）** 而非延迟（Latency），属于 Fig. 5 中针对单设备算子性能验证的子图（如 Softmax、LayerNorm 或 All-reduce）。

### 319ed35afd23191387dde46ddb54d806720cb2c207f69d410590ecc86be47718.jpg

![319ed35afd23191387dde46ddb54d806720cb2c207f69d410590ecc86be47718.jpg](images/319ed35afd23191387dde46ddb54d806720cb2c207f69d410590ecc86be47718.jpg)

- **图片标识**：Figure 5(g)，用于验证 **LLMCompass** 框架对 **GELU** 算子的性能模拟准确性。
- **坐标轴定义**：
  - **X轴**：**Data Size (Bytes)**，范围从 $2^{11}$ 到 $2^{35}$，表示输入数据规模的对数刻度。
  - **Y轴**：**Bandwidth (GB/s)**，范围从 0 到 150，表示算子执行时的有效内存带宽。
- **数据系列与趋势分析**：
  - **NVIDIA A100 Node**：**Real**（绿色圆点）与 **Simulated**（绿色叉号）曲线高度重合。带宽随数据量增加呈非线性增长，在数据量大于 $2^{27}$ Bytes 时趋于饱和，峰值带宽稳定在 **150 GB/s** 左右。
  - **Google TPUv3 Node**：**Real**（蓝色圆点）与 **Simulated**（蓝色叉号）曲线高度重合。带宽增长趋势与 A100 类似，但受限于硬件架构与内存带宽，峰值带宽饱和在 **40 GB/s** 左右。
- **核心结论**：
  - **GELU** 作为 **element-wise** 算子，计算与访存模式简单，**LLMCompass** 对其模拟的平均误差率仅为 **5.0%**，是所有验证算子中精度最高的。
  - 模拟器能够精准捕捉不同硬件平台（**NVIDIA A100** 与 **Google TPUv3**）在 **GELU** 算子上的带宽瓶颈与饱和趋势，证明了其在 **element-wise** 算子建模上的高保真度。

| 硬件平台 | 数据系列 | 峰值带宽 (GB/s) | 算子类型 | 模拟误差率 |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA A100 Node** | Real / Simulated | ~150 | **element-wise** | **5.0%** |
| **Google TPUv3 Node** | Real / Simulated | ~40 | **element-wise** | **5.0%** |

### 5a4546c5013eb12e8343364cad8c9439d1c20af45db753448f35d158b6983718.jpg

![5a4546c5013eb12e8343364cad8c9439d1c20af45db753448f35d158b6983718.jpg](images/5a4546c5013eb12e8343364cad8c9439d1c20af45db753448f35d158b6983718.jpg)

- **图表基本信息**
  - 图表类型：**堆叠柱状图 (Stacked Bar Chart)**。
  - 对应论文图表：**Fig. 5(i) GPU Prefill Latency**。
  - X轴类别：**Real A100**、**Simulated A100**、**Roofline Model**。
  - Y轴指标：**Latency (s)**，刻度范围介于 0.00 至 0.065 秒之间。

- **数据对比与分析**
  - **Real A100** 与 **Simulated A100** 的总延迟高度吻合，均约为 **0.065s**，验证了 **LLMCompass** 性能模型在 **Prefill** 阶段的高准确性。
  - **Roofline Model** 的总延迟约为 **0.058s**，明显低于实际硬件和仿真结果，表明传统 **Roofline Model** 在评估 **LLM Inference** 时过于乐观，无法准确反映真实硬件的隐藏开销。
  - 堆叠色块代表 **Transformer Layer** 中不同算子（如 **Matmul**、**Softmax**、**LayerNorm** 等）的延迟贡献，**Simulated A100** 成功复现了 **Real A100** 中各算子的延迟分布比例。

- **延迟数据估算表**

| 评估模型 | 总延迟 (Latency) | 相对误差/特征 |
| :--- | :--- | :--- |
| **Real A100** | ~0.065 s | 真实硬件基准 (Ground Truth) |
| **Simulated A100** | ~0.065 s | 与真实硬件高度一致，误差极小 |
| **Roofline Model** | ~0.058 s | 低估延迟，结果过于乐观 |

### 0e5a6b877d1ef2b91f6fcb771efea41032535fd81c0fd9e5f8475b1f74b958e5.jpg

![0e5a6b877d1ef2b91f6fcb771efea41032535fd81c0fd9e5f8475b1f74b958e5.jpg](images/0e5a6b877d1ef2b91f6fcb771efea41032535fd81c0fd9e5f8475b1f74b958e5.jpg)

- **图表标识**：该图为论文中的 **Figure 5(j)**，展示了 **TPU Prefill Latency** 的模型验证结果。
- **对比维度**：图表通过堆叠柱状图对比了 **Real TPUv3**（真实硬件测试）、**Simulated TPUv3**（LLMCompass 仿真）与 **Roofline Model**（理论分析模型）的延迟（**Latency**）表现。
- **数据量化对比**：

| 评估对象 | 总延迟 (Latency) | 误差与特征分析 |
| :--- | :--- | :--- |
| **Real TPUv3** | **~0.21s** | 真实硬件基准，包含所有底层微架构与调度开销 |
| **Simulated TPUv3** | **~0.21s** | 与真实硬件**高度吻合**，验证了仿真框架的**高准确性** |
| **Roofline Model** | **~0.17s** | 显著低于真实值，表现出**过度乐观 (overly optimistic)** 的缺陷 |

- **核心洞察**：
  - **仿真精度验证**：**Simulated TPUv3** 的柱状图总高度及各颜色分层（代表不同算子如 **Matmul**, **Softmax**, **LayerNorm** 等的耗时）与 **Real TPUv3** 几乎完全一致，证明 LLMCompass 能精准映射 TPU 架构的 **Prefill** 阶段性能。
  - **Roofline 模型局限**：**Roofline Model** 仅考虑了计算与内存带宽的理论峰值，忽略了硬件调度、通信同步及微架构开销，导致其预测结果**严重偏离**真实场景。
  - **算子延迟分布**：堆叠结构清晰展示了 **Prefill** 阶段中计算密集型算子（如 **Matmul**）占据主导延迟，而 LLMCompass 成功捕获了这种细粒度的**算子级延迟分布**，具备极强的可解释性。

### 7901581bea7cfe37fb0300c22cbf06d3cb03ca14f877e78631182e9ae513294e.jpg

![7901581bea7cfe37fb0300c22cbf06d3cb03ca14f877e78631182e9ae513294e.jpg](images/7901581bea7cfe37fb0300c22cbf06d3cb03ca14f877e78631182e9ae513294e.jpg)

- **图表标识**：Figure 5(k) GPU Decoding Latency。
- **图表类型**：堆叠柱状图，用于对比不同评估方法下的 **GPU Decoding Latency** 表现。
- **对比维度**：包含 **Real A100**（真实硬件基准）、**Simulated A100**（LLMCompass 模拟结果）与 **Roofline Model**（理论峰值模型）。
- **核心发现**：
  - **Simulated A100** 与 **Real A100** 的总延迟高度吻合，验证了 LLMCompass 在 Decoding 阶段的**高准确性**。
  - **Roofline Model** 的延迟显著偏低，证明传统 Roofline 模型在评估 LLM 推理时**过于乐观**，无法反映真实硬件的算子调度、内存层级与通信开销。
  - 柱状图的**多色堆叠层**精准复现了 Transformer 层内各算子（如 Matmul、Softmax、LayerNorm 等）在真实硬件上的耗时占比分布。

| 评估模型 | 预估总延迟 (ms) | 准确性与特性评价 |
| :--- | :--- | :--- |
| **Real A100** | ~1.10 | 真实基准 (Ground Truth)，包含所有硬件与软件开销。 |
| **Simulated A100** | ~1.02 | **高度拟合**，误差极小，具备极强的架构描述能力。 |
| **Roofline Model** | ~0.65 | **严重低估**，仅考虑理论带宽与算力上限，缺乏实际指导意义。 |

### daff8e8777cdba97a562fe1f168144765cec0939ae2f8fd4b5e55c369fb0f75f.jpg

![daff8e8777cdba97a562fe1f168144765cec0939ae2f8fd4b5e55c369fb0f75f.jpg](images/daff8e8777cdba97a562fe1f168144765cec0939ae2f8fd4b5e55c369fb0f75f.jpg)

- **图表身份**：该图片为论文中的 **Fig. 5(l)**，展示了 **TPU Decoding Latency**（TPU解码阶段延迟）的性能模型验证结果。
- **图表类型**：堆叠柱状图（Stacked Bar Chart），用于对比不同评估方法下各算子延迟的分布与总和。
- **坐标轴定义**：
  - **X轴**：评估方法，包含 **Real TPUv3**（真实硬件）、**Simulated TPUv3**（LLMCompass模拟）和 **Roofline Model**（理论上限模型）。
  - **Y轴**：延迟时间 **Latency (ms)**，刻度范围从 0 到 6。
- **算子分解（图例）**：图表将解码阶段的总延迟拆解为多个底层算子与通信原语，具体分类如下表所示：

| 算子类别 | 包含的具体操作 (英文术语) |
| :--- | :--- |
| **通信原语** | `AllReduce_FFN`, `AllReduce_MHA` |
| **激活与归一化** | `GeLU`, `LayerNorm_FFN`, `LayerNorm_MHA`, `Softmax` |
| **FFN层投影** | `W1_proj`, `W2_proj`, `Wo_proj` |
| **注意力机制计算** | `Q_K_V`, `Q_mul_K`, `A_mul_V` |

- **核心数据对比与分析**：
  - **高保真度验证**：**Real TPUv3** 与 **Simulated TPUv3** 的总延迟高度吻合（均在 **5.5ms - 5.8ms** 区间），且各算子的堆叠比例几乎一致，证明了 **LLMCompass** 在 TPUv3 架构上对解码阶段性能预测的**极高准确性**。
  - **Roofline 模型的局限性**：**Roofline Model** 预测的总延迟显著偏低（约 **4.0ms**），因为它仅基于理论计算峰值与内存带宽，**忽略了实际硬件的调度开销、通信延迟及算子启动成本**，导致结果过于乐观。
  - **延迟瓶颈分布**：在解码阶段，**注意力机制计算**（`Q_K_V`, `Q_mul_K`, `A_mul_V`）与 **FFN层投影**（`W1_proj`, `W2_proj`）占据了绝大部分延迟，表明该阶段属于典型的 **IO-bound（访存受限）** 场景，矩阵乘法主要受限于内存带宽而非计算单元。
- **结论支撑**：此图直接支撑了论文的核心论点，即 **LLMCompass** 能够比传统的 **Roofline Model** 更精准地捕捉复杂硬件架构（如 TPUv3）上的真实运行趋势，为后续的硬件设计空间探索提供了可靠的评估基准。

### 7ca1fbddec15530805f5c6405d0f293ee6953acf99befcbf53319f3d22b0d405.jpg

![7ca1fbddec15530805f5c6405d0f293ee6953acf99befcbf53319f3d22b0d405.jpg](images/7ca1fbddec15530805f5c6405d0f293ee6953acf99befcbf53319f3d22b0d405.jpg)

* **图表主题**：该图展示了 **NVIDIA GA100** 与 **AMD Aldebaran** 芯片的实际（Real）与模拟（Simulated）芯片面积（Die Area）分解对比，旨在验证 **LLMCompass** 面积模型的准确性。
* **坐标轴信息**：
  * **X轴**：包含四个对比类别，分别为 **Real GA100**、**Simulated GA100**、**Real Aldebaran** 和 **Simulated Aldebaran**。
  * **Y轴**：表示芯片面积 **Area (mm²)**，刻度范围从 0 延伸至 800 以上。
* **图例组件构成**：图表采用堆叠柱状图形式，自下而上详细拆解了以下硬件模块：
  * **Cores**（计算核心，占比最大，呈红色）
  * **On-chip interconnect**（片上互连，绿色）
  * **Global buffer**（全局缓冲区，浅绿色）
  * **Memory(PHY)**（内存物理层，棕色）
  * **Memory(Control)**（内存控制器，浅棕色）
  * **Device-device interconnect(PHY)**（设备间互连物理层，蓝色）
  * **Device-device interconnect(Control)**（设备间互连控制器，浅蓝色）
  * **Other**（其他未明确分类的组件，灰色）
* **模型验证结果**：
  * **GA100 精度**：模拟面积与实际面积高度吻合，误差率仅为 **5.1%**。
  * **Aldebaran 精度**：模拟面积与实际面积的误差率为 **8.1%**。
  * **误差归因**：差异主要来源于专有的核心微架构（micro-architecture）以及难以精确估算的核心间通信开销（core-to-core communication overheads）。
* **组件面积分布与功能解析**：

| 组件层级 (Component Level) | 英文术语 (Terminology) | 面积占比特征 (Area Characteristic) |
| :--- | :--- | :--- |
| **计算单元** | **Cores** | 占据绝对主导地位，是芯片面积的最大消耗者 |
| **缓存与互连** | **Global buffer**, **On-chip interconnect** | 占据中等比例，支撑核心间数据交换与暂存 |
| **内存子系统** | **Memory(PHY)**, **Memory(Control)** | 占据一定比例，负责高带宽内存（如 HBM）接口 |
| **设备间通信** | **Device-device interconnect(PHY/Control)** | 占比较小，用于多芯片高速互联（如 NVLink） |
| **其他杂项** | **Other** | 包含控制逻辑、IO 等未单独列出的开销 |

### f0a09d8cf53f1762d2c6e261b6691baf31f3f6e133b233a9e867597855cf352a.jpg

![f0a09d8cf53f1762d2c6e261b6691baf31f3f6e133b233a9e867597855cf352a.jpg](images/f0a09d8cf53f1762d2c6e261b6691baf31f3f6e133b233a9e867597855cf352a.jpg)

* **图表主题**：该图展示了 **NVIDIA GA100**（Stream Multiprocessor）与 **AMD Aldebaran**（Compute Unit）的**核心面积分解（Core Area Breakdown）** 验证结果，用于评估 **LLMCompass** 面积模型的准确性。
* **坐标轴与图例**：
  * **Y轴**：表示面积（Area），单位为 **mm²**，刻度范围从 0 到 4。
  * **X轴**：包含四个对比类别，分别为 **Real GA100**、**Simulated GA100**、**Real Aldebaran** 和 **Simulated Aldebaran**。
  * **图例组件**：堆叠柱状图由五个核心硬件组件构成，自上而下依次为 **Local buffer**（浅蓝色）、**Register file**（深蓝色）、**Systolic array**（浅绿色）、**ALUs**（深绿色）和 **Control logic**（红色）。
* **数据对比分析**：
  * **真实硬件（Real）**：以**灰色单色柱**表示，**Real GA100** 与 **Real Aldebaran** 的核心总面积均约为 **4 mm²**。
  * **模拟硬件（Simulated）**：以**彩色堆叠柱**表示，**Simulated GA100** 的总面积略高于 4 mm²，而 **Simulated Aldebaran** 的总面积略低于 4 mm²，整体与真实硬件面积高度吻合，证明了模型的可靠性。
* **面积占比分布**：
  * 在两款模拟核心中，**Register file** 均占据了**最大的面积比例**。
  * **Control logic** 与 **ALUs** 占据了中等比例面积。
  * **Systolic array** 与 **Local buffer** 的面积占比较小。

* **组件面积分布概览**：

| 硬件组件 (Component) | 颜色标识 | 面积占比趋势 (Simulated) |
| :--- | :--- | :--- |
| **Local buffer** | 浅蓝色 | 最小 |
| **Register file** | 深蓝色 | **最大** |
| **Systolic array** | 浅绿色 | 较小 |
| **ALUs** | 深绿色 | 中等 |
| **Control logic** | 红色 | 中等偏大 |

### 8531387920e7e6a2a6bb78925379b4ad903d4410090ad3101776b19486e1a915.jpg

![8531387920e7e6a2a6bb78925379b4ad903d4410090ad3101776b19486e1a915.jpg](images/8531387920e7e6a2a6bb78925379b4ad903d4410090ad3101776b19486e1a915.jpg)

- **图表基本信息**：该图展示了五种不同计算系统配置（A至E）下，单层 GPT-3 模型的 **Prefill Latency (TTFT)** 与芯片 **Area** 的对比关系，对应论文 Figure 7(a)。
- **坐标轴与图例说明**：
  - **左 Y 轴**：**Latency (s)**，通过堆叠柱状图呈现各底层算子的延迟贡献。
  - **右 Y 轴**：**Area (mm²)**，通过带 'x' 标记的灰色虚线折线图呈现芯片面积。
  - **X 轴**：**Configurations**，对应论文 Table III 中的五种硬件设计（A为1/4算力，B为完整 GA100，C/D/E为同总算力下的大核心设计）。
- **算子延迟拆解**：
  | 算子类别 | 包含的具体操作 | 视觉颜色标识 |
  | --- | --- | --- |
  | **通信原语** | AllReduce_FFN, AllReduce_MHA | 浅蓝、深蓝 |
  | **激活与归一化** | GeLU, LayerNorm_FFN, LayerNorm_MHA, Softmax | 棕、浅绿、深绿 |
  | **矩阵乘法 (MLP)** | W2_proj, W1_proj, Wo_proj | 橙、红、深红 |
  | **矩阵乘法 (Attention)** | A_mul_V, Q_mul_K, Q_K_V | 紫红、深紫 |
- **数据趋势分析**：
  - **配置 A 到 B（算力提升）**：**Latency** 从约 0.18s 骤降至 0.055s，但 **Area** 从约 480 mm² 激增至 826 mm²。表明 **Prefill 阶段是 Compute-bound**，增加计算能力能显著降低延迟。
  - **配置 B 到 E（核心合并）**：保持总算力和总 Buffer 不变，减少核心数并增大单核的 **Systolic Array** 和 **Local Buffer**。
    - **Latency** 呈现轻微上升趋势（从 0.055s 增至约 0.06s）。原因是更大的 Tile 尺寸导致 Padding 增加，硬件难以被完全利用。
    - **Area** 呈现缓慢下降趋势（从 826 mm² 降至约 760 mm²）。表明大核心设计具有更高的面积效率。
- **架构设计启示**：
  - **算力与延迟的强相关性**：在 **Prefill** 阶段，计算资源的增加直接转化为延迟的降低。
  - **面积效率与硬件利用率的权衡**：更大的 **Systolic Array** 和 **Vector Unit** 虽然能减少芯片面积，但会增加调度难度并降低利用率。配置 B（类似 NVIDIA GA100）在延迟表现和面积成本之间取得了较优的平衡。

### 630dfd7bb1626bfa508a452eeac393a096b938e256fceeb2d4216ae7e3f48332.jpg

![630dfd7bb1626bfa508a452eeac393a096b938e256fceeb2d4216ae7e3f48332.jpg](images/630dfd7bb1626bfa508a452eeac393a096b938e256fceeb2d4216ae7e3f48332.jpg)

- **图表概述**：该图为双Y轴组合图，直观展示了五种不同计算系统配置（Configurations A-E）在 LLM 推理 **Decoding** 阶段的延迟分解及对应的芯片面积（Area）对比。
- **坐标轴与图例解析**：
  - **左Y轴**：Latency (ms)，衡量每生成一个 token 的解码延迟，范围 0.0 至 1.0 ms。
  - **右Y轴**：Area (mm²)，衡量硬件设计的芯片面积，范围 0 至 1000 mm²。
  - **X轴**：五种硬件配置 A、B、C、D、E（基于 Table III，总计算能力与 Buffer 大小基本一致，仅核心规模与向量宽度不同）。
  - **图例**：堆叠柱状图代表各 Transformer 算子的延迟贡献（如 Q_K_V, W1_proj 等）；灰色虚线带 'x' 标记代表芯片面积。
- **数据趋势与洞察**：
  - **延迟表现（Latency）**：配置 A 到 E 的总解码延迟差异极小（均在 0.8ms - 0.9ms 之间）。配置 A 的延迟略低于配置 B，而配置 E 的延迟略高于配置 B。这印证了论文核心结论：**Decoding 阶段是 IO-bound（内存带宽受限）**，单纯增加计算单元规模（如更大的 Systolic array）对降低延迟帮助甚微。
  - **面积表现（Area）**：配置 B（基准 GA100）面积最大（约 820 mm²）。配置 A 通过削减计算能力，面积大幅下降至约 500 mm²（约为 B 的 57.8%）。配置 C、D、E 通过调整核心数量与向量宽度，面积控制在 750 mm² - 800 mm² 之间，实现了面积与性能的平衡。
  - **算子分布**：延迟主要由矩阵乘法算子（如 **W1_proj**, **W2_proj**, **Q_K_V**）主导，Softmax、LayerNorm 和 AllReduce 等算子占比较小。
- **配置对比与架构启示**：

| 配置 | 核心特征 | 相对面积 (Area) | 解码延迟 (Latency) | 架构启示 |
|---|---|---|---|---|
| **A** | 1/4 计算能力 | 最小 (~500 mm²) | 略低 (~0.8 ms) | 极致面积优化，适合 IO-bound 场景，无性能损失 |
| **B** | 基准 GA100 | 最大 (~820 mm²) | 基准 (~0.85 ms) | 计算能力过剩，存在面积与良率浪费 |
| **C/D/E** | 大核心/宽向量 | 中等 (~750-800 mm²) | 略高 (~0.88 ms) | 大核心难以完全利用，面积收益递减，调度难度增加 |

- **总结**：该图有力支撑了论文提出 **Latency-Oriented Design** 的合理性。在 Decoding 阶段，由于受限于内存带宽，大幅削减计算单元（如配置 A）不仅能保持几乎相同的延迟，还能显著降低芯片面积（Area）和制造成本。

### 8357eb57ceb20f06aed13992d6370a03a78c4b738e5ebb8b4873b14f5c170a9c.jpg

![8357eb57ceb20f06aed13992d6370a03a78c4b738e5ebb8b4873b14f5c170a9c.jpg](images/8357eb57ceb20f06aed13992d6370a03a78c4b738e5ebb8b4873b14f5c170a9c.jpg)

- **图表主题**：该图（Fig. 8）展示了 **Memory Bandwidth** 对 LLM 推理 **Latency** 的影响，通过堆叠柱状图呈现不同带宽配置下各算子的延迟分布。
- **坐标轴与图例**：
  - **X轴**：**Memory bandwidth (GB/s)**，范围从 400 到 3200，以 400 为步长递增。
  - **Y轴**：**Latency (s)**，数值范围在 0.00 至 0.09 之间。
  - **图例构成**：涵盖 Transformer 架构中的核心算子，包括 **AllReduce**、**GeLU**、**LayerNorm**、**Softmax** 以及各类矩阵乘法（如 **W1_proj**, **W2_proj**, **Q_mul_K** 等），并区分了 **MHA** (Multi-Head Attention) 和 **FFN** (Feed-Forward Network) 模块。
- **数据趋势**：
  - 随着 **Memory bandwidth** 的提升，总 **Latency** 呈现明显的下降趋势。
  - 在低带宽区间（如 400 GB/s 至 800 GB/s），延迟下降幅度最大；在高带宽区间（如 2000 GB/s 至 3200 GB/s），延迟下降趋于平缓，呈现边际收益递减。
- **算子延迟分布与特性**：
| 算子类别 | 具体算子 | 颜色标识 | 特性分析 |
| --- | --- | --- | --- |
| **通信原语** | AllReduce_FFN, AllReduce_MHA | 蓝色系 | 占比极小，受带宽影响有限 |
| **激活与归一化** | GeLU, LayerNorm_FFN/MHA, Softmax | 绿色/棕色系 | 属于 **IO-bound** 操作，对内存带宽高度敏感 |
| **矩阵乘法 (FFN)** | W1_proj, W2_proj | 橙色系 | 计算密集型，高带宽下易转为 **compute-bound** |
| **矩阵乘法 (MHA)** | Wo_proj, A_mul_V, Q_mul_K, Q_K_V | 红色/紫色系 | 核心计算单元，**Decoding** 阶段因矩阵变窄而呈 **IO-bound** |
- **核心结论**：
  - **Prefill 阶段**：**Matmul** 在带宽从 400GB/s 提升至 800GB/s 时显著加速，随后因转为 **compute-bound** 而收益递减；**IO-bound** 算子（如 **GELU**, **LayerNorm**, **Softmax**）则持续从大带宽中获益。
  - **Decoding 阶段**：由于矩阵乘法退化为向量乘矩阵（**narrow**），整体呈 **IO-bound** 特性，因此 **Decoding** 对 **Memory bandwidth** 的敏感度远高于 **Prefill**。
  - **设计启示**：单纯增加计算能力对 **Decoding** 帮助甚微，提升 **Memory bandwidth** 或增大 **Batch size** 才是优化吞吐量的关键。

### fa579dc6f4d10f2d898ffe8c890f89f21d856c1cf9094afa83d841e90647a134.jpg

![fa579dc6f4d10f2d898ffe8c890f89f21d856c1cf9094afa83d841e90647a134.jpg](images/fa579dc6f4d10f2d898ffe8c890f89f21d856c1cf9094afa83d841e90647a134.jpg)

- **图片基本信息**
  - **图表类型**：堆叠柱状图 (Stacked Bar Chart)。
  - **图表主题**：**Decoding Latency (TBT)** per GPT-3 Layer per Output Token（GPT-3 单层每输出 Token 的解码延迟）。
  - **X轴**：**Memory bandwidth (GB/s)**，取值范围从 400 到 3200，步长为 400。
  - **Y轴**：**Latency (ms)**，反映单 Token 解码阶段的总延迟。
  - **图例 (Legend)**：涵盖 Transformer 解码阶段的各类算子，包括矩阵乘法 (**Matmuls**)、归一化与激活 (**Norm/Activation**) 以及通信原语 (**Communication**)。

- **核心数据趋势**
  - **带宽与延迟呈强负相关**：随着 Memory bandwidth 从 400 GB/s 提升至 3200 GB/s，总解码延迟从约 3.5 ms 骤降至约 0.8 ms。
  - **加速比与边际收益**：带宽从 800 GB/s 提升至 2000 GB/s 带来 **1.88x** 的显著加速；而从 2000 GB/s 进一步提升至 3200 GB/s，仅带来约 **26%** 的额外增益，呈现边际收益递减趋势。

- **算子延迟拆解分析**

| 算子类别 (Operators) | 包含算子 (Examples) | 对 Memory Bandwidth 敏感度 | 延迟主导因素 (Dominant Factor) |
| :--- | :--- | :--- | :--- |
| **矩阵乘法 (Matmuls)** | Q_K_V, Q_mul_K, A_mul_V, Wo_proj, W1_proj, W2_proj | **极高** | **内存带宽** (IO-bound) |
| **归一化与激活 (Norm & Activation)** | LayerNorm_FFN, LayerNorm_MHA, Softmax, GeLU | **极低** | **Kernel launch overhead** |
| **通信原语 (Communication)** | AllReduce_FFN, AllReduce_MHA | **中等** | 设备间带宽与延迟 (Device-device interconnect) |

- **架构启示 (Architectural Implications)**
  - **Decoding 阶段的 IO-bound 特性**：在解码阶段，由于 Batch size 较小，矩阵乘法退化为向量-矩阵乘法 (vector-matrix multiplication)，导致计算单元无法被充分喂饱，性能高度依赖 **Memory bandwidth**。
  - **小算子的固定开销瓶颈**：GeLU、LayerNorm 和 Softmax 等算子因输入尺寸较小，其延迟主要由 **kernel launch overhead** 主导。这意味着单纯提升内存带宽无法优化这些算子的延迟，需通过算子融合 (Operator fusion) 等软件层面手段解决。
  - **硬件设计权衡**：针对以 Decoding 为主的推理场景，配置高带宽内存 (如 HBM) 是降低延迟的关键；但硬件架构师需注意带宽提升的**边际收益递减**效应，避免过度设计导致成本浪费。

### 3345b7690bdace42259e0d18ab6f95e7053e3be56cfc210bb08e4f34f0d23da2.jpg

![3345b7690bdace42259e0d18ab6f95e7053e3be56cfc210bb08e4f34f0d23da2.jpg](images/3345b7690bdace42259e0d18ab6f95e7053e3be56cfc210bb08e4f34f0d23da2.jpg)

- **图表核心主题**：评估 **Local buffer size** 对 **Prefill Latency (TTFT)** 及硬件 **Area** 的影响，揭示 **NVIDIA A100** 的设计权衡。
- **坐标轴与图例**：X轴为 **Local buffer size (KB)**（64至1024）；左Y轴为 **Latency (s)**（堆叠柱状图，包含各类算子延迟）；右Y轴为 **Area (mm²)**（折线图）。
- **性能与面积权衡**：
  - **64KB 至 192KB**：**Prefill Latency** 显著降低，性能提升 **18.0%**，而 **Area** 仅增加 **5.8%**，属于高收益区间。
  - **192KB 至 1024KB**：**Prefill Latency** 仅微降 **0.2%**，但 **Area** 激增 **28.8%**，呈现严重的边际收益递减。
- **算子延迟分布**：**矩阵乘法算子**（如 W2_proj, W1_proj, Wo_proj, Q_mul_K）占据主导延迟。增大 **Local buffer** 能有效减少矩阵乘法的 Tiling 开销，从而降低整体延迟。
- **架构设计启示**：
  - **192KB** 为最优设计点。该容量完美匹配 FP16 精度下 128×128×128 的矩阵乘法及 **Double buffering** 需求。
  - 该容量足以完全喂饱 **16×16 systolic arrays**，解释了 **NVIDIA A100** 采用 192KB Local buffer 的设计逻辑。

| Local buffer size (KB) | Latency 变化 | Area 变化 | 综合评估 |
| :--- | :--- | :--- | :--- |
| **64 -> 192** | 显著下降 | +5.8% | **高收益** (性能提升 18.0%) |
| **192 -> 1024** | 趋于平缓 | +28.8% | **低收益** (性能仅提升 0.2%) |

### 35e3e952505586325fc463f1218d190d515951a91431eb7a3441ecb19494caa0.jpg

![35e3e952505586325fc463f1218d190d515951a91431eb7a3441ecb19494caa0.jpg](images/35e3e952505586325fc463f1218d190d515951a91431eb7a3441ecb19494caa0.jpg)

- **图表概述**：本图（Figure 9b）量化分析了 **Local buffer size** 对 LLM **Decoding Latency (TBT)** 及硬件 **Area** 的影响。
- **视觉元素解析**：
  - **X轴**：Local buffer size (KB)，涵盖 64 至 1024 的六组配置。
  - **左Y轴与柱状图**：表示 Latency (ms)，堆叠柱状图细分了 Q_K_V、W1_proj、Softmax 等 12 个算子的延迟占比。
  - **右Y轴与折线图**：表示 Area (mm²)，折线直观反映了芯片面积随 buffer 扩大的膨胀情况。
- **关键数据趋势**：
  - **延迟停滞**：无论 Local buffer size 如何翻倍，总 Decoding Latency 始终维持在 1.0 ms 左右，未出现实质性下降。
  - **面积激增**：硬件面积与 buffer size 呈强正相关，从 64KB 的约 800 mm² 飙升至 1024KB 的约 1100 mm²。
- **数据对比表**：

| Local buffer size (KB) | 性能表现 (Decoding Latency) | 硬件代价 (Area) |
| :--- | :--- | :--- |
| **64** | 基准水平 | 约 800 mm² |
| **192** | 无明显改善 | 约 850 mm² |
| **1024** | 仅提升约 **0.5%** | 约 1100 mm² |

- **架构学启示**：
  - **IO-bound 瓶颈**：Decoding 阶段的核心瓶颈在于读取模型参数和 KV cache 的内存带宽（**IO-bound**），而非片上计算或缓存容量。
  - **设计冗余**：在 Decoding 阶段过度配置 Local buffer 属于资源浪费，无法带来有效的性能回报，验证了论文中“**Buffers should be large enough to fully utilize the systolic arrays, but larger buffers do not help decoding**”的设计准则。

### 9d5b6a187d5b300c06bda7d6f8e0bbf16b39c7a2832ef34e53e181a337ee630c.jpg

![9d5b6a187d5b300c06bda7d6f8e0bbf16b39c7a2832ef34e53e181a337ee630c.jpg](images/9d5b6a187d5b300c06bda7d6f8e0bbf16b39c7a2832ef34e53e181a337ee630c.jpg)

* **图表概述**：该热力图展示了 **Latency-Oriented Design** 对比 NVIDIA **GA100** 的端到端性能归一化结果。性能指标为延迟的倒数（数值越高越好），颜色从深紫色（0.800）渐变至黄色（1.000）。实验设置为 batch size 16、4-way tensor parallelism，运行 48 层 GPT-3 模型。
* **数据矩阵**：横轴为 **Output Length**，纵轴为 **Input Length**，具体归一化性能数值如下表所示：

| Input Length \ Output Length | 256 | 512 | 768 | 1024 | 1280 | 1536 | 1792 | 2048 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **256** | 0.95 | 0.97 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| **512** | 0.92 | 0.96 | 0.97 | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 |
| **1024** | 0.87 | 0.92 | 0.94 | 0.96 | 0.97 | 0.97 | 0.97 | 0.98 |
| **2048** | 0.80 | 0.88 | 0.91 | 0.93 | 0.94 | 0.95 | 0.96 | 0.96 |

* **趋势分析**：
  * **极值表现**：性能最低点为 **0.80**（Input 2048, Output 256），性能最高点为 **0.99**（Input 256, Output ≥ 1024）。
  * **输出长度正相关**：在固定输入长度下，随着 **Output Length** 的增加，性能比例稳步上升并趋近于 1.0。
  * **输入长度负相关**：在固定输出长度下，随着 **Input Length** 的增加，性能比例呈现下降趋势。
* **架构启示**：
  * **Prefill 阶段瓶颈**：长输入序列与短输出序列组合时，**prefill** 阶段（compute-bound）占据主导地位。由于该设计裁剪了一半的计算能力，导致此场景下性能下降最为明显（降至 80%）。
  * **Decoding 阶段免疫**：短输入序列与长输出序列组合时，**decoding** 阶段（IO-bound）占据主导地位。由于 **decoding** 主要受限于内存带宽而非算力，裁剪计算能力对其几乎不产生负面影响，性能得以保持在 **GA100** 的 99% 水平。
  * **设计合理性验证**：数据证实了通过裁剪冗余算力来降低芯片面积和成本，同时维持核心 **IO-bound** 推理性能的策略是高度有效的。

### 0ef2325a780837696db7581d1682ead58a60f8a1a588c8fe6ceadabcb0ac2853.jpg

![0ef2325a780837696db7581d1682ead58a60f8a1a588c8fe6ceadabcb0ac2853.jpg](images/0ef2325a780837696db7581d1682ead58a60f8a1a588c8fe6ceadabcb0ac2853.jpg)

* **图表基本信息**
  * **图表类型**：热力图（Heatmap），用于直观展示不同序列长度组合下的归一化 TBT（Time Between Tokens）性能表现。
  * **X轴**：Output Length，取值范围从 256 递增至 2048。
  * **Y轴**：Input Length，取值包括 256、512、1024、2048。
  * **Colorbar**：右侧颜色条表示 Normalized TBT 的数值区间，介于 **1.0020** 至 **1.0030** 之间，色彩由深紫（低值）向黄绿（高值）渐变。

* **数据分布特征**
  * **数值高度集中**：图表内所有网格的数值均极度收敛于 **1.002** 至 **1.003** 的极窄区间内。
  * **无显著波动**：无论 Input Length 还是 Output Length 如何改变，Normalized TBT 均未表现出明显的性能衰减或提升趋势。
  * **具体数据矩阵**：

| Input Length \ Output Length | 256 | 512 | 768 | 1024 | 1280 | 1536 | 1792 | 2048 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2048** | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 |
| **1024** | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.003 | 1.003 |
| **512** | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 |
| **256** | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 | 1.002 |

* **核心结论与架构启示**
  * **Decoding 性能无损**：该图对应论文提出的 **Latency-Oriented Design**（延迟导向设计）。数据确凿地表明，即便将 Compute Capability（计算能力）与 Buffer Size 硬性削减一半，其 Decoding 阶段的 TBT 与基准 **NVIDIA GA100** 相比几乎完全等同（归一化值趋近于 1）。
  * **IO-bound 特性验证**：此现象完美印证了论文的核心洞察，即 LLM 推理的 Decoding 阶段本质上是 **IO-bound**（受限于内存带宽读取模型参数与 KV Cache），而非 Compute-bound。因此，大幅裁剪计算单元对 Decoding 延迟几乎不产生负面拖累。
  * **设计冗余揭示**：图表数据直接揭示了当前主流商用硬件（如 GA100）在处理 LLM Decoding 任务时存在严重的计算资源过剩，为后续提出低成本、高能效的硬件裁剪与重构方案提供了坚实的量化支撑。

### fb0f2464edd6738d964f4856cf176861ff7dc45c39ad3bf77a2b2f918d5c6ef6.jpg

![fb0f2464edd6738d964f4856cf176861ff7dc45c39ad3bf77a2b2f918d5c6ef6.jpg](images/fb0f2464edd6738d964f4856cf176861ff7dc45c39ad3bf77a2b2f918d5c6ef6.jpg)

* **图表基本信息**
  * **图表类型**：水平条形图 (Horizontal Bar Chart)。
  * **图表主题**：评估 **Latency-Oriented Design** 相对于基准 **NVIDIA GA100** 的 **Normalized TTFT** (Time To First Token) 表现。
  * **Y轴参数**：序列长度配置（2048, 1024, 512, 256），代表不同的输入/输出长度组合。
  * **X轴参数**：**Normalized TTFT**，基准值为 1.0。

* **核心数据表现**
  * **延迟增加**：在所有测试的序列长度下，**Normalized TTFT** 均稳定在 **1.8** 左右。
  * **具体倍数**：论文明确指出，削减一半计算能力会导致 TTFT 产生 **1.82x slowdown**。

| 序列长度配置 | Normalized TTFT (约) | 性能变化说明 |
| :--- | :--- | :--- |
| 2048 | ~1.82 | 计算能力减半导致 Prefill 延迟增加 |
| 1024 | ~1.82 | 趋势保持一致 |
| 512 | ~1.82 | 趋势保持一致 |
| 256 | ~1.82 | 趋势保持一致 |

* **架构与性能分析**
  * **Prefill 阶段特性**：**TTFT** 主要由 **Prefill** 阶段决定。该阶段属于 **compute-bound**（计算密集型），因此对计算单元数量高度敏感。削减 50% 的计算能力直接导致了 **1.82x** 的延迟惩罚。
  * **Decoding 阶段特性**：与 TTFT 形成鲜明对比的是，该设计对 **TBT** (Time Between Tokens) 几乎无负面影响。因为 **Decoding** 阶段属于 **IO-bound**（IO密集型），主要受限于内存带宽而非算力。
  * **设计权衡 (Trade-off)**：该 **Latency-Oriented Design** 通过牺牲部分 **Prefill** 性能，成功减少了 **42.1%** 的芯片面积 (die area)。由于实际推理延迟通常由 **Decoding** 主导，该设计在保持整体推理性能（**95.3%**）的同时，大幅降低了硬件成本。

### 8f30742cb4620614ba67918f37cfef8a6ba20df296f72f6075dada0e61a8aa54.jpg

![8f30742cb4620614ba67918f37cfef8a6ba20df296f72f6075dada0e61a8aa54.jpg](images/8f30742cb4620614ba67918f37cfef8a6ba20df296f72f6075dada0e61a8aa54.jpg)

- **图片概述**：该图片为 Figure 12(a)，直观展示了 **Throughput-Oriented Design** 在不同输入与输出序列长度下的 **Throughput** 表现，核心评估指标为 **Tokens/s**。
- **图表结构解析**：
  - **Y轴**：代表 **Input Length**，设定了 256、512、1024、2048 四个梯度。
  - **X轴**：代表 **Output Length**，设定了从 256 到 2048 的八个梯度。
  - **视觉映射**：右侧 Colorbar 标示了吞吐量区间，深紫色对应低吞吐量（约 500），亮黄色对应高吞吐量（2000 以上）。
- **核心数据矩阵**：

| Input Length \ Output Length | 256 | 512 | 768 | 1024 | 1280 | 1536 | 1792 | 2048 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2048** | 444 | 515 | 526 | 519 | 505 | 489 | 472 | 455 |
| **1024** | 872 | 952 | 923 | 872 | 819 | 768 | 722 | 680 |
| **512** | 1568 | 1572 | 1443 | 1310 | 1180 | 1070 | 977 | 898 |
| **256** | 2464 | 2304 | 1985 | 1718 | 1510 | 1329 | 1186 | 1070 |

- **趋势与架构洞察**：
  - **极值分布**：在 **Input Length** 与 **Output Length** 均为 256 的短序列场景下，系统吞吐量达到峰值 **2464 Tokens/s**；而在双 2048 的长序列场景下，吞吐量衰减至最低点 **455 Tokens/s**。
  - **序列长度负相关**：热力图颜色由左上角的亮黄向右下角的深紫渐变，表明随着 **Input Length** 和 **Output Length** 的增加，系统吞吐量呈现**显著的下降趋势**。
  - **底层瓶颈机制**：该数据走势印证了论文的核心发现。在 **Throughput-Oriented Design** 采用大 Batch Size 策略时，虽然减少了模型参数的读取次数，但**KV cache** 的访问开销随之剧增。序列越长，**KV cache** 读取越成为制约 **Throughput** 的核心瓶颈。

### 17c09bc314edd5b43d6b21f5a8158e1d80fcc572a0a7dc61b5161d27cfa2479e.jpg

![17c09bc314edd5b43d6b21f5a8158e1d80fcc572a0a7dc61b5161d27cfa2479e.jpg](images/17c09bc314edd5b43d6b21f5a8158e1d80fcc572a0a7dc61b5161d27cfa2479e.jpg)

* **图表基本信息**
  * **图表类型**：热力图（Heatmap），用于可视化不同序列长度下的性能对比。
  * **坐标轴定义**：横轴（X轴）为 **Output Length**（256至2048），纵轴（Y轴）为 **Input Length**（256至2048）。
  * **数据含义**：单元格内的数值代表提出的 **Throughput-Oriented Design** 的吞吐量相对于 **8-GA100 GPU Node** 的归一化倍数。
  * **颜色映射**：颜色从深紫到亮黄，代表吞吐量提升倍数从低（约1.25）到高（约1.60）的渐变。

* **核心数据矩阵**

| Input Length \ Output Length | 256 | 512 | 768 | 1024 | 1280 | 1536 | 1792 | 2048 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2048** | 1.28 | 1.30 | 1.32 | 1.31 | 1.37 | 1.38 | 1.41 | 1.44 |
| **1024** | 1.32 | 1.33 | 1.36 | 1.40 | 1.40 | 1.45 | 1.48 | 1.47 |
| **512** | 1.30 | 1.36 | 1.41 | 1.47 | 1.50 | 1.54 | 1.52 | 1.58 |
| **256** | 1.23 | 1.36 | 1.46 | 1.52 | 1.57 | 1.60 | 1.63 | 1.61 |

* **趋势与特征分析**
  * **全面性能优势**：在所有测试的输入和输出长度组合下，归一化吞吐量均**大于1.2**，证明该设计在吞吐量上全面优于传统的 **8-GA100 GPU Node**。
  * **Output Length 正向增益**：随着 **Output Length** 的增加，吞吐量提升倍数总体呈**上升趋势**。例如在 Input Length 为 256 时，提升倍数从 1.23 攀升至峰值 **1.63**。
  * **Input Length 负向衰减**：随着 **Input Length** 的增加，吞吐量提升倍数呈现**轻微下降趋势**。例如在 Output Length 为 2048 时，提升倍数从 1.61 降至 1.44。
  * **极值分布**：
    * **最高提升倍数**：**1.63**（出现在 Input Length 256，Output Length 1792）。
    * **最低提升倍数**：**1.23**（出现在 Input Length 256，Output Length 256）。

* **架构设计启示**
  * **Batch Size 红利**：该设计通过用传统 **DRAM** 替换昂贵的 **HBM**，大幅增加了内存容量，从而支持更大的 **Batch Size**。模型参数只需为整个 Batch 读取一次，显著摊薄了内存带宽压力，实现了吞吐量的跃升。
  * **KV Cache 瓶颈**：当 Input Length 和 Output Length 同时增大时，**KV Cache** 的访问开销急剧增加，抵消了部分大 Batch Size 带来的收益，导致吞吐量提升倍数出现边际递减。
  * **成本效益验证**：结合论文上下文，该设计在实现平均 **1.42倍** 吞吐量提升的同时，将硬件成本降低了 **58.3%**，最终实现了 **3.41倍** 的 **Performance/Cost** 提升，是后台数据处理等吞吐量敏感场景的理想选择。

### c30173839d264c12ad2eda7840346b8d574e98e0540a72f82e5778ee22b1cb2e.jpg

![c30173839d264c12ad2eda7840346b8d574e98e0540a72f82e5778ee22b1cb2e.jpg](images/c30173839d264c12ad2eda7840346b8d574e98e0540a72f82e5778ee22b1cb2e.jpg)

* **图表基本信息**
  * **X轴**：**Normalized Latency**（归一化延迟），数值越大代表推理延迟越高。
  * **Y轴**：**Normalized Throughput**（归一化吞吐量），数值越大代表系统吞吐量越高。
  * **颜色映射**：**Normalized TFLOPs**（归一化算力），色阶从深紫到亮黄代表计算能力从低到高。
  * **数据点形态**：通过不同几何形状区分底层内存架构与容量配置。

* **图例与硬件配置参数**
  | 标记符号 | 内存协议类型 | 内存容量 | 内存带宽 |
  | :--- | :--- | :--- | :--- |
  | ● 圆形 | **HBM** | 80GB | 2TB/s |
  | ■ 方形 | **CXL** | 256GB | 1TB/s |
  | ▲ 三角形 | **CXL** | 512GB | 1TB/s |
  | ♦ 菱形 | **CXL** | 1TB | 1TB/s |
  | ✖ 红色叉号 | **Latency-opt Design** | - | - |
  | ★ 红色星号 | **Throughput-opt Design** | - | - |

* **数据分布与硬件权衡 (Trade-off) 分析**
  * **容量与吞吐量的正相关**：数据点整体呈现从左下向右上的分布趋势。随着内存容量从 **HBM 80GB** 跃升至 **CXL 1TB**，**Normalized Throughput** 显著提升，但 **Normalized Latency** 也随之恶化。
  * **算力对延迟的压制**：在同一内存配置簇内，颜色越亮（**TFLOPs** 越高）的数据点越偏向左侧，证明**堆叠计算单元能有效压缩延迟**，但对突破内存带宽瓶颈带来的吞吐量上限作用有限。
  * **带宽瓶颈与边际效应**：**CXL 1TB** 配置（菱形）虽然容量巨大，但受限于 **1TB/s** 的带宽，其吞吐量增长出现明显的边际收益递减，且延迟代价高昂。

* **核心设计点 (Design Space Exploration) 评估**
  * **Latency-opt Design (✖)**：精准落在图表左下角区域。该设计通过**裁剪算力并保留高带宽 HBM**，实现了**极致的低延迟**，完美匹配 Chatbot 等交互式场景对 **Time to First Token (TTFT)** 的严苛需求。
  * **Throughput-opt Design (★)**：位于图表中上部的高吞吐量区间。该设计通过**用大容量低成本 CXL DRAM 替换 HBM**，在**容忍一定延迟增加**的前提下，实现了**吞吐量的最大化**。
  * **帕累托前沿 (Pareto Frontier)**：两个红色标记点均处于数据散点图的**帕累托最优边界**，证明 **LLMCompass** 能够自动过滤无效设计，精准定位特定目标下的最优硬件架构。

* **图表核心结论**
  * 直观揭示了 **LLM Inference** 中**延迟与吞吐量不可兼得**的物理规律。
  * 验证了打破传统“大算力+高带宽 HBM”设计范式的可行性，证明**通过调整内存层级与算力配比**，能够以极低的硬件成本实现特定场景下的性能最优，为 **LLM 的民主化部署**提供了坚实的理论支撑。

