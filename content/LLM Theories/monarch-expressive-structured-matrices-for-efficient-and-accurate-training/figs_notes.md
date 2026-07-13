# Monarch: Expressive Structured Matrices for Efficient and Accurate Training 图表详解

### Monarch matrices unlock several ways to train sparse and dense models: end-to-end training a sparse (Monarch) model can be 2x faster than dense training thanks to its hardware efficiency; sparse-to-dense “reverse sparsification” can speed up training of large models such as GPT-2; and our dense-to-sparse Monarch projection algorithm can transfer knowledge from pretrained dense model to Monarch model and speed up BERT fine-tuning.

![monarch-main.png](monarch-main.png)

- 图像以 **“Sparse / Dense”表示模型权重表示空间**，以中间灰色流程区表示 Monarch 支持的训练或转换策略。核心信息是：**Monarch matrices 不只是压缩权重的一种结构，还连接了稀疏训练、稀疏到稠密训练、稠密到稀疏微调三类工作流。**

- 左右两侧的竖向面板分别表示流程的**输入模型状态**与**输出模型状态**：
  - 上半部分是 **Sparse**：橙色、白色交错的网格象征结构化的 Monarch matrices。它不是普通的随机非结构化剪枝，而是由两个置换后的 block-diagonal factors 构成的可学习结构。
  - 下半部分是 **Dense**：蓝色满格矩阵象征常规 dense weight matrices。
  - 中间的点状横线是清晰的表示边界：**线以上为结构化/稀疏表示，线以下为稠密表示。**
  - 左下角的 “**Pretrained BERT / Random Init**” 特别说明 Dense 起点可有两种来源：一是随机初始化的新模型，二是已经预训练完成的 Dense BERT。

| 图中编号/路径 | 起点 → 终点 | 图中文字 | 对应论文设置 | 核心作用 |
|---|---|---|---|---|
| **①** | Sparse → Sparse | **Sparse E2E Training** | End-to-End sparse training | 从随机初始化的 Monarch 模型直接端到端训练 |
| **②** | Dense → Sparse | **Reverse Fine-tuning** | Dense-to-Sparse fine-tuning, D2S | 将预训练 Dense 模型投影到 Monarch 空间，再进行下游微调 |
| **③** | Dense → Dense | **Dense E2E Training** | Dense baseline | 标准稠密端到端训练，作为质量和时间基线 |
| 蓝色斜线 | Sparse → Dense | **Sparse-to-Dense** | Reverse sparsification, S2D | 先高效训练 Monarch，再解构为 Dense 权重继续训练 |

- **路径①：Sparse E2E Training**
  - 该路径从左上角 Monarch matrices 出发，水平指向右上角 Monarch matrices，表示模型从初始化到训练结束始终保持 Monarch 结构。
  - 图中橙色强调该路径，视觉上对应论文的主要效率主张：Monarch 通过 block-diagonal matrix multiplication 与 GPU 的 **batched matrix multiply (BMM)** 相匹配，避免普通非结构化稀疏矩阵“FLOPs 少但实际训练慢”的问题。
  - 论文实验支撑：
  
| 任务 | Dense 质量 | Monarch 质量 | 训练加速 | 参数变化 |
|---|---:|---:|---:|---:|
| ViT-B/16, ImageNet | 78.5% | **78.9%** | **2.0×** | 86.6M → 33.0M |
| GPT-2 Medium, WikiText-103 | 20.9 PPL | **20.3 PPL** | **2.0×** | 355M → 165M |
| Mixer-B/16, ImageNet | 77.7% | **77.8%** | 1.9× | 59.9M → 20.9M |

  - 因而，①传达的不是“牺牲精度换速度”，而是 **在相近甚至略优任务指标下，实现约 2× 训练加速**。

- **路径③：Dense E2E Training**
  - 图的底部水平箭头是标准 Dense 模型训练流程：Dense initialization 或预训练权重进入 Dense training，输出仍为 Dense 模型。
  - 它在图中以浅蓝色处理，功能上是**传统基线**：没有结构约束，因此具有完整的 \(O(n^2)\) 参数、存储和矩阵乘法成本。
  - 这条路线的存在凸显 Monarch 的实际定位：论文并不声称 Dense 模型无用；相反，Dense 表示仍是最终模型需要最大容量时的目标形式，尤其适合大规模语言建模后期。

- **Sparse-to-Dense：图中从左上到右下的蓝色斜线**
  - 这一路径对应论文提出的 **“reverse sparsification”**。其语义是：
    - 训练前期：采用硬件高效的 Sparse Monarch matrices；
    - 训练后期：将 Monarch 因子直接相乘、展开为 Dense matrices；
    - 随后：以 Dense 参数继续优化。
  - “Sparse-to-Dense”与传统 pruning 的方向相反。传统 pruning 通常是 Dense → Sparse，用于压缩推理；这里则是 **Sparse → Dense，用于降低大模型预训练的大部分成本**。
  - 图中的蓝色路径从 Sparse 输入下降到 Dense 输出，准确表达“先受结构约束、后释放结构约束”的容量扩展过程。
  - 论文中 GPT-2 使用约 **90% Monarch training + 10% Dense training**：
  
| 模型/数据 | Dense 结果 | Reverse sparsification 结果 | 加速 |
|---|---:|---:|---:|
| GPT-2 Medium, OpenWebText | 18.0 PPL；38.9 avg acc | 18.0 PPL；38.8 avg acc | **2×** |
| BERT-Large pretraining | Nvidia MLPerf 1.1：30.2 h | Monarch：**23.8 h** | **23% faster** |

  - 图中心的 monarch butterfly 图标在此有双重含义：它既代表 Monarch 参数化，也是将“稀疏高效阶段”过渡到“稠密高容量阶段”的中间表示。

- **路径②：Reverse Fine-tuning，即 Dense-to-Sparse**
  - 橙色上行斜线从左下 Dense 区域指向右上 Sparse 区域，表达的是与 S2D 完全相反的转换：**Dense pretrained model → Monarch model**。
  - 图中称其为 **“Reverse Fine-tuning”**，论文正文更精确地称为 **Dense-to-Sparse (D2S) fine-tuning**。
  - 该路径的难点在于：Dense → Sparse 不只是格式转换，而是一个非凸逼近问题。给定 Dense 权重矩阵 \(A\)，需要求解最接近的 Monarch 矩阵：
    \[
    \min_{M\in\mathcal{M}}\|A-M\|_F^2.
    \]
  - Monarch 的关键理论贡献是：尽管参数化为因子乘积而整体问题非凸，作者可将矩阵重排为四维张量，并把目标分解为多个独立的 **rank-1 approximation** 子问题；对每个切片做 SVD 即可得到**全局最优**投影。
  - 因而，图中的该路径不是依赖启发式剪枝或长时间重新训练，而是由 **analytical optimal projection algorithm** 支撑的知识迁移流程。

| BERT 微调结果 | Dense BERT-base | Monarch BERT-base | Dense BERT-large | Monarch BERT-large |
|---|---:|---:|---:|---:|
| GLUE average | 78.6 | 78.3 | 80.4 | 79.6 |
| 参数量 | 109M | **55M** | 335M | **144M** |
| 微调速度 | 1.0× | **1.5×** | 1.0× | **1.7×** |

  - 这说明路径②的价值是：**保留预训练 Dense 模型的知识，同时将下游微调迁移为更快、更小的 Monarch 模型。**

- 图像的颜色编码服务于方法论区分：
  
| 颜色/符号 | 视觉对象 | 语义 |
|---|---|---|
| 橙色 | Sparse E2E、Reverse Fine-tuning、Monarch 非零块 | Monarch 结构化稀疏表示及其相关流程 |
| 蓝色 | Dense E2E、Sparse-to-Dense、Dense 矩阵 | Dense 表示及最终释放结构约束的过程 |
| 灰色中央圆角框 | 方法空间 | Monarch 是连接不同表示与训练范式的桥梁 |
| 蝴蝶图标 | Monarch butterfly | 呼应 Monarch 名称，并强调其源于/涵盖 butterfly-style structured transforms |
| 黑色箭头 | 模型状态迁移 | 指明权重表示与训练流程的方向性 |
| ①②③ | 三条主训练范式 | Sparse E2E、D2S、Dense E2E 的分类索引 |

- 从构图逻辑看，这张图最重要的论点是：**Monarch 不是仅替换 Dense layer 的一种“稀疏层”，而是一个可双向连接 Sparse 与 Dense 模型空间的中间表示。**
  - 向右上：可直接训练高效 Sparse Monarch 模型；
  - 向右下：可先 Sparse 后 Dense，以加速大型 Dense 模型预训练；
  - 向右上逆向迁移：可把 Dense pretrained model 最优投影为 Sparse Monarch model，以加速微调；
  - 底部：保留传统 Dense E2E 作为对照和必要时的最终高容量模型。

- 该图也隐含了论文的完整技术闭环：
  - **硬件效率**支撑 Sparse E2E 与 S2D 前期训练；
  - **表达能力**保证 Monarch 不只是低容量稀疏模式，能够涵盖 butterfly matrices 及 Fourier、convolution、DCT/DST 等快速变换；
  - **最优投影算法**使 Dense → Sparse 可行；
  - **直接展开为 Dense matrix**使 Sparse → Dense 几乎无转换障碍。
  - 因此，图中四条方向并非独立技巧，而是由 Monarch 的 **efficient multiplication、expressiveness、tractable projection** 三项性质共同实现。

### Monarch matrices are parametrized as products of two block-diagonal matrices up to permutation, allowing efficient multiplication algorithm that leverages batch matrix multiply.

![monarch-1.png](monarch-1.png)

- 图中以橙色小方格表示**可学习的非零参数**，白色区域表示固定为零。两个大方阵均呈现清晰的 **block-diagonal** 稀疏模式：沿主对角线分布多个稠密小块，而块间连接被置零。

- 图示对应 Monarch 的核心参数化：
  
  \[
  \mathbf{M}=\mathbf{P}\mathbf{L}\mathbf{P}^{\top}\mathbf{R},
  \]
  
  其中 \(\mathbf{L}\) 与 \(\mathbf{R}\) 都是 block-diagonal matrices，\(\mathbf{P}\) 是固定 permutation matrix，不引入额外可学习参数。

- 从视觉结构看，示例将大矩阵划分为 **4 个沿主对角线排列的 \(4\times4\) 稠密块**。因此，示例整体可理解为 \(n=16\)、\(m=\sqrt n=4\) 的 Monarch layout。

| 图中元素 | 视觉特征 | 数学含义 | 计算作用 |
|---|---|---|---|
| 左侧方阵 | 主对角线上有 4 个橙色稠密块 | 一个 block-diagonal factor，如 \(\mathbf{R}\) | 对输入按连续分组执行独立的小矩阵乘法 |
| 右侧方阵 | 同样有 4 个橙色稠密块 | 另一个 block-diagonal factor，如 \(\mathbf{L}\) | 在 permutation 后的另一维分组上执行独立乘法 |
| 橙色 \(4\times4\) 块 | 块内完全连接 | 每块是独立稠密参数矩阵 | 可直接映射为小型 GEMM / BMM |
| 白色块间区域 | 无连接 | 参数严格为零 | 避免完整 \(n\times n\) dense multiplication |
| \(\mathbf{P},\mathbf{P}^{\top}\) | 固定重排，通常不以参数块显示 | reshape–transpose–reshape 的索引变换 | 改变信息分组方式，使第二次 block multiplication 能跨组混合 |

- 该图想表达的重点不是“单个 block-diagonal matrix 足够表达”，而是：**两个不同坐标系下的 block-diagonal transformations，经 permutation 交错组合后，能够形成远强于单层块对角矩阵的全局信息交互能力。**

- 对输入 \(\mathbf{x}\in\mathbb{R}^{n}\)，其计算可以按如下流程理解：

| 步骤 | 运算 | 张量视角 | 信息混合范围 |
|---|---|---|---|
| 1 | \(\mathbf{y}=\mathbf{R}\mathbf{x}\) | 将 \(\mathbf{x}\) 分为 \(m\) 个长度为 \(m\) 的连续分组 | 每个组内局部混合 |
| 2 | \(\mathbf{y}'=\mathbf{P}^{\top}\mathbf{y}\) | 将长度 \(m^2\) 的向量 reshape 成 \(m\times m\) 矩阵并转置 | 将原本不同组的元素重新编组 |
| 3 | \(\mathbf{z}'=\mathbf{L}\mathbf{y}'\) | 在转置后的分组上执行另一批 block multiplication | 跨原始分组进行混合 |
| 4 | \(\mathbf{z}=\mathbf{P}\mathbf{z}'\) | 恢复原始布局 | 得到最终输出 \(\mathbf{M}\mathbf{x}\) |

- 若将输入重塑为 \(m\times m\) 张量，Monarch 的本质是**沿两个正交维度分别执行 batched dense linear transforms**：
  
  \[
  y_{kj}=\sum_i R_{kji}x_{ki},\qquad
  z_{\ell j}=\sum_k L_{j\ell k}y_{kj}.
  \]
  
  - \(\mathbf{R}\) 先在一个维度上进行局部混合；
  - permutation 等价于二维张量的转置；
  - \(\mathbf{L}\) 再在另一个维度上混合；
  - 因而每个输出位置可依赖于来自多个原始 block 的输入，而不是局限在一个对角块内。

- 图中的结构与普通 block sparsity 的关键差异如下：

| 方法 | 单次连接模式 | 跨 block 信息交互 | 硬件实现特征 |
|---|---|---|---|
| 单层 block-diagonal matrix | 仅块内连接 | **无**，各块彼此隔离 | 高效，但表达力有限 |
| 非结构化 sparse matrix | 任意稀疏连接 | 有，但不规则 | GPU 利用率通常较差 |
| Dense matrix | 全连接 | 完全 | 表达力强，但 \(O(n^2)\) 成本高 |
| **Monarch** | 两个块对角层加 permutation | **有**，通过交错分组实现 | 规则 BMM，兼顾效率和表达力 |

- 参数量与复杂度是该图背后的直接收益。令 \(n=m^2\)：

| 矩阵类型 | 可学习参数量 | 单次 matrix-vector multiplication 复杂度 |
|---|---:|---:|
| Dense \(n\times n\) | \(n^2\) | \(O(n^2)\) |
| 单个 Monarch factor | \(nm=n^{3/2}\) | \(O(n^{3/2})\) |
| **Monarch matrix \(\mathbf{P}\mathbf{L}\mathbf{P}^{\top}\mathbf{R}\)** | **\(2nm=2n^{3/2}\)** | **\(O(n^{3/2})\)** |
| Butterfly matrix | \(O(n\log n)\) | \(O(n\log n)\) |

- 虽然 Butterfly 在渐近 FLOPs 上更低，Monarch 图示强调的是实际 GPU 执行效率：
  
  - **橙色稠密块可批量堆叠**，对应 batch matrix multiply；
  - 不依赖不规则 sparse kernel；
  - permutation 是固定索引操作，无训练参数；
  - block 内计算是高算术强度的 dense GEMM，更符合 GPU / Tensor Core 的硬件偏好；
  - 因此论文报告该结构在训练中可达到接近 **\(2\times\)** 的实际 wall-clock speedup。

- 图所示的两个 block-diagonal factors 也解释了 Monarch 的表达力来源。单独看每个因子都很受限；但其组合满足：
  
  \[
  \mathcal{B}^{(n)}\subset\mathcal{M}^{(n)},
  \]
  
  即 Monarch matrix class 包含 butterfly matrices。进一步，\(\mathcal{M}\mathcal{M}^{*}\) 或其多层乘积可表示 convolution、Toeplitz、Hadamard、Fourier、DCT/DST、ACDC 等多类 fast transforms。

- 对论文后续的 dense-to-sparse projection 而言，该图的二维交错结构尤其关键。将 dense matrix 重排成四维张量后，Monarch 的每个对应 slice 被约束为 rank-1：
  
  \[
  M_{\ell jki}=L_{j\ell k}R_{kji}.
  \]
  
  因而非凸的因子分解问题被分解为多个独立的 rank-1 SVD 问题。这正是 Monarch 能够对 dense pretrained weights 实现**解析最优投影**的结构基础。

- 总结而言，该图浓缩了 Monarch 的设计逻辑：**用两次规则的局部稠密计算替代一次全局 dense multiplication；通过固定 permutation 打破局部 block 的隔离；最终以 GPU-friendly BMM 获得实际加速，同时保留足以覆盖多种快速变换的表达能力。**

### With the “reverse sparsification” process, Monarch matrices can speed up GPT-2 training by 2x.

![monarch-2.png](monarch-2.png)

- 图片以**上下两条训练路径**对比说明：使用 **Monarch reverse sparsification** 的 GPT-2 预训练流程，可在维持最终 Dense 模型形式的前提下，将总训练时间从 **2000 hrs 降至 1000 hrs**，即实现 **2× 加速**。

- 图中核心数值关系如下：

| 训练方案 | 阶段 | 矩阵表示 | 时间 | 总时间 | 相对速度 |
|---|---|---:|---:|---:|---:|
| Conventional dense training | 全程训练 | Dense matrices | 2000 hrs | **2000 hrs** | 1× |
| Monarch reverse sparsification | 前期训练 | Monarch structured matrices | **800 hrs** |  |  |
| Monarch reverse sparsification | 后期训练 | Dense matrices | **200 hrs** | **1000 hrs** | **2×** |

- 上方路径表示论文提出的 **Sparse-to-Dense（S2D）training**：
  - 左侧两个带有局部非零块的矩阵，对应 Monarch 参数化中的结构化因子；可理解为由两个高效的 block-diagonal / permuted block-diagonal 因子组成。
  - 橙色和浅橙色块强调这些阶段采用的是**结构化、低参数且硬件友好**的 Monarch 表示，而不是完整 Dense weight matrix。
  - **800 hrs** 表示大部分训练过程在 Monarch 表示下完成。根据正文，该比例约为总迭代数的 **90%**。
  - 中间的 **“Sparse to dense conversion”** 表示将训练得到的 Monarch factors 显式相乘并展开为 Dense matrix；这是一次直接的表示转换，而不是重新初始化或从头训练。
  - 右侧深蓝色满矩阵代表最终 Dense GPT-2 权重；在转换后继续进行 **200 hrs** 的 Dense training，对应最后约 **10%** 的训练迭代，用于恢复或匹配全程 Dense 训练的最终质量。

- 下方路径是对照组：
  - 左右均为浅蓝/深蓝的满矩阵，表达模型在整个预训练期间始终保持 **Dense representation**。
  - 底部 **2000 hrs** 是达到相同目标质量所需的常规 Dense training 时间。
  - 因而，图的比较对象不是“Monarch-only 模型 vs Dense 模型”，而是：
    - **先用 Monarch 快速完成主要学习，再转为 Dense 完成收敛**；
    - 对比**全程 Dense** 训练。

- 该图隐含的加速计算为：

| 项目 | 数值 |
|---|---:|
| Monarch 阶段耗时 | 800 hrs |
| Dense 收尾阶段耗时 | 200 hrs |
| Reverse sparsification 总耗时 | **1000 hrs** |
| 全程 Dense 耗时 | **2000 hrs** |
| 节省时间 | **1000 hrs** |
| 加速比 | **2000 / 1000 = 2×** |
| 时间降幅 | **50%** |

- 该图的关键技术逻辑是：
  - Monarch matrix 的乘法由块矩阵运算和固定 permutation 构成，可利用 GPU 的 **batched matrix multiplication（BMM）**，因此其实际 wall-clock efficiency 优于不规则 sparse matrix。
  - 前期训练主要学习通用的语言模式、表示和优化方向；这一阶段可由更便宜的 Monarch parameterization 承担。
  - 后期切换到 Dense matrix 后，模型获得完整的参数自由度，从而降低结构化约束可能带来的表达或优化限制。
  - 因此，该方法不是传统意义上“先 Dense、再压缩”的 pruning，而是相反的 **reverse sparsification**：**先结构化/稀疏化训练，后稠密化完成训练**。

- 与论文中的实验结论一致，图旨在支持以下结论：
  - 在 OpenWebText 上，Monarch-GPT-2-medium 的 reverse sparsification 可达到与 Dense GPT-2-medium 相同的上游 perplexity：**18.0 vs. 18.0**。
  - 下游分类平均准确率也几乎一致：**38.8 vs. 38.9**。
  - 因此，图中的 **2× speedup** 不只是 FLOPs 层面的估计，而是以达到相近模型质量为条件的训练时间节省。

- 视觉设计上：
  - **橙色**表示 Monarch / structured training 阶段，突出其是主要的加速来源。
  - **蓝色**表示 Dense matrix 与 Dense training 阶段，强调最终模型仍可回归标准 Dense GPT-2。
  - 上方的“橙色 800 hrs + 蓝色 200 hrs”与下方单独的“蓝色 2000 hrs”形成直接视觉对照，清晰传达“**两阶段 1000 hrs 替代全程 2000 hrs**”这一主张。

### With [alg:project] for our Monarch parameterization, we can convert a pretrained model into a model with Monarch weight matrices and speed up downstream fine-tuning.

![monarch-3.png](monarch-3.png)

- 图片展示论文的第 **3 种使用范式：Dense-to-Sparse (D2S) fine-tuning**。核心信息是：借助 **[alg:project]**，将预训练模型中的 dense weight matrix 投影为 Monarch 结构，再进行下游微调，从而获得实际训练加速。

- 图中从左至右表达的流程可概括为：

| 阶段 | 图形元素 | 含义 | 对应操作 |
|---|---|---|---|
| ① 输入预训练权重 | 左侧蓝色完整 \(4\times4\) 方格矩阵 | 原始的 **dense matrix**；每个位置都可独立取值 | 读取 pretrained model 的 dense linear weights |
| ② Monarch 投影 / 分解 | 中间浅橙色结构方格 | 用 **Monarch parameterization** 逼近 dense 权重 | 运行 [alg:project]，求解最优 Frobenius-norm Monarch approximation |
| ③ 下游适配 | “Fine-tuning” 标注 | 在结构化参数空间继续优化 | 使用下游数据训练 Monarch factors |
| ④ 输出结构化模型 | 右侧深橙色结构方格 | 经过微调后的 Monarch 权重 | 以较少参数、更低计算成本完成任务 |

- 左侧蓝色矩阵强调 dense 权重的两个特点：
  - **完全稠密**：所有 \(16\) 个格子均被填充，表示任意输入—输出维度之间都存在独立连接。
  - **参数与计算成本高**：对 \(n\times n\) dense layer，参数量和矩阵乘法复杂度均为 \(O(n^2)\)。
  - 这是 BERT 等预训练 Transformer 中 attention projection 与 FFN linear layer 的常规形式。

- 中间矩阵不是普通“剪枝掩码”，而是在表达 Monarch 的关键代数结构。Monarch 不是直接保留部分原始元素，而是将矩阵写为：
  \[
  \mathbf{M}=\mathbf{P}\mathbf{L}\mathbf{P}^{\top}\mathbf{R},
  \]
  其中：
  - \(\mathbf{L}\) 与 \(\mathbf{R}\) 是 **block-diagonal matrices**；
  - \(\mathbf{P}\) 是固定的 reshape-transpose permutation；
  - 图中的方格排布和浅色对角位置，意在抽象表示这种**分块、置换、再分块**的计算连接方式；
  - 因此，Monarch 既不是 unstructured sparsity，也不是 low-rank，而是一种**具有可学习块结构的因子化线性变换**。

- 图中最重要但未直接写出的技术步骤是 **projection**。对给定 dense 权重 \(\mathbf A\)，论文求解：
  \[
  \min_{\mathbf M\in\mathcal M}\|\mathbf A-\mathbf M\|_F^2.
  \]
  尽管 \(\mathbf M\) 是两个可学习因子的乘积，整体问题表面上非凸，但论文证明该问题可解析地化为多个独立的 rank-1 approximation 问题。

- [alg:project] 的图外数学含义如下：

| 步骤 | 操作 | 作用 |
|---|---|---|
| 1 | 将 \(\mathbf A\in\mathbb R^{n\times n}\) reshape 为 \(m\times m\times m\times m\)，其中 \(n=m^2\) | 将 Monarch 的双层块结构显式化 |
| 2 | 固定两个索引 \(j,k\)，取一个 \(m\times m\) slice | 得到一个局部矩阵块 |
| 3 | 对每个 slice 作最佳 rank-1 SVD approximation | 独立求出该块对应的最优 Monarch 局部因子 |
| 4 | 将所有 rank-1 左右因子重新组织为 \(\mathbf L,\mathbf R\) | 构成全局 Monarch matrix |
| 5 | 在下游任务上 fine-tune \(\mathbf L,\mathbf R\) | 弥补投影误差并适配任务 |

- 该投影过程的理论价值在于：
  - 对每个局部 slice 使用 SVD，得到的是 **global optimum under the Monarch constraint**，而不是交替优化、启发式 pruning 或无保证的 gradient fitting。
  - 其时间复杂度为 **\(O(n^{5/2})\)**。
  - 这使 dense pretrained model 能够直接转换到 Monarch 模型；相比 butterfly 或一般 sparse-factor products 缺少可处理投影算法的情况，这是论文的关键差异化贡献。

- 右侧的深橙色结构表示：经过 fine-tuning 后，模型仍保持 Monarch 参数化，而不是重新恢复 dense 权重。其实际收益来自：
  - **参数减少**：标准方形 Monarch 由 \(2n\sqrt n\) 个参数描述，低于 dense 的 \(n^2\)；
  - **计算降低**：矩阵乘法主要分解为 block-diagonal multiplication、permutation 和 batched matrix multiplication；
  - **硬件友好**：可调用 GPU 上高效的 **BMM (batch matrix multiply)**，避免常规 unstructured sparsity 的低利用率问题；
  - **可迁移预训练知识**：初始点不是随机 Monarch model，而是 dense pretrained weights 的最优 Monarch 近似。

- 图片中的 “Fine-tuning” 位于中间和右侧之间，强调训练顺序不是“随机初始化 Monarch 后重新预训练”，而是：
  - **Dense pretrained model**
  - \(\rightarrow\) **optimal Monarch projection**
  - \(\rightarrow\) **downstream task fine-tuning**
  - \(\rightarrow\) **fast structured model**

- 这一区别十分关键：

| 方法 | 是否利用 dense 预训练权重 | 结构化初始化质量 | 是否保留 Monarch 结构 | 主要风险 |
|---|---:|---:|---:|---|
| 从头训练 Monarch | 否 | 随机初始化 | 是 | 需要重新学习全部知识 |
| Magnitude pruning | 是 | 基于权重幅度删除连接 | 通常是 unstructured sparse | GPU 训练未必更快 |
| Low-rank SVD | 是 | 全局最优低秩近似 | 否，结构为 low-rank | 表达能力受 rank 限制 |
| 迭代式 butterfly fitting | 是 | 通常为启发式 | 是 | 非凸优化、无全局最优保证 |
| **Monarch projection + fine-tuning** | **是** | **Monarch 约束下全局最优** | **是** | 受 block shape 与结构容量限制 |

- 该图对应的实验结论是 BERT 的 GLUE fine-tuning：
  
| 模型 | GLUE 平均分 | 参数量 | FLOPs | 微调加速 |
|---|---:|---:|---:|---:|
| BERT-base | 78.6 | 109M | 11.2G | — |
| Monarch-BERT-base | 78.3 | 55M | 6.2G | **1.5×** |
| BERT-large | 80.4 | 335M | 39.5G | — |
| Monarch-BERT-large | 79.6 | 144M | 14.6G | **1.7×** |

- 因而，这张图所传达的结论是：**Monarch 不只是用于从零开始训练的高效结构化层，也可以作为 dense pretrained model 的可计算、可证明最优的结构化目标空间。** 通过先投影、后微调，模型以约 **2× 更少参数** 和最高 **1.7× 更快微调速度**，维持与 dense BERT 接近的 GLUE 表现。

### Time required (in A100 GPU hours) to reach the same perplexity (18.0) for GPT-2-small on OpenWebText. With “reverse sparsification”, Monarch can speed up GPT-2 training by 2×.

![rv-bar-temp.png](rv-bar-temp.png)

- 图像采用**堆叠柱状图**，比较 GPT-2-small 在 OpenWebText 上达到相同困惑度 **perplexity = 18.0** 时所需的 **A100 GPU hours**。核心比较对象如下：

| 柱状类别 | 橙色部分 | 蓝色部分 | 训练含义 |
|---|---:|---:|---|
| 左侧柱 | Train w Monarch Weights | Train w Dense Weights | **reverse sparsification**：先用 Monarch，再切换为 Dense |
| 右侧柱 | 无 | Train w Dense Weights | 全程 conventional dense training |

- 图例明确区分两种训练阶段：
  - **橙色：Train w Monarch Weights**，即用 Monarch structured matrices 训练。
  - **蓝色：Train w Dense Weights**，即普通 dense weight matrices 训练。
  - 右侧基线柱完全为蓝色，说明其从训练开始到达到目标 perplexity 都使用 Dense Weights。
  - 左侧柱由橙色和蓝色组成，说明 Monarch 不是最终模型的永久约束，而是用于前期加速，随后恢复为 dense 模型继续训练。

- 图中最重要的结论是：在达到相同质量门槛时，**reverse sparsification 将 GPT-2-small 训练总时间降低约 2×**。
  - 右侧 dense baseline 的柱高约为左侧 reverse-sparsification 柱高的两倍。
  - 因而，该结果不是“更少训练步数”或“更低目标质量”带来的表观收益，而是在**相同 perplexity = 18.0** 条件下的真实时间节省。
  - 若以 dense baseline 的总 GPU-hours 为 \(T\)，则 Monarch reverse sparsification 的时间约为 **\(T/2\)**；节省约 **50% GPU 时间**。

- 左侧堆叠柱表达了该方法的训练日程：
  - 前一大段使用 **Monarch Weights**，对应橙色主体，承担绝大部分优化过程。
  - 最后一小段切回 **Dense Weights**，对应顶部蓝色部分，用于解除结构化参数化约束、恢复 dense 表达能力并完成收敛。
  - 这与论文的机制一致：先利用 Monarch 的 block-diagonal / batched matrix multiply 硬件效率完成大部分预训练，再以 dense 阶段弥补结构化模型可能存在的表示或优化限制。

- 从视觉比例看，左侧柱中橙色部分明显占主导，蓝色收尾阶段只占较小比例。这说明加速并非来自短暂使用 Monarch，而是来自将**大部分训练计算迁移到 Monarch 表示**。
  - 其设计逻辑可概括为：  
    \[
    \text{fast structured pretraining} \rightarrow \text{dense recovery / finishing}
    \]
  - 这不同于传统 pruning：传统 pruning 往往是 dense 模型训练后压缩；Monarch 则直接将结构化计算用于**预训练加速**，最后才“反向稀疏化”回 dense。

- 图的实验控制变量较清晰：

| 项目 | 设定 |
|---|---|
| 模型 | GPT-2-small |
| 数据集 | OpenWebText |
| 质量目标 | perplexity = **18.0** |
| 资源度量 | A100 GPU hours |
| 基线 | 全程 Train w Dense Weights |
| 方法 | Monarch + Dense 的 reverse sparsification |
| 结论 | 达到同等 perplexity 的时间约 **2× 更少** |

- 该图也为论文中更大规模的 GPT-2-medium 实验提供了机制性证据。论文报告 GPT-2-medium 在 OpenWebText 上，Monarch reverse sparsification 达到：
  - **OpenWebText perplexity：18.0，与 Dense GPT-2m 相同**；
  - 分类任务平均准确率：**38.8 vs. 38.9**；
  - 训练速度：**2× speedup**。
  - 因此，这张 GPT-2-small 图不仅展示时间优势，也支持“先结构化、后稠密化”不会导致明显最终质量损失的主张。

- 需要注意的图表限制：
  - 当前图像未显示完整的横轴类别文本、纵轴刻度和绝对 GPU-hour 数值，因此不能仅依据图片可靠读取具体小时数。
  - 不过，图注已经明确给出比较目标和结论：**达到 perplexity 18.0 时，Monarch reverse sparsification 相对 dense training 实现约 2× 加速**。
  - 柱高的像素比例可能受图例遮挡、渲染裁剪或视觉布局影响；解读时应以图注明示的 **2×** 为准，而不应从像素高度反推出精确小时数。

