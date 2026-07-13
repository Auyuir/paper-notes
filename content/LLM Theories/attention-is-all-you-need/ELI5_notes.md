# Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works. 通俗讲解

### 0. 整体创新点通俗解读

**痛点直击：这篇论文真正想解决的“大难受”**

- 这篇论文的核心问题不是“再把机器翻译 BLEU 提高一点”，而是要解决当时主流 **Seq2Seq** 架构里的一个根本矛盾：
  - **RNN/LSTM/GRU** 很擅长按顺序处理句子，但它们天然是“一步接一步”的。
  - 第 10 个词的状态要等第 9 个词算完，第 100 个词要等前面 99 个词一路传过来。
  - 这会带来两个很难受的问题：
    - **训练慢**：同一句话内部无法充分并行，GPU 很多算力闲着等。
    - **长距离依赖难学**：两个相隔很远的词之间，信息要穿过很多层时间步，路径太长，梯度和语义都容易“走丢”。

- 当时另一条路线是 **CNN/ConvS2S/ByteNet**：
  - CNN 可以并行，比 RNN 快。
  - 但它看长距离关系也不直接。
  - 如果卷积核很小，远处两个词要通过多层卷积慢慢传播。
  - 如果卷积核很大，计算又贵，而且局部归纳偏置未必适合语言中的灵活依赖。

- 所以当时的尴尬是：
  - **RNN**：懂顺序，但太慢，远距离传递太累。
  - **CNN**：能并行，但远距离关系还得绕路。
  - **Attention**：能直接看全句，但通常只是挂在 RNN/CNN 旁边当辅助模块。

- Transformer 的激进之处在于：
  - 它不是“给 RNN 加一个更强的 Attention”。
  - 它是直接说：**RNN 和 CNN 都不要了，整套序列建模只靠 Attention。**

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

---

**通俗比方：从“传话游戏”变成“全员会议”**

- 把一句话想成一排人，每个人代表一个 **Token**。

- **RNN** 像传话游戏：
  - 第一个人把信息传给第二个人，第二个人再传给第三个人。
  - 如果第 1 个词和第 30 个词有关，它们不能直接交流。
  - 信息必须经过中间 28 个人。
  - 中间任何一步表达不清，后面都会受影响。

- **CNN** 像邻桌讨论：
  - 每个人先和附近几个人交流。
  - 多轮之后，远处信息也能传过来。
  - 但仍然需要多轮传播。
  - 距离越远，绕的路越长。

- **Self-Attention** 像全员会议：
  - 每个 Token 一上来就可以“看向”句子里的所有 Token。
  - 它会自己判断：
    - 我现在最该听谁？
    - 哪些词和我关系最大？
    - 哪些词可以忽略？
  - 这就把“长距离依赖”从一条漫长传话链，变成了一次直接查询。

- **Multi-Head Attention** 更像不是开一个会议，而是开多个专题小组：
  - 一个 Head 关注主谓关系。
  - 一个 Head 关注指代关系。
  - 一个 Head 关注短语边界。
  - 一个 Head 关注位置或修饰关系。
  - 最后把这些视角合并起来。
  - 这就是为什么论文里说单个 Attention 会有“平均化”的问题，而 **Multi-Head** 可以让模型在不同子空间里同时看不同关系。

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

---

**关键一招：把“时间递推”替换成“内容寻址”**

- 作者最巧妙的逻辑转换是：
  - 不再让序列信息沿着时间步一步步流动。
  - 而是让每个位置直接向全序列发起一次“查询”。

- 换句话说，原来的流程是：
  - 当前 Token 的表示，主要来自前一个隐藏状态。
  - 信息传递依赖固定的时间顺序。
  - 模型结构本身规定了“你只能从邻近状态慢慢拿信息”。

- Transformer 的流程变成：
  - 当前 Token 生成一个 **Query**。
  - 所有 Token 提供自己的 **Key** 和 **Value**。
  - Query 去和所有 Key 做匹配。
  - 匹配度高的 Token，其 Value 就被更多吸收。
  - 当前 Token 的新表示，就是“从全句中按相关性取回来的信息”。

- 这一步非常关键：
  - 它把序列建模从 **位置驱动** 变成了 **关系驱动**。
  - RNN 问的是：“前一个状态告诉了我什么？”
  - Self-Attention 问的是：“整句话里，谁和我最相关？”

- 这也是标题 **Attention Is All You Need** 的真正含义：
  - 不是说 Attention 是一个好用插件。
  - 而是说 Attention 本身可以成为主干计算机制。
  - 它可以替代原本负责信息流动的 **recurrence** 和 **convolution**。

---

**为什么这一招厉害：它同时解决了速度和依赖路径**

| 层类型                           | 并行能力 |          长距离依赖路径 | 直觉解释              |
| ----------------------------- | ---: | ---------------: | ----------------- |
| **Self-Attention**            |    强 |         **O(1)** | 任意两个 Token 可以直接交流 |
| **Recurrent**                 |    弱 |         **O(n)** | 信息要沿时间步一个个传       |
| **Convolutional**             |    强 | **O(log n)** 或更长 | 远距离关系要多层卷积传播      |
| **Restricted Self-Attention** |    强 |       **O(n/r)** | 为省计算只看局部，但远距离又要绕路 |

- 这张表背后的直觉很简单：
  - **Self-Attention** 的优势不是每一步计算永远最便宜。
  - 它的优势是：
    - 每层可以高度并行。
    - 任意两个位置之间的信息路径极短。
    - 对机器翻译这类中等长度文本任务，非常划算。

- 所以 Transformer 的贡献不是单点 trick，而是一次架构范式转换：
  - 用 **Self-Attention** 负责全局信息交换。
  - 用 **Feed-Forward Network** 负责每个位置上的非线性变换。
  - 用 **Residual Connection + LayerNorm** 保证深层训练稳定。
  - 用 **Positional Encoding** 补上顺序信息。
  - 用 **Encoder-Decoder Attention** 完成源语言和目标语言之间的对齐。

---

**一个更准确的整体 Mental Model**

- Transformer 可以理解成一个反复执行的“读句子—重写表示”的系统：

  - 每一层里，每个 Token 都会问：
    - **我和谁有关？**
    - **我该从谁那里拿信息？**
    - **我应该忽略谁？**

  - Attention 负责回答：
    - 当前词与全句其他词之间的关系。

  - FFN 负责回答：
    - 拿到这些关系信息后，如何更新当前词自身的表示。

  - 多层堆叠后：
    - 低层可能学局部短语关系。
    - 中层可能学依存、指代、修饰。
    - 高层可能学更抽象的语义对齐。

- 论文附录里的可视化正是这个直觉的证据：
  - 某些 Head 会追踪长距离短语依赖。
  - 某些 Head 会做类似指代消解的行为。
  - 某些 Head 会捕捉句法结构。

![](57e3ad00e7c57fe0dc66b468b013f2fcf447ef78a7d2ee01be8b434fe6ef0669.jpg)

![](1678a839f4e6f07663c6d0c31aa1125d433138f2e6617aae82d039d1529967ea.jpg) *Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.*

![](b616696f63c30ad27de5f095b0cac5975742209d1231c70057510010187c1512.jpg)

---

**这篇论文的整体贡献**

- **核心贡献一句话版**：
  - Transformer 把序列建模里的主干计算，从“按时间递推”改成了“全局 Attention 交互”，从而同时获得了更强并行性、更短依赖路径和更好的翻译质量。

- 它真正改变的是模型设计哲学：
  - 以前默认序列模型必须有某种“沿序列传播”的结构。
    - RNN 沿时间传播。
    - CNN 沿局部窗口传播。
  - Transformer 说：
    - 不必让信息慢慢传播。
    - 每个 Token 可以直接从全局取信息。
    - 顺序信息不靠递推结构承载，而是通过 **Positional Encoding** 显式注入。

- 性能结果说明这不是理论上的优雅，而是工程上也有效：
  
| 模型 | EN-DE BLEU | EN-FR BLEU | 训练成本直觉 |
|---|---:|---:|---|
| **ConvS2S Ensemble** | 26.36 | 41.29 | 高 |
| **GNMT + RL Ensemble** | 26.30 | 41.16 | 很高 |
| **Transformer base** | 27.3 | 38.1 | 低 |
| **Transformer big** | **28.4** | **41.8** | 相对更划算 |

- 最有历史意义的一点：
  - 它不是在旧框架里修补一个模块。
  - 它直接给出了一个新的默认架构模板：
    - **Embedding + Positional Encoding**
    - **Multi-Head Self-Attention**
    - **Feed-Forward Network**
    - **Residual + LayerNorm**
    - **Stacked Encoder-Decoder**

- 后来的 **BERT**、**GPT**、**T5**、**ViT** 等模型，本质上都继承了这个范式：
  - 把输入拆成 Token 或 Patch。
  - 让它们通过 Attention 全局交互。
  - 通过多层堆叠形成越来越抽象的表示。

---

**最应该记住的直觉**

- Transformer 的“妙”不在于某个公式复杂，而在于它做了一个非常大胆的替换：
  - 把原来负责序列信息流动的 **RNN/CNN 主干** 拿掉。
  - 用 **Self-Attention** 让任意位置直接建立联系。
  - 再用 **Positional Encoding** 告诉模型顺序。
  - 用 **Multi-Head** 避免单一注意力视角太粗糙。

- 如果用一句最朴素的话说：
  - **RNN 是排队传话，CNN 是邻里扩散，Transformer 是每个词直接开全局检索。**

- 这就是为什么它能成为后来大模型时代的底层架构：
  - 它让模型更适合 GPU 并行。
  - 它让远距离依赖更容易学习。
  - 它把“关系建模”放到了架构中心。
  - 它证明了：对很多序列任务来说，**Attention 本身就足够强大**。

### 1. Scaled Dot-Product Attention

**痛点直击：为什么要除以√d_k**

- **Scaled Dot-Product Attention**要解决的不是“Attention能不能算”，而是**维度变大后，Attention分数会失控**。

- 在普通 **Dot-Product Attention** 里，模型会拿一个 **Query** 和很多 **Key** 做点积：
  - 点积越大，说明 Query 和 Key 越“匹配”。
  - 然后把这些分数丢进 **softmax**，变成对不同 **Value** 的权重。
  - 最后按权重加权求和，得到输出。

- 难受的地方在于：
  - 当 **d_k** 很大时，点积是很多维度乘积的累加。
  - 维度越多，累加出来的数值波动越大。
  - 这些分数一旦变得很大或很小，**softmax**就会变得非常“极端”：
    - 最大的那个位置接近 1。
    - 其他位置接近 0。
    - 梯度变得很小。
    - 训练时模型很难调整注意力分布。

- 这就像模型一开始还没学明白，就被迫做出“极其自信”的选择：
  - 明明只是初步判断，却被 softmax 放大成“非它不可”。
  - 一旦选错，梯度又很弱，模型改起来很慢。
  - 这会让训练变得不稳定，尤其是在 **Key/Query 维度较大**时更明显。

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

---

**通俗比方：这是给softmax前的分数做“温度校准”**

- 可以把 **QK^T** 理解成一组候选答案的“打分”。

- 没有 scaling 时，就像一个面试官给候选人打分：
  - A：800 分
  - B：760 分
  - C：720 分

- 分数本身看起来有差距，但丢给 **softmax** 后会变得非常极端：
  - A 几乎拿走全部概率。
  - B 和 C 几乎没有机会。
  - 模型过早锁死在 A 上。

- 除以 **√d_k** 的作用，就像把分数压回合理区间：
  - A：8.0
  - B：7.6
  - C：7.2

- 排名没有改变，但判断不再“暴躁”：
  - A 仍然更重要。
  - B、C 也还能保留一定梯度。
  - 模型还有空间慢慢学习“到底该关注谁”。

- 更直白地说：
  - **Dot-Product**负责衡量相似度。
  - **softmax**负责把相似度变成注意力分布。
  - **scaling**负责防止相似度数值太大，把 softmax 推进饱和区。

- 这个操作的直觉很像经典机器学习里的**标准化**：
  - 不是改变问题本身。
  - 不是换一个更复杂的模型。
  - 而是让输入到下一步的数值尺度更健康。

---

**关键一招：不是换Attention，而是给打分结果“降温”**

- 作者最巧妙的地方在于：
  - 没有放弃 **Dot-Product Attention**。
  - 没有改用更慢的 **Additive Attention**。
  - 也没有额外加复杂网络去学兼容性函数。
  - 而是在 **softmax之前**，把点积分数除以 **√d_k**。

- 这个逻辑转换很关键：
  - 原流程是：
    - **Query** 和 **Key** 做点积。
    - 直接送入 **softmax**。
    - 得到权重后加权 **Value**。
  - 新流程只是改了一步：
    - **Query** 和 **Key** 做点积。
    - 先除以 **√d_k**。
    - 再送入 **softmax**。
    - 得到权重后加权 **Value**。

- 这一步看似很小，但解决了一个训练中的核心数值问题：
  - 点积大小会随 **d_k** 增长而膨胀。
  - 除以 **√d_k** 后，分数尺度被拉回稳定范围。
  - softmax 不容易过早饱和。
  - 梯度保持更可用。
  - 训练更稳定。

- 作者并没有重造 Attention，而是巧妙地在 **QK^T** 和 **softmax** 中间插了一个**尺度校准器**。

- 它的本质不是“让 Attention 更聪明”，而是：
  - **让 Attention 的打分别太激动。**
  - **让 softmax 还能正常工作。**
  - **让模型在高维空间里依然有可训练的梯度。**

---

**一句话抓住核心**

- **Scaled Dot-Product Attention**就是：  
  - 用 **Dot-Product**高效计算 Query-Key 匹配度；
  - 用 **√d_k**把高维点积的数值压回合理尺度；
  - 避免 **softmax**过早变成近似 one-hot；
  - 从而让 **Attention**既快，又稳定好训。

### 2. Multi-Head Attention

**痛点直击：为什么需要Multi-Head Attention**

- **单个Attention head的问题，不是“看不到”，而是“看得太平均”**
  - 普通Attention会把一个位置对其他位置的关注结果加权平均成一个向量。
  - 这个机制很强，但也有一个天然问题：**一次Attention只能形成一种关注模式**。
  - 如果一句话里同时存在多种关系，比如：
    - 主谓关系
    - 指代关系
    - 修饰关系
    - 长距离依赖
    - 局部短语结构
  - 单个head必须把这些关系揉在一起看，结果容易变成“什么都看一点，但没有一个关系看得特别干净”。

- **真正难受的场景是：语言里不同关系经常同时存在**
  - 例如句子里某个词既要：
    - 找它的语法主语
    - 找它修饰的对象
    - 找远处的依赖词
    - 保留当前位置附近的局部上下文
  - 如果只有一个Attention head，它就像一个人只戴一副眼镜：
    - 既要看近处
    - 又要看远处
    - 既要看语法
    - 又要看语义
  - 最后往往只能折中。

- **Transformer的核心需求不是单纯“注意力更强”，而是“并行地看不同类型的信息”**
  - **Self-Attention**已经解决了长距离依赖的问题：任意两个Token之间一步可达。
  - 但它还需要解决另一个问题：**同一个Token该从哪些“角度”理解上下文？**
  - **Multi-Head Attention**就是为这个问题设计的。

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

---

**通俗比方：不是一个专家开大会，而是一组专家分头审稿**

- 可以把一个Token理解成一篇论文，把上下文里的其他Token理解成参考文献。
- 单头Attention像是：
  - 只请一位专家审稿。
  - 这位专家要同时判断：
    - 方法是否相关
    - 实验是否支持
    - 引文是否对应
    - 逻辑是否连贯
  - 专家当然能做，但判断会混在一起。

- **Multi-Head Attention像是请多个专家并行审稿**
  - 一个head专门看**语法结构**。
  - 一个head专门看**指代关系**。
  - 一个head专门看**长距离依赖**。
  - 一个head专门看**局部搭配**。
  - 一个head专门看**语义相似性**。
  - 每个专家只在自己的视角里做判断，最后把意见汇总。

- 这个比方最关键的地方在于：
  - **不是复制8个一样的Attention**。
  - 而是让每个head先进入一个不同的低维表示空间。
  - 换句话说，每个专家看的不是同一份材料，而是经过不同“滤镜”处理过的材料。

- 更像经典算法里的一个思想：
  - **不要在一个巨大空间里做一次复杂判断**。
  - 而是把问题投影到多个子空间里，各自做一个简单判断，再合并。
  - 这有点像集成学习里的多个弱视角，但这里不是投票，而是**表示拼接与融合**。

---

**关键一招：把“一次全维度注意力”拆成“多次低维度注意力”**

- 作者没有让一个Attention head在完整的**d_model**维度里硬看所有关系。
- 作者做了一个很巧妙的替换：
  - 原来是：
    - **一个大Attention**直接处理完整的Query、Key、Value。
  - 现在变成：
    - 把Query、Key、Value分别用不同的线性投影，送进多个低维子空间。
    - 每个子空间里各自做一次Attention。
    - 多个Attention结果并行产生。
    - 把这些结果拼接起来，再做一次线性变换融合。

- 这一步的直觉是：
  - **不是让模型“多看几遍同一个东西”**。
  - 而是让模型“从不同角度看同一个东西”。

- 论文里的典型配置很有代表性：

| 配置项 | 数值 | 直觉解释 |
|---|---:|---|
| **d_model** | 512 | 每个Token的总表示维度 |
| **head数h** | 8 | 同时开8个观察视角 |
| **每个head的d_k** | 64 | 每个视角只看一个低维子空间 |
| **每个head的d_v** | 64 | 每个视角输出一个低维结果 |
| **拼接后维度** | 8 × 64 = 512 | 回到原来的表示规模 |

- 这个设计非常精妙：
  - 如果直接做8个完整512维Attention，计算量会暴涨。
  - 但作者把每个head降到64维。
  - 所以8个head合起来的计算量，和一个512维单头Attention大致相近。
  - 也就是说，**几乎不多花计算，却换来了多个表示子空间的并行建模能力**。

---

**最该抓住的直觉**

- **Single-Head Attention**像一个总工程师一次性看完整蓝图。
- **Multi-Head Attention**像把蓝图拆给多个专项工程师：
  - 结构工程师看承重关系。
  - 电气工程师看线路关系。
  - 给排水工程师看管道关系。
  - 安全工程师看风险关系。
- 最后总工程师把他们的结论合并，形成更完整的判断。

- 对语言模型来说：
  - 一个head可能学会看**主谓依赖**。
  - 一个head可能学会看**代词指代**。
  - 一个head可能学会看**短语边界**。
  - 一个head可能学会看**远距离搭配**。

- 这也是为什么论文后面的可视化里，不同head会表现出不同功能：
  - 有的head关注“making...more difficult”这种长距离依赖。
  - 有的head关注“its”这样的指代关系。
  - 有的head呈现出类似句法结构的模式。

![](57e3ad00e7c57fe0dc66b468b013f2fcf447ef78a7d2ee01be8b434fe6ef0669.jpg)

---

**一句话点破**

- **Multi-Head Attention的本质不是“多个注意力叠加”，而是“把注意力分配到多个表示子空间里并行工作”**。
- 它解决的不是Attention能不能看远的问题，而是：
  - **同一个Token需要同时按多种关系去理解上下文，单个head太容易把这些关系混成一锅粥。**
- 作者的关键一招是：
  - **把一次全维度Attention，替换成多次低维Attention并行，再把结果拼回去。**
- 这就是它的顿悟点：
  - **用几乎相同的计算预算，换来多个互补的“观察角度”。**

### 3. Encoder-Decoder Stack without Recurrence or Convolution

**痛点直击：为什么要去掉 Recurrence 和 Convolution**

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

- 过去的 **Encoder-Decoder** 最大的问题，不是“不够聪明”，而是**计算流程太别扭**。

- 对 **RNN/LSTM/GRU** 来说，难受点在于：
  - 它必须按时间步一个一个处理 Token。
  - 第 10 个 Token 的表示，依赖第 9 个；第 100 个 Token 的表示，依赖第 99 个。
  - 这意味着训练时很难在一个句子内部充分并行。
  - GPU 很擅长“大矩阵一起算”，但 RNN 偏偏让它“排队办业务”。
  - 序列越长，这种排队越痛苦。

- 对 **Convolutional Seq2Seq** 来说，难受点在于：
  - Convolution 可以并行，但它的感受野是局部的。
  - 一个 Token 想看见很远处的 Token，需要堆很多层卷积，或者用 dilated convolution 绕一下。
  - 这就像传话：
    - 近处的人可以直接听见。
    - 远处的人要经过很多中间人转述。
  - 路径一长，长距离依赖就更难学。

- 机器翻译这种任务里，最难受的场景正是：
  - 源句和目标句中，语义关联可能隔得很远。
  - 主谓、指代、修饰关系不一定相邻。
  - Decoder 生成某个词时，可能需要同时参考源句多个位置。
  - RNN 太慢，CNN 又要绕远路。

- Transformer 的判断很直接：
  - 既然翻译的核心是“这个词该看哪些词”，那为什么还要让信息按时间步流动？
  - 不如让每个 Token **直接和所有 Token 建立联系**。
  - 这就是 **Encoder-Decoder Stack without Recurrence or Convolution** 的出发点。

| Layer 类型 | 信息传递方式 | 并行性 | 长距离依赖路径 |
|---|---|---|---|
| **Recurrent** | 一个 Token 接一个 Token 传递 | 差 | **O(n)** |
| **Convolutional** | 局部窗口逐层扩散 | 好 | **O(log n)** 或更长 |
| **Self-Attention** | 任意两个 Token 直接交互 | 好 | **O(1)** |

---

**通俗比方：从“流水线传话”变成“全员开会”**

- 可以把一句话里的 Token 想成一个会议室里的参会者。

- **RNN** 像排队传话：
  - 第一个人告诉第二个人。
  - 第二个人再告诉第三个人。
  - 信息从头传到尾。
  - 优点是顺序天然存在。
  - 缺点是慢，而且传远了容易变味。

- **CNN** 像邻桌讨论：
  - 每个人只能先和旁边几个人交流。
  - 多讨论几轮后，远处的信息才能传过来。
  - 优点是大家可以同时说话。
  - 缺点是远距离关系需要多层堆叠才能覆盖。

- **Transformer** 像全员视频会议：
  - 每个人都可以直接看向任何一个人。
  - 某个词如果需要理解另一个远处的词，可以一步到位。
  - 不需要等信息一级一级传过来。
  - 也不需要靠卷积层一圈一圈扩散。

- 这个比方里，**Self-Attention** 就是会议中的“目光分配机制”：
  - 每个 Token 都问一句：
    - “我现在要更新自己的表示，最该参考谁？”
  - 它可以看前面的词，也可以看后面的词。
  - 在 Encoder 里，它可以看整个输入句子。
  - 在 Decoder 里，它只能看已经生成的部分，不能偷看未来。

- 这带来的顿悟点是：
  - Transformer 不是在 RNN 上加一个更强的 Attention。
  - 它是把原来负责“传递上下文”的 RNN/CNN 整个拿掉。
  - 然后让 **Attention 本身承担建模上下文的主任务**。

---

**关键一招：把“沿序列传播信息”替换成“按相关性聚合信息”**

- 作者最巧妙的地方在于：
  - 没有继续优化 RNN 的门控。
  - 没有继续加深 CNN 的层数。
  - 而是直接把 Encoder 和 Decoder 里的核心计算单元换掉。

- 原来的流程大致是：
  - Token 进入 Embedding。
  - 通过 RNN 或 CNN 混合上下文信息。
  - Decoder 再通过 Attention 对 Encoder 输出进行对齐。
  - Attention 只是辅助模块，主要骨架仍然是 RNN/CNN。

- Transformer 的流程被扭转成：
  - Token 进入 Embedding。
  - 加上 **Positional Encoding**，补充位置信息。
  - 通过多层 **Self-Attention** 直接混合全局上下文。
  - 再通过 **Position-wise Feed-Forward Network** 做逐位置非线性变换。
  - Encoder 和 Decoder 都重复堆叠这个结构。

- 换句话说：
  - 作者并没有让模型“按顺序读完整句话”。
  - 而是让每个位置一次性看到全句，然后自己决定该吸收哪些位置的信息。
  - 原来是“信息沿着序列流动”。
  - 现在是“信息按照相关性跳转”。

- Encoder 里的关键替换：
  - 原来：
    - 用 RNN/CNN 生成每个输入位置的上下文表示。
  - 现在：
    - 用 **Multi-Head Self-Attention** 让每个输入 Token 直接关注所有输入 Token。
  - 结果：
    - 每个位置的表示从一开始就是全局感知的。

- Decoder 里的关键替换：
  - 原来：
    - 用 RNN 维护已经生成内容的历史状态。
  - 现在：
    - 用 **Masked Self-Attention** 让当前位置只能关注左侧已生成 Token。
  - 结果：
    - 保留自回归生成约束。
    - 同时避免 RNN 式逐步状态传递。

- Encoder-Decoder 之间的连接方式：
  - Decoder 不是只接收一个压缩向量。
  - Decoder 的每一层都可以通过 **Encoder-Decoder Attention** 去读取 Encoder 的所有输出位置。
  - 这相当于生成每个目标词时，都可以重新对源句做一次软检索。

- 这里有一个容易忽略但非常重要的点：
  - 去掉 Recurrence 和 Convolution 后，模型本身不知道顺序。
  - 所以作者必须引入 **Positional Encoding**。
  - 它的作用不是建模语义，而是告诉模型：
    - “这个 Token 在第几个位置。”
    - “两个 Token 相隔多远。”
  - 也就是说，Transformer 把“顺序建模”和“内容交互”拆开了：
    - **Positional Encoding** 负责顺序。
    - **Self-Attention** 负责关系。
    - **Feed-Forward Network** 负责特征变换。

- 这就是这项设计的本质：
  - RNN 把顺序、记忆、上下文混在一个时间递推里。
  - CNN 把上下文压进局部窗口和层级扩张里。
  - Transformer 把问题拆成更清楚的三件事：
    - 用位置编码告诉模型顺序。
    - 用 Attention 直接建模任意 Token 间关系。
    - 用 FFN 对每个位置做表达能力增强。

- 直觉上可以这样记：
  - **RNN** 是“按顺序读”。
  - **CNN** 是“按邻域扫”。
  - **Transformer** 是“全局查表 + 按需取信息”。

- 所以 **Encoder-Decoder Stack without Recurrence or Convolution** 的真正创新，不是简单地“少用了两个模块”，而是：
  - 把序列建模从**时间递推问题**改造成了**全局信息路由问题**。
  - 把模型瓶颈从“怎么把信息一步步传过去”改成“怎么学会该看谁”。
  - 这一步，正是 Transformer 后来能扩展到大模型的关键基础。

### 4. Masked Decoder Self-Attention

**痛点直击：Decoder最怕“偷看答案”**

- **Masked Decoder Self-Attention**要解决的不是“让Attention更强”，而是防止它**强过头**。

- 在机器翻译这类**auto-regressive generation**里，Decoder生成第i个Token时，理论上只能看见：
  - 已经生成的Token：第1到第i-1个
  - 当前输入侧Encoder提供的信息
  - 不能看见未来Token：第i+1、第i+2……

- 问题在于，普通**Self-Attention**天生是“全局互看”的：
  - 一个位置的Query可以和所有位置的Key做Attention
  - 如果不加限制，Decoder里第i个位置会直接看到后面的真实答案Token
  - 训练时这会造成**信息泄漏**
  - 模型看似学得很好，实际推理时会崩，因为推理阶段未来Token根本不存在

- 这就像考试训练时给学生做填空题：
  - 正常训练：只能看前文，根据上下文预测下一个词
  - 不加mask：答案就在后面印着，学生只是在抄答案
  - 训练分数很高，但真正考试时不会做

- 所以这里的核心痛点是：
  - **训练阶段可以并行看到完整目标句子**
  - **生成逻辑却要求只能从左到右逐步生成**
  - 二者天然冲突
  - **Masked Decoder Self-Attention**就是在保留并行训练效率的同时，强行维护auto-regressive规则

![](da0cb167628b8c102175cfb8905c35ca892193b2792f27c2ecc67f25752338a5.jpg) *Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.*

---

**通俗比方：给Decoder戴上“防作弊眼罩”**

- 你可以把Decoder想象成一排学生，每个学生负责预测一个位置的Token。

- 普通**Self-Attention**像是开卷讨论：
  - 每个学生都能看所有人的草稿
  - 第3个学生可以问第5个学生：“你后面写了什么？”
  - 这在Encoder里没问题，因为输入句子本来就是完整给定的

- 但Decoder不是阅读理解，而是**接龙写作**：
  - 第1个词写完，才能写第2个
  - 第2个词写完，才能写第3个
  - 后面的词在真实生成时还不存在

- **Masked Decoder Self-Attention**就像给每个学生加了一块遮挡板：
  - 第1个位置：只能看自己或起始符号
  - 第2个位置：只能看第1个位置
  - 第3个位置：只能看第1、2个位置
  - 第i个位置：只能看第1到第i个位置，不能看右边

- 这个Mental Model很关键：
  - **Encoder Self-Attention**是“全班自由讨论”
  - **Decoder Self-Attention**是“只能向左看”
  - **Encoder-Decoder Attention**是“Decoder去查输入资料”

- 一句话理解：
  - **Mask不是为了减少计算，而是为了维持生成因果性。**

---

**关键一招：不是放弃并行，而是在Attention矩阵里划掉未来**

- 作者没有回到RNN那种真正一步一步串行训练的老路。

- 巧妙之处在于：
  - 训练时仍然把整个目标序列一次性送进Decoder
  - 仍然用矩阵乘法并行计算Self-Attention
  - 但在Attention权重进入softmax之前，把“看未来”的位置直接屏蔽掉

- 具体逻辑转换是：
  - 原本第i个位置可以Attention到所有位置
  - 现在第i个位置只能Attention到不超过i的位置
  - 右上角那些代表“看未来”的Attention连接被设成不可用
  - softmax之后，这些位置的权重变成0

- 你可以把Decoder Self-Attention的可见性理解成一个下三角矩阵：

| 预测位置 | 可看的Token位置 | 被禁止看的位置 |
|---|---|---|
| 第1个Token | 1 | 2, 3, 4, ... |
| 第2个Token | 1, 2 | 3, 4, 5, ... |
| 第3个Token | 1, 2, 3 | 4, 5, 6, ... |
| 第i个Token | 1到i | i+1之后 |

- 这个设计解决了一个很漂亮的矛盾：
  - **想要Transformer的并行训练效率**
  - **又不能破坏auto-regressive generation的因果约束**
  - Mask让两者同时成立

- 更直白地说：
  - RNN靠时间顺序天然防止偷看未来
  - Transformer没有时间步递归，所以必须人工加规则
  - **Masked Decoder Self-Attention**就是给并行Attention补上的“时间箭头”

- 最值得记住的一句话：
  - **它不是让Decoder看得更多，而是精确规定Decoder哪些地方不能看。**

### 5. Position-wise Feed-Forward Network

**痛点直击：为什么需要Position-wise Feed-Forward Network**

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

- **Self-Attention很擅长“混合信息”，但不擅长“加工信息”**。
  - **Attention**做的核心事情是：让每个Token去看别的Token，把相关位置的信息加权拿过来。
  - 但这个过程本质上更像是**信息路由**或**信息搬运**：谁该看谁、看多少、从哪里拿信息。
  - 它并不特别擅长对每个位置拿到的新表示做复杂的**非线性变换**。

- 如果一层里只有**Multi-Head Attention**，模型会有一个很明显的问题：
  - 每个Token确实能拿到全局上下文；
  - 但拿到之后，只是线性组合出来的表示；
  - 缺少一个“局部思考器”来问：
    - 这个Token现在的语义该怎么重组？
    - 哪些特征该放大？
    - 哪些特征该压掉？
    - 这个位置上的表示是否应该进入更抽象的语义空间？

- 直白说，**Attention解决的是Token之间的沟通问题**，而**Position-wise FFN解决的是每个Token自己的思考问题**。
  - 没有Attention，Token之间彼此隔离。
  - 没有FFN，Token之间虽然沟通了，但每个Token自己的表达能力不够强。

---

**通俗比方：Attention像开会，FFN像会后独立消化**

- 可以把一层Transformer想象成一次团队会议：
  - **Multi-Head Attention**：
    - 每个人，也就是每个Token，去听其他人的意见。
    - 有的人重点听主语，有的人重点听动词，有的人重点听指代词。
    - 这一步是在做**信息交换**。
  - **Position-wise FFN**：
    - 会议结束后，每个人回到自己的座位上，把刚才听到的信息重新整理成自己的判断。
    - 这一步不再互相交流，而是每个Token独立地做一次**内部加工**。

- 所以**Position-wise**这个词很关键：
  - 它不是跨位置操作。
  - 它不会让第3个Token再去看第7个Token。
  - 它只对每个位置自己的向量做同一套变换。

- 更贴近神经网络的比方：
  - **Attention**像是在序列维度上做信息混合。
  - **FFN**像是在特征维度上做非线性变换。
  - 一个管“横向交流”，一个管“纵向加工”。

- 这和CNN里常见的**1×1 Convolution**很像：
  - 不看邻居；
  - 只在当前位置上，对channel进行重新组合；
  - 但所有位置共享同一套参数。
  - Transformer论文里也明确说，Position-wise FFN可以理解成两个kernel size为1的卷积。

---

**关键一招：把“序列建模”和“特征加工”拆开**

- 作者最巧妙的地方，不是设计了一个很复杂的MLP，而是做了一个非常干净的职责划分：
  - **Self-Attention负责跨Token的信息交互**；
  - **Position-wise FFN负责单个Token表示的非线性升级**。

- 原来的RNN或CNN里，信息传播和特征变换常常是混在一起的：
  - RNN在时间步之间传递状态，同时也更新隐藏表示；
  - CNN通过局部窗口聚合邻居，同时也做特征变换。
  - 这会带来一个问题：结构本身限制了并行性，尤其是RNN。

- Transformer这里把流程扭转了：
  - 不再用RNN一步步传状态；
  - 不再靠CNN一层层扩大感受野；
  - 而是先用**Attention**一次性让每个位置看到全局；
  - 再用**Position-wise FFN**对每个位置独立加工。

- 这一步的核心逻辑是：
  - **跨位置关系**交给Attention；
  - **位置内部表达能力**交给FFN；
  - 两者之间用**Residual Connection**和**LayerNorm**稳定训练。

---

**它到底在每个Token上做了什么**

- 对每个位置的向量，FFN做的是一个两层变换：
  - 先把维度从**d_model=512**扩展到**d_ff=2048**；
  - 经过一次**ReLU**，引入非线性；
  - 再把维度从**2048**压回**512**。

| 模块 | 输入维度 | 中间维度 | 输出维度 | 是否跨Token交互 |
|---|---:|---:|---:|---|
| **Position-wise FFN** | 512 | 2048 | 512 | 否 |
| **Multi-Head Attention** | 512 | 多个head | 512 | 是 |

- 为什么要先升维再降维？
  - 升维相当于给每个Token一个更大的“思考空间”。
  - ReLU在高维空间里做选择和过滤。
  - 降维再把结果压回Transformer主干所需的表示维度。

- 这有点像：
  - 你原本只有512个特征槽位；
  - 先临时展开成2048个候选解释；
  - 经过ReLU筛掉不重要的；
  - 再浓缩回512维，交给下一层继续处理。

---

**为什么叫Position-wise，而不是普通FFN**

- **Position-wise**强调两点：
  - 每个Token位置都各自通过同一个FFN；
  - 不同位置之间在这一步完全不通信。

- 举个例子：
  - 句子有10个Token；
  - 每个Token都有一个512维表示；
  - FFN会对这10个512维向量分别处理；
  - 用的是同一组参数；
  - 输出仍然是10个512维向量。

- 所以它保持了序列长度不变：
  - 输入是n个Token；
  - 输出还是n个Token；
  - 改变的是每个Token的内部表示质量。

---

**一句话抓住本质**

- **Position-wise Feed-Forward Network就是Transformer里给每个Token配的“独立思考模块”**。
- **Attention让Token彼此交换信息，FFN让每个Token把交换来的信息消化成更高级的表示**。
- 作者并没有让FFN去建模序列关系，而是巧妙地把它限制在每个位置内部，从而保留了Transformer的高度并行性。

### 6. Sinusoidal Positional Encoding

**痛点直击：Transformer天生是“路痴”**

- **Sinusoidal Positional Encoding**要解决的不是“模型不够大”，而是一个更根本的问题：**纯Attention本身不知道顺序**。
- 在RNN里，顺序是天然存在的：
  - Token一个接一个进来。
  - “我爱你”和“你爱我”经过RNN时，计算路径本身就不同。
  - 时间步就是位置。
- 在CNN里，顺序也有一定结构：
  - 卷积核扫过局部窗口。
  - 相邻Token会先发生局部交互。
  - 位置关系藏在卷积窗口里。
- 但Transformer把这些都拿掉了：
  - 没有recurrence。
  - 没有convolution。
  - Self-Attention看一整句话时，本质上像是在看一个**Token集合**。
- 这就很尴尬：
  - Attention知道“哪些词相关”。
  - 但如果不额外告诉它位置，它不知道“谁在前，谁在后”。
  - 对它来说，某种程度上“猫追狗”和“狗追猫”的Token内容差不多，只是排列不同。
- 所以痛点不是简单的“提高准确率”，而是：
  - **Transformer为了并行化，牺牲了序列结构的天然来源**。
  - 作者必须把“顺序信息”重新塞回模型里，而且不能破坏Transformer的并行优势。

![](f7896a22ff43c1f81531754bb9c3f1e738ea4cf8f64eb0a2e62ca12ec9f973de.jpg) *Figure 1: The Transformer - model architecture.*

---

**通俗比方：给每个Token发一张“坐标身份证”**

- 你可以把Token Embedding理解成一个人的“语义身份证”：
  - “dog”这个词是什么含义。
  - “chased”这个词是什么含义。
  - “cat”这个词是什么含义。
- 但光有语义身份证还不够，因为句子不是一袋词。
- **Positional Encoding**就是给每个Token再发一张“座位号身份证”：
  - 第0个位置有自己的位置向量。
  - 第1个位置有自己的位置向量。
  - 第2个位置有自己的位置向量。
- 然后作者做了一件很朴素但很关键的事：
  - 把**Token Embedding**和**Positional Encoding**直接相加。
  - 得到一个同时包含“这个词是谁”和“这个词坐在哪”的表示。
- 更直观地说：
  - Token Embedding回答：**你是谁？**
  - Positional Encoding回答：**你站在哪里？**
  - 两者相加后，Self-Attention看到的是：**某个位置上的某个词**。
- 为什么用sin和cos？
  - 可以把它想成给每个位置生成一组不同频率的“条形码”。
  - 低频维度像粗粒度坐标：大概在哪一段。
  - 高频维度像细粒度坐标：具体偏移多少。
  - 多个频率叠加后，每个位置都有一个比较独特的指纹。
- 这个设计的妙处在于：
  - 它不是简单写一个整数位置，比如“第17个词”。
  - 它是把位置变成一个和Embedding同维度的连续向量。
  - 这样Attention和FFN都可以自然处理它，不需要额外改模型结构。

---

**关键一招：不是改Attention，而是在输入处“偷偷加坐标”**

- 作者没有去重写Self-Attention机制。
- 作者也没有引入RNN或CNN来恢复顺序。
- 作者的关键一招是：
  - **在进入Encoder和Decoder之前，把位置向量加到Token Embedding上**。
- 这个操作非常轻：
  - 不引入序列依赖。
  - 不影响并行计算。
  - 不改变Transformer主体结构。
- 原来的流程可以理解为：
  - Token → Embedding → Transformer
- 加入Sinusoidal Positional Encoding后变成：
  - Token → Embedding
  - Position → Sinusoidal Positional Encoding
  - 两者相加 → Transformer
- 这一步的本质转换是：
  - 原来Attention只比较“内容与内容”的关系。
  - 现在Attention比较的是“带位置的内容与带位置的内容”的关系。
- 于是模型可以学到：
  - 某个词应该看前一个词。
  - 某个动词应该看远处的主语。
  - Decoder当前位置不能看未来位置。
  - 某些head可以专门关注相对位置模式。
- 这也是为什么论文里强调：
  - 对任意固定偏移，模型有机会从当前位置表示中推断出相对位置关系。
  - 例如“当前位置后面第3个Token”这种关系，不需要每个长度都死记硬背。

---

**为什么不用Learned Positional Embedding就完了？**

| 方案 | 直觉 | 优点 | 隐患 |
|---|---|---|---|
| **Learned Positional Embedding** | 每个位置学一个专属向量 | 简单直接，训练集长度内效果好 | 训练没见过更长位置时，外推能力弱 |
| **Sinusoidal Positional Encoding** | 用固定的sin/cos规律生成位置条形码 | 不需要学习位置参数，理论上可生成任意长度 | 位置表达形式被人为设定，灵活性稍弱 |

- 论文里其实做了对比：
  - Learned版本和Sinusoidal版本效果几乎一样。
- 但作者最后选择**Sinusoidal Positional Encoding**，核心考虑是：
  - **它可能更容易外推到比训练时更长的序列**。
- 这点很关键：
  - Learned Positional Embedding像是背了一张固定座位表。
  - Sinusoidal Positional Encoding像是掌握了一套坐标生成规则。
  - 如果来了第1000个位置，前者可能没见过，后者仍然能算出来。

---

**一句话抓住本质**

- **Sinusoidal Positional Encoding**不是为了让Transformer更复杂，而是为了在不牺牲并行性的前提下，把被Self-Attention抹掉的**顺序感**重新注入进去。
- 它最聪明的地方是：
  - **不改Attention，不加RNN，不加CNN，只在Embedding入口处加一个可泛化的位置坐标。**
- 所以它像是在告诉模型：
  - “你可以继续全局并行地看所有Token，但别忘了，每个Token在句子里都有座位。”
