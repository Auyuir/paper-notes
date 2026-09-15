# Group Normalization 通俗讲解

### 0. 整体创新点通俗解读

**痛点直击**

- 深度学习界的“万金油” **Batch Normalization (BN)** 有个致命软肋：极度依赖 **batch size**。
- 在 ImageNet 分类这种显存充裕的任务里，batch size 开到 32 以上，BN 如鱼得水；但在目标检测、视频分类等吃显存的任务中，**batch size** 往往被迫降到 1 或 2。
- 此时 BN 估算的均值和方差极度不准，模型误差直线飙升。这导致研究者在“模型容量”和“batch size”之间被迫妥协，甚至不敢设计更大的模型。

**通俗比方**

- 想象你在给一批画作打分。BN 的做法是“跨画师比较”，把所有画师的画放一起算平均分。如果只有 2 个画师（小 batch），这个平均分毫无意义。
- **Layer Normalization (LN)** 是看单个画师的所有作品算平均分；**Instance Normalization (IN)** 是看单幅画的不同区域算平均分。
- **Group Normalization (GN)** 的思路是：把单个画师的作品按“风格”分组（比如风景组、人物组），只在同风格组内算平均分。这样不管总共有多少画师（batch size 多大），只要组内维度够多，统计量就稳如泰山。

![](images/x2.png) *Figure 2:Normalization methods. Each subplot shows a feature map tensor, with $N$ as the batch axis, $C$ as the channel axis, and $(H,W)$ as the spatial axes. The pixels in blue are normalized by the same mean and variance, computed by aggregating the values of these pixels.*

**关键一招**

- 作者没有去修补 BN 估算不准的数学缺陷，而是直接**在计算维度上做了外科手术**。
- 在特征张量 **(N, C, H, W)** 中，BN 是沿着 **(N, H, W)** 算均值方差。作者巧妙地把 C（通道）维度切分成 G 个组，然后在 **(C/G, H, W)** 这个子空间内计算统计量。
- 这一招直接让计算过程与 N（batch 维度）彻底解耦。无论 batch size 是 32 还是 2，GN 的统计量完全一致。
- 在 **batch size = 2** 的极端情况下，GN 的误差比 BN 低了惊人的 **10.6%**。

| batch size | 32 | 16 | 8 | 4 | 2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BN | 23.6 | 23.7 | 24.8 | 27.3 | 34.7 |
| GN | 24.1 | 24.2 | 24.0 | 24.2 | 24.1 |
| 差值 | +0.5 | +0.5 | -0.8 | -3.1 | -10.6 |

### 1. Group Normalization (GN)

**痛点直击**

- Batch Normalization (BN) 的命门在于**死磕 batch 维度**。
- 在 ImageNet 这种大 batch size（如 32/GPU）下，BN 算出来的 mean 和 variance 很准，模型收敛极好。
- 但在计算机视觉的下游任务（如目标检测、分割、视频分类）中，输入分辨率大，显存吃紧，batch size 往往只有 1 或 2。
- 此时 BN 估算的 batch 统计量**极度失真**，误差飙升。为了不崩盘，大家只能把 BN 冻结（frozen）成线性层，但这又导致 pre-training 和 fine-tuning 阶段不一致，性能受损。

![](images/x1.png) *Figure 1:ImageNet classification error \vsbatch sizes. This is a ResNet-50 model trained in the ImageNet training set using 8 workers (GPUs), evaluated in the validation set.*

---

**通俗比方**

- 假设你要给一个班级的学生成绩定一个“标准化”的及格线。
- **BN 的做法**是看“全班同学”（一个 batch 内的所有样本）的卷子，算出全班的平均分和方差。如果全班有 32 个人，这个平均分很靠谱；但如果全班只有 2 个人（batch size=2），这 2 个人的平均分波动极大，今天可能 90 分，明天可能 30 分，你拿这个去定及格线，模型直接懵了。
- **GN 的做法**是不看全班了，改成**按学科兴趣小组**来算。把所有科目（channels）分成几个小组（比如 32 组），在每个小组内部算平均分和方差。
- 这样一来，不管全班来了几个人（哪怕只有 1 个人），每个兴趣小组内部的统计量都是稳定的，因为它是沿着 channel 和空间维度算的，彻底摆脱了对 batch 人数的依赖。

![](images/x2.png) *Figure 2:Normalization methods. Each subplot shows a feature map tensor, with $N$ as the batch axis, $C$ as the channel axis, and $(H,W)$ as the spatial axes. The pixels in blue are normalized by the same mean and variance, computed by aggregating the values of these pixels.*

---

**关键一招**

- 作者并没有去修补 batch 统计量的估算方法（比如 Batch Renormalization 或者 Sync BN），而是**直接釜底抽薪，切断了与 batch 维度的联系**。
- 在具体的 tensor 操作上，特征图形状是 `(N, C, H, W)`。
- BN 是沿着 `(N, H, W)` 切，对每个 channel 算统计量。
- 作者巧妙地把 `C` 轴切成了 `G` 份，变成沿着 `(H, W)` 以及**同一组内的多个 channels** 来算统计量。
- 这一步替换，让 normalization 的计算域从“跨样本的单通道”变成了“单样本的跨通道组”。不仅彻底摆脱了 batch size 的束缚，还保留了 channel 间的依赖关系（比 Instance Norm 强）。

| batch size | BN 误差 (%) | GN 误差 (%) | 性能差距 |
| :--- | :--- | :--- | :--- |
| 32 | 23.6 | 24.1 | +0.5 |
| 8 | 24.8 | 24.0 | -0.8 |
| 4 | 27.3 | 24.2 | -3.1 |
| 2 | 34.7 | 24.1 | -10.6 |
