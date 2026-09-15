# Group Normalization 原文翻译

# Group Normalization

*Yuxin Wu Kaiming He Facebook AI Research (FAIR) {yuxinwu,kaiminghe}@fb.com*

## 摘要

Batch Normalization (BN) 是深度学习发展历程中的一项里程碑式技术，使得各种网络得以训练。然而，沿批次维度进行归一化引入了问题——当批次大小变小时，由于批次统计量估计不准确，BN 的误差会迅速增加。这限制了 BN 在训练更大模型以及将特征迁移到计算机视觉任务（包括检测、分割和视频）中的使用，这些任务由于内存消耗的限制而需要小批次。在本文中，我们提出了 Group Normalization (GN) 作为 BN 的简单替代方案。GN 将通道划分为组，并在每个组内计算用于归一化的均值和方差。GN 的计算独立于批次大小，且其精度在很宽的批次大小范围内保持稳定。在 ImageNet 上训练的 ResNet-50 中，当批次大小为 2 时，GN 的误差比对应的 BN 低 10.6%；在使用典型批次大小时，GN 与 BN 表现相当，且优于其他归一化变体。此外，GN 可以自然地从预训练迁移到微调。在 COCO 中的目标检测和分割任务中，111https://github.com/facebookresearch/Detectron/blob/master/projects/GN. 以及在 Kinetics 中的视频分类任务中，GN 的表现优于其基于 BN 的对应方法，这表明 GN 能够在各种任务中有效替代强大的 BN。在现代库中，GN 可以通过几行代码轻松实现。

---

\iccvfinalcopy

## 1 引言

Batch Normalization (Batch Norm 或 BN) [26] 已被确立为深度学习中非常有效的组件，在很大程度上帮助推动了计算机视觉 [59, 20] 及更广泛领域 [54] 的前沿发展。BN 通过在（小）批次内计算的均值和方差来对特征进行归一化。许多实践表明，这能简化优化过程并使非常深的网络得以收敛。批次统计量的随机不确定性也起到了正则化器的作用，有利于泛化。BN 一直是许多最先进计算机视觉算法的基础。

![Figure 1:ImageNet classification error \vsbatch sizes. This is a ResNet-50 model trained in the ImageNet training set using 8 workers (GPUs), evaluated in the validation set.](images/x1.png)

尽管取得了巨大成功，但 BN 也表现出一些缺点，这些缺点同样是由其沿批次维度进行归一化的独特行为引起的。具体而言，BN 需要在足够大的批次大小下工作（\eg，每个 worker 32 个样本222在本文语境中，我们使用“批次大小”来指代每个 worker（\eg，GPU）的样本数。BN 的统计量是为每个 worker 计算的，而不是跨 worker 广播的，这是许多库中的标准做法。[26, 59, 20]）。小批次会导致批次统计量的估计不准确，而减小 BN 的批次大小会显著增加模型误差（Figure 1）。因此，许多最近的模型 [59, 20, 57, 24, 63] 都使用消耗大量内存的不可忽视的批次大小进行训练。对 BN 训练模型有效性的严重依赖，反过来又限制了人们探索受内存限制的更高容量模型。

在计算机视觉任务中，包括检测 [12, 47, 18]、分割 [38, 18]、视频识别 [60, 6] 以及基于它们构建的其他高级系统，对批次大小的限制更为苛刻。例如，Fast/er 和 Mask R-CNN 框架 [12, 47, 18] 由于分辨率较高，使用 1 或 2 张图像的批次大小，其中 BN 通过转化为线性层而被“冻结” [20]；在使用 3D 卷积的视频分类 [60, 6] 中，时空特征的存在引入了时间长度和批次大小之间的权衡。BN 的使用通常要求这些系统在模型设计和批次大小之间进行妥协。

本文提出了 Group Normalization (GN) 作为 BN 的简单替代方案。我们注意到，许多经典特征如 SIFT [39] 和 HOG [9] 都是分组特征，并涉及分组归一化。例如，一个 HOG 向量是几个空间单元的输出，其中每个单元由一个归一化的方向直方图表示。类似地，我们提出将 GN 作为一种将通道划分为组并在每个组内对特征进行归一化的层（Figure 2）。GN 不利用批次维度，其计算独立于批次大小。

GN 在很宽的批次大小范围内表现非常稳定（Figure 1）。在批次大小为 2 个样本时，对于 ImageNet [50] 中的 ResNet-50 [20]，GN 的误差比对应的 BN 低 10.6%。在常规批次大小下，GN 与 BN 表现相当（差距为 $\scriptstyle\sim$0.5%），并优于其他归一化变体 [3, 61, 51]。此外，尽管批次大小可能发生变化，GN 可以自然地从预训练迁移到微调。在用于 COCO 目标检测和分割 [37] 的 Mask R-CNN 上，以及用于 Kinetics 视频分类 [30] 的 3D 卷积网络上，GN 展示了 \vs 其 BN 对应方法的改进结果。GN 在 ImageNet、COCO 和 Kinetics 中的有效性表明，GN 是在这些任务中一直占主导地位的 BN 的有力竞争替代方案。

已有的一些方法，如 Layer Normalization (LN) [3] 和 Instance Normalization (IN) [61]（Figure 2），也避免了沿批次维度进行归一化。这些方法对于训练序列模型（RNN/LSTM [49, 22]）或生成模型（GANs [15, 27]）是有效的。但正如我们将通过实验展示的，LN 和 IN 在视觉识别方面的成功有限，而 GN 则呈现出更好的结果。反过来，GN 可以替代 LN 和 IN，因此也适用于序列或生成模型。这超出了本文的重点，但对未来的研究具有启示意义。

## 2 相关工作

#### 归一化。

众所周知，对输入数据进行归一化可以加快训练速度 [33]。为了对隐藏特征进行归一化，基于特征分布的强假设推导出了一些初始化方法 [33, 14, 19]，但随着训练的进行，这些假设可能会变得无效。

在 BN 出现之前，深度网络中的归一化层已被广泛使用。Local Response Normalization (LRN) [40, 28, 32] 是 AlexNet [32] 及后续模型 [64, 53, 58] 中的一个组件。与最近的方法 [26, 3, 61] 不同，LRN 在每个像素的小邻域内计算统计量。

Batch Normalization [26] 沿批次维度执行更全局的归一化（同样重要的是，它建议对所有层都这样做）。但“批次”的概念并不总是存在，或者可能会不时发生变化。例如，逐批次的归一化在推理时是不合理的，因此均值和方差是从训练集 [26] 预先计算的，通常通过运行平均值计算；因此，在测试时不执行归一化。当目标数据分布发生变化时，预计算的统计量也可能发生改变 [45]。这些问题导致了在训练、迁移和测试时的不一致性。此外，如前所述，减小批次大小会对估计的批次统计量产生巨大影响。

已经提出了几种归一化方法 [3, 61, 51, 2, 46] 来避免利用批次维度。Layer Normalization (LN) [3] 沿通道维度操作，而 Instance Normalization (IN) [61] 执行类似 BN 的计算，但仅针对每个样本（Figure 2）。Weight Normalization (WN) [51] 不对特征进行操作，而是提出对滤波器权重进行归一化。这些方法不受批次维度引起的问题的影响，但在许多视觉识别任务中，它们一直无法达到 BN 的精度。我们将在后续章节的语境中提供与这些方法的比较。

#### 应对小批量。

Ioffe [25] 提出了 Batch Renormalization (BR)，以缓解 BN 在小批量方面的问题。BR 引入了两个额外的参数，将 BN 估计的均值和方差约束在一定范围内，减少了它们在批量较小时的漂移。在小批量情况下，BR 比 BN 具有更好的精度。但 BR 仍然依赖于批量，当批量减小时，其精度仍会下降 [25]。

也有一些尝试旨在避免使用小批量。[43] 中的目标检测器执行了 synchronized BN，其均值和方差是在多个 GPU 上计算的。然而，这种方法并没有解决小批量的问题；相反，它将算法问题转移到了工程和硬件需求上，使用的 GPU 数量与 BN 的需求成正比。此外，synchronized BN 的计算阻碍了异步求解器（ASGD [10]）的使用，而后者是工业界广泛使用的大规模训练的实用解决方案。这些问题可能会限制 synchronized BN 的使用范围。

不同于解决批量统计计算的方法（\eg, [25, 43]），我们的归一化方法从根本上避免了这种计算。

![Figure 2:Normalization methods. Each subplot shows a feature map tensor, with $N$ as the batch axis, $C$ as the channel axis, and $(H,W)$ as the spatial axes. The pixels in blue are normalized by the same mean and variance, computed by aggregating the values of these pixels.](images/x2.png)

#### 分组计算。

Group convolutions 最早由 AlexNet [32] 提出，用于将模型分配到两个 GPU 上。近年来，将组作为模型设计的一个维度的概念得到了更广泛的研究。ResNeXt [63] 的工作研究了深度、宽度和组之间的权衡，并建议在相似的计算成本下，更多的组数量可以提高精度。MobileNet [23] 和 Xception [7] 利用了 channel-wise（也称为“depth-wise”）卷积，这是组数等于通道数的 group convolutions。ShuffleNet [65] 提出了一种 channel shuffle 操作，用于置换分组特征的轴。这些方法都涉及将通道维度划分为组。尽管与这些方法有关联，GN 并不需要 group convolutions。GN 是一个通用层，正如我们在标准 ResNets [20] 中评估的那样。

## 3 Group Normalization

视觉表示的通道并非完全独立。SIFT [39]、HOG [9] 和 GIST [41] 等经典特征在设计上就是分组表示，其中每组通道由某种直方图构建而成。这些特征通常通过对每个直方图或每个方向进行分组归一化来处理。更高层的特征，如 VLAD [29] 和 Fisher Vectors (FV) [44]，也是分组特征，其中一个组可以被视为相对于某个聚类计算的子向量。

类似地，没有必要将深度神经网络特征视为非结构化向量。例如，对于网络的 conv1（第一个卷积层），可以合理地预期一个滤波器及其水平翻转在自然图像上表现出相似的滤波器响应分布。如果 conv1 恰好近似学习了这对滤波器，或者如果水平翻转（或其他变换）在设计上被引入到架构中 [11, 8]，那么这些滤波器对应的通道就可以一起进行归一化。

更高层的层更加抽象，其行为也不那么直观。然而，除了方向（SIFT [39]、HOG [9] 或 [11, 8]）之外，还有许多因素可能导致分组，\eg, 频率、形状、光照、纹理。它们的系数可能是相互依赖的。事实上，神经科学中一个被广泛接受的计算模型是对细胞响应进行归一化 [21, 52, 55, 5]，“具有各种感受野中心（覆盖视野）以及各种时空频率调谐”（p183, [21]）；这不仅发生在初级视觉皮层，而且发生在“整个视觉系统”中 [5]。受这些工作的启发，我们为深度神经网络提出了新的通用分组归一化方法。

### 3.1 Formulation

我们首先描述特征归一化的一般公式，然后在该公式中介绍 GN。一系列特征归一化方法，包括 BN、LN、IN 和 GN，执行以下计算：

$$
\hat{x}_{i}=\frac{1}{\sigma_{i}}(x_{i}-\mu_{i}).
$$

这里 $x$ 是由一层计算的特征，$i$ 是一个索引。对于 2D 图像，$i=(i_{N},i_{C},i_{H},i_{W})$ 是一个 4D 向量，按 $(N,C,H,W)$ 顺序索引特征，其中 $N$ 是批量轴，$C$ 是通道轴，$H$ 和 $W$ 分别是空间高度和宽度轴。

(1) 式中的 $\mu$ 和 $\sigma$ 是通过以下公式计算的均值和标准差：

$$
\mu_{i}=\frac{1}{m}\sum_{k\in\mathcal{S}_{i}}x_{k},\quad\sigma_{i}=\sqrt{\frac%
{1}{m}\sum_{k\in\mathcal{S}_{i}}(x_{k}-\mu_{i})^{2}+\epsilon},
$$

其中 $\epsilon$ 是一个小常数。$\mathcal{S}_{i}$ 是计算均值和标准差的像素集合，$m$ 是该集合的大小。许多类型的特征归一化方法主要区别在于如何定义集合 $\mathcal{S}_{i}$（Figure 2），讨论如下。

在 Batch Norm [26] 中，集合 $\mathcal{S}_{i}$ 定义为：

$$
\mathcal{S}_{i}=\{k\leavevmode\nobreak\ |\leavevmode\nobreak\ k_{C}=i_{C}\},
$$

其中 $i_{C}$（和 $k_{C}$）表示 $i$（和 $k$）沿 $C$ 轴的子索引。这意味着共享相同通道索引的像素被一起归一化，\ie, 对于每个通道，BN 沿 $(N,H,W)$ 轴计算 $\mu$ 和 $\sigma$。在 Layer Norm [3] 中，集合为：

$$
\mathcal{S}_{i}=\{k\leavevmode\nobreak\ |\leavevmode\nobreak\ k_{N}=i_{N}\},
$$

这意味着 LN 为每个样本沿 $(C,H,W)$ 轴计算 $\mu$ 和 $\sigma$。在 Instance Norm [61] 中，集合为：

$$
\mathcal{S}_{i}=\{k\leavevmode\nobreak\ |\leavevmode\nobreak\ k_{N}=i_{N},k_{C%
}=i_{C}\}.
$$

这意味着 IN 为每个样本和每个通道沿 $(H,W)$ 轴计算 $\mu$ 和 $\sigma$。BN、LN 和 IN 之间的关系如 Figure 2 所示。

如 [26] 所示，BN、LN 和 IN 的所有方法都学习一个逐通道的线性变换，以补偿可能丢失的表示能力：

$$
y_{i}=\gamma\hat{x}_{i}+\beta,
$$

其中 $\gamma$ 和 $\beta$ 是可训练的缩放和平移（在所有情况下均由 $i_{C}$ 索引，为简化符号我们将其省略）。

#### Group Norm.

形式上，Group Norm 层在定义为以下形式的集合 $\mathcal{S}_{i}$ 中计算 $\mu$ 和 $\sigma$：

$$
\mathcal{S}_{i}=\{k\leavevmode\nobreak\ |\leavevmode\nobreak\ k_{N}=i_{N},%
\lfloor\frac{k_{C}}{C/G}\rfloor=\lfloor\frac{i_{C}}{C/G}\rfloor\}.
$$

这里 $G$ 是组的数量，是一个预定义的超参数（默认 $G=32$）。$C/G$ 是每组的通道数。$\lfloor\cdot\rfloor$ 是向下取整操作，“$\lfloor\frac{k_{C}}{C/G}\rfloor=\lfloor\frac{i_{C}}{C/G}\rfloor$”表示索引 $i$ 和 $k$ 在同一组通道中，假设每组通道沿 $C$ 轴按顺序存储。GN 沿 $(H,W)$ 轴以及沿一组 $\frac{C}{G}$ 个通道计算 $\mu$ 和 $\sigma$。GN 的计算如 Figure 2（最右侧）所示，这是一个包含 2 个组（$G=2$）、每组有 3 个通道的简单示例。

给定 Eqn.(7) 中的 $\mathcal{S}_{i}$，GN 层由 Eqn.(1)、(2) 和 (6) 定义。具体来说，同一组中的像素由相同的 $\mu$ 和 $\sigma$ 一起归一化。GN 也学习逐通道的 $\gamma$ 和 $\beta$。

#### 与先前工作的关系。

LN、IN 和 GN 都沿 batch 轴执行独立计算。GN 的两个极端情况分别等价于 LN 和 IN（图 2）。

与 Layer Normalization [3] 的关系。如果我们将组数设为 $G=1$，GN 就变成了 LN。LN 假设一层中的所有通道做出“相似的贡献”[3]。与 [3] 中研究的全连接层情况不同，在存在卷积的情况下，这一假设可能不那么成立，正如 [3] 中所讨论的。GN 的限制比 LN 更少，因为假设每一组通道（而不是所有通道）服从共享的均值和方差；模型仍然具有为每组学习不同分布的灵活性。这导致 GN 比 LN 具有更强的表示能力，正如实验中较低的训练和验证误差所示（图 4）。

与 Instance Normalization [61] 的关系。如果我们将组数设为 $G=C$（\ie，每组一个通道），GN 就变成了 IN。但 IN 只能依靠空间维度来计算均值和方差，错失了利用通道依赖性的机会。

图 3：基于 TensorFlow 的 Group Norm 的 Python 代码。

### 3.2 实现

在支持自动微分的 PyTorch [42] 和 TensorFlow [1] 中，只需几行代码即可轻松实现 GN。图 3 展示了基于 TensorFlow 的代码。实际上，我们只需要指定如何沿由归一化方法定义的适当轴计算均值和方差（“矩”）。

![Figure 4:Comparison of error curves with a batch size of 32 images/GPU. We show the ImageNet training error (left) and validation error (right) \vsnumbers of training epochs. The model is ResNet-50.](images/x3.png)

![Figure 4:Comparison of error curves with a batch size of 32 images/GPU. We show the ImageNet training error (left) and validation error (right) \vsnumbers of training epochs. The model is ResNet-50.](images/x4.png)

![Figure 5:Sensitivity to batch sizes: ResNet-50’s validation error of BN (left) and GN (right) trained with 32, 16, 8, 4, and 2 images/GPU.](images/x5.png)

![Figure 5:Sensitivity to batch sizes: ResNet-50’s validation error of BN (left) and GN (right) trained with 32, 16, 8, 4, and 2 images/GPU.](images/x6.png)

## 4 实验

### 4.1 ImageNet 中的图像分类

我们在包含 1000 个类别的 ImageNet 分类数据集 [50] 上进行实验。我们使用 ResNet 模型 [20] 在 $\scriptstyle\sim$1.28M 张训练图像上进行训练，并在 50,000 张验证图像上进行评估。

#### 实现细节。

作为标准做法 [20, 17]，我们使用 8 个 GPU 来训练所有模型，并且 BN 的 batch 均值和方差在每个 GPU 内部计算。我们使用 [19] 的方法初始化所有模型的所有卷积层。我们使用 1 来初始化所有 $\gamma$ 参数，但每个残差块的最后一个归一化层除外，在该层我们按照 [16] 的做法将 $\gamma$ 初始化为 0（使得残差块的初始状态为恒等映射）。我们对所有权重层使用 0.0001 的权重衰减，包括 $\gamma$ 和 $\beta$（遵循 [17] 但不同于 [20, 16]）。我们为所有模型训练 100 个 epoch，并在 30、60 和 90 个 epoch 时将学习率降低 10$\times$。在训练期间，我们采用 [58] 的数据增强方法，具体实现遵循 [17]。我们在验证集的 224$\times$224 像素中心裁剪图像上评估 top-1 分类误差。为了减少随机变化，我们报告最后 5 个 epoch 的中值误差率 [16]。其他实现细节遵循 [17]。

我们的基线是使用 BN [20] 训练的 ResNet。为了与 LN、IN 和 GN 进行比较，我们用特定的变体替换 BN。我们对所有模型使用相同的超参数。默认情况下，我们将 GN 的 $G$ 设为 32。

#### 特征归一化方法的比较。

我们首先使用常规的 32 张图像（每 GPU）的 batch size 进行实验 [26, 20]。BN 在这种设置下工作成功，因此这是一个用于比较的强基线。图 4 展示了误差曲线，表 1 展示了最终结果。

表 1：在 ImageNet 验证集上，使用 32 张图像/GPU 的 batch size 训练的 ResNet-50 的错误率（%）比较。误差曲线见图 4。

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.25pt 8.0pt;"></th>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">BN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">LN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">IN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">GN</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">验证集误差</span></th>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">23.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">25.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">28.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">24.1</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r" style="padding:0.25pt 8.0pt;">
<math id="S4.T1.m1" class="ltx_Math" alttext="\triangle" display="inline"><mi mathsize="90%" mathvariant="normal">△</mi></math><span class="ltx_text" style="font-size:90%;"> (与</span><span class="ltx_ERROR undefined">\vs</span><span class="ltx_text" style="font-size:90%;">BN相比)</span>
</th>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">-</span></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.25pt 8.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">1.7</em></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.25pt 8.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">4.8</em></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.25pt 8.0pt;"><em class="ltx_emph ltx_font_bold ltx_font_italic" style="font-size:90%;">0.5</em></td>
</tr>
</tbody>
</table>

图 4 显示所有这些归一化方法都能够收敛。与 BN 相比，LN 有 1.7% 的小幅下降。这是一个令人鼓舞的结果，因为它表明对卷积网络的所有通道进行归一化（如 LN 所做的那样）是相当不错的。IN 也能使模型收敛，但比 BN 差 4.8%。333为了完整性，我们还使用 WN [51] 训练了 ResNet-50，这是一种滤波器（而非特征）归一化方法。WN 的结果为 28.2%。

在 BN 工作良好的这种设置下，GN 能够逼近 BN 的精度，在验证集上有 0.5% 的适度下降。实际上，图 4（左）显示 GN 的训练误差低于 BN，这表明 GN 在简化优化方面是有效的。GN 略高的验证集误差意味着 GN 损失了 BN 的部分正则化能力。这是可以理解的，因为 BN 的均值和方差计算引入了由随机 batch 采样引起的不确定性，这有助于正则化 [26]。这种不确定性在 GN（以及 LN/IN）中是不存在的。但是，GN 结合合适的正则化器可能会改善结果。这可以作为一个未来的研究课题。

表 2：对 batch size 的敏感性。我们展示了 ResNet-50 在 ImageNet 上的验证集误差（%）。最后一行显示了 BN 和 GN 之间的差异。误差曲线见图 5。该表在图 1 中进行了可视化。

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">batch size</span></th>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">32</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">16</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">8</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">4</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">2</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">BN</span></th>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">23.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">23.7</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">24.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">27.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">34.7</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">GN</span></th>
<td class="ltx_td ltx_align_center" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">24.1</span></td>
<td class="ltx_td ltx_align_center" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">24.2</span></td>
<td class="ltx_td ltx_align_center" style="padding:0.25pt 8.0pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">24.0</span></td>
<td class="ltx_td ltx_align_center" style="padding:0.25pt 8.0pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">24.2</span></td>
<td class="ltx_td ltx_align_center" style="padding:0.25pt 8.0pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">24.1</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r" style="padding:0.25pt 8.0pt;"><math id="S4.T2.m1" class="ltx_Math" alttext="\triangle" display="inline"><mi mathsize="90%" mathvariant="normal">△</mi></math></th>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.25pt 8.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">0.5</em></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.25pt 8.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">0.5</em></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.25pt 8.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">-0.8</em></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.25pt 8.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">-3.1</em></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.25pt 8.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">-10.6</em></td>
</tr>
</tbody>
</table>

![Figure 6:Evolution of feature distributions of conv5_3's output (before normalization and ReLU) from VGG-16, shown as the {1, 20, 80, 99} percentile of responses. The table on the right shows the ImageNet validation error (%). Models are trained with 32 images/GPU.](images/x7.png)

![Figure 6:Evolution of feature distributions of conv5_3's output (before normalization and ReLU) from VGG-16, shown as the {1, 20, 80, 99} percentile of responses. The table on the right shows the ImageNet validation error (%). Models are trained with 32 images/GPU.](images/x8.png)

![Figure 6:Evolution of feature distributions of conv5_3's output (before normalization and ReLU) from VGG-16, shown as the {1, 20, 80, 99} percentile of responses. The table on the right shows the ImageNet validation error (%). Models are trained with 32 images/GPU.](images/x9.png)

#### 小 batch size。

尽管 BN 在某些情况下受益于随机性，但当 batch size 变小且不确定性增大时，其误差会上升。我们在 Figure 1、Figure 5 和 Table 2 中展示了这一点。

我们评估了每 GPU 32、16、8、4、2 张图像的 batch size。在所有情况下，BN 的均值和方差在每 GPU 内计算，不进行同步。所有模型均在 8 个 GPU 上训练。在这组实验中，我们采用线性学习率缩放规则 [31, 4, 16] 来适应 batch size 的变化——对于 batch size 为 32 的情况，我们使用 0.1 [20] 的学习率，对于 batch size 为 $N$ 的情况使用 0.1$N/$32。如果总 batch size 发生变化（通过改变 GPU 数量）而每 GPU 的 batch size 不变，则该线性缩放规则对 BN 效果良好 [16]。我们在所有情况下保持相同的训练轮数（Figure 5，x 轴）。所有其他超参数保持不变。

Figure 5（左）显示 BN 的误差在小 batch size 下显著增大。GN 的行为更加稳定，对 batch size 不敏感。实际上，Figure 5（右）显示 GN 在从 32 到 2 的广泛 batch size 范围内具有非常相似的曲线（受随机变化影响）。在 batch size 为 2 的情况下，GN 的误差率比其 BN 对应模型低 10.6%（24.1% \vs34.7%）。

这些结果表明，batch 均值和方差估计可能过于随机和不准确，尤其是在仅基于 4 或 2 张图像计算时。然而，如果统计量仅从 1 张图像计算，这种随机性就会消失，此时 BN 在训练时变得类似于 IN。我们看到 IN 的结果（28.4%）优于 batch size 为 2 时的 BN（34.7%）。

Table 2 中 GN 的稳健结果展示了 GN 的优势。它能够消除 BN 所施加的 batch size 约束，从而可以释放大量显存（\eg，16$\times$ 或更多）。这将使得训练更高容量的模型成为可能，而这些模型原本会受到显存限制的瓶颈。我们希望这将在架构设计中创造新的机遇。

Table 3：分组方式。我们展示了 ResNet-50 在 ImageNet 上的验证误差（%），使用 32 张图像/GPU 训练。（上）：给定组数。（下）：每组给定通道数。最后一行显示与最佳数量的差异。

<table class="ltx_tabular ltx_figure_panel ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_align_top ltx_th ltx_th_column" style="padding:0.25pt 2.0pt;" colspan="7">
<span class="ltx_text" style="font-size:90%;">组数 (</span><math id="S4.T3.m1" class="ltx_Math" alttext="G" display="inline"><mi mathsize="90%">G</mi></math><span class="ltx_text" style="font-size:90%;">)</span>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">64</span></span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">32</span></span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">16</span></span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">8</span></span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">4</span></span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_r ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">2</span></span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:32.0pt;"><span class="ltx_text" style="font-size:90%;">1 (=LN)</span></span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">24.6</span></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">24.1</span></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">24.6</span></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">24.4</span></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">24.6</span></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_r ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">24.7</span></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_t" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:32.0pt;"><span class="ltx_text" style="font-size:90%;">25.3</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">0.5</em></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><span class="ltx_text" style="font-size:90%;">-</span></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">0.5</em></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">0.3</em></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">0.5</em></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b ltx_border_r" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:22.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">0.6</em></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b" style="padding:0.25pt 2.0pt;">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p" style="width:32.0pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">1.2</em></span>
</span>
</td>
</tr>
</tbody>
</table>

#### 与 Batch Renorm (BR) 的比较。

BR [25] 引入了两个额外的参数（[25] 中的 $r$ 和 $d$）来约束 BN 的估计均值和方差。它们的值由 $r_{\text{max}}$ 和 $d_{\text{max}}$ 控制。为了将 BR 应用于 ResNet-50，我们仔细选择了这些超参数，并发现 $r_{\text{max}}=1.5$ 和 $d_{\text{max}}=0.5$ 对 ResNet-50 效果最好。在 batch size 为 4 的情况下，使用 BR 训练的 ResNet-50 的错误率为 26.3%。这优于 BN 的 27.3%，但仍比 GN 的 24.2% 高 2.1%。

#### 分组划分。

到目前为止，所有展示的 GN 模型均在组数 $G=32$ 下进行训练。接下来我们评估不同的分组方式。在给定固定组数的情况下，GN 对我们研究的所有 $G$ 值均表现良好（表 3，上表）。在 $G=1$ 的极端情况下，GN 等价于 LN，其错误率高于所研究的所有 $G>1$ 的情况。

我们还评估了固定每组通道数的情况（表 3，下表）。请注意，由于各层可以有不同的通道数，在这种设置下，组数 $G$ 会跨层变化。在每组 1 个通道的极端情况下，GN 等价于 IN。即使每组仅使用少至 2 个通道，GN 的错误率也大幅低于 IN（25.6% \vs28.4%）。这一结果表明了在执行归一化时对通道进行分组的效果。

#### 更深的模型。

我们还在 ResNet-101 [20] 上比较了 GN 与 BN。在 batch size 为 32 时，我们的 ResNet-101 BN 基线验证误差为 22.0%，而对应的 GN 为 22.4%，略差 0.4%。在 batch size 为 2 时，GN ResNet-101 的误差为 23.0%。考虑到非常小的 batch size，这仍然是一个相当稳定的结果，并且比 BN 对应的 31.9% 的误差好 8.9%。

#### VGG 模型的结果与分析。

为了研究 GN/BN 与无归一化的对比，我们考虑了在没有归一化层的情况下也能健康训练的 VGG-16 [56]。我们在每个卷积层之后直接应用 BN 或 GN。图 6 展示了 conv5_3（最后一个卷积层）特征分布的演变。GN 和 BN 在定性行为上相似，而与无归一化的变体存在显著差异；在所有其他卷积层中也观察到了这一现象。这一比较表明，执行归一化对于控制特征分布至关重要。

对于 VGG-16，GN 比 BN 好 0.4%（图 6，右）。这可能意味着 VGG-16 从 BN 的正则化效应中获益较少，在这种情况下，GN（导致更低的训练误差）优于 BN。

### 4.2 COCO 中的目标检测与分割

接下来，我们评估对模型进行微调以迁移到目标检测和分割任务。这些计算机视觉任务通常受益于更高分辨率的输入，因此在常见实践中 batch size 往往很小（1 或 2 张图像/GPU [12, 47, 18, 36]）。结果，BN 变成了一个线性层 $y=\frac{\gamma}{\sigma}(x-\mu)+\beta$，其中 $\mu$ 和 $\sigma$ 是从预训练模型中预先计算并冻结的 [20]。我们将其记为 BN${}^{\text{*}}$，它在微调期间实际上不执行归一化。我们还尝试了微调 BN 的变体（执行归一化且不冻结），发现其效果很差（在 batch size 为 2 时降低了 $\scriptstyle\sim$6 AP），因此我们忽略该变体。

我们在 Mask R-CNN 基线 [18] 上进行实验，该基线在公开可用的 Detectron [13] 代码库中实现。我们使用与 [13] 中超参数相同的端到端变体。我们在微调期间用 GN 替换 BN${}^{\text{*}}$，使用从 ImageNet 预训练的相应模型。444Detectron [13] 使用 [20] 作者提供的预训练模型。为了公平比较，我们改用本文中预训练的模型。这些预训练模型之间的目标检测和分割精度在统计上是相似的。在微调期间，我们对 $\gamma$ 和 $\beta$ 参数使用 0 的权重衰减，这在调整 $\gamma$ 和 $\beta$ 时对于获得良好的检测结果很重要。我们以 1 张图像/GPU 和 8 个 GPU 的 batch size 进行微调。

模型在 COCO train2017 集上训练，并在 COCO val2017 集（即 minival）上评估。我们报告了边界框检测（AP${}^{\text{bbox}}$）和实例分割（AP${}^{\text{mask}}$）的标准 COCO 指标：平均精度（AP）、AP${}_{\text{50}}$ 和 AP${}_{\text{75}}$。

![Figure 7:Error curves in Kinetics with an input length of 32 frames. We show ResNet-50 I3D’s validation error of BN (left) and GN (right) using a batch size of 8 and 4 clips/GPU. The monitored validation error is the 1-clip error under the same data augmentation as the training set, while the final validation accuracy in Table 8 is 10-clip testing without data augmentation.](images/x10.png)

![Figure 7:Error curves in Kinetics with an input length of 32 frames. We show ResNet-50 I3D’s validation error of BN (left) and GN (right) using a batch size of 8 and 4 clips/GPU. The monitored validation error is the 1-clip error under the same data augmentation as the training set, while the final validation accuracy in Table 8 is 10-clip testing without data augmentation.](images/x11.png)

#### C4 骨干网络的结果。

表4展示了在使用 conv4 骨干网络（“C4” [18]）的 Mask R-CNN 上，GN \vsBN${}^{\text{*}}$ 的比较结果。这种 C4 变体使用 ResNet 中直到 conv4 的层来提取特征图，并使用 ResNet 的 conv5 层作为用于分类和回归的感兴趣区域（RoI）头。由于它们继承自预训练模型，骨干网络和头都包含归一化层。

在这个基线上，GN 比 BN${}^{\text{*}}$ 提高了 1.1 个 box AP 和 0.8 个 mask AP。我们注意到，预训练的 GN 模型在 ImageNet 上比 BN 略差（24.1% \vs23.6%），但在微调方面 GN 仍然优于 BN${}^{\text{*}}$。BN${}^{\text{*}}$ 在预训练和微调（冻结）之间造成了不一致性，这可能解释了其性能下降的原因。

我们还对 LN 变体进行了实验，发现它的 box AP 比 GN 差 1.9，比 BN${}^{\text{*}}$ 差 0.8。尽管 LN 也与批量大小无关，但其表示能力弱于 GN。

表4：在 COCO 上使用带有 ResNet-50 C4 的 Mask R-CNN 的检测和分割消融实验结果。BN${}^{\text{*}}$ 表示 BN 被冻结。

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t" style="padding:0.65pt 3.5pt;"><span class="ltx_text" style="font-size:90%;">骨干网络</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.65pt 3.5pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T4.m1" class="ltx_Math" alttext="{}^{\text{bbox}}" display="inline"><msup><mi></mi><mtext mathsize="90%">bbox</mtext></msup></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.65pt 3.5pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T4.m2" class="ltx_Math" alttext="{}^{\text{bbox}}_{\text{50}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">50</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">bbox</mtext></mmultiscripts></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t" style="padding:0.65pt 3.5pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T4.m3" class="ltx_Math" alttext="{}^{\text{bbox}}_{\text{75}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">75</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">bbox</mtext></mmultiscripts></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.65pt 3.5pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T4.m4" class="ltx_Math" alttext="{}^{\text{mask}}" display="inline"><msup><mi></mi><mtext mathsize="90%">mask</mtext></msup></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.65pt 3.5pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T4.m5" class="ltx_Math" alttext="{}^{\text{mask}}_{\text{50}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">50</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">mask</mtext></mmultiscripts></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.65pt 3.5pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T4.m6" class="ltx_Math" alttext="{}^{\text{mask}}_{\text{75}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">75</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">mask</mtext></mmultiscripts></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.65pt 3.5pt;">
<span class="ltx_text" style="font-size:90%;">BN</span><math id="S4.T4.m7" class="ltx_Math" alttext="{}^{\text{*}}" display="inline"><msup><mi></mi><mtext mathsize="90%">*</mtext></msup></math>
</th>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.65pt 3.5pt;"><span class="ltx_text" style="font-size:90%;">37.7</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.65pt 3.5pt;"><span class="ltx_text" style="font-size:90%;">57.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" style="padding:0.65pt 3.5pt;"><span class="ltx_text" style="font-size:90%;">40.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.65pt 3.5pt;"><span class="ltx_text" style="font-size:90%;">32.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.65pt 3.5pt;"><span class="ltx_text" style="font-size:90%;">54.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.65pt 3.5pt;"><span class="ltx_text" style="font-size:90%;">34.7</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r" style="padding:0.65pt 3.5pt;"><span class="ltx_text" style="font-size:90%;">GN</span></th>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.65pt 3.5pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">38.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.65pt 3.5pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">59.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r" style="padding:0.65pt 3.5pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">42.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.65pt 3.5pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">33.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.65pt 3.5pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">55.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.65pt 3.5pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">35.4</span></td>
</tr>
</tbody>
</table>

表5：在 COCO 上使用带有 ResNet-50 FPN 和 4conv1fc 边界框头的 Mask R-CNN 的检测和分割消融实验结果。BN${}^{\text{*}}$ 表示 BN 被冻结。

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">骨干网络</span></th>
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">检测框头</span></th>
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.65pt 1.8pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T5.m1" class="ltx_Math" alttext="{}^{\text{bbox}}" display="inline"><msup><mi></mi><mtext mathsize="90%">bbox</mtext></msup></math>
</th>
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.65pt 1.8pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T5.m2" class="ltx_Math" alttext="{}^{\text{bbox}}_{\text{50}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">50</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">bbox</mtext></mmultiscripts></math>
</th>
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t" style="padding:0.65pt 1.8pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T5.m3" class="ltx_Math" alttext="{}^{\text{bbox}}_{\text{75}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">75</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">bbox</mtext></mmultiscripts></math>
</th>
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.65pt 1.8pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T5.m4" class="ltx_Math" alttext="{}^{\text{mask}}" display="inline"><msup><mi></mi><mtext mathsize="90%">mask</mtext></msup></math>
</th>
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.65pt 1.8pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T5.m5" class="ltx_Math" alttext="{}^{\text{mask}}_{\text{50}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">50</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">mask</mtext></mmultiscripts></math>
</th>
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.65pt 1.8pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T5.m6" class="ltx_Math" alttext="{}^{\text{mask}}_{\text{75}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">75</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">mask</mtext></mmultiscripts></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.65pt 1.8pt;">
<span class="ltx_text" style="font-size:90%;">BN</span><math id="S4.T5.m7" class="ltx_Math" alttext="{}^{\text{*}}" display="inline"><msup><mi></mi><mtext mathsize="90%">*</mtext></msup></math>
</th>
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">-</span></th>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">38.6</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">59.5</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r ltx_border_t" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">41.9</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">34.2</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">56.2</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">36.1</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_row ltx_border_r" style="padding:0.65pt 1.8pt;">
<span class="ltx_text" style="font-size:90%;">BN</span><math id="S4.T5.m8" class="ltx_Math" alttext="{}^{\text{*}}" display="inline"><msup><mi></mi><mtext mathsize="90%">*</mtext></msup></math>
</th>
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_row ltx_border_r" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">GN</span></th>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">39.5</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">60.0</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">43.2</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">34.4</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">56.4</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center" style="padding:0.65pt 1.8pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">36.3</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">GN</span></th>
<th class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r" style="padding:0.65pt 1.8pt;"><span class="ltx_text" style="font-size:90%;">GN</span></th>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_b" style="padding:0.65pt 1.8pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">40.0</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_b" style="padding:0.65pt 1.8pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">61.0</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_b ltx_border_r" style="padding:0.65pt 1.8pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">43.3</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_b" style="padding:0.65pt 1.8pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">34.8</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_b" style="padding:0.65pt 1.8pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">57.3</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_b" style="padding:0.65pt 1.8pt;"><span class="ltx_text ltx_font_bold" style="font-size:90%;">36.3</span></td>
</tr>
</tbody>
</table>

表6：在 COCO 上使用 Mask R-CNN 和 FPN 的检测与分割结果。其中 BN${}^{\text{*}}$ 是 Detectron 的默认基线 [13]，GN 被应用于骨干网络、检测框头和掩码头。"long" 表示使用更多迭代次数进行训练。这些结果的代码可在 https://github.com/facebookresearch/Detectron/blob/master/projects/GN 中找到。

| | $\text{AP}^{\text{bbox}}$ | $\text{AP}^{\text{bbox}}_{\text{50}}$ | $\text{AP}^{\text{bbox}}_{\text{75}}$ | $\text{AP}^{\text{mask}}$ | $\text{AP}^{\text{mask}}_{\text{50}}$ | $\text{AP}^{\text{mask}}_{\text{75}}$ |
|---|---|---|---|---|---|---|
| R50 BN$^{*}$ | 38.6 | 59.8 | 42.1 | 34.5 | 56.4 | 36.3 |
| R50 GN | 40.3 | 61.0 | 44.0 | 35.7 | 57.9 | 37.7 |
| R50 GN, long | **40.8** | **61.6** | **44.4** | **36.1** | **58.5** | **38.2** |
| R101 BN$^{*}$ | 40.9 | 61.9 | 44.8 | 36.4 | 58.5 | 38.7 |
| R101 GN | 41.8 | 62.5 | 45.4 | 36.8 | 59.2 | 39.0 |
| R101 GN, long | **42.3** | **62.8** | **46.2** | **37.2** | **59.7** | **39.5** |

#### FPN 主干网络的结果。

接下来，我们使用特征金字塔网络（FPN）主干 [35]（目前 COCO 中最先进的框架）在 Mask R-CNN 上比较 GN 和 BN${}^{\text{*}}$。与 C4 变体不同，FPN 利用所有预训练层来构建金字塔，并将随机初始化的层作为头部附加其后。在 [35] 中，边界框头部由两个隐藏的全连接层（2fc）组成。我们发现将 2fc 边界框头部替换为 4conv1fc（类似于 [48]）可以更好地利用 GN。比较结果见表 5。

作为基线，使用 4conv1fc 头部的 BN${}^{\text{*}}$ 实现了 38.6 的边界框 AP，与使用相同预训练模型的 2fc 对应版本（38.5 AP）相当。通过将 GN 添加到边界框头部的所有卷积层（但仍使用 BN${}^{\text{*}}$ 主干），我们将边界框 AP 提高了 0.9，达到 39.5（表 5 第 2 行）。该消融实验表明，GN 对检测任务的提升有很大一部分来自于头部的归一化（C4 变体也进行了此操作）。相反，将 BN 应用于边界框头部（每张图像有 512 个 RoIs）未能提供令人满意的结果，并且 AP 下降了 $\scriptstyle\sim$9 —— 在检测任务中，RoIs 的批次是从同一张图像中采样的，其分布不是独立同分布（i.i.d.）的，而这种非独立同分布的分布也是导致 BN 批次统计量估计退化的问题 [25]。GN 不受此问题影响。

接下来，我们将 FPN 主干替换为基于 GN 的对应主干，即，在微调期间使用 GN 预训练模型（表 5 第 3 行）。仅将 GN 应用于主干就带来了 0.5 AP 的提升（从 39.5 到 40.0），这表明 GN 在迁移特征时具有帮助作用。

表 6 展示了 GN（应用于主干、边界框头部和掩码头部）的完整结果，并与基于 BN${}^{\text{*}}$ 的标准 Detectron 基线 [13] 进行了比较。使用与 [13] 相同的超参数，GN 相比 BN${}^{\text{*}}$ 有了大幅度的提升。此外，我们发现 GN 在 [13] 的默认时间表下并未得到充分训练，因此我们还尝试将迭代次数从 180k 增加到 270k（BN${}^{\text{*}}$ 并未从更长时间的训练中受益）。我们最终的 ResNet-50 GN 模型（“long”，表 6）比其 BN${}^{\text{*}}$ 变体在边界框 AP 上好 2.2 个点，在掩码 AP 上好 1.6 个点。

表 7：在 COCO 中使用 Mask R-CNN 和 FPN 从头开始训练的检测和分割结果。这里的 BN 结果来自 [34]，BN 在 GPU 之间进行同步 [43] 且未被冻结。这些结果的代码位于 https://github.com/facebookresearch/Detectron/blob/master/projects/GN。

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t" style="padding:0.9pt 2.2pt;"><em class="ltx_emph ltx_font_italic" style="font-size:90%;">从零开始</em></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.9pt 2.2pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T7.m1" class="ltx_Math" alttext="{}^{\text{bbox}}" display="inline"><msup><mi></mi><mtext mathsize="90%">bbox</mtext></msup></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.9pt 2.2pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T7.m2" class="ltx_Math" alttext="{}^{\text{bbox}}_{\text{50}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">50</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">bbox</mtext></mmultiscripts></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t" style="padding:0.9pt 2.2pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T7.m3" class="ltx_Math" alttext="{}^{\text{bbox}}_{\text{75}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">75</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">bbox</mtext></mmultiscripts></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.9pt 2.2pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T7.m4" class="ltx_Math" alttext="{}^{\text{mask}}" display="inline"><msup><mi></mi><mtext mathsize="90%">mask</mtext></msup></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.9pt 2.2pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T7.m5" class="ltx_Math" alttext="{}^{\text{mask}}_{\text{50}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">50</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">mask</mtext></mmultiscripts></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.9pt 2.2pt;">
<span class="ltx_text" style="font-size:90%;">AP</span><math id="S4.T7.m6" class="ltx_Math" alttext="{}^{\text{mask}}_{\text{75}}" display="inline"><mmultiscripts><mi></mi><mprescripts></mprescripts><mtext mathsize="90%">75</mtext><mrow></mrow><mrow></mrow><mtext mathsize="90%">mask</mtext></mmultiscripts></math>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t" style="padding:0.9pt 2.2pt;">
<span class="ltx_text" style="font-size:90%;">R50 BN </span><cite class="ltx_cite ltx_citemacro_cite"><span class="ltx_text" style="font-size:90%;">[</span><a href="#bib.bib34" title="" class="ltx_ref">34</a><span class="ltx_text" style="font-size:90%;">]</span></cite>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">34.5</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">55.2</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">37.7</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">-</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">-</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">-</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">R50 GN</span></th>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">39.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">59.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">43.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">35.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">56.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">37.6</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b ltx_border_r" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">R101 GN</span></th>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">41.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">61.1</span></td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">44.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">36.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">58.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.9pt 2.2pt;"><span class="ltx_text" style="font-size:90%;">38.7</span></td>
</tr>
</tbody>
</table>

#### 从零开始训练 Mask R-CNN。

GN 使我们能够轻松研究从零开始（无需任何预训练）训练目标检测器。我们在表 7 中展示了结果，其中 GN 模型训练了 27 万次迭代。555对于从零开始训练的模型，我们关闭了 Detectron 中默认的 StopGrad，该操作会冻结前几层。据我们所知，我们的数值（41.0 box AP 和 36.4 mask AP）是迄今为止 COCO 上报道的最佳从零开始结果；它们甚至可以与表 6 中 ImageNet 预训练的结果相媲美。作为参考，借助同步 BN [43]，一项同期工作 [34] 使用 R50 实现了 34.5 box AP 的从零开始结果（表 7），并使用专用骨干网络实现了 36.3 的结果。

### 4.3 Kinetics 中的视频分类

最后，我们在 Kinetics 数据集 [30] 上评估视频分类。许多视频分类模型 [60, 6] 将特征扩展到 3D 时空维度。这非常消耗内存，并对批量大小和模型设计施加了限制。

我们使用 Inflated 3D (I3D) 卷积网络 [6] 进行实验。我们使用 [62] 中描述的 ResNet-50 I3D 基线模型。这些模型在 ImageNet 上进行预训练。对于 BN 和 GN，我们将归一化范围从 $(H,W)$ 扩展到 $(T,H,W)$，其中 $T$ 是时间轴。我们在 400 类 Kinetics 训练集中进行训练，并在验证集中进行评估。我们报告了 top-1 和 top-5 分类准确率，使用标准的 10-clip 测试，即对定期采样的 10 个片段的 softmax 分数取平均值。

我们研究了两种不同的时间长度：32 帧和 64 帧输入片段。32 帧片段从原始视频中以帧间隔为 2 定期采样，而 64 帧片段则是连续采样的。该模型在时空上是全卷积的，因此 64 帧变体消耗的内存大约是 2$\times$。由于内存限制，对于 32 帧变体，我们研究了 8 或 4 clips/GPU 的批量大小，而对于 64 帧变体，我们研究了 4 clips/GPU 的批量大小。

Table 8:Kinetics 中的视频分类结果：ResNet-50 I3D 基线模型的 top-1 / top-5 准确率 (%)。

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">片段长度</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">32</span></th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">32</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">64</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">批量大小</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">8</span></th>
<td class="ltx_td ltx_align_center ltx_border_r" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">4</span></td>
<td class="ltx_td ltx_align_center" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">4</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">BN</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t" style="padding:0.25pt 8.0pt;">
<span class="ltx_text ltx_font_bold" style="font-size:90%;">73.3</span><span class="ltx_text" style="font-size:90%;"> / </span><span class="ltx_text ltx_font_bold" style="font-size:90%;">90.7</span>
</th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">72.1 / 90.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_t" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">73.3 / 90.8</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">GN</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r" style="padding:0.25pt 8.0pt;"><span class="ltx_text" style="font-size:90%;">73.0 / 90.6</span></th>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r" style="padding:0.25pt 8.0pt;">
<span class="ltx_text ltx_font_bold" style="font-size:90%;">72.8</span><span class="ltx_text" style="font-size:90%;"> / </span><span class="ltx_text ltx_font_bold" style="font-size:90%;">90.6</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_b" style="padding:0.25pt 8.0pt;">
<span class="ltx_text ltx_font_bold" style="font-size:90%;">74.5</span><span class="ltx_text" style="font-size:90%;"> / </span><span class="ltx_text ltx_font_bold" style="font-size:90%;">91.7</span>
</td>
</tr>
</tbody>
</table>

#### 32 帧输入的结果。

Table 8 (第 1、2 列) 显示了使用 32 帧片段在 Kinetics 中的视频分类准确率。对于 8 的批量大小，GN 的 top-1 准确率比 BN 略差 0.3%，top-5 略差 0.1%。这表明当 BN 表现良好时，GN 与 BN 具有竞争力。对于 4 的较小批量大小，GN 的准确率保持相似 (72.8 / 90.6 \vs73.0 / 90.6)，但优于 BN 的 72.1 / 90.0。当批量大小从 8 减小到 4 时，BN 的准确率下降了 1.2%。

Figure 7 显示了误差曲线。当批量大小从 8 减小到 4 时，BN 的误差曲线（左）有明显的差距，而 GN 的误差曲线（右）则非常相似。

#### 64 帧输入的结果。

Table 8 (第 3 列) 显示了使用 64 帧片段的结果。在这种情况下，BN 的结果为 73.3 / 90.8。这些数字看起来是可以接受的（\vs32 帧、批量大小 8 时的 73.3 / 90.7），但时间长度（64 \vs32）和批量大小（4 \vs8）之间的权衡可能被忽略了。比较 Table 8 中的第 3 列和第 2 列，我们发现时间长度实际上有积极影响（+1.2%），但这被较小批量大小对 BN 的负面影响所掩盖。

GN 不受这种权衡的影响。GN 的 64 帧变体具有 74.5 / 91.7 的准确率，显示出相对于其 BN 对应版本和所有 BN 变体的健康增益。GN 帮助模型从时间长度中受益，在相同的批量大小下，更长的片段使 top-1 准确率提高了 1.7%（top-5 提高 1.1%）。

GN 在检测、分割和视频分类上的改进表明，在这些任务中，GN 是强大且目前占主导地位的 BN 技术的有力替代方案。

## 5 讨论与未来工作

我们已经将 GN 作为一种不利用批量维度的有效归一化层提出。我们已经在各种应用中评估了 GN 的行为。然而，我们注意到 BN 具有极大的影响力，以至于许多最先进的系统及其超参数都是为它设计的，这对于基于 GN 的模型可能不是最优的。重新设计系统或为 GN 搜索新的超参数可能会给出更好的结果。

此外，我们已经表明 GN 与 LN 和 IN 有关，这两种归一化方法在训练循环 (RNN/LSTM) 或生成 (GAN) 模型方面特别成功。这建议我们在未来研究 GN 在这些领域的应用。我们还将研究 GN 在强化学习 (RL) 任务的表示学习上的性能，\eg, [54]，其中 BN 在训练非常深的模型中发挥着重要作用 [20]。

#### 致谢。

我们要感谢 Piotr Dollár 和 Ross Girshick 进行了有益的讨论。

Generated on Mon Sep 14 22:56:36 2026 by LaTeXML