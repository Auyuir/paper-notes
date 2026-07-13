# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints 原文翻译

# GQA: 从 Multi-Head Checkpoints 训练 Generalized Multi-Query Transformer 模型

*Joshua Ainslie {\- \ Equal contribution.}, James Lee-Thorp [1], Michiel de Jong [1] \ [2] {\- \ University of Southern California. Work done at Google Research.} { Yury Zemlyanskiy}, { Federico Lebr\'{o}n}, { Sumit Sanghai} { Google Research}*

## 摘要

Multi-query attention () 仅使用单个 key-value head，极大地加速了 decoder 推理。然而， 可能导致质量下降，此外，仅仅为了更快的推理而训练一个单独的模型可能并不理想。我们 (1) 提出了一种方案，将现有的 multi-head 语言模型 checkpoints 升级训练为使用 的模型，仅消耗原始预训练计算量的 5%，并且 (2) 引入了 attention ()，这是 multi-query attention 的一种泛化，它使用中间数量（多于一个，少于 query heads 的数量）的 key-value heads。我们表明，升级训练后的 实现了接近 multi-head attention 的质量，且速度与 相当。

---

# 引言

由于在每个解码步骤加载 decoder 权重和所有 attention keys 与 values 产生的内存带宽开销，自回归 decoder 推理是 Transformer 模型的严重瓶颈 (Shazeer 2019; Pope et al. 2022; Jong et al. 2022)。通过使用多个 query heads 但单个 key 和 value head 的 *multi-query attention* (Shazeer 2019)，可以大幅减少加载 keys 和 values 产生的内存带宽。

然而，multi-query attention (MQA) 可能导致质量下降和训练不稳定，并且训练分别针对质量和推理进行优化的独立模型可能并不可行。此外，虽然一些语言模型已经使用了 multi-query attention，例如 PaLM (Chowdhery et al. 2022)，但许多模型并没有，包括公开可用的语言模型，如 T5 (Raffel et al. 2020) 和 LLaMA (Touvron et al. 2023)。

这项工作包含两个关于大型语言模型更快推理的贡献。首先，我们展示了具有 multi-head attention (MHA) 的语言模型 checkpoints 可以被 *uptrained* (Komatsuzaki et al. 2022) 以使用 MQA，且仅消耗原始训练计算量的一小部分。这提供了一种具有成本效益的方法，以获得快速的 multi-query 以及高质量的 MHA checkpoints。

其次，我们提出了 grouped-query attention (GQA)，它是 multi-head 和 multi-query attention 之间的一种插值，在 *每个 query heads 子组* 中具有单个 key 和 value head。我们表明，升级训练的 GQA 实现了接近 multi-head attention 的质量，同时几乎与 multi-query attention 一样快。

# 方法

## 升级训练

从 multi-head 模型生成 multi-query 模型分两步进行：首先，转换 checkpoint；其次，进行额外的预训练以使模型适应其新结构。图 1 展示了将 multi-head checkpoint 转换为 multi-query checkpoint 的过程。Key 和 value heads 的投影矩阵被平均池化为单个投影矩阵，我们发现这比选择单个 key 和 value head 或从头随机初始化新的 key 和 value head 效果更好。

![Overview of conversion from multi-head to multi-query attention. Key and value projection matrices from all heads are mean pooled into a single head.](recycling.png)

然后使用相同的预训练方案，对转换后的 checkpoint 进行预训练，训练步数为原始训练步数的一小部分 $\alpha$。

## Grouped-query attention


![image](gmq-architecture.png)


Grouped-query attention 将 query heads 划分为 $G$ 个 *groups*，每个 group 共享一个 key head 和 value head。GQA-g 指的是具有 $G$ 个 groups 的 grouped-query。GQA-$1$ 具有单个 group，因此具有单个 key 和 value head，等同于 MQA；而 GQA-h 的 groups 数量等于 heads 的数量，等同于 MHA。图 [fig:gmq_architecture] 展示了 grouped-query attention 与 multi-head/multi-query attention 的比较。在将 multi-head checkpoint 转换为 GQA checkpoint 时，我们通过对该 group 内所有原始 heads 进行平均池化来构建每个 group 的 key 和 value head。

中间数量的 groups 产生了一个插值模型，其质量高于 MQA 但速度快于 MHA，并且正如我们将展示的，代表了一种有利的权衡。从 MHA 到 MQA 将 $H$ 个 key 和 value heads 减少为单个 key 和 value head，从而将 key-value cache 的大小以及因此需要加载的数据量减少了 $H$ 倍。然而，较大的模型通常会按比例增加 heads 的数量，因此 multi-query attention 代表了在内存带宽和容量上更激进的削减。GQA 使我们能够在模型尺寸增加时，保持带宽和容量相同比例的减少。

此外，较大的模型相对较少受到 attention 带来的内存带宽开销的影响，因为 KV-cache 随模型维度缩放，而模型 FLOPs 和参数随模型维度的平方缩放。最后，大型模型的标准分片通过模型分区的数量来复制单个 key 和 value head (Pope et al. 2022)；GQA 消除了这种分区造成的浪费。因此，我们期望 GQA 能为大型模型提供特别好的权衡。

我们注意到 GQA 未应用于 encoder self-attention 层；encoder 表示是并行计算的，因此内存带宽通常不是主要瓶颈。

# 实验

## 实验设置

#### 配置

所有模型均基于 T5.1.1 架构 (Raffel et al. 2020)，使用 JAX (Bradbury et al. 2018)、Flax (Heek et al. 2020) 和 Flaxformer (<https://github.com/google/flaxformer>) 实现。对于我们的主要实验，我们考虑具有 multi-head attention 的 T5 Large 和 XXL，以及具有 multi-query 和 grouped-query attention 的 T5 XXL 升级训练版本。我们使用 Adafactor 优化器，其超参数和学习率调度与 T5 (Raffel et al. 2020) 相同。我们将 MQA 和 GQA 应用于 decoder self-attention 和 cross-attention，但不应用于 encoder self-attention。

#### 升级训练

升级训练的模型从公开的 T5.1.1 checkpoints 初始化。Key 和 value heads 被平均池化为适当的 MQA 或 GQA 结构，然后使用 (Raffel et al. 2020) 中的原始预训练设置和数据集，进一步预训练原始预训练步数的 $\alpha$ 比例。对于 $\alpha=0.05$，训练花费了大约 600 个 TPUv3 芯片日。

#### 数据

我们在摘要数据集 CNN/Daily Mail (Nallapati et al. 2016)、arXiv 和 PubMed (Cohan et al. 2018)、MediaSum (Zhu et al. 2021) 以及 Multi-News (Fabbri et al. 2019)；翻译数据集 WMT 2014 英德翻译；以及问答数据集 TriviaQA (Joshi et al. 2017) 上进行评估。我们不在诸如 GLUE (Wang et al. 2019) 等流行的分类基准上进行评估，因为自回归推理不太适用于这些任务。

#### 微调

对于微调，我们在所有任务中使用恒定的学习率 0.001，batch size 为 128，dropout rate 为 0.1。CNN/Daily Mail 和 WMT 使用输入长度 512 和输出长度 256。其他摘要数据集使用输入长度 2048 和输出长度 512。最后，TriviaQA 使用输入长度 2048 和输出长度 32。我们训练直到收敛，并选择在开发集上性能最高的 checkpoint。我们使用贪婪解码进行推理。

#### 计时

我们报告了每个 TPUv4 芯片每个样本的时间，由 xprof (Google 2020) 测量。对于计时实验，我们使用 8 个 TPU，采用每个 TPU 最多容纳 32 的最大 batch size，并为每个模型分别优化并行化。


| **模型**                                | **T<sub>infer</sub>** | **平均** |      **CNN**      |     **arXiv**     |    **PubMed**     |   **MediaSum**    |   **MultiNews**   | **WMT**  | **TriviaQA** |     |     |
|:-----------------------------------------|:---------------------:|:-----------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:--------:|:------------:|:---:|:---:|
|                                          |         **s**         |             | **R<sub>1</sub>** | **R<sub>1</sub>** | **R<sub>1</sub>** | **R<sub>1</sub>** | **R<sub>1</sub>** | **BLEU** |    **F1**    |     |     |
| MHA-Large |         0.37          |    46.0     |       42.9        |       44.6        |       46.2        |       35.5        |       46.6        |   27.7   |     78.2     |     |     |
| MHA-XXL   |         1.51          |    47.2     |       43.8        |       45.6        |       47.5        |       36.4        |       46.9        |   28.4   |     81.9     |     |     |
| MQA-XXL   |         0.24          |    46.6     |       43.0        |       45.0        |       46.9        |       36.1        |       46.5        |   28.5   |     81.3     |     |     |
| GQA-8-XXL |         0.28          |    47.1     |       43.5        |       45.4        |       47.7        |       36.3        |       47.2        |   28.4   |     81.6     |     |     |


与 MHA 相比，Uptrained 的 MQA 实现了良好的折衷，其质量和速度均高于 MHA-Large，而 GQA 取得了更好的性能，在获得相似速度提升的同时，质量与 MHA-XXL 相当。所有任务的平均性能作为每个样本平均推理时间的函数，针对具有 multi-head attention 的 T5-Large 和 T5-XXL，以及经过 5% uptrained 并具有 MQA 和 GQA-8 attention 的 T5-XXL。

## 主要结果

图 2 显示了 MHA T5-Large 和 T5-XXL，以及 uptraining 比例为 $\alpha= 0.05$ 的 uptrained MQA 和 GQA-$8$ XXL 模型在所有数据集上的平均性能随平均推理时间变化的函数关系。我们看到，更大的 uptrained MQA 模型相对于 MHA 模型提供了有利的折衷，比 MHA-Large 具有更高的质量和更快的推理速度。此外，GQA 获得了显著的额外质量提升，实现了接近 MHA-XXL 的性能和接近 MQA 的速度。表 [table:headline_results] 包含了所有数据集的完整结果。

## 消融实验

本节介绍了研究不同建模选择效果的实验。我们在具有代表性的任务子样本上评估性能：CNN/Daily Mail（短文本摘要）、MultiNews（长文本摘要）和 TriviaQA（问答）。

针对以比例 α = 0.05 uptrained 到 MQA 的 T5-Large，比较不同 checkpoint 转换方法的性能。‘Mean’ 对 key 和 value head 进行均值池化，‘First’ 选择第一个 head，‘Random’ 从头初始化 head。

#### Checkpoint 转换

图 3 比较了不同 checkpoint 转换方法的性能。均值池化效果最好，其次是选择单个 head，然后是随机初始化。直观上，结果的排序与从预训练模型中保留信息的程度一致。

#### Uptraining 步骤

图 4 显示了具有 MQA 和 GQA 的 T5 XXL 的性能如何随 uptraining 比例变化。首先，我们注意到 GQA 在转换后已经达到了合理的性能，而 MQA 需要 uptraining 才能发挥作用。MQA 和 GQA 都从 5% 的 uptraining 中获益，但从 10% 开始收益递减。

具有 MQA 和 GQA-8 的 T5 XXL 模型的性能随 uptraining 比例变化的函数。

#### 组数

图 5 展示了 GQA 组数对推理速度的影响。对于较大的模型，来自 KV cache 的内存带宽开销限制较小 (Shazeer 2019)，而由于 head 数量增加，key-value 大小的减小更为明显。因此，从 MQA 增加组数最初只会导致适度的减速，但随着我们接近 MHA，成本会不断增加。我们选择了 8 个组作为有利的折衷点。

在输入长度为 2048 且输出长度为 512 的情况下，GQA-XXL 的每个样本时间随 GQA 组数变化的函数。从 1 组 (MQA) 增加到 8 组会增加适度的推理开销，而增加更多组的成本会随之增加。

# 相关工作

这项工作重点是通过减少加载 key 和 value 带来的内存带宽开销 (Williams, Waterman, and Patterson 2009) 来实现 decoder 质量和推理时间之间更好的折衷。Shazeer (2019) 首先提出通过 multi-query attention 来减少这种开销。后续工作表明，multi-query attention 对长输入特别有帮助 (Pope et al. 2022; Jong et al. 2022)。Rabe (2023) 独立开发了带有公开实现的 GQA。其他工作探索了将 attention head 分组以提高计算效率 (Park et al. 2020; Luo et al. 2022; Ni et al. 2023)，但没有专门关注决定内存带宽开销的 key-value head。

还提出了许多其他方法来减少 key 和 value 以及参数带来的内存带宽开销。Flash attention (Dao et al. 2022) 构建了 attention 计算，以避免具象化二次 attention 分数，从而减少内存并加速训练。Quantization (Dettmers et al. 2022; Frantar et al. 2022) 通过降低精度来减小权重和激活（包括 key 和 value）的大小。Model distillation (Hinton, Vinyals, and Dean 2015; Gou et al. 2021) 则在给定精度下减小模型大小，使用从较大模型生成的数据来微调较小的模型。Layer-sparse cross-attention (Jong et al. 2022) 消除了大部分 cross-attention 层，这些层构成了长输入的主要开销。Speculative sampling (Chen et al. 2023; Leviathan, Kalman, and Matias 2022) 通过使用较小的模型提出多个 token，然后由较大的模型并行评分，从而缓解了内存带宽瓶颈。

最后，我们提出的 uptraining 过程受到了 Komatsuzaki et al. (2022) 的启发，该方法将标准 T5 checkpoint uptrain 为稀疏激活的 Mixture-of-Experts 模型。

# 结论

语言模型的推理成本高昂，主要是由于加载 key 和 value 带来的内存带宽开销。Multi-query attention 以降低模型容量和质量为代价减少了这种开销。我们提出使用一小部分原始预训练计算量将 multi-head attention 模型转换为 multi-query 模型。此外，我们引入了 grouped-query attention，它是 multi-query 和 multi-head attention 的插值，以与 multi-query attention 相当的速度实现了接近 multi-head 的质量。

# 局限性

本文重点改善加载 key 和 value 带来的内存带宽开销。这种开销在生成较长序列时最为重要，而长序列的质量本身就难以评估。对于摘要，我们使用 Rouge score，我们知道这是一个有缺陷的评估，无法说明全部情况；因此，很难确定我们的折衷是否正确。由于计算资源有限，我们也没有将我们的 XXL GQA 模型与从头训练的对比模型进行比较，因此我们不知道 uptraining 与从头训练的相对性能。最后，我们仅在 encoder-decoder 模型上评估了 uptraining 和 GQA 的影响。最近，decoder-only 模型非常受欢迎，由于这些模型没有单独的 self-attention 和 cross-attention，我们预计 GQA 将比 MQA 具有更大的优势。

# 致谢

我们感谢 Google Research 的 Santiago Ontañón、Afroz Mohiuddin、William Cohen 等人提供的深刻建议和讨论。

# 训练稳定性

我们发现 multi-query attention 在微调期间可能导致训练不稳定，特别是与长输入任务结合时。我们从头训练了多个带有 multi-query attention 的 T5-Large 模型。在每种情况下，预训练都遭受了频繁的 loss 尖峰，并且最终模型在长输入任务上进行微调时立即发散。Uptrained 的 multi-query attention 模型更稳定，但仍然表现出高方差，因此对于在不稳定任务上的 multi-query 模型，我们报告了三次微调运行的平均性能。然而，Uptrained 的 grouped-query attention 模型似乎是稳定的，因此我们没有进一步调查 multi-query 不稳定的根本原因。

## 参考文献

Bradbury, James, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, et al. 2018. “JAX：Python+NumPy程序的可组合变换。” <http://github.com/google/jax>.

Chen, Charlie, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. “使用投机采样加速大型语言模型解码。” *CoRR* abs/2302.01318. <https://doi.org/10.48550/arXiv.2302.01318>.

Chowdhery, Aakanksha, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, et al. 2022. “PaLM：使用Pathways扩展语言建模。” arXiv. <https://doi.org/10.48550/ARXIV.2204.02311>.

Cohan, Arman, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. 2018. “用于长文档抽象摘要的篇章感知Attention模型。” 载于 *2018年计算语言学协会北美分会人类语言技术会议论文集，第2卷（短论文）*，615–21。路易斯安那州新奥尔良：计算语言学协会。 <https://doi.org/10.18653/v1/N18-2097>.

Dao, Tri, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. “FlashAttention：具备IO感知的快速且内存高效的精确Attention。” *CoRR* abs/2205.14135. <https://doi.org/10.48550/arXiv.2205.14135>.

Dettmers, Tim, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. “LLM.int8()：用于大规模Transformer的8位矩阵乘法。” *CoRR* abs/2208.07339. <https://doi.org/10.48550/arXiv.2208.07339>.

Fabbri, Alexander R., Irene Li, Tianwei She, Suyi Li, and Dragomir R. Radev. 2019. “Multi-News：一个大规模多文档摘要数据集与抽象层次模型。” 载于 *第57届计算语言学协会会议论文集，ACL 2019，意大利佛罗伦萨，2019年7月28日-8月2日，第1卷：长论文*，Anna Korhonen、David R. Traum和Lluı́s Màrquez编，1074–84。计算语言学协会。 <https://doi.org/10.18653/v1/p19-1102>.

Frantar, Elias, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. “GPTQ：用于生成式预训练Transformer的精确训练后量化。” *CoRR* abs/2210.17323. <https://doi.org/10.48550/arXiv.2210.17323>.

Google. 2020. “使用Cloud TPU工具分析您的模型。” <https://cloud.google.com/tpu/docs/cloud-tpu-tools>.

Gou, Jianping, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. 2021. “知识蒸馏：一项综述。” *Int. J. Comput. Vis.* 129 (6): 1789–819. <https://doi.org/10.1007/s11263-021-01453-z>.

Heek, Jonathan, Anselm Levskaya, Avital Oliver, Marvin Ritter, Bertrand Rondepierre, Andreas Steiner, and Marc van Zee. 2020. “Flax：一个用于JAX的神经网络库与生态系统。” <http://github.com/google/flax>.

Hinton, Geoffrey E., Oriol Vinyals, and Jeffrey Dean. 2015. “蒸馏神经网络中的知识。” *CoRR* abs/1503.02531. <http://arxiv.org/abs/1503.02531>.

Jong, Michiel de, Yury Zemlyanskiy, Joshua Ainslie, Nicholas FitzGerald, Sumit Sanghai, Fei Sha, and William Cohen. 2022. “FiDO：针对更强性能和更快推理优化的Fusion-in-Decoder。” *arXiv Preprint arXiv:2212.08153*. <https://arxiv.org/abs/2212.08153>.

Joshi, Mandar, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. “TriviaQA：一个用于阅读理解的大规模远监督挑战数据集。” 载于 *第55届计算语言学协会年会论文集*。加拿大温哥华：计算语言学协会。

Komatsuzaki, Aran, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. 2022. “稀疏升级循环：从稠密检查点训练Mixture-of-Experts。” arXiv. <https://doi.org/10.48550/ARXIV.2212.05055>.

Leviathan, Yaniv, Matan Kalman, and Yossi Matias. 2022. “通过投机解码实现Transformer的快速推理。” *CoRR* abs/2211.17192. <https://doi.org/10.48550/arXiv.2211.17192>.

Luo, Gen, Yiyi Zhou, Xiaoshuai Sun, Yan Wang, Liujuan Cao, Yongjian Wu, Feiyue Huang, and Rongrong Ji. 2022. “通过面向视觉与语言任务的分组变换迈向轻量级Transformer。” *IEEE Trans. Image Process.* 31: 3386–98. <https://doi.org/10.1109/TIP.2021.3139234>.

Nallapati, Ramesh, Bowen Zhou, Cı́cero Nogueira dos Santos, Çaglar Gülçehre, and Bing Xiang. 2016. “使用Sequence-to-Sequence RNNs及超越方法进行抽象文本摘要。” 载于 *第20届SIGNLL计算自然语言学习会议论文集，CoNLL 2016，德国柏林，2016年8月11-12日*，Yoav Goldberg和Stefan Riezler编，280–90。ACL。 <https://doi.org/10.18653/v1/k16-1028>.

Ni, Jinjie, Rui Mao, Zonglin Yang, Han Lei, and Erik Cambria. 2023. “寻找Multi-Head Attention的支柱力量。” 载于 *第61届计算语言学协会年会论文集（第1卷：长论文），ACL 2023，加拿大多伦多，2023年7月9-14日*，Anna Rogers、Jordan L. Boyd-Graber和Naoaki Okazaki编，14526–40。计算语言学协会。 <https://doi.org/10.18653/V1/2023.ACL-LONG.812>.

Park, Sungrae, Geewook Kim, Junyeop Lee, Junbum Cha, Ji-Hoon Kim, and Hwalsuk Lee. 2020. “通过特征分组缩小Transformer以实现轻量级字符级语言模型。” 载于 *第28届国际计算语言学会议论文集，COLING 2020，西班牙巴塞罗那（在线），2020年12月8-13日*，Donia Scott、Núria Bel和Chengqing Zong编，6883–93。国际计算语言学委员会。 <https://doi.org/10.18653/V1/2020.COLING-MAIN.607>.

Pope, Reiner, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2022. “高效扩展Transformer推理。” *arXiv Preprint arXiv:2211.05102*.

Rabe, Markus. 2023. “内存高效的Attention。” <https://github.com/google/flaxformer/blob/main/flaxformer/components/attention/memory_efficient_attention.py>.

Raffel, Colin, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. “使用统一的文本到文本Transformer探索迁移学习的极限。” *J. Mach. Learn. Res.* 21: 140:1–67. <http://jmlr.org/papers/v21/20-074.html>.

Shazeer, Noam. 2019. “快速Transformer解码：一个写头足矣。” *arXiv Preprint arXiv:1911.02150>.

Touvron, Hugo, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, et al. 2023. “LLaMA：开放且高效的基础语言模型。” arXiv. <https://doi.org/10.48550/ARXIV.2302.13971>.

Wang, Alex, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019. “GLUE：一个用于自然语言理解的多任务基准与分析平台。” 载于 *第7届国际学习表征会议，ICLR 2019，美国路易斯安那州新奥尔良，2019年5月6-9日*。OpenReview.net。 <https://openreview.net/forum?id=rJ4km2R5t7>.

Williams, Samuel, Andrew Waterman, and David A. Patterson. 2009. “Roofline：一种用于多核架构的富有洞察力的可视化性能模型。” *Commun. ACM* 52 (4): 65–76. <https://doi.org/10.1145/1498765.1498785>.

Zhu, Chenguang, Yang Liu, Jie Mei, 和 Michael Zeng. 2021. “MediaSum：一个用于对话摘要的大规模媒体访谈数据集。” 载于 *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021*，由 Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tür, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, 和 Yichao Zhou 编辑，5927–34。Association for Computational Linguistics. <https://doi.org/10.18653/v1/2021.naacl-main.474>.