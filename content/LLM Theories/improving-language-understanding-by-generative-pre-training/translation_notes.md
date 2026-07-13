# Improving Language Understanding by Generative Pre-Training 原文翻译

# 通过生成式预训练提升语言理解能力

Alec Radford   
OpenAI   
alec@openai.com

Karthik Narasimhan OpenAI karthikn@openai.com

Tim Salimans OpenAI tim@openai.com

Ilya Sutskever OpenAI ilyasu@openai.com

## 摘要

自然语言理解包含广泛的多样化任务，如文本蕴含、问答、语义相似度评估和文档分类。尽管大量未标注文本语料库非常丰富，但用于学习这些特定任务的标注数据却很稀缺，这使得判别式训练模型难以表现出色。我们证明了，通过在多样化的未标注文本语料库上对语言模型进行生成式预训练，随后在每个特定任务上进行判别式微调，可以在这些任务上实现巨大的性能提升。与以往的方法不同，我们在微调过程中使用了任务感知的输入转换，以实现有效的迁移，同时对模型架构的改动最小。我们在广泛的自然语言理解基准测试上证明了该方法的有效性。我们的通用任务无关模型优于使用为每个任务专门设计的架构的判别式训练模型，在所研究的 12 项任务中有 9 项显著超越了现有技术水平。例如，我们在常识推理上取得了 8.9% 的绝对提升，在问答上取得了 5.7% 的提升，在文本蕴含上取得了 1.5% 的提升。

## 1 引言

从原始文本中有效学习的能力，对于减轻自然语言处理（NLP）对监督学习的依赖至关重要。大多数深度学习方法需要大量人工标注的数据，这限制了它们在许多缺乏标注资源的领域中的适用性 [61]。在这些情况下，能够利用未标注数据中的语言信息的模型，为收集更多标注提供了一种有价值的替代方案，因为收集标注既耗时又昂贵。此外，即使在有大量监督可用的情况下，以无监督方式学习良好的表示也能显著提升性能。迄今为止，最令人信服的证据是广泛使用预训练的词 Embedding [10, 39, 42] 来提升一系列 NLP 任务的性能 [8, 11, 26, 45]。

然而，利用未标注文本中超出词级别的信息面临两个主要挑战，因此具有挑战性。首先，尚不清楚哪种类型的优化目标最有效地学习对迁移有用的文本表示。最近的研究考察了各种目标，如语言建模 [44]、机器翻译 [38] 和语篇连贯性 [22]，每种方法在不同的任务上都互有胜负。<sup>1</sup> 其次，对于如何将这些学习到的表示最有效地迁移到目标任务上，目前还没有共识。现有技术包括对模型架构进行特定于任务的修改 [43, 44]、使用复杂的学习方案 [21] 以及添加辅助学习目标 [50] 的组合。这些不确定性使得开发有效的语言处理半监督学习方法变得困难。

在本文中，我们探索了一种结合无监督预训练和有监督微调的半监督方法，用于语言理解任务。我们的目标是学习一种通用表示，只需很少的调整即可迁移到各种任务中。我们假设可以访问大量未标注文本语料库和几个带有手动标注训练示例的数据集（目标任务）。我们的设置不要求这些目标任务与未标注语料库处于同一领域。我们采用两阶段训练过程。首先，我们在未标注数据上使用语言建模目标来学习神经网络模型的初始参数。随后，我们使用相应的监督目标使这些参数适应目标任务。

对于我们的模型架构，我们使用 Transformer [62]，它已被证明在机器翻译 [62]、文档生成 [34] 和句法分析 [29] 等各种任务上表现出色。与循环网络等替代方案相比，这种模型选择为我们提供了更结构化的记忆来处理文本中的长期依赖关系，从而在多样化任务中实现了稳健的迁移性能。在迁移过程中，我们利用了源自遍历式方法 [52] 的特定于任务的输入适配，该方法将结构化文本输入作为单个连续的 Token 序列进行处理。正如我们在实验中所展示的，这些适配使我们能够在对预训练模型架构进行最小改动的情况下进行有效微调。

我们在四种类型的语言理解任务上评估了我们的方法——自然语言推理、问答、语义相似度和文本分类。我们的通用任务无关模型优于采用为每个任务专门设计的架构的判别式训练模型，在所研究的 12 项任务中有 9 项显著超越了现有技术水平。例如，我们在常识推理上取得了 8.9% 的绝对提升 [40]，在问答上取得了 5.7% 的提升 [30]，在文本蕴含上取得了 1.5% 的提升 [66]，在最近引入的 GLUE 多任务基准上取得了 5.5% 的提升 [64]。我们还分析了预训练模型在四种不同设置下的零样本行为，并证明它为下游任务获取了有用的语言知识。

## 2 相关工作

NLP的半监督学习 我们的工作广泛属于自然语言的半监督学习范畴。这种范式引起了极大的兴趣，并应用于序列标注 [24, 33, 57] 或文本分类 [41, 70] 等任务。最早的方法使用无标签数据来计算词级或短语级的统计信息，然后将其作为监督模型中的特征 [33]。在过去的几年中，研究人员已经证明了使用在无标签语料库上训练的词嵌入 [11, 39, 42] 来提高各种任务性能的好处 [8, 11, 26, 45]。然而，这些方法主要转移词级信息，而我们的目标是捕获更高层次的语义。

最近的方法研究了从无标签数据中学习和利用超越词级别的语义。短语级或句子级的嵌入可以使用无标签语料库进行训练，已被用于将文本编码为适用于各种目标任务的向量表示 [28, 32, 1, 36, 22, 12, 56, 31]。

无监督预训练 无监督预训练是半监督学习的一个特例，其目标是找到一个良好的初始化点，而不是修改监督学习的目标。早期的工作探索了将该技术用于图像分类 [20, 49, 63] 和回归任务 [3]。随后的研究 [15] 表明，预训练作为一种正则化方案，能够使深度神经网络获得更好的泛化能力。在最近的工作中，该方法已被用于帮助训练深度神经网络完成各种任务，如图像分类 [69]、语音识别 [68]、实体消歧 [17] 和机器翻译 [48]。

与我们最接近的工作路线涉及使用语言建模目标对神经网络进行预训练，然后在有监督的目标任务上对其进行微调。Dai 等人 [13] 以及 Howard 和 Ruder [21] 遵循此方法来改进文本分类。然而，尽管预训练阶段有助于捕获一些语言信息，但他们使用的 LSTM 模型将其预测能力限制在短距离范围内。相比之下，我们对 Transformer 网络的选择使我们能够捕获更长距离的语言结构，正如我们在实验中所展示的那样。此外，我们还在更广泛的任务上展示了我们模型的有效性，包括自然语言推理、复述检测和故事补全。其他方法 [43, 44, 38] 在目标任务上训练监督模型时，使用来自预训练语言或机器翻译模型的隐藏表示作为辅助特征。这为每个单独的目标任务涉及大量新参数，而我们在迁移过程中只需要对模型架构进行极少的修改。

辅助训练目标 添加辅助无监督训练目标是半监督学习的一种替代形式。Collobert 和 Weston [10] 的早期工作使用了各种辅助 NLP 任务，如词性标注、分块、命名实体识别和语言建模，以改进语义角色标注。最近，Rei [50] 在其目标任务目标中添加了辅助语言建模目标，并展示了在序列标注任务上的性能提升。我们的实验也使用了辅助目标，但正如我们所展示的，无监督预训练已经学习到了与目标任务相关的若干语言方面。

## 3 框架

我们的训练过程包括两个阶段。第一阶段是在大型文本语料库上学习高容量的语言模型。随后是微调阶段，我们使用标记数据使模型适应判别性任务。

## 3.1 无监督预训练

给定一个无监督的 Token 语料库 $\mathcal { U } = \{ u _ { 1 } , \ldots , u _ { n } \}$ ，我们使用标准的语言建模目标来最大化以下似然：

$$
L _ { 1 } ( \mathcal { U } ) = \sum _ { i } \log P ( u _ { i } | u _ { i - k } , . . . , u _ { i - 1 } ; \Theta )\tag{1}
$$

其中 k 是上下文窗口的大小，条件概率 $P$ 使用具有参数 Θ 的神经网络进行建模。这些参数使用随机梯度下降 [51] 进行训练。

在我们的实验中，我们使用多层 Transformer 解码器 [34] 作为语言模型，它是 Transformer [62] 的一种变体。该模型对输入上下文 Token 应用多头自注意力操作，随后是位置级前馈层，以在目标 Token 上产生输出分布：

$$
\begin{array} { r l } & { \boldsymbol { h } _ { 0 } = \boldsymbol { U } \boldsymbol { W } _ { e } + \boldsymbol { W } _ { p } } \\ & { \quad \boldsymbol { h } _ { l } = \mathbf { t r a n s f o r m e r \_ b l o c k } ( \boldsymbol { h } _ { l - 1 } ) \forall i \in [ 1 , \boldsymbol { n } ] } \\ & { \boldsymbol { P } ( u ) = \mathbf { s o f } \mathbf { t m a x } ( \boldsymbol { h } _ { n } \boldsymbol { W } _ { e } ^ { T } ) } \end{array}\tag{2}
$$

其中 $U = \left( u _ { - k } , \dots , u _ { - 1 } \right)$ 是 Token 的上下文向量，n 是层数，$W _ { e }$ 是 Token Embedding 矩阵，$W _ { p }$ 是位置 Embedding 矩阵。

## 3.2 监督微调

在使用公式 1 中的目标训练模型之后，我们将参数调整至有监督的目标任务。我们假设有一个带标签的数据集 C，其中每个实例由一系列输入 Token $x ^ { 1 } , \ldots , x ^ { m }$ 以及一个标签 y 组成。输入通过我们的预训练模型以获得最终 Transformer 块的激活值 $h _ { l } ^ { m }$ ，然后将其输入到带有参数 $W _ { y }$ 的附加线性输出层以预测 $y \colon$

$$
P ( y | x ^ { 1 } , \ldots , x ^ { m } ) = { \tt s o f t m a x } ( h _ { l } ^ { m } W _ { y } ) .\tag{3}
$$

这给出了我们要最大化的以下目标：

$$
L _ { 2 } ( \mathcal { C } ) = \sum _ { ( x , y ) } \log P ( y | x ^ { 1 } , . . . , x ^ { m } ) .\tag{4}
$$

我们还发现，将语言建模作为微调的辅助目标有助于学习，方式包括 提高监督模型的泛化能力，以及 加速收敛。这与先前的工作 [50, 43] 一致，他们也在使用此类辅助目标时观察到了性能提升。具体而言，我们优化以下目标（权重为 λ）：

$$
L _ { 3 } ( \mathcal { C } ) = L _ { 2 } ( \mathcal { C } ) + \lambda * L _ { 1 } ( \mathcal { C } )\tag{5}
$$

总体而言，我们在微调期间需要的唯一额外参数是 $W _ { y }$ ，以及分隔符 Token 的 Embedding（在下文 3.3 节中描述）。

![](2c30b602030ba602a6891860329f4ebe9dbe5eabfbe2e49fa9f7fc3469333c1d.jpg)  
图 1：（左）本工作中使用的 Transformer 架构和训练目标。（右）针对不同任务进行微调的输入转换。我们将所有结构化输入转换为 Token 序列，以便由我们的预训练模型处理，随后是一个 linear+softmax 层。

## 3.3 针对特定任务的输入转换

对于某些任务，如文本分类，我们可以如上所述直接微调我们的模型。某些其他任务，如问答或文本蕴含，具有结构化输入，例如有序的句子对，或文档、问题和答案的三元组。由于我们的预训练模型是在连续的文本序列上训练的，我们需要进行一些修改才能将其应用于这些任务。先前的工作提出在迁移表示之上学习特定任务的架构 [44]。这种方法重新引入了大量的特定任务定制，并且没有对这些额外的架构组件使用迁移学习。相反，我们使用一种遍历风格的方法 [52]，将结构化输入转换为我们的预训练模型可以处理的有序序列。这些输入转换使我们能够避免在不同任务中对架构进行大量修改。我们在下面提供了这些输入转换的简要描述，图 1 提供了直观的说明。所有转换都包括添加随机初始化的起始和结束 Token (hsi, hei)。

文本蕴含 对于蕴含任务，我们将前提 p 和假设 h 的 Token 序列拼接起来，中间带有一个分隔 Token (\$)。

相似度 对于相似度任务，被比较的两个句子没有固有的顺序。为了反映这一点，我们修改输入序列以包含两种可能的句子顺序（中间带有分隔符），并独立处理每一个以产生两个序列表示 $h _ { l } ^ { m }$，它们在输入到线性输出层之前按元素相加。

问答与常识推理 对于这些任务，我们给定一个上下文文档 z、一个问题 q 和一组可能的答案 $\{ a _ { k } \}$ 。我们将文档上下文和问题与每个可能的答案拼接起来，中间添加一个分隔 Token 以获得 $[ z ; q ; \$ 3; a _ { k } ]$ 。这些序列中的每一个都使用我们的模型独立处理，然后通过 softmax 层进行归一化，以产生可能答案上的输出分布。

## 4 实验

## 4.1 设置

无监督预训练 我们使用 BooksCorpus 数据集 [71] 来训练语言模型。它包含超过 7,000 本来自各种类型（包括冒险、奇幻和浪漫）的独特未出版书籍。至关重要的是，它包含长段的连续文本，这使得生成模型能够学习以长程信息为条件。另一个数据集，1B Word Benchmark，被类似的方法 ELMo [44] 所使用，其大小大致相同，但在句子级别进行了打乱——破坏了长程结构。我们的语言模型在该语料库上实现了 18.4 的极低 Token 级别困惑度。

表 1：我们实验中使用的不同任务和数据集列表。
<table><tr><td>任务</td><td>数据集</td></tr><tr><td>自然语言推理</td><td>SNLI [5], MultiNLI [66], Question NLI [64], RTE [4], SciTail [25]</td></tr><tr><td>问答</td><td>RACE [30], Story Cloze [40]</td></tr><tr><td>句子相似度</td><td>MSR Paraphrase Corpus [14], Quora Question Pairs [9], STS Benchmark [6]</td></tr><tr><td>分类</td><td>Stanford Sentiment Treebank-2 [54], CoLA [65]</td></tr></table>

模型规格 我们的模型在很大程度上遵循了原始的 Transformer 工作 [62]。我们训练了一个 12 层的仅包含解码器的 Transformer，带有掩码自注意力头（768 维状态和 12 个注意力头）。对于位置式前馈网络，我们使用了 3072 维的内部状态。我们使用了 Adam 优化方案 [27]，最大学习率为 2.5e-4。学习率在前 2000 次更新中从零线性增加，然后使用余弦调度退火至 0。我们在包含 64 个随机采样的 512 个 Token 的连续序列的小批量上训练了 100 个 epoch。由于 layernorm [2] 在整个模型中被广泛使用，简单的 N(0, 0.02) 权重初始化就足够了。我们使用了包含 40,000 次合并的字节对编码 (BPE) 词汇表 [53]，以及比率为 0.1 的残差、Embedding 和 Attention dropout 进行正则化。我们还采用了 [37] 中提出的修改版 L2 正则化，对所有非偏置或增益权重设置 w = 0.01。对于激活函数，我们使用了高斯误差线性单元 (GELU) [18]。我们使用了学习的位置 Embedding，而不是原始工作中提出的正弦版本。我们使用 ftfy 库<sup>2</sup> 来清理 BooksCorpus 中的原始文本，标准化一些标点符号和空格，并使用 spaCy 分词器。<sup>3</sup>

微调细节 除非特别说明，我们复用无监督预训练的超参数设置。我们向分类器添加了比率为 0.1 的 dropout。对于大多数任务，我们使用 6.25e-5 的学习率和 32 的批量大小。我们的模型微调速度很快，在大多数情况下 3 个 epoch 的训练就足够了。我们使用带有预热（在 0.2% 的训练过程中）的线性学习率衰减调度。λ 被设置为 0.5。

## 4.2 监督微调

我们在各种监督任务上进行了实验，包括自然语言推理、问答、语义相似度和文本分类。其中一些任务是最近发布的 GLUE 多任务基准 [64] 的一部分，我们利用了这些任务。图 1 提供了所有任务和数据集的概述。

Natural Language Inference 自然语言推理（NLI）任务，也称为识别文本蕴含，涉及阅读一对句子并判断它们之间的关系是蕴含、矛盾还是中立。尽管最近有很多关注 [58, 35, 44]，但由于存在词汇蕴含、共指以及词汇和句法歧义等多种现象，该任务仍然具有挑战性。我们在五个来源多样的数据集上进行了评估，包括图像描述（SNLI）、转录语音、流行小说和政府报告（MNLI）、维基百科文章（QNLI）、科学考试或新闻文章（RTE）。

表 2 详细列出了我们的模型和先前最先进方法在不同 NLI 任务上的各种结果。我们的方法在五个数据集中的四个上显著优于基线，在 MNLI 上比先前的最佳结果实现了高达 1.5% 的绝对提升，在 SciTail 上为 5%，在 QNLI 上为 5.8%，在 SNLI 上为 0.6%。这证明了我们的模型能够更好地对多个句子进行推理，并处理语言歧义的各个方面。在 RTE（我们评估的较小数据集之一，包含 2490 个样本）上，我们达到了 56% 的准确率，低于多任务 biLSTM 模型报告的 61.7%。鉴于我们的方法在较大的 NLI 数据集上的强劲表现，我们的模型很可能也会从多任务训练中受益，但我们目前尚未对此进行探索。

表 2：自然语言推理任务的实验结果，将我们的模型与当前最先进的方法进行比较。5x 表示 5 个模型的集成。所有数据集均使用准确率作为评估指标。
<table><tr><td>方法</td><td>MNLI-m</td><td>MNLI-mm</td><td>SNLI</td><td>SciTail</td><td>QNLI</td><td>RTE</td></tr><tr><td>ESIM + ELMo [44] (5x)</td><td></td><td></td><td>89.3</td><td>-</td><td>=</td><td>=</td></tr><tr><td>CAFE [58] (5x)</td><td>80.2</td><td>79.0</td><td>89.3</td><td>=</td><td></td><td></td></tr><tr><td>Stochastic Answer Network [35] (3x)</td><td>80.6</td><td>80.1</td><td></td><td></td><td></td><td></td></tr><tr><td>CAFE [58]</td><td>78.7</td><td>77.9</td><td>88.5</td><td>83.3</td><td></td><td></td></tr><tr><td>GenSen [64]</td><td>71.4</td><td>71.3</td><td></td><td>=</td><td>82.3</td><td>59.2</td></tr><tr><td>Multi-task BiLSTM + Attn [64]</td><td>72.2</td><td>72.1</td><td>=</td><td></td><td>82.1</td><td>61.7</td></tr><tr><td>Finetuned Transformer LM（我们的）</td><td>82.1</td><td>81.4</td><td>89.9</td><td>88.3</td><td>88.1</td><td>56.0</td></tr></table>

表 3：问答和常识推理的结果，将我们的模型与当前最先进的方法进行比较。9x 表示 9 个模型的集成。
<table><tr><td>方法</td><td>Story Cloze</td><td>RACE-m</td><td>RACE-h</td><td>RACE</td></tr><tr><td>val-LS-skip [55]</td><td>76.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Hidden Coherence Model [7]</td><td>77.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Dynamic Fusion Net [67] (9x)</td><td>-</td><td>55.6</td><td>49.4</td><td>51.2</td></tr><tr><td>BiAttention MRU [59] (9x)</td><td>-</td><td>60.2</td><td>50.3</td><td>53.3</td></tr><tr><td>Finetuned Transformer LM（我们的）</td><td>86.5</td><td>62.9</td><td>57.4</td><td>59.0</td></tr></table>

Question answering and commonsense reasoning 另一项需要单句和多句推理能力的任务是问答。我们使用最近发布的 RACE 数据集 [30]，其中包含来自中学和高中考试的英语文章及相关问题。该语料库已被证明包含比 CNN [19] 或 SQuaD [47] 等其他数据集更多的推理型问题，为经过训练以处理长程上下文的模型提供了完美的评估。此外，我们还在 Story Cloze Test [40] 上进行了评估，该任务涉及从两个选项中为多句故事选择正确的结尾。在这些任务上，我们的模型再次以显著优势超越了先前的最佳结果——在 Story Cloze 上高达 8.9%，在 RACE 上总体为 5.7%。这证明了我们的模型有效处理长程上下文的能力。

Semantic Similarity 语义相似度（或复述检测）任务涉及预测两个句子在语义上是否等价。挑战在于识别概念的重新表述、理解否定以及处理句法歧义。我们为此任务使用了三个数据集——微软复述语料库（MRPC）[14]（从新闻来源收集）、Quora Question Pairs（QQP）数据集 [9] 以及语义文本相似度基准（STS-B）[6]。我们在三个语义相似度任务中的两个上获得了最先进的结果（表 4），在 STS-B 上获得了 1 个点的绝对提升。在 QQP 上的性能差异显著，比 Single-task BiLSTM + ELMo + Attn 绝对提升了 4.2%。

Classification 最后，我们还在两个不同的文本分类任务上进行了评估。语言可接受性语料库（CoLA）[65] 包含专家关于句子是否符合语法的判断，并测试训练模型的内在语言偏置。另一方面，斯坦福情感树库（SST-2）[54] 是一个标准的二分类任务。我们的模型在 CoLA 上获得了 45.4 的分数，这比先前 35.0 的最佳结果有特别大的飞跃，展示了我们模型学习到的内在语言偏置。该模型在 SST-2 上也达到了 91.3% 的准确率，与最先进的结果具有竞争力。我们还在 GLUE 基准上获得了 72.8 的总分，显著优于先前 68.9 的最佳结果。

表 4：语义相似度和分类结果，将我们的模型与当前最先进的方法进行比较。此表中的所有任务评估均使用 GLUE 基准完成。（mc=马修斯相关系数，acc=准确率，pc=皮尔逊相关系数）
<table><tr><td rowspan="2">方法</td><td colspan="2">分类</td><td colspan="3">语义相似度</td><td rowspan="2">GLUE</td></tr><tr><td>CoLA (mc)</td><td>SST2 (acc)</td><td>MRPC (F1)</td><td>STSB (pc)</td><td>QQP (F1)</td></tr><tr><td>Sparse byte mLSTM [16]</td><td>=</td><td>93.2</td><td>-</td><td>-</td><td>-</td><td></td></tr><tr><td>TF-KLD [23]</td><td>一</td><td>-</td><td>86.0</td><td>-</td><td>一</td><td>一</td></tr><tr><td>ECNU (mixed ensemble) [60]</td><td>-</td><td>-</td><td>-</td><td>81.0</td><td>-</td><td></td></tr><tr><td>Single-task BiLSTM + ELMo + Attn [64]</td><td>35.0</td><td>90.2</td><td>80.2</td><td>55.5</td><td>66.1</td><td>64.8</td></tr><tr><td>Multi-task BiLSTM + ELMo + Attn [64]</td><td>18.9</td><td>91.6</td><td>83.5</td><td>72.8</td><td>63.3</td><td>68.9</td></tr><tr><td>Finetuned Transformer LM（我们的）</td><td>45.4</td><td>91.3</td><td>82.3</td><td>82.0</td><td>70.3</td><td>72.8</td></tr></table>

总体而言，我们的方法在我们评估的 12 个数据集中的 9 个上取得了新的最先进结果，在许多情况下优于集成模型。我们的结果还表明，我们的方法在不同规模的数据集上均表现良好，从较小的数据集（如 STS-B，约 5.7k 训练样本）到最大的数据集（SNLI，约 550k 训练样本）。

## 5 分析

迁移层数的影响 我们观察了从无监督预训练到有监督目标任务迁移不同层数所产生的影响。图 2(左) 展示了我们的方法在 MultiNLI 和 RACE 上的性能随迁移层数变化的函数关系。我们观察到了标准的结果，即迁移 embeddings 可以提升性能，并且每一层 transformer 都能带来进一步的收益，在 MultiNLI 上进行全量迁移时收益高达 9%。这表明预训练模型中的每一层都包含用于解决目标任务的有用功能。

![](2c69fed3c6ecd2c00e8a0cf6d4ca50400f8cce340bc2f13127851049bc9ba718.jpg)

![](25b6444944c565e5b1eb049c975849f8e9b2426aeefee490f47a7c9290b2da1d.jpg)  
图 2：(左) 从预训练语言模型迁移递增层数对 RACE 和 MultiNLI 的影响。(右) 显示不同任务上 zero-shot 性能随 LM 预训练更新次数变化的曲线图。每个任务的性能在随机猜测基线和当前最先进的单模型之间进行了归一化。

Zero-shot 行为 我们希望更好地理解为什么 transformer 的语言模型预训练是有效的。一个假设是，底层的生成模型为了提高其语言建模能力，学习执行了我们评估的许多任务，并且与 LSTM 相比，transformer 更具结构化的注意力记忆有助于迁移。我们设计了一系列启发式解决方案，使用底层的生成模型在没有监督微调的情况下执行任务。我们在图 2(右) 中可视化了这些启发式解决方案在生成式预训练过程中的有效性。我们观察到这些启发式方法的性能稳定并且在训练期间稳步提升，这表明生成式预训练支持学习与各种任务相关的功能。我们还观察到 LSTM 在其 zero-shot 性能上表现出更高的方差，这表明 Transformer 架构的归纳偏置有助于迁移。

表 5：不同任务上各种模型消融的分析。平均分是所有结果的无权重平均值。(mc=马修斯相关系数, acc=准确率, pc=皮尔逊相关系数)
<table><tr><td>方法</td><td>平均分</td><td>CoLA (mc)</td><td>SST2 (acc)</td><td>MRPC (F1)</td><td>STSB (pc)</td><td>QQP (F1)</td><td>MNLI (acc)</td><td>QNLI (acc)</td><td>RTE (acc)</td></tr><tr><td>Transformer w/ aux LM (full)</td><td>74.7</td><td>45.4</td><td>91.3</td><td>82.3</td><td>82.0</td><td>70.3</td><td>81.8</td><td>88.1</td><td>56.0</td></tr><tr><td>Transformer w/o pre-training</td><td>59.9</td><td>18.9</td><td>84.0</td><td>79.4</td><td>30.9</td><td>65.5</td><td>75.7</td><td>71.2</td><td>53.8</td></tr><tr><td>Transformer w/o aux LM</td><td>75.0</td><td>47.9</td><td>92.0</td><td>84.9</td><td>83.2</td><td>69.8</td><td>81.1</td><td>86.9</td><td>54.4</td></tr><tr><td>LSTM w/ aux LM</td><td>69.1</td><td>30.3</td><td>90.5</td><td>83.2</td><td>71.8</td><td>68.1</td><td>73.7</td><td>81.1</td><td>54.6</td></tr></table>

对于 CoLA（语言可接受性），样本的得分依据是生成模型分配的平均 token 对数概率，并通过设定阈值进行预测。对于 SST-2（情感分析），我们在每个样本后附加 token very，并将语言模型的输出分布限制为 positive 和 negative 这两个词，猜测其分配更高概率的 token 作为预测。对于 RACE（问答），我们选择在以文档和问题为条件时，生成模型分配最高平均 token 对数概率的答案。对于 DPRD [46]（winograd schemas），我们用两个可能的指代对象替换定代词，并预测在替换后生成模型对序列其余部分分配更高平均 token 对数概率的解析。

消融研究 我们进行了三项不同的消融研究（表 5）。首先，我们检验了在微调期间不使用辅助 LM 目标时我们方法的性能。我们观察到辅助目标在 NLI 任务和 QQP 上有帮助。总体而言，趋势表明较大的数据集从辅助目标中受益，但较小的数据集则不然。其次，我们通过将 Transformer 与使用相同框架的单层 2048 单元 LSTM 进行比较，分析了 Transformer 的效果。我们观察到使用 LSTM 代替 Transformer 时平均分下降了 5.6。LSTM 仅在一个数据集——MRPC 上优于 Transformer。最后，我们还与直接在有监督目标任务上训练、没有预训练的 transformer 架构进行了比较。我们观察到缺乏预训练会损害所有任务的性能，与我们的完整模型相比导致下降了 14.8%。

## 6 结论

我们引入了一个框架，通过生成式预训练和判别式微调，使用单一的任务无关模型来实现强大的自然语言理解。通过在包含长连续文本的多样化语料库上进行预训练，我们的模型获得了丰富的世界知识和处理长程依赖的能力，随后这些知识和能力被成功迁移以解决判别式任务，如问答、语义相似性评估、蕴含判断和文本分类，在我们研究的 12 个数据集中有 9 个提升了当前的最优水平。使用无监督（预）训练来提升判别式任务的性能长期以来一直是机器学习研究的一个重要目标。我们的工作表明，实现显著的性能提升确实是可能的，并提供了关于哪些模型和数据集（具有长程依赖的文本）最适合这种方法的提示。我们希望这将有助于推动无监督学习的新研究，无论是在自然语言理解还是其他领域，从而进一步提高我们对无监督学习如何以及何时起作用的理解。

## 参考文献

[1] S. Arora, Y. Liang, and T. Ma. 句子嵌入的一个简单但难以超越的基线. 2016.

[2] J. L. Ba, J. R. Kiros, and G. E. Hinton. 层归一化. arXiv preprint arXiv:1607.06450, 2016.

[3] Y. Bengio, P. Lamblin, D. Popovici, and H. Larochelle. 深度网络的逐层贪婪训练. 载于 Advances in neural information processing systems，第 153–160 页，2007 年。

[4] L. Bentivogli, P. Clark, I. Dagan, and D. Giampiccolo. 第五届 PASCAL 文本蕴含识别挑战赛. 载于 TAC, 2009.

[5] S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning. 用于学习自然语言推理的大型标注语料库. EMNLP, 2015.

[6] D. Cer, M. Diab, E. Agirre, I. Lopez-Gazpio, and L. Specia. Semeval-2017 任务 1：语义文本相似度——聚焦多语言与跨语言评估. arXiv preprint arXiv:1708.00055, 2017.

[7] S. Chaturvedi, H. Peng, and D. Roth. 用于预测后续发展的故事理解. 载于 Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing，第 1603–1614 页，2017 年。

[8] D. Chen and C. Manning. 使用神经网络的快速准确的依存句法分析器. 载于 Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)，第 740–750 页，2014 年。

[9] Z. Chen, H. Zhang, X. Zhang, and L. Zhao. Quora 问题对. https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs, 2018.

[10] R. Collobert and J. Weston. 用于自然语言处理的统一架构：基于多任务学习的深度神经网络. 载于 Proceedings of the 25th international conference on Machine learning，第 160–167 页. ACM, 2008.

[11] R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, and P. Kuksa. （几乎）从零开始的自然语言处理. Journal of Machine Learning Research, 12(Aug):2493–2537, 2011.

[12] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, and A. Bordes. 从自然语言推理数据中监督学习通用句子表示. EMNLP, 2017.

[13] A. M. Dai and Q. V. Le. 半监督序列学习. 载于 Advances in Neural Information Processing Systems，第 3079–3087 页，2015 年。

[14] W. B. Dolan and C. Brockett. 自动构建句子释义语料库. 载于 Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005.

[15] D. Erhan, Y. Bengio, A. Courville, P.-A. Manzagol, P. Vincent, and S. Bengio. 为什么无监督预训练有助于深度学习？ Journal of Machine Learning Research, 11(Feb):625–660, 2010.

[16] S. Gray, A. Radford, and K. P. Diederik. 用于块稀疏权重的 GPU 内核. 2017.

[17] Z. He, S. Liu, M. Li, M. Zhou, L. Zhang, and H. Wang. 学习用于实体消歧的实体表示. 载于 Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), volume 2, 第 30–34 页，2013 年。

[18] D. Hendrycks and K. Gimpel. 用高斯误差线性单元桥接非线性与随机正则化器. arXiv preprint arXiv:1606.08415, 2016.

[19] K. M. Hermann, T. Kocisky, E. Grefenstette, L. Espeholt, W. Kay, M. Suleyman, and P. Blunsom. 教机器阅读和理解. 载于 Advances in Neural Information Processing Systems，第 1693– 1701 页，2015 年。

[20] G. E. Hinton, S. Osindero, and Y.-W. Teh. 深度信念网络的快速学习算法. Neural computation, 18(7):1527–1554, 2006.

[21] J. Howard and S. Ruder. 用于文本分类的通用语言模型微调. Association for Computational Linguistics (ACL), 2018.

[22] Y. Jernite, S. R. Bowman, and D. Sontag. 用于快速无监督句子表示学习的基于话语的目标. arXiv preprint arXiv:1705.00557, 2017.

[23] Y. Ji and J. Eisenstein. 对分布式句子相似度的判别式改进. 载于 Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing，第 891–896 页，2013 年。

[24] F. Jiao, S. Wang, C.-H. Lee, R. Greiner, and D. Schuurmans. 用于改进序列分割与标注的半监督条件随机场. 载于 Proceedings of the 21st International Conference on Computational Linguistics and the 44th annual meeting of the Association for Computational Linguistics，第 209–216 页. Association for Computational Linguistics, 2006.

[25] T. Khot, A. Sabharwal, and P. Clark. Scitail：一个来自科学问答的文本蕴含数据集. 载于 Proceedings of AAAI, 2018.

[26] Y. Kim. 用于句子分类的卷积神经网络. EMNLP, 2014.

[27] D. P. Kingma and J. Ba. Adam：一种随机优化方法. arXiv preprint arXiv:1412.6980, 2014.

[28] R. Kiros, Y. Zhu, R. R. Salakhutdinov, R. Zemel, R. Urtasun, A. Torralba, and S. Fidler. Skip-thought 向量. 载于 Advances in neural information processing systems，第 3294–3302 页，2015 年。

[29] N. Kitaev and D. Klein. 使用自注意力编码器进行成分句法分析. ACL, 2018.

[30] G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy. Race：来自考试的大规模阅读理解数据集. EMNLP, 2017.

[31] G. Lample, L. Denoyer, and M. Ranzato. 仅使用单语语料库的无监督机器翻译. ICLR, 2018.

[32] Q. Le and T. Mikolov. 句子和文档的分布式表示. 载于 International Conference on Machine Learning，第 1188–1196 页，2014 年。

[33] P. Liang. 面向自然语言的半监督学习. 博士论文，麻省理工学院，2005 年。

[34] P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, L. Kaiser, and N. Shazeer. 通过总结长序列生成维基百科. ICLR, 2018.

[35] X. Liu, K. Duh, and J. Gao. 用于自然语言推理的随机答案网络. arXiv preprint arXiv:1804.07888, 2018.

[36] L. Logeswaran and H. Lee. 一个学习句子表示的高效框架. ICLR, 2018.

[37] I. Loshchilov and F. Hutter. 修复 Adam 中的权重衰减正则化. arXiv preprint arXiv:1711.05101, 2017.

[38] B. McCann, J. Bradbury, C. Xiong, and R. Socher. 在翻译中学习：上下文相关的词向量. 载于 Advances in Neural Information Processing Systems，第 6297–6308 页，2017 年。

[39] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. 词和短语的分布式表示及其组合性. 载于 Advances in neural information processing systems，第 3111–3119 页，2013 年。

[40] N. Mostafazadeh, M. Roth, A. Louis, N. Chambers, and J. Allen. Lsdsem 2017 共享任务：故事完形填空测试. 载于 Proceedings of the 2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics，第 46–51 页，2017 年。

[41] K. Nigam, A. McCallum, and T. Mitchell. 使用 EM 的半监督文本分类. Semi-Supervised Learning，第 33–56 页，2006 年。

[42] J. Pennington, R. Socher, and C. Manning. Glove：用于词表示的全局向量. 载于 Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)，第 1532–1543 页，2014 年。

[43] M. E. Peters, W. Ammar, C. Bhagavatula, and R. Power. 使用双向语言模型的半监督序列标注. ACL, 2017.

[44] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer. 深度上下文相关的词表示. NAACL, 2018.

[45] Y. Qi, D. S. Sachan, M. Felix, S. J. Padmanabhan, and G. Neubig. 预训练词嵌入何时以及为何对神经机器翻译有用？ NAACL, 2018.

[46] A. Rahman and V. Ng. 解决定代词的复杂情况：Winograd 模式挑战. 载于 Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning，第 777–789 页. Association for Computational Linguistics, 2012.

[47] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. Squad：用于文本机器理解的 100,000+ 个问题. EMNLP, 2016.

[48] P. Ramachandran, P. J. Liu, 和 Q. V. Le. 用于序列到序列学习的无监督预训练. arXiv 预印本 arXiv:1611.02683, 2016.

[49] M. Ranzato, C. Poultney, S. Chopra, 和 Y. LeCun. 使用基于能量的模型高效学习稀疏表示. 载于《神经信息处理系统进展》, 页 1137–1144, 2007.

[50] M. Rei. 用于序列标注的半监督多任务学习. ACL, 2017.

[51] H. Robbins 和 S. Monro. 一种随机近似方法. 《数理统计年鉴》, 页 400–407, 1951.

[52] T. Rocktäschel, E. Grefenstette, K. M. Hermann, T. Kociskˇ y, 和 P. Blunsom. 蕴含推理.\` 使用神经 Attention. arXiv 预印本 arXiv:1509.06664, 2015.

[53] R. Sennrich, B. Haddow, 和 A. Birch. 使用子词单元进行罕见词的神经机器翻译. arXiv 预印本 arXiv:1508.07909, 2015.

[54] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng, 和 C. Potts. 基于情感树库语义组合性的递归深度模型. 载于《2013 年自然语言处理经验方法会议论文集》, 页 1631–1642, 2013.

[55] S. Srinivasan, R. Arora, 和 M. Riedl. 一种简单有效的故事完形填空测试方法. arXiv 预印本 arXiv:1803.05547, 2018.

[56] S. Subramanian, A. Trischler, Y. Bengio, 和 C. J. Pal. 通过大规模多任务学习通用分布式句子表示. arXiv 预印本 arXiv:1804.00079, 2018.

[57] J. Suzuki 和 H. Isozaki. 使用十亿词级无标注数据的半监督序列标注与分割. 《ACL-08: HLT 论文集》, 页 665–673, 2008.

[58] Y. Tay, L. A. Tuan, 和 S. C. Hui. 用于自然语言推理的带有对齐因式分解的比较-传播架构. arXiv 预印本 arXiv:1801.00102, 2017.

[59] Y. Tay, L. A. Tuan, 和 S. C. Hui. 用于机器理解的多范围推理. arXiv 预印本 arXiv:1803.09074, 2018.

[60] J. Tian, Z. Zhou, M. Lan, 和 Y. Wu. Ecnu 在 semeval-2017 任务 1：利用基于核的传统 nlp 特征和神经网络构建多语言和跨语言语义文本相似度的通用模型. 载于《第 11 届语义评测国际研讨会论文集》, 页 191–197, 2017.

[61] Y. Tsvetkov. 处理低资源语言的机遇与挑战. CMU, 2017.

[62] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, 和 I. Polosukhin. Attention is all you need. 载于《神经信息处理系统进展》, 页 6000–6010, 2017.

[63] P. Vincent, H. Larochelle, Y. Bengio, 和 P.-A. Manzagol. 使用去噪自编码器提取和组合鲁棒特征. 载于《第 25 届国际机器学习会议论文集》, 页 1096–1103. ACM, 2008.

[64] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, 和 S. R. Bowman. Glue：一个用于自然语言理解的多任务基准和分析平台. arXiv 预印本 arXiv:1804.07461, 2018.

[65] A. Warstadt, A. Singh, 和 S. R. Bowman. 语言可接受性语料库. http://nyu-mll.github.io/cola, 2018.

[66] A. Williams, N. Nangia, 和 S. R. Bowman. 一个通过推理进行句子理解的广泛覆盖挑战语料库. NAACL, 2018.

[67] Y. Xu, J. Liu, J. Gao, Y. Shen, 和 X. Liu. 迈向人类水平的机器阅读理解：多策略推理与推断. arXiv 预印本 arXiv:1711.04964, 2017.

[68] D. Yu, L. Deng, 和 G. Dahl. 预训练和微调在真实世界语音识别的上下文相关 dbn-hmms 中的作用. 载于《NIPS 深度学习与无监督特征学习研讨会论文集》, 2010.

[69] R. Zhang, P. Isola, 和 A. A. Efros. Split-brain 自编码器：通过跨通道预测进行无监督学习. 载于 CVPR, 卷 1, 页 6, 2017.

[71] Y. Zhu, R. Kiros, R. Zemel, R. Salakhutdinov, R. Urtasun, A. Torralba, 和 S. Fidler. 对齐书籍与电影：通过看电影和读书实现故事般的视觉解释. 载于《IEEE 国际计算机视觉会议论文集》, 页 19–27, 2015.

[70] X. Zhu. 半监督学习文献综述. 2005.