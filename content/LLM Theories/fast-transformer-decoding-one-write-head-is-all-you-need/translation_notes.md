# Fast Transformer Decoding: One Write-Head is All You Need 原文翻译

# 快速 Transformer 解码：一个写头足矣

*Noam Shazeer Google noam@google.com*

## 摘要

Transformer 神经序列模型中使用的 Multi-head attention 层，是 RNN 在序列内和序列间移动信息的强大替代方案。尽管由于可以跨序列长度进行并行化，这些层的训练通常快速且简单，但增量推理（由于无法进行这种并行化）通常很慢，这是由于反复加载大型 "keys" 和 "values" 张量会带来内存带宽开销。我们提出了一种称为 multi-query attention 的变体，其中 keys 和 values 在所有不同的 attention "heads" 之间共享，大大减小了这些张量的尺寸，从而降低了增量解码对内存带宽的需求。我们通过实验验证了由此产生的模型确实可以更快地进行解码，并且与基线相比仅产生轻微的质量下降。

---

# 引言

Transformer 神经序列模型 (Vaswani et al. 2017) 已成为循环序列模型的一种流行替代方案。Transformer 依赖 attention 层在序列之间和序列内部传递信息。Transformer 的一个主要挑战是增量推理的速度。正如我们将要讨论的，在现代计算硬件上，增量 Transformer 推理的速度受限于重新加载大型 "keys" 和 "values" 张量所需的内存带宽，这些张量编码了 attention 层的状态。在接下来的章节中，我们将回顾 Transformer 使用的 multi-head-attention 层，进行性能分析，并提出一种架构变体，该变体极大地提高了推理速度，且仅有轻微的质量下降。

# 背景：Neural Attention

(Bahdanau, Cho, and Bengio 2014) 引入的 Neural Attention 是处理可变长度表示的强大工具。神经 attention 函数接收单个 query-vector $q$ 和一组 $m$ 个不同的 (key-vector, value-vector) 对（由矩阵 $K$ 和 $V$ 表示），并产生一个输出向量 $y$。输出 $y$ 的计算方式是对不同的 value 向量进行加权求和，其中权重是通过将 query 与 keys 进行比较得出的。

## Dot-Product Attention

以下代码描述了一种常见的公式表示，其中权重计算为 query 与不同 keys 的点积的 softmax。

``` python
def DotProductAttention(q, K, V):
  """Dot-Product Attention on one query.
  Args:
    q: a vector with shape [k]
    K: a matrix with shape [m, k]
    V: a matrix with shape [m, v]
  Returns:
    y: a vector with shape [v]
  """
  logits = tf.einsum("k,mk->m", q, K)
  weights = tf.softmax(logits)
  return tf.einsum("m,mv->v", weights, V)
```

我们的代码示例使用 TensorFlow 和 numpy 中定义的 **einsum** 表示法，用于任意维度张量之间的广义缩并。在这种表示法中，方程命名了输入和输出张量的维度。该计算在数值上等价于将每个输入广播为具有所有维度的并集，逐元素相乘，并对不在所需输出形状中的所有维度求和。

## Multi-head Attention

"Transformer" 序列到序列模型 (Vaswani et al. 2017) 并行使用 $h$ 个不同的 attention 层，作者将其称为 "Multi-head attention"。$h$ 个不同层的 query 向量由输入向量 $x$ 的 $h$ 个不同的学习线性投影 $P_q$ 导出。类似地，keys 和 values 由 $m$ 个不同输入向量的集合 $M$ 的 $h$ 个不同的学习线性投影 $P_k, P_v$ 导出。$h$ 个层的输出本身通过不同的学习线性投影 $P_o$，然后求和。为简单起见，我们给输入和输出向量相同的维度 $d$。该计算可以表示如下：

``` python
def MultiheadAttention(
    x, M, P_q, P_k, P_v, P_o):
  """Multi-head Attention on one query.
  Args:
    x: a vector with shape [d]
    M: a matrix with shape [m, d]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [h, d, k]
    P_v: a tensor with shape [h, d, v]
    P_o: a tensor with shape [h, d, v]
  Returns:
    y: a vector with shape [d]
  """
  q = tf.einsum("d,hdk->hk", x, P_q)
  K = tf.einsum("md,hdk->hmk", M, P_k)
  V = tf.einsum("md,hdv->hmv", M, P_v)
  logits = tf.einsum("hk,hmk->hm", q, K)
  weights = tf.softmax(logits)
  o = tf.einsum("hm,hmv->hv", weights, V)
  y = tf.einsum("hv,hdv->d", o, P_o)
  return y
```

注意：(Vaswani et al. 2017) 在 logits 上包含一个常数缩放因子。我们在代码中省略了这一点，因为它可以被合并到线性投影 $P_q$ 或 $P_k$ 中。

## Multi-head Attention (Batched)

在实践中，将多个 query 组合在一起进行批处理效率要高得多。下面的代码添加了两种类型的批处理。首先，我们从序列中的 $n$ 个不同位置生成 queries。这些 queries 都与相同的 keys 和 values 交互。此外，我们一次处理一批 $b$ 个不同的非交互序列。遵循 (Vaswani et al. 2017)，在自回归模型中，我们可以通过在 logits 中添加一个 "mask"（在非法位置包含值 $-\infty$）来防止信息向后流动。

``` python
def MultiheadAttentionBatched(
    X, M, mask, P_q, P_k, P_v, P_o):
  """Multi-head Attention.
  Args:
    X: a tensor with shape [b, n, d]
    M: a tensor with shape [b, m, d]
    mask: a tensor with shape [b, h, n, m]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [h, d, k]
    P_v: a tensor with shape [h, d, v]
    P_o: a tensor with shape [h, d, v]
  Returns:
    Y: a tensor with shape [b, n, d]
  """
  Q = tf.einsum("bnd,hdk->bhnk", X, P_q)
  K = tf.einsum("bmd,hdk->bhmk", M, P_k)
  V = tf.einsum("bmd,hdv->bhmv", M, P_v)
  logits = tf.einsum("bhnk,bhmk->bhnm", Q, K)
  weights = tf.softmax(logits + mask)
  O = tf.einsum("bhnm,bhmv->bhnv", weights, V)
  Y = tf.einsum("bhnv,hdv->bnd", O, P_o)
  return Y
```

### 批处理 Multi-head Attention 的性能分析

为了简化性能分析，我们将做出几个简化假设：

- $m = n$

- $k = v = \frac{d}{h}$，如 (Vaswani et al. 2017) 所建议

- $n \leq d$

算术运算的总数为 $\Theta(bnd^2)$。（鉴于上述简化假设，上述每个 `tf.einsum` 操作的复杂度为 $O(bnd^2)$。

要访问的内存总大小等于所有涉及张量的尺寸之和：$O(bnd + bhn^2 + d^2)$。第一项归因于 $X$、$M$、$Q$、$K$、$V$、$O$ 和 $Y$，第二项归因于 logits 和 weights，第三项归因于投影张量 $P_q$、$P_k$、$P_v$ 和 $P_o$。

将两者相除，我们发现内存访问与算术运算的比率为 $O(\frac{1}{k} + \frac{1}{bn})$。这种低比率对于在现代 GPU/TPU 硬件上获得良好性能是必要的，在这些硬件上，计算能力可能比内存带宽高出两个数量级。

## 多头注意力（增量式）

在某些设置中，数据依赖性使得无法并行处理来自多个位置的查询。一个例子是自回归语言模型（例如 Transformer (Vaswani et al. 2017)）中的自注意力层。在每个位置产生的查询会关注直到并包括该位置在内的所有位置产生的键值对。在训练期间，已知真实目标序列，我们可以使用类似于 2.3 节中的高效并行实现。然而，当从训练好的模型生成时，特定位置的自注意力层输出会影响下一个位置生成的 token，这反过来又影响下一个位置该层的输入。这阻碍了并行计算。增量计算此自注意力层的代码如下所示。

``` python
def MultiheadSelfAttentionIncremental(
    x, prev_K, prev_V, P_q, P_k, P_v, P_o):
  """Multi-head Self-Attention (one step).
  Args:
    x: a tensor with shape [b, d]
    prev_K: tensor with shape [b, h, m, k]
    prev_V: tensor with shape [b, h, m, v]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [h, d, k]
    P_v: a tensor with shape [h, d, v]
    P_o: a tensor with shape [h, d, v]
  Returns:
    y: a tensor with shape [b, d]
    new_K: tensor with shape [b, h, m+1, k]
    new_V: tensor with shape [b, h, m+1, v]
  """
  q = tf.einsum("bd,hdk->bhk", x, P_q)
  new_K = tf.concat(
    [prev_K, tf.expand_dims(tf.einsum("bd,hdk->bhk", M, P_k),axis=2)],
    axis=2)
  new_V = tf.concat(
    [prev_V, tf.expand_dims(tf.einsum("bd,hdv->bhv", M, P_v), axis=2)],
    axis=2)
  logits = tf.einsum("bhk,bhmk->bhm", q, new_K)
  weights = tf.softmax(logits)
  o = tf.einsum("bhm,bhmv->bhv", weights, new_V)
  y = tf.einsum("bhv,hdv->bd", O, P_o)
  return y, new_K, new_V
```

### 性能分析

我们做出与 2.3.1 节相同的简化假设。

在 $n$ 次调用中，总算术运算量再次为 $\Theta(bnd^2)$。

在 $n$ 次调用中，总内存访问量为 $\Theta(bn^2d + nd^2)$，第一项归因于 $K$ 和 $V$，第二项归因于 $P_q$、$P_k$、$P_v$ 和 $P_o$。

将内存访问量除以计算量，我们发现内存访问与算术运算的比率为 $\Theta(\frac{n}{d} + \frac{1}{b})$。当 $n \approx d$ 或 $b \approx 1$ 时，该比率接近 1，导致内存带宽成为现代计算硬件的主要性能瓶颈。为了使增量生成高效，我们必须将这两项都降低到 $\ll1$。$\frac{1}{b}$ 项更容易处理——只要内存大小允许，我们只需使用更大的批量大小即可。

降低 $\frac{n}{d}$ 项更为困难。这一项与每一步重新加载表示内存的 $K$ 和 $V$ 张量（其大小为 $bhmk=bn^2$）的开销有关。一种解决方案是限制序列长度 $n$。另一种是减少被关注的位置数量，可以通过关注局部邻域，或者通过其他方式压缩内存位置的数量来实现，如 (Liu et al. 2018)、(Zhang, Xiong, and Su 2018)、(Povey et al. 2018) 所述。在本文中，我们提出了一种正交的方法来减小 $K$ 和 $V$ 张量的大小——即移除它们的“头”维度，同时在查询中保留“头”维度。

# 多查询注意力

我们引入**多查询注意力**（multi-query Attention）作为 (Vaswani et al. 2017) 中描述的多头注意力的一种变体。多头注意力由多个并行的注意力层（头）组成，对查询、键、值和输出进行不同的线性变换。多查询注意力与此相同，只是不同的头共享单组键和值。（增量式）多查询（自）注意力的代码与上面列出的多头注意力代码相同，只是我们在 `tf.einsum` 等式中移除了字母 "h"，当它代表 $K$、$V$、$P_k$ 或 $P_v$ 的“头”维度时。

``` python
def MultiqueryAttentionBatched(
    X, M, mask, P_q, P_k, P_v, P_o):
  """Multi-Query Attention.
  Args:
    X: a tensor with shape [b, n, d]
    M: a tensor with shape [b, m, d]
    mask: a tensor with shape [b, h, n, m]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [d, k]
    P_v: a tensor with shape d, v]
    P_o: a tensor with shape [h, d, v]
  Returns:
    Y: a tensor with shape [b, n, d]
  """
  Q = tf.einsum("bnd,hdk->bhnk", X, P_q)
  K = tf.einsum("bmd,dk->bmk", M, P_k)
  V = tf.einsum("bmd,dv->bmv", M, P_v)
  logits = tf.einsum("bhnk,bmk->bhnm", Q, K)
  weights = tf.softmax(logits + mask)
  O = tf.einsum("bhnm,bmv->bhnv", weights, V)
  Y = tf.einsum("bhnv,hdv->bnd", O, P_o)
  return Y
```

``` python
def MultiquerySelfAttentionIncremental(
    x, prev_K, prev_V, P_q, P_k, P_v, P_o):
  """Multi-query Self-Attention (one step).
  Args:
    x: a tensor with shape [b, d]
    prev_K: tensor with shape [b, m, k]
    prev_V: tensor with shape [b, m, v]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [d, k]
    P_v: a tensor with shape [d, v]
    P_o: a tensor with shape [h, d, v]
  Returns:
    y: a tensor with shape [b, d]
    new_K: tensor with shape [b, m+1, k]
    new_V: tensor with shape [b, m+1, v]
  """
   q = tf.einsum("bd,hdk->bhk", x, P_q)
  K = tf.concat(
    [prev_K, tf.expand_dims(tf.einsum("bd,dk->bk", M, P_k), axis=2)],
    axis=2)
  V = tf.concat(
    [prev_V, tf.expand_dims(tf.einsum("bd,dv->bv", M, P_v), axis=2)],
    axis=2)
  logits = tf.einsum("bhk,bmk->bhm", q, K)
  weights = tf.softmax(logits)
  o = tf.einsum("bhm,bmv->bhv", weights, V)
  y = tf.einsum("bhv,hdv->bd", O, P_o)
  return y, K, V
```

## 增量式多查询注意力的性能分析

我们做出与 2.3.1 节相同的简化假设。

在 $n$ 次调用中，总算术运算量再次为 $\Theta(bnd^2)$。

在 $n$ 次调用中，总内存访问量为 $\Theta(bnd + bn^2k + nd^2)$，第一项归因于 $x$、$q$、$o$ 和 $y$，第二项归因于 $K$ 和 $V$，第三项归因于 $P_q$、$P_k$、$P_v$、$P_o$。

将内存访问量除以计算量，我们发现内存访问与算术运算的比率为 $\Theta(\frac{1}{d} + \frac{n}{dh} + \frac{1}{b})$。我们已经将棘手的 $\frac{n}{d}$ 减小了 $h$ 倍。理论上，给定较大的批量大小 $b$，这将显著提升增量生成的性能。在我们的实验部分，我们将展示性能提升是真实的，并且模型质量保持在高水平。

## 实验与结果

## 实验设置

遵循 (Vaswani et al. 2017)，我们在 WMT 2014 英德翻译任务上进行评估。作为基线，我们使用了一个 6 层的 encoder-decoder Transformer 模型，使用 $d_{model}=1024$ $d_{ff}=4096$，$h=8$，$d_k=d_v=128$，learned positional embeddings，以及 token-embedding 和输出层之间的权重共享。基线模型和所有变体都有 2.11 亿个参数。所有模型都训练了 100,000 步（约 20 个 epoch）。每个训练 batch 包含 128 个样本，每个样本由一个 256-token 的输入序列和一个 256-token 的目标序列组成（多个训练句子被拼接在一起以达到此长度）。模型在 32 核 TPUv3 集群上训练，每个模型训练大约需要 2 小时。我们使用了 tensor2tensor 和 mesh-tensorflow 库中的实现。使用的配置可以在 \[to be added before publication\] 中找到，包括关于学习率、dropout、label smoothing 等的详细信息。

在我们的“multi-query”模型中，我们将模型中的所有 attention 层替换为 multi-query attention。这包括 encoder-self-attention、decoder-self-attention 和 encoder-decoder-attention 层。我们将 feed-forward 隐藏层从 4096 扩展到 5440，以使总参数量与基线相等。

为了证明 local-attention 和 multi-query attention 是正交的，我们还训练了基线和 multi-query 模型的“local”版本，其中 decoder-self-attention 层（但不包括其他 attention 层）将 attention 限制在当前位置和前 31 个位置。

减小 $K$ 和 $V$ 大小的一种更简单的替代方法是减少头数 $h$ 和/或减少键和值的维度 $k$ 和 $v$。我们训练了几个这样的模型进行比较，同样通过扩展 feed-forward 隐藏层来使总参数量与基线相等。

我们在 Billion-Word Language Modeling Benchmark (Chelba et al. 2013) 上使用“transformer-decoder”语言模型进行了一组类似的实验。对于基线，我们使用了一个 6 层的模型，$d_{model}=1024$ $d_{ff}=8192$，$h=8$，$d_k=d_v=128$。基线和所有变体的总参数量为 1.92 亿。我们在 64K token 的 batch size 下训练了 136K 步（10 个 epoch）。同样，我们使用 32 核 TPUv3 集群训练每个模型大约 3 小时。

## 模型质量

表 1 展示了机器翻译实验的结果。我们使用贪婪最大似然解码对 dev 集进行解码，并使用 sacrebleu `"sacrebleu -t wmt13 -l en-de -tok intl"` 计算 BLEU 分数。我们还列出了 dev 集上每个 subword-token 的困惑度。根据这两个指标，multi-query attention 模型似乎比基线略差，但比任何涉及减少 $h$、$d_k$ 和 $d_v$ 的替代方案都要接近得多。

我们通过使用贪婪解码和 beam search（beam 4，$\alpha=0.6$）对测试集进行解码来验证结果，并使用 sacrebleu `"sacrebleu -t wmt14 -l en-de -tok intl"` 进行评估。同样，multi-query 模型的表现与基线相似，并且在 beam-4 解码中实际上获得了最高的 BLEU 分数（28.5）。

表 [tab:lm1b] 展示了 billion-word 语言建模基准测试的结果。模型通过 dev 集上的每个单词（而非每个 subword-token）的困惑度进行评估。结果描绘了与翻译结果相似的情况。multi-query attention 模型比基线略差，但明显优于任何涉及减少 $h$、$d_k$ 和 $d_v$ 的替代方案。

## 速度

表 2 展示了各种模型的训练和推理时间。训练和推理速度均在一个 TPUv2（8 核）上进行评估。如上所述，一个训练步（包含 32,768 个输入 token 和 32,768 个目标 token）对于基线模型耗时 433ms，对于 multi-query 模型耗时 425ms。除以 32,768，我们发现训练时间为每个（输入 token + 目标 token）13.2$\mu s$，如表 2 所列。

我们在一个包含 1024 个序列的 batch（每个核心 128 个）上运行了增量贪婪推理，使用 128 个 token 的源序列长度和 128 的目标序列长度。（由于需要固定形状的系统限制，我们在 decoder-self-attention 实现中使用了 padding 和 masking。因此，内存张量被填充到最大长度（128），或者在 local attention 的情况下填充到窗口大小（32）。因此，每个解码步骤花费相同的时间。一种增量增长张量的替代实现可以在序列开头附近节省时间。）对于基线模型，模型的 encoder 部分耗时 222ms，decoder 的每个增量步骤耗时 47ms。除以各自的 token 数量，我们发现 encoder 的摊销推理时间为每个 token $1.7\mu s$，而 decoder 的摊销推理时间则要大得多，为每个 token $46\mu s$，如表 2 所列。对于 multi-query 模型，encoder 耗时 195ms，decoder 每步耗时 3.9ms，摊销的每 token 成本分别为 $1.5\mu s$ 和 $3.8\mu s$。表 2 展示了这些值以及 beam-search 的类似结果。

|                   |     |            |          |           |          |                 |
|:-----------------:|:---:|:----------:|:--------:|:---------:|:--------:|:---------------:|
|     Attention     | $h$ | $d_k, d_v$ | $d_{ff}$ |  ln(PPL)  |   BLEU   |   BLEU (test)   |
|       类型        |     |            |          |   (dev)   |  (dev)   |   beam 1 / 4    |
|    multi-head     |  8  |    128     |   4096   | **1.424** | **26.7** |   27.7 / 28.4   |
|    multi-query    |  8  |    128     |   5440   |   1.439   |   26.5   | 27.5 / **28.5** |
| multi-head local  |  8  |    128     |   4096   |   1.427   |   26.6   |   27.5 / 28.3   |
| multi-query local |  8  |    128     |   5440   |   1.437   |   26.5   |   27.6 / 28.2   |
|    multi-head     |  1  |    128     |   6784   |   1.518   |   25.8   |                 |
|    multi-head     |  2  |     64     |   6784   |   1.480   |   26.2   |   26.8 / 27.9   |
|    multi-head     |  4  |     32     |   6784   |   1.488   |   26.1   |                 |
|    multi-head     |  8  |     16     |   6784   |   1.513   |   25.8   |                 |

WMT14 英德结果。

|                   |          |               |               |
|------------------:|:--------:|:-------------:|:-------------:|
|         Attention |   训练   |     推理      | Beam-4 搜索   |
|              类型 |          |  enc. + dec.  |  enc. + dec.  |
|        multi-head |   13.2   |   1.7 + 46    |   2.0 + 203   |
|       multi-query | **13.0** |   1.5 + 3.8   |   1.6 + 32    |
|  multi-head local |   13.2   |   1.7 + 23    |   1.9 + 47    |
| multi-query local | **13.0** | **1.5 + 3.3** | **1.6 + 16**  |

序列长度为 128 的 WMT14 英德翻译任务的摊销训练和推理成本。列出的值以每个输出 token 的 TPUv2 微秒为单位。

# 结论

我们提出了 multi-query attention——一种在增量设置中具有更低内存带宽需求的 multi-head attention 替代方案。我们相信这使得基于 attention 的序列模型能够在对推理性能要求严格的应用中得到更广泛的采用。

## 参考文献


Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 2014. “通过联合学习对齐与翻译的神经机器翻译。”



Chelba, Ciprian, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, and Phillipp Koehn. 2013. “用于衡量统计语言建模进展的十亿词基准。” *CoRR* abs/1312.3005. <http://arxiv.org/abs/1312.3005>.



Liu, Peter J, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer. 2018. “通过总结长序列生成维基百科。” 载于 *Proceedings of the International Conference on Learning Representations*。



Povey, Daniel, Hossein Hadian, Pegah Ghahremani, Ke Li, and Sanjeev Khudanpur. 2018. “用于 ASR 的时间受限自注意力层。” 载于 *Proceddings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*。IEEE。



Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. “Attention Is All You Need.” 载于 *NIPS*。



Zhang, Biao, Deyi Xiong, and Jinsong Su. 2018. “通过平均注意力网络加速神经 Transformer。”