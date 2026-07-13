# TileBench 45 个 AI Kernel 简短介绍

本文按前面列出的分类和顺序整理 TileBench 中的 45 个 AI kernel。每个条目包含：来源、ELI5 三句话解释、极简伪代码、伪代码说明。

来源链接说明：

- TritonBench: [GitHub: thunlp/TritonBench](https://github.com/thunlp/TritonBench)
- LeetGPU: 论文引用为 `LeetGPU. 2025. Challenges.`，本地论文材料没有给出可核验的 GitHub 仓库链接；下文保留来源标注为 LeetGPU。

## Matrix Multiplication / Attention 类

### 1. flash-attention

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. 它是在做 Transformer 里的注意力，让每个 token 去看其他 token。
2. 普通 attention 会把很大的注意力矩阵完整存下来，flash-attention 尽量边算边用，不把大矩阵全放进显存。
3. 这样能省显存，也能减少慢速显存访问。

```text
for each batch, head:
  for each query block Qb:
    acc = 0
    norm = 0
    for each key/value block Kb, Vb:
      scores = Qb @ Kb.T / sqrt(d)
      probs = softmax_update(scores, norm)
      acc = update(acc, probs @ Vb)
    O[Qb] = acc
```

伪代码说明：外层按 batch/head 和 query block 分块。内层逐块读取 K/V，在线更新 softmax 的归一化项和输出累加器，避免显式 materialize 完整 `QK^T` 矩阵。

### 2. flash-decode

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. 它是大模型生成下一个 token 时用的 attention。
2. 这时 query 通常很短，但历史 K/V cache 很长。
3. 算子的重点是快速从长缓存里读信息，并把一个新 token 的答案算出来。

```text
for each batch, head, query:
  acc = 0
  norm = 0
  for each cache block Kb, Vb:
    scores = query @ Kb.T / sqrt(d)
    probs = softmax_update(scores, norm)
    acc = update(acc, probs @ Vb)
  out = acc
```

伪代码说明：和 flash-attention 类似，但 decode 场景的 query 数量很少，瓶颈常在长 K/V cache 的 streaming 读取、mask 和归约上。

### 3. block-sparse-attn

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. 它也是 attention，但不是每个 token 都看所有 token。
2. 它只看预先允许的若干块，就像只翻相关章节而不是整本书。
3. 这样可以减少无用计算，但访问模式会更不规则。

```text
for each query block Qb:
  acc = 0
  for block_id in sparse_layout[Qb]:
    Kb = K[block_id]
    Vb = V[block_id]
    scores = Qb @ Kb.T
    acc += softmax(scores) @ Vb
  O[Qb] = acc
```

伪代码说明：核心多了一张 sparse layout 表。每个 query block 只访问表中列出的 K/V block，因此计算量降低，但索引和加载不再是完全连续规则的。

### 4. matmul_fp32_fp16_fp8

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. 它做标准矩阵乘法，也就是 `C = A x B`。
2. 它覆盖 FP32、FP16、FP8 等不同精度路径。
3. 对 GPU 来说，这是最经典的 Tensor Core 友好型工作负载。

```text
for i in blocks(M):
  for j in blocks(N):
    acc = 0
    for k in blocks(K):
      acc += A[i, k] @ B[k, j]
    C[i, j] = acc
```

伪代码说明：把 M/N/K 三个维度切成 tile。每个输出 tile 通过遍历 K 维 tile 累加得到，实际 GPU 实现会尽量让 `@` 映射到 Tensor Core/MMA 指令。

### 5. matmul-int8

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. 它也是矩阵乘法，但输入里有低比特或 INT8 量化数据。
2. 量化数据更省空间，但计算前通常要解包、缩放或转换。
3. 难点是别让解包和数据搬运吃掉矩阵乘法本该省下来的时间。

```text
for each output tile Cij:
  acc = 0
  for each k tile:
    A_tile = load_int8(A)
    B_tile = unpack_lowbit(B_packed)
    acc += mma(A_tile, B_tile)
  Cij = scale_or_cast(acc)
```

伪代码说明：和普通 GEMM 的骨架相同，但 B 或 A 可能是 packed 低比特格式。伪代码中的 `unpack_lowbit` 是性能关键，因为它可能引入额外寄存器、shared memory 和重排开销。

### 6. streamk-matmul

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. 它还是矩阵乘法，但重点在任务怎么切给不同 GPU blocks。
2. Stream-K 会把 K 维工作拆得更细，让更多计算单元同时忙起来。
3. 它适合处理普通分块 GEMM 负载不均或尾部利用率不好的情况。

```text
split output tiles and K ranges into work units
for each work unit:
  partial = A_tile @ B_tile
  write partial sum
reduce partial sums into final C tile
```

伪代码说明：普通 GEMM 通常一个 CTA 负责一个输出 tile 的完整 K 累加。Stream-K 允许多个 CTA 分摊同一个输出 tile 的 K 维片段，最后再合并 partial sums。

### 7. linear-self-attn

来源：LeetGPU

ELI5：
1. 它是一种线性 attention，想把传统 attention 的平方复杂度降下来。
2. 它通常会先对 Q/K 做特征变换，再用可结合的方式重排计算。
3. 这样长序列时更省，但公式和数据流比普通矩阵乘更绕。

```text
Qp = feature_map(Q)
Kp = feature_map(K)
KV = Kp.T @ V
den = Qp @ sum(Kp)
O = (Qp @ KV) / den
```

伪代码说明：传统 attention 先形成 `QK^T`，linear attention 用 feature map 把计算改写为先聚合 `K^T V`，再与 Q 相乘。这样避免显式构造长度平方的 attention 矩阵。

### 8. batched-matmul

来源：LeetGPU

ELI5：
1. 它一次做很多个小矩阵乘法。
2. 每个 batch 都有自己的 A、B、C。
3. 重点是让 GPU 同时处理很多独立矩阵，减少调度浪费。

```text
for b in batches:
  for i in blocks(M):
    for j in blocks(N):
      acc = 0
      for k in blocks(K):
        acc += A[b, i, k] @ B[b, k, j]
      C[b, i, j] = acc
```

伪代码说明：比普通 GEMM 多了 batch 维。每个 batch 的矩阵乘互不依赖，GPU 可以把 batch 维和输出 tile 维一起映射到并行 block。

## Reduction / Normalization 类

### 9. softmax

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. softmax 把一排数字变成一排概率。
2. 大数字会得到更高概率，小数字会被压低。
3. 为了数值稳定，通常先减掉最大值再指数化。

```text
for each row x:
  m = max(x)
  e = exp(x - m)
  y = e / sum(e)
```

伪代码说明：一行 softmax 包含 max reduction、exp、sum reduction 和逐元素除法。GPU 实现通常让一个或多个 block 负责一行或一段行。

### 10. layernorm

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. layernorm 把每一行特征调整到更稳定的尺度。
2. 它先算这一行的平均值和方差。
3. 然后把每个元素减均值、除标准差，再乘权重加偏置。

```text
for each row x:
  mean = sum(x) / N
  var = sum((x - mean)^2) / N
  y = (x - mean) / sqrt(var + eps) * gamma + beta
```

伪代码说明：核心是每行两次归约：一次求均值，一次求方差。最后的归一化和 affine 变换是逐元素操作。

### 11. rmsnorm

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. rmsnorm 像 layernorm 的简化版。
2. 它不减平均值，只看平方平均的大小。
3. 这样计算更轻，在很多 LLM 里很常见。

```text
for each row x:
  rms = sqrt(sum(x * x) / N + eps)
  y = x / rms * weight
```

伪代码说明：RMSNorm 只需要一轮平方和归约，然后做逐元素缩放。它省掉了 mean 和 variance 的部分计算。

### 12. cross-entropy

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. cross-entropy 衡量模型预测和正确答案差多远。
2. 正确类别概率越高，loss 越小。
3. 它常和 softmax 合在一起算，避免中间概率不稳定。

```text
for each sample:
  m = max(logits)
  logsum = log(sum(exp(logits - m))) + m
  loss = logsum - logits[target]
```

伪代码说明：这是 log-softmax 加负对数似然的稳定写法。它先做 max 和 sum-exp 归约，再取目标类别的 logit 计算 loss。

### 13. kl-divergence

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. KL divergence 衡量两个概率分布有多不一样。
2. 如果两个分布很像，值就小。
3. 它常用于蒸馏、概率模型或分布约束。

```text
loss = 0
for i in elements:
  loss += p[i] * (log(p[i]) - log(q[i]))
```

伪代码说明：每个元素贡献一个 `p * log(p/q)`，最后做归约求和。实现上要注意 `p=0` 或数值下溢等边界。

### 14. mean-reduction

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. 它把一堆数字求平均。
2. 对 GPU 来说，难点不是公式，而是很多线程怎么高效合并结果。
3. 数据越大，归约树和内存访问越重要。

```text
sum = 0
for x in input:
  sum += x
mean = sum / count(input)
```

伪代码说明：逻辑上是加总再除以元素数。GPU 里通常先在 block 内局部求和，再跨 block 合并。

### 15. argmax

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. argmax 找出一组数字里最大的那个位置。
2. 它不只要最大值，还要最大值的索引。
3. 如果有并列最大值，还要有一致的 tie-breaking 规则。

```text
best_val = -inf
best_idx = -1
for i, x in enumerate(input):
  if x > best_val:
    best_val = x
    best_idx = i
return best_idx
```

伪代码说明：这是带索引的 max reduction。并行实现会在每个线程块里维护 `(value, index)` 对，再合并局部最优。

### 16. l2-norm

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. L2 norm 衡量一个向量的长度。
2. 它先把每个元素平方，再全部加起来。
3. 最后开平方，得到欧几里得长度。

```text
sum_sq = 0
for x in input:
  sum_sq += x * x
norm = sqrt(sum_sq)
```

伪代码说明：核心是平方和归约。相比 mean reduction，它多了逐元素平方和最后的平方根。

### 17. histogramming

来源：LeetGPU

ELI5：
1. histogramming 是数数：每个值落在哪个桶里，就给那个桶加一。
2. 比如统计图片里每种灰度出现多少次。
3. GPU 上难点是很多线程可能同时给同一个桶加数。

```text
hist = zeros(num_bins)
for x in input:
  bin = bucket(x)
  atomic_add(hist[bin], 1)
```

伪代码说明：每个输入元素映射到一个 bin。并行时多个线程会竞争同一个 `hist[bin]`，所以常需要 atomic 或先做局部 histogram 再合并。

### 18. batch-normalization

来源：LeetGPU

ELI5：
1. batch norm 用一批数据的统计量来归一化特征。
2. 它先算每个通道的均值和方差。
3. 然后把该通道的数据拉回稳定尺度，再乘 gamma 加 beta。

```text
for each channel c:
  mean = average(x[:, c, ...])
  var = average((x[:, c, ...] - mean)^2)
  y[:, c, ...] = (x[:, c, ...] - mean) / sqrt(var + eps) * gamma[c] + beta[c]
```

伪代码说明：归约维度通常跨 batch 和空间维度，但按 channel 分组。每个 channel 有独立的统计量和 affine 参数。

### 19. moe-topk-gating

来源：LeetGPU

ELI5：
1. MoE gating 是给每个 token 选择最合适的几个专家。
2. 它先看每个专家的分数，再挑 top-k。
3. 最后对这几个专家的分数做 softmax，得到路由权重。

```text
for each token:
  scores = gate[token, :]
  idx = topk(scores, k)
  weights = softmax(scores[idx])
  output_indices[token] = idx
  output_weights[token] = weights
```

伪代码说明：这个算子结合了 top-k selection 和小范围 softmax。输出既包括专家编号，也包括分配给专家的概率权重。

## Point-wise 类

### 20. rope-embedding

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. RoPE 给 token 的向量加入位置信息。
2. 它把向量的相邻维度成对旋转，位置不同旋转角度不同。
3. 这样模型能感知 token 在序列里的相对位置。

```text
for each position p:
  for each dim pair (a, b):
    cosv = cos(freq[p, pair])
    sinv = sin(freq[p, pair])
    out[a] = x[a] * cosv - x[b] * sinv
    out[b] = x[a] * sinv + x[b] * cosv
```

伪代码说明：RoPE 是逐元素/逐维度对操作，不需要大归约。关键是按位置和维度读取正确的 sin/cos 参数。

### 21. vector-addition

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. 它把两个同样长的数组逐元素相加。
2. 第 i 个输出只依赖两个输入的第 i 个元素。
3. 这是最基础的 GPU pointwise kernel。

```text
for i in range(N):
  out[i] = x[i] + y[i]
```

伪代码说明：每个元素完全独立，非常适合并行。性能通常受显存带宽和 kernel launch 开销影响。

### 22. mul2

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. 它做逐元素乘法。
2. 每个输出元素等于对应输入元素相乘，或者把一个输入乘以常数。
3. 它和 vector-addition 一样，是轻量 pointwise 算子。

```text
for i in range(N):
  out[i] = x[i] * y[i]
```

伪代码说明：每个位置独立，没有跨元素通信。GPU 实现重点是连续读取、合并写回和减少额外开销。

### 23. relu

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. ReLU 是神经网络里常见的激活函数。
2. 它把负数变成 0，正数保持不变。
3. 像一个简单的阀门，只让正信号通过。

```text
for i in range(N):
  out[i] = max(x[i], 0)
```

伪代码说明：这是逐元素比较和选择。没有归约或复杂索引，通常是内存带宽型 kernel。

### 24. quantize-global

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. quantize 把高精度数字压缩成低精度表示。
2. 它通常会先除以 scale，再四舍五入和截断。
3. 这样模型权重或激活更省显存，也可能跑得更快。

```text
for i in range(N):
  q = round(x[i] / scale)
  out[i] = clamp(q, qmin, qmax)
```

伪代码说明：global quantization 使用全局 scale。每个元素独立处理，但要注意 rounding、clamp 和目标整数类型范围。

### 25. dequantize-rowwise

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. dequantize 是把低精度数还原成近似的浮点数。
2. rowwise 表示每一行可能有自己的 scale。
3. 它常用于压缩权重恢复到可计算的精度。

```text
for row in rows:
  s = scale[row]
  for col in cols:
    out[row, col] = q[row, col] * s
```

伪代码说明：每行读取一个 scale，然后对该行元素逐个乘回去。性能关键是 scale 复用和连续加载量化数据。

### 26. dropout

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. dropout 会随机关掉一部分神经元输出。
2. 训练时它让模型别太依赖某几个特征。
3. 没被关掉的元素通常会按概率缩放，保持期望不变。

```text
for i in range(N):
  keep = random(i, seed) > p
  out[i] = x[i] * keep / (1 - p)
```

伪代码说明：每个元素需要生成一个确定可复现的随机数。输出是 mask 后的输入，并用 `1/(1-p)` 做训练期缩放。

### 27. swiglu

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. SwiGLU 是 LLM 前馈网络里常用的门控激活。
2. 它把一部分输入做 SiLU 激活，再和另一部分相乘。
3. 可以理解为一个可学习的开关控制信息流。

```text
for i in range(N):
  gate = x1[i] * sigmoid(x1[i])
  out[i] = gate * x2[i]
```

伪代码说明：SwiGLU 常把输入拆成两半。第一半经过 SiLU，第二半作为被门控的值，二者逐元素相乘。

### 28. fused-activation

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. fused activation 把多个小操作合成一个 kernel。
2. 这样中间结果不用反复写回显存再读出来。
3. 它的目标是减少内存搬运，而不是改变数学结果。

```text
for i in range(N):
  t = affine_or_bias(x[i])
  out[i] = activation(t)
```

伪代码说明：伪代码中的 `affine_or_bias` 和 `activation` 代表被融合的小算子。融合后每个元素在寄存器里完成多个步骤，再一次性写回。

### 29. weight-dequant

来源：LeetGPU

ELI5：
1. 它把压缩存储的权重恢复成浮点权重。
2. 每个 tile 或 block 可能有自己的 scale。
3. 它常出现在量化模型推理的矩阵乘法前后。

```text
for each tile:
  s = scale[tile]
  for each packed value in tile:
    q = unpack(value)
    out = q * s
```

伪代码说明：和 rowwise dequant 类似，但 scale 的粒度是 tile。额外难点是 packed 数据的解包和 scale 对齐。

### 30. leaky-relu

来源：LeetGPU

ELI5：
1. Leaky ReLU 和 ReLU 很像。
2. 它不会把负数直接变成 0，而是保留一个很小的斜率。
3. 这样负数区域也能传递一点信号。

```text
for i in range(N):
  if x[i] >= 0:
    out[i] = x[i]
  else:
    out[i] = alpha * x[i]
```

伪代码说明：这是逐元素条件选择。相比 ReLU，负数分支不是 0，而是乘以 `alpha`。

### 31. sigmoid

来源：LeetGPU

ELI5：
1. sigmoid 把任意实数压到 0 到 1 之间。
2. 很大的正数接近 1，很小的负数接近 0。
3. 它常被用作概率或门控值。

```text
for i in range(N):
  out[i] = 1 / (1 + exp(-x[i]))
```

伪代码说明：每个元素独立计算一个指数函数。性能通常取决于特殊函数单元、近似实现和内存访问。

## Stencil / Convolution 类

### 32. 2d-conv

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. 2D convolution 用一个小滤波器在图片或特征图上滑动。
2. 每个输出点来自邻域输入和卷积核的加权求和。
3. 它像拿一个小窗口在图上扫，每扫一次算一个新像素。

```text
for n, oc, oh, ow:
  acc = 0
  for ic, kh, kw:
    acc += x[n, ic, oh+kh, ow+kw] * w[oc, ic, kh, kw]
  y[n, oc, oh, ow] = acc
```

伪代码说明：输出的每个空间位置都需要访问输入邻域。实际实现常把卷积变形成 im2col 风格的 tile，再用矩阵乘或类似 MMA 的路径计算。

### 33. jacobi-stencil-2d

来源：LeetGPU

ELI5：
1. Jacobi stencil 用周围邻居更新网格上的每个点。
2. 它常出现在数值模拟，比如热扩散。
3. 每个新点通常是上下左右和自己的某种平均。

```text
for i in 1..H-2:
  for j in 1..W-2:
    out[i, j] = (x[i,j] + x[i-1,j] + x[i+1,j] + x[i,j-1] + x[i,j+1]) / 5
```

伪代码说明：这是典型 stencil 访问：每个输出依赖固定形状的邻域。边界通常单独处理或保持不变。

### 34. 2d-max-pooling

来源：LeetGPU

ELI5：
1. max pooling 用小窗口在特征图上滑动。
2. 每个窗口只保留最大的值。
3. 它可以缩小特征图，同时保留最强响应。

```text
for n, c, oh, ow:
  best = -inf
  for kh, kw in pool_window:
    best = max(best, x[n, c, oh*stride+kh, ow*stride+kw])
  y[n, c, oh, ow] = best
```

伪代码说明：每个输出点是一个窗口内的 max reduction。与卷积相比，它没有乘加权重，只做比较。

### 35. gaussian-blur

来源：LeetGPU

ELI5：
1. Gaussian blur 是给图片做平滑模糊。
2. 它用周围像素的加权平均替换当前像素。
3. 越靠近中心的像素权重通常越大。

```text
for y, x in image:
  acc = 0
  for ky, kx in kernel:
    acc += image[y+ky, x+kx] * gaussian[ky, kx]
  out[y, x] = acc
```

伪代码说明：这是固定卷积核的 2D stencil/conv。权重来自 Gaussian kernel，边界位置需要 padding、clamp 或跳过。

### 36. 1d-conv

来源：LeetGPU

ELI5：
1. 1D convolution 在一维序列上滑动小窗口。
2. 每个输出是附近几个输入和卷积核的加权和。
3. 它常用于音频、时间序列或序列特征处理。

```text
for out_pos in range(L_out):
  acc = 0
  for k in range(K):
    acc += x[out_pos + k] * w[k]
  y[out_pos] = acc
```

伪代码说明：和 2D conv 相同思想，但只有一个空间维度。窗口越大，每个输出需要的加载和乘加越多。

### 37. 3d-conv

来源：LeetGPU

ELI5：
1. 3D convolution 在三维体数据或视频特征上滑动窗口。
2. 每个输出点看的是深度、高度、宽度三个方向的邻域。
3. 它比 2D convolution 搬更多数据，也更吃计算和缓存。

```text
for od, oh, ow:
  acc = 0
  for kd, kh, kw:
    acc += x[od+kd, oh+kh, ow+kw] * w[kd, kh, kw]
  y[od, oh, ow] = acc
```

伪代码说明：3D conv 的邻域是立方体。访存复用和 tile 设计更重要，否则会反复读取相邻体素。

## Data Layout 类

### 38. destindex

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. destindex 是按目标索引搬数据。
2. 每个元素不一定写到连续位置，而是写到索引指定的位置。
3. 它像给每个包裹贴上目的地，再按地址投递。

```text
for i in range(N):
  dst = index[i]
  out[dst] = x[i]
```

伪代码说明：这是 scatter/gather 类 layout 操作。性能难点是不规则写入可能导致非合并访存和写冲突。

### 39. matrix-transpose

来源：TritonBench ([GitHub](https://github.com/thunlp/TritonBench))

ELI5：
1. matrix transpose 把矩阵行列互换。
2. 原来的第 i 行第 j 列，变成第 j 行第 i 列。
3. 数学很简单，但 GPU 上要小心读写方向造成的非连续访问。

```text
for i in range(M):
  for j in range(N):
    out[j, i] = x[i, j]
```

伪代码说明：读和写有一边会跨 stride。高性能实现通常用 tile 暂存，保证读写尽量合并。

### 40. top-k-selection

来源：LeetGPU

ELI5：
1. top-k selection 找出一组数里最大的 k 个。
2. 它不一定要把全部数据完整排序。
3. 只关心前 k 个时，可以比完整 sort 更省。

```text
top = empty_min_heap(k)
for x in input:
  if top not full or x > min(top):
    insert_or_replace_min(top, x)
return sorted(top)
```

伪代码说明：概念上维护一个大小为 k 的候选集合。GPU 实现可能用分块局部 top-k，再把局部结果合并。

### 41. bitonic-sort

来源：LeetGPU

ELI5：
1. bitonic sort 是一种适合并行硬件的排序网络。
2. 它通过固定模式反复比较和交换元素。
3. 它不需要复杂分支，很适合小块并行排序。

```text
for size in powers_of_two:
  for stride in descending_powers(size):
    for i in range(N):
      j = i xor stride
      compare_and_swap(a[i], a[j], direction)
```

伪代码说明：bitonic sort 的比较配对由 `xor stride` 决定。每一轮 compare-and-swap 都能大量并行执行。

### 42. radix-sort

来源：LeetGPU

ELI5：
1. radix sort 按数字的二进制位分多轮排序。
2. 每轮只看某几位，把元素分桶。
3. 对整数排序很常见，因为它不依赖大小比较链。

```text
for shift in bit_groups:
  counts = histogram((x >> shift) & mask)
  offsets = prefix_sum(counts)
  scatter x into temp by bucket offsets
  swap(x, temp)
```

伪代码说明：每一轮包含 histogram、prefix sum 和 scatter。它把整数按低位到高位逐步整理成全局有序。

### 43. matrix-copy

来源：LeetGPU

ELI5：
1. matrix copy 把一个矩阵原样复制到另一个地方。
2. 每个输出元素等于对应输入元素。
3. 它看起来简单，但能测试纯内存带宽和访问合并。

```text
for i in range(M):
  for j in range(N):
    out[i, j] = x[i, j]
```

伪代码说明：这是没有计算负担的数据搬运。性能主要取决于读取和写入是否连续、是否对齐、是否充分并行。

### 44. reverse-array

来源：LeetGPU

ELI5：
1. reverse-array 把数组顺序倒过来。
2. 第一个元素去最后一个位置，最后一个元素去第一个位置。
3. 它是简单但有反向索引的数据重排。

```text
for i in range(N):
  out[N - 1 - i] = x[i]
```

伪代码说明：每个元素独立搬运，但读写方向相反。若原地反转，还需要避免两个线程同时交换同一对元素。

### 45. interleave

来源：LeetGPU

ELI5：
1. interleave 把两个数组交错合成一个数组。
2. 一个元素来自 A，下一个元素来自 B，像拉拉链一样合在一起。
3. 它常用于数据布局转换或打包。

```text
for i in range(N):
  out[2*i] = a[i]
  out[2*i + 1] = b[i]
```

伪代码说明：每个输入位置产生两个输出写入。访问模式规则，但输出 stride 是 2，需要注意写入合并和对齐。
