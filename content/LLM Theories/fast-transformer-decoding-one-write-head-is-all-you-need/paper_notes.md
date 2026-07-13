# Fast Transformer Decoding: One Write-Head is All You Need 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Noam Shazeer

**发表期刊/会议 (Journal/Conference)**: ArXiv

**发表年份 (Publication Year)**: 2019

**研究机构 (Affiliations)**: Google

---

## 1. 摘要

**目的**

- 解决 Transformer 模型在增量推理阶段因无法并行化导致的速度瓶颈。
- 消除反复加载大型 **Keys** 和 **Values** 张量带来的高内存带宽开销。

---

**方法**

- 提出 **Multi-query Attention** 架构变体。
- 核心机制：
  - 保留 **Queries** 的多头机制。
  - 让所有不同的 Attention heads 共享单组 **Keys** 和 **Values**。
  - 移除 K、V 张量中的 "heads" 维度。
- 参数控制：通过扩大 Feed-forward 层的隐藏层维度（从 4096 增至 5440），保持模型总参数量与基线一致。
- 正交性验证：结合 Local-attention 机制进行对比实验，验证两种优化方法的正交性。

---

**结果**

- 模型质量评估（WMT14 EN-DE 翻译与 Billion-Word 语言建模）：
  - **Multi-query Attention** 模型的 Perplexity 和 BLEU 分数略低于基线，但远优于减少 head 数量或降低维度的替代方案。
  - 在 Beam-4 解码下，**Multi-query** 模型取得最高 BLEU 分数（28.5）。
- 推理速度对比（序列长度 128，TPUv2 微秒/Token）：

| Attention Type | Training | Inference (enc. + dec.) | Beam-4 Search (enc. + dec.) |
| :--- | :--- | :--- | :--- |
| multi-head | 13.2 | 1.7 + 46 | 2.0 + 203 |
| multi-query | **13.0** | 1.5 + 3.8 | 1.6 + 32 |
| multi-head local | 13.2 | 1.7 + 23 | 1.9 + 47 |
| multi-query local | **13.0** | **1.5 + 3.3** | **1.6 + 16** |

- 性能提升：Decoder 增量推理速度提升超 10 倍（从 46μs 降至 3.8μs），内存访问与算术运算的比率从 $\Theta(\frac{n}{d} + \frac{1}{b})$ 降至 $\Theta(\frac{1}{d} + \frac{n}{dh} + \frac{1}{b})$。

---

**结论**

- **Multi-query Attention** 成功打破了 Transformer 增量推理的内存带宽限制。
- 在仅产生极小质量损失的前提下，实现了推理速度的大幅飞跃。
- 该架构使得基于 Attention 的序列模型更适用于对推理性能要求极高的实际应用场景。

---

## 2. 背景知识与核心贡献

**研究背景与动机**

- **Transformer** 模型作为 RNN 的替代方案，在序列建模中广泛应用，其核心依赖 **Attention** 机制进行序列内和序列间的信息交互。
- 在模型训练阶段，由于可以跨序列长度进行并行计算，速度较快且高效。
- 在 **Incremental Inference**（如自回归语言模型生成）阶段，由于数据依赖导致无法并行计算，推理速度成为主要挑战。
- 现代计算硬件（如 **GPU/TPU**）的计算能力远超内存带宽，而增量推理的瓶颈正是 **Memory Bandwidth** 成本。
- 每次解码步骤都需要重复加载庞大的 **Keys** 和 **Values** 张量以更新 Attention 状态，导致内存访问与计算操作的比例严重失衡，极大限制了推理速度。

---

**核心贡献**

- 提出 **Multi-query Attention** 架构变体，作为传统 **Multi-head Attention** 的高效替代方案。
- 核心改进在于让所有不同的 Attention **Heads** 共享单组 **Keys** 和 **Values**，仅保留 **Queries** 和输出的多头维度。
- 该设计大幅减小了增量解码中需重复加载的 **Keys** 和 **Values** 张量体积，显著降低了 **Memory Bandwidth** 需求。
- 实验验证表明，采用 **Multi-query Attention** 的模型在大幅提升解码速度的同时，仅产生极小的模型质量下降，且效果远优于直接减少 **Heads** 数量或降低 **Keys/Values** 维度的简单降维方案。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**核心架构：Multi-query Attention**

本文提出了一种针对 Transformer 增量推理优化的架构变体——**Multi-query Attention**。该架构旨在解决标准 **Multi-head Attention** 在自回归生成时，因反复加载庞大的 **Keys** 和 **Values** 张量而导致的 **Memory Bandwidth** 瓶颈问题。

---

**架构设计与实现细节**

- **共享 Key/Value 机制**：与标准 Multi-head Attention 中每个 Head 拥有独立的 Keys 和 Values 不同，Multi-query Attention 让所有不同的 Attention Heads 共享单一的一组 Keys 和 Values。
- **保留多头 Query**：Queries 依然保持多头特性，确保模型具备从不同表示子空间提取信息的能力。
- **代码层面修改**：在基于 `tf.einsum` 的实现中，针对 $K$, $V$, $P_k$, $P_v$ 的张量运算移除代表 Heads 维度的 "h" 维度。
- **参数量补偿**：为了与 Baseline 模型保持相同的总参数量，Multi-query 模型拓宽了 Feed-Forward 层的隐藏层维度（例如在翻译任务中从 4096 增至 5440）。

---

**性能分析对比**

通过简化假设（$m=n$, $k=v=d/h$, $n \leq d$），对增量推理的性能瓶颈进行理论分析：

- **计算复杂度**：Multi-query Attention 的总计算量保持不变，为 $\Theta(bnd^2)$。
- **内存访问量**：标准 Multi-head Attention 的总内存访问量为 $\Theta(bn^2d + nd^2)$；Multi-query Attention 将其降低至 $\Theta(bnd + bn^2k + nd^2)$。
- **访存/计算比率**：标准架构的比率为 $\Theta(\frac{n}{d} + \frac{1}{b})$，在 $n \approx d$ 或 $b \approx 1$ 时成为严重瓶颈；Multi-query Attention 将比率优化为 $\Theta(\frac{1}{d} + \frac{n}{dh} + \frac{1}{b})$，成功将关键的 $\frac{n}{d}$ 项缩小了 $h$ 倍。

---

**实验配置与结果**

- **任务与基准**：在 WMT 2014 English-German 翻译任务和 Billion-Word Language Modeling Benchmark 上进行评估。
- **模型质量**：Multi-query 模型的 BLEU 和 Perplexity 指标略低于 Baseline，但显著优于通过减少 Heads 数量（$h$）或降低 Key/Value 维度（$d_k, d_v$）来压缩张量的替代方案。
- **推理速度提升**：在 TPUv2 上的测试表明，Multi-query 架构大幅提升了增量解码速度。

**WMT14 EN-DE 推理速度对比 (单位: TPUv2-microseconds/token)**

| Attention Type | Training | Inference (enc. + dec.) | Beam-4 Search (enc. + dec.) |
| :--- | :---: | :---: | :---: |
| multi-head | 13.2 | 1.7 + 46 | 2.0 + 203 |
| multi-query | **13.0** | 1.5 + **3.8** | 1.6 + **32** |
| multi-head local | 13.2 | 1.7 + 23 | 1.9 + 47 |
| multi-query local | **13.0** | **1.5 + 3.3** | **1.6 + 16** |

### 1. Multi-Query Attention

**核心原理**
- **Multi-Query Attention** 是 **Multi-head Attention** 的一种架构变体。
- 核心改进：在不同的 Attention **heads** 之间共享单一的一组 **Keys** 和 **Values**。
- **Queries** 和输出投影依然保留 **heads** 维度，确保模型具备多视角的查询能力。
- 目的：大幅减小 **K** 和 **V** tensors 的体积，降低 **Incremental Decoding** 阶段的 **Memory Bandwidth** 消耗。

---

**算法流程与参数设置**
- **参数维度对比**：
  - **Queries** 投影矩阵 `P_q`：维度为 `[h, d, k]`，保留 **heads** 维度 `h`。
  - **Keys** 投影矩阵 `P_k`：维度为 `[d, k]`，移除 **heads** 维度。
  - **Values** 投影矩阵 `P_v`：维度为 `[d, v]`，移除 **heads** 维度。
  - 输出投影矩阵 `P_o`：维度为 `[h, d, v]`，保留 **heads** 维度。
- **计算步骤**：
  - 生成 **Queries**：`Q = tf.einsum("bnd,hdk->bhnk", X, P_q)`
  - 生成共享 **Keys**：`K = tf.einsum("bmd,dk->bmk", M, P_k)`
  - 生成共享 **Values**：`V = tf.einsum("bmd,dv->bmv", M, P_v)`
  - 计算 **Logits**：`logits = tf.einsum("bhnk,bmk->bhnm", Q, K)`
  - 计算 **Weights**：`weights = tf.softmax(logits + mask)`
  - 计算中间输出：`O = tf.einsum("bhnm,bmv->bhnv", weights, V)`
  - 计算最终输出：`Y = tf.einsum("bhnv,hdv->bnd", O, P_o)`
- **参数量平衡策略**：
  - 由于移除了 **K** 和 **V** 的 **heads** 维度，模型总参数量减少。
  - 实验中通过扩大 Feed-Forward 层的隐藏层维度（如从 4096 扩大至 5440），使总参数量与 Baseline 保持一致。

---

**输入输出关系**
- **Batched 模式输入**：
  - `X`：维度 `[b, n, d]` 的 Query 输入序列。
  - `M`：维度 `[b, m, d]` 的 Memory 输入序列。
  - `mask`：维度 `[b, h, n, m]` 的掩码张量。
- **Batched 模式输出**：
  - `Y`：维度 `[b, n, d]` 的最终表示。
- **Incremental 模式输入输出**：
  - 输入当前步 `x` (`[b, d]`) 及历史状态 `prev_K` (`[b, m, k]`)、`prev_V` (`[b, m, v]`)。
  - 输出当前步结果 `y` (`[b, d]`) 及更新后的 `new_K` (`[b, m+1, k]`)、`new_V` (`[b, m+1, v]`)。

---

**在整体架构中的作用与性能影响**
- **解决 Memory Bandwidth 瓶颈**：
  - 传统 **Multi-head Attention** 在 **Incremental Decoding** 中，内存访问与算术运算的比例为 $\Theta(\frac{n}{d} + \frac{1}{b})$。当序列长度 $n$ 接近维度 $d$ 时，比例接近 1，导致硬件算力受限于内存带宽。
  - **Multi-Query Attention** 将该比例降至 $\Theta(\frac{1}{d} + \frac{n}{dh} + \frac{1}{b})$，将主要瓶颈项 $\frac{n}{d}$ 缩小了 $h$ 倍。
- **推理速度提升**：
  - 在 TPUv2 上的测试表明，Decoder 每步推理时间从 46$\mu s$ 骤降至 3.8$\mu s$。
  - Beam-4 Search 下的 Decoder 每步耗时从 203$\mu s$ 降至 32$\mu s$。
- **模型质量评估**：
  - 在 WMT14 EN-DE 翻译任务中，**Multi-Query Attention** 的 BLEU 分数与 Baseline 接近（Beam-4 下甚至达到最高 28.5）。
  - 显著优于直接减少 **heads** 数量 $h$ 或降低 $d_k, d_v$ 维度的替代方案。

---

**性能与质量对比数据**

| Attention Type | $d_{ff}$ | BLEU (dev) | BLEU (test, beam 4) | Inference (enc. + dec.) $\mu s$/token |
| :--- | :---: | :---: | :---: | :---: |
| multi-head | 4096 | **26.7** | 28.4 | 1.7 + 46 |
| multi-query | 5440 | 26.5 | **28.5** | 1.5 + **3.8** |
| multi-head (h=2, d=64) | 6784 | 26.2 | 27.9 | - |


---

## 4. 实验方法与实验结果

**实验设置**

研究团队在两个核心自然语言处理任务上验证了 **Multi-query Attention** 的有效性与效率，并严格控制变量以保证对比的公平性。

*   **机器翻译任务**
    *   **数据集**：WMT 2014 English-German 翻译数据集。
    *   **Baseline 模型**：6 层 Encoder-Decoder Transformer，参数量 **211M**。
    *   **模型配置**：$d_{model}=1024$，$d_{ff}=4096$，$h=8$，$d_k=d_v=128$，使用 learned positional embeddings，共享 token-embedding 与输出层权重。
    *   **训练配置**：训练 100,000 steps (约 20 epochs)，batch size 为 128 (包含 256-token 输入与 256-token 目标序列)。硬件使用 32-core TPUv3 集群，单模型训练耗时约 2 小时。
*   **语言建模任务**
    *   **数据集**：Billion-Word Language Modeling Benchmark。
    *   **模型配置**：6 层 Transformer-decoder，$d_{model}=1024$，$d_{ff}=8192$，$h=8$，$d_k=d_v=128$，参数量 **192M**。
    *   **训练配置**：训练 136K steps (10 epochs)，batch size 为 64K tokens，硬件同样使用 32-core TPUv3 集群，耗时约 3 小时。
*   **变量控制策略**
    *   在 **Multi-query** 模型中，网络所有的 Attention 层（包含 Encoder-self-attention, Decoder-self-attention, Encoder-decoder-attention）均被替换。
    *   为保持总参数量与 Baseline 一致，**Multi-query** 模型的 feed-forward hidden layers 维度从 4096 扩大至 5440。
    *   为验证局部注意力机制的正交性，额外训练了限制 Decoder-self-attention 仅关注当前及前 31 个位置的 local 模型版本。

---

**结果数据分析**

实验结果从模型生成质量与推理速度两个维度进行评估，证明了 **Multi-query Attention** 在几乎不损失质量的前提下实现了显著的推理加速。

*   **模型质量评估**
    *   **机器翻译**：在 WMT14 EN-DE 数据集上，**Multi-query** 模型的 BLEU 得分与 Baseline 极为接近。在使用 Beam-4 解码时，**Multi-query** 模型甚至取得了最高的 BLEU 得分 (28.5)，超越 Baseline 的 28.4。
    *   **语言建模**：在 Billion-Word 基准测试中，**Multi-query** 模型的 per-word perplexity 仅略高于 Baseline，但远优于其他缩减维度的变体。

| Attention Type | $h$ | $d_k, d_v$ | $d_{ff}$ | ln(PPL) (dev) | BLEU (dev) | BLEU (test) beam 1 / 4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **multi-head** | 8 | 128 | 4096 | **1.424** | **26.7** | 27.7 / 28.4 |
| **multi-query** | 8 | 128 | 5440 | 1.439 | 26.5 | 27.5 / **28.5** |
| multi-head local | 8 | 128 | 4096 | 1.427 | 26.6 | 27.5 / 28.3 |
| multi-query local | 8 | 128 | 5440 | 1.437 | 26.5 | 27.6 / 28.2 |

*   **推理速度评估**
    *   测试硬件为单块 TPUv2 (8 cores)，序列长度设为 128。
    *   **训练速度**：**Multi-query** 模型与 Baseline 训练耗时几乎一致 (13.0 vs 13.2 $\mu s$/token)。
    *   **增量推理速度**：**Multi-query** 模型的 Decoder 推理速度获得极大提升。Baseline 的 Decoder 耗时 46 $\mu s$/token，而 **Multi-query** 仅需 3.8 $\mu s$/token，实现了超过 **12倍** 的加速。
    *   **Beam-4 搜索速度**：在 Beam-4 解码场景下，**Multi-query** 的 Decoder 耗时从 203 $\mu s$/token 骤降至 32 $\mu s$/token，加速比超过 **6倍**。

| Attention Type | Training | Inference (enc. + dec.) | Beam-4 Search (enc. + dec.) |
| :--- | :---: | :---: | :---: |
| **multi-head** | 13.2 | 1.7 + 46 | 2.0 + 203 |
| **multi-query** | **13.0** | 1.5 + 3.8 | 1.6 + 32 |
| multi-head local | 13.2 | 1.7 + 23 | 1.9 + 47 |
| multi-query local | **13.0** | **1.5 + 3.3** | **1.6 + 16** |

*(注：时间单位为 TPUv2-microseconds per output token)*

---

**消融实验分析**

为了证明共享 Key 和 Value 是最优的参数缩减策略，论文对比了另一种直接降低内存访问量的替代方案：减少 head 数量 $h$ 或降低 Key/Value 维度 $d_k, d_v$。为保证参数量一致，这些变体同样扩大了 $d_{ff}$。

*   **对比变体设计**
    *   变体 1：$h=1$，$d_k=d_v=128$，$d_{ff}=6784$
    *   变体 2：$h=2$，$d_k=d_v=64$，$d_{ff}=6784$
    *   变体 3：$h=4$，$d_k=d_v=32$，$d_{ff}=6784$
    *   变体 4：$h=8$，$d_k=d_v=16$，$d_{ff}=6784$
*   **质量退化分析**
    *   所有减少 $h$ 或 $d_k, d_v$ 的变体均出现了显著的模型质量退化。
    *   在 WMT14 翻译任务中，变体 1 的 BLEU (dev) 下降至 25.8，ln(PPL) 升高至 1.518；而 **Multi-query** 的 BLEU 为 26.5，ln(PPL) 仅为 1.439。
    *   变体 2 ($h=2, d_k=64$) 虽然表现稍好 (BLEU 26.2)，但仍不及 **Multi-query**，且其并未像 **Multi-query** 那样在推理阶段完全消除 K/V 张量的 heads 维度，内存带宽优化效果有限。
*   **结论**
    *   简单地削减 Attention 头数或维度会破坏模型学习复杂表示的能力，导致严重的性能损失。
    *   **Multi-query Attention** 通过保留 Query 的多头多样性，同时共享 Key 和 Value，在突破内存带宽瓶颈与保持模型容量之间找到了最佳平衡点。

---

