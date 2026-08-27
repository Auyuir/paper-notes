# Eficient Top-K Query Processing on Massively Parallel Hardware 原文翻译

# 大规模并行硬件上的高效 Top-K 查询处理

Anil Shanbhag MIT anil@csail.mit.edu

Holger Pirk Imperial College London pirk@imperial.ac.uk

## 摘要

Samuel Madden MIT madden@csail.mit.edu

在许多数据分析工作负载中，一个常见的操作是寻找 top-k 项，即根据某种排序顺序找到最大或最小的操作（在 SQL 中通过 LIMIT 或 ORDER BY 表达式实现）。top-k 的一种简单实现是对所有项进行排序然后返回前 k 项，但这比实际所需的工作量大得多。尽管在传统多核处理器上已经探索了 top-k 的高效实现，但迄今为止尚未对 GPU 上的 top-k 实现进行过系统研究，尽管在 TensorFlow<sup>1</sup> 和 ArrayFire<sup>2</sup> 等 GPU 框架中已有对此类实现的公开需求。在这项工作中，我们提出了几种针对 GPU 的 top-k 算法，包括一种基于双调排序（bitonic sort）的新算法，称为双调 top-k（bitonic top-k）。对于高达 256 的 k 值，双调 top-k 算法比排序快 15 倍，比其他多种可能的实现快 4 倍。我们还开发了一个成本模型来预测我们几种算法的性能，并表明它能够准确预测现代 GPU 上的实际性能。

## CCS 概念

• 信息系统 → 查询算子；• 计算理论 → 大规模并行算法；• 计算机系统组织 → 异构（混合）系统；

## 关键词

GPU 的 Top-K 算法；双调 Top-K

## ACM 参考格式:

Anil Shanbhag, Holger Pirk, and Samuel Madden. 2018. Eficient Top-K Query Processing on Massively Parallel Hardware. In Proceedings of 2018 International Conference on Management of Data, Houston, TX, USA, June 10–15, 2018 (SIGMOD’18), 14 pages.   
https://doi.org/10.1145/3183713.3183735

## 1 引言

一种常见的分析型 SQL 查询涉及运行 top-k，即在给定排名函数的情况下找出 n 个元组中最高（或最低）的 k 个。top-k 查询的示例包括查询电商网站上最贵的产品、评论网站上评分最高的餐厅，或查询日志中性能最差的查询。Top-k 是计算机科学特别是数据管理中一个被广泛研究的问题，因为几乎每个数据分析系统都支持 top-k 计算（order-by/limit 子句）。该问题有许多实例和多种高效的解决方案（参见 [13] 的综述）。

<table><tr><td>Top-K</td><td>优先队列</td><td>???</td></tr><tr><td>排序</td><td>堆排序</td><td>双调排序</td></tr><tr><td colspan="2">串行</td><td>并行</td></tr></table>

图 1：Top-K 与排序的对偶性

寻找 top-k 元素的一种简单方法是对它们进行排序并返回前 k 个。然而，排序所做的工作超出了必要的范围，因为没有必要对 top-k 之外的元素进行排序。一种更好的方法是维护一个大小为 k 的优先队列（即最大堆），并在插入较大元素的同时移除较小元素。该方法的运行时间在 n log (k) 的数量级。该算法可以通过在逻辑上对数据进行分区来在 m 个处理器上并行化，让每个处理器计算每个分区的 top-k，并从 m 个分区堆中计算全局 top-k。虽然这种方法可以在多核处理器上高效实现（参见第 6.7 节），但它不适合大规模并行系统的单指令多线程执行模型<sup>3</sup>。随着最近对基于 GPU 的查询处理的关注 [3, 12, 14, 16, 19, 23]，显然需要一种高效的大规模并行算法来解决 top-k 问题。事实上，我们发现两个最主流的 GPU 编程框架（Tensorflow 和 Arrayfire）[1, 2] 已经有添加 top-k 算子的公开功能请求。

培养对该问题解的存在性甚至特征直觉的一种方法是考虑 top-k 和排序算法的对偶性。我们在图 1 中说明了这种对偶性：与优先队列对应的排序算法是堆排序。事实上，可以将堆排序视为构建一个 k = n 的优先队列，然后按排序顺序提取元素。当然，这隐藏了许多实现细节，但有助于形成直觉。在考虑大规模并行架构背景下的排序和 top-k 时，人们发现教科书上的大规模并行排序算法是双调排序。然而，目前还没有已知与双调排序相对应的 top-k 算法。不过，我们可以假设，就像双调排序一样，它很可能基于双调归并，并且需要结合一些低级优化，使其在计算和带宽上都高效。

在这项工作中，我们通过广泛研究 GPU 上现有的 top-k 解决方案，并开发一种针对大规模并行架构的新颖解决方案，从而系统地将这种直觉发展为一个可行的算法。我们发现它实际上是基于双调归并的，并将其称为双调 top-k。我们研究了 GPU 上其他几种潜在 top-k 算法的特性，包括排序和基于堆的算法，以及使用高位来查找顶部项的基于基数的算法。最终，我们发现对于高达 256 的 k 值，双调 top-k 比其他 top-k 方法快 4 倍，比排序快 15 倍。

这些新算法有潜力直接影响现代基于 GPU 的数据库系统的性能：我们所知所有的系统（PG Strom、Ocelot 和 MapD）目前都使用排序或将整个数据集传输到 CPU 进行 top-k 计算。因此，它们可以通过集成我们的算法直接获得我们方法带来的好处。虽然我们没有明确研究缓解 PCI-E 瓶颈的方法，但拥有一个高效的基于 GPU 的 top-k 算子将使这些系统通过 PCI-E 总线传输更少的数据，从而获得更高的性能。

总而言之，我们做出了以下贡献：

• 我们在各种基准测试上研究了多种不同 top-k 算法的性能特征，改变了数据集大小、k 的值、数据类型（整数与浮点数）以及数据的初始分布。

• 我们开发了一种新颖的、大规模并行的算法，用于高效评估 top-k 查询。

• 我们设计了许多优化（部分基于已知技术，部分完全新颖），并表明我们新的双调 top-k 算法通常优于所有其他算法，对于高达 256 的 k 值，通常有 4 倍或以上的性能提升。此外，我们展示了它对倾斜输入数据分布的鲁棒性。

• 我们证明了该算法能够集成到现有系统中（特别是 MapD）。

• 最后，我们为我们的双调 top-k 以及其他算法开发了详细的成本模型，并表明这些成本模型可以准确预测运行时间，这在查询规划器需要选择 top-k 实现时非常有价值。

在描述我们的算法、优化和实验的细节之前，我们首先讨论 GPU 性能以及它们现有的排序和 top-k 算法。

## 2 背景

## 2.1 GPU 数据访问

许多在 GPU 上编写的数据库操作仍然受限于内存子系统（共享内存或全局内存）[23] 的性能。因此，为了刻画不同算法在 GPU 上的性能，正确理解其内存层次结构至关重要。

图 2 展示了现代 GPU 的简化层次结构。层次结构中最低且最大的内存是全局内存。全局内存位于片外，在现代 GPU 上具有 150-920 GBps 的内存带宽。每个 GPU 都有若干被称为流多处理器 (Streaming Multiprocessors, SMs) 的计算单元。每个 SM 拥有多个核心和一组固定的寄存器。每个 SM 还拥有一个可被所有核心访问的共享内存，以及一个用于缓存对全局内存请求的 L1 缓存。来自 SM 对全局内存的访问会被缓存在 L2 缓存中。L2 缓存被所有流多处理器 (SM) 共享，且位于片上。

![](images/0f2818dd54f63b96b66a8cc0126e39a5ed0d16609fdcb7ad6029d25a6a33122e.jpg)  
图 2：GPU 内存层次结构

GPU 上的工作由大量线程完成，这些线程被组织成线程块（每个由一个 SM 运行）。线程块进一步被划分为 warp（通常为 32 个线程）。warp 中的线程以单指令多线程 (Single Instruction Multiple Threads, SIMT) 模型执行。设备会合并来自单个 warp 的全局内存加载和存储操作。当 warp 合并对全局内存的访问，导致相邻位置被访问时，可以达到最大带宽。

该编程模型允许用户在每个线程块中显式分配全局内存和共享内存。共享内存比全局内存快一个数量级。在 Nvidia Titan X Maxwell GPU 上，全局内存带宽约为 250 GBps，而共享内存带宽约为 3 TBps。同时，与数 GB 的全局内存相比，GPU 跨 SM 仅有几 MB 的共享内存。为了最大化性能，共享内存被组织为 32 个 bank，使得 warp 中的线程可以并行访问不同的内存 bank。然而，如果 warp 中的两个线程访问同一个内存 bank，就会发生 bank 冲突，对该 bank 的访问将被串行化。

最后，寄存器是内存层次结构中最快的一层。如果一个线程块需要的寄存器数量超过可用数量，寄存器值会溢出到线程本地内存。尽管名字如此，本地内存仅意味着它只能被该线程访问——它存储在 SM 之外的慢速全局内存中。

## 2.2 GPU 上的排序

多年来已有许多排序算法被提出。早期的实现通常基于双调排序 [6, 10, 17]。后来，基于基数的排序算法被提出，其性能优于双调排序 [15, 21, 22]。

双调排序 双调排序基于双调序列，即两个以相反方向排序的子序列的拼接。给定长度为 l = 2<sup>r</sup> 的双调序列 S，S 可以在 r 步内按升序（或降序）排序。在第一步中，比较元素对 (S[0] S[l/2]), (S[1] S[l/2 + 1]) (S[l/2 − 1] S[l − 1])，如果第二个元素小于第一个元素，则进行交换。这会产生两个双调序列，(S[0] S[l/2 − 1]) 和 (S[l/2] S[l − 1])，其中第一个子序列中的所有元素都小于第二个子序列中的任何元素。

![](images/67f25745b19e68e4554147f288150fd40c6ca378d1c3af19951f8432cf859dbb.jpg)  
(b) 示例数据（红色条按升序排序，蓝色条按降序排序）  
图 3：双调排序网络

在第二步中，对两个子序列应用相同的过程，产生四个双调序列。第一个子序列中的所有元素都小于第二个子序列中的任何元素，第二个子序列中的所有元素都小于第三个子序列中的任何元素，第三个子序列中的所有元素都小于第四个子序列中的任何元素。第三步、第四步、……、第 r 步依此类推。处理第 r 步会产生 $2 ^ { r }$ 个长度为 1 的子序列，因此序列 S 被排序。

设 A 为待排序的输入数组，并设 $n = 2 ^ { k }$ 为 A 的长度。对 A 进行排序的过程由 k 个阶段组成。长度为 $2 \ ( A [ 0 ] , A [ 1 ] ) , ( A [ 2 ] , A [ 3 ] ) , . . . , ( A [ n - 2 ] , A [ n - 1 ] )$ 的子序列根据定义是双调序列。在第一阶段，这些子序列被交替按升序和降序排序（如上所述）。这会创建长度为 4 的双调子序列，$( A [ 0 ] , A [ 1 ] , A [ 2 ] , A [ 3 ] ) , . . . , ( A [ n - 4 ] , A [ n - 3 ] , A [ n - 2 ] , A [ n - 1 ] )$ 。在第二阶段，这些长度为 4 的子序列被交替按升序和降序排序，产生长度为 8 的双调序列子序列。在双调排序的第 i 阶段，被排序的子序列总数为 $2 ^ { k - i }$ ，每个子序列的长度为 $2 ^ { i }$ ，因此第 i 阶段由 i 步组成。在第 (k-1) 阶段之后，数组 A 是一个双调序列。A 在最后一个阶段 k 中被排序。

在每一步中处理 n/2 个比较/交换操作。共有 loдn 个阶段，其中第 i 阶段有 i 步。因此，比较次数为 $O ( n l o g ^ { 2 } n )$ 。因此，在串行 CPU 上，双调排序比其他 O(nloдn) 排序算法慢。双调排序的优势在于它可以很容易地在 SIMT 和 SIMD 架构上并行化，并且需要更少的进程间通信。图 3a 展示了大小为 8 的任意序列的双调排序网络。这里有 $l o g _ { 2 } 8 = 3$ 个阶段，其中阶段 i 有 i 步。每一步由 $8 / 2 = 4$ 次比较组成。图 3a(a) 中的排序遵循上述过程：在阶段 1，比较元素 0 和 1 并按升序排序；元素 2 和 3 按降序排序；元素 4 和 5 按升序排序，依此类推。这些比较中的每一个都可以在单独的线程上并行完成。在阶段 1 结束时，得到 4 个长度为 2 的已排序序列。在阶段 2，步长为 2，首先比较元素 0 和 2 以及 1 和 3 并按降序排序，而 4 和 6 以及 5 和 7 按升序排序。

这些比较也可以并行完成。然后，执行步长为 1 的阶段 2，使得元素 0 和 1 以及 2 和 3 按降序排序，而 4 和 5 以及 6 和 7 按升序排序。这些比较同样被并行化。在阶段 2 结束时，我们剩下两个长度为 4 的已排序列表。最后，阶段 3 使用从 3 递减到 1 的步长合并这两个列表。

双调排序最快的实现是 Peters 等人 [17] 提出的实现。后面讨论的双调 top-k 算法复用了他们论文中的一些思想。

基数排序 基数排序基于将 k 位键重新解释为 d 位数字序列，并每次考虑其中一位。基本思想是，将 k 位数字拆分为更小的 d 位数字会得到足够小的基数 $\mathbf { r } = 2 ^ { d }$ ，使得键可以被划分到 r 个不同的桶中。由于每个数字的排序可以以与键数量 n 成线性的代价完成，整个排序过程的时间复杂度为 $O ( \lceil k / d \rceil n )$ 。对键的数字进行迭代可以从最高有效位到最低有效位进行（MSD 基数排序 [22]），或者反之（LSD 基数排序 [15, 21]）。

在任一情况下，第一步都是在顺序扫描中计算输入值的直方图。由于直方图反映了应放入 r 个桶中每个桶的键数量，因此对这些计数计算排他性前缀和即可得到每个桶的内存偏移量。最后，根据键的数字值将键分散到各个桶中。对后续数字递归地在生成的桶上重复这些步骤，最终得到排序后的序列。目前性能最好的排序算法是基于 MSD 基数排序 [22] 的。

## 2.3 K-选择问题

k-selection 问题要求在一个包含 n 个元素的列表中找到第 k 大的值。有了 k-selection 问题的解决方案，只需对数据进行一次额外的遍历，就可以轻松找到 top-k 元素。Alabi 等人 [5] 对此问题进行了广泛研究。除了对元素进行排序并选择第 k 大的元素外，他们还研究了另外两种算法：Radix Select 和 Bucket Select。

Radix Select：Radix select 源于 MSD radix sort 算法。与 MSD radix sort 一样，它作为一系列步骤来执行，每个步骤处理一个 d 位的数字。它执行相同的直方图和前缀求和步骤。然而，radix select 不是将所有条目写入划分的桶中，而是使用直方图找到包含第 $k ^ { t h }$ 大条目的桶 B。然后，它只写出 B 的条目，并继续仅检查匹配桶中元素的下一个 d 位数字。

Bucket Select：Bucket select 不再基于 radix 位创建桶，而是尝试通过基于最小值和最大值计算桶来提高鲁棒性。该算法对数据集进行一次显式的初始遍历，以计算最小值和最大值。随后，我们执行一系列遍历。每次遍历包含三个步骤：首先，在最小值和最大值之间创建多个等距的桶，并计算每个线程在每个桶中的条目数。其次，进行前缀求和并找到包含第 $k ^ { t h }$ 大元素的桶。最后，读取输入并写出匹配桶中的元素。我们在匹配桶的条目上运行下一次遍历。

Algorithm 1: Per-Thread Top-K   
Input : List L of length n; const int k   
Output : List O of the top-k elements per thread   
1 int t ← getGlobalThreadId();   
2 int nt ← numThreads();   
3 MinHeap heap;   
4 for i ← t; i < n; i += nt do   
5 T xi = L[i]; // T 是键的类型   
6 if xi > heap.min() then   
7 heap.pop(); heap.push(xi);   
8 for j ← 0; j < k; j++ do   
9 O[t + j\*nt] = heap.pop();

## 3 算法

基于目前的讨论，我们有 3 种寻找 top-k 元素的算法：

• Sort and Choose：使用 radix sort 对整个向量进行排序，并从中选择 top-k 元素。

• Using Radix Select：我们可以使用基于 radix 的选择算法来获取第 $k ^ { t h }$ 大的元素，并通过在输入数组上进行一次额外的遍历来找到 top-k。

• Using Bucket Select：我们可以执行与上述相同的操作，只是这次使用 Bucket Select 而不是 Radix Select。

在本节中，我们描述了两种用于寻找 top-k 元素的新算法。在第一种算法中，每个线程独立维护其目前所见过的 top-k 元素，并在局部（每个线程的）top-k 中找到全局 top-k。其次，我们提出了基于 bitonic sorting 的 bitonic top-k 算法。为了便于表述，我们的描述假设元组仅由一个键组成。当然，实际应用可能需要在其他设置上执行 top-k，包括 (key,value) 对、多个键以及不同的数据类型和分布；我们的评估表明我们的算法涵盖了所有这些情况（第 6 节）。

## 3.1 Per-Thread Top-K

单线程版本的 top-k 会在 min-heap 中维护 top-k 元素，并为每个看到的新元素进行更新。并行化的自然方法是划分输入，计算每个分区的 top-k，并作为最终的归约步骤从这些局部 top-k 中计算出全局 top-k。Algorithm 1 展示了将在每个线程中并行运行的伪代码（nt 个线程并行运行）。我们为每个线程使用一个 min-heap 来维护该线程目前所见过的 top-k 元素。在初始化堆之后，我们从 t 开始以线程数为步长迭代元素。这种（合并的）内存访问模式已被证明有利于 GPU 上的内存访问 [11]。我们检查当前元素是否大于所见 top-k 中的最小值。如果是，我们弹出最小值并添加当前元素。最后，我们以合并的方式将 top-k 值写入 O。这种方法在内存使用方面非常高效。它对全局内存进行一次完整的读取遍历，并写入显著减少的数据。然而，它存在线程发散和占用率问题，这将在第 4.1 节中更详细地讨论。

Algorithm 2: Bitonic Top-K Local Sort   
Input : List L of length n   
Output : L with sorted sequences of length k   
1 int t = getGlobalThreadId();   
2 for len ← 1; len < k; len ← len ≪ 1 do   
3 dir ← len ≪ 1;   
4 for inc ← len; inc > 0; inc ← inc ≫ 1 do   
5 int low ← t & (inc − 1);   
6 int i ← (t ≪ 1) − low;   
7 bool reverse ← ((dir & i) == 0);   
8 x0, x1 ← L[i], L[i + inc];   
9 bool swap ← reverse ⊕ (x0 < x1) ;   
10 if swap: x0, x1 ← x1, x0;   
11 L[i], L[i + inc] ← x0, x1;

## 3.2 Bitonic Top-K

虽然完整的双调排序是解决 top-k 问题的一种方法，但它在对整个输入进行排序时执行了大量不必要的工作，就像堆排序在选取 top-k 时效率远不如使用优先队列一样。

在双调排序中，我们从一个未排序的数组开始，这等价于长度为 1 的已排序序列，然后构建长度为 2, 4, ... 直到 n 的更长已排序序列，此时整个列表便已排序完成。我们的基本方法是开发一种算法，它执行尽可能少的不必要工作，同时保持双调排序的大规模并行特性。为了实现这一点，我们将复杂的双调排序操作分解为一系列具有不同比较距离的并行步骤。我们仔细地将这些步骤重新组装为三个算子，这些算子组合起来，允许对向量的 top-k 元素进行高效的完全并行计算。这些算子是局部排序、合并和重建。

在局部排序中，我们使用（部分）双调排序生成大小为 k 的已排序序列。在合并中，我们双调合并两个大小为 k 的已排序序列，从而创建两个双调序列，其中第一个序列包含 k 个最大元素（w.l.o.g），第二个序列包含 k 个最小元素。在重建中，我们对包含最大元素的序列进行排序；包含较小 k 个元素的第二个序列则被丢弃。在排序时，我们利用了第二个算子的输出已经满足双调特性这一事实。此时，我们已经有效地将问题规模减半。我们将合并和重建算子递归地应用于（减半后的）序列，直到只剩下构成 top-k 的 k 个元素。由此产生的算法不执行任何不必要的工作，并且具有双调排序的大规模并行性。在本节的其余部分，我们将更详细地描述各个算子。

(1) 局部排序。该算子的目标是从一个未排序的数组开始，生成长度为 k 的交替升序和降序的已排序序列。算法 2 展示了伪代码。未排序序列等价于长度为 len = 1 的已排序序列。从 len = 1 开始，我们生成长度为 len = 2<sub>,</sub> 4 ... k 的已排序序列。当 len = k 时，我们就完成了。这是第 3 行的外层循环。当 len = x 时，两个相邻的长度为 x 的已排序序列组成长度为 2x 的双调序列，并且可以在 loд(x) + 1 步内完成排序。这由第 4 行的内层循环处理。在第一步中，当 inc = len 时，我们比较元素对 (L[0] L[len]), (L[1] L[1 + len]), ... (L[len − 1] L[2len − 1])。这是并行完成的，每个线程比较一对元素。通常，线程 t 将元素 L[i] 与 L[i + len] 进行比较，其中索引 i 作为 t 和 inc 的函数计算，如第 5 − 6 行所示。元素被比较并交换（12-13）（如果需要）并写回原数组（14-15）。交换的方向由 len 决定。当 len = x 时，我们想要生成长度为 2x 的交替升序降序已排序序列，即：方向每 dir = 2 ∗ len 个元素改变一次（第 3 行）。比较的实际方向由 (i/dir) 是奇数还是偶数决定（第 7 行）。图 5 中的阶段 1 说明了局部排序算子的访问过程，以从 16 个元素中找出 top-4。

![](images/68f9a161f82064ab0f17779101199c1a3873396017ff8a8cd1f0de6e809eaf16.jpg)  
(a) Top-K 合并前 (b) Top-K 合并后

图 4：Top-K 合并
<table><tr><td>算法 3：Bitonic Top-K 合并</td></tr><tr><td>输入：具有长度为 k 的已排序序列的列表 L</td></tr><tr><td>输出：大小为 |L|/2 且具有长度为 k 的双调序列的列表 $L _ { 2 }$ </td></tr><tr><td>1 int t ← getGlobalThreadId(); 2 int low ← t &amp; (k-1);</td></tr><tr><td>3 int i ← (t « 1) - low;</td></tr><tr><td> $L _ { 2 } [ \mathrm { t } ] \gets \operatorname* { m a x } ( \mathrm { L } [ \mathrm { i } ] , \mathrm { L } [ \mathrm { i } + \mathrm { k } ] ) ;$  4</td></tr><tr><td></td></tr></table>

(2) 合并。在局部排序结束时，我们得到了长度为 k 的交替升序降序已排序（即双调）序列。我们成对比较相邻的序列，并在每一对中选择较大的元素。虽然我们不知道每个序列中有多少元素被选中，但我们知道 top-k 元素已被选中，并且它们构成了一个双调序列。这是我们工作的关键洞察。为了说明这一点，请看图 4，它说明了 top-8 的计算：在图 4b 中（合并步骤之后），左侧的所有元素都在 top-8 之中，因为它们大于其比较伙伴，这意味着它们大于其比较伙伴左侧（或右侧）的所有元素（由于双调特性）。这一步将 top-k 候选集减半。算法 3 展示了伪代码。

(3) 重建。重建的输入是一个包含长度为 k 的双调序列的列表 L，而不是局部排序算子中的未排序序列。因此，我们可以通过从 $l e n = k / 2$ 开始应用局部排序的内层循环，在 loд(k) 步内生成长度为 k 的已排序序列。为了完整性，附录 B 中的算法 4 展示了伪代码。其流程与局部排序相同。合并和重建的组合将一个长度为 n 且包含长度为 k 的已排序序列的列表缩减为长度为 n/2 且包含长度为 k 的已排序序列的列表。合并和重建重复进行，直到我们得到一个长度为 k 的列表。

![](images/3781d10310c1e142d18104a7885fad628112a932eb5d3ac5419d41898daa13b4.jpg)

(a) 算法  
![](images/072f71f12113371d6180710257529055de03c6a49c654ebc95641b499a50813b.jpg)  
(b) 可视化（灰色：非活跃，橙色：候选）
图 5：Bitonic Top-K (K=4)

分析。在局部排序中，每一步进行 n/2 次比较。共有 loдk 次外层循环迭代，其中第 i 次有 i 步。在合并中，我们进行 n/2 次比较。在重建中，我们有 loдk 步，每步 n/2 次比较。每次运行合并时，列表大小减半。合并和重建运行多次，直到我们得到一个大小为 k 的列表。比较的总次数为 $O ( n l o g ^ { 2 } k )$ 。与双调排序网络一样，双调 top-k 的运行时间与数据分布无关，仅取决于 |L| 和 k。

## 4 优化与实现

在本节中，我们描述了我们在逻辑和实现层面上应用于不同方法以优化性能的若干优化措施。本节中的所有性能数据均来自在 Nvidia Titan X Maxwell GPU 上对由均匀分布 U(0 1) 生成的 $2 ^ { 2 9 }$ 个浮点值的数据集运行算法的结果（有关硬件设置的详细信息，请参见第 6.1 节）。关于更多样化数据的数据在第 6 节中给出。

## 4.1 Per-Thread Top-K

为了高效实现 per-thread top-k 算法（Algorithm 1），我们使用共享内存来存储堆。每个线程块在共享内存中分配一个大小为 k ∗ wд 的数组，其中 wд 是线程块的大小。每个线程在共享内存中使用一个大小为 k 的数组来维护自己的堆。为了避免 bank conflicts，我们以交错方式存储数组，其中线程 t 使用条目 sdata[t + wg\*i]，这里 sdata 是线程块使用的共享内存数组，且 i 在 0<sub>...</sub>k 之间变化。由于 wд 总是 32 的倍数，每个线程的数组映射到一个共享内存 bank，且 warp 中的多个线程更新它们各自的数组不会导致共享内存 bank conflicts。

该实现存在两个问题：

Thread Divergence：堆更新是数据相关的。在 GPU 上，一个 warp（32 个线程）在 SIMT 模型下运行。因此，即使只有一个线程想要更新其条目，warp 中的所有其他线程也必须遵循相同的指令路径，从而导致速度下降。Occupancy：每个线程使用的共享内存随 k 的增加而增加。随着一个 block 使用的共享内存增加，可以运行的并发 warp 数量会减少。超过一定程度后，occupancy 的降低会导致 GPU 没有足够的活跃 warp 来饱和全局内存带宽。对于 $k \geq 5 1 2 ,$ 即使使用最小的线程块大小 32，我们也需要 64KB 的共享内存，这大于我们 GPU 上每个线程块可用的 48KB。

我们还使用寄存器实现了 per-thread top-k 算法，发现其性能较差。Appendix A 包含对基于寄存器版本的更详细讨论和性能比较。

## 4.2 Selection-based Top-K

所使用的 radix select 和 bucket select 实现来自 GGKS 包 [4]。我们修改了 radix select 的实现，使用 8 位数字（基于 MSD radix sort [22]）而不是原始代码中的 4 位数字。这使得 32 位（int 和 float）键需要进行 4 趟处理。每趟处理都可以减少数据大小。然而，如果在 prefix sum 之后我们没有看到数据减少，则跳过聚类步骤，我们只需在下一趟处理中重用输入。Bucket select 也一次将数据划分为 16 个桶，并选择一个包含第 k 个元素的桶。感兴趣的读者可以参考 [5] 获取更多实现细节。radix select 实现会在每趟处理后写出整个输入数组，然后更新数组指针以指向包含 $k ^ { t h }$ 个元素的桶。我们修复了这一低效问题，仅写出正确的桶。

给定第 $k ^ { t h }$ 高的元素 X，我们可以对数据进行额外的一趟处理以找到 top-k 元素。然而，这是没有必要的。一旦我们选择了包含 X 的桶，在第二次扫描数组以写出落入该桶的元组时，我们也可以将更高桶中的元素写到一个单独的结果数组中。在最后一趟处理中，我们将已识别桶中所有值小于 X 的元素复制到结果中，并用 X 填充结果使其大小为 k。这消除了我们之前在给定 X 时寻找 top-k 元素所需的最后一趟处理。

## 4.3 优化双调 Top-K

在本节中，我们讨论了为实现新双调 top-k 算法接近最优性能而设计的一些优化。虽然其中一些是受其他算法的类似优化启发，但据我们所知，它们均未应用于 top-k 计算的上下文中。然而，由于我们的优化可能适用于其他问题，我们在每项优化的描述中都包含了一段关于新颖性和适用性的内容。为了说明每项优化的重要性，我们在每个小节的末尾都用一张图表展示了优化对本节开头描述的数据集中寻找 top-32 元素的运行时间的影响。

在共享内存中操作。第一项优化可以单独应用于三个算子中的每一个：我们不再在每个大规模并行步骤之后向全局内存读写数据，而是每次操作只读写一次。所需的数据在操作开始时被加载到共享内存中。所有操作的中间步骤都在共享内存中进行。最后，将结果写回全局内存。例如，图 5a 中的局部排序操作有 3 个中间步骤。通过此优化，每个 threadblock 会将所需数据读取到共享内存中，在共享内存内运行这 3 个步骤，然后在最后写回结果。回想一下，共享内存比全局内存快一个数量级。此优化将全局内存读写转移到共享内存读写，从而提高了性能。

![](images/e3ad636e3f0af2b5fc9a6b2da59b639d792c1d194b58b487530b30e530fd0a4f.jpg)

这使得性能得到了显著提升，从 521ms 降至 122ms。局部排序算子变为受限于共享内存，而合并和重建仍然受限于全局内存。

请注意，此优化取决于 k 是否小于或等于 2∗最大线程块大小（在现代 GPU 上为 2048）。它也不能应用于具有 inc 高达 $n / 2 ,$ 步骤的通用双调排序算法的所有步骤，因为这需要将整个数组加载到共享内存中，但只要 k 足够小，这在我们的双调 top-k 算法中并不是限制。此优化已应用于双调排序以最小化对全局内存的访问 [17]。

合并算子。如第 3.2 节所述，我们的双调 top-k 算法可以分解为三个操作：（1）局部排序以生成长度为 k 的已排序序列，（2）合并两个长度为 k 的已排序序列以生成长度为 k 的双调序列，以及（3）在合并后重建长度为 k 的双调序列。虽然局部排序操作只在开始时执行一次，但合并和重建阶段会交替进行，直到找到结果。

朴素的实现会为每个算子运行一个 kernel。然而，各个 kernel 之间不存在跨线程块的依赖关系。这带来了显著的优化潜力：多个算子可以融合为一个 kernel，并且可以使用共享内存来传递算子之间的结果。除了减少 kernel 调用开销外，此优化还消除了由中间结果引起的全局内存流量。

每次合并会使元素数量减半。为了确保融合 kernel 中的最后一个操作中的每个线程都有工作可做，我们需要确保每个线程的数据项数量至少为 $2 ^ { x }$ ，其中 x 是融合 kernel 中的合并阶段数。我们发现每个线程处理 8 个数据项是最佳数量。超过这个数值，每将每个线程的元素数量翻倍，就会使共享内存 bank 冲突的数量翻倍，并且不会带来任何性能提升。由于每次合并都会使每个线程的元素数量减半，因此每个线程处理 8 个元素允许我们在每个 kernel 中进行三次（即 ld(8)）合并阶段。这导致了两个独立的 kernel：第一个执行局部排序，随后是两个合并-重建算子和一次单独的合并（SortReducer）。第二个 kernel 执行三个重建-合并算子序列（BitonicReducer）。据我们所知，这是一项新颖的优化。

![](images/ddbbccc621eae31a6c8fa0e8fb0dd0d17c42a591e17f508f5e788a92a5a1c9fa.jpg)  
(a) 单步  
(b) 合并 2 步  
(c) 合并 3 步  
图 6：合并多个步骤

此优化将 top-32 的运行时间从 122ms 减少到 48.15ms。两个 kernel（以及整个应用程序）现在都受限于共享内存带宽。SortReducer kernel 和 BitonicReducer kernel 分别实现了 2.75TBps 和 2.7TBps 的共享内存带宽。这大于在重复读取工作负载上观察到的 2.9TBps 共享内存峰值带宽的 90% 利用率。因此，我们将注意力转向优化共享内存访问。

![](images/bd5ed3fe8788d038de11a3405705819ea3d6177b016eba555f21a88dbc7f5ab7.jpg)

合并/串行化多个步骤。对于下一项优化，我们重新安排数据项到线程的分配，以减少内存流量。图 6a 显示了默认分配（线程用颜色编码），每个线程从共享内存中读取两个值，对它们进行比较，然后将它们写回共享内存。由于每个线程负责 8 个元素，它对其他 3 对执行相同的操作。然而，如果我们在每轮中每个线程处理多于两个值，读写操作就可以被共享。例如，在图 6b 中，橙色线程读取位置 0、2、4 和 6 的值，并对每个值执行两次比较。这将共享内存流量减半，并且可以推广到更多元素（见图 6c）。虽然这（部分地）串行化了处理过程（从三个各有四个操作的完全并行步骤变为 12 个串行操作），但它并没有增加比较的总次数。此优化类似于优化 1，后者将读写全局内存的多个步骤合并为读写共享内存。而在这里，我们将读写共享内存的多个步骤合并为在寄存器中工作。这将运行时间减少到了 33.7ms。

![](images/8400bf98bbae7c31681786eeb996e31a026864902c9d2290a32d12c46395ec.jpg)

![](images/8d5494c64c45e5623ce6b818f54d78900dd580ecbad0a85f757d32f322413f9d.jpg)  
图 7：通过填充避免 Bank 冲突（框中的数字表示访问线程的 ID）

在写入前执行工作。传统经验是以合并的方式将一块数据从全局内存复制到共享内存，并仅在共享内存中处理数据。然而，通过摒弃这一传统经验，我们可以减少共享内存访问。每个线程从全局内存中将 8 个连续元素加载到寄存器中，执行创建长度为 8 的局部排序序列所需的所有中间步骤，而不触及共享内存，然后再写入共享内存。请注意，由于此优化的结果，对全局内存的访问不再是合并的，因为线程以 8 的步长访问数据元素。然而，由于现代 GPU 具有数据缓存，这不会导致任何明显的性能差异。此优化可能具有广泛的适用性。在我们的实验中，它使运行时间有效减少到了 27.1ms。

![](images/b5a0b15c015c3c9421ae24cf14cc51953bb55b9af23e74f10472a4a348ccdd63.jpg)

通过填充打破冲突。在本小节及下一小节中，我们将介绍三种有助于避免内存 bank 冲突的优化。虽然目前大多数 GPU 都有 32 个共享内存 bank 和 32 个线程的 warp，但在 32 个内存 bank 上展示我们优化的效果会不必要地增加图表的尺寸。因此，我们在图示中假设有 8 个内存 bank（以及大小为 8 的 warp）（请注意，实验是在具有 32 个内存 bank 的真实 GPU 上进行的）。

第一个优化是一种广为人知技术的实例：填充数组以避免内存冲突。大小为 n 的共享内存数组可以被视为维度为 [ <sup>n</sup> <sub>,</sub> 8] 的 2D 数组（其中 8 是 bank 的数量）。核心思想是分配稍多的内存以创建一个维度为 [ <sup>n</sup> 9] 的更大数组。增加的额外列不存储任何元素，然而，它有助于打破共享内存冲突。图 7 展示了在填充后，在时间步 0 结合 inc = 2<sub>,</sub> 1 的组合步骤所执行的访问。灰色的单元格不包含任何值，仅仅是空间开销。每个线程希望读取 4 个连续的元素。线程 0 希望读取条目 0-3，线程 2 希望读取 8-12。如果没有填充，这两个线程将会冲突（0 和 8 在同一个 bank 中）。该图说明了填充如何防止冲突（填充后线程 0 和 2 访问不同的内存 bank）。这将 top-32 的运行时间减少到 22.3ms。请注意，由于 bitonic sorting 受限于全局内存带宽，填充对其没有帮助。相比之下，bitonic top-k 受限于共享内存带宽。

填充还有第二个好处：它允许我们将更多算子合并到一个 kernel 中。回想一下，处理超过 8 个元素

![](images/e717c280a842929a412d3866e67ecdd2ea24b13885a01f8d4eb546969a2111ad.jpg)  
会引起冲突（在本节前面关于算子合并的部分讨论过），并且这种效应限制我们只能合并三个算子。

有了填充，这不再成立，这允许我们合并四个甚至更多步骤（每个线程处理 16 个或更多元素）。然而，超过 16 后，分配的寄存器数量迫使编译器降低占用率，导致性能下降：图 8 显示了改变处理元素数量（B）时的性能。将 B 从 16 增加到 32 几乎没有好处，而将 B 增加到 64 则有不利影响。因此，我们将 B 固定为 16。
![](images/81601aa9dd307b76d0ff5fb3fd600af006ff56e9123a989f91ede36ebada93ca.jpg)  
图 8：改变每个线程的元素数量

Chunk Permutation。图 9 展示了应用迄今为止讨论的优化后，local sort 操作的共享内存访问模式。在这里，每个轮廓形状代表一个不访问共享内存的操作（共享内存访问在每个形状的边缘执行），坐标轴代表 kernel 内顺序循环的迭代，数字代表被比较元素在输入数组中的距离。虽然大多数 kernel 是无 bank 冲突的，但我们观察到，当比较距离为 4 时，内存访问会引起 bank 冲突。为了说明这一点，请看图 10a：它展示了图 9 中红框内执行的比较（距离为 4 的元素成对比较）。该图展示了 warp 中线程的内存访问：每个箭头代表一个线程中执行的比较，颜色表示访问的时间（因此也代表顺序）。我们观察到，尽管进行了填充，但在时钟时间 0 的内存访问在内存 bank 方面仍然重叠。我们可以通过改变每个线程读取（和写入）的内存位置来避免这种情况。我们将此优化称为 Chunk Permutation，并在图 10b 中进行了说明：在时钟 0 时，每个线程不再从冲突的 bank 读取，而是访问不同的内存 bank。人们可能会注意到，访问的值之间仍然存在重叠。然而，这些访问是在不同时间执行的。从图中很明显可以看出，通过观察一列中没有两个颜色相同的框，可知不存在冲突。

重新排列 chunk 以避免 bank 冲突的更广泛思想可以应用于其他受共享内存 bank 冲突影响的算法。

Reassigning Partitions。我们开发的最后一个优化针对的是第一次 reduction 后数据项到线程的分配：由于 reduction 将元素数量减半，但线程数量保持不变，因此每个线程的工作量减少。这导致被合并的步骤减少，因为可以合并的步骤数是每个线程输入数据项数的对数。为了在 reduction 后保持每个线程具有相同数量的输入数据项，我们让一半的线程执行所有工作。虽然这使一半的线程没有工作，但由于更大的组合步骤而减少的共享内存流量抵消了这一代价。该优化进一步将性能提升到 15.4ms。此优化是新颖的，可能适用于分阶段 reduction 输入数据的 kernel。

## 讨论：

内存使用。内存使用对于基于 GPU 的数据管理系统至关重要。对于大小为 n 的数据集，out-of-place bitonic top-k 使用一个大小为 n/8 的额外 buffer。这显著少于需要大小为 n 的额外 buffer 的排序和基于选择的方法。

大于 GPU 内存的数据。当数据大于 GPU 内存可容纳的大小时，需要通过 PCI 总线将数据移动到 GPU。有大量研究致力于使用异步传输 [10, 22]、近似 [18]、压缩 [9, 20] 和成本感知的设备选择 [7] 来减轻该瓶颈的压力。虽然我们在本文中没有明确解决 PCI 瓶颈问题，但 top-k 查询的归约特性使得按内存大小的 chunk 处理数据并将计算与传输重叠变得很简单（类似于排序所做的工作 [22]）。

CPU 上的 Bitonic Top-K。所描述的 bitonic top-k 算法也可以在 CPU 上实现。我们在附录 C 中描述了我们的实现。应用了本节描述的所有优化后，bitonic top-k 在 GPU 上接近于计算受限。当相同的算法在 CPU 上运行时，由于 CPU 的计算与带宽比较低，它严格受限于计算。

## 5 数据库集成

在开发出高度优化的、大规模并行的 top-k 实现之后，我们自然对其在完整系统中的可用性产生了兴趣。作为概念验证，我们将 bitonic top-k kernel 集成到了开源 GPU 数据库 MapD [16] 中。在本节中，我们将讨论在数据库分析中可用于提升性能的两种优化机会。

与 filter 融合。一种常见的查询模板是在满足选择谓词的数据子集中寻找 top-k 项。执行此操作的简单方法是让一个单独的 kernel 执行 filter，并让随后的 top-k kernel 使用其输出来查找 top-k 项。基于 GPU 的数据库目前最终就是这样做的，因为它们将 top-k kernel（通过排序完成）视为黑盒。我们可以通过将 select 融合到 bitonic top-k 例程中来优化这一点。

运行 SortReducer kernel 的每个 thread block 读入 16nt 个元素并写出 t 个元素，其中 t 是 thread block 中的线程数。融合 kernel 的一种方法是读入 16nt 个元素，应用 filter 谓词，并在匹配的元素上运行 SortReducer。然而，此时 SortReducer kernel 实际上是在 s ∗ 16nt 上运行，其中 s 是 selectivity。如前一节所示，每个线程拥有 16 个元素对 SortReducer 的性能至关重要，因为这使其能够运行组合步骤。相反，FusedSortReducer 使用选择步骤作为缓冲区填充器。它一次读入 nt 个元素到 shared memory 中，应用 filter 谓词以找到匹配元素的数量，计算 prefix sum，然后将其写出到大小为 16nt 的 shared memory 缓冲区中。接着它读入下一批 nt 个元素，直到我们有超过 15nt 个匹配的元素。其余条目用 min/max value 填充，以便它们永远不会出现在 top-k 结果中。然后 SortReducer 在 16nt 个元素的缓冲区上工作，并写出包含 top-k 的 nt 个元素。

自定义排名函数 自定义排名函数是形如 $f ( A _ { 1 } , A _ { 2 } , A _ { 3 } . . )$ 的 order by 子句，其中 $f$ 是任意函数，$A _ { 1 } , A _ { 2 } , . .$ 是 A 的列。排名函数可以在 SortReducer kernel 的开始处进行评估，而不是作为一个单独的 project 步骤运行以输出该函数的值。

## 6 评估

在本节中，我们比较了在第 3 节中提出的五种不同算法的性能：

(1) Sort：通过排序查找 top-k

(2) PerThread TopK：使用每个线程一个堆来查找 top-k

(3) Radix Select：改编 radix select 来查找 top-k

(4) Bucket Select：改编 bucket select 来查找 top-k

(5) Bitonic TopK：在应用了第 4 节中的优化之后，使用 bitonic top-k 算法，并改变以下参数：(1) K 的值 (2) 键数据类型 (3) 数据分布 (4) 数据大小 (5) 键和值列的数量，最后是 (6) 设备（CPU vs. GPU）。之后，我们通过在 twitter 数据集上评估 top-k 查询，展示了将 BitonicTopK 集成到 MapD 数据库中所达到的性能。

## 6.1 设置

所有结果均为在单插槽 Intel i7- 6900 @ 3.20GHz（具有 8 个核心、16 个硬件线程的 Skylake）上运行 3 次的平均值，配备

Nvidia GTX Titan X Maxwell GPU，运行于 Ubuntu 15.10 (Kernel 4.2.0-30) 和 CUDA 8.0。

## 6.2 不同 K 值下的性能

我们生成了 $2 ^ { 2 9 }$ 个随机均匀分布 (U (0 1)) 的浮点数，并观察了 K 以 2 的幂从 1 变化到 1024 时不同算法的性能。图 11a 显示了结果。

Memory Bandwidth 显示了从 global memory 读取整个数据所需的时间。由于所有数据至少需要被读取一次，这构成了任何算法运行时间的下界。实际上，大多数算法会写入/读取中间数据并产生其他开销。我们观察到 Sort 方法的运行时间在 k 之间几乎保持不变，因为无论 K 多大，它都必须对整个输入进行排序。

正如预期的那样，Radix Select 和 Bucket Select 在不同 K 值下花费的时间几乎相同。由于使用了更昂贵的原子操作，后者的性能比前者差。当 k = 1 时，Bucket Select 很快，因为它在找到数组的 min-max 后终止，并直接将其作为结果返回。

PerThread TopK 曲线从 $k = 3 2$ 开始具有陡峭的斜率，这是由于如前面第 4.1 节所解释的 occupancy 降低和 thread divergence 造成的。由于所需的 shared memory 数量，该方法在 K > 256 时失败。对于 $K = 5 1 2$ ，即使 thread block 大小最小为 32，我们也需要 $5 1 2 * 3 2 * 4 = 6 4 K B$ （每个键 4 字节），这超过了每个 thread block 可用的 48KB。

最后，对于 $K \leq 2 5 6 .$ ，Bitonic 的表现优于所有其他算法。对于 $K > 2 5 6 ,$ ，Radix Select 方法表现更好。

## 6.3 对数据类型的依赖性

接下来，我们在包含 $2 ^ { 2 9 }$ 个取自 $U ( 0 , 2 ^ { 3 1 } - 1 )$ 的无符号整数的数据集上运行这些算法（见图 11b）。除了 Radix Select 之外，所有方法所花费的时间与使用 float 数据类型观察到的时间几乎完全相同。Radix Select 表现更好，因为对于均匀分布的数据，每次扫描消除的元组数量是最大的（假设使用 8-bit radices，则减少 256 倍）。

其次，我们在取自 U(0 1) 的 $2 ^ { 2 8 }$ 个 double 上运行这些算法。数据大小相同，但是每个键的字长增加了。图 11c 显示了结果。基于 Sort 的方法必须执行两倍的扫描次数（因为位数翻倍了），但扫描的值更少。然而，在大多数 GPU 上，处理 64 位值比处理 32 位值要昂贵得多，这解释了成本增加的原因。Radix Select 也有同样的问题，但是，这种效果不太明显，因为该算法在后续传递中操作较少的元素。Bucket Select 最终比处理 floats 时略快，因为键数量的减少导致原子操作更少。PerThread TopK 曲线与处理 float 时看到的曲线相似，向左移动且略低：这很自然，因为每个读取字节需要执行的处理更少。对于每个 $K ,$ ，该方法在处理 doubles 时使用的 shared memory 是处理 floats 时的两倍。因此，该方法更早失败（对于 $K > 1 2 8 )$ 。最后，Bitonic TopK 基本保持不变，因为数据大小相同且成本主要由 memory bandwidth 决定。

(a) 递增  
![](images/91d17c6a9174845eefd205a691a30190e36cd45fa14bc6309e995e349531c660.jpg)  
(a) 使用 float 键

![](images/a731102cf4ca2e7dd57ee326d0c15b2926ac4e2e51dd2b14f97d7e51118fdd40.jpg)  
(b) 使用无符号整数键

![](images/cfaa34b26689d14fa32f3c14905ba81d3388c9870b66a59a52444c801c1c7f6e.jpg)  
(c) 使用 double 键  
图 11：不同 k 值下的耗时

![](images/81140ee6831345b0dd397b385232afab3515417a8076729e65aeeead5e34af8c.jpg)

![](images/0dba963827bcd87c8615cacba4c3eae92f3eba70a2172c26487367ac932d53f0.jpg)  
图 12：不同分布下的性能

## 6.4 对数据分布的依赖

保持数据大小固定为 $2 ^ { 2 9 }$，我们在两种分布下检验算法在不同 k 下的性能：

• Increasing：来自 U (0 1) 的已排序浮点数

• Bucket Killer：包含全1（浮点数），除了4个数字，每个数字在一个8位数字上与1.0不同。这最小化了单次基数扫描（radix-scan）中实现的数据缩减。

图12展示了结果。唯一不随元素分布改变而改变性能的算法是 Sort 和 Bitonic TopK。两者执行的操作完全相同。

Increasing（图12(a)）分布导致 PerThread TopK 的性能下降高达3倍，而其他算法没有变化。这是因为 PerThread TopK 的性能取决于堆插入的次数。在 Increasing 分布下，每个元素都会导致一次堆插入，使其成为该算法的近乎最坏情况。

对于大多数选择算法而言，相对容易找到会导致算法最坏情况的分布。Bucket killer 是 Radix Select 的对抗性分布。在 Bucket killer（图12(b)）下，Radix Select 最终花费的时间与 Sort 相同，因为每次基数遍历只导致一个数字被排除在考虑范围之外（即在该8位数字上与1不同的那个数字）。每次遍历最终都会像 Sort 一样读写整个数据集。由于中间步骤的数据缩减较少，Bucket Select 也出现了2倍的减速。请注意，由于双调合并（bitonic merges）的模式可预测，Bitonic TopK 方法不存在对抗性输入分布，这使其成为一个非常稳健的选择。

## 6.5 对数据大小的依赖

为了展示算法在不同数据大小下的性能，我们在固定 k = 64 的情况下运行它们，并选择从 U (0 1) 中抽取的随机浮点数数据集，数据大小从 $2 ^ { 2 1 }$ 到 $2 ^ { 2 9 }$ 不等。图13展示了结果。Bitonic TopK 和 Sort 随输入大小线性增长。PerThread TopK 维护每个线程的 top-k，并运行固定数量的线程以保持所有 GPU 核心繁忙。随着数据大小的增加，每个线程处理的元素数量也会增加。此外，对于均匀分布，随着看到的数据越来越多，堆插入的概率会降低。这导致了初始的向外凸起。Radix Select 和 Bucket Select 在较大的数据大小下呈线性增长。当数据大小低于 $2 ^ { 2 4 }$ 时，前缀和（prefix sum）所花费的时间（在不同数据大小间是一个常数）变得显著，导致曲线变平。

![](images/119c3134a790f9e35e65431dc215ed0cb18e734e06ec6f79aea2dbdb072df435.jpg)  
图13：不同大小下的性能

![](images/5ae75733db03798779bf0a46731825af9898408da0b063f8575ff41aa9370e22.jpg)  
图14：不同键数量

## 6.6 键+值

到目前为止，我们使用的元组仅包含一个键。然而，许多应用需要键+值或多个键+值。在本节中，我们展示了 Radix Select 和 Bitonic TopK 在键+值（KV）、两键+值（KKV）和三键+值（KKKV）下的性能。每个键是从 U(0 1) 中抽取的浮点数，值是一个4字节整数。数据集中元素的大小为 $2 ^ { 2 8 }$。图14展示了结果。随着我们从 KV 变为 KKKV，由于数据大小增加，两种方法的运行时间都呈线性增长。不同键数量下的交叉点（cut-off point）保持不变。为了可读性，我们未展示其他方法的结果。

我们不展示具有更大值负载的实验，因为传递元组ID并在 top-k 结束时构造完整元组总是更好的选择。例如，考虑一个包含1000万个元组的数据集，每个元组具有4字节键和12字节负载。在 (key,id) 而不是 (key,payload) 上运行 top-k 可以将移动的数据大小减半。在最后组装结果几乎不花费时间。

## 6.7 与 CPU 的比较

在本节中，我们将基于 CPU 的 top-k 与基于 GPU 的 top-k 的性能进行比较。对于基于 CPU 的 top-k，我们有两种基于堆的方法：一种使用 C++ STL 优先队列作为最小堆（STL PQ），另一种是手工优化的最小堆（Hand PQ）。对于每个元素，我们通过与堆的根进行比较来检查它是否大于堆的最小值。如果更大，我们就弹出根（最小值）并插入新元素。我们还展示了 CPU 版本的双调 top-k。对于基于 GPU 的 top-k，我们展示了 Bitonic TopK 和 Radix Select。

![](images/e5e9d429789816062a7de7708888066eb866ca8740d78ec6917999d5cef96a52.jpg)

![](images/0174ee80a508fb350790638c5791f20d1073755ce690b2562c81fd82b22efa8c.jpg)  
图15：与 CPU Top-K 的比较

首先，我们在从均匀分布 U (0 1) 中抽取的 $2 ^ { 2 9 }$ 个浮点数数据集上对它们进行比较。图15(a)展示了结果。由于数据是均匀分布的，大多数元素在与堆最小值进行比较时都会被丢弃，只有极少数会触发堆插入。为了说明这一点，请注意，对于这个数据集，当 k = 32 时，每个核心处理 671k 个元素，最终只进行约500次插入（包括总是会被插入的前32个元素）。因此，性能很可能受限于内存。当 k = 32 时，Bitonic TopK 比 Hand PQ 好3倍。CPU 上的双调 top-k 性能明显差于基于堆的方法，因为它比基于堆的方法进行了更多的计算，而后者只进行了500次插入。

接下来，我们考虑相同但按递增顺序排序的数据集。图15(b)展示了结果。由于数据已排序，每个元素都会导致一次堆弹出/插入。这接近最坏情况。Bitonic TopK 和 Radix Select 花费的时间相同，而 CPU 算法的表现则差得多。当 k = 32 时，Bitonic TopK 比 Hand PQ 好60倍，比 STL PQ 好120倍。尽管进行了更多的比较，CPU 上的双调 top-k 所花费的时间却接近 Hand PQ。这是由于使用了 SIMD 指令。

正如本节通过实验证明的那样，Bitonic TopK 是较小 K (K ≤ 256) 下性能最好的方法，而 Radix Select 适用于较大的 K。为了提供支持这些发现的分析论证，并预测在不同硬件上的性能，我们在第7节中开发了一个硬件感知的成本模型。

## 6.8 MapD 集成

为了评估 Bitonic TopK 在真实场景中带来的性能提升，我们在一个包含 2017 年 5 月的 2.5 亿条推文的 twitter 数据集上对该系统进行了评估。我们评估了四个查询：

1) SELECT id FROM tweets WHERE tweet\_time < X ORDER BY retweet\_count DESC LIMIT 50

该查询查找在指定时间范围内转发次数排名前 50 的推文。我们改变时间范围，使得选择率从 0 到 1，步长为 0.1。MapD 默认在时间范围上运行过滤器，然后在 GPU 上进行排序。然后它复制 top-k 推文 id 并组装推文（Filter+Sort）。我们评估了两种替代方案：1）用 bitonic top-k 替换排序（Filter+Bitonic TopK），2）将过滤器和 bitonic top-k 一起运行的组合 kernel（Combined Bitonic TopK）。图 16a 显示了结果。基于 bitonic top-k 的方法优于现有方法。过滤器融合优化节省了将过滤后的 id、retweet count 条目写出和读入全局内存的时间。在选择率为 1 时，过滤器融合优化将总 kernel 运行时间（在 GPU 上花费的时间）减少了 30%，将端到端运行时间减少了 23%。

![](images/3bbf3f845f8c2ed33b86ed7969bbdb64020241eb54742efda24f359351db04ec.jpg)

![](images/c2ee6569963d2b312f90c2606d90aa0ec1727a6ad4c2aca0a64ce9e7462d8a9b.jpg)  
(a) 获取时间范围内转发次数最多的 top-k 推文 (b) 查找时间范围内最流行的 top-k 推文  
图 16：MapD 实验

2) SELECT id FROM tweets ORDER BY retweet\_count + 0.5 \*likes\_count DESC LIMIT K

该查询基于复杂的排名函数查找最流行的推文。MapD 默认运行一个投影步骤来计算排名函数的值，随后是一个排序步骤（Project+Sort）。我们评估了两种替代方案：1）用 bitonic top-k 替换排序（Project+Bitonic TopK），2）在 SortReducer 内部计算排名函数值的组合 kernel（Combined Bitonic TopK）。图 16b 显示了结果。组合 kernel 节省了必须写出和读入投影排名值的时间。与 Project+Bitonic TopK 相比，这将组合方法的运行时间减少了 10ms。

3) SELECT id FROM tweets WHERE lang=’en’ OR lang=’es’ ORDER BY retweet\_count DESC LIMIT K

该查询查找英语或西班牙语中按转发次数排名的前 K 条推文。我们评估了与查询 (1) 中使用的相同的 3 种方法。过滤器的设定选择率约为 80%。我们看到了与上一个查询相同的趋势。组合 kernel 节省了必须读/写过滤后的 id、retweet count 条目的时间。与 Filter+Bitonic TopK 相比，这在所有 K 值下将运行时间减少了 16ms。 4) SELECT uid, COUNT() AS num\_tweets FROM tweets GROUP BY uid ORDER BY num\_tweets DESC LIMIT 50 该查询查找按推文数量排名的前 50 名用户。数据集中大约有 5700 万独立用户。在 MapD 默认情况下，查询执行耗时 97ms，其中排序步骤耗时 44ms。使用 bitonic top-k 将运行时间减少了 39%，因为它将排序步骤所耗时间减少了 38ms。一个查找例如 50 个最流行 hash 标签的查询不会从 bitonic top-k 中受益那么多，因为大部分时间都花在了 group by 步骤上。

## 7 成本模型

由于篇幅限制，我们将建模工作限制在性能最好的两种算法（见上一节）：Radix Select 和 Bitonic TopK。我们使用通过 Nvidia 提供的基准测试经验确定的硬件参数对它们进行建模。这些参数是 (1) 全局内存带宽 $( B _ { G } ) , ( 2 )$ 共享内存带宽 $( B _ { S } )$ ，(3) 以字节为单位的键大小，(4) 以字节为单位的输入数据大小 以及 (5) 总线程数。

### 7.1 基于 Radix 的 Top-K

基于 Radix 的 top-k（Radix Select）作为一系列 pass 运行，每个 pass 查看 8 位的一个数字。每个 pass 都会减少数据大小，pass 总数最多为 w/8。Pass i 包括：

• 从全局内存中读取该 pass 的输入，以写出每个线程每个数字值的条目数（总计：每个线程 16 个整数）。$D _ { i I }$ 是该 pass 的输入大小（以字节为单位）。对于第一个 pass，$D _ { i I } = D$。

$$
T _ { i 1 } = \frac { D _ { i I } } { B _ { G } } + \frac { 1 6 * 4 * n _ { t } } { B _ { G } }
$$

• 计算前缀和以找到包含第 k 个值的数字值 d。

$$
T _ { i 2 } = \frac { 2 * 1 6 * 4 * n _ { t } } { B _ { G } }
$$

• 扫描输入并将数字值为 d 的条目写出到全局内存中的另一个数组。设 η<sub>i</sub> 为数字值为 d 的条目的比例。请注意，如果 $\eta _ { i } = 1$，则跳过此步骤

$$
T _ { i 3 } = { \frac { D _ { i I } } { B _ { G } } } + \eta _ { i } { \frac { D _ { i I } } { B _ { G } } }
$$

Pass i 的总时间为 $T _ { i } = T _ { i 1 } + T _ { i 2 } + T _ { i 3 }$。总成本是各个 pass 所耗时间的总和。

## 7.2 Bitonic Top-K

Bitonic top-k 运行一系列 kernel：首先是 SortReducer kernel，接着是一系列 BitonicReducer kernel。设 x 为每个线程的元素数量。每个 kernel 将问题规模缩小 x 倍。对于每个 kernel，根据 K 的不同，有两个组件可能会主导性能：全局内存访问或共享内存访问。由于高并行性和低上下文切换开销，GPU 将有效地把两者中开销较低的那个隐藏在开销较高的那个之后。因此，其开销是两者中的最大值。

我们从 SortReducer kernel 的全局内存访问开销开始分析。该 kernel 对来自全局内存的输入进行一次扫描，并将输入的 $ 1 / x $ 写回（至全局内存）。因此，全局内存数据访问时间的建模非常直观：

$$
T _ { g } = \frac { D } { B _ { G } } + \frac { 1 } { x } \frac { D } { B _ { G } }
$$

共享内存数据访问时间的估算则较为困难：除了访问次数外，我们还需要考虑共享内存 bank 冲突的数量。由于只要访问同一个 bank 上的两个值就会发生 bank 冲突，我们需要考虑内存访问的具体地址。

如果 kernel 受限于共享内存带宽，所花费的时间是每个组合步骤所花费时间的总和：

$$
T _ { s } = \Sigma _ { i } \delta _ { i } \frac { D _ { I i } + D _ { O i } } { B _ { s } }
$$

![](images/e8d0ac1e397ea7b84abc57e3ed767e5070c7b73977ae5a527c7fc9b3fc6224ee.jpg)  
图17：估计运行时间与实际运行时间对比

其中 $\delta _ { i }$ 是一个 warp 的共享内存 bank 冲突数，$D _ { I i }$ 和 $D _ { O i }$ 分别是该阶段读取和写入的数据大小。将此应用于寻找 top-32 的 SortReducer 以求 $T _ { s }$，我们得到 $T _ { s } = 1 7 . 5 D / B _ { s }$

SortReducer kernel 的估计耗时为 max $( T _ { g } , T _ { s } )$ 。对于 Titan X Maxwell，$B _ { S } = 2 . 9 T B p$ s 且 $B _ { G } =$ 251GBps。估计总时间为 $m a x ( 8 . 9 6 m s , 1 2 . 1 m s ) =$ 12 1ms，这接近于实际运行时间 14 2ms。BitonicReducer 的开销可以用非常相似的方式进行估算，只是它直接从 len $= k / 2$ 开始

图17比较了在从 $U ( 0 , 1 )$ 提取的包含 $2 ^ { 2 9 }$ 个浮点数的数据集上，寻找不同 K 值的 top-k 时，各方法的实际时间与基于模型预测的时间。预测时间显示出与观察时间相同的趋势，且交点保持不变。两个模型都低估了所花费的时间。这是因为受限于全局或共享内存的 kernel 可能无法达到最大可能带宽。例如，基于基数排序的 top-k 的第一个 kernel 根据模型应耗时 8 6ms，而实际上耗时 9 8ms，并且 k = 32 时 SortReducer kernel 使用的有效共享内存带宽约为 2 5TBps，而最大为 2 9TBps。

如本节所示，bitonic top-k 不仅在实验上更快，而且在理论上也比我们评估的最佳替代方案更高效。

## 8 结论

GPU 上的数据分析日益普遍，而一项频繁的分析任务是根据某个属性对一组数据项进行排名，并提取 top-k 值。在本文中，我们提出了许多在 GPU 上高效计算 top-k 的算法，包括一种基于 bitonic sort 的新算法。通过对多种不同算法的广泛性能评估，我们展示了我们的 bitonic-top-k 算法比基于完全排序列表元素的最快算法快一个数量级，并且根据 k 值的不同，比其他几种高效计算 top-k 的算法快几倍。我们还提出了一个成本模型，该模型能准确预测多种算法在 k 值变化时的性能，从而允许查询优化器为特定查询选择最佳的 top-k 实现。

我们认为这一领域仍有创新空间。虽然我们已经深入研究了现有算法并提出了新算法，但我们有意将混合和自适应解决方案排除在本文范围之外。此类混合解决方案可能涉及多个设备（CPU 和 GPU），也可能是所提出算法的混合体。

## REFERENCES

[1] [n. d.]. Arrayfire discussion on top-k. http://bit.ly/2lLuFS1. ([n. d.]).

[2] [n. d.]. Issue to add gpu verion of top-k to tensorflow. https://github.com/ tensorflow/tensorflow/issues/5719. ([n. d.])

[3] Martín Abadi et al. 2016. TensorFlow: A system for large-scale machine learning. In OSDI.

[4] Tolu Alabi, Jefrey D Blanchard, Bradley Gordon, and Russel Steinbach. 2010. GGKS: Grinnell GPU k-selection. http://code.google.com/p/ggks/. (2010).

[5] Tolu Alabi, Jefrey D Blanchard, Bradley Gordon, and Russel Steinbach. 2012. Fast k-selection algorithms for graphics processing units. Journal ofExperimental Algorithmics (JEA) (2012).

[6] Kenneth E Batcher. 1968. Sorting networks and their applications. In Proceedings ofthe spring joint computer conference.

[7] Sebastian Breß, Henning Funke, and Jens Teubner. 2016. Robust query processing in co-processor-accelerated databases. In SIGMOD. ACM.

[8] Jatin Chhugani, Anthony D Nguyen, Victor W Lee, William Macy, Mostafa Hagog, Yen-Kuang Chen, Akram Baransi, Sanjeev Kumar, and Pradeep Dubey. 2008. Eficient implementation of sorting on multi-core SIMD CPU architecture. PVLDB (2008).

[9] Wenbin Fang, Bingsheng He, and Qiong Luo. 2010. Database compression on graphics processors. PVLDB (2010).

[10] Naga Govindaraju, Jim Gray, Ritesh Kumar, and Dinesh Manocha. 2006. GPUTeraSort: high performance graphics co-processor sorting for large database management. In SIGMOD.

[11] Mark Harris. 2007. Optimizing cuda. SC07: High Performance Computing With CUDA (2007).

[12] Max Heimel, Michael Saecker, Holger Pirk, Stefan Manegold, and Volker Markl. 2013. Hardware-oblivious parallelism for in-memory column-stores. PVLDB (2013).

[14] James Malcolm et al. 2012. ArrayFire: a GPU acceleration platform. In SPIE.

[13] Ihab F Ilyas, George Beskales, and Mohamed A Soliman. 2008. A survey of top-k query processing techniques in relational database systems. CSUR (2008).

[15] Duane Merrill and Andrew Grimshaw. 2011. High performance and scalable radix sorting: A case study of implementing dynamic parallelism for GPU computing. Parallel Processing Letters (2011).

[16] Todd Mostak. 2013. An overview of MapD (massively parallel database). White paper, Massachusetts Institute of Technology (2013).

[17] Hagen Peters, Ole Schulz-Hildebrandt, and Norbert Luttenberger. 2010. Fast in-place sorting with cuda based on bitonic sort. Parallel Processing and Applied Mathematics (2010), 403–410.

[18] Holger Pirk, Stefan Manegold, and Martin Kersten. 2014. Waste notâĂę Eficient co-processing of relational data. In ICDE. IEEE

[19] Holger Pirk, Oscar Moll, Matei Zaharia, and Sam Madden. 2016. Voodoo-a vector algebra for portable database performance on modern hardware. PVLDB (2016).

[20] Eyal Rozenberg and Peter Boncz. 2017. Faster across the PCIe bus: a GPU library for lightweight decompression: including support for patched compression schemes. In DaMoN. ACM.

[21] Nadathur Satish, Mark Harris, and Michael Garland. 2009. Designing eficient sorting algorithms for manycore GPUs. In Parallel & Distributed Processing, 2009. IPDPS 2009. IEEE International Symposium on. IEEE, 1–10.

[22] Elias Stehle and Hans-Arno Jacobsen. 2017. A Memory Bandwidth-Eficient Hybrid Radix Sort on GPUs. In SIGMOD. ACM.

[23] Yuan Yuan, Rubao Lee, and Xiaodong Zhang. 2013. The Yin and Yang of processing data warehousing queries on GPU devices. PVLDB (2013)



## 使用寄存器的每线程 Top-K

寄存器是内存层次结构中最快的一层。然而，如第 2.1 节所述，当前一代的 GPU 没有线程本地内存。只有当线程本地数组的所有访问都是静态已知时，才能使其使用寄存器。否则，编译器将被迫在片外本地内存中分配该数组，这会对性能产生严重的负面影响。这阻止了我们使用寄存器实现堆，因为在堆更新期间进行的数组访问无法静态确定。我们发现，通过将迄今为止看到的 top-k 维护为一个列表，并保留最小值的索引和值，我们仍然可以在寄存器中为每个线程维护 top-k。

T buf[k];

T minValue; int minIndex;

如果看到的元素大于 minValue，我们将更新 minIndex 并按如下方式找到新的 minIndex 和 minValue：

minValue = xi

for j in range(0,k):

if j == minIndex: buf[j] = xi

if buf[j] < minValue:

minIndex, minValue = j, buf[j]

虽然遍历缓冲数组 buf 的元素会产生 k 量级的开销，但它允许编译器将 buf 的元素放置在寄存器中。更快的数据访问抵消了 k 值较低时的开销。对于较高的 k 值，可用寄存器数量的限制迫使编译器将 buf 的一些条目分配在本地内存中，即使访问是按照上述方式实现的。

图 18 比较了基于寄存器的版本和基于共享内存的版本在从 $2 ^ { 2 9 }$ 个浮点数中寻找 top-k 时所花费的时间，其中 k 值是变化的。我们改变了分布： 均匀分布：从均匀分布 U(0 1) 中抽取的数字， 增序：从 U(0 1) 中抽取并按升序排序的数字， 降序：从 U (0<sub>,</sub> 1) 中抽取并按降序排序的数字。

对于较大的 k，基于寄存器的 top-k 比等效的基于共享内存的 top-k 方法更慢，因为基于寄存器的方法开始将寄存器溢出到本地内存，这导致了显著的减速。这从图中 k = 32 到 k = 64 的陡峭斜率可以明显看出。比较增序和降序分布，我们看到两种方法之间的差距扩大了。这是因为在增序中，每个数字都会更新 top-k。与堆相比，列表中的更新更为昂贵。在降序中，在插入前 k 个元素之后没有堆更新。

## B 重建算法

算法 4 展示了双调 top-k 重建操作的伪代码。

Algorithm 4: Bitonic Top-K Rebuild   
Input : 包含长度为 k 的双调序列的列表 L   
Output : 包含长度为 k 的已排序序列的 L   
1 int t = getGlobalThreadId();   
2 int len ← k ≫ 1;   
3 int dir ← len ≪ 1;   
4 for inc ← len; inc > 0; inc ← inc ≫ 1 do   
5 int low ← t & (inc − 1);   
6 int i ← (t ≪ 1) − low;   
7 bool reverse ← ((dir & i) == 0);   
8 x0, x1 ← L[i], L[i + inc];   
9 bool swap ← reverse ⊕ (x0 < x1) ;   
10 if swap: x0, x1 ← x1, x0;   
11 L[i], L[i + inc] ← x0, x1;

## C CPU 上的双调 Top-K

本文提出的双调 top-k 算法也可以适配在 CPU 上运行。双调 top-k 算法是归约性的，它将大小为 n 的数组归约为包含 top-k 元素的大小为 k 的数组。为了利用所有可用的核心，我们将输入数组划分为大小相等的分区，并让每个核心独立处理分区以输出 top-k。各个核心输出的 top-k 元素在最终的全局步骤中合并，以找到全局 top-k。

![](images/a22e580ce9e2c4525afda660fa6bcfeb873d4cc7a9cd755bfb675180f9407935.jpg)  
均匀分布

![](images/cbfcee84ce35e1a1ba22022ea06688ce1ecdbfac91d2e4c7b9b5a520eb12b169.jpg)  
增序

![](images/23cc8fbdb2a6b76c6a09fe90d720b4505b145211bb2efda7f4f6ce8722487258.jpg)  
降序

图 18：不同的每线程 Top-K 方法  
Algorithm 5: CPU Bitonic Top-K Thread   
Input : 长度为 n 的输入分区 S；int k   
Output : 每个线程的 top-k 元素列表 O   
1 int numElements ← n;   
2 int numVectors ← numElements / vectorSize;   
3 int temp[2][n/16];   
4 int current ← 0;   
5 for i ← 0; i < numVectors; i += 1 do   
6 SortReducer(S, temp[current], i, k)   
7 numElements ← numElements / 16;   
8 while numElements >= vectorSize do   
9 for i ← 0; i < numVectors; i += 1 do   
10 BitonicReducer(temp[current], temp[1-current], i, k);   
11 numElements ← numElements / 16;   
12 numVectors ← numElements / vectorSize;   
13 current ← 1 - current;   
14 O ← sort(temp[current], numElements);

在每个核心上，我们进一步将输入分区分解为固定大小的向量（在实现中，我们使用 2048 个元素作为向量大小）。我们分阶段处理输入分区。第一阶段执行 SortReducer 的功能。它一次读取一个向量的未排序输入分区，并输出包含长度为 k 的双调序列的输入的 $( 1 / 1 6 ) ^ { t h }$。后续阶段执行 BitonicReducer 的功能。它们一次读取一个包含长度为 k 的双调序列的输入向量，并输出包含长度为 k 的双调序列的输入的 $( 1 / 1 6 ) ^ { t h }$。算法 5 展示了伪代码。

在 GPU 上，每个向量由一个线程块并行处理。线程块中的每个线程从共享内存中读取 16 个元素，运行一个组合步骤并将其输出回共享内存。然而，在 CPU 上，我们在每个核心上以单线程方式处理向量。该线程一次从主内存中读取 16 个元素，运行一个组合步骤并将其输出回共享内存。我们处理小尺寸向量（此处大小为 2048）的原因是为了让数据被缓存在 L1 cache 中。这使得向量中的随机访问不会导致主内存读取的延迟。

现代 CPU 也支持单指令多数据流（SIMD）指令。用于处理组合步骤的双调排序网络可以使用 SIMD 指令来实现，以提高性能。在实现中，我们使用了 [8] 中基于 128 位 SSE 的实现。此外，第 4.3 节中详述的一些优化在 CPU 上是不需要的。特别是，填充和块置换在 CPU 上没有用，因为不存在 bank conflict 的概念。

双调 top-k 算法并不是工作高效的。如第 3.2 节所示，它进行了 $O ( n ( l o g k ) ^ { 2 } )$ 次比较。这严格劣于进行 O(nloдk) 次比较的基于堆的方法。然而，双调 top-k 可以利用 SIMD 指令来提高性能。总的来说，在发生大量堆插入的情况下（例如：当输入数据已排序时），尽管比较次数较多，双调 top-k 的性能仍接近基于堆的方法。此外，双调 top-k 在具有更宽向量指令支持的平台（如 Intel Knights Landing 处理器中的 AVX-512）上可能会更好。我们计划在未来探索这一点。