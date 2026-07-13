# TileBench: A Controlled Benchmark for Performance Evaluation and Bottleneck Diagnosis of Tile-Based Programming Models 图表详解

### c46cba47f4d618a2e72eb29c059a5c2144d83ca4d63ed90b4f4c0c5a37a4b113.jpg

![c46cba47f4d618a2e72eb29c059a5c2144d83ca4d63ed90b4f4c0c5a37a4b113.jpg](c46cba47f4d618a2e72eb29c059a5c2144d83ca4d63ed90b4f4c0c5a37a4b113.jpg)

- **图像类型与核心含义**

  - 该图是 **Triton 与 cuTile 相对 PyTorch 的 speedup 散点图**，对应论文中的 **Figure 2: Triton and cuTile speedup over PyTorch across operators**。
  - 每个点代表 **一个 TileBench operator**。
  - 横轴表示 **Triton speedup vs PyTorch**。
  - 纵轴表示 **cuTile speedup vs PyTorch**。
  - 坐标轴均为 **log scale**，因此点之间的距离反映的是倍数差异，而非线性差异。
  - 灰色虚线为 **y = x**：
    - 点在虚线上方：**cuTile 比 Triton 更快**。
    - 点在虚线下方：**Triton 比 cuTile 更快**。
    - 点接近虚线：两者性能接近。

- **图中元素说明**

  | 图形元素 | 含义 |
  |---|---|
  | **x-axis: Triton speedup vs PyTorch** | Triton 相对 PyTorch 的加速比 |
  | **y-axis: cuTile speedup vs PyTorch** | cuTile 相对 PyTorch 的加速比 |
  | **灰色虚线 y = x** | Triton 与 cuTile 性能相等边界 |
  | **绿色圆点** | **Stencil/Conv** |
  | **紫色方块** | **Matrix Mult./Attn** |
  | **红色菱形** | **Reduction/Norm** |
  | **黄色三角形** | **Point-wise** |
  | **青色星形** | **Data Layout** |
  | **横纵坐标 1× 附近** | 与 PyTorch 性能相当 |
  | **坐标大于 1×** | 快于 PyTorch |
  | **坐标小于 1×** | 慢于 PyTorch |

- **整体趋势**

  - 大多数点集中在 **x > 1、y > 1** 区域，说明 **Triton 和 cuTile 在多数 operator 上都快于 PyTorch**。
  - 散点整体沿着 **y = x** 分布，表明两个 DSL 在很多任务上的性能具有一定相关性：如果某个 operator 容易被 tile-based kernel 加速，通常 Triton 和 cuTile 都能取得一定收益。
  - 但大量点位于虚线下方，说明 **Triton 在整体上更强、更稳定**。
  - 少数点明显位于虚线上方，说明 **cuTile 在特定 workload 上有显著优势**，尤其是更适合 Blackwell tile/TMA/Tensor Core 路径的任务。

- **论文给出的关键统计结论**

  | 指标 | Triton | cuTile | 结论 |
  |---|---:|---:|---|
  | **Geometric-mean speedup vs PyTorch** | **2.7×** | **2.2×** | Triton 整体更快 |
  | **Median speedup vs PyTorch** | **3.1×** | **2.7×** | Triton 中位数也更高 |
  | **快于 PyTorch 的 operator 数量** | **36/45** | **34/45** | 两者多数情况下都优于 PyTorch |
  | **cuTile 快于 Triton 的 operator 数量** | — | **11/45** | cuTile 优势集中在少数任务 |
  | **Triton 慢于 PyTorch 的 operator 数量** | **9/45** | — | 少数任务 PyTorch 更优 |
  | **cuTile 慢于 PyTorch 的 operator 数量** | — | **11/45** | cuTile 低于 PyTorch 的情况略多 |

- **按区域解读**

  | 图中区域 | 性能含义 | 观察 |
  |---|---|---|
  | **右上区域，x > 1 且 y > 1** | 两个后端都快于 PyTorch | 大多数点落在此区域，是主趋势 |
  | **虚线下方** | Triton 快于 cuTile | 点数较多，体现 Triton 整体鲁棒性 |
  | **虚线上方** | cuTile 快于 Triton | 点数较少，但部分点差距明显 |
  | **左下区域，x < 1 且 y < 1** | 两者都慢于 PyTorch | 少量 operator，通常是不适合简单 tile 化或 PyTorch 已高度优化的任务 |
  | **右下区域，x > 1 且 y < 1** | Triton 快于 PyTorch，但 cuTile 慢于 PyTorch | 体现 Triton 对 irregular/lightweight kernel 更稳 |
  | **左上区域，x < 1 且 y > 1** | cuTile 快于 PyTorch，但 Triton 慢于 PyTorch | 较少见，表示 cuTile 对个别 tile-friendly workload 更合适 |

- **类别层面观察**

  | 类别 | 图中表现 | 解释 |
  |---|---|---|
  | **Matrix Mult./Attn** | 紫色方块分布跨度最大，既有高加速点，也有低于 1× 的点 | GEMM/Attention 对实现细节、Tensor Core、TMA、tile reuse 非常敏感 |
  | **Stencil/Conv** | 绿色圆点中有若干位于虚线上方 | cuTile 对规则 tile staging 和局部复用较友好 |
  | **Reduction/Norm** | 红色菱形多集中在 1× 到 10× 附近 | 两者均能加速，但 Triton 通常更稳 |
  | **Point-wise** | 黄色三角形多位于中等加速区间 | 主要受 HBM bandwidth 限制，Triton 指针模型较自然 |
  | **Data Layout** | 青色星形数量少但分布分散 | irregular access、scatter/gather、layout transform 对后端表达能力要求高 |

- **Triton 优势区域**

  - 图中较多点在 **y = x 下方**，说明 **Triton 在多数 operator 上快于 cuTile**。
  - Triton 的优势主要来自：
    - **pointer-level programming model 更灵活**。
    - 对 **runtime-computed indices**、**masked loads**、**boundary-dependent control** 表达更自然。
    - 对 **irregular access pattern** 和 **bandwidth-bound streaming kernels** 更稳。
  - 论文中特别提到 Triton 优势明显的 operator 包括：
    - **block_sparse_attention**
    - **flash_decode**
    - **linear_self_attention**
    - **weight_dequant**
  - 这些任务通常包含：
    - 稀疏布局
    - 动态索引
    - 非规则访存
    - 低比特 packing/unpacking
    - 边界 mask
    - 轻量级 memory-bound 计算

- **cuTile 优势区域**

  - 少数点明显位于 **y = x 上方**，说明 **cuTile 在部分 operator 上超过 Triton**。
  - cuTile 的优势集中在：
    - **tile-reuse-heavy workload**
    - **Tensor Core-friendly workload**
    - **TMA-friendly memory movement**
    - 规则矩阵块搬运与复用
  - 论文中提到 cuTile 表现较好的例子包括：
    - **matmul_fp32_fp16_fp8**
    - **1d_conv**
    - **2d_conv**
    - 部分 **dense matrix multiplication**
    - 部分 **attention**
    - 部分 **stencil/convolution**
  - 这些场景更符合 cuTile 的静态 tile 抽象：
    - 使用 **ct.load**
    - 使用 **ct.mma**
    - 使用 **ct.store**
    - 更容易映射到 **Blackwell-native TMA / Tensor Core path**

- **异常点与长尾现象**

  - 图中右上角存在极高加速点，说明某些 operator 中 PyTorch reference 较弱，而手写 tile kernel 可以获得 **几十倍甚至上百倍 speedup**。
  - Triton 横轴最右侧有超过 **100×** 的点，说明 Triton 在某些 operator 上对 PyTorch 有极强加速。
  - cuTile 纵轴上也有接近或超过 **80×** 的点，说明其在特定 workload 上也可以非常高效。
  - 左下角有少量点小于 **1×**，表示手写 Triton/cuTile kernel 反而慢于 PyTorch，可能原因包括：
    - operator 粒度太小，launch overhead 占比高。
    - PyTorch 调用了高度优化的 vendor/library kernel。
    - tile 参数或 lowering 不理想。
    - irregular kernel 中 cuTile 表达代价较高。
    - memory staging 额外开销超过收益。

- **该图支持的论文 RQ1 结论**

  | RQ1 问题 | 图中证据 | 结论 |
  |---|---|---|
  | Triton/cuTile 是否优于 PyTorch？ | 大多数点位于 **x > 1, y > 1** | 两者多数情况下都能加速 PyTorch |
  | 哪个后端整体更强？ | 更多点位于 **y = x 下方**；Triton 几何平均加速更高 | **Triton 整体更强** |
  | cuTile 是否有优势场景？ | 少数点明显在 **y = x 上方** | **cuTile 在 tile/TMA/TC-friendly workload 上有优势** |
  | 性能差异是否 workload-dependent？ | 不同颜色类别分布差异明显 | 性能强弱高度依赖 operator 结构 |

- **与论文诊断结论的关系**

  - 该图只是展示 speedup 分布，但背后的原因在论文后续 RQ3 中进一步解释。
  - **cuTile 胜出的本质原因**：
    - 对规则 tile 访问建模更贴近硬件。
    - 更容易使用 **TMA** 和 **tcgen05 / Tensor Core** 路径。
    - 在规则 GEMM、attention、stencil 场景中减少搬运和 staging 开销。
  - **Triton 胜出的本质原因**：
    - 对不规则地址、mask、动态索引的表达成本更低。
    - `tl.load` 可直接携带 predicate。
    - 避免 cuTile 中 `ct.gather`、`ct.where`、额外 staging 造成的依赖链。
  - 因此，图中的性能分化不是简单的“哪个 DSL 更高级”，而是 **programming model 与 operator memory/computation pattern 的匹配程度**。

- **对开发者的实际启示**

  | 场景 | 更推荐后端 | 原因 |
  |---|---|---|
  | **规则 GEMM / dense attention** | **cuTile 或 Triton 都可，cuTile 可能更优** | cuTile 更容易走 Blackwell-native tile path |
  | **TMA 可复用 tile movement** | **cuTile** | 静态 tile 抽象更适合 TMA |
  | **Stencil/Conv 且 tile reuse 明显** | **cuTile 可能更优** | 局部数据复用明显 |
  | **Sparse attention / dynamic indexing** | **Triton** | pointer + mask 表达更自然 |
  | **Point-wise / bandwidth-bound streaming** | **Triton** | 简单 `tl.load/tl.store` 避免额外 staging |
  | **Data Layout / gather/scatter-heavy** | **Triton** | 对 arbitrary pointer 更友好 |
  | **低比特 unpack / irregular quantization** | **Triton 通常更稳，但需看 lowering** | cuTile 可能因 gather/staging 复杂化 |

- **图像结论概括**

  - **Triton 和 cuTile 都能显著提升 PyTorch baseline 性能**。
  - **Triton 在 45 个 operator 的整体表现更强，速度更鲁棒**。
  - **cuTile 并非全面落后，而是在少数规则、tile-reuse-heavy、Tensor-Core/TMA-friendly workload 上表现突出**。
  - **性能差异高度依赖 workload 类型，而不是单一后端绝对优劣**。
  - 该图直接支撑论文核心观点：**tile-based DSL 的性能比较必须在多 operator、多 dtype、多 shape 的 controlled benchmark 下进行，不能只看 GEMM 或 attention 等少数任务。**

### 95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg

![95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg](95dfec873c7ae7fcbe730596092e7371185f93a2769559a2e7893c62d90d1ca1.jpg)

- **图像类型与核心含义**
  - 该图是 **Default-to-autotuned roofline utilization across operators**，用于比较 **Triton** 与 **cuTile** 在默认配置和 autotuned 配置下的 **roofline utilization R** 变化。
  - 横轴表示 **Default roofline utilization R**。
  - 纵轴表示 **Autotuned roofline utilization R**。
  - 每个点代表一个 operator 在某个后端下的平均 roofline utilization。
  - 蓝色圆点表示 **Triton**，橙色三角表示 **cuTile**。
  - 灰色虚线为 **y = x**，表示 autotuning 前后没有变化。
  - 从默认点到 autotuned 点之间的竖直箭头表示 **autotuning 带来的提升幅度**。

- **坐标轴与判读方式**

| 图中元素 | 含义 | 解释 |
|---|---:|---|
| x-axis | **Default roofline utilization R** | 默认配置下的硬件利用率 |
| y-axis | **Autotuned roofline utilization R** | autotuning 后的硬件利用率 |
| 灰色虚线 | **y = x** | autotuned 与 default 性能相同 |
| 点在虚线上方 | **autotuning 有收益** | autotuned R > default R |
| 点接近虚线 | **收益很小** | 默认配置已接近搜索空间内较优点 |
| 点远高于虚线 | **显著 autotuning 收益** | tile size、occupancy、num_warps 等参数选择影响明显 |
| R 接近 1.0 | **接近 roofline 上限** | 接近理论硬件性能边界 |
| R 低于 0.4 | **大量性能 headroom** | 仍远未充分利用硬件 |

- **整体趋势**
  - 大多数点位于 **y = x 上方或接近 y = x**，说明 autotuning 通常能提升 roofline utilization。
  - 但多数点仍集中在 **R = 0.2 到 0.8** 区间，说明即使经过 autotuning，许多 kernels 仍未接近硬件 roofline。
  - 图中只有少量点达到 **R ≥ 0.8**，表明高硬件利用率只出现在少数 operator 上。
  - 大量点沿对角线分布，说明 **autotuning 的平均收益是温和的，而不是根本性性能跃迁**。
  - 少数竖直箭头较长，说明某些 operator 对配置参数非常敏感。

- **Triton 表现分析**

| 观察项 | Triton 表现 |
|---|---|
| 标记 | **蓝色圆点** |
| 整体分布 | 覆盖从低 R 到高 R 的较宽范围 |
| autotuning 收益 | 多数有小到中等收益，少数收益显著 |
| 高利用率点 | 有多个点达到 **0.8–0.95** 区间 |
| 与论文数值对应 | 几何平均 autotune gain 为 **1.20×** |
| 达到 R ≥ 0.8 的数量 | **8/45** |
| 主要含义 | Triton 默认配置已较稳健，但 autotuning 仍能恢复部分 tile/occupancy 参数收益 |

- **cuTile 表现分析**

| 观察项 | cuTile 表现 |
|---|---|
| 标记 | **橙色三角** |
| 整体分布 | 与 Triton 类似，但高 R 点略少 |
| autotuning 收益 | 多数收益较小，个别 operator 有明显提升 |
| 高利用率点 | 少数点进入 **0.8–0.9** 区间 |
| 与论文数值对应 | 几何平均 autotune gain 为 **1.15×** |
| 达到 R ≥ 0.8 的数量 | **4/45** |
| 主要含义 | cuTile 的 autotuning 有帮助，但受限于 exposed tuning knobs 和 compiler lowering |

- **关键数值结论**

| 指标 | Triton | cuTile | 解读 |
|---|---:|---:|---|
| 几何平均 autotune gain | **1.20×** | **1.15×** | Triton 平均调优收益略高 |
| R ≥ 0.8 的 operator 数量 | **8/45** | **4/45** | Triton 达到高 roofline utilization 的 operator 更多 |
| 默认配置质量 | 较强 | 较强 | 默认配置不是弱 baseline |
| 剩余 headroom | 明显 | 更明显 | 参数调优不足以解决全部瓶颈 |

- **图中明显模式**
  - **低 R 区域**：
    - 左下角聚集了不少点，default R 和 autotuned R 都低于 **0.3**。
    - 这说明部分 operator 无论怎么调 tile 参数，都难以充分利用 B200 的计算或带宽能力。
    - 这类 kernel 往往可能受限于 **irregular memory access、control divergence、scatter/gather、low arithmetic intensity、compiler lowering**。
  - **中 R 区域**：
    - 大量点位于 **0.4–0.7** 区间。
    - 这些 operator 能获得一定硬件利用率，但仍有明显优化空间。
    - autotuning 在这里主要微调 **BLOCK_SIZE、num_warps、num_stages、TILE_SIZE、occupancy**。
  - **高 R 区域**：
    - 少数点接近 **0.9–1.0**。
    - 这些通常是更规则、更容易充分利用 memory bandwidth 或 Tensor Core 的 kernel。
    - 但数量较少，说明 TileBench 中多数 operator 并非简单 dense GEMM 型负载。

- **箭头长度的含义**
  - 短箭头：
    - 表示 default 配置已经比较接近 autotuned 配置。
    - 这也支持论文中的说法：default configurations 是 **manually selected reasonable baselines**，不是随机参数。
  - 长箭头：
    - 表示该 operator 对 tuning space 非常敏感。
    - 可能原因包括：
      - **tile size 改变 occupancy**
      - **num_warps 改变并行粒度**
      - **num_stages 改变 pipeline overlap**
      - **cuTile occupancy hint 改变 CTA residency**
      - **shared memory/register pressure 被重新平衡**

- **为什么 autotuning 收益有限**
  - 图中大部分点没有从低 R 直接跃迁到接近 1.0，说明 bottleneck 不只是参数选择。
  - 论文明确指出，Triton 和 cuTile 的 autotuning 范围主要覆盖：
    - Triton：**BLOCK_SIZE、num_warps、num_stages**
    - cuTile：**TILE_SIZE、occupancy**
  - 这些参数可以调节 tile shape、parallelism 和 occupancy，但通常不能改变：
    - **compiler lowering path**
    - **instruction selection**
    - **Tensor Core / TMA 是否被有效使用**
    - **shared-memory layout**
    - **register spilling**
    - **bank conflict**
    - **global memory coalescing**
  - 因此，autotuning 更像是局部搜索，而不是算法级或编译器级重写。

- **与 RQ2 的关系**
  - 该图直接支撑论文 RQ2 的回答：**Autotuning provides measurable gains, but substantial roofline headroom remains.**
  - 换言之：
    - autotuning 有用；
    - 但 autotuning 不是充分条件；
    - 许多性能差距来自 backend compiler 和 memory behavior，而不是单纯 tile 参数。

- **Triton 与 cuTile 的差异解读**
  - **Triton**
    - 在图中高 R 点更多。
    - 说明 Triton 在更广泛 operator 上更容易通过参数调优达到较好 roofline utilization。
    - 这与论文 RQ1 中 Triton 更适合 irregular、streaming、bandwidth-bound kernels 的结论一致。
  - **cuTile**
    - cuTile 的高 R 点更少，但也有若干接近高利用率区域的点。
    - 说明 cuTile 在适合其 static tile abstraction、TMA、Tensor Core 路径的 operator 上可以表现很好。
    - 但对不规则访问或 runtime-computed indexing 的 kernel，单纯调 TILE_SIZE 和 occupancy 难以解决根本瓶颈。

- **图中低利用率点的潜在原因**

| 潜在瓶颈 | 对 R 的影响 | 相关后端表现 |
|---|---|---|
| **Irregular memory access** | 降低 coalescing，增加 load latency | cuTile 更容易受影响 |
| **ct.gather / ct.where 链较重** | 增加 index、mask、staging 开销 | cuTile 常见 |
| **Pointer-tile masked load** | 表达不规则访问更自然 | Triton 相对有利 |
| **Tensor Core 未充分使用** | compute roofline 利用率低 | 两者都可能 |
| **TMA latency 未被隐藏** | stall_long_scoreboard 增多 | cuTile case study 中出现 |
| **Register pressure / spilling** | 降低 occupancy，增加 local memory traffic | 两者都可能 |
| **Shared-memory bank conflict** | 影响 shared memory throughput | 两者都可能 |
| **Compiler lowering 不理想** | 无法走 Blackwell-native path | Triton 在部分低比特 GEMM 中明显 |

- **图像给出的核心结论**
  - **autotuning 确实提升 Triton 和 cuTile 的 roofline utilization**。
  - **Triton 的平均 autotuning 收益略高于 cuTile：1.20× vs 1.15×**。
  - **绝大多数 operator 即使 autotuned 后仍没有接近 roofline 上限**。
  - **Triton 有更多 operator 达到 R ≥ 0.8，说明其整体调优后性能更稳健**。
  - **cuTile 在部分 regular tile-reuse-heavy kernel 上有竞争力，但总体 high-utilization 覆盖面较窄**。
  - **性能瓶颈主要不在 tuning space，而在 backend lowering、memory staging、instruction selection 和硬件路径映射**。

- **一句话总结**
  - 该图表明：**autotuning 能带来稳定但有限的 roofline utilization 提升；Triton 整体收益和高利用率覆盖更好，而 cuTile 的瓶颈更多受 compiler lowering 与 memory/tile abstraction 适配性的限制。**

### f9e643fecb6869d7899b6b7a50bbcaf63d4acc756e36ecba362971843a1ab893.jpg

![f9e643fecb6869d7899b6b7a50bbcaf63d4acc756e36ecba362971843a1ab893.jpg](f9e643fecb6869d7899b6b7a50bbcaf63d4acc756e36ecba362971843a1ab893.jpg)

- **图像核心含义**：该图展示了 TileBench 中 **Triton 与 cuTile 在 default mode 下的 Top-20 最大延迟差距**。每一行是一个 **operator–dtype–input size** 组合，横轴为 **slower-backend latency / faster-backend latency**，即慢的一方比快的一方慢多少倍。数值越大，说明两个 DSL 的性能差距越明显。

- **坐标与编码说明**：

| 元素 | 含义 |
|---|---|
| 图标题 | **Top 20 Triton vs cuTile latency gaps (default mode, sweep-max input)** |
| 横轴 | **Default-mode latency gap (slower / faster) [×]** |
| 横轴尺度 | 对数尺度，标注约为 **1×、2×、3×、4×、6×、8×** |
| 纵轴 | Top-20 最大差距的 **operator / dtype / input-size** |
| 蓝色标签 | **Triton faster** |
| 橙色标签 | **cuTile faster** |
| 标签数值 | 慢端延迟 / 快端延迟，例如 **Triton 5.8×** 表示 Triton 比 cuTile 快，cuTile 延迟约为 Triton 的 5.8 倍 |

- **Top-20 性能差距概览**：

| 排名趋势 | Operator / dtype / size | 更快后端 | 延迟差距 |
|---:|---|---|---:|
| 1 | **matmul_fp32_fp16_fp8 / fp32 / K=20480** | **cuTile** | **6.1×** |
| 2 | **2d_conv / fp32 / H=320** | **Triton** | **5.8×** |
| 3 | **streamk_matmul / bf16 / m=8192, n=28672** | **Triton** | **5.7×** |
| 4 | **streamk_matmul / fp16 / m=8192, n=28672** | **Triton** | **5.6×** |
| 5 | **linear_self_attention / fp32 / D=256, M=10000** | **Triton** | **3.8×** |
| 6 | **flash_decode / fp32 / seq_len=40960** | **Triton** | **3.8×** |
| 7 | **block_sparse_attention / fp16 / M=10240** | **Triton** | **3.7×** |
| 8 | **mean_reduction / fp16 / N=20480** | **Triton** | **3.3×** |
| 9 | **gaussian_blur / fp32 / input_rows=10240** | **Triton** | **3.2×** |
| 10 | **gaussian_blur / fp16 / input_rows=10240** | **Triton** | **2.9×** |
| 11 | **weight_dequant / bf16 / M=10240** | **Triton** | **2.9×** |
| 12 | **weight_dequant / fp16 / M=10240** | **Triton** | **2.8×** |
| 13 | **matmul_fp32_fp16_fp8 / fp8_e4m3fn / K=20480** | **cuTile** | **2.5×** |
| 14 | **1d_conv / fp32 / n=20.0M** | **cuTile** | **2.5×** |
| 15 | **2d_conv / fp16 / H=320** | **Triton** | **2.5×** |
| 16 | **2d_max_pooling / fp16 / H=640** | **Triton** | **2.4×** |
| 17 | **2d_max_pooling / fp32 / H=640** | **Triton** | **2.4×** |
| 18 | **2d_max_pooling / bf16 / H=640** | **Triton** | **2.4×** |
| 19 | **matmul_fp32_fp16_fp8 / fp8_e5m2 / K=20480** | **cuTile** | **2.3×** |
| 20 | **weight_dequant / fp32 / M=10240** | **Triton** | **1.8×** |

- **总体分布结论**：

| 更快后端 | Top-20 中数量 | 典型优势场景 |
|---|---:|---|
| **Triton faster** | **16 / 20** | irregular access、streaming、mask load、runtime index、bandwidth-bound kernel |
| **cuTile faster** | **4 / 20** | Tensor Core / TMA-friendly、规则 tile reuse、dense GEMM、部分 convolution |
| 最大差距 | **cuTile 6.1× faster** | matmul_fp32_fp16_fp8 / fp32 |
| Triton 最大差距 | **Triton 5.8× faster** | 2d_conv / fp32 |

- **最显著现象 1：Triton 在 Top-20 中占绝对多数优势**
  - 图中 **蓝色条目明显多于橙色条目**，说明在最大性能差距案例中，更多场景是 **Triton 明显快于 cuTile**。
  - 这与论文 RQ1 的总体结论一致：**Triton 在 aggregate performance 上更强，default mode 下几何平均 speedup 为 2.7×，高于 cuTile 的 2.2×**。
  - 图中 Triton 的优势集中在：
    - **2d_conv / fp32**
    - **streamk_matmul / bf16、fp16**
    - **linear_self_attention**
    - **flash_decode**
    - **block_sparse_attention**
    - **weight_dequant**
    - **2d_max_pooling**
  - 这些任务普遍包含 **边界判断、非规则访存、runtime-computed indices、低复用 streaming pattern 或 memory-bound 行为**。

- **最显著现象 2：cuTile 的胜利虽少，但最大单点优势最强**
  - 排名第一的案例是 **matmul_fp32_fp16_fp8 / fp32 / K=20480**，cuTile 快 **6.1×**。
  - 这说明 cuTile 在高度匹配其抽象的任务上可以非常强，尤其是：
    - **规则 tile**
    - **Tensor Core-friendly**
    - **TMA-friendly**
    - **可复用 tile movement**
    - **Blackwell-native memory path**
  - 图中 cuTile 更快的 4 个案例分别来自：
    - **matmul_fp32_fp16_fp8 / fp32**
    - **matmul_fp32_fp16_fp8 / fp8_e4m3fn**
    - **1d_conv / fp32**
    - **matmul_fp32_fp16_fp8 / fp8_e5m2**
  - 这表明 cuTile 的优势并非广泛分布，而是集中在 **GEMM / convolution-like regular computation**。

- **算子类别层面的解释**：

| 算子类别 | 图中代表 | 优势后端 | 原因 |
|---|---|---|---|
| **Dense GEMM / Tensor Core** | matmul_fp32_fp16_fp8 | **cuTile** | ct.load、ct.mma、ct.store 更容易映射到 Blackwell-native TMA / Tensor Core 路径 |
| **Stream-K GEMM** | streamk_matmul | **Triton** | cuTile 可能暴露 TMEM / MMA pipeline wait，导致 stall_long_sb |
| **Irregular Attention** | flash_decode、block_sparse_attention、linear_self_attention | **Triton** | Triton pointer-level model 更适合 arbitrary address 与 masked load |
| **Stencil / Conv** | 2d_conv、gaussian_blur、1d_conv | 混合 | 规则 tile 时 cuTile 有优势；virtual-im2col / gather-mask-stage 复杂时 Triton 更优 |
| **Pooling / Reduction** | mean_reduction、2d_max_pooling | **Triton** | 简单 streaming / reduction 更适合 Triton 的轻量级 pointer load |
| **Quantization / Dequantization** | weight_dequant | **Triton** | runtime index、低比特 unpack、非规则访存更适合 Triton |

- **2d_conv / fp32：Triton 5.8× 优势的含义**
  - 图中第二大差距是 **2d_conv / fp32 / H=320**，Triton 快 **5.8×**。
  - 论文诊断指出，该场景中两者都不是 Tensor Core 路径，而是 **true-FP32 arithmetic path**。
  - 差距主要来自 **virtual-im2col operands materialization**：
    - **Triton**：直接构造 pointer tile，并把 boundary predicate 绑定到 **tl.load**。
    - **cuTile**：更依赖 **ct.gather + ct.where + staging**，会产生更重的 index computation、indirect load、mask selection、local/shared memory staging。
  - 因此，该案例体现了 **Triton 对非规则地址表达更自然**，而 cuTile 在偏离规则 tile-space access 时开销更大。

- **streamk_matmul：Triton 5.7× / 5.6× 优势的含义**
  - **streamk_matmul / bf16** 和 **streamk_matmul / fp16** 分别显示 Triton 快 **5.7×** 和 **5.6×**。
  - 论文的 Nsight Compute 分析指出，cuTile 在此类场景中可能出现 **Tensor Core / TMEM pipeline 等待**。
  - 关键瓶颈不是普通分支，而是：
    - stalled predicated branch
    - predicate 由 **SYNCS.PHASECHK.TRANS64.TRYWAIT** 产生
    - SASS 中出现 **STTM、FENCE.VIEW.ASYNC.T、SYNCS.ARRIVE**
  - 这说明 cuTile 的异步 MMA / TMEM pipeline overlap 不足，导致 **stall_long_sb**。

- **attention 系列：Triton 在 irregular attention 上更稳**
  - 图中 **linear_self_attention / fp32**、**flash_decode / fp32**、**block_sparse_attention / fp16** 均为 Triton 更快，差距约 **3.7×–3.8×**。
  - 这些 attention 变体通常具有：
    - 非规则访问
    - 动态 mask
    - sequence-dependent control
    - sparse block layout
    - KV cache / decode-oriented memory pattern
  - 这类模式更适合 Triton 的 **pointer arithmetic + tl.load(mask=...)**。
  - cuTile 的静态 tile abstraction 在此类场景中容易引入额外 gather、mask、staging 成本。

- **weight_dequant：Triton 在多个 dtype 上稳定领先**
  - 图中 **weight_dequant** 出现三次：
    - **bf16：Triton 2.9×**
    - **fp16：Triton 2.8×**
    - **fp32：Triton 1.8×**
  - 说明 Triton 对 **dequantization / low-bit unpack / scale application** 这类轻量但索引复杂的 kernel 更稳。
  - 这类 workload 通常不是纯算力瓶颈，而是受：
    - memory bandwidth
    - bit manipulation
    - pointer indirection
    - register pressure
    - mask handling
    影响更大。

- **matmul_fp32_fp16_fp8：cuTile 在多个 dtype 上重复领先**
  - 图中 cuTile 胜出的 4 个案例中有 3 个来自 **matmul_fp32_fp16_fp8**：
    - **fp32：cuTile 6.1×**
    - **fp8_e4m3fn：cuTile 2.5×**
    - **fp8_e5m2：cuTile 2.3×**
  - 这说明该 matmul 实现高度契合 cuTile 的设计优势：
    - **ct.mma**
    - **Tensor Core**
    - **TMA**
    - **static tile staging**
    - **regular reusable tile movement**
  - 与论文 Appendix E 的 matmul_int8 诊断类似，cuTile 更可能走到 **Blackwell-native path**，如 **TMA / tcgen05 / TMEM accumulator**，而 Triton 某些场景可能退化到较传统的 staging 路径。

- **gaussian_blur 与 pooling：Triton 在 stencil-like memory-bound 场景中表现更好**
  - **gaussian_blur / fp32**：Triton 快 **3.2×**
  - **gaussian_blur / fp16**：Triton 快 **2.9×**
  - **2d_max_pooling / fp16、fp32、bf16**：Triton 均快约 **2.4×**
  - 这些算子虽然属于 stencil / pooling，但不一定有足够高的 tile reuse 来抵消 cuTile 的 staging 成本。
  - 对于一次性访问或低复用窗口操作，Triton 的轻量级 load/store 往往更合适。

- **图像反映出的核心 trade-off**：

| 维度 | Triton | cuTile |
|---|---|---|
| 抽象风格 | **pointer-level block programming** | **static tile-level programming** |
| irregular address | **强** | 较弱，常需 ct.gather / ct.where |
| masked load | **自然，tl.load(mask=...)** | 可能需要额外 tile selection / staging |
| Tensor Core / TMA 映射 | 有时不如 cuTile 直接 | **强，适合 Blackwell-native path** |
| streaming kernel | **通常更稳** | 可能 staging overhead 较高 |
| regular tile reuse | 表现好但不总是最优 | **优势明显** |
| LLM 生成友好度 | **更成熟、更 token-efficient** | 较新，生成难度更高 |

- **与论文主结论的对应关系**
  - 图 4 是 RQ3 的核心证据之一，用于解释 **why performance gaps happen**。
  - 它支持论文中的两个主要判断：
    - **cuTile excels on a small cluster of Tensor-Core/TMA-friendly kernels**。
    - **Triton is stronger on many irregular, streaming, and bandwidth-bound operators**。
  - 图中蓝色条目占多数，说明在最大差距案例中，**Triton 的鲁棒性优势更常出现**。
  - 但最高单点是橙色 **cuTile 6.1×**，说明 cuTile 并非弱，而是 **高度 workload-dependent**。

- **对开发者的实践启示**
  - 如果算子是 **dense GEMM、FP8/INT8 matmul、regular attention、large tile reuse、TMA-friendly**，应优先尝试 **cuTile**。
  - 如果算子包含 **dynamic indexing、sparse layout、decode attention、dequantization、mask-heavy load、boundary-dependent control**，应优先尝试 **Triton**。
  - 对 stencil / convolution 类算子不能简单归类：
    - **规则、可复用、tile-staged convolution** 可能适合 cuTile。
    - **virtual-im2col、gather-heavy、mask-heavy convolution** 更可能适合 Triton。
  - 性能差距不能只靠 tile size autotuning 消除，因为很多瓶颈来自 **compiler lowering、instruction selection、shared-memory staging、register spilling、bank conflict、async pipeline overlap**。

- **一句话总结**
  - 该图表明：**Triton 在 Top-20 最大性能差距中更常胜出，尤其适合 irregular / streaming / bandwidth-bound workload；cuTile 胜出次数较少但在 Tensor Core / TMA-friendly regular tile workload 上可取得最大单点优势。**

### 336730bb607696cb1e402d8e5305e120d7e5b75eee5b3cbb91f30035d728ffb6.jpg

![336730bb607696cb1e402d8e5305e120d7e5b75eee5b3cbb91f30035d728ffb6.jpg](336730bb607696cb1e402d8e5305e120d7e5b75eee5b3cbb91f30035d728ffb6.jpg)

- **图像主题**：该图展示 TileBench 论文中 **LLM-generated kernels** 轨道的结果，对比两种编程后端 **Triton** 与 **cuTile** 在两个大模型 **GPT-5.5** 和 **Claude Opus 4.7** 下的 **TokenCost@10** 与 **TokenEfficiency@10**。

- **图像结构**：
  - 左半部分为 **TokenCost@10**：
    - 衡量 10 次迭代 refinement 中平均消耗的 token 数。
    - 单位为 **M tokens**。
    - 数值越低，表示生成和优化代码所需 token 越少。
  - 右半部分为 **TokenEfficiency@10**：
    - 衡量单位 token 带来的有效性能收益。
    - 定义为：
      **BestSpeedup@10 / TokenCost@10(M tokens)**
    - 数值越高，表示 LLM 用更少 token 生成更高性能 kernel 的能力越强。
  - 蓝色柱表示 **Triton**。
  - 橙色柱表示 **cuTile**。
  - 右半部分柱子带斜线纹理，用于区分 TokenEfficiency@10 指标。

- **核心数据汇总**：

| Model | Backend | TokenCost@10 | TokenEfficiency@10 |
|---|---:|---:|---:|
| **GPT-5.5** | **Triton** | **0.26M tokens** | **20.9** |
| **GPT-5.5** | **cuTile** | **0.32M tokens** | **13.4** |
| **Claude Opus 4.7** | **Triton** | **0.28M tokens** | **17.1** |
| **Claude Opus 4.7** | **cuTile** | **0.42M tokens** | **8.3** |

- **TokenCost@10 观察**：
  - **Triton 的 token 成本始终低于 cuTile**。
  - 对 **GPT-5.5**：
    - Triton 平均消耗约 **0.26M tokens**。
    - cuTile 平均消耗约 **0.32M tokens**。
    - cuTile 比 Triton 多消耗约 **23%** token。
  - 对 **Claude Opus 4.7**：
    - Triton 平均消耗约 **0.28M tokens**。
    - cuTile 平均消耗约 **0.42M tokens**。
    - cuTile 比 Triton 多消耗约 **50%** token。
  - 这说明在相同 10 轮 refinement budget 下，**cuTile 更容易导致更长的代码生成、更多修复尝试或更复杂的提示反馈消耗**。

- **TokenEfficiency@10 观察**：
  - **Triton 的 token efficiency 明显高于 cuTile**。
  - 对 **GPT-5.5**：
    - Triton 为 **20.9**。
    - cuTile 为 **13.4**。
    - Triton 约为 cuTile 的 **1.56×**。
  - 对 **Claude Opus 4.7**：
    - Triton 为 **17.1**。
    - cuTile 为 **8.3**。
    - Triton 约为 cuTile 的 **2.06×**。
  - 这表明 LLM 使用 Triton 时，不仅消耗更少 token，而且更容易产生高性能、正确的 kernel。

- **模型维度对比**：

| 对比项 | 结论 |
|---|---|
| **GPT-5.5 vs Claude Opus 4.7 on Triton** | GPT-5.5 token 成本更低，效率更高 |
| **GPT-5.5 vs Claude Opus 4.7 on cuTile** | GPT-5.5 明显优于 Claude，尤其在 TokenEfficiency@10 上 |
| **Triton vs cuTile for GPT-5.5** | Triton 更省 token，效率更高 |
| **Triton vs cuTile for Claude Opus 4.7** | Triton 优势更大，cuTile token 成本最高、效率最低 |

- **最优组合**：
  - **GPT-5.5 + Triton** 是图中表现最好的组合：
    - TokenCost@10 最低之一：**0.26M tokens**。
    - TokenEfficiency@10 最高：**20.9**。
  - 说明 GPT-5.5 对 Triton 的代码生成能力最成熟，能够以较低 token 成本达到较高性能。

- **最差组合**：
  - **Claude Opus 4.7 + cuTile** 是图中表现最弱的组合：
    - TokenCost@10 最高：**0.42M tokens**。
    - TokenEfficiency@10 最低：**8.3**。
  - 这说明 Claude 在 cuTile 目标上的生成过程更不稳定或更低效，可能需要更多迭代修复，却难以获得相应性能提升。

- **后端差异大于模型差异**：
  - 图中最重要的结论是：**backend choice 的影响大于 model choice**。
  - 无论使用 GPT-5.5 还是 Claude Opus 4.7：
    - Triton 都比 cuTile **更省 token**。
    - Triton 都比 cuTile **token efficiency 更高**。
  - 这与论文正文结论一致：**Triton is easier and more token-efficient for LLM-generated kernels than cuTile**。

- **原因分析**：
  - **Triton 更成熟**：
    - Triton 已在 vLLM、LightLLM、Liger-Kernel、Unsloth 等生产系统中广泛使用。
    - LLM 预训练语料中更可能包含大量 Triton 示例代码。
    - 因此模型对 Triton API、常见 kernel 模式、masking、tl.load、tl.store、tl.dot 等更熟悉。
  - **cuTile 更新且语料稀缺**：
    - cuTile 是较新的 CUDA Tile 编程接口。
    - 公开代码较少，LLM prior knowledge 不足。
    - 生成 cuTile kernel 时更容易出现 API 误用、tile shape 约束错误、padding 处理错误或性能配置不佳。
  - **cuTile 抽象更依赖静态 tile 结构**：
    - cuTile 对规则 tile、Tensor Core、TMA-friendly workload 更友好。
    - 但 LLM 在面对复杂或不规则 operator 时，可能更难正确表达 ct.load、ct.store、ct.gather、ct.scatter、ct.where 等逻辑。
  - **Triton pointer-level model 更适合 LLM 生成**：
    - Triton 使用显式 pointer arithmetic 与 mask load/store。
    - 对 LLM 来说，这种模式更接近已有 CUDA/Triton 代码范式。
    - 因此更容易在少量迭代内生成正确且性能可接受的实现。

- **与论文 RQ4 的关系**：
  - 该图直接回答 **RQ4: LLM Usability and Token Efficiency**。
  - 结论是：
    - **Triton 更适合作为 LLM-generated GPU kernels 的目标 DSL**。
    - **cuTile 虽然在部分人工实现的 Tensor Core/TMA-friendly kernels 上表现强，但作为 LLM 代码生成目标仍不够成熟**。
    - 当前阶段，cuTile 的瓶颈不是单纯 refinement iteration 不够，而是 LLM 对该 DSL 的先验知识不足。

- **实验含义**：
  - 对研究者：
    - 评估 GPU DSL 时不能只看人工实现性能，还应关注 **LLM usability**。
    - 一个后端即使硬件映射能力强，如果 LLM 难以生成正确高效代码，也会影响实际自动化开发价值。
  - 对工具开发者：
    - cuTile 需要更多公开样例、文档、模板和错误反馈机制。
    - 若未来要提升 cuTile 的 LLM 生成能力，应重点补充高质量训练语料和标准 kernel patterns。
  - 对工程实践：
    - 当前若使用 LLM 辅助生成 GPU kernel，**Triton 是更稳妥的目标后端**。
    - cuTile 更适合由熟悉 Blackwell、TMA、Tensor Core pipeline 的专家手写或深度调优。

- **总体结论**：
  - 图中结果清晰表明：**Triton 在 LLM 代码生成场景下同时具备更低 token 成本和更高 token 效率**。
  - **cuTile 在 LLM usability 上明显落后**，尤其在 Claude Opus 4.7 上差距更大。
  - 因此，TileBench 不仅揭示了 Triton 与 cuTile 的运行性能差异，也揭示了二者作为 **LLM-programmable DSL** 的可用性差异。

### 56a81d119d5f0e2475b0ac4c118839dcfc5eb04af7067a25c7639e684180dcf7.jpg

![56a81d119d5f0e2475b0ac4c118839dcfc5eb04af7067a25c7639e684180dcf7.jpg](56a81d119d5f0e2475b0ac4c118839dcfc5eb04af7067a25c7639e684180dcf7.jpg)

- **图片对象**：该图对应论文附录 Figure 8(a)，主题是 **Default mode speedup distribution across operators**，展示 **Triton** 与 **cuTile** 在默认配置下相对于 **PyTorch** 的加速比分布。

- **图表类型**：  
  | 图表元素 | 含义 |
  |---|---|
  | **箱线图 Boxplot** | 展示每个 backend 在 45 个 operator 上的 speedup 分布 |
  | **散点 Strip plot** | 每个点代表一个 operator 的平均 speedup |
  | **对数 y 轴** | Speedup 跨度很大，从低于 0.1× 到超过 100× |
  | **虚线 y=1** | 与 PyTorch 持平的基准线，超过 1× 表示快于 PyTorch |
  | **蓝色 Triton** | Triton 默认配置结果 |
  | **橙色 cuTile** | cuTile 默认配置结果 |

- **坐标轴解读**：  
  | 坐标轴 | 内容 |
  |---|---|
  | **x 轴** | 两个 backend：**Triton** 与 **cuTile** |
  | **y 轴** | **Speedup vs PyTorch (×)** |
  | **y 轴刻度** | 对数尺度，主要刻度约为 **0.1×、1×、10×、100×** |
  | **1× 虚线** | PyTorch 性能基线 |

- **核心结论**：  
  | 对比项 | Triton | cuTile | 观察 |
  |---|---:|---:|---|
  | **几何平均 speedup** | **2.7×** | **2.2×** | Triton 整体更强 |
  | **中位数 speedup** | **3.1×** | **2.7×** | Triton 中位表现更好 |
  | **快于 PyTorch 的 operator 数** | **36/45** | **34/45** | 两者多数场景均优于 PyTorch |
  | **慢于 PyTorch 的 operator 数** | **9/45** | **11/45** | cuTile 慢于 PyTorch 的情况略多 |
  | **最高尾部** | **超过 100×** | **超过 80×** | 两者都有极高加速 outlier |

- **Triton 分布特征**：  
  - **Triton 的箱体整体略高于 cuTile**，说明其默认配置下的典型 operator 表现更好。  
  - **中位线约在 3× 左右**，与正文给出的 **3.1× median speedup** 一致。  
  - 大多数 Triton 点分布在 **1× 到 10×** 区间，说明多数 operator 能稳定超过 PyTorch。  
  - 存在少数低于 **1×** 的点，最低约接近 **0.1×**，代表部分 operator 上 Triton 默认 kernel 不如 PyTorch。  
  - 上方有明显长尾，最高点超过 **100×**，说明某些 PyTorch reference 很弱，Triton 自定义 kernel 能获得极大收益。

- **cuTile 分布特征**：  
  - **cuTile 的中位线略低于 Triton**，约在 **2–3×** 范围。  
  - 大部分 cuTile 点同样集中在 **1× 到 10×**，说明 cuTile 在多数 operator 上也能超过 PyTorch。  
  - 低于 **1×** 的点数量略多，且最低点也明显低于 PyTorch，说明默认配置鲁棒性稍弱。  
  - cuTile 也有高加速长尾，最高点接近或超过 **80×**，说明在适合其 tile abstraction 的场景中收益很高。  
  - 其分布上界低于 Triton 的最高 outlier，但整体仍表现出强于 PyTorch 的趋势。

- **箱线图细节解释**：  
  | 统计特征 | Triton | cuTile | 含义 |
  |---|---|---|---|
  | **箱体位置** | 略高 | 略低 | Triton 典型 speedup 更高 |
  | **箱体跨度** | 较宽 | 较宽 | operator 间差异明显 |
  | **须线范围** | 从低于 1× 延伸到十几倍 | 从远低于 1× 延伸到接近十倍 | 两者均存在强 workload-dependence |
  | **离群点** | 超过 100× | 接近 100× | 高加速由少数 operator 驱动 |
  | **低端点** | 约 0.1× | 约 0.05× | 某些默认 kernel 远慢于 PyTorch |

- **与论文 RQ1 的关系**：  
  - 该图是主文 Figure 2 的补充视角，重点从 **分布统计** 角度说明默认模式下的性能差异。  
  - 主文结论是：**Triton 与 cuTile 都显著优于 PyTorch，但 Triton aggregate performance 更强**。  
  - 图中可见两组分布高度重叠，因此论文强调 **neither abstraction is uniformly superior**。  
  - Triton 的优势不是绝对碾压，而是体现在 **median、geomean、快于 PyTorch 的 operator 数量** 等整体指标上。

- **对数尺度的重要性**：  
  - y 轴使用 log scale，说明 speedup 的跨度非常大。  
  - 如果使用线性坐标，低速与中等 speedup 的差异会被高 outlier 淹没。  
  - 对数轴更适合展示 **0.1× 到 100×** 这种跨三个数量级的性能分布。  
  - 这也表明 TileBench 中 operator 的性能行为非常异质，不能仅用单个平均值概括。

- **低于 1× 的点说明什么**：  
  - 低于 **1×** 表示自定义 Triton/cuTile kernel 慢于 PyTorch reference。  
  - 这通常可能来自：  
    - **kernel launch overhead** 在小规模 case 中占比过高。  
    - 默认 tile size 与实际输入形状不匹配。  
    - backend compiler lowering 不理想。  
    - memory access pattern 不适合 tile-based abstraction。  
    - PyTorch 内部调用了高度优化的 library kernel。  
  - 图中两者都有低于 1× 的点，说明 **tile-based DSL 并不自动保证性能优势**。

- **高于 10× 甚至 100× 的点说明什么**：  
  - 高 speedup 通常意味着 PyTorch reference 存在较高框架开销、未融合操作、或不适合该 operator 的通用实现。  
  - 自定义 kernel 通过 **fusion、direct memory access、tile-level computation、减少中间张量** 等方式获得大幅收益。  
  - 但这些 outlier 不代表所有 operator 都有同等收益，因此论文同时报告了 **median** 和 **geometric mean**。

- **Triton 相对优势的可能来源**：  
  - **pointer-level programming model** 更适合表达 irregular access、masked load、runtime-computed indices。  
  - 在 lightweight、streaming、bandwidth-bound operator 上通常更稳健。  
  - 对 LLM / AI kernel 社区更成熟，已有更多优化经验与 compiler 支持。  
  - 对默认配置而言，Triton 的通用性使其在多类型 operator 上更不容易严重退化。

- **cuTile 相对劣势的可能来源**：  
  - cuTile 更依赖规则 tile abstraction。  
  - 对 irregular access、scatter/gather、boundary-heavy control flow 的表达成本较高。  
  - 当使用 **ct.gather、ct.where、mask staging** 等路径时，可能引入额外寄存器压力、shared/local memory traffic。  
  - 默认配置下部分 workload 不能充分触发 Blackwell-native TMA / Tensor Core 优势。

- **但 cuTile 并非弱势 backend**：  
  - 图中 cuTile 仍有大量点高于 **1×**，说明其默认实现多数情况下优于 PyTorch。  
  - cuTile 的高端 outlier 接近 **100×**，表明在适合场景下极具潜力。  
  - 论文正文指出 cuTile 在 **dense matmul、attention、stencil/convolution、Tensor Core/TMA-friendly kernels** 上具有竞争力甚至胜过 Triton。

- **从分布看 workload-dependence**：  
  - 两个 backend 的散点分布都很分散，说明性能主要由 operator 结构决定。  
  - 同一 backend 上既有低于 **1×** 的失败案例，也有超过 **100×** 的极佳案例。  
  - 因此评价 tile-based DSL 不能只看 GEMM 或 attention，应覆盖 pointwise、reduction、layout、stencil、irregular attention 等多类 operator。  
  - 这正是 TileBench 设计 45 个 operator 的动机。

- **图中最重要的信息压缩**：  
  | 结论 | 证据 |
  |---|---|
  | **两者多数时候快于 PyTorch** | 大多数散点在 **1×** 虚线以上 |
  | **Triton 默认模式整体更强** | Triton 箱体和中位线略高，正文给出 geomean **2.7× vs 2.2×** |
  | **性能差异高度依赖 operator** | 两组散点跨度从低于 **0.1×** 到超过 **100×** |
  | **cuTile 有强项但更不稳定** | 高端 outlier 明显，但低于 1× 的点略多 |
  | **平均值不足以解释全貌** | 分布长尾明显，必须结合 category / case study 分析 |

- **与 Figure 8(b) 的互补关系**：  
  - Figure 8(a) 展示 **speedup over PyTorch** 的整体分布。  
  - Figure 8(b) 展示 **roofline utilization by category**。  
  - 前者回答“相对 PyTorch 快多少”，后者回答“离硬件极限还有多远”。  
  - 两者结合后可以看到：即使 speedup 很高，也不一定代表接近硬件上限，因为 PyTorch baseline 可能较弱。

- **方法学意义**：  
  - 该图支持 TileBench 的核心观点：**仅用 PyTorch speedup 评价 kernel 不够，需要结合 roofline utilization、autotuning、NCU profiling**。  
  - 某个 operator 相对 PyTorch 很快，可能只是 PyTorch reference 低效。  
  - 某个 operator speedup 一般，也可能已经接近 memory bandwidth ceiling。  
  - 因此论文进一步使用 roofline 和 profiling 分析 bottleneck。

- **最终解读**：  
  - 这张图清楚表明，在 **default mode** 下，**Triton 与 cuTile 都是有效的 PyTorch 替代 backend**。  
  - **Triton 的整体分布更优、更稳健**，尤其体现在 median 和 geomean speedup。  
  - **cuTile 的性能更依赖 workload 是否匹配其 tile/TMA/Tensor Core 抽象**。  
  - 两者均存在长尾和失败点，说明 tile-based programming model 的性能仍需依赖 operator-specific tuning、compiler lowering 和 memory behavior 分析。

### fb9b61904f598daa037da72176e1ff88850595260aed8af1924d698112584fca.jpg

![fb9b61904f598daa037da72176e1ff88850595260aed8af1924d698112584fca.jpg](fb9b61904f598daa037da72176e1ff88850595260aed8af1924d698112584fca.jpg)

- **图像内容概述**
  - 该图展示了 TileBench 中五类算子的 **default 配置下 roofline utilization \(R\)**。
  - 横轴为算子类别，纵轴为 **Roofline utilization \(R\)**，范围为 **0 到 1**。
  - 蓝色柱表示 **Triton**，橙色柱表示 **cuTile**。
  - 每个柱上的散点表示该类别中各个具体 operator 的 \(R\) 值，柱高表示类别平均值。
  - 类别包括：
    - **Stencil/Conv**
    - **MatMul/Attn**
    - **Reduce/Norm**
    - **Point-wise**
    - **Data Layout**

- **主要数值解读**

| 类别 | #Ops | Triton 平均 R | cuTile 平均 R | 更优后端 | 主要特征 |
|---|---:|---:|---:|---|---|
| **Stencil/Conv** | 6 | 约 **0.14** | 约 **0.11** | **Triton 略优** | 整体利用率低，个别 cuTile 点较高 |
| **MatMul/Attn** | 8 | 约 **0.14** | 约 **0.14** | **接近持平** | 方差较大，部分 cuTile 算子表现突出 |
| **Reduce/Norm** | 11 | 约 **0.40** | 约 **0.34** | **Triton 更优** | 中等利用率，散点分布很宽 |
| **Point-wise** | 12 | 约 **0.73** | 约 **0.64** | **Triton 明显更优** | 整体最高，接近带宽上限 |
| **Data Layout** | 8 | 约 **0.42** | 约 **0.40** | **Triton 略优** | 平均中等，但内部差异极大 |

- **核心结论**
  - **Point-wise 是 roofline utilization 最高的类别**。
    - Triton 平均约 **0.73**，cuTile 平均约 **0.64**。
    - 多数散点集中在 **0.6–0.9** 区间。
    - 说明简单 elementwise / fused pointwise kernel 更容易接近 B200 的内存带宽或计算上限。
  - **Stencil/Conv 和 MatMul/Attn 的平均 R 最低**。
    - 两类平均值大多低于 **0.15**。
    - 表明这些算子虽然理论上可能具有较高复用或 Tensor Core 潜力，但默认实现中存在明显低层瓶颈。
    - 可能受限于 **operand staging、shared memory layout、TMA 使用不足、Tensor Core pipeline 不充分、边界处理复杂度** 等因素。
  - **Triton 在多数类别平均值上高于 cuTile**。
    - 特别是在 **Point-wise** 和 **Reduce/Norm** 中优势较明显。
    - 这与论文主结论一致：**Triton 对 irregular、streaming、bandwidth-bound operators 更稳健**。
  - **cuTile 并非全面落后，而是高低分化明显**。
    - 在 MatMul/Attn 和 Stencil/Conv 中，cuTile 的部分散点达到较高 R。
    - 说明当 workload 符合 **static tile abstraction、Tensor Core、TMA-friendly memory movement** 时，cuTile 可以表现很好。
    - 但类别平均值被大量低利用率算子拉低。

- **类别内部分布分析**
  - **Stencil/Conv**
    - 两个后端平均值都低。
    - Triton 柱略高于 cuTile。
    - cuTile 有个别点接近 **0.4**，说明某些 stencil/convolution kernel 适合 cuTile 的 tile staging。
    - 但多数点接近 **0–0.15**，表明该类别整体难以充分利用硬件。
  - **MatMul/Attn**
    - Triton 与 cuTile 平均值几乎相同。
    - 散点跨度较大，cuTile 有一个点接近 **0.58**，Triton 最高约 **0.29**。
    - 这说明 cuTile 在部分 Tensor Core / TMA 友好的 GEMM 或 attention kernel 上有明显潜力。
    - 但大量 attention 变体，如 **flash_decode、linear_self_attention、block_sparse_attention**，可能因 irregular access 或低复用导致利用率偏低。
  - **Reduce/Norm**
    - Triton 平均约 **0.40**，cuTile 约 **0.34**。
    - Triton 有点超过 **0.8**，cuTile 也有点达到 **0.7+**。
    - 但也存在接近 **0** 的点，说明 reduction 类算子对实现细节非常敏感。
    - 如 **rmsnorm / layernorm** 可能表现较好，而 **argmax、histogramming、top-k-like reduction** 可能受分支、同步、访存模式影响。
  - **Point-wise**
    - 是全图表现最好的类别。
    - Triton 平均最高，且多个点位于 **0.75–0.95**。
    - cuTile 也较高，但平均低于 Triton。
    - 该类别通常是 **streaming memory-bound**，实现结构简单，Triton 的 pointer-based load/store 与 mask 机制更直接。
  - **Data Layout**
    - Triton 和 cuTile 平均接近，约 **0.42 vs 0.40**。
    - 散点呈现明显两极化：
      - 有些算子接近 **0.8**。
      - 有些算子接近 **0**。
    - 说明 Data Layout 类内部差异很大，简单 copy / transpose 可能较高，而 sort、scatter/gather、复杂 indexing 可能很低。

- **Triton 与 cuTile 对比**
  
| 观察点 | Triton | cuTile |
|---|---|---|
| **整体平均趋势** | 多数类别更高 | 少数类别接近或局部胜出 |
| **Point-wise** | **最强，R 最高** | 较强但低于 Triton |
| **Reduction/Norm** | **平均更优** | 有高点但稳定性较弱 |
| **MatMul/Attn** | 平均不占优 | **部分 Tensor Core/TMA 友好算子更强** |
| **Stencil/Conv** | 平均略高 | 个别点较高，但整体偏低 |
| **Data Layout** | 略优 | 接近 Triton，但方差大 |

- **与论文结论的对应关系**
  - 图中结果支撑论文中的判断：
    - **Triton 更 robust**，尤其适合 **irregular、streaming、bandwidth-bound** 算子。
    - **cuTile 更依赖 workload 是否匹配其 tile abstraction**。
    - **operator structure 的影响大于 backend 平均差异**。
  - 论文 Appendix D.1 中也指出：
    - **Point-wise 平均 R：Triton 0.73，cuTile 0.64**。
    - **Reduction/Norm 和 Data Layout 位于中间区间**。
    - **Stencil/Conv 与 MatMul/Attn 平均不超过 0.15**。
    - 同一类别内部散点跨度往往大于 Triton 与 cuTile 的均值差距。

- **性能瓶颈含义**
  - **低 R 不一定代表算法 FLOPs 少，而是实现没有接近 roofline bound**。
  - 对低利用率类别，可能瓶颈包括：
    - **memory access irregularity**
    - **shared-memory bank conflict**
    - **register pressure**
    - **local memory spill**
    - **Tensor Core / TMA lowering 不理想**
    - **boundary masking 和 gather/scatter 开销**
    - **pipeline overlap 不充分**
  - 对高利用率 Point-wise 类，瓶颈通常更接近 **HBM bandwidth**，代码生成路径也更直接。

- **实践启示**
  - **Point-wise / simple streaming kernel 优先使用 Triton**，通常能快速获得较高 roofline utilization。
  - **GEMM、attention、convolution 中规则且 tile reuse 强的 kernel 可尝试 cuTile**，尤其是能映射到 **TMA + Tensor Core / tcgen05** 的场景。
  - **Reduction、Data Layout、irregular indexing** 需要逐算子诊断，不能仅根据类别或后端平均值判断。
  - **默认配置下仍存在大量 roofline headroom**，说明后续优化重点不只是 tile size autotuning，还包括 compiler lowering、memory staging、shared-memory layout 和 instruction selection。

### 6e1f4d31e8492995f3941738cd3ec827bbdd3969b7866160a9a882e8d2f8f2f7.jpg

![6e1f4d31e8492995f3941738cd3ec827bbdd3969b7866160a9a882e8d2f8f2f7.jpg](6e1f4d31e8492995f3941738cd3ec827bbdd3969b7866160a9a882e8d2f8f2f7.jpg)

- **图像对象**：该图对应论文 Appendix D.2 中的 **Figure 9: Triton and cuTile autotune gain \(R_{\mathrm{autotune}} / R_{\mathrm{default}}\) per operator**，展示 45 个算子在 **autotuning 后相对 default 配置的 roofline utilization 提升倍率**。

- **核心含义**：
  - 纵轴是 **Autotune gain \(R_{\mathrm{autotune}} / R_{\mathrm{default}}\)**。
  - 数值 **大于 1×** 表示 autotuning 后 roofline utilization 提升。
  - 数值 **等于 1×** 表示 autotuning 基本无收益。
  - 数值 **小于 1×** 表示 autotuned 配置反而比 default 配置更差，即出现轻微回退。
  - 图中虚线水平线位于 **1×**，作为 default/autotuned 性能持平基准。
  - 两组箱线图分别对应 **Triton** 与 **cuTile**。
  - 每个散点代表一个 operator 的 autotune gain。
  - 纵轴采用接近对数刻度，突出 **长尾收益**。

- **图中主要统计特征**：

| Backend | 中位数趋势 | 几何平均收益 | 最大收益 | 回退数量 | 高收益分布特征 |
|---|---:|---:|---:|---:|---|
| **Triton** | 约 **1.07×** | **1.20×** | 约 **2.60×**，出现在 **argmax** | **5/45** | 1.5×–2.0× 区间算子更多，约 **7 个** |
| **cuTile** | 约 **1.04×** | **1.15×** | 约 **2.90×**，出现在 **2d_conv** | **8/45** | 极端长尾更明显，少数算子收益很高 |

- **视觉观察**：
  - **Triton** 的箱体略高于 cuTile，说明其典型 autotuning 收益稍强。
  - **cuTile** 的大部分点集中在 1× 附近，说明多数算子 default 配置已经接近其搜索空间内较优配置，或者 autotuning 能调整的参数有限。
  - 两个 backend 都存在明显上方离群点，表明 autotuning 对少数算子非常关键。
  - **cuTile** 有两个接近或超过 **2.5×** 的高收益点，说明某些 tile shape / occupancy 对 cuTile 性能影响极大。
  - **Triton** 的高收益点分布更连续，从 1.5× 到 2.6× 都有较多样本。
  - 两个 backend 都有低于 1× 的点，说明 autotuning 并不保证单调提升。

- **与论文主文结论的对应关系**：
  - 该图支撑 RQ2 的结论：**Autotuning provides measurable gains, but most implementations still leave substantial roofline headroom**。
  - 主文 Figure 3 给出 default-to-autotuned roofline utilization 的整体箭头变化，而该图进一步展示 **每个 operator 的 gain 分布**。
  - 图中显示 autotuning 的收益主要来自少数长尾算子，而不是所有算子均匀提升。
  - 因此，几何平均收益 **Triton 1.20×、cuTile 1.15×** 被少量大收益算子显著拉高。
  - 中位数仅为 **Triton 1.07×、cuTile 1.04×**，说明典型算子的收益相对温和。

- **对 Triton 的分析**：
  - Triton 的可调参数包括 **BLOCK_SIZE、num_warps、num_stages** 等。
  - 图中 Triton 的收益分布显示，这些参数对部分算子仍有显著优化空间。
  - 特别是 reduction、argmax、部分 layout 或 irregular operator，可能对 block size、并行粒度和 occupancy 更敏感。
  - Triton 的最大收益约 **2.60×**，说明 default 配置虽然是人工选择，但在某些算子上仍可能远离最优 tile/parallelism 组合。
  - Triton 的回退点较少，说明其 autotuning 搜索空间相对稳定，出现错误选择或噪声导致劣化的概率较低。

- **对 cuTile 的分析**：
  - cuTile 的可调参数主要包括 **TILE_SIZE** 和 **occupancy**。
  - 大多数 cuTile 点集中在 1× 附近，说明 autotuning 对多数算子提升有限。
  - 但 cuTile 存在更极端的高收益长尾，最大约 **2.90×**，对应 **2d_conv**。
  - 这说明对于 cuTile，某些算子高度依赖合适的 tile shape 与 occupancy；一旦 default 配置不匹配硬件执行路径，性能损失可能很大。
  - cuTile 回退数量更多，为 **8/45**，表明其 autotuning 搜索结果更容易受到测量噪声、occupancy 选择、tile padding 或 compiler lowering 差异影响。

- **为什么 autotuning 收益有限**：
  - 图中大量点接近 **1×**，说明 default 配置并非弱基线，而是人工选择的合理配置。
  - autotuning 只搜索 backend 暴露出的表层参数，无法改变底层实现结构。
  - 对 Triton 而言，autotuning 不能直接改变是否使用 **TMA descriptor**、是否触发更优 **Tensor Core lowering** 或 shared-memory staging 策略。
  - 对 cuTile 而言，autotuning 也不能根本改变 **ct.load / ct.mma / ct.store** 的 compiler lowering、shared memory layout 或 TMEM pipeline 调度。
  - 因此该图体现的是 **tile-parameter tuning 的收益上限**，不是完整 kernel 重写或 compiler lowering 改进后的潜力。

- **性能回退现象解释**：
  - 图中低于 1× 的点说明 autotuning 后 roofline utilization 下降。
  - 可能原因包括：
    - **测量噪声**导致选择了非最优配置。
    - default 配置本身已接近最优。
    - 搜索空间中某些配置增加 register pressure 或 shared-memory pressure。
    - 更大 tile 造成 occupancy 下降。
    - cuTile 的 power-of-two tile constraint 导致 padding 开销增加。
    - 某些配置触发不同 compiler lowering path，产生更差指令序列。

- **与 roofline headroom 的关系**：
  - 该图只显示 **相对提升倍率**，不显示绝对 roofline utilization。
  - 即使某个算子 gain 达到 2×，它的最终 roofline utilization 也可能仍然较低。
  - 论文指出 autotuning 后只有 **8/45 Triton** 和 **4/45 cuTile** 达到至少 **80% roofline utilization**。
  - 因此，autotuning 能改善局部配置，但无法普遍接近 hardware roofline。
  - 主要瓶颈仍可能来自 **instruction selection、memory staging、Tensor Core usage、shared-memory bank conflict、register spilling、TMA/TMEM pipeline scheduling**。

- **方法学意义**：
  - 该图强调 TileBench 的 autotuning 评估不是简单比较 default 性能，而是进一步检验 backend 在合理调参后的潜力。
  - 通过展示每个 operator 的 gain，论文避免只用平均值掩盖长尾行为。
  - 图中分布说明，**operator-specific tuning** 很重要，但其收益高度 workload-dependent。
  - 对 benchmark 设计而言，这证明单个算子或单个配置不足以代表一个 tile-based programming model 的整体性能。

- **对开发者的实践启示**：
  - 对大多数算子，若已有合理 default 配置，autotuning 可能只带来 **小幅收益**。
  - 对性能敏感且 shape 多变的算子，仍应进行 autotuning，因为少数场景可能出现 **2× 以上提升**。
  - 如果 autotuning 后仍远低于 roofline，应优先检查：
    - **compiler lowering path**
    - **Tensor Core / TMA 是否被有效使用**
    - **shared memory bank conflict**
    - **register pressure**
    - **occupancy**
    - **global memory coalescing**
    - **predicate / mask handling**
  - 对 Triton，重点关注 **BLOCK_SIZE、num_warps、num_stages** 与 pointer load / tensor descriptor 的选择。
  - 对 cuTile，重点关注 **TILE_SIZE、occupancy、ct.load/ct.gather 使用方式、tile regularity**。

- **总体结论**：
  - 该图清楚表明：**autotuning 对 Triton 和 cuTile 都有效，但收益主要集中在少数算子上**。
  - **Triton 的典型收益略高且更稳定**。
  - **cuTile 的中位收益更低，但极端收益更高，同时回退更多**。
  - 该结果支持论文核心观点：**tile 参数搜索只能解决一部分性能问题，真正限制性能的往往是 backend compiler lowering、memory behavior 与硬件路径映射能力**。

### eecac1f3f846e36c1c4ebbc4ff6729c5eda72ef2563106488aac0344e4ef83b5.jpg

![eecac1f3f846e36c1c4ebbc4ff6729c5eda72ef2563106488aac0344e4ef83b5.jpg](eecac1f3f846e36c1c4ebbc4ff6729c5eda72ef2563106488aac0344e4ef83b5.jpg)

- **图像类型与位置**
  - 该图对应论文附录中的 **Figure 10: Top 20 bank-conflict cases based on conflict score**。
  - 主题是比较 **Triton** 与 **cuTile** 在 TileBench 中的 **shared-memory bank conflict** 严重程度。
  - 图中每一行是一个 **operator / backend / dtype** 组合。
  - 横轴为 **Bank conflicts / shared wavefront (%)**，采用 **log scale**。
  - 颜色编码：
    - **蓝色：Triton**
    - **橙色：cuTile**

- **核心指标含义**
  - 图中的 conflict score 来自 Nsight Compute 指标，论文定义为：
  
  | 指标 | 含义 |
  |---|---|
  | **Bank conflicts** | shared-memory 访问中的 bank conflict 事件数 |
  | **Shared wavefronts** | 服务 shared-memory 访问所需的 L1TEX wavefront 数 |
  | **Conflict score** | `100 × bank conflicts / shared wavefronts` |
  
  - 该指标衡量的是 **单位 shared-memory wavefront 上的冲突密度**。
  - 它用于定位潜在的 shared-memory layout 或 staging 问题，但 **不是精确的性能损失估计**。

- **Top-20 案例分布**
  
  | Backend | Top-20 数量 | 占比 | 观察 |
  |---|---:|---:|---|
  | **Triton** | **14** | **70%** | 高 conflict case 更多 |
  | **cuTile** | **6** | **30%** | 也存在严重冲突，但数量较少 |
  
  - 图中显示 **Triton 在 Top-20 中占多数**，说明 Triton 生成代码在若干 tile-heavy kernel 上更容易暴露 shared-memory bank conflict。
  - 但 cuTile 也有多个高冲突案例，因此不能简单认为 bank conflict 是某一个 backend 的单边问题。

- **排名靠前的典型案例**
  
  | 排名区间 | Operator / Backend / dtype | 颜色 | 主要含义 |
  |---|---|---|---|
  | 最高 | **streamk_matmul / Triton / fp32** | 蓝色 | 最严重 bank-conflict case |
  | 前列 | **batched_matmul / cuTile / fp32** | 橙色 | cuTile 中最突出的冲突案例之一 |
  | 前列 | **matmul_int8 / cuTile / int8** | 橙色 | 低比特 GEMM 中 unpack / staging 可能导致冲突 |
  | 前列 | **2d_conv / Triton / fp32** | 蓝色 | im2col-like operand materialization 相关 |
  | 前列 | **flash_attention / Triton / fp16** | 蓝色 | attention tile staging 存在冲突信号 |
  | 中段 | **batched_matmul / Triton / fp16/bf16** | 蓝色 | batched GEMM 多 dtype 均出现 |
  | 中段 | **batched_matmul / cuTile / fp16/bf16** | 橙色 | cuTile 同类 kernel 也出现冲突 |
  | 后段 | **l2_norm / Triton / fp16/bf16/fp32** | 蓝色 | reduction 类 kernel 出现多 dtype 冲突 |
  | 后段 | **rmsnorm / Triton / fp16/bf16** | 蓝色 | normalization kernel 中也有 shared-memory 冲突 |

- **dtype 层面的观察**
  
  | dtype | 图中表现 | 解释 |
  |---|---|---|
  | **fp32** | 多个高排名案例 | fp32 数据宽度更大，shared-memory layout 与 staging 更容易放大冲突 |
  | **fp16 / bf16** | 在 matmul、norm、attention 中频繁出现 | Tensor Core 或 tile staging 路径中仍可能产生 bank conflict |
  | **int8** | 主要出现在 **matmul_int8 / cuTile / int8** | packed / unpacked low-bit 数据路径可能引入额外 shared-memory 重排压力 |

- **Operator 类型分布**
  
  | Operator 类型 | 代表案例 | 现象 |
  |---|---|---|
  | **Matrix Multiplication / Attention** | streamk_matmul, batched_matmul, flash_attention, block_sparse_attention, matmul_int8 | Top-20 中最集中，说明 tile staging 和 MMA operand layout 是主要风险源 |
  | **Stencil / Convolution** | 2d_conv | irregular operand materialization 可能引入 shared-memory 冲突 |
  | **Reduction / Normalization** | l2_norm, rmsnorm, softmax | reduction staging 或 shared-memory partial accumulation 也会触发冲突 |
  | **Sorting / Layout** | bitonic_sort | 数据重排型 kernel 中存在访问模式不规则问题 |

- **与论文诊断结论的对应关系**
  - 论文指出，Top-10 的 NCU Source view 检查结果大致分为：
  
  | 冲突来源 | 数量 | 含义 |
  |---|---:|---|
  | **shared-memory stores** | **4** | 写入 shared-memory 时 bank layout 不佳 |
  | **shared-memory loads / global-to-shared copies** | **4** | load staging 或 LDGSTS 路径存在冲突 |
  | **MMA source lines** | **2** | 表面看是 MMA 行，实际多为 operand staging 问题 |
  
  - 关键结论是：**bank conflict 通常不是 Tensor Core 算术本身的问题，而是 operand staging、shared-memory layout、load/store tiling 的问题**。

- **最重要案例：streamk_matmul / Triton / fp32**
  - 图中最高冲突案例是 **streamk_matmul / Triton / fp32**。
  - 论文说明该案例对应到源代码中的 **B-tile load**：
    - `b = tl.load(B_ptrs, ...)`
    - 以及后续 **LDGSTS.E global-to-shared copies**
  - 这说明 Triton 在该 kernel 中选择的 staging pattern 可能导致 shared-memory bank 访问分布不均。
  - 优化方向应集中在：
    - **B tile layout**
    - **shared-memory swizzle**
    - **padding**
    - **load vectorization**
    - **K tile / N tile shape 调整**

- **Triton 的表现解读**
  - Triton 在 Top-20 中占 **14/20**。
  - 这不意味着 Triton 整体性能一定更差，因为论文主结果显示 Triton 在整体速度上仍强于 cuTile。
  - 更合理的解释是：
    - Triton 的 pointer-level model 更灵活；
    - 但在某些 tile-heavy kernel 中，compiler-selected staging pattern 可能不够理想；
    - 特别是 matmul、attention、reduction 中，shared-memory layout 可能出现高冲突。

- **cuTile 的表现解读**
  - cuTile 在 Top-20 中占 **6/20**。
  - cuTile 的 tile abstraction 更贴近 Blackwell native path，例如 **TMA**、**tcgen05**、**ct.mma**。
  - 但图中仍有明显高冲突案例，例如：
    - **batched_matmul / cuTile / fp32**
    - **matmul_int8 / cuTile / int8**
    - **softmax / cuTile / fp32**
  - 这说明 cuTile 虽然更容易利用硬件 tile path，但在 **intermediate tile layout**、**unpacked data staging**、**shared-memory write/read pattern** 上仍可能产生冲突。

- **与 matmul_int8 case study 的联系**
  - 论文在 Appendix E.1 中分析 **matmul_int8**：
    - cuTile 使用 **UTMALDG.2D、UTCIMMA、TMEM accumulator**，整体比 Triton 快。
    - 但 cuTile 仍报告严重 shared-memory layout inefficiency。
    - 包括 **store conflicts**、**conflicted shared-store wavefronts**、**uncoalesced MMA reads**。
  - 因此图中 **matmul_int8 / cuTile / int8** 的高冲突并不矛盾：
    - cuTile 总体更快；
    - 但其 intermediate unpacked-B scratch tile layout 仍有优化空间。

- **性能诊断价值**
  - 该图的价值不是直接判断哪个 backend 更快，而是帮助定位：
    - **shared-memory tiling 是否合理**
    - **operand staging 是否过重**
    - **MMA 输入布局是否需要 swizzle / padding**
    - **global-to-shared copy 是否产生 bank conflict**
  - 对性能工程师而言，Top-20 中的 case 是优先检查对象。

- **需要避免的误读**
  - **高 bank conflict 不等于一定低性能**：
    - 某些 kernel 即使 conflict 高，仍可能因 Tensor Core 利用率高而整体较快。
  - **低 bank conflict 不等于性能最优**：
    - 性能还受 occupancy、register pressure、TMA overlap、instruction selection、memory bandwidth 等影响。
  - **该指标只覆盖 LSU shared-memory path**：
    - 不覆盖所有 pipeline，也不能替代完整 NCU 分析。

- **总体结论**
  - 图 10 表明：**严重 shared-memory bank conflict 集中在少数 tile-heavy kernels 中**。
  - **Triton Top-20 数量更多**，主要暴露在 streamk_matmul、batched_matmul、attention、norm 等场景。
  - **cuTile 也存在显著冲突**，尤其是 batched_matmul、matmul_int8、softmax 等。
  - 最核心的优化方向不是简单更换 backend，而是检查：
    - **load/store tiling**
    - **shared-memory layout**
    - **operand staging**
    - **MMA input swizzling**
    - **padding strategy**
    - **compiler lowering path**

### f40e30cb49a6e88710c31c7a1255f22ad48c646bcd4612828aa878bf469b88af.jpg

![f40e30cb49a6e88710c31c7a1255f22ad48c646bcd4612828aa878bf469b88af.jpg](f40e30cb49a6e88710c31c7a1255f22ad48c646bcd4612828aa878bf469b88af.jpg)

- **图像对象**：该图为论文 Appendix D.3.3 中的 **Figure 11: Automated NCU bank-conflict diagnosis by backend**，用于比较 **Triton** 与 **cuTile** 在 Nsight Compute（NCU）自动化分析中的 **shared-memory bank conflict** 诊断分布。

- **图表类型**：水平堆叠条形图。
  - y 轴：backend，包含 **triton** 与 **cutile**。
  - x 轴：**reports**，表示 NCU profiling report 数量。
  - 每个条形被划分为 5 类互斥诊断结果：
    - **Severe/Moderate**
    - **Mild**
    - **Likely**
    - **Confounded**
    - **No direct conflict**

- **近似读数如下**：

| Backend | Severe/Moderate | Mild | Likely | Confounded | No direct conflict | 总体趋势 |
|---|---:|---:|---:|---:|---:|---|
| **triton** | 约 13 | 约 23 | 约 5 | 约 3 | 约 59 | **多数无直接 bank conflict，但严重/中等冲突比例更高** |
| **cutile** | 约 6 | 约 35 | 约 17 | 约 5 | 约 40 | **轻度或疑似冲突更多，无直接冲突比例更低** |

- **核心观察**：
  - **Triton 的 “No direct conflict” 占比最高**，大约接近 60 个 report，说明在多数 Triton profile 中，NCU 没有发现直接的 LSU shared-memory bank conflict 证据。
  - **cuTile 的 “No direct conflict” 明显少于 Triton**，约 40 个 report，说明 cuTile 中更多 case 被归入某种冲突相关类别。
  - **Triton 的 Severe/Moderate 数量高于 cuTile**，约为 cuTile 的 2 倍左右。这与论文中 Figure 10 的结论一致：top-20 bank-conflict cases 中 **14 个是 Triton，6 个是 cuTile**。
  - **cuTile 的 Mild 与 Likely 合计更多**，说明 cuTile 不一定出现最多的严重冲突，但更频繁地产生弱冲突或疑似冲突信号。
  - **Confounded 类别占比较小**，两者都只有少量 report，说明大多数诊断没有被 branch divergence 等因素严重混淆。

- **诊断类别含义可理解为**：

| 类别 | 含义 | 性能分析意义 |
|---|---|---|
| **Severe/Moderate** | NCU 指标显示较强的 LSU shared-memory bank conflict 证据 | **需要优先检查 shared-memory layout、tile staging、operand swizzling** |
| **Mild** | 有一定冲突迹象，但强度较弱 | 可能影响性能，但未必是主瓶颈 |
| **Likely** | 间接或来源级证据暗示 bank conflict | 需要结合 Source View / SASS 进一步确认 |
| **Confounded** | 冲突信号被 branch divergence 等因素干扰 | **不能直接归因于 bank conflict** |
| **No direct conflict** | 未发现直接 LSU bank conflict 证据 | 性能瓶颈可能来自 memory latency、register pressure、TMA wait、occupancy 等其他因素 |

- **与论文上下文的关系**：
  - 论文在 Appendix D.3.1 中说明，作者使用 NCU profile 了 **204 个 kernels**，并根据 normalized LSU shared-memory conflict signal 进行自动化分类。
  - Figure 11 的作用不是证明某个 backend 绝对更好，而是说明：**bank conflict 在 Triton 和 cuTile 中都会出现，但表现形式不同**。
  - Triton 的严重案例更多，常与 **compiler-selected staging pattern**、global-to-shared copy、MMA operand staging 有关。
  - cuTile 的轻度/疑似冲突更多，可能与其 tile abstraction、ct.load / ct.store / ct.mma 的 shared-memory staging 方式有关。

- **关键结论**：
  - **不存在一个 backend 在 bank conflict 上全面占优**。
  - **Triton 更容易出现少数严重 bank-conflict case**。
  - **cuTile 更频繁出现轻度或疑似 conflict signal**。
  - **大部分 report，尤其是 Triton，仍属于 No direct conflict**，说明 bank conflict 不是 TileBench 中所有性能差距的主因。
  - 该图支持论文的总体判断：性能差异更多来自 **memory access expression、compiler lowering、operand staging、TMA/tcgen05 使用、register pressure** 等综合因素，而不是单一的 bank conflict 指标。

- **对优化的启示**：
  - 对 **Severe/Moderate** case，应优先检查：
    - **shared-memory tile layout**
    - **load/store coalescing**
    - **MMA operand staging**
    - **swizzling / padding**
    - **LDS / STS / LDGSTS 指令分布**
  - 对 **Likely / Mild** case，应结合：
    - **NCU Source View**
    - **SASS inspection**
    - **branch efficiency**
    - **stall_long_scoreboard**
    - **register spilling**
  - 对 **No direct conflict** case，不应继续把 bank conflict 当作主瓶颈，而应转向分析：
    - **TMA latency hiding**
    - **Tensor Core utilization**
    - **occupancy**
    - **instruction mix**
    - **register pressure**
    - **HBM bandwidth utilization**

### e055de22c98b3a5a7cd8160a32a8750a8815cc4733f2537d3189b67514330c8a.jpg

![e055de22c98b3a5a7cd8160a32a8750a8815cc4733f2537d3189b67514330c8a.jpg](e055de22c98b3a5a7cd8160a32a8750a8815cc4733f2537d3189b67514330c8a.jpg)

- **图像类型**：该图是一个 **heatmap / 热力图**，展示 TileBench 中 **bank conflict score** 较高的前 25 个 **operator/backend** 组合在不同 **dtype** 下的冲突强度。

- **图像主题**：标题为 **“Top 25 bank-conflict operator/backend rows”**，用于分析 Triton 与 cuTile 在 NVIDIA B200 上执行不同算子时的 **shared-memory bank conflict** 分布。

- **横轴含义**：

| 横轴 dtype | 含义 |
|---|---|
| **bf16** | bfloat16 |
| **fp16** | half precision floating point |
| **fp32** | single precision floating point |
| **int32** | 32-bit integer |
| **int8** | 8-bit integer |

- **纵轴含义**：

| 纵轴项目 | 含义 |
|---|---|
| **operator / backend** | 某个算子在某个后端上的实现 |
| **triton** | Triton backend |
| **cutile** | cuTile backend |

- **颜色含义**：

| 颜色 | conflict score 含义 |
|---|---|
| **浅色 / 接近白色** | bank conflict 很低或几乎没有 |
| **橙色** | 中等冲突 |
| **红色 / 紫色** | 较高冲突 |
| **接近黑色** | 极高冲突，约接近 **60%–70%** |

- **最显著现象**：
  - **bank conflict 高值高度集中在少数 tile-heavy 算子中**，并不是所有 operator/backend/dtype 都普遍存在高冲突。
  - **fp32 列出现最强冲突最频繁**，尤其在 matmul、conv、softmax、norm、transpose 等涉及 shared-memory staging 或 tile reuse 的算子中。
  - **Triton 与 cuTile 都会出现严重 bank conflict**，图中没有显示某个 backend 在所有情况下都绝对更优。
  - **部分极端冲突集中在 GEMM / convolution / attention 类算子**，说明 bank conflict 与 **operand staging、shared-memory layout、MMA 输入组织** 强相关。

- **冲突最严重的条目**：

| 排名趋势 | operator / backend | dtype | 观察 |
|---|---|---|---|
| 最高 | **streamk_matmul / triton** | **fp32** | 最深色，conflict score 接近图例上限，属于最严重 bank conflict |
| 最高 | **batched_matmul / cutile** | **fp32** | 与 streamk_matmul/triton 类似，fp32 冲突极高 |
| 最高 | **matmul_int8 / cutile** | **int8** | int8 下出现极深色，说明低比特 GEMM 的 shared-memory staging 存在严重冲突 |
| 很高 | **2d_conv / triton** | **fp32** | fp32 下冲突极高，和论文中 2d_conv case study 的 operand materialization 问题一致 |
| 较高 | **flash_attention / triton** | **fp16** | fp16 下有明显高冲突，可能来自 attention tile staging 或 shared-memory layout |

- **按 dtype 分析**：

| dtype | 主要现象 |
|---|---|
| **bf16** | 高冲突较少，主要集中在 **streamk_matmul / triton**、**batched_matmul / triton**、**l2_norm / triton**、**rmsnorm / triton** 等 |
| **fp16** | 冲突分布较广，**flash_attention / triton** 显著偏高，多个 norm、attention、transpose 算子有中等冲突 |
| **fp32** | **最容易出现高冲突**，streamk_matmul、batched_matmul、2d_conv、softmax、norm、transpose 等都有明显信号 |
| **int32** | 只有少量算子支持，整体冲突较低，图中主要是 **radix_sort / cutile** 有轻微冲突 |
| **int8** | 主要集中在 **matmul_int8 / cutile** 极高冲突，以及部分 transpose 相关算子有轻微冲突 |

- **按 backend 分析**：

| backend | 观察 |
|---|---|
| **Triton** | 高冲突案例更多地出现在 **streamk_matmul、2d_conv、flash_attention、l2_norm、rmsnorm、transpose** 等 |
| **cuTile** | 高冲突集中在 **batched_matmul、matmul_int8、softmax、block_sparse_attention、2d_conv、transpose** 等 |
| **总体判断** | 图中支持论文结论：**severe bank conflict occurs in both backends**，不是 Triton 或 cuTile 单方面的问题 |

- **与论文内容的对应关系**：
  - 该图对应 Appendix D.3.3 中的 **operator–dtype heatmap**。
  - 论文指出该热力图用于说明：**high conflict scores are concentrated in a small number of tile-heavy kernels rather than spread uniformly across the benchmark**。
  - 图中确实可以看到，大部分单元格接近白色，只有少数 operator/backend/dtype 组合呈现橙色、红色、紫色或黑色。

- **核心技术解释**：
  - **bank conflict** 主要来自 shared memory 多 bank 并行访问时，多个线程同时访问同一个 bank。
  - 在 GPU kernel 中，bank conflict 通常会降低 shared-memory throughput，引发 stall，尤其影响：
    - **shared-memory load / store**
    - **global-to-shared copy**
    - **MMA operand staging**
    - **tile transpose / swizzle**
    - **low-bit unpacking 后的中间 tile 存储**

- **典型算子解读**：

| operator/backend | 图中表现 | 可能原因 |
|---|---|---|
| **streamk_matmul / triton** | fp32 极高冲突 | B-tile load 或 LDGSTS global-to-shared staging 产生冲突 |
| **batched_matmul / cutile** | fp32 极高冲突 | cuTile tile staging 或 MMA operand layout 存在 shared-memory bank conflict |
| **matmul_int8 / cutile** | int8 极高冲突 | 2-bit / int8 unpack 后的 scratch tile layout 不佳，MMA read 不够 coalesced |
| **2d_conv / triton** | fp32 极高冲突 | virtual-im2col operand tile 的 shared-memory materialization 可能产生冲突 |
| **flash_attention / triton** | fp16 高冲突 | attention 中 Q/K/V tile staging 与 shared-memory layout 有冲突风险 |
| **matrix_transpose / triton/cutile** | 多 dtype 中等冲突 | transpose 天然容易触发 bank conflict，除非使用 padding/swizzling |

- **重要结论**：
  - **fp32 是 bank conflict 的高风险 dtype**。图中最深色冲突大多出现在 fp32 列。
  - **GEMM-like 和 attention-like 算子更容易触发严重冲突**，因为它们依赖 shared-memory staging 和 MMA operand preparation。
  - **Data layout 类算子如 matrix_transpose 也容易产生冲突**，因为访问模式本身容易导致 bank 对齐冲突。
  - **cuTile 的 Blackwell-native 路径并不自动避免 bank conflict**。例如 matmul_int8/cutile 在 int8 下冲突极高。
  - **Triton 的 pointer-based flexibility 也不保证 shared-memory layout 最优**。例如 streamk_matmul/triton 和 2d_conv/triton 的 fp32 冲突非常突出。

- **从优化角度看，该图给出的启示**：
  - 对深色单元格，应优先检查 **shared-memory layout**。
  - 对 GEMM / attention，应检查 **MMA operand staging** 是否引入 bank conflict。
  - 对 transpose / convolution，应考虑 **padding、swizzling、tile shape 调整**。
  - 对 low-bit GEMM，应特别检查 **unpack 后的中间 tile** 是否造成 shared-memory store conflict 或 uncoalesced MMA read。
  - 单纯 autotuning TILE_SIZE / BLOCK_SIZE 可能不足，需要改进 **compiler lowering、shared-memory swizzle、operand layout**。

- **综合评价**：
  - 该图不是在比较整体性能，而是在定位 **shared-memory bank conflict 的热点分布**。
  - 它支持论文 RQ3 的核心观点：Triton 与 cuTile 的性能差异，很多时候来自 **memory access expression、shared-memory staging、compiler lowering 与 hardware-native path 使用方式**。
  - 图中最值得关注的异常点是 **streamk_matmul/triton/fp32**、**batched_matmul/cutile/fp32**、**matmul_int8/cutile/int8** 和 **2d_conv/triton/fp32**，这些组合很可能是后续优化的优先对象。

### 98c4463baedfff6877d1206ce66cafa179292649e329275fea02456eac9f6331.jpg

![98c4463baedfff6877d1206ce66cafa179292649e329275fea02456eac9f6331.jpg](98c4463baedfff6877d1206ce66cafa179292649e329275fea02456eac9f6331.jpg)

- **图像对象**：该图为论文 TileBench 附录中的 **Figure 13: Matched Triton–cuTile conflict-score comparison**，用于比较同一 **operator–dtype** 组合下，**Triton** 与 **cuTile** 的共享内存 bank conflict 严重程度。

- **图像核心含义**：
  - 横轴为 **Triton conflict score (%)**。
  - 纵轴为 **cuTile conflict score (%)**。
  - 每个散点表示一个匹配的 **operator–dtype** 测试案例。
  - 虚线为 **y = x** 对角线：
    - 点在虚线上方：**cuTile conflict score 更高**，即 cuTile 的 bank conflict 更严重。
    - 点在虚线下方：**Triton conflict score 更高**，即 Triton 的 bank conflict 更严重。
    - 点接近虚线：两者 conflict 行为相近。

- **坐标尺度分析**：
  - 横纵轴均为 **log scale**。
  - conflict score 覆盖范围很宽：
    - 约从 **10⁻³%** 到 **10²%**。
    - 表明不同 kernel 的 bank conflict 严重程度差异达到多个数量级。
  - 大多数点集中在：
    - Triton conflict score：约 **10⁰%–10¹%**。
    - cuTile conflict score：约 **10⁰%–10¹%**。
  - 少数异常点达到 **几十到接近 100%**，说明特定 operator–dtype 会触发严重 shared-memory conflict。

- **图例与 dtype 分布**：

| 颜色 | dtype | 图中现象 |
|---|---|---|
| 蓝色 | **fp16** | 数量较多，分布较散，既有 Triton 更高也有 cuTile 更高 |
| 橙色 | **fp32** | 覆盖范围最广，存在高 conflict 离群点 |
| 绿色 | **bf16** | 点较少，多集中在中低 conflict 区域 |
| 红色 | **int8** | 出现非常高的 cuTile conflict score 离群点 |
| 紫色 | **int32** | 点较少，主要位于中等 conflict 区域 |
| 棕色 | **fp8_e4m3fn** | 数量少，靠近中低 conflict 区 |
| 粉色 | **fp8_e5m2** | 图中可见度较低，样本较少或与其他点重叠 |

- **主要观察结果**：

| 观察 | 解释 |
|---|---|
| **点分布在对角线两侧** | 说明 **Triton 与 cuTile 没有一方在 bank conflict 上绝对占优** |
| **多数点接近 1%–10% 区间** | 大部分 kernel 存在一定 shared-memory conflict，但并非极端严重 |
| **cuTile 有高纵轴离群点** | 某些 cuTile kernel 出现较严重 bank conflict，可能来自 shared-memory staging、tile layout 或 ct.mma operand staging |
| **Triton 也有高横轴点** | Triton 在部分 tile-heavy kernel 中也会产生明显 bank conflict，尤其可能与 compiler-selected staging pattern 相关 |
| **fp32 点分布较宽** | 与论文附录描述一致：部分 **FP32 cases** 的 bank conflict 更严重 |
| **低 conflict 区域存在重叠点** | 有些 operator–dtype 在两个 backend 上 bank conflict 都很低，说明 conflict 并非普遍瓶颈 |

- **与论文上下文的对应关系**：
  - 论文在 Appendix D.3.3 中指出：
    - **Backend comparison** 表明 matched operator–dtype pairs 在对角线两侧都有点。
    - 因此，**neither backend universally dominates in bank-conflict behavior**。
  - 该图正是这一结论的证据：
    - cuTile 并非总是更差。
    - Triton 也并非总是更优。
    - bank conflict 更依赖于 **具体 operator、dtype、tile layout、shared-memory staging 与 compiler lowering path**。

- **对角线区域解读**：
  - 接近虚线的点表示：
    - Triton 与 cuTile 的 conflict score 接近。
    - 这类 case 中，性能差异若存在，可能更多来自：
      - Tensor Core / TMA 使用差异；
      - register pressure；
      - instruction selection；
      - occupancy；
      - memory coalescing；
      - asynchronous pipeline scheduling。
  - 远离虚线的点表示：
    - bank conflict 是更值得优先排查的 backend 差异来源。

- **高 conflict 离群点分析**：

| 区域 | 含义 | 可能原因 |
|---|---|---|
| 左上方 | **cuTile conflict 显著高于 Triton** | cuTile tile staging、ct.load/ct.store layout、ct.mma operand staging 产生 bank conflict |
| 右下方 | **Triton conflict 显著高于 cuTile** | Triton lowering 到 shared-memory copy / LDGSTS / LDS 路径时布局不佳 |
| 右上方 | **两者 conflict 都高** | operator 本身 tile-heavy，shared memory 访问模式复杂 |
| 左下方 | **两者 conflict 都低** | kernel 可能是 streaming / pointwise / 较少 shared-memory staging |

- **从图中可见的典型模式**：
  - **fp32** 样本在高 conflict 区较明显，说明 FP32 路径可能更容易因 tile 大小、shared-memory footprint 或 staging 方式导致冲突。
  - **int8** 出现高 cuTile conflict 点，与论文 E.1 中 **matmul_int8** case study 呼应：
    - cuTile 使用 Blackwell-native path，如 **UTMALDG.2D、UTCIMMA、TMEM accumulator**。
    - 但 unpacked-B scratch tile 仍存在 shared-memory layout inefficiency。
    - 论文提到 cuTile 在该 case 中存在 **4.4-way store conflicts**、大量 conflicted shared-store wavefronts 与 uncoalesced MMA reads。
  - **Triton 高 conflict 点**也存在，说明即使 Triton 在不规则访问上更灵活，其 compiler lowering 仍可能选择不理想的 shared-memory staging pattern。

- **该图支持的论文结论**：
  - **bank conflict 不是 Triton 或 cuTile 的单边问题**。
  - **严重 conflict 集中在少数 tile-heavy kernels**，而不是均匀分布在所有 benchmark。
  - conflict score 适合用作 **诊断信号**，但不能单独作为性能预测指标。
  - 若某 case 中点远离对角线，应优先检查：
    - shared-memory load/store；
    - global-to-shared copy；
    - MMA operand staging；
    - tile swizzling；
    - padding；
    - compiler-generated SASS。

- **方法学评价**：
  - 图中的 conflict score 来自 Nsight Compute 指标，论文定义为：

| 指标 | 含义 |
|---|---|
| **B_LSU** | LSU shared-memory bank-conflict events |
| **W_LSU** | LSU shared-memory wavefronts |
| **C_LSU = 100 × B_LSU / W_LSU** | normalized conflict severity |

  - 这种归一化方式可以消除 kernel 规模影响，更适合跨 operator 比较。
  - 但论文也强调：
    - **C_LSU 不是精确的 source-level conflict 数量**。
    - 它也不是直接的 speedup 估计。
    - 它不覆盖非 LSU pipeline 上的 conflict 或其他瓶颈。

- **对性能诊断的启示**：

| 诊断场景 | 建议 |
|---|---|
| 点在虚线上方很多 | 优先检查 cuTile 的 **ct.load / ct.store / ct.mma** staging 和 tile layout |
| 点在虚线下方很多 | 优先检查 Triton 的 **tl.load lowering、LDGSTS、shared-memory copy** |
| fp32 case conflict 高 | 检查 FP32 tile size、shared-memory footprint、register pressure |
| int8/fp8 case conflict 高 | 检查 packed / unpacked operand staging、swizzling、padding |
| 点接近对角线但性能差距大 | bank conflict 可能不是主因，应检查 TMA、Tensor Core、occupancy、scheduler stall |

- **与 TileBench 总体结论的关系**：
  - 主文认为：
    - **cuTile** 在 Tensor Core / TMA-friendly kernel 上有优势。
    - **Triton** 在 irregular、streaming、bandwidth-bound operator 上更稳健。
  - 该图补充说明：
    - 即使 cuTile 更贴近 Blackwell-native tile path，也可能因为 shared-memory layout 出现 conflict。
    - 即使 Triton 更灵活，也可能在 compiler-selected staging 中产生 conflict。
    - 因此性能差距不能只看 DSL 抽象层，还要结合 **NCU + PTX/SASS** 做底层诊断。

- **一句话总结**：
  - 该图表明 **Triton 与 cuTile 在 bank conflict 上没有绝对优劣，严重 conflict 主要由具体 operator–dtype、tile staging 与 compiler lowering 决定；因此 conflict score 应作为定位 shared-memory layout 问题的诊断工具，而不是单独的性能判据**。

### 658915d4c4dc89964f3708c8d937a87d133acabceb646e3365b9e7470215528e.jpg

![658915d4c4dc89964f3708c8d937a87d133acabceb646e3365b9e7470215528e.jpg](658915d4c4dc89964f3708c8d937a87d133acabceb646e3365b9e7470215528e.jpg)

- **图像核心信息**

  | 项目 | 内容 |
  |---|---|
  | 图名 | **Branch-divergence confounder** |
  | 对应论文位置 | Appendix D.3.3，Figure 14 |
  | 分析目标 | 检查 **branch divergence** 是否会干扰 **shared-memory bank-conflict** 归因 |
  | 横轴 | **Branch efficiency (%)** |
  | 纵轴 | **Conflict score (%)**，采用 **log scale** |
  | 后端颜色 | **orange = cuTile**，**blue = Triton** |
  | 点形状 | 表示 bank-conflict 诊断类别 |
  | 关键阈值线 | 约 **95%** 与 **99%** branch efficiency 分界线 |

- **坐标轴含义**

  | 轴 | 含义 | 解读方式 |
  |---|---|---|
  | **Branch efficiency (%)** | 有效分支执行效率，越高说明 warp 内控制流越一致 | 接近 **100%** 表示几乎无 branch divergence |
  | **Conflict score (%)** | 归一化 LSU shared-memory bank-conflict 信号 | 越高表示 bank-conflict 证据越强 |
  | **log-scale y-axis** | conflict score 跨越多个数量级 | 便于同时观察极低冲突与严重冲突样本 |

- **图例说明**

  | 编码 | 含义 |
  |---|---|
  | **orange** | cuTile backend |
  | **blue** | Triton backend |
  | **circle** | **No direct conflict** |
  | **x** | **Likely** conflict |
  | **plus** | **Mild** conflict |
  | **square** | **Severe/Moderate** conflict |
  | **diamond** | **Confounded**，即可能被 branch divergence 混淆 |

- **主要视觉结论**

  | 观察 | 说明 |
  |---|---|
  | 大多数点集中在 **99%–100% branch efficiency** | 说明绝大多数 profiling case 几乎没有明显 branch divergence |
  | 高 conflict score 点主要也位于 **接近 100% branch efficiency** 区域 | 表明强 bank-conflict 信号通常不是由 branch divergence 人为造成 |
  | 少量点落在 **95% 以下**，甚至约 **80%–92%** | 这些样本被标记为 **Confounded**，归因可信度较低 |
  | Triton 与 cuTile 都存在高 conflict score 样本 | bank-conflict 并非某一个 backend 独有问题 |
  | 低 conflict score 点同样集中在高 branch efficiency 区域 | 高 branch efficiency 本身不保证无 bank conflict，只是排除了 divergence 混淆 |

- **阈值线解读**

  | 阈值 | 视觉位置 | 含义 |
  |---|---|---|
  | **约 95% dotted line** | 图中偏右的灰色虚线 | 低于该值时，branch divergence 可能显著影响 bank-conflict 归因 |
  | **约 99% dashed line** | 靠近右侧的黑色虚线 | 高于该值时，控制流基本一致，bank-conflict 诊断更可信 |

- **样本分布分析**

  | 区域 | 样本特征 | 解释 |
  |---|---|---|
  | **Branch efficiency ≈ 99%–100%，Conflict score 高** | 大量 orange/blue 点，包含 plus、square、x | 这是最有价值区域，说明 **bank-conflict 信号较干净** |
  | **Branch efficiency ≈ 99%–100%，Conflict score 极低** | 若干 circle 点，y 约 10^-3% | 说明部分 kernel 没有直接 LSU conflict 证据 |
  | **Branch efficiency < 95%** | 少量 diamond 点 | 被视为 **divergence-confounded attribution** |
  | **Branch efficiency 80%–92%** | 极少数异常点 | 控制流发散明显，不能简单把性能问题归咎于 bank conflict |

- **高冲突点分析**

  | 特征 | 说明 |
  |---|---|
  | **Conflict score 可达到 10%–100% 级别** | 表示某些 kernel 的 shared-memory bank-conflict 非常严重 |
  | 高冲突点多数位于 **branch efficiency 接近 100%** | 支持论文观点：这些冲突更可能源自 **shared-memory layout / staging pattern / MMA operand staging**，而不是分支发散 |
  | Triton 与 cuTile 均有高冲突点 | 说明 compiler lowering 和 memory staging 都可能导致冲突，不能简单归因于某个 DSL |

- **低冲突点分析**

  | 特征 | 说明 |
  |---|---|
  | 一些点位于 **Conflict score ≈ 10^-3%** | 几乎没有可观测 LSU bank-conflict |
  | 这些点也大多具有高 branch efficiency | 表示 branch behavior 干净，但没有 bank-conflict 问题 |
  | 圆形 marker 较多 | 与 **No direct conflict** 诊断一致 |

- **Confounded 样本分析**

  | 位置 | 含义 |
  |---|---|
  | 左侧少量 diamond 点 | branch efficiency 明显低于主群体 |
  | y 值从约 10^-3% 到数 % 不等 | conflict score 可能受 active-lane mask、控制流路径差异影响 |
  | 论文处理方式 | 将低 branch efficiency 视为 **confounder**，不直接作为 bank-conflict 强证据 |

- **与论文 bank-conflict 分析的关系**

  | 论文问题 | Figure 14 的作用 |
  |---|---|
  | 是否能直接根据 conflict score 判断 bank conflict？ | 不能，需要排除 branch divergence 混淆 |
  | 高 conflict score 是否可能只是分支发散造成？ | 图中显示多数高 conflict score 样本 branch efficiency 很高，因此通常不是 |
  | 是否需要过滤低 branch efficiency case？ | 是，低 branch efficiency case 应标记为 **Confounded** |
  | 对 Figure 10、Figure 12、Figure 13 的支撑 | Figure 14 是 sanity check，增强 bank-conflict 诊断可信度 |

- **对 Triton 与 cuTile 的比较**

  | 维度 | Triton | cuTile |
  |---|---|---|
  | 高 branch efficiency 样本 | 很多 | 很多 |
  | 高 conflict score 样本 | 存在 | 存在 |
  | Confounded 样本 | 少量 | 少量 |
  | 总体结论 | **不存在单一 backend 被 branch divergence 系统性污染的现象** | **同样不存在系统性污染** |

- **方法学意义**

  | 点 | 说明 |
  |---|---|
  | **branch efficiency 是归因过滤器** | 用于判断 bank-conflict 证据是否可靠 |
  | **conflict score 不能孤立解释** | 必须结合 branch efficiency、NCU Source view、SASS/PTX 分析 |
  | **高 branch efficiency + 高 conflict score** | 更可信地指向 shared-memory bank conflict |
  | **低 branch efficiency + 高 conflict score** | 需要谨慎，可能是控制流发散导致统计失真 |

- **图像支持的关键论断**

  | 论断 | 支持程度 |
  |---|---|
  | **严重 bank-conflict 信号集中在少数 kernel** | 支持 |
  | **branch divergence 不是大多数 conflict score 的主要来源** | 强支持 |
  | **Triton 与 cuTile 都可能出现 bank conflict** | 支持 |
  | **低 branch efficiency case 应从直接冲突证据中剥离** | 支持 |
  | **IPC 或单一性能指标不能替代 conflict diagnosis** | 间接支持，与 Figure 15 呼应 |

- **潜在优化启示**

  | 现象 | 优化方向 |
  |---|---|
  | 高 conflict score 且 branch efficiency 高 | 优先检查 **shared-memory layout**、**tile swizzling**、**padding**、**operand staging** |
  | 高 conflict score 出现在 MMA source line | 不一定是 Tensor Core 算术问题，可能是 **MMA operand staging** 问题 |
  | cuTile 高冲突 | 检查 **ct.load / ct.store / ct.mma** 的 tile shape、shared-memory staging、TMA path |
  | Triton 高冲突 | 检查 **tl.load 到 shared memory lowering**、block pointer layout、LDGSTS/LDS pattern |
  | branch efficiency 低 | 先处理 **masking / boundary condition / branch structure**，再判断 bank conflict |

- **综合评价**

  | 维度 | 评价 |
  |---|---|
  | 诊断价值 | **较高**，用于验证 bank-conflict 归因是否被 branch divergence 混淆 |
  | 结论清晰度 | **较强**，绝大多数样本 branch efficiency 接近 100% |
  | 局限性 | 图中不直接显示 operator 名称与 dtype，需要结合 Figure 10/12/13 定位具体 kernel |
  | 对论文主张的贡献 | 强化了论文关于 **profiling-guided diagnosis** 的可信度 |

- **一句话总结**

  | 总结 |
  |---|
  | **Figure 14 表明，大多数高 bank-conflict 信号出现在 branch efficiency 接近 100% 的样本中，因此这些冲突更可能来自 shared-memory staging/layout，而不是 branch divergence；少量低 branch-efficiency 样本被合理标记为 Confounded，避免了错误归因。** |

### 98dcb8a46260be12cbf9061048965ee05ee7785ee754a725ee8a32a6184688d8.jpg

![98dcb8a46260be12cbf9061048965ee05ee7785ee754a725ee8a32a6184688d8.jpg](98dcb8a46260be12cbf9061048965ee05ee7785ee754a725ee8a32a6184688d8.jpg)

- **图像对象**：该图为论文 TileBench 附录 D.3.3 中的 **Figure 15: IPC-gap sanity check**，用于验证 **issued/executed IPC gap** 是否可以作为 **shared-memory bank conflict** 的代理指标。

- **核心结论**：图中几乎所有点都聚集在 **Issued/executed IPC gap 接近 0** 的位置，但对应的 **Conflict score** 从极低到接近 **100%** 均有分布，因此 **IPC gap 不能预测 bank conflict 严重程度**。这也是图标题 **“IPC gap is not a conflict proxy”** 的直接含义。

- **图像元素概览**：

| 元素 | 内容 |
|---|---|
| 图标题 | **IPC gap is not a conflict proxy** |
| 横轴 | **Issued/executed IPC gap** |
| 纵轴 | **Conflict score (%)** |
| 纵轴尺度 | **log scale**，约从 **10⁻³ 到 10²** |
| 横轴范围 | 约 **0.000 到 0.010** |
| 点颜色 | backend：**cuTile** 为橙色，**Triton** 为蓝色 |
| 点形状 | diagnosis group：No direct conflict、Likely、Mild、Severe/Moderate、Confounded |
| 分析目的 | 判断 **IPC gap** 是否能代表 **bank conflict signal** |

- **坐标轴含义**：

| 轴 | 指标 | 含义 |
|---|---|---|
| x-axis | **Issued/executed IPC gap** | issued IPC 与 executed IPC 之间的差值，用于观察指令发射与实际执行之间是否存在明显落差 |
| y-axis | **Conflict score (%)** | 归一化后的 shared-memory bank conflict 严重程度，基于 Nsight Compute 的 LSU shared-memory conflict counters |
| y-axis log scale | **对数坐标** | 用于同时展示极小冲突值和高冲突值，说明冲突强度跨越多个数量级 |

- **主要视觉现象**：

| 观察 | 说明 |
|---|---|
| **点高度分布很宽** | Conflict score 从约 **10⁻³%** 到接近 **10²%**，跨度极大 |
| **点横向分布很窄** | 大多数样本的 IPC gap 聚集在 **0 附近** |
| **高冲突点并不对应高 IPC gap** | 很多 Conflict score 高达 **10%、100%** 的点，x 值仍接近 **0** |
| **少量点向右偏移** | 有少数 cuTile 点出现在约 **0.001–0.003** 附近，但并未形成清晰趋势 |
| **Triton 与 cuTile 混合分布** | 两个 backend 都存在低冲突和高冲突样本，说明该现象不是单一 backend 特有 |

- **对趋势的判断**：

| 关系 | 图中证据 | 结论 |
|---|---|---|
| IPC gap 与 Conflict score 是否正相关 | 高 Conflict score 样本大量集中在 x≈0 | **否** |
| IPC gap 是否能识别 severe conflict | Severe/Moderate 标记同样主要集中在 x≈0 | **不能** |
| IPC gap 是否能区分 backend | cuTile 与 Triton 点在 x≈0 区域高度重叠 | **不能** |
| IPC gap 是否适合作为 bank conflict proxy | 横向变化不足，纵向变化巨大 | **不适合** |

- **诊断分组含义**：

| diagnosis group | 图例形状 | 含义 |
|---|---|---|
| **No direct conflict** | 圆点 | NCU 中没有直接 LSU bank-conflict 证据 |
| **Likely** | x 标记 | 可能存在 bank conflict，但证据较间接 |
| **Mild** | 加号 | 存在轻度 bank conflict 信号 |
| **Severe/Moderate** | 方块 | 存在中到重度 bank conflict 信号 |
| **Confounded** | 菱形 | 结果可能被其他因素干扰，例如 branch divergence |

- **backend 分布分析**：

| backend | 颜色 | 图中表现 |
|---|---|---|
| **cuTile** | 橙色 | 样本数量较多，Conflict score 覆盖低、中、高多个区间；少数点在 x 方向有轻微偏移 |
| **Triton** | 蓝色 | 多数点也集中在 x≈0；既有极低 conflict，也有接近高 conflict 的样本 |
| **共同点** | — | 两者均不呈现 IPC gap 与 Conflict score 的线性或单调关系 |

- **为什么 IPC gap 不是可靠 proxy**：

| 原因 | 解释 |
|---|---|
| **指标层级不同** | IPC gap 是整体指令发射/执行层面的粗粒度指标；bank conflict 是 shared-memory LSU 路径上的局部结构性问题 |
| **bank conflict 可被其他 stall 掩盖** | 即使 shared-memory bank conflict 严重，总体 IPC gap 也可能很小，因为瓶颈可能体现在 memory pipeline、scoreboard、synchronization 或 operand staging |
| **高 IPC gap 不必然来自 bank conflict** | IPC gap 增大也可能由 branch divergence、dependency chain、async wait、register pressure 等因素造成 |
| **低 IPC gap 不代表无 conflict** | 图中大量高 Conflict score 样本仍位于 x≈0，说明低 IPC gap 无法排除 bank conflict |

- **与论文上下文的关系**：

| 论文位置 | 作用 |
|---|---|
| Appendix D.3.3 | 作为 **Additional Bank-conflict Diagnostics** 的 sanity check |
| Figure 10 | 展示 top-20 bank-conflict cases |
| Figure 11 | 给出自动化 NCU bank-conflict diagnosis 分布 |
| Figure 12 | 展示 operator/backend/dtype heatmap |
| Figure 13 | 比较 Triton 与 cuTile 的 matched conflict score |
| Figure 14 | 检查 branch-divergence confounder |
| **Figure 15** | 验证 **IPC gap 不应被用作 bank conflict proxy** |

- **对 TileBench 性能诊断方法的启示**：

| 启示 | 说明 |
|---|---|
| **必须使用 NCU counter** | bank conflict 需要依赖 Nsight Compute 中的 shared-memory conflict counters，而不是 IPC gap |
| **需要 source-level diagnosis** | 单个全局指标不足以定位问题，需要结合 Source View、SASS、LDS/STL、LDGSTS、MMA operand staging 等信息 |
| **不能用 IPC gap 替代 conflict score** | IPC gap 只能作为 sanity check，不能作为主要诊断依据 |
| **性能瓶颈需多指标联合判断** | bank conflict、branch divergence、stall_long_scoreboard、register spilling、TMA wait、TMEM pipeline wait 都可能共同影响性能 |

- **图中最重要的信息**：

| 关键点 | 解释 |
|---|---|
| **Conflict score 高低与 IPC gap 无明显关系** | 纵向分布很宽，横向几乎不变 |
| **高 bank conflict 不一定导致明显 IPC gap** | 说明 IPC gap 对 shared-memory 局部冲突不敏感 |
| **Triton 与 cuTile 均受影响** | bank conflict 不是某一个 backend 独有问题 |
| **诊断应基于硬件 counter，而非间接 proxy** | 论文选择 conflict score 而非 IPC gap 是合理的 |

- **总体解读**：该图通过散点分布证明，**issued/executed IPC gap 与 normalized LSU shared-memory conflict signal 之间没有稳定对应关系**。因此，在 TileBench 的 profiling-guided diagnosis 中，作者只把 IPC gap 作为 **sanity check**，而不将其作为 **bank conflict proxy**。真正的 bank conflict 判断仍需依赖 **Nsight Compute shared-memory bank-conflict metrics**、source-level attribution 和 SASS/NCU 证据。

### 8280c07c994efbcaeb79af11fbfd5ac7be69c82490fbbab5d2e2b7956adea84c.jpg

![8280c07c994efbcaeb79af11fbfd5ac7be69c82490fbbab5d2e2b7956adea84c.jpg](8280c07c994efbcaeb79af11fbfd5ac7be69c82490fbbab5d2e2b7956adea84c.jpg)

- **图像对象**：该图为论文 Appendix D.3.4 中的 **Figure 16: Triton tensor-descriptor ablation after autotuning**，用于比较 Triton 在启用 **tensor descriptor / TMA-style path** 后，相对于普通 **pointer-based baseline** 的速度变化。

- **核心含义**：横轴是 **Descriptor speedup over pointer baseline**，即：
  
  | 数值位置 | 含义 |
  |---|---|
  | **> 1×** | 使用 **tensor descriptor** 更快 |
  | **= 1×** | 与 pointer baseline 持平 |
  | **< 1×** | 使用 **tensor descriptor** 更慢 |
  | **虚线 1×** | 性能分界线，右侧为 faster，左侧为 slower |

- **图形编码方式**：

  | 视觉元素 | 含义 |
  |---|---|
  | y 轴 | 不同 operator |
  | x 轴 | descriptor 相对 pointer baseline 的 speedup |
  | 黑色菱形 | **operator median**，表示该算子的中位性能变化 |
  | 彩色圆点 | 不同 dtype 下的测量结果 |
  | 横向灰线 | 同一 operator 在不同 dtype / case 下的性能跨度 |
  | 背景左侧浅红区域 | descriptor 更慢 |
  | 背景右侧浅绿区域 | descriptor 更快 |
  | 竖直虚线 | **1× parity line** |

- **总体结论**：**Triton tensor descriptor 并不是通用优化手段**。图中大多数 operator 的黑色菱形位于 **1× 左侧**，说明在多数场景下，descriptor/TMA 路径相比普通 pointer load 会带来性能回退。

- **性能显著回退的 operator**：

  | Operator | 观察 | 解释 |
  |---|---|---|
  | **layernorm** | 中位值约在 0.35× 左右 | descriptor 对流式 reduction/norm 无明显复用收益，反而引入额外 staging |
  | **dequantize_rowwise** | 约 0.45×–0.55× | 数据访问偏轻量/解包逻辑主导，TMA 不划算 |
  | **rmsnorm** | 约 0.5× | reduction/norm 类 kernel 更适合直接 pointer load |
  | **cross_entropy** | 约 0.5×–0.6× | 控制与 reduction 逻辑复杂，descriptor 难以带来 tile reuse |
  | **matrix_transpose** | 分布跨度较大，但中位低于 1× | descriptor 对部分 dtype/case 有帮助，但整体不稳定 |
  | **matrix_copy** | 中位约 0.55×–0.6× | 纯 copy 属于 one-pass streaming，TMA staging 成本高于收益 |

- **轻度回退或接近持平的 operator**：

  | Operator | 观察 | 含义 |
  |---|---|---|
  | **softmax** | 中位约 0.65×–0.7× | descriptor 多数情况下减速 |
  | **interleave** | 中位约 0.67× | 数据布局类 streaming kernel 不适合 descriptor |
  | **block_sparse_attention** | 中位约 0.67× | irregular sparse access 不适合规则 descriptor |
  | **matmul_fp16/fp8** | 中位约 0.67×，但有个别点接近/超过 1× | 部分矩阵乘可能受益，但整体实现未稳定利用 TMA |
  | **kl_divergence** | 约 0.7× | reduction/loss 类仍偏 pointer-friendly |

- **接近 1× 的 operator**：

  | Operator | 观察 | 解释 |
  |---|---|---|
  | **streamk_matmul** | 中位略高于 1× | tile movement 与 computation 有一定复用/重叠，descriptor 开始有效 |
  | **matmul_int8** | 接近或略高于 1× | int8 GEMM 更容易从 TMA / descriptor-like movement 获益 |
  | **flash_attention** | 明显高于 1× | attention 中 KV tile 复用较强，descriptor 更适合 |
  | **flash_decode** | 中位约 0.9×–1.0× | 接近持平，但未形成稳定收益 |

- **明显受益的 operator**：

  | Operator | 图中位置 | 主要原因 |
  |---|---|---|
  | **flash_attention** | 最右侧之一，约 1.1× | dense attention 具有较强 tile reuse，适合 TMA-style path |
  | **matmul_int8** | 略高于 1× | Tensor Core / TMA 友好，数据搬运可被计算掩盖 |
  | **streamk_matmul** | 略高于 1× | Stream-K GEMM 存在更复杂 tile 调度，descriptor 可改善部分访存 |

- **dtype 层面的观察**：

  | dtype | 图中表现 |
  |---|---|
  | **bf16** | 多数跟随 operator 主趋势，少数 matmul/attention 场景接近或超过 1× |
  | **fp16** | 在 attention / matmul 类中更可能受益 |
  | **fp32** | 多数 operator 中 descriptor 不占优 |
  | **fp8_e5m2 / fp8_e4m3fn** | 出现在 matmul 相关 kernel 中，表现差异较大 |
  | **int8** | 在 **matmul_int8** 附近表现较好，是 descriptor/TMA 较可能获益的场景 |

- **重要异常点**：
  - **matrix_transpose** 存在较长横向跨度，说明 descriptor 对不同 dtype 或 shape 的影响不稳定。
  - **matmul_fp16/fp8** 中有个别彩色点靠近甚至超过 **1×**，但 operator median 仍低于 1×，说明不是所有 GEMM-like kernel 都能自动受益。
  - **streamk_matmul** 和 **flash_attention** 是少数明确落在右侧绿色区域的 operator，代表 descriptor 更适合 **tile-reuse-heavy** 负载。

- **与论文结论的对应关系**：
  - 论文指出：**Triton tensor descriptors are not a general replacement for pointer loads**。
  - 图像直接支持这一点：大部分 operator 位于 **1× 左侧**。
  - descriptor 仅在少数 **tile-reuse-heavy kernels** 中有效，例如 **flash_attention、matmul_int8、streamk_matmul**。
  - 对于 **pointwise、layout、normalization、reduction** 等 one-pass streaming kernel，descriptor 通常带来额外 shared-memory staging 或 TMA setup 成本，导致回退。

- **性能机制解释**：
  - **pointer-based tl.load** 更适合：
    - irregular address
    - masked load
    - one-pass streaming
    - lightweight bandwidth-bound kernel
    - reduction/norm kernel
  - **tensor descriptor / TMA path** 更适合：
    - regular tile movement
    - tile reuse
    - computation can overlap memory transfer
    - Tensor Core-heavy workload
    - attention / GEMM-like kernel

- **实践建议**：
  - **不要默认启用 Triton tensor descriptor**。
  - 对 **streaming pointwise / reduction / norm / layout** kernel，应优先使用 **plain pointer loads**。
  - 对 **flash_attention、matmul_int8、streamk_matmul** 等具有 tile reuse 和计算重叠潜力的 kernel，可以尝试 descriptor/TMA。
  - descriptor 是否有效需要结合 **operator structure、dtype、shape、reuse pattern** 逐 case 验证。
  - 单纯替换访存 API 不足以保证加速，关键在于 compiler lowering 是否能生成合适的 **TMA / Tensor Core / shared-memory staging** 路径。

- **一句话总结**：该图表明，**Triton tensor descriptor 是一种条件性优化，而不是通用加速开关；它主要服务于规则、可复用、计算密集的 tile workload，而会拖慢大量 irregular 或 streaming kernel**。

### e9e45d81864f1e99ab5bd4542c44ef7ee151bd4710d6eae1fd3a21ac42153072.jpg

![e9e45d81864f1e99ab5bd4542c44ef7ee151bd4710d6eae1fd3a21ac42153072.jpg](e9e45d81864f1e99ab5bd4542c44ef7ee151bd4710d6eae1fd3a21ac42153072.jpg)

- **图片对象**：该图是 TileBench 论文附录中的 **Figure 17: Iterative refinement trajectory of the four (model, backend) combinations on softmax**，展示 **softmax** 算子在 **10 次 LLM iterative refinement** 中，不同 **LLM + backend** 组合生成 kernel 的性能变化。

- **图中变量含义**：

| 元素 | 含义 |
|---|---|
| **x-axis: Iteration** | LLM 代码生成/优化迭代轮次，范围约为 **0–9** |
| **y-axis: Speedup vs PyTorch** | 相对 PyTorch baseline 的加速比，计算为 **Ttorch / Tbackend** |
| **灰色虚线 PyTorch parity (1×)** | 与 PyTorch 持平的基准线；高于该线表示生成 kernel 快于 PyTorch |
| **蓝色实线** | **GPT-5.5 + Triton** |
| **橙色实线** | **GPT-5.5 + cuTile** |
| **蓝色虚线** | **Claude Opus 4.7 + Triton** |
| **橙色虚线** | **Claude Opus 4.7 + cuTile** |
| **× 标记** | verification failed，未记录有效 speedup |

- **总体结论**：该图清楚表明，在 softmax 任务上，**Triton 作为 LLM 生成目标明显更稳定、更容易达到 PyTorch parity 以上**；而 **cuTile 的迭代轨迹波动更大，且更容易出现 verification failure 或低于 PyTorch 的性能**。

- **各组合表现概览**：

| 组合 | 初始表现 | 迭代稳定性 | 是否超过 PyTorch 1× | 最终/峰值趋势 |
|---|---:|---|---|---|
| **GPT-5.5 + Triton** | 约 **1.5×** | **最稳定** | 全程基本高于 1× | 峰值约 **2.64×**，出现在 iteration 8 |
| **GPT-5.5 + cuTile** | iteration 0 verification failed | **波动较大** | 多数后期超过 1× | iteration 4 降至约 **0.76×**，后期恢复到约 **2.45×** |
| **Claude Opus 4.7 + Triton** | 约 **1.4–1.5×** | 较稳定 | 全程高于 1× | 最终约 **1.73×** |
| **Claude Opus 4.7 + cuTile** | 约 **0.3–0.4×** | 低位波动 | **始终低于 1×** | 始终约 **0.2–0.45×**，未达到 PyTorch parity |

- **GPT-5.5 + Triton 的轨迹分析**：
  - 从 **iteration 0** 开始就超过 PyTorch，说明 GPT-5.5 对 **Triton softmax kernel** 的初始生成能力较强。
  - 整个曲线维持在 **约 2× 左右或以上**，说明后续 refinement 没有造成明显退化。
  - 在 **iteration 8** 达到最高点，论文文本给出的峰值为 **2.64×**。
  - 该轨迹体现出 **Triton API、masking、tl.load/tl.store、row-wise reduction** 等模式更符合 LLM 已学习到的常见代码分布。

- **GPT-5.5 + cuTile 的轨迹分析**：
  - **iteration 0 verification failed**，说明初始 cuTile 代码在 correctness 上存在问题。
  - iteration 1 后开始出现有效性能，约高于 **1×**。
  - iteration 4 明显跌破 PyTorch parity，约 **0.76×**，说明 refinement 过程中配置或实现策略发生退化。
  - 后期性能持续恢复，并在 iteration 8–9 达到较高水平，最终约 **2.45×**。
  - 这说明 **GPT-5.5 能够通过反馈逐步修正 cuTile kernel**，但搜索路径不稳定，代价更高。

- **Claude Opus 4.7 + Triton 的轨迹分析**：
  - 初始即超过 PyTorch parity，约 **1.4–1.5×**。
  - 后续曲线变化较小，整体稳定在 **1.4×–1.7×** 左右。
  - iteration 9 达到约 **1.73×**。
  - 相比 GPT-5.5 + Triton，Claude + Triton 的峰值较低，但仍体现出 **Triton backend 的可生成性较好**。

- **Claude Opus 4.7 + cuTile 的轨迹分析**：
  - 全程处于 PyTorch parity 线下方，约 **0.2×–0.45×**。
  - 没有任何 iteration 成功超过 **1×**。
  - 说明 Claude 在该实验设置下对 **cuTile softmax kernel** 的优化能力明显不足。
  - 该结果与论文 RQ4 的总体结论一致：**cuTile 是更陌生、更难被 LLM 稳定生成高性能代码的目标 DSL**。

- **关键数值解读**：

| 指标 | 观察 |
|---|---|
| **最高 speedup** | **GPT-5.5 + Triton ≈ 2.64×** |
| **cuTile 最好表现** | **GPT-5.5 + cuTile ≈ 2.45×** |
| **最稳定组合** | **GPT-5.5 + Triton** |
| **最弱组合** | **Claude Opus 4.7 + cuTile** |
| **最明显退化点** | **GPT-5.5 + cuTile iteration 4 ≈ 0.76×** |
| **始终低于 PyTorch 的组合** | **Claude Opus 4.7 + cuTile** |

- **图中最重要的现象**：
  - **backend 差异大于 model 差异**：同一个模型在 Triton 上通常明显优于 cuTile。
  - **Triton 收敛更快**：iteration 0 即能生成超过 PyTorch 的实现。
  - **cuTile refinement 波动更强**：可能出现 verification failure、性能退化和后期恢复。
  - **LLM 对 Triton 的先验更强**：Triton 已有更多公开代码、文档和实际系统使用案例，LLM 更容易生成正确且高性能的 kernel。
  - **cuTile 的抽象较新**：LLM 对其 API、tile shape、padding、ct.load/ct.store/reduction 写法掌握不足，导致 correctness 与 performance 都更不稳定。

- **与论文 RQ4 的关系**：
  - 该图是 RQ4 的局部案例，展示 **softmax** 上的 iteration-level 行为。
  - 主文 RQ4 结论是：**Triton is easier and more token-efficient for LLM-generated kernels than cuTile**。
  - 图中 softmax 轨迹支持该结论：Triton 不仅最终性能更好，而且从初始 iteration 就更可靠。
  - cuTile 即使最终可能追上，例如 **GPT-5.5 + cuTile** 后期达到约 **2.45×**，也需要更多迭代、更高搜索成本和更大的 correctness 风险。

- **性能原因推测**：
  - **softmax** 是典型 **row-wise reduction + normalization** 算子，Triton 中常见实现模式为：
    - 使用 **tl.arange** 构造 block offsets；
    - 使用 **tl.load with mask** 处理边界；
    - 使用 **tl.max / tl.sum** 完成 reduction；
    - 使用 **tl.exp** 和归一化；
    - 使用 **tl.store with mask** 写回。
  - 这些 Triton 模式在公开代码中非常常见，因此 LLM 容易生成。
  - cuTile 中需要更严格地处理：
    - **power-of-two tile shape**；
    - **padding_mode**；
    - tile-level reduction；
    - ct.load/ct.store 的静态 tile index；
    - occupancy 与 TILE 的协调。
  - 这些约束使 LLM 更容易写出 correctness 问题或性能较差的实现。

- **最终评价**：
  - 这张图的核心信息是：在 LLM 生成 softmax kernel 的场景下，**Triton 表现出更好的初始正确性、更快的性能收敛和更稳定的迭代轨迹**。
  - **cuTile 并非没有性能潜力**，GPT-5.5 后期能达到接近 Triton 的 speedup；但其过程明显更不稳定。
  - 对于当前 LLM 代码生成生态，**Triton 是更成熟、更 token-efficient、更适合作为自动 kernel generation target 的 tile-based DSL**。

