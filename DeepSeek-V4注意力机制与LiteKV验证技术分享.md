# 1M 上下文不是把窗口拉大：DeepSeek-V4 如何把长文本变成“可检索的压缩记忆”

> 这篇文章试图回答一个问题：DeepSeek-V4 宣称把 1M（一百万）上下文变成官方服务标配，背后的关键到底是不是“窗口更大”？我的结论是：不是。真正的变化是注意力机制从“逐 token 全量回看”，转向“远程压缩、局部保真、按需稀疏取回”的记忆系统。

本文基于公开资料、用户提供的微信文章，以及本仓库 LiteKV demo。需要先划清边界：LiteKV 不是 DeepSeek-V4 官方复现，不加载官方权重，也不证明真实 1M 上下文能力。它是一个 mechanism microscope（机制显微镜）：用最小代码把 token 维度压缩、DSA 稀疏注意力（DeepSeek Sparse Attention）、CSA（Compressed Sparse Attention，压缩稀疏注意力）和 local window（局部窗口）拆开，分析这条路线为什么能降低计算与显存压力。

## 一、长上下文真正贵在哪里？

很多人听到 1M context window（上下文窗口）时，第一反应是“模型能塞进更多字了”。但从系统角度看，难点不是“能不能放进去”，而是“放进去之后每一步生成还算不算得动”。

传统 Transformer 的注意力（Attention）会让当前 query（查询向量）和历史 token 的 key/value（键/值向量）交互。上下文变长后，两个成本同时爆炸：

第一是 attention FLOPs（Floating Point Operations，浮点运算量）。如果 query 要和所有历史 token 打分，历史越长，每一步注意力成本越高。

第二是 KV cache（Key-Value Cache，键值缓存）。自回归生成时，模型会缓存历史 token 的 K/V，避免每一步重新计算。上下文到几十万甚至百万 token 后，KV cache 本身就会变成显存大户。

所以，长上下文不是一个单纯的“位置编码问题”，而是一个 attention 计算图和 memory layout（内存布局）问题。谁能让模型少看、不乱看、还能找回关键远程信息，谁就有机会把长上下文做成可用能力。

## 二、已有路线：窗口、稀疏、压缩，各有代价

市面上处理长上下文，大体有三类路线。

第一类是 sliding window attention（滑动窗口注意力）。Longformer 用局部窗口加少量 global tokens（全局 token）把复杂度降下来；Mistral 7B 也使用 SWA（Sliding Window Attention）降低推理成本。它的优点是简单、便宜、工程友好；缺点也明显：窗口外的信息默认不可见。如果目标信息在很早的位置，模型可能根本看不到。

第二类是 sparse attention（稀疏注意力）。BigBird 通过局部、随机、全局连接构造稀疏图，让注意力不必覆盖全部 token。它证明了稀疏连接在理论上也能保留较强表达能力。但固定稀疏模式有一个问题：它不知道当前 query 真正需要哪段历史。

第三类是 memory compression（记忆压缩）。Infini-attention 这类方法会把长期历史压进 bounded memory（有界记忆），最近上下文仍然保留局部注意力。很多工程系统也会用摘要、检索、分层缓存解决长文本问题。它们共同的思想是：远程历史不应该永远以原始 token 级别留在主 attention 里。

DeepSeek-V4 的精彩之处在于，它不是简单选一条路线，而是把“压缩”和“稀疏选择”组合进模型原生注意力里。

## 三、DeepSeek-V4 的关键：不是看更多，而是更会记

公开解读中，DeepSeek-V4 的长上下文注意力采用 hybrid attention（混合注意力），核心包括 CSA（Compressed Sparse Attention，压缩稀疏注意力）和 HCA（Heavily Compressed Attention，高压缩注意力）。

CSA 可以先用一句话理解：

> 把远程上下文从 token 级 KV 压缩成 block 级 KV，再让 query 只选择 top-k 个相关 block。

更具体一点：远程 K/V 不再每个 token 都原样保存，而是每 4 个 token 压成 1 个 compressed KV entry（压缩 KV 条目）。这样序列维度先缩短 4 倍。然后 lightning indexer（轻量索引器）对这些 compressed blocks（压缩块）打分，为当前 query 选出 top-k 个相关 block。最后 query 只对这些 block 做注意力。

同时，模型还保留 sliding-window branch（滑动窗口分支）处理最近的未压缩 token。因为近处信息通常包含语法、局部依赖和当前指令细节，压得太狠会损伤质量。

HCA 则更像一个全局摘要层：它把远程 KV 进一步高倍率压缩，例如 128 倍，然后在更短的压缩序列上做 dense attention（稠密注意力）。如果 CSA 像“从远程档案库里按需抽几份卷宗”，HCA 就像“先读一份全局摘要”。

这套设计的底层通解是：

- 近处 token：保真，直接看。
- 远处 token：压缩，减少 KV cache。
- 重要远程信息：用 indexer 找回来。
- 不重要远程信息：只保留统计摘要。

这就是我认为 DeepSeek-V4 最值得研究的地方：它不是把 RAG（Retrieval-Augmented Generation，检索增强生成）外挂到模型外部，而是在 attention 内部重构了一套“压缩可检索记忆层”。

## 四、LiteKV：用一个小 demo 验证核心机制

为了验证这个想法，我做了 LiteKV。它只做一件事：构造一个 synthetic retrieval case（合成检索样例），让远程某个 token 的 key 和 query 高度相似，并把目标 value 做上标记。然后比较不同注意力机制能否找回这个远程目标，以及要付出多少理论 KV cache 和 FLOPs 成本。

核心入口非常小：

```python
result = run_attention(
    case,
    "csa_lite_local",
    top_k=32,
    compression_ratio=4,
    local_window=128,
)
metrics = result.metrics.as_dict()
```

LiteKV 目前实现四种模式：

- `dense`：完整看历史 token，作为正确性与成本基线。
- `sliding_window`：只看最近窗口，成本低，但远程目标可能不可达。
- `csa_lite`：每 4 个 token 均值压成 block，query 对 compressed key 打分，选择 top-k blocks。
- `csa_lite_local`：远程使用 compressed top-k，本地窗口保留未压缩 token。

运行方式：

```bash
source .venv/bin/activate
python experiments/run_litekv.py
```

它会生成：

- `results/metrics.csv`
- `results/metrics.json`
- `results/kv_cache_vs_context.png`
- `results/flops_vs_context.png`
- `results/retrieval_accuracy.png`
- `results/topk_tradeoff.png`

这些结果不是为了证明 LiteKV 很强，而是为了证明一个机制判断：如果远程上下文可以压缩成 block，而且 query 能选中相关 block，那么模型不需要全量回看所有 token，也能保留远程检索信号。

## 五、结果：sliding window 便宜但漏检，CSA-lite 便宜且能找回

默认实验比较 512、1024、2048、4096 context 下的四种模式。取 `top_k=32`、`local_window=128` 的代表 slice，4096 context 下结果如下：

```text
dense            KV=8,388,608   FLOPs=2,097,152   recall=1.0
sliding_window   KV=262,144     FLOPs=65,536      recall=0.0
csa_lite         KV=2,097,152   FLOPs=540,672     recall=1.0
csa_lite_local   KV=2,293,760   FLOPs=589,824     recall=1.0
```

这组数字很直观：

- Dense attention 能找回目标，但成本最大。
- Sliding window 成本最低，但远程目标在窗口外，recall（召回率）直接变成 0。
- CSA-lite 把 KV cache 降到 dense 的约 1/4，FLOPs 约下降 3.88 倍，同时 recall 仍为 1。
- CSA-lite + local window 略贵于纯 CSA-lite，但同时保留远程检索和局部未压缩细节。

下面是 demo 生成的几张图。

![KV cache vs context](results/kv_cache_vs_context.png)

![FLOPs vs context](results/flops_vs_context.png)

![Retrieval accuracy](results/retrieval_accuracy.png)

![Top-k tradeoff](results/topk_tradeoff.png)

最值得看的不是 latency 图，而是 KV cache、FLOPs 和 recall。因为当前 LiteKV 为了可读性保留了 pure Python fallback（纯 Python 回退实现），不是优化过的 CUDA/MPS kernel（GPU/Apple Metal 后端内核）。因此 latency 只能当本地 demo timing（本地演示计时），不能当生产性能结论。

## 六、DeepSeek 的亮点到底在哪里？

我认为亮点有四个层次。

第一，压缩发生在 token/KV 维度，而不是只在文本层面做摘要。文本摘要会丢结构，外部 RAG 会引入检索系统边界；KV 维度压缩更接近模型内部记忆表示。

第二，稀疏选择不是固定模板，而是 query-aware（查询感知）。当前 query 需要什么，就从 compressed blocks 里选什么。这比固定窗口更适合远程 needle 检索。

第三，它不是只做远程压缩，也保留本地窗口。长上下文模型不能只会“翻档案”，还要能处理眼前这句话的局部关系。

第四，它把 CSA、HCA、低精度 KV、专用 kernel 和训练目标组合成系统工程。单独看每个 idea，学术界和工业界都有类似方向；但真正困难的是让它们在大模型里稳定训练、稳定推理、成本可控。

## 七、这篇 demo 能证明什么？

它能证明的是机制趋势：

> 远程 KV 压缩 + query-aware sparse top-k，可以在显著降低理论注意力成本的同时，保留构造任务中的远程检索能力。

它不能证明的是：

- 不能证明 DeepSeek-V4 官方 1M 上下文能力。
- 不能证明真实语言任务准确率。
- 不能证明本地 Python latency 等价于生产推理性能。

但我觉得这正是 demo 的价值。它不是拿玩具模型冒充大模型，而是把大模型论文/解读里的关键结构拆成可运行、可画图、可讨论的最小机械模型。对于技术分享来说，这比堆概念更有说服力。

## 八、我会如何概括 DeepSeek-V4 的长上下文路线

如果只用一句话：

> DeepSeek-V4 的 1M 上下文不是“把所有历史都看一遍”，而是把历史组织成一套分层记忆：近处保真，远处压缩，重要信息按需取回。

这也是未来长上下文模型的大方向。窗口会越来越长，但真正决定体验的，不是窗口数字，而是模型是否有一套高效的记忆管理机制。

LiteKV 给我的验证结论也很明确：sliding window 能省钱，但会漏掉远程信息；dense attention 不漏，但太贵；CSA-lite 这种“压缩后稀疏检索”的路线，正好站在两者之间，既保留远程检索信号，又把理论成本压下来。

这就是 DeepSeek-V4 这类架构最值得关注的地方。

## 参考链接

- 背景文章：[微信公众号原文](https://mp.weixin.qq.com/s/8bxXqS2R8Fx5-1TLDBiEDg)
- DeepSeek-V4 解读：[Hugging Face Blog: DeepSeek-V4](https://huggingface.co/blog/deepseekv4)
- DeepSeek Sparse Attention 公开仓库：[DeepSeek-V3.2-Exp GitHub](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- DeepSeek-V3.2 论文：[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)
- Longformer：[arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
- BigBird：[arXiv:2007.14062](https://arxiv.org/abs/2007.14062)
- Mistral 7B：[arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- Infini-attention：[arXiv:2404.07143](https://arxiv.org/abs/2404.07143)
