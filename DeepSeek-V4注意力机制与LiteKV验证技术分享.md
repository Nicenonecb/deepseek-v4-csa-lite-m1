# DeepSeek-V4 的长上下文注意力：从 token 压缩、DSA 稀疏选择到 LiteKV 验证

> 本文基于公开资料与本仓库 LiteKV demo。它不是 DeepSeek-V4 的官方复现，也不声称验证 1M（一百万）上下文真实能力；它的目标是把“token 维度压缩 + DSA 稀疏注意力（DeepSeek Sparse Attention）+ local window（局部窗口）”这条路线拆开，用一个可运行的小实验解释为什么它能降低长上下文的计算和显存压力。

## 1. 为什么 1M 上下文难，不只是“窗口开大”

传统 Transformer 的注意力（Attention）在每一步生成时都要拿当前 query（查询向量）和历史 token 的 key/value（键/值向量）交互。上下文越长，两个成本越明显：

第一是 attention FLOPs（Floating Point Operations，浮点运算量）。如果每个 token 都看所有历史 token，理论成本会随序列长度快速增长。第二是 KV cache（Key-Value Cache，键值缓存）。自回归生成时，为了避免重复计算，模型会缓存历史 token 的 K/V；上下文到几十万、上百万 token 后，缓存本身就会成为显存瓶颈。

这也是 DeepSeek-V4 相关解读里强调的重点：1M context window（上下文窗口）只是容量，真正能不能用，取决于每次 forward pass（前向计算）在这个长度下是否足够便宜。Hugging Face 对 DeepSeek-V4 的解读提到，V4 的核心是面向长上下文推理降低 FLOPs 与 KV cache，而不是简单把位置编码拉长。

## 2. 市面上长上下文常见路线

过去几年，长上下文模型大致有几条路线。

第一类是 sliding window attention（滑动窗口注意力）。Longformer 用局部窗口加少量全局 token，让注意力复杂度从二次变成近似线性；Mistral 7B 也使用 sliding window attention（SWA）降低推理成本。这条路线便宜、好实现，但天然问题是：窗口外的信息看不到，远程 needle/passkey 检索会受损。

第二类是 sparse attention（稀疏注意力）。BigBird 用局部、随机、全局 token 的稀疏图结构，把全注意力的二次依赖降到线性量级，同时保留一定理论表达能力。DeepSeek Sparse Attention（DSA）可以理解为更细粒度、更工程化的稀疏选择：不是固定看哪些位置，而是由轻量 indexer（索引器）为每个 query 选择相关历史 token 或 block。

第三类是 memory/compression（记忆压缩）。比如 Infini-attention 把长期上下文压进 bounded memory（有界记忆），最近 token 仍用局部注意力；很多工程系统也会做摘要、检索、分层缓存。它们的共性是：不要把所有历史 token 原样放进主 attention。

DeepSeek-V4 的亮点在于把第二类和第三类结合得更紧：先在 token 维度压缩 K/V，再在压缩后的空间上做 sparse top-k（稀疏 top-k）选择。

## 3. DeepSeek-V4 的关键：CSA + HCA + DSA 思想

公开解读中，DeepSeek-V4 的长上下文注意力由 hybrid attention（混合注意力）构成，核心包括 CSA（Compressed Sparse Attention，压缩稀疏注意力）和 HCA（Heavily Compressed Attention，高压缩注意力）。

CSA 的直觉是：远程上下文不必每个 token 都原样参与注意力。可以先把每 4 个 token 的 K/V 压缩成 1 个 compressed KV entry（压缩 KV 条目），相当于序列维度缩短 4 倍。然后 lightning indexer（轻量索引器）在这些 compressed blocks（压缩块）上打分，为当前 query 选 top-k 个最相关 block。最后 query 只对这些 block 做注意力，并保留一个 sliding-window branch（滑动窗口分支）处理最近的未压缩 token。

HCA 则更激进，把远程 KV 压得更重，例如 128 倍压缩，然后直接对压缩后的短序列做 dense attention（稠密注意力）。V4 通过层间交替，让不同层承载不同注意力模式：有些层适合稀疏检索，有些层适合重压缩的全局摘要。

这背后的通解是：长上下文不是“全保真地看全部历史”，而是“把历史变成多级表示”。近处信息保真，远处信息压缩；重要远程信息通过 indexer 找回来；不重要信息只保留统计摘要。DeepSeek 的亮点是把这套结构放进模型内部，并配套低精度存储、专用 kernel 与训练，使它不是外挂 RAG，而是原生 attention 机制的一部分。

## 4. LiteKV demo 怎么做简单验证

本仓库 LiteKV 做的是一个机制级最小实验。它不加载 DeepSeek 权重，也不训练语言模型，只构造一个 synthetic retrieval case（合成检索样例）：目标 key 与 query 相似，目标 value 带有标记；如果注意力机制能找到远程目标，就说明它保留了远程检索信号。

核心入口在 `src/litekv/attention.py`：

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

四种模式共用同一个接口：

- `dense`：完整看历史，作为正确性和成本基线。
- `sliding_window`：只看最近窗口，成本低但远程目标不可达。
- `csa_lite`：每 4 个 token 均值压成 block，query 对 compressed key 打分，选 top-k block。
- `csa_lite_local`：远程用 compressed top-k，本地窗口保留未压缩 token。

实验由 `experiments/run_litekv.py` 触发：

```bash
source .venv/bin/activate
python experiments/run_litekv.py
```

它会生成 `results/metrics.csv`、`results/metrics.json` 和五张图。这里最重要的是理论 KV cache、理论 FLOPs 和 retrieval recall（检索召回），而不是 Python fallback 的本地 latency（延迟）。

## 5. Demo 结果：成本下降，但远程检索仍命中

默认实验在 512、1024、2048、4096 context 上比较四种模式。取 `top_k=32`、`local_window=128` 的代表 slice，4096 context 下结果如下：

```text
dense            KV=8,388,608   FLOPs=2,097,152   recall=1.0
sliding_window   KV=262,144     FLOPs=65,536      recall=0.0
csa_lite         KV=2,097,152   FLOPs=540,672     recall=1.0
csa_lite_local   KV=2,293,760   FLOPs=589,824     recall=1.0
```

这组数字很符合我们的验证猜想：

- dense 能找到目标，但 KV cache 和 FLOPs 最大。
- sliding window 成本最低，但远程目标在窗口外，recall 变成 0。
- csa_lite 把 KV cache 降到 dense 的约 1/4，FLOPs 也约下降 3.88 倍，同时 recall 仍为 1。
- csa_lite_local 额外保留本地未压缩 token，成本略高于纯 csa_lite，但仍比 dense 明显便宜，并保留远程检索能力。

下面几张图就是 demo 产物，可以直接作为技术分享中的案例图。

![KV cache vs context](results/kv_cache_vs_context.png)

![FLOPs vs context](results/flops_vs_context.png)

![Retrieval accuracy](results/retrieval_accuracy.png)

![Top-k tradeoff](results/topk_tradeoff.png)

## 6. 这能证明什么，不能证明什么

它能证明的是机制趋势：如果远程上下文能被压缩成 block，而且 query 能选中相关 block，那么注意力不必看全部历史 token，也能保留远程检索信号。这个趋势正是 CSA/DSA 类方法想解决的问题。

它不能证明三件事：第一，不能证明 DeepSeek-V4 官方 1M 上下文能力；第二，不能证明真实语言任务上的准确率；第三，不能用本 demo 的 latency 图代表生产 kernel 性能。当前实现故意保留纯 Python fallback，目的是可读、可测、可讲，不是追求高性能。

真正的 DeepSeek-V4 强在系统组合：模型内部原生压缩、稀疏 indexer、局部窗口、不同层的 CSA/HCA 交替、低精度 KV 存储，以及面向长链路 agent 的训练与基础设施。LiteKV 只是把其中“token 维度压缩 + sparse top-k + local window”的骨架拆出来，让我们能用几十行核心逻辑看懂它为什么可能有效。

## 7. 写技术文章时的主线

如果把这篇分享压成一句话，我建议是：

> DeepSeek-V4 的长上下文能力，不是把窗口硬拉到 1M，而是把历史上下文变成“压缩可检索的记忆层”：远处压缩、近处保真、重要信息稀疏取回。

文章结构可以这样展开：

1. 长上下文为什么贵：FLOPs 和 KV cache。
2. 传统路线的取舍：dense、sliding window、sparse attention、memory compression。
3. DeepSeek 的路线：DSA 到 CSA/HCA，把稀疏选择放在压缩后的 token/block 空间。
4. LiteKV demo：用合成 passkey case 验证机制。
5. 结果图：成本降低，远程检索不丢。
6. 边界：这不是官方复现，而是机制解释器。

## 参考链接

- 用户提供背景文章：[微信公众号原文](https://mp.weixin.qq.com/s/8bxXqS2R8Fx5-1TLDBiEDg)
- DeepSeek-V4 解读：[Hugging Face Blog: DeepSeek-V4](https://huggingface.co/blog/deepseekv4)
- DeepSeek Sparse Attention 公开仓库：[DeepSeek-V3.2-Exp GitHub](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- DeepSeek-V3.2 论文：[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)
- Longformer：[arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
- BigBird：[arXiv:2007.14062](https://arxiv.org/abs/2007.14062)
- Mistral 7B：[arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- Infini-attention：[arXiv:2404.07143](https://arxiv.org/abs/2404.07143)
