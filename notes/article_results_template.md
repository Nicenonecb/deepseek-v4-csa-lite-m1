# LiteKV Article Results Template

Use this note after generating `results/metrics.csv`, `results/metrics.json`, and the plot PNGs.

## Safe Claims

- LiteKV is a synthetic attention-level demo for explaining long-context mechanisms.
- The KV cache and FLOPs plots compare theoretical accounting across dense attention, sliding window, CSA-lite, and CSA-lite + local window.
- The latency plot is local machine evidence only. Treat it as a sanity check, not a production throughput benchmark.
- Retrieval accuracy reflects constructed passkey-style cases, not general language-model quality.

## Non-Claims

- Do not claim this reproduces DeepSeek-V4.
- Do not claim the local M1 timing predicts datacenter inference performance.
- Do not claim synthetic retrieval recall proves real workload accuracy.

## Generated Figures

- `results/kv_cache_vs_context.png`: estimated KV cache bytes as context length grows.
- `results/flops_vs_context.png`: theoretical attention FLOPs as context length grows.
- `results/latency_vs_context.png`: local forward latency measured by the demo runner.
- `results/retrieval_accuracy.png`: retrieval recall on constructed synthetic cases.
- `results/topk_tradeoff.png`: top-k sweep trade-off when multiple top-k values are present.

## Suggested Wording

LiteKV does not prove a production model result. It makes the mechanism visible: compressing remote KV entries and selecting a small number of relevant blocks can reduce the amount of attention work while retaining a measurable retrieval signal in controlled synthetic cases.
