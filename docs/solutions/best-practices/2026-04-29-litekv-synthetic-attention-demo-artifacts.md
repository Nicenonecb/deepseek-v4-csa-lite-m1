---
title: Build Auditable Synthetic Attention Demos With Metrics-First Artifacts
date: 2026-04-29
category: docs/solutions/best-practices
module: litekv
problem_type: best_practice
component: development_workflow
severity: medium
applies_when:
  - building synthetic attention demos for mechanism-level validation
  - generating reproducible experiment artifacts and article-ready plots
  - comparing fallback attention modes, CSA-lite, and local-window variants
tags: [litekv, synthetic-demo, attention, experiment-artifacts, plots, toy-decoder]
---

# Build Auditable Synthetic Attention Demos With Metrics-First Artifacts

## Context

LiteKV was built to explain long-context attention mechanisms, not to reproduce DeepSeek-V4 or benchmark production kernels. Units 3-6 added a pure Python fallback attention implementation, deterministic retrieval cases, metrics artifacts, plots, and a toy decoder wrapper. The final verified state had 33 tests passing.

Session history surfaced a few useful constraints: local `torch` was initially unavailable, system `python3` lacked `pytest`, JSON artifacts first stringified list fields, and the initial top-k tradeoff chart was unreadable because it plotted too many slices at once (session history).

## Guidance

Keep the mechanism inspectable and make the metrics the durable contract. `run_attention(...)` returns both the output vector and an `AttentionMetrics` object, so the experiment runner, tests, plots, and toy decoder all use the same path.

```python
result = run_attention(
    case,
    "csa_lite_local",
    top_k=top_k,
    compression_ratio=config.compression_ratio,
    local_window=local_window,
    measure_latency=measure_latency,
)
metrics = result.metrics.as_dict()
```

Use one shared interface for all comparable mechanisms. Dense attention, sliding window, CSA-lite, and CSA-lite + local-window dispatch through the same validation, metric-building, and timing code. This prevents plots from comparing unrelated implementations.

Record raw data before plotting. `experiment.py` writes `metrics.csv` and `metrics.json`; `plots.py` regenerates figures from those files instead of rerunning experiments. CSV serializes list fields for spreadsheets, while JSON keeps arrays as arrays for machine readers.

For multi-dimensional sweeps, plot representative slices deliberately. The top-k tradeoff chart now selects the largest context and representative local window, then separates recall from theoretical FLOPs in two subplots. This keeps the chart readable while preserving the stress-case comparison.

## Why This Matters

Synthetic demos are convincing only when readers can audit the mechanism. LiteKV exposes selected blocks, attended positions, compressed entry counts, KV entries, FLOP estimates, retrieval hit/recall, and local latency. That lets the article explain why a mechanism works instead of only showing a final score.

The separation also reduces drift. The same attention path powers tests, experiments, plots, and the toy decoder. If the mechanism changes, all downstream artifacts change through one controlled interface.

## When to Apply

- Use this pattern for article demos that validate a mechanism rather than production performance.
- Use it when comparing attention, cache, routing, compression, or retrieval strategies.
- Use it when a project needs CPU-friendly or pure-Python fallbacks for tests.
- Avoid using local fallback latency as a production throughput claim.

## Examples

The CSA-lite + local path combines compressed remote memory with exact local tokens:

```python
remote_blocks = _compressed_blocks(case, compression_ratio, end_position=remote_end)
selected_remote = _select_blocks(case.query, remote_blocks, top_k)

candidates = [
    _Candidate(key=key, value=value, block=block_index)
    for block_index, key, value in selected_remote
]
candidates.extend(
    _Candidate(key=case.keys[position], value=case.values[position], position=position)
    for position in local_positions
)
```

The toy decoder wrapper demonstrates integration without duplicating attention logic:

```python
result = run_attention(
    case,
    self.config.attention_mode,
    top_k=self.config.top_k,
    compression_ratio=self.config.compression_ratio,
    local_window=self.config.local_window,
    measure_latency=False,
)
```

## Related

- `src/litekv/attention.py`
- `src/litekv/experiment.py`
- `src/litekv/plots.py`
- `src/litekv/model.py`
- `notes/article_results_template.md`
