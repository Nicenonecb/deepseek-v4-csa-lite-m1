# LiteKV

LiteKV is a small demo for a DeepSeek-V4 technical article. Its goal is to make the long-context attention trade-off easy to run, measure, and explain on a local machine.

LiteKV is not a DeepSeek-V4 reproduction. It does not load official model weights, reproduce the full DSA/HCA/CSA stack, or prove official 1M-context behavior. It is an experimental microscope for mechanism-level claims: dense attention cost, sliding-window limitations, remote KV compression, sparse top-k block selection, and local-window compensation.


## Quick Check

Use the project virtual environment when running experiment dependencies:

```bash
uv venv .venv --python /usr/local/bin/python3.10
uv pip install -e '.[experiment,test]'
```

```bash
.venv/bin/python -m pytest -q
.venv/bin/python experiments/run_litekv.py --dry-run
.venv/bin/python experiments/run_litekv.py
```

The full run writes:

- `results/metrics.csv`
- `results/metrics.json`
- `results/kv_cache_vs_context.png`
- `results/flops_vs_context.png`
- `results/latency_vs_context.png`
- `results/retrieval_accuracy.png`
- `results/topk_tradeoff.png`

## Scope

Safe claim: LiteKV helps explain why token-level KV compression plus sparse block selection can reduce long-context attention cost.

Unsafe claim: LiteKV proves DeepSeek-V4's official 1M-context capability.
