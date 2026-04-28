# LiteKV

LiteKV is a small demo for a DeepSeek-V4 technical article. Its goal is to make the long-context attention trade-off easy to run, measure, and explain on a local machine.

LiteKV is not a DeepSeek-V4 reproduction. It does not load official model weights, reproduce the full DSA/HCA/CSA stack, or prove official 1M-context behavior. It is an experimental microscope for mechanism-level claims: dense attention cost, sliding-window limitations, remote KV compression, sparse top-k block selection, and local-window compensation.

## Current Status

Unit 1 provides the project scaffold and configuration layer:

- `src/litekv/config.py` loads experiment config.
- `experiments/configs/default.yaml` defines M1-friendly defaults.
- `experiments/run_litekv.py` validates and prints the resolved config.
- `tests/test_experiment.py` covers config loading and validation.

Unit 2 adds deterministic synthetic retrieval cases:

- `src/litekv/data.py` creates passkey-style task metadata and constructed Q/K/V cases.
- `tests/test_data.py` verifies deterministic generation, sliding-window reachability, block ids, and validation errors.

Unit 3 adds comparable attention implementations and theoretical accounting:

- `src/litekv/attention.py` runs dense, sliding-window, CSA-lite, and CSA-lite + local-window attention over constructed cases.
- `src/litekv/metrics.py` reports shared metrics for KV entries, estimated KV bytes, attention score counts, FLOPs, retrieval hit/recall, and local latency.
- `tests/test_attention.py` and `tests/test_metrics.py` verify expected retrieval and accounting behavior.

Unit 4 adds the experiment data pipeline and result artifacts:

- `src/litekv/experiment.py` runs the configured matrix of contexts, modes, top-k values, and local windows.
- `experiments/run_litekv.py` writes `results/metrics.csv` and `results/metrics.json` when run without `--dry-run`.
- `tests/test_experiment.py` verifies smoke runs, deterministic rows, missing-directory creation, invalid mode handling, and artifact columns.

Later units will add plots and article-facing notes.

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

## Scope

Safe claim: LiteKV helps explain why token-level KV compression plus sparse block selection can reduce long-context attention cost.

Unsafe claim: LiteKV proves DeepSeek-V4's official 1M-context capability.
