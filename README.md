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

Later units will add attention implementations, metrics, experiment outputs, and plots.

## Quick Check

```bash
PYTHONPATH=src python3 -m unittest tests/test_experiment.py
PYTHONPATH=src python3 -m unittest tests/test_data.py
PYTHONPATH=src python3 experiments/run_litekv.py --dry-run
```

## Scope

Safe claim: LiteKV helps explain why token-level KV compression plus sparse block selection can reduce long-context attention cost.

Unsafe claim: LiteKV proves DeepSeek-V4's official 1M-context capability.
