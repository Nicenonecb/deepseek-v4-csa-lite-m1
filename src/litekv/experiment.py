import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from litekv.attention import run_attention
from litekv.config import ATTENTION_MODES, ExperimentConfig
from litekv.data import generate_retrieval_case


METRICS_CSV = "metrics.csv"
METRICS_JSON = "metrics.json"

METRIC_FIELDNAMES = [
    "run_timestamp",
    "seed",
    "device",
    "resolved_device",
    "attention_heads",
    "batch_size",
    "mode",
    "context_length",
    "hidden_size",
    "compression_ratio",
    "top_k",
    "local_window",
    "kv_entries",
    "estimated_kv_bytes",
    "attention_score_count",
    "estimated_attention_flops",
    "selected_block_count",
    "retrieval_hit",
    "retrieval_recall",
    "forward_latency_ms",
    "target_position",
    "target_block",
    "selected_blocks",
    "attended_positions",
    "retrieved_position",
    "retrieved_block",
    "compressed_entry_count",
    "local_token_count",
]


@dataclass(frozen=True)
class ExperimentArtifacts:
    output_dir: Path
    csv_path: Path
    json_path: Path
    rows: List[Dict[str, Any]]


def run_experiment(
    config: ExperimentConfig,
    run_timestamp: Optional[str] = None,
    measure_latency: bool = True,
) -> ExperimentArtifacts:
    _validate_attention_modes(config.attention_modes)
    timestamp = run_timestamp or _utc_timestamp()
    rows = list(_build_rows(config, timestamp, measure_latency=measure_latency))

    output_dir = Path(config.output_dir)
    csv_path = output_dir / METRICS_CSV
    json_path = output_dir / METRICS_JSON

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)

    return ExperimentArtifacts(
        output_dir=output_dir,
        csv_path=csv_path,
        json_path=json_path,
        rows=rows,
    )


def _validate_attention_modes(modes: Iterable[str]) -> None:
    unknown_modes = sorted(set(modes) - set(ATTENTION_MODES))
    if unknown_modes:
        raise ValueError("Unknown attention modes: {}".format(", ".join(unknown_modes)))


def _build_rows(
    config: ExperimentConfig,
    run_timestamp: str,
    measure_latency: bool,
) -> Iterable[Dict[str, Any]]:
    resolved_device = config.resolved_device()
    for context_length in config.context_lengths:
        for local_window in config.local_window_values:
            case = generate_retrieval_case(
                context_length=context_length,
                hidden_size=config.hidden_size,
                compression_ratio=config.compression_ratio,
                local_window=local_window,
                seed=config.seed,
            )
            for top_k in config.top_k_values:
                for mode in config.attention_modes:
                    result = run_attention(
                        case,
                        mode,
                        top_k=top_k,
                        compression_ratio=config.compression_ratio,
                        local_window=local_window,
                        measure_latency=measure_latency,
                    )
                    metrics = result.metrics.as_dict()
                    yield {
                        "run_timestamp": run_timestamp,
                        "seed": config.seed,
                        "device": config.device,
                        "resolved_device": resolved_device,
                        "attention_heads": config.attention_heads,
                        "batch_size": config.batch_size,
                        **metrics,
                    }


def _serialize_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    serialized = dict(row)
    serialized["selected_blocks"] = json.dumps(serialized["selected_blocks"])
    serialized["attended_positions"] = json.dumps(serialized["attended_positions"])
    return serialized


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_serialize_csv_row(row) for row in rows)
    temp_path.replace(path)


def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w") as output:
        json.dump(rows, output, indent=2, sort_keys=True)
        output.write("\n")
    temp_path.replace(path)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
