import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PLOT_FILENAMES = (
    "kv_cache_vs_context.png",
    "flops_vs_context.png",
    "latency_vs_context.png",
    "retrieval_accuracy.png",
    "topk_tradeoff.png",
)

REQUIRED_METRIC_FIELDS = (
    "mode",
    "context_length",
    "top_k",
    "local_window",
    "estimated_kv_bytes",
    "estimated_attention_flops",
    "forward_latency_ms",
    "retrieval_recall",
)


@dataclass(frozen=True)
class PlotArtifacts:
    output_dir: Path
    plot_paths: List[Path]
    warnings: List[str]


def generate_plots(metrics_path: Path, output_dir: Path = None) -> PlotArtifacts:
    metrics_path = Path(metrics_path)
    rows = _load_metric_rows(metrics_path)
    if not rows:
        raise ValueError("metrics input is empty")
    _validate_rows(rows)

    output = Path(output_dir) if output_dir is not None else metrics_path.parent
    output.mkdir(parents=True, exist_ok=True)

    plt = _load_pyplot()
    representative_rows = _representative_rows(rows)
    plot_paths = [
        _plot_by_context(
            plt,
            representative_rows,
            output / "kv_cache_vs_context.png",
            metric="estimated_kv_bytes",
            ylabel="Estimated KV cache bytes",
            title="LiteKV synthetic demo: KV cache vs context",
        ),
        _plot_by_context(
            plt,
            representative_rows,
            output / "flops_vs_context.png",
            metric="estimated_attention_flops",
            ylabel="Estimated attention FLOPs",
            title="LiteKV synthetic demo: theoretical FLOPs vs context",
        ),
        _plot_by_context(
            plt,
            representative_rows,
            output / "latency_vs_context.png",
            metric="forward_latency_ms",
            ylabel="Forward latency (ms, local machine)",
            title="LiteKV synthetic demo: local M1 timing vs context",
        ),
        _plot_by_context(
            plt,
            representative_rows,
            output / "retrieval_accuracy.png",
            metric="retrieval_recall",
            ylabel="Retrieval recall",
            title="LiteKV synthetic demo: retrieval recall vs context",
        ),
    ]

    topk_path, warning = _plot_topk_tradeoff(plt, rows, output / "topk_tradeoff.png")
    warnings = []
    if topk_path is not None:
        plot_paths.append(topk_path)
    if warning is not None:
        warnings.append(warning)

    return PlotArtifacts(output_dir=output, plot_paths=plot_paths, warnings=warnings)


def _load_metric_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".json":
        with path.open() as input_file:
            return [dict(row) for row in json.load(input_file)]
    if path.suffix == ".csv":
        with path.open(newline="") as input_file:
            return [_coerce_csv_row(row) for row in csv.DictReader(input_file)]
    raise ValueError("Unsupported metrics file type: {}".format(path.suffix))


def _coerce_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    coerced: Dict[str, Any] = dict(row)
    for field in (
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
        "target_position",
        "target_block",
        "compressed_entry_count",
        "local_token_count",
    ):
        if field in coerced and coerced[field] != "":
            coerced[field] = int(coerced[field])
    for field in ("forward_latency_ms", "retrieval_recall"):
        if field in coerced and coerced[field] != "":
            coerced[field] = float(coerced[field])
    return coerced


def _validate_rows(rows: Sequence[Dict[str, Any]]) -> None:
    missing = sorted(
        field
        for field in REQUIRED_METRIC_FIELDS
        if any(field not in row for row in rows)
    )
    if missing:
        raise ValueError("Missing metric fields: {}".format(", ".join(missing)))


def _representative_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_mode_context: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["mode"]), int(row["context_length"]))
        existing = by_mode_context.get(key)
        if existing is None or _slice_key(row) < _slice_key(existing):
            by_mode_context[key] = row
    return [by_mode_context[key] for key in sorted(by_mode_context)]


def _slice_key(row: Dict[str, Any]) -> Tuple[int, int]:
    return (int(row["top_k"]), int(row["local_window"]))


def _plot_by_context(plt: Any, rows: Sequence[Dict[str, Any]], path: Path, metric: str, ylabel: str, title: str) -> Path:
    _new_figure(plt)
    for mode in _sorted_modes(rows):
        mode_rows = sorted((row for row in rows if row["mode"] == mode), key=lambda row: int(row["context_length"]))
        plt.plot(
            [int(row["context_length"]) for row in mode_rows],
            [float(row[metric]) for row in mode_rows],
            marker="o",
            label=mode,
        )
    _finish_plot(plt, title=title, ylabel=ylabel)
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _plot_topk_tradeoff(plt: Any, rows: Sequence[Dict[str, Any]], path: Path) -> Tuple[Optional[Path], Optional[str]]:
    sweep_groups = _topk_sweep_groups(rows)
    if not sweep_groups:
        return None, "Skipped topk_tradeoff.png because metrics include no top-k sweep."

    figure, recall_axis = plt.subplots(figsize=(7, 4.5))
    flops_axis = recall_axis.twinx()
    for label, group_rows in sweep_groups:
        sorted_rows = sorted(group_rows, key=lambda row: int(row["top_k"]))
        recall_axis.plot(
            [int(row["top_k"]) for row in sorted_rows],
            [float(row["retrieval_recall"]) for row in sorted_rows],
            marker="o",
            label="{} recall".format(label),
        )
        flops_axis.plot(
            [int(row["top_k"]) for row in sorted_rows],
            [float(row["estimated_attention_flops"]) for row in sorted_rows],
            linestyle="--",
            marker="x",
            alpha=0.65,
            label="{} FLOPs".format(label),
        )
    recall_axis.set_title("LiteKV synthetic demo: top-k cost/recall trade-off")
    recall_axis.set_xlabel("top-k selected blocks")
    recall_axis.set_ylabel("Retrieval recall")
    flops_axis.set_ylabel("Estimated attention FLOPs")
    recall_axis.grid(True, alpha=0.25)
    handles, labels = recall_axis.get_legend_handles_labels()
    flops_handles, flops_labels = flops_axis.get_legend_handles_labels()
    recall_axis.legend(handles + flops_handles, labels + flops_labels, fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path, None


def _topk_sweep_groups(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    groups: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        if str(row["mode"]) not in {"csa_lite", "csa_lite_local"}:
            continue
        key = (str(row["mode"]), int(row["context_length"]), int(row["local_window"]))
        groups.setdefault(key, []).append(row)

    sweep_groups = []
    for key, group_rows in sorted(groups.items()):
        top_k_values = {int(row["top_k"]) for row in group_rows}
        if len(top_k_values) > 1:
            mode, context_length, local_window = key
            label = "{} / ctx {} / local {}".format(mode, context_length, local_window)
            sweep_groups.append((label, group_rows))
    return sweep_groups


def _sorted_modes(rows: Iterable[Dict[str, Any]]) -> List[str]:
    return sorted({str(row["mode"]) for row in rows})


def _new_figure(plt: Any) -> None:
    plt.figure(figsize=(7, 4.5))


def _finish_plot(plt: Any, title: str, ylabel: str, xlabel: str = "Context length") -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
