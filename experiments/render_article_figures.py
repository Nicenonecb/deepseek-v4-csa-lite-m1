#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    _render_attention_bridge(RESULTS / "attention_architecture_bridge.png")
    _render_active_kv_projection(RESULTS / "active_kv_projection.png")
    return 0


def _render_attention_bridge(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    _box(ax, 0.4, 5.8, 3.2, 0.7, "Native Sparse Attention", "#234b6d", "white", 14)
    _box(ax, 4.4, 5.8, 3.2, 0.7, "DeepSeek-V4 CSA", "#2f6f5e", "white", 14)
    _box(ax, 8.4, 5.8, 3.2, 0.7, "DeepSeek-V4 HCA", "#7a5b20", "white", 14)

    nsa = [
        ("Compressed attention", "coarse block memory"),
        ("Selected attention", "top-n contiguous raw blocks"),
        ("Sliding attention", "recent token fidelity"),
        ("Gated merge", "branch specialization"),
    ]
    csa = [
        ("Token-level compressor", "m tokens -> 1 compressed KV"),
        ("Lightning indexer", "query-aware top-k compressed entries"),
        ("Core MQA", "selected compressed KV + window KV"),
        ("Sliding window", "restore local causality"),
    ]
    hca = [
        ("Heavier compressor", "m' >> m tokens -> 1 KV"),
        ("Dense compressed attention", "global coarse summary"),
        ("Sliding window", "local fine-grained tokens"),
        ("Hybrid interleave", "CSA precision + HCA reach"),
    ]

    for x, items, color in ((0.4, nsa, "#d9e8f5"), (4.4, csa, "#dcefe8"), (8.4, hca, "#f1e6ca")):
        y = 4.75
        for title, subtitle in items:
            _box(ax, x, y, 3.2, 0.72, title + "\n" + subtitle, color, "#1f2933", 10)
            y -= 1.02

    for x1, x2 in ((3.75, 4.25), (7.75, 8.25)):
        ax.annotate(
            "",
            xy=(x2, 3.45),
            xytext=(x1, 3.45),
            arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#56616b"},
        )

    ax.text(
        6,
        0.45,
        "LiteKV demo maps this family into runnable toy modes: dense, sliding_window, csa_lite, csa_lite_local, nsa_lite.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_active_kv_projection(path: Path) -> None:
    contexts = [512, 1024, 2048, 4096, 8192, 16384]
    m = 4
    heavy_m = 128
    top_k = 32
    window = 128

    series = {
        "dense": [length for length in contexts],
        "sliding_window": [min(length, window) for length in contexts],
        "csa_lite_local": [max(0, length - window) / m + window for length in contexts],
        "nsa_lite": [max(0, length - window) / m + top_k * m + window for length in contexts],
        "hca_lite_formula": [max(0, length - window) / heavy_m + window for length in contexts],
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for label, values in series.items():
        ax.plot(contexts, values, marker="o", linewidth=2, label=label)
    ax.set_title("Active KV entries per decoding step (formula projection)")
    ax.set_xlabel("Context length")
    ax.set_ylabel("Active KV entries")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.text(
        0.02,
        0.95,
        "m=4, heavy m'=128, top-k=32, local window=128",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _box(ax, x, y, width, height, text, facecolor, textcolor, fontsize):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#4b5563",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        linespacing=1.25,
    )


if __name__ == "__main__":
    raise SystemExit(main())
