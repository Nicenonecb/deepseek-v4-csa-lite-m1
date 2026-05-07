import json
import tempfile
import unittest
from pathlib import Path

from litekv.plots import PLOT_FILENAMES, generate_plots


REFERENCE_DOC = Path(__file__).resolve().parents[1] / "DeepSeek-V4注意力机制与LiteKV验证技术分享.md"


class PlotGenerationTest(unittest.TestCase):
    def test_generates_expected_plots_from_metrics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics_path.write_text(json.dumps(_metric_rows(top_k_values=[1, 2])))

            artifacts = generate_plots(metrics_path)

            self.assertEqual(
                sorted(path.name for path in artifacts.plot_paths),
                sorted(PLOT_FILENAMES),
            )
            self.assertEqual(artifacts.warnings, [])
            for path in artifacts.plot_paths:
                self.assertTrue(path.exists())
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_skips_topk_tradeoff_when_no_topk_sweep_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics_path.write_text(json.dumps(_metric_rows(top_k_values=[1])))

            artifacts = generate_plots(metrics_path)

            self.assertNotIn("topk_tradeoff.png", [path.name for path in artifacts.plot_paths])
            self.assertEqual(len(artifacts.plot_paths), 5)
            self.assertEqual(
                artifacts.warnings,
                ["Skipped topk_tradeoff.png because metrics include no top-k sweep."],
            )

    def test_empty_metrics_input_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics_path.write_text("[]")

            with self.assertRaisesRegex(ValueError, "metrics input is empty"):
                generate_plots(metrics_path)

    def test_plot_filenames_match_article_plan(self):
        reference_text = REFERENCE_DOC.read_text()

        for filename in PLOT_FILENAMES:
            self.assertIn(filename, reference_text)


def _metric_rows(top_k_values):
    rows = []
    for context_length in [16, 32]:
        for top_k in top_k_values:
            for mode in ["dense", "csa_lite"]:
                multiplier = 1 if mode == "dense" else 0.5
                rows.append(
                    {
                        "mode": mode,
                        "context_length": context_length,
                        "top_k": top_k,
                        "local_window": 4,
                        "estimated_kv_bytes": int(context_length * 64 * multiplier),
                        "estimated_attention_flops": int(context_length * top_k * 10 * multiplier),
                        "forward_latency_ms": float(context_length) / 10.0 * multiplier,
                        "retrieval_recall": 1.0 if mode == "dense" else float(top_k) / max(top_k_values),
                        "answer_signal": multiplier * float(top_k) / max(top_k_values),
                        "retrieved_signal": multiplier,
                    }
                )
    return rows


if __name__ == "__main__":
    unittest.main()
