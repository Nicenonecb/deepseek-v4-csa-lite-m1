import csv
import json
import tempfile
import unittest
from pathlib import Path

from litekv.config import (
    ATTENTION_MODES,
    ExperimentConfig,
    load_config,
    resolve_device,
)
from litekv.experiment import METRIC_FIELDNAMES, run_experiment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "default.yaml"


class ExperimentConfigTest(unittest.TestCase):
    def test_loads_default_config(self):
        config = load_config(DEFAULT_CONFIG)

        self.assertEqual(config.context_lengths, [512, 1024, 2048, 4096])
        self.assertEqual(config.attention_modes, list(ATTENTION_MODES))
        self.assertEqual(config.output_dir, Path("results"))

    def test_output_dir_does_not_need_to_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "missing-results"
            config = ExperimentConfig(output_dir=output_dir)

        self.assertEqual(config.output_dir, output_dir)

    def test_mps_request_falls_back_when_unavailable(self):
        device = resolve_device("mps")

        self.assertIn(device, {"mps", "cpu"})

    def test_rejects_invalid_compression_ratio(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(compression_ratio=0)


class ExperimentRunnerTest(unittest.TestCase):
    def test_smoke_experiment_writes_csv_and_json_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _smoke_config(
                output_dir=Path(temp_dir) / "results",
                context_lengths=[16, 32],
                attention_modes=["dense", "csa_lite"],
            )

            artifacts = run_experiment(config, run_timestamp="2026-04-28T00:00:00+00:00", measure_latency=False)

            self.assertEqual(len(artifacts.rows), 4)
            self.assertTrue(artifacts.csv_path.exists())
            self.assertTrue(artifacts.json_path.exists())

            with artifacts.csv_path.open(newline="") as input_file:
                csv_rows = list(csv.DictReader(input_file))
            with artifacts.json_path.open() as input_file:
                json_rows = json.load(input_file)

            self.assertEqual(len(csv_rows), 4)
            self.assertEqual(len(json_rows), 4)
            self.assertEqual(list(csv_rows[0].keys()), METRIC_FIELDNAMES)
            self.assertIn("estimated_kv_bytes", json_rows[0])
            self.assertIn("forward_latency_ms", json_rows[0])
            self.assertIsInstance(json_rows[0]["selected_blocks"], list)

    def test_repeated_runs_with_same_seed_are_deterministic_without_latency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = _smoke_config(output_dir=Path(temp_dir) / "first")
            second = _smoke_config(output_dir=Path(temp_dir) / "second")

            first_rows = run_experiment(first, run_timestamp="first", measure_latency=False).rows
            second_rows = run_experiment(second, run_timestamp="second", measure_latency=False).rows

            self.assertEqual(
                [_without_timestamp(row) for row in first_rows],
                [_without_timestamp(row) for row in second_rows],
            )

    def test_missing_results_directory_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "missing" / "results"
            config = _smoke_config(output_dir=output_dir)

            self.assertFalse(output_dir.exists())
            run_experiment(config, measure_latency=False)

            self.assertTrue((output_dir / "metrics.csv").exists())
            self.assertTrue((output_dir / "metrics.json").exists())

    def test_invalid_mode_fails_before_writing_partial_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            config = _smoke_config(output_dir=output_dir)
            config.attention_modes = ["dense", "mystery"]

            with self.assertRaisesRegex(ValueError, "Unknown attention modes"):
                run_experiment(config, measure_latency=False)

            self.assertFalse(output_dir.exists())

    def test_rows_cover_every_configured_mode_and_context_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _smoke_config(
                output_dir=Path(temp_dir) / "results",
                context_lengths=[16, 32],
                attention_modes=list(ATTENTION_MODES),
                top_k_values=[1, 2],
                local_window_values=[4, 8],
            )

            artifacts = run_experiment(config, measure_latency=False)

            self.assertEqual(len(artifacts.rows), 32)
            for row in artifacts.rows:
                self.assertIn(row["mode"], ATTENTION_MODES)
                self.assertIn(row["context_length"], [16, 32])
                self.assertIn(row["top_k"], [1, 2])
                self.assertIn(row["local_window"], [4, 8])
                self.assertIn("attention_score_count", row)
                self.assertIn("estimated_attention_flops", row)
                self.assertIn("forward_latency_ms", row)


def _smoke_config(
    output_dir: Path,
    context_lengths=None,
    attention_modes=None,
    top_k_values=None,
    local_window_values=None,
) -> ExperimentConfig:
    return ExperimentConfig(
        context_lengths=context_lengths or [16],
        hidden_size=8,
        attention_heads=2,
        batch_size=1,
        compression_ratio=4,
        top_k_values=top_k_values or [1],
        local_window_values=local_window_values or [4],
        seed=7,
        device="cpu",
        output_dir=output_dir,
        attention_modes=attention_modes or ["dense"],
    )


def _without_timestamp(row):
    without_timestamp = dict(row)
    without_timestamp.pop("run_timestamp")
    return without_timestamp


if __name__ == "__main__":
    unittest.main()
