import tempfile
import unittest
from pathlib import Path

from litekv.config import (
    ATTENTION_MODES,
    ExperimentConfig,
    load_config,
    resolve_device,
)


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


if __name__ == "__main__":
    unittest.main()
