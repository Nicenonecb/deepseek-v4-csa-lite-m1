import unittest

from litekv.metrics import estimate_attention_flops, estimate_kv_bytes


class AttentionMetricsTest(unittest.TestCase):
    def test_estimates_kv_bytes_for_keys_and_values(self):
        self.assertEqual(estimate_kv_bytes(kv_entries=8, hidden_size=4, dtype_bytes=4), 256)

    def test_estimates_attention_flops_for_dot_products(self):
        self.assertEqual(estimate_attention_flops(score_count=8, hidden_size=4), 64)

    def test_rejects_invalid_metric_inputs(self):
        with self.assertRaises(ValueError):
            estimate_kv_bytes(kv_entries=-1, hidden_size=4)

        with self.assertRaises(ValueError):
            estimate_attention_flops(score_count=1, hidden_size=0)


if __name__ == "__main__":
    unittest.main()
