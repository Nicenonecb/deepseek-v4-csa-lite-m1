import unittest

from litekv.attention import run_attention
from litekv.config import ATTENTION_MODES
from litekv.data import TargetRegion, generate_retrieval_case


class AttentionImplementationTest(unittest.TestCase):
    def test_dense_attention_retrieves_constructed_target(self):
        case = generate_retrieval_case(
            context_length=64,
            hidden_size=8,
            compression_ratio=4,
            local_window=8,
            target_region=TargetRegion.EARLY,
            seed=1,
        )

        result = run_attention(case, "dense")

        self.assertEqual(result.metrics.retrieved_position, case.task.target_position)
        self.assertTrue(result.metrics.retrieval_hit)
        self.assertEqual(result.metrics.kv_entries, case.task.query_position + 1)
        self.assertEqual(result.metrics.attention_score_count, case.task.query_position + 1)

    def test_sliding_window_misses_remote_target_outside_window(self):
        case = generate_retrieval_case(
            context_length=64,
            hidden_size=8,
            compression_ratio=4,
            local_window=8,
            target_region=TargetRegion.EARLY,
            seed=2,
        )

        result = run_attention(case, "sliding_window", local_window=8)

        self.assertNotIn(case.task.target_position, result.metrics.attended_positions)
        self.assertFalse(result.metrics.retrieval_hit)
        self.assertEqual(result.metrics.kv_entries, 8)

    def test_csa_lite_selects_target_block(self):
        case = generate_retrieval_case(
            context_length=64,
            hidden_size=8,
            compression_ratio=4,
            local_window=8,
            target_region=TargetRegion.EARLY,
            seed=3,
        )

        result = run_attention(case, "csa_lite", top_k=1, compression_ratio=4)

        self.assertIn(case.task.target_block, result.metrics.selected_blocks)
        self.assertTrue(result.metrics.retrieval_hit)
        self.assertEqual(result.metrics.selected_block_count, 1)
        self.assertEqual(result.metrics.kv_entries, 16)
        self.assertGreater(result.metrics.compressed_entry_count, result.metrics.selected_block_count)

    def test_csa_lite_caps_top_k_to_available_blocks(self):
        case = generate_retrieval_case(
            context_length=16,
            hidden_size=8,
            compression_ratio=4,
            local_window=4,
            target_region=TargetRegion.EARLY,
            seed=4,
        )

        result = run_attention(case, "csa_lite", top_k=99, compression_ratio=4)

        self.assertEqual(result.metrics.selected_block_count, 4)
        self.assertEqual(sorted(result.metrics.selected_blocks), [0, 1, 2, 3])

    def test_csa_lite_local_accounts_for_remote_blocks_and_local_tokens(self):
        case = generate_retrieval_case(
            context_length=64,
            hidden_size=8,
            compression_ratio=4,
            local_window=8,
            target_region=TargetRegion.EARLY,
            seed=5,
        )

        result = run_attention(case, "csa_lite_local", top_k=1, compression_ratio=4, local_window=8)

        self.assertIn(case.task.target_block, result.metrics.selected_blocks)
        self.assertEqual(result.metrics.local_token_count, 8)
        self.assertEqual(result.metrics.kv_entries, result.metrics.compressed_entry_count + 8)
        self.assertTrue(result.metrics.retrieval_hit)

    def test_local_window_larger_than_context_behaves_like_full_local_coverage(self):
        case = generate_retrieval_case(
            context_length=16,
            hidden_size=8,
            compression_ratio=4,
            local_window=32,
            target_region=TargetRegion.MIDDLE,
            seed=6,
        )

        result = run_attention(case, "csa_lite_local", top_k=2, compression_ratio=4, local_window=32)

        self.assertEqual(result.metrics.local_token_count, 16)
        self.assertEqual(result.metrics.compressed_entry_count, 0)
        self.assertEqual(result.metrics.selected_block_count, 0)
        self.assertTrue(result.metrics.retrieval_hit)

    def test_unknown_attention_mode_raises_clear_error(self):
        case = generate_retrieval_case(context_length=16, hidden_size=8)

        with self.assertRaisesRegex(ValueError, "Unknown attention mode"):
            run_attention(case, "mystery")

    def test_all_modes_share_the_same_metric_schema(self):
        case = generate_retrieval_case(
            context_length=64,
            hidden_size=8,
            compression_ratio=4,
            local_window=8,
            target_region=TargetRegion.EARLY,
            seed=7,
        )

        schemas = [
            set(run_attention(case, mode, top_k=2, compression_ratio=4, local_window=8).metrics.as_dict())
            for mode in ATTENTION_MODES
        ]

        self.assertTrue(all(schema == schemas[0] for schema in schemas))


if __name__ == "__main__":
    unittest.main()
