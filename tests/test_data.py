import unittest

from litekv.data import (
    TargetRegion,
    generate_retrieval_case,
    sliding_window_reaches_target,
    target_position_for_region,
)


class SyntheticRetrievalDataTest(unittest.TestCase):
    def test_generation_is_deterministic_for_same_seed(self):
        first = generate_retrieval_case(context_length=512, hidden_size=8, seed=7)
        second = generate_retrieval_case(context_length=512, hidden_size=8, seed=7)

        self.assertEqual(first.task, second.task)
        self.assertEqual(first.query, second.query)
        self.assertEqual(first.keys, second.keys)
        self.assertEqual(first.values, second.values)

    def test_remote_target_outside_sliding_window_is_unreachable(self):
        case = generate_retrieval_case(
            context_length=512,
            hidden_size=8,
            compression_ratio=4,
            local_window=128,
            target_region=TargetRegion.EARLY,
            seed=11,
        )

        self.assertFalse(case.task.reachable_by_sliding_window)
        self.assertFalse(
            sliding_window_reaches_target(
                target_position=case.task.target_position,
                query_position=case.task.query_position,
                local_window=128,
            )
        )

    def test_context_shorter_than_local_window_is_handled(self):
        case = generate_retrieval_case(
            context_length=64,
            hidden_size=8,
            compression_ratio=4,
            local_window=128,
            target_region=TargetRegion.MIDDLE,
            seed=13,
        )

        self.assertGreaterEqual(case.task.local_window_start, 0)
        self.assertTrue(case.task.reachable_by_sliding_window)

    def test_target_blocks_are_valid_at_context_edges(self):
        early = target_position_for_region(512, TargetRegion.EARLY)
        late = target_position_for_region(512, TargetRegion.LATE)

        early_case = generate_retrieval_case(
            context_length=512,
            hidden_size=8,
            compression_ratio=4,
            target_position=early,
        )
        late_case = generate_retrieval_case(
            context_length=512,
            hidden_size=8,
            compression_ratio=4,
            target_position=late,
        )

        self.assertEqual(early_case.task.target_block, early // 4)
        self.assertEqual(late_case.task.target_block, late // 4)
        self.assertGreaterEqual(early_case.task.target_block, 0)
        self.assertLess(late_case.task.target_block, 512 // 4)

    def test_rejects_non_divisible_context_and_compression_ratio(self):
        with self.assertRaises(ValueError):
            generate_retrieval_case(context_length=510, hidden_size=8, compression_ratio=4)


if __name__ == "__main__":
    unittest.main()
