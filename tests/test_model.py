import unittest

from litekv.model import ToyDecoderBlock, ToyDecoderConfig


class ToyDecoderBlockTest(unittest.TestCase):
    def test_dense_forward_returns_output_for_each_token(self):
        block = ToyDecoderBlock(ToyDecoderConfig(attention_mode="dense", local_window=2))
        sequence = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]

        output = block.forward(sequence)

        self.assertEqual(len(output.hidden_states), 3)
        self.assertEqual([len(vector) for vector in output.hidden_states], [2, 2, 2])
        self.assertEqual([metric.mode for metric in output.attention_metrics], ["dense", "dense", "dense"])

    def test_can_swap_to_csa_lite_mode_through_config(self):
        block = ToyDecoderBlock(
            ToyDecoderConfig(
                attention_mode="csa_lite",
                compression_ratio=2,
                top_k=1,
                local_window=2,
            )
        )
        sequence = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.25, 0.75]]

        output = block.forward(sequence)

        self.assertEqual(len(output.hidden_states), 4)
        self.assertTrue(all(metric.mode == "csa_lite" for metric in output.attention_metrics))
        self.assertTrue(all(metric.selected_block_count <= 1 for metric in output.attention_metrics))

    def test_sequence_shorter_than_local_window_returns_valid_shape(self):
        block = ToyDecoderBlock(ToyDecoderConfig(attention_mode="sliding_window", local_window=16))
        sequence = [[1.0, 0.0], [0.0, 1.0]]

        output = block.forward(sequence)

        self.assertEqual(len(output.hidden_states), 2)
        self.assertEqual([len(vector) for vector in output.hidden_states], [2, 2])
        self.assertEqual(output.attention_metrics[-1].local_token_count, 2)

    def test_unsupported_attention_mode_uses_attention_validation(self):
        block = ToyDecoderBlock(ToyDecoderConfig(attention_mode="mystery"))

        with self.assertRaisesRegex(ValueError, "Unknown attention mode"):
            block.forward([[1.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
