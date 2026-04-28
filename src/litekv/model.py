from dataclasses import dataclass
from typing import List

from litekv.attention import run_attention
from litekv.data import (
    Matrix,
    RetrievalCase,
    SyntheticTask,
    Vector,
    sliding_window_reaches_target,
    sliding_window_start,
)
from litekv.metrics import AttentionMetrics


@dataclass(frozen=True)
class ToyDecoderConfig:
    attention_mode: str = "dense"
    compression_ratio: int = 4
    top_k: int = 1
    local_window: int = 4


@dataclass(frozen=True)
class ToyDecoderOutput:
    hidden_states: Matrix
    attention_metrics: List[AttentionMetrics]


class ToyDecoderBlock:
    def __init__(self, config: ToyDecoderConfig = None):
        self.config = config or ToyDecoderConfig()

    def forward(self, hidden_states: Matrix) -> ToyDecoderOutput:
        _validate_hidden_states(hidden_states)

        outputs: Matrix = []
        metrics: List[AttentionMetrics] = []
        for query_position, query in enumerate(hidden_states):
            case = _decoder_case(
                hidden_states=hidden_states,
                query=query,
                query_position=query_position,
                config=self.config,
            )
            result = run_attention(
                case,
                self.config.attention_mode,
                top_k=self.config.top_k,
                compression_ratio=self.config.compression_ratio,
                local_window=self.config.local_window,
                measure_latency=False,
            )
            outputs.append(result.output)
            metrics.append(result.metrics)

        return ToyDecoderOutput(hidden_states=outputs, attention_metrics=metrics)


def _validate_hidden_states(hidden_states: Matrix) -> None:
    if not hidden_states:
        raise ValueError("hidden_states must not be empty")
    hidden_size = len(hidden_states[0])
    if hidden_size == 0:
        raise ValueError("hidden vectors must not be empty")
    for vector in hidden_states:
        if len(vector) != hidden_size:
            raise ValueError("all hidden vectors must share hidden size")


def _decoder_case(
    hidden_states: Matrix,
    query: Vector,
    query_position: int,
    config: ToyDecoderConfig,
) -> RetrievalCase:
    target_position = max(0, query_position - 1)
    local_start = sliding_window_start(query_position, config.local_window)
    task = SyntheticTask(
        context_length=len(hidden_states),
        hidden_size=len(query),
        compression_ratio=config.compression_ratio,
        local_window=config.local_window,
        query_position=query_position,
        target_position=target_position,
        target_block=target_position // config.compression_ratio,
        local_window_start=local_start,
        reachable_by_sliding_window=sliding_window_reaches_target(
            target_position=target_position,
            query_position=query_position,
            local_window=config.local_window,
        ),
        passkey="TOY-{:04d}".format(target_position),
        prompt="Toy decoder query at token {} attends over prior states.".format(query_position),
    )
    return RetrievalCase(
        task=task,
        query=list(query),
        keys=[list(vector) for vector in hidden_states],
        values=[list(vector) for vector in hidden_states],
    )
