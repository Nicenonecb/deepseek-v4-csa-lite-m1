from dataclasses import dataclass
from enum import Enum
import random
from typing import List, Optional


class TargetRegion(str, Enum):
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


Vector = List[float]
Matrix = List[Vector]


@dataclass(frozen=True)
class SyntheticTask:
    context_length: int
    hidden_size: int
    compression_ratio: int
    local_window: int
    query_position: int
    target_position: int
    target_block: int
    local_window_start: int
    reachable_by_sliding_window: bool
    passkey: str
    prompt: str


@dataclass(frozen=True)
class RetrievalCase:
    task: SyntheticTask
    query: Vector
    keys: Matrix
    values: Matrix


def target_position_for_region(context_length: int, region: TargetRegion) -> int:
    if context_length <= 1:
        raise ValueError("context_length must be greater than 1")

    # Key note: these fixed anchors make early/middle/late cases article-friendly.
    if region == TargetRegion.EARLY:
        return max(0, context_length // 8)
    if region == TargetRegion.MIDDLE:
        return context_length // 2
    if region == TargetRegion.LATE:
        return max(0, context_length - context_length // 8 - 2)
    raise ValueError("Unknown target region: {}".format(region))


def sliding_window_start(query_position: int, local_window: int) -> int:
    if local_window <= 0:
        raise ValueError("local_window must be positive")
    return max(0, query_position - local_window + 1)


def sliding_window_reaches_target(
    target_position: int,
    query_position: int,
    local_window: int,
) -> bool:
    start = sliding_window_start(query_position, local_window)
    return start <= target_position <= query_position


def _validate_case_inputs(
    context_length: int,
    hidden_size: int,
    compression_ratio: int,
    local_window: int,
) -> None:
    if context_length <= 1:
        raise ValueError("context_length must be greater than 1")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive")
    if local_window <= 0:
        raise ValueError("local_window must be positive")
    if context_length % compression_ratio != 0:
        raise ValueError("context_length must be divisible by compression_ratio")


def _background_vector(rng: random.Random, hidden_size: int, scale: float = 0.01) -> Vector:
    return [rng.uniform(-scale, scale) for _ in range(hidden_size)]


def _signal_vector(hidden_size: int) -> Vector:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    vector = [0.0 for _ in range(hidden_size)]
    vector[0] = 1.0
    return vector


def _value_vector(hidden_size: int, marker: float) -> Vector:
    vector = [0.0 for _ in range(hidden_size)]
    vector[0] = marker
    if hidden_size > 1:
        vector[1] = marker / 10.0
    return vector


def generate_retrieval_case(
    context_length: int,
    hidden_size: int,
    compression_ratio: int = 4,
    local_window: int = 128,
    target_region: TargetRegion = TargetRegion.EARLY,
    target_position: Optional[int] = None,
    query_position: Optional[int] = None,
    seed: int = 2026,
) -> RetrievalCase:
    _validate_case_inputs(context_length, hidden_size, compression_ratio, local_window)

    if query_position is None:
        query_position = context_length - 1
    if not 0 <= query_position < context_length:
        raise ValueError("query_position must be inside the context")

    if target_position is None:
        target_position = target_position_for_region(context_length, target_region)
    if not 0 <= target_position < query_position:
        raise ValueError("target_position must be before query_position")

    rng = random.Random(seed)
    query = _signal_vector(hidden_size)
    keys = [_background_vector(rng, hidden_size) for _ in range(context_length)]
    values = [_value_vector(hidden_size, 0.0) for _ in range(context_length)]

    # Key note: the target key mirrors the query, so attention scoring should find it.
    keys[target_position] = list(query)
    values[target_position] = _value_vector(hidden_size, 1.0)

    # Key note: the query position is also marked, but the expected remote answer remains target_position.
    keys[query_position] = _background_vector(rng, hidden_size)
    values[query_position] = _value_vector(hidden_size, -1.0)

    window_start = sliding_window_start(query_position, local_window)
    target_block = target_position // compression_ratio
    reachable = sliding_window_reaches_target(target_position, query_position, local_window)
    passkey = "PASSKEY-{:04d}".format(target_position)
    prompt = (
        "Needle passkey {} is placed at token {}. "
        "The query at token {} should retrieve block {}."
    ).format(passkey, target_position, query_position, target_block)

    task = SyntheticTask(
        context_length=context_length,
        hidden_size=hidden_size,
        compression_ratio=compression_ratio,
        local_window=local_window,
        query_position=query_position,
        target_position=target_position,
        target_block=target_block,
        local_window_start=window_start,
        reachable_by_sliding_window=reachable,
        passkey=passkey,
        prompt=prompt,
    )

    return RetrievalCase(task=task, query=query, keys=keys, values=values)
