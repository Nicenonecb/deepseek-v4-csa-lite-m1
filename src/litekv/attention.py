from dataclasses import dataclass
from math import exp
from typing import List, Optional, Sequence, Tuple

from litekv.config import ATTENTION_MODES
from litekv.data import RetrievalCase, Vector
from litekv.metrics import AttentionMetrics, estimate_attention_flops, estimate_kv_bytes
from litekv.timing import timed_call


@dataclass(frozen=True)
class AttentionResult:
    output: Vector
    metrics: AttentionMetrics


@dataclass(frozen=True)
class _Candidate:
    key: Vector
    value: Vector
    position: Optional[int] = None
    block: Optional[int] = None


def run_attention(
    case: RetrievalCase,
    mode: str,
    top_k: int = 1,
    compression_ratio: Optional[int] = None,
    local_window: Optional[int] = None,
    dtype_bytes: int = 4,
    measure_latency: bool = True,
) -> AttentionResult:
    if mode not in ATTENTION_MODES:
        raise ValueError("Unknown attention mode: {}".format(mode))

    ratio = case.task.compression_ratio if compression_ratio is None else compression_ratio
    window = case.task.local_window if local_window is None else local_window
    _validate_inputs(case, top_k=top_k, compression_ratio=ratio, local_window=window)

    def compute() -> Tuple[Vector, _MetricInputs]:
        if mode == "dense":
            return _dense_attention(case)
        if mode == "sliding_window":
            return _sliding_window_attention(case, window)
        if mode == "csa_lite":
            return _csa_lite_attention(case, top_k, ratio)
        if mode == "csa_lite_local":
            return _csa_lite_local_attention(case, top_k, ratio, window)
        raise ValueError("Unknown attention mode: {}".format(mode))

    if measure_latency:
        (output, inputs), latency_ms = timed_call(compute)
    else:
        output, inputs = compute()
        latency_ms = 0.0
    metrics = _build_metrics(
        case=case,
        mode=mode,
        top_k=top_k,
        compression_ratio=ratio,
        local_window=window,
        dtype_bytes=dtype_bytes,
        forward_latency_ms=latency_ms,
        inputs=inputs,
    )
    return AttentionResult(output=output, metrics=metrics)


def _validate_inputs(
    case: RetrievalCase,
    top_k: int,
    compression_ratio: int,
    local_window: int,
) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive")
    if local_window <= 0:
        raise ValueError("local_window must be positive")
    if len(case.query) == 0:
        raise ValueError("query must not be empty")
    if len(case.keys) != len(case.values):
        raise ValueError("keys and values must have the same length")
    for vector in case.keys + case.values:
        if len(vector) != len(case.query):
            raise ValueError("query, keys, and values must share hidden size")


@dataclass(frozen=True)
class _MetricInputs:
    kv_entries: int
    attention_score_count: int
    selected_blocks: List[int]
    attended_positions: List[int]
    retrieved_position: Optional[int]
    retrieved_block: Optional[int]
    compressed_entry_count: int
    local_token_count: int
    retrieval_hit: bool


def _dense_attention(case: RetrievalCase) -> Tuple[Vector, _MetricInputs]:
    positions = _causal_positions(case)
    candidates = [
        _Candidate(key=case.keys[position], value=case.values[position], position=position)
        for position in positions
    ]
    output, best = _attend(case.query, candidates)
    retrieved_position = best.position if best else None
    retrieval_hit = retrieved_position == case.task.target_position
    return output, _MetricInputs(
        kv_entries=len(candidates),
        attention_score_count=len(candidates),
        selected_blocks=[],
        attended_positions=positions,
        retrieved_position=retrieved_position,
        retrieved_block=None,
        compressed_entry_count=0,
        local_token_count=len(candidates),
        retrieval_hit=retrieval_hit,
    )


def _sliding_window_attention(
    case: RetrievalCase,
    local_window: int,
) -> Tuple[Vector, _MetricInputs]:
    start = max(0, case.task.query_position - local_window + 1)
    positions = list(range(start, case.task.query_position + 1))
    candidates = [
        _Candidate(key=case.keys[position], value=case.values[position], position=position)
        for position in positions
    ]
    output, best = _attend(case.query, candidates)
    retrieved_position = best.position if best else None
    retrieval_hit = retrieved_position == case.task.target_position
    return output, _MetricInputs(
        kv_entries=len(candidates),
        attention_score_count=len(candidates),
        selected_blocks=[],
        attended_positions=positions,
        retrieved_position=retrieved_position,
        retrieved_block=None,
        compressed_entry_count=0,
        local_token_count=len(candidates),
        retrieval_hit=retrieval_hit,
    )


def _csa_lite_attention(
    case: RetrievalCase,
    top_k: int,
    compression_ratio: int,
) -> Tuple[Vector, _MetricInputs]:
    blocks = _compressed_blocks(case, compression_ratio, end_position=case.task.query_position)
    selected = _select_blocks(case.query, blocks, top_k)
    candidates = [
        _Candidate(key=key, value=value, block=block_index)
        for block_index, key, value in selected
    ]
    output, best = _attend(case.query, candidates)
    selected_blocks = [block_index for block_index, _, _ in selected]
    retrieved_block = best.block if best else None
    retrieval_hit = case.task.target_block in selected_blocks
    return output, _MetricInputs(
        kv_entries=len(blocks),
        attention_score_count=len(blocks) + len(selected),
        selected_blocks=selected_blocks,
        attended_positions=[],
        retrieved_position=None,
        retrieved_block=retrieved_block,
        compressed_entry_count=len(blocks),
        local_token_count=0,
        retrieval_hit=retrieval_hit,
    )


def _csa_lite_local_attention(
    case: RetrievalCase,
    top_k: int,
    compression_ratio: int,
    local_window: int,
) -> Tuple[Vector, _MetricInputs]:
    local_start = max(0, case.task.query_position - local_window + 1)
    local_positions = list(range(local_start, case.task.query_position + 1))
    remote_end = local_start - 1
    remote_blocks = _compressed_blocks(case, compression_ratio, end_position=remote_end)
    selected_remote = _select_blocks(case.query, remote_blocks, top_k)

    candidates = [
        _Candidate(key=key, value=value, block=block_index)
        for block_index, key, value in selected_remote
    ]
    candidates.extend(
        _Candidate(
            key=case.keys[position],
            value=case.values[position],
            position=position,
            block=position // compression_ratio,
        )
        for position in local_positions
    )

    output, best = _attend(case.query, candidates)
    selected_blocks = [block_index for block_index, _, _ in selected_remote]
    target_in_local = case.task.target_position in local_positions
    retrieval_hit = target_in_local or case.task.target_block in selected_blocks
    return output, _MetricInputs(
        kv_entries=len(remote_blocks) + len(local_positions),
        attention_score_count=len(remote_blocks) + len(candidates),
        selected_blocks=selected_blocks,
        attended_positions=local_positions,
        retrieved_position=best.position if best and best.position is not None else None,
        retrieved_block=best.block if best else None,
        compressed_entry_count=len(remote_blocks),
        local_token_count=len(local_positions),
        retrieval_hit=retrieval_hit,
    )


def _causal_positions(case: RetrievalCase) -> List[int]:
    return list(range(0, case.task.query_position + 1))


def _compressed_blocks(
    case: RetrievalCase,
    compression_ratio: int,
    end_position: int,
) -> List[Tuple[int, Vector, Vector]]:
    if end_position < 0:
        return []

    blocks: List[Tuple[int, Vector, Vector]] = []
    for block_start in range(0, end_position + 1, compression_ratio):
        block_end = min(block_start + compression_ratio - 1, end_position)
        positions = list(range(block_start, block_end + 1))
        block_index = block_start // compression_ratio
        key = _mean_vectors([case.keys[position] for position in positions])
        value = _mean_vectors([case.values[position] for position in positions])
        blocks.append((block_index, key, value))
    return blocks


def _select_blocks(
    query: Vector,
    blocks: Sequence[Tuple[int, Vector, Vector]],
    top_k: int,
) -> List[Tuple[int, Vector, Vector]]:
    capped_top_k = min(top_k, len(blocks))
    ranked = sorted(blocks, key=lambda block: _dot(query, block[1]), reverse=True)
    return ranked[:capped_top_k]


def _attend(query: Vector, candidates: Sequence[_Candidate]) -> Tuple[Vector, Optional[_Candidate]]:
    if not candidates:
        return [0.0 for _ in query], None

    scores = [_dot(query, candidate.key) for candidate in candidates]
    weights = _softmax(scores)
    output = [0.0 for _ in query]
    for weight, candidate in zip(weights, candidates):
        for index, value in enumerate(candidate.value):
            output[index] += weight * value
    best_index = max(range(len(scores)), key=lambda index: scores[index])
    return output, candidates[best_index]


def _build_metrics(
    case: RetrievalCase,
    mode: str,
    top_k: int,
    compression_ratio: int,
    local_window: int,
    dtype_bytes: int,
    forward_latency_ms: float,
    inputs: _MetricInputs,
) -> AttentionMetrics:
    return AttentionMetrics(
        mode=mode,
        context_length=case.task.context_length,
        hidden_size=case.task.hidden_size,
        compression_ratio=compression_ratio,
        top_k=top_k,
        local_window=local_window,
        kv_entries=inputs.kv_entries,
        estimated_kv_bytes=estimate_kv_bytes(inputs.kv_entries, case.task.hidden_size, dtype_bytes),
        attention_score_count=inputs.attention_score_count,
        estimated_attention_flops=estimate_attention_flops(
            inputs.attention_score_count,
            case.task.hidden_size,
        ),
        selected_block_count=len(inputs.selected_blocks),
        retrieval_hit=inputs.retrieval_hit,
        retrieval_recall=1.0 if inputs.retrieval_hit else 0.0,
        forward_latency_ms=forward_latency_ms,
        target_position=case.task.target_position,
        target_block=case.task.target_block,
        selected_blocks=inputs.selected_blocks,
        attended_positions=inputs.attended_positions,
        retrieved_position=inputs.retrieved_position,
        retrieved_block=inputs.retrieved_block,
        compressed_entry_count=inputs.compressed_entry_count,
        local_token_count=inputs.local_token_count,
    )


def _dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right))


def _softmax(scores: Sequence[float]) -> List[float]:
    largest = max(scores)
    exps = [exp(score - largest) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def _mean_vectors(vectors: Sequence[Vector]) -> Vector:
    if not vectors:
        raise ValueError("vectors must not be empty")
    width = len(vectors[0])
    totals = [0.0 for _ in range(width)]
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    return [value / len(vectors) for value in totals]
