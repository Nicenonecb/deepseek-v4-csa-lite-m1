from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_DTYPE_BYTES = 4


@dataclass(frozen=True)
class AttentionMetrics:
    mode: str
    context_length: int
    hidden_size: int
    compression_ratio: int
    top_k: int
    local_window: int
    kv_entries: int
    estimated_kv_bytes: int
    attention_score_count: int
    estimated_attention_flops: int
    selected_block_count: int
    retrieval_hit: bool
    retrieval_recall: float
    answer_signal: float
    retrieved_signal: float
    forward_latency_ms: float
    target_position: int
    target_block: int
    selected_blocks: List[int] = field(default_factory=list)
    attended_positions: List[int] = field(default_factory=list)
    retrieved_position: Optional[int] = None
    retrieved_block: Optional[int] = None
    compressed_entry_count: int = 0
    local_token_count: int = 0
    selected_token_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "context_length": self.context_length,
            "hidden_size": self.hidden_size,
            "compression_ratio": self.compression_ratio,
            "top_k": self.top_k,
            "local_window": self.local_window,
            "kv_entries": self.kv_entries,
            "estimated_kv_bytes": self.estimated_kv_bytes,
            "attention_score_count": self.attention_score_count,
            "estimated_attention_flops": self.estimated_attention_flops,
            "selected_block_count": self.selected_block_count,
            "retrieval_hit": self.retrieval_hit,
            "retrieval_recall": self.retrieval_recall,
            "answer_signal": self.answer_signal,
            "retrieved_signal": self.retrieved_signal,
            "forward_latency_ms": self.forward_latency_ms,
            "target_position": self.target_position,
            "target_block": self.target_block,
            "selected_blocks": list(self.selected_blocks),
            "attended_positions": list(self.attended_positions),
            "retrieved_position": self.retrieved_position,
            "retrieved_block": self.retrieved_block,
            "compressed_entry_count": self.compressed_entry_count,
            "local_token_count": self.local_token_count,
            "selected_token_count": self.selected_token_count,
        }


def estimate_kv_bytes(kv_entries: int, hidden_size: int, dtype_bytes: int = DEFAULT_DTYPE_BYTES) -> int:
    if kv_entries < 0:
        raise ValueError("kv_entries must not be negative")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")
    return kv_entries * hidden_size * 2 * dtype_bytes


def estimate_attention_flops(score_count: int, hidden_size: int) -> int:
    if score_count < 0:
        raise ValueError("score_count must not be negative")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    return score_count * hidden_size * 2
