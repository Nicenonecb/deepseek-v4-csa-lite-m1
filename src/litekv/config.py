from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


ATTENTION_MODES = (
    "dense",
    "sliding_window",
    "csa_lite",
    "csa_lite_local",
    "nsa_lite",
)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _parse_inline_list(value: str) -> List[Any]:
    inner = value.strip()[1:-1].strip()
    if not inner:
        return []
    return [_parse_scalar(part) for part in inner.split(",")]


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError("Invalid config line: {}".format(raw_line))
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            config[key] = _parse_inline_list(value)
        else:
            config[key] = _parse_scalar(value)
    return config


def resolve_device(requested: str = "auto") -> str:
    if requested == "cpu":
        return "cpu"

    if requested in {"auto", "mps"}:
        try:
            import torch  # type: ignore

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass

        if requested == "mps":
            return "cpu"
        return "cpu"

    raise ValueError("Unsupported device: {}".format(requested))


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError("{} must be positive".format(name))


def _validate_positive_ints(name: str, values: Iterable[int]) -> None:
    for value in values:
        _validate_positive_int(name, int(value))


@dataclass
class ExperimentConfig:
    context_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    hidden_size: int = 256
    attention_heads: int = 4
    batch_size: int = 1
    compression_ratio: int = 4
    top_k_values: List[int] = field(default_factory=lambda: [32, 64, 128])
    local_window_values: List[int] = field(default_factory=lambda: [128, 256])
    seed: int = 2026
    device: str = "auto"
    output_dir: Path = Path("results")
    attention_modes: List[str] = field(default_factory=lambda: list(ATTENTION_MODES))

    def __post_init__(self) -> None:
        self.context_lengths = [int(value) for value in self.context_lengths]
        self.top_k_values = [int(value) for value in self.top_k_values]
        self.local_window_values = [int(value) for value in self.local_window_values]
        self.hidden_size = int(self.hidden_size)
        self.attention_heads = int(self.attention_heads)
        self.batch_size = int(self.batch_size)
        self.compression_ratio = int(self.compression_ratio)
        self.seed = int(self.seed)
        self.output_dir = Path(self.output_dir)
        self.attention_modes = list(self.attention_modes)

        _validate_positive_ints("context length", self.context_lengths)
        _validate_positive_int("hidden_size", self.hidden_size)
        _validate_positive_int("attention_heads", self.attention_heads)
        _validate_positive_int("batch_size", self.batch_size)
        _validate_positive_int("compression_ratio", self.compression_ratio)
        _validate_positive_ints("top_k", self.top_k_values)
        _validate_positive_ints("local_window", self.local_window_values)

        unknown_modes = sorted(set(self.attention_modes) - set(ATTENTION_MODES))
        if unknown_modes:
            raise ValueError("Unknown attention modes: {}".format(", ".join(unknown_modes)))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        fields = {
            "context_lengths",
            "hidden_size",
            "attention_heads",
            "batch_size",
            "compression_ratio",
            "top_k_values",
            "local_window_values",
            "seed",
            "device",
            "output_dir",
            "attention_modes",
        }
        kwargs = {key: value for key, value in values.items() if key in fields}
        return cls(**kwargs)

    def resolved_device(self) -> str:
        return resolve_device(self.device)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "context_lengths": self.context_lengths,
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "batch_size": self.batch_size,
            "compression_ratio": self.compression_ratio,
            "top_k_values": self.top_k_values,
            "local_window_values": self.local_window_values,
            "seed": self.seed,
            "device": self.device,
            "resolved_device": self.resolved_device(),
            "output_dir": str(self.output_dir),
            "attention_modes": self.attention_modes,
        }


def load_config(path: Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return ExperimentConfig.from_mapping(_load_simple_yaml(path))
