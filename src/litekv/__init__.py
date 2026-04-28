"""LiteKV demo package."""

from litekv.config import ATTENTION_MODES, ExperimentConfig, load_config
from litekv.data import RetrievalCase, SyntheticTask, TargetRegion, generate_retrieval_case
from litekv.attention import AttentionResult, run_attention
from litekv.metrics import AttentionMetrics
from litekv.experiment import ExperimentArtifacts, run_experiment
from litekv.model import ToyDecoderBlock, ToyDecoderConfig, ToyDecoderOutput
from litekv.plots import PlotArtifacts, generate_plots

__all__ = [
    "ATTENTION_MODES",
    "ExperimentArtifacts",
    "AttentionMetrics",
    "AttentionResult",
    "ExperimentConfig",
    "RetrievalCase",
    "PlotArtifacts",
    "SyntheticTask",
    "TargetRegion",
    "ToyDecoderBlock",
    "ToyDecoderConfig",
    "ToyDecoderOutput",
    "generate_retrieval_case",
    "load_config",
    "run_attention",
    "run_experiment",
    "generate_plots",
]
