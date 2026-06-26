"""LLM Quant Benchmark — evaluate LLMs on quantitative-trading tasks.

A small, provider-agnostic harness for benchmarking large language models on
indicator calculation, K-line structure analysis, trading-rule execution and
data-anomaly detection, with embedded JSON-compliance scoring.
"""
from __future__ import annotations

__version__ = "0.1.0"

from llm_quant_bench.client import LLMClient
from llm_quant_bench.config import get_api_key, load_config
from llm_quant_bench.loader import load_all, load_dimension
from llm_quant_bench.models import ModelResponse, RunResult, ScoreResult, TestCase
from llm_quant_bench.runner import BenchmarkRunner
from llm_quant_bench.scorer import score

__all__ = [
    "__version__",
    "LLMClient",
    "BenchmarkRunner",
    "score",
    "load_all",
    "load_dimension",
    "load_config",
    "get_api_key",
    "TestCase",
    "ModelResponse",
    "ScoreResult",
    "RunResult",
]
