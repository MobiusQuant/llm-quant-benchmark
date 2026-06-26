from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    id: str
    dimension: str
    prompt: str
    system_prompt: str | None = None
    expected: Any
    scoring_method: str
    scoring_params: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    json_spec: dict | None = None


class ModelResponse(BaseModel):
    model_id: str
    test_case_id: str
    raw_response: str = ""
    parsed_response: Any = None
    latency_ms: float = 0.0
    token_usage: dict = Field(default_factory=dict)
    error: str | None = None


class ScoreResult(BaseModel):
    model_id: str
    test_case_id: str
    dimension: str
    score: float
    details: dict = Field(default_factory=dict)


class RunResult(BaseModel):
    run_id: str
    scores: list[ScoreResult] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)
