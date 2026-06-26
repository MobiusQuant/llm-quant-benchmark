"""Unit tests for the scoring engine — run with `pytest`."""
from __future__ import annotations

from llm_quant_bench.models import ModelResponse, TestCase
from llm_quant_bench.scorer import (
    check_json_compliance,
    extract_number,
    score,
)


def _case(**kw) -> TestCase:
    base = dict(
        id="t",
        dimension="t1",
        prompt="p",
        expected=0,
        scoring_method="numeric_tolerance",
    )
    base.update(kw)
    return TestCase(**base)


def _resp(text: str) -> ModelResponse:
    return ModelResponse(model_id="m", test_case_id="t", raw_response=text)


# --- parsing ---------------------------------------------------------------

def test_extract_number_takes_last():
    assert extract_number("DIF=12.5, answer is 67783.66") == 67783.66
    assert extract_number("no digits here") is None


# --- numeric tolerance -----------------------------------------------------

def test_numeric_tolerance_pass_and_fail():
    tc = _case(expected=100.0, scoring_params={"tolerance": 0.5})
    assert score(tc, _resp("The SMA is 100.3")).score == 1.0
    assert score(tc, _resp("The SMA is 105")).score == 0.0


# --- anomaly F1 ------------------------------------------------------------

def test_anomaly_f1_perfect_match():
    expected = [{"row": 5, "type": "price_spike"}, {"row": 9, "type": "missing_value"}]
    tc = _case(dimension="t5", scoring_method="anomaly_f1", expected=expected)
    resp = _resp('[{"row": 5, "type": "price_spike"}, {"row": 9, "type": "missing_value"}]')
    assert score(tc, resp).score == 1.0


def test_anomaly_f1_partial():
    expected = [{"row": 5, "type": "price_spike"}, {"row": 9, "type": "missing_value"}]
    tc = _case(dimension="t5", scoring_method="anomaly_f1", expected=expected)
    resp = _resp('[{"row": 5, "type": "price_spike"}]')  # 1 of 2 found, no false positives
    result = score(tc, resp)
    assert 0.0 < result.score < 1.0
    assert result.details["tp"] == 1 and result.details["fn"] == 1


# --- JSON compliance -------------------------------------------------------

def test_json_compliance_clean_beats_fenced():
    spec = {"type": "object", "fields": {"signal": {"type": "str", "enum": ["BUY", "SELL"]}}}
    clean = check_json_compliance('{"signal": "BUY"}', spec)
    fenced = check_json_compliance('```json\n{"signal": "BUY"}\n```', spec)
    assert clean["checks"]["direct_parseable"] is True
    assert fenced["checks"]["direct_parseable"] is False   # code fence breaks json.loads
    assert clean["score"] > fenced["score"]


def test_error_response_scores_zero():
    tc = _case(expected=100.0)
    result = score(tc, ModelResponse(model_id="m", test_case_id="t", error="timeout"))
    assert result.score == 0.0
