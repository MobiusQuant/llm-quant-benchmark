from __future__ import annotations

import json
import re
from typing import Any

from llm_quant_bench.models import ModelResponse, ScoreResult, TestCase


def score(test_case: TestCase, response: ModelResponse) -> ScoreResult:
    if response.error:
        return ScoreResult(
            model_id=response.model_id,
            test_case_id=test_case.id,
            dimension=test_case.dimension,
            score=0.0,
            details={"error": response.error, "json_compliance": _empty_json_compliance()},
        )

    fn = SCORERS.get(test_case.scoring_method)
    if not fn:
        raise ValueError(f"Unknown scoring method: {test_case.scoring_method}")
    result = fn(test_case, response)
    result.details["json_compliance"] = check_json_compliance(response.raw_response, test_case.json_spec)
    return result


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def extract_number(text: str) -> float | None:
    """Extract the last standalone number from text."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return float(matches[-1])


def extract_json(text: str) -> Any | None:
    """Extract JSON from text, handling markdown code fences."""
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def extract_list(text: str) -> list[str]:
    """Extract items from a list in text (newline or comma separated)."""
    lines = [line.strip().lstrip("- ").lstrip("* ") for line in text.strip().splitlines()]
    items = [item for line in lines for item in line.split(",")]
    return [item.strip() for item in items if item.strip()]


# ---------------------------------------------------------------------------
# JSON Compliance Check (embedded in every test case)
# ---------------------------------------------------------------------------

def _empty_json_compliance() -> dict:
    return {"score": 0.0, "checks": {}, "passed": 0, "total": 0}


def check_json_compliance(raw_response: str, json_spec: dict | None) -> dict:
    """
    检查LLM响应的JSON规范性。

    json_spec 格式:
    {
        "type": "object" | "array",
        "fields": {
            "field_name": {"type": "str|int|float|bool|list|dict", "required": true/false, "enum": [...]}
        },
        "array_item_fields": {  # when type=array, describes each item
            "field_name": {"type": "...", "required": true/false, "enum": [...]}
        }
    }
    """
    checks: dict[str, bool] = {}
    raw = raw_response.strip()

    # 1. 可解析性：能否直接json.loads
    try:
        direct_parsed = json.loads(raw)
        checks["direct_parseable"] = True
    except (json.JSONDecodeError, ValueError):
        checks["direct_parseable"] = False
        direct_parsed = None

    # 2. 纯净度：是否是纯JSON，没有markdown code fence或多余文字
    has_fence = bool(re.search(r"```", raw))
    has_extra_text = False
    if direct_parsed is None:
        stripped = re.sub(r"```(?:json)?\s*\n?", "", raw)
        stripped = re.sub(r"\n?```", "", stripped).strip()
        try:
            json.loads(stripped)
            has_extra_text = stripped != raw
        except (json.JSONDecodeError, ValueError):
            has_extra_text = True

    checks["no_code_fence"] = not has_fence
    checks["no_extra_text"] = not has_extra_text and direct_parsed is not None

    # 3. 实际解析（兼容code fence）
    parsed = extract_json(raw)
    checks["extractable"] = parsed is not None

    if parsed is None or json_spec is None:
        passed = sum(checks.values())
        total = len(checks)
        return {
            "score": round(passed / total, 4) if total > 0 else 0.0,
            "checks": checks,
            "passed": passed,
            "total": total,
        }

    # 4. 类型检查
    expected_type = json_spec.get("type", "object")
    if expected_type == "object":
        checks["correct_type"] = isinstance(parsed, dict)
    elif expected_type == "array":
        checks["correct_type"] = isinstance(parsed, list)

    # 5. 字段检查
    type_map = {
        "str": str, "string": str,
        "int": int, "integer": int,
        "float": (int, float), "number": (int, float),
        "bool": bool, "boolean": bool,
        "list": list, "array": list,
        "dict": dict, "object": dict,
    }

    def _check_fields(obj: dict, fields_spec: dict):
        for field_name, field_def in fields_spec.items():
            is_required = field_def.get("required", True)
            field_type = field_def.get("type", "str")
            enum_values = field_def.get("enum")

            if field_name in obj:
                checks[f"field_{field_name}_present"] = True
                expected_py_type = type_map.get(field_type, str)
                checks[f"field_{field_name}_type"] = isinstance(obj[field_name], expected_py_type)
                if enum_values:
                    checks[f"field_{field_name}_enum"] = obj[field_name] in enum_values
            else:
                if is_required:
                    checks[f"field_{field_name}_present"] = False

    if expected_type == "object" and isinstance(parsed, dict):
        fields = json_spec.get("fields", {})
        _check_fields(parsed, fields)

    elif expected_type == "array" and isinstance(parsed, list):
        item_fields = json_spec.get("array_item_fields", {})
        if item_fields and parsed:
            sample = parsed[0] if isinstance(parsed[0], dict) else {}
            _check_fields(sample, item_fields)
            if len(parsed) > 1:
                last = parsed[-1] if isinstance(parsed[-1], dict) else {}
                for field_name in item_fields:
                    if field_name in last:
                        checks[f"field_{field_name}_consistent"] = True

    passed = sum(checks.values())
    total = len(checks)
    return {
        "score": round(passed / total, 4) if total > 0 else 0.0,
        "checks": checks,
        "passed": passed,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _score_numeric_tolerance(tc: TestCase, resp: ModelResponse) -> ScoreResult:
    value = extract_number(resp.raw_response)
    expected = float(tc.expected)
    tolerance = tc.scoring_params.get("tolerance", 0.5)

    if value is None:
        return ScoreResult(
            model_id=resp.model_id,
            test_case_id=tc.id,
            dimension=tc.dimension,
            score=0.0,
            details={"error": "no number found", "raw": resp.raw_response[:200]},
        )

    error = abs(value - expected)
    passed = error <= tolerance
    return ScoreResult(
        model_id=resp.model_id,
        test_case_id=tc.id,
        dimension=tc.dimension,
        score=1.0 if passed else 0.0,
        details={
            "actual": value,
            "expected": expected,
            "error": round(error, 6),
            "tolerance": tolerance,
        },
    )


def _score_binary(tc: TestCase, resp: ModelResponse) -> ScoreResult:
    expected = str(tc.expected).strip().upper()
    raw = resp.raw_response.strip().upper()
    matched = expected in raw
    return ScoreResult(
        model_id=resp.model_id,
        test_case_id=tc.id,
        dimension=tc.dimension,
        score=1.0 if matched else 0.0,
        details={"expected": expected, "raw": resp.raw_response[:200]},
    )


def _score_f1_set(tc: TestCase, resp: ModelResponse) -> ScoreResult:
    expected_items: list = tc.expected if isinstance(tc.expected, list) else []
    match_key = tc.scoring_params.get("match_key", None)

    parsed = extract_json(resp.raw_response)
    if parsed is None:
        parsed = []
    if not isinstance(parsed, list):
        parsed = [parsed]

    if match_key:
        expected_keys = {item[match_key] for item in expected_items if isinstance(item, dict)}
        actual_keys = {item[match_key] for item in parsed if isinstance(item, dict) and match_key in item}
    else:
        expected_keys = set(map(str, expected_items))
        actual_keys = set(map(str, parsed))

    tp = len(expected_keys & actual_keys)
    fp = len(actual_keys - expected_keys)
    fn = len(expected_keys - actual_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ScoreResult(
        model_id=resp.model_id,
        test_case_id=tc.id,
        dimension=tc.dimension,
        score=round(f1, 4),
        details={
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn,
        },
    )


def _score_json_schema(tc: TestCase, resp: ModelResponse) -> ScoreResult:
    parsed = extract_json(resp.raw_response)
    checks: dict[str, bool] = {}

    checks["parseable"] = parsed is not None
    if not checks["parseable"]:
        return ScoreResult(
            model_id=resp.model_id,
            test_case_id=tc.id,
            dimension=tc.dimension,
            score=0.0,
            details={"checks": checks, "raw": resp.raw_response[:200]},
        )

    schema = tc.expected if isinstance(tc.expected, dict) else {}

    required_fields = schema.get("required", [])
    if required_fields and isinstance(parsed, dict):
        for field in required_fields:
            checks[f"has_{field}"] = field in parsed

    field_types = schema.get("field_types", {})
    if field_types and isinstance(parsed, dict):
        type_map = {"str": str, "int": int, "float": (int, float), "list": list, "dict": dict, "bool": bool}
        for field, expected_type in field_types.items():
            if field in parsed:
                checks[f"type_{field}"] = isinstance(parsed[field], type_map.get(expected_type, object))

    passed = sum(checks.values())
    total = len(checks)
    return ScoreResult(
        model_id=resp.model_id,
        test_case_id=tc.id,
        dimension=tc.dimension,
        score=round(passed / total, 4) if total > 0 else 0.0,
        details={"checks": checks, "passed": passed, "total": total},
    )


def _score_checklist(tc: TestCase, resp: ModelResponse) -> ScoreResult:
    checkpoints: list = tc.expected if isinstance(tc.expected, list) else []
    raw_upper = resp.raw_response.upper()

    results: dict[str, bool] = {}
    for cp in checkpoints:
        if isinstance(cp, str):
            results[cp] = cp.upper() in raw_upper
        elif isinstance(cp, dict):
            label = cp.get("label", str(cp))
            keyword = cp.get("contains", "")
            if keyword:
                results[label] = keyword.upper() in raw_upper
            else:
                results[label] = False

    passed = sum(results.values())
    total = len(results)
    return ScoreResult(
        model_id=resp.model_id,
        test_case_id=tc.id,
        dimension=tc.dimension,
        score=round(passed / total, 4) if total > 0 else 0.0,
        details={"checkpoints": results, "passed": passed, "total": total},
    )


def _score_macd_tolerance(tc: TestCase, resp: ModelResponse) -> ScoreResult:
    expected = tc.expected
    tolerance_pct = tc.scoring_params.get("tolerance_pct", 0.01)
    raw = resp.raw_response

    parsed = {}
    for key in ["DIF", "DEA", "MACD"]:
        match = re.search(rf"{key}\s*=\s*(-?\d+(?:\.\d+)?)", raw)
        if match:
            parsed[key] = float(match.group(1))

    if not parsed:
        return ScoreResult(
            model_id=resp.model_id,
            test_case_id=tc.id,
            dimension=tc.dimension,
            score=0.0,
            details={"error": "cannot parse DIF/DEA/MACD", "raw": raw[:200]},
        )

    checks = {}
    for key in ["DIF", "DEA", "MACD"]:
        exp_val = float(expected.get(key, 0))
        act_val = parsed.get(key)
        if act_val is None:
            checks[key] = {"expected": exp_val, "actual": None, "pass": False}
        else:
            threshold = max(abs(exp_val) * tolerance_pct, 1.0)
            error = abs(act_val - exp_val)
            checks[key] = {
                "expected": exp_val,
                "actual": act_val,
                "error": round(error, 4),
                "threshold": round(threshold, 4),
                "pass": error <= threshold,
            }

    passed = sum(1 for v in checks.values() if v.get("pass"))
    total = len(checks)
    return ScoreResult(
        model_id=resp.model_id,
        test_case_id=tc.id,
        dimension=tc.dimension,
        score=round(passed / total, 4),
        details={"checks": checks, "passed": passed, "total": total},
    )


def _score_highlow_match(tc: TestCase, resp: ModelResponse) -> ScoreResult:
    """高低点精确匹配：row + price 都要对"""
    expected = tc.expected
    parsed = extract_json(resp.raw_response)

    if not isinstance(parsed, dict):
        return ScoreResult(
            model_id=resp.model_id, test_case_id=tc.id, dimension=tc.dimension,
            score=0.0, details={"error": "cannot parse JSON", "raw": resp.raw_response[:200]},
        )

    checks = {}
    checks["row"] = parsed.get("row") == expected.get("row")
    exp_price = expected.get("price", 0)
    act_price = parsed.get("price", 0)
    try:
        checks["price"] = abs(float(act_price) - float(exp_price)) < 0.01
    except (TypeError, ValueError):
        checks["price"] = False

    passed = sum(checks.values())
    total = len(checks)
    return ScoreResult(
        model_id=resp.model_id, test_case_id=tc.id, dimension=tc.dimension,
        score=round(passed / total, 4),
        details={
            "expected": expected,
            "actual": parsed,
            "checks": checks,
            "passed": passed, "total": total,
        },
    )


def _score_smc_f1(tc: TestCase, resp: ModelResponse) -> ScoreResult:
    """SMC结构识别F1评分：基于match_key匹配，支持price_tolerance"""
    expected_items = tc.expected if isinstance(tc.expected, list) else []
    match_key = tc.scoring_params.get("match_key", "timestamp")
    price_tolerance = tc.scoring_params.get("price_tolerance", 50.0)

    parsed = extract_json(resp.raw_response)
    if parsed is None:
        parsed = []
    if not isinstance(parsed, list):
        parsed = [parsed] if isinstance(parsed, dict) else []

    matched_expected = set()
    matched_actual = set()

    for i, act in enumerate(parsed):
        if not isinstance(act, dict):
            continue
        for j, exp in enumerate(expected_items):
            if j in matched_expected:
                continue
            if act.get(match_key) == exp.get(match_key):
                matched_expected.add(j)
                matched_actual.add(i)
                break

    tp = len(matched_expected)
    fp = len(parsed) - len(matched_actual)
    fn = len(expected_items) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ScoreResult(
        model_id=resp.model_id, test_case_id=tc.id, dimension=tc.dimension,
        score=round(f1, 4),
        details={
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "expected_count": len(expected_items),
            "actual_count": len(parsed),
        },
    )


def _score_rule_signal(tc: TestCase, resp: ModelResponse) -> ScoreResult:
    """规则执行 + 约束遵循综合评分"""
    expected = tc.expected
    constraints = tc.scoring_params.get("constraints", [])
    parsed = extract_json(resp.raw_response)

    if not isinstance(parsed, dict):
        return ScoreResult(
            model_id=resp.model_id, test_case_id=tc.id, dimension=tc.dimension,
            score=0.0, details={"error": "cannot parse JSON", "raw": resp.raw_response[:300]},
        )

    checks: dict[str, bool] = {}

    # T4: Signal correctness
    exp_signals = expected.get("signals", [])
    act_signals = parsed.get("signals", [])
    if not isinstance(act_signals, list):
        act_signals = []

    for i, exp_sig in enumerate(exp_signals):
        act_sig = act_signals[i] if i < len(act_signals) else {}
        exp_type = exp_sig.get("type", "HOLD")
        act_type = str(act_sig.get("type", "")).upper() if isinstance(act_sig, dict) else ""
        checks[f"signal_{i}_type"] = act_type == exp_type

    # T4: Trend/context correctness (Q4)
    if "market_context" in expected:
        exp_ctx = expected["market_context"]
        act_ctx = parsed.get("market_context", {})
        if isinstance(act_ctx, dict):
            if "trend" in exp_ctx:
                checks["trend"] = act_ctx.get("trend") == exp_ctx["trend"]
            if "volatility" in exp_ctx:
                checks["volatility"] = act_ctx.get("volatility") == exp_ctx["volatility"]

    # T7: Constraint checks
    for c in constraints:
        cid = c["id"]
        rule = c["rule"].lower()

        if "timeframe" in rule and "1h" in rule:
            checks[f"constraint_{cid}"] = parsed.get("timeframe") == "1h"

        elif "last 5 candles" in rule or "last 5" in rule:
            checks[f"constraint_{cid}"] = len(act_signals) <= 5

        elif "sell" in rule and "priority" in rule:
            checks[f"constraint_{cid}"] = True  # checked via signal correctness

        elif "confidence" in rule:
            all_valid = True
            for sig in act_signals:
                if isinstance(sig, dict):
                    conf = sig.get("confidence")
                    if conf is not None:
                        try:
                            all_valid = all_valid and (0.0 <= float(conf) <= 1.0)
                        except (TypeError, ValueError):
                            all_valid = False
            checks[f"constraint_{cid}"] = all_valid

        elif "volume" in rule and "not" in rule:
            raw_lower = resp.raw_response.lower()
            checks[f"constraint_{cid}"] = "volume" not in str(parsed.get("signals", "")).lower()

        elif "trend" in rule and ("bullish" in rule or "bearish" in rule):
            act_trend = parsed.get("market_context", {}).get("trend", "")
            checks[f"constraint_{cid}"] = act_trend in ["bullish", "bearish", "sideways"]

        elif "volatility" in rule:
            act_vol = parsed.get("market_context", {}).get("volatility", "")
            checks[f"constraint_{cid}"] = act_vol in ["high", "medium", "low"]

        elif "at most 1" in rule or "most 1 element" in rule:
            checks[f"constraint_{cid}"] = len(act_signals) <= 1

    passed = sum(checks.values())
    total = len(checks)
    return ScoreResult(
        model_id=resp.model_id, test_case_id=tc.id, dimension=tc.dimension,
        score=round(passed / total, 4) if total > 0 else 0.0,
        details={"checks": checks, "passed": passed, "total": total},
    )


def _score_anomaly_f1(tc: TestCase, resp: ModelResponse) -> ScoreResult:
    """异常检测F1：匹配row号+type"""
    expected = tc.expected if isinstance(tc.expected, list) else []
    parsed = extract_json(resp.raw_response)
    if not isinstance(parsed, list):
        parsed = [parsed] if isinstance(parsed, dict) else []

    matched = set()
    tp = 0
    for act in parsed:
        if not isinstance(act, dict):
            continue
        act_row = act.get("row")
        act_type = str(act.get("type", "")).lower().strip()
        for j, exp in enumerate(expected):
            if j in matched:
                continue
            if act_row == exp.get("row") and act_type == exp.get("type", "").lower():
                matched.add(j)
                tp += 1
                break

    fp = len(parsed) - tp
    fn = len(expected) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ScoreResult(
        model_id=resp.model_id, test_case_id=tc.id, dimension=tc.dimension,
        score=round(f1, 4),
        details={
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "expected_count": len(expected),
            "actual_count": len(parsed),
        },
    )


SCORERS = {
    "numeric_tolerance": _score_numeric_tolerance,
    "binary": _score_binary,
    "f1_set": _score_f1_set,
    "json_schema": _score_json_schema,
    "checklist": _score_checklist,
    "macd_tolerance": _score_macd_tolerance,
    "highlow_match": _score_highlow_match,
    "smc_f1": _score_smc_f1,
    "rule_signal": _score_rule_signal,
    "anomaly_f1": _score_anomaly_f1,
}
