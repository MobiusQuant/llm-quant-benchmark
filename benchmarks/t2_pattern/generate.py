"""
T2 K线结构识别 — 测试用例生成脚本

从真实BTC K线数据生成高低点识别和SMC结构识别的测试用例。
每个题目: 1段数据 × 2种模式（给定义 / 不给定义）= 2个用例
代码支持多段数据配置，供开源社区扩展。

用法：
    cd tests/t2_pattern
    python3 generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from reference import (
    load_klines,
    find_highest,
    find_lowest,
    compute_smc,
    extract_fvg,
    extract_order_blocks,
    extract_bos_choch,
    extract_eqhl,
)

OUTPUT_DIR = Path(__file__).resolve().parent

# 使用200根K线，给SMC足够的数据
DATA_FILES = [
    ("20240101", "btc_usdt_1h_20240101.csv"),
]
KLINE_LIMIT = 200


# ============================================================
# Prompt 模板
# ============================================================

FVG_DEFINITION = """FVG (Fair Value Gap) Definition:
A Fair Value Gap occurs when three consecutive candles create a price gap:
- Bullish FVG: The LOW of candle 3 is HIGHER than the HIGH of candle 1, creating a gap between candle 1's high and candle 3's low
- Bearish FVG: The HIGH of candle 3 is LOWER than the LOW of candle 1, creating a gap between candle 3's high and candle 1's low
The FVG zone is defined by the top and bottom of this gap."""

OB_DEFINITION = """Order Block (OB) Definition:
An Order Block is the last opposing candle before a strong directional move that breaks market structure:
- Bullish OB: The last bearish candle before a strong bullish move that breaks above a previous swing high
- Bearish OB: The last bullish candle before a strong bearish move that breaks below a previous swing low
The OB zone is defined by the high and low of that candle."""

BOS_CHOCH_DEFINITION = """BOS and CHoCH Definition:
- BOS (Break of Structure): When price breaks beyond a swing point IN THE SAME direction as the current trend, confirming trend continuation
  - In uptrend: price breaks above the previous swing high
  - In downtrend: price breaks below the previous swing low
- CHoCH (Change of Character): When price breaks beyond a swing point in the OPPOSITE direction of the current trend, signaling potential reversal
  - In uptrend: price breaks below the previous swing low
  - In downtrend: price breaks above the previous swing high"""

EQHL_DEFINITION = """EQH/EQL Definition:
- EQH (Equal Highs): Two or more swing highs at approximately the same price level (within a small threshold, typically 0.1 × ATR), indicating liquidity accumulation above
- EQL (Equal Lows): Two or more swing lows at approximately the same price level, indicating liquidity accumulation below"""


# ============================================================
# JSON Spec 定义（用于JSON合规性评分）
# ============================================================

HIGHLOW_JSON_SPEC = {
    "type": "object",
    "fields": {
        "row": {"type": "int", "required": True},
        "timestamp": {"type": "str", "required": True},
        "price": {"type": "float", "required": True},
    },
}

FVG_JSON_SPEC = {
    "type": "array",
    "array_item_fields": {
        "timestamp": {"type": "str", "required": True},
        "top": {"type": "float", "required": True},
        "bottom": {"type": "float", "required": True},
        "bias": {"type": "str", "required": True, "enum": ["bull", "bear"]},
    },
}

OB_JSON_SPEC = {
    "type": "array",
    "array_item_fields": {
        "timestamp": {"type": "str", "required": True},
        "top": {"type": "float", "required": True},
        "bottom": {"type": "float", "required": True},
        "bias": {"type": "str", "required": True, "enum": ["bull", "bear"]},
    },
}

BOS_CHOCH_JSON_SPEC = {
    "type": "array",
    "array_item_fields": {
        "pivot_timestamp": {"type": "str", "required": True},
        "pivot_price": {"type": "float", "required": True},
        "kind": {"type": "str", "required": True, "enum": ["BOS", "CHoCH"]},
        "bias": {"type": "str", "required": True, "enum": ["bull", "bear"]},
    },
}

EQHL_JSON_SPEC = {
    "type": "array",
    "array_item_fields": {
        "type": {"type": "str", "required": True, "enum": ["EQH", "EQL"]},
        "anchor_timestamp": {"type": "str", "required": True},
        "confirm_timestamp": {"type": "str", "required": True},
        "level": {"type": "float", "required": True},
    },
}


def format_kline_table(df) -> str:
    lines = ["Row,Timestamp,Open,High,Low,Close,Volume"]
    for i, (_, row) in enumerate(df.iterrows()):
        lines.append(
            f"{i+1},{row['timestamp']},"
            f"{float(row['open']):.2f},{float(row['high']):.2f},"
            f"{float(row['low']):.2f},{float(row['close']):.2f},"
            f"{float(row['volume']):.4f}"
        )
    return "\n".join(lines)


def generate_highlow():
    """生成高低点识别用例"""
    cases = []
    for tag, filename in DATA_FILES:
        df, _ = load_klines(filename, KLINE_LIMIT)
        highest = find_highest(df)
        lowest = find_lowest(df)
        kline_text = format_kline_table(df)

        # --- 最高点 ---
        cases.append({
            "id": f"t2_highest_with_def_{tag}",
            "prompt": (
                "Find the candle with the highest price in the following BTC/USDT 1-hour K-line data.\n\n"
                "Definition: The highest price is the maximum value in the 'High' column across all rows.\n\n"
                f"K-line data ({len(df)} candles):\n{kline_text}\n\n"
                "Return your answer in this exact JSON format:\n"
                '{"row": <row_number>, "timestamp": "<timestamp>", "price": <highest_price>}'
            ),
            "system_prompt": "You are a quantitative analyst. Return only the JSON result, no explanation.",
            "expected": highest,
            "scoring_method": "highlow_match",
            "scoring_params": {},
            "metadata": {"task": "highest", "with_definition": True, "data_segment": tag},
            "json_spec": HIGHLOW_JSON_SPEC,
        })
        cases.append({
            "id": f"t2_highest_no_def_{tag}",
            "prompt": (
                "Find the candle with the highest price in the following BTC/USDT 1-hour K-line data.\n\n"
                f"K-line data ({len(df)} candles):\n{kline_text}\n\n"
                "Return your answer in this exact JSON format:\n"
                '{"row": <row_number>, "timestamp": "<timestamp>", "price": <highest_price>}'
            ),
            "system_prompt": "You are a quantitative analyst. Return only the JSON result, no explanation.",
            "expected": highest,
            "scoring_method": "highlow_match",
            "scoring_params": {},
            "metadata": {"task": "highest", "with_definition": False, "data_segment": tag},
            "json_spec": HIGHLOW_JSON_SPEC,
        })

        # --- 最低点 ---
        cases.append({
            "id": f"t2_lowest_with_def_{tag}",
            "prompt": (
                "Find the candle with the lowest price in the following BTC/USDT 1-hour K-line data.\n\n"
                "Definition: The lowest price is the minimum value in the 'Low' column across all rows.\n\n"
                f"K-line data ({len(df)} candles):\n{kline_text}\n\n"
                "Return your answer in this exact JSON format:\n"
                '{"row": <row_number>, "timestamp": "<timestamp>", "price": <lowest_price>}'
            ),
            "system_prompt": "You are a quantitative analyst. Return only the JSON result, no explanation.",
            "expected": lowest,
            "scoring_method": "highlow_match",
            "scoring_params": {},
            "metadata": {"task": "lowest", "with_definition": True, "data_segment": tag},
            "json_spec": HIGHLOW_JSON_SPEC,
        })
        cases.append({
            "id": f"t2_lowest_no_def_{tag}",
            "prompt": (
                "Find the candle with the lowest price in the following BTC/USDT 1-hour K-line data.\n\n"
                f"K-line data ({len(df)} candles):\n{kline_text}\n\n"
                "Return your answer in this exact JSON format:\n"
                '{"row": <row_number>, "timestamp": "<timestamp>", "price": <lowest_price>}'
            ),
            "system_prompt": "You are a quantitative analyst. Return only the JSON result, no explanation.",
            "expected": lowest,
            "scoring_method": "highlow_match",
            "scoring_params": {},
            "metadata": {"task": "lowest", "with_definition": False, "data_segment": tag},
            "json_spec": HIGHLOW_JSON_SPEC,
        })

    output = {"dimension": "t2", "cases": cases}
    with open(OUTPUT_DIR / "highlow.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Generated {len(cases)} high/low test cases -> highlow.yaml")


def _smc_prompt(task_name: str, definition: str, kline_text: str, n_candles: int, output_format: str, with_def: bool) -> str:
    parts = [f"Analyze the following BTC/USDT 1-hour K-line data and identify all {task_name}.\n"]
    if with_def:
        parts.append(f"{definition}\n")
    parts.append(f"K-line data ({n_candles} candles):\n{kline_text}\n")
    parts.append(f"Return your answer as a JSON array. Each item should have: {output_format}")
    parts.append("If none found, return an empty array: []")
    return "\n".join(parts)


def generate_fvg():
    cases = []
    for tag, filename in DATA_FILES:
        df, api_klines = load_klines(filename, KLINE_LIMIT)
        objects = compute_smc(api_klines)
        expected = extract_fvg(objects)
        kline_text = format_kline_table(df)
        fmt = '{"timestamp": "<candle_timestamp>", "top": <upper_price>, "bottom": <lower_price>, "bias": "bull|bear"}'

        cases.append({
            "id": f"t2_fvg_with_def_{tag}",
            "prompt": _smc_prompt("Fair Value Gaps (FVG)", FVG_DEFINITION, kline_text, len(df), fmt, True),
            "system_prompt": "You are an SMC (Smart Money Concepts) analyst. Return only the JSON array, no explanation.",
            "expected": expected,
            "scoring_method": "smc_f1",
            "scoring_params": {"match_key": "timestamp", "price_tolerance": 50.0},
            "metadata": {"task": "fvg", "with_definition": True, "data_segment": tag, "expected_count": len(expected)},
            "json_spec": FVG_JSON_SPEC,
        })
        cases.append({
            "id": f"t2_fvg_no_def_{tag}",
            "prompt": _smc_prompt("Fair Value Gaps (FVG)", FVG_DEFINITION, kline_text, len(df), fmt, False),
            "system_prompt": "You are an SMC (Smart Money Concepts) analyst. Return only the JSON array, no explanation.",
            "expected": expected,
            "scoring_method": "smc_f1",
            "scoring_params": {"match_key": "timestamp", "price_tolerance": 50.0},
            "metadata": {"task": "fvg", "with_definition": False, "data_segment": tag, "expected_count": len(expected)},
            "json_spec": FVG_JSON_SPEC,
        })

    output = {"dimension": "t2", "cases": cases}
    with open(OUTPUT_DIR / "fvg.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Generated {len(cases)} FVG test cases -> fvg.yaml")


def generate_orderblock():
    cases = []
    for tag, filename in DATA_FILES:
        df, api_klines = load_klines(filename, KLINE_LIMIT)
        objects = compute_smc(api_klines)
        expected = extract_order_blocks(objects)
        kline_text = format_kline_table(df)
        fmt = '{"timestamp": "<candle_timestamp>", "top": <upper_price>, "bottom": <lower_price>, "bias": "bull|bear"}'

        cases.append({
            "id": f"t2_ob_with_def_{tag}",
            "prompt": _smc_prompt("Order Blocks (OB)", OB_DEFINITION, kline_text, len(df), fmt, True),
            "system_prompt": "You are an SMC (Smart Money Concepts) analyst. Return only the JSON array, no explanation.",
            "expected": expected,
            "scoring_method": "smc_f1",
            "scoring_params": {"match_key": "timestamp", "price_tolerance": 50.0},
            "metadata": {"task": "order_block", "with_definition": True, "data_segment": tag, "expected_count": len(expected)},
            "json_spec": OB_JSON_SPEC,
        })
        cases.append({
            "id": f"t2_ob_no_def_{tag}",
            "prompt": _smc_prompt("Order Blocks (OB)", OB_DEFINITION, kline_text, len(df), fmt, False),
            "system_prompt": "You are an SMC (Smart Money Concepts) analyst. Return only the JSON array, no explanation.",
            "expected": expected,
            "scoring_method": "smc_f1",
            "scoring_params": {"match_key": "timestamp", "price_tolerance": 50.0},
            "metadata": {"task": "order_block", "with_definition": False, "data_segment": tag, "expected_count": len(expected)},
            "json_spec": OB_JSON_SPEC,
        })

    output = {"dimension": "t2", "cases": cases}
    with open(OUTPUT_DIR / "orderblock.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Generated {len(cases)} Order Block test cases -> orderblock.yaml")


def generate_bos_choch():
    cases = []
    for tag, filename in DATA_FILES:
        df, api_klines = load_klines(filename, KLINE_LIMIT)
        objects = compute_smc(api_klines)
        expected = extract_bos_choch(objects)
        kline_text = format_kline_table(df)
        fmt = '{"pivot_timestamp": "<timestamp>", "pivot_price": <price>, "kind": "BOS|CHoCH", "bias": "bull|bear"}'

        cases.append({
            "id": f"t2_bos_with_def_{tag}",
            "prompt": _smc_prompt("BOS (Break of Structure) and CHoCH (Change of Character) events", BOS_CHOCH_DEFINITION, kline_text, len(df), fmt, True),
            "system_prompt": "You are an SMC (Smart Money Concepts) analyst. Return only the JSON array, no explanation.",
            "expected": expected,
            "scoring_method": "smc_f1",
            "scoring_params": {"match_key": "pivot_timestamp", "price_tolerance": 50.0},
            "metadata": {"task": "bos_choch", "with_definition": True, "data_segment": tag, "expected_count": len(expected)},
            "json_spec": BOS_CHOCH_JSON_SPEC,
        })
        cases.append({
            "id": f"t2_bos_no_def_{tag}",
            "prompt": _smc_prompt("BOS (Break of Structure) and CHoCH (Change of Character) events", BOS_CHOCH_DEFINITION, kline_text, len(df), fmt, False),
            "system_prompt": "You are an SMC (Smart Money Concepts) analyst. Return only the JSON array, no explanation.",
            "expected": expected,
            "scoring_method": "smc_f1",
            "scoring_params": {"match_key": "pivot_timestamp", "price_tolerance": 50.0},
            "metadata": {"task": "bos_choch", "with_definition": False, "data_segment": tag, "expected_count": len(expected)},
            "json_spec": BOS_CHOCH_JSON_SPEC,
        })

    output = {"dimension": "t2", "cases": cases}
    with open(OUTPUT_DIR / "bos_choch.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Generated {len(cases)} BOS/CHoCH test cases -> bos_choch.yaml")


def generate_eqhl():
    cases = []
    for tag, filename in DATA_FILES:
        df, api_klines = load_klines(filename, KLINE_LIMIT)
        objects = compute_smc(api_klines)
        expected = extract_eqhl(objects)
        kline_text = format_kline_table(df)
        fmt = '{"type": "EQH|EQL", "anchor_timestamp": "<timestamp>", "confirm_timestamp": "<timestamp>", "level": <price>}'

        cases.append({
            "id": f"t2_eqhl_with_def_{tag}",
            "prompt": _smc_prompt("Equal Highs (EQH) and Equal Lows (EQL)", EQHL_DEFINITION, kline_text, len(df), fmt, True),
            "system_prompt": "You are an SMC (Smart Money Concepts) analyst. Return only the JSON array, no explanation.",
            "expected": expected,
            "scoring_method": "smc_f1",
            "scoring_params": {"match_key": "type", "price_tolerance": 100.0},
            "metadata": {"task": "eqhl", "with_definition": True, "data_segment": tag, "expected_count": len(expected)},
            "json_spec": EQHL_JSON_SPEC,
        })
        cases.append({
            "id": f"t2_eqhl_no_def_{tag}",
            "prompt": _smc_prompt("Equal Highs (EQH) and Equal Lows (EQL)", EQHL_DEFINITION, kline_text, len(df), fmt, False),
            "system_prompt": "You are an SMC (Smart Money Concepts) analyst. Return only the JSON array, no explanation.",
            "expected": expected,
            "scoring_method": "smc_f1",
            "scoring_params": {"match_key": "type", "price_tolerance": 100.0},
            "metadata": {"task": "eqhl", "with_definition": False, "data_segment": tag, "expected_count": len(expected)},
            "json_spec": EQHL_JSON_SPEC,
        })

    output = {"dimension": "t2", "cases": cases}
    with open(OUTPUT_DIR / "eqhl.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Generated {len(cases)} EQH/EQL test cases -> eqhl.yaml")


# Save SMC API raw response for reference
def save_smc_raw():
    for tag, filename in DATA_FILES:
        _, api_klines = load_klines(filename, KLINE_LIMIT)
        objects = compute_smc(api_klines)
        raw_path = OUTPUT_DIR / f"smc_raw_{tag}.json"
        with open(raw_path, "w") as f:
            json.dump(objects, f, indent=2, ensure_ascii=False)
        print(f"Saved SMC raw objects -> smc_raw_{tag}.json")


if __name__ == "__main__":
    save_smc_raw()
    generate_highlow()
    generate_fvg()
    generate_orderblock()
    generate_bos_choch()
    generate_eqhl()
    print("\nAll T2 test cases generated.")
