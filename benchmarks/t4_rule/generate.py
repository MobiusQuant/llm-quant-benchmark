"""
T4+T7 规则执行 + 约束遵循 — 测试用例生成脚本

2题 × 2模式 = 4个用例。代码支持扩展。

用法：
    cd tests/t4_rule
    python3 generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from reference import load_klines_with_indicators, judge_q3, judge_q4

OUTPUT_DIR = Path(__file__).resolve().parent

DATA_FILES = [
    ("20240101", "btc_usdt_1h_20240101.csv"),
]
KLINE_LIMIT = 50


# ============================================================
# JSON Spec
# ============================================================

Q3_JSON_SPEC = {
    "type": "object",
    "fields": {
        "timeframe": {"type": "str", "required": True, "enum": ["1h", "4h", "1d"]},
        "signals": {"type": "list", "required": True},
    },
}

Q4_JSON_SPEC = {
    "type": "object",
    "fields": {
        "timeframe": {"type": "str", "required": True, "enum": ["1h", "4h", "1d"]},
        "signals": {"type": "list", "required": True},
        "market_context": {"type": "dict", "required": True},
    },
}

# ============================================================
# Constraint definitions
# ============================================================

Q3_CONSTRAINTS = [
    {"id": "C1", "rule": "timeframe must be '1h'"},
    {"id": "C2", "rule": "only analyze the last 5 candles"},
    {"id": "C3", "rule": "SELL signal takes priority over BUY for the same candle"},
    {"id": "C4", "rule": "confidence must be between 0.0 and 1.0"},
]

Q4_CONSTRAINTS = [
    {"id": "C1", "rule": "timeframe must be '1h'"},
    {"id": "C2", "rule": "do NOT use volume data for any judgment"},
    {"id": "C3", "rule": "market_context.trend must be one of: bullish, bearish, sideways"},
    {"id": "C4", "rule": "market_context.volatility must be one of: high, medium, low"},
    {"id": "C5", "rule": "signals array must contain at most 1 element (analyze only the last candle)"},
]


# ============================================================
# Prompt builders
# ============================================================

def format_candle_table(candles: list[dict], last_n: int | None = None) -> str:
    subset = candles[-last_n:] if last_n else candles
    lines = ["Row,Timestamp,Close,RSI(14),MACD_DIF,MACD_DEA,MACD_HIST,EMA20,BB_Upper,BB_Lower"]
    for c in subset:
        lines.append(
            f"{c['row']},{c['timestamp']},{c['close']:.2f},"
            f"{c['rsi'] if c['rsi'] is not None else 'N/A'},"
            f"{c['macd_dif'] if c['macd_dif'] is not None else 'N/A'},"
            f"{c['macd_dea'] if c['macd_dea'] is not None else 'N/A'},"
            f"{c['macd_hist'] if c['macd_hist'] is not None else 'N/A'},"
            f"{c['ema20'] if c['ema20'] is not None else 'N/A'},"
            f"{c['bb_upper'] if c['bb_upper'] is not None else 'N/A'},"
            f"{c['bb_lower'] if c['bb_lower'] is not None else 'N/A'}"
        )
    return "\n".join(lines)


Q3_RULES_FULL = """## Trading Rules

Group A (BUY): RSI(14) < 35 AND MACD_DIF > MACD_DEA (golden cross)
Group B (BUY): Close < BB_Lower AND EMA20 > Close (price below lower Bollinger Band and below EMA20)
Group C (SELL): RSI(14) > 65 AND MACD_DIF < MACD_DEA (death cross)
Group D (SELL): Close > BB_Upper AND EMA20 < Close (price above upper Bollinger Band and above EMA20)

Signal logic: If Group C OR Group D triggers → SELL. If Group A OR Group B triggers → BUY. Otherwise → HOLD.

## Constraints
1. timeframe must be "1h"
2. Only analyze the last 5 candles for signals
3. For each candle, if both SELL and BUY conditions trigger, SELL takes priority
4. confidence must be between 0.0 and 1.0 (set based on how many rule groups triggered)"""

Q3_RULES_BRIEF = """Determine BUY/SELL/HOLD signals based on these conditions:
- BUY if: (RSI<35 AND MACD golden cross) OR (Close < BB_Lower AND Close < EMA20)
- SELL if: (RSI>65 AND MACD death cross) OR (Close > BB_Upper AND Close > EMA20)
- SELL priority over BUY. Analyze last 5 candles only. timeframe="1h"."""

Q4_RULES_FULL = """## Trading Rules (Nested)

### Step 1: Determine Market Trend (based on last 10 candles)
- If EMA20 of last candle > EMA20 of 10th-from-last candle, AND last Close > EMA20 → "bullish"
- If EMA20 of last candle < EMA20 of 10th-from-last candle, AND last Close < EMA20 → "bearish"
- Otherwise → "sideways"

### Step 2: Generate Signal (based on trend, analyze ONLY the last candle)
- Bullish trend:
  - RSI < 40 AND MACD_HIST > 0 → BUY (pullback entry)
  - RSI > 80 → SELL (overbought exit)
  - Otherwise → HOLD

- Bearish trend:
  - RSI > 60 AND MACD_HIST < 0 → SELL (bounce short)
  - RSI < 20 → BUY (oversold bottom)
  - Otherwise → HOLD

- Sideways:
  - Close < BB_Lower → BUY
  - Close > BB_Upper → SELL
  - Otherwise → HOLD

## Constraints
1. timeframe must be "1h"
2. Do NOT use volume data for any judgment
3. market_context.trend must be exactly one of: "bullish", "bearish", "sideways"
4. market_context.volatility must be exactly one of: "high", "medium", "low"
5. signals array must contain at most 1 element (only the last candle)"""

Q4_RULES_BRIEF = """Nested trading logic:
Step 1 — Trend: Compare EMA20 over last 10 candles + Close vs EMA20 → bullish/bearish/sideways
Step 2 — Signal based on trend:
  bullish: RSI<40 & MACD_HIST>0 → BUY; RSI>80 → SELL
  bearish: RSI>60 & MACD_HIST<0 → SELL; RSI<20 → BUY
  sideways: Close<BB_Lower → BUY; Close>BB_Upper → SELL
Constraints: timeframe="1h", no volume data, max 1 signal, include market_context with trend and volatility."""

OUTPUT_FORMAT = """\nReturn your answer in this exact JSON format:
```
{
  "timeframe": "1h",
  "signals": [
    {
      "timestamp": "YYYY-MM-DD HH:MM:SS",
      "type": "BUY|SELL|HOLD",
      "confidence": 0.0-1.0,
      "triggered_rules": ["rule_description"]
    }
  ],
  "market_context": {
    "trend": "bullish|bearish|sideways",
    "volatility": "high|medium|low"
  },
  "constraints_acknowledged": ["C1_description", "C2_description"]
}
```
Return ONLY the JSON, no explanation."""


def generate():
    cases = []

    for tag, filename in DATA_FILES:
        candles = load_klines_with_indicators(filename, KLINE_LIMIT)
        q3_answer = judge_q3(candles)
        q4_answer = judge_q4(candles)

        table_last10 = format_candle_table(candles, last_n=10)

        # Q3 with full rules
        cases.append({
            "id": f"t4_q3_with_rules_{tag}",
            "prompt": (
                "You are a trading signal analyzer Skill. Analyze the market data and determine trading signals.\n\n"
                f"{Q3_RULES_FULL}\n\n"
                f"## Market Data (BTC/USDT 1H, last 10 candles with pre-computed indicators)\n{table_last10}\n"
                f"{OUTPUT_FORMAT}"
            ),
            "system_prompt": "You are a quantitative trading agent. Follow all rules and constraints exactly. Return only JSON.",
            "expected": q3_answer,
            "scoring_method": "rule_signal",
            "scoring_params": {"constraints": Q3_CONSTRAINTS},
            "metadata": {"task": "q3", "with_rules": True, "data_segment": tag},
            "json_spec": Q3_JSON_SPEC,
        })

        # Q3 brief
        cases.append({
            "id": f"t4_q3_no_rules_{tag}",
            "prompt": (
                "You are a trading signal analyzer. Analyze the market data and determine trading signals.\n\n"
                f"{Q3_RULES_BRIEF}\n\n"
                f"## Market Data (BTC/USDT 1H, last 10 candles)\n{table_last10}\n"
                f"{OUTPUT_FORMAT}"
            ),
            "system_prompt": "You are a quantitative trading agent. Follow all rules and constraints exactly. Return only JSON.",
            "expected": q3_answer,
            "scoring_method": "rule_signal",
            "scoring_params": {"constraints": Q3_CONSTRAINTS},
            "metadata": {"task": "q3", "with_rules": False, "data_segment": tag},
            "json_spec": Q3_JSON_SPEC,
        })

        # Q4 with full rules
        cases.append({
            "id": f"t4_q4_with_rules_{tag}",
            "prompt": (
                "You are a trading signal analyzer Skill. Analyze the market data using nested trading logic.\n\n"
                f"{Q4_RULES_FULL}\n\n"
                f"## Market Data (BTC/USDT 1H, last 10 candles with pre-computed indicators)\n{table_last10}\n"
                f"{OUTPUT_FORMAT}"
            ),
            "system_prompt": "You are a quantitative trading agent. Follow all rules and constraints exactly. Return only JSON.",
            "expected": q4_answer,
            "scoring_method": "rule_signal",
            "scoring_params": {"constraints": Q4_CONSTRAINTS},
            "metadata": {"task": "q4", "with_rules": True, "data_segment": tag},
            "json_spec": Q4_JSON_SPEC,
        })

        # Q4 brief
        cases.append({
            "id": f"t4_q4_no_rules_{tag}",
            "prompt": (
                "You are a trading signal analyzer. Analyze the market data using nested trading logic.\n\n"
                f"{Q4_RULES_BRIEF}\n\n"
                f"## Market Data (BTC/USDT 1H, last 10 candles)\n{table_last10}\n"
                f"{OUTPUT_FORMAT}"
            ),
            "system_prompt": "You are a quantitative trading agent. Follow all rules and constraints exactly. Return only JSON.",
            "expected": q4_answer,
            "scoring_method": "rule_signal",
            "scoring_params": {"constraints": Q4_CONSTRAINTS},
            "metadata": {"task": "q4", "with_rules": False, "data_segment": tag},
            "json_spec": Q4_JSON_SPEC,
        })

    # Save candle data for reference
    for tag, filename in DATA_FILES:
        candles = load_klines_with_indicators(filename, KLINE_LIMIT)
        with open(OUTPUT_DIR / f"candles_{tag}.json", "w") as f:
            json.dump(candles, f, indent=2, ensure_ascii=False)

    output = {"dimension": "t4", "cases": cases}
    with open(OUTPUT_DIR / "rules.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Generated {len(cases)} T4 test cases -> rules.yaml")


if __name__ == "__main__":
    generate()
    print("All T4 test cases generated.")
