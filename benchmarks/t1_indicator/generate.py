"""
T1 指标计算 — 测试用例生成脚本

从真实BTC K线数据生成 SMA / RSI / MACD 的测试用例。
每个指标生成 6 个用例：3段数据 × 2种方式（给公式 / 不给公式）

用法：
    cd tests/t1_indicator
    python3 generate.py
"""

from __future__ import annotations

import yaml
import pandas as pd
from pathlib import Path

from reference import sma, rsi, macd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent

DATA_FILES = [
    ("20240101", "btc_usdt_1h_20240101.csv"),
    ("20240601", "btc_usdt_1h_20240601.csv"),
    ("20250101", "btc_usdt_1h_20250101.csv"),
]

SMA_PERIOD = 10
RSI_PERIOD = 14
MACD_PARAMS = (12, 26, 9)
KLINE_WINDOW = 30
MACD_KLINE_WINDOW = 40


def format_kline_table(df: pd.DataFrame) -> str:
    lines = ["timestamp,open,high,low,close,volume"]
    for _, row in df.iterrows():
        lines.append(
            f"{row['timestamp']},"
            f"{float(row['open']):.2f},"
            f"{float(row['high']):.2f},"
            f"{float(row['low']):.2f},"
            f"{float(row['close']):.2f},"
            f"{float(row['volume']):.4f}"
        )
    return "\n".join(lines)


def generate_sma():
    cases = []
    for tag, filename in DATA_FILES:
        df = pd.read_csv(DATA_DIR / filename)
        window = df.head(KLINE_WINDOW)
        closes = window["close"].astype(float)
        expected = sma(closes, SMA_PERIOD)
        kline_text = format_kline_table(window)

        cases.append({
            "id": f"t1_sma_with_formula_{tag}",
            "prompt": (
                f"Based on the following BTC/USDT 1-hour K-line data, "
                f"calculate the {SMA_PERIOD}-period Simple Moving Average (SMA) of the closing price.\n\n"
                f"Calculation method:\n"
                f"SMA = (C1 + C2 + ... + C{SMA_PERIOD}) / {SMA_PERIOD}\n"
                f"where C1 to C{SMA_PERIOD} are the last {SMA_PERIOD} closing prices.\n\n"
                f"K-line data:\n{kline_text}\n\n"
                f"Return ONLY the final SMA value as a number, rounded to 2 decimal places."
            ),
            "system_prompt": "You are a quantitative analyst. Return only the numeric result, no explanation.",
            "expected": expected,
            "scoring_method": "numeric_tolerance",
            "scoring_params": {"tolerance": round(float(expected) * 0.001, 2)},
            "metadata": {"indicator": "SMA", "period": SMA_PERIOD, "with_formula": True, "data_segment": tag},
        })

        cases.append({
            "id": f"t1_sma_no_formula_{tag}",
            "prompt": (
                f"Based on the following BTC/USDT 1-hour K-line data, "
                f"calculate the {SMA_PERIOD}-period SMA of the closing price.\n\n"
                f"K-line data:\n{kline_text}\n\n"
                f"Return ONLY the final SMA value as a number, rounded to 2 decimal places."
            ),
            "system_prompt": "You are a quantitative analyst. Return only the numeric result, no explanation.",
            "expected": expected,
            "scoring_method": "numeric_tolerance",
            "scoring_params": {"tolerance": round(float(expected) * 0.001, 2)},
            "metadata": {"indicator": "SMA", "period": SMA_PERIOD, "with_formula": False, "data_segment": tag},
        })

    output = {"dimension": "t1", "cases": cases}
    with open(OUTPUT_DIR / "sma.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Generated {len(cases)} SMA test cases -> sma.yaml")


def generate_rsi():
    cases = []
    for tag, filename in DATA_FILES:
        df = pd.read_csv(DATA_DIR / filename)
        window = df.head(KLINE_WINDOW)
        closes = window["close"].astype(float)
        expected = rsi(closes, RSI_PERIOD)
        kline_text = format_kline_table(window)

        cases.append({
            "id": f"t1_rsi_with_formula_{tag}",
            "prompt": (
                f"Based on the following BTC/USDT 1-hour K-line data, "
                f"calculate the {RSI_PERIOD}-period RSI of the closing price.\n\n"
                f"Calculation method:\n"
                f"1. delta = close[i] - close[i-1]\n"
                f"2. Separate gains (delta > 0) and losses (delta < 0, take absolute value)\n"
                f"3. First avg_gain = mean of first {RSI_PERIOD} gains, first avg_loss = mean of first {RSI_PERIOD} losses\n"
                f"4. Subsequent: avg_gain = (prev_avg_gain * {RSI_PERIOD - 1} + current_gain) / {RSI_PERIOD}\n"
                f"   avg_loss = (prev_avg_loss * {RSI_PERIOD - 1} + current_loss) / {RSI_PERIOD}\n"
                f"5. RS = avg_gain / avg_loss\n"
                f"6. RSI = 100 - (100 / (1 + RS))\n\n"
                f"K-line data:\n{kline_text}\n\n"
                f"Return ONLY the final RSI value as a number, rounded to 2 decimal places."
            ),
            "system_prompt": "You are a quantitative analyst. Return only the numeric result, no explanation.",
            "expected": expected,
            "scoring_method": "numeric_tolerance",
            "scoring_params": {"tolerance": 1.0},
            "metadata": {"indicator": "RSI", "period": RSI_PERIOD, "with_formula": True, "data_segment": tag},
        })

        cases.append({
            "id": f"t1_rsi_no_formula_{tag}",
            "prompt": (
                f"Based on the following BTC/USDT 1-hour K-line data, "
                f"calculate the {RSI_PERIOD}-period RSI of the closing price.\n\n"
                f"K-line data:\n{kline_text}\n\n"
                f"Return ONLY the final RSI value as a number, rounded to 2 decimal places."
            ),
            "system_prompt": "You are a quantitative analyst. Return only the numeric result, no explanation.",
            "expected": expected,
            "scoring_method": "numeric_tolerance",
            "scoring_params": {"tolerance": 1.0},
            "metadata": {"indicator": "RSI", "period": RSI_PERIOD, "with_formula": False, "data_segment": tag},
        })

    output = {"dimension": "t1", "cases": cases}
    with open(OUTPUT_DIR / "rsi.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Generated {len(cases)} RSI test cases -> rsi.yaml")


def generate_macd():
    fast, slow, signal = MACD_PARAMS
    cases = []
    for tag, filename in DATA_FILES:
        df = pd.read_csv(DATA_DIR / filename)
        window = df.head(MACD_KLINE_WINDOW)
        closes = window["close"].astype(float)
        result = macd(closes, fast, slow, signal)
        kline_text = format_kline_table(window)

        expected_str = f"DIF={result['DIF']}, DEA={result['DEA']}, MACD={result['MACD']}"

        cases.append({
            "id": f"t1_macd_with_formula_{tag}",
            "prompt": (
                f"Based on the following BTC/USDT 1-hour K-line data, "
                f"calculate the MACD({fast},{slow},{signal}) indicator.\n\n"
                f"Calculation method:\n"
                f"1. EMA_fast = EMA(close, {fast})\n"
                f"2. EMA_slow = EMA(close, {slow})\n"
                f"3. DIF = EMA_fast - EMA_slow\n"
                f"4. DEA = EMA(DIF, {signal})\n"
                f"5. MACD histogram = (DIF - DEA) * 2\n\n"
                f"EMA calculation:\n"
                f"- First EMA value = SMA of the first N closing prices\n"
                f"- Multiplier = 2 / (N + 1)\n"
                f"- EMA[i] = close[i] * multiplier + EMA[i-1] * (1 - multiplier)\n\n"
                f"K-line data:\n{kline_text}\n\n"
                f"Return the result in this exact format:\n"
                f"DIF=<value>, DEA=<value>, MACD=<value>\n"
                f"All values rounded to 2 decimal places."
            ),
            "system_prompt": "You are a quantitative analyst. Return only the result in the specified format, no explanation.",
            "expected": result,
            "scoring_method": "macd_tolerance",
            "scoring_params": {"tolerance_pct": 0.01},
            "metadata": {"indicator": "MACD", "params": [fast, slow, signal], "with_formula": True, "data_segment": tag},
        })

        cases.append({
            "id": f"t1_macd_no_formula_{tag}",
            "prompt": (
                f"Based on the following BTC/USDT 1-hour K-line data, "
                f"calculate the MACD({fast},{slow},{signal}) indicator.\n\n"
                f"K-line data:\n{kline_text}\n\n"
                f"Return the result in this exact format:\n"
                f"DIF=<value>, DEA=<value>, MACD=<value>\n"
                f"All values rounded to 2 decimal places."
            ),
            "system_prompt": "You are a quantitative analyst. Return only the result in the specified format, no explanation.",
            "expected": result,
            "scoring_method": "macd_tolerance",
            "scoring_params": {"tolerance_pct": 0.01},
            "metadata": {"indicator": "MACD", "params": [fast, slow, signal], "with_formula": False, "data_segment": tag},
        })

    output = {"dimension": "t1", "cases": cases}
    with open(OUTPUT_DIR / "macd.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Generated {len(cases)} MACD test cases -> macd.yaml")


if __name__ == "__main__":
    generate_sma()
    generate_rsi()
    generate_macd()
    print("\nAll T1 test cases generated.")
