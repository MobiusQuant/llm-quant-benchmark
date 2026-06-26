"""
T5+T6 数据处理 + 长上下文 — 测试用例生成脚本

3个数据长度(50/200/500) × 1段数据 = 3个用例。

用法：
    cd tests/t5_anomaly
    python3 generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from reference import (
    load_raw_klines,
    get_anomaly_positions,
    inject_anomalies,
    format_kline_csv,
)

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_FILE = "btc_usdt_1h_20240101.csv"
LENGTHS = [50, 200, 500]

JSON_SPEC = {
    "type": "array",
    "array_item_fields": {
        "row": {"type": "int", "required": True},
        "type": {"type": "str", "required": True, "enum": [
            "missing_value", "ohlc_logic_error", "duplicate_timestamp",
            "price_spike", "volume_anomaly",
        ]},
        "detail": {"type": "str", "required": True},
    },
}

ANOMALY_DEFINITIONS = """## Anomaly Types to Look For

1. **missing_value**: A required field (open, high, low, close, volume) is empty or missing
2. **ohlc_logic_error**: OHLC logic violation — high should be >= low, and open/close should be within [low, high]
3. **duplicate_timestamp**: Two consecutive rows share the same timestamp
4. **price_spike**: A candle's price (close) deviates dramatically from surrounding candles (e.g., 10x drop or spike)
5. **volume_anomaly**: Volume is negative or clearly invalid"""

OUTPUT_FORMAT = """
Return your answer as a JSON array. Each anomaly should have:
{"row": <row_number>, "type": "<anomaly_type>", "detail": "<brief description>"}

anomaly_type must be exactly one of: missing_value, ohlc_logic_error, duplicate_timestamp, price_spike, volume_anomaly

If no anomalies found, return: []
Return ONLY the JSON array, no explanation."""


def generate():
    cases = []

    for length in LENGTHS:
        klines = load_raw_klines(DATA_FILE, length)
        positions = get_anomaly_positions(length)
        data, answers = inject_anomalies(klines, positions)
        csv_text = format_kline_csv(data)

        # Save injected data and answers for reference
        with open(OUTPUT_DIR / f"injected_{length}.json", "w") as f:
            json.dump({"data": data, "answers": answers, "positions": {str(k): v for k, v in positions.items()}},
                      f, indent=2, ensure_ascii=False)

        cases.append({
            "id": f"t5_anomaly_{length}",
            "prompt": (
                f"Analyze the following BTC/USDT 1-hour K-line data ({length} candles) "
                f"and identify ALL anomalies.\n\n"
                f"{ANOMALY_DEFINITIONS}\n\n"
                f"## K-line Data ({length} candles)\n{csv_text}\n"
                f"{OUTPUT_FORMAT}"
            ),
            "system_prompt": "You are a data quality analyst. Find all anomalies in the data. Return only the JSON array.",
            "expected": answers,
            "scoring_method": "anomaly_f1",
            "scoring_params": {},
            "metadata": {"data_length": length, "anomaly_count": len(answers)},
            "json_spec": JSON_SPEC,
        })

    output = {"dimension": "t5", "cases": cases}
    with open(OUTPUT_DIR / "anomaly.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Generated {len(cases)} T5 test cases -> anomaly.yaml")


if __name__ == "__main__":
    generate()
    print("All T5+T6 test cases generated.")
