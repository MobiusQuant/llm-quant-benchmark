"""
T5+T6 数据处理 + 长上下文 — 标准参考实现

在真实K线数据中注入已知异常，生成不同长度的测试数据。
异常位置和类型完全确定，作为标准答案。
"""

from __future__ import annotations

import csv
import copy
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

ANOMALY_TYPES = {
    "missing_value": "Missing value (close is empty)",
    "ohlc_logic_error": "OHLC logic error (high < low)",
    "duplicate_timestamp": "Duplicate timestamp",
    "price_spike": "Abnormal price spike (deviates significantly from surrounding candles)",
    "volume_anomaly": "Volume anomaly (negative value)",
}


def load_raw_klines(filename: str, limit: int) -> list[dict]:
    rows = []
    with open(DATA_DIR / filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append({
                "row": i + 1,
                "timestamp": row["timestamp"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
    return rows


def inject_anomalies(klines: list[dict], positions: dict[int, str]) -> tuple[list[dict], list[dict]]:
    """
    在指定位置注入异常。
    positions: {row_index: anomaly_type}
    返回：(修改后的klines, 异常标准答案列表)
    """
    data = copy.deepcopy(klines)
    answers = []

    for idx, atype in positions.items():
        if idx >= len(data):
            continue
        row = data[idx]

        if atype == "missing_value":
            row["close"] = ""
            answers.append({"row": row["row"], "type": "missing_value",
                           "detail": "close field is empty"})

        elif atype == "ohlc_logic_error":
            row["high"], row["low"] = row["low"], row["high"]
            answers.append({"row": row["row"], "type": "ohlc_logic_error",
                           "detail": f"high ({row['high']}) < low ({row['low']})"})

        elif atype == "duplicate_timestamp":
            if idx > 0:
                row["timestamp"] = data[idx - 1]["timestamp"]
                answers.append({"row": row["row"], "type": "duplicate_timestamp",
                               "detail": f"same timestamp as row {data[idx-1]['row']}"})

        elif atype == "price_spike":
            original = row["close"]
            row["close"] = round(original * 0.1, 2)
            row["low"] = min(row["low"], row["close"])
            answers.append({"row": row["row"], "type": "price_spike",
                           "detail": f"close changed from ~{original:.0f} to {row['close']}"})

        elif atype == "volume_anomaly":
            row["volume"] = -500.0
            answers.append({"row": row["row"], "type": "volume_anomaly",
                           "detail": "volume is negative (-500)"})

    return data, answers


def get_anomaly_positions(total_rows: int) -> dict[int, str]:
    """根据数据长度，在头部/中部/尾部均匀分布5个异常"""
    types = ["missing_value", "ohlc_logic_error", "duplicate_timestamp",
             "price_spike", "volume_anomaly"]

    step = total_rows // 6
    positions = {}
    for i, atype in enumerate(types):
        pos = step * (i + 1)
        pos = min(pos, total_rows - 2)
        positions[pos] = atype

    return positions


def format_kline_csv(klines: list[dict]) -> str:
    lines = ["Row,Timestamp,Open,High,Low,Close,Volume"]
    for row in klines:
        c = row["close"]
        close_str = f"{c:.2f}" if isinstance(c, (int, float)) and c != "" else str(c)
        lines.append(
            f"{row['row']},{row['timestamp']},"
            f"{row['open']:.2f},{row['high']:.2f},"
            f"{row['low']:.2f},{close_str},"
            f"{row['volume']:.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    for length in [50, 200, 500]:
        klines = load_raw_klines("btc_usdt_1h_20240101.csv", length)
        positions = get_anomaly_positions(length)
        data, answers = inject_anomalies(klines, positions)
        print(f"\n=== {length} candles, {len(answers)} anomalies ===")
        for a in answers:
            print(f"  Row {a['row']}: {a['type']} — {a['detail']}")
