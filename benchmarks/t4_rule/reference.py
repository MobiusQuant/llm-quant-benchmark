"""
T4+T7 规则执行 + 约束遵循 — 标准参考实现

提供交易信号判定的标准答案生成。
给定交易规则 + 指标数据，确定性地判断信号是否触发。
"""

from __future__ import annotations

import csv
import httpx
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MOBIUSQUANT_URL = "https://api.mobiusquant.ai/api/indicators/compute"


def load_klines_with_indicators(filename: str, limit: int = 50) -> list[dict]:
    """加载K线并通过MobiusQuant API计算指标，返回带指标的数据列表"""
    klines = []
    with open(DATA_DIR / filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            ts = int(datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
            klines.append([ts, float(row["open"]), float(row["high"]),
                           float(row["low"]), float(row["close"]), float(row["volume"])])

    resp = httpx.post(MOBIUSQUANT_URL,
        headers={"Content-Type": "application/json"},
        json={"klines": klines, "interval": "1h", "echo_klines": True,
              "calc": [
                  {"name": "rsi", "params": {"period": 14}, "id": "rsi"},
                  {"name": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}, "id": "macd"},
                  {"name": "ema", "params": {"period": 20}, "id": "ema20"},
                  {"name": "bollinger", "params": {"period": 20, "std_dev": 2}, "id": "bb"},
              ]},
        timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for i, k in enumerate(data["klines"]):
        ts = datetime.fromtimestamp(k[0] / 1000).strftime("%Y-%m-%d %H:%M:%S")
        r = data["indicators"]["rsi"]["data"][i]
        m = data["indicators"]["macd"]["data"][i]
        e = data["indicators"]["ema20"]["data"][i]
        b = data["indicators"]["bb"]["data"][i]
        results.append({
            "row": i + 1,
            "timestamp": ts,
            "open": round(k[1], 2),
            "high": round(k[2], 2),
            "low": round(k[3], 2),
            "close": round(k[4], 2),
            "volume": round(k[5], 4),
            "rsi": round(r[1], 2) if r[1] is not None else None,
            "macd_dif": round(m[1], 2) if m[1] is not None else None,
            "macd_dea": round(m[2], 2) if m[2] is not None else None,
            "macd_hist": round(m[3], 2) if m[3] is not None else None,
            "ema20": round(e[1], 2) if e[1] is not None else None,
            "bb_upper": round(b[1], 2) if b[1] is not None else None,
            "bb_lower": round(b[3], 2) if b[3] is not None else None,
        })
    return results


# ============================================================
# Q3: OR+AND 组合条件判定
# ============================================================

def judge_q3(candles: list[dict]) -> dict:
    """
    规则：
    条件组A：RSI < 35 且 MACD_DIF > MACD_DEA（金叉）
    条件组B：Close < BB_Lower 且 EMA20 > Close（价格跌破布林下轨且低于EMA20）
    触发 A 或 B → BUY

    条件组C：RSI > 65 且 MACD_DIF < MACD_DEA（死叉）
    条件组D：Close > BB_Upper 且 EMA20 < Close（价格突破布林上轨且高于EMA20）
    触发 C 或 D → SELL

    否则 → HOLD

    约束：
    1. timeframe 必须填 "1h"
    2. 只分析最后5根K线的信号
    3. 每根K线只能有一个信号（SELL优先于BUY）
    4. confidence 必须在 0.0-1.0 之间
    """
    signals = []
    last_5 = candles[-5:]

    for c in last_5:
        if c["rsi"] is None or c["macd_dif"] is None or c["bb_upper"] is None:
            signals.append({"timestamp": c["timestamp"], "type": "HOLD", "triggered_rules": []})
            continue

        sell_rules = []
        buy_rules = []

        # Group C: RSI > 65 AND DIF < DEA
        if c["rsi"] > 65 and c["macd_dif"] < c["macd_dea"]:
            sell_rules.append("C_rsi_gt_65_and_macd_death_cross")
        # Group D: Close > BB_Upper AND EMA20 < Close
        if c["close"] > c["bb_upper"] and c["ema20"] < c["close"]:
            sell_rules.append("D_close_gt_bb_upper_and_above_ema20")

        # Group A: RSI < 35 AND DIF > DEA
        if c["rsi"] < 35 and c["macd_dif"] > c["macd_dea"]:
            buy_rules.append("A_rsi_lt_35_and_macd_golden_cross")
        # Group B: Close < BB_Lower AND EMA20 > Close
        if c["close"] < c["bb_lower"] and c["ema20"] > c["close"]:
            buy_rules.append("B_close_lt_bb_lower_and_below_ema20")

        # SELL priority over BUY
        if sell_rules:
            signals.append({"timestamp": c["timestamp"], "type": "SELL", "triggered_rules": sell_rules})
        elif buy_rules:
            signals.append({"timestamp": c["timestamp"], "type": "BUY", "triggered_rules": buy_rules})
        else:
            signals.append({"timestamp": c["timestamp"], "type": "HOLD", "triggered_rules": []})

    return {"signals": signals, "timeframe": "1h"}


# ============================================================
# Q4: 嵌套条件 + 优先级
# ============================================================

def judge_q4(candles: list[dict]) -> dict:
    """
    规则（嵌套）：
    Step 1 — 判断趋势方向（基于最后10根K线）：
      - 若 EMA20 持续上升（最后一根 > 第一根）且 最后一根 Close > EMA20 → 上升趋势
      - 若 EMA20 持续下降（最后一根 < 第一根）且 最后一根 Close < EMA20 → 下降趋势
      - 否则 → 震荡

    Step 2 — 基于趋势判断信号（只看最后一根K线）：
      上升趋势：
        - RSI < 40 且 MACD_HIST > 0 → BUY (回调买入)
        - RSI > 80 → SELL (超买离场)
        - 否则 → HOLD

      下降趋势：
        - RSI > 60 且 MACD_HIST < 0 → SELL (反弹做空)
        - RSI < 20 → BUY (超卖抄底)
        - 否则 → HOLD

      震荡：
        - Close < BB_Lower → BUY
        - Close > BB_Upper → SELL
        - 否则 → HOLD

    约束：
    1. timeframe 必须填 "1h"
    2. 不得使用 volume 数据做判断依据
    3. market_context.trend 必须是 "bullish"/"bearish"/"sideways" 之一
    4. market_context.volatility 必须是 "high"/"medium"/"low" 之一
    5. signals 数组最多1个元素（只看最后一根K线）
    """
    last_10 = candles[-10:]
    last = candles[-1]

    if last["ema20"] is None or last["rsi"] is None:
        return {
            "signals": [{"timestamp": last["timestamp"], "type": "HOLD", "triggered_rules": []}],
            "timeframe": "1h",
            "market_context": {"trend": "sideways", "volatility": "medium"},
        }

    # Step 1: Trend
    ema_first = last_10[0]["ema20"]
    ema_last = last_10[-1]["ema20"]
    if ema_first and ema_last:
        if ema_last > ema_first and last["close"] > last["ema20"]:
            trend = "bullish"
        elif ema_last < ema_first and last["close"] < last["ema20"]:
            trend = "bearish"
        else:
            trend = "sideways"
    else:
        trend = "sideways"

    # Volatility based on BB width
    if last["bb_upper"] and last["bb_lower"]:
        bb_width = (last["bb_upper"] - last["bb_lower"]) / last["close"]
        if bb_width > 0.02:
            volatility = "high"
        elif bb_width > 0.01:
            volatility = "medium"
        else:
            volatility = "low"
    else:
        volatility = "medium"

    # Step 2: Signal
    signal_type = "HOLD"
    rules = []

    if trend == "bullish":
        if last["rsi"] < 40 and last["macd_hist"] and last["macd_hist"] > 0:
            signal_type = "BUY"
            rules = ["bullish_pullback_rsi_lt_40_macd_hist_gt_0"]
        elif last["rsi"] > 80:
            signal_type = "SELL"
            rules = ["bullish_overbought_rsi_gt_80"]
    elif trend == "bearish":
        if last["rsi"] > 60 and last["macd_hist"] and last["macd_hist"] < 0:
            signal_type = "SELL"
            rules = ["bearish_bounce_rsi_gt_60_macd_hist_lt_0"]
        elif last["rsi"] < 20:
            signal_type = "BUY"
            rules = ["bearish_oversold_rsi_lt_20"]
    else:
        if last["bb_lower"] and last["close"] < last["bb_lower"]:
            signal_type = "BUY"
            rules = ["sideways_close_lt_bb_lower"]
        elif last["bb_upper"] and last["close"] > last["bb_upper"]:
            signal_type = "SELL"
            rules = ["sideways_close_gt_bb_upper"]

    return {
        "signals": [{"timestamp": last["timestamp"], "type": signal_type, "triggered_rules": rules}],
        "timeframe": "1h",
        "market_context": {"trend": trend, "volatility": volatility},
    }


if __name__ == "__main__":
    candles = load_klines_with_indicators("btc_usdt_1h_20240101.csv", 50)
    print(f"Loaded {len(candles)} candles")
    print(f"Last candle: {candles[-1]['timestamp']} C={candles[-1]['close']} RSI={candles[-1]['rsi']}")

    q3 = judge_q3(candles)
    print(f"\nQ3 signals:")
    for s in q3["signals"]:
        print(f"  {s['timestamp']} -> {s['type']}  rules={s['triggered_rules']}")

    q4 = judge_q4(candles)
    print(f"\nQ4 trend={q4['market_context']['trend']} volatility={q4['market_context']['volatility']}")
    for s in q4["signals"]:
        print(f"  {s['timestamp']} -> {s['type']}  rules={s['triggered_rules']}")
