"""
T1 指标计算 — 标准参考实现

提供 SMA、RSI、MACD 的标准计算函数。
这些函数用于生成测试用例的标准答案，也供开源社区参考和验证。
"""

from __future__ import annotations

import pandas as pd


def sma(closes: pd.Series, period: int) -> float:
    """简单移动平均线 (Simple Moving Average)

    计算方式：取最近 period 个收盘价的算术平均值
    SMA = (C1 + C2 + ... + Cn) / n
    """
    return round(float(closes.tail(period).mean()), 2)


def rsi(closes: pd.Series, period: int = 14) -> float:
    """相对强弱指数 (Relative Strength Index)

    计算步骤：
    1. 计算每根K线相对前一根的价格变动 delta = close[i] - close[i-1]
    2. 分离涨幅(gain)和跌幅(loss)
    3. 第一个周期：avg_gain = mean(gains[:period]), avg_loss = mean(losses[:period])
    4. 后续周期：avg_gain = (prev_avg_gain * (period-1) + current_gain) / period
                 avg_loss = (prev_avg_loss * (period-1) + current_loss) / period
    5. RS = avg_gain / avg_loss
    6. RSI = 100 - (100 / (1 + RS))
    """
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.iloc[1:period + 1].mean()
    avg_loss = loss.iloc[1:period + 1].mean()

    for i in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = float(avg_gain) / float(avg_loss)
    return round(100 - (100 / (1 + rs)), 2)


def macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict:
    """MACD 指标 (Moving Average Convergence Divergence)

    计算步骤：
    1. 计算快线EMA：EMA(close, fast_period)  默认12
    2. 计算慢线EMA：EMA(close, slow_period)  默认26
    3. DIF = 快线EMA - 慢线EMA
    4. DEA = EMA(DIF, signal_period)  默认9
    5. MACD柱 = (DIF - DEA) * 2

    EMA计算方式：
    - 第一个值 = 前N个收盘价的SMA
    - 后续值：EMA = close * multiplier + prev_EMA * (1 - multiplier)
    - multiplier = 2 / (period + 1)
    """
    ema_fast = _ema_series(closes, fast)
    ema_slow = _ema_series(closes, slow)

    dif = ema_fast - ema_slow
    dea = _ema_series(dif.dropna(), signal)

    histogram = (dif.iloc[-1] - dea.iloc[-1]) * 2

    return {
        "DIF": round(float(dif.iloc[-1]), 2),
        "DEA": round(float(dea.iloc[-1]), 2),
        "MACD": round(float(histogram), 2),
    }


def _ema_series(series: pd.Series, period: int) -> pd.Series:
    """计算EMA序列"""
    values = series.values.astype(float)
    multiplier = 2.0 / (period + 1)
    ema = pd.Series(index=series.index, dtype=float)

    ema.iloc[period - 1] = values[:period].mean()
    for i in range(period, len(values)):
        ema.iloc[i] = values[i] * multiplier + ema.iloc[i - 1] * (1 - multiplier)
    return ema


if __name__ == "__main__":
    df = pd.read_csv("../../data/btc_usdt_1h_20240101.csv")
    closes = df["close"].astype(float).head(30)

    print(f"SMA(10):  {sma(closes, 10)}")
    print(f"RSI(14):  {rsi(closes, 14)}")

    closes_40 = df["close"].astype(float).head(40)
    result = macd(closes_40)
    print(f"MACD:     DIF={result['DIF']}, DEA={result['DEA']}, MACD={result['MACD']}")
