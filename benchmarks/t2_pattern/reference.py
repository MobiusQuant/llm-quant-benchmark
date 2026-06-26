"""
T2 K线结构识别 — 标准参考实现

提供高低点识别和SMC结构识别的标准答案生成。
SMC部分通过 MobiusQuant API 计算：https://docs.mobiusquant.ai/zh/compute.html
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

MOBIUSQUANT_URL = "https://api.mobiusquant.ai/api/indicators/compute"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_klines(filename: str, limit: int | None = None) -> tuple[pd.DataFrame, list]:
    """加载K线数据，返回DataFrame和API格式的klines列表"""
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)

    api_klines = []
    for _, row in df.iterrows():
        ts = int(datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        api_klines.append([
            ts,
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]),
        ])

    return df, api_klines


# ============================================================
# 高低点识别
# ============================================================

def find_highest(df: pd.DataFrame) -> dict:
    """找出最高价及其位置（基于high列）"""
    idx = df["high"].astype(float).idxmax()
    row = df.iloc[idx]
    return {
        "row": int(idx) + 1,
        "timestamp": row["timestamp"],
        "price": round(float(row["high"]), 2),
    }


def find_lowest(df: pd.DataFrame) -> dict:
    """找出最低价及其位置（基于low列）"""
    idx = df["low"].astype(float).idxmin()
    row = df.iloc[idx]
    return {
        "row": int(idx) + 1,
        "timestamp": row["timestamp"],
        "price": round(float(row["low"]), 2),
    }


# ============================================================
# SMC 结构识别 (via MobiusQuant API)
# ============================================================

def compute_smc(api_klines: list, swing_size: int = 10, internal_size: int = 5) -> dict:
    """调用MobiusQuant API计算SMC指标，返回objects结构化数据"""
    resp = httpx.post(
        MOBIUSQUANT_URL,
        headers={"Content-Type": "application/json"},
        json={
            "klines": api_klines,
            "interval": "1h",
            "calc": [{"name": "smc", "params": {"swing_size": swing_size, "internal_size": internal_size}, "id": "smc"}],
            "echo_klines": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["indicators"]["smc"]["objects"]


def extract_fvg(objects: dict) -> list[dict]:
    """提取FVG（公允价值缺口）列表"""
    return [
        {
            "timestamp": _ms_to_str(item["anchor_time"]),
            "top": round(item["top"], 2),
            "bottom": round(item["bottom"], 2),
            "bias": item["bias"],
        }
        for item in objects.get("fair_value_gaps", [])
    ]


def extract_order_blocks(objects: dict) -> list[dict]:
    """提取Order Block（订单块）列表，合并internal和swing"""
    seen = set()
    results = []
    for key in ["order_blocks_internal", "order_blocks_swing"]:
        for item in objects.get(key, []):
            uid = (item["anchor_time"], item["top"], item["bottom"])
            if uid in seen:
                continue
            seen.add(uid)
            results.append({
                "timestamp": _ms_to_str(item["anchor_time"]),
                "top": round(item["top"], 2),
                "bottom": round(item["bottom"], 2),
                "bias": item["bias"],
                "status": item["status"],
            })
    return results


def extract_bos_choch(objects: dict) -> list[dict]:
    """提取BOS/CHoCH结构突破事件，合并swing和internal"""
    seen = set()
    results = []
    for key in ["swing_structures", "internal_structures"]:
        for item in objects.get(key, []):
            uid = (item["pivot_time"], item["kind"], item["bias"])
            if uid in seen:
                continue
            seen.add(uid)
            results.append({
                "pivot_timestamp": _ms_to_str(item["pivot_time"]),
                "confirm_timestamp": _ms_to_str(item["confirm_time"]),
                "pivot_price": round(item["pivot_price"], 2),
                "kind": item["kind"],
                "bias": item["bias"],
            })
    return results


def extract_eqhl(objects: dict) -> list[dict]:
    """提取EQH/EQL（等值高低点）"""
    results = []
    for item in objects.get("equal_highs", []):
        results.append({
            "type": "EQH",
            "anchor_timestamp": _ms_to_str(item["anchor_time"]),
            "confirm_timestamp": _ms_to_str(item["confirm_time"]),
            "level": round(item["level"], 2),
        })
    for item in objects.get("equal_lows", []):
        results.append({
            "type": "EQL",
            "anchor_timestamp": _ms_to_str(item["anchor_time"]),
            "confirm_timestamp": _ms_to_str(item["confirm_time"]),
            "level": round(item["level"], 2),
        })
    return results


def _ms_to_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    df, api_klines = load_klines("btc_usdt_1h_20240101.csv", limit=200)

    print("=== High/Low ===")
    print(f"Highest: {find_highest(df)}")
    print(f"Lowest:  {find_lowest(df)}")

    print("\n=== SMC (via MobiusQuant API) ===")
    objects = compute_smc(api_klines)

    fvgs = extract_fvg(objects)
    print(f"FVG: {len(fvgs)} items")
    for f in fvgs[:3]:
        print(f"  {f}")

    obs = extract_order_blocks(objects)
    print(f"Order Blocks: {len(obs)} items")
    for o in obs[:3]:
        print(f"  {o}")

    bos = extract_bos_choch(objects)
    print(f"BOS/CHoCH: {len(bos)} items")
    for b in bos[:3]:
        print(f"  {b}")

    eqhl = extract_eqhl(objects)
    print(f"EQH/EQL: {len(eqhl)} items")
    for e in eqhl[:3]:
        print(f"  {e}")
