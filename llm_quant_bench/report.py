from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_quant_bench.config import PROJECT_ROOT
from llm_quant_bench.models import RunResult

DIMENSION_LABELS = {
    "t1": "T1 计算",
    "t2": "T2 分析",
    "t3": "T3 JSON",
    "t4": "T4 规则",
    "t5": "T5 数据",
    "t6": "T6 长文本",
    "t7": "T7 约束",
}


def save_results(run_result: RunResult, output_dir: str | Path | None = None):
    """
    结果归档结构：
    results/
    └── {run_id}/
        ├── summary.json                          # 汇总
        ├── t1/                                   # 按维度
        │   ├── claude-opus-4.8/                  # 按模型
        │   │   ├── t1_sma_with_formula_20240101/ # 按用例
        │   │   │   ├── round_1.json
        │   │   │   ├── round_2.json
        │   │   │   └── round_3.json
        │   │   └── ...
        │   └── ...
        └── ...
    """
    out = Path(output_dir) if output_dir else PROJECT_ROOT / "results"
    run_dir = out / run_result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save per-round detail files
    for s in run_result.scores:
        model_slug = s.model_id.replace("/", "_")
        case_dir = run_dir / s.dimension / model_slug / s.test_case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        round_num = s.details.get("round", 1)
        detail_path = case_dir / f"round_{round_num}.json"
        detail_path.write_text(json.dumps(s.model_dump(), ensure_ascii=False, indent=2))

    # Save summary
    summary_path = run_dir / "summary.json"
    summary_data = {
        "run_id": run_result.run_id,
        "summary": run_result.summary,
    }
    summary_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2))

    return run_dir


def print_report(run_result: RunResult, verbose: bool = False):
    console = Console()
    run_cfg = run_result.summary.pop("_run_config", {})
    rounds = run_cfg.get("rounds", 1)

    if run_cfg:
        console.print(f"\n[dim]Rounds: {rounds} | Cases: {run_cfg.get('total_cases', '?')} | "
                       f"Total calls: {run_cfg.get('total_calls', '?')}[/dim]")

    dims_in_run = sorted({s.dimension for s in run_result.scores})

    table = Table(title="LLM Quant Benchmark Results", show_lines=True)
    table.add_column("Model", style="bold", min_width=18)
    for dim in dims_in_run:
        table.add_column(DIMENSION_LABELS.get(dim, dim), justify="center", min_width=8)
    table.add_column("Average", justify="center", style="bold", min_width=8)
    table.add_column("JSON", justify="center", min_width=6)
    table.add_column("Cost($)", justify="right", min_width=8)
    table.add_column("Tokens", justify="right", min_width=10)

    for model_id, info in run_result.summary.items():
        if model_id.startswith("_"):
            continue
        row = [info["label"]]
        for dim in dims_in_run:
            val = info["dimensions"].get(dim, 0.0)
            color = "green" if val >= 0.8 else "yellow" if val >= 0.6 else "red"
            row.append(f"[{color}]{val:.2f}[/{color}]")
        avg = info["average"]
        avg_color = "green" if avg >= 0.8 else "yellow" if avg >= 0.6 else "red"
        row.append(f"[{avg_color}]{avg:.2f}[/{avg_color}]")
        jc = info.get("json_compliance", 0.0)
        jc_color = "green" if jc >= 0.8 else "yellow" if jc >= 0.6 else "red"
        row.append(f"[{jc_color}]{jc:.2f}[/{jc_color}]")
        cost = info["token_usage"].get("total_cost_usd", 0)
        row.append(f"{cost:.4f}")
        row.append(f"{info['token_usage']['total_tokens']:,}")
        table.add_row(*row)

    console.print(table)

    if rounds > 1:
        _print_pass_rate_table(console, run_result)

    run_result.summary["_run_config"] = run_cfg

    if verbose:
        _print_details(console, run_result)


def _print_pass_rate_table(console: Console, run_result: RunResult):
    console.print(f"\n[bold]Per-case pass rate:[/bold]\n")

    for model_id, info in run_result.summary.items():
        if model_id.startswith("_"):
            continue
        console.print(f"  [bold]{info['label']}[/bold]")
        case_rates = info.get("case_pass_rates", {})
        for case_id, rate_info in sorted(case_rates.items()):
            pr = rate_info["pass_rate"]
            scores_str = " ".join(f"{s:.0f}" if s in (0, 1) else f"{s:.2f}" for s in rate_info["scores"])
            color = "green" if pr >= 1.0 else "yellow" if pr > 0 else "red"
            console.print(f"    [{color}]{pr:.0%}[/{color}]  {case_id:40s}  rounds=[{scores_str}]")
        console.print()


def _print_details(console: Console, run_result: RunResult):
    console.print("\n[bold]Per-call details:[/bold]\n")
    for s in run_result.scores:
        rd = s.details.get("round", "?")
        color = "green" if s.score >= 0.8 else "yellow" if s.score >= 0.6 else "red"
        console.print(
            f"  R{rd} [{color}]{s.score:.2f}[/{color}]  "
            f"[dim]{s.model_id}[/dim]  {s.test_case_id}  "
            f"latency={s.details.get('latency_ms', 0):.0f}ms  "
            f"tokens={s.details.get('token_usage', {}).get('total_tokens', 0)}"
        )
        raw = s.details.get("raw_response", "")
        if raw:
            console.print(f"         [dim]response: {raw[:100]}[/dim]")
