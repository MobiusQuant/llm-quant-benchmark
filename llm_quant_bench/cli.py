from __future__ import annotations

import argparse
import asyncio

from llm_quant_bench.client import LLMClient
from llm_quant_bench.config import PROJECT_ROOT, get_api_key, load_config
from llm_quant_bench.loader import load_all
from llm_quant_bench.report import print_report, save_results
from llm_quant_bench.runner import BenchmarkRunner

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "openrouter.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llm-quant-bench",
        description="Run the LLM quant-trading benchmark.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to a provider config YAML (default: configs/openrouter.yaml)",
    )
    parser.add_argument("--dimensions", nargs="+", help="Dimensions to run, e.g. t1 t2 t4 t5")
    parser.add_argument("--models", nargs="+", help="Model IDs to test (overrides config)")
    parser.add_argument("--rounds", type=int, help="Rounds per test case (overrides config)")
    parser.add_argument("--resume", help="Resume a previous run by run_id")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-call details")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    provider = cfg.get("provider", {})
    api_key = get_api_key(provider.get("api_key_env", "OPENROUTER_API_KEY"))

    if args.dimensions:
        cfg.setdefault("run", {})["dimensions"] = args.dimensions
    if args.models:
        cfg["models"] = [{"id": m, "label": m} for m in args.models]
    if args.rounds:
        cfg.setdefault("run", {})["rounds"] = args.rounds

    dimensions = cfg.get("run", {}).get("dimensions", ["t1", "t2", "t4", "t5"])
    test_cases = load_all(dimensions=dimensions)
    if not test_cases:
        print("No test cases found. Check the benchmarks/ directory and --dimensions.")
        return

    n_dims = len({tc.dimension for tc in test_cases})
    print(f"Loaded {len(test_cases)} test cases across {n_dims} dimension(s)")
    print(f"Testing {len(cfg['models'])} model(s)\n")

    client = LLMClient(
        api_key=api_key,
        base_url=provider.get("base_url", "https://openrouter.ai/api/v1"),
        timeout=provider.get("timeout", 120),
        max_retries=provider.get("max_retries", 2),
        temperature=provider.get("temperature", 0.0),
        max_tokens=provider.get("max_tokens"),
        extra_body=provider.get("extra_body"),
    )
    runner = BenchmarkRunner(client, cfg)
    try:
        result = await runner.run(test_cases, cfg["models"], resume_run_id=args.resume)
        path = save_results(result)
        print(f"\nResults saved to {path}\n")
        print_report(result, verbose=args.verbose)
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_run(parse_args(argv)))


if __name__ == "__main__":
    main()
