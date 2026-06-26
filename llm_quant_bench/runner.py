from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from rich.progress import BarColumn, SpinnerColumn, Progress, TextColumn, TimeElapsedColumn

from llm_quant_bench.client import LLMClient
from llm_quant_bench.config import PROJECT_ROOT
from llm_quant_bench.models import ModelResponse, RunResult, ScoreResult, TestCase
from llm_quant_bench.scorer import score


class BenchmarkRunner:
    def __init__(self, client: LLMClient, config: dict):
        self.client = client
        self.config = config
        self.concurrency = config.get("run", {}).get("concurrency", 3)
        self.rounds = config.get("run", {}).get("rounds", 1)
        self.output_dir = PROJECT_ROOT / config.get("run", {}).get("output_dir", "results")

    def _round_path(self, run_id: str, model_id: str, dimension: str, test_case_id: str, round_num: int) -> Path:
        model_slug = model_id.replace("/", "_")
        return self.output_dir / run_id / dimension / model_slug / test_case_id / f"round_{round_num}.json"

    def _save_round(self, run_id: str, result: ScoreResult):
        path = self._round_path(run_id, result.model_id, result.dimension, result.test_case_id, result.details.get("round", 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

    def _load_existing(self, run_id: str, model_id: str, test_case_id: str, round_num: int, dimension: str) -> ScoreResult | None:
        path = self._round_path(run_id, model_id, dimension, test_case_id, round_num)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return ScoreResult(**data)
        except Exception:
            return None

    async def _run_one(
        self,
        model: dict,
        test_case: TestCase,
        round_num: int,
        run_id: str,
        semaphore: asyncio.Semaphore,
        progress: Progress,
        task_id,
    ) -> ScoreResult:
        existing = self._load_existing(run_id, model["id"], test_case.id, round_num, test_case.dimension)
        if existing is not None:
            progress.advance(task_id)
            return existing

        messages: list[dict] = []
        if test_case.system_prompt:
            messages.append({"role": "system", "content": test_case.system_prompt})
        messages.append({"role": "user", "content": test_case.prompt})

        async with semaphore:
            response: ModelResponse = await self.client.chat(
                model=model["id"],
                messages=messages,
                test_case_id=test_case.id,
            )
            progress.advance(task_id)

        result = score(test_case, response)
        result.details["round"] = round_num
        result.details["latency_ms"] = round(response.latency_ms, 1)
        result.details["token_usage"] = response.token_usage
        result.details["prompt"] = test_case.prompt
        result.details["system_prompt"] = test_case.system_prompt
        result.details["expected"] = test_case.expected
        result.details["scoring_method"] = test_case.scoring_method
        result.details["scoring_params"] = test_case.scoring_params
        result.details["metadata"] = test_case.metadata
        if response.error:
            result.details["error"] = response.error
        result.details["raw_response"] = response.raw_response

        self._save_round(run_id, result)
        return result

    async def run(
        self,
        test_cases: list[TestCase],
        models: list[dict],
        resume_run_id: str | None = None,
    ) -> RunResult:
        run_id = resume_run_id or self._generate_run_id(test_cases)
        semaphore = asyncio.Semaphore(self.concurrency)
        total = len(models) * len(test_cases) * self.rounds

        if resume_run_id:
            run_dir = self.output_dir / run_id
            existing_count = len(list(run_dir.rglob("round_*.json"))) if run_dir.exists() else 0
            print(f"Resuming run {run_id}: {existing_count}/{total} already completed")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task_id = progress.add_task(
                f"Running benchmark ({self.rounds} rounds)", total=total,
            )

            tasks = [
                self._run_one(model, tc, r + 1, run_id, semaphore, progress, task_id)
                for r in range(self.rounds)
                for model in models
                for tc in test_cases
            ]
            scores: list[ScoreResult] = await asyncio.gather(*tasks)

        summary = self._build_summary(scores, models)

        run_result = RunResult(run_id=run_id, scores=scores, summary=summary)
        run_result.summary["_run_config"] = {
            "rounds": self.rounds,
            "concurrency": self.concurrency,
            "dimensions": self.config.get("run", {}).get("dimensions", []),
            "models": [m["id"] for m in models],
            "total_cases": len(test_cases),
            "total_calls": total,
        }
        return run_result

    def _generate_run_id(self, test_cases: list[TestCase]) -> str:
        dims = sorted(set(tc.dimension for tc in test_cases))
        prefix = "_".join(dims)
        run_dir = self.output_dir
        # 取已有序号的最大值+1，避免归档/改名目录导致 len+1 撞上现存 run 而覆盖数据
        seq = 1
        if run_dir.exists():
            pattern = re.compile(rf"^{re.escape(prefix)}_run(\d+)")
            nums = [
                int(m.group(1))
                for d in run_dir.iterdir()
                if d.is_dir() and (m := pattern.match(d.name))
            ]
            if nums:
                seq = max(nums) + 1
        return f"{prefix}_run{seq}"

    def _build_summary(self, scores: list[ScoreResult], models: list[dict]) -> dict:
        summary: dict = {}
        for model in models:
            mid = model["id"]
            label = model.get("label", mid)
            model_scores = [s for s in scores if s.model_id == mid]

            dims: dict[str, list[float]] = {}
            for s in model_scores:
                dims.setdefault(s.dimension, []).append(s.score)

            dim_avgs = {d: round(sum(v) / len(v), 4) for d, v in dims.items()}
            all_scores = [s.score for s in model_scores]
            avg = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

            case_rounds: dict[str, list[float]] = {}
            for s in model_scores:
                case_rounds.setdefault(s.test_case_id, []).append(s.score)

            case_pass_rates = {
                case_id: {
                    "scores": round_scores,
                    "pass_rate": round(sum(1 for s in round_scores if s >= 1.0) / len(round_scores), 4),
                    "avg_score": round(sum(round_scores) / len(round_scores), 4),
                    "best": max(round_scores),
                }
                for case_id, round_scores in case_rounds.items()
            }

            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0
            total_cost = 0.0
            for s in model_scores:
                usage = s.details.get("token_usage", {})
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)
                cost_details = usage.get("cost_details", {})
                total_cost += cost_details.get("upstream_inference_cost", 0) or 0

            json_scores = []
            for s in model_scores:
                jc = s.details.get("json_compliance", {})
                if "score" in jc:
                    json_scores.append(jc["score"])
            json_compliance_avg = round(sum(json_scores) / len(json_scores), 4) if json_scores else 0.0

            summary[mid] = {
                "label": label,
                "dimensions": dim_avgs,
                "average": avg,
                "json_compliance": json_compliance_avg,
                "case_pass_rates": case_pass_rates,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                },
            }
        return summary
