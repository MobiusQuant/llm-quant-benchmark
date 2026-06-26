# LLM Quant Benchmark

A small, provider-agnostic harness for benchmarking large language models on **quantitative-trading tasks** — the things an LLM actually has to do inside a quant agent, not chat. Every task is scored automatically against a ground-truth answer.

It works with **any OpenAI-compatible endpoint**: OpenRouter, a vendor's first-party API, or a self-hosted gateway.

## What it measures

| Dimension | Task | Engineering scenario |
|---|---|---|
| **T1** calculation | SMA / RSI / MACD by hand | indicator computation |
| **T2** analysis | swing highs/lows, FVG, Order Block, BOS·CHoCH, EQH·EQL | K-line structure analysis |
| **T4+T7** rules | multi-condition signal judgment + constraint adherence | signal generation |
| **T5+T6** data | anomaly detection over 50 / 200 / 500 candles | data-quality checks |
| **JSON** | embedded in T2/T4/T5 | agent data exchange |

Data is real **BTC/USDT 1-hour** K-line from Binance; SMC ground truth is generated from a reference implementation. See [`docs/design.md`](docs/design.md) for the full methodology, and [`docs/results/`](docs/results/) for a worked example evaluating 13 models (987 API calls).

> **📖 The story behind the numbers:** [*We Benchmarked 13 LLMs on Quant Trading — Which Capabilities Are Reliable, and Which You Still Can't Trust*](docs/blog.md)
>
> Headline from the example run: frontier models now break 80% overall, but **K-line structure analysis tops out at ~52%** across every model — it isn't solved by scaling. Rule execution and anomaly detection, by contrast, are already production-grade (>92%). Full data and methodology in [`docs/results/technical_report.md`](docs/results/technical_report.md).

## Install

```bash
git clone https://github.com/<you>/llm-quant-benchmark.git
cd llm-quant-benchmark
pip install -e .            # or: pip install -r requirements.txt
```

Requires Python ≥ 3.10.

## Configure your API key

Keys are read from environment variables (or a `.env` file) — never from tracked config:

```bash
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=...
```

## Run

```bash
# Full benchmark, all default models, via OpenRouter
llm-quant-bench

# A subset of dimensions / a single model / more rounds
llm-quant-bench --dimensions t1 t4 --models openai/gpt-5.5 --rounds 3

# A different provider (see configs/)
llm-quant-bench --config configs/bigmodel-direct.yaml

# Resume an interrupted run (results are persisted per call)
llm-quant-bench --resume t1_t2_t4_t5_run1
```

Results stream to a Rich table and are saved under `results/<run_id>/` (one JSON per model × case × round, plus `summary.json`). `results/` is git-ignored.

## Providers

Three ready-made configs under [`configs/`](configs/) — copy one and edit the `models:` list:

| Config | Endpoint | Key env var |
|---|---|---|
| `openrouter.yaml` (default) | OpenRouter | `OPENROUTER_API_KEY` |
| `bigmodel-direct.yaml` | Zhipu BigModel (first-party) | `BIGMODEL_API_KEY` |
| `openai-compatible-proxy.yaml` | any OpenAI-compatible gateway / local server | `PROXY_API_KEY` |

A config has three blocks: `provider` (base_url, key env var, timeout, temperature, optional `max_tokens` / `extra_body` for thinking models), `models` (id + display label), and `run` (dimensions, concurrency, rounds).

## Project structure

```
llm-quant-benchmark/
├── llm_quant_bench/        # the framework (installable package)
│   ├── client.py           # LLMClient — async OpenAI-compatible client (retries, backoff)
│   ├── runner.py           # BenchmarkRunner — concurrency + per-call persistence + resume
│   ├── scorer.py           # 10 scoring methods + JSON-compliance checker
│   ├── loader.py           # load test cases from benchmarks/*.yaml
│   ├── models.py           # pydantic models (TestCase / ModelResponse / ScoreResult)
│   ├── report.py           # save_results + Rich report
│   ├── config.py           # config + env-based key loading
│   └── cli.py              # `llm-quant-bench` entry point
├── benchmarks/             # the test suite (YAML cases + generators)
│   └── t{1,2,4,5}_*/       # generate.py + reference.py + *.yaml cases
├── data/                   # raw BTC/USDT 1h K-line CSVs
├── configs/                # provider configs (sanitized examples)
├── docs/                   # design.md + results/ (example findings, 13 models)
└── tests/                  # unit tests for the framework
```

## Extending

**Add a model** — append to `models:` in your config (`{ id: "...", label: "..." }`).

**Add a test case** — drop a YAML file in `benchmarks/<dim>/`. Each case:

```yaml
cases:
  - id: t1_sma_my_case
    prompt: "Compute the 10-period SMA of the last 10 closes: ..."
    expected: 67783.66
    scoring_method: numeric_tolerance
    scoring_params: { tolerance: 0.5 }
    json_spec: null          # or a spec to score JSON compliance
```

The generators (`benchmarks/*/generate.py`) show how the bundled cases are derived from raw data — run them with the `generate` extra (`pip install -e ".[generate]"`).

**Add a scorer** — write a `def _score_x(tc, resp) -> ScoreResult` in `scorer.py` and register it in the `SCORERS` dict. Available methods: `numeric_tolerance`, `macd_tolerance`, `binary`, `highlow_match`, `smc_f1`, `f1_set`, `anomaly_f1`, `rule_signal`, `checklist`, `json_schema`. JSON compliance is scored on every case automatically.

## How scoring works

Each response goes through `score(test_case, response)`: the case's `scoring_method` produces a 0–1 score, and a JSON-compliance check (`direct_parseable`, `no_code_fence`, field types/enums against `json_spec`) is attached to every result. The runner averages per dimension and per case across rounds.

## License

MIT — see [LICENSE](LICENSE).
