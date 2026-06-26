from __future__ import annotations

import asyncio
import time

import httpx

from llm_quant_bench.models import ModelResponse


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
        max_retries: int = 2,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        # None 表示不发送 temperature 参数（如 claude-fable-5 拒绝该参数）
        self._temperature = temperature
        # None 表示不发送 max_tokens（用服务端默认值）；
        # 思考型模型在重型任务上可能把默认额度全烧在 thinking 上导致无 text 输出
        self._max_tokens = max_tokens
        # 透传额外的请求体字段（如 GLM-5.2 的 thinking / reasoning_effort），
        # 用于显式钉死服务端默认值以保证可复现
        self._extra_body = extra_body or {}
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "LLMQuantBenchmark",
                },
                timeout=self._timeout,
            )
        return self._http

    async def chat(
        self,
        model: str,
        messages: list[dict],
        test_case_id: str = "",
    ) -> ModelResponse:
        body = {
            "model": model,
            "messages": messages,
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        if self._extra_body:
            body.update(self._extra_body)

        last_error: str | None = None
        for attempt in range(1 + self._max_retries):
            try:
                client = await self._get_http()
                t0 = time.perf_counter()
                resp = await client.post("/chat/completions", json=body)
                latency = (time.perf_counter() - t0) * 1000

                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    # 429 多为并发/速率限制，退避加长
                    await asyncio.sleep(2 ** attempt * 2)
                    continue

                resp.raise_for_status()
                data = resp.json()

                message = (data.get("choices") or [{}])[0].get("message") or {}
                content = message.get("content")
                if not content:
                    # 部分代理/模型偶发返回无 content 的 message（如深度思考耗尽输出），
                    # 重试而不是记录一个空响应的 0 分
                    last_error = "Response missing message content"
                    await asyncio.sleep(2 ** attempt)
                    continue
                usage = data.get("usage", {})

                return ModelResponse(
                    model_id=model,
                    test_case_id=test_case_id,
                    raw_response=content,
                    latency_ms=latency,
                    token_usage=usage,
                )

            except Exception as e:
                last_error = str(e)
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)

        return ModelResponse(
            model_id=model,
            test_case_id=test_case_id,
            error=last_error,
        )

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()
