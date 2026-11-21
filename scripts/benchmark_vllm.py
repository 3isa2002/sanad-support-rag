"""Benchmark OpenAI-compatible vLLM / Ollama serving (TTFT, p50/p95 latency, tokens/sec, VRAM)."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

BENCHMARK_PROMPTS = [
    ("ar", "مصاريف الشحن كام جوه القاهرة والجيزة؟ جاوب باختصار مع ذكر المصدر [1]."),
    ("ar", "إيه شروط إرجاع الأجهزة الإلكترونية والموبايلات؟"),
    ("en", "How many business days does a credit card refund take?"),
    ("en", "Why would a marketplace listing get rejected during review?"),
    ("ar", "الدفع فشل بس الفلوس اتخصمت من الكارت، أعمل إيه؟"),
    ("en", "Someone asked me for my OTP code on WhatsApp, is that official support?"),
]

SYSTEM_MSG = (
    "You are Sanad, a bilingual Egyptian Arabic & English customer-support assistant. "
    "Answer concisely in 1-3 sentences."
)


@dataclass
class ConcurrencyResult:
    concurrency: int
    total_requests: int
    succeeded: int
    failed: int
    wall_seconds: float
    requests_per_sec: float
    output_tokens_total: int
    output_tokens_per_sec: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_avg_ms: float


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return round(s[lo], 2)
    return round(s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo), 2)


def read_gpu_vram() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) != 3:
        return None
    return {"gpu_name": parts[0], "vram_used_mb": int(parts[1]), "vram_total_mb": int(parts[2])}


async def _run_one(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> tuple[bool, float, float, int]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with sem:
        t0 = time.perf_counter()
        ttft_ms = 0.0
        tokens = 0
        try:
            async with client.stream("POST", "chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    if chunk.get("usage") and chunk["usage"].get("completion_tokens"):
                        tokens = int(chunk["usage"]["completion_tokens"])
                    choices = chunk.get("choices") or []
                    if choices and choices[0].get("delta", {}).get("content"):
                        if ttft_ms == 0.0:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        if not chunk.get("usage"):
                            tokens += 1
            total_ms = (time.perf_counter() - t0) * 1000.0
            return True, (ttft_ms or total_ms), total_ms, max(1, tokens)
        except Exception:
            return False, 0.0, 0.0, 0


async def benchmark_level(
    base_url: str,
    model: str,
    *,
    api_key: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    timeout: float,
) -> ConcurrencyResult:
    normalized = base_url.rstrip("/") + "/"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    sem = asyncio.Semaphore(concurrency)
    prompts = [BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)][1] for i in range(num_requests)]

    async with httpx.AsyncClient(base_url=normalized, headers=headers, timeout=timeout) as client:
        t_start = time.perf_counter()
        results = await asyncio.gather(
            *[_run_one(client, model, p, max_tokens, sem) for p in prompts]
        )
        wall = max(1e-6, time.perf_counter() - t_start)

    ok_rows = [r for r in results if r[0]]
    ttfts = [r[1] for r in ok_rows]
    lats = [r[2] for r in ok_rows]
    tok_total = sum(r[3] for r in ok_rows)

    return ConcurrencyResult(
        concurrency=concurrency,
        total_requests=num_requests,
        succeeded=len(ok_rows),
        failed=len(results) - len(ok_rows),
        wall_seconds=round(wall, 3),
        requests_per_sec=round(len(ok_rows) / wall, 3),
        output_tokens_total=tok_total,
        output_tokens_per_sec=round(tok_total / wall, 2),
        ttft_p50_ms=_pct(ttfts, 0.50),
        ttft_p95_ms=_pct(ttfts, 0.95),
        latency_p50_ms=_pct(lats, 0.50),
        latency_p95_ms=_pct(lats, 0.95),
        latency_avg_ms=round(sum(lats) / len(lats), 2) if lats else 0.0,
    )


def format_markdown(payload: dict[str, Any]) -> str:
    gpu = payload.get("gpu") or {}
    gpu_str = (
        f"{gpu['gpu_name']} (`{gpu['vram_used_mb']} / {gpu['vram_total_mb']} MiB`)"
        if gpu
        else "N/A (remote or CPU)"
    )
    lines = [
        "# vLLM Serving Benchmark Report",
        "",
        f"- **Endpoint:** `{payload['base_url']}`",
        f"- **Model:** `{payload['model']}`",
        f"- **GPU / VRAM:** {gpu_str}",
        "",
        "| Concurrency | Ok / Total | Req/s | Output Tokens/s | TTFT p50 (ms) | TTFT p95 (ms) | Latency p50 (ms) | Latency p95 (ms) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["results"]:
        lines.append(
            f"| {r['concurrency']} | {r['succeeded']}/{r['total_requests']} | "
            f"{r['requests_per_sec']} | {r['output_tokens_per_sec']} | "
            f"{r['ttft_p50_ms']} | {r['ttft_p95_ms']} | {r['latency_p50_ms']} | {r['latency_p95_ms']} |"
        )
    lines.append("")
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    # Warm-up request
    await benchmark_level(
        args.base_url,
        args.model,
        api_key=args.api_key,
        concurrency=1,
        num_requests=1,
        max_tokens=32,
        timeout=args.timeout,
    )
    levels = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]
    results = []
    for c in levels:
        res = await benchmark_level(
            args.base_url,
            args.model,
            api_key=args.api_key,
            concurrency=c,
            num_requests=max(c * args.rounds, args.min_requests),
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        results.append(asdict(res))

    return {
        "base_url": args.base_url,
        "model": args.model,
        "gpu": read_gpu_vram(),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark vLLM / OpenAI-compatible serving.")
    parser.add_argument("--base-url", default="http://localhost:8002/v1")
    parser.add_argument("--model", default="sanad-qwen25-ft")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-requests", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("data/eval/vllm_benchmark.json"))
    args = parser.parse_args()

    payload = asyncio.run(_main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md = format_markdown(payload)
    md_path.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
