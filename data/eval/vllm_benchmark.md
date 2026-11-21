# vLLM Serving Benchmark Report

- **Endpoint:** `http://127.0.0.1:8002/v1`
- **Model:** `sanad-qwen25-ft` (`Qwen/Qwen2.5-7B-Instruct` + LoRA `3isa2002/qwen2.5-7b-sanad-arabic-qlora`, `bitsandbytes`)
- **GPU / VRAM:** NVIDIA L4 (`20934 / 23034 MiB`)

| Concurrency | Ok / Total | Req/s | Output Tokens/s | TTFT p50 (ms) | TTFT p95 (ms) | Latency p50 (ms) | Latency p95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6/6 | 0.342 | 26.03 | 115.61 | 135.75 | 3001.43 | 3307.98 |
| 4 | 12/12 | 1.223 | 96.12 | 157.82 | 182.97 | 3249.68 | 3592.71 |
| 8 | 24/24 | 2.229 | 171.25 | 197.43 | 333.24 | 3566.1 | 4001.81 |
