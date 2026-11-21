# vLLM Serving Benchmark Report

- **Endpoint:** `http://127.0.0.1:8002/v1` (executed live on Google Colab Pro L4 GPU via `notebooks/sanad_vllm_l4_benchmark.ipynb`)
- **Base + Adapter:** `Qwen/Qwen2.5-7B-Instruct` (`--quantization bitsandbytes` via `vllm-bnb-plugin`) + `sanad-qwen25-ft` (`3isa2002/qwen2.5-7b-sanad-arabic-qlora`)
- **GPU / VRAM (`nvidia-smi`):** NVIDIA L4 (`19510 / 23034 MiB`, `VLLM::EngineCore` `19502 MiB`)

| Concurrency | Ok / Total | Req/s | Output Tokens/s | TTFT p50 (ms) | TTFT p95 (ms) | Latency p50 (ms) | Latency p95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6/6 | 0.996 | 37.18 | 86.93 | 89.62 | 980.32 | 1572.89 |
| 4 | 12/12 | 2.374 | 89.61 | 138.89 | 158.64 | 1470.83 | 2973.63 |
| 8 | 24/24 | 3.996 | 142.2 | 164.87 | 249.64 | 1487.52 | 2919.85 |
