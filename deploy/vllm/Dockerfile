# GPU-enabled vLLM OpenAI-compatible container serving Qwen2.5-7B-Instruct + Sanad QLoRA adapter
FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir bitsandbytes vllm-bnb-plugin

ENV VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
    VLLM_LORA_MODULES=sanad-qwen25-ft=3isa2002/qwen2.5-7b-sanad-arabic-qlora \
    VLLM_QUANTIZATION=bitsandbytes \
    VLLM_MAX_MODEL_LEN=2048 \
    VLLM_GPU_MEMORY_UTILIZATION=0.90 \
    VLLM_PORT=8000

EXPOSE 8000

ENTRYPOINT ["/bin/sh", "-c", "exec vllm serve \"$VLLM_MODEL\" --host 0.0.0.0 --port \"$VLLM_PORT\" --dtype half --quantization \"$VLLM_QUANTIZATION\" --enable-lora --lora-modules \"$VLLM_LORA_MODULES\" --max-model-len \"$VLLM_MAX_MODEL_LEN\" --gpu-memory-utilization \"$VLLM_GPU_MEMORY_UTILIZATION\""]
