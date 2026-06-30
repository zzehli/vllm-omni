#!/bin/bash
# AudioX online serving startup script.
# AudioX requires an explicit pipeline class (`--model-class-name AudioXPipeline`).
# Per-request task and sampler knobs are sent via `extra_body` on the standard
# chat-completions endpoint (see run_curl_audiox.sh).

MODEL="${MODEL:-zhangj1an/AudioX}"
PORT="${PORT:-8099}"
DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-FLASH_ATTN}"

echo "Starting AudioX server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Diffusion attention backend: $DIFFUSION_ATTENTION_BACKEND"

DIFFUSION_ATTENTION_BACKEND="$DIFFUSION_ATTENTION_BACKEND" \
    vllm serve "$MODEL" --omni \
        --model-class-name AudioXPipeline \
        --port "$PORT"
