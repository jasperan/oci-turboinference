#!/usr/bin/env bash
set -eo pipefail
LOG_PREFIX="[start-inference]"

CONFIG_FILE="${1:-/opt/turboinference/inference-config.json}"
API_PORT="${API_PORT:-8080}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "$LOG_PREFIX Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "$LOG_PREFIX Reading config from $CONFIG_FILE..."

# Extract all fields from JSON config in a single Python call
eval "$(python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))
print(f'BACKEND={c.get(\"backend\", \"llamacpp\")}')
print(f'MODEL_URL={c.get(\"model_url\", \"\")}')
print(f'QUANT_TYPE={c.get(\"quant_type\", \"Q4_K_M\")}')
print(f'N_GPU_LAYERS={c.get(\"n_gpu_layers\", 999)}')
print(f'CTX_SIZE={c.get(\"ctx_size\", 8192)}')
print(f'CPU_OFFLOAD={c.get(\"cpu_offload_gb\", 0)}')
" "$CONFIG_FILE")"

echo "$LOG_PREFIX Backend: $BACKEND"
echo "$LOG_PREFIX Model: $MODEL_URL"
echo "$LOG_PREFIX Quant: $QUANT_TYPE, GPU layers: $N_GPU_LAYERS, Context: $CTX_SIZE"

if [ "$BACKEND" = "vllm" ]; then
    echo "$LOG_PREFIX Starting vLLM server..."
    exec vllm serve "$MODEL_URL" \
        --host 0.0.0.0 \
        --port "$API_PORT" \
        --dtype auto \
        --gpu-memory-utilization 0.92 \
        --max-model-len "$CTX_SIZE" \
        --cpu-offload-gb "$CPU_OFFLOAD"

elif [ "$BACKEND" = "llamacpp" ]; then
    MODEL_FILE="$MODEL_URL"

    # If model_url contains ":" it's a repo:quant format, download GGUF
    if echo "$MODEL_URL" | grep -q ":"; then
        REPO=$(echo "$MODEL_URL" | cut -d: -f1)
        QUANT=$(echo "$MODEL_URL" | cut -d: -f2)
        MODEL_DIR="/opt/turboinference/models"
        mkdir -p "$MODEL_DIR"

        echo "$LOG_PREFIX Downloading GGUF: repo=$REPO quant=$QUANT..."
        pip install -q huggingface-hub 2>/dev/null || true

        # Use snapshot_download with allow_patterns (supports globs, unlike hf_hub_download)
        MODEL_FILE=$(python3 -c "
import os, sys, glob
from huggingface_hub import snapshot_download

repo = sys.argv[1]
quant = sys.argv[2]
model_dir = sys.argv[3]

local = snapshot_download(
    repo_id=repo,
    allow_patterns=[f'*{quant}*.gguf'],
    local_dir=model_dir,
    local_dir_use_symlinks=False,
)
files = glob.glob(os.path.join(local, '**/*.gguf'), recursive=True)
if not files:
    files = glob.glob(os.path.join(model_dir, '**/*.gguf'), recursive=True)
if files:
    print(sorted(files, key=os.path.getsize, reverse=True)[0])
else:
    sys.exit(1)
" "$REPO" "$QUANT" "$MODEL_DIR" 2>&1) || true

        if [ -z "$MODEL_FILE" ] || [ ! -f "$MODEL_FILE" ]; then
            echo "$LOG_PREFIX Failed to download model. Exiting."
            exit 1
        fi
        echo "$LOG_PREFIX Model file: $MODEL_FILE"
    fi

    # Build llama-server args
    ARGS=(
        --host 0.0.0.0
        --port "$API_PORT"
        --model "$MODEL_FILE"
        --ctx-size "$CTX_SIZE"
        --n-gpu-layers "$N_GPU_LAYERS"
        --flash-attn on
    )

    # Add KV cache quantization if GPU present
    if command -v nvidia-smi &>/dev/null; then
        ARGS+=(--cache-type-k q8_0 --cache-type-v q8_0)
    fi

    echo "$LOG_PREFIX Starting llama-server..."
    exec llama-server "${ARGS[@]}"
else
    echo "$LOG_PREFIX Unknown backend: $BACKEND"
    exit 1
fi
