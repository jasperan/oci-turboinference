#!/usr/bin/env bash
set -e
LOG_PREFIX="[start-inference]"

CONFIG_FILE="${1:-/opt/turboinference/inference-config.json}"
API_PORT="${API_PORT:-8080}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "$LOG_PREFIX Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "$LOG_PREFIX Reading config from $CONFIG_FILE..."

# Extract fields from JSON config
BACKEND=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('backend','llamacpp'))")
MODEL_URL=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('model_url',''))")
QUANT_TYPE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('quant_type','Q4_K_M'))")
N_GPU_LAYERS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('n_gpu_layers',999))")
CTX_SIZE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('ctx_size',8192))")
CPU_OFFLOAD=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('cpu_offload_gb',0))")

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
        MODEL_FILE=$(python3 -c "
from huggingface_hub import hf_hub_download
import glob, os
path = hf_hub_download(
    repo_id='$REPO',
    filename='*${QUANT}*.gguf',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False
)
print(path)
" 2>/dev/null || true)

        # Fallback: search for downloaded file
        if [ -z "$MODEL_FILE" ] || [ ! -f "$MODEL_FILE" ]; then
            echo "$LOG_PREFIX Trying glob-based download..."
            MODEL_FILE=$(python3 << 'PYEOF'
import os, glob
from huggingface_hub import snapshot_download
local = snapshot_download(
    repo_id="REPO_PLACEHOLDER",
    allow_patterns=["*QUANT_PLACEHOLDER*.gguf"],
    local_dir="/opt/turboinference/models",
    local_dir_use_symlinks=False
)
files = glob.glob(os.path.join(local, "**/*.gguf"), recursive=True)
if files:
    print(sorted(files, key=os.path.getsize, reverse=True)[0])
PYEOF
            )
            MODEL_FILE=$(echo "$MODEL_FILE" | sed "s|REPO_PLACEHOLDER|$REPO|;s|QUANT_PLACEHOLDER|$QUANT|")
        fi

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
        --flash-attn
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
