#!/usr/bin/env bash
set -e
LOG_PREFIX="[install-vllm]"

if ! command -v nvidia-smi &>/dev/null; then
    echo "$LOG_PREFIX No GPU detected (nvidia-smi not found). Skipping vLLM installation."
    exit 0
fi

if python3 -c "import vllm" &>/dev/null; then
    VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    echo "$LOG_PREFIX vLLM already installed (version: $VLLM_VER). Skipping."
    exit 0
fi

echo "$LOG_PREFIX Installing vLLM..."
pip install vllm

echo "$LOG_PREFIX vLLM installed successfully."
python3 -c "import vllm; print('$LOG_PREFIX Version:', vllm.__version__)"
