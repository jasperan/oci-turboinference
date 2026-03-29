#!/usr/bin/env bash
set -eo pipefail
LOG_PREFIX="[setup]"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
API_PORT="${API_PORT:-8080}"
INSTALL_PI="${INSTALL_PI:-true}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPT_DIR="/opt/turboinference"

echo "$LOG_PREFIX Starting TurboInference setup..."
echo "$LOG_PREFIX Model: $MODEL_ID"
echo "$LOG_PREFIX API Port: $API_PORT"
echo "$LOG_PREFIX Install Pi: $INSTALL_PI"

# Create working directory
mkdir -p "$OPT_DIR"

# Copy profiler modules to /opt/turboinference/
echo "$LOG_PREFIX Copying profiler to $OPT_DIR..."
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -d "$REPO_DIR/profiler" ]; then
    cp -r "$REPO_DIR/profiler" "$OPT_DIR/"
else
    echo "$LOG_PREFIX WARNING: profiler/ not found in repo. Skipping profiler copy."
fi

# Step 1: Install NVIDIA drivers
echo "$LOG_PREFIX Step 1/7: Installing NVIDIA drivers..."
bash "$SCRIPT_DIR/install-drivers.sh"

# Step 2: Install llama.cpp
echo "$LOG_PREFIX Step 2/7: Installing llama.cpp..."
bash "$SCRIPT_DIR/install-llama-cpp.sh"

# Step 3: Install vLLM
echo "$LOG_PREFIX Step 3/7: Installing vLLM..."
bash "$SCRIPT_DIR/install-vllm.sh"

# Step 4: Install llmfit
echo "$LOG_PREFIX Step 4/7: Installing llmfit..."
bash "$SCRIPT_DIR/install-llmfit.sh"

# Step 5: Run profiler to generate config
echo "$LOG_PREFIX Step 5/7: Running hardware profiler..."
pip install pyyaml httpx 2>/dev/null || true

CONFIG_FILE="$OPT_DIR/inference-config.json"
python3 -c "
import sys, json
from dataclasses import asdict

sys.path.insert(0, sys.argv[1])
from profiler.detect import detect_hardware
from profiler.strategy import pick_strategy

hw = detect_hardware()
config = pick_strategy(sys.argv[2], hw)
output = asdict(config)
output['model_id'] = sys.argv[2]

with open(sys.argv[3], 'w') as f:
    json.dump(output, f, indent=2)
print(json.dumps(output, indent=2))
" "$OPT_DIR" "$MODEL_ID" "$CONFIG_FILE"
echo "$LOG_PREFIX Config written to $CONFIG_FILE"

# Step 6: Install Pi agent
if [ "$INSTALL_PI" = "true" ]; then
    echo "$LOG_PREFIX Step 6/7: Installing Pi agent..."
    API_PORT="$API_PORT" bash "$SCRIPT_DIR/install-pi-agent.sh"
else
    echo "$LOG_PREFIX Step 6/7: Skipping Pi agent (INSTALL_PI=$INSTALL_PI)"
fi

# Step 7: Start inference server in background
echo "$LOG_PREFIX Step 7/7: Starting inference server..."
mkdir -p /var/log
nohup bash "$SCRIPT_DIR/start-inference.sh" "$CONFIG_FILE" \
    > /var/log/turboinference.log 2>&1 &
INFERENCE_PID=$!
echo "$LOG_PREFIX Inference server started (PID: $INFERENCE_PID)"
echo "$LOG_PREFIX Logs: /var/log/turboinference.log"

echo "$LOG_PREFIX Setup complete."
