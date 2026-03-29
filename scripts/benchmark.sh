#!/usr/bin/env bash
set -eo pipefail
LOG_PREFIX="[benchmark]"

# Defaults
PORT="${PORT:-8080}"
PROMPTS="short,medium,long,context"
OUTPUT_DIR="benchmarks"

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --prompts)
            PROMPTS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "$LOG_PREFIX Unknown arg: $1"
            echo "Usage: $0 [--port 8080] [--prompts short,medium,long,context] [--output-dir benchmarks/]"
            exit 1
            ;;
    esac
done

BASE_URL="http://localhost:${PORT}"

echo "$LOG_PREFIX Checking inference server at ${BASE_URL}..."

# Health check: try /v1/models (standard OpenAI-compatible endpoint)
if ! curl -sf --max-time 5 "${BASE_URL}/v1/models" > /dev/null 2>&1; then
    echo "$LOG_PREFIX Server not reachable at ${BASE_URL}/v1/models"
    echo "$LOG_PREFIX Make sure the inference server is running on port ${PORT}."
    exit 1
fi
echo "$LOG_PREFIX Server is up."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Try to read model info from inference-config.json (if available)
CONFIG_FILE="${CONFIG_FILE:-/opt/turboinference/inference-config.json}"
if [ -f "$CONFIG_FILE" ]; then
    echo "$LOG_PREFIX Reading config from $CONFIG_FILE"
    MODEL_ID=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('model_url','unknown'))" 2>/dev/null || echo "unknown")
    echo "$LOG_PREFIX Model from config: $MODEL_ID"
else
    echo "$LOG_PREFIX No inference-config.json found, model info will come from /v1/models"
fi

# Run the benchmark
echo "$LOG_PREFIX Running benchmark suite (prompts: $PROMPTS)..."
echo ""

python3 -m profiler.benchmark \
    --port "$PORT" \
    --prompts "$PROMPTS" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "$LOG_PREFIX Benchmark complete. Results saved to ${OUTPUT_DIR}/"
    echo "$LOG_PREFIX Files:"
    ls -1t "$OUTPUT_DIR"/ | head -5 | while read -r f; do
        echo "$LOG_PREFIX   ${OUTPUT_DIR}/${f}"
    done
else
    echo "$LOG_PREFIX Benchmark failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
