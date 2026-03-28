#!/usr/bin/env bash
set -e
LOG_PREFIX="[install-pi-agent]"

API_PORT="${API_PORT:-8080}"

if command -v pi &>/dev/null; then
    echo "$LOG_PREFIX Pi coding agent already installed. Skipping install."
else
    echo "$LOG_PREFIX Installing Pi coding agent..."

    # Install Node.js 22 LTS if not present
    if ! command -v node &>/dev/null || [ "$(node -v | cut -d. -f1 | tr -d v)" -lt 22 ]; then
        echo "$LOG_PREFIX Installing Node.js 22 LTS..."
        if command -v apt &>/dev/null; then
            curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
            apt-get install -y nodejs
        elif command -v dnf &>/dev/null; then
            curl -fsSL https://rpm.nodesource.com/setup_22.x | bash -
            dnf install -y nodejs
        fi
    fi

    echo "$LOG_PREFIX Installing @mariozechner/pi-coding-agent globally..."
    npm install -g @mariozechner/pi-coding-agent
fi

# Configure Pi to use local inference
echo "$LOG_PREFIX Configuring Pi for local inference (port $API_PORT)..."
PI_CONFIG_DIR="$HOME/.pi/agent"
mkdir -p "$PI_CONFIG_DIR"

cat > "$PI_CONFIG_DIR/models.json" << PIEOF
{
    "providers": {
        "local-turboinfer": {
            "baseUrl": "http://localhost:${API_PORT}/v1",
            "api": "openai-completions",
            "compatFlags": {
                "supportsDeveloperRole": false,
                "supportsReasoningEffort": false
            }
        }
    },
    "models": {
        "turbo-model": {
            "provider": "local-turboinfer"
        }
    }
}
PIEOF

echo "$LOG_PREFIX Pi agent configured. Models config: $PI_CONFIG_DIR/models.json"
