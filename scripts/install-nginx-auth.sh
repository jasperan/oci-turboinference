#!/usr/bin/env bash
set -eo pipefail
LOG_PREFIX="[nginx-auth]"

API_PORT="${API_PORT:-8080}"
PROXY_PORT="${PROXY_PORT:-8443}"
API_KEY="${API_KEY:-}"
KEY_FILE="/opt/turboinference/api-key"

echo "$LOG_PREFIX Installing nginx reverse proxy with API key authentication..."

# Install nginx
if command -v apt-get &>/dev/null; then
    echo "$LOG_PREFIX Installing nginx via apt..."
    apt-get update -qq
    apt-get install -y -qq nginx
elif command -v dnf &>/dev/null; then
    echo "$LOG_PREFIX Installing nginx via dnf..."
    dnf install -y nginx
elif command -v yum &>/dev/null; then
    echo "$LOG_PREFIX Installing nginx via yum..."
    yum install -y nginx
else
    echo "$LOG_PREFIX ERROR: No supported package manager found (apt, dnf, yum)."
    exit 1
fi

# Generate or use provided API key
if [ -z "$API_KEY" ]; then
    API_KEY="$(openssl rand -hex 32)"
    echo "$LOG_PREFIX Generated random API key."
else
    echo "$LOG_PREFIX Using provided API key."
fi

# Save key to file
mkdir -p "$(dirname "$KEY_FILE")"
echo "$API_KEY" > "$KEY_FILE"
chmod 600 "$KEY_FILE"
echo "$LOG_PREFIX API key saved to $KEY_FILE"

# Write nginx config
cat > /etc/nginx/conf.d/turboinference.conf <<NGINX_EOF
map \$http_authorization \$auth_valid {
    default                    0;
    "Bearer ${API_KEY}"        1;
}

server {
    listen ${PROXY_PORT};

    location / {
        # Reject missing or invalid API keys
        if (\$auth_valid = 0) {
            return 401 '{"error": "Unauthorized. Provide a valid Authorization: Bearer <key> header."}';
        }

        # CORS headers
        add_header Access-Control-Allow-Origin "*" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Authorization, Content-Type" always;

        # Handle CORS preflight
        if (\$request_method = OPTIONS) {
            add_header Access-Control-Allow-Origin "*";
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
            add_header Access-Control-Allow-Headers "Authorization, Content-Type";
            add_header Content-Length 0;
            add_header Content-Type text/plain;
            return 204;
        }

        # Proxy to inference server
        proxy_pass http://localhost:${API_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Streaming support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
NGINX_EOF

echo "$LOG_PREFIX Nginx config written to /etc/nginx/conf.d/turboinference.conf"

# Test nginx config
nginx -t
echo "$LOG_PREFIX Nginx config test passed."

# Enable and start nginx
systemctl enable nginx
systemctl restart nginx
echo "$LOG_PREFIX Nginx started and enabled."

echo "$LOG_PREFIX ============================================"
echo "$LOG_PREFIX API Key: $API_KEY"
echo "$LOG_PREFIX Proxy URL: http://localhost:${PROXY_PORT}"
echo "$LOG_PREFIX Backend: http://localhost:${API_PORT}"
echo "$LOG_PREFIX Key file: $KEY_FILE"
echo "$LOG_PREFIX ============================================"
echo "$LOG_PREFIX Usage: curl -H 'Authorization: Bearer $API_KEY' http://<host>:${PROXY_PORT}/v1/models"
