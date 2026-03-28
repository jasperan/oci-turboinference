#!/usr/bin/env bash
set -e
LOG_PREFIX="[install-llmfit]"

if command -v llmfit &>/dev/null; then
    echo "$LOG_PREFIX llmfit already installed. Skipping."
    exit 0
fi

echo "$LOG_PREFIX Installing llmfit..."

# Install Rust toolchain if not present
if ! command -v cargo &>/dev/null; then
    echo "$LOG_PREFIX Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Install build dependencies
if command -v apt &>/dev/null; then
    apt-get update
    apt-get install -y build-essential pkg-config libssl-dev git
elif command -v dnf &>/dev/null; then
    dnf install -y gcc gcc-c++ pkg-config openssl-devel git
fi

# Clone and build
LLMFIT_DIR="/tmp/llmfit-build"
rm -rf "$LLMFIT_DIR"
echo "$LOG_PREFIX Cloning llmfit..."
git clone https://github.com/AlexsJones/llmfit "$LLMFIT_DIR"
cd "$LLMFIT_DIR"

echo "$LOG_PREFIX Building llmfit (release)..."
cargo build --release

# Install binary
cp target/release/llmfit /usr/local/bin/llmfit
chmod +x /usr/local/bin/llmfit

# Cleanup
rm -rf "$LLMFIT_DIR"

echo "$LOG_PREFIX llmfit installed successfully."
