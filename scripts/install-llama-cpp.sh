#!/usr/bin/env bash
set -eo pipefail
LOG_PREFIX="[install-llama-cpp]"

INSTALL_DIR="/opt/llama.cpp"

if command -v llama-server &>/dev/null; then
    echo "$LOG_PREFIX llama.cpp already installed (llama-server found in PATH). Skipping."
    exit 0
fi

echo "$LOG_PREFIX Installing llama.cpp..."

# Install build dependencies
if command -v apt &>/dev/null; then
    apt-get update
    apt-get install -y build-essential cmake git
elif command -v dnf &>/dev/null; then
    dnf install -y cmake gcc gcc-c++ git make
fi

# Clone
if [ -d "$INSTALL_DIR" ]; then
    echo "$LOG_PREFIX $INSTALL_DIR already exists, pulling latest..."
    cd "$INSTALL_DIR" && git pull
    # Clean stale build artifacts to avoid CMake cache issues
    echo "$LOG_PREFIX Cleaning stale build directory..."
    rm -rf build/
else
    echo "$LOG_PREFIX Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
mkdir -p build && cd build

# Build with or without CUDA
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
if command -v nvidia-smi &>/dev/null; then
    echo "$LOG_PREFIX GPU detected, building with CUDA support..."
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"
else
    echo "$LOG_PREFIX No GPU detected, building CPU-only..."
fi

cmake .. $CMAKE_ARGS
cmake --build . --config Release -j "$(nproc)"

# Symlink binaries
echo "$LOG_PREFIX Creating symlinks..."
for BIN in llama-server llama-cli; do
    BUILT_BIN="$INSTALL_DIR/build/bin/$BIN"
    if [ -f "$BUILT_BIN" ]; then
        ln -sf "$BUILT_BIN" "/usr/local/bin/$BIN"
        echo "$LOG_PREFIX Linked $BIN -> /usr/local/bin/$BIN"
    fi
done

echo "$LOG_PREFIX llama.cpp installed successfully."
