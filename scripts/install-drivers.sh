#!/usr/bin/env bash
set -e
LOG_PREFIX="[install-drivers]"

echo "$LOG_PREFIX Checking for NVIDIA GPU..."

if ! lspci | grep -qi nvidia; then
    echo "$LOG_PREFIX No NVIDIA GPU detected. Skipping driver installation."
    exit 0
fi

if command -v nvidia-smi &>/dev/null; then
    echo "$LOG_PREFIX NVIDIA drivers already installed:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
    echo "$LOG_PREFIX Skipping."
    exit 0
fi

echo "$LOG_PREFIX Installing NVIDIA drivers + CUDA 12.4..."

if command -v dnf &>/dev/null; then
    # Oracle Linux / RHEL / Fedora
    echo "$LOG_PREFIX Detected dnf-based system (Oracle Linux / RHEL)"
    dnf install -y epel-release
    dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo || \
    dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
    dnf clean all
    dnf -y module install nvidia-driver:latest-dkms || dnf -y install nvidia-driver
    dnf -y install cuda-toolkit-12-4
elif command -v apt &>/dev/null; then
    # Ubuntu / Debian
    echo "$LOG_PREFIX Detected apt-based system (Ubuntu / Debian)"
    apt-get update
    apt-get install -y software-properties-common
    # Add NVIDIA CUDA repo
    DISTRO="ubuntu$(lsb_release -rs | tr -d '.')"
    ARCH="x86_64"
    apt-get install -y wget
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb" \
        -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update
    apt-get install -y cuda-toolkit-12-4 nvidia-driver-550
else
    echo "$LOG_PREFIX Unsupported package manager. Install NVIDIA drivers manually."
    exit 1
fi

# Add CUDA to PATH for all users
echo "$LOG_PREFIX Configuring CUDA PATH..."
cat > /etc/profile.d/cuda.sh << 'CUDA_EOF'
export PATH="/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
CUDA_EOF
chmod 644 /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

echo "$LOG_PREFIX NVIDIA drivers + CUDA 12.4 installed successfully."
