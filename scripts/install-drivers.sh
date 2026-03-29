#!/usr/bin/env bash
set -eo pipefail
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

# Detect OS version from /etc/os-release
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID="${ID}"
    OS_VERSION_ID="${VERSION_ID}"
    OS_VERSION_MAJOR="${VERSION_ID%%.*}"
else
    echo "$LOG_PREFIX Cannot detect OS: /etc/os-release not found."
    exit 1
fi

echo "$LOG_PREFIX Detected OS: ${OS_ID} ${OS_VERSION_ID} (major: ${OS_VERSION_MAJOR})"
echo "$LOG_PREFIX Installing NVIDIA drivers + CUDA 12.4..."

if command -v dnf &>/dev/null; then
    # Oracle Linux / RHEL / Fedora
    echo "$LOG_PREFIX Detected dnf-based system (${OS_ID})"
    dnf install -y epel-release

    # Select CUDA repo URL based on OS major version
    case "$OS_VERSION_MAJOR" in
        8)
            CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo"
            ;;
        9)
            CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo"
            ;;
        *)
            echo "$LOG_PREFIX Unsupported RHEL/OL major version: ${OS_VERSION_MAJOR}. Trying rhel9 repo."
            CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo"
            ;;
    esac

    echo "$LOG_PREFIX Adding CUDA repo: ${CUDA_REPO}"
    dnf config-manager --add-repo "$CUDA_REPO"
    dnf clean all
    dnf -y module install nvidia-driver:latest-dkms || dnf -y install nvidia-driver
    dnf -y install cuda-toolkit-12-4
elif command -v apt &>/dev/null; then
    # Ubuntu / Debian
    echo "$LOG_PREFIX Detected apt-based system (${OS_ID})"
    apt-get update
    apt-get install -y software-properties-common

    # Get Ubuntu version from /etc/os-release (no lsb_release dependency)
    UBUNTU_VERSION="$(echo "${OS_VERSION_ID}" | tr -d '.')"
    DISTRO="ubuntu${UBUNTU_VERSION}"
    ARCH="x86_64"

    echo "$LOG_PREFIX Using CUDA repo for ${DISTRO}/${ARCH}"
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
