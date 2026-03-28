# oci-turboinference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** One-click OCI Resource Manager Stack that deploys a GPU or CPU-only instance, auto-profiles hardware, picks optimal quantization + offload strategy, downloads the model, starts inference (vLLM or llama.cpp), and configures Pi coding agent.

**Architecture:** Terraform provisions the OCI instance. Cloud-init runs a master `setup.sh` that installs all dependencies, runs a Python profiler to pick the best backend/quant/offload strategy, downloads the model, starts the inference server, and configures Pi. The profiler uses a curated lookup table for popular models and falls back to llmfit for unknowns.

**Tech Stack:** Terraform (OCI provider), Python 3.11+ (profiler), Bash (install scripts), llama.cpp (GGUF inference), vLLM (AWQ/GPTQ inference), llmfit (Rust, hardware profiling), Pi coding agent (Node.js), cloud-init (YAML).

---

### Task 1: Profiler - Hardware Detection

**Files:**
- Create: `profiler/__init__.py`
- Create: `profiler/detect.py`
- Create: `tests/test_detect.py`

**Step 1: Write the failing test**

```python
# tests/test_detect.py
import pytest
from unittest.mock import patch
from profiler.detect import detect_hardware, HardwareInfo


def test_detect_hardware_with_gpu():
    """Detect GPU VRAM, system RAM, and disk."""
    mock_smi = "24576 MiB\n"
    mock_free = (
        "              total        used        free\n"
        "Mem:         245760       12345      200000\n"
    )
    mock_disk = (
        "Filesystem     1G-blocks  Used Available\n"
        "/dev/sda1            200    50       150\n"
    )
    with patch("profiler.detect._run_cmd") as mock_run:
        mock_run.side_effect = [mock_smi, mock_free, mock_disk]
        info = detect_hardware()

    assert info.gpu_model is not None or info.vram_gb == pytest.approx(24, abs=1)
    assert info.vram_gb == pytest.approx(24, abs=1)
    assert info.ram_gb > 200
    assert info.disk_gb > 100
    assert info.has_gpu is True


def test_detect_hardware_cpu_only():
    """CPU-only: vram_gb=0, has_gpu=False."""
    mock_free = (
        "              total        used        free\n"
        "Mem:          65536       12345       50000\n"
    )
    mock_disk = (
        "Filesystem     1G-blocks  Used Available\n"
        "/dev/sda1            500   100       400\n"
    )
    with patch("profiler.detect._run_cmd") as mock_run:
        # nvidia-smi fails (no GPU)
        mock_run.side_effect = [None, mock_free, mock_disk]
        info = detect_hardware()

    assert info.vram_gb == 0
    assert info.has_gpu is False
    assert info.ram_gb > 60
```

**Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/git/personal/oci-turboinference && python -m pytest tests/test_detect.py -v`
Expected: FAIL with "No module named 'profiler'"

**Step 3: Write minimal implementation**

```python
# profiler/__init__.py
# oci-turboinference profiler package

# profiler/detect.py
from dataclasses import dataclass
import subprocess
import re


@dataclass
class HardwareInfo:
    has_gpu: bool
    gpu_model: str | None
    vram_gb: float
    ram_gb: float
    disk_gb: float


def _run_cmd(cmd: list[str]) -> str | None:
    """Run a shell command, return stdout or None on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _parse_vram() -> tuple[str | None, float]:
    """Query nvidia-smi for GPU model and VRAM in GB."""
    # Get VRAM
    output = _run_cmd([
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits"
    ])
    if output is None:
        return None, 0.0

    vram_mib = float(output.strip().split("\n")[0])
    vram_gb = vram_mib / 1024.0

    # Get GPU model name
    name_output = _run_cmd([
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader"
    ])
    gpu_model = name_output.strip() if name_output else None

    return gpu_model, vram_gb


def _parse_ram() -> float:
    """Get total system RAM in GB from free command."""
    output = _run_cmd(["free", "-m"])
    if output is None:
        return 0.0

    for line in output.strip().split("\n"):
        if line.startswith("Mem:"):
            parts = line.split()
            return float(parts[1]) / 1024.0
    return 0.0


def _parse_disk() -> float:
    """Get available disk in GB."""
    output = _run_cmd(["df", "-BG", "--output=avail", "/"])
    if output is None:
        return 0.0

    lines = output.strip().split("\n")
    if len(lines) >= 2:
        avail = lines[1].strip().rstrip("G")
        return float(avail)
    return 0.0


def detect_hardware() -> HardwareInfo:
    """Detect GPU, RAM, and disk available on this machine."""
    gpu_model, vram_gb = _parse_vram()
    ram_gb = _parse_ram()
    disk_gb = _parse_disk()

    return HardwareInfo(
        has_gpu=vram_gb > 0,
        gpu_model=gpu_model,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        disk_gb=disk_gb,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/git/personal/oci-turboinference && python -m pytest tests/test_detect.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add profiler/ tests/test_detect.py
git commit -m "feat: hardware detection module (GPU/RAM/disk)"
```

---

### Task 2: Profiler - Curated Models Table

**Files:**
- Create: `profiler/curated_models.yaml`
- Create: `profiler/strategy.py`
- Create: `tests/test_strategy.py`

**Step 1: Write the failing test**

```python
# tests/test_strategy.py
import pytest
from profiler.detect import HardwareInfo
from profiler.strategy import pick_strategy, InferenceConfig


def test_strategy_qwen35_a3b_on_a10():
    """Qwen3.5-35B-A3B (MoE) fits on A10 with vLLM AWQ."""
    hw = HardwareInfo(has_gpu=True, gpu_model="NVIDIA A10", vram_gb=24, ram_gb=240, disk_gb=150)
    config = pick_strategy("Qwen/Qwen3.5-35B-A3B", hw)

    assert config.backend == "vllm"
    assert config.quant_type in ("AWQ", "GPTQ", "Q4_K_M")
    assert config.n_gpu_layers > 0


def test_strategy_llama70b_on_a10():
    """Llama 70B needs extreme quant + heavy offload on A10."""
    hw = HardwareInfo(has_gpu=True, gpu_model="NVIDIA A10", vram_gb=24, ram_gb=240, disk_gb=150)
    config = pick_strategy("meta-llama/Llama-3.1-70B", hw)

    assert config.backend == "llamacpp"
    assert config.quant_type in ("IQ2_XXS", "Q2_K", "IQ1_S")
    assert config.n_gpu_layers > 0
    assert config.n_gpu_layers < 999  # partial offload


def test_strategy_phi4_cpu_only():
    """Phi-4 14B on CPU-only should use higher quality quant."""
    hw = HardwareInfo(has_gpu=False, gpu_model=None, vram_gb=0, ram_gb=128, disk_gb=200)
    config = pick_strategy("microsoft/phi-4", hw)

    assert config.backend == "llamacpp"
    assert config.n_gpu_layers == 0
    assert config.quant_type in ("Q4_K_M", "Q6_K", "Q8_0")


def test_strategy_deepseek_v3_on_a10():
    """DeepSeek-V3 671B: extreme quant, extreme offload."""
    hw = HardwareInfo(has_gpu=True, gpu_model="NVIDIA A10", vram_gb=24, ram_gb=240, disk_gb=500)
    config = pick_strategy("deepseek-ai/DeepSeek-V3", hw)

    assert config.backend == "llamacpp"
    assert config.quant_type in ("IQ1_S", "IQ2_XXS")
    assert config.estimated_tps < 2.0


def test_strategy_unknown_model_uses_fallback():
    """Unknown model should still return a config (via estimation)."""
    hw = HardwareInfo(has_gpu=True, gpu_model="NVIDIA A10", vram_gb=24, ram_gb=240, disk_gb=200)
    config = pick_strategy("some-org/unknown-model-42B", hw)

    assert config is not None
    assert config.backend in ("vllm", "llamacpp")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_strategy.py -v`
Expected: FAIL with "No module named 'profiler.strategy'"

**Step 3: Write the curated models table**

```yaml
# profiler/curated_models.yaml
# Known-good inference configs per model per hardware tier.
# gpu_a10: VM.GPU.A10.1 (24GB VRAM, 240GB RAM)
# cpu_64: CPU-only with 64GB RAM
# cpu_128: CPU-only with 128GB RAM
# cpu_256: CPU-only with 256GB+ RAM

models:
  "Qwen/Qwen3.5-35B-A3B":
    params_b: 35
    active_params_b: 3  # MoE
    gpu_a10:
      backend: vllm
      quant_type: AWQ
      model_url: "Qwen/Qwen3.5-35B-A3B-AWQ"
      n_gpu_layers: 999
      ctx_size: 16384
      cpu_offload_gb: 0
      estimated_tps: 25
    cpu_128:
      backend: llamacpp
      quant_type: Q4_K_M
      model_url: "bartowski/Qwen3.5-35B-A3B-GGUF:Q4_K_M"
      n_gpu_layers: 0
      ctx_size: 8192
      estimated_tps: 4

  "Qwen/Qwen3.5-27B":
    params_b: 27
    gpu_a10:
      backend: llamacpp
      quant_type: IQ4_XS
      model_url: "bartowski/Qwen3.5-27B-GGUF:IQ4_XS"
      n_gpu_layers: 40
      ctx_size: 8192
      cpu_offload_gb: 0
      estimated_tps: 8
    cpu_128:
      backend: llamacpp
      quant_type: Q4_K_M
      model_url: "bartowski/Qwen3.5-27B-GGUF:Q4_K_M"
      n_gpu_layers: 0
      ctx_size: 8192
      estimated_tps: 2

  "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF":
    params_b: 27
    gpu_a10:
      backend: llamacpp
      quant_type: Q2_K
      model_url: "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF:Q2_K"
      n_gpu_layers: 45
      ctx_size: 8192
      estimated_tps: 10
    cpu_128:
      backend: llamacpp
      quant_type: Q4_K_M
      model_url: "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF:Q4_K_M"
      n_gpu_layers: 0
      ctx_size: 8192
      estimated_tps: 2

  "meta-llama/Llama-3.1-70B":
    params_b: 70
    gpu_a10:
      backend: llamacpp
      quant_type: IQ2_XXS
      model_url: "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:IQ2_XXS"
      n_gpu_layers: 25
      ctx_size: 4096
      estimated_tps: 3
    cpu_256:
      backend: llamacpp
      quant_type: Q4_K_M
      model_url: "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF:Q4_K_M"
      n_gpu_layers: 0
      ctx_size: 4096
      estimated_tps: 1

  "meta-llama/Llama-3.3-70B-Instruct":
    params_b: 70
    gpu_a10:
      backend: llamacpp
      quant_type: IQ2_XXS
      model_url: "bartowski/Llama-3.3-70B-Instruct-GGUF:IQ2_XXS"
      n_gpu_layers: 25
      ctx_size: 4096
      estimated_tps: 3
    cpu_256:
      backend: llamacpp
      quant_type: Q4_K_M
      model_url: "bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M"
      n_gpu_layers: 0
      ctx_size: 4096
      estimated_tps: 1

  "deepseek-ai/DeepSeek-V3":
    params_b: 671
    gpu_a10:
      backend: llamacpp
      quant_type: IQ1_S
      model_url: "bartowski/DeepSeek-V3-0324-GGUF:IQ1_S"
      n_gpu_layers: 10
      ctx_size: 2048
      estimated_tps: 0.5
    cpu_256:
      backend: llamacpp
      quant_type: Q2_K
      model_url: "bartowski/DeepSeek-V3-0324-GGUF:Q2_K"
      n_gpu_layers: 0
      ctx_size: 2048
      estimated_tps: 0.2

  "mistralai/Mistral-Large-Instruct-2411":
    params_b: 123
    gpu_a10:
      backend: llamacpp
      quant_type: IQ1_S
      model_url: "bartowski/Mistral-Large-Instruct-2411-GGUF:IQ1_S"
      n_gpu_layers: 15
      ctx_size: 4096
      estimated_tps: 1.5
    cpu_256:
      backend: llamacpp
      quant_type: Q2_K
      model_url: "bartowski/Mistral-Large-Instruct-2411-GGUF:Q2_K"
      n_gpu_layers: 0
      ctx_size: 4096
      estimated_tps: 0.5

  "CohereForAI/c4ai-command-r-plus":
    params_b: 104
    gpu_a10:
      backend: llamacpp
      quant_type: IQ2_XXS
      model_url: "bartowski/c4ai-command-r-plus-GGUF:IQ2_XXS"
      n_gpu_layers: 15
      ctx_size: 4096
      estimated_tps: 1.5
    cpu_256:
      backend: llamacpp
      quant_type: Q2_K
      model_url: "bartowski/c4ai-command-r-plus-GGUF:Q2_K"
      n_gpu_layers: 0
      ctx_size: 4096
      estimated_tps: 0.3

  "microsoft/phi-4":
    params_b: 14
    gpu_a10:
      backend: vllm
      quant_type: AWQ
      model_url: "TechxGenus/phi-4-AWQ"
      n_gpu_layers: 999
      ctx_size: 16384
      cpu_offload_gb: 0
      estimated_tps: 40
    cpu_64:
      backend: llamacpp
      quant_type: Q6_K
      model_url: "bartowski/phi-4-GGUF:Q6_K"
      n_gpu_layers: 0
      ctx_size: 16384
      estimated_tps: 5

  "google/gemma-3-27b-it":
    params_b: 27
    gpu_a10:
      backend: llamacpp
      quant_type: Q3_K_M
      model_url: "bartowski/gemma-3-27b-it-GGUF:Q3_K_M"
      n_gpu_layers: 35
      ctx_size: 8192
      estimated_tps: 8
    cpu_128:
      backend: llamacpp
      quant_type: Q4_K_M
      model_url: "bartowski/gemma-3-27b-it-GGUF:Q4_K_M"
      n_gpu_layers: 0
      ctx_size: 8192
      estimated_tps: 2
```

**Step 4: Write the strategy module**

```python
# profiler/strategy.py
from dataclasses import dataclass, field
from pathlib import Path
import re
import yaml

from profiler.detect import HardwareInfo


@dataclass
class InferenceConfig:
    backend: str              # "vllm" or "llamacpp"
    model_url: str            # HuggingFace model ID or GGUF repo:quant
    quant_type: str           # "AWQ", "GPTQ", "Q4_K_M", "IQ2_XXS", etc.
    n_gpu_layers: int         # 0 = CPU-only, 999 = all on GPU
    ctx_size: int             # context window tokens
    cpu_offload_gb: int = 0   # vLLM --cpu-offload-gb
    estimated_tps: float = 0  # expected tokens/sec
    extra_args: dict = field(default_factory=dict)


def _load_curated_models() -> dict:
    """Load curated_models.yaml from same directory as this module."""
    yaml_path = Path(__file__).parent / "curated_models.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


def _select_hw_tier(hw: HardwareInfo) -> str:
    """Map hardware to the best matching tier key in curated_models.yaml."""
    if hw.has_gpu and hw.vram_gb >= 20:
        return "gpu_a10"
    elif hw.ram_gb >= 256:
        return "cpu_256"
    elif hw.ram_gb >= 128:
        return "cpu_128"
    else:
        return "cpu_64"


def _estimate_model_size_gb(model_id: str) -> float:
    """Rough estimate of model size from name (e.g. '42B' -> ~84GB at fp16)."""
    match = re.search(r"(\d+)[Bb]", model_id)
    if match:
        params_b = int(match.group(1))
        return params_b * 2.0  # ~2 bytes/param at fp16
    return 14.0  # default guess: 7B-ish model


def _fallback_strategy(model_id: str, hw: HardwareInfo) -> InferenceConfig:
    """Estimate a config for models not in the curated table."""
    estimated_size_gb = _estimate_model_size_gb(model_id)

    if hw.has_gpu:
        if estimated_size_gb <= hw.vram_gb * 0.85:
            # Fits in VRAM with AWQ
            return InferenceConfig(
                backend="vllm",
                model_url=model_id,
                quant_type="AWQ",
                n_gpu_layers=999,
                ctx_size=8192,
                estimated_tps=15,
            )
        elif estimated_size_gb <= hw.vram_gb * 2:
            # Needs moderate quantization
            return InferenceConfig(
                backend="llamacpp",
                model_url=model_id,
                quant_type="Q4_K_M",
                n_gpu_layers=int(hw.vram_gb / estimated_size_gb * 80),
                ctx_size=4096,
                estimated_tps=5,
            )
        else:
            # Extreme: heavy quant + offload
            return InferenceConfig(
                backend="llamacpp",
                model_url=model_id,
                quant_type="IQ2_XXS" if estimated_size_gb > hw.ram_gb * 0.5 else "Q2_K",
                n_gpu_layers=max(5, int(hw.vram_gb / estimated_size_gb * 60)),
                ctx_size=2048,
                estimated_tps=1,
            )
    else:
        # CPU-only
        if estimated_size_gb * 0.25 <= hw.ram_gb * 0.8:
            quant = "Q4_K_M"
        elif estimated_size_gb * 0.15 <= hw.ram_gb * 0.8:
            quant = "Q2_K"
        else:
            quant = "IQ1_S"

        return InferenceConfig(
            backend="llamacpp",
            model_url=model_id,
            quant_type=quant,
            n_gpu_layers=0,
            ctx_size=2048,
            estimated_tps=0.5,
        )


def pick_strategy(model_id: str, hw: HardwareInfo) -> InferenceConfig:
    """Pick the optimal inference config for a model on given hardware."""
    curated = _load_curated_models()
    tier = _select_hw_tier(hw)

    # Check curated table (exact match or partial match)
    for key, model_data in curated.items():
        if key == model_id or model_id in key or key in model_id:
            # Find the best tier: try exact, then fall through to lower
            tiers_to_try = [tier]
            if tier == "gpu_a10":
                tiers_to_try += ["cpu_256", "cpu_128", "cpu_64"]
            elif tier == "cpu_256":
                tiers_to_try += ["cpu_128", "cpu_64"]
            elif tier == "cpu_128":
                tiers_to_try += ["cpu_64"]

            for t in tiers_to_try:
                if t in model_data:
                    cfg = model_data[t]
                    return InferenceConfig(
                        backend=cfg["backend"],
                        model_url=cfg["model_url"],
                        quant_type=cfg["quant_type"],
                        n_gpu_layers=cfg["n_gpu_layers"],
                        ctx_size=cfg["ctx_size"],
                        cpu_offload_gb=cfg.get("cpu_offload_gb", 0),
                        estimated_tps=cfg.get("estimated_tps", 0),
                    )

    # Not in curated table: use estimation fallback
    return _fallback_strategy(model_id, hw)
```

**Step 5: Run tests to verify they pass**

Run: `pip install pyyaml && python -m pytest tests/test_strategy.py -v`
Expected: 5 passed

**Step 6: Commit**

```bash
git add profiler/curated_models.yaml profiler/strategy.py tests/test_strategy.py
git commit -m "feat: strategy engine with curated model table and fallback estimator"
```

---

### Task 3: Profiler - llmfit Client (Fallback)

**Files:**
- Create: `profiler/llmfit_client.py`
- Create: `tests/test_llmfit_client.py`

**Step 1: Write the failing test**

```python
# tests/test_llmfit_client.py
import pytest
from unittest.mock import patch, MagicMock
from profiler.llmfit_client import query_llmfit, LlmfitResult


def test_query_llmfit_success():
    """llmfit REST API returns a recommendation."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "some-org/model-42B",
        "fit": "good",
        "quantization": "Q4_K_M",
        "estimated_vram_gb": 20.5,
        "runtime": "llamacpp",
    }

    with patch("profiler.llmfit_client.httpx.get", return_value=mock_response):
        result = query_llmfit("some-org/model-42B", vram_gb=24, ram_gb=240)

    assert result is not None
    assert result.fit in ("perfect", "good", "marginal", "too_tight")
    assert result.quantization == "Q4_K_M"


def test_query_llmfit_unreachable():
    """llmfit not running returns None (graceful fallback)."""
    with patch("profiler.llmfit_client.httpx.get", side_effect=Exception("Connection refused")):
        result = query_llmfit("some-org/model-42B", vram_gb=24, ram_gb=240)

    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llmfit_client.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# profiler/llmfit_client.py
from dataclasses import dataclass
import httpx


LLMFIT_BASE_URL = "http://localhost:8787/api/v1"


@dataclass
class LlmfitResult:
    model: str
    fit: str             # "perfect", "good", "marginal", "too_tight"
    quantization: str    # "Q4_K_M", "Q2_K", etc.
    estimated_vram_gb: float
    runtime: str         # "llamacpp", "vllm", etc.


def query_llmfit(
    model_id: str,
    vram_gb: float = 0,
    ram_gb: float = 0,
    base_url: str = LLMFIT_BASE_URL,
) -> LlmfitResult | None:
    """Query llmfit REST API for a model recommendation. Returns None if llmfit is unavailable."""
    try:
        response = httpx.get(
            f"{base_url}/models/plan",
            params={"model": model_id, "vram_gb": vram_gb, "ram_gb": ram_gb},
            timeout=10,
        )
        if response.status_code != 200:
            return None

        data = response.json()
        return LlmfitResult(
            model=data.get("model", model_id),
            fit=data.get("fit", "unknown"),
            quantization=data.get("quantization", "Q4_K_M"),
            estimated_vram_gb=data.get("estimated_vram_gb", 0),
            runtime=data.get("runtime", "llamacpp"),
        )
    except Exception:
        return None
```

**Step 4: Run tests**

Run: `pip install httpx && python -m pytest tests/test_llmfit_client.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add profiler/llmfit_client.py tests/test_llmfit_client.py
git commit -m "feat: llmfit REST client with graceful fallback"
```

---

### Task 4: Install Scripts

**Files:**
- Create: `scripts/setup.sh`
- Create: `scripts/install-drivers.sh`
- Create: `scripts/install-llama-cpp.sh`
- Create: `scripts/install-vllm.sh`
- Create: `scripts/install-llmfit.sh`
- Create: `scripts/install-pi-agent.sh`
- Create: `scripts/start-inference.sh`

**Step 1: Write install-drivers.sh**

```bash
#!/bin/bash
# scripts/install-drivers.sh
# Install NVIDIA drivers + CUDA on Oracle Linux / Ubuntu.
# Skip gracefully if no GPU detected.
set -e

LOG_PREFIX="[install-drivers]"

if ! lspci | grep -qi nvidia; then
    echo "$LOG_PREFIX No NVIDIA GPU detected. Skipping driver install."
    exit 0
fi

if command -v nvidia-smi &>/dev/null; then
    echo "$LOG_PREFIX NVIDIA drivers already installed."
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    exit 0
fi

echo "$LOG_PREFIX Installing NVIDIA drivers + CUDA 12.4..."

# Detect distro
if [ -f /etc/oracle-release ] || [ -f /etc/redhat-release ]; then
    # Oracle Linux / RHEL
    dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    dnf install -y cuda-12-4
elif [ -f /etc/lsb-release ]; then
    # Ubuntu
    apt update
    apt install -y build-essential wget
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
    apt update
    apt install -y cuda-12-4
fi

echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

echo "$LOG_PREFIX NVIDIA driver installation complete."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

**Step 2: Write install-llama-cpp.sh**

```bash
#!/bin/bash
# scripts/install-llama-cpp.sh
# Build llama.cpp from source with CUDA (if available) or CPU-only.
set -e

LOG_PREFIX="[install-llama-cpp]"
INSTALL_DIR="/opt/llama.cpp"

if [ -x "$INSTALL_DIR/build/bin/llama-server" ]; then
    echo "$LOG_PREFIX llama.cpp already installed."
    exit 0
fi

echo "$LOG_PREFIX Cloning llama.cpp..."
apt install -y cmake build-essential git 2>/dev/null || dnf install -y cmake gcc-c++ git 2>/dev/null
git clone --depth 1 https://github.com/ggml-org/llama.cpp "$INSTALL_DIR"
cd "$INSTALL_DIR"

if command -v nvidia-smi &>/dev/null; then
    echo "$LOG_PREFIX Building with CUDA support..."
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
else
    echo "$LOG_PREFIX Building CPU-only..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
fi

cmake --build build --config Release -j$(nproc)

ln -sf "$INSTALL_DIR/build/bin/llama-server" /usr/local/bin/llama-server
ln -sf "$INSTALL_DIR/build/bin/llama-cli" /usr/local/bin/llama-cli

echo "$LOG_PREFIX llama.cpp installed at $INSTALL_DIR"
```

**Step 3: Write install-vllm.sh**

```bash
#!/bin/bash
# scripts/install-vllm.sh
# Install vLLM. Only useful on GPU instances.
set -e

LOG_PREFIX="[install-vllm]"

if ! command -v nvidia-smi &>/dev/null; then
    echo "$LOG_PREFIX No GPU detected. Skipping vLLM install (CPU-only uses llama.cpp)."
    exit 0
fi

if python3 -c "import vllm" 2>/dev/null; then
    echo "$LOG_PREFIX vLLM already installed."
    exit 0
fi

echo "$LOG_PREFIX Installing vLLM..."
pip install --upgrade pip
pip install vllm

echo "$LOG_PREFIX vLLM installed."
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

**Step 4: Write install-llmfit.sh**

```bash
#!/bin/bash
# scripts/install-llmfit.sh
# Build and install llmfit from source (Rust).
set -e

LOG_PREFIX="[install-llmfit]"

if command -v llmfit &>/dev/null; then
    echo "$LOG_PREFIX llmfit already installed."
    exit 0
fi

# Install Rust if not present
if ! command -v cargo &>/dev/null; then
    echo "$LOG_PREFIX Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "$LOG_PREFIX Cloning and building llmfit..."
TMPDIR=$(mktemp -d)
git clone --depth 1 https://github.com/AlexsJones/llmfit "$TMPDIR/llmfit"
cd "$TMPDIR/llmfit"
cargo build --release
cp target/release/llmfit /usr/local/bin/llmfit

rm -rf "$TMPDIR"
echo "$LOG_PREFIX llmfit installed."
```

**Step 5: Write install-pi-agent.sh**

```bash
#!/bin/bash
# scripts/install-pi-agent.sh
# Install Pi coding agent and configure it to use local inference.
set -e

LOG_PREFIX="[install-pi-agent]"
API_PORT="${API_PORT:-8080}"

# Install Node.js if not present
if ! command -v node &>/dev/null; then
    echo "$LOG_PREFIX Installing Node.js 22 LTS..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt install -y nodejs 2>/dev/null || dnf install -y nodejs 2>/dev/null
fi

# Install Pi coding agent
if command -v pi &>/dev/null; then
    echo "$LOG_PREFIX Pi coding agent already installed."
else
    echo "$LOG_PREFIX Installing Pi coding agent..."
    npm install -g @mariozechner/pi-coding-agent
fi

# Configure Pi to point to local inference server
PI_CONFIG_DIR="$HOME/.pi/agent"
mkdir -p "$PI_CONFIG_DIR"

cat > "$PI_CONFIG_DIR/models.json" << PIEOF
{
  "providers": {
    "local-turboinfer": {
      "baseUrl": "http://localhost:${API_PORT}/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        { "id": "turbo-model" }
      ]
    }
  }
}
PIEOF

echo "$LOG_PREFIX Pi coding agent configured. Run 'pi' to start."
```

**Step 6: Write start-inference.sh**

```bash
#!/bin/bash
# scripts/start-inference.sh
# Read profiler output and start the appropriate inference backend.
set -e

LOG_PREFIX="[start-inference]"
CONFIG_FILE="${1:-/opt/turboinference/inference-config.json}"
API_PORT="${API_PORT:-8080}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "$LOG_PREFIX Error: config file not found at $CONFIG_FILE"
    echo "$LOG_PREFIX Run the profiler first: python -m profiler.strategy"
    exit 1
fi

BACKEND=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['backend'])")
MODEL_URL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['model_url'])")
QUANT_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['quant_type'])")
N_GPU_LAYERS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['n_gpu_layers'])")
CTX_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['ctx_size'])")
CPU_OFFLOAD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('cpu_offload_gb', 0))")

echo "$LOG_PREFIX Backend: $BACKEND"
echo "$LOG_PREFIX Model: $MODEL_URL"
echo "$LOG_PREFIX Quant: $QUANT_TYPE"
echo "$LOG_PREFIX GPU layers: $N_GPU_LAYERS"
echo "$LOG_PREFIX Context: $CTX_SIZE"

if [ "$BACKEND" = "vllm" ]; then
    echo "$LOG_PREFIX Starting vLLM server on port $API_PORT..."
    exec vllm serve "$MODEL_URL" \
        --host 0.0.0.0 \
        --port "$API_PORT" \
        --dtype auto \
        --gpu-memory-utilization 0.92 \
        --max-model-len "$CTX_SIZE" \
        --cpu-offload-gb "$CPU_OFFLOAD"

elif [ "$BACKEND" = "llamacpp" ]; then
    # Download GGUF model if it's a HuggingFace URL
    MODEL_DIR="/opt/turboinference/models"
    mkdir -p "$MODEL_DIR"

    if [[ "$MODEL_URL" == *":"* ]]; then
        # Format: repo:quant_type - download via huggingface-cli
        REPO="${MODEL_URL%%:*}"
        QUANT="${MODEL_URL##*:}"
        echo "$LOG_PREFIX Downloading $REPO ($QUANT)..."
        pip install -q huggingface-hub
        MODEL_FILE=$(python3 -c "
from huggingface_hub import hf_hub_download
import glob, os
path = hf_hub_download('$REPO', filename='*${QUANT}*.gguf', local_dir='$MODEL_DIR')
print(path)
" 2>/dev/null || echo "")

        # Fallback: search for downloaded file
        if [ -z "$MODEL_FILE" ] || [ ! -f "$MODEL_FILE" ]; then
            MODEL_FILE=$(find "$MODEL_DIR" -name "*${QUANT}*.gguf" | head -1)
        fi
    else
        MODEL_FILE="$MODEL_URL"
    fi

    if [ -z "$MODEL_FILE" ] || [ ! -f "$MODEL_FILE" ]; then
        echo "$LOG_PREFIX Error: could not find/download model file."
        exit 1
    fi

    echo "$LOG_PREFIX Starting llama-server on port $API_PORT..."
    ARGS=(
        llama-server
        --host 0.0.0.0
        --port "$API_PORT"
        --model "$MODEL_FILE"
        --ctx-size "$CTX_SIZE"
        --n-gpu-layers "$N_GPU_LAYERS"
        --flash-attn
    )

    # Add KV cache quantization for large models to save VRAM
    if command -v nvidia-smi &>/dev/null; then
        ARGS+=(--cache-type-k q8_0 --cache-type-v q8_0)
    fi

    exec "${ARGS[@]}"
else
    echo "$LOG_PREFIX Unknown backend: $BACKEND"
    exit 1
fi
```

**Step 7: Write setup.sh (master script)**

```bash
#!/bin/bash
# scripts/setup.sh
# Master install script called by cloud-init.
# Installs everything, runs profiler, starts inference.
set -e

LOG_PREFIX="[turboinference-setup]"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="/opt/turboinference"
API_PORT="${API_PORT:-8080}"

# Model selection (passed via cloud-init template variable)
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
INSTALL_PI="${INSTALL_PI:-true}"

echo "$LOG_PREFIX Starting oci-turboinference setup"
echo "$LOG_PREFIX Model: $MODEL_ID"
echo "$LOG_PREFIX Install Pi agent: $INSTALL_PI"
echo ""

mkdir -p "$INSTALL_DIR"
cp -r "$PROJECT_DIR/profiler" "$INSTALL_DIR/"
cp -r "$PROJECT_DIR/scripts" "$INSTALL_DIR/"

# Step 1: NVIDIA drivers (skips if no GPU)
echo "$LOG_PREFIX [1/7] Installing NVIDIA drivers..."
bash "$SCRIPT_DIR/install-drivers.sh"

# Step 2: llama.cpp
echo "$LOG_PREFIX [2/7] Installing llama.cpp..."
bash "$SCRIPT_DIR/install-llama-cpp.sh"

# Step 3: vLLM (GPU only)
echo "$LOG_PREFIX [3/7] Installing vLLM..."
bash "$SCRIPT_DIR/install-vllm.sh"

# Step 4: llmfit
echo "$LOG_PREFIX [4/7] Installing llmfit..."
bash "$SCRIPT_DIR/install-llmfit.sh"

# Step 5: Run profiler
echo "$LOG_PREFIX [5/7] Running hardware profiler..."
pip install -q pyyaml httpx
python3 -c "
import json
from profiler.detect import detect_hardware
from profiler.strategy import pick_strategy
from dataclasses import asdict

hw = detect_hardware()
print(f'Hardware: GPU={hw.gpu_model}, VRAM={hw.vram_gb:.1f}GB, RAM={hw.ram_gb:.1f}GB, Disk={hw.disk_gb:.1f}GB')

config = pick_strategy('$MODEL_ID', hw)
print(f'Strategy: backend={config.backend}, quant={config.quant_type}, gpu_layers={config.n_gpu_layers}')
print(f'Estimated speed: {config.estimated_tps} tok/s')

with open('$INSTALL_DIR/inference-config.json', 'w') as f:
    json.dump(asdict(config), f, indent=2)
"

# Step 6: Pi coding agent
if [ "$INSTALL_PI" = "true" ]; then
    echo "$LOG_PREFIX [6/7] Installing Pi coding agent..."
    API_PORT="$API_PORT" bash "$SCRIPT_DIR/install-pi-agent.sh"
else
    echo "$LOG_PREFIX [6/7] Skipping Pi coding agent (disabled)."
fi

# Step 7: Start inference
echo "$LOG_PREFIX [7/7] Starting inference server..."
API_PORT="$API_PORT" nohup bash "$SCRIPT_DIR/start-inference.sh" "$INSTALL_DIR/inference-config.json" \
    > /var/log/turboinference.log 2>&1 &

echo ""
echo "$LOG_PREFIX Setup complete!"
echo "$LOG_PREFIX API endpoint: http://$(hostname -I | awk '{print $1}'):$API_PORT/v1"
echo "$LOG_PREFIX Logs: /var/log/turboinference.log"
echo "$LOG_PREFIX Run 'pi' to start the coding agent."
```

**Step 8: Make scripts executable and commit**

```bash
chmod +x scripts/*.sh
git add scripts/
git commit -m "feat: install scripts (drivers, llama.cpp, vllm, llmfit, pi, inference launcher)"
```

---

### Task 5: Terraform - OCI Resource Manager Stack

**Files:**
- Create: `terraform/main.tf`
- Create: `terraform/variables.tf`
- Create: `terraform/outputs.tf`
- Create: `terraform/schema.yaml`
- Create: `terraform/cloud-init.yaml`
- Create: `terraform/provider.tf`

**Step 1: Write provider.tf**

```hcl
# terraform/provider.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 5.0"
    }
  }
}

provider "oci" {
  region = var.region
}
```

**Step 2: Write variables.tf**

```hcl
# terraform/variables.tf

variable "compartment_ocid" {
  type        = string
  description = "OCI compartment to deploy into."
}

variable "region" {
  type        = string
  description = "OCI region."
  default     = "us-ashburn-1"
}

variable "availability_domain_number" {
  type        = number
  description = "Availability domain number (1, 2, or 3)."
  default     = 1
}

variable "model_selection" {
  type        = string
  description = "Model to deploy."
  default     = "Qwen/Qwen3.5-35B-A3B"
}

variable "custom_model_url" {
  type        = string
  description = "Custom HuggingFace model ID (only used when model_selection is 'custom')."
  default     = ""
}

variable "instance_type" {
  type        = string
  description = "GPU or CPU-only instance."
  default     = "gpu"
  validation {
    condition     = contains(["gpu", "cpu"], var.instance_type)
    error_message = "instance_type must be 'gpu' or 'cpu'."
  }
}

variable "cpu_ocpus" {
  type        = number
  description = "OCPUs for CPU-only flex shape."
  default     = 16
}

variable "cpu_ram_gb" {
  type        = number
  description = "RAM in GB for CPU-only flex shape."
  default     = 128
}

variable "ssh_public_key" {
  type        = string
  description = "SSH public key for instance access."
}

variable "api_allowed_cidr" {
  type        = string
  description = "CIDR block allowed to access the API endpoint."
  default     = "0.0.0.0/0"
}

variable "install_pi_agent" {
  type        = bool
  description = "Install Pi coding agent."
  default     = true
}

variable "api_port" {
  type        = number
  description = "Port for the inference API."
  default     = 8080
}

variable "boot_volume_gb" {
  type        = number
  description = "Boot volume size in GB."
  default     = 200
}
```

**Step 3: Write cloud-init.yaml**

```yaml
# terraform/cloud-init.yaml
#cloud-config
package_update: true
packages:
  - git
  - cmake
  - build-essential
  - python3-pip
  - curl
  - wget

write_files:
  - path: /opt/turboinference/setup-env.sh
    permissions: '0755'
    content: |
      export MODEL_ID="${model_id}"
      export API_PORT="${api_port}"
      export INSTALL_PI="${install_pi}"

runcmd:
  - echo "=== oci-turboinference cloud-init starting ==="
  - git clone --depth 1 https://github.com/jasperan/oci-turboinference /opt/turboinference/repo
  - source /opt/turboinference/setup-env.sh
  - cd /opt/turboinference/repo && bash scripts/setup.sh
  - echo "=== oci-turboinference cloud-init complete ==="
```

**Step 4: Write main.tf**

```hcl
# terraform/main.tf

locals {
  model_id = var.model_selection == "custom" ? var.custom_model_url : var.model_selection
  shape    = var.instance_type == "gpu" ? "VM.GPU.A10.1" : "VM.Standard.E5.Flex"
}

data "oci_identity_availability_domain" "ad" {
  compartment_id = var.compartment_ocid
  ad_number      = var.availability_domain_number
}

# Latest Oracle Linux 8 image (GPU or standard)
data "oci_core_images" "ol8" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Oracle Linux"
  operating_system_version = "8"
  shape                    = local.shape
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

# ─── Network ─────────────────────────────────────────────────
resource "oci_core_vcn" "turboinfer_vcn" {
  compartment_id = var.compartment_ocid
  display_name   = "turboinfer-vcn"
  cidr_blocks    = ["10.0.0.0/16"]
  dns_label      = "turboinfer"
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.turboinfer_vcn.id
  display_name   = "turboinfer-igw"
  enabled        = true
}

resource "oci_core_route_table" "rt" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.turboinfer_vcn.id
  display_name   = "turboinfer-rt"

  route_rules {
    network_entity_id = oci_core_internet_gateway.igw.id
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
  }
}

resource "oci_core_subnet" "public_subnet" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.turboinfer_vcn.id
  display_name      = "turboinfer-public"
  cidr_block        = "10.0.1.0/24"
  route_table_id    = oci_core_route_table.rt.id
  security_list_ids = [oci_core_security_list.sl.id]
  dns_label         = "public"
}

resource "oci_core_security_list" "sl" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.turboinfer_vcn.id
  display_name   = "turboinfer-sl"

  # Egress: allow all
  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  # SSH
  ingress_security_rules {
    protocol = "6" # TCP
    source   = var.api_allowed_cidr
    tcp_options {
      min = 22
      max = 22
    }
  }

  # Inference API
  ingress_security_rules {
    protocol = "6"
    source   = var.api_allowed_cidr
    tcp_options {
      min = var.api_port
      max = var.api_port
    }
  }
}

# ─── Compute ─────────────────────────────────────────────────
resource "oci_core_instance" "turboinfer" {
  availability_domain = data.oci_identity_availability_domain.ad.name
  compartment_id      = var.compartment_ocid
  display_name        = "turboinfer-${var.instance_type}"
  shape               = local.shape

  # Flex shape config (CPU-only needs this; GPU shapes ignore it)
  dynamic "shape_config" {
    for_each = var.instance_type == "cpu" ? [1] : []
    content {
      ocpus         = var.cpu_ocpus
      memory_in_gbs = var.cpu_ram_gb
    }
  }

  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ol8.images[0].id
    boot_volume_size_in_gbs = var.boot_volume_gb
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.public_subnet.id
    assign_public_ip = true
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data = base64encode(templatefile("${path.module}/cloud-init.yaml", {
      model_id   = local.model_id
      api_port   = var.api_port
      install_pi = var.install_pi_agent ? "true" : "false"
    }))
  }
}
```

**Step 5: Write outputs.tf**

```hcl
# terraform/outputs.tf

output "instance_ip" {
  description = "Public IP of the inference instance."
  value       = oci_core_instance.turboinfer.public_ip
}

output "api_endpoint" {
  description = "OpenAI-compatible API endpoint."
  value       = "http://${oci_core_instance.turboinfer.public_ip}:${var.api_port}/v1"
}

output "ssh_command" {
  description = "SSH into the instance."
  value       = "ssh opc@${oci_core_instance.turboinfer.public_ip}"
}

output "model_deployed" {
  description = "Model selected for deployment."
  value       = local.model_id
}

output "instance_type" {
  description = "Deployment type (gpu or cpu)."
  value       = var.instance_type
}
```

**Step 6: Write schema.yaml (OCI Resource Manager UI)**

```yaml
# terraform/schema.yaml
title: "OCI TurboInference - Run Huge LLMs on Small Hardware"
description: "One-click deploy: auto-profiles hardware, picks optimal quantization, starts inference, configures Pi coding agent."
schemaVersion: 1.1.0
version: "1.0.0"
locale: "en"

variableGroups:
  - title: "Model Selection"
    visible: true
    variables:
      - model_selection
      - custom_model_url

  - title: "Instance Configuration"
    visible: true
    variables:
      - compartment_ocid
      - region
      - availability_domain_number
      - instance_type
      - cpu_ocpus
      - cpu_ram_gb
      - boot_volume_gb

  - title: "Network & Security"
    visible: true
    variables:
      - ssh_public_key
      - api_allowed_cidr
      - api_port

  - title: "Coding Agent"
    visible: true
    variables:
      - install_pi_agent

variables:
  model_selection:
    type: enum
    title: "Model"
    description: "Select a model to deploy, or choose 'custom' to enter a HuggingFace model ID."
    default: "Qwen/Qwen3.5-35B-A3B"
    required: true
    enum:
      - "Qwen/Qwen3.5-35B-A3B"
      - "Qwen/Qwen3.5-27B"
      - "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF"
      - "meta-llama/Llama-3.1-70B"
      - "meta-llama/Llama-3.3-70B-Instruct"
      - "deepseek-ai/DeepSeek-V3"
      - "mistralai/Mistral-Large-Instruct-2411"
      - "CohereForAI/c4ai-command-r-plus"
      - "microsoft/phi-4"
      - "google/gemma-3-27b-it"
      - "custom"

  custom_model_url:
    type: string
    title: "Custom HuggingFace Model ID"
    description: "e.g., 'some-org/model-name' or full GGUF URL. Only used when Model is set to 'custom'."
    default: ""
    required: false
    visible: ${model_selection == "custom"}

  compartment_ocid:
    type: oci:identity:compartment:id
    title: "Compartment"
    required: true

  region:
    type: oci:identity:region:name
    title: "Region"
    required: true

  availability_domain_number:
    type: integer
    title: "Availability Domain"
    default: 1
    minimum: 1
    maximum: 3
    required: true

  instance_type:
    type: enum
    title: "Instance Type"
    description: "GPU (A10 24GB VRAM, ~$1/hr) or CPU-only (flex shape, ~$0.03/hr per OCPU)."
    default: "gpu"
    required: true
    enum:
      - "gpu"
      - "cpu"

  cpu_ocpus:
    type: integer
    title: "CPU OCPUs (CPU-only)"
    description: "Number of OCPUs for CPU-only shape."
    default: 16
    minimum: 4
    maximum: 64
    required: false
    visible: ${instance_type == "cpu"}

  cpu_ram_gb:
    type: integer
    title: "RAM in GB (CPU-only)"
    description: "System RAM for CPU-only shape. More RAM = bigger models."
    default: 128
    minimum: 32
    maximum: 1024
    required: false
    visible: ${instance_type == "cpu"}

  boot_volume_gb:
    type: integer
    title: "Boot Volume (GB)"
    description: "Disk space. Large models need 100-500GB."
    default: 200
    minimum: 100
    maximum: 1000
    required: true

  ssh_public_key:
    type: string
    title: "SSH Public Key"
    description: "Public SSH key for instance access."
    required: true

  api_allowed_cidr:
    type: string
    title: "Allowed CIDR for API"
    description: "IP range allowed to access the inference API. Default: open to all."
    default: "0.0.0.0/0"
    required: true

  api_port:
    type: integer
    title: "API Port"
    default: 8080
    minimum: 1024
    maximum: 65535
    required: true

  install_pi_agent:
    type: boolean
    title: "Install Pi Coding Agent"
    description: "Auto-install and configure Pi coding agent to use the deployed model."
    default: true
    required: false
```

**Step 7: Validate terraform config**

Run: `cd /home/ubuntu/git/personal/oci-turboinference/terraform && terraform init -backend=false && terraform validate`
Expected: "Success! The configuration is valid."

**Step 8: Commit**

```bash
git add terraform/
git commit -m "feat: OCI Resource Manager Stack (Terraform + cloud-init + schema)"
```

---

### Task 6: README and CLAUDE.md

**Files:**
- Create: `README.md`
- Create: `CLAUDE.md`

**Step 1: Write README.md**

Content should cover:
- Project mission (one paragraph)
- Quick start: OCI Resource Manager deploy (3 steps with screenshots placeholder)
- Quick start: Manual deploy on any Linux machine (`bash scripts/setup.sh`)
- Supported models table (from curated_models.yaml)
- Architecture diagram (ASCII)
- Three deployment tiers explained
- How the profiler works
- Pi coding agent integration
- v2 roadmap
- Contributing

**Step 2: Write CLAUDE.md**

Content should cover:
- Project overview
- Key commands (terraform, scripts, profiler)
- Architecture summary
- Testing: `python -m pytest tests/ -v`
- Directory structure
- Dependencies

**Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: README and CLAUDE.md"
```

---

### Task 7: Integration Test - End-to-End Profiler

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""
Integration test: run the full profiler pipeline on the current machine.
This test uses real hardware detection (no mocks).
"""
import pytest
from profiler.detect import detect_hardware
from profiler.strategy import pick_strategy


# All curated models should produce a valid config
CURATED_MODELS = [
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-27B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "mistralai/Mistral-Large-Instruct-2411",
    "CohereForAI/c4ai-command-r-plus",
    "microsoft/phi-4",
    "google/gemma-3-27b-it",
]


@pytest.fixture(scope="module")
def hardware():
    return detect_hardware()


def test_hardware_detection(hardware):
    """Hardware detection should return sane values."""
    assert hardware.ram_gb > 0
    assert hardware.disk_gb > 0
    # GPU may or may not be present


@pytest.mark.parametrize("model_id", CURATED_MODELS)
def test_curated_model_produces_config(hardware, model_id):
    """Every curated model should produce a valid InferenceConfig."""
    config = pick_strategy(model_id, hardware)

    assert config.backend in ("vllm", "llamacpp")
    assert config.quant_type != ""
    assert config.model_url != ""
    assert config.ctx_size > 0
    assert config.n_gpu_layers >= 0
    assert config.estimated_tps >= 0


def test_unknown_model_fallback(hardware):
    """Unknown models should produce a config via the fallback estimator."""
    config = pick_strategy("totally-unknown/mystery-model-99B", hardware)

    assert config is not None
    assert config.backend in ("vllm", "llamacpp")
```

**Step 2: Run integration tests**

Run: `python -m pytest tests/test_integration.py -v`
Expected: All pass (hardware detection uses real machine, strategy uses curated table)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end profiler integration tests"
```

---

### Task 8: GitHub Repo Setup + Push

**Step 1: Create GitHub repo**

```bash
cd /home/ubuntu/git/personal/oci-turboinference
gh repo create jasperan/oci-turboinference --public --source=. --push --description "Run huge LLMs on small hardware. One-click OCI deploy with auto-profiling."
```

**Step 2: Verify**

```bash
gh repo view jasperan/oci-turboinference
```

---

## Summary

| Task | What | Commit |
|------|------|--------|
| 1 | Hardware detection (GPU/RAM/disk) | `feat: hardware detection module` |
| 2 | Strategy engine + curated models | `feat: strategy engine with curated model table` |
| 3 | llmfit REST client | `feat: llmfit REST client with graceful fallback` |
| 4 | Install scripts (7 scripts) | `feat: install scripts` |
| 5 | Terraform OCI RM Stack | `feat: OCI Resource Manager Stack` |
| 6 | README + CLAUDE.md | `docs: README and CLAUDE.md` |
| 7 | Integration tests | `test: end-to-end profiler integration tests` |
| 8 | GitHub repo + push | (push only) |
