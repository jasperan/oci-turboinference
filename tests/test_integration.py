"""End-to-end integration tests that run the profiler on real hardware."""

from __future__ import annotations

import pytest

from profiler.detect import detect_hardware, HardwareInfo
from profiler.strategy import pick_strategy, InferenceConfig


@pytest.fixture(scope="module")
def hardware() -> HardwareInfo:
    """Detect hardware once for the entire module."""
    return detect_hardware()


CURATED_MODEL_IDS = [
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-27B",
    "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "mistralai/Mistral-Large-Instruct-2411",
    "CohereForAI/c4ai-command-r-plus",
    "microsoft/phi-4",
    "google/gemma-3-27b-it",
]


def test_hardware_detection(hardware: HardwareInfo) -> None:
    """detect_hardware() returns sane values on real hardware."""
    assert hardware.ram_gb > 0, "RAM must be detected"
    assert hardware.disk_gb > 0, "Disk space must be detected"
    # GPU may or may not be present, so we just check the types
    assert isinstance(hardware.has_gpu, bool)
    assert isinstance(hardware.vram_gb, float)


@pytest.mark.parametrize("model_id", CURATED_MODEL_IDS)
def test_curated_model_produces_config(
    hardware: HardwareInfo, model_id: str
) -> None:
    """Each curated model must produce a valid InferenceConfig."""
    config = pick_strategy(model_id, hardware)

    assert isinstance(config, InferenceConfig)
    assert config.backend in ("vllm", "llamacpp")
    assert config.quant_type, "quant_type must be non-empty"
    assert config.model_url, "model_url must be non-empty"
    assert config.ctx_size > 0, "ctx_size must be positive"
    assert config.n_gpu_layers >= -1, "n_gpu_layers must be >= -1 (-1 means all layers)"
    assert config.estimated_tps >= 0, "estimated_tps must be non-negative"


def test_unknown_model_fallback(hardware: HardwareInfo) -> None:
    """Unknown models must still return a valid config via fallback."""
    config = pick_strategy("totally-unknown/mystery-model-99B", hardware)

    assert config is not None
    assert isinstance(config, InferenceConfig)
    assert config.backend in ("vllm", "llamacpp")
    assert config.quant_type, "quant_type must be non-empty"
    assert config.model_url, "model_url must be non-empty"
    assert config.ctx_size > 0
    assert config.n_gpu_layers >= -1
    assert config.estimated_tps >= 0
