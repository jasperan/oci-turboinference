"""Tests for the strategy engine."""

from profiler.detect import HardwareInfo
from profiler.strategy import InferenceConfig, pick_strategy


# -- Hardware fixtures --

def _hw_a10() -> HardwareInfo:
    """A10 GPU with 24GB VRAM, 64GB RAM."""
    return HardwareInfo(has_gpu=True, gpu_model="NVIDIA A10", vram_gb=24.0, ram_gb=64.0, disk_gb=500.0)


def _hw_cpu_64() -> HardwareInfo:
    """CPU-only with 64GB RAM."""
    return HardwareInfo(has_gpu=False, gpu_model=None, vram_gb=0.0, ram_gb=64.0, disk_gb=500.0)


def _hw_cpu_128() -> HardwareInfo:
    """CPU-only with 128GB RAM."""
    return HardwareInfo(has_gpu=False, gpu_model=None, vram_gb=0.0, ram_gb=128.0, disk_gb=500.0)


def _hw_cpu_256() -> HardwareInfo:
    """CPU-only with 256GB RAM."""
    return HardwareInfo(has_gpu=False, gpu_model=None, vram_gb=0.0, ram_gb=256.0, disk_gb=1000.0)


# -- Tests --

def test_strategy_qwen35_a3b_on_a10():
    """Qwen3.5-35B-A3B on A10 should use vllm with AWQ."""
    cfg = pick_strategy("Qwen/Qwen3.5-35B-A3B", _hw_a10())
    assert isinstance(cfg, InferenceConfig)
    assert cfg.backend == "vllm"
    assert cfg.quant_type in ("AWQ", "GPTQ", "Q4_K_M")
    assert cfg.n_gpu_layers == -1  # full GPU
    assert cfg.estimated_tps > 0


def test_strategy_llama70b_on_a10():
    """Llama-3.1-70B on A10 should use llamacpp with heavy quant and partial offload."""
    cfg = pick_strategy("meta-llama/Llama-3.1-70B", _hw_a10())
    assert isinstance(cfg, InferenceConfig)
    assert cfg.backend == "llamacpp"
    assert cfg.quant_type in ("IQ2_XXS", "Q2_K", "IQ1_S")
    assert 0 < cfg.n_gpu_layers < 80  # partial, not full
    assert cfg.estimated_tps > 0


def test_strategy_phi4_cpu_only():
    """phi-4 on CPU-only (64GB) should use llamacpp with no GPU layers."""
    cfg = pick_strategy("microsoft/phi-4", _hw_cpu_64())
    assert isinstance(cfg, InferenceConfig)
    assert cfg.backend == "llamacpp"
    assert cfg.n_gpu_layers == 0
    assert cfg.quant_type in ("Q4_K_M", "Q6_K", "Q8_0")
    assert cfg.estimated_tps > 0


def test_strategy_deepseek_v3_on_a10():
    """DeepSeek-V3 (671B) on A10 should be llamacpp with extreme quant and low tps."""
    cfg = pick_strategy("deepseek-ai/DeepSeek-V3", _hw_a10())
    assert isinstance(cfg, InferenceConfig)
    assert cfg.backend == "llamacpp"
    assert cfg.quant_type in ("IQ1_S", "IQ2_XXS")
    assert cfg.estimated_tps < 2.0


def test_strategy_unknown_model_uses_fallback():
    """An unknown model should still return a valid InferenceConfig via fallback."""
    cfg = pick_strategy("some-org/totally-unknown-model-13B", _hw_a10())
    assert isinstance(cfg, InferenceConfig)
    assert cfg.backend in ("vllm", "llamacpp")
    assert cfg.quant_type != ""
    assert cfg.ctx_size > 0
    assert cfg.estimated_tps > 0
