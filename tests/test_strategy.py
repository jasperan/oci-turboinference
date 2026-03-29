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
    """Qwen3.5-35B-A3B on A10 should use vllm (MoE model fits in VRAM)."""
    cfg = pick_strategy("Qwen/Qwen3.5-35B-A3B", _hw_a10())
    assert isinstance(cfg, InferenceConfig)
    assert cfg.backend == "vllm"
    assert cfg.quant_type in ("fp16", "AWQ", "GPTQ", "Q4_K_M")
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


def _hw_2xa10() -> HardwareInfo:
    """2x A10 GPUs with 48GB total VRAM, 128GB RAM."""
    return HardwareInfo(has_gpu=True, gpu_model="NVIDIA A10", vram_gb=48.0, ram_gb=128.0, disk_gb=500.0, gpu_count=2)


def _hw_4xa10() -> HardwareInfo:
    """4x A10 GPUs with 96GB total VRAM, 256GB RAM."""
    return HardwareInfo(has_gpu=True, gpu_model="NVIDIA A10", vram_gb=96.0, ram_gb=256.0, disk_gb=1000.0, gpu_count=4)


def _hw_8xa100() -> HardwareInfo:
    """8x A100-80GB GPUs with 640GB total VRAM, 1TB RAM."""
    return HardwareInfo(has_gpu=True, gpu_model="NVIDIA A100-SXM4-80GB", vram_gb=640.0, ram_gb=1024.0, disk_gb=2000.0, gpu_count=8)


def _hw_t4() -> HardwareInfo:
    """T4 GPU with 16GB VRAM, 64GB RAM."""
    return HardwareInfo(has_gpu=True, gpu_model="NVIDIA T4", vram_gb=16.0, ram_gb=64.0, disk_gb=500.0)


def test_moe_size_estimation():
    """MoE models should estimate size from active params, not total."""
    from profiler.strategy import _estimate_model_size_gb
    # Qwen3.5-35B-A3B: total=35B but active=3B, so 3*2=6GB not 35*2=70GB
    size = _estimate_model_size_gb("Qwen3.5-35B-A3B")
    assert size == 6.0
    # Non-MoE model should use total params
    size = _estimate_model_size_gb("Llama-3.1-70B")
    assert size == 70.0 * 2.0


def test_gpu_small_tier_fallback():
    """T4 16GB should get gpu_small tier with more aggressive quant."""
    cfg = pick_strategy("some-org/unknown-7B-model", _hw_t4())
    assert isinstance(cfg, InferenceConfig)
    assert cfg.backend == "llamacpp"
    # 7B fp16 = 14GB, T4 has 16GB, so it should fit with Q4_K_M
    assert cfg.quant_type in ("Q4_K_M", "IQ2_XXS", "IQ1_S")
    assert cfg.estimated_tps > 0


def test_gpu_small_large_model():
    """Large model on T4 should get aggressive quant with partial offload."""
    cfg = pick_strategy("some-org/unknown-70B-model", _hw_t4())
    assert isinstance(cfg, InferenceConfig)
    assert cfg.backend == "llamacpp"
    assert cfg.quant_type in ("IQ2_XXS", "IQ1_S")
    assert cfg.n_gpu_layers > 0
    assert cfg.cpu_offload_gb > 0


def test_tighter_matching_exact_preferred():
    """Exact full-ID match should beat substring match."""
    cfg = pick_strategy("Qwen/Qwen3.5-27B", _hw_a10())
    assert isinstance(cfg, InferenceConfig)
    # Should match the exact Qwen/Qwen3.5-27B entry, not the longer distilled variant
    assert "unsloth/Qwen3.5-27B-GGUF" == cfg.model_url


def test_partial_match_prefers_exact():
    """'Qwen/Qwen3.5-27B' should match the 27B entry, not the Jackrong distilled variant."""
    cfg = pick_strategy("Qwen/Qwen3.5-27B", _hw_a10())
    assert isinstance(cfg, InferenceConfig)
    # Must NOT pick the Jackrong distilled model URL
    assert "Jackrong" not in cfg.model_url
    assert "unsloth/Qwen3.5-27B-GGUF" == cfg.model_url


def test_gpu_64_tier_fallback():
    """2x A10 (48GB) should get gpu_64 tier for a 27B model."""
    cfg = pick_strategy("some-org/unknown-27B-model", _hw_2xa10())
    assert isinstance(cfg, InferenceConfig)
    # 27B fp16 = 54GB > 48GB, so AWQ (54*0.25=13.5GB) fits easily
    assert cfg.backend == "vllm"
    assert cfg.quant_type in ("fp16", "AWQ")
    assert cfg.tensor_parallel == 2  # auto-set from gpu_count


def test_gpu_96_tier_fallback():
    """4x A10 (96GB) should handle 70B model at fp16."""
    cfg = pick_strategy("some-org/unknown-70B-model", _hw_4xa10())
    assert isinstance(cfg, InferenceConfig)
    # 70B fp16 = 140GB > 96GB, but AWQ (140*0.25=35GB) fits
    assert cfg.backend == "vllm"
    assert cfg.tensor_parallel == 4


def test_gpu_192_tier_fallback():
    """8x A100 (640GB) should run 120B at fp16 with tensor parallel."""
    cfg = pick_strategy("some-org/unknown-120B-model", _hw_8xa100())
    assert isinstance(cfg, InferenceConfig)
    # 120B fp16 = 240GB < 640GB, fits at fp16
    assert cfg.backend == "vllm"
    assert cfg.quant_type == "fp16"
    assert cfg.tensor_parallel == 8
    assert cfg.ctx_size == 32768  # large ctx for 192+ tier


def test_gpu_192_huge_model():
    """8x A100 with a truly massive model should still produce a valid config."""
    cfg = pick_strategy("some-org/unknown-671B-model", _hw_8xa100())
    assert isinstance(cfg, InferenceConfig)
    # 671B fp16 = 1342GB > 640GB, needs quant or offload
    assert cfg.quant_type in ("AWQ", "Q4_K_M", "IQ2_XXS", "IQ4_XS")
    assert cfg.estimated_tps > 0


def test_cpu_256_tier():
    """Llama 70B on CPU-only 512GB RAM should get a CPU config with decent quant."""
    hw = HardwareInfo(has_gpu=False, gpu_model=None, vram_gb=0.0, ram_gb=512.0, disk_gb=2000.0)
    cfg = pick_strategy("meta-llama/Llama-3.1-70B", hw)
    assert isinstance(cfg, InferenceConfig)
    assert cfg.n_gpu_layers == 0  # no GPU
    assert cfg.backend == "llamacpp"
    # 70B fp16 = 140GB, 512GB RAM is plenty, so should get decent quant (not extreme)
    assert cfg.quant_type in ("Q4_K_M", "Q6_K", "Q8_0", "Q2_K")
    assert cfg.estimated_tps > 0
