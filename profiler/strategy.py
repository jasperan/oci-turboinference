"""Strategy engine for selecting inference configurations."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from profiler.detect import HardwareInfo


@dataclass
class InferenceConfig:
    """Selected inference configuration for a model + hardware combo."""

    backend: str
    model_url: str
    quant_type: str
    n_gpu_layers: int
    ctx_size: int
    cpu_offload_gb: float = 0.0
    estimated_tps: float = 0.0


@functools.lru_cache(maxsize=1)
def _load_curated_models() -> dict:
    """Load the curated_models.yaml file and return parsed dict (cached)."""
    yaml_path = Path(__file__).parent / "curated_models.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


def _select_hw_tier(hw: HardwareInfo) -> str:
    """Map hardware capabilities to a tier key."""
    if hw.has_gpu and hw.vram_gb >= 20:
        return "gpu_a10"
    if hw.has_gpu and hw.vram_gb >= 8:
        return "gpu_small"
    if hw.ram_gb >= 200:
        return "cpu_256"
    if hw.ram_gb >= 100:
        return "cpu_128"
    return "cpu_64"


def _estimate_model_size_gb(model_id: str) -> float:
    """Estimate model size in GB from param count in the name.

    Looks for patterns like '70B', '35B', '14B', '671B' etc.
    Returns N * 2.0 as an fp16 size estimate.

    For MoE models with active-param markers (e.g. 'A3B', 'A2.7B'),
    uses the active params instead, since only a fraction activates per token.
    """
    match = re.search(r"(\d+(?:\.\d+)?)B", model_id, re.IGNORECASE)
    if match:
        params_b = float(match.group(1))
        # Check for MoE active params pattern like "A3B", "A2.7B"
        moe_match = re.search(r"A(\d+(?:\.\d+)?)B", model_id, re.IGNORECASE)
        if moe_match:
            active_b = float(moe_match.group(1))
            return active_b * 2.0  # MoE active params at fp16
        return params_b * 2.0
    return 14.0  # default guess: 7B model fp16


def _fallback_strategy(model_id: str, hw: HardwareInfo) -> InferenceConfig:
    """Generate a best-effort config for models not in the curated table."""
    size_gb = _estimate_model_size_gb(model_id)
    tier = _select_hw_tier(hw)

    if tier == "gpu_a10":
        # Small models fit entirely on GPU with vllm
        if size_gb <= hw.vram_gb * 0.9:
            return InferenceConfig(
                backend="vllm",
                model_url=model_id,
                quant_type="AWQ",
                n_gpu_layers=-1,
                ctx_size=16384,
                estimated_tps=30.0,
            )
        # Larger models: llamacpp with partial offload
        offload = max(0.0, size_gb - hw.vram_gb)
        # Estimate layers: more VRAM = more layers on GPU
        ratio = min(1.0, hw.vram_gb / size_gb) if size_gb > 0 else 0.5
        gpu_layers = int(ratio * 60)
        return InferenceConfig(
            backend="llamacpp",
            model_url=model_id,
            quant_type="IQ4_XS" if size_gb < 80 else "IQ2_XXS",
            n_gpu_layers=gpu_layers,
            ctx_size=4096,
            cpu_offload_gb=round(offload, 1),
            estimated_tps=max(0.5, 20.0 * ratio),
        )

    if tier == "gpu_small":
        # Small GPU (T4 16GB, RTX 3060 12GB): more aggressive quant than gpu_a10
        if size_gb <= hw.vram_gb * 0.8:
            return InferenceConfig(
                backend="llamacpp",
                model_url=model_id,
                quant_type="Q4_K_M",
                n_gpu_layers=-1,
                ctx_size=8192,
                estimated_tps=20.0,
            )
        # Partial offload with aggressive quantization
        offload = max(0.0, size_gb - hw.vram_gb)
        ratio = min(1.0, hw.vram_gb / size_gb) if size_gb > 0 else 0.3
        gpu_layers = int(ratio * 40)
        return InferenceConfig(
            backend="llamacpp",
            model_url=model_id,
            quant_type="IQ2_XXS" if size_gb < 40 else "IQ1_S",
            n_gpu_layers=gpu_layers,
            ctx_size=2048 if size_gb > 60 else 4096,
            cpu_offload_gb=round(offload, 1),
            estimated_tps=max(0.3, 12.0 * ratio),
        )

    # CPU-only tiers
    available_ram = hw.ram_gb * 0.85  # leave headroom
    if size_gb * 0.3 <= available_ram:
        quant = "Q4_K_M"
    elif size_gb * 0.2 <= available_ram:
        quant = "Q2_K"
    else:
        quant = "IQ1_S"

    return InferenceConfig(
        backend="llamacpp",
        model_url=model_id,
        quant_type=quant,
        n_gpu_layers=0,
        ctx_size=4096 if size_gb > 100 else 8192,
        estimated_tps=max(0.3, 10.0 * (available_ram / max(size_gb, 1))),
    )


def pick_strategy(model_id: str, hw: HardwareInfo) -> InferenceConfig:
    """Pick the best inference config for a model on given hardware.

    Checks the curated model table first (with partial matching on the
    model name), then falls back to estimation for unknown models.
    """
    curated = _load_curated_models()
    tier = _select_hw_tier(hw)

    # Matching with priority scoring:
    #   1 = exact match on full model ID
    #   2 = exact match on model name (after last "/")
    #   3 = substring containment
    # At the same priority, prefer the longer curated key (more specific).
    best_entry = None
    best_priority = 99
    best_key_len = 0

    model_short = model_id.split("/")[-1].lower()

    for curated_id, entry in curated.items():
        tiers = entry.get("tiers", {})
        if tier not in tiers:
            continue

        curated_short = curated_id.split("/")[-1].lower()

        if curated_id == model_id:
            priority = 1
        elif curated_short == model_short:
            priority = 2
        elif curated_short in model_short or model_short in curated_short:
            priority = 3
        else:
            continue

        key_len = len(curated_id)
        if priority < best_priority or (priority == best_priority and key_len > best_key_len):
            best_priority = priority
            best_key_len = key_len
            best_entry = tiers[tier]

    if best_entry is not None:
        cfg = best_entry
        return InferenceConfig(
            backend=cfg["backend"],
            model_url=cfg["model_url"],
            quant_type=cfg["quant_type"],
            n_gpu_layers=cfg["n_gpu_layers"],
            ctx_size=cfg["ctx_size"],
            cpu_offload_gb=cfg.get("cpu_offload_gb", 0),
            estimated_tps=cfg.get("estimated_tps", 0),
        )

    # No curated match: fall back to estimation
    return _fallback_strategy(model_id, hw)
