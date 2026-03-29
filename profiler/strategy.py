"""Strategy engine for selecting inference configurations."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
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
    extra_args: dict = field(default_factory=dict)


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
    if hw.ram_gb >= 200:
        return "cpu_256"
    if hw.ram_gb >= 100:
        return "cpu_128"
    return "cpu_64"


def _estimate_model_size_gb(model_id: str) -> float:
    """Estimate model size in GB from param count in the name.

    Looks for patterns like '70B', '35B', '14B', '671B' etc.
    Returns N * 2.0 as an fp16 size estimate.
    """
    match = re.search(r"(\d+(?:\.\d+)?)B", model_id, re.IGNORECASE)
    if match:
        params_b = float(match.group(1))
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

    # Exact match first
    if model_id in curated:
        tiers = curated[model_id].get("tiers", {})
        if tier in tiers:
            cfg = tiers[tier]
            return InferenceConfig(
                backend=cfg["backend"],
                model_url=cfg["model_url"],
                quant_type=cfg["quant_type"],
                n_gpu_layers=cfg["n_gpu_layers"],
                ctx_size=cfg["ctx_size"],
                cpu_offload_gb=cfg.get("cpu_offload_gb", 0),
                estimated_tps=cfg.get("estimated_tps", 0),
            )

    # Partial match: check if model_id appears as substring in any curated key
    # or if a curated key appears as substring in model_id
    for curated_id, entry in curated.items():
        curated_short = curated_id.split("/")[-1].lower()
        model_short = model_id.split("/")[-1].lower()
        if curated_short in model_short or model_short in curated_short:
            tiers = entry.get("tiers", {})
            if tier in tiers:
                cfg = tiers[tier]
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
