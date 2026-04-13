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
    tensor_parallel: int = 1
    tier: str = "curated"
    throughput_class: str = "interactive"
    estimated_ttft_s: float = 0.0
    warning: str | None = None
    draft_model: str | None = None
    draft_quantization: str | None = None
    draft_gpu_layers: int | None = None


def classify_throughput(tok_s: float) -> str:
    """Classify tokens/sec into a human-readable throughput tier."""
    if tok_s >= 5.0:
        return "interactive"
    if tok_s >= 1.0:
        return "batch"
    if tok_s >= 0.1:
        return "offline"
    return "glacial"


@functools.lru_cache(maxsize=1)
def _load_curated_models() -> dict:
    """Load the curated_models.yaml file and return parsed dict (cached)."""
    yaml_path = Path(__file__).parent / "curated_models.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


def _select_hw_tier(hw: HardwareInfo) -> str:
    """Map hardware capabilities to a tier key."""
    if hw.has_gpu:
        if hw.vram_gb >= 150:
            return "gpu_192"
        if hw.vram_gb >= 72:
            return "gpu_96"
        if hw.vram_gb >= 48:
            return "gpu_64"
        if hw.vram_gb >= 20:
            return "gpu_a10"
        if hw.vram_gb >= 8:
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


# Quantization ladder from highest quality to most aggressive.
# Each entry: (name, compression_ratio_vs_fp16)
_QUANT_LADDER = [
    ("fp16", 1.0),
    ("AWQ", 4.0),
    ("Q4_K_M", 3.56),
    ("IQ4_XS", 3.76),
    ("IQ2_XXS", 6.4),
    ("IQ1_S", 10.67),
]

_MIN_DISK_HEADROOM_GB = 20.0


def _try_gpu_fit(model_id, fp16_size_gb, hw):
    """Tier 2: Try to fit model on GPU with quantization + partial offload."""
    vram = hw.vram_gb
    ram = hw.ram_gb
    tp = hw.gpu_count if hw.gpu_count > 1 else 1

    for quant, ratio in _QUANT_LADDER:
        quant_size = fp16_size_gb / ratio

        # Fits entirely on GPU?
        if quant_size <= vram * 0.9:
            backend = "vllm" if quant in ("fp16", "AWQ") else "llamacpp"
            ctx = 32768 if vram >= 150 else 16384 if vram >= 48 else 8192
            tps = 50.0 if quant == "fp16" else 40.0 if quant == "AWQ" else 25.0
            return InferenceConfig(
                backend=backend, model_url=model_id, quant_type=quant,
                n_gpu_layers=-1, ctx_size=ctx, estimated_tps=tps,
                tensor_parallel=tp, tier="auto_fit",
                throughput_class=classify_throughput(tps),
            )

        # Fits with partial CPU offload?
        available_ram = ram * 0.85
        if quant_size <= vram + available_ram:
            offload = max(0.0, quant_size - vram)
            ratio_on_gpu = min(1.0, vram / quant_size) if quant_size > 0 else 0.5
            max_layers = 80 if vram >= 48 else 60 if vram >= 20 else 40
            gpu_layers = int(ratio_on_gpu * max_layers)
            ctx = 4096 if quant_size > 60 else 8192
            tps = max(0.5, 30.0 * ratio_on_gpu * (ratio / 4.0))
            return InferenceConfig(
                backend="llamacpp", model_url=model_id, quant_type=quant,
                n_gpu_layers=gpu_layers, ctx_size=ctx,
                cpu_offload_gb=round(offload, 1), estimated_tps=round(tps, 1),
                tensor_parallel=tp, tier="auto_fit",
                throughput_class=classify_throughput(tps),
            )

    return None


def _try_cpu_only(model_id, fp16_size_gb, hw):
    """Tier 3: Pure CPU with quantization."""
    available_ram = hw.ram_gb * 0.85
    for quant, ratio in _QUANT_LADDER:
        quant_size = fp16_size_gb / ratio
        if quant_size <= available_ram:
            tps = min(10.0, max(0.3, 8.0 * (available_ram / max(quant_size, 1)) * 0.5))
            ctx = 4096 if quant_size > 100 else 8192
            return InferenceConfig(
                backend="llamacpp", model_url=model_id, quant_type=quant,
                n_gpu_layers=0, ctx_size=ctx, estimated_tps=round(tps, 1),
                tier="cpu_only", throughput_class=classify_throughput(tps),
                warning="CPU-only mode: no GPU acceleration. Suitable for batch workloads.",
            )
    return None


def _try_layer_stream(model_id, fp16_size_gb, hw):
    """Tier 4: Layer streaming - load one layer at a time from disk."""
    min_disk = fp16_size_gb + _MIN_DISK_HEADROOM_GB
    if hw.disk_gb < min_disk:
        return None

    if hw.ram_gb > fp16_size_gb * 1.1:
        seconds_per_token = fp16_size_gb / 12.0
    else:
        seconds_per_token = fp16_size_gb / 3.0
    tps = round(max(0.01, 1.0 / seconds_per_token if seconds_per_token > 0 else 0.01), 3)
    ctx = 2048 if fp16_size_gb > 200 else 4096

    return InferenceConfig(
        backend="layer_stream", model_url=model_id, quant_type="fp16",
        n_gpu_layers=0, ctx_size=ctx, estimated_tps=tps,
        estimated_ttft_s=round(1.0 / tps, 1) if tps > 0 else 999.0,
        tier="layer_stream", throughput_class=classify_throughput(tps),
        warning="Layer streaming mode: very slow, suitable for batch/offline use only.",
    )


def _fallback_strategy(model_id: str, hw: HardwareInfo) -> InferenceConfig:
    """Generate a best-effort config for models not in the curated table.

    Walks through tiers 2-5 of the progressive fallback chain:
      Tier 2: Auto-fit with quantization on GPU
      Tier 3: CPU-only with quantization
      Tier 4: Layer streaming from disk
      Tier 5: Impossible (disk too small)
    """
    fp16_size_gb = _estimate_model_size_gb(model_id)

    # Tier 2: Try GPU fit
    if hw.has_gpu:
        config = _try_gpu_fit(model_id, fp16_size_gb, hw)
        if config is not None:
            return config

    # Tier 3: Try CPU-only
    config = _try_cpu_only(model_id, fp16_size_gb, hw)
    if config is not None:
        return config

    # Tier 4: Try layer streaming
    config = _try_layer_stream(model_id, fp16_size_gb, hw)
    if config is not None:
        return config

    # Tier 5: Impossible
    return InferenceConfig(
        backend="none", model_url=model_id, quant_type="none",
        n_gpu_layers=0, ctx_size=0, estimated_tps=0.0,
        tier="impossible", throughput_class="glacial",
        warning=f"Cannot run {model_id}: estimated {fp16_size_gb:.0f}GB fp16, "
                f"insufficient disk ({hw.disk_gb:.0f}GB available).",
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
        tps = cfg.get("estimated_tps", 0)
        config = InferenceConfig(
            backend=cfg["backend"],
            model_url=cfg["model_url"],
            quant_type=cfg["quant_type"],
            n_gpu_layers=cfg["n_gpu_layers"],
            ctx_size=cfg["ctx_size"],
            cpu_offload_gb=cfg.get("cpu_offload_gb", 0),
            estimated_tps=tps,
            tier="curated",
            throughput_class=classify_throughput(tps),
        )
    else:
        config = _fallback_strategy(model_id, hw)

    # Auto-set tensor parallelism for multi-GPU vLLM setups
    if config.backend == "vllm" and hw.gpu_count > 1:
        config.tensor_parallel = hw.gpu_count

    return config
