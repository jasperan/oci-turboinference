"""REST client for the llmfit model-fitting API."""

from dataclasses import dataclass

import httpx

LLMFIT_BASE_URL = "http://localhost:8787/api/v1"


@dataclass
class LlmfitResult:
    model: str
    fit: str  # "perfect" | "good" | "marginal" | "too_tight"
    quantization: str
    estimated_vram_gb: float
    runtime: str


def query_llmfit(
    model_id: str,
    vram_gb: float = 0,
    ram_gb: float = 0,
    base_url: str = LLMFIT_BASE_URL,
) -> LlmfitResult | None:
    """Query the llmfit REST API for a model fitting plan.

    Returns None on any failure (connection refused, timeout, bad JSON, etc.).
    """
    try:
        resp = httpx.get(
            f"{base_url}/models/plan",
            params={"model": model_id, "vram_gb": vram_gb, "ram_gb": ram_gb},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return LlmfitResult(
            model=data["model"],
            fit=data["fit"],
            quantization=data["quantization"],
            estimated_vram_gb=float(data["estimated_vram_gb"]),
            runtime=data["runtime"],
        )
    except Exception:
        return None
