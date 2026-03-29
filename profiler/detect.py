"""Hardware detection for OCI TurboInference profiler."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""

    has_gpu: bool
    gpu_model: str | None
    vram_gb: float
    ram_gb: float
    disk_gb: float
    gpu_count: int = 1


def _run_cmd(cmd: list[str]) -> str | None:
    """Run a shell command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _parse_vram() -> tuple[str | None, float, int]:
    """Query nvidia-smi for GPU model name, total VRAM in GB, and GPU count."""
    output = _run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if output is None:
        return None, 0.0, 0

    lines = [l for l in output.splitlines() if l.strip()]
    if not lines:
        return None, 0.0, 0

    # Take GPU model from first line
    first_parts = [p.strip() for p in lines[0].split(",")]
    if len(first_parts) < 2:
        return None, 0.0, 0

    gpu_model = first_parts[0]

    # Sum VRAM across all GPUs
    total_vram_gb = 0.0
    gpu_count = 0
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                total_vram_gb += float(parts[1]) / 1024.0
                gpu_count += 1
            except ValueError:
                pass

    return gpu_model, total_vram_gb, gpu_count


def _parse_ram() -> float:
    """Get total system RAM in GB from `free -m`."""
    output = _run_cmd(["free", "-m"])
    if output is None:
        return 0.0

    for line in output.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1]) / 1024.0
                except ValueError:
                    return 0.0
    return 0.0


def _parse_disk() -> float:
    """Get available disk space in GB from `df` on root filesystem."""
    output = _run_cmd(["df", "--output=avail", "-BG", "/"])
    if output is None:
        return 0.0

    lines = output.splitlines()
    # Second line has the value (first is header)
    if len(lines) >= 2:
        val = lines[1].strip().rstrip("G")
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0


def detect_hardware() -> HardwareInfo:
    """Detect hardware capabilities and return a HardwareInfo summary."""
    gpu_model, vram_gb, gpu_count = _parse_vram()
    ram_gb = _parse_ram()
    disk_gb = _parse_disk()

    return HardwareInfo(
        has_gpu=gpu_model is not None and vram_gb > 0,
        gpu_model=gpu_model,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        disk_gb=disk_gb,
        gpu_count=gpu_count,
    )
