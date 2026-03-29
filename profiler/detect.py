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


def _parse_vram() -> tuple[str | None, float]:
    """Query nvidia-smi for GPU model name and total VRAM in GB."""
    output = _run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if output is None:
        return None, 0.0

    lines = output.splitlines()
    if not lines:
        return None, 0.0

    # Take the first GPU line
    line = lines[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None, 0.0

    gpu_model = parts[0]
    try:
        vram_mb = float(parts[1])
        vram_gb = vram_mb / 1024.0
    except ValueError:
        vram_gb = 0.0

    return gpu_model, vram_gb


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
    gpu_model, vram_gb = _parse_vram()
    ram_gb = _parse_ram()
    disk_gb = _parse_disk()

    return HardwareInfo(
        has_gpu=gpu_model is not None and vram_gb > 0,
        gpu_model=gpu_model,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        disk_gb=disk_gb,
    )
