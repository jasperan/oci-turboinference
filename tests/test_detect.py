"""Tests for profiler.detect hardware detection."""

from unittest.mock import patch

from profiler.detect import detect_hardware


NVIDIA_SMI_OUTPUT = "NVIDIA A10, 24564"

FREE_OUTPUT = """\
              total        used        free      shared  buff/cache   available
Mem:         257836       12345      200000        1234       45491      243210
Swap:          8191        1234        6957"""

DF_OUTPUT = """\
 Avail
  432G"""

FREE_CPU_ONLY = """\
              total        used        free      shared  buff/cache   available
Mem:          64000       12345       40000        1234       11655       50000
Swap:          8191        1234        6957"""


@patch("profiler.detect._run_cmd")
def test_detect_hardware_with_gpu(mock_run_cmd):
    """GPU present: nvidia-smi returns A10 with ~24GB VRAM."""

    def side_effect(cmd):
        if cmd[0] == "nvidia-smi":
            return NVIDIA_SMI_OUTPUT
        if cmd[0] == "free":
            return FREE_OUTPUT
        if cmd[0] == "df":
            return DF_OUTPUT
        return None

    mock_run_cmd.side_effect = side_effect

    info = detect_hardware()

    assert info.has_gpu is True
    assert info.gpu_model == "NVIDIA A10"
    assert 23.0 < info.vram_gb < 25.0  # ~24 GB
    assert info.ram_gb > 200.0
    assert info.disk_gb > 100.0


@patch("profiler.detect._run_cmd")
def test_detect_hardware_cpu_only(mock_run_cmd):
    """No GPU: nvidia-smi fails, system still reports RAM/disk."""

    def side_effect(cmd):
        if cmd[0] == "nvidia-smi":
            return None
        if cmd[0] == "free":
            return FREE_CPU_ONLY
        if cmd[0] == "df":
            return DF_OUTPUT
        return None

    mock_run_cmd.side_effect = side_effect

    info = detect_hardware()

    assert info.has_gpu is False
    assert info.gpu_model is None
    assert info.vram_gb == 0.0
    assert info.ram_gb > 60.0
