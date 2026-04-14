# CLAUDE.md

## Project Overview

oci-turboinference provisions OCI compute instances for LLM inference, automatically selecting the best backend and quantization for the available hardware. It ships as a Terraform stack (with OCI Resource Manager support) and a Python profiler that maps models to tested configurations.

## Key Commands

```bash
# Terraform
cd terraform && terraform init && terraform plan
terraform apply -auto-approve

# Run setup manually (on an OCI instance)
sudo MODEL_ID="Qwen/Qwen3.5-35B-A3B" API_PORT=8080 bash scripts/setup.sh

# Disable Pi agent install during setup
sudo MODEL_ID="Qwen/Qwen3.5-35B-A3B" INSTALL_PI=false bash scripts/setup.sh

# Run the profiler standalone
python -c "from profiler.detect import detect_hardware; print(detect_hardware())"
python -c "from profiler.strategy import pick_strategy; print(pick_strategy('Qwen/Qwen3.5-35B-A3B'))"

# Run the benchmark against a live inference server
python -m profiler.benchmark --port 8080
python -m profiler.benchmark --base-url http://localhost:8080 --prompts short,medium --output-dir benchmarks/

# Tests (unit only, no GPU or OCI credentials needed)
python -m pytest tests/ -v -k "not test_integration"

# Integration tests (require real hardware)
python -m pytest tests/test_integration.py -v

# Manage the systemd service (on a provisioned instance)
sudo systemctl status turboinference
sudo journalctl -u turboinference -f
sudo systemctl restart turboinference
```

## Directory Structure

```
oci-turboinference/
  profiler/
    detect.py            # Hardware detection (GPU count, VRAM sum, RAM, disk)
    strategy.py          # Strategy engine: curated lookup + llmfit fallback + tensor parallel
    llmfit_client.py     # Client wrapper for the llmfit Rust binary
    benchmark.py         # Benchmark runner: TTFT + tok/s against live server, writes JSON + log
    curated_models.yaml  # 10 pre-tested model configs across hardware tiers
  scripts/
    setup.sh             # Main orchestrator (calls all install scripts + profiler)
    install-drivers.sh   # NVIDIA driver installation
    install-llama-cpp.sh # llama.cpp build from source with CUDA
    install-vllm.sh      # vLLM pip install
    install-llmfit.sh    # llmfit Rust binary install
    install-nginx-auth.sh # nginx reverse proxy with Bearer token auth on port 8443
    install-pi-agent.sh  # Pi coding agent install (skipped if INSTALL_PI=false)
    start-inference.sh   # Launches vLLM or llama.cpp based on profiler output
    turboinference.service # systemd unit (deployed to /etc/systemd/system/)
    wait-for-health.sh   # Polls /health until server is ready
  terraform/
    main.tf              # OCI compute instance resource
    variables.tf         # Input variables (shape, model, compartment)
    outputs.tf           # Instance IP, inference URL
    provider.tf          # OCI provider config
    schema.yaml          # OCI Resource Manager UI schema
    cloud-init.yaml      # Cloud-init template that runs setup.sh
  tests/
    test_detect.py       # Unit tests for hardware detection
    test_strategy.py     # Unit tests for strategy engine
    test_llmfit_client.py # Unit tests for llmfit client
    test_benchmark.py    # Unit tests for benchmark parsing/formatting
    test_integration.py  # Integration tests (real hardware, no mocks)
  benchmarks/            # JSON + log output from benchmark runs (gitignored)
  docs/plans/            # Design docs (local only, gitignored)
```

## Architecture Summary

Terraform creates an OCI compute instance with cloud-init. Cloud-init runs `setup.sh`, which installs NVIDIA drivers, llama.cpp, vLLM, and llmfit. The profiler's `detect.py` reads hardware specs, then `strategy.py` looks up the requested model in `curated_models.yaml`. If found, it returns a tested config. If not, it calls llmfit to estimate memory and pick a viable quant. Finally, `start-inference.sh` launches the chosen backend with an OpenAI-compatible API on port 8080.

## Testing

Unit tests mock subprocess calls and hardware detection. No GPU or OCI credentials needed:

```bash
python -m pytest tests/ -v -k "not test_integration"
```

Integration tests in `test_integration.py` call `detect_hardware()` and `pick_strategy()` for all 10 curated model IDs against real hardware. Run only on provisioned OCI instances:

```bash
python -m pytest tests/test_integration.py -v
```

Benchmark output (JSON + .log) lands in `benchmarks/` by default. Existing results from real A10 runs are committed there as reference data.

## Dependencies

- Python 3.11+
- pyyaml (curated model config parsing)
- httpx (llmfit client HTTP calls and benchmark streaming)
- Terraform 1.5+ (infrastructure provisioning)
- Rust toolchain (for building llmfit from source, handled by install-llmfit.sh)

No `pyproject.toml` or `requirements.txt` — dependencies are installed by `setup.sh` on the instance directly.

## Gotchas

- **nginx auth proxy**: `install-nginx-auth.sh` puts an nginx proxy on port 8443 (Bearer token auth) in front of the raw inference port 8080. The API key is saved to `/opt/turboinference/api-key`. The raw port 8080 is not firewalled by this script — OCI security list rules handle that.
- **systemd service**: `turboinference.service` reads env from `/opt/turboinference/inference-env` and the config from `/opt/turboinference/inference-config.json`. Both are written by `setup.sh`. To reload after a config change, restart the unit: `sudo systemctl restart turboinference`.
- **Pi agent**: installed by default via `install-pi-agent.sh`. Set `INSTALL_PI=false` to skip it.
- **benchmark module**: import path is `profiler.benchmark`, not a standalone script. Run via `python -m profiler.benchmark` from the repo root.
- **integration tests vs unit tests**: `test_integration.py` will fail on dev machines without GPUs or matching RAM. Always exclude with `-k "not test_integration"` locally.
