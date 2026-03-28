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

# Run the profiler standalone
python -c "from profiler.detect import detect_hardware; print(detect_hardware())"
python -c "from profiler.strategy import pick_strategy; print(pick_strategy('Qwen/Qwen3.5-35B-A3B'))"

# Tests
python -m pytest tests/ -v
```

## Directory Structure

```
oci-turboinference/
  profiler/
    detect.py            # Hardware detection (GPU, RAM, disk)
    strategy.py          # Strategy engine: curated lookup + llmfit fallback
    llmfit_client.py     # Client wrapper for the llmfit Rust binary
    curated_models.yaml  # 10 pre-tested model configs across hardware tiers
  scripts/
    setup.sh             # Main orchestrator (calls all install scripts + profiler)
    install-drivers.sh   # NVIDIA driver installation
    install-llama-cpp.sh # llama.cpp build from source with CUDA
    install-vllm.sh      # vLLM pip install
    install-llmfit.sh    # llmfit Rust binary install
    install-pi-agent.sh  # Pi coding agent install
    start-inference.sh   # Launches vLLM or llama.cpp based on profiler output
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
  docs/plans/            # Design docs (local only, gitignored)
```

## Architecture Summary

Terraform creates an OCI compute instance with cloud-init. Cloud-init runs `setup.sh`, which installs NVIDIA drivers, llama.cpp, vLLM, and llmfit. The profiler's `detect.py` reads hardware specs, then `strategy.py` looks up the requested model in `curated_models.yaml`. If found, it returns a tested config. If not, it calls llmfit to estimate memory and pick a viable quant. Finally, `start-inference.sh` launches the chosen backend with an OpenAI-compatible API on port 8080.

## Testing

```bash
python -m pytest tests/ -v
```

Tests mock subprocess calls and hardware detection. No GPU or OCI credentials needed to run them.

## Dependencies

- Python 3.11+
- pyyaml (curated model config parsing)
- httpx (llmfit client HTTP calls)
- Terraform 1.5+ (infrastructure provisioning)
- Rust toolchain (for building llmfit from source, handled by install-llmfit.sh)
