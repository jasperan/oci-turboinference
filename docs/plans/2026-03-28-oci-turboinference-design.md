# oci-turboinference Design Doc

**Date**: 2026-03-28
**Status**: Approved
**Author**: jasperan + Claude

## Mission

Run LLMs way too big for your GPU on a $1/hr OCI A10 instance. One-click deploy via OCI Resource Manager Stack. Auto-profiles hardware, picks optimal quantization + offload strategy, starts inference, hooks up Pi coding agent.

Target: 400B+ parameter models on 24GB VRAM (or even CPU-only).

## Background

People on Twitter (0xsero and others) have demonstrated running 400B+ parameter models on low VRAM at ~1 tok/s using extreme quantization and CPU offloading. The techniques exist but require manual configuration: picking the right GGUF quant, calculating layer splits, tuning context sizes. This project automates all of that and wraps it in a one-click OCI deployment.

## Three Deployment Tiers

| Tier | OCI Shape | Hardware | Backend | Strategy |
|------|-----------|----------|---------|----------|
| **GPU (preferred)** | VM.GPU.A10.1 | 24GB VRAM + 240GB RAM | vLLM (AWQ/GPTQ) or llama.cpp (extreme GGUF) | GPU-first, spill to RAM if needed |
| **GPU + heavy offload** | VM.GPU.A10.1 | Same | llama.cpp | Extreme quants (IQ1/Q2), most layers in RAM |
| **CPU-only** | VM.Standard.E5.Flex | 0 VRAM, up to 1TB RAM | llama.cpp (`-ngl 0`) | All layers in RAM, higher quality quants (Q4/Q6/Q8) since RAM is abundant |

## Technique Stack

Three independent techniques that stack together to maximize what fits:

| Layer | Technique | What it shrinks |
|-------|-----------|----------------|
| Weights | GGUF IQ1/Q2/Q4 or AWQ/GPTQ | Model parameters on disk + VRAM |
| KV Cache | TurboQuant (v2) | Attention state during inference |
| Offload | CPU/RAM layer split | Moves layers off GPU to system memory |

### Backend Selection Logic

| Model size vs VRAM | Backend | Why |
|---|---|---|
| Fits in VRAM with AWQ/GPTQ | **vLLM** (preferred) | Native quant, Marlin kernels, best throughput |
| Needs mild offload (a few GB) | **vLLM** | AWQ + `--cpu-offload-gb` |
| Needs extreme quant (IQ1/Q2) or heavy offload | **llama.cpp** | Best support for extreme GGUF quants + fine-grained layer splitting |
| CPU-only (no GPU) | **llama.cpp** | Only option, `-ngl 0` |

## Architecture

```
User Flow:
┌─────────────────────────────────────────────────────┐
│  OCI Console > Resource Manager > Create Stack      │
│  User picks: model (dropdown or HF URL) + shape     │
└──────────────────────┬──────────────────────────────┘
                       │ provisions instance
                       ▼
┌─────────────────────────────────────────────────────┐
│  cloud-init (runs on first boot)                     │
│                                                      │
│  1. Install NVIDIA drivers + CUDA (if GPU shape)     │
│  2. Build llama.cpp from source (CUDA or CPU)        │
│  3. Install vLLM (if GPU shape)                      │
│  4. Install llmfit (Rust binary)                     │
│  5. Install Pi coding agent (badlogic/pi-mono)       │
│  6. Run profiler: detect HW, pick strategy           │
│  7. Download optimal model file (GGUF or AWQ)        │
│  8. Start inference server (llama.cpp or vLLM)       │
│  9. Configure Pi -> localhost:8080                    │
│ 10. Expose OpenAI-compatible API on :8080            │
└─────────────────────────────────────────────────────┘
```

## Profiler Design

Python module that decides the optimal strategy.

```
detect_hardware()
  -> gpu_model, vram_gb (nvidia-smi or None)
  -> ram_gb (free -g)
  -> disk_gb

if model in CURATED_TABLE:
    config = CURATED_TABLE[model][gpu_type]
else:
    # Query llmfit REST API for recommendation
    llmfit_result = llmfit_recommend(model, vram_gb, ram_gb)
    config = translate_llmfit_to_config(llmfit_result)

config = {
    "backend": "vllm" | "llamacpp",
    "model_url": "https://huggingface.co/.../Q2_K.gguf",
    "quant_type": "Q2_K" | "AWQ" | "GPTQ",
    "n_gpu_layers": 28,        # 0 for CPU-only
    "ctx_size": 4096,           # reduced for huge models
    "cpu_offload_gb": 0,        # vLLM --cpu-offload-gb
    "estimated_tps": 1.2,       # tokens/sec estimate
    "vllm_args": {},            # extra vLLM flags if applicable
    "llamacpp_args": {}         # extra llama.cpp flags if applicable
}
```

## Curated Model Table (v1)

Known-good configs for ~10 popular models on A10 24GB:

| Model | Params | GPU (A10 24GB) | CPU-only (64GB RAM) |
|-------|--------|----------------|---------------------|
| Qwen3.5-35B-A3B | 35B (3B active) | vLLM AWQ, full GPU | llama.cpp Q4_K_M, -ngl 0 |
| Qwen3.5-27B | 27B | llama.cpp IQ4_XS, partial offload | llama.cpp Q4_K_M, -ngl 0 |
| Qwen3.5-27B-Claude-Distilled | 27B | llama.cpp Q2_K, partial offload | llama.cpp Q4_K_M, -ngl 0 |
| Llama 3.1 70B | 70B | llama.cpp IQ2_XXS, heavy offload | llama.cpp Q4_K_M, -ngl 0 |
| Llama 3.3 70B | 70B | llama.cpp IQ2_XXS, heavy offload | llama.cpp Q4_K_M, -ngl 0 |
| DeepSeek-V3 | 671B | llama.cpp IQ1_S, extreme offload | Needs 512GB+ RAM |
| Mistral Large 2 | 123B | llama.cpp IQ1_S, heavy offload | llama.cpp Q2_K, -ngl 0 |
| Command R+ | 104B | llama.cpp IQ2_XXS, heavy offload | llama.cpp Q2_K, -ngl 0 |
| Phi-4 | 14B | vLLM AWQ, full GPU | llama.cpp Q6_K, -ngl 0 |
| Gemma 3 27B | 27B | llama.cpp Q3_K_M, partial offload | llama.cpp Q4_K_M, -ngl 0 |

## Pi Coding Agent Integration

Pi (from badlogic/pi-mono) connects to any OpenAI-compatible endpoint.

Setup script will:
1. Clone `https://github.com/badlogic/pi-mono`
2. Install the coding-agent package (`packages/coding-agent`)
3. Configure it to point at `http://localhost:8080/v1`
4. Create a systemd service or convenience alias so user can just run `pi`

The inference server (llama.cpp or vLLM) already exposes `/v1/chat/completions`, so Pi works out of the box.

## OCI Resource Manager Stack

### Terraform Variables (schema.yaml)

User-facing inputs in OCI Console:

| Variable | Type | Description |
|----------|------|-------------|
| `compartment_id` | OCID | OCI compartment |
| `model_selection` | enum | Dropdown of curated models + "Custom HuggingFace URL" |
| `custom_model_url` | string | HF URL (only if model_selection = custom) |
| `instance_type` | enum | "GPU (A10 24GB)" or "CPU-only" |
| `cpu_shape_ocpus` | number | OCPUs for CPU-only shape (default: 16) |
| `cpu_shape_ram_gb` | number | RAM for CPU-only shape (default: 128) |
| `ssh_public_key` | string | SSH key for access |
| `api_allowed_cidr` | string | CIDR for API access (default: 0.0.0.0/0) |
| `install_pi_agent` | bool | Install Pi coding agent (default: true) |

### Infrastructure Created

- VCN + public subnet
- Network Security Group (SSH 22, API 8080, restricted by CIDR)
- Compute instance (GPU or CPU shape)
- Cloud-init bootstraps everything

### Outputs

- `instance_ip`: Public IP
- `api_endpoint`: `http://<ip>:8080/v1`
- `ssh_command`: `ssh -i <key> opc@<ip>`
- `model_loaded`: Which model + quant was selected
- `estimated_tps`: Expected tokens/sec

## Directory Structure

```
oci-turboinference/
├── terraform/
│   ├── main.tf                 # Compute, VCN, subnet, NSG
│   ├── variables.tf            # User inputs
│   ├── outputs.tf              # IP, endpoint, SSH command
│   ├── cloud-init.yaml         # Bootstrap template
│   └── schema.yaml             # OCI RM UI schema (dropdowns)
├── profiler/
│   ├── __init__.py
│   ├── detect.py               # VRAM/RAM/disk detection
│   ├── strategy.py             # Pick backend + quant + offload config
│   ├── curated_models.yaml     # Known-good configs per model per GPU
│   └── llmfit_client.py        # Query llmfit REST API
├── scripts/
│   ├── setup.sh                # Master install (called by cloud-init)
│   ├── install-drivers.sh      # NVIDIA drivers + CUDA
│   ├── install-llama-cpp.sh    # Build llama.cpp with CUDA or CPU
│   ├── install-vllm.sh         # pip install vllm
│   ├── install-llmfit.sh       # Build llmfit from source (Rust)
│   ├── install-pi-agent.sh     # Clone + install Pi coding agent
│   └── start-inference.sh      # Launch server with profiled config
├── config/
│   └── pi-config.template      # Pi agent config template
├── README.md
├── CLAUDE.md
└── .gitignore
```

## Security

- NSG restricts API port (8080) to user-specified CIDR. Default: open (user's responsibility to restrict).
- SSH key required, no password auth.
- No secrets stored in Terraform state (model URLs are public HuggingFace links).
- Inference endpoint has no auth by default (same as running Ollama locally). Users who want auth can put nginx + basic auth in front.

## v2 Roadmap (Deferred)

- **TurboQuant KV cache compression**: vLLM + Triton kernels, 2-3x KV cache savings at long context
- **Multi-GPU tensor parallelism**: A10.2 shape (2x A10 48GB total)
- **Web dashboard**: Model switching, VRAM monitoring, live tok/s metrics
- **More OCI shapes**: V100, A100 for users who want speed over cost
- **Auto-scaling**: Scale down to CPU-only when idle, scale up to GPU on demand
- **Model marketplace**: Pre-profiled configs contributed by community
