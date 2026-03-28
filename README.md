# oci-turboinference

**Run Huge LLMs on Small Hardware.**

Run 400B+ parameter models on OCI GPU instances that cost ~$1/hr. The profiler detects your hardware, picks the best backend (vLLM or llama.cpp), selects the right quantization, and configures CPU/GPU offloading automatically. You get an OpenAI-compatible API endpoint without touching a config file.

## Quick Start: OCI One-Click Deploy

1. Download the [latest release zip](https://github.com/jasperan/oci-turboinference/releases) (contains Terraform + Resource Manager schema)
2. Upload it to **OCI Resource Manager** > Create Stack
3. Fill in your compartment, SSH key, and desired model. Click **Create**.

Cloud-init handles everything: drivers, backends, profiler, inference server.

## Quick Start: Manual Deploy

```bash
git clone https://github.com/jasperan/oci-turboinference.git
cd oci-turboinference
sudo MODEL_ID="Qwen/Qwen3.5-35B-A3B" bash scripts/setup.sh
```

The server starts on port 8080 by default. Override with `API_PORT=9000`.

## Three Deployment Tiers

| Tier | Hardware | Backends | Use Case |
|------|----------|----------|----------|
| **GPU** | A10 (24 GB VRAM) | vLLM, llama.cpp | Models up to ~35B at full speed |
| **GPU + Offload** | A10 (24 GB VRAM) + system RAM | llama.cpp with extreme quants (IQ1_S, IQ2_XXS) | 70B to 671B models, partial GPU acceleration |
| **CPU-only** | Flex shape (64 to 256 GB RAM) | llama.cpp in RAM | No GPU needed, runs anything that fits in memory |

## Supported Models

| Model | Params | GPU Strategy | CPU Strategy | Est. tok/s (GPU) | Est. tok/s (CPU) |
|-------|--------|-------------|-------------|-------------------|------------------|
| Qwen3.5-35B-A3B | 35B | vLLM AWQ, full GPU | Q4_K_M, 128 GB RAM | 45 | 8 |
| Qwen3.5-27B | 27B | llama.cpp IQ4_XS, 40 layers + 6 GB offload | Q4_K_M, 128 GB RAM | 12 | 4 |
| Qwen3.5-27B-Claude-Distilled | 27B | llama.cpp Q2_K, 50 layers + 4 GB offload | Q4_K_M, 128 GB RAM | 10 | 3.5 |
| Llama-3.1-70B | 70B | llama.cpp IQ2_XXS, 25 layers + 12 GB offload | Q4_K_M, 256 GB RAM | 3 | 2.5 |
| Llama-3.3-70B-Instruct | 70B | llama.cpp IQ2_XXS, 25 layers + 12 GB offload | Q4_K_M, 256 GB RAM | 3 | 2.5 |
| DeepSeek-V3 | 671B | llama.cpp IQ1_S, 10 layers + 18 GB offload | Q2_K, 256 GB RAM | 0.8 | 0.5 |
| Mistral-Large-2411 | 123B | llama.cpp IQ1_S, 15 layers + 16 GB offload | Q2_K, 256 GB RAM | 1.5 | 1 |
| Command R+ | 104B | llama.cpp IQ2_XXS, 15 layers + 14 GB offload | Q2_K, 256 GB RAM | 1.8 | 1.2 |
| Phi-4 | 14B | vLLM AWQ, full GPU | Q6_K, 64 GB RAM | 55 | 10 |
| Gemma-3-27B-IT | 27B | llama.cpp Q3_K_M, 35 layers + 5 GB offload | Q4_K_M, 128 GB RAM | 10 | 3.5 |

## How the Profiler Works

The profiler runs in 3 steps:

1. **Detect hardware.** Reads GPU model/VRAM via `nvidia-smi`, total RAM, and disk space.
2. **Check curated table.** Looks up the requested model in `curated_models.yaml` (10 pre-tested configs with known-good quant, layer split, and context size).
3. **Fall back to llmfit.** If the model isn't curated, the profiler calls [llmfit](https://github.com/mozilla-ai/llmfit) (a Rust tool) to estimate memory requirements and pick a viable quant + offload strategy.

The output is a single `InferenceConfig` that tells `start-inference.sh` exactly which backend to launch and how.

## Pi Coding Agent

The setup script installs [Pi](https://github.com/jasperan/pi) automatically (disable with `INSTALL_PI=false`). Once provisioned, SSH in and run:

```bash
pi
```

Pi auto-detects the local inference server and uses it as its backend. You get a coding agent on the same box as your model.

## Architecture

```
+-------------------+
|  OCI Resource Mgr |
|  (Terraform Stack)|
+--------+----------+
         |
         v
+--------+----------+
|    cloud-init      |
|  (cloud-init.yaml) |
+--------+----------+
         |
         v
+--------+----------+     +------------------+
|    setup.sh        +---->  install-drivers  |
|  (orchestrator)    |     |  install-llama   |
+--------+----------+     |  install-vllm    |
         |                 |  install-llmfit  |
         v                 |  install-pi      |
+--------+----------+     +------------------+
|  Profiler          |
|  detect.py         |
|  strategy.py       |
|  curated_models.yaml|
+--------+----------+
         |
         v
+--------+----------+
| start-inference.sh |
| (vLLM or llama.cpp)|
+--------+----------+
         |
         v
+--------+----------+
|  OpenAI-compatible |
|  API on :8080      |
+--------------------+
```

## v2 Roadmap

- **TurboQuant**: KV cache compression to squeeze larger context windows from the same VRAM budget
- **Multi-GPU**: Tensor parallel across 2x or 4x A10 instances for full-precision 70B+ inference
- **Web Dashboard**: Browser UI for model switching, throughput monitoring, and log tailing

## License

MIT
