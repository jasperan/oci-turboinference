"""
Benchmarking module for oci-turboinference.

Runs a suite of prompts against an OpenAI-compatible inference server,
measures latency/throughput, and logs results to JSON + human-readable text.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    gpu_name: str = ""
    gpu_vram_total_mb: int = 0
    gpu_vram_used_before_mb: int = 0
    gpu_vram_used_after_mb: int = 0
    system_ram_total_mb: int = 0
    system_ram_used_mb: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelInfo:
    model_id: str = ""
    quant_type: str = ""
    backend: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PromptResult:
    prompt_name: str = ""
    prompt_text: str = ""
    ttft_ms: float = 0.0
    tokens_per_second: float = 0.0
    total_tokens: int = 0
    wall_clock_s: float = 0.0
    prompt_eval_time_ms: Optional[float] = None
    generation_text: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SummaryStats:
    avg_ttft_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    total_prompts: int = 0
    successful_prompts: int = 0
    total_wall_clock_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkRun:
    timestamp: str = ""
    hardware_info: HardwareInfo = field(default_factory=HardwareInfo)
    model_info: ModelInfo = field(default_factory=ModelInfo)
    prompt_results: list[PromptResult] = field(default_factory=list)
    summary: SummaryStats = field(default_factory=SummaryStats)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "hardware_info": self.hardware_info.to_dict(),
            "model_info": self.model_info.to_dict(),
            "prompt_results": [r.to_dict() for r in self.prompt_results],
            "summary": self.summary.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkRun":
        run = cls()
        run.timestamp = d.get("timestamp", "")
        run.hardware_info = HardwareInfo(**d.get("hardware_info", {}))
        run.model_info = ModelInfo(**d.get("model_info", {}))
        run.prompt_results = [PromptResult(**r) for r in d.get("prompt_results", [])]
        run.summary = SummaryStats(**d.get("summary", {}))
        return run


# ---------------------------------------------------------------------------
# Standard prompts
# ---------------------------------------------------------------------------

CONTEXT_PASSAGE = (
    "The history of computing spans several centuries, beginning with mechanical "
    "devices designed to automate arithmetic. Charles Babbage conceived the "
    "Analytical Engine in 1837, widely regarded as the first general-purpose "
    "computer design. Ada Lovelace wrote what is considered the first computer "
    "program for this machine. The development of vacuum tubes in the early 20th "
    "century enabled the construction of electronic computers. ENIAC, completed "
    "in 1945, was one of the earliest electronic general-purpose computers. It "
    "could perform 5000 additions per second. The invention of the transistor at "
    "Bell Labs in 1947 revolutionized computing by making machines smaller and "
    "more reliable. Jack Kilby and Robert Noyce independently invented the "
    "integrated circuit in 1958-1959, paving the way for microprocessors. Intel "
    "released the 4004 in 1971, the first commercially available microprocessor. "
    "The personal computer revolution began in the late 1970s with machines like "
    "the Apple II and the IBM PC. Tim Berners-Lee invented the World Wide Web in "
    "1989 at CERN, transforming how people access and share information. The "
    "rise of mobile computing in the 2000s, led by smartphones, brought "
    "computing power to billions of people worldwide. Cloud computing emerged as "
    "a dominant paradigm, allowing on-demand access to shared computing "
    "resources. Artificial intelligence, once a niche research area, has become "
    "central to modern computing. Machine learning techniques, particularly deep "
    "learning with neural networks, have achieved breakthroughs in image "
    "recognition, natural language processing, and game playing. Large language "
    "models trained on vast text corpora can generate human-like text and assist "
    "with a wide range of tasks. Quantum computing promises to solve certain "
    "problems exponentially faster than classical computers, though practical "
    "large-scale quantum computers remain years away. Edge computing brings "
    "processing closer to data sources, reducing latency for real-time "
    "applications. The convergence of AI, cloud, and edge computing is shaping "
    "the next generation of technology infrastructure. Security and privacy "
    "concerns continue to drive innovation in encryption, authentication, and "
    "data protection. Open source software has become foundational to modern "
    "computing, with Linux powering most servers and Android dominating mobile. "
    "The semiconductor industry follows Moore's Law, though physical limits are "
    "being approached. New architectures like neuromorphic chips and photonic "
    "processors may extend computational progress beyond silicon. Computing has "
    "transformed every aspect of human life, from healthcare and education to "
    "entertainment and transportation. The next decade will likely see further "
    "integration of AI into daily life, new forms of human-computer interaction, "
    "and continued advances in computational power and efficiency. "
    "Supercomputers now exceed exaflop performance, enabling climate modeling, "
    "drug discovery, and fundamental physics research at unprecedented scales."
)

STANDARD_PROMPTS: dict[str, str] = {
    "short": "What is 2+2?",
    "medium": "Explain how a CPU works in 200 words.",
    "long": (
        "Write a Python function that implements binary search, with docstrings "
        "and type hints. Then write 5 unit tests for it."
    ),
    "context": f"{CONTEXT_PASSAGE}\n\nSummarize the above in 3 bullet points.",
}


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return ""


def get_gpu_info() -> tuple[str, int, int]:
    """Returns (gpu_name, total_vram_mb, used_vram_mb)."""
    out = _run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return ("N/A", 0, 0)
    parts = out.split("\n")[0].split(",")
    if len(parts) < 3:
        return ("N/A", 0, 0)
    name = parts[0].strip()
    total = int(parts[1].strip())
    used = int(parts[2].strip())
    return (name, total, used)


def get_ram_info() -> tuple[int, int]:
    """Returns (total_mb, used_mb)."""
    out = _run_cmd(["free", "-m"])
    for line in out.split("\n"):
        if line.startswith("Mem:"):
            parts = line.split()
            return (int(parts[1]), int(parts[2]))
    return (0, 0)


def collect_hardware_info() -> HardwareInfo:
    gpu_name, gpu_total, gpu_used = get_gpu_info()
    ram_total, ram_used = get_ram_info()
    return HardwareInfo(
        gpu_name=gpu_name,
        gpu_vram_total_mb=gpu_total,
        gpu_vram_used_before_mb=gpu_used,
        system_ram_total_mb=ram_total,
        system_ram_used_mb=ram_used,
    )


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

def fetch_model_info(base_url: str) -> ModelInfo:
    """Query /v1/models to get model name and details."""
    info = ModelInfo()
    try:
        resp = httpx.get(f"{base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if models:
            info.model_id = models[0].get("id", "unknown")
    except Exception as exc:
        info.model_id = f"error: {exc}"
    return info


# ---------------------------------------------------------------------------
# Streaming benchmark
# ---------------------------------------------------------------------------

def run_prompt(
    base_url: str,
    prompt_name: str,
    prompt_text: str,
    model: str = "",
) -> PromptResult:
    """Send a chat completion with streaming and measure TTFT + throughput."""
    result = PromptResult(prompt_name=prompt_name, prompt_text=prompt_text)

    payload = {
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": True,
        "max_tokens": 1024,
    }
    if model:
        payload["model"] = model

    url = f"{base_url}/v1/chat/completions"
    token_count = 0
    chunks_text: list[str] = []
    first_token_time: Optional[float] = None

    try:
        start = time.perf_counter()
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", url, json=payload) as stream:
                stream.raise_for_status()
                for line in stream.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "") or delta.get("reasoning_content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
                        chunks_text.append(content)

                    # Check for usage info (some servers send it in the last chunk)
                    usage = chunk.get("usage")
                    if usage and usage.get("prompt_tokens"):
                        result.prompt_eval_time_ms = usage.get(
                            "prompt_eval_duration_ms"
                        )

        end = time.perf_counter()
        wall_clock = end - start

        result.total_tokens = token_count
        result.wall_clock_s = round(wall_clock, 3)
        result.generation_text = "".join(chunks_text)

        if first_token_time is not None:
            result.ttft_ms = round((first_token_time - start) * 1000, 1)

        if wall_clock > 0 and token_count > 0:
            result.tokens_per_second = round(token_count / wall_clock, 1)

    except Exception as exc:
        result.error = str(exc)

    return result


def parse_streaming_chunks(lines: list[str]) -> tuple[int, list[str]]:
    """Parse SSE lines and return (token_count, content_pieces).

    Useful for testing. Each line should be a raw SSE line like
    ``data: {"choices": [{"delta": {"content": "hi"}}]}``.
    """
    token_count = 0
    pieces: list[str] = []
    for line in lines:
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "") or delta.get("reasoning_content", "")
        if content:
            token_count += 1
            pieces.append(content)
    return token_count, pieces


# ---------------------------------------------------------------------------
# Summary + formatting
# ---------------------------------------------------------------------------

def compute_summary(results: list[PromptResult]) -> SummaryStats:
    successful = [r for r in results if r.error is None]
    total_wall = sum(r.wall_clock_s for r in results)
    avg_ttft = (
        sum(r.ttft_ms for r in successful) / len(successful) if successful else 0.0
    )
    avg_tps = (
        sum(r.tokens_per_second for r in successful) / len(successful)
        if successful
        else 0.0
    )
    return SummaryStats(
        avg_ttft_ms=round(avg_ttft, 1),
        avg_tokens_per_second=round(avg_tps, 1),
        total_prompts=len(results),
        successful_prompts=len(successful),
        total_wall_clock_s=round(total_wall, 3),
    )


def format_summary_table(results: list[PromptResult], summary: SummaryStats) -> str:
    """Build a plain-text summary table."""
    header = (
        f"{'Prompt':<12} {'TTFT(ms)':>10} {'Tok/s':>8} "
        f"{'Tokens':>8} {'Wall(s)':>9} {'Status':>8}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        status = "OK" if r.error is None else "FAIL"
        lines.append(
            f"{r.prompt_name:<12} {r.ttft_ms:>10.1f} {r.tokens_per_second:>8.1f} "
            f"{r.total_tokens:>8} {r.wall_clock_s:>9.3f} {status:>8}"
        )
    lines.append(sep)
    lines.append(
        f"{'AVERAGE':<12} {summary.avg_ttft_ms:>10.1f} "
        f"{summary.avg_tokens_per_second:>8.1f} "
        f"{'':>8} {summary.total_wall_clock_s:>9.3f} "
        f"{summary.successful_prompts}/{summary.total_prompts:>7}"
    )
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def model_slug(model_id: str) -> str:
    """Turn a model ID into a filesystem-safe slug."""
    return model_id.replace("/", "_").replace(":", "_").replace(" ", "_")[:60] or "unknown"


def write_results(run: BenchmarkRun, output_dir: str) -> tuple[str, str]:
    """Write JSON and log files. Returns (json_path, log_path)."""
    os.makedirs(output_dir, exist_ok=True)
    ts = run.timestamp.replace(":", "-")
    slug = model_slug(run.model_info.model_id)
    base = f"{ts}_{slug}"

    json_path = os.path.join(output_dir, f"{base}.json")
    log_path = os.path.join(output_dir, f"{base}.log")

    with open(json_path, "w") as f:
        json.dump(run.to_dict(), f, indent=2)

    table = format_summary_table(run.prompt_results, run.summary)
    hw = run.hardware_info
    mi = run.model_info
    log_lines = [
        f"Benchmark: {run.timestamp}",
        f"Model: {mi.model_id} (quant={mi.quant_type}, backend={mi.backend})",
        f"GPU: {hw.gpu_name} ({hw.gpu_vram_total_mb}MB total)",
        f"VRAM before: {hw.gpu_vram_used_before_mb}MB, after: {hw.gpu_vram_used_after_mb}MB",
        f"RAM: {hw.system_ram_used_mb}/{hw.system_ram_total_mb}MB",
        "",
        table,
        "",
    ]
    # Per-prompt details
    for r in run.prompt_results:
        log_lines.append(f"--- {r.prompt_name} ---")
        log_lines.append(f"Prompt: {r.prompt_text[:120]}...")
        if r.error:
            log_lines.append(f"Error: {r.error}")
        else:
            preview = r.generation_text[:200].replace("\n", " ")
            log_lines.append(f"Response preview: {preview}")
        log_lines.append("")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    return json_path, log_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    base_url: str,
    prompt_names: list[str],
    output_dir: str,
) -> BenchmarkRun:
    """Execute the full benchmark suite and return results."""
    now = datetime.now(timezone.utc)
    run = BenchmarkRun(timestamp=now.strftime("%Y-%m-%d_%H-%M-%S"))

    print(f"[benchmark] Collecting hardware info...")
    run.hardware_info = collect_hardware_info()

    print(f"[benchmark] Fetching model info from {base_url}...")
    run.model_info = fetch_model_info(base_url)
    model_name = run.model_info.model_id
    print(f"[benchmark] Model: {model_name}")

    prompts_to_run = {
        k: v for k, v in STANDARD_PROMPTS.items() if k in prompt_names
    }
    if not prompts_to_run:
        prompts_to_run = STANDARD_PROMPTS

    for name, text in prompts_to_run.items():
        print(f"[benchmark] Running '{name}' prompt...")
        result = run_prompt(base_url, name, text, model=model_name)
        run.prompt_results.append(result)
        if result.error:
            print(f"[benchmark]   FAIL: {result.error}")
        else:
            print(
                f"[benchmark]   TTFT={result.ttft_ms}ms  "
                f"tok/s={result.tokens_per_second}  "
                f"tokens={result.total_tokens}  "
                f"wall={result.wall_clock_s}s"
            )

    # Snapshot VRAM after all prompts
    _, _, gpu_used_after = get_gpu_info()
    run.hardware_info.gpu_vram_used_after_mb = gpu_used_after

    run.summary = compute_summary(run.prompt_results)

    # Print summary
    table = format_summary_table(run.prompt_results, run.summary)
    print()
    print(table)

    # Write files
    json_path, log_path = write_results(run, output_dir)
    print(f"\n[benchmark] JSON: {json_path}")
    print(f"[benchmark] Log:  {log_path}")

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an inference server")
    parser.add_argument(
        "--port", type=int, default=8080, help="Inference server port (default: 8080)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Full base URL (overrides --port). Example: http://localhost:8080",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="short,medium,long,context",
        help="Comma-separated prompt names to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks",
        help="Output directory for results (default: benchmarks/)",
    )
    args = parser.parse_args()

    base_url = args.base_url or f"http://localhost:{args.port}"
    prompt_names = [p.strip() for p in args.prompts.split(",")]

    run_benchmark(base_url, prompt_names, args.output_dir)


if __name__ == "__main__":
    main()
