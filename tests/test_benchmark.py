"""Unit tests for profiler.benchmark module."""

import json

from profiler.benchmark import (
    BenchmarkRun,
    HardwareInfo,
    ModelInfo,
    PromptResult,
    SummaryStats,
    compute_summary,
    format_summary_table,
    parse_streaming_chunks,
)


class TestParseStreamingResponse:
    """Test SSE chunk parsing logic."""

    def test_basic_chunks(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            "data: [DONE]",
        ]
        count, pieces = parse_streaming_chunks(lines)
        assert count == 3
        assert pieces == ["Hello", " world", "!"]
        assert "".join(pieces) == "Hello world!"

    def test_empty_deltas_skipped(self):
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"Yes"}}]}',
            'data: {"choices":[{"delta":{"content":""}}]}',
            "data: [DONE]",
        ]
        count, pieces = parse_streaming_chunks(lines)
        # Empty string content is falsy, so it should be skipped
        assert count == 1
        assert pieces == ["Yes"]

    def test_non_data_lines_ignored(self):
        lines = [
            ": heartbeat",
            "",
            'data: {"choices":[{"delta":{"content":"OK"}}]}',
            "data: [DONE]",
        ]
        count, pieces = parse_streaming_chunks(lines)
        assert count == 1
        assert pieces == ["OK"]

    def test_malformed_json_skipped(self):
        lines = [
            "data: {not valid json}",
            'data: {"choices":[{"delta":{"content":"fine"}}]}',
            "data: [DONE]",
        ]
        count, pieces = parse_streaming_chunks(lines)
        assert count == 1
        assert pieces == ["fine"]

    def test_done_stops_processing(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"B"}}]}',
        ]
        count, pieces = parse_streaming_chunks(lines)
        assert count == 1
        assert pieces == ["A"]


class TestBenchmarkResultSerialization:
    """Test BenchmarkRun round-trip serialization."""

    def _make_run(self) -> BenchmarkRun:
        return BenchmarkRun(
            timestamp="2026-03-29_12-00-00",
            hardware_info=HardwareInfo(
                gpu_name="NVIDIA A10",
                gpu_vram_total_mb=23028,
                gpu_vram_used_before_mb=512,
                gpu_vram_used_after_mb=8192,
                system_ram_total_mb=65536,
                system_ram_used_mb=4096,
            ),
            model_info=ModelInfo(
                model_id="Qwen/Qwen3.5-35B-A3B",
                quant_type="Q4_K_M",
                backend="llamacpp",
            ),
            prompt_results=[
                PromptResult(
                    prompt_name="short",
                    prompt_text="What is 2+2?",
                    ttft_ms=45.2,
                    tokens_per_second=82.3,
                    total_tokens=12,
                    wall_clock_s=0.146,
                    generation_text="4",
                ),
                PromptResult(
                    prompt_name="medium",
                    prompt_text="Explain CPUs",
                    ttft_ms=102.5,
                    tokens_per_second=65.1,
                    total_tokens=210,
                    wall_clock_s=3.225,
                    prompt_eval_time_ms=80.0,
                    generation_text="A CPU works by...",
                ),
            ],
            summary=SummaryStats(
                avg_ttft_ms=73.85,
                avg_tokens_per_second=73.7,
                total_prompts=2,
                successful_prompts=2,
                total_wall_clock_s=3.371,
            ),
        )

    def test_serialize_to_json(self):
        run = self._make_run()
        d = run.to_dict()
        text = json.dumps(d)
        assert '"timestamp"' in text
        assert '"hardware_info"' in text
        assert '"prompt_results"' in text

    def test_round_trip(self):
        run = self._make_run()
        d = run.to_dict()
        text = json.dumps(d)
        restored_d = json.loads(text)
        restored = BenchmarkRun.from_dict(restored_d)

        assert restored.timestamp == run.timestamp
        assert restored.hardware_info.gpu_name == "NVIDIA A10"
        assert restored.hardware_info.gpu_vram_total_mb == 23028
        assert restored.model_info.model_id == "Qwen/Qwen3.5-35B-A3B"
        assert restored.model_info.backend == "llamacpp"
        assert len(restored.prompt_results) == 2
        assert restored.prompt_results[0].prompt_name == "short"
        assert restored.prompt_results[0].ttft_ms == 45.2
        assert restored.prompt_results[1].prompt_eval_time_ms == 80.0
        assert restored.summary.avg_ttft_ms == 73.85
        assert restored.summary.successful_prompts == 2

    def test_all_fields_present(self):
        run = self._make_run()
        d = run.to_dict()
        assert "timestamp" in d
        assert "hardware_info" in d
        assert "model_info" in d
        assert "prompt_results" in d
        assert "summary" in d
        hw = d["hardware_info"]
        for key in [
            "gpu_name", "gpu_vram_total_mb", "gpu_vram_used_before_mb",
            "gpu_vram_used_after_mb", "system_ram_total_mb", "system_ram_used_mb",
        ]:
            assert key in hw

    def test_error_field_serializes(self):
        pr = PromptResult(prompt_name="fail", error="Connection refused")
        d = pr.to_dict()
        assert d["error"] == "Connection refused"
        restored = PromptResult(**d)
        assert restored.error == "Connection refused"


class TestFormatSummaryTable:
    """Test that the summary table has expected columns and structure."""

    def _make_results(self):
        results = [
            PromptResult(
                prompt_name="short",
                ttft_ms=40.0,
                tokens_per_second=80.0,
                total_tokens=10,
                wall_clock_s=0.125,
            ),
            PromptResult(
                prompt_name="medium",
                ttft_ms=100.0,
                tokens_per_second=60.0,
                total_tokens=200,
                wall_clock_s=3.333,
            ),
        ]
        summary = compute_summary(results)
        return results, summary

    def test_contains_column_headers(self):
        results, summary = self._make_results()
        table = format_summary_table(results, summary)
        assert "Prompt" in table
        assert "TTFT(ms)" in table
        assert "Tok/s" in table
        assert "Tokens" in table
        assert "Wall(s)" in table
        assert "Status" in table

    def test_contains_prompt_names(self):
        results, summary = self._make_results()
        table = format_summary_table(results, summary)
        assert "short" in table
        assert "medium" in table

    def test_contains_average_row(self):
        results, summary = self._make_results()
        table = format_summary_table(results, summary)
        assert "AVERAGE" in table

    def test_shows_ok_status(self):
        results, summary = self._make_results()
        table = format_summary_table(results, summary)
        assert "OK" in table

    def test_shows_fail_status(self):
        results = [
            PromptResult(prompt_name="broken", error="timeout"),
        ]
        summary = compute_summary(results)
        table = format_summary_table(results, summary)
        assert "FAIL" in table

    def test_summary_stats_correct(self):
        results, summary = self._make_results()
        assert summary.avg_ttft_ms == 70.0
        assert summary.avg_tokens_per_second == 70.0
        assert summary.total_prompts == 2
        assert summary.successful_prompts == 2
