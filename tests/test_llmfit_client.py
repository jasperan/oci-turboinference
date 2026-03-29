"""Tests for the llmfit REST client."""

from unittest.mock import MagicMock, patch

import httpx

from profiler.llmfit_client import LlmfitResult, query_llmfit


@patch("profiler.llmfit_client.httpx.get")
def test_query_llmfit_success(mock_get: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "meta-llama/Llama-3-8B",
        "fit": "perfect",
        "quantization": "Q4_K_M",
        "estimated_vram_gb": 5.2,
        "runtime": "llama.cpp",
    }
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = query_llmfit("meta-llama/Llama-3-8B", vram_gb=24)

    assert result is not None
    assert isinstance(result, LlmfitResult)
    assert result.model == "meta-llama/Llama-3-8B"
    assert result.fit == "perfect"
    assert result.quantization == "Q4_K_M"
    assert result.estimated_vram_gb == 5.2
    assert result.runtime == "llama.cpp"

    mock_get.assert_called_once()


@patch("profiler.llmfit_client.httpx.get")
def test_query_llmfit_unreachable(mock_get: MagicMock) -> None:
    mock_get.side_effect = Exception("Connection refused")

    result = query_llmfit("meta-llama/Llama-3-8B", vram_gb=24)

    assert result is None


@patch("profiler.llmfit_client.httpx.get")
def test_query_llmfit_http_error(mock_get: MagicMock) -> None:
    """Server returns 500. Should return None."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error", request=MagicMock(), response=mock_resp
    )
    mock_get.return_value = mock_resp

    result = query_llmfit("meta-llama/Llama-3-8B", vram_gb=24)

    assert result is None


@patch("profiler.llmfit_client.httpx.get")
def test_query_llmfit_malformed_json(mock_get: MagicMock) -> None:
    """Server returns 200 but body is not valid JSON. Should return None."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.side_effect = ValueError("No JSON object could be decoded")
    mock_get.return_value = mock_resp

    result = query_llmfit("meta-llama/Llama-3-8B", vram_gb=24)

    assert result is None


@patch("profiler.llmfit_client.httpx.get")
def test_query_llmfit_custom_base_url(mock_get: MagicMock) -> None:
    """Custom base_url should be used in the request."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "model": "meta-llama/Llama-3-8B",
        "fit": "perfect",
        "quantization": "Q4_K_M",
        "estimated_vram_gb": 5.2,
        "runtime": "llama.cpp",
    }
    mock_get.return_value = mock_resp

    result = query_llmfit(
        "meta-llama/Llama-3-8B", vram_gb=24, base_url="http://custom:9999/api/v1"
    )

    assert result is not None
    # Verify the correct URL was called
    call_args = mock_get.call_args
    assert call_args[0][0] == "http://custom:9999/api/v1/models/plan"
