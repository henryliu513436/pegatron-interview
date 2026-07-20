import pytest
from unittest.mock import patch, MagicMock
import requests
from llm_advisor import generate_suggestion, template_suggestion

def test_template_suggestion_content():
    """
    Verify template suggestions are generated based on severity and sensors.
    """
    # Case 1: CRITICAL
    rec_critical = {
        "severity": "CRITICAL",
        "triggered_sensors": "temp;vibration",
        "temp": 55.0, "pressure": 1.02, "vibration": 0.08
    }
    sugg_critical = template_suggestion(rec_critical)
    assert "嚴重警告" in sugg_critical
    assert "temp" in sugg_critical
    assert "vibration" in sugg_critical

    # Case 2: MEDIUM
    rec_medium = {
        "severity": "MEDIUM",
        "triggered_sensors": "pressure",
        "temp": 47.0, "pressure": 1.10, "vibration": 0.03
    }
    sugg_medium = template_suggestion(rec_medium)
    assert "中度警告" in sugg_medium
    assert "pressure" in sugg_medium

def test_generate_suggestion_no_llm():
    """
    Verify that use_llm=False immediately uses the template.
    """
    record = {
        "severity": "HIGH",
        "triggered_sensors": "temp",
        "temp": 53.0, "pressure": 1.02, "vibration": 0.03
    }
    suggestion, is_fallback = generate_suggestion(record, use_llm=False)

    assert is_fallback is True
    assert suggestion == template_suggestion(record)

@patch("llm_advisor.requests.post")
def test_generate_suggestion_llm_success(mock_post):
    """
    Verify successful LLM response.
    """
    # Mocking Ollama response
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Ollama usually returns a stream of JSON objects or a single JSON object if stream=False
    # We simulate the response body based on how llm_advisor is implemented
    mock_response.json.return_value = {"response": "LLM suggested: check the cooling system."}
    mock_post.return_value = mock_response

    record = {
        "severity": "CRITICAL",
        "triggered_sensors": "temp",
        "temp": 55.0, "pressure": 1.02, "vibration": 0.08
    }

    suggestion, is_fallback = generate_suggestion(record, use_llm=True)

    assert is_fallback is False
    assert "check the cooling system" in suggestion
    mock_post.assert_called_once()

@patch("llm_advisor.requests.post")
def test_generate_suggestion_llm_failure_fallback(mock_post):
    """
    Verify that LLM failure (exception or bad status) falls back to template.
    """
    # Simulate a timeout or connection error
    mock_post.side_effect = requests.exceptions.Timeout()

    record = {
        "severity": "HIGH",
        "triggered_sensors": "temp",
        "temp": 53.0, "pressure": 1.02, "vibration": 0.03
    }

    suggestion, is_fallback = generate_suggestion(record, use_llm=True)

    assert is_fallback is True
    assert suggestion == template_suggestion(record)

@patch("llm_advisor.requests.post")
def test_generate_suggestion_llm_empty_response(mock_post):
    """
    Verify that an empty or invalid LLM response falls back to template.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": ""} # Empty string
    mock_post.return_value = mock_response

    record = {
        "severity": "MEDIUM",
        "triggered_sensors": "pressure",
        "temp": 47.0, "pressure": 1.10, "vibration": 0.03
    }

    suggestion, is_fallback = generate_suggestion(record, use_llm=True)

    assert is_fallback is True
    assert suggestion == template_suggestion(record)
