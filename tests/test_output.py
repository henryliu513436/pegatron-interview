import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from output import render_alert, append_to_log, replay
from config import ALERTS_LOG_PATH

def test_render_alert_content():
    """
    Verify render_alert produces a string containing the three mandatory elements:
    1. Anomaly marker/severity
    2. ml_score
    3. actionable suggestion
    """
    alert_row = {
        "timestamp": "2024-06-03 19:00:00",
        "severity": "CRITICAL",
        "triggered_sensors": "temp",
        "ml_score": -0.15,
        "temp": 55.0,
        "pressure": 1.02,
        "vibration": 0.03
    }
    suggestion = "Check cooling system immediately."
    is_fallback = False

    formatted = render_alert(alert_row, suggestion, is_fallback)

    # Check for mandatory elements
    assert "CRITICAL" in formatted
    assert "-0.15" in formatted
    assert "Check cooling system immediately" in formatted
    assert "2024-06-03 19:00:00" in formatted
    assert "temp" in formatted

def test_append_to_log(tmp_path):
    """
    Verify that alerts are appended to the log file and directory is created.
    """
    log_file = str(tmp_path / "alerts.log")
    formatted_alert = "[2024-06-03 19:00:00] CRITICAL | sensors=temp | suggestion=Check it | fallback=False"

    append_to_log(formatted_alert, log_path=log_file)

    with open(log_file, "r") as f:
        content = f.read()
        assert formatted_alert in content

def test_replay_integration(tmp_path):
    """
    Verify that replay calls render_alert and append_to_log for each row.
    """
    log_file = str(tmp_path / "alerts.log")

    # Create a small alerts DataFrame
    data = {
        "timestamp": ["2024-06-03 19:00:00", "2024-06-03 19:01:00"],
        "severity": ["HIGH", "MEDIUM"],
        "triggered_sensors": ["temp", "pressure"],
        "ml_score": [-0.1, -0.05],
        "temp": [47.0, 47.0],
        "pressure": [1.02, 1.10],
        "vibration": [0.03, 0.03]
    }
    alerts_df = pd.DataFrame(data)
    suggestions = [
        ("Check temp", False),
        ("Check pressure", True)
    ]

    # We mock append_to_log to avoid global file writes,
    # but here we just use a temp path
    import output
    with patch("output.append_to_log", wraps=output.append_to_log) as mock_log:
        output.replay(alerts_df, suggestions, replay_speed=0, log_path=log_file)
        assert mock_log.call_count == 2

def test_render_alert_fallback_marker():
    """
    Verify that fallback=True is correctly marked in the output.
    """
    alert_row = {
        "timestamp": "2024-06-03 19:00:00",
        "severity": "MEDIUM",
        "triggered_sensors": "vibration",
        "ml_score": -0.02,
        "temp": 47.0, "pressure": 1.02, "vibration": 0.05
    }
    suggestion = "Monitor vibration."
    is_fallback = True

    formatted = render_alert(alert_row, suggestion, is_fallback)
    assert "fallback=True" in formatted
