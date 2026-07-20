import pytest
from unittest.mock import patch, MagicMock
import main
import pandas as pd

def test_main_evaluate_mode():
    """
    Verify that --evaluate calls evaluate() and does not call generate_suggestion or replay.
    """
    with patch("main.run_pipeline") as mock_pipe, \
         patch("main.evaluate") as mock_eval, \
         patch("main.generate_suggestion") as mock_sugg, \
         patch("main.replay") as mock_replay:

        # Setup mocks
        mock_pipe.return_value = {
            "alerts_df": pd.DataFrame({"is_anomaly_final": [True]}),
            "test_df": pd.DataFrame({"label": ["normal"], "rule_flag": [False], "ml_flag": [False], "is_anomaly_final": [False]})
        }
        mock_eval.return_value = {"rule": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "cm": []}}

        # Simulate python main.py --evaluate
        with patch("sys.argv", ["main.py", "--evaluate"]):
            main.main()

        # Assertions
        mock_pipe.assert_called_once()
        mock_eval.assert_called_once()
        mock_sugg.assert_not_called()
        mock_replay.assert_not_called()

def test_main_default_mode():
    """
    Verify that default mode calls run_pipeline, generate_suggestion, and replay.
    """
    with patch("main.run_pipeline") as mock_pipe, \
         patch("main.generate_suggestion") as mock_sugg, \
         patch("main.replay") as mock_replay:

        # Setup mocks
        # alerts_df needs to be non-empty to trigger suggestions and replay
        alerts_df = pd.DataFrame({
            "timestamp": ["2024-06-03 19:00:00"],
            "is_anomaly_final": [True]
        })
        mock_pipe.return_value = {
            "alerts_df": alerts_df,
            "test_df": pd.DataFrame()
        }
        mock_sugg.return_value = ("Check it", False)

        with patch("sys.argv", ["main.py"]):
            main.main()

        # Assertions
        mock_pipe.assert_called_once()
        assert mock_sugg.call_count == 1
        mock_replay.assert_called_once()

def test_main_no_llm_mode():
    """
    Verify that --no-llm passes use_llm=False to generate_suggestion.
    """
    with patch("main.run_pipeline") as mock_pipe, \
         patch("main.generate_suggestion") as mock_sugg, \
         patch("main.replay") as mock_replay:

        alerts_df = pd.DataFrame({
            "timestamp": ["2024-06-03 19:00:00"],
            "is_anomaly_final": [True]
        })
        mock_pipe.return_value = {
            "alerts_df": alerts_df,
            "test_df": pd.DataFrame()
        }
        mock_sugg.return_value = ("Template suggestion", True)

        with patch("sys.argv", ["main.py", "--no-llm"]):
            main.main()

        # Verify use_llm=False was passed to generate_suggestion
        # The call is: generate_suggestion(record, use_llm=use_llm)
        args, kwargs = mock_sugg.call_args
        assert kwargs["use_llm"] is False

def test_main_no_anomalies():
    """
    Verify that when no anomalies are detected, no suggestions or replay occur.
    """
    with patch("main.run_pipeline") as mock_pipe, \
         patch("main.generate_suggestion") as mock_sugg, \
         patch("main.replay") as mock_replay:

        mock_pipe.return_value = {
            "alerts_df": pd.DataFrame(), # Empty
            "test_df": pd.DataFrame()
        }

        with patch("sys.argv", ["main.py"]):
            main.main()

        mock_sugg.assert_not_called()
        mock_replay.assert_not_called()

def test_main_replay_speed():
    """
    Verify that --replay-speed is passed correctly to replay().
    """
    with patch("main.run_pipeline") as mock_pipe, \
         patch("main.generate_suggestion") as mock_sugg, \
         patch("main.replay") as mock_replay:

        alerts_df = pd.DataFrame({
            "timestamp": ["2024-06-03 19:00:00"],
            "is_anomaly_final": [True]
        })
        mock_pipe.return_value = {
            "alerts_df": alerts_df,
            "test_df": pd.DataFrame()
        }
        mock_sugg.return_value = ("Check it", False)

        with patch("sys.argv", ["main.py", "--replay-speed", "0.5"]):
            main.main()

        # Check if replay was called with replay_speed=0.5
        args, kwargs = mock_replay.call_args
        assert kwargs["replay_speed"] == 0.5
