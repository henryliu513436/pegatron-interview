import time
import os
import pandas as pd
from config import ALERTS_LOG_PATH, DEFAULT_REPLAY_SPEED

def render_alert(alert_row: dict, suggestion: str, is_fallback: bool) -> str:
    """
    Formats a single alert into a readable string for CLI output and logging.

    Mandatory elements per spec:
    1. Anomaly markers (timestamp, severity)
    2. ML Score (ml_score)
    3. Actionable suggestion (suggestion text)
    """
    ts = alert_row.get("timestamp", "Unknown Time")
    severity = alert_row.get("severity", "UNKNOWN")
    sensors = alert_row.get("triggered_sensors", "unknown")
    score = alert_row.get("ml_score", 0.0)

    # Formatting with visual markers for severity
    marker = f"[{severity}]" if severity else "[NORMAL]"

    formatted = (
        f"[{ts}] {marker} | "
        f"score={score:.4f} | "
        f"sensors={sensors} | "
        f"suggestion={suggestion} | "
        f"fallback={is_fallback}"
    )

    # Print to terminal (CLI rendering)
    print(formatted)
    return formatted

def append_to_log(formatted_alert: str, log_path: str = ALERTS_LOG_PATH) -> None:
    """
    Appends a formatted alert string to the log file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(formatted_alert + "\n")

def replay(alerts_df: pd.DataFrame, suggestions: list, replay_speed: float = DEFAULT_REPLAY_SPEED, log_path: str = ALERTS_LOG_PATH) -> None:
    """
    Replays alerts sequentially, calling render and log.
    """
    # suggestions is a list of (text, is_fallback) tuples
    for idx, row in alerts_df.iterrows():
        sugg_text, is_fallback = suggestions[idx]

        # Convert row to dict for render_alert
        alert_dict = row.to_dict()

        formatted = render_alert(alert_dict, sugg_text, is_fallback)
        append_to_log(formatted, log_path=log_path)

        if replay_speed > 0:
            time.sleep(replay_speed)
