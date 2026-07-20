import time
import os
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from config import ALERTS_LOG_PATH, DEFAULT_REPLAY_SPEED

console = Console()

def render_alert(alert_row: dict, suggestion: str, is_fallback: bool) -> str:
    """
    Formats a single alert for CLI output using 'rich' and returns a plain string for logging.
    """
    ts = alert_row.get("timestamp", "Unknown Time")
    severity = alert_row.get("severity", "UNKNOWN")
    sensors = alert_row.get("triggered_sensors", "unknown")
    score = alert_row.get("ml_score", 0.0)

    # 1. Build the plain string for logging
    log_marker = f"[{severity}]" if severity else "[NORMAL]"
    formatted_log = (
        f"[{ts}] {log_marker} | "
        f"score={score:.4f} | "
        f"sensors={sensors} | "
        f"suggestion={suggestion} | "
        f"fallback={is_fallback}"
    )

    # 2. Build the rich display for CLI
    # Severity colors
    severity_color = "white"
    if severity == "CRITICAL":
        severity_color = "bold red"
    elif severity == "HIGH":
        severity_color = "bold yellow"
    elif severity == "MEDIUM":
        severity_color = "cyan"

    # Construct visual content
    content = Text()
    content.append(f"{ts} ", style="dim")
    content.append(f" {severity} ", style=severity_color)
    content.append(f" | Sensors: {sensors} | Score: {score:.4f}\n", style="white")
    content.append(f"Suggestion: {suggestion}", style="italic green")
    if is_fallback:
        content.append(" (Fallback Template)", style="dim yellow")

    # Render as a panel
    panel = Panel(
        content,
        title="⚠️ Factory Alert",
        border_style=severity_color,
        expand=False
    )

    console.print(panel)
    return formatted_log

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
    for idx, row in alerts_df.iterrows():
        # suggestions is a list of (text, is_fallback) tuples
        # Ensure index match (alerts_df has reset index in agent.py)
        if idx >= len(suggestions):
            break

        sugg_text, is_fallback = suggestions[idx]

        alert_dict = row.to_dict()

        formatted = render_alert(alert_dict, sugg_text, is_fallback)
        append_to_log(formatted, log_path=log_path)

        if replay_speed > 0:
            time.sleep(replay_speed)
