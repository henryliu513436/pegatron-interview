import requests
import json
from typing import Tuple
from config import OLLAMA_MODEL, OLLAMA_HOST, LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES

def template_suggestion(record: dict) -> str:
    """
    Generates a fixed template suggestion based on severity and triggered sensors.
    """
    severity = record.get("severity", "UNKNOWN")
    sensors = record.get("triggered_sensors", "unknown sensors")

    # Simple template matrix (Traditional Chinese)
    templates = {
        "CRITICAL": f"嚴重警告：多個感測器 ({sensors}) 數值極端。請立即緊急停機並進行人工檢查。",
        "HIGH": f"高度警告：感測器 {sensors} 超出閾值。請立即檢查設備狀態。",
        "MEDIUM": f"中度警告：ML 偵測到 {sensors} 異常模式。請密切監控是否存在漸進劣化 (drift)。",
    }

    return templates.get(severity, f"警告：偵測到 {sensors} 異常活動。請進行調查。")

def generate_suggestion(record: dict, use_llm: bool = True) -> Tuple[str, bool]:
    """
    Generates a suggestion for the alert.
    - If use_llm=True, attempts to call Ollama.
    - If use_llm=False, or LLM call fails, falls back to template_suggestion.
    - Returns (suggestion_text, is_fallback).
    """
    if not use_llm:
        return template_suggestion(record), True

    # LLM Implementation
    prompt = (
        f"The factory sensor system detected an anomaly.\n"
        f"Severity: {record.get('severity')}\n"
        f"Triggered Sensors: {record.get('triggered_sensors')}\n"
        f"Current Values: temp={record.get('temp')}, pressure={record.get('pressure')}, vibration={record.get('vibration')}\n"
        f"Please provide a brief, actionable maintenance suggestion for the technician. "
        f"Respond in Traditional Chinese and keep it concise."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False
    }

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=LLM_TIMEOUT_SECONDS
            )
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "").strip()
                if text:
                    return text, False
        except Exception:
            # Log error internally if needed, but do not raise
            pass

    # Fallback if all retries fail or response is empty
    return template_suggestion(record), True
