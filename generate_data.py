import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Optional
from config import (
    N_ROWS, ANOMALY_RATIO, MISSING_RATIO, START_TIME, INTERVAL_MINUTES,
    RANDOM_SEED, RAW_DATA_PATH, MIN_REQUIRED_SAMPLES, GEN_RANGES,
    TRAIN_RATIO, CAL_RATIO, NORMAL_DIST_PARAMS
)

def generate_dataset(
    n_rows: int = N_ROWS,
    anomaly_ratio: float = ANOMALY_RATIO,
    missing_ratio: float = MISSING_RATIO,
    start_time: str = START_TIME,
    interval_minutes: int = INTERVAL_MINUTES,
    seed: int = RANDOM_SEED,
    output_path: str = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Generates a synthetic factory sensor dataset with anomalies.
    Anomalies are distributed across three layers: Train, Cal, and Test.
    """
    if n_rows < MIN_REQUIRED_SAMPLES:
        raise ValueError(f"n_rows must be at least {MIN_REQUIRED_SAMPLES}")

    rng = np.random.default_rng(seed)
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

    # Precision map for rounding
    PRECISION = {"temp": 1, "pressure": 2, "vibration": 2}

    def round_val(sensor: str, val: float) -> float:
        return round(val, PRECISION[sensor])

    def is_abnormal(sensor: str, val: float) -> bool:
        # Rounding must be applied before label check
        v = round_val(sensor, val)
        norm_min, norm_max = GEN_RANGES[sensor]["normal"]
        return v < norm_min or v > norm_max

    def sample_abnormal(sensor: str, direction: str = None) -> float:
        norm_min, norm_max = GEN_RANGES[sensor]["normal"]

        if direction is None:
            if sensor == "vibration":
                direction = "high"
            else:
                if "min_limit" in GEN_RANGES[sensor]:
                    direction = rng.choice(["high", "low"])
                else:
                    direction = "high"

        if direction == "high":
            return rng.uniform(norm_max, GEN_RANGES[sensor]["max_limit"])
        else:
            return rng.uniform(GEN_RANGES[sensor]["min_limit"], norm_min)

    # Initial generation (Normal state) - Now using Normal Distribution
    data = []
    for i in range(n_rows):
        ts = start_dt + timedelta(minutes=interval_minutes * i)
        row = {"timestamp": ts}
        for sensor in ["temp", "pressure", "vibration"]:
            mean, std = NORMAL_DIST_PARAMS[sensor]
            low, high = GEN_RANGES[sensor]["normal"]
            # Gaussian noise clipped to normal range
            val = rng.normal(mean, std)
            row[sensor] = round_val(sensor, np.clip(val, low, high))
        row["label"] = "normal"
        data.append(row)

    # --- Layered Anomaly Configuration ---

    train_split = int(n_rows * TRAIN_RATIO)
    cal_split = int(n_rows * (TRAIN_RATIO + CAL_RATIO))

    blocks = [
        {"name": "train", "start": 0, "end": train_split, "ratio": TRAIN_RATIO},
        {"name": "cal", "start": train_split, "end": cal_split, "ratio": CAL_RATIO},
        {"name": "test", "start": cal_split, "end": n_rows, "ratio": 1.0 - (TRAIN_RATIO + CAL_RATIO)},
    ]

    est_avg_rows_per_event = 3.0
    total_event_budget = max(3, int((n_rows * anomaly_ratio) / est_avg_rows_per_event))

    for block in blocks:
        num_events = max(1, int(total_event_budget * block["ratio"]))

        for _ in range(num_events):
            start_idx = rng.integers(block["start"], block["end"])
            anomaly_type = rng.choice(["spike", "drift", "stuck"])

            if anomaly_type == "spike":
                # Spike: All 3 sensors abnormal (joint failure)
                for s in ["temp", "pressure", "vibration"]:
                    direction = rng.choice(["high", "low"]) if "min_limit" in GEN_RANGES[s] else "high"
                    val = sample_abnormal(s, direction)
                    data[start_idx][s] = round_val(s, val)
                    if is_abnormal(s, data[start_idx][s]):
                        data[start_idx]["label"] = "abnormal"

            elif anomaly_type == "drift":
                # Drift: All 3 sensors abnormal (joint failure)
                length = rng.integers(3, 7)
                drift_dirs = {s: rng.choice(["high", "low"]) if "min_limit" in GEN_RANGES[s] else "high"
                              for s in ["temp", "pressure", "vibration"]}
                drift_targets = {s: sample_abnormal(s, drift_dirs[s]) for s in ["temp", "pressure", "vibration"]}
                start_vals = {s: data[start_idx][s] for s in ["temp", "pressure", "vibration"]}

                for step in range(length):
                    idx = start_idx + step
                    if idx >= n_rows: break
                    progress = (step + 1) / length
                    for s in ["temp", "pressure", "vibration"]:
                        val = start_vals[s] + (drift_targets[s] - start_vals[s]) * progress
                        data[idx][s] = round_val(s, val)
                    # Drift: entire block is abnormal
                    data[idx]["label"] = "abnormal"

            elif anomaly_type == "stuck":
                # Stuck: Only single sensor affected
                stuck_sensor = rng.choice(["temp", "pressure", "vibration"])
                base_val = sample_abnormal(stuck_sensor)
                length = rng.integers(3, 6)

                for step in range(length):
                    idx = start_idx + step
                    if idx >= n_rows: break
                    val = base_val + rng.normal(0, 0.001)
                    data[idx][stuck_sensor] = round_val(stuck_sensor, val)
                    if is_abnormal(stuck_sensor, data[idx][stuck_sensor]):
                        data[idx]["label"] = "abnormal"

    df = pd.DataFrame(data)
    cols = ["timestamp", "temp", "pressure", "vibration", "label"]
    df = df[cols]
    df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

    # Missing values
    sensor_cols = ["temp", "pressure", "vibration"]
    n_missing = int(len(df) * missing_ratio)
    missing_idx = rng.choice(len(df), size=n_missing, replace=False)
    missing_cols = rng.choice(sensor_cols, size=n_missing)
    for idx, col in zip(missing_idx, missing_cols):
        df.loc[idx, col] = np.nan

    df = df.sort_values("timestamp").reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(f"Dataset generated and saved to {RAW_DATA_PATH}")
    print(df.head())
