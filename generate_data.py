import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Optional
from config import (
    N_ROWS, ANOMALY_RATIO, MISSING_RATIO, START_TIME, INTERVAL_MINUTES,
    RANDOM_SEED, RAW_DATA_PATH, MIN_REQUIRED_SAMPLES, GEN_RANGES
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
    Anomalies include spike, drift, and stuck patterns.
    """
    if n_rows < MIN_REQUIRED_SAMPLES:
        raise ValueError(f"n_rows must be at least {MIN_REQUIRED_SAMPLES}")

    rng = np.random.default_rng(seed)
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

    rows = []
    i = 0

    def get_val(sensor: str, range_type: str):
        low, high = GEN_RANGES[sensor][range_type]
        return rng.uniform(low, high)

    # To control the exact label ratio, we track how many abnormal rows we've created
    target_abnormal_count = int(n_rows * anomaly_ratio)
    current_abnormal_count = 0

    while i < n_rows:
        ts = start_dt + timedelta(minutes=interval_minutes * i)

        # Trigger anomaly if we haven't reached target and random chance hits
        # We adjust probability to account for event lengths
        if current_abnormal_count < target_abnormal_count and rng.random() < 0.05:
            anomaly_type = rng.choice(["spike", "drift", "stuck"])

            if anomaly_type == "spike":
                sensors = ["temp", "pressure", "vibration"]
                abn_sensors = [s for s in sensors if rng.random() < 0.5]
                if not abn_sensors:
                    abn_sensors = [rng.choice(sensors)]

                row_vals = {}
                for s in sensors:
                    if s in abn_sensors:
                        options = [k for k in GEN_RANGES[s].keys() if "abnormal" in k]
                        rtype = rng.choice(options)
                        row_vals[s] = get_val(s, rtype)
                    else:
                        row_vals[s] = get_val(s, "normal")

                rows.append([ts, row_vals["temp"], row_vals["pressure"], row_vals["vibration"], "abnormal"])
                i += 1
                current_abnormal_count += 1

            elif anomaly_type == "drift":
                length = rng.integers(3, 7)
                drift_sensor = rng.choice(["temp", "pressure", "vibration"])
                options = [k for k in GEN_RANGES[drift_sensor].keys() if "abnormal" in k]
                drift_dir = rng.choice(options)

                base_vals = {s: get_val(s, "normal") for s in ["temp", "pressure", "vibration"]}
                target_low, target_high = GEN_RANGES[drift_sensor][drift_dir]
                target_val = rng.uniform(target_low, target_high)

                # We want the drift to end in the abnormal range and we only
                # label the parts that are actually abnormal.
                # To make a 'drift' event, we start from normal and move to abnormal.
                norm_low, norm_high = GEN_RANGES[drift_sensor]["normal"]
                start_val = rng.uniform(norm_low, norm_high)

                for step in range(length):
                    if i >= n_rows: break
                    progress = (step + 1) / length
                    val = start_val + (target_val - start_val) * progress

                    current_vals = base_vals.copy()
                    current_vals[drift_sensor] = val

                    ts_step = start_dt + timedelta(minutes=interval_minutes * i)

                    # CRITICAL: Only label as abnormal if it actually exceeds the boundary
                    is_abnormal = (val < norm_low or val > norm_high)
                    label = "abnormal" if is_abnormal else "normal"

                    rows.append([ts_step, current_vals["temp"], current_vals["pressure"], current_vals["vibration"], label])
                    if is_abnormal:
                        current_abnormal_count += 1
                    i += 1

            elif anomaly_type == "stuck":
                length = rng.integers(3, 6)
                stuck_sensor = rng.choice(["temp", "pressure", "vibration"])
                options = [k for k in GEN_RANGES[stuck_sensor].keys() if "abnormal" in k]
                stuck_type = rng.choice(options)

                base_val = get_val(stuck_sensor, stuck_type)
                base_others = {s: get_val(s, "normal") for s in ["temp", "pressure", "vibration"] if s != stuck_sensor}

                for step in range(length):
                    if i >= n_rows: break
                    val = base_val + rng.normal(0, 0.001)
                    current_vals = base_others.copy()
                    current_vals[stuck_sensor] = val
                    ts_step = start_dt + timedelta(minutes=interval_minutes * i)

                    # Verify it's still abnormal after noise
                    norm_low, norm_high = GEN_RANGES[stuck_sensor]["normal"]
                    label = "abnormal" if (val < norm_low or val > norm_high) else "normal"

                    rows.append([ts_step, current_vals["temp"], current_vals["pressure"], current_vals["vibration"], label])
                    if label == "abnormal":
                        current_abnormal_count += 1
                    i += 1
        else:
            temp = get_val("temp", "normal")
            pressure = get_val("pressure", "normal")
            vibration = get_val("vibration", "normal")
            rows.append([ts, temp, pressure, vibration, "normal"])
            i += 1

    df = pd.DataFrame(rows, columns=["timestamp", "temp", "pressure", "vibration", "label"])
    df = df.iloc[:n_rows].reset_index(drop=True)
    df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

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
