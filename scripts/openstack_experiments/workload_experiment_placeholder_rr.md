## PART1: workload experiment: live metrics collection (rr=0)

# STEP1: live metrics collection keeping request_rate 0 as palceholder
# In openstack, ssh to vm (test_model), go to fcdm folder and activate venv
# 1 — Install psutil (inside venv) (for live metrics collection)
# pip install psutil
# Verify installation: python -c "import psutil; print('psutil OK')". You should see "psutil OK" if it’s installed correctly.

# 2 - Install the CPU stress tool (workload preparation)
# sudo apt-get update
# sudo apt-get install -y stress
# check: stress --version

# 3 - Create the live metrics logger (writes CSV every minute) (runs for 75 minutes)
# Create the file: nano log_openstack_metrics.py
# code below: (it imports csv)

"""
import time
from datetime import datetime
import psutil

OUT_PATH = "data/openstack_metrics.csv"
TOTAL_MINUTES = 75
SLEEP_SECONDS = 60

def main():
    # Warm-up call for more stable first CPU reading
    psutil.cpu_percent(interval=None)

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "cpu_mean", "memory_mean", "request_rate_mean"])

        for minute in range(TOTAL_MINUTES):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cpu_frac = psutil.cpu_percent(interval=1) / 100.0   # 0..1
            mem_frac = psutil.virtual_memory().percent / 100.0  # 0..1

            # Placeholder (no web app in this experiment)
            rr = 0.0

            w.writerow([ts, f"{cpu_frac:.6f}", f"{mem_frac:.6f}", f"{rr:.1f}"])
            f.flush()

            print(f"[{minute+1:02d}/{TOTAL_MINUTES}] ts={ts} cpu={cpu_frac:.3f} mem={mem_frac:.3f} rr={rr}")

            # sleep remaining time of the minute (we already used ~1s for cpu_percent interval)
            if minute < TOTAL_MINUTES - 1:
                time.sleep(max(0, SLEEP_SECONDS - 1))

    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()

"""

# 4 - Run the logger: python log_openstack_metrics.py
# You should see output every minute and at the end a CSV file with 75 rows (1 header + 75 data rows).
# 5 - In a SECOND SSH session, start stress at minute 10 command: stress --cpu $(nproc) --timeout 45m
# Verify the CSV exists and has 75 rows (+ header) : wc -l data/openstack_metrics.csv

# 6 - Run your model on this NEW OpenStack metrics CSV : python infer_openstack_next10min.py
# copy: cp infer_next10min.py infer_openstack_next10min.py
# Edit the copied script:
# RAW_CSV_PATH = "data/openstack_metrics.csv"     #this is the new openstack csv with live metrics
# OUT_PATH = "output/openstack_live/predictions_next10min.csv" # save predictions in a separate folder to avoid confusion with previous experiments
# Run: python infer_openstack_next10min.py

# code below: 
"""
# infer_next10min.py is copied and modified to infer_openstack_next10min.py with the above changes in paths and names. The rest of the code is the same as infer_next10min.py
# Goal: Load trained HF TimeSeriesTransformer + saved MinMax scaler,
#       read a small input CSV, forecast next 10 steps, save:
#       output/predictions_next10min.csv

import pickle
import numpy as np
import pandas as pd
import torch
from transformers import TimeSeriesTransformerForPrediction

RAW_CSV_PATH = "data/openstack_metrics.csv"
MODEL_DIR = "model"
SCALER_PATH = "minmax_scaler.pkl"
OUT_PATH = "output/openstack_live/predictions_next10min.csv"

# Columns your model was trained on (keep same order!)
COLS = ["cpu_mean", "memory_mean", "request_rate_mean"]

# Must match training context/lag settings
CONTEXT_LEN = 60
LAGS_SEQUENCE = [1]
HISTORY_LEN = CONTEXT_LEN + max(LAGS_SEQUENCE)  # 61
PRED_LEN = 10  # next 10 minutes/steps

def main():
    # ---- Load input ----
    df = pd.read_csv(RAW_CSV_PATH)

    # Keep only needed columns and drop NaNs
    df = df[COLS].dropna().reset_index(drop=True)

    if len(df) < HISTORY_LEN:
        raise ValueError(f"Need at least {HISTORY_LEN} rows, but got {len(df)}")

    # ---- Load scaler (the one from training) ----
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Normalize
    x = df.values.astype(np.float32)
    x_norm = scaler.transform(x)

    # Take last HISTORY_LEN as past context
    past = x_norm[-HISTORY_LEN:]                 # (61, 3)
    past = torch.tensor(past).unsqueeze(0)       # (1, 61, 3)

    # ---- Load trained model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    past = past.to(device)

    # ---- Forecast ----
    # Some transformers versions infer horizon from future_time_features shape
    past_observed_mask = torch.ones_like(past, device=device)

    num_time_features = getattr(model.config, "num_time_features", 0) or 0
    if num_time_features > 0:
        past_time_features = torch.zeros((1, HISTORY_LEN, num_time_features), device=device)
        future_time_features = torch.zeros((1, PRED_LEN, num_time_features), device=device)
    else:
        past_time_features = None
        future_time_features = None

    with torch.no_grad():
        kwargs = {
            "past_values": past,
            "past_observed_mask": past_observed_mask,
        }
        if past_time_features is not None:
            kwargs["past_time_features"] = past_time_features
        if future_time_features is not None:
            kwargs["future_time_features"] = future_time_features

        out = model.generate(**kwargs)

    # Depending on transformers version, output could be:
    # - out.sequences (common)
    # - out (tensor)
    if hasattr(out, "sequences"):
        pred_norm = out.sequences[0].detach().cpu().numpy()
    else:
        pred_norm = out[0].detach().cpu().numpy()
    
            # Ensure we have exactly (PRED_LEN, num_targets)
    pred_norm = np.asarray(pred_norm)
    if pred_norm.ndim == 3:
        pred_norm = pred_norm[0]
    if pred_norm.shape[0] != PRED_LEN:
        # If model returns longer, keep first PRED_LEN
        pred_norm = pred_norm[:PRED_LEN]

    # Inverse transform back to real units
    pred_real = scaler.inverse_transform(pred_norm)

    # ---- Save output ----
    pred_df = pd.DataFrame(
        pred_real,
        columns=["cpu_pred", "memory_pred", "request_rate_pred"],
    )
    pred_df.insert(0, "t_plus_step", np.arange(1, PRED_LEN + 1))

    pred_df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

"""

# checkpoint: 
# wc -l output/openstack_live/predictions_next10min.csv
# head output/openstack_live/predictions_next10min.csv

## OR automate the process:: 
# instead of using two ssh sessions, you can run the logger in background and then start stress in one script together.
# run_experiment.sh
# create the file: nano run_experiment.sh
# code below:
"""
#!/bin/bash
cd ~/fcdm_inference

python log_openstack_metrics.py &
sleep 600
stress --cpu $(nproc) --timeout 45m
wait
wc -l data/openstack_metrics.csv
head data/openstack_metrics.csv

"""
# to execute: chmod +x ~/fcdm_inference/run_experiment.sh
# RUN : ./run_experiment.sh

# then Run your model on this NEW OpenStack metrics CSV : python infer_openstack_next10min.py
# checkpoint: 
# wc -l output/openstack_live/predictions_next10min.csv
# head output/openstack_live/predictions_next10min.csv
