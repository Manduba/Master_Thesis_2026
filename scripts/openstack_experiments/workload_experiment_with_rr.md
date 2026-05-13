# ##PART2: workload experiment: live metrics collection with request_rate (rr) as feature
# In this experiment, we will collect live metrics including request_rate (rr) and use them for inference.
# The steps are similar to the previous experiment, but we will have a web server generating requests

# STEP1: Install and start Nginx on the VM
# sudo apt-get update
# sudo apt-get install -y nginx 
# start nginx: sudo systemctl start nginx   
# check acitve running: sudo systemctl status nginx
# check nginx is serving: curl http://localhost

# 2 - Create NEW logger script for HTTP run: nano log_openstack_http_metrics.py
# run the logger: python log_openstack_http_metrics.py
#code below:
"""    
import csv
import time
from datetime import datetime
import psutil
import subprocess
from pathlib import Path

OUT_PATH = "data/openstack_metrics_http_75min.csv"
TOTAL_MINUTES = 75
SLEEP_SECONDS = 60
ACCESS_LOG = "/var/log/nginx/access.log"

def count_log_lines():
    out = subprocess.check_output(["wc", "-l", ACCESS_LOG], text=True).strip()
    return int(out.split()[0])


def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    
    psutil.cpu_percent(interval=None)
    prev_lines = count_log_lines()

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "cpu_mean", "memory_mean", "request_rate_mean"])

        for minute in range(TOTAL_MINUTES):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cpu_frac = psutil.cpu_percent(interval=1) / 100.0
            mem_frac = psutil.virtual_memory().percent / 100.0

            curr_lines = count_log_lines()
            req_last_min = max(0, curr_lines - prev_lines)
            prev_lines = curr_lines

            # request_rate_mean = requests per second (mean over the minute)
            rr = req_last_min / 60.0

            w.writerow([ts, f"{cpu_frac:.6f}", f"{mem_frac:.6f}", f"{rr:.6f}"])
            f.flush()

            print(f"[{minute+1:02d}/{TOTAL_MINUTES}] ts={ts} cpu={cpu_frac:.3f} mem={mem_frac:.3f} req/min={req_last_min} rr={rr:.2f}")

            if minute < TOTAL_MINUTES - 1:
                time.sleep(max(0, SLEEP_SECONDS - 1))

    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
"""
#check: wc -l data/openstack_metrics_http_75min.csv
#check: head data/openstack_metrics_http_75min.csv

# 3 - Generate HTTP load (Terminal 2, same VM1)
# use a tool that can run continuously. Apache Benchmarking Tool (ab) did not work and it stopped after a while.
# so we use siege for 45 minutes
# Install siege: 
# sudo apt update
# sudo apt install -y siege
# siege -c 50 -t 45M http://127.0.0.1/
# after it finishes, check : 
# wc -l data/openstack_metrics_http_75min.csv
# head data/openstack_metrics_http_75min.csv

# 4 - Run your model on this NEW OpenStack metrics CSV : python infer_openstack_http_next10min.py
# copy: cp infer_next10min.py infer_openstack_http_next10min.py 
# Edit the copied script:nano infer_openstack_http_next10min.py
# RAW_CSV_PATH = "data/openstack_metrics_http_75min.csv"     #this is the new openstack csv with live metrics with rr
# OUT_PATH = "output/openstack_live/predictions_http_next10min.csv" # save predictions in a separate folder to avoid confusion with previous experiments
# Run: python infer_openstack_http_next10min.py
#code below:
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

RAW_CSV_PATH = "data/openstack_metrics_http_75min.csv"
MODEL_DIR = "model"
SCALER_PATH = "minmax_scaler.pkl"
OUT_PATH = "output/openstack_live/predictions_http_next10min.csv"

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
# wc -l data/openstack_live/predictions_http_next10min.csv
# head data/openstack_live/predictions_http_next10min.csv

## OR automate the process:: 
# instead of using two ssh sessions, you can run the logger in background and then start stress in one script together.
# run_experiment_http.sh
# 1 - create the file: nano run_experiment_http.sh
#code below:

"""
#!/bin/bash

cd ~/fcdm_inference || exit 1

mkdir -p data output/openstack_live

echo "[0/6] Clearing old files and nginx access log..."
rm -f data/openstack_metrics_http_75min.csv
rm -f output/openstack_live/http_logger.log
sudo truncate -s 0 /var/log/nginx/access.log

echo "[1/6] Starting logger at $(date)"
nohup python log_openstack_http_metrics.py > output/openstack_live/http_logger.log 2>&1 &
LOGGER_PID=$!
echo "Logger PID: $LOGGER_PID"

echo "[2/6] Baseline period starts at $(date)"
sleep 600

echo "[3/6] Starting siege at $(date)"
siege -c 50 -t 45M http://127.0.0.1/ || true

echo "[4/6] Waiting for logger to finish..."
wait "$LOGGER_PID"

echo "[5/6] CSV summary:"
wc -l data/openstack_metrics_http_75min.csv
head data/openstack_metrics_http_75min.csv
tail data/openstack_metrics_http_75min.csv

echo "[6/6] Finished at $(date)"
"""

# to execute: chmod +x ~/fcdm_inference/run_experiment_http.sh
# 2 - RUN : ./run_experiment_http.sh

# 3 - then Run your model on this NEW OpenStack metrics CSV : python infer_openstack_http_next10min.py
# checkpoint: 
# wc -l output/openstack_live/predictions_next10min_http.csv
# head output/openstack_live/predictions_next10min_http.csv

# 4 - copy live metrics and forecasted files from vm to local terminal for analysis and plotting:
# Command from local terminal: scp ubuntu@10.196.243.15:~/fcdm_inference/data/openstack_metrics_http_75min.csv /Users/annka/Documents/Study/MSC_in_Cloud/MS_THESIS/PHASE2/Implementation/Final_Thesis_Project/output/openstack_csv/
