# infer_next10min.py is copied and modified to infer_openstack_http_live.py with few  changes in paths and names. The rest of the code is the same as infer_next10min.py
# Goal: Load trained HF TimeSeriesTransformer + saved MinMax scaler,
#       read a small input CSV, forecast next 10 steps, save:
#       output/predictions_next10min.csv

import pickle
import numpy as np
import pandas as pd
import torch
from transformers import TimeSeriesTransformerForPrediction

RAW_CSV_PATH = "data/openstack_metrics_http_75min_live.csv"
MODEL_DIR = "model"
SCALER_PATH = "minmax_scaler.pkl"
OUT_PATH = "output/openstack_live/predictions_http_live.csv"

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

