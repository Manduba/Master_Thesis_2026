import os                           #create folders, join paths (output/hf_run_eval/.
import json                         #save metrics/run info as .json
import pickle                       #save metrics/run info as .json
import numpy as np                  #handle arrays + CSV data
import pandas as pd                 #handle arrays + CSV data
import torch                        #tensors + training loop
from torch.utils.data import Dataset, DataLoader            #PyTorch way to feed batches
import matplotlib.pyplot as plt                             #save loss curve plot


#MinMaxScaler → scales 3 features into [0, 1]
#metrics → MAE + MSE (used for RMSE)
#HF model classes → configure and create the time series transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


# =========================
# Configuration: PATHS (edit if needed)
# =========================
RAW_CSV_PATH = "output/final_service_timebased.csv"  # raw data with many columns
TIME_COL = "ts"
COLS = ["cpu_mean", "memory_mean", "request_rate_mean"]

#Everything saved (model, scaler, metrics, plots) goes into this folder.
OUT_DIR = "output/hf_run_eval"  # outputs saved here


#Model window length (edit if needed)
CONTEXT_LEN = 60        #(use the last 60 minutes as input history)
LAGS_SEQUENCE = [1]     #HF TimeSeriesTransformer uses lagged values internally
HISTORY_LEN = CONTEXT_LEN + max(LAGS_SEQUENCE)  # 61: #HF requirement: past length = context_length + max(lags) = 61
PRED_LEN = 10           #10minutes into the future (5 steps of 2min each): forecast next 10 minutes ahead.
STRIDE = 1              #when you build sliding windows, move by 1 step each sample

#Train/val/test split settings
TRAIN_RATIO = 0.70              #First 70% windows → train
VAL_RATIO = 0.15                #Next 15% windows → validation
#Remaining 15% → test


#Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4                   #learning rate for AdamW optimizer
WEIGHT_DECAY = 1e-2         #Weight decay regularization
PATIENCE = 10               #Early stopping: stop if validation doesn’t improve for 10 epochs

#Transformer model size
#This defines the architecture. Small model (good for your dataset and laptop. I tested largem model but result is bad).
D_MODEL = 64
ENC_LAYERS = 2
DEC_LAYERS = 2
HEADS = 4
DROPOUT = 0.1

#runs are reproducible (same random splits/initialization)
SEED = 42


#Helper functions
#Create directory (if missing)
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

#rmse: Computes RMSE from sklearn MSE.
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

#sMape
#sMAPE = symmetric MAPE (safe even when values are small). If denom is 0, avoid division error by using tiny 1e-8.
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denom))

#Dataset class: WindowDataset
#WindowDataset takes the normalized array arr of shape (N, D) and converts it into many sliding windows.

class WindowDataset(Dataset):           #turning time series into supervised learning samples
    def __init__(self, arr: np.ndarray, history_len: int, pred_len: int, stride: int = 1):
        self.arr = arr.astype(np.float32)  # (N, D)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.D = self.arr.shape[1]          #number of features (3)

        #Figure out how many windows can be extracted:
        max_start = len(self.arr) - self.history_len - self.pred_len + 1
        if max_start <= 0:
            raise ValueError(
                f"Not enough rows. Need at least {self.history_len + self.pred_len}, got {len(self.arr)}"
            )
        self.starts = list(range(0, max_start, self.stride))        #starts stores all valid start indices.

    #length
    def __len__(self):
        return len(self.starts)

    #Creates a simple time feature: [0, 1) increasing.
    #shape (length, 1)
    #HF needs time features, so we give at least 1.
    def _age(self, length: int) -> np.ndarray:
        return (np.arange(length, dtype=np.float32) / max(1.0, float(length))).reshape(length, 1)

    #Past window = 61 rows, future window = next 10 rows.
    def __getitem__(self, idx):
        s = self.starts[idx]

        past = self.arr[s : s + self.history_len]  # (HISTORY_LEN, D)
        future = self.arr[s + self.history_len : s + self.history_len + self.pred_len]  # (PRED_LEN, D)

        #Observed mask tells model which values exist. Here everything exists (no missing), so all ones.
        past_mask = np.ones_like(past, dtype=np.float32)
        future_mask = np.ones_like(future, dtype=np.float32)

        #This is exactly the format HF model expects as in documentation
        return {
            "past_values": torch.from_numpy(past),
            "future_values": torch.from_numpy(future),
            "past_observed_mask": torch.from_numpy(past_mask),
            "future_observed_mask": torch.from_numpy(future_mask),
            "past_time_features": torch.from_numpy(self._age(self.history_len)),  # (HISTORY_LEN, 1)
            "future_time_features": torch.from_numpy(self._age(self.pred_len)),   # (PRED_LEN, 1)
        }

#Fix HF generate() output shape
#Hugging Face model.generate() can return predictions in different tensor shapes (sometimes with a “num_samples” dimension). The helper _extract_preds_to_match_truth() normalizes that output into the expected shape (B, PRED_LEN, D) by selecting one sample if multiple exist.This prevents shape-mismatch bugs when comparing predictions vs ground truth."

def _extract_preds_to_match_truth(gen_out, true_shape):
    # Pull tensor out of HF output object
    if hasattr(gen_out, "sequences") and gen_out.sequences is not None:
        pt = gen_out.sequences
    elif hasattr(gen_out, "predictions") and gen_out.predictions is not None:
        pt = gen_out.predictions
    else:
        pt = gen_out

    if not torch.is_tensor(pt):
        raise ValueError(f"generate() returned unsupported type: {type(pt)}")

    # Normalize to (B, pred, D)
    if pt.ndim == 4:
        B, P, D = true_shape
        # likely (B, num_samples, P, D)
        if pt.shape[0] == B and pt.shape[2] == P and pt.shape[3] == D:
            pt = pt[:, 0]      # take sample 0
        # or (num_samples, B, P, D)
        elif pt.shape[1] == B and pt.shape[2] == P and pt.shape[3] == D:
            pt = pt[0]         # take sample 0
        else:
            pt = pt[0]
    elif pt.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected prediction tensor shape: {tuple(pt.shape)}")

    B, P, D = true_shape
    pt = pt[:, :P, :D]
    return pt

#This computes MAE/RMSE/sMAPE for each of 3 features.
def _metrics_dict(yt_flat, yp_flat, cols):
    out = {}
    for i, name in enumerate(cols):
        out[name] = {
            "MAE": float(mean_absolute_error(yt_flat[:, i], yp_flat[:, i])),
            "RMSE": rmse(yt_flat[:, i], yp_flat[:, i]),
            "sMAPE": smape(yt_flat[:, i], yp_flat[:, i]),
        }
    return out


def main():
    # Set seeds + device
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(OUT_DIR)

    # ---- Load RAW CSV
    df = pd.read_csv(RAW_CSV_PATH)
    missing = [c for c in [TIME_COL] + COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nAvailable: {df.columns.tolist()}")

    df = df.sort_values(TIME_COL).reset_index(drop=True)    #Checks required columns exist
    #Make sure those three columns are numeric and drop rows with missing values.
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=COLS).reset_index(drop=True)

    #Scaling (this is the key part)
    # ---- Fit MinMaxScaler on ONLY the 3 model columns (compatible with your current venv)
    #Fit MinMax on raw data → learns min/max per column. 
    # Transform values into normalized
    scaler = MinMaxScaler()
    X_raw = df[COLS].values.astype(np.float32)
    X_norm = scaler.fit_transform(X_raw)

    # Save the NEW scaler (this one will load fine in this venv + in OpenStack)
    scaler_out = os.path.join(OUT_DIR, "minmax_scaler.pkl")
    with open(scaler_out, "wb") as f:
        pickle.dump(scaler, f)

    # training array is normalized
    arr = X_norm.astype(np.float32)
    D = arr.shape[1]

    #Save run metadata
    #Saves your setup so you can reproduce.
    with open(os.path.join(OUT_DIR, "run_info.json"), "w") as f:
        json.dump(
            {
                "raw_csv_path": RAW_CSV_PATH,
                "cols": COLS,
                "context_len": CONTEXT_LEN,
                "lags_sequence": LAGS_SEQUENCE,
                "history_len": HISTORY_LEN,
                "pred_len": PRED_LEN,
                "rows": int(arr.shape[0]),
                "saved_scaler": scaler_out,
            },
            f,
            indent=2,
        )

    # ---- Build Dataset + split
    #Build windows and compute split sizes.
    dataset = WindowDataset(arr, HISTORY_LEN, PRED_LEN, STRIDE)
    n = len(dataset)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    #Sequential splits (keeps time ordering).
    train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    test_ds = torch.utils.data.Subset(dataset, range(n_train + n_val, n))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Create HF model: Define the TimeSeriesTransformer model configuration
    # This defines the architecture and expected input shapes.
    model_cfg = TimeSeriesTransformerConfig(
        context_length=CONTEXT_LEN,
        prediction_length=PRED_LEN,
        input_size=D,
        num_time_features=1,
        lags_sequence=LAGS_SEQUENCE,
        d_model=D_MODEL,
        encoder_layers=ENC_LAYERS,
        decoder_layers=DEC_LAYERS,
        attention_heads=HEADS,
        dropout=DROPOUT,
    )

    #Create model.
    model = TimeSeriesTransformerForPrediction(model_cfg).to(device)
    print("Using lags_sequence:", model.config.lags_sequence)
    print("HF required past length:", model.config.context_length + max(model.config.lags_sequence))
    print("We provide HISTORY_LEN:", HISTORY_LEN)

    #Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop + validation + early stopping
    # ---- Training
    best_val = float("inf")
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        tr = 0.0

        for batch in train_loader:
            past_values = batch["past_values"].to(device)
            future_values = batch["future_values"].to(device)
            past_observed_mask = batch["past_observed_mask"].to(device)
            future_observed_mask = batch["future_observed_mask"].to(device)
            past_time_features = batch["past_time_features"].to(device)
            future_time_features = batch["future_time_features"].to(device)

            optimizer.zero_grad()
            out = model(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_time_features=future_time_features,
                future_values=future_values,
                future_observed_mask=future_observed_mask,
            )
            out.loss.backward()         # loss comes FROM the HF model, not defined by you, but it’s typically MSE loss between predicted vs true future values in normalized space.
            optimizer.step()
            tr += out.loss.item()

        tr /= max(1, len(train_loader))
        train_losses.append(tr)

        #Test prediction
        #Generates predictions.
        #Fixes shape from (B,100,10,3) to (B,10,3).
        # Saves all predictions and true values.
        model.eval()
        va = 0.0
        with torch.no_grad():
            for batch in val_loader:
                past_values = batch["past_values"].to(device)
                future_values = batch["future_values"].to(device)
                past_observed_mask = batch["past_observed_mask"].to(device)
                future_observed_mask = batch["future_observed_mask"].to(device)
                past_time_features = batch["past_time_features"].to(device)
                future_time_features = batch["future_time_features"].to(device)

                out = model(
                    past_values=past_values,
                    past_time_features=past_time_features,
                    past_observed_mask=past_observed_mask,
                    future_time_features=future_time_features,
                    future_values=future_values,
                    future_observed_mask=future_observed_mask,
                )
                va += out.loss.item()

        va /= max(1, len(val_loader))
        val_losses.append(va)

        print(f"Epoch {epoch+1}/{EPOCHS}  train={tr:.4f}  val={va:.4f}")

        if va < best_val:
            best_val = va
            wait = 0
            model.save_pretrained(os.path.join(OUT_DIR, "model"))
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    # ---- Loss curve: plot
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    #plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"))
    plt.close()

    # ---- Test prediction
    #Generates predictions.
    #Fixes shape from (B,100,10,3) to (B,10,3).
    #Saves all predictions and true values.
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            past_values = batch["past_values"].to(device)
            past_observed_mask = batch["past_observed_mask"].to(device)
            past_time_features = batch["past_time_features"].to(device)
            future_time_features = batch["future_time_features"].to(device)

            future_values_cpu = batch["future_values"]  # normalized truth (CPU)
            true_shape = future_values_cpu.shape        # (B, PRED_LEN, D)

            gen_out = model.generate(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_time_features=future_time_features,
            )

            pred_tensor = _extract_preds_to_match_truth(gen_out, true_shape)

            if batch_idx == 0:
                raw = gen_out.sequences if hasattr(gen_out, "sequences") and gen_out.sequences is not None else (
                      gen_out.predictions if hasattr(gen_out, "predictions") and gen_out.predictions is not None else gen_out)
                print("DEBUG true_shape:", true_shape)
                if torch.is_tensor(raw):
                    print("DEBUG raw pred shape:", tuple(raw.shape))
                print("DEBUG normalized pred shape:", tuple(pred_tensor.shape))

            y_pred.append(pred_tensor.detach().cpu().numpy())
            y_true.append(future_values_cpu.numpy())

    y_pred = np.concatenate(y_pred, axis=0)  # normalized (N, P, D)
    y_true = np.concatenate(y_true, axis=0)  # normalized (N, P, D)

    # ---- Metrics (normalized):Compute metrics in normalized space
    yp_norm = y_pred.reshape(-1, D)
    yt_norm = y_true.reshape(-1, D)
    metrics_norm = _metrics_dict(yt_norm, yp_norm, COLS)
    with open(os.path.join(OUT_DIR, "metrics_normalized.json"), "w") as f:
        json.dump(metrics_norm, f, indent=2)

    pd.DataFrame({
        f"true_{COLS[0]}": yt_norm[:, 0], f"pred_{COLS[0]}": yp_norm[:, 0],
        f"true_{COLS[1]}": yt_norm[:, 1], f"pred_{COLS[1]}": yp_norm[:, 1],
        f"true_{COLS[2]}": yt_norm[:, 2], f"pred_{COLS[2]}": yp_norm[:, 2],
    }).to_csv(os.path.join(OUT_DIR, "predictions_flat_normalized.csv"), index=False)

    # ---- Metrics (REAL units) via inverse-transform using the NEW scaler
    #Convert predictions back to raw units.
    #Compute real-unit errors.
    yp_real = scaler.inverse_transform(yp_norm)
    yt_real = scaler.inverse_transform(yt_norm)
    metrics_real = _metrics_dict(yt_real, yp_real, COLS)
    with open(os.path.join(OUT_DIR, "metrics_real.json"), "w") as f:
        json.dump(metrics_real, f, indent=2)

    pd.DataFrame({
        f"true_{COLS[0]}": yt_real[:, 0], f"pred_{COLS[0]}": yp_real[:, 0],
        f"true_{COLS[1]}": yt_real[:, 1], f"pred_{COLS[1]}": yp_real[:, 1],
        f"true_{COLS[2]}": yt_real[:, 2], f"pred_{COLS[2]}": yp_real[:, 2],
    }).to_csv(os.path.join(OUT_DIR, "predictions_flat_real.csv"), index=False)

    # ---- Save prediction-only CSV (NO ground truth)
    pred_only_df = pd.DataFrame({
        "pred_cpu_mean": yp_real[:, 0],
        "pred_memory_mean": yp_real[:, 1],
        "pred_request_rate_mean": yp_real[:, 2],
    })

    pred_only_path = os.path.join(OUT_DIR, "predictions_only.csv")
    pred_only_df.to_csv(pred_only_path, index=False)

    print("Saved prediction-only CSV to:", pred_only_path)

    print("Done.")
    print("Normalized metrics:\n", json.dumps(metrics_norm, indent=2))
    print("Real-unit metrics:\n", json.dumps(metrics_real, indent=2))
    print("Saved NEW scaler to:", scaler_out)       #This is the scaler you will reuse in OpenStack inference.


if __name__ == "__main__":
    main()
