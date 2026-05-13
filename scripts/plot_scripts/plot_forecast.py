# Script: plot_forecast.py
# Goal: Load the CSV with true vs predicted values, and create line plots for each metric woth rolling evaluation. This is for visual inspection of how well the model is doing over time, especially in the last few minutes. We will plot CPU, Memory, and Request Rate separately, showing true vs predicted values. The plots will be saved to "plots/after_training/".
# Goal: showing true vs predicted over time. Save plots to "plots/after_training/".

import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Configuration
# --------------------------

CSV_PATH = "output/hf_run_eval/predictions_flat_real.csv"
PLOT_DIR = "plots/after_training"

# Create output folder if it does not exist
os.makedirs(PLOT_DIR, exist_ok=True)

# --------------------------
# Load Data
# --------------------------

df = pd.read_csv(CSV_PATH)

# Each row corresponds to 1 minute interval
df["t"] = range(len(df))  # Time in minutes

# Plot only last N minutes for clarity
LAST_N = 300
if len(df) > LAST_N:
    df = df.tail(LAST_N).reset_index(drop=True)
    df["t"] = range(len(df))

# --------------------------
# CPU Plot (Percentage)
# --------------------------

plt.figure(figsize=(10, 5))
plt.plot(df["t"], df["true_cpu_mean"] * 100, label="True CPU", linewidth=2)
plt.plot(df["t"], df["pred_cpu_mean"] * 100, label="Predicted CPU", linestyle="--", linewidth=2)

#plt.title("CPU: True vs Predicted")
plt.xlabel("Time (minutes)")
plt.ylabel("CPU Utilization (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/cpu_true_vs_pred.png", dpi=300)
plt.close()

# --------------------------
# Memory Plot (Percentage)
# --------------------------

plt.figure(figsize=(10, 5))
plt.plot(df["t"], df["true_memory_mean"] * 100, label="True Memory", linewidth=2)
plt.plot(df["t"], df["pred_memory_mean"] * 100, label="Predicted Memory", linestyle="--", linewidth=2)

#plt.title("Memory: True vs Predicted")
plt.xlabel("Time (minutes)")
plt.ylabel("Memory Utilization (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/memory_true_vs_pred.png", dpi=300)
plt.close()

# --------------------------
# Request Rate Plot (Real Units)
# --------------------------

plt.figure(figsize=(10, 5))
plt.plot(df["t"], df["true_request_rate_mean"], label="True Request Rate", linewidth=2)
plt.plot(df["t"], df["pred_request_rate_mean"], label="Predicted Request Rate", linestyle="--", linewidth=2)

#plt.title("Request Rate: True vs Predicted")
plt.xlabel("Time (minutes)")
plt.ylabel("Request Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/request_rate_true_vs_pred.png", dpi=300)
plt.close()

print("Plots saved in:", PLOT_DIR)
