#plots the 10-minute ahead forecast for CPU, using the predictions_only.csv and final_service_timebased.csv files for only one block of 10 minutes (first 10 rows). This is a sanity check to see if the predictions look reasonable for a single block before plotting the full timeline. The plot is saved as cpu_10min_example.png in the plots/after_training folder.
# Note: This is a simplified example for one block of 10 minutes. For a full timeline plot, see plot_forecast.py which plots the entire predicted vs true values over time with rolling evaluation. 
# This script is just to visualize the 10-minute ahead forecast for CPU for one block of predictions to ensure they look reasonable before plotting the full timeline.


import pandas as pd
import matplotlib.pyplot as plt

# Load predictions (10-step blocks)
#pred = pd.read_csv("output/hf_run_eval/predictions_only.csv")
pred = pd.read_csv("output/hf_run_eval/predictions_flat_real.csv")

# Load full true dataset
true = pd.read_csv("output/final_service_timebased.csv")

# ---- Pick block 0 (first 10 rows) ----
block_index = 0
start = block_index * 10
end = start + 10

pred_block = pred.iloc[start:end].reset_index(drop=True)

# For simplicity, take first 10 true rows
true_block = true.iloc[:10].reset_index(drop=True)

# Create minute ahead axis
minutes_ahead = range(1, 11)

# Convert CPU to percentage
true_cpu = true_block["cpu_mean"] * 100
pred_cpu = pred_block["pred_cpu_mean"] * 100

# Plot
plt.figure(figsize=(8, 5))
plt.plot(minutes_ahead, true_cpu, label="True CPU", linewidth=2)
plt.plot(minutes_ahead, pred_cpu, label="Predicted CPU", linestyle="--", linewidth=2)

plt.xlabel("Minutes Ahead")
plt.ylabel("CPU Utilization (%)")
plt.title("Example 10-Minute Ahead Forecast (CPU)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/after_training/cpu_10min_example.png", dpi=300)
plt.close()

print("Saved: cpu_10min_example.png")

