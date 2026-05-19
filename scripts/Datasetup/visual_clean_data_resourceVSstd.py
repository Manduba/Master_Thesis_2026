import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Read CSV
df = pd.read_csv("output/final_service_timebased.csv")
print(f"Loaded {len(df)} rows")

# Convert ts to datetime
df["ts"] = pd.to_datetime(df["ts"], format="%H:%M:%S")

# 4-minute resolution
df_4min = df.iloc[::4].reset_index(drop=True)

# --- VERIFICATION ---
print("Sample of 4min data (first 3 rows):")
print(df_4min[["ts", "cpu_mean", "cpu_std", "memory_mean", "memory_std", "request_rate_mean", "request_rate_std"]].head(3))
print(f"Memory std range: {df_4min['memory_std'].min():.4f} - {df_4min['memory_std'].max():.4f}")
print(f"Request rate range: {df_4min['request_rate_mean'].min():.0f} - {df_4min['request_rate_mean'].max():.0f}")

# Create output directory
save_dir = Path("plots/errorbar_4min")
save_dir.mkdir(parents=True, exist_ok=True)

def format_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# CPU plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.errorbar(df_4min["ts"], df_4min["cpu_mean"], yerr=df_4min["cpu_std"],
            fmt='-o', markersize=3, capsize=3, label="CPU mean ± std")
ax.set_xlabel("Time"); ax.set_ylabel("CPU Utilization (mean)")
#ax.set_title("CPU Mean Utilization with Std Dev (4‑min resolution)")
ax.grid(True, linestyle="--", alpha=0.6); ax.legend()
format_time_axis(ax)
plt.tight_layout()
plt.savefig(save_dir / "cpu_mean_vs_std_cpu_4min.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# Memory plot (same structure)
fig, ax = plt.subplots(figsize=(12, 6))
ax.errorbar(df_4min["ts"], df_4min["memory_mean"], yerr=df_4min["memory_std"],
            fmt='-o', markersize=3, capsize=3, label="Memory mean ± std")
ax.set_xlabel("Time"); ax.set_ylabel("Memory Utilization (mean)")
#ax.set_title("Memory Mean Utilization with Std Dev (4‑min resolution)")
ax.grid(True, linestyle="--", alpha=0.6); ax.legend()
format_time_axis(ax)
plt.tight_layout()
plt.savefig(save_dir / "memory_mean_vs_std_memory_4min.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# Request rate plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.errorbar(df_4min["ts"], df_4min["request_rate_mean"], yerr=df_4min["request_rate_std"],
            fmt='-o', markersize=3, capsize=3, label="Request rate mean ± std")
ax.set_xlabel("Time"); ax.set_ylabel("Request Rate (mean)")
#ax.set_title("Request Rate Mean with Std Dev (4‑min resolution)")
ax.grid(True, linestyle="--", alpha=0.6); ax.legend()
format_time_axis(ax)
plt.tight_layout()
plt.savefig(save_dir / "request_rate_mean_vs_std_4min.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)