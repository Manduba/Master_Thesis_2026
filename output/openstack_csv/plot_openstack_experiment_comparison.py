import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# INPUT PATHS
# Run this script from your project root folder
# ------------------------------------------------------------
BASE_DIR = "output/openstack_csv"

HTTP_METRICS = os.path.join(BASE_DIR, "openstack_metrics_http_75min.csv")
PLACEHOLDER_METRICS = os.path.join(BASE_DIR, "openstack_metrics_placeholder_rr.csv")

PRED_HIST = os.path.join(BASE_DIR, "predictions_historical_input_next10min.csv")
PRED_HTTP = os.path.join(BASE_DIR, "predictions_http_next10min.csv")
PRED_PLACEHOLDER = os.path.join(BASE_DIR, "predictions_next10min_placeholder_rr.csv")

# ------------------------------------------------------------
# OUTPUT PATH
# ------------------------------------------------------------
OUT_DIR = "plots/after_training/openstack_live"

# How many last real points to show before forecast begins
LAST_REAL_POINTS_TO_SHOW = 25

# If logger is ~1 row/min, keep this 1
STEP_MINUTES = 1


def make_future_minutes(last_real_minute: float, steps: int, step_minutes: int):
    return [last_real_minute + i * step_minutes for i in range(1, steps + 1)]


def plot_one_series(x, y, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(9, 4))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_two_series(x1, y1, label1, x2, y2, label2, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(9, 4))
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_three_series(x, y1, label1, y2, label2, y3, label3, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(9, 4))
    plt.plot(x, y1, marker="o", label=label1)
    plt.plot(x, y2, marker="o", label=label2)
    plt.plot(x, y3, marker="o", label=label3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_collected_vs_predicted(x_real, y_real, x_pred, y_pred, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(9, 4))
    plt.plot(x_real, y_real, label="Collected (OpenStack)")
    plt.plot(x_pred, y_pred, label="Predicted (next 10 min)")
    plt.axvline(x=x_real[-1], linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_prediction_only(x_pred, y_pred, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(9, 4))
    plt.plot(x_pred, y_pred, marker="o", label="Predicted (next 10 min)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load live metrics
    # --------------------------------------------------------
    df_http = pd.read_csv(HTTP_METRICS)
    df_placeholder = pd.read_csv(PLACEHOLDER_METRICS)

    df_http["ts"] = pd.to_datetime(df_http["ts"])
    df_placeholder["ts"] = pd.to_datetime(df_placeholder["ts"])

    df_http["minute"] = (df_http["ts"] - df_http["ts"].iloc[0]).dt.total_seconds() / 60.0
    df_placeholder["minute"] = (df_placeholder["ts"] - df_placeholder["ts"].iloc[0]).dt.total_seconds() / 60.0

    # --------------------------------------------------------
    # Plot 1: HTTP live metrics over time
    # CPU and memory converted to percentage
    # --------------------------------------------------------
    plot_one_series(
        df_http["minute"], df_http["cpu_mean"] * 100,
        "Minutes since start", "CPU Utilization (%)",
        "HTTP live experiment: CPU over time",
        os.path.join(OUT_DIR, "http_live_cpu.png")
    )

    plot_one_series(
        df_http["minute"], df_http["memory_mean"] * 100,
        "Minutes since start", "Memory Utilization (%)",
        "HTTP live experiment: Memory over time",
        os.path.join(OUT_DIR, "http_live_memory.png")
    )

    plot_one_series(
        df_http["minute"], df_http["request_rate_mean"],
        "Minutes since start", "Request Rate (req/s)",
        "HTTP live experiment: Request rate over time",
        os.path.join(OUT_DIR, "http_live_request_rate.png")
    )

    # --------------------------------------------------------
    # Plot 2: Placeholder live metrics over time
    # CPU and memory converted to percentage
    # --------------------------------------------------------
    plot_one_series(
        df_placeholder["minute"], df_placeholder["cpu_mean"] * 100,
        "Minutes since start", "CPU Utilization (%)",
        "Placeholder experiment: CPU over time",
        os.path.join(OUT_DIR, "placeholder_live_cpu.png")
    )

    plot_one_series(
        df_placeholder["minute"], df_placeholder["memory_mean"] * 100,
        "Minutes since start", "Memory Utilization (%)",
        "Placeholder experiment: Memory over time",
        os.path.join(OUT_DIR, "placeholder_live_memory.png")
    )

    plot_one_series(
        df_placeholder["minute"], df_placeholder["request_rate_mean"],
        "Minutes since start", "Request Rate (req/s)",
        "Placeholder experiment: Request rate over time",
        os.path.join(OUT_DIR, "placeholder_live_request_rate.png")
    )

    # --------------------------------------------------------
    # Plot 3: Request-rate comparison between experiments
    # --------------------------------------------------------
    plot_two_series(
        df_http["minute"], df_http["request_rate_mean"], "HTTP experiment",
        df_placeholder["minute"], df_placeholder["request_rate_mean"], "Placeholder rr=0",
        "Minutes since start", "Request Rate (req/s)",
        "Request-rate comparison: HTTP vs placeholder experiment",
        os.path.join(OUT_DIR, "request_rate_comparison_http_vs_placeholder.png")
    )

    # --------------------------------------------------------
    # Load prediction files
    # --------------------------------------------------------
    pred_hist = pd.read_csv(PRED_HIST)
    pred_http = pd.read_csv(PRED_HTTP)
    pred_placeholder = pd.read_csv(PRED_PLACEHOLDER)

    # --------------------------------------------------------
    # Plot 4: Forecast comparison across 3 OpenStack cases
    # CPU and memory converted to percentage
    # --------------------------------------------------------
    x_pred_steps = pred_hist["t_plus_step"]

    plot_three_series(
        x_pred_steps,
        pred_hist["cpu_pred"] * 100, "Historical input",
        pred_placeholder["cpu_pred"] * 100, "Placeholder rr=0",
        pred_http["cpu_pred"] * 100, "HTTP request rate",
        "Forecast step (t+)", "CPU Utilization (%)",
        "CPU forecast comparison across OpenStack cases",
        os.path.join(OUT_DIR, "forecast_comparison_cpu.png")
    )

    plot_three_series(
        x_pred_steps,
        pred_hist["memory_pred"] * 100, "Historical input",
        pred_placeholder["memory_pred"] * 100, "Placeholder rr=0",
        pred_http["memory_pred"] * 100, "HTTP request rate",
        "Forecast step (t+)", "Memory Utilization (%)",
        "Memory forecast comparison across OpenStack cases",
        os.path.join(OUT_DIR, "forecast_comparison_memory.png")
    )

    plot_three_series(
        x_pred_steps,
        pred_hist["request_rate_pred"], "Historical input",
        pred_placeholder["request_rate_pred"], "Placeholder rr=0",
        pred_http["request_rate_pred"], "HTTP request rate",
        "Forecast step (t+)", "Request Rate Prediction (req/s)",
        "Request-rate forecast comparison across OpenStack cases",
        os.path.join(OUT_DIR, "forecast_comparison_request_rate.png")
    )

    # --------------------------------------------------------
    # Plot 5: Collected vs predicted for HTTP experiment
    # CPU and memory converted to percentage
    # --------------------------------------------------------
    df_http_tail = df_http.tail(LAST_REAL_POINTS_TO_SHOW).reset_index(drop=True)
    x_real = df_http_tail["minute"].tolist()
    last_real_minute = float(x_real[-1])

    steps = len(pred_http)
    x_future = make_future_minutes(last_real_minute, steps, STEP_MINUTES)

    plot_collected_vs_predicted(
        x_real, (df_http_tail["cpu_mean"] * 100).tolist(),
        x_future, (pred_http["cpu_pred"] * 100).tolist(),
        "Minutes since start", "CPU Utilization (%)",
        "HTTP experiment: collected vs predicted CPU",
        os.path.join(OUT_DIR, "http_cpu_collected_vs_predicted.png")
    )

    plot_collected_vs_predicted(
        x_real, (df_http_tail["memory_mean"] * 100).tolist(),
        x_future, (pred_http["memory_pred"] * 100).tolist(),
        "Minutes since start", "Memory Utilization (%)",
        "HTTP experiment: collected vs predicted memory",
        os.path.join(OUT_DIR, "http_memory_collected_vs_predicted.png")
    )

    plot_collected_vs_predicted(
        x_real, df_http_tail["request_rate_mean"].tolist(),
        x_future, pred_http["request_rate_pred"].tolist(),
        "Minutes since start", "Request Rate (req/s)",
        "HTTP experiment: collected vs predicted request rate",
        os.path.join(OUT_DIR, "http_request_rate_collected_vs_predicted.png")
    )

    # --------------------------------------------------------
    # Plot 6: Prediction-only plots for each forecast case
    # CPU and memory converted to percentage
    # --------------------------------------------------------
    x_hist = pred_hist["t_plus_step"]
    x_http = pred_http["t_plus_step"]
    x_placeholder = pred_placeholder["t_plus_step"]

    # Historical input predictions only
    plot_prediction_only(
        x_hist, pred_hist["cpu_pred"] * 100,
        "Forecast step (t+)", "CPU Utilization (%)",
        "Historical input: predicted CPU vs time",
        os.path.join(OUT_DIR, "historical_cpu_predicted_vs_time.png")
    )

    plot_prediction_only(
        x_hist, pred_hist["memory_pred"] * 100,
        "Forecast step (t+)", "Memory Utilization (%)",
        "Historical input: predicted memory vs time",
        os.path.join(OUT_DIR, "historical_memory_predicted_vs_time.png")
    )

    plot_prediction_only(
        x_hist, pred_hist["request_rate_pred"],
        "Forecast step (t+)", "Request Rate (req/s)",
        "Historical input: predicted request rate vs time",
        os.path.join(OUT_DIR, "historical_request_rate_predicted_vs_time.png")
    )

    # Placeholder rr=0 predictions only
    plot_prediction_only(
        x_placeholder, pred_placeholder["cpu_pred"] * 100,
        "Forecast step (t+)", "CPU Utilization (%)",
        "Placeholder rr=0: predicted CPU vs time",
        os.path.join(OUT_DIR, "placeholder_cpu_predicted_vs_time.png")
    )

    plot_prediction_only(
        x_placeholder, pred_placeholder["memory_pred"] * 100,
        "Forecast step (t+)", "Memory Utilization (%)",
        "Placeholder rr=0: predicted memory vs time",
        os.path.join(OUT_DIR, "placeholder_memory_predicted_vs_time.png")
    )

    plot_prediction_only(
        x_placeholder, pred_placeholder["request_rate_pred"],
        "Forecast step (t+)", "Request Rate (req/s)",
        "Placeholder rr=0: predicted request rate vs time",
        os.path.join(OUT_DIR, "placeholder_request_rate_predicted_vs_time.png")
    )

    # HTTP request-rate predictions only
    plot_prediction_only(
        x_http, pred_http["cpu_pred"] * 100,
        "Forecast step (t+)", "CPU Utilization (%)",
        "HTTP request-rate: predicted CPU vs time",
        os.path.join(OUT_DIR, "http_cpu_predicted_vs_time.png")
    )

    plot_prediction_only(
        x_http, pred_http["memory_pred"] * 100,
        "Forecast step (t+)", "Memory Utilization (%)",
        "HTTP request-rate: predicted memory vs time",
        os.path.join(OUT_DIR, "http_memory_predicted_vs_time.png")
    )

    plot_prediction_only(
        x_http, pred_http["request_rate_pred"],
        "Forecast step (t+)", "Request Rate (req/s)",
        "HTTP request-rate: predicted request rate vs time",
        os.path.join(OUT_DIR, "http_request_rate_predicted_vs_time.png")
    )

    print("Saved plots to:", OUT_DIR)


if __name__ == "__main__":
    main()