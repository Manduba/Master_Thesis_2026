#!/usr/bin/env python3

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path


# -------------------------
# Paths
# -------------------------

def default_csv_path() -> str:
    return (
        Path(__file__).resolve().parents[2]
        / "output"
        / "final_service_timebased.csv"
    ).as_posix()


CSV_PATH = default_csv_path()


LABELS = {
    "cpu_mean": "CPU mean",
    "cpu_std": "CPU std.",
    "memory_mean": "Memory mean",
    "memory_std": "Memory std.",
    "request_rate_mean": "Request rate mean",
    "request_rate_std": "Request rate std.",
    "total_request_rate_mean": "Request rate mean",
    "total_request_rate_std": "Request rate std.",
}

COLORS = {
    "cpu": "#0072B2",
    "memory": "#009E73",
    "request": "#D55E00",
}


# -------------------------
# Helpers
# -------------------------

def pick_col(df: pd.DataFrame, *candidates: str):
    """Return the first existing column name from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_time(df: pd.DataFrame) -> pd.DataFrame:
    df["time"] = pd.to_datetime(df["ts"], format="%H:%M:%S")
    return df


def style_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def apply_paper_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def label_for(col):
    return LABELS.get(col, col.replace("_", " ").title())


def save_figure(fig, outpath_base):
    outbase = Path(outpath_base)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".eps"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), bbox_inches="tight")


def set_zoomed_ylim(ax, values, include_zero=False):
    ymin = float(values.min())
    ymax = float(values.max())
    if include_zero:
        ymin = min(0.0, ymin)
    if ymin == ymax:
        pad = abs(ymin) * 0.05 if ymin else 0.05
    else:
        pad = (ymax - ymin) * 0.12
    ax.set_ylim(ymin - pad, ymax + pad)


# -------------------------
# Plot functions
# -------------------------

def plot_simple(df, cols, ylabel, outpath_base, force01=False, zoom_y=False, ylim=None):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    plotted = False
    for c in cols:
        if c and c in df.columns:
            ax.plot(df["time"], df[c], label=label_for(c))
            plotted = True

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)

    style_time_axis(ax)

    if ylim:
        ax.set_ylim(*ylim)
    elif force01:
        ax.set_ylim(0, 1)
    elif zoom_y and plotted:
        y_values = df[[c for c in cols if c and c in df.columns]].stack()
        set_zoomed_ylim(ax, y_values)

    if plotted:
        ax.legend(frameon=False)

    fig.tight_layout()

    save_figure(fig, outpath_base)
    plt.close(fig)


def plot_dual(
    df,
    left_cols,
    right_col,
    left_ylabel,
    right_ylabel,
    outpath_base,
    force01_left=False,
    zoom_left=False,
):
    fig, ax1 = plt.subplots(figsize=(7.2, 3.4))

    plotted_left = False
    for c in left_cols:
        if c in df.columns:
            ax1.plot(df["time"], df[c], label=label_for(c))
            plotted_left = True

    ax1.set_xlabel("Time")
    ax1.set_ylabel(left_ylabel)
    style_time_axis(ax1)

    if force01_left:
        ax1.set_ylim(0, 1)
    elif zoom_left and plotted_left:
        y_values = df[[c for c in left_cols if c in df.columns]].stack()
        set_zoomed_ylim(ax1, y_values)

    ax2 = ax1.twinx()
    plotted_right = False
    if right_col and right_col in df.columns:
        ax2.plot(
            df["time"],
            df[right_col],
            label=label_for(right_col),
            linestyle="--",
            color=COLORS["request"],
        )
        plotted_right = True

    ax2.set_ylabel(right_ylabel)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if plotted_left or plotted_right:
        fig.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout()

    save_figure(fig, outpath_base)
    plt.close(fig)


def plot_scatter_cpu_vs_req(df, req_mean_col, outpath_base):
    if not req_mean_col:
        print("⚠️ Skipping scatter (no request-rate mean column).")
        return

    # Safer if any non-positive values exist
    tmp = df[(df[req_mean_col] > 0) & (df["cpu_mean"].notna())].copy()
    if tmp.empty:
        print("⚠️ Skipping scatter (no positive request-rate values).")
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(tmp[req_mean_col], tmp["cpu_mean"], s=12, edgecolors="none")
    ax.set_xlabel("Request rate mean")
    ax.set_ylabel("CPU mean")
    ax.set_xscale("log")

    fig.tight_layout()

    save_figure(fig, outpath_base)
    plt.close(fig)


def plot_paper_summary(df, panels, outpath_base):
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        constrained_layout=True,
    )

    for ax, panel in zip(axes, panels):
        col = panel["col"]
        ax.plot(
            df["time"],
            df[col],
            color=panel["color"],
            label=label_for(col),
        )
        ax.set_title(panel["title"], loc="left", pad=4)
        ax.set_ylabel(panel["ylabel"])
        style_time_axis(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if panel.get("log_y"):
            positive_vals = df[col][df[col] > 0]
            if not positive_vals.empty and len(positive_vals) == len(df[col]):
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            else:
                set_zoomed_ylim(ax, df[col])
        else:
            set_zoomed_ylim(ax, df[col])

    axes[-1].set_xlabel("Time")
    save_figure(fig, outpath_base)
    plt.close(fig)


def plot_mean_with_std_lines(
    df,
    mean_col,
    std_col,
    ylabel,
    outpath_base,
    color,
    force01=False,
    zoom_y=False,
    use_band=False,
):
    """
    Adds a new plot without affecting old ones.
    Two modes:
    - use_band=False -> mean line + upper/lower dashed std lines
    - use_band=True  -> mean line + shaded mean±std band
    """
    if mean_col not in df.columns or std_col not in df.columns:
        print(f"⚠️ Skipping {mean_col} with {std_col} plot (missing columns).")
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    mean_vals = df[mean_col]
    std_vals = df[std_col]

    upper = mean_vals + std_vals
    lower = mean_vals - std_vals

    if force01:
        lower = lower.clip(lower=0)
        upper = upper.clip(upper=1)

    ax.plot(
        df["time"],
        mean_vals,
        color=color,
        label=label_for(mean_col),
        linewidth=1.4,
    )

    if use_band:
        ax.fill_between(
            df["time"],
            lower,
            upper,
            color=color,
            alpha=0.18,
            label="±1 std.",
        )
    else:
        ax.plot(
            df["time"],
            upper,
            color=color,
            linestyle="--",
            linewidth=1.0,
            label=f"{label_for(mean_col)} + std.",
        )
        ax.plot(
            df["time"],
            lower,
            color=color,
            linestyle="--",
            linewidth=1.0,
            label=f"{label_for(mean_col)} - std.",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    style_time_axis(ax)

    if force01:
        ax.set_ylim(0, 1)
    elif zoom_y:
        y_values = pd.concat([lower, mean_vals, upper], ignore_index=True)
        set_zoomed_ylim(ax, y_values)

    ax.legend(frameon=False)
    fig.tight_layout()

    save_figure(fig, outpath_base)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def main():
    apply_paper_style()

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    print("Loaded columns:", list(df.columns))

    req_mean_col = pick_col(df, "request_rate_mean", "total_request_rate_mean")
    req_std_col = pick_col(df, "request_rate_std", "total_request_rate_std")

    df = parse_time(df)
    df = df.sort_values("time")

    outdir = Path("plots") / "before_training"

    # =========================
    # SEPARATE MEAN PLOTS
    # =========================

    plot_simple(
        df,
        ["cpu_mean"],
        "CPU Mean Utilization",
        (outdir / "cpu_mean_only").as_posix(),
        zoom_y=True,
    )

    plot_simple(
        df,
        ["cpu_mean"],
        "CPU Mean (0-1)",
        (outdir / "cpu_mean_only_fullscale").as_posix(),
        force01=True,
    )

    plot_simple(
        df,
        ["memory_mean"],
        "Memory Mean Utilization",
        (outdir / "memory_mean_only").as_posix(),
        zoom_y=True,
    )

    plot_simple(
        df,
        ["memory_mean"],
        "Memory Mean (0-1)",
        (outdir / "memory_mean_only_fullscale").as_posix(),
        force01=True,
    )

    if req_mean_col:
        plot_simple(
            df,
            [req_mean_col],
            "Request Rate Mean",
            (outdir / "request_rate_mean_only").as_posix(),
        )

    # =========================
    # SEPARATE STD PLOTS
    # =========================

    plot_simple(
        df,
        ["cpu_std"],
        "CPU Std (spatial)",
        (outdir / "cpu_std_only").as_posix(),
    )

    plot_simple(
        df,
        ["memory_std"],
        "Memory Std (spatial)",
        (outdir / "memory_std_only").as_posix(),
    )

    if req_std_col:
        plot_simple(
            df,
            [req_std_col],
            "Request Rate Std",
            (outdir / "request_rate_std_only").as_posix(),
        )

    # =========================
    # NEW: MEAN + STD IN SINGLE PLOT
    # These ADD new files only. They do not replace old plots.
    # =========================

    plot_mean_with_std_lines(
        df,
        mean_col="cpu_mean",
        std_col="cpu_std",
        ylabel="CPU utilization (0-1)",
        outpath_base=(outdir / "cpu_mean_with_std_lines").as_posix(),
        color=COLORS["cpu"],
        force01=True,
        use_band=False,
    )

    plot_mean_with_std_lines(
        df,
        mean_col="cpu_mean",
        std_col="cpu_std",
        ylabel="CPU utilization (0-1)",
        outpath_base=(outdir / "cpu_mean_with_std_band").as_posix(),
        color=COLORS["cpu"],
        force01=True,
        use_band=True,
    )

    plot_mean_with_std_lines(
        df,
        mean_col="memory_mean",
        std_col="memory_std",
        ylabel="Memory utilization (0-1)",
        outpath_base=(outdir / "memory_mean_with_std_lines").as_posix(),
        color=COLORS["memory"],
        force01=True,
        use_band=False,
    )

    plot_mean_with_std_lines(
        df,
        mean_col="memory_mean",
        std_col="memory_std",
        ylabel="Memory utilization (0-1)",
        outpath_base=(outdir / "memory_mean_with_std_band").as_posix(),
        color=COLORS["memory"],
        force01=True,
        use_band=True,
    )

    # =========================
    # COMBINED & SUPPORTING
    # =========================

    plot_simple(
        df,
        ["cpu_mean", "memory_mean"],
        "Utilization Mean",
        (outdir / "cpu_memory_mean").as_posix(),
        zoom_y=True,
    )

    plot_simple(
        df,
        ["cpu_mean", "memory_mean"],
        "Utilization Mean (0-1)",
        (outdir / "cpu_memory_mean_fullscale").as_posix(),
        force01=True,
    )

    plot_simple(
        df,
        ["cpu_std", "memory_std"],
        "Standard Deviation",
        (outdir / "cpu_memory_std").as_posix(),
    )

    if req_mean_col:
        plot_dual(
            df,
            ["cpu_mean", "memory_mean"],
            req_mean_col,
            "CPU/Memory Mean",
            "Request Rate Mean",
            (outdir / "means_vs_request_rate_mean").as_posix(),
            zoom_left=True,
        )

        plot_dual(
            df,
            ["cpu_mean", "memory_mean"],
            req_mean_col,
            "CPU/Memory Mean (0-1)",
            "Request Rate Mean",
            (outdir / "means_vs_request_rate_mean_fullscale").as_posix(),
            force01_left=True,
        )

    if req_std_col:
        plot_dual(
            df,
            ["cpu_std", "memory_std"],
            req_std_col,
            "CPU/Memory Std",
            "Request Rate Std",
            (outdir / "stds_vs_request_rate_std").as_posix(),
        )

    plot_scatter_cpu_vs_req(
        df,
        req_mean_col,
        (outdir / "scatter_cpu_vs_request_rate_mean").as_posix(),
    )

    # Thesis-friendly alternatives to a dense 3x2 subfigure page.
    if req_mean_col:
        plot_paper_summary(
            df,
            [
                {
                    "col": "cpu_mean",
                    "title": "(a) Average CPU utilization",
                    "ylabel": "Utilization",
                    "color": COLORS["cpu"],
                },
                {
                    "col": "memory_mean",
                    "title": "(b) Average memory utilization",
                    "ylabel": "Utilization",
                    "color": COLORS["memory"],
                },
                {
                    "col": req_mean_col,
                    "title": "(c) Average request rate",
                    "ylabel": "Requests/min",
                    "color": COLORS["request"],
                    "log_y": True,
                },
            ],
            (outdir / "paper_mean_summary").as_posix(),
        )

    if req_std_col:
        plot_paper_summary(
            df,
            [
                {
                    "col": "cpu_std",
                    "title": "(a) Standard deviation of CPU utilization",
                    "ylabel": "Std.",
                    "color": COLORS["cpu"],
                },
                {
                    "col": "memory_std",
                    "title": "(b) Standard deviation of memory utilization",
                    "ylabel": "Std.",
                    "color": COLORS["memory"],
                },
                {
                    "col": req_std_col,
                    "title": "(c) Standard deviation of request rate",
                    "ylabel": "Requests/min",
                    "color": COLORS["request"],
                    "log_y": True,
                },
            ],
            (outdir / "paper_std_summary").as_posix(),
        )

    print(f"✅ All plots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()