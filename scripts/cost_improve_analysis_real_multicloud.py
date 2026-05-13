#!/usr/bin/env python3
"""
Calculate cost benefit from the complete real multi-cloud decision log.

Default CSV:
    output/decision_log_liveMode_real_aws.csv

Expected CSV columns:
- ts_iso
- aws_total_usd_h
- openstack_total_usd_h
- chosen_provider

Where:
- chosen_provider = "A" means OpenStack selected
- chosen_provider = "B" means AWS selected

This script computes:
- interval durations between consecutive rows
- MWPA/FCDM switching cost
- always-AWS baseline cost
- always-OpenStack baseline cost
- savings vs both baselines
- weekly/monthly extrapolated savings
- percentage savings

Usage:
    python scripts/cost_saving_real_multicloud.py

Optional:
    python scripts/cost_saving_real_multicloud.py --csv output/decision_log_liveMode_real_aws.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class IntervalResult:
    start_time: str
    end_time: str
    duration_hours: float
    chosen_provider: str
    selected_hourly_cost: float
    aws_hourly_cost: float
    openstack_hourly_cost: float
    interval_cost_mwpa: float
    interval_cost_aws: float
    interval_cost_openstack: float


def parse_args() -> argparse.Namespace:
    default_csv = Path(__file__).resolve().parents[1] / "output" / "decision_log_liveMode_real_aws.csv"

    parser = argparse.ArgumentParser(
        description="Calculate cost benefit from a real multi-cloud decision log."
    )
    parser.add_argument(
        "--csv",
        default=str(default_csv),
        help=f"Path to the complete decision log CSV (default: {default_csv})",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=4,
        help="Decimal places for printed values (default: 4).",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        "ts_iso",
        "aws_total_usd_h",
        "openstack_total_usd_h",
        "chosen_provider",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_columns(df)

    df = df.copy()
    df["ts_iso"] = pd.to_datetime(df["ts_iso"], errors="raise")
    df = df.sort_values("ts_iso").reset_index(drop=True)

    df["aws_total_usd_h"] = pd.to_numeric(df["aws_total_usd_h"], errors="raise")
    df["openstack_total_usd_h"] = pd.to_numeric(df["openstack_total_usd_h"], errors="raise")
    df["chosen_provider"] = df["chosen_provider"].astype(str).str.strip()

    return df


def build_intervals(df: pd.DataFrame) -> List[IntervalResult]:
    results: List[IntervalResult] = []

    for i in range(len(df) - 1):
        current = df.iloc[i]
        nxt = df.iloc[i + 1]

        delta_hours = (nxt["ts_iso"] - current["ts_iso"]).total_seconds() / 3600.0

        if delta_hours < 0:
            raise ValueError("Negative interval found. Check timestamp ordering.")

        chosen = current["chosen_provider"]
        if chosen == "A":
            selected_cost = float(current["openstack_total_usd_h"])
            chosen_label = "OpenStack"
        elif chosen == "B":
            selected_cost = float(current["aws_total_usd_h"])
            chosen_label = "AWS"
        else:
            raise ValueError(
                f"Unexpected chosen_provider value at row {i}: {chosen!r}. Expected 'A' or 'B'."
            )

        aws_cost = float(current["aws_total_usd_h"])
        op_cost = float(current["openstack_total_usd_h"])

        results.append(
            IntervalResult(
                start_time=current["ts_iso"].isoformat(sep=" ", timespec="seconds"),
                end_time=nxt["ts_iso"].isoformat(sep=" ", timespec="seconds"),
                duration_hours=delta_hours,
                chosen_provider=chosen_label,
                selected_hourly_cost=selected_cost,
                aws_hourly_cost=aws_cost,
                openstack_hourly_cost=op_cost,
                interval_cost_mwpa=selected_cost * delta_hours,
                interval_cost_aws=aws_cost * delta_hours,
                interval_cost_openstack=op_cost * delta_hours,
            )
        )

    return results


def summarize(intervals: List[IntervalResult]) -> dict:
    total_hours = sum(x.duration_hours for x in intervals)
    total_mwpa = sum(x.interval_cost_mwpa for x in intervals)
    total_aws = sum(x.interval_cost_aws for x in intervals)
    total_op = sum(x.interval_cost_openstack for x in intervals)

    savings_vs_aws = total_aws - total_mwpa
    savings_vs_op = total_op - total_mwpa

    week_hours = 24 * 7
    month_hours = 24 * 30

    weekly_savings_vs_aws = savings_vs_aws * (week_hours / total_hours) if total_hours else 0.0
    weekly_savings_vs_op = savings_vs_op * (week_hours / total_hours) if total_hours else 0.0

    monthly_savings_vs_aws = savings_vs_aws * (month_hours / total_hours) if total_hours else 0.0
    monthly_savings_vs_op = savings_vs_op * (month_hours / total_hours) if total_hours else 0.0

    pct_vs_aws = (savings_vs_aws / total_aws * 100.0) if total_aws else 0.0
    pct_vs_op = (savings_vs_op / total_op * 100.0) if total_op else 0.0

    return {
        "observed_hours": total_hours,
        "mwpa_cost": total_mwpa,
        "always_aws_cost": total_aws,
        "always_openstack_cost": total_op,
        "savings_vs_aws": savings_vs_aws,
        "savings_vs_openstack": savings_vs_op,
        "weekly_savings_vs_aws": weekly_savings_vs_aws,
        "weekly_savings_vs_openstack": weekly_savings_vs_op,
        "monthly_savings_vs_aws": monthly_savings_vs_aws,
        "monthly_savings_vs_openstack": monthly_savings_vs_op,
        "pct_saving_vs_aws": pct_vs_aws,
        "pct_saving_vs_openstack": pct_vs_op,
    }


def print_interval_table(intervals: List[IntervalResult], digits: int) -> None:
    print("\nINTERVAL-LEVEL VIEW")
    print("-" * 120)
    header = (
        f"{'Start':19}  {'End':19}  {'Hours':>8}  {'Chosen':>10}  "
        f"{'Selected USD/h':>14}  {'MWPA cost':>12}  {'AWS cost':>12}  {'OP cost':>12}"
    )
    print(header)
    print("-" * 120)

    for x in intervals:
        print(
            f"{x.start_time:19}  {x.end_time:19}  "
            f"{x.duration_hours:8.{digits}f}  {x.chosen_provider:>10}  "
            f"{x.selected_hourly_cost:14.{digits}f}  "
            f"{x.interval_cost_mwpa:12.{digits}f}  "
            f"{x.interval_cost_aws:12.{digits}f}  "
            f"{x.interval_cost_openstack:12.{digits}f}"
        )


def print_summary(summary: dict, digits: int) -> None:
    print("\nSUMMARY")
    print("-" * 60)
    for key, label in [
        ("observed_hours", "Observed duration (hours)"),
        ("mwpa_cost", "MWPA/FCDM switching cost"),
        ("always_aws_cost", "Always AWS cost"),
        ("always_openstack_cost", "Always OpenStack cost"),
        ("savings_vs_aws", "Savings vs always AWS"),
        ("savings_vs_openstack", "Savings vs always OpenStack"),
        ("weekly_savings_vs_aws", "Estimated weekly savings vs AWS"),
        ("weekly_savings_vs_openstack", "Estimated weekly savings vs OpenStack"),
        ("monthly_savings_vs_aws", "Estimated monthly savings vs AWS"),
        ("monthly_savings_vs_openstack", "Estimated monthly savings vs OpenStack"),
        ("pct_saving_vs_aws", "Percentage saving vs AWS"),
        ("pct_saving_vs_openstack", "Percentage saving vs OpenStack"),
    ]:
        value = summary[key]
        suffix = "%" if "pct_" in key else ""
        print(f"{label:40}: {value:.{digits}f}{suffix}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)

    print(f"Using CSV: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Either place the file there or run with:\n"
            f"python scripts/cost_saving_real_multicloud.py --csv <your_csv_path>"
        )

    df = load_and_prepare(csv_path)
    intervals = build_intervals(df)
    summary = summarize(intervals)

    print_interval_table(intervals, args.round)
    print_summary(summary, args.round)


if __name__ == "__main__":
    main()