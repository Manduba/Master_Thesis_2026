#!/usr/bin/env python3
"""
FCDM Decision Engine

Reads all configuration from decision.yaml (same directory as this file).
No hardcoded values here — edit decision.yaml to change run behaviour.

Modes (set in decision.yaml):
  static — reads a local forecast CSV, no SSH/SCP
  live   — SSHes to remote VM, runs inference, SCPs forecast back
"""

import csv
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from cost_module import get_hourly_price, refresh_prices


# ================================================================
# LOAD CONFIG
# ================================================================

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "decision.yaml"

with open(_CONFIG_PATH) as _f:
    _C = yaml.safe_load(_f)

# run behaviour
MODE                = _C["mode"]
DRY_RUN             = _C["dry_run"]
CHECK_EVERY_SECONDS = _C["check_every_seconds"]

# openstack CLI
OS_CLOUD    = _C["os_cloud"]
CLOUDA_MAIN = _C["clouda_main"]
CLOUDB_MAIN = _C["cloudb_main"]


# pricing tuples
AWS_PROVIDER = (
    _C["aws_provider"]["provider"],
    _C["aws_provider"]["region"],
    _C["aws_provider"]["instance_type"],
)
OS_PROVIDER = (
    _C["openstack_provider"]["provider"],
    _C["openstack_provider"]["region"],
    _C["openstack_provider"]["instance_type"],
)

# paths — all relative to the directory this file lives in
_ROOT  = Path(__file__).resolve().parent
_PATHS = _C["paths"]

STATIC_FORECAST_CSV = _ROOT / _PATHS["static_forecast"]
STATIC_LOG_CSV      = _ROOT / _PATHS["static_log"]
LIVE_FORECAST_CSV   = _ROOT / _PATHS["live_forecast"]
LIVE_LOG_CSV        = _ROOT / _PATHS["live_log"]

# remote VM (live mode only)
_R                  = _C["remote"]
VM_HOST             = _R["host"]
VM_USER             = _R["user"]
VM_PROJECT_DIR      = _R["project_dir"]
REMOTE_INFER_SCRIPT = _R["infer_script"]
REMOTE_FORECAST     = _R["forecast_out"]


# ================================================================
# ACTIVE PATHS
# ================================================================

def _active_paths() -> tuple[Path, Path]:
    if MODE == "live":
        return LIVE_FORECAST_CSV, LIVE_LOG_CSV
    return STATIC_FORECAST_CSV, STATIC_LOG_CSV


# ================================================================
# DECISION LOG
# ================================================================

def _init_decision_log(log_csv: Path) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    if log_csv.exists():
        return
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "ts_iso",
            "forecast_cpu_max",
            "forecast_mem_max",
            "forecast_rr_max",
            "required_units",
            "aws_price_per_vm_usd_h",
            "openstack_price_per_vm_usd_h",
            "aws_total_usd_h",
            "openstack_total_usd_h",
            "chosen_provider",
            "chosen_name",
            "clouda_status",
            "cloudb_status",
        ])


def _append_decision_log(
    log_csv: Path,
    cpu_f: float, mem_f: float, rr_f: float,
    required_units: int,
    aws_p: float, os_p: float,
    aws_total: float, os_total: float,
    choice: str,
    a_st: str, b_st: str,
) -> None:
    chosen_name = "CloudA-openstack" if choice == "A" else "CloudB-aws"
    with open(log_csv, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            f"{cpu_f:.6f}", f"{mem_f:.6f}", f"{rr_f:.6f}",
            str(required_units),
            f"{aws_p:.6f}", f"{os_p:.6f}",
            f"{aws_total:.6f}", f"{os_total:.6f}",
            choice, chosen_name,
            a_st, b_st,
        ])


# ================================================================
# OPENSTACK CLI HELPERS
# ================================================================

def _os_cmd(args: list[str]) -> str:
    cmd = ["openstack", "--os-cloud", OS_CLOUD] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip())
    return (p.stdout or "").strip()


def server_status(name: str) -> str:
    return _os_cmd(["server", "show", name, "-f", "value", "-c", "status"])


def server_start(name: str) -> None:
    print(f"Starting {name}")
    if DRY_RUN:
        return
    try:
        _os_cmd(["server", "start", name])
    except RuntimeError as e:
        msg = str(e)
        if any(k in msg for k in ("task_state", "powering-on", "is not ready")):
            print(f"NOTE: {name} is already starting / not ready. Skipping.")
            return
        raise


def server_stop(name: str) -> None:
    print(f"Stopping {name}")
    if DRY_RUN:
        return
    try:
        _os_cmd(["server", "stop", name])
    except RuntimeError as e:
        msg = str(e)
        if any(k in msg for k in ("task_state", "powering-off")):
            print(f"NOTE: {name} is already stopping. Skipping.")
            return
        raise


def wait_for_status(name: str, desired: str = "ACTIVE", timeout_sec: int = 300) -> bool:
    if DRY_RUN:
        return True
    start = time.time()
    while True:
        st = server_status(name)
        if st == desired:
            return True
        if st == "ERROR":
            raise RuntimeError(f"{name} entered ERROR state while waiting for {desired}")
        if time.time() - start > timeout_sec:
            print(f"WARNING: Timeout waiting for {name} → {desired} (current={st})")
            return False
        time.sleep(5)


# ================================================================
# FORECAST HELPERS
# ================================================================

def _fetch_live_forecast() -> None:
    """SSH to remote VM, run inference, SCP the CSV back. Live mode only."""
    LIVE_FORECAST_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"Triggering remote inference on {VM_HOST} ...")
    ssh_cmd = [
        "ssh", "-o", "BatchMode=yes", f"{VM_USER}@{VM_HOST}",
        f"cd {VM_PROJECT_DIR} && source .venv/bin/activate && python {REMOTE_INFER_SCRIPT}",
    ]
    p = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Remote inference failed:\n{p.stderr or p.stdout}")

    print("Copying forecast CSV from remote VM ...")
    scp_cmd = [
        "scp", "-o", "BatchMode=yes",
        f"{VM_USER}@{VM_HOST}:{REMOTE_FORECAST}",
        str(LIVE_FORECAST_CSV),
    ]
    p2 = subprocess.run(scp_cmd, capture_output=True, text=True)
    if p2.returncode != 0:
        raise RuntimeError(f"SCP failed:\n{p2.stderr or p2.stdout}")

    print(f"Forecast copied to: {LIVE_FORECAST_CSV}\n")


def read_forecast_summary(csv_path: Path) -> tuple[float, float, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Forecast CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return (
        float(df["cpu_pred"].max()),
        float(df["memory_pred"].max()),
        float(df["request_rate_pred"].max()),
    )


def required_units_from_forecast(cpu_f: float, mem_f: float) -> int:
    return max(1, math.ceil(max(cpu_f, mem_f)))


# ================================================================
# DECISION LOGIC
# ================================================================

def _decide(os_total: float, aws_total: float, a_st: str, b_st: str) -> tuple[str, str]:
    if os_total < aws_total:
        return "A", "OpenStack cheaper → CloudA-openstack"
    if aws_total < os_total:
        return "B", "AWS cheaper → CloudB-aws"

    # equal cost — keep whichever is already active to avoid churn
    if a_st == "ACTIVE" and b_st != "ACTIVE":
        return "A", "Equal cost → keep active CloudA-openstack"
    if b_st == "ACTIVE" and a_st != "ACTIVE":
        return "B", "Equal cost → keep active CloudB-aws"
    return "A", "Equal cost, state unclear → default CloudA-openstack"


# ================================================================
# ONE CYCLE
# ================================================================

def one_cycle(forecast_csv: Path, log_csv: Path) -> None:
    # 1. get forecast
    if MODE == "live":
        _fetch_live_forecast()

    cpu_f, mem_f, rr_f = read_forecast_summary(forecast_csv)
    required_units = required_units_from_forecast(cpu_f, mem_f)

    # 2. prices
    refresh_prices()
    aws_p     = get_hourly_price(*AWS_PROVIDER)
    os_p      = get_hourly_price(*OS_PROVIDER)
    aws_total = aws_p * required_units
    os_total  = os_p  * required_units

    mode_label = "static CSV" if MODE == "static" else "live remote inference"
    print(f"\n=== FCDM Decision Cycle ({mode_label}) ===")
    print(f"Forecast max: CPU={cpu_f:.4f}  MEM={mem_f:.4f}  RR={rr_f:.2f}")
    print(f"Required VM units: {required_units}")
    print(f"AWS       {aws_p:.4f} USD/h/vm  →  total {aws_total:.4f} USD/h")
    print(f"OpenStack {os_p:.4f} USD/h/vm  →  total {os_total:.4f} USD/h")

    # 3. current VM states
    a_st = server_status(CLOUDA_MAIN)
    b_st = server_status(CLOUDB_MAIN)
    print(f"Current: {CLOUDA_MAIN}={a_st}  {CLOUDB_MAIN}={b_st}")

    # 4. decide
    choice, reason = _decide(os_total, aws_total, a_st, b_st)
    print(f"Decision: {reason}")

    # 5. actuate
    if choice == "A":
        if a_st != "ACTIVE":
            server_start(CLOUDA_MAIN)
            wait_for_status(CLOUDA_MAIN, "ACTIVE")
        if b_st == "ACTIVE":
            server_stop(CLOUDB_MAIN)
            wait_for_status(CLOUDB_MAIN, "SHUTOFF")
    else:
        if b_st != "ACTIVE":
            server_start(CLOUDB_MAIN)
            wait_for_status(CLOUDB_MAIN, "ACTIVE")
        if a_st == "ACTIVE":
            server_stop(CLOUDA_MAIN)
            wait_for_status(CLOUDA_MAIN, "SHUTOFF")

    # 6. log
    a_st_after = server_status(CLOUDA_MAIN)
    b_st_after = server_status(CLOUDB_MAIN)
    _append_decision_log(
        log_csv,
        cpu_f, mem_f, rr_f, required_units,
        aws_p, os_p, aws_total, os_total,
        choice, a_st_after, b_st_after,
    )


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    if MODE not in ("static", "live"):
        raise ValueError(f"Unknown mode={MODE!r} in config.yaml. Must be 'static' or 'live'.")

    forecast_csv, log_csv = _active_paths()

    print(f"FCDM controller started  [mode={MODE}  dry_run={DRY_RUN}]")
    print(f"Config       : {_CONFIG_PATH}")
    print(f"Forecast CSV : {forecast_csv}")
    print(f"Decision log : {log_csv}")

    _init_decision_log(log_csv)

    while True:
        try:
            one_cycle(forecast_csv, log_csv)
        except KeyboardInterrupt:
            print("\nFCDM stopped manually.")
            return
        except Exception as e:
            print(f"ERROR: {e}")

        print(f"\nSleeping {CHECK_EVERY_SECONDS}s ...\n")
        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()