import sys
import argparse
from pathlib import Path
import pandas as pd

try:
    import yaml
except ImportError:
    print("Please: pip install pyyaml")
    sys.exit(1)


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def run_job(job_name: str, root_cfg: dict):
    if job_name not in root_cfg:
        print(f"[ERROR] Job '{job_name}' not found in YAML. Available: {list(root_cfg.keys())}")
        sys.exit(1)

    cfg = root_cfg[job_name]

    inputs       = cfg["inputs"]
    output       = cfg["output"]
    sort_by      = cfg.get("sort_by", "ts")
    dedup_on     = cfg.get("deduplicate_on", [sort_by])
    keep         = cfg.get("keep", "last")
    keep_cols    = cfg.get("keep_cols")  # optional

    dfs = []
    for f in inputs:
        p = Path(f)
        if not p.exists():
            print(f"[WARN] Missing file: {f} (skipping)")
            continue
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"[INFO][{job_name}] read {f}: rows={len(df):,}")

    if not dfs:
        print(f"[ERROR][{job_name}] No input files found. Check 'inputs' in YAML.")
        sys.exit(1)

    big = pd.concat(dfs, ignore_index=True, sort=False)

        # -------------------------------------------------
    # Rename request-rate columns to final canonical names
    # -------------------------------------------------
    rename_map = {
        "total_request_rate_mean": "request_rate_mean",
        "total_request_rate_std": "request_rate_std",
    }

    big = big.rename(columns={k: v for k, v in rename_map.items() if k in big.columns})


    # Sort
    if sort_by in big.columns:
        big = big.sort_values(sort_by, kind="stable")
    else:
        print(f"[WARN][{job_name}] sort_by '{sort_by}' not found; skipping sort.")

    # De-duplicate
    missing_keys = [c for c in dedup_on if c not in big.columns]
    if missing_keys:
        print(f"[WARN][{job_name}] deduplicate_on keys missing: {missing_keys}; skipping de-dup.")
    else:
        big = big.drop_duplicates(subset=dedup_on, keep=keep)

    # Optional: enforce final column order / presence
    if keep_cols:
        for c in keep_cols:
            if c not in big.columns:
                big[c] = pd.NA
        big = big[keep_cols]

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    big.to_csv(output, index=False)
    print(f"[OK][{job_name}] wrote {output} rows={len(big):,}")
    if "ts" in big.columns and len(big):
        print(f"[INFO][{job_name}] ts_min={big['ts'].iloc[0]}  ts_max={big['ts'].iloc[-1]}")


def main():
    parser = argparse.ArgumentParser(description="Join multiple CSVs per YAML job")
    parser.add_argument("config", nargs="?", default="config/join_cluster_service_batch.yaml",
                        help="Path to YAML (default: config/join_cluster_service_batch.yaml)")
    parser.add_argument("--job", default=None,
                        help="YAML job key to run (e.g., 'join' or 'join_service'). "
                             "If omitted, defaults to 'join' (legacy).")
    parser.add_argument("--all", action="store_true",
                        help="Run ALL top-level jobs in the YAML (e.g., 'join' and 'join_service') in a sensible order.")
    args = parser.parse_args()

    root_cfg = load_cfg(args.config)
    if not root_cfg:
        print("[ERROR] YAML is empty.")
        sys.exit(1)

    if args.all:
        # Sensible order: run 'join' first if present, then 'join_service', then any other remaining jobs
        jobs = list(root_cfg.keys())
        ordered = []
        for preferred in ("join", "join_service"):
            if preferred in jobs:
                ordered.append(preferred)
                jobs.remove(preferred)
        ordered.extend(sorted(jobs))  # remaining jobs alphabetically
        print(f"[INFO] Running ALL jobs in order: {ordered}")
        for j in ordered:
            run_job(j, root_cfg)
        return

    # Backward-compatible single job
    job = args.job
    if job is None:
        if "join" in root_cfg:
            job = "join"
        else:
            # fallback to first key
            job = next(iter(root_cfg.keys()))
            print(f"[INFO] --job not provided; using first job '{job}'")

    print(f"[INFO] Running job: {job}")
    run_job(job, root_cfg)


if __name__ == "__main__":
    main()
