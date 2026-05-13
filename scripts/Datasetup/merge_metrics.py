import pandas as pd
from pathlib import Path
import sys

try:
    import yaml
except ImportError:
    print("Please: pip install pyyaml")
    sys.exit(1)

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df

def ensure_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    target_to_aliases = {}
    for alias, target in rename_map.items():
        target_to_aliases.setdefault(target, []).append(alias)
    for target, aliases in target_to_aliases.items():
        if target in df.columns: continue
        for alias in aliases:
            if alias in df.columns:
                df = df.rename(columns={alias: target})
                break
    return df

def main(cfg_path="config/merge.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    m = cfg["merge"]
    res_path  = Path(m["resource_csv"])
    qps_path  = Path(m["msrtqps_csv"])
    out_full  = Path(m["out_full"]) if m.get("out_full") else None   # optional
    out_final = Path(m["out_final"])
    join_keys = [k.lower() for k in m.get("join_keys", ["ts","msname"])]

    # load + normalize
    res = normalize_cols(pd.read_csv(res_path, low_memory=False))
    qps = normalize_cols(pd.read_csv(qps_path, low_memory=False))

    # align columns
    for df in (res, qps):
        df = df  # just clarity

    res = ensure_columns(res, {"time_str":"ts","time":"ts","time_1min":"ts"})
    qps = ensure_columns(qps, {"time_str":"ts","time":"ts","time_1min":"ts"})

    res = ensure_columns(res, {
        "cpu":"cpu_mean", "cpu_mean":"cpu_mean",
        "memory":"memory_mean", "memory_mean":"memory_mean",
        "cpu_stddev":"cpu_std", "cpu_stdev":"cpu_std",
        "mem_std":"memory_std", "memory_stdev":"memory_std",
        "active_instance":"active_instances",
    })
    qps = ensure_columns(qps, {
        "total_request_rate":"request_rate",
        "total_req_rate":"request_rate",
        "total_requests_per_second":"request_rate",
        "requests_rate":"request_rate",
        "request_mean":"request_rate_mean",
        "request_stddev":"request_rate_std",
        "request_stdev":"request_rate_std",
        "request_std":"request_rate_std",
        "active_instance":"active_instances",
    })

    # check join keys
    for k in join_keys:
        if k not in res.columns: raise KeyError(f"Join key '{k}' missing in resource: {res.columns}")
        if k not in qps.columns: raise KeyError(f"Join key '{k}' missing in msrtqps: {qps.columns}")

    # merge
    merged_full = res.merge(qps, on=join_keys, how="inner", suffixes=("_res","_qps"))

    # coalesce active_instances → single column (prefer non-zero on mismatch)
    def _find(df, cands):
        for c in cands:
            if c in df.columns: return c
        return None

    res_col = _find(merged_full, ["active_instances_res","active_instance_res"])
    qps_col = _find(merged_full, ["active_instances_qps","active_instance_qps"])
    if res_col and qps_col:
        a = merged_full[res_col].astype("Int64")
        b = merged_full[qps_col].astype("Int64")
        eq = (a == b) | (a.isna() & b.isna())
        prefer_b = (a.fillna(0) == 0) & (b.fillna(0) > 0)
        merged_full["active_instances"] = a.where(eq, a.where(~prefer_b, b))
        merged_full.drop(columns=[res_col, qps_col], inplace=True)
    else:
        lone = _find(merged_full, ["active_instances","active_instance"])
        if lone and lone != "active_instances":
            merged_full.rename(columns={lone:"active_instances"}, inplace=True)
    """
    # Calculate request_rate statistics
    if "request_rate" in merged_full.columns:
        # Calculate cluster-level statistics (mean and std across all services for each timestamp)
        cluster_stats = merged_full.groupby("ts")["request_rate"].agg(["mean", "std"]).reset_index()
        cluster_stats = cluster_stats.rename(columns={
            "mean": "request_rate_mean", 
            "std": "request_rate_std"
        })
        
        # Merge the statistics back to the main dataframe
        merged_full = merged_full.merge(cluster_stats, on="ts", how="left")
        
        # Fill NaN std values with 0 (when only one service has data at that timestamp)
        merged_full["request_rate_std"] = merged_full["request_rate_std"].fillna(0)
    """
    # write full (ONLY if configured)
    if out_full:
        out_full.parent.mkdir(parents=True, exist_ok=True)
        merged_full.to_csv(out_full, index=False)
        print(f"[OK] Wrote full merged file: {out_full}")

    # final required columns only
    final_cols = [
        "ts","msname",
        "cpu_mean","memory_mean",
        "cpu_std","memory_std",
        "total_request_rate",
        "total_request_rate_mean",
        "total_request_rate_std",
    ]

    def pick(col):
        if col in merged_full.columns: return col
        if col+"_res" in merged_full.columns: return col+"_res"
        if col+"_qps" in merged_full.columns: return col+"_qps"
        return None

    choose = {c: pick(c) for c in final_cols}
    final_df = merged_full[[v for v in choose.values() if v is not None]].copy()
    final_df = final_df.rename(columns={v:k for k,v in choose.items() if v is not None})

    out_final.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_final, index=False)
    print(f"[OK] Wrote final merged file: {out_final}")
    print(final_df.head())

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/merge.yaml"
    main(cfg)