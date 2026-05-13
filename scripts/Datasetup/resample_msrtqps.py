import pandas as pd
import numpy as np
import yaml, glob, os, sys

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_output_folder(output_path):
    """Create output folder if it doesn't exist"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output folder: {output_dir}")
    return output_path

def process_chunk(df, cols, time_unit, bucket):
    """
    Standardize column names and bucket timestamps to minute windows.
    """
    df = df.rename(columns={
        cols["timestamp"]:   "timestamp",
        cols["msname"]:      "msname",
        cols["msinstanceid"]: "msinstanceid",
        cols["metric"]:      "metric",
        cols["value"]:       "value",
    })
    df["time_1min"] = pd.to_datetime(df["timestamp"], unit=time_unit).dt.floor(bucket)
    return df

def main(cfg_path="config/msrtqps.yaml", chunksize=500_000, keep_http_mq_rt=False):
    cfg = load_cfg(cfg_path)["msrtqps"]

    # --- Inputs ---
    paths = sorted(glob.glob(cfg["input_glob"]))
    if not paths:
        raise FileNotFoundError(f"No files match: {cfg['input_glob']}")
    print(f"Found {len(paths)} file(s): {paths}")

    cols = cfg["columns"]
    drop_index_col = cfg.get("drop_index_col")
    mcr_metrics = cfg["mcr_metrics"]
    rt_metrics  = cfg["rt_metrics"]
    keep_metrics = mcr_metrics + rt_metrics
    time_unit = cfg.get("time_unit", "ms")
    bucket    = cfg.get("bucket", "60s")

    # --- Stream read + per-instance-minute reduction ---
    agg_chunks = []
    for path in paths:
        print(f"\nProcessing {path}")
        for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
            if drop_index_col and drop_index_col in chunk.columns:
                chunk = chunk.drop(columns=[drop_index_col])

            # filter metrics we care about
            if "metric" not in chunk.columns:
                chunk = chunk.rename(columns={cols["metric"]: "metric"})
            chunk = chunk[chunk["metric"].isin(keep_metrics)]
            if chunk.empty:
                continue

            chunk = process_chunk(chunk, cols, time_unit, bucket)

            # mean within (msname, instance, minute, metric)
            inst_min = (
                chunk.groupby(["msname","msinstanceid","time_1min","metric"], observed=True)["value"]
                     .mean()
                     .reset_index()
            )
            agg_chunks.append(inst_min)
            print(f"  ⏳ Chunk {i+1}: {len(chunk):,} rows → {len(inst_min):,} inst-min rows")

    if not agg_chunks:
        raise RuntimeError("No data after filtering; check your metrics in YAML.")

    # --- Concatenate all chunk aggregates ---
    print("\nMerging all chunks ...")
    inst_min_all = pd.concat(agg_chunks, ignore_index=True)
    del agg_chunks

    # --- Pivot to wide: one row = (service, instance, minute), columns = metrics ---
    inst_wide = (
        inst_min_all.pivot_table(
            index=["msname","msinstanceid","time_1min"],
            columns="metric", values="value", aggfunc="first"
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # --- ADD: total request rate per INSTANCE per minute ---
    mcr_cols = ["consumerRPC_MCR","providerRPC_MCR","consumerMQ_MCR","HTTP_MCR"]
    inst_wide["total_request_rate_inst"] = inst_wide[mcr_cols].sum(axis=1, skipna=True)

    # ensure columns for any missing metric names exist
    for m in keep_metrics:
        if m not in inst_wide.columns:
            inst_wide[m] = np.nan  # fill missing metrics with NaN

    # --- Service-level aggregation (per minute) ---
    agg_dict = {
        "consumerRPC_MCR": ("consumerRPC_MCR","sum"),
        "providerRPC_MCR": ("providerRPC_MCR","sum"),
        "consumerMQ_MCR":  ("consumerMQ_MCR","sum"),
        "HTTP_MCR":        ("HTTP_MCR","sum"),
        "consumerRPC_RT":  ("consumerRPC_RT","mean"),
        "providerRPC_RT":  ("providerRPC_RT","mean"),
        "active_instances":("msinstanceid","nunique"),
        "total_request_rate": ("total_request_rate_inst","sum"),
        "total_request_rate_mean": ("total_request_rate_inst","mean"),
        "total_request_rate_std": ("total_request_rate_inst", lambda x: x.std(ddof=0)),
    }

    if keep_http_mq_rt:
        if "HTTP_RT" in inst_wide.columns:
            agg_dict["HTTP_RT"] = ("HTTP_RT","mean")
        if "consumerMQ_RT" in inst_wide.columns:
            agg_dict["consumerMQ_RT"] = ("consumerMQ_RT","mean")

    svc_min = (
        inst_wide.groupby(["msname","time_1min"], observed=True)
                 .agg(**agg_dict)
                 .reset_index()
    )

    # --- Derived metrics: latency_mean from RPC RTs ---
    svc_min["latency_mean"] = svc_min[["consumerRPC_RT","providerRPC_RT"]].mean(axis=1, skipna=True)

    # --- Total + mean + std for request rates across FOUR MCR metrics ---
    #mcr_cols = ["consumerRPC_MCR","providerRPC_MCR","consumerMQ_MCR","HTTP_MCR"]

    # total_request_rate = sum of the four MCRs (ignoring NaN)
    #svc_min["total_request_rate"] = svc_min[mcr_cols].sum(axis=1, skipna=True)

    # request_mean = mean of those four MCRs (ignoring NaN)
    #svc_min["request_mean"] = svc_min[mcr_cols].mean(axis=1, skipna=True)

    # request_std = std of those four MCRs (population std, ddof=0, ignoring NaN)
    #svc_min["request_std"] = svc_min[mcr_cols].std(axis=1, ddof=0)

    # --- Human-readable minute label ---
    svc_min["ts"] = svc_min["time_1min"].dt.strftime("%H:%M:%S")

    # --- Final columns in your requested order ---
    base_cols = [
        "ts","msname",
        "consumerRPC_MCR","providerRPC_MCR","consumerMQ_MCR","HTTP_MCR",
        "latency_mean",
        "total_request_rate","total_request_rate_mean","total_request_rate_std",
        "active_instances",
    ]

    out = svc_min[base_cols]

    # --- Save ---
    out_path = cfg["output_csv"]
    out_path = ensure_output_folder(out_path)
    out.to_csv(out_path, index=False)

    print(f"\n✅ Saved: {out_path}")
    print(f"📊 Shape: {out.shape}")
    print(out.head(10).to_string(index=False))
    print("\nColumns saved:", list(out.columns))

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/msrtqps.yaml"
    main(cfg_path, keep_http_mq_rt=False)
