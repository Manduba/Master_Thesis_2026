# resample_resource.py
import pandas as pd
import glob
from collections import defaultdict
from pathlib import Path
import os

# ---- settings (edit to taste) ----
INPUT_GLOB = "data/batch3/MSResource_*.csv"

OUTPUT_FOLDER = "output"
# 👇 NEW: unique filename so you don't mix with any older runs
OUTPUT_FILENAME = "batch3_out_resource_service_60s.csv"

CHUNKSIZE = 2_000_000   # adjust to your RAM (e.g., 500_000 .. 5_000_000)

# column names
TS_COL   = "timestamp"
NAME_COL = "msname"
INST_COL = "msinstanceid"
CPU_COL  = "instance_cpu_usage"
MEM_COL  = "instance_memory_usage"

TIME_UNIT = "ms"        # timestamps are in milliseconds
INTERVAL  = "60s"       # resample target
TIME_FMT  = "%H:%M:%S"  # output label

# 🔴 Keep clipping OFF to preserve true distribution
CLIP_01   = False

# 0=population std (std=0 if one instance); 1=sample std (NaN if one)
STD_DDOF  = 0

# ---- helper: running aggregator for Stage 1 (instance, minute) ----
# we'll accumulate sum and count so we can compute the mean at the end.
acc_cpu_sum = defaultdict(float)
acc_mem_sum = defaultdict(float)
acc_count   = defaultdict(int)

def ensure_output_folder():
    """Create output folder if it doesn't exist"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, OUTPUT_FOLDER)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"📁 Created output folder: {output_path}")
    return output_path

def get_output_filepath():
    """Get the full output file path"""
    ensure_output_folder()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, OUTPUT_FOLDER, OUTPUT_FILENAME)

def process_chunk(df: pd.DataFrame):
    # use only needed columns
    df = df[[TS_COL, NAME_COL, INST_COL, CPU_COL, MEM_COL]]

    # --- DIAGNOSTIC A: did raw values ever leave [0,1]?
    cpu_outside = ((df[CPU_COL] < 0) | (df[CPU_COL] > 1)).sum()
    mem_outside = ((df[MEM_COL] < 0) | (df[MEM_COL] > 1)).sum()
    if cpu_outside or mem_outside:
        print(f"[diag] Outside [0,1] in this chunk: CPU={cpu_outside}, MEM={mem_outside}")

    # optional clip (disabled)
    if CLIP_01:
        df[CPU_COL] = df[CPU_COL].clip(0, 1)
        df[MEM_COL] = df[MEM_COL].clip(0, 1)

    # convert to minute bucket
    t = pd.to_datetime(df[TS_COL], unit=TIME_UNIT)
    time_1min = t.dt.floor(INTERVAL)

    # make a small frame for grouping
    tmp = pd.DataFrame({
        NAME_COL: df[NAME_COL].values,
        INST_COL: df[INST_COL].values,
        "time_1min": time_1min.values,
        CPU_COL: df[CPU_COL].values,
        MEM_COL: df[MEM_COL].values,
    })

    # Stage 1 partials: per (msname, msinstanceid, minute) -> sum, count
    grp = tmp.groupby([NAME_COL, INST_COL, "time_1min"], observed=True)
    part = grp.agg(
        cpu_sum=(CPU_COL, "sum"),
        mem_sum=(MEM_COL, "sum"),
        n=("time_1min", "count"),
    ).reset_index()

    # Accumulate into global dicts (constant memory w.r.t. rows)
    for row in part.itertuples(index=False):
        key = (row.msname, row.msinstanceid, row.time_1min)
        acc_cpu_sum[key] += float(row.cpu_sum)
        acc_mem_sum[key] += float(row.mem_sum)
        acc_count[key]   += int(row.n)

def finalize_and_save():
    # Build instance-minute means from accumulators  (Stage 1 result)
    recs = []
    for key, n in acc_count.items():
        msname, msinstanceid, time_1min = key
        cpu_inst_min = acc_cpu_sum[key] / max(n, 1)
        mem_inst_min = acc_mem_sum[key] / max(n, 1)
        recs.append((msname, msinstanceid, time_1min, cpu_inst_min, mem_inst_min))

    inst_min = pd.DataFrame(
        recs,
        columns=["msname", "msinstanceid", "time_1min", "cpu_inst_min", "mem_inst_min"]
    )

    # Stage 2: service-level aggregation across instances (per minute)
    svc_min = (
        inst_min.groupby(["msname", "time_1min"], observed=True)
                .agg(
                    cpu_mean        = ("cpu_inst_min",  "mean"),
                    cpu_std         = ("cpu_inst_min",  lambda x: x.std(ddof=STD_DDOF)),
                    memory_mean     = ("mem_inst_min",  "mean"),
                    memory_std      = ("mem_inst_min",  lambda x: x.std(ddof=STD_DDOF)),
                    active_instances= ("msinstanceid",  "nunique"),
                )
                .reset_index()
                .sort_values(["msname", "time_1min"])
    )

    # --- DIAGNOSTIC B: see the aggregated ranges (this run)
    print("[diag] Aggregated ranges (service-minute):")
    for col in ["cpu_mean", "memory_mean", "cpu_std", "memory_std"]:
        print(f"  {col}: min={svc_min[col].min():.6f}  max={svc_min[col].max():.6f}")

    # add HH:MM:SS label
    svc_min["ts"] = pd.to_datetime(svc_min["time_1min"]).dt.strftime(TIME_FMT)

    # final column order
    out = svc_min[[
        "ts", "msname",
        "cpu_mean", "memory_mean",
        "cpu_std", "memory_std",
        "active_instances"
    ]].copy()

    output_file = get_output_filepath()
    out.to_csv(output_file, index=False)
    print(f"✅ Saved to: {output_file}")
    print(f"📊 Shape: {out.shape}")
    print(out.head())

def main():
    paths = sorted(glob.glob(INPUT_GLOB))
    if not paths:
        raise FileNotFoundError(f"No files match {INPUT_GLOB}")

    usecols = [TS_COL, NAME_COL, INST_COL, CPU_COL, MEM_COL]

    # stream each file in chunks
    for p in paths:
        print(f"Processing {p} ...")
        for chunk in pd.read_csv(p, usecols=usecols, chunksize=CHUNKSIZE):
            # Use smaller dtypes to save memory
            chunk[CPU_COL] = chunk[CPU_COL].astype("float32")
            chunk[MEM_COL] = chunk[MEM_COL].astype("float32")
            # IDs as category to reduce memory
            chunk[NAME_COL] = chunk[NAME_COL].astype("category")
            chunk[INST_COL] = chunk[INST_COL].astype("category")
            process_chunk(chunk)

    finalize_and_save()

if __name__ == "__main__":
    main()
