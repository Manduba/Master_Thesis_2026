import pandas as pd
from pathlib import Path

# ⬅️ your file
path = Path("msresource_0.csv")

# Read a small sample to see columns (and drop any unnamed index column)
sample = pd.read_csv(path, nrows=20)
if any(c.startswith("Unnamed") or c=="" for c in sample.columns):
    sample = pd.read_csv(path, nrows=20, index_col=0)  # drop first index-like column

print("Columns:", list(sample.columns))
sample.head(5)

# Small sample sanity
print("Sample dtypes:\n", sample.dtypes)
print("\nNon-null ratio (sample):\n", 1 - sample.isna().mean())

# Grab a few non-zero timestamps
ts = sample["timestamp"].dropna()
print("First few timestamps:", ts.head(5).tolist())

# Guess unit: show conversions for one example value
ex = ts.iloc[0]
print("\nTry interpreting the same number as seconds vs milliseconds:")
try:
    print("as seconds    →", pd.to_datetime(ex, unit="s",  utc=True))
    print("as milliseconds→", pd.to_datetime(ex, unit="ms", utc=True))
except Exception as e:
    print("Conversion error:", e)

# Use a slightly larger slice to measure gaps
gaps = (pd.read_csv(path, nrows=50_000)
          .pipe(lambda df: df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore"))
          ["timestamp"]
          .dropna()
          .astype("int64")
          .sort_values()
          .drop_duplicates()
          .diff()
          .dropna())

print("Most common gap values (raw units):")
print(gaps.value_counts().head(5))

df_small = pd.read_csv(path, nrows=100_000)
df_small = df_small.drop(columns=[c for c in df_small.columns if c.startswith("Unnamed")], errors="ignore")

for col in ["instance_cpu_usage", "instance_memory_usage"]:
    s = pd.to_numeric(df_small[col], errors="coerce")
    print(f"\n{col}: min={s.min()}, max={s.max()}, p1={s.quantile(0.01)}, p99={s.quantile(0.99)}")

    # Quick sanity flag
    out_of_bounds = ((s < 0) | (s > 1)).mean()
    print(f"  Fraction outside [0,1]: {out_of_bounds:.4f}")

df_small = df_small.dropna(subset=["msinstanceid","timestamp"])
top_inst = (df_small["msinstanceid"].value_counts().index[:1])[0]
one = df_small[df_small["msinstanceid"]==top_inst].sort_values("timestamp")

print("Example instance:", top_inst)
print("First 10 timestamps for that instance:")
print(one["timestamp"].head(10).tolist())

print("\nUnique counts in the sample (~100k rows):")
for col in ["msname","msinstanceid","nodeid"]:
    if col in df_small.columns:
        print(f"  {col}: {df_small[col].nunique()}")

#timespan check
import pandas as pd

ts = pd.read_csv("msresource_0.csv", usecols=["timestamp"])["timestamp"].astype("int64")
span_ms = ts.max() - ts.min()
span_hr = span_ms / 1000 / 3600
print(f"Span ≈ {span_hr:.2f} hours")  # should be ~0.99

gaps_s = ts.sort_values().drop_duplicates().diff().dropna() / 1000
print("Common gaps (s):")
print(gaps_s.round().value_counts().head(5))

