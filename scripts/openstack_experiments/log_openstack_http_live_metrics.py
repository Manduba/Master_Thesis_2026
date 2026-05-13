import csv
import time
from datetime import datetime
import psutil
import subprocess
from pathlib import Path

OUT_PATH = "data/openstack_metrics_http_75min_live.csv"
SLEEP_SECONDS = 60
ACCESS_LOG = "/var/log/nginx/access.log"

def count_log_lines():
    out = subprocess.check_output(["wc", "-l", ACCESS_LOG], text=True).strip()
    return int(out.split()[0])

def main():
    Path("data").mkdir(parents=True, exist_ok=True)

    psutil.cpu_percent(interval=None)
    prev_lines = count_log_lines()
    row_count = 0

    out_file = Path(OUT_PATH)
    file_empty = (not out_file.exists()) or (out_file.stat().st_size == 0)

    try:
        with open(OUT_PATH, "a", newline="") as f:
            w = csv.writer(f)

            if file_empty:
                w.writerow(["ts", "cpu_mean", "memory_mean", "request_rate_mean"])
                f.flush()

            while True:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                cpu_frac = psutil.cpu_percent(interval=1) / 100.0
                mem_frac = psutil.virtual_memory().percent / 100.0

                curr_lines = count_log_lines()
                req_last_min = max(0, curr_lines - prev_lines)
                prev_lines = curr_lines

                rr = req_last_min / 60.0

                w.writerow([ts, f"{cpu_frac:.6f}", f"{mem_frac:.6f}", f"{rr:.6f}"])
                f.flush()

                row_count += 1
                print(f"[{row_count:04d}] ts={ts} cpu={cpu_frac:.3f} mem={mem_frac:.3f} rr={rr:.2f}")

                time.sleep(max(0, SLEEP_SECONDS - 1))

    except KeyboardInterrupt:
        print(f"\nStopped by user. Appended rows to: {OUT_PATH}")

if __name__ == "__main__":
    main()