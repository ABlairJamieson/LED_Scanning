#!/usr/bin/env python3
import os
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

WATCH_DIR = "."
CHECK_INTERVAL = 10   # seconds between scans
ANALYZER = "waveform_analysis.py"
MAX_PARALLEL = 8      # up to 8 analyses at once


def get_dat_files():
    """List all .dat files that match wave_*.dat"""
    return sorted(
        f for f in os.listdir(WATCH_DIR)
        if f.startswith("wave_") and f.endswith(".dat")
    )


def analysis_dir_for(fname):
    """Determine output directory name for a given .dat file"""
    base = fname[:-4]  # strip .dat
    return f"WaveformAnalysis_{base}"


def run_analysis(dat):
    """Run the waveform analysis subprocess for one file"""
    outdir = analysis_dir_for(dat)
    cmd = ["python3", ANALYZER, dat]
    print(f"\n=== Starting analysis: {dat}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✔ Finished {dat}")
        return (dat, True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR running analysis on {dat}: {e}")
        return (dat, False)


def main():
    print(f"Watching directory: {os.path.abspath(WATCH_DIR)}")
    print(f"Every {CHECK_INTERVAL} seconds")
    print(f"Will run up to {MAX_PARALLEL} analyses in parallel.\n")

    running = set()  # filenames currently being processed

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {}

        while True:
            dat_files = get_dat_files()

            # Queue new analyses
            for dat in dat_files:
                outdir = analysis_dir_for(dat)
                if dat not in running and not os.path.exists(outdir):
                    print(f"\n→ New file detected: {dat}")
                    fut = pool.submit(run_analysis, dat)
                    futures[fut] = dat
                    running.add(dat)

            # Check for completed jobs
            done_futures = [f for f in futures if f.done()]
            for f in done_futures:
                dat = futures.pop(f)
                running.remove(dat)
                try:
                    _, ok = f.result()
                    if ok:
                        print(f"✅ Completed: {dat}")
                    else:
                        print(f"⚠ Failed: {dat}")
                except Exception as e:
                    print(f"⚠ Exception while analyzing {dat}: {e}")

            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
