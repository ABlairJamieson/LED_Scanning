#!/usr/bin/env python3
# coding: utf-8
import struct
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import re
import glob
import gc

# ==========================================================
# Config
# ==========================================================
FULL_SCALE_V = 2.0
N_BITS = 12
SAMPLE_RATE_HZ = 250e6
SAMPLE_PERIOD_NS = 1e9 / SAMPLE_RATE_HZ  # 4 ns

PED_WINDOW_NS = 100.0
INTEG_WINDOW = (110.0, 160.0)
R_OHM = 50.0

TT_MOD = 2**31
BATCH_SIZE = 16384   # 1024 * 16 events per batch

# ==========================================================
# Geometry helpers
# ==========================================================
def sph_to_cart(theta_deg, phi_deg):
    """
    Convert (theta, phi) degrees to Cartesian unit vector.

    theta: polar angle from +z axis
    phi:   azimuth angle in degrees
    """
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = np.cos(th)
    return np.array([x, y, z], dtype=np.float32)


def cart_to_sph(v):
    """Inverse of sph_to_cart: return (theta_deg, phi_deg)."""
    x, y, z = v
    r = np.sqrt(x*x + y*y + z*z)
    if r == 0:
        return (np.nan, np.nan)
    th = np.arccos(z / r)
    ph = np.arctan2(y, x)
    return (np.rad2deg(th), np.rad2deg(ph))


# Rotation: gantry frame → LED frame
R_g_to_led = np.array([
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
    [1.0, 0.0,  0.0],
], dtype=np.float32)


# ==========================================================
# CAEN iterator
# ==========================================================
def iter_wavedump_dt5720_headered(filename):
    with open(filename, "rb") as f:
        while True:
            header = f.read(24)
            if len(header) < 24:
                return

            try:
                event_size, board_id, pattern, chmask, evnum, timetag = struct.unpack("6I", header)
            except struct.error:
                print("WARNING: malformed header → stopping")
                return

            payload_size = event_size - 24
            if payload_size <= 0:
                print("WARNING: invalid event_size → skipping")
                continue

            wf_bytes = f.read(payload_size)
            if len(wf_bytes) < payload_size:
                print("WARNING: truncated event → stopping")
                return

            if len(wf_bytes) % 2 != 0:
                print("WARNING: odd waveform length → skipping")
                continue

            wf = np.frombuffer(wf_bytes, dtype=np.int16)
            volts = (wf.astype(np.float32) / (2**(N_BITS-1))) * (FULL_SCALE_V/2)
            t_ns = np.arange(len(volts), dtype=np.float32) * SAMPLE_PERIOD_NS

            yield (volts, t_ns, timetag)


# ==========================================================
# FAST trapezoid for batch
# ==========================================================
def trapz_batch(Y, T):
    """
    Vectorized trapezoidal integration over axis=1.
    Y, T shape = (B, M)
    """
    return np.sum((Y[:, :-1] + Y[:, 1:]) * (T[:, 1:] - T[:, :-1]) * 0.5, axis=1)


# ==========================================================
# Safe histogram helper (drop NaNs, avoid crashes)
# ==========================================================
def finite_hist(arr, bins=60, range=None):
    """
    Histogram with NaNs/inf removed.
    Returns (counts, bin_edges).
    If no finite values, returns a zero histogram with a dummy range.
    """
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if a.size == 0:
        # dummy edges so np.savez consumers still see something sane
        if range is None:
            edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
        else:
            edges = np.linspace(range[0], range[1], bins + 1, dtype=np.float32)
        return np.zeros(bins, dtype=np.int64), edges
    return np.histogram(a, bins=bins, range=range)


# ==========================================================
# MAIN
# ==========================================================
def run(filename):
    t_start = time.time()
    base = os.path.basename(filename).replace(".dat", "")
    outdir = f"WaveformAnalysis_{base}"
    if os.path.exists(outdir):
        print(f"[skip] {outdir} exists")
        return
    os.makedirs(outdir, exist_ok=True)

    print(f"Streaming read of {filename} (batch size = {BATCH_SIZE})")

    # --- First event to determine waveform length ---
    it = iter_wavedump_dt5720_headered(filename)
    first = next(it, None)
    if first is None:
        print("No events")
        return

    w0, t0, tt0 = first
    N = len(w0)

    # Precompute masks for fixed length
    t_template = np.arange(N, dtype=np.float32) * SAMPLE_PERIOD_NS
    i_ped = np.where(t_template < PED_WINDOW_NS)[0]
    i_sig = np.where((t_template >= INTEG_WINDOW[0]) &
                     (t_template <= INTEG_WINDOW[1]))[0]

    # Storage lists (final results)
    B_list  = []
    Qv_list = []
    Qp_list = []
    TC_list = []
    TP_list = []
    A_list  = []
    TT_list = []
    ped_q_list = []
    sig_q_list = []

    # --- Batch buffers ---
    batch_w = []
    batch_t = []
    batch_tt = []

    # Insert first event into batch
    batch_w.append(w0)
    batch_t.append(t0)
    batch_tt.append(tt0)

    total_events = 1
    PRINT_INTERVAL = 100000

    # ======================================================
    # Helper: process a batch
    # ======================================================
    def process_batch(batch_w, batch_t, batch_tt):
        if len(batch_w) == 0:
            return

        B = len(batch_w)  # batch size
        W = np.stack(batch_w, axis=0)    # shape B×N
        T = np.stack(batch_t, axis=0)    # shape B×N
        TT = np.array(batch_tt, dtype=np.uint64)

        # Baseline
        baseline = np.mean(W[:, i_ped], axis=1)

        # Amplitude + peak index
        peak_idx = np.argmin(W, axis=1)
        t_peak = T[np.arange(B), peak_idx]
        v_min = W[np.arange(B), peak_idx]
        amp = baseline - v_min

        # Signal integration
        Ysig = (baseline[:, None] - W[:, i_sig])
        Ysig = np.clip(Ysig, 0.0, None)
        Tsig = T[:, i_sig]
        q_vns = trapz_batch(Ysig, Tsig)
        q_pc = (q_vns * 1e-9) / R_OHM * 1e12

        # Time centroid
        wsum = np.sum(Ysig, axis=1)
        t_centroid = np.sum(Tsig * Ysig, axis=1) / np.where(wsum > 0, wsum, np.nan)

        # Pedestal integration
        Yped = (baseline[:, None] - W[:, i_ped])
        Yped = np.clip(Yped, 0.0, None)
        Tped = T[:, i_ped]
        ped_q = trapz_batch(Yped, Tped)

        # Fill master lists
        B_list.extend(baseline.tolist())
        Qv_list.extend(q_vns.tolist())
        Qp_list.extend(q_pc.tolist())
        TC_list.extend(t_centroid.tolist())
        TP_list.extend(t_peak.tolist())
        A_list.extend(amp.tolist())
        TT_list.extend(TT.tolist())
        ped_q_list.extend(ped_q.tolist())
        sig_q_list.extend(q_vns.tolist())  # signal = q_vns

    # ======================================================
    # Loop over remaining events
    # ======================================================
    for (w, t, tt) in it:
        total_events += 1

        # progress
        if (total_events % PRINT_INTERVAL) == 0:
            print(f"[{total_events:,} events] elapsed={time.time()-t_start:6.1f}s")

        batch_w.append(w)
        batch_t.append(t)
        batch_tt.append(tt)

        if len(batch_w) >= BATCH_SIZE:
            process_batch(batch_w, batch_t, batch_tt)
            batch_w.clear()
            batch_t.clear()
            batch_tt.clear()

    # Process last partial batch
    process_batch(batch_w, batch_t, batch_tt)

    good_events = len(B_list)
    print(f"Finished streaming. Total events = {total_events}, good = {good_events}")

    # ======================================================
    # Convert final arrays for summary + plotting
    # ======================================================
    B_arr  = np.asarray(B_list,  dtype=np.float32)
    Qv_arr = np.asarray(Qv_list, dtype=np.float32)
    Qp_arr = np.asarray(Qp_list, dtype=np.float32)
    TC_arr = np.asarray(TC_list, dtype=np.float32)
    TP_arr = np.asarray(TP_list, dtype=np.float32)
    A_arr  = np.asarray(A_list,  dtype=np.float32)
    TT_arr = np.asarray(TT_list, dtype=np.uint64)
    ped_arr = np.asarray(ped_q_list, dtype=np.float32)
    sig_arr = np.asarray(sig_q_list, dtype=np.float32)

    # ======================================================
    # Histograms and plots (NaN-safe)
    # ======================================================
    amp_counts,   amp_bins   = finite_hist(A_arr,  bins=60)
    qv_counts,    qv_bins    = finite_hist(Qv_arr, bins=60)
    qp_counts,    qp_bins    = finite_hist(Qp_arr, bins=60)
    base_counts,  base_bins  = finite_hist(B_arr,  bins=60)
    tcent_counts, tcent_bins = finite_hist(TC_arr, bins=60)
    tpeak_counts, tpeak_bins = finite_hist(TP_arr, bins=60)

    # PNGs
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1); plt.hist(A_arr[np.isfinite(A_arr)],  60); plt.title("Amplitude")
    plt.subplot(2, 3, 2); plt.hist(Qv_arr[np.isfinite(Qv_arr)], 60); plt.title("Charge [V ns]")
    plt.subplot(2, 3, 3); plt.hist(Qp_arr[np.isfinite(Qp_arr)], 60); plt.title("Charge [pC]")
    plt.subplot(2, 3, 4); plt.hist(B_arr[np.isfinite(B_arr)],  60); plt.title("Baseline")
    plt.subplot(2, 3, 5); plt.hist(TC_arr[np.isfinite(TC_arr)], 60); plt.title("t_centroid")
    plt.subplot(2, 3, 6); plt.hist(TP_arr[np.isfinite(TP_arr)], 60); plt.title("t_peak")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "histograms.png"), dpi=200)
    plt.close()

    # Baseline vs signal charge
    ped_mean = np.mean(ped_arr[np.isfinite(ped_arr)]) if ped_arr.size > 0 else 0.0
    pedc = ped_arr - ped_mean
    sigc = sig_arr - ped_mean

    ped_counts, ped_bins = finite_hist(pedc, bins=100)
    sig_counts, sig_bins = finite_hist(sigc, bins=100)

    plt.figure(figsize=(7, 5))
    plt.hist(pedc[np.isfinite(pedc)], 100, histtype='step', label="Pedestal")
    plt.hist(sigc[np.isfinite(sigc)], 100, histtype='step', label="Signal")
    plt.legend()
    plt.title("Baseline vs Signal")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "charge_baseline_overlay.png"), dpi=200)
    plt.close()

    # Time vs baseline and rate
    tick_ns = 8.0
    tt_unwrapped = np.empty_like(TT_arr, dtype=np.float64)
    rollover = 0.0
    for i, ttag in enumerate(TT_arr):
        if i > 0 and ttag < TT_arr[i - 1]:
            rollover += TT_MOD
        tt_unwrapped[i] = float(ttag) + rollover

    times_s = tt_unwrapped * (tick_ns * 1e-9)

    plt.figure(figsize=(8, 5))
    plt.plot(times_s, B_arr, '.', alpha=0.3)
    plt.title("Baseline vs Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Baseline [V]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "baseline_vs_time.png"), dpi=200)
    plt.close()

    t0 = times_s[0]
    t_rel = times_s - t0
    bin_edges = np.arange(0, t_rel[-1] + 1.0, 1.0)
    rate, be = np.histogram(t_rel, bins=bin_edges)

    plt.figure(figsize=(8, 5))
    plt.plot(be[:-1], rate)
    plt.title("Rate vs Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Events / s")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rate.png"), dpi=200)
    plt.close()

    # ======================================================
    # Summary
    # ======================================================
    def stats(x):
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return (np.nan, np.nan, np.nan)
        mean = float(np.mean(x))
        std  = float(np.std(x))
        se   = float(std / np.sqrt(len(x)))
        return (mean, std, se)

    q_pc_valid = Qp_arr[np.isfinite(Qp_arr)]
    if q_pc_valid.size > 0:
        q_pc_median = float(np.median(q_pc_valid))
        q_pc_peak   = float(np.percentile(q_pc_valid, 99))
    else:
        q_pc_median = np.nan
        q_pc_peak   = np.nan

    total_span = float(times_s[-1] - times_s[0])
    if rate.size > 0:
        mean_rate = float(np.mean(rate))
        std_rate  = float(np.std(rate))
        se_rate   = float(std_rate / np.sqrt(len(rate)))
    else:
        mean_rate = std_rate = se_rate = np.nan

    # Geometry from filename
    r_scan = theta_led = phi_led = np.nan
    v_led = v_gantry = np.array([np.nan]*3, dtype=np.float32)
    theta_gantry = phi_gantry = np.nan

    m = re.search(r"r([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)_phi([+-]?\d+\.?\d*)", base)
    if m:
        r_scan = float(m.group(1))
        theta_led = float(m.group(2))
        phi_led = float(m.group(3))
        v_led = sph_to_cart(theta_led, phi_led)
        v_gantry = R_g_to_led.T @ v_led
        theta_gantry, phi_gantry = cart_to_sph(v_gantry)

    # scan index
    dirpath = os.path.dirname(os.path.abspath(filename)) or "."
    dat_files = sorted(
        glob.glob(os.path.join(dirpath, "wave_*.dat")),
        key=lambda f: os.path.getmtime(f)
    )
    scan_index_0 = dat_files.index(os.path.abspath(filename))
    scan_index_1 = scan_index_0 + 1

    summary = {
        "Filename": filename,
        "Output dir": outdir,
        "Total events": total_events,
        "Good events": good_events,
        "Acq span [s]": total_span,
        "Runtime [s]": total_span,
        "Rate mean [Hz]": mean_rate,
        "Rate std [Hz]": std_rate,
        "Rate SE [Hz]": se_rate,
        "amp": stats(A_arr),
        "charge_vns": stats(Qv_arr),
        "charge_pc": stats(Qp_arr),
        "charge_pc_median": q_pc_median,
        "charge_pc_peak": q_pc_peak,
        "baseline": stats(B_arr),
        "t_centroid": stats(TC_arr),
        "t_peak": stats(TP_arr),
        "scan_index_0based": scan_index_0,
        "scan_index_1based": scan_index_1,
        "r_scan [mm]": r_scan,
        "theta_LED [deg]": theta_led,
        "phi_LED [deg]": phi_led,
        "theta_gantry [deg]": theta_gantry,
        "phi_gantry [deg]": phi_gantry,
        "v_LED": v_led.tolist(),
        "v_gantry": v_gantry.tolist(),
        "normal_LED": v_led.tolist(),
        "normal_gantry": v_gantry.tolist(),
        "R_g_to_led_row0": R_g_to_led[0].tolist(),
        "R_g_to_led_row1": R_g_to_led[1].tolist(),
        "R_g_to_led_row2": R_g_to_led[2].tolist(),
    }

    # summary.txt
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            if isinstance(v, tuple):
                f.write(f"{k:18s} mean={v[0]:.5g} std={v[1]:.5g} se={v[2]:.5g}\n")
            else:
                f.write(f"{k:18s} {v}\n")

    # npz (use NaN-safe histograms)
    np.savez(
        os.path.join(outdir, "histograms_all.npz"),
        amp_counts=amp_counts,   amp_bins=amp_bins,
        qv_counts=qv_counts,     qv_bins=qv_bins,
        qp_counts=qp_counts,     qp_bins=qp_bins,
        base_counts=base_counts, base_bins=base_bins,
        tcent_counts=tcent_counts, tcent_bins=tcent_bins,
        tpeak_counts=tpeak_counts, tpeak_bins=tpeak_bins,
        ped_counts=ped_counts,   ped_bins=ped_bins,
        sig_counts=sig_counts,   sig_bins=sig_bins,
        rate_counts=rate,        rate_bin_edges=be
    )

    print(f"Analysis completed in {time.time()-t_start:.1f} s.")

    gc.collect()


# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: waveform_analysis.py <wavefile.dat>")
        sys.exit(1)
    run(sys.argv[1])
