#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# ==============================================================
#  Helpers: Geometry conversions
# ==============================================================

def sph_to_cart(theta_deg, phi_deg):
    """Convert spherical (deg) to Cartesian unit vector."""
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    return np.array([
        np.sin(th)*np.cos(ph),
        np.sin(th)*np.sin(ph),
        np.cos(th)
    ])

def cart_to_sph(v):
    """Convert Cartesian to spherical (deg). Returned θ from +z."""
    x, y, z = v
    th = np.arccos(np.clip(z, -1, 1))
    ph = np.arctan2(y, x)
    return np.rad2deg(th), np.rad2deg(ph)

# Rotation matrix converting gantry xyz → LED xyz
# LED direction = +x_gantry → becomes +z_LED
# z_gantry → y_LED
# produces a right-handed basis
R_g_to_led = np.array([
    [  0,  0, -1 ],   # x_LED
    [  0,  1,  0 ],   # y_LED
    [  1,  0,  0 ]    # z_LED (LED beam direction)
])

def gantry_to_led_angles(theta_g, phi_g):
    """Convert gantry scan θ,φ to LED-centric θ_LED, φ_LED."""
    vg = sph_to_cart(theta_g, phi_g)
    vL = R_g_to_led @ vg
    return cart_to_sph(vL)


# ==============================================================
#  Helpers: filenames and summary.txt parsing
# ==============================================================

def snap_angle(val, tol=0.25):
    return round(val / tol) * tol

def parse_angles_from_path(path):
    m = re.search(r"r([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)_phi([+-]?\d+\.?\d*)", path)
    if not m:
        return None, None, None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def read_summary(fname, tol=1.0):
    """Reads summary.txt including charge median/peak lines."""
    out = {}
    charge_re = re.compile(r"^(charge_vns|charge_pc)\s+mean=([0-9eE+\-.]+)\s+std=([0-9eE+\-.]+)\s+se=([0-9eE+\-.]+)")
    amp_re    = re.compile(r"^amp\s+mean=([0-9eE+\-.]+)\s+std=([0-9eE+\-.]+)\s+se=([0-9eE+\-.]+)")
    rate_re   = re.compile(r"^Rate\s+(mean|std|SE)\s+\[Hz\]\s+([0-9eE+\-.]+)")
    median_re = re.compile(r"^charge_pc_median\s+([0-9eE+\-.]+)")
    peak_re   = re.compile(r"^charge_pc_peak\s+([0-9eE+\-.]+)")
    acqspan_re = re.compile(r"^Acq span \[s\]\s+([0-9eE+\-.]+)")
    
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = acqspan_re.match(line)
            if m:
                out["Acq span [s]"] = float(m.group(1))
                continue

            m = charge_re.match(line)
            if m:
                name = m.group(1)
                out[f"{name}_mean"] = float(m.group(2))
                out[f"{name}_std"]  = float(m.group(3))
                out[f"{name}_se"]   = float(m.group(4))
                continue

            m = amp_re.match(line)
            if m:
                out["amp_mean"] = float(m.group(1))
                out["amp_std"]  = float(m.group(2))
                out["amp_se"]   = float(m.group(3))
                continue

            m = rate_re.match(line)
            if m:
                key = f"Rate {m.group(1)} [Hz]"
                out[key] = float(m.group(2))
                continue

            m = median_re.match(line)
            if m:
                out["charge_pc_median"] = float(m.group(1))
                continue

            m = peak_re.match(line)
            if m:
                out["charge_pc_peak"] = float(m.group(1))
                continue

            # fallback
            if ":" in line:
                key, val = line.split(":", 1)
            elif " " in line:
                parts = line.split()
                if len(parts) == 2:
                    key, val = parts
                else:
                    continue
            else:
                continue

            key, val = key.strip(), val.strip()
            try:
                val = float(val)
            except:
                pass
            out[key] = val

    # parse filename-derived angles
    d = os.path.dirname(fname)
    r, th, ph = parse_angles_from_path(d)
    if th is not None:
        out["theta"] = snap_angle(th, tol)
    if ph is not None:
        out["phi"] = snap_angle(ph, tol)
    if r is not None:
        out["r"] = snap_angle(r, tol)

    return out


# ===============================================================
# NEW: φ_LED-wrapped intensity-vs-φ plots (linear and log)
# ===============================================================

def plot_intensity_vs_phi_wrapped(df, outdir):
    # Wrap φ so that -170° → +190° instead of isolating them on the far left
    df["phi_LED_wrap"] = df["phi_LED"].apply(lambda p: p if p >= -15 else p + 360)

    out = os.path.join(outdir, "PhiSlices")
    os.makedirs(out, exist_ok=True)

    # --- Linear scale ---
    plt.figure(figsize=(8,6))
    for th in sorted(df["theta_LED"].unique()):
        sub = df[df["theta_LED"] == th].sort_values("phi_LED_wrap")
        if sub.empty: 
            continue
        plt.errorbar(sub["phi_LED_wrap"], sub["intensity"],
                     yerr=sub["intensity_err"], fmt='o-', capsize=3,
                     alpha=0.7, label=f"θ={th:.1f}°")

    plt.xlim(-15, 195)
    plt.xlabel("phi_LED (wrapped) [deg]")
    plt.ylabel("Intensity [Hz·pC]")
    plt.title("Intensity vs phi_LED (wrapped, linear)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    linname = os.path.join(out, "intensity_vs_phi_led_linear.png")
    plt.savefig(linname, dpi=200)
    plt.close()
    print(f"[PhiSlices] Saved {linname}")

    # --- Log scale ---
    plt.figure(figsize=(8,6))
    for th in sorted(df["theta_LED"].unique()):
        sub = df[df["theta_LED"] == th].sort_values("phi_LED_wrap")
        if sub.empty:
            continue
        vals = sub["intensity"].clip(lower=1e-6)
        errs = sub["intensity_err"]
        plt.errorbar(sub["phi_LED_wrap"], vals, yerr=errs,
                     fmt='o-', capsize=3, alpha=0.7,
                     label=f"θ={th:.1f}°")

    plt.xlim(-15, 195)
    plt.yscale("log")
    plt.xlabel("phi_LED (wrapped) [deg]")
    plt.ylabel("Intensity [Hz·pC]")
    plt.title("Intensity vs phi_LED (wrapped, log)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    logname = os.path.join(out, "intensity_vs_phi_led_log.png")
    plt.savefig(logname, dpi=200)
    plt.close()
    print(f"[PhiSlices] Saved {logname}")


# ===============================================================
# NEW: overlay intensity vs theta_LED (linear + log)
#       grouped by phi_LED (analogous to PhiSlices)
# ===============================================================

def plot_intensity_vs_theta_overlay(df, outdir):

    out = os.path.join(outdir, "ThetaSlices")
    os.makedirs(out, exist_ok=True)

    # --- Linear scale ---
    plt.figure(figsize=(8,6))

    for phi_val in sorted(df["phi_LED"].unique()):
        sub = df[df["phi_LED"] == phi_val].sort_values("theta_LED")
        if sub.empty:
            continue

        plt.errorbar(sub["theta_LED"], sub["intensity"],
                     yerr=sub["intensity_err"],
                     fmt='o-', capsize=3, alpha=0.7,
                     label=f"φ={phi_val:.1f}°")

    plt.xlabel("theta_LED (deg)")
    plt.ylabel("Intensity [Hz·pC]")
    plt.title("Intensity vs theta_LED (linear)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out, "intensity_vs_theta_linear.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[ThetaSlices] Saved {fname}")

    # --- Log scale ---
    plt.figure(figsize=(8,6))

    for phi_val in sorted(df["phi_LED"].unique()):
        sub = df[df["phi_LED"] == phi_val].sort_values("theta_LED")
        if sub.empty:
            continue

        vals = sub["intensity"].clip(lower=1e-6)
        errs = sub["intensity_err"]

        plt.errorbar(sub["theta_LED"], vals,
                     yerr=errs,
                     fmt='o-', capsize=3, alpha=0.7,
                     label=f"φ={phi_val:.1f}°")

    plt.yscale("log")
    plt.xlabel("theta_LED (deg)")
    plt.ylabel("Intensity [Hz·pC]")
    plt.title("Intensity vs theta_LED (log)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out, "intensity_vs_theta_log.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[ThetaSlices] Saved {fname}")


# ===============================================================
# NEW: Save phi-slices and theta-slices to a single NPZ
# ===============================================================

def save_slice_npz(df, outdir):
    """
    Save all slice data used for plotting into a single NPZ file.
    - phi-slices contain I(theta) for each phi
    - theta-slices contain I(phi) for each theta
    """

    # --- Wrap phi exactly like plot_intensity_vs_phi_wrapped ---
    df = df.copy()
    df["phi_LED_wrap"] = df["phi_LED"].apply(lambda p: p if p >= -15 else p + 360)

    phi_vals = sorted(df["phi_LED_wrap"].unique())
    theta_vals = sorted(df["theta_LED"].unique())

    out = {}

    # -------------------------
    # Build phi-slices
    # -------------------------
    out["phi_slices/phi_list"] = np.array(phi_vals)
    out["phi_slices/theta_list"] = np.array(theta_vals)

    for phi in phi_vals:
        sub = df[df["phi_LED_wrap"] == phi].sort_values("theta_LED")
        if sub.empty:
            continue
        key = f"phi_slices/slice_{phi:+07.2f}"
        out[key] = np.vstack([
            sub["theta_LED"].to_numpy(),
            sub["intensity"].to_numpy(),
            sub["intensity_err"].to_numpy()
        ])

    # -------------------------
    # Build theta-slices
    # -------------------------
    out["theta_slices/theta_list"] = np.array(theta_vals)
    out["theta_slices/phi_list"] = np.array(phi_vals)

    for th in theta_vals:
        sub = df[df["theta_LED"] == th].sort_values("phi_LED_wrap")
        if sub.empty:
            continue
        key = f"theta_slices/slice_{th:+07.2f}"
        out[key] = np.vstack([
            sub["phi_LED_wrap"].to_numpy(),
            sub["intensity"].to_numpy(),
            sub["intensity_err"].to_numpy()
        ])

    # -------------------------
    # Meta
    # -------------------------
    out["meta/num_points"] = len(df)
    out["meta/cols"] = np.array(df.columns.tolist())

    outpath = os.path.join(outdir, "lambert_slices.npz")
    np.savez_compressed(outpath, **out)

    print(f"[NPZ] Saved Lambert slices → {outpath}")


# ==============================================================
#  MAIN: Lambertian summary
# ==============================================================

def main():
    print("\n=== Collecting Lambertian-scan summaries... ===\n")

    dirs = sorted(glob.glob("WaveformAnalysis_*"))
    summaries = []

    for d in dirs:
        summ = os.path.join(d, "summary.txt")
        if os.path.exists(summ):
            info = read_summary(summ)
            info["dir"] = d
            summaries.append(info)
        else:
            print(f"WARNING: no summary.txt in {d}")

    if not summaries:
        print("No summary files found.")
        return

    df = pd.DataFrame(summaries)

    # ---------------------------------------------------------
    # Compute LED-centric angles
    # ---------------------------------------------------------
    df["theta_LED"] = df["theta"]
    df["phi_LED"] = df["phi"]


    # Save combined CSV
    outdir = "ScanSummary_Lambert"
    os.makedirs(outdir, exist_ok=True)
    csv_out = os.path.join(outdir, "scan_results_lamb.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    # ---------------------------------------------------------
    # Now repeat your existing maps/slices but using theta_LED
    # ---------------------------------------------------------

    thetas_LED = sorted(df["theta_LED"].dropna().unique())

    def plot_slices(df, value_col, err_col=None, ylabel="", title="", outname=""):
        if value_col not in df.columns:
            return
        plt.figure(figsize=(8,6))
        for th in thetas_LED:
            sub = df[df["theta_LED"] == th].sort_values("phi_LED")
            if sub.empty:
                continue
            yerr = sub[err_col].values if err_col and err_col in sub.columns else None
            plt.errorbar(sub["phi_LED"], sub[value_col], yerr=yerr,
                         fmt='o-', capsize=3, alpha=0.7,
                         label=f"theta_LED={th:.1f}°")
        plt.xlabel("phi_LED (deg)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend(title="LED-theta slices", ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, outname), dpi=200)
        plt.close()
        print(f"Saved: {outname}")

    # =========================================================
    # Intensity and errorbars
    # =========================================================
    DEFAULT_DURATION = 60.0
    if "Acq span [s]" not in df.columns:
        print(f"WARNING: no Acq span [s], assuming {DEFAULT_DURATION}s.")
        df["Acq span [s]"] = DEFAULT_DURATION

    if all(c in df.columns for c in ["Rate mean [Hz]", "Rate SE [Hz]",
                                     "Acq span [s]", "charge_pc_median"]):

        df["N_events"] = df["Rate mean [Hz]"] * df["Acq span [s]"]
        df["intensity"] = df["Rate mean [Hz]"] * df["charge_pc_median"]

        charge_err = df["charge_pc_se"] if "charge_pc_se" in df.columns else 0.0

        frac_rate_err = 1.0 / np.sqrt(np.clip(df["N_events"], 1, None))
        frac_charge_err = np.where(
            df["charge_pc_median"] > 0,
            (charge_err / df["charge_pc_median"]) ** 2,
            0.0
        )

        df["intensity_err"] = df["intensity"] * np.sqrt(frac_rate_err + frac_charge_err)
        df["intensity_err"].replace([np.nan, np.inf, -np.inf], 0.0, inplace=True)

        # 2D map
        pivot = df.pivot_table(index="theta_LED", columns="phi_LED", values="intensity")
        plt.figure(figsize=(8,6))
        im = plt.imshow(
            pivot.values, origin="lower", aspect="auto",
            cmap="inferno",
            extent=[pivot.columns.min(), pivot.columns.max(),
                    pivot.index.min(), pivot.index.max()]
        )
        plt.colorbar(im, label="Rate × Median Charge [Hz·pC]")
        plt.xlabel("phi_LED (deg)")
        plt.ylabel("theta_LED (deg)")
        plt.title("LED-centric Intensity Map")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "intensity_led_map.png"), dpi=200)
        plt.close()
        print("Saved intensity_led_map.png")

        # Slices
        plot_slices(df,
                    value_col="intensity",
                    err_col="intensity_err",
                    ylabel="Rate × Median Charge [Hz·pC]",
                    title="Intensity vs phi_LED for LED-theta slices",
                    outname="intensity_slices_led.png")


    # NEW: φ-wrapped linear/log plots
    plot_intensity_vs_phi_wrapped(df, outdir)

    # NEW: reverse slices: intensity vs θ grouped by φ
    plot_intensity_vs_theta_overlay(df, outdir)

    # Lambertian fits
    #run_lambertian_fits(df, outdir, angle_sigma_deg=2.0)  # set to 0.0 to ignore angle uncertainty

    # Alignment diagnostic
    #plot_intensity_vs_phi_for_each_theta(df, outdir)

    save_slice_npz(df, outdir)

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
