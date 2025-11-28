#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# ==============================================================
#  Helpers
# ==============================================================

def snap_angle(val, tol=0.25):
    """Snap angle to nearest multiple of tol."""
    return round(val / tol) * tol

def parse_angles_from_path(path):
    """Extract r, theta, phi from directory name like WaveformAnalysis_wave_r450.0_+46.38_phi-27.00."""
    m = re.search(r"r([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)_phi([+-]?\d+\.?\d*)", path)
    if not m:
        return None, None, None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))

# ==============================================================
#  Read summary.txt
# ==============================================================

def read_summary(fname, tol=1.0):
    """
    Reads summary.txt and extracts charge + rate quantities,
    including new median/peak charge entries.
    Works with either space- or colon-separated format.
    """
    out = {}
    charge_re = re.compile(r"^(charge_vns|charge_pc)\s+mean=([0-9eE+\-.]+)\s+std=([0-9eE+\-.]+)\s+se=([0-9eE+\-.]+)")
    amp_re    = re.compile(r"^amp\s+mean=([0-9eE+\-.]+)\s+std=([0-9eE+\-.]+)\s+se=([0-9eE+\-.]+)")
    rate_re   = re.compile(r"^Rate\s+(mean|std|SE)\s+\[Hz\]\s+([0-9eE+\-.]+)")
    median_re = re.compile(r"^charge_pc_median\s+([0-9eE+\-.]+)")
    peak_re   = re.compile(r"^charge_pc_peak\s+([0-9eE+\-.]+)")

    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Charge mean/std/se
            m = charge_re.match(line)
            if m:
                name = m.group(1)
                out[f"{name}_mean"] = float(m.group(2))
                out[f"{name}_std"]  = float(m.group(3))
                out[f"{name}_se"]   = float(m.group(4))
                continue

            # Amplitude
            m = amp_re.match(line)
            if m:
                out["amp_mean"] = float(m.group(1))
                out["amp_std"]  = float(m.group(2))
                out["amp_se"]   = float(m.group(3))
                continue

            # Rate mean/std/se
            m = rate_re.match(line)
            if m:
                key = f"Rate {m.group(1)} [Hz]"
                out[key] = float(m.group(2))
                continue

            # Median / Peak charge lines
            m = median_re.match(line)
            if m:
                out["charge_pc_median"] = float(m.group(1))
                continue

            m = peak_re.match(line)
            if m:
                out["charge_pc_peak"] = float(m.group(1))
                continue

            # Generic fallback: key:value or key value
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

    # Extract and snap angles
    d = os.path.dirname(fname)
    r, th, ph = parse_angles_from_path(d)
    if th is not None: out["theta"] = snap_angle(th, tol)
    if ph is not None: out["phi"]   = snap_angle(ph, tol)
    if r  is not None: out["r"]     = snap_angle(r, tol)
    return out


def main():
    print("\n=== Collecting charge + rate summaries... ===\n")
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
    outdir = "ScanSummary"
    os.makedirs(outdir, exist_ok=True)
    csv_out = os.path.join(outdir, "scan_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved summary CSV : {csv_out}")

    # =========================================================
    # Mean / Median / Peak charge maps
    # =========================================================
    for col, title, cmap, fname in [
        ("charge_pc_mean",   "Mean Charge [pC]",   "viridis", "charge_mean_map.png"),
        ("charge_pc_median", "Median Charge [pC]", "magma",   "charge_median_map.png"),
        ("charge_pc_peak",   "Peak Charge [pC]",   "plasma",  "charge_peak_map.png"),
    ]:
        if col in df.columns:
            pivot = df.pivot_table(index="theta", columns="phi", values=col)
            plt.figure(figsize=(8,6))
            im = plt.imshow(pivot.values, origin="lower", aspect="auto", cmap=cmap,
                            extent=[pivot.columns.min(), pivot.columns.max(),
                                    pivot.index.min(), pivot.index.max()])
            plt.colorbar(im, label=title)
            plt.xlabel("phi (deg)")
            plt.ylabel("theta (deg)")
            plt.title(f"{title} vs pointing")
            plt.tight_layout()
            fn = os.path.join(outdir, fname)
            plt.savefig(fn, dpi=200)
            plt.close()
            print(f"Saved : {fn}")

    # =========================================================
    # Rate map
    # =========================================================
    if "Rate mean [Hz]" in df.columns:
        rate_col = "Rate mean [Hz]"
        pivot = df.pivot_table(index="theta", columns="phi", values=rate_col)
        plt.figure(figsize=(8,6))
        im = plt.imshow(pivot.values, origin="lower", aspect="auto", cmap="plasma",
                        extent=[pivot.columns.min(), pivot.columns.max(),
                                pivot.index.min(), pivot.index.max()])
        plt.colorbar(im, label="Rate [Hz]")
        plt.xlabel("phi (deg)")
        plt.ylabel("theta (deg)")
        plt.title("Trigger rate vs pointing")
        plt.tight_layout()
        fn = os.path.join(outdir, "rate_map.png")
        plt.savefig(fn, dpi=200)
        plt.close()
        print(f"Saved : {fn}")

    # =========================================================
    # 1D slices helper
    # =========================================================
    thetas = sorted(df["theta"].dropna().unique())

    def plot_slices(df, value_col, err_col=None, ylabel="", title="", outname=""):
        if value_col not in df.columns:
            return
        plt.figure(figsize=(8,6))
        for th in thetas:
            sub = df[df["theta"] == th].sort_values("phi")
            if sub.empty:
                continue
            yerr = sub[err_col].values if err_col and err_col in sub.columns else None
            plt.errorbar(sub["phi"], sub[value_col], yerr=yerr,
                         fmt='o-', capsize=3, alpha=0.7, label=f"theta={th:.1f}°")
        plt.xlabel("phi (deg)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend(title="theta slices", ncol=2, fontsize=8)
        plt.tight_layout()
        fn = os.path.join(outdir, outname)
        plt.savefig(fn, dpi=200)
        plt.close()
        print(f"Saved : {fn}")

    # --- Rate slices with SE ---
    if "Rate mean [Hz]" in df.columns:
        plot_slices(df, "Rate mean [Hz]", err_col="Rate SE [Hz]",
                    ylabel="Rate [Hz]",
                    title="Trigger rate vs φ for each θ (with SE)",
                    outname="rate_slices_all_theta_err.png")

    # --- Charge slices (mean, median, peak) ---
    if "charge_pc_mean" in df.columns:
        plot_slices(df, "charge_pc_mean", err_col="charge_pc_se",
                    ylabel="Mean charge [pC]",
                    title="Mean charge vs φ for each θ (with SE)",
                    outname="charge_mean_slices_all_theta.png")
    if "charge_pc_median" in df.columns:
        plot_slices(df, "charge_pc_median",
                    ylabel="Median charge [pC]",
                    title="Median charge vs φ for each θ",
                    outname="charge_median_slices_all_theta.png")
    if "charge_pc_peak" in df.columns:
        plot_slices(df, "charge_pc_peak",
                    ylabel="Peak charge [pC]",
                    title="Peak charge vs φ for each θ",
                    outname="charge_peak_slices_all_theta.png")

        # =========================================================
        # Combined Intensity map and propagated uncertainty
        # =========================================================
        # -------------------------------------------------
        # Ensure Acq span exists (fallback to default)
        # -------------------------------------------------
        DEFAULT_DURATION = 60.0   # change if needed

        if "Acq span [s]" not in df.columns:
            print(f"WARNING: 'Acq span [s]' missing : assuming {DEFAULT_DURATION} seconds.")
            df["Acq span [s]"] = DEFAULT_DURATION


        
        if all(c in df.columns for c in [
                "Rate mean [Hz]", "Rate SE [Hz]",
                "Acq span [s]", "charge_pc_median"]):

            # Compute event counts and intensity
            df["N_events"] = df["Rate mean [Hz]"] * df["Acq span [s]"]
            df["intensity"] = df["Rate mean [Hz]"] * df["charge_pc_median"]

            # ---------------------------------------------------------
            # Determine available charge uncertainty
            # (median has no std in summary -- fall back to mean charge std or SE)
            # ---------------------------------------------------------
            if "charge_pc_se" in df.columns:
                charge_err = df["charge_pc_se"]
            else:
                charge_err = 0.0  # ultimate fallback

            # ---------------------------------------------------------
            # Propagate fractional uncertainties:
            # sigma_I / I = sqrt( (1/N)  + (sigma_charge/charge_median)^2 )
            # ---------------------------------------------------------
            frac_rate_err = 1.0 / np.sqrt(np.clip(df["N_events"], 1, None))
            frac_charge_err = np.where(
                df["charge_pc_median"] > 0,
                (charge_err / df["charge_pc_median"]) ** 2,
                0.0
            )

            df["intensity_err"] = df["intensity"] * np.sqrt(frac_rate_err + frac_charge_err)

            # ---------------------------------------------------------
            # Clean all problem values so plotting NEVER fails
            # ---------------------------------------------------------
            df["intensity"].replace([np.nan, np.inf, -np.inf], 0.0, inplace=True)
            df["intensity_err"].replace([np.nan, np.inf, -np.inf], 0.0, inplace=True)

            # =========================================================
            # Combined Intensity 2D map
            # =========================================================
            pivot_int = df.pivot_table(index="theta", columns="phi", values="intensity")

            plt.figure(figsize=(8,6))
            im = plt.imshow(
                pivot_int.values,
                origin="lower",
                aspect="auto",
                cmap="inferno",
                extent=[
                    pivot_int.columns.min(), pivot_int.columns.max(),
                    pivot_int.index.min(), pivot_int.index.max()
                ]
            )
            plt.colorbar(im, label="Rate × Median Charge [Hz·pC]")
            plt.xlabel("phi (deg)")
            plt.ylabel("theta (deg)")
            plt.title("Combined Light Intensity (Rate × Median Charge)")
            plt.tight_layout()
            fn = os.path.join(outdir, "intensity_combined_map.png")
            plt.savefig(fn, dpi=200)
            plt.close()
            print(f"Saved: {fn}")

            # =========================================================
            # 1D intensity slices with propagated error
            # =========================================================
            plt.figure(figsize=(8,6))
            any_plotted = False

            for th in sorted(df["theta"].dropna().unique()):
                sub = df[df["theta"] == th].sort_values("phi")

                # No dropping — but replace invalid with 0
                sub_int = sub["intensity"].replace([np.nan, np.inf, -np.inf], 0.0)
                sub_err = sub["intensity_err"].replace([np.nan, np.inf, -np.inf], 0.0)

                # Skip only if *every* entry is zero
                if np.all(sub_int == 0):
                    print(f"NOTE: No valid intensity for theta={th:.1f}, skipping slice.")
                    continue

                plt.errorbar(
                    sub["phi"], sub_int, yerr=sub_err,
                    fmt="o-", capsize=3, alpha=0.7,
                    label=f"theta={th:.1f}°"
                )
                any_plotted = True

            plt.xlabel("phi (deg)")
            plt.ylabel("Rate × Median Charge [Hz·pC]")
            plt.title("Combined Light Intensity (Rate × Median Charge) vs phi for each theta")
            plt.grid(True)

            if any_plotted:
                plt.legend(title="θ slices", ncol=2, fontsize=8)

            plt.tight_layout()
            fn = os.path.join(outdir, "intensity_slices_all_theta_err.png")
            plt.savefig(fn, dpi=200)
            plt.close()
            print(f"Saved: {fn}")
    

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
