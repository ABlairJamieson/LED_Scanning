#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Helpers for intensity and quantities
# ============================================================

def ensure_intensity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has 'intensity' and 'intensity_err' columns.

    intensity       = (Rate mean [Hz]) * (charge_pc_median)
    σ_intensity/intensity ≈ sqrt[(σ_R/R)^2 + (σ_C/C)^2]

    where:
      R      = Rate mean [Hz]
      σ_R    = Rate SE [Hz]
      C      = charge_pc_median
      σ_C    = charge_pc_se (used as a proxy; no median SE in CSV)

    If some columns are missing, raises RuntimeError.
    """
    if "intensity" in df.columns and "intensity_err" in df.columns:
        return df

    required = ["Rate mean [Hz]", "Rate SE [Hz]", "charge_pc_median"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Cannot compute intensity: missing columns {missing}")

    R = df["Rate mean [Hz]"].astype(float).to_numpy()
    dR = df["Rate SE [Hz]"].astype(float).to_numpy()

    C = df["charge_pc_median"].astype(float).to_numpy()
    if "charge_pc_se" in df.columns:
        dC = df["charge_pc_se"].astype(float).to_numpy()
    else:
        dC = np.zeros_like(C)

    intensity = R * C

    # Fractional errors; guard against division by zero
    frac_R = np.zeros_like(R)
    mask_R = R != 0
    frac_R[mask_R] = (dR[mask_R] / np.abs(R[mask_R])) ** 2

    frac_C = np.zeros_like(C)
    mask_C = C != 0
    frac_C[mask_C] = (dC[mask_C] / np.abs(C[mask_C])) ** 2

    dI = np.abs(intensity) * np.sqrt(frac_R + frac_C)

    out = df.copy()
    out["intensity"] = intensity
    out["intensity_err"] = dI
    return out


def get_quantity_series(df: pd.DataFrame, quantity: str):
    """
    For a given *subset* DataFrame (typically at fixed θ), return:
       values, errors, ylabel, title_prefix

    quantity ∈ { 'rate', 'charge_pc_mean', 'charge_pc_median', 'intensity' }
    """
    if quantity == "rate":
        val_col = "Rate mean [Hz]"
        err_col = "Rate SE [Hz]"
        if val_col not in df.columns:
            raise RuntimeError(f"Missing column '{val_col}' for quantity 'rate'")
        vals = df[val_col].astype(float).to_numpy()
        errs = df[err_col].astype(float).to_numpy() if err_col in df.columns else None
        ylabel = "Rate [Hz]"
        prefix = "Trigger rate"
        return vals, errs, ylabel, prefix

    elif quantity == "charge_pc_mean":
        val_col = "charge_pc_mean"
        err_col = "charge_pc_se"
        if val_col not in df.columns:
            raise RuntimeError(f"Missing column '{val_col}' for quantity 'charge_pc_mean'")
        vals = df[val_col].astype(float).to_numpy()
        errs = df[err_col].astype(float).to_numpy() if err_col in df.columns else None
        ylabel = "Mean charge [pC]"
        prefix = "Mean charge"
        return vals, errs, ylabel, prefix

    elif quantity == "charge_pc_median":
        val_col = "charge_pc_median"
        if val_col not in df.columns:
            raise RuntimeError(f"Missing column '{val_col}' for quantity 'charge_pc_median'")
        vals = df[val_col].astype(float).to_numpy()
        # Use the SE of the mean as a proxy for median uncertainty if present
        if "charge_pc_se" in df.columns:
            errs = df["charge_pc_se"].astype(float).to_numpy()
        else:
            errs = None
        ylabel = "Median charge [pC]"
        prefix = "Median charge"
        return vals, errs, ylabel, prefix

    elif quantity == "intensity":
        df_i = ensure_intensity_columns(df)
        vals = df_i["intensity"].astype(float).to_numpy()
        errs = df_i["intensity_err"].astype(float).to_numpy()
        ylabel = "Rate × Median Charge [Hz·pC]"
        prefix = "Combined light intensity (Rate × Median Charge)"
        return vals, errs, ylabel, prefix

    else:
        raise ValueError(f"Unknown quantity '{quantity}'")


# shading helper
def shade(color, factor):
    """Return lighter or darker shade of the base color."""
    color = np.array(color[:3])  # drop alpha if present
    return tuple((factor * color + (1 - factor) * np.array([1, 1, 1])).clip(0, 1))

    
def plot_overlap_slices(datasets, quantity, outdir, logy=False):
    """
    Overlays multiple scans for a given quantity.
    datasets: dict[label] = dict with keys:
        'theta': array
        'phi': array
        'vals': dict[theta] -> array of values
        'errs': dict[theta] -> array of errors
    """

    print(f"Plotting overlay slices for quantity '{quantity}' ...")

    os.makedirs(outdir, exist_ok=True)

    # --- Collect all theta values that appear in ANY dataset ---
    all_thetas = sorted({
        float(theta)
        for d in datasets.values()
        for theta in d["vals"].keys()
    })

    # --- Base colormap: 10 well-separated colors ---
    base_cmap = plt.cm.tab10

    # Assign each theta a base color
    theta_base_colors = {
        th: base_cmap[i % 10]
        for i, th in enumerate(all_thetas)
    }

    # --- Assign each scan label a shade factor and marker type ---
    labels = list(datasets.keys())


    # Strongly separated shade factors:
    strong_shades = [0.25, 0.55, 0.85, 0.40, 0.70, 0.15, 0.95]
    # extend if needed
    while len(strong_shades) < len(labels):
        strong_shades += strong_shades

    shade_factors = {
        lbl: strong_shades[i]
        for i, lbl in enumerate(labels)
    }

    markers = ["o", "s", "^", "D", "P", "v", "H", "X"]
    label_markers = {lbl: markers[i % len(markers)] for i, lbl in enumerate(labels)}

    # --- Now produce the plot for each theta ---
    for th in thetas:
        # Base color per theta
        base_rgb = base_colors[th]  # e.g. from a colormap

        for label, df in datasets.items():
            sub = df[df["theta"] == th].sort_values("phi")
            if sub.empty:
                continue

            # Apply shading dependent on scan label
            sh = shade_factors[label]        # e.g. 0.25, 0.55, 0.85...
            this_color = shade(base_rgb[:3], sh) # <-- THIS IS THE IMPORTANT FIX

            
            # Marker style per scan
            marker = label_markers[label]

            plt.errorbar(
                sub["phi"],
                sub[value_col],
                yerr=sub[err_col] if err_col else None,
                fmt=marker + '-',
                markersize=4,
                capsize=3,
                color=this_color,            # <-- use shaded color
                alpha=0.9,
                label=f"{label} (θ={th:.1f}°)"
            )

        plt.xlabel("phi (deg)")
        plt.ylabel(quantity)
        plt.title(f"{quantity} vs phi for theta = {th:.1f}°")
        plt.grid(True)

        if logy:
            plt.yscale("log")
            plt.ylim(bottom=max(np.min(vals[vals > 0]) * 0.5, 1e-6))

        plt.legend(title="Scan", fontsize=8)
        plt.tight_layout()

        outname = os.path.join(outdir, f"{quantity}_theta_{th:.1f}.png")
        plt.savefig(outname, dpi=200)
        plt.close()

        print(f"Saved slice plot : {outname}")

    
def run_dark_subtraction(scan_dfs, quantity, outdir, logy=False):
    """
    For each non-dark scan:
       result = LED_quantity - DARK_quantity
    Error propagation: sigma^2 = sigma_LED^2 + sigma_dark^2
    Produces 1D theta–phi slice plots of subtracted quantity.
    """

    # =====================================================
    #  PALETTE A — RGBY base colors for each LED
    # =====================================================
    base_colors = {
        "LED4": (0.17, 0.63, 0.17),   # green
        "LED5": (0.12, 0.47, 0.71),   # blue
        "LED6": (0.58, 0.0, 0.55),   # green
        "LED7": (1.00, 0.50, 0.05),   # orange
    }

    # default fallback if an LED label not in map
    default_color = (0.4, 0.4, 0.4)   # grey

    # =====================================================
    #  Identify LED → DARK mappings
    # =====================================================
    mapping = {}
    for lbl in scan_dfs:
        if lbl.lower().startswith("dark"):
            continue
        dark_lbl = match_dark_scan(lbl, scan_dfs)
        mapping[lbl] = dark_lbl

    subtracted = {}

    # =====================================================
    #  Build subtracted datasets
    # =====================================================
    for lbl, dark_lbl in mapping.items():

        df_led  = scan_dfs[lbl]
        df_dark = scan_dfs[dark_lbl]

        merged = pd.merge(
            df_led, df_dark,
            on=["theta", "phi"],
            suffixes=("_LED", "_DARK")
        )

        # Strip suffixes before sending to get_quantity_series
        led_df  = merged.filter(regex="_LED$").rename(columns=lambda c: c.replace("_LED", ""))
        dark_df = merged.filter(regex="_DARK$").rename(columns=lambda c: c.replace("_DARK", ""))

        vals_L, err_L, ylabel, prefix = get_quantity_series(led_df, quantity)
        vals_D, err_D, _, _          = get_quantity_series(dark_df, quantity)

        vals = vals_L - vals_D
        errs = None
        if err_L is not None and err_D is not None:
            errs = np.sqrt(err_L**2 + err_D**2)

        df_out = pd.DataFrame({
            "theta": merged["theta"],
            "phi":   merged["phi"],
            "value": vals,
            "error": errs
        })

        subtracted[lbl] = df_out

    # =====================================================
    #  Plotting — each LED its own color,
    #              each theta a shade of that LED color
    # =====================================================
    all_thetas = sorted({
        float(t)
        for df in subtracted.values()
        for t in df["theta"].unique()
    })

    # shade factors mapped to theta index
    n_theta = max(len(all_thetas), 1)
    shade_factors = {
        th: (0.5 + 0.4 * (i / (n_theta - 1 if n_theta > 1 else 1)))
        for i, th in enumerate(all_thetas)
    }

    markers = ["o", "s", "^", "D", "x", "P", "v", "H"]
    labels = list(subtracted.keys())
    lbl_to_marker = {lbl: markers[i % len(markers)] for i, lbl in enumerate(labels)}

    plt.figure(figsize=(9, 7))
    any_plot = False

    for lbl, dfS in subtracted.items():

        # pick the LED's base color
        base_color = base_colors.get(lbl.upper(), default_color)

        for th in all_thetas:
            sub = dfS[dfS["theta"] == th].sort_values("phi")
            if sub.empty:
                continue

            # shaded color based on theta index
            factor = shade_factors[th]
            this_color = shade(base_color, factor)

            phi_vals = sub["phi"]
            y_vals   = sub["value"]
            y_errs   = sub["error"]

            if logy:
                mask = y_vals > 0
                if not np.any(mask):
                    continue
                phi_vals = phi_vals[mask]
                y_vals   = y_vals[mask]
                y_errs   = y_errs[mask] if y_errs is not None else None

            fmt = lbl_to_marker[lbl] + "-"

            plt.errorbar(
                phi_vals, y_vals,
                yerr=y_errs,
                fmt=fmt,
                color=this_color,
                capsize=3,
                alpha=0.85,
                label=f"{lbl}, theta={th:.1f}"
            )

            any_plot = True

    if not any_plot:
        print("No valid points for dark subtraction.")
        plt.close()
        return

    plt.xlabel("phi (deg)")
    plt.ylabel(ylabel + " (LED minus DARK)")
    title = f"{prefix} minus DARK vs phi"
    if logy:
        plt.yscale("log")
        title += " (log scale)"

    plt.title(title)
    plt.grid(True)
    plt.legend(title="Scan and theta", ncol=2, fontsize=8)
    plt.tight_layout()

    fname = f"{quantity}_subtract_dark{'_logy' if logy else ''}.png"
    savepath = os.path.join(outdir, fname)
    plt.savefig(savepath, dpi=200)
    plt.close()

    print(f"Saved : {savepath}")

    

def match_dark_scan(label, scan_dfs):
    """
    Match a scan label (e.g., 'LED4') to a dark scan.
    Rules:
       - If scan label is DARK or darkX → it is its own dark
       - If LEDX exists → use DARKX if present, otherwise DARK
       - If no DARK exists, raise error
    """
    if label.lower().startswith("dark"):
        return label  # dark is its own reference

    # Extract number (LED4 → "4")
    import re
    m = re.search(r'(\d+)', label)
    idx = m.group(1) if m else None

    # Case 1: DARK + X exists (DARK4)
    if idx:
        dark_label = f"DARK{idx}"
        if dark_label in scan_dfs:
            return dark_label

    # Case 2: plain DARK exists
    if "DARK" in scan_dfs:
        return "DARK"

    raise RuntimeError(f"No matching dark scan found for '{label}'")


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple LED/PMT scans using the ScanSummary/scan_results.csv files.\n\n"
            "Example:\n"
            "  compare_scans.py --scan LED4 ../14nov2025water_100kHz/ScanSummary \\\n"
            "                   --scan LED5 ../17nov2025_led5_100kHz/ScanSummary \\\n"
            "                   --scan LED6 ../17nov2025_led6_100kHz/ScanSummary \\\n"
            "                   --scan DARK ../17nov2025_dark/ScanSummary \\\n"
            "                   --mode overlay --quantity intensity --logy\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--scan", "-s",
        action="append",
        nargs=2,
        metavar=("LABEL", "SCANDIR"),
        required=True,
        help=(
            "Add a scan: LABEL and path to its ScanSummary directory.\n"
            "Each SCANDIR must contain a scan_results.csv.\n"
            "Example: --scan LED4 ../14nov2025water_100kHz/ScanSummary"
        )
    )

    parser.add_argument(
        "--mode",
        choices=["overlay","subtract-dark"],
        default="overlay",
        help="Comparison mode. overlay: 1D theta–phi slices\nsubtract-dark: subtract dark scan(s) from corresponding LED scans."
    )

    parser.add_argument(
        "--quantity", "-q",
        choices=["rate", "charge_pc_mean", "charge_pc_median", "intensity"],
        default="rate",
        help=(
            "Quantity to plot:\n"
            "  rate             → Rate mean [Hz]\n"
            "  charge_pc_mean   → Mean charge [pC]\n"
            "  charge_pc_median → Median charge [pC]\n"
            "  intensity        → Rate × median charge [Hz·pC]"
        )
    )

    parser.add_argument(
        "--outdir", "-o",
        default=".",
        help="Output directory for plots (default: current directory)."
    )

    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic scale on the y-axis (non-positive points are omitted)."
    )

    return parser.parse_args()


def load_scan_results(scan_specs):
    """
    scan_specs : list of (label, scandir)
    Returns: dict[label -> DataFrame]
    """
    scan_dfs = {}
    for label, scandir in scan_specs:
        csv_path = os.path.join(scandir, "scan_results.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No scan_results.csv found in {scandir}")

        df = pd.read_csv(csv_path)
        # Basic sanity: require theta/phi
        if "theta" not in df.columns or "phi" not in df.columns:
            raise RuntimeError(f"{csv_path} is missing 'theta' or 'phi' columns")

        scan_dfs[label] = df
        print(f"Loaded {csv_path} as '{label}'")
    return scan_dfs


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    scan_dfs = load_scan_results(args.scan)

    if args.mode == "overlay":
        plot_overlay_slices(scan_dfs, args.quantity, args.outdir, logy=args.logy)
    elif args.mode == "subtract-dark":
        run_dark_subtraction(scan_dfs, args.quantity, args.outdir, logy=args.logy)
    else:
        raise NotImplementedError(f"Mode '{args.mode}' not implemented.")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
