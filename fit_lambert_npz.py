#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

def lambert_model(theta_deg, A, dtheta_deg):
    """Lambertian: I = A * cos(theta + dtheta)."""
    return A * np.cos(np.deg2rad(theta_deg + dtheta_deg))


def exp_tail_model(theta_deg, B, tau_deg, theta_cut_deg):
    """
    Rayleigh-like tail: I = B * exp(-(theta - theta_cut)/tau)
    with independent B (no continuity constraint).
    """
    return B * np.exp(-(theta_deg - theta_cut_deg) / tau_deg)


# ------------------------------------------------------------
# Fitting helpers
# ------------------------------------------------------------

def fit_lambert_segment(theta_deg, I, sigma_I,
                        dtheta_grid_deg=np.linspace(-10, 10, 401)):
    """
    Weighted least-squares fit of A * cos(theta + dtheta) to one segment.
    dtheta scanned on a grid; for each dtheta we solve linear LS for A.
    """
    theta_deg = np.asarray(theta_deg, float)
    I = np.asarray(I, float)
    sigma = np.asarray(sigma_I, float)

    mask = np.isfinite(theta_deg) & np.isfinite(I) & np.isfinite(sigma) & (sigma > 0)
    theta_deg = theta_deg[mask]
    I = I[mask]
    sigma = sigma[mask]

    if theta_deg.size < 3:
        return np.nan, np.nan, np.nan, 0  # A, dtheta, chi2_red, N

    w = 1.0 / sigma**2
    theta_rad = np.deg2rad(theta_deg)

    best_chi2 = np.inf
    best_A = np.nan
    best_dtheta = np.nan

    for dth_deg in dtheta_grid_deg:
        c = np.cos(theta_rad + np.deg2rad(dth_deg))
        num = np.sum(w * I * c)
        den = np.sum(w * c * c)
        if den <= 0:
            continue
        A = num / den
        model = A * c
        chi2 = np.sum(w * (I - model)**2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_A = A
            best_dtheta = dth_deg

    ndof = max(theta_deg.size - 2, 1)
    chi2_red = best_chi2 / ndof if np.isfinite(best_chi2) else np.nan

    return best_A, best_dtheta, chi2_red, theta_deg.size


def fit_exponential_tail(theta_deg, I, sigma_I, theta_cut_deg):
    """
    Fit I(theta) = B * exp(-(theta - theta_cut)/tau) in the high-theta region.

    Do a weighted linear fit in log space:
        ln I = ln B - (theta - theta_cut)/tau
             = a + b * x,  x = theta - theta_cut
        with a = ln B, b = -1/tau
    """
    theta_deg = np.asarray(theta_deg, float)
    I = np.asarray(I, float)
    sigma = np.asarray(sigma_I, float)

    mask = np.isfinite(theta_deg) & np.isfinite(I) & np.isfinite(sigma) \
           & (I > 0) & (sigma > 0)
    theta_deg = theta_deg[mask]
    I = I[mask]
    sigma = sigma[mask]

    if theta_deg.size < 3:
        return np.nan, np.nan, np.nan, 0  # B, tau, chi2_red, N

    x = theta_deg - theta_cut_deg
    y = np.log(I)
    sigma_y = sigma / I
    w = 1.0 / (sigma_y**2)

    S  = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)

    denom = S * Sxx - Sx * Sx
    if denom <= 0:
        return np.nan, np.nan, np.nan, theta_deg.size

    a = (Sxx * Sy - Sx * Sxy) / denom
    b = (S * Sxy - Sx * Sy) / denom

    # Convert to B, tau
    if b == 0:
        return np.nan, np.nan, np.nan, theta_deg.size

    tau_deg = -1.0 / b
    B = np.exp(a)

    # Compute chi2 in linear space
    model = exp_tail_model(theta_deg, B, tau_deg, theta_cut_deg)
    chi2 = np.sum((I - model)**2 / (sigma**2))
    ndof = max(theta_deg.size - 2, 1)
    chi2_red = chi2 / ndof

    return B, tau_deg, chi2_red, theta_deg.size


# ------------------------------------------------------------
# Main driver: read NPZ, loop over phi-slices, fit & plot
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Piecewise Lambertian + exponential tail fits "
                    "from lambert_slices.npz."
    )
    ap.add_argument(
        "npzfile",
        nargs="?",
        default="ScanSummary_Lambert/lambert_slices.npz",
        help="NPZ file produced by summarize_lamb_scan.py (default: %(default)s)",
    )
    ap.add_argument(
        "--theta-cut", type=float, default=45.0,
        help="Boundary between Lambertian and Rayleigh regions in degrees (default: 45)"
    )
    ap.add_argument(
        "--phi", type=float, nargs="*",
        help="Optional list of phi_LED values to fit (use wrapped values, e.g. -5,14,33). "
             "If omitted, fit all."
    )
    args = ap.parse_args()

    data = np.load(args.npzfile)
    phi_list_all = np.array(data["phi_slices/phi_list"])
    theta_list = np.array(data["phi_slices/theta_list"])

    # If user specified a subset of phi, restrict
    if args.phi is not None and len(args.phi) > 0:
        phi_targets = set(args.phi)
        phi_list = [p for p in phi_list_all if np.any(np.isclose(p, list(phi_targets)))]
    else:
        phi_list = phi_list_all

    base_dir = os.path.dirname(args.npzfile)
    fit_dir = os.path.join(base_dir, "LambertFits_piecewise")
    os.makedirs(fit_dir, exist_ok=True)

    rows = []

    for phi in sorted(phi_list):
        key = f"phi_slices/slice_{phi:+07.2f}"
        if key not in data:
            print(f"[WARN] Missing slice for phi={phi:.2f} in NPZ; skipping.")
            continue

        slice_arr = data[key]  # shape (3, N): [theta, I, I_err]
        theta_deg = slice_arr[0, :]
        I = slice_arr[1, :]
        sigma_I = slice_arr[2, :]

        # Sort by theta
        order = np.argsort(theta_deg)
        theta_deg = theta_deg[order]
        I = I[order]
        sigma_I = sigma_I[order]

        # Split into low/high theta regions
        mask_low = theta_deg <= args.theta_cut
        mask_high = theta_deg >= args.theta_cut

        # --- Fit Lambertian part ---
        A_L, dtheta_L, chi2_L, N_L = fit_lambert_segment(
            theta_deg[mask_low], I[mask_low], sigma_I[mask_low]
        )

        # --- Fit exponential tail ---
        B_H, tau_H, chi2_H, N_H = fit_exponential_tail(
            theta_deg[mask_high], I[mask_high], sigma_I[mask_high],
            theta_cut_deg=args.theta_cut
        )

        # --- Build model curve for plotting ---
        th_grid = np.linspace(theta_deg.min(), theta_deg.max(), 400)
        I_model = np.zeros_like(th_grid)

        if np.isfinite(A_L) and np.isfinite(dtheta_L):
            mask_g_low = th_grid <= args.theta_cut
            I_model[mask_g_low] = lambert_model(
                th_grid[mask_g_low], A_L, dtheta_L
            )

        if np.isfinite(B_H) and np.isfinite(tau_H) and tau_H > 0:
            mask_g_high = th_grid >= args.theta_cut
            I_model[mask_g_high] = exp_tail_model(
                th_grid[mask_g_high], B_H, tau_H, args.theta_cut
            )

        # --- Linear plot ---
        plt.figure(figsize=(7,5))
        plt.errorbar(theta_deg, I, yerr=sigma_I,
                     fmt='o', capsize=3, label="data")
        plt.plot(th_grid, I_model, '-', label="fit")

        plt.axvline(args.theta_cut, color='gray', linestyle='--')

        title = (f"phi_LED={phi:.1f}° : "
                 f"A={A_L:.2e}, dtheta={dtheta_L:+.1f}°, "
                 f"B={B_H:.2e}, tau={tau_H:.1f}°, "
                 f"chi2_L={chi2_L:.2f}, chi2_H={chi2_H:.2f}")
        plt.title(title)
        plt.xlabel("theta_LED (deg)")
        plt.ylabel("Intensity [Hz·pC]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        png_lin = os.path.join(
            fit_dir, f"fit_phi_{phi:+06.2f}_linear.png"
        )
        plt.savefig(png_lin, dpi=200)
        plt.close()
        print(f"[Fit] Saved {png_lin}")

        # --- Log plot ---
        plt.figure(figsize=(7,5))
        plt.errorbar(theta_deg, I, yerr=sigma_I,
                     fmt='o', capsize=3, label="data")
        plt.plot(th_grid, I_model, '-', label="fit")

        plt.axvline(args.theta_cut, color='gray', linestyle='--')

        plt.yscale("log")
        plt.xlabel("theta_LED (deg)")
        plt.ylabel("Intensity [Hz·pC]")
        plt.title(title + " (log)")
        plt.grid(True, which="both", ls=':')
        plt.legend()
        plt.tight_layout()

        png_log = os.path.join(
            fit_dir, f"fit_phi_{phi:+06.2f}_log.png"
        )
        plt.savefig(png_log, dpi=200)
        plt.close()
        print(f"[Fit] Saved {png_log}")

        # --- Collect summary row ---
        rows.append({
            "phi_LED": phi,
            "theta_cut_deg": args.theta_cut,
            "A_L": A_L,
            "dtheta_L_deg": dtheta_L,
            "chi2_red_L": chi2_L,
            "N_L": N_L,
            "B_tail": B_H,
            "tau_tail_deg": tau_H,
            "chi2_red_tail": chi2_H,
            "N_H": N_H,
        })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(fit_dir, "lambert_piecewise_fit_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Fit] Wrote summary CSV → {csv_path}")
    else:
        print("[Fit] No slices fitted; nothing to write.")


if __name__ == "__main__":
    main()

