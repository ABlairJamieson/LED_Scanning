# WCTE / LED Lambertian Scan Data Package Format
### *Co-authored by Blair A. Jamieson & ChatGPT (OpenAI)*  
*Version: 2025-11-28*

## 1. Introduction
This document describes the data format used to store processed waveform analyses and Lambertian angular scan summaries for LED calibration runs in the WCTE water Cherenkov detector test environment. Each dataset directory contains per-angle waveform analysis summaries, a global Lambertian slice summary, and a per-point CSV summary.

## 2. Directory Structure Overview
A typical dataset looks like:
```
20nov2025_r450_led4/
├── WaveformAnalysis_wave_r450.0_+12.50_phi+14.00/
│    ├── histograms_all.npz
│    ├── histograms.png
│    ├── charge_baseline_overlay.png
│    ├── baseline_vs_time.png
│    ├── rate.png
│    └── summary.txt
├── ScanSummary_Lambert/
│    ├── lambert_slices.npz
│    └── scan_results_lamb.csv
```

## 3. File Naming Convention
Raw waveform file names follow:
```
wave_r{r_mm}_{theta_LED}_{phi_LED}.dat
```
Example:
```
wave_r450.0_+12.50_phi+14.00.dat
```

## 4. Per-Angle Analysis Output (`histograms_all.npz`)
Load with:
```python
import numpy as np
d = np.load("histograms_all.npz")
```

### Keys
```
amp_bins, amp_counts
base_bins, base_counts
ped_bins, ped_counts
qp_bins, qp_counts
qv_bins, qv_counts
rate_bin_edges, rate_counts
sig_bins, sig_counts
tcent_bins, tcent_counts
tpeak_bins, tpeak_counts
```

### Meaning
- **Amplitude** (`amp_*`): baseline – waveform minimum  
- **Baseline** (`base_*`): 0–100 ns region  
- **Charge (V·ns)** (`qv_*`): integrated waveform  
- **Charge (pC)** (`qp_*`): converted via \( q = \frac{1}{50\Omega} \int(V-V_b) dt \)  
- **Timing centroid** (`tcent_*`) and **peak time** (`tpeak_*`)  
- **Baseline vs signal** windows (`ped_*`, `sig_*`)  
- **DAQ rate** (`rate_*`): events/s using CAEN timetags  

# Summary.txt Field Definitions

## File Metadata
- **Filename** — Original waveform file name.
- **Output dir** — Directory containing plots and NPZ output for this run.
- **Total events** — Number of events read from the CAEN file.
- **Good events** — Number of successfully processed events (should match Total events).
- **Acq span [s]** — Acquisition time span in seconds (from CAEN timetags, unwrapped).
- **Runtime [s]** — Same as Acq span (difference between first and last event timestamps).

## Event Rate Statistics
- **Rate mean [Hz]** — Mean event rate (computed from 1-second bins).
- **Rate std [Hz]** — Standard deviation of the per-second rate histogram.
- **Rate SE [Hz]** — Standard error of the mean rate.

## Pulse & Charge Statistics
- **amp** — Pulse amplitude statistics: baseline − waveform minimum.
- **charge_vns** — Integrated charge in volt-nanoseconds (over 110–160 ns window).
- **charge_pc** — Charge converted to picoCoulombs.
- **charge_pc_median** — Median of the finite charge distribution.
- **charge_pc_peak** — 99th percentile (“peak”) of the charge distribution.
- **baseline** — Mean ADC baseline (from first 100 ns of waveform).
- **t_centroid** — Charge-weighted timing centroid of the pulse.
- **t_peak** — Time of the waveform minimum.

## Scan Identification
- **scan_index_0based** — Index of this run among all wave_*.dat files (0-based).
- **scan_index_1based** — Same, but 1-based.

## Geometry (from filename and rotation)
- **r_scan [mm]** — Radial distance of gantry scan.
- **theta_LED [deg]** — LED polar angle (from filename).
- **phi_LED [deg]** — LED azimuthal angle (from filename).
- **theta_gantry [deg]** — LED direction expressed in gantry coordinates (polar).
- **phi_gantry [deg]** — LED direction in gantry coordinates (azimuth).

## Direction Vectors
- **v_LED** — Unit vector (x, y, z) for LED direction in LED frame.
- **v_gantry** — Same vector rotated into gantry frame.
- **normal_LED** — Same as v_LED.
- **normal_gantry** — Same as v_gantry.

## Rotation Matrix
- **R_g_to_led_row0** — First row of gantry→LED rotation matrix.
- **R_g_to_led_row1** — Second row of rotation matrix.
- **R_g_to_led_row2** — Third row of rotation matrix.
ng  

## 6. Lambert Scan Summary (`ScanSummary_Lambert/`)
### 6.1 `lambert_slices.npz`
Keys include:
```
meta/cols
meta/num_points
phi_slices/phi_list
phi_slices/slice_+014.00
theta_slices/theta_list
theta_slices/slice_+005.00
```
Contains structured arrays for constant-φ and constant-θ slices.

### 6.2 `scan_results_lamb.csv`
Columns:
```
Filename,Acq span [s],
Rate mean [Hz],Rate std [Hz],Rate SE [Hz],
amp_mean,amp_std,amp_se,
charge_vns_mean,charge_vns_std,charge_vns_se,
charge_pc_mean,charge_pc_std,charge_pc_se,
charge_pc_median,charge_pc_peak,
scan_index_0based,scan_index_1based,
theta,phi,r,dir,theta_LED,phi_LED
```

## 7. Loading Data in Python
```python
import numpy as np
import pandas as pd

# Load histogram
d = np.load("histograms_all.npz")

# Load Lambert slices
L = np.load("lambert_slices.npz", allow_pickle=True)

# Load CSV
df = pd.read_csv("scan_results_lamb.csv")
```

## 8. Geometry and Direction
LED direction:
\[
v_{\mathrm{LED}} = (\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta)
\]

Rotation matrix:
\[
R_{g\rightarrow LED} =
\begin{bmatrix}
0 & 0 & -1 \\
0 & 1 & 0 \\
1 & 0 & 0
\end{bmatrix}
\]

## 9. Intended Physics Use Cases
- LED emission mapping  
- Lambertian fitting  
- Optical calibration  
- Water scattering studies  
- Photogrammetry alignment  
- Simulation tuning  
- DAQ stability checks  

## 10. Attribution
This dataset format was documented collaboratively by:
- **Blair A. Jamieson (University of Winnipeg)**  
- **ChatGPT (OpenAI)**  

## 11. License
Suggested:
```
CC-BY 4.0 — Attribution required.
© 2025 Blair A. Jamieson & OpenAI ChatGPT
```
