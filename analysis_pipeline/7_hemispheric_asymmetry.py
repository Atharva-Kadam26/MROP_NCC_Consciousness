"""
Hemispheric band-power asymmetry: spike vs baseline.

Reuses the same memory-mapped window loading and Welch-based band power
extraction as 4_analyze_compare.py, but instead of per-channel power,
computes a normalized left/right asymmetry index per homologous channel
pair, per band, per window:

    asymmetry = (left_power - right_power) / (left_power + right_power)

Bounded [-1, 1]. Positive = left-dominant, negative = right-dominant.
Compares this index between spike and baseline windows to test whether
epileptiform activity is associated with a shift in hemispheric balance --
directly relevant to the split-brain / hemispheric-independence framing.

Run per subject/session, same as 4_analyze_compare.py:
    python 7_hemispheric_asymmetry.py
(uses config.py's output_dir/subject/session, same as the rest of the
pipeline)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
from config import output_dir, bands as BANDS

EEG_CHANNEL_NAMES = [
    "C3", "C4", "O1", "O2", "A1", "A2", "Cz", "F3", "F4", "F7", "F8", "Fz",
    "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6",
]
CH_IDX = {name: i for i, name in enumerate(EEG_CHANNEL_NAMES)}

LR_PAIRS = [
    ("Fp1", "Fp2"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("C3", "C4"),
    ("T3", "T4"),
    ("T5", "T6"),
    ("P3", "P4"),
    ("O1", "O2"),
]

windows_dir = output_dir / "windows"
results_dir = output_dir / "analysis"
results_dir.mkdir(exist_ok=True)

spike_windows = np.load(windows_dir / "spike_windows.npy", mmap_mode="r")
baseline_windows = np.load(windows_dir / "baseline_windows.npy", mmap_mode="r")
n_spike = spike_windows.shape[0]
n_base = baseline_windows.shape[0]
n_t = spike_windows.shape[2]
print(f"Spike windows: {spike_windows.shape} | Baseline windows: {baseline_windows.shape}")

pre_spike_sec = 1.0
post_spike_sec = 2.0
window_sec = pre_spike_sec + post_spike_sec
sfreq = n_t / window_sec
print(f"Derived sampling rate: {sfreq} Hz")

batch_size = 500


def band_power_batch(windows_batch):
    w = windows_batch.astype(np.float32, copy=False)
    freqs, psd = signal.welch(w, fs=sfreq, nperseg=min(256, n_t), axis=2)
    out = {}
    for band_name, (lo, hi) in BANDS.items():
        idx = (freqs >= lo) & (freqs <= hi)
        out[band_name] = np.trapz(psd[:, :, idx], freqs[idx], axis=2)
    return out


def asymmetry_for_windows(windows_mmap, label):
    n = windows_mmap.shape[0]
    rows = {f"{band}_{l}-{r}": [] for band in BANDS for (l, r) in LR_PAIRS}

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = np.asarray(windows_mmap[start:stop], dtype=np.float32)
        bp = band_power_batch(batch)
        for band in BANDS:
            for (l, r) in LR_PAIRS:
                lp = bp[band][:, CH_IDX[l]]
                rp = bp[band][:, CH_IDX[r]]
                asym = (lp - rp) / (lp + rp + 1e-20)
                rows[f"{band}_{l}-{r}"].extend(asym.tolist())
        print(f"  [{label}] {stop}/{n}", end="\r")
    print()
    return pd.DataFrame(rows)


print("\nComputing hemispheric asymmetry (spike windows)...")
spike_asym = asymmetry_for_windows(spike_windows, "spike")
print("Computing hemispheric asymmetry (baseline windows)...")
base_asym = asymmetry_for_windows(baseline_windows, "baseline")

spike_asym.to_csv(results_dir / "spike_asymmetry.csv", index=False)
base_asym.to_csv(results_dir / "baseline_asymmetry.csv", index=False)

stat_rows = []
for col in spike_asym.columns:
    sv = spike_asym[col].values
    bv = base_asym[col].values
    u_stat, p = stats.mannwhitneyu(sv, bv, alternative="two-sided")
    effect_r = 1 - (2 * u_stat) / (n_spike * n_base)
    stat_rows.append({
        "band_pair": col,
        "spike_mean_asym": sv.mean(),
        "baseline_mean_asym": bv.mean(),
        "u_stat": u_stat,
        "p_value": p,
        "effect_size_r": effect_r,
    })

stats_df = pd.DataFrame(stat_rows)
reject, p_fdr, _, _ = multipletests(stats_df["p_value"], alpha=0.05, method="fdr_bh")
stats_df["p_fdr"] = p_fdr
stats_df["significant"] = reject
stats_df = stats_df.sort_values("effect_size_r", key=lambda s: s.abs(), ascending=False)
stats_df.to_csv(results_dir / "hemispheric_asymmetry_stats.csv", index=False)

print(f"\n{stats_df['significant'].sum()} / {len(stats_df)} band-pair asymmetries "
      f"significant after FDR correction")
print("\nTop 10 by effect size:")
print(stats_df.head(10).to_string(index=False))

plt.figure(figsize=(12, 6))
x = np.arange(len(stats_df))
width = 0.35
order = stats_df.sort_values("band_pair")
plt.bar(x - width / 2, order["spike_mean_asym"], width, label="Spike", color="crimson")
plt.bar(x + width / 2, order["baseline_mean_asym"], width, label="Baseline", color="steelblue")
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(x, order["band_pair"], rotation=90, fontsize=7)
plt.ylabel("Mean asymmetry index (L-R)/(L+R)")
plt.title("Hemispheric Asymmetry by Band and Channel Pair: Spike vs Baseline")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "hemispheric_asymmetry.png", dpi=150)
plt.close()

print(f"\nSaved to: {results_dir}")
print("  - spike_asymmetry.csv, baseline_asymmetry.csv")
print("  - hemispheric_asymmetry_stats.csv")
print("  - hemispheric_asymmetry.png")