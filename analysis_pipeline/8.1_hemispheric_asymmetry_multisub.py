"""
Multi-subject/session hemispheric asymmetry aggregation.

Loops over every subject/session under outputs/{subject}/{session}/windows/
that has spike_windows.npy + baseline_windows.npy already saved (from
3_extract_windows.py), computes the same per-window L/R band-power
asymmetry index as 7_hemispheric_asymmetry.py, and:

  1. Pools everything for an overall spike-vs-baseline comparison.
  2. ALSO computes per-SUBJECT mean asymmetry so we can check whether the
     direction of the effect is consistent across subjects, rather than
     being dominated by whichever subject has the most windows.

Run from the project root:
    python 8_hemispheric_asymmetry_multisubject.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
from config import bands as BANDS

EEG_CHANNEL_NAMES = [
    "C3", "C4", "O1", "O2", "A1", "A2", "Cz", "F3", "F4", "F7", "F8", "Fz",
    "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6",
]
CH_IDX = {name: i for i, name in enumerate(EEG_CHANNEL_NAMES)}

LR_PAIRS = [
    ("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("C3", "C4"),
    ("T3", "T4"), ("T5", "T6"), ("P3", "P4"), ("O1", "O2"),
]

PRE_SPIKE_SEC = 1.0
POST_SPIKE_SEC = 2.0
WINDOW_SEC = PRE_SPIKE_SEC + POST_SPIKE_SEC
BATCH_SIZE = 500
MIN_WINDOWS_FOR_SUBJECT_MEAN = 10

project_root = Path("/mnt/data/shyam/sanjana/MROP_NCC_Consciousness")
outputs_root = project_root / "outputs"
multi_dir = outputs_root / "multi_subject"
multi_dir.mkdir(parents=True, exist_ok=True)


def find_sessions_with_windows(outputs_root):
    for sub_dir in sorted(outputs_root.glob("sub-*")):
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            wdir = ses_dir / "windows"
            spike_f = wdir / "spike_windows.npy"
            base_f = wdir / "baseline_windows.npy"
            if spike_f.exists() and base_f.exists():
                yield sub_dir.name, ses_dir.name, spike_f, base_f


def band_power_batch(windows_batch, sfreq, n_t):
    w = windows_batch.astype(np.float32, copy=False)
    freqs, psd = signal.welch(w, fs=sfreq, nperseg=min(256, n_t), axis=2)
    out = {}
    for band_name, (lo, hi) in BANDS.items():
        idx = (freqs >= lo) & (freqs <= hi)
        out[band_name] = np.trapz(psd[:, :, idx], freqs[idx], axis=2)
    return out


def asymmetry_for_windows(windows_mmap, sfreq, n_t):
    n = windows_mmap.shape[0]
    rows = {f"{band}_{l}-{r}": [] for band in BANDS for (l, r) in LR_PAIRS}
    for start in range(0, n, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, n)
        batch = np.asarray(windows_mmap[start:stop], dtype=np.float32)
        bp = band_power_batch(batch, sfreq, n_t)
        for band in BANDS:
            for (l, r) in LR_PAIRS:
                lp = bp[band][:, CH_IDX[l]]
                rp = bp[band][:, CH_IDX[r]]
                asym = (lp - rp) / (lp + rp + 1e-20)
                rows[f"{band}_{l}-{r}"].extend(asym.tolist())
    return pd.DataFrame(rows)


sessions = list(find_sessions_with_windows(outputs_root))
print(f"Found {len(sessions)} sessions with saved windows across "
      f"{len(set(s for s, _, _, _ in sessions))} subjects")

all_spike_asym = []
all_base_asym = []

for subject, session, spike_f, base_f in sessions:
    spike_windows = np.load(spike_f, mmap_mode="r")
    baseline_windows = np.load(base_f, mmap_mode="r")
    if spike_windows.shape[0] == 0 or baseline_windows.shape[0] == 0:
        print(f"  SKIP {subject}/{session}: empty spike or baseline windows")
        continue

    n_t = spike_windows.shape[2]
    sfreq = n_t / WINDOW_SEC

    spike_asym = asymmetry_for_windows(spike_windows, sfreq, n_t)
    base_asym = asymmetry_for_windows(baseline_windows, sfreq, n_t)
    spike_asym["subject"] = subject
    spike_asym["session"] = session
    base_asym["subject"] = subject
    base_asym["session"] = session

    all_spike_asym.append(spike_asym)
    all_base_asym.append(base_asym)

    print(f"  {subject}/{session}: {len(spike_asym)} spike, {len(base_asym)} baseline windows")

pooled_spike = pd.concat(all_spike_asym, ignore_index=True)
pooled_base = pd.concat(all_base_asym, ignore_index=True)
pooled_spike.to_csv(multi_dir / "pooled_spike_asymmetry.csv", index=False)
pooled_base.to_csv(multi_dir / "pooled_baseline_asymmetry.csv", index=False)

band_pair_cols = [c for c in pooled_spike.columns if c not in ("subject", "session")]

print(f"\nPooled: {len(pooled_spike)} spike windows, {len(pooled_base)} baseline windows "
      f"across {pooled_spike['subject'].nunique()} subjects")

stat_rows = []
for col in band_pair_cols:
    sv = pooled_spike[col].values
    bv = pooled_base[col].values
    u_stat, p = stats.mannwhitneyu(sv, bv, alternative="two-sided")
    effect_r = 1 - (2 * u_stat) / (len(sv) * len(bv))
    stat_rows.append({
        "band_pair": col,
        "pooled_spike_mean": sv.mean(),
        "pooled_baseline_mean": bv.mean(),
        "u_stat": u_stat,
        "p_value": p,
        "effect_size_r": effect_r,
    })
stats_df = pd.DataFrame(stat_rows)
reject, p_fdr, _, _ = multipletests(stats_df["p_value"], alpha=0.05, method="fdr_bh")
stats_df["p_fdr"] = p_fdr
stats_df["significant"] = reject

consistency_rows = []
for col in band_pair_cols:
    per_subj = pooled_spike.groupby("subject")[col].agg(["mean", "count"])
    per_subj = per_subj[per_subj["count"] >= MIN_WINDOWS_FOR_SUBJECT_MEAN]
    pooled_dir = np.sign(stats_df.loc[stats_df["band_pair"] == col, "pooled_spike_mean"].values[0])
    n_agree = int((np.sign(per_subj["mean"]) == pooled_dir).sum())
    n_subjects_checked = len(per_subj)
    consistency_rows.append({
        "band_pair": col,
        "n_subjects_checked": n_subjects_checked,
        "n_subjects_agree_with_pooled_direction": n_agree,
        "consistency_fraction": n_agree / n_subjects_checked if n_subjects_checked else np.nan,
    })
consistency_df = pd.DataFrame(consistency_rows)

merged = stats_df.merge(consistency_df, on="band_pair")
merged = merged.sort_values("effect_size_r", key=lambda s: s.abs(), ascending=False)
merged.to_csv(multi_dir / "hemispheric_asymmetry_multisubject_stats.csv", index=False)

print(f"\n{merged['significant'].sum()} / {len(merged)} band-pairs significant (pooled, FDR-corrected)")
print("\nTop 10 by effect size, with per-subject consistency:")
print(merged.head(10)[["band_pair", "pooled_spike_mean", "pooled_baseline_mean",
                        "effect_size_r", "significant",
                        "n_subjects_checked", "n_subjects_agree_with_pooled_direction",
                        "consistency_fraction"]].to_string(index=False))

plt.figure(figsize=(10, 6))
top10 = merged.head(10).sort_values("band_pair")
x = np.arange(len(top10))
width = 0.35
plt.bar(x - width / 2, top10["pooled_spike_mean"], width, label="Spike", color="crimson")
plt.bar(x + width / 2, top10["pooled_baseline_mean"], width, label="Baseline", color="steelblue")
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(x, top10["band_pair"], rotation=45, ha="right")
plt.ylabel("Mean asymmetry index (L-R)/(L+R)")
plt.title("Top 10 Hemispheric Asymmetry Effects (Pooled Across Subjects)")
plt.legend()
plt.tight_layout()
plt.savefig(multi_dir / "hemispheric_asymmetry_multisubject.png", dpi=150)
plt.close()

print(f"\nSaved to: {multi_dir}")
print("  - pooled_spike_asymmetry.csv, pooled_baseline_asymmetry.csv")
print("  - hemispheric_asymmetry_multisubject_stats.csv")
print("  - hemispheric_asymmetry_multisubject.png")