
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import mne

from config import bands as BANDS

EEG_CHANNEL_NAMES = [
    "C3", "C4", "O1", "O2", "A1", "A2", "Cz", "F3", "F4", "F7", "F8", "Fz",
    "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6",
]
CH_IDX = {name: i for i, name in enumerate(EEG_CHANNEL_NAMES)}

PRE_SPIKE_SEC = 1.0
POST_SPIKE_SEC = 2.0
WINDOW_SEC = PRE_SPIKE_SEC + POST_SPIKE_SEC
BATCH_SIZE = 500
MIN_WINDOWS_FOR_SUBJECT_MEAN = 10

project_root = Path("/mnt/data/shyam/sanjana/MROP_NCC_Consciousness")
outputs_root = project_root / "outputs"
multi_dir = outputs_root / "multi_subject"
topomap_dir = multi_dir / "topomaps"
topomap_dir.mkdir(parents=True, exist_ok=True)


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


def mean_channel_power(windows_mmap, sfreq, n_t):
    n = windows_mmap.shape[0]
    band_sums = {band: np.zeros(len(EEG_CHANNEL_NAMES), dtype=np.float64) for band in BANDS}
    for start in range(0, n, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, n)
        batch = np.asarray(windows_mmap[start:stop], dtype=np.float32)
        bp = band_power_batch(batch, sfreq, n_t)
        for band in BANDS:
            band_sums[band] += bp[band].sum(axis=0)
    return {band: band_sums[band] / n for band in BANDS}, n


sessions = list(find_sessions_with_windows(outputs_root))
print(f"Found {len(sessions)} sessions with saved windows across "
      f"{len(set(s for s, _, _, _ in sessions))} subjects")

session_rows = []

for subject, session, spike_f, base_f in sessions:
    spike_windows = np.load(spike_f, mmap_mode="r")
    baseline_windows = np.load(base_f, mmap_mode="r")
    if spike_windows.shape[0] == 0 or baseline_windows.shape[0] == 0:
        print(f"  SKIP {subject}/{session}: empty spike or baseline windows")
        continue

    n_t = spike_windows.shape[2]
    sfreq = n_t / WINDOW_SEC

    spike_power, n_spike = mean_channel_power(spike_windows, sfreq, n_t)
    base_power, n_base = mean_channel_power(baseline_windows, sfreq, n_t)

    # Normalize within this session BEFORE pooling across sessions: express
    # spike power as a ratio to this session's own baseline power, per
    # channel/band. This controls for inter-session amplitude scale drift
    # (electrode impedance, amplifier gain, day-to-day differences) the same
    # way 6_multi_subject_training.py's per-session baseline normalization
    # does for the classical feature pipeline.
    spike_ratio = {band: spike_power[band] / (base_power[band] + 1e-30) for band in BANDS}
    # Clip to a plausible physiological range (1/16x to 16x, i.e. log2 in
    # [-4, 4]) before aggregating. Without this, a single session with a
    # near-zero baseline-power denominator for some channel can produce a
    # ratio in the hundreds or thousands, which then dominates the average
    # even after subject-balancing -- not a real effect, just numerical
    # instability from dividing by a very small number.
    for band in BANDS:
        spike_ratio[band] = np.clip(spike_ratio[band], 1 / 16, 16)

    session_rows.append({
        "subject": subject, "session": session,
        "n_spike": n_spike, "n_base": n_base,
        "spike_power": spike_power, "base_power": base_power,
        "spike_ratio": spike_ratio,
    })
    print(f"  {subject}/{session}: {n_spike} spike, {n_base} baseline windows")

# 1. Window-weighted grand average of the within-session ratio
grand_ratio = {band: np.zeros(len(EEG_CHANNEL_NAMES)) for band in BANDS}
total_n = sum(r["n_spike"] for r in session_rows)
for r in session_rows:
    for band in BANDS:
        grand_ratio[band] += r["spike_ratio"][band] * r["n_spike"]
for band in BANDS:
    grand_ratio[band] /= total_n

# 2. Subject-balanced average of the within-session ratio
subjects = sorted(set(r["subject"] for r in session_rows))
subject_ratio_means = {band: [] for band in BANDS}

for subj in subjects:
    subj_rows = [r for r in session_rows if r["subject"] == subj]
    n_spike_subj = sum(r["n_spike"] for r in subj_rows)
    n_base_subj = sum(r["n_base"] for r in subj_rows)
    if n_spike_subj < MIN_WINDOWS_FOR_SUBJECT_MEAN or n_base_subj < MIN_WINDOWS_FOR_SUBJECT_MEAN:
        print(f"  Excluding {subj} from subject-balanced average "
              f"(only {n_spike_subj} spike / {n_base_subj} baseline windows)")
        continue
    for band in BANDS:
        weighted = sum(r["spike_ratio"][band] * r["n_spike"] for r in subj_rows) / n_spike_subj
        subject_ratio_means[band].append(weighted)

balanced_ratio = {band: np.mean(subject_ratio_means[band], axis=0) for band in BANDS}
n_subjects_in_balanced = len(subject_ratio_means[list(BANDS.keys())[0]])
print(f"\nSubject-balanced average uses {n_subjects_in_balanced} / {len(subjects)} subjects "
      f"(others excluded for too few windows)")

# Save raw numbers
rows = []
for band in BANDS:
    for i, ch in enumerate(EEG_CHANNEL_NAMES):
        rows.append({
            "band": band, "channel": ch,
            "grand_avg_spike_to_baseline_ratio": grand_ratio[band][i],
            "subject_balanced_spike_to_baseline_ratio": balanced_ratio[band][i],
        })
pd.DataFrame(rows).to_csv(multi_dir / "channel_band_power_ratio_multisubject.csv", index=False)
print(f"Saved per-channel ratio table to: "
      f"{multi_dir / 'channel_band_power_ratio_multisubject.csv'}")

# Plot topomaps: log2(spike/baseline) ratio, centered at 0 (= no difference)
info = mne.create_info(ch_names=EEG_CHANNEL_NAMES, sfreq=256.0, ch_types="eeg")
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage, match_case=False, on_missing="warn")


def plot_ratio_topomaps(ratio_dict, out_dir, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    for band in BANDS:
        log_ratio = np.log2(ratio_dict[band])
        vmax = np.abs(log_ratio).max()

        fig, ax = plt.subplots(figsize=(5, 4.5))
        im, _ = mne.viz.plot_topomap(log_ratio, info, axes=ax, show=False,
                                      vlim=(-vmax, vmax), cmap="RdBu_r")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("log2(spike / baseline power)")
        ax.set_title(f"{band.capitalize()}: spike vs baseline ({label})\n"
                      f"red = spike higher, blue = baseline higher")
        fig.tight_layout()
        out_path = out_dir / f"topomap_{band}_ratio.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


print("\nPlotting window-weighted grand average ratio topomaps...")
plot_ratio_topomaps(grand_ratio, topomap_dir / "grand_average",
                     f"grand average, {len(sessions)} sessions, {len(subjects)} subjects")

print("\nPlotting subject-balanced average ratio topomaps...")
plot_ratio_topomaps(balanced_ratio, topomap_dir / "subject_balanced",
                     f"subject-balanced, n={n_subjects_in_balanced} subjects")

print(f"\nAll topomaps saved to: {topomap_dir}")
print("  - grand_average/     (window-weighted; can be dominated by data-rich subjects)")
print("  - subject_balanced/  (equal weight per subject; preferred for reporting a general finding)")
print("\nNote: these show the WITHIN-SESSION spike/baseline ratio, averaged across")
print("sessions/subjects -- correcting for inter-session amplitude scale differences.")