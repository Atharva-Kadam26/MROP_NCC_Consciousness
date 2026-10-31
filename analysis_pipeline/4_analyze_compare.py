
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.integrate import trapezoid
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg") 
from config import output_dir, data_root, subject, session ,bands as BANDS
 
# Channel order as produced by 2_preprocessing.py's raw.copy().pick('eeg') --
# MNE preserves original EDF order, and these are every channel NOT reassigned
# to eog/ecg/emg/misc in that script's channel_types dict. Index i here matches
# column "..._ch{i}" in feature_matrix.csv / feature_stats.csv.
EEG_CHANNEL_NAMES = [
    "C3", "C4", "O1", "O2", "A1", "A2", "Cz", "F3", "F4", "F7", "F8", "Fz",
    "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6",
]
# A1/A2 are mastoid reference electrodes, not scalp recording sites -- a
# "significant" result there likely reflects reference/montage artifact
# rather than a genuine cortical spike source. Flagged, not excluded, so the
# plot stays honest about what was actually tested.
REFERENCE_CHANNELS = {"A1", "A2"}
 
 
windows_dir = output_dir/ "windows"
results_dir = output_dir/ "analysis"
results_dir.mkdir(exist_ok=True)
 
print("memory mapped not fully loadded into RAM")
 
spike_windows = np.load(windows_dir/"spike_windows.npy", mmap_mode='r')
baseline_windows = np.load(windows_dir/"baseline_windows.npy", mmap_mode='r')
spike_times = np.load(windows_dir/"spike_times.npy")  #THESE two are not big just a few few kb so no need of memory mapping
baseline_times = np.load(windows_dir/"baseline_times.npy")
 
n_spike, n_ch,n_t = spike_windows.shape
n_base = baseline_windows.shape[0]
print(f"Spike windows: {spike_windows.shape} | Baseline windows: {baseline_windows.shape}")
print(f"On-disk size: spike {spike_windows.nbytes/1e6:.1f} MB, baseline {baseline_windows.nbytes/1e6:.1f} MB")
 
pre_spike_sec = 1.0
post_spike_sec = 2.0
window_sec = pre_spike_sec + post_spike_sec
sfreq = n_t / window_sec
print(f"Derived sampling rate: {sfreq} Hz (from window length {n_t} samples / {window_sec} sec)")
 
batch_size = 500    #windows per batch
 
def extract_features_batch(windows_batch):
    w = windows_batch.astype(np.float32,copy = False)
 
    feats = {
        "mean_amp" : w.mean(axis=2),  #axis =2 -> time axis (768 samples)  #for each window for each channel compute one number by looking across 768 timepoints #average voltage of each channel in each window
        "var_amp " : w.var(axis=2),    #how much each channel's voltage bounces around
        "ptp_amp" : w.max(axis=2) - w.min(axis=2),  #biggest swing top to bottom
        "line_length" : np.abs(np.diff(w,axis=2)).sum(axis=2),  #first we get the differences of the consecutive timepoints, then we take the absolute value of those differences, then we sum them up
 
    }
 
 
    freqs,psd =signal.welch(w,fs=sfreq,nperseg = min(256,n_t),axis=2) #signal.wlech computes the  power spectrum how much power is in each frequency band
    for band_name,(lo,hi) in BANDS.items():
        idx = (freqs >= lo) & (freqs <= hi)
        feats[f"power_{band_name}"] = trapezoid(psd[:,:,idx],freqs[idx],axis=2)  
    
    del w,psd
    return feats
 
 
def extract_features(windows, label): 
    n = windows.shape[0]   #windows is a fully mamapped array all 4360 spike windows 
    batch_feat_arrays = None    #dict of feature_name -> list of batch arrays
    feature_names = None 
    # we started with none bcoz we dont know the feature names until the first batch runs
 
    for start in range(0,n,batch_size):
        stop= min(start + batch_size,n)
        batch = np.asarray(windows[start:stop],dtype = np.float32)
        feats = extract_features_batch(batch)
 
        del batch
 
        if batch_feat_arrays is None:
            feature_names = list(feats.keys())
            batch_feat_arrays = {name: [] for name in feature_names}
        
        for name in feature_names:
            batch_feat_arrays[name].append(feats[name])
        
        del feats
 
        print(f"  [{label}] processed {stop}/{n} windows", end="\r")
 
    print()
 
    col_data = {}
    for name in feature_names:
        arr = np.concatenate(batch_feat_arrays[name], axis =0)
        for ch_idx in range(arr.shape[1]):
            col_data[f"{name}_ch{ch_idx}"] = arr[:,ch_idx]
        del arr
 
    gc.collect()
    return pd.DataFrame(col_data)
 
print("\nExtracting features (time-domain + spectral band power)...")
spike_feats = extract_features(spike_windows, "spike")
baseline_feats = extract_features(baseline_windows, "baseline")
print(f"Feature matrix shape: spike {spike_feats.shape}, baseline {baseline_feats.shape}")
 
X = pd.concat([spike_feats,baseline_feats],axis=0,ignore_index = True)
y = np.array([1] * n_spike + [0] * n_base)   #1 = spike , 0 = baseline
 
X.to_csv(results_dir / "feature_matrix.csv",index = False)
np.save(results_dir/"labels.npy",y)
print(f"Saved feature matrix ({X.shape}) and labels to {results_dir}")
 
 
#Statistical comparison
# For each feature: Mann-Whitney U test (p_value) tells us IF spike and baseline
# differ. Rank-biserial correlation (effect_size_r) tells us HOW MUCH they differ,
# in a way that doesn't inflate just because n_spike is large. Report both --
# a tiny p-value with a tiny effect size is not a strong finding.
 
n1 = len(spike_feats)  # n_spike
n2 = len(baseline_feats)  # n_base
 
stat_rows=[]
for col in X.columns:
    spike_vals = spike_feats[col].values
    base_vals = baseline_feats[col].values
    stat,p = stats.mannwhitneyu(spike_vals,base_vals,alternative="two-sided")
    # Rank-biserial correlation from the U statistic (Wendt, 1972):
    # ranges from -1 to +1. |r| near 0 = negligible effect even if p is tiny.
    # |r| ~0.1 small, ~0.3 medium, ~0.5 large (common rule-of-thumb cutoffs).
    effect_size_r = 1 - (2 * stat) / (n1 * n2)
    stat_rows.append({"feature":col,"u_stat":stat,"p_value":p,"effect_size_r":effect_size_r})
 
 
stats_df = pd.DataFrame(stat_rows)
reject, p_fdr, _, _ = multipletests(stats_df["p_value"], alpha=0.05, method="fdr_bh")
stats_df["p_fdr"] = p_fdr
stats_df["significant"] = reject
# Sort by effect size magnitude first, not just p_fdr -- with n_spike=4630 many
# features will hit tiny p-values regardless of how meaningful the difference is.
stats_df = stats_df.sort_values("effect_size_r", key=lambda s: s.abs(), ascending=False)
stats_df.to_csv(results_dir / "feature_stats.csv", index=False)
 
n_sig = stats_df["significant"].sum()
n_sig_and_meaningful = ((stats_df["significant"]) & (stats_df["effect_size_r"].abs() >= 0.3)).sum()
print(f"{n_sig} / {len(stats_df)} features significant after FDR correction (alpha=0.05)")
print(f"{n_sig_and_meaningful} / {len(stats_df)} features significant AND have at least a medium effect size (|r| >= 0.3)")
print("\nTop 10 features by effect size:")
print(stats_df.head(10).to_string(index=False))
 
 
 
print("\nGenerating plots...")
 
# 3a. Grand-average ERP: spike-locked vs baseline, averaged across channels & windows.
# Computed via chunked running sum over the mmap'd arrays instead of window_arr.mean(),
# so the full float64 array is never materialized in RAM at once.
def grand_average(windows_mmap):
    total = np.zeros(windows_mmap.shape[2], dtype=np.float64)
    n = windows_mmap.shape[0]
    for start in range(0, n,batch_size):
        stop = min(start + batch_size, n)
        batch = np.asarray(windows_mmap[start:stop], dtype=np.float32)
        total += batch.mean(axis=1).sum(axis=0)  # mean over channels, sum over windows in batch
        del batch
    return total / n
 
 
t = np.arange(n_t) / sfreq - 1.0  # pre_spike_sec = 1 from step 3
spike_grand_avg = grand_average(spike_windows)
baseline_grand_avg = grand_average(baseline_windows)
 
plt.figure(figsize=(8, 4))
plt.plot(t, spike_grand_avg, label="Spike windows", color="crimson")
plt.plot(t, baseline_grand_avg, label="Baseline windows", color="steelblue")
plt.axvline(0, color="black", linestyle="--", linewidth=0.8, label="Spike onset")
plt.xlabel("Time relative to event (s)")
plt.ylabel("Amplitude (grand average, all channels)")
plt.title("Grand-Average Waveform: Spike vs Baseline")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "grand_average_erp.png", dpi=150)
plt.close()
 
# From here on we only need the (small) feature DataFrames, so the big window
# arrays can be dropped for good.
del spike_windows, baseline_windows
gc.collect()
 
# 3b. Band power comparison (mean across channels & windows) per band
band_summary = []
for band_name in BANDS:
    cols = [c for c in X.columns if c.startswith(f"power_{band_name}_ch")]
    band_summary.append({
        "band": band_name,
        "spike_mean": spike_feats[cols].values.mean(),
        "baseline_mean": baseline_feats[cols].values.mean(),
    })
band_summary_df = pd.DataFrame(band_summary)
band_summary_df.to_csv(results_dir / "band_power_summary.csv", index=False)
 
x_pos = np.arange(len(BANDS))
width = 0.35
plt.figure(figsize=(8, 4))
plt.bar(x_pos - width / 2, band_summary_df["spike_mean"], width, label="Spike", color="crimson")
plt.bar(x_pos + width / 2, band_summary_df["baseline_mean"], width, label="Baseline", color="steelblue")
plt.xticks(x_pos, band_summary_df["band"])
plt.ylabel("Mean band power (a.u.)")
plt.title("Band Power: Spike vs Baseline")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "band_power_comparison.png", dpi=150)
plt.close()
 
# 3c. Per-channel significance summary (line-length feature, as an example seizure marker)
ll_stats = stats_df[stats_df["feature"].str.startswith("line_length_ch")].copy()
ll_stats["channel"] = ll_stats["feature"].str.extract(r"ch(\d+)").astype(int)
ll_stats = ll_stats.sort_values("channel")
ll_stats["channel_name"] = ll_stats["channel"].map(lambda i: EEG_CHANNEL_NAMES[i])
ll_stats["is_reference"] = ll_stats["channel_name"].isin(REFERENCE_CHANNELS)
 
plt.figure(figsize=(11, 4.5))
 
 
def bar_color(row):
    if row["is_reference"]:
        return "gray"  # mastoid reference electrode -- flagged, not a scalp finding
    return "crimson" if row["significant"] else "lightgray"
 
 
colors = ll_stats.apply(bar_color, axis=1)
plt.bar(ll_stats["channel"], -np.log10(ll_stats["p_fdr"]), color=colors)
plt.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.8, label="FDR p=0.05")
plt.xticks(ll_stats["channel"], ll_stats["channel_name"], rotation=45, ha="right")
plt.xlabel("Channel")
plt.ylabel("-log10(p_fdr)")
plt.title("Line-Length Feature: Per-Channel Significance (Spike vs Baseline)")
 
# Manual legend so "reference electrode" gets its own entry alongside significant/not
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="crimson", label="Significant (scalp channel)"),
    Patch(facecolor="lightgray", label="Not significant"),
    Patch(facecolor="gray", label="Reference electrode (A1/A2) -- interpret with caution"),
]
plt.legend(handles=legend_handles, fontsize=8)
plt.tight_layout()
plt.savefig(results_dir / "per_channel_significance.png", dpi=150)
plt.close()
 
ll_stats.to_csv(results_dir / "line_length_per_channel.csv", index=False)
 
print(f"\nSaved plots and tables to: {results_dir}")
print("  - feature_matrix.csv, labels.npy   (for downstream ML model)")
print("  - feature_stats.csv                (per-feature spike vs baseline stats + effect size)")
print("  - band_power_summary.csv")
print("  - grand_average_erp.png")
print("  - band_power_comparison.png")
print("  - per_channel_significance.png     (now labeled with channel names, A1/A2 flagged)")
print("  - line_length_per_channel.csv")