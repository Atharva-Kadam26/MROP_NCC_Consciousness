import mne
import pandas as pd
from config import windows_dir,bands

window_labels = ['baseline','ictal_aware','ictal_impaired','ictal_unresponsive']

results = {}

for label in window_labels:
    seg = mne.io.read_raw_fif(windows_dir/f"{label}_raw.fif",preload=True)
    psd = seg.compute_psd(method="welch",fmin=0.5,fmax=70,verbose= False)
    psd_data,freqs = psd.get_data(return_freqs = True)

    band_power ={}
    for band_name,(lo,hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        band_power[band_name] = psd_data[:,mask].mean()

    results[label] = band_power
    print(f"\n--- {label} ({seg.times[-1]:.1f}s) ---")

    for band_name, power in band_power.items():
        print(f" {band_name} : {power:.4e}")

summary = pd.DataFrame(results).T
print("\n\n ==== Summary: band power across consciousness states ====")
print(summary)

print("\n === Gamma Power relative to baseline ===")
baseline_gamma = results["baseline"]["gamma"]

for label in window_labels:
    ratio = results[label]["gamma"]/baseline_gamma
    print(f"{label} : {ratio:.2f}x baseline")

summary.to_csv(windows_dir.parent/"band_power_summary.csv")
print(f"\n Saved summary to: {windows_dir.parent / 'band_power_summary.csv'}")