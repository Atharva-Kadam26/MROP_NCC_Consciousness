import mne
from config import preprocessed_file,windows_dir,baseline_duration,seizure_offset,seizure_onset,first_automatism,unresponsive_marker

raw = mne.io.read_raw_fif(preprocessed_file,preload = True)
print(f"Loaded preprocessed data: {raw.times[-1]:.1f}s total")

windows = {
    "baseline": (max(0,seizure_onset - baseline_duration),seizure_onset),
    "ictal_aware": (seizure_onset,first_automatism),
    "ictal_impaired": (first_automatism,unresponsive_marker),
    "ictal_unresponsive": (unresponsive_marker,seizure_offset)
}


for label,(start,end) in windows.items():
    segment = raw.copy().crop(tmin=start,tmax=end)
    out_path = windows_dir/f"{label}_raw.fif"
    segment.save(out_path,overwrite=True)
    print(f"{label}: {start:.1f}s to {end:.1f}s"
          f"({segment.times[-1]:.1f}s long) -> saved to {out_path}")

print("\n All four windows saved")