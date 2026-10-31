#cutting the 3 second window around the spike event


import mne
import numpy as np
import pandas as pd
from config import output_dir,data_root,subject,session

preprocessed_file = output_dir / "preprocessed_eeg.fif"
print("Loading p[reprocessed file:")
raw = mne.io.read_raw_fif(preprocessed_file, preload=True)
sfreq = raw.info['sfreq']    #256 Hz
print(f"Loaded: {len(raw.ch_names)} channels, {raw.times[-1]:.1f} secs, sampling rate: {sfreq} Hz")

#Spike Annotations
annotations_file = (data_root/subject/session/"eeg"/f"{subject}_{session}_task-EEG.annotations.csv")
ann_df = pd.read_csv(annotations_file)


spikes = ann_df[ann_df["Text"].isin(["spike","xlspike","xlevent"])].copy()
print(f"\nTotal spike annotations : {len(spikes)}")


recording_start_stamp = ann_df['Stamp'].min()
spikes['time_sec'] = (spikes['Stamp'] - recording_start_stamp)/sfreq
print(f"Spike times range: {spikes['time_sec'].min():.1f} sec to {spikes['time_sec'].max():.1f} sec")


#Window params
pre_spike_sec = 1
post_spike_sec = 2
window_sec = pre_spike_sec + post_spike_sec

print((f"\n Extracting {window_sec} sec window around each spike event: "))

spike_windows = []
spike_labels = []
valid_spike_times = []

for _ ,row in spikes.iterrows():
    t_spike = row['time_sec']
    t_start = t_spike - pre_spike_sec
    t_end = t_spike + post_spike_sec

    if t_start < 0 or t_end > raw.times[-1]:  
        continue  # Skip spikes that would result in out-of-bounds windows

    start_sample = int(t_start*sfreq)
    stop_sample = int(t_end*sfreq)

    window = raw.get_data(start = start_sample,stop = stop_sample)
    spike_windows.append(window)
    spike_labels.append(row['Text'])
    valid_spike_times.append(t_spike)

print(f"Successfully extracted: {len(spike_windows)} spike windows")



print("\n Extracting matching baseline windows ")


spike_times_array = np.array(spikes['time_sec'].values)
baseline_windows = []
baseline_times = []


for t in np.arange(60,raw.times[-1]-60, 30):  # Start from 60 sec to avoid early spikes and end 60 sec before the end
    too_close = np.any(np.abs(spike_times_array - t) < 15.0)
    if too_close:
        continue  # Skip this time point if it's too close to any spike

    t_start = t - pre_spike_sec
    t_end = t + post_spike_sec

    start_sample = int(t_start * sfreq)
    stop_sample = int(t_end * sfreq)

    window = raw.get_data(start=start_sample, stop=stop_sample)
    baseline_windows.append(window)
    baseline_times.append(t)


    if len(baseline_windows) >= len(spike_windows):
        break  #Stop if we have same number as spike windows


print(f"Successfully extracted: {len(baseline_windows)} baseline windows")


windows_dir = output_dir / "windows"
windows_dir.mkdir(exist_ok=True)

spike_array = np.array(spike_windows)
baseline_array = np.array(baseline_windows)

np.save(windows_dir / "spike_windows.npy", spike_array)
np.save(windows_dir / "baseline_windows.npy", baseline_array)
np.save(windows_dir / "spike_times.npy", np.array(valid_spike_times))
np.save(windows_dir / "baseline_times.npy", np.array(baseline_times))
pd.Series(spike_labels).to_csv(windows_dir / "spike_labels.csv", index=False)

print(f"\nSpike windows shape: {spike_array.shape}")
print(f"Baseline windows shape: {baseline_array.shape}")
print(f"Shape means: (n_windows, n_channels, n_timepoints)")
print(f"\nSaved to: {windows_dir}")




