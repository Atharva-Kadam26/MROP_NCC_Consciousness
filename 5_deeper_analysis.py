import mne

edf_file = "/mnt/data/shyam/data/bids/I0002/sub-I0002150000031/ses-1/eeg/sub-I0002150000031_ses-1_task-EEG_eeg.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)
print(raw)