from pathlib import Path

data_root = Path("./epilepsy_ieeg_data")

subject = "sub-pt01"
session = "ses-presurgery"
run = "run-01"

raw_dir = data_root/subject/session/"ieeg"
vhdr_file = raw_dir/f"{subject}_{session}_task-ictal_acq-ecog_{run}_ieeg.vhdr"
events_file = raw_dir/f"{subject}_{session}_task-ictal_acq-ecog_{run}_events.tsv"
channels_file = raw_dir/f"{subject}_{session}_task-ictal_acq-ecog_{run}_channels.tsv"

#to keep raw data untouched we create a separate folder
output_dir = Path("./pipeline_outputs")/subject/run
output_dir.mkdir(parents=True, exist_ok=True)


preprocessed_file = output_dir/"preprocessed_raw.fif"
windows_dir = output_dir/"windows"
windows_dir.mkdir(parents=True,exist_ok= True)

filter_low_hz = 0.5
filter_high_hz = 70.0
notch_hz = 50

bands = {
    "delta":(0.5,4),
    "theta":(4,8),
    "alpha":(8,13),
    "beta":(13,30),
    "gamma":(30,70),
}

baseline_duration = 30 # how much time to take bfore the seizure starts

seizure_onset = 75.95
first_automatism = 84.94
unresponsive_marker = 130.82
seizure_offset = 161.82


