from pathlib import Path
import os

data_root = Path("/mnt/data/shyam/data/bids/I0002")
# subject = "sub-I0002150000031"
subject = os.environ.get("MROP_SUBJECT", "sub-I0002150000031")

print(f"Using subject: {subject}")
session = os.environ.get("MROP_SESSION", "ses-1")


raw_dir = data_root / subject / session / "eeg"
edf_file = raw_dir / f"{subject}_{session}_task-EEG_eeg.edf"
channels_file = raw_dir / f"{subject}_{session}_task-EEG_channels.tsv"
json_file = raw_dir / f"{subject}_{session}_task-EEG_eeg.json"

output_dir = Path("/mnt/data/shyam/sanjana/MROP_NCC_Consciousness/outputs")/subject/session
output_dir.mkdir(parents=True, exist_ok=True)

filter_low_hz = 0.5
filter_high_hz = 70.0
notch_hz = 60   #This is Harvard dataset from USA there the notch frequency is 60 Hz.

bands={
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 70)
}

