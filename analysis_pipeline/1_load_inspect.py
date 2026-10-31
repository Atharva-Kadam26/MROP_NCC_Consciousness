import mne
import pandas as pd
from config import edf_file, channels_file, json_file

raw = mne.io.read_raw_edf(edf_file, preload=True)

# print("Basic info :")
# print(raw.info)
# print(f"\nChannels: {len(raw.ch_names)}")
# print(f"Channel names: {raw.ch_names}")
# print(f"Duration: {raw.times[-1]:.1f} sec")
# print(f"Sampling rate: {raw.info['sfreq']} Hz")


# channels_df = pd.read_csv(channels_file, sep="\t")
# print("\nChannels metadata:")
# print(channels_df)


# print("Annotations:")
# print(raw.annotations)


# import pandas as pd
# from config import data_root,subject,session

# annotations_file = (data_root/subject/session/"eeg"/f"{subject}_{session}_task-EEG.annotations.csv")

# ann_df = pd.read_csv(annotations_file,header=None,names=["type", "sample", "timestamp", "duration",
#                             "label", "annotation_type", "timestamp2",
#                             "sample2", "duration2", "software",
#                             "extra1", "extra2", "score"])

# spikes = ann_df[ann_df["label"].isin(["spike","xlspike"])]
# print(f"\nTotal spikes: {len(spikes)}")
# print(f"Spike types: {spikes['label'].value_counts()}")
# print(f"\nFirst 5 spikes:")
# print(spikes[["timestamp", "label", "score"]].head())


# import pandas as pd
# from pathlib import Path

# annotations_file = Path("/mnt/data/shyam/data/bids/I0002/sub-I0002150000031/ses-1/eeg/sub-I0002150000031_ses-1_task-EEG.annotations.csv")

# # Read without assuming column names
# ann_df = pd.read_csv(annotations_file, header=None)
# print(f"Shape: {ann_df.shape}")
# print(f"\nFirst 3 rows:")
# print(ann_df.head(3).to_string())
# print(f"\nNumber of columns: {len(ann_df.columns)}")

import pandas as pd
from pathlib import Path

annotations_file = Path("/mnt/data/shyam/data/bids/I0002/sub-I0002150000031/ses-1/eeg/sub-I0002150000031_ses-1_task-EEG.annotations.csv")

# Read WITH header this time
ann_df = pd.read_csv(annotations_file)

# Show column names
# print("Columns:", list(ann_df.columns[:10]))  # first 10 columns

# # Show first 3 rows, just key columns
# print(f"\nShape: {ann_df.shape}")
# print(f"\nFirst 3 rows (key columns):")
# print(ann_df[['Origin', 'Stamp', 'Text', 'Type', 'StampTimeLocal']].head(3).to_string())

# # Count spike types
# print(f"\nUnique types in 'Type' column:")
# print(ann_df['Type'].value_counts())

import pandas as pd
from pathlib import Path

annotations_file = Path("/mnt/data/shyam/data/bids/I0002/sub-I0002150000031/ses-1/eeg/sub-I0002150000031_ses-1_task-EEG.annotations.csv")

ann_df = pd.read_csv(annotations_file)

# # Check what's in the Text column
# print("Unique Text values (first 20):")
# print(ann_df['Text'].value_counts().head(20))

# # Check all columns we have
# print(f"\nAll column names:")
# for col in ann_df.columns:
#     print(f"  {col}")

seizure = ann_df[ann_df['Text'] == 'seizure']
print("Seizure Annotation:")
print(seizure[["Text", "Stamp","StampTimeLocal"]].head().to_string())

spikes = ann_df[ann_df['Text'].isin(['spike', 'xlspike', 'xlevent'])]
print(f"\nTotal spike events: {len(spikes)}")
print(f"\nFirst 5 spikes:")
print(spikes[['Text', 'Stamp', 'StampTimeLocal']].head().to_string())

spikes = spikes.copy()
spikes['time_sec'] = (spikes['Stamp'] - ann_df['Stamp'].min()) / 256.0  # Converting  to seconds

print(f"\n First 5 spikes with time in seconds:")
print(spikes[['Text', 'Stamp', 'time_sec']].head().to_string())
