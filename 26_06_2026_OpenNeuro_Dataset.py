import openneuro

# openneuro.download(
#     dataset = 'ds003029',
#     target_dir = './epilepsy_ieeg_data',
#     include=['participants.tsv','dataset_description.json','README']
# )

import pandas as pd

participants = pd.read_csv('./epilepsy_ieeg_data/participants.tsv',sep='\t')
# print(participants.head(20))
# print({f"\nTotal subjects: {len(participants)}"})
# print(f"Column names: {list(participants.columns)}")
# print("\n")
# print(list(participants.columns))
# print("\n")
# print(participants[participants['site'] == 'NIH'].to_string())

# openneuro.download(
#     dataset ='ds003029',
#     target_dir ='./epilepsy_ieeg_data',
#     include=['sub-pt01']
# )


import os

# for root,dirs,files in os.walk('./epilepsy_ieeg_data/sub-pt01'):
#     for file in files:
#         print(os.path.join(root,file))


import mne

raw = mne.io.read_raw_brainvision(
    './epilepsy_ieeg_data/sub-pt01/ses-presurgery/ieeg/sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg.vhdr',
    preload = True
)

# print(raw.info)
# print(f"Channels: {raw.ch_names}")
# print(f"Duration: {raw.times[-1]:.1f} seconds")
# print(f"Sampling Rate: {raw.info['sfreq']} Hz")

import pandas as pd

events_df = pd.read_csv(
'./epilepsy_ieeg_data/sub-pt01/ses-presurgery/ieeg/sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_events.tsv',
sep='\t'
)

# print(events_df)

channels_df = pd.read_csv('./epilepsy_ieeg_data/sub-pt01/ses-presurgery/ieeg/sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_events.tsv',
sep='\t')

# print(channels_df)

onset_time = events_df[events_df['trial_type'] == 'onset']['onset'].values[0]
offset_time = events_df[events_df['trial_type'] == 'offset']['onset'].values[0]

# print(f"Seizure : {onset_time}s to {offset_time}s (duration: {offset_time - onset_time:.1f}s)")

semiology = events_df[~events_df['trial_type'].str.startswith('+',na=False)]
# print(semiology[['onset','trial_type']])


# raw_filt =