import mne
import pandas as pd
from config import vhdr_file,events_file,channels_file

raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

print("---BAsic Info---")
print(raw.info)
print(f"\n Channels : {len(raw.ch_names)}")
print(f"duration: {raw.times[-1]:.1f} secs")
print(f"Sampling rate: {raw.info['sfreq']} Hz")


events_df = pd.read_csv(events_file,sep="\t")
pd.set_option("display.max_colwidth",None)

print("\n--Events--")
print(events_df.to_string())

channels_df = pd.read_csv(channels_file,sep="\t")
print("\n--Channels--")
print(channels_df)

#getting only the meaningful rows removing the +N.000 rows those are not real sample recordings it is like a time stamp
semiology = events_df[~events_df["trial_type"].str.startswith("+",na=False)]
print("\n --- CLinical Timeline (semiology) ---")
print(semiology.to_string())