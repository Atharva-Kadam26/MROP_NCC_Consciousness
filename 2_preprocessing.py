import mne
from config import vhdr_file,filter_high_hz,filter_low_hz,notch_hz,preprocessed_file

raw = mne.io.read_raw_brainvision(vhdr_file,preload=True)
print(f"Loaded raw : {len(raw.ch_names)} channels , {raw.times[-1]:.1f}s")

raw.filter(l_freq= filter_low_hz,h_freq=filter_high_hz,fir_design = "firwin")
print(f"Band-pass filtered: {filter_low_hz}-{filter_high_hz} Hz")

raw.notch_filter(notch_hz)
print(f"Notch filtered: {notch_hz} Hz")

print(f"\nHigh pass is now: {raw.info['highpass']} Hz")
print(f"\nLow pass is now: {raw.info['lowpass']} Hz")

raw.save(preprocessed_file,overwrite=True)
print(f"\n Saved preprocessed data to {preprocessed_file}")