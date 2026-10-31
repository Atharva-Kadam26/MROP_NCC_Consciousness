import mne
from pathlib import Path
from config import edf_file,output_dir,filter_low_hz,filter_high_hz,notch_hz

print("Loading raw EDF file")
raw = mne.io.read_raw_edf(edf_file, preload=True)
print(f"Raw data loaded: {len(raw.ch_names)} channels, {raw.times[-1]:.1f} secs")

#printing channels MNE needs to know which channels are EEG, EOG, ECG

channel_types={
    'LOC':'eog',    #Left outer canthus
    'ROC':'eog',    #Right outer canthus
    'ECGL':'ecg',   #ECG left
    'ECGR':'ecg',   #ECG right
    'CHIN1':'emg',   #Chin Muscle
    'CHIN2':'emg',   #Chin Muscle
    'CHEST':'misc',  #Breathing sensors
    'ABD':'misc',     #Abdominal sensors
    'FLOW': 'misc',
    'SNORE': 'misc',
    'DIF5': 'misc',
    'DIF6': 'misc',
    'POS': 'misc',
    'DC2': 'misc',
    'DC3': 'misc',
    'DC4': 'misc',
    'DC5': 'misc',
    'DC6': 'misc',
    'DC7': 'misc',
    'DC8': 'misc',
    'DC9': 'misc',
    'DC10': 'misc',
    'OSAT': 'misc',  # Oxygen saturation
    'PR': 'misc',    # Pulse rate
    'LAT1': 'misc',
    'LAT2': 'misc',
    'RAT1': 'misc',
    'RAT2': 'misc',
}
raw.set_channel_types(channel_types)
print("Channel types set correctly")




#Picking only EEG channels for filtering and saving
raw_eeg = raw.copy().pick('eeg')
print(f"EEG channels picked: {len(raw_eeg.ch_names)} channels")

#Band pass filtering
raw_eeg.filter(l_freq=filter_low_hz, h_freq=filter_high_hz,fir_design='firwin')
print("Bandpass filtering done")


#Notch filtering
raw_eeg.notch_filter(freqs=notch_hz)
print("Notch filtering done")


print(f"\nHighpass: {raw_eeg.info['highpass']} Hz")
print(f"Lowpass: {raw_eeg.info['lowpass']} Hz")

preprocessed_file = output_dir / "preprocessed_eeg.fif"
raw_eeg.save(preprocessed_file, overwrite=True)

print(f"\nPreprocessed EEG data saved to: {preprocessed_file}")


