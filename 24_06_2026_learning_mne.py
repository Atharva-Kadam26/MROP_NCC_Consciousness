#Loading Sample Data
import mne

sample_data_folder = mne.datasets.sample.data_path()
raw_file = sample_data_folder/'MEG'/'sample'/'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(raw_file,preload=True)

# print("---Basic Info---")
# print(raw.info)
#
# print("\n---Channel Names---")
# print(raw.ch_names)
#
#
# print("\n---Duratiion---")
# print(f"recording is {raw.times[-1]:.1f} seconds long")
#
# print("\n---Sampling rate---")
# print(f"Sampled at {raw.info['sfreq']} Hz")



#Preprocessing
#picking only EEG channels(ignoring MEG)

raw_eeg = raw.copy().pick('eeg')
# print(f"\n---EEG Channels Only---")
# print(f"Number of EEG channels: {len(raw_eeg.ch_names)}")
# print("EEG channel names:",raw_eeg.ch_names)



#Filtering
#band pass filter: keep 0.5-70Hz
#(wide enough to preserve gamma waves, 30-70 Hz)
#below 0.5Hz are bcoz slow drifts like(sweat,low movements)
#removes high frequency muscle noise above 70Hz
raw_filtered = raw.copy().pick(['eeg','eog'])
raw_filtered.filter(l_freq=0.5,h_freq=70.0,fir_design="firwin")
print("\n ---Filtering Done---")


#remove powerline noise (in india it is 50Hz)
raw_filtered.notch_filter(freqs=50)
print("---Notch filter done---")


print(f"\nHighpass is now: {raw_filtered.info['highpass']} Hz")
print(f"\nLowpass is now: {raw_filtered.info['lowpass']} Hz")


#ICA for eye blink removal
raw_eeg = raw_filtered.copy().pick('eeg')

from mne.preprocessing import ICA


ica = ICA(n_components=20,random_state=42,max_iter = "auto")
ica.fit(raw_eeg)
print('ICA fitting done')

# eog_channel=raw_filtered.copy().pick('eog')
eog_indices,eog_scores = ica.find_bads_eog(raw_filtered)
print(f"Components flagged as eye blink related: {eog_indices}")


raw_eeg.plot(block=True)

ica.plot_components()
ica.plot_properties(raw_eeg,picks=eog_indices)
ica.plot_sources(raw_eeg)



ica.exclude = eog_indices
raw_clean = raw_eeg.copy()
ica.apply(raw_clean)
print('Eye-blink components removed')

raw_clean.plot(block = True,title='After ICA')


#Creating Epochs
events = mne.find_events(raw)

epochs = mne.Epochs(
    raw_clean,  #Cleaned EEG
    events,     #Event Markers
    tmin=-0.2,   #200 milliseconds before event
    tmax=0.5,   #500 milliseconds after event
    baseline=(None,0),
    preload = True
)

print(epochs)


evoked = epochs.average()    #Averaging all 320 epochs together and shows the brain;s average response to the stimuli
print(evoked)

evoked.plot()


#Spectral Analysis
psd = epochs.compute_psd(method = 'multitaper',fmin = 0.5,fmax = 70)
psd_data ,freqs= psd.get_data(return_freqs=True)
fig=psd.plot(average=True)

import matplotlib.pyplot as plt


psd.plot_topomap()

alpha_mask = (freqs >= 8) & (freqs <= 12)
alpha_power = psd_data[:,:,alpha_mask].mean(axis=2)
print(f"Alpha Power Shape : {alpha_power.shape}")

gamma_mask = (freqs >= 30) & (freqs <= 70)
gamma_power = psd_data[:, :, gamma_mask].mean(axis=2)
print(f"Gamma power shape: {gamma_power.shape}")