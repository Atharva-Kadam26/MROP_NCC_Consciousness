"""
Time-frequency analysis of spike vs baseline EEG windows.

Computes average spectrograms using Welch/STFT for all spike windows
and baseline windows, then visualizes:

1. Spike spectrogram
2. Baseline spectrogram
3. Difference spectrogram (Spike - Baseline)

Outputs:
    spike_spectrogram.png
    baseline_spectrogram.png
    difference_spectrogram.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

from config import output_dir

# -------------------------------------------------------
# Load windows
# -------------------------------------------------------

windows_dir = output_dir / "windows"
results_dir = output_dir / "analysis"
results_dir.mkdir(exist_ok=True)

spike_windows = np.load(
    windows_dir / "spike_windows.npy",
    mmap_mode="r"
)

baseline_windows = np.load(
    windows_dir / "baseline_windows.npy",
    mmap_mode="r"
)

print("Spike:", spike_windows.shape)
print("Baseline:", baseline_windows.shape)

# -------------------------------------------------------
# Sampling frequency
# -------------------------------------------------------

pre_spike_sec = 1.0
post_spike_sec = 2.0
window_sec = pre_spike_sec + post_spike_sec

n_time = spike_windows.shape[2]
sfreq = n_time / window_sec

print("Sampling rate:", sfreq)

# -------------------------------------------------------
# Average spectrogram
# -------------------------------------------------------

batch_size = 200

def average_spectrogram(windows, label):

    spectrogram_sum = None
    total = 0

    for start in range(0, windows.shape[0], batch_size):

        stop = min(start + batch_size, windows.shape[0])

        batch = np.asarray(
            windows[start:stop],
            dtype=np.float32
        )

        for sample in batch:

            # Average across channels
            eeg = sample.mean(axis=0)

            f, t, Sxx = signal.spectrogram(
                eeg,
                fs=sfreq,
                nperseg=128,
                noverlap=64,
                scaling="density"
            )

            if spectrogram_sum is None:
                spectrogram_sum = np.zeros_like(Sxx)

            spectrogram_sum += Sxx
            total += 1

        print(f"{label}: {stop}/{windows.shape[0]}", end="\r")

    print()

    return f, t, spectrogram_sum / total

# -------------------------------------------------------
# Compute
# -------------------------------------------------------

print("Computing spike spectrogram...")
f, t, spike_spec = average_spectrogram(
    spike_windows,
    "Spike"
)

print("Computing baseline spectrogram...")
_, _, baseline_spec = average_spectrogram(
    baseline_windows,
    "Baseline"
)

difference = spike_spec - baseline_spec

# -------------------------------------------------------
# Plot helper
# -------------------------------------------------------

def save_plot(data, title, filename):

    plt.figure(figsize=(8,5))

    plt.pcolormesh(
        t,
        f,
        10*np.log10(data + 1e-12),
        shading="gouraud"
    )

    plt.colorbar(label="Power (dB)")

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)

    plt.ylim(0,80)

    plt.tight_layout()

    plt.savefig(
        results_dir / filename,
        dpi=200
    )

    plt.close()

# -------------------------------------------------------
# Save figures
# -------------------------------------------------------

save_plot(
    spike_spec,
    "Average Spike Spectrogram",
    "spike_spectrogram.png"
)

save_plot(
    baseline_spec,
    "Average Baseline Spectrogram",
    "baseline_spectrogram.png"
)

save_plot(
    difference,
    "Spike - Baseline Spectrogram",
    "difference_spectrogram.png"
)

print("\nSaved:")
print(results_dir / "spike_spectrogram.png")
print(results_dir / "baseline_spectrogram.png")
print(results_dir / "difference_spectrogram.png")