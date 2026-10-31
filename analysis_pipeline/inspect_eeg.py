#!/usr/bin/env python3

"""
01_inspect_eeg.py

Purpose:
    Inspect a BIDS EEG recording before preprocessing or running REVE.

Author:
    Atharva Kadam
"""

import os
import mne


# ---------------------------------------------------------------------
# CHANGE THIS PATH IF YOU WANT TO INSPECT A DIFFERENT SUBJECT
# ---------------------------------------------------------------------
EEG_FILE = "/mnt/data/shyam/data/bids/I0002/sub-I0002150000002/ses-1/eeg/sub-I0002150000002_ses-1_task-EEG_eeg.edf"


def main():

    print("=" * 70)
    print("EEG INSPECTION")
    print("=" * 70)

    # Check file exists
    if not os.path.exists(EEG_FILE):
        print(f"\nERROR: File not found:\n{EEG_FILE}")
        return

    print("\nLoading EEG file...")
    raw = mne.io.read_raw_edf(EEG_FILE, preload=False, verbose=False)

    print("\nSuccessfully loaded EEG!\n")

    print("=" * 70)
    print("BASIC INFORMATION")
    print("=" * 70)

    print(raw)

    print(f"\nSampling Frequency : {raw.info['sfreq']} Hz")
    print(f"Number of Channels: {len(raw.ch_names)}")
    print(f"Recording Duration: {raw.times[-1]:.2f} seconds")
    print(f"Number of Samples : {raw.n_times}")

    print("\n" + "=" * 70)
    print("CHANNEL NAMES")
    print("=" * 70)

    for i, ch in enumerate(raw.ch_names, start=1):
        print(f"{i:02d}. {ch}")

    print("\n" + "=" * 70)
    print("CHANNEL TYPES")
    print("=" * 70)

    print(raw.get_channel_types())

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()