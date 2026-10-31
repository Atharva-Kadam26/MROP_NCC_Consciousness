
import os
import pickle
import argparse
import gc
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("WARNING")

parser = argparse.ArgumentParser()
parser.add_argument("--bids_root", type=str, default="/mnt/data/shyam/data/bids/I0002")
parser.add_argument("--out_dir", type=str,
                     default="/mnt/data/shyam/sanjana/MROP_NCC_Consciousness/reve_eeg/harvard_data")
parser.add_argument("--filter_low", type=float, default=0.5)
parser.add_argument("--filter_high", type=float, default=70.0)
parser.add_argument("--notch", type=float, default=60.0)
parser.add_argument("--pre_sec", type=float, default=2.5)
parser.add_argument("--post_sec", type=float, default=2.5)
parser.add_argument("--min_gap_from_spike", type=float, default=15.0)
parser.add_argument("--baseline_stride", type=float, default=30.0)
# parser.add_argument("--eval_frac", type=float, default=0.2)
# parser.add_argument("--seed", type=int, default=42)
# args = parser.parse_args()
parser.add_argument("--eval_frac", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--subjects", type=str, default=None,
                     help="Comma-separated subject IDs to process (e.g. sub-I0002150000031). "
                          "Default: process all subjects found under bids_root.")
parser.add_argument("--dedup_gap_sec", type=float, default=2.0,
                     help="Collapse spike detections within this many seconds of each other "
                          "into a single event (the automated detector often fires multiple "
                          "times on the same physical spike).")
parser.add_argument("--max_spikes_per_session", type=int, default=200,
                     help="Cap on spike windows kept per session, randomly subsampled after "
                          "dedup. Prevents a few noisy sessions from dominating the dataset.")
parser.add_argument("--eval_subjects", type=str, default=None,
                     help="Comma-separated subject IDs to force into the eval split "
                          "(overrides random eval_frac selection). E.g. "
                          "sub-I0002150000012,sub-I0002150000008")
args = parser.parse_args()
WINDOW_SEC = args.pre_sec + args.post_sec  # 5.0, matches harvard_seizure.yaml duration

BIPOLAR_PAIRS = [
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]
ANODES = [a for a, _ in BIPOLAR_PAIRS]
CATHODES = [c for _, c in BIPOLAR_PAIRS]
BIPOLAR_NAMES = [f"{a}-{c}" for a, c in BIPOLAR_PAIRS]

SPIKE_LABELS = {"spike", "xlspike"}


def find_sessions(bids_root):
    bids_root = Path(bids_root)
    for sub_dir in sorted(bids_root.glob("sub-*")):
        subject = sub_dir.name
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            session = ses_dir.name
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.is_dir():
                continue
            edf = eeg_dir / f"{subject}_{session}_task-EEG_eeg.edf"
            ann = eeg_dir / f"{subject}_{session}_task-EEG.annotations.csv"
            if edf.exists() and ann.exists():
                yield subject, session, edf, ann


def dedup_times(times, gap_sec):
    """Collapse detections within gap_sec of each other into a single event (keep the first)."""
    if len(times) == 0:
        return times
    times = np.sort(times)
    kept = [times[0]]
    for t in times[1:]:
        if t - kept[-1] >= gap_sec:
            kept.append(t)
    return np.array(kept)


def get_spike_and_baseline_times(ann_path, sfreq, recording_dur_sec, rng):
    ann_df = pd.read_csv(ann_path, low_memory=False)
    recording_start_stamp = ann_df["Stamp"].min()

    spikes = ann_df[ann_df["Text"].astype(str).str.lower().isin(SPIKE_LABELS)].copy()
    spikes["time_sec"] = (spikes["Stamp"] - recording_start_stamp) / sfreq
    spike_times = dedup_times(spikes["time_sec"].values, args.dedup_gap_sec)

    if len(spike_times) > args.max_spikes_per_session:
        spike_times = rng.choice(spike_times, size=args.max_spikes_per_session, replace=False)
        spike_times.sort()

    n_baseline_target = len(spike_times)
    baseline_times = []
    for t in np.arange(60, recording_dur_sec - 60, args.baseline_stride):
        if len(spike_times) and np.any(np.abs(spike_times - t) < args.min_gap_from_spike):
            continue
        baseline_times.append(t)
        if len(baseline_times) >= n_baseline_target:
            break

    return spike_times, np.array(baseline_times)


def cut_windows(raw_bipolar, times, sfreq, total_dur):
    windows = []
    n_target = int(round(WINDOW_SEC * sfreq))
    for t in times:
        t_start = t - args.pre_sec
        t_end = t + args.post_sec
        if t_start < 0 or t_end > total_dur:
            continue
        start_sample = int(round(t_start * sfreq))
        stop_sample = start_sample + n_target
        data = raw_bipolar.get_data(start=start_sample, stop=stop_sample)
        if data.shape[1] != n_target:
            continue
        windows.append(data.astype(np.float32))
    return windows


def process_session(subject, session, edf_path, ann_path,rng):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    # sfreq = raw.info["sfreq"]

    missing = [ch for pair in BIPOLAR_PAIRS for ch in pair if ch not in raw.ch_names]
    if missing:
        print(f"  SKIP {subject}/{session}: missing channels {set(missing)}")
        return [], []
    
    raw.pick(list(set(ANODES + CATHODES)))

    raw.filter(
    l_freq=args.filter_low,
    h_freq=args.filter_high,
    fir_design="firwin",
    verbose=False,
)

    raw.notch_filter(
    freqs=args.notch,
    verbose=False,
    )

    raw.resample(200, verbose=False)
    sfreq = raw.info["sfreq"]

    raw_bipolar = mne.set_bipolar_reference(
    raw,
    anode=ANODES,
    cathode=CATHODES,
    ch_name=BIPOLAR_NAMES,
    drop_refs=True,
    verbose=False,
    )

    
    raw_bipolar.reorder_channels(BIPOLAR_NAMES)

    total_dur = raw_bipolar.times[-1]
    spike_times, baseline_times = get_spike_and_baseline_times(ann_path, sfreq, total_dur,rng)

    spike_windows = cut_windows(raw_bipolar, spike_times, sfreq, total_dur)
    baseline_windows = cut_windows(raw_bipolar, baseline_times, sfreq, total_dur)

    print(f"  {subject}/{session}: {len(spike_windows)} spike windows, "
          f"{len(baseline_windows)} baseline windows (sfreq={sfreq}Hz)")
    
    del raw, raw_bipolar
    gc.collect()

    return spike_windows, baseline_windows


def main():
    # sessions = list(find_sessions(args.bids_root))
    # subjects = sorted(set(s for s, _, _, _ in sessions))
    sessions = list(find_sessions(args.bids_root))
    if args.subjects:
        wanted = set(s.strip() for s in args.subjects.split(","))
        sessions = [s for s in sessions if s[0] in wanted]
    subjects = sorted(set(s for s, _, _, _ in sessions))
    print(f"Found {len(sessions)} sessions across {len(subjects)} subjects")

    rng = np.random.default_rng(args.seed)
    if args.eval_subjects:
        eval_subjects = set(s.strip() for s in args.eval_subjects.split(","))
        missing = eval_subjects - set(subjects)
        if missing:
            print(f"WARNING: requested eval_subjects not found in data: {missing}")
        eval_subjects = eval_subjects & set(subjects)
        train_subjects = set(subjects) - eval_subjects
    else:
        shuffled = subjects.copy()
        rng.shuffle(shuffled)
        n_eval = max(1, int(round(len(subjects) * args.eval_frac)))
        eval_subjects = set(shuffled[:n_eval])
        train_subjects = set(shuffled[n_eval:])
    print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"Eval subjects  ({len(eval_subjects)}): {sorted(eval_subjects)}")
    
    out_dir = Path(args.out_dir)
    train_dir = out_dir / "processed_train"
    eval_dir = out_dir / "processed_eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": {"spike": 0, "baseline": 0}, "eval": {"spike": 0, "baseline": 0}}

    for subject, session, edf_path, ann_path in sessions:
        split = "eval" if subject in eval_subjects else "train"
        dest_dir = eval_dir if split == "eval" else train_dir
        already_done = list(dest_dir.glob(f"{subject}_{session}_*.pkl"))
        if already_done:
            print(f"  SKIP {subject}/{session}: already processed ({len(already_done)} files exist)")
            continue

        import multiprocessing as mp

        def _worker(q, subject, session, edf_path, ann_path, session_seed, dest_dir):
            try:
                session_rng = np.random.default_rng(session_seed)
                sw, bw = process_session(subject, session, edf_path, ann_path, session_rng)
                for i, w in enumerate(sw):
                    fname = f"{subject}_{session}_spike_{i}.pkl"
                    with open(dest_dir / fname, "wb") as f:
                        pickle.dump({"signal": w, "label": 2}, f)
                for i, w in enumerate(bw):
                    fname = f"{subject}_{session}_baseline_{i}.pkl"
                    with open(dest_dir / fname, "wb") as f:
                        pickle.dump({"signal": w, "label": 1}, f)
                q.put(("ok", len(sw), len(bw)))
            except Exception as e:
                q.put(("error", str(e), None))

        session_seed = args.seed + hash((subject, session)) % 100000
        q = mp.Queue()
        p = mp.Process(target=_worker, args=(q, subject, session, edf_path, ann_path, session_seed, dest_dir))
        p.start()

        try:
            status, n_spike, n_baseline = q.get(timeout=3600)
            p.join(timeout=30)
        except Exception:
            status, n_spike, n_baseline = "timeout", 0, 0
            p.terminate()
            p.join()

        if status == "timeout":
            print(f"  TIMEOUT {subject}/{session}: killing after 30 min")
            continue
        if status == "error":
            print(f"  ERROR {subject}/{session}: {n_spike}")
            continue

        counts[split]["spike"] += n_spike
        counts[split]["baseline"] += n_baseline
        continue

        for i, w in enumerate(spike_windows):
            fname = f"{subject}_{session}_spike_{i}.pkl"
            with open(dest_dir / fname, "wb") as f:
                pickle.dump({"signal": w, "label": 2}, f)
        for i, w in enumerate(baseline_windows):
            fname = f"{subject}_{session}_baseline_{i}.pkl"
            with open(dest_dir / fname, "wb") as f:
                pickle.dump({"signal": w, "label": 1}, f)

        counts[split]["spike"] += len(spike_windows)
        counts[split]["baseline"] += len(baseline_windows)

    print("\nDone.")
    print(f"Train: spike={counts['train']['spike']}, baseline={counts['train']['baseline']}")
    print(f"Eval:  spike={counts['eval']['spike']}, baseline={counts['eval']['baseline']}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()