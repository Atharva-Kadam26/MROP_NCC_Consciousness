# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from scipy.signal import coherence

# from config import output_dir

# EEG_CHANNEL_NAMES = [
#     "C3","C4","O1","O2","A1","A2","Cz","F3","F4","F7","F8","Fz",
#     "Fp1","Fp2","Fpz","P3","P4","Pz","T3","T4","T5","T6"
# ]

# windows_dir = output_dir / "windows"
# results_dir = output_dir / "analysis"
# results_dir.mkdir(exist_ok=True)

# spike_windows = np.load(
#     windows_dir / "spike_windows.npy",
#     mmap_mode="r"
# )

# baseline_windows = np.load(
#     windows_dir / "baseline_windows.npy",
#     mmap_mode="r"
# )

# n_channels = spike_windows.shape[1]
# n_time = spike_windows.shape[2]

# pre_spike = 1.0
# post_spike = 2.0

# sfreq = n_time / (pre_spike + post_spike)

# batch_size = 100

"""
Multi-subject functional connectivity analysis.

Computes correlation-based functional connectivity for every
subject/session that already contains extracted spike and baseline
windows.

Pipeline
--------
1. Discover all processed subjects.
2. Compute a correlation matrix for every spike window.
3. Compute a correlation matrix for every baseline window.
4. Average matrices within each subject.
5. Pool across subjects.
6. Compare spike vs baseline.
7. Perform statistics with FDR correction.
8. Produce publication-quality heatmaps.

Outputs
-------
outputs/multi_subject/

pooled_spike_connectivity.csv
pooled_baseline_connectivity.csv
difference_connectivity.csv

connectivity_multisubject_stats.csv

spike_connectivity_heatmap.png
baseline_connectivity_heatmap.png
difference_connectivity_heatmap.png
significant_connectivity_heatmap.png

run_info.json
"""

from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests


EEG_CHANNEL_NAMES = [
    "C3","C4","O1","O2","A1","A2",
    "Cz","F3","F4","F7","F8","Fz",
    "Fp1","Fp2","Fpz",
    "P3","P4","Pz",
    "T3","T4","T5","T6"
]

N_CHANNELS = len(EEG_CHANNEL_NAMES)

BATCH_SIZE = 250

MIN_WINDOWS= 10

project_root = Path("/mnt/data/shyam/sanjana/MROP_NCC_Consciousness")

outputs_root = project_root / "outputs"

multi_dir = outputs_root / "multi_subject"
multi_dir.mkdir(parents=True, exist_ok=True)

def find_sessions_with_windows(outputs_root):
    """
    Finds every subject/session that already contains

        spike_windows.npy
        baseline_windows.npy
    """

    for sub_dir in sorted(outputs_root.glob("sub-*")):

        for ses_dir in sorted(sub_dir.glob("ses-*")):

            windows_dir = ses_dir / "windows"

            spike_file = windows_dir / "spike_windows.npy"
            baseline_file = windows_dir / "baseline_windows.npy"

            if spike_file.exists() and baseline_file.exists():

                yield (
                    sub_dir.name,
                    ses_dir.name,
                    spike_file,
                    baseline_file
                )


def save_heatmap(matrix,
                 title,
                 filename,
                 cmap="viridis",
                 vmin=None,
                 vmax=None):

    plt.figure(figsize=(9,8))

    plt.imshow(
        matrix,
        cmap=cmap,
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax
    )

    plt.xticks(
        np.arange(N_CHANNELS),
        EEG_CHANNEL_NAMES,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        np.arange(N_CHANNELS),
        EEG_CHANNEL_NAMES,
        fontsize=7
    )

    plt.colorbar()

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        multi_dir / filename,
        dpi=200
    )

    plt.close()

    # ============================================================
# Connectivity computation
# ============================================================

def correlation_matrix(window):
    """
    Compute a Pearson correlation matrix for a single EEG window.

    Parameters
    ----------
    window : ndarray
        Shape = (channels, timepoints)

    Returns
    -------
    ndarray
        Shape = (channels, channels)
    """

    corr = np.corrcoef(window)

    # Replace NaN values that can arise from constant channels
    corr = np.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return corr


def average_connectivity_matrix(windows_mmap,
                                label,
                                subject,
                                session):
    """
    Average connectivity matrix over all windows.

    Parameters
    ----------
    windows_mmap : numpy.memmap

    Returns
    -------
    mean_matrix

    window_values

        Dictionary containing every edge value across
        every window.

        Used later for statistical testing.
    """

    n_windows = windows_mmap.shape[0]

    accumulator = np.zeros(
        (N_CHANNELS, N_CHANNELS),
        dtype=np.float64
    )

    edge_values = {}

    for i in range(N_CHANNELS):
        for j in range(i + 1, N_CHANNELS):
            edge_values[(i, j)] = []

    print(
        f"\n{label}: {subject}/{session}"
    )

    for start in range(0,
                       n_windows,
                       BATCH_SIZE):

        stop = min(
            start + BATCH_SIZE,
            n_windows
        )

        batch = np.asarray(
            windows_mmap[start:stop],
            dtype=np.float32
        )

        for window in batch:

            corr = correlation_matrix(window)

            accumulator += corr

            for i in range(N_CHANNELS):

                for j in range(i + 1,
                               N_CHANNELS):

                    edge_values[(i, j)].append(
                        corr[i, j]
                    )

        print(
            f"   {stop}/{n_windows}",
            end="\r"
        )

    print()

    mean_matrix = accumulator / n_windows

    return (
        mean_matrix,
        edge_values
    )

# ============================================================
# Discover sessions
# ============================================================

sessions = list(
    find_sessions_with_windows(outputs_root)
)

print(
    f"Found {len(sessions)} sessions across "
    f"{len(set(s for s, _, _, _ in sessions))} subjects"
)

subject_spike_matrices = []

subject_baseline_matrices = []

all_spike_edges = {}

all_baseline_edges = {}

for i in range(N_CHANNELS):

    for j in range(i + 1, N_CHANNELS):

        all_spike_edges[(i, j)] = []

        all_baseline_edges[(i, j)] = []


# ============================================================
# Process every subject/session
# ============================================================

print("\nPooling subject connectivity matrices...")
MIN_WINDOWS = 10
skipped_sessions = []
for subject, session, spike_file, baseline_file in sessions:

    spike_windows = np.load(
        spike_file,
        mmap_mode="r"
    )

    baseline_windows = np.load(
        baseline_file,
        mmap_mode="r"
    )

    n_spike = len(spike_windows)
    n_baseline = len(baseline_windows)

    # Skip sessions with too few windows
    if n_spike < MIN_WINDOWS or n_baseline < MIN_WINDOWS:

        print(
            f"Skipping {subject}/{session} "
            f"(Spike={n_spike}, Baseline={n_baseline})"
        )

        skipped_sessions.append(
            (subject, session, n_spike, n_baseline)
        )

        continue

    spike_matrix, spike_edges = average_connectivity_matrix(
        spike_windows,
        "Spike",
        subject,
        session
    )

    baseline_matrix, baseline_edges = average_connectivity_matrix(
        baseline_windows,
        "Baseline",
        subject,
        session
    )

    subject_spike_matrices.append({
        "subject": subject,
        "session": session,
        "matrix": spike_matrix
    })

    subject_baseline_matrices.append({
        "subject": subject,
        "session": session,
        "matrix": baseline_matrix
    })

    # ============================================================
    # Save individual session connectivity matrices
    # ============================================================

    SESSION_DIR = multi_dir / "session_matrices"
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
     SESSION_DIR / f"{subject}_{session}_spike.npy",
     spike_matrix
    )

    np.save(
     SESSION_DIR / f"{subject}_{session}_baseline.npy",
     baseline_matrix
    )

    for key in spike_edges:

        all_spike_edges[key].extend(
            spike_edges[key]
        )

        all_baseline_edges[key].extend(
            baseline_edges[key]
        )

# ============================================================
# Pool subject connectivity matrices
# ============================================================

print("\nPooling subject connectivity matrices...")

pooled_spike = np.mean(
    [x["matrix"] for x in subject_spike_matrices],
    axis=0
)

pooled_baseline = np.mean(
    [x["matrix"] for x in subject_baseline_matrices],
    axis=0
)

difference_matrix = pooled_spike - pooled_baseline


# ============================================================
# Save pooled matrices
# ============================================================

pd.DataFrame(
    pooled_spike,
    index=EEG_CHANNEL_NAMES,
    columns=EEG_CHANNEL_NAMES
).to_csv(
    multi_dir / "pooled_spike_connectivity.csv"
)

pd.DataFrame(
    pooled_baseline,
    index=EEG_CHANNEL_NAMES,
    columns=EEG_CHANNEL_NAMES
).to_csv(
    multi_dir / "pooled_baseline_connectivity.csv"
)

pd.DataFrame(
    difference_matrix,
    index=EEG_CHANNEL_NAMES,
    columns=EEG_CHANNEL_NAMES
).to_csv(
    multi_dir / "difference_connectivity.csv"
)

# ============================================================
# Statistical comparison
# ============================================================
print("\nRunning session-level statistical tests...")

rows = []

for i in range(N_CHANNELS):

    for j in range(i + 1, N_CHANNELS):

        spike_values = np.asarray([
            x["matrix"][i, j]
            for x in subject_spike_matrices
        ])

        baseline_values = np.asarray([
            x["matrix"][i, j]
            for x in subject_baseline_matrices
        ])

        if len(spike_values) < 3:
            continue

        # Paired Wilcoxon signed-rank test
        try:
            statistic, p = stats.wilcoxon(
                spike_values,
                baseline_values,
                zero_method="wilcox",
                alternative="two-sided"
            )
        except ValueError:
            statistic = np.nan
            p = 1.0

        difference = spike_values - baseline_values

        effect_size = (
            np.mean(difference)
            /
            (np.std(difference, ddof=1) + 1e-12)
        )

        rows.append({

            "channel_1":
                EEG_CHANNEL_NAMES[i],

            "channel_2":
                EEG_CHANNEL_NAMES[j],

            "spike_mean":
                spike_values.mean(),

            "baseline_mean":
                baseline_values.mean(),

            "difference":
                difference.mean(),

            "wilcoxon_stat":
                statistic,

            "p_value":
                p,

            "effect_size":
                effect_size

        })

stats_df = pd.DataFrame(rows)

reject, p_fdr, _, _ = multipletests(

    stats_df["p_value"],

    alpha=0.05,

    method="fdr_bh"

)

stats_df["p_fdr"] = p_fdr
stats_df["significant"] = reject

# ============================================================
# Subject consistency
# ============================================================

consistency_rows = []

for i in range(N_CHANNELS):

    for j in range(i + 1, N_CHANNELS):

        spike_subject_means = []

        baseline_subject_means = []

        for s in subject_spike_matrices:

            spike_subject_means.append(
                s["matrix"][i, j]
            )

        for s in subject_baseline_matrices:

            baseline_subject_means.append(
                s["matrix"][i, j]
            )

        spike_subject_means = np.asarray(
            spike_subject_means
        )

        baseline_subject_means = np.asarray(
            baseline_subject_means
        )

        pooled_direction = np.sign(

            spike_subject_means.mean()
            -
            baseline_subject_means.mean()

        )

        subject_direction = np.sign(

            spike_subject_means
            -
            baseline_subject_means

        )

        n_agree = np.sum(
            subject_direction == pooled_direction
        )

        consistency_rows.append({

            "channel_1":
                EEG_CHANNEL_NAMES[i],

            "channel_2":
                EEG_CHANNEL_NAMES[j],

            "subjects":
                len(subject_direction),

            "agree":
                int(n_agree),

            "consistency":
                n_agree / len(subject_direction)

        })


consistency_df = pd.DataFrame(
    consistency_rows
)

merged = stats_df.merge(

    consistency_df,

    on=["channel_1",
        "channel_2"]

)

merged = merged.sort_values(

    "effect_size",

    key=lambda s: s.abs(),

    ascending=False

)

merged.to_csv(

    multi_dir /
    "connectivity_multisubject_stats.csv",

    index=False

)

print(
    f"\n"
    f"{merged['significant'].sum()} / "
    f"{len(merged)} connections "
    f"significant after FDR correction"
)

print("\nTop 10 connections:")

print(

    merged.head(10)[[

        "channel_1",
        "channel_2",
        "difference",
        "effect_size",
        "p_fdr",
        "consistency"

    ]].to_string(index=False)

)


# ============================================================
# Connectivity heatmaps
# ============================================================

print("\nGenerating figures...")

save_heatmap(
    pooled_spike,
    title="Pooled Spike Connectivity",
    filename="spike_connectivity_heatmap.png",
    cmap="viridis",
    vmin=0,
    vmax=1
)

save_heatmap(
    pooled_baseline,
    title="Pooled Baseline Connectivity",
    filename="baseline_connectivity_heatmap.png",
    cmap="viridis",
    vmin=0,
    vmax=1
)

diff_limit = np.max(np.abs(difference_matrix))

save_heatmap(
    difference_matrix,
    title="Spike - Baseline Connectivity",
    filename="difference_connectivity_heatmap.png",
    cmap="RdBu_r",
    vmin=-diff_limit,
    vmax=diff_limit
)

# ============================================================
# Significant connections only
# ============================================================

significant_matrix = np.zeros(
    (N_CHANNELS, N_CHANNELS),
    dtype=np.float64
)

for _, row in merged.iterrows():

    if not row["significant"]:
        continue

    ch1 = EEG_CHANNEL_NAMES.index(row["channel_1"])
    ch2 = EEG_CHANNEL_NAMES.index(row["channel_2"])

    significant_matrix[ch1, ch2] = row["difference"]
    significant_matrix[ch2, ch1] = row["difference"]

limit = np.max(np.abs(significant_matrix))

if limit == 0:
    limit = 1e-6

save_heatmap(
    significant_matrix,
    title="Significant Connectivity Changes (FDR < 0.05)",
    filename="significant_connectivity_heatmap.png",
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit
)



# ============================================================
# Save strongest connections
# ============================================================

top20 = merged.head(20)

top20.to_csv(
    multi_dir / "top20_connectivity_changes.csv",
    index=False
)


# ============================================================
# Save metadata
# ============================================================
subjects_used = sorted(
    set(
        x["subject"]
        for x in subject_spike_matrices
    )
)

run_info = {

    "subjects":
        len(subjects_used),

    "sessions":
        len(subject_spike_matrices),

    "channels":

        N_CHANNELS,

    "connections":

        int(N_CHANNELS * (N_CHANNELS - 1) / 2),

    "batch_size":

        BATCH_SIZE,

    "significant_connections":

        int(
            merged["significant"].sum()
        )

}

with open(
    multi_dir / "run_info.json",
    "w"
) as f:

    json.dump(
        run_info,
        f,
        indent=4
    )



# ============================================================
# Finished
# ============================================================

print("\n==========================================")
print(" Multi-subject connectivity analysis done")
print("==========================================")

print(f"Subjects : {run_info['subjects']}")
print(f"Sessions : {run_info['sessions']}")
print(f"Channels : {run_info['channels']}")
print(f"Connections tested : {run_info['connections']}")
print(f"Significant (FDR) : {run_info['significant_connections']}")

print("\nOutputs saved to")

print(multi_dir)

print("\nGenerated files")

print("  pooled_spike_connectivity.csv")
print("  pooled_baseline_connectivity.csv")
print("  difference_connectivity.csv")

print("  connectivity_multisubject_stats.csv")
print("  top20_connectivity_changes.csv")

print("  spike_connectivity_heatmap.png")
print("  baseline_connectivity_heatmap.png")
print("  difference_connectivity_heatmap.png")
print("  significant_connectivity_heatmap.png")

print("  run_info.json")