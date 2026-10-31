"""
==============================================================
Graph Theory Analysis
Spike vs Baseline Functional Connectivity

Author: Atharva Kadam
==============================================================
"""

import os
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import networkx as nx

from scipy import stats
from scipy.stats import wilcoxon

from statsmodels.stats.multitest import multipletests

# ============================================================
# Configuration
# ============================================================

BASE_DIR = "/mnt/data/shyam/sanjana/MROP_NCC_Consciousness"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "graph_theory"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

CONNECTIVITY_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "multi_subject"
)

EDGE_THRESHOLD = 0.30

FIG_DPI = 300


EEG_CHANNEL_NAMES = [

    "C3","C4",

    "O1","O2",

    "A1","A2",

    "Cz",

    "F3","F4",

    "F7","F8",

    "Fz",

    "Fp1","Fp2","Fpz",

    "P3","P4","Pz",

    "T3","T4","T5","T6"

]

N_CHANNELS = len(EEG_CHANNEL_NAMES)

NODE_POSITIONS = {

    "Fp1":(-2,4),
    "Fpz":(0,4),
    "Fp2":(2,4),

    "F7":(-4,3),
    "F3":(-2,3),
    "Fz":(0,3),
    "F4":(2,3),
    "F8":(4,3),

    "C3":(-2,1),
    "Cz":(0,1),
    "C4":(2,1),

    "T3":(-4,1),
    "T4":(4,1),

    "P3":(-2,-1),
    "Pz":(0,-1),
    "P4":(2,-1),

    "T5":(-4,-1),
    "T6":(4,-1),

    "O1":(-2,-3),
    "O2":(2,-3),

    "A1":(-5,0),
    "A2":(5,0)

}


def load_session_matrices():

    session_dir = os.path.join(
        CONNECTIVITY_DIR,
        "session_matrices"
    )

    spike_files = sorted(

        f for f in os.listdir(session_dir)

        if f.endswith("_spike.npy")

    )

    baseline_files = sorted(

        f for f in os.listdir(session_dir)

        if f.endswith("_baseline.npy")

    )

    sessions = []

    for spike_file, baseline_file in zip(

        spike_files,

        baseline_files

    ):

        spike = np.load(

            os.path.join(

                session_dir,

                spike_file

            )

        )

        baseline = np.load(

            os.path.join(

                session_dir,

                baseline_file

            )

        )

        session_name = spike_file.replace(

            "_spike.npy",

            ""

        )

        sessions.append({

            "name":session_name,

            "spike":spike,

            "baseline":baseline

        })

    return sessions


def matrix_to_graph(matrix):

    G = nx.Graph()

    for node in EEG_CHANNEL_NAMES:

        G.add_node(node)

    for i in range(N_CHANNELS):

        for j in range(i+1,N_CHANNELS):

            weight = matrix[i,j]

            if abs(weight) >= EDGE_THRESHOLD:

                G.add_edge(

                    EEG_CHANNEL_NAMES[i],

                    EEG_CHANNEL_NAMES[j],

                    weight=float(weight)

                )

    return G


print()

print("="*60)
print(" Loading session connectivity matrices")
print("="*60)

sessions = load_session_matrices()

print()

print(

    f"Loaded {len(sessions)} sessions"

)

# ============================================================
# Load significant connectivity edges
# ============================================================

STATS_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "multi_subject",
    "connectivity_multisubject_stats.csv"
)

edge_stats = pd.read_csv(STATS_FILE)

significant_edges = edge_stats[
    edge_stats["p_fdr"] < 0.05
].copy()

print(f"Loaded {len(significant_edges)} significant edges.")



# ============================================================
# Graph metric helper functions
# ============================================================

def graph_density(G):

    return nx.density(G)


def graph_global_efficiency(G):

    if G.number_of_edges() == 0:
        return np.nan

    return nx.global_efficiency(G)


def graph_clustering(G):

    if G.number_of_edges() == 0:
        return np.nan

    return nx.average_clustering(
        G,
        weight="weight"
    )


def graph_path_length(G):

    if G.number_of_edges() == 0:
        return np.nan

    if nx.is_connected(G):

        return nx.average_shortest_path_length(G)

    largest = max(
        nx.connected_components(G),
        key=len
    )

    H = G.subgraph(largest).copy()

    if H.number_of_nodes() < 2:
        return np.nan

    return nx.average_shortest_path_length(H)


def node_strength(G):

    strength = {}

    for node in G.nodes():

        total = 0.0

        for _, _, data in G.edges(node, data=True):

            total += abs(data["weight"])

        strength[node] = total

    return strength


def safe_betweenness(G):

    if G.number_of_edges() == 0:

        return {
            node: np.nan
            for node in G.nodes()
        }

    return nx.betweenness_centrality(
        G,
        weight="weight",
        normalized=True
    )


def safe_eigenvector(G):

    if G.number_of_edges() == 0:

        return {
            node: np.nan
            for node in G.nodes()
        }

    try:

        return nx.eigenvector_centrality(
            G,
            weight="weight",
            max_iter=1000
        )

    except nx.PowerIterationFailedConvergence:

        return {
            node: np.nan
            for node in G.nodes()
        }
    

# ============================================================
# Compute graph metrics
# ============================================================

print("\nComputing graph metrics...\n")

global_results = []

node_results = []

for session in sessions:

    print(session["name"])

    spike_graph = matrix_to_graph(
        session["spike"]
    )

    baseline_graph = matrix_to_graph(
        session["baseline"]
    )

    #####################################################
    # Global metrics
    #####################################################

    spike_global = {

        "session": session["name"],

        "condition": "Spike",

        "density":
            graph_density(spike_graph),

        "global_efficiency":
            graph_global_efficiency(spike_graph),

        "clustering":
            graph_clustering(spike_graph),

        "path_length":
            graph_path_length(spike_graph)

    }

    baseline_global = {

        "session": session["name"],

        "condition": "Baseline",

        "density":
            graph_density(baseline_graph),

        "global_efficiency":
            graph_global_efficiency(baseline_graph),

        "clustering":
            graph_clustering(baseline_graph),

        "path_length":
            graph_path_length(baseline_graph)

    }

    global_results.append(
        spike_global
    )

    global_results.append(
        baseline_global
    )

    #####################################################
    # Node metrics
    #####################################################

    spike_strength = node_strength(
        spike_graph
    )

    baseline_strength = node_strength(
        baseline_graph
    )

    spike_between = safe_betweenness(
        spike_graph
    )

    baseline_between = safe_betweenness(
        baseline_graph
    )

    spike_eigen = safe_eigenvector(
        spike_graph
    )

    baseline_eigen = safe_eigenvector(
        baseline_graph
    )

    for node in EEG_CHANNEL_NAMES:

        node_results.append({

            "session":
                session["name"],

            "condition":
                "Spike",

            "electrode":
                node,

            "strength":
                spike_strength[node],

            "betweenness":
                spike_between[node],

            "eigenvector":
                spike_eigen[node]

        })

        node_results.append({

            "session":
                session["name"],

            "condition":
                "Baseline",

            "electrode":
                node,

            "strength":
                baseline_strength[node],

            "betweenness":
                baseline_between[node],

            "eigenvector":
                baseline_eigen[node]

        })

print("\nFinished computing graph metrics.")


global_df = pd.DataFrame(global_results)

node_df = pd.DataFrame(node_results)

print()
print(global_df.head())
print()
print(node_df.head())

# ============================================================
# Statistical Analysis
# ============================================================

print("\n" + "=" * 60)
print(" Statistical Analysis")
print("=" * 60)

global_metrics = [
    "density",
    "global_efficiency",
    "clustering",
    "path_length"
]

global_stats = []

for metric in global_metrics:

    spike = global_df[
        global_df["condition"] == "Spike"
    ][metric].values

    baseline = global_df[
        global_df["condition"] == "Baseline"
    ][metric].values

    valid = ~(np.isnan(spike) | np.isnan(baseline))

    spike = spike[valid]
    baseline = baseline[valid]

    statistic, pvalue = wilcoxon(
        spike,
        baseline
    )

    global_stats.append({

        "metric": metric,

        "n_sessions": len(spike),

        "statistic": statistic,

        "p_value": pvalue

    })

global_stats_df = pd.DataFrame(global_stats)

reject, p_corrected, _, _ = multipletests(

    global_stats_df["p_value"],

    method="fdr_bh"

)

global_stats_df["p_fdr"] = p_corrected
global_stats_df["significant"] = reject

print()
print(global_stats_df)


# ============================================================
# Node-wise Statistics
# ============================================================

node_metrics = [

    "strength",

    "betweenness",

    "eigenvector"

]

node_stats = []

for metric in node_metrics:

    for electrode in EEG_CHANNEL_NAMES:

        spike = node_df[
            (node_df["condition"]=="Spike") &
            (node_df["electrode"]==electrode)
        ][metric].values

        baseline = node_df[
            (node_df["condition"]=="Baseline") &
            (node_df["electrode"]==electrode)
        ][metric].values

        valid = ~(np.isnan(spike) | np.isnan(baseline))

        spike = spike[valid]
        baseline = baseline[valid]

        if len(spike) < 2:

            continue

        statistic, pvalue = wilcoxon(
            spike,
            baseline
        )

        node_stats.append({

            "electrode": electrode,

            "metric": metric,

            "n_sessions": len(spike),

            "statistic": statistic,

            "p_value": pvalue

        })

node_stats_df = pd.DataFrame(node_stats)

reject, p_corrected, _, _ = multipletests(

    node_stats_df["p_value"],

    method="fdr_bh"

)

node_stats_df["p_fdr"] = p_corrected
node_stats_df["significant"] = reject

print()
print("\nSignificant node metrics:\n")

print(

    node_stats_df[
        node_stats_df["significant"]
    ].sort_values("p_fdr")

)


# ============================================================
# Save Results
# ============================================================

global_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "global_metrics.csv"
    ),

    index=False

)

node_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "node_metrics.csv"
    ),

    index=False

)

global_stats_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "global_statistics.csv"
    ),

    index=False

)

node_stats_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "node_statistics.csv"
    ),

    index=False

)

print("\nResults saved to:")

print(OUTPUT_DIR)


# ============================================================
# Boxplots of Global Metrics
# ============================================================

print("\nCreating boxplots...")

for metric in [

    "density",

    "global_efficiency",

    "clustering",

    "path_length"

]:

    plt.figure(figsize=(5,5))

    spike = global_df[
        global_df["condition"]=="Spike"
    ][metric]

    baseline = global_df[
        global_df["condition"]=="Baseline"
    ][metric]

    plt.boxplot(

        [spike, baseline],

        labels=["Spike","Baseline"]

    )

    plt.ylabel(metric.replace("_"," ").title())

    plt.title(metric.replace("_"," ").title())

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            f"{metric}_boxplot.png"

        ),

        dpi=FIG_DPI

    )

    plt.close()

print("Boxplots saved.")

# ============================================================
# Save Run Information
# ============================================================

run_info = {

    "n_sessions": len(sessions),

    "n_channels": N_CHANNELS,

    "edge_threshold": EDGE_THRESHOLD,

    "global_metrics": [

        "density",

        "global_efficiency",

        "clustering",

        "path_length"

    ],

    "node_metrics": [

        "strength",

        "betweenness",

        "eigenvector"

    ],

    "significant_global_metrics":

        global_stats_df[
            global_stats_df["significant"]
        ]["metric"].tolist(),

    "n_significant_node_metrics":

        int(node_stats_df["significant"].sum())

}

with open(

    os.path.join(

        OUTPUT_DIR,

        "run_info.json"

    ),

    "w"

) as f:

    json.dump(

        run_info,

        f,

        indent=4

    )

print("Run information saved.")


# ============================================================
# Summary
# ============================================================

print("\n" + "="*60)
print("GRAPH THEORY ANALYSIS COMPLETE")
print("="*60)

print(f"Sessions analysed          : {len(sessions)}")

print(f"Global metrics tested      : {len(global_stats_df)}")

print(f"Node metrics tested        : {len(node_stats_df)}")

print(f"Significant global metrics : {int(global_stats_df['significant'].sum())}")

print(f"Significant node metrics   : {int(node_stats_df['significant'].sum())}")

print(f"\nResults saved to:\n{OUTPUT_DIR}")



# ============================================================
# Average Connectivity Networks
# ============================================================

print("\nComputing average connectivity matrices...")

spike_stack = np.stack(
    [session["spike"] for session in sessions],
    axis=0
)

baseline_stack = np.stack(
    [session["baseline"] for session in sessions],
    axis=0
)

avg_spike = np.mean(
    spike_stack,
    axis=0
)

avg_baseline = np.mean(
    baseline_stack,
    axis=0
)

difference = avg_spike - avg_baseline



# ============================================================
# Draw Brain Network
# ============================================================

# ============================================================
# Plot Significant Difference Network
# ============================================================
def draw_significant_network():

    G = nx.Graph()

    for node in EEG_CHANNEL_NAMES:
        G.add_node(node)

    edge_colors = []
    edge_widths = []

    for _, row in significant_edges.iterrows():

        node1 = row["channel_1"]
        node2 = row["channel_2"]

        diff = row["difference"]

        G.add_edge(
            node1,
            node2,
            weight=diff
        )

        edge_widths.append(
            3 + 20 * abs(diff)
        )

        if diff > 0:
            edge_colors.append("red")
        else:
            edge_colors.append("blue")

    plt.figure(figsize=(8,8))

    # ---------- node strength ----------
    strength = {}

    for node in EEG_CHANNEL_NAMES:

        total = 0

        for _, _, data in G.edges(node, data=True):

            total += abs(data["weight"])

        strength[node] = total

    node_sizes = [

        700 + 1500 * strength[node]

        for node in G.nodes()

    ]

    nx.draw_networkx_nodes(

        G,

        NODE_POSITIONS,

        node_size=node_sizes,

        node_color="lightblue",

        edgecolors="black",

        linewidths=1.5

    )

    nx.draw_networkx_labels(

        G,

        NODE_POSITIONS,

        font_size=10,

        font_weight="bold"

    )

    nx.draw_networkx_edges(

        G,

        NODE_POSITIONS,

        edge_color=edge_colors,

        width=edge_widths,

        alpha=0.9

    )

    plt.title(
        "Significant Connectivity Changes\nSpike vs Baseline"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            "significant_difference_network.png"

        ),

        dpi=FIG_DPI

    )

    plt.close()

# ============================================================
# Save Network Figures
# ============================================================

# draw_network(

#     avg_spike,

#     "Average Spike Connectivity",

#     "average_spike_network.png"

# )

# draw_network(

#     avg_baseline,

#     "Average Baseline Connectivity",

#     "average_baseline_network.png"

# )

draw_significant_network()

print("\nNetwork figures saved.")