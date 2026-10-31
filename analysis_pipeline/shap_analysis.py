
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
 
project_root = Path("/mnt/data/shyam/sanjana/MROP_NCC_Consciousness")
multi_dir = project_root / "outputs" / "multi_subject"
out_dir = multi_dir / "shap"
out_dir.mkdir(parents=True, exist_ok=True)
 
N_SAMPLE = 3000  # stratified subsample size for SHAP value computation
RANDOM_STATE = 42
 
EEG_CHANNEL_NAMES = [
    "C3", "C4", "O1", "O2", "A1", "A2", "Cz", "F3", "F4", "F7", "F8", "Fz",
    "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6",
]
 
 
def readable_feature_name(col):
    """'power_alpha_ch16' -> 'power_alpha_P4'"""
    if "_ch" in col:
        base, ch_idx = col.rsplit("_ch", 1)
        try:
            return f"{base}_{EEG_CHANNEL_NAMES[int(ch_idx)]}"
        except (ValueError, IndexError):
            return col
    return col
 
 
print("Loading saved model and feature matrix...")
model = joblib.load(multi_dir / "final_random_forest_all_subjects.joblib")
X = pd.read_csv(multi_dir / "combined_feature_matrix.csv")
y = np.load(multi_dir / "combined_labels.npy")
print(f"X: {X.shape}, y: {y.shape}")
print(f"Class counts -> spike: {(y == 1).sum()}, baseline: {(y == 0).sum()}")
 
readable_cols = [readable_feature_name(c) for c in X.columns]
 
# Stratified subsample for speed
rng = np.random.default_rng(RANDOM_STATE)
idx_spike = np.where(y == 1)[0]
idx_base = np.where(y == 0)[0]
n_each = min(N_SAMPLE // 2, len(idx_spike), len(idx_base))
sample_idx = np.concatenate([
    rng.choice(idx_spike, size=n_each, replace=False),
    rng.choice(idx_base, size=n_each, replace=False),
])
X_sample = X.iloc[sample_idx].reset_index(drop=True)
y_sample = y[sample_idx]
print(f"Using stratified subsample of {len(X_sample)} windows for SHAP "
      f"({n_each} spike, {n_each} baseline)")
 
print("\nComputing SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
 
# shap.TreeExplainer on a binary RandomForestClassifier returns either a
# list [class0_array, class1_array] or a single 3D array depending on the
# installed shap version -- handle both, and take the "spike" (class 1) view.
if isinstance(shap_values, list):
    shap_spike = shap_values[1]
else:
    shap_spike = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
 
print(f"SHAP values shape: {shap_spike.shape}")
 
# --- Mean absolute SHAP value per feature (overall importance, with direction info) ---
mean_abs_shap = np.abs(shap_spike).mean(axis=0)
mean_shap = shap_spike.mean(axis=0)  # signed: positive = pushes toward spike on average
 
shap_summary_df = pd.DataFrame({
    "feature": X.columns,
    "feature_readable": readable_cols,
    "mean_abs_shap": mean_abs_shap,
    "mean_signed_shap": mean_shap,
}).sort_values("mean_abs_shap", ascending=False)
shap_summary_df.to_csv(out_dir / "shap_feature_summary.csv", index=False)
 
print("\nTop 15 features by mean |SHAP value|:")
print(shap_summary_df.head(15)[["feature_readable", "mean_abs_shap", "mean_signed_shap"]]
      .to_string(index=False))
 
# --- Bar plot: top 20 features by mean |SHAP|, colored by direction ---
top_n = 20
top = shap_summary_df.head(top_n).iloc[::-1]
colors = ["crimson" if v > 0 else "steelblue" for v in top["mean_signed_shap"]]
 
plt.figure(figsize=(9, 7))
plt.barh(top["feature_readable"], top["mean_abs_shap"], color=colors)
plt.xlabel("Mean |SHAP value| (impact on spike-probability prediction)")
plt.title(f"Top {top_n} Features by SHAP Importance\n"
          f"(red = higher value pushes toward SPIKE, blue = pushes toward BASELINE)")
plt.tight_layout()
plt.savefig(out_dir / "shap_bar_top20.png", dpi=150)
plt.close()
print(f"\nSaved: {out_dir / 'shap_bar_top20.png'}")
 
# --- Beeswarm summary plot (standard SHAP visualization) ---
plt.figure(figsize=(9, 7))
shap.summary_plot(shap_spike, X_sample, feature_names=readable_cols,
                   max_display=20, show=False)
plt.tight_layout()
plt.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_dir / 'shap_beeswarm.png'}")
 
print(f"\nAll SHAP outputs saved to: {out_dir}")
print("  - shap_feature_summary.csv  (all features, mean |SHAP| + signed mean)")
print("  - shap_bar_top20.png        (top 20, colored by direction)")
print("  - shap_beeswarm.png         (standard SHAP distribution plot)")