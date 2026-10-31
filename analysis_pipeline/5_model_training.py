import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
 
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,confusion_matrix,ConfusionMatrixDisplay
from config import output_dir,bands as BANDS
 
eeg_channel_names = ["C3", "C4", "O1", "O2", "A1", "A2", "Cz", "F3", "F4", "F7", "F8", "Fz",
    "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6", ]
 
reference_channels = {"A1","A2"}
 
analysis_dir = output_dir/"analysis"
model_dir = output_dir/"model"
model_dir.mkdir(exist_ok = True)
 
print("Loading feature matrix and labels")
X = pd.read_csv(analysis_dir/"feature_matrix.csv")
y = np.load(analysis_dir/"labels.npy")
 
n_spike = int((y ==1).sum())
n_base = int((y ==0).sum())
 
print(f"X: {X.shape}, y: {y.shape}")
print(f"Class counts -> spike: {n_spike}, baseline: {n_base}  (ratio {n_spike / n_base:.1f}:1)")
 
 
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,stratify = y , random_state = 42)
 
print(f"\nTrain: {X_train.shape[0]} windows, Test: {X_test.shape[0]} windows (80/20 split, stratified)")
print(f"Train class counts: spike={int((y_train == 1).sum())}, baseline={int((y_train == 0).sum())}")
print(f"Test class counts:  spike={int((y_test == 1).sum())}, baseline={int((y_test == 0).sum())}")
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
 
print("\n Training logistic regression ...")
logistic_model = LogisticRegression(class_weight = "balanced",max_iter = 2000, random_state = 42)
logistic_model.fit(X_train_scaled,y_train)
 
 
print("|n Training random forest")
rf_model = RandomForestClassifier(n_estimators=300, class_weight="balanced",random_state = 42,n_jobs=-1)
rf_model.fit(X_train,y_train)
 
 
models = {"logistic_regression":(logistic_model,X_test_scaled),"random_forest": (rf_model,X_test)}
 
train_data = {"logistic_regression": X_train_scaled, "random_forest": X_train}
 
print("\nTuning decision thresholds via 5-fold CV on the training set (test set untouched)...")
tuned_thresholds = {}
for name, (model, _) in models.items():
    X_tr = train_data[name]
    cv_proba = cross_val_predict(model, X_tr, y_train, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, cv_proba)
    gmean = np.sqrt(tpr * (1 - fpr))
    best_idx = np.argmax(gmean)
    best_threshold = thresholds[best_idx]
    tuned_thresholds[name] = float(best_threshold)
    print(f"  {name}: tuned threshold = {best_threshold:.3f} (default is 0.5)")
 
 
 
results = {}
results_summary_tuned={}
 
for name,(model,X_te) in models.items():
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:,1]
 
 
    report = classification_report(y_test,y_pred,target_names=['baseline','spike'],output_dict=True)
    auc = roc_auc_score(y_test,y_proba)
 
    print("-- Default threshold (0.5) --")
    print(classification_report(y_test, y_pred, target_names=["baseline", "spike"]))
    print(f"ROC-AUC: {auc:.4f}")
 
    results[name] = {
        "roc_auc": auc,
        "baseline_precision": report["baseline"]["precision"],
        "baseline_recall": report["baseline"]["recall"],
        "baseline_f1": report["baseline"]["f1-score"],
        "spike_precision": report["spike"]["precision"],
        "spike_recall": report["spike"]["recall"],
        "spike_f1": report["spike"]["f1-score"],
        "accuracy": report["accuracy"],
    }
 
 
    #Confusion matrix plot (default threshold)
 
    cm = confusion_matrix(y_test,y_pred)
    fig,ax = plt.subplots(figsize=(4.5,4.5))
    ConfusionMatrixDisplay(cm,display_labels=["baseline","spike"]).plot(ax=ax,cmap = "Blues", colorbar=False)
    ax.set_title(f"{name.replace('_', ' ').title()} -- Confusion Matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig(model_dir/f"confusion_matrix_{name}.png",dpi=150)
    plt.close()
 
    # --- same model, tuned threshold (probabilities y_proba are unchanged --
    # only the yes/no cutoff applied to them changes) ---
    threshold = tuned_thresholds[name]
    y_pred_tuned = (y_proba >= threshold).astype(int)
    report_tuned = classification_report(
        y_test, y_pred_tuned, target_names=["baseline", "spike"], output_dict=True
    )
 
    print(f"\n-- Tuned threshold ({threshold:.3f}) --")
    print(classification_report(y_test, y_pred_tuned, target_names=["baseline", "spike"]))
 
    results_summary_tuned[name] = {
        "threshold": threshold,
        "roc_auc": auc,
        "baseline_precision": report_tuned["baseline"]["precision"],
        "baseline_recall": report_tuned["baseline"]["recall"],
        "baseline_f1": report_tuned["baseline"]["f1-score"],
        "spike_precision": report_tuned["spike"]["precision"],
        "spike_recall": report_tuned["spike"]["recall"],
        "spike_f1": report_tuned["spike"]["f1-score"],
        "accuracy": report_tuned["accuracy"],
    }
 
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay(cm_tuned, display_labels=["baseline", "spike"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name.replace('_', ' ').title()} -- Confusion Matrix (threshold={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(model_dir / f"confusion_matrix_{name}_tuned.png", dpi=150)
    plt.close()
 
results_df = pd.DataFrame(results).T
results_df.to_csv(model_dir / "model_comparison.csv")
print(f"\nModel comparison (default threshold=0.5):\n{results_df.round(3).to_string()}")
 
results_df_tuned = pd.DataFrame(results_summary_tuned).T
results_df_tuned.to_csv(model_dir / "model_comparison_tuned.csv")
print(f"\nModel comparison (tuned threshold):\n{results_df_tuned.round(3).to_string()}")
 
 
 
 
 
#ROC curves, both models on one plot
 
plt.figure(figsize=(6,5))
for name, (model, X_te) in models.items():
    y_proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = results[name]["roc_auc"]
    plt.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve: Spike vs Baseline Classification")
plt.legend()
plt.tight_layout()
plt.savefig(model_dir / "roc_curves.png", dpi=150)
plt.close()
 
 
 
#Feature importance 
 
 
def readable_feature_name(col):
    """'power_alpha_ch16' -> 'power_alpha_T5'"""
    if "_ch" in col:
        base, ch_idx = col.rsplit("_ch", 1)
        return f"{base}_{eeg_channel_names[int(ch_idx)]}"
    return col
 
 
rf_importance = pd.DataFrame({
    "feature": X.columns,
    "feature_readable": [readable_feature_name(c) for c in X.columns],
    "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False)
rf_importance.to_csv(model_dir / "rf_feature_importance.csv", index=False)
 
print("\nTop 15 features by random forest importance:")
print(rf_importance.head(15)[["feature_readable", "importance"]].to_string(index=False))
 
top_n = 20
plt.figure(figsize=(9, 6))
top_feats = rf_importance.head(top_n).iloc[::-1]  # reverse so most important is on top
plt.barh(top_feats["feature_readable"], top_feats["importance"], color="steelblue")
plt.xlabel("Random forest importance")
plt.title(f"Top {top_n} Features (Random Forest)")
plt.tight_layout()
plt.savefig(model_dir / "feature_importance.png", dpi=150)
plt.close()
 
# Cross-check against step 4's effect-size ranking, if available
stats_path = analysis_dir / "feature_stats.csv"
if stats_path.exists():
    step4_stats = pd.read_csv(stats_path)
    step4_top15 = set(step4_stats.reindex(
        step4_stats["effect_size_r"].abs().sort_values(ascending=False).index
    ).head(15)["feature"])
    model_top15 = set(rf_importance.head(15)["feature"])
    overlap = step4_top15 & model_top15
    print(f"\nOverlap between model's top-15 features and step 4's top-15 effect-size features: "
          f"{len(overlap)}/15")
    if overlap:
        print("Shared features:", ", ".join(sorted(readable_feature_name(f) for f in overlap)))
 
 
 
 
 #save models
 
joblib.dump(logistic_model, model_dir / "logistic_regression.joblib")
joblib.dump(rf_model, model_dir / "random_forest.joblib")
joblib.dump(scaler, model_dir / "scaler.joblib")
 
with open(model_dir / "run_info.json", "w") as f:
    json.dump({
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts": {"spike": n_spike, "baseline": n_base},
        "results_default_threshold": results,
        "tuned_thresholds": tuned_thresholds,
        "results_tuned_threshold": results_summary_tuned,
    }, f, indent=2)
 
print(f"\nSaved to: {model_dir}")
print("  - logistic_regression.joblib, random_forest.joblib, scaler.joblib")
print("  - model_comparison.csv              (default threshold=0.5)")
print("  - model_comparison_tuned.csv        (per-model tuned threshold)")
print("  - run_info.json")
print("  - confusion_matrix_logistic_regression.png, confusion_matrix_random_forest.png")
print("  - confusion_matrix_logistic_regression_tuned.png, confusion_matrix_random_forest_tuned.png")
print("  - roc_curves.png")
print("  - rf_feature_importance.csv, feature_importance.png")
print("\nDone!")