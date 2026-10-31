import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
 
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
 
eeg_channel_names = [
    "C3", "C4", "O1", "O2", "A1", "A2", "Cz", "F3", "F4", "F7", "F8", "Fz",
    "Fp1", "Fp2", "Fpz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6",
]
 
subjects = [
    "sub-I0002150000008",
    "sub-I0002150000012",
    "sub-I0002150000014",
    "sub-I0002150000031",
]
 
project_root = Path("/mnt/data/shyam/sanjana/MROP_NCC_Consciousness")
multi_dir = project_root / "outputs" / "multi_subject"
multi_dir.mkdir(parents=True, exist_ok=True)
 
 
print("Loading per-subject feature matrices from step 4 (ALL available sessions per subject)...")
print("Applying per-session baseline normalization before pooling (see note below)...")
all_X = []
all_y = []
all_subjects = []
 
MIN_RELIABLE_N = 30  # folds with fewer test windows than this are reported separately, not blended into the headline average
 
for sub in subjects:
    sub_output_dir = project_root / "outputs" / sub
    # Glob every session folder this subject has processed output for, not
    # just ses-1 -- lets richer subjects (e.g. sub-031 with many usable
    # sessions) contribute far more data than a single session would.
    session_dirs = sorted(sub_output_dir.glob("ses-*")) if sub_output_dir.exists() else []
 
    per_subject_X = []
    per_subject_y = []
    sessions_loaded = []
 
    for ses_dir in session_dirs:
        analysis_dir = ses_dir / "analysis"
        x_path = analysis_dir / "feature_matrix.csv"
        y_path = analysis_dir / "labels.npy"
        if not x_path.exists() or not y_path.exists():
            continue  # this session wasn't processed (steps 2-4 not run for it) -- skip silently
 
        X_ses = pd.read_csv(x_path)
        y_ses = np.load(y_path)
 
        # --- Per-SESSION baseline normalization ---
        # Same reasoning as per-subject normalization, applied one level
        # finer: even within one person, electrode reapplication between
        # separate recording sessions/admissions can shift the absolute
        # scale. Normalizing each session against ITS OWN baseline windows
        # (not the subject's pooled baseline across all sessions) controls
        # for this before sessions get combined.
        baseline_mask = y_ses == 0
        if baseline_mask.sum() == 0:
            continue  # no baseline windows in this session, can't normalize -- skip
        baseline_mean = X_ses[baseline_mask].mean()
        baseline_std = X_ses[baseline_mask].std().replace(0, 1)
        X_ses = (X_ses - baseline_mean) / baseline_std
 
        per_subject_X.append(X_ses)
        per_subject_y.append(y_ses)
        sessions_loaded.append(ses_dir.name)
 
    if len(per_subject_X) == 0:
        print(f"  {sub}: MISSING feature_matrix.csv/labels.npy for all sessions -- skipped. Run steps 2-4 first.")
        continue
 
    X_sub = pd.concat(per_subject_X, axis=0, ignore_index=True)
    y_sub = np.concatenate(per_subject_y)
    n_spike = int((y_sub == 1).sum())
    n_base = int((y_sub == 0).sum())
 
    print(f"  {sub}: {len(sessions_loaded)} sessions pooled ({', '.join(sessions_loaded)}) "
          f"-> {X_sub.shape[0]} windows (spike={n_spike}, baseline={n_base})"
          f"{' [SMALL FOLD]' if len(y_sub) < MIN_RELIABLE_N else ''}")
 
    all_X.append(X_sub)
    all_y.append(y_sub)
    all_subjects.extend([sub] * len(y_sub))
 
X = pd.concat(all_X, axis=0, ignore_index=True)
y = np.concatenate(all_y)
groups = np.array(all_subjects)
 
print(f"\nCombined: X {X.shape}, y {y.shape}, from {len(set(groups))} subjects")
print(f"Total class counts -> spike: {int((y == 1).sum())}, baseline: {int((y == 0).sum())}")
 
X.to_csv(multi_dir / "combined_feature_matrix.csv", index=False)
np.save(multi_dir / "combined_labels.npy", y)
np.save(multi_dir / "combined_groups.npy", groups)
 
# ---------------------------------------------------------------------------
#Leave-one-subject-out cross-validation. Each fold: train on all subjects
#    except one, test on the held-out subject entirely.
# ---------------------------------------------------------------------------
 
logo = LeaveOneGroupOut()
n_folds = logo.get_n_splits(groups=groups)
print(f"\nRunning leave-one-subject-out CV ({n_folds} folds)...")
 
fold_results = []
rf_importances_per_fold = []
 
for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
    held_out_subject = groups[test_idx][0]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    logistic_model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    logistic_model.fit(X_train_scaled, y_train)
 
    rf_model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
 
    rf_importances_per_fold.append(rf_model.feature_importances_)
 
    fold_models = {
        "logistic_regression": (logistic_model, X_test_scaled),
        "random_forest": (rf_model, X_test),
    }
    fold_train_data = {
        "logistic_regression": X_train_scaled,
        "random_forest": X_train,
    }
 
    print(f"\n[Fold {fold_idx + 1}/{n_folds}] Held-out subject: {held_out_subject} "
          f"(n_test={len(y_test)}, spike={int((y_test == 1).sum())}, baseline={int((y_test == 0).sum())})")
 
    # --- Nested threshold tuning ---
    # High AUC but poor recall at the default 0.5 cutoff means the model
    # separates the classes fine, it's just using a miscalibrated decision
    # boundary (same issue fixed in step 5). Fix: find a better threshold
    # using ONLY the training subjects (inner cross-validation, grouped by
    # subject so windows from the same person don't leak across the inner
    # split), then apply that threshold to the held-out subject. The
    # held-out subject's own data is never used to choose its threshold --
    # this stays a fair, leakage-free test of generalization.
    train_groups = groups[train_idx]
    n_inner_groups = len(set(train_groups))
    inner_cv = GroupKFold(n_splits=min(4, n_inner_groups))
 
    for name, (model, X_te) in fold_models.items():
        X_tr = fold_train_data[name]
        try:
            cv_proba = cross_val_predict(
                model, X_tr, y_train, groups=train_groups, cv=inner_cv,
                method="predict_proba", n_jobs=-1
            )[:, 1]
            fpr, tpr, thresholds = roc_curve(y_train, cv_proba)
            gmean = np.sqrt(tpr * (1 - fpr))
            threshold = thresholds[np.argmax(gmean)]
        except ValueError:
            # Falls back to 0.5 if the inner CV can't run (e.g. too few
            # training subjects/groups left in some fold configuration)
            threshold = 0.5
 
        y_proba = model.predict_proba(X_te)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
 
        # Some held-out subjects may have very few baseline windows -- guard
        # against a fold where a class is entirely absent from the test set,
        # which would make roc_auc_score undefined.
        if len(set(y_test)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_test, y_proba)
 
        report = classification_report(
            y_test, y_pred, target_names=["baseline", "spike"], output_dict=True, zero_division=0
        )
 
        fold_results.append({
            "fold": fold_idx + 1,
            "held_out_subject": held_out_subject,
            "model": name,
            "n_test": len(y_test),
            "threshold": threshold,
            "roc_auc": auc,
            "baseline_precision": report["baseline"]["precision"],
            "baseline_recall": report["baseline"]["recall"],
            "spike_precision": report["spike"]["precision"],
            "spike_recall": report["spike"]["recall"],
            "accuracy": report["accuracy"],
        })
        print(f"    {name}: threshold={threshold:.3f}, AUC={auc if np.isnan(auc) else round(auc, 3)}, "
              f"accuracy={report['accuracy']:.3f}, "
              f"baseline_recall={report['baseline']['recall']:.3f}, "
              f"spike_recall={report['spike']['recall']:.3f}")
 
fold_results_df = pd.DataFrame(fold_results)
fold_results_df.to_csv(multi_dir / "loso_fold_results.csv", index=False)
 
# ---------------------------------------------------------------------------
# Summarize across folds. Folds vary hugely in size -- a plain mean would let
# a small-sample fold count exactly as much as a large one, which is
# statistically misleading. Two things fix this:
#   (a) weight the average by n_test, so large/reliable folds dominate
#   (b) report folds below MIN_RELIABLE_N separately, flagged, rather than
#       silently blending a noisy small-sample result into the headline
# ---------------------------------------------------------------------------
 
 
def weighted_summary(df, min_n):
    reliable = df[df["n_test"] >= min_n]
    small = df[df["n_test"] < min_n]
    rows = []
    for name, group in reliable.groupby("model"):
        w = group["n_test"]
        row = {"model": name, "n_folds_used": len(group), "n_windows_used": int(w.sum())}
        for metric in ["roc_auc", "accuracy", "baseline_recall", "spike_recall"]:
            valid = group.dropna(subset=[metric])
            if len(valid) == 0:
                row[f"{metric}_weighted_mean"] = np.nan
            else:
                row[f"{metric}_weighted_mean"] = np.average(valid[metric], weights=valid["n_test"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("model"), small
 
 
print(f"\n{'=' * 60}\nLOSO summary across {n_folds} held-out subjects\n{'=' * 60}")
 
summary, small_folds = weighted_summary(fold_results_df, MIN_RELIABLE_N)
summary.to_csv(multi_dir / "loso_summary.csv")
print(f"Reliable folds only (n_test >= {MIN_RELIABLE_N}), weighted by fold size:")
print(summary.round(3).to_string())
 
if len(small_folds) > 0:
    small_folds.to_csv(multi_dir / "loso_small_folds_excluded.csv", index=False)
    print(f"\nExcluded from the summary above -- {small_folds['held_out_subject'].nunique()} "
          f"subject(s) with fewer than {MIN_RELIABLE_N} test windows (too few to trust a per-fold estimate):")
    print(small_folds[["held_out_subject", "model", "n_test", "roc_auc", "baseline_recall", "spike_recall"]]
          .to_string(index=False))
    print("These are saved separately in loso_small_folds_excluded.csv -- report them as individual "
          "case notes if useful, but don't average them into the headline generalization number.")
 
 
plt.figure(figsize=(10, 5))
for name in fold_results_df["model"].unique():
    sub_df = fold_results_df[fold_results_df["model"] == name].sort_values("held_out_subject")
    plt.plot(sub_df["held_out_subject"], sub_df["roc_auc"], marker="o", label=name.replace("_", " ").title())
plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Chance")
plt.xticks(rotation=45, ha="right")
plt.ylabel("ROC-AUC (held-out subject)")
plt.title("Leave-One-Subject-Out Generalization")
plt.legend()
plt.tight_layout()
plt.savefig(multi_dir / "loso_per_subject_auc.png", dpi=150)
plt.close()
 
 
#Average feature importance across folds -- which features/channels are consistently useful across different subjects, not just one recording.
 
 
 
def readable_feature_name(col):
    if "_ch" in col:
        base, ch_idx = col.rsplit("_ch", 1)
        return f"{base}_{eeg_channel_names[int(ch_idx)]}"
    return col
 
 
mean_importance = np.mean(rf_importances_per_fold, axis=0)
importance_df = pd.DataFrame({
    "feature": X.columns,
    "feature_readable": [readable_feature_name(c) for c in X.columns],
    "mean_importance": mean_importance,
}).sort_values("mean_importance", ascending=False)
importance_df.to_csv(multi_dir / "loso_feature_importance.csv", index=False)
 
print("\nTop 15 features by mean random forest importance across all LOSO folds:")
print(importance_df.head(15)[["feature_readable", "mean_importance"]].to_string(index=False))
 
top_n = 20
plt.figure(figsize=(9, 6))
top_feats = importance_df.head(top_n).iloc[::-1]
plt.barh(top_feats["feature_readable"], top_feats["mean_importance"], color="steelblue")
plt.xlabel("Mean random forest importance (across LOSO folds)")
plt.title(f"Top {top_n} Features -- Consistent Across Subjects")
plt.tight_layout()
plt.savefig(multi_dir / "loso_feature_importance.png", dpi=150)
plt.close()
 
 
final_scaler = StandardScaler()
X_scaled_all = final_scaler.fit_transform(X)
 
final_logistic = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
final_logistic.fit(X_scaled_all, y)
 
final_rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
final_rf.fit(X, y)
 
joblib.dump(final_logistic, multi_dir / "final_logistic_regression_all_subjects.joblib")
joblib.dump(final_rf, multi_dir / "final_random_forest_all_subjects.joblib")
joblib.dump(final_scaler, multi_dir / "final_scaler_all_subjects.joblib")
 
with open(multi_dir / "run_info.json", "w") as f:
    json.dump({
        "subjects": subjects,
        "n_folds": n_folds,
        "n_total_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts": {"spike": int((y == 1).sum()), "baseline": int((y == 0).sum())},
        "preprocessing_note": "Features are per-SESSION baseline-normalized (z-scored against each "
                               "session's own baseline-window mean/std), then pooled across all "
                               "available sessions per subject, then across subjects. Any new data "
                               "run through final_*_all_subjects.joblib must be normalized the same way "
                               "using its own baseline windows first.",
        "min_reliable_fold_size": MIN_RELIABLE_N,
    }, f, indent=2)
 
print(f"\nSaved to: {multi_dir}")
print("  - combined_feature_matrix.csv, combined_labels.npy, combined_groups.npy  (session-normalized, pooled)")
print("  - loso_fold_results.csv           (per-subject, per-model results, all folds)")
print("  - loso_summary.csv                (weighted mean, reliable folds only, n_test >= "
      f"{MIN_RELIABLE_N})")
print("  - loso_small_folds_excluded.csv   (small folds reported separately, not in the headline average)")
print("  - loso_per_subject_auc.png")
print("  - loso_feature_importance.csv, loso_feature_importance.png")
print("  - final_*_all_subjects.joblib (models trained on everything, for future reuse)")
print("  - run_info.json")
print("\nDone!")