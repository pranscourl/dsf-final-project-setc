"""Retrain both screening models on one shared feature set and compare synthetic training.

Run from the repository root:
    python scripts/retrain_and_compare_models.py

The script uses the saved train/test CSVs. It never uses the held-out test data for fitting,
feature selection, or threshold selection. Synthetic examples are generated inside each training
fold for the threshold estimate, matching the leakage safeguards in the modeling notebooks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET = "Diabetes_binary"
RANDOM_STATE = 42
TARGET_RECALL = 0.85
REAL_PREVALENCE = 0.14
N_SPLITS = 5

# Shared by both saved models. This is the former 70k feature set with the two
# healthcare-access proxies removed. It keeps the models directly comparable.
FEATURES = [
    "HighBP",
    "HighChol",
    "BMI",
    "Stroke",
    "HeartDiseaseorAttack",
    "Fruits",
    "HvyAlcoholConsump",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Income",
]
NON_BINARY = {"BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"}

TRAIN_70K = ROOT / "data/05_cmb_train_test_for_models/diabetes_model_train.csv"
TRAIN_200K = ROOT / "data/07_cmu+synthetic/diabetes_cmu_synthetic_train.csv"
TEST = ROOT / "data/05_cmb_train_test_for_models/diabetes_model_test.csv"
MODEL_70K = ROOT / "application/models/logreg_70k.joblib"
MODEL_200K = ROOT / "application/models/logreg_200k.joblib"
CONTROL_OUTPUT = ROOT / "data/07_cmu+synthetic/diabetes_synthetic_control_comparison.csv"


def make_model(class_weight=None):
    scaled = [name for name in FEATURES if name in NON_BINARY]
    prep = ColumnTransformer(
        [("scale", StandardScaler(), scaled)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("prep", prep),
            (
                "model",
                LogisticRegression(C=np.inf, max_iter=2000, class_weight=class_weight),
            ),
        ]
    )


def choose_threshold(y_true, probabilities):
    """Highest OOF threshold whose observed recall meets the target."""
    from sklearn.metrics import precision_recall_curve

    _, recall, thresholds = precision_recall_curve(y_true, probabilities)
    valid = thresholds[recall[:-1] >= TARGET_RECALL]
    if not len(valid):
        raise RuntimeError("No out-of-fold threshold reaches the configured recall goal.")
    return float(valid.max())


def metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    flagged_share = recall * REAL_PREVALENCE + fpr * (1 - REAL_PREVALENCE)
    return {
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, predictions),
        "Recall (sensitivity)": recall,
        "Specificity": 1 - fpr,
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, probabilities),
        "Brier score": brier_score_loss(y_true, probabilities),
        "False negatives": int(fn),
        "False positives": int(fp),
        "Precision @ 14%": recall * REAL_PREVALENCE / flagged_share,
        "Share flagged @ 14%": flagged_share,
    }


def require_features(frame, label):
    absent = [name for name in FEATURES if name not in frame.columns]
    if absent:
        raise ValueError(f"{label} is missing shared model features: {absent}")


def load_synthetic_generator():
    helper_dir = ROOT / "notebooks/09_utils"
    sys.path.insert(0, str(helper_dir))
    from synth_utils import generate_synthetic

    return generate_synthetic


def synthetic_fold_oof(real_rows, generator):
    """OOF predictions with synthetic diabetes examples made within each fold."""
    X_real = real_rows.drop(columns=TARGET).reset_index(drop=True)
    y_real = real_rows[TARGET].reset_index(drop=True)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    probabilities = np.empty(len(y_real), dtype=float)

    for fold_index, (train_idx, valid_idx) in enumerate(cv.split(X_real[FEATURES], y_real)):
        x_train, y_train = X_real.iloc[train_idx], y_real.iloc[train_idx]
        x_valid, y_valid = X_real.iloc[valid_idx], y_real.iloc[valid_idx]
        n_synthetic = int((y_train == 0).sum() - (y_train == 1).sum())
        synthetic = generator(
            x_train.loc[y_train == 1],
            n_synthetic,
            seed=RANDOM_STATE + fold_index,
            avoid=x_valid,
        )
        fit_x = pd.concat([x_train[FEATURES], synthetic[FEATURES]], ignore_index=True)
        fit_y = pd.concat(
            [y_train.reset_index(drop=True), pd.Series(1, index=range(n_synthetic))],
            ignore_index=True,
        )
        model = make_model().fit(fit_x[FEATURES], fit_y)
        probabilities[valid_idx] = model.predict_proba(x_valid[FEATURES])[:, 1]

        print(
            f"Synthetic CV fold {fold_index + 1}/{N_SPLITS}: "
            f"{len(fit_x):,} training rows, {len(x_valid):,} real validation rows"
        )

    return y_real, probabilities


def rows_for_model(name, train_note, y_test, probabilities, threshold):
    output = []
    for point, cutoff in (("Chosen threshold", threshold), ("Default (0.50)", 0.5)):
        output.append(
            {
                "Model": name,
                "Training data": train_note,
                "Point": point,
                **metrics(y_test, probabilities, cutoff),
            }
        )
    return output


def main():
    train_70k = pd.read_csv(TRAIN_70K)
    train_200k = pd.read_csv(TRAIN_200K)
    test = pd.read_csv(TEST)
    for label, frame in (("70k training data", train_70k), ("200k training data", train_200k), ("test data", test)):
        require_features(frame, label)
        if TARGET not in frame:
            raise ValueError(f"{label} is missing {TARGET!r}")

    y_test = test[TARGET]
    x_test = test[FEATURES]
    y_70k = train_70k[TARGET]
    x_70k = train_70k[FEATURES]

    # 70k real balanced model.
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_70k = cross_val_predict(
        make_model(), x_70k, y_70k, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold_70k = choose_threshold(y_70k, oof_70k)
    model_70k = make_model().fit(x_70k, y_70k)
    p_70k = model_70k.predict_proba(x_test)[:, 1]
    bundle_70k = {
        "model": model_70k,
        "features": FEATURES,
        "threshold": threshold_70k,
        "target_recall": TARGET_RECALL,
        "feature_set_name": "Shared 14-feature set (excludes healthcare-access proxies)",
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "training_data": f"{len(train_70k):,} real rows (balanced)",
    }
    joblib.dump(bundle_70k, MODEL_70K)
    print(f"Saved 70k model: {threshold_70k:.4f} threshold; test AUC {roc_auc_score(y_test, p_70k):.4f}")

    # 200k synthetic model. CV folds are built using real rows only, then synthesized independently.
    real_200k = train_200k.loc[train_200k["is_synthetic"].eq(0)].drop(
        columns=["is_synthetic"], errors="ignore"
    ).reset_index(drop=True)
    full_200k = train_200k.drop(columns=["is_synthetic"], errors="ignore")
    if not train_200k["is_synthetic"].eq(1).any():
        raise ValueError("The 200k training CSV contains no rows marked as synthetic.")
    y_synth_oof, p_synth_oof = synthetic_fold_oof(real_200k, load_synthetic_generator())
    threshold_200k = choose_threshold(y_synth_oof, p_synth_oof)
    model_200k = make_model().fit(full_200k[FEATURES], full_200k[TARGET])
    p_200k = model_200k.predict_proba(x_test)[:, 1]
    bundle_200k = {
        "model": model_200k,
        "features": FEATURES,
        "threshold": threshold_200k,
        "target_recall": TARGET_RECALL,
        "feature_set_name": "Shared 14-feature set (excludes healthcare-access proxies)",
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "training_data": f"{len(full_200k):,} rows (real + synthetic)",
    }
    joblib.dump(bundle_200k, MODEL_200K)
    print(f"Saved 200k model: {threshold_200k:.4f} threshold; test AUC {roc_auc_score(y_test, p_200k):.4f}")

    # Matched-feature real-only control: class weighting is chosen inside each CV fit.
    y_real_oof = real_200k[TARGET]
    x_real = real_200k[FEATURES]
    control_oof = cross_val_predict(
        make_model(class_weight="balanced"),
        x_real,
        y_real_oof,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold_control = choose_threshold(y_real_oof, control_oof)
    control = make_model(class_weight="balanced").fit(x_real, y_real_oof)
    p_control = control.predict_proba(x_test)[:, 1]

    comparison = []
    comparison += rows_for_model(
        "70k real balanced", bundle_70k["training_data"], y_test, p_70k, threshold_70k
    )
    comparison += rows_for_model(
        "200k real + synthetic", bundle_200k["training_data"], y_test, p_200k, threshold_200k
    )
    comparison += rows_for_model(
        "128k real only, class weighted",
        f"{len(real_200k):,} real rows (class_weight='balanced')",
        y_test,
        p_control,
        threshold_control,
    )
    CONTROL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(CONTROL_OUTPUT, index=False)

    # Keep the existing 70k evaluation artifact current and save all comparisons in one CSV.
    pd.DataFrame(
        rows_for_model("Logistic regression (70k real rows)", bundle_70k["training_data"], y_test, p_70k, threshold_70k)
    ).to_csv(ROOT / "data/06_cmb_logreg_70kdataset/diabetes_eval_logreg_70k.csv", index=False)

    result = pd.DataFrame(comparison)
    print("\nHeld-out test comparison (same rows and 14 features):")
    print(result[["Model", "Point", "Threshold", "Recall (sensitivity)", "Specificity", "ROC AUC", "Precision @ 14%"]].to_string(index=False))
    print(f"\nSaved comparison to {CONTROL_OUTPUT.relative_to(ROOT)}")
    print("Shared feature list:", json.dumps(FEATURES))


if __name__ == "__main__":
    main()
