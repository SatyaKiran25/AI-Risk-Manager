from pathlib import Path

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "train_v2.csv"
VAL_FILE = DATA_DIR / "validation_v2.csv"
TEST_FILE = DATA_DIR / "test_v2.csv"


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)


# ============================================================
# TARGET
# ============================================================

TARGET = "is_fraud"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "amount",
    "merchant_risk_tier",
    "txns_prev_1h",
    "txns_prev_24h",
    "amount_z_score",
    "new_device",
    "new_location",
    "location_mismatch",
    "new_merchant",
    "time_of_day_anomaly",
    "velocity_anomaly",
    "customer_txn_count_prior",
    "customer_avg_amount_prior",
    "customer_std_amount_prior",
    "hour",
    "day_of_week",
]


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in train.columns
]

if missing_features:
    raise ValueError(
        f"Missing features in dataset: {missing_features}"
    )


# ============================================================
# X / y
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = val[FEATURES]
y_val = val[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ============================================================
# CLASS IMBALANCE
# ============================================================

fraud_count = y_train.sum()
legit_count = len(y_train) - fraud_count

scale_pos_weight = legit_count / fraud_count

print("\nClass imbalance:")
print("Legitimate:", legit_count)
print("Fraud:", fraud_count)
print(
    "scale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Training completed.")


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

val_prob = model.predict_proba(X_val)[:, 1]


# ============================================================
# THRESHOLD TUNING
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_prob
)

f1_scores = (
    2 * precision * recall
    /
    (precision + recall + 1e-9)
)

# precision/recall contains one extra element
best_idx = np.argmax(f1_scores[:-1])

best_threshold = thresholds[best_idx]

print(
    f"Best threshold: {best_threshold:.4f}"
)

print(
    f"Precision: {precision[best_idx]:.4f}"
)

print(
    f"Recall: {recall[best_idx]:.4f}"
)

print(
    f"F1: {f1_scores[best_idx]:.4f}"
)


# ============================================================
# VALIDATION RESULTS
# ============================================================

val_pred = (
    val_prob >= best_threshold
).astype(int)

val_precision = precision_score(
    y_val,
    val_pred,
    zero_division=0
)

val_recall = recall_score(
    y_val,
    val_pred,
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    val_pred,
    zero_division=0
)

val_roc_auc = roc_auc_score(
    y_val,
    val_prob
)

val_pr_auc = average_precision_score(
    y_val,
    val_prob
)

val_cm = confusion_matrix(
    y_val,
    val_pred
)

tn, fp, fn, tp = val_cm.ravel()

val_fpr = fp / (fp + tn)

val_fnr = fn / (fn + tp)


print("\n" + "=" * 60)
print("XGBOOST VALIDATION RESULTS")
print("=" * 60)

print(
    f"Threshold          : {best_threshold:.4f}"
)

print(
    f"Precision           : {val_precision:.4f}"
)

print(
    f"Recall              : {val_recall:.4f}"
)

print(
    f"F1 Score            : {val_f1:.4f}"
)

print(
    f"ROC-AUC             : {val_roc_auc:.4f}"
)

print(
    f"PR-AUC              : {val_pr_auc:.4f}"
)

print(
    f"False Positive Rate : {val_fpr:.4f}"
)

print(
    f"False Negative Rate : {val_fnr:.4f}"
)

print("\nConfusion Matrix:")
print(val_cm)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating test set...")

test_prob = model.predict_proba(X_test)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)


test_precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    test_prob
)

test_pr_auc = average_precision_score(
    y_test,
    test_prob
)

test_cm = confusion_matrix(
    y_test,
    test_pred
)

tn, fp, fn, tp = test_cm.ravel()

test_fpr = fp / (fp + tn)

test_fnr = fn / (fn + tp)


# ============================================================
# TEST RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL XGBOOST TEST RESULTS")
print("=" * 60)

print(
    f"Threshold          : {best_threshold:.4f}"
)

print(
    f"Precision           : {test_precision:.4f}"
)

print(
    f"Recall              : {test_recall:.4f}"
)

print(
    f"F1 Score            : {test_f1:.4f}"
)

print(
    f"ROC-AUC             : {test_roc_auc:.4f}"
)

print(
    f"PR-AUC              : {test_pr_auc:.4f}"
)

print(
    f"False Positive Rate : {test_fpr:.4f}"
)

print(
    f"False Negative Rate : {test_fnr:.4f}"
)

print("\nConfusion Matrix:")
print(test_cm)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION vs XGBOOST")
print("=" * 60)

print("\nXGBoost test metrics:")

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1        : {test_f1:.4f}"
)

print(
    f"ROC-AUC   : {test_roc_auc:.4f}"
)

print(
    f"PR-AUC    : {test_pr_auc:.4f}"
)

print("\n" + "=" * 60)
print("XGBOOST COMPLETE")
print("=" * 60)