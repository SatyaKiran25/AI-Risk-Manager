from pathlib import Path

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    log_loss,
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
# DATA
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


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

xgb_model = XGBClassifier(
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

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("XGBoost training completed.")


# ============================================================
# RAW PROBABILITIES
# ============================================================

val_raw_prob = xgb_model.predict_proba(X_val)[:, 1]

test_raw_prob = xgb_model.predict_proba(X_test)[:, 1]


# ============================================================
# RAW PROBABILITY METRICS
# ============================================================

print("\n" + "=" * 60)
print("RAW XGBOOST PROBABILITY")
print("=" * 60)

raw_brier = brier_score_loss(
    y_val,
    val_raw_prob
)

raw_logloss = log_loss(
    y_val,
    val_raw_prob
)

raw_roc_auc = roc_auc_score(
    y_val,
    val_raw_prob
)

raw_pr_auc = average_precision_score(
    y_val,
    val_raw_prob
)

print(
    f"Validation Brier Score : {raw_brier:.6f}"
)

print(
    f"Validation Log Loss    : {raw_logloss:.6f}"
)

print(
    f"Validation ROC-AUC     : {raw_roc_auc:.4f}"
)

print(
    f"Validation PR-AUC      : {raw_pr_auc:.4f}"
)


# ============================================================
# ISOTONIC CALIBRATION
# ============================================================

print("\n" + "=" * 60)
print("CALIBRATING PROBABILITIES")
print("=" * 60)

calibrator = IsotonicRegression(
    y_min=0.0,
    y_max=1.0,
    out_of_bounds="clip",
)

# IMPORTANT:
# Calibration mapping is learned ONLY from validation data.

calibrator.fit(
    val_raw_prob,
    y_val
)

print("Calibration completed using validation data.")


# ============================================================
# CALIBRATED PROBABILITIES
# ============================================================

val_calibrated_prob = calibrator.predict(
    val_raw_prob
)

test_calibrated_prob = calibrator.predict(
    test_raw_prob
)


# ============================================================
# CALIBRATED VALIDATION METRICS
# ============================================================

cal_brier = brier_score_loss(
    y_val,
    val_calibrated_prob
)

cal_logloss = log_loss(
    y_val,
    val_calibrated_prob
)

cal_roc_auc = roc_auc_score(
    y_val,
    val_calibrated_prob
)

cal_pr_auc = average_precision_score(
    y_val,
    val_calibrated_prob
)


print("\n" + "=" * 60)
print("CALIBRATED VALIDATION RESULTS")
print("=" * 60)

print(
    f"Brier Score : {cal_brier:.6f}"
)

print(
    f"Log Loss    : {cal_logloss:.6f}"
)

print(
    f"ROC-AUC     : {cal_roc_auc:.4f}"
)

print(
    f"PR-AUC      : {cal_pr_auc:.4f}"
)


# ============================================================
# TEST EVALUATION
# ============================================================

test_brier = brier_score_loss(
    y_test,
    test_calibrated_prob
)

test_logloss = log_loss(
    y_test,
    test_calibrated_prob
)

test_roc_auc = roc_auc_score(
    y_test,
    test_calibrated_prob
)

test_pr_auc = average_precision_score(
    y_test,
    test_calibrated_prob
)


print("\n" + "=" * 60)
print("FINAL TEST - CALIBRATED MODEL")
print("=" * 60)

print(
    f"Brier Score : {test_brier:.6f}"
)

print(
    f"Log Loss    : {test_logloss:.6f}"
)

print(
    f"ROC-AUC     : {test_roc_auc:.4f}"
)

print(
    f"PR-AUC      : {test_pr_auc:.4f}"
)


# ============================================================
# EXAMPLE PROBABILITIES
# ============================================================

print("\n" + "=" * 60)
print("PROBABILITY EXAMPLES")
print("=" * 60)

comparison = pd.DataFrame({
    "raw_probability": val_raw_prob[:20],
    "calibrated_probability": val_calibrated_prob[:20],
    "actual": y_val.iloc[:20].values,
})

print(comparison.to_string(index=False))


# ============================================================
# SAVE CALIBRATED TEST PREDICTIONS
# ============================================================

output = test.copy()

output["raw_risk_probability"] = test_raw_prob

output["calibrated_risk_probability"] = (
    test_calibrated_prob
)

output_path = (
    DATA_DIR / "test_calibrated_predictions.csv"
)

output.to_csv(
    output_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("CALIBRATION COMPLETE")
print("=" * 60)

print(
    "\nSaved:"
)

print(output_path)

print(
    "\nNext step: Cost Model"
)