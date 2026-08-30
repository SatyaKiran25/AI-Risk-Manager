import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)


# ============================================================
# 1. LOAD DATA
# ============================================================

TRAIN_PATH = "data/processed/train_v2.csv"
VAL_PATH = "data/processed/validation_v2.csv"
TEST_PATH = "data/processed/test_v2.csv"

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)


# ============================================================
# 2. CREATE TIME FEATURES
# ============================================================

# for df in [train, val, test]:

    # df["timestamp"] = pd.to_datetime(df["timestamp"])

    # df["hour"] = df["timestamp"].dt.hour

    # df["day_of_week"] = df["timestamp"].dt.dayofweek


# ============================================================
# 3. DEFINE TARGET
# ============================================================

TARGET = "is_fraud"

y_train = train[TARGET]
y_val = val[TARGET]
y_test = test[TARGET]


# ============================================================
# 4. DEFINE FEATURES
# ============================================================

numeric_features = [
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
    "day_of_week"
]


FEATURES = numeric_features

X_train = train[FEATURES]
X_val = val[FEATURES]
X_test = test[FEATURES]


# ============================================================
# 5. DIAGNOSTIC CHECKS
# ============================================================

print("\n" + "=" * 60)
print("DIAGNOSTIC CHECKS")
print("=" * 60)

print(
    "\nmerchant_risk_tier unique values:",
    sorted(train["merchant_risk_tier"].dropna().unique())
)

print("\nFeature means:")
comparison = pd.DataFrame({
    "train": X_train[numeric_features].mean(),
    "validation": X_val[numeric_features].mean(),
    "test": X_test[numeric_features].mean()
})

print(comparison)


# ============================================================
# 6. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 7. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


# ============================================================
# 8. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    ]
)


# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


# ============================================================
# 10. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 11. TRAIN
# ============================================================

print("\nTraining Logistic Regression...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 12. VALIDATION PROBABILITIES
# ============================================================

val_prob = pipeline.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 13. THRESHOLD TUNING
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

precisions, recalls, thresholds = precision_recall_curve(
    y_val,
    val_prob
)

# thresholds has one fewer element than precision/recall
f1_scores = (
    2 * (precisions[:-1] * recalls[:-1])
    /
    (
        precisions[:-1]
        + recalls[:-1]
        + 1e-9
    )
)

best_idx = np.argmax(f1_scores)

best_threshold = thresholds[best_idx]

print(
    f"Best threshold: {best_threshold:.4f}"
)

print(
    f"Precision: {precisions[best_idx]:.4f}"
)

print(
    f"Recall: {recalls[best_idx]:.4f}"
)

print(
    f"F1: {f1_scores[best_idx]:.4f}"
)


# ============================================================
# 14. CHOSEN THRESHOLD
# ============================================================
# For now we use validation-best F1.
#
# Later, when we build the actual Cost Model, this will be
# replaced by the cost-optimal threshold.
# ============================================================

CHOSEN_THRESHOLD = best_threshold


# ============================================================
# 15. VALIDATION PREDICTIONS
# ============================================================

val_pred = (
    val_prob >= CHOSEN_THRESHOLD
).astype(int)


# ============================================================
# 16. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    y_true,
    y_pred,
    y_prob,
    dataset_name
):

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    false_positive_rate = fp / (fp + tn)

    false_negative_rate = fn / (fn + tp)

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(
        f"Threshold          : {CHOSEN_THRESHOLD:.4f}"
    )

    print(
        f"Precision          : {precision:.4f}"
    )

    print(
        f"Recall             : {recall:.4f}"
    )

    print(
        f"F1 Score           : {f1:.4f}"
    )

    print(
        f"ROC-AUC            : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC             : {pr_auc:.4f}"
    )

    print(
        f"False Positive Rate: {false_positive_rate:.4f}"
    )

    print(
        f"False Negative Rate: {false_negative_rate:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate
    }


# ============================================================
# 17. VALIDATION RESULTS
# ============================================================

val_metrics = evaluate_model(
    y_val,
    val_pred,
    val_prob,
    "VALIDATION RESULTS"
)


# ============================================================
# 18. FINAL TEST EVALUATION
# ============================================================
# IMPORTANT:
# The threshold was selected using VALIDATION only.
# We do NOT tune the threshold using the test set.
# ============================================================

print("\nEvaluating test set...")

test_prob = pipeline.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= CHOSEN_THRESHOLD
).astype(int)

test_metrics = evaluate_model(
    y_test,
    test_pred,
    test_prob,
    "FINAL TEST RESULTS"
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("BASELINE MODEL COMPLETE")
print("=" * 60)

print(
    "Logistic Regression baseline completed successfully."
)

print(
    "The test set was evaluated using the "
    "validation-selected threshold."
)

print(
    "Next step: XGBoost."
)