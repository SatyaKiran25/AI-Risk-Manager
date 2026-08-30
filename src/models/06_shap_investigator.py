from pathlib import Path

import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "train_v2.csv"
ROUTED_FILE = DATA_DIR / "test_routed.csv"
OUTPUT_FILE = DATA_DIR / "test_investigated.csv"

TARGET = "is_fraud"

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
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
routed = pd.read_csv(ROUTED_FILE)

X_train = train[FEATURES]
y_train = train[TARGET]

print("=" * 70)
print("AI RISK MANAGER - SHAP INVESTIGATOR")
print("=" * 70)

print("\nTraining XGBoost model for SHAP analysis...")

fraud_count = y_train.sum()
legit_count = len(y_train) - fraud_count
scale_pos_weight = legit_count / fraud_count

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

model.fit(X_train, y_train, verbose=False)

print("XGBoost training completed.")


# ============================================================
# SELECT REVIEW TRANSACTIONS
# ============================================================

review_mask = routed["router_decision"] == "REVIEW"
review_data = routed.loc[review_mask].copy()

print("\nReview transactions:", len(review_data))

if len(review_data) == 0:
    print("\nNo REVIEW transactions found.")

    routed["investigation_status"] = "NOT_REQUIRED"
    routed["top_risk_feature"] = ""
    routed["top_shap_contribution"] = np.nan
    routed["shap_explanation"] = ""
    routed["shap_features"] = ""
    routed["shap_values"] = ""
    routed["shap_risk_factors"] = ""

    routed.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)
    raise SystemExit


# ============================================================
# SHAP
# ============================================================

print("\nCalculating SHAP explanations...")

explainer = shap.TreeExplainer(model)

X_review = review_data[FEATURES]

shap_values = explainer.shap_values(X_review)
shap_values = np.asarray(shap_values)

# Handle possible XGBoost/SHAP output shape
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]


# ============================================================
# GENERATE SHAP DATA
# ============================================================

feature_explanations = []

top_features = []
top_contributions = []

all_shap_features = []
all_shap_values = []
all_shap_risk_factors = []


for row_idx in range(len(review_data)):

    values = shap_values[row_idx]

    ranked_indices = np.argsort(
        np.abs(values)
    )[::-1]

    explanations = []

    ranked_features = []
    ranked_values = []
    ranked_risk_factors = []

    # --------------------------------------------------------
    # STORE TOP 8 CONTRIBUTORS
    # --------------------------------------------------------

    for feature_idx in ranked_indices[:8]:

        feature = FEATURES[feature_idx]
        contribution = float(values[feature_idx])

        feature_value = X_review.iloc[
            row_idx,
            feature_idx
        ]

        # Store feature
        ranked_features.append(feature)

        # Store exact SHAP value
        ranked_values.append(
            f"{contribution:.6f}"
        )

        # Human-readable factor
        direction = (
            "increases fraud risk"
            if contribution > 0
            else "decreases fraud risk"
        )

        ranked_risk_factors.append(
            f"{feature}={feature_value} "
            f"({direction}, SHAP={contribution:.4f})"
        )

        explanations.append(
            f"{feature}={feature_value} "
            f"({'increased' if contribution > 0 else 'decreased'} "
            f"fraud risk, SHAP={contribution:.4f})"
        )

    feature_explanations.append(
        " | ".join(explanations)
    )

    top_idx = ranked_indices[0]

    top_features.append(
        FEATURES[top_idx]
    )

    top_contributions.append(
        float(values[top_idx])
    )

    # --------------------------------------------------------
    # SAVE RANKED SHAP DATA
    # --------------------------------------------------------

    all_shap_features.append(
        " | ".join(ranked_features)
    )

    all_shap_values.append(
        " | ".join(ranked_values)
    )

    all_shap_risk_factors.append(
        " | ".join(ranked_risk_factors)
    )


# ============================================================
# ATTACH INVESTIGATION
# ============================================================

review_data["investigation_status"] = "INVESTIGATED"

review_data["top_risk_feature"] = top_features

review_data["top_shap_contribution"] = top_contributions

review_data["shap_explanation"] = feature_explanations

# NEW — required by dashboard
review_data["shap_features"] = all_shap_features

review_data["shap_values"] = all_shap_values

review_data["shap_risk_factors"] = all_shap_risk_factors


# ============================================================
# NON-REVIEW
# ============================================================

non_review = routed.loc[
    ~review_mask
].copy()

non_review["investigation_status"] = "NOT_REQUIRED"

non_review["top_risk_feature"] = ""

non_review["top_shap_contribution"] = np.nan

non_review["shap_explanation"] = ""

non_review["shap_features"] = ""

non_review["shap_values"] = ""

non_review["shap_risk_factors"] = ""


# ============================================================
# COMBINE
# ============================================================

final = pd.concat(
    [
        non_review,
        review_data,
    ],
    axis=0
)

final = final.sort_index()


# ============================================================
# SAVE
# ============================================================

final.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INVESTIGATION SUMMARY")
print("=" * 70)

print("Total transactions:", len(final))

print(
    "Investigated:",
    (final["investigation_status"] == "INVESTIGATED").sum()
)

print(
    "Not required:",
    (final["investigation_status"] == "NOT_REQUIRED").sum()
)

print("\nSHAP columns created:")

print("✓ top_risk_feature")
print("✓ top_shap_contribution")
print("✓ shap_explanation")
print("✓ shap_features")
print("✓ shap_values")
print("✓ shap_risk_factors")


# ============================================================
# SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE SHAP EXPLANATIONS")
print("=" * 70)

sample = final[
    final["investigation_status"] == "INVESTIGATED"
].head(5)

for _, row in sample.iterrows():

    print("\nTransaction:", row["transaction_id"])

    print(
        "Top feature:",
        row["top_risk_feature"]
    )

    print(
        "Top SHAP:",
        row["top_shap_contribution"]
    )

    print(
        "SHAP factors:",
        row["shap_risk_factors"]
    )


print("\n" + "=" * 70)
print("SHAP INVESTIGATOR COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nNext step: AI INVESTIGATOR / LLM EXPLANATION")