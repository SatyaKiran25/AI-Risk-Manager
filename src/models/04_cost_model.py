from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "test_calibrated_predictions.csv"
OUTPUT_FILE = DATA_DIR / "test_cost_analysis.csv"


# ============================================================
# DEMO BUSINESS COSTS
# ============================================================

# Simulation assumptions for this project.
# These can later be replaced with real business costs.

FRAUD_LOSS = 5000.0
FALSE_POSITIVE_COST = 250.0
REVIEW_COST = 50.0


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

if "transaction_id" not in df.columns:
    df.insert(
        0,
        "transaction_id",
        [f"TXN_{i+1}" for i in range(len(df))]
    )

PROBABILITY_COLUMN = "calibrated_risk_probability"


# ============================================================
# EXPECTED COST CALCULATIONS
# ============================================================

# ------------------------------------------------------------
# ALLOW
# ------------------------------------------------------------
# If fraud is allowed, expected loss is:
#
# P(fraud) × fraud loss

df["allow_cost"] = (
    df[PROBABILITY_COLUMN] * FRAUD_LOSS
)


# ------------------------------------------------------------
# BLOCK
# ------------------------------------------------------------
# If legitimate transaction is blocked:
#
# P(legitimate) × false-positive cost

df["block_cost"] = (
    (1 - df[PROBABILITY_COLUMN])
    * FALSE_POSITIVE_COST
)


# ------------------------------------------------------------
# REVIEW
# ------------------------------------------------------------

df["review_cost"] = REVIEW_COST


# ============================================================
# COST-BASED RECOMMENDATION
# ============================================================

cost_columns = [
    "allow_cost",
    "review_cost",
    "block_cost",
]

df["recommended_action"] = (
    df[cost_columns]
    .idxmin(axis=1)
    .map({
        "allow_cost": "ALLOW",
        "review_cost": "REVIEW",
        "block_cost": "BLOCK",
    })
)


# ============================================================
# EXPECTED COST
# ============================================================

df["expected_cost"] = (
    df[cost_columns]
    .min(axis=1)
)


# ============================================================
# INVESTIGATION RISK SCORE
# ============================================================
#
# IMPORTANT:
#
# calibrated_risk_probability = MODEL PROBABILITY
#
# risk_score = INVESTIGATION SEVERITY SCORE
#
# They intentionally represent different things.
#
# Risk score combines:
#   70% calibrated fraud probability
#   30% contextual / behavioral risk signals
#
# This score is for reviewer prioritization and dashboard
# presentation. It does NOT replace calibrated probability
# for routing decisions.
# ============================================================


def clip01(value):
    """
    Convert a numeric value into the range [0, 1].
    """
    try:
        value = float(value)
    except (ValueError, TypeError):
        return 0.0

    return max(0.0, min(1.0, value))


def calculate_contextual_risk(row):
    """
    Calculate contextual transaction risk from existing
    behavioral and transaction-level risk signals.

    Returns a value between 0 and 1.
    """

    signals = []

    # --------------------------------------------------------
    # Merchant risk
    # merchant_risk_tier is expected to be 0-3.
    # --------------------------------------------------------

    if "merchant_risk_tier" in row.index:
        merchant_risk = clip01(
            float(row["merchant_risk_tier"]) / 3.0
        )
        signals.append(merchant_risk)

    # --------------------------------------------------------
    # New device
    # --------------------------------------------------------

    if "new_device" in row.index:
        signals.append(
            clip01(row["new_device"])
        )

    # --------------------------------------------------------
    # New location
    # --------------------------------------------------------

    if "new_location" in row.index:
        signals.append(
            clip01(row["new_location"])
        )

    # --------------------------------------------------------
    # Location mismatch
    # --------------------------------------------------------

    if "location_mismatch" in row.index:
        signals.append(
            clip01(row["location_mismatch"])
        )

    # --------------------------------------------------------
    # New merchant
    # --------------------------------------------------------

    if "new_merchant" in row.index:
        signals.append(
            clip01(row["new_merchant"])
        )

    # --------------------------------------------------------
    # Transaction velocity anomaly
    # --------------------------------------------------------

    if "velocity_anomaly" in row.index:
        signals.append(
            clip01(row["velocity_anomaly"])
        )

    # --------------------------------------------------------
    # Time-of-day anomaly
    # --------------------------------------------------------

    if "time_of_day_anomaly" in row.index:
        signals.append(
            clip01(row["time_of_day_anomaly"])
        )

    # --------------------------------------------------------
    # Amount anomaly
    #
    # amount_z_score is converted into 0-1.
    # 0 = normal
    # 3+ = highly unusual
    # --------------------------------------------------------

    if "amount_z_score" in row.index:

        try:
            z = abs(float(row["amount_z_score"]))
        except (ValueError, TypeError):
            z = 0.0

        amount_risk = clip01(z / 3.0)

        signals.append(amount_risk)

    # --------------------------------------------------------
    # If no contextual features are available
    # --------------------------------------------------------

    if not signals:
        return 0.0

    return sum(signals) / len(signals)


df["contextual_risk"] = (
    df.apply(
        calculate_contextual_risk,
        axis=1
    )
)


# ------------------------------------------------------------
# Final risk score
# ------------------------------------------------------------

probability_score = (
    df[PROBABILITY_COLUMN] * 70
)

context_score = (
    df["contextual_risk"].clip(0, 1) * 30
)

df["risk_score"] = (
    probability_score + context_score
).clip(0, 100).round(2)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("AI RISK MANAGER - COST MODEL")
print("=" * 70)

print("\nBusiness cost assumptions:")

print(
    f"Fraud loss       : ₹{FRAUD_LOSS:,.2f}"
)

print(
    f"False positive   : ₹{FALSE_POSITIVE_COST:,.2f}"
)

print(
    f"Human review     : ₹{REVIEW_COST:,.2f}"
)


# ============================================================
# DECISION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("COST-BASED RECOMMENDATION DISTRIBUTION")
print("=" * 70)

print(
    df["recommended_action"]
    .value_counts()
)


# ============================================================
# SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE COST & RISK RESULTS")
print("=" * 70)

columns_to_show = [
    PROBABILITY_COLUMN,
    "contextual_risk",
    "risk_score",
    "allow_cost",
    "review_cost",
    "block_cost",
    "expected_cost",
    "recommended_action",
]

print(
    df[columns_to_show]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("COST MODEL COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)