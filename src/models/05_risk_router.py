from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "test_cost_analysis.csv"
OUTPUT_FILE = DATA_DIR / "test_routed.csv"


# ============================================================
# ROUTER SETTINGS
# ============================================================

# Very high probability of fraud → automatic BLOCK
BLOCK_RISK_THRESHOLD = 0.95

# Medium / uncertain probability → HUMAN REVIEW
REVIEW_RISK_THRESHOLD = 0.10


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# SAFETY CHECK
# ============================================================

required_columns = [
    "calibrated_risk_probability",
    "risk_score",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# ROUTING LOGIC
# ============================================================

def route_transaction(row):

    probability = float(
        row["calibrated_risk_probability"]
    )

    # --------------------------------------------------------
    # HIGH CONFIDENCE FRAUD
    # --------------------------------------------------------

    if probability >= BLOCK_RISK_THRESHOLD:

        return "BLOCK"

    # --------------------------------------------------------
    # MEDIUM / UNCERTAIN RISK
    # --------------------------------------------------------

    elif probability >= REVIEW_RISK_THRESHOLD:

        return "REVIEW"

    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    else:

        return "ALLOW"


df["router_decision"] = (
    df.apply(
        route_transaction,
        axis=1
    )
)


# ============================================================
# ROUTING REASON
# ============================================================

def routing_reason(row):

    probability = float(
        row["calibrated_risk_probability"]
    )

    if probability >= BLOCK_RISK_THRESHOLD:

        return (
            "High-confidence fraud probability; "
            "automatic block recommended."
        )

    elif probability >= REVIEW_RISK_THRESHOLD:

        return (
            "Moderate or uncertain fraud probability; "
            "requires human investigation."
        )

    else:

        return (
            "Low fraud probability; "
            "transaction can proceed automatically."
        )


df["routing_reason"] = (
    df.apply(
        routing_reason,
        axis=1
    )
)


# ============================================================
# ROUTING DISTRIBUTION
# ============================================================

print("=" * 70)
print("AI RISK MANAGER - RISK & COST ROUTER")
print("=" * 70)

print("\nRouting distribution:")

print(
    df["router_decision"]
    .value_counts()
)


# ============================================================
# SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE ROUTING DECISIONS")
print("=" * 70)

columns = [
    "calibrated_risk_probability",
    "risk_score",
    "contextual_risk",
    "allow_cost",
    "review_cost",
    "block_cost",
    "expected_cost",
    "router_decision",
    "routing_reason",
]

available_columns = [
    col
    for col in columns
    if col in df.columns
]

print(
    df[available_columns]
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
print("RISK ROUTER COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)