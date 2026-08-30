from pathlib import Path
import pandas as pd
import numpy as np


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
validation = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)


# ============================================================
# FEATURES WE NEED TO AUDIT
# ============================================================

features = [
    "txns_prev_1h",
    "txns_prev_24h",
    "amount_z_score",
    "customer_std_amount_prior",
    "velocity_anomaly",
    "new_device",
    "new_location",
    "new_merchant",
    "location_mismatch",
]


# ============================================================
# BASIC DATASET AUDIT
# ============================================================

print("=" * 70)
print("AI RISK MANAGER - FINAL FEATURE AUDIT")
print("=" * 70)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Rows:", len(df))

    print(
        "Fraud rate:",
        f"{df['is_fraud'].mean():.4%}"
    )

    print(
        "Step range:",
        df["step"].min(),
        "→",
        df["step"].max()
    )


# ============================================================
# CHECK REQUIRED FEATURES EXIST
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXISTENCE CHECK")
print("=" * 70)

for feature in features:

    exists = all(
        feature in df.columns
        for df in [train, validation, test]
    )

    print(
        f"{feature:<30} : "
        f"{'PASS' if exists else 'MISSING'}"
    )


# ============================================================
# FEATURE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION")
print("=" * 70)

for feature in features:

    print("\n" + "-" * 70)
    print(feature)

    for name, df in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:

        s = df[feature]

        print(
            f"{name:<12} "
            f"mean={s.mean():.4f} | "
            f"std={s.std():.4f} | "
            f"min={s.min():.4f} | "
            f"median={s.median():.4f} | "
            f"max={s.max():.4f} | "
            f"unique={s.nunique()}"
        )


# ============================================================
# ZERO / CONSTANT FEATURE CHECK
# ============================================================

print("\n" + "=" * 70)
print("CONSTANT / ZERO FEATURE CHECK")
print("=" * 70)

for feature in features:

    print(f"\n{feature}")

    for name, df in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:

        s = df[feature]

        all_zero = (s == 0).all()
        constant = s.nunique() <= 1

        if all_zero:
            status = "FAIL - ALL ZERO"
        elif constant:
            status = "WARNING - CONSTANT"
        else:
            status = "PASS"

        print(
            f"  {name:<12}: {status}"
        )


# ============================================================
# MISSING / INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING / INFINITE VALUE CHECK")
print("=" * 70)

for feature in features:

    print(f"\n{feature}")

    for name, df in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:

        missing = df[feature].isna().sum()

        infinite = np.isinf(
            df[feature]
        ).sum()

        print(
            f"  {name:<12}: "
            f"missing={missing}, "
            f"infinite={infinite}"
        )


# ============================================================
# FRAUD VS LEGITIMATE FEATURE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FRAUD vs LEGITIMATE FEATURE MEANS")
print("=" * 70)

for feature in features:

    print("\n" + "-" * 70)
    print(feature)

    for name, df in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:

        fraud_mean = (
            df.loc[
                df["is_fraud"] == 1,
                feature
            ].mean()
        )

        legit_mean = (
            df.loc[
                df["is_fraud"] == 0,
                feature
            ].mean()
        )

        print(
            f"{name:<12} "
            f"legit={legit_mean:.4f} | "
            f"fraud={fraud_mean:.4f}"
        )


# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(
    "\nIf important features have meaningful variation "
    "and are not all zero/constant, the dataset is ready "
    "for the baseline model."
)

print(
    "\nIf any important feature shows "
    "'FAIL - ALL ZERO', do NOT train the model yet."
)