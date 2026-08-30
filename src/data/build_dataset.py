# ============================================================
# AI RISK MANAGER
# Dataset Construction Pipeline
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# 2. LOAD PAYSIM
# ============================================================

def load_paysim():

    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found inside {RAW_DIR}"
        )

    # Prefer PaySim-looking filename
    paysim_files = [
        f for f in csv_files
        if "paysim" in f.name.lower()
        or "ps_" in f.name.lower()
    ]

    file_path = paysim_files[0] if paysim_files else csv_files[0]

    print(f"\nLoading dataset:")
    print(file_path)

    df = pd.read_csv(file_path)

    print(f"Raw shape: {df.shape}")

    return df


# ============================================================
# 3. CLEAN PAYSim
# ============================================================

def clean_paysim(df):

    df = df.copy()

    # Rename PaySim columns
    rename_map = {
        "step": "step",
        "type": "payment_type",
        "amount": "amount",
        "nameOrig": "customer_id",
        "oldbalanceOrg": "old_balance",
        "newbalanceOrig": "new_balance",
        "nameDest": "merchant_id",
        "oldbalanceDest": "merchant_old_balance",
        "newbalanceDest": "merchant_new_balance",
        "isFraud": "is_fraud",
        "isFlaggedFraud": "is_flagged_fraud"
    }

    df = df.rename(columns=rename_map)

    required = [
        "step",
        "payment_type",
        "amount",
        "customer_id",
        "merchant_id",
        "is_fraud"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required PaySim columns: {missing}"
        )

    # Remove impossible rows
    df = df[df["amount"] > 0].copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Sort chronologically
    df = df.sort_values("step").reset_index(drop=True)

    # Binary fraud label
    df["is_fraud"] = df["is_fraud"].astype(int)

    print("\nAfter cleaning:")
    print(f"Shape: {df.shape}")
    print(f"Fraud transactions: {df['is_fraud'].sum()}")
    print(
        f"Fraud rate: {df['is_fraud'].mean():.4%}"
    )

    return df


# ============================================================
# 4. CREATE CUSTOMER PROFILES
# IMPORTANT:
# Profiles are calculated from FULL cleaned PaySim data.
# They are NOT calculated from the sampled dataset.
# ============================================================

def create_customer_profiles(df):

    customer_profiles = (
        df.groupby("customer_id")
        .agg(
            customer_txn_count=("amount", "count"),
            customer_avg_amount=("amount", "mean"),
            customer_std_amount=("amount", "std"),
            customer_max_amount=("amount", "max")
        )
        .reset_index()
    )

    customer_profiles["customer_std_amount"] = (
        customer_profiles["customer_std_amount"]
        .fillna(0)
    )

    return customer_profiles


# ============================================================
# 5. CREATE MERCHANT PROFILES
# ============================================================

def create_merchant_profiles(df):

    merchant_profiles = (
        df.groupby("merchant_id")
        .agg(
            merchant_txn_count=("amount", "count"),
            merchant_avg_amount=("amount", "mean"),
            merchant_fraud_rate=("is_fraud", "mean")
        )
        .reset_index()
    )

    # Risk tier based on merchant fraud rate
    merchant_profiles["merchant_risk_tier"] = pd.qcut(
        merchant_profiles["merchant_fraud_rate"].rank(
            method="first"
        ),
        q=3,
        labels=[1, 2, 3]
    ).astype(int)

    return merchant_profiles


# ============================================================
# 6. SYNTHETIC DEVICE PROFILES
# ============================================================

def create_device_profiles(df):

    customers = df["customer_id"].unique()

    # Give each customer a stable device
    device_profiles = pd.DataFrame({
        "customer_id": customers
    })

    device_profiles["device_id"] = (
        "DEV_" +
        pd.Series(
            np.arange(len(customers)),
            index=device_profiles.index
        ).astype(str)
    )

    return device_profiles


# ============================================================
# 7. SYNTHETIC LOCATION PROFILES
# ============================================================

def create_location_profiles(df):

    customers = df["customer_id"].unique()

    indian_locations = [
        "IN_DEL",
        "IN_MUM",
        "IN_BLR",
        "IN_HYD",
        "IN_CHE",
        "IN_PUN",
        "IN_KOL",
        "IN_AHM"
    ]

    location_profiles = pd.DataFrame({
        "customer_id": customers
    })

    location_profiles["home_location"] = rng.choice(
        indian_locations,
        size=len(customers)
    )

    return location_profiles


# ============================================================
# 8. STRATIFIED SAMPLING
#
# Keep ALL fraud.
# Keep approximately 3% of legitimate transactions.
#
# Profiles are already calculated from FULL history.
# ============================================================

def sample_transactions(df, legitimate_fraction=0.03):

    fraud = df[df["is_fraud"] == 1].copy()

    legitimate = df[df["is_fraud"] == 0].copy()

    legitimate_sample = legitimate.sample(
        frac=legitimate_fraction,
        random_state=RANDOM_STATE
    )

    sampled = pd.concat(
        [fraud, legitimate_sample],
        ignore_index=True
    )

    sampled = sampled.sort_values("step").reset_index(drop=True)

    print("\nSampling:")
    print(f"Fraud kept: {len(fraud):,}")
    print(
        f"Legitimate kept: {len(legitimate_sample):,}"
    )
    print(f"Final sampled rows: {len(sampled):,}")
    print(
        f"Fraud rate after sampling: "
        f"{sampled['is_fraud'].mean():.4%}"
    )

    return sampled


# ============================================================
# 9. MERGE SYNTHETIC CONTEXT
# ============================================================

def add_synthetic_context(
    sampled,
    customer_profiles,
    merchant_profiles,
    device_profiles,
    location_profiles
):

    df = sampled.copy()

    # Customer profile
    df = df.merge(
        customer_profiles,
        on="customer_id",
        how="left"
    )

    # Merchant profile
    df = df.merge(
        merchant_profiles,
        on="merchant_id",
        how="left"
    )

    # Device
    df = df.merge(
        device_profiles,
        on="customer_id",
        how="left"
    )

    # Location
    df = df.merge(
        location_profiles,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------------
    # Synthetic transaction-level device/location
    # --------------------------------------------------------

    df["transaction_device_id"] = df["device_id"]
    df["transaction_location"] = df["home_location"]

    # --------------------------------------------------------
    # New device / location
    # --------------------------------------------------------

    # Start with normal transactions
    df["new_device"] = 0
    df["new_location"] = 0

    # Inject approximately 6% new-device transactions
    device_mask = rng.random(len(df)) < 0.06

    df.loc[device_mask, "transaction_device_id"] = (
        "NEWDEV_" +
        pd.Series(
            np.arange(device_mask.sum()),
            index=df.index[device_mask]
        ).astype(str)
    )

    df.loc[device_mask, "new_device"] = 1

    # Inject approximately 6% new-location transactions
    location_mask = rng.random(len(df)) < 0.06

    new_locations = [
        "IN_NEW_1",
        "IN_NEW_2",
        "IN_NEW_3",
        "IN_NEW_4"
    ]

    df.loc[
        location_mask,
        "transaction_location"
    ] = rng.choice(
        new_locations,
        size=location_mask.sum()
    )

    df.loc[location_mask, "new_location"] = 1

    # Location mismatch
    df["location_mismatch"] = (
        df["transaction_location"]
        != df["home_location"]
    ).astype(int)

    # New merchant
    df["new_merchant"] = (
        rng.random(len(df)) < 0.01
    ).astype(int)

    return df


# ============================================================
# 10. TRANSACTION HISTORY FEATURES
# ============================================================

def add_history_features(df):

    df = df.copy()

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    df = df.sort_values(
        ["customer_id", "step"]
    ).reset_index(drop=True)

    # ========================================================
    # 1. PREVIOUS TRANSACTION COUNT
    # ========================================================

    df["customer_txn_count_prior"] = (
        df.groupby("customer_id")
        .cumcount()
    )

    # ========================================================
    # 2. PREVIOUS 1-HOUR TRANSACTIONS
    # ========================================================
    # PaySim step = 1 hour.
    #
    # Count previous transactions by the same customer
    # occurring in the previous hour.

    df["txns_prev_1h"] = (
        df.groupby("customer_id")["step"]
        .transform(
            lambda x:
            x.rolling(
                window=2,
                min_periods=1
            ).count() - 1
        )
    )

    # ========================================================
    # 3. PREVIOUS 24-HOUR TRANSACTIONS
    # ========================================================

    def calculate_24h_count(group):

        steps = group["step"].to_numpy()

        result = np.zeros(
            len(group),
            dtype=np.int32
        )

        left = 0

        for i in range(len(steps)):

            while (
                left < i
                and steps[left] < steps[i] - 24
            ):
                left += 1

            result[i] = i - left

        return pd.Series(
            result,
            index=group.index
        )

    df["txns_prev_24h"] = (
        df.groupby(
            "customer_id",
            group_keys=False
        )
        .apply(
            calculate_24h_count,
            include_groups=False
        )
        .reset_index(level=0, drop=True)
    )

    # ========================================================
    # 4. PREVIOUS CUSTOMER AVERAGE AMOUNT
    # ========================================================

    df["customer_avg_amount_prior"] = (
        df.groupby("customer_id")["amount"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    # ========================================================
    # 5. PREVIOUS CUSTOMER STANDARD DEVIATION
    # ========================================================

    df["customer_std_amount_prior"] = (
        df.groupby("customer_id")["amount"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .std()
        )
    )

    # ========================================================
    # 6. HANDLE FIRST TRANSACTION
    # ========================================================

    df["customer_avg_amount_prior"] = (
        df["customer_avg_amount_prior"]
        .fillna(0)
    )

    df["customer_std_amount_prior"] = (
        df["customer_std_amount_prior"]
        .fillna(0)
    )

    # ========================================================
    # 7. AMOUNT Z-SCORE
    # ========================================================

    df["amount_z_score"] = 0.0

    valid_history = (
        df["customer_std_amount_prior"] > 0
    )

    df.loc[valid_history, "amount_z_score"] = (
        (
            df.loc[
                valid_history,
                "amount"
            ]
            -
            df.loc[
                valid_history,
                "customer_avg_amount_prior"
            ]
        )
        /
        df.loc[
            valid_history,
            "customer_std_amount_prior"
        ]
    )

    # Avoid extreme values
    df["amount_z_score"] = (
        df["amount_z_score"]
        .clip(-10, 10)
    )

    # ========================================================
    # 8. VELOCITY ANOMALY
    # ========================================================

    df["velocity_anomaly"] = (
        (
            df["txns_prev_1h"] >= 3
        )
        |
        (
            df["txns_prev_24h"] >= 10
        )
    ).astype(int)

    # ========================================================
    # 9. CLEAN NUMERICAL TYPES
    # ========================================================

    df["txns_prev_1h"] = (
        df["txns_prev_1h"]
        .fillna(0)
        .astype(int)
    )

    df["txns_prev_24h"] = (
        df["txns_prev_24h"]
        .fillna(0)
        .astype(int)
    )

    df["customer_txn_count_prior"] = (
        df["customer_txn_count_prior"]
        .fillna(0)
        .astype(int)
    )

    return df

# ============================================================
# 11. TIME FEATURES
# ============================================================

def add_time_features(df):

    # PaySim step represents an hourly time unit.
    df["hour"] = df["step"] % 24

    df["day_of_week"] = (
        df["step"] // 24
    ) % 7

    # Unusual transaction hours
    df["time_of_day_anomaly"] = (
        (df["hour"] < 5) |
        (df["hour"] > 23)
    ).astype(int)

    return df


# ============================================================
# 12. FRAUD ARCHETYPE INJECTION
#
# These create controlled, explainable fraud patterns:
#
# 1. Card testing
# 2. Device/location account takeover
# 3. Synthetic identity
# 4. Look-alike legitimate transactions
# ============================================================

def inject_fraud_archetypes(df):

    df = df.copy()

    fraud_candidates = df[
        df["is_fraud"] == 1
    ].index.to_numpy(copy=True)

    if len(fraud_candidates) == 0:
        return df

    rng.shuffle(fraud_candidates)

    n = len(fraud_candidates)

    # Card testing
    n_card = int(n * 0.25)

    card_idx = fraud_candidates[:n_card]
    # --------------------------------------------------------
    # Card testing
    # --------------------------------------------------------

    n_card = int(n * 0.25)

    card_idx = fraud_candidates[:n_card]

    df.loc[card_idx, "txns_prev_1h"] = rng.integers(
        4, 9, size=n_card
    )

    df.loc[card_idx, "velocity_anomaly"] = 1

    # --------------------------------------------------------
    # Device/location ATO
    # --------------------------------------------------------

    start = n_card
    end = start + int(n * 0.25)

    ato_idx = fraud_candidates[start:end]

    df.loc[ato_idx, "new_device"] = 1
    df.loc[ato_idx, "new_location"] = 1
    df.loc[ato_idx, "location_mismatch"] = 1

    # --------------------------------------------------------
    # Synthetic identity
    # --------------------------------------------------------

    start = end
    end = start + int(n * 0.25)

    identity_idx = fraud_candidates[start:end]

    df.loc[
        identity_idx,
        "customer_txn_count_prior"
    ] = rng.integers(0, 2, size=len(identity_idx))

    df.loc[
        identity_idx,
        "new_merchant"
    ] = 1

    # --------------------------------------------------------
    # Look-alike fraud
    # --------------------------------------------------------

    # Remaining fraud deliberately receives less obvious
    # signals. This prevents the model from simply learning
    # "fraud = obvious anomaly".
    lookalike_idx = fraud_candidates[end:]

    df.loc[
        lookalike_idx,
        "time_of_day_anomaly"
    ] = 0

    df.loc[
        lookalike_idx,
        "new_device"
    ] = 0

    df.loc[
        lookalike_idx,
        "new_location"
    ] = 0

    return df


# ============================================================
# 13. FINAL FEATURE PREPARATION
# ============================================================

def prepare_final_dataset(df):

    df = df.copy()

    # Remove columns that should not enter ML model
    drop_columns = [
        "payment_type",
        "customer_id",
        "merchant_id",
        "device_id",
        "home_location",
        "transaction_device_id",
        "transaction_location",
        "previous_amount",
        "historical_amount_mean",
        "historical_amount_std",
        "merchant_fraud_rate",
        "merchant_txn_count",
        "customer_max_amount"
    ]

    df = df.drop(
        columns=[
            c for c in drop_columns
            if c in df.columns
        ]
    )

    # Rename merchant risk tier if needed
    df["merchant_risk_tier"] = (
        df["merchant_risk_tier"]
        .astype(int)
    )

    # Clean infinities
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill numerical missing values
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_columns:

        if col == "is_fraud":
            continue

        df[col] = df[col].fillna(
            df[col].median()
        )

    # Sort by time
    df = df.sort_values(
        "step"
    ).reset_index(drop=True)

    return df


# ============================================================
# 14. TEMPORAL TRAIN / VALIDATION / TEST SPLIT
#
# Same dense time window.
# Split by TIME RANGE, not random row count.
#
# 70% train
# 15% validation
# 15% test
# ============================================================

def temporal_split(df):

    df = df.sort_values("step").reset_index(drop=True)

    min_step = df["step"].min()
    max_step = df["step"].max()

    total_range = max_step - min_step

    train_end = min_step + total_range * 0.70
    val_end = min_step + total_range * 0.85

    train = df[
        df["step"] <= train_end
    ].copy()

    validation = df[
        (df["step"] > train_end) &
        (df["step"] <= val_end)
    ].copy()

    test = df[
        df["step"] > val_end
    ].copy()

    return train, validation, test


# ============================================================
# 15. DATASET REPORT
# ============================================================

def dataset_report(name, df):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Shape:", df.shape)

    print(
        "Fraud:",
        int(df["is_fraud"].sum())
    )

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
# 16. MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("AI RISK MANAGER - DATASET BUILD PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Load
    # --------------------------------------------------------

    raw = load_paysim()

    # --------------------------------------------------------
    # Step 2: Clean
    # --------------------------------------------------------

    cleaned = clean_paysim(raw)

    # --------------------------------------------------------
    # Step 3: Profiles from FULL HISTORY
    # --------------------------------------------------------

    print("\nCreating profiles from FULL transaction history...")

    customer_profiles = create_customer_profiles(
        cleaned
    )

    merchant_profiles = create_merchant_profiles(
        cleaned
    )

    device_profiles = create_device_profiles(
        cleaned
    )

    location_profiles = create_location_profiles(
        cleaned
    )

    # --------------------------------------------------------
    # Step 4: Sampling
    # --------------------------------------------------------

    sampled = sample_transactions(
        cleaned,
        legitimate_fraction=0.03
    )

    # --------------------------------------------------------
    # Step 5: Synthetic context
    # --------------------------------------------------------

    print("\nAdding synthetic customer/device/location context...")

    contextual = add_synthetic_context(
        sampled,
        customer_profiles,
        merchant_profiles,
        device_profiles,
        location_profiles
    )

    # --------------------------------------------------------
    # Step 6: Archetype injection
    # --------------------------------------------------------

    print("\nInjecting fraud archetypes...")

    contextual = inject_fraud_archetypes(
        contextual
    )

    # --------------------------------------------------------
    # Step 7: Transaction history
    # --------------------------------------------------------

    print("\nCreating transaction history features...")

    contextual = add_history_features(
        contextual
    )

    # --------------------------------------------------------
    # Step 8: Time features
    # --------------------------------------------------------

    contextual = add_time_features(
        contextual
    )

    # --------------------------------------------------------
    # Step 9: Final ML dataset
    # --------------------------------------------------------

    final_df = prepare_final_dataset(
        contextual
    )

    # --------------------------------------------------------
    # Save complete dataset
    # --------------------------------------------------------

    final_path = (
        PROCESSED_DIR /
        "final_ml_dataset.csv"
    )

    final_df.to_csv(
        final_path,
        index=False
    )

    print("\nFinal dataset saved:")
    print(final_path)

    # --------------------------------------------------------
    # Step 10: Temporal split
    # --------------------------------------------------------

    train, validation, test = temporal_split(
        final_df
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    dataset_report(
        "TRAIN",
        train
    )

    dataset_report(
        "VALIDATION",
        validation
    )

    dataset_report(
        "TEST",
        test
    )

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    train.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    validation.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False
    )

    test.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    print("\n")
    print("=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)

    print("\nCreated files:")

    print(
        "1.",
        PROCESSED_DIR / "final_ml_dataset.csv"
    )

    print(
        "2.",
        PROCESSED_DIR / "train.csv"
    )

    print(
        "3.",
        PROCESSED_DIR / "validation.csv"
    )

    print(
        "4.",
        PROCESSED_DIR / "test.csv"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()