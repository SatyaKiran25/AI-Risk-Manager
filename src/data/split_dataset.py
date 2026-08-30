from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_ml_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# We use the stable early time window.
# PaySim steps 1-520 contain the densest/stablest region
# before the large fraud-rate increase seen later.

START_STEP = 126
END_STEP = 400

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("AI RISK MANAGER - TEMPORAL DATASET SPLIT")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset:")
print("Shape:", df.shape)

print(
    "Step range:",
    df["step"].min(),
    "→",
    df["step"].max()
)


# ============================================================
# SELECT STABLE TIME WINDOW
# ============================================================

df = df[
    (df["step"] >= START_STEP) &
    (df["step"] <= END_STEP)
].copy()

df = df.sort_values("step").reset_index(drop=True)

print("\nSelected stable window:")
print(
    "Step range:",
    df["step"].min(),
    "→",
    df["step"].max()
)

print("Rows:", len(df))

print(
    "Fraud rate:",
    f"{df['is_fraud'].mean():.4%}"
)


# ============================================================
# CREATE TIME-BASED BOUNDARIES
# ============================================================

min_step = df["step"].min()
max_step = df["step"].max()

total_range = max_step - min_step

train_end = (
    min_step
    + total_range * TRAIN_RATIO
)

validation_end = (
    min_step
    + total_range
    * (TRAIN_RATIO + VALIDATION_RATIO)
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train = df[
    df["step"] <= train_end
].copy()

validation = df[
    (df["step"] > train_end) &
    (df["step"] <= validation_end)
].copy()

test = df[
    df["step"] > validation_end
].copy()


# ============================================================
# REPORT FUNCTION
# ============================================================

def report(name, data):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Rows:", len(data))

    print(
        "Fraud:",
        int(data["is_fraud"].sum())
    )

    print(
        "Fraud rate:",
        f"{data['is_fraud'].mean():.4%}"
    )

    print(
        "Step range:",
        data["step"].min(),
        "→",
        data["step"].max()
    )


# ============================================================
# REPORT SPLITS
# ============================================================

report("TRAIN", train)

report("VALIDATION", validation)

report("TEST", test)


# ============================================================
# CHECK FOR OVERLAPPING TIME
# ============================================================

assert train["step"].max() < validation["step"].min()

assert validation["step"].max() < test["step"].min()

print("\nTemporal separation: PASS")


# ============================================================
# CHECK DUPLICATES
# ============================================================

if "source_row_id" in df.columns:

    train_ids = set(train["source_row_id"])
    val_ids = set(validation["source_row_id"])
    test_ids = set(test["source_row_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("source_row_id overlap: PASS")


# ============================================================
# SAVE DATASETS
# ============================================================

train_path = OUTPUT_DIR / "train_v2.csv"
validation_path = OUTPUT_DIR / "validation_v2.csv"
test_path = OUTPUT_DIR / "test_v2.csv"

train.to_csv(
    train_path,
    index=False
)

validation.to_csv(
    validation_path,
    index=False
)

test.to_csv(
    test_path,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("DATASET SPLIT COMPLETE")
print("=" * 60)

print("\nCreated:")

print(train_path)

print(validation_path)

print(test_path)

print("\nThese are now the datasets used by baseline.py.")