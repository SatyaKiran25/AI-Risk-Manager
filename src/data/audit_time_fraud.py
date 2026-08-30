from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_ml_dataset.csv"
)


df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("AI RISK MANAGER - TIME / FRAUD AUDIT")
print("=" * 70)

print("\nDataset:")
print("Rows:", len(df))
print("Step range:", df["step"].min(), "→", df["step"].max())


# ------------------------------------------------------------
# FRAUD RATE BY 25-STEP BLOCK
# ------------------------------------------------------------

df["step_block"] = (
    ((df["step"] - 1) // 25) * 25 + 1
)

audit = (
    df.groupby("step_block")
    .agg(
        transactions=("is_fraud", "size"),
        fraud_count=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean")
    )
    .reset_index()
)

audit["fraud_rate_percent"] = (
    audit["fraud_rate"] * 100
)

print("\nFraud rate by 25-step block:")
print(
    audit[
        [
            "step_block",
            "transactions",
            "fraud_count",
            "fraud_rate_percent"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# FIND BLOCKS WITH ENOUGH DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BLOCK SUMMARY")
print("=" * 70)

for _, row in audit.iterrows():

    print(
        f"Steps {int(row['step_block']):>3}–"
        f"{int(row['step_block']) + 24:<3} | "
        f"Rows: {int(row['transactions']):>6} | "
        f"Fraud: {int(row['fraud_count']):>5} | "
        f"Rate: {row['fraud_rate_percent']:.2f}%"
    )


# ------------------------------------------------------------
# OVERALL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL FRAUD RATE")
print("=" * 70)

print(
    f"{df['is_fraud'].mean() * 100:.4f}%"
)