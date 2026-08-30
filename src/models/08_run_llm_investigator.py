"""
AI RISK MANAGER - BATCH LLM INVESTIGATOR

Pipeline:
    SHAP Investigator
          ↓
    REVIEW transactions
          ↓
    Gemini batch explanations
          ↓
    Cache
          ↓
    test_llm_investigated.csv

441 REVIEW transactions
25 transactions / batch
18 Gemini API calls
"""

import os
import json
import time
from typing import List

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

INPUT_PATH = "data/processed/test_investigated.csv"
OUTPUT_PATH = "data/processed/test_llm_investigated.csv"
CACHE_PATH = "data/processed/llm_batch_cache_v2.json"

MODEL_NAME = "gemini-3.5-flash-lite"

BATCH_SIZE = 25
MIN_SECONDS_BETWEEN_CALLS = 5
MAX_RETRIES = 4


# ============================================================
# OUTPUT SCHEMA
# ============================================================

class ExplanationOutput(BaseModel):
    transaction_id: str
    explanation: str
    evidence_summary: str


class BatchExplanationOutput(BaseModel):
    explanations: List[ExplanationOutput]


# ============================================================
# SHAP FEATURE EXPLANATION
# ============================================================

def build_shap_explanation(txn):
    """
    Convert important transaction features into a human-readable
    risk-factor description for Gemini.
    """

    factors = []

    merchant_risk_tier = txn.get("merchant_risk_tier", 0)

    if pd.notna(merchant_risk_tier) and float(merchant_risk_tier) >= 3:
        factors.append(
            "high-risk merchant tier (merchant_risk_tier=3)"
        )

    if float(txn.get("new_device", 0) or 0) == 1:
        factors.append(
            "new device (new_device=1)"
        )

    if float(txn.get("new_location", 0) or 0) == 1:
        factors.append(
            "new location (new_location=1)"
        )

    if float(txn.get("location_mismatch", 0) or 0) == 1:
        factors.append(
            "location mismatch (location_mismatch=1)"
        )

    if float(txn.get("new_merchant", 0) or 0) == 1:
        factors.append(
            "new merchant (new_merchant=1)"
        )

    if float(txn.get("velocity_anomaly", 0) or 0) == 1:
        factors.append(
            "unusual transaction velocity (velocity_anomaly=1)"
        )

    if float(txn.get("time_of_day_anomaly", 0) or 0) == 1:
        factors.append(
            f"unusual transaction time "
            f"(time_of_day_anomaly=1, hour={txn.get('hour', 'n/a')})"
        )

    amount_z = txn.get("amount_z_score", 0)

    if pd.notna(amount_z):
        try:
            amount_z = float(amount_z)

            if abs(amount_z) >= 3:
                factors.append(
                    f"unusually large transaction amount "
                    f"(amount_z_score={amount_z:.2f})"
                )
        except Exception:
            pass

    if not factors:
        factors.append(
            "no additional binary risk anomalies identified"
        )

    return "; ".join(factors)


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_batch_prompt(transactions):

    transaction_blocks = []

    for txn in transactions:

        transaction_id = txn.get("transaction_id", "UNKNOWN")
        amount = txn.get("amount", "n/a")
        risk_score = txn.get("risk_score", "n/a")
        probability = txn.get(
            "calibrated_risk_probability",
            "n/a"
        )

        shap_explanation = build_shap_explanation(txn)

        block = f"""
Transaction ID: {transaction_id}
Amount: {amount}
Fraud Probability: {probability}
Risk Score: {risk_score}

Risk Factors:
{shap_explanation}
"""

        transaction_blocks.append(block)

    all_transactions = "\n".join(transaction_blocks)

    prompt = f"""
You are a professional fraud investigation assistant
for a financial transaction risk-management system.

The transactions below have already been routed to REVIEW
by a risk engine.

For EACH transaction produce:

1. explanation
2. evidence_summary

STRICT REQUIREMENTS:

- Explain WHY the transaction requires review.
- Explicitly mention the actual named risk factors provided.
- Do not merely repeat the risk score and amount.
- If merchant_risk_tier=3 is present, explicitly mention
  "high-risk merchant tier".
- If new_device=1 is present, explicitly mention
  "new device".
- If new_location=1 is present, explicitly mention
  "new location".
- If location_mismatch=1 is present, explicitly mention
  "location mismatch".
- If new_merchant=1 is present, explicitly mention
  "new merchant".
- If velocity_anomaly=1 is present, explicitly mention
  "unusual transaction velocity".
- If time_of_day_anomaly=1 is present, explicitly mention
  "unusual transaction time".
- Mention the transaction amount only when useful.
- Do not invent facts.
- Do not claim that a transaction is definitely fraudulent.
- Do not mention SHAP.
- Do not mention machine learning.
- Do not mention AI or Gemini.
- Use neutral professional language.
- Explanation must be 2-3 sentences.
- Evidence summary must be 1-2 factual sentences.
- Preserve transaction_id exactly.
- Return exactly one object per transaction.

The explanation should be useful to a human fraud reviewer.

Example of GOOD explanation:

"This transaction was routed for review because it involves
a high-risk merchant tier and a new device. The combination
of these risk indicators increases the need for manual
verification before the payment is cleared."

Example of BAD explanation:

"This transaction was reviewed because the risk score is 50.42."

The second example is insufficient because it does not explain
the actual risk factors.

TRANSACTIONS:
{all_transactions}
"""

    return prompt


# ============================================================
# GEMINI CALL
# ============================================================

def generate_batch_explanations(transactions):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. "
            "Check your .env file."
        )

    client = genai.Client(api_key=api_key)

    prompt = build_batch_prompt(transactions)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchExplanationOutput,
                    temperature=0.2,
                ),
            )

            parsed = response.parsed

            if parsed is None:
                raise ValueError(
                    "Gemini returned no structured response."
                )

            results = []

            expected_ids = {
                str(txn["transaction_id"])
                for txn in transactions
            }

            returned_ids = set()

            for item in parsed.explanations:

                transaction_id = str(
                    item.transaction_id
                )

                if transaction_id not in expected_ids:
                    continue

                returned_ids.add(transaction_id)

                results.append({
                    "transaction_id": transaction_id,
                    "explanation": item.explanation,
                    "evidence_summary": item.evidence_summary,
                    "source": "gemini",
                    "error": None,
                })

            missing_ids = expected_ids - returned_ids

            if missing_ids:
                raise ValueError(
                    f"Gemini did not return explanations for "
                    f"{len(missing_ids)} transactions."
                )

            return results

        except Exception as e:

            last_error = e
            error_text = str(e)

            is_rate_limit = (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            )

            print()
            print("Gemini ERROR:")
            print(error_text)

            if attempt >= MAX_RETRIES:
                break

            if is_rate_limit:

                wait_time = 15 * attempt

                print(
                    f"Rate limit detected. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                wait_time = 5 * attempt

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

    raise RuntimeError(
        f"Gemini failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# CACHE
# ============================================================

def load_cache():

    if not os.path.exists(CACHE_PATH):
        return {}

    try:

        with open(
            CACHE_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            cache = json.load(f)

        if not isinstance(cache, dict):
            return {}

        return cache

    except Exception as e:

        print(
            f"Warning: could not load cache: {e}"
        )

        return {}


def save_cache(cache):

    os.makedirs(
        os.path.dirname(CACHE_PATH),
        exist_ok=True
    )

    temp_path = CACHE_PATH + ".tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False
        )

    os.replace(
        temp_path,
        CACHE_PATH
    )

    print(
        f"Cache saved: {len(cache)} transactions"
    )


# ============================================================
# FALLBACK
# ============================================================

def fallback_explanation(transaction):

    transaction_id = transaction.get(
        "transaction_id",
        "UNKNOWN"
    )

    risk_factors = build_shap_explanation(
        transaction
    )

    return {
        "transaction_id": str(transaction_id),

        "explanation": (
            "This transaction requires manual review due to "
            f"the following identified risk indicators: "
            f"{risk_factors}. "
            "Manual verification is recommended before "
            "final clearance."
        ),

        "evidence_summary": (
            f"Identified risk indicators include: "
            f"{risk_factors}."
        ),

        "source": "fallback",

        "error": "Gemini unavailable",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI RISK MANAGER - BATCH LLM INVESTIGATOR")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # LOAD INPUT
    # --------------------------------------------------------

    print("Loading investigated transactions...")

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Total transactions: {len(df)}"
    )

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "transaction_id",
        "router_decision",
        "amount",
        "risk_score",
        "calibrated_risk_probability",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # SELECT REVIEW
    # --------------------------------------------------------

    review_df = df[
        df["router_decision"]
        .astype(str)
        .str.upper()
        .eq("REVIEW")
    ].copy()

    print(
        f"REVIEW transactions: {len(review_df)}"
    )

    total_batches = (
        len(review_df) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(
        f"Batch size: {BATCH_SIZE} transactions"
    )

    print(
        f"Total Gemini API calls required: "
        f"{total_batches}"
    )

    print()

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:

        print("Gemini API key found.")

    else:

        print(
            "WARNING: GEMINI_API_KEY not found."
        )

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    cache = load_cache()

    print(
        f"Already cached LLM explanations: "
        f"{len(cache)}"
    )

    # --------------------------------------------------------
    # PREPARE TRANSACTIONS
    # --------------------------------------------------------

    transactions = review_df.to_dict(
        orient="records"
    )

    remaining = [
        txn
        for txn in transactions
        if str(txn["transaction_id"])
        not in cache
    ]

    print(
        f"Transactions remaining: "
        f"{len(remaining)}"
    )

    print()

    # --------------------------------------------------------
    # PROCESS BATCHES
    # --------------------------------------------------------

    for batch_start in range(
        0,
        len(remaining),
        BATCH_SIZE
    ):

        batch = remaining[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch_number = (
            batch_start // BATCH_SIZE
        ) + 1

        print(
            f"[Batch {batch_number}/{total_batches}] "
            f"Processing {len(batch)} transactions..."
        )

        try:

            results = generate_batch_explanations(
                batch
            )

            for result in results:

                cache[
                    str(result["transaction_id"])
                ] = result

            print(
                f"[Batch {batch_number}/{total_batches}] "
                "LLM SUCCESS"
            )

            save_cache(cache)

        except Exception as e:

            print(
                f"[Batch {batch_number}/{total_batches}] "
                f"LLM FAILED: {e}"
            )

            print(
                "Using fallback explanations "
                "for this batch."
            )

            for txn in batch:

                result = fallback_explanation(
                    txn
                )

                cache[
                    str(txn["transaction_id"])
                ] = result

            save_cache(cache)

        # ----------------------------------------------------
        # RATE LIMIT DELAY
        # ----------------------------------------------------

        if (
            batch_start + BATCH_SIZE
            < len(remaining)
        ):

            print(
                f"Waiting "
                f"{MIN_SECONDS_BETWEEN_CALLS} seconds..."
            )

            time.sleep(
                MIN_SECONDS_BETWEEN_CALLS
            )

    # --------------------------------------------------------
    # MERGE RESULTS
    # --------------------------------------------------------

    print()
    print("Merging explanations...")
    print()

    df["llm_explanation"] = None
    df["evidence_summary"] = None
    df["llm_source"] = None
    df["llm_error"] = None

    for idx, row in df.iterrows():

        transaction_id = str(
            row["transaction_id"]
        )

        if transaction_id in cache:

            result = cache[transaction_id]

            df.at[
                idx,
                "llm_explanation"
            ] = result.get(
                "explanation"
            )

            df.at[
                idx,
                "evidence_summary"
            ] = result.get(
                "evidence_summary"
            )

            df.at[
                idx,
                "llm_source"
            ] = result.get(
                "source"
            )

            df.at[
                idx,
                "llm_error"
            ] = result.get(
                "error"
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    review_output = df[
        df["router_decision"]
        .astype(str)
        .str.upper()
        .eq("REVIEW")
    ]

    gemini_count = (
        review_output["llm_source"]
        .eq("gemini")
        .sum()
    )

    fallback_count = (
        review_output["llm_source"]
        .eq("fallback")
        .sum()
    )

    explanation_count = (
        review_output["llm_explanation"]
        .notna()
        .sum()
    )

    print("=" * 70)
    print("LLM INVESTIGATOR COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Total transactions: {len(df)}"
    )

    print(
        f"REVIEW transactions: "
        f"{len(review_output)}"
    )

    print(
        f"LLM explanations: "
        f"{explanation_count}"
    )

    print(
        f"Gemini explanations: "
        f"{gemini_count}"
    )

    print(
        f"Fallback explanations: "
        f"{fallback_count}"
    )

    print()

    print("Output saved:")
    print(OUTPUT_PATH)

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()