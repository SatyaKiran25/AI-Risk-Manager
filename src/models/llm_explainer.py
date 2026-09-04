"""
AI RISK MANAGER - LLM EXPLANATION LAYER

Uses Google Gemini to generate human-readable explanations
for flagged transactions.

Architecture:
    Transaction
        ↓
    SHAP features
        ↓
    Gemini
        ↓
    Explanation + Evidence

Batch mode:
    25 transactions → 1 Gemini API call

Therefore:
    441 REVIEW transactions → 18 Gemini API calls
"""

import os
import time
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

# Transactions per Gemini API call
BATCH_SIZE = 25

# Keep safely below the free-tier request limit
MIN_SECONDS_BETWEEN_CALLS = 5

# Retry configuration
MAX_RETRIES = 4


# ============================================================
# OUTPUT SCHEMAS
# ============================================================

class ExplanationOutput(BaseModel):
    transaction_id: str
    explanation: str
    evidence_summary: str


class BatchExplanationOutput(BaseModel):
    explanations: List[ExplanationOutput]


# ============================================================
# HELPER - SAFE VALUE
# ============================================================

def safe_value(value, default="n/a"):
    """
    Convert missing / invalid values into a readable value.
    """

    if value is None:
        return default

    try:
        if str(value).lower() in ["nan", "none", ""]:
            return default
    except Exception:
        pass

    return value


# ============================================================
# SHAP FEATURE DESCRIPTION
# ============================================================

def build_risk_factor_text(txn):
    """
    Build an explicit, human-readable list of risk factors.

    FIX: this now reads directly from the transaction's actual ranked SHAP
    contributions (txn["shap_explanation"], already sorted by |contribution|
    in 06_shap_investigator.py) instead of a fixed set of hand-picked
    features with hardcoded thresholds.

    The previous version only ever surfaced merchant_risk_tier, new_device,
    new_location, location_mismatch, new_merchant, velocity_anomaly,
    time_of_day_anomaly, amount_z_score, and hour (and only when hour fell
    in a hardcoded 22:00-05:00 window) - day_of_week had NO handling at
    all, and any feature could have a large real SHAP contribution for a
    specific transaction yet never be mentioned, because the selection
    logic never consulted the actual SHAP values. That caused explanations
    to silently omit real top contributors (e.g. hour=+1.2154,
    day_of_week=+0.3690) while still calling out smaller or unrelated ones.

    Now: take the top N SHAP factors that INCREASE risk (positive
    contribution), by actual magnitude, whatever features they happen to
    be - then apply a friendly label only for readability, never as a
    filter that can drop a feature.
    """

    TOP_N = 4

    # Friendly labels are cosmetic only - every feature still gets included
    # based on its real SHAP contribution, whether or not it has a label here.
    FRIENDLY_LABELS = {
        "merchant_risk_tier": "high-risk merchant tier",
        "new_device": "transaction from a new device",
        "new_location": "transaction from a new location",
        "location_mismatch": "location mismatch detected",
        "new_merchant": "transaction involves a new merchant",
        "velocity_anomaly": "unusual transaction velocity",
        "time_of_day_anomaly": "unusual transaction time",
        "hour": "unusual hour of transaction",
        "day_of_week": "unusual day of week",
        "amount_z_score": "unusually large transaction amount",
    }

    shap_explanation = safe_value(txn.get("shap_explanation"))

    if shap_explanation == "n/a" or not str(shap_explanation).strip():
        return "No specific contextual risk factors were available."

    factors = []

    # Format produced by 06_shap_investigator.py:
    #   "feature=value (increased fraud risk, SHAP=0.1234) | feature2=..."
    # Already sorted by |SHAP contribution| descending.
    for entry in str(shap_explanation).split(" | "):
        entry = entry.strip()
        if not entry or "increased fraud risk" not in entry:
            continue  # only surface risk-INCREASING factors, matching the dashboard's "Risk Factors Detected" panel

        feature_name = entry.split("=", 1)[0].strip()
        label = FRIENDLY_LABELS.get(feature_name)

        if label:
            factors.append(f"{entry} \u2014 {label}")
        else:
            factors.append(entry)  # no friendly label yet, but NEVER dropped

        if len(factors) >= TOP_N:
            break

    if not factors:
        return "No risk-increasing factors were identified for this transaction."

    return "\n".join(f"- {factor}" for factor in factors)


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_batch_prompt(transactions):
    """
    Build one prompt containing up to 25 REVIEW transactions.

    The prompt explicitly requires Gemini to reference actual
    supplied risk factors instead of producing generic text.
    """

    transaction_blocks = []

    for txn in transactions:

        transaction_id = safe_value(
            txn.get("transaction_id"),
            "UNKNOWN"
        )

        amount = safe_value(
            txn.get("amount")
        )

        risk_score = safe_value(
            txn.get("risk_score")
        )

        probability = safe_value(
            txn.get("calibrated_risk_probability")
        )

        hour = safe_value(
            txn.get("hour")
        )

        risk_factors = build_risk_factor_text(txn)

        block = f"""
============================================================
TRANSACTION
============================================================

Transaction ID:
{transaction_id}

Transaction Amount:
{amount}

Calibrated Fraud Probability:
{probability}

Risk Score:
{risk_score}

Hour:
{hour}

IDENTIFIED RISK FACTORS:
{risk_factors}
"""

        transaction_blocks.append(block)

    all_transactions = "\n".join(
        transaction_blocks
    )

    prompt = f"""
You are an AI fraud investigation assistant working for a
payments risk-management system.

You will receive multiple transactions that have already been
classified as REVIEW by a fraud risk system.

Your task is to produce a concise, evidence-based explanation
for EACH transaction.

============================================================
CRITICAL EXPLANATION REQUIREMENTS
============================================================

For EVERY transaction:

1. Explain WHY the transaction was routed to REVIEW.

2. You MUST mention at least TWO specific risk factors from
   the "IDENTIFIED RISK FACTORS" section whenever at least two
   meaningful factors are available.

3. You MUST use the actual feature names or clear human-readable
   versions of those features.

   Examples:

   merchant_risk_tier=3
   → "high-risk merchant tier"

   new_device=1
   → "new device"

   new_location=1
   → "new location"

   location_mismatch=1
   → "location mismatch"

   new_merchant=1
   → "new merchant"

   time_of_day_anomaly=1
   → "unusual transaction time"

   velocity_anomaly=1
   → "unusual transaction velocity"

   amount_z_score
   → "unusual transaction amount"

4. Do NOT merely repeat the risk score and transaction amount.

5. Risk score and amount MAY be mentioned, but they are NOT
   sufficient as the explanation.

6. If a high-risk merchant tier is present, mention it.

7. If new device, new location, location mismatch, new merchant,
   velocity anomaly, time anomaly, or amount anomaly is present,
   mention the relevant factors.

8. Only describe a factor as a risk indicator when its supplied
   value actually indicates the factor is present.

9. Do NOT invent any transaction behavior.

10. Do NOT claim that the transaction is definitely fraudulent.

11. Explain why the combination of the observed factors justifies
    human review.

12. Use neutral, professional fraud-investigation language.

13. Keep the explanation to 2-3 sentences.

============================================================
EVIDENCE SUMMARY REQUIREMENTS
============================================================

For evidence_summary:

1. Provide 1-2 factual sentences.

2. Include the transaction amount or risk score when useful.

3. Include the most important identified risk factors.

4. Do NOT simply repeat:
   "Transaction X has amount Y and risk score Z."

5. The evidence summary must explain the observable reasons
   supporting the review decision.

============================================================
IMPORTANT
============================================================

The purpose of this layer is explainability.

BAD:

"This transaction was placed in review due to a risk score
of 37.16 and an amount of 45271.85."

GOOD:

"This transaction was routed for review because it has a
high-risk merchant tier and a fraud probability that produces
an elevated risk score. The high transaction value provides
additional context for manual verification."

If only one meaningful risk factor is available, mention that
factor and explain its relevance without inventing another one.

Do not mention SHAP, machine learning, model internals, or AI
in the final explanation.

Return exactly ONE explanation object for EACH transaction.

Preserve every transaction_id EXACTLY as provided.

Do not omit transactions.

============================================================
TRANSACTIONS
============================================================

{all_transactions}
"""

    return prompt


# ============================================================
# GEMINI BATCH CALL
# ============================================================

def generate_batch_explanations(transactions):
    """
    Send ONE Gemini request for an entire batch.

    Returns:
        list of dictionaries
    """

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY not found. "
            "Check your .env file."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = build_batch_prompt(
        transactions
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchExplanationOutput,
                    temperature=0.1,
                ),
            )

            parsed = response.parsed

            if parsed is None:

                raise ValueError(
                    "Gemini returned no structured response."
                )

            results = []

            for item in parsed.explanations:

                results.append({
                    "transaction_id":
                        item.transaction_id,

                    "explanation":
                        item.explanation,

                    "evidence_summary":
                        item.evidence_summary,

                    "source":
                        "gemini",

                    "error":
                        None,
                })

            # ------------------------------------------------
            # VALIDATE TRANSACTION COUNT
            # ------------------------------------------------

            expected_ids = {
                str(
                    txn.get(
                        "transaction_id",
                        ""
                    )
                )
                for txn in transactions
            }

            returned_ids = {
                str(
                    item["transaction_id"]
                )
                for item in results
            }

            missing_ids = (
                expected_ids
                - returned_ids
            )

            if missing_ids:

                raise ValueError(
                    "Gemini did not return explanations "
                    f"for transactions: {sorted(missing_ids)}"
                )

            return results

        except Exception as e:

            last_error = e

            error_text = str(e)

            is_rate_limit = (
                "429" in error_text
                or "RESOURCE_EXHAUSTED"
                in error_text
            )

            print()
            print(
                "Gemini ERROR:"
            )
            print(
                error_text
            )

            if attempt >= MAX_RETRIES:

                break

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if is_rate_limit:

                wait_time = (
                    15 * attempt
                )

                print(
                    f"Rate limit detected. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

            else:

                wait_time = (
                    5 * attempt
                )

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

    raise RuntimeError(
        f"Gemini failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# FALLBACK
# ============================================================

def fallback_explanation(transaction):
    """
    Emergency fallback.

    This is NOT counted as an LLM explanation.
    """

    transaction_id = transaction.get(
        "transaction_id",
        "UNKNOWN"
    )

    amount = transaction.get(
        "amount",
        "n/a"
    )

    risk_score = transaction.get(
        "risk_score",
        "n/a"
    )

    risk_factors = build_risk_factor_text(
        transaction
    )

    return {
        "transaction_id":
            transaction_id,

        "explanation": (
            "This transaction requires manual review "
            f"based on a risk score of {risk_score}. "
            "The identified risk indicators are: "
            f"{risk_factors.replace(chr(10), '; ')}"
        ),

        "evidence_summary": (
            f"Transaction amount: {amount}. "
            f"Risk score: {risk_score}. "
            "Identified risk indicators: "
            f"{risk_factors.replace(chr(10), '; ')}"
        ),

        "source":
            "fallback",

        "error":
            "Gemini unavailable",
    }