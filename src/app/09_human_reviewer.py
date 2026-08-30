"""
AI RISK MANAGER - HUMAN REVIEWER DASHBOARD

Step 9 of the pipeline.

Features:
    - Transaction details
    - Fraud probability
    - Risk score
    - Merchant risk tier
    - Active risk factors
    - SHAP investigation
    - Gemini investigation
    - Evidence summary
    - AI recommendation
    - Mandatory human decision
    - Mandatory reviewer rationale
    - Human-review audit trail
    - Model-reviewer agreement tracker
    - AI override tracking
    - Recalibration warning
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/processed/test_llm_investigated.csv"

SHAP_FALLBACK_PATHS = [
    "data/processed/test_cost_analysis.csv",
    "data/processed/test_routed.csv",
]

AUDIT_PATH = "data/processed/human_review_audit.csv"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Risk Manager",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .section-box {
        padding: 14px 16px;
        border-radius: 8px;
        border: 1px solid #30343f;
        margin-bottom: 10px;
    }

    .tier-box {
        padding: 14px 16px;
        border-radius: 8px;
        border: 1px solid #30343f;
        background-color: #17202a;
        margin-bottom: 12px;
    }

    .factor-box {
        padding: 10px 14px;
        border-radius: 7px;
        border: 1px solid #30343f;
        margin-bottom: 7px;
    }

    .decision-required {
        background-color: #3b3214;
        border-left: 4px solid #f5c542;
        padding: 10px 14px;
        border-radius: 5px;
        margin-top: 8px;
        margin-bottom: 10px;
    }

    .required-label {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:
        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def clean_value(value, default="N/A"):

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return default

    return text


def is_active(value):

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    text = str(value).strip().lower()

    return text in {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "active",
        "high",
    }


def tier_information(tier):

    tier_value = safe_float(tier, 0)

    if tier_value >= 3:

        return (
            "Tier 3",
            "High Risk",
            "Elevated merchant risk classification. "
            "Transactions through this merchant require additional scrutiny.",
        )

    if tier_value >= 2:

        return (
            "Tier 2",
            "Medium Risk",
            "Moderate merchant risk classification. "
            "Transactions require additional monitoring.",
        )

    return (
        "Tier 1",
        "Low Risk",
        "Lower merchant risk classification based on the available merchant-risk history.",
    )


def find_first_value(row, candidates):

    for column in candidates:

        if column not in row.index:
            continue

        value = clean_value(
            row.get(column),
            "",
        )

        if value:
            return value

    return ""


def merchant_tier_basis(row):

    candidates = [
        ("merchant_risk_score", "Merchant Risk Score"),
        ("merchant_fraud_rate", "Merchant Fraud Rate"),
        ("merchant_chargeback_rate", "Merchant Chargeback Rate"),
        ("merchant_dispute_rate", "Merchant Dispute Rate"),
        ("merchant_risk", "Merchant Risk Score"),
        ("historical_merchant_risk", "Historical Merchant Risk"),
        ("merchant_transaction_count", "Merchant Transaction Count"),
        ("merchant_fraud_count", "Merchant Fraud Count"),
    ]

    found = []

    for column, label in candidates:

        if column not in row.index:
            continue

        value = clean_value(
            row.get(column),
            "",
        )

        if value:
            found.append(
                (label, value)
            )

    return found


def risk_factor_description(feature):

    descriptions = {

        "new_device":
            "Transaction originated from a device not previously observed for the customer.",

        "new_location":
            "Transaction originated from a location not previously observed for the customer.",

        "location_mismatch":
            "Transaction location differs from the customer's expected location.",

        "new_merchant":
            "The customer is interacting with a merchant not previously observed.",

        "velocity_anomaly":
            "Transaction frequency is unusually high compared with normal activity.",

        "time_of_day_anomaly":
            "Transaction occurred at an unusual time for the observed customer behavior.",

        "merchant_risk_tier":
            "Merchant belongs to an elevated historical risk classification.",

        "amount_z_score":
            "Transaction amount is unusually large compared with the customer's normal amount.",
    }

    return descriptions.get(
        feature,
        "Risk indicator detected by the fraud risk system.",
    )


def get_ai_recommendation(row):

    candidates = [
        "recommended_action",
        "cost_decision",
        "cost_recommendation",
        "recommended_decision",
    ]

    for column in candidates:

        if column not in row.index:
            continue

        value = clean_value(
            row.get(column),
            "",
        )

        if value:
            return value.upper()

    return "REVIEW"


# ============================================================
# LOAD MAIN DATA
# ============================================================

@st.cache_data
def load_main_data():

    if not os.path.exists(INPUT_PATH):
        return None

    data = pd.read_csv(INPUT_PATH)

    if "router_decision" not in data.columns:

        raise ValueError(
            "Column 'router_decision' is missing from "
            "test_llm_investigated.csv"
        )

    data = data[
        data["router_decision"]
        .astype(str)
        .str.upper()
        .eq("REVIEW")
    ].copy()

    if "transaction_id" not in data.columns:

        data["transaction_id"] = [
            f"REVIEW_{i + 1}"
            for i in range(len(data))
        ]

    data["transaction_id"] = (
        data["transaction_id"]
        .astype(str)
        .str.strip()
    )

    data.reset_index(
        drop=True,
        inplace=True,
    )

    return data


df = load_main_data()


# ============================================================
# DATASET CHECK
# ============================================================

if df is None:

    st.error(
        f"Input file not found:\n\n{INPUT_PATH}"
    )

    st.stop()


if len(df) == 0:

    st.success(
        "🎉 No transactions require human review."
    )

    st.stop()


# ============================================================
# LOAD SHAP FALLBACK DATA
# ============================================================

@st.cache_data
def load_shap_fallback():

    frames = []

    for path in SHAP_FALLBACK_PATHS:

        if not os.path.exists(path):
            continue

        try:

            candidate = pd.read_csv(path)

            if "transaction_id" in candidate.columns:

                candidate["transaction_id"] = (
                    candidate["transaction_id"]
                    .astype(str)
                    .str.strip()
                )

                frames.append(candidate)

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    result = frames[0].copy()

    for extra in frames[1:]:

        extra = extra.drop_duplicates(
            "transaction_id"
        )

        result = result.merge(
            extra,
            on="transaction_id",
            how="left",
            suffixes=("", "__fallback"),
        )

        fallback_cols = [
            c
            for c in result.columns
            if c.endswith("__fallback")
        ]

        for fallback_col in fallback_cols:

            base_col = fallback_col[:-10]

            if base_col in result.columns:

                result[base_col] = (
                    result[base_col]
                    .where(
                        result[base_col].notna()
                        &
                        result[base_col]
                        .astype(str)
                        .str.strip()
                        .ne(""),
                        result[fallback_col],
                    )
                )

                result.drop(
                    columns=[fallback_col],
                    inplace=True,
                )

            else:

                result.rename(
                    columns={
                        fallback_col: base_col
                    },
                    inplace=True,
                )

    return result.drop_duplicates(
        "transaction_id"
    )


shap_df = load_shap_fallback()


if (
    not shap_df.empty
    and "transaction_id" in shap_df.columns
):

    shap_df = shap_df.drop_duplicates(
        "transaction_id"
    )

    shap_columns = [
        c
        for c in shap_df.columns
        if (
            "shap" in c.lower()
            or c.lower()
            in {
                "top_risk_feature",
                "top_feature",
            }
        )
    ]

    if shap_columns:

        merge_columns = [
            "transaction_id"
        ] + [
            c
            for c in shap_columns
            if (
                c != "transaction_id"
                and c not in df.columns
            )
        ]

        if len(merge_columns) > 1:

            df = df.merge(
                shap_df[merge_columns],
                on="transaction_id",
                how="left",
            )


# ============================================================
# AUDIT CONFIGURATION
# ============================================================

AUDIT_COLUMNS = [

    "timestamp",
    "transaction_id",
    "amount",
    "risk_score",
    "calibrated_risk_probability",
    "contextual_risk",
    "merchant_risk_tier",
    "router_decision",
    "ai_recommendation",
    "recommended_action",
    "expected_cost",
    "routing_reason",
    "top_risk_feature",
    "top_shap_contribution",
    "shap_explanation",
    "llm_explanation",
    "evidence_summary",
    "llm_source",
    "human_decision",
    "reviewer_rationale",
    "reviewer_agreement",
    "ai_overridden",
]


# ============================================================
# LOAD AUDIT DATA
# ============================================================

@st.cache_data(ttl=0)
def load_audit():

    if not os.path.exists(AUDIT_PATH):

        return pd.DataFrame(
            columns=AUDIT_COLUMNS
        )

    try:

        audit = pd.read_csv(
            AUDIT_PATH
        )

        if audit.empty:

            return pd.DataFrame(
                columns=AUDIT_COLUMNS
            )

        for column in AUDIT_COLUMNS:

            if column not in audit.columns:
                audit[column] = None

        audit = audit[AUDIT_COLUMNS].copy()

        audit["transaction_id"] = (
            audit["transaction_id"]
            .astype(str)
            .str.strip()
        )

        audit = audit[
            audit["transaction_id"].notna()
            &
            audit["transaction_id"].ne("")
            &
            audit["transaction_id"].ne("nan")
        ].copy()

        return audit

    except Exception as e:

        st.warning(
            f"Audit file could not be read: {e}"
        )

        return pd.DataFrame(
            columns=AUDIT_COLUMNS
        )


# ============================================================
# AGREEMENT CALCULATION
# ============================================================

def calculate_agreement(audit):

    if audit is None or audit.empty:
        return None

    if "human_decision" not in audit.columns:
        return None

    working = audit.copy()

    # --------------------------------------------------------
    # Prefer the explicitly saved AI recommendation.
    # --------------------------------------------------------

    if "ai_recommendation" in working.columns:

        working["ai_action"] = (
            working["ai_recommendation"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    elif "recommended_action" in working.columns:

        working["ai_action"] = (
            working["recommended_action"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    else:

        return None

    working["human_action"] = (
        working["human_decision"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid_ai_actions = [
        "ALLOW",
        "BLOCK",
        "ESCALATE",
        "REVIEW",
    ]

    valid_human_actions = [
        "ALLOW",
        "BLOCK",
        "ESCALATE",
    ]

    valid = working[
        working["ai_action"].isin(
            valid_ai_actions
        )
        &
        working["human_action"].isin(
            valid_human_actions
        )
    ].copy()

    if valid.empty:
        return None

    valid["agreement"] = (
        valid["ai_action"]
        ==
        valid["human_action"]
    )

    total = len(valid)

    agreed = int(
        valid["agreement"].sum()
    )

    overridden = total - agreed

    percentage = (
        agreed / total
    ) * 100

    return {
        "total": total,
        "agreed": agreed,
        "overridden": overridden,
        "percentage": percentage,
    }


# ============================================================
# IMPORTANT:
# LOAD LATEST AUDIT DATA BEFORE DISPLAYING DASHBOARD
# ============================================================

audit_df = load_audit()


# ============================================================
# REVIEWED TRANSACTIONS
# ============================================================

reviewed_ids = set()

if (
    not audit_df.empty
    and "transaction_id" in audit_df.columns
):

    reviewed_ids = set(
        audit_df[
            "transaction_id"
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )


# ============================================================
# PENDING TRANSACTIONS
# ============================================================

pending_df = df[
    ~df["transaction_id"]
    .astype(str)
    .str.strip()
    .isin(reviewed_ids)
].copy()

pending_df.reset_index(
    drop=True,
    inplace=True,
)


# ============================================================
# AGREEMENT
# ============================================================

agreement = calculate_agreement(
    audit_df
)


# ============================================================
# SESSION STATE
# ============================================================

if "current_index" not in st.session_state:

    st.session_state.current_index = 0


if "form_version" not in st.session_state:

    st.session_state.form_version = 0


if len(pending_df) > 0:

    if (
        st.session_state.current_index
        >= len(pending_df)
    ):

        st.session_state.current_index = (
            len(pending_df) - 1
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ AI Risk Manager"
)

st.subheader(
    "Human Review Center"
)

st.caption(
    "Human-in-the-loop fraud investigation "
    "and decision system"
)


# ============================================================
# MODEL–REVIEWER AGREEMENT
# ============================================================

st.subheader(
    "📈 Model–Reviewer Agreement"
)


if agreement is None:

    st.info(
        "No human reviews completed yet."
    )

else:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Agreement",
            f"{agreement['percentage']:.1f}%"
        )

    with col2:

        st.metric(
            "Agreed",
            agreement["agreed"]
        )

    with col3:

        st.metric(
            "AI Overridden",
            agreement["overridden"]
        )

    if agreement["percentage"] < 70:

        st.warning(
            "⚠️ Reviewer agreement is below 70%. "
            "Consider model recalibration."
        )

    elif agreement["percentage"] < 85:

        st.info(
            "ℹ️ Moderate reviewer agreement. "
            "Continue monitoring model decisions."
        )

    else:

        st.success(
            "✅ High reviewer agreement. "
            "Model decisions are strongly aligned with reviewers."
        )


# ============================================================
# ALL REVIEWED
# ============================================================

if len(pending_df) == 0:

    st.success(
        "🎉 All REVIEW transactions have been processed."
    )

    st.write(
        f"Total REVIEW transactions: {len(df)}"
    )

    st.write(
        f"Completed human reviews: {len(reviewed_ids)}"
    )

    st.stop()


# ============================================================
# CURRENT TRANSACTION
# ============================================================

row = pending_df.iloc[
    st.session_state.current_index
]

transaction_id = str(
    row["transaction_id"]
).strip()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Review Queue"
    )

    st.metric(
        "Total REVIEW",
        len(df)
    )

    st.metric(
        "Reviewed",
        len(reviewed_ids)
    )

    st.metric(
        "Remaining",
        len(pending_df)
    )

    st.divider()

    st.write(
        "**Pipeline Status**"
    )

    st.write(
        "✅ Fraud Prediction"
    )

    st.write(
        "✅ Risk Calibration"
    )

    st.write(
        "✅ Cost-Based Routing"
    )

    st.write(
        "✅ SHAP Investigation"
    )

    st.write(
        "✅ Gemini Investigation"
    )

    st.write(
        "🔵 Human Review"
    )


# ============================================================
# PROGRESS
# ============================================================

progress = (
    st.session_state.current_index + 1
) / len(pending_df)

st.progress(
    progress
)

st.caption(
    f"Reviewing transaction "
    f"{st.session_state.current_index + 1} "
    f"of {len(pending_df)} pending reviews"
)


# ============================================================
# FORM KEYS
# ============================================================

form_version = (
    st.session_state.form_version
)

decision_key = (
    f"decision_{transaction_id}_{form_version}"
)

rationale_key = (
    f"rationale_{transaction_id}_{form_version}"
)


# ============================================================
# TRANSACTION HEADER
# ============================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Transaction ID
# ------------------------------------------------------------

with col1:

    st.metric(
        "Transaction ID",
        transaction_id
    )


# ------------------------------------------------------------
# Amount
# ------------------------------------------------------------

with col2:

    amount = row.get(
        "amount",
        row.get(
            "transaction_amount",
            0
        )
    )

    amount_value = safe_float(
        amount,
        0
    )

    st.metric(
        "Transaction Amount",
        f"₹{amount_value:,.2f}"
    )


# ------------------------------------------------------------
# Fraud Probability
# ------------------------------------------------------------

with col3:

    probability = row.get(
        "calibrated_risk_probability",
        row.get(
            "fraud_probability",
            row.get(
                "calibrated_probability",
                0
            )
        )
    )

    probability_value = safe_float(
        probability,
        0
    )

    if probability_value <= 1:

        probability_percent = (
            probability_value * 100
        )

    else:

        probability_percent = (
            probability_value
        )

    st.metric(
        "Fraud Probability",
        f"{probability_percent:.2f}%"
    )

    st.caption(
        "Calibrated Fraud Model"
    )


# ------------------------------------------------------------
# Risk Score
# ------------------------------------------------------------

with col4:

    risk_score = safe_float(
        row.get(
            "risk_score",
            0
        ),
        0
    )

    st.metric(
        "Risk Score",
        f"{risk_score:.2f} / 100"
    )

    st.caption(
        "Calibrated Probability + Contextual Risk"
    )


# ============================================================
# SYSTEM DECISION
# ============================================================

st.divider()

decision_col1, decision_col2 = st.columns(2)


router_decision = clean_value(
    row.get(
        "router_decision",
        "REVIEW"
    )
).upper()


recommended_action = get_ai_recommendation(
    row
)


with decision_col1:

    st.subheader(
        "System Routing"
    )

    st.info(
        f"AI Router Decision: **{router_decision}**"
    )


with decision_col2:

    st.subheader(
        "AI Recommendation"
    )

    if recommended_action == "BLOCK":

        st.error(
            f"Recommended Action: **{recommended_action}**"
        )

    elif recommended_action == "ALLOW":

        st.success(
            f"Recommended Action: **{recommended_action}**"
        )

    else:

        st.warning(
            f"Recommended Action: **{recommended_action}**"
        )


# ============================================================
# ROUTING REASON
# ============================================================

routing_reason = clean_value(
    row.get(
        "routing_reason",
        ""
    ),
    ""
)

if routing_reason:

    st.caption(
        "Routing Reason"
    )

    st.write(
        routing_reason
    )


# ============================================================
# AI INVESTIGATION
# ============================================================

st.divider()

st.header(
    "🔍 AI Investigation"
)


# ============================================================
# MERCHANT RISK TIER
# ============================================================

tier_value = row.get(
    "merchant_risk_tier",
    1
)

tier_name, tier_level, tier_description = (
    tier_information(tier_value)
)


st.subheader(
    "Merchant Risk Tier"
)


st.markdown(
    f"""
    <div class="tier-box">
        <h3>{tier_name} — {tier_level}</h3>
        <p>{tier_description}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


tier_basis = merchant_tier_basis(
    row
)


if tier_basis:

    st.caption(
        "Tier Assignment Factors"
    )

    basis_cols = st.columns(
        min(3, len(tier_basis))
    )

    for i, (label, value) in enumerate(
        tier_basis
    ):

        with basis_cols[
            i % len(basis_cols)
        ]:

            st.metric(
                label,
                value
            )

else:

    st.caption(
        "Tier basis: merchant-risk classification "
        "available in the dataset. No underlying "
        "merchant-risk metrics were provided for "
        "this transaction."
    )


# ============================================================
# ACTIVE RISK FACTORS
# ============================================================

st.subheader(
    "⚠️ Risk Factors Detected"
)


factor_candidates = [

    (
        "merchant_risk_tier",
        "Merchant Risk Tier",
    ),

    (
        "new_device",
        "New Device",
    ),

    (
        "new_location",
        "New Location",
    ),

    (
        "location_mismatch",
        "Location Mismatch",
    ),

    (
        "new_merchant",
        "New Merchant",
    ),

    (
        "velocity_anomaly",
        "Velocity Anomaly",
    ),

    (
        "time_of_day_anomaly",
        "Time-of-Day Anomaly",
    ),

    (
        "amount_z_score",
        "Unusual Transaction Amount",
    ),
]


active_factors = []


for feature, label in factor_candidates:

    if feature not in row.index:
        continue

    value = row.get(
        feature
    )

    if feature == "merchant_risk_tier":

        tier_number = safe_float(
            value,
            0
        )

        if tier_number >= 2:

            active_factors.append(
                (
                    label,
                    f"{tier_name} — {tier_level}",
                    risk_factor_description(
                        feature
                    ),
                )
            )

    elif feature == "amount_z_score":

        z_score = safe_float(
            value,
            0
        )

        if abs(z_score) >= 2:

            active_factors.append(
                (
                    label,
                    f"Z-score: {z_score:+.2f}",
                    risk_factor_description(
                        feature
                    ),
                )
            )

    elif is_active(value):

        active_factors.append(
            (
                label,
                "Detected",
                risk_factor_description(
                    feature
                ),
            )
        )


if active_factors:

    for label, status, description in active_factors:

        st.markdown(
            f"""
            <div class="factor-box">
                <strong>🔴 {label}</strong>
                <br>
                <span>{status}</span>
                <br>
                <small>{description}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    st.success(
        "No specific active risk factors were identified."
    )


# ============================================================
# SHAP INVESTIGATION
# ============================================================

st.divider()

st.subheader(
    "SHAP Risk Factors"
)


def format_feature_name(name):

    if name is None:
        return "Unknown"

    text_name = str(
        name
    ).strip()

    if not text_name:
        return "Unknown"

    return (
        text_name
        .replace("_", " ")
        .title()
    )


def parse_shap_rows(row):

    results = []

    # --------------------------------------------------------
    # 1. Structured SHAP feature/value columns
    # --------------------------------------------------------

    feature_text = clean_value(
        row.get(
            "shap_features",
            ""
        ),
        "",
    )

    value_text = clean_value(
        row.get(
            "shap_values",
            ""
        ),
        "",
    )


    if feature_text and value_text:

        features = [
            item.strip()
            for item in str(
                feature_text
            ).split("|")
            if item.strip()
        ]

        values = []

        for item in str(
            value_text
        ).split("|"):

            item = item.strip()

            if not item:
                continue

            try:

                values.append(
                    float(item)
                )

            except (
                ValueError,
                TypeError,
            ):

                values.append(
                    None
                )


        for feature, value in zip(
            features,
            values,
        ):

            if value is not None:

                results.append(
                    (
                        feature,
                        value
                    )
                )


    # --------------------------------------------------------
    # 2. Individual SHAP columns
    # --------------------------------------------------------

    if not results:

        excluded = {
            "shap_explanation",
            "shap_features",
            "shap_feature_names",
            "shap_feature",
            "shap_values",
            "shap_value",
            "shap_contributions",
            "shap_risk_factors",
        }


        for column in row.index:

            column_name = str(
                column
            )

            lower = (
                column_name.lower()
            )


            if (
                not lower.startswith(
                    "shap_"
                )
                or lower in excluded
            ):

                continue


            value = row.get(
                column
            )


            try:

                if pd.isna(value):
                    continue

                numeric = float(
                    value
                )

            except (
                ValueError,
                TypeError,
            ):

                continue


            results.append(
                (
                    column_name[5:],
                    numeric
                )
            )


    # --------------------------------------------------------
    # 3. Top-feature fallback
    # --------------------------------------------------------

    if not results:

        top_feature = find_first_value(
            row,
            [
                "top_risk_feature",
                "top_feature",
            ],
        )

        contribution = find_first_value(
            row,
            [
                "top_shap_contribution",
                "top_feature_shap",
            ],
        )


        if top_feature and contribution:

            try:

                results.append(
                    (
                        top_feature,
                        float(contribution)
                    )
                )

            except (
                ValueError,
                TypeError,
            ):

                pass


    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    best = {}

    for feature, value in results:

        key = str(
            feature
        ).strip()

        if not key:
            continue

        if (
            key not in best
            or abs(value)
            > abs(best[key])
        ):

            best[key] = value


    return sorted(
        best.items(),
        key=lambda item:
            abs(item[1]),
        reverse=True,
    )


shap_rows = parse_shap_rows(
    row
)


if shap_rows:

    shap_display = pd.DataFrame(
        [
            {
                "Risk Feature":
                    format_feature_name(
                        feature
                    ),

                "SHAP Impact":
                    value,

                "Impact":
                    (
                        "Increases Risk"
                        if value > 0
                        else
                        "Decreases Risk"
                        if value < 0
                        else
                        "Neutral"
                    ),
            }

            for feature, value
            in shap_rows
        ]
    )


    shap_display[
        "SHAP Impact"
    ] = shap_display[
        "SHAP Impact"
    ].map(
        lambda x:
            f"{x:+.4f}"
    )


    st.dataframe(
        shap_display,
        hide_index=True,
        use_container_width=True,
    )


else:

    st.info(
        "Structured SHAP values are not available "
        "for this transaction."
    )


# ============================================================
# TOP SHAP FEATURE
# ============================================================

feature_col1, feature_col2 = (
    st.columns(2)
)


with feature_col1:

    st.subheader(
        "Top Risk Feature"
    )

    top_feature = find_first_value(
        row,
        [
            "top_risk_feature",
            "top_feature",
        ],
    )

    if not top_feature and shap_rows:

        top_feature = shap_rows[0][0]

    st.metric(
        "Feature",
        (
            format_feature_name(
                top_feature
            )
            if top_feature
            else "N/A"
        )
    )


with feature_col2:

    if shap_rows:

        top_contribution = (
            shap_rows[0][1]
        )

        st.metric(
            "SHAP Contribution",
            f"{top_contribution:+.4f}"
        )

    else:

        contribution = find_first_value(
            row,
            [
                "top_shap_contribution",
                "top_feature_shap",
            ],
        )

        if contribution:

            st.metric(
                "SHAP Contribution",
                contribution
            )


# ============================================================
# GEMINI EXPLANATION
# ============================================================

st.divider()

st.subheader(
    "🤖 AI Investigation Explanation"
)


llm_explanation = row.get(
    "llm_explanation",
    None
)


if (
    llm_explanation is not None
    and pd.notna(llm_explanation)
    and str(
        llm_explanation
    ).strip()
    and str(
        llm_explanation
    ).lower() != "nan"
):

    st.write(
        str(llm_explanation)
    )

else:

    st.warning(
        "No LLM explanation available."
    )


# ============================================================
# EVIDENCE SUMMARY
# ============================================================

st.subheader(
    "📄 Evidence Summary"
)


evidence = row.get(
    "evidence_summary",
    None
)


if (
    evidence is not None
    and pd.notna(evidence)
    and str(evidence).strip()
    and str(evidence).lower() != "nan"
):

    st.info(
        str(evidence)
    )

else:

    st.info(
        "No evidence summary available."
    )


# ============================================================
# HUMAN REVIEW
# ============================================================

st.divider()

st.header(
    "👤 Human Review"
)


st.markdown(
    """
    <div class="section-box">
        Review the AI evidence and make the final decision.
        An explicit decision and rationale are required
        for auditability.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DECISION
# ============================================================

st.markdown(
    """
    <div class="required-label">
        Final Decision
        <span style="color:#ff4b4b">*</span>
    </div>
    """,
    unsafe_allow_html=True,
)


decision = st.radio(
    "Select one:",
    [
        "ALLOW",
        "BLOCK",
        "ESCALATE",
    ],
    index=None,
    horizontal=True,
    key=decision_key,
)


decision_valid = (
    decision is not None
)


if not decision_valid:

    st.markdown(
        """
        <div class="decision-required">
            ⚠️ Please select ALLOW, BLOCK, or ESCALATE.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RATIONALE
# ============================================================

st.markdown(
    """
    <div class="required-label">
        Reviewer Rationale
        <span style="color:#ff4b4b">*</span>
    </div>
    """,
    unsafe_allow_html=True,
)


rationale = st.text_area(
    "Explain why you made this decision:",
    placeholder=(
        "Example: Merchant is high risk and the "
        "transaction originated from a new device. "
        "After reviewing the evidence, I recommend BLOCK."
    ),
    height=120,
    key=rationale_key,
)


rationale_valid = (
    rationale is not None
    and bool(
        rationale.strip()
    )
)


if not rationale_valid:

    st.caption(
        "⚠️ Reviewer rationale is required "
        "for the audit trail."
    )


can_submit = (
    decision_valid
    and rationale_valid
)


# ============================================================
# SAVE DECISION
# ============================================================

def save_decision(
    transaction_id,
    decision,
    rationale,
    row,
    ai_recommendation,
):

    os.makedirs(
        os.path.dirname(
            AUDIT_PATH
        ),
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Duplicate protection
    # --------------------------------------------------------

    if os.path.exists(
        AUDIT_PATH
    ):

        try:

            existing = pd.read_csv(
                AUDIT_PATH
            )

            if (
                "transaction_id"
                in existing.columns
            ):

                existing_ids = set(
                    existing[
                        "transaction_id"
                    ]
                    .astype(str)
                    .str.strip()
                    .tolist()
                )

                if (
                    str(
                        transaction_id
                    ).strip()
                    in existing_ids
                ):

                    return False

        except Exception:

            pass


    # --------------------------------------------------------
    # Safe value
    # --------------------------------------------------------

    def safe_value(
        column,
        default=None,
    ):

        value = row.get(
            column,
            default,
        )

        try:

            if pd.isna(value):
                return default

        except Exception:

            pass

        return value


    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    ai_action = str(
        ai_recommendation
    ).strip().upper()


    human_action = str(
        decision
    ).strip().upper()


    reviewer_agreement = (
        ai_action
        ==
        human_action
    )


    # --------------------------------------------------------
    # Audit record
    # --------------------------------------------------------

    record = {

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "transaction_id":
            str(transaction_id),

        "amount":
            safe_value(
                "amount",
                safe_value(
                    "transaction_amount"
                ),
            ),

        "risk_score":
            safe_value(
                "risk_score"
            ),

        "calibrated_risk_probability":
            safe_value(
                "calibrated_risk_probability",
                safe_value(
                    "fraud_probability"
                ),
            ),

        "contextual_risk":
            safe_value(
                "contextual_risk"
            ),

        "merchant_risk_tier":
            safe_value(
                "merchant_risk_tier"
            ),

        "router_decision":
            safe_value(
                "router_decision"
            ),

        "ai_recommendation":
            ai_action,

        "recommended_action":
            safe_value(
                "recommended_action"
            ),

        "expected_cost":
            safe_value(
                "expected_cost"
            ),

        "routing_reason":
            safe_value(
                "routing_reason"
            ),

        "top_risk_feature":
            safe_value(
                "top_risk_feature"
            ),

        "top_shap_contribution":
            safe_value(
                "top_shap_contribution"
            ),

        "shap_explanation":
            safe_value(
                "shap_explanation"
            ),

        "llm_explanation":
            safe_value(
                "llm_explanation"
            ),

        "evidence_summary":
            safe_value(
                "evidence_summary"
            ),

        "llm_source":
            safe_value(
                "llm_source"
            ),

        "human_decision":
            human_action,

        "reviewer_rationale":
            str(
                rationale
            ).strip(),

        "reviewer_agreement":
            reviewer_agreement,

        "ai_overridden":
            not reviewer_agreement,
    }


    # --------------------------------------------------------
    # Write audit record
    # --------------------------------------------------------

    record_df = pd.DataFrame(
        [record]
    )


    if os.path.exists(
        AUDIT_PATH
    ):

        record_df.to_csv(
            AUDIT_PATH,
            mode="a",
            header=False,
            index=False,
        )

    else:

        record_df.to_csv(
            AUDIT_PATH,
            index=False,
        )


    return True


# ============================================================
# SUBMIT / SKIP
# ============================================================

submit_col, skip_col = (
    st.columns([4, 1])
)


# ============================================================
# SUBMIT
# ============================================================

with submit_col:

    if st.button(
        "✅ Submit Decision",
        type="primary",
        use_container_width=True,
        disabled=not can_submit,
    ):

        saved = save_decision(
            transaction_id=transaction_id,
            decision=decision,
            rationale=rationale,
            row=row,
            ai_recommendation=recommended_action,
        )


        if not saved:

            st.warning(
                "This transaction has already been reviewed."
            )

        else:

            if (
                str(decision).upper()
                ==
                str(
                    recommended_action
                ).upper()
            ):

                st.success(
                    f"Decision saved: {decision} "
                    f"• AI recommendation agreed"
                )

            else:

                st.info(
                    f"Decision saved: {decision} "
                    f"• AI recommendation overridden"
                )


            # ------------------------------------------------
            # IMPORTANT:
            # Clear Streamlit cache so the next rerun reads
            # the newly updated audit CSV.
            # ------------------------------------------------

            load_audit.clear()


            st.session_state.current_index += 1

            st.session_state.form_version += 1

            st.rerun()


# ============================================================
# SKIP
# ============================================================

with skip_col:

    if st.button(
        "Skip for Now →",
        use_container_width=True,
        help=(
            "Move to another transaction without "
            "recording a human decision."
        ),
    ):

        if (
            st.session_state.current_index
            <
            len(pending_df) - 1
        ):

            st.session_state.current_index += 1

        else:

            st.session_state.current_index = 0

        st.session_state.form_version += 1

        st.rerun()


# ============================================================
# SUBMISSION STATUS
# ============================================================

if not can_submit:

    st.caption(
        "Submit is disabled until you select a decision "
        "and provide a reviewer rationale."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Risk Manager • Human-in-the-loop fraud decision "
    "system • Explainable • Audit-ready"
)