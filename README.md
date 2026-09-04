# 🛡️ AI Risk Manager

### End-to-End Fraud Detection, Risk Routing, Explainable AI & Human-in-the-Loop Investigation

AI Risk Manager is an end-to-end fraud risk management system designed to go beyond simple fraud classification.

The system predicts transaction fraud risk and converts that prediction into an operational decision:

**ALLOW → REVIEW → BLOCK**

For suspicious transactions, the system provides SHAP-based explanations, generates an LLM investigation summary, and supports human review with an auditable final decision.

---

## 🚀 Project Overview

Traditional fraud detection systems often focus only on:
> "Is this transaction fraudulent?"

A real-world risk management system needs to answer additional questions:

- How likely is the transaction to be fraudulent?
- How reliable is the predicted probability?
- What factors are driving the risk?
- Should the transaction be allowed, reviewed, or blocked?
- Why was the transaction flagged?
- Can an investigator understand the evidence?
- Can a human override the AI decision?
- Can the final decision be audited?

AI Risk Manager addresses these requirements through a complete decision pipeline combining:

- Machine Learning
- Probability Calibration
- Contextual Risk Scoring
- Cost-Sensitive Decision Making
- Risk Routing
- SHAP Explainability
- LLM Investigation Summaries
- Human-in-the-Loop Review
- Audit Logging

---

## 🏗️ System Architecture

```text
                         Transaction Data
                                │
                                ▼
                     Data Preprocessing
                                │
                                ▼
                     Feature Engineering
                                │
                                ▼
                       Fraud ML Model
                                │
                                ▼
                  Probability Calibration
                                │
                                ▼
                    Contextual Risk Score
                                │
                                ▼
                    Cost-Sensitive Router
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
              ALLOW           REVIEW          BLOCK
                                │
                                ▼
                         SHAP Investigation
                                │
                                ▼
                         LLM Explanation
                                │
                                ▼
                       Human Reviewer
                                │
                                ▼
                  Final Decision + Rationale
                                │
                                ▼
                         Audit Trail
```

---

## 📊 Results

Evaluated on a held-out temporal test set (18,968 transactions, 479 real fraud cases, 2.53% fraud rate) built from Kaggle's [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) synthetic mobile-money dataset, enriched with synthetic customer/device/location context and injected fraud archetypes (card-testing, device/location account takeover, synthetic identity) beyond PaySim's native fraud pattern.

### Routing outcome

| Decision | Transactions | % of total |
|---|---|---|
| ALLOW | 18,230 | 96.1% |
| REVIEW | 441 | 2.3% |
| BLOCK | 297 | 1.6% |

### Detection performance (flagged for REVIEW or BLOCK vs. actual fraud)

| Metric | Value |
|---|---|
| Precision | 0.629 |
| Recall | 0.969 |
| F1 Score | 0.763 |
| False Negatives (fraud missed) | 15 of 479 |
| False Positives | 274 |

For comparison, an untuned logistic regression baseline (default 0.5 threshold, no cost model) scored **0.19 precision / 0.90 recall / 0.31 F1** on an equivalent split — the full pipeline (XGBoost + calibration + cost-aware routing) more than doubled F1 while keeping recall high.

### Business impact: cost-aware routing vs. naive "allow everything"

| Approach | Total expected cost (test set) |
|---|---|
| Allow every transaction (naive baseline) | ₹2,596,047 |
| AI Risk Manager (cost-aware routing) | ₹93,215 |
| **Cost reduction** | **96.4%** |

Cost assumptions used for this simulation (documented in `src/models/04_cost_model.py`, intended to be replaced with real business figures in production): fraud loss ₹5,000, false-positive cost ₹250, review cost ₹50 per transaction.

> Every number above is computed directly from committed output files in `data/processed/` — nothing here is estimated by hand.

---

## 🖥️ Human Review Dashboard

<p align="center">
  <img src="assets/dashboard_review_center.png" alt="Human Review Center showing risk assessment, cost-based routing, and SHAP investigation" width="800">
</p>

<p align="center">
  <img src="assets/dashboard_human_decision.png" alt="AI Investigation explanation, evidence summary, and human decision panel" width="800">
</p>

The dashboard walks a reviewer through every stage of the pipeline for a flagged transaction: risk score and fraud probability, the router's cost-based recommendation and *why* it routed to REVIEW, SHAP-derived risk factors, an LLM-generated plain-English explanation grounded in those factors, and a final human decision (ALLOW / BLOCK / ESCALATE) with mandatory reviewer rationale — every decision is logged to the audit trail regardless of outcome.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| ML Model | XGBoost, scikit-learn |
| Calibration | scikit-learn (isotonic / Platt scaling) |
| Explainability | SHAP |
| LLM Investigation | Google Gemini API (`google-genai`) |
| Dashboard | Streamlit |
| Data | pandas, numpy, scipy |
| Config | python-dotenv |

Full pinned versions in [`requirements.txt`](requirements.txt).

---

## ⚙️ Setup & How to Run

### 1. Clone and install dependencies

```bash
git clone https://github.com/SatyaKiran25/AI-Risk-Manager.git
cd AI-Risk-Manager
pip install -r requirements.txt
```

### 2. Get a Gemini API key (free tier, no credit card required)

Create a key at [aistudio.google.com](https://aistudio.google.com), then set it as an environment variable:

```bash
export GEMINI_API_KEY="your-key-here"     # macOS/Linux
$env:GEMINI_API_KEY="your-key-here"       # Windows PowerShell
```

Or place it in a `.env` file in the project root:
```
GEMINI_API_KEY=your-key-here
```

### 3. Get the raw data

The raw PaySim dataset (`data/raw/PS_20174392719_1491204439457_log.csv`) is tracked via **Git LFS**. Pull it with:

```bash
git lfs pull
```

Or download it directly from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it at that path.

### 4. Run the pipeline, in order

```bash
python src/data/build_dataset.py        # PaySim -> enriched synthetic transaction dataset
python src/data/split_dataset.py        # Temporal train / validation / test split
python src/models/01_baseline.py        # Logistic regression baseline
python src/models/02_xgboost_model.py   # XGBoost risk model
python src/models/03_calibrate.py       # Probability calibration
python src/models/04_cost_model.py      # Cost-sensitive expected-cost calculation
python src/models/05_risk_router.py     # ALLOW / REVIEW / BLOCK routing
python src/models/06_shap_investigator.py   # SHAP risk factor extraction for REVIEW cases
python src/models/08_run_llm_investigator.py  # Batch LLM explanations via Gemini
```

### 5. Launch the human review dashboard

```bash
streamlit run src/app/09_human_reviewer.py
```

---

## ⚠️ Known Limitations & Honest Notes

- Cost figures (fraud loss, false-positive cost, review cost) are **simulation assumptions**, not real business data — see `src/models/04_cost_model.py` for exact values and how to replace them.
- PaySim's native customer IDs are ~99.85% one-time-use in the raw data; this project remaps transactions to synthetic recurring customer identities so behavioral/velocity features have real history to learn from (see `src/data/build_dataset.py`).
- The LLM explanation layer falls back to a template built directly from SHAP values if the Gemini API is unavailable (rate limit, no key, timeout) — the pipeline never breaks silently.