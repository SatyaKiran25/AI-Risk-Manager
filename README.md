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

# 🏗️ System Architecture

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