# Instamart Customer Intelligence & Support Copilot

> Synthetic educational capstone. This repository uses fictional data and offline model simulation only. It is not real Instamart data or policy.

## Overview

A reproducible four-part Python pipeline: (A) clean and persist customer/order data, (B) build RFM segments and a lapse classifier, (C) redact and classify support tickets with retry/budget/routing controls, and (D) retrieve policy passages and run a tool-using refund agent.

## Reproduce

Run from the repository root with Python 3.11+ and SQLite 3.25+.

1. `python -m venv .venv`
2. Windows: `.venv\Scripts\activate` — macOS/Linux: `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_all.py`

The generator and mock model are copied verbatim from the supplied brief. All model calls are offline through `mock_llm.py`; no API key or network service is used.

## Part A

Cleaning rules:
- City: strip whitespace and title-case; the result contains the six canonical cities.
- Gender: normalize male/M and female/F variants; blank and unrecognized values become `unknown`.
- Age: parse numeric values, mark values outside 10–100 missing, then fill missing values using the city median.
- Orders: parse timestamps; remove exact duplicate rows; retain the latest timestamp per order ID; coerce amounts to numeric and reconstruct missing amounts from catalogue price × quantity × (1 − discount/100); negative delivery times become missing; completed-order missing delivery times are imputed from the completed-order product-category median; cancelled-order delivery stays missing.
- Clean CSV files and the SQLite database are written under `data/`.

`src/part_a.py` creates seven SQL analyses in `sql/queries.sql` and asserts the core reference counts, sums and missing-value rules. A successful run writes `outputs/part_a_validation.txt` containing `ALL CHECKPOINTS PASSED`. The executed clean-data checks confirmed 1,200 customers, 12,309 orders (11,110 completed and 1,199 cancelled), and completed-order revenue of INR 2,998,668.80.

The SQL analyses cover WHERE/HAVING, a three-table city/category revenue join, scalar subquery, LEFT JOIN, CTE/average spend, ranking window functions and cumulative daily revenue.

## Part B

The RFM snapshot is 2025-10-01. K-means metrics for K=2..8 are written to `outputs/kmeans_metrics.csv`; the implementation selects K=4, profiles clusters, saves `outputs/customer_segments.csv`, and reports two-component PCA variance in `outputs/pca_variance.csv`.

The lapse cohort uses only pre-2025-09-01 order history for features. September completed-order activity defines the target label, not a feature. Encoders and scalers are inside a pipeline fitted only on training rows. Segment labels are not model features. Logistic regression and random forest are tuned by five-fold F1 grid search; metrics are saved in `outputs/lapse_model_metrics.csv`. The selected model is persisted and reloaded from `outputs/lapse_model.joblib`. In the executed run, the tuned logistic model achieved test precision/recall/F1 of 0.644/0.652/0.648; the tuned random forest achieved 0.659/0.652/0.655. The random forest was selected.

**Leakage limitation:** the Premium flag is based on whole-window RFM and therefore includes the September label window; the retention list is illustrative and is not a leakage-free evaluation.

## Part C

Tickets are processed in file order. Empty and over-400-token tickets are skipped before model invocation. Redaction order is email, phone, then order ID; assertions verify that no matching personal identifier remains in model input. Calls retry transient errors with exponential backoff and jitter. Responses are accepted only as valid allowed-category JSON or a single unambiguous allowed category extracted from lowercase alphabetic tokens; all other outputs are routed to human review.

The zero-shot budget demonstration is saved as `outputs/triage_budget_demo.csv`; the final few-shot routing output is `outputs/triage_results.csv`. The run produced 55 email, 91 phone and 132 order-ID substitutions (278 total); the few-shot run yielded 135 JSON classifications, 12 keyword classifications and 9 invalid outputs, with 19 transient errors recovered. The 6,000-token zero-shot budget demo exhausted the budget on 29 tickets. Fraud goes to the security team; premium customers go to the premium desk; other valid tickets go to the support queue; guardrail/invalid/budget cases go to human review.

## Part D

Five policy documents are chunked into 40-word passages with 10-word overlap, vectorized with `TfidfVectorizer`, and ranked using cosine similarity. TF measures within-document term frequency; IDF downweights terms common across documents. TF-IDF is lexical, so semantically equivalent wording with different terms can retrieve poorly.

The policy assistant answers from the top three retrieved chunks and returns source filenames. Evaluation outputs are saved for overlapping and zero-overlap chunking.

The refund agent exposes `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. It logs ledger actions, enforces iteration/token/time limits, checkpoints after tool observations, and includes a crash/resume demonstration. The executed scenario outcomes were S1 REFUND INR 128.25 (5 calls), S2 REFUND INR 135.00 (5), S3 ESCALATE INR 1,999.00 (5), S4 DENIED (4), S5 DENIED (3), S6 NOT_FOUND (2), and S7 FALLBACK_MAX_ITERATIONS (6). The stop-safe demos returned FALLBACK_TOKEN_LIMIT and FALLBACK_TIMEOUT. The crash/resume demonstration returned REFUND INR 128.25. See `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `outputs/agent_demos.json`, and `outputs/crash_resume_result.json`.

## Limitations and Next Steps

- The supplied model is a deterministic offline mock, not a production LLM; classification and agent behavior are intentionally constrained by its implementation.
- TF-IDF does not perform semantic retrieval; a production system should evaluate embeddings and a reranker while preserving source citations.
- The synthetic dataset is small and generated from fixed archetypes; model metrics should not be interpreted as real customer-risk performance.
- The Premium flag is a post-hoc whole-window label that includes the September target window, so the retention list is illustrative and not a leakage-free evaluation.
- Production deployment would require authorization, audit controls, real policy governance, payment-provider integration, idempotency keys, human approval workflows, monitoring and privacy/security review.

### Implementation coverage note

This is an executable project baseline, not a claim that every narrative/reporting subtask in the 100-mark rubric is complete. In particular, the current README does not yet include the full raw-column missingness audit, every SQL query's result table, the complete Part A descriptive-statistics/correlation discussion, the logistic-regression oversampling comparison and coefficient interpretation, the random-forest feature-importance interpretation, the zero-shot confusion matrix/accuracy comparison, and the detailed overlap-versus-no-overlap retrieval score table. Those should be completed and independently checked before treating the repository as final for grading.
