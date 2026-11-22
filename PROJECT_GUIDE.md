# Fraud Detection Project Guide

This guide explains what the project runs, which features each model sees, and
where each result is written. It is intended for reproducing the live project
demonstration and preparing the presentation.

## Start here

Run every experiment from the project root:

```powershell
python -m src.run_all_experiments
```

This reads the two raw datasets, rebuilds the cleaned and engineered data, runs
all unsupervised cases, then runs every supervised case.
It takes time because PaySim contains 6.36 million transactions. The console
prints each model's test F1-score when it finishes.

If data cleaning and feature engineering have already completed, rerun only the
models:

```powershell
python -m src.run_all_experiments --resume-engineered
```

Use only the normal chronological split for the presentation. It assigns the
first 70% of rows to training, the next 15% to validation, and the last 15% to
testing. No test labels are used to select thresholds or models.

## What is run

The complete matrix contains 16 cases.

| Learning type | Algorithms | Feature sets | Cases per dataset |
| --- | --- | --- | ---: |
| Unsupervised | Isolation Forest, Local Outlier Factor (LOF) | Reduced, full | 4 |
| Supervised | Logistic Regression, Histogram Gradient Boosting | Reduced, full | 4 |

Both BankSim and PaySim are evaluated, producing 8 unsupervised and 8
supervised result rows. The anomaly models do not receive fraud labels during
training. The supervised models use labels during fitting; all models select
their final anomaly/fraud cutoff using validation F1 only.

Every final model uses all rows in its chronological 70% training partition.
For LOF, every training row is indexed and receives a local-density estimate;
FAISS IVF supplies approximate neighbour search so this is feasible on PaySim.
Only the IVF coarse-quantizer training uses a capped 200,000-row sample. That
sample does not remove any row from LOF fitting or reference lookup.

## Dataset and split summary

| Dataset | Total rows | Training 70% | Validation 15% | Test 15% | Label column |
| --- | ---: | ---: | ---: | ---: | --- |
| BankSim | 594,643 | 416,250 | 89,196 | 89,197 | `fraud` |
| PaySim | 6,362,620 | 4,453,834 | 954,393 | 954,393 | `isFraud` |

Validation chooses the decision threshold. Testing is the final held-out
measurement. This distinction is important: a high validation result is not
presented as a final result until its score is measured on the test partition.

## Feature sets in plain language

All features are created from transaction fields. Account/customer identifiers,
postcodes, and PaySim's existing `isFlaggedFraud` rule flag are excluded. The
fraud label is also excluded from the unsupervised models.

### BankSim

The **reduced set** has 20 features: log transaction amount, cyclic day and
weekday features (`sin_day`, `cos_day`, `sin_weekday`, `cos_weekday`), and 15
merchant-category indicators. It deliberately keeps the features easiest to
explain in a presentation: how much was spent, when it happened, and what was
being purchased.

The **full set** has 36 features. It contains all reduced features plus raw
simulation time (`step`, `month`, `day`, `weekday`), eight age-group indicators,
and four gender indicators. The exact list is saved in
`project_output/banksim/phase_3_5_full.json`.

### PaySim

The **reduced set** has 14 features: log amount; four log balance values
(`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`);
cyclic hour and weekday features; and five transaction-type
indicators (Cash In, Cash Out, Debit, Payment, Transfer).

The **full set** has 22 features. It adds simulation time (`step`, `hour`,
`weekday`), log-transformed sender/destination balance changes, residuals
between balance movement and transaction amount, and a zero-origin-balance
indicator. The engineered `day` column is removed from the full set because it
is near-perfectly correlated with another time representation in the training
sample. The exact final list is saved in
`project_output/paysim/phase_3_5_full.json`.

## Algorithms

- **Isolation Forest:** isolates unusual transactions using random trees. Higher
  negated `score_samples` values mean a transaction is more anomalous.
- **LOF:** compares a transaction's local density with its neighbours. It uses
  20 Euclidean neighbours and FAISS IVF approximate lookup over every training
  row; this is an all-row approximation to exact LOF, not a sampled LOF model.
  The validation partition also establishes score orientation so that higher
  saved anomaly scores always mean greater suspicion.
- **Logistic Regression:** estimates fraud probability using a linear boundary.
  Inputs are standardized and class balancing is enabled.
- **Histogram Gradient Boosting:** learns nonlinear decision rules from the
  labelled transactions. It uses 200 iterations, 31 leaf nodes, learning rate
  0.08, and seed 42.

## Metrics

- **Precision:** proportion of flagged transactions that are fraud.
- **Recall:** proportion of all fraud transactions that were flagged.
- **F1-score:** harmonic balance of precision and recall; this selects the
  validation threshold.
- **PR-AUC / average precision:** ranking quality for the rare-fraud setting.
- **ROC-AUC:** ranking quality across false-positive thresholds.
- **Specificity and false-positive rate:** how often normal transactions remain
  unflagged and how often they are incorrectly flagged.

Do not use accuracy as the main result because normal transactions dominate both
datasets.

## Where to find results

| Location | Contents |
| --- | --- |
| `larger_training_output/metrics.json` | Final all-row unsupervised metric records |
| `larger_training_output/comparison.csv` | Flat all-row unsupervised comparison table |
| `larger_training_output/<dataset>/*_test_predictions.parquet` | Per-test-row anomaly score and flag |
| `supervised_output/metrics.json` | All eight supervised metric records |
| `supervised_output/comparison.csv` | Combined supervised and unsupervised comparison table |
| `supervised_output/<dataset>/*_test_predictions.parquet` | Per-test-row fraud probability and flag |
| `phase_3_5_reduced.json`, `phase_3_5_full.json` | The exact feature lists used by each dataset |

The current completed test results are in
`supervised_output/comparison.csv`. It contains all 16 rows (2 datasets × 2
feature sets × 4 algorithms); the `learning_type` column separates the two
approaches. For a report, quote the held-out `test_f1`, `test_pr_auc`, and
`test_roc_auc` columns rather than a validation score.

### PaySim real-time scope check

The full PaySim representation contains `newbalanceOrig`, `newbalanceDest`, and
derived balance deltas/residuals. Those values are available after transaction
processing and are therefore unsuitable for a pre-authorization claim. The
all-row HGB audit excluding these post-transaction features reports test
precision .940, recall .707, F1 .807, PR-AUC .899, and ROC-AUC .999. The
near-perfect full-feature HGB result is retained as a post-event simulator
classification result, not presented as a real-time fraud-decision result.

## Useful commands

```powershell
# Fresh complete run (all 16 cases)
python -m src.run_all_experiments

# Fastest model rerun after a completed feature build
python -m src.run_all_experiments --resume-engineered

# Only the all-row unsupervised comparisons
python -m src.large_training

# Only the supervised comparisons after the baseline
python -m src.supervised_compare

# Rebuild the combined table from saved results (no training)
python -m src.supervised_compare --report-only
```

## Presentation advice

Present the supervised and unsupervised tables separately first, then show the
combined comparison. Explain that supervised methods can use known fraud labels
and therefore can learn the simulated fraud patterns directly. Unsupervised
methods are still useful when labels are unavailable, delayed, or unreliable.
The unusually high PaySim supervised performance reflects strong signals in a
simulated dataset; it is not a production guarantee.
