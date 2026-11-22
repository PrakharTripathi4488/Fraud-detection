# Controlled model enhancement results

## Method

The original run in `project_output/` is the baseline. `tune_models.py` fixes a
small candidate list before scoring it: amount/category/time subsets for BankSim,
amount/balance/type/time subsets for PaySim, Isolation Forest leaf and tree
settings, and LOF neighbor count. Its first pass selects by **validation F1**;
thresholds also maximize validation F1. A separate BankSim comparison selects
by **validation average precision (AP)**, a ranking metric unaffected by the
threshold. A fixed literature-inspired PaySim subset trial and a fixed
`RobustScaler` LOF trial are recorded separately. All models and scalers fit
without fraud labels; they retain the exact original 70/15/15 chronological
partitions. No candidate is chosen by a test metric in code.

## Confirmed improvements on the existing test partitions

| Dataset, detector | Change | Validation F1, old → new | Test precision | Test recall | Test F1, old → new | Test AP, old → new |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BankSim, Isolation Forest | Amount + 15 category indicators (selected by validation F1) | .211 → .301 | .143 → .223 | .320 → .455 | **.197 → .299** | .121 → .162 |
| BankSim, Isolation Forest | 1024 samples per tree rather than automatic 256 (selected by validation AP) | .211 → .297 | .143 → .213 | .320 → .494 | **.197 → .298** | **.121 → .233** |
| PaySim, LOF reduced | Median/IQR scaling instead of mean/std scaling | .022 → .051 | .080 → .175 | .476 → .225 | **.136 → .197** | **.055 → .131** |

The BankSim IF amount/category variant reduced false positives from **1,849 to
1,527** and increased detected fraud from **308 to 438** in 89,197 test rows.
The AP-selected 1024-leaf variant offers much better ranking with almost the
same F1; it is saved as a separate option. The PaySim robust LOF variant reduced
false positives from **22,041 to 4,266**, but caught **902 instead of 1,907**
fraud cases in 954,393 test rows. It is useful where alert capacity and
precision matter; it is worse where catching more fraud matters most.

The best overall test F1 in each dataset remains **BankSim reduced LOF .410**
and **PaySim reduced Isolation Forest .312**. The changes above improve the
other detector configurations, not those overall best baselines.

## Negative and mixed trials (retained as evidence)

- PaySim IF: the transaction/balance/type, time, residual and 1024-leaf
  candidates all scored below the reduced baseline on validation, so the
  baseline remained selected and its test results were reused.
- BankSim LOF: the baseline retained the highest validation F1. The 50-neighbor
  variant raised test AP from .276 to .286 when selected separately by
  validation AP, while test F1 slightly fell from .410 to .405.
- PaySim LOF using the paper-inspired S/log features raised validation F1 from
  .050 to .065 but test F1 fell from .035 to .011; test AP rose from .023 to
  .029. This is not a test F1 improvement.
- PaySim full LOF with robust scaling raised validation F1 from .050 to .139,
  yet test F1 **fell** from .035 to .018 and test AP from .023 to .010.
- BankSim reduced LOF with robust scaling lowered validation F1 from .431 to
  .116; it was not advanced to the test set.

The PaySim fraud rate shifts across the chronological partitions: **562 fraud
cases in validation** versus **4,008 in testing**, despite equal partition
sizes. This makes a validation threshold difficult to transfer and explains
why a validation gain can reverse on test. That explanation is an inference
from the observed counts, not a causal proof.

## Reproduce and audit

From the project root:

```powershell
python -m src.tune_models
python -c "from tune_models import evaluate_ap_selected_bank; evaluate_ap_selected_bank()"
python -c "from tune_models import evaluate_literature_paysim; evaluate_literature_paysim()"
python -c "from tune_models import evaluate_robust_lof; evaluate_robust_lof()"
```

`enhancement_output/evidence_comparison.csv` and each dataset's
`*_validation_candidates.csv` contain the complete primary search. The three
follow-up JSON files contain full validation and test counts, not just rounded
table values. Selected model bundles and test predictions are stored under the
dataset directory. The original test partition had already been examined in
the initial project run before this enhancement work. Therefore these results
are transparent comparisons on the existing test set, **not a fresh external
replication**. Confirm generalization with new unseen data before treating
small differences as established gains.
