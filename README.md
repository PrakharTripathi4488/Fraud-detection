# Fraud detection

This project implements sections 3.3–3.9 and section 4 of
`Midsem_Project_Report.pdf` for **only** these two inputs:

| Input | Rows | Fraud | Source |
| --- | ---: | ---: | --- |
| `data/raw/BankSim_Dataset.csv` | 594,643 | 7,200 | BankSim CSV corresponding to `ealaxi/banksim1` |
| `data/raw/PaySim_Dataset.zip` | 6,362,620 | 8,213 | The 493,534,783-byte `PS_20174392719_1491204439457_log.csv` from `ealaxi/paysim1` |

The PaySim decompressed CSV has SHA-256
`16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.
The previous 571,037-row PaySim export and 1,048,575-row standalone CSV are not
used. The raw zip stays compressed; the pipeline reads it in chunks.

## Reproduce

For the final all-row experiment, run `python -m src.run_all_experiments` from this
directory. Dependencies: Python 3.12, NumPy, pandas, pyarrow, scikit-learn,
matplotlib, joblib, and `faiss-cpu` (`python -m pip install faiss-cpu`). It
prepares the data, then fits every model using the complete 70% training
partition. The all-row unsupervised artifacts are written to
`larger_training_output/`; supervised artifacts are in `supervised_output/`.

`python -m src.rebuild_project` and `python -m src.verify_project` remain available for
the original sampled baseline only; they are not the final all-row workflow.

## Supervised comparison

Run `python -m src.supervised_compare` after the baseline. It trains Logistic
Regression and histogram gradient boosting with fraud labels on the same 70/15/15
chronological partitions. Thresholds are chosen only on validation. Results and a
presentation-ready comparison table are written to `supervised_output/`.

## Complete live run and documentation

Use `python -m src.run_all_experiments` to rebuild and evaluate every unsupervised
and supervised reduced/full feature-set case. See [PROJECT_GUIDE.md](PROJECT_GUIDE.md)
for the commands, exact features, algorithms, metrics, outputs, and presentation
interpretation.

## Implementation by report phase

| Phase | Work done | Output per dataset |
| --- | --- | --- |
| 3.3 | Audited all raw rows for nulls, numeric validity, data types, duplicate row hashes, categorical frequencies, and fraud prevalence. Calculated full-data mean, variance, standard deviation, min, quartiles, max, skewness, and IQR outlier counts. Cleaned BankSim string quoting and added `log1p` to amounts and PaySim balances. Produced amount/step plots and PaySim balance plots. No outliers were removed merely for being rare. | `cleaned.parquet`, `phase_3_3.json`, diagnostic PNGs |
| 3.4 | PaySim: hour, relative day/weekday, cyclic sine/cosine, transaction-type indicators, balance changes/residuals. BankSim: relative month/day/weekday, cycles, gender/category/age indicators. IDs, postcodes and the existing PaySim rule flag do not become model inputs. | `engineered.parquet`, `phase_3_4.json` |
| 3.5 | Compared a reduced interpretable set with the full engineered set. Removed constant and near perfectly correlated columns using an unlabeled training sample only. Fitted the scaler on **every row of the training partition** only. | `phase_3_5_reduced.json`, `phase_3_5_full.json` |
| 3.6 | Isolation Forest with 100 trees, automatic tree sample size, seed 42; novelty-mode LOF with 20 Euclidean/Minkowski neighbors. Every model emits a numeric anomaly score increasing with suspicion. | Four `.joblib` model/scaler/threshold bundles |
| 3.7 | Stable chronological row-order 70% train, 15% validation, 15% test. Gold labels excluded from model fitting and scaling. | `phase_3_7.json` |
| 3.8 | Used validation labels to choose the anomaly cutoff maximizing F1. Labels from the test set were never used for selection. | Thresholds in model bundles and results JSON |
| 3.9 / 4 | Compared eight dataset × model × feature-set cases on the held-out test partition. Reported precision, recall, F1, average precision (PR-AUC summary), ROC-AUC, specificity, false-positive rate, confusion counts, and fit/inference times. | `project_output/comparison.csv`, `metrics.json`, four test prediction Parquet files per dataset |

The per-transaction test files contain `source_row`, `anomaly_score`, `anomaly_flag`
(0 normal, 1 suspected anomaly), and `reference_fraud` for checking performance.
Their model identity is recorded by the file name and model bundle. To score a
new transaction, derive its features with `engineer()`, select the bundle's feature
list, transform with its scaler, negate `model.score_samples()`, and compare to
its stored threshold.

### Split sizes

| Dataset | Training | Validation | Testing | Validation fraud | Test fraud |
| --- | ---: | ---: | ---: | ---: | ---: |
| BankSim | 416,250 | 89,196 | 89,197 | 970 | 962 |
| PaySim | 4,453,834 | 954,393 | 954,393 | 562 | 4,008 |

Rows at the same simulated step may fall on opposite sides of a percentage
boundary. Time features are relative simulation calendars, because the files do
not supply a true calendar start date. The full feature sets start with 36 BankSim
and 23 PaySim engineered columns; the training-only selection removes one
near-duplicate PaySim column before modeling.

### Test comparison

| Dataset | Model | Set | Precision | Recall | F1 | AP / PR-AUC | ROC-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| BankSim | Isolation Forest | Reduced | .103 | .500 | .170 | .094 | .936 |
| BankSim | LOF (approximate all-row) | Reduced | .011 | 1.000 | .021 | .006 | .063 |
| BankSim | Isolation Forest | Full | .065 | .905 | .122 | .066 | .915 |
| BankSim | LOF (approximate all-row) | Full | .011 | 1.000 | .021 | .006 | .053 |
| PaySim | Isolation Forest | Reduced | .470 | .263 | .337 | .275 | .837 |
| PaySim | LOF (approximate all-row) | Reduced | .038 | .085 | .053 | .021 | .712 |
| PaySim | Isolation Forest | Full | .155 | .208 | .177 | .088 | .926 |
| PaySim | LOF (approximate all-row) | Full | .652 | .202 | .308 | .275 | .967 |

**Resource and interpretation limits:** In the final all-row run, the scaler,
Isolation Forest training input, supervised training input, and LOF reference
index all contain every row in the chronological 70% partition. LOF uses FAISS
IVF approximate nearest-neighbour search so that complete-partition local-density
calculation is practical; its IVF coarse quantizer is trained on at most 200,000
rows, but no training rows are excluded from the index or density estimate. The PaySim validation
partition has 562 fraud cases while the later test partition has 4,008, so the
fraud prevalence shifts sharply across time. PR-AUC is calculated as average
precision, a threshold-independent ranking statistic; only F1 selects the
binary decision threshold.

**PaySim deployment scope:** The full PaySim feature set includes post-transaction
new balances and residuals derived from them. It produces a simulator-specific
post-event HGB result of F1=.9998. A full-training pre-event ablation that excludes
these post-transaction fields obtains F1=.8068 and PR-AUC=.8994. Use the ablation,
not the near-perfect full-feature result, when discussing a real-time transaction
decision; both results are retained for transparent comparison.

## Validation-driven enhancement experiments

Run `python -m src.tune_models` to compare the fixed feature and parameter search
against the baseline. Additional published-subset, robust-scaling and
average-precision choices and their exact commands are documented in
`ENHANCEMENT_REPORT.md`. Their models and full candidate metrics are stored in
`enhancement_output/`. The original `project_output/` baseline is preserved so
every reported change can be checked. The strongest verified changes are a
BankSim Isolation Forest F1 increase from .197 to .299 and a PaySim reduced
LOF F1 increase from .136 to .197, with the latter losing substantial recall.
The report also records candidates that failed on the test partition.
