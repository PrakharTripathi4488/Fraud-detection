# Project artifact map

This project keeps only the outputs needed to reproduce, explain, and present
the final experiments.

| Location | Purpose | Keep? |
| --- | --- | --- |
| `data/raw/` | Original BankSim CSV and PaySim archive | Yes |
| `src/` | All executable Python modules | Yes |
| `project_output/` | Cleaned/engineered Parquet files, diagnostics, feature lists, and the original sampled unsupervised baseline | Yes; final all-row scripts reuse its engineered files |
| `larger_training_output/` | Final all-row unsupervised models, predictions, and metrics | Yes |
| `supervised_output/` | Final full-training supervised models, comparison table, and PaySim pre-event leakage audit | Yes |
| `enhancement_output/` | Earlier validation-driven enhancement experiments retained as supporting evidence | Yes |
| `output/report/` | Generated report document and its source figures | Yes |
| `tmp/` and `__pycache__/` | Rendered-review images, temporary matrices, and Python bytecode | No; safe to delete and ignored by Git |

For the final presentation, use `supervised_output/comparison.csv` for the
complete 16-case comparison, `supervised_output/leakage_audit.json` for the
PaySim pre-event scope check, and `larger_training_output/metrics.json` for the
all-row unsupervised results.
