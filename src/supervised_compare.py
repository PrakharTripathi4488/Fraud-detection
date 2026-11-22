"""Train supervised fraud classifiers and compare them to the unsupervised baseline.

The split boundaries and evaluation protocol deliberately match rebuild_project.py:
70% chronological training, 15% validation for the decision threshold, and 15%
held-out testing. Unlike the anomaly detectors, this script uses reference fraud
labels while fitting. Run after ``python -m src.rebuild_project``:

    python -m src.supervised_compare
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .rebuild_project import CONFIG, OUT, ROOT, dump, evaluate, threshold_f1

DEST = ROOT / "supervised_output"


def partition_labels(path: Path, label: str, start: int, stop: int) -> np.ndarray:
    values, offset = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=[label]):
        end = offset + len(batch)
        lo, hi = max(start, offset), min(stop, end)
        if hi > lo:
            values.append(batch.column(0).to_numpy()[lo-offset:hi-offset].astype(np.uint8))
        offset = end
        if offset >= stop:
            break
    result = np.concatenate(values)
    assert len(result) == stop-start
    return result


def training_frame(path: Path, columns: list[str], labels: np.ndarray, size: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep all available fraud rows and a seeded random normal sample."""
    if size < int(labels.sum()):
        raise ValueError("train-size is smaller than the number of training fraud rows")
    rng = np.random.default_rng(42)
    fraud = np.flatnonzero(labels == 1)
    normal = np.flatnonzero(labels == 0)
    chosen = np.sort(np.concatenate((fraud, rng.choice(normal, size-len(fraud), replace=False))))
    parts, offset = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=columns):
        end = offset + len(batch)
        a, b = np.searchsorted(chosen, (offset, end))
        if b > a:
            parts.append(batch.to_pandas().iloc[chosen[a:b]-offset])
        offset = end
        if offset >= len(labels):
            break
    frame = pd.concat(parts, ignore_index=True)
    y = labels[chosen]
    assert len(frame) == size and int(y.sum()) == len(fraud)
    return frame, y


def full_training_matrix(path: Path, columns: list[str], stop: int, scratch: Path) -> np.memmap:
    """Stream every row of the 70% training partition into a disk-backed matrix."""
    matrix = np.lib.format.open_memmap(scratch / "training.npy", mode="w+",
                                      dtype=np.float32, shape=(stop, len(columns)))
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=25_000, columns=columns):
        count = min(len(batch), stop-offset)
        if count <= 0:
            break
        matrix[offset:offset+count] = batch.to_pandas().iloc[:count].to_numpy(dtype=np.float32)
        offset += count
    assert offset == stop
    matrix.flush()
    return matrix


def scores(path: Path, columns: list[str], model, start: int, stop: int) -> np.ndarray:
    result, offset = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000, columns=columns):
        end = offset + len(batch)
        lo, hi = max(start, offset), min(stop, end)
        if hi > lo:
            frame = batch.to_pandas().iloc[lo-offset:hi-offset]
            result.append(model.predict_proba(frame.to_numpy(dtype=np.float32))[:, 1])
        offset = end
        if offset >= stop:
            break
    values = np.concatenate(result)
    assert len(values) == stop-start
    return values


def run_one(dataset: str, feature_set: str, train_size: int | None) -> list[dict]:
    folder = OUT / dataset.lower()
    source = folder / "engineered.parquet"
    n = pq.ParquetFile(source).metadata.num_rows
    train_end, validation_end = int(.7*n), int(.85*n)
    label = CONFIG[dataset]["label"]
    features = json.loads((folder / f"phase_3_5_{feature_set}.json").read_text(
        encoding="utf-8"))["features"]
    y_train = partition_labels(source, label, 0, train_end)
    y_validation = partition_labels(source, label, train_end, validation_end)
    y_test = partition_labels(source, label, validation_end, n)
    temporary = None
    if train_size is None:
        DEST.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=DEST, prefix=f"{dataset}_{feature_set}_full_")
        x_train = full_training_matrix(source, features, train_end, Path(temporary.name))
        sampled_y = y_train
    else:
        sample, sampled_y = training_frame(source, features, y_train, min(train_size, train_end))
        x_train = sample.to_numpy(dtype=np.float32)
    results = []
    try:
        for name, model in (
            ("LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=300, class_weight="balanced", solver="lbfgs"))),
            ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=200, learning_rate=.08,
                max_leaf_nodes=31, l2_regularization=1.0, random_state=42,
                class_weight="balanced")),
        ):
            started = time.perf_counter()
            model.fit(x_train, sampled_y)
            fit_seconds = time.perf_counter()-started
            validation_score = scores(source, features, model, train_end, validation_end)
            threshold, validation_f1 = threshold_f1(y_validation, validation_score)
            test_started = time.perf_counter()
            test_score = scores(source, features, model, validation_end, n)
            test_seconds = time.perf_counter()-test_started
            result = {"dataset": dataset, "model": name, "feature_set": feature_set,
                      "training_rows": len(sampled_y), "training_fraud_rows": int(sampled_y.sum()),
                      "threshold": threshold, "validation_best_f1": validation_f1,
                      "validation": evaluate(y_validation, validation_score, threshold),
                      "test": evaluate(y_test, test_score, threshold),
                      "fit_seconds": fit_seconds, "test_inference_seconds": test_seconds}
            results.append(result)
            target = DEST / dataset.lower(); target.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "features": features, "threshold": threshold,
                         "dataset": dataset, "feature_set": feature_set, "supervised": True},
                        target / f"{name}_{feature_set}.joblib")
            pq.write_table(pa.table({"source_row": np.arange(validation_end, n),
                                     "fraud_probability": test_score,
                                     "fraud_flag": (test_score >= threshold).astype(np.uint8),
                                     "reference_fraud": y_test}),
                           target / f"{name}_{feature_set}_test_predictions.parquet", compression="zstd")
            print(dataset, name, feature_set, "test F1", round(result["test"]["f1"], 4), flush=True)
    finally:
        if temporary is not None:
            del x_train
            temporary.cleanup()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=0,
                        help="training rows per case; 0 means every row in the 70%% partition")
    parser.add_argument("--dataset", choices=("BankSim", "PaySim", "both"), default="both")
    parser.add_argument("--feature-set", choices=("reduced", "full", "both"), default="both")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild the combined comparison CSV from saved results without training")
    args = parser.parse_args()
    if args.train_size != 0 and args.train_size < 10_000:
        parser.error("train-size must be 0 (full training) or at least 10,000")
    DEST.mkdir(exist_ok=True)
    datasets = ("BankSim", "PaySim") if args.dataset == "both" else (args.dataset,)
    feature_sets = ("reduced", "full") if args.feature_set == "both" else (args.feature_set,)
    supervised = [] if args.report_only else [row for dataset in datasets
                                               for feature_set in feature_sets
                                               for row in run_one(dataset, feature_set,
                                                                  None if args.train_size == 0 else args.train_size)]
    # The all-row unsupervised outputs are the presentation comparison, when present.
    full_unsupervised = ROOT / "larger_training_output" / "metrics.json"
    unsupervised_source = full_unsupervised if full_unsupervised.exists() else OUT / "metrics.json"
    unsupervised = json.loads(unsupervised_source.read_text(encoding="utf-8"))
    if args.report_only:
        saved = json.loads((DEST / "metrics.json").read_text(encoding="utf-8"))
    elif args.dataset == "both" and args.feature_set == "both":
        saved = supervised
    else:
        existing = json.loads((DEST / "metrics.json").read_text(encoding="utf-8")) if (DEST / "metrics.json").exists() else []
        saved = [row for row in existing if not (row["dataset"] in datasets and row["feature_set"] in feature_sets)] + supervised
    dump(DEST / "metrics.json", saved)
    comparison = []
    comparison.extend({"learning_type": "unsupervised", **row} for row in unsupervised)
    comparison.extend({"learning_type": "supervised", **row} for row in saved)
    pd.DataFrame([{"dataset": r["dataset"], "learning_type": r["learning_type"], "model": r["model"], "feature_set": r["feature_set"],
                   "test_precision": r["test"]["precision"], "test_recall": r["test"]["recall"],
                   "test_f1": r["test"]["f1"], "test_pr_auc": r["test"]["pr_auc_average_precision"],
                   "test_roc_auc": r["test"]["roc_auc"], "fit_seconds": r.get("fit_seconds")}
                 for r in comparison]).to_csv(DEST / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
