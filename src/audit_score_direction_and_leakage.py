"""Audit LOF score orientation and PaySim post-transaction balance dependence.

The script is read-only with respect to final experiment outputs. It writes one
audit JSON file. The PaySim ablation deliberately excludes post-transaction
balances and features derived from them before retraining HGB on the exact 70%
chronological training partition.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .rebuild_project import CONFIG, OUT, ROOT, dump, evaluate, threshold_f1
from .supervised_compare import full_training_matrix, partition_labels, scores


RESULT = ROOT / "supervised_output" / "leakage_audit.json"


def lof_orientation() -> list[dict]:
    folder = ROOT / "larger_training_output" / "paysim"
    rows = []
    for feature_set in ("reduced", "full"):
        table = pq.read_table(folder / f"LOF_{feature_set}_test_predictions.parquet",
                              columns=["anomaly_score", "reference_fraud"]).to_pandas()
        y = table["reference_fraud"].to_numpy()
        score = table["anomaly_score"].to_numpy()
        rows.append({
            "feature_set": feature_set,
            "stored_score_roc_auc": float(roc_auc_score(y, score)),
            "reversed_score_roc_auc": float(roc_auc_score(y, -score)),
            "stored_score_average_precision": float(average_precision_score(y, score)),
            "reversed_score_average_precision": float(average_precision_score(y, -score)),
            "mean_score_fraud": float(score[y == 1].mean()),
            "mean_score_nonfraud": float(score[y == 0].mean()),
        })
    return rows


def post_transaction_ablation() -> dict:
    """Train HGB only on information available before the transaction completes."""
    source = OUT / "paysim" / "engineered.parquet"
    total = pq.ParquetFile(source).metadata.num_rows
    train_end, validation_end = int(.7 * total), int(.85 * total)
    # Excluded: new balances, balance deltas, and residuals. These are known only
    # after applying a transaction and can act as simulator-specific target proxies.
    features = [
        "step", "hour", "weekday", "sin_hour", "cos_hour", "sin_weekday", "cos_weekday",
        "amount_log1p", "oldbalanceOrg_log1p", "oldbalanceDest_log1p", "is_zero_origin_balance",
        "is_type_CASH_IN", "is_type_CASH_OUT", "is_type_DEBIT", "is_type_PAYMENT", "is_type_TRANSFER",
    ]
    label = CONFIG["PaySim"]["label"]
    y_train = partition_labels(source, label, 0, train_end)
    y_validation = partition_labels(source, label, train_end, validation_end)
    y_test = partition_labels(source, label, validation_end, total)
    with tempfile.TemporaryDirectory(dir=ROOT / "supervised_output", prefix="paysim_pre_event_") as temp:
        x_train = full_training_matrix(source, features, train_end, Path(temp))
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=.08,
                                                max_leaf_nodes=31, l2_regularization=1.0,
                                                random_state=42, class_weight="balanced")
        model.fit(x_train, y_train)
        validation_score = scores(source, features, model, train_end, validation_end)
        threshold, validation_f1 = threshold_f1(y_validation, validation_score)
        test_score = scores(source, features, model, validation_end, total)
        del x_train
    return {
        "protocol": "HGB trained on all 70% rows with post-transaction balance features excluded",
        "features": features,
        "training_rows": int(train_end),
        "validation_best_f1": float(validation_f1),
        "test": evaluate(y_test, test_score, threshold),
    }


def main() -> None:
    result = {"lof_score_orientation": lof_orientation(),
              "paysim_pre_transaction_hgb_ablation": post_transaction_ablation()}
    dump(RESULT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
