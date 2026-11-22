"""Rebuild the midterm project from the two raw Kaggle-format datasets.

Run `python -m src.rebuild_project` in this directory. Every stage writes its audit
before the next begins. Requires numpy, pandas, pyarrow, scikit-learn, matplotlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "project_output"
CHUNK = 50_000
CATEGORIES = ("es_barsandrestaurants", "es_contents", "es_fashion", "es_food",
              "es_health", "es_home", "es_hotelservices", "es_hyper", "es_leisure",
              "es_otherservices", "es_sportsandtoys", "es_tech",
              "es_transportation", "es_travel", "es_wellnessandbeauty")
TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
CONFIG = {
    "BankSim": {"raw": RAW / "BankSim_Dataset.csv", "rows": 594643,
                "label": "fraud", "numerics": ["step", "amount"]},
    "PaySim": {"raw": RAW / "PaySim_Dataset.zip", "rows": 6362620,
               "label": "isFraud", "numerics": ["step", "amount", "oldbalanceOrg",
               "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]},
}


def dump(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, default=float), encoding="utf-8")


def input_chunks(name: str):
    path = CONFIG[name]["raw"]
    if name == "PaySim":
        with zipfile.ZipFile(path) as z:
            members = z.namelist()
            if members != ["PS_20174392719_1491204439457_log.csv"]:
                raise ValueError(f"Unexpected PaySim archive contents: {members}")
            with z.open(members[0]) as stream:
                yield from pd.read_csv(stream, chunksize=CHUNK)
    else:
        yield from pd.read_csv(path, chunksize=CHUNK)


def clean(name: str, folder: Path) -> dict:
    """3.3: validate and clean the complete raw input; audit distributions."""
    spec = CONFIG[name]
    path = folder / "cleaned.parquet"
    hashes = folder / "row_hashes.bin"
    writer = None
    count = kept = invalid = 0
    nulls = {}
    dtypes = {}
    class_counts = {0: 0, 1: 0}
    step_min, step_max = 10**9, -1
    sample = []
    started = time.perf_counter()
    with hashes.open("wb") as hash_out:
        try:
            for frame in input_chunks(name):
                n = len(frame)
                count += n
                if name == "BankSim":
                    for col in frame.select_dtypes(include="object"):
                        frame[col] = frame[col].str.strip("'\" \t")
                    if set(frame.category.dropna()) - set(CATEGORIES):
                        raise ValueError("Unexpected BankSim category")
                elif set(frame.type.dropna()) - set(TYPES):
                    raise ValueError("Unexpected PaySim type")
                for col in frame:
                    nulls[col] = nulls.get(col, 0) + int(frame[col].isna().sum())
                dtypes = {col: str(dtype) for col, dtype in frame.dtypes.items()}
                label = spec["label"]
                valid = frame[label].isin([0, 1])
                for col in spec["numerics"]:
                    valid &= np.isfinite(frame[col]) & (frame[col] >= 0)
                invalid += int((~valid).sum())
                frame = frame.loc[valid].copy()
                class_counts[0] += int((frame[label] == 0).sum())
                class_counts[1] += int((frame[label] == 1).sum())
                step_min = min(step_min, int(frame.step.min()))
                step_max = max(step_max, int(frame.step.max()))
                if not frame.step.is_monotonic_increasing:
                    raise ValueError("Raw input is not chronological by step")
                for col in (['amount'] if name == 'BankSim' else
                            ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                             'oldbalanceDest', 'newbalanceDest']):
                    frame[col + '_log1p'] = np.log1p(frame[col])
                pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64).tofile(hash_out)
                if len(sample) < 2:
                    sample.append(frame.sample(min(25000, len(frame)), random_state=42))
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                kept += len(frame)
                print(f"{name} 3.3: {count:,}/{spec['rows']:,}", end="\r", flush=True)
        finally:
            if writer is not None:
                writer.close()
    if count != spec["rows"]:
        raise ValueError(f"Expected {spec['rows']} raw rows but found {count}")
    # 64-bit row hashes provide a memory-bounded global duplicate audit.
    hashes_array = np.fromfile(hashes, dtype=np.uint64)
    hashes_array.sort()
    duplicate_hashes = int(np.count_nonzero(hashes_array[1:] == hashes_array[:-1]))
    del hashes_array
    hashes.unlink()
    if duplicate_hashes:
        raise ValueError(f"{duplicate_hashes} repeated row hashes; inspect before modeling")
    statistics = {}
    for col in spec["numerics"] + [c + "_log1p" for c in spec["numerics"] if c != "step"]:
        values = pd.read_parquet(path, columns=[col])[col].to_numpy(dtype=np.float64)
        q = np.quantile(values, [.25, .5, .75])
        statistics[col] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                           "variance": float(values.var(ddof=1)), "min": float(values.min()),
                           "q1": float(q[0]), "median": float(q[1]), "q3": float(q[2]),
                           "max": float(values.max()), "skewness": float(pd.Series(values).skew()),
                           "iqr_outliers": int(((values < q[0] - 1.5*(q[2]-q[0])) |
                                                 (values > q[2] + 1.5*(q[2]-q[0]))).sum())}
        del values
    plot_sample = pd.concat(sample)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(plot_sample.amount, bins=60, log=True)
    axs[0].set(title=f"{name}: raw amount (log count)", xlabel="Amount")
    axs[1].boxplot(plot_sample.amount_log1p.to_numpy(), vert=False)
    axs[1].set(title=f"{name}: log1p(amount) sample", xlabel="log1p(amount)")
    fig.tight_layout()
    fig.savefig(folder / "amount_diagnostics.png", dpi=150)
    plt.close(fig)
    audit = {"source": str(spec['raw']), "raw_rows": count, "cleaned_rows": kept,
             "invalid_rows_removed": invalid, "missing_by_raw_column": nulls,
             "duplicate_row_hashes": duplicate_hashes, "dtypes": dtypes,
             "fraud_counts": class_counts, "step_range": [step_min, step_max],
             "numeric_statistics": statistics, "seconds": time.perf_counter()-started}
    dump(folder / "phase_3_3.json", audit)
    print(f"{name} 3.3 complete: {kept:,} rows, {duplicate_hashes} duplicate hashes")
    return audit


def finish_diagnostics(name: str, folder: Path) -> None:
    """3.3 plots and category audit sampled across the entire simulation window."""
    category_columns = ['gender', 'category'] if name == 'BankSim' else ['type']
    balance_columns = ([] if name == 'BankSim' else
                       ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])
    needed = ['step', 'amount', 'amount_log1p'] + category_columns + balance_columns
    needed += [c+'_log1p' for c in balance_columns]
    counts = {c:{} for c in category_columns}
    steps = {}
    samples = []
    for i, batch in enumerate(pq.ParquetFile(folder/'cleaned.parquet').iter_batches(
            batch_size=CHUNK, columns=needed)):
        part = batch.to_pandas()
        for c in category_columns:
            for value, count in part[c].value_counts(dropna=False).items():
                key = str(value)
                counts[c][key] = counts[c].get(key,0)+int(count)
        for value, count in part.step.value_counts().items():
            steps[int(value)] = steps.get(int(value),0)+int(count)
        samples.append(part.sample(min(400,len(part)),random_state=42+i))
    sample = pd.concat(samples,ignore_index=True)
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(sample.amount,bins=60,log=True)
    axes[0].set(title=f'{name}: raw amount (distributed sample)',xlabel='Amount')
    axes[1].boxplot(sample.amount_log1p.to_numpy(),vert=False)
    axes[1].set(title=f'{name}: transformed amount',xlabel='log1p(Amount)')
    fig.tight_layout();fig.savefig(folder/'amount_diagnostics.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3))
    ax.plot(sorted(steps),[steps[k] for k in sorted(steps)])
    ax.set(title=f'{name}: transactions per simulation step',xlabel='Step',ylabel='Count')
    fig.tight_layout();fig.savefig(folder/'step_distribution.png',dpi=150);plt.close(fig)
    if balance_columns:
        fig,axes=plt.subplots(4,2,figsize=(12,12))
        for row,col in enumerate(balance_columns):
            axes[row,0].hist(sample[col],bins=50,log=True)
            axes[row,0].set(title=col+' raw',xlabel='Balance')
            axes[row,1].boxplot(sample[col+'_log1p'].to_numpy(),vert=False)
            axes[row,1].set(title=col+' log1p',xlabel='log1p(Balance)')
        fig.tight_layout();fig.savefig(folder/'balance_diagnostics.png',dpi=150);plt.close(fig)
    audit=json.loads((folder/'phase_3_3.json').read_text(encoding='utf-8'))
    audit['categorical_counts']=counts
    audit['plots_sample_rows']=len(sample)
    audit['plot_sampling']='400 rows per 50,000-row batch across complete dataset'
    dump(folder/'phase_3_3.json',audit)


def engineer(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """3.4: relative simulation calendars, cyclic features, and fixed one-hot schema."""
    step = frame.step.to_numpy(dtype=np.int32)
    x = pd.DataFrame(index=frame.index)
    x["step"] = step
    if name == "PaySim":
        hour = step % 24
        day = (step - 1) // 24
        weekday = day % 7
        x["hour"], x["day"], x["weekday"] = hour, day, weekday
        x["sin_hour"], x["cos_hour"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
        x["sin_weekday"], x["cos_weekday"] = (np.sin(2*np.pi*weekday/7),
                                               np.cos(2*np.pi*weekday/7))
        for col in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
                    "newbalanceDest"):
            x[col + "_log1p"] = frame[col + "_log1p"].to_numpy(dtype=np.float32)
        origin = frame.oldbalanceOrg - frame.newbalanceOrig
        dest = frame.newbalanceDest - frame.oldbalanceDest
        for col, difference in (("origin_delta_log1p", origin),
                                ("destination_delta_log1p", dest),
                                ("origin_residual_log1p", origin-frame.amount),
                                ("destination_residual_log1p", dest-frame.amount)):
            x[col] = np.sign(difference) * np.log1p(np.abs(difference))
        x["is_zero_origin_balance"] = (frame.oldbalanceOrg == 0).astype(np.uint8)
        for kind in TYPES:
            x["is_type_" + kind] = (frame.type == kind).astype(np.uint8)
    else:
        day, month, weekday = step % 30, step // 30, step % 7
        x["month"], x["day"], x["weekday"] = month, day, weekday
        x["sin_day"], x["cos_day"] = np.sin(2*np.pi*day/30), np.cos(2*np.pi*day/30)
        x["sin_weekday"], x["cos_weekday"] = np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)
        x["amount_log1p"] = frame.amount_log1p.to_numpy(dtype=np.float32)
        for age in ("0", "1", "2", "3", "4", "5", "6", "U"):
            x["is_age_"+age] = (frame.age.astype(str) == age).astype(np.uint8)
        for gender in ("E", "F", "M", "U"):
            x["is_gender_"+gender] = (frame.gender == gender).astype(np.uint8)
        for category in CATEGORIES:
            x["is_category_"+category] = (frame.category == category).astype(np.uint8)
    x[CONFIG[name]["label"]] = frame[CONFIG[name]["label"]].to_numpy(dtype=np.uint8)
    return x


def feature_stage(name: str, folder: Path, rows: int) -> list[str]:
    writer = None
    path = folder / "engineered.parquet"
    try:
        for batch in pq.ParquetFile(folder / "cleaned.parquet").iter_batches(batch_size=CHUNK):
            engineered = engineer(batch.to_pandas(), name)
            table = pa.Table.from_pandas(engineered, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    columns = pq.ParquetFile(path).schema_arrow.names
    features = [c for c in columns if c != CONFIG[name]["label"]]
    assert len(features) == (23 if name == "PaySim" else 36)
    assert pq.ParquetFile(path).metadata.num_rows == rows
    dump(folder / "phase_3_4.json", {"engineered_columns": features,
         "rows": rows, "relative_calendar": True,
         "categorical_encoding": "fixed schema; identifiers and rule flag excluded"})
    print(f"{name} 3.4 complete: {len(features)} engineered features")
    return features


def choose(candidate: list[str], sampled: pd.DataFrame) -> tuple[list[str], dict]:
    constant = [c for c in candidate if sampled[c].nunique() <= 1]
    selected = [c for c in candidate if c not in constant]
    continuous = [c for c in selected if not c.startswith('is_')]
    corr = sampled[continuous].corr().abs()
    redundant = []
    for i, a in enumerate(continuous):
        if a in redundant:
            continue
        redundant.extend(b for b in continuous[i+1:]
                         if b not in redundant and corr.loc[a,b] >= .999)
    return [c for c in selected if c not in redundant], {"constant": constant,
                                                              "correlated": redundant}


def threshold_f1(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = np.divide(2*precision[:-1]*recall[:-1], precision[:-1]+recall[:-1],
                   out=np.zeros_like(thresholds), where=(precision[:-1]+recall[:-1]) != 0)
    idx = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[idx]), float(f1[idx])


def evaluate(y: np.ndarray, score: np.ndarray, cutoff: float) -> dict:
    pred = score >= cutoff
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y,pred,zero_division=0)),
            "recall": float(recall_score(y,pred,zero_division=0)),
            "f1": float(f1_score(y,pred,zero_division=0)),
            "pr_auc_average_precision": float(average_precision_score(y,score)),
            "roc_auc": float(roc_auc_score(y,score)), "specificity": float(tn/(tn+fp)),
            "fpr": float(fp/(fp+tn)), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)}


def model_stage(name: str, folder: Path, columns: list[str], n: int,
                if_size: int, lof_size: int) -> list[dict]:
    """3.5–3.9: train-only feature reduction and scaling, fitting, tuning, test."""
    train_end, val_end = int(.7*n), int(.85*n)
    rng = np.random.default_rng(42)
    selected_indices = np.sort(rng.choice(train_end, min(if_size,train_end), replace=False))
    sample_parts = []
    scaler_all = StandardScaler()
    label = CONFIG[name]["label"]
    offset = 0
    split_labels = {"train": 0, "validation": [], "test": []}
    parquet = pq.ParquetFile(folder / "engineered.parquet")
    for batch in parquet.iter_batches(batch_size=CHUNK):
        part = batch.to_pandas()
        start, end = offset, offset+len(part)
        train_count = max(0, min(end,train_end)-start)
        if train_count:
            scaler_all.partial_fit(part.iloc[:train_count][columns].to_numpy(dtype=np.float32))
            split_labels['train'] += int(part.iloc[:train_count][label].sum())
            a, b = np.searchsorted(selected_indices, [start, start+train_count])
            if b > a:
                sample_parts.append(part.iloc[selected_indices[a:b]-start][columns])
        v1, v2 = max(start,train_end), min(end,val_end)
        t1 = max(start,val_end)
        if v2>v1:
            split_labels['validation'].append(part.iloc[v1-start:v2-start][label].to_numpy(dtype=np.uint8))
        if end>t1:
            split_labels['test'].append(part.iloc[t1-start:][label].to_numpy(dtype=np.uint8))
        offset = end
    assert offset == n
    sample = pd.concat(sample_parts, ignore_index=True)
    ys = {s: np.concatenate(split_labels[s]) for s in ('validation','test')}
    assert all(set(np.unique(y)) == {0, 1} for y in ys.values())
    dump(folder / "phase_3_7.json", {"method":"chronological row order",
         "counts":{"train":train_end,"validation":val_end-train_end,"test":n-val_end},
         "fraud_counts":{"train":split_labels['train'],
                         "validation":int(ys['validation'].sum()),
                         "test":int(ys['test'].sum())},
         "warning":"Equal step values can cross exact row percentage boundaries."})
    reduced = (["amount_log1p", "sin_day", "cos_day", "sin_weekday", "cos_weekday"]+
               ["is_category_"+c for c in CATEGORIES] if name == 'BankSim' else
               ["amount_log1p", "oldbalanceOrg_log1p", "newbalanceOrig_log1p",
                "oldbalanceDest_log1p", "newbalanceDest_log1p", "sin_hour", "cos_hour",
                "sin_weekday", "cos_weekday"]+["is_type_"+t for t in TYPES])
    rows = []
    for set_name, candidates in (("reduced",reduced),("full",columns)):
        features, removed = choose(candidates, sample)
        scaler = StandardScaler()
        indices = [columns.index(c) for c in features]
        scaler.mean_ = scaler_all.mean_[indices].copy()
        scaler.var_ = scaler_all.var_[indices].copy()
        scaler.scale_ = scaler_all.scale_[indices].copy()
        scaler.n_features_in_ = len(features)
        scaler.n_samples_seen_ = train_end
        dump(folder / f"phase_3_5_{set_name}.json", {"features":features,
             "removed_from_training_sample":removed,
             "training_rows_for_scaler":train_end,
             "sampling_seed":42,"model_reference_rows":len(sample)})
        matrix = scaler.transform(sample[features].to_numpy(dtype=np.float32))
        models = {}
        times = {}
        for model_name in ('IsolationForest','LOF'):
            start = time.perf_counter()
            if model_name == 'IsolationForest':
                model = IsolationForest(n_estimators=100,max_samples='auto',
                                        random_state=42,n_jobs=1).fit(matrix)
                fit_count = len(matrix)
            else:
                inds = rng.choice(len(matrix), min(lof_size,len(matrix)), replace=False)
                model = LocalOutlierFactor(n_neighbors=20,metric='minkowski',p=2,
                                           novelty=True,n_jobs=1).fit(matrix[inds])
                fit_count = len(inds)
            models[model_name] = (model,fit_count)
            times[model_name] = {"fit_seconds":time.perf_counter()-start}
        scores = {m:{'validation':np.empty(len(ys['validation']),dtype=np.float32),
                     'test':np.empty(len(ys['test']),dtype=np.float32)} for m in models}
        offset = 0
        cursors = {'validation':0,'test':0}
        for batch in parquet.iter_batches(batch_size=5000, columns=features):
            part = batch.to_pandas()
            end = offset+len(part)
            scaled = scaler.transform(part.to_numpy(dtype=np.float32))
            for split, lo, hi in (('validation',max(offset,train_end),min(end,val_end)),
                                  ('test',max(offset,val_end),end)):
                if hi <= lo:
                    continue
                subsection = scaled[lo-offset:hi-offset]
                for model_name, (model,_) in models.items():
                    start = time.perf_counter()
                    vals = -model.score_samples(subsection)
                    scores[model_name][split][cursors[split]:cursors[split]+len(vals)] = vals
                    times[model_name][split+'_inference_seconds'] = (
                        times[model_name].get(split+'_inference_seconds',0)+time.perf_counter()-start)
                cursors[split] += len(subsection)
            offset = end
        assert offset == n and all(cursors[s]==len(ys[s]) for s in cursors)
        for model_name, (model,fit_count) in models.items():
            val, test = scores[model_name]['validation'], scores[model_name]['test']
            cutoff, val_f1 = threshold_f1(ys['validation'],val)
            result = {"dataset":name,"feature_set":set_name,"model":model_name,
                      "features":len(features), "fit_reference_rows":fit_count,
                      "threshold":cutoff,"validation_best_f1":val_f1,
                      "validation":evaluate(ys['validation'],val,cutoff),
                      "test":evaluate(ys['test'],test,cutoff), **times[model_name]}
            rows.append(result)
            stem = f'{model_name}_{set_name}'
            joblib.dump({"model":model,"scaler":scaler,"features":features,
                         "threshold":cutoff,"dataset":name},folder / f'{stem}.joblib')
            pq.write_table(pa.table({"source_row":np.arange(val_end,n),
                                     "anomaly_score":test,
                                     "anomaly_flag":(test>=cutoff).astype(np.uint8),
                                     "reference_fraud":ys['test']}),
                           folder / f'{stem}_test_predictions.parquet',compression='zstd')
            print(f'{name} {stem}: test F1 {result["test"]["f1"]:.4f}, '
                  f'AP {result["test"]["pr_auc_average_precision"]:.4f}',flush=True)
    dump(folder / 'phase_3_6_to_3_9.json',rows)
    return rows


def model_stage_random(name: str, folder: Path, columns: list[str], n: int,
                       if_size: int, lof_size: int) -> list[dict]:
    """Run phases 3.5-3.9 with an exact label-independent random 70/15/15 split."""
    train_n, validation_n = int(.7*n), int(.15*n)
    rng = np.random.default_rng(42)
    partitions = np.empty(n, dtype=np.uint8)
    permutation = rng.permutation(n)
    partitions[permutation[:train_n]] = 0
    partitions[permutation[train_n:train_n+validation_n]] = 1
    partitions[permutation[train_n+validation_n:]] = 2
    del permutation
    sample_mask = np.zeros(n, dtype=bool)
    sample_mask[rng.choice(np.flatnonzero(partitions == 0), min(if_size, train_n), replace=False)] = True
    label, scaler_all, sample_parts = CONFIG[name]['label'], StandardScaler(), []
    split_labels, test_source_rows, offset, train_fraud = {'validation': [], 'test': []}, [], 0, 0
    parquet = pq.ParquetFile(folder / 'engineered.parquet')
    for batch in parquet.iter_batches(batch_size=CHUNK):
        part = batch.to_pandas(); end = offset + len(part); group = partitions[offset:end]
        train_mask = group == 0
        if train_mask.any():
            scaler_all.partial_fit(part.loc[train_mask, columns].to_numpy(dtype=np.float32))
            train_fraud += int(part.loc[train_mask, label].sum())
        if sample_mask[offset:end].any():
            sample_parts.append(part.loc[sample_mask[offset:end], columns])
        for split, code in (('validation', 1), ('test', 2)):
            selected = group == code
            if selected.any():
                split_labels[split].append(part.loc[selected, label].to_numpy(dtype=np.uint8))
                if split == 'test': test_source_rows.append(np.flatnonzero(selected).astype(np.int64) + offset)
        offset = end
    sample = pd.concat(sample_parts, ignore_index=True)
    ys = {split: np.concatenate(values) for split, values in split_labels.items()}
    source_rows = np.concatenate(test_source_rows)
    assert len(sample) == min(if_size, train_n) and len(ys['validation']) == validation_n
    assert len(ys['test']) == n-train_n-validation_n and all(set(np.unique(y)) == {0, 1} for y in ys.values())
    dump(folder / 'phase_3_7.json', {'method': 'seeded random row assignment independent of labels', 'seed': 42,
         'counts': {'train': train_n, 'validation': validation_n, 'test': n-train_n-validation_n},
         'fraud_counts': {'train': train_fraud, 'validation': int(ys['validation'].sum()), 'test': int(ys['test'].sum())},
         'label_use': 'Fraud labels are excluded from allocation, fitting and scaling.'})
    reduced = (["amount_log1p", "sin_day", "cos_day", "sin_weekday", "cos_weekday"] + ["is_category_"+c for c in CATEGORIES] if name == 'BankSim' else ["amount_log1p", "oldbalanceOrg_log1p", "newbalanceOrig_log1p", "oldbalanceDest_log1p", "newbalanceDest_log1p", "sin_hour", "cos_hour", "sin_weekday", "cos_weekday"] + ["is_type_"+t for t in TYPES])
    rows = []
    for set_name, candidates in (("reduced", reduced), ("full", columns)):
        features, removed = choose(candidates, sample); indexes = [columns.index(c) for c in features]
        scaler = StandardScaler(); scaler.mean_ = scaler_all.mean_[indexes].copy(); scaler.var_ = scaler_all.var_[indexes].copy(); scaler.scale_ = scaler_all.scale_[indexes].copy(); scaler.n_features_in_ = len(features); scaler.n_samples_seen_ = train_n
        dump(folder / f'phase_3_5_{set_name}.json', {'features': features, 'removed_from_training_sample': removed, 'training_rows_for_scaler': train_n, 'sampling_seed': 42, 'model_reference_rows': len(sample)})
        matrix = scaler.transform(sample[features].to_numpy(dtype=np.float32)); models, times = {}, {}
        for model_name in ('IsolationForest', 'LOF'):
            started = time.perf_counter()
            if model_name == 'IsolationForest': model, fit_count = IsolationForest(n_estimators=100, max_samples='auto', random_state=42, n_jobs=1).fit(matrix), len(matrix)
            else:
                inds = rng.choice(len(matrix), min(lof_size, len(matrix)), replace=False); model, fit_count = LocalOutlierFactor(n_neighbors=20, metric='minkowski', p=2, novelty=True, n_jobs=1).fit(matrix[inds]), len(inds)
            models[model_name] = (model, fit_count); times[model_name] = {'fit_seconds': time.perf_counter()-started}
        scores = {m: {'validation': [], 'test': []} for m in models}; offset = 0
        for batch in parquet.iter_batches(batch_size=5000, columns=features):
            part = batch.to_pandas(); end = offset+len(part); scaled = scaler.transform(part.to_numpy(dtype=np.float32)); group = partitions[offset:end]
            for split, code in (('validation', 1), ('test', 2)):
                selected = group == code
                if selected.any():
                    for model_name, (model, _) in models.items():
                        started = time.perf_counter(); scores[model_name][split].append(-model.score_samples(scaled[selected])); key = split+'_inference_seconds'; times[model_name][key] = times[model_name].get(key, 0)+time.perf_counter()-started
            offset = end
        for model_name, (model, fit_count) in models.items():
            validation, test = np.concatenate(scores[model_name]['validation']), np.concatenate(scores[model_name]['test'])
            cutoff, validation_f1 = threshold_f1(ys['validation'], validation)
            result = {'dataset': name, 'feature_set': set_name, 'model': model_name, 'features': len(features), 'fit_reference_rows': fit_count, 'threshold': cutoff, 'validation_best_f1': validation_f1, 'validation': evaluate(ys['validation'], validation, cutoff), 'test': evaluate(ys['test'], test, cutoff), **times[model_name]}
            rows.append(result); stem = f'{model_name}_{set_name}'
            joblib.dump({'model': model, 'scaler': scaler, 'features': features, 'threshold': cutoff, 'dataset': name}, folder / f'{stem}.joblib')
            pq.write_table(pa.table({'source_row': source_rows, 'anomaly_score': test, 'anomaly_flag': (test >= cutoff).astype(np.uint8), 'reference_fraud': ys['test']}), folder / f'{stem}_test_predictions.parquet', compression='zstd')
            print(f'{name} {stem}: test F1 {result["test"]["f1"]:.4f}, AP {result["test"]["pr_auc_average_precision"]:.4f}', flush=True)
    dump(folder / 'phase_3_6_to_3_9.json', rows)
    return rows


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--if-fit-size',type=int,default=200000)
    parser.add_argument('--lof-fit-size',type=int,default=2000)
    parser.add_argument('--resume-engineered',action='store_true',
                        help='reuse verified engineered.parquet files and rerun phases 3.5-3.9')
    parser.add_argument('--prepare-only', action='store_true',
                        help='clean/engineer the data but do not fit the sampled baseline models')
    parser.add_argument('--split-strategy', choices=('chronological', 'random'),
                        default='chronological',
                        help='row allocation; random is seeded and independent of labels')
    args=parser.parse_args()
    if args.if_fit_size<256 or args.lof_fit_size<21:
        parser.error('Fitting samples must exceed IF 256 and LOF 20')
    OUT.mkdir(exist_ok=True)
    all_rows=[]
    for name in ('BankSim','PaySim'):
        folder=OUT / name.lower()
        folder.mkdir(exist_ok=True)
        engineered=folder/'engineered.parquet'
        if args.resume_engineered and engineered.exists():
            audit=json.loads((folder/'phase_3_3.json').read_text(encoding='utf-8'))
            if int(audit['cleaned_rows']) != pq.ParquetFile(engineered).metadata.num_rows:
                raise ValueError(f'{name}: engineered rows do not match the cleaning audit')
            features=[c for c in pq.ParquetFile(engineered).schema_arrow.names
                      if c != CONFIG[name]['label']]
            print(f'{name}: reusing verified engineered features for phases 3.5-3.9')
        else:
            audit=clean(name,folder)
            finish_diagnostics(name,folder)
            features=feature_stage(name,folder,audit['cleaned_rows'])
        if not args.prepare_only:
            stage = model_stage_random if args.split_strategy == 'random' else model_stage
            all_rows.extend(stage(name,folder,features,audit['cleaned_rows'],
                                  args.if_fit_size,args.lof_fit_size))
    if args.prepare_only:
        print('Preparation complete; no sampled baseline models were fitted.', flush=True)
        return
    dump(OUT/'metrics.json',all_rows)
    pd.DataFrame([{k:v for k,v in row.items() if k not in ('validation','test')}
                  |{f'test_{k}':v for k,v in row['test'].items()}
                  for row in all_rows]).to_csv(OUT/'comparison.csv',index=False)


if __name__=='__main__':
    main()
