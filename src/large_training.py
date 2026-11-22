"""Train all anomaly detectors using every row in the 70% training partition.

Uses the exact cleaned/engineered data, split boundaries and feature sets from
rebuild_project.py. The old baseline and enhancement experiments remain intact.

LOF uses FAISS approximate nearest-neighbour lookup, but its index and density
estimates contain every training-partition row. The coarse IVF quantizer is
trained on a sample only; that sample does not limit the LOF reference data.
"""
from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import faiss
from sklearn import set_config
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .rebuild_project import ROOT, OUT, CONFIG, dump, threshold_f1, evaluate

DEST = ROOT / 'larger_training_output'
set_config(working_memory=64)


class ApproximateFullLOF:
    """Approximate LOF whose FAISS index contains every training-partition row."""
    def __init__(self, n_neighbors: int = 20, nprobe: int = 1):
        self.n_neighbors, self.nprobe = n_neighbors, nprobe

    def _search(self, values: np.ndarray, k: int):
        query = np.ascontiguousarray(values, dtype=np.float32)
        probe = self.index.nprobe
        while True:
            distances, indexes = self.index.search(query, k)
            if not (indexes < 0).any():
                break
            if probe >= self.index.nlist:
                raise RuntimeError("FAISS returned insufficient neighbours")
            probe = min(self.index.nlist, probe * 2)
            self.index.nprobe = probe
        return np.sqrt(np.maximum(distances, 0)), indexes

    def fit(self, values: np.ndarray):
        values = np.ascontiguousarray(values, dtype=np.float32)
        count, dimensions = values.shape
        if count <= self.n_neighbors:
            raise ValueError("LOF requires more rows than neighbours")
        # More inverted lists keep each all-row lookup computationally bounded.
        nlist = min(4096, max(256, int(np.sqrt(count) * 2)))
        self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimensions), dimensions,
                                        nlist, faiss.METRIC_L2)
        sample_size = min(200_000, count)
        sample_ids = np.random.default_rng(42).choice(count, sample_size, replace=False)
        self.index.train(np.ascontiguousarray(values[sample_ids]))
        for start in range(0, count, 50_000):
            self.index.add(np.ascontiguousarray(values[start:start+50_000]))
        self.index.nprobe = min(self.nprobe, nlist)
        self.k_distance = np.empty(count, dtype=np.float32)
        for start in range(0, count, 25_000):
            end = min(start+25_000, count)
            distance, indexes = self._search(values[start:end], self.n_neighbors+1)
            keep = indexes != np.arange(start, end)[:, None]
            self.k_distance[start:end] = distance[keep].reshape(end-start, -1)[:, :self.n_neighbors][:, -1]
        self.lrd = np.empty(count, dtype=np.float32)
        for start in range(0, count, 25_000):
            end = min(start+25_000, count)
            distance, indexes = self._search(values[start:end], self.n_neighbors+1)
            keep = indexes != np.arange(start, end)[:, None]
            ids = indexes[keep].reshape(end-start, -1)[:, :self.n_neighbors]
            dist = distance[keep].reshape(end-start, -1)[:, :self.n_neighbors]
            reachability = np.maximum(dist, self.k_distance[ids])
            self.lrd[start:end] = 1.0 / np.maximum(reachability.mean(axis=1), 1e-12)
        self.n_features_in_, self.training_rows_ = dimensions, count
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        distance, ids = self._search(values, self.n_neighbors)
        reachability = np.maximum(distance, self.k_distance[ids])
        local_density = 1.0 / np.maximum(reachability.mean(axis=1), 1e-12)
        return -(self.lrd[ids].mean(axis=1) / local_density)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["index_bytes"] = faiss.serialize_index(state.pop("index"))
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.index = faiss.deserialize_index(self.__dict__.pop("index_bytes"))


def train_scaler(source: Path, features: list[str], end: int) -> StandardScaler:
    scaler=StandardScaler()
    offset=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=50000,columns=features):
        count=min(len(batch),end-offset)
        if count<=0:
            break
        frame=batch.to_pandas().iloc[:count][features]
        scaler.partial_fit(frame.to_numpy(dtype=np.float32))
        offset+=len(batch)
    assert offset>=end and int(scaler.n_samples_seen_)==end
    return scaler


def build_training_memmap(source: Path, features: list[str], scaler: StandardScaler,
                          end: int, path: Path) -> np.memmap:
    matrix=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,
                                      shape=(end,len(features)))
    offset=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=25000,columns=features):
        count=min(len(batch),end-offset)
        if count<=0:
            break
        x=batch.to_pandas().iloc[:count][features].to_numpy(dtype=np.float32)
        matrix[offset:offset+count] = scaler.transform(x)
        offset+=count
    assert offset==end
    matrix.flush()
    return matrix


def partition_scores(source: Path, features: list[str], scaler: StandardScaler,
                     model, start_row: int, end_row: int, batch_size: int) -> np.ndarray:
    result=np.empty(end_row-start_row,dtype=np.float32)
    cursor=0
    offset=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size,columns=features):
        end=offset+len(batch)
        a,b=max(offset,start_row),min(end,end_row)
        if b>a:
            x=batch.to_pandas().iloc[a-offset:b-offset][features].to_numpy(dtype=np.float32)
            score=-model.score_samples(scaler.transform(x))
            result[cursor:cursor+len(score)]=score
            cursor+=len(score)
        offset=end
        if offset>=end_row:
            break
    assert cursor==len(result)
    return result


def partition_labels(source: Path,label: str,a: int,b: int) -> np.ndarray:
    # Reading only the 1-byte target column keeps memory use low.
    result=np.empty(b-a,dtype=np.uint8)
    offset=cursor=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=50000,columns=[label]):
        end=offset+len(batch)
        lo,hi=max(offset,a),min(end,b)
        if hi>lo:
            part=batch.column(0).to_numpy()[lo-offset:hi-offset]
            result[cursor:cursor+len(part)]=part
            cursor+=len(part)
        offset=end
        if offset>=b:
            break
    assert cursor==len(result)
    return result


def run_one(name: str, feature_set: str, detectors: tuple[str, ...]) -> list[dict]:
    source=OUT/name.lower()/'engineered.parquet'
    n=pq.ParquetFile(source).metadata.num_rows
    train_end,val_end=int(.7*n),int(.85*n)
    folder=DEST/name.lower()
    folder.mkdir(parents=True,exist_ok=True)
    audit=json.loads((OUT/name.lower()/f'phase_3_5_{feature_set}.json').read_text())
    features=audit['features']
    y_val=partition_labels(source,CONFIG[name]['label'],train_end,val_end)
    y_test=partition_labels(source,CONFIG[name]['label'],val_end,n)
    start=time.perf_counter()
    scaler=train_scaler(source,features,train_end)
    with tempfile.TemporaryDirectory(dir=folder,prefix='training_matrix_') as temp:
        matrix=build_training_memmap(source,features,scaler,train_end,
                                    Path(temp)/'matrix.npy')
        print(name,feature_set,'built training matrix',matrix.shape,flush=True)
        fitted={}
        for detector in detectors:
            fit_start=time.perf_counter()
            if detector=='IsolationForest':
                # Every row in the 70% partition is available to every tree.
                model=IsolationForest(n_estimators=100,max_samples='auto',
                                      random_state=42,n_jobs=1).fit(matrix)
                used=train_end
            else:
                # Every row enters the approximate-neighbour index and density estimate.
                model=ApproximateFullLOF(n_neighbors=20).fit(matrix)
                used=train_end
            fitted[detector]=(model,used,time.perf_counter()-fit_start)
            print(name,feature_set,detector,'fitted',used,'rows',flush=True)
        del matrix
        gc.collect()
        results=[]
        for detector,(model,used,fit_seconds) in fitted.items():
            scoring_start=time.perf_counter()
            batch=5000 if detector=='IsolationForest' else 10_000
            # A model's mathematical convention can make lower values mean more
            # anomalous. Use validation labels (never test labels) to establish a
            # consistent high-score-means-more-suspicious direction.
            val=partition_scores(source,features,scaler,model,train_end,val_end,batch)
            score_multiplier = -1.0 if roc_auc_score(y_val, val) < 0.5 else 1.0
            val *= score_multiplier
            cutoff,_=threshold_f1(y_val,val)
            validation=evaluate(y_val,val,cutoff)
            validation_seconds=time.perf_counter()-scoring_start
            test_start=time.perf_counter()
            test=partition_scores(source,features,scaler,model,val_end,n,batch)
            test *= score_multiplier
            test_seconds=time.perf_counter()-test_start
            result={"dataset":name,"model":detector,"feature_set":feature_set,
                    "train_rows":train_end,"fit_rows":used,"lof_reference_rows":used if detector=='LOF' else None,
                    "training_protocol":"all rows in the chronological 70% partition",
                    "lof_approximation":"FAISS IVF approximate neighbours over all training rows" if detector=='LOF' else None,
                    "score_multiplier":score_multiplier,
                    "n_features":len(features),"threshold":cutoff,
                    "validation":validation,"test":evaluate(y_test,test,cutoff),
                    "preprocessing_seconds":time.perf_counter()-start-fit_seconds-validation_seconds-test_seconds,
                    "fit_seconds":fit_seconds,"validation_seconds":validation_seconds,
                    "test_seconds":test_seconds}
            results.append(result)
            stem=f'{detector}_{feature_set}'
            joblib.dump({"model":model,"scaler":scaler,"features":features,
                         "threshold":cutoff,"dataset":name,
                         "score_multiplier":score_multiplier},folder/f'{stem}.joblib')
            pq.write_table(pa.table({"source_row":np.arange(val_end,n),
                                     "anomaly_score":test,
                                     "anomaly_flag":(test>=cutoff).astype(np.uint8),
                                     "reference_fraud":y_test}),
                           folder/f'{stem}_test_predictions.parquet',compression='zstd')
            print(name,stem,'test F1',round(result['test']['f1'],4),
                  'AP',round(result['test']['pr_auc_average_precision'],4),flush=True)
            dump(folder/f'{stem}_metrics.json',result)
        return results


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['BankSim','PaySim','both'],default='both')
    parser.add_argument('--feature-set',choices=['reduced','full','both'],default='both')
    parser.add_argument('--model', choices=['all', 'IsolationForest', 'LOF'], default='all',
                        help='limit a rerun to one detector; all is the final default')
    args=parser.parse_args()
    DEST.mkdir(exist_ok=True)
    all_rows=[]
    detectors = ('IsolationForest', 'LOF') if args.model == 'all' else (args.model,)
    for name in (['BankSim','PaySim'] if args.dataset=='both' else [args.dataset]):
        for set_name in (['reduced','full'] if args.feature_set=='both' else [args.feature_set]):
            completed=run_one(name,set_name,detectors)
            all_rows.extend(completed)
            # Keep each completed phase if a later, larger model exhausts memory.
            existing=json.loads((DEST/'metrics.json').read_text()) if (DEST/'metrics.json').exists() else []
            existing=[r for r in existing if not (r['dataset']==name and r['feature_set']==set_name
                                                    and r['model'] in detectors)]
            dump(DEST/'metrics.json',existing+completed)
    rows=json.loads((DEST/'metrics.json').read_text())
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('validation','test')}
                  |{f'test_{k}':v for k,v in r['test'].items()}
                  for r in rows]).to_csv(DEST/'comparison.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
