"""IEEE-CIS: dataset statistics and the stratified random 80/20 split (separate pass to keep memory below ~6 GB)."""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud_detection.data import read_ieee_files  # noqa: E402
from fraud_detection.features import IEEEFeatureBuilder  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).parent / "cache"
SEED = 42
day = 86400

ieee = read_ieee_files(DATA / "ieee" / "train_transaction.csv", DATA / "ieee" / "train_identity.csv")
h = pd.util.hash_pandas_object(ieee.drop(columns=["TransactionID"]), index=False)
n_dup = int(h.duplicated().sum())
del h
gc.collect()
ieee = ieee.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
cut = int(len(ieee) * 0.8)
y = ieee["isFraud"].values
info = json.load(open(OUT / "data_info.json")) if (OUT / "data_info.json").exists() else {}
info["ieee"] = dict(raw_rows=len(ieee), raw_frauds=int(y.sum()), rows_with_identity=int(ieee["id_01"].notna().sum()),
                    duplicate_rows_excluding_id=n_dup, rows=len(ieee), frauds=int(y.sum()),
                    train_rows=cut, train_frauds=int(y[:cut].sum()),
                    test_rows=len(ieee) - cut, test_frauds=int(y[cut:].sum()),
                    train_day_range=[float(ieee["TransactionDT"].iloc[0] / day), float(ieee["TransactionDT"].iloc[cut - 1] / day)],
                    test_day_range=[float(ieee["TransactionDT"].iloc[cut] / day), float(ieee["TransactionDT"].iloc[-1] / day)],
                    n_features=454)
json.dump(info, open(OUT / "data_info.json", "w"), indent=2)
print(info["ieee"], flush=True)

idx_tr, idx_te = train_test_split(np.arange(len(ieee)), test_size=0.2, stratify=y, random_state=SEED)
rtr = ieee.iloc[idx_tr].reset_index(drop=True)
rte = ieee.iloc[idx_te].reset_index(drop=True)
del ieee
gc.collect()
fb = IEEEFeatureBuilder().fit(rtr)
Xtr = fb.transform(rtr).values.astype(np.float32)
ytr = rtr["isFraud"].values.astype(np.int8)
del rtr
gc.collect()
Xte = fb.transform(rte).values.astype(np.float32)
np.savez_compressed(OUT / "ieee_random.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=rte["isFraud"].values.astype(np.int8),
                    amt_te=rte["TransactionAmt"].values)
print("saved", Xtr.shape, Xte.shape)
