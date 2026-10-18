"""Data preparation for both datasets (Methods: Data Sources, Preparation and Splitting).

ULB : drop exact duplicate rows, sort by Time, first 80% -> train, last 20% -> test.
      Features: V1..V28, log(1+Amount), hour of day as sin/cos (raw Time dropped).
IEEE: join train_transaction + train_identity on TransactionID, sort by TransactionDT,
      first 80% -> train, last 20% -> test. Features: the leakage-free
      IEEEFeatureBuilder of the project (fit on the training period only).
Also stores a stratified random 80/20 split of ULB (IEEE: prep_ieee_random.py) for the sensitivity check.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud_detection.data import read_ieee_files  # noqa: E402
from fraud_detection.features import IEEEFeatureBuilder  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True)
SEED = 42
info = {}


def ulb_features(df):
    X = df[[f"V{i}" for i in range(1, 29)]].copy()
    X["amount_log"] = np.log1p(df["Amount"])
    h = (df["Time"] / 3600.0) % 24
    X["hour_sin"] = np.sin(2 * np.pi * h / 24)
    X["hour_cos"] = np.cos(2 * np.pi * h / 24)
    return X.astype(np.float32)


def save(name, Xtr, ytr, Xte, yte, extra):
    np.savez_compressed(OUT / f"{name}.npz", Xtr=Xtr.values.astype(np.float32), ytr=ytr.astype(np.int8),
                        Xte=Xte.values.astype(np.float32), yte=yte.astype(np.int8),
                        feat=np.array(list(Xtr.columns)), **extra)


# ---------------------------------------------------------------- ULB
t = time.time()
ulb = pd.read_csv(DATA / "ulb" / "creditcard.csv")
n_raw, f_raw = len(ulb), int(ulb["Class"].sum())
dups = ulb.duplicated()
n_dup, f_dup = int(dups.sum()), int(ulb.loc[dups, "Class"].sum())
ulb = ulb.loc[~dups].sort_values("Time", kind="stable").reset_index(drop=True)
cut = int(len(ulb) * 0.8)
tr, te = ulb.iloc[:cut], ulb.iloc[cut:]
save("ulb_time", ulb_features(tr), tr["Class"].values, ulb_features(te), te["Class"].values,
     dict(t_tr=tr["Time"].values, t_te=te["Time"].values, amt_te=te["Amount"].values, amt_tr=tr["Amount"].values))
info["ulb"] = dict(raw_rows=n_raw, raw_frauds=f_raw, duplicates_removed=n_dup, duplicate_frauds=f_dup,
                   rows=len(ulb), frauds=int(ulb["Class"].sum()),
                   train_rows=len(tr), train_frauds=int(tr["Class"].sum()),
                   test_rows=len(te), test_frauds=int(te["Class"].sum()),
                   train_time_range_h=[float(tr["Time"].min() / 3600), float(tr["Time"].max() / 3600)],
                   test_time_range_h=[float(te["Time"].min() / 3600), float(te["Time"].max() / 3600)],
                   n_features=ulb_features(tr.head(1)).shape[1])
rtr, rte = train_test_split(ulb, test_size=0.2, stratify=ulb["Class"], random_state=SEED)
save("ulb_random", ulb_features(rtr), rtr["Class"].values, ulb_features(rte), rte["Class"].values,
     dict(amt_te=rte["Amount"].values))
print("ULB done", round(time.time() - t, 1), "s", info["ulb"], flush=True)

# ---------------------------------------------------------------- IEEE
t = time.time()
ieee = read_ieee_files(DATA / "ieee" / "train_transaction.csv", DATA / "ieee" / "train_identity.csv")
n_raw, f_raw = len(ieee), int(ieee["isFraud"].sum())
n_id = int(ieee["id_01"].notna().sum())
ieee = ieee.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
cut = int(len(ieee) * 0.8)
tr, te = ieee.iloc[:cut], ieee.iloc[cut:]
fb = IEEEFeatureBuilder().fit(tr)
Xtr, Xte = fb.transform(tr), fb.transform(te)
save("ieee_time", Xtr, tr["isFraud"].values, Xte, te["isFraud"].values,
     dict(t_tr=tr["TransactionDT"].values, t_te=te["TransactionDT"].values,
          amt_te=te["TransactionAmt"].values, amt_tr=tr["TransactionAmt"].values,
          prod_te=te["ProductCD"].astype(str).values))
day = 86400
info["ieee"] = dict(raw_rows=n_raw, raw_frauds=f_raw, rows_with_identity=n_id, rows=len(ieee),
                    frauds=int(ieee["isFraud"].sum()),
                    train_rows=len(tr), train_frauds=int(tr["isFraud"].sum()),
                    test_rows=len(te), test_frauds=int(te["isFraud"].sum()),
                    train_day_range=[float(tr["TransactionDT"].min() / day), float(tr["TransactionDT"].max() / day)],
                    test_day_range=[float(te["TransactionDT"].min() / day), float(te["TransactionDT"].max() / day)],
                    n_features=Xtr.shape[1])
del Xtr, Xte
json.dump(info, open(OUT / "data_info.json", "w"), indent=2)
