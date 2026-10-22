"""IEEE-CIS: features built and selected inside each forward-chaining fold.

For fold k (k = 0..4: train on time blocks 1..k+1, validate on block k+2) the IEEEFeatureBuilder (frequency
maps, group amount statistics, category codes) is fitted on the training part of the fold only, and the
validation block is transformed with it. The 100 features with the highest LightGBM gain are then selected on
the training part only. "full": the builder is fitted on the whole training period (identical to
cache/ieee_time.npz) and the selection is made on the whole training period; it is used for the final model.

Writes cache/ieee_time_fold{k}.npz (Xtr, ytr, Xva, yva, feat: selected columns only) and
results/ieee_time/selected_features_{k|full}.npy (indices into the 454 builder features).
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
for src in (ROOT / "src", ROOT.parent / "src"):
    if src.exists():
        sys.path.insert(0, str(src))
from fraud_detection.data import read_ieee_files  # noqa: E402
from fraud_detection.features import IEEEFeatureBuilder  # noqa: E402

from common import CACHE, RES, load, select_top_features, time_folds  # noqa: E402

DATA = next(p for p in (ROOT.parent / "data" / "ieee", ROOT / "data") if (p / "train_transaction.csv").exists())
out_dir = RES / "ieee_time"
out_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()

ieee = read_ieee_files(DATA / "train_transaction.csv", DATA / "train_identity.csv")
ieee = ieee.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
tr = ieee.iloc[: int(len(ieee) * 0.8)].reset_index(drop=True)
del ieee
gc.collect()
y = tr["isFraud"].values.astype(np.int8)
cache = load("ieee_time")
assert np.array_equal(cache["ytr"], y), "training rows differ from cache/ieee_time.npz"
feat_all = list(cache["feat"])

info = {}
# final model: builder on the whole training period = the cached features; selection on the whole period
keep_full = select_top_features(cache["Xtr"], y)
np.save(out_dir / "selected_features_full.npy", keep_full)
info["full"] = dict(train_rows=int(len(y)), features=[feat_all[i] for i in keep_full])
del cache
gc.collect()
print("full selection done", round(time.time() - t0), "s", flush=True)

for k, (itr, iva) in enumerate(time_folds(len(tr))):
    fb = IEEEFeatureBuilder().fit(tr.iloc[itr])
    Xa = fb.transform(tr.iloc[itr])
    assert list(Xa.columns) == feat_all, "feature columns differ between folds"
    Xa = Xa.values.astype(np.float32)
    Xv = fb.transform(tr.iloc[iva]).values.astype(np.float32)
    keep = select_top_features(Xa, y[itr])
    np.save(out_dir / f"selected_features_{k}.npy", keep)
    np.savez_compressed(CACHE / f"ieee_time_fold{k}.npz", Xtr=Xa[:, keep], ytr=y[itr], Xva=Xv[:, keep], yva=y[iva],
                        feat=np.array([feat_all[i] for i in keep]))
    info[str(k)] = dict(train_rows=int(len(itr)), val_rows=int(len(iva)), features=[feat_all[i] for i in keep],
                        overlap_with_full=int(len(set(keep) & set(keep_full))))
    print(f"fold {k}: train {len(itr)}, val {len(iva)}, overlap with full selection {info[str(k)]['overlap_with_full']}/100,"
          f" {round(time.time() - t0)} s", flush=True)
    del fb, Xa, Xv
    gc.collect()

json.dump(info, open(out_dir / "selected_features.json", "w"), indent=1)
print("done", round(time.time() - t0), "s")
