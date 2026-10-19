"""Sparkov sensitivity splits (date, random): the training part is subsampled (stratified, seed 42) to the
same number of rows as the card-split training set, so that the protocols differ only in how the data are split."""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from common import CACHE, SEED
n_target = json.load(open(CACHE / "data_info.json"))["sparkov"]["train_rows"]
for name in ("sparkov_date", "sparkov_random"):
    d = dict(np.load(CACHE / f"{name}.npz", allow_pickle=True))
    if len(d["ytr"]) <= n_target:
        print(name, "already subsampled", len(d["ytr"])); continue
    keep, _ = train_test_split(np.arange(len(d["ytr"])), train_size=n_target, stratify=d["ytr"], random_state=SEED)
    keep = np.sort(keep)
    d["Xtr"], d["ytr"] = d["Xtr"][keep], d["ytr"][keep]
    np.savez_compressed(CACHE / f"{name}.npz", **d)
    print(name, len(d["ytr"]), int(d["ytr"].sum()), "test", len(d["yte"]), int(d["yte"].sum()))
