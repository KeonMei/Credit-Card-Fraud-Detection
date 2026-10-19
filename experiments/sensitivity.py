"""Additional analyses (robustness / leakage), using the hyperparameters chosen on the time split.

1. Stratified random 80/20 split instead of the time split (best configuration and the LR baseline).
2. "Leaky" protocol seen in several reviewed papers: SMOTE (ratio 1.0) applied to the whole
   dataset before the random split, so synthetic frauds are present in the test set.
Usage: python sensitivity.py <dataset prefix: ulb|ieee>
"""
import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from common import RES, SEED, Prep, Timer, all_metrics, balance, load, make_model

warnings.filterwarnings("ignore")
ds = sys.argv[1]
MAIN = {"ulb": "ulb_time", "ieee": "ieee_time", "sparkov": "sparkov_card"}[ds]
MAIN_LABEL = "card split (main)" if ds == "sparkov" else "time split (main)"
EXTRA = [("stratified random split", f"{ds}_random")]
if ds == "sparkov":
    EXTRA = [("date split (last 6 months)", "sparkov_date")] + EXTRA
tdir = RES / "tables"
best = pd.read_csv(tdir / f"{MAIN}_main.csv").iloc[0]
keep_f = RES / MAIN / "selected_features.npy"
keep = np.load(keep_f) if keep_f.exists() else None
configs = [(best["classifier"], best["balancing"]), ("LR", "none")]
rows = []
for clf, bal in configs:
    r = pickle.load(open(RES / MAIN / f"{clf}__{bal}.pkl", "rb"))
    # (a) main split, numbers from the main run at threshold 0.5
    d = load(MAIN)
    m = all_metrics(d["yte"].astype(int), r["test_pred"], 0.5)
    rows.append(dict(config=f"{clf}+{bal}", protocol=MAIN_LABEL, **m))
    # (b) other splits, same hyperparameters
    for label, cache_name in EXTRA:
        d = load(cache_name)
        X, y, Xt, yt = d["Xtr"], d["ytr"].astype(int), d["Xte"], d["yte"].astype(int)
        if keep is not None:
            X, Xt = X[:, keep], Xt[:, keep]
        pp = Prep().fit(X)
        Xa, ya = balance(pp.transform(X), y, bal, r["ratio"]) if bal != "none" else (pp.transform(X), y)
        with Timer() as t:
            mdl = make_model(clf, r["params"]).fit(Xa, ya)
        p = mdl.predict_proba(pp.transform(Xt))[:, 1]
        rows.append(dict(config=f"{clf}+{bal}", protocol=label, **all_metrics(yt, p, 0.5)))
        print(rows[-1], flush=True)

# (c) leaky protocol: SMOTE on all data, then random split (best classifier, no further balancing)
clf = best["classifier"]
r = pickle.load(open(RES / MAIN / f"{clf}__{best['balancing']}.pkl", "rb"))
d = load(f"{ds}_random")
Xall = np.vstack([d["Xtr"], d["Xte"]])
yall = np.concatenate([d["ytr"], d["yte"]]).astype(int)
if keep is not None:
    Xall = Xall[:, keep]
pp = Prep().fit(Xall)  # also leaky: statistics of the test rows are used
Xs, ys = SMOTE(sampling_strategy=1.0, random_state=SEED).fit_resample(pp.transform(Xall), yall)
Xa, Xt, ya, yt = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=SEED)
mdl = make_model(clf, r["params"]).fit(Xa, ya)
p = mdl.predict_proba(Xt)[:, 1]
rows.append(dict(config=f"{clf}", protocol="SMOTE before split (leaky)", **all_metrics(yt, p, 0.5),
                 n_test=len(yt), n_test_fraud=int(yt.sum())))
out = pd.DataFrame(rows)
out.insert(0, "dataset", ds)
out.to_csv(tdir / f"{ds}_sensitivity.csv", index=False)
print(out.round(4).to_string())
