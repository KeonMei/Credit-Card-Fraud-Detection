"""Baseline grid: 5 classifiers x 4 balancing settings (Methods: Compared Methods, Tuning).

For every configuration:
  1. random search (n_trials) over the Methods ranges; the balancing ratio
     (minority/majority after resampling) is searched in {0.1, 0.5, 1.0};
     each trial is scored by mean PR-AUC on the tuning folds (last forward-chaining folds);
  2. the chosen setting is re-run on all 5 forward-chaining folds -> out-of-fold (OOF)
     predictions for blocks 2..6 and per-fold PR-AUC (mean, std);
  3. refit on the full training period and score the untouched test period.
Balancing is applied only to the training part of each fold.
Usage: python run_grid.py <dataset> <n_trials> <tune_folds> [classifiers] [balancings]
"""
import json
import pickle
import sys
import warnings

import numpy as np

from common import (BALANCING, CLASSIFIERS, RES, SEED, Prep, Timer, average_precision_score, balance, load,
                    make_model, sample_params, time_folds)

warnings.filterwarnings("ignore")
name = sys.argv[1]
# per-classifier number of random-search trials, e.g. "LR:10,RF:5,XGB:10,LGBM:10,MLP:6"
TRIALS = {k: int(v) for k, v in (kv.split(":") for kv in sys.argv[2].split(","))}
n_tune = int(sys.argv[3])
clfs = sys.argv[4].split(",") if len(sys.argv) > 4 else CLASSIFIERS
bals = sys.argv[5].split(",") if len(sys.argv) > 5 else BALANCING
RATIOS = [0.1, 0.5, 1.0]
# ENN is O(n^2): on IEEE-CIS the SMOTE-ENN ratio is fixed to 0.1 (compute budget, see deviations)
ENN_RATIOS = [0.1] if name.startswith("ieee") else RATIOS
out_dir = RES / name
out_dir.mkdir(parents=True, exist_ok=True)

d = load(name)
X, y = d["Xtr"], d["ytr"].astype(int)
Xte, yte = d["Xte"], d["yte"].astype(int)
sel = out_dir / "selected_features.npy"
if sel.exists():
    keep = np.load(sel)
    X, Xte = X[:, keep], Xte[:, keep]
folds = time_folds(len(y))
tune_ids = list(range(5 - n_tune, 5))

# preprocessed folds (imputer + scaler fitted on the training part of each fold)
prep_cache = {}


def fold_data(k):
    if k not in prep_cache:
        if k == "full":
            pp = Prep().fit(X)
            prep_cache[k] = (pp.transform(X), y, pp.transform(Xte), yte)
        else:
            tr, va = folds[k]
            pp = Prep().fit(X[tr])
            prep_cache[k] = (pp.transform(X[tr]), y[tr], pp.transform(X[va]), y[va])
    return prep_cache[k]


bal_cache = {}


def balanced(k, method, ratio):
    key = (k, method, ratio)
    if key not in bal_cache:
        Xa, ya, _, _ = fold_data(k)
        with Timer() as t:
            bal_cache[key] = balance(Xa, ya, method, ratio)
        bal_cache[key] = (*bal_cache[key], t.s)
        # keep memory bounded: only the most recent few resampled sets
        if len(bal_cache) > 8:
            bal_cache.pop(next(iter(bal_cache)))
    return bal_cache[key]


def run_fold(k, clf, params, method, ratio, seed=SEED):
    Xa, ya, Xb, yb = fold_data(k)
    if method != "none":
        Xa, ya, bal_s = balanced(k, method, ratio)
    else:
        bal_s = 0.0
    with Timer() as t:
        mdl = make_model(clf, params, seed).fit(Xa, ya)
    p = mdl.predict_proba(Xb)[:, 1]
    return p, t.s + bal_s, len(ya), int(ya.sum())


for method in bals:
    for clf in clfs:
        tag = f"{clf}__{method}"
        if (out_dir / f"{tag}.pkl").exists():
            print("skip", tag, flush=True)
            continue
        rng = np.random.default_rng(SEED)
        trials = []
        for i in range(TRIALS[clf]):
            params = sample_params(clf, rng)
            rset = ENN_RATIOS if method == "SMOTE-ENN" else RATIOS
            ratio = rset[int(rng.integers(0, len(rset)))] if method != "none" else None
            scores, preds, secs = [], {}, 0.0
            for k in tune_ids:
                p, s, _, _ = run_fold(k, clf, params, method, ratio)
                scores.append(average_precision_score(fold_data(k)[3], p))
                preds[k] = p
                secs += s
            trials.append(dict(params=params, ratio=ratio, score=float(np.mean(scores)), preds=preds, secs=secs))
            print(f"{tag} trial {i} {np.mean(scores):.4f} {secs:.0f}s {params} r={ratio}", flush=True)
        best = max(trials, key=lambda t: t["score"])
        oof = np.full(len(y), np.nan, dtype=np.float32)
        fold_scores, fold_secs = [], []
        for k in range(5):
            if k in best["preds"]:
                p = best["preds"][k]
                s = np.nan
            else:
                p, s, _, _ = run_fold(k, clf, best["params"], method, best["ratio"])
            oof[folds[k][1]] = p
            fold_scores.append(float(average_precision_score(y[folds[k][1]], p)))
            fold_secs.append(s)
        p_test, fit_s, n_fit, n_pos = run_fold("full", clf, best["params"], method, best["ratio"])
        res = dict(dataset=name, clf=clf, balancing=method, params=best["params"], ratio=best["ratio"],
                   tune_score=best["score"], fold_pr_auc=fold_scores, cv_mean=float(np.mean(fold_scores)),
                   cv_std=float(np.std(fold_scores, ddof=1)), oof=oof, test_pred=p_test.astype(np.float32),
                   fit_seconds_full=fit_s, n_fit=n_fit, n_pos_fit=n_pos,
                   test_pr_auc=float(average_precision_score(yte, p_test)),
                   search_seconds=float(sum(t["secs"] for t in trials)),
                   trials=[{k: v for k, v in t.items() if k != "preds"} for t in trials])
        pickle.dump(res, open(out_dir / f"{tag}.pkl", "wb"))
        print(f"DONE {tag} cv={res['cv_mean']:.4f}+-{res['cv_std']:.4f} test={res['test_pr_auc']:.4f} "
              f"fit={fit_s:.0f}s", flush=True)
