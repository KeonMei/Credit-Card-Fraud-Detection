"""Shared pieces: data loading, folds, balancing, classifiers, search spaces, metrics."""
import time
from pathlib import Path

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, fbeta_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 42
N_JOBS = 2
CLASSIFIERS = ["LR", "RF", "XGB", "LGBM", "MLP"]
BALANCING = ["none", "RUS", "SMOTE", "SMOTE-ENN"]
BETA = 2


def load(name):
    d = np.load(CACHE / f"{name}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def time_folds(n, n_blocks=6):
    """Forward-chaining folds over n time-ordered rows: train blocks 1..k, validate block k+1."""
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    return [(np.arange(0, edges[k]), np.arange(edges[k], edges[k + 1])) for k in range(1, n_blocks)]


# ------------------------------------------------------------------ preprocessing
class Prep:
    """Median imputation + standardisation, fitted on the training part only."""

    def fit(self, X):
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        self.sc = StandardScaler().fit(self.imp.transform(X))
        return self

    def transform(self, X):
        Z = self.sc.transform(self.imp.transform(X)).astype(np.float32)
        return np.clip(Z, -50, 50)


# ------------------------------------------------------------------ balancing
def balance(X, y, method, ratio, seed=SEED):
    """Applied to the training part of a fold only. ratio = minority/majority after resampling."""
    if method == "none":
        return X, y
    if method == "RUS":
        return RandomUnderSampler(sampling_strategy=ratio, random_state=seed).fit_resample(X, y)
    if method == "SMOTE":
        return SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=seed).fit_resample(X, y)
    if method == "SMOTE-ENN":
        return SMOTEENN(smote=SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=seed),
                        enn=EditedNearestNeighbours(n_neighbors=3, n_jobs=N_JOBS),
                        random_state=seed).fit_resample(X, y)
    raise ValueError(method)


# ------------------------------------------------------------------ classifiers
def sample_params(clf, rng):
    """Search ranges from Methods (Tuning)."""
    if clf == "LR":
        return dict(C=float(loguniform(1e-4, 1e2).rvs(random_state=rng)))
    if clf == "RF":
        return dict(n_estimators=int(randint(100, 501).rvs(random_state=rng)),
                    max_depth=[None, *range(5, 31)][int(rng.integers(0, 27))],
                    min_samples_leaf=int(randint(1, 21).rvs(random_state=rng)),
                    max_features=["sqrt", "log2"][int(rng.integers(0, 2))])
    if clf == "XGB":
        return dict(n_estimators=int(randint(100, 1001).rvs(random_state=rng)),
                    learning_rate=float(loguniform(0.01, 0.3).rvs(random_state=rng)),
                    max_depth=int(randint(3, 11).rvs(random_state=rng)),
                    subsample=float(uniform(0.6, 0.4).rvs(random_state=rng)),
                    colsample_bytree=float(uniform(0.6, 0.4).rvs(random_state=rng)))
    if clf == "LGBM":
        return dict(n_estimators=int(randint(100, 1001).rvs(random_state=rng)),
                    learning_rate=float(loguniform(0.01, 0.3).rvs(random_state=rng)),
                    num_leaves=int(randint(15, 256).rvs(random_state=rng)),
                    min_child_samples=int(randint(5, 101).rvs(random_state=rng)))
    if clf == "MLP":
        return dict(n_layers=int(rng.integers(1, 4)), units=[32, 64, 128, 256][int(rng.integers(0, 4))],
                    lr=float(loguniform(1e-4, 1e-2).rvs(random_state=rng)),
                    dropout=float(uniform(0, 0.5).rvs(random_state=rng)))
    raise ValueError(clf)


def make_model(clf, p, seed=SEED):
    if clf == "LR":
        return LogisticRegression(C=p["C"], max_iter=300, solver="lbfgs")
    if clf == "RF":
        return RandomForestClassifier(n_jobs=N_JOBS, random_state=seed, **p)
    if clf == "XGB":
        return XGBClassifier(tree_method="hist", n_jobs=N_JOBS, random_state=seed, eval_metric="aucpr",
                             max_bin=128, **p)
    if clf == "LGBM":
        return LGBMClassifier(n_jobs=N_JOBS, random_state=seed, verbose=-1, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, **p)
    if clf == "MLP":
        return TorchMLP(seed=seed, **p)
    raise ValueError(clf)


class TorchMLP:
    """Feed-forward network, ReLU hidden layers, sigmoid output, BCE loss, Adam,
    early stopping on a random 10% of the training part of the fold (patience 10, max 100 epochs)."""

    def __init__(self, n_layers, units, lr, dropout, seed=SEED, batch=1024, max_epochs=100, patience=10):
        self.n_layers, self.units, self.lr, self.dropout = n_layers, units, lr, dropout
        self.seed, self.batch, self.max_epochs, self.patience = seed, batch, max_epochs, patience

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        torch.set_num_threads(N_JOBS)
        rng = np.random.default_rng(self.seed)
        n_val = max(int(0.1 * len(y)), 1)
        perm = rng.permutation(len(y))
        vi, ti = perm[:n_val], perm[n_val:]
        layers, d = [], X.shape[1]
        for _ in range(self.n_layers):
            layers += [nn.Linear(d, self.units), nn.ReLU(), nn.Dropout(self.dropout)]
            d = self.units
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.BCEWithLogitsLoss()
        Xt, yt = torch.from_numpy(X[ti]), torch.from_numpy(y[ti].astype(np.float32))
        Xv, yv = X[vi], y[vi]
        best, best_state, bad = -1.0, None, 0
        for ep in range(self.max_epochs):
            self.net.train()
            order = torch.randperm(len(ti))
            for s in range(0, len(ti), self.batch):
                b = order[s:s + self.batch]
                opt.zero_grad()
                loss = lossf(self.net(Xt[b]).squeeze(1), yt[b])
                loss.backward()
                opt.step()
            pv = self.predict_proba(Xv)[:, 1]
            score = average_precision_score(yv, pv) if yv.sum() > 0 else -loss.item()
            if score > best + 1e-5:
                best, bad = score, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.epochs_ = ep + 1
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 65536):
                out.append(torch.sigmoid(self.net(torch.from_numpy(X[s:s + 65536]))).squeeze(1).numpy())
        p = np.concatenate(out)
        return np.c_[1 - p, p]


# ------------------------------------------------------------------ metrics
def threshold_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat),
                f1=fbeta_score(y, yhat, beta=1, zero_division=0), f2=fbeta_score(y, yhat, beta=2, zero_division=0),
                mcc=matthews_corrcoef(y, yhat), fpr=fp / (fp + tn), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def all_metrics(y, p, thr=0.5):
    m = dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p), threshold=thr)
    m.update(threshold_metrics(y, p, thr))
    return m


def best_f2_threshold(y, p):
    grid = np.round(np.arange(0.001, 1.0, 0.001), 3)
    # vectorised F2 over the grid
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    tp_cum = np.cumsum(ys)
    P = ys.sum()
    best_t, best_f = 0.5, -1
    idx = np.searchsorted(-ps, -grid, side="right")  # number of scores >= t
    for t, k in zip(grid, idx):
        if k == 0:
            continue
        tp = tp_cum[k - 1]
        prec, rec = tp / k, tp / P
        f = 0 if tp == 0 else 5 * prec * rec / (4 * prec + rec)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t
