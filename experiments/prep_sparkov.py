"""Sparkov (simulated) data: features and card-based split (Methods: Second Dataset, Splitting).

Both Kaggle files (fraudTrain.csv, fraudTest.csv) are joined into one table of 1.85 M transactions.
Features: log amount, merchant category (one-hot), hour and day of week (sin/cos), customer age,
gender, customer-merchant distance, and per-card number of transactions and mean amount in the
previous 24 hours (only earlier transactions of the same card). Names, addresses, transaction ids
and the card number itself are not used as features; the card number is only used for grouping.

Main split: 20% of the cards (with all their transactions) form the test set; the cards are drawn
separately among cards with and without fraud, so the fraud rate stays similar. Because of the compute
budget, training uses a random 50% of the remaining cards. 5-fold cross-validation
on the training cards is grouped by card (StratifiedGroupKFold). Extra splits for the sensitivity check:
by date (last six months = test) and stratified random by transaction.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

DATA = Path(__file__).resolve().parents[1] / "data" / "sparkov"
OUT = Path(__file__).parent / "cache"
SEED = 42

df = pd.concat([pd.read_csv(DATA / "fraudTrain.csv", index_col=0), pd.read_csv(DATA / "fraudTest.csv", index_col=0)],
               ignore_index=True)
n_raw, f_raw = len(df), int(df["is_fraud"].sum())
n_dup = int(df.duplicated(subset=["trans_num"]).sum())
df = df.drop_duplicates(subset=["trans_num"])
df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
df = df.sort_values(["ts", "trans_num"], kind="stable").reset_index(drop=True)

# ---------------------------------------------------------------- features
X = pd.DataFrame(index=df.index)
X["amount_log"] = np.log1p(df["amt"])
h = df["ts"].dt.hour + df["ts"].dt.minute / 60
X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
dw = df["ts"].dt.dayofweek
X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)
X["age"] = (df["ts"] - pd.to_datetime(df["dob"])).dt.days / 365.25
X["gender_m"] = (df["gender"] == "M").astype(np.float32)
lat1, lon1, lat2, lon2 = map(np.radians, [df["lat"], df["long"], df["merch_lat"], df["merch_long"]])
a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
X["distance_km"] = 2 * 6371 * np.arcsin(np.sqrt(a))
# per-card activity in the previous 24 hours, strictly before the current transaction
g = df[["cc_num", "ts", "amt"]].copy()
g = g.sort_values(["cc_num", "ts"], kind="stable")
roll = g.groupby("cc_num").rolling("24h", on="ts", closed="left")["amt"]
cnt = roll.count().reset_index(level=0, drop=True)
mean = roll.mean().reset_index(level=0, drop=True)
X.loc[g.index, "card_n_24h"] = cnt.values
X.loc[g.index, "card_mean_amt_24h"] = mean.values
X["card_n_24h"] = X["card_n_24h"].fillna(0)
X["card_amt_to_mean_24h"] = (df["amt"] / X["card_mean_amt_24h"]).replace([np.inf, -np.inf], np.nan)
X["card_mean_amt_24h"] = np.log1p(X["card_mean_amt_24h"])
cats = pd.get_dummies(df["category"], prefix="cat", dtype=np.float32)
X = pd.concat([X, cats], axis=1).astype(np.float32)
y = df["is_fraud"].values.astype(np.int8)
cards = df["cc_num"].values

# ---------------------------------------------------------------- card-based split
card_fraud = df.groupby("cc_num")["is_fraud"].max()
rng = np.random.default_rng(SEED)
test_cards = []
for flag in (0, 1):
    cs = card_fraud.index[card_fraud.values == flag].values
    test_cards.append(rng.choice(cs, size=int(round(0.2 * len(cs))), replace=False))
test_cards = set(np.concatenate(test_cards))
te = np.isin(cards, list(test_cards))
# compute budget: the models are trained on a random 50% of the training cards (drawn separately
# among cards with and without fraud); the test set keeps all 200 test cards
train_card_ids = card_fraud.index[~card_fraud.index.isin(list(test_cards))]
keep_cards = []
for flag in (0, 1):
    cs = train_card_ids[card_fraud.loc[train_card_ids].values == flag].values
    keep_cards.append(rng.choice(cs, size=int(round(0.5 * len(cs))), replace=False))
keep_cards = np.concatenate(keep_cards)
tr = np.isin(cards, keep_cards)
# 5 grouped folds on the training cards
fold_id = np.full(len(y), -1, dtype=np.int8)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
idx_tr = np.where(tr)[0]
for k, (_, va) in enumerate(sgkf.split(np.zeros(len(idx_tr)), y[idx_tr], groups=cards[idx_tr])):
    fold_id[idx_tr[va]] = k

np.savez_compressed(OUT / "sparkov_card.npz", Xtr=X.values[tr], ytr=y[tr], Xte=X.values[te], yte=y[te],
                    fold_tr=fold_id[tr], groups_tr=cards[tr], groups_te=cards[te], feat=np.array(list(X.columns)),
                    amt_te=df["amt"].values[te], cat_te=df["category"].astype(str).values[te])

# ---------------------------------------------------------------- extra splits
date_cut = pd.Timestamp("2020-07-01")
dte = (df["ts"] >= date_cut).values
np.savez_compressed(OUT / "sparkov_date.npz", Xtr=X.values[~dte], ytr=y[~dte], Xte=X.values[dte], yte=y[dte])
itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
np.savez_compressed(OUT / "sparkov_random.npz", Xtr=X.values[itr], ytr=y[itr], Xte=X.values[ite], yte=y[ite])

info = json.load(open(OUT / "data_info.json"))
info["sparkov"] = dict(raw_rows=n_raw, raw_frauds=f_raw, duplicates_removed=n_dup, rows=len(df), frauds=int(y.sum()),
                       cards=int(len(card_fraud)), cards_with_fraud=int(card_fraud.sum()),
                       train_rows=int(tr.sum()), train_frauds=int(y[tr].sum()), train_cards=int(len(keep_cards)),
                       train_cards_available=int(len(card_fraud) - len(test_cards)),
                       train_rows_available=int((~te).sum()), train_frauds_available=int(y[~te].sum()),
                       test_rows=int(te.sum()), test_frauds=int(y[te].sum()), test_cards=len(test_cards),
                       period=[str(df["ts"].min()), str(df["ts"].max())], n_features=X.shape[1],
                       fold_sizes=[int((fold_id == k).sum()) for k in range(5)],
                       fold_frauds=[int(y[fold_id == k].sum()) for k in range(5)],
                       date_split_test_rows=int(dte.sum()), date_split_test_frauds=int(y[dte].sum()))
json.dump(info, open(OUT / "data_info.json", "w"), indent=2)
print(json.dumps(info["sparkov"], indent=1))
print(list(X.columns))
