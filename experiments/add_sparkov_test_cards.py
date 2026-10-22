"""Adds the card number of every Sparkov test transaction (groups_te) to cache/sparkov_card.npz.

Needed for the card-level bootstrap. Repeats the card split of prep_sparkov.py (same order, same seed)
and checks that labels and amounts match the cached test set before writing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "sparkov"
if not DATA.exists():
    DATA = Path(__file__).parent / "data" / "sparkov"
CACHE = Path(__file__).parent / "cache" / "sparkov_card.npz"
SEED = 42

df = pd.concat([pd.read_csv(DATA / "fraudTrain.csv", index_col=0), pd.read_csv(DATA / "fraudTest.csv", index_col=0)],
               ignore_index=True)
df = df.drop_duplicates(subset=["trans_num"])
df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
df = df.sort_values(["ts", "trans_num"], kind="stable").reset_index(drop=True)
cards = df["cc_num"].values
card_fraud = df.groupby("cc_num")["is_fraud"].max()
rng = np.random.default_rng(SEED)
test_cards = []
for flag in (0, 1):
    cs = card_fraud.index[card_fraud.values == flag].values
    test_cards.append(rng.choice(cs, size=int(round(0.2 * len(cs))), replace=False))
te = np.isin(cards, np.concatenate(test_cards))

d = dict(np.load(CACHE, allow_pickle=True))
assert np.array_equal(d["yte"].astype(int), df["is_fraud"].values[te].astype(int))
assert np.allclose(d["amt_te"], df["amt"].values[te])
d["groups_te"] = cards[te]
np.savez_compressed(CACHE, **d)
print("groups_te added:", len(d["groups_te"]), "rows,", len(np.unique(d["groups_te"])), "cards")
