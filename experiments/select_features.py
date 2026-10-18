"""IEEE-CIS: keep the 100 features with the highest LightGBM gain, computed on the training period only."""
import numpy as np
from lightgbm import LGBMClassifier
from common import RES, load
d = load("ieee_time")
m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, n_jobs=2, verbose=-1, random_state=42)
m.fit(d["Xtr"], d["ytr"].astype(int))
gain = m.booster_.feature_importance("gain")
keep = np.sort(np.argsort(-gain)[:100])
(RES / "ieee_time").mkdir(parents=True, exist_ok=True)
np.save(RES / "ieee_time" / "selected_features.npy", keep)
print(len(keep), list(d["feat"][keep][:30]))
