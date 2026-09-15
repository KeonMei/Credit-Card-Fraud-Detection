"""Model zoo and class-imbalance handling.

Models
------
logreg - logistic regression: interpretable linear baseline
rf     - random forest: classic bagging baseline used in most fraud papers
xgb    - XGBoost gradient boosting
lgbm   - LightGBM gradient boosting (usually the strongest on tabular data)

Imbalance strategies (applied to the training set only, never to val/test)
---------------------------------------------------------------------------
none        - train on the natural class distribution
weight      - cost-sensitive learning: fraud errors weighted by n_negative / n_positive
smote       - SMOTE oversampling of frauds (Chawla et al., 2002)
undersample - random undersampling of legitimate transactions (Dal Pozzolo et al., 2015)

Resampling distorts predicted probabilities, which is why the decision threshold is
later tuned on the untouched validation set rather than fixed at 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = ["logreg", "rf", "xgb", "lgbm"]
IMBALANCE_STRATEGIES = ["none", "weight", "smote", "undersample"]
BOOSTING_MODELS = {"xgb", "lgbm"}

EARLY_STOPPING_ROUNDS = 100


class FraudModel:
    """preprocessing -> (optional) resampling -> estimator, with a uniform interface."""

    def __init__(self, name: str, imbalance: str = "weight", random_state: int = 42, n_jobs: int = -1):
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model {name!r}, expected one of {MODEL_NAMES}")
        if imbalance not in IMBALANCE_STRATEGIES:
            raise ValueError(f"Unknown imbalance strategy {imbalance!r}, expected one of {IMBALANCE_STRATEGIES}")
        self.name = name
        self.imbalance = imbalance
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ building blocks

    def _make_preprocessor(self):
        # Trees handle NaN natively; linear models need imputation + scaling;
        # SMOTE interpolates between neighbours, so it cannot work with NaN either.
        if self.name == "logreg":
            steps = [SimpleImputer(strategy="median"), StandardScaler()]
        elif self.name == "rf" or self.imbalance == "smote":
            steps = [SimpleImputer(strategy="median")]
        else:
            return None
        return make_pipeline(*steps).set_output(transform="pandas")

    def _make_sampler(self):
        if self.imbalance == "smote":
            return SMOTE(random_state=self.random_state)
        if self.imbalance == "undersample":
            return RandomUnderSampler(random_state=self.random_state)
        return None

    def _make_estimator(self, pos_weight: float):
        rs, n_jobs = self.random_state, self.n_jobs
        balanced = self.imbalance == "weight"

        if self.name == "logreg":
            return LogisticRegression(max_iter=2000, class_weight="balanced" if balanced else None)
        if self.name == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample" if balanced else None,
                n_jobs=n_jobs,
                random_state=rs,
            )
        if self.name == "xgb":
            return XGBClassifier(
                n_estimators=3000,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.6,
                min_child_weight=1,
                tree_method="hist",
                eval_metric="aucpr",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                scale_pos_weight=pos_weight if balanced else 1.0,
                n_jobs=n_jobs,
                random_state=rs,
            )
        return LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            scale_pos_weight=pos_weight if balanced else 1.0,
            n_jobs=n_jobs,
            metric="average_precision",
            random_state=rs,
            verbose=-1,
        )

    # ------------------------------------------------------------------ public API

    def fit(self, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame | None = None, y_val=None) -> FraudModel:
        y_train = np.asarray(y_train)
        n_pos = max(int(y_train.sum()), 1)
        pos_weight = (len(y_train) - n_pos) / n_pos

        self.preprocessor_ = self._make_preprocessor()
        Xt = self.preprocessor_.fit_transform(X_train) if self.preprocessor_ is not None else X_train

        sampler = self._make_sampler()
        if sampler is not None:
            Xt, y_train = sampler.fit_resample(Xt, y_train)

        self.estimator_ = self._make_estimator(pos_weight)
        fit_kwargs = {}
        if self.name in BOOSTING_MODELS and X_val is not None:
            X_eval, y_eval = self._preprocess(X_val), np.asarray(y_val)
            if self.name == "xgb":
                fit_kwargs = {"eval_set": [(X_eval, y_eval)], "verbose": False}
            else:
                fit_kwargs = {
                    "eval_X": X_eval,
                    "eval_y": y_eval,
                    "callbacks": [early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
                }
        elif self.name == "xgb":
            # early stopping needs a validation set; fall back to a fixed number of trees
            self.estimator_.set_params(early_stopping_rounds=None, n_estimators=500)

        self.estimator_.fit(Xt, y_train, **fit_kwargs)
        self.feature_names_ = list(Xt.columns)
        return self

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor_.transform(X) if self.preprocessor_ is not None else X

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of fraud (positive class) for each row."""
        return self.estimator_.predict_proba(self._preprocess(X))[:, 1]

    @property
    def best_iteration(self) -> int | None:
        if self.name == "xgb":
            return getattr(self.estimator_, "best_iteration", None)
        if self.name == "lgbm":
            return self.estimator_.best_iteration_ or None
        return None

    def feature_importance(self, X: pd.DataFrame, max_rows: int = 20_000) -> pd.Series:
        """Mean |SHAP value| per feature (Lundberg & Lee, 2017).

        Exact TreeSHAP is built into XGBoost and LightGBM, so no extra dependency is
        needed. Other models fall back to coefficients / impurity importance.
        """
        Xp = self._preprocess(X)
        if len(Xp) > max_rows:
            Xp = Xp.sample(max_rows, random_state=self.random_state)

        if self.name == "lgbm":
            contrib = self.estimator_.predict(Xp, pred_contrib=True)[:, :-1]
            values = np.abs(contrib).mean(axis=0)
        elif self.name == "xgb":
            from xgboost import DMatrix

            best = self.best_iteration
            iteration_range = (0, best + 1) if best is not None else (0, 0)
            contrib = self.estimator_.get_booster().predict(
                DMatrix(Xp), pred_contribs=True, iteration_range=iteration_range
            )[:, :-1]
            values = np.abs(contrib).mean(axis=0)
        elif self.name == "logreg":
            values = np.abs(self.estimator_.coef_[0])
        else:
            values = self.estimator_.feature_importances_

        return pd.Series(values, index=self.feature_names_).sort_values(ascending=False)
