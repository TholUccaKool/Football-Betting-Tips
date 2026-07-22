"""XGBoost multi-class classifier for match outcome prediction."""

import numpy as np
import pandas as pd
import xgboost as xgb


FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_form_ppg", "away_form_ppg",
    "home_rolling_xg_for", "home_rolling_xg_against",
    "away_rolling_xg_for", "away_rolling_xg_against",
    "home_rest_days", "away_rest_days",
    "is_home",
]


class GBMClassifier:
    """XGBoost wrapper for home/draw/away classification."""

    def __init__(self, **xgb_params):
        defaults = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
        defaults.update(xgb_params)
        self.model = xgb.XGBClassifier(**defaults)
        self.feature_cols = FEATURE_COLS

    def train(self, df: pd.DataFrame) -> "GBMClassifier":
        """Train on a feature DataFrame. Expects 'target' column (0=H, 1=D, 2=A)."""
        X = df[self.feature_cols].values
        y = df["target"].values
        self.model.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return (n, 3) array of [p_home, p_draw, p_away] probabilities."""
        X = df[self.feature_cols].values
        return self.model.predict_proba(X)
