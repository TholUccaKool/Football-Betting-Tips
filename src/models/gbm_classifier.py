"""XGBoost multi-class classifier for match outcome prediction."""

import numpy as np
import pandas as pd
import xgboost as xgb

# TODO: Re-add these once Understat xG data is persisted into matches.parquet.
# They are currently all-NaN and contribute nothing to the model.
_XG_FEATURES = [
    "home_rolling_xg_for", "home_rolling_xg_against",
    "away_rolling_xg_for", "away_rolling_xg_against",
]

FEATURES_OWN_SIGNAL = [
    "elo_diff",
    "home_form_ppg", "away_form_ppg",
    "home_rest_days", "away_rest_days",
    "is_home",
]

# Own signal + raw decimal odds.
FEATURES_MARKET_BLEND = FEATURES_OWN_SIGNAL + [
    "home_odds", "draw_odds", "away_odds",
]

# V2: own signal + richer xG features
_XG_V2_FEATURES = [
    "home_xg_overperformance", "away_xg_overperformance",
    "home_opp_adjusted_xg_for", "away_opp_adjusted_xg_for",
]

FEATURES_OWN_SIGNAL_V2 = FEATURES_OWN_SIGNAL + _XG_V2_FEATURES

FEATURES_MARKET_BLEND_V2 = FEATURES_MARKET_BLEND + _XG_V2_FEATURES

# --- Regularization defaults ---
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_N_ESTIMATORS = 500  # ceiling; early stopping cuts well before this
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_ALPHA = 1.0   # L1 regularization
XGB_REG_LAMBDA = 1.0  # L2 regularization
XGB_EARLY_STOPPING_ROUNDS = 20
XGB_VALIDATION_FRAC = 0.15  # last 15% of training window (by date) for early stopping


class GBMClassifier:
    """XGBoost wrapper for home/draw/away classification."""

    def __init__(self, feature_cols: list[str] | None = None, **xgb_params):
        if feature_cols is None:
            feature_cols = FEATURES_OWN_SIGNAL
        defaults = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE,
            "n_estimators": XGB_N_ESTIMATORS,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            "reg_alpha": XGB_REG_ALPHA,
            "reg_lambda": XGB_REG_LAMBDA,
            "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
        defaults.update(xgb_params)
        self.model = xgb.XGBClassifier(**defaults)
        self.feature_cols = list(feature_cols)
        self.best_ntree_limit = None

    def train(self, df: pd.DataFrame) -> "GBMClassifier":
        """Train on a feature DataFrame with time-based early stopping.

        The last XGB_VALIDATION_FRAC of training matches (by date order) are
        held out as a validation set for early stopping. The dataframe must
        already be sorted by date.
        """
        X = df[self.feature_cols].values
        y = df["target"].values

        n_val = max(1, int(len(df) * XGB_VALIDATION_FRAC))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.best_ntree_limit = self.model.best_iteration + 1
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return (n, 3) array of [p_home, p_draw, p_away] probabilities."""
        X = df[self.feature_cols].values
        return self.model.predict_proba(X)
