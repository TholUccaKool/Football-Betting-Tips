"""Elo-based baseline probability model.

Uses multinomial logistic regression on elo_diff to convert raw Elo rating
differences into calibrated (home, draw, away) probabilities.  The logistic
regression must be fit on training data per walk-forward split to avoid
leakage.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_elo_calibrator(train_df) -> LogisticRegression:
    """Fit a multinomial logistic regression on elo_diff from training data.

    Args:
        train_df: DataFrame with 'elo_diff' and 'target' (0=H, 1=D, 2=A).

    Returns:
        Fitted LogisticRegression model.
    """
    X = train_df[["elo_diff"]].values
    y = train_df["target"].values.astype(int)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    return model


def predict_calibrated(model: LogisticRegression, test_df) -> np.ndarray:
    """Predict (p_home, p_draw, p_away) using a fitted calibrator.

    Returns array of shape (n, 3).
    """
    X = test_df[["elo_diff"]].values
    return model.predict_proba(X)
