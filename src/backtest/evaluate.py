"""Evaluation metrics and walk-forward splitting for backtesting."""

import numpy as np
import pandas as pd

MIN_TRAIN_MATCHES = 200  # Exclude splits with fewer training matches from aggregate metrics


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Multi-class Brier score.

    Args:
        y_true: 1-D array of class labels (0, 1, 2).
        y_proba: (n, 3) array of predicted probabilities.
    """
    n_classes = y_proba.shape[1]
    one_hot = np.zeros_like(y_proba)
    one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return np.mean(np.sum((y_proba - one_hot) ** 2, axis=1))


def log_loss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-15) -> float:
    """Multi-class log loss.

    Args:
        y_true: 1-D array of class labels (0, 1, 2).
        y_proba: (n, 3) array of predicted probabilities.
        eps: Clipping epsilon to avoid log(0).
    """
    y_proba = np.clip(y_proba, eps, 1 - eps)
    n = len(y_true)
    return -np.sum(np.log(y_proba[np.arange(n), y_true.astype(int)])) / n


def walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    train_months: int = 12,
    test_months: int = 3,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate chronological train/test splits — never shuffled.

    Slides a window forward: train on [start, start + train_months),
    test on [start + train_months, start + train_months + test_months).

    Args:
        df: DataFrame with a date column, sorted chronologically.
        date_col: Name of the date column.
        train_months: Size of training window in months.
        test_months: Size of test window in months.

    Returns:
        List of (train_df, test_df) tuples.
    """
    df = df.sort_values(date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col])

    min_date = df[date_col].min()
    max_date = df[date_col].max()

    splits = []
    current = min_date

    while True:
        train_end = current + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if train_end >= max_date:
            break

        train = df[(df[date_col] >= current) & (df[date_col] < train_end)]
        test = df[(df[date_col] >= train_end) & (df[date_col] < test_end)]

        if len(train) > 0 and len(test) > 0:
            splits.append((train, test))

        current = current + pd.DateOffset(months=test_months)

    return splits


def market_implied_probabilities(
    df: pd.DataFrame,
    home_odds_col: str = "home_odds",
    draw_odds_col: str = "draw_odds",
    away_odds_col: str = "away_odds",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert bookmaker odds to devigged probabilities via normalisation.

    Returns:
        (probs, mask) where probs is (n_valid, 3) and mask is a boolean array
        indicating which rows had complete odds.
    """
    has_odds = df[[home_odds_col, draw_odds_col, away_odds_col]].notna().all(axis=1)
    valid = df[has_odds]
    raw_h = 1.0 / valid[home_odds_col].values
    raw_d = 1.0 / valid[draw_odds_col].values
    raw_a = 1.0 / valid[away_odds_col].values
    total = raw_h + raw_d + raw_a
    probs = np.column_stack([raw_h / total, raw_d / total, raw_a / total])
    return probs, has_odds.values
