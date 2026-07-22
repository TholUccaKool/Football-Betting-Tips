"""Elo-based baseline probability model."""

import numpy as np


def elo_probabilities(
    home_elo: float,
    away_elo: float,
    home_advantage: float = 100,
    draw_width: float = 0.1,
) -> tuple[float, float, float]:
    """Convert Elo ratings to home/draw/away probabilities.

    Uses a logistic model with a draw margin derived from draw_width.

    Args:
        home_elo: Home team Elo rating.
        away_elo: Away team Elo rating.
        home_advantage: Elo points added for home advantage.
        draw_width: Controls the probability mass allocated to draws.

    Returns:
        (p_home, p_draw, p_away) probabilities summing to 1.
    """
    diff = (home_elo + home_advantage - away_elo) / 400.0
    p_home_win_or_draw = 1.0 / (1.0 + 10.0 ** (-(diff + draw_width)))
    p_away_win_or_draw = 1.0 / (1.0 + 10.0 ** (diff - draw_width))

    p_draw = max(0, p_home_win_or_draw + p_away_win_or_draw - 1.0)
    p_home = max(0, p_home_win_or_draw - p_draw)
    p_away = max(0, p_away_win_or_draw - p_draw)

    # Normalize
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    return p_home, p_draw, p_away


def predict_from_df(df, home_advantage: float = 100) -> np.ndarray:
    """Predict probabilities for a DataFrame with home_elo and away_elo columns.

    Returns array of shape (n, 3) with columns [p_home, p_draw, p_away].
    """
    probs = [
        elo_probabilities(row["home_elo"], row["away_elo"], home_advantage)
        for _, row in df.iterrows()
    ]
    return np.array(probs)
