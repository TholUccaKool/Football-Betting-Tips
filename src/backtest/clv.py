"""Closing Line Value (CLV) calculator for betting analysis."""

import numpy as np
import pandas as pd


def implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def clv_single(model_prob: float, closing_odds: float) -> float:
    """Compute CLV% for a single prediction.

    CLV% = (model_prob - implied_prob) / implied_prob * 100

    Positive CLV means the model found value vs the closing line.
    """
    impl_prob = implied_probability(closing_odds)
    if impl_prob <= 0:
        return 0.0
    return (model_prob - impl_prob) / impl_prob * 100.0


def aggregate_clv(
    model_probs: np.ndarray,
    closing_odds: np.ndarray,
    results: np.ndarray,
    threshold: float = 0.0,
) -> dict:
    """Compute aggregate CLV stats across a backtest period.

    Args:
        model_probs: (n, 3) array of [p_home, p_draw, p_away] model probabilities.
        closing_odds: (n, 3) array of [home_odds, draw_odds, away_odds] closing odds.
        results: 1-D array of actual outcomes (0=H, 1=D, 2=A).
        threshold: Minimum CLV% to count as a "value bet".

    Returns:
        Dict with avg_clv, n_value_bets, value_bet_hit_rate, etc.
    """
    clvs = []
    value_bets = []
    value_hits = []

    for i in range(len(results)):
        for outcome in range(3):
            mp = model_probs[i, outcome]
            co = closing_odds[i, outcome]
            if np.isnan(co) or co <= 1.0:
                continue
            c = clv_single(mp, co)
            clvs.append(c)

            if c > threshold:
                value_bets.append(c)
                value_hits.append(1 if int(results[i]) == outcome else 0)

    return {
        "avg_clv_pct": np.mean(clvs) if clvs else 0.0,
        "n_predictions": len(clvs),
        "n_value_bets": len(value_bets),
        "avg_value_bet_clv_pct": np.mean(value_bets) if value_bets else 0.0,
        "value_bet_hit_rate": np.mean(value_hits) if value_hits else 0.0,
    }
