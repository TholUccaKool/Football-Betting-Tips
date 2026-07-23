"""Closing Line Value (CLV) analysis using Pinnacle opening/closing odds."""

import numpy as np
import pandas as pd

EDGE_THRESHOLD = 0.03  # Minimum model edge over opening implied prob to flag a bet


def devig_odds(home_odds: np.ndarray, draw_odds: np.ndarray, away_odds: np.ndarray) -> np.ndarray:
    """Convert decimal odds to devigged fair probabilities via normalisation.

    Args:
        home_odds, draw_odds, away_odds: 1-D arrays of decimal odds.

    Returns:
        (n, 3) array of [p_home, p_draw, p_away] fair probabilities.
    """
    raw_h = 1.0 / home_odds
    raw_d = 1.0 / draw_odds
    raw_a = 1.0 / away_odds
    total = raw_h + raw_d + raw_a
    return np.column_stack([raw_h / total, raw_d / total, raw_a / total])


def compute_clv(
    model_probs: np.ndarray,
    opening_odds: np.ndarray,
    closing_odds: np.ndarray,
    edge_threshold: float = EDGE_THRESHOLD,
) -> pd.DataFrame:
    """Compute CLV for each outcome of each match.

    For each match and each outcome (H/D/A):
    - Devig opening odds to get opening implied prob.
    - edge = model_prob - opening_implied_prob
    - If edge > threshold, flag as a bet at the opening price.
    - CLV% = (opening_odds_taken / closing_fair_odds - 1) * 100
      where closing_fair_odds = 1 / devigged_closing_prob.

    Args:
        model_probs: (n, 3) array of model probabilities [H, D, A].
        opening_odds: (n, 3) array of [home, draw, away] opening decimal odds.
        closing_odds: (n, 3) array of [home, draw, away] closing decimal odds.
        edge_threshold: Minimum edge to flag a bet.

    Returns:
        DataFrame with columns: match_idx, outcome, model_prob, opening_impl,
        closing_impl, edge, is_bet, opening_odds, closing_fair_odds, clv_pct.
    """
    open_impl = devig_odds(opening_odds[:, 0], opening_odds[:, 1], opening_odds[:, 2])
    close_impl = devig_odds(closing_odds[:, 0], closing_odds[:, 1], closing_odds[:, 2])

    records = []
    outcome_names = ["H", "D", "A"]
    for oc in range(3):
        edge = model_probs[:, oc] - open_impl[:, oc]
        is_bet = edge > edge_threshold
        closing_fair_odds = 1.0 / close_impl[:, oc]
        clv_pct = (opening_odds[:, oc] / closing_fair_odds - 1.0) * 100.0

        for i in range(len(model_probs)):
            records.append({
                "match_idx": i,
                "outcome": outcome_names[oc],
                "model_prob": model_probs[i, oc],
                "opening_impl": open_impl[i, oc],
                "closing_impl": close_impl[i, oc],
                "edge": edge[i],
                "is_bet": is_bet[i],
                "opening_odds": opening_odds[i, oc],
                "closing_fair_odds": closing_fair_odds[i],
                "clv_pct": clv_pct[i],
            })

    return pd.DataFrame(records)
