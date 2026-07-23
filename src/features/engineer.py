"""Feature engineering from processed match data."""

import pandas as pd
import numpy as np

from src.features.elo import EloRatingSystem
from src.utils.io import load_config, get_processed_dir


def _rolling_team_stats(matches: pd.DataFrame, window: int = 5) -> dict[str, pd.DataFrame]:
    """Compute per-team rolling stats with proper time ordering.

    Returns a dict mapping team -> DataFrame with rolling features indexed by match date.
    """
    # Build a long-form record: one row per team per match
    home = matches[["date", "home_team", "home_goals", "away_goals", "home_xg", "away_xg", "result"]].copy()
    home.columns = ["date", "team", "goals_for", "goals_against", "xg_for", "xg_against", "result"]
    home["points"] = home["result"].map({"H": 3, "D": 1, "A": 0})
    home["is_home"] = True

    away = matches[["date", "away_team", "away_goals", "home_goals", "away_xg", "home_xg", "result"]].copy()
    away.columns = ["date", "team", "goals_for", "goals_against", "xg_for", "xg_against", "result"]
    away["points"] = away["result"].map({"H": 0, "D": 1, "A": 3})
    away["is_home"] = False

    long = pd.concat([home, away]).sort_values("date").reset_index(drop=True)

    team_stats = {}
    for team, grp in long.groupby("team"):
        grp = grp.sort_values("date").copy()
        # Shift by 1 so we only use info BEFORE the current match
        grp["rolling_ppg"] = grp["points"].shift(1).rolling(window, min_periods=1).mean()
        grp["rolling_xg_for"] = grp["xg_for"].shift(1).rolling(window, min_periods=1).mean()
        grp["rolling_xg_against"] = grp["xg_against"].shift(1).rolling(window, min_periods=1).mean()
        grp["prev_date"] = grp["date"].shift(1)
        grp["rest_days"] = (grp["date"] - grp["prev_date"]).dt.days
        team_stats[team] = grp

    return team_stats


def build_features(matches: pd.DataFrame | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Build the full feature matrix from matches.parquet.

    All features use only pre-match information (no lookahead).
    """
    if cfg is None:
        cfg = load_config()
    if matches is None:
        matches = pd.read_parquet(get_processed_dir() / "matches.parquet")

    matches = matches.sort_values("date").reset_index(drop=True)

    # --- Elo ratings (pre-match) ---
    elo_cfg = cfg.get("elo", {})
    elo = EloRatingSystem(
        k_factor=elo_cfg.get("k_factor", 20),
        home_advantage=elo_cfg.get("home_advantage", 100),
        initial_rating=elo_cfg.get("initial_rating", 1500),
    )

    home_elos, away_elos = [], []
    for _, row in matches.iterrows():
        h_elo, a_elo = elo.get_pre_match_ratings(row["home_team"], row["away_team"])
        home_elos.append(h_elo)
        away_elos.append(a_elo)
        elo.update(row["home_team"], row["away_team"], int(row["home_goals"]), int(row["away_goals"]))

    matches["home_elo"] = home_elos
    matches["away_elo"] = away_elos
    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]

    # --- Rolling form features ---
    team_stats = _rolling_team_stats(matches)

    # Build lookup: (team, date, is_home) -> features
    home_rolling = []
    away_rolling = []
    for _, row in matches.iterrows():
        ht, at, d = row["home_team"], row["away_team"], row["date"]

        if ht in team_stats:
            ht_data = team_stats[ht]
            ht_match = ht_data[(ht_data["date"] == d) & (ht_data["is_home"] == True)]
            if not ht_match.empty:
                home_rolling.append(ht_match.iloc[0])
            else:
                home_rolling.append(pd.Series(dtype=float))
        else:
            home_rolling.append(pd.Series(dtype=float))

        if at in team_stats:
            at_data = team_stats[at]
            at_match = at_data[(at_data["date"] == d) & (at_data["is_home"] == False)]
            if not at_match.empty:
                away_rolling.append(at_match.iloc[0])
            else:
                away_rolling.append(pd.Series(dtype=float))
        else:
            away_rolling.append(pd.Series(dtype=float))

    home_roll_df = pd.DataFrame(home_rolling).reset_index(drop=True)
    away_roll_df = pd.DataFrame(away_rolling).reset_index(drop=True)

    matches["home_form_ppg"] = home_roll_df.get("rolling_ppg", np.nan)
    matches["away_form_ppg"] = away_roll_df.get("rolling_ppg", np.nan)
    matches["home_rolling_xg_for"] = home_roll_df.get("rolling_xg_for", np.nan)
    matches["home_rolling_xg_against"] = home_roll_df.get("rolling_xg_against", np.nan)
    matches["away_rolling_xg_for"] = away_roll_df.get("rolling_xg_for", np.nan)
    matches["away_rolling_xg_against"] = away_roll_df.get("rolling_xg_against", np.nan)
    matches["home_rest_days"] = home_roll_df.get("rest_days", np.nan)
    matches["away_rest_days"] = away_roll_df.get("rest_days", np.nan)
    matches["is_home"] = 1  # Always 1 from home team perspective

    # Target encoding
    matches["target"] = matches["result"].map({"H": 0, "D": 1, "A": 2})

    feature_cols = [
        "home_elo", "away_elo", "elo_diff",
        "home_form_ppg", "away_form_ppg",
        "home_rolling_xg_for", "home_rolling_xg_against",
        "away_rolling_xg_for", "away_rolling_xg_against",
        "home_rest_days", "away_rest_days",
        "is_home",
    ]

    meta_cols = ["date", "league", "home_team", "away_team",
                 "home_goals", "away_goals", "result", "target"]
    odds_cols = ["home_odds", "draw_odds", "away_odds"]
    pin_cols = [c for c in ["pin_open_home", "pin_open_draw", "pin_open_away",
                            "pin_close_home", "pin_close_draw", "pin_close_away"]
                if c in matches.columns]
    return matches[meta_cols + feature_cols + odds_cols + pin_cols]


if __name__ == "__main__":
    df = build_features()
    print(f"Feature matrix: {df.shape}")
    print(df.head())
