"""Merge match history + Understat xG into a single clean parquet dataset."""

import logging

import pandas as pd

from src.utils.io import load_config, get_raw_dir, get_processed_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LEAGUE_CODE_TO_NAME = {
    # Core leagues
    "E0": "EPL",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    # Secondary leagues
    "E1": "Championship",
    "SP2": "Segunda División",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
}


def _load_raw_parquets(subdir: str) -> pd.DataFrame:
    """Load and concatenate all parquet files from a raw subdirectory."""
    raw_dir = get_raw_dir() / subdir
    if not raw_dir.exists():
        return pd.DataFrame()
    frames = []
    for f in sorted(raw_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        # Extract league code from filename (e.g. E0_2023.parquet)
        league_code = f.stem.split("_")[0]
        df["league_code"] = league_code
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Understat name → football-data.co.uk name (canonical for matches.parquet)
_UNDERSTAT_TO_FD = {
    "AC Milan": "Milan",
    "Arminia Bielefeld": "Bielefeld",
    "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia M.Gladbach": "M'gladbach",
    "Celta Vigo": "Celta",
    "Clermont Foot": "Clermont",
    "FC Cologne": "FC Koln",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Espanyol": "Espanol",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf",
    "Hamburger SV": "Hamburg",
    "Hannover 96": "Hannover",
    "Hertha Berlin": "Hertha",
    "Mainz 05": "Mainz",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Nuernberg": "Nurnberg",
    "Paris Saint Germain": "Paris SG",
    "Parma Calcio 1913": "Parma",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Oviedo": "Oviedo",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "Saint-Etienne": "St Etienne",
    "SD Huesca": "Huesca",
    "SPAL 2013": "Spal",
    "VfB Stuttgart": "Stuttgart",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}


def _normalize_team_name(name: str) -> str:
    """Basic team name normalization for matching across sources."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("fc ", "")
        .replace(" fc", "")
        .replace(".", "")
        .replace("'", "")
    )


def _map_understat_team(name: str) -> str:
    """Map an Understat team name to its football-data.co.uk equivalent."""
    return _UNDERSTAT_TO_FD.get(name, name)


def build(cfg: dict | None = None) -> pd.DataFrame:
    """Build the merged match dataset."""
    if cfg is None:
        cfg = load_config()

    # Determine which league codes to include
    all_league_codes = set(cfg.get("leagues", []) + cfg.get("secondary_leagues", []))

    logger.info("Loading raw match history...")
    mh = _load_raw_parquets("match_history")
    if not mh.empty and "league_code" in mh.columns:
        mh = mh[mh["league_code"].isin(all_league_codes)]
    logger.info(f"  {len(mh)} match history rows")

    logger.info("Loading raw Understat xG data...")
    us = _load_raw_parquets("understat")
    logger.info(f"  {len(us)} Understat rows")

    if mh.empty:
        logger.error("No match history data found. Run ingest_match_history first.")
        return pd.DataFrame()

    # Reset multi-index if present (soccerdata often uses multi-index)
    if isinstance(mh.index, pd.MultiIndex):
        mh = mh.reset_index()
    if not us.empty and isinstance(us.index, pd.MultiIndex):
        us = us.reset_index()

    # Standardize match history columns
    mh_cols = mh.columns.str.lower()
    mh.columns = mh_cols

    # Try to identify key columns (soccerdata column names vary by version)
    date_col = next((c for c in mh.columns if c in ("date", "match_date")), None)
    home_col = next((c for c in mh.columns if c in ("home", "hometeam", "home_team")), None)
    away_col = next((c for c in mh.columns if c in ("away", "awayteam", "away_team")), None)
    hg_col = next((c for c in mh.columns if c in ("hg", "fthg", "home_goals", "homegoals")), None)
    ag_col = next((c for c in mh.columns if c in ("ag", "ftag", "away_goals", "awaygoals")), None)

    if any(c is None for c in [date_col, home_col, away_col, hg_col, ag_col]):
        logger.error(f"Cannot identify required columns. Available: {list(mh.columns)}")
        return pd.DataFrame()

    # Build core match dataframe
    matches = pd.DataFrame({
        "date": pd.to_datetime(mh[date_col]),
        "league": mh["league_code"].map(LEAGUE_CODE_TO_NAME),
        "home_team": mh[home_col].astype(str),
        "away_team": mh[away_col].astype(str),
        "home_goals": pd.to_numeric(mh[hg_col], errors="coerce"),
        "away_goals": pd.to_numeric(mh[ag_col], errors="coerce"),
    })

    # Extract odds — prefer Bet365, fall back to average
    for prefix, target in [("home", "home_odds"), ("draw", "draw_odds"), ("away", "away_odds")]:
        b365_col = next(
            (c for c in mh.columns if c in (f"b365{prefix[0]}", f"bet365{prefix[0]}", f"b365_{prefix}")),
            None,
        )
        avg_col = next(
            (c for c in mh.columns if c in (f"avg{prefix[0]}", f"avg_{prefix}", f"market_avg_{prefix}")),
            None,
        )
        odds_col = b365_col or avg_col
        if odds_col is not None:
            matches[target] = pd.to_numeric(mh[odds_col], errors="coerce")
        else:
            matches[target] = float("nan")

    # Pinnacle opening odds (PSH/PSD/PSA)
    for col_raw, target in [("psh", "pin_open_home"), ("psd", "pin_open_draw"), ("psa", "pin_open_away")]:
        if col_raw in mh.columns:
            matches[target] = pd.to_numeric(mh[col_raw], errors="coerce")
        else:
            matches[target] = float("nan")

    # Pinnacle closing odds (PSCH/PSCD/PSCA)
    for col_raw, target in [("psch", "pin_close_home"), ("pscd", "pin_close_draw"), ("psca", "pin_close_away")]:
        if col_raw in mh.columns:
            matches[target] = pd.to_numeric(mh[col_raw], errors="coerce")
        else:
            matches[target] = float("nan")

    # Result column
    matches["result"] = "D"
    matches.loc[matches["home_goals"] > matches["away_goals"], "result"] = "H"
    matches.loc[matches["home_goals"] < matches["away_goals"], "result"] = "A"

    # Merge xG from Understat if available
    matches["home_xg"] = float("nan")
    matches["away_xg"] = float("nan")

    if not us.empty:
        us_cols = us.columns.str.lower()
        us.columns = us_cols

        us_date = next((c for c in us.columns if c in ("date", "match_date", "datetime")), None)
        us_home = next((c for c in us.columns if c in ("home", "hometeam", "home_team")), None)
        us_away = next((c for c in us.columns if c in ("away", "awayteam", "away_team")), None)
        us_hxg = next((c for c in us.columns if "home" in c and "xg" in c), None)
        us_axg = next((c for c in us.columns if "away" in c and "xg" in c), None)

        if all(c is not None for c in [us_date, us_home, us_away, us_hxg, us_axg]):
            # Deduplicate Understat rows (overlapping season files)
            us = us.drop_duplicates(subset=[us_date, us_home, us_away], keep="first")
            logger.info(f"  {len(us)} Understat rows after dedup")

            us_clean = pd.DataFrame({
                "us_date": pd.to_datetime(us[us_date]).dt.date,
                "us_home_norm": us[us_home].apply(_map_understat_team).apply(_normalize_team_name),
                "us_away_norm": us[us_away].apply(_map_understat_team).apply(_normalize_team_name),
                "home_xg": pd.to_numeric(us[us_hxg], errors="coerce"),
                "away_xg": pd.to_numeric(us[us_axg], errors="coerce"),
            })

            matches["_date_key"] = matches["date"].dt.date
            matches["_home_norm"] = matches["home_team"].apply(_normalize_team_name)
            matches["_away_norm"] = matches["away_team"].apply(_normalize_team_name)

            merged = matches.merge(
                us_clean,
                left_on=["_date_key", "_home_norm", "_away_norm"],
                right_on=["us_date", "us_home_norm", "us_away_norm"],
                how="left",
                suffixes=("", "_us"),
            )

            if "home_xg_us" in merged.columns:
                merged["home_xg"] = merged["home_xg_us"].combine_first(merged["home_xg"])
                merged["away_xg"] = merged["away_xg_us"].combine_first(merged["away_xg"])

            matches = merged

    # Final column selection and cleanup
    output_cols = [
        "date", "league", "home_team", "away_team",
        "home_goals", "away_goals", "home_xg", "away_xg",
        "home_odds", "draw_odds", "away_odds",
        "pin_open_home", "pin_open_draw", "pin_open_away",
        "pin_close_home", "pin_close_draw", "pin_close_away",
        "result",
    ]
    matches = matches[[c for c in output_cols if c in matches.columns]]
    matches = matches.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    matches = matches.dropna(subset=["home_goals", "away_goals", "league"])
    matches = matches.sort_values("date").reset_index(drop=True)

    # Report xG coverage
    xg_present = matches["home_xg"].notna().sum()
    logger.info(f"xG coverage: {xg_present}/{len(matches)} matches ({100*xg_present/len(matches):.1f}%)")

    # Save
    out_path = get_processed_dir() / "matches.parquet"
    matches.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(matches)} matches -> {out_path}")

    return matches


if __name__ == "__main__":
    import sys
    result = build()
    if result is None or result.empty:
        print("::error::build_dataset produced no matches - matches.parquet was NOT written")
        sys.exit(1)
