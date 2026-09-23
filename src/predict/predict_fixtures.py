"""Fetch upcoming fixtures with odds, run trained models, and display probabilities side-by-side.

Data source (live mode): football-data.co.uk/fixtures.csv (free, updated weekly).
Demo mode (--as-of):     uses a past date from matches.parquet as a stand-in.
Models are fit on all available historical data up to the target date.
"""

import argparse
import html as html_mod
import io
import json
import logging
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data._soccerdata_patch import _BROWSER_UA
from src.features.elo import EloRatingSystem
from src.features.engineer import _rolling_team_stats
from src.models.elo_baseline import fit_elo_calibrator
from src.models.dixon_coles import DixonColesModel
from src.models.gbm_classifier import GBMClassifier, FEATURES_MARKET_BLEND
from src.models.market_derivations import over_under, btts, correct_score_top_n, asian_handicap
from src.utils.io import load_config, get_processed_dir, get_raw_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DIV_TO_LEAGUE = {
    # Core 5
    "E0": "EPL", "SP1": "La Liga", "I1": "Serie A",
    "D1": "Bundesliga", "F1": "Ligue 1",
    # Secondary 4
    "E1": "Championship", "SP2": "Segunda Division",
    "N1": "Eredivisie", "P1": "Primeira Liga",
}
TARGET_DIVS = set(DIV_TO_LEAGUE.keys())
LEAGUE_NAMES = set(DIV_TO_LEAGUE.values())

ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_LABELS = ["Market", "Elo", "Dixon-Coles", "XGBoost"]
OUTCOME_MAP = {0: "H", 1: "D", 2: "A"}
OUTCOME_WORDS = {"H": "Home win", "D": "Draw", "A": "Away win"}

# ── TheSportsDB fallback ────────────────────────────────────────────────
TSDB_LEAGUE_IDS = {
    "EPL": 4328, "La Liga": 4335, "Serie A": 4332,
    "Bundesliga": 4331, "Ligue 1": 4334,
    "Championship": 4329, "Segunda Division": 4400,
    "Eredivisie": 4337, "Primeira Liga": 4344,
}
# TheSportsDB team name → football-data.co.uk canonical name (matches.parquet)
_TSDB_TO_FD = {
    # La Liga
    "Deportivo Alavés": "Alaves", "Athletic Bilbao": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid", "Celta Vigo": "Celta",
    "Real Betis": "Betis", "Rayo Vallecano": "Vallecano",
    "Real Sociedad": "Sociedad", "Deportivo de A Coruña": "La Coruna",
    "Racing de Santander": "Santander", "Málaga": "Malaga",
    # EPL
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Leeds United": "Leeds", "Leicester City": "Leicester",
    "Wolverhampton Wanderers": "Wolves", "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham", "Brighton and Hove Albion": "Brighton",
    "Ipswich Town": "Ipswich", "Coventry City": "Coventry",
    "Sheffield United": "Sheffield United", "Hull City": "Hull",
    "Sunderland AFC": "Sunderland", "AFC Bournemouth": "Bournemouth",
    # Serie A
    "Inter Milan": "Inter", "AC Milan": "Milan",
    "Hellas Verona": "Verona",
    # Bundesliga
    "Borussia Dortmund": "Dortmund", "Eintracht Frankfurt": "Ein Frankfurt",
    "Borussia Mönchengladbach": "M'gladbach", "Bayer Leverkusen": "Leverkusen",
    "FC Cologne": "FC Koln", "SC Freiburg": "Freiburg",
    "TSG Hoffenheim": "Hoffenheim", "VfL Wolfsburg": "Wolfsburg",
    "Werder Bremen": "Werder Bremen", "FC Augsburg": "Augsburg",
    "Hertha Berlin": "Hertha", "Schalke 04": "Schalke 04",
    "SC Paderborn 07": "Paderborn", "Holstein Kiel": "Holstein Kiel",
    "FC St. Pauli": "St Pauli", "Mainz 05": "Mainz",
    "VfB Stuttgart": "Stuttgart", "Hamburger SV": "Hamburg",
    # Ligue 1
    "Paris Saint-Germain": "Paris SG", "AS Monaco": "Monaco",
    "Saint-Étienne": "St Etienne", "Stade Rennais": "Rennes",
    "Stade Brestois 29": "Brest", "Montpellier HSC": "Montpellier",
    "RC Lens": "Lens", "OGC Nice": "Nice",
    "Olympique Lyonnais": "Lyon", "Olympique de Marseille": "Marseille",
    "FC Nantes": "Nantes", "Stade de Reims": "Reims",
    "Clermont Foot": "Clermont", "Angers SCO": "Angers",
    # Eredivisie
    "NEC Nijmegen": "Nijmegen", "ADO Den Haag": "Den Haag",
    "Fortuna Sittard": "For Sittard", "RKC Waalwijk": "Waalwijk",
    "FC Twente": "Twente", "FC Utrecht": "Utrecht",
    "FC Groningen": "Groningen", "SC Heerenveen": "Heerenveen",
    "PEC Zwolle": "Zwolle", "SC Cambuur": "Cambuur",
    # Primeira Liga
    "Sporting CP": "Sp Lisbon", "Sporting Braga": "Sp Braga",
    "Vitória de Guimarães": "Guimaraes", "Marítimo": "Maritimo",
    "Estoril Praia": "Estoril", "Estrela Amadora": "Estrela",
    "Académico de Viseu": "Vizela", "Paços de Ferreira": "Pacos Ferreira",
    # La Liga (accent variant)
    "Espanyol": "Espanol",
    # Championship
    "Blackburn Rovers": "Blackburn", "Bolton Wanderers": "Bolton",
    "Preston North End": "Preston", "Norwich City": "Norwich",
    "Lincoln City": "Lincoln", "Queens Park Rangers": "QPR",
    "Swansea City": "Swansea", "Sheffield Wednesday": "Sheffield Weds",
    "Stoke City": "Stoke", "Cardiff City": "Cardiff",
    "Derby County": "Derby", "Huddersfield Town": "Huddersfield",
    "Plymouth Argyle": "Plymouth", "Wigan Athletic": "Wigan",
    "Peterborough United": "Peterboro", "Oxford United": "Oxford",
    "Portsmouth FC": "Portsmouth", "Luton Town": "Luton",
    # Segunda Division
    "Sporting de Gijón": "Sp Gijon", "Real Zaragoza": "Zaragoza",
    "CD Tenerife": "Tenerife", "Gimnàstic de Tarragona": "Gimnastic",
    "Real Sociedad B": "Sociedad B", "Castellón": "Castellon",
    "Real Oviedo": "Oviedo", "Cádiz": "Cadiz",
    "Celta Fortuna": "Celta Fortuna",  # new team, no historical data
    "Real Valladolid": "Valladolid", "FC Andorra": "Andorra",
}


# ── Data loading ─────────────────────────────────────────────────────────

def fetch_fixtures() -> pd.DataFrame:
    """Download the current fixtures CSV from football-data.co.uk."""
    resp = requests.get(FIXTURES_URL, headers={"User-Agent": _BROWSER_UA}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content), encoding="utf-8-sig")
    if df.empty:
        return df
    df = df[df["Div"].isin(TARGET_DIVS)].copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

    # Merge kickoff time into date so cards show real times
    if "Time" in df.columns:
        kick = df["date"] + pd.to_timedelta(
            df["Time"].fillna("23:59").astype(str).str.strip() + ":00"
        )
    else:
        kick = df["date"] + pd.Timedelta(hours=23, minutes=59)
    df["date"] = kick  # replace midnight-only with full datetime

    # Drop matches that have already kicked off (date+time < now)
    now = datetime.now()
    n_before = len(df)
    df = df[df["date"] >= now].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("  Dropped %d already-kicked-off fixture(s)", n_dropped)
    if df.empty:
        return df

    df["league"] = df["Div"].map(DIV_TO_LEAGUE)
    df = df.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team"})
    for src, tgt in [("AvgH", "home_odds"), ("AvgD", "draw_odds"), ("AvgA", "away_odds")]:
        fallback = src.replace("Avg", "B365")
        if src in df.columns:
            df[tgt] = pd.to_numeric(df[src], errors="coerce")
            if fallback in df.columns:
                df[tgt] = df[tgt].fillna(pd.to_numeric(df[fallback], errors="coerce"))
        elif fallback in df.columns:
            df[tgt] = pd.to_numeric(df[fallback], errors="coerce")
        else:
            df[tgt] = np.nan

    # Over/Under 2.5 odds (prefer Avg, fallback B365)
    for src, tgt in [("Avg>2.5", "over25_odds"), ("Avg<2.5", "under25_odds")]:
        fallback = src.replace("Avg", "B365")
        if src in df.columns:
            df[tgt] = pd.to_numeric(df[src], errors="coerce")
        elif fallback in df.columns:
            df[tgt] = pd.to_numeric(df[fallback], errors="coerce")
        else:
            df[tgt] = np.nan

    # Asian handicap line + odds
    if "AHh" in df.columns:
        df["ah_line"] = pd.to_numeric(df["AHh"], errors="coerce")
    else:
        df["ah_line"] = np.nan
    for src, tgt in [("AvgAHH", "ah_home_odds"), ("AvgAHA", "ah_away_odds")]:
        fallback = src.replace("Avg", "B365")
        if src in df.columns:
            df[tgt] = pd.to_numeric(df[src], errors="coerce")
        elif fallback in df.columns:
            df[tgt] = pd.to_numeric(df[fallback], errors="coerce")
        else:
            df[tgt] = np.nan

    keep = ["date", "league", "home_team", "away_team",
            "home_odds", "draw_odds", "away_odds",
            "over25_odds", "under25_odds",
            "ah_line", "ah_home_odds", "ah_away_odds"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def _map_tsdb_team(name: str) -> str:
    """Map a TheSportsDB team name to its football-data.co.uk canonical name."""
    return _TSDB_TO_FD.get(name, name)


TSDB_REQUEST_DELAY = 2.5          # seconds between requests; free key allows ~30/min
TSDB_RETRY_WAITS = (15, 30, 60)   # backoff on HTTP 429 / 5xx, seconds


def _tsdb_get(url: str) -> dict | None:
    """GET a TheSportsDB URL with fixed pacing and backoff on 429/5xx.

    Every request (success or failure) is followed by TSDB_REQUEST_DELAY so the
    loop never bursts past the free-tier rate limit. On 429 or 5xx the request
    is retried up to len(TSDB_RETRY_WAITS) times with increasing waits.
    Returns the parsed JSON dict, or None if the request ultimately failed.
    """
    n_retries = len(TSDB_RETRY_WAITS)
    for attempt in range(n_retries + 1):
        try:
            resp = requests.get(url, timeout=15)
        except Exception as exc:
            logger.warning("  TheSportsDB request error for %s: %s", url, exc)
            time.sleep(TSDB_REQUEST_DELAY)
            return None

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < n_retries:
                wait = TSDB_RETRY_WAITS[attempt]
                logger.warning("  TheSportsDB HTTP %d for %s - backing off %ds (retry %d/%d)",
                               resp.status_code, url, wait, attempt + 1, n_retries)
                time.sleep(wait)
                continue
            logger.warning("  TheSportsDB HTTP %d for %s - giving up after %d retries",
                           resp.status_code, url, n_retries)
            time.sleep(TSDB_REQUEST_DELAY)
            return None

        time.sleep(TSDB_REQUEST_DELAY)
        try:
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("  TheSportsDB bad response for %s: %s", url, exc)
            return None
    return None


def fetch_tsdb_fixtures(leagues: list[str]) -> pd.DataFrame:
    """Fetch upcoming fixtures from TheSportsDB for the given league names.

    Uses the eventsday endpoint across the next 7 days (the free key caps
    eventsnextleague at a single event per league, so per-day queries are the
    only way to get a full matchweek). Requests are paced and retried with
    backoff via _tsdb_get so the free-tier rate limit is never exceeded.
    Returns a DataFrame with the same columns as fetch_fixtures but odds set to NaN.
    """
    if not leagues:
        return pd.DataFrame()

    today = datetime.now().date()
    rows = []
    n_failed = 0
    for league in leagues:
        tsdb_id = TSDB_LEAGUE_IDS.get(league)
        if tsdb_id is None:
            continue
        for day_offset in range(7):
            day = today + timedelta(days=day_offset)
            url = (f"https://www.thesportsdb.com/api/v1/json/3/"
                   f"eventsday.php?d={day.isoformat()}&l={tsdb_id}")
            data = _tsdb_get(url)
            if data is None:
                n_failed += 1
                continue
            events = data.get("events") or []
            for ev in events:
                if ev.get("strStatus") != "NS":
                    continue  # skip already-played or postponed
                kick_str = ev.get("strTimestamp", "")
                try:
                    kick_dt = pd.Timestamp(kick_str)
                except Exception:
                    kick_dt = pd.Timestamp(day)
                if kick_dt < pd.Timestamp.now():
                    continue  # already kicked off
                rows.append({
                    "date": kick_dt,
                    "league": league,
                    "home_team": _map_tsdb_team(ev["strHomeTeam"]),
                    "away_team": _map_tsdb_team(ev["strAwayTeam"]),
                    "home_odds": np.nan,
                    "draw_odds": np.nan,
                    "away_odds": np.nan,
                    "_from_tsdb": True,
                })
    if n_failed:
        logger.warning("  TheSportsDB: %d of %d day-queries failed after retries",
                       n_failed, 7 * len([l for l in leagues if l in TSDB_LEAGUE_IDS]))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_from_tsdb"] = True
    return df.drop_duplicates(subset=["date", "home_team", "away_team"]).reset_index(drop=True)


def _load_extra_odds_from_raw(target_date: pd.Timestamp) -> pd.DataFrame:
    """Load O/U 2.5 and AH odds from raw match_history parquets for a specific date."""
    raw_dir = get_raw_dir() / "match_history"
    if not raw_dir.exists():
        return pd.DataFrame()
    frames = []
    for f in sorted(raw_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"])
        day_df = df[df["date"].dt.normalize() == target_date.normalize()]
        if day_df.empty:
            continue
        keep = {"date": "date", "home_team": "home_team", "away_team": "away_team"}
        # O/U 2.5 — prefer Avg, fallback B365
        for raw_col, fallback, tgt in [
            ("Avg>2.5", "B365>2.5", "over25_odds"),
            ("Avg<2.5", "B365<2.5", "under25_odds"),
        ]:
            col = raw_col if raw_col in day_df.columns else (fallback if fallback in day_df.columns else None)
            if col:
                keep[col] = tgt
        # AH
        if "AHh" in day_df.columns:
            keep["AHh"] = "ah_line"
        for raw_col, fallback, tgt in [
            ("AvgAHH", "B365AHH", "ah_home_odds"),
            ("AvgAHA", "B365AHA", "ah_away_odds"),
        ]:
            col = raw_col if raw_col in day_df.columns else (fallback if fallback in day_df.columns else None)
            if col:
                keep[col] = tgt

        sub = day_df[[c for c in keep if c in day_df.columns]].copy()
        sub = sub.rename(columns=keep)
        for c in ["over25_odds", "under25_odds", "ah_line", "ah_home_odds", "ah_away_odds"]:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")


def fetch_demo_fixtures(as_of: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull matches on a past date from matches.parquet as stand-in fixtures."""
    hist = load_historical_data()
    target = pd.Timestamp(as_of)
    day = hist[hist["date"].dt.normalize() == target].copy()
    day = day[day["league"].isin(LEAGUE_NAMES)]
    if day.empty:
        return pd.DataFrame(), pd.DataFrame()
    fx_cols = ["date", "league", "home_team", "away_team",
               "home_odds", "draw_odds", "away_odds"]
    act_cols = ["home_team", "away_team", "home_goals", "away_goals", "result"]
    fixtures = day[fx_cols].reset_index(drop=True)
    actuals = day[act_cols].reset_index(drop=True)

    # Merge extra odds from raw data
    extra = _load_extra_odds_from_raw(target)
    if not extra.empty:
        fixtures["_date_key"] = fixtures["date"].dt.normalize()
        extra["_date_key"] = pd.to_datetime(extra["date"]).dt.normalize()
        n_before = len(fixtures)
        fixtures = fixtures.merge(
            extra.drop(columns=["date"]),
            on=["_date_key", "home_team", "away_team"],
            how="left",
        )
        fixtures.drop(columns=["_date_key"], inplace=True)
        if len(fixtures) > n_before:
            fixtures = fixtures.drop_duplicates(
                subset=["date", "home_team", "away_team"], keep="first")

    return fixtures.reset_index(drop=True), actuals


def load_historical_data() -> pd.DataFrame:
    path = get_processed_dir() / "matches.parquet"
    if not path.exists():
        logger.error(f"No historical data at {path}. Run the data pipeline first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Feature / model fitting ─────────────────────────────────────────────

def build_elo_and_features(matches: pd.DataFrame, cfg: dict):
    matches = matches.sort_values("date").reset_index(drop=True)
    elo_cfg = cfg.get("elo", {})
    elo = EloRatingSystem(
        k_factor=elo_cfg.get("k_factor", 20),
        home_advantage=elo_cfg.get("home_advantage", 100),
        initial_rating=elo_cfg.get("initial_rating", 1500),
    )
    home_elos, away_elos = [], []
    for _, row in matches.iterrows():
        h, a = elo.get_pre_match_ratings(row["home_team"], row["away_team"])
        home_elos.append(h)
        away_elos.append(a)
        elo.update(row["home_team"], row["away_team"],
                   int(row["home_goals"]), int(row["away_goals"]))
    matches["home_elo"] = home_elos
    matches["away_elo"] = away_elos
    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]
    team_stats = _rolling_team_stats(matches)
    home_rolling, away_rolling = [], []
    for _, row in matches.iterrows():
        ht, at, d = row["home_team"], row["away_team"], row["date"]
        for team, is_home, coll in [(ht, True, home_rolling), (at, False, away_rolling)]:
            if team in team_stats:
                td = team_stats[team]
                tm = td[(td["date"] == d) & (td["is_home"] == is_home)]
                coll.append(tm.iloc[0] if not tm.empty else pd.Series(dtype=float))
            else:
                coll.append(pd.Series(dtype=float))
    hdf = pd.DataFrame(home_rolling).reset_index(drop=True)
    adf = pd.DataFrame(away_rolling).reset_index(drop=True)
    matches["home_form_ppg"] = hdf.get("rolling_ppg", np.nan)
    matches["away_form_ppg"] = adf.get("rolling_ppg", np.nan)
    matches["home_rolling_xg_for"] = hdf.get("rolling_xg_for", np.nan)
    matches["home_rolling_xg_against"] = hdf.get("rolling_xg_against", np.nan)
    matches["away_rolling_xg_for"] = adf.get("rolling_xg_for", np.nan)
    matches["away_rolling_xg_against"] = adf.get("rolling_xg_against", np.nan)
    matches["home_rest_days"] = hdf.get("rest_days", np.nan)
    matches["away_rest_days"] = adf.get("rest_days", np.nan)
    matches["is_home"] = 1
    matches["target"] = matches["result"].map({"H": 0, "D": 1, "A": 2})
    return elo, matches, team_stats


def fit_models(train_clean: pd.DataFrame, reference_date):
    logger.info("  Fitting Elo calibrator...")
    elo_model = fit_elo_calibrator(train_clean)
    logger.info("  Fitting Dixon-Coles model...")
    dc_cutoff = train_clean["date"].max() - pd.DateOffset(months=24)
    dc_train = train_clean[train_clean["date"] >= dc_cutoff]
    dc = DixonColesModel()
    dc.fit(dc_train["home_team"].values, dc_train["away_team"].values,
           dc_train["home_goals"].values.astype(int),
           dc_train["away_goals"].values.astype(int),
           match_dates=dc_train["date"].values, reference_date=reference_date)
    logger.info("  Fitting XGBoost (market-blend)...")
    xgb_train = train_clean.dropna(subset=FEATURES_MARKET_BLEND + ["target"])
    xgb_model = GBMClassifier(feature_cols=FEATURES_MARKET_BLEND)
    xgb_model.train(xgb_train)
    return elo_model, dc, xgb_model


def compute_fixture_features(fixtures: pd.DataFrame, elo: EloRatingSystem,
                             team_stats: dict) -> pd.DataFrame:
    rows = []
    for _, fx in fixtures.iterrows():
        ht, at = fx["home_team"], fx["away_team"]
        h_elo, a_elo = elo.get_rating(ht), elo.get_rating(at)
        feat = {"home_elo": h_elo, "away_elo": a_elo, "elo_diff": h_elo - a_elo,
                "is_home": 1, "home_odds": fx["home_odds"],
                "draw_odds": fx["draw_odds"], "away_odds": fx["away_odds"]}
        for team, pfx in [(ht, "home"), (at, "away")]:
            if team in team_stats:
                ts = team_stats[team].iloc[-1]
                feat[f"{pfx}_form_ppg"] = ts.get("rolling_ppg", np.nan)
                feat[f"{pfx}_rolling_xg_for"] = ts.get("rolling_xg_for", np.nan)
                feat[f"{pfx}_rolling_xg_against"] = ts.get("rolling_xg_against", np.nan)
                ld = ts.get("date")
                feat[f"{pfx}_rest_days"] = (
                    (fx["date"] - pd.Timestamp(ld)).days if pd.notna(ld) else np.nan)
            else:
                for s in ("form_ppg", "rolling_xg_for", "rolling_xg_against", "rest_days"):
                    feat[f"{pfx}_{s}"] = np.nan
        rows.append(feat)
    return pd.DataFrame(rows)


# ── Compute predictions + consensus ──────────────────────────────────────

def _consensus_pick(m):
    """Return the consensus outcome name for a match using team names."""
    code = m["consensus"]["pick"]
    if code == "H":
        return f"{m['home_team']} to win"
    elif code == "A":
        return f"{m['away_team']} to win"
    return "Draw"


def _agreement_text(m):
    """Return a short string describing source agreement."""
    con = m["consensus"]
    n = con.get("n_sources", 4)
    no_mkt = con.get("no_market", False)
    suffix = f" — {n} sources, no market" if no_mkt else ""
    if con["unanimous"]:
        return f"all sources agree{suffix}"
    # Count per-source top picks
    counts = {"H": 0, "D": 0, "A": 0}
    for src in SOURCE_LABELS:
        p = m["probs"][src]
        if not np.any(np.isnan(p)):
            counts[OUTCOME_MAP[int(np.argmax(p))]] += 1
    parts = []
    for code, label in [("H", "Home"), ("D", "Draw"), ("A", "Away")]:
        if counts[code] > 0:
            parts.append(f"{counts[code]} {label}")
    return "sources split: " + ", ".join(parts) + suffix


def _best_secondary_pick(m):
    """Return the best secondary-market pick for a match, or None.

    Returns dict with keys: market, pick_text, prob, tag.
    """
    candidates = _secondary_candidates(m)
    if not candidates:
        return None
    best = max(candidates, key=lambda x: x[2])
    return {"market": best[0], "pick_text": best[1], "prob": best[2], "tag": best[3]}


def _secondary_candidates(m):
    """Return list of (market, pick_text, prob, tag) for all secondary markets."""
    candidates = []

    if "over_under" in m:
        ou = m["over_under"]
        p_over, p_under = ou["model_over"], ou["model_under"]
        if p_over >= p_under:
            candidates.append(("O/U", "Over 2.5 goals", p_over, ""))
        else:
            candidates.append(("O/U", "Under 2.5 goals", p_under, ""))

    if "btts" in m:
        b = m["btts"]
        if b["yes"] >= b["no"]:
            candidates.append(("BTTS", "Both teams to score", b["yes"], "(unvalidated)"))
        else:
            candidates.append(("BTTS", "One or both teams kept clean", b["no"], "(unvalidated)"))

    if "asian_handicap" in m:
        ah = m["asian_handicap"]
        line = ah["line"]
        line_str = f"{line:+.2g}" if line != 0 else "0"
        if ah["home_cover"] >= ah["home_lose"]:
            candidates.append(("AH", f"{m['home_team']} covers AH {line_str}", ah["home_cover"], ""))
        else:
            candidates.append(("AH", f"{m['away_team']} covers AH {line_str}", ah["home_lose"], ""))

    return candidates


def _best_overall_pick(m):
    """Return the single highest-confidence pick across ALL markets (H/D/A + O/U + BTTS + AH).

    Returns dict with keys: market, pick_text, prob, tag.
    """
    # H/D/A consensus pick
    candidates = []
    con = m["consensus"]
    pct = con["avg_pct"]
    candidates.append(("1X2", _consensus_pick(m), pct, ""))

    # Secondary markets
    candidates.extend(_secondary_candidates(m))

    best = max(candidates, key=lambda x: x[2])
    return {"market": best[0], "pick_text": best[1], "prob": best[2], "tag": best[3]}


def compute_all_predictions(fixtures, fx_features, elo_model, dc, xgb_model, actuals=None):
    """Return list of dicts, one per match, with all probabilities and consensus."""
    show_actuals = actuals is not None and not actuals.empty
    results = []
    for i in range(len(fixtures)):
        fx = fixtures.iloc[i]
        feat = fx_features.iloc[i:i + 1]
        is_tsdb = bool(fx.get("_from_tsdb", False))

        odds_ok = all(pd.notna(fx[c]) for c in ["home_odds", "draw_odds", "away_odds"])
        if odds_ok:
            raw = np.array([1 / fx["home_odds"], 1 / fx["draw_odds"], 1 / fx["away_odds"]])
            mkt = raw / raw.sum()
        else:
            mkt = np.array([np.nan, np.nan, np.nan])

        try:
            elo_p = elo_model.predict_proba(feat[["elo_diff"]].values)[0]
        except Exception:
            elo_p = np.array([np.nan, np.nan, np.nan])

        try:
            dc_p = np.array(dc.predict_proba(fx["home_team"], fx["away_team"]))
        except KeyError:
            dc_p = np.array([np.nan, np.nan, np.nan])

        # Skip XGBoost market-blend for TSDB fixtures (no odds to blend with)
        if is_tsdb:
            xgb_p = np.array([np.nan, np.nan, np.nan])
        else:
            try:
                xgb_p = xgb_model.predict_proba(feat)[0]
            except Exception:
                xgb_p = np.array([np.nan, np.nan, np.nan])

        all_probs = [mkt, elo_p, dc_p, xgb_p]

        # Consensus: average across sources that have valid values
        valid = [p for p in all_probs if not np.any(np.isnan(p))]
        n_sources = len(valid)
        if valid:
            avg = np.mean(valid, axis=0)
            pick_idx = int(np.argmax(avg))
            pick_code = OUTCOME_MAP[pick_idx]
            avg_pct = avg[pick_idx]
            individual_picks = [OUTCOME_MAP[int(np.argmax(p))] for p in valid]
            unanimous = len(set(individual_picks)) == 1
        else:
            pick_code = "H"
            avg_pct = 0.0
            unanimous = False

        entry = {
            "home_team": fx["home_team"], "away_team": fx["away_team"],
            "date": fx["date"], "league": fx["league"],
            "probs": {"Market": mkt, "Elo": elo_p, "Dixon-Coles": dc_p, "XGBoost": xgb_p},
            "consensus": {"pick": pick_code, "avg_pct": avg_pct, "unanimous": unanimous,
                          "n_sources": n_sources, "no_market": is_tsdb},
        }

        # ── Secondary markets from Dixon-Coles scoreline grid ──
        try:
            grid = dc.predict_scoreline_matrix(fx["home_team"], fx["away_team"])

            # Over/Under 2.5
            p_over, p_under = over_under(grid, 2.5)
            ou_mkt = {}
            if "over25_odds" in fx.index and pd.notna(fx.get("over25_odds")) and pd.notna(fx.get("under25_odds")):
                raw_ou = np.array([1 / fx["over25_odds"], 1 / fx["under25_odds"]])
                norm_ou = raw_ou / raw_ou.sum()
                ou_mkt = {"over": float(norm_ou[0]), "under": float(norm_ou[1])}
            entry["over_under"] = {
                "model_over": p_over, "model_under": p_under,
                "market": ou_mkt,
            }

            # BTTS (no market odds available — football-data.co.uk doesn't track BTTS)
            p_btts_yes, p_btts_no = btts(grid)
            entry["btts"] = {"yes": p_btts_yes, "no": p_btts_no}

            # Correct score top 5
            entry["correct_score"] = correct_score_top_n(grid, n=5)

            # Asian handicap — only if a line is present in the data
            ah_line_val = fx.get("ah_line") if "ah_line" in fx.index else None
            if pd.notna(ah_line_val):
                ah_line_val = float(ah_line_val)
                p_cover, p_push, p_lose = asian_handicap(grid, ah_line_val, side="home")
                ah_mkt = {}
                if pd.notna(fx.get("ah_home_odds")) and pd.notna(fx.get("ah_away_odds")):
                    # AH odds are 2-way (push returns stake), devig by normalising
                    raw_ah = np.array([1 / fx["ah_home_odds"], 1 / fx["ah_away_odds"]])
                    norm_ah = raw_ah / raw_ah.sum()
                    ah_mkt = {"home_cover": float(norm_ah[0]), "away_cover": float(norm_ah[1])}
                entry["asian_handicap"] = {
                    "line": ah_line_val,
                    "home_cover": p_cover, "push": p_push, "home_lose": p_lose,
                    "market": ah_mkt,
                }
        except KeyError:
            pass  # DC doesn't know one of the teams

        if show_actuals:
            act = actuals.iloc[i]
            entry["actual"] = {
                "home_goals": int(act["home_goals"]),
                "away_goals": int(act["away_goals"]),
                "result": act["result"],
            }
        results.append(entry)
    return results


# ── Terminal output ──────────────────────────────────────────────────────

def _fmt_pct(v):
    return f"{v * 100:.1f}%" if pd.notna(v) else "  -  "


def render_terminal(predictions, demo_mode):
    """Print top picks summary + stacked per-match blocks grouped by league."""
    # Top picks section
    ranked = sorted(predictions, key=lambda m: m["consensus"]["avg_pct"], reverse=True)
    print(f"\n{'━' * 64}")
    print("  TOP PICKS THIS ROUND (by consensus probability)")
    print(f"{'━' * 64}")
    for i, m in enumerate(ranked, 1):
        pick_text = _consensus_pick(m)
        pct = m["consensus"]["avg_pct"] * 100
        matchup = f"{m['home_team']} v {m['away_team']}"
        tag = "unanimous" if m["consensus"]["unanimous"] else "split"
        print(f"  {i:>2}. {pick_text:<28} ({matchup}) — {pct:.1f}% [{tag}]")

    # Other Markets section
    secondary_picks = []
    for m in predictions:
        pick = _best_secondary_pick(m)
        if pick:
            secondary_picks.append((m, pick))
    if secondary_picks:
        secondary_picks.sort(key=lambda x: x[1]["prob"], reverse=True)
        print(f"\n{'━' * 64}")
        print("  OTHER MARKETS THIS ROUND (O/U, BTTS, Asian Handicap)")
        print(f"{'━' * 64}")
        print("  Note: these run ~7pp overconfident on average (stated ~65%,")
        print("  actual ~58% in 8-season backtest). Directionally useful,")
        print("  not literally accurate. BTTS is unvalidated against odds.")
        for i, (m, pick) in enumerate(secondary_picks, 1):
            matchup = f"{m['home_team']} v {m['away_team']}"
            tag = f" {pick['tag']}" if pick["tag"] else ""
            print(f"  {i:>2}. [{pick['market']}] {pick['pick_text']:<30}"
                  f" ({matchup}) — {pick['prob']*100:.1f}%{tag}")

    # Per-league blocks
    by_league = {}
    for m in predictions:
        by_league.setdefault(m["league"], []).append(m)

    for league in sorted(by_league):
        print(f"\n{'━' * 64}")
        print(f"  {league}")
        print(f"{'━' * 64}")

        for m in by_league[league]:
            title = f"{m['home_team']} vs {m['away_team']}"
            date_str = m["date"].strftime("%Y-%m-%d")
            print(f"\n  {title:<48} {date_str}")

            no_mkt = m["consensus"].get("no_market", False)
            if no_mkt:
                print("    [Schedule from backup source — no market odds available]")

            # Consensus headline
            pick_text = _consensus_pick(m)
            pct = m["consensus"]["avg_pct"] * 100
            agree = _agreement_text(m)
            print(f"    >>> Model favors: {pick_text} — {pct:.1f}% avg ({agree})")

            active_sources = [s for s in SOURCE_LABELS
                              if not (no_mkt and s in ("Market", "XGBoost"))]
            for src in active_sources:
                p = m["probs"][src]
                h, d, a = _fmt_pct(p[0]), _fmt_pct(p[1]), _fmt_pct(p[2])
                print(f"    {src:<14} Home {h:>6}   Draw {d:>6}   Away {a:>6}")

            # ── Secondary markets ──
            if "over_under" in m:
                ou = m["over_under"]
                model_str = f"Over {ou['model_over']*100:.1f}%  Under {ou['model_under']*100:.1f}%"
                if ou["market"]:
                    mkt_str = (f"  (Market: Over {ou['market']['over']*100:.1f}%"
                               f"  Under {ou['market']['under']*100:.1f}%)")
                else:
                    mkt_str = ""
                print(f"    {'O/U 2.5':<14} {model_str}{mkt_str}")

            if "btts" in m:
                b = m["btts"]
                print(f"    {'BTTS':<14} Yes {b['yes']*100:.1f}%  No {b['no']*100:.1f}%"
                      f"  (model only — no market odds available)")

            if "correct_score" in m:
                scores = m["correct_score"]
                parts = [f"{s}: {p*100:.1f}%" for s, p in scores]
                print(f"    {'Top scores':<14} {', '.join(parts)}")

            if "asian_handicap" in m:
                ah = m["asian_handicap"]
                line = ah["line"]
                line_str = f"{line:+.2g}" if line != 0 else "0"
                model_str = (f"Home covers {ah['home_cover']*100:.1f}%"
                             f"  Push {ah['push']*100:.1f}%"
                             f"  Loses {ah['home_lose']*100:.1f}%")
                if ah["market"]:
                    mkt_str = (f"  (Market: Home {ah['market']['home_cover']*100:.1f}%"
                               f"  Away {ah['market']['away_cover']*100:.1f}%)")
                else:
                    mkt_str = ""
                print(f"    {'AH ' + line_str:<14} {model_str}{mkt_str}")

            if "actual" in m:
                act = m["actual"]
                print(f"    {'Result':<14} {act['home_goals']}-{act['away_goals']}"
                      f" ({OUTCOME_WORDS[act['result']]})")


# ── HTML output ──────────────────────────────────────────────────────────

_HTML_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       background: #f5f5f5; color: #1a1a1a; max-width: 820px; margin: 0 auto; padding: 16px 12px; }
h1 { font-size: 1.4rem; margin-bottom: 4px; }
.meta { color: #666; font-size: 0.82rem; margin-bottom: 12px; }
.demo-banner { background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
               padding: 8px 12px; margin-bottom: 14px; font-size: 0.85rem; }
.about-box { background: #f0f4ff; border: 1px solid #c7d2fe; border-radius: 8px;
             padding: 12px 16px; margin-bottom: 16px; font-size: 0.84rem; line-height: 1.45; }
.about-box strong { color: #4338ca; }
/* ── Date tabs ── */
.date-tabs { display: flex; gap: 6px; overflow-x: auto; padding: 4px 0 12px 0;
             -webkit-overflow-scrolling: touch; scrollbar-width: none; position: sticky;
             top: 0; background: #f5f5f5; z-index: 10; }
.date-tabs::-webkit-scrollbar { display: none; }
.date-tab { flex-shrink: 0; padding: 6px 14px; border-radius: 20px; border: 1px solid #ddd;
            background: #fff; font-size: 0.82rem; font-weight: 500; color: #555; cursor: pointer;
            transition: all 0.15s; white-space: nowrap; }
.date-tab:hover { border-color: #999; }
.date-tab.active { background: #1a1a1a; color: #fff; border-color: #1a1a1a; }
.date-group { display: none; }
.date-group.active { display: block; }
/* ── Top picks ── */
.top-picks { background: #fff; border-radius: 8px; border: 1px solid #ddd;
             padding: 12px 16px; margin-bottom: 16px; }
.top-picks-hdr { font-size: 0.92rem; font-weight: 600; margin-bottom: 8px; }
.top-picks ol { padding-left: 22px; }
.top-picks li { padding: 2px 0; font-size: 0.85rem; }
.top-picks .pick-name { font-weight: 600; }
.top-picks .pick-matchup { color: #666; }
.top-picks .pick-pct { font-weight: 600; }
.top-picks .pick-tag { font-size: 0.75rem; color: #888; }
.top-picks .market-tag { font-size: 0.7rem; font-weight: 600; color: #fff;
            border-radius: 3px; padding: 1px 5px; margin-right: 4px; }
.top-picks .mt-1x2 { background: #4338ca; }
.top-picks .mt-ou { background: #f59e0b; }
.top-picks .mt-btts { background: #22c55e; }
.top-picks .mt-ah { background: #3b82f6; }
.top-picks .caveat { font-size: 0.82rem; color: #666; margin-bottom: 8px; line-height: 1.4; }
.top-picks .section-note { font-size: 0.82rem; color: #666; margin-bottom: 8px; line-height: 1.4; }
/* ── League sections ── */
.league-section { margin-bottom: 16px; }
.league-hdr { font-size: 0.88rem; font-weight: 600; color: #555; padding: 8px 12px;
              background: #e9ecef; border-radius: 6px 6px 0 0; }
/* ── Match cards (compact) ── */
.match-card { background: #fff; border: 1px solid #ddd; border-top: none;
              padding: 10px 14px; cursor: pointer; transition: background 0.1s; }
.match-card:last-child { border-radius: 0 0 6px 6px; }
.league-section .match-card:first-of-type { border-top: 1px solid #ddd; }
.match-card:hover { background: #fafafa; }
.match-summary { display: flex; align-items: center; gap: 10px; }
.match-teams { flex: 1; font-size: 0.92rem; font-weight: 600; }
.match-time { font-size: 0.78rem; color: #888; flex-shrink: 0; min-width: 44px; text-align: right; }
.match-pick { font-size: 0.78rem; color: #4338ca; flex-shrink: 0; max-width: 180px;
              text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.match-pick strong { font-weight: 600; }
.backup-tag { display: inline-block; font-size: 0.65rem; font-weight: 600; color: #b45309;
              background: #fef3c7; border: 1px solid #fcd34d; border-radius: 3px;
              padding: 0 4px; margin-left: 6px; vertical-align: middle; }
.expand-icon { font-size: 0.7rem; color: #bbb; flex-shrink: 0; transition: transform 0.2s; }
.match-card.open .expand-icon { transform: rotate(180deg); }
/* ── Detail panel (hidden by default) ── */
.match-detail { display: none; padding-top: 10px; border-top: 1px solid #eee; margin-top: 10px; }
.match-card.open .match-detail { display: block; }
.consensus-badge { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 6px;
                   padding: 7px 10px; margin-bottom: 10px; font-size: 0.84rem; }
.consensus-badge strong { color: #4338ca; }
.consensus-badge .agree-tag { font-size: 0.78rem; color: #666; }
.source-row { display: flex; align-items: center; margin-bottom: 5px; }
.source-label { width: 90px; font-size: 0.78rem; font-weight: 500; color: #555; flex-shrink: 0; }
.bar-wrap { flex: 1; display: flex; height: 20px; border-radius: 4px; overflow: hidden; }
.seg { display: flex; align-items: center; justify-content: center;
       font-size: 0.68rem; font-weight: 600; color: #fff; min-width: 26px; }
.seg-home { background: #3b82f6; }
.seg-draw { background: #9ca3af; }
.seg-away { background: #ef4444; }
.secondary-markets { margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee; }
.secondary-markets h4 { font-size: 0.78rem; font-weight: 600; color: #555; margin: 6px 0 3px 0; }
.secondary-markets h4:first-child { margin-top: 0; }
.two-bar { display: flex; height: 20px; border-radius: 4px; overflow: hidden; margin-bottom: 2px; }
.two-bar .seg-over { background: #f59e0b; }
.two-bar .seg-under { background: #6366f1; }
.two-bar .seg-yes { background: #22c55e; }
.two-bar .seg-no { background: #ef4444; }
.two-bar .seg-cover { background: #3b82f6; }
.two-bar .seg-push { background: #9ca3af; }
.two-bar .seg-lose { background: #ef4444; }
.bar-label { font-size: 0.7rem; color: #888; margin-bottom: 4px; }
.score-list { display: flex; flex-wrap: wrap; gap: 5px; margin: 3px 0; }
.score-chip { background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 4px;
              padding: 2px 7px; font-size: 0.75rem; font-weight: 500; }
.score-chip .pct { color: #666; font-weight: 400; }
.result-line { margin-top: 8px; padding-top: 7px; border-top: 1px solid #eee;
               font-size: 0.84rem; font-weight: 500; }
.correct { border-left: 4px solid #22c55e; }
.incorrect { border-left: 4px solid #ef4444; }
.footer { margin-top: 24px; padding-top: 14px; border-top: 1px solid #ccc;
          font-size: 0.78rem; color: #888; }
.unvalidated { font-size: 0.72rem; color: #b45309; font-style: italic; }
/* ── Archive navigation ── */
.archive-nav { display: flex; justify-content: space-between; align-items: center;
               padding: 8px 12px; margin-bottom: 12px; background: #fff; border: 1px solid #ddd;
               border-radius: 8px; font-size: 0.88rem; }
.archive-nav-link { color: #4338ca; text-decoration: none; font-weight: 500; min-width: 80px; }
.archive-nav-link:first-child { text-align: left; }
.archive-nav-link:last-child { text-align: right; }
.archive-nav-link:hover { text-decoration: underline; }
.archive-nav-link.disabled { visibility: hidden; }
.archive-nav-current { font-weight: 600; text-align: center; }
.archive-banner { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px;
                  padding: 8px 12px; margin-bottom: 14px; font-size: 0.85rem; text-align: center; }
.archive-banner a { color: #4338ca; font-weight: 600; }
"""

_HTML_JS = """\
function switchTab(dateKey) {
  document.querySelectorAll('.date-tab').forEach(t => t.classList.toggle('active', t.dataset.date === dateKey));
  document.querySelectorAll('.date-group').forEach(g => g.classList.toggle('active', g.id === 'day-' + dateKey));
  // Update top picks visibility
  document.querySelectorAll('.top-pick-item').forEach(li => {
    li.style.display = (!li.dataset.date || li.dataset.date === dateKey) ? '' : 'none';
  });
  // Re-number visible picks
  let n = 0;
  document.querySelectorAll('.top-pick-item').forEach(li => {
    if (li.style.display !== 'none') { n++; li.setAttribute('value', n); }
  });
}
function toggleCard(el) {
  el.closest('.match-card').classList.toggle('open');
}
document.addEventListener('DOMContentLoaded', function() {
  var first = document.querySelector('.date-tab');
  if (first) switchTab(first.dataset.date);
});
"""


def _bar_segments(probs):
    """Generate HTML for a 3-segment probability bar."""
    parts = []
    labels = ["H", "D", "A"]
    classes = ["seg-home", "seg-draw", "seg-away"]
    for v, lbl, cls in zip(probs, labels, classes):
        if pd.isna(v):
            continue
        pct = v * 100
        width = max(pct, 4)
        parts.append(
            f'<div class="seg {cls}" style="width:{width:.1f}%">'
            f'{pct:.0f}%</div>'
        )
    return "".join(parts)


def _render_match_detail(m, body):
    """Append expanded detail HTML for a single match card."""
    has_actual = "actual" in m
    no_mkt = m["consensus"].get("no_market", False)

    pick = html_mod.escape(_consensus_pick(m))
    pct = m["consensus"]["avg_pct"] * 100
    agree = html_mod.escape(_agreement_text(m))
    body.append(
        f'<div class="consensus-badge">Model favors: <strong>{pick}</strong>'
        f' &mdash; {pct:.1f}% avg '
        f'<span class="agree-tag">({agree})</span></div>')

    active_sources = [s for s in SOURCE_LABELS
                      if not (no_mkt and s in ("Market", "XGBoost"))]
    for src in active_sources:
        p = m["probs"][src]
        border_cls = ""
        if has_actual and not np.any(np.isnan(p)):
            pred = OUTCOME_MAP[int(np.argmax(p))]
            border_cls = " correct" if pred == m["actual"]["result"] else " incorrect"
        body.append(f'<div class="source-row{border_cls}">')
        body.append(f'<div class="source-label">{html_mod.escape(src)}</div>')
        body.append(f'<div class="bar-wrap">{_bar_segments(p)}</div>')
        body.append("</div>")

    has_secondary = any(k in m for k in ("over_under", "btts", "correct_score", "asian_handicap"))
    if has_secondary:
        body.append('<div class="secondary-markets">')
        if "over_under" in m:
            ou = m["over_under"]
            body.append('<h4>Over/Under 2.5 Goals</h4>')
            p_o, p_u = ou["model_over"], ou["model_under"]
            w_o, w_u = max(p_o * 100, 4), max(p_u * 100, 4)
            body.append(
                f'<div class="two-bar">'
                f'<div class="seg seg-over" style="width:{w_o:.1f}%">O {p_o*100:.0f}%</div>'
                f'<div class="seg seg-under" style="width:{w_u:.1f}%">U {p_u*100:.0f}%</div>'
                f'</div>')
            if ou["market"]:
                body.append(
                    f'<div class="bar-label">Market: Over {ou["market"]["over"]*100:.1f}%'
                    f' / Under {ou["market"]["under"]*100:.1f}%</div>')
            else:
                body.append('<div class="bar-label">Model only</div>')
        if "btts" in m:
            b = m["btts"]
            body.append('<h4>Both Teams to Score</h4>')
            w_y, w_n = max(b["yes"] * 100, 4), max(b["no"] * 100, 4)
            body.append(
                f'<div class="two-bar">'
                f'<div class="seg seg-yes" style="width:{w_y:.1f}%">Yes {b["yes"]*100:.0f}%</div>'
                f'<div class="seg seg-no" style="width:{w_n:.1f}%">No {b["no"]*100:.0f}%</div>'
                f'</div>')
            body.append('<div class="bar-label">Model only &mdash; no BTTS odds in data source</div>')
        if "correct_score" in m:
            body.append('<h4>Most Likely Scores</h4>')
            body.append('<div class="score-list">')
            for score, prob in m["correct_score"]:
                body.append(
                    f'<div class="score-chip">{html_mod.escape(score)}'
                    f' <span class="pct">{prob*100:.1f}%</span></div>')
            body.append('</div>')
        if "asian_handicap" in m:
            ah = m["asian_handicap"]
            line = ah["line"]
            line_str = f"{line:+.2g}" if line != 0 else "0"
            body.append(f'<h4>Asian Handicap ({html_mod.escape(line_str)})</h4>')
            w_c = max(ah["home_cover"] * 100, 4)
            w_p = max(ah["push"] * 100, 2) if ah["push"] > 0.005 else 0
            w_l = max(ah["home_lose"] * 100, 4)
            seg_parts = (
                f'<div class="seg seg-cover" style="width:{w_c:.1f}%">'
                f'Cover {ah["home_cover"]*100:.0f}%</div>')
            if w_p > 0:
                seg_parts += (
                    f'<div class="seg seg-push" style="width:{w_p:.1f}%">'
                    f'Push {ah["push"]*100:.0f}%</div>')
            seg_parts += (
                f'<div class="seg seg-lose" style="width:{w_l:.1f}%">'
                f'Lose {ah["home_lose"]*100:.0f}%</div>')
            body.append(f'<div class="two-bar">{seg_parts}</div>')
            if ah["market"]:
                body.append(
                    f'<div class="bar-label">Market: Home {ah["market"]["home_cover"]*100:.1f}%'
                    f' / Away {ah["market"]["away_cover"]*100:.1f}%</div>')
            else:
                body.append('<div class="bar-label">Model only</div>')
        body.append('</div>')

    if has_actual:
        act = m["actual"]
        rw = OUTCOME_WORDS[act["result"]]
        body.append(
            f'<div class="result-line">Result: {act["home_goals"]}-{act["away_goals"]}'
            f' ({rw})</div>')


def _render_archive_nav(nav):
    """Return HTML for the prev/next archive navigation bar."""
    parts = []
    parts.append('<div class="archive-nav">')

    if nav.get("prev_date"):
        pd_label = datetime.strptime(nav["prev_date"], "%Y-%m-%d").strftime("%b %d")
        prev_href = f"archive/predictions-{nav['prev_date']}.html" if not nav.get("is_archive") else f"predictions-{nav['prev_date']}.html"
        parts.append(f'<a class="archive-nav-link" href="{prev_href}">&larr; {pd_label}</a>')
    else:
        parts.append('<span class="archive-nav-link disabled"></span>')

    cur_label = datetime.strptime(nav["current_date"], "%Y-%m-%d").strftime("%b %d")
    if nav.get("is_archive"):
        parts.append(f'<span class="archive-nav-current">{cur_label}</span>')
    else:
        parts.append(f'<span class="archive-nav-current">{cur_label} (latest)</span>')

    if nav.get("next_date"):
        nd_label = datetime.strptime(nav["next_date"], "%Y-%m-%d").strftime("%b %d")
        next_href = f"archive/predictions-{nav['next_date']}.html" if not nav.get("is_archive") else f"predictions-{nav['next_date']}.html"
        parts.append(f'<a class="archive-nav-link" href="{next_href}">&rarr; {nd_label}</a>')
    else:
        parts.append('<span class="archive-nav-link disabled"></span>')

    parts.append('</div>')

    if nav.get("is_archive"):
        parts.append(
            f'<div class="archive-banner">Archived prediction round from {cur_label} &mdash; '
            f'<a href="../index.html">view latest predictions</a></div>')

    return "\n".join(parts)


def render_html(predictions, demo_mode, as_of_date, archive_nav=None):
    """Write a self-contained HTML report and return the output path.

    archive_nav: optional dict with keys 'prev_date', 'next_date', 'current_date',
                 'is_archive' for inter-page navigation.
    """
    out_dir = ROOT_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    date_label = as_of_date if demo_mode else datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"predictions_{date_label}.html"

    # Group by date → league → matches
    by_date = {}
    for m in predictions:
        dk = m["date"].strftime("%Y-%m-%d")
        by_date.setdefault(dk, []).append(m)
    sorted_dates = sorted(by_date.keys())

    # Build date tab labels
    today = datetime.now().date()
    def _tab_label(ds):
        d = datetime.strptime(ds, "%Y-%m-%d").date()
        if d == today:
            return "Today"
        if d == today + timedelta(days=1):
            return "Tomorrow"
        return d.strftime("%a %d %b")

    body = []

    # Archive navigation bar (injected at top)
    if archive_nav:
        body.append(_render_archive_nav(archive_nav))

    # Header
    body.append("<h1>Match Predictions</h1>")
    body.append(f'<p class="meta">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} '
                f'&middot; {len(predictions)} matches across {len(sorted_dates)} day(s)</p>')
    if demo_mode:
        body.append(
            f'<div class="demo-banner">[DEMO MODE] Using historical date '
            f'{html_mod.escape(as_of_date)} as stand-in fixtures &mdash; NOT live data</div>')

    # About box
    body.append(
        '<div class="about-box">'
        '<strong>What is this?</strong> '
        'Automated football match predictions for the top European leagues, '
        'built from three statistical models (Elo ratings, Dixon-Coles, and XGBoost) '
        'trained on eight seasons of historical data. '
        'Updates Mon/Tue/Fri. Click any match to expand full detail. '
        '<strong>Important:</strong> these are model estimates, not betting advice. '
        'None of these models consistently beat the market.'
        '</div>')

    # Date tabs
    body.append('<div class="date-tabs">')
    for ds in sorted_dates:
        label = html_mod.escape(_tab_label(ds))
        n = len(by_date[ds])
        body.append(f'<div class="date-tab" data-date="{ds}" onclick="switchTab(\'{ds}\')">'
                    f'{label} <span style="opacity:0.6">({n})</span></div>')
    body.append('</div>')

    # ── Primary list: Best Pick Every Match (All Markets) ──
    all_market_picks = []
    for m in predictions:
        best = _best_overall_pick(m)
        all_market_picks.append((m, best))
    all_market_picks.sort(key=lambda x: x[1]["prob"], reverse=True)

    body.append('<div class="top-picks">')
    body.append('<div class="top-picks-hdr">Best Pick Every Match (All Markets)</div>')
    body.append(
        '<p class="caveat">A full 8-season backtest of this exact approach: raises hit rate '
        'to ~58% (vs ~51% for match-result-only picks), but displayed confidence runs ~7 '
        'points hot on average (stated ~65%, actual ~58%). About 83% of entries here are '
        'O/U or BTTS &mdash; single-model estimates, not full 4-source consensus. BTTS '
        'specifically has never been checked against real bookmaker odds.</p>')
    body.append('<ol>')
    for m, best in all_market_picks:
        dk = m["date"].strftime("%Y-%m-%d")
        market = html_mod.escape(best["market"])
        market_cls = {"1X2": "1x2", "O/U": "ou", "BTTS": "btts", "AH": "ah"}.get(best["market"], "1x2")
        pick_text = html_mod.escape(best["pick_text"])
        matchup = html_mod.escape(f"{m['home_team']} v {m['away_team']}")
        pct = best["prob"] * 100
        unvalidated = ' <span class="pick-tag">(unvalidated)</span>' if best["tag"] else ""
        body.append(
            f'<li class="top-pick-item" data-date="{dk}">'
            f'<span class="market-tag mt-{market_cls}">{market}</span>'
            f'<span class="pick-name">{pick_text}</span> '
            f'<span class="pick-matchup">({matchup})</span> '
            f'&mdash; <span class="pick-pct">{pct:.1f}%</span>'
            f'{unvalidated}</li>')
    body.append('</ol></div>')

    # ── Secondary list: Best-Calibrated Picks (Match Result Only) ──
    ranked_hda = sorted(predictions, key=lambda m: m["consensus"]["avg_pct"], reverse=True)
    body.append('<div class="top-picks">')
    body.append('<div class="top-picks-hdr">Best-Calibrated Picks (Match Result Only)</div>')
    body.append(
        '<p class="section-note">This list alone is well-calibrated in backtesting '
        '(states ~50%, actual ~51%) &mdash; the safer, more honest-confidence view.</p>')
    body.append('<ol>')
    for m in ranked_hda:
        dk = m["date"].strftime("%Y-%m-%d")
        pick = html_mod.escape(_consensus_pick(m))
        matchup = html_mod.escape(f"{m['home_team']} v {m['away_team']}")
        pct = m["consensus"]["avg_pct"] * 100
        tag = "unanimous" if m["consensus"]["unanimous"] else "split"
        body.append(
            f'<li class="top-pick-item" data-date="{dk}">'
            f'<span class="market-tag mt-1x2">1X2</span>'
            f'<span class="pick-name">{pick}</span> '
            f'<span class="pick-matchup">({matchup})</span> '
            f'&mdash; <span class="pick-pct">{pct:.1f}%</span> '
            f'<span class="pick-tag">[{tag}]</span></li>')
    body.append('</ol></div>')

    # Date groups
    for ds in sorted_dates:
        day_matches = by_date[ds]
        body.append(f'<div class="date-group" id="day-{ds}">')

        # Group by league within this date
        day_by_league = {}
        for m in day_matches:
            day_by_league.setdefault(m["league"], []).append(m)

        for league in sorted(day_by_league):
            body.append('<div class="league-section">')
            body.append(f'<div class="league-hdr">{html_mod.escape(league)}</div>')

            for m in day_by_league[league]:
                ht = html_mod.escape(m["home_team"])
                at = html_mod.escape(m["away_team"])
                no_mkt = m["consensus"].get("no_market", False)
                kick_time = m["date"].strftime("%H:%M")
                pick_text = html_mod.escape(_consensus_pick(m))
                pct = m["consensus"]["avg_pct"] * 100

                body.append('<div class="match-card">')
                body.append(f'<div class="match-summary" onclick="toggleCard(this)">')
                backup_tag = '<span class="backup-tag">NO ODDS</span>' if no_mkt else ''
                body.append(f'<div class="match-teams">{ht} v {at}{backup_tag}</div>')
                body.append(f'<div class="match-pick"><strong>{pct:.0f}%</strong> {pick_text}</div>')
                body.append(f'<div class="match-time">{kick_time}</div>')
                body.append('<div class="expand-icon">&#9660;</div>')
                body.append('</div>')

                # Detail panel (hidden by default)
                body.append('<div class="match-detail">')
                if no_mkt:
                    body.append(
                        '<div class="demo-banner" style="margin-bottom:8px;font-size:0.8rem">'
                        'Schedule from backup source &mdash; no market odds available'
                        '</div>')
                _render_match_detail(m, body)
                body.append('</div>')  # match-detail
                body.append('</div>')  # match-card

            body.append('</div>')  # league-section

        body.append('</div>')  # date-group

    # Footer
    body.append('<div class="footer">')
    body.append("<p>Probabilities are model estimates, not recommendations. "
                "Highest probability = most likely outcome, not necessarily value "
                "against the bookmaker's price.</p>")
    if demo_mode:
        body.append(
            "<p>Actual results shown for gut-check only &mdash; not a backtest.</p>")
    body.append("</div>")

    html_content = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>Predictions {html_mod.escape(date_label)}</title>\n"
        f"<style>{_HTML_CSS}</style>\n"
        "</head>\n<body>\n"
        + "\n".join(body)
        + f"\n<script>{_HTML_JS}</script>\n"
        "</body>\n</html>\n"
    )

    out_path.write_text(html_content, encoding="utf-8")
    return out_path


def build_archive(html_path):
    """Copy current HTML into archive/, update index.json, and inject nav into all pages.

    Returns the archive directory path.
    """
    archive_dir = ROOT_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)

    # Determine today's date label from the HTML filename (predictions_YYYY-MM-DD.html)
    m = re.search(r"predictions_(\d{4}-\d{2}-\d{2})\.html", html_path.name)
    if not m:
        logger.warning("Could not parse date from %s — skipping archive", html_path.name)
        return archive_dir
    current_date = m.group(1)

    # Copy into archive with hyphenated name
    archive_file = archive_dir / f"predictions-{current_date}.html"
    shutil.copy2(html_path, archive_file)
    logger.info("Archived: %s", archive_file)

    # Build/update index.json — sorted list of all archived dates
    index_path = archive_dir / "index.json"
    if index_path.exists():
        existing = json.loads(index_path.read_text())
    else:
        existing = []
    dates = sorted(set(existing) | {current_date})
    index_path.write_text(json.dumps(dates, indent=2) + "\n")
    logger.info("Archive index: %s dates", len(dates))

    # Now inject navigation into all archive pages + the live page
    for i, d in enumerate(dates):
        nav = {
            "prev_date": dates[i - 1] if i > 0 else None,
            "next_date": dates[i + 1] if i < len(dates) - 1 else None,
            "current_date": d,
            "is_archive": True,
        }
        _inject_nav_into_html(archive_dir / f"predictions-{d}.html", nav)

    # Inject nav into the live page (index.html will be copied from this)
    cur_idx = dates.index(current_date)
    live_nav = {
        "prev_date": dates[cur_idx - 1] if cur_idx > 0 else None,
        "next_date": None,
        "current_date": current_date,
        "is_archive": False,
    }
    _inject_nav_into_html(html_path, live_nav)

    return archive_dir


def _inject_nav_into_html(html_path, nav):
    """Replace or insert archive nav into an existing HTML file."""
    if not html_path.exists():
        return
    content = html_path.read_text(encoding="utf-8")
    nav_html = _render_archive_nav(nav)

    # Remove any existing archive nav + banner
    content = re.sub(
        r'<div class="archive-nav">.*?</div>\s*(?:<div class="archive-banner">.*?</div>)?\s*',
        '', content, flags=re.DOTALL)

    # Insert after <body>
    content = content.replace("<body>\n", "<body>\n" + nav_html + "\n", 1)
    html_path.write_text(content, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare model probabilities against market odds for upcoming fixtures.")
    parser.add_argument("--as-of", metavar="YYYY-MM-DD",
                        help="Demo mode: use matches from this past date as stand-in fixtures.")
    parser.add_argument("--html", action="store_true",
                        help="Also generate an HTML report in output/.")
    parser.add_argument("--archive", action="store_true",
                        help="Archive the HTML report (implies --html).")
    args = parser.parse_args()
    if args.archive:
        args.html = True

    demo_mode = args.as_of is not None
    actuals = None

    print("=" * 64)
    print("UPCOMING FIXTURE PROBABILITY COMPARISON")
    print("=" * 64)

    if demo_mode:
        print(
            f"\n[DEMO MODE] Using historical date {args.as_of} as a stand-in"
            f" for upcoming fixtures — NOT live data"
        )
        fixtures, actuals = fetch_demo_fixtures(args.as_of)
        if fixtures.empty:
            print(f"\nNo matches found on {args.as_of} for the {len(TARGET_DIVS)} target leagues.")
            return
        print(f"Found {len(fixtures)} matches on {args.as_of} across: "
              f"{', '.join(sorted(fixtures['league'].unique()))}")
    else:
        print("\nFetching upcoming fixtures from football-data.co.uk...")
        fixtures = fetch_fixtures()
        if "_from_tsdb" not in fixtures.columns and not fixtures.empty:
            fixtures["_from_tsdb"] = False

        # Check which leagues are missing from football-data.co.uk
        fd_leagues = set(fixtures["league"].unique()) if not fixtures.empty else set()
        missing_leagues = sorted(set(LEAGUE_NAMES) - fd_leagues)

        if missing_leagues:
            print(f"  football-data.co.uk missing: {', '.join(missing_leagues)}")
            print(f"  Checking TheSportsDB backup for {len(missing_leagues)} league(s)...")
            tsdb_fx = fetch_tsdb_fixtures(missing_leagues)
            if not tsdb_fx.empty:
                print(f"  TheSportsDB found {len(tsdb_fx)} fixture(s) across: "
                      f"{', '.join(sorted(tsdb_fx['league'].unique()))}")
                fixtures = pd.concat([fixtures, tsdb_fx], ignore_index=True)
            else:
                print("  TheSportsDB returned no upcoming fixtures either.")

        if fixtures.empty:
            print(f"\nNo upcoming fixtures found for the {len(TARGET_DIVS)} target leagues.")
            print("This is expected during the off-season (June-August).")
            print("Tip: use --as-of YYYY-MM-DD to demo against a past matchday.")
            return

        n_tsdb = int(fixtures["_from_tsdb"].sum()) if "_from_tsdb" in fixtures.columns else 0
        n_fd = len(fixtures) - n_tsdb
        parts = []
        if n_fd:
            parts.append(f"{n_fd} from football-data.co.uk")
        if n_tsdb:
            parts.append(f"{n_tsdb} from TheSportsDB (no odds)")
        print(f"Found {len(fixtures)} upcoming fixtures ({', '.join(parts)}) across: "
              f"{', '.join(sorted(fixtures['league'].unique()))}")

    # Load historical data, cut off strictly before fixture date
    cfg = load_config()
    hist = load_historical_data()
    if demo_mode:
        cutoff = pd.Timestamp(args.as_of)
        train_data = hist[hist["date"] < cutoff]
        print(f"\nTraining on {len(train_data)} matches strictly before {args.as_of}")
    else:
        train_data = hist

    print("Fitting models...")
    elo, train_df, team_stats = build_elo_and_features(train_data, cfg)
    train_clean = train_df.dropna(subset=["target", "elo_diff"])
    ref_date = pd.Timestamp(args.as_of) if demo_mode else pd.Timestamp.now()
    elo_model, dc, xgb_model = fit_models(train_clean, ref_date)

    # Compute features and predictions ONCE
    fx_features = compute_fixture_features(fixtures, elo, team_stats)
    predictions = compute_all_predictions(
        fixtures, fx_features, elo_model, dc, xgb_model, actuals=actuals)

    # Render terminal
    render_terminal(predictions, demo_mode)

    print(f"\n{'=' * 64}")
    print("Probabilities shown are model estimates, not recommendations.")
    print("Highest probability = what the model considers most likely, not")
    print("value against the bookmaker's price.")
    if demo_mode:
        print("Actual results shown for gut-check only — this is not a backtest.")
    print("=" * 64)

    # Render HTML if requested
    if args.html:
        html_path = render_html(predictions, demo_mode, args.as_of or "")
        print(f"\nHTML report written to: {html_path}")

        if args.archive:
            archive_dir = build_archive(html_path)
            print(f"Archive updated: {archive_dir}")


if __name__ == "__main__":
    main()
