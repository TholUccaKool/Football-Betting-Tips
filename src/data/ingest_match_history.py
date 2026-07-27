"""Pull historical match results + closing odds from football-data.co.uk.

Core leagues (E0, SP1, I1, D1, F1) are fetched via soccerdata, which handles
caching and auto-refresh of in-progress seasons.

Secondary leagues (E1, SP2, N1, P1, etc.) are downloaded directly from
football-data.co.uk since soccerdata doesn't have built-in mappings for them.
"""

import io
import logging

import pandas as pd
import requests
import soccerdata as sd

from src.data._soccerdata_patch import patch_soccerdata_session, _BROWSER_UA

patch_soccerdata_session()

from src.utils.io import load_config, get_raw_dir, season_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# soccerdata uses league names, not codes — map config codes to soccerdata names
LEAGUE_CODE_TO_NAME = {
    "E0": "ENG-Premier League",
    "SP1": "ESP-La Liga",
    "I1": "ITA-Serie A",
    "D1": "GER-Bundesliga",
    "F1": "FRA-Ligue 1",
}

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"


def _fetch_direct(league_code: str, season: int, output_dir) -> tuple[str, int] | None:
    """Download a league/season CSV directly from football-data.co.uk.

    Returns (label, n_rows) on success, None on failure.
    """
    season_str = str(season)[-2:] + str(season + 1)[-2:]
    url = FOOTBALL_DATA_URL.format(season=season_str, code=league_code)
    label = f"{league_code} {season_str}"

    try:
        resp = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return None

    try:
        df = pd.read_csv(io.BytesIO(resp.content), encoding="utf-8-sig")
    except Exception:
        return None

    if df.empty or len(df) < 5:
        return None

    # Standardise column names to match soccerdata output format
    col_map = {}
    for col in df.columns:
        low = col.strip()
        if low == "HomeTeam":
            col_map[col] = "home_team"
        elif low == "AwayTeam":
            col_map[col] = "away_team"
        elif low == "Date":
            col_map[col] = "date"
    df = df.rename(columns=col_map)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)

    out_path = output_dir / f"{league_code}_{season}.parquet"
    df.to_parquet(out_path)
    return label, len(df)


def ingest(cfg: dict | None = None) -> None:
    """Download match history for all configured leagues/seasons."""
    if cfg is None:
        cfg = load_config()

    output_dir = get_raw_dir() / "match_history"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.resolve()}")

    seasons = season_range(cfg)
    saved = []
    skipped = []

    # --- Core leagues via soccerdata (handles caching for in-progress seasons) ---
    for league_code in cfg["leagues"]:
        league_name = LEAGUE_CODE_TO_NAME.get(league_code)
        if league_name is None:
            logger.warning(f"No soccerdata mapping for core league {league_code}, skipping")
            continue
        for season in seasons:
            season_str = str(season)[-2:] + str(season + 1)[-2:]
            label = f"{league_name} {season_str}"
            logger.info(f"Fetching {label}")
            try:
                mh = sd.MatchHistory(league_name, [season])
                df = mh.read_games()
                out_path = output_dir / f"{league_code}_{season}.parquet"
                df.to_parquet(out_path)
                logger.info(f"  Saved {len(df)} matches -> {out_path.name}")
                saved.append((label, len(df)))
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                logger.warning(f"  Skipped {label}: {reason}")
                skipped.append((label, reason))

    # --- Secondary leagues via direct download ---
    secondary = cfg.get("secondary_leagues", [])
    if secondary:
        logger.info(f"\nFetching {len(secondary)} secondary leagues via direct download...")
    for league_code in secondary:
        for season in seasons:
            season_str = str(season)[-2:] + str(season + 1)[-2:]
            label = f"{league_code} {season_str}"
            logger.info(f"Fetching {label}")
            result = _fetch_direct(league_code, season, output_dir)
            if result is not None:
                label, n = result
                logger.info(f"  Saved {n} matches -> {league_code}_{season}.parquet")
                saved.append((label, n))
            else:
                logger.warning(f"  Skipped {label}: not available or empty")
                skipped.append((label, "not available"))

    logger.info("=" * 60)
    logger.info(f"SUMMARY: {len(saved)} saved, {len(skipped)} skipped")
    for label, n in saved:
        logger.info(f"  OK   {label} ({n} matches)")
    for label, reason in skipped:
        logger.info(f"  SKIP {label}: {reason}")
    logger.info("=" * 60)


if __name__ == "__main__":
    ingest()
