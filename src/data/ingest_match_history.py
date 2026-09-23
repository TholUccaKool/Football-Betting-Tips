"""Pull historical match results + closing odds from football-data.co.uk.

Core leagues (E0, SP1, I1, D1, F1) are fetched via soccerdata, which handles
caching and auto-refresh of in-progress seasons.

Secondary leagues (E1, SP2, N1, P1, etc.) are downloaded directly from
football-data.co.uk since soccerdata doesn't have built-in mappings for them.
"""

import io
import logging
import sys
import time

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

RETRY_WAITS = (5, 15, 30)       # backoff between attempts, seconds
MAX_CONSECUTIVE_FAILURES = 3    # abort the whole ingest after this many fetches fail in a row


class IngestAborted(RuntimeError):
    """Raised when the data source looks down (too many consecutive failures)."""


def _with_retries(fn, label: str):
    """Call fn() with backoff retries. Returns fn's result or raises the last error."""
    last_exc = None
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise after retries
            last_exc = exc
            if attempt < len(RETRY_WAITS):
                wait = RETRY_WAITS[attempt]
                logger.warning(f"  {label}: {type(exc).__name__}: {exc} - retrying in {wait}s "
                               f"({attempt + 1}/{len(RETRY_WAITS)})")
                time.sleep(wait)
    raise last_exc


def _fetch_direct(league_code: str, season: int, output_dir) -> tuple[str, int] | None:
    """Download a league/season CSV directly from football-data.co.uk.

    Returns (label, n_rows) on success, None on failure.
    """
    season_str = str(season)[-2:] + str(season + 1)[-2:]
    url = FOOTBALL_DATA_URL.format(season=season_str, code=league_code)
    label = f"{league_code} {season_str}"

    def _get():
        resp = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=30)
        if resp.status_code == 404:
            return None  # season file genuinely not published yet - not retryable
        resp.raise_for_status()
        return resp

    resp = _with_retries(_get, label)   # raises on persistent HTTP/network failure
    if resp is None:
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
    consecutive_failures = 0

    def _note_failure(label, reason):
        nonlocal consecutive_failures
        consecutive_failures += 1
        skipped.append((label, reason))
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise IngestAborted(
                f"{consecutive_failures} consecutive fetch failures (last: {label}: {reason}) "
                f"- football-data.co.uk looks unavailable")

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
                df = _with_retries(
                    lambda: sd.MatchHistory(league_name, [season]).read_games(), label)
                out_path = output_dir / f"{league_code}_{season}.parquet"
                df.to_parquet(out_path)
                logger.info(f"  Saved {len(df)} matches -> {out_path.name}")
                saved.append((label, len(df)))
                consecutive_failures = 0
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                logger.warning(f"  Skipped {label}: {reason}")
                _note_failure(label, reason)

    # --- Secondary leagues via direct download ---
    secondary = cfg.get("secondary_leagues", [])
    if secondary:
        logger.info(f"\nFetching {len(secondary)} secondary leagues via direct download...")
    for league_code in secondary:
        for season in seasons:
            season_str = str(season)[-2:] + str(season + 1)[-2:]
            label = f"{league_code} {season_str}"
            logger.info(f"Fetching {label}")
            try:
                result = _fetch_direct(league_code, season, output_dir)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                logger.warning(f"  Skipped {label}: {reason}")
                _note_failure(label, reason)
                continue
            if result is not None:
                label, n = result
                logger.info(f"  Saved {n} matches -> {league_code}_{season}.parquet")
                saved.append((label, n))
                consecutive_failures = 0
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

    if not saved:
        # GitHub Actions annotation so the failure is visible in the run summary
        print("::error::Match-history ingest saved 0 files - football-data.co.uk unavailable?")
        sys.exit(1)
    if skipped:
        print(f"::warning::Match-history ingest skipped {len(skipped)} of "
              f"{len(saved) + len(skipped)} league-seasons")


if __name__ == "__main__":
    try:
        ingest()
    except IngestAborted as exc:
        logger.error(str(exc))
        print(f"::error::Match-history ingest aborted: {exc}")
        sys.exit(1)
