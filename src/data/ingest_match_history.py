"""Pull historical match results + closing odds from football-data.co.uk via soccerdata."""

import logging

import soccerdata as sd

from src.data._soccerdata_patch import patch_soccerdata_session  # fix tls_requests 503s

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


def ingest(cfg: dict | None = None) -> None:
    """Download match history for all configured leagues/seasons."""
    if cfg is None:
        cfg = load_config()

    output_dir = get_raw_dir() / "match_history"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.resolve()}")

    leagues = [LEAGUE_CODE_TO_NAME[c] for c in cfg["leagues"]]
    seasons = season_range(cfg)

    saved = []
    skipped = []

    for league_code, league_name in zip(cfg["leagues"], leagues):
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

    logger.info("=" * 60)
    logger.info(f"SUMMARY: {len(saved)} saved, {len(skipped)} skipped")
    for label, n in saved:
        logger.info(f"  OK   {label} ({n} matches)")
    for label, reason in skipped:
        logger.info(f"  SKIP {label}: {reason}")
    logger.info("=" * 60)


if __name__ == "__main__":
    ingest()
