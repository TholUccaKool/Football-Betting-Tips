"""Pull match-level xG data from Understat via soccerdata."""

import logging

import soccerdata as sd

from src.utils.io import load_config, get_raw_dir, season_range

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Understat league names
LEAGUE_CODE_TO_UNDERSTAT = {
    "E0": "ENG-Premier League",
    "SP1": "ESP-La Liga",
    "I1": "ITA-Serie A",
    "D1": "GER-Bundesliga",
    "F1": "FRA-Ligue 1",
}


def ingest(cfg: dict | None = None) -> None:
    """Download Understat xG data for all configured leagues/seasons."""
    if cfg is None:
        cfg = load_config()

    output_dir = get_raw_dir() / "understat"
    output_dir.mkdir(parents=True, exist_ok=True)

    seasons = season_range(cfg)

    for league_code in cfg["leagues"]:
        league_name = LEAGUE_CODE_TO_UNDERSTAT.get(league_code)
        if league_name is None:
            logger.warning(f"No Understat mapping for {league_code}, skipping")
            continue

        for season in seasons:
            season_str = str(season)[-2:] + str(season + 1)[-2:]
            logger.info(f"Fetching Understat {league_name} {season_str}")
            try:
                us = sd.Understat(league_name, [season])
                df = us.read_schedule()
                out_path = output_dir / f"{league_code}_{season}.parquet"
                df.to_parquet(out_path)
                logger.info(f"  Saved {len(df)} matches -> {out_path}")
            except Exception as e:
                logger.warning(f"  Skipped {league_name} {season}: {e}")


if __name__ == "__main__":
    ingest()
