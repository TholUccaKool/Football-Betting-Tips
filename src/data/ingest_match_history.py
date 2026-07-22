"""Pull historical match results + closing odds from football-data.co.uk via soccerdata."""

import logging
from pathlib import Path

import soccerdata as sd

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

    leagues = [LEAGUE_CODE_TO_NAME[c] for c in cfg["leagues"]]
    seasons = season_range(cfg)

    for league_code, league_name in zip(cfg["leagues"], leagues):
        for season in seasons:
            season_str = str(season)[-2:] + str(season + 1)[-2:]
            logger.info(f"Fetching {league_name} {season_str}")
            try:
                mh = sd.MatchHistory(league_name, [season])
                df = mh.read_games()
                out_path = output_dir / f"{league_code}_{season}.parquet"
                df.to_parquet(out_path)
                logger.info(f"  Saved {len(df)} matches -> {out_path}")
            except Exception as e:
                logger.warning(f"  Skipped {league_name} {season}: {e}")


if __name__ == "__main__":
    ingest()
