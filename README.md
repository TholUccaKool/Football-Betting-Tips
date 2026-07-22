# Football Betting Tips

A data-driven football (soccer) match prediction and betting analysis pipeline. Ingests historical match results and xG data, engineers features (Elo ratings, rolling form, rest days), trains multiple models (Elo baseline, Dixon-Coles, XGBoost), and evaluates predictions via walk-forward backtesting with Closing Line Value analysis.

## Leagues Covered

- EPL (E0), La Liga (SP1), Serie A (I1), Bundesliga (D1), Ligue 1 (F1)
- Seasons 2018-2026 (configurable in `config/config.yaml`)

## Setup

```bash
# Clone and enter repo
cd Football-Betting-Tips

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Ingestion

```bash
# 1. Pull historical match results + closing odds
python -m src.data.ingest_match_history

# 2. Pull Understat xG data
python -m src.data.ingest_understat

# 3. Merge into a single clean dataset
python -m src.data.build_dataset
```

This produces `data/processed/matches.parquet` with match results, xG, and closing odds.

## Feature Engineering

```bash
python -m src.features.engineer
```

## Models

- **Elo Baseline** (`src.models.elo_baseline`) — logistic conversion of Elo difference to probabilities
- **Dixon-Coles** (`src.models.dixon_coles`) — Poisson model with low-score correlation adjustment
- **XGBoost** (`src.models.gbm_classifier`) — gradient-boosted multi-class classifier

## Running Tests

```bash
python -m pytest tests/
```

## Project Structure

```
config/config.yaml        — leagues, seasons, Elo params
src/data/                  — data ingestion and dataset building
src/features/              — Elo ratings, feature engineering
src/models/                — prediction models
src/backtest/              — evaluation metrics, CLV analysis
src/utils/                 — config/IO helpers
notebooks/                 — exploratory analysis
tests/                     — unit tests
```

## Notes

- **soccerdata** scrapes football-data.co.uk and Understat. It requires an internet connection and may need `chromium`/`geckodriver` for Understat scraping depending on the version. Check [soccerdata docs](https://github.com/probberechts/soccerdata) if you hit issues.
- Copy `.env.example` to `.env` and add your `ODDS_API_KEY` when ready for live odds integration.
