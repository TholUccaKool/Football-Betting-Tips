"""Fetch upcoming fixtures with odds, run trained models, and display probabilities side-by-side.

Data source (live mode): football-data.co.uk/fixtures.csv (free, updated weekly).
Demo mode (--as-of):     uses a past date from matches.parquet as a stand-in.
Models are fit on all available historical data up to the target date.
"""

import argparse
import html as html_mod
import io
import logging
import sys
from datetime import datetime
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
from src.utils.io import load_config, get_processed_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
TARGET_DIVS = {"E0", "SP1", "I1", "D1", "F1"}
LEAGUE_NAMES = {"EPL", "La Liga", "Serie A", "Bundesliga", "Ligue 1"}
DIV_TO_LEAGUE = {
    "E0": "EPL", "SP1": "La Liga", "I1": "Serie A",
    "D1": "Bundesliga", "F1": "Ligue 1",
}

ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_LABELS = ["Market", "Elo", "Dixon-Coles", "XGBoost"]
OUTCOME_MAP = {0: "H", 1: "D", 2: "A"}
OUTCOME_WORDS = {"H": "Home win", "D": "Draw", "A": "Away win"}


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
    return df[["date", "league", "home_team", "away_team",
               "home_odds", "draw_odds", "away_odds"]].reset_index(drop=True)


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
    return day[fx_cols].reset_index(drop=True), day[act_cols].reset_index(drop=True)


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
    if con["unanimous"]:
        return "all sources agree"
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
    return "sources split: " + ", ".join(parts)


def compute_all_predictions(fixtures, fx_features, elo_model, dc, xgb_model, actuals=None):
    """Return list of dicts, one per match, with all probabilities and consensus."""
    show_actuals = actuals is not None and not actuals.empty
    results = []
    for i in range(len(fixtures)):
        fx = fixtures.iloc[i]
        feat = fx_features.iloc[i:i + 1]

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

        try:
            xgb_p = xgb_model.predict_proba(feat)[0]
        except Exception:
            xgb_p = np.array([np.nan, np.nan, np.nan])

        all_probs = [mkt, elo_p, dc_p, xgb_p]

        # Consensus: average across sources that have valid values
        valid = [p for p in all_probs if not np.any(np.isnan(p))]
        if valid:
            avg = np.mean(valid, axis=0)
            pick_idx = int(np.argmax(avg))
            pick_code = OUTCOME_MAP[pick_idx]
            avg_pct = avg[pick_idx]
            # Check unanimity
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
            "consensus": {"pick": pick_code, "avg_pct": avg_pct, "unanimous": unanimous},
        }
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

            # Consensus headline
            pick_text = _consensus_pick(m)
            pct = m["consensus"]["avg_pct"] * 100
            agree = _agreement_text(m)
            print(f"    >>> Model favors: {pick_text} — {pct:.1f}% avg ({agree})")

            for src in SOURCE_LABELS:
                p = m["probs"][src]
                h, d, a = _fmt_pct(p[0]), _fmt_pct(p[1]), _fmt_pct(p[2])
                print(f"    {src:<14} Home {h:>6}   Draw {d:>6}   Away {a:>6}")

            if "actual" in m:
                act = m["actual"]
                print(f"    {'Result':<14} {act['home_goals']}-{act['away_goals']}"
                      f" ({OUTCOME_WORDS[act['result']]})")


# ── HTML output ──────────────────────────────────────────────────────────

_HTML_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       background: #f5f5f5; color: #1a1a1a; max-width: 820px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 1.5rem; margin-bottom: 4px; }
h2 { font-size: 1.2rem; margin: 24px 0 10px 0; }
.meta { color: #666; font-size: 0.85rem; margin-bottom: 16px; }
.demo-banner { background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
               padding: 10px 14px; margin-bottom: 20px; font-size: 0.9rem; }
.top-picks { background: #fff; border-radius: 8px; border: 1px solid #ddd;
             padding: 14px 18px; margin-bottom: 24px; }
.top-picks ol { padding-left: 24px; }
.top-picks li { padding: 3px 0; font-size: 0.9rem; }
.top-picks .pick-name { font-weight: 600; }
.top-picks .pick-matchup { color: #666; }
.top-picks .pick-tag { font-size: 0.78rem; color: #888; }
.league-hdr { font-size: 1.15rem; font-weight: 600; margin: 28px 0 12px 0;
              padding-bottom: 6px; border-bottom: 2px solid #333; }
.card { background: #fff; border-radius: 8px; border: 1px solid #ddd;
        padding: 16px 18px; margin-bottom: 14px; }
.card-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 2px; }
.card-date { font-size: 0.82rem; color: #888; margin-bottom: 8px; }
.consensus-badge { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 6px;
                   padding: 8px 12px; margin-bottom: 12px; font-size: 0.88rem; }
.consensus-badge strong { color: #4338ca; }
.consensus-badge .agree-tag { font-size: 0.8rem; color: #666; }
.source-row { display: flex; align-items: center; margin-bottom: 6px; }
.source-label { width: 100px; font-size: 0.82rem; font-weight: 500; color: #555; flex-shrink: 0; }
.bar-wrap { flex: 1; display: flex; height: 22px; border-radius: 4px; overflow: hidden; }
.seg { display: flex; align-items: center; justify-content: center;
       font-size: 0.72rem; font-weight: 600; color: #fff; min-width: 28px; }
.seg-home { background: #3b82f6; }
.seg-draw { background: #9ca3af; }
.seg-away { background: #ef4444; }
.result-line { margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;
               font-size: 0.88rem; font-weight: 500; }
.correct { border-left: 4px solid #22c55e; }
.incorrect { border-left: 4px solid #ef4444; }
.footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #ccc;
          font-size: 0.8rem; color: #888; }
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


def render_html(predictions, demo_mode, as_of_date):
    """Write a self-contained HTML report and return the output path."""
    out_dir = ROOT_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    date_label = as_of_date if demo_mode else datetime.now().strftime("%Y-%m-%d")
    out_path = out_dir / f"predictions_{date_label}.html"

    by_league = {}
    for m in predictions:
        by_league.setdefault(m["league"], []).append(m)

    body = []

    # Header
    body.append("<h1>Fixture Probability Comparison</h1>")
    body.append(f'<p class="meta">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>')
    if demo_mode:
        body.append(
            f'<div class="demo-banner">[DEMO MODE] Using historical date '
            f'{html_mod.escape(as_of_date)} as a stand-in for upcoming fixtures '
            f'&mdash; NOT live data</div>')

    # Top picks
    ranked = sorted(predictions, key=lambda m: m["consensus"]["avg_pct"], reverse=True)
    body.append('<h2>Top Picks This Round</h2>')
    body.append('<div class="top-picks"><ol>')
    for m in ranked:
        pick = html_mod.escape(_consensus_pick(m))
        matchup = html_mod.escape(f"{m['home_team']} v {m['away_team']}")
        pct = m["consensus"]["avg_pct"] * 100
        tag = "unanimous" if m["consensus"]["unanimous"] else "split"
        body.append(
            f'<li><span class="pick-name">{pick}</span> '
            f'<span class="pick-matchup">({matchup})</span> '
            f'&mdash; {pct:.1f}% '
            f'<span class="pick-tag">[{tag}]</span></li>')
    body.append('</ol></div>')

    # Per-league cards
    for league in sorted(by_league):
        body.append(f'<div class="league-hdr">{html_mod.escape(league)}</div>')
        for m in by_league[league]:
            has_actual = "actual" in m
            ht = html_mod.escape(m["home_team"])
            at = html_mod.escape(m["away_team"])
            ds = m["date"].strftime("%Y-%m-%d")

            body.append('<div class="card">')
            body.append(f'<div class="card-title">{ht} vs {at}</div>')
            body.append(f'<div class="card-date">{ds}</div>')

            # Consensus badge
            pick = html_mod.escape(_consensus_pick(m))
            pct = m["consensus"]["avg_pct"] * 100
            agree = html_mod.escape(_agreement_text(m))
            body.append(
                f'<div class="consensus-badge">Model favors: <strong>{pick}</strong>'
                f' &mdash; {pct:.1f}% avg '
                f'<span class="agree-tag">({agree})</span></div>')

            for src in SOURCE_LABELS:
                p = m["probs"][src]
                border_cls = ""
                if has_actual and not np.any(np.isnan(p)):
                    pred = OUTCOME_MAP[int(np.argmax(p))]
                    border_cls = " correct" if pred == m["actual"]["result"] else " incorrect"
                body.append(f'<div class="source-row{border_cls}">')
                body.append(f'<div class="source-label">{html_mod.escape(src)}</div>')
                body.append(f'<div class="bar-wrap">{_bar_segments(p)}</div>')
                body.append("</div>")

            if has_actual:
                act = m["actual"]
                rw = OUTCOME_WORDS[act["result"]]
                body.append(
                    f'<div class="result-line">Result: {act["home_goals"]}-{act["away_goals"]}'
                    f' ({rw})</div>')
            body.append("</div>")

    # Footer
    body.append('<div class="footer">')
    body.append("<p>Probabilities shown are model estimates, not recommendations.</p>")
    body.append("<p>Highest probability reflects what the model considers most likely "
                "&mdash; it does not by itself indicate value against the price offered "
                "by a bookmaker.</p>")
    if demo_mode:
        body.append(
            "<p>Actual results shown for gut-check only &mdash; this is not a backtest.</p>")
    body.append("</div>")

    html_content = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>Predictions {html_mod.escape(date_label)}</title>\n"
        f"<style>{_HTML_CSS}</style>\n"
        "</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )

    out_path.write_text(html_content, encoding="utf-8")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare model probabilities against market odds for upcoming fixtures.")
    parser.add_argument("--as-of", metavar="YYYY-MM-DD",
                        help="Demo mode: use matches from this past date as stand-in fixtures.")
    parser.add_argument("--html", action="store_true",
                        help="Also generate an HTML report in output/.")
    args = parser.parse_args()

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
            print(f"\nNo matches found on {args.as_of} for the 5 target leagues.")
            return
        print(f"Found {len(fixtures)} matches on {args.as_of} across: "
              f"{', '.join(sorted(fixtures['league'].unique()))}")
    else:
        print("\nFetching upcoming fixtures from football-data.co.uk...")
        fixtures = fetch_fixtures()
        if fixtures.empty:
            print("\nNo upcoming fixtures found for the 5 target leagues.")
            print("This is expected during the off-season (June-August).")
            print("Tip: use --as-of YYYY-MM-DD to demo against a past matchday.")
            print("\nDivisions currently in the fixtures file:")
            resp = requests.get(FIXTURES_URL, headers={"User-Agent": _BROWSER_UA}, timeout=30)
            if resp.ok:
                all_fx = pd.read_csv(io.BytesIO(resp.content), encoding="utf-8-sig")
                if not all_fx.empty:
                    for div in sorted(all_fx["Div"].unique()):
                        print(f"  {div}: {len(all_fx[all_fx['Div'] == div])} fixtures")
                else:
                    print("  (none — fixtures file is empty)")
            return
        print(f"Found {len(fixtures)} upcoming fixtures across: "
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


if __name__ == "__main__":
    main()
