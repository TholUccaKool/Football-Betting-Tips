"""Best-of-all-markets backtest: does picking the highest-confidence option
across H/D/A + O/U 2.5 + BTTS + Asian Handicap genuinely improve hit rate,
or is it a selection-effect illusion?

Reuses the existing walk-forward model fitting from parlay_backtest.py and
adds secondary market picks from Dixon-Coles scoreline grids.
"""

import logging
import random

import numpy as np
import pandas as pd

from src.features.engineer import build_features
from src.models.elo_baseline import fit_elo_calibrator
from src.models.dixon_coles import DixonColesModel
from src.models.gbm_classifier import GBMClassifier, FEATURES_MARKET_BLEND
from src.models.market_derivations import over_under, btts, asian_handicap
from src.backtest.evaluate import walk_forward_splits, MIN_TRAIN_MATCHES
from src.utils.io import get_raw_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTCOME_MAP = {0: "H", 1: "D", 2: "A"}
RANDOM_SEED = 42
N_RANDOM_TRIALS = 50


def _load_ah_lines() -> pd.DataFrame:
    """Load Asian handicap lines from raw match_history parquets."""
    raw_dir = get_raw_dir() / "match_history"
    if not raw_dir.exists():
        return pd.DataFrame()
    frames = []
    for f in sorted(raw_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if "AHh" not in df.columns:
            continue
        keep = {}
        if "date" in df.columns:
            keep["date"] = "date"
        for hcol in ["HomeTeam", "home_team"]:
            if hcol in df.columns:
                keep[hcol] = "home_team"
                break
        for acol in ["AwayTeam", "away_team"]:
            if acol in df.columns:
                keep[acol] = "away_team"
                break
        keep["AHh"] = "ah_line"
        sub = df[[c for c in keep if c in df.columns]].copy()
        sub = sub.rename(columns=keep)
        sub["ah_line"] = pd.to_numeric(sub["ah_line"], errors="coerce")
        sub["date"] = pd.to_datetime(sub["date"])
        frames.append(sub.dropna(subset=["ah_line"]))
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")


def build_all_market_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward predictions for all 4 market types per match.

    Returns one row per match with picks and actuals for each market.
    """
    ah_lines = _load_ah_lines()
    if not ah_lines.empty:
        ah_lines["_date_key"] = ah_lines["date"].dt.normalize()
        logger.info(f"Loaded {len(ah_lines)} AH lines from raw data")
    else:
        logger.info("No AH line data found")

    splits = walk_forward_splits(df)
    logger.info(f"Walk-forward splits: {len(splits)}")

    records = []

    for split_idx, (train, test) in enumerate(splits):
        if len(train) < MIN_TRAIN_MATCHES:
            continue

        # --- Fit models on training data ---
        def market_probs(row):
            h, d, a = row["home_odds"], row["draw_odds"], row["away_odds"]
            if pd.notna(h) and pd.notna(d) and pd.notna(a):
                raw = np.array([1/h, 1/d, 1/a])
                return raw / raw.sum()
            return np.array([np.nan, np.nan, np.nan])

        elo_model = fit_elo_calibrator(train)

        dc_cutoff = train["date"].max() - pd.DateOffset(months=24)
        dc_train = train[train["date"] >= dc_cutoff]
        dc = DixonColesModel()
        dc.fit(
            dc_train["home_team"].values,
            dc_train["away_team"].values,
            dc_train["home_goals"].values.astype(int),
            dc_train["away_goals"].values.astype(int),
            match_dates=dc_train["date"].values,
            reference_date=test["date"].min(),
        )

        xgb_train = train.dropna(subset=FEATURES_MARKET_BLEND + ["target"])
        xgb_model = None
        if len(xgb_train) >= 50:
            xgb_model = GBMClassifier(feature_cols=FEATURES_MARKET_BLEND)
            xgb_model.train(xgb_train)

        # --- Predict on test data ---
        for idx, row in test.iterrows():
            sources = []

            mkt = market_probs(row)
            if not np.any(np.isnan(mkt)):
                sources.append(mkt)

            try:
                elo_p = elo_model.predict_proba(
                    np.array([[row["elo_diff"]]])
                )[0]
                sources.append(elo_p)
            except Exception:
                pass

            dc_grid = None
            try:
                dc_p = np.array(dc.predict_proba(row["home_team"], row["away_team"]))
                sources.append(dc_p)
                dc_grid = dc.predict_scoreline_matrix(row["home_team"], row["away_team"])
            except KeyError:
                pass

            if xgb_model is not None:
                try:
                    test_row = pd.DataFrame([row])
                    for c in FEATURES_MARKET_BLEND:
                        if test_row[c].isna().any():
                            test_row[c] = test_row[c].fillna(train[c].median())
                    xgb_p = xgb_model.predict_proba(test_row)[0]
                    sources.append(xgb_p)
                except Exception:
                    pass

            if not sources or dc_grid is None:
                continue

            # --- H/D/A consensus ---
            avg = np.mean(sources, axis=0)
            hda_pick_idx = int(np.argmax(avg))
            hda_prob = avg[hda_pick_idx]
            actual_result = int(row["target"])
            hda_correct = hda_pick_idx == actual_result

            # --- Actuals for secondary markets ---
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
            total_goals = hg + ag
            actual_over25 = total_goals > 2.5
            actual_btts = hg >= 1 and ag >= 1

            # --- O/U 2.5 from Dixon-Coles ---
            p_over, p_under = over_under(dc_grid, 2.5)
            if p_over >= p_under:
                ou_pick = "over"
                ou_prob = p_over
                ou_correct = actual_over25
            else:
                ou_pick = "under"
                ou_prob = p_under
                ou_correct = not actual_over25

            # --- BTTS from Dixon-Coles ---
            p_btts_yes, p_btts_no = btts(dc_grid)
            if p_btts_yes >= p_btts_no:
                btts_pick = "yes"
                btts_prob = p_btts_yes
                btts_correct = actual_btts
            else:
                btts_pick = "no"
                btts_prob = p_btts_no
                btts_correct = not actual_btts

            # --- Asian Handicap from Dixon-Coles (where line exists) ---
            ah_prob = np.nan
            ah_correct = np.nan
            ah_pick = None
            if not ah_lines.empty:
                match_date = pd.Timestamp(row["date"]).normalize()
                ah_match = ah_lines[
                    (ah_lines["_date_key"] == match_date) &
                    (ah_lines["home_team"] == row["home_team"]) &
                    (ah_lines["away_team"] == row["away_team"])
                ]
                if not ah_match.empty:
                    line_val = float(ah_match.iloc[0]["ah_line"])
                    p_cover, p_push, p_lose = asian_handicap(dc_grid, line_val, side="home")
                    # AH pick: home covers vs away covers (push = refund, ignore)
                    # Effective probabilities excluding push
                    if p_cover >= p_lose:
                        ah_pick = "home_cover"
                        ah_prob = p_cover
                        # Did home actually cover?
                        adjusted = (hg - ag) + line_val
                        ah_correct = adjusted > 1e-9  # covers
                    else:
                        ah_pick = "away_cover"
                        ah_prob = p_lose
                        adjusted = (hg - ag) + line_val
                        ah_correct = adjusted < -1e-9  # home loses = away covers

            # --- Build candidate picks ---
            candidates = [
                ("H/D/A", hda_prob, hda_correct),
                ("O/U 2.5", ou_prob, ou_correct),
                ("BTTS", btts_prob, btts_correct),
            ]
            if ah_pick is not None and not np.isnan(ah_prob):
                candidates.append(("AH", ah_prob, ah_correct))

            # Best pick = highest stated probability
            best = max(candidates, key=lambda x: x[1])

            records.append({
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                # H/D/A baseline
                "hda_prob": hda_prob,
                "hda_correct": hda_correct,
                # Best-of-all
                "best_market": best[0],
                "best_prob": best[1],
                "best_correct": best[2],
                # Individual market probs (for random control)
                "ou_prob": ou_prob,
                "ou_correct": ou_correct,
                "btts_prob": btts_prob,
                "btts_correct": btts_correct,
                "ah_prob": ah_prob if ah_pick else np.nan,
                "ah_correct": ah_correct if ah_pick else np.nan,
            })

        if (split_idx + 1) % 5 == 0:
            logger.info(f"  Completed split {split_idx + 1}/{len(splits)} "
                        f"({len(records)} predictions so far)")

    logger.info(f"Total walk-forward predictions: {len(records)}")
    return pd.DataFrame(records)


def run_backtest(preds: pd.DataFrame):
    """Compute hit rates for Strategy A (best-of-all), B (H/D/A), C (random)."""
    n = len(preds)

    # Strategy B: H/D/A baseline
    hda_hits = preds["hda_correct"].sum()
    hda_avg_prob = preds["hda_prob"].mean()
    hda_hit_rate = hda_hits / n

    # Strategy A: Best of all markets
    best_hits = preds["best_correct"].sum()
    best_avg_prob = preds["best_prob"].mean()
    best_hit_rate = best_hits / n

    # Strategy C: Random market control (50 trials)
    rng = random.Random(RANDOM_SEED)
    rand_hits_total = 0
    rand_prob_total = 0.0
    rand_n_total = 0

    for trial in range(N_RANDOM_TRIALS):
        for _, row in preds.iterrows():
            candidates = [
                (row["hda_prob"], row["hda_correct"]),
                (row["ou_prob"], row["ou_correct"]),
                (row["btts_prob"], row["btts_correct"]),
            ]
            if pd.notna(row["ah_prob"]):
                candidates.append((row["ah_prob"], row["ah_correct"]))
            pick = rng.choice(candidates)
            rand_prob_total += pick[0]
            rand_hits_total += int(pick[1])
            rand_n_total += 1

    rand_avg_prob = rand_prob_total / rand_n_total
    rand_hit_rate = rand_hits_total / rand_n_total

    # Market type breakdown for Strategy A
    market_counts = preds["best_market"].value_counts()
    total_picks = len(preds)

    return {
        "n": n,
        "hda_avg_prob": hda_avg_prob,
        "hda_hit_rate": hda_hit_rate,
        "best_avg_prob": best_avg_prob,
        "best_hit_rate": best_hit_rate,
        "rand_avg_prob": rand_avg_prob,
        "rand_hit_rate": rand_hit_rate,
        "market_breakdown": {m: c / total_picks for m, c in market_counts.items()},
    }


def print_results(r):
    """Print summary table and analysis."""
    n = r["n"]

    print(f"\n{'='*78}")
    print("BEST-OF-ALL-MARKETS BACKTEST — Walk-forward, 8 seasons, raw hit rate")
    print(f"{'='*78}")
    print(f"\n{'Strategy':<30} {'N matches':>10} {'Avg stated prob':>16} "
          f"{'Actual hit rate':>16} {'Gap':>8}")
    print("─" * 78)

    rows = [
        ("A: Best of all markets", n,
         r["best_avg_prob"], r["best_hit_rate"]),
        ("B: H/D/A only (baseline)", n,
         r["hda_avg_prob"], r["hda_hit_rate"]),
        ("C: Random market (50 trials)", n,
         r["rand_avg_prob"], r["rand_hit_rate"]),
    ]

    for label, nm, avg_p, hit in rows:
        gap = avg_p - hit
        print(f"{label:<30} {nm:>10} {avg_p*100:>15.1f}% {hit*100:>15.1f}% "
              f"{gap*100:>+7.1f}pp")

    print("─" * 78)

    # Market type breakdown
    print("\nSTRATEGY A — PICK SOURCE BREAKDOWN:")
    for market, frac in sorted(r["market_breakdown"].items(),
                                key=lambda x: x[1], reverse=True):
        print(f"  {market:<10} {frac*100:5.1f}% of picks")

    # Calibration analysis
    print("\nCALIBRATION ANALYSIS:")
    best_gap = r["best_avg_prob"] - r["best_hit_rate"]
    hda_gap = r["hda_avg_prob"] - r["hda_hit_rate"]

    print(f"  Strategy A (best-of-all): avg stated {r['best_avg_prob']*100:.1f}%, "
          f"actual {r['best_hit_rate']*100:.1f}% — gap {best_gap*100:+.1f}pp")
    print(f"  Strategy B (H/D/A only):  avg stated {r['hda_avg_prob']*100:.1f}%, "
          f"actual {r['hda_hit_rate']*100:.1f}% — gap {hda_gap*100:+.1f}pp")

    # Conclusion
    print(f"\n{'='*78}")
    print("CONCLUSION:")
    print(f"{'='*78}")

    hit_diff = r["best_hit_rate"] - r["hda_hit_rate"]
    rand_diff = r["best_hit_rate"] - r["rand_hit_rate"]

    if best_gap > hda_gap + 0.02:
        print(f"\n  Strategy A's actual hit rate ({r['best_hit_rate']*100:.1f}%) is "
              f"{hit_diff*100:+.1f}pp vs the H/D/A baseline ({r['hda_hit_rate']*100:.1f}%).")
        print(f"  However, the calibration gap is LARGER for Strategy A ({best_gap*100:+.1f}pp) "
              f"than for")
        print(f"  H/D/A alone ({hda_gap*100:+.1f}pp). "
              f"This is the selection-effect signature: picking the")
        print(f"  highest number from a bigger pool inflates the stated confidence more")
        print(f"  than it inflates the actual hit rate.")
        print(f"\n  The random-market control hits at {r['rand_hit_rate']*100:.1f}% "
              f"(vs Strategy A's {r['best_hit_rate']*100:.1f}%),")
        if abs(rand_diff) < 0.01:
            print(f"  confirming that most of the apparent improvement comes from selection")
            print(f"  bias rather than genuinely better predictions.")
        elif rand_diff > 0.01:
            print(f"  suggesting some marginal improvement from picking higher-confidence")
            print(f"  markets, but the widened calibration gap means the stated probabilities")
            print(f"  overstate the true edge.")
        else:
            print(f"  which is actually HIGHER than Strategy A — the selection effect is")
            print(f"  pure illusion in this case.")
    elif abs(hit_diff) < 0.005:
        print(f"\n  Strategy A ({r['best_hit_rate']*100:.1f}%) and the H/D/A baseline "
              f"({r['hda_hit_rate']*100:.1f}%) have")
        print(f"  essentially identical hit rates. The higher stated confidence "
              f"({r['best_avg_prob']*100:.1f}% vs")
        print(f"  {r['hda_avg_prob']*100:.1f}%) is entirely selection bias — picking the "
              f"biggest number from")
        print(f"  more options produces bigger numbers without improving accuracy.")
    else:
        print(f"\n  Strategy A hits at {r['best_hit_rate']*100:.1f}% vs H/D/A baseline "
              f"{r['hda_hit_rate']*100:.1f}% ({hit_diff*100:+.1f}pp).")
        print(f"  Calibration gaps: A={best_gap*100:+.1f}pp, B={hda_gap*100:+.1f}pp.")
        if best_gap > hda_gap:
            print(f"  The wider calibration gap for Strategy A suggests some selection-effect")
            print(f"  inflation, though the hit rate improvement is real.")
        else:
            print(f"  The calibration gap is comparable, suggesting the improvement is genuine.")

    # BTTS note
    btts_frac = r["market_breakdown"].get("BTTS", 0)
    ou_frac = r["market_breakdown"].get("O/U 2.5", 0)
    if btts_frac > 0.1 or ou_frac > 0.1:
        secondary = btts_frac + ou_frac + r["market_breakdown"].get("AH", 0)
        print(f"\n  Note: {secondary*100:.0f}% of Strategy A's picks come from secondary markets")
        print(f"  (BTTS: {btts_frac*100:.0f}%, O/U: {ou_frac*100:.0f}%, "
              f"AH: {r['market_breakdown'].get('AH', 0)*100:.0f}%).")
        if btts_frac > 0.1:
            print(f"  BTTS probabilities are from Dixon-Coles only (single model, no market")
            print(f"  cross-check) — treat with appropriate skepticism.")


def main():
    logger.info("Building feature matrix...")
    df = build_features()
    logger.info(f"Feature matrix: {df.shape}")

    logger.info("\nRunning walk-forward predictions across all splits...")
    preds = build_all_market_predictions(df)

    logger.info("\nComputing hit rates...")
    results = run_backtest(preds)

    print_results(results)


if __name__ == "__main__":
    main()
