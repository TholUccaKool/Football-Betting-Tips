"""Parlay backtest: does combining top-confidence picks into parlays help or hurt?

Walk-forward consistent predictions across 8 seasons of real data, using
Pinnacle opening odds as the actual price available before kickoff.
"""

import logging
import random

import numpy as np
import pandas as pd

from src.features.engineer import build_features
from src.models.elo_baseline import fit_elo_calibrator
from src.models.dixon_coles import DixonColesModel
from src.models.gbm_classifier import GBMClassifier, FEATURES_MARKET_BLEND
from src.backtest.evaluate import walk_forward_splits, MIN_TRAIN_MATCHES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTCOME_MAP = {0: "H", 1: "D", 2: "A"}
RANDOM_SEED = 42
N_RANDOM_TRIALS = 50  # average over this many random draws per week for control


def build_walk_forward_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Run walk-forward model fitting and return per-match predictions.

    Returns a DataFrame aligned with df's index containing:
      consensus_prob, consensus_pick (0/1/2), pin_odds_for_pick,
      actual_result (0/1/2), pick_correct (bool).
    """
    splits = walk_forward_splits(df)
    logger.info(f"Walk-forward splits: {len(splits)}")

    records = []

    for split_idx, (train, test) in enumerate(splits):
        if len(train) < MIN_TRAIN_MATCHES:
            continue

        # --- Fit models on training data ---

        # Market implied
        def market_probs(row):
            h, d, a = row["home_odds"], row["draw_odds"], row["away_odds"]
            if pd.notna(h) and pd.notna(d) and pd.notna(a):
                raw = np.array([1/h, 1/d, 1/a])
                return raw / raw.sum()
            return np.array([np.nan, np.nan, np.nan])

        # Elo calibrator
        elo_model = fit_elo_calibrator(train)

        # Dixon-Coles (last 24 months of training data)
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

        # XGBoost market-blend
        xgb_train = train.dropna(subset=FEATURES_MARKET_BLEND + ["target"])
        xgb_model = None
        if len(xgb_train) >= 50:
            xgb_model = GBMClassifier(feature_cols=FEATURES_MARKET_BLEND)
            xgb_model.train(xgb_train)

        # --- Predict on test data ---
        for idx, row in test.iterrows():
            sources = []

            # Market
            mkt = market_probs(row)
            if not np.any(np.isnan(mkt)):
                sources.append(mkt)

            # Elo
            try:
                elo_p = elo_model.predict_proba(
                    np.array([[row["elo_diff"]]])
                )[0]
                sources.append(elo_p)
            except Exception:
                pass

            # Dixon-Coles
            try:
                dc_p = np.array(dc.predict_proba(row["home_team"], row["away_team"]))
                sources.append(dc_p)
            except KeyError:
                pass

            # XGBoost
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

            if not sources:
                continue

            # Consensus: average across valid sources, pick highest
            avg = np.mean(sources, axis=0)
            pick_idx = int(np.argmax(avg))
            consensus_prob = avg[pick_idx]

            # Pinnacle opening odds for the picked side
            pin_cols = ["pin_open_home", "pin_open_draw", "pin_open_away"]
            pin_odds = [row.get(c) for c in pin_cols]
            if any(pd.isna(o) for o in pin_odds):
                pin_odds_for_pick = np.nan
            else:
                pin_odds_for_pick = pin_odds[pick_idx]

            actual = int(row["target"])

            records.append({
                "idx": idx,
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "consensus_prob": consensus_prob,
                "consensus_pick": pick_idx,
                "pin_odds_for_pick": pin_odds_for_pick,
                "actual_result": actual,
                "pick_correct": pick_idx == actual,
            })

        if (split_idx + 1) % 5 == 0:
            logger.info(f"  Completed split {split_idx + 1}/{len(splits)} "
                        f"({len(records)} predictions so far)")

    logger.info(f"Total walk-forward predictions: {len(records)}")
    return pd.DataFrame(records)


def run_parlay_backtest(preds: pd.DataFrame, max_n: int = 5):
    """Run the parlay backtest for N=1..max_n, both top-confidence and random.

    Returns a list of result dicts for the summary table.
    """
    # Filter to matches with valid Pinnacle odds
    valid = preds[preds["pin_odds_for_pick"].notna()].copy()
    logger.info(f"Matches with Pinnacle opening odds for picked side: {len(valid)}/{len(preds)}")

    # Group by ISO year-week
    valid["iso_year"] = valid["date"].dt.isocalendar().year.astype(int)
    valid["iso_week"] = valid["date"].dt.isocalendar().week.astype(int)
    valid["year_week"] = valid["iso_year"] * 100 + valid["iso_week"]

    weeks = valid.groupby("year_week")
    week_data = []
    for yw, grp in weeks:
        # Sort by consensus confidence descending
        grp_sorted = grp.sort_values("consensus_prob", ascending=False).reset_index(drop=True)
        week_data.append(grp_sorted)

    logger.info(f"Distinct ISO weeks: {len(week_data)}")

    results = []
    rng = random.Random(RANDOM_SEED)

    for n in range(1, max_n + 1):
        # --- Top-confidence strategy ---
        top_parlays = 0
        top_wins = 0
        top_pnl = 0.0
        top_bankroll = 100.0
        top_peak = 100.0
        top_max_dd = 0.0

        for grp in week_data:
            if len(grp) < n:
                continue
            picks = grp.iloc[:n]
            combined_odds = picks["pin_odds_for_pick"].prod()
            all_correct = picks["pick_correct"].all()

            top_parlays += 1
            if all_correct:
                top_wins += 1
                top_pnl += combined_odds - 1.0
                top_bankroll += combined_odds - 1.0
            else:
                top_pnl -= 1.0
                top_bankroll -= 1.0

            top_peak = max(top_peak, top_bankroll)
            dd = top_peak - top_bankroll
            top_max_dd = max(top_max_dd, dd)

        top_roi = (top_pnl / top_parlays * 100) if top_parlays > 0 else 0.0

        # Track average combined odds for winning parlays
        win_payouts = []
        for grp in week_data:
            if len(grp) < n:
                continue
            picks = grp.iloc[:n]
            combined_odds = picks["pin_odds_for_pick"].prod()
            if picks["pick_correct"].all():
                win_payouts.append(combined_odds)

        avg_payout = np.mean(win_payouts) if win_payouts else 0.0

        results.append({
            "N": n,
            "strategy": "Top confidence",
            "parlays": top_parlays,
            "hit_rate": f"{top_wins/top_parlays*100:.1f}%" if top_parlays > 0 else "-",
            "roi_pct": top_roi,
            "final_bankroll": top_bankroll,
            "max_drawdown": top_max_dd,
            "wins": top_wins,
            "avg_win_payout": avg_payout,
        })

        # --- Random selection control (averaged over N_RANDOM_TRIALS) ---
        rand_parlays_total = 0
        rand_wins_total = 0
        rand_pnl_total = 0.0
        rand_final_br_total = 0.0
        rand_max_dd_total = 0.0

        for trial in range(N_RANDOM_TRIALS):
            r_parlays = 0
            r_wins = 0
            r_pnl = 0.0
            r_bankroll = 100.0
            r_peak = 100.0
            r_max_dd = 0.0

            for grp in week_data:
                if len(grp) < n:
                    continue
                # Pick N random matches from this week
                indices = list(range(len(grp)))
                rng.shuffle(indices)
                picks = grp.iloc[indices[:n]]
                combined_odds = picks["pin_odds_for_pick"].prod()
                all_correct = picks["pick_correct"].all()

                r_parlays += 1
                if all_correct:
                    r_wins += 1
                    r_pnl += combined_odds - 1.0
                    r_bankroll += combined_odds - 1.0
                else:
                    r_pnl -= 1.0
                    r_bankroll -= 1.0

                r_peak = max(r_peak, r_bankroll)
                dd = r_peak - r_bankroll
                r_max_dd = max(r_max_dd, dd)

            rand_parlays_total += r_parlays
            rand_wins_total += r_wins
            rand_pnl_total += r_pnl
            rand_final_br_total += r_bankroll
            rand_max_dd_total += r_max_dd

        avg_parlays = rand_parlays_total / N_RANDOM_TRIALS
        avg_wins = rand_wins_total / N_RANDOM_TRIALS
        avg_pnl = rand_pnl_total / N_RANDOM_TRIALS
        avg_roi = (avg_pnl / avg_parlays * 100) if avg_parlays > 0 else 0.0
        avg_final_br = rand_final_br_total / N_RANDOM_TRIALS
        avg_max_dd = rand_max_dd_total / N_RANDOM_TRIALS

        results.append({
            "N": n,
            "strategy": "Random (avg 50 trials)",
            "parlays": int(round(avg_parlays)),
            "hit_rate": f"{avg_wins/avg_parlays*100:.1f}%" if avg_parlays > 0 else "-",
            "roi_pct": avg_roi,
            "final_bankroll": avg_final_br,
            "max_drawdown": avg_max_dd,
            "wins": avg_wins,
        })

    return results


def print_results(results):
    """Print the summary table."""
    print(f"\n{'='*90}")
    print("PARLAY BACKTEST RESULTS — Pinnacle opening odds, flat 1-unit stake, 100-unit start")
    print(f"{'='*90}")
    print(f"{'N':<4} {'Strategy':<24} {'Parlays':>8} {'Hit Rate':>10} "
          f"{'ROI%':>8} {'Final BR':>10} {'Max DD':>8}")
    print("─" * 90)

    prev_n = None
    for r in results:
        if prev_n is not None and r["N"] != prev_n:
            print("─" * 90)
        prev_n = r["N"]
        roi_str = f"{r['roi_pct']:+.1f}%"
        br_str = f"{r['final_bankroll']:.1f}"
        dd_str = f"{r['max_drawdown']:.1f}"
        print(f"{r['N']:<4} {r['strategy']:<24} {r['parlays']:>8} {r['hit_rate']:>10} "
              f"{roi_str:>8} {br_str:>10} {dd_str:>8}")

    print("─" * 90)

    # Variance context
    print("\nSAMPLE SIZE CONTEXT:")
    for n in range(1, 6):
        top = next(r for r in results if r["N"] == n and "Top" in r["strategy"])
        wins = top["wins"]
        total = top["parlays"]
        avg_pay = top.get("avg_win_payout", 0)
        hit = wins / total if total > 0 else 0
        se = (hit * (1 - hit) / total) ** 0.5 * 100 if total > 0 else 0
        # Each winning parlay pays avg_pay units, so +/-1 win swings P&L by avg_pay units
        swing_pp = avg_pay / total * 100 if total > 0 and avg_pay > 0 else 0
        print(f"  N={n}: {wins} wins / {total} parlays. "
              f"Avg win pays {avg_pay:.1f}x. "
              f"Hit rate 95% CI: {hit*100 - 1.96*se:.1f}%–{hit*100 + 1.96*se:.1f}%. "
              f"+/- 1 win swings ROI by ~{swing_pp:.1f}pp.")

    # Interpretation
    print("\nINTERPRETATION:")
    n1 = next(r for r in results if r["N"] == 1 and r["strategy"] == "Top confidence")
    n5 = next(r for r in results if r["N"] == 5 and r["strategy"] == "Top confidence")

    if n1["roi_pct"] < 0 and n5["roi_pct"] < n1["roi_pct"]:
        print("  ROI is negative and worsens as parlay size increases — the compounding-vig")
        print("  effect is clearly visible. Each additional leg multiplies the bookmaker's")
        print("  built-in margin, making the parlay progressively harder to beat even when")
        print("  picking the most confident selections.")
    elif n5["roi_pct"] > 0 and n5["wins"] < 80:
        print("  Positive ROI at larger N is driven by a small number of winning parlays.")
        print(f"  At N=5, just {n5['wins']} wins out of {n5['parlays']} parlays determine the result.")
        print("  A handful of extra wins or misses would flip the sign entirely — this is")
        print("  high-variance territory where ~195 weekly parlays across 8 seasons is not")
        print("  enough to draw reliable conclusions. Do not treat this as a discovered strategy.")
    else:
        print("  Results are mixed. Treat with caution given the sample sizes involved.")

    if n1["roi_pct"] < 0:
        print(f"\n  N=1 (singles baseline) shows {n1['roi_pct']:+.1f}% ROI — the models cannot")
        print("  beat Pinnacle's opening odds on a flat-stake basis even when selecting the")
        print("  single most confident pick each week. This is consistent with prior CLV")
        print("  analysis showing no model achieves positive average CLV against Pinnacle.")

    # Compare top-confidence vs random
    print("\nTOP-CONFIDENCE vs RANDOM CONTROL:")
    for n in range(1, 6):
        top = next(r for r in results if r["N"] == n and "Top" in r["strategy"])
        rand = next(r for r in results if r["N"] == n and "Random" in r["strategy"])
        diff = top["roi_pct"] - rand["roi_pct"]
        direction = "better" if diff > 0 else "worse"
        print(f"  N={n}: Top-confidence ROI {top['roi_pct']:+.1f}% vs Random {rand['roi_pct']:+.1f}% "
              f"(confidence selection is {abs(diff):.1f}pp {direction})")

    # Note on random showing positive ROI
    rand_pos = [r for r in results if "Random" in r["strategy"] and r["roi_pct"] > 0]
    if rand_pos:
        print("\n  Note: The random control also shows positive ROI at some N values.")
        print("  With ~195 parlays and low hit rates at N>=3 (single-digit wins),")
        print("  variance overwhelms signal. A few lucky payouts dominate the total.")


def main():
    logger.info("Building feature matrix...")
    df = build_features()
    logger.info(f"Feature matrix: {df.shape}")

    # Verify Pinnacle odds coverage
    pin_cols = ["pin_open_home", "pin_open_draw", "pin_open_away"]
    pin_valid = df[pin_cols].notna().all(axis=1)
    logger.info(f"Pinnacle opening odds coverage: {pin_valid.sum()}/{len(df)}")

    logger.info("\nRunning walk-forward predictions across all splits...")
    preds = build_walk_forward_predictions(df)

    logger.info("\nRunning parlay backtests (N=1..5, top-confidence + random control)...")
    results = run_parlay_backtest(preds, max_n=5)

    print_results(results)


if __name__ == "__main__":
    main()
