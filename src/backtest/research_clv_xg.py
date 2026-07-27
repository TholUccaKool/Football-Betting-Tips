"""Research script: softer-bookmaker CLV + richer xG features evaluation.

Part A: CLV analysis using William Hill (softest bookmaker) vs Pinnacle
Part B: Walk-forward backtest comparing V1 vs V2 xG features
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features.engineer import build_features
from src.models.elo_baseline import fit_elo_calibrator
from src.models.dixon_coles import DixonColesModel
from src.models.gbm_classifier import (
    GBMClassifier,
    FEATURES_OWN_SIGNAL, FEATURES_MARKET_BLEND,
    FEATURES_OWN_SIGNAL_V2, FEATURES_MARKET_BLEND_V2,
)
from src.backtest.evaluate import walk_forward_splits, brier_score, log_loss, MIN_TRAIN_MATCHES
from src.backtest.clv import compute_clv, EDGE_THRESHOLD
from src.utils.io import get_raw_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_wh_odds() -> pd.DataFrame:
    """Load William Hill opening/closing odds from raw match_history parquets.

    Returns DataFrame with date, home_team, away_team, and WH odds columns.
    """
    raw_dir = get_raw_dir() / "match_history"
    frames = []
    for f in sorted(raw_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        # Check if WH columns exist (both open and close)
        open_cols = ["WHH", "WHD", "WHA"]
        close_cols = ["WHCH", "WHCD", "WHCA"]
        if not all(c in df.columns for c in open_cols + close_cols):
            continue
        sub = df[["date", "home_team", "away_team"] + open_cols + close_cols].copy()
        sub.columns = [
            "date", "home_team", "away_team",
            "wh_open_home", "wh_open_draw", "wh_open_away",
            "wh_close_home", "wh_close_draw", "wh_close_away",
        ]
        sub["date"] = pd.to_datetime(sub["date"])
        for c in ["wh_open_home", "wh_open_draw", "wh_open_away",
                   "wh_close_home", "wh_close_draw", "wh_close_away"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    wh = pd.concat(frames, ignore_index=True)
    # Deduplicate: same match can appear in overlapping season files
    wh = wh.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return wh


def market_implied_probs(home_odds, draw_odds, away_odds):
    """Normalised implied probabilities from decimal odds."""
    raw = np.column_stack([1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds])
    return raw / raw.sum(axis=1, keepdims=True)


def run_clv_analysis(model_probs, opening_odds, closing_odds, label=""):
    """Run CLV analysis and return summary dict."""
    clv_df = compute_clv(model_probs, opening_odds, closing_odds)
    bets = clv_df[clv_df["is_bet"]]
    n_bets = len(bets)
    n_matches = len(model_probs)
    if n_bets == 0:
        return {
            "label": label,
            "n_matches": n_matches,
            "n_bets": 0,
            "pct_positive_clv": np.nan,
            "avg_clv_pct": np.nan,
            "edge_clv_corr": np.nan,
            "edge_clv_pval": np.nan,
        }
    pct_pos = (bets["clv_pct"] > 0).mean() * 100
    avg_clv = bets["clv_pct"].mean()
    if len(bets) >= 5:
        corr, pval = spearmanr(bets["edge"], bets["clv_pct"])
    else:
        corr, pval = np.nan, np.nan
    return {
        "label": label,
        "n_matches": n_matches,
        "n_bets": n_bets,
        "pct_positive_clv": pct_pos,
        "avg_clv_pct": avg_clv,
        "edge_clv_corr": corr,
        "edge_clv_pval": pval,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ══════════════════════════════════════════════════════════════════════════
    # Load and prepare data
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Building feature matrix (includes new V2 xG features)...")
    df = build_features()
    logger.info(f"Feature matrix: {df.shape}")

    # Check V2 feature coverage
    for col in ["home_xg_overperformance", "away_xg_overperformance",
                 "home_opp_adjusted_xg_for", "away_opp_adjusted_xg_for"]:
        n_valid = df[col].notna().sum()
        logger.info(f"  {col}: {n_valid}/{len(df)} non-NaN")

    # Load WH odds
    logger.info("\nLoading William Hill odds from raw data...")
    wh = load_wh_odds()
    logger.info(f"  WH rows loaded: {len(wh)}")

    # Merge WH odds into feature matrix using date-only key
    # (raw parquet dates may differ in time component from processed data)
    df["_date_key"] = df["date"].dt.normalize()
    wh["_date_key"] = wh["date"].dt.normalize()
    n_before = len(df)
    df = df.merge(
        wh.drop(columns=["date"]),
        on=["_date_key", "home_team", "away_team"],
        how="left",
    )
    df.drop(columns=["_date_key"], inplace=True)
    # Guard against merge-induced duplication
    if len(df) > n_before:
        df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
        logger.info(f"  (deduped merge result from {n_before} to {len(df)})")

    wh_mask = df[["wh_open_home", "wh_open_draw", "wh_open_away",
                   "wh_close_home", "wh_close_draw", "wh_close_away"]].notna().all(axis=1)
    logger.info(f"  WH open+close matched to features: {wh_mask.sum()}/{len(df)}")
    wh_dates = df.loc[wh_mask, "date"]
    logger.info(f"  WH date range: {wh_dates.min().date()} to {wh_dates.max().date()}")

    pin_mask = df[["pin_open_home", "pin_open_draw", "pin_open_away",
                    "pin_close_home", "pin_close_draw", "pin_close_away"]].notna().all(axis=1)
    logger.info(f"  Pinnacle open+close: {pin_mask.sum()}/{len(df)}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART B: Walk-forward backtest — V1 vs V2 features
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PART B: Walk-forward backtest — V1 vs V2 xG features")
    logger.info("=" * 70)

    # Restrict to rows with xG (nearly all) and valid target
    df_xg = df.dropna(subset=["target", "elo_diff", "home_xg_overperformance"]).copy()
    logger.info(f"\nCommon xG-covered subset: {len(df_xg)} matches")

    splits = walk_forward_splits(df_xg)
    logger.info(f"Walk-forward splits: {len(splits)}")

    # Collect per-split metrics
    model_names = [
        "Naive", "Market", "Elo", "Dixon-Coles",
        "XGB-own-v1", "XGB-mkt-v1",
        "XGB-own-v2", "XGB-mkt-v2",
    ]
    all_metrics = {m: {"brier": [], "logloss": []} for m in model_names}

    # Also collect model predictions for CLV analysis (on common Pinnacle+WH subset)
    clv_collectors = {
        "Elo": [], "Dixon-Coles": [],
        "XGB-mkt-v1": [], "XGB-mkt-v2": [],
    }
    clv_meta = []  # track indices for CLV matching

    for split_idx, (train, test) in enumerate(splits):
        if len(train) < MIN_TRAIN_MATCHES:
            continue

        y_test = test["target"].values.astype(int)
        n_test = len(test)

        # Naive: 1/3 each
        naive_probs = np.full((n_test, 3), 1.0 / 3)
        all_metrics["Naive"]["brier"].append(brier_score(y_test, naive_probs))
        all_metrics["Naive"]["logloss"].append(log_loss(y_test, naive_probs))

        # Market implied
        mkt_valid = test[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)
        if mkt_valid.sum() > 0:
            mkt_probs_full = np.full((n_test, 3), 1.0 / 3)
            mkt_sub = test[mkt_valid]
            mkt_p = market_implied_probs(
                mkt_sub["home_odds"].values,
                mkt_sub["draw_odds"].values,
                mkt_sub["away_odds"].values,
            )
            mkt_probs_full[mkt_valid.values] = mkt_p
            all_metrics["Market"]["brier"].append(brier_score(y_test, mkt_probs_full))
            all_metrics["Market"]["logloss"].append(log_loss(y_test, mkt_probs_full))

        # Elo calibrator
        from sklearn.linear_model import LogisticRegression
        elo_model = fit_elo_calibrator(train)
        elo_probs = elo_model.predict_proba(test[["elo_diff"]].values)
        all_metrics["Elo"]["brier"].append(brier_score(y_test, elo_probs))
        all_metrics["Elo"]["logloss"].append(log_loss(y_test, elo_probs))

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
        dc_probs = np.array([
            dc.predict_proba(row["home_team"], row["away_team"])
            if row["home_team"] in dc.teams and row["away_team"] in dc.teams
            else [1/3, 1/3, 1/3]
            for _, row in test.iterrows()
        ])
        all_metrics["Dixon-Coles"]["brier"].append(brier_score(y_test, dc_probs))
        all_metrics["Dixon-Coles"]["logloss"].append(log_loss(y_test, dc_probs))

        # XGBoost V1
        for feat_name, feat_cols in [("XGB-own-v1", FEATURES_OWN_SIGNAL),
                                      ("XGB-mkt-v1", FEATURES_MARKET_BLEND)]:
            train_clean = train.dropna(subset=feat_cols + ["target"])
            if len(train_clean) < 50:
                continue
            xgb_model = GBMClassifier(feature_cols=feat_cols)
            xgb_model.train(train_clean)
            test_clean = test.copy()
            # Fill NaN features with median for prediction
            for c in feat_cols:
                if test_clean[c].isna().any():
                    test_clean[c] = test_clean[c].fillna(train_clean[c].median())
            xgb_probs = xgb_model.predict_proba(test_clean)
            all_metrics[feat_name]["brier"].append(brier_score(y_test, xgb_probs))
            all_metrics[feat_name]["logloss"].append(log_loss(y_test, xgb_probs))
            if feat_name in clv_collectors:
                clv_collectors[feat_name].append((test.index.values, xgb_probs))

        # XGBoost V2
        for feat_name, feat_cols in [("XGB-own-v2", FEATURES_OWN_SIGNAL_V2),
                                      ("XGB-mkt-v2", FEATURES_MARKET_BLEND_V2)]:
            train_clean = train.dropna(subset=feat_cols + ["target"])
            if len(train_clean) < 50:
                continue
            xgb_model = GBMClassifier(feature_cols=feat_cols)
            xgb_model.train(train_clean)
            test_clean = test.copy()
            for c in feat_cols:
                if test_clean[c].isna().any():
                    test_clean[c] = test_clean[c].fillna(train_clean[c].median())
            xgb_probs = xgb_model.predict_proba(test_clean)
            all_metrics[feat_name]["brier"].append(brier_score(y_test, xgb_probs))
            all_metrics[feat_name]["logloss"].append(log_loss(y_test, xgb_probs))
            if feat_name in clv_collectors:
                clv_collectors[feat_name].append((test.index.values, xgb_probs))

            # Feature importance for V2 market-blend (last split)
            if feat_name == "XGB-mkt-v2" and split_idx == len(splits) - 1:
                importances = xgb_model.model.feature_importances_
                feat_imp = sorted(zip(feat_cols, importances),
                                  key=lambda x: x[1], reverse=True)

        # Collect Elo and DC predictions for CLV
        clv_collectors["Elo"].append((test.index.values, elo_probs))
        clv_collectors["Dixon-Coles"].append((test.index.values, dc_probs))

    # Print backtest results table
    logger.info(f"\n{'Model':<18} {'Brier':>8} {'Log Loss':>10} {'Splits':>7}")
    logger.info("─" * 45)
    for name in model_names:
        m = all_metrics[name]
        if m["brier"]:
            avg_b = np.mean(m["brier"])
            avg_l = np.mean(m["logloss"])
            logger.info(f"{name:<18} {avg_b:8.4f} {avg_l:10.4f} {len(m['brier']):>7}")
        else:
            logger.info(f"{name:<18} {'no data':>8}")

    # Feature importance for V2
    logger.info("\nXGB-mkt-v2 feature importance (last split):")
    for feat, imp in feat_imp:
        marker = " ← NEW" if feat in [
            "home_xg_overperformance", "away_xg_overperformance",
            "home_opp_adjusted_xg_for", "away_opp_adjusted_xg_for",
        ] else ""
        logger.info(f"  {feat:<30} {imp:.4f}{marker}")

    # Determine if V2 improves over V1
    v1_brier = np.mean(all_metrics["XGB-mkt-v1"]["brier"])
    v2_brier = np.mean(all_metrics["XGB-mkt-v2"]["brier"])
    v1_logloss = np.mean(all_metrics["XGB-mkt-v1"]["logloss"])
    v2_logloss = np.mean(all_metrics["XGB-mkt-v2"]["logloss"])
    v2_improves = v2_brier < v1_brier or v2_logloss < v1_logloss

    logger.info(f"\nXGB-mkt-v2 vs v1: Brier {v2_brier:.4f} vs {v1_brier:.4f} "
                f"({'better' if v2_brier < v1_brier else 'worse'}), "
                f"LogLoss {v2_logloss:.4f} vs {v1_logloss:.4f} "
                f"({'better' if v2_logloss < v1_logloss else 'worse'})")

    # ══════════════════════════════════════════════════════════════════════════
    # PART A: CLV analysis — Pinnacle vs William Hill
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PART A: CLV analysis — Pinnacle vs William Hill")
    logger.info("=" * 70)

    # Reassemble model predictions from walk-forward splits
    def assemble_predictions(collector):
        """Merge per-split predictions into arrays aligned with df_xg index."""
        idx_all, probs_all = [], []
        for idxs, probs in collector:
            idx_all.extend(idxs)
            probs_all.append(probs)
        if not probs_all:
            return np.array([]), np.array([])
        probs_all = np.vstack(probs_all)
        return np.array(idx_all), probs_all

    # Models to test CLV for
    clv_models = ["Elo", "Dixon-Coles", "XGB-mkt-v1"]
    if v2_improves:
        clv_models.append("XGB-mkt-v2")
        logger.info("V2 shows improvement → including XGB-mkt-v2 in CLV analysis")
    else:
        logger.info("V2 shows NO improvement → skipping XGB-mkt-v2 in CLV analysis")

    clv_results = []

    for model_name in clv_models:
        idx, probs = assemble_predictions(clv_collectors[model_name])
        if len(idx) == 0:
            continue

        # Build aligned DataFrame for this model's predictions
        pred_df = df_xg.loc[idx].copy()
        pred_df["_model_h"] = probs[:, 0]
        pred_df["_model_d"] = probs[:, 1]
        pred_df["_model_a"] = probs[:, 2]

        # --- Pinnacle CLV ---
        pin_valid = pred_df[["pin_open_home", "pin_open_draw", "pin_open_away",
                              "pin_close_home", "pin_close_draw", "pin_close_away"]].notna().all(axis=1)
        pin_sub = pred_df[pin_valid]
        if len(pin_sub) > 0:
            pin_open = pin_sub[["pin_open_home", "pin_open_draw", "pin_open_away"]].values
            pin_close = pin_sub[["pin_close_home", "pin_close_draw", "pin_close_away"]].values
            model_p = pin_sub[["_model_h", "_model_d", "_model_a"]].values
            result = run_clv_analysis(model_p, pin_open, pin_close, f"{model_name} vs Pinnacle")
            clv_results.append(result)

        # --- William Hill CLV ---
        wh_valid = pred_df[["wh_open_home", "wh_open_draw", "wh_open_away",
                             "wh_close_home", "wh_close_draw", "wh_close_away"]].notna().all(axis=1)
        wh_sub = pred_df[wh_valid]
        if len(wh_sub) > 0:
            wh_open = wh_sub[["wh_open_home", "wh_open_draw", "wh_open_away"]].values
            wh_close = wh_sub[["wh_close_home", "wh_close_draw", "wh_close_away"]].values
            model_p = wh_sub[["_model_h", "_model_d", "_model_a"]].values
            result = run_clv_analysis(model_p, wh_open, wh_close, f"{model_name} vs WH")
            clv_results.append(result)

    # Print CLV results table
    logger.info(f"\n{'Label':<30} {'Matches':>8} {'Bets':>6} {'%Pos CLV':>9} "
                f"{'Avg CLV%':>9} {'Edge-CLV r':>11} {'p-val':>8}")
    logger.info("─" * 85)
    for r in clv_results:
        corr_str = f"{r['edge_clv_corr']:.3f}" if pd.notna(r["edge_clv_corr"]) else "   -"
        pval_str = f"{r['edge_clv_pval']:.4f}" if pd.notna(r["edge_clv_pval"]) else "    -"
        pos_str = f"{r['pct_positive_clv']:.1f}%" if pd.notna(r["pct_positive_clv"]) else "   -"
        avg_str = f"{r['avg_clv_pct']:+.2f}%" if pd.notna(r["avg_clv_pct"]) else "    -"
        logger.info(f"{r['label']:<30} {r['n_matches']:>8} {r['n_bets']:>6} "
                    f"{pos_str:>9} {avg_str:>9} {corr_str:>11} {pval_str:>8}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # Any model with positive avg CLV?
    for r in clv_results:
        if pd.notna(r["avg_clv_pct"]) and r["avg_clv_pct"] > 0:
            logger.info(f"  POSITIVE avg CLV: {r['label']} at {r['avg_clv_pct']:+.2f}%")
    neg_all = all(pd.isna(r["avg_clv_pct"]) or r["avg_clv_pct"] <= 0 for r in clv_results)
    if neg_all:
        logger.info("  No model achieves positive average CLV against either bookmaker.")

    # Edge-CLV correlation summary
    for r in clv_results:
        if pd.notna(r["edge_clv_corr"]) and pd.notna(r["edge_clv_pval"]):
            sig = "significant" if r["edge_clv_pval"] < 0.05 else "not significant"
            logger.info(f"  Edge-CLV corr for {r['label']}: r={r['edge_clv_corr']:.3f} ({sig})")


if __name__ == "__main__":
    main()
