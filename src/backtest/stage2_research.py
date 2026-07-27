"""Stage 2 research: leakage-safe ensemble, secondary-market CLV, new-league CLV.

Part A: Stacking ensemble (Elo + DC + XGB -> multinomial logistic meta-learner)
Part B: Over/Under 2.5 and Asian Handicap CLV using Dixon-Coles scoreline grid
Part C: Core-5 vs Secondary-4 league CLV comparison
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.engineer import build_features
from src.models.elo_baseline import fit_elo_calibrator
from src.models.dixon_coles import DixonColesModel
from src.models.gbm_classifier import GBMClassifier, FEATURES_MARKET_BLEND
from src.models.market_derivations import over_under, asian_handicap
from src.backtest.evaluate import walk_forward_splits, brier_score, log_loss, MIN_TRAIN_MATCHES
from src.backtest.clv import compute_clv, devig_odds, EDGE_THRESHOLD
from src.utils.io import get_raw_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CORE_LEAGUES = {"EPL", "La Liga", "Serie A", "Bundesliga", "Ligue 1"}
SECONDARY_LEAGUES = {"Championship", "Segunda División", "Eredivisie", "Primeira Liga"}


# ─── Data loading helpers ─────────────────────────────────────────────────────

def load_secondary_market_odds() -> pd.DataFrame:
    """Load Pinnacle O/U 2.5 and AH odds from raw match_history parquets.

    Returns DataFrame with date, home_team, away_team, and Pinnacle
    O/U opening/closing + AH opening/closing columns.
    """
    raw_dir = get_raw_dir() / "match_history"
    ou_cols = ["P>2.5", "P<2.5", "PC>2.5", "PC<2.5"]
    ah_cols = ["AHh", "PAHH", "PAHA", "AHCh", "PCAHH", "PCAHA"]
    needed = ou_cols + ah_cols

    frames = []
    for f in sorted(raw_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if not all(c in df.columns for c in needed):
            continue
        # Need date + team identifiers
        if "date" not in df.columns or "home_team" not in df.columns:
            continue
        sub = df[["date", "home_team", "away_team"] + needed].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        for c in needed:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return result


def _fit_dc(train_df, test_start_date):
    """Fit Dixon-Coles on last 24 months of training data."""
    dc_cutoff = train_df["date"].max() - pd.DateOffset(months=24)
    dc_train = train_df[train_df["date"] >= dc_cutoff]
    dc = DixonColesModel()
    dc.fit(
        dc_train["home_team"].values,
        dc_train["away_team"].values,
        dc_train["home_goals"].values.astype(int),
        dc_train["away_goals"].values.astype(int),
        match_dates=dc_train["date"].values,
        reference_date=test_start_date,
    )
    return dc


def _dc_probs_array(dc, df):
    """Generate DC probability array for a DataFrame of matches."""
    probs = []
    for _, row in df.iterrows():
        if row["home_team"] in dc.teams and row["away_team"] in dc.teams:
            probs.append(dc.predict_proba(row["home_team"], row["away_team"]))
        else:
            probs.append((1/3, 1/3, 1/3))
    return np.array(probs)


def _xgb_probs_array(xgb_model, df, train_df):
    """Generate XGB probability array, filling NaN features with train medians."""
    test_clean = df.copy()
    for c in xgb_model.feature_cols:
        if test_clean[c].isna().any():
            test_clean[c] = test_clean[c].fillna(train_df[c].median())
    return xgb_model.predict_proba(test_clean)


# ─── Part A: Ensemble ─────────────────────────────────────────────────────────

def run_part_a(df):
    """Build leakage-safe stacking ensemble and compare to base models."""
    logger.info("=" * 70)
    logger.info("PART A: Leakage-safe stacking ensemble")
    logger.info("=" * 70)

    splits = walk_forward_splits(df)
    logger.info(f"Walk-forward splits: {len(splits)}")

    model_names = ["Naive", "Market", "Elo", "Dixon-Coles", "XGB-mkt-v1", "Ensemble"]
    all_metrics = {m: {"brier": [], "logloss": []} for m in model_names}

    # Collectors for CLV (Part C needs ensemble predictions with indices)
    ensemble_collector = []  # (test_indices, ensemble_probs)
    base_collectors = {
        "Elo": [], "Dixon-Coles": [], "XGB-mkt-v1": [],
    }

    # Track meta-learner coefficients
    meta_coefs_all = []

    for split_idx, (train, test) in enumerate(splits):
        if len(train) < MIN_TRAIN_MATCHES:
            continue

        y_test = test["target"].values.astype(int)
        n_test = len(test)

        # ── Naive ──
        naive_probs = np.full((n_test, 3), 1/3)
        all_metrics["Naive"]["brier"].append(brier_score(y_test, naive_probs))
        all_metrics["Naive"]["logloss"].append(log_loss(y_test, naive_probs))

        # ── Market implied ──
        mkt_valid = test[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)
        if mkt_valid.sum() > 0:
            mkt_probs_full = np.full((n_test, 3), 1/3)
            mkt_sub = test[mkt_valid]
            raw = np.column_stack([
                1.0 / mkt_sub["home_odds"].values,
                1.0 / mkt_sub["draw_odds"].values,
                1.0 / mkt_sub["away_odds"].values,
            ])
            mkt_probs_full[mkt_valid.values] = raw / raw.sum(axis=1, keepdims=True)
            all_metrics["Market"]["brier"].append(brier_score(y_test, mkt_probs_full))
            all_metrics["Market"]["logloss"].append(log_loss(y_test, mkt_probs_full))

        # ══════════════════════════════════════════════════════════════════
        # ENSEMBLE: inner split for meta-learner training
        # ══════════════════════════════════════════════════════════════════

        # Inner split: last ~15% of training window by date for meta-learner
        train_sorted = train.sort_values("date")
        n_inner_val = max(1, int(len(train_sorted) * 0.15))
        inner_train = train_sorted.iloc[:-n_inner_val]
        inner_val = train_sorted.iloc[-n_inner_val:]

        # Fit base models on inner_train only
        # --- Elo on inner_train ---
        inner_elo = fit_elo_calibrator(inner_train)
        inner_elo_probs_val = inner_elo.predict_proba(inner_val[["elo_diff"]].values)

        # --- DC on inner_train ---
        inner_dc = _fit_dc(inner_train, inner_val["date"].min())
        inner_dc_probs_val = _dc_probs_array(inner_dc, inner_val)

        # --- XGB on inner_train ---
        inner_xgb_train = inner_train.dropna(subset=FEATURES_MARKET_BLEND + ["target"])
        inner_xgb_probs_val = None
        if len(inner_xgb_train) >= 50:
            inner_xgb = GBMClassifier(feature_cols=FEATURES_MARKET_BLEND)
            inner_xgb.train(inner_xgb_train)
            inner_xgb_probs_val = _xgb_probs_array(inner_xgb, inner_val, inner_xgb_train)

        # Build meta-learner features on inner_val (out-of-fold predictions)
        if inner_xgb_probs_val is not None:
            meta_X_val = np.hstack([
                inner_elo_probs_val,
                inner_dc_probs_val,
                inner_xgb_probs_val,
            ])
        else:
            # Fall back to just Elo + DC if XGB can't be trained
            meta_X_val = np.hstack([
                inner_elo_probs_val,
                inner_dc_probs_val,
            ])

        meta_y_val = inner_val["target"].values.astype(int)

        # Fit meta-learner (multinomial logistic regression) on inner_val
        meta = LogisticRegression(
            solver="lbfgs", max_iter=1000,
            C=1.0,  # light regularization
        )
        meta.fit(meta_X_val, meta_y_val)
        meta_coefs_all.append(meta.coef_.copy())

        # ── Now refit ALL base models on FULL training window ──
        full_elo = fit_elo_calibrator(train)
        full_elo_probs_test = full_elo.predict_proba(test[["elo_diff"]].values)

        full_dc = _fit_dc(train, test["date"].min())
        full_dc_probs_test = _dc_probs_array(full_dc, test)

        full_xgb_train = train.dropna(subset=FEATURES_MARKET_BLEND + ["target"])
        full_xgb_probs_test = None
        if len(full_xgb_train) >= 50:
            full_xgb = GBMClassifier(feature_cols=FEATURES_MARKET_BLEND)
            full_xgb.train(full_xgb_train)
            full_xgb_probs_test = _xgb_probs_array(full_xgb, test, full_xgb_train)

        # Record base model metrics
        all_metrics["Elo"]["brier"].append(brier_score(y_test, full_elo_probs_test))
        all_metrics["Elo"]["logloss"].append(log_loss(y_test, full_elo_probs_test))
        all_metrics["Dixon-Coles"]["brier"].append(brier_score(y_test, full_dc_probs_test))
        all_metrics["Dixon-Coles"]["logloss"].append(log_loss(y_test, full_dc_probs_test))
        if full_xgb_probs_test is not None:
            all_metrics["XGB-mkt-v1"]["brier"].append(brier_score(y_test, full_xgb_probs_test))
            all_metrics["XGB-mkt-v1"]["logloss"].append(log_loss(y_test, full_xgb_probs_test))

        # Collect for Part C
        base_collectors["Elo"].append((test.index.values, full_elo_probs_test))
        base_collectors["Dixon-Coles"].append((test.index.values, full_dc_probs_test))
        if full_xgb_probs_test is not None:
            base_collectors["XGB-mkt-v1"].append((test.index.values, full_xgb_probs_test))

        # ── Apply meta-learner to test predictions ──
        if full_xgb_probs_test is not None and inner_xgb_probs_val is not None:
            meta_X_test = np.hstack([
                full_elo_probs_test,
                full_dc_probs_test,
                full_xgb_probs_test,
            ])
        elif inner_xgb_probs_val is None:
            # Meta was trained without XGB, apply same way
            meta_X_test = np.hstack([
                full_elo_probs_test,
                full_dc_probs_test,
            ])
        else:
            # Meta was trained with XGB but we don't have it for test (shouldn't happen)
            meta_X_test = np.hstack([
                full_elo_probs_test,
                full_dc_probs_test,
                full_xgb_probs_test,
            ])

        ensemble_probs = meta.predict_proba(meta_X_test)
        all_metrics["Ensemble"]["brier"].append(brier_score(y_test, ensemble_probs))
        all_metrics["Ensemble"]["logloss"].append(log_loss(y_test, ensemble_probs))
        ensemble_collector.append((test.index.values, ensemble_probs))

        if (split_idx + 1) % 5 == 0:
            logger.info(f"  Completed split {split_idx + 1}/{len(splits)}")

    # Print comparison table
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

    # Meta-learner coefficients analysis
    logger.info("\nMeta-learner coefficients (averaged across splits):")
    avg_coefs = np.mean(meta_coefs_all, axis=0)  # shape (3, 9) or (3, 6)
    n_features = avg_coefs.shape[1]
    if n_features == 9:
        feat_names = [
            "Elo_H", "Elo_D", "Elo_A",
            "DC_H", "DC_D", "DC_A",
            "XGB_H", "XGB_D", "XGB_A",
        ]
    else:
        feat_names = [
            "Elo_H", "Elo_D", "Elo_A",
            "DC_H", "DC_D", "DC_A",
        ]
    outcome_names = ["Home", "Draw", "Away"]
    logger.info(f"  {'Feature':<12}" + "".join(f"{o:>10}" for o in outcome_names))
    logger.info("  " + "─" * 42)
    for j, feat in enumerate(feat_names):
        vals = "".join(f"{avg_coefs[i, j]:+10.3f}" for i in range(3))
        logger.info(f"  {feat:<12}{vals}")

    # Summarize which base model the meta-learner leans on most
    if n_features == 9:
        # Sum absolute coefficients by model group
        elo_weight = np.abs(avg_coefs[:, 0:3]).sum()
        dc_weight = np.abs(avg_coefs[:, 3:6]).sum()
        xgb_weight = np.abs(avg_coefs[:, 6:9]).sum()
        total_w = elo_weight + dc_weight + xgb_weight
        logger.info(f"\n  Relative weight by |coef| sum:")
        logger.info(f"    Elo:  {elo_weight/total_w*100:.1f}%")
        logger.info(f"    DC:   {dc_weight/total_w*100:.1f}%")
        logger.info(f"    XGB:  {xgb_weight/total_w*100:.1f}%")

    return ensemble_collector, base_collectors


# ─── Part B: Secondary market CLV ─────────────────────────────────────────────

def run_part_b(df):
    """Run CLV analysis for Over/Under 2.5 and Asian Handicap using DC grid."""
    logger.info("\n" + "=" * 70)
    logger.info("PART B: Secondary market CLV (O/U 2.5, Asian Handicap)")
    logger.info("=" * 70)

    # Load secondary market odds
    logger.info("\nLoading Pinnacle O/U and AH odds from raw data...")
    sec_odds = load_secondary_market_odds()
    logger.info(f"  Rows loaded: {len(sec_odds)}")

    # Merge into feature matrix
    df["_date_key"] = df["date"].dt.normalize()
    sec_odds["_date_key"] = sec_odds["date"].dt.normalize()
    n_before = len(df)
    df = df.merge(
        sec_odds.drop(columns=["date"]),
        on=["_date_key", "home_team", "away_team"],
        how="left",
    )
    df.drop(columns=["_date_key"], inplace=True)
    if len(df) > n_before:
        df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
        logger.info(f"  (deduped from {n_before} to {len(df)})")

    ou_mask = df[["P>2.5", "P<2.5", "PC>2.5", "PC<2.5"]].notna().all(axis=1)
    ah_mask = df[["AHh", "PAHH", "PAHA", "AHCh", "PCAHH", "PCAHA"]].notna().all(axis=1)
    logger.info(f"  O/U 2.5 coverage: {ou_mask.sum()}/{len(df)}")
    logger.info(f"  AH coverage:      {ah_mask.sum()}/{len(df)}")
    logger.info(f"\n  Note: BTTS has NO market odds available — excluded from CLV analysis.")

    # Walk-forward with DC for scoreline grids
    splits = walk_forward_splits(df)
    logger.info(f"  Walk-forward splits: {len(splits)}")

    # Collect per-match: model probs, opening odds, closing odds for O/U and AH
    ou_records = []
    ah_records = []

    for split_idx, (train, test) in enumerate(splits):
        if len(train) < MIN_TRAIN_MATCHES:
            continue

        # Fit DC on training data
        dc = _fit_dc(train, test["date"].min())

        for idx, row in test.iterrows():
            ht, at = row["home_team"], row["away_team"]
            if ht not in dc.teams or at not in dc.teams:
                continue

            grid = dc.predict_scoreline_matrix(ht, at)

            # ── Over/Under 2.5 ──
            if (pd.notna(row.get("P>2.5")) and pd.notna(row.get("P<2.5"))
                    and pd.notna(row.get("PC>2.5")) and pd.notna(row.get("PC<2.5"))):
                p_over, p_under = over_under(grid, line=2.5)
                ou_records.append({
                    "idx": idx,
                    "model_probs": np.array([p_over, p_under]),
                    "open_odds": np.array([row["P>2.5"], row["P<2.5"]]),
                    "close_odds": np.array([row["PC>2.5"], row["PC<2.5"]]),
                    "actual_total": row["home_goals"] + row["away_goals"],
                })

            # ── Asian Handicap ──
            if (pd.notna(row.get("AHh")) and pd.notna(row.get("PAHH"))
                    and pd.notna(row.get("PAHA")) and pd.notna(row.get("AHCh"))
                    and pd.notna(row.get("PCAHH")) and pd.notna(row.get("PCAHA"))):
                ah_line = row["AHh"]
                p_cover, p_push, p_lose = asian_handicap(grid, ah_line, side="home")
                # For a two-way market, fold push into half-win/half-lose
                # (standard for CLV analysis with Asian odds)
                if p_push > 0:
                    p_home_ah = p_cover + p_push * 0.5
                    p_away_ah = p_lose + p_push * 0.5
                else:
                    p_home_ah = p_cover
                    p_away_ah = p_lose
                ah_records.append({
                    "idx": idx,
                    "model_probs": np.array([p_home_ah, p_away_ah]),
                    "open_odds": np.array([row["PAHH"], row["PAHA"]]),
                    "close_odds": np.array([row["PCAHH"], row["PCAHA"]]),
                    "ah_line": ah_line,
                    "actual_margin": row["home_goals"] - row["away_goals"],
                })

        if (split_idx + 1) % 5 == 0:
            logger.info(f"    Completed split {split_idx + 1}/{len(splits)}")

    # ── O/U CLV analysis ──
    logger.info(f"\n  O/U 2.5 predictions: {len(ou_records)}")
    _run_two_way_clv(ou_records, "Over/Under 2.5", ["Over", "Under"])

    # ── AH CLV analysis ──
    logger.info(f"\n  AH predictions: {len(ah_records)}")
    _run_two_way_clv(ah_records, "Asian Handicap", ["AH Home", "AH Away"])

    return df


def _run_two_way_clv(records, market_name, outcome_names):
    """Run CLV analysis for a two-outcome market."""
    if not records:
        logger.info(f"  No {market_name} records to analyse.")
        return

    n = len(records)
    model_probs = np.array([r["model_probs"] for r in records])
    open_odds = np.array([r["open_odds"] for r in records])
    close_odds = np.array([r["close_odds"] for r in records])

    # Devig opening and closing
    open_raw = 1.0 / open_odds
    open_impl = open_raw / open_raw.sum(axis=1, keepdims=True)
    close_raw = 1.0 / close_odds
    close_impl = close_raw / close_raw.sum(axis=1, keepdims=True)

    results_by_outcome = []
    all_bets_clv = []

    for oc in range(2):
        edge = model_probs[:, oc] - open_impl[:, oc]
        is_bet = edge > EDGE_THRESHOLD
        closing_fair_odds = 1.0 / close_impl[:, oc]
        clv_pct = (open_odds[:, oc] / closing_fair_odds - 1.0) * 100.0

        n_bets = is_bet.sum()
        if n_bets > 0:
            bet_clv = clv_pct[is_bet]
            pct_pos = (bet_clv > 0).mean() * 100
            avg_clv = bet_clv.mean()
            all_bets_clv.extend(bet_clv.tolist())
        else:
            pct_pos = np.nan
            avg_clv = np.nan

        results_by_outcome.append({
            "outcome": outcome_names[oc],
            "n_bets": n_bets,
            "pct_positive": pct_pos,
            "avg_clv": avg_clv,
        })

    # Print results
    logger.info(f"\n  {market_name} CLV (DC model vs Pinnacle, threshold={EDGE_THRESHOLD}):")
    logger.info(f"  {'Outcome':<12} {'Matches':>8} {'Bets':>6} {'%Pos CLV':>10} {'Avg CLV%':>10}")
    logger.info("  " + "─" * 50)
    for r in results_by_outcome:
        pos_str = f"{r['pct_positive']:.1f}%" if pd.notna(r["pct_positive"]) else "-"
        avg_str = f"{r['avg_clv']:+.2f}%" if pd.notna(r["avg_clv"]) else "-"
        logger.info(f"  {r['outcome']:<12} {n:>8} {r['n_bets']:>6} {pos_str:>10} {avg_str:>10}")

    total_bets = sum(r["n_bets"] for r in results_by_outcome)
    if all_bets_clv:
        overall_avg = np.mean(all_bets_clv)
        overall_pct_pos = (np.array(all_bets_clv) > 0).mean() * 100
        logger.info(f"  {'TOTAL':<12} {n:>8} {total_bets:>6} "
                    f"{overall_pct_pos:.1f}%{'':<4} {overall_avg:+.2f}%")


# ─── Part C: Core vs Secondary league CLV ─────────────────────────────────────

def run_part_c(df, ensemble_collector, base_collectors):
    """Compare CLV between core-5 and secondary-4 leagues."""
    logger.info("\n" + "=" * 70)
    logger.info("PART C: Core-5 vs Secondary-4 league CLV")
    logger.info("=" * 70)

    # Assemble predictions
    def assemble(collector):
        idx_all, probs_all = [], []
        for idxs, probs in collector:
            idx_all.extend(idxs)
            probs_all.append(probs)
        if not probs_all:
            return np.array([]), np.array([])
        return np.array(idx_all), np.vstack(probs_all)

    models_to_test = {
        "Ensemble": ensemble_collector,
        "Elo": base_collectors["Elo"],
        "Dixon-Coles": base_collectors["Dixon-Coles"],
        "XGB-mkt-v1": base_collectors["XGB-mkt-v1"],
    }

    pin_open_cols = ["pin_open_home", "pin_open_draw", "pin_open_away"]
    pin_close_cols = ["pin_close_home", "pin_close_draw", "pin_close_away"]

    for league_group_name, league_set in [("Core-5", CORE_LEAGUES), ("Secondary-4", SECONDARY_LEAGUES)]:
        logger.info(f"\n{'─' * 70}")
        logger.info(f"  {league_group_name} leagues: {sorted(league_set)}")
        logger.info(f"{'─' * 70}")

        clv_results = []

        for model_name, collector in models_to_test.items():
            idx, probs = assemble(collector)
            if len(idx) == 0:
                continue

            pred_df = df.loc[idx].copy()
            pred_df["_model_h"] = probs[:, 0]
            pred_df["_model_d"] = probs[:, 1]
            pred_df["_model_a"] = probs[:, 2]

            # Filter to league group
            league_mask = pred_df["league"].isin(league_set)
            pred_sub = pred_df[league_mask]

            # Filter to matches with Pinnacle open+close
            pin_valid = pred_sub[pin_open_cols + pin_close_cols].notna().all(axis=1)
            pin_sub = pred_sub[pin_valid]

            if len(pin_sub) == 0:
                clv_results.append({
                    "model": model_name, "n_matches": 0, "n_bets": 0,
                    "pct_pos": np.nan, "avg_clv": np.nan,
                })
                continue

            model_p = pin_sub[["_model_h", "_model_d", "_model_a"]].values
            pin_open = pin_sub[pin_open_cols].values
            pin_close = pin_sub[pin_close_cols].values

            clv_df = compute_clv(model_p, pin_open, pin_close)
            bets = clv_df[clv_df["is_bet"]]
            n_bets = len(bets)
            n_matches = len(pin_sub)

            if n_bets > 0:
                pct_pos = (bets["clv_pct"] > 0).mean() * 100
                avg_clv = bets["clv_pct"].mean()
            else:
                pct_pos = np.nan
                avg_clv = np.nan

            clv_results.append({
                "model": model_name, "n_matches": n_matches, "n_bets": n_bets,
                "pct_pos": pct_pos, "avg_clv": avg_clv,
            })

        # Print table
        logger.info(f"\n  {'Model':<18} {'Matches':>8} {'Bets':>6} {'%Pos CLV':>10} {'Avg CLV%':>10}")
        logger.info("  " + "─" * 55)
        for r in clv_results:
            pos_str = f"{r['pct_pos']:.1f}%" if pd.notna(r["pct_pos"]) else "-"
            avg_str = f"{r['avg_clv']:+.2f}%" if pd.notna(r["avg_clv"]) else "-"
            logger.info(f"  {r['model']:<18} {r['n_matches']:>8} {r['n_bets']:>6} "
                        f"{pos_str:>10} {avg_str:>10}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("Building feature matrix...")
    df = build_features()
    logger.info(f"Feature matrix: {df.shape}")
    logger.info(f"Leagues: {sorted(df['league'].dropna().unique())}")
    logger.info(f"Core-5 matches: {df['league'].isin(CORE_LEAGUES).sum()}")
    logger.info(f"Secondary-4 matches: {df['league'].isin(SECONDARY_LEAGUES).sum()}")

    # Part A
    ensemble_collector, base_collectors = run_part_a(df)

    # Part B (needs its own DC walk-forward for scoreline grids)
    df = run_part_b(df)

    # Part C
    run_part_c(df, ensemble_collector, base_collectors)

    # ── Final summary ──
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("""
Stage 2 tested three angles:
  A) Stacking ensemble (Elo + DC + XGB -> logistic meta-learner)
  B) Secondary markets (O/U 2.5, Asian Handicap) via Dixon-Coles
  C) Secondary leagues (Championship, Segunda, Eredivisie, Primeira Liga)

See tables above for all results. Any positive CLV findings are flagged
inline. BTTS is excluded from CLV analysis (no market odds available).
""")


if __name__ == "__main__":
    main()
