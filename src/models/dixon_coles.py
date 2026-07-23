"""Dixon-Coles (1997) Poisson model for football match prediction.

Simplification vs. the original paper: walk-forward evaluation window
(~12 months) handles recency at the split level. Exponential time-decay
weighting within each window is controlled by DC_TIME_DECAY_XI.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

# Half-life of ~180 days: xi = ln(2) / 180 ≈ 0.00385
DC_TIME_DECAY_XI = np.log(2) / 180.0


class DixonColesModel:
    """Dixon-Coles model with per-team attack/defense + home advantage + rho correction.

    Parameters are fit via MLE with analytic gradient, fully vectorized.
    Optional exponential time-decay weighting on training matches.
    """

    def __init__(self, max_goals: int = 7):
        self.max_goals = max_goals
        self.teams: list[str] = []
        self.params: dict[str, float] = {}
        self._n_teams = 0

    def fit(self, home_teams, away_teams, home_goals, away_goals,
            match_dates=None, reference_date=None,
            xi: float = DC_TIME_DECAY_XI) -> "DixonColesModel":
        """Fit model parameters via MLE.

        Args:
            home_teams, away_teams: arrays of team name strings.
            home_goals, away_goals: arrays of integer goal counts.
            match_dates: optional array of datetime-like match dates for
                time-decay weighting. If None, all matches weighted equally.
            reference_date: the date from which to measure recency (typically
                the first date of the test window). Required if match_dates
                is provided.
            xi: exponential decay rate. Default uses DC_TIME_DECAY_XI
                (~180-day half-life).
        """
        self.teams = sorted(set(list(home_teams) + list(away_teams)))
        self._n_teams = n = len(self.teams)
        team_to_idx = {t: i for i, t in enumerate(self.teams)}

        hi = np.array([team_to_idx[t] for t in home_teams], dtype=int)
        ai = np.array([team_to_idx[t] for t in away_teams], dtype=int)
        hg = np.asarray(home_goals, dtype=int)
        ag = np.asarray(away_goals, dtype=int)

        # Time-decay weights
        if match_dates is not None and reference_date is not None:
            dates = np.asarray(match_dates, dtype="datetime64[D]")
            ref = np.datetime64(reference_date, "D")
            days_before = (ref - dates).astype(float)
            weights = np.exp(-xi * np.clip(days_before, 0, None))
        else:
            weights = np.ones(len(hg))

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25
        x0[2 * n + 1] = -0.05

        # Precompute constants
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        hg_f = hg.astype(np.float64)
        ag_f = ag.astype(np.float64)
        log_hg_fact = gammaln(hg_f + 1.0)
        log_ag_fact = gammaln(ag_f + 1.0)

        def neg_ll_and_grad(params):
            atk = params[:n]
            dfn = params[n:2*n]
            home_adv = params[2*n]
            rho = params[2*n + 1]

            log_lam = atk[hi] + dfn[ai] + home_adv
            log_mu = atk[ai] + dfn[hi]
            lam = np.exp(log_lam)
            mu = np.exp(log_mu)

            ll_pois = (hg_f * log_lam - lam - log_hg_fact
                       + ag_f * log_mu - mu - log_ag_fact)

            tau = np.ones(len(hg))
            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m10] = 1.0 + mu[m10] * rho
            tau[m11] = 1.0 - rho

            if np.any(tau <= 0):
                return 1e10, np.zeros_like(params)

            log_tau = np.log(tau)
            # Weighted log-likelihood
            ll = np.sum(weights * (ll_pois + log_tau))

            atk_sum = atk.sum()
            ll -= 0.01 * atk_sum ** 2

            # --- Gradient (weighted) ---
            grad = np.zeros_like(params)

            dll_dloglam = hg_f - lam
            dll_dlogmu = ag_f - mu

            dlogtau_dlam = np.zeros(len(hg))
            dlogtau_dmu = np.zeros(len(hg))
            dlogtau_drho = np.zeros(len(hg))

            dlogtau_dlam[m00] = -mu[m00] * rho / tau[m00]
            dlogtau_dmu[m00] = -lam[m00] * rho / tau[m00]
            dlogtau_drho[m00] = -lam[m00] * mu[m00] / tau[m00]
            dlogtau_dlam[m01] = rho / tau[m01]
            dlogtau_drho[m01] = lam[m01] / tau[m01]
            dlogtau_dmu[m10] = rho / tau[m10]
            dlogtau_drho[m10] = mu[m10] / tau[m10]
            dlogtau_drho[m11] = -1.0 / tau[m11]

            dll_dloglam += dlogtau_dlam * lam
            dll_dlogmu += dlogtau_dmu * mu

            # Apply weights to per-match gradient contributions
            w_dll_dloglam = weights * dll_dloglam
            w_dll_dlogmu = weights * dll_dlogmu

            np.add.at(grad[:n], hi, w_dll_dloglam)
            np.add.at(grad[n:2*n], ai, w_dll_dloglam)
            grad[2*n] += w_dll_dloglam.sum()

            np.add.at(grad[:n], ai, w_dll_dlogmu)
            np.add.at(grad[n:2*n], hi, w_dll_dlogmu)

            grad[2*n + 1] = (weights * dlogtau_drho).sum()

            grad[:n] -= 0.02 * atk_sum

            return -ll, -grad

        bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 2.0), (-1.0, 1.0)]

        result = minimize(
            neg_ll_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 500},
        )

        self.params = {}
        for i, t in enumerate(self.teams):
            self.params[f"attack_{t}"] = result.x[i]
            self.params[f"defense_{t}"] = result.x[self._n_teams + i]
        self.params["home_adv"] = result.x[2 * self._n_teams]
        self.params["rho"] = result.x[2 * self._n_teams + 1]
        return self

    def predict_scoreline_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Return (max_goals+1, max_goals+1) matrix of scoreline probabilities."""
        lam = np.exp(
            self.params[f"attack_{home_team}"]
            + self.params[f"defense_{away_team}"]
            + self.params["home_adv"]
        )
        mu = np.exp(
            self.params[f"attack_{away_team}"]
            + self.params[f"defense_{home_team}"]
        )
        rho = self.params["rho"]
        mg = self.max_goals + 1

        h_pmf = poisson.pmf(np.arange(mg), lam)
        a_pmf = poisson.pmf(np.arange(mg), mu)
        matrix = np.outer(h_pmf, a_pmf)

        matrix[0, 0] *= (1.0 - lam * mu * rho)
        matrix[0, 1] *= (1.0 + lam * rho)
        matrix[1, 0] *= (1.0 + mu * rho)
        matrix[1, 1] *= (1.0 - rho)

        np.clip(matrix, 0.0, None, out=matrix)
        matrix /= matrix.sum()
        return matrix

    def predict_proba(self, home_team: str, away_team: str) -> tuple[float, float, float]:
        """Return (p_home, p_draw, p_away) from the scoreline matrix."""
        m = self.predict_scoreline_matrix(home_team, away_team)
        p_draw = np.trace(m)
        p_home = np.tril(m, -1).sum()
        p_away = np.triu(m, 1).sum()
        return float(p_home), float(p_draw), float(p_away)
