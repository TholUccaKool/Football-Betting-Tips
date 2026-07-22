"""Simplified Dixon-Coles Poisson model for football match prediction."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def _tau(x: int, y: int, lambda_: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-scoring correction factor."""
    if x == 0 and y == 0:
        return 1.0 - lambda_ * mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_ * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    """Dixon-Coles model with per-team attack/defense + home advantage + rho correction."""

    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self.teams: list[str] = []
        self.params: dict[str, float] = {}

    def _build_param_vector(self, teams: list[str]) -> np.ndarray:
        """Initial parameter vector: [attack_1..n, defense_1..n, home_adv, rho]."""
        n = len(teams)
        # attack params ~0, defense params ~0, home_adv ~0.25, rho ~0
        return np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.05]])

    def _unpack(self, params: np.ndarray, teams: list[str]) -> dict[str, float]:
        """Unpack parameter vector into named dict."""
        n = len(teams)
        d = {}
        for i, t in enumerate(teams):
            d[f"attack_{t}"] = params[i]
            d[f"defense_{t}"] = params[n + i]
        d["home_adv"] = params[2 * n]
        d["rho"] = params[2 * n + 1]
        return d

    def _neg_log_likelihood(self, params: np.ndarray, home_teams, away_teams,
                            home_goals, away_goals, teams: list[str]) -> float:
        """Negative log-likelihood for the Dixon-Coles model."""
        p = self._unpack(params, teams)
        n = len(teams)

        # Constraint: sum of attack params = n (via softmax-like normalization)
        log_lik = 0.0
        for ht, at, hg, ag in zip(home_teams, away_teams, home_goals, away_goals):
            lambda_ = np.exp(p[f"attack_{ht}"] + p[f"defense_{at}"] + p["home_adv"])
            mu = np.exp(p[f"attack_{at}"] + p[f"defense_{ht}"])

            hg, ag = int(hg), int(ag)
            tau = _tau(hg, ag, lambda_, mu, p["rho"])

            if tau <= 0 or lambda_ <= 0 or mu <= 0:
                return 1e10

            log_lik += (
                np.log(tau + 1e-10)
                + poisson.logpmf(hg, lambda_)
                + poisson.logpmf(ag, mu)
            )

        # Regularization: sum of attack params should be near 0
        attack_sum = sum(params[i] for i in range(n))
        log_lik -= 0.01 * attack_sum ** 2

        return -log_lik

    def fit(self, home_teams, away_teams, home_goals, away_goals) -> "DixonColesModel":
        """Fit model parameters via MLE."""
        self.teams = sorted(set(list(home_teams) + list(away_teams)))
        x0 = self._build_param_vector(self.teams)

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_teams, away_teams, home_goals, away_goals, self.teams),
            method="L-BFGS-B",
            options={"maxiter": 500, "disp": False},
        )

        self.params = self._unpack(result.x, self.teams)
        return self

    def predict_scoreline_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Return (max_goals+1, max_goals+1) matrix of scoreline probabilities."""
        lambda_ = np.exp(
            self.params[f"attack_{home_team}"]
            + self.params[f"defense_{away_team}"]
            + self.params["home_adv"]
        )
        mu = np.exp(
            self.params[f"attack_{away_team}"]
            + self.params[f"defense_{home_team}"]
        )
        rho = self.params["rho"]

        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                base = poisson.pmf(i, lambda_) * poisson.pmf(j, mu)
                tau = _tau(i, j, lambda_, mu, rho)
                matrix[i, j] = base * tau

        # Normalize
        matrix /= matrix.sum()
        return matrix

    def predict_proba(self, home_team: str, away_team: str) -> tuple[float, float, float]:
        """Return (p_home, p_draw, p_away) from the scoreline matrix."""
        m = self.predict_scoreline_matrix(home_team, away_team)
        n = m.shape[0]
        p_home = sum(m[i, j] for i in range(n) for j in range(n) if i > j)
        p_draw = sum(m[i, i] for i in range(n))
        p_away = sum(m[i, j] for i in range(n) for j in range(n) if i < j)
        return p_home, p_draw, p_away
