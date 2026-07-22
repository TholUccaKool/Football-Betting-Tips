"""Elo rating system for football match prediction."""

from collections import defaultdict


class EloRatingSystem:
    """Configurable Elo rating tracker with home advantage."""

    def __init__(self, k_factor: float = 20, home_advantage: float = 100, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.ratings: dict[str, float] = defaultdict(lambda: self.initial_rating)

    def get_rating(self, team: str) -> float:
        return self.ratings[team]

    def get_pre_match_ratings(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return (home_rating, away_rating) BEFORE the match is played."""
        return self.ratings[home_team], self.ratings[away_team]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for team A against team B (no home advantage baked in)."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> tuple[float, float]:
        """Update ratings after a match. Returns (new_home_rating, new_away_rating).

        Home team gets home_advantage added to their rating for expectation calc,
        but the stored rating is updated without the temporary boost.
        """
        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]

        # Expected scores with home advantage
        exp_home = self.expected_score(home_rating + self.home_advantage, away_rating)
        exp_away = 1.0 - exp_home

        # Actual scores: 1 for win, 0.5 for draw, 0 for loss
        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        # Update ratings
        new_home = home_rating + self.k_factor * (actual_home - exp_home)
        new_away = away_rating + self.k_factor * (actual_away - exp_away)

        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away

        return new_home, new_away

    def reset(self) -> None:
        """Reset all ratings to initial values."""
        self.ratings.clear()
