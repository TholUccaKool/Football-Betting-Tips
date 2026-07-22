"""Unit tests for the Elo rating system."""

import unittest

from src.features.elo import EloRatingSystem


class TestEloRatingSystem(unittest.TestCase):

    def setUp(self):
        self.elo = EloRatingSystem(k_factor=20, home_advantage=100, initial_rating=1500)

    def test_initial_ratings(self):
        self.assertEqual(self.elo.get_rating("TeamA"), 1500)
        self.assertEqual(self.elo.get_rating("TeamB"), 1500)

    def test_winner_rating_increases(self):
        pre_home, pre_away = self.elo.get_pre_match_ratings("TeamA", "TeamB")
        new_home, new_away = self.elo.update("TeamA", "TeamB", 2, 0)
        self.assertGreater(new_home, pre_home)

    def test_loser_rating_decreases(self):
        pre_home, pre_away = self.elo.get_pre_match_ratings("TeamA", "TeamB")
        new_home, new_away = self.elo.update("TeamA", "TeamB", 2, 0)
        self.assertLess(new_away, pre_away)

    def test_draw_moves_ratings_toward_each_other(self):
        # Give teams different ratings first
        self.elo.ratings["Strong"] = 1600
        self.elo.ratings["Weak"] = 1400
        new_strong, new_weak = self.elo.update("Strong", "Weak", 1, 1)
        # After a draw, the stronger team loses rating (draw is a disappointment)
        self.assertLess(new_strong, 1600)
        # The weaker team gains rating (draw is a good result)
        self.assertGreater(new_weak, 1400)

    def test_upset_produces_larger_change(self):
        """A surprising result (upset) should produce a larger rating change."""
        # Scenario 1: Expected win (strong home team beats weak away)
        elo1 = EloRatingSystem(k_factor=20, home_advantage=100)
        elo1.ratings["Strong"] = 1700
        elo1.ratings["Weak"] = 1300
        new_strong_1, _ = elo1.update("Strong", "Weak", 2, 0)
        change_expected = abs(new_strong_1 - 1700)

        # Scenario 2: Upset (weak home team beats strong away)
        elo2 = EloRatingSystem(k_factor=20, home_advantage=100)
        elo2.ratings["Weak"] = 1300
        elo2.ratings["Strong"] = 1700
        new_weak_2, _ = elo2.update("Weak", "Strong", 2, 0)
        change_upset = abs(new_weak_2 - 1300)

        self.assertGreater(change_upset, change_expected)

    def test_pre_match_ratings_no_leakage(self):
        """Pre-match ratings should not reflect the current match result."""
        pre_h, pre_a = self.elo.get_pre_match_ratings("X", "Y")
        self.assertEqual(pre_h, 1500)
        self.assertEqual(pre_a, 1500)
        self.elo.update("X", "Y", 3, 0)
        # After update, ratings have changed
        self.assertNotEqual(self.elo.get_rating("X"), 1500)

    def test_reset(self):
        self.elo.update("A", "B", 1, 0)
        self.elo.reset()
        self.assertEqual(self.elo.get_rating("A"), 1500)


if __name__ == "__main__":
    unittest.main()
