"""Derive secondary market probabilities from a Dixon-Coles scoreline grid.

All functions take an (N, N) scoreline probability matrix where entry [h, a]
is P(home_goals=h, away_goals=a). The grid is assumed already normalised.
"""

import numpy as np


def over_under(grid: np.ndarray, line: float = 2.5) -> tuple[float, float]:
    """P(total goals > line), P(total goals <= line).

    For integer lines (e.g. 2.0), goals == line is under (standard convention).
    For half lines (e.g. 2.5), no push is possible.
    """
    mg = grid.shape[0]
    p_over = 0.0
    for h in range(mg):
        for a in range(mg):
            if h + a > line:
                p_over += grid[h, a]
    return float(p_over), float(1.0 - p_over)


def btts(grid: np.ndarray) -> tuple[float, float]:
    """P(both teams score >= 1), P(at least one team scores 0)."""
    # BTTS Yes = exclude row 0 (home=0) and column 0 (away=0)
    p_yes = grid[1:, 1:].sum()
    return float(p_yes), float(1.0 - p_yes)


def correct_score_top_n(grid: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    """Top N most likely scorelines with their probabilities.

    Returns list of ("H-A", probability) tuples, sorted by probability desc.
    """
    mg = grid.shape[0]
    scores = []
    for h in range(mg):
        for a in range(mg):
            scores.append((f"{h}-{a}", float(grid[h, a])))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]


def asian_handicap(
    grid: np.ndarray,
    line: float,
    side: str = "home",
) -> tuple[float, float, float]:
    """P(covers), P(push), P(loses) for an Asian handicap bet.

    Args:
        grid: (N, N) scoreline probability matrix.
        line: The handicap line from the home team's perspective (e.g. -0.5,
              -1.0, -0.75, +0.25). Negative means home gives goals.
        side: 'home' or 'away'. If 'away', we flip the perspective.

    Returns:
        (p_cover, p_push, p_lose) — probabilities for the specified side.

    Quarter-line convention (e.g. -0.75):
        A quarter line is split into two equal half-stakes at the adjacent
        half/whole lines. For example, line=-0.75 splits into:
          - Half the stake at -0.5
          - Half the stake at -1.0
        The outcome probabilities are averaged across both legs.
        This mirrors the standard Asian handicap settlement rule where
        half the stake is settled at each adjacent line.
    """
    if side == "away":
        # Flip perspective: away handicap of L is same as home handicap of -L
        line = -line

    # Check if quarter line (not a multiple of 0.5)
    remainder = abs(line) % 0.5
    is_quarter = abs(remainder - 0.25) < 1e-9

    if is_quarter:
        # Split into two half-lines
        line_lo = np.floor(line * 2) / 2  # round toward negative infinity
        line_hi = line_lo + 0.5
        p1 = _ah_single_line(grid, line_lo)
        p2 = _ah_single_line(grid, line_hi)
        # Average the two legs
        return tuple((np.array(p1) + np.array(p2)) / 2)
    else:
        return _ah_single_line(grid, line)


def _ah_single_line(
    grid: np.ndarray,
    line: float,
) -> tuple[float, float, float]:
    """Compute AH probabilities for a single (non-quarter) line.

    Returns (p_cover, p_push, p_lose) from home perspective.
    """
    mg = grid.shape[0]
    p_cover = 0.0
    p_push = 0.0
    p_lose = 0.0

    for h in range(mg):
        for a in range(mg):
            # Home adjusted margin = (h - a) + line
            adjusted = (h - a) + line
            p = grid[h, a]
            if adjusted > 1e-9:
                p_cover += p
            elif abs(adjusted) < 1e-9:
                p_push += p
            else:
                p_lose += p

    return float(p_cover), float(p_push), float(p_lose)
