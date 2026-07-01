"""Evaluation harness — win rate of one policy vs another over many games.

The single north-star measurement tool. Because the coin/first-player choice and
shuffles make games high-variance, `evaluate` (1) plays many games and (2) swaps
which side each policy pilots every other game, so a side advantage can't inflate
the number. It returns the *A-policy's* win rate with a 95% Wilson interval.

Policies are passed as **factories** (``() -> policy``) so each game gets a fresh
policy (important once policies carry per-game state).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from src.env import DRAW, Policy, play_game

PolicyFactory = Callable[[], Policy]


@dataclass
class MatchResult:
    games: int
    wins: int          # A wins
    losses: int        # B wins
    draws: int
    aborts: int
    avg_turns: float

    @property
    def win_rate(self) -> float:
        decisive = self.wins + self.losses
        return self.wins / decisive if decisive else 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        """95% Wilson score interval on win rate over decisive games."""
        n = self.wins + self.losses
        if n == 0:
            return (0.0, 0.0)
        z = 1.96
        p = self.wins / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, center - half), min(1.0, center + half))

    def summary(self, label_a: str = "A", label_b: str = "B") -> str:
        lo, hi = self.ci95
        return (
            f"{label_a} vs {label_b}: {self.win_rate:.1%} win "
            f"[{lo:.1%}, {hi:.1%}]  "
            f"(W{self.wins} L{self.losses} D{self.draws} abort{self.aborts} "
            f"/ {self.games} games, avg {self.avg_turns:.0f} turns)"
        )


def evaluate(
    factory_a: PolicyFactory,
    factory_b: PolicyFactory,
    deck_a: list[int],
    deck_b: list[int],
    games: int = 100,
    *,
    progress: bool = False,
) -> MatchResult:
    """Play `games` games of policy A (deck_a) vs policy B (deck_b), swapping sides.

    Returns A's record. On even-indexed games A is player 0; on odd, A is player 1
    — so first-player/coin advantage is shared evenly.
    """
    wins = losses = draws = aborts = 0
    total_turns = 0
    for g in range(games):
        pol_a, pol_b = factory_a(), factory_b()
        a_is_p0 = (g % 2 == 0)
        if a_is_p0:
            res = play_game(pol_a, pol_b, deck_a, deck_b)
        else:
            res = play_game(pol_b, pol_a, deck_b, deck_a)
        total_turns += res.turns
        a_index = 0 if a_is_p0 else 1
        if res.aborted:
            aborts += 1
        elif res.winner == DRAW:
            draws += 1
        elif res.won_by(a_index):
            wins += 1
        else:
            losses += 1
        if progress and (g + 1) % max(1, games // 10) == 0:
            print(f"  {g + 1}/{games}: A {wins}W-{losses}L-{draws}D", flush=True)
    return MatchResult(
        games=games, wins=wins, losses=losses, draws=draws, aborts=aborts,
        avg_turns=total_turns / games if games else 0.0,
    )
