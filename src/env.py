"""Engine driver — the foundation the whole learning stack plugs into.

The native `cg` library is a **process-global singleton**: there is exactly one
battle at a time (`Battle.battle_ptr`). So rather than a Gym `Env` object per
game, the natural primitive is a *driver* that runs one full game between two
policies and returns the outcome.

A **policy** is any callable ``policy(obs_dict) -> list[int]``: given the raw
observation dict the engine hands out, return the chosen option indices (a subset
of ``obs.select.option``, length in ``[minCount, maxCount]``, no duplicates).
This is exactly the Kaggle `agent()` contract, so the same policy object drops
straight into a submission.

`play_game` alternates control between the two policies based on
``obs.current.yourIndex`` — the engine tells us whose decision each step is and
hands that player a perspective-correct observation (the opponent's hand/deck are
hidden). Training-time trajectory recording is done by the *policy* (see
`src/agents`), keeping this driver agnostic about who is learning.
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
# Repo layout keeps the engine under engine/; the packed Kaggle submission has
# the cg/ package at the agent root instead. Support both.
_ENGINE = ROOT / "engine" if (ROOT / "engine").exists() else ROOT

# The cg package loads its native lib and resolves relative paths against the cwd,
# so it must be importable and we must run from engine/. Import lazily-safe here.
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))
if Path.cwd() != _ENGINE:
    os.chdir(_ENGINE)

from cg.api import Observation, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

Policy = Callable[[dict], list[int]]

# result / winner sentinels (engine: State.result)
ONGOING = -1
DRAW = 2


@dataclass
class GameResult:
    winner: int          # 0, 1, or DRAW(2); ABORTED(-1) if the guard tripped
    turns: int           # final turn counter
    selections: int      # total decisions made across both players
    aborted: bool = False

    def won_by(self, player_index: int) -> bool:
        return self.winner == player_index


# --------------------------------------------------------------------------- #
# Legal-action helpers                                                          #
# --------------------------------------------------------------------------- #
def is_terminal(o: Observation) -> bool:
    """True once the battle has finished (a winner/draw is decided)."""
    return o.current is not None and o.current.result != ONGOING


def to_move(o: Observation) -> int | None:
    """Index of the player whose decision it currently is (None during setup)."""
    return o.current.yourIndex if o.current is not None else None


def option_count(o: Observation) -> int:
    return len(o.select.option) if o.select is not None else 0


def random_legal(o: Observation, rng: random.Random | None = None) -> list[int]:
    """A guaranteed-legal selection: the safety fallback and the random policy.

    Length is clamped into ``[minCount, maxCount]`` and indices are unique and in
    range, satisfying every constraint `battle_select` enforces.
    """
    if o.select is None:
        return []
    rng = rng or random
    n = len(o.select.option)
    k = max(o.select.minCount, min(o.select.maxCount, n))
    return rng.sample(range(n), k) if n else []


def is_legal(o: Observation, sel: object) -> bool:
    """Validate a selection against the current select constraints."""
    if o.select is None:
        return sel == [] or sel is None
    if not isinstance(sel, list) or not all(isinstance(i, int) for i in sel):
        return False
    n = len(o.select.option)
    if not (o.select.minCount <= len(sel) <= o.select.maxCount):
        return False
    if len(set(sel)) != len(sel):
        return False
    return all(0 <= i < n for i in sel)


# --------------------------------------------------------------------------- #
# Game driver                                                                   #
# --------------------------------------------------------------------------- #
def play_game(
    policy0: Policy,
    policy1: Policy,
    deck0: list[int],
    deck1: list[int],
    *,
    max_selections: int = 20_000,
    safe: bool = True,
    rng: random.Random | None = None,
) -> GameResult:
    """Run one full game between two policies; return the outcome.

    ``policy0`` pilots ``deck0`` (player index 0), ``policy1`` pilots ``deck1``.
    With ``safe=True`` (the default), a policy that raises or returns an illegal
    selection is transparently replaced by a random legal move for that decision
    — so a buggy learner never crashes a training rollout. Turn ``safe=False``
    off during development to surface such bugs.
    """
    obs, start = battle_start(list(deck0), list(deck1))
    if obs is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start.errorPlayer} errorType={start.errorType}"
        )
    policies = (policy0, policy1)
    winner, turns, nsel, aborted = ONGOING, 0, 0, False
    try:
        while True:
            o = to_observation_class(obs)
            if is_terminal(o):
                winner, turns = o.current.result, o.current.turn
                break
            yi = o.current.yourIndex
            sel = _query(policies[yi], obs, o, safe, rng)
            obs = battle_select(sel)
            nsel += 1
            if nsel > max_selections:
                aborted = True
                turns = o.current.turn if o.current else 0
                break
    finally:
        battle_finish()
    return GameResult(winner=winner, turns=turns, selections=nsel, aborted=aborted)


def _query(policy: Policy, obs: dict, o: Observation, safe: bool, rng) -> list[int]:
    """Get a legal selection from a policy, falling back to random if unsafe."""
    if not safe:
        return policy(obs)
    try:
        sel = policy(obs)
    except Exception:
        return random_legal(o, rng)
    return sel if is_legal(o, sel) else random_legal(o, rng)


# --------------------------------------------------------------------------- #
# Deck convenience                                                              #
# --------------------------------------------------------------------------- #
def load_deck(name: str = "dragapult_ex.draft") -> list[int]:
    """Load our authoring-format deck by name from ``deck/``."""
    from src.decklib import expand_deck

    return expand_deck(ROOT / "deck" / f"{name}.csv")


def sample_deck() -> list[int]:
    """The engine's bundled sample deck (a generic opponent for smoke tests)."""
    from src.decklib import load_ids

    return load_ids(_ENGINE / "deck.csv")
