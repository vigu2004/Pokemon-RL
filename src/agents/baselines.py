"""Baseline policies: the random floor and a loop-safe greedy pilot.

These are non-learning ``policy(obs_dict) -> list[int]`` callables. They serve
four roles from the design (`docs/rl_design.md`):

1. **Sanity floor** — `RandomPolicy` is the win-rate bar every learned agent must
   clear.
2. **Reference opponent** — measure progress against `GreedyPolicy`.
3. **Self-play seed** — the league starts with these before checkpoints exist.
4. **Safety fallback** — `GreedyPolicy` is what the submission wraps its learned
   agent in when search times out or throws.

`GreedyPolicy` is deliberately *domain-general* (no Dragapult-specific lines yet)
and, crucially, **loop-safe**: on the main phase it only repeats one-shot
developing actions (attach/evolve/play — each consumes a once-per-turn resource
or a hand card) and takes repeatable Abilities only probabilistically, so a turn
always progresses toward an attack or end. A proper Dragapult pilot replaces this
later without changing the interface.
"""
from __future__ import annotations

import random

# Importing src.env first puts the engine on sys.path (and chdirs into it), so the
# subsequent `cg.api` import resolves regardless of who imports this module first.
from src.env import random_legal
from cg.api import OptionType, SelectType, to_observation_class

# OptionType groups we reason about on the MAIN selection.
_PLAY = OptionType.PLAY          # 7
_ATTACH = OptionType.ATTACH      # 8
_EVOLVE = OptionType.EVOLVE      # 9
_ABILITY = OptionType.ABILITY    # 10
_ATTACK = OptionType.ATTACK      # 13
_END = OptionType.END            # 14


class RandomPolicy:
    """Uniform-random over legal selections. The floor."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def __call__(self, obs: dict) -> list[int]:
        return random_legal(to_observation_class(obs), self.rng)


class GreedyPolicy:
    """A simple, loop-safe developing-then-attacking pilot.

    On the main phase: develop (attach → evolve → play) while such one-shot
    actions exist, use an Ability ~half the time, otherwise attack, otherwise end.
    Every other selection (targets, setup, discards, coin flips) is random-legal —
    good enough for a reference opponent and self-play seed.
    """

    def __init__(self, seed: int | None = None, ability_prob: float = 0.5):
        self.rng = random.Random(seed)
        self.ability_prob = ability_prob

    def __call__(self, obs: dict) -> list[int]:
        o = to_observation_class(obs)
        if o.select is None:
            return []
        if o.select.type != SelectType.MAIN:
            return random_legal(o, self.rng)
        return self._main(o.select.option)

    def _main(self, options) -> list[int]:
        by_type: dict[int, list[int]] = {}
        for i, op in enumerate(options):
            by_type.setdefault(int(op.type), []).append(i)

        # 1. One-shot developing actions (each terminates: consumes hand card or
        #    the once-per-turn energy attachment), so no infinite loop.
        for t in (_ATTACH, _EVOLVE, _PLAY):
            if t in by_type:
                return [self.rng.choice(by_type[t])]
        # 2. Abilities may be repeatable — take only sometimes so the turn advances.
        if _ABILITY in by_type and self.rng.random() < self.ability_prob:
            return [self.rng.choice(by_type[_ABILITY])]
        # 3. Attack (the win condition; ends the turn).
        if _ATTACK in by_type:
            return [self.rng.choice(by_type[_ATTACK])]
        # 4. Otherwise end the turn (or take whatever remains).
        if _END in by_type:
            return [by_type[_END][0]]
        return [self.rng.randrange(len(options))]
