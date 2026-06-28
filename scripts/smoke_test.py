"""Smoke test: load the native engine, dump card data, run one random self-play game.

Run from the repo root:
    .venv/bin/python scripts/smoke_test.py

The cg package loads its native lib (libcg.dylib on macOS) at import time and
reads deck.csv from the cwd, so we add engine/ to sys.path and chdir there.
"""
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"
sys.path.insert(0, str(ENGINE))
os.chdir(ENGINE)  # so relative lib + deck.csv paths resolve

from cg.api import all_card_data, to_observation_class  # noqa: E402
from cg.game import battle_start, battle_finish, battle_select  # noqa: E402


def read_deck(path: Path) -> list[int]:
    ids = [int(x) for x in path.read_text().split("\n") if x.strip()]
    assert len(ids) == 60, f"deck must be 60 cards, got {len(ids)}"
    return ids


def random_agent(obs_dict: dict, deck: list[int]) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:  # deck-selection step
        return deck
    n = len(obs.select.option)
    k = obs.select.maxCount
    # length must be in [minCount, maxCount], no dups, each in [0, n)
    k = max(obs.select.minCount, min(k, n))
    return random.sample(range(n), k) if n else []


def main() -> None:
    # 1) dump structured card data from the engine itself
    cards = all_card_data()
    out = ROOT / "data" / "engine_cards.json"
    out.write_text(json.dumps([c.__dict__ for c in cards], default=str, ensure_ascii=False, indent=0))
    print(f"all_card_data(): {len(cards)} cards -> {out.relative_to(ROOT)}")

    # 2) run one random vs random game (both sides play the sample deck)
    deck = read_deck(ENGINE / "deck.csv")
    obs, start = battle_start(deck[:], deck[:])
    if obs is None:
        print(f"battle_start FAILED: errorPlayer={start.errorPlayer} errorType={start.errorType}")
        return

    turn_guard = 0
    while True:
        o = to_observation_class(obs)
        if o.current is not None and o.current.result != -1:
            print(f"GAME OVER after {turn_guard} selections. result(winner index)={o.current.result}, "
                  f"final turn={o.current.turn}")
            break
        whose = o.current.yourIndex if o.current else "setup"
        sel = random_agent(obs, deck)
        obs = battle_select(sel)
        turn_guard += 1
        if turn_guard % 200 == 0:
            print(f"  ...{turn_guard} selections so far (turn {o.current.turn if o.current else '?'})")
        if turn_guard > 20000:
            print("guard tripped — too many selections, aborting")
            break

    battle_finish()
    print("Engine OK ✅ — native lib loads, self-play loop runs to completion.")


if __name__ == "__main__":
    main()
