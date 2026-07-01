"""CLI: evaluate baseline (or learned) policies head-to-head.

    .venv/bin/python scripts/eval.py --a greedy --b random --games 100

The A-policy pilots the Dragapult deck; B pilots the engine sample deck by default.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents import GreedyPolicy, RandomPolicy  # noqa: E402
from src.env import load_deck, sample_deck  # noqa: E402
from src.eval import evaluate  # noqa: E402

FACTORIES = {
    "random": lambda: RandomPolicy(),
    "greedy": lambda: GreedyPolicy(),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="greedy", choices=FACTORIES)
    ap.add_argument("--b", default="random", choices=FACTORIES)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--a-deck", default="dragapult_ex.draft")
    ap.add_argument("--b-deck", default=None, help="deck name; default = engine sample deck")
    args = ap.parse_args()

    deck_a = load_deck(args.a_deck)
    deck_b = load_deck(args.b_deck) if args.b_deck else sample_deck()

    res = evaluate(FACTORIES[args.a], FACTORIES[args.b], deck_a, deck_b,
                   games=args.games, progress=True)
    print(res.summary(args.a, args.b))


if __name__ == "__main__":
    main()
