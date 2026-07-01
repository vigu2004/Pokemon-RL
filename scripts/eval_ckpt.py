"""Load a trained checkpoint and evaluate the learned agent vs the baselines.

    .venv/bin/python scripts/eval_ckpt.py models/dragapult_v1.pt --games 200

Uses greedy (argmax) action selection — the deployment setting. Mirror matches
(both sides pilot the same deck) so the number is *piloting skill*, not deck luck.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.agents import GreedyPolicy, RandomPolicy  # noqa: E402
from src.agents.net_agent import NetPolicy  # noqa: E402
from src.encode import Encoder  # noqa: E402
from src.env import load_deck  # noqa: E402
from src.eval import evaluate  # noqa: E402
from src.model import PolicyValueNet  # noqa: E402


def load_net(ckpt_path: str, encoder: Encoder) -> PolicyValueNet:
    ck = torch.load(ckpt_path, map_location="cpu")
    net = PolicyValueNet(ck["state_dim"], ck["option_dim"])
    net.load_state_dict(ck["model"])
    net.eval()
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--deck", default="dragapult_ex.draft")
    ap.add_argument("--games", type=int, default=200)
    args = ap.parse_args()

    deck = load_deck(args.deck)
    encoder = Encoder(deck)
    net = load_net(args.ckpt, encoder)
    print(f"Loaded {args.ckpt} (iter {torch.load(args.ckpt, map_location='cpu').get('iter','?')})")

    for name, opp in [("Random", lambda: RandomPolicy()), ("Greedy", lambda: GreedyPolicy())]:
        res = evaluate(lambda: NetPolicy(net, encoder, greedy=True), opp,
                       deck, deck, games=args.games)
        print("  " + res.summary("Net", name))


if __name__ == "__main__":
    main()
