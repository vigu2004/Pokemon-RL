"""CLI: train the Dragapult agent by self-play PPO.

    .venv/bin/python scripts/train.py --iters 40 --games 64

Checkpoints land in models/. Watch the EVAL lines: win rate vs Random/Greedy is
the north-star.
"""
import argparse
import sys
from pathlib import Path

# Stream progress live even when stdout is redirected to a file (`> train.log`):
# by default Python block-buffers non-TTY stdout, so lines wouldn't appear until
# the process exits. Line-buffering flushes every line.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import load_deck  # noqa: E402
from src.selfplay import TrainConfig, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200,
                    help="TOTAL target iterations (a resumed run trains up to this)")
    ap.add_argument("--games", type=int, default=64,
                    help="self-play games per iteration")
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--deck", default="dragapult_ex.draft")
    ap.add_argument("--opp-decks", default=None,
                    help="comma-separated deck names under deck/ (e.g. 'meta/gholdengo,"
                         "meta/raging_bolt') added to the league piloted by Greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="checkpoint path (default models/<deck>.pt)")
    ap.add_argument("--resume", default=None,
                    help="resume from a checkpoint (defaults to --out if it exists when set)")
    args = ap.parse_args()

    deck = load_deck(args.deck)
    opp_decks = None
    if args.opp_decks:
        opp_decks = {name: load_deck(name) for name in args.opp_decks.split(",")}
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    # Resolve the checkpoint against the repo root — the engine chdirs the process
    # into engine/ on import, so relative paths would otherwise land there.
    ckpt = Path(args.out) if args.out else models / f"{args.deck}.pt"
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ckpt = str(ckpt)
    metrics = str(Path(ckpt).with_suffix(".metrics.csv"))

    resume = args.resume
    if resume and not Path(resume).is_absolute():
        resume = str(ROOT / resume)
    # Fresh run: start a clean metrics file (resume appends to keep continuity).
    if not resume and Path(metrics).exists():
        Path(metrics).unlink()

    cfg = TrainConfig(
        iters=args.iters, games_per_iter=args.games, ppo_epochs=args.ppo_epochs,
        lr=args.lr, eval_every=args.eval_every, eval_games=args.eval_games, seed=args.seed,
    )
    print(f"Training {args.deck}: up to {cfg.iters} iters x {cfg.games_per_iter} games -> {ckpt}")
    print(f"Metrics CSV (updates each eval): {metrics}")
    if resume:
        print(f"Resuming from: {resume}")
    train(deck, cfg, ckpt_path=ckpt, metrics_path=metrics, resume_from=resume,
          opp_decks=opp_decks, log=lambda m: print(m, flush=True))
    print(f"Done. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
