"""Build and validate the Kaggle submission tarball.

    .venv/bin/python scripts/pack_submission.py --ckpt models/dragapult_ex.draft.pt

Stages the agent bundle (main.py, deck.csv, model.pt, cg/, src/, data/) into
build/submission/, validates it in a CLEAN subprocess (imports only from the
staging dir, plays a full main-vs-main game through the real engine — an illegal
selection raises there, so passing means the contract holds), then tars it up.
The checkpoint is stripped to model weights (no optimizer state).
"""
import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SRC_FILES = [
    "src/__init__.py", "src/env.py", "src/encode.py", "src/model.py",
    "src/agents/__init__.py", "src/agents/baselines.py", "src/agents/net_agent.py",
]
DATA_FILES = ["data/engine_cards.json", "data/engine_attacks.json"]

VALIDATOR = r'''
import sys
sys.path.insert(0, ".")

# Load the agent the way kaggle-environments does: exec the SOURCE with
# filename "<string>" and no __file__ in the namespace. Catches anything that
# only works under a normal import.
g = {}
exec(compile(open("main.py").read(), "<string>", "exec"), g)
agent = g["agent"]

assert len(g["_DECK"]) == 60, f"deck has {len(g['_DECK'])} cards"
assert g["_POLICIES"], "no policies built (net AND baselines failed)"
print("policies:", [n for n, _ in g["_POLICIES"]])

ans = agent({"select": None, "current": None, "logs": []})
assert ans == g["_DECK"], "deck-selection step wrong"

from cg.game import battle_start, battle_select, battle_finish
obs, sd = battle_start(list(g["_DECK"]), list(g["_DECK"]))
assert obs is not None, f"battle_start failed: {sd.errorPlayer} {sd.errorType}"
nsel = 0
try:
    while obs["current"]["result"] == -1 and nsel < 20000:
        obs = battle_select(agent(obs))  # raises IndexError if illegal
        nsel += 1
finally:
    battle_finish()
assert obs["current"]["result"] != -1, "game did not finish in 20000 selections"
print(f"OK: full game (exec-context agent), result={obs['current']['result']}, "
      f"turns={obs['current']['turn']}, selections={nsel}")
'''


def stage(ckpt: Path, deck_name: str, stage_dir: Path) -> None:
    import torch

    sys.path.insert(0, str(ROOT))
    from src.decklib import expand_deck, write_ids

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    shutil.copy(ROOT / "src" / "submit" / "main.py", stage_dir / "main.py")
    write_ids(stage_dir / "deck.csv", expand_deck(ROOT / "deck" / f"{deck_name}.csv"))

    ck = torch.load(ckpt, map_location="cpu")
    torch.save({k: ck[k] for k in ("model", "state_dim", "option_dim", "iter") if k in ck},
               stage_dir / "model.pt")

    shutil.copytree(ROOT / "engine" / "cg", stage_dir / "cg",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for rel in SRC_FILES + DATA_FILES:
        dst = stage_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dst)


def validate(stage_dir: Path) -> None:
    proc = subprocess.run([sys.executable, "-c", VALIDATOR], cwd=stage_dir,
                          capture_output=True, text=True, timeout=600)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"VALIDATION FAILED (exit {proc.returncode})")


def pack(stage_dir: Path, out: Path) -> None:
    # validate() runs python inside the staging dir, littering __pycache__ —
    # keep bytecode (and anything else compiled locally) out of the tarball.
    def _filter(ti: tarfile.TarInfo):
        name = Path(ti.name).name
        return None if name == "__pycache__" or name.endswith(".pyc") else ti

    with tarfile.open(out, "w:gz") as tar:
        for p in sorted(stage_dir.iterdir()):
            tar.add(p, arcname=p.name, filter=_filter)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--deck", default="dragapult_ex.draft")
    ap.add_argument("--stage", default=str(ROOT / "build" / "submission"))
    ap.add_argument("--out", default=str(ROOT / "build" / "submission.tar.gz"))
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    stage_dir = Path(args.stage)
    stage(Path(args.ckpt), args.deck, stage_dir)
    if not args.skip_validate:
        validate(stage_dir)
    pack(stage_dir, Path(args.out))


if __name__ == "__main__":
    main()
