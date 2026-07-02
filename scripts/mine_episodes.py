"""Mine Kaggle ladder episode replays: decks, results, archetypes.

Reads episode JSONs straight out of the daily dataset zip (no 21GB extract):

    .venv/bin/python scripts/mine_episodes.py path/to/episodes-2026-07-01.zip \
        --out data/episodes_2026-07-01.jsonl

Pass 1 writes one JSONL row per episode: {episode_id, teams, rewards, decks,
n_steps}. Pass 2 (automatic) aggregates and prints the meta report: archetype
frequency, archetype win rates, top teams and their decks. Re-running with an
existing --out skips pass 1.
"""
import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CARDS = {c["cardId"]: c for c in json.load(open(ROOT / "data" / "engine_cards.json"))}


def card_name(cid: int) -> str:
    return CARDS.get(cid, {}).get("name", f"#{cid}")


def archetype(deck: list[int]) -> str:
    """Label a deck by its marquee attackers: ex/tera/mega Pokémon, highest
    count then highest HP. Falls back to the beefiest Pokémon for ex-less decks."""
    counts = collections.Counter(deck)
    marquee = [
        (n, CARDS[cid]["hp"], cid) for cid, n in counts.items()
        if cid in CARDS and (CARDS[cid]["ex"] or CARDS[cid]["tera"] or CARDS[cid]["megaEx"])
    ]
    if not marquee:
        marquee = [
            (n, CARDS[cid]["hp"], cid) for cid, n in counts.items()
            if cid in CARDS and CARDS[cid]["hp"] >= 90
        ]
    if not marquee:
        return "unknown"
    marquee.sort(reverse=True)
    return " / ".join(card_name(cid) for _, _, cid in marquee[:2])


def extract_episode(raw: bytes) -> dict | None:
    d = json.loads(raw)
    steps = d.get("steps") or []
    if len(steps) < 2:
        return None
    decks = [steps[1][i].get("action") or [] for i in range(2)]
    if any(len(dk) != 60 for dk in decks):
        return None
    return {
        "episode_id": d.get("info", {}).get("EpisodeId"),
        "teams": d.get("info", {}).get("TeamNames", ["?", "?"]),
        "rewards": d.get("rewards"),
        "decks": decks,
        "n_steps": len(steps),
    }


def pass1_extract(zip_path: Path, out_path: Path) -> None:
    n_ok = n_bad = 0
    with zipfile.ZipFile(zip_path) as zf, open(out_path, "w") as out:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        for i, name in enumerate(names):
            try:
                row = extract_episode(zf.read(name))
            except Exception:
                row = None
            if row is None:
                n_bad += 1
            else:
                out.write(json.dumps(row) + "\n")
                n_ok += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(names)} episodes ({n_bad} skipped)", flush=True)
    print(f"Extracted {n_ok} episodes ({n_bad} skipped) -> {out_path}")


def pass2_report(out_path: Path, top: int = 25) -> None:
    arch_games = collections.Counter()
    arch_wins = collections.Counter()
    team_games = collections.Counter()
    team_wins = collections.Counter()
    team_arch = {}
    n = 0
    with open(out_path) as f:
        for line in f:
            row = json.loads(line)
            n += 1
            rewards = row["rewards"] or [0, 0]
            for i in (0, 1):
                arch = archetype(row["decks"][i])
                team = row["teams"][i]
                won = (rewards[i] or 0) > (rewards[1 - i] or 0)
                arch_games[arch] += 1
                arch_wins[arch] += won
                team_games[team] += 1
                team_wins[team] += won
                team_arch[team] = arch

    print(f"\n=== {n} episodes, {len(team_games)} teams, {len(arch_games)} archetypes ===")
    print(f"\n--- Archetypes by play count (win rate) ---")
    for arch, g in arch_games.most_common(top):
        print(f"  {arch_wins[arch]/g:6.1%}  {g:5d}  {arch}")
    print(f"\n--- Top teams by win rate (>= 20 games) ---")
    ranked = [(team_wins[t] / g, g, t) for t, g in team_games.items() if g >= 20]
    for wr, g, t in sorted(ranked, reverse=True)[:top]:
        print(f"  {wr:6.1%}  {g:5d}  {t}  [{team_arch[t]}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", nargs="?", help="daily episodes .zip (skippable if --out exists)")
    ap.add_argument("--out", default=None, help="episodes JSONL (default: data/episodes_<zipname>.jsonl)")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "data" / (Path(args.zip_path).stem + ".jsonl")
    if not out.exists():
        if not args.zip_path:
            ap.error(f"{out} does not exist and no zip_path given")
        pass1_extract(Path(args.zip_path), out)
    else:
        print(f"Using existing {out}")
    pass2_report(out, args.top)


if __name__ == "__main__":
    main()
