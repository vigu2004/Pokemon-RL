"""Extract representative meta decklists from mined episodes -> deck/meta/*.csv.

    .venv/bin/python scripts/make_meta_decks.py data/episodes_2026-07-01.jsonl --top 8

For each of the top-N archetypes by play count, picks the exact 60-card list
with the most wins (exact-list dedup, so it's a real list some team piloted,
not a Frankenstein average). Writes authoring-format (card_id,count) files
that `load_deck("meta/<slug>")` can read, with provenance in the header.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.mine_episodes import archetype, card_name  # noqa: E402


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower().replace("'", ""))
    return s.strip("_")[:48]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="output of mine_episodes.py")
    ap.add_argument("--top", type=int, default=8, help="archetypes to export")
    ap.add_argument("--outdir", default=str(ROOT / "deck" / "meta"))
    args = ap.parse_args()

    # exact list -> stats
    lists: dict[tuple, dict] = {}
    arch_games = collections.Counter()
    with open(args.jsonl) as f:
        for line in f:
            row = json.loads(line)
            rewards = row["rewards"] or [0, 0]
            for i in (0, 1):
                key = tuple(sorted(row["decks"][i]))
                st = lists.setdefault(key, {"games": 0, "wins": 0,
                                            "teams": collections.Counter()})
                st["games"] += 1
                st["wins"] += (rewards[i] or 0) > (rewards[1 - i] or 0)
                st["teams"][row["teams"][i]] += 1
                arch_games[archetype(list(key))] += 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    exported = []
    for arch, total in arch_games.most_common(args.top):
        candidates = [(st["wins"], st["games"], key) for key, st in lists.items()
                      if archetype(list(key)) == arch]
        wins, games, key = max(candidates)
        st = lists[key]
        slug = slugify(arch)
        path = outdir / f"{slug}.csv"
        counts = collections.Counter(key)
        with open(path, "w") as f:
            f.write(f"# Meta deck: {arch}\n")
            f.write(f"# Mined from ladder episodes ({Path(args.jsonl).name}); "
                    f"archetype seen in {total} games.\n")
            f.write(f"# This exact list: {games} games, {st['wins']/max(games,1):.1%} win rate, "
                    f"teams: {', '.join(t for t, _ in st['teams'].most_common(3))}\n")
            f.write("# card_id,count,name\n")
            for cid in sorted(counts):
                f.write(f"{cid},{counts[cid]}  # {card_name(cid)}\n")
        exported.append((f"meta/{slug}", games, st["wins"] / max(games, 1), arch))
        print(f"  wrote {path.name:44s} {games:4d} games {st['wins']/max(games,1):6.1%}  {arch}")

    print("\n--opp-decks " + ",".join(name for name, *_ in exported))


if __name__ == "__main__":
    main()
