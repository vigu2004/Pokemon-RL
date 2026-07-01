"""Deck I/O for the PTCG agent.

Two deck formats live in this repo:

* **Authoring format** (`deck/*.csv`) — human-editable ``card_id,count,name`` rows
  with ``#`` comments. One line per distinct card.
* **Engine format** (`engine/deck.csv`) — 60 raw card IDs, one per line. This is
  what ``battle_start`` and the Kaggle submission expect.

``expand_deck`` turns the authoring format into the flat 60-ID list; ``load_ids``
reads an already-flat engine-format file.
"""
from __future__ import annotations

from pathlib import Path

DECK_SIZE = 60


def expand_deck(path: str | Path) -> list[int]:
    """Expand an authoring-format ``card_id,count,name`` deck into 60 raw IDs.

    Blank lines and ``#`` comments (whole-line or trailing) are ignored.
    """
    ids: list[int] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()  # drop trailing comments
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        cid, count = int(parts[0]), int(parts[1])
        ids.extend([cid] * count)
    _validate(ids, path)
    return ids


def load_ids(path: str | Path) -> list[int]:
    """Read a flat engine-format deck (one card ID per line) into a list."""
    ids = [int(x) for x in Path(path).read_text().split("\n") if x.strip()]
    _validate(ids, path)
    return ids


def write_ids(path: str | Path, ids: list[int]) -> None:
    """Write a flat engine-format deck (one card ID per line)."""
    _validate(ids, path)
    Path(path).write_text("\n".join(str(i) for i in ids) + "\n")


def _validate(ids: list[int], path: str | Path) -> None:
    if len(ids) != DECK_SIZE:
        raise ValueError(f"deck {path} must be {DECK_SIZE} cards, got {len(ids)}")


if __name__ == "__main__":
    import sys
    from collections import Counter

    root = Path(__file__).resolve().parent.parent
    src = sys.argv[1] if len(sys.argv) > 1 else root / "deck" / "dragapult_ex.draft.csv"
    ids = expand_deck(src)
    print(f"{src}: {len(ids)} cards, {len(set(ids))} distinct")
    for cid, n in Counter(ids).most_common():
        print(f"  {cid}: {n}")
