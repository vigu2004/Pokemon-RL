"""Loader for the PTCG AI Battle Challenge card data.

The competition's `EN_Card_Data.csv` is one row *per move/ability*, so a single
card can span several rows. This module collapses those rows back into one
structured `Card` per Card ID, which is the unit you actually reason about when
building decks and writing the agent.

Usage:
    from src.cardlib import load_cards
    cards = load_cards()                 # dict: card_id (int) -> Card
    print(cards[121])                    # Dragapult ex
    pikachu = [c for c in cards.values() if "Pikachu" in c.name]
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "EN_Card_Data.csv"

# CSV column names (verbatim from the header, including the awkward long one).
_C_ID = "Card ID"
_C_NAME = "Card Name"
_C_EXP = "Expansion"
_C_NO = "Collection No."
_C_STAGE = "Stage (Pokémon)/Type (Energy and Trainer)"
_C_RULE = "Rule"
_C_CATEGORY = "Category"
_C_PREV = "Previous stage"
_C_HP = "HP"
_C_TYPE = "Type"
_C_WEAK = "Weakness"
_C_RESIST = "Resistance (Type)"
_C_RETREAT = "Retreat"
_C_MOVE = "Move Name"
_C_COST = "Cost"
_C_DAMAGE = "Damage"
_C_EFFECT = "Effect Explanation"


@dataclass
class Move:
    """A move, ability, or rule line attached to a card."""

    name: str
    cost: str  # raw energy-cost string, e.g. "{G}{L}{M}" or "●"
    damage: str
    effect: str

    @property
    def is_ability(self) -> bool:
        return self.name.startswith("[Ability]")


@dataclass
class Card:
    card_id: int
    name: str
    expansion: str
    collection_no: str
    stage_or_type: str  # "Basic Pokémon", "Stage 1/2", "Supporter", "Item", "Basic Energy", ...
    rule: str           # "Pokémon ex", "Tera", "ACE SPEC", ...
    category: str
    previous_stage: str
    hp: str
    types: str
    weakness: str
    resistance: str
    retreat: str
    moves: list[Move] = field(default_factory=list)

    # --- convenience predicates -------------------------------------------
    @property
    def is_pokemon(self) -> bool:
        return "Pokémon" in self.stage_or_type or self.stage_or_type.endswith("Pokémon")

    @property
    def is_trainer(self) -> bool:
        return self.stage_or_type in {"Supporter", "Item", "Stadium", "Pokémon Tool"}

    @property
    def is_energy(self) -> bool:
        return "Energy" in self.stage_or_type

    @property
    def is_basic(self) -> bool:
        return self.stage_or_type == "Basic Pokémon"

    @property
    def is_ex(self) -> bool:
        return "ex" in self.rule.lower()

    def __repr__(self) -> str:
        bits = [f"#{self.card_id}", self.name, f"({self.expansion}-{self.collection_no})"]
        if self.is_pokemon and self.hp not in ("", "n/a"):
            bits.append(f"HP{self.hp} {self.types}")
        if self.moves:
            bits.append(f"[{len(self.moves)} move/ability]")
        return "Card<" + " ".join(bits) + ">"


def load_cards(csv_path: Path | str = DATA_CSV) -> dict[int, Card]:
    """Return {card_id: Card}, collapsing multi-row cards into one Card each."""
    cards: dict[int, Card] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw_id = (row.get(_C_ID) or "").strip()
            if not raw_id:
                # Continuation/blank rows: a handful exist; attach the move to the
                # most recently seen card if there is one, otherwise skip.
                if cards:
                    last = cards[next(reversed(cards))]
                    _add_move(last, row)
                continue
            cid = int(raw_id)
            if cid not in cards:
                cards[cid] = Card(
                    card_id=cid,
                    name=row[_C_NAME].strip(),
                    expansion=row[_C_EXP].strip(),
                    collection_no=row[_C_NO].strip(),
                    stage_or_type=row[_C_STAGE].strip(),
                    rule=(row.get(_C_RULE) or "").strip(),
                    category=(row.get(_C_CATEGORY) or "").strip(),
                    previous_stage=(row.get(_C_PREV) or "").strip(),
                    hp=(row.get(_C_HP) or "").strip(),
                    types=(row.get(_C_TYPE) or "").strip(),
                    weakness=(row.get(_C_WEAK) or "").strip(),
                    resistance=(row.get(_C_RESIST) or "").strip(),
                    retreat=(row.get(_C_RETREAT) or "").strip(),
                )
            _add_move(cards[cid], row)
    return cards


def _add_move(card: Card, row: dict) -> None:
    name = (row.get(_C_MOVE) or "").strip()
    if not name or name == "n/a":
        return
    card.moves.append(
        Move(
            name=name,
            cost=(row.get(_C_COST) or "").strip(),
            damage=(row.get(_C_DAMAGE) or "").strip(),
            effect=(row.get(_C_EFFECT) or "").strip(),
        )
    )


if __name__ == "__main__":
    cards = load_cards()
    print(f"Loaded {len(cards)} unique cards from {DATA_CSV}")
    for cid in (121, 120, 119, 133):  # Dragapult line + Dusknoir
        c = cards.get(cid)
        if c:
            print(c)
            for m in c.moves:
                print(f"    - {m.name}  cost={m.cost!r} dmg={m.damage!r} :: {m.effect[:70]}")
