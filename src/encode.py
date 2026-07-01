"""Observation + option encoding: obs_dict -> fixed-length tensors.

The learned policy has to consume two variable things and produce a decision:

* a **state** (board, hands-counts, prizes, whose turn, what kind of decision) —
  encoded to a fixed-length vector ``state[F]``;
* a **variable list of legal options** — each encoded to ``opt[G]``, giving
  ``options[n, G]``. The net scores each option row and picks among them, so the
  action head naturally handles the dynamic action set (see `src/model.py`).

Everything is perspective-correct: "me" = the player to move (``yourIndex``),
"opp" = the other. Hidden info (opp hand/deck contents) is never encoded — only
public counts — which is exactly the imperfect-information the agent must handle.

Card/attack static features come from the engine's own dumps
(`data/engine_cards.json`, `data/engine_attacks.json`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.env import ROOT
from cg.api import AreaType, OptionType, to_observation_class

# ---- enum sizes (leave headroom; the API notes enums may gain elements) ----
N_CARDTYPE = 7
N_ENERGY = 12
N_AREA = 13          # AreaType is 1..12
N_OPTTYPE = 17       # OptionType 0..16
N_SELTYPE = 11       # SelectType 0..10
N_CONTEXT = 49       # SelectContext 0..48

_HP_SCALE = 350.0
_DMG_SCALE = 300.0


# --------------------------------------------------------------------------- #
# Static card / attack feature tables                                          #
# --------------------------------------------------------------------------- #
def _load_json(name: str) -> list[dict]:
    return json.loads((ROOT / "data" / name).read_text())


def _one_hot(idx: int | None, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if idx is not None and 0 <= int(idx) < size:
        v[int(idx)] = 1.0
    return v


class _CardTable:
    """Static per-card and per-attack features, keyed by engine ID."""

    CARD_F = 1 + N_CARDTYPE + N_ENERGY + 6  # hp + type + energy + flags/scalars
    ATTACK_F = 2 + N_ENERGY                 # damage,#energy + energy multi-hot

    def __init__(self):
        self.cards = {c["cardId"]: c for c in _load_json("engine_cards.json")}
        self.attacks = {a["attackId"]: a for a in _load_json("engine_attacks.json")}
        self._card_cache: dict[int, np.ndarray] = {}
        self._atk_cache: dict[int, np.ndarray] = {}

    def card(self, card_id: int | None) -> np.ndarray:
        if card_id is None:
            return np.zeros(self.CARD_F, dtype=np.float32)
        if card_id in self._card_cache:
            return self._card_cache[card_id]
        c = self.cards.get(card_id)
        if c is None:
            v = np.zeros(self.CARD_F, dtype=np.float32)
        else:
            best_dmg = max((self.attacks.get(a, {}).get("damage", 0) for a in c["attacks"]),
                           default=0)
            v = np.concatenate([
                [min(c["hp"] or 0, _HP_SCALE) / _HP_SCALE],
                _one_hot(c["cardType"], N_CARDTYPE),
                _one_hot(c["energyType"], N_ENERGY),
                np.array([
                    float(bool(c["basic"])), float(bool(c["stage1"])),
                    float(bool(c["stage2"])), float(bool(c["ex"])),
                    (c["retreatCost"] or 0) / 4.0,
                    best_dmg / _DMG_SCALE,
                ], dtype=np.float32),
            ]).astype(np.float32)
        self._card_cache[card_id] = v
        return v

    def attack(self, attack_id: int | None) -> np.ndarray:
        if attack_id is None:
            return np.zeros(self.ATTACK_F, dtype=np.float32)
        if attack_id in self._atk_cache:
            return self._atk_cache[attack_id]
        a = self.attacks.get(attack_id)
        if a is None:
            v = np.zeros(self.ATTACK_F, dtype=np.float32)
        else:
            energies = a.get("energies") or []
            multi = np.zeros(N_ENERGY, dtype=np.float32)
            for e in energies:
                if 0 <= int(e) < N_ENERGY:
                    multi[int(e)] += 1.0
            v = np.concatenate([
                [a.get("damage", 0) / _DMG_SCALE, len(energies) / 4.0],
                multi,
            ]).astype(np.float32)
        self._atk_cache[attack_id] = v
        return v


# --------------------------------------------------------------------------- #
# Encoder                                                                       #
# --------------------------------------------------------------------------- #
class Encoder:
    """Turn an obs_dict into (state[F], options[n, G]).

    ``deck_ids`` fixes the hand/board card vocabulary (we always pilot one deck),
    so the hand multi-hot has a stable meaning across games.
    """

    def __init__(self, deck_ids: list[int]):
        self.tab = _CardTable()
        self.vocab = sorted(set(deck_ids))
        self.vidx = {cid: i for i, cid in enumerate(self.vocab)}
        self.V = len(self.vocab)
        self.state_dim = self._state_dim()
        self.option_dim = self._option_dim()

    # -- dimensions (compute once by encoding a zero-ish probe) --------------
    def _state_dim(self) -> int:
        # 8 global flags + 2*(7 per-player scalars) + 2*active(hp,dmg,ncards + card_feat)
        # + 2*status(5) + hand vocab V + select meta (selType+context)
        return (8 + 2 * 7 + 2 * (3 + self.tab.CARD_F) + 2 * 5
                + self.V + N_SELTYPE + N_CONTEXT)

    def _option_dim(self) -> int:
        return (N_OPTTYPE + N_AREA + N_AREA + 3
                + self.tab.CARD_F + self.tab.ATTACK_F)

    # -- state ---------------------------------------------------------------
    def encode_state(self, o) -> np.ndarray:
        st = o.current
        me = st.yourIndex
        opp = 1 - me
        pm, po = st.players[me], st.players[opp]

        glob = np.array([
            min(st.turn, 40) / 40.0,
            min(st.turnActionCount, 20) / 20.0,
            float(st.firstPlayer == me),
            float(st.supporterPlayed), float(st.stadiumPlayed),
            float(st.energyAttached), float(st.retreated),
            float(len(st.stadium) > 0),
        ], dtype=np.float32)

        def player_scalars(p):
            return np.array([
                len(p.prize) / 6.0,
                p.deckCount / 60.0,
                len(p.discard) / 60.0,
                p.handCount / 12.0,
                len(p.bench) / 5.0,
                float(len(p.active) > 0 and p.active[0] is not None),
                p.benchMax / 5.0,
            ], dtype=np.float32)

        def active_feat(p):
            if p.active and p.active[0] is not None:
                a = p.active[0]
                hp_frac = (a.hp / a.maxHp) if a.maxHp else 0.0
                dmg_frac = 1.0 - hp_frac
                ncards = (len(a.energyCards) + len(a.tools)) / 6.0
                return np.concatenate([[hp_frac, dmg_frac, ncards], self.tab.card(a.id)])
            return np.zeros(3 + self.tab.CARD_F, dtype=np.float32)

        def status(p):
            return np.array([float(p.poisoned), float(p.burned), float(p.asleep),
                             float(p.paralyzed), float(p.confused)], dtype=np.float32)

        hand = np.zeros(self.V, dtype=np.float32)
        if pm.hand is not None:
            for c in pm.hand:
                j = self.vidx.get(c.id)
                if j is not None:
                    hand[j] += 1.0
        hand /= 4.0  # at most 4 copies of a card

        sel = o.select
        sel_meta = np.concatenate([
            _one_hot(int(sel.type) if sel else None, N_SELTYPE),
            _one_hot(int(sel.context) if sel else None, N_CONTEXT),
        ])

        return np.concatenate([
            glob,
            player_scalars(pm), player_scalars(po),
            active_feat(pm), active_feat(po),
            status(pm), status(po),
            hand, sel_meta,
        ]).astype(np.float32)

    # -- options -------------------------------------------------------------
    def encode_options(self, o) -> np.ndarray:
        sel = o.select
        if sel is None or not sel.option:
            return np.zeros((0, self.option_dim), dtype=np.float32)
        st = o.current
        me = st.yourIndex
        rows = [self._option_row(op, st, me) for op in sel.option]
        return np.stack(rows).astype(np.float32)

    def _option_row(self, op, st, me) -> np.ndarray:
        card_id = _resolve_card_id(op, st, me)
        scalars = np.array([
            (op.index or 0) / 12.0,
            (op.count or 0) / 12.0,
            float(op.playerIndex == me) if op.playerIndex is not None else 0.5,
        ], dtype=np.float32)
        return np.concatenate([
            _one_hot(int(op.type), N_OPTTYPE),
            _one_hot(int(op.area) if op.area is not None else None, N_AREA),
            _one_hot(int(op.inPlayArea) if op.inPlayArea is not None else None, N_AREA),
            scalars,
            self.tab.card(card_id),
            self.tab.attack(op.attackId),
        ]).astype(np.float32)


def _resolve_card_id(op, st, me) -> int | None:
    """Best-effort: the engine ID of the card an option refers to, for features."""
    try:
        if op.type == OptionType.PLAY and op.index is not None:
            hand = st.players[me].hand
            if hand is not None and op.index < len(hand):
                return hand[op.index].id
        pi = op.playerIndex if op.playerIndex is not None else me
        p = st.players[pi]
        area, idx = op.area, op.index
        if area is None or idx is None:
            return None
        if area == AreaType.ACTIVE and p.active and p.active[0] is not None:
            return p.active[0].id
        if area == AreaType.BENCH and idx < len(p.bench):
            return p.bench[idx].id
        if area == AreaType.HAND and p.hand is not None and idx < len(p.hand):
            return p.hand[idx].id
        if area == AreaType.DISCARD and idx < len(p.discard):
            return p.discard[idx].id
    except Exception:
        return None
    return None


def encode(obs: dict, encoder: Encoder) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: obs_dict -> (state[F], options[n, G])."""
    o = to_observation_class(obs)
    return encoder.encode_state(o), encoder.encode_options(o)
