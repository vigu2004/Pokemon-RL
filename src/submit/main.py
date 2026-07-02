"""Kaggle submission agent — trained policy net with layered safe fallbacks.

Packed to the tarball root as main.py by scripts/pack_submission.py, next to
deck.csv, model.pt, cg/, src/ and data/. The contract (engine/main.py sample):

    agent(obs_dict) -> list[int]
      * obs.select is None  -> return the 60-card deck (IDs from deck.csv)
      * otherwise           -> option indices, len in [minCount, maxCount],
                               no duplicates, each < len(select.option)

A crash or contract violation is an auto-loss, so every layer is wrapped:
net policy -> greedy heuristic -> random legal -> static minCount slice. The
answer is always passed through a contract sanitizer before returning.
"""
import os
import sys


def _agent_dir() -> str:
    """Locate the bundle dir. kaggle-environments loads this file by exec()ing
    its source into a bare namespace: __file__ is NEVER defined (that killed
    submission #1). But it compiles the source with the real path as filename,
    so this function's code object carries it. Fallbacks: the server's fixed
    extract dir, then cwd. Pick the first candidate that has deck.csv."""
    cands = []
    try:
        cands.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    try:
        f = sys._getframe().f_code.co_filename
        if f and not f.startswith("<"):
            cands.append(os.path.dirname(os.path.abspath(f)))
    except Exception:
        pass
    cands += ["/kaggle_simulations/agent", os.getcwd()]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "deck.csv")):
            return c
    return next((c for c in cands if c), os.getcwd())


_HERE = _agent_dir()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_DECK: list[int] = []
_POLICIES = []  # ordered (name, policy) fallback chain


def _read_deck() -> list[int]:
    path = os.path.join(_HERE, "deck.csv")
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(x) for x in f.read().split("\n") if x.strip()][:60]


def _build() -> None:
    global _DECK
    _DECK = _read_deck()
    try:
        import torch
        from src.agents.net_agent import NetPolicy
        from src.encode import Encoder
        from src.model import PolicyValueNet

        ck = torch.load(os.path.join(_HERE, "model.pt"), map_location="cpu")
        net = PolicyValueNet(ck["state_dim"], ck["option_dim"])
        net.load_state_dict(ck["model"])
        net.eval()
        _POLICIES.append(("net", NetPolicy(net, Encoder(_DECK), greedy=True)))
    except Exception as e:  # torch missing / bad ckpt -> heuristic still plays
        print(f"[agent] net unavailable, falling back: {e!r}", file=sys.stderr)
    try:
        from src.agents import GreedyPolicy, RandomPolicy

        _POLICIES.append(("greedy", GreedyPolicy()))
        _POLICIES.append(("random", RandomPolicy()))
    except Exception as e:
        print(f"[agent] baselines unavailable: {e!r}", file=sys.stderr)


try:
    _build()
except Exception as e:
    print(f"[agent] init failed: {e!r}", file=sys.stderr)
    if not _DECK:
        try:
            _DECK = _read_deck()
        except Exception:
            _DECK = []


def _sanitize(ans, sel: dict) -> list[int]:
    """Force an answer into the engine contract; never raises."""
    n = len(sel.get("option") or [])
    lo = int(sel.get("minCount") or 0)
    hi = int(sel.get("maxCount") or 0)
    lo, hi = max(0, min(lo, n)), max(0, min(hi, n))
    out, seen = [], set()
    for x in ans if isinstance(ans, (list, tuple)) else []:
        try:
            i = int(x)
        except Exception:
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
        if len(out) >= hi:
            break
    for i in range(n):  # pad up to minCount with unused legal indices
        if len(out) >= lo:
            break
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def agent(obs_dict: dict) -> list[int]:
    try:
        if obs_dict.get("select") is None:
            return list(_DECK)
        sel = obs_dict["select"]
        for _name, pol in _POLICIES:
            try:
                return _sanitize(pol(obs_dict), sel)
            except Exception:
                continue
        return _sanitize([], sel)  # static minCount slice
    except Exception:
        try:
            return _sanitize([], obs_dict.get("select") or {})
        except Exception:
            return []
