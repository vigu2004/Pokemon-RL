# Plan — PTCG AI Battle Challenge (Strategy track)

Master plan for our agent. Living document: update the **Status** boxes as we go.
See also `docs/rl_design.md` (deep dive on the learning stack) and `README.md`
(setup + verified engine API).

---

## 1. The problem, precisely

Build an agent `agent(obs_dict) -> list[int]` that pilots a **single fixed
60-card deck** of Standard-format Pokémon TCG on the official simulator, and wins
automated ladder matches against *other competitors' decks*.

- **Imperfect information** — we see our hand + board; opponent's hand/deck are hidden.
- **Stochastic** — shuffles, draws, coin flips.
- **Large, dynamic action space** — each decision the engine hands us a *list of
  legal options*; we return indices into it (`len ∈ [minCount, maxCount]`, no dups).
- **Hard constraints** — agent must **never crash** (always return a legal fallback)
  and must obey a **per-move time limit**. Violations auto-lose.

### What we're optimizing
Strategy track is scored **70% model approach / 20% deck concept / 10% written
report (≤2000 words)** — *not* pure Elo. So: strong agent **and** a coherent,
well-documented deck concept + reasoning log. We keep a running design journal.

### Locked decisions
| Decision | Choice | Why |
|---|---|---|
| Deck | **Dragapult ex** (`deck/dragapult_ex.draft.csv`) | Flexible spread/snipe (Phantom Dive + Dusknoir Cursed Blast); robust across matchups |
| Approach | **Heavy RL/ML** | Organizers say rule-based alone won't rank; we want a learned policy + search |
| Generalization | Over **opponents**, not our deck | We pilot one deck always; train vs a *diverse opponent pool* |

---

## 2. Approaches we're taking (the strategy)

> **RL concepts/methods** (the theory: POSG→PPO, league/PFSP self-play, IS-MCTS
> with the true model, etc.) live in **`docs/rl_approach.md`**. This section is the
> high-level strategy; that doc is the *why* behind each RL choice.

We layer four ideas, each shipping independently so we always have a working agent:

### A. Fix the deck to make learning tractable
A 2000-card game is intractable to learn end-to-end. Committing to one deck
collapses *our* state/action distribution to a few dozen distinct cards and a
handful of recurring lines (set up Dreepy→Dragapult, Rare-Candy Dusknoir, snipe,
gust + KO). The agent specializes in piloting *this* deck extremely well.

### B. Strong heuristic core (baseline + self-play seed + safety net)
A hand-coded Dragapult pilot that plays correct turn flow and **never crashes**.
It is (1) our first valid submission, (2) the reference opponent we measure
against, (3) the seed policy for self-play, and (4) the time-out/illegal fallback
the learned agent wraps for safety.

### C. Self-play RL with a learned evaluator (the engine that improves)
PPO over a masked action head, trained by self-play against a **league** of past
checkpoints + the heuristic (not just mirror matches) for opponent robustness.
We learn a **value function** (board → win probability) as much as a policy —
the value net is reused inside search (D). Reward = +1 win / −1 loss, optional
small shaping for prizes taken.

### D. Determinized search at inference (the real edge — engine-native)
The engine exposes a forward model: `search_begin(obs, your_deck, your_prize,
opp_deck, opp_prize, opp_hand, opp_active)` → `search_step` → `search_end`. We
sample plausible hidden info (determinization), roll out a few plies guided by the
value net, average the outcomes, and act. Even 1–2 ply beats reactive play. Hard
wall-clock budget; fall back to policy/heuristic if time runs out. **This is what
"forward thinking under uncertainty" means here, and it's first-class in the engine.**

### Supporting: imitation / opponent modeling from ladder replays
Daily **episodes** datasets (~750MB/day) are real ladder game replays. Use them to
(1) learn what opponent archetypes we must beat (shapes the self-play league),
and (2) optionally warm-start/behavior-clone a policy before RL.

---

## 3. Architecture

```
                    ┌────────────────────────────────────────────┐
                    │           cg/ native engine (lib)           │
                    │  battle_start · battle_select · search_*     │
                    └───────────────┬───────────────┬─────────────┘
                          forward    │ play loop     │ forward model
                                     ▼               ▼
        ┌──────────────┐    ┌─────────────────┐   ┌──────────────────┐
        │ obs encoder  │───▶│ policy/value net │   │ determinized     │
        │ (State→tensor│    │  (masked action  │◀──│ search (MCTS/    │
        │  + opt feats)│    │   head + V)      │   │  expectimax)     │
        └──────────────┘    └────────┬─────────┘   └──────────────────┘
                                     │ self-play PPO
                                     ▼
                    ┌────────────────────────────────────────────┐
                    │ league: {heuristic, past checkpoints, meta} │
                    └────────────────────────────────────────────┘

  main.py (submission)  =  load net → search(budget) → indices, wrapped in
                           try/except → heuristic fallback → never crash
```

Components → files:
- `src/cardlib.py` — CSV loader (done). `data/engine_cards.json` — typed card data (done).
- `src/env.py` — Gym-style wrapper over `battle_start`/`battle_select`.
- `src/agents/heuristic.py` — rule-based Dragapult pilot.
- `src/encode.py` — `obs_dict` → fixed-length features (+ per-option features).
- `src/model.py` — policy/value net with illegal-action masking.
- `src/selfplay.py` — PPO + league self-play loop.
- `src/search.py` — determinized search wrapping `search_begin/step/end`.
- `src/submit/` (`main.py`, packer) — build `submission.tar.gz`.

---

## 4. Phased roadmap

### Phase 0 — Foundation ✅ DONE
- [x] Card data + engine downloaded; native lib runs on macOS.
- [x] Verified agent/obs/search API (`README.md`).
- [x] `src/cardlib.py`; engine card dump `data/engine_cards.json`.
- [x] Legal Dragapult deck (`battle_start` accepts it; 1 ACE SPEC).
- [x] `scripts/smoke_test.py` runs a full random self-play game.

### Phase 1 — Env + baselines + first submission
- [x] `src/env.py` — `play_game` driver, legal-action plumbing, +1/−1 reward, game-over detect.
- [x] `src/agents/baselines.py` — `RandomPolicy` + loop-safe `GreedyPolicy` (never crashes).
- [x] Eval harness (`src/eval.py`): win rate + Wilson CI, side-swapped. Greedy beats
      random **61.7%** in the mirror.
- [ ] `main.py` + packer → first **valid submission** on the leaderboard (ship Greedy first).

### Phase 2 — Learning substrate ✅ DONE
- [x] `src/encode.py` — obs → state[174] + per-option features[n,86] (perspective-correct).
- [x] `src/model.py` — policy/value net; variable-count masked action head (STOP-token
      Plackett–Luce); log-prob recompute verified exact.

### Phase 3 — Self-play RL  ◀ IN PROGRESS
- [x] `src/selfplay.py` — PPO; league of {random, greedy, frozen self-snapshots}.
- [~] North-star metric: win rate vs random & greedy, tracked across iters (first run running).
- [ ] Diverse opponent *decks* (currently mirror self-play only).

### Phase 4 — Search at inference
- [ ] `src/search.py` — determinization + value-guided rollouts via `search_*`, time-budgeted.
- [ ] Ablation: net-only vs net+search win rate; tune ply/budget under the move time limit.

### Phase 5 — Meta hardening + report
- [ ] Pull episodes data; mine opponent archetypes → expand self-play league.
- [ ] Tune deck list against the live meta (re-validate legality each change).
- [ ] Write the ≤2000-word strategy report from the design journal.

---

## 5. Risks & guardrails
- **Crash / timeout = auto-loss** → `main.py` wraps everything in try/except with a
  legal heuristic fallback; search is deadline-bounded.
- **Engine ≠ real-world TCG** (e.g. Maximum Belt is ACE SPEC here) → trust
  `data/engine_cards.json`, never netdecks. Re-validate deck legality on every edit.
- **High variance** → evaluate over hundreds of games before trusting a delta.
- **RL instability/cost** → heuristic baseline always shippable; learn a value
  function for search even if full policy RL underperforms.
- **API may change mid-comp** (enums note new elements may be appended) → keep the
  engine wrapper thin and defensive.

---

## 6. Current status snapshot
Phases 0–2 complete; Phase 3 (self-play PPO) in progress — the full learning stack
(`env → agents → eval → encode → model → selfplay`) is built and a first training run
is under way (`scripts/train.py`, checkpoints in `models/`). Next: read the first run's
win-rate trend, then (a) first Greedy submission via `main.py`+packer, (b) determinized
search at inference (Phase 4), (c) diverse opponent decks (Phase 5). See
`docs/design_log.md` for the running journal.
