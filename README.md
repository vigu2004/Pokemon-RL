# RL-Pokémon — PTCG AI Battle Challenge (Strategy track)

Building an AI agent for the [Pokémon TCG AI Battle Challenge — Strategy track](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)
($300k+ prize pool, deadline ~Aug 16 2026). The agent plays Standard-format
Pokémon TCG on the official simulator and is rated via automated ladder matches.

## Decisions locked
- **Deck:** Dragapult ex (`deck/dragapult_ex.draft.csv`) — chosen for flexible
  spread/snipe (`Phantom Dive` + Dusknoir `Cursed Blast`).
- **Approach:** heavy RL/ML — self-play PPO with a learned evaluator + inference-time
  determinized search. See `docs/rl_design.md`.

## What the submission actually is
`submission.tar.gz` containing:
- `main.py` — the agent: `agent(obs_dict) -> list[int]` (returns chosen legal option
  indices; returns 60 card IDs at deck-selection). Must **never crash** and must
  respect a per-move **time limit**.
- `deck.csv` — 60 card IDs.
- `cg/` — the official game package (from Kaggle, see below).

## Status
- [x] Card data in `data/EN_Card_Data.csv` (1267 cards) + engine dumps
      (`data/engine_cards.json`, `data/engine_attacks.json`).
- [x] `src/cardlib.py` — loads/collapses the CSV into structured `Card` objects.
- [x] Legal Dragapult deck (`battle_start` accepts it).
- [x] **Official engine RUNS LOCALLY** (`engine/`, native `libcg.dylib`).
- [x] **Full learning stack built:** engine driver (`src/env.py`), baseline agents
      (`src/agents/`), win-rate eval (`src/eval.py`), feature encoder (`src/encode.py`),
      policy-value net (`src/model.py`), **self-play PPO** (`src/selfplay.py`).
- [~] Self-play RL training runs (`scripts/train.py`) — tuning in progress.
- [ ] First submission (`main.py` + packer); inference-time determinized search.

See `docs/design_log.md` for the running journal and `plan.md` for the roadmap.

## Engine interface (verified, from `engine/cg/api.py`)
- `agent(obs_dict) -> list[int]`: if `obs.select is None` → return 60 card IDs (deck);
  else return option indices, length in `[select.minCount, select.maxCount]`, no dups,
  each `< len(select.option)`.
- `obs.current` = full `State`: both players' active/bench/discard/prizes/deck counts,
  your hand visible, opponent's hand hidden (counts only). `obs.logs` = events since
  last decision. `result != -1` means game over (winner index).
- Forward model for search: `search_begin(obs, your_deck, your_prize, opp_deck,
  opp_prize, opp_hand, opp_active)` → simulate with sampled hidden info. `search_step`,
  `search_end`, `search_release`. This is the engine-native determinized search.
- `all_card_data()` / `all_attack()` dump typed card/attack data (costs as EnergyType
  enums, ex/tera/aceSpec flags, evolvesFrom, attackIds).

## ⚠️ Key finding: the card pool is CURATED, not real-world Standard
The ~1267-card pool **excludes** many real-world staples — there is **no**
`Professor's Research`, `Iono`, `Arven`, `Nest Ball`, or `Pidgeot ex`. Draw/consistency
engines differ (e.g. `Judge`, `Cyrano`, `Hyper Aroma`, `Buddy-Buddy Poffin`). **Do not
copy netdecks** — build from what's in `data/EN_Card_Data.csv`.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.cardlib              # smoke-test the loader
```

## Getting the engine (do this next)
1. Accept competition rules in the browser (required, else API 403s).
2. Create a Kaggle API token → `~/.kaggle/kaggle.json` (chmod 600).
3. `bash scripts/download_data.sh` → unpacks the simulator into `engine/`.

## Training the agent

The agent is a policy-value neural net (`src/model.py`) trained by **self-play
PPO** (`src/selfplay.py`): each iteration it plays N games against a league of
{random, greedy, frozen past selves}, then learns from the ±1 win/loss outcomes.
The north-star is win rate vs the baselines, measured every few iterations.

### Quickstart
```bash
# Defaults: 200 iters × 64 games/iter, evaluate every 10 iters over 100 games.
# CPU-bound; ~20s/iter here, so a full run is ~1–1.5 h. Ctrl-C any time —
# the latest checkpoint and every eval row are already saved.
.venv/bin/python scripts/train.py

# Shorter run:
.venv/bin/python scripts/train.py --iters 40 --games 64

# Run in the background and watch progress live:
.venv/bin/python scripts/train.py > models/train.log 2>&1 &
tail -f models/train.log                       # streamed, line-buffered
tail -f models/dragapult_ex.draft.metrics.csv  # iter,vs_random,vs_greedy
```

### Outputs (in `models/`)
- `<deck>.pt` — checkpoint (model weights **+ optimizer state + iteration**),
  overwritten every iteration.
- `<deck>.metrics.csv` — one row per eval (`iter,vs_random,vs_greedy`), flushed
  immediately so it survives a kill.

### Resume from a checkpoint
`--iters` is the **total** target; a resumed run trains up to it. Restores weights,
optimizer momentum, and the iteration counter (league snapshots rebuild as it goes).
```bash
# Continue the default checkpoint and push the target to 300 iters:
.venv/bin/python scripts/train.py --resume models/dragapult_ex.draft.pt --iters 300
```

### Key flags
| Flag | Default | Meaning |
|---|---|---|
| `--iters` | `200` | total target iterations (resume trains up to this) |
| `--games` | `64` | self-play games per iteration (more = lower-variance gradient) |
| `--eval-every` / `--eval-games` | `10` / `100` | eval cadence and games per eval |
| `--lr`, `--ppo-epochs`, `--seed` | `3e-4`, `4`, `0` | PPO knobs |
| `--deck` | `dragapult_ex.draft` | deck name under `deck/` |
| `--out`, `--resume` | — | checkpoint path / resume source |

### Evaluate a trained checkpoint
```bash
.venv/bin/python scripts/eval_ckpt.py models/dragapult_ex.draft.pt --games 200
.venv/bin/python scripts/eval.py --a greedy --b random --games 200   # baselines only
```

## Layout
```
data/      card CSV + engine card/attack dumps
src/       cardlib, decklib, env (driver), encode, model, eval, selfplay; agents/
deck/      deck lists (authoring format: card_id,count)
docs/      rl_design.md, rl_approach.md, design_log.md (journal)
scripts/   train.py, eval.py, eval_ckpt.py, smoke_test.py, download_data.sh
models/    checkpoints + metrics CSVs (gitignored)
engine/    official simulator (gitignored; download separately)
```
