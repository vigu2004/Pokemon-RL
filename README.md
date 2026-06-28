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
- [x] Card data in `data/EN_Card_Data.csv` (1267 cards) + `Card_ID List_EN.pdf`.
- [x] `src/cardlib.py` — loads/collapses the CSV into structured `Card` objects.
- [x] Draft Dragapult deck from **in-pool** cards (unvalidated).
- [x] **Official engine downloaded + RUNS LOCALLY** (`engine/`, native `libcg.dylib`).
      `.venv/bin/python scripts/smoke_test.py` plays a full random self-play game. ✅
- [x] `data/engine_cards.json` — structured card data dumped from the engine itself.
- [ ] Validate Dragapult deck legality via `battle_start`.
- [ ] Gym wrapper (`battle_start`/`battle_select` loop), heuristic baseline,
      first valid submission, then self-play RL + determinized search.

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

## Layout
```
data/      card CSV + ID reference PDF (the only data we have so far)
src/       cardlib.py (loader); env wrapper, agent, model go here
deck/      deck lists
docs/      rl_design.md
scripts/   download_data.sh; submission packer (TODO)
engine/    official simulator (gitignored; download separately)
```
