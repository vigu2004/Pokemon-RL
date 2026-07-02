# Design log — PTCG AI Battle Challenge (Strategy track)

A running journal of what we built, why, and what we learned. Newest entry on
top. This doubles as raw material for the ≤2000-word strategy report (10% of the
score).

> **What is "the agent"?** The competition contract is a single function
> `agent(obs_dict) -> list[int]`: each decision, the engine hands us a list of
> legal options and we return the indices we pick. *Anything* implementing that
> is an agent. Behind the interface we run a layered stack that grows over time:
>
> 1. **Rule-based baselines** — `RandomPolicy`, `GreedyPolicy`. Not learned.
> 2. **Learned policy-value net** — *this is the neural network*. A policy head
>    scores the legal options; a value head estimates win probability. Trained by
>    self-play PPO. This is what "training the agent" refers to.
> 3. **Determinized search** at inference — uses the value net as an evaluator.
>
> The submitted agent = net (+ search) wrapped in `try/except → GreedyPolicy`, so
> it always returns a legal move and never crashes.

---

## 2026-07-02 — First run evaluated; ladder meta mined; meta league; FIRST SUBMISSION

**First full training run: it learns.** The 200-iter mirror-league run finished.
Checkpoint evaluated over 300 games/side (`eval_ckpt.py`): **vs Random 96.0%
[93.1, 97.7], vs Greedy 80.5% [75.6, 84.6]**, zero aborts, avg 22 turns. The
metrics trend is a clean learning curve (45%→95% vs Random, 20%→83% vs Greedy).
Gotcha found: the engine import `os.chdir`s into `engine/`, so `eval_ckpt.py`
with a *relative* checkpoint path resolves against `engine/` and dies — pass
absolute paths.

**Episodes = full replays, and we can read them.** Kaggle auth works with the
new-style `~/.kaggle/access_token`. Daily datasets
`kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD` (~750MB zip, ~21.5GB raw,
5–6k episodes/day, manifest in `...-episodes-index`). Each episode JSON has BOTH
players' 60-card decks (step[1] actions), every decision's full obs (same schema
as our engine wrapper: `current`/`select`/`logs`) + chosen indices, and final
rewards. So: meta mining now, behavior cloning later, all offline.

**Mined the 2026-07-01 day** (5,266 episodes, 125 teams) with the new
`scripts/mine_episodes.py` (streams the zip, no extract). The live meta:
Marnie's Grimmsnarl ex is the deck to beat (1,827 games, 55.8%; its +Fezandipiti
variant 58.4% — best large-sample list), then Alakazam/Dudunsparce (1,648 @
42.1% — popular but weak), Archaludon ex (1,630 @ 48.0%), Fezandipiti ex (1,351
@ 44.9%), Cornerstone Ogerpon ex (679 @ **56.1%**), Mega Kangaskhan/Clefairy
(542 @ 54.1%), Mega Starmie ex (527), Mega Lucario ex (400 @ 54.2%). Sobering:
**Dragapult ex barely exists on the ladder (86 games) and wins 43.0%** — our
deck list needs tuning even if the concept stays.

**Meta league built and training resumed.** `scripts/make_meta_decks.py` exports
the top-8 archetypes' best exact lists (real lists top teams piloted) to
`deck/meta/*.csv`; all 8 verified engine-legal. `selfplay.py` league entries are
now `(policy, opp_deck)` pairs and `train.py --opp-decks` adds Greedy piloting
each meta deck; eval gained a **vs_meta** column. Resumed iter 200→400 with the
8-deck league. First reads: vs Meta ~56–60% while mirror metrics hold (98%/81%)
— cross-deck skill is now the thing being trained.

**FIRST SUBMISSION shipped** to the Simulation ladder (2026-07-02, pending
validation): `scripts/pack_submission.py` stages main.py + deck.csv + stripped
model.pt + cg/ + src/ + data/, then validates in a clean subprocess (imports
only the staged bundle, plays a full main-vs-main game through the real engine —
any illegal selection raises), then tars (3.2MB). The agent
(`src/submit/main.py`) is a fallback chain net→greedy→random with a contract
sanitizer — it cannot crash and cannot return an illegal selection.
`src/env.py` now supports both layouts (repo `engine/`, submission root `cg/`).

**Facts pinned down:** Strategy track deadline **Sep 13 2026** ($240k, judged,
no leaderboard); Simulation ladder ends Aug 16 2026 (~4k teams, top Elo ~1253).
Competition data files were refreshed 2026-07-01; our card CSV is byte-identical,
engine unchanged.

**Next**
- Read the 400-iter run: does vs_meta climb without mirror regressing?
- Watch the first submission validate; then ship the meta-league checkpoint.
- Behavior-clone from top teams' episodes (decks AND decisions are in the data)
  — both as a warm-start comparison and as stronger league opponents.
- Deck-list tuning against the mined meta (Dragapult's 43% ladder WR is a flag).
- Determinized search at inference (Phase 4) — still the biggest untapped edge.

---

## 2026-07-01 — Learning substrate: env, baselines, eval, encoder

Goal for the session: *"start training the agent to play."* Training needs a
substrate that didn't exist yet (only `cardlib.py` was in `src/`). Built it
bottom-up. Nothing here is a neural net yet — this is the scaffolding a net
plugs into.

**Built**
- `src/decklib.py` — expand authoring-format decks (`card_id,count`) into the
  flat 60-ID list the engine and submission want.
- `src/env.py` — the engine **driver**. The native `cg` lib is a process-global
  singleton (one battle at a time), so instead of a Gym `Env` object we expose
  `play_game(policy0, policy1, deck0, deck1)`, which alternates control between
  two policies based on `obs.current.yourIndex`. A **policy** is any
  `policy(obs_dict) -> list[int]` — exactly the Kaggle contract, so policies drop
  straight into a submission. Includes legal-move helpers and a `safe=True` mode
  that replaces an illegal/throwing move with a random legal one (a training
  rollout never crashes).
- `src/agents/baselines.py` — `RandomPolicy` (floor) and `GreedyPolicy` (a
  loop-safe develop-then-attack pilot). These are the self-play seed, the
  reference opponent, and the submission's safety fallback.
- `src/eval.py` + `scripts/eval.py` — win-rate harness. Swaps sides every other
  game (so first-player/coin advantage can't bias the number) and reports a 95%
  Wilson interval.
- `src/encode.py` — featurizer: `obs_dict → state[174]` + `options[n, 86]`.
  Perspective-correct (me = player to move); encodes only *public* info about the
  opponent (counts, visible board), never hidden hand/deck. Static card/attack
  features come from the engine's own dumps.

**Verified**
- Engine still runs on this Mac (smoke test: full self-play game to completion).
- Selection protocol with two *different* decks: `battle_start` takes both decks
  up front, so driving directly there is **no** `select=None` deck step (that
  only happens in the Kaggle `agent()` harness). `result` is `-1` until terminal,
  then `0`/`1`/`2` (p0 / p1 / draw).
- `GreedyPolicy` vs `RandomPolicy`, **mirror match** (both pilot Dragapult):
  **61.7% [49.0, 72.9]** over 60 games — the heuristic has real skill.
- `GreedyPolicy` (Dragapult) vs `RandomPolicy` (engine sample deck): **38.3%** —
  *lost*. This is deck asymmetry, not policy weakness: our draft list is weaker
  than the sample deck under unskilled pilots. Deck tuning is a later phase; the
  takeaway is to **measure policy skill in the mirror**, and treat cross-deck
  results as a matchup signal.
- Encoder is dimension-stable across a full game (state 174, option 86, option
  counts 1–30).

**Decisions / notes**
- Action space is a *variable-length set selection* (`minCount..maxCount`,
  no dups). The net will handle it with a per-option scoring head + a STOP token
  (Plackett-Luce-style sequential selection), so any `(min,max)` is covered.
- Torch installed into `.venv`.

**Then built** — `src/model.py` (policy-value net; STOP-token action head, log-probs
verified exact), `src/selfplay.py` + `scripts/train.py` (PPO self-play loop). Loop
runs end-to-end: rollouts → PPO update → eval → checkpoint.

**Training ops**
- **Logging gotcha:** redirecting stdout to a file block-buffers Python output, so a
  killed run loses everything. Fixed: `scripts/train.py` forces line-buffering, and
  each eval flushes a row to `models/<deck>.metrics.csv` — progress survives a kill.
- **Resume:** checkpoints now save optimizer state + iteration; `--resume` continues
  a run. `--iters` is the *total* target.
- **Defaults:** 64 games/iter, 200 total iters, eval every 10 iters over 100 games
  (~20s/iter → ~1–1.5 h full run on this Mac). README has a Training section.

**Open question — is it actually learning?** Tiny/short runs are dominated by eval
variance and the net (greedy-argmax) can sit *below* Random early. Needs a full run
with the flushing metrics CSV to read the real win-rate trend. That's the next check.

**Next**
- Kick off a full run; read `metrics.csv` trend. If flat, suspect credit assignment
  (terminal-only reward over long games) — consider prize-diff reward shaping / GAE.
- First `main.py` + packer submission (ship `GreedyPolicy` first, learned net after).
