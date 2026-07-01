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
