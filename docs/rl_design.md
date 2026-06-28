# RL design — Dragapult ex agent (Strategy track)

Goal: a self-play-trained agent for a **fixed Dragapult ex deck** that maximizes
win rate on Kaggle's rating ladder, under a per-move time limit, and **never crashes**.

## Why not naive end-to-end PPO
PTCG is imperfect-information, high-variance, with a huge/variable discrete action
space (the engine hands us a *list of legal option indices* each decision). Training
a policy from scratch over 2000 cards via self-play is slow and brittle. We fix the
deck (Dragapult ex) to collapse the state/action distribution, then learn.

## Architecture (staged — each stage ships independently)

1. **Engine wrapper / Gym env.** Wrap the official simulator as `step(action_idx) ->
   obs, reward, done`. Reward = +1 win / -1 loss (optionally small shaping: prize
   taken = +0.16). Action space = "pick an index into the legal-options list".
   This is the foundation everything else needs.

2. **Heuristic policy (baseline + reference opponent).** Hand-coded turn logic:
   attach energy, evolve via Rare Candy, set up Dusknoir for `Cursed Blast` snipe,
   `Phantom Dive` to spread + KO, gust with Boss's Orders, manage prize race. Must
   always return a legal fallback. This is our self-play seed and our sanity bar.

3. **Observation encoding.** Featurize obs_dict -> fixed-length tensor:
   - own/opp active + bench (species id embedding, HP, energy attached, damage,
     status, tool), hand contents (multi-hot over deck's ~25 distinct cards),
     prizes remaining (both sides), deck/discard counts, turn number, who's on the
     play. Hidden info (opp hand/deck) stays hidden — that's the point.

4. **Policy/value net + masked action head.** Since the action set is a dynamic
   list, score each legal option (pointer/embedding over option features) and softmax
   *only over legal indices* (illegal-action masking). Shared trunk -> value head for
   PPO advantage and for use as a search-time evaluator.

5. **Self-play PPO.** League/self-play vs a pool {heuristic, past checkpoints} to
   avoid overfitting one opponent. Track win rate vs heuristic as the north-star.

6. **Search at inference (the real edge).** Plug the value net into **time-budgeted
   determinized search**: sample plausible hidden info, do shallow expectimax/MCTS
   rollouts using the simulator as forward model, act on the averaged value. Even
   1–2 ply beats reactive play and is where "forward thinking" scores. Hard-cap on
   wall-clock per move; fall back to raw policy / heuristic if time runs out.

## Risks / guardrails
- **Time limit + crashes auto-lose.** Wrap `agent()` in try/except -> legal fallback;
  budget search with a deadline.
- **Variance.** Evaluate over many games (hundreds) before trusting a win-rate delta.
- **Engine fidelity.** Train against the *official* simulator only; no homemade rules.

## Build order
Stage 1 (env) -> Stage 2 (heuristic, first valid submission) -> 3+4 (encode + net) ->
5 (self-play) -> 6 (search). Ship a submission at the end of Stage 2 to de-risk the pipeline.
