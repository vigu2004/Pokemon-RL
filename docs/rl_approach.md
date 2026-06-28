# RL approach — the concepts we're using (and why)

This doc is about the **RL ideas/methods**, not the file layout (that's `plan.md` /
`docs/rl_design.md`). Each choice is tied to a concrete property of *this* game.

---

## 0. The game's properties dictate the method

| Property of PTCG | RL consequence |
|---|---|
| Hidden hand/deck/prizes | It's a **POMDP** / two-player zero-sum **POSG**, not an MDP. Decisions condition on the *information set*, not full state. |
| Coin flips, shuffles, draws | **Stochastic transitions** → expectations, variance reduction matter. |
| Opponent is adaptive | Objective is **low exploitability (≈ Nash)**, not best-response to one fixed bot. Mixed strategies may be optimal. |
| Variable, typed legal-option list | **Structured/variable action space** → masking + per-option scoring, not a fixed softmax. |
| Win only at the end | **Sparse terminal reward**, long horizon → credit-assignment problem. |
| **We have the real simulator as a forward model** | We do **planning/search with the true model** and **don't need to learn dynamics** (no MuZero). Huge simplification. |
| Deck is fixed (Dragapult) | Collapsed state distribution → **specialize**, sample-efficient. Generalize over *opponents*. |

---

## 1. Problem formulation: zero-sum POSG, target ≈ Nash
We treat it as a **two-player zero-sum partially observable stochastic game**. The
goal is not "beat the heuristic" but a **low-exploitability policy** that holds up
against the whole field. That's why training is **self-play against a population**
(below), the game-theoretic way to approximate a Nash/best-response equilibrium
rather than overfitting one opponent.

## 2. Core optimizer: PPO actor–critic
On-policy **PPO** (clipped objective) with:
- **GAE** for advantage estimation (bias/variance trade-off over long episodes),
- **entropy regularization** for exploration *and* to keep the policy **stochastic**
  (mixed strategies are often correct under imperfect info),
- a **value (critic) head** estimating info-set win-probability — reused by search (§6).

*Considered & rejected:* tabular/Q-learning (state space far too large);
SAC (continuous-action); pure AlphaZero policy-iteration as the only signal (assumes
perfect info — see §6).

## 3. Self-play scheme: league / prioritized fictitious self-play (PFSP)
Naive self-play (always vs the current net) cycles and overfits — decks have
**non-transitive (rock-paper-scissors) dynamics**, so a single "best" policy can be
exploited. Instead, the **fictitious self-play family**:
- Maintain a **population/league**: current agent, **past checkpoints**, the
  **heuristic**, and a few **exploiters** (agents trained specifically to beat the
  main agent — they surface its blind spots).
- **PFSP opponent sampling**: weight opponents by how hard/instructive they are
  (more games vs agents we *barely* beat), à la AlphaStar.
- Conceptually this is **NFSP/PSRO**-style best-response-to-an-average, which is what
  pushes us toward low exploitability instead of a brittle local optimum.

*Considered & rejected:* **Deep CFR / MCCFR** — the principled imperfect-info answer
(poker), but heavy to implement over PTCG's rich sequential/typed action structure;
we get most of the benefit from population self-play + search. We keep CFR ideas in
reserve for §6.

## 4. Function approximation for a variable, typed action space
The net's **action head scores each legal option** rather than emitting a fixed
vector:
- Embed every `Option` (its `OptionType` + attributes: area/index/attackId/…),
  **attention/pointer** over the option set, one logit per option.
- **Invalid-action masking**: softmax over legal indices only (no wasted probability,
  no illegal picks).
- **Autoregressive selection** when `maxCount > 1` (pick option-by-option conditioned
  on prior picks) — same idea AlphaStar used for structured actions.
- Shared trunk encodes the **information set**: our board+hand, opponent's *visible*
  board + **counts** of hidden zones, prizes, turn flags, recent `logs`.

## 5. Sample efficiency: imitation warm-start, then RL
Cold-start self-play on a 30-turn sparse-reward game is slow. So:
- **Behavior cloning** (supervised) from the **heuristic** and from **ladder-replay
  episodes** to pretrain the policy/value to a sane starting point (offline).
- Then **fine-tune with PPO self-play**. (Optionally an **offline-RL** pass on replays,
  e.g. CQL, but BC-then-RL is the simpler first cut.)
This also seeds **opponent modeling** — replays tell us which archetypes to put in the
league (§3) and what hidden cards to expect (§6).

## 6. Decision-time planning: search with the true model (imperfect-info aware)
Because we hold the real simulator, inference does **planning**, guided by the
learned net — the AlphaZero recipe, adapted for hidden information:
- **Determinization / Information-Set MCTS (IS-MCTS, PIMC-style):** sample plausible
  hidden info (opponent deck/hand/prize) consistent with what we've seen, run the
  engine's `search_begin/step/end` forward, and **average** over samples. Policy net
  = action **priors**; value net = **leaf evaluation**; the simulator = exact dynamics.
- We explicitly accept PIMC's known weaknesses (**strategy fusion, non-locality**) as a
  pragmatic trade — and note the principled upgrade path: **belief-state RL+search
  (ReBeL)** / **Player of Games** (CFR-style counterfactual values inside the search),
  if determinized search proves exploitable.
- **Belief model:** start with a uniform prior over remaining legal cards (constraint
  propagation from public info); upgrade to a **learned opponent/belief model** from
  replays. This is what we feed the determinizer.
- Strictly **time-budgeted** (move time limit); degrade gracefully to net-only, then
  heuristic, on timeout — never crash.

*Why no MuZero:* MuZero learns a latent dynamics model because its games hide the
rules. We have exact rules via the engine, so we **plan with the real model** — no
learned-dynamics complexity.

## 7. Reward & credit assignment
- **Sparse terminal** reward: +1 win / −1 loss (true zero-sum signal).
- **Potential-based reward shaping** on prize differential / board advantage to ease
  credit assignment — potential-based so it's **policy-invariant** (doesn't change the
  optimum, just speeds learning).
- Discount γ close to 1 (short, decisive horizons).

## 8. Exploration & stability
- **Entropy bonus** (PPO) + **Dirichlet noise at the search root** (AlphaZero-style)
  to keep exploring and to avoid deterministic exploitable lines.
- **Opponent diversity** (§3) as the main regularizer against overfitting.
- **Exploitability / win-rate-vs-league** as eval metrics (not just vs one bot);
  evaluate over **hundreds of games** because variance is high.
- Optional **population-based training (PBT)** for hyperparameters later.

---

## TL;DR — the stack
**Zero-sum POSG → PPO actor-critic with GAE + entropy → masked pointer/attention
action head → BC warm-start from heuristic+replays → league / PFSP self-play toward
low exploitability → determinized IS-MCTS at inference using the engine's true
forward model, net-guided, time-budgeted → potential-based shaping for credit
assignment.** No learned dynamics (we own the simulator); CFR / ReBeL kept as the
upgrade path if PIMC-style search is too exploitable.
