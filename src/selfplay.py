"""Self-play PPO — the training loop that makes the agent improve.

Each iteration:

1. **Collect rollouts.** The learner (current net) plays games against a *league*
   of opponents — `RandomPolicy`, `GreedyPolicy`, and frozen snapshots of past
   selves — alternating sides for fairness. Every decision the learner makes is
   recorded (`NetPolicy(record=True)`); the game's ±1 outcome is stamped onto all
   of that game's transitions as the reward (terminal-only, γ=1 Monte-Carlo).
2. **PPO update.** Advantage = reward − value baseline (normalized). Clipped
   surrogate policy loss + value regression + entropy bonus, a few epochs over
   minibatches of decisions.
3. **Evaluate & checkpoint.** Greedy net vs Random and vs Greedy (mirror) for the
   north-star win rate; save the net; periodically add a frozen snapshot to the
   league.

This trains *piloting skill* in the (mirror) Dragapult matchup. Diverse opponent
*decks* (episodes data, Phase 5) plug in by extending the opponent/deck lists.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from src.agents import GreedyPolicy, RandomPolicy
from src.agents.net_agent import NetPolicy
from src.encode import Encoder
from src.env import play_game, DRAW
from src.eval import evaluate
from src.model import PolicyValueNet, action_log_prob


@dataclass
class TrainConfig:
    iters: int = 40
    games_per_iter: int = 64
    ppo_epochs: int = 4
    minibatch: int = 256
    clip: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    snapshot_every: int = 5      # add a frozen self to the league this often
    eval_every: int = 2
    eval_games: int = 40
    seed: int = 0
    device: str = "cpu"


def _append_metrics(path: str, row: dict) -> None:
    """Append one eval row to a CSV, flushed immediately (survives a killed run)."""
    import os
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                         for v in row.values()) + "\n")
        f.flush()


def _reward(winner: int, aborted: bool, learner_index: int) -> float:
    if aborted or winner == DRAW:
        return 0.0
    return 1.0 if winner == learner_index else -1.0


def collect_rollouts(net, encoder, deck, opponents, games, device, rng):
    """Play `games` self-play games; return the learner's transitions with rewards.

    ``opponents`` entries are ``(policy_factory, opp_deck)``; ``opp_deck=None``
    means a mirror match (the opponent pilots our deck too).
    """
    batch: list[dict] = []
    wins = losses = draws = 0
    for g in range(games):
        learner = NetPolicy(net, encoder, record=True, device=device)
        factory, opp_deck = rng.choice(opponents)
        opp = factory()
        opp_deck = opp_deck or deck
        learner_p0 = (g % 2 == 0)
        if learner_p0:
            res = play_game(learner, opp, deck, opp_deck)
            li = 0
        else:
            res = play_game(opp, learner, opp_deck, deck)
            li = 1
        r = _reward(res.winner, res.aborted, li)
        for t in learner.transitions:
            t["reward"] = r
        batch.extend(learner.transitions)
        wins += r > 0
        losses += r < 0
        draws += r == 0
    return batch, (wins, losses, draws)


def ppo_update(net, optimizer, batch, cfg: TrainConfig, rng):
    """One PPO update (several epochs) over a batch of recorded decisions."""
    rewards = torch.tensor([t["reward"] for t in batch], dtype=torch.float32)
    values = torch.tensor([t["value"] for t in batch], dtype=torch.float32)
    adv = rewards - values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for i, t in enumerate(batch):
        t["adv"] = adv[i].item()
        t["ret"] = rewards[i].item()

    idxs = list(range(len(batch)))
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "clipfrac": 0.0, "n": 0}
    for _ in range(cfg.ppo_epochs):
        rng.shuffle(idxs)
        for start in range(0, len(idxs), cfg.minibatch):
            mb = idxs[start:start + cfg.minibatch]
            optimizer.zero_grad()
            pi_loss = v_loss = ent_term = clip_count = 0.0
            for j in mb:
                t = batch[j]
                zl, sl, v = net(t["state"], t["options"])
                lp, ent = action_log_prob(zl, sl, t["order"], t["min"], t["max"])
                ratio = torch.exp(lp - t["old_log_prob"])
                a = t["adv"]
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a
                pi_loss = pi_loss - torch.min(unclipped, clipped)
                v_loss = v_loss + (v - t["ret"]) ** 2
                ent_term = ent_term + ent
                clip_count += float((torch.abs(ratio - 1.0) > cfg.clip).item())
            m = len(mb)
            loss = (pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent_term) / m
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            stats["pi_loss"] += float(pi_loss.item())
            stats["v_loss"] += float(v_loss.item())
            stats["entropy"] += float(ent_term.item())
            stats["clipfrac"] += clip_count
            stats["n"] += m
    n = max(stats["n"], 1)
    return {k: v / n for k, v in stats.items() if k != "n"}


def train(deck, cfg: TrainConfig, log=print, ckpt_path: str | None = None,
          metrics_path: str | None = None, resume_from: str | None = None,
          opp_decks: dict[str, list[int]] | None = None):
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    encoder = Encoder(deck)
    net = PolicyValueNet(encoder.state_dim, encoder.option_dim).to(cfg.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    # Resume: restore weights, optimizer momentum, and the iteration counter so a
    # continued run picks up exactly where it left off (frozen league snapshots are
    # rebuilt over subsequent iters rather than persisted).
    start_iter = 0
    if resume_from:
        ck = torch.load(resume_from, map_location=cfg.device)
        if ck["state_dim"] != encoder.state_dim or ck["option_dim"] != encoder.option_dim:
            raise ValueError(
                f"checkpoint dims ({ck['state_dim']},{ck['option_dim']}) != "
                f"encoder ({encoder.state_dim},{encoder.option_dim}) — different deck/encoder?")
        net.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_iter = ck.get("iter", 0)
        log(f"Resumed from {resume_from} at iter {start_iter} "
            f"(training through iter {cfg.iters}).")
        if start_iter >= cfg.iters:
            log("Nothing to do: checkpoint iter >= --iters. Raise --iters to train further.")

    # League: static baselines + frozen snapshots of past selves (added over time).
    # Entries are (policy_factory, opp_deck); opp_deck=None -> mirror match.
    league = [(lambda: RandomPolicy(), None), (lambda: GreedyPolicy(), None)]
    for name, odeck in (opp_decks or {}).items():
        league.append((lambda: GreedyPolicy(), odeck))
        log(f"League: + Greedy piloting {name} ({len(odeck)} cards)")

    def snapshot():
        frozen = copy.deepcopy(net).eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        return (lambda: NetPolicy(frozen, encoder, greedy=False, device=cfg.device), None)

    history = []
    for it in range(start_iter + 1, cfg.iters + 1):
        net.train()
        batch, wld = collect_rollouts(net, encoder, deck, league,
                                      cfg.games_per_iter, cfg.device, rng)
        upd = ppo_update(net, optimizer, batch, cfg, rng)
        w, l, d = wld
        msg = (f"[iter {it:3d}] selfplay W{w} L{l} D{d} | "
               f"decisions={len(batch)} | pi={upd['pi_loss']:+.3f} "
               f"v={upd['v_loss']:.3f} ent={upd['entropy']:.2f} clip={upd['clipfrac']:.2f}")

        if it % cfg.eval_every == 0 or it == cfg.iters:
            net.eval()
            vs_rand = evaluate(lambda: NetPolicy(net, encoder, greedy=True), lambda: RandomPolicy(),
                               deck, deck, games=cfg.eval_games)
            vs_greedy = evaluate(lambda: NetPolicy(net, encoder, greedy=True), lambda: GreedyPolicy(),
                                 deck, deck, games=cfg.eval_games)
            row = {"iter": it, "vs_random": vs_rand.win_rate,
                   "vs_greedy": vs_greedy.win_rate}
            msg += (f"\n           EVAL  vs Random {vs_rand.win_rate:.1%}  "
                    f"vs Greedy {vs_greedy.win_rate:.1%}")
            if opp_decks:
                # Win rate vs Greedy piloting the meta decks, eval games split evenly.
                per = max(2, cfg.eval_games // len(opp_decks))
                w = g = 0
                for odeck in opp_decks.values():
                    r = evaluate(lambda: NetPolicy(net, encoder, greedy=True),
                                 lambda: GreedyPolicy(), deck, odeck, games=per)
                    w += r.wins
                    g += r.games
                row["vs_meta"] = w / max(g, 1)
                msg += f"  vs Meta {row['vs_meta']:.1%}"
            history.append(row)
            if metrics_path:
                _append_metrics(metrics_path, row)

        log(msg)

        if it % cfg.snapshot_every == 0:
            league.append(snapshot())
        if ckpt_path:
            torch.save({"model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "state_dim": encoder.state_dim,
                        "option_dim": encoder.option_dim,
                        "iter": it}, ckpt_path)
    return net, history
