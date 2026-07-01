"""The policy-value network — *the* neural network the agent learns.

Two heads on a shared trunk that reads the encoded state (`src/encode.py`):

* **policy** — scores each legal option (a per-option MLP conditioned on the
  state context) into a logit. The action is a *set* of option indices with size
  in ``[minCount, maxCount]``; we model it as sequential selection without
  replacement plus a STOP token (a Plackett–Luce distribution), which gives a
  proper log-prob and entropy for **any** ``(min, max)`` — including the common
  single-pick case (which reduces to a plain categorical) and multi-select
  discards/damage-placement.
* **value** — estimates win probability of the state as a scalar in ``[-1, 1]``
  (``tanh``), matching the ±1 game reward. Used as the PPO critic and, later, as
  the search-time evaluator.

The net processes **one decision at a time** (variable option count `n`), which
keeps the code simple and correct; PPO recomputes log-probs per stored decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# A large negative used to forbid choices (not -inf, to keep softmax finite).
_NEG = -1e9


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, option_dim: int, hidden: int = 256,
                 opt_embed: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.opt_embed = nn.Sequential(
            nn.Linear(option_dim, opt_embed), nn.ReLU(),
        )
        # Score an option from [option_embedding ; state_context].
        self.scorer = nn.Sequential(
            nn.Linear(opt_embed + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.stop_head = nn.Linear(hidden, 1)   # logit for the STOP token
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, options: torch.Tensor):
        """state[state_dim], options[n, option_dim] -> (opt_logits[n], stop_logit, value)."""
        h = self.trunk(state)                                  # [hidden]
        if options.shape[0] == 0:
            opt_logits = options.new_zeros(0)
        else:
            oe = self.opt_embed(options)                       # [n, opt_embed]
            hrep = h.unsqueeze(0).expand(oe.shape[0], -1)      # [n, hidden]
            opt_logits = self.scorer(torch.cat([oe, hrep], dim=-1)).squeeze(-1)  # [n]
        stop_logit = self.stop_head(h).squeeze(-1)             # scalar
        value = self.value_head(h).squeeze(-1)                 # scalar in [-1,1]
        return opt_logits, stop_logit, value


# --------------------------------------------------------------------------- #
# Action distribution: sequential set-selection with a STOP token              #
# --------------------------------------------------------------------------- #
@dataclass
class ActionDist:
    """Everything PPO needs about one decision's action."""
    order: list[int]        # chosen option indices, in sampling order (submit any order)
    log_prob: torch.Tensor  # scalar, differentiable
    entropy: torch.Tensor   # scalar, differentiable


def _step_logits(opt_logits, stop_logit, available, can_stop):
    """Assemble the categorical over {available options} ∪ {STOP?} for one sub-step.

    Returns (logits[k], index_map) where index_map[j] is the option index for row
    j, or -1 for the STOP row (present only when can_stop).
    """
    idxs = list(available)
    rows = [opt_logits[i] for i in idxs]
    index_map = list(idxs)
    if can_stop:
        rows.append(stop_logit)
        index_map.append(-1)
    return torch.stack(rows), index_map


def sample_action(opt_logits, stop_logit, min_c: int, max_c: int,
                  greedy: bool = False) -> ActionDist:
    """Sample (or argmax) a legal set selection and return its log-prob/entropy."""
    n = opt_logits.shape[0]
    max_c = min(max_c, n)
    min_c = min(min_c, max_c)
    available = list(range(n))
    order: list[int] = []
    log_prob = opt_logits.new_zeros(())
    entropy = opt_logits.new_zeros(())

    while True:
        if len(order) >= max_c or not available:
            break
        can_stop = len(order) >= min_c
        logits, index_map = _step_logits(opt_logits, stop_logit, available, can_stop)
        logp = F.log_softmax(logits, dim=0)
        if logits.numel() == 1:
            choice = 0                      # forced move: no decision, adds 0 entropy
        elif greedy:
            choice = int(torch.argmax(logits).item())
        else:
            choice = int(torch.multinomial(logp.exp(), 1).item())
        log_prob = log_prob + logp[choice]
        entropy = entropy - (logp.exp() * logp).sum()
        picked = index_map[choice]
        if picked == -1:                    # STOP
            break
        order.append(picked)
        available.remove(picked)
    return ActionDist(order=order, log_prob=log_prob, entropy=entropy)


def action_log_prob(opt_logits, stop_logit, order: list[int], min_c: int, max_c: int):
    """Recompute (log_prob, entropy) for a *fixed* ordered selection (PPO update)."""
    n = opt_logits.shape[0]
    max_c = min(max_c, n)
    min_c = min(min_c, max_c)
    available = list(range(n))
    log_prob = opt_logits.new_zeros(())
    entropy = opt_logits.new_zeros(())

    steps = list(order) + ([-1] if len(order) < max_c else [])  # trailing STOP if we stopped early
    for picked in steps:
        can_stop = (n - len(available)) >= min_c  # #picked-so-far >= minCount
        logits, index_map = _step_logits(opt_logits, stop_logit, available, can_stop)
        logp = F.log_softmax(logits, dim=0)
        row = index_map.index(picked)
        log_prob = log_prob + logp[row]
        entropy = entropy - (logp.exp() * logp).sum()
        if picked == -1:
            break
        available.remove(picked)
    return log_prob, entropy
