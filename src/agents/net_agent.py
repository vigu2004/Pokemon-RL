"""Net-backed policy: the learned agent, usable for both play and training.

`NetPolicy` wraps the `PolicyValueNet` + `Encoder` into the standard
``policy(obs_dict) -> list[int]`` interface, so it drops into `play_game`, the
eval harness, and (eventually) the submission unchanged.

With ``record=True`` it also stores, for every decision it made, exactly what PPO
needs later: the encoded state/options, the chosen selection order, and the
log-prob/value under the *current* net. The self-play loop assigns the terminal
reward to those transitions after the game ends.
"""
from __future__ import annotations

import torch

from cg.api import to_observation_class

from src.encode import Encoder
from src.model import PolicyValueNet, sample_action


class NetPolicy:
    def __init__(self, net: PolicyValueNet, encoder: Encoder, *,
                 greedy: bool = False, record: bool = False, device: str = "cpu"):
        self.net = net
        self.enc = encoder
        self.greedy = greedy
        self.record = record
        self.device = device
        self.transitions: list[dict] = []

    @torch.no_grad()
    def __call__(self, obs: dict) -> list[int]:
        o = to_observation_class(obs)
        if o.select is None:
            return []
        state = torch.as_tensor(self.enc.encode_state(o), device=self.device)
        options = torch.as_tensor(self.enc.encode_options(o), device=self.device)
        opt_logits, stop_logit, value = self.net(state, options)
        ad = sample_action(opt_logits, stop_logit,
                           o.select.minCount, o.select.maxCount, greedy=self.greedy)
        if self.record:
            self.transitions.append({
                "state": state.cpu(),
                "options": options.cpu(),
                "order": list(ad.order),
                "old_log_prob": float(ad.log_prob.item()),
                "value": float(value.item()),
                "min": o.select.minCount,
                "max": o.select.maxCount,
            })
        return list(ad.order)
