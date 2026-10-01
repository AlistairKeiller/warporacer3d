"""Actor-critic MLP (torch) with a state-independent Gaussian policy."""

import torch
from torch import nn
from torch.distributions import Normal

LOGSTD_MIN, LOGSTD_MAX = -1.6, -0.3


def _layer_init(layer, std=2.0**0.5):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp(obs_dim, out_dim, head_std, hidden):
    return nn.Sequential(
        _layer_init(nn.Linear(obs_dim, hidden)),
        nn.Tanh(),
        _layer_init(nn.Linear(hidden, hidden)),
        nn.Tanh(),
        _layer_init(nn.Linear(hidden, out_dim), std=head_std),
    )


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.actor = _mlp(obs_dim, act_dim, 0.01, hidden)
        self.critic = _mlp(obs_dim, 1, 1.0, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        # The bounded scale is positive; validation would synchronize CUDA here.
        return Normal(
            self.actor(obs),
            self.log_std.clamp(LOGSTD_MIN, LOGSTD_MAX).exp(),
            validate_args=False,
        )

    def value(self, obs):
        return self.critic(obs).squeeze(-1)
