"""Compare all Mojo PPO derivatives with PyTorch autograd (test-only dependency)."""

import sys

import numpy as np
import torch


def uniform(seed):
    x = seed & 0xFFFFFFFF
    x ^= x >> 16
    x = x * 0x7FEB352D & 0xFFFFFFFF
    x ^= x >> 15
    x = x * 0x846CA68B & 0xFFFFFFFF
    x ^= x >> 16
    return np.float32(np.float32(x >> 9) + np.float32(0.5)) / np.float32(8388608)


def permutation(index, n=2, seed=42):
    row, col = divmod(index, n)
    row = (row + int(uniform(col + seed) * np.float32(32))) % 32
    col = (col + int(uniform(row + seed + 1) * np.float32(n))) % n
    return row * n + col


def check(path):
    data = np.fromfile(path, dtype=np.float32)
    n, samples, batch, count = 2, 64, 16, 9029
    start = n * (14 + 72 + 4)
    params = torch.tensor(data[start : start + count], requires_grad=True)
    actual = data[start + count : start + count * 2]
    obs_start = start + count * 4
    obs = data[obs_start : obs_start + samples * 72].reshape(samples, 72)
    action_start = obs_start + samples * 72
    actions = data[action_start : action_start + samples * 2].reshape(samples, 2)
    value_start = action_start + samples * 2
    old_values, old_logp, _, _, advantages, targets = data[
        value_start : value_start + samples * 6
    ].reshape(6, samples)
    idx = [permutation(i) for i in range(batch)]
    x = torch.tensor(obs[idx])
    h1 = torch.tanh(x @ params[:4608].reshape(72, 64) + params[4608:4672])
    h2 = torch.tanh(h1 @ params[4672:8768].reshape(64, 64) + params[8768:8832])
    out = h2 @ params[8832:9024].reshape(64, 3) + params[9024:9027]
    dist = torch.distributions.Normal(out[:, :2], params[9027:].exp())
    ratio = (
        dist.log_prob(torch.tensor(actions[idx])).sum(-1) - torch.tensor(old_logp[idx])
    ).exp()
    adv = torch.tensor(advantages[idx])
    actor = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
    value = out[:, 2]
    old, target = torch.tensor(old_values[idx]), torch.tensor(targets[idx])
    clipped = old + (value - old).clamp(-0.2, 0.2)
    critic = (
        0.25
        * torch.maximum((value - target).square(), (clipped - target).square()).mean()
    )
    (actor + critic).backward()
    expected = params.grad.numpy()
    # The fast GPU matmul permits tensor-core precision (TF32). Check its
    # aggregate error as well as every individual derivative. CPU is IEEE FP32.
    gpu = "gpu" in str(path)
    np.testing.assert_allclose(
        actual, expected, atol=3e-4 if gpu else 2e-5, rtol=3e-3 if gpu else 2e-4
    )
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < (
        0.002 if gpu else 1e-5
    )
    print(
        f"{path}: all {count} derivatives match autograd; max error {abs(actual - expected).max():.3g}"
    )


for path in sys.argv[1:]:
    check(path)
