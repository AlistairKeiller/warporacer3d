"""Readable PPO with device-resident rollouts, observation normalization, and GAE."""

from collections import deque

import torch

from warporacer.agent import LOGSTD_MAX, LOGSTD_MIN

GAMMA, GAE_LAMBDA = 0.99, 0.95
CLIP, MAX_GRAD_NORM, TARGET_KL = 0.2, 0.5, 0.02


class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.full((), 1e-4, device=device)

    def update(self, x):
        x = x.reshape(-1, *self.mean.shape)
        variance, mean = torch.var_mean(x, dim=0, unbiased=False)
        n = len(x)
        delta = mean - self.mean
        total = self.count + n
        self.var.copy_(
            (
                self.var * self.count
                + variance * n
                + delta.square() * self.count * n / total
            )
            / total
        )
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    def normalize(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)


def gae(rewards, dones, values, gamma=GAMMA, lam=GAE_LAMBDA):
    advantage = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        live = 1 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * live - values[t]
        last = delta + gamma * lam * live * last
        advantage[t] = last
    return advantage


class PPO:
    def __init__(self, env, agent, rollouts=24, epochs=5, minibatches=4, lr=3e-4):
        self.env, self.agent = env, agent
        self.T, self.N, self.epochs, self.minibatches = (
            rollouts,
            env.num_envs,
            epochs,
            minibatches,
        )
        self.batch_size = rollouts * env.num_envs
        if min(rollouts, epochs, minibatches) <= 0 or self.batch_size % minibatches:
            raise ValueError(
                "Positive rollout dimensions must divide evenly into minibatches"
            )
        self.lr, self.global_step, self.iteration = lr, 0, 0
        self.opt = torch.optim.Adam(
            agent.parameters(), lr=lr, eps=1e-5, fused=env.device.is_cuda
        )
        self.obs_rms = RunningMeanStd((env.obs_dim,), env.torch_device)
        self.return_rms = RunningMeanStd((), env.torch_device)
        self.returns = torch.zeros(self.N, device=env.torch_device)
        self.ep_return = torch.zeros_like(self.returns)
        self.ep_length = torch.zeros_like(self.returns)
        self.finished_returns, self.finished_lengths = (
            deque(maxlen=100),
            deque(maxlen=100),
        )
        shape, dev = (self.T, self.N), env.torch_device
        self.buffer = (
            torch.empty((*shape, env.obs_dim), device=dev),
            torch.empty((*shape, env.act_dim), device=dev),
            *(torch.empty(shape, device=dev) for _ in range(3)),
            torch.empty((self.T + 1, self.N), device=dev),
        )
        self.episode_stats = torch.empty((*shape, 2), device=dev)
        self.reset_env_stats()

    def reset_env_stats(self):
        self.obs_rms.update(self.env.obs)
        self.obs = self.obs_rms.normalize(self.env.obs)
        self.ep_return.zero_()
        self.ep_length.zero_()
        self.returns.zero_()

    @torch.no_grad()
    def rollout(self):
        obs, actions, logp, rewards, dones, values = self.buffer
        for t in range(self.T):
            obs[t] = self.obs
            distribution = self.agent.dist(self.obs)
            actions[t] = distribution.sample()
            logp[t] = distribution.log_prob(actions[t]).sum(-1)
            values[t] = self.agent.value(self.obs)
            raw, reward, done = self.env.step(actions[t])
            self.returns.mul_(GAMMA).add_(reward)
            self.return_rms.update(self.returns)
            rewards[t] = reward / torch.sqrt(self.return_rms.var + 1e-8)
            dones[t] = done.float()
            self.returns.mul_(1 - dones[t])
            self.ep_return.add_(reward)
            self.ep_length.add_(1)
            finished = done.bool()
            self.episode_stats[t, :, 0] = torch.where(
                finished, self.ep_return, torch.nan
            )
            self.episode_stats[t, :, 1] = self.ep_length
            self.ep_return.mul_(1 - dones[t])
            self.ep_length.mul_(1 - dones[t])
            self.obs_rms.update(raw)
            self.obs = self.obs_rms.normalize(raw)
            if self.env.viewer is not None and t % 10 == 0:
                self.env.viewer.render()
        values[-1] = self.agent.value(self.obs)
        # Transfer completed-episode statistics once, rather than blocking each step.
        stats = self.episode_stats.flatten(0, 1).cpu()
        completed = stats[torch.isfinite(stats[:, 0])]
        self.finished_returns.extend(completed[:, 0].tolist())
        self.finished_lengths.extend(completed[:, 1].tolist())
        advantages = gae(rewards, dones, values)
        returns = advantages + values[:-1]
        return (
            obs.flatten(0, 1),
            actions.flatten(0, 1),
            logp.flatten(),
            advantages.flatten(),
            returns.flatten(),
            values[:-1].flatten(),
        )

    def iterate(self):
        with self.env.scope():
            return self._iterate()

    def _iterate(self):
        obs, actions, old_logp, advantage, returns, old_values = self.rollout()
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-8
        )
        metrics = torch.zeros(5, device=obs.device)
        updates, stopped = 0, False
        size = self.batch_size // self.minibatches
        for _ in range(self.epochs):
            epoch_kl = torch.zeros((), device=obs.device)
            for ids in torch.randperm(self.batch_size, device=obs.device).split(size):
                distribution = self.agent.dist(obs[ids])
                logp = distribution.log_prob(actions[ids]).sum(-1)
                logratio = logp - old_logp[ids]
                ratio = logratio.exp()
                policy = -torch.minimum(
                    ratio * advantage[ids],
                    ratio.clamp(1 - CLIP, 1 + CLIP) * advantage[ids],
                ).mean()
                value = self.agent.value(obs[ids])
                clipped = old_values[ids] + (value - old_values[ids]).clamp(-CLIP, CLIP)
                value_loss = (
                    0.5
                    * torch.maximum(
                        (value - returns[ids]).square(),
                        (clipped - returns[ids]).square(),
                    ).mean()
                )
                loss = policy + 0.5 * value_loss
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), MAX_GRAD_NORM)
                self.opt.step()
                with torch.no_grad():
                    self.agent.log_std.clamp_(LOGSTD_MIN, LOGSTD_MAX)
                    kl = ((ratio - 1) - logratio).mean()
                    fraction = ((ratio - 1).abs() > CLIP).float().mean()
                    metrics += torch.stack(
                        [
                            policy,
                            value_loss,
                            distribution.entropy().sum(-1).mean(),
                            kl,
                            fraction,
                        ]
                    )
                    epoch_kl.add_(kl)
                updates += 1
            if float(epoch_kl) / self.minibatches > 1.5 * TARGET_KL:
                stopped = True
                break
        policy, value, entropy, kl, fraction = (metrics / updates).tolist()
        if kl > 2 * TARGET_KL:
            self.lr = max(1e-6, self.lr / 1.5)
        elif kl < 0.5 * TARGET_KL:
            self.lr = min(3e-3, self.lr * 1.5)
        self.opt.param_groups[0]["lr"] = self.lr
        self.global_step += self.batch_size
        self.iteration += 1
        log = {
            "policy_loss": policy,
            "value_loss": value,
            "entropy": entropy,
            "approx_kl": kl,
            "clipfrac": fraction,
            "kl_stop": int(stopped),
            "lr": self.lr,
            "iteration": self.iteration,
        }
        if self.finished_returns:
            log["ep_return"] = sum(self.finished_returns) / len(self.finished_returns)
            log["ep_length"] = sum(self.finished_lengths) / len(self.finished_lengths)
        return log
