"""PPO with an asymmetric actor-critic, following rsl_rl (as configured in rl_cfg.py)."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOCfg:
  hidden_dims: tuple = (512, 256, 128)
  init_std: float = 1.0
  num_steps_per_env: int = 24
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.01
  learning_rate: float = 1.0e-3
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  use_clipped_value_loss: bool = True


class EmpiricalNormalization(nn.Module):
  """Running mean/std observation normalizer (rsl_rl EmpiricalNormalization)."""

  def __init__(self, dim, eps=1e-2):
    super().__init__()
    self.eps = eps
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("std", torch.ones(dim))
    self.register_buffer("count", torch.tensor(0, dtype=torch.long))

  def forward(self, x):
    return (x - self.mean) / (self.std + self.eps)

  @torch.no_grad()
  def update(self, x):
    n = x.shape[0]
    self.count += n
    rate = n / self.count
    var_x, mean_x = torch.var_mean(x, dim=0, unbiased=False)
    delta = mean_x - self.mean
    self.mean += rate * delta
    self.var += rate * (var_x - self.var + delta * (mean_x - self.mean))
    self.std = torch.sqrt(self.var)


def mlp(inp, hidden, out):
  layers, d = [], inp
  for h in hidden:
    layers += [nn.Linear(d, h), nn.ELU()]
    d = h
  layers.append(nn.Linear(d, out))
  return nn.Sequential(*layers)


class ActorCritic(nn.Module):
  def __init__(self, num_obs, num_critic_obs, num_actions, cfg: PPOCfg):
    super().__init__()
    self.actor_norm = EmpiricalNormalization(num_obs)
    self.critic_norm = EmpiricalNormalization(num_critic_obs)
    self.actor = mlp(num_obs, cfg.hidden_dims, num_actions)
    self.critic = mlp(num_critic_obs, cfg.hidden_dims, 1)
    self.std = nn.Parameter(cfg.init_std * torch.ones(num_actions))  # std_type="scalar"

  def distribution(self, obs):
    mean = self.actor(self.actor_norm(obs))
    std = self.std.clamp(min=1e-3).expand_as(mean)
    return torch.distributions.Normal(mean, std)

  def act_inference(self, obs):
    return self.actor(self.actor_norm(obs))

  def value(self, critic_obs):
    return self.critic(self.critic_norm(critic_obs))


class OnnxPolicy(nn.Module):
  """Deterministic actor with the normalizer baked in: obs -> actions."""

  def __init__(self, ac: ActorCritic):
    super().__init__()
    self.norm = ac.actor_norm
    self.actor = ac.actor

  def forward(self, obs):
    return self.actor(self.norm(obs))


class RolloutStorage:
  def __init__(self, T, N, num_obs, num_critic_obs, num_actions):
    z = lambda *s: torch.zeros(T, N, *s)
    self.obs, self.critic_obs = z(num_obs), z(num_critic_obs)
    self.actions, self.rewards, self.dones = z(num_actions), z(1), z(1)
    self.values, self.log_probs = z(1), z(1)
    self.mu, self.sigma = z(num_actions), z(num_actions)
    self.returns, self.advantages = z(1), z(1)
    self.step = 0


class PPO:
  def __init__(self, ac: ActorCritic, cfg: PPOCfg, num_envs, num_obs, num_critic_obs, num_actions):
    self.ac, self.cfg = ac, cfg
    self.lr = cfg.learning_rate
    self.optimizer = torch.optim.Adam(ac.parameters(), lr=self.lr)
    self.storage = RolloutStorage(cfg.num_steps_per_env, num_envs, num_obs, num_critic_obs,
                                  num_actions)

  @torch.no_grad()
  def act(self, obs, critic_obs):
    self.ac.actor_norm.update(obs)
    self.ac.critic_norm.update(critic_obs)
    dist = self.ac.distribution(obs)
    actions = dist.sample()
    s, t = self.storage, self.storage.step
    s.obs[t], s.critic_obs[t], s.actions[t] = obs, critic_obs, actions
    s.values[t] = self.ac.value(critic_obs)
    s.log_probs[t] = dist.log_prob(actions).sum(-1, keepdim=True)
    s.mu[t], s.sigma[t] = dist.mean, dist.stddev
    return actions

  @torch.no_grad()
  def process_step(self, rewards, dones, time_outs):
    s, t = self.storage, self.storage.step
    # Bootstrap on time-outs (rsl_rl behaviour).
    rewards = rewards + self.cfg.gamma * s.values[t, :, 0] * time_outs
    s.rewards[t, :, 0], s.dones[t, :, 0] = rewards, dones
    s.step += 1

  @torch.no_grad()
  def compute_returns(self, last_critic_obs):
    s, c = self.storage, self.cfg
    last_value = self.ac.value(last_critic_obs)
    adv = 0
    for t in reversed(range(c.num_steps_per_env)):
      next_value = last_value if t == c.num_steps_per_env - 1 else s.values[t + 1]
      not_done = 1.0 - s.dones[t]
      delta = s.rewards[t] + not_done * c.gamma * next_value - s.values[t]
      adv = delta + not_done * c.gamma * c.lam * adv
      s.returns[t] = adv + s.values[t]
    s.advantages = s.returns - s.values
    s.advantages = (s.advantages - s.advantages.mean()) / (s.advantages.std() + 1e-8)

  def update(self):
    s, c = self.storage, self.cfg
    flat = lambda x: x.reshape(-1, x.shape[-1])
    data = [flat(x) for x in (s.obs, s.critic_obs, s.actions, s.values, s.advantages,
                              s.returns, s.log_probs, s.mu, s.sigma)]
    batch = data[0].shape[0]
    mb = batch // c.num_mini_batches
    stats = {"value_loss": 0.0, "surrogate_loss": 0.0, "entropy": 0.0, "kl": 0.0}
    n_updates = 0
    for _ in range(c.num_learning_epochs):
      perm = torch.randperm(batch)
      for i in range(c.num_mini_batches):
        idx = perm[i * mb:(i + 1) * mb]
        obs, cobs, act, old_v, adv, ret, old_logp, old_mu, old_sigma = (d[idx] for d in data)
        dist = self.ac.distribution(obs)
        logp = dist.log_prob(act).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()
        value = self.ac.value(cobs)

        with torch.no_grad():
          mu, sigma = dist.mean, dist.stddev
          kl = torch.sum(
            torch.log(sigma / old_sigma + 1e-5)
            + (old_sigma**2 + (old_mu - mu) ** 2) / (2.0 * sigma**2) - 0.5, dim=-1
          ).mean()
          if kl > c.desired_kl * 2.0:
            self.lr = max(1e-5, self.lr / 1.5)
          elif 0.0 < kl < c.desired_kl / 2.0:
            self.lr = min(1e-2, self.lr * 1.5)
          for g in self.optimizer.param_groups:
            g["lr"] = self.lr

        ratio = torch.exp(logp - old_logp)
        surrogate = -torch.min(ratio * adv,
                               torch.clamp(ratio, 1 - c.clip_param, 1 + c.clip_param) * adv).mean()
        if c.use_clipped_value_loss:
          v_clipped = old_v + (value - old_v).clamp(-c.clip_param, c.clip_param)
          value_loss = torch.max((value - ret) ** 2, (v_clipped - ret) ** 2).mean()
        else:
          value_loss = ((value - ret) ** 2).mean()

        loss = surrogate + c.value_loss_coef * value_loss - c.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
        self.optimizer.step()

        stats["value_loss"] += value_loss.item()
        stats["surrogate_loss"] += surrogate.item()
        stats["entropy"] += entropy.item()
        stats["kl"] += kl.item()
        n_updates += 1
    s.step = 0
    return {k: v / n_updates for k, v in stats.items()}
