"""PPO with the HojaeGomtang mirror loss and adaptive Lipschitz regularization.

Mirror loss: with left/right maps set by :meth:`HojaeGomtangPPO.set_mirror` (see
``mirror.py``), it penalizes ``mean(w * (mu(mirror(obs)) - mirror(mu(obs)))^2)``
with per-action weights ``w``, pushing the policy toward left/right symmetry.

Lipschitz loss: it penalizes the sensitivity of the actor's mean action ``mu`` to its
raw (unnormalized) observation, with the observation normalizer frozen. ``mode``
selects the sensitivity measure, averaged over the batch (J = d mu / d obs):

- ``"original"``: ``||J^T 1||^2``, as in the original. The gradient of the summed
  actions, so per-action gradients can cancel without reducing ``J``.
- ``"hutchinson"``: ``||J^T v||^2`` with fresh ``v ~ N(0, I)``, an unbiased estimate
  of ``||J||_F^2`` at the same cost.
- ``"jacobian_l2"``: ``||J||_F^2``, exact (one backward pass per action).
- ``"jacobian_l1"``: ``sum |J_ij|``, exact (one backward pass per action).

``max_samples`` optionally evaluates the term on a random subset of each mini-batch
to bound the cost of the Jacobian modes. The weight adapts once per iteration so that
``weight * lipschitz_loss`` stays near ``target_ratio * |surrogate_loss|``, within
``[min_weight, max_weight]``. This follows the HojaeGomtang rsl_rl fork (PPO and
ModularOnPolicyRunner).

``update`` is the rsl-rl 5.5.1 ``PPO.update`` with these terms added; keep it in
sync if rsl-rl is upgraded.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

_LIPSCHITZ_MODES = ("original", "hutchinson", "jacobian_l2", "jacobian_l1")


class HojaeGomtangPPO(PPO):
  def __init__(
    self,
    *args: Any,
    lipschitz_cfg: dict | None = None,
    mirror_cfg: dict | None = None,
    **kwargs: Any,
  ):
    super().__init__(*args, **kwargs)
    mirror = mirror_cfg or {}
    self.mirror_enabled: bool = mirror.get("enabled", False)
    self.mirror_weight: float = mirror.get("weight", 0.1)
    self.mirror_cfg = mirror
    self._mirror_obs_idx: torch.Tensor | None = None
    self._mirror_obs_sign = torch.ones(0)
    self._mirror_act_idx = torch.zeros(0, dtype=torch.long)
    self._mirror_act_sign = torch.ones(0)
    self._mirror_act_weight = torch.ones(0)
    cfg = lipschitz_cfg or {}
    self.lipschitz_enabled: bool = cfg.get("enabled", False)
    self.lipschitz_weight: float = cfg.get("initial_weight", 0.1)
    self.lipschitz_adaptive: bool = cfg.get("adaptive", True)
    self.lipschitz_target_ratio: float = cfg.get("target_ratio", 0.05)
    self.lipschitz_adapt_factor: float = cfg.get("adapt_factor", 1.05)
    self.lipschitz_deadband: float = cfg.get("deadband", 0.7)
    self.lipschitz_min_weight: float = cfg.get("min_weight", 0.005)
    self.lipschitz_max_weight: float = cfg.get("max_weight", 0.01)
    self.lipschitz_mode: str = cfg.get("mode", "original")
    self.lipschitz_max_samples: int | None = cfg.get("max_samples", None)
    if self.lipschitz_mode not in _LIPSCHITZ_MODES:
      raise ValueError(
        f"Unknown Lipschitz mode '{self.lipschitz_mode}', expected one of "
        f"{_LIPSCHITZ_MODES}."
      )

  def set_mirror(
    self,
    obs_idx: list[int],
    obs_sign: list[float],
    act_idx: list[int],
    act_sign: list[float],
    act_weight: list[float],
  ) -> None:
    """Set the mirror maps for the concatenated actor observation and the action."""
    dev = self.device
    self._mirror_obs_idx = torch.tensor(obs_idx, dtype=torch.long, device=dev)
    self._mirror_obs_sign = torch.tensor(obs_sign, device=dev)
    self._mirror_act_idx = torch.tensor(act_idx, dtype=torch.long, device=dev)
    self._mirror_act_sign = torch.tensor(act_sign, device=dev)
    self._mirror_act_weight = torch.tensor(act_weight, device=dev)

  def _mirror_loss(self, observations, mean_action: torch.Tensor) -> torch.Tensor:
    assert self._mirror_obs_idx is not None, "Call set_mirror() first."
    actor = self.actor
    obs = torch.cat([observations[g] for g in actor.obs_groups], dim=-1)
    obs_m = obs[:, self._mirror_obs_idx] * self._mirror_obs_sign
    mu_of_mirror = actor.mlp(actor.obs_normalizer(obs_m))
    mirror_of_mu = mean_action[:, self._mirror_act_idx] * self._mirror_act_sign
    diff = mu_of_mirror - mirror_of_mu
    return torch.mean(torch.square(diff) * self._mirror_act_weight)

  def _lipschitz_loss(self, observations) -> torch.Tensor:
    actor = self.actor
    obs = torch.cat([observations[g] for g in actor.obs_groups], dim=-1)
    n = self.lipschitz_max_samples
    if n is not None and obs.shape[0] > n:
      obs = obs[torch.randperm(obs.shape[0], device=obs.device)[:n]]
    obs = obs.detach().requires_grad_(True)
    normalizer = actor.obs_normalizer
    was_training = normalizer.training
    normalizer.eval()  # Don't update normalizer statistics for this term.
    try:
      mean_action = actor.mlp(normalizer(obs))
    finally:
      normalizer.train(was_training)
    mode = self.lipschitz_mode
    if mode in ("original", "hutchinson"):
      v = (
        torch.ones_like(mean_action)
        if mode == "original"
        else torch.randn_like(mean_action)
      )
      (grad,) = torch.autograd.grad(mean_action, obs, v, create_graph=True)
      return torch.mean(torch.sum(grad * grad, dim=1))
    # Full Jacobian (num_actions, batch, obs_dim) via batched vector-Jacobian products.
    num_actions = mean_action.shape[1]
    basis = torch.eye(num_actions, device=obs.device).unsqueeze(1)
    basis = basis.expand(num_actions, *mean_action.shape)
    (jac,) = torch.autograd.grad(
      mean_action, obs, basis, create_graph=True, is_grads_batched=True
    )
    per_sample = (jac * jac if mode == "jacobian_l2" else jac.abs()).sum(dim=(0, 2))
    return torch.mean(per_sample)

  def _adapt_lipschitz_weight(self, raw_loss: float, surrogate_loss: float) -> None:
    ratio = self.lipschitz_weight * raw_loss / (abs(surrogate_loss) + 1e-8)
    lo = self.lipschitz_target_ratio * max(0.0, 1.0 - self.lipschitz_deadband)
    hi = self.lipschitz_target_ratio * (1.0 + self.lipschitz_deadband)
    factor = max(self.lipschitz_adapt_factor, 1.0001)
    weight = self.lipschitz_weight
    if ratio > hi:
      weight /= factor
    elif ratio < lo and raw_loss > 0.0:
      weight *= factor
    self.lipschitz_weight = min(
      max(weight, self.lipschitz_min_weight), self.lipschitz_max_weight
    )

  def update(self) -> dict[str, float]:
    """Run optimization epochs over stored batches and return mean losses."""
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_rnd_loss = 0 if self.rnd else None
    mean_symmetry_loss = 0 if self.symmetry else None
    mean_lipschitz_loss = 0.0
    last_lipschitz_loss = 0.0
    mean_mirror_loss = 0.0

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    symmetry_loss = torch.zeros((), device=self.device)
    for batch in generator:
      observations = batch.observations
      assert observations is not None
      original_batch_size = observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        advantages = batch.advantages
        assert advantages is not None
        with torch.no_grad():
          batch.advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
          )

      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      with torch.amp.autocast(
        device_type=torch.device(self.device).type,
        enabled=self.use_mixed_precision,
        dtype=torch.bfloat16,
      ):
        self.actor(
          batch.observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[0],
          stochastic_output=True,
        )
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
        values = self.critic(
          batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
        )
        distribution_params = tuple(
          p[:original_batch_size] for p in self.actor.output_distribution_params
        )
        entropy = self.actor.output_entropy[:original_batch_size]

        if self.desired_kl is not None and self.schedule == "adaptive":
          with torch.inference_mode():
            kl = self.actor.get_kl_divergence(
              batch.old_distribution_params,  # type: ignore
              distribution_params,
            )
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
              torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
              kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
              if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
              elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
              lr_tensor = torch.tensor(self.learning_rate, device=self.device)
              torch.distributed.broadcast(lr_tensor, src=0)
              self.learning_rate = lr_tensor.item()
            for param_group in self.optimizer.param_groups:
              param_group["lr"] = self.learning_rate

        ratio = torch.exp(
          actions_log_prob - torch.squeeze(batch.old_actions_log_prob)  # type: ignore
        )
        surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
        surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
          value_clipped = batch.values + (values - batch.values).clamp(
            -self.clip_param, self.clip_param
          )
          value_losses = (values - batch.returns).pow(2)
          value_losses_clipped = (value_clipped - batch.returns).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = (batch.returns - values).pow(2).mean()

        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy.mean()
        )

        rnd_loss = (
          self.rnd.compute_loss(batch.observations[:original_batch_size])  # type: ignore
          if self.rnd
          else None
        )

        if self.symmetry:
          symmetry_loss = self.symmetry.compute_loss(
            self.actor, batch, original_batch_size
          )
          if self.symmetry.use_mirror_loss:
            loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

        if self.mirror_enabled:
          mirror_loss = self._mirror_loss(
            observations[:original_batch_size],
            self.actor.output_mean[:original_batch_size],
          )
          loss = loss + self.mirror_weight * mirror_loss
          mean_mirror_loss += mirror_loss.item()

        if self.lipschitz_enabled:
          lipschitz_loss = self._lipschitz_loss(observations[:original_batch_size])
          loss = loss + self.lipschitz_weight * lipschitz_loss
          last_lipschitz_loss = lipschitz_loss.item()
          mean_lipschitz_loss += last_lipschitz_loss

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        rnd_loss.backward()  # type: ignore

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()  # type: ignore
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    stored_obs = self.storage.observations
    assert stored_obs is not None
    obs = cast(TensorDict, stored_obs.flatten(0, 1))
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    if self.rnd:
      self.rnd.update_normalization(obs)

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    loss_dict = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    if self.mirror_enabled:
      loss_dict["mirror"] = mean_mirror_loss / num_updates
    if self.lipschitz_enabled:
      loss_dict["lipschitz"] = mean_lipschitz_loss / num_updates
      loss_dict["lipschitz_weight"] = self.lipschitz_weight
      # As in the original, adapt from the last mini-batch Lipschitz loss and the
      # iteration's mean surrogate loss.
      if self.lipschitz_adaptive:
        self._adapt_lipschitz_weight(last_lipschitz_loss, mean_surrogate_loss)

    self.storage.clear()

    return loss_dict
