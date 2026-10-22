"""Tests for the HojaeGomtang Lipschitz PPO extension."""

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.tasks.tocabi.rl import HojaeGomtangPPO


def _bare_ppo(**attrs) -> HojaeGomtangPPO:
  """HojaeGomtangPPO with only the Lipschitz attributes, for unit tests."""
  ppo = HojaeGomtangPPO.__new__(HojaeGomtangPPO)
  defaults = {
    "lipschitz_weight": 0.1,
    "lipschitz_target_ratio": 0.05,
    "lipschitz_adapt_factor": 1.05,
    "lipschitz_deadband": 0.7,
    "lipschitz_min_weight": 0.005,
    "lipschitz_max_weight": 0.01,
    "lipschitz_mode": "original",
    "lipschitz_max_samples": None,
  }
  for name, value in {**defaults, **attrs}.items():
    setattr(ppo, name, value)
  return ppo


def test_adapt_weight_clamps_and_moves_toward_target() -> None:
  # Initial 0.1 is clamped into [min, max] after the first adaptation.
  ppo = _bare_ppo()
  ppo._adapt_lipschitz_weight(raw_loss=1.0, surrogate_loss=1.0)
  assert ppo.lipschitz_weight == 0.01

  # Ratio far above target: weight shrinks by the adapt factor.
  ppo = _bare_ppo(lipschitz_weight=0.008)
  ppo._adapt_lipschitz_weight(raw_loss=100.0, surrogate_loss=1.0)
  assert abs(ppo.lipschitz_weight - 0.008 / 1.05) < 1e-12

  # Ratio far below target: weight grows.
  ppo = _bare_ppo(lipschitz_weight=0.008)
  ppo._adapt_lipschitz_weight(raw_loss=1e-3, surrogate_loss=1.0)
  assert abs(ppo.lipschitz_weight - 0.008 * 1.05) < 1e-12

  # Within the deadband: unchanged.
  ppo = _bare_ppo(lipschitz_weight=0.008)
  ppo._adapt_lipschitz_weight(raw_loss=0.05 / 0.008, surrogate_loss=1.0)
  assert ppo.lipschitz_weight == 0.008


def test_lipschitz_loss_matches_input_gradient() -> None:
  obs = TensorDict({"actor": torch.randn(16, 5)}, batch_size=[16])
  actor = MLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    3,
    hidden_dims=(8,),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  ppo = _bare_ppo()
  ppo.actor = actor
  loss = ppo._lipschitz_loss(obs)
  assert loss.requires_grad and loss.item() > 0.0

  x = obs["actor"].clone().requires_grad_(True)
  actor.obs_normalizer.eval()
  (grad,) = torch.autograd.grad(actor.mlp(actor.obs_normalizer(x)).sum(), x)
  torch.testing.assert_close(loss, (grad * grad).sum(dim=1).mean())


def test_mirror_loss_is_zero_for_identity_maps() -> None:
  obs = TensorDict({"actor": torch.randn(16, 5)}, batch_size=[16])
  actor = MLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    3,
    hidden_dims=(8,),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  ppo = _bare_ppo(device="cpu")
  ppo.actor = actor
  ppo.set_mirror(list(range(5)), [1.0] * 5, [0, 1, 2], [1.0] * 3, [2.5, 1.0, 1.0])
  mean = actor.mlp(actor.obs_normalizer(obs["actor"]))
  assert ppo._mirror_loss(obs, mean).item() == 0.0
  # Swapping two actions breaks symmetry for a generic policy.
  ppo.set_mirror(list(range(5)), [1.0] * 5, [1, 0, 2], [1.0] * 3, [1.0] * 3)
  assert ppo._mirror_loss(obs, mean).item() > 0.0


def test_lipschitz_modes_match_explicit_jacobian() -> None:
  torch.manual_seed(0)
  obs = TensorDict({"actor": torch.randn(64, 5)}, batch_size=[64])
  actor = MLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    3,
    hidden_dims=(8,),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  actor.obs_normalizer.eval()
  x = obs["actor"]

  def policy(v: torch.Tensor) -> torch.Tensor:
    return actor.mlp(actor.obs_normalizer(v.unsqueeze(0)))[0]

  jac = torch.stack(
    [torch.autograd.functional.jacobian(policy, xi) for xi in x]
  )  # (B, actions, obs)

  def loss(mode: str, **attrs) -> torch.Tensor:
    ppo = _bare_ppo(lipschitz_mode=mode, **attrs)
    ppo.actor = actor
    return ppo._lipschitz_loss(obs)

  torch.testing.assert_close(loss("jacobian_l2"), (jac**2).sum((1, 2)).mean())
  torch.testing.assert_close(loss("jacobian_l1"), jac.abs().sum((1, 2)).mean())
  # Hutchinson is an unbiased estimate of the squared Frobenius norm.
  estimate = torch.stack([loss("hutchinson") for _ in range(2000)]).mean()
  torch.testing.assert_close(estimate, loss("jacobian_l2"), rtol=0.05, atol=0.0)
