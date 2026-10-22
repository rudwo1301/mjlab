"""RL configuration for the TOCABI velocity task.

Hyperparameters follow the DYROS IsaacLab TOCABI PPO config.

Not ported (DYROS): the action bound loss (coef 10, range 1.1) and the Lipschitz
constraint (LCP) gradient penalty (coef 0.002) of DyrosPPO.
"""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def tocabi_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the TOCABI velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=5.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.001,
      num_learning_epochs=5,
      num_mini_batches=2,
      learning_rate=1.0e-5,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.004,
      max_grad_norm=1.0,
    ),
    experiment_name="tocabi_velocity",
    save_interval=50,
    num_steps_per_env=64,
    max_iterations=5000,
  )


@dataclass
class LipschitzCfg:
  """Adaptive Lipschitz regularization of the HojaeGomtang leg policy."""

  enabled: bool = True
  initial_weight: float = 0.1
  adaptive: bool = True
  target_ratio: float = 0.05
  """Target of weight * lipschitz_loss / |surrogate_loss|."""
  adapt_factor: float = 1.05
  deadband: float = 0.7
  """Relative band around ``target_ratio`` in which the weight is kept."""
  min_weight: float = 0.005
  max_weight: float = 0.01
  mode: Literal["original", "hutchinson", "jacobian_l2", "jacobian_l1"] = "original"
  """Sensitivity measure (see ``HojaeGomtangPPO``). "original" matches the source;
  "hutchinson" and the exact Jacobian modes penalize ``J`` itself."""
  max_samples: int | None = None
  """Evaluate the term on at most this many samples per mini-batch (None: all)."""


@dataclass
class MirrorCfg:
  """Left/right mirror loss of the HojaeGomtang leg policy (see ``mirror.py``)."""

  enabled: bool = True
  weight: float = 0.1
  action_weights: dict[str, float] = field(
    default_factory=lambda: {".*_Knee_Joint": 2.5, ".*_HipPitch_Joint": 2.5}
  )
  """Per-action loss weights by joint-name regex (default 1)."""
  joint_sign_flip: tuple[str, ...] = ()
  """Joints whose values change sign when mirrored. Empty as in the original; a
  physical sagittal mirror would use (".*_HipYaw_Joint", ".*_HipRoll_Joint",
  ".*_AnkleRoll_Joint")."""
  shift_phase: bool = False
  """Mirror the gait phase by half a cycle (negates phase sin/cos). Off as in the
  original."""


@dataclass
class HojaeGomtangPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO with the HojaeGomtang mirror and Lipschitz losses (``HojaeGomtangPPO``)."""

  class_name: str = "mjlab.tasks.tocabi.rl:HojaeGomtangPPO"
  mirror_cfg: MirrorCfg = field(default_factory=MirrorCfg)
  lipschitz_cfg: LipschitzCfg = field(default_factory=LipschitzCfg)


def tocabi_hojae_gomtang_ppo_runner_cfg(
  lipschitz: bool = False,
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the HojaeGomtang TOCABI task.

  Hyperparameters and the mirror loss follow the HojaeGomtang leg policy.
  ``lipschitz`` adds its adaptive Lipschitz loss (on by default in the original).
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 256),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=HojaeGomtangPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.12143,
      entropy_coef=0.001,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=4.1e-4,
      schedule="adaptive",
      gamma=0.9751,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      lipschitz_cfg=LipschitzCfg(enabled=lipschitz),
    ),
    experiment_name="tocabi_hojae_gomtang",
    run_name="lipschitz" if lipschitz else "",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
