"""RL configuration for the TOCABI velocity task.

Hyperparameters follow the DYROS IsaacLab TOCABI PPO config.

Not ported (DYROS): the action bound loss (coef 10, range 1.1) and the Lipschitz
constraint (LCP) gradient penalty (coef 0.002) of DyrosPPO.
"""

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
