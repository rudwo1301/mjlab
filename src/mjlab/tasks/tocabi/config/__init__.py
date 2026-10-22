from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tocabi.rl import HojaeGomtangOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import tocabi_flat_env_cfg, tocabi_rough_env_cfg
from .hojae_gomtang_env_cfgs import tocabi_hojae_gomtang_flat_env_cfg
from .rl_cfg import tocabi_hojae_gomtang_ppo_runner_cfg, tocabi_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Tocabi",
  env_cfg=tocabi_rough_env_cfg(),
  play_env_cfg=tocabi_rough_env_cfg(play=True),
  rl_cfg=tocabi_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi",
  env_cfg=tocabi_flat_env_cfg(),
  play_env_cfg=tocabi_flat_env_cfg(play=True),
  rl_cfg=tocabi_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(play=True),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-Lipschitz",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(play=True),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(lipschitz=True),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

# Startup weight shift (0.4 s) and the linear hip roll penalty.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V2",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(startup_steps=20, linear_hip_roll=True),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True, startup_steps=20, linear_hip_roll=True
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)
