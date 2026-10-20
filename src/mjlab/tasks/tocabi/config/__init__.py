from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import tocabi_flat_env_cfg, tocabi_rough_env_cfg
from .rl_cfg import tocabi_ppo_runner_cfg

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
