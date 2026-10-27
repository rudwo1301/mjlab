from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tocabi.rl import HojaeGomtangOnPolicyRunner
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import tocabi_flat_env_cfg, tocabi_rough_env_cfg
from .hojae_gomtang_env_cfgs import tocabi_hojae_gomtang_flat_env_cfg
from .rl_cfg import (
  tocabi_hojae_gomtang_ppo_runner_cfg,
  tocabi_ppo_runner_cfg,
  tocabi_tracking_ppo_runner_cfg,
)
from .tracking_env_cfgs import tocabi_flat_tracking_env_cfg

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

# V2 plus a stance-leg hip roll bound widened to 0.15 rad.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V3",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=20, linear_hip_roll=True, stance_hip_roll_bound=0.15
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True, startup_steps=20, linear_hip_roll=True, stance_hip_roll_bound=0.15
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

# V3 plus a lateral CoM sway reference over the stance foot (4 cm).
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V4",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

# V4 plus a squared ankle roll penalty, so the sway comes from the hips.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V5",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

# V3 plus the ankle roll penalty, without the CoM sway reward.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V6",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    ankle_roll_weight=200.0,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    ankle_roll_weight=200.0,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

# V5 plus a squared pelvis roll penalty.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V7",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

# Deployment-MJCF friction: ankle roll frictionloss randomized over 0-25 N·m.
_ANKLE_FRICTION = (0.0, 25.0)
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-Startup-AnkleFric",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=20, ankle_roll_friction_range=_ANKLE_FRICTION
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True, startup_steps=20, ankle_roll_friction_range=_ANKLE_FRICTION
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V7-AnkleFric",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
    ankle_roll_friction_range=_ANKLE_FRICTION,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
    ankle_roll_friction_range=_ANKLE_FRICTION,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Tocabi",
  env_cfg=tocabi_flat_tracking_env_cfg(),
  play_env_cfg=tocabi_flat_tracking_env_cfg(play=True),
  rl_cfg=tocabi_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Tocabi-Privileged",
  env_cfg=tocabi_flat_tracking_env_cfg(privileged_actor=True),
  play_env_cfg=tocabi_flat_tracking_env_cfg(play=True, privileged_actor=True),
  rl_cfg=tocabi_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
