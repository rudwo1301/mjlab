from mjlab.envs import ManagerBasedRlEnvCfg
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

# Hip roll frictionloss randomized over 10-25 N·m on top of the ankle roll range: the
# real robot showed more hip roll friction than the deployment MJCF (9.9 N·m).
_HIP_FRICTION = (10.0, 25.0)
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-Startup-HipFric",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=20,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=20,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V7-HipFric",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=20,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
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
    hip_roll_friction_range=_HIP_FRICTION,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

# A visible startup: 0.8 s of double support in which the CoM moves 4 cm toward the
# first stance foot (the 0.4 s startup shifted it about 0.5 cm).
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-Startup-ShiftStartup",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=40,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=40,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V7-ShiftStartup",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=40,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=40,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

# ShiftStartup plus a penalty on the gait-cycle mean lateral velocity error: a 5 cm/s
# drift costs 0.3 of the lateral tracking weight (10).
_DRIFT_WEIGHT = 1200.0
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-Startup-NoDrift",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=40,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
    lateral_drift_weight=_DRIFT_WEIGHT,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=40,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
    lateral_drift_weight=_DRIFT_WEIGHT,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V7-NoDrift",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=40,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
    lateral_drift_weight=_DRIFT_WEIGHT,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=40,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
    lateral_drift_weight=_DRIFT_WEIGHT,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

# ShiftStartup plus a reward for each step landing 21 cm beside the stance foot (the
# mean touchdown width), against step-to-step placement scatter that adds up to a
# sideways drift. "Vanilla" is the original HojaeGomtang reward set (the -Startup
# tasks above), as opposed to the V7 rewards.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-Vanilla-StepWidth",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=40,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
    step_width=0.21,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=40,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
    step_width=0.21,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V7-StepWidth",
  env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    startup_steps=40,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
    step_width=0.21,
  ),
  play_env_cfg=tocabi_hojae_gomtang_flat_env_cfg(
    play=True,
    startup_steps=40,
    linear_hip_roll=True,
    stance_hip_roll_bound=0.15,
    com_sway_amplitude=0.04,
    ankle_roll_weight=200.0,
    base_roll_weight=1000.0,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
    step_width=0.21,
  ),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)


def _lateral_env_cfg(play: bool = False, v7: bool = False) -> ManagerBasedRlEnvCfg:
  """StepWidth task with a wider kernel under vy and forced pure lateral commands."""
  return tocabi_hojae_gomtang_flat_env_cfg(
    play=play,
    startup_steps=40,
    linear_hip_roll=v7,
    stance_hip_roll_bound=0.15 if v7 else None,
    com_sway_amplitude=0.04 if v7 else None,
    ankle_roll_weight=200.0 if v7 else 0.0,
    base_roll_weight=1000.0 if v7 else 0.0,
    ankle_roll_friction_range=_ANKLE_FRICTION,
    hip_roll_friction_range=_HIP_FRICTION,
    startup_com_shift=0.04,
    step_width=0.21,
    step_width_sigma_per_vy=0.8,
    rel_lateral_envs=0.15,
  )


register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-Vanilla-Lateral",
  env_cfg=_lateral_env_cfg(),
  play_env_cfg=_lateral_env_cfg(play=True),
  rl_cfg=tocabi_hojae_gomtang_ppo_runner_cfg(),
  runner_cls=HojaeGomtangOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Tocabi-HojaeGomtang-V7-Lateral",
  env_cfg=_lateral_env_cfg(v7=True),
  play_env_cfg=_lateral_env_cfg(play=True, v7=True),
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
