"""TOCABI flat velocity task with the HojaeGomtang leg policy MDP.

Observations, actions, commands, rewards and terminations follow the leg actor-critic
of the sim-to-real validated HojaeGomtang code (LearningHumanoidArmMotion, tocabi
branch, ``tocabi_arm_motion`` with the arm policy disabled). The robot, simulation
timing and domain randomization are shared with :func:`tocabi_flat_env_cfg`.

The mirror and Lipschitz losses live in the PPO update (``config/rl_cfg.py``,
``tasks/tocabi/rl``).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tocabi import mdp
from mjlab.tasks.tocabi.mdp import hojae_gomtang as hg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise

from .env_cfgs import (
  ACTOR_HISTORY_LENGTH,
  D_GAINS,
  LEG_JOINT_NAMES,
  LEGS,
  P_GAINS,
  TORQUE_LIMITS,
  UPPER_JOINT_NAMES,
  tocabi_flat_env_cfg,
)

# Absolute PD target ranges for actions in [-1, 1], per leg joint in
# LEG_JOINT_NAMES order (HipYaw, HipRoll, HipPitch, Knee, AnklePitch, AnkleRoll).
_LEG_RANGES = (
  (-0.3, 0.3),
  (-1.0, 1.0),
  (-1.0, 0.5),
  (-0.3, 1.5),
  (-0.8, 0.5),
  (-0.6, 0.6),
)
LEG_JOINT_POS_LIMITS = _LEG_RANGES * 2

FEET = SceneEntityCfg(
  "robot", body_names=("L_Foot_Link", "R_Foot_Link"), preserve_order=True
)
_LR = ("L", "R")
HIP_YAW = SceneEntityCfg(
  "robot", joint_names=tuple(f"{s}_HipYaw_Joint" for s in _LR), preserve_order=True
)
HIP_ROLL = SceneEntityCfg(
  "robot", joint_names=tuple(f"{s}_HipRoll_Joint" for s in _LR), preserve_order=True
)
HIP_YAW_ROLL = SceneEntityCfg(
  "robot",
  joint_names=HIP_YAW.joint_names + HIP_ROLL.joint_names,  # type: ignore[operator]
  preserve_order=True,
)
ANKLE_ROLL = SceneEntityCfg(
  "robot", joint_names=tuple(f"{s}_AnkleRoll_Joint" for s in _LR), preserve_order=True
)
KNEES = SceneEntityCfg(
  "robot", joint_names=tuple(f"{s}_Knee_Joint" for s in _LR), preserve_order=True
)

ILLEGAL_CONTACT_BODIES = (
  "Pelvis_Link",
  "L_HipRoll_Link",
  "L_Thigh_Link",
  "L_Knee_Link",
  "L_AnkleRoll_Link",
  "R_HipRoll_Link",
  "R_Thigh_Link",
  "R_Knee_Link",
  "R_AnkleRoll_Link",
)


def tocabi_hojae_gomtang_flat_env_cfg(
  play: bool = False,
  encoder_bias: bool = True,
  startup_steps: int = 0,
  linear_hip_roll: bool = False,
  stance_hip_roll_bound: float | None = None,
  com_sway_amplitude: float | None = None,
  ankle_roll_weight: float = 0.0,
  base_roll_weight: float = 0.0,
  ankle_roll_friction_range: tuple[float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the TOCABI flat velocity configuration with the HojaeGomtang MDP.

  Args:
    play: Build the evaluation variant.
    encoder_bias: Randomize a per-episode joint encoder bias. HojaeGomtang
      deployment training had this off.
    startup_steps: Policy steps of double support at the start of every episode,
      in which the weight moves onto the first stance foot before the gait
      clock starts (0 disables, as in the original).
    linear_hip_roll: Charge only the excess past the hip roll bounds instead of
      the original constant-plus-excess penalty.
    stance_hip_roll_bound: Widen the stance leg's inward hip roll bound to this
      magnitude [rad], following the gait phase (None keeps both bounds fixed).
    com_sway_amplitude: Add a reward for the CoM following a lateral sway of
      this amplitude [m] over the stance foot (None disables).
    ankle_roll_weight: Weight of a squared ankle roll angle penalty, so lateral
      weight shift comes from the hips rather than leaning on the ankles (0
      disables).
    base_roll_weight: Weight of a squared pelvis roll penalty, on top of the
      original tilt reward (0 disables).
    ankle_roll_friction_range: Separate frictionloss range [N·m] for the ankle
      roll joints, e.g. (0, 25) to cover the 22 N·m of the deployment MJCF. The
      other leg joints keep their range (None: one range for all leg joints).
  """
  cfg = tocabi_flat_env_cfg(play=play, encoder_bias=encoder_bias)

  ##
  # Sensors
  ##

  # Net contact forces over all contacts (ground and self), as in the original.
  feet_contact = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="body", pattern=r"^(L_Foot_Link|R_Foot_Link)$", entity="robot"
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  illegal_contact = ContactSensorCfg(
    name="illegal_contact",
    primary=ContactMatch(
      mode="body", pattern=f"^({'|'.join(ILLEGAL_CONTACT_BODIES)})$", entity="robot"
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=1,
  )
  foot_foot_contact = ContactSensorCfg(
    name="foot_foot_contact",
    primary=ContactMatch(mode="body", pattern="R_Foot_Link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="L_Foot_Link", entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=1,
  )
  cfg.scene.sensors = (feet_contact, illegal_contact, foot_foot_contact)

  ##
  # Observations
  ##

  hist = ACTOR_HISTORY_LENGTH
  phase = {"command_name": "step_phase"}
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=hg.base_ang_vel_lpf,
      params={"cutoff_hz": 25.0},
      noise=Gnoise(std=0.10),
      history_length=hist,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity, noise=Gnoise(std=0.05), history_length=hist
    ),
    "velocity_commands": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "twist"}
    ),
    "phase_sin": ObservationTermCfg(func=hg.phase_sin, params=phase),
    "phase_cos": ObservationTermCfg(func=hg.phase_cos, params=phase),
    "joint_pos": ObservationTermCfg(
      func=hg.joint_pos_abs,
      params={"asset_cfg": LEGS, "biased": True},
      noise=Gnoise(std=0.05),
      history_length=hist,
    ),
    "joint_vel": ObservationTermCfg(
      func=hg.joint_vel_lpf,
      params={"asset_cfg": LEGS, "cutoff_hz": 25.0},
      noise=Gnoise(std=1.0),
      history_length=hist,
    ),
    "last_leg_action": ObservationTermCfg(
      func=hg.last_raw_action,
      params={"action_name": "joint_pos"},
      noise=Gnoise(std=0.1),
    ),
  }
  # The critic adds privileged base state and drops noise, history and encoder bias.
  critic_terms = {
    "base_height": ObservationTermCfg(func=hg.base_height),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_heading": ObservationTermCfg(func=hg.base_heading),
    "base_ang_vel": ObservationTermCfg(
      func=hg.base_ang_vel_lpf, params={"cutoff_hz": 25.0}
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "velocity_commands": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "twist"}
    ),
    "phase_sin": ObservationTermCfg(func=hg.phase_sin, params=phase),
    "phase_cos": ObservationTermCfg(func=hg.phase_cos, params=phase),
    "joint_pos": ObservationTermCfg(func=hg.joint_pos_abs, params={"asset_cfg": LEGS}),
    "joint_vel": ObservationTermCfg(
      func=hg.joint_vel_lpf, params={"asset_cfg": LEGS, "cutoff_hz": 25.0}
    ),
    "last_leg_action": ObservationTermCfg(
      func=hg.last_raw_action, params={"action_name": "joint_pos"}
    ),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.LowerBodyActionCfg(
      entity_name="robot",
      lower_joint_names=LEG_JOINT_NAMES,
      upper_joint_names=UPPER_JOINT_NAMES,
      p_gains=P_GAINS,
      d_gains=D_GAINS,
      torque_limits=TORQUE_LIMITS,
      pd_control=True,
      joint_pos_limits=LEG_JOINT_POS_LIMITS,
      enable_torque_bias=not play,
      torque_bias_episode=(0.0, 3.0),
      torque_bias_tick=(0.0, 5.0),
    )
  }
  cfg.actions = actions

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": hg.HojaeGomtangVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(8.0, 12.0),
      heading_command=False,
      debug_vis=True,
      ranges=mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.6, 0.6)
      ),
    ),
    # Sampled only at reset: per-env step period and start phase.
    "step_phase": hg.StepPhaseCommandCfg(
      resampling_time_range=(1.0e9, 1.0e9),
      step_period_range=(35, 45),
      startup_steps=startup_steps,
    ),
  }
  cfg.commands = commands

  ##
  # Rewards (original sign convention: penalties are negative, weights positive)
  ##

  feet_sensor = {"sensor_name": "feet_contact"}
  cfg.rewards = {
    "survival": RewardTermCfg(func=hg.survival, weight=1.0),
    "action_smoothness1": RewardTermCfg(func=hg.action_smoothness1, weight=2e-3),
    "action_smoothness2": RewardTermCfg(func=hg.action_smoothness2, weight=2e-4),
    "joint_torque": RewardTermCfg(
      func=hg.joint_torque_penalty,
      weight=1e-4,
      params={"action_name": "joint_pos"},
    ),
    "joint_velocity": RewardTermCfg(
      func=hg.joint_velocity_penalty, weight=2e-3, params={"asset_cfg": LEGS}
    ),
    "foot_flat_contact": RewardTermCfg(
      func=hg.foot_flat_orientation_penalty, weight=100.0, params={"asset_cfg": FEET}
    ),
    "swing_foot_height": RewardTermCfg(
      func=hg.swing_foot_height_penalty,
      weight=500.0,
      params={
        **phase,
        "asset_cfg": FEET,
        "max_height": 0.18,
        "min_height": 0.14,
        "rise_end": 0.24,
        "hold_end": 0.25,
        "swing_end": 0.5,
      },
    ),
    "base_lin_vel_z": RewardTermCfg(func=hg.base_lin_vel_z_penalty, weight=0.1),
    "base_ang_vel_xy": RewardTermCfg(func=hg.base_ang_vel_xy_penalty, weight=5.0),
    "joint_pos_limits": RewardTermCfg(
      func=hg.joint_pos_limits_penalty,
      weight=10.0,
      params={"asset_cfg": LEGS, "margin": 0.10, "outside_weight": 1.0},
    ),
    "joint_torque_limits": RewardTermCfg(
      func=hg.joint_torque_limits_penalty,
      weight=1e-2,
      params={"action_name": "joint_pos"},
    ),
    "joint_regularization": RewardTermCfg(
      func=hg.joint_regularization, weight=1.0, params={"asset_cfg": HIP_YAW_ROLL}
    ),
    "knee_negative": RewardTermCfg(
      func=hg.knee_negative_penalty,
      weight=100.0,
      params={"asset_cfg": KNEES, "min_knee_pos": 0.6},
    ),
    "hip_yaw_negative": RewardTermCfg(
      func=hg.hip_yaw_negative_penalty, weight=10.0, params={"asset_cfg": HIP_YAW}
    ),
    "hip_yaw_sum_zero": RewardTermCfg(
      func=hg.hip_yaw_sum_zero_reward, weight=1.0, params={"asset_cfg": HIP_YAW}
    ),
    # Linear: 15 makes a 0.1 rad excess cost 1.5, about what the original charges
    # for touching the bound.
    "hip_roll_threshold": RewardTermCfg(
      func=hg.hip_roll_threshold_penalty,
      weight=15.0 if linear_hip_roll else 1.0,
      params={
        "asset_cfg": HIP_ROLL,
        "left_min": -0.06,
        "right_max": 0.06,
        "linear": linear_hip_roll,
        "stance_bound": stance_hip_roll_bound,
        "command_name": "step_phase",
      },
    ),
    "foot_yaw_frame_y_dist": RewardTermCfg(
      func=hg.foot_yaw_frame_y_distance_penalty,
      weight=10.0,
      params={"asset_cfg": FEET, "threshold": 0.18},
    ),
    "grf_limit": RewardTermCfg(
      func=hg.grf_limit_penalty,
      weight=50.0,
      params={**feet_sensor, "threshold": 1500.0},
    ),
    "foot_slip": RewardTermCfg(
      func=hg.foot_slip_penalty,
      weight=20.0,
      params={
        **feet_sensor,
        "asset_cfg": FEET,
        "contact_force_threshold": 1.0,
        "vel_scale": 0.5,
      },
    ),
    "base_z_orientation": RewardTermCfg(func=hg.base_z_orientation_reward, weight=11.5),
    "tracking_lin_vel_x": RewardTermCfg(
      func=hg.tracking_lin_vel_x_reward, weight=7.0, params={"command_name": "twist"}
    ),
    "tracking_lin_vel_y": RewardTermCfg(
      func=hg.tracking_lin_vel_y_reward, weight=10.0, params={"command_name": "twist"}
    ),
    "tracking_yaw_vel": RewardTermCfg(
      func=hg.tracking_yaw_vel_reward, weight=10.0, params={"command_name": "twist"}
    ),
    "contact_schedule": RewardTermCfg(
      func=hg.contact_schedule_reward, weight=30.0, params={**phase, **feet_sensor}
    ),
    "termination": RewardTermCfg(func=hg.termination_penalty, weight=20.0),
  }
  if com_sway_amplitude is not None:
    cfg.rewards["com_lateral_tracking"] = RewardTermCfg(
      func=hg.com_lateral_tracking_reward,
      weight=5.0,
      params={**phase, "asset_cfg": FEET, "amplitude": com_sway_amplitude},
    )
  if ankle_roll_weight > 0.0:
    cfg.rewards["ankle_roll"] = RewardTermCfg(
      func=hg.joint_pos_square_penalty,
      weight=ankle_roll_weight,
      params={"asset_cfg": ANKLE_ROLL},
    )
  if base_roll_weight > 0.0:
    cfg.rewards["base_roll"] = RewardTermCfg(
      func=hg.base_roll_penalty, weight=base_roll_weight
    )
  if startup_steps > 0:
    cfg.rewards["startup_weight_shift"] = RewardTermCfg(
      func=hg.startup_weight_shift_reward,
      weight=30.0,
      params={**phase, **feet_sensor},
    )

  ##
  # Terminations
  ##

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "illegal_contact", "force_threshold": 1.0},
    ),
    "base_termination": TerminationTermCfg(
      func=hg.base_termination,
      params={"max_lin_vel": 15.0, "max_ang_vel": 10.0, "max_tilting": 0.8},
    ),
    "hip_yaw_limits": TerminationTermCfg(
      func=hg.joint_pos_out_of_bounds,
      params={"bounds": (-0.5, 0.5), "asset_cfg": HIP_YAW},
    ),
    "foot_crossing": TerminationTermCfg(
      func=hg.foot_crossing_yaw_frame, params={"asset_cfg": FEET, "min_y_dist": 0.16}
    ),
    "foot_foot_collision": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "foot_foot_contact", "force_threshold": 1.0},
    ),
  }

  ##
  # Events: scheduled pushes on the pelvis (training only)
  ##

  if not play:
    cfg.events["scheduled_push"] = EventTermCfg(
      func=hg.scheduled_push,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("Pelvis_Link",)),
        "impulse_range": (-50.0, 50.0),
        "lateral_scale": 1.0,
        "torque_impulse_range": (-1.5, 1.5),
        "windows": ((0.2, 0.4), (0.6, 0.8)),
        "halfspan_steps": (0, 2),
        "curriculum_iterations": (3000, 10000),
        "steps_per_iteration": 24,
      },
    )

  if ankle_roll_friction_range is not None:
    leg_friction = cfg.events["leg_friction"]
    leg_friction.params["asset_cfg"] = SceneEntityCfg(
      "robot",
      joint_names=tuple(n for n in LEG_JOINT_NAMES if "AnkleRoll" not in n),
    )
    cfg.events["ankle_roll_friction"] = EventTermCfg(
      mode="reset",
      func=leg_friction.func,
      params={
        **leg_friction.params,
        "asset_cfg": ANKLE_ROLL,
        "ranges": ankle_roll_friction_range,
      },
    )

  cfg.curriculum = {}
  cfg.episode_length_s = int(1e9) if play else 20.0

  return cfg
