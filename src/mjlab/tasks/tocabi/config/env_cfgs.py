"""DYROS TOCABI velocity environment configurations.

Task structure (observations, rewards, commands, terminations) follows the DYROS
IsaacLab TOCABI locomotion task. Simulation timing, observation history and domain
randomization follow the sim-to-real validated HojaeGomtang TOCABI setup.
"""

from dataclasses import replace

from mjlab.asset_zoo.robots import get_tocabi_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tocabi import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

##
# Joints and gains.
##

LEG_JOINT_NAMES = tuple(
  f"{side}_{joint}_Joint"
  for side in ("L", "R")
  for joint in ("HipYaw", "HipRoll", "HipPitch", "Knee", "AnklePitch", "AnkleRoll")
)
_ARM = ("Shoulder1", "Shoulder2", "Shoulder3", "Armlink", "Elbow", "Forearm")
UPPER_JOINT_NAMES = (
  ("Waist1_Joint", "Waist2_Joint", "Upperbody_Joint")
  + tuple(f"L_{j}_Joint" for j in (*_ARM, "Wrist1", "Wrist2"))
  + ("Neck_Joint", "Head_Joint")
  + tuple(f"R_{j}_Joint" for j in (*_ARM, "Wrist1", "Wrist2"))
)

# Real-robot PD gains and torque limits (divided by 9 and 3 in the action term).
_LEG_P = (2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0)
_LEG_D = (15.0, 50.0, 20.0, 25.0, 24.0, 24.0)
_LEG_TAU = (333.0, 232.0, 263.0, 289.0, 222.0, 166.0)
_ARM_P = (400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0)
_ARM_D = (10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0)
_ARM_TAU = (64.0, 64.0, 64.0, 64.0, 23.0, 23.0, 10.0, 10.0)
P_GAINS = _LEG_P * 2 + (6000.0, 10000.0, 10000.0) + _ARM_P + (100.0, 100.0) + _ARM_P
D_GAINS = _LEG_D * 2 + (200.0, 100.0, 100.0) + _ARM_D + (3.0, 3.0) + _ARM_D
TORQUE_LIMITS = (
  _LEG_TAU * 2 + (303.0, 303.0, 303.0) + _ARM_TAU + (10.0, 10.0) + _ARM_TAU
)

FEET = SceneEntityCfg(
  "robot", body_names=("L_AnkleRoll_Link", "R_AnkleRoll_Link"), preserve_order=True
)
LEGS = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)

TWIST_RANGES = mdp.UniformVelocityCommandCfg.Ranges(
  lin_vel_x=(-0.8, 0.8), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.8, 0.8)
)
"""Training velocity command ranges."""

ACTOR_HISTORY_LENGTH = 40
"""HojaeGomtang actor history: 40 consecutive steps, except clock, command and last
action."""


def tocabi_rough_env_cfg(
  play: bool = False,
  encoder_bias: bool = True,
  height_scan_in_actor: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create TOCABI rough terrain velocity configuration.

  Args:
    play: Build the evaluation variant.
    encoder_bias: Randomize a per-episode joint encoder bias. HojaeGomtang deployment
      training had this off (enable_obs_joint_pos_episode_bias=False).
    height_scan_in_actor: Give the actor the ground-truth height map. A policy
      trained with it needs a height map on the real robot too. If False, only the
      critic sees it and the actor walks blind on proprioception and history.
  """

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="Pelvis_Link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
  )
  feet_contact = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(L_AnkleRoll_Link|R_AnkleRoll_Link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=3,
  )
  pelvis_contact = ContactSensorCfg(
    name="pelvis_contact",
    primary=ContactMatch(mode="body", pattern="Pelvis_Link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=3,
  )

  ##
  # Observations
  ##

  hist = ACTOR_HISTORY_LENGTH
  # Not ported (HojaeGomtang): low-pass filters on observed base angular velocity
  # (60 Hz) and joint velocity (25 Hz). Add them if the deployment code filters
  # these too.
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel, noise=Unoise(n_min=-0.03, n_max=0.03), history_length=hist
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel, noise=Unoise(n_min=-0.03, n_max=0.03), history_length=hist
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.02, n_max=0.02),
      history_length=hist,
    ),
    "clock_input": ObservationTermCfg(
      func=mdp.clock_input, params={"command_name": "phase_time"}
    ),
    "velocity_commands": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "twist"}
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": LEGS, "biased": True},
      noise=Gnoise(std=0.00005),
      history_length=hist,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": LEGS},
      noise=Gnoise(std=0.005),
      history_length=hist,
    ),
    "actions": ObservationTermCfg(
      func=mdp.lower_body_action, params={"action_name": "joint_pos"}
    ),
    # Ground-truth height map, not camera data: 17 x 11 = 187 downward rays on a
    # 1.6 x 1.0 m yaw-aligned grid around the pelvis, giving pelvis height above
    # each hit point. Fed as a flat vector straight into the MLP, without history
    # (HojaeGomtang has no height scan, and 40 steps would add ~7.5k inputs). DYROS
    # fed 5 strided frames. Removed in the flat task.
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / terrain_scan.max_distance,
    ),
  }
  # The critic sees the same quantities without history, noise or encoder bias.
  critic_terms = {
    name: replace(term, history_length=0, noise=None)
    for name, term in actor_terms.items()
  }
  critic_terms["joint_pos"] = ObservationTermCfg(
    func=mdp.joint_pos_rel, params={"asset_cfg": LEGS}
  )
  if not height_scan_in_actor:
    del actor_terms["height_scan"]

  observations = {
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

  # Not ported: HojaeGomtang joint velocity limits (20-50 rad/s, enforced by the PhysX
  # solver; MuJoCo has no equivalent) and DYROS ERFI random force injection (off by
  # default in DYROS).
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.LowerBodyActionCfg(
      entity_name="robot",
      lower_joint_names=LEG_JOINT_NAMES,
      upper_joint_names=UPPER_JOINT_NAMES,
      p_gains=P_GAINS,
      d_gains=D_GAINS,
      torque_limits=TORQUE_LIMITS,
      pd_control=False,
      rand_motor_scale_range=(0.8, 1.2),
      enable_torque_bias=not play,
      torque_bias_episode=(0.0, 3.0),
      torque_bias_tick=(0.0, 5.0),
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": mdp.UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 5.0),
      rel_standing_envs=0.05,
      heading_command=False,
      debug_vis=True,
      ranges=replace(TWIST_RANGES),
    ),
    "phase_time": mdp.WalkingPhaseCommandCfg(
      resampling_time_range=(10.0, 10.0), phase_time_range=(1.2, 1.2)
    ),
  }

  ##
  # Events (HojaeGomtang deployment randomization)
  ##

  events = {
    # Ground friction * robot friction range, as in HojaeGomtang. Foot geoms have
    # contact priority, so their friction alone sets the foot-ground value.
    # (Python's 0.8 * (0.6, 1.0) would repeat the tuple, so scale each bound.)
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=(r"^[LR]_Foot_Link_collision\d+$",)
        ),
        "operation": "abs",
        "ranges": (0.8 * 0.6, 0.8 * 1.0),
      },
    ),
    "body_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
    "body_mass": EventTermCfg(
      mode="reset",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "operation": "scale",
        "ranges": (0.5, 1.5),
      },
    ),
    "joint_armature": EventTermCfg(
      mode="reset",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "scale",
        "ranges": (0.5, 1.5),
      },
    ),
    # HojaeGomtang's load-dependent PhysX friction has no MuJoCo equivalent; randomize a
    # constant leg friction instead.
    "leg_friction": EventTermCfg(
      mode="reset",
      func=dr.joint_friction,
      params={"asset_cfg": LEGS, "operation": "abs", "ranges": (0.0, 12.0)},
    ),
    # Base damping is 1.0, so this spans [0.1, 4.0] as in HojaeGomtang.
    "joint_damping": EventTermCfg(
      mode="reset",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "add",
        "ranges": (-0.9, 3.0),
      },
    ),
    # Removed below when encoder_bias=False.
    "encoder_bias": EventTermCfg(
      mode="reset",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.02, 0.02)},
    ),
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Not ported (HojaeGomtang): scheduled pushes, i.e. force/torque pulses (+-50 N,
    # +-1.5 N·m) in two windows per gait cycle, ramped in by a curriculum.
  }
  if not encoder_bias:
    del events["encoder_bias"]

  ##
  # Rewards
  ##

  phase = {"command_name": "phase_time"}
  rewards = {
    "lin_vel_xy_tracking": RewardTermCfg(
      func=mdp.track_lin_vel_xy_base_exp,
      weight=1.0,
      params={"command_name": "twist", "omega": 5.0},
    ),
    "ang_vel_z_tracking": RewardTermCfg(
      func=mdp.track_ang_vel_z_world_exp,
      weight=1.0,
      params={"command_name": "twist", "omega": 7.0},
    ),
    "orientation_tracking": RewardTermCfg(
      func=mdp.orientation_tracking, weight=1.0, params={"omega": 5.0}
    ),
    "base_height_tracking": RewardTermCfg(
      func=mdp.base_height_tracking,
      weight=0.5,
      params={"height": 0.928, "omega": 10.0},
    ),
    "periodic_force": RewardTermCfg(
      func=mdp.periodic_force,
      weight=1.0,
      params={**phase, "sensor_name": "feet_contact", "scale": 1300.0},
    ),
    "feet_height_tracking": RewardTermCfg(
      func=mdp.feet_height_tracking,
      weight=1.0,
      params={
        **phase,
        "omega": 5.0,
        "foot_height": 0.1,
        "kappa": 2.0,
        "offset": -0.77,
        "asset_cfg": FEET,
      },
    ),
    "feet_velocity_z_tracking": RewardTermCfg(
      func=mdp.feet_velocity_z_tracking,
      weight=0.5,
      params={
        **phase,
        "omega": 3.0,
        "foot_height": 0.1,
        "kappa": 2.0,
        "asset_cfg": FEET,
      },
    ),
    "large_contact": RewardTermCfg(
      func=mdp.large_contact_force,
      weight=-0.005,
      params={"sensor_name": "feet_contact", "threshold": 100.0},
    ),
    "default_joint": RewardTermCfg(
      func=mdp.default_joint_pos,
      weight=0.2,
      params={"omega": 2.0, "asset_cfg": LEGS},
    ),
    "feet_flat": RewardTermCfg(
      func=mdp.feet_flat, weight=-0.1, params={"asset_cfg": FEET}
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "base_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "pelvis_contact", "force_threshold": 1.0},
    ),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": 0.79}
    ),
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds, time_out=True
    ),
  }

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel, params={"command_name": "twist"}
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG, curriculum=True),
        max_init_terrain_level=5,
      ),
      entities={"robot": get_tocabi_robot_cfg()},
      sensors=(terrain_scan, feet_contact, pelvis_contact),
      num_envs=1,
      extent=2.5,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="Pelvis_Link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      # The host model compiles at qpos0 (root at z=0, buried in the terrain), which
      # needs ~185 contacts; the per-world pool must cover that.
      nconmax=200,
      njmax=1500,
      contact_sensor_maxmatch=500,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=10.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain, mode="reset", params={}
    )
    assert cfg.scene.terrain is not None
    generator = cfg.scene.terrain.terrain_generator
    if generator is not None:
      generator.curriculum = False
      generator.num_cols = 5
      generator.num_rows = 5
      generator.border_width = 10.0
    twist = cfg.commands["twist"]
    assert isinstance(twist, mdp.UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (0.0, 0.5)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg


def tocabi_flat_env_cfg(
  play: bool = False, encoder_bias: bool = True
) -> ManagerBasedRlEnvCfg:
  """Create TOCABI flat terrain velocity configuration.

  Arguments are as in :func:`tocabi_rough_env_cfg`.
  """
  cfg = tocabi_rough_env_cfg(play=play, encoder_bias=encoder_bias)

  cfg.sim.njmax = 1000
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)
  cfg.events.pop("randomize_terrain", None)

  if play:
    # As in DYROS, flat play keeps the training command ranges (rough play narrows
    # them).
    twist = cfg.commands["twist"]
    assert isinstance(twist, mdp.UniformVelocityCommandCfg)
    twist.ranges = replace(TWIST_RANGES)
    twist.resampling_time_range = (10.0, 10.0)

  return cfg
