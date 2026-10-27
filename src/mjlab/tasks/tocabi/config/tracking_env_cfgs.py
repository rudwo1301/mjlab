"""TOCABI motion-tracking task (BeyondMimic, mjlab's tracking task).

Robot, simulation timing, actions and domain randomization come from the
HojaeGomtang leg-policy task, so the two policies run under the same conditions and
the same deployment controller: absolute leg PD targets with the upper body held at
COM728, ankle roll friction randomized up to 25 N·m, no pushes. Observations,
commands, rewards and terminations follow mjlab's tracking task without state
estimation (no base linear velocity or anchor position in the actor), with the
anchor at the pelvis and a slip penalty on the retargeted motion's planted feet.

The motion file is passed at run time: --env.commands.motion.motion-file <npz>.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tocabi.mdp import LowerBodyActionCfg
from mjlab.tasks.tocabi.mdp import tracking as tt
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .env_cfgs import LEGS
from .hojae_gomtang_env_cfgs import (
  LEG_JOINT_POS_LIMITS,
  tocabi_hojae_gomtang_flat_env_cfg,
)

# Retargeted walking reaches ankle pitch -0.86 rad; the walking policy range stops at
# -0.8, so the tracking action range extends it to -1.0.
TRACKING_LEG_JOINT_POS_LIMITS = tuple(
  (-1.0, hi) if i % 6 == 4 else (lo, hi)
  for i, (lo, hi) in enumerate(LEG_JOINT_POS_LIMITS)
)
TRACKED_BODIES = (
  "Pelvis_Link",
  "L_Thigh_Link",
  "L_Knee_Link",
  "L_Foot_Link",
  "R_Thigh_Link",
  "R_Knee_Link",
  "R_Foot_Link",
  "Upperbody_Link",
)
FEET = SceneEntityCfg(
  "robot", body_names=("L_Foot_Link", "R_Foot_Link"), preserve_order=True
)


def tocabi_flat_tracking_env_cfg(
  play: bool = False, privileged_actor: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create the TOCABI flat motion-tracking configuration.

  Args:
    play: Build the evaluation variant.
    privileged_actor: Also give the actor the base linear velocity and the anchor
      position relative to the reference (mjlab's state-estimation variant), to check
      whether the motion can be tracked before removing them.
  """
  cfg = tocabi_hojae_gomtang_flat_env_cfg(
    play=play, ankle_roll_friction_range=(0.0, 25.0)
  )
  tracking = make_tracking_env_cfg()

  ##
  # Sensors
  ##

  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Pelvis_Link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Pelvis_Link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision,)

  ##
  # Actions: HojaeGomtang leg PD, with the ankle pitch range extended.
  ##

  action = cfg.actions["joint_pos"]
  assert isinstance(action, LowerBodyActionCfg)
  action.joint_pos_limits = TRACKING_LEG_JOINT_POS_LIMITS

  ##
  # Commands
  ##

  motion = tracking.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.anchor_body_name = "Pelvis_Link"
  motion.body_names = TRACKED_BODIES
  cfg.commands = {"motion": motion}

  ##
  # Observations (no state estimation: no base linear velocity or anchor position)
  ##

  command = {"command_name": "motion"}
  actor_terms = {
    "command": ObservationTermCfg(func=mdp.generated_commands, params=command),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params=command,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True, "asset_cfg": LEGS},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": LEGS},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic_terms = {
    "command": ObservationTermCfg(func=mdp.generated_commands, params=command),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params=command
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params=command
    ),
    "body_pos": ObservationTermCfg(func=mdp.robot_body_pos_b, params=command),
    "body_ori": ObservationTermCfg(func=mdp.robot_body_ori_b, params=command),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": LEGS}),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": LEGS}),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  if privileged_actor:
    actor_terms["motion_anchor_pos_b"] = ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params=command,
      noise=Unoise(n_min=-0.25, n_max=0.25),
    )
    actor_terms["base_lin_vel"] = ObservationTermCfg(
      func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)
    )
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Rewards and terminations: mjlab tracking, plus planted-foot slip.
  ##

  cfg.rewards = dict(tracking.rewards)
  cfg.rewards["joint_limit"].params["asset_cfg"] = LEGS
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=tt.reference_foot_slip,
    weight=1.0,
    params={**command, "asset_cfg": FEET},
  )
  cfg.terminations = dict(tracking.terminations)
  cfg.terminations["ee_body_pos"].params["body_names"] = ("L_Foot_Link", "R_Foot_Link")

  ##
  # Events: keep the HojaeGomtang randomization; the motion command sets the
  # reset state, and pushes are off.
  ##

  for name in ("reset_base", "reset_robot_joints", "scheduled_push"):
    cfg.events.pop(name, None)

  cfg.curriculum = {}
  cfg.viewer.body_name = "Pelvis_Link"
  cfg.episode_length_s = int(1e9) if play else 10.0
  if play:
    # As mjlab's G1 tracking play: no reset-state randomization, start at frame 0.
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.sampling_mode = "start"
  return cfg
