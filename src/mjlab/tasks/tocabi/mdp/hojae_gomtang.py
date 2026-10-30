"""TOCABI leg-policy MDP terms ported from the HojaeGomtang code.

Source: LearningHumanoidArmMotion (tocabi branch), extensions/tocabi. The reward
functions keep the original sign convention: penalties return negative values and
are configured with positive weights, so the original weights carry over.

Foot conventions: feet are ``asset_cfg`` bodies or contact-sensor primaries ordered
``(left, right)``. The gait phase runs over one full cycle (two steps): the left foot
swings for phase < 0.5 and the right foot for phase >= 0.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  sample_uniform,
  yaw_quat,
)

from .actions import LowerBodyAction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

REWARD_TRACKING_SIGMA = 0.25


def _negsqrd_exp(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
  """exp(-(x / scale)^2 / REWARD_TRACKING_SIGMA)."""
  return torch.exp(-torch.square(x / scale) / REWARD_TRACKING_SIGMA)


def _lpf_dt(env: ManagerBasedRlEnv) -> float:
  # The original computes observation filters and action-smoothness rates with
  # step_dt * decimation (0.08 s at 50 Hz), not step_dt. Kept so that the filter
  # response and the reward weights match.
  return env.step_dt * env.cfg.decimation


##
# Commands.
##


class HojaeGomtangVelocityCommand(UniformVelocityCommand):
  """Uniform velocity command that zeroes small samples."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    cfg = self.cfg
    assert isinstance(cfg, HojaeGomtangVelocityCommandCfg)
    cmd = self.vel_command_b[env_ids]
    keep_lin = torch.norm(cmd[:, :2], dim=1) > cfg.lin_vel_threshold
    keep_ang = torch.abs(cmd[:, 2]) > cfg.ang_vel_threshold
    cmd[:, :2] *= keep_lin.unsqueeze(1)
    cmd[:, 2] *= keep_ang
    self.vel_command_b[env_ids] = cmd
    self.vel_command_w[env_ids] = cmd


@dataclass(kw_only=True)
class HojaeGomtangVelocityCommandCfg(UniformVelocityCommandCfg):
  lin_vel_threshold: float = 0.2
  """Linear commands with a smaller xy norm are set to zero [m/s]."""
  ang_vel_threshold: float = 0.1
  """Yaw-rate commands with a smaller magnitude are set to zero [rad/s]."""

  def build(self, env: ManagerBasedRlEnv) -> HojaeGomtangVelocityCommand:
    return HojaeGomtangVelocityCommand(self, env)


class StepPhaseCommand(CommandTerm):
  """Gait phase with a per-env step period and a random start phase.

  Each env gets a fixed step period, spread evenly over ``step_period_range``
  (policy steps per step). A full gait cycle is two steps. Every episode starts at
  phase 0 or 0.5 with equal probability, so either foot may swing first.

  With ``startup_steps`` > 0 every episode begins with a double-support window of
  that many policy steps: the phase holds at its start value while
  ``phase_scale`` ramps from 0 to 1, so the clock observation r * (sin, cos)
  starts at the origin, a value walking never produces.
  """

  cfg: StepPhaseCommandCfg

  def __init__(self, cfg: StepPhaseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    lo, hi = cfg.step_period_range
    step = torch.linspace(lo, hi, self.num_envs, device=self.device)
    step = torch.round(step).clamp(min=1).long()
    self.full_step_period = (2 * step).unsqueeze(1)
    self.phase_offset = torch.zeros_like(self.full_step_period)

  @property
  def command(self) -> torch.Tensor:
    """Full gait cycle in policy steps. Shape (num_envs, 1)."""
    return self.full_step_period.float()

  @property
  def phase(self) -> torch.Tensor:
    """Gait phase in [0, 1). Shape (num_envs,)."""
    walked = (self._env.episode_length_buf - self.cfg.startup_steps).clamp(min=0)
    steps = walked.unsqueeze(1) + self.phase_offset
    return (torch.remainder(steps, self.full_step_period) / self.full_step_period)[:, 0]

  @property
  def phase_scale(self) -> torch.Tensor:
    """Clock observation scale r in [0, 1], 1 after startup. Shape (num_envs,)."""
    if self.cfg.startup_steps <= 0:
      return torch.ones(self.num_envs, device=self.device)
    buf = self._env.episode_length_buf.float()
    return (buf / self.cfg.startup_steps).clamp(max=1.0)

  @property
  def in_startup(self) -> torch.Tensor:
    """True during the double-support startup window. Shape (num_envs,)."""
    return self._env.episode_length_buf < self.cfg.startup_steps

  @property
  def first_stance_foot(self) -> torch.Tensor:
    """Foot that carries the first swing, 0 left / 1 right. Shape (num_envs,)."""
    # Phase 0 swings the left foot first, so the right foot is the stance foot.
    return (self.phase_offset[:, 0] == 0).long()

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    half = self.full_step_period[env_ids] // 2
    start_at_half = torch.rand(len(env_ids), 1, device=self.device) < 0.5
    self.phase_offset[env_ids] = torch.where(start_at_half, half, 0)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass


@dataclass(kw_only=True)
class StepPhaseCommandCfg(CommandTermCfg):
  step_period_range: tuple[int, int] = (35, 45)
  """Range of policy steps per step (half gait cycle)."""
  startup_steps: int = 0
  """Policy steps of double support before the clock starts (0 disables)."""

  def build(self, env: ManagerBasedRlEnv) -> StepPhaseCommand:
    return StepPhaseCommand(self, env)


def _phase_term(env: ManagerBasedRlEnv, command_name: str) -> StepPhaseCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, StepPhaseCommand)
  return term


def _phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _phase_term(env, command_name).phase


def smooth_sqr_wave(phase: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
  """Smoothed square wave: positive for phase in (0, 0.5), negative after."""
  p = torch.sin(2.0 * math.pi * phase)
  return p / torch.sqrt(p * p + eps * eps)


##
# Observations.
##


def phase_sin(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  term = _phase_term(env, command_name)
  return (term.phase_scale * torch.sin(2 * math.pi * term.phase)).unsqueeze(1)


def phase_cos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  term = _phase_term(env, command_name)
  return (term.phase_scale * torch.cos(2 * math.pi * term.phase)).unsqueeze(1)


def joint_pos_abs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  biased: bool = False,
) -> torch.Tensor:
  """Absolute joint positions (not relative to the default pose)."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return joint_pos[:, asset_cfg.joint_ids]


def last_raw_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  return env.action_manager.get_term(action_name).raw_action


def base_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def base_heading(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.heading_w.unsqueeze(1)


class _LowPassObs:
  """First-order low-pass filter over an observation, applied before noise.

  As in the original, the filter state is not reset between episodes.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    cutoff_hz = cfg.params.get("cutoff_hz", 25.0)
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = _lpf_dt(env)
    self._alpha = dt / (rc + dt)
    self._state: torch.Tensor | None = None

  def _raw(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    raise NotImplementedError

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    cutoff_hz: float = 25.0,
  ) -> torch.Tensor:
    del cutoff_hz  # Used in __init__.
    raw = self._raw(env, asset_cfg)
    if self._state is None:
      state = raw.clone()
    else:
      state = self._state + self._alpha * (raw - self._state)
    self._state = state
    return state.clone()


class base_ang_vel_lpf(_LowPassObs):
  """Low-pass filtered base angular velocity in the base frame."""

  def _raw(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b


class joint_vel_lpf(_LowPassObs):
  """Low-pass filtered joint velocities."""

  def _raw(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


##
# Rewards.
##


def _feet_force_z(sensor: ContactSensor) -> torch.Tensor:
  """Vertical ground reaction force per foot, (left, right). Shape (N, 2).

  The sensor reports the force the foot exerts (negative z while standing); the
  original uses the force acting on the foot, so the sign is flipped.
  """
  assert sensor.data.force is not None
  return -sensor.data.force[..., 2]


def _feet_pos_yaw_frame(asset: Entity, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Feet positions relative to the root in the yaw-aligned frame. (N, 2, 3)."""
  rel = asset.data.body_link_pos_w[:, asset_cfg.body_ids] - (
    asset.data.root_link_pos_w.unsqueeze(1)
  )
  quat = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, 2, -1)
  return quat_apply_inverse(quat, rel)


def survival(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def action_smoothness1(env: ManagerBasedRlEnv) -> torch.Tensor:
  """-sum((a_t - a_{t-1})^2) / dt^2 over the clipped policy action."""
  am = env.action_manager
  a, a1 = am.action.clamp(-1.0, 1.0), am.prev_action.clamp(-1.0, 1.0)
  return -torch.sum(torch.square(a - a1), dim=1) / _lpf_dt(env) ** 2


def action_smoothness2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """-sum((a_t - 2 a_{t-1} + a_{t-2})^2) / dt^2 over the clipped policy action."""
  am = env.action_manager
  a = am.action.clamp(-1.0, 1.0)
  a1 = am.prev_action.clamp(-1.0, 1.0)
  a2 = am.prev_prev_action.clamp(-1.0, 1.0)
  return -torch.sum(torch.square(a - 2 * a1 + a2), dim=1) / _lpf_dt(env) ** 2


def joint_torque_penalty(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """-sum(tau^2) of the lower-body PD torque (before bias and clamping)."""
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, LowerBodyAction)
  return -torch.sum(torch.square(term.pd_effort[:, : term.action_dim]), dim=1)


def joint_torque_limits_penalty(
  env: ManagerBasedRlEnv, action_name: str, soft_limit_factor: float = 0.8
) -> torch.Tensor:
  """-sum(max(|tau| - soft_limit_factor * tau_max, 0)) over the lower body."""
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, LowerBodyAction)
  n = term.action_dim
  limits = soft_limit_factor * term.torque_limits[:n]
  excess = torch.abs(term.applied_effort[:, :n]) - limits
  return -torch.sum(excess.clamp(min=0.0), dim=1)


def joint_velocity_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return -torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  margin: float = 0.1,
  outside_weight: float = 1.0,
) -> torch.Tensor:
  """Quadratic penalty within ``margin`` of the soft limits, linear outside."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  dist_lower = q - limits[..., 0]
  dist_upper = limits[..., 1] - q
  outside = torch.relu(-dist_lower) + torch.relu(-dist_upper)
  inside = torch.zeros_like(q)
  if margin > 0.0:
    in_lower = torch.relu(margin - dist_lower) * (dist_lower >= 0.0)
    in_upper = torch.relu(margin - dist_upper) * (dist_upper >= 0.0)
    inside = torch.square(in_lower / margin) + torch.square(in_upper / margin)
  return -torch.sum(inside + outside_weight * outside, dim=1)


def foot_flat_orientation_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """-sum(roll^2 + pitch^2) of the feet."""
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids].reshape(-1, 4)
  roll, pitch, _ = euler_xyz_from_quat(quat)
  roll = roll.view(env.num_envs, -1).clamp(-math.pi / 2, math.pi / 2)
  pitch = pitch.view(env.num_envs, -1).clamp(-math.pi / 2, math.pi / 2)
  return -torch.sum(roll * roll + pitch * pitch, dim=1)


def swing_foot_height_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  max_height: float = 0.18,
  min_height: float = 0.14,
  rise_end: float = 0.24,
  hold_end: float = 0.25,
  swing_end: float = 0.5,
) -> torch.Tensor:
  """Penalize the swing foot for being below a smooth rise-hold-fall profile.

  Heights are world z of the foot bodies. The stance foot is not penalized, and
  neither foot is during the startup window.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  term = _phase_term(env, command_name)
  phase = term.phase

  def smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

  def profile(s: torch.Tensor) -> torch.Tensor:
    amp = max_height - min_height
    rise = min_height + amp * smoothstep(s / rise_end)
    fall = min_height + amp * (
      1.0 - smoothstep((s - hold_end) / (swing_end - hold_end))
    )
    hold = torch.full_like(s, max_height)
    low = torch.full_like(s, min_height)
    return torch.where(
      s < rise_end,
      rise,
      torch.where(s < hold_end, hold, torch.where(s < swing_end, fall, low)),
    )

  left_swing = phase < 0.5
  target = torch.stack(
    [
      torch.where(left_swing, profile(phase), 0.0),
      torch.where(~left_swing, profile(phase - 0.5), 0.0),
    ],
    dim=1,
  )
  swing = torch.stack([left_swing, ~left_swing], dim=1).float()
  penalty = -torch.sum((target - foot_z).clamp(min=0.0) * swing, dim=1)
  return penalty * (~term.in_startup).float()


def base_lin_vel_z_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return -torch.square(asset.data.root_link_lin_vel_b[:, 2])


def base_ang_vel_xy_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return -torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_regularization(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Mean Gaussian closeness of the selected joints to their default positions."""
  asset: Entity = env.scene[asset_cfg.name]
  error = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.mean(_negsqrd_exp(error), dim=1)


def knee_negative_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, min_knee_pos: float = 0.6
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return -torch.sum(torch.relu(min_knee_pos - q), dim=1)


def hip_yaw_negative_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """-sum(|q|) of the hip yaw joints."""
  asset: Entity = env.scene[asset_cfg.name]
  return -torch.sum(torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)


def hip_yaw_sum_zero_reward(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Reward left and right hip yaw summing to zero."""
  asset: Entity = env.scene[asset_cfg.name]
  return _negsqrd_exp(torch.sum(asset.data.joint_pos[:, asset_cfg.joint_ids], dim=1))


def hip_roll_threshold_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  left_min: float = -0.06,
  right_max: float = 0.06,
  linear: bool = False,
  stance_bound: float | None = None,
  command_name: str = "step_phase",
) -> torch.Tensor:
  """Penalize left hip roll below ``left_min`` and right above ``right_max``.

  The original charges 1 plus the excess as soon as a bound is crossed; with
  ``linear`` only the excess is charged. With ``stance_bound`` the stance leg's
  inward bound widens to that magnitude, so the pelvis can shift over the stance
  foot while the swing leg keeps the original bound. ``asset_cfg`` joints are
  ordered (left, right).
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  lo = torch.full_like(q[:, 0], left_min)
  hi = torch.full_like(q[:, 1], right_max)
  if stance_bound is not None:
    term = _phase_term(env, command_name)
    # Right stance weight, smoothly 1 in the first half cycle and 0 in the second.
    right_stance = 0.5 * (1.0 + smooth_sqr_wave(term.phase, eps=0.1))
    right_stance = torch.where(
      term.in_startup, term.first_stance_foot.float(), right_stance
    )
    lo = torch.lerp(lo, torch.full_like(lo, -stance_bound), 1.0 - right_stance)
    hi = torch.lerp(hi, torch.full_like(hi, stance_bound), right_stance)
  left = torch.relu(lo - q[:, 0])
  right = torch.relu(q[:, 1] - hi)
  if linear:
    return -(left + right)
  return -((left > 0.0) * (1.0 + left) + (right > 0.0) * (1.0 + right))


def foot_yaw_frame_y_distance_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.18
) -> torch.Tensor:
  """Penalize a lateral foot separation below ``threshold`` in the yaw frame."""
  asset: Entity = env.scene[asset_cfg.name]
  feet = _feet_pos_yaw_frame(asset, asset_cfg)
  y_dist = torch.abs(feet[:, 0, 1] - feet[:, 1, 1])
  return -torch.relu(threshold - y_dist)


def grf_limit_penalty(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1500.0
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  excess = (_feet_force_z(sensor) - threshold).clamp(min=0.0)
  return -torch.sum(torch.square(excess / threshold), dim=1)


def foot_slip_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  contact_force_threshold: float = 1.0,
  vel_scale: float = 0.5,
) -> torch.Tensor:
  """Penalize horizontal foot speed while the foot is loaded."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  speed_sq = torch.sum(
    torch.square(asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=-1
  )
  contact = (_feet_force_z(sensor) > contact_force_threshold).float()
  return -torch.sum(contact * speed_sq / max(vel_scale * vel_scale, 1e-6), dim=1)


def base_z_orientation_reward(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  error = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
  return _negsqrd_exp(error, scale=0.2)


def _tracking(
  command: torch.Tensor, actual: torch.Tensor, scale: float
) -> torch.Tensor:
  error = (command - actual) / (1.0 + torch.abs(command))
  return _negsqrd_exp(error, scale=scale)


def tracking_lin_vel_x_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return _tracking(command[:, 0], asset.data.root_link_lin_vel_b[:, 0], 0.3)


def tracking_lin_vel_y_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return _tracking(command[:, 1], asset.data.root_link_lin_vel_b[:, 1], 0.25)


def tracking_yaw_vel_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return _tracking(command[:, 2], asset.data.root_link_ang_vel_b[:, 2], 0.3)


def contact_schedule_reward(
  env: ManagerBasedRlEnv, command_name: str, sensor_name: str
) -> torch.Tensor:
  """Reward right-foot contact in the first half cycle and left-foot contact in the
  second (the opposite foot swings). During startup, reward double support."""
  sensor: ContactSensor = env.scene[sensor_name]
  contact = (_feet_force_z(sensor) > 0.0).float()
  term = _phase_term(env, command_name)
  schedule = smooth_sqr_wave(term.phase)
  walking = (contact[:, 1] - contact[:, 0]) * schedule
  standing = contact[:, 0] + contact[:, 1] - 1.0
  return torch.where(term.in_startup, standing, walking)


def joint_pos_square_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """-sum(q^2) of the selected joints (zero is their default position)."""
  asset: Entity = env.scene[asset_cfg.name]
  return -torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)


def base_roll_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """-g_y^2 of the projected gravity, about sin^2 of the base roll angle."""
  asset: Entity = env.scene[asset_cfg.name]
  return -torch.square(asset.data.projected_gravity_b[:, 1])


def com_lateral_tracking_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  amplitude: float = 0.04,
  sigma: float = 0.03,
) -> torch.Tensor:
  """Reward the whole-body CoM for following a lateral sway over the stance foot.

  In the yaw frame, relative to the midpoint of the feet (``asset_cfg`` bodies),
  the reference is ``-amplitude * sin(2 pi phase)``: toward the right foot while
  the left swings and toward the left foot after. Off during startup.
  """
  asset: Entity = env.scene[asset_cfg.name]
  com_w = asset.data.data.subtree_com[:, asset.data.indexing.root_body_id]
  mid_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids].mean(dim=1)
  rel = quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), com_w - mid_w)
  term = _phase_term(env, command_name)
  target = -amplitude * torch.sin(2.0 * math.pi * term.phase)
  reward = torch.exp(-torch.square((rel[:, 1] - target) / sigma))
  return reward * (~term.in_startup).float()


def startup_weight_shift_reward(
  env: ManagerBasedRlEnv, command_name: str, sensor_name: str
) -> torch.Tensor:
  """During startup, reward the share of vertical load on the first stance foot.

  Weighted by the clock ramp r, so the load is expected on that foot by the time
  the first swing begins.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = _feet_force_z(sensor).clamp(min=0.0)
  term = _phase_term(env, command_name)
  stance = force.gather(1, term.first_stance_foot.unsqueeze(1))[:, 0]
  share = stance / (force.sum(dim=1) + 1e-6)
  return share * term.phase_scale * term.in_startup.float()


class lateral_drift_penalty:
  """Penalize the lateral velocity error averaged over one gait cycle.

  The per-step lateral sway of the pelvis averages out over a cycle (two steps, the
  env's own period), so this charges only a steady sideways drift that the
  instantaneous velocity tracking barely notices. The error is measured in the yaw
  frame against the twist command; the penalty is its squared cycle mean, 0 during
  startup and until a full cycle has been walked.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    term = _phase_term(env, cfg.params["command_name"])
    self._period = term.full_step_period[:, 0]
    n, size = env.num_envs, int(self._period.max().item())
    self._buf = torch.zeros(n, size, device=env.device)
    self._age = torch.arange(size, device=env.device)
    self._head = 0
    self._count = torch.zeros(n, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    twist_command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    term = _phase_term(env, command_name)
    vel = quat_apply_inverse(
      yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w
    )
    command = env.command_manager.get_command(twist_command_name)
    assert command is not None
    error = vel[:, 1] - command[:, 1]
    size = self._buf.shape[1]
    self._head = (self._head + 1) % size
    self._buf[:, self._head] = error
    walking = ~term.in_startup
    self._count = torch.where(walking, self._count + 1, 0)
    # Entries written within the last period steps, newest has age 0.
    age = (self._head - self._age) % size
    window = age.unsqueeze(0) < self._period.unsqueeze(1)
    mean = (self._buf * window).sum(dim=1) / self._period
    full = self._count >= self._period
    return -torch.square(mean) * full.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._count[env_ids] = 0


def startup_com_shift_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  shift: float = 0.04,
  sigma: float = 0.03,
) -> torch.Tensor:
  """During startup, reward the CoM for moving toward the first stance foot.

  In the yaw frame, relative to the midpoint of the feet (``asset_cfg`` bodies), the
  reference ramps with the clock scale r from 0 to ``shift`` on the stance side, so
  the weight shift is a visible body motion rather than only a change of the foot
  loads.
  """
  asset: Entity = env.scene[asset_cfg.name]
  com_w = asset.data.data.subtree_com[:, asset.data.indexing.root_body_id]
  mid_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids].mean(dim=1)
  rel = quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), com_w - mid_w)
  term = _phase_term(env, command_name)
  side = 1.0 - 2.0 * term.first_stance_foot.float()  # +y toward the left foot
  target = side * shift * term.phase_scale
  reward = torch.exp(-torch.square((rel[:, 1] - target) / sigma))
  return reward * term.in_startup.float()


def termination_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """-1/dt on non-timeout terminations (so weight w costs w per termination)."""
  return -env.termination_manager.terminated.float() / env.step_dt


##
# Terminations.
##


def base_termination(
  env: ManagerBasedRlEnv,
  max_lin_vel: float = 15.0,
  max_ang_vel: float = 10.0,
  max_tilting: float = 0.8,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate on excessive base speed or a projected gravity x/y beyond
  ``max_tilting``."""
  asset: Entity = env.scene[asset_cfg.name]
  data = asset.data
  return (
    (torch.norm(data.root_link_lin_vel_b, dim=1) > max_lin_vel)
    | (torch.norm(data.root_link_ang_vel_b, dim=1) > max_ang_vel)
    | torch.any(torch.abs(data.projected_gravity_b[:, :2]) > max_tilting, dim=1)
  )


def joint_pos_out_of_bounds(
  env: ManagerBasedRlEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.any((q < bounds[0]) | (q > bounds[1]), dim=1)


def foot_crossing_yaw_frame(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, min_y_dist: float = 0.16
) -> torch.Tensor:
  """Terminate when the left foot is not at least ``min_y_dist`` left of the right
  foot in the yaw frame."""
  asset: Entity = env.scene[asset_cfg.name]
  feet = _feet_pos_yaw_frame(asset, asset_cfg)
  return feet[:, 0, 1] <= feet[:, 1, 1] + min_y_dist


##
# Events.
##


_PUSH_DEFAULTS: dict[str, Any] = {
  "impulse_range": (-50.0, 50.0),
  "lateral_scale": 1.0,
  "torque_impulse_range": (-1.5, 1.5),
  "windows": ((0.2, 0.4), (0.6, 0.8)),
  "halfspan_steps": (0, 2),
  "curriculum_iterations": (3000, 10000),
  "steps_per_iteration": 24,
}


class scheduled_push:
  """Two body-frame force/torque pulses on a body per episode, ramped in by a
  curriculum.

  Each episode samples one pulse inside each of ``windows`` (fractions of the
  episode). A pulse delivers an impulse sampled uniformly per axis from
  ``impulse_range`` [N·s] (lateral y scaled by ``lateral_scale``) and
  ``torque_impulse_range`` [N·m·s], spread over 2h+1 policy steps with h in
  ``halfspan_steps``. The impulse is scaled from 0 to 1 between training iterations
  ``curriculum_iterations`` (``steps_per_iteration`` env steps each).

  The original applies impulse / (duration * physics_dt) for one physics step per
  policy step; here impulse / (duration * step_dt) is held over each policy step,
  which gives the same total impulse. Use with ``mode="step"``.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._p: dict[str, Any] = {**_PUSH_DEFAULTS, **cfg.params}
    asset_cfg: SceneEntityCfg = self._p["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list) and len(body_ids) == 1, "Push one body."
    self._body_ids: list[int] = body_ids
    self._env = env
    n, device = env.num_envs, env.device
    self._center = torch.zeros(n, 2, dtype=torch.long, device=device)
    self._halfspan = torch.zeros(n, 2, dtype=torch.long, device=device)
    self._impulse = torch.zeros(n, 2, 3, device=device)
    self._torque_impulse = torch.zeros(n, 2, 3, device=device)
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env = self._env
    params = self._p
    ids = torch.arange(env.num_envs, device=env.device)[
      env_ids if env_ids is not None else slice(None)
    ]
    n = len(ids)
    if n == 0:
      return
    f_lo, f_hi = params["impulse_range"]
    t_lo, t_hi = params["torque_impulse_range"]
    impulse = sample_uniform(f_lo, f_hi, (n, 2, 3), env.device)
    impulse[..., 1] *= params["lateral_scale"]
    self._impulse[ids] = impulse
    self._torque_impulse[ids] = sample_uniform(t_lo, t_hi, (n, 2, 3), env.device)
    max_ep = int(env.max_episode_length)
    h_min, h_max = params["halfspan_steps"]
    for k, (lo, hi) in enumerate(params["windows"]):
      start = int(lo * max_ep)
      end = max(start + 1, int(hi * max_ep))  # Exclusive.
      h_cap = (end - start - 1) // 2
      h = torch.randint(
        min(h_min, h_cap), min(h_max, h_cap) + 1, (n,), device=env.device
      )
      # Center in [start + h, end - 1 - h].
      span = (end - 1 - 2 * h - start + 1).clamp(min=1)
      self._center[ids, k] = (
        start + h + (torch.rand(n, device=env.device) * span).long()
      )
      self._halfspan[ids, k] = h
    zeros = torch.zeros(n, len(self._body_ids), 3, device=env.device)
    self._asset.write_external_wrench_to_sim(
      zeros, zeros, env_ids=ids, body_ids=self._body_ids
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    **params: Any,
  ) -> None:
    del env_ids, asset_cfg, params  # Read in __init__ (see _PUSH_DEFAULTS).
    p = self._p
    lo, hi = p["curriculum_iterations"]
    steps_per_iteration = p["steps_per_iteration"]
    iteration = env.common_step_counter / steps_per_iteration
    scale = min(max((iteration - lo) / max(hi - lo, 1), 0.0), 1.0)

    step = env.episode_length_buf.unsqueeze(1)  # (N, 1)
    active = (step >= self._center - self._halfspan) & (
      step <= self._center + self._halfspan
    )  # (N, 2)
    duration_s = ((2 * self._halfspan + 1) * env.step_dt).unsqueeze(-1)  # (N, 2, 1)
    weight = active.unsqueeze(-1).float() * scale / duration_s
    force_b = torch.sum(self._impulse * weight, dim=1)  # (N, 3)
    torque_b = torch.sum(self._torque_impulse * weight, dim=1)
    quat = self._asset.data.body_link_quat_w[:, self._body_ids[0]]
    force_w = quat_apply(quat, force_b).unsqueeze(1)
    torque_w = quat_apply(quat, torque_b).unsqueeze(1)
    self._asset.write_external_wrench_to_sim(force_w, torque_w, body_ids=self._body_ids)
