"""TOCABI lower-body action term.

Ported from the DYROS IsaacLab ``LowerBodyActions`` term, with the actuator-side
behavior of the HojaeGomtang ``TocabiAction`` term (sim-to-real validated):

- The policy drives the lower body, either as normalized torques (``pd_control=False``)
  or as joint position targets tracked by a PD loop (``pd_control=True``). Position
  targets are offsets from the default pose (DYROS) or, with ``joint_pos_limits``,
  actions in [-1, 1] mapped onto absolute joint ranges (HojaeGomtang).
- The upper body is held at its default pose by a PD loop.
- PD gains are divided by 9 (P) and 3 (D), matching both reference implementations.
- PD feedback uses encoder-biased joint positions and low-pass filtered joint
  velocities.
- Optional motor-strength scaling, uniform torque injection and Gaussian torque bias
  (per episode and per physics step) on the lower body.
- Efforts are clamped to the torque limits.

Command delay is handled by the actuators (see ``tocabi_constants``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class LowerBodyActionCfg(ActionTermCfg):
  """Configuration for :class:`LowerBodyAction`."""

  lower_joint_names: tuple[str, ...]
  """Policy-controlled joints, in action order."""

  upper_joint_names: tuple[str, ...]
  """Joints held at their default position by PD control."""

  p_gains: tuple[float, ...]
  """P gains for lower then upper joints (before the /9 scaling)."""

  d_gains: tuple[float, ...]
  """D gains for lower then upper joints (before the /3 scaling)."""

  torque_limits: tuple[float, ...]
  """Torque limits for lower then upper joints. In torque mode the lower-body
  action is normalized by these limits."""

  pd_control: bool = False
  """If True, lower-body actions are joint position targets tracked by PD (see
  ``joint_pos_limits``). If False, they are torques normalized by
  ``torque_limits``."""

  joint_pos_limits: tuple[tuple[float, float], ...] | None = None
  """PD mode only. If set, actions in [-1, 1] map linearly onto these absolute
  (min, max) joint ranges, one per lower joint (HojaeGomtang). If None, actions
  are offsets from the default pose."""

  scale: float = 1.0
  """Action scale."""

  action_clip: tuple[float, float] | None = (-1.0, 1.0)
  """Bounds applied to the scaled action. None disables clipping."""

  p_gain_divisor: float = 9.0
  d_gain_divisor: float = 3.0

  joint_vel_lpf_cutoff_hz: float = 25.0
  """Cutoff of the first-order low-pass filter on the joint velocity used in the
  D term. 0 disables filtering."""

  rand_motor_scale_range: tuple[float, float] = (1.0, 1.0)
  """Uniform motor strength scale applied to lower-body torques in torque mode,
  resampled every physics step."""

  rand_torque_inj_range: tuple[float, float] = (0.0, 0.0)
  """Uniform torque added to the lower body in torque mode, resampled every
  physics step [N·m]."""

  enable_torque_bias: bool = False
  """Add Gaussian torque bias on the lower body."""

  torque_bias_episode: tuple[float, float] = (0.0, 0.0)
  """(mean, std) of the lower-body torque bias sampled at reset [N·m]."""

  torque_bias_tick: tuple[float, float] = (0.0, 0.0)
  """(mean, std) of the lower-body torque bias sampled every physics step [N·m]."""

  def __post_init__(self) -> None:
    num_joints = len(self.lower_joint_names) + len(self.upper_joint_names)
    for name in ("p_gains", "d_gains", "torque_limits"):
      if len(getattr(self, name)) != num_joints:
        raise ValueError(
          f"{name} has {len(getattr(self, name))} entries, expected {num_joints}"
          " (lower then upper joints)."
        )
    if self.joint_pos_limits is not None and len(self.joint_pos_limits) != len(
      self.lower_joint_names
    ):
      raise ValueError("joint_pos_limits needs one (min, max) per lower joint.")

  def build(self, env: ManagerBasedRlEnv) -> LowerBodyAction:
    return LowerBodyAction(self, env)


class LowerBodyAction(ActionTerm):
  """Lower-body policy action with upper-body PD hold."""

  cfg: LowerBodyActionCfg

  def __init__(self, cfg: LowerBodyActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    lower_ids, _ = self._entity.find_joints(cfg.lower_joint_names, preserve_order=True)
    upper_ids, _ = self._entity.find_joints(cfg.upper_joint_names, preserve_order=True)
    self._num_lower = len(lower_ids)
    self._joint_ids = torch.tensor(
      lower_ids + upper_ids, device=self.device, dtype=torch.long
    )

    self._p_gains = torch.tensor(cfg.p_gains, device=self.device) / cfg.p_gain_divisor
    self._d_gains = torch.tensor(cfg.d_gains, device=self.device) / cfg.d_gain_divisor
    self._torque_limits = torch.tensor(cfg.torque_limits, device=self.device)
    self._default_pos = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    self._joint_pos_limits = (
      torch.tensor(cfg.joint_pos_limits, device=self.device)
      if cfg.joint_pos_limits is not None
      else None
    )

    self._raw_actions = torch.zeros(self.num_envs, self._num_lower, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    num_joints = len(self._joint_ids)
    self._joint_vel_lpf = torch.zeros(self.num_envs, num_joints, device=self.device)
    if cfg.joint_vel_lpf_cutoff_hz > 0.0:
      rc = 1.0 / (2.0 * math.pi * cfg.joint_vel_lpf_cutoff_hz)
      self._lpf_alpha = env.physics_dt / (rc + env.physics_dt)
    else:
      self._lpf_alpha = 1.0
    self._episode_torque_bias = torch.zeros(
      self.num_envs, self._num_lower, device=self.device
    )
    self._applied_effort = torch.zeros(self.num_envs, num_joints, device=self.device)
    self._pd_effort = torch.zeros_like(self._applied_effort)

  @property
  def action_dim(self) -> int:
    return self._num_lower

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def processed_action(self) -> torch.Tensor:
    """Scaled and clipped action (position offset or normalized torque)."""
    return self._processed_actions

  @property
  def joint_ids(self) -> torch.Tensor:
    """Entity joint indices, lower then upper."""
    return self._joint_ids

  @property
  def torque_limits(self) -> torch.Tensor:
    """Torque limits, lower then upper joints."""
    return self._torque_limits

  @property
  def pd_effort(self) -> torch.Tensor:
    """Control effort before torque bias and clamping on the last physics step."""
    return self._pd_effort

  @property
  def applied_effort(self) -> torch.Tensor:
    """Efforts sent to the actuators on the last physics step, lower then upper."""
    return self._applied_effort

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    processed = actions * self.cfg.scale
    if self.cfg.action_clip is not None:
      processed = processed.clamp(*self.cfg.action_clip)
    self._processed_actions[:] = processed

  def apply_actions(self) -> None:
    n = self._num_lower
    data = self._entity.data
    joint_pos = data.joint_pos_biased[:, self._joint_ids]
    joint_vel = data.joint_vel[:, self._joint_ids]
    self._joint_vel_lpf += self._lpf_alpha * (joint_vel - self._joint_vel_lpf)

    target_pos = self._default_pos.clone()
    if self.cfg.pd_control and self._joint_pos_limits is not None:
      lo, hi = self._joint_pos_limits[:, 0], self._joint_pos_limits[:, 1]
      action = self._processed_actions.clamp(-1.0, 1.0)
      target_pos[:, :n] = lo + 0.5 * (action + 1.0) * (hi - lo)
    elif self.cfg.pd_control:
      target_pos[:, :n] += self._processed_actions
    effort = (
      self._p_gains * (target_pos - joint_pos) - self._d_gains * self._joint_vel_lpf
    )

    if not self.cfg.pd_control:
      motor_scale = sample_uniform(
        *self.cfg.rand_motor_scale_range, (self.num_envs, n), device=self.device
      )
      torque_inj = sample_uniform(
        *self.cfg.rand_torque_inj_range, (self.num_envs, n), device=self.device
      )
      effort[:, :n] = (
        self._processed_actions * self._torque_limits[:n] * motor_scale + torque_inj
      )

    self._pd_effort[:] = effort
    if self.cfg.enable_torque_bias:
      mean, std = self.cfg.torque_bias_tick
      tick_bias = torch.randn_like(self._episode_torque_bias) * std + mean
      effort[:, :n] += self._episode_torque_bias + tick_bias

    effort = torch.clamp(effort, -self._torque_limits, self._torque_limits)
    self._applied_effort[:] = effort
    self._entity.set_joint_effort_target(effort, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0
    self._joint_vel_lpf[env_ids] = self._entity.data.joint_vel[env_ids][
      :, self._joint_ids
    ]
    if self.cfg.enable_torque_bias:
      mean, std = self.cfg.torque_bias_episode
      bias = self._episode_torque_bias[env_ids]
      self._episode_torque_bias[env_ids] = torch.randn_like(bias) * std + mean
