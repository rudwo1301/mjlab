"""TOCABI reward terms, ported from the DYROS IsaacLab locomotion task.

Feet are given as ``asset_cfg`` bodies or contact-sensor primaries ordered
``(left, right)``. The gait clock follows :func:`.observations.gait_phase`: the right
foot swings during the first half of the cycle and the left foot during the second.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse

from .observations import gait_phase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _swing_angle(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Von Mises swing angle per foot, (left, right). Shape (N, 2).

  The right foot peaks at a quarter cycle and the left foot half a cycle later.
  """
  phase = gait_phase(env, command_name)
  right = 2 * math.pi * phase - math.pi / 2
  left = 2 * math.pi * torch.remainder(phase - 0.5, 1.0) - math.pi / 2
  return torch.cat([left, right], dim=1)


def _stance_indicator(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """1 for the foot that should be in stance, (left, right). Shape (N, 2)."""
  left = (gait_phase(env, command_name) < 0.5).float()
  return torch.cat([left, 1.0 - left], dim=1)


def _feet_force(sensor: ContactSensor) -> torch.Tensor:
  """Peak contact force magnitude over the sensor history. Shape (N, 2)."""
  data = sensor.data
  if data.force_history is not None:
    return data.force_history.norm(dim=-1).max(dim=-1)[0]
  assert data.force is not None
  return data.force.norm(dim=-1)


def track_lin_vel_xy_base_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  omega: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-omega * |v_cmd_xy - v_xy|^2), linear velocity in the base frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-omega * error)


def track_ang_vel_z_world_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  omega: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-omega * (w_cmd_z - w_z)^2), yaw rate in the world frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
  return torch.exp(-omega * error)


def orientation_tracking(
  env: ManagerBasedRlEnv,
  omega: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-omega * (roll^2 + pitch^2)) of the base."""
  asset: Entity = env.scene[asset_cfg.name]
  roll, pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
  return torch.exp(-omega * (torch.square(roll) + torch.square(pitch)))


def base_height_tracking(
  env: ManagerBasedRlEnv,
  height: float,
  omega: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-omega * (z - height)^2) of the base in the world frame."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-omega * torch.square(asset.data.root_link_pos_w[:, 2] - height))


def periodic_force(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  scale: float,
) -> torch.Tensor:
  """Reward contact force on the stance foot, normalized by ``scale`` and clipped to
  [0, 1]."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = _feet_force(sensor)
  reward = torch.sum(_stance_indicator(env, command_name) * force, dim=1) / scale
  return reward.clamp(0.0, 1.0)


def feet_height_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  omega: float,
  foot_height: float,
  kappa: float,
  offset: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Track a Von Mises foot height profile, foot height in the base frame."""
  asset: Entity = env.scene[asset_cfg.name]
  b, norm = math.exp(-kappa), math.exp(kappa) - math.exp(-kappa)
  ref = foot_height * (
    torch.exp(kappa * torch.cos(_swing_angle(env, command_name))) - b
  )
  ref = ref / norm + offset

  root_pos = asset.data.root_link_pos_w.unsqueeze(1)
  root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
  feet_pos_b = quat_apply_inverse(
    root_quat, asset.data.body_link_pos_w[:, asset_cfg.body_ids] - root_pos
  )
  error = torch.sum(torch.square(feet_pos_b[..., 2] - ref), dim=1)
  return torch.exp(-omega * error)


def feet_velocity_z_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  omega: float,
  foot_height: float,
  kappa: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Track the time derivative of the Von Mises foot height profile."""
  asset: Entity = env.scene[asset_cfg.name]
  theta = _swing_angle(env, command_name)
  phase_time = env.command_manager.get_command(command_name)
  assert phase_time is not None, f"Command '{command_name}' not found."
  dtheta = 2 * math.pi / phase_time
  norm = math.exp(kappa) - math.exp(-kappa)
  d_gx = -kappa * torch.sin(theta) * torch.exp(kappa * torch.cos(theta)) * dtheta
  ref = foot_height * d_gx / norm

  vel_z = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, 2]
  error = torch.sum(torch.square(vel_z - ref), dim=1)
  return torch.exp(-omega * error)


def feet_flat(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize the horizontal component of gravity in each foot frame."""
  asset: Entity = env.scene[asset_cfg.name]
  feet_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
  gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand_as(
    feet_quat[..., :3]
  )
  projected = quat_apply_inverse(feet_quat, gravity)
  return torch.sum(torch.square(projected[..., :2]), dim=(1, 2))


def default_joint_pos(
  env: ManagerBasedRlEnv,
  omega: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-omega * |q - q_default|) over the selected joints."""
  asset: Entity = env.scene[asset_cfg.name]
  error = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.exp(-omega * torch.norm(error, dim=1))


class large_contact_force:
  """Penalize foot contact force above 1.3x robot weight, clipped per foot to
  ``threshold``."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset.indexing.body_ids.cpu().numpy()
    mass = float(env.sim.mj_model.body_mass[body_ids].sum())
    self.max_force = 1.3 * 9.81 * mass

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del asset_cfg  # Used in __init__.
    sensor: ContactSensor = env.scene[sensor_name]
    excess = (_feet_force(sensor) - self.max_force).clamp(0.0, threshold)
    return torch.sum(excess, dim=1)
