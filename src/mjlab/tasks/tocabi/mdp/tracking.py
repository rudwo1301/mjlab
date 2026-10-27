"""TOCABI motion-tracking terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class reference_foot_slip:
  """Penalize foot speed while the reference foot is flat on the ground.

  The motion file's ``contact`` array (T, 2, 2) holds heel/toe contact labels of
  the retargeted human, left then right; a foot counts as planted when both heel
  and toe are down. Returns -sum over planted feet of the squared horizontal speed.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    command = env.command_manager.get_term(cfg.params["command_name"])
    assert isinstance(command, MotionCommand)
    contact = np.load(command.cfg.motion_file)["contact"]
    planted = contact[:, :, 0] & contact[:, :, 1]  # (T, 2)
    self._planted = torch.tensor(planted, dtype=torch.float32, device=env.device)
    self._command = command

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del command_name  # Used in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    planted = self._planted[self._command.time_steps]  # (N, 2)
    return -torch.sum(planted * torch.sum(torch.square(vel), dim=-1), dim=1)
