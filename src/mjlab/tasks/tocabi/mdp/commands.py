"""TOCABI walking phase command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class WalkingPhaseCommand(CommandTerm):
  """Gait period command: one full left/right step cycle in seconds."""

  cfg: WalkingPhaseCommandCfg

  def __init__(self, cfg: WalkingPhaseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.phase_time = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Gait period. Shape (num_envs, 1)."""
    return self.phase_time

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.phase_time[env_ids, 0] = torch.empty(
      len(env_ids), device=self.device
    ).uniform_(*self.cfg.phase_time_range)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass


@dataclass(kw_only=True)
class WalkingPhaseCommandCfg(CommandTermCfg):
  phase_time_range: tuple[float, float]
  """Range of the gait period in seconds."""

  def build(self, env: ManagerBasedRlEnv) -> WalkingPhaseCommand:
    return WalkingPhaseCommand(self, env)
