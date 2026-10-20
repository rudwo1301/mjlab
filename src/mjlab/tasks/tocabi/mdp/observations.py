"""TOCABI observation terms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .actions import LowerBodyAction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def gait_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Elapsed time in the current gait cycle, normalized to [0, 1). Shape (N, 1)."""
  phase_time = env.command_manager.get_command(command_name)
  assert phase_time is not None, f"Command '{command_name}' not found."
  elapsed = env.episode_length_buf.unsqueeze(1) * env.step_dt
  return torch.remainder(elapsed, phase_time) / phase_time


def clock_input(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Gait clock [sin(2*pi*phase), -sin(2*pi*phase)]. Shape (N, 2)."""
  s = torch.sin(2 * math.pi * gait_phase(env, command_name))
  return torch.cat([s, -s], dim=1)


def lower_body_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Processed (scaled, clipped) action of a :class:`LowerBodyAction` term."""
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, LowerBodyAction)
  return term.processed_action
