"""Tests for the play script's --velocity-command override."""

import pytest

from mjlab.scripts.play import fix_velocity_command
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg
from mjlab.tasks.tocabi.config.env_cfgs import tocabi_rough_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_fix_velocity_command() -> None:
  cfg = tocabi_rough_env_cfg(play=True)
  fix_velocity_command(cfg, (0.5, -0.1, 0.2))
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert twist.ranges.lin_vel_x == (0.5, 0.5)
  assert twist.ranges.lin_vel_y == (-0.1, -0.1)
  assert twist.ranges.ang_vel_z == (0.2, 0.2)
  assert twist.rel_standing_envs == 0.0
  assert not twist.heading_command
  assert cfg.curriculum == {}


def test_fix_velocity_command_requires_velocity_task() -> None:
  with pytest.raises(ValueError, match="velocity command"):
    fix_velocity_command(cartpole_balance_env_cfg(), (0.5, 0.0, 0.0))
