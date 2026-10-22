"""Tests for the TOCABI velocity environment configs."""

import pytest

from mjlab.tasks.tocabi.config import env_cfgs

SINGLE_STEP_TERMS = {"clock_input", "velocity_commands", "actions", "height_scan"}


@pytest.mark.parametrize(
  "make_cfg", [env_cfgs.tocabi_rough_env_cfg, env_cfgs.tocabi_flat_env_cfg]
)
def test_actor_history_follows_hojae_gomtang(make_cfg) -> None:
  """Proprioception keeps 40 steps; clock, command and last action keep one."""
  cfg = make_cfg()
  for name, term in cfg.observations["actor"].terms.items():
    expected = 0 if name in SINGLE_STEP_TERMS else env_cfgs.ACTOR_HISTORY_LENGTH
    assert term.history_length == expected, name
    assert term.history_stride == 1, name
  for name, term in cfg.observations["critic"].terms.items():
    assert term.history_length == 0, name


def test_flat_removes_terrain_terms() -> None:
  cfg = env_cfgs.tocabi_flat_env_cfg()
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "terrain_levels" not in cfg.curriculum
  assert all(s.name != "terrain_scan" for s in cfg.scene.sensors or ())


def test_encoder_bias_toggle() -> None:
  assert "encoder_bias" in env_cfgs.tocabi_flat_env_cfg().events
  assert "encoder_bias" not in env_cfgs.tocabi_flat_env_cfg(encoder_bias=False).events


def test_height_scan_in_actor_toggle() -> None:
  cfg = env_cfgs.tocabi_rough_env_cfg()
  assert "height_scan" in cfg.observations["actor"].terms
  blind = env_cfgs.tocabi_rough_env_cfg(height_scan_in_actor=False)
  assert "height_scan" not in blind.observations["actor"].terms
  assert "height_scan" in blind.observations["critic"].terms


def test_gain_tables_match_joint_lists() -> None:
  n = len(env_cfgs.LEG_JOINT_NAMES) + len(env_cfgs.UPPER_JOINT_NAMES)
  assert n == 33
  assert len(env_cfgs.P_GAINS) == len(env_cfgs.D_GAINS) == n
  assert len(env_cfgs.TORQUE_LIMITS) == n
