"""Tests for the HojaeGomtang TOCABI task."""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tocabi.config.hojae_gomtang_env_cfgs import (
  LEG_JOINT_POS_LIMITS,
  tocabi_hojae_gomtang_flat_env_cfg,
)
from mjlab.tasks.tocabi.mdp import LowerBodyAction
from mjlab.tasks.tocabi.mdp import hojae_gomtang as hg
from mjlab.tasks.tocabi.rl.mirror import build_mirror_maps


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def env(device):
  cfg = tocabi_hojae_gomtang_flat_env_cfg(encoder_bias=False)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  yield env
  env.close()


def _default_pose_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Action whose absolute PD targets equal the default leg pose."""
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, LowerBodyAction)
  limits = torch.tensor(LEG_JOINT_POS_LIMITS, device=env.device)
  robot = env.scene["robot"]
  q0 = robot.data.default_joint_pos[:, term.joint_ids[: term.action_dim]]
  return 2 * (q0 - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1


def test_observation_and_action_sizes(env) -> None:
  obs = env.observation_manager.compute()
  # 40 steps of (ang vel 3, gravity 3, joint pos 12, joint vel 12), plus command 3,
  # phase sin/cos 2 and last action 12.
  assert obs["actor"].shape == (4, 40 * 30 + 17)
  assert obs["critic"].shape == (4, 52)
  assert env.action_manager.total_action_dim == 12


def test_default_pose_action_holds_pose(env) -> None:
  """At reset, targets equal the current pose, so leg PD effort is ~0."""
  env.reset()
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, LowerBodyAction)
  term.process_actions(_default_pose_action(env))
  term.apply_actions()
  assert term.pd_effort[:, : term.action_dim].abs().max() < 1.0


def test_ground_reaction_force_is_positive_when_standing(env) -> None:
  env.reset()
  action = _default_pose_action(env)
  for _ in range(5):
    env.step(action)
  grf = hg._feet_force_z(env.scene["feet_contact"])
  assert (grf.sum(dim=1) > 100.0).all()


def test_step_phase_command(env) -> None:
  term = env.command_manager.get_term("step_phase")
  assert isinstance(term, hg.StepPhaseCommand)
  period = term.full_step_period[:, 0]
  assert ((period >= 70) & (period <= 90)).all()
  assert torch.all(
    (term.phase_offset[:, 0] == 0) | (term.phase_offset[:, 0] == period // 2)
  )
  phase = term.phase
  assert ((phase >= 0.0) & (phase < 1.0)).all()


def test_startup_holds_phase_and_ramps_clock(env) -> None:
  term = env.command_manager.get_term("step_phase")
  assert isinstance(term, hg.StepPhaseCommand)
  term.cfg.startup_steps = 4
  try:
    start = term.phase_offset[:, 0] / term.full_step_period[:, 0]
    for step, scale in ((0, 0.0), (2, 0.5), (4, 1.0)):
      env.episode_length_buf[:] = step
      torch.testing.assert_close(term.phase, start.float())
      assert torch.allclose(term.phase_scale, torch.full_like(term.phase_scale, scale))
      clock = hg.phase_cos(env, "step_phase")[:, 0]
      expected = scale * torch.cos(2 * torch.pi * start.float())
      torch.testing.assert_close(clock, expected)
    assert not term.in_startup.any()
    env.episode_length_buf[:] = 5
    assert (term.phase > start).all()
    # Phase 0 swings the left foot first, so the right foot carries the weight.
    assert torch.equal(term.first_stance_foot, (term.phase_offset[:, 0] == 0).long())
  finally:
    term.cfg.startup_steps = 0


def test_linear_hip_roll_penalty_charges_only_the_excess(env) -> None:
  robot = env.scene["robot"]
  ids, _ = robot.find_joints(
    ("L_HipRoll_Joint", "R_HipRoll_Joint"), preserve_order=True
  )
  asset_cfg = SceneEntityCfg("robot", joint_ids=ids)
  pos = torch.tensor([[-0.08, 0.0]], device=env.device).expand(env.num_envs, -1)
  robot.write_joint_position_to_sim(pos, joint_ids=torch.tensor(ids, device=env.device))
  linear = hg.hip_roll_threshold_penalty(env, asset_cfg, linear=True)
  step = hg.hip_roll_threshold_penalty(env, asset_cfg)
  torch.testing.assert_close(linear, torch.full_like(linear, -0.02))
  torch.testing.assert_close(step, torch.full_like(step, -1.02))
  env.reset()


def test_env_steps_with_finite_rewards(env) -> None:
  env.reset()
  action = _default_pose_action(env)
  for _ in range(9):
    env.step(action)
  _, reward, *_ = env.step(action)
  assert torch.isfinite(reward).all()
  for name in env.reward_manager.active_terms:
    assert torch.isfinite(env.reward_manager.get_term(name)).all(), name


def test_mirror_maps_are_involutions(env) -> None:
  obs_idx, obs_sign, act_idx, act_sign, joints = build_mirror_maps(env)
  obs_idx_t, obs_sign_t = torch.tensor(obs_idx), torch.tensor(obs_sign)
  assert len(obs_idx) == env.observation_manager.group_obs_dim["actor"][0]
  # Mirroring twice is the identity.
  assert torch.equal(obs_idx_t[obs_idx_t], torch.arange(len(obs_idx)))
  assert torch.equal(obs_sign_t * obs_sign_t[obs_idx_t], torch.ones(len(obs_idx)))
  # Left and right leg joints swap.
  for i, name in enumerate(joints):
    other = joints[act_idx[i]]
    assert other == name.replace("L_", "R_", 1) or other == name.replace("R_", "L_", 1)
  assert act_sign == [1.0] * len(joints)
  # Projected gravity (first history step starts at 120) flips y.
  assert obs_sign[120:123] == [1.0, -1.0, 1.0]


def test_scheduled_push_applies_impulse_in_window(env) -> None:
  env.reset()
  term_cfg = env.event_manager.get_term_cfg("scheduled_push")
  term = term_cfg.func
  assert isinstance(term, hg.scheduled_push)
  env.common_step_counter = 24 * 20000  # Past the curriculum: full strength.
  robot = env.scene["robot"]
  pelvis = term._body_ids[0]
  center, h = term._center[0, 0].item(), term._halfspan[0, 0].item()
  for step in (center - h - 1, center):
    env.episode_length_buf[:] = step
    term(env, None, **term_cfg.params)
    force_w = robot.data.body_external_wrench[0, pelvis, :3]
    if step < center - h:
      assert torch.allclose(force_w, torch.zeros(3, device=env.device))
    else:
      expected = term._impulse[0, 0].norm() / ((2 * h + 1) * env.step_dt)
      torch.testing.assert_close(force_w.norm(), expected, rtol=1e-4, atol=1e-3)
