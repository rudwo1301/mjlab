"""Tests for the TOCABI MDP terms."""

import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots.tocabi import tocabi_constants
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tocabi import mdp
from mjlab.terrains import TerrainEntityCfg

LOWER = tuple(
  f"{s}_{j}_Joint"
  for s in "LR"
  for j in ("HipYaw", "HipRoll", "HipPitch", "Knee", "AnklePitch", "AnkleRoll")
)
FEET = SceneEntityCfg(
  "robot", body_names=("L_AnkleRoll_Link", "R_AnkleRoll_Link"), preserve_order=True
)


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _make_env_cfg(**action_kwargs) -> ManagerBasedRlEnvCfg:
  robot = tocabi_constants.get_tocabi_robot_cfg()
  upper = tuple(n for n in _joint_names() if n not in LOWER)
  n = len(LOWER) + len(upper)
  feet_contact = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(L_AnkleRoll_Link|R_AnkleRoll_Link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=3,
  )
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.LowerBodyActionCfg(
      entity_name="robot",
      lower_joint_names=LOWER,
      upper_joint_names=upper,
      p_gains=(400.0,) * n,
      d_gains=(10.0,) * n,
      torque_limits=(100.0,) * n,
      **action_kwargs,
    )
  }
  phase = {"command_name": "phase_time"}
  rewards = {
    "lin_vel": RewardTermCfg(
      func=mdp.track_lin_vel_xy_base_exp,
      weight=1.0,
      params={"command_name": "base_velocity", "omega": 5.0},
    ),
    "ang_vel": RewardTermCfg(
      func=mdp.track_ang_vel_z_world_exp,
      weight=1.0,
      params={"command_name": "base_velocity", "omega": 7.0},
    ),
    "orientation": RewardTermCfg(
      func=mdp.orientation_tracking, weight=1.0, params={"omega": 5.0}
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height_tracking,
      weight=0.5,
      params={"height": 0.928, "omega": 10.0},
    ),
    "periodic_force": RewardTermCfg(
      func=mdp.periodic_force,
      weight=1.0,
      params={**phase, "sensor_name": "feet_contact", "scale": 1300.0},
    ),
    "feet_height": RewardTermCfg(
      func=mdp.feet_height_tracking,
      weight=1.0,
      params={
        **phase,
        "omega": 5.0,
        "foot_height": 0.1,
        "kappa": 2.0,
        "offset": -0.77,
        "asset_cfg": FEET,
      },
    ),
    "feet_vel_z": RewardTermCfg(
      func=mdp.feet_velocity_z_tracking,
      weight=0.5,
      params={
        **phase,
        "omega": 3.0,
        "foot_height": 0.1,
        "kappa": 2.0,
        "asset_cfg": FEET,
      },
    ),
    "large_contact": RewardTermCfg(
      func=mdp.large_contact_force,
      weight=-0.005,
      params={"sensor_name": "feet_contact", "threshold": 100.0},
    ),
    "default_joint": RewardTermCfg(
      func=mdp.default_joint_pos,
      weight=0.2,
      params={"omega": 2.0, "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER)},
    ),
    "feet_flat": RewardTermCfg(
      func=mdp.feet_flat, weight=-0.1, params={"asset_cfg": FEET}
    ),
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=2,
      extent=2.0,
      entities={"robot": robot},
      sensors=(feet_contact,),
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "clock": ObservationTermCfg(func=mdp.clock_input, params=phase),
          "action": ObservationTermCfg(
            func=mdp.lower_body_action, params={"action_name": "joint_pos"}
          ),
        },
      ),
    },
    actions=actions,
    commands={
      "phase_time": mdp.WalkingPhaseCommandCfg(
        resampling_time_range=(10.0, 10.0), phase_time_range=(1.2, 1.2)
      ),
      "base_velocity": mdp.UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(5.0, 5.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(-0.8, 0.8), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.8, 0.8)
        ),
      ),
    },
    rewards=rewards,
    terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
    decimation=4,
    episode_length_s=10.0,
  )


def _joint_names() -> tuple[str, ...]:
  return Entity(tocabi_constants.get_tocabi_robot_cfg()).joint_names


def test_env_steps_with_finite_rewards(device):
  env = ManagerBasedRlEnv(cfg=_make_env_cfg(), device=device)
  env.reset()
  action = torch.zeros(2, len(LOWER), device=device)
  for _ in range(9):
    env.step(action)
  obs, reward, *_ = env.step(action)
  assert torch.isfinite(reward).all()
  for name in env.reward_manager.active_terms:
    assert torch.isfinite(env.reward_manager.get_term(name)).all(), name
  actor_obs = obs["actor"]
  assert isinstance(actor_obs, torch.Tensor)
  assert actor_obs.shape == (2, 2 + len(LOWER))
  env.close()


def test_torque_mode_maps_action_to_torque_limits(device):
  env = ManagerBasedRlEnv(cfg=_make_env_cfg(), device=device)
  env.reset()
  action = torch.full((2, len(LOWER)), 0.5, device=device)
  action[:, 0] = 3.0  # Clipped to 1.0 by action_clip.
  env.step(action)
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, mdp.LowerBodyAction)
  lower_effort = term.applied_effort[:, : len(LOWER)]
  assert torch.allclose(lower_effort[:, 0], torch.full((2,), 100.0, device=device))
  assert torch.allclose(lower_effort[:, 1:], torch.full_like(lower_effort[:, 1:], 50.0))
  env.close()


def test_efforts_clamped_to_torque_limits(device):
  env = ManagerBasedRlEnv(
    cfg=_make_env_cfg(
      action_clip=None, enable_torque_bias=True, torque_bias_tick=(500.0, 0.0)
    ),
    device=device,
  )
  env.reset()
  env.step(torch.zeros(2, len(LOWER), device=device))
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, mdp.LowerBodyAction)
  assert term.applied_effort.abs().max() <= 100.0 + 1e-4
  lower_effort = term.applied_effort[:, : len(LOWER)]
  assert torch.allclose(lower_effort, torch.full_like(lower_effort, 100.0))
  env.close()


def test_left_swing_is_half_cycle_after_right(device):
  env = ManagerBasedRlEnv(cfg=_make_env_cfg(), device=device)
  env.reset()
  phase_time = 1.2
  half = round(phase_time / 2 / env.step_dt)
  angles = []
  for step in range(2 * half):
    env.episode_length_buf[:] = step
    angles.append(mdp.rewards._swing_angle(env, "phase_time")[0])
  angles = torch.stack(angles)
  # Left at t equals right at t - T/2 (mod 2*pi).
  left, right = angles[half:, 0], angles[:half, 1]
  assert torch.allclose(torch.cos(left), torch.cos(right), atol=1e-4)
  assert torch.allclose(torch.sin(left), torch.sin(right), atol=1e-4)
  # The right foot peaks (angle 0) a quarter cycle in.
  assert torch.cos(angles[half // 2, 1]).item() == pytest.approx(1.0, abs=1e-4)
  env.close()
