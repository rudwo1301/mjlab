"""Tests for tocabi_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.tocabi import tocabi_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def tocabi_entity() -> Entity:
  return Entity(tocabi_constants.get_tocabi_robot_cfg())


@pytest.fixture(scope="module")
def tocabi_model(tocabi_entity: Entity) -> mujoco.MjModel:
  return tocabi_entity.spec.compile()


def test_robot_only_spec(tocabi_model) -> None:
  """Scene geoms, XML actuators and XML keyframes are stripped."""
  assert tocabi_model.nu == 33
  assert tocabi_model.nkey == 1
  assert tocabi_model.body(1).name == tocabi_constants.ROOT_BODY_NAME
  world_geoms = tocabi_model.geom_bodyid == 0
  assert not world_geoms.any()


def test_total_mass(tocabi_model) -> None:
  np.testing.assert_allclose(tocabi_model.body_subtreemass[1], 96.32, atol=1e-2)


@pytest.mark.parametrize(
  "actuator_cfg",
  tocabi_constants.TOCABI_ARTICULATION.actuators,
)
def test_actuator_parameters(tocabi_model, actuator_cfg) -> None:
  matched = 0
  for i in range(tocabi_model.nu):
    joint = tocabi_model.joint(tocabi_model.actuator_trnid[i, 0])
    if not any(re.match(p, joint.name) for p in actuator_cfg.target_names_expr):
      continue
    matched += 1
    dof = joint.dofadr[0]
    assert tocabi_model.actuator_forcerange[i, 1] == actuator_cfg.effort_limit
    assert tocabi_model.dof_armature[dof] == actuator_cfg.armature
    assert tocabi_model.dof_frictionloss[dof] == actuator_cfg.frictionloss
    assert tocabi_model.dof_damping[dof] == actuator_cfg.viscous_damping
  assert matched > 0


def test_command_delay_groups(tocabi_entity) -> None:
  """Legs and active arm joints are delayed, locked joints are not.

  Follows HojaeGomtang.
  """
  delayed_expr = r".*_(HipYaw|HipRoll|HipPitch|Knee|AnklePitch|AnkleRoll"
  delayed_expr += r"|Shoulder1|Shoulder2|Shoulder3|Elbow)_Joint$"
  for actuator_cfg in tocabi_constants.TOCABI_ARTICULATION.actuators:
    for name in tocabi_entity.joint_names:
      if any(re.match(p, name) for p in actuator_cfg.target_names_expr):
        expected = tocabi_constants.TOCABI_DELAY_MAX_LAG
        expected = expected if re.match(delayed_expr, name) else 0
        assert actuator_cfg.delay_max_lag == expected, name


def test_keyframe_joint_positions(tocabi_entity, tocabi_model) -> None:
  key = tocabi_model.key("init_state")
  expected = tocabi_constants.COM728_KEYFRAME
  assert expected.joint_pos is not None
  np.testing.assert_array_equal(key.qpos[:3], expected.pos)
  values = resolve_expr(expected.joint_pos, tocabi_entity.joint_names, 0.0)
  for joint_name, value in zip(tocabi_entity.joint_names, values, strict=True):
    qpos_idx = tocabi_model.joint(joint_name).qposadr[0]
    np.testing.assert_allclose(key.qpos[qpos_idx], value, rtol=1e-5)


def test_collision_geoms_hidden_by_default(tocabi_model) -> None:
  """Collision geoms sit in group 3, which viewers hide by default."""
  for i in range(tocabi_model.ngeom):
    collides = tocabi_model.geom_contype[i] or tocabi_model.geom_conaffinity[i]
    if collides:
      assert tocabi_model.geom_group[i] == tocabi_constants.COLLISION_GEOM_GROUP
    else:
      assert tocabi_model.geom_group[i] < 3


def test_feet_above_ground_at_init(tocabi_model) -> None:
  data = mujoco.MjData(tocabi_model)
  mujoco.mj_resetDataKeyframe(tocabi_model, data, 0)
  mujoco.mj_forward(tocabi_model, data)
  assert data.ncon == 0
  for side in "LR":
    geom = tocabi_model.geom(f"{side}_Foot_Link_collision0")
    sole_z = data.geom_xpos[geom.id][2] - geom.size[2]
    assert 0.0 < sole_z < 0.02
