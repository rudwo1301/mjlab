"""Left/right mirror maps for the HojaeGomtang leg policy.

Builds index and sign vectors such that ``obs[:, idx] * sign`` is the observation of
the left/right mirrored state, and likewise for actions. Joint terms swap ``L_`` and
``R_`` joints. Base terms follow the original mapping: angular velocity
(-x, y, -z), gravity and linear velocity (x, -y, z), command (vx, -vy, -wz) and
heading -1.

Two choices of the original are kept as defaults and can be changed:
- Joint values are swapped without a sign flip (``joint_sign_flip=()``). A physical
  mirror about the sagittal plane would also negate hip yaw, hip roll and ankle roll.
- The gait phase is left unchanged (``shift_phase=False``). Swapping legs would
  otherwise correspond to a half-cycle shift, which negates phase sin and cos.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from mjlab.tasks.tocabi.mdp.actions import LowerBodyAction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_BASE_TERM_SIGNS = {
  "base_ang_vel": (-1.0, 1.0, -1.0),
  "projected_gravity": (1.0, -1.0, 1.0),
  "base_lin_vel": (1.0, -1.0, 1.0),
  "velocity_commands": (1.0, -1.0, -1.0),
  "base_heading": (-1.0,),
  "base_height": (1.0,),
}
_JOINT_TERMS = ("joint_pos", "joint_vel")
_PHASE_TERMS = ("phase_sin", "phase_cos")


def _swap_side(name: str) -> str:
  if name.startswith("L_"):
    return "R_" + name[2:]
  if name.startswith("R_"):
    return "L_" + name[2:]
  return name


def joint_mirror_map(
  joint_names: list[str], joint_sign_flip: tuple[str, ...] = ()
) -> tuple[list[int], list[float]]:
  """Index and sign per joint so that ``x[idx] * sign`` mirrors joint values."""
  index = {n: i for i, n in enumerate(joint_names)}
  idx = [index.get(_swap_side(n), i) for i, n in enumerate(joint_names)]
  sign = [
    -1.0 if any(re.fullmatch(p, n) for p in joint_sign_flip) else 1.0
    for n in joint_names
  ]
  return idx, sign


def build_mirror_maps(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  group: str = "actor",
  joint_sign_flip: tuple[str, ...] = (),
  shift_phase: bool = False,
) -> tuple[list[int], list[float], list[int], list[float], list[str]]:
  """Return (obs_idx, obs_sign, action_idx, action_sign, action_joint_names)."""
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, LowerBodyAction)
  robot = env.scene["robot"]
  action_joints = [robot.joint_names[i] for i in term.joint_ids[: term.action_dim]]
  act_idx, act_sign = joint_mirror_map(action_joints, joint_sign_flip)

  om = env.observation_manager
  obs_idx: list[int] = []
  obs_sign: list[float] = []
  offset = 0

  def add_blocks(count: int, local_idx: list[int], local_sign: list[float], name):
    block = len(local_idx)
    if block == 0 or count % block != 0:
      raise ValueError(
        f"Mirror: term '{name}' of size {count} is not a multiple of {block}."
      )
    for b in range(count // block):
      base = offset + b * block
      obs_idx.extend(base + j for j in local_idx)
      obs_sign.extend(local_sign)

  for name, dims, cfg in zip(
    om._group_obs_term_names[group],
    om._group_obs_term_dim[group],
    om._group_obs_term_cfgs[group],
    strict=True,
  ):
    count = int(np.prod(dims))
    if name in _BASE_TERM_SIGNS:
      signs = list(_BASE_TERM_SIGNS[name])
      add_blocks(count, list(range(len(signs))), signs, name)
    elif name in _PHASE_TERMS:
      add_blocks(count, [0], [-1.0 if shift_phase else 1.0], name)
    elif name in _JOINT_TERMS:
      joints = list(cfg.params["asset_cfg"].joint_names)
      idx, sign = joint_mirror_map(joints, joint_sign_flip)
      add_blocks(count, idx, sign, name)
    elif name == "last_leg_action":
      add_blocks(count, act_idx, act_sign, name)
    else:
      raise ValueError(f"Mirror: no mapping for observation term '{name}'.")
    offset += count
  return obs_idx, obs_sign, act_idx, act_sign, action_joints
