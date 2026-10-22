"""Runner for the HojaeGomtang TOCABI tasks."""

import re

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .hojae_gomtang_ppo import HojaeGomtangPPO
from .mirror import build_mirror_maps


class HojaeGomtangOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity runner that wires the mirror maps into :class:`HojaeGomtangPPO`."""

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    alg = self.alg
    if isinstance(alg, HojaeGomtangPPO) and alg.mirror_enabled:
      cfg = alg.mirror_cfg
      obs_idx, obs_sign, act_idx, act_sign, joints = build_mirror_maps(
        self.env.unwrapped,
        joint_sign_flip=tuple(cfg.get("joint_sign_flip", ())),
        shift_phase=cfg.get("shift_phase", False),
      )
      weights = cfg.get("action_weights", {})
      act_weight = [
        next((w for p, w in weights.items() if re.fullmatch(p, j)), 1.0) for j in joints
      ]
      alg.set_mirror(obs_idx, obs_sign, act_idx, act_sign, act_weight)
