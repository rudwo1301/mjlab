"""DYROS TOCABI constants.

The MJCF is copied verbatim from ``dyros_tocabi_v2/tocabi_description``
(``mujoco_model/dyros_tocabi_ec2_small.xml``). Scene elements, XML actuators and
keyframes are stripped in :func:`get_spec`. Actuator and joint parameters follow
the HojaeGomtang TOCABI config rather than the XML joint classes.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

TOCABI_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "tocabi"
  / "xmls"
  / "mujoco_model"
  / "dyros_tocabi_ec2_small.xml"
)
assert TOCABI_XML.exists()

# The XML names the root body "base_link"; the URDF and IsaacLab use "Pelvis_Link".
ROOT_BODY_NAME = "Pelvis_Link"

# Viewers show geom groups 0-2 by default; collision geoms go to 3, as for G1.
COLLISION_GEOM_GROUP = 3


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TOCABI_XML))

  # Keep only the robot: terrain, lights and sim options come from the mjlab scene.
  for geom in list(spec.worldbody.geoms):
    spec.delete(geom)
  for light in list(spec.worldbody.lights):
    spec.delete(light)
  for texture in [
    t for t in spec.textures if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX
  ]:
    spec.delete(texture)
  # Actuators are defined in Python, initial state by EntityCfg.init_state.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for key in list(spec.keys):
    spec.delete(key)
  # The included simulation_scene.xml sets solver options for the standalone
  # simulator; mjlab ignores them on attach (MujocoCfg applies), so reset them to
  # avoid attach-conflict warnings.
  default = mujoco.MjSpec()
  spec.option = default.option
  spec.visual.map.znear = default.visual.map.znear
  spec.njmax = default.njmax
  spec.nconmax = default.nconmax

  spec.body("base_link").name = ROOT_BODY_NAME

  # Name collision geoms so CollisionCfg can address them, and move them to a group
  # the viewers hide by default (the XML puts them in group 2, next to the meshes).
  for body in spec.bodies:
    i = 0
    for geom in body.geoms:
      if geom.contype != 0 or geom.conaffinity != 0:
        geom.name = f"{body.name}_collision{i}"
        geom.group = COLLISION_GEOM_GROUP
        i += 1

  return spec


##
# Actuator config.
##

# Values follow the HojaeGomtang TOCABI config (LearningHumanoidArmMotion, tocabi
# branch, extensions/tocabi/assets), which has been validated sim-to-real.
# - Damping: HojaeGomtang uses explicit PD actuators with zero stiffness and
#   damping 1.0, i.e. a -1.0 * qd torque on every joint. Here this is passive
#   joint damping.
# - Friction: HojaeGomtang uses a load-dependent PhysX friction coefficient (0.3
#   on the legs), which MuJoCo cannot represent. Constant friction is zero and joint
#   friction is covered by domain randomization in the task instead.
# - Delay: legs and actively controlled arm joints (DelayedPDActuator in HojaeGomtang)
#   get 0-2 physics steps of command delay; locked joints have none.

TOCABI_VISCOUS_DAMPING = 1.0
TOCABI_DELAY_MAX_LAG = 2


def _motor(
  names: tuple[str, ...], effort_limit: float, armature: float, delayed: bool
) -> BuiltinMotorActuatorCfg:
  return BuiltinMotorActuatorCfg(
    target_names_expr=names,
    effort_limit=effort_limit,
    armature=armature,
    frictionloss=0.0,
    viscous_damping=TOCABI_VISCOUS_DAMPING,
    delay_min_lag=0,
    delay_max_lag=TOCABI_DELAY_MAX_LAG if delayed else 0,
  )


# Legs.
TOCABI_ACTUATOR_HIP_YAW = _motor((".*_HipYaw_Joint",), 333.0, 0.614, True)
TOCABI_ACTUATOR_HIP_ROLL = _motor((".*_HipRoll_Joint",), 232.0, 0.862, True)
TOCABI_ACTUATOR_HIP_PITCH = _motor((".*_HipPitch_Joint",), 263.0, 1.09, True)
TOCABI_ACTUATOR_KNEE = _motor((".*_Knee_Joint",), 289.0, 1.09, True)
TOCABI_ACTUATOR_ANKLE_PITCH = _motor((".*_AnklePitch_Joint",), 222.0, 1.09, True)
TOCABI_ACTUATOR_ANKLE_ROLL = _motor((".*_AnkleRoll_Joint",), 166.0, 0.360, True)

# Arms.
TOCABI_ACTUATOR_SHOULDER = _motor(
  (".*_Shoulder1_Joint", ".*_Shoulder2_Joint", ".*_Shoulder3_Joint"),
  64.0,
  0.18,
  True,
)
TOCABI_ACTUATOR_ELBOW = _motor((".*_Elbow_Joint",), 23.0, 0.0032, True)

# Locked arm joints.
TOCABI_ACTUATOR_ARMLINK = _motor((".*_Armlink_Joint",), 64.0, 0.18, False)
TOCABI_ACTUATOR_FOREARM = _motor((".*_Forearm_Joint",), 23.0, 0.0032, False)
TOCABI_ACTUATOR_WRIST = _motor(
  (".*_Wrist1_Joint", ".*_Wrist2_Joint"), 10.0, 0.0032, False
)

# Locked trunk and head.
TOCABI_ACTUATOR_WAIST = _motor(
  ("Waist1_Joint", "Waist2_Joint", "Upperbody_Joint"), 303.0, 0.078, False
)
TOCABI_ACTUATOR_HEAD = _motor(("Neck_Joint", "Head_Joint"), 10.0, 0.0032, False)

##
# Keyframe config.
##

# CoM height ~0.728 m.
COM728_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.93),
  joint_pos={
    ".*_HipYaw_Joint": 0.0,
    ".*_HipRoll_Joint": 0.0,
    ".*_HipPitch_Joint": -0.24,
    ".*_Knee_Joint": 0.6,
    ".*_AnklePitch_Joint": -0.36,
    ".*_AnkleRoll_Joint": 0.0,
    "Waist1_Joint": 0.0,
    "Waist2_Joint": 0.0,
    "Upperbody_Joint": 0.0,
    "L_Shoulder1_Joint": 0.3,
    "L_Shoulder2_Joint": 0.3,
    "L_Shoulder3_Joint": 1.5,
    "L_Armlink_Joint": -1.27,
    "L_Elbow_Joint": -1.0,
    "L_Forearm_Joint": 0.0,
    "L_Wrist1_Joint": -1.0,
    "L_Wrist2_Joint": 0.0,
    "Neck_Joint": 0.0,
    "Head_Joint": 0.0,
    "R_Shoulder1_Joint": -0.3,
    "R_Shoulder2_Joint": -0.3,
    "R_Shoulder3_Joint": -1.5,
    "R_Armlink_Joint": 1.27,
    "R_Elbow_Joint": 1.0,
    "R_Forearm_Joint": 0.0,
    "R_Wrist1_Joint": 1.0,
    "R_Wrist2_Joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# CoM height ~0.68 m (more crouched).
COM068_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.88),
  joint_pos={
    **COM728_KEYFRAME.joint_pos,  # type: ignore[dict-item]
    ".*_HipPitch_Joint": -0.46,
    ".*_Knee_Joint": 1.04,
    ".*_AnklePitch_Joint": -0.58,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_COLLISION_EXPR = r"^[LR]_Foot_Link_collision\d+$"

# All collisions including self collisions (IsaacLab enables self collisions).
# Feet get condim=3, everything else condim=1.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision\\d+",),
  contype=1,
  conaffinity=1,
  condim={FOOT_COLLISION_EXPR: 3, ".*_collision\\d+": 1},
  priority={FOOT_COLLISION_EXPR: 1, ".*": 0},
  friction={FOOT_COLLISION_EXPR: (1.0,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(FOOT_COLLISION_EXPR,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(1.0,),
)

##
# Final config.
##

TOCABI_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    TOCABI_ACTUATOR_HIP_YAW,
    TOCABI_ACTUATOR_HIP_ROLL,
    TOCABI_ACTUATOR_HIP_PITCH,
    TOCABI_ACTUATOR_KNEE,
    TOCABI_ACTUATOR_ANKLE_PITCH,
    TOCABI_ACTUATOR_ANKLE_ROLL,
    TOCABI_ACTUATOR_SHOULDER,
    TOCABI_ACTUATOR_ELBOW,
    TOCABI_ACTUATOR_ARMLINK,
    TOCABI_ACTUATOR_FOREARM,
    TOCABI_ACTUATOR_WRIST,
    TOCABI_ACTUATOR_WAIST,
    TOCABI_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=1.0,
)


def get_tocabi_robot_cfg(
  init_state: EntityCfg.InitialStateCfg = COM728_KEYFRAME,
) -> EntityCfg:
  """Get a fresh TOCABI robot configuration instance."""
  return EntityCfg(
    init_state=init_state,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=TOCABI_ARTICULATION,
    sort_actuators=True,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_tocabi_robot_cfg())

  viewer.launch(robot.spec.compile())
