"""Model loading, name discovery and scene helpers for the evidence room."""

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = ROOT / "models" / "panda" / "evidence_intake.xml"  # Phase 1: sealed evidence bags
PHASE0B_SCENE_XML = ROOT / "models" / "panda" / "evidence_room.xml"  # kept unchanged; runs with code at tag phase0b
OUT_DIR = ROOT / "out"

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATORS = [f"actuator{i}" for i in range(1, 8)]
GRIPPER_ACTUATOR = "actuator8"
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
FINGER_BODIES = ["left_finger", "right_finger"]
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0

ITEM_CLASSES = ["phone", "blade", "garment", "carton"]  # content classes of the sealed evidence bag
SLOT_NAMES = [f"slot_{i}" for i in range(4)]
COUNTER_TOP_Z = 0.0

# Flange-to-fingertip-pad offset in the hand frame (Franka's standard TCP offset).
TCP_OFFSET = 0.1034
WRIST_CAM_X = 0.075
WRIST_CAM_ABOVE_TCP = 0.110
WRIST_CAM_FOVY = 60.0
WRIST_CAMS = ("wrist_cam_px", "wrist_cam_nx")
WRIST_RES = (1280, 960)
SCANNER_CAM = "scanner_cam"
SCANNER_RES = (4000, 3260)
PAD_FRICTION = [1.5, 0.05, 0.0001]
# Gripper servo stiffness/damping on the `split` tendon (N/m, N*s/m). Menagerie ships kp=100, which squeezes only
# ~1.4 N per pad and cannot lift 0.5 kg; the real Franka Hand is rated 70 N continuous. kp=1500 gives ~21 N per pad
# on a 56 mm item (see NOTES.md, M2). The ctrl mapping (0 = closed, 255 = 0.04 m open) is unchanged.
GRIPPER_KP = 1500.0
GRIPPER_KV = 120.0

_OBJ = mujoco.mjtObj


def build_spec(path=SCENE_XML):
    spec = mujoco.MjSpec.from_file(str(path))
    spec.body("hand").add_site(
        name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.006, 0, 0], rgba=[1, 0.1, 0.1, 0.8], group=4
    )
    # Verify-scan wrist cameras (lower resolution than the intake scanner). The label sits at one end of the bag
    # and either grasp yaw (180 deg apart) may be chosen, so there is one camera beyond each x-edge of the hand
    # body (which spans +-32 mm), centred over where labels sit (label centres are 73-89 mm from the grasp centre),
    # 110 mm above the TCP, looking straight down the hand's +z (the finger direction). At 52 mm / 83 mm the labels
    # were cut off at the frame edge (verify 8/12); see NOTES.md, Phase 1 Task B.
    for name, x in (("wrist_cam_px", WRIST_CAM_X), ("wrist_cam_nx", -WRIST_CAM_X)):
        spec.body("hand").add_camera(name=name, pos=[x, 0, TCP_OFFSET - WRIST_CAM_ABOVE_TCP], quat=[0, 1, 0, 0],
                                     fovy=WRIST_CAM_FOVY)
    # The real Franka compensates gravity in its joint controller; the Menagerie position servos do not, and sag
    # ~7 mm at the TCP under their own weight. Model the robot's internal compensation on the arm bodies only
    # (never on items, which must be carried by contact forces).
    for body_name in [f"link{i}" for i in range(1, 8)] + ["hand"] + FINGER_BODIES:
        spec.body(body_name).gravcomp = 1.0
    grip = spec.actuator(GRIPPER_ACTUATOR)
    gain, bias = list(grip.gainprm), list(grip.biasprm)
    gain[0] = GRIPPER_KP * 0.04 / GRIPPER_OPEN
    bias[1], bias[2] = -GRIPPER_KP, -GRIPPER_KV
    grip.gainprm, grip.biasprm = gain, bias
    # Bag bodies get a randomised centre-of-mass offset at runtime (body_ipos). MuJoCo compiles bodies whose inertial
    # frame equals the body frame as "simple"/"sameframe" and then refuses (or ignores) a moved CoM, so opt out here;
    # randomise.py also sets body_sameframe to 0 so kinematics computes the true CoM position.
    for cls in ITEM_CLASSES:
        spec.body(item_body(cls)).simple = False
    for body_name in FINGER_BODIES:
        for geom in spec.body(body_name).geoms:
            if geom.contype or geom.conaffinity:
                geom.friction = PAD_FRICTION
    return spec


def load(path=SCENE_XML):
    """Compile the scene and return (model, data) at the `home` keyframe."""
    model = build_spec(path).compile()
    # The `home` keyframe comes from panda.xml and only covers the arm; MuJoCo zero-pads the rest,
    # which would teleport every item to the world origin. Fill the item part from qpos0 instead.
    key = key_id(model, "home")
    arm_nq = len(ARM_JOINTS) + len(FINGER_JOINTS)
    model.key_qpos[key, arm_nq:] = model.qpos0[arm_nq:]
    data = mujoco.MjData(model)
    reset_home(model, data)
    return model, data


def reset_home(model, data):
    mujoco.mj_resetDataKeyframe(model, data, key_id(model, "home"))
    mujoco.mj_forward(model, data)


def _id(model, obj, name):
    i = mujoco.mj_name2id(model, obj, name)
    if i < 0:
        raise KeyError(f"no {obj.name} named {name!r} in model")
    return i


def body_id(model, name):
    return _id(model, _OBJ.mjOBJ_BODY, name)


def site_id(model, name):
    return _id(model, _OBJ.mjOBJ_SITE, name)


def geom_id(model, name):
    return _id(model, _OBJ.mjOBJ_GEOM, name)


def actuator_id(model, name):
    return _id(model, _OBJ.mjOBJ_ACTUATOR, name)


def joint_id(model, name):
    return _id(model, _OBJ.mjOBJ_JOINT, name)


def key_id(model, name):
    return _id(model, _OBJ.mjOBJ_KEY, name)


def camera_id(model, name):
    return _id(model, _OBJ.mjOBJ_CAMERA, name)


def arm_qpos_adr(model):
    return np.array([model.jnt_qposadr[joint_id(model, j)] for j in ARM_JOINTS])


def arm_dof_adr(model):
    return np.array([model.jnt_dofadr[joint_id(model, j)] for j in ARM_JOINTS])


def arm_ranges(model):
    return np.array([model.jnt_range[joint_id(model, j)] for j in ARM_JOINTS])


def item_body(cls):
    return f"item_{cls}"


def item_geom(cls):
    return f"item_{cls}_geom"


def item_joint(cls):
    return f"item_{cls}_joint"


def item_label(cls):
    return f"item_{cls}_label"


def body_geoms(model, bid):
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]


def slot_volume(model, data, slot):
    """World-frame centre and half-extents of a slot's usable volume (box site, axis-aligned)."""
    sid = site_id(model, slot)
    return data.site_xpos[sid].copy(), model.site_size[sid].copy()


def discover(model):
    """Print every actuator, joint range, named body, site, camera and keyframe actually in the model."""
    name = lambda obj, i: mujoco.mj_id2name(model, obj, i)
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom} "
          f"nsite={model.nsite} timestep={model.opt.timestep}")
    print("-- actuators")
    for i in range(model.nu):
        print(f"  [{i}] {name(_OBJ.mjOBJ_ACTUATOR, i):12s} ctrlrange={model.actuator_ctrlrange[i]} "
              f"forcerange={model.actuator_forcerange[i]}")
    print("-- joints")
    for i in range(model.njnt):
        limited = "range=" + str(model.jnt_range[i]) if model.jnt_limited[i] else "unlimited"
        print(f"  [{i}] {name(_OBJ.mjOBJ_JOINT, i):20s} type={model.jnt_type[i]} qadr={model.jnt_qposadr[i]} {limited}")
    print("-- bodies")
    for i in range(model.nbody):
        print(f"  [{i}] {name(_OBJ.mjOBJ_BODY, i)}")
    print("-- sites")
    for i in range(model.nsite):
        print(f"  [{i}] {name(_OBJ.mjOBJ_SITE, i)} size={model.site_size[i]}")
    print("-- cameras")
    for i in range(model.ncam):
        print(f"  [{i}] {name(_OBJ.mjOBJ_CAMERA, i)}")
    print("-- keyframes")
    for i in range(model.nkey):
        print(f"  [{i}] {name(_OBJ.mjOBJ_KEY, i)}")


def render(model, data, camera="demo_cam", width=1280, height=720, renderer=None):
    own = renderer is None
    if own:
        renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera)
    img = renderer.render()
    if own:
        renderer.close()
    return img


def save_png(img, path):
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path


if __name__ == "__main__":
    m, _ = load()
    discover(m)
