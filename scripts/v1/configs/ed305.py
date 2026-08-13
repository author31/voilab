"""
EnvConfig instances for the project ED305 scene.

Folds together what used to live in ``base_config`` (scene/robot/camera/IK
constants), the ``registry/`` task classes (ArUco pose, robot/camera pose,
success predicate) and the ``motion_plan`` per-task planners (pick/place offsets).
The robot/IK/camera spec is identical across the three tasks, so it is built once
by ``_ed305_manipulator()``; only scene-localization, object set, success
predicate and motion plan differ per task.
"""

import numpy as np

from config import (
    SceneCfg, ManipulatorCfg, GripperCfg, IKCfg, CameraCfg, PoseCfg,
    PreloadObject, ArucoSpawnCfg, TaskCfg, MotionPlanCfg, EnvConfig,
    GRASP_QUAT_WXYZ,
)
from . import _tasklib as T


# ----------------------------------------------------------------------
# Shared ED305 constants (from base_config)
# ----------------------------------------------------------------------
BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/franka_panda/franka_panda_arm_v2.usd"
FRANKA_PANDA_PRIM_PATH = "/World/Franka"
GOPRO_PRIM_PATH = "/World/Franka/panda/panda_link7/gopro_link"
CAMERA_PRIM_PATH = f"{GOPRO_PRIM_PATH}/Camera"
END_EFFECTOR_PRIM_PATH = FRANKA_PANDA_PRIM_PATH + "/panda/panda_rightfinger"

LULA_ROBOT_DESCRIPTION_PATH = "/workspace/voilab/assets/lula/frank_umi_descriptor.yaml"
LULA_URDF_PATH = "/workspace/voilab/assets/franka_panda/franka_panda_umi-isaacsim.urdf"
END_EFFECTOR_FRAME_NAME = "umi_tcp"

# Per-object grasp orientation presets (motion-planning placement mode).
OBJECT_ORIENTATION_PRESETS = {
    "fork": [0.707, 0.0, 0.0, 0.707],
    "knife": [0.707, 0.0, 0.0, -0.707],
    "blue_block": [0.707107, 0.707107, 0, 0],
    "red_block": [0.0677732, -0.7038514, -0.0677732, -0.7038514],
    "green_block": [0.5, 0.5, 0.5, -0.5],
}

OBJECT_MAXIMUM_Z_HEIGHT = 1.1


def _ed305_scene():
    return SceneCfg(usd_path=BASE_SCENE_FP)


def _ed305_manipulator():
    return ManipulatorCfg(
        usd_path=FRANKA_PANDA_FP,
        prim_path=FRANKA_PANDA_PRIM_PATH,
        name="my_franka",
        end_effector_prim_path=END_EFFECTOR_PRIM_PATH,
        gripper=GripperCfg(
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.00, 0.00]),
            action_deltas=np.array([0.005, 0.005]),
        ),
        ik=IKCfg(
            kind="files",
            end_effector_frame_name=END_EFFECTOR_FRAME_NAME,
            robot_description_path=LULA_ROBOT_DESCRIPTION_PATH,
            urdf_path=LULA_URDF_PATH,
        ),
        camera=CameraCfg(prim_path=CAMERA_PRIM_PATH),
        variant_selections={"Gripper": "AlternateFinger", "Mesh": "Quality"},
    )


# ----------------------------------------------------------------------
# Kitchen — stack the blue cup onto the pink cup
# ----------------------------------------------------------------------
def kitchen():
    pink = "/World/pink_cup"
    blue = "/World/blue_cup"
    return EnvConfig(
        name="ed305_kitchen",
        scene=_ed305_scene(),
        manipulator=_ed305_manipulator(),
        robot_pose=PoseCfg(
            translation=np.array([4.5, 2.7, 0.9000000134110451]),
            rotation_quat_wxyz=T.euler_deg_to_wxyz([0.0, 0.0, 0.0]),
        ),
        view_camera_translation=np.array([7.5, 2.68664950400609, 2.2]),
        object_spawn=ArucoSpawnCfg(
            preload_objects=[
                PreloadObject("pink cup", "cup_pink.usd", pink, np.array([1.0, 0.0, 0.0, 0.0])),
                PreloadObject("blue cup", "cup_blue.usd", blue, np.array([1.0, 0.0, 0.0, 0.0])),
            ],
            aruco_tag_pose={
                "translation": np.array([4.9652, 2.45, 0.9]),
                "rotation_quat": T.euler_deg_to_wxyz([0.0, 0.0, 180.0]),
            },
            object_maximum_z_height=OBJECT_MAXIMUM_Z_HEIGHT,
            object_pos_offset=np.array([-0.05, 0.0, 0.0]),
            orientation_presets=OBJECT_ORIENTATION_PRESETS,
        ),
        task=TaskCfg(
            pickplace_task_key="kitchen",
            init_ee_offset=np.array([-0.16, 0.0, 0.13]),
            init_ee_quat_wxyz=GRASP_QUAT_WXYZ,
            success_predicate=T.make_kitchen_predicate(blue, pink),
        ),
        motion_plan=MotionPlanCfg(steps=T.kitchen_motion_steps(blue, pink)),
    )


# ----------------------------------------------------------------------
# Dining room — fork left / knife right of the plate
# ----------------------------------------------------------------------
def dining_room():
    fork = "/World/fork"
    knife = "/World/knife"
    plate = "/plate"
    return EnvConfig(
        name="ed305_dining_room",
        scene=_ed305_scene(),
        manipulator=_ed305_manipulator(),
        robot_pose=PoseCfg(
            translation=np.array([1.4471314866267897, 4.953638444125494, 0.7547650876392805]),
            rotation_quat_wxyz=T.euler_deg_to_wxyz([0.0, 0.0, -18.0]),
        ),
        view_camera_translation=np.array([5.300000078976154, 4.90000007301569, 1.600000023841858]),
        object_spawn=ArucoSpawnCfg(
            preload_objects=[
                PreloadObject("knife", "knife.usd", knife, np.array([1.0, 0.0, 0.0, 0.0])),
                PreloadObject("fork", "fork.usd", fork, np.array([0.707, 0.0, 0.0, -0.707])),
            ],
            aruco_tag_pose={
                "translation": np.array([1.65, 4.75, 0.8]),
                "rotation_quat": T.euler_deg_to_wxyz([0.0, 0.0, 90.0]),
            },
            object_maximum_z_height=OBJECT_MAXIMUM_Z_HEIGHT,
            object_pos_offset=np.array([0.0, 0.0, -0.25]),
            orientation_presets=OBJECT_ORIENTATION_PRESETS,
        ),
        task=TaskCfg(
            pickplace_task_key="dining-room",
            init_ee_offset=np.array([-0.16, 0.0, 0.13]),
            init_ee_quat_wxyz=GRASP_QUAT_WXYZ,
            success_predicate=T.make_dining_predicate(plate, fork, knife),
        ),
        motion_plan=MotionPlanCfg(steps=T.dining_motion_steps(fork, knife, plate)),
    )


# ----------------------------------------------------------------------
# Living room — drop the three blocks into the storage box
# ----------------------------------------------------------------------
def living_room():
    blue_block = "/World/cylinder"
    green_block = "/World/bridge"
    red_block = "/World/triangle"
    storage_box = "/storage_box"
    return EnvConfig(
        name="ed305_living_room",
        scene=_ed305_scene(),
        manipulator=_ed305_manipulator(),
        robot_pose=PoseCfg(
            translation=np.array([1.045, 11.31, 0.50]),
            rotation_quat_wxyz=T.euler_deg_to_wxyz([0.0, 0.0, -30.0]),
        ),
        view_camera_translation=np.array([2.66, 11.41, 1.96]),
        object_spawn=ArucoSpawnCfg(
            preload_objects=[
                PreloadObject("blue_block", "cylinder.usd", blue_block, np.array([0.707107, 0.707107, 0, 0])),
                PreloadObject("green_block", "bridge.usd", green_block, np.array([0.5, 0.5, 0.5, -0.5])),
                PreloadObject("red_block", "triangle.usd", red_block,
                              np.array([0.0677732, -0.7038514, -0.0677732, -0.7038514])),
            ],
            aruco_tag_pose={
                "translation": np.array([1.18, 11.31, 0.83]),
                "rotation_quat": T.euler_deg_to_wxyz([0.0, 0.0, 90.0]),
            },
            object_maximum_z_height=OBJECT_MAXIMUM_Z_HEIGHT,
            object_pos_offset=np.array([0.0, 0.0, -0.25]),
            orientation_presets=OBJECT_ORIENTATION_PRESETS,
        ),
        task=TaskCfg(
            pickplace_task_key="living-room",
            init_ee_offset=np.array([-0.1, 0.2, 0.20]),
            init_ee_quat_wxyz=GRASP_QUAT_WXYZ,
            success_predicate=T.make_living_predicate(blue_block, green_block, red_block),
        ),
        motion_plan=MotionPlanCfg(
            steps=T.living_motion_steps(red_block, blue_block, green_block, storage_box)
        ),
    )
