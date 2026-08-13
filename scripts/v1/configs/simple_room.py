"""
EnvConfig instance for Isaac Sim's built-in ``simple_room.usd`` + Franka FR3.

This is the simple_room analog of the ED305 configs. It differs in three ways
that map onto the config axes:
  * scene:        stock simple_room.usd (default prim /Root renamed to /World).
  * manipulator:  Isaac's built-in FR3 (fr3_* joints, supported-Lula IK config,
                  fr3_hand_tcp frame, a wrist camera created under fr3_hand).
  * object spawn: table-anchored spawning + per-episode jitter (no ArUco/JSON).

NOTE: the scene + robot USDs are resolved from the Isaac assets root, so this
factory MUST be called after ``SimulationApp`` has been created.
The kitchen task (stack blue cup onto pink cup) reuses ED305's kitchen success
predicate and scripted motion steps — the cup prim paths are identical.
"""

import numpy as np

from config import (
    SceneCfg, PrimPose, ManipulatorCfg, GripperCfg, IKCfg, CameraCfg, PoseCfg,
    PreloadObject, TableSpawnCfg, TaskCfg, MotionPlanCfg, EnvConfig,
    GRASP_QUAT_WXYZ,
)
from . import _tasklib as T


SIMPLE_ROOM_RELATIVE_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"
FR3_RELATIVE_PATH = "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd"

# Project CAD assets for the cups (mounted at /workspace/voilab in the Isaac
# container -> host assets/CADs/). Set explicitly so this config does not rely on
# the schema's DEFAULT_ASSETS_DIR.
CAD_ASSETS_DIR = "/workspace/voilab/assets/CADs"
PINK_CUP_ASSET = "cup_pink.usd"
BLUE_CUP_ASSET = "cup_blue.usd"

FRANKA_PRIM_PATH = "/World/Franka"
HAND_PRIM_PATH = FRANKA_PRIM_PATH + "/fr3_hand"
CAMERA_PRIM_PATH = f"{HAND_PRIM_PATH}/gopro_camera"
END_EFFECTOR_PRIM_PATH = FRANKA_PRIM_PATH + "/fr3_rightfinger"
END_EFFECTOR_FRAME_NAME = "fr3_hand_tcp"
FR3_LULA_ROBOT_NAME = "FR3"

# Wrist-cam local pose relative to fr3_hand (sensible default; calibrate to match
# the gopro view a given checkpoint was trained on). Pose is expressed in ROS
# camera convention (+Z forward, +Y down) to match the IsaacLab
# TiledCameraCfg.OffsetCfg(pos=(0.04, 0.0, 0.0), rot=(0.707, 0, 0, 0.707),
# convention="ros") the policy was trained with.
WRIST_CAMERA_TRANSLATION = np.array([0.04, 0.0, 0.0])
WRIST_CAMERA_ORIENTATION_WXYZ = np.array([0.707, 0.0, 0.0, 0.707])
WRIST_CAMERA_CONVENTION = "ros"

# Stock-table layout (world-frame, tabletop coordinates).
# The table_low asset is ~0.5 m tall; placed at the origin its top surface sits at
# world z≈0.5, so the FR3 stands on the tabletop at z=TABLE_TOP_Z and objects rest
# at TABLE_TOP_Z + their origin offset.
TABLE_TOP_Z = 0.8
# Stock table prim + the pose we force it to (translation + Z-up yaw in degrees).
TABLE_PRIM_PATH = "/World/table_low_327"
TABLE_TRANSLATION = np.array([0.0, 0.0, 0.0])
TABLE_ROTATION_EULER_DEG = np.array([0.0, 0.0, 90.0])
# The 90° yaw swaps the table's long/short axes in world frame, so the X/Y spawn
# half-extents are swapped relative to the un-rotated table.
TABLE_HALF_EXTENT_XY = np.array([0.7, 1.4])
SPAWN_JITTER_XY = np.array([0.05, 0.05])


def _assets_root():
    # Imported lazily: valid only after SimulationApp has been created.
    from isaacsim.storage.native import get_assets_root_path
    root = get_assets_root_path()
    assert root is not None, (
        "Could not resolve the Isaac Sim assets root for simple_room.usd / fr3.usd. "
        "Check the Nucleus/local assets configuration."
    )
    return root


def kitchen():
    root = _assets_root()
    pink = "/World/pink_cup"
    blue = "/World/blue_cup"
    return EnvConfig(
        name="simple_room_kitchen",
        scene=SceneCfg(
            usd_path=root + SIMPLE_ROOM_RELATIVE_PATH,
            rename_default_prim_to="World",
            # Move the stock table to the origin with a 90° yaw.
            prim_pose_overrides=[
                PrimPose(
                    prim_path=TABLE_PRIM_PATH,
                    translation=TABLE_TRANSLATION,
                    rotation_quat_wxyz=T.euler_deg_to_wxyz(TABLE_ROTATION_EULER_DEG),
                    name_fallback="table_low_327",
                ),
            ],
        ),
        manipulator=ManipulatorCfg(
            usd_path=root + FR3_RELATIVE_PATH,
            prim_path=FRANKA_PRIM_PATH,
            name="my_franka",
            end_effector_prim_path=END_EFFECTOR_PRIM_PATH,
            gripper=GripperCfg(
                joint_prim_names=["fr3_finger_joint1", "fr3_finger_joint2"],
                joint_opened_positions=np.array([0.04, 0.04]),
                joint_closed_positions=np.array([0.00, 0.00]),
                action_deltas=np.array([0.005, 0.005]),
            ),
            ik=IKCfg(
                kind="supported",
                end_effector_frame_name=END_EFFECTOR_FRAME_NAME,
                robot_name=FR3_LULA_ROBOT_NAME,
            ),
            camera=CameraCfg(
                prim_path=CAMERA_PRIM_PATH,
                translation=WRIST_CAMERA_TRANSLATION,
                orientation_wxyz=WRIST_CAMERA_ORIENTATION_WXYZ,
                convention=WRIST_CAMERA_CONVENTION,
            ),
            variant_selections={},
        ),
        robot_pose=PoseCfg(
            translation=np.array([-0.31423972563714314, 0.0, TABLE_TOP_Z]),
            rotation_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        ),
        view_camera_translation=np.array([1.6, 0.0, 0.9]),
        object_spawn=TableSpawnCfg(
            # Both cups sit on the tabletop directly IN FRONT of the FR3 (+X),
            # within reach; the cup origin is ~0.12 m above its base, so
            # z = TABLE_TOP_Z + 0.12 drops the base onto the tabletop.
            preload_objects=[
                PreloadObject("pink cup", PINK_CUP_ASSET, pink,
                              np.array([1.0, 0.0, 0.0, 0.0]),
                              table_position=np.array([0.05, -0.12, TABLE_TOP_Z + 0.12])),
                PreloadObject("blue cup", BLUE_CUP_ASSET, blue,
                              np.array([1.0, 0.0, 0.0, 0.0]),
                              table_position=np.array([0.05, 0.12, TABLE_TOP_Z + 0.12])),
            ],
            table_half_extent_xy=TABLE_HALF_EXTENT_XY,
            spawn_jitter_xy=SPAWN_JITTER_XY,
            assets_dir=CAD_ASSETS_DIR,
        ),
        task=TaskCfg(
            pickplace_task_key="kitchen",
            init_ee_offset=np.array([-0.16, 0.0, 0.13]),
            init_ee_quat_wxyz=GRASP_QUAT_WXYZ,
            success_predicate=T.make_kitchen_predicate(blue, pink),
        ),
        motion_plan=MotionPlanCfg(steps=T.kitchen_motion_steps(blue, pink)),
    )
