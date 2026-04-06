import argparse
import json
import os
import sys
import time

import dill
import hydra
import numpy as np
import registry
import torch
import zarr
from loguru import logger
from numcodecs import Blosc
from zarr.storage import ZipStore

from utils import pose_to_transform_matrix
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.real_world.real_inference_util import (
    get_real_obs_dict,
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--task", type=str, choices=["kitchen", "dining-room", "living-room"], required=True
)
parser.add_argument(
    "--object_poses_path",
    type=str,
    default=None,
    help="Path to JSON file containing object poses to initialize objects in the simulation scene. "
         "Overwrites PRELOAD_OBJECTS positions and orientations from the registry.",
)
parser.add_argument(
    "--episode_index",
    type=int,
    default=0,
    help="Index of the episode to use from the object poses JSON file (default: 0).",
)
args = parser.parse_args()

from isaacsim import SimulationApp

config = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": ["isaacsim.robot_motion.motion_generation"],
}
logger.info("Initializing IsaacSim App Launcher.")
simulation_app = SimulationApp(config)

# Import enable_extension BEFORE using it (must come after SimulationApp)
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.robot_motion.motion_generation")

import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim, RigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)

import omni.ui as ui
import omni.kit.app
import omni.kit.viewport.utility as vp_util

from motion_plan import PickPlace
from scipy.spatial.transform import Slerp

# --- Constants ---
BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/franka_panda/franka_panda_arm_v2.usd"
FRANKA_PANDA_PRIM_PATH = "/World/Franka"
GOPRO_PRIM_PATH = "/World/Franka/panda/panda_link7/gopro_link"
ASSETS_DIR = "/workspace/voilab/assets/CADs"

# Lula IK config paths
LULA_ROBOT_DESCRIPTION_PATH = "/workspace/voilab/assets/lula/frank_umi_descriptor.yaml"
LULA_URDF_PATH = "/workspace/voilab/assets/franka_panda/franka_panda_umi-isaacsim.urdf"

# Task name to runner class mapping
TASK_RUNNER_MAP = {
    "kitchen": "diffusion_policy.env_runner.isaacsim_registry_runners.KitchenIsaacSimAppRunner",
    "dining-room": "diffusion_policy.env_runner.isaacsim_registry_runners.DiningRoomIsaacSimAppRunner",
    "living-room": "diffusion_policy.env_runner.isaacsim_registry_runners.LivingRoomIsaacSimAppRunner",
}


def setup_dual_viewports():
    """Setup dual viewports: main perspective view and GoPro camera view."""
    perspective_path = "/OmniverseKit_Persp"

    # Get main viewport window
    v1_window = ui.Workspace.get_window("Viewport")
    if not v1_window:
        print("Error: Main viewport window not found")
        return

    v1_api = vp_util.get_viewport_from_window_name("Viewport")
    if v1_api:
        v1_api.camera_path = perspective_path

    # Get or create secondary viewport window
    v2_window = ui.Workspace.get_window("Viewport 2")
    if not v2_window:
        v2_window = vp_util.create_viewport_window("Viewport 2")
        # Important: Wait for UI to register the new window
        omni.kit.app.get_app().update()  # Synchronous frame update

    v2_api = vp_util.get_viewport_from_window_name("Viewport 2")
    if v2_api:
        v2_api.camera_path = f"{GOPRO_PRIM_PATH}/Camera"

    # Ensure both windows exist before docking
    if v1_window and v2_window:
        # Wait for UI to stabilize before docking
        omni.kit.app.get_app().update()

        # Attempt docking with error handling
        try:
            v2_window.dock_in(v1_window, ui.DockPosition.RIGHT)
            print("Viewports docked: [Viewport (Persp)] | [Viewport 2 (Camera)]")
        except Exception as e:
            print(f"Docking failed: {str(e)}")
            # Alternative docking approach if direct docking fails
            try:
                # Try docking after another frame
                omni.kit.app.get_app().update()
                v2_window.dock_in(v1_window, ui.DockPosition.RIGHT)
                print("Viewports docked on second attempt")
            except Exception as e2:
                print(f"Second docking attempt failed: {str(e2)}")
    else:
        print("Error: Could not find one or both viewport windows for docking.")


import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R


# --- Helper classes and functions for motion planning ---

class RigidPrimManager:
    """Cache for RigidPrim objects to avoid re-initialization."""
    def __init__(self):
        self._cache = {}

    def clear(self):
        self._cache.clear()

    def get(self, prim_path):
        if prim_path not in self._cache:
            prim = RigidPrim(
                prim_path,
                name=prim_path.replace("/", "_")
            )
            prim.initialize()
            self._cache[prim_path] = prim
        return self._cache[prim_path]


def make_get_object_world_pose(prim_mgr):
    """Create a function to get object world pose given a prim manager."""
    def _get_object_world_pose(prim_path):
        prim = prim_mgr.get(prim_path)
        pos_batch, quat_wxyz_batch = prim.get_world_poses()
        pos = pos_batch[0]
        quat_wxyz = quat_wxyz_batch[0]

        quat_xyzw = np.array([
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
        ])
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = pos
        return T
    return _get_object_world_pose


def wxyz_to_xyzw(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def xyzw_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def plan_line_cartesian(
    p_start: np.ndarray,
    q_start_wxyz: np.ndarray,
    p_goal: np.ndarray,
    q_goal_wxyz: np.ndarray,
    step_m: float = 0.005,
):
    """Plan a linear Cartesian trajectory with orientation interpolation."""
    p_start = np.asarray(p_start, dtype=float)
    p_goal = np.asarray(p_goal, dtype=float)

    dist = np.linalg.norm(p_goal - p_start)
    n_steps = max(2, int(np.ceil(dist / step_m)))

    positions = np.linspace(p_start, p_goal, n_steps)

    q0_xyzw = wxyz_to_xyzw(np.asarray(q_start_wxyz, dtype=float))
    q1_xyzw = wxyz_to_xyzw(np.asarray(q_goal_wxyz, dtype=float))

    key_rots = R.from_quat([q0_xyzw, q1_xyzw])
    slerp = Slerp([0.0, 1.0], key_rots)
    interp_rots = slerp(np.linspace(0.0, 1.0, n_steps))
    quats_xyzw = interp_rots.as_quat()
    quats_wxyz = np.array([xyzw_to_wxyz(q) for q in quats_xyzw])

    return [np.concatenate([p, q_wxyz]) for p, q_wxyz in zip(positions, quats_wxyz)]


def step_world(world, render=True, sleep_dt=0.01):
    """Advance the simulation world by one step."""
    world.step(render=render)
    time.sleep(sleep_dt)


def load_object_poses_from_json(json_path: str, episode_index: int = 0) -> list:
    """
    Load object poses from a JSON file.

    Args:
        json_path: Path to the JSON file containing object poses.
        episode_index: Index of the episode to use from the JSON file.

    Returns:
        List of dicts with object_name, position (tvec), and quat_wxyz (from rvec).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if episode_index >= len(data):
        raise ValueError(
            f"Episode index {episode_index} out of range. "
            f"JSON file contains {len(data)} episodes."
        )

    episode = data[episode_index]
    objects = episode.get("objects", [])

    object_poses = []
    for obj in objects:
        object_name = obj.get("object_name", "")
        rvec = np.array(obj.get("rvec", [0, 0, 0]))
        tvec = np.array(obj.get("tvec", [0, 0, 0]))

        # Convert rotation vector to quaternion (wxyz format)
        rotation = R.from_rotvec(rvec)
        quat_xyzw = rotation.as_quat()  # scipy returns xyzw
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        object_poses.append({
            "object_name": object_name,
            "tvec": tvec,
            "quat_wxyz": quat_wxyz,
        })

    logger.info(f"Loaded {len(object_poses)} object poses from episode {episode_index}")
    return object_poses


def update_preload_objects_with_poses(
    preload_objects: list,
    object_poses: list,
    aruco_translation: np.ndarray,
    aruco_rotation_quat_xyzw: np.ndarray,
) -> list:
    """
    Update PRELOAD_OBJECTS with positions and orientations from object_poses.

    The object poses (tvec, rvec) are assumed to be in the ArUco tag frame.
    This function transforms them to world coordinates using the ArUco tag pose.

    Args:
        preload_objects: List of PRELOAD_OBJECTS dicts from registry.
        object_poses: List of object poses loaded from JSON.
        aruco_translation: ArUco tag position in world frame.
        aruco_rotation_quat_xyzw: ArUco tag orientation (xyzw) in world frame.

    Returns:
        Updated preload_objects list with new positions and orientations.
    """
    # Build transformation matrix from ArUco frame to world frame
    aruco_rot = R.from_quat(aruco_rotation_quat_xyzw)
    T_world_aruco = np.eye(4)
    T_world_aruco[:3, :3] = aruco_rot.as_matrix()
    T_world_aruco[:3, 3] = aruco_translation

    # Create a mapping from object_name to pose
    pose_map = {}
    for pose in object_poses:
        # Normalize object name (replace underscores with spaces for matching)
        name_variants = [
            pose["object_name"],
            pose["object_name"].replace("_", " "),
            pose["object_name"].replace(" ", "_"),
        ]
        for name in name_variants:
            pose_map[name.lower()] = pose

    updated_objects = []
    for obj in preload_objects:
        obj_copy = obj.copy()
        obj_name = obj.get("name", "").lower()
        obj_name_underscore = obj_name.replace(" ", "_")

        # Try to find matching pose
        matched_pose = pose_map.get(obj_name) or pose_map.get(obj_name_underscore)

        if matched_pose:
            # Transform tvec from ArUco frame to world frame
            tvec_aruco = matched_pose["tvec"]
            pos_aruco_homogeneous = np.array([*tvec_aruco, 1.0])
            pos_world_homogeneous = T_world_aruco @ pos_aruco_homogeneous
            pos_world = pos_world_homogeneous[:3]
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

            obj_copy["default_position"] = pos_world.tolist()
            obj_copy["quat_wxyz"] = quat_wxyz

            logger.info(
                f"Updated '{obj.get('name')}' pose: "
                f"position={pos_world}, quat_wxyz={quat_wxyz}"
            )
        else:
            logger.warning(
                f"No matching pose found for object '{obj.get('name')}' in JSON file"
            )

        updated_objects.append(obj_copy)

    return updated_objects


class UMISimEnv:
    def __init__(
        self,
        world,
        manipulator,
        camera,
        art_kine_solver,
        frequency: float = 10.0,  # policy frequency
        camera_obs_horizon: int = 2,
        robot_obs_horizon: int = 2,
        gripper_obs_horizon: int = 2,
        camera_down_sample_steps: int = 3,
        robot_down_sample_steps: int = 1,
        gripper_down_sample_steps: int = 1,
    ):
        self.world = world
        self.manipulator = manipulator
        self.camera = camera
        self.art_kine_solver = art_kine_solver

        self.frequency = frequency
        self.dt = 1.0 / frequency

        # Observation horizons
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps

        # Observation buffers (ring buffers for history)
        buffer_size = (
            max(
                camera_obs_horizon * camera_down_sample_steps,
                robot_obs_horizon * robot_down_sample_steps,
                gripper_obs_horizon * gripper_down_sample_steps,
            )
            + 10
        )  # extra margin

        self.camera_buffer = deque(maxlen=buffer_size)
        self.robot_buffer = deque(maxlen=buffer_size)
        self.gripper_buffer = deque(maxlen=buffer_size)

        # Gripper joint indices (adjust for your robot)
        self.gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

    def _get_eef_pose_axis_angle(self) -> np.ndarray:
        """Get EEF pose as [x, y, z, ax, ay, az] in robot base frame."""
        ee_pos, ee_T = self.art_kine_solver.compute_end_effector_pose()

        # Convert to base frame if needed (see your earlier code)
        base_pos, base_quat = self.manipulator.get_world_pose()
        T_base_world = pose_to_transform_matrix(base_pos, base_quat)
        T_world_base = np.linalg.inv(T_base_world)

        T_eef_world = np.eye(4)
        T_eef_world[:3, :3] = ee_T[:3, :3]
        T_eef_world[:3, 3] = ee_pos

        T_eef_base = T_world_base @ T_eef_world

        pos = T_eef_base[:3, 3]
        rot_matrix = T_eef_base[:3, :3]

        # Convert to axis-angle (UMI format)
        axis_angle = R.from_matrix(rot_matrix).as_rotvec()

        return np.concatenate([pos, axis_angle])

    def _get_gripper_width(self) -> float:
        """Get gripper width in meters."""
        joint_positions = self.manipulator.get_joint_positions()
        # For Franka: width = finger1_pos + finger2_pos
        # Adjust indices based on your articulation
        finger1_idx = self.manipulator.get_dof_index(self.gripper_joint_names[0])
        finger2_idx = self.manipulator.get_dof_index(self.gripper_joint_names[1])
        return joint_positions[finger1_idx] + joint_positions[finger2_idx]

    def _get_camera_rgb(self) -> np.ndarray:
        """Get camera RGB image [H, W, 3] uint8."""
        return self.camera.get_rgb()
        

    def step_accumulate(self):
        """Call this every sim step to accumulate observations."""
        current_time = self.world.current_time

        # Accumulate camera
        self.camera_buffer.append(
            {"timestamp": current_time, "color": self._get_camera_rgb()}
        )

        # Accumulate robot
        self.robot_buffer.append(
            {"timestamp": current_time, "eef_pose": self._get_eef_pose_axis_angle()}
        )

        # Accumulate gripper
        self.gripper_buffer.append(
            {"timestamp": current_time, "gripper_width": self._get_gripper_width()}
        )

    def get_obs(self) -> dict:
        """
        Get aligned observations matching UMI format.
        Call after sufficient history is accumulated.
        """
        # Convert buffers to arrays
        camera_timestamps = np.array([d["timestamp"] for d in self.camera_buffer])
        camera_colors = np.stack([d["color"] for d in self.camera_buffer])

        robot_timestamps = np.array([d["timestamp"] for d in self.robot_buffer])
        robot_poses = np.stack([d["eef_pose"] for d in self.robot_buffer])

        gripper_timestamps = np.array([d["timestamp"] for d in self.gripper_buffer])
        gripper_widths = np.array([d["gripper_width"] for d in self.gripper_buffer])

        # Reference timestamp (latest camera)
        last_timestamp = camera_timestamps[-1]

        # === Align camera obs ===
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1]
            * self.camera_down_sample_steps
            * self.dt
        )
        camera_idxs = [
            np.argmin(np.abs(camera_timestamps - t)) for t in camera_obs_timestamps
        ]
        camera_obs = {
            "camera0_rgb": camera_colors[camera_idxs]  # [obs_horizon, H, W, 3]
        }

        # === Align robot obs (interpolate) ===
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1]
            * self.robot_down_sample_steps
            * self.dt
        )
        robot_pose_interp = self._interpolate_poses(
            robot_timestamps, robot_poses, robot_obs_timestamps
        )
        robot_obs = {
            "robot0_eef_pos": robot_pose_interp[..., :3],
            "robot0_eef_rot_axis_angle": robot_pose_interp[..., 3:],
        }

        # === Align gripper obs (interpolate) ===
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1]
            * self.gripper_down_sample_steps
            * self.dt
        )
        gripper_interp = np.interp(
            gripper_obs_timestamps, gripper_timestamps, gripper_widths
        )
        gripper_obs = {"robot0_gripper_width": gripper_interp[..., None]}

        # Combine
        obs_data = {}
        obs_data.update(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data["timestamp"] = camera_obs_timestamps

        return obs_data

    def _interpolate_poses(
        self, timestamps: np.ndarray, poses: np.ndarray, query_times: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate poses (pos + axis-angle).
        For rotation, use SLERP via scipy.
        """
        from scipy.interpolate import interp1d
        from scipy.spatial.transform import Slerp

        # Interpolate position
        pos_interp = interp1d(
            timestamps, poses[:, :3], axis=0, fill_value="extrapolate"
        )
        interp_pos = pos_interp(query_times)

        # Interpolate rotation (SLERP)
        rotations = R.from_rotvec(poses[:, 3:])
        slerp = Slerp(timestamps, rotations)

        # Clamp query times to valid range for SLERP
        query_times_clamped = np.clip(query_times, timestamps[0], timestamps[-1])
        interp_rot = slerp(query_times_clamped).as_rotvec()

        return np.concatenate([interp_pos, interp_rot], axis=-1)


def get_end_effector_pos_quat_wxyz(manipulator, lula_solver, art_kine_solver):
    base_pos, base_quat = manipulator.get_world_pose()
    lula_solver.set_robot_base_pose(
        robot_position=base_pos, robot_orientation=base_quat
    )

    ee_pos, ee_T = art_kine_solver.compute_end_effector_pose()  # ee_T[:3,:3] rotation
    quat_xyzw = R.from_matrix(ee_T[:3, :3]).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return ee_pos.astype(np.float64), quat_wxyz.astype(np.float64)


def set_to_init_pose(manipulator, lula_solver, art_kine_solver, task_name):
    curr_pos, _ = get_end_effector_pos_quat_wxyz(
        manipulator, lula_solver, art_kine_solver
    )
    print(f"Setting to init pose: {task_name}")

    if task_name == "kitchen":
        INIT_EE_POS = curr_pos + np.array([-0.16, 0.0, 0.13])
        INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
    elif task_name == "dining-room":
        INIT_EE_POS = curr_pos + np.array([-0.16, 0.0, 0.13])
        INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
    elif task_name == "living-room":
        INIT_EE_POS = curr_pos + np.array([-0.1, 0.2, 0.20])
        INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
    else:
        raise RuntimeError(
            f"Unknown task, expected one of 'kitchen', 'dining-room', 'living-room', got {task_name}"
        )

    success = apply_ik_solution(
        manipulator,
        art_kine_solver,
        INIT_EE_POS,
        INIT_EE_QUAT_WXYZ,
    )


def load_preload_objects(world, registry_config, stage_utils):
    """Load objects specified in registry config's PRELOAD_OBJECTS at their default positions."""

    env_vars = registry_config.get("environment_vars", {})
    preload_objects = env_vars.get("PRELOAD_OBJECTS", [])

    object_prims = {}

    assert len(preload_objects) > 0, (
        f"Registry {registry_config.__name__} got empty list of PRELOAD_OBJECTS"
    )

    for entry in preload_objects:
        raw_name = entry.get("name", "unknown")
        asset_filename = entry.get("assets")
        prim_path = entry.get("prim_path")

        if not asset_filename or not prim_path:
            logger.info(f" Skipping invalid preload entry: {entry}")
            continue

        full_asset_path = os.path.join(ASSETS_DIR, asset_filename)
        if not os.path.exists(full_asset_path):
            logger.info(f" Asset not found: {full_asset_path}")
            continue

        try:
            stage_utils.add_reference_to_stage(
                usd_path=full_asset_path, prim_path=prim_path
            )

            default_position = entry.get("default_position")
            if default_position is not None:
                default_position = np.array(default_position, dtype=np.float64)

            orientation = np.array(entry.get("quat_wxyz", [1, 0, 0, 0]))

            obj_prim = SingleXFormPrim(
                prim_path=prim_path,
                name=raw_name.replace(" ", "_"),
                position=default_position,
                orientation=orientation,
            )
            world.scene.add(obj_prim)
            object_prims[raw_name] = obj_prim
            logger.info(
                f" Loaded {raw_name} at {prim_path}, position={default_position}"
            )

        except Exception as e:
            logger.error(f" Failed to load {raw_name}: {e}")
            raise e

    return object_prims


def init_environment(task_name, cfg, object_poses_path=None, episode_index=0):
    assert hasattr(cfg, "shape_meta"), (
        "Missing shape_meta attribute from cfg. Please check your checkpoint."
    )

    # Get registry config FIRST (before it's used for robot/camera setup)
    registry_class = registry.get_task_registry(task_name)
    if not registry_class.validate_environment():
        logger.warning(f"Registry validation failed for {task_name}")
    registry_config = registry_class.get_config()

    # Override PRELOAD_OBJECTS with poses from JSON file if provided
    if object_poses_path is not None:
        logger.info(f"Loading object poses from: {object_poses_path}")
        object_poses = load_object_poses_from_json(object_poses_path, episode_index)

        # Get ArUco tag pose from registry config for coordinate transformation
        aruco_pose = registry_config.get("aruco_tag_pose", {})
        aruco_translation = np.array(aruco_pose.get("translation", [0.0, 0.0, 0.0]))
        # Convert wxyz to xyzw for scipy
        aruco_quat_wxyz = aruco_pose.get("rotation_quat", (1.0, 0.0, 0.0, 0.0))
        aruco_quat_xyzw = np.array([
            aruco_quat_wxyz[1],  # x
            aruco_quat_wxyz[2],  # y
            aruco_quat_wxyz[3],  # z
            aruco_quat_wxyz[0],  # w
        ])

        # Update PRELOAD_OBJECTS with new poses
        preload_objects = registry_config.get("environment_vars", {}).get("PRELOAD_OBJECTS", [])
        updated_preload_objects = update_preload_objects_with_poses(
            preload_objects,
            object_poses,
            aruco_translation,
            aruco_quat_xyzw,
        )
        registry_config["environment_vars"]["PRELOAD_OBJECTS"] = updated_preload_objects

    stage_utils.open_stage(BASE_SCENE_FP)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # --- Setup robot ---
    robot = stage_utils.add_reference_to_stage(
        usd_path=FRANKA_PANDA_FP, prim_path=FRANKA_PANDA_PRIM_PATH
    )
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

    robot_xform = SingleXFormPrim(prim_path=FRANKA_PANDA_PRIM_PATH)

    gripper = ParallelGripper(
        end_effector_prim_path=f"{FRANKA_PANDA_PRIM_PATH}/panda/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.02, 0.02]),
        action_deltas=np.array([0.01, 0.01]),
    )

    manipulator = world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PANDA_PRIM_PATH,
            name="franka_panda",
            end_effector_prim_path=f"{FRANKA_PANDA_PRIM_PATH}/panda/panda_rightfinger",
            gripper=gripper,
        )
    )
    manipulator.gripper.set_default_state(manipulator.gripper.joint_opened_positions)

    # --- Set robot pose from registry config ---
    franka_pose = registry_config.get("franka_pose", {})
    franka_translation = np.array(franka_pose.get("translation", [0.0, 0.0, 0.0]))
    franka_rotation = np.array(franka_pose.get("rotation_quat", [1.0, 0.0, 0.0, 0.0]))
    robot_xform.set_local_pose(
        translation=franka_translation / stage_utils.get_stage_units(),
        orientation=franka_rotation,
    )

    # --- Set camera view from registry config ---
    camera_pose = registry_config.get("camera_pose", {})
    camera_translation = np.array(camera_pose.get("translation", [0.0, 0.0, 0.0]))
    set_camera_view(camera_translation, franka_translation)

    # --- Setup observation camera ---
    cam_meta = cfg.shape_meta.get("obs", {}).get("camera0_rgb", {})
    cam_shape = cam_meta.get("shape", (3, 224, 224))
    if len(cam_shape) >= 3:
        _, height, width = cam_shape[:3]
    else:
        height = width = 224

    camera = Camera(
        prim_path=f"{GOPRO_PRIM_PATH}/Camera",
        name="gopro_camera",
        resolution=(int(width), int(height)),
    )
    camera.initialize()
    world.reset()

    # --- Load preload objects from registry ---
    load_preload_objects(world, registry_config, stage_utils)
    return world, manipulator, camera, registry_config


def initialize_ik_solvers(manipulator) -> tuple:
    """Initialize IK solvers for the manipulator.

    Returns:
        tuple: (lula_solver, art_kine_solver)
    """
    logger.info("Initializing IK solvers...")

    lula_solver = LulaKinematicsSolver(
        robot_description_path=LULA_ROBOT_DESCRIPTION_PATH,
        urdf_path=LULA_URDF_PATH,
    )

    art_kine_solver = ArticulationKinematicsSolver(
        manipulator,
        kinematics_solver=lula_solver,
        end_effector_frame_name="umi_tcp",  # TODO: franka_panda_umi_v2.usd has updated the tcp
    )

    # Calibrate robot base pose
    robot_pos, robot_quat = manipulator.get_world_pose()  # w.r.t world's frame
    lula_solver.set_robot_base_pose(
        robot_position=robot_pos, robot_orientation=robot_quat
    )

    logger.info("IK solvers initialized.")
    return lula_solver, art_kine_solver


def apply_ik_solution(manipulator, art_kine_solver, target_pos, target_quat_wxyz):
    """
    Compute and apply IK solution for target pose.

    Args:
        manipulator: manipulator articulation object
        art_kine_solver: ArticulationKinematicsSolver instance
        target_pos: Target position (3,)
        target_quat_wxyz: Target orientation as quaternion WXYZ (4,)
        step_idx: Current step index (for logging)

    Returns:
        bool: True if IK succeeded
    """
    action, success = art_kine_solver.compute_inverse_kinematics(
        target_position=target_pos, target_orientation=target_quat_wxyz
    )

    if success:
        manipulator.set_joint_positions(action.joint_positions, np.arange(7))
        return True

    return False


def main():
    logger.info("Initializing inference Workspace.")
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    instance = hydra.utils.get_class(cfg._target_)

    # Create session directory for debug outputs
    session_dir = f"./data/eval_outputs/session_{int(time.time())}"
    debug_img_dir = os.path.join(session_dir, "DEBUG_IMGS")
    os.makedirs(debug_img_dir, exist_ok=True)
    logger.info(f"Debug images will be saved to: {debug_img_dir}")

    workspace: BaseWorkspace = instance(cfg, output_dir=session_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    assert "diffusion" in cfg.name, "Unsupported policy type. Missing 'diffusion' key"

    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device("cuda")
    policy.eval().to(device)

    # set inference params
    logger.info("Policy initialized successfully.")

    world, manipulator, camera, registry_config = init_environment(
        args.task, cfg, args.object_poses_path, args.episode_index
    )

    # Setup dual viewports (perspective + GoPro camera)
    setup_dual_viewports()

    # Initialize step counter for debug image saving
    inference_step_counter = 0

    # Initialize IK solvers
    lula_solver, art_kine_solver = initialize_ik_solvers(manipulator)

    # Initialize RigidPrimManager for getting object poses
    prim_mgr = RigidPrimManager()
    get_object_world_pose = make_get_object_world_pose(prim_mgr)

    # Create PickPlace instance for motion planning
    pickplace = PickPlace(
        get_end_effector_pose_fn=get_end_effector_pos_quat_wxyz,
        get_object_world_pose_fn=get_object_world_pose,
        apply_ik_solution_fn=apply_ik_solution,
        plan_line_cartesian_fn=plan_line_cartesian,
        world=world,
        task=args.task,
    )

    # Get motion planner from registry
    motion_planner = registry.get_motion_planner(
        args.task,
        registry_config,
        get_object_world_pose_fn=get_object_world_pose,
        pickplace=pickplace,
    )

    # Create environment
    env = UMISimEnv(world, manipulator, camera, art_kine_solver)

    obs_pose_rep = "relative"  # Policy trained with relative poses
    action_pose_repr = "relative"  # Actions are in relative frame
    tx_robot1_robot0 = None  # Single robot, no inter-robot transform

    set_to_init_pose(manipulator, lula_solver, art_kine_solver, args.task)
    # Warm up buffer (run sim steps before inference)
    for _ in range(50):
        world.step(render=True)
        env.step_accumulate()

    # Capture episode start pose for computing robot0_eef_rot_axis_angle_wrt_start
    # This is required by the policy's shape_meta
    initial_obs = env.get_obs()
    # episode_start_pose is a list of poses, one per robot
    # Each pose is [x, y, z, ax, ay, az] (position + axis-angle rotation)
    episode_start_pose = [
        np.concatenate(
            [
                initial_obs["robot0_eef_pos"][-1],  # Last observation in horizon
                initial_obs["robot0_eef_rot_axis_angle"][-1],
            ]
        )
    ]

    curr_pos, _ = get_end_effector_pos_quat_wxyz(
        manipulator, lula_solver, art_kine_solver
    )

    if args.task == "kitchen":
        INIT_EE_POS = curr_pos + np.array([-0.16, 0.0, 0.13])
        INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
    elif args.task == "dining-room":
        INIT_EE_POS = curr_pos + np.array([-0.16, 0.0, 0.13])
        INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
    elif args.task == "living-room":
        INIT_EE_POS = curr_pos + np.array([-0.1, 0.2, 0.20])
        INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
    else:
        raise RuntimeError(
            f"Unknown task, expected one of 'kitchen', 'dining-room', 'living-room', got {args.task}"
        )

    success = apply_ik_solution(
        manipulator,
        art_kine_solver,
        INIT_EE_POS,
        INIT_EE_QUAT_WXYZ,
    )

    # Let simulation settle
    for _ in range(100):
        world.step(render=True)
        time.sleep(1 / 60)

    # Move to "move_above" position without recording (like in generate_data.py)
    print("[Main] Moving to position above object...")
    while simulation_app.is_running():
        motion_planner.step(manipulator, lula_solver, art_kine_solver)
        step_world(world, render=True)
        env.step_accumulate()

        # Phase becomes "descend" when robot reaches move_above position
        if motion_planner.pickplace.phase == "descend":
            print("[Main] Reached position above object.")
            break

    # Recapture episode start pose after moving above object
    initial_obs = env.get_obs()
    episode_start_pose = [
        np.concatenate(
            [
                initial_obs["robot0_eef_pos"][-1],
                initial_obs["robot0_eef_rot_axis_angle"][-1],
            ]
        )
    ]

    # Wait for user to type "start" before beginning evaluation
    print("[Main] Ready to start evaluation. Type 'start' to begin...")
    while True:
        user_input = input().strip().lower()
        if user_input == "start":
            break
        print("[Main] Invalid input. Type 'start' to begin.")

    while simulation_app.is_running():
        world.step(render=True)
        env.step_accumulate()

        obs = env.get_obs()

        # Save debug image every 50 inference steps
        if inference_step_counter % 50 == 0:
            # Get the latest RGB image from observations
            rgb_img = obs["camera0_rgb"][-1]  # Last frame in horizon, shape [H, W, 3]
            img_path = os.path.join(
                debug_img_dir, f"step_{inference_step_counter:06d}.png"
            )
            # Save using PIL
            from PIL import Image

            Image.fromarray(rgb_img).save(img_path)
            logger.info(f"Saved debug image: {img_path}")

        inference_step_counter += 1

        with torch.no_grad():
            s = time.time()
            obs_dict_np = get_real_umi_obs_dict(
                env_obs=obs,
                shape_meta=cfg.task.shape_meta,
                obs_pose_repr=obs_pose_rep,
                tx_robot1_robot0=tx_robot1_robot0,
                episode_start_pose=episode_start_pose,
            )
            obs_dict = dict_apply(
                obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
            )

            result = policy.predict_action(obs_dict)
            raw_action = result["action_pred"][0].detach().to("cpu").numpy()
            action = get_real_umi_action(raw_action, obs, action_pose_repr)
            logger.info(f"Inference latency: {time.time() - s:.3f}s")

        # Execute actions
        # action shape: (horizon, 7) = [x, y, z, rx, ry, rz, gripper_width]
        # Actions are in robot base frame, need to transform to world frame for IK
        n_action_steps = min(cfg.n_action_steps, len(action))
        base_pos, base_quat = manipulator.get_world_pose()
        T_base_world = pose_to_transform_matrix(base_pos, base_quat)

        for step_idx in range(n_action_steps):
            action_step = action[step_idx]

            # Extract pose and gripper from action (in robot base frame)
            target_pos_base = action_step[:3]
            target_rot_axis_angle = action_step[3:6]
            target_gripper_width = action_step[6]

            # Convert axis-angle to rotation matrix
            target_rot_matrix_base = R.from_rotvec(target_rot_axis_angle).as_matrix()

            # Build 4x4 transform matrix for target pose in base frame
            T_target_base = np.eye(4)
            T_target_base[:3, :3] = target_rot_matrix_base
            T_target_base[:3, 3] = target_pos_base

            # Transform from base frame to world frame: T_target_world = T_base_world @ T_target_base
            T_target_world = T_base_world @ T_target_base

            # Extract world-frame position and rotation
            target_pos_world = T_target_world[:3, 3]
            target_rot_matrix_world = T_target_world[:3, :3]

            # Convert rotation matrix to quaternion (wxyz) for IK
            target_quat_xyzw = R.from_matrix(target_rot_matrix_world).as_quat()
            target_quat_wxyz = np.array(
                [
                    target_quat_xyzw[3],  # w
                    target_quat_xyzw[0],  # x
                    target_quat_xyzw[1],  # y
                    target_quat_xyzw[2],  # z
                ]
            )


            # Apply IK to move robot arm (expects world frame)
            success = apply_ik_solution(
                manipulator, art_kine_solver, target_pos_world, target_quat_wxyz
            )

            if not success:
                logger.warning(f"IK failed at step {step_idx}")
                continue

            # Set gripper position
            # Franka gripper: width = finger1 + finger2, so each finger = width/2
            finger_pos = target_gripper_width / 2.0
            finger_positions = np.array([finger_pos, finger_pos])
            gripper_joint_indices = np.array(
                [
                    manipulator.get_dof_index("panda_finger_joint1"),
                    manipulator.get_dof_index("panda_finger_joint2"),
                ]
            )
            manipulator.set_joint_positions(finger_positions, gripper_joint_indices)

            # Step simulation
            world.step(render=True)
            env.step_accumulate()


if __name__ == "__main__":
    main()
    simulation_app.close()
