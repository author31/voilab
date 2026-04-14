import cv2
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
from umi_replay import set_gripper_width


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
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R


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


class RigidPrimManager:
    """Cache for RigidPrim objects to avoid re-initialization."""

    def __init__(self):
        self._cache = {}

    def clear(self):
        self._cache.clear()

    def get(self, prim_path):
        if prim_path not in self._cache:
            prim = RigidPrim(prim_path, name=prim_path.replace("/", "_"))
            prim.initialize()
            self._cache[prim_path] = prim
        return self._cache[prim_path]


class PolicyWrapper:
    def __init__(self, checkpoint_path, device="cuda:0"):
        print(f"[Policy] Loading checkpoint: {checkpoint_path}")
        self.device = torch.device(device)

        # Load checkpoint payload using dill
        payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
        self.cfg = payload["cfg"]

        # Instantiate Policy
        self.policy = hydra.utils.instantiate(self.cfg.policy)

        # Load State Dict
        if "state_dicts" in payload:
            self.policy.load_state_dict(payload["state_dicts"]["model"])
        else:
            self.policy.load_state_dict(payload["state_dict"])

        self.policy.to(self.device)
        self.policy.eval()

        # Get Obs config
        self.obs_horizon = self.cfg.shape_meta.obs.camera0_rgb.horizon
        print(f"[Policy] Observation Horizon: {self.obs_horizon}")

        # Target image shape
        target_shape = self.cfg.shape_meta.obs.camera0_rgb.shape
        self.target_c = target_shape[0]
        self.target_h = target_shape[1]
        self.target_w = target_shape[2]
        print(
            f"[Policy] Target Image Shape: {self.target_w}x{self.target_h} (Channels: {self.target_c})"
        )

        # Save as absolute coord buffer
        self.obs_buffer_abs = deque(maxlen=self.obs_horizon)

    def reset(self):
        self.obs_buffer_abs.clear()

    def get_action(self, obs_dict):
        """
        Input observation at current step (absolute coords), store in buffer,
        transform to relative coords for model inference, then convert predicted
        actions back to absolute coords.

        Argsh:
            obs_dict: {
                'camera0_rgb': (H, W, 3) uint8                      # RGB image
                'robot0_eef_pos': (3,) float32                      # EE position (absolute)
                'robot0_eef_rot_axis_angle': (3,) float32           # EE rotation axis-angle (absolute)
                'robot0_eef_rot_axis_angle_wrt_start': (6,) float32 # EE rotation 6D relative to init pose (absolute)
                'robot0_gripper_width': (1,) float32                # Gripper width
            }
        Returns:
            actions_abs: list of dict {'pos', 'rot_mat', 'gripper'} in absolute coords
        """

        # 1. Process Image
        img = obs_dict["camera0_rgb"]
        img = cv2.resize(
            img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR
        )
        img = img.astype(np.float32)
        img = np.moveaxis(img, -1, 0) / 255.0  # (C, H, W), [0,1]

        # 2. Save Curr observation to buffer (absolute coords)
        current_frame_abs = {
            "camera0_rgb": img,
            "robot0_eef_pos": obs_dict["robot0_eef_pos"].astype(np.float32).copy(),
            "robot0_eef_rot_axis_angle": obs_dict["robot0_eef_rot_axis_angle"]
            .astype(np.float32)
            .copy(),
            "robot0_eef_rot_axis_angle_wrt_start": obs_dict[
                "robot0_eef_rot_axis_angle_wrt_start"
            ]
            .astype(np.float32)
            .copy(),
            "robot0_gripper_width": obs_dict["robot0_gripper_width"]
            .astype(np.float32)
            .copy(),
        }

        self.obs_buffer_abs.append(current_frame_abs)

        # Padding if buffer not full
        while len(self.obs_buffer_abs) < self.obs_horizon:
            self.obs_buffer_abs.appendleft(self.obs_buffer_abs[0])

        # 3. Use buffer[-1] as refernce, tranform to related coord
        ref_frame = self.obs_buffer_abs[-1]
        ref_pos = ref_frame["robot0_eef_pos"]
        ref_rot_aa = ref_frame["robot0_eef_rot_axis_angle"]
        ref_pose6 = np.concatenate([ref_pos, ref_rot_aa])
        ref_mat = pose_to_mat(ref_pose6)  # (4, 4)
        ref_mat_inv = np.linalg.inv(ref_mat)

        # Transform all frames to relative coordinate
        batch_obs_rel = {
            "camera0_rgb": [],
            "robot0_eef_pos": [],
            "robot0_eef_rot_axis_angle": [],
            "robot0_eef_rot_axis_angle_wrt_start": [],
            "robot0_gripper_width": [],
        }

        for frame in self.obs_buffer_abs:
            batch_obs_rel["camera0_rgb"].append(frame["camera0_rgb"])

            # Convert to relative pose
            frame_pos = frame["robot0_eef_pos"]
            frame_rot_aa = frame["robot0_eef_rot_axis_angle"]
            frame_pose6 = np.concatenate([frame_pos, frame_rot_aa])
            frame_mat = pose_to_mat(frame_pose6)

            # T_rel = inv(T_ref) @ T_frame
            rel_mat = ref_mat_inv @ frame_mat
            rel_pose10d = mat_to_pose10d(rel_mat)  # (10,) = pos(3) + rot6d(6)

            batch_obs_rel["robot0_eef_pos"].append(rel_pose10d[:3])
            batch_obs_rel["robot0_eef_rot_axis_angle"].append(rel_pose10d[3:9])  # 6D

            # wrt_start
            batch_obs_rel["robot0_eef_rot_axis_angle_wrt_start"].append(
                frame["robot0_eef_rot_axis_angle_wrt_start"]
            )
            batch_obs_rel["robot0_gripper_width"].append(frame["robot0_gripper_width"])

        # 4. Stack to Tensor
        batch_obs_tensor = {}
        for key, val_list in batch_obs_rel.items():
            val_stack = np.stack(val_list, axis=0)  # (T, ...)
            val_tensor = (
                torch.from_numpy(val_stack).unsqueeze(0).to(self.device)
            )  # (1, T, ...)
            batch_obs_tensor[key] = val_tensor

        # 5. Inference
        with torch.no_grad():
            result = self.policy.predict_action(batch_obs_tensor)
            action_tensor = result["action_pred"]  # (1, T_action, 10)
            actions_rel = action_tensor[0].cpu().numpy()  # (T_action, 10)
            # print(f"[Policy] Predicted {actions_rel.shape} actions: {actions_rel}")

        # 6. Transferm predicted actions back to absolute coord
        # [pos(3), rot6d(6), gripper(1)], Action is related to ref_mat (from buffer[-1])
        actions_abs = []
        for i in range(len(actions_rel)):
            action_rel = actions_rel[i]
            action_pos_rel = action_rel[:3]
            action_rot6d_rel = action_rel[3:9]
            action_gripper = action_rel[9]

            # Construct action reference matrix (4x4)
            action_rot_mat_rel = rot6d_to_matrix(action_rot6d_rel)
            action_mat_rel = np.eye(4)
            action_mat_rel[:3, :3] = action_rot_mat_rel
            action_mat_rel[:3, 3] = action_pos_rel

            # Transform to absolute coordinate: T_abs = T_ref @ T_rel
            action_mat_abs = ref_mat @ action_mat_rel

            actions_abs.append(
                {
                    "pos": action_mat_abs[:3, 3].copy(),
                    "rot_mat": action_mat_abs[:3, :3].copy(),
                    "gripper": float(action_gripper),
                }
            )

        return actions_abs, ref_mat, actions_rel


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


def get_end_effector_pos_quat_wxyz(manipulator, lula_solver, art_kine_solver):
    base_pos, base_quat = manipulator.get_world_pose()
    lula_solver.set_robot_base_pose(
        robot_position=base_pos, robot_orientation=base_quat
    )

    ee_pos, ee_T = art_kine_solver.compute_end_effector_pose()  # ee_T[:3,:3] rotation
    quat_xyzw = R.from_matrix(ee_T[:3, :3]).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return ee_pos.astype(np.float64), quat_wxyz.astype(np.float64)


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

        object_poses.append(
            {
                "object_name": object_name,
                "tvec": tvec,
                "quat_wxyz": quat_wxyz,
            }
        )

    logger.info(f"Loaded {len(object_poses)} object poses from episode {episode_index}")
    return object_poses


def make_get_object_world_pose(prim_mgr):
    """Create a function to get object world pose given a prim manager."""

    def _get_object_world_pose(prim_path):
        prim = prim_mgr.get(prim_path)
        pos_batch, quat_wxyz_batch = prim.get_world_poses()
        pos = pos_batch[0]
        quat_wxyz = quat_wxyz_batch[0]

        quat_xyzw = np.array(
            [
                quat_wxyz[1],
                quat_wxyz[2],
                quat_wxyz[3],
                quat_wxyz[0],
            ]
        )
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = pos
        return T

    return _get_object_world_pose


def init_environment(task_name, object_poses_path=None, episode_index=0):
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
        aruco_quat_xyzw = np.array(
            [
                aruco_quat_wxyz[1],  # x
                aruco_quat_wxyz[2],  # y
                aruco_quat_wxyz[3],  # z
                aruco_quat_wxyz[0],  # w
            ]
        )

        # Update PRELOAD_OBJECTS with new poses
        preload_objects = registry_config.get("environment_vars", {}).get(
            "PRELOAD_OBJECTS", []
        )
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
    cam_shape = (3, 224, 224)
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
    return (
        world,
        manipulator,
        camera,
        registry_config,
        registry_class.is_episode_completed,
    )


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


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts 3x3 rotation matrix to 6D rotation representation.
    Takes the first two columns of the rotation matrix.
    Args:
        matrix: (3, 3) or (N, 3, 3)
    Returns:
        rot6d: (6,) or (N, 6) - [col0, col1] flattened
    """
    if len(matrix.shape) == 2:
        # Single matrix
        x_axis = matrix[:, 0]
        y_axis = matrix[:, 1]
        return np.concatenate([x_axis, y_axis])
    else:
        # Batch
        return np.concatenate([matrix[:, :, 0], matrix[:, :, 1]], axis=-1)


def rot6d_to_matrix(rot6d):
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    Uses Gram-Schmidt orthogonalization.
    Args:
        rot6d: (6,) or (N, 6)
    Returns:
        matrix: (3, 3) or (N, 3, 3)
    """
    is_batch = len(rot6d.shape) == 2
    if not is_batch:
        rot6d = rot6d[None, :]

    x_raw = rot6d[:, 0:3]
    y_raw = rot6d[:, 3:6]

    # Gram-Schmidt
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)

    # Stack as columns
    matrix = np.stack([x, y, z], axis=-1)

    if not is_batch:
        matrix = matrix[0]
    return matrix


# ==================== Action / Pose Conversions ====================


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose [x, y, z, rx, ry, rz] (axis-angle) to 4x4 matrix.

    Args:
        pose: (6,) or (N, 6) - [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z]
    Returns:
        mat: (4, 4) or (N, 4, 4)
    """
    is_batch = len(pose.shape) == 2
    if not is_batch:
        pose = pose[None, :]

    pos = pose[:, :3]
    rot_axis_angle = pose[:, 3:6]

    # Axis-angle to rotation matrix
    rot_mat = R.from_rotvec(rot_axis_angle).as_matrix()

    # Build 4x4 matrix
    mat = np.zeros((pose.shape[0], 4, 4))
    mat[:, :3, :3] = rot_mat
    mat[:, :3, 3] = pos
    mat[:, 3, 3] = 1.0

    if not is_batch:
        mat = mat[0]
    return mat


def mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 matrix to 10D pose [x, y, z, rot6d(6)].
    Args:
        mat: (4, 4) or (N, 4, 4)
    Returns:
        pose10d: (10,) or (N, 10)
    """
    is_batch = len(mat.shape) == 3
    if not is_batch:
        mat = mat[None, :]

    pos = mat[:, :3, 3]
    rot_mat = mat[:, :3, :3]
    rot6d = matrix_to_rot6d(rot_mat)  # 使用我們已有的函數

    pose10d = np.concatenate([pos, rot6d], axis=-1)

    if not is_batch:
        pose10d = pose10d[0]
    return pose10d


# ================================================================


def world_to_robot_base(pos_world, rot_mat_world, base_pos, base_rot_mat):
    """
    Transform pose from World Frame to Robot Base Frame.
    T_base_obj = inv(T_world_base) @ T_world_obj
    """
    T_world_base = np.eye(4)
    T_world_base[:3, :3] = base_rot_mat
    T_world_base[:3, 3] = base_pos

    T_world_obj = np.eye(4)
    T_world_obj[:3, :3] = rot_mat_world
    T_world_obj[:3, 3] = pos_world

    T_base_obj = np.linalg.inv(T_world_base) @ T_world_obj

    return T_base_obj[:3, 3], T_base_obj[:3, :3]


def robot_base_to_world(pos_base, rot_mat_base, base_pos, base_rot_mat):
    """
    Transform pose from Robot Base Frame to World Frame.
    T_world_obj = T_world_base @ T_base_obj
    """
    T_world_base = np.eye(4)
    T_world_base[:3, :3] = base_rot_mat
    T_world_base[:3, 3] = base_pos

    T_base_obj = np.eye(4)
    T_base_obj[:3, :3] = rot_mat_base
    T_base_obj[:3, 3] = pos_base

    T_world_obj = T_world_base @ T_base_obj

    return T_world_obj[:3, 3], T_world_obj[:3, :3]


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


def main():
    logger.info("Initializing inference Workspace.")
    # Create session directory for debug outputs
    session_dir = f"./data/eval_outputs/session_{int(time.time())}"
    debug_img_dir = os.path.join(session_dir, "DEBUG_IMGS")
    os.makedirs(debug_img_dir, exist_ok=True)
    logger.info(f"Debug images will be saved to: {debug_img_dir}")

    # set inference params
    logger.info("Policy initialized successfully.")

    world, manipulator, camera, registry_config, is_episode_completed = (
        init_environment(args.task, args.object_poses_path, args.episode_index)
    )

    policy_wrapper = PolicyWrapper(args.checkpoint)

    # Setup dual viewports (perspective + GoPro camera)
    setup_dual_viewports()

    # Initialize IK solvers
    lula_solver, art_kine_solver = initialize_ik_solvers(manipulator)
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

    # Warm up buffer (run sim steps before inference)
    for _ in range(100):
        world.step(render=True)
        time.sleep(1 / 60)

    # Move to "move_above" position without recording (like in generate_data.py)
    print("[Main] Moving to position above object...")
    while simulation_app.is_running():
        motion_planner.step(manipulator, lula_solver, art_kine_solver)
        for _ in range(50):
            world.step(render=True)

        # Phase becomes "descend" when robot reaches move_above position
        if motion_planner.pickplace.phase == "descend":
            print("[Main] Reached position above object.")
            break

    # Wait for user to type "start" before beginning evaluation
    print("[Main] Ready to start evaluation. Type 'start' to begin...")
    while True:
        user_input = input().strip().lower()
        if user_input == "start":
            break
        print("[Main] Invalid input. Type 'start' to begin.")

    step_count = 0
    eef_pos_list = []
    eef_rot_list = []
    gripper_list = []
    episode_start_pose = None
    episode_start_mat_base = None
    while simulation_app.is_running():
        world.step(render=True)

        # 1. Update solver with robot base (World Frame)
        cur_base_pos, cur_base_quat = (
            manipulator.get_world_pose()
        )  # base pose in world frame
        lula_solver.set_robot_base_pose(cur_base_pos, cur_base_quat)

        # 2. Get EE(End Effector) Pose in World Frame
        ee_pos_world, ee_rot_mat_world = (
            art_kine_solver.compute_end_effector_pose()
        )  # end effector pose in world frame
        ee_pos_world = ee_pos_world.astype(np.float64)
        ee_rot_mat_world = ee_rot_mat_world[:3, :3]

        # record start pose
        if episode_start_pose is None:
            start_rot_vec = R.from_matrix(ee_rot_mat_world).as_rotvec()
            episode_start_pose = np.concatenate([ee_pos_world, start_rot_vec])

        # 3. Prepare EE Pose for Model (World Frame)
        ee_pos_for_model = ee_pos_world.astype(np.float32)
        ee_rot_mat_for_model = ee_rot_mat_world

        # 4. Transform rotation matrix to axis-angle
        ee_rot_axis_angle_for_model = (
            R.from_matrix(ee_rot_mat_for_model).as_rotvec().astype(np.float32)
        )

        # 5. Get Gripper state
        joint_pos = manipulator.get_joint_positions()
        gripper_width = joint_pos[-2] + joint_pos[-1]

        # 5.5 Get EE Rotation relative to start (World Frame)
        curr_ee_mat_world = np.eye(4)
        curr_ee_mat_world[:3, 3] = ee_pos_world
        curr_ee_mat_world[:3, :3] = ee_rot_mat_world

        if episode_start_mat_base is None:  # Save the first frame ee pose as base
            episode_start_mat_base = curr_ee_mat_world.copy()

        rel_mat = (
            np.linalg.inv(episode_start_mat_base) @ curr_ee_mat_world
        )  # relative to start
        rel_rot_mat = rel_mat[:3, :3]
        rel_rot_6d = matrix_to_rot6d(rel_rot_mat)  # relative rot6d to start

        # 6. Get Image (Wrist Camera)
        wrist_img = camera.get_rgb()  # (H, W, C)

        # Error handling: If image is missing, step simulation and retry
        if wrist_img is None:
            world.step(render=True)
            continue

        # Remove alpha channel if present
        if wrist_img.shape[-1] == 4:
            wrist_img = wrist_img[..., :3]

        obs_dict = {
            "camera0_rgb": wrist_img,
            "robot0_eef_pos": ee_pos_for_model,  # World Frame
            "robot0_eef_rot_axis_angle": ee_rot_axis_angle_for_model,  # World Frame
            "robot0_eef_rot_axis_angle_wrt_start": rel_rot_6d.astype(np.float32),
            "robot0_gripper_width": np.array([gripper_width], dtype=np.float32),
        }

        actions_abs, ref_mat, raw_actions = policy_wrapper.get_action(
            obs_dict
        )  # return absolute coord actions
        n_execution = min(1000, len(actions_abs))

        for i in range(n_execution):
            if not simulation_app.is_running():
                break

            # Get action in absolute coord
            action = actions_abs[i]
            target_pos_base = action["pos"]
            target_rot_mat_base = action["rot_mat"]
            action_gripper = action["gripper"]

            target_pos_world = target_pos_base.copy()  # at World Frame
            target_rot_mat_world = target_rot_mat_base.copy()  # at World Frame
            target_quat_xyzw = R.from_matrix(target_rot_mat_world).as_quat()
            target_quat_wxyz = np.array(
                [
                    target_quat_xyzw[3],
                    target_quat_xyzw[0],
                    target_quat_xyzw[1],
                    target_quat_xyzw[2],
                ]
            )

            # --- Apply Action (IK) ---
            ik_action, success = art_kine_solver.compute_inverse_kinematics(
                target_position=target_pos_world, target_orientation=target_quat_wxyz
            )

            if success:
                manipulator.set_joint_positions(ik_action.joint_positions, np.arange(7))
            else:
                print(f"[Warning] IK Failed at step {step_count}")

            # Apply Gripper command
            set_gripper_width(manipulator, action_gripper)

            # [Important] Step the physics engine forward for one tick
            for _ in range(30):
                world.step(render=True)

            # Record Data
            rec_pos, rec_rot_mat = art_kine_solver.compute_end_effector_pose()
            rec_rot_vec = R.from_matrix(rec_rot_mat[:3, :3]).as_rotvec()
            rec_joint = manipulator.get_joint_positions()
            rec_grip = rec_joint[-2] + rec_joint[-1]

            eef_pos_list.append(rec_pos.astype(np.float32))
            eef_rot_list.append(rec_rot_vec.astype(np.float32))
            gripper_list.append(np.array([rec_grip], dtype=np.float32))
            # ----------------------------------------------------

            step_count += 1
            if step_count % 50 == 0:
                print(
                    f"Step {step_count}: Target {target_pos_base} | Grip {action_gripper:.3f}"
                )

                print("\n[Main] Episode finished. Checking success...")

        if len(eef_pos_list) > 0:
            episode_end_pose = np.concatenate([eef_pos_list[-1], eef_rot_list[-1]])
        else:
            episode_end_pose = episode_start_pose

        # construct demo start/end lists
        num_steps = len(eef_pos_list)
        demo_start_list = np.repeat(episode_start_pose[None, :], num_steps, axis=0)
        demo_end_list = np.repeat(episode_end_pose[None, :], num_steps, axis=0)

        # Construct Record Dict
        episode_record = {
            "episode_idx": args.episode_index,
            "eef_pos": np.stack(eef_pos_list, 0) if eef_pos_list else np.array([]),
            "eef_rot": np.stack(eef_rot_list, 0) if eef_rot_list else np.array([]),
            "gripper": np.stack(gripper_list, 0) if gripper_list else np.array([]),
            "demo_start": demo_start_list,
            "demo_end": demo_end_list,
        }

        # Success Check
        try:
            episode_success = is_episode_completed(episode_record)
            episode_record["success"] = episode_success

            print(
                "===================================================================="
            )
            if episode_success:
                print(f"[Main] Episode {args.episode_idx}: Task SUCCESS")
            else:
                print(f"[Main] Episode {args.episode_idx}: Task FAIL")
            print(
                "===================================================================="
            )
        except Exception as e:
            print(f"[Main] Error checking success: {e}")
            print(
                "Make sure registry.is_episode_completed works with the recorded data format."
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
