import argparse
import importlib
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import dill
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None  # type: ignore

import registry

# --- Constants ---
BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/franka_panda/franka_panda_arm.usd"
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

def get_end_effector_pos_quat_wxyz(panda, lula_solver, art_kine_solver):
    base_pos, base_quat = panda.get_world_pose()
    lula_solver.set_robot_base_pose(robot_position=base_pos, robot_orientation=base_quat)

    ee_pos, ee_T = art_kine_solver.compute_end_effector_pose()  # ee_T[:3,:3] rotation
    quat_xyzw = R.from_matrix(ee_T[:3, :3]).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return ee_pos.astype(np.float64), quat_wxyz.astype(np.float64)

def calibrate_robot_base(panda, lula_solver):
    """
    Update Lula solver with current robot base pose.
    Must be called before computing IK.
    
    Args:
        panda: Panda articulation object
        lula_solver: LulaKinematicsSolver instance
    """
    robot_pos, robot_quat = panda.get_world_pose()
    lula_solver.set_robot_base_pose(
        robot_position=robot_pos,
        robot_orientation=robot_quat
    )

def apply_ik_solution(panda, art_kine_solver, target_pos, target_quat_wxyz):
    """
    Compute and apply IK solution for target pose.
    
    Args:
        panda: Panda articulation object
        art_kine_solver: ArticulationKinematicsSolver instance
        target_pos: Target position (3,)
        target_quat_wxyz: Target orientation as quaternion WXYZ (4,)
        step_idx: Current step index (for logging)
        
    Returns:
        bool: True if IK succeeded
    """
    action, success = art_kine_solver.compute_inverse_kinematics(
        target_position=target_pos,
        target_orientation=target_quat_wxyz
    )

    if success:
        panda.set_joint_positions(action.joint_positions, np.arange(7))
        return True

    return False

def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Uses Gram-Schmidt orthogonalization (Zhou et al., 2019).

    Args:
        rot6d: (6,) array representing first two columns of rotation matrix

    Returns:
        (3, 3) rotation matrix
    """
    a1 = rot6d[:3]
    a2 = rot6d[3:6]

    # Normalize first vector
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)

    # Orthogonalize second vector
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)

    # Cross product for third vector
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=1)


def set_gripper_width(panda, width: float, threshold: float = 0.04, step: float = 0.01):
    """
    Threshold-based gripper control with gradual movement.

    Args:
        panda: Robot articulation
        width: Input gripper width from policy
        threshold: Threshold to determine open (>=) or closed (<)
        step: Amount to change finger position per call
    """
    target_pos = 1.0 if width >= threshold else 0.0

    idx1 = panda.get_dof_index("panda_finger_joint1")
    idx2 = panda.get_dof_index("panda_finger_joint2")

    if idx1 is not None and idx2 is not None:
        current_positions = panda.get_joint_positions(
            joint_indices=np.array([idx1, idx2])
        )
        current_pos = current_positions[0]

        # Smaller steps for closing (grasping)
        if target_pos < current_pos:
            step = min(step, 0.005)
            if abs(current_pos - target_pos) < 0.02:
                step = 0.002

        # Move gradually toward target
        if current_pos < target_pos:
            finger_pos = min(current_pos + step, target_pos)
        elif current_pos > target_pos:
            finger_pos = max(current_pos - step, target_pos)
        else:
            finger_pos = target_pos

        panda.set_joint_positions(
            positions=np.array([finger_pos, finger_pos]),
            joint_indices=np.array([idx1, idx2]),
        )


class IsaacSimEnvironment:
    """
    Isaac Sim environment wrapper with IK-based action execution.

    Handles:
    - SimulationApp lifecycle
    - World/scene setup from registry config
    - Robot spawning and IK solver initialization
    - Action execution (position + rot6d + gripper -> IK -> joint commands)
    - Observation capture (RGB + proprioception)

    Coordinate System Note:
    - Training data uses RELATIVE pose representation by default (obs_pose_repr='rel')
    - Actions predicted by policy are RELATIVE to the last observation pose
    - This class converts relative actions to absolute world-frame poses before IK
    """

    def __init__(
        self,
        task_name: str,
        sim_config: Optional[Dict[str, Any]] = None,
        shape_meta: Optional[Dict[str, Any]] = None,
        pose_repr: Optional[Dict[str, str]] = None,
    ) -> None:
        self.task_name = task_name
        self.shape_meta = shape_meta or {}
        self._obs_history: Dict[str, List[np.ndarray]] = {}

        # Pose representation configuration (from training)
        # Defaults match umi_dataset.py defaults
        self.pose_repr = pose_repr or {}
        self.obs_pose_repr = self.pose_repr.get("obs_pose_repr", "rel")
        self.action_pose_repr = self.pose_repr.get("action_pose_repr", "rel")

        # Load configuration from registry
        registry_class = registry.get_task_registry(task_name)
        if not registry_class.validate_environment():
            print(
                f"[IsaacSimEnvironment] WARNING: Registry validation failed for {task_name}"
            )
        self.registry_config = registry_class.get_config()
        self.is_episode_completed = registry_class.is_episode_completed

        # IK solver references (initialized after world setup)
        self.lula_solver = None
        self.art_kine_solver = None
        self.manipulator = None

        # Pose tracking for relative observations and actions
        self._initial_eef_rot: Optional[np.ndarray] = None
        self._last_eef_pos: Optional[np.ndarray] = None
        self._last_eef_rot: Optional[np.ndarray] = None

        default_sim_config = {
            "headless": False,
            "width": 1280,
            "height": 720,
            "enable_streaming": False,
            "extensions": ["isaacsim.robot_motion.motion_generation"],
        }
        merged_sim_config = {**default_sim_config, **(sim_config or {})}

        print("[IsaacSimEnvironment] Starting SimulationApp...")
        from isaacsim import SimulationApp

        self.simulation_app = SimulationApp(merged_sim_config)
        self.world = None
        self.camera = None

        try:
            self._initialize_world_and_camera()
            self._initialize_ik_solvers()
        except Exception as e:
            print(f"[IsaacSimEnvironment] Initialization error: {e}")
            import traceback

            traceback.print_exc()


    def _set_to_init_pose(self):
        curr_pos, _ = get_end_effector_pos_quat_wxyz(
            self.manipulator, self.lula_solver, self.art_kine_solver
        )
        print(f'Setting to init pose: {self.task_name}')

        if self.task_name=="kitchen":
            INIT_EE_POS = curr_pos + np.array([-0.16, 0., 0.13])
            INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
        elif self.task_name=="dining-room":
            INIT_EE_POS = curr_pos + np.array([-0.16, 0., 0.13])
            INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
        elif self.task_name=="living-room":
            INIT_EE_POS = curr_pos + np.array([-0.1, 0.2, 0.20])
            INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
        else:
            raise RuntimeError(
                f"Unknown task, expected one of 'kitchen', 'dining-room', 'living-room', got {self.task_name}"
            )

        calibrate_robot_base(self.manipulator, self.lula_solver)
        success = apply_ik_solution(
            self.manipulator,
            self.art_kine_solver,
            INIT_EE_POS,
            INIT_EE_QUAT_WXYZ,
        )


    def _initialize_world_and_camera(self) -> None:
        import os

        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise RuntimeError("Could not find Isaac Sim assets folder.")

        enable_extension("isaacsim.robot_motion.motion_generation")
        stage_utils.open_stage(BASE_SCENE_FP)

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # --- Setup robot ---
        robot = stage_utils.add_reference_to_stage(
            usd_path=FRANKA_PANDA_FP, prim_path=FRANKA_PANDA_PRIM_PATH
        )
        robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
        robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

        self.robot_xform = SingleXFormPrim(prim_path=FRANKA_PANDA_PRIM_PATH)

        gripper = ParallelGripper(
            end_effector_prim_path=f"{FRANKA_PANDA_PRIM_PATH}/panda/panda_rightfinger",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.02, 0.02]),
            action_deltas=np.array([0.01, 0.01]),
        )

        self.manipulator = self.world.scene.add(
            SingleManipulator(
                prim_path=FRANKA_PANDA_PRIM_PATH,
                name="my_franka",
                end_effector_prim_path=f"{FRANKA_PANDA_PRIM_PATH}/panda/panda_rightfinger",
                gripper=gripper,
            )
        )
        self.manipulator.gripper.set_default_state(
            self.manipulator.gripper.joint_opened_positions
        )

        # --- Set robot pose from registry config ---
        franka_pose = self.registry_config.get("franka_pose", {})
        franka_translation = np.array(franka_pose.get("translation", [0.0, 0.0, 0.0]))
        franka_rotation = np.array(
            franka_pose.get("rotation_quat", [1.0, 0.0, 0.0, 0.0])
        )
        self.robot_xform.set_local_pose(
            translation=franka_translation / stage_utils.get_stage_units(),
            orientation=franka_rotation,
        )

        # --- Set camera view from registry config ---
        camera_pose = self.registry_config.get("camera_pose", {})
        camera_translation = np.array(camera_pose.get("translation", [0.0, 0.0, 0.0]))
        set_camera_view(camera_translation, franka_translation)

        # --- Setup observation camera ---
        cam_meta = self.shape_meta.get("obs", {}).get("camera0_rgb", {})
        cam_shape = cam_meta.get("shape", (3, 224, 224))
        if len(cam_shape) >= 3:
            _, height, width = cam_shape[:3]
        else:
            height = width = 224

        self.camera = Camera(
            prim_path=f"{GOPRO_PRIM_PATH}/Camera",
            name="gopro_camera",
            resolution=(int(width), int(height)),
        )
        self.camera.initialize()
        self.world.reset()

        # --- Load preload objects from registry ---
        self._load_preload_objects(stage_utils)

    def _load_preload_objects(self, stage_utils) -> None:
        """Load objects specified in registry config's PRELOAD_OBJECTS at their default positions."""
        import os

        from isaacsim.core.prims import SingleXFormPrim

        env_vars = self.registry_config.get("environment_vars", {})
        preload_objects = env_vars.get("PRELOAD_OBJECTS", [])

        self.object_prims: Dict[str, Any] = {}

        for entry in preload_objects:
            raw_name = entry.get("name", "unknown")
            asset_filename = entry.get("assets")
            prim_path = entry.get("prim_path")

            if not asset_filename or not prim_path:
                print(f"[IsaacSimEnvironment] Skipping invalid preload entry: {entry}")
                continue

            full_asset_path = os.path.join(ASSETS_DIR, asset_filename)
            if not os.path.exists(full_asset_path):
                print(f"[IsaacSimEnvironment] Asset not found: {full_asset_path}")
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
                self.world.scene.add(obj_prim)
                self.object_prims[raw_name] = obj_prim
                print(
                    f"[IsaacSimEnvironment] Loaded {raw_name} at {prim_path}, position={default_position}"
                )
            except Exception as e:
                print(f"[IsaacSimEnvironment] Failed to load {raw_name}: {e}")

    def _reposition_objects_to_default(self) -> None:
        """Reset all preloaded objects to their default positions (for inference)."""
        if not hasattr(self, "object_prims") or not self.object_prims:
            return

        env_vars = self.registry_config.get("environment_vars", {})
        preload_objects = env_vars.get("PRELOAD_OBJECTS", [])

        for entry in preload_objects:
            raw_name = entry.get("name", "unknown")
            if raw_name not in self.object_prims:
                continue

            default_position = entry.get("default_position")
            if default_position is None:
                continue

            default_position = np.array(default_position, dtype=np.float64)
            orientation = np.array(entry.get("quat_wxyz", [1, 0, 0, 0]))

            self.object_prims[raw_name].set_local_pose(
                translation=default_position,
                orientation=orientation,
            )

    def _initialize_ik_solvers(self) -> None:
        """Initialize Lula IK solvers for action execution."""
        from isaacsim.robot_motion.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
        )

        print("[IsaacSimEnvironment] Initializing IK solvers...")

        self.lula_solver = LulaKinematicsSolver(
            robot_description_path=LULA_ROBOT_DESCRIPTION_PATH,
            urdf_path=LULA_URDF_PATH,
        )

        self.art_kine_solver = ArticulationKinematicsSolver(
            self.manipulator,
            kinematics_solver=self.lula_solver,
            end_effector_frame_name="umi_tcp",
        )

        # Calibrate robot base pose
        robot_pos, robot_quat = self.manipulator.get_world_pose()
        self.lula_solver.set_robot_base_pose(
            robot_position=robot_pos, robot_orientation=robot_quat
        )

        print("[IsaacSimEnvironment] IK solvers initialized.")

    def _get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector position and rotation matrix.

        Returns:
            (position (3,), rotation_matrix (3,3))
        """
        robot_pos, robot_quat = self.manipulator.get_world_pose()
        self.lula_solver.set_robot_base_pose(
            robot_position=robot_pos, robot_orientation=robot_quat
        )
        ee_pos, ee_rot_matrix = self.art_kine_solver.compute_end_effector_pose()
        return ee_pos.astype(np.float64), ee_rot_matrix[:3, :3].astype(np.float64)

    def _rotation_matrix_to_rot6d(self, rot_mat: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to 6D representation."""
        return rot_mat[:, :2].flatten()

    def is_running(self) -> bool:
        return bool(self.simulation_app and self.simulation_app.is_running())

    def update(self) -> None:
        if self.simulation_app:
            self.simulation_app.update()

    def _get_horizon(self, key: str) -> int:
        return int(self.shape_meta.get("obs", {}).get(key, {}).get("horizon", 1))

    def _capture_observation(self) -> Dict[str, np.ndarray]:
        """Capture current observations from the environment."""
        obs: Dict[str, np.ndarray] = {}

        # Camera RGB
        if self.camera is not None:
            rgb = self.camera.get_rgb()
            if rgb is not None:
                obs["camera0_rgb"] = rgb

        # Fallback placeholder for camera
        cam_meta = self.shape_meta.get("obs", {}).get("camera0_rgb", {})
        cam_shape = cam_meta.get("shape")
        if "camera0_rgb" not in obs and cam_shape is not None:
            if len(cam_shape) >= 3:
                _, height, width = cam_shape[:3]
            else:
                height = width = 224
            obs["camera0_rgb"] = np.zeros((int(height), int(width), 3), dtype=np.uint8)

        # End-effector pose
        if self.art_kine_solver is not None and self.manipulator is not None:
            ee_pos, ee_rot_mat = self._get_end_effector_pose()

            obs["robot0_eef_pos"] = ee_pos.astype(np.float32)

            # Rotation as 6D representation
            rot6d = self._rotation_matrix_to_rot6d(ee_rot_mat)
            obs["robot0_eef_rot_axis_angle"] = rot6d.astype(np.float32)

            # Rotation relative to episode start
            if self._initial_eef_rot is not None:
                rel_rot = self._initial_eef_rot.T @ ee_rot_mat
                rel_rot6d = self._rotation_matrix_to_rot6d(rel_rot)
                obs["robot0_eef_rot_axis_angle_wrt_start"] = rel_rot6d.astype(
                    np.float32
                )
            else:
                obs["robot0_eef_rot_axis_angle_wrt_start"] = rot6d.astype(np.float32)

            # Gripper width
            joint_pos = self.manipulator.get_joint_positions()
            gripper_width = joint_pos[-2] + joint_pos[-1]
            obs["robot0_gripper_width"] = np.array([gripper_width], dtype=np.float32)

        # Fill any missing observations with zeros based on shape_meta
        for key, meta in self.shape_meta.get("obs", {}).items():
            if key in obs:
                continue
            shape = meta.get("shape") or meta.get("raw_shape")
            if shape is None:
                continue
            obs[key] = np.zeros(shape, dtype=np.float32)

        return obs

    def _update_history(self, obs: Dict[str, np.ndarray]) -> None:
        for key, value in obs.items():
            history = self._obs_history.setdefault(key, [])
            history.append(value)
            max_h = self._get_horizon(key)
            if len(history) > max_h:
                self._obs_history[key] = history[-max_h:]

    def _stack_history(self) -> Dict[str, np.ndarray]:
        """
        Stack observation history and convert to relative pose representation.

        During training, observations are converted to be relative to the LAST
        observation in the sequence (pose_mat[-1]). This method replicates that
        behavior for inference consistency.
        """
        stacked: Dict[str, np.ndarray] = {}
        for key, history in self._obs_history.items():
            if not history:
                continue
            horizon = self._get_horizon(key)
            if len(history) < horizon:
                pad = [history[0]] * (horizon - len(history))
                data = pad + history
            else:
                data = history[-horizon:]
            stacked[key] = np.stack(data, axis=0)

        # Convert pose observations to relative representation if configured
        if (
            self.obs_pose_repr in ("rel", "relative")
            and "robot0_eef_pos" in stacked
            and "robot0_eef_rot_axis_angle" in stacked
        ):
            stacked = self._convert_obs_to_relative(stacked)

        return stacked

    def _convert_obs_to_relative(
        self, stacked: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Convert stacked observations to relative pose representation.

        Replicates the training data processing from umi_dataset.py:
        - All poses become relative to the LAST observation in the sequence
        - For pose_rep='rel': pos = pos - base_pos, rot = rot @ inv(base_rot)

        Args:
            stacked: Dictionary of stacked observations

        Returns:
            Dictionary with pose observations converted to relative format
        """
        pos = stacked["robot0_eef_pos"]  # [horizon, 3]
        rot6d = stacked["robot0_eef_rot_axis_angle"]  # [horizon, 6]

        # Get the last (current) pose as base
        base_pos = pos[-1]  # [3]
        base_rot6d = rot6d[-1]  # [6]
        base_rot_mat = rot6d_to_rotation_matrix(base_rot6d)  # [3, 3]
        base_rot_inv = base_rot_mat.T  # Inverse of rotation matrix

        # Convert each observation to be relative to the last one
        rel_pos = np.zeros_like(pos)
        rel_rot6d = np.zeros_like(rot6d)

        for i in range(len(pos)):
            # Position: relative = current - base
            rel_pos[i] = pos[i] - base_pos

            # Rotation: relative = current @ inv(base)
            curr_rot_mat = rot6d_to_rotation_matrix(rot6d[i])
            rel_rot_mat = curr_rot_mat @ base_rot_inv
            rel_rot6d[i] = self._rotation_matrix_to_rot6d(rel_rot_mat)

        stacked["robot0_eef_pos"] = rel_pos.astype(np.float32)
        stacked["robot0_eef_rot_axis_angle"] = rel_rot6d.astype(np.float32)

        return stacked

    def reset(self) -> Dict[str, np.ndarray]:
        if self.world is None:
            raise RuntimeError("IsaacSimEnvironment is not initialized with a world.")

        self.world.reset()
        self._reposition_objects_to_default()

        # Always return the arm to the canonical start pose before collecting observations.
        self._set_to_init_pose()

        for _ in range(20):
            self.world.step(render=True)

        self._obs_history.clear()
        self._initial_eef_rot = None
        self._last_eef_pos = None
        self._last_eef_rot = None

        # Recalibrate IK solver after reset
        if self.lula_solver is not None and self.manipulator is not None:
            robot_pos, robot_quat = self.manipulator.get_world_pose()
            self.lula_solver.set_robot_base_pose(
                robot_position=robot_pos, robot_orientation=robot_quat
            )

        # Capture initial observation and set initial rotation
        obs = self._capture_observation()

        # Store initial rotation for relative observations
        if self.art_kine_solver is not None:
            ee_pos, ee_rot_mat = self._get_end_effector_pose()
            self._initial_eef_rot = ee_rot_mat.copy()
            # Initialize last pose tracking for relative action conversion
            self._last_eef_pos = ee_pos.copy()
            self._last_eef_rot = ee_rot_mat.copy()

        self._update_history(obs)
        return self._stack_history()

    def _convert_relative_to_absolute_pose(
        self,
        rel_pos: np.ndarray,
        rel_rot_mat: np.ndarray,
        base_pos: np.ndarray,
        base_rot_mat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert relative pose to absolute pose.

        This is the backward pass of the 'rel' pose representation from
        pose_repr_util.py:convert_pose_mat_rep().

        For pose_rep='rel':
            Forward (training):  rel_pos = abs_pos - base_pos
                                 rel_rot = abs_rot @ inv(base_rot)
            Backward (inference): abs_pos = rel_pos + base_pos
                                  abs_rot = rel_rot @ base_rot

        Args:
            rel_pos: Relative position (3,)
            rel_rot_mat: Relative rotation matrix (3, 3)
            base_pos: Base position (last observation eef pos) (3,)
            base_rot_mat: Base rotation matrix (last observation eef rot) (3, 3)

        Returns:
            (absolute_pos, absolute_rot_mat): Tuple of absolute position and rotation
        """
        abs_pos = rel_pos + base_pos
        abs_rot_mat = rel_rot_mat @ base_rot_mat
        return abs_pos, abs_rot_mat

    def _apply_action(self, action: np.ndarray) -> None:
        """
        Apply action to the robot using IK.

        Action format: [N, 10] where each row is:
            - [0:3]: target position (RELATIVE to last observation)
            - [3:9]: target rotation in 6D representation (RELATIVE to last observation)
            - [9]: gripper width

        The policy outputs actions in RELATIVE pose representation (default for UMI training).
        This method converts relative actions to absolute world-frame poses before IK.

        For now, we apply only the first action in the horizon.
        """
        if self.art_kine_solver is None or self.manipulator is None:
            print("[IsaacSimEnvironment] IK solver not initialized, skipping action.")
            return

        # Handle action shape: could be (10,) or (N, 10)
        if action.ndim == 1:
            action_step = action
        else:
            action_step = action[0]  # Take first action in horizon

        rel_pos = action_step[:3]
        rel_rot6d = action_step[3:9]
        gripper_width = action_step[9]

        # Convert 6D rotation to rotation matrix
        rel_rot_mat = rot6d_to_rotation_matrix(rel_rot6d)

        # Convert relative pose to absolute pose based on pose representation
        if self.action_pose_repr == "abs":
            # Actions are already absolute - use directly
            abs_pos = rel_pos
            abs_rot_mat = rel_rot_mat
        elif self.action_pose_repr in ("rel", "relative"):
            # Actions are relative to last observation pose - convert to absolute
            if self._last_eef_pos is None or self._last_eef_rot is None:
                print(
                    "[IsaacSimEnvironment] WARNING: Last EEF pose not available, using action as absolute"
                )
                abs_pos = rel_pos
                abs_rot_mat = rel_rot_mat
            else:
                abs_pos, abs_rot_mat = self._convert_relative_to_absolute_pose(
                    rel_pos, rel_rot_mat, self._last_eef_pos, self._last_eef_rot
                )
        elif self.action_pose_repr == "delta":
            # Delta actions: cumulative from base
            if self._last_eef_pos is None or self._last_eef_rot is None:
                print(
                    "[IsaacSimEnvironment] WARNING: Last EEF pose not available, using action as absolute"
                )
                abs_pos = rel_pos
                abs_rot_mat = rel_rot_mat
            else:
                # Delta: position is additive, rotation is multiplicative
                abs_pos = rel_pos + self._last_eef_pos
                abs_rot_mat = rel_rot_mat @ self._last_eef_rot
        else:
            print(
                f"[IsaacSimEnvironment] WARNING: Unknown action_pose_repr '{self.action_pose_repr}', using as absolute"
            )
            abs_pos = rel_pos
            abs_rot_mat = rel_rot_mat

        # Convert absolute rotation matrix to quaternion for IK
        rot = R.from_matrix(abs_rot_mat)
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Compute IK
        action_result, success = self.art_kine_solver.compute_inverse_kinematics(
            target_position=abs_pos, target_orientation=quat_wxyz
        )

        if success:
            self.manipulator.set_joint_positions(
                action_result.joint_positions, np.arange(7)
            )
            # Update last pose tracking after successful action
            self._last_eef_pos = abs_pos.copy()
            self._last_eef_rot = abs_rot_mat.copy()
        else:
            print(
                f"[IsaacSimEnvironment] IK failed for target pos={abs_pos}, quat={quat_wxyz}"
            )

        # Apply gripper action
        set_gripper_width(self.manipulator, gripper_width)

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        if self.world is None:
            raise RuntimeError("IsaacSimEnvironment is not initialized with a world.")

        self._apply_action(action)
        self.world.step(render=True)

        obs = self._capture_observation()
        self._update_history(obs)

        # Update last EEF pose for next action's relative-to-absolute conversion
        # This ensures actions are relative to the actual current observation
        if self.art_kine_solver is not None:
            ee_pos, ee_rot_mat = self._get_end_effector_pose()
            self._last_eef_pos = ee_pos.copy()
            self._last_eef_rot = ee_rot_mat.copy()

        reward = 0.0
        done = False
        info: Dict[str, Any] = {"episode_record": obs}
        return self._stack_history(), reward, done, info

    def close(self) -> None:
        if self.simulation_app:
            self.simulation_app.close()
            self.simulation_app = None


# --- Utility functions ---


def _register_resolvers() -> None:
    if OmegaConf is None:
        return
    try:
        OmegaConf.register_new_resolver("eval", eval, replace=True)  # type: ignore
    except Exception:
        pass


class AttrDict(dict):
    """Minimal dict that supports attribute access (obj.key)."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _import_from_string(path: str):
    candidates = [path]
    if "env_runners." in path:
        candidates.append(path.replace("env_runners.", "env_runner."))

    last_err: Optional[Exception] = None
    for candidate in candidates:
        try:
            module_name, attr_name = candidate.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except Exception as exc:
            last_err = exc
            continue
    raise last_err if last_err else ImportError(f"Could not import {path}")


def _to_container(cfg: Any) -> Dict[str, Any]:
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "items"):
        return dict(cfg)  # type: ignore
    return {}


def _instantiate(cfg: Any):
    if isinstance(cfg, dict):
        if "_target_" in cfg:
            target = cfg["_target_"]
            kwargs = {k: _instantiate(v) for k, v in cfg.items() if k != "_target_"}
            cls = _import_from_string(target)
            return cls(**kwargs)
        return AttrDict({k: _instantiate(v) for k, v in cfg.items()})
    if isinstance(cfg, list):
        return [_instantiate(v) for v in cfg]
    if isinstance(cfg, tuple):
        return tuple(_instantiate(list(cfg)))
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run Isaac Sim policy evaluation without Hydra."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["kitchen", "dining-room", "living-room"],
        required=True,
        help="Task name for registry configuration",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint .ckpt file"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save logs")
    parser.add_argument("--device", default="cuda:0", help="Torch device for policy")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run simulation in headless mode",
    )
    args = parser.parse_args()

    _register_resolvers()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    # Load checkpoint to CPU first to avoid initializing CUDA before Isaac Sim
    payload = torch.load(
        open(args.checkpoint, "rb"), pickle_module=dill, map_location="cpu"
    )
    cfg = payload["cfg"]

    shape_meta = None
    if hasattr(cfg, "shape_meta"):
        shape_meta = _to_container(cfg.shape_meta)

    # Extract pose representation configuration from checkpoint
    # This determines how observations and actions are represented (absolute vs relative)
    pose_repr = {}
    if getattr(cfg, "task", None) is not None:
        task_cfg = cfg.task
        if hasattr(task_cfg, "dataset") and hasattr(task_cfg.dataset, "pose_repr"):
            pose_repr = _to_container(task_cfg.dataset.pose_repr)
        elif hasattr(task_cfg, "pose_repr"):
            pose_repr = _to_container(task_cfg.pose_repr)

    # Also check env_runner for pose_repr
    if not pose_repr and getattr(cfg, "task", None) is not None:
        env_runner_cfg = getattr(cfg.task, "env_runner", None)
        if env_runner_cfg is not None and hasattr(env_runner_cfg, "pose_repr"):
            pose_repr = _to_container(env_runner_cfg.pose_repr)

    print(f"Pose representation config: {pose_repr}")

    # Get runner target from task name
    runner_target = TASK_RUNNER_MAP.get(args.task)
    if not runner_target:
        raise RuntimeError(
            f"No runner configured for task: {args.task}. "
            f"Available tasks: {list(TASK_RUNNER_MAP.keys())}"
        )

    # Extract runner configuration from checkpoint (for n_episodes, max_steps, etc.)
    runner_cfg = {}
    if getattr(cfg, "task", None) is not None:
        runner_cfg = _to_container(getattr(cfg.task, "env_runner", {}))

    # Build sim_config from runner config and CLI args
    sim_config = runner_cfg.get("sim_config", {})
    if args.headless:
        sim_config["headless"] = True

    # 1. Initialize Isaac Sim Environment FIRST (before CUDA)
    # Uses registry for configuration based on --task argument
    print(f"Initializing Isaac Sim Environment for task: {args.task}...")
    isaac_env = IsaacSimEnvironment(
        task_name=args.task,
        sim_config=sim_config,
        shape_meta=shape_meta,
        pose_repr=pose_repr,
    )

    # 2. Instantiate Policy and move to GPU (after SimApp is running)
    print("Instantiating Policy...")
    policy_cfg = _to_container(getattr(cfg, "policy", {}))
    if shape_meta is not None:
        policy_cfg.setdefault("shape_meta", shape_meta)

    policy = _instantiate(policy_cfg)

    state_dicts = payload.get("state_dicts", {})
    if "model" in state_dicts:
        policy.load_state_dict(state_dicts["model"], strict=False)

    ema_policy = None
    if (
        getattr(getattr(cfg, "training", None), "use_ema", False)
        and "ema_model" in state_dicts
    ):
        ema_policy = _instantiate(policy_cfg)
        ema_policy.load_state_dict(state_dicts["ema_model"], strict=False)

    active_policy = ema_policy or policy

    device = torch.device(args.device)
    active_policy.to(device)
    active_policy.eval()

    # 3. Instantiate the Runner with the existing environment
    print(f"Instantiating Runner: {runner_target}")
    # Extract runner kwargs from checkpoint config (excluding _target_)
    runner_kwargs = {k: v for k, v in runner_cfg.items() if k != "_target_"}

    if shape_meta is not None:
        runner_kwargs.setdefault("shape_meta", shape_meta)

    # Inject the pre-initialized environment and task_name for registry lookup
    runner_kwargs["env"] = isaac_env
    runner_kwargs["task_name"] = args.task

    # Instantiate the runner
    env_runner = _instantiate(
        {"_target_": runner_target, **runner_kwargs, "output_dir": str(output_dir)}
    )


    isaac_env._set_to_init_pose()
    import time;time.sleep(1)

    # 4. Run Evaluation
    print("Starting evaluation...")
    runner_log = env_runner.run(active_policy)

    # Save logs
    json_log = {}
    for key, value in runner_log.items():
        if hasattr(value, "_path"):  # wandb video-like objects
            json_log[key] = value._path
        else:
            try:
                json.dumps(value)
                json_log[key] = value
            except TypeError:
                json_log[key] = str(value)

    out_path = output_dir.joinpath("eval_log.json")
    with out_path.open("w") as f:
        json.dump(json_log, f, indent=2, sort_keys=True)

    print(f"Evaluation complete. Results saved to {out_path}")
    print(f"Success rate: {runner_log.get('success_rate', 'N/A')}")

    if hasattr(env_runner, "close"):
        try:
            env_runner.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
