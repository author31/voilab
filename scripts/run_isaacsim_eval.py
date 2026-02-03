"""
Standalone Isaac Sim evaluation without Hydra (Python 3.11 friendly).
Uses the shared registry system for task configuration (kitchen, dining-room, living-room).
.venv/bin/python scripts/run_isaacsim_eval.py --task kitchen --checkpoint data/outputs/2026.01.25/22.51.20_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt --output-dir ./tmp/eval
.venv/bin/python scripts/run_isaacsim_eval.py --task kitchen --checkpoint data/outputs/2026.01.30/12.06.28_train_diffusion_unet_timm_umi/checkpoints/epoch=0060-train_loss=0.014.ckpt --output-dir ./tmp/eval
Example:
Example:
    python run_isaacsim_eval.py --task kitchen --checkpoint /path/to.ckpt --output-dir /tmp/eval
"""

import argparse
import importlib
import json
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import dill
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os
import cv2 # Ensure cv2 is imported


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

# --- Replay Policy Class ---
import zarr
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

class ReplayDatasetPolicy(BaseImagePolicy):
    def __init__(self, dataset_path: str, n_action_steps: int = 8):
        super().__init__()
        self.dataset_path = dataset_path
        self.n_action_steps = n_action_steps
        
        print(f"[ReplayPolicy] Loading dataset from {dataset_path}...")
        try:
            # Open in read-only mode
            self.store = zarr.ZipStore(dataset_path, mode='r')
            self.root = zarr.group(self.store)
            self.data = self.root['data']
            self.meta = self.root['meta']
            self.episode_ends = self.meta['episode_ends'][:]
            print(f"[ReplayPolicy] Loaded {len(self.episode_ends)} episodes.")
        except Exception as e:
            raise RuntimeError(f"Error opening replay dataset: {e}")

        self.current_episode_idx = -1
        self.current_step_in_episode = 0
        self.episode_start_idx = 0
        self.episode_end_idx = 0
        
    def reset(self):
        """Called at the beginning of each episode"""
        self.current_episode_idx += 1
        
        # Wrap around if we run out of episodes
        if self.current_episode_idx >= len(self.episode_ends):
            print(f"[ReplayPolicy] Warning: Requested episode {self.current_episode_idx} exceeds dataset size. Looping back to 0.")
            self.current_episode_idx = 0
            
        if self.current_episode_idx == 0:
            self.episode_start_idx = 0
        else:
            self.episode_start_idx = int(self.episode_ends[self.current_episode_idx - 1])
            
        self.episode_end_idx = int(self.episode_ends[self.current_episode_idx])
        self.current_step_in_episode = 0
        
        print(f"[ReplayPolicy] Resetting for Episode {self.current_episode_idx} (Global steps {self.episode_start_idx} to {self.episode_end_idx})")

    def get_current_episode_length(self) -> int:
        return self.episode_end_idx - self.episode_start_idx

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns action dictionary with key 'action' of shape (B, T, D).
        """
        global_step = self.episode_start_idx + self.current_step_in_episode
        chunk_size = self.n_action_steps
        actions_list = []
        
        for i in range(chunk_size):
            curr_idx = min(global_step + i, self.episode_end_idx - 1)
            
            # Extract pose
            target_pos = self.data['robot0_eef_pos'][curr_idx]
            target_rot_aa = self.data['robot0_eef_rot_axis_angle'][curr_idx]
            target_gripper = self.data['robot0_gripper_width'][curr_idx]
            
            # Convert rotation to 6D
            rot_mat = R.from_rotvec(target_rot_aa).as_matrix()
            # Assuming rot6d_to_rotation_matrix's inverse format (first 2 cols flattened)
            target_rot6d = rot_mat[:3, :2].T.flatten()
            
            # Construct action vector: [pos(3), rot6d(6), gripper(1)]
            action_vec = np.zeros(10, dtype=np.float32)
            action_vec[:3] = target_pos
            action_vec[3:9] = target_rot6d
            action_vec[9] = target_gripper[0]
            
            actions_list.append(action_vec)
            
        # Stack to (1, T, D)
        action_tensor = torch.from_numpy(np.array(actions_list)).float().unsqueeze(0)
        
        # Determine device from input obs
        device = "cpu"
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
        
        # Advance step counter
        # Corresponds to how many steps the environment will execute
        self.current_step_in_episode += chunk_size
        
        return {
            "action": action_tensor.to(device)
        }


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
    """
    target_pos = 1.0 if width >= threshold else 0.0

    # Try to find indices robustly
    dof_names = panda.dof_names
    idx1 = panda.get_dof_index("panda_finger_joint1")
    idx2 = panda.get_dof_index("panda_finger_joint2")

    if (idx1 is None or idx2 is None) and dof_names is not None:
        for i, name in enumerate(dof_names):
            if name.endswith("panda_finger_joint1"): idx1 = i
            if name.endswith("panda_finger_joint2"): idx2 = i

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
        use_kinematic_control: bool = True,
        object_poses_path: Optional[str] = None,
        assets_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.task_name = task_name
        self.output_dir = output_dir  # Store output_dir for saving videos
        if self.output_dir:
            self.video_dir = os.path.join(self.output_dir, "debug_videos")
            os.makedirs(self.video_dir, exist_ok=True)
            print(f"[IsaacSimEnvironment] Video recording enabled. Output dir: {self.video_dir}")
        
        self.video_frames = []
        self.video_segment_idx = 0
        self.episode_idx = -1 # Will be incremented on reset

        self.shape_meta = shape_meta or {}
        self._obs_history: Dict[str, List[np.ndarray]] = {}

        # Pose representation configuration (from training)
        # Defaults match umi_dataset.py defaults
        self.pose_repr = pose_repr or {}
        self.obs_pose_repr = self.pose_repr.get("obs_pose_repr", "rel")
        self.action_pose_repr = self.pose_repr.get("action_pose_repr", "rel")
        self.use_kinematic_control = use_kinematic_control
        
        self.object_poses_path = object_poses_path
        
        # Resolve assets dir
        if assets_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            voilab_root = os.path.dirname(script_dir)
            self.assets_dir = os.path.join(voilab_root, "assets", "CADs")
        else:
            self.assets_dir = assets_dir

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
        self.first = False
        self.tmp = None
        import omni.kit.app

        # 獲取擴展管理器
        ext_manager = omni.kit.app.get_app().get_extension_manager()

        # 檢查並立即啟動 debug_draw 擴展
        extension_id = "isaacsim.util.debug_draw"
        if not ext_manager.is_extension_enabled(extension_id):
            ext_manager.set_extension_enabled_immediate(extension_id, True)
        try:
            self._initialize_world_and_camera()
            self._initialize_ik_solvers()
        except Exception as e:
            print(f"[IsaacSimEnvironment] Initialization error: {e}")
            import traceback

            traceback.print_exc()

    def show_cameras(env):
        rgb_wrist = env.cameras["wrist"].get_rgb()
        rgb_top   = env.cameras["top"].get_rgb()
        rgb_angle = env.cameras["angle"].get_rgb()

        # 轉成 uint8 (假設是 0-1 float)
        rgb_wrist = (rgb_wrist * 255).astype(np.uint8) if rgb_wrist.dtype==np.float32 else rgb_wrist
        rgb_top   = (rgb_top * 255).astype(np.uint8) if rgb_top.dtype==np.float32 else rgb_top
        rgb_angle = (rgb_angle * 255).astype(np.uint8) if rgb_angle.dtype==np.float32 else rgb_angle

        # RGB -> BGR
        rgb_wrist = cv2.cvtColor(rgb_wrist, cv2.COLOR_RGB2BGR)
        rgb_top   = cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR)
        rgb_angle = cv2.cvtColor(rgb_angle, cv2.COLOR_RGB2BGR)

        # 顯示
        cv2.imshow("Wrist Camera", rgb_wrist)
        cv2.imshow("Top Camera", rgb_top)
        cv2.imshow("Angle Camera", rgb_angle)

        cv2.waitKey(1)  # 1 ms 更新

    def draw_line(self, p0, p1, color=(0, 1, 0, 1), thickness=2):
        # 所有的參數都必須包裝成 List
        self.debug_draw.draw_lines(
            [tuple(p0.tolist())],      # 起點列表
            [tuple(p1.tolist())],      # 終點列表
            [color],                   # 顏色列表 (關鍵：加中括號)
            [float(thickness)],        # 粗細列表 (關鍵：加中括號)
        )

    def draw_point(self, pos, color=(1, 0, 0, 1), size=5):
        # 同理，draw_points 也需要 List
        self.debug_draw.draw_points(
            [tuple(pos.tolist())],     # 點位列表
            [color],                   # 顏色列表
            [float(size)],             # 大小列表
        )
    def draw_frame(self, pos, rot_mat, scale=0.1):
        pos = pos.reshape(3)

        x = rot_mat[:, 0]
        y = rot_mat[:, 1]
        z = rot_mat[:, 2]

        self.draw_line(pos, pos + scale * x, color=(1, 0, 0, 1), thickness=3)  # X
        self.draw_line(pos, pos + scale * y, color=(0, 1, 0, 1), thickness=3)  # Y
        self.draw_line(pos, pos + scale * z, color=(0, 0, 1, 1), thickness=3)  # Z
    def draw_trajectory(self, points, color=(1, 1, 1, 1), thickness=2, closed=False):
        pts = [tuple(p.tolist()) for p in points]
        self.debug_draw.draw_lines_spline(
            pts,
            color,
            thickness,
            closed,
        )
        
    def calculate_camera_orientation(self, eye_pos, target_pos, up_axis=np.array([0, 0, 1])):
        """
        Helper function.
        Computes the quaternion (WXYZ) for a Robotics Camera (Look=X, Up=Z)
        at eye_pos looking at target_pos.
        """
        eye_pos = np.array(eye_pos)
        target_pos = np.array(target_pos)
        
        # 1. Forward Vector (Camera X-axis)
        fwd = target_pos - eye_pos
        fwd = fwd / np.linalg.norm(fwd)
        
        # 2. Right Vector
        right = np.cross(fwd, up_axis)
        
        # Handle degenerate case (looking straight up/down)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0, 1, 0])
            
        right = right / np.linalg.norm(right)
        
        # 3. Up Vector (Camera Z-axis)
        z_axis = np.cross(right, fwd)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Y_axis (Camera Left)
        y_axis = np.cross(z_axis, fwd)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Basis: X=fwd, Y=y_axis, Z=z_axis
        R_matrix = np.column_stack((fwd, y_axis, z_axis))
        
        # Convert to quaternion (Scalar-Last xyzw -> WXYZ for Isaac Sim)
        quat_xyzw = R.from_matrix(R_matrix).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return quat_wxyz

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

        # Discovery and store gripper joint indices for robust control/obs
        self._gripper_joint_indices = []
        
        # --- Set robot pose ---
        franka_pose = self.registry_config.get("franka_pose", {})
        franka_translation = np.array(franka_pose.get("translation", [0.0, 0.0, 0.0]))
        franka_rotation = np.array(franka_pose.get("rotation_quat", [1.0, 0.0, 0.0, 0.0]))
        self.robot_xform.set_local_pose(
            translation=franka_translation / stage_utils.get_stage_units(),
            orientation=franka_rotation,
        )
        camera_translation = self.registry_config.get("camera_pose", {}).get("translation", [0, 0, 0])
        set_camera_view(camera_translation, franka_translation)

        # --- Setup Multi-Camera (UMI Style) ---
        self.cameras = {}
        res = (224, 224) # 統一解析度

        # 1. Wrist Camera
        self.cameras["wrist"] = Camera(
            prim_path=f"{GOPRO_PRIM_PATH}/Camera",
            name="wrist_camera",
            resolution=res
        )

        # 2. Top-Down Camera
        top_pos = franka_translation + np.array([0.6, 0.0, 1.8])
        top_target = franka_translation + np.array([0.6, 0.0, 0.0])
        top_quat = self.calculate_camera_orientation(top_pos, top_target, up_axis=np.array([1, 0, 0]))
        self.cameras["top"] = Camera(
            prim_path="/World/TopCamera",
            name="top_camera",
            position=top_pos,
            orientation=top_quat,
            resolution=res
        )

        # 3. Angle Camera
        angle_pos = franka_translation + np.array([1.6, -2.0, 1.3])
        angle_target = franka_translation + np.array([0.3, 0.0, 0.2])
        angle_quat = self.calculate_camera_orientation(angle_pos, angle_target, up_axis=np.array([0, 0, 1]))
        self.cameras["angle"] = Camera(
            prim_path="/World/AngleCamera",
            name="angle_camera",
            position=angle_pos,
            orientation=angle_quat,
            resolution=res
        )

        for cam in self.cameras.values():
            cam.initialize()

        self.world.reset()

        # --- Debug Draw Setup ---
        from isaacsim.util.debug_draw import _debug_draw
        try:
            self.debug_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            import isaacsim.util.debug_draw.impl as debug_impl
            self.debug_draw = debug_impl.acquire_debug_draw_interface()

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

    def apply_ik_solution(self, target_pos, target_quat_wxyz):
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
        action, success = self.art_kine_solver.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_quat_wxyz
        )

        if success:
            self.manipulator.set_joint_positions(action.joint_positions, np.arange(7))
            # Default to open gripper (0.04 per finger = 0.08 total)
            if hasattr(self, "_gripper_joint_indices") and self._gripper_joint_indices:
                self.manipulator.set_joint_positions(
                    positions=np.array([0.04] * len(self._gripper_joint_indices)),
                    joint_indices=np.array(self._gripper_joint_indices)
                )
            return True

        return False

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
        from isaacsim.core.prims import SingleXFormPrim
        import isaacsim.core.utils.prims as prims_utils
        
        base_link_path = "/World/Franka/panda/panda_link0"
        if not prims_utils.is_prim_path_valid(base_link_path):
            base_link_path = f"{self.manipulator.prim_path}/panda/panda_link0"
            
        base_link_prim = SingleXFormPrim(base_link_path)
        robot_pos, robot_quat = base_link_prim.get_world_pose()
        self.lula_solver.set_robot_base_pose(
            robot_position=robot_pos, robot_orientation=robot_quat
        )

        # Discovery and store gripper joint indices for robust control/obs
        # We step the world once to ensure articulation is fully initialized in Isaac Sim
        for _ in range(5):
            self.world.step(render=False)

        self._gripper_joint_indices = []
        dof_names = self.manipulator.dof_names
        if dof_names is not None:
            for target_name in ["panda_finger_joint1", "panda_finger_joint2"]:
                idx = self.manipulator.get_dof_index(target_name)
                if idx is None:
                    # Try suffix match (e.g. "panda/panda_finger_joint1")
                    for i, name in enumerate(dof_names):
                        if name.endswith(target_name):
                            idx = i
                            break
                if idx is not None:
                    self._gripper_joint_indices.append(idx)
            print(f"[IsaacSimEnvironment] Robot Joint Names discovered: {dof_names}")
            print(f"[IsaacSimEnvironment] Gripper joint indices discovered: {self._gripper_joint_indices}")
        else:
            print("[IsaacSimEnvironment] WARNING: dof_names is None, gripper control may fail.")
        
            

    def _get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector position and rotation matrix.

        Returns:
            (position (3,), rotation_matrix (3,3))
        """
        self.calibrate_robot_base(self.manipulator, self.lula_solver)
        ee_pos, ee_rot_matrix = self.art_kine_solver.compute_end_effector_pose()
        return ee_pos.astype(np.float64), ee_rot_matrix[:3, :3].astype(np.float64)

    def get_end_effector_pos_quat_wxyz(self, panda, lula_solver, art_kine_solver) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end-effector position and orientation as quaternion WXYZ."""
        self.calibrate_robot_base(panda, lula_solver)

        ee_pos, ee_T = art_kine_solver.compute_end_effector_pose()  # ee_T[:3,:3] rotation
        quat_xyzw = R.from_matrix(ee_T[:3, :3]).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return ee_pos.astype(np.float64), quat_wxyz.astype(np.float64)

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
        """Capture current observations from the multi-camera environment, policy-ready format."""
        obs: Dict[str, np.ndarray] = {}

        # -------------------------------------------------
        # Helper: process RGB -> CHW float32
        # -------------------------------------------------
        def _process_rgb(rgb: np.ndarray) -> np.ndarray:
            if rgb is None:
                #print("[DEBUG][_capture_observation] RGB is None")
                return None
            rgb = np.asarray(rgb)
            if rgb.ndim != 3:
                raise ValueError(f"RGB must be 3D, got shape {rgb.shape}")
            # HWC -> CHW
            if rgb.shape[-1] == 3:
                rgb = np.transpose(rgb, (2, 0, 1))
            # Add batch & time dimension -> (1,1,C,H,W)
            return rgb.astype(np.float32)[None, None, :, :, :]

        # -------------------------------------------------
        # 1. Multi-camera RGB
        # -------------------------------------------------
        if hasattr(self, "cameras"):
            for cam_name, key_name in [("angle", "angle_rgb"),
                                    ("top", "top_rgb"),
                                    ("wrist", "wrist_rgb")]:
                if cam_name in self.cameras:
                    rgb = _process_rgb(self.cameras[cam_name].get_rgb())
                    if rgb is not None:
                        obs[key_name] = rgb
                        #print(f"[DEBUG][_capture_observation] {key_name} shape: {rgb.shape}")
                    else:
                        pass
                        #print(f"[DEBUG][_capture_observation] {key_name} is None")
        elif hasattr(self, "camera") and self.camera is not None:
            rgb = _process_rgb(self.camera.get_rgb())
            if rgb is not None:
                obs["camera0_rgb"] = rgb
                #print(f"[DEBUG][_capture_observation] camera0_rgb shape: {rgb.shape}")
            else:
                pass
                #print("[DEBUG][_capture_observation] camera0_rgb is None")

        # -------------------------------------------------
        # 2. End-effector pose (EEF) low-dim
        # -------------------------------------------------
        if self.art_kine_solver is not None and self.manipulator is not None:
            ee_pos, ee_rot_mat = self._get_end_effector_pose()
            #print(f"[DEBUG][_capture_observation] EEF pos: {ee_pos}")
            #print(f"[DEBUG][_capture_observation] EEF rot_mat:\n{ee_rot_mat}")

            # Position (B,T,D) -> (1,1,3)
            obs["robot0_eef_pos"] = ee_pos.astype(np.float32)[None, None, :]

            # Rotation (axis-angle or 6D) -> (1,1,6)
            rot6d = self._rotation_matrix_to_rot6d(ee_rot_mat).astype(np.float32)
            if rot6d.ndim == 1:
                rot6d = rot6d[None, None, :]
            obs["robot0_eef_rot_axis_angle"] = rot6d
            #print(f"[DEBUG][_capture_observation] EEF rot6d: {rot6d}")

            # Relative rotation wrt start
            if self._initial_eef_rot is not None:
                rel_rot = self._initial_eef_rot.T @ ee_rot_mat
                rel_rot6d = self._rotation_matrix_to_rot6d(rel_rot).astype(np.float32)
                if rel_rot6d.ndim == 1:
                    rel_rot6d = rel_rot6d[None, None, :]
                obs["robot0_eef_rot_axis_angle_wrt_start"] = rel_rot6d
                #print(f"[DEBUG][_capture_observation] EEF rel_rot6d wrt start: {rel_rot6d}")
            else:
                obs["robot0_eef_rot_axis_angle_wrt_start"] = rot6d

            # Gripper width (robust using discovered indices)
            joint_pos = np.asarray(self.manipulator.get_joint_positions(), dtype=np.float32)
            if self._gripper_joint_indices:
                # Sum the finger positions to get the total width
                gripper_width = sum([joint_pos[idx] for idx in self._gripper_joint_indices])
            else:
                gripper_width = 0.0
            obs["robot0_gripper_width"] = np.array([gripper_width], dtype=np.float32)[None, None, :]
            #print(f"[DEBUG][_capture_observation] Gripper width: {gripper_width}")

        # -------------------------------------------------
        # 3. Shape-meta-aware fallback (zeros, batch/time dims)
        # -------------------------------------------------
        for key, meta in self.shape_meta.get("obs", {}).items():
            if key in obs:
                continue
            shape = meta.get("shape") or meta.get("raw_shape")
            if shape is None:
                continue
            obs[key] = np.zeros((1, 1, *shape), dtype=np.float32)
            #print(f"[DEBUG][_capture_observation] Fallback obs {key} shape: {obs[key].shape}")

        return obs



    def _update_history(self, obs: Dict[str, np.ndarray]) -> None:
        for key, value in obs.items():
            history = self._obs_history.setdefault(key, [])
            history.append(value)
            max_h = self._get_horizon(key)
            if len(history) > max_h:
                self._obs_history[key] = history[-max_h:]

    def _stack_history(self) -> Dict[str, np.ndarray]:
        stacked: Dict[str, np.ndarray] = {}
        for key, history in self._obs_history.items():
            horizon = self._get_horizon(key)
            if not history:
                # 用 zeros 填滿
                shape = self.shape_meta.get("obs", {}).get(key, {}).get("shape", (1,))
                data = [np.zeros(shape, dtype=np.float32) for _ in range(horizon)]
            else:
                if len(history) < horizon:
                    pad = [history[0]] * (horizon - len(history))
                    data = pad + history
                else:
                    data = history[-horizon:]

            stacked_arr = np.stack(data, axis=0)  # [horizon, ...]
            # 只移除 size=1 的維度，不要指定 axis
            stacked_arr = stacked_arr.squeeze()
            stacked[key] = stacked_arr.astype(np.float32)

        if (
            self.obs_pose_repr in ("rel", "relative")
            and "robot0_eef_pos" in stacked
            and "robot0_eef_rot_axis_angle" in stacked
        ):
            stacked = self._convert_obs_to_relative(stacked)

        return stacked

    def _convert_obs_to_relative(self, stacked: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pos = stacked["robot0_eef_pos"]  # [horizon, 3]
        rot6d = stacked["robot0_eef_rot_axis_angle"]  # [horizon, 6]

        if rot6d.size == 0:
            stacked["robot0_eef_pos"] = np.zeros_like(pos, dtype=np.float32)
            stacked["robot0_eef_rot_axis_angle"] = np.zeros_like(rot6d, dtype=np.float32)
            return stacked

        base_pos = pos[-1]  # [3]
        base_rot6d = rot6d[-1]  # [6]
        base_rot_mat = rot6d_to_rotation_matrix(base_rot6d)
        base_rot_inv = base_rot_mat.T

        rel_pos = np.zeros_like(pos)
        rel_rot6d = np.zeros_like(rot6d)

        for i in range(len(pos)):
            rel_pos[i] = pos[i] - base_pos
            curr_rot_mat = rot6d_to_rotation_matrix(rot6d[i])
            rel_rot_mat = curr_rot_mat @ base_rot_inv
            rel_rot6d[i] = self._rotation_matrix_to_rot6d(rel_rot_mat)

        stacked["robot0_eef_pos"] = rel_pos.astype(np.float32)
        stacked["robot0_eef_rot_axis_angle"] = rel_rot6d.astype(np.float32)

        return stacked


    def reset(self, seed=None, options=None) -> Dict[str, np.ndarray]:
        if self.world is None:
            raise RuntimeError("IsaacSimEnvironment is not initialized with a world.")

        self.world.reset()
        
        # [Replay Mode] Load Ground Truth Object Poses
        episode_index = options.get('episode_index') if options else None
        loaded_custom_objects = False
        
        if episode_index is not None and self.object_poses_path:
            from object_loader import load_objects_from_json
            # We assume objects are spawned under /World. 
            # Ideally we should clean up previous replay objects, but for now we rely on overwrite/move.
            # If using specific parent path in loader, use it here.
            print(f"[IsaacSimEnvironment] Loading objects for episode {episode_index} from {self.object_poses_path}")
            
            # Ensure assets_dir is available
            assets_dir = getattr(self, "assets_dir", None)
            
            load_objects_from_json(
                json_path=self.object_poses_path,
                assets_dir=assets_dir,
                world=self.world,
                episode_index=episode_index,
                cfg=self.registry_config,
            )
            loaded_custom_objects = True
            
        if not loaded_custom_objects:
            self._reposition_objects_to_default()

        # [Alignment] Step physics to settle robot base before calculating INIT_EE_POS/offsets
        # Matches generate_data.py behavior (which steps 100 times)
        for _ in range(60):
            self.world.step(render=False)

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
        
        # [Video Recording] Reset video state
        if self.output_dir:
            # Save any remaining frames from previous episode if not empty
            if self.video_frames:
                self._save_video_segment(force=True)
            
            self.video_frames = []
            self.video_segment_idx = 0
            self.episode_idx += 1
            
        return self._stack_history()

    def _save_video_segment(self, force=False):
        """Save captured frames to a video segment file."""
        if not self.output_dir or not self.video_frames:
            return

        # Save if forced (end of episode) or we hit the chunk size (e.g. 100)
        # Note: Caller checks the size before calling, so we just check valid buffer
        if self.video_frames:

            filename = f"episode_{self.episode_idx:04d}_part_{self.video_segment_idx:03d}.mp4"
            filepath = os.path.join(self.video_dir, filename)
            
            # Get dimensions from first frame
            h, w, _ = self.video_frames[0].shape
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, 30.0, (w, h))
            
            for frame in self.video_frames:
                out.write(frame)
            out.release()
            
            print(f"[IsaacSimEnvironment] Saved video segment: {filepath}")
            
            self.video_frames = [] # Clear buffer
            self.video_segment_idx += 1


    def _convert_relative_to_absolute_pose(
        self,
        rel_pos: np.ndarray,
        rel_rot_mat: np.ndarray,
        base_pos: np.ndarray,
        base_rot_mat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Position: Rotate relative displacement into world frame, then add to base_pos
        abs_pos = base_rot_mat @ rel_pos + base_pos
        
        # 2. Rotation: 
        # For local-frame relative actions (UMI default): abs_rot = base_rot @ rel_rot
        abs_rot_mat = base_rot_mat @ rel_rot_mat
        
        return abs_pos, abs_rot_mat

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        # 1. 視覺化清理
        self.debug_draw.clear_lines()
        self.debug_draw.clear_points()
        
        # 2. 核心：鎖定這一動（Chunk）開始前的「絕對基準」
        # 這樣即便物理上有微小延遲，目標點也不會跟著偏移
        start_ee_pos, start_ee_rot_mat = self._get_end_effector_pose()

        # [Safety Check] Detect NaN/Inf in robot state
        if not np.all(np.isfinite(start_ee_pos)) or not np.all(np.isfinite(start_ee_rot_mat)):
            print(f"[IsaacSimEnvironment] CRITICAL: NaN detected in EE pose: {start_ee_pos}. Ending episode failure.")
            # Construct a dummy obs (zeros) to return
            obs = self._capture_observation()
            # Mark as done-fail
            return self._stack_history(), 0.0, True, {"episode_record": obs, "ik_success": False, "error": "nan_detected"}
        
        # [DEBUG] 打印當前 EE 位置
        print(f"[DEBUG] Current EE pos: {start_ee_pos}")
        
        # 3. 預先生成整條絕對座標軌跡 (Target Trajectory)
        # 避免在 _apply_action 裡面重複計算導致邏輯不對稱
        target_trajectory = []
        current_ref_pos = start_ee_pos
        current_ref_rot = start_ee_rot_mat

        for i in range(len(action)):
            # Check if action is already absolute
            is_abs = self.pose_repr.get('action_pose_repr', 'rel') == 'abs'
            
            raw_pos = action[i, :3]
            raw_rot6d = action[i, 3:9]
            raw_rot_mat = rot6d_to_rotation_matrix(raw_rot6d)
            
            if is_abs:
                # Direct assignment for absolute actions
                abs_pos = raw_pos
                abs_rot_mat = raw_rot_mat
            else:
                # Convert relative to absolute
                abs_pos, abs_rot_mat = self._convert_relative_to_absolute_pose(
                    raw_pos,
                    raw_rot_mat,
                    current_ref_pos,
                    current_ref_rot
                )
            
            target_trajectory.append((abs_pos, abs_rot_mat, action[i, 9])) # 存入位姿與夾具寬度
            
            # 更新預測的下一點基準
            current_ref_pos, current_ref_rot = abs_pos, abs_rot_mat

        # [DEBUG] 打印第一個和最後一個 target
        if len(target_trajectory) > 0:
            first_target = target_trajectory[0][0]
            last_target = target_trajectory[-1][0]
            print(f"[DEBUG] First target pos: {first_target}, Last target pos: {last_target}")
            print(f"[DEBUG] Movement direction: {last_target - start_ee_pos}")

        # [DEBUG] Movement direction
        # if len(target_trajectory) > 0:
            # print(f"[DEBUG] Movement direction: {target_trajectory[-1][0] - start_ee_pos}")

        # --- 軌跡視覺化 (已停用以確保 Policy 輸入乾淨) ---
        # traj_points = [p[0] for p in target_trajectory]
        # if len(traj_points) > 1:
        #     self.draw_trajectory(traj_points[:8], color=(0, 1, 0, 1), thickness=3) # 改用綠色表示目標
        #     self.draw_frame(traj_points[0], target_trajectory[0][1], scale=0.05)


        # 4. 執行 (傳入預先算好的軌跡)
        success = self._apply_action_sequence(target_trajectory)

        # 5. 常規更新 (注意：world.step 已經在 _apply_action_sequence 結尾呼叫了)
        # 這裡不需要再 capture 了，因為 _apply_action_sequence 已經更新了 history
        # 我們只需要返回最後 stack 的 history
        
        # [DEBUG] 打印執行後的 EE 位置 (使用 cached last pos)
        if self._last_eef_pos is not None:
             print(f"[DEBUG] After step EE pos: {self._last_eef_pos}")
         
         # 補充: 為了保持回傳介面一致，我們取最後一次 step 的 obs (已經在 history 中)
         # 但 step() 簽章要求回傳 (obs_stacked, reward, done, info)
         # 我們從 _capture_observation() 取得最後一幀作為 info 的 episode_record (雖然有點重複)
         # 為了簡化，我們直接拿 history 的最後一筆 stack 起來

        return self._stack_history(), 0.0, False, {"episode_record": {}, "ik_success": success}


        # 5. 常規更新 (注意：world.step 已經在 _apply_action_sequence 結尾呼叫了)
        obs = self._capture_observation()
        self._update_history(obs)
        self._last_eef_pos, self._last_eef_rot = self._get_end_effector_pose()
        
        # [DEBUG] 打印執行後的 EE 位置
        print(f"[DEBUG] After step EE pos: {self._last_eef_pos}")

        # [Video Recording] Capture and accumulate frames
        print(f"[DEBUG] Video recording check: output_dir={self.output_dir}, has_cameras={hasattr(self, 'cameras')}")
        if self.output_dir and hasattr(self, "cameras"):


            try:
                # Capture from cameras (RGB is typically float or uint8, careful with range)
                # Note: get_rgb() usually returns float [0,1] or uint8 [0,255]. 
                # Isaac Sim Camera.get_rgb() returns float32 array in range [0, 1] by default? 
                # Let's check existing code usage or assume based on standard Isaac Sim.
                # Actually in show_cameras method above (line 325), it handles conversion.
                
                frames = []
                for cam_name in ["wrist", "top", "angle"]:
                    if cam_name in self.cameras:
                        rgb = self.cameras[cam_name].get_rgb()
                        if rgb is not None:
                            # print(f"[DEBUG] Got RGB from {cam_name}: shape={rgb.shape}, dtype={rgb.dtype}")


                            # Handle float [0,1] -> uint8 [0,255]
                            if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                            
                            # Convert RGB to BGR for OpenCV
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            frames.append(bgr)
                
                if len(frames) == 3:
                     # Horizontal concat: Wrist | Top | Angle
                    combined_frame = cv2.hconcat(frames)
                    self.video_frames.append(combined_frame)
                    print(f"[DEBUG] Accumulated video frames: {len(self.video_frames)}")

                    
                    # Check if we need to save segment

                    if len(self.video_frames) >= 100:
                        self._save_video_segment()
            except Exception as e:
                print(f"[Video Recording] Error capturing frame: {e}")

        return self._stack_history(), 0.0, False, {"episode_record": obs, "ik_success": success}

    def _apply_action_sequence(self, trajectory: list) -> bool:
        """
        關鍵修正：執行多步（n_action_steps），每步之後都呼叫 world.step()
        
        Policy 輸出 16 步的軌跡，設計上預期會連續執行多步。
        如果只執行第一步，位移會非常小（因為 policy 預期你會繼續執行後面的步驟）。
        正確做法：執行 n_action_steps 步，每步之後都呼叫 world.step()
        
        [OBS FIX]: 為了確保 observation history 是連續的 (t, t-1, t-2...)，
        我們必須在每一步 world.step() 之後都 capture observation 並更新 history。
        否則 history 會變成 (t, t-8, t-16...)，造成嚴重分佈偏移。
        
        [更寬容的 IK]: 即使 IK 失敗 (可能因為碰撞檢查過嚴)，只要有計算出 joint_positions，
        就強制執行。這符合使用者要求的 "permissive" 行為，避免機器人猶豫。
        """
        if len(trajectory) == 0:
            self.world.step(render=True)
            # Capture obs for no-op
            obs = self._capture_observation()
            self._update_history(obs)
            return True
        
        # 執行多步（典型 diffusion policy 設定是 8 步）
        n_action_steps = min(8, len(trajectory))
        all_success = True
        
        for step_idx in range(n_action_steps):
            target_pos, target_rot_mat, gripper_width = trajectory[step_idx]
            
            # 校準 robot base（每步都校準確保準確）
            self.calibrate_robot_base(self.manipulator, self.lula_solver)
            
            # 計算目標四元數
            # Safety for SVD convergence
            if not np.all(np.isfinite(target_rot_mat)):
                 print(f"[IK ERROR] NaN in target_rot_mat at step {step_idx}")
                 continue
                 
            target_rot_quat = R.from_matrix(target_rot_mat).as_quat(scalar_first=True)

            # IK 解算
            action, success = self.art_kine_solver.compute_inverse_kinematics(
                target_position=target_pos,
                target_orientation=target_rot_quat
            )

            # [關鍵修改] 即使 success=False，只要有動作就執行 (Permissive Mode)
            if action.joint_positions is not None:
                # 設定手臂關節（前 7 個）
                self.manipulator.set_joint_positions(action.joint_positions, np.arange(7))
                
                if not success:
                    # 雖然失敗但強制執行，記錄警告但不標記為全失敗 (視需求而定)
                    print(f"[IK WARNING] Step {step_idx} failed (collision?), FORCING execution. Target: {target_pos}")
                    # all_success = False # 選擇不標記為失敗，讓流程繼續
            else:
                all_success = False
                print(f"[IK ERROR] Step {step_idx} failed and NO solution found. Target: {target_pos}")

            # 設定夾爪
            set_gripper_width(self.manipulator, gripper_width)

            # [關鍵] 每步之後都呼叫 world.step()！
            # 並且 Capture Observation 以更新 History
            # 只在最後一步 render? 不，如果要 capture image，通常需要 render=True
            # 為了正確的 visual history，每步都需要 render。
            self.world.step(render=True)
            
            # [CRITICAL UPDATE] Capture and update history per step
            obs = self._capture_observation()
            self._update_history(obs)
            
            # Update legacy tracking
            self._last_eef_pos, self._last_eef_rot = self._get_end_effector_pose()

            # [Video Recording] Accumulate frames PER STEP
            # (Moved from step() to here so we capture full smooth video)
            if self.output_dir and hasattr(self, "cameras"):
                 try:
                    frames = []
                    for cam_name in ["wrist", "top", "angle"]:
                        if cam_name in self.cameras:
                            rgb = self.cameras[cam_name].get_rgb()
                            if rgb is not None:
                                # Handle float [0,1] -> uint8 [0,255]
                                if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                                    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                frames.append(bgr)
                    
                    if len(frames) == 3:
                        combined_frame = cv2.hconcat(frames)
                        self.video_frames.append(combined_frame)
                        if len(self.video_frames) >= 100:  # Use small threshold as requested
                             self._save_video_segment()
                 except Exception as e:
                    print(f"[Video Recording] Error capturing frame: {e}")

        # time.sleep(2) # Remove sleep to go faster
        return all_success


    def close(self) -> None:
        if self.simulation_app:
            self.simulation_app.close()
            self.simulation_app = None
    def calibrate_robot_base(self, panda, lula_solver):
        """
        Update Lula solver with current robot base pose.
        Must be called before computing IK.
        """
        if lula_solver is None:
            return

        from isaacsim.core.prims import SingleXFormPrim
        import isaacsim.core.utils.prims as prims_utils
        
        # Try both common paths for Franka link0
        base_link_path = "/World/Franka/panda/panda_link0"
        if not prims_utils.is_prim_path_valid(base_link_path):
            base_link_path = f"{panda.prim_path}/panda/panda_link0"
            
        base_link_prim = SingleXFormPrim(base_link_path)
        robot_pos, robot_quat = base_link_prim.get_world_pose()
        
        lula_solver.set_robot_base_pose(
            robot_position=robot_pos,
            robot_orientation=robot_quat
        )
    def get_end_effector_pos_quat_wxyz(self, panda, lula_solver, art_kine_solver):
        self.calibrate_robot_base(panda, lula_solver)

        ee_pos, ee_T = art_kine_solver.compute_end_effector_pose()  # ee_T[:3,:3] rotation
        quat_xyzw = R.from_matrix(ee_T[:3, :3]).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return ee_pos.astype(np.float64), quat_wxyz.astype(np.float64)
    def _set_to_init_pose(self):
        print(f'[Init] Setting to init pose for task: {self.task_name}')

        self.calibrate_robot_base(self.manipulator, self.lula_solver)

        # 1. 取得當前 EE 位置（用於計算 task 特定偏移）
        curr_pos, _ = self.get_end_effector_pos_quat_wxyz(
            self.manipulator, self.lula_solver, self.art_kine_solver
        )

        # 2. 根據 task 設定固定初始 EE 位姿
        if self.task_name == "kitchen":
            INIT_EE_POS = curr_pos + np.array([-0.16, 0., 0.13])
            INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
        elif self.task_name == "dining-room":
            INIT_EE_POS = curr_pos + np.array([-0.16, 0., 0.13])
            INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
        elif self.task_name == "living-room":
            INIT_EE_POS = curr_pos + np.array([-0.1, 0.2, 0.20])
            INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
        else:
            raise RuntimeError(f"Unknown task, expected one of 'kitchen', 'dining-room', 'living-room', got {self.task_name}")

        # 3. IK 與設定
        success = self.apply_ik_solution(
            INIT_EE_POS,
            INIT_EE_QUAT_WXYZ,
        )

        if success:
            print(f"[Init] Success: Moved EE to INIT_EE_POS {INIT_EE_POS}")
        else:
            print("=" * 50)
            print("[IsaacSimEnvironment] WARNING: Failed to apply initial EE pose")
            print(f"  Target Pos: {INIT_EE_POS}, Target Quat WXYZ: {INIT_EE_QUAT_WXYZ}")
            print(f"  Current Pos: {curr_pos}")
            print("Proceeding with current pose...")
            print("=" * 50)



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
    parser.add_argument(
        "--replay_dataset",
        type=str,
        default=None,
        nargs="?",
        const=None,
        help="Path to Zarr dataset for ground truth replay. Overrides checkpoint policy.",
    )
    parser.add_argument(
        "--object_poses",
        type=str,
        default=None,
        help="Path to object_poses.json for faithful scene reconstruction in replay.",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=None,
        help="Directory containing USD assets for objects.",
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

    # [Replay Mode Override]
    if args.replay_dataset:
        print(f"[Replay Mode] Overriding pose_repr to absolute for dataset replay.")
        pose_repr = {'action_pose_repr': 'abs', 'obs_pose_repr': 'abs'}

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
        output_dir=str(output_dir), # Pass output_dir
    )

    # 2. Instantiate Policy and move to GPU (after SimApp is running)
    print("Instantiating Policy...")
    
    state_dicts = {}
    
    if args.replay_dataset:
        print(f"[Replay Mode] Instantiating ReplayDatasetPolicy with dataset: {args.replay_dataset}")
        # Assuming n_action_steps is available in runner_cfg, otherwise default to 8
        n_action_steps = runner_cfg.get("n_action_steps", 8)
        policy = ReplayDatasetPolicy(args.replay_dataset, n_action_steps=n_action_steps)
        # Replay policy handles its own device placement (usually CPU for simple lookup)
    else:
        policy_cfg = _to_container(getattr(cfg, "policy", {}))
        if shape_meta is not None:
            policy_cfg.setdefault("shape_meta", shape_meta)

        policy = _instantiate(policy_cfg)

        state_dicts = payload.get("state_dicts", {})
        if "model" in state_dicts:
            policy.load_state_dict(state_dicts["model"], strict=False)

    ema_policy = None
    if (
        not args.replay_dataset # [Replay Mode] Skip EMA
        and getattr(getattr(cfg, "training", None), "use_ema", False)
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

    # [Replay Mode Override]
    if args.replay_dataset and isinstance(active_policy, ReplayDatasetPolicy):
        total_episodes = len(active_policy.episode_ends)
        print(f"[Replay Mode] Overriding n_episodes to {total_episodes} (Full Dataset)")
        runner_kwargs['n_episodes'] = total_episodes
        # Ensure long episodes are not truncated
        runner_kwargs['max_steps_per_episode'] = 10000

    # Inject the pre-initialized environment and task_name for registry lookup
    runner_kwargs["env"] = isaac_env
    runner_kwargs["task_name"] = args.task

    # Instantiate the runner
    env_runner = _instantiate(
        {"_target_": runner_target, **runner_kwargs, "output_dir": str(output_dir)}
    )

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
