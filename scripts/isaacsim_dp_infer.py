"""
Isaac Sim Inference for UMI Diffusion Policy.

Usage:
    python isaacsim_dp_infer.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --task dining-room \
        --num_steps 1000 \
        --session_dir /path/to/session_dir \
        --episode_idx 0 \
        --steps_per_inference 1 \
"""

import os
import argparse
import numpy as np
import time
import sys

# -----------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
voilab_root = os.path.dirname(current_dir)
policy_source_root = os.path.join(voilab_root, "packages", "diffusion_policy", "src")
if policy_source_root not in sys.path:
    sys.path.append(policy_source_root)
    print(f"[System] Added to sys.path: {policy_source_root}")
# -----------------------------------------------------

import json
import cv2
import torch
import hydra
import dill
from omegaconf import OmegaConf
from collections import deque
from scipy.spatial.transform import Rotation as R

# --- Isaac Sim Initialization ---
from isaacsim import SimulationApp

# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)")
parser.add_argument("--task", type=str, default="dining-room", choices=["kitchen", "dining-room", "living-room"])
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--num_steps", type=int, default=1000, help="Max simulation steps")
parser.add_argument("--device", type=str, default="cuda:0")
# Arguments for loading real object poses (inherited from scene loader)
parser.add_argument("--session_dir", type=str, default=None, help="Path to session directory")
parser.add_argument("--episode_idx", type=int, default=0, help="Index of the episode")
parser.add_argument("--steps_per_inference", type=int, default=1, help="Action steps to execute per inference (Action Chunking)")

args = parser.parse_args()

config = {
    "headless": args.headless,
    "width": 1280,
    "height": 720,
    "extensions": ["isaacsim.robot_motion.motion_generation"]
}
simulation_app = SimulationApp(config)

# --- Standard Imports after Isaac Sim ---
import omni.usd
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils import prims
from isaacsim.util.debug_draw import _debug_draw

# Project imports
import registry
from umi_replay import set_gripper_width
from object_loader import load_object_transforms_from_json

# Check assets
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("[Error] Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

enable_extension("isaacsim.robot_motion.motion_generation")

# --- Constants & Configs ---
BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/franka_panda/franka_panda_arm.usd"
FRANKA_PANDA_PRIM_PATH = "/World/Franka"
GOPRO_PRIM_PATH = "/World/Franka/panda/panda_link7/gopro_link"
ASSETS_DIR = "/workspace/voilab/assets/CADs"
LULA_ROBOT_DESCRIPTION_PATH = "/workspace/voilab/assets/lula/frank_umi_descriptor.yaml"
LULA_URDF_PATH = "/workspace/voilab/assets/franka_panda/franka_panda_umi-isaacsim.urdf"

# =============================================================================
# Math & Transformation Helpers
# =============================================================================

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

def calculate_camera_orientation(eye_pos, target_pos, up_axis=np.array([0, 0, 1])):
    # Copied from scene loader
    eye_pos = np.array(eye_pos)
    target_pos = np.array(target_pos)
    fwd = target_pos - eye_pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up_axis)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0, 1, 0])
    right = right / np.linalg.norm(right)
    z_axis = np.cross(right, fwd)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, fwd)
    y_axis = y_axis / np.linalg.norm(y_axis)
    R_matrix = np.column_stack((fwd, y_axis, z_axis))
    quat_xyzw = R.from_matrix(R_matrix).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz

def _normalize_object_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

# =============================================================================
# Helper Class: Policy Wrapper
# =============================================================================

class PolicyWrapper:
    
    def __init__(self, checkpoint_path, device="cuda:0"):
        print(f"[Policy] Loading checkpoint: {checkpoint_path}")
        self.device = torch.device(device)
        
        # Load checkpoint payload using dill
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = payload['cfg']
        
        # Instantiate Policy
        self.policy = hydra.utils.instantiate(self.cfg.policy)
        
        # Load State Dict
        if 'state_dicts' in payload:
             self.policy.load_state_dict(payload['state_dicts']['model'])
        else:
             self.policy.load_state_dict(payload['state_dict'])
        
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
        print(f"[Policy] Target Image Shape: {self.target_w}x{self.target_h} (Channels: {self.target_c})")
        
        # Save as absolute coord buffer
        self.obs_buffer_abs = deque(maxlen=self.obs_horizon)
        

    def reset(self):
        self.obs_buffer_abs.clear()

    def get_action(self, obs_dict):
        """
        Input observation at current step (absolute coords), store in buffer,
        transform to relative coords for model inference, then convert predicted
        actions back to absolute coords.
        
        Args:
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
        img = obs_dict['camera0_rgb']
        img = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) 
        img = np.moveaxis(img, -1, 0) / 255.0  # (C, H, W), [0,1]
        
        # 2. Save Curr observation to buffer (absolute coords)
        current_frame_abs = {
            'camera0_rgb': img,
            'robot0_eef_pos': obs_dict['robot0_eef_pos'].astype(np.float32).copy(),
            'robot0_eef_rot_axis_angle': obs_dict['robot0_eef_rot_axis_angle'].astype(np.float32).copy(),
            'robot0_eef_rot_axis_angle_wrt_start': obs_dict['robot0_eef_rot_axis_angle_wrt_start'].astype(np.float32).copy(),
            'robot0_gripper_width': obs_dict['robot0_gripper_width'].astype(np.float32).copy()
        }
        
        self.obs_buffer_abs.append(current_frame_abs)
        
        # Padding if buffer not full
        while len(self.obs_buffer_abs) < self.obs_horizon:
            self.obs_buffer_abs.appendleft(self.obs_buffer_abs[0])
        
        # 3. Use buffer[-1] as refernce, tranform to related coord
        ref_frame = self.obs_buffer_abs[-1]
        ref_pos = ref_frame['robot0_eef_pos']
        ref_rot_aa = ref_frame['robot0_eef_rot_axis_angle']
        ref_pose6 = np.concatenate([ref_pos, ref_rot_aa])
        ref_mat = pose_to_mat(ref_pose6)  # (4, 4)
        ref_mat_inv = np.linalg.inv(ref_mat)
        
        # Transform all frames to relative coordinate
        batch_obs_rel = {
            'camera0_rgb': [],
            'robot0_eef_pos': [],
            'robot0_eef_rot_axis_angle': [],
            'robot0_eef_rot_axis_angle_wrt_start': [],
            'robot0_gripper_width': []
        }
        
        for frame in self.obs_buffer_abs:
            batch_obs_rel['camera0_rgb'].append(frame['camera0_rgb'])
            
            # Convert to relative pose
            frame_pos = frame['robot0_eef_pos']
            frame_rot_aa = frame['robot0_eef_rot_axis_angle']
            frame_pose6 = np.concatenate([frame_pos, frame_rot_aa])
            frame_mat = pose_to_mat(frame_pose6)
            
            # T_rel = inv(T_ref) @ T_frame
            rel_mat = ref_mat_inv @ frame_mat
            rel_pose10d = mat_to_pose10d(rel_mat)  # (10,) = pos(3) + rot6d(6)
            
            batch_obs_rel['robot0_eef_pos'].append(rel_pose10d[:3])
            batch_obs_rel['robot0_eef_rot_axis_angle'].append(rel_pose10d[3:9])  # 6D
            
            # wrt_start
            batch_obs_rel['robot0_eef_rot_axis_angle_wrt_start'].append(
                frame['robot0_eef_rot_axis_angle_wrt_start']
            )
            batch_obs_rel['robot0_gripper_width'].append(frame['robot0_gripper_width'])
        
        # 4. Stack to Tensor
        batch_obs_tensor = {}
        for key, val_list in batch_obs_rel.items():
            val_stack = np.stack(val_list, axis=0)  # (T, ...)
            val_tensor = torch.from_numpy(val_stack).unsqueeze(0).to(self.device)  # (1, T, ...)
            batch_obs_tensor[key] = val_tensor
        
        # 5. Inference
        with torch.no_grad():
            result = self.policy.predict_action(batch_obs_tensor)
            action_tensor = result['action_pred']  # (1, T_action, 10)
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
            
            actions_abs.append({
                'pos': action_mat_abs[:3, 3].copy(),
                'rot_mat': action_mat_abs[:3, :3].copy(),
                'gripper': float(action_gripper)
            })
        
        return actions_abs, ref_mat, actions_rel


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"[Main] Starting Diffusion Inference on task: {args.task}")
    
    # --- 1. Load Registry & Config ---
    registry_class = registry.get_task_registry(args.task)
    cfg = registry_class.get_config()
    
    franka_pose = cfg.get("franka_pose", {})
    franka_translation = np.array(franka_pose.get("translation", [0, 0, 0]))
    franka_rotation = np.array(franka_pose.get("rotation_quat", [1, 0, 0, 0])) # wxyz
    # Convert wxyz to matrix for transforms
    r_base = R.from_quat([franka_rotation[1], franka_rotation[2], franka_rotation[3], franka_rotation[0]])
    franka_rot_mat = r_base.as_matrix()
    
    camera_translation = cfg.get("camera_pose", {}).get("translation", [0, 0, 0])
    aruco_tag_pose = cfg.get("aruco_tag_pose", {})

    # --- 2. Setup Isaac Sim World ---
    stage_utils.open_stage(BASE_SCENE_FP)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # --- 3. Setup Robot ---
    robot = stage_utils.add_reference_to_stage(usd_path=FRANKA_PANDA_FP, prim_path=FRANKA_PANDA_PRIM_PATH)
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")
    robot_xform = SingleXFormPrim(prim_path=FRANKA_PANDA_PRIM_PATH)
    
    gripper = ParallelGripper(
        end_effector_prim_path=FRANKA_PANDA_PRIM_PATH + "/panda/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.02, 0.02]),
        action_deltas=np.array([0.01, 0.01]),
    )
    panda = world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PANDA_PRIM_PATH,
            name="my_franka",
            end_effector_prim_path=FRANKA_PANDA_PRIM_PATH + "/panda/panda_rightfinger",
            gripper=gripper,
        )
    )
    panda.gripper.set_default_state(panda.gripper.joint_opened_positions)
    
    # Move robot to configured location
    robot_xform.set_local_pose(
        translation=franka_translation,
        orientation=franka_rotation
    )
    set_camera_view(camera_translation, franka_translation)

    # --- 4. Setup Cameras ---
    cameras = {}
    cameras["wrist"] = Camera(prim_path=f"{GOPRO_PRIM_PATH}/Camera", name="wrist_camera", resolution=(224, 224))
    cameras["top"] = Camera(prim_path="/World/TopCamera", name="top_camera", resolution=(224, 224)) 
    # Initialize top camera pose
    top_cam_pos = franka_translation + np.array([0.4, 0.0, 1.6])
    top_cam_target = franka_translation + np.array([0.4, 0.0, 0.0])
    top_cam_quat = calculate_camera_orientation(top_cam_pos, top_cam_target, up_axis=np.array([1, 0, 0]))
    cameras["top"].set_world_pose(position=top_cam_pos, orientation=top_cam_quat)

    for cam in cameras.values(): cam.initialize()

    # --- 5. Load Objects ---
    print("\n[Init] Loading objects...")
    preload_objects = cfg.get("environment_vars", {}).get("PRELOAD_OBJECTS", [])
    object_prims = {}
    
    for entry in preload_objects:
        raw_name = entry.get("name")
        asset_filename = entry.get("assets")
        prim_path = entry.get("prim_path")
        full_asset_path = os.path.join(ASSETS_DIR, asset_filename)
        if not os.path.exists(full_asset_path): continue
        try:
            stage_utils.add_reference_to_stage(usd_path=full_asset_path, prim_path=prim_path)
            default_pos = entry.get("position", [0.5, 0.0, 0.8])
            obj_prim = SingleXFormPrim(prim_path=prim_path, name=_normalize_object_name(raw_name), position=np.array(default_pos))
            world.scene.add(obj_prim)
            object_prims[_normalize_object_name(raw_name)] = obj_prim
        except Exception as e: print(e)

    # Apply Poses from JSON
    if args.session_dir:
        object_poses_path = os.path.join(args.session_dir, 'object_poses.json')
        if os.path.exists(object_poses_path):
            try:
                object_transforms = load_object_transforms_from_json(
                    object_poses_path, args.episode_idx, aruco_tag_pose, cfg
                )
                for obj in object_transforms:
                    object_name = _normalize_object_name(obj["object_name"])
                    if object_name == "plate": continue
                    if object_name in object_prims:
                        obj_prim = object_prims[object_name]
                        obj_pos = np.array(obj["position"], dtype=np.float64)
                        obj_prim.set_world_pose(position=obj_pos)
                        obj_prim.set_default_state(position=obj_pos)
                        print(f"  -> Placed {object_name} at {obj_pos}")
                    else:
                        print(f"  -> WARNING: JSON contains {object_name}, but it was not preloaded in the scene.")
            except Exception as e: print(e)

    # --- 6. Init Kinematics ---
    world.reset() # This applies default states
    
    # Re-apply robot pose after reset just in case
    robot_xform.set_local_pose(translation=franka_translation, orientation=franka_rotation)
    
    lula_solver = LulaKinematicsSolver(robot_description_path=LULA_ROBOT_DESCRIPTION_PATH, urdf_path=LULA_URDF_PATH)
    art_kine_solver = ArticulationKinematicsSolver(panda, kinematics_solver=lula_solver, end_effector_frame_name="umi_tcp")

    # --- 7. Load Policy ---
    policy_wrapper = PolicyWrapper(args.checkpoint, device=args.device)

    # --- 8. Inference Loop ---
    
    # Initialize Robot to starting pose
    for _ in range(10):
        world.step(render=True)
    
    # Get EE position (World Frame)
    cur_base_pos, cur_base_quat = panda.get_world_pose()
    lula_solver.set_robot_base_pose(cur_base_pos, cur_base_quat)
    curr_pos, _ = art_kine_solver.compute_end_effector_pose()
    curr_pos = curr_pos.astype(np.float64)
    
    # Set initial EE pose based on task
    if args.task == "kitchen":
        INIT_EE_POS = curr_pos + np.array([-0.16, 0., 0.13])
        INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
    elif args.task == "dining-room":
        INIT_EE_POS = curr_pos + np.array([-0.16, 0., 0.13])
        INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
    elif args.task == "living-room":
        INIT_EE_POS = curr_pos + np.array([-0.1, 0.2, 0.20])
        INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
    else:
        raise RuntimeError(f"Unknown task: {args.task}")
    
    # Move to initial EE pose by IK
    ik_action, success = art_kine_solver.compute_inverse_kinematics(
        target_position=INIT_EE_POS,
        target_orientation=INIT_EE_QUAT_WXYZ
    )
    
    if success:
        panda.set_joint_positions(ik_action.joint_positions, np.arange(7))
        print(f"[Init] Moved EE to initial position: {INIT_EE_POS}")
    else:
        print(f"[Init] WARNING: Failed to move to initial EE position")
    set_gripper_width(panda, width=0.08) # Open gripper

    for _ in range(30):
        world.step(render=True)
    
    print(f"[Init] Robot initialized for task: {args.task}")
    
    # load success check
    is_episode_completed = registry_class.is_episode_completed
    
    # get monitored object
    monitored_objects = ["fork", "knife", "plate"]
    monitored_prims = {}
    for name in monitored_objects:
        normalized_name = _normalize_object_name(name)
        if normalized_name in object_prims:
            monitored_prims[normalized_name] = object_prims[normalized_name]
            print(f"[Loop Init] Monitoring object: {normalized_name}")
    
    
    print("\n[Main] Starting Inference Loop...")
    for _ in range(50): # Settle
        world.step(render=True)
        
    step_count = 0
    eef_pos_list = []
    eef_rot_list = []
    gripper_list = []
    episode_start_pose = None
    episode_start_mat_base = None   # based frame matrix at start
    
    while simulation_app.is_running() and step_count < args.num_steps:
        
        # --- A. Get Observations (Sense) ---
        
        # Print status every 50 steps
        if step_count % 50 == 0:        
            for name, prim in monitored_prims.items():
                pos, _ = prim.get_world_pose()
                print(f"  Object {name:<6}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # 1. Update solver with robot base (World Frame)
        cur_base_pos, cur_base_quat = panda.get_world_pose() # base pose in world frame
        lula_solver.set_robot_base_pose(cur_base_pos, cur_base_quat)
        
        # 2. Get EE(End Effector) Pose in World Frame
        ee_pos_world, ee_rot_mat_world = art_kine_solver.compute_end_effector_pose() # end effector pose in world frame
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
        ee_rot_axis_angle_for_model = R.from_matrix(ee_rot_mat_for_model).as_rotvec().astype(np.float32)
        
        # 5. Get Gripper state
        joint_pos = panda.get_joint_positions()
        gripper_width = joint_pos[-2] + joint_pos[-1]
        
        # 5.5 Get EE Rotation relative to start (World Frame)
        curr_ee_mat_world = np.eye(4)
        curr_ee_mat_world[:3, 3] = ee_pos_world
        curr_ee_mat_world[:3, :3] = ee_rot_mat_world
        
        if episode_start_mat_base is None:  # Save the first frame ee pose as base
            episode_start_mat_base = curr_ee_mat_world.copy()
        
        rel_mat = np.linalg.inv(episode_start_mat_base) @ curr_ee_mat_world # relative to start
        rel_rot_mat = rel_mat[:3, :3]
        rel_rot_6d = matrix_to_rot6d(rel_rot_mat)   # relative rot6d to start
            
        # 6. Get Image (Wrist Camera)
        wrist_img = cameras["wrist"].get_rgb() # (H, W, C)
        
        # Error handling: If image is missing, step simulation and retry
        if wrist_img is None: 
            world.step(render=True) 
            continue
        
        # Remove alpha channel if present
        if wrist_img.shape[-1] == 4: wrist_img = wrist_img[..., :3]

        # --- B. Predict Action ---
        obs_dict = {
            'camera0_rgb': wrist_img,
            'robot0_eef_pos': ee_pos_for_model,  # World Frame
            'robot0_eef_rot_axis_angle': ee_rot_axis_angle_for_model,  # World Frame
            'robot0_eef_rot_axis_angle_wrt_start': rel_rot_6d.astype(np.float32),
            'robot0_gripper_width': np.array([gripper_width], dtype=np.float32)
        }
        
        # Model Inference 
        actions_abs, ref_mat, raw_actions = policy_wrapper.get_action(obs_dict) # return absolute coord actions
        
        # --- C. Execute Actions (Act) ---
        n_execution = min(args.steps_per_inference, len(actions_abs))

        for i in range(n_execution):
            if not simulation_app.is_running(): break
            
            # Get action in absolute coord
            action = actions_abs[i]
            target_pos_base = action['pos']
            target_rot_mat_base = action['rot_mat']
            action_gripper = action['gripper']
            
            # ================= [DEBUG START] =================
            if i == 0 and step_count % 2 == 0:
                print(f"\n{'='*60}")
                print(f"  DEBUG Step {step_count}")
                print(f"{'='*60}")
                
                # --- Raw Model Output ---
                print(f"\n[Raw Model Output]")
                raw_action_i = raw_actions[i]
                print(f"  Position (rel):  [{raw_action_i[0]:.4f}, {raw_action_i[1]:.4f}, {raw_action_i[2]:.4f}]")
                print(f"  Rotation (6D):   [{raw_action_i[3]:.3f}, {raw_action_i[4]:.3f}, {raw_action_i[5]:.3f}, {raw_action_i[6]:.3f}, {raw_action_i[7]:.3f}, {raw_action_i[8]:.3f}]")
                print(f"  Gripper (raw):   {raw_action_i[9]:.4f}")
                
                # --- EE Position (World Frame now!) ---
                target_pos_world_debug = target_pos_base  # 其實是 World Frame
                delta = target_pos_world_debug - ee_pos_world
                print(f"\n[EE Position (World Frame)]")
                print(f"  Current:  [{ee_pos_world[0]:.4f}, {ee_pos_world[1]:.4f}, {ee_pos_world[2]:.4f}]")
                print(f"  Target:   [{target_pos_world_debug[0]:.4f}, {target_pos_world_debug[1]:.4f}, {target_pos_world_debug[2]:.4f}]")
                print(f"  Delta:    [{delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}]")
                print(f"  Delta Magnitude: {np.linalg.norm(delta)*1000:.2f} mm")
                
                # --- Gripper ---
                print(f"\n[Gripper]")
                print(f"  Current Width: {gripper_width:.4f} m ({gripper_width*1000:.1f} mm)")
                print(f"  Target Width:  {action_gripper:.4f} m ({action_gripper*1000:.1f} mm)")
                print(f"  Action:        {'CLOSING' if action_gripper < gripper_width else 'OPENING' if action_gripper > gripper_width else 'HOLD'}")
                
                # --- Rotation ---
                target_rot_mat_world_debug = target_rot_mat_base  # 其實是 World Frame
                current_euler = R.from_matrix(ee_rot_mat_world).as_euler('xyz', degrees=True)
                target_euler = R.from_matrix(target_rot_mat_world_debug).as_euler('xyz', degrees=True)
                print(f"\n[EE Rotation (World Frame, Euler XYZ deg)]")
                print(f"  Current: [{current_euler[0]:.1f}, {current_euler[1]:.1f}, {current_euler[2]:.1f}]")
                print(f"  Target:  [{target_euler[0]:.1f}, {target_euler[1]:.1f}, {target_euler[2]:.1f}]")
                
                # --- Object Info ---
                print(f"\n[Objects]")
                fork_found = False
                for obj_name in ["fork", "knife", "plate"]:
                    if obj_name in monitored_prims:
                        obj_pos_world, _ = monitored_prims[obj_name].get_world_pose()
                        
                        vec_to_obj = obj_pos_world - ee_pos_world
                        dist_to_obj = np.linalg.norm(vec_to_obj)
                        
                        print(f"  {obj_name}:")
                        print(f"    Position (world): [{obj_pos_world[0]:.3f}, {obj_pos_world[1]:.3f}, {obj_pos_world[2]:.3f}]")
                        print(f"    Vector from EE:  [{vec_to_obj[0]:.3f}, {vec_to_obj[1]:.3f}, {vec_to_obj[2]:.3f}]")
                        print(f"    Distance:        {dist_to_obj*100:.1f} cm")
                        
                        # Calculate angle between delta and vec_to_obj (objective direction)
                        if obj_name == "fork":
                            fork_found = True
                            if np.linalg.norm(delta) > 1e-6:
                                vec_norm = vec_to_obj / (np.linalg.norm(vec_to_obj) + 1e-8)
                                delta_norm = delta / (np.linalg.norm(delta) + 1e-8)
                                cos_angle = np.clip(np.dot(vec_norm, delta_norm), -1, 1)
                                angle_deg = np.arccos(cos_angle) * 180 / np.pi
                                
                                # Direction status
                                direction_status = '✓ GOOD' if angle_deg < 45 else '⚠️ DRIFT' if angle_deg < 90 else '❌ WRONG'
                                print(f"    Angle (Delta→Fork): {angle_deg:.1f}° {direction_status}")
                            else:
                                print(f"    Angle: N/A (Delta too small)")
                
                if not fork_found:
                    print(f"  [WARNING] Fork not in monitored_prims!")
                    print(f"  monitored_prims keys: {list(monitored_prims.keys())}")
                
                # --- Direction Analysis ---
                print(f"\n[Direction Analysis]")
                print(f"  To reach fork, EE should move:")
                if fork_found:
                    print(f"    X: {'BACKWARD (-)' if vec_to_obj[0] < 0 else 'FORWARD (+)'} by {abs(vec_to_obj[0])*100:.1f} cm")
                    print(f"    Y: {'LEFT (-)' if vec_to_obj[1] < 0 else 'RIGHT (+)'} by {abs(vec_to_obj[1])*100:.1f} cm")
                    print(f"    Z: {'DOWN (-)' if vec_to_obj[2] < 0 else 'UP (+)'} by {abs(vec_to_obj[2])*100:.1f} cm")
                print(f"  Model predicts:")
                print(f"    X: {'BACKWARD' if delta[0] < 0 else 'FORWARD'} by {abs(delta[0])*1000:.2f} mm")
                print(f"    Y: {'LEFT' if delta[1] < 0 else 'RIGHT'} by {abs(delta[1])*1000:.2f} mm")
                print(f"    Z: {'DOWN' if delta[2] < 0 else 'UP'} by {abs(delta[2])*1000:.2f} mm")
                
                # --- wrt_start info ---
                print(f"\n[Relative to Episode Start]")
                print(f"  rot6d_wrt_start: [{rel_rot_6d[0]:.3f}, {rel_rot_6d[1]:.3f}, {rel_rot_6d[2]:.3f}, {rel_rot_6d[3]:.3f}, {rel_rot_6d[4]:.3f}, {rel_rot_6d[5]:.3f}]")
                
                print(f"{'='*60}\n")
                
                # [DEBUG: Save Image]
                debug_img_path = os.path.join(current_dir, "debug_images", f"debug_view_step{step_count}.png")
                if wrist_img.shape[-1] == 3:
                    save_img = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                else:
                    save_img = wrist_img
                cv2.imwrite(debug_img_path, save_img)

            # ================= [DEBUG END] =================
            
            # --- Prepare Target Pose ---
            target_pos_world = target_pos_base.copy()  # at World Frame
            target_rot_mat_world = target_rot_mat_base.copy()  # at World Frame
            
            # Matrix -> Quaternion (WXYZ)
            target_quat_xyzw = R.from_matrix(target_rot_mat_world).as_quat()
            target_quat_wxyz = np.array([target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]])
            
            # --- Apply Action (IK) ---
            ik_action, success = art_kine_solver.compute_inverse_kinematics(
                target_position=target_pos_world,
                target_orientation=target_quat_wxyz
            )
            
            if success:
                panda.set_joint_positions(ik_action.joint_positions, np.arange(7))
            else:
                print(f"[Warning] IK Failed at step {step_count}")
                
            # Apply Gripper command
            set_gripper_width(panda, action_gripper)
            
            # [Important] Step the physics engine forward for one tick
            world.step(render=True)

            # ----------------------------------------------------
            # Record Data
            rec_pos, rec_rot_mat = art_kine_solver.compute_end_effector_pose()
            rec_rot_vec = R.from_matrix(rec_rot_mat[:3, :3]).as_rotvec()
            rec_joint = panda.get_joint_positions()
            rec_grip = rec_joint[-2] + rec_joint[-1]
            
            eef_pos_list.append(rec_pos.astype(np.float32))
            eef_rot_list.append(rec_rot_vec.astype(np.float32))
            gripper_list.append(np.array([rec_grip], dtype=np.float32))
            # ----------------------------------------------------
            
            step_count += 1
            if step_count % 50 == 0:
                print(f"Step {step_count}: Target {target_pos_base} | Grip {action_gripper:.3f}")


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
        "episode_idx": args.episode_idx,
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

        print("====================================================================")
        if episode_success:
            print(f"[Main] Episode {args.episode_idx}: Task SUCCESS")
        else:
            print(f"[Main] Episode {args.episode_idx}: Task FAIL")
        print("====================================================================")
    except Exception as e:
        print(f"[Main] Error checking success: {e}")
        print("Make sure registry.is_episode_completed works with the recorded data format.")

    simulation_app.close()

if __name__ == "__main__":
    main()