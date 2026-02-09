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
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)

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

import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R


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


def init_environment(task_name, cfg):
    assert hasattr(cfg, "shape_meta"), (
        "Missing shape_meta attribute from cfg. Please check your checkpoint."
    )

    # Get registry config FIRST (before it's used for robot/camera setup)
    registry_class = registry.get_task_registry(task_name)
    if not registry_class.validate_environment():
        logger.warning(f"Registry validation failed for {task_name}")
    registry_config = registry_class.get_config()

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
    return world, manipulator, camera


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

    world, manipulator, camera = init_environment(args.task, cfg)

    # Initialize step counter for debug image saving
    inference_step_counter = 0

    # Initialize IK solvers
    lula_solver, art_kine_solver = initialize_ik_solvers(manipulator)

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

    time.sleep(5)

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
