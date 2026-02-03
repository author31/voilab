"""
Debug script to replay a UMI dataset in Isaac Sim using the runner environment.
This verifies if the action execution and coordinate transformations are correct.
"""

import sys
import os
import numpy as np
import zarr
import torch
from scipy.spatial.transform import Rotation as R

# Add directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
voilab_root = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(os.path.join(voilab_root, "packages/diffusion_policy/src"))
sys.path.append(os.path.join(voilab_root, "packages/umi/src"))

from run_isaacsim_eval import IsaacSimEnvironment, rot6d_to_rotation_matrix

def rotation_matrix_to_rot6d(rot_mat: np.ndarray) -> np.ndarray:
    # UMI uses the first two columns of the rotation matrix
    return rot_mat[:3, :2].T.flatten()

def main():
    # 1. Dataset Path
    dataset_path = "./video/raw_videos/simulation_dataset.zarr.zip"
    print(f"Loading dataset from {dataset_path}...")
    
    # Open dataset
    try:
        store = zarr.ZipStore(dataset_path, mode='r')
        root = zarr.group(store)
    except Exception as e:
        print(f"Error opening zarr: {e}")
        return

    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    # 2. Pick an episode to replay
    episode_idx = 0
    start_idx = 0 if episode_idx == 0 else int(episode_ends[episode_idx-1])
    end_idx = int(episode_ends[episode_idx])
    print(f"Replaying episode {episode_idx} (steps {start_idx} to {end_idx}, total {end_idx-start_idx} steps)")
    
    # 3. Initialize Environment
    # We use action_pose_repr='abs' because we will feed absolute poses from the dataset
    env = IsaacSimEnvironment(
        task_name='kitchen',
        sim_config={'headless': False},
        shape_meta={'obs': {}}, # dummy shape_meta
        pose_repr={'action_pose_repr': 'abs', 'obs_pose_repr': 'abs'},
        use_kinematic_control=True
    )
    
    # 4. Reset to match initial state
    obs = env.reset()
    
    # 5. Debug Base Pose
    from isaacsim.core.prims import SingleXFormPrim
    import isaacsim.core.utils.prims as prims_utils
    
    base_link_path = "/World/Franka/panda/panda_link0"
    if not prims_utils.is_prim_path_valid(base_link_path):
        base_link_path = f"{env.manipulator.prim_path}/panda/panda_link0"
        
    base_link_prim = SingleXFormPrim(base_link_path)
    articulation_pos, articulation_quat = env.manipulator.get_world_pose()
    base_link_pos, base_link_quat = base_link_prim.get_world_pose()
    print(f"[Debug] Articulation Prim Path: {env.manipulator.prim_path}")
    print(f"[Debug] Articulation Prim Pos: {articulation_pos}")
    print(f"[Debug] Base Link Path: {base_link_path}")
    print(f"[Debug] Base Link Pos (panda_link0): {base_link_pos}")
    print(f"[Debug] Difference: {base_link_pos - articulation_pos}")
    
    print("Starting replay...")
    for i in range(start_idx, end_idx):
        if not env.is_running():
            break
            
        # Extract GT pose from dataset
        target_pos = data['robot0_eef_pos'][i]
        target_rot_aa = data['robot0_eef_rot_axis_angle'][i]
        target_gripper = data['robot0_gripper_width'][i][0]
        
        # Check for joint positions (High-Precision Replay)
        target_joint_pos = None
        if 'robot0_joint_pos' in data:
            target_joint_pos = data['robot0_joint_pos'][i]
        
        # Convert axis-angle to 6D for the runner's action format
        rot_mat = R.from_rotvec(target_rot_aa).as_matrix()
        target_rot6d = rotation_matrix_to_rot6d(rot_mat)
        
        # Construct action: [pos(3), rot6d(6), gripper(1)]
        action = np.zeros((1, 10))
        action[0, :3] = target_pos
        action[0, 3:9] = target_rot6d
        action[0, 9] = target_gripper
        
        # Step environment
        if target_joint_pos is not None:
            # PERFECT REPLAY: Bypass IK entirely to verify ground truth
            env.manipulator.set_joint_positions(target_joint_pos[:7], np.arange(7))
            
            # Gripper control (absolute)
            f_pos = np.clip(target_gripper / 2.0, 0.0, 0.04)
            if hasattr(env, "_gripper_joint_indices") and env._gripper_joint_indices:
                env.manipulator.set_joint_positions(
                    positions=np.array([f_pos] * len(env._gripper_joint_indices)),
                    joint_indices=np.array(env._gripper_joint_indices)
                )
            
            env.world.step(render=True)
            obs = env._capture_observation()
            info = {'ik_success': 'Bypassed (Perfect Replay)'}
        else:
            # Standard IK-based replay with refinement
            obs, reward, done, info = env.step(action)
        
        # Measure actual achieved pose
        actual_pos, actual_rot_mat = env._get_end_effector_pose()
        pos_error = np.linalg.norm(actual_pos - target_pos)
        z_error = actual_pos[2] - target_pos[2]
        
        # Rotation error (angle in degrees)
        target_rot = R.from_matrix(rot6d_to_rotation_matrix(target_rot6d))
        actual_rot = R.from_matrix(actual_rot_mat)
        combined_rot = target_rot.inv() * actual_rot
        rot_error_deg = combined_rot.magnitude() * 180 / np.pi
        
        if (i - start_idx) % 10 == 0:
            print(f"Step {i-start_idx}/{end_idx-start_idx}")
            print(f"  Target: {target_pos}")
            print(f"  Actual: {actual_pos}")
            print(f"  Pos Error: {pos_error*1000:.2f} mm | Z Error: {z_error*1000:.2f} mm | Rot Error: {rot_error_deg:.2f} deg")
            print(f"  IK Success: {info.get('ik_success', 'N/A')} | Gripper Target: {target_gripper:.3f} | Actual: {np.array(obs['robot0_gripper_width']).item():.3f}")
            
    print("Replay complete.")
    env.close()

if __name__ == "__main__":
    main()
