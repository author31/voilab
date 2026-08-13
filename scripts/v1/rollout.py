"""
Policy rollout / evaluation in Isaac Sim — config-based v1.

Loads a trained diffusion policy checkpoint and evaluates it in the environment
described by ``--config``. Scene setup, robot/gripper config, object placement,
initial pose and the observation pipeline are shared with ``sdg.py`` via
``engine`` so the observations fed to the policy match the training data.

This file owns only the rollout specifics: the policy wrapper (relative-pose
inference), the inference loop, action execution and viewport setup. The scripted
motion planner is used ONLY to reach the above-object start pose (not for control).
"""

import os
import sys
import time
import argparse
from collections import deque

# --- import bootstrap: make both scripts/v1 and scripts/ importable ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _p in (_SCRIPTS, _HERE):  # insert order -> _HERE ends up first on sys.path
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import cv2
import dill
import hydra
import torch
from loguru import logger
from scipy.spatial.transform import Rotation as R

from configs import config_names, get_config  # import-safe (no isaacsim at module load)


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--config", type=str, required=True, choices=config_names(),
                    help="Environment config to load; must match the one used for data collection")
parser.add_argument("--session_dir", type=str, default=None,
                    help="Session dir containing demos/mapping/object_poses.json "
                         "(ArUco-spawn configs). If omitted, --object_poses_path may be given.")
parser.add_argument("--object_poses_path", type=str, default=None,
                    help="Explicit path to object_poses.json. Overrides --session_dir.")
parser.add_argument("--episode_index", type=int, default=0,
                    help="Episode index to initialize objects from (default: 0).")
parser.add_argument("--num_eval_rounds", type=int, default=1,
                    help="Number of evaluation rounds to run back-to-back (default: 1).")
parser.add_argument("--round_timeout", type=float, default=30.0,
                    help="Wall-clock timeout per evaluation round in seconds (default: 30).")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

from isaacsim import SimulationApp

logger.info("Initializing IsaacSim App Launcher.")
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": ["isaacsim.robot_motion.motion_generation"],
})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.robot_motion.motion_generation")

import omni.ui as ui
import omni.kit.app
import omni.kit.viewport.utility as vp_util

# Imported AFTER SimulationApp so isaacsim modules inside the engine resolve.
import engine as bc
from config import TableSpawnCfg

# Policy stack
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: F401
from diffusion_policy.policy.base_image_policy import BaseImagePolicy  # noqa: F401


# Physics ticks simulated per *data frame* (one obs/action sample at the
# data-collection rate). See the original rollout for the full rationale.
PHYSICS_STEPS_PER_FRAME = 10


class PolicyWrapper:
    """Wraps a diffusion policy: absolute obs in -> absolute action waypoints out."""

    def __init__(self, checkpoint_path, device="cuda:0"):
        print(f"[Policy] Loading checkpoint: {checkpoint_path}")
        self.device = torch.device(device)

        payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
        self.cfg = payload["cfg"]

        self.policy = hydra.utils.instantiate(self.cfg.policy)
        if "state_dicts" in payload:
            self.policy.load_state_dict(payload["state_dicts"]["model"])
        else:
            self.policy.load_state_dict(payload["state_dict"])
        self.policy.to(self.device)
        self.policy.eval()

        PRODUCED_KEYS = {
            "camera0_rgb", "robot0_eef_pos", "robot0_eef_rot_axis_angle",
            "robot0_eef_rot_axis_angle_wrt_start", "robot0_gripper_width",
        }
        self.key_horizons = {
            k: attr.horizon
            for k, attr in self.cfg.shape_meta.obs.items()
            if not attr.get("ignore_by_policy", False)
        }
        self.key_down_sample = {
            k: self.cfg.shape_meta.obs[k].down_sample_steps
            for k in self.key_horizons
        }
        missing = set(self.key_horizons) - PRODUCED_KEYS
        if missing:
            raise RuntimeError(
                f"shape_meta.obs requires keys {sorted(missing)} that "
                f"build_obs_dict() does not produce; update engine.build_obs_dict."
            )

        self.action_down_sample = self.cfg.shape_meta.action.down_sample_steps

        self.obs_window = max(
            (h - 1) * self.key_down_sample[k] + 1
            for k, h in self.key_horizons.items()
        )
        target_shape = self.cfg.shape_meta.obs.camera0_rgb.shape
        self.target_c, self.target_h, self.target_w = target_shape[:3]
        print(f"[Policy] Per-key obs horizons: {self.key_horizons}, "
              f"down_sample: {self.key_down_sample} "
              f"(buffer window={self.obs_window} frames, "
              f"action stride={self.action_down_sample}), "
              f"target image: {self.target_w}x{self.target_h} (C={self.target_c})")

        self.obs_buffer_abs = deque(maxlen=self.obs_window)

    def reset(self):
        self.obs_buffer_abs.clear()

    def append_obs(self, obs_dict):
        """Push one raw absolute frame (resized image + abs poses) onto buffer."""
        img = cv2.resize(
            obs_dict["camera0_rgb"], (self.target_w, self.target_h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        img = np.moveaxis(img, -1, 0) / 255.0
        self.obs_buffer_abs.append({
            "camera0_rgb": img,
            "robot0_eef_pos": obs_dict["robot0_eef_pos"].astype(np.float32).copy(),
            "robot0_eef_rot_axis_angle": obs_dict["robot0_eef_rot_axis_angle"].astype(np.float32).copy(),
            "robot0_eef_rot_axis_angle_wrt_start": obs_dict["robot0_eef_rot_axis_angle_wrt_start"].astype(np.float32).copy(),
            "robot0_gripper_width": obs_dict["robot0_gripper_width"].astype(np.float32).copy(),
        })

    def _strided_indices(self, n_frames, horizon, down_sample):
        """Buffer indices for one key, oldest->newest, spaced by down_sample."""
        newest = n_frames - 1
        idxs = [max(0, newest - i * down_sample) for i in range(horizon)]
        return idxs[::-1]

    def predict(self):
        """Run inference on the current buffer; return absolute action waypoints."""
        assert len(self.obs_buffer_abs) > 0, "predict() called with empty obs buffer"
        frames = list(self.obs_buffer_abs)  # oldest -> newest
        n_frames = len(frames)

        ref_frame = frames[-1]
        ref_pose6 = np.concatenate(
            [ref_frame["robot0_eef_pos"], ref_frame["robot0_eef_rot_axis_angle"]]
        )
        ref_mat = bc.pose_to_mat(ref_pose6)
        ref_mat_inv = np.linalg.inv(ref_mat)

        rel10 = []
        for frame in frames:
            frame_pose6 = np.concatenate(
                [frame["robot0_eef_pos"], frame["robot0_eef_rot_axis_angle"]]
            )
            rel10.append(bc.mat_to_pose10d(ref_mat_inv @ bc.pose_to_mat(frame_pose6)))

        batch_obs_tensor = {}
        for key, horizon in self.key_horizons.items():
            idxs = self._strided_indices(n_frames, horizon, self.key_down_sample[key])
            if key == "camera0_rgb":
                seq = [frames[i]["camera0_rgb"] for i in idxs]
            elif key == "robot0_eef_pos":
                seq = [rel10[i][:3] for i in idxs]
            elif key == "robot0_eef_rot_axis_angle":
                seq = [rel10[i][3:9] for i in idxs]
            elif key == "robot0_eef_rot_axis_angle_wrt_start":
                seq = [frames[i]["robot0_eef_rot_axis_angle_wrt_start"] for i in idxs]
            elif key == "robot0_gripper_width":
                seq = [frames[i]["robot0_gripper_width"] for i in idxs]
            else:
                raise RuntimeError(f"unhandled obs key {key}")
            val_stack = np.stack(seq, axis=0).astype(np.float32)
            batch_obs_tensor[key] = torch.from_numpy(val_stack).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.policy.predict_action(batch_obs_tensor)
            actions_rel = result["action"][0].cpu().numpy()  # (n_action_steps, 10)

        actions_abs = []
        for action_rel in actions_rel:
            action_mat_rel = np.eye(4)
            action_mat_rel[:3, :3] = bc.rot6d_to_matrix(action_rel[3:9])
            action_mat_rel[:3, 3] = action_rel[:3]
            action_mat_abs = ref_mat @ action_mat_rel
            actions_abs.append({
                "pos": action_mat_abs[:3, 3].copy(),
                "rot_mat": action_mat_abs[:3, :3].copy(),
                "gripper": float(action_rel[9]),
            })

        return actions_abs


def setup_dual_viewports(camera_prim_path):
    """Dual viewports: main perspective view and wrist/gopro camera view."""
    v1_window = ui.Workspace.get_window("Viewport")
    if not v1_window:
        print("Error: Main viewport window not found")
        return

    v1_api = vp_util.get_viewport_from_window_name("Viewport")
    if v1_api:
        v1_api.camera_path = "/OmniverseKit_Persp"

    v2_window = ui.Workspace.get_window("Viewport 2")
    if not v2_window:
        v2_window = vp_util.create_viewport_window("Viewport 2")
        omni.kit.app.get_app().update()

    v2_api = vp_util.get_viewport_from_window_name("Viewport 2")
    if v2_api:
        v2_api.camera_path = camera_prim_path

    if v1_window and v2_window:
        omni.kit.app.get_app().update()
        try:
            v2_window.dock_in(v1_window, ui.DockPosition.RIGHT)
            print("Viewports docked: [Persp] | [Camera]")
        except Exception as e:
            print(f"Docking failed: {e}")


def resolve_object_poses_path(cfg) -> str:
    if args.object_poses_path is not None:
        return args.object_poses_path
    if args.session_dir is not None:
        return os.path.join(args.session_dir, "demos", "mapping", "object_poses.json")
    # Table-spawn configs don't read object_poses.json, so it's optional there.
    if isinstance(cfg.object_spawn, TableSpawnCfg):
        return None
    raise RuntimeError("Provide either --object_poses_path or --session_dir")


def drive_to_action(manipulator, art_kine_solver, action, step_count):
    """Command the arm + gripper toward one predicted waypoint (no stepping)."""
    target_quat_xyzw = R.from_matrix(action["rot_mat"]).as_quat()
    target_quat_wxyz = target_quat_xyzw[[3, 0, 1, 2]]

    success = bc.apply_ik_solution(
        manipulator, art_kine_solver, action["pos"], target_quat_wxyz
    )
    if not success:
        print(f"[Warning] IK Failed at step {step_count}")

    bc.set_gripper_width(manipulator, action["gripper"])


def advance_one_frame(world, camera, manipulator, lula_solver, art_kine_solver,
                      policy_wrapper, episode_start_mat):
    """Step one data frame of physics, then capture an obs into the buffer."""
    for _ in range(PHYSICS_STEPS_PER_FRAME):
        world.step(render=True)

    obs_dict, episode_start_mat = bc.build_obs_dict(
        camera, manipulator, lula_solver, art_kine_solver, episode_start_mat
    )
    if obs_dict is not None:
        policy_wrapper.append_obs(obs_dict)
    return obs_dict, episode_start_mat


def record_eef(art_kine_solver, manipulator, eef_pos_list, eef_rot_list, gripper_list):
    """Append the current measured EE pose + gripper to the rollout record."""
    rec_pos, rec_rot_mat = art_kine_solver.compute_end_effector_pose()
    eef_pos_list.append(rec_pos.astype(np.float32))
    eef_rot_list.append(R.from_matrix(rec_rot_mat[:3, :3]).as_rotvec().astype(np.float32))
    gripper_list.append(np.array([bc.get_gripper_width(manipulator)], dtype=np.float32))


# A manipulated object is considered grasped once lifted at least this far above
# the height it settled at before inference started.
GRASP_LIFT_THRESHOLD_M = 0.04


def object_prim_paths(cfg):
    """Prim paths of every spawnable object in the config (for grasp tracking)."""
    return [e.prim_path for e in cfg.object_spawn.preload_objects if e.prim_path]


def object_heights(get_object_world_pose, prim_paths):
    """Current world-frame z of each tracked object, keyed by prim path."""
    heights = {}
    for prim_path in prim_paths:
        try:
            heights[prim_path] = float(get_object_world_pose(prim_path)[2, 3])
        except Exception:
            continue
    return heights


def reset_scene_for_round(ctx):
    """Re-place objects and drive the arm back to the above-object start pose."""
    placed = bc.place_episode_objects(
        ctx["cfg"], args.episode_index, ctx["object_poses_path"],
        ctx["object_prims"], ctx["preload_by_name"],
    )
    assert placed > 0, f"No objects placed for episode {args.episode_index}"

    bc.settle_simulation(ctx["world"], ctx["manipulator"])
    curr_pos, _ = bc.get_end_effector_pos_quat_wxyz(
        ctx["manipulator"], ctx["lula_solver"], ctx["art_kine_solver"]
    )
    bc.apply_initial_ee_pose(
        ctx["cfg"], ctx["manipulator"], ctx["lula_solver"], ctx["art_kine_solver"], curr_pos,
    )

    print("[Rollout] Moving to position above object...")
    bc.move_to_above(
        ctx["motion_planner"], ctx["manipulator"], ctx["lula_solver"],
        ctx["art_kine_solver"], ctx["world"],
        is_running_fn=simulation_app.is_running,
    )
    print("[Rollout] Reached position above object.")

    return object_heights(ctx["get_object_world_pose"], ctx["object_prim_paths"])


def run_eval_round(ctx, round_idx, start_heights):
    """Run one timed inference round; return a result dict for the summary."""
    policy_wrapper = ctx["policy_wrapper"]
    world = ctx["world"]
    camera = ctx["camera"]
    manipulator = ctx["manipulator"]
    lula_solver = ctx["lula_solver"]
    art_kine_solver = ctx["art_kine_solver"]
    get_object_world_pose = ctx["get_object_world_pose"]
    prim_paths = ctx["object_prim_paths"]
    action_ds = policy_wrapper.action_down_sample

    step_count = 0
    eef_pos_list, eef_rot_list, gripper_list = [], [], []
    episode_start_pose = None
    episode_start_mat = None  # threaded through build_obs_dict for wrt_start
    max_lift = 0.0  # largest height gain of any tracked object this round

    def update_max_lift():
        nonlocal max_lift
        heights = object_heights(get_object_world_pose, prim_paths)
        for prim_path, z in heights.items():
            base = start_heights.get(prim_path)
            if base is not None:
                max_lift = max(max_lift, z - base)

    # Seed the obs buffer with one real frame so the first inference has data.
    policy_wrapper.reset()
    obs_dict = None
    while simulation_app.is_running() and obs_dict is None:
        obs_dict, episode_start_mat = advance_one_frame(
            world, camera, manipulator, lula_solver, art_kine_solver,
            policy_wrapper, episode_start_mat,
        )
    if obs_dict is not None:
        episode_start_pose = np.concatenate([
            obs_dict["robot0_eef_pos"].astype(np.float64),
            obs_dict["robot0_eef_rot_axis_angle"].astype(np.float64),
        ])

    deadline = time.time() + args.round_timeout
    timed_out = False
    while simulation_app.is_running():
        if time.time() >= deadline:
            timed_out = True
            break
        actions_abs = policy_wrapper.predict()

        for action in actions_abs:
            if not simulation_app.is_running() or time.time() >= deadline:
                timed_out = timed_out or time.time() >= deadline
                break
            drive_to_action(manipulator, art_kine_solver, action, step_count)
            for _ in range(action_ds):
                if not simulation_app.is_running():
                    break
                _, episode_start_mat = advance_one_frame(
                    world, camera, manipulator, lula_solver, art_kine_solver,
                    policy_wrapper, episode_start_mat,
                )
            update_max_lift()
            record_eef(art_kine_solver, manipulator,
                       eef_pos_list, eef_rot_list, gripper_list)
            step_count += 1
            if step_count % 50 == 0:
                print(f"Step {step_count}: pos {action['pos']} | grip {action['gripper']:.3f}")

    # --- Grasp + task success ---
    grasp_success = max_lift >= GRASP_LIFT_THRESHOLD_M

    if eef_pos_list:
        episode_end_pose = np.concatenate([eef_pos_list[-1], eef_rot_list[-1]])
    else:
        episode_end_pose = episode_start_pose
    num_steps = len(eef_pos_list)

    task_success = False
    completed = num_steps > 0 and episode_start_pose is not None
    if completed:
        episode_record = {
            "episode_idx": args.episode_index,
            "eef_pos": np.stack(eef_pos_list, 0),
            "eef_rot": np.stack(eef_rot_list, 0),
            "gripper": np.stack(gripper_list, 0),
            "demo_start": np.repeat(episode_start_pose[None, :], num_steps, axis=0),
            "demo_end": np.repeat(episode_end_pose[None, :], num_steps, axis=0),
        }
        try:
            task_success = bool(ctx["success_predicate"](episode_record))
        except Exception as e:
            print(f"[Rollout] Error checking task success: {e}")
            completed = False

    result = {
        "round": round_idx,
        "completed": completed,
        "timed_out": timed_out,
        "steps": num_steps,
        "max_lift": max_lift,
        "grasp_success": grasp_success,
        "task_success": task_success,
    }
    print("=" * 68)
    print(f"[Rollout] Round {round_idx + 1}/{args.num_eval_rounds} "
          f"({'timed out' if timed_out else 'finished'} after {num_steps} steps) | "
          f"grasp {'OK' if grasp_success else 'FAIL'} "
          f"(lift {max_lift * 100:.1f}cm) | "
          f"task {'SUCCESS' if task_success else 'FAIL'}")
    print("=" * 68)
    return result


def print_summary(results):
    """Aggregate per-round results into a process / grasp / task success report."""
    n = len(results)
    completed = sum(r["completed"] for r in results)
    grasped = sum(r["grasp_success"] for r in results)
    succeeded = sum(r["task_success"] for r in results)

    def rate(count):
        return f"{count}/{n} ({100.0 * count / n:.1f}%)" if n else "0/0"

    print("\n" + "#" * 68)
    print(f"# Evaluation summary over {n} round(s)")
    print("#" * 68)
    print(f"  Process completed : {rate(completed)}")
    print(f"  Grasp success     : {rate(grasped)}")
    print(f"  Task success      : {rate(succeeded)}")
    print("#" * 68)
    print(f"  {'Round':>5} | {'Completed':>9} | {'TimedOut':>8} | "
          f"{'Steps':>5} | {'Lift(cm)':>8} | {'Grasp':>5} | {'Task':>5}")
    print("  " + "-" * 64)
    for r in results:
        print(f"  {r['round'] + 1:>5} | {str(r['completed']):>9} | "
              f"{str(r['timed_out']):>8} | {r['steps']:>5} | "
              f"{r['max_lift'] * 100:>8.1f} | "
              f"{'OK' if r['grasp_success'] else '-':>5} | "
              f"{'OK' if r['task_success'] else '-':>5}")
    print("#" * 68 + "\n")


def main():
    logger.info("Initializing rollout workspace.")
    session_dir = args.session_dir or f"./data/eval_outputs/session_{int(time.time())}"
    os.makedirs(os.path.join(session_dir, "DEBUG_IMGS"), exist_ok=True)

    cfg = get_config(args.config)

    # --- Shared environment (identical to sdg.py) ---
    env = bc.build_environment(cfg)
    world = env["world"]
    manipulator = env["panda"]
    camera = env["camera"]

    lula_solver, art_kine_solver = bc.initialize_ik_solvers(cfg, manipulator)
    prim_mgr = bc.RigidPrimManager()
    get_object_world_pose = bc.make_get_object_world_pose(prim_mgr)

    # --- Object placement (identical to sdg.py) ---
    object_prims, preload_by_name = bc.preload_objects(world, cfg)
    object_poses_path = resolve_object_poses_path(cfg)
    logger.info(f"Loading object poses from: {object_poses_path}")

    # --- Policy ---
    policy_wrapper = PolicyWrapper(args.checkpoint, device=args.device)
    setup_dual_viewports(cfg.manipulator.camera.prim_path)

    # --- Motion planner to reach the above-object start pose ---
    pickplace = bc.build_pickplace(
        cfg, get_object_world_pose_fn=get_object_world_pose, world=world,
    )
    motion_planner = bc.build_motion_planner(
        cfg, get_object_world_pose_fn=get_object_world_pose, pickplace=pickplace
    )

    ctx = {
        "cfg": cfg,
        "world": world,
        "manipulator": manipulator,
        "camera": camera,
        "lula_solver": lula_solver,
        "art_kine_solver": art_kine_solver,
        "get_object_world_pose": get_object_world_pose,
        "object_prims": object_prims,
        "preload_by_name": preload_by_name,
        "object_poses_path": object_poses_path,
        "object_prim_paths": object_prim_paths(cfg),
        "policy_wrapper": policy_wrapper,
        "motion_planner": motion_planner,
        "success_predicate": cfg.task.success_predicate,
    }

    print(f"[Rollout] Running {args.num_eval_rounds} round(s), "
          f"{args.round_timeout:.0f}s each.")

    results = []
    for round_idx in range(args.num_eval_rounds):
        if not simulation_app.is_running():
            break
        print(f"\n[Rollout] === Round {round_idx + 1}/{args.num_eval_rounds} ===")
        start_heights = reset_scene_for_round(ctx)
        results.append(run_eval_round(ctx, round_idx, start_heights))

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
    simulation_app.close()
