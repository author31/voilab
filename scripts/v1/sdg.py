"""
Synthetic Data Generation (SDG) for Isaac Sim — config-based v1.

Replays the scripted pick-place motion planner across every episode and records
the resulting observation stream (camera + end-effector + gripper) into a
multi-episode zarr dataset.

The whole simulation environment — scene, manipulator, and object spawn
coordinates — is described by a single ``EnvConfig`` selected with ``--config``
(see ``configs/``). All scene/robot/observation behaviour is shared with
``rollout.py`` via ``engine`` so the recorded observations match inference.
This file owns only the data-generation specifics: action-noise injection,
per-episode recording, success bookkeeping and zarr I/O.
"""

import os
import sys
import json
import argparse
import shutil
import tempfile
import zipfile

# --- import bootstrap: make both scripts/v1 and scripts/ importable ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _p in (_SCRIPTS, _HERE):  # insert order -> _HERE ends up first on sys.path
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from numcodecs import Blosc
import zarr
from zarr.storage import ZipStore, DirectoryStore
from scipy.spatial.transform import Rotation as R

from configs import config_names, get_config  # import-safe (no isaacsim at module load)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, choices=config_names(),
                    help="Environment config (scene + manipulator + task) to load")
parser.add_argument("--session_dir", type=str, default=None)
parser.add_argument("--num_episodes", type=int, default=None,
                    help="Number of episodes to run. Required for table-spawn configs "
                         "(e.g. simple_room); defaults to the object_poses.json length "
                         "for ArUco-spawn configs.")
parser.add_argument("--output", type=str, default="simulation_dataset.zarr.zip",
                    help="Output filename ONLY (no path); saved under session_dir")
# --- Action noise injection (MolmoBOT-style, action-proportional) ---
parser.add_argument("--inject_wp_action_noise", action="store_true",
                    help="Inject action-proportional noise into Cartesian waypoints during collection")
parser.add_argument("--inject_gripper_noise", action="store_true",
                    help="Inject action-proportional noise into the recorded gripper width during collection")
parser.add_argument("--noise_seed", type=int, default=0,
                    help="Base seed for noise RNG; per-episode seed is noise_seed + episode_idx")
parser.add_argument("--wp_pos_noise_alpha", type=float, default=0.15,
                    help="Waypoint position noise std as a fraction of the commanded displacement magnitude")
parser.add_argument("--wp_pos_noise_cap", type=float, default=0.01,
                    help="Maximum waypoint position noise std (meters)")
parser.add_argument("--wp_rot_noise_alpha", type=float, default=0.10,
                    help="Waypoint orientation noise std as a fraction of the commanded angular displacement")
parser.add_argument("--wp_rot_noise_cap", type=float, default=0.05,
                    help="Maximum waypoint orientation noise std (radians)")
parser.add_argument("--gripper_noise_alpha", type=float, default=0.10,
                    help="Gripper-width noise std as a fraction of the per-step commanded width change")
parser.add_argument("--gripper_noise_cap", type=float, default=0.005,
                    help="Maximum gripper-width noise std (meters)")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": ["isaacsim.robot_motion.motion_generation"],
})

from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path

enable_extension("isaacsim.robot_motion.motion_generation")

# Imported AFTER SimulationApp so isaacsim modules inside the engine resolve.
import engine as bc
from config import TableSpawnCfg


assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("[SDG] ERROR: Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()


# ======================================================================
# Action noise injection (MolmoBOT-style) — data-generation only
# ======================================================================
def _quat_geodesic_angle(q0_wxyz, q1_wxyz) -> float:
    r0 = R.from_quat(bc.wxyz_to_xyzw(np.asarray(q0_wxyz, dtype=float)))
    r1 = R.from_quat(bc.wxyz_to_xyzw(np.asarray(q1_wxyz, dtype=float)))
    return float((r0.inv() * r1).magnitude())


def _inject_waypoint_noise(waypoints, p_start, q_start_wxyz, p_goal, q_goal_wxyz,
                           pos_alpha, pos_cap, rot_alpha, rot_cap, rng):
    """Add action-proportional noise to interior Cartesian waypoints.

    First/last waypoints stay clean to preserve continuity with the current pose
    and grasp/place accuracy at the goal.
    """
    if len(waypoints) <= 2:
        return waypoints

    total_disp = float(np.linalg.norm(np.asarray(p_goal, dtype=float) - np.asarray(p_start, dtype=float)))
    sigma_p = min(pos_alpha * total_disp, pos_cap)
    sigma_r = min(rot_alpha * _quat_geodesic_angle(q_start_wxyz, q_goal_wxyz), rot_cap)

    if sigma_p <= 0.0 and sigma_r <= 0.0:
        return waypoints

    noisy = [waypoints[0]]
    for wp in waypoints[1:-1]:
        p = wp[:3] + (rng.normal(0.0, sigma_p, size=3) if sigma_p > 0.0 else 0.0)
        if sigma_r > 0.0:
            dr = R.from_rotvec(rng.normal(0.0, sigma_r, size=3))
            q_xyzw = (dr * R.from_quat(bc.wxyz_to_xyzw(wp[3:]))).as_quat()
            q = bc.xyzw_to_wxyz(q_xyzw)
        else:
            q = wp[3:]
        noisy.append(np.concatenate([p, q]))
    noisy.append(waypoints[-1])
    return noisy


def make_plan_line_cartesian(inject_noise=False, pos_alpha=0.15, pos_cap=0.01,
                             rot_alpha=0.10, rot_cap=0.05, rng=None):
    """Wrap engine.plan_line_cartesian with optional waypoint noise.

    Keeps the (p_start, q_start, p_goal, q_goal, step_m) signature so it is a
    drop-in for PickPlace's plan_line_cartesian_fn.
    """
    def _plan(p_start, q_start_wxyz, p_goal, q_goal_wxyz, step_m: float = 0.005):
        waypoints = bc.plan_line_cartesian(p_start, q_start_wxyz, p_goal, q_goal_wxyz, step_m)
        if inject_noise and rng is not None:
            waypoints = _inject_waypoint_noise(
                waypoints, p_start, q_start_wxyz, p_goal, q_goal_wxyz,
                pos_alpha, pos_cap, rot_alpha, rot_cap, rng,
            )
        return waypoints
    return _plan


class GripperNoiseInjector:
    """Action-proportional gripper-width noise (clean width tracked internally)."""

    def __init__(self, alpha, cap, rng):
        self.alpha = alpha
        self.cap = cap
        self.rng = rng
        self.prev_clean = None

    def __call__(self, clean_width: float) -> float:
        if self.prev_clean is None:
            noised = clean_width
        else:
            delta = abs(clean_width - self.prev_clean)
            sigma = min(self.alpha * delta, self.cap)
            noised = clean_width + self.rng.normal(0.0, sigma) if sigma > 0.0 else clean_width
        self.prev_clean = clean_width
        return max(0.0, float(noised))


# ======================================================================
# Recording
# ======================================================================
def record_state(camera, panda, lula_solver, art_kine_solver,
                 rgb_list, eef_pos_list, eef_rot_list, gripper_list,
                 gripper_noise_injector=None):
    """Record one frame of observation; returns the 6D end-effector pose."""
    img = bc.get_camera_rgb(camera)
    if img is not None:
        rgb_list.append(img)

    eef_pose6d = bc.get_end_effector_pose(panda, lula_solver, art_kine_solver)
    eef_pos_list.append(eef_pose6d[:3])
    eef_rot_list.append(eef_pose6d[3:])

    gripper_width = bc.get_gripper_width(panda)
    if gripper_noise_injector is not None:
        gripper_width = gripper_noise_injector(gripper_width)
    gripper_list.append([gripper_width])

    return eef_pose6d


# ======================================================================
# Progress / dataset I/O
# ======================================================================
def _load_progress(session_dir: str) -> set:
    progress_path = os.path.join(session_dir, ".previous_progress.json")
    if not os.path.exists(progress_path):
        return set()
    try:
        with open(progress_path, "r") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[SDG] WARNING: Failed to read progress file: {exc}")
        return set()
    completed = payload.get("completed_episodes", [])
    return set(int(x) for x in completed)


def _save_progress(session_dir: str, completed: set) -> None:
    progress_path = os.path.join(session_dir, ".previous_progress.json")
    payload = {"completed_episodes": sorted(completed)}
    with open(progress_path, "w") as f:
        json.dump(payload, f, indent=2)


def save_multi_episode_dataset(output_path: str, episodes: list) -> None:
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    CHUNK_FRAMES = 128

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".zarr.zip", dir=out_dir)
    os.close(fd)
    staging_dir = tempfile.mkdtemp(prefix=".stage_", dir=out_dir)

    try:
        dir_store = DirectoryStore(staging_dir)
        root = zarr.group(dir_store)
        data = root.create_group("data")

        specs = [
            ("rgb", "camera0_rgb", np.uint8),
            ("demo_start", "robot0_demo_start_pose", np.float64),
            ("demo_end", "robot0_demo_end_pose", np.float64),
            ("eef_pos", "robot0_eef_pos", np.float32),
            ("eef_rot", "robot0_eef_rot_axis_angle", np.float32),
            ("gripper", "robot0_gripper_width", np.float32),
        ]

        episode_lengths = [len(ep["rgb"]) for ep in episodes]
        total = int(sum(episode_lengths))

        for key, dataset_name, dtype in specs:
            sample = np.asarray(episodes[0][key])
            item_shape = sample.shape[1:]
            arr = data.create_dataset(
                dataset_name,
                shape=(total,) + item_shape,
                dtype=dtype,
                chunks=(min(CHUNK_FRAMES, total),) + item_shape,
                compressor=compressor,
            )
            offset = 0
            for ep in episodes:
                block = np.asarray(ep[key], dtype=dtype)
                n = block.shape[0]
                arr[offset:offset + n] = block
                offset += n

        episode_ends = np.cumsum(episode_lengths)
        meta = root.create_group("meta")
        meta.create_dataset("episode_ends", data=episode_ends)

        zip_store = ZipStore(tmp_path, mode="w")
        try:
            zarr.copy_store(dir_store, zip_store)
        finally:
            zip_store.close()

        if not zipfile.is_zipfile(tmp_path):
            raise RuntimeError(f"Wrote {tmp_path} but it is not a valid zip archive")

        os.replace(tmp_path, output_path)  # atomic within the same directory
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print("[SAVE] dataset saved at:", output_path)


# ======================================================================
# Episode counting
# ======================================================================
def resolve_total_episodes(cfg, object_poses_path):
    """Episode count: explicit --num_episodes wins, else object_poses.json length."""
    if args.num_episodes is not None:
        return int(args.num_episodes)
    if isinstance(cfg.object_spawn, TableSpawnCfg):
        raise RuntimeError(
            "Table-spawn configs have no object_poses.json; pass --num_episodes."
        )
    print(f"[SDG] Looking for object poses at: {object_poses_path}")
    with open(object_poses_path, "r") as f:
        return len(json.load(f))


# ======================================================================
# Main
# ======================================================================
def main():
    print(f"[SDG] Starting with config: {args.config}")

    if args.session_dir is None:
        print("[SDG] ERROR: session_dir is required for multi-episode replay.")
        simulation_app.close()
        return

    cfg = get_config(args.config)

    # --- Shared environment ---
    env = bc.build_environment(cfg)
    world = env["world"]
    panda = env["panda"]
    camera = env["camera"]
    robot_xform = env["robot_xform"]

    lula_solver, art_kine_solver = bc.initialize_ik_solvers(cfg, panda)
    prim_mgr = bc.RigidPrimManager()

    object_prims, preload_by_name = bc.preload_objects(world, cfg)

    object_poses_path = os.path.join(args.session_dir, "demos", "mapping", "object_poses.json")
    total_episodes = resolve_total_episodes(cfg, object_poses_path)
    print(f"[SDG] Replay initialized for {total_episodes} episodes.")

    completed_episodes = _load_progress(args.session_dir)
    episodes_to_run = [ep for ep in range(total_episodes) if ep not in completed_episodes]
    collected_episodes = []

    idx = 0
    while idx < len(episodes_to_run):
        episode_idx = episodes_to_run[idx]
        if not simulation_app.is_running():
            break

        print(f"[SDG] Starting episode {episode_idx}")
        world.reset()
        prim_mgr.clear()
        bc.reset_robot_pose(
            robot_xform, env["franka_translation"], env["franka_rotation"], env["camera_translation"]
        )

        # --- Object placement (shared with rollout) ---
        placed = bc.place_episode_objects(
            cfg, episode_idx, object_poses_path, object_prims, preload_by_name
        )
        if placed == 0:
            print(f"[SDG] Skipping episode {episode_idx}: objects not constructed successfully.")
            idx += 1
            continue

        # --- Settle + initial EE pose (shared with rollout) ---
        bc.settle_simulation(world, panda)
        curr_pos, _ = bc.get_end_effector_pos_quat_wxyz(panda, lula_solver, art_kine_solver)
        get_object_world_pose = bc.make_get_object_world_pose(prim_mgr)

        # --- Per-episode noise RNG + injectors ---
        episode_rng = np.random.default_rng(args.noise_seed + episode_idx)
        plan_fn = make_plan_line_cartesian(
            inject_noise=args.inject_wp_action_noise,
            pos_alpha=args.wp_pos_noise_alpha,
            pos_cap=args.wp_pos_noise_cap,
            rot_alpha=args.wp_rot_noise_alpha,
            rot_cap=args.wp_rot_noise_cap,
            rng=episode_rng,
        )
        gripper_noise_injector = (
            GripperNoiseInjector(args.gripper_noise_alpha, args.gripper_noise_cap, episode_rng)
            if args.inject_gripper_noise else None
        )
        if args.inject_wp_action_noise or args.inject_gripper_noise:
            print(f"[Noise] episode {episode_idx}: waypoint={args.inject_wp_action_noise}, "
                  f"gripper={args.inject_gripper_noise}")

        pickplace = bc.build_pickplace(
            cfg, get_object_world_pose_fn=get_object_world_pose,
            world=world, plan_line_cartesian_fn=plan_fn,
        )
        bc.apply_initial_ee_pose(cfg, panda, lula_solver, art_kine_solver, curr_pos)
        motion_planner = bc.build_motion_planner(
            cfg, get_object_world_pose_fn=get_object_world_pose, pickplace=pickplace
        )

        rgb_list, eef_pos_list, eef_rot_list, gripper_list = [], [], [], []
        episode_start_pose = None
        episode_end_pose = None

        # --- Move above object (no recording) ---
        print("[SDG] Moving to position above object...")
        bc.move_to_above(
            motion_planner, panda, lula_solver, art_kine_solver, world,
            is_running_fn=simulation_app.is_running,
        )

        # First recorded frame at the above-object pose
        eef_pose6d = record_state(
            camera, panda, lula_solver, art_kine_solver,
            rgb_list, eef_pos_list, eef_rot_list, gripper_list,
            gripper_noise_injector=gripper_noise_injector,
        )
        episode_start_pose = eef_pose6d.copy()

        # --- Record the scripted episode ---
        print("[SDG] Recording episode...")
        while simulation_app.is_running():
            motion_planner.step(panda, lula_solver, art_kine_solver)
            bc.step_world(world, render=True)
            eef_pose6d = record_state(
                camera, panda, lula_solver, art_kine_solver,
                rgb_list, eef_pos_list, eef_rot_list, gripper_list,
                gripper_noise_injector=gripper_noise_injector,
            )
            if motion_planner.is_done():
                episode_end_pose = eef_pose6d.copy()
                print("[SDG] Motion plan finished")
                break

        if episode_end_pose is None and eef_pos_list:
            episode_end_pose = np.concatenate([eef_pos_list[-1], eef_rot_list[-1]])

        if not rgb_list:
            print(f"[SDG] WARNING: No frames captured for episode {episode_idx}")
            idx += 1
            continue

        demo_start_list = np.repeat(episode_start_pose[None, :], len(rgb_list), axis=0)
        demo_end_list = np.repeat(episode_end_pose[None, :], len(rgb_list), axis=0)
        episode_record = {
            "episode_idx": episode_idx,
            "rgb": np.stack(rgb_list, 0),
            "eef_pos": np.stack(eef_pos_list, 0),
            "eef_rot": np.stack(eef_rot_list, 0),
            "gripper": np.stack(gripper_list, 0),
            "demo_start": demo_start_list,
            "demo_end": demo_end_list,
        }

        episode_success = cfg.task.success_predicate(episode_record)
        episode_record["success"] = episode_success
        print("[SDG] Task success" if episode_success else "[SDG] Task fail")

        collected_episodes.append(episode_record)
        if episode_success:
            completed_episodes.add(episode_idx)
            _save_progress(args.session_dir, completed_episodes)

        idx += 1

    successful_episodes = [ep for ep in collected_episodes if ep.get("success", False)]
    print(f"[SDG] Total successful trials collected: {len(successful_episodes)}")
    if successful_episodes:
        output_filename = os.path.basename(args.output)
        assert output_filename == args.output, "--output must be a filename only, not a path"
        output_zarr = os.path.join(args.session_dir, output_filename)
        save_multi_episode_dataset(output_zarr, successful_episodes)

    simulation_app.close()


if __name__ == "__main__":
    main()
