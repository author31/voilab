"""
Preview a configured Isaac Sim environment before SDG / rollout — config-based v1.

Loads the environment described by ``--config`` exactly the way ``sdg.py`` and
``rollout.py`` do (same scene, manipulator and object spawn coordinates), places
the objects for one episode, settles the scene, and then holds the viewport open
so you can visually verify the layout, robot pose and camera framing before
committing to a data-generation or evaluation run.

By default it stops after spawning + settling. Pass ``--move_to_start`` to also
drive the arm to the per-task initial EE pose and run the scripted move-to-above,
previewing the exact state SDG/rollout begin recording/inference from.
"""

import os
import sys
import argparse

# --- import bootstrap: make both scripts/v1 and scripts/ importable ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _p in (_SCRIPTS, _HERE):  # insert order -> _HERE ends up first on sys.path
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs import config_names, get_config  # import-safe (no isaacsim at module load)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, choices=config_names(),
                    help="Environment config (scene + manipulator + task) to preview")
parser.add_argument("--session_dir", type=str, default=None,
                    help="Session dir containing demos/mapping/object_poses.json "
                         "(ArUco-spawn configs only).")
parser.add_argument("--object_poses_path", type=str, default=None,
                    help="Explicit path to object_poses.json. Overrides --session_dir.")
parser.add_argument("--episode_index", type=int, default=0,
                    help="Episode index whose object layout to preview (default: 0).")
parser.add_argument("--move_to_start", action="store_true",
                    help="Also drive to the initial EE pose and run move-to-above.")
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

enable_extension("isaacsim.robot_motion.motion_generation")

import omni.ui as ui
import omni.kit.app
import omni.kit.viewport.utility as vp_util

import engine as bc
from config import TableSpawnCfg


def setup_dual_viewports(camera_prim_path):
    """Dual viewports: main perspective view and wrist/gopro observation camera."""
    v1_window = ui.Workspace.get_window("Viewport")
    if not v1_window:
        print("[Preview] Error: Main viewport window not found")
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
            print("[Preview] Viewports docked: [Persp] | [Camera]")
        except Exception as e:
            print(f"[Preview] Docking failed: {e}")


def resolve_object_poses_path(cfg):
    if args.object_poses_path is not None:
        return args.object_poses_path
    if args.session_dir is not None:
        return os.path.join(args.session_dir, "demos", "mapping", "object_poses.json")
    if isinstance(cfg.object_spawn, TableSpawnCfg):
        return None
    raise RuntimeError("Provide either --object_poses_path or --session_dir")


def main():
    print(f"[Preview] Loading config: {args.config}")
    cfg = get_config(args.config)

    env = bc.build_environment(cfg)
    world = env["world"]
    panda = env["panda"]

    lula_solver, art_kine_solver = bc.initialize_ik_solvers(cfg, panda)
    prim_mgr = bc.RigidPrimManager()
    get_object_world_pose = bc.make_get_object_world_pose(prim_mgr)

    object_prims, preload_by_name = bc.preload_objects(world, cfg)
    object_poses_path = resolve_object_poses_path(cfg)

    world.reset()
    bc.reset_robot_pose(
        env["robot_xform"], env["franka_translation"],
        env["franka_rotation"], env["camera_translation"],
    )

    placed = bc.place_episode_objects(
        cfg, args.episode_index, object_poses_path, object_prims, preload_by_name
    )
    print(f"[Preview] Placed {placed} object(s) for episode {args.episode_index}.")

    bc.settle_simulation(world, panda)
    setup_dual_viewports(cfg.manipulator.camera.prim_path)

    if args.move_to_start:
        curr_pos, _ = bc.get_end_effector_pos_quat_wxyz(panda, lula_solver, art_kine_solver)
        pickplace = bc.build_pickplace(
            cfg, get_object_world_pose_fn=get_object_world_pose, world=world,
        )
        bc.apply_initial_ee_pose(cfg, panda, lula_solver, art_kine_solver, curr_pos)
        motion_planner = bc.build_motion_planner(
            cfg, get_object_world_pose_fn=get_object_world_pose, pickplace=pickplace
        )
        print("[Preview] Moving to position above object...")
        bc.move_to_above(
            motion_planner, panda, lula_solver, art_kine_solver, world,
            is_running_fn=simulation_app.is_running,
        )
        print("[Preview] Reached above-object start pose.")

    print("[Preview] Environment ready. Close the window to exit.")
    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
