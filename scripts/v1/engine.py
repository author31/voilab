"""
Generic Isaac Sim engine for the v1 config-based pipeline.

This module owns ALL behaviour: it interprets an ``EnvConfig`` (see ``config.py``)
to open the scene, build the robot/gripper/camera, construct IK solvers, spawn and
place objects, settle the sim, run the observation pipeline, and drive the scripted
motion planner. There is exactly ONE implementation of each step; the differences
between ED305 and simple_room (and between tasks) live entirely in the config data.

This collapses the previously-duplicated ``base_config`` and ``simple_room_config``
modules into a single engine, and replaces the per-task ``registry`` + per-task
``motion_plan`` planners with one config-driven sequencer.

IMPORTANT: this module imports ``isaacsim.*`` at load time, so it MUST be imported
AFTER ``SimulationApp(...)`` has been instantiated. Pure pose math lives in the
import-safe ``geometry`` module; the declarative schema lives in ``config``.
"""

import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

# --- isaacsim imports (only valid after SimulationApp has been created) ---
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim, RigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)
from pxr import Sdf

# --- legacy project modules live in the parent scripts/ dir. Ensure it is on
# sys.path so v1 works even when only scripts/v1 is importable (some launchers
# don't add scripts/). Appended so v1's own modules keep precedence. ---
import sys
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.append(_SCRIPTS_DIR)

from object_loader import load_object_transforms_from_json
from umi_replay import set_gripper_width  # re-exported for entry points
from motion_plan import PickPlace          # re-exported for entry points

# --- v1 modules ---
import geometry
from geometry import (  # re-exported so entry points can use bc.* style helpers
    wxyz_to_xyzw, xyzw_to_wxyz, matrix_to_rot6d, rot6d_to_matrix,
    pose_to_mat, mat_to_pose10d, plan_line_cartesian,
)
from config import ArucoSpawnCfg, TableSpawnCfg


# ======================================================================
# Stepping / settle constants (shared; identical to the legacy configs)
# ======================================================================
SETTLE_STEPS = 100
MOVE_ABOVE_SETTLE_STEPS = 1
STEP_SLEEP_DT = 1.0 / 60.0


# ======================================================================
# Small data helpers
# ======================================================================
def normalize_object_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def get_object_orientation(name, presets, default_orientation=(1.0, 0.0, 0.0, 0.0)):
    for key, quat in presets.items():
        if key in name.lower():
            return np.array(quat)
    return np.array(default_orientation)


# ======================================================================
# Scene / robot / camera construction
# ======================================================================
def rename_default_prim(new_name: str = "World"):
    """Rename the opened stage's default prim (e.g. ``/Root`` -> ``/World``).

    In-memory namespace edit on the root layer (not saved back to the asset);
    used for simple_room.usd which ships with ``/Root`` as its default prim.
    """
    stage = stage_utils.get_current_stage()
    default_prim = stage.GetDefaultPrim()
    old_path = default_prim.GetPath().pathString if default_prim else "/Root"
    new_path = "/" + new_name
    if old_path == new_path:
        return

    edit = Sdf.BatchNamespaceEdit()
    edit.Add(old_path, new_path)
    if not stage.GetRootLayer().Apply(edit):
        raise RuntimeError(f"Failed to rename default prim {old_path} -> {new_path}")
    stage.SetDefaultPrim(stage.GetPrimAtPath(new_path))


def apply_prim_pose(override):
    """Reposition an existing in-scene prim per a PrimPose override.

    Tries ``override.prim_path`` first; if it is missing, falls back to a name
    search (``override.name_fallback``). Warns loudly rather than silently
    no-op'ing so a wrong path is visible in the logs.
    """
    stage = stage_utils.get_current_stage()
    target = None
    prim = stage.GetPrimAtPath(override.prim_path)
    if prim and prim.IsValid():
        target = override.prim_path
    elif override.name_fallback:
        for p in stage.Traverse():
            if override.name_fallback in p.GetName():
                target = p.GetPath().pathString
                break

    if target is None:
        print(f"[Engine] WARNING: prim '{override.prim_path}' "
              f"(fallback '{override.name_fallback}') not found; pose override skipped. "
              f"Run scripts/launch_simpleroom.py to dump prim paths.")
        return

    xform = SingleXFormPrim(prim_path=target)
    xform.set_world_pose(
        position=np.array(override.translation, dtype=float),
        orientation=np.array(override.rotation_quat_wxyz, dtype=float),
    )
    print(f"[Engine] Set prim '{target}' pose -> t={np.array(override.translation)}, "
          f"q_wxyz={np.array(override.rotation_quat_wxyz)}")


def build_environment(cfg):
    """Open the scene and create the robot, gripper, and observation camera.

    Returns a dict bundle of the long-lived handles. SDG, rollout and preview all
    call this so the world, robot articulation, gripper config and camera are
    identical across the three pipelines.
    """
    scene = cfg.scene
    man = cfg.manipulator
    franka_translation = np.array(cfg.robot_pose.translation, dtype=float)
    franka_rotation = np.array(cfg.robot_pose.rotation_quat_wxyz, dtype=float)
    camera_translation = np.array(cfg.view_camera_translation, dtype=float)

    stage_utils.open_stage(scene.usd_path)
    if scene.rename_default_prim_to:
        rename_default_prim(scene.rename_default_prim_to)
    world = World(stage_units_in_meters=scene.stage_units_in_meters)
    if scene.add_default_ground_plane:
        world.scene.add_default_ground_plane()

    for override in getattr(scene, "prim_pose_overrides", []):
        apply_prim_pose(override)

    robot = stage_utils.add_reference_to_stage(
        usd_path=man.usd_path, prim_path=man.prim_path
    )
    for variant_set, selection in man.variant_selections.items():
        robot.GetVariantSet(variant_set).SetVariantSelection(selection)

    robot_xform = SingleXFormPrim(prim_path=man.prim_path)

    g = man.gripper
    gripper = ParallelGripper(
        end_effector_prim_path=man.end_effector_prim_path,
        joint_prim_names=g.joint_prim_names,
        joint_opened_positions=np.array(g.joint_opened_positions),
        joint_closed_positions=np.array(g.joint_closed_positions),
        action_deltas=np.array(g.action_deltas),
    )

    panda = world.scene.add(
        SingleManipulator(
            prim_path=man.prim_path,
            name=man.name,
            end_effector_prim_path=man.end_effector_prim_path,
            gripper=gripper,
        )
    )
    panda.gripper.set_default_state(panda.gripper.joint_opened_positions)

    reset_robot_pose(robot_xform, franka_translation, franka_rotation, camera_translation)

    cam = man.camera
    camera = Camera(
        prim_path=cam.prim_path,
        name=cam.name,
        resolution=(int(cam.resolution[1]), int(cam.resolution[0])),
    )
    # Apply the local pose explicitly so the offset is interpreted in the
    # camera-axes convention the policy was trained with (e.g. simple_room's
    # FR3 wrist cam uses the "ros" convention). The Camera constructor would
    # otherwise force the "world" convention.
    if cam.translation is not None or cam.orientation_wxyz is not None:
        camera.set_local_pose(
            translation=None if cam.translation is None else np.array(cam.translation),
            orientation=None if cam.orientation_wxyz is None else np.array(cam.orientation_wxyz),
            camera_axes=cam.convention,
        )
    camera.initialize()
    world.reset()

    return {
        "world": world,
        "robot_xform": robot_xform,
        "panda": panda,
        "camera": camera,
        "franka_translation": franka_translation,
        "franka_rotation": franka_rotation,
        "camera_translation": camera_translation,
    }


def reset_robot_pose(robot_xform, franka_translation, franka_rotation, camera_translation):
    """Re-apply the robot base pose and viewport camera (called per episode)."""
    robot_xform.set_local_pose(
        translation=np.array(franka_translation) / stage_utils.get_stage_units(),
        orientation=np.array(franka_rotation),
    )
    set_camera_view(camera_translation, franka_translation)


# ======================================================================
# IK
# ======================================================================
def initialize_ik_solvers(cfg, panda):
    """Create the Lula + Articulation IK solvers and calibrate the base pose."""
    ik = cfg.manipulator.ik
    if ik.kind == "files":
        lula_solver = LulaKinematicsSolver(
            robot_description_path=ik.robot_description_path,
            urdf_path=ik.urdf_path,
        )
    elif ik.kind == "supported":
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config(
            ik.robot_name
        )
        lula_solver = LulaKinematicsSolver(**kinematics_config)
    else:
        raise RuntimeError(f"Unknown IK kind: {ik.kind!r}")

    art_kine_solver = ArticulationKinematicsSolver(
        panda,
        kinematics_solver=lula_solver,
        end_effector_frame_name=ik.end_effector_frame_name,
    )
    calibrate_robot_base(panda, lula_solver)
    return lula_solver, art_kine_solver


def calibrate_robot_base(panda, lula_solver):
    """Sync the Lula solver with the robot's current world base pose."""
    robot_pos, robot_quat = panda.get_world_pose()
    lula_solver.set_robot_base_pose(robot_position=robot_pos, robot_orientation=robot_quat)


def apply_ik_solution(panda, art_kine_solver, target_pos, target_quat_wxyz):
    """Compute IK for the target pose and drive the arm via PD targets."""
    action, success = art_kine_solver.compute_inverse_kinematics(
        target_position=target_pos, target_orientation=target_quat_wxyz
    )
    if success:
        panda.apply_action(action)
        return True
    return False


# ======================================================================
# Object world-pose access
# ======================================================================
class RigidPrimManager:
    """Cache for RigidPrim handles to avoid re-initialization."""

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


def make_get_object_world_pose(prim_mgr):
    def _get_object_world_pose(prim_path):
        prim = prim_mgr.get(prim_path)
        pos_batch, quat_wxyz_batch = prim.get_world_poses()
        pos = pos_batch[0]
        quat_wxyz = quat_wxyz_batch[0]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = pos
        return T

    return _get_object_world_pose


# ======================================================================
# Object loading / placement (dispatched on the spawn strategy)
# ======================================================================
def preload_objects(world, cfg):
    """Reference every spawnable object into the stage once.

    Returns ``(object_prims, preload_by_name)`` where object_prims maps the
    normalized object name to its prim handle and preload_by_name maps it to the
    PreloadObject entry (used to lazily (re)create prims during placement).
    """
    spawn = cfg.object_spawn
    if isinstance(spawn, ArucoSpawnCfg):
        return _preload_aruco(world, spawn)
    if isinstance(spawn, TableSpawnCfg):
        return _preload_table(world, spawn)
    raise RuntimeError(f"Unknown spawn cfg: {type(spawn).__name__}")


def place_episode_objects(cfg, episode_idx, object_poses_path, object_prims, preload_by_name):
    """Place every object for ``episode_idx``. Returns the number placed."""
    spawn = cfg.object_spawn
    if isinstance(spawn, ArucoSpawnCfg):
        return _place_aruco(spawn, episode_idx, object_poses_path, object_prims, preload_by_name)
    if isinstance(spawn, TableSpawnCfg):
        return _place_table(spawn, episode_idx, object_prims, preload_by_name)
    raise RuntimeError(f"Unknown spawn cfg: {type(spawn).__name__}")


def _preload_aruco(world, spawn):
    object_prims = {}
    preload_by_name = {}
    for entry in spawn.preload_objects:
        object_name = normalize_object_name(entry.name)
        preload_by_name[object_name] = entry
        if object_name in object_prims:
            continue

        full_asset_path = os.path.join(spawn.assets_dir, entry.assets)
        if not os.path.exists(full_asset_path):
            print(f"[ObjectLoader] WARNING: Asset not found: {full_asset_path}, skipping {entry.name}")
            continue

        try:
            stage_utils.add_reference_to_stage(usd_path=full_asset_path, prim_path=entry.prim_path)
        except Exception as e:
            print(f"[ObjectLoader] ERROR: Failed to load asset {full_asset_path}: {str(e)}")
            continue

        obj_prim = SingleXFormPrim(
            prim_path=entry.prim_path,
            name=object_name,
            orientation=np.array(entry.quat_wxyz),
        )
        world.scene.add(obj_prim)
        object_prims[object_name] = obj_prim
        print(f"[ObjectLoader] Preloaded {entry.name} as {entry.prim_path}")

    return object_prims, preload_by_name


def _place_aruco(spawn, episode_idx, object_poses_path, object_prims, preload_by_name):
    shim_cfg = {"environment_vars": {"OBJECT_MAXIMUM_Z_HEIGHT": spawn.object_maximum_z_height}}
    object_transforms = load_object_transforms_from_json(
        object_poses_path,
        episode_index=episode_idx,
        aruco_tag_pose=spawn.aruco_tag_pose,
        cfg=shim_cfg,
    )
    if len(object_transforms) == 0:
        return 0

    skip = {normalize_object_name(s) for s in spawn.skip_names}
    placed = 0
    for obj in object_transforms:
        object_name = normalize_object_name(obj["object_name"])
        if object_name in skip:
            continue

        if object_name not in object_prims:
            preload_entry = preload_by_name.get(object_name)
            assert preload_entry, f"Object {object_name} missing from preload_objects"
            full_asset_path = os.path.join(spawn.assets_dir, preload_entry.assets)
            if not os.path.exists(full_asset_path):
                print(f"[ObjectLoader] WARNING: Asset not found: {full_asset_path}, skipping {object_name}")
                continue
            try:
                stage_utils.add_reference_to_stage(usd_path=full_asset_path, prim_path=preload_entry.prim_path)
            except Exception as e:
                print(f"[ObjectLoader] ERROR: Failed to load asset {full_asset_path}: {str(e)}")
                continue
            obj_prim = RigidPrim(preload_entry.prim_path, object_name)
            obj_prim.initialize()
            object_prims[object_name] = obj_prim

        obj_prim = object_prims[object_name]
        obj_pos = np.array(obj["position"], dtype=np.float64) + spawn.object_pos_offset
        orientation = get_object_orientation(object_name, spawn.orientation_presets)
        obj_prim.set_world_pose(position=obj_pos, orientation=orientation)
        placed += 1
        print(f"[ObjectLoader] Positioned {object_name} at {obj_pos}")

    return placed


def _preload_table(world, spawn):
    object_prims = {}
    preload_by_name = {}
    for entry in spawn.preload_objects:
        object_name = normalize_object_name(entry.name)
        preload_by_name[object_name] = entry
        if object_name in object_prims:
            continue

        full_asset_path = os.path.join(spawn.assets_dir, entry.assets)
        if not os.path.exists(full_asset_path):
            print(f"[ObjectLoader] WARNING: Asset not found: {full_asset_path}, skipping {entry.name}")
            continue

        try:
            stage_utils.add_reference_to_stage(usd_path=full_asset_path, prim_path=entry.prim_path)
        except Exception as e:
            print(f"[ObjectLoader] ERROR: Failed to load asset {full_asset_path}: {str(e)}")
            continue

        # RigidPrim + scene.add so the per-episode world.reset() adds it to the
        # physics view, letting it fall and rest on the table.
        obj_prim = RigidPrim(entry.prim_path, name=object_name)
        world.scene.add(obj_prim)
        object_prims[object_name] = obj_prim
        print(f"[ObjectLoader] Preloaded {entry.name} as {entry.prim_path}")

    return object_prims, preload_by_name


def _place_table(spawn, episode_idx, object_prims, preload_by_name):
    rng = np.random.default_rng(episode_idx)
    placed = 0
    for entry in spawn.preload_objects:
        object_name = normalize_object_name(entry.name)

        if object_name not in object_prims:
            full_asset_path = os.path.join(spawn.assets_dir, entry.assets)
            if not os.path.exists(full_asset_path):
                print(f"[ObjectLoader] WARNING: Asset not found: {full_asset_path}, skipping {object_name}")
                continue
            try:
                stage_utils.add_reference_to_stage(usd_path=full_asset_path, prim_path=entry.prim_path)
            except Exception as e:
                print(f"[ObjectLoader] ERROR: Failed to load asset {full_asset_path}: {str(e)}")
                continue
            obj_prim = RigidPrim(entry.prim_path, name=object_name)
            obj_prim.initialize()
            object_prims[object_name] = obj_prim

        obj_prim = object_prims[object_name]
        base_pos = np.array(entry.table_position, dtype=np.float64)
        jitter = rng.uniform(-spawn.spawn_jitter_xy, spawn.spawn_jitter_xy)
        obj_pos = base_pos.copy()
        obj_pos[:2] = np.clip(
            base_pos[:2] + jitter, -spawn.table_half_extent_xy, spawn.table_half_extent_xy
        )
        orientation = np.array(entry.quat_wxyz)
        # RigidPrim is the batched view API in Isaac 5.1: poses are (N, ...).
        obj_prim.set_world_poses(positions=obj_pos[None, :], orientations=orientation[None, :])
        placed += 1
        print(f"[ObjectLoader] Positioned {object_name} at {obj_pos}")

    return placed


# ======================================================================
# Simulation stepping / settling
# ======================================================================
def step_world(world, render=True, sleep_dt=0.01):
    world.step(render=render)
    time.sleep(sleep_dt)


def settle_simulation(world, panda, steps=SETTLE_STEPS):
    """Open the gripper and let the scene settle before an episode."""
    for _ in range(steps):
        set_gripper_width(panda, width=0.1, threshold=0.0, step=0.05)
        world.step(render=True)
        time.sleep(STEP_SLEEP_DT)


def apply_initial_ee_pose(cfg, panda, lula_solver, art_kine_solver, curr_pos):
    """Drive the EE to the per-task initial pose (offset from settled position)."""
    init_pos = np.asarray(curr_pos, dtype=np.float64) + cfg.task.init_ee_offset
    init_quat = cfg.task.init_ee_quat_wxyz
    calibrate_robot_base(panda, lula_solver)
    return apply_ik_solution(panda, art_kine_solver, init_pos, init_quat)


def move_to_above(motion_planner, panda, lula_solver, art_kine_solver, world,
                  is_running_fn=lambda: True, settle_steps=MOVE_ABOVE_SETTLE_STEPS):
    """Run the motion planner until it reaches the "above object" pose.

    The planner transitions ``pickplace.phase`` to "descend" exactly when the
    above-object waypoint is reached; this is where recording (sdg) and inference
    (rollout) begin.
    """
    while is_running_fn():
        motion_planner.step(panda, lula_solver, art_kine_solver)
        for _ in range(settle_steps):
            world.step(render=True)
        if motion_planner.pickplace.phase == "descend":
            return True
    return False


# ======================================================================
# Observation pipeline — the consistency-critical part
# ======================================================================
def read_ee_state(panda, lula_solver, art_kine_solver):
    """Return (ee_pos_world (3,), ee_rot_mat_world (3,3)) in the world frame."""
    base_pos, base_quat = panda.get_world_pose()
    lula_solver.set_robot_base_pose(robot_position=base_pos, robot_orientation=base_quat)
    ee_pos, ee_T = art_kine_solver.compute_end_effector_pose()
    return ee_pos.astype(np.float64), ee_T[:3, :3]


def get_gripper_width(panda) -> float:
    joint_pos = panda.get_joint_positions()
    return float(joint_pos[-2] + joint_pos[-1])


def get_end_effector_pose(panda, lula_solver, art_kine_solver) -> np.ndarray:
    """6D EE pose [x,y,z, rx,ry,rz] (axis-angle), world frame."""
    ee_pos, ee_rot_mat = read_ee_state(panda, lula_solver, art_kine_solver)
    eef_rot = R.from_matrix(ee_rot_mat).as_rotvec()
    return np.concatenate([ee_pos, eef_rot.astype(np.float64)])


def get_end_effector_pos_quat_wxyz(panda, lula_solver, art_kine_solver):
    """EE pose as (position (3,), quaternion wxyz (4,)), world frame.

    This is the callback consumed by the motion planner / PickPlace.
    """
    ee_pos, ee_rot_mat = read_ee_state(panda, lula_solver, art_kine_solver)
    quat_xyzw = R.from_matrix(ee_rot_mat).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return ee_pos, quat_wxyz.astype(np.float64)


def get_camera_rgb(camera):
    """Return an HxWx3 uint8 RGB frame (alpha stripped) or None if not ready."""
    img = camera.get_rgb()
    if img is None:
        return None
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def build_obs_dict(camera, panda, lula_solver, art_kine_solver, episode_start_mat):
    """Build the policy observation dict for the current world state.

    Mirrors the recorded training fields exactly:
        camera0_rgb                          (H, W, 3) uint8
        robot0_eef_pos                       (3,) float32   (world)
        robot0_eef_rot_axis_angle            (3,) float32   (world, axis-angle)
        robot0_eef_rot_axis_angle_wrt_start  (6,) float32   (rot6d rel. to start)
        robot0_gripper_width                 (1,) float32
    """
    img = get_camera_rgb(camera)
    if img is None:
        return None, episode_start_mat

    ee_pos, ee_rot_mat = read_ee_state(panda, lula_solver, art_kine_solver)
    ee_rot_aa = R.from_matrix(ee_rot_mat).as_rotvec().astype(np.float32)

    curr_mat = np.eye(4)
    curr_mat[:3, :3] = ee_rot_mat
    curr_mat[:3, 3] = ee_pos
    if episode_start_mat is None:
        episode_start_mat = curr_mat.copy()

    rel_mat = np.linalg.inv(episode_start_mat) @ curr_mat
    rel_rot_6d = matrix_to_rot6d(rel_mat[:3, :3]).astype(np.float32)

    obs_dict = {
        "camera0_rgb": img,
        "robot0_eef_pos": ee_pos.astype(np.float32),
        "robot0_eef_rot_axis_angle": ee_rot_aa,
        "robot0_eef_rot_axis_angle_wrt_start": rel_rot_6d,
        "robot0_gripper_width": np.array([get_gripper_width(panda)], dtype=np.float32),
    }
    return obs_dict, episode_start_mat


# ======================================================================
# Generic scripted motion planner (config-driven; SDG-only)
# ======================================================================
class SequencedPickPlace:
    """Runs the pick-place steps in ``cfg.motion_plan`` in order.

    Subsumes the per-task Kitchen/Dining/LivingRoom motion planners: each is just
    a list of ``PickPlaceStep`` with different offsets, object prim paths and the
    fix-target / retreat flags. Steps with ``fix_target`` snapshot the target
    object's world pose once at construction (objects are already placed/settled),
    matching the legacy planners that captured plate_pos / box_pos in __init__.
    """

    def __init__(self, cfg, *, get_object_world_pose_fn, pickplace):
        self.steps = cfg.motion_plan.steps
        self.pickplace = pickplace
        self.get_object_pose = get_object_world_pose_fn
        self.pickplace.grasp_quat = np.asarray(cfg.motion_plan.grasp_quat_wxyz)

        self._fixed_target_pos = []
        for s in self.steps:
            if s.fix_target and s.target_object is not None:
                self._fixed_target_pos.append(self.get_object_pose(s.target_object)[:3, 3])
            else:
                self._fixed_target_pos.append(None)

        self.current_idx = 0
        self.started = False

    def _start_current_step(self):
        s = self.steps[self.current_idx]
        self.pickplace.reset()
        self.pickplace.grasp_mode = s.grasp_mode
        self.pickplace.grasp_extra_descend = s.grasp_extra_descend
        self.pickplace.start(
            pick_above=s.pick_above,
            pick=s.pick,
            lift_offset=s.lift,
            place_above=s.place_above,
            place=s.place,
            attached_object_path=s.attached_object,
            target_object_path=s.target_object,
            fix_target_pose=self._fixed_target_pos[self.current_idx],
            retreat_after_place=s.retreat_after_place,
        )
        self.started = True

    def step(self, panda, lula, ik):
        if self.current_idx >= len(self.steps):
            return
        if not self.started:
            self._start_current_step()
            return
        self.pickplace.step(panda, lula, ik)
        if self.pickplace.is_done():
            self.current_idx += 1
            self.started = False

    def is_done(self):
        return self.current_idx >= len(self.steps)


def build_pickplace(cfg, *, get_object_world_pose_fn, world=None, plan_line_cartesian_fn=None):
    """Construct the low-level PickPlace state machine for ``cfg``."""
    return PickPlace(
        get_end_effector_pose_fn=get_end_effector_pos_quat_wxyz,
        get_object_world_pose_fn=get_object_world_pose_fn,
        apply_ik_solution_fn=apply_ik_solution,
        plan_line_cartesian_fn=plan_line_cartesian_fn or plan_line_cartesian,
        world=world,
        task=cfg.task.pickplace_task_key,
    )


def build_motion_planner(cfg, *, get_object_world_pose_fn, pickplace):
    """Construct the config-driven scripted motion planner."""
    return SequencedPickPlace(
        cfg, get_object_world_pose_fn=get_object_world_pose_fn, pickplace=pickplace
    )
