"""
Declarative configuration schema for the v1 config-based Isaac Sim pipeline.

This module is the SINGLE SOURCE OF TRUTH for an environment's specification.
Everything that used to be scattered across ``base_config`` / ``simple_room_config``
(scene + robot + camera construction data), the ``registry/`` task classes (ArUco
pose, robot/camera pose, success predicate) and ``motion_plan`` per-task planners
(pick/place offsets) is promoted into ONE ``EnvConfig`` dataclass tree.

The dataclasses are pure DATA (plus a couple of callables for success/predicate
logic). All behaviour that interprets them lives in ``engine.py``. This keeps the
module import-safe (no ``isaacsim`` import) so configs can be described and
inspected without a running ``SimulationApp``.

Axes promoted into the config, as requested:
  * scene        -> SceneCfg
  * manipulator  -> ManipulatorCfg (asset, prim paths, gripper, IK, camera)
  * object spawn -> ArucoSpawnCfg | TableSpawnCfg (coordinates + strategy)
  * task         -> TaskCfg (initial EE pose, success predicate)
                  + MotionPlanCfg (the scripted pick-place sequence; SDG-only)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Shared default grasp orientation (was GRASP_QUAT_WXYZ in both config modules
# and the default in motion_plan.PickPlace).
GRASP_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])

# Default observation camera resolution (H, W); matches the legacy configs.
DEFAULT_CAMERA_RESOLUTION = (224, 224)

# Default CAD assets directory for spawnable objects.
DEFAULT_ASSETS_DIR = "/workspace/voilab/assets/CADs"


# ======================================================================
# Scene
# ======================================================================
@dataclass
class PrimPose:
    """Override the world pose of an existing scene prim after the stage opens.

    Used to reposition geometry that ships inside the scene USD (e.g. moving
    simple_room's stock table to the origin). ``prim_path`` is tried first; if it
    is missing, ``name_fallback`` (a substring of the prim's leaf name) is used to
    locate the prim by traversal, so a slightly different authored path still works.
    """
    prim_path: str
    translation: np.ndarray
    rotation_quat_wxyz: np.ndarray
    name_fallback: Optional[str] = None


@dataclass
class SceneCfg:
    """The environment stage."""
    usd_path: str
    # simple_room.usd ships with /Root as its default prim; set this to "World"
    # to rename it in-memory so the rest of the pipeline's /World/* prim paths
    # line up. None (ED305) leaves the stage untouched.
    rename_default_prim_to: Optional[str] = None
    add_default_ground_plane: bool = True
    stage_units_in_meters: float = 1.0
    # Existing in-scene prims to reposition after the stage opens (and after the
    # default-prim rename), e.g. the stock simple_room table.
    prim_pose_overrides: list = field(default_factory=list)  # list[PrimPose]


# ======================================================================
# Manipulator (robot + gripper + IK + observation camera)
# ======================================================================
@dataclass
class GripperCfg:
    joint_prim_names: list
    joint_opened_positions: np.ndarray
    joint_closed_positions: np.ndarray
    action_deltas: np.ndarray


@dataclass
class IKCfg:
    """Lula IK solver configuration.

    Two construction modes:
      * kind="files"     -> explicit robot_description_path + urdf_path (ED305's
                            custom UMI descriptor).
      * kind="supported" -> Isaac's built-in config loaded at runtime via
                            interface_config_loader(robot_name) (FR3).
    """
    kind: str  # "files" | "supported"
    end_effector_frame_name: str
    robot_description_path: Optional[str] = None
    urdf_path: Optional[str] = None
    robot_name: Optional[str] = None


@dataclass
class CameraCfg:
    """Observation (wrist/gopro) camera.

    If ``translation``/``orientation_wxyz`` are None the camera attaches to an
    existing prim without overriding its local pose (ED305's baked gopro mount).
    If provided, the Camera sensor creates the prim with that local pose
    (simple_room's FR3 wrist cam).

    ``convention`` selects the camera-axes frame ``orientation_wxyz`` is
    expressed in when applied as a local pose ("world": +Z up/+X forward,
    "ros": +Z forward/+Y down, "usd": -Z forward/+Y up).
    """
    prim_path: str
    name: str = "gopro_camera"
    resolution: tuple = DEFAULT_CAMERA_RESOLUTION  # (H, W)
    translation: Optional[np.ndarray] = None
    orientation_wxyz: Optional[np.ndarray] = None
    convention: str = "world"


@dataclass
class ManipulatorCfg:
    usd_path: str
    prim_path: str
    name: str
    end_effector_prim_path: str
    gripper: GripperCfg
    ik: IKCfg
    camera: CameraCfg
    # USD variant selections to apply after referencing the robot (ED305:
    # {"Gripper": "AlternateFinger", "Mesh": "Quality"}; FR3: {}).
    variant_selections: dict = field(default_factory=dict)


# ======================================================================
# Robot base pose + viewport (debug) camera
# ======================================================================
@dataclass
class PoseCfg:
    translation: np.ndarray
    rotation_quat_wxyz: np.ndarray


# ======================================================================
# Object spawning (coordinates + strategy)
# ======================================================================
@dataclass
class PreloadObject:
    """An object referenced into the stage once and (re)positioned per episode."""
    name: str
    assets: str
    prim_path: str
    quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    # Table-anchored spawning only: nominal world position of the asset origin.
    table_position: Optional[np.ndarray] = None


@dataclass
class ArucoSpawnCfg:
    """ED305-style placement: ArUco-tag -> world transform from object_poses.json.

    Per episode, each object's pose is read from object_poses.json (in the ArUco
    tag frame), transformed to world, z-clamped, nudged by ``object_pos_offset``
    and oriented by ``orientation_presets``. ``skip_names`` are objects present in
    the JSON but not spawned here (e.g. the pre-baked "plate").
    """
    kind = "aruco"
    preload_objects: list  # list[PreloadObject]
    aruco_tag_pose: dict   # {"translation": (3,), "rotation_quat": wxyz (4,)}
    object_maximum_z_height: float
    object_pos_offset: np.ndarray
    orientation_presets: dict  # name-substring -> quat wxyz (list/np)
    skip_names: tuple = ("plate",)
    assets_dir: str = DEFAULT_ASSETS_DIR


@dataclass
class TableSpawnCfg:
    """simple_room-style placement: spawn objects directly onto the stock table.

    No ArUco/JSON; each object spawns at its ``table_position`` plus a
    deterministic per-episode XY jitter (seeded by episode index).
    """
    kind = "table"
    preload_objects: list  # list[PreloadObject] (must have table_position set)
    table_half_extent_xy: np.ndarray
    spawn_jitter_xy: np.ndarray
    assets_dir: str = DEFAULT_ASSETS_DIR


# ======================================================================
# Task (initial EE pose + success predicate)
# ======================================================================
@dataclass
class TaskCfg:
    # Name passed through to motion_plan.PickPlace; selects its per-task gripper
    # threshold. Kept as the legacy hyphenated values ("kitchen"/"dining-room"/
    # "living-room") so PickPlace.GRIPPER_THRESHOLDS resolves identically.
    pickplace_task_key: str
    # Initial EE pose, expressed as an offset from the settled EE position.
    init_ee_offset: np.ndarray
    init_ee_quat_wxyz: np.ndarray
    # (episode_record) -> bool. Reads live object world poses to decide success.
    success_predicate: Callable[[dict], bool]


# ======================================================================
# Motion plan (the scripted pick-place sequence; SDG-only)
# ======================================================================
@dataclass
class PickPlaceStep:
    """One pick-and-place leg of a scripted task.

    Offsets mirror motion_plan.PickPlace.start()'s arguments. ``target_object``
    is the place reference; when ``fix_target`` is True its world pose is
    snapshotted once at planner-build time (dining/living) instead of tracked
    live (kitchen).
    """
    attached_object: str          # prim path of the object to pick
    target_object: Optional[str]  # prim path of the place reference (or None)
    pick_above: np.ndarray
    pick: np.ndarray
    lift: np.ndarray
    place_above: np.ndarray
    place: np.ndarray
    fix_target: bool = False
    retreat_after_place: bool = False
    grasp_mode: str = "regular"
    grasp_extra_descend: float = 0.0


@dataclass
class MotionPlanCfg:
    steps: list  # list[PickPlaceStep]
    grasp_quat_wxyz: np.ndarray = field(default_factory=lambda: GRASP_QUAT_WXYZ.copy())


# ======================================================================
# Top-level environment config
# ======================================================================
@dataclass
class EnvConfig:
    name: str
    scene: SceneCfg
    manipulator: ManipulatorCfg
    robot_pose: PoseCfg
    view_camera_translation: np.ndarray
    object_spawn: object  # ArucoSpawnCfg | TableSpawnCfg
    task: TaskCfg
    motion_plan: MotionPlanCfg
