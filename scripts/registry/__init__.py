from .kitchen_registry import KitchenTaskRegistry
from .dining_room_registry import DiningRoomTaskRegistry
from .living_room_registry import LivingRoomTaskRegistry
from motion_plan import KitchenMotionPlanner, DiningRoomMotionPlanner, LivingRoomMotionPlanner

# Registry mapping
TASK_REGISTRIES = {
    "kitchen": KitchenTaskRegistry,
    "dining-room": DiningRoomTaskRegistry,
    "living-room": LivingRoomTaskRegistry,
}

def get_task_registry(task_name: str):
    """Get task registry by name"""
    if task_name not in TASK_REGISTRIES:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_REGISTRIES.keys())}")
    return TASK_REGISTRIES[task_name]


def get_episode_completion_fn(task_name: str):
    registry_cls = get_task_registry(task_name)
    return getattr(registry_cls, "is_episode_completed", lambda record: True)


MOTION_PLANNER_FACTORIES = {
    "kitchen": lambda cfg, *, get_object_world_pose_fn, pickplace: (
        KitchenMotionPlanner(
            cfg,
            get_object_world_pose_fn=get_object_world_pose_fn,
            pickplace=pickplace
        )
    ),
    "dining-room": lambda cfg, *, get_object_world_pose_fn, pickplace: (
        DiningRoomMotionPlanner(
            cfg,
            get_object_world_pose_fn=get_object_world_pose_fn,
            pickplace=pickplace
        )
    ),
    "living-room": lambda cfg, *, get_object_world_pose_fn, pickplace: (
        LivingRoomMotionPlanner(
            cfg,
            get_object_world_pose_fn=get_object_world_pose_fn,
            pickplace=pickplace
        )
    ),
}


def get_motion_planner(
    task_name: str,
    cfg,
    *,
    get_object_world_pose_fn=None,
    pickplace=None,
):
    if task_name not in MOTION_PLANNER_FACTORIES:
        raise ValueError(
            f"Unknown task: {task_name}. Available tasks: {list(MOTION_PLANNER_FACTORIES.keys())}"
        )

    missing = [
        name
        for name, fn in {
            "get_object_world_pose_fn": get_object_world_pose_fn,
            "pickplace": pickplace,
        }.items()
        if fn is None
    ]
    if missing:
        raise ValueError(f"Missing motion planner dependencies: {', '.join(missing)}")

    return MOTION_PLANNER_FACTORIES[task_name](
        cfg,
        get_object_world_pose_fn=get_object_world_pose_fn,
        pickplace=pickplace,
    )
