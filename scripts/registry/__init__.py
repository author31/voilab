from .kitchen_registry import KitchenTaskRegistry
from .dining_table_registry import DiningTableTaskRegistry
from .living_room_registry import LivingRoomTaskRegistry
from motion_plan import KitchenMotionPlanner, DiningRoomMotionPlanner, LivingRoomMotionPlanner

# Registry mapping
TASK_REGISTRIES = {
    "kitchen": KitchenTaskRegistry,
    "dining-table": DiningTableTaskRegistry,
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
    "kitchen": lambda cfg, *, get_end_effector_pose_fn, get_object_world_pose_fn, apply_ik_solution_fn: (
        KitchenMotionPlanner(
            cfg,
            get_end_effector_pose_fn=get_end_effector_pose_fn,
            get_object_world_pose_fn=get_object_world_pose_fn,
            apply_ik_solution_fn=apply_ik_solution_fn,
        )
    ),
    "dining-table": lambda cfg, *, get_end_effector_pose_fn, get_object_world_pose_fn, apply_ik_solution_fn: (
        DiningRoomMotionPlanner(
            cfg,
            get_end_effector_pose_fn=get_end_effector_pose_fn,
            get_object_world_pose_fn=get_object_world_pose_fn,
            apply_ik_solution_fn=apply_ik_solution_fn,
        )
    ),
    "living-room": lambda cfg, *, get_end_effector_pose_fn, get_object_world_pose_fn, apply_ik_solution_fn: (
        LivingRoomMotionPlanner(
            cfg,
            get_end_effector_pose_fn=get_end_effector_pose_fn,
            get_object_world_pose_fn=get_object_world_pose_fn,
            apply_ik_solution_fn=apply_ik_solution_fn,
        )
    ),
}


def get_motion_planner(
    task_name: str,
    cfg,
    *,
    get_end_effector_pose_fn=None,
    get_object_world_pose_fn=None,
    apply_ik_solution_fn=None,
):
    if task_name not in MOTION_PLANNER_FACTORIES:
        raise ValueError(
            f"Unknown task: {task_name}. Available tasks: {list(MOTION_PLANNER_FACTORIES.keys())}"
        )

    missing = [
        name
        for name, fn in {
            "get_end_effector_pose_fn": get_end_effector_pose_fn,
            "get_object_world_pose_fn": get_object_world_pose_fn,
            "apply_ik_solution_fn": apply_ik_solution_fn,
        }.items()
        if fn is None
    ]
    if missing:
        raise ValueError(f"Missing motion planner dependencies: {', '.join(missing)}")

    return MOTION_PLANNER_FACTORIES[task_name](
        cfg,
        get_end_effector_pose_fn=get_end_effector_pose_fn,
        get_object_world_pose_fn=get_object_world_pose_fn,
        apply_ik_solution_fn=apply_ik_solution_fn,
    )
