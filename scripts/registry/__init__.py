from .kitchen_registry import KitchenTaskRegistry
from .dining_table_registry import DiningTableTaskRegistry
from .living_room_registry import LivingRoomTaskRegistry

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