"""
Config registry for the v1 pipeline.

Maps a ``--config`` name to a factory returning a fully-populated ``EnvConfig``.
Factories are called (not imported) by the entry points AFTER ``SimulationApp``
has been created, because some configs (simple_room) resolve USD paths from the
Isaac assets root at construction time.

Each name now encodes scene + task in one identifier (task was folded into the
config), e.g. ``ed305_kitchen`` instead of the old ``--config ed305 --task kitchen``.
"""

from . import ed305
from . import simple_room

CONFIG_REGISTRY = {
    "ed305_kitchen": ed305.kitchen,
    "ed305_dining_room": ed305.dining_room,
    "ed305_living_room": ed305.living_room,
    "simple_room_kitchen": simple_room.kitchen,
}


def config_names():
    return list(CONFIG_REGISTRY)


def get_config(name):
    if name not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown config '{name}'. Available: {config_names()}"
        )
    return CONFIG_REGISTRY[name]()
