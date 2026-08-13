"""
Task logic shared by the v1 environment configs.

Holds the success predicates (folded out of the ``registry/`` task classes) and
the scripted pick-place step builders (folded out of the per-task planners in
``motion_plan``). Keeping them here lets ED305 and simple_room reuse the same
kitchen-stacking logic, and keeps the EnvConfig assembly files declarative.

Import-safe: only numpy/scipy plus ``utils.get_object_pose`` (which imports
isaacsim lazily inside the function).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils import get_object_pose
from config import PickPlaceStep


def euler_deg_to_wxyz(euler_deg) -> np.ndarray:
    """Z-up euler (xyz, degrees) -> quaternion wxyz, matching the registries."""
    q_xyzw = R.from_euler("xyz", np.asarray(euler_deg, dtype=float), degrees=True).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


# ======================================================================
# Success predicates (signature: (episode_record) -> bool)
# ======================================================================
def make_kitchen_predicate(blue_prim, pink_prim):
    def predicate(episode_record):
        blue_pos, _ = get_object_pose(blue_prim)
        pink_pos, _ = get_object_pose(pink_prim)
        vertical_order_ok = blue_pos[2] > pink_pos[2]
        xy_dist = np.linalg.norm(blue_pos[:2] - pink_pos[:2])
        xy_alignment_ok = xy_dist < 0.03
        return bool(vertical_order_ok and xy_alignment_ok)
    return predicate


def make_dining_predicate(plate_prim, fork_prim, knife_prim):
    def predicate(episode_record):
        plate_pos, _ = get_object_pose(plate_prim)
        fork_pos, _ = get_object_pose(fork_prim)
        knife_pos, _ = get_object_pose(knife_prim)

        max_dist_xy = 0.15
        fork_near_plate = np.linalg.norm(fork_pos[:2] - plate_pos[:2]) <= max_dist_xy
        knife_near_plate = np.linalg.norm(knife_pos[:2] - plate_pos[:2]) <= max_dist_xy
        fork_on_left = fork_pos[1] > plate_pos[1]
        knife_on_right = knife_pos[1] < plate_pos[1]
        return bool(fork_near_plate and knife_near_plate and fork_on_left and knife_on_right)
    return predicate


def make_living_predicate(blue_prim, green_prim, red_prim,
                          box_min=(1.41298, 10.87098, 0.70),
                          box_max=(1.61221, 11.08388, 0.81762)):
    box_min = np.array(box_min)
    box_max = np.array(box_max)

    def in_box(p):
        return (box_min[0] <= p[0] <= box_max[0]
                and box_min[1] <= p[1] <= box_max[1]
                and box_min[2] <= p[2] <= box_max[2])

    def predicate(episode_record):
        blue_pos, _ = get_object_pose(blue_prim)
        green_pos, _ = get_object_pose(green_prim)
        red_pos, _ = get_object_pose(red_prim)
        return bool(in_box(blue_pos) and in_box(green_pos) and in_box(red_pos))
    return predicate


# ======================================================================
# Scripted pick-place step builders (the SDG-only motion plan)
# ======================================================================
def kitchen_motion_steps(blue_prim, pink_prim):
    """Stack the blue cup onto the pink cup (mirrors KitchenMotionPlanner)."""
    return [
        PickPlaceStep(
            attached_object=blue_prim,
            target_object=pink_prim,
            pick_above=np.array([-0.0, -0.0, 0.20]),
            pick=np.array([-0.055, -0.08, -0.12]),
            lift=np.array([-0.050, 0.0, 0.4]),
            place_above=np.array([-0.045, -0.07, 0.2]),
            place=np.array([-0.045, -0.07, 0.01]),
            fix_target=False,
            retreat_after_place=False,
            grasp_mode="regular",
            # Descend an extra 3 cm so the fingers span the cup wall, not the
            # oval bottom edge (less slip on lift).
            grasp_extra_descend=0.03,
        )
    ]


def dining_motion_steps(fork_prim, knife_prim, plate_prim):
    """Place fork left / knife right of the plate (mirrors DiningRoomMotionPlanner)."""
    common = dict(
        target_object=plate_prim,
        pick_above=np.array([-0.06, -0.06, 0.10]),
        lift=np.array([0.0, -0.05, 0.25]),
        place_above=np.array([0.0, -0.05, 0.20]),
        fix_target=True,
        retreat_after_place=True,
        grasp_mode="object_based",
    )
    return [
        PickPlaceStep(
            attached_object=fork_prim,
            pick=np.array([-0.05, -0.06, -0.08]),
            place=np.array([-0.05, 0.03, 0.04]),
            **common,
        ),
        PickPlaceStep(
            attached_object=knife_prim,
            pick=np.array([-0.06, -0.06, -0.08]),
            place=np.array([-0.05, -0.15, 0.04]),
            **common,
        ),
    ]


def living_motion_steps(red_prim, blue_prim, green_prim, box_prim):
    """Drop the three blocks into the storage box (mirrors LivingRoomMotionPlanner).

    Order matches the legacy planner: red (triangle), blue (cylinder), green
    (bridge).
    """
    common = dict(
        target_object=box_prim,
        lift=np.array([0.0, 0.0, 0.2]),
        place_above=np.array([-0.20, -0.10, 0.20]),
        fix_target=True,
        retreat_after_place=True,
        grasp_mode="regular",
    )
    specs = [
        (red_prim, np.array([-0.06, -0.075, 0.10]), np.array([-0.055, -0.075, -0.088]),
         np.array([-0.20, -0.10, 0.09])),
        (blue_prim, np.array([-0.06, -0.062, 0.10]), np.array([-0.06, -0.062, -0.088]),
         np.array([-0.15, -0.10, 0.09])),
        (green_prim, np.array([-0.065, -0.06, 0.10]), np.array([-0.065, -0.06, -0.088]),
         np.array([-0.25, -0.10, 0.09])),
    ]
    return [
        PickPlaceStep(
            attached_object=prim,
            pick_above=pick_above,
            pick=pick,
            place=place,
            **common,
        )
        for prim, pick_above, pick, place in specs
    ]
