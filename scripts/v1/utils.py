"""
Utility helpers for the v1 pipeline.

v1 runs put ``scripts/v1`` first on ``sys.path``, so this module SHADOWS the
legacy ``scripts/utils.py`` for every importer in a v1 run — not just v1's own
modules but also the parent modules v1 pulls in (``motion_plan`` imports
``set_prim_world_pose``/``get_preload_prim_path``; ``object_loader`` imports
``set_prim_scale``/``pose_to_transform_matrix``). It therefore provides the full
set of ``from utils import ...`` symbols that v1's dependency graph needs, so v1
is self-contained and does not require ``scripts/`` to be importable.

These are copied verbatim from the legacy ``scripts/utils.py``. The pose-math
helpers are import-safe (numpy/scipy only); the stage helpers import ``isaacsim``/
``omni``/``pxr`` lazily inside the function bodies, so this module is import-safe
and can be loaded before ``SimulationApp`` if needed.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_to_transform_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transformation matrix from position + quaternion (wxyz)."""
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rot_matrix = R.from_quat(quat_xyzw).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = position
    return T


def set_prim_scale(prim, scale) -> None:
    """Set the local scale of a prim (uniform float or [sx, sy, sz])."""
    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale, scale])
    else:
        scale = np.array(scale)

    prim.set_local_scale(scale)


def get_object_pose(object_prim_path: str):
    """Return (position, quat_wxyz) of a prim in the world frame."""
    import isaacsim.core.utils.xforms as xforms_utils
    import isaacsim.core.utils.prims as prims_utils

    prim = prims_utils.get_prim_at_path(object_prim_path)
    pos, rot_quat_wxyz = xforms_utils.get_world_pose(prims_utils.get_prim_path(prim))
    return pos, rot_quat_wxyz


def set_prim_world_pose(prim_path, position, quat_wxyz):
    """Set translate + orient xform ops on an existing prim."""
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")

    xform = UsdGeom.Xformable(prim)

    t_op = None
    r_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            r_op = op

    if t_op is None:
        t_op = xform.AddTranslateOp()

    t_op.Set(Gf.Vec3d(
        float(position[0]),
        float(position[1]),
        float(position[2]),
    ))

    w, x, y, z = [float(v) for v in quat_wxyz]

    if r_op is None:
        r_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        r_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        if r_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            r_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        else:
            r_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def get_preload_prim_path(preload_objects, object_name: str):
    """Look up a preload entry's prim_path by its 'name' field (dict entries)."""
    for entry in preload_objects:
        if entry.get("name") == object_name:
            prim_path = entry.get("prim_path")
            if prim_path:
                return prim_path
    return None
