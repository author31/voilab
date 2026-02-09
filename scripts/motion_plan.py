import numpy as np
from scipy.spatial.transform import Rotation as R
from umi_replay import set_gripper_width
from utils import set_prim_world_pose, get_preload_prim_path
from scipy.spatial.transform import Rotation as R
from pynput import keyboard

class PickPlace:
    GRIPPER_THRESHOLDS = {
        "kitchen": 0.038,
        "dining-room": 0.004,
        "living-room": 0.004,
    }
    def __init__(
        self,
        *,
        get_end_effector_pose_fn,
        get_object_world_pose_fn,
        apply_ik_solution_fn,
        plan_line_cartesian_fn,
        grasp_quat_wxyz=np.array([
            0.0081739, -0.9366365, 0.350194, 0.0030561
        ]),
        grasp_mode="regular",
        open_width=0.04,
        close_width=0.00,
        close_steps=30,
        hold_steps=10,
        step_move=0.005,
        step_descend=0.005,
        #
        attach_dist_thresh=0.1,
        release_dist_thresh = 0.13,
        gripper_close_thresh = 0.085,
        gripper_open_thresh = 0.0875,
        world=None,
        task=None
        ):
        self.get_ee_pose = get_end_effector_pose_fn
        self.get_obj_pose = get_object_world_pose_fn
        self.apply_ik = apply_ik_solution_fn
        self.plan = plan_line_cartesian_fn

        self.grasp_quat = np.asarray(grasp_quat_wxyz)
        self.grasp_mode = grasp_mode
        self.open_width = open_width
        self.close_width = close_width
        self.close_steps = close_steps
        self.hold_steps = hold_steps
        self.step_move = step_move
        self.step_descend = step_descend
        
        self.world = world
        self.task = task
        self.prev_grip_width = None
        self.stall_count = 0
        self.cnt = 0
        self.reach = False
        self.stall = False
        self.width = None

        self.attach_dist_thresh = attach_dist_thresh
        self.release_dist_thresh = release_dist_thresh
        self.gripper_close_thresh = gripper_close_thresh
        self.gripper_open_thresh  = gripper_open_thresh
        self.reset()

    
    def reset(self):
        self.phase = "idle"
        self.traj = []
        self.i = 0
        self.counter = 0

        # --- attach ---
        self.attached = False
        self.T_ee_to_obj = None
        self.attached_object_path = None
        self.target_object_path = None

    
    def start(self, pick_above, pick, lift_offset, place_above, place, 
            attached_object_path=None, target_object_path=None,
            fix_target_pose=None, retreat_after_place=False):
        self.pick_above = pick_above
        self.pick = pick
        self.place_above = place_above
        self.lift_offset = lift_offset
        self.place = place

        self.attached_object_path = attached_object_path
        self.target_object_path = target_object_path
        self.fix_target_pose = fix_target_pose
        self.retreat_after_place = retreat_after_place

        self.attached = False
        self.T_ee_to_obj = None

        self.phase = "move_above"
        self.traj = []
        self.i = 0
        self.counter = 0
        self.close_counter = 0
        self.prev_grip_width = None
        self.stall_count = 0
    
    def _compute_grasp_quat_from_object(self, obj_path):
        # --- 1. Get object pose in world frame ---
        T_obj = self.get_obj_pose(obj_path)
        R_obj = T_obj[:3, :3]

        # Object local axes in world coordinates
        x_obj = R_obj[:, 0]  # object long axis
        y_obj = R_obj[:, 1]
        z_obj = R_obj[:, 2]  # object up axis

        # --- 2. Compute yaw of the object long axis projected onto the XY plane ---
        x_xy = x_obj.copy()
        x_xy[2] = 0.0  # project onto XY plane

        if np.linalg.norm(x_xy) < 1e-6:
            yaw = 0.0
        else:
            x_xy /= np.linalg.norm(x_xy)
            yaw = np.arctan2(x_xy[1], x_xy[0])  # radians

        # --- 3. Yaw rotation matrix around world Z-axis ---
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw),  np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]
        ])

        # --- 4. (Redundant but explicit) Extract yaw from object local X axis ---
        # Object local X axis expressed in world frame
        x_axis_world = R_obj @ np.array([1.0, 0.0, 0.0])

        # Project to XY plane
        x_axis_xy = x_axis_world.copy()
        x_axis_xy[2] = 0.0

        # Safety check for degenerate cases
        if np.linalg.norm(x_axis_xy) < 1e-6:
            yaw = 0.0
        else:
            x_axis_xy /= np.linalg.norm(x_axis_xy)
            yaw = np.arctan2(x_axis_xy[1], x_axis_xy[0])

        yaw_deg = np.degrees(yaw)

        # --- 5. Construct end-effector frame using yaw only ---
        # Tool Z-axis points downward in world frame
        z_ee = np.array([0.0, 0.0, -1.0])

        # Tool X-axis aligned with object yaw direction
        x_ee = np.array([
            np.cos(yaw),
            np.sin(yaw),
            0.0
        ])

        # Tool Y-axis via right-hand rule
        y_ee = np.cross(z_ee, x_ee)

        # Normalize axes for numerical stability
        x_ee /= np.linalg.norm(x_ee)
        y_ee /= np.linalg.norm(y_ee)

        # End-effector rotation matrix (geometry-based)
        R_ee_geom = np.column_stack([x_ee, y_ee, z_ee])

        # --- 6. Apply grasp-to-tool rotational offset (yaw-only) ---
        # Example: rotate tool by -45 degrees around Z-axis
        R_offset = R.from_euler('z', -45.0, degrees=True).as_matrix()
        R_ee = R_ee_geom @ R_offset

        # --- 7. Disambiguate object orientation (knife / fork) using up direction ---
        z_after_yaw = R_yaw @ np.array([0.0, 0.0, 1.0])
        dot = np.dot(z_obj, z_after_yaw)

        # If object orientation is flipped, rotate end-effector yaw by 180 degrees
        if ("knife" in obj_path.lower() and dot > 0) or ("fork" in obj_path.lower() and dot < 0):
            R_flip = R.from_euler('z', 180.0, degrees=True).as_matrix()
            R_ee = R_ee @ R_flip

        # --- 8. Convert final rotation matrix to quaternion (w, x, y, z) ---
        quat = R.from_matrix(R_ee).as_quat(scalar_first=True)
        return quat

    def _run_traj(self, panda, lula, ik, target, step):
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        # ---------- 初始化 trajectory ----------
        if not self.traj:

            # 重新計算 grasp quaternion（跟著物體姿態）
            if self.attached_object_path is not None and self.task in ["dining_room"]:
                self.grasp_quat = self._compute_grasp_quat_from_object(self.attached_object_path)

            ee_pos, ee_quat = self.get_ee_pose(panda, lula, ik)

            # 用新的 grasp_quat 進行 motion planning
            cart_traj = self.plan(ee_pos, ee_quat, target, self.grasp_quat, step)

            self.traj = []
            for wp in cart_traj:
                action, success = ik.compute_inverse_kinematics(wp[:3], wp[3:])
                if not success:
                    raise RuntimeError("IK failed for waypoint")
                qpos = np.array(action.joint_positions, dtype=np.float32)
                self.traj.append(qpos)

            self.i = 0

        # ---------- trajectory 結束 ----------
        if self.i >= len(self.traj):
            self.traj = []
            return True

        # ---------- 開爪 ----------
        if self.phase not in ["close", "lift", "move_place", "descend_place"]:
            set_gripper_width(panda, 0.04)
        else:
            panda.gripper.apply_action(
                    ArticulationAction(
                        joint_positions=np.array([self.close_width, self.close_width], dtype=np.float32),
                        joint_indices=np.array([7, 8])
                    )
                )

        # ---------- 移動手臂 ----------
        panda.apply_action(
            ArticulationAction(
                joint_positions=self.traj[self.i],
                joint_indices=np.arange(7)
            )
        )

        self.i += 1
        return False
    
    def _object_target(self, obj_path, offset_obj):
        # Get object pose in world frame
        T_obj = self.get_obj_pose(obj_path)
        p_obj = T_obj[:3, 3]
        return p_obj + offset_obj

    def _place_target(self, offset):
        if self.fix_target_pose is not None:
            return np.asarray(self.fix_target_pose) + offset
        return self._object_target(self.target_object_path, offset)
    
    def close_gripper(self, panda, target_width, step=0.005, threshold=0.005, min_step=0.0005):
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        idx1, idx2 = 7, 8
        current_positions = panda.gripper.get_joint_positions()
        current_width = np.mean(current_positions)

        # ===== 初始化 prev_grip_width =====
        if self.prev_grip_width is None:
            self.prev_grip_width = current_width
            return current_width, False, False

        # ===== 計算剩餘距離並動態縮小步伐 =====
        diff = target_width - current_width
        dynamic_step = max(min_step, step * abs(diff) / step)  # 距離越小，步伐越小
        move = np.clip(diff, -dynamic_step, dynamic_step)
        new_positions = current_positions + move

        # ===== 套用位置控制 =====
        panda.gripper.apply_action(
            ArticulationAction(
                joint_positions=new_positions.astype(np.float32),
                joint_indices=np.array([idx1, idx2])
            )
        )

        # ===== detect stall =====
        delta = abs(current_width - self.prev_grip_width)
        self.prev_grip_width = current_width

        stalled = False
        if delta < 1e-3:
            self.stall_count += 1
        else:
            self.stall_count = 0

        if self.stall_count > 100:
            stalled = True

        reached = abs(target_width - current_width) < threshold

        return current_width, reached, stalled

    def step(self, panda, lula, ik):
        if self.phase in ["idle", "done"]:
            return

        if self.phase == "move_above":
            self._step_move_above(panda, lula, ik)
        elif self.phase == "descend":
            self._step_descend(panda, lula, ik)
        elif self.phase == "close":
            self._step_close_gripper(panda)
        elif self.phase == "lift":
            self._step_lift(panda, lula, ik)
        elif self.phase == "move_place":
            self._step_move_place(panda, lula, ik)
        elif self.phase == "descend_place":
            self._step_descend_place(panda, lula, ik)
        elif self.phase == "release":
            self._step_release(panda)
        elif self.phase == "post_place_lift":
            self._step_post_place_lift(panda, lula, ik)
        else:
            raise RuntimeError(f"Unknown phase: {self.phase}")

    # ---------------- PHASE FUNCTIONS ----------------
    def _step_move_above(self, panda, lula, ik):
        target = self._object_target(self.attached_object_path, self.pick_above)
        if self._run_traj(panda, lula, ik, target, self.step_move):
            self.phase = "descend"

    def _step_descend(self, panda, lula, ik):
        target = self._object_target(self.attached_object_path, self.pick)
        if self._run_traj(panda, lula, ik, target, self.step_descend):
            self.phase = "close"
            self.close_counter = 0
            self.prev_grip_width = None
            self.reach = False
            self.stall = False
            self.stall_count = 0

    def _step_close_gripper(self, panda):
        self.close_counter += 1
        threshold = self.GRIPPER_THRESHOLDS.get(self.task, 0.004)

        self.width, self.reached, self.stalled = self.close_gripper(
            panda,
            target_width=self.close_width,
            step=0.01,
            threshold=threshold
        )

        if self.reached or self.stalled:
            self.phase = "lift"

        if self.close_counter > 300:
            self.phase = "lift"

    def _step_lift(self, panda, lula, ik):
        ee_pos, _ = self.get_ee_pose(panda, lula, ik)
        target = ee_pos + self.lift_offset
        if self._run_traj(panda, lula, ik, target, self.step_move):
            self.phase = "move_place"

    def _step_move_place(self, panda, lula, ik):
        target = self._place_target(self.place_above)
        if self._run_traj(panda, lula, ik, target, self.step_move):
            self.phase = "descend_place"

    def _step_descend_place(self, panda, lula, ik):
        target = self._place_target(self.place)
        if self._run_traj(panda, lula, ik, target, self.step_descend):
            self.phase = "release"
            self.cnt = 0

    def _step_release(self, panda):
        self.cnt += 1
        # 直接打開 gripper
        from isaacsim.core.utils.types import ArticulationAction
        panda.gripper.apply_action(
            ArticulationAction(
                joint_efforts=np.array([0.0, 0.0], dtype=np.float32),
                joint_positions=np.array([0.04, 0.04], dtype=np.float32),
                joint_indices=np.array([7, 8])
            )
        )

        if self.cnt > 50:
            self.phase = "post_place_lift" if self.retreat_after_place else "done"

    def _step_post_place_lift(self, panda, lula, ik):
        ee_pos, _ = self.get_ee_pose(panda, lula, ik)
        target = ee_pos + self.lift_offset
        if self._run_traj(panda, lula, ik, target, self.step_move):
            self.phase = "done"

    # ---------------- UTILS ----------------
    def is_done(self):
        return self.phase == "done"


class DiningRoomMotionPlanner:
    def __init__(self, cfg, *, get_object_world_pose_fn, pickplace):
        self.cfg = cfg
        self.get_object_pose = get_object_world_pose_fn
        self.pickplace = pickplace
        self.started = False

        env = cfg["environment_vars"]
        self.cutlery = [
            env["FORK_PATH"],
            env["KNIFE_PATH"],
        ]
        self.plate = env["PLATE_PATH"]
        plate_T = self.get_object_pose(self.plate)
        self.plate_pos = plate_T[:3, 3]

        self.pick_above_offset  = [np.array([-0.06, -0.06,  0.10])
                                   , np.array([-0.06, -0.06,  0.10])]
        self.pick_offset        = [np.array([-0.05, -0.06, -0.08])
                                   , np.array([-0.06, -0.06, -0.08])]
        self.lift_offset        = [np.array([ 0.0,  -0.05,  0.25])
                                   , np.array([ 0.0,  -0.05,  0.25])]
        self.place_above_offset = [np.array([ 0.0,  -0.05,  0.20])
                                      , np.array([ 0.0,  -0.05,  0.20])]
        self.place_offsets = [
            np.array([-0.05, 0.03, 0.04]),
            np.array([-0.05, -0.15, 0.04]),
        ]
        
        self.current_idx = 0
        self.started = False
    
    def _start_pickplace_for_current_cutlery(self):
        self.pickplace.reset()
        self.pickplace.grasp_mode = "object_based"
        self.pickplace.start(
            pick_above  = self.pick_above_offset[self.current_idx],
            pick        = self.pick_offset[self.current_idx],
            lift_offset = self.lift_offset[self.current_idx],
            place_above = self.place_above_offset[self.current_idx],
            place       = self.place_offsets[self.current_idx],
            attached_object_path = self.cutlery[self.current_idx],
            target_object_path   = self.plate,
            fix_target_pose = self.plate_pos,
            retreat_after_place = True,
        )
        self.started = True

    def step(self, panda, lula, ik):
        if self.current_idx >= len(self.cutlery):
            return

        if not self.started:
            self._start_pickplace_for_current_cutlery()
            return

        self.pickplace.step(panda, lula, ik)

        if self.pickplace.is_done():
            self.current_idx += 1
            self.started = False
    
    def is_done(self):
        return self.current_idx >= len(self.cutlery)

class KitchenMotionPlanner:
    def __init__(self, cfg, *, get_object_world_pose_fn, pickplace):
        self.cfg = cfg
        self.get_object_pose = get_object_world_pose_fn
        self.pickplace = pickplace
        self.started = False

        env = cfg.get("environment_vars", {})
        preload_objects = env.get("PRELOAD_OBJECTS", [])
        self.blue = get_preload_prim_path(preload_objects, "blue cup")
        self.pink = get_preload_prim_path(preload_objects, "pink cup")
        if self.blue is None or self.pink is None:
            fallback = [entry.get("prim_path") for entry in preload_objects]
            fallback = [path for path in fallback if path]
            if self.blue is None and len(fallback) > 0:
                self.blue = fallback[0]
            if self.pink is None and len(fallback) > 1:
                self.pink = fallback[1]
        if self.blue is None or self.pink is None:
            raise ValueError("Missing PRELOAD_OBJECTS prim_path for kitchen cups.")


        self.pick_above_offset  = np.array([-0.0, -0.0,  0.20])
        self.pick_offset        = np.array([-0.055, -0.08, -0.12])
        self.lift_offset        = np.array([-0.050,   0.0,    0.4])
        self.place_above_offset = np.array([-0.045, -0.07,  0.2])
        self.place_offset       = np.array([-0.045, -0.07,  0.01])
    


    def step(self, panda, lula, ik):
        if not self.started:
            self.pickplace.reset()
            self.pickplace.grasp_mode = "regular"
            self.pickplace.start(
                pick_above  = self.pick_above_offset,
                pick        = self.pick_offset,
                lift_offset = self.lift_offset,
                place_above = self.place_above_offset,
                place       = self.place_offset,
                attached_object_path = self.blue,
                target_object_path = self.pink
            )
            self.started = True
            return

        self.pickplace.step(panda, lula, ik)

    def is_done(self):
        return self.pickplace.is_done()
    
class LivingRoomMotionPlanner:
    def __init__(self, cfg, *, get_object_world_pose_fn, pickplace):
        self.cfg = cfg
        self.get_object_pose = get_object_world_pose_fn
        self.pickplace = pickplace
        self.started = False

        env = cfg["environment_vars"]
        self.blocks = [
            env["RED_BLOCK_PATH"],
            env["BLUE_BLOCK_PATH"],
            env["GREEN_BLOCK_PATH"],
        ]
        self.storage_box = env["STORAGE_BOX_PATH"]
        box_T = self.get_object_pose(self.storage_box)
        self.box_pos = box_T[:3, 3]

        self.pick_above_offset  = [np.array([-0.06, -0.075,  0.10])
                                   , np.array([-0.06, -0.062,  0.10])
                                      , np.array([-0.065, -0.06, 0.10])]
        self.pick_offset        = [np.array([-0.055, -0.075, -0.088])
                                   , np.array([-0.06, -0.062, -0.088])
                                      , np.array([-0.065, -0.06, -0.088])]
        self.lift_offset        = [np.array([ 0.0,   0.0,    0.2])
                                   , np.array([ 0.0,   0.0,    0.2])
                                      , np.array([ 0.0,   0.0,    0.2])]
        self.place_above_offset = [np.array([-0.20, -0.10,  0.20])
                                   , np.array([-0.20, -0.10,  0.20])
                                      , np.array([-0.20, -0.10,  0.20])]
        self.place_offsets = [
            np.array([-0.20, -0.10, 0.09]),
            np.array([-0.15, -0.10, 0.09]),
            np.array([-0.25, -0.10, 0.09]),
        ]

        self.current_idx = 0
        self.started = False
    
    def _start_pickplace_for_current_block(self):
        self.pickplace.reset()
        self.pickplace.grasp_mode = "regular"
        self.pickplace.start(
            pick_above  = self.pick_above_offset[self.current_idx],
            pick        = self.pick_offset[self.current_idx],
            lift_offset = self.lift_offset[self.current_idx],
            place_above = self.place_above_offset[self.current_idx],
            place       = self.place_offsets[self.current_idx],
            attached_object_path = self.blocks[self.current_idx],
            target_object_path   = self.storage_box,
            fix_target_pose = self.box_pos,
            retreat_after_place=True,
        )
        self.started = True

    def step(self, panda, lula, ik):
        if self.current_idx >= len(self.blocks):
            return

        if not self.started:
            self._start_pickplace_for_current_block()
            return

        self.pickplace.step(panda, lula, ik)

        if self.pickplace.is_done():
            self.current_idx += 1
            self.started = False
    
    def is_done(self):
        return self.current_idx >= len(self.blocks)