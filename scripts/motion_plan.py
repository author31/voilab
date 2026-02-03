import numpy as np
from scipy.spatial.transform import Rotation as R
from umi_replay import set_gripper_width
from utils import set_prim_world_pose, get_preload_prim_path
from scipy.spatial.transform import Rotation as R
from pynput import keyboard

class PickPlace:
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
        open_width=0.08,
        close_width=0.03,
        close_steps=30,
        hold_steps=10,
        step_move=0.01,
        step_descend=0.005,
        #
        attach_dist_thresh=0.1,
        release_dist_thresh = 0.13,
        gripper_close_thresh = 0.085,
        gripper_open_thresh = 0.0875,
        teleop=None
        ):
        self.teleop = teleop
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

        self.attach_dist_thresh = attach_dist_thresh
        self.release_dist_thresh = release_dist_thresh
        self.gripper_close_thresh = gripper_close_thresh
        self.gripper_open_thresh  = gripper_open_thresh
        self.teleop_yaw = 0.0
        self.R_grasp_to_tool = R.from_euler(
            'xyz', [0.0, 0.0, 45.0], degrees=True).as_matrix()
        self._teleop_prev_enabled = False
        self._force_rebuild_once = False
        self.reset()

    # -------------------------
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

    # -------------------------
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

    # -------------------------
    def _teleop_step(self, panda, lula, ik):
        # sync object
        self._sync_attached_object(panda, lula, ik)

        dpos, dyaw, grip, next_ep = self.teleop.consume()

        if np.linalg.norm(dpos) < 1e-4:
            dpos[:] = 0.0
        if abs(dyaw) < 1e-3:
            dyaw = 0.0

        ee_pos, _ = self.get_ee_pose(panda, lula, ik)
        target_pos = ee_pos + dpos

        # yaw state
        self.teleop_yaw += dyaw
        self.teleop_yaw = (self.teleop_yaw + 180) % 360 - 180

        R_base = R.from_quat(self.grasp_quat, scalar_first=True)
        R_yaw  = R.from_euler('z', self.teleop_yaw, degrees=True)
        quat_wxyz = (R_yaw * R_base).as_quat(scalar_first=True)

        self.apply_ik(panda, ik, target_pos, quat_wxyz)

        if grip == "close":
            set_gripper_width(panda, self.close_width)
            self._try_attach_object(panda, lula, ik)

        elif grip == "open":
            set_gripper_width(panda, self.open_width)
            self.attached = False
            self.T_ee_to_obj = None

        return next_ep
    

    def _try_attach_object(self, panda, lula, ik):
        if self.attached:
            return

        if self.attached_object_path is None:
            return

        ee_pos, ee_quat = self.get_ee_pose(panda, lula, ik)
        T_obj = self.get_obj_pose(self.attached_object_path)
        obj_pos = T_obj[:3, 3]

        dist = np.linalg.norm(obj_pos - ee_pos)
        if dist > self.attach_dist_thresh:
            return

        # --- build EE transform ---
        quat_xyzw = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
        T_ee = np.eye(4)
        T_ee[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T_ee[:3, 3] = ee_pos

        # --- attach ---
        self.T_ee_to_obj = np.linalg.inv(T_ee) @ T_obj
        self.attached = True

        print(f"[Teleop] Object attached: {self.attached_object_path}")
        
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


    # -------------------------
    def _run_traj(self, panda, lula, ik, target, step):
        if (not self.traj) or self._force_rebuild_once:
            self._force_rebuild_once = False   # 🔑 CHỈ 1 LẦN

            p, q = self.get_ee_pose(panda, lula, ik)

            if self.phase in ["move_above", "descend", "close"]:
                if self.grasp_mode == "object_based":
                    grasp_quat = self._compute_grasp_quat_from_object(
                        self.attached_object_path)
                else:
                    grasp_quat = self.grasp_quat
            else:
                grasp_quat = self.grasp_quat

            self.traj = self.plan(p, q, target, grasp_quat, step_m=step)
            self.i = 0
            print("[PickPlace] new plan (force rebuild =", self._force_rebuild_once, ")")

        if self.i >= len(self.traj):
            self.traj = []
            self.i = 0
            return True

        wp = self.traj[self.i]
        self.apply_ik(panda, ik, wp[:3], wp[3:])
        set_gripper_width(
            panda,
            self.close_width if self.attached else self.open_width
        )

        self.i += 1
        return False

    
    # -------------------------
    def _sync_attached_object(self, panda, lula, ik):
        if (not self.attached) or (self.T_ee_to_obj is None) or (self.attached_object_path is None):
            return
        ee_pos, ee_quat_wxyz = self.get_ee_pose(panda, lula, ik)
        
        quat_xyzw = np.array([ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]])
        T_ee = np.eye(4)
        T_ee[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T_ee[:3, 3] = ee_pos
        T_obj = T_ee @ self.T_ee_to_obj
        pos = T_obj[:3, 3]
        quat_wxyz = R.from_matrix(T_obj[:3, :3]).as_quat(scalar_first=True)  # wxyz

        set_prim_world_pose(self.attached_object_path, pos, quat_wxyz)

    # -------------------------
# -------------------------
    def _object_target(self, obj_path, offset_obj):
        # Get object pose in world frame
        T_obj = self.get_obj_pose(obj_path)
        R_obj = T_obj[:3, :3]
        p_obj = T_obj[:3, 3]

        # Object local axes
        x_obj = R_obj[:, 0]  # object long axis
        z_obj = R_obj[:, 2]  # object up axis

        # --- 1. Compute yaw of the object long axis projected onto the XY plane ---
        x_xy = x_obj.copy()
        x_xy[2] = 0  # project onto XY plane
        if np.linalg.norm(x_xy) < 1e-6:
            yaw = 0.0
        else:
            x_xy /= np.linalg.norm(x_xy)
            yaw = np.arctan2(x_xy[1], x_xy[0])

        # --- 2. Yaw rotation matrix around world Z-axis ---
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # --- 3. Check object up direction after yaw alignment ---
        # Used to disambiguate orientation (e.g., knife vs. fork)
        z_after_yaw = R_yaw @ np.array([0, 0, 1])
        dot = np.dot(z_obj, z_after_yaw)

        # Apply object-specific offset correction if orientation is flipped
        #if ("knife" in obj_path.lower() and dot > 0) or ("fork" in obj_path.lower() and dot < 0):
        #    offset_obj[1] = 0.12
        #    offset_obj[0] = -0.06
            # Optional: rotate yaw by 180 degrees if full orientation flip is required
            # R_knife_flip = R.from_euler('z', -180, degrees=True).as_matrix()
            # R_yaw = R_yaw @ R_knife_flip

        # --- 4. Transform offset from object-aligned frame to world frame ---
        offset_world = R_yaw @ offset_obj
        return p_obj + offset_obj


    # -------------------------
    def _place_target(self, offset):
        if self.fix_target_pose is not None:
            return np.asarray(self.fix_target_pose) + offset
        return self._object_target(self.target_object_path, offset)

    # -------------------------
    def step(self, panda, lula, ik):
        # ---- TELEOP EDGE DETECTION ----
        teleop_enabled = self.teleop.enabled if self.teleop else False

        # phát hiện True -> False
        if self._teleop_prev_enabled and not teleop_enabled:
            self._force_rebuild_once = True
            self.traj = []
            self.i = 0
            print("[PickPlace] Teleop → Auto: force rebuild traj once")

        self._teleop_prev_enabled = teleop_enabled

        if self.teleop and self.teleop.enabled:
            next_ep = self._teleop_step(panda, lula, ik)
            if next_ep:
                self.phase = "done"
            return

        self._sync_attached_object(panda, lula, ik)

        if self.phase == "idle" or self.phase == "done":
            return

        if self.phase == "move_above":
            target = self._object_target(self.attached_object_path, self.pick_above)
            if self._run_traj(panda, lula, ik, target, self.step_move):
                self.phase = "descend"

        elif self.phase == "descend":
            target = self._object_target(self.attached_object_path, self.pick)
            if self._run_traj(panda, lula, ik, target, self.step_descend):
                self.phase = "close"
                self.counter = 0

        elif self.phase == "close":
            set_gripper_width(panda, self.close_width, 0.02, 0.02)
            self.counter += 1

            if (self.counter == 20
                and not self.attached
                and self.attached_object_path is not None):

                ee_pos, ee_quat = self.get_ee_pose(panda, lula, ik)
                # print(
                #     f"[EE]| "
                #     f"pos={ee_pos} | "
                #     f"quat(wxyz)={ee_quat}"
                # )
                quat_xyzw = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
                T_ee = np.eye(4)
                T_ee[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
                T_ee[:3, 3] = ee_pos

                T_obj = self.get_obj_pose(self.attached_object_path)
                self.T_ee_to_obj = np.linalg.inv(T_ee) @ np.array(T_obj)
                self.attached = True

            if self.counter >= self.close_steps:
                self.phase = "hold"
                self.traj = []
                self.i = 0
                self.counter = 0

        elif self.phase == "hold":
            set_gripper_width(panda, self.close_width, 0.02, 0.02)
            self.counter += 1
            if self.counter >= self.hold_steps:
                self.phase = "lift"

        elif self.phase == "lift":
            target, _ = self.get_ee_pose(panda, lula, ik)
            target += self.lift_offset
            if self._run_traj(panda, lula, ik, target, self.step_move):
                self.phase = "move_place"

        elif self.phase == "move_place":
            target = self._place_target(self.place_above)
            if self._run_traj(panda, lula, ik, target, self.step_move):
                self.phase = "descend_place"

        elif self.phase == "descend_place":
            target = self._place_target(self.place)
            if self._run_traj(panda, lula, ik, target, self.step_descend):
                self.phase = "release"

        elif self.phase == "release":
            set_gripper_width(panda, self.open_width, 0.0, 0.05)
            self.traj = []
            self.i = 0
            self.counter = 0
            self.attached = False
            self.T_ee_to_obj = None
            self.attached_object_path = None
            self.target_object_path = None
            if self.retreat_after_place:
                self.phase = "post_place_lift"
            else:
                self.phase = "done"

        elif self.phase == "post_place_lift":
            ee_pos, _ = self.get_ee_pose(panda, lula, ik)
            target = ee_pos + self.lift_offset
            if self._run_traj(panda, lula, ik, target, self.step_move):
                self.phase = "done"

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

        self.pick_above_offset  = np.array([0.06, -0.12,  0.10])
        self.pick_offset        = np.array([0.06, -0.12, -0.07])
        self.lift_offset        = np.array([ 0.0,  -0.05,  0.25])
        self.place_above_offset = np.array([ 0.0,  -0.05,  0.20])
        self.place_offsets = [
            np.array([-0.05, 0.03, -0.02]),
            np.array([-0.05, -0.15, -0.02]),
        ]

        self.current_idx = 0
        self.started = False
    
    def _start_pickplace_for_current_cutlery(self):
        self.pickplace.reset()
        self.pickplace.grasp_mode = "object_based"
        self.pickplace.start(
            pick_above  = self.pick_above_offset,
            pick        = self.pick_offset,
            lift_offset = self.lift_offset,
            place_above = self.place_above_offset,
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
    
class TeleopEEController:
    def __init__(self):
        self.delta_pos = np.zeros(3)
        self.delta_yaw = 0.0
        self.grip_cmd = None
        self.next_episode = False
        self.enabled = False

        self.step_xy = 0.01
        self.step_z  = 0.01
        self.step_yaw = 5.0  # deg

        self.listener = keyboard.Listener(
            on_press=self._on_press
        )
        self.listener.start()

    def _on_press(self, key):
        try:
            k = key.char.lower()
        except:
            return

        if k == 't':
            self.enabled = not self.enabled
            print(f"[Teleop] enabled = {self.enabled}")
        
        if not self.enabled:
            return

        if k == 'w': self.delta_pos[1] += self.step_xy
        if k == 's': self.delta_pos[1] -= self.step_xy
        if k == 'a': self.delta_pos[0] -= self.step_xy
        if k == 'd': self.delta_pos[0] += self.step_xy
        if k == 'q': self.delta_pos[2] += self.step_z
        if k == 'e': self.delta_pos[2] -= self.step_z

        if k == 'j': self.delta_yaw += self.step_yaw
        if k == 'l': self.delta_yaw -= self.step_yaw

        if k == 'g': self.grip_cmd = "close"
        if k == 'r': self.grip_cmd = "open"
        if k == 'n': self.next_episode = True

    def consume(self):
        dpos = self.delta_pos.copy()
        dyaw = self.delta_yaw
        grip = self.grip_cmd
        next_ep = self.next_episode

        self.delta_pos[:] = 0
        self.delta_yaw = 0
        self.grip_cmd = None
        self.next_episode = False

        return dpos, dyaw, grip, next_ep
    

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


        self.pick_above_offset  = np.array([-0.0, -0.0,  0.10])
        self.pick_offset        = np.array([-0.05, -0.05, -0.1])
        self.lift_offset        = np.array([ 0.0,   0.0,    0.25])
        self.place_above_offset = np.array([-0.05, -0.05,  0.15])
        self.place_offset       = np.array([-0.05, -0.05,  0.06])
    


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

        self.pick_above_offset  = np.array([-0.06, -0.075,  0.10])
        self.pick_offset        = np.array([-0.06, -0.075, -0.06])
        self.lift_offset        = np.array([ 0.0,   0.0,    0.25])
        self.place_above_offset = np.array([-0.20, -0.10,  0.20])
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
            pick_above  = self.pick_above_offset,
            pick        = self.pick_offset,
            lift_offset = self.lift_offset,
            place_above = self.place_above_offset,
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