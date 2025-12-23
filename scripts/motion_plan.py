import numpy as np
from scipy.spatial.transform import Rotation as R
from umi_replay import set_gripper_width
from utils import set_prim_world_pose


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
        gripper_open_thresh = 0.0875
        ):

        self.get_ee_pose = get_end_effector_pose_fn
        self.get_obj_pose = get_object_world_pose_fn
        self.apply_ik = apply_ik_solution_fn
        self.plan = plan_line_cartesian_fn

        self.grasp_quat = np.asarray(grasp_quat_wxyz)
        self.open_width = open_width
        self.close_width = close_width
        self.close_steps = close_steps
        self.hold_steps = hold_steps
        self.step_move = step_move
        self.step_descend = step_descend

        # Attachment threshold
        self.attach_dist_thresh = attach_dist_thresh
        self.release_dist_thresh = release_dist_thresh
        self.gripper_close_thresh = gripper_close_thresh
        self.gripper_open_thresh  = gripper_open_thresh

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
            attached_object_path=None, target_object_path=None):
        self.pick_above = pick_above
        self.pick = pick
        self.place_above = place_above
        self.lift_offset = lift_offset
        self.place = place

        self.attached_object_path = attached_object_path
        self.target_object_path = target_object_path

        self.attached = False
        self.T_ee_to_obj = None

        self.phase = "move_above"
        self.traj = []
        self.i = 0
        self.counter = 0

    # -------------------------
    def _run_traj(self, panda, lula, ik, target, step):
        if not self.traj:
            p, q = self.get_ee_pose(panda, lula, ik)
            self.traj = self.plan(p, q, target, self.grasp_quat, step_m=step)
            self.i = 0

        if self.i >= len(self.traj):
            self.traj = []
            self.i = 0
            return True

        wp = self.traj[self.i]
        self.apply_ik(panda, ik, wp[:3], wp[3:])
        if self.attached:
            set_gripper_width(panda, self.close_width, 0.02, 0.02)
        else:
            set_gripper_width(panda, self.open_width, 0.0, 0.05)
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

    # def _should_attach(self, panda, lula, ik):
    #     if self.attached or self.attached_object_path is None:
    #         return False

    #     ee_pos, _ = self.get_ee_pose(panda, lula, ik)
    #     obj_pos = self.get_obj_pose(self.attached_object_path)[:3, 3]
    #     dist = np.linalg.norm(ee_pos - obj_pos)

    #     joint_pos = panda.get_joint_positions()
    #     gripper_width = joint_pos[-2] + joint_pos[-1]

    #     if dist < self.attach_dist_thresh and gripper_width < self.gripper_close_thresh:
    #         return True

    #     return False

    # def _should_release(self, panda, lula, ik):
    #     if not self.attached or self.target_object_path is None:
    #         return False

    #     ee_pos, _ = self.get_ee_pose(panda, lula, ik)
    #     target_pos = self.get_obj_pose(self.target_object_path)[:3, 3]
    #     dist = np.linalg.norm(ee_pos - target_pos)

    #     if dist > self.release_dist_thresh:
    #         self.gripper_above_open_cnt = 0
    #         return False

    #     joint_pos = panda.get_joint_positions()
    #     gripper_width = joint_pos[-2] + joint_pos[-1]

    #     if gripper_width >= self.gripper_open_thresh:
    #         self.gripper_above_open_cnt += 1
    #     else:
    #         self.gripper_above_open_cnt = 0

    #     return self.gripper_above_open_cnt >= 2
    
    # -------------------------
    def _object_target(self, obj_path, offset):
        T = self.get_obj_pose(obj_path)
        pos = T[:3, 3]
        return pos + offset

    # -------------------------
    def step(self, panda, lula, ik):

        # curr_ee_pos, curr_ee_quat = self.get_ee_pose(panda, lula, ik)
        # T_blue = self.get_obj_pose(self.attached_object_path)
        # T_pink = self.get_obj_pose(self.target_object_path)
        # blue_pos = T_blue[:3, 3]
        # pink_pos = T_pink[:3, 3]
        # joint_pos = panda.get_joint_positions() 
        # gripper_width = joint_pos[-2] + joint_pos[-1] 

        # print("Distance from eef to blue cup (need to grasp)= ", 
        #       np.linalg.norm(curr_ee_pos - blue_pos))
        # print("Distance from eef to pink cup (need to be place on)= ", 
        #       np.linalg.norm(curr_ee_pos - pink_pos))
        # print("Gripper width= ", gripper_width)

        # check if need attachment
        # if self._should_attach(panda, lula, ik):
        #     ee_pos, ee_quat = self.get_ee_pose(panda, lula, ik)
        #     quat_xyzw = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
        #     T_ee = np.eye(4)
        #     T_ee[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        #     T_ee[:3, 3] = ee_pos
        #     T_obj = self.get_obj_pose(self.attached_object_path)
        #     self.T_ee_to_obj = np.linalg.inv(T_ee) @ np.array(T_obj)
        #     self.attached = True
        
        # # check if need release
        # if self._should_release(panda, lula, ik):
        #     self.attached = False
        #     self.T_ee_to_obj = None
        #     self.attached_object_path = None
        #     self.target_object_path = None

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
            target = self._object_target(self.target_object_path, self.place_above)
            if self._run_traj(panda, lula, ik, target, self.step_move):
                self.phase = "descend_place"

        elif self.phase == "descend_place":
            target = self._object_target(self.target_object_path, self.place)
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
            self.phase = "done"

    def is_done(self):
        return self.phase == "done"


class KitchenMotionPlanner:
    def __init__(self, cfg, *, get_object_world_pose_fn, pickplace):
        self.cfg = cfg
        self.get_object_pose = get_object_world_pose_fn
        self.pickplace = pickplace
        self.started = False

        env = cfg["environment_vars"]
        self.blue = env["TARGET_OBJECT_PATH"]
        self.pink = env["SUPPORT_OBJECT_PATH"]

        self.pick_above_offset  = np.array([-0.05, -0.075,  0.10])
        self.pick_offset        = np.array([-0.05, -0.075, -0.12])
        self.lift_offset        = np.array([ 0.0,   0.0,    0.25])
        self.place_above_offset = np.array([-0.05, -0.07,  0.15])
        self.place_offset       = np.array([-0.05, -0.07,  0.03])

    def step(self, panda, lula, ik):
        if not self.started:
            self.pickplace.reset()
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


class DiningRoomMotionPlanner:
    def __init__(self, cfg, *, get_object_world_pose_fn, pickplace):
        self.cfg = cfg
        self.get_object_pose = get_object_world_pose_fn
        self.pickplace = pickplace
        self.started = False
    
    def step(self, panda, lula, ik):
        return 0
    
    def is_done(self):
        return self.pickplace.is_done()


class LivingRoomMotionPlanner:
    def __init__(self, cfg, *, get_object_world_pose_fn, pickplace):
        self.cfg = cfg
        self.get_object_pose = get_object_world_pose_fn
        self.pickplace = pickplace
        self.started = False
    
    def step(self, panda, lula, ik):
        return 0
    
    def is_done(self):
        return self.pickplace.is_done()