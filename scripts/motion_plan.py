import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from umi_replay import set_gripper_width


def wxyz_to_xyzw(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

def xyzw_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def plan_line_cartesian(
    p_start: np.ndarray,
    q_start_wxyz: np.ndarray,
    p_goal: np.ndarray,
    q_goal_wxyz: np.ndarray,
    step_m: float = 0.005,
    ):
    p_start = np.asarray(p_start, dtype=float)
    p_goal = np.asarray(p_goal, dtype=float)

    dist = np.linalg.norm(p_goal - p_start)
    n_steps = max(2, int(np.ceil(dist / step_m)))

    positions = np.linspace(p_start, p_goal, n_steps)

    q0_xyzw = wxyz_to_xyzw(np.asarray(q_start_wxyz, dtype=float))
    q1_xyzw = wxyz_to_xyzw(np.asarray(q_goal_wxyz, dtype=float))

    key_rots = R.from_quat([q0_xyzw, q1_xyzw])
    slerp = Slerp([0.0, 1.0], key_rots)
    interp_rots = slerp(np.linspace(0.0, 1.0, n_steps))
    quats_xyzw = interp_rots.as_quat()
    quats_wxyz = np.array([xyzw_to_wxyz(q) for q in quats_xyzw])

    return [np.concatenate([p, q_wxyz]) for p, q_wxyz in zip(positions, quats_wxyz)]


class KitchenMotionPlanner:
    def __init__(
        self,
        cfg,
        *,
        get_end_effector_pose_fn,
        get_object_world_pose_fn,
        apply_ik_solution_fn,
        ):
        self.cfg = cfg
        self.get_end_effector_pose = get_end_effector_pose_fn
        self.get_object_world_pose = get_object_world_pose_fn
        self.apply_ik_solution = apply_ik_solution_fn

        self.phase = "init"
        self.traj = []
        self.traj_idx = 0
        self.close_gripper_steps = 0

    def step(self, panda, lula_solver, art_kine_solver):
        # =========================
        # Phase 0: initialize plan
        # =========================
        if self.phase == "init":
            blue_cup_path = self.cfg["environment_vars"]["TARGET_OBJECT_PATH"]
            blue_cup_T = self.get_object_world_pose(blue_cup_path)
            cup_pos = blue_cup_T[:3, 3]

            ee_pos, ee_quat_wxyz = self.get_end_effector_pose(
                panda, lula_solver, art_kine_solver
            )

            # Heuristic approach offset
            target_pos = cup_pos.copy()
            target_pos[0] -= 0.15
            target_pos[1] -= 0.06
            target_pos[2] -= 0.15

            target_quat = ee_quat_wxyz.copy()

            self.traj = plan_line_cartesian(
                ee_pos,
                ee_quat_wxyz,
                target_pos,
                target_quat,
            )
            self.traj_idx = 0
            self.phase = "approach"
            return

        # =========================
        # Phase 1: approach
        # =========================
        if self.phase == "approach":
            if self.traj_idx >= len(self.traj):
                self.phase = "pregrasp"
                return

            waypoint = self.traj[self.traj_idx]
            pos = waypoint[:3]
            quat = waypoint[3:]

            self.apply_ik_solution(panda, art_kine_solver, pos, quat)
            set_gripper_width(panda, width=0.1, threshold=0.0, step=0.05)

            self.traj_idx += 1
            return

        # =========================
        # Phase 2: pregrasp (small forward move)
        # =========================
        if self.phase == "pregrasp":
            ee_pos, ee_quat_wxyz = self.get_end_effector_pose(
                panda, lula_solver, art_kine_solver
            )

            pregrasp_pos = ee_pos.copy()
            pregrasp_pos[0] += 0.20
            pregrasp_quat = ee_quat_wxyz.copy()

            self.traj = plan_line_cartesian(
                ee_pos,
                ee_quat_wxyz,
                pregrasp_pos,
                pregrasp_quat,
            )
            self.traj_idx = 0
            self.phase = "pregrasp_exec"
            return

        if self.phase == "pregrasp_exec":
            if self.traj_idx >= len(self.traj):
                self.phase = "close_gripper"
                self.close_gripper_steps = 0
                return

            waypoint = self.traj[self.traj_idx]
            pos = waypoint[:3]
            quat = waypoint[3:]

            self.apply_ik_solution(panda, art_kine_solver, pos, quat)
            set_gripper_width(panda, width=0.1, threshold=0.0, step=0.05)

            self.traj_idx += 1
            return

        # =========================
        # Phase 3: close gripper
        # =========================
        if self.phase == "close_gripper":
            set_gripper_width(panda, width=0.0, threshold=0.04, step=0.03)
            self.close_gripper_steps += 1

            if self.close_gripper_steps >= 50:
                self.phase = "done"
            return

        # =========================
        # Phase 4: done
        # =========================
        if self.phase == "done":
            return

    def is_done(self) -> bool:
        return self.phase == "done"


class DiningRoomMotionPlanner:
    def __init__(self, cfg):
        self.cfg = cfg

        # Internal state
        self.phase = "init"
        self.traj = []
        self.traj_idx = 0
        self.close_gripper_steps = 0
    
    def step(self, panda, lula_solver, art_kine_solver):
        return 0
    
    def is_done(self) -> bool:
        return self.phase == "done"


class LivingRoomMotionPlanner:
    def __init__(self, cfg):
        self.cfg = cfg

        # Internal state
        self.phase = "init"
        self.traj = []
        self.traj_idx = 0
        self.close_gripper_steps = 0
    
    def step(self, panda, lula_solver, art_kine_solver):
        return 0
    
    def is_done(self) -> bool:
        return self.phase == "done"