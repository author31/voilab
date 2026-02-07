from pynput import keyboard
from pynput.keyboard import Key
from scipy.spatial.transform import Rotation as R
import numpy as np

class TeleopController:
    def __init__(self, get_end_effector_pose_fn, ik_solution_fn, init_ee_pos, init_ee_quat_wxyz):
        # ----------------- Basic flags -----------------
        self.start = False
        self.restart = False
        self.done = False

        # ----------------- Callbacks -----------------
        self.get_ee_pose = get_end_effector_pose_fn
        self.ik_solution = ik_solution_fn

        # ----------------- Keyboard listener -----------------
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start() 

        # ----------------- Teleop parameters -----------------
        self.open_width = 0.08
        self.close_width = 0.03
        self.pos_step_size = 0.002   # meters per key press
        self.rot_step_size = 0.2     # degrees per key press

        # Thresholds to decide when to apply IK
        self.POS_THRESHOLD = 0.01   # meters
        self.ROT_THRESHOLD = 1.0     # degrees

        # ----------------- Current EE state -----------------
        self.curr_ee_pos = np.array(init_ee_pos, dtype=float)
        self.curr_ee_quat = np.array(init_ee_quat_wxyz, dtype=float)

        # ----------------- Accumulated deltas -----------------
        self.accum_dpos = np.zeros(3)
        self.accum_drot = 0.0  # rotation around Z in degrees

        # ----------------- Gripper state -----------------
        self.gripper_width = self.open_width

        # ----------------- Key pressed tracking -----------------
        self.pressed = dict.fromkeys([
            "forward", "backward", "left", "right", 
            "up", "down", "open_gripper", "close_gripper", 
            "rotate_cw", "rotate_ccw"
        ], False)

    # ----------------- Keyboard callbacks -----------------
    def _on_press(self, key):
        k = key.char.lower() if hasattr(key, 'char') and key.char else None

        if k == 'e': self.start = True
        elif k == 'r': self.restart = True
        elif k == 'q': self.done = True

        # translation
        elif k == 'w': self.pressed["forward"] = True
        elif k == 's': self.pressed["backward"] = True
        elif k == 'a': self.pressed["left"] = True
        elif k == 'd': self.pressed["right"] = True

        # rotation
        elif k == 'j': self.pressed["rotate_ccw"] = True
        elif k == 'k': self.pressed["rotate_cw"] = True

        # arrow keys
        elif key == Key.up: self.pressed["up"] = True
        elif key == Key.down: self.pressed["down"] = True
        elif key == Key.left: self.pressed["close_gripper"] = True
        elif key == Key.right: self.pressed["open_gripper"] = True

    def _on_release(self, key):
        k = key.char.lower() if hasattr(key, 'char') and key.char else None

        if k == 'w': self.pressed["forward"] = False
        elif k == 's': self.pressed["backward"] = False
        elif k == 'a': self.pressed["left"] = False
        elif k == 'd': self.pressed["right"] = False
        elif k == 'j': self.pressed["rotate_ccw"] = False
        elif k == 'k': self.pressed["rotate_cw"] = False

        elif key == Key.up: self.pressed["up"] = False
        elif key == Key.down: self.pressed["down"] = False
        elif key == Key.left: self.pressed["close_gripper"] = False
        elif key == Key.right: self.pressed["open_gripper"] = False

    # ----------------- Teleop status -----------------
    def is_done(self): return self.done
    def started(self): return self.start
    def requested_restart(self): return self.restart

    # ----------------- Instructions -----------------
    def print_instructions(self):
        print("Teleoperation Controls:")
        print("  W/S: Move Forward/Backward")
        print("  A/D: Move Left/Right")
        print("  UP/Down Arrow: Move Up/Down")
        print("  Left/Right Arrow: Close/Open Gripper")
        print("  J/K: Rotate CCW/CW")
        print("  E: Start Recording")
        print("  R: Restart Same Episode")
        print("  Q: End Recording")
        print("Turn on gopro camera view on isaacsim by clicking Window > Viewport > Viewport 2")

    # ----------------- Main teleop step -----------------
    def step(self, panda, lula, ik):
        if not self.started(): return

        articulation_controller = panda.get_articulation_controller()

        dx = dy = dz = drot = dw = 0.0

        # accumulate delta based on keys
        if self.pressed["forward"]: dx += 1
        if self.pressed["backward"]: dx -= 1
        if self.pressed["left"]: dy += 1
        if self.pressed["right"]: dy -= 1
        if self.pressed["up"]: dz += 1
        if self.pressed["down"]: dz -= 1
        if self.pressed["rotate_ccw"]: drot += 1
        if self.pressed["rotate_cw"]: drot -= 1
        if self.pressed["open_gripper"]: dw += 1
        if self.pressed["close_gripper"]: dw -= 1

        # ----- update accumulated position -----
        norm = np.linalg.norm([dx, dy, dz])
        if norm > 0:
            self.accum_dpos += np.array([dx, dy, dz]) / norm * self.pos_step_size

        # ----- update accumulated rotation -----
        self.accum_drot += drot * self.rot_step_size

        # ----- check if accumulated deltas exceed threshold -----
        if np.linalg.norm(self.accum_dpos) > self.POS_THRESHOLD or abs(self.accum_drot) > self.ROT_THRESHOLD:
            # apply IK
            dq = R.from_euler('z', self.accum_drot, degrees=True)
            new_quat = (dq * R.from_quat(self.curr_ee_quat, scalar_first=True)).as_quat(scalar_first=True)
            new_pos = self.curr_ee_pos + self.accum_dpos

            action = self.ik_solution(panda, ik, new_pos, new_quat)
            if action is not None:
                articulation_controller.apply_action(action)

            # update current pose and reset accumulated deltas
            self.curr_ee_pos = new_pos
            self.curr_ee_quat = new_quat
            self.accum_dpos[:] = 0
            self.accum_drot = 0.0

        # ----- gripper control -----
        if dw != 0.0:
            if dw > 0:
                action = panda.gripper.forward(action="open")
            else:
                action = panda.gripper.forward(action="close")
            articulation_controller.apply_action(action)
