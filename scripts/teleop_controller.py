from pynput import keyboard
from pynput.keyboard import Key
from umi_replay import set_gripper_width
from scipy.spatial.transform import Rotation as R
import numpy as np

class TeleopController:
    def __init__(self, get_end_effector_pose_fn, apply_ik_solution_fn, init_ee_pos, init_ee_quat_wxyz):
        self.start = False
        self.restart = False
        self.done = False
        self.get_ee_pose = get_end_effector_pose_fn
        self.apply_ik = apply_ik_solution_fn
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start() 
        
        self.open_width=0.08
        self.close_width=0.03
        self.pos_step_size=0.002
        self.rot_step_size=1.0 # degrees
        self.gripper_step_size=0.001
        self.init_ee_pos = init_ee_pos
        self.init_ee_quat_wxyz = init_ee_quat_wxyz

        self.pressed = dict.fromkeys(["forward", "backward", "left", "right", 
                                      "up", "down", "open_gripper", "close_gripper", 
                                      "rotate_cw", "rotate_ccw"], False)
        
        self.cumulated_dpos = np.array([0.0, 0.0, 0.0])
        self.cumulated_drot = 0.0
        self.gripper_width = self.open_width

    def _on_press(self, key):

        k = key.char.lower() if hasattr(key, 'char') and key.char else None

        if k == 'e':
            self.start = True
        elif k == 'r':
            self.restart = True
        elif k == 'q':
            self.done = True
        elif k == 'w':
            self.pressed["forward"] = True
        elif k == 's':
            self.pressed["backward"] = True
        elif k == 'a':
            self.pressed["left"] = True
        elif k == 'd':
            self.pressed["right"] = True
        elif key == Key.space:
            self.pressed["up"] = True
        elif key == Key.caps_lock:
            self.pressed["down"] = True
        elif key == Key.up:
            self.pressed["open_gripper"] = True
        elif key == Key.down:
            self.pressed["close_gripper"] = True
        elif key == Key.left:
            self.pressed["rotate_ccw"] = True
        elif key == Key.right:
            self.pressed["rotate_cw"] = True


    def _on_release(self, key):

        k = key.char.lower() if hasattr(key, 'char') and key.char else None
 
        if k == 'w':
            self.pressed["forward"] = False
        elif k == 's':
            self.pressed["backward"] = False
        elif k == 'a':
            self.pressed["left"] = False
        elif k == 'd':
            self.pressed["right"] = False
        elif key == Key.space:
            self.pressed["up"] = False
        elif key == Key.caps_lock:
            self.pressed["down"] = False
        elif key == Key.up:
            self.pressed["open_gripper"] = False
        elif key == Key.down:
            self.pressed["close_gripper"] = False
        elif key == Key.left:
            self.pressed["rotate_ccw"] = False
        elif key == Key.right:
            self.pressed["rotate_cw"] = False


    def is_done(self):
        return self.done
      
    def started(self):
        return self.start
      
    def requested_restart(self):
        return self.restart
      
    def gripper_control(self, panda, width=0.0):
        idx1 = panda.get_dof_index("panda_finger_joint1")
        idx2 = panda.get_dof_index("panda_finger_joint2")
        if idx1 is not None and idx2 is not None:
            panda.set_joint_positions(
                positions=np.array([width, width]),
                joint_indices=np.array([idx1, idx2])
            )
    
    def print_instructions(self):
        print("Teleoperation Controls:")
        print("  W/S: Move Forward/Backward")
        print("  A/D: Move Left/Right")
        print("  Space/Caps Lock: Move Up/Down")
        print("  Up/Down Arrow: Open/Close Gripper")
        print("  Left/Right Arrow: Rotate CCW/CW")
        print("  E: Start Recording")
        print("  R: Restart Same Episode")
        print("  Q: End Recording")
        print("Turn on gopro camera view on isaacsim by clicking Window > Viewport > Viewport 2")
      
    def step(self, panda, lula, ik):

        if not self.started():
            return
        
        dx, dy, dz, drot, dw = 0.0, 0.0, 0.0, 0.0, 0.0

        if self.pressed["forward"]:
            dx += 1
        if self.pressed["backward"]:
            dx -= 1
        if self.pressed["left"]:
            dy += 1
        if self.pressed["right"]:
            dy -= 1
        if self.pressed["up"]:
            dz += 1
        if self.pressed["down"]:
            dz -= 1
        if self.pressed["rotate_ccw"]:
            drot += 1
        if self.pressed["rotate_cw"]:
            drot -= 1
        if self.pressed["open_gripper"]:
            dw += 1
        if self.pressed["close_gripper"]:
            dw -= 1

        # Normalize position step to self.pos_step_size
        norm = (dx**2 + dy**2 + dz**2)**0.5
        dpos = [(v / norm * self.pos_step_size if norm > 0 else 0.0) 
                      for v in (dx, dy, dz)]
        drot = drot * self.rot_step_size
        
        self.cumulated_dpos += np.array(dpos)
        self.cumulated_drot += drot
        
        new_p = self.init_ee_pos + self.cumulated_dpos
        # Apply rotation around Z axis
        if self.cumulated_drot != 0.0:
            dq  = R.from_euler('z', self.cumulated_drot, degrees=True)
            new_q = (dq * R.from_quat(self.init_ee_quat_wxyz, scalar_first=True)).as_quat(scalar_first=True)
        else:
            new_q = self.init_ee_quat_wxyz
        self.apply_ik(panda, ik, new_p, new_q)
        
        self.gripper_width += dw * self.gripper_step_size
        self.gripper_width = np.clip(self.gripper_width, self.close_width, self.open_width)
            
        self.gripper_control(panda, width=self.gripper_width)