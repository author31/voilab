#!/usr/bin/env python3
"""
ROS2 node to publish a "go to home" position command.
Publishes a target pose to /joint_command topic using cuRobo IK solver.

Target pose:
- Position: [0.6, 0, 0.0]
- Quaternion: [0, 1, 0, 0] (w, x, y, z format)
"""

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import time
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation

# cuRobo imports
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState as CuroboJointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.util_file import load_yaml
from curobo.geom.sdf.world import CollisionCheckerType

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TENSOR_ARGS = TensorDeviceType()
JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
]


def get_neutral_joint_angles():
    """Get a neutral joint configuration for the Franka Panda robot."""
    return [0.0, -0.0002, 0.0, -0.0698, 0.0, 0.0005, 0.0]


def init_motion_gen():
    """Initialize a cuRobo MotionGen for the Franka Panda robot."""
    logger.info('Initializing MotionGen')

    robot_path = "franka.yml"
    trajopt_tsteps = 32

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_path,
        None,
        TENSOR_ARGS,
        trajopt_tsteps=trajopt_tsteps,
        collision_checker_type=CollisionCheckerType.BLOX,
        use_cuda_graph=True,
        num_trajopt_seeds=2,
        num_graph_seeds=2,
        evaluate_interpolated_trajectory=True,
        filter_robot_command=True
    )

    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    return motion_gen


def solve_ik_for_target_pose(target_pos, target_quat, current_joint_states):
    """
    Solve inverse kinematics for target pose using cuRobo's MotionGen.

    Args:
        target_pos: Target position [x, y, z]
        target_quat: Target quaternion [w, x, y, z]
        current_joint_states: Current joint angles to use as starting state

    Returns:
        List of joint angles or None if failed
    """
    try:
        # Initialize motion gen
        motion_gen = init_motion_gen()

        # Create goal pose
        goal = Pose(
            position=TENSOR_ARGS.to_device(
                torch.tensor(target_pos, dtype=torch.float32).unsqueeze(0)
            ),
            quaternion=TENSOR_ARGS.to_device(
                torch.tensor(target_quat, dtype=torch.float32).unsqueeze(0)
            ),
        )

        # Use current joint states as starting state
        start_joint_state = CuroboJointState.from_position(
            position=TENSOR_ARGS.to_device(current_joint_states).unsqueeze(0),
            joint_names=JOINT_NAMES
        )

        # Plan motion
        result = motion_gen.plan_single(
            start_joint_state,
            goal,
            MotionGenPlanConfig(max_attempts=3, enable_finetune_trajopt=True)
        )

        success = bool(result.success.view(-1)[0].item())
        if not success:
            logger.error(f"MotionGen plan failed: {result.error_msg.view(-1)[0].item()}")
            return None

        # Get the final joint position from the trajectory
        trajectories = result.get_interpolated_plan().position.tolist()
        return trajectories[-1]  # Return the final pose

    except Exception as e:
        logger.warning(f"IK solver failed: {e}")
        return None


class GoToHomePublisher(Node):
    """ROS2 node to publish go-to-home joint commands."""

    def __init__(self):
        super().__init__('go_to_home_publisher')

        # Create publisher for joint commands
        self.joint_pub = self.create_publisher(
            JointState,
            '/joint_command',
            10
        )

        # Create subscriber for current joint states
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )

        # Timer for publishing (10 Hz)
        self.timer = self.create_timer(0.1, self.publish_joint_command)
        self.get_logger().info('GoToHomePublisher node initialized')

        # State variables
        self.current_joint_states = None
        self.target_joint_angles = None
        self.ik_solved = False
        self.finger_joint_positions = [0.0, 0.0]  # Default finger positions

    def joint_states_callback(self, msg):
        """Callback for receiving current joint states."""
        try:
            # Extract positions for the arm joints (first 7 joints)
            self.current_joint_states = []
            for joint_name in JOINT_NAMES:
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    self.current_joint_states.append(msg.position[idx])
                else:
                    self.get_logger().warn(f"Joint {joint_name} not found in /joint_states")
                    # Use default value if joint not found
                    self.current_joint_states.append(0.0)

            # Extract finger joint positions for publishing
            if "panda_finger_joint1" in msg.name:
                idx = msg.name.index("panda_finger_joint1")
                self.finger_joint_positions[0] = msg.position[idx]
            if "panda_finger_joint2" in msg.name:
                idx = msg.name.index("panda_finger_joint2")
                self.finger_joint_positions[1] = msg.position[idx]

            self.get_logger().debug(f'Updated current joint states: {self.current_joint_states}')

            # Solve IK if we haven't solved it yet and we have current states
            if not self.ik_solved and self.current_joint_states:
                self.solve_target_pose()

        except Exception as e:
            self.get_logger().error(f"Error processing joint states: {e}")

    def solve_target_pose(self):
        """Solve IK for the target home pose using current joint states."""
        self.get_logger().info('Solving IK for target pose with current joint states...')

        target_pos = np.array([0.61, 0.0, 0.56])
        target_quat = np.array([0.0, 0.0, 1.0, 0.0])

        if self.current_joint_states:
            self.target_joint_angles = solve_ik_for_target_pose(
                target_pos, target_quat, self.current_joint_states)
        else:
            self.get_logger().warn('No current joint states available, using neutral configuration')
            # Fallback to neutral configuration
            neutral_joints = [0.0, -0.0002, 0.0, -0.0698, 0.0, 0.0005, 0.0]
            self.target_joint_angles = solve_ik_for_target_pose(
                target_pos, target_quat, neutral_joints)

        if self.target_joint_angles is not None:
            self.ik_solved = True
            self.get_logger().info(f'IK solution found: {self.target_joint_angles}')
        else:
            self.get_logger().error('Failed to solve IK for target pose')

    def publish_joint_command(self):
        """Publish joint command message."""
        if self.target_joint_angles is None:
            # Don't warn continuously, just skip publishing
            return

        # Create joint state message with all joints (arm + fingers)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "panda_link0"
        msg.name = JOINT_NAMES

        # Combine arm joint angles with finger positions
        msg.position = self.target_joint_angles
        msg.velocity = []
        msg.effort = []

        # Publish the message
        self.joint_pub.publish(msg)
        self.get_logger().debug(f'Published joint command: {msg.position}')


def main(args=None):
    """Main function to run the ROS2 node."""
    rclpy.init(args=args)

    try:
        # Create and spin the node
        node = GoToHomePublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
