#!/bin/bash
set -e

# To have ros2.bridge extension worked with IsaacSim
# Source the ROS environment setup file(s)
source /workspace/humble_ws/install/setup.bash
source /workspace/build_ws/install/local_setup.bash

exec "$@"
