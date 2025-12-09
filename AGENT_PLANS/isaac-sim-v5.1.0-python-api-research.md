# Isaac Sim v5.1.0 Python API Research Report

**Date:** 2025-12-09  
**Version:** Isaac Sim 5.1.0  
**Repository:** author31/voilab  
**Purpose:** Comprehensive reference for Isaac Sim Python APIs used in the voilab project

---

## Table of Contents

1. [Overview](#overview)
2. [Core Python APIs](#core-python-apis)
3. [Simulation Application](#simulation-application)
4. [World and Scene Management](#world-and-scene-management)
5. [Robot Control and Articulation](#robot-control-and-articulation)
6. [Prims and Transforms](#prims-and-transforms)
7. [Stage Utilities](#stage-utilities)
8. [ROS2 Bridge Integration](#ros2-bridge-integration)
9. [OmniGraph Action Graphs](#omnigraph-action-graphs)
10. [Camera and Sensors](#camera-and-sensors)
11. [Physics and Collision](#physics-and-collision)
12. [Extensions System](#extensions-system)
13. [Best Practices and Patterns](#best-practices-and-patterns)
14. [Common Pitfalls](#common-pitfalls)
15. [Migration from Previous Versions](#migration-from-previous-versions)

---

## Overview

### What is Isaac Sim v5.1.0?

Isaac Sim 5.1.0 is NVIDIA's robotics simulation platform built on NVIDIA Omniverse. It provides:
- High-fidelity physics simulation (PhysX)
- Photorealistic rendering (RTX)
- ROS/ROS2 integration
- Python scripting API
- USD (Universal Scene Description) file format support

### Installation in Voilab

The project uses Isaac Sim via pip installation:
```bash
python3.11 -m pip install "isaacsim[all,extscache,isaacsim-gui,isaacsim-kernel,isaacsim-app,isaacsim-ros2]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
```

**Key Package Groups:**
- `isaacsim-gui` - GUI components
- `isaacsim-kernel` - Core kernel
- `isaacsim-app` - Application framework
- `isaacsim-ros2` - ROS2 bridge

### Import Structure

Isaac Sim 5.1.0 uses the `isaacsim` package namespace (changed from `omni.isaac` in previous versions):

```python
from isaacsim import SimulationApp        # Main application
from isaacsim.core.api import World       # World management
from isaacsim.core.prims import Articulation, XFormPrim  # Prim types
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.extensions import enable_extension
```

---

## Core Python APIs

### 1. SimulationApp - Main Application Entry Point

**Module:** `isaacsim.SimulationApp`

The `SimulationApp` is the primary interface for initializing and running Isaac Sim.

#### Initialization

```python
from isaacsim import SimulationApp

config = {
    "headless": False,           # GUI enabled
    "width": 1280,              # Window width
    "height": 720,              # Window height
    "enable_streaming": False,  # Disable streaming
    "extensions": ["isaacsim.ros2.bridge"]  # Extensions to load
}

simulation_app = SimulationApp(config)
```

**Configuration Options:**
- `headless` (bool): Run without GUI (default: False)
- `width` (int): Window width in pixels
- `height` (int): Window height in pixels
- `enable_streaming`: Enable Omniverse streaming
- `extensions` (list): Extensions to load at startup
- `active_gpu` (int): GPU device ID
- `renderer`: Rendering mode (e.g., "RayTracedLighting")
- `physics_gpu` (int): Physics GPU device

#### Core Methods

```python
# Update simulation one step
simulation_app.update()

# Check if app is still running
while simulation_app.is_running():
    simulation_app.update()

# Shutdown application
simulation_app.close()
```

**Important:** `SimulationApp` must be initialized before importing other Isaac Sim modules.

#### Pattern Used in Voilab

```python
from isaacsim import SimulationApp

config = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": ["isaacsim.ros2.bridge"]
}
simulation_app = SimulationApp(config)

# NOW import other modules
import cv2
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
```

---

## World and Scene Management

### 2. World - Scene Container and Simulation Control

**Module:** `isaacsim.core.api.World`

The `World` class manages the simulation scene, physics, and timesteps.

#### Initialization

```python
from isaacsim.core.api import World

world = World()
# Or with custom physics timestep
world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
```

#### Core Methods

```python
# Reset the world to initial state
world.reset()

# Step simulation
world.step(render=True)  # Step with rendering

# Add objects to scene
world.scene.add(robot)
world.scene.add(object_prim)

# Get current timestep
current_time = world.current_time_step_index

# Pause/Resume simulation
world.pause()
world.play()
world.stop()
```

#### Scene Management

```python
# Access scene
scene = world.scene

# Add named objects
scene.add(articulation, name="robot")
scene.add(xform_prim, name="object_1")

# Get objects by name
robot = scene.get_object("robot")

# Remove objects
scene.remove_object("object_1")

# Clear all objects
scene.clear()
```

**Pattern in Voilab:**
```python
world = World()
stage_utils.add_reference_to_stage(usd_path=FRANKA_PANDA_FP, prim_path=FRANKA_PANDA_PRIM_PATH)
panda = Articulation(prim_paths_expr=FRANKA_PANDA_PRIM_PATH, name="panda_robot")
world.scene.add(panda)
world.reset()
```

---

## Robot Control and Articulation

### 3. Articulation - Robot Joint Control

**Module:** `isaacsim.core.prims.Articulation`

Represents articulated robots (multi-joint systems).

#### Initialization

```python
from isaacsim.core.prims import Articulation

# Create articulation from prim path
panda = Articulation(
    prim_paths_expr="/World/panda",  # USD prim path
    name="panda_robot"               # Logical name
)
```

#### Joint Control Methods

```python
# Get joint information
num_dof = panda.num_dof
dof_names = panda.dof_names
joint_indices = panda.get_joints_indices()

# Get specific joint DOF index
joint_index = panda.get_dof_index("panda_joint1")

# Set joint positions
positions = np.array([[0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]])
panda.set_joint_positions(positions=positions, joint_indices=joint_indices)

# Get joint positions
current_positions = panda.get_joint_positions()

# Set joint velocities
velocities = np.array([[0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
panda.set_joint_velocities(velocities=velocities)

# Get joint velocities
current_velocities = panda.get_joint_velocities()

# Apply joint efforts (torques)
efforts = np.array([[10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
panda.set_joint_efforts(efforts=efforts)
```

#### Gains and Control Parameters

```python
# Set PD control gains
kps = np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0])  # Stiffness
kds = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])         # Damping
joint_indices = np.array([0, 1, 2, 3, 4, 5, 6])

panda.set_gains(kps=kps, kds=kds, joint_indices=joint_indices)

# Get current gains
current_kps, current_kds = panda.get_gains()
```

#### World Pose Control

```python
# Set world position and orientation
position = np.array([4.9, 3.22, 0.9])  # x, y, z
orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z quaternion

panda.set_world_poses(positions=np.array([position]))

# Get world pose
pos, rot = panda.get_world_poses()
```

**Voilab Example:**
```python
PANDA_ARM_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2"
]

joint_indices = []
kps = []
kds = []
default_positions = []

for joint_name in PANDA_ARM_JOINTS:
    dof_index = panda.get_dof_index(joint_name)
    if dof_index is not None:
        config = DRIVE_CONFIGS.get(joint_name)
        if config:
            joint_indices.append(dof_index)
            kps.append(config["stiffness"])
            kds.append(config["damping"])
            default_positions.append(config["pos"])

kps_array = np.array([kps], dtype=np.float32)
kds_array = np.array([kds], dtype=np.float32)
indices_array = np.array(joint_indices, dtype=np.int32)
positions_array = np.array([default_positions], dtype=np.float32)

panda.set_gains(kps=kps_array, kds=kds_array, joint_indices=indices_array)
panda.set_joint_positions(positions=positions_array, joint_indices=indices_array)
```

---

## Prims and Transforms

### 4. XFormPrim - Transform Hierarchy Node

**Module:** `isaacsim.core.prims.XFormPrim`

Represents a transform (XForm) in the USD hierarchy. Used for objects, groups, and coordinate frames.

#### Initialization

```python
from isaacsim.core.prims import XFormPrim

# Create XFormPrim from existing prim
obj_prim = XFormPrim(
    prim_path="/World/objects/cup_0",
    name="cup_instance",
    position=np.array([1.0, 2.0, 1.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
)
```

#### Position and Orientation

```python
# Set world pose (position + orientation)
position = np.array([4.5, 3.2, 1.0])
orientation = np.array([0.98, 0.01, -0.02, 0.15])  # quaternion (w, x, y, z)

obj_prim.set_world_pose(position=position, orientation=orientation)

# Get world pose
pos, rot = obj_prim.get_world_pose()

# Set local pose (relative to parent)
obj_prim.set_local_pose(
    translation=np.array([0.1, 0.2, 0.3]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0])
)
```

#### Visibility and Properties

```python
# Set visibility
obj_prim.set_visibility(visible=True)

# Get visibility
is_visible = obj_prim.get_visibility()

# Enable/disable collision
obj_prim.set_collision_enabled(enabled=True)
```

**Voilab Pattern:**
```python
# Spawn object with pose
obj_prim = XFormPrim(prim_path=prim_path, name=f"{object_name}_{idx}")
world.scene.add(obj_prim)

# Convert rotation vector to quaternion
quaternion = rvec_to_quaternion(rvec)
position = np.array(tvec, dtype=np.float64)

# Set pose
obj_prim.set_world_pose(position=position, orientation=quaternion)
```

---

## Stage Utilities

### 5. Stage Utils - USD Stage Management

**Module:** `isaacsim.core.utils.stage`

Utilities for working with USD stages (scenes).

#### Stage Operations

```python
import isaacsim.core.utils.stage as stage_utils

# Open an existing USD file
stage_utils.open_stage("/path/to/scene.usd")

# Create new stage
stage_utils.create_new_stage()

# Save current stage
stage_utils.save_stage()

# Close stage
stage_utils.close_stage()

# Get current stage
stage = stage_utils.get_current_stage()
```

#### Adding References

```python
# Add USD file as reference to stage
stage_utils.add_reference_to_stage(
    usd_path="/workspace/voilab/assets/CADs/cup_finalized.usd",
    prim_path="/World/objects/cup_0"
)
```

**Key Method:**
- `add_reference_to_stage(usd_path, prim_path)`: Loads a USD file into the scene
  - `usd_path`: Path to USD file
  - `prim_path`: Where to place it in the hierarchy

#### Prim Path Operations

```python
# Check if prim exists
exists = stage_utils.is_prim_path_valid("/World/panda")

# Get prim at path
prim = stage_utils.get_prim_at_path("/World/panda")

# Delete prim
stage_utils.delete_prim("/World/objects/cup_0")
```

**Voilab Usage:**
```python
# Load base scene
stage_utils.open_stage(BASE_SCENE_FP)

# Add robot
stage_utils.add_reference_to_stage(
    usd_path=FRANKA_PANDA_FP,
    prim_path=FRANKA_PANDA_PRIM_PATH
)

# Add object
usd_path = CAD_ASSETS_DIR / usd_filename
stage_utils.add_reference_to_stage(
    usd_path=str(usd_path),
    prim_path=prim_path
)
```

---

## ROS2 Bridge Integration

### 6. ROS2 Bridge - ROS2 Communication

**Module:** `isaacsim.ros2.bridge`

Isaac Sim 5.1.0 includes comprehensive ROS2 integration.

#### Extension Initialization

```python
from isaacsim.core.utils.extensions import enable_extension

# Enable ROS2 bridge
enable_extension("isaacsim.ros2.bridge")
```

**Must be called after `SimulationApp` initialization.**

#### Action Graph ROS2 Nodes

Available ROS2 OmniGraph nodes:

**Context Node:**
```python
("Context", "isaacsim.ros2.bridge.ROS2Context")
# Inputs:
#   - domain_id: ROS2 domain ID
#   - useDomainIDEnvVar: Use ROS_DOMAIN_ID env var
```

**Joint State Publisher:**
```python
("PublisherJointState", "isaacsim.ros2.bridge.ROS2PublishJointState")
# Inputs:
#   - topicName: Topic name (e.g., "/joint_states")
#   - nodeNamespace: Namespace for the node
#   - targetPrim: Robot prim path
```

**Joint State Subscriber:**
```python
("SubscriberJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState")
# Inputs:
#   - topicName: Topic name (e.g., "/joint_command")
#   - nodeNamespace: Namespace for the node
```

**Generic Publisher:**
```python
("Publisher", "isaacsim.ros2.bridge.ROS2Publisher")
# Inputs:
#   - topicName: Topic name
#   - messageName: ROS message type (e.g., "Pose")
#   - messagePackage: Package name (e.g., "geometry_msgs")
```

**Generic Subscriber:**
```python
("Subscriber", "isaacsim.ros2.bridge.ROS2Subscriber")
# Inputs:
#   - topicName: Topic name
#   - messageName: ROS message type
#   - messagePackage: Package name
```

**Camera Helper:**
```python
("CameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper")
# Inputs:
#   - topicName: Image topic name
#   - frameId: TF frame ID
#   - nodeNamespace: Node namespace
```

**Voilab Example:**
```python
keys.CREATE_NODES: [
    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
    ("PublisherJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
    ("SubscriberJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
]

keys.SET_VALUES: [
    ("Context.inputs:domain_id", 0),
    ("PublisherJointState.inputs:topicName", "/joint_states"),
    ("PublisherJointState.inputs:targetPrim", "/World/panda/root_joint"),
    ("SubscriberJointState.inputs:topicName", "/joint_command"),
]
```

---

## OmniGraph Action Graphs

### 7. OmniGraph - Visual Programming System

**Module:** `omni.graph.core`

OmniGraph is Isaac Sim's visual programming system for creating execution pipelines.

#### Graph Creation

```python
import omni.graph.core as og

keys = og.Controller.Keys

(graph_handle, nodes, _, _) = og.Controller.edit(
    {
        "graph_path": "/World/ActionGraph",
        "evaluator_name": "execution",
    },
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "ReadSimTime.inputs:execIn"),
        ],
        keys.SET_VALUES: [
            ("OnPlaybackTick.inputs:onlyPlayback", True),
        ],
    }
)
```

#### Common Node Types

**Timing Nodes:**
- `omni.graph.action.OnPlaybackTick` - Execute on each simulation tick
- `omni.graph.action.OnTick` - Execute on every frame
- `isaacsim.core.nodes.IsaacReadSimulationTime` - Read simulation time

**Computation Nodes:**
- `omni.graph.nodes.Add` - Addition
- `omni.graph.nodes.Subtract` - Subtraction
- `omni.graph.nodes.Multiply` - Multiplication
- `omni.graph.nodes.Magnitude` - Vector magnitude

**Vector Nodes:**
- `omni.graph.nodes.BreakVector3` - Split 3D vector into components
- `omni.graph.nodes.BreakVector4` - Split 4D vector (quaternion) into components
- `omni.graph.nodes.MakeVector3` - Create 3D vector

**USD Nodes:**
- `omni.graph.nodes.ReadPrimAttribute` - Read USD attribute
- `omni.graph.nodes.WritePrimAttribute` - Write USD attribute

**Isaac Sim Nodes:**
- `isaacsim.core.nodes.IsaacArticulationController` - Control articulated robots
- `isaacsim.core.nodes.IsaacCreateRenderProduct` - Create camera render
- `isaacsim.core.nodes.OmnIsaacRunOneSimulationFrame` - Step simulation

#### Graph Connections

```python
keys.CONNECT: [
    # Source node.output -> Target node.input
    ("OnPlaybackTick.outputs:tick", "ReadSimTime.inputs:execIn"),
    ("ReadSimTime.outputs:simulationTime", "Publisher.inputs:timeStamp"),
]
```

#### Setting Node Values

```python
keys.SET_VALUES: [
    ("ReadPrimAttribute.inputs:name", "xformOp:translate"),
    ("ReadPrimAttribute.inputs:prim", "/World/panda/panda_link7"),
    ("ReadPrimAttribute.inputs:usePath", False),
]
```

**Voilab Full Example (action_graph.py):**
```python
class IsaacSimActionGraph:
    def create_action_graph(self) -> None:
        keys = og.Controller.Keys
        (graph_handle, nodes, _, _) = og.Controller.edit(
            {"graph_path": "/World/ROS_JointStates", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublisherJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublisherJointState.inputs:execIn"),
                    ("Context.outputs:context", "PublisherJointState.inputs:context"),
                ],
                keys.SET_VALUES: [
                    ("Context.inputs:domain_id", 0),
                    ("PublisherJointState.inputs:topicName", "/joint_states"),
                ],
            }
        )
        self.graph_handle = graph_handle
        self.nodes = nodes
```

---

## Camera and Sensors

### 8. Camera System

**Module:** `isaacsim.core.nodes`

#### Render Product Creation

```python
# Create render product for camera
("isaac_create_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct")

# Set camera prim path
keys.SET_VALUES: [
    ("isaac_create_render_product.inputs:cameraPrim", "/World/Camera"),
    ("isaac_create_render_product.inputs:enabled", True),
]
```

#### ROS2 Camera Helper

```python
("ros2_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper")

keys.SET_VALUES: [
    ("ros2_camera_helper.inputs:topicName", "/camera/image_raw"),
    ("ros2_camera_helper.inputs:frameId", "camera_frame"),
    ("ros2_camera_helper.inputs:nodeNamespace", ""),
]

keys.CONNECT: [
    ("isaac_create_render_product.outputs:renderProductPath", 
     "ros2_camera_helper.inputs:renderProductPath"),
]
```

#### Camera Viewport Control

```python
from isaacsim.core.utils.viewports import set_camera_view

# Set camera position and target
set_camera_view(
    eye=np.array([2.5, 1.8, 2.0]),    # Camera position
    target=np.array([0.0, 0.0, 0.8])  # Look-at target
)
```

---

## Physics and Collision

### 9. Physics Simulation

Isaac Sim 5.1.0 uses NVIDIA PhysX for physics simulation.

#### Physics Scene Settings

```python
from isaacsim.core.api import World

# Create world with physics settings
world = World(
    physics_dt=1.0/60.0,      # Physics timestep (60 Hz)
    rendering_dt=1.0/60.0,    # Rendering timestep (60 Hz)
    physics_prim_path="/physicsScene"
)
```

#### Articulation Physics

Drive settings control joint behavior:

```python
# Position drive (PD controller)
panda.set_gains(
    kps=np.array([80.0, 80.0, ...]),  # Proportional gain (stiffness)
    kds=np.array([4.0, 4.0, ...]),    # Derivative gain (damping)
    joint_indices=indices
)

# Drive types:
# - Position: PD controller to target position
# - Velocity: Velocity controller
# - Effort: Direct torque control
```

**Voilab Drive Configuration:**
```python
DRIVE_CONFIGS = {
    "panda_joint1": {"stiffness": 80.0, "damping": 4.0, "pos": 0.0},
    "panda_joint2": {"stiffness": 80.0, "damping": 4.0, "pos": -0.569},
    # ... more joints
    "panda_leftfinger": {"stiffness": 2e3, "damping": 1e2, "pos": 0.04},
    "panda_rightfinger": {"stiffness": 2e3, "damping": 1e2, "pos": 0.04},
}
```

Higher stiffness = stronger position control  
Higher damping = more resistance to velocity

---

## Extensions System

### 10. Extensions Management

**Module:** `isaacsim.core.utils.extensions`

#### Enable Extensions

```python
from isaacsim.core.utils.extensions import enable_extension

# Enable ROS2 bridge
enable_extension("isaacsim.ros2.bridge")

# Enable physics
enable_extension("omni.physx")

# Enable render products
enable_extension("omni.syntheticdata")
```

**Common Extensions:**
- `isaacsim.ros2.bridge` - ROS2 integration
- `omni.physx` - Physics simulation
- `omni.syntheticdata` - Synthetic data generation
- `omni.kit.viewport.window` - Viewport rendering
- `omni.replicator.core` - Data replication

**Voilab Pattern:**
```python
enable_extension("isaacsim.ros2.bridge")
update_simulation()  # Let extension initialize
```

---

## Best Practices and Patterns

### 11. Recommended Workflows

#### Initialization Order

**CRITICAL:** Follow this exact order:

```python
# 1. Import and create SimulationApp FIRST
from isaacsim import SimulationApp

config = {"headless": False, "width": 1280, "height": 720}
simulation_app = SimulationApp(config)

# 2. Import other Isaac Sim modules AFTER SimulationApp
import cv2
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.prims import Articulation, XFormPrim

# 3. Enable extensions
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

# 4. Update to initialize extensions
for _ in range(5):
    simulation_app.update()

# 5. Load scene and create world
stage_utils.open_stage("/path/to/scene.usd")
world = World()

# 6. Add objects and configure
# ...

# 7. Reset world before simulation
world.reset()

# 8. Main loop
while simulation_app.is_running():
    world.step(render=True)

# 9. Cleanup
simulation_app.close()
```

#### Object Spawning Pattern

```python
# 1. Add USD reference to stage
stage_utils.add_reference_to_stage(
    usd_path="/path/to/object.usd",
    prim_path="/World/objects/object_0"
)

# 2. Create XFormPrim wrapper
obj = XFormPrim(
    prim_path="/World/objects/object_0",
    name="object_0"
)

# 3. Add to world scene
world.scene.add(obj)

# 4. Set initial pose
obj.set_world_pose(
    position=np.array([1.0, 2.0, 3.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0])
)

# 5. Reset world to apply changes
world.reset()
```

#### Robot Control Pattern

```python
# 1. Add robot USD to stage
stage_utils.add_reference_to_stage(
    usd_path=ROBOT_USD_PATH,
    prim_path="/World/robot"
)

# 2. Create Articulation
robot = Articulation(
    prim_paths_expr="/World/robot",
    name="robot"
)

# 3. Add to world
world.scene.add(robot)

# 4. Reset world (important!)
world.reset()

# 5. Set world pose
robot.set_world_poses(positions=np.array([[x, y, z]]))

# 6. Configure joints
robot.set_gains(kps=kps_array, kds=kds_array, joint_indices=indices)
robot.set_joint_positions(positions=positions_array, joint_indices=indices)

# 7. Control in loop
while simulation_app.is_running():
    # Get state
    joint_positions = robot.get_joint_positions()
    
    # Apply control
    robot.set_joint_positions(target_positions)
    
    # Step
    world.step(render=True)
```

---

## Common Pitfalls

### 12. Issues and Solutions

#### 1. Import Order Error
**Problem:** Importing Isaac modules before SimulationApp
```python
# WRONG
from isaacsim.core.api import World  # Error!
from isaacsim import SimulationApp
```

**Solution:** SimulationApp must be created first
```python
# CORRECT
from isaacsim import SimulationApp
simulation_app = SimulationApp(config)
# Now import others
from isaacsim.core.api import World
```

#### 2. Forgetting world.reset()
**Problem:** Changes don't take effect
```python
panda.set_world_poses(positions=pos)
# Forgot world.reset()!
# Pose won't be applied
```

**Solution:** Call reset after major changes
```python
panda.set_world_poses(positions=pos)
world.reset()  # Apply changes
```

#### 3. Incorrect Quaternion Format
**Problem:** Using x,y,z,w instead of w,x,y,z
```python
# WRONG - OpenCV format
quaternion = [x, y, z, w]
```

**Solution:** Isaac Sim uses w-first
```python
# CORRECT - Isaac Sim format
quaternion = [w, x, y, z]

# Convert from OpenCV (x,y,z,w) to Isaac (w,x,y,z)
def xyzw_to_wxyz(q):
    return [q[3], q[0], q[1], q[2]]
```

#### 4. Path Issues in Docker
**Problem:** Absolute paths don't work in container
```python
# May not exist in container
usd_path = "/home/user/assets/robot.usd"
```

**Solution:** Use workspace-relative paths
```python
# Works in container
usd_path = "/workspace/voilab/assets/robot.usd"
```

#### 5. Extension Not Loading
**Problem:** Extension features not working
```python
enable_extension("isaacsim.ros2.bridge")
# Immediately try to use ROS2 - may fail!
```

**Solution:** Update after enabling
```python
enable_extension("isaacsim.ros2.bridge")
for _ in range(5):
    simulation_app.update()
# Now ROS2 is ready
```

---

## Migration from Previous Versions

### 13. Changes from Isaac Sim 4.x to 5.1.0

#### Package Namespace Changes

**Old (4.x):**
```python
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
```

**New (5.1.0):**
```python
from isaacsim import SimulationApp
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
```

#### Installation Changes

**Old:**
```bash
# Standalone package
./isaac-sim.sh
```

**New:**
```bash
# pip installable
pip install "isaacsim[all,isaacsim-gui,isaacsim-ros2]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
```

#### ROS2 Bridge Changes

**Old:**
```python
from omni.isaac.ros2_bridge import enable_ros2_bridge
```

**New:**
```python
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
```

---

## API Reference Summary

### Quick Reference Table

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `isaacsim.SimulationApp` | Main application | `SimulationApp()`, `.update()`, `.close()` |
| `isaacsim.core.api` | World management | `World()`, `.reset()`, `.step()` |
| `isaacsim.core.prims` | Prim wrappers | `Articulation`, `XFormPrim`, `RigidPrim` |
| `isaacsim.core.utils.stage` | USD stage ops | `open_stage()`, `add_reference_to_stage()` |
| `isaacsim.core.utils.prims` | Prim utilities | `get_prim_at_path()`, `is_prim_path_valid()` |
| `isaacsim.core.utils.extensions` | Extension system | `enable_extension()` |
| `isaacsim.ros2.bridge` | ROS2 integration | ROS2 OmniGraph nodes |
| `omni.graph.core` | Action graphs | `og.Controller`, graph creation |

### Core Data Types

**Numpy Arrays:**
- Positions: `np.array([x, y, z])` - Shape (3,)
- Quaternions: `np.array([w, x, y, z])` - Shape (4,)
- Joint arrays: `np.array([[j1, j2, ...]])` - Shape (1, N)

**Quaternion Convention:**
- Isaac Sim: **(w, x, y, z)**
- OpenCV/ROS: (x, y, z, w)

**Prim Paths:**
- Root: `/World`
- Robot example: `/World/robot`
- Object example: `/World/objects/cup_0`

---

## Voilab-Specific Patterns

### Current Implementation Highlights

#### 1. Launch Script Pattern (launch_isaacsim_workspace.py)

```python
from isaacsim import SimulationApp

config = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": ["isaacsim.ros2.bridge"]
}
simulation_app = SimulationApp(config)

# Then imports
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils

# Main workflow
enable_extension("isaacsim.ros2.bridge")
stage_utils.open_stage(BASE_SCENE_FP)
world = World()
stage_utils.add_reference_to_stage(usd_path=FRANKA_PANDA_FP, prim_path="/World/panda")
panda = Articulation(prim_paths_expr="/World/panda", name="panda_robot")
world.scene.add(panda)
world.reset()

# Object spawning
spawn_objects_from_poses(world, object_poses)

# Simulation loop
while simulation_app.is_running():
    world.step(render=True)
```

#### 2. Action Graph Pattern (action_graph.py)

```python
import omni.graph.core as og

class IsaacSimActionGraph:
    def create_action_graph(self):
        keys = og.Controller.Keys
        (graph_handle, nodes, _, _) = og.Controller.edit(
            {"graph_path": "/World/ROS_JointStates", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublisherJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublisherJointState.inputs:execIn"),
                ],
                keys.SET_VALUES: [
                    ("PublisherJointState.inputs:topicName", "/joint_states"),
                ],
            }
        )
```

#### 3. Task Registry Pattern

```python
class KitchenTaskRegistry:
    TASK_NAME = "kitchen"
    FRANKA_TRANSLATION = np.array([4.9, 3.22, 0.9])
    
    @classmethod
    def get_config(cls):
        return {
            "franka_pose": {
                "translation": cls.FRANKA_TRANSLATION,
            }
        }
```

---

## Additional Resources

### Official Documentation
- Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/
- Python API Reference: https://docs.omniverse.nvidia.com/py/isaacsim/
- USD Documentation: https://graphics.pixar.com/usd/docs/index.html

### Voilab Resources
- Docker setup: `/docs/DOCKER.md`
- Implementation plans: `/AGENT_PLANS/`
- Example code: `/scripts/launch_isaacsim_workspace.py`

---

## Conclusion

Isaac Sim 5.1.0 provides a powerful Python API for robotics simulation. Key takeaways:

1. **Initialization Order Matters**: `SimulationApp` before other imports
2. **Use world.reset()**: Required after major scene changes
3. **Quaternion Convention**: w-first format (w, x, y, z)
4. **Extension System**: Enable and update before use
5. **ROS2 Integration**: Seamless via OmniGraph action graphs

This research document provides comprehensive coverage of the APIs used and patterns followed in the voilab project. For implementation details, refer to the example code in the repository.

---

**End of Report**
