# Isaac Sim V5.1.0 Standalone API - Object Spawning Research Report

## Executive Summary

This document provides a comprehensive research report on using Isaac Sim V5.1.0 standalone API to dynamically load and position objects from an external `object_poses.json` file. The goal is to modify `scripts/launch_isaacsim_workspace.py` to read reconstructed object poses and spawn corresponding USD models from `voilab/assets/CADs` at the specified positions and orientations.

## Table of Contents

1. [IsaacSim V5.1.0 API Overview](#isaacgsim-v510-api-overview)
2. [Object Spawning Mechanisms](#object-spawning-mechanisms)
3. [Pose Application Methods](#pose-application-methods)
4. [JSON Data Structure](#json-data-structure)
5. [Implementation Architecture](#implementation-architecture)
6. [Code Examples](#code-examples)
7. [Best Practices](#best-practices)
8. [Error Handling](#error-handling)

---

## IsaacSim V5.1.0 API Overview

### Core Modules

Isaac Sim V5.1.0 provides several key modules for programmatic scene manipulation:

#### 1. **isaacsim.core.api**
- `World`: Main simulation context manager
- Handles simulation stepping, physics, and scene lifecycle

#### 2. **isaacsim.core.utils.stage**
- `stage_utils.add_reference_to_stage()`: Primary method for adding USD assets to scene
- `stage_utils.open_stage()`: Opens existing USD stages
- `stage_utils.get_current_stage()`: Gets active USD stage for manipulation

#### 3. **isaacsim.core.prims**
- `XFormPrim`: Basic transformable USD primitive with pose manipulation
- `RigidPrimView`: Physics-enabled rigid body wrapper
- `GeometryPrim`: Base class for geometric objects

#### 4. **Pxr.Usd and Pxr.UsdGeom** (Pixar USD)
- Low-level USD manipulation for fine-grained control
- `UsdGeom.Xformable`: Direct transformation manipulation
- `UsdGeom.SetTransform()`: Apply transformation matrices

### API Version Considerations

Isaac Sim V5.1.0 represents a significant update from V5.0.0 with the following key changes:

1. **Namespace Migration**: Changed from `omni.isaac.*` to `isaacsim.*`
   - Old: `omni.isaac.core.utils.stage`
   - New: `isaacsim.core.utils.stage`

2. **Improved World API**: Better lifecycle management and scene reset capabilities

3. **Enhanced Prim Management**: More robust reference handling and pose setting

---

## Object Spawning Mechanisms

### Method 1: Using `stage_utils.add_reference_to_stage()`

This is the **recommended approach** for Isaac Sim V5.1.0, as it's used throughout the codebase.

```python
import isaacsim.core.utils.stage as stage_utils

# Add a USD reference to the stage
stage_utils.add_reference_to_stage(
    usd_path="/workspace/voilab/assets/CADs/cup_finalized.usd",
    prim_path="/World/objects/cup_001"
)
```

**Key Features:**
- Creates a reference (not a copy) to the USD file
- Automatically handles namespace and prim hierarchy
- Preserves original USD structure
- Memory efficient for multiple instances

**When to Use:**
- Loading external USD assets
- When you need to spawn multiple instances of the same object
- When object USD has internal structure (materials, physics, etc.)

### Method 2: Using XFormPrim with World.scene

```python
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim

world = World()

# Create XFormPrim wrapper after adding reference
stage_utils.add_reference_to_stage(
    usd_path="/workspace/voilab/assets/CADs/plate.usd",
    prim_path="/World/objects/plate_001"
)

# Wrap in XFormPrim for pose manipulation
plate_prim = XFormPrim(
    prim_path="/World/objects/plate_001",
    name="plate_001"
)

# Add to scene for World management
world.scene.add(plate_prim)
```

**Key Features:**
- Provides high-level pose manipulation interface
- Integrates with World scene management
- Easier pose setting with `set_world_pose()`
- Better for objects that need dynamic updates

**When to Use:**
- When you need to manipulate object poses programmatically
- When objects are part of simulation logic
- When you want World to track and reset objects

### Method 3: Direct USD Manipulation (Advanced)

```python
from pxr import Usd, UsdGeom, Gf
import isaacsim.core.utils.stage as stage_utils

# Get current stage
stage = stage_utils.get_current_stage()

# Create or get prim
prim = stage.DefinePrim("/World/objects/knife_001", "Xform")

# Add USD reference as payload or reference
prim.GetReferences().AddReference(
    "/workspace/voilab/assets/CADs/knife_final.usd"
)

# Get xformable and set transform
xformable = UsdGeom.Xformable(prim)
xform_op = xformable.AddTransformOp()
```

**Key Features:**
- Fine-grained control over USD structure
- Direct manipulation of USD properties
- Can batch operations for performance

**When to Use:**
- When you need advanced USD features
- Batch loading many objects
- Custom USD manipulation requirements

---

## Pose Application Methods

### Understanding Coordinate Systems

Isaac Sim uses the following conventions:
- **Units**: Meters for distance
- **Rotation**: Quaternions (WXYZ or XYZW depending on API)
- **Coordinate System**: Right-handed, Y-up (standard USD)
- **Transform Order**: Translation → Rotation → Scale

### Method 1: `XFormPrim.set_world_pose()` (Recommended)

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# Position as [x, y, z] in meters
position = np.array([0.5, 0.3, 0.1])

# Rotation as quaternion [w, x, y, z] 
# Note: Isaac Sim uses WXYZ format
rotation_matrix = R.from_euler('xyz', [0, 0, np.pi/4]).as_matrix()
quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

# Apply pose
prim.set_world_pose(position=position, orientation=quat_wxyz)
```

**Key Points:**
- Position in world coordinates (meters)
- Quaternion format: WXYZ (w, x, y, z)
- Call after World.reset() or during World.step()
- Requires object to be added to scene

### Method 2: Direct USD Transform Setting

```python
from pxr import Gf, UsdGeom
import isaacsim.core.utils.stage as stage_utils

stage = stage_utils.get_current_stage()
prim = stage.GetPrimAtPath("/World/objects/fork_001")

# Create xformable
xformable = UsdGeom.Xformable(prim)

# Create translation
translate_op = xformable.AddTranslateOp()
translate_op.Set(Gf.Vec3d(0.5, 0.3, 0.1))

# Create rotation (as quaternion or euler)
orient_op = xformable.AddOrientOp()
# Quaternion in USD: (w, x, y, z) - WXYZ format
quat = Gf.Quatd(1.0, 0.0, 0.0, 0.0)  
orient_op.Set(quat)
```

**Key Points:**
- More control over transform stack
- Can create custom transform ops
- Persists in USD file
- Lower level, more complex

### Method 3: Setting Transform Matrix

```python
import numpy as np
from pxr import Gf, UsdGeom

# Build 4x4 transformation matrix
T = np.eye(4)
T[:3, 3] = [0.5, 0.3, 0.1]  # translation
T[:3, :3] = rotation_matrix   # rotation

# Convert to USD Gf.Matrix4d
gf_matrix = Gf.Matrix4d(*T.flatten().tolist())

# Apply to prim
xformable = UsdGeom.Xformable(prim)
xform_op = xformable.AddTransformOp()
xform_op.Set(gf_matrix)
```

**Key Points:**
- Useful when working with transformation matrices
- Single operation for position and rotation
- Good for importing from external systems

---

## JSON Data Structure

### Expected `object_poses.json` Format

Based on the problem statement and common robotics conventions:

```json
{
  "objects": [
    {
      "name": "cup",
      "asset_path": "cup_finalized.usd",
      "translation": [0.45, 0.2, 0.85],
      "rotation": {
        "quaternion": [1.0, 0.0, 0.0, 0.0]
      }
    },
    {
      "name": "plate",
      "asset_path": "plate.usd",
      "translation": [0.3, -0.1, 0.82],
      "rotation": {
        "quaternion": [0.9239, 0.0, 0.0, 0.3827]
      }
    },
    {
      "name": "knife",
      "asset_path": "knife_final.usd",
      "translation": [0.35, 0.0, 0.83],
      "rotation": {
        "quaternion": [0.7071, 0.0, 0.0, 0.7071]
      }
    }
  ],
  "metadata": {
    "coordinate_frame": "world",
    "units": "meters",
    "quaternion_format": "wxyz"
  }
}
```

### Alternative Format (Axis-Angle or Euler)

```json
{
  "objects": [
    {
      "name": "fork",
      "asset_path": "fork_final.usd",
      "translation": [0.4, 0.15, 0.83],
      "rotation": {
        "type": "axis_angle",
        "axis": [0, 0, 1],
        "angle": 1.5708
      }
    },
    {
      "name": "spoon",
      "asset_path": "spoon.usd",
      "translation": [0.25, 0.15, 0.83],
      "rotation": {
        "type": "euler",
        "angles": [0, 0, 90],
        "sequence": "xyz",
        "degrees": true
      }
    }
  ]
}
```

### Minimal Format

```json
{
  "objects": [
    {
      "asset": "cup_finalized.usd",
      "pose": {
        "position": [0.45, 0.2, 0.85],
        "orientation": [1.0, 0.0, 0.0, 0.0]
      }
    }
  ]
}
```

---

## Implementation Architecture

### High-Level Flow

```
1. Parse command-line arguments (session_dir, episode)
2. Construct path to object_poses.json
3. Load and parse JSON file
4. Open base scene (ED305.usd)
5. Initialize World
6. Add robot to scene
7. For each object in JSON:
   a. Construct full USD asset path
   b. Generate unique prim path
   c. Add reference to stage
   d. Create XFormPrim wrapper
   e. Apply pose from JSON
   f. Add to World scene
8. Reset world
9. Start simulation loop
```

### Module Structure

```
scripts/
├── launch_isaacsim_workspace.py    (main script - to be modified)
├── object_loader.py                 (new - object loading utilities)
└── pose_utils.py                    (new - pose conversion utilities)
```

### Key Classes/Functions

#### 1. **ObjectPoseLoader** (New Class)

```python
class ObjectPoseLoader:
    """Loads object poses from JSON and spawns them in Isaac Sim"""
    
    def __init__(self, json_path: str, assets_dir: str):
        self.json_path = json_path
        self.assets_dir = assets_dir
        self.objects_data = self._load_json()
    
    def _load_json(self) -> dict:
        """Load and validate JSON file"""
        pass
    
    def spawn_objects(self, world: World, parent_prim: str = "/World/objects"):
        """Spawn all objects from JSON into the scene"""
        pass
    
    def _spawn_single_object(self, obj_data: dict, prim_path: str):
        """Spawn a single object with its pose"""
        pass
    
    def _convert_rotation(self, rotation_data: dict) -> np.ndarray:
        """Convert rotation from JSON format to quaternion WXYZ"""
        pass
```

#### 2. **Pose Conversion Utilities** (New Module)

```python
def json_to_quaternion(rotation_data: dict) -> np.ndarray:
    """Convert various rotation formats to quaternion WXYZ"""
    pass

def quaternion_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion from XYZW to WXYZ format"""
    pass

def euler_to_quaternion(angles: list, sequence: str = 'xyz', degrees: bool = False) -> np.ndarray:
    """Convert Euler angles to quaternion WXYZ"""
    pass

def axis_angle_to_quaternion(axis: list, angle: float) -> np.ndarray:
    """Convert axis-angle to quaternion WXYZ"""
    pass
```

---

## Code Examples

### Example 1: Basic Object Loading

```python
import json
import os
import numpy as np
from pathlib import Path
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim

def load_objects_from_json(json_path: str, assets_dir: str, world: World):
    """
    Load objects from object_poses.json and spawn them in the scene.
    
    Args:
        json_path: Path to object_poses.json
        assets_dir: Base directory for CAD assets
        world: Isaac Sim World instance
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    objects = data.get('objects', [])
    
    # Spawn each object
    for idx, obj in enumerate(objects):
        name = obj.get('name', f'object_{idx}')
        asset_path = obj.get('asset_path', obj.get('asset'))
        translation = np.array(obj.get('translation', obj.get('pose', {}).get('position', [0, 0, 0])))
        
        # Handle rotation
        rotation_data = obj.get('rotation', obj.get('pose', {}).get('orientation'))
        if isinstance(rotation_data, dict):
            quat = np.array(rotation_data.get('quaternion', [1, 0, 0, 0]))
        else:
            quat = np.array(rotation_data)
        
        # Ensure quaternion is WXYZ format
        if len(quat) == 4 and quat[0] < 1.0:  # Heuristic check
            # Might be XYZW, convert to WXYZ
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
        
        # Construct full paths
        full_asset_path = os.path.join(assets_dir, asset_path)
        prim_path = f"/World/objects/{name}_{idx}"
        
        # Add reference to stage
        stage_utils.add_reference_to_stage(
            usd_path=full_asset_path,
            prim_path=prim_path
        )
        
        # Create XFormPrim and set pose
        obj_prim = XFormPrim(prim_path=prim_path, name=f"{name}_{idx}")
        obj_prim.set_world_pose(position=translation, orientation=quat)
        
        # Add to world scene
        world.scene.add(obj_prim)
        
        print(f"[ObjectLoader] Spawned {name} at {translation}")
```

### Example 2: Robust Rotation Conversion

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_rotation_to_quaternion_wxyz(rotation_data):
    """
    Convert various rotation formats to quaternion WXYZ.
    
    Args:
        rotation_data: Dict with rotation information or list/array
        
    Returns:
        np.ndarray: Quaternion in WXYZ format [w, x, y, z]
    """
    # If already a list/array, assume it's a quaternion
    if isinstance(rotation_data, (list, np.ndarray)):
        quat = np.array(rotation_data)
        # Check if it's XYZW (common) or WXYZ (Isaac Sim)
        # Heuristic: if first element > last element, likely WXYZ already
        if len(quat) == 4:
            if quat[0] > quat[3]:  # Likely WXYZ
                return quat
            else:  # Likely XYZW, convert
                return np.array([quat[3], quat[0], quat[1], quat[2]])
    
    # Handle dict with type specification
    if isinstance(rotation_data, dict):
        rot_type = rotation_data.get('type', 'quaternion')
        
        if rot_type == 'quaternion' or 'quaternion' in rotation_data:
            quat = np.array(rotation_data.get('quaternion', [1, 0, 0, 0]))
            # Assume WXYZ if w is first
            format_hint = rotation_data.get('format', 'wxyz')
            if format_hint == 'xyzw':
                return np.array([quat[3], quat[0], quat[1], quat[2]])
            return quat
        
        elif rot_type == 'euler':
            angles = rotation_data.get('angles', [0, 0, 0])
            sequence = rotation_data.get('sequence', 'xyz')
            degrees = rotation_data.get('degrees', False)
            
            if degrees:
                angles = np.deg2rad(angles)
            
            rot = R.from_euler(sequence, angles)
            quat_xyzw = rot.as_quat()
            return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        elif rot_type == 'axis_angle':
            axis = np.array(rotation_data.get('axis', [0, 0, 1]))
            angle = rotation_data.get('angle', 0.0)
            
            rot = R.from_rotvec(axis * angle)
            quat_xyzw = rot.as_quat()
            return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        elif rot_type == 'matrix':
            matrix = np.array(rotation_data.get('matrix'))
            rot = R.from_matrix(matrix)
            quat_xyzw = rot.as_quat()
            return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    # Default: identity rotation
    return np.array([1.0, 0.0, 0.0, 0.0])
```

### Example 3: Integration into main()

```python
def main():
    print(f"[Main] Starting with task: {args.task}")
    
    # Registry and configuration
    registry_class = registry.get_task_registry(args.task)
    if not registry_class.validate_environment():
        print(f"[Main] WARNING: Registry validation failed")
    
    cfg = registry_class.get_config()
    franka_translation = cfg.get("franka_pose", {}).get("translation", [0, 0, 0])
    camera_translation = cfg.get("camera_pose", {}).get("translation", [0, 0, 0])
    
    # Open stage and create world
    stage_utils.open_stage(BASE_SCENE_FP)
    world = World()
    
    # Add robot
    stage_utils.add_reference_to_stage(usd_path=FRANKA_PANDA_FP, prim_path=FRANKA_PANDA_PRIM_PATH)
    panda = SingleArticulation(FRANKA_PANDA_PRIM_PATH, name="panda_robot")
    world.scene.add(panda)
    
    # Load objects from JSON if session_dir provided
    if args.session_dir:
        object_poses_path = os.path.join(
            args.session_dir, 
            'demos',
            args.episode if isinstance(args.episode, str) else f'demo_{args.episode}',
            'object_poses.json'
        )
        
        if os.path.exists(object_poses_path):
            print(f"[Main] Loading objects from: {object_poses_path}")
            assets_dir = "/workspace/voilab/assets/CADs"
            load_objects_from_json(object_poses_path, assets_dir, world)
        else:
            print(f"[Main] WARNING: object_poses.json not found at {object_poses_path}")
    
    # Set camera and reset world
    set_camera_view(camera_translation, franka_translation)
    world.reset()
    
    # Set robot pose
    print(f"[Main] Setting robot world pose to: {franka_translation}")
    panda.set_world_pose(position=franka_translation)
    setup_robot_drives(panda)
    
    # Settle simulation
    print("[Main] Settling robot into calibration pose (200 steps)...")
    for _ in range(200):
        world.step(render=True)
    
    # Continue with replay logic...
```

---

## Best Practices

### 1. Object Naming and Hierarchy

```python
# Good: Organized hierarchy
/World/objects/kitchen/cup_001
/World/objects/kitchen/plate_001
/World/objects/dining/fork_001

# Bad: Flat structure
/World/cup
/World/plate
/World/fork
```

### 2. Error Handling

```python
# Always validate paths
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Object poses JSON not found: {json_path}")

if not os.path.exists(usd_asset_path):
    print(f"WARNING: Asset not found: {usd_asset_path}, skipping...")
    continue
```

### 3. Performance Considerations

```python
# Batch operations when possible
objects_to_spawn = []
for obj in objects_data:
    objects_to_spawn.append((obj['asset_path'], obj['translation'], obj['rotation']))

# Spawn all at once
for asset, trans, rot in objects_to_spawn:
    spawn_object(asset, trans, rot)

# Then reset world once
world.reset()
```

### 4. Coordinate Frame Validation

```python
# Validate coordinate frame matches expectations
metadata = data.get('metadata', {})
coord_frame = metadata.get('coordinate_frame', 'unknown')

if coord_frame != 'world':
    print(f"WARNING: Coordinate frame is '{coord_frame}', expected 'world'")
    # Apply transformation if needed
```

### 5. Physics Properties

```python
from isaacsim.core.prims import RigidPrim

# If objects need physics
rigid_obj = RigidPrim(prim_path=prim_path, name=name)
rigid_obj.enable_rigid_body_physics()
rigid_obj.set_mass(0.5)  # kg
world.scene.add(rigid_obj)
```

---

## Error Handling

### Common Issues and Solutions

#### Issue 1: Asset Not Found

```python
def safe_load_asset(asset_path: str, prim_path: str) -> bool:
    """Safely load USD asset with error handling"""
    try:
        if not os.path.exists(asset_path):
            print(f"ERROR: Asset not found: {asset_path}")
            return False
        
        stage_utils.add_reference_to_stage(
            usd_path=asset_path,
            prim_path=prim_path
        )
        return True
    except Exception as e:
        print(f"ERROR: Failed to load asset {asset_path}: {str(e)}")
        return False
```

#### Issue 2: Invalid Quaternion

```python
def validate_quaternion(quat: np.ndarray) -> np.ndarray:
    """Validate and normalize quaternion"""
    quat = np.array(quat)
    
    if len(quat) != 4:
        print(f"ERROR: Invalid quaternion length: {len(quat)}, using identity")
        return np.array([1, 0, 0, 0])
    
    # Normalize
    norm = np.linalg.norm(quat)
    if norm < 0.001:
        print("ERROR: Zero-length quaternion, using identity")
        return np.array([1, 0, 0, 0])
    
    return quat / norm
```

#### Issue 3: JSON Parse Errors

```python
def load_json_safe(json_path: str) -> dict:
    """Safely load JSON with error handling"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON format in {json_path}: {str(e)}")
        return {"objects": []}
    except FileNotFoundError:
        print(f"ERROR: File not found: {json_path}")
        return {"objects": []}
    except Exception as e:
        print(f"ERROR: Unexpected error loading JSON: {str(e)}")
        return {"objects": []}
```

---

## Summary and Recommendations

### Recommended Implementation Approach

1. **Minimal Changes**: Modify only `launch_isaacsim_workspace.py`
2. **Use Existing Patterns**: Follow patterns from robot loading
3. **Robust Error Handling**: Gracefully handle missing files/assets
4. **Flexible JSON Format**: Support multiple rotation representations
5. **Clear Logging**: Print what objects are loaded and where

### Key API Functions to Use

1. `stage_utils.add_reference_to_stage()` - Object spawning
2. `XFormPrim.set_world_pose()` - Pose application
3. `World.scene.add()` - Scene management
4. `scipy.spatial.transform.Rotation` - Rotation conversions

### Integration Points in launch_isaacsim_workspace.py

```python
# After line 104: stage_utils.add_reference_to_stage(...)
# Add:
if args.session_dir:
    load_objects_from_poses_json(args.session_dir, args.episode, world)

# Before line 110: world.reset()
# Objects should be added before world.reset()
```

### Testing Strategy

1. Create sample `object_poses.json` in expected location
2. Run launcher with `--session_dir` and `--episode` arguments
3. Verify objects appear in scene at correct positions
4. Test with missing files (graceful degradation)
5. Test with various rotation formats

---

## Appendix: Reference Documentation

### Isaac Sim V5.1.0 API References

- **Core API**: `isaacsim.core.api`
- **Stage Utils**: `isaacsim.core.utils.stage`
- **Prims**: `isaacsim.core.prims`
- **USD**: Pixar's USD Python API

### Key Differences from V5.0.0

- Namespace change from `omni.isaac.*` to `isaacsim.*`
- Improved `World` API with better scene management
- Enhanced error messages and logging
- Better support for dynamic object loading

### Quaternion Convention

Isaac Sim consistently uses **WXYZ** format:
- `[w, x, y, z]` where `w² + x² + y² + z² = 1`
- This matches the Pixar USD Gf.Quatd convention
- SciPy uses XYZW, so conversion is often needed

---

## Conclusion

This research provides a comprehensive guide to implementing object loading from `object_poses.json` using Isaac Sim V5.1.0 standalone API. The recommended approach uses:

1. `stage_utils.add_reference_to_stage()` for spawning
2. `XFormPrim.set_world_pose()` for pose application
3. `scipy.spatial.transform.Rotation` for rotation conversion
4. Robust error handling for production use

The implementation should be minimal, focused, and follow existing patterns in the codebase.
