
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
9. [Detailed Implementation Plan](#detailed-implementation-plan)

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

### Actual `object_poses.json` Format (Per-Episode Reconstruction)

The reconstruction system generates per-episode object pose data with the following structure:

```json
{
  "video_name": "converted_60fps_raw_video.mp4",
  "episode_range": [1878, 2285],
  "objects": [
    {
      "object_name": "blue_cup",
      "rvec": [-1.7764613338639368, -0.35174590123848837, -0.2668578462849374],
      "tvec": [-0.017231631611681134, -0.2640980962680338, 0.11816561790000263]
    },
    {
      "object_name": "plate",
      "rvec": [-1.234, 0.456, 0.789],
      "tvec": [0.3, -0.1, 0.82]
    }
  ],
  "status": "partial"
}
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

## Detailed Implementation Plan

### Step-by-Step Tasks

1. **Create Utility Modules**
   - Create `scripts/pose_utils.py` with rotation vector to quaternion conversion functions.
   - Create `scripts/object_loader.py` with object spawning and mapping logic.
   - Ensure both modules have proper error handling and logging.

2. **Modify `launch_isaacsim_workspace.py`**
   - Import the new utility modules.
   - Add logic to construct the path to `object_poses.json` (fixed subdirectory `mapping`).
   - After robot is added but before `world.reset()`, call `load_objects_from_json`.
   - Handle missing JSON gracefully (log warning, continue without objects).

3. **Update Object‑to‑Asset Mapping**
   - Define a registry mapping object names (e.g., `blue_cup`) to USD filenames (e.g., `cup_finalized.usd`).
   - Place the registry in `object_loader.py` for easy maintenance.

4. **Test with Sample Data**
   - Use an existing `object_poses.json` (e.g., `datasets/1124_gopro3_kitchen/demos/mapping/object_poses.json`).
   - Run the modified launcher with `--session_dir` pointing to the dataset.
   - Verify objects appear at correct positions and orientations.

5. **Integrate with Existing Workflow**
   - Ensure object spawning does not interfere with robot initialization or replay.
   - Confirm that physics properties (if any) are correctly applied.

6. **Documentation and Cleanup**
   - Update this research report with any changes made during implementation.
   - Add inline comments in the code for clarity.

### Integration Points in `launch_isaacsim_workspace.py`

```python
# After line 104: stage_utils.add_reference_to_stage(...)
# Add:
if args.session_dir:
    # Construct path to object_poses.json (fixed subdirectory 'mapping')
    json_path = os.path.join(args.session_dir, 'demos', 'mapping', 'object_poses.json')
    assets_dir = "/workspace/voilab/assets/CADs"
    load_objects_from_json(json_path, assets_dir, world, episode_index=args.episode)

# Before line 110: world.reset()
# Objects should be added before world.reset()
```

### Testing Strategy

1. **Unit Tests for Pose Conversion**
   - Test `rvec_to_quaternion_wxyz` with known rotation vectors.
   - Verify quaternion normalization and WXYZ format.

2. **Integration Test**
   - Run the launcher with a known session directory and episode.
   - Visually inspect that objects are placed correctly in Isaac Sim.

3. **Error‑Handling Tests**
   - Provide a missing asset path; ensure graceful skipping.
   - Provide malformed JSON; ensure script continues.

### Timeline and Dependencies

- **Day 1**: Create utility modules and basic object loading.
- **Day 2**: Integrate into main script and test with sample data.
- **Day 3**: Refine error handling, add logging, and finalize documentation.

---

## Conclusion

This research provides a comprehensive guide to implementing object loading from `object_poses.json` using Isaac Sim V5.1.0 standalone API. The recommended approach uses:

1. **Object Data Format**: Per‑episode JSON with `rvec` (rotation vector) and `tvec` (translation vector)
2. **Path Structure**: `$SESSION_DIR/demos/mapping/object_poses.json`
3. **Spawning**: `stage_utils.add_reference_to_stage()` for USD reference loading
4. **Pose Application**: `XFormPrim.set_world_pose()` with WXYZ quaternions
5. **Rotation Conversion**: `scipy.spatial.transform.Rotation.from_rotvec()` to convert rvec to quaternion
6. **Object Mapping**: Custom mapping function to match object names to USD asset files
7. **Error Handling**: Graceful degradation if objects or files are missing

**Key Implementation Steps:**
1. Parse `object_poses.json` to extract objects with rvec/tvec
2. Convert rvec to quaternion WXYZ using scipy
3. Map object_name to USD asset file (e.g., "blue_cup" → "cup_finalized.usd")
4. Spawn objects using `add_reference_to_stage()` after robot, before `world.reset()`
5. Apply poses using `set_world_pose()` with converted quaternions

The implementation should be minimal, focused, and follow existing patterns in the codebase.

---

**Key Fields:**
- `video_name`: Source video file for this episode
- `episode_range`: Frame range [start, end] for this episode
- `objects`: List of reconstructed objects with poses
  - `object_name`: Name identifier for the object
  - `rvec`: Rotation vector (axis-angle representation) [x, y, z] in radians
  - `tvec`: Translation vector [x, y, z] in meters
- `status`: Reconstruction status ("partial", "complete", etc.)

**Important Notes:**
- **Rotation Format**: `rvec` is a **rotation vector** (also called axis-angle or Rodrigues vector)
  - The direction of the vector is the rotation axis
  - The magnitude (length) is the rotation angle in radians
  - This is OpenCV's standard representation from `cv2.Rodrigues()`
- **Translation Format**: `tvec` is in meters, representing object position in the camera/world frame
- **File Location**: `$SESSION_DIR/demos/$EPISODE/object_poses.json`

### Example with Multiple Objects

```json
{
  "video_name": "converted_60fps_raw_video.mp4",
  "episode_range": [1878, 2285],
  "objects": [
    {
      "object_name": "blue_cup",
      "rvec": [-1.7764613338639368, -0.35174590123848837, -0.2668578462849374],
      "tvec": [-0.017231631611681134, -0.2640980962680338, 0.11816561790000263]
    },
    {
      "object_name": "white_plate",
      "rvec": [0.123, -0.456, 0.789],
      "tvec": [0.25, -0.15, 0.80]
    },
    {
      "object_name": "fork",
      "rvec": [0.0, 0.0, 1.5708],
      "tvec": [0.35, 0.0, 0.83]
    }
  ],
  "status": "complete"
}
```

---

## Implementation Architecture

### High-Level Flow

```
1. Parse command-line arguments (session_dir, episode)
2. Construct path to object_poses.json (fixed: $SESSION_DIR/demos/mapping/object_poses.json)
3. Load and parse JSON file (array of episodes)
4. Open base scene (ED305.usd)
5. Initialize World
6. Add robot to scene
7. Load the specified episode (or default episode 0) from the JSON array
   a. For each object in episode.objects:
        i. Construct full USD asset path using mapping
        ii. Generate unique prim path
        iii. Add reference to stage
        iv. Create XFormPrim wrapper
        v. Apply pose from JSON (convert rvec/tvec to quaternion WXYZ)
        vi. Add to World scene
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

#### **Pose Conversion Utilities** (Updated for rvec/tvec Format)

```python
def rvec_to_quaternion_wxyz(rvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector (rvec) to quaternion WXYZ format"""
    pass

def quaternion_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion from XYZW to WXYZ format"""
    pass

def validate_quaternion(quat: np.ndarray) -> np.ndarray:
    """Validate and normalize quaternion"""
    pass

def rvec_tvec_to_transform_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert rvec/tvec to 4x4 transformation matrix"""
    pass

def map_object_name_to_asset(object_name: str) -> str:
    """Map object name from reconstruction to USD asset filename"""
    pass

def transform_tvec_to_world_frame(tvec: np.ndarray, world_frame: np.ndarray) -> np.ndarray:
    """Transform translation vector from camera frame to world frame"""
    pass
```

---

## Code Examples

### Example 1: Basic Object Loading (Updated for Actual Format)

```python
import json
import os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim

def load_objects_from_json(json_path: str, assets_dir: str, world: World, episode_index: int = 0):
    """
    Load objects from a specific episode in object_poses.json and spawn them in the scene.
    
    The JSON format is an array of episodes, each containing objects with rvec (rotation vector)
    and tvec (translation vector) from the reconstruction system.
    
    Args:
        json_path: Path to object_poses.json (contains array of episodes)
        assets_dir: Base directory for CAD assets (e.g., /workspace/voilab/assets/CADs)
        world: Isaac Sim World instance
        episode_index: Index of the episode to load (default 0)
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # data is an array of episodes
    if isinstance(data, list):
        episodes = data
    else:
        episodes = [data]
    
    # Validate episode index
    if episode_index < 0 or episode_index >= len(episodes):
        print(f"[ObjectLoader] ERROR: Episode index {episode_index} out of range (0-{len(episodes)-1}), skipping object loading")
        return
    
    episode = episodes[episode_index]
    objects = episode.get('objects', [])
    print(f"[ObjectLoader] Found {len(objects)} objects in episode {episode_index}")
    
    total_objects = 0
    for idx, obj in enumerate(objects):
        object_name = obj.get('object_name', f'object_{idx}')
        rvec = np.array(obj.get('rvec', [0, 0, 0]))
        tvec = np.array(obj.get('tvec', [0, 0, 0]))
        
        # Convert rotation vector (axis-angle) to quaternion WXYZ
        # rvec is a rotation vector where:
        # - Direction is the axis of rotation
        # - Magnitude is the angle in radians
        rot = R.from_rotvec(rvec)
        quat_xyzw = rot.as_quat()  # scipy returns [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        # Map object_name to USD asset file
        asset_filename = map_object_name_to_asset(object_name)
        full_asset_path = os.path.join(assets_dir, asset_filename)
        # Check if asset exists
        if not os.path.exists(full_asset_path):
            print(f"[ObjectLoader] WARNING: Asset not found: {full_asset_path}, skipping {object_name}")
            continue
        
        # Create unique prim path
        prim_path = f"/World/objects/{object_name}_{total_objects}"
        
        # Add reference to stage
        stage_utils.add_reference_to_stage(
            usd_path=full_asset_path,
            prim_path=prim_path
        )
        
        # Create XFormPrim and set pose
        obj_prim = XFormPrim(prim_path=prim_path, name=f"{object_name}_{total_objects}")
        obj_prim.set_world_pose(position=tvec, orientation=quat_wxyz)
        
        # Add to world scene
        world.scene.add(obj_prim)
        
        print(f"[ObjectLoader] Spawned {object_name} at position {tvec}")
        total_objects += 1
    
    print(f"[ObjectLoader] Total objects spawned: {total_objects}")

def map_object_name_to_asset(object_name: str) -> str:
    """
    Map object name from reconstruction to USD asset filename.
    
    Args:
        object_name: Name from object_poses.json (e.g., "blue_cup", "white_plate")
        
    Returns:
        str: USD filename (e.g., "cup_finalized.usd")
    """
    # Define mapping from object names to USD files
    # Customize this based on your asset library
    # THIS REGISTRY IS USED IN THE RECONSTRUCTION SYSTEM
    # BE PLACED TO THE BEGINNING OF THE FILE
    REGISTRY = {
        'blue_cup': 'cup_finalized.usd',
        'cup': 'cup_finalized.usd',
        'white_plate': 'plate.usd',
        'plate': 'plate.usd',
        'fork': 'fork_final.usd',
        'knife': 'knife_final.usd',
        'spoon': 'spoon.usd',
        'storage_box': 'storage_box.usd',
        'table': 'table.usd',
    }
    
    # Try exact match first
    if object_name in REGISTRY:
        return REGISTRY[object_name]
    
    # Try partial match (e.g., "blue_cup" contains "cup")
    for key, asset in REGISTRY.items():
        if key in object_name.lower():
            return asset
    
    # Default: try to construct filename from object_name
    # e.g., "blue_cup" -> "cup.usd"
    base_name = object_name.lower().split('_')[-1]  # Get last word
    return f"{base_name}.usd"
```
               
