#!/usr/bin/env python3
"""
Unit tests for launch_isaacsim_workspace.py functions

These tests validate the object pose loading and conversion logic
without requiring Isaac Sim to be installed.

Note: Functions are duplicated here rather than imported because
launch_isaacsim_workspace.py imports Isaac Sim modules at the top level,
which are not available in the test environment. This duplication is
intentional to enable testing without Isaac Sim dependencies.
"""

import json
import tempfile
import numpy as np
import cv2
from pathlib import Path
import sys


def load_object_poses(session_dir: Path, episode: int):
    """
    Load object poses from object_poses.json for a specific episode.
    (Extracted copy of the function from launch_isaacsim_workspace.py)
    """
    object_poses_path = session_dir / "demos" / "mapping" / "object_poses.json"
    
    if not object_poses_path.exists():
        print(f"[ObjectPoses] WARNING: object_poses.json not found at {object_poses_path}")
        return []
    
    try:
        with open(object_poses_path, "r") as f:
            all_episodes_data = json.load(f)
        
        if episode >= len(all_episodes_data):
            print(f"[ObjectPoses] WARNING: Episode {episode} not found in object_poses.json (only {len(all_episodes_data)} episodes)")
            return []
        
        episode_data = all_episodes_data[episode]
        objects = episode_data.get("objects", [])
        status = episode_data.get("status", "unknown")
        
        print(f"[ObjectPoses] Loaded episode {episode} with status: {status}")
        print(f"[ObjectPoses] Found {len(objects)} objects: {[obj['object_name'] for obj in objects]}")
        
        return objects
    
    except Exception as e:
        print(f"[ObjectPoses] ERROR loading object_poses.json: {e}")
        return []


def rvec_to_quaternion(rvec):
    """
    Convert rotation vector (OpenCV format) to quaternion (w, x, y, z).
    (Extracted copy of the function from launch_isaacsim_workspace.py)
    """
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z], dtype=np.float64)


OBJECT_NAME_TO_USD = {
    # Kitchen objects
    "pink_cup": "cup_finalized.usd",
    "blue_cup": "cup_finalized.usd",
    # Dining room objects
    "fork": "fork_final.usd",
    "knife": "knife_final.usd",
    "plate": "plate.usd",
    # Living room objects
    "blue_block": "storage_box.usd",
    "green_block": "storage_box.usd",
    "red_block": "storage_box.usd",
}


def test_load_object_poses():
    """Test loading object poses from JSON file"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create directory structure
        demos_dir = tmpdir / "demos" / "mapping"
        demos_dir.mkdir(parents=True)
        
        # Create mock object_poses.json
        object_poses_data = [
            {
                "video_name": "test_video.mp4",
                "episode_range": [0, 100],
                "objects": [
                    {
                        "object_name": "pink_cup",
                        "rvec": [0.1, 0.2, 0.3],
                        "tvec": [1.0, 2.0, 3.0]
                    },
                    {
                        "object_name": "blue_cup",
                        "rvec": [0.4, 0.5, 0.6],
                        "tvec": [4.0, 5.0, 6.0]
                    }
                ],
                "status": "full"
            },
            {
                "video_name": "test_video2.mp4",
                "episode_range": [100, 200],
                "objects": [
                    {
                        "object_name": "fork",
                        "rvec": [0.7, 0.8, 0.9],
                        "tvec": [7.0, 8.0, 9.0]
                    }
                ],
                "status": "partial"
            }
        ]
        
        json_path = demos_dir / "object_poses.json"
        with open(json_path, "w") as f:
            json.dump(object_poses_data, f)
        
        # Test loading episode 0
        objects = load_object_poses(tmpdir, 0)
        assert len(objects) == 2
        assert objects[0]["object_name"] == "pink_cup"
        assert objects[0]["tvec"] == [1.0, 2.0, 3.0]
        assert objects[1]["object_name"] == "blue_cup"
        
        # Test loading episode 1
        objects = load_object_poses(tmpdir, 1)
        assert len(objects) == 1
        assert objects[0]["object_name"] == "fork"
        
        # Test loading non-existent episode
        objects = load_object_poses(tmpdir, 999)
        assert len(objects) == 0
        
        print("✓ test_load_object_poses passed")


def test_rvec_to_quaternion():
    """Test rotation vector to quaternion conversion"""
    
    # Test identity rotation (zero rotation vector)
    rvec = [0.0, 0.0, 0.0]
    quat = rvec_to_quaternion(rvec)
    assert quat.shape == (4,)
    # Identity quaternion should be approximately [1, 0, 0, 0]
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    
    # Test 90-degree rotation around Z-axis
    rvec = [0.0, 0.0, np.pi/2]
    quat = rvec_to_quaternion(rvec)
    
    # Quaternion norm should be 1
    assert np.abs(np.linalg.norm(quat) - 1.0) < 1e-6
    
    # Test another rotation
    rvec2 = [0.1, 0.2, 0.3]
    quat2 = rvec_to_quaternion(rvec2)
    assert quat2.shape == (4,)
    assert np.abs(np.linalg.norm(quat2) - 1.0) < 1e-6
    
    print("✓ test_rvec_to_quaternion passed")


def test_object_name_to_usd_mapping():
    """Test that all expected object names have USD mappings"""
    
    # Kitchen objects
    assert "pink_cup" in OBJECT_NAME_TO_USD
    assert "blue_cup" in OBJECT_NAME_TO_USD
    
    # Dining room objects
    assert "fork" in OBJECT_NAME_TO_USD
    assert "knife" in OBJECT_NAME_TO_USD
    assert "plate" in OBJECT_NAME_TO_USD
    
    # Living room objects
    assert "blue_block" in OBJECT_NAME_TO_USD
    assert "green_block" in OBJECT_NAME_TO_USD
    assert "red_block" in OBJECT_NAME_TO_USD
    
    # Verify all mappings point to valid file names (end with .usd)
    for obj_name, usd_file in OBJECT_NAME_TO_USD.items():
        assert usd_file.endswith(".usd"), f"USD file '{usd_file}' for '{obj_name}' should end with .usd"
    
    print("✓ test_object_name_to_usd_mapping passed")


def test_load_object_poses_missing_file():
    """Test loading object poses when file doesn't exist"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # No object_poses.json file created
        objects = load_object_poses(tmpdir, 0)
        assert len(objects) == 0
        
        print("✓ test_load_object_poses_missing_file passed")


def test_load_object_poses_empty_objects():
    """Test loading object poses when episode has no objects"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create directory structure
        demos_dir = tmpdir / "demos" / "mapping"
        demos_dir.mkdir(parents=True)
        
        # Create object_poses.json with empty objects
        object_poses_data = [
            {
                "video_name": "test_video.mp4",
                "episode_range": [0, 100],
                "objects": [],
                "status": "none"
            }
        ]
        
        json_path = demos_dir / "object_poses.json"
        with open(json_path, "w") as f:
            json.dump(object_poses_data, f)
        
        objects = load_object_poses(tmpdir, 0)
        assert len(objects) == 0
        
        print("✓ test_load_object_poses_empty_objects passed")


if __name__ == "__main__":
    print("Running tests for launch_isaacsim_workspace.py...\n")
    
    try:
        test_load_object_poses()
        test_rvec_to_quaternion()
        test_object_name_to_usd_mapping()
        test_load_object_poses_missing_file()
        test_load_object_poses_empty_objects()
        
        print("\n" + "="*50)
        print("All tests passed! ✓")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
