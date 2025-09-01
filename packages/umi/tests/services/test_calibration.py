#!/usr/bin/env python3
"""
Unit tests for CalibrationService

Run these tests independently:
    python -m pytest umi/tests/services/test_calibration.py -v
"""

import pytest
import tempfile
import json
from pathlib import Path

from umi.services.calibration import CalibrationService


class TestCalibrationService:
    """Test cases for CalibrationService"""

    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {"slam_tag_calibration_timeout": 600, "gripper_range_timeout": 600}
        service = CalibrationService(config)
        assert service.slam_tag_timeout == 600
        assert service.gripper_range_timeout == 600

    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = CalibrationService({})
        assert service.slam_tag_timeout == 300
        assert service.gripper_range_timeout == 300

    def test_run_calibrations_complete(self):
        """Test running complete calibration suite"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock calibration files
            (input_dir / "slam_tag_calibration.json").write_text(
                json.dumps(
                    {
                        "tag_positions": {"tag1": {"x": 0, "y": 0, "z": 0}},
                        "world_to_camera_transforms": {
                            "demo1": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
                        },
                    }
                )
            )

            (input_dir / "gripper_range_calibration.json").write_text(
                json.dumps(
                    {
                        "workspace_bounds": {
                            "x": [-0.5, 0.5],
                            "y": [-0.5, 0.5],
                            "z": [0, 1],
                        },
                        "min_position": [-0.5, -0.5, 0],
                        "max_position": [0.5, 0.5, 1],
                    }
                )
            )

            output_dir = tmpdir / "output"

            service = CalibrationService(
                {"slam_tag_calibration_timeout": 10, "gripper_range_timeout": 10}
            )
            result = service.run_calibrations(str(input_dir), str(output_dir))

            assert "slam_tag_calibration" in result
            assert "gripper_range_calibration" in result
            assert len(result["errors"]) == 0
            assert result["slam_tag_calibration"]["success"] is True
            assert result["gripper_range_calibration"]["success"] is True

    def test_calibrate_slam_tag_with_data(self):
        """Test SLAM tag calibration with existing data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock ArUco detection files
            aruco_file = input_dir / "demo1_aruco.json"
            aruco_data = {
                "detections": [
                    {
                        "frame": 0,
                        "markers": [
                            {
                                "id": 1,
                                "corners": [[0, 0], [100, 0], [100, 100], [0, 100]],
                            }
                        ],
                    }
                ]
            }
            aruco_file.write_text(json.dumps(aruco_data))

            output_dir = tmpdir / "output"

            service = CalibrationService({"slam_tag_calibration_timeout": 10})
            result = service._calibrate_slam_tag(input_dir, output_dir)

            assert result["success"] is True
            assert "calibration_file" in result
            assert result["num_tags_calibrated"] > 0

    def test_calibrate_slam_tag_no_data(self):
        """Test SLAM tag calibration with no data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            output_dir = tmpdir / "output"

            service = CalibrationService({"slam_tag_calibration_timeout": 10})

            with pytest.raises(ValueError, match="No ArUco detection files found"):
                service._calibrate_slam_tag(input_dir, output_dir)

    def test_calibrate_gripper_range_with_data(self):
        """Test gripper range calibration with existing data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock trajectory files
            traj_file = input_dir / "trajectory.txt"
            traj_file.write_text("# SLAM trajectory data\n1 2 3 0 0 0 1\n4 5 6 0 0 0 1")

            output_dir = tmpdir / "output"

            service = CalibrationService({"gripper_range_timeout": 10})
            result = service._calibrate_gripper_range(input_dir, output_dir)

            assert result["success"] is True
            assert "calibration_file" in result
            assert "workspace_volume" in result

    def test_calibrate_gripper_range_no_data(self):
        """Test gripper range calibration with no data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            output_dir = tmpdir / "output"

            service = CalibrationService({"gripper_range_timeout": 10})

            with pytest.raises(ValueError, match="No trajectory files found"):
                service._calibrate_gripper_range(input_dir, output_dir)

    def test_run_calibrations_partial_failure(self):
        """Test calibration handling partial failures"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Only create SLAM tag calibration file
            (input_dir / "slam_tag_calibration.json").write_text(
                json.dumps({"tag_positions": {}})
            )

            output_dir = tmpdir / "output"

            service = CalibrationService(
                {"slam_tag_calibration_timeout": 10, "gripper_range_timeout": 10}
            )
            result = service.run_calibrations(str(input_dir), str(output_dir))

            # Should have errors but still return results
            assert len(result["errors"]) > 0
            assert (
                result["slam_tag_calibration"] is not None
                or result["gripper_range_calibration"] is not None
            )

    def test_validate_calibrations_success(self):
        """Test successful validation of calibration results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            output_dir = tmpdir / "output"
            output_dir.mkdir()

            (output_dir / "slam_tag_calibration.json").write_text(
                json.dumps({"test": True})
            )
            (output_dir / "gripper_range_calibration.json").write_text(
                json.dumps({"test": True})
            )

            service = CalibrationService({})
            assert service.validate_calibrations(str(output_dir)) is True

    def test_validate_calibrations_failure(self):
        """Test validation failure cases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Missing one file
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            (output_dir / "slam_tag_calibration.json").write_text(
                json.dumps({"test": True})
            )

            service = CalibrationService({})
            assert service.validate_calibrations(str(output_dir)) is False

            # Empty directory
            empty_dir = tmpdir / "empty"
            empty_dir.mkdir()

            assert service.validate_calibrations(str(empty_dir)) is False

    def test_calibration_file_structure(self):
        """Test that calibration files are created with proper structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock files
            (input_dir / "slam_tag_calibration.json").write_text(
                json.dumps({"tag_positions": {"tag1": {"x": 0, "y": 0, "z": 0}}})
            )

            output_dir = tmpdir / "output"

            service = CalibrationService({"slam_tag_calibration_timeout": 10})
            service.run_calibrations(str(input_dir), str(output_dir))

            # Check files are created
            assert (output_dir / "slam_tag_calibration.json").exists()
            assert (output_dir / "gripper_range_calibration.json").exists()

    def test_slam_tag_calibration_file_content(self):
        """Test SLAM tag calibration file content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock ArUco detection
            aruco_file = input_dir / "demo1_aruco.json"
            aruco_data = {"detections": [{"frame": 0, "markers": [{"id": 1}]}]}
            aruco_file.write_text(json.dumps(aruco_data))

            output_dir = tmpdir / "output"

            service = CalibrationService({"slam_tag_calibration_timeout": 10})
            result = service._calibrate_slam_tag(input_dir, output_dir)

            # Check file content
            calib_file = Path(result["calibration_file"])
            assert calib_file.exists()

            with open(calib_file, "r") as f:
                calib_data = json.load(f)

            assert "calibration_type" in calib_data
            assert calib_data["calibration_type"] == "slam_tag"
            assert "tag_positions" in calib_data

    def test_gripper_range_calibration_file_content(self):
        """Test gripper range calibration file content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock trajectory
            traj_file = input_dir / "trajectory.txt"
            traj_file.write_text("# SLAM trajectory\n1 2 3 0 0 0 1")

            output_dir = tmpdir / "output"

            service = CalibrationService({"gripper_range_timeout": 10})
            result = service._calibrate_gripper_range(input_dir, output_dir)

            # Check file content
            calib_file = Path(result["calibration_file"])
            assert calib_file.exists()

            with open(calib_file, "r") as f:
                calib_data = json.load(f)

            assert "calibration_type" in calib_data
            assert calib_data["calibration_type"] == "gripper_range"
            assert "workspace_bounds" in calib_data
            assert "min_position" in calib_data
            assert "max_position" in calib_data

    def test_timeout_configurations(self):
        """Test timeout configuration values"""
        service = CalibrationService(
            {"slam_tag_calibration_timeout": 123, "gripper_range_timeout": 456}
        )

        assert service.slam_tag_timeout == 123
        assert service.gripper_range_timeout == 456


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
