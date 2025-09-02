#!/usr/bin/env python3
"""
Unit tests for DatasetPlanningService

Run these tests independently:
    python -m pytest umi/tests/services/test_dataset_planning.py -v
"""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path

from umi.services.dataset_planning import DatasetPlanningService


class TestDatasetPlanningService:
    """Test cases for DatasetPlanningService"""

    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {
            "tcp_offset": [0.1, 0.2, 0.3],
            "nominal_z": 0.5,
            "min_episode_length": 15,
        }
        service = DatasetPlanningService(config)
        assert np.array_equal(service.tcp_offset, [0.1, 0.2, 0.3])
        assert service.nominal_z == 0.5
        assert service.min_episode_length == 15

    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = DatasetPlanningService({})
        assert np.array_equal(service.tcp_offset, [0.0, 0.0, 0.0])
        assert service.nominal_z == 0.0
        assert service.min_episode_length == 10

    def test_generate_plan_single_episode(self):
        """Test dataset plan generation with single episode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock dataset plan
            dataset_plan = {
                "episodes": [
                    {
                        "demo_name": "demo1",
                        "frame_count": 100,
                        "duration": 3.3,
                        "metadata": {},
                    }
                ]
            }

            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))

            # Create supporting files
            (input_dir / "slam_tag_calibration.json").write_text(json.dumps({"tags": {}}))
            (input_dir / "gripper_range_calibration.json").write_text(json.dumps({"range": {}}))
            (input_dir / "demo1_trajectory.txt").write_text("trajectory data")
            (input_dir / "demo1_aruco.json").write_text(json.dumps({"detections": []}))

            output_dir = tmpdir / "output"

            service = DatasetPlanningService(
                {
                    "tcp_offset": [0.0, 0.0, 0.0],
                    "nominal_z": 0.0,
                    "min_episode_length": 10,
                }
            )
            result = service.generate_plan(str(input_dir), str(output_dir))

            assert result["total_episodes"] == 1
            assert result["total_frames"] == 100
            assert "plan_file" in result
            assert result["plan"] is not None

    def test_generate_plan_multiple_episodes(self):
        """Test dataset plan generation with multiple episodes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock dataset plan
            dataset_plan = {
                "episodes": [
                    {"demo_name": "demo1", "frame_count": 100, "duration": 3.3},
                    {"demo_name": "demo2", "frame_count": 200, "duration": 6.6},
                    {"demo_name": "demo3", "frame_count": 50, "duration": 1.65},
                ]
            }

            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))

            # Create supporting files
            for demo in ["demo1", "demo2", "demo3"]:
                (input_dir / f"{demo}_trajectory.txt").write_text("trajectory")
                (input_dir / f"{demo}_aruco.json").write_text(json.dumps({"detections": []}))

            output_dir = tmpdir / "output"

            service = DatasetPlanningService({"min_episode_length": 10})
            result = service.generate_plan(str(input_dir), str(output_dir))

            assert result["total_episodes"] == 3
            assert result["total_frames"] == 350  # 100+200+50

    def test_generate_plan_filter_short_episodes(self):
        """Test filtering episodes shorter than minimum length"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock dataset plan with mixed lengths
            dataset_plan = {
                "episodes": [
                    {"demo_name": "long_demo", "frame_count": 100, "duration": 3.3},
                    {"demo_name": "short_demo", "frame_count": 5, "duration": 0.165},
                ]
            }

            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))

            # Create supporting files
            for demo in ["long_demo", "short_demo"]:
                (input_dir / f"{demo}_trajectory.txt").write_text("trajectory")
                (input_dir / f"{demo}_aruco.json").write_text(json.dumps({"detections": []}))

            output_dir = tmpdir / "output"

            service = DatasetPlanningService({"min_episode_length": 10})
            result = service.generate_plan(str(input_dir), str(output_dir))

            assert result["total_episodes"] == 1  # Only long_demo should pass
            assert result["episodes"][0]["demo_name"] == "long_demo"

    def test_load_calibrations(self):
        """Test loading calibration data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock calibration files
            slam_data = {"tags": {"tag1": {"x": 0, "y": 0, "z": 0}}}
            gripper_data = {"range": {"x": [-0.5, 0.5]}}

            (input_dir / "slam_tag_calibration.json").write_text(json.dumps(slam_data))
            (input_dir / "gripper_range_calibration.json").write_text(json.dumps(gripper_data))

            service = DatasetPlanningService({})
            calibrations = service._load_calibrations(input_dir)

            assert "slam_tag" in calibrations
            assert "gripper_range" in calibrations
            assert calibrations["slam_tag"] == slam_data
            assert calibrations["gripper_range"] == gripper_data

    def test_load_trajectories(self):
        """Test loading trajectory files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock trajectory files
            (input_dir / "demo1_trajectory.txt").write_text("trajectory1")
            (input_dir / "demo2_trajectory.txt").write_text("trajectory2")
            (input_dir / "demo3_trajectory.txt").write_text("trajectory3")

            service = DatasetPlanningService({})
            trajectories = service._load_trajectories(input_dir)

            assert len(trajectories) == 3
            assert "demo1" in trajectories
            assert "demo2" in trajectories
            assert "demo3" in trajectories

    def test_load_aruco_detections(self):
        """Test loading ArUco detection files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock detection files
            for demo in ["demo1", "demo2"]:
                (input_dir / f"{demo}_aruco.json").write_text(json.dumps({"detections": [{"frame": 0, "markers": []}]}))

            service = DatasetPlanningService({})
            detections = service._load_aruco_detections(input_dir)

            assert len(detections) == 2
            assert "demo1" in detections
            assert "demo2" in detections

    def test_validate_plan_success(self):
        """Test successful validation of dataset plan"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            output_dir = tmpdir / "output"
            output_dir.mkdir()

            plan_data = {"episodes": [{"demo_name": "test"}]}
            (output_dir / "dataset_plan.json").write_text(json.dumps(plan_data))

            service = DatasetPlanningService({})
            assert service.validate_plan(str(output_dir)) is True

    def test_validate_plan_failure(self):
        """Test validation failure cases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Missing file
            empty_dir = tmpdir / "empty"
            empty_dir.mkdir()

            service = DatasetPlanningService({})
            assert service.validate_plan(str(empty_dir)) is False

            # Empty plan
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            (output_dir / "dataset_plan.json").write_text(json.dumps({"episodes": []}))

            assert service.validate_plan(str(output_dir)) is False

    def test_create_dataset_plan_structure(self):
        """Test dataset plan structure creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock data
            calibrations = {"slam_tag": {"tags": {}}, "gripper_range": {"range": {}}}
            trajectories = {"demo1": str(input_dir / "demo1_trajectory.txt")}
            detections = {"demo1": {"total_frames": 100, "detections": []}}

            service = DatasetPlanningService({"tcp_offset": [0.1, 0.2, 0.3], "nominal_z": 0.5})

            plan = service._create_dataset_plan(calibrations, trajectories, detections)

            assert "episodes" in plan
            assert "total_episodes" in plan
            assert "total_duration" in plan
            assert "config" in plan

            if len(plan["episodes"]) > 0:
                episode = plan["episodes"][0]
                assert "demo_name" in episode
                assert "frame_count" in episode
                assert "duration" in episode
                assert "metadata" in episode
                assert "tcp_offset" in episode["metadata"]

    def test_episode_sorting_by_duration(self):
        """Test episodes are sorted by duration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create mock dataset plan
            dataset_plan = {
                "episodes": [
                    {"demo_name": "short", "frame_count": 50, "duration": 1.0},
                    {"demo_name": "long", "frame_count": 100, "duration": 3.0},
                    {"demo_name": "medium", "frame_count": 75, "duration": 2.0},
                ]
            }

            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))

            # Create supporting files
            for demo in ["short", "long", "medium"]:
                (input_dir / f"{demo}_trajectory.txt").write_text("trajectory")
                (input_dir / f"{demo}_aruco.json").write_text(json.dumps({"detections": []}))

            output_dir = tmpdir / "output"

            service = DatasetPlanningService({"min_episode_length": 10})
            result = service.generate_plan(str(input_dir), str(output_dir))

            # Check episodes are sorted by duration (descending)
            durations = [ep["duration"] for ep in result["plan"]["episodes"]]
            assert durations == sorted(durations, reverse=True)

    def test_empty_input_handling(self):
        """Test handling empty input data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create empty dataset plan
            dataset_plan = {"episodes": []}
            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))

            output_dir = tmpdir / "output"

            service = DatasetPlanningService({})
            result = service.generate_plan(str(input_dir), str(output_dir))

            assert result["total_episodes"] == 0
            assert result["total_frames"] == 0

    def test_config_serialization(self):
        """Test config serialization in dataset plan"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            dataset_plan = {"episodes": []}
            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))

            output_dir = tmpdir / "output"

            config = {
                "tcp_offset": [0.1, 0.2, 0.3],
                "nominal_z": 0.5,
                "min_episode_length": 15,
            }

            service = DatasetPlanningService(config)
            result = service.generate_plan(str(input_dir), str(output_dir))

            config_in_plan = result["plan"]["config"]
            assert config_in_plan["tcp_offset"] == [0.1, 0.2, 0.3]
            assert config_in_plan["nominal_z"] == 0.5
            assert config_in_plan["min_episode_length"] == 15


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
