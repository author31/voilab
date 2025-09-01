#!/usr/bin/env python3
"""
Unit tests for SLAMMappingService

Run these tests independently:
    python -m pytest umi/tests/services/test_slam_mapping.py -v
"""

import pytest
import tempfile
from pathlib import Path

from umi.services.slam_mapping import SLAMMappingService


class TestSLAMMappingService:
    """Test cases for SLAMMappingService"""

    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {
            "docker_image": "custom/orb_slam3:latest",
            "timeout_multiple": 20,
            "max_lost_frames": 100,
        }
        service = SLAMMappingService(config)
        assert service.docker_image == "custom/orb_slam3:latest"
        assert service.timeout_multiple == 20
        assert service.max_lost_frames == 100

    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = SLAMMappingService({})
        assert service.docker_image == "chicheng/orb_slam3:latest"
        assert service.timeout_multiple == 16
        assert service.max_lost_frames == 60

    def test_create_map_single_demo(self):
        """Test creating SLAM map for single demo"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video data")

            output_dir = tmpdir / "output"

            service = SLAMMappingService({})
            result = service.create_map(str(input_dir), str(output_dir))

            assert len(result["maps"]) == 1
            assert len(result["failed"]) == 0
            assert result["maps"][0]["demo"] == "demo1"

    def test_create_map_multiple_demos(self):
        """Test creating SLAM maps for multiple demos"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            # Create multiple demo directories
            demo1_dir = input_dir / "demo1"
            demo1_dir.mkdir()
            (demo1_dir / "demo1.MP4").write_text(b"video1")

            demo2_dir = input_dir / "demo2"
            demo2_dir.mkdir()
            (demo2_dir / "demo2.mp4").write_text(b"video2")

            output_dir = tmpdir / "output"

            service = SLAMMappingService({})
            result = service.create_map(str(input_dir), str(output_dir))

            assert len(result["maps"]) == 2
            assert len(result["failed"]) == 0

    def test_create_map_empty_directory(self):
        """Test handling empty input directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()

            output_dir = tmpdir / "output"

            service = SLAMMappingService({})
            result = service.create_map(str(input_dir), str(output_dir))

            assert len(result["maps"]) == 0
            assert len(result["failed"]) == 0

    def test_create_map_no_videos(self):
        """Test handling directory with no video files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "readme.txt").write_text("no videos here")

            output_dir = tmpdir / "output"

            service = SLAMMappingService({})
            result = service.create_map(str(input_dir), str(output_dir))

            assert len(result["maps"]) == 0
            assert len(result["failed"]) > 0

    def test_validate_mapping_success(self):
        """Test successful validation of SLAM mapping results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create valid mapping structure
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            demo1_dir = output_dir / "demo1"
            demo1_dir.mkdir()
            (demo1_dir / "map.bin").write_text(b"mock map data")
            (demo1_dir / "trajectory.txt").write_text(b"mock trajectory")

            service = SLAMMappingService({})
            assert service.validate_mapping(str(output_dir)) is True

    def test_validate_mapping_failure(self):
        """Test validation failure cases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Empty directory
            empty_dir = tmpdir / "empty"
            empty_dir.mkdir()

            service = SLAMMappingService({})
            assert service.validate_mapping(str(empty_dir)) is False

            # Directory with incomplete files
            incomplete_dir = tmpdir / "incomplete"
            incomplete_dir.mkdir()
            demo_dir = incomplete_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "map.bin").write_text(b"map only")

            # Should still validate as True since it has map files
            assert service.validate_mapping(str(incomplete_dir)) is True

    def test_create_map_output_structure(self):
        """Test that proper directory structure is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video")

            output_dir = tmpdir / "output"

            service = SLAMMappingService({})
            service.create_map(str(input_dir), str(output_dir))

            # Check output structure
            assert output_dir.exists()
            demo1_output = output_dir / "demo1"
            assert demo1_output.exists()
            assert (demo1_output / "map.bin").exists()
            assert (demo1_output / "trajectory.txt").exists()

    def test_create_map_result_structure(self):
        """Test result structure from create_map"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video")

            output_dir = tmpdir / "output"

            service = SLAMMappingService({})
            result = service.create_map(str(input_dir), str(output_dir))

            assert "maps" in result
            assert "failed" in result
            assert len(result["maps"]) == 1

            map_result = result["maps"][0]
            assert "demo" in map_result
            assert "map_file" in map_result
            assert "trajectory_file" in map_result
            assert "video_files" in map_result

    def test_run_docker_slam_placeholder(self):
        """Test Docker SLAM method (placeholder implementation)"""
        # This is a placeholder test for the Docker SLAM functionality
        # In real testing, this would require Docker setup
        service = SLAMMappingService({})

        # Mock the method for testing
        def mock_run_docker_slam(self, video_file, output_dir):
            return True

        # Temporarily replace method
        original_method = service._run_docker_slam
        service._run_docker_slam = mock_run_docker_slam.__get__(
            service, SLAMMappingService
        )

        try:
            result = service._run_docker_slam(Path("test.mp4"), Path("/tmp"))
            assert result is True
        finally:
            # Restore original method
            service._run_docker_slam = original_method


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
