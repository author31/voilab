#!/usr/bin/env python3
"""
Unit tests for VideoOrganizationService

Run these tests independently:
    python -m pytest umi/tests/services/test_video_organization.py -v
"""

import pytest
import tempfile
from pathlib import Path

from umi.services.video_organization import VideoOrganizationService


class TestVideoOrganizationService:
    """Test cases for VideoOrganizationService"""

    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {"input_patterns": ["*.MP4", "*.mp4", "*.avi"]}
        service = VideoOrganizationService(config)
        assert service.input_patterns == ["*.MP4", "*.mp4", "*.avi"]

    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = VideoOrganizationService({})
        assert service.input_patterns == ["*.MP4", "*.mp4"]

    def test_organize_videos_single_demo(self):
        """Test organizing videos for single demo"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create session directory with video
            session_dir = tmpdir / "session"
            session_dir.mkdir()
            (session_dir / "demo1_video.MP4").write_bytes(b"mock video data")

            output_dir = tmpdir / "output"

            service = VideoOrganizationService({})
            result = service.organize_videos(str(session_dir), str(output_dir))

            assert result["moved_to_raw_videos"] >= 0
            assert result["organized_demos"] >= 1

    def test_organize_videos_multiple_demos(self):
        """Test organizing videos for multiple demos"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            session_dir = tmpdir / "session"
            session_dir.mkdir()

            # Create multiple demo videos
            (session_dir / "demo1_video.MP4").write_text("mock1")
            (session_dir / "demo2_video.mp4").write_text("mock2")
            (session_dir / "demo3_clip.avi").write_text("mock3")

            output_dir = tmpdir / "output"

            service = VideoOrganizationService({"input_patterns": ["*.MP4", "*.mp4", "*.avi"]})
            result = service.organize_videos(str(session_dir), str(output_dir))

            assert result["total_videos"] == 3
            assert len(result["demos"]) == 3
            assert "demo1" in result["demos"]
            assert "demo2" in result["demos"]
            assert "demo3" in result["demos"]

    def test_organize_videos_empty_directory(self):
        """Test handling empty input directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            session_dir = tmpdir / "session"
            session_dir.mkdir()

            output_dir = tmpdir / "output"

            service = VideoOrganizationService({})
            result = service.organize_videos(str(session_dir), str(output_dir))

            assert result["total_videos"] == 0
            assert len(result["demos"]) == 0

    def test_extract_demo_name_variations(self):
        """Test demo name extraction from different filename patterns"""
        service = VideoOrganizationService({})

        # Test various filename patterns
        assert service._extract_demo_name("demo1_video") == "demo1"
        assert service._extract_demo_name("test_demo.MP4") == "test"
        assert service._extract_demo_name("single_name") == "single_name"
        assert service._extract_demo_name("multi_part_video_name") == "multi"

    def test_validate_organization_success(self):
        """Test successful validation of organized videos"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create valid structure
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            demo1_dir = output_dir / "demo1"
            demo1_dir.mkdir()
            (demo1_dir / "demo1.MP4").write_text("video1")

            demo2_dir = output_dir / "demo2"
            demo2_dir.mkdir()
            (demo2_dir / "demo2.mp4").write_text("video2")

            service = VideoOrganizationService({})
            assert service.validate_organization(str(output_dir)) is True

    def test_validate_organization_failure(self):
        """Test validation failure cases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Empty directory
            empty_dir = tmpdir / "empty"
            empty_dir.mkdir()

            service = VideoOrganizationService({})
            assert service.validate_organization(str(empty_dir)) is False

            # Directory with no videos
            no_videos_dir = tmpdir / "no_videos"
            no_videos_dir.mkdir()
            (no_videos_dir / "demo1").mkdir()

            assert service.validate_organization(str(no_videos_dir)) is False

    def test_organize_videos_no_matching_patterns(self):
        """Test with no matching file patterns"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            session_dir = tmpdir / "session"
            session_dir.mkdir()
            (session_dir / "video.txt").write_text("not a video")

            output_dir = tmpdir / "output"

            service = VideoOrganizationService({"input_patterns": ["*.MP4"]})
            result = service.organize_videos(str(session_dir), str(output_dir))

            assert result["total_videos"] == 0
            assert len(result["demos"]) == 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
