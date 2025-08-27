#!/usr/bin/env python3
"""
Unit tests for ReplayBufferService

Run these tests independently:
    python -m pytest umi/tests/services/test_replay_buffer.py -v
"""

import pytest
import tempfile
import json
import gzip
from pathlib import Path

from umi.services.replay_buffer import ReplayBufferService


class TestReplayBufferService:
    """Test cases for ReplayBufferService"""
    
    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {
            'output_resolution': [512, 512],
            'output_fov': 120,
            'compression_level': 9,
            'num_workers': 8
        }
        service = ReplayBufferService(config)
        assert service.output_resolution == (512, 512)
        assert service.output_fov == 120
        assert service.compression_level == 9
        assert service.num_workers == 8
    
    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = ReplayBufferService({})
        assert service.output_resolution == (256, 256)
        assert service.output_fov == 90
        assert service.compression_level == 6
        assert service.num_workers == 4
    
    def test_generate_replay_buffer_single_episode(self):
        """Test replay buffer generation for single episode"""
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
                        "metadata": {}
                    }
                ]
            }
            
            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))
            
            # Create mock video file
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video data")
            
            # Create supporting files
            (input_dir / "slam_tag_calibration.json").write_text(json.dumps({"tags": {}}))
            (input_dir / "gripper_range_calibration.json").write_text(json.dumps({"range": {}}))
            (input_dir / "demo1_trajectory.txt").write_text("trajectory")
            (input_dir / "demo1_aruco.json").write_text(json.dumps({"detections": []}))
            
            output_dir = tmpdir / "output"
            
            service = ReplayBufferService({
                'output_resolution': [64, 64],
                'output_fov': 90,
                'compression_level': 1,
                'num_workers': 1
            })
            result = service.generate_replay_buffer(str(input_dir), str(output_dir))
            
            assert result["total_episodes"] == 1
            assert result["summary"]["total_episodes"] == 1
            assert len(result["episodes"]) == 1
    
    def test_generate_replay_buffer_multiple_episodes(self):
        """Test replay buffer generation for multiple episodes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            
            # Create mock dataset plan
            dataset_plan = {
                "episodes": [
                    {"demo_name": "demo1", "frame_count": 100, "duration": 3.3},
                    {"demo_name": "demo2", "frame_count": 200, "duration": 6.6},
                    {"demo_name": "demo3", "frame_count": 50, "duration": 1.65}
                ]
            }
            
            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))
            
            # Create mock video files and supporting files
            for demo in ["demo1", "demo2", "demo3"]:
                demo_dir = input_dir / demo
                demo_dir.mkdir()
                (demo_dir / f"{demo}.MP4").write_text(b"mock video data")
                (input_dir / f"{demo}_trajectory.txt").write_text("trajectory")
                (input_dir / f"{demo}_aruco.json").write_text(json.dumps({"detections": []}))
            
            # Create calibration files
            (input_dir / "slam_tag_calibration.json").write_text(json.dumps({"tags": {}}))
            (input_dir / "gripper_range_calibration.json").write_text(json.dumps({"range": {}}))
            
            output_dir = tmpdir / "output"
            
            service = ReplayBufferService({
                'output_resolution': [64, 64],
                'compression_level': 1,
                'num_workers': 1
            })
            result = service.generate_replay_buffer(str(input_dir), str(output_dir))
            
            assert result["total_episodes"] == 3
            assert result["summary"]["total_episodes"] == 3
            assert len(result["episodes"]) == 3
    
    def test_generate_replay_buffer_empty_dataset(self):
        """Test handling empty dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            
            # Create empty dataset plan
            dataset_plan = {"episodes": []}
            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))
            
            output_dir = tmpdir / "output"
            
            service = ReplayBufferService({
                'output_resolution': [64, 64],
                'compression_level': 1,
                'num_workers': 1
            })
            result = service.generate_replay_buffer(str(input_dir), str(output_dir))
            
            assert result["total_episodes"] == 0
            assert result["summary"]["total_episodes"] == 0
    
    def test_validate_replay_buffer_success(self):
        """Test successful validation of replay buffer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            
            summary_data = {
                "total_episodes": 2,
                "total_frames": 300,
                "episodes": [{"demo_name": "demo1"}, {"demo_name": "demo2"}]
            }
            (output_dir / "replay_buffer_summary.json").write_text(json.dumps(summary_data))
            
            service = ReplayBufferService({})
            assert service.validate_replay_buffer(str(output_dir)) is True
    
    def test_validate_replay_buffer_failure(self):
        """Test validation failure cases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Missing file
            empty_dir = tmpdir / "empty"
            empty_dir.mkdir()
            
            service = ReplayBufferService({})
            assert service.validate_replay_buffer(str(empty_dir)) is False
            
            # Empty summary
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            (output_dir / "replay_buffer_summary.json").write_text(json.dumps({"total_episodes": 0}))
            
            assert service.validate_replay_buffer(str(output_dir)) is False
    
    def test_num_workers_auto_detection(self):
        """Test automatic worker count detection"""
        service = ReplayBufferService({'num_workers': None})
        assert service.num_workers is not None
        assert isinstance(service.num_workers, int)
        assert service.num_workers > 0
    
    def test_process_episode_output_structure(self):
        """Test episode processing output structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            
            # Create mock episode data
            episode = {
                "demo_name": "demo1",
                "frame_count": 50,
                "duration": 1.65,
                "metadata": {}
            }
            
            # Create mock video
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video data")
            
            output_dir = tmpdir / "output"
            
            service = ReplayBufferService({
                'output_resolution': [64, 64],
                'compression_level': 1,
                'num_workers': 1
            })
            
            result = service._process_episode(episode, input_dir, output_dir)
            
            assert "demo_name" in result
            assert "episode_file" in result
            assert "frame_count" in result
            assert "episode_dir" in result
            assert result["demo_name"] == "demo1"
    
    def test_extract_frames_placeholder(self):
        """Test frame extraction (placeholder implementation)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create mock video file
            video_file = tmpdir / "test.MP4"
            video_file.write_text(b"mock video data")
            
            episode = {
                "demo_name": "test",
                "frame_count": 10,
                "metadata": {}
            }
            
            service = ReplayBufferService({
                'output_resolution': [64, 64]
            })
            
            frames = service._extract_frames(video_file, episode)
            
            assert isinstance(frames, list)
            if len(frames) > 0:
                frame = frames[0]
                assert "frame_idx" in frame
                assert "timestamp" in frame
                assert "image_shape" in frame
                assert "camera_pose" in frame
                assert "action" in frame
    
    def test_create_summary(self):
        """Test summary creation"""
        service = ReplayBufferService({})
        
        mock_results = {
            "episodes": [
                {"demo_name": "demo1", "frame_count": 100},
                {"demo_name": "demo2", "frame_count": 200}
            ]
        }
        
        summary = service._create_summary(mock_results)
        
        assert "total_episodes" in summary
        assert "total_frames" in summary
        assert summary["total_episodes"] == 2
        assert summary["total_frames"] == 300
        assert "resolution" in summary
        assert "compression_level" in summary
        assert len(summary["episodes"]) == 2
    
    def test_generate_replay_buffer_no_dataset_plan(self):
        """Test handling missing dataset plan"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            
            output_dir = tmpdir / "output"
            
            service = ReplayBufferService({})
            
            with pytest.raises(ValueError, match="Dataset plan not found"):
                service.generate_replay_buffer(str(input_dir), str(output_dir))
    
    def test_episode_file_compression(self):
        """Test that episode files are properly compressed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            
            # Create mock dataset plan
            dataset_plan = {
                "episodes": [{"demo_name": "demo1", "frame_count": 10, "duration": 0.33}]
            }
            (input_dir / "dataset_plan.json").write_text(json.dumps(dataset_plan))
            
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video data")
            
            # Create supporting files
            (input_dir / "slam_tag_calibration.json").write_text(json.dumps({"tags": {}}))
            (input_dir / "gripper_range_calibration.json").write_text(json.dumps({"range": {}}))
            (input_dir / "demo1_trajectory.txt").write_text("trajectory")
            (input_dir / "demo1_aruco.json").write_text(json.dumps({"detections": []}))
            
            output_dir = tmpdir / "output"
            
            service = ReplayBufferService({
                'compression_level': 9,
                'num_workers': 1
            })
            
            result = service.generate_replay_buffer(str(input_dir), str(output_dir))
            
            # Check compressed file exists
            episode_file = output_dir / "demo1" / "replay_buffer.json.gz"
            assert episode_file.exists()
            assert episode_file.suffix == ".gz"
    
    def test_get_camera_pose_placeholder(self):
        """Test camera pose generation (placeholder)"""
        service = ReplayBufferService({})
        
        pose = service._get_camera_pose(0, {"demo_name": "test"})
        
        assert isinstance(pose, list)
        assert len(pose) == 16  # 4x4 matrix flattened
    
    def test_get_action_placeholder(self):
        """Test action generation (placeholder)"""
        service = ReplayBufferService({})
        
        action = service._get_action(0, {"demo_name": "test"})
        
        assert isinstance(action, dict)
        assert "gripper_position" in action
        assert "gripper_rotation" in action
        assert "gripper_open" in action
        assert isinstance(action["gripper_position"], list)
        assert isinstance(action["gripper_rotation"], list)
        assert isinstance(action["gripper_open"], (int, float))


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])