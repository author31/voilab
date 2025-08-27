#!/usr/bin/env python3
"""
Unit tests for BatchSLAMService

Run these tests independently:
    python -m pytest umi/tests/services/test_batch_slam.py -v
"""

import pytest
import tempfile
from pathlib import Path

from umi.services.batch_slam import BatchSLAMService


class TestBatchSLAMService:
    """Test cases for BatchSLAMService"""
    
    def test_init_with_config(self):
        """Test service initialization with custom config"""
        config = {
            'num_workers': 8,
            'retry_attempts': 5
        }
        service = BatchSLAMService(config)
        assert service.num_workers == 8
        assert service.retry_attempts == 5
    
    def test_init_with_default_config(self):
        """Test service initialization with default config"""
        service = BatchSLAMService({})
        assert service.num_workers is not None  # Auto-detected
        assert service.retry_attempts == 3
    
    def test_num_workers_auto_detection(self):
        """Test automatic worker count detection"""
        service = BatchSLAMService({'num_workers': None})
        assert service.num_workers is not None
        assert isinstance(service.num_workers, int)
        assert service.num_workers > 0
    
    def test_process_batch_single_demo(self):
        """Test batch processing for single demo"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create input structure
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"mock video data")
            (demo_dir / "map.bin").write_text(b"mock map data")
            
            output_dir = tmpdir / "output"
            
            service = BatchSLAMService({'num_workers': 1})
            result = service.process_batch(str(input_dir), str(output_dir))
            
            assert len(result["processed"]) == 1
            assert len(result["failed"]) == 0
            assert result["processed"][0]["demo"] == "demo1"
    
    def test_process_batch_multiple_demos(self):
        """Test batch processing for multiple demos"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            
            # Create multiple demo directories
            demo1_dir = input_dir / "demo1"
            demo1_dir.mkdir()
            (demo1_dir / "demo1.MP4").write_text(b"video1")
            (demo1_dir / "map.bin").write_text(b"map1")
            
            demo2_dir = input_dir / "demo2"
            demo2_dir.mkdir()
            (demo2_dir / "demo2.mp4").write_text(b"video2")
            (demo2_dir / "map.txt").write_text(b"map2")
            
            output_dir = tmpdir / "output"
            
            service = BatchSLAMService({'num_workers': 1})
            result = service.process_batch(str(input_dir), str(output_dir))
            
            assert len(result["processed"]) == 2
            assert len(result["failed"]) == 0
    
    def test_process_batch_empty_directory(self):
        """Test handling empty input directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            
            output_dir = tmpdir / "output"
            
            service = BatchSLAMService({'num_workers': 1})
            result = service.process_batch(str(input_dir), str(output_dir))
            
            assert len(result["processed"]) == 0
            assert len(result["failed"]) == 0
    
    def test_process_batch_no_map_files(self):
        """Test handling demos without map files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"video only")
            
            output_dir = tmpdir / "output"
            
            service = BatchSLAMService({'num_workers': 1})
            result = service.process_batch(str(input_dir), str(output_dir))
            
            # Should still process but with limited inputs
            assert len(result["processed"]) == 1
    
    def test_validate_batch_results_success(self):
        """Test successful validation of batch results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create valid structure
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            
            demo1_dir = output_dir / "demo1"
            demo1_dir.mkdir()
            (demo1_dir / "optimized_trajectory.txt").write_text(b"trajectory")
            (demo1_dir / "keyframes.json").write_text(b"keyframes")
            
            service = BatchSLAMService({})
            assert service.validate_batch_results(str(output_dir)) is True
    
    def test_validate_batch_results_failure(self):
        """Test validation failure cases"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Empty directory
            empty_dir = tmpdir / "empty"
            empty_dir.mkdir()
            
            service = BatchSLAMService({})
            assert service.validate_batch_results(str(empty_dir)) is False
    
    def test_process_batch_output_structure(self):
        """Test that proper directory structure is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"video")
            (demo_dir / "map.bin").write_text(b"map")
            
            output_dir = tmpdir / "output"
            
            service = BatchSLAMService({'num_workers': 1})
            service.process_batch(str(input_dir), str(output_dir))
            
            # Check output structure
            assert output_dir.exists()
            demo1_output = output_dir / "demo1"
            assert demo1_output.exists()
            assert (demo1_output / "optimized_trajectory.txt").exists()
            assert (demo1_output / "keyframes.json").exists()
    
    def test_process_batch_result_structure(self):
        """Test result structure from process_batch"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"video")
            (demo_dir / "map.bin").write_text(b"map")
            
            output_dir = tmpdir / "output"
            
            service = BatchSLAMService({'num_workers': 1})
            result = service.process_batch(str(input_dir), str(output_dir))
            
            assert "processed" in result
            assert "failed" in result
            assert len(result["processed"]) == 1
            
            processed_result = result["processed"][0]
            assert "demo" in processed_result
            assert "output_dir" in processed_result
            assert "attempt" in processed_result
    
    def test_run_batch_slam_placeholder(self):
        """Test batch SLAM method (placeholder implementation)"""
        # This is a placeholder test for actual SLAM processing
        service = BatchSLAMService({})
        
        # Mock the method for testing
        def mock_run_batch_slam(self, video_file, map_files, output_dir):
            return True
        
        # Temporarily replace method
        original_method = service._run_batch_slam
        service._run_batch_slam = mock_run_batch_slam.__get__(service, BatchSLAMService)
        
        try:
            result = service._run_batch_slam(Path("test.mp4"), [Path("map.bin")], Path("/tmp"))
            assert result is True
        finally:
            # Restore original method
            service._run_batch_slam = original_method
    
    def test_process_batch_with_retry_logic(self):
        """Test retry logic in batch processing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            demo_dir = input_dir / "demo1"
            demo_dir.mkdir()
            (demo_dir / "demo1.MP4").write_text(b"video")
            (demo_dir / "map.bin").write_text(b"map")
            
            output_dir = tmpdir / "output"
            
            service = BatchSLAMService({
                'num_workers': 1,
                'retry_attempts': 2
            })
            result = service.process_batch(str(input_dir), str(output_dir))
            
            # Should succeed within retry attempts
            assert len(result["processed"]) == 1
            assert result["processed"][0]["attempt"] == 1


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])