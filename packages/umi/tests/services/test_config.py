#!/usr/bin/env python3
"""
Unit tests for ConfigService

Run these tests independently:
    python -m pytest umi/tests/services/test_config.py -v
"""

import pytest
import tempfile
import json
from pathlib import Path

from umi.services.config import ConfigService


class TestConfigService:
    """Test cases for ConfigService"""
    
    def test_init_with_valid_config(self):
        """Test initialization with valid config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            config_file = tmpdir / "test_config.yaml"
            config_data = {
                "video_organization": {
                    "instance": "umi.services.video_organization.VideoOrganizationService",
                    "config": {"input_patterns": ["*.MP4"]}
                }
            }
            config_file.write_text(json.dumps(config_data))
            
            service = ConfigService(str(config_file))
            assert service.config_path == config_file
    
    def test_init_with_missing_config(self):
        """Test initialization with missing config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            config_file = tmpdir / "missing_config.yaml"
            
            with pytest.raises(FileNotFoundError):
                ConfigService(str(config_file))
    
    def test_get_service_config(self):
        """Test getting service-specific config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            config_file = tmpdir / "test_config.yaml"
            config_data = {
                "video_organization": {
                    "instance": "test.service.VideoService",
                    "config": {"input_patterns": ["*.MP4"], "test_param": True}
                }
            }
            config_file.write_text(json.dumps(config_data))
            
            service = ConfigService(str(config_file))
            config = service.get_service_config("video_organization")
            
            assert config == {"input_patterns": ["*.MP4"], "test_param": True}
    
    def test_get_service_instance(self):
        """Test getting service instance path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            config_file = tmpdir / "test_config.yaml"
            config_data = {
                "imu_extraction": {
                    "instance": "umi.services.imu_extraction.IMUExtractionService",
                    "config": {"num_workers": 4}
                }
            }
            config_file.write_text(json.dumps(config_data))
            
            service = ConfigService(str(config_file))
            instance = service.get_service_instance("imu_extraction")
            
            assert instance == "umi.services.imu_extraction.IMUExtractionService"
    
    def test_get_full_config(self):
        """Test getting complete configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            config_file = tmpdir / "test_config.yaml"
            config_data = {
                "service1": {
                    "instance": "test.service1",
                    "config": {"param1": "value1"}
                },
                "service2": {
                    "instance": "test.service2",
                    "config": {"param2": "value2"}
                }
            }
            config_file.write_text(json.dumps(config_data))
            
            service = ConfigService(str(config_file))
            full_config = service.get_full_config()
            
            assert full_config == config_data
    
    def test_get_nonexistent_service_config(self):
        """Test getting config for non-existent service"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            config_file = tmpdir / "test_config.yaml"
            config_data = {"existing_service": {"config": {"test": True}}}
            config_file.write_text(json.dumps(config_data))
            
            service = ConfigService(str(config_file))
            config = service.get_service_config("nonexistent_service")
            
            assert config == {}
    
    def test_get_nonexistent_service_instance(self):
        """Test getting instance for non-existent service"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            config_file = tmpdir / "test_config.yaml"
            config_data = {"existing_service": {"instance": "test.service"}}
            config_file.write_text(json.dumps(config_data))
            
            service = ConfigService(str(config_file))
            instance = service.get_service_instance("nonexistent_service")
            
            assert instance == ""