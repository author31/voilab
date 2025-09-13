# UMI (Universal Manipulation Interface)

UMI is a comprehensive robotics data processing pipeline for SLAM mapping, ArUco detection, calibration, and dataset generation.
voilab has been refactored to use YAML-based configuration, making debugging and parameter tuning easier across different generations of GoPro cameras. While the official implementation only supports GoPro 9, voilab extends compatibility up to GoPro 13.
## Quick Start

### Data collection
1. Please follow the instructions from the official documentation at [here](https://swanky-sphere-ad1.notion.site/UMI-Data-Collection-Tutorial-4db1a1f0f2aa4a2e84d9742720428b4c?pvs=4)
2. Create a new directory, then add a subdirectory named /raw_videos. Copy all recorded videos into $new_directory/raw_videos.
3. Update the session_dir argument in the YAML configuration file under the 00_process_video stage.
4. (Optional) Run the calibration 
4. Verify and, if necessary, adjust the video_resolution argument in the YAML configuration file.

### Command
```bash
umi run-slam-pipeline umi_pipeline_configs/gopro13_wide_angle_pipeline_config.yaml
```

## Pipeline Configuration

### Configuration Structure
Pipeline configurations are located in `umi_pipeline_configs/`:

- **gopro13_wide_angle_pipeline_config.yaml** - Complete pipeline for GoPro 13 wide angle processing

### Pipeline Stages
The configuration includes these sequential processing stages:

1. **00_process_video**: Video organization and preprocessing
2. **01_extract_gopro_imu**: IMU data extraction from GoPro metadata
3. **02_create_map**: Initial SLAM map creation
4. **03_batch_slam**: Batch SLAM processing
5. **04_detect_aruco**: ArUco marker detection
6. **05_run_calibrations**: Camera and system calibration
7. **06_generate_dataset_plan**: Dataset planning generation
8. **07_generate_replay_buffer**: Final replay buffer creation

### Key Configuration Options
- **SLAM Mapping**: Uses ORB-SLAM3 Docker container
- **Camera Intrinsics**: Configured for GoPro 13 2.7K resolution
- **Aruco Detection**: Custom marker configuration
- **Output**: Compressed zarr datasets for robot learning

## Package Details

For complete package information and dependencies, see:
- `packages/umi/pyproject.toml`

## Integration with Voilab

The processed datasets from UMI pipelines can be visualized using the Voilab replay buffer viewer:
- Use `voilab launch-viewer` to explore generated `.zarr.zip` files
- Visualize RGB streams, robot poses, and demonstration data