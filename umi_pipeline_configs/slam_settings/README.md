# SLAM Settings Configuration Files

This folder contains ORB-SLAM3 configuration files extracted from the Docker image `chicheng/orb_slam3:latest`.

## Available Configuration Files

### GoPro 10 Max Lens (Fisheye)
- **gopro10_maxlens_fisheye_setting_v1_720.yaml** - 720p resolution (960x720)
  - Default setting used in `slam_mapping.py`
  - Camera parameters calibrated for GoPro 10 with Max Lens Mod
  - FPS: 60
  
- **gopro10_maxlens_fisheye_setting_v1_480.yaml** - 480p resolution
  - Lower resolution alternative for faster processing

### GoPro 9
- **gopro9_maxlens_fisheye_setting.yaml** - GoPro 9 with Max Lens Mod (fisheye)
- **gopro9_wide_setting.yaml** - GoPro 9 wide angle mode

## Usage

To use a custom SLAM settings file, specify it in your configuration with a path relative to the project root:

```yaml
slam_settings_file: "umi_pipeline_configs/slam_settings/gopro10_maxlens_fisheye_setting_v1_720.yaml"
```

The path should be relative to the project root directory (where `umi_pipeline_configs/` folder is located). The system will automatically mount this file into the Docker container.

## Configuration Parameters

Key parameters you can adjust:

- **Camera Intrinsics**: `Camera1.fx`, `Camera1.fy`, `Camera1.cx`, `Camera1.cy`
- **Distortion Parameters**: `Camera1.k1`, `Camera1.k2`, `Camera1.k3`, `Camera1.k4`
- **Resolution**: `Camera.width`, `Camera.height`
- **Frame Rate**: `Camera.fps`
- **ORB Features**: `ORBextractor.nFeatures` (number of features per image)
- **IMU Noise Parameters**: `IMU.NoiseGyro`, `IMU.NoiseAcc`, `IMU.GyroWalk`, `IMU.AccWalk`

## Notes

- These files are based on the Kannala-Brandt fisheye camera model
- IMU-to-camera transformation matrix (`IMU.T_b_c1`) is calibrated for specific camera setups
- You may need to recalibrate camera parameters for your specific hardware setup
