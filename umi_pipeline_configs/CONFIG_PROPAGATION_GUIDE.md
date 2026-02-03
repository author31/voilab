# UMI Pipeline Configuration Propagation Guide

This guide explains the enhanced configuration system that allows configurations to be propagated between pipeline stages, reducing duplication and enabling more flexible pipeline design.

## Overview

The enhanced pipeline executor now supports configuration inheritance between stages, allowing later stages to automatically inherit configuration from previous stages. This reduces the need for duplicated configuration values and enables more maintainable pipeline configurations.

## New YAML Syntax Features

### 1. Configuration Inheritance Control

```yaml
stage_name:
  instance: "umi.services.example.ExampleService"
  inherit_config: true  # Default: true
  config:
    # Local configuration values
```

- `inherit_config: true` (default): Inherit configuration from all previous stages
- `inherit_config: false`: Use only local configuration, ignore previous stages

### 2. Configuration Overrides

```yaml
stage_name:
  instance: "umi.services.example.ExampleService"
  config_override:
    key_to_override: "new_value"
    nested:
      key: "overridden_value"
```

- `config_override`: Explicitly override inherited configuration values
- Applied after inheritance and local config merging

### 3. Configuration Exclusions

```yaml
stage_name:
  instance: "umi.services.example.ExampleService"
  config_exclude:
    - "unwanted_key"
    - "another_key"
```

- `config_exclude`: List of keys to exclude from inherited configuration
- Useful for removing configuration that doesn't apply to a specific stage

## Configuration Resolution Order

For each stage, the effective configuration is built in this order:

1. **Inherited configuration** (from previous stages, if `inherit_config: true`)
2. **Filtered by exclusions** (keys in `config_exclude` are removed)
3. **Local configuration** (values in `config` section)
4. **Overrides applied** (values in `config_override` section)

## Example Usage Scenarios

### Scenario 1: Shared Output Directory

```yaml
# First stage sets the base output directory
video_organization:
  instance: "umi.services.video_organization.VideoOrganizationService"
  config:
    output_dir: "./data"

# Later stages inherit and extend
imu_extraction:
  instance: "umi.services.imu_extraction.IMUExtractionService"
  config_override:
    output_dir: "./data/imu"  # Override to subdirectory
```

### Scenario 2: Opting Out of Inheritance

```yaml
# Stage that should not inherit Docker settings
aruco_detection:
  instance: "umi.services.aruco_detection.ArucoDetectionService"
  inherit_config: false  # Don't inherit any previous config
  config:
    num_workers: 4
    specific_setting: "value"
```

### Scenario 3: Selective Inheritance

```yaml
# Stage that needs most config but wants to exclude Docker settings
create_map:
  instance: "umi.services.slam_mapping.SLAMMappingService"
  inherit_config: true
  config_exclude:
    - "docker_image"  # Don't inherit Docker image from previous stages
  config:
    # Only specify what's unique to this stage
    max_lost_frames: 60
```

## Logging and Debugging

The enhanced pipeline executor provides detailed logging:

```
[INFO] Starting pipeline execution with 8 stages
[INFO] Configuration propagation enabled - configs from previous stages will be passed forward
[INFO] Stage 1/8: video_organization (inherit_config: false)
[INFO] Configuration for stage 'video_organization':
[INFO]   No inherited configuration (first stage or inherit_config=false)
[INFO]   Completed stage: video_organization
[INFO] Stage 2/8: imu_extraction (inherit_config: true)
[INFO] Configuration for stage 'imu_extraction':
[INFO]   Inherited 1 keys, added 2 keys, overrode 1 keys
[INFO]   Overridden keys: ['output_dir']
[INFO]   Updated propagated configuration with 3 keys
```

## Migration Guide

### From Basic Configuration

**Before (basic config):**
```yaml
video_organization:
  instance: "umi.services.video_organization.VideoOrganizationService"
  config:
    output_dir: "./data"

imu_extraction:
  instance: "umi.services.imu_extraction.IMUExtractionService"
  config:
    output_dir: "./data"
    num_workers: 4
```

**After (with inheritance):**
```yaml
video_organization:
  instance: "umi.services.video_organization.VideoOrganizationService"
  inherit_config: false  # First stage
  config:
    output_dir: "./data"

imu_extraction:
  instance: "umi.services.imu_extraction.IMUExtractionService"
  # output_dir is automatically inherited! No need to specify
  config:
    num_workers: 4
```

### Backward Compatibility

All existing configurations will continue to work without modification. The new features are opt-in:
- `inherit_config` defaults to `true` for backward compatibility
- `config_override` and `config_exclude` are optional
- Existing configs behave as before

## Best Practices

1. **First Stage**: Set `inherit_config: false` for the first stage to avoid inheriting empty configs
2. **Shared Settings**: Define common settings in early stages for inheritance
3. **Specific Overrides**: Use `config_override` sparingly for clarity
4. **Documentation**: Comment complex inheritance scenarios
5. **Validation**: Use `validate_stages()` to check configuration before execution

## Troubleshooting

### Issue: Configuration not inherited
- Check that `inherit_config: true` is set (or omitted, as it's the default)
- Verify that previous stages don't have `inherit_config: false`
- Look for `config_exclude` that might be filtering out expected keys

### Issue: Values being overridden unexpectedly
- Check `config_override` sections for conflicting keys
- Review the logging output which shows which keys are being overridden

### Issue: Complex nested configurations
- The merging system performs deep merging of dictionaries
- Lists are replaced, not merged - use `config_override` for list values