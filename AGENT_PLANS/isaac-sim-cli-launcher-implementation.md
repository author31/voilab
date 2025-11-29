# Isaac Sim CLI Launcher Implementation Plan

## Executive Summary

This plan details the implementation of a new CLI command to launch Isaac Sim GUI programmatically via Python API for three task environments: kitchen, dining-table, and living-room. The implementation will leverage the existing Docker infrastructure, USD assets, and CLI patterns in the Voilab codebase.

## Current State Analysis

### Existing Infrastructure
- **Docker Setup**: Complete Isaac Sim container with `nvcr.io/nvidia/isaac-sim:5.0.0`
- **USD Assets**: Franka Panda robot scenes, physics configs, sensor configurations
- **CLI Framework**: Click-based command structure with existing commands
- **Task Directories**: Organized data for kitchen, dining-table, living-room environments

### Missing Components
- Task-specific camera coordinates and action graphs
- Python API integration for Isaac Sim
- Task-specific USD scene configurations
- CLI command for Isaac Sim launcher

## Implementation Approach

### Phase 1: Core Infrastructure Setup

#### 1.1 Create Task Registry Structure
**Directory**: `src/voilab/simulator/registry/`

**File**: `src/voilab/simulator/registry/__init__.py`
```python
from .kitchen_registry import KitchenTaskRegistry
from .dining_table_registry import DiningTableTaskRegistry
from .living_room_registry import LivingRoomTaskRegistry

# Registry mapping
TASK_REGISTRIES = {
    "kitchen": KitchenTaskRegistry,
    "dining-table": DiningTableTaskRegistry,
    "living-room": LivingRoomTaskRegistry,
}

def get_task_registry(task_name: str):
    """Get task registry by name"""
    if task_name not in TASK_REGISTRIES:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_REGISTRIES.keys())}")
    return TASK_REGISTRIES[task_name]
```

**File**: `src/voilab/simulator/registry/kitchen_registry.py`
```python
import numpy as np
from typing import Dict, Any

class KitchenTaskRegistry:
    """Registry for kitchen task configuration"""

    TASK_NAME = "kitchen"
    USD_PATH = "assets/franka_panda/franka_panda_umi-isaacsim/franka_panda_umi-isaacsim.usd"
    CAMERA_POSITION = np.array([2.5, 1.8, 2.0])
    CAMERA_TARGET = np.array([0.0, 0.0, 0.8])

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "usd_path": cls.USD_PATH,
            "camera_position": cls.CAMERA_POSITION,
            "camera_target": cls.CAMERA_TARGET,
            "environment_vars": {
                "TASK_NAME": cls.TASK_NAME,
                "SCENE_CONFIG": "kitchen_scene",
                "SCENE_USD": cls.USD_PATH
            }
        }

    @classmethod
    def validate_environment(cls) -> bool:
        """Validate kitchen environment setup"""
        from pathlib import Path
        return Path(cls.USD_PATH).exists()
```

**File**: `src/voilab/simulator/registry/dining_table_registry.py`
```python
import numpy as np
from typing import Dict, Any

class DiningTableTaskRegistry:
    """Registry for dining table task configuration"""

    TASK_NAME = "dining-table"
    USD_PATH = "assets/Collected_franka-umi-scene/franka-umi-scene.usd"
    CAMERA_POSITION = np.array([1.8, 2.2, 1.8])
    CAMERA_TARGET = np.array([0.0, 0.0, 0.7])

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "usd_path": cls.USD_PATH,
            "camera_position": cls.CAMERA_POSITION,
            "camera_target": cls.CAMERA_TARGET,
            "environment_vars": {
                "TASK_NAME": cls.TASK_NAME,
                "SCENE_CONFIG": "dining_scene",
                "SCENE_USD": cls.USD_PATH
            }
        }

    @classmethod
    def validate_environment(cls) -> bool:
        """Validate dining table environment setup"""
        from pathlib import Path
        return Path(cls.USD_PATH).exists()
```

**File**: `src/voilab/simulator/registry/living_room_registry.py`
```python
import numpy as np
from typing import Dict, Any

class LivingRoomTaskRegistry:
    """Registry for living room task configuration"""

    TASK_NAME = "living-room"
    USD_PATH = "assets/Collected_franka-umi-scene/franka-umi-scene.usd"
    CAMERA_POSITION = np.array([3.2, 1.2, 1.9])
    CAMERA_TARGET = np.array([0.0, 0.5, 0.8])

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "usd_path": cls.USD_PATH,
            "camera_position": cls.CAMERA_POSITION,
            "camera_target": cls.CAMERA_TARGET,
            "environment_vars": {
                "TASK_NAME": cls.TASK_NAME,
                "SCENE_CONFIG": "living_scene",
                "SCENE_USD": cls.USD_PATH
            }
        }

    @classmethod
    def validate_environment(cls) -> bool:
        """Validate living room environment setup"""
        from pathlib import Path
        return Path(cls.USD_PATH).exists()
```

#### 1.2 Create Isaac Sim Action Graph Component

**File**: `src/voilab/simulator/action_graph.py`

See detailed specification: `/home/hcis-s17/author_workdir/voilab/AGENT_PLANS/isaac-sim-action-graph-component.md`

#### 1.3 Create Generic Isaac Sim Launcher Script
**File**: `src/voilab/simulator/launch.py`

```python
#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from voilab.simulator.registry import get_task_registry

# Isaac Sim runtime configuration
ISAAC_SIM_CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "renderer": "RayTracedLighting",
    "physics_engine": "PhysX"
}

class IsaacSimLauncher:
    def __init__(self, task_name: str, width: int = 1280, height: int = 720):
        self.task_name = task_name
        self.task_registry = get_task_registry(task_name)
        self.config = self.task_registry.get_config()
        self.width = width
        self.height = height

        # Update config with resolution
        self.config["width"] = width
        self.config["height"] = height

    def setup_environment(self) -> Dict[str, str]:
        """Setup environment variables for Isaac Sim"""
        env_vars = os.environ.copy()
        env_vars.update(self.config["environment_vars"])
        env_vars.update({
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "DISPLAY": os.getenv("DISPLAY", ":1"),
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "NVIDIA_DRIVER_CAPABILITIES": "all,graphics,display,utility,compute",
            "XAUTHORITY": os.getenv("XAUTHORITY", f"{os.getenv('HOME', '')}/.Xauthority"),
            "ROS_LOCALHOST_ONLY": "0",
            "ROS_DOMAIN_ID": "0"
        })
        return env_vars

    def launch_isaac_sim(self) -> None:
        """Launch Isaac Sim locally using Python API"""
        env_vars = self.setup_environment()

        # Set environment variables for Isaac Sim
        for key, value in env_vars.items():
            os.environ[key] = value

        print(f"Launching Isaac Sim for task: {self.task_name}")
        print(f"USD scene: {self.config['usd_path']}")
        print(f"Camera position: {self.config['camera_position']}")
        print(f"Camera target: {self.config['camera_target']}")
        print(f"Resolution: {self.width}x{self.height}")

        try:
            # Import Isaac Sim modules (must happen after environment setup)
            from omni.isaac.kit import SimulationApp
            import omni.usd
            from omni.isaac.core.utils.viewports import set_camera_view

            # Initialize simulation app with config
            sim_config = ISAAC_SIM_CONFIG.copy()
            sim_config["width"] = self.width
            sim_config["height"] = self.height
            simulation_app = SimulationApp(sim_config)

            # Load USD scene
            usd_path = Path.cwd() / self.config["usd_path"]
            if not usd_path.exists():
                raise FileNotFoundError(f"USD file not found: {usd_path}")

            print(f"Loading USD scene: {usd_path}")
            omni.usd.get_context().open_stage(str(usd_path))

            # Perform update steps to ensure stage is fully loaded
            print("Initializing scene...")
            for i in range(10):
                simulation_app.update()
                if i % 5 == 0:
                    print(f"Update step {i+1}/10")

            # Set camera view
            print("Setting camera view...")
            set_camera_view(
                eye=self.config["camera_position"],
                target=self.config["camera_target"]
            )

            print(f"Isaac Sim ready for task: {self.task_name}")
            print("Close window to exit...")

            # Main simulation loop
            while simulation_app.is_running():
                simulation_app.update()

            # Cleanup on exit
            print("Shutting down Isaac Sim...")
            simulation_app.close()

        except ImportError as e:
            print(f"Failed to import Isaac Sim modules: {e}")
            print("Make sure Isaac Sim is properly installed and environment is set up.")
            raise
        except Exception as e:
            print(f"Error launching Isaac Sim: {e}")
            raise

    def validate_environment(self) -> bool:
        """Validate task environment setup"""
        return self.task_registry.validate_environment()

def main():
    parser = argparse.ArgumentParser(description="Isaac Sim launcher script")
    parser.add_argument("--task", required=True,
                       choices=["kitchen", "dining-table", "living-room"],
                       help="Task environment to load")
    parser.add_argument("--width", type=int, default=1280,
                       help="Window width")
    parser.add_argument("--height", type=int, default=720,
                       help="Window height")
    args = parser.parse_args()

    try:
        launcher = IsaacSimLauncher(
            task_name=args.task,
            width=args.width,
            height=args.height
        )

        # Validate environment
        if not launcher.validate_environment():
            print(f"Error: Task environment '{args.task}' validation failed")
            sys.exit(1)

        # Launch Isaac Sim
        launcher.launch_isaac_sim()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Phase 2: CLI Integration

#### 2.1 Extend CLI Command Structure
**File**: `src/voilab/cli.py` (modify existing)

```python
import click
from pathlib import Path
import subprocess
import os

# ... existing imports ...

@cli.command()
@click.option('--task', type=click.Choice(['kitchen', 'dining-table', 'living-room']),
              required=True, help='Task environment to load')
@click.option('--width', default=1280, help='Window width')
@click.option('--height', default=720, help='Window height')
def launch_simulator(task, width, height):
    """Launch Isaac Sim simulation with specified robot and task environment."""

    try:
        click.echo(f"Preparing Isaac Sim for task: {task}")
        click.echo(f"Resolution: {width}x{height}")

        # Prepare environment variables for Docker
        env_vars = os.environ.copy()
        env_vars.update({
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "DISPLAY": os.getenv("DISPLAY", ":1"),
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "NVIDIA_DRIVER_CAPABILITIES": "all,graphics,display,utility,compute",
            "XAUTHORITY": os.getenv("XAUTHORITY", f"{os.getenv('HOME', '')}/.Xauthority"),
            "ROS_LOCALHOST_ONLY": "0",
            "ROS_DOMAIN_ID": "0",
            "TASK_NAME": task,
            "WINDOW_WIDTH": str(width),
            "WINDOW_HEIGHT": str(height)
        })

        # Build Docker command to use Isaac Sim's built-in Python launcher
        workspace_path = Path.cwd()
        docker_cmd = [
            "docker", "run", "--runtime=nvidia", "--gpus", "all",
            "-e", "ACCEPT_EULA=Y",
            "-e", "PRIVACY_CONSENT=Y",
            "-e", f"DISPLAY={env_vars['DISPLAY']}",
            "-e", f"VK_ICD_FILENAMES={env_vars['VK_ICD_FILENAMES']}",
            "-e", f"NVIDIA_DRIVER_CAPABILITIES={env_vars['NVIDIA_DRIVER_CAPABILITIES']}",
            "-e", f"XAUTHORITY={env_vars['XAUTHORITY']}",
            "-e", f"ROS_LOCALHOST_ONLY={env_vars['ROS_LOCALHOST_ONLY']}",
            "-e", f"ROS_DOMAIN_ID={env_vars['ROS_DOMAIN_ID']}",
            "-e", f"TASK_NAME={env_vars['TASK_NAME']}",
            "-e", f"WINDOW_WIDTH={env_vars['WINDOW_WIDTH']}",
            "-e", f"WINDOW_HEIGHT={env_vars['WINDOW_HEIGHT']}",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            "-v", f"{workspace_path}:/workspace/voilab",
            "-v", "/usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro",
            "--network", "host",
            "--ipc", "host",
            "--rm", "-it",
            "nvcr.io/nvidia/isaac-sim:5.0.0",
            "/isaac-sim/python.sh",  # Use Isaac Sim's built-in Python launcher
            str(workspace_path / "src/voilab/simulator/launch.py"),  # Our script path
            "--task", task,
            "--width", str(width),
            "--height", str(height)
        ]

        click.echo(f"Launching Isaac Sim in Docker container...")
        click.echo(f"Task: {task}")
        click.echo(f"Resolution: {width}x{height}")

        # Launch Docker container
        subprocess.run(docker_cmd, env=env_vars, check=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"Error launching Docker container: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error launching Isaac Sim: {str(e)}", err=True)
        sys.exit(1)

# Fix existing broken command
@cli.command()
@click.argument("task")
def launch_dataset_visualizer_task(task):
    """Launch dataset visualizer for a specific task."""
    argv = ["--no-browser", "nbs/dataset_visualizer.ipynb"]
    Voila.launch_instance(argv=argv)
```

### Phase 3: Docker and Configuration Management

#### 3.1 Update Dependencies
**File**: `pyproject.toml` (add to dependencies)

```toml
[project.dependencies]
# ... existing dependencies ...
"docker>=6.0.0",
"numpy>=1.24",
```

#### 3.2 Create Isaac Sim Docker Compose Extension
**File**: `docker-compose.isaacsim.yaml`

```yaml
services:
  isaac-sim-gui:
    build:
      context: .
      dockerfile: Dockerfile.isaacsim
    volumes:
      - .:/workspace/voilab
      - /tmp/.X11-unix:/tmp/.X11-unix
      - /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro
    stdin_open: true
    tty: true
    environment:
      - ACCEPT_EULA=Y
      - PRIVACY_CONSENT=Y
      - DISPLAY=${DISPLAY:-:1}
      - VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
      - NVIDIA_DRIVER_CAPABILITIES=all,graphics,display,utility,compute
      - XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
      - ROS_LOCALHOST_ONLY=0
      - ROS_DOMAIN_ID=0
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    runtime: nvidia
    network_mode: host
    command: >
      bash -c "source /opt/ros/humble/setup.bash &&
               python3 /workspace/voilab/src/voilab/applications/isaac_sim_runner.py"
```

### Phase 4: Testing and Validation

#### 4.1 Create Test Scripts
**File**: `tests/test_isaac_sim_launcher.py`

```python
import pytest
from unittest.mock import Mock, patch
from voilab.applications.isaac_sim_launcher import IsaacSimLauncher
from voilab.applications.isaac_sim_config import TASK_CONFIGURATIONS

def test_launcher_initialization():
    """Test Isaac Sim launcher initialization"""
    launcher = IsaacSimLauncher(task="kitchen", use_docker=True)
    assert launcher.task == "kitchen"
    assert launcher.use_docker == True
    assert launcher.config == TASK_CONFIGURATIONS["kitchen"]

def test_environment_setup():
    """Test environment variable setup"""
    launcher = IsaacSimLauncher(task="dining-table")
    env_vars = launcher.setup_environment()

    assert env_vars["ACCEPT_EULA"] == "Y"
    assert env_vars["PRIVACY_CONSENT"] == "Y"
    assert "TASK_NAME" in env_vars
    assert env_vars["TASK_NAME"] == "dining_table"

@patch('subprocess.run')
def test_docker_launch(mock_subprocess):
    """Test Docker launch functionality"""
    launcher = IsaacSimLauncher(task="kitchen")
    launcher.launch_in_docker()

    mock_subprocess.assert_called_once()
    args, kwargs = mock_subprocess.call_args
    assert "docker" in args[0]
    assert "nvcr.io/nvidia/isaac-sim:5.0.0" in args[0]

def test_invalid_task():
    """Test error handling for invalid task"""
    with pytest.raises(ValueError):
        IsaacSimLauncher(task="invalid_task")
```

## Critical Files for Implementation

### Primary Files to Modify/Create:
1. **src/voilab/cli.py** - Add new CLI command
2. **src/voilab/applications/isaac_sim_config.py** - Task configurations
3. **src/voilab/applications/isaac_sim_launcher.py** - Main launcher class
4. **src/voilab/applications/isaac_sim_runner.py** - Docker runner script
5. **pyproject.toml** - Add Docker dependency
6. **docker-compose.isaacsim.yaml** - Docker compose configuration
7. **tests/test_isaac_sim_launcher.py** - Test suite

### Assets to Reference:
1. **assets/franka_panda/franka_panda_umi-isaacsim/** - USD scenes and configurations
2. **assets/Collected_franka-umi-scene/** - Task-specific scenes
3. **Dockerfile.isaacsim** - Existing Isaac Sim Docker setup
4. **launch_isaacsim.sh** - Existing launch script patterns

## Implementation Dependencies

### Required Python Packages:
- `click` - CLI framework (already exists)
- `docker` - Docker SDK for Python (new)
- `numpy` - Array operations (already exists)
- `pathlib` - Path handling (built-in)

### Isaac Sim Dependencies:
- `omni.isaac.kit` - Isaac Sim core (in container)
- `omni.usd` - USD file handling (in container)
- `omni.isaac.core.utils.viewports` - Camera utilities (in container)

## Deployment and Usage

### CLI Commands:
```bash
# Launch kitchen environment in Docker
uv run voilab launch-simulator --task kitchen --docker

# Launch dining table locally
uv run voilab launch-simulator --task dining-table --no-docker

# Launch living room with custom USD
uv run voilab launch-simulator --task living-room --local-usd /path/to/custom.usd

# Launch with custom resolution
uv run voilab launch-simulator --task kitchen --width 1920 --height 1080
```

### Docker Usage:
```bash
# Using Docker Compose
docker-compose -f docker-compose.isaacsim.yaml up isaac-sim-gui

# With custom task
TASK=kitchen docker-compose -f docker-compose.isaacsim.yaml up isaac-sim-gui
```

## Risk Mitigation

### Technical Risks:
1. **Docker X11 Forwarding**: Ensure proper display configuration
2. **GPU Driver Compatibility**: Verify NVIDIA driver versions
3. **USD File Paths**: Validate path resolution in container vs host
4. **Memory Requirements**: Isaac Sim is resource-intensive

### Mitigation Strategies:
1. **Fallback Options**: Provide local launch option if Docker fails
2. **Resource Limits**: Set appropriate memory and GPU limits
3. **Error Handling**: Comprehensive error messages and logging
4. **Documentation**: Clear setup instructions and troubleshooting

## Success Criteria

1. **Functional CLI Command**: Successfully launches Isaac Sim for all three tasks
2. **Docker Integration**: Proper containerization with GPU support
3. **Camera Configuration**: Task-specific camera positions working correctly
4. **USD Loading**: Proper scene loading for different environments
5. **Error Handling**: Graceful failure modes and informative error messages
6. **Test Coverage**: Comprehensive test suite for all components

## Timeline Estimate

- **Phase 1 (Core Infrastructure)**: 2-3 days
- **Phase 2 (CLI Integration)**: 1 day
- **Phase 3 (Docker & Config)**: 1-2 days
- **Phase 4 (Testing & Validation)**: 1-2 days

**Total Estimated Time**: 5-8 days for full implementation and testing.

This plan leverages existing infrastructure while providing a robust, extensible solution for Isaac Sim integration into Voilab CLI toolkit.