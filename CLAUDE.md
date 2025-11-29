# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voilab is a visualization toolkit for robotics datasets built on JupyterLab with Voila for interactive applications. The project consists of:

1. **Main voilab package**: Interactive visualization applications for robotics data
2. **UMI package**: Universal Manipulation Interface for SLAM pipelines and dataset processing
3. **Diffusion Policy package**: Dependencies for diffusion policy implementation

## Development Environment

### Package Management
- Uses `uv` for dependency management
- Python version: >=3.10, <3.13
- Virtual environment: `.venv/`

### Common Commands

```bash
# Install dependencies
make install
# or
uv sync

# Install development dependencies
make install-dev
# or
uv sync --extra dev

# Launch JupyterLab environment
make launch-jupyterlab
# or
uv run jupyter lab --ip 0.0.0.0 --port 8888 --no-browser

# Run linter
ruff check

# Run CLI tools
uv run voilab launch-viewer
uv run umi run-slam-pipeline <config_file>
```

## Project Structure

### Key Directories
- `src/voilab/`: Main source code
  - `applications/`: Interactive visualization components
  - `utils/`: Data loading and processing utilities
  - `cli.py`: Command-line interface
- `packages/`: Workspace packages
  - `umi/`: UMI robotics data processing pipeline
  - `diffusion_policy/`: Diffusion policy dependencies
- `nbs/`: Jupyter notebooks for Voila applications
- `umi_pipeline_configs/`: Configuration files for UMI pipelines
- `assets/`: Robot models and assets (e.g., Franka Panda URDF)
- `experiments/`: Experimental code and notebooks

### Core Applications
- **Replay Buffer Viewer** (`nbs/replay_buffer_viewer.ipynb`): Interactive exploration of UMI datasets
- **ArUco Tag Viewer** (`nbs/aruco_detection_viewer.ipynb`): Marker detection and visualization
- **SLAM Viewer** (`nbs/slam_viewer.ipynb`): SLAM mapping visualization
- **Dataset Visualizer** (`nbs/dataset_visualizer.ipynb`): Comprehensive dataset review tool

### Data Processing Pipeline
The UMI pipeline processes robotics data through these stages:
1. Video preprocessing and IMU extraction
2. SLAM mapping with ORB-SLAM3
3. ArUco marker detection
4. Camera calibration
5. Dataset generation and compression

## Development Workflow

### Adding New Applications
1. Create notebook in `nbs/` for interactive development
2. Implement core logic in `src/voilab/applications/`
3. Add utilities in `src/voilab/utils/`
4. Ensure notebooks work with Voila rendering

### Testing
- Tests are located in `packages/*/tests/`
- Use pytest with configuration in package pyproject.toml files
- Test paths include `src/` via pytest configuration

### Dependencies
- Main voilab package includes visualization libraries (ipywidgets, plotly, viser)
- UMI package includes robotics frameworks (robomimic, free-mujoco-py, PyTorch)
- Development dependencies include JupyterLab extensions and testing tools

## Key Technologies
- **JupyterLab + Voila**: Interactive web applications
- **Zarr**: Compressed dataset storage
- **Plotly**: 3D visualization
- **OpenCV**: Computer vision processing
- **PyTorch**: Machine learning framework
- **ORB-SLAM3**: SLAM mapping
- **URDF**: Robot model visualization via jupyterlab-urdf extension