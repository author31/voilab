# Voilab

A lightweight visualization toolkit for exploring robotics datasets, built on a pre-configured JupyterLab environment with Voila for interactive applications.

## Overview

Voilab provides a set of tools to interactively view and debug robotics data. The primary workflow is through a custom JupyterLab environment that includes built-in extensions for launching web applications and viewing URDF models directly from the UI.

-----

## Documentation

This repository contains several packages. For more detailed information on each, please refer to their respective documentation files:

  - [`packages/umi`](./docs/UMI_README.md): Tools and configurations for running SLAM pipelines with UMI datasets.
  - [`packages/diffusion_policy`](./docs/diffusion_policy.md): To train the diffusion policy with UMI datasets.
  - [`diffusion_policy_layers`](./docs/diffusion_policy_layers.md): Overview of the diffusion policy package layers.
  - [`ros2_integration`](./docs/ros2_integration.md): Overview of the ROS2 integration.
  - [`Isaac Sim docker setup`](./docs/DOCKER.md): How to setup the Docker containers for Isaac Sim.


-----

## Getting Started

### Installation

Voilab uses `uv` for dependency management. You can install everything needed with:

```bash
# Install uv (if not already installed) and project dependencies
make install

# Or manually:
uv sync
```

## Core Workflow: JupyterLab Environment

The main functionalities of Voilab are accessed through a customized JupyterLab instance, which includes pre-installed extensions for visualization.

### 1. Launch the Environment

Start the JupyterLab server using the following command:

```bash
make launch-jupyterlab
```

This will open a JupyterLab interface in your web browser.

### 2. Launching Interactive Applications (Voila)

The interactive visualization tools are built as Jupyter notebooks that can be run as standalone web applications using Voila.

**Usage**:

1.  In the JupyterLab file browser (left panel), navigate to the `nbs/` directory.
2.  Right-click on an application notebook (e.g., `replay_buffer_viewer.ipynb`).
3.  Select **"Open with -\> voila"** from the context menu. This will open the application in a new browser tab.

#### Example: Replay Buffer Viewer

  - **Location**: `nbs/replay_buffer_viewer.ipynb`
  - **Goal**: An interactive tool for exploring UMI-style datasets for debugging, validation, and quick data analysis.
  - **Features**:
      - Interactive slider to navigate through time-series data.
      - Visualizes RGB camera streams.
      - Displays robot end-effector positions, orientations, and gripper states.
      - Supports both `.zarr.zip` and `.zarr` datasets.
  ![Replay Buffer Viewer](./media/replay_buffer_viewer.gif)

#### Example: ArUco Tag Viewer

  - **Location**: `nbs/aruco_detection_viewer.ipynb`
  - **Goal**: An interactive tool to detect and visualize ArUco markers in the camera data from a dataset. This is useful for validating marker detection quality, camera calibration, and pose estimation.
  - **Features**:
      - Interactive slider to navigate through time-series data.
      - Performs ArUco marker detection on each image frame.
  ![ArUco Tag Viewer](./media/aruco_tag_viewer.png)

#### Example: Dataset Visualizer

  - **Location**: `nbs/dataset_visualizer.ipynb`
  - **Goal**: A visualization tool for data collectors to review and refine their collected human demonstrations. Helps identify issues such as lost SLAM frames, low ArUco detection rates, and trajectory anomalies before final processing.
  - **Features**:
      - **Pipeline Status**: Overview of which UMI processing stages have completed
      - **Demo Quality Metrics**: Detection rates, lost frames, and trajectory quality for each demo
      - **Trajectory & Video**: Side-by-side 3D camera trajectory visualization and frame-by-frame video preview
      - **ArUco Tags**: ArUco marker detection viewer with detection statistics and marker overlays
  - **CLI Usage**:
    ```bash
    # Launch the dataset visualizer web app
    uv run voilab launch-dataset-visualizer
    ```
  - **Python Usage**:
    ```python
    from voilab.applications.dataset_visualizer import show
    show("/path/to/session/directory")
    ```

### 3. Viewing Robot Models (Built-in URDF Viewer)

The JupyterLab environment comes with a built-in viewer for Universal Robot Description Format (URDF) files.

**Usage**:

1.  Use the file browser to locate a `.urdf` file.
2.  Double-click the file to open it in a new tab with an interactive 3D viewer.

An example model for the Franka Emika Panda robot is provided in `assets/franka_panda`. You can test the viewer by opening `assets/franka_panda/franka_panda.urdf`.

![URDF Viewer](./media/urdf_viewer.png)

-----

## How to Contribute

Follow the established pattern when adding new applications:

1.  **Notebook interface**: Create `.ipynb` files in `nbs/` for interactive development. Ensure they can be rendered correctly with Voila.
2.  **Core logic**: Implement visualization components in `src/voilab/applications/`.
3.  **Utilities**: Add reusable data loading/processing in `src/voilab/utils/`.
4.  **CLI integration**: (Optional) Register new commands in the voilab CLI following existing patterns.

Use `uv sync` to manage dependencies and test changes.
