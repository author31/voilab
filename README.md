# Voilab

A lightweight visualization toolkit for exploring robotics datasets, built on a pre-configured JupyterLab environment with Voila for interactive applications.

## Overview

Voilab provides a set of tools to interactively view and debug robotics data. The primary workflow is through a custom JupyterLab environment that includes built-in extensions for launching web applications and viewing URDF models directly from the UI.

-----

## Packages

This repository contains several packages. For more detailed information on each, please refer to their respective README files:

  - [`packages/umi`](https://www.google.com/search?q=./packages/umi/README.md): Tools and configurations for running SLAM pipelines with UMI datasets.

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

### 3. Viewing Robot Models (Built-in URDF Viewer)

The JupyterLab environment comes with a built-in viewer for Universal Robot Description Format (URDF) files.

**Usage**:

1.  Use the file browser to locate a `.urdf` file.
2.  Double-click the file to open it in a new tab with an interactive 3D viewer.

An example model for the Franka Emika Panda robot is provided in `assets/franka_panda`. You can test the viewer by opening `assets/franka_panda/franka_panda.urdf`.

-----

## How to Contribute

Follow the established pattern when adding new applications:

1.  **Notebook interface**: Create `.ipynb` files in `nbs/` for interactive development. Ensure they can be rendered correctly with Voila.
2.  **Core logic**: Implement visualization components in `src/voilab/applications/`.
3.  **Utilities**: Add reusable data loading/processing in `src/voilab/utils/`.
4.  **CLI integration**: (Optional) Register new commands in the voilab CLI following existing patterns.

Use `uv sync` to manage dependencies and test changes.
