import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from .isaac_sim_config import TASK_CONFIGURATIONS, ISAAC_SIM_CONFIG


class IsaacSimLauncher:
    def __init__(self, task: str, use_docker: bool = True, enable_gpu: bool = True):
        if task not in TASK_CONFIGURATIONS:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(TASK_CONFIGURATIONS.keys())}")

        self.task = task
        self.use_docker = use_docker
        self.enable_gpu = enable_gpu
        self.config = TASK_CONFIGURATIONS[task].copy()

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

    def launch_in_docker(self) -> None:
        """Launch Isaac Sim in Docker container"""
        env_vars = self.setup_environment()
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
            "-e", f"SCENE_CONFIG={env_vars['SCENE_CONFIG']}",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            "-v", f"{workspace_path}:/workspace/voilab",
            "-v", "/usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro",
            "--network", "host",
            "--ipc", "host",
            "--rm", "-it",
            "nvcr.io/nvidia/isaac-sim:5.0.0",
            "python3", "/workspace/voilab/src/voilab/applications/isaac_sim_runner.py",
            "--task", self.task
        ]

        print(f"Launching Isaac Sim in Docker container for task: {self.task}")
        print(f"Container command: {' '.join(docker_cmd)}")

        try:
            subprocess.run(docker_cmd, env=env_vars, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error launching Docker container: {e}")
            raise

    def launch_locally(self) -> None:
        """Launch Isaac Sim locally"""
        env_vars = self.setup_environment()

        # Set environment variables for Isaac Sim
        for key, value in env_vars.items():
            os.environ[key] = value

        print(f"Launching Isaac Sim locally for task: {self.task}")
        print(f"USD scene: {self.config['usd_path']}")
        print(f"Camera position: {self.config['camera_position']}")
        print(f"Camera target: {self.config['camera_target']}")

        try:
            # Import Isaac Sim modules (must happen after environment setup)
            from omni.isaac.kit import SimulationApp
            import omni.usd
            from omni.isaac.core.utils.viewports import set_camera_view

            # Initialize simulation app with task-specific config
            sim_config = ISAAC_SIM_CONFIG.copy()
            simulation_app = SimulationApp(sim_config)

            # Load USD scene
            usd_path = Path.cwd() / self.config["usd_path"]
            if not usd_path.exists():
                raise FileNotFoundError(f"USD file not found: {usd_path}")

            print(f"Loading USD scene: {usd_path}")
            omni.usd.get_context().open_stage(str(usd_path))

            # Perform update steps to ensure the stage is fully loaded
            print("Initializing scene...")
            for i in range(10):
                simulation_app.update()
                if i % 5 == 0:
                    print(f"Update step {i+1}/10")

            # Set the camera view
            print("Setting camera view...")
            set_camera_view(
                eye=self.config["camera_position"],
                target=self.config["camera_target"]
            )

            print(f"Isaac Sim ready for task: {self.task}")
            print("Close the window to exit...")

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
            print(f"Error launching Isaac Sim locally: {e}")
            raise

    def update_resolution(self, width: int, height: int) -> None:
        """Update the resolution configuration"""
        self.config["width"] = width
        self.config["height"] = height

    def launch(self) -> None:
        """Main launch method"""
        if self.use_docker:
            self.launch_in_docker()
        else:
            self.launch_locally()