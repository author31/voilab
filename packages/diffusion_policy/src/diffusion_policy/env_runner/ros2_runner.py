from typing import Dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class ROS2Runner(BaseImageRunner):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy) -> Dict:
        raise NotImplementedError()
