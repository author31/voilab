import yaml
from pathlib import Path


class ConfigService:
    """Configuration management service using YAML format."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self):
        """Load and parse YAML configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get_service_config(self, service_name: str) -> dict:
        """Get configuration for specific service."""
        service_config = self._config.get(service_name, {})
        return service_config.get("config", {})

    def get_service_instance(self, service_name: str) -> str:
        """Get service instance class path."""
        service_config = self._config.get(service_name, {})
        return service_config.get("instance", "")

    def get_full_config(self) -> dict:
        """Get complete configuration."""
        return self._config.copy()
