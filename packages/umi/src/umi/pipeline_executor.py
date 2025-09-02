import copy
import importlib
import logging
from pathlib import Path
from typing import Any, Dict
from loguru import logger

import yaml

from .services.base_service import BaseService


class PipelineExecutor:
    """Pipeline executor for UMI services."""

    def __init__(self, config_path: str):
        """Initialize the pipeline executor.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: dict = {}
        self.services: dict = {}

        self._load_config()

    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries.

        Args:
            base_config: Base configuration to merge into
            override_config: Configuration to override base values

        Returns:
            Merged configuration dictionary
        """
        result = copy.deepcopy(base_config)

        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    def _filter_config(self, config: Dict[str, Any], exclude_keys: list) -> Dict[str, Any]:
        """Filter out specified keys from configuration.

        Args:
            config: Configuration to filter
            exclude_keys: List of keys to exclude

        Returns:
            Filtered configuration dictionary
        """
        result = copy.deepcopy(config)
        for key in exclude_keys:
            if key in result:
                del result[key]
        return result

    def _log_config_diff(
        self,
        stage_name: str,
        inherited_config: Dict[str, Any],
        stage_config: Dict[str, Any],
        final_config: Dict[str, Any],
    ) -> None:
        """Log configuration differences for a stage.

        Args:
            stage_name: Name of the stage
            inherited_config: Configuration inherited from previous stages
            stage_config: Local stage configuration
            final_config: Final merged configuration
        """
        logger.info(f"Configuration for stage '{stage_name}':")

        if not inherited_config:
            logger.info("  No inherited configuration (first stage or inherit_config=false)")
        else:
            inherited_keys = set(inherited_config.keys())
            local_keys = set(stage_config.keys())
            overridden_keys = inherited_keys.intersection(local_keys)

            if overridden_keys:
                logger.info(f"  Overridden keys: {list(overridden_keys)}")

            new_keys = local_keys - inherited_keys
            if new_keys:
                logger.info(f"  New keys: {list(new_keys)}")

            logger.info(
                f"Inherited {len(inherited_config)} keys, added {len(new_keys)} keys, overrode {len(overridden_keys)} keys"
            )

    def _load_config(self) -> None:
        """Load and parse the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

    def _import_class(self, class_path: str) -> type:
        """Dynamically import a class from a string path.

        Args:
            class_path: String path like 'umi.services.video_organization.VideoOrganizationService'

        Returns:
            The imported class

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import class {class_path}: {e}")

    def load_service(self, stage_name: str, propagated_config: Dict[str, Any] = None) -> BaseService:
        """Load and instantiate a service for a given stage.

        Args:
            stage_name: Name of the pipeline stage
            propagated_config: Configuration inherited from previous stages

        Returns:
            Instantiated service

        Raises:
            KeyError: If stage_name not found in configuration
            ImportError: If service class cannot be imported
        """
        if stage_name not in self.config:
            raise KeyError(f"Stage '{stage_name}' not found in configuration")

        stage_config = self.config[stage_name]

        if "instance" not in stage_config:
            raise KeyError(f"Missing 'instance' field for stage '{stage_name}'")

        class_path = stage_config["instance"]

        # Handle configuration inheritance
        inherit_config = stage_config.get("inherit_config", True)
        local_config = stage_config.get("config", {})
        config_override = stage_config.get("config_override", {})
        config_exclude = stage_config.get("config_exclude", [])

        if inherit_config and propagated_config:
            # Start with propagated config, apply overrides and exclusions
            effective_config = self._filter_config(propagated_config, config_exclude)
            effective_config = self._merge_configs(effective_config, local_config)
            effective_config = self._merge_configs(effective_config, config_override)
        else:
            # Use only local configuration
            effective_config = self._merge_configs(local_config, config_override)

        # Log configuration details
        self._log_config_diff(stage_name, propagated_config or {}, local_config, effective_config)

        try:
            service_class = self._import_class(class_path)
            service_instance = service_class(effective_config)

            if not isinstance(service_instance, BaseService):
                raise TypeError(f"Service {class_path} does not inherit from BaseService")

            self.services[stage_name] = service_instance
            logger.info(f"Loaded service for stage '{stage_name}': {class_path}")
            return service_instance

        except Exception as e:
            logger.error(f"Failed to load service for stage '{stage_name}': {e}")
            raise

    def get_service(self, stage_name: str) -> BaseService | None:
        """Get a loaded service instance for a stage.

        Args:
            stage_name: Name of the pipeline stage

        Returns:
            Service instance if loaded, None otherwise
        """
        return self.services.get(stage_name)

    def list_stages(self) -> list:
        """List all available pipeline stages.

        Returns:
            List of stage names
        """
        return list(self.config.keys())

    def execute_stage(self, stage_name: str, propagated_config: Dict[str, Any] = None, *args, **kwargs):
        """Execute a specific pipeline stage.

        Args:
            stage_name: Name of the stage to execute
            propagated_config: Configuration inherited from previous stages
            *args: Positional arguments to pass to execute()
            **kwargs: Keyword arguments to pass to execute()

        Returns:
            Result from service execution

        Raises:
            KeyError: If stage_name not found
            Exception: Any exception from service execution
        """
        # Handle legacy calls without propagated_config
        if propagated_config is None and args and isinstance(args[0], dict):
            propagated_config = args[0]
            args = args[1:]
        elif propagated_config is None:
            propagated_config = {}

        if stage_name not in self.services:
            self.load_service(stage_name, propagated_config)

        service = self.services[stage_name]
        logger.info(f"Executing stage: {stage_name}")

        try:
            result = service.execute(*args, **kwargs)
            logger.info(f"Completed stage: {stage_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to execute stage '{stage_name}': {e}")
            raise

    def execute_all(self, *args, **kwargs) -> dict:
        """Execute all pipeline stages in sequence with configuration propagation.

        Args:
            *args: Positional arguments to pass to each service
            **kwargs: Keyword arguments to pass to each service

        Returns:
            Dictionary mapping stage names to their execution results
        """
        results = {}
        stages = self.list_stages()
        propagated_config = {}

        logger.info(f"Starting pipeline execution with {len(stages)} stages")
        logger.info("Configuration propagation enabled - configs from previous stages will be passed forward")

        for i, stage_name in enumerate(stages, 1):
            stage_config = self.config[stage_name]
            inherit_config = stage_config.get("inherit_config", True)

            logger.info(f"Stage {i}/{len(stages)}: {stage_name} (inherit_config: {inherit_config})")

            try:
                # Load service with propagated configuration
                service_instance = self.load_service(stage_name, propagated_config if inherit_config else {})

                # Get the effective configuration used for this stage
                stage_effective_config = service_instance.config

                # Execute the stage
                result = service_instance.execute(*args, **kwargs)
                results[stage_name] = result

                logger.info(f"Stage {i}/{len(stages)}: {stage_name} completed")

                # Update propagated configuration for next stages
                if inherit_config:
                    propagated_config = self._merge_configs(propagated_config, stage_effective_config)
                    logger.info(f"Updated propagated configuration with {len(stage_effective_config)} keys")
                else:
                    logger.info("Skipped configuration propagation for this stage")

            except Exception as e:
                logger.error(f"Pipeline failed at stage '{stage_name}': {e}")
                raise

        logger.info("Pipeline execution completed successfully")
        return results

    def reload_config(self) -> None:
        """Reload the configuration file."""
        self.services.clear()
        self._load_config()
        logger.info("Configuration reloaded")

    def validate_stages(self) -> dict:
        """Validate all stages by attempting to load their services.

        Returns:
            Dictionary mapping stage names to validation results (True/False)
        """
        validation_results = {}
        propagated_config = {}

        for stage_name in self.list_stages():
            try:
                stage_config = self.config[stage_name]
                inherit_config = stage_config.get("inherit_config", True)

                self.load_service(stage_name, propagated_config if inherit_config else {})
                validation_results[stage_name] = True

                # Update propagated configuration for validation
                if inherit_config and stage_name in self.services:
                    stage_effective_config = self.services[stage_name].config
                    propagated_config = self._merge_configs(propagated_config, stage_effective_config)

            except Exception as e:
                validation_results[stage_name] = False
                logger.warning(f"Validation failed for stage '{stage_name}': {e}")

        return validation_results
