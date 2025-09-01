import logging
import json
from typing import Any

from .base_service import BaseService


class LoggingService(BaseService):
    """Structured logging service for pipeline operations."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        output_dir = self.config.get("output_dir", ".")
        log_level = self.config.get("log_level", "INFO")
        self.output_dir = self._ensure_output_dir(output_dir)

        # Configure main logger (use the one from BaseService)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # File handler
        log_file = self.output_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))

        # JSON formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def execute(self, *args, **kwargs) -> Any:
        """Execute logging service (no-op for this service)."""
        return {
            "status": "initialized",
            "log_file": str(self.output_dir / "pipeline.log"),
        }

    def info(self, message: str, **kwargs):
        """Log info message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)

    def error(self, message: str, **kwargs):
        """Log error message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.debug(message)

    def log_stage_start(self, stage_name: str, **kwargs):
        """Log stage start."""
        self.info(f"Starting stage: {stage_name}", stage=stage_name, **kwargs)

    def log_stage_complete(self, stage_name: str, **kwargs):
        """Log stage completion."""
        self.info(f"Completed stage: {stage_name}", stage=stage_name, **kwargs)

    def log_stage_error(self, stage_name: str, error: str, **kwargs):
        """Log stage error."""
        self.error(
            f"Error in stage: {stage_name}", stage=stage_name, error=error, **kwargs
        )
