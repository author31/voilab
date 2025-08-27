from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path
import logging


class BaseService(ABC):
    """Base service class for all UMI services."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the service with configuration.
        
        Args:
            config: Configuration dictionary for the service
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the service's main functionality.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of the service execution
        """
        pass
        
    def _ensure_output_dir(self, output_dir: str) -> Path:
        """Ensure output directory exists.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Path: Path object for output directory
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
        
    def _get_num_workers(self, default_factor: float = 1.0) -> int:
        """Get number of worker threads based on CPU count.
        
        Args:
            default_factor: Factor to multiply CPU count by
            
        Returns:
            int: Number of worker threads
        """
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        return max(1, int(cpu_count * default_factor))
        
    def validate_output(self, output_dir: str) -> bool:
        """Validate service output.
        
        Args:
            output_dir: Directory containing service outputs
            
        Returns:
            bool: True if output is valid
        """
        return True  # Override in subclasses