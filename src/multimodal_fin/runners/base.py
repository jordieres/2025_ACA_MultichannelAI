"""
Base interface for runner classes in the multimodal_fin CLI.

All runners must inherit from this class and implement the `run()` method.
"""

from abc import ABC, abstractmethod


class Runner(ABC):
    """Abstract base class for all runners used in the CLI."""

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Execute the runner's logic.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Each runner must implement the `run()` method.")
