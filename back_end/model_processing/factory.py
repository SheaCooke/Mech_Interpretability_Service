"""
factory.py

ModelStrategyFactory resolves the correct ModelStrategy subclass for a
given file extension. New formats are registered with a single line —
no changes to Model_Processor or any existing strategy are required.

The registry is a class-level dict so it is shared across all instances,
but because it maps to *classes* (not instances), it is effectively
immutable after the initial registrations and therefore thread-safe.
"""

from __future__ import annotations
from .base import ModelStrategy
from .strategies.keras_strategy import KerasStrategy


class ModelStrategyFactory:

    # Maps lowercase file extension to strategy class (not instance).
    _registry: dict[str, type[ModelStrategy]] = {}

    @classmethod
    def register(cls, ext: str, strategy_class: type[ModelStrategy]) -> None:
        """
        Register a strategy class for a file extension.
        """
        cls._registry[ext.lower()] = strategy_class

    @classmethod
    def create(cls, file_path: str) -> ModelStrategy:
        """
        Resolve and instantiate the correct strategy for file_path.
        """
        ext = file_path.rsplit('.', 1)[-1].lower()
        if ext not in cls._registry:
            supported = sorted(cls._registry.keys())
            raise ValueError(
                f"Unsupported model format: .{ext}. "
                f"Supported formats: {supported}"
            )
        return cls._registry[ext]()



ModelStrategyFactory.register('keras', KerasStrategy)
#TODO: add other model formats after implementation