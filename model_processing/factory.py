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


class ModelStrategyFactory:

    # Maps lowercase file extension → strategy class (not instance).
    # Storing classes rather than instances means each call to create()
    # produces a fresh strategy, avoiding shared mutable state between
    # concurrent requests.
    _registry: dict[str, type[ModelStrategy]] = {}

    @classmethod
    def register(cls, ext: str, strategy_class: type[ModelStrategy]) -> None:
        """
        Register a strategy class for a file extension.

        Called at module import time for built-in formats. Can also be
        called at runtime to add support for new formats without modifying
        this file — the Open/Closed principle applied to format support.

        Args:
            ext:            lowercase file extension without the dot (e.g. 'keras')
            strategy_class: a concrete subclass of ModelStrategy
        """
        cls._registry[ext.lower()] = strategy_class

    @classmethod
    def create(cls, file_path: str) -> ModelStrategy:
        """
        Resolve and instantiate the correct strategy for file_path.

        Returns a fresh strategy instance — each call to create() produces
        an independent object so concurrent requests cannot share state.

        Raises:
            ValueError: if the file extension is not registered
        """
        ext = file_path.rsplit('.', 1)[-1].lower()
        if ext not in cls._registry:
            supported = sorted(cls._registry.keys())
            raise ValueError(
                f"Unsupported model format: .{ext}. "
                f"Supported formats: {supported}"
            )
        return cls._registry[ext]()

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return a sorted list of all registered extensions."""
        return sorted(cls._registry.keys())


# ── Built-in registrations ────────────────────────────────────────────────────
# Import strategies here so the registrations happen once at module load time.
# Adding a new format requires only two lines: the import and the register call.

from .strategies.keras_strategy import KerasStrategy

ModelStrategyFactory.register('keras', KerasStrategy)

# Future formats — uncomment as strategies are implemented:
# from .strategies.onnx_strategy    import OnnxStrategy
# from .strategies.pytorch_strategy import PyTorchStrategy
# ModelStrategyFactory.register('onnx', OnnxStrategy)
# ModelStrategyFactory.register('pt',   PyTorchStrategy)
# ModelStrategyFactory.register('pth',  PyTorchStrategy)