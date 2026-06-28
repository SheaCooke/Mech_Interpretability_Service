
from __future__ import annotations
from .base import ModelStrategy
from .strategies.keras_strategy import KerasStrategy


class ModelStrategyFactory:

    @classmethod
    def create(cls, file_path: str) -> ModelStrategy:
        """
        Resolve and instantiate the correct strategy for the file type
        """
        ext = file_path.rsplit('.', 1)[-1].lower()

        if ext == 'keras':
            return KerasStrategy()
        raise ValueError(f"Model file type is not currently supported.")