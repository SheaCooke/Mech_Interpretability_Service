
from __future__ import annotations
from .base import ModelStrategy
from .strategies.keras_strategy import KerasStrategy


class ModelStrategyFactory:

    @classmethod
    def create(cls, file_path: str) -> ModelStrategy:
        ext = file_path.rsplit('.', 1)[-1].lower()

        if ext == 'keras':
            return KerasStrategy()
        #TODO: extend with other model types
        raise ValueError(f"Model file type is not currently supported.")