"""
model_processor.py

Model_Processor is a thin coordinator. It holds no format-specific logic
of its own — all of that lives in the concrete strategy classes. Its
responsibilities are:

  1. Use ModelStrategyFactory to select the right strategy for a file
  2. Delegate load() and extract_model_data() to the strategy
  3. Convert DataRecord lists to InferenceRecord lists via the strategy
  4. Expose a stable public interface to main.py

Adding support for a new model format requires only:
  - A new strategy class in strategies/
  - One ModelStrategyFactory.register() call in factory.py
  - No changes to this file
"""

from __future__ import annotations
from typing import Optional

from .factory        import ModelStrategyFactory
from .dataset_loader import load_dataset
from .summariser     import summarise_results
from .types          import ModelMetadata, DataRecord, InferenceRecord


class Model_Processor:

    def __init__(self, file_path: str):
        self.file_path  = file_path
        self._strategy  = ModelStrategyFactory.create(file_path)
        self.model      = self._strategy.load(file_path)
        self.model_data: ModelMetadata = self._strategy.extract_model_data(self.model)


    @property
    def supported_formats(self) -> list[str]:
        return ModelStrategyFactory.supported_extensions()

    def load_dataset(
        self,
        file_path:    str,
        label_column: Optional[str] = None,
    ) -> list[DataRecord]:
        """
        Load a dataset file and return immutable DataRecord objects.
        Delegates to dataset_loader — no dataset logic lives here.
        """
        return load_dataset(file_path, label_column)

    def run_inference(
        self,
        records: list[DataRecord],
    ) -> list[InferenceRecord]:
        """
        Run all records through the model and return InferenceRecord objects.

        Each call to run_inference creates a fresh execution context via
        the strategy's _prepare/_teardown hooks. Concurrent requests each
        call this independently on their own records list and their own
        strategy instance (created fresh by the factory), so there is no
        shared mutable state between concurrent calls.
        """
        return self._strategy.run_inference(self.model, records)

    def summarise(self, results: list[InferenceRecord]) -> dict:
        """
        Compute accuracy statistics from inference results.
        Delegates to summariser — no statistics logic lives here.
        """
        return summarise_results(results)

    def results_to_dicts(self, results: list[InferenceRecord]) -> list[dict]:
        """
        Convert InferenceRecord objects to plain dicts for downstream
        consumers (Vector_Analyzer, layer_analysis, JSON serialisation).

        This is the bridge between the immutable domain model and the
        dict-based interface expected by the rest of the pipeline.
        """
        return [r.to_dict() for r in results]