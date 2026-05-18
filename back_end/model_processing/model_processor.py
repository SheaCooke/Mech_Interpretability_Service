"""
Thin coordinator class
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

    def load_dataset(
        self,
        file_bytes: bytes,
        ext: str,
        label_column: Optional[str] = None
    ) -> list[DataRecord]:
        return load_dataset(file_bytes, ext, label_column)

    def run_inference(
        self,
        records: list[DataRecord],
    ) -> list[InferenceRecord]:
        """
        Run all records through the model and return InferenceRecord objects.
        """
        return self._strategy.run_inference(self.model, records)

    def summarise(self, results: list[InferenceRecord]) -> dict:
        """
        Compute accuracy statistics from inference results.
        """
        return summarise_results(results)

    def results_to_dicts(self, results: list[InferenceRecord]) -> list[dict]:
        """
        Convert InferenceRecord objects to plain dicts for downstream
        consumers (Vector_Analyzer, layer_analysis, JSON serialisation).
        """
        return [r.to_dict() for r in results]