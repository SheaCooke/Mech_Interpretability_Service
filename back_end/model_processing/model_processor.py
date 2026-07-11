"""
coordinator class
"""

from __future__ import annotations
from typing import Optional
from .factory import ModelStrategyFactory
from .functional_components.dataset_functions import convert_to_parquet
from .functional_components.summarizer import summarize_results
from .types import ModelMetadata, InferenceRecord


class Model_Processor: #TODO: model processor name is not an accurate description of these methods

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._strategy = ModelStrategyFactory.create(file_path)
        self.model = self._strategy.load(file_path)
        self.model_data: ModelMetadata = self._strategy.extract_model_data(self.model)

    def convert_to_parquet(self, file_extension: str, file_bytes: bytes, session_id: str, label_column: Optional[str] = None) -> tuple[int,str,str,list]:
        return convert_to_parquet(file_extension, file_bytes, session_id, label_column)

    def run_inference(self, x_test_path: str, y_test_path: str, class_names: Optional[list], record_limit: int) -> list[InferenceRecord]:
        """
        Run all records through the model and return InferenceRecord objects.
        """
        return self._strategy.run_inference(self.model, x_test_path, y_test_path, class_names, record_limit)

    def summarize(self, results: list[InferenceRecord]) -> dict:
        """
        Compute accuracy statistics from inference results.
        """
        return summarize_results(results)

    def results_to_dicts(self, results: list[InferenceRecord]) -> list[dict]:
        """
        Convert InferenceRecord objects to plain dicts for downstream
        consumers (Net_Path_Vector_Analyzer, layer_analysis, JSON serialisation).
        """
        return [r.to_dict() for r in results]