"""Tests for model_processing.vector_analyzer.Vector_Analyzer.

Vector_Analyzer is pure post-processing on inference results, so we can build
synthetic results in pure Python/NumPy and avoid loading any real model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from model_processing.vector_analyzer import Vector_Analyzer


class TestVectorAnalyzerConstruction:
    def test_builds_id_map_in_input_order(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        assert va.id_map == ["rec_0", "rec_1", "rec_2", "rec_3"]

    def test_activation_matrix_has_one_row_per_record(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        assert va.activation_matrix.shape == (4, 4)

    def test_activation_matrix_preserves_record_values(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        # Row 0 should still be [1, 0, 0, 0]
        np.testing.assert_array_almost_equal(
            va.activation_matrix[0],
            np.array([1.0, 0.0, 0.0, 0.0]),
        )

    def test_distance_matrix_is_square_and_has_zero_diagonal(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        n = len(simple_inference_results)
        assert va.distance_matrix.shape == (n, n)
        # cosine distance from a vector to itself must be ~0
        for i in range(n):
            assert va.distance_matrix[i][i] == pytest.approx(0.0, abs=1e-9)

    def test_distance_matrix_is_symmetric(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        np.testing.assert_array_almost_equal(
            va.distance_matrix, va.distance_matrix.T
        )


class TestFindAllSimilarPairs:
    def test_returns_only_upper_triangle_pairs(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        pairs = va.find_all_similar_pairs(threshold=2.1)  # everything passes
        # Each unordered pair appears exactly once
        seen = {(p["id_a"], p["id_b"]) for p in pairs}
        for a, b in seen:
            assert (b, a) not in seen
            assert a < b  # ids are 'rec_0', 'rec_1', ... so lex order matches

    def test_pairs_are_sorted_by_distance_ascending(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        pairs = va.find_all_similar_pairs(threshold=2.1)
        distances = [p["distance"] for p in pairs]
        assert distances == sorted(distances)

    def test_threshold_filters_out_distant_pairs(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        # rec_0 and rec_1 are nearly identical, rec_2 is orthogonal to them.
        close = va.find_all_similar_pairs(threshold=0.01)
        ids = {(p["id_a"], p["id_b"]) for p in close}
        assert ("rec_0", "rec_1") in ids
        assert ("rec_0", "rec_2") not in ids

    def test_pair_dict_includes_labels_for_both_records(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        pairs = va.find_all_similar_pairs(threshold=0.01)
        for pair in pairs:
            for key in ("id_a", "id_b", "distance", "label_a", "label_b"):
                assert key in pair

    def test_distance_value_is_python_float(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        pairs = va.find_all_similar_pairs(threshold=2.1)
        assert pairs, "expected at least one pair"
        for pair in pairs:
            assert isinstance(pair["distance"], float)
            assert math.isfinite(pair["distance"])

    def test_zero_threshold_returns_no_pairs(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        # cosine distance is always >= 0, the strict-less-than means nothing matches
        assert va.find_all_similar_pairs(threshold=0.0) == []


class TestGetIdMappingAndMatrix:
    def test_get_id_mapping_extracts_ids_in_order(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        result = va.get_id_mapping(simple_inference_results)
        assert result == [r["id"] for r in simple_inference_results]

    def test_get_activation_matrix_stacks_rows_correctly(self, simple_inference_results):
        va = Vector_Analyzer(simple_inference_results)
        matrix = va.get_activation_matrix(simple_inference_results)
        assert matrix.shape[0] == len(simple_inference_results)
        for i, record in enumerate(simple_inference_results):
            np.testing.assert_array_almost_equal(matrix[i], record["activations"])

    def test_get_distance_matrix_uses_cosine(self):
        # Two identical and one anti-aligned vector → distances 0, 1.
        m = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        va = Vector_Analyzer.__new__(Vector_Analyzer)  # bypass full __init__
        d = va.get_distance_matrix(m)
        assert d[0][1] == pytest.approx(0.0, abs=1e-9)
        assert d[0][2] == pytest.approx(1.0, abs=1e-9)
