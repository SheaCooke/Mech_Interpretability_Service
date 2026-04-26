"""Tests for the (currently mostly empty) top-level Analyzer class."""

from __future__ import annotations

from analyzer import Analyzer


class TestAnalyzer:
    def test_can_be_instantiated_without_arguments(self):
        analyzer = Analyzer()
        assert analyzer is not None

    def test_analyzer_class_is_importable_as_attribute(self):
        # Sanity check: the symbol exists at module scope
        import analyzer as analyzer_module
        assert hasattr(analyzer_module, "Analyzer")
        assert isinstance(Analyzer(), analyzer_module.Analyzer)
