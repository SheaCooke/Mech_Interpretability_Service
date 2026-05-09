"""
Pure function for summarising inference results.
Extracted from Model_Processor so it has no dependency on model loading
or inference — it works on a list of InferenceRecord objects and returns
a plain dict. This makes it independently testable and reusable.
"""

from __future__ import annotations
from .types import InferenceRecord


def summarise_results(results: list[InferenceRecord]) -> dict:
    """
    Compute accuracy statistics from a list of InferenceRecord objects.

    Returns a plain dict suitable for JSON serialisation. Per-class
    breakdown is included only when ground-truth labels are present.
    """
    total      = len(results)
    has_labels = all(r.label is not None for r in results)

    if not has_labels:
        return {
            'total_records': total,
            'has_labels':    False,
        }

    correct   = sum(1 for r in results if r.correct)
    incorrect = total - correct
    accuracy  = correct / total if total > 0 else 0.0

    class_results: dict[int, dict] = {}
    for record in results:
        label = record.label
        if label is None:
            continue
        if label not in class_results:
            class_results[label] = {'total': 0, 'correct': 0}
        class_results[label]['total'] += 1
        if record.correct:
            class_results[label]['correct'] += 1

    per_class_accuracy = {
        label: {
            'total':    stats['total'],
            'correct':  stats['correct'],
            'accuracy': stats['correct'] / stats['total'],
        }
        for label, stats in sorted(class_results.items())
    }

    return {
        'total_records':      total,
        'has_labels':         True,
        'correct':            correct,
        'incorrect':          incorrect,
        'accuracy':           accuracy,
        'per_class_accuracy': per_class_accuracy,
    }