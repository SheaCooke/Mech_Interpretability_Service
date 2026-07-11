
from __future__ import annotations
from ..types import InferenceRecord


def summarize_results(results: list[InferenceRecord]) -> dict:
    total = len(results)
    has_labels = all(r.label is not None for r in results)

    if not has_labels:
        return {
            'total_records': total,
            'has_labels':    False
        }

    correct = sum(1 for r in results if r.correct)
    incorrect = total - correct
    accuracy = correct / total if total > 0 else 0.0

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
            'accuracy': stats['correct'] / stats['total']
        }
        for label, stats in sorted(class_results.items())
    }

    return {
        'total_records':      total,
        'has_labels':         True,
        'correct':            correct,
        'incorrect':          incorrect,
        'accuracy':           accuracy,
        'per_class_accuracy': per_class_accuracy
    }