"""
Layer-wise analysis helpers.
These operate directly on inference_results (list of record dicts),
each of which must contain a 'layer_activations' key:
  {layer_name: [float, ...], ...}
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance


def compute_prototypes(inference_results: list[dict]) -> dict[str, dict[int, list[float]]]:
    """
    For every (layer, label) combination, compute the mean activation vector
    across all CORRECTLY classified records with that label.

    Returns:
        {layer_name: {label: [mean_activation_float, ...]}}
    """
    correct = [r for r in inference_results if r.get('correct') is True]
    if not correct: #if no correct inference results
        return {}

    layer_names = list(correct[0]['layer_activations'].keys())

    # Group correct records by label
    by_label: dict[int, list[dict]] = {}
    for r in correct:
        label = r['label']
        if label is None:
            continue
        by_label.setdefault(label, []).append(r)

    prototypes: dict[str, dict[int, list[float]]] = {}

    for layer in layer_names:
        prototypes[layer] = {}
        for label, records in by_label.items():
            vectors = np.array([r['layer_activations'][layer] for r in records])
            prototypes[layer][label] = np.mean(vectors, axis=0).tolist()

    return prototypes


def compute_layer_deviations(
    record: dict,
    prototypes: dict[str, dict[int, list[float]]],
) -> dict:
    """
    For a single (incorrectly classified) record, compute the cosine distance
    between the record's activation at each layer and:
      - the prototype for the TRUE label   (correct_label_deviation)
      - the prototype for the PREDICTED label (predicted_label_deviation)

    Returns a dict ready to be serialised and sent to the frontend.
    """
    layer_names = list(record['layer_activations'].keys())
    true_label      = record['label']
    predicted_label = record['predicted']

    true_deviations      = []
    predicted_deviations = []

    for layer in layer_names:
        vec = np.array(record['layer_activations'][layer])

        # Distance to true-label prototype
        if (true_label is not None
                and layer in prototypes
                and true_label in prototypes[layer]):
            proto_true = np.array(prototypes[layer][true_label])
            # Guard against zero vectors (e.g. all-zero dropout outputs)
            if np.any(vec) and np.any(proto_true):
                true_deviations.append(float(cosine_distance(vec, proto_true)))
            else:
                true_deviations.append(0.0)
        else:
            true_deviations.append(None)

        # Distance to predicted-label prototype
        if (predicted_label is not None
                and layer in prototypes
                and predicted_label in prototypes[layer]):
            proto_pred = np.array(prototypes[layer][predicted_label])
            if np.any(vec) and np.any(proto_pred):
                predicted_deviations.append(float(cosine_distance(vec, proto_pred)))
            else:
                predicted_deviations.append(0.0)
        else:
            predicted_deviations.append(None)

    return {
        'record_id':               record['id'],
        'true_label':              true_label,
        'predicted_label':         predicted_label,
        'layer_names':             layer_names,
        'true_label_deviations':   true_deviations,
        'predicted_deviations':    predicted_deviations,
    }