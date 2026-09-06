"""
Works out how many records a session can hold for a given RAM ceiling.

The user supplies a max RAM figure in the UI. Everything downstream of inference
scales with the record count, so that ceiling has to be turned into a record
count before inference starts. The estimate deliberately assumes the user will
go on to use every analysis feature on the resulting session, because those
features allocate far more than inference itself does:

  * find_all_similar_pairs caches a full N x N cosine distance matrix
  * get_cluster_plot_data runs UMAP over the whole activation matrix
  * compute_layer_deviations builds per-layer prototype matrices

Cost is therefore modelled as a quadratic in the record count:

    bytes(N) = baseline + QUADRATIC_BYTES_PER_PAIR * N^2 + per_record_bytes * N

and the record budget is the largest N that fits under the ceiling.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from ..types import ModelMetadata

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024

# CPython object sizes (64 bit).
# Activations survive in session["inference_results"] as dicts of plain python
# lists, so every single activation value costs a boxed float plus a pointer.
# This dominates the linear term, which is why it is spelled out rather than
# folded into one fudge factor.
PY_FLOAT_BYTES = 24        # sys.getsizeof(0.0)
POINTER_BYTES = 8          # one list / tuple slot
LIST_HEADER_BYTES = 56     # sys.getsizeof([])
TUPLE_HEADER_BYTES = 40    # sys.getsizeof(())
DICT_ENTRY_BYTES = 100     # a layer name key plus its hash table slot
RECORD_DICT_BYTES = 400    # the record dict itself: id, label, predicted, correct

FLOAT32_BYTES = 4
FLOAT64_BYTES = 8

# N^2 terms.
# Net_Path_Vector_Analyzer caches distance_matrix as float64, and
# find_all_similar_pairs then materialises three N x N boolean masks over it
# ((dm >= low), (dm <= high) and their conjunction).
DISTANCE_MATRIX_BYTES = FLOAT64_BYTES
SIMILARITY_MASK_BYTES = 3
QUADRATIC_BYTES_PER_PAIR = DISTANCE_MATRIX_BYTES + SIMILARITY_MASK_BYTES

# UMAP: n_neighbors defaults to 15; the kNN graph, the fuzzy simplicial set and
# the 2D embedding are all roughly linear in the record count.
UMAP_NEIGHBORS = 15
UMAP_GRAPH_BYTES_PER_RECORD = UMAP_NEIGHBORS * 3 * FLOAT64_BYTES

# Covers what the model above does not name: the pairs list returned to the
# client, fragmentation, numpy scratch buffers, prototype dicts.
SAFETY_FACTOR = 1.15

# Used only when psutil is unavailable. A loaded tensorflow + keras process with
# a small model sits somewhere in this range before any records are held.
FALLBACK_BASELINE_MB = 700

# When a layer's output shape could not be resolved, fall back to this many
# activation values for it so the estimate stays conservative rather than absent.
UNKNOWN_LAYER_ACTIVATIONS = 512


def current_process_memory_mb() -> tuple[float, bool]:
    """
    Resident set size of this process right now, in MB.

    Returns (megabytes, measured). measured is False when psutil is not
    installed and FALLBACK_BASELINE_MB was used instead.
    """
    try:
        import psutil #optional dependency, keep the import local
        return psutil.Process().memory_info().rss / BYTES_PER_MB, True
    except Exception:
        logger.warning("psutil unavailable, falling back to a %s MB baseline estimate", FALLBACK_BASELINE_MB)
        return float(FALLBACK_BASELINE_MB), False


def activation_profile(model_data: ModelMetadata) -> tuple[int, int, int, bool]:
    """
    Describe the activations one record produces.

    Only layers flagged relevant_inference are counted, matching the activation
    model that KerasStrategy._prepare builds. Training only layers such as
    Dropout are the identity at inference and are not collected.

    Returns (total_values, num_layers, largest_layer_values, exact).
    """
    total = 0
    largest = 0
    collected = 0
    exact = True

    for layer in model_data.layers:
        if not layer.relevant_inference:
            continue

        size = layer.output_size

        if size is None: #shape was not resolvable, approximate so the caller still gets a number
            size = layer.num_neurons or UNKNOWN_LAYER_ACTIVATIONS
            exact = False

        total += size
        largest = max(largest, size)
        collected += 1

    return total, collected, largest, exact


def per_record_bytes(activation_values: int, num_layers: int, largest_layer_values: int) -> dict:
    """
    Bytes consumed by one record, broken down by what holds it.

    Every entry here is either retained for the life of the session or is live
    at the same time as the retained data, so they add rather than overlap.
    """
    d = activation_values
    l = num_layers

    # session["inference_results"]: one dict per record, activations as python lists
    inference_results = (
        d * (PY_FLOAT_BYTES + POINTER_BYTES)
        + l * (LIST_HEADER_BYTES + DICT_ENTRY_BYTES)
        + RECORD_DICT_BYTES
    )

    # The InferenceRecord tuples stay alive alongside the dicts for the whole
    # /inference/run request. They share the float objects, so only the slots cost.
    inference_records = d * POINTER_BYTES + l * TUPLE_HEADER_BYTES + RECORD_DICT_BYTES

    # Net_Path_Vector_Analyzer.activation_matrix, retained on the analyzer
    activation_matrix = d * FLOAT32_BYTES

    # cdist upcasts and row-normalises before the dot product
    distance_matrix_scratch = d * FLOAT64_BYTES

    # UMAP's own float32 copy of the activation matrix, plus the kNN graph
    umap_scratch = d * FLOAT32_BYTES + UMAP_GRAPH_BYTES_PER_RECORD

    # compute_prototypes stacks the largest layer across records as float64
    prototype_scratch = largest_layer_values * FLOAT64_BYTES

    # similar-pairs and cluster-plot are separate requests, so their scratch
    # does not coexist. Charge whichever is worse.
    analysis_scratch = max(distance_matrix_scratch, umap_scratch)

    total = (
        inference_results
        + inference_records
        + activation_matrix
        + analysis_scratch
        + prototype_scratch
    )

    return {
        "inference_results": inference_results,
        "inference_records": inference_records,
        "activation_matrix": activation_matrix,
        "analysis_scratch": analysis_scratch,
        "prototype_scratch": prototype_scratch,
        "total": total,
    }


def project_memory_bytes(per_record: int, num_records: int) -> int:
    """
    Bytes above the baseline needed to hold num_records, safety factor included.
    """
    n = max(0, int(num_records))
    raw = QUADRATIC_BYTES_PER_PAIR * n * n + per_record * n
    return int(raw * SAFETY_FACTOR)


def max_records_for_budget(per_record: int, available_bytes: float) -> int:
    """
    Largest N satisfying  a*N^2 + b*N <= available_bytes / SAFETY_FACTOR.

    Solved with the quadratic formula rather than a search so this stays cheap
    enough to call on every keystroke in the UI.
    """
    if available_bytes <= 0:
        return 0

    usable = available_bytes / SAFETY_FACTOR
    a = QUADRATIC_BYTES_PER_PAIR
    b = per_record

    n = (-b + math.sqrt(b * b + 4 * a * usable)) / (2 * a)
    return max(0, int(math.floor(n)))


def estimate_record_budget(
    model_data: ModelMetadata,
    max_memory_mb: Optional[float],
    total_records: int,
    selected_records: Optional[int] = None,
) -> dict:
    """
    Turn a RAM ceiling into a record count.

    max_memory_mb of None or 0 means "no ceiling": the budget is the whole
    dataset. selected_records, when given, is projected against the same model
    so the UI can show what the current slider position will actually cost.
    """
    activation_values, num_layers, largest_layer, exact = activation_profile(model_data)
    breakdown = per_record_bytes(activation_values, num_layers, largest_layer)
    per_record = breakdown["total"]

    baseline_mb, baseline_measured = current_process_memory_mb()

    capped = bool(max_memory_mb and max_memory_mb > 0)

    if not capped:
        max_records = total_records
        available_mb = None
    else:
        available_mb = float(max_memory_mb) - baseline_mb
        max_records = min(total_records, max_records_for_budget(per_record, available_mb * BYTES_PER_MB))

    projected_for = selected_records if selected_records is not None else max_records
    projected_for = max(0, min(int(projected_for), total_records))
    projected_mb = baseline_mb + project_memory_bytes(per_record, projected_for) / BYTES_PER_MB

    return {
        "max_records": int(max_records),
        "total_records": int(total_records),
        "activation_values_per_record": int(activation_values),
        "bytes_per_record": int(per_record),
        "bytes_per_record_breakdown": {k: int(v) for k, v in breakdown.items()},
        "baseline_mb": round(baseline_mb, 1),
        "baseline_measured": baseline_measured,
        "available_mb": round(available_mb, 1) if available_mb is not None else None,
        "capped": capped,
        "exact": exact,
        "projected_records": int(projected_for),
        "projected_mb": round(projected_mb, 1),
        "safety_factor": SAFETY_FACTOR,
    }
