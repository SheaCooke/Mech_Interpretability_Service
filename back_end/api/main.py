import os
import uuid
import tempfile
import numpy as np
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from back_end.model_processing.vector_analyzer import Vector_Analyzer
from back_end.model_processing.model_processor import Model_Processor
from back_end.model_processing.layer_analysis import compute_prototypes, compute_layer_deviations
from .types import PredictionFilter, SimilarPairsRequest, InferenceRequest
from .utilities import apply_filter, get_extension, numpy_safe
from ..common import SUPPORTED_MODEL_EXTENSIONS, SUPPORTED_DATASET_EXTENSIONS




app = FastAPI(title="Neural Network Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# session store
sessions: dict[str, dict] = {} #TODO: replace with redis or a DB

#TODO: move to utilities after changing how sessions are managed
def require_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Upload a model first.")
    return sessions[session_id]

@app.post("/upload/model")
async def upload_model(file: UploadFile = File(...)):
    """
    Accepts a model file (.keras, .onnx, .pt, .pth).
    Saves it to a temp file, loads it via Model_Processor,
    and returns a session_id for subsequent calls.
    """
    ext = get_extension(file.filename)
    if ext not in SUPPORTED_MODEL_EXTENSIONS: #TODO: replace
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model format '.{ext}'. Supported: {SUPPORTED_MODEL_EXTENSIONS}",
        )

    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        processor = Model_Processor(tmp.name)
    except Exception as e:
        os.unlink(tmp.name)
        raise HTTPException(status_code=422, detail=f"Failed to load model: {str(e)}")

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "processor":          processor,
        "model_tmp_path":     tmp.name,
        "model_filename":     file.filename,
        "dataset_records":    None,
        "inference_results":  None,
        "vector_analyzer":    None,
    }

    return {
        "session_id": session_id,
        "filename":   file.filename,
        "model_data": numpy_safe(processor.model_data.to_dict()),
    }


@app.post("/upload/dataset")
async def upload_dataset(
    session_id:   str,
    label_column: Optional[str] = None,
    file: UploadFile = File(...),
):
    """
    Accepts a dataset file (.csv or .npz) and associates it with an existing session.
    """
    session = require_session(session_id)

    ext = get_extension(file.filename)
    if ext not in SUPPORTED_DATASET_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dataset format '.{ext}'. Supported: {SUPPORTED_DATASET_EXTENSIONS}",
        )

    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        processor: "Model_Processor" = session["processor"]
        records = processor.load_dataset(tmp.name, label_column)
    except Exception as e:
        os.unlink(tmp.name)
        raise HTTPException(status_code=422, detail=f"Failed to load dataset: {str(e)}")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    session["dataset_records"]   = records
    session["inference_results"] = None
    session["vector_analyzer"]   = None

    return {
        "session_id":  session_id,
        "filename":    file.filename,
        "num_records": len(records),
        "sample": [{"id": r.id, "label": r.label} for r in records[:5]],
    }


@app.post("/inference/run")
def run_inference(body: InferenceRequest):
    """
    Runs the full inference pipeline on the previously uploaded dataset.
    Stores results in the session.
    """
    session = require_session(body.session_id)

    if session["dataset_records"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded for this session. Upload a dataset first.")

    processor: "Model_Processor" = session["processor"]
    records = session["dataset_records"]

    print('----------------')

    if body.limit is not None and 0 < body.limit < len(records):
        records = records[:body.limit]

    try:
        results = processor.run_inference(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Vector_Analyzer and any downstream code that uses dict-style access.
    result_dicts = processor.results_to_dicts(results)

    analyzer = Vector_Analyzer(result_dicts)

    # Store both: InferenceRecord list (typed) and dict list (for analysis modules)
    session["inference_results"] = result_dicts
    session["vector_analyzer"]   = analyzer

    summary = processor.summarise(results)

    return {
        "session_id":    body.session_id,
        "num_results":   len(results),
        "limit_applied": body.limit if body.limit and body.limit < len(session["dataset_records"]) else None,
        "summary":       numpy_safe(summary),
    }


@app.get("/inference/results") #TODO: not used?
def get_inference_results(session_id: str, limit: int = 100, offset: int = 0):
    """
    Returns a paginated slice of inference results (without raw activation vectors
    to keep payload size manageable).
    """
    session = require_session(session_id)

    if session["inference_results"] is None:
        raise HTTPException(status_code=400, detail="No inference results available. Run inference first.")

    results = session["inference_results"]
    page = results[offset: offset + limit]

    stripped = [
        {k: v for k, v in r.items() if k != "activations" and k != "input"}
        for r in page
    ]

    return {
        "session_id": session_id,
        "total":      len(results),
        "offset":     offset,
        "limit":      limit,
        "results":    numpy_safe(stripped),
    }


@app.post("/analysis/similar-pairs")
def similar_pairs(body: SimilarPairsRequest):
    session = require_session(body.session_id)
    if session["vector_analyzer"] is None:
        raise HTTPException(status_code=400, detail="No inference results available.")

    filtered = apply_filter(session["inference_results"], body.filter)
    if not filtered:
        raise HTTPException(status_code=400, detail=f"No records match filter '{body.filter}'.")

    if body.threshold_low >= body.threshold_high:
        raise HTTPException(
            status_code=400,
            detail=f"threshold_low ({body.threshold_low}) must be less than "
                   f"threshold_high ({body.threshold_high})."
        )

    analyzer = Vector_Analyzer(filtered)

    try:
        pairs = analyzer.find_all_similar_pairs(low=body.threshold_low, high=body.threshold_high)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return {
        "session_id":     body.session_id,
        "threshold_low":  body.threshold_low,
        "threshold_high": body.threshold_high,
        "filter":         body.filter,
        "num_pairs":      len(pairs),
        "pairs":          numpy_safe(pairs),
    }


@app.get("/model/data")
def get_model_data(session_id: str):
    session = require_session(session_id)
    processor: "Model_Processor" = session["processor"]
    return numpy_safe(processor.model_data.to_dict())


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    session = require_session(session_id)
    tmp = session.get("model_tmp_path")
    if tmp and os.path.exists(tmp):
        os.unlink(tmp)
    del sessions[session_id]
    return {"deleted": session_id}


@app.post("/analysis/cluster-plot")
def cluster_plot(body: dict):
    session_id = body.get("session_id")
    filter_val = body.get("filter", "all")
    session = require_session(session_id)

    if session["inference_results"] is None:
        raise HTTPException(status_code=400, detail="No inference results available.")

    filtered = apply_filter(session["inference_results"], filter_val)
    if not filtered:
        raise HTTPException(status_code=400, detail=f"No records match filter '{filter_val}'.")

    analyzer = Vector_Analyzer(filtered)

    try:
        plot_data = analyzer.get_cluster_plot_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

    return numpy_safe({
        "session_id": session_id,
        "filter":     filter_val,
        **plot_data,
    })


@app.get("/analysis/incorrect-records")
def get_incorrect_records(session_id: str):
    session = require_session(session_id)
    if session["inference_results"] is None:
        raise HTTPException(status_code=400, detail="No inference results. Run inference first.")

    incorrect = [
        {
            "id":        r["id"],
            "label":     r["label"],
            "predicted": r["predicted"],
        }
        for r in session["inference_results"]
        if r.get("correct") is False
    ]

    return {
        "session_id": session_id,
        "total":      len(incorrect),
        "records":    incorrect,
    }


@app.post("/analysis/layer-deviation")
def layer_deviation(body: dict):
    session_id = body.get("session_id")
    record_id  = body.get("record_id")

    session = require_session(session_id)

    if session["inference_results"] is None:
        raise HTTPException(status_code=400, detail="No inference results. Run inference first.")

    results = session["inference_results"]

    target = next((r for r in results if r["id"] == record_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found.")

    if "layer_activations" not in target or not target["layer_activations"]:
        raise HTTPException(
            status_code=422,
            detail="Record does not contain per-layer activations. "
                   "Ensure the model was loaded and inference run with the updated Model_Processor."
        )

    try:
        prototypes = compute_prototypes(results)
        deviations = compute_layer_deviations(target, prototypes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layer deviation failed: {str(e)}")

    return numpy_safe({
        "session_id": session_id,
        **deviations,
    })