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
from .types import PredictionFilter, SimilarPairsRequest, InferenceRequest, ClusterPlotRequest
from ..model_processing.types import InferenceRecord
from .utilities import get_extension, numpy_safe
from ..common import SUPPORTED_MODEL_EXTENSIONS, SUPPORTED_DATASET_EXTENSIONS
import logging
from logging.config import dictConfig



dictConfig({
  "version": 1,
  "formatters": {
    "default": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}
  },
  "handlers": {
    "console": {"class": "logging.StreamHandler", "formatter": "default"}
    # "file": {
    #   "class": "logging.handlers.RotatingFileHandler",
    #   "formatter": "default",
    #   "filename": "logs/app.log",
    #   "mode": "a",
    #   "maxBytes": 10_485_760,   # 10 MB
    #   "backupCount": 5,
    #   "encoding": "utf-8"
    # }
  },
  "root": {"level": "INFO", "handlers": ["console"]}
})

logger = logging.getLogger(__name__)
logger.info("App starting")


app = FastAPI(title="Neural Network Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, dict] = {} #TODO: replace with redis or a DB

#TODO: move to utilities after changing how sessions are managed
def require_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Upload a model first.")
    return sessions[session_id]

@app.post("/upload/model")
async def upload_model(file: UploadFile = File(...)):

    ext = get_extension(file.filename)
    if ext not in SUPPORTED_MODEL_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model format '.{ext}'. Supported: {SUPPORTED_MODEL_EXTENSIONS}",
        )

    logger.info(f"user uploaded model {file.filename}")

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

    logger.info(f"created session {session_id}")
    
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
    session = require_session(session_id)

    ext = get_extension(file.filename)
    if ext not in SUPPORTED_DATASET_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dataset format '.{ext}'. Supported: {SUPPORTED_DATASET_EXTENSIONS}",
        )
    
    logger.info(f"loaded dataset file {file.filename} for session {session_id}")

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

    return {
        "session_id":  session_id,
        "filename":    file.filename,
        "num_records": len(records)
    }


@app.post("/inference/run")
def run_inference(body: InferenceRequest):
    session = require_session(body.session_id)

    if session["dataset_records"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded for this session. Upload a dataset first.")

    processor: Model_Processor = session["processor"]
    records = session["dataset_records"]

    if body.limit is not None and 0 < body.limit < len(records):
        records = records[:body.limit]

    logger.info(f"running inference for session {body.session_id}")

    try:
        results: list[InferenceRecord] = processor.run_inference(records)
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    result_dicts = processor.results_to_dicts(results) #convert to dictionaries to support json responses
    analyzer = Vector_Analyzer(result_dicts)

    session["inference_results"] = result_dicts
    session["vector_analyzer"] = analyzer

    summary = processor.summarise(results)

    return {
        "session_id":    body.session_id,
        "num_results":   len(results),
        "limit_applied": body.limit if body.limit and body.limit < len(session["dataset_records"]) else None,
        "summary":       numpy_safe(summary)
    }



@app.post("/analysis/similar-pairs")
def similar_pairs(body: SimilarPairsRequest):

    session = require_session(body.session_id)

    if session["inference_results"] is None:
        raise HTTPException(status_code=400, detail="No inference results available.")
    elif body.threshold_low >= body.threshold_high:
        raise HTTPException(
            status_code=400,
            detail=f"threshold_low ({body.threshold_low}) must be less than "
                   f"threshold_high ({body.threshold_high})."
        )

    logger.info(f"session: {body.session_id}, found {len(session['inference_results'])} inference results to process")

    analyzer = session["vector_analyzer"]

    try:
        pairs: list[dict] = analyzer.find_all_similar_pairs(session["inference_results"], low=body.threshold_low, high=body.threshold_high, filter=body.filter)
        logger.info(f"returning {len(pairs)} similar pairs for session>: {body.session_id}")
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


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    session = require_session(session_id)
    tmp = session.get("model_tmp_path")
    if tmp and os.path.exists(tmp):
        os.unlink(tmp)
    del sessions[session_id]
    return {"deleted": session_id}


@app.post("/analysis/cluster-plot")
def cluster_plot(body: ClusterPlotRequest):
    session_id = body.session_id
    session = require_session(session_id)

    if session["inference_results"] is None:
        raise HTTPException(status_code=400, detail="No inference results available.")

    analyzer = session["vector_analyzer"]

    try:
        plot_data = analyzer.get_cluster_plot_data(session["inference_results"], body.filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

    return numpy_safe({
        "session_id": session_id,
        "filter": body.filter,
        **plot_data
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