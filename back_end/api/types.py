from enum import Enum
from pydantic import BaseModel
from typing import Optional

class PredictionFilter(str, Enum):
    ALL       = "all"
    CORRECT   = "correct"
    INCORRECT = "incorrect"


class SimilarPairsRequest(BaseModel):
    session_id:     str
    threshold_low:  float = 0.0
    threshold_high: float = 0.2
    filter: PredictionFilter = PredictionFilter.ALL


class InferenceRequest(BaseModel):
    session_id:   str
    label_column: Optional[str] = None
    batch_size:   Optional[int] = None
    limit:        Optional[int] = None  # num records to run; None = all