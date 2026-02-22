from pydantic import BaseModel
from typing import Optional, Literal


class RunCreateRequest(BaseModel):
    prompt: Optional[str] = None
    spec: Optional[dict] = None
    compute_mode: Literal["cpu", "gpu", "auto"] = "auto"


class RunStatusResponse(BaseModel):
    model_config = {"from_attributes": True}
    id: str
    status: str
    compute_mode: str
    modality: Optional[str] = None
    metrics: Optional[dict] = None
    diagnosis_verdict: Optional[str] = None
    diagnosis_confidence: Optional[float] = None
    error_message: Optional[str] = None
    local_path: Optional[str] = None
    modal_cost_usd: Optional[float] = None
