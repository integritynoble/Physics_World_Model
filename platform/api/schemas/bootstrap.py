from pydantic import BaseModel, Field
from typing import Optional


class BootstrapCreateRequest(BaseModel):
    name: str = Field(..., min_length=3, max_length=128)
    description: str = Field(..., min_length=20, max_length=4000)
    physics_class: Optional[str] = None
    sensor_type: Optional[str] = None
    geometry: Optional[str] = None


class SimilarityMatch(BaseModel):
    modality_id: str
    modality_name: str
    score: float
    explanation: str


class BootstrapResponse(BaseModel):
    model_config = {"from_attributes": True}
    id: str
    status: str
    name: str
    similar_modalities: list[SimilarityMatch] = []
    operator_graph_template: Optional[dict] = None
    experiment_spec_template: Optional[dict] = None
    sim_dataset_plan: Optional[dict] = None
    real_data_checklist: list[str] = []
    calibration_modes: list[str] = []
    benchmark_metrics: list[str] = []
    uncertainty_notes: list[str] = []
    viability_checklist: list[str] = []
