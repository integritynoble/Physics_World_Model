import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.deps import get_db
from api.models.bootstrap import BootstrapProposal
from api.schemas.bootstrap import BootstrapCreateRequest, BootstrapResponse, SimilarityMatch
from bootstrap.generator import (
    generate_operator_graph_template,
    generate_experiment_spec_template,
    generate_sim_dataset_plan,
    generate_real_data_checklist,
    generate_viability_checklist,
)
from bootstrap.similarity import find_similar
from bootstrap.knowledge_base import get_all_modalities

router = APIRouter(tags=["bootstrap"])

_similarity_model = None


def get_similarity_model():
    """Lazy-load the sentence-transformers model (80MB, CPU-fast)."""
    global _similarity_model
    if _similarity_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _similarity_model = None
    return _similarity_model


@router.post("/bootstrap", response_model=BootstrapResponse, status_code=201)
async def create_bootstrap_proposal(
    body: BootstrapCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    # Fetch all modalities from the knowledge base
    all_modalities = await get_all_modalities(db)
    all_modalities_dict = {m.id: m for m in all_modalities}

    # Run similarity search if model available and we have modalities
    similar: list[dict] = []
    model = get_similarity_model()
    if model and all_modalities:
        raw_similar = find_similar(
            query_description=body.description,
            query_physics_class=body.physics_class,
            query_sensor_type=body.sensor_type,
            all_modalities=all_modalities,
            model=model,
            top_k=5,
        )
        similar = raw_similar

    # Generate templates based on similar modalities
    op_graph = generate_operator_graph_template(similar, all_modalities_dict) if similar else None
    exp_spec = generate_experiment_spec_template(body.name, similar, all_modalities_dict) if similar else None
    sim_plan = generate_sim_dataset_plan(similar, all_modalities_dict)
    real_checklist = generate_real_data_checklist(body.physics_class or "")
    viability = generate_viability_checklist()

    proposal = BootstrapProposal(
        id=f"bp_{uuid.uuid4().hex[:12]}",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        status="draft",
        name=body.name,
        description=body.description,
        physics_class=body.physics_class,
        sensor_type=body.sensor_type,
        geometry=body.geometry,
        similar_modalities=similar,
        operator_graph_template=op_graph,
        experiment_spec_template=exp_spec,
        sim_dataset_plan=sim_plan,
        real_data_checklist=real_checklist,
        calibration_modes=[],
        benchmark_metrics=[],
        uncertainty_notes=[],
        viability_checklist=viability,
    )
    db.add(proposal)
    await db.commit()
    await db.refresh(proposal)

    # Convert similar_modalities list to SimilarityMatch objects for response
    similar_matches = [SimilarityMatch(**m) for m in (proposal.similar_modalities or [])]

    return BootstrapResponse(
        id=proposal.id,
        status=proposal.status,
        name=proposal.name,
        similar_modalities=similar_matches,
        operator_graph_template=proposal.operator_graph_template,
        experiment_spec_template=proposal.experiment_spec_template,
        sim_dataset_plan=proposal.sim_dataset_plan,
        real_data_checklist=proposal.real_data_checklist or [],
        calibration_modes=proposal.calibration_modes or [],
        benchmark_metrics=proposal.benchmark_metrics or [],
        uncertainty_notes=proposal.uncertainty_notes or [],
        viability_checklist=proposal.viability_checklist or [],
    )


@router.get("/bootstrap/{proposal_id}", response_model=BootstrapResponse)
async def get_bootstrap_proposal(
    proposal_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(BootstrapProposal).where(BootstrapProposal.id == proposal_id)
    )
    proposal = result.scalar_one_or_none()
    if proposal is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id!r} not found")

    similar_matches = [SimilarityMatch(**m) for m in (proposal.similar_modalities or [])]
    return BootstrapResponse(
        id=proposal.id,
        status=proposal.status,
        name=proposal.name,
        similar_modalities=similar_matches,
        operator_graph_template=proposal.operator_graph_template,
        experiment_spec_template=proposal.experiment_spec_template,
        sim_dataset_plan=proposal.sim_dataset_plan,
        real_data_checklist=proposal.real_data_checklist or [],
        calibration_modes=proposal.calibration_modes or [],
        benchmark_metrics=proposal.benchmark_metrics or [],
        uncertainty_notes=proposal.uncertainty_notes or [],
        viability_checklist=proposal.viability_checklist or [],
    )
