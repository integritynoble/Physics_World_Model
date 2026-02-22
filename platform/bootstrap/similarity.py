"""CPU-viable embedding similarity for modality matching."""
from __future__ import annotations
import numpy as np
from typing import Optional, TYPE_CHECKING

MODEL_NAME = "all-MiniLM-L6-v2"   # ~80MB, 384-dim, CPU fast

def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)

def embed_modality(mod, model) -> list:
    primitives = getattr(mod, 'primitives', []) or []
    noise_models = getattr(mod, 'noise_models', []) or []
    task_types = getattr(mod, 'task_types', []) or []
    description = getattr(mod, 'description', '') or ''
    text = (
        f"{mod.name}. Physics: {mod.physics_class}. "
        f"Forward model: {getattr(mod, 'forward_model_family', '')}. "
        f"Sensor: {mod.sensor_type}. Geometry: {mod.geometry}. "
        f"Primitives: {', '.join(str(p) for p in primitives[:6])}. "
        f"Noise: {', '.join(str(n) for n in noise_models)}. "
        f"Tasks: {', '.join(str(t) for t in task_types)}. "
        f"{str(description)[:200]}"
    )
    return model.encode(text).tolist()

def find_similar(
    query_description: str,
    query_physics_class: Optional[str],
    query_sensor_type: Optional[str],
    all_modalities: list,
    model,
    top_k: int = 5,
) -> list:
    """Return top-k similar modalities with scores and explanations."""
    query_emb = model.encode(query_description)
    results = []
    for mod in all_modalities:
        emb = getattr(mod, 'embedding', None)
        if emb is None:
            continue
        emb_arr = np.array(emb)
        norm_q = np.linalg.norm(query_emb)
        norm_e = np.linalg.norm(emb_arr)
        cosine = float(np.dot(query_emb, emb_arr) / (norm_q * norm_e + 1e-9))
        bonus = 0.0
        explanation_parts = [f"semantic={cosine:.2f}"]
        if query_physics_class and getattr(mod, 'physics_class', '') == query_physics_class:
            bonus += 0.10
            explanation_parts.append(f"physics_class={query_physics_class}")
        if query_sensor_type and getattr(mod, 'sensor_type', '') == query_sensor_type:
            bonus += 0.08
            explanation_parts.append(f"sensor={query_sensor_type}")
        score = cosine + bonus
        results.append({
            "modality_id": mod.id,
            "modality_name": mod.name,
            "score": round(score, 3),
            "explanation": "; ".join(explanation_parts),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
