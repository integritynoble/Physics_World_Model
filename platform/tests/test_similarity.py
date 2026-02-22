from unittest.mock import MagicMock
import numpy as np
from bootstrap.similarity import find_similar
from bootstrap.generator import (
    generate_operator_graph_template,
    generate_real_data_checklist,
    generate_viability_checklist,
    generate_experiment_spec_template,
)


def _make_mod(id_, physics_class="coherent", sensor_type="camera", emb=None):
    m = MagicMock()
    m.id = id_
    m.name = id_
    m.physics_class = physics_class
    m.sensor_type = sensor_type
    m.forward_model_family = "linear"
    m.geometry = "2d"
    m.embedding = emb if emb is not None else [1.0, 0.0, 0.0]
    m.primitives = ["fresnel_prop", "absorption"]
    m.noise_models = ["poisson"]
    m.task_types = ["recon"]
    m.description = "test modality"
    return m


def test_find_similar_top_k():
    model = MagicMock()
    model.encode.return_value = np.array([1.0, 0.0, 0.0])
    mods = [
        _make_mod("a", "coherent", "camera", [1.0, 0.0, 0.0]),
        _make_mod("b", "incoherent", "detector", [0.0, 1.0, 0.0]),
        _make_mod("c", "coherent", "camera", [0.9, 0.1, 0.0]),
    ]
    results = find_similar("coherent imaging", "coherent", "camera", mods, model, top_k=2)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


def test_find_similar_physics_class_bonus():
    model = MagicMock()
    model.encode.return_value = np.array([0.6, 0.4, 0.0])
    mods = [
        _make_mod("match", "coherent", "camera", [0.6, 0.4, 0.0]),
        _make_mod("nomatch", "tomographic", "detector", [0.6, 0.4, 0.0]),
    ]
    results = find_similar("query", "coherent", "camera", mods, model, top_k=2)
    assert results[0]["modality_id"] == "match"
    assert results[0]["score"] > results[1]["score"]


def test_find_similar_skips_none_embedding():
    model = MagicMock()
    model.encode.return_value = np.array([1.0, 0.0, 0.0])
    m = _make_mod("no_emb")
    m.embedding = None
    results = find_similar("query", None, None, [m], model, top_k=5)
    assert len(results) == 0


def test_generate_operator_graph_template():
    similar = [{"modality_id": "a", "modality_name": "A", "score": 0.9, "explanation": "x"}]
    mod_a = _make_mod("a")
    result = generate_operator_graph_template(similar, {"a": mod_a})
    assert "primitives" in result
    assert "fresnel_prop" in result["primitives"]
    assert result["source"] == "bootstrap_generated"


def test_generate_real_data_checklist_base_length():
    items = generate_real_data_checklist("")
    assert len(items) >= 8


def test_generate_real_data_checklist_coherent():
    items = generate_real_data_checklist("coherent")
    assert any("coherence" in i.lower() for i in items)


def test_generate_viability_checklist_length():
    items = generate_viability_checklist()
    assert len(items) >= 6


def test_generate_experiment_spec_template_structure():
    similar = [{"modality_id": "cassi_sci", "modality_name": "CASSI", "score": 0.9, "explanation": "x"}]
    result = generate_experiment_spec_template("THz TDS", similar, {})
    assert result["version"] == "0.2.1"
    assert "states" in result
    assert "input" in result
