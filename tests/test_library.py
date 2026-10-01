"""The library grows only through a validator, and a run reads one pinned state."""
import pytest

from pwm_mode.library import Library, LibraryError

BASE = [{"id": "p1", "layer": "L1", "field": "ldct", "title": "Edge-preserving restoration", "conditions": {}, "payload": {}},
        {"id": "b1", "layer": "L3", "field": "ldct", "title": "Lesion benchmark", "conditions": {}, "payload": {}}]


def _sub(lib, hu=150.0, r=5, sid="s1"):
    return lib.submit(field="ldct", layer="L4", id=f"sol-c{int(hu)}-r{r}", title="tuned", summary="",
                      payload={"params": {"sigma_s": 3}}, conditions={"anatomy": "chest", "lesion_hu": hu, "lesion_r": r},
                      source_problems=["dev-1"], submitter="engine-search")


def test_seed_once_and_states_have_digests(tmp_path):
    lib = Library(tmp_path)
    st = lib.seed(BASE)
    assert st["n"] == 0 and len(st["digest"]) == 64 and len(st["artifacts"]) == 2
    with pytest.raises(LibraryError):
        lib.seed(BASE)


def test_only_a_named_independent_passing_validation_adds_to_the_library(tmp_path):
    lib = Library(tmp_path)
    lib.seed(BASE)
    sub = _sub(lib)
    for bad in ({"passed": False, "validator": "v"}, {"passed": True}, {"passed": True, "validator": "engine-search"}):
        with pytest.raises(LibraryError):
            lib.accept(sub["id"], bad)
    assert lib.head() == 0
    st = lib.accept(sub["id"], {"passed": True, "validator": "validator-ldct-1", "objective": 4.2})
    assert st["n"] == 1 and st["parent"] == 0 and st["added"] == ["sol-c150-r5@1"]
    with pytest.raises(LibraryError):                                   # once
        lib.accept(sub["id"], {"passed": True, "validator": "validator-ldct-1"})


def test_a_run_pinned_to_an_old_state_does_not_see_later_solutions(tmp_path):
    lib = Library(tmp_path)
    lib.seed(BASE)
    lib.accept(_sub(lib)["id"], {"passed": True, "validator": "v", "objective": 4.0})
    cond = {"anatomy": "chest", "lesion_hu": 150.0, "lesion_r": 5}
    assert lib.best_solution("ldct", cond, state=0) is None
    assert lib.best_solution("ldct", cond, state=1)["id"] == "sol-c150-r5"
    with pytest.raises(LibraryError):
        lib.get("sol-c150-r5@1", state=0)


def test_best_solution_prefers_the_closest_conditions_then_the_better_result(tmp_path):
    lib = Library(tmp_path)
    lib.seed(BASE)
    lib.accept(_sub(lib, 150.0, 5)["id"], {"passed": True, "validator": "v", "objective": 3.5})
    lib.accept(_sub(lib, 100.0, 3)["id"], {"passed": True, "validator": "v", "objective": 9.0})
    best = lib.best_solution("ldct", {"anatomy": "chest", "lesion_hu": 150.0, "lesion_r": 5})
    assert best["id"] == "sol-c150-r5"                                  # matching beats a higher score elsewhere


def test_a_new_version_replaces_the_old_in_the_next_state(tmp_path):
    lib = Library(tmp_path)
    lib.seed(BASE)
    lib.accept(_sub(lib)["id"], {"passed": True, "validator": "v", "objective": 3.5})
    st = lib.accept(_sub(lib)["id"], {"passed": True, "validator": "v", "objective": 4.5})
    sols = [r for r in st["artifacts"] if r.startswith("sol-")]
    assert sols == ["sol-c150-r5@2"]
