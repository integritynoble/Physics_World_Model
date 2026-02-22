import tempfile, os, base64, pathlib
from storage.local import LocalStorage

def test_runbundle_path_format():
    store = LocalStorage(workspace="/data/workspace")
    assert store.runbundle_path("run_abc") == "/data/workspace/runbundles/run_abc"

def test_dataset_path_format():
    store = LocalStorage(workspace="/data/workspace")
    p = store.dataset_path("cassi_sci", "sim", "v1.0.0")
    assert "datasets/sim/cassi_sci/v1.0.0" in p

def test_ensure_run_dir_creates_directory():
    with tempfile.TemporaryDirectory() as workspace:
        store = LocalStorage(workspace=workspace)
        path = store.ensure_runbundle_dir("run_test_001")
        assert os.path.isdir(path)

def test_save_artifacts_writes_files():
    with tempfile.TemporaryDirectory() as workspace:
        store = LocalStorage(workspace=workspace)
        bundle_dir = store.ensure_runbundle_dir("run_test_002")
        artifacts_b64 = {
            "spec.json": base64.b64encode(b'{"id": "test"}').decode(),
            "diagnosis.json": base64.b64encode(b'{"verdict": "ok"}').decode(),
        }
        store.save_artifacts(bundle_dir, artifacts_b64)
        assert (pathlib.Path(bundle_dir) / "spec.json").read_bytes() == b'{"id": "test"}'
        assert (pathlib.Path(bundle_dir) / "diagnosis.json").read_bytes() == b'{"verdict": "ok"}'

def test_list_runbundles():
    with tempfile.TemporaryDirectory() as workspace:
        store = LocalStorage(workspace=workspace)
        for rid in ["run_a", "run_b", "run_c"]:
            store.ensure_runbundle_dir(rid)
        bundles = store.list_runbundles()
        assert set(bundles) == {"run_a", "run_b", "run_c"}

def test_runbundle_exists_false():
    with tempfile.TemporaryDirectory() as workspace:
        store = LocalStorage(workspace=workspace)
        assert store.runbundle_exists("nonexistent") is False

def test_runbundle_exists_true():
    with tempfile.TemporaryDirectory() as workspace:
        store = LocalStorage(workspace=workspace)
        store.ensure_runbundle_dir("run_exists")
        assert store.runbundle_exists("run_exists") is True
