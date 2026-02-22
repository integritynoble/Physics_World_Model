import importlib

def test_api_importable():
    importlib.import_module("api")

def test_workers_importable():
    importlib.import_module("workers")

def test_bootstrap_importable():
    importlib.import_module("bootstrap")

def test_storage_importable():
    importlib.import_module("storage")

def test_ingestion_importable():
    importlib.import_module("ingestion")
