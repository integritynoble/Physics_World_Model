from api.models.run import RunRecord, RunStatus
from api.models.dataset import DatasetRecord
from api.models.modality import ModalityRecord
from api.models.bootstrap import BootstrapProposal, BootstrapStatus

def test_run_status_values():
    assert RunStatus.queued == "queued"
    assert RunStatus.running == "running"
    assert RunStatus.done == "done"
    assert RunStatus.failed == "failed"

def test_bootstrap_status_values():
    assert BootstrapStatus.draft == "draft"
    assert BootstrapStatus.approved == "approved"
    assert BootstrapStatus.rejected == "rejected"

def test_run_record_tablename():
    assert RunRecord.__tablename__ == "runs"

def test_bootstrap_proposal_tablename():
    assert BootstrapProposal.__tablename__ == "bootstrap_proposals"

def test_dataset_record_tablename():
    assert DatasetRecord.__tablename__ == "datasets"

def test_modality_record_tablename():
    assert ModalityRecord.__tablename__ == "modalities"
