from unittest.mock import patch, MagicMock

def test_dispatch_cpu_run_success():
    with patch("workers.tasks.should_use_gpu", return_value=False),          patch("workers.tasks.pwm_run") as mock_run,          patch("workers.tasks._update_run") as mock_update,          patch("pathlib.Path.mkdir"):
        mock_run.return_value = {"spec_id": "test", "recon": [], "diagnosis": None}
        from workers.tasks import dispatch_pwm_run
        dispatch_pwm_run.run("run_001", {"states": {"physics": {"modality": "ct_diagnostic"}}}, "cpu")
        mock_update.assert_called_once()
        call_args = mock_update.call_args
        assert call_args[1]["status"] == "done" or call_args[0][1] == "done"
