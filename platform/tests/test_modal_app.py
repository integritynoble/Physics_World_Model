# platform/tests/test_modal_app.py
# These tests mock Modal so no actual GPU or Modal account is needed
from unittest.mock import patch, MagicMock


def test_modal_app_importable():
    """modal_app.py should import without error when Modal is mocked."""
    with patch.dict("sys.modules", {
        "modal": MagicMock(),
        "modal.Image": MagicMock(),
        "modal.gpu": MagicMock(),
    }):
        import importlib
        import sys
        # Remove any cached version
        for key in list(sys.modules.keys()):
            if "workers.modal_app" in key:
                del sys.modules[key]
        import workers.modal_app
        assert True


def test_simulate_gpu_function_exists():
    """simulate_gpu should be defined at module level."""
    with patch.dict("sys.modules", {
        "modal": MagicMock(),
        "modal.Image": MagicMock(),
        "modal.gpu": MagicMock(),
    }):
        import sys
        for key in list(sys.modules.keys()):
            if "workers.modal_app" in key:
                del sys.modules[key]
        import workers.modal_app as ma
        assert hasattr(ma, "simulate_gpu")


def test_modal_app_has_pwm_app():
    """The Modal app should be named pwm-platform."""
    mock_modal = MagicMock()
    mock_app_instance = MagicMock()
    mock_app_instance.function = MagicMock(return_value=lambda f: f)
    mock_modal.App.return_value = mock_app_instance
    mock_modal.Image = MagicMock()
    mock_modal.Image.debian_slim = MagicMock(return_value=MagicMock(
        pip_install=MagicMock(return_value=MagicMock(
            run_commands=MagicMock(return_value=MagicMock())
        ))
    ))
    mock_modal.gpu = MagicMock()
    mock_modal.Retries = MagicMock()

    with patch.dict("sys.modules", {"modal": mock_modal}):
        import sys
        for key in list(sys.modules.keys()):
            if "workers.modal_app" in key:
                del sys.modules[key]
        import workers.modal_app as ma
        # App should be created with the name "pwm-platform"
        mock_modal.App.assert_called_once_with("pwm-platform")
