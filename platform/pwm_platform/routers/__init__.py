"""PWM Platform routers."""

from pwm_platform.routers.auth import router as auth_router
from pwm_platform.routers.bootstrap import router as bootstrap_router
from pwm_platform.routers.datasets import router as datasets_router
from pwm_platform.routers.gcs_proxy import router as gcs_proxy_router
from pwm_platform.routers.modalities import router as modalities_router
from pwm_platform.routers.pages import router as pages_router
from pwm_platform.routers.runs import router as runs_router
from pwm_platform.routers.spec_chat import router as spec_chat_router
from pwm_platform.routers.submissions import router as submissions_router

__all__ = [
    "auth_router",
    "bootstrap_router",
    "datasets_router",
    "gcs_proxy_router",
    "modalities_router",
    "pages_router",
    "runs_router",
    "spec_chat_router",
    "submissions_router",
]
