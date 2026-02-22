"""PWM Platform routers."""

from pwm_platform.routers.auth import router as auth_router
from pwm_platform.routers.bootstrap import router as bootstrap_router
from pwm_platform.routers.datasets import router as datasets_router
from pwm_platform.routers.modalities import router as modalities_router
from pwm_platform.routers.pages import router as pages_router
from pwm_platform.routers.runs import router as runs_router

__all__ = [
    "auth_router",
    "bootstrap_router",
    "datasets_router",
    "modalities_router",
    "pages_router",
    "runs_router",
]
