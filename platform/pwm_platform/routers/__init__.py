"""PWM Platform routers."""

from pwm_platform.routers.audit import router as audit_router
from pwm_platform.routers.admin_users import router as admin_users_router
from pwm_platform.routers.instruments import router as instruments_router
from pwm_platform.routers.auth import router as auth_router
from pwm_platform.routers.billing import router as billing_router
from pwm_platform.routers.bootstrap import router as bootstrap_router
from pwm_platform.routers.claim_review import router as claim_review_router
from pwm_platform.routers.contributors import router as contributors_router
from pwm_platform.routers.datasets import router as datasets_router
from pwm_platform.routers.gcs_proxy import router as gcs_proxy_router
from pwm_platform.routers.modalities import router as modalities_router
from pwm_platform.routers.pages import router as pages_router
from pwm_platform.routers.runs import router as runs_router
from pwm_platform.routers.spec_chat import router as spec_chat_router
from pwm_platform.routers.submissions import router as submissions_router
from pwm_platform.routers.system_design_chat import router as system_design_chat_router
from pwm_platform.routers.trust_promotion import router as trust_promotion_router

__all__ = [
    "audit_router",
    "admin_users_router",
    "instruments_router",
    "auth_router",
    "billing_router",
    "bootstrap_router",
    "claim_review_router",
    "contributors_router",
    "datasets_router",
    "gcs_proxy_router",
    "modalities_router",
    "pages_router",
    "runs_router",
    "spec_chat_router",
    "submissions_router",
    "system_design_chat_router",
    "trust_promotion_router",
]
