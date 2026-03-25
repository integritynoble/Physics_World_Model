"""pwm_core.spec.problem_instance

ProblemInstance — concrete binding of CoreSpec + DomainProfile for a specific run.

A ProblemInstance binds concrete values to the combined CoreSpec + DomainProfile
schema. It is ephemeral — one per run or benchmark row. It carries the specific
scanner, dataset split, patient, phantom, or benchmark configuration for a single
evaluation.

Formula:
    CoreSpec + DomainProfile + ProblemInstance = executable PWM object
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pwm_core.cards.utils import new_card_id, now_iso


class ProblemInstance(BaseModel):
    """Concrete binding for a single PWM evaluation.

    Fields
    ------
    instance_id:
        Unique run/evaluation identifier.
    spec_id:
        CoreSpec ID this instance is bound to.
    domain_profile_id:
        DomainProfile ID (e.g. "imaging/v1", "ct_qc/v1").
    dataset_id:
        DatasetCard ID for the input dataset.
    method_id:
        MethodCard ID for the solver being evaluated.
    instrument_id:
        InstrumentCard ID (optional — physical instrument context).
    split:
        Dataset split identifier: "public", "dev", "hidden".
    sample_indices:
        Specific sample indices from the dataset split (None = all).
    solver_params:
        Concrete solver hyperparameters for this run.
    budget_s:
        Declared compute budget in seconds for this run.
    seed:
        Random seed for reproducibility.
    extra:
        Arbitrary instance-specific metadata.
    created_at:
        ISO-8601 creation timestamp.
    """

    instance_id: str = Field(default_factory=lambda: f"instance_{new_card_id()}")
    spec_id: str = ""
    domain_profile_id: str = "imaging/v1"
    dataset_id: str = ""
    method_id: str = ""
    instrument_id: Optional[str] = None
    split: str = "public"
    sample_indices: Optional[List[int]] = None
    solver_params: Dict[str, Any] = Field(default_factory=dict)
    budget_s: Optional[float] = None
    seed: int = 42
    extra: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)

    def describe(self) -> str:
        return (
            f"ProblemInstance({self.instance_id}) "
            f"spec={self.spec_id} domain={self.domain_profile_id} "
            f"dataset={self.dataset_id} method={self.method_id} split={self.split}"
        )
