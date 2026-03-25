"""pwm_core.core.primitive_registry

PrimitiveRegistry -- typed representation of a versioned primitive registry.

PrimitiveRegistry is a deep-sun object. It represents the full content of
a versioned primitive registry (e.g., general/v1 or imaging/v1).

The actual YAML content lives at:
  contrib/primitive_registry/{domain}/v{n}/primitives.yaml

This model provides typed access, validation, and JSON serialization.
"""

from __future__ import annotations
import datetime
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover -- fallback kept for minimal envs
    from dataclasses import dataclass as BaseModel  # type: ignore[assignment]

    def Field(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[misc]
        return None


class PrimitiveDefinition(BaseModel):
    """One primitive in a registry."""

    name: str = Field(..., description="Primitive name, e.g. 'Transform'")
    symbol: Optional[str] = Field(default=None, description="Short symbol, e.g. 'T'")
    scope: str = Field(..., description="What this primitive does")
    category: str = Field(default="general", description="'general' or domain name")
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    decomposition: Optional[List[str]] = Field(
        default=None,
        description="How this decomposes into general/v1 primitives (for domain primitives)",
    )


class PrimitiveRegistry(BaseModel):
    """Typed representation of a versioned primitive registry.

    Deep-sun object. OperatorGraph references primitives by
    (registry_id, version, name) triples resolved against this registry.
    """

    registry_id: str = Field(
        ..., description="e.g. 'general/v1' or 'imaging/v1'"
    )
    domain: str = Field(
        ..., description="Domain name: 'general', 'imaging', etc."
    )
    version: str = Field(..., description="Registry version, e.g. 'v1'")
    description: str = Field(default="", description="Human-readable description")
    primitives: List[PrimitiveDefinition] = Field(default_factory=list)
    is_universal: bool = Field(
        default=False,
        description="True for general/v1 (part of semantic kernel)",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    # -- helpers --------------------------------------------------------

    @property
    def primitive_count(self) -> int:
        return len(self.primitives)

    def get_primitive(self, name: str) -> Optional[PrimitiveDefinition]:
        for p in self.primitives:
            if p.name == name:
                return p
        return None

    def primitive_names(self) -> List[str]:
        return [p.name for p in self.primitives]

    def to_dict(self) -> Dict[str, Any]:
        try:
            return self.model_dump()
        except AttributeError:
            return self.__dict__

    # -- factory --------------------------------------------------------

    @classmethod
    def from_yaml_file(cls, path: str) -> "PrimitiveRegistry":
        """Load a PrimitiveRegistry from a primitives.yaml file."""
        import yaml
        from pathlib import Path

        p = Path(path)
        with open(p) as f:
            data = yaml.safe_load(f)

        # Infer domain and version from path
        parts = p.parts
        domain = "unknown"
        version = "v1"
        for i, part in enumerate(parts):
            if part == "primitive_registry" and i + 2 < len(parts):
                domain = parts[i + 1]
                version = parts[i + 2]
                break

        registry_id = f"{domain}/{version}"
        primitives: List[PrimitiveDefinition] = []
        for entry in data.get("primitives", []) if isinstance(data, dict) else []:
            if isinstance(entry, dict):
                primitives.append(
                    PrimitiveDefinition(
                        name=entry.get("name", ""),
                        symbol=entry.get("symbol"),
                        scope=entry.get("scope", entry.get("description", "")),
                        category=domain,
                        inputs=entry.get("inputs", []),
                        outputs=entry.get("outputs", []),
                        decomposition=entry.get("decomposition"),
                    )
                )

        return cls(
            registry_id=registry_id,
            domain=domain,
            version=version,
            description=data.get("description", "") if isinstance(data, dict) else "",
            primitives=primitives,
            is_universal=(domain == "general"),
        )
