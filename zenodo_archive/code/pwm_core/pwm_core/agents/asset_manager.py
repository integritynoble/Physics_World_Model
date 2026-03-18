"""pwm_core.agents.asset_manager

Illustration stage management and asset licensing.

Manages optional illustration assets (system schematics, diagrams) that
can be included in RunBundles. Tracks licensing information to ensure
compliance when distributing bundles.

Entirely deterministic — no LLM required.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .contracts import StrictBaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class AssetLicense(StrictBaseModel):
    """License metadata for an illustration asset."""

    asset_id: str
    source: str
    license_type: str  # e.g., "CC-BY-4.0", "internal", "public_domain"
    attribution: str
    url: Optional[str] = None
    date_added: str = ""


class AssetEntry(StrictBaseModel):
    """One managed illustration asset."""

    asset_id: str
    filename: str
    description: str
    modality: str
    asset_type: str  # "system_schematic", "element_diagram", "physics_illustration"
    license: AssetLicense


class AssetManifest(StrictBaseModel):
    """Complete manifest of illustration assets for a RunBundle."""

    version: str = "1.0"
    generated_at: str = ""
    assets: List[AssetEntry] = []
    illustrations_enabled: bool = True


# ---------------------------------------------------------------------------
# Asset Manager
# ---------------------------------------------------------------------------

class AssetManager:
    """Manage illustration assets for RunBundles.

    Provides a registry of modality-specific illustrations with licensing
    metadata. Can generate system schematics as simple SVG diagrams.

    The ``illustrations_enabled`` toggle allows users to disable all
    illustration generation (for minimal bundles or CI environments).
    """

    # Built-in modality schematics (descriptions for generation)
    MODALITY_SCHEMATICS: Dict[str, str] = {
        "cassi": "Coded aperture + disperser + detector array",
        "cacti": "Temporal coded aperture + detector",
        "ct": "X-ray source -> sample -> detector (rotate)",
        "mri": "RF coil -> gradient coils -> sample -> receiver",
        "oct": "Broadband source -> interferometer -> sample + reference",
        "light_field": "Microlens array -> sensor (4D capture)",
        "ptychography": "Coherent beam -> overlapping scan -> detector",
        "holography": "Laser -> beamsplitter -> sample + reference -> detector",
        "lensless": "LED -> coded mask -> sensor (no lens)",
        "spc": "DMD pattern -> single pixel detector",
        "dot": "NIR sources -> tissue -> detectors (diffuse light)",
        "photoacoustic": "Pulsed laser -> tissue -> ultrasound transducers",
        "flim": "Pulsed excitation -> sample -> time-resolved detector",
        "phase_retrieval": "Coherent beam -> sample -> far-field detector (no lens)",
        "integral": "Scene -> microlens array -> sensor (multi-view)",
        "fpm": "LED array -> sample -> low-NA objective -> camera",
    }

    def __init__(self, illustrations_enabled: bool = True):
        self.illustrations_enabled = illustrations_enabled

    def generate_manifest(
        self,
        modality_key: str,
        output_dir: Optional[str] = None,
    ) -> AssetManifest:
        """Generate an asset manifest for a given modality.

        Parameters
        ----------
        modality_key : str
            Modality identifier.
        output_dir : str, optional
            Directory to write manifest and assets.

        Returns
        -------
        AssetManifest
            Manifest listing all available assets.
        """
        manifest = AssetManifest(
            generated_at=datetime.utcnow().isoformat() + "Z",
            illustrations_enabled=self.illustrations_enabled,
        )

        if not self.illustrations_enabled:
            logger.info("Illustrations disabled — empty manifest.")
            return manifest

        # Generate system schematic
        if modality_key in self.MODALITY_SCHEMATICS:
            schematic = self._create_schematic_entry(modality_key)
            manifest.assets.append(schematic)

            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                svg_path = os.path.join(output_dir, schematic.filename)
                self._write_schematic_svg(
                    svg_path, modality_key,
                    self.MODALITY_SCHEMATICS[modality_key],
                )

        # Write manifest
        if output_dir is not None:
            manifest_path = os.path.join(output_dir, "asset_license.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest.model_dump(), f, indent=2)
            logger.info(f"Wrote asset manifest: {manifest_path}")

        return manifest

    def _create_schematic_entry(self, modality_key: str) -> AssetEntry:
        """Create an asset entry for a modality system schematic."""
        return AssetEntry(
            asset_id=f"{modality_key}_system_schematic",
            filename=f"system_schematic.svg",
            description=f"System schematic for {modality_key} imaging",
            modality=modality_key,
            asset_type="system_schematic",
            license=AssetLicense(
                asset_id=f"{modality_key}_system_schematic",
                source="pwm_core (auto-generated)",
                license_type="internal",
                attribution="Physics World Model",
                date_added=datetime.utcnow().strftime("%Y-%m-%d"),
            ),
        )

    @staticmethod
    def _write_schematic_svg(
        path: str,
        modality_key: str,
        description: str,
    ) -> None:
        """Write a simple SVG system schematic.

        Generates a minimal block diagram showing the imaging chain
        described in the modality's schematic description.
        """
        # Parse chain: "A -> B -> C" or "A + B + C"
        if "->" in description:
            blocks = [b.strip() for b in description.split("->")]
        elif "+" in description:
            blocks = [b.strip() for b in description.split("+")]
        else:
            blocks = [description.strip()]

        n = len(blocks)
        block_w = 120
        block_h = 40
        gap = 30
        total_w = n * block_w + (n - 1) * gap + 40
        total_h = block_h + 60

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">',
            f'  <text x="{total_w // 2}" y="18" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#333">'
            f'{modality_key.upper()} System</text>',
        ]

        y_top = 30
        for i, block in enumerate(blocks):
            x = 20 + i * (block_w + gap)
            # Box
            lines.append(
                f'  <rect x="{x}" y="{y_top}" width="{block_w}" '
                f'height="{block_h}" rx="5" fill="#e8f0fe" stroke="#4285f4" '
                f'stroke-width="1.5"/>'
            )
            # Label
            # Truncate long labels
            label = block[:16] + "..." if len(block) > 16 else block
            lines.append(
                f'  <text x="{x + block_w // 2}" y="{y_top + block_h // 2 + 4}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="10" '
                f'fill="#333">{label}</text>'
            )
            # Arrow to next
            if i < n - 1:
                ax = x + block_w
                ay = y_top + block_h // 2
                lines.append(
                    f'  <line x1="{ax + 2}" y1="{ay}" '
                    f'x2="{ax + gap - 2}" y2="{ay}" '
                    f'stroke="#666" stroke-width="1.5" '
                    f'marker-end="url(#arrow)"/>'
                )

        # Arrow marker
        lines.insert(1,
            '  <defs><marker id="arrow" markerWidth="8" markerHeight="6" '
            'refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" '
            'fill="#666"/></marker></defs>'
        )

        lines.append("</svg>")

        with open(path, "w") as f:
            f.write("\n".join(lines))
