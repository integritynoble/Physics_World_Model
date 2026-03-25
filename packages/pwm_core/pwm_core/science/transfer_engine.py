"""Cross-modality transfer engine.

Suggests algorithm and technique transfers between modalities based on:
- Shared primitive chain structure (OperatorGraph similarity)
- Domain diagnostic patterns (Triad/diagnostic decomposition similarity)
- Historical performance data (which algorithms generalize well)

Part of P3 (Scale) -- domain-diagnostic-based hypothesis generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransferSuggestion:
    """A suggested algorithm transfer between modalities."""

    source_modality: str
    target_modality: str
    algorithm: str
    confidence: float  # 0-1
    rationale: str
    shared_primitives: List[str]
    expected_benefit: str  # e.g., "PSNR improvement 2-5 dB"
    risk_factors: List[str]


class CrossModalityTransferEngine:
    """Suggests cross-modality algorithm transfers based on structural similarity.

    Loads primitive chain signatures from ``contrib/graph_templates.yaml`` and
    uses Jaccard similarity of those chains to rank transfer candidates.
    """

    def __init__(self) -> None:
        # Primitive chain signatures per modality (from graph_templates)
        self._modality_signatures: Dict[str, List[str]] = {}
        # Known successful transfers
        self._transfer_history: List[TransferSuggestion] = []
        self._load_signatures()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _load_signatures(self) -> None:
        """Load primitive chain signatures from graph templates."""
        try:
            import yaml
            from pathlib import Path

            # contrib/ sits next to the inner pwm_core/ package directory:
            #   packages/pwm_core/pwm_core/science/transfer_engine.py
            #   packages/pwm_core/contrib/graph_templates.yaml
            templates_path = (
                Path(__file__).resolve().parent.parent.parent
                / "contrib"
                / "graph_templates.yaml"
            )
            if not templates_path.exists():
                logger.debug("graph_templates.yaml not found at %s", templates_path)
                return

            with open(templates_path) as f:
                raw = yaml.safe_load(f) or {}

            # Top-level structure: {version: ..., templates: {key: {...}, ...}}
            templates = raw.get("templates", raw)

            for key, tmpl in templates.items():
                if not isinstance(tmpl, dict):
                    continue
                nodes = tmpl.get("nodes", [])
                primitives: List[str] = []
                for n in nodes:
                    if isinstance(n, dict):
                        prim = n.get("primitive_id", n.get("primitive", n.get("type", "unknown")))
                        primitives.append(prim)

                # Derive a short modality name by stripping the _graph_vN suffix
                mod = key
                for suffix in ("_graph_v1", "_graph_v2", "_graph_v3"):
                    mod = mod.replace(suffix, "")
                self._modality_signatures[mod] = primitives

            logger.debug(
                "Loaded %d modality signatures from %s",
                len(self._modality_signatures),
                templates_path,
            )
        except Exception as e:
            logger.debug("Could not load graph templates: %s", e)

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def compute_similarity(self, modality_a: str, modality_b: str) -> float:
        """Compute structural similarity between two modalities (0-1).

        Based on Jaccard similarity of their primitive chains.
        """
        sig_a = set(self._modality_signatures.get(modality_a, []))
        sig_b = set(self._modality_signatures.get(modality_b, []))
        if not sig_a or not sig_b:
            return 0.0
        union = sig_a | sig_b
        if not union:
            return 0.0
        return len(sig_a & sig_b) / len(union)

    def find_shared_primitives(self, modality_a: str, modality_b: str) -> List[str]:
        """Return primitives shared between two modalities."""
        sig_a = set(self._modality_signatures.get(modality_a, []))
        sig_b = set(self._modality_signatures.get(modality_b, []))
        return sorted(sig_a & sig_b)

    # ------------------------------------------------------------------
    # Transfer suggestions
    # ------------------------------------------------------------------

    def suggest_transfers(
        self,
        target_modality: str,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> List[TransferSuggestion]:
        """Suggest algorithm transfers from other modalities to *target_modality*.

        Ranks source modalities by structural similarity and generates
        concrete transfer suggestions with risk analysis.
        """
        similarities: List[Tuple[str, float]] = []
        for mod in self._modality_signatures:
            if mod == target_modality:
                continue
            sim = self.compute_similarity(mod, target_modality)
            if sim >= min_similarity:
                similarities.append((mod, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        suggestions: List[TransferSuggestion] = []
        for source_mod, sim in similarities[:top_k]:
            shared = self.find_shared_primitives(source_mod, target_modality)
            suggestion = TransferSuggestion(
                source_modality=source_mod,
                target_modality=target_modality,
                algorithm=f"best_from_{source_mod}",
                confidence=sim,
                rationale=(
                    f"Modalities share {len(shared)} primitive(s): {', '.join(shared)}. "
                    f"Algorithms optimized for {source_mod} may transfer to "
                    f"{target_modality} with adaptation of domain-specific parameters."
                ),
                shared_primitives=shared,
                expected_benefit=(
                    f"Potential PSNR improvement based on "
                    f"{sim:.0%} structural similarity"
                ),
                risk_factors=self._identify_risks(source_mod, target_modality, shared),
            )
            suggestions.append(suggestion)

        return suggestions

    # ------------------------------------------------------------------
    # Risk analysis
    # ------------------------------------------------------------------

    def _identify_risks(
        self, source: str, target: str, shared: List[str]
    ) -> List[str]:
        """Identify risk factors for a proposed transfer."""
        risks: List[str] = []
        sig_source = set(self._modality_signatures.get(source, []))
        sig_target = set(self._modality_signatures.get(target, []))

        unique_to_target = sig_target - sig_source
        if unique_to_target:
            risks.append(
                f"Target has primitives not in source: {', '.join(sorted(unique_to_target))}"
            )

        unique_to_source = sig_source - sig_target
        if unique_to_source:
            risks.append(
                f"Source has primitives not needed by target: "
                f"{', '.join(sorted(unique_to_source))}"
            )

        if len(shared) < 2:
            risks.append(
                "Low primitive overlap -- transfer may require significant adaptation"
            )

        return risks

    # ------------------------------------------------------------------
    # Bulk analysis
    # ------------------------------------------------------------------

    def get_all_similarities(self) -> Dict[Tuple[str, str], float]:
        """Return similarity matrix for all modality pairs."""
        result: Dict[Tuple[str, str], float] = {}
        mods = sorted(self._modality_signatures.keys())
        for i, a in enumerate(mods):
            for b in mods[i + 1 :]:
                sim = self.compute_similarity(a, b)
                result[(a, b)] = sim
        return result
