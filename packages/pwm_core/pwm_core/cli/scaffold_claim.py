"""pwm_core.cli.scaffold_claim

``pwm scaffold-claim`` — fetch an arXiv paper and scaffold a Draft ClaimCard.

Usage
-----
    pwm scaffold-claim 2401.12345
    pwm scaffold-claim 2401.12345 --modality ct --out claims/

Fetches title/authors/abstract from the arXiv API, extracts claimed metrics
from the abstract text, and writes a Draft ClaimCard JSON to the output
directory.  Covers eess.IV, physics.optics, and cs.CV categories.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"

# Regex patterns to extract claimed metric values from abstract text
_PSNR_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(?:dB|db)\b", re.IGNORECASE
)
_SSIM_PATTERN = re.compile(
    r"SSIM\s+(?:of\s+)?(\d+\.\d+)", re.IGNORECASE
)
_METRIC_ALIASES = {
    "psnr": ["PSNR", "peak signal-to-noise ratio"],
    "ssim": ["SSIM", "structural similarity"],
    "nmse": ["NMSE", "normalized mean squared error"],
}


def add_scaffold_claim_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "scaffold-claim",
        help="Scaffold a Draft ClaimCard from an arXiv paper ID",
    )
    p.add_argument("arxiv_id", help="arXiv paper ID (e.g. 2401.12345)")
    p.add_argument("--modality", default="", help="Physics modality (e.g. ct, mri)")
    p.add_argument("--out", default="claims", help="Output directory for ClaimCard JSON")
    p.add_argument("--method-id", default="", help="MethodCard ID to link")
    p.add_argument("--spec-id", default="", help="SpecCard ID to link")
    p.set_defaults(func=cmd_scaffold_claim)


def cmd_scaffold_claim(args: argparse.Namespace) -> int:
    arxiv_id = args.arxiv_id.strip()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pwm scaffold-claim] Fetching arXiv:{arxiv_id} ...")

    paper = _fetch_arxiv(arxiv_id)
    if paper is None:
        print(f"[pwm scaffold-claim] ERROR: could not fetch arXiv:{arxiv_id}")
        return 1

    title = paper.get("title", "")
    authors = paper.get("authors", [])
    abstract = paper.get("abstract", "")
    categories = paper.get("categories", [])

    print(f"[pwm scaffold-claim] Title: {title}")
    print(f"[pwm scaffold-claim] Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
    print(f"[pwm scaffold-claim] Categories: {categories}")

    # Infer modality from categories + abstract if not specified
    modality = args.modality or _infer_modality(abstract, categories)

    # Extract claimed metrics
    claimed_metric, claimed_value = _extract_best_metric(abstract)

    # Build ClaimCard
    from pwm_core.cards.claim_card import ClaimCard
    from pwm_core.cards.utils import new_card_id, now_iso

    claim = ClaimCard(
        claim_id=f"arxiv_{arxiv_id.replace('.', '_')}_0",
        arxiv_id=arxiv_id,
        title=title,
        claimed_metric=claimed_metric,
        claimed_value=claimed_value,
        modality=modality,
        method_id=args.method_id or None,
        spec_id=args.spec_id or None,
        trust_tier="draft",
        authors=authors,
        tags=_infer_tags(abstract, categories),
        extra={
            "abstract_excerpt": abstract[:500],
            "arxiv_categories": categories,
            "scaffolded_by": "pwm scaffold-claim",
        },
    )

    out_path = out_dir / f"claim_arxiv_{arxiv_id.replace('.', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(claim.model_dump(), f, indent=2, default=str)

    print(f"[pwm scaffold-claim] ClaimCard written to {out_path}")
    print(f"  modality      = {modality or '(undetected)'}")
    print(f"  claimed_metric= {claimed_metric}")
    print(f"  claimed_value = {claimed_value}")
    print(f"  trust_tier    = draft")
    return 0


# ---------------------------------------------------------------------------
# arXiv API helpers
# ---------------------------------------------------------------------------

def _fetch_arxiv(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Fetch paper metadata from the arXiv Atom API."""
    url = ARXIV_API.format(arxiv_id=arxiv_id)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml = resp.read().decode("utf-8")
    except Exception as exc:
        logger.warning("arXiv fetch failed: %s", exc)
        return None

    return _parse_arxiv_atom(xml)


def _parse_arxiv_atom(xml: str) -> Optional[Dict[str, Any]]:
    """Parse the Atom XML response from arXiv into a flat dict."""
    import xml.etree.ElementTree as ET

    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    try:
        root = ET.fromstring(xml)
        entry = root.find("atom:entry", NS)
        if entry is None:
            return None

        title_el = entry.find("atom:title", NS)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""

        abstract_el = entry.find("atom:summary", NS)
        abstract = abstract_el.text.strip().replace("\n", " ") if abstract_el is not None else ""

        authors = [
            a.find("atom:name", NS).text.strip()
            for a in entry.findall("atom:author", NS)
            if a.find("atom:name", NS) is not None
        ]

        categories = [
            c.get("term", "")
            for c in entry.findall("atom:category", NS)
        ]

        return {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
        }
    except Exception as exc:
        logger.warning("arXiv XML parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Metric extraction + modality inference
# ---------------------------------------------------------------------------

def _extract_best_metric(abstract: str) -> Tuple[str, Optional[float]]:
    """Return (metric_name, value) for the highest PSNR claim in the abstract."""
    # Try PSNR first (most common in imaging papers)
    psnr_matches = _PSNR_PATTERN.findall(abstract)
    if psnr_matches:
        values = [float(v) for v in psnr_matches if 10.0 <= float(v) <= 60.0]
        if values:
            return "PSNR", max(values)

    # Try SSIM
    ssim_matches = _SSIM_PATTERN.findall(abstract)
    if ssim_matches:
        values = [float(v) for v in ssim_matches if 0.0 <= float(v) <= 1.0]
        if values:
            return "SSIM", max(values)

    return "PSNR", None


_MODALITY_KEYWORDS: Dict[str, List[str]] = {
    "ct": ["CT", "computed tomography", "X-ray CT", "Radon", "sinogram"],
    "mri": ["MRI", "magnetic resonance", "k-space", "undersampled MRI"],
    "cassi": ["CASSI", "coded aperture", "spectral imaging", "hyperspectral"],
    "cacti": ["CACTI", "video SCI", "snapshot compressive imaging"],
    "pet": ["PET", "positron emission"],
    "ultrasound": ["ultrasound", "US imaging", "acoustic"],
    "oct": ["OCT", "optical coherence tomography"],
}


def _infer_modality(abstract: str, categories: List[str]) -> str:
    abstract_lower = abstract.lower()
    for modality, keywords in _MODALITY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in abstract_lower:
                return modality
    return ""


def _infer_tags(abstract: str, categories: List[str]) -> List[str]:
    tags = list(categories)
    abstract_lower = abstract.lower()
    for kw in ["deep learning", "neural network", "unrolled", "plug-and-play",
               "diffusion model", "transformer", "iterative"]:
        if kw in abstract_lower:
            tags.append(kw.replace(" ", "_"))
    return tags[:8]
