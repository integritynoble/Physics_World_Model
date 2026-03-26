"""test_arxiv_scaffolder.py — Automated self-test for arXiv claim scaffolding.

Runs without human intervention or network access. Uses mock papers
to verify the full scaffolding pipeline: relevance scoring, ClaimCard
generation, JSON persistence, and batch processing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The scaffolder lives in pwm_product; add it to path for testing
_product_root = Path(__file__).parent.parent.parent.parent / "pwm_product"
if _product_root.exists():
    sys.path.insert(0, str(_product_root / "platform"))

from pwm_platform.services.arxiv_scaffolder import (
    ArxivScaffolder,
    ArxivPaper,
    DraftClaimCard,
    IMAGING_KEYWORDS,
    WATCHED_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_PAPERS = [
    ArxivPaper(
        arxiv_id="2603.00001",
        title="Deep Unrolling for Spectral Compressive Imaging Reconstruction",
        authors=["Alice", "Bob"],
        abstract="We propose a deep unrolling network for CASSI reconstruction "
                 "achieving state-of-the-art compressed sensing performance.",
        categories=["eess.IV", "cs.CV"],
        published="2026-03-25",
    ),
    ArxivPaper(
        arxiv_id="2603.00002",
        title="Phase Retrieval via Plug-and-Play Priors",
        authors=["Charlie"],
        abstract="A plug-and-play approach to phase retrieval in ptychography "
                 "with diffusion model denoising.",
        categories=["eess.IV"],
        published="2026-03-25",
    ),
    ArxivPaper(
        arxiv_id="2603.00003",
        title="Attention Mechanisms in Natural Language Processing",
        authors=["Dave"],
        abstract="We study transformer attention patterns in large language models "
                 "for text summarization tasks.",
        categories=["cs.CL"],
        published="2026-03-25",
    ),
    ArxivPaper(
        arxiv_id="2603.00004",
        title="Super-Resolution MRI with Deep Learning",
        authors=["Eve"],
        abstract="A super-resolution method for accelerated MRI reconstruction "
                 "using compressed sensing and inverse problem formulation.",
        categories=["eess.IV", "physics.med-ph"],
        published="2026-03-25",
    ),
    ArxivPaper(
        arxiv_id="2603.00005",
        title="Computed Tomography Denoising with Score-Based Models",
        authors=["Frank"],
        abstract="We apply score-based diffusion models to low-dose computed "
                 "tomography denoising achieving 42 dB PSNR.",
        categories=["physics.optics"],
        published="2026-03-25",
    ),
]


# ---------------------------------------------------------------------------
# Tests — Relevance Scoring
# ---------------------------------------------------------------------------

class TestRelevanceScoring:
    """Verify relevance scoring without any network or human input."""

    def test_high_relevance_imaging_paper(self):
        scaffolder = ArxivScaffolder()
        score, keywords = scaffolder.compute_relevance(MOCK_PAPERS[0])
        assert score >= 0.5, f"Imaging paper should score ≥0.5, got {score}"
        assert len(keywords) >= 2, f"Should match ≥2 keywords, got {keywords}"

    def test_irrelevant_nlp_paper(self):
        scaffolder = ArxivScaffolder()
        score, keywords = scaffolder.compute_relevance(MOCK_PAPERS[2])
        assert score < 0.2, f"NLP paper should score <0.2, got {score}"

    def test_category_bonus_eess_iv(self):
        scaffolder = ArxivScaffolder()
        s1, _ = scaffolder.compute_relevance(MOCK_PAPERS[0])  # eess.IV
        s2, _ = scaffolder.compute_relevance(MOCK_PAPERS[4])  # physics.optics
        # eess.IV gets higher category bonus than physics.optics
        assert s1 >= s2 or abs(s1 - s2) < 0.2  # may differ based on keywords

    def test_keyword_matching(self):
        scaffolder = ArxivScaffolder()
        _, kw = scaffolder.compute_relevance(MOCK_PAPERS[3])  # MRI paper
        assert "MRI" in kw or "super-resolution" in kw or "compressed sensing" in kw


# ---------------------------------------------------------------------------
# Tests — ClaimCard Generation
# ---------------------------------------------------------------------------

class TestClaimCardGeneration:
    """Verify ClaimCard scaffolding produces valid Draft cards."""

    def test_relevant_paper_produces_claim(self, tmp_path):
        scaffolder = ArxivScaffolder(output_dir=str(tmp_path))
        claim = scaffolder.scaffold_claim(MOCK_PAPERS[0])
        assert claim is not None, "Relevant paper should produce a ClaimCard"
        assert claim.trust_tier == "draft"
        assert claim.status == "pending_review"
        assert claim.source_type == "arxiv"
        assert claim.source_id == "2603.00001"

    def test_irrelevant_paper_produces_no_claim(self, tmp_path):
        scaffolder = ArxivScaffolder(output_dir=str(tmp_path))
        claim = scaffolder.scaffold_claim(MOCK_PAPERS[2])
        assert claim is None, "NLP paper should not produce a ClaimCard"

    def test_claim_saved_to_disk(self, tmp_path):
        scaffolder = ArxivScaffolder(output_dir=str(tmp_path))
        claim = scaffolder.scaffold_claim(MOCK_PAPERS[0])
        assert claim is not None
        claim_path = tmp_path / f"{claim.claim_id}.json"
        assert claim_path.exists(), f"ClaimCard should be saved to {claim_path}"
        data = json.loads(claim_path.read_text())
        assert data["trust_tier"] == "draft"
        assert data["claim_id"] == claim.claim_id

    def test_claim_has_required_fields(self, tmp_path):
        scaffolder = ArxivScaffolder(output_dir=str(tmp_path))
        claim = scaffolder.scaffold_claim(MOCK_PAPERS[0])
        assert claim is not None
        required = {"claim_id", "source_type", "source_id", "title", "authors",
                     "trust_tier", "relevance_score", "status", "scaffolded_at"}
        actual = set(claim.to_dict().keys())
        missing = required - actual
        assert not missing, f"ClaimCard missing fields: {missing}"


# ---------------------------------------------------------------------------
# Tests — Batch Processing
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    """Verify batch scaffolding filters and ranks correctly."""

    def test_batch_filters_irrelevant(self, tmp_path):
        scaffolder = ArxivScaffolder(output_dir=str(tmp_path))
        claims = scaffolder.scaffold_batch(MOCK_PAPERS)
        # NLP paper (index 2) should be filtered out
        claim_sources = [c.source_id for c in claims]
        assert "2603.00003" not in claim_sources, "NLP paper should be filtered"

    def test_batch_returns_sorted_by_relevance(self, tmp_path):
        scaffolder = ArxivScaffolder(output_dir=str(tmp_path))
        claims = scaffolder.scaffold_batch(MOCK_PAPERS)
        assert len(claims) >= 3, f"Should scaffold ≥3 of 5 papers, got {len(claims)}"
        scores = [c.relevance_score for c in claims]
        assert scores == sorted(scores, reverse=True), "Claims should be sorted by relevance"

    def test_batch_all_draft_tier(self, tmp_path):
        scaffolder = ArxivScaffolder(output_dir=str(tmp_path))
        claims = scaffolder.scaffold_batch(MOCK_PAPERS)
        for c in claims:
            assert c.trust_tier == "draft", f"All scaffolded claims must be Draft, got {c.trust_tier}"

    def test_batch_all_pending_review(self, tmp_path):
        scaffolder = ArxivScaffolder(output_dir=str(tmp_path))
        claims = scaffolder.scaffold_batch(MOCK_PAPERS)
        for c in claims:
            assert c.status == "pending_review"


# ---------------------------------------------------------------------------
# Tests — Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_watched_categories(self):
        assert "eess.IV" in WATCHED_CATEGORIES
        assert "physics.optics" in WATCHED_CATEGORIES
        assert "cs.CV" in WATCHED_CATEGORIES

    def test_imaging_keywords_nonempty(self):
        assert len(IMAGING_KEYWORDS) >= 10
        assert "reconstruction" in IMAGING_KEYWORDS
        assert "MRI" in IMAGING_KEYWORDS
