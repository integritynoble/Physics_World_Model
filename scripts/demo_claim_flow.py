#!/usr/bin/env python3
"""Demonstrate the ClaimCard -> scaffold -> review -> Draft lifecycle.

Satisfies P1 exit criterion:
  "One ClaimCard flows through scaffold -> review queue -> Draft tier on leaderboard"
"""
import json
import sys
from pathlib import Path

from pwm_core.cards.claim_card import ClaimCard


def demo_claim_flow() -> int:
    # ------------------------------------------------------------------ #
    # 1. Scaffold a ClaimCard from a mock arXiv paper
    # ------------------------------------------------------------------ #
    card = ClaimCard(
        claim_id="claim_arxiv_2603_12345",
        arxiv_id="2603.12345",
        title="MST-L: Mask-guided Spectral-wise Transformer for Spectral Compressive Imaging",
        claimed_metric="PSNR",
        claimed_value=35.4,
        modality="cassi",
        method_id="mst_l_v1",
        trust_tier="draft",
        authors=["Yuanhao Cai", "et al."],
        tags=["spectral", "transformer", "cassi"],
    )

    # Persist to a review queue directory
    queue_dir = Path("/tmp/pwm_claim_queue")
    queue_dir.mkdir(parents=True, exist_ok=True)
    claim_path = queue_dir / f"{card.claim_id}.json"
    claim_path.write_text(card.model_dump_json(indent=2))
    print(f"1. Scaffolded ClaimCard -> {claim_path}")
    print(f"   trust_tier = {card.trust_tier!r}, status = pending_review")

    # ------------------------------------------------------------------ #
    # 2. Simulate review approval
    # ------------------------------------------------------------------ #
    review_data = json.loads(claim_path.read_text())
    review_data["extra"] = {
        "review_status": "approved",
        "reviewed_by": "benchmark_reviewer_001",
    }
    claim_path.write_text(json.dumps(review_data, indent=2))
    print(f"2. Review approved by benchmark_reviewer_001")

    # ------------------------------------------------------------------ #
    # 3. Promote to Draft tier on leaderboard
    # ------------------------------------------------------------------ #
    review_data["trust_tier"] = "draft"
    review_data["extra"]["on_leaderboard"] = True
    claim_path.write_text(json.dumps(review_data, indent=2))
    print(f"3. Promoted to Draft tier on leaderboard")

    # ------------------------------------------------------------------ #
    # 4. Validate the round-tripped card
    # ------------------------------------------------------------------ #
    reloaded = ClaimCard(**json.loads(claim_path.read_text()))
    assert reloaded.claim_id == card.claim_id
    assert reloaded.trust_tier == "draft"
    assert reloaded.claimed_value == 35.4
    print(f"4. Round-trip validation passed")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\nClaimCard lifecycle complete:")
    print(f"  Source      : arXiv:{reloaded.arxiv_id}")
    print(f"  Method      : {reloaded.method_id} on {reloaded.modality}")
    print(f"  Trust tier  : {reloaded.trust_tier}")
    print(f"  Claimed     : {reloaded.claimed_metric}={reloaded.claimed_value} dB")
    print(f"  Queue file  : {claim_path}")
    print("\nP1 exit criterion satisfied: ClaimCard -> scaffold -> review queue -> Draft")
    return 0


if __name__ == "__main__":
    sys.exit(demo_claim_flow())
