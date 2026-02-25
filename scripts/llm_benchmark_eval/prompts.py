"""Prompt templates for B1 (Spec Selection) and B3 (Spec Validation)."""

from __future__ import annotations

import random
from typing import Any

# ---------------------------------------------------------------------------
# B1 — Spec Selection (Multiple Choice)
# ---------------------------------------------------------------------------

B1_SYSTEM = (
    "You are an expert in computational imaging and physics-based forward models. "
    "You will be given a natural-language description of an imaging system and four "
    "candidate spec strings (labelled A, B, C, D). Exactly one is correct.\n\n"
    "Reply with ONLY a JSON object: {\"answer\": \"<letter>\"}\n"
    "where <letter> is one of A, B, C, or D."
)


def build_b1_user_prompt(sample: dict[str, Any], seed: int | None = None) -> tuple[str, str]:
    """Build the B1 user prompt from a sample dict.

    Returns
    -------
    (user_prompt, correct_letter) — the prompt text and the correct answer letter.
    """
    choices = [sample["correct_spec"]] + sample["distractors"]
    rng = random.Random(seed if seed is not None else sample["id"])
    rng.shuffle(choices)

    correct_letter = "ABCD"[choices.index(sample["correct_spec"])]
    labels = "ABCD"

    lines = [
        "### Imaging System Description",
        sample["description"],
        "",
        "### Candidate Specs",
    ]
    for letter, spec in zip(labels, choices):
        lines.append(f"{letter}. {spec}")

    lines.append("")
    lines.append("Which spec correctly describes this imaging system?")

    return "\n".join(lines), correct_letter


# ---------------------------------------------------------------------------
# B3 — Spec Validation (Binary Classification)
# ---------------------------------------------------------------------------

B3_SYSTEM = (
    "You are an expert in physics-based forward models. "
    "You will be given a candidate spec and the true spec of an imaging system. "
    "Determine whether the candidate spec matches the true spec.\n\n"
    "Reply with ONLY a JSON object: {\"answer\": \"match\"} or {\"answer\": \"no_match\"}"
)


def build_b3_user_prompt(sample: dict[str, Any]) -> tuple[str, str]:
    """Build the B3 user prompt from a sample dict.

    Returns
    -------
    (user_prompt, correct_label) — the prompt text and the correct label.
    """
    lines = [
        "### True Spec",
        sample["true_spec"],
        "",
        "### Candidate Spec",
        sample["candidate_spec"],
        "",
        "Does the candidate spec match the true spec?",
    ]
    return "\n".join(lines), sample["label"]
