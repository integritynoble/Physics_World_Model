"""Validate modality reports at pwm/reports/*.md against the report contract.

If no reports exist the parametrize list is empty and zero tests are collected
(no failures).
"""

import glob
import re
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_DIR = Path(__file__).resolve().parent.parent / "pwm" / "reports"

REQUIRED_HEADINGS = [
    "## Modality overview",
    "## Standard dataset",
    "## PWM pipeline flowchart (mandatory)",
    "## Element inventory & mismatch parameters",
    "## Node-by-node trace (one sample)",
    "## Workflow W1: Prompt-driven simulation + reconstruction",
    "## Workflow W2: Operator correction mode (measured y + operator A)",
    "## Test results summary",
    "## Reproducibility",
    "## Saved artifacts",
    "## Next actions",
]

VALID_A_DEFINITIONS = {"matrix", "sparse", "callable", "LinearOperator"}
VALID_A_METHODS = {"graph_stripped", "provided", "linearized"}

# ---------------------------------------------------------------------------
# Helpers — collection
# ---------------------------------------------------------------------------


def _collect_reports():
    """Return sorted list of *.md report paths (excluding scoreboard etc.)."""
    pattern = str(REPORT_DIR / "*.md")
    return sorted(glob.glob(pattern))


# ---------------------------------------------------------------------------
# Helpers — text extraction
# ---------------------------------------------------------------------------


def _read_report(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _heading_line(heading: str) -> str:
    """Normalise a heading for comparison (strip trailing whitespace)."""
    return heading.strip()


def extract_section(text: str, heading: str) -> str:
    """Extract the body text under *heading* up to the next heading of equal
    or higher level.  Returns empty string if heading not found."""
    level = len(heading) - len(heading.lstrip("#"))
    # Build regex: match the heading, capture everything until next heading of
    # same-or-higher level (or EOF).
    escaped = re.escape(heading.strip())
    pattern = rf"^{escaped}\s*\n(.*?)(?=^#{{{1},{level}}} |\Z)"
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return m.group(1) if m else ""


def extract_flowchart_block(text: str) -> list[str]:
    """Return non-blank lines inside the flowchart section."""
    section = extract_section(text, "## PWM pipeline flowchart (mandatory)")
    # Flowchart is typically inside a fenced code block or plain text lines.
    # Strip fenced-code markers if present.
    lines: list[str] = []
    in_fence = False
    for raw in section.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or (not in_fence and stripped):
            # When inside a fence keep all lines (including blank for ordering);
            # outside a fence only keep non-blank lines.
            if stripped:
                lines.append(stripped)
    return lines


def parse_table_rows(section_text: str) -> list[list[str]]:
    """Parse markdown tables in *section_text*.
    Returns a list of data rows (each row is a list of cell strings).
    The header row and separator row are excluded."""
    rows: list[list[str]] = []
    header_seen = False
    separator_seen = False
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            # Reset if we hit non-table content after a table
            if header_seen and separator_seen:
                break
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not header_seen:
            header_seen = True
            continue  # skip header
        if not separator_seen:
            # separator row looks like |---|---|
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                separator_seen = True
                continue
        if header_seen and separator_seen:
            rows.append(cells)
    return rows


def parse_table_header(section_text: str) -> list[str]:
    """Return the column names of the first markdown table in *section_text*."""
    for line in section_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            return [c.strip() for c in stripped.strip("|").split("|")]
    return []


# ---------------------------------------------------------------------------
# Flowchart pattern validators
# ---------------------------------------------------------------------------

_ELEMENT_RE = re.compile(
    r"Element\s+(\d+)\s+\(subrole=(transport|interaction|encoding|transduction)\):"
)
_ARROW_RE = re.compile(r"[↓⬇]")


def _element_numbers(lines: list[str]) -> list[int]:
    nums = []
    for ln in lines:
        m = _ELEMENT_RE.search(ln)
        if m:
            nums.append(int(m.group(1)))
    return nums


def _line_index(lines: list[str], predicate) -> int | None:
    for i, ln in enumerate(lines):
        if predicate(ln):
            return i
    return None


def check_pattern_a(lines: list[str]) -> list[str]:
    """Validate Pattern A (linear chain).  Returns list of error strings."""
    errors: list[str] = []
    if not lines:
        errors.append("Flowchart is empty")
        return errors

    # First non-blank line == "x (world)"
    if lines[0] != "x (world)":
        errors.append(f"Pattern A: first line must be 'x (world)', got '{lines[0]}'")

    # Last non-blank line == "y"
    if lines[-1] != "y":
        errors.append(f"Pattern A: last line must be 'y', got '{lines[-1]}'")

    # Exactly 1 SourceNode
    src_lines = [i for i, ln in enumerate(lines) if ln.startswith("SourceNode:")]
    if len(src_lines) != 1:
        errors.append(f"Pattern A: expected 1 SourceNode, found {len(src_lines)}")

    # Elements: >=1, sequential from 1
    elem_nums = _element_numbers(lines)
    if len(elem_nums) < 1:
        errors.append("Pattern A: need >=1 Element lines")
    elif elem_nums != list(range(1, len(elem_nums) + 1)):
        errors.append(f"Pattern A: element numbers not sequential from 1: {elem_nums}")

    # Exactly 1 SensorNode
    sensor_lines = [i for i, ln in enumerate(lines) if ln.startswith("SensorNode:")]
    if len(sensor_lines) != 1:
        errors.append(f"Pattern A: expected 1 SensorNode, found {len(sensor_lines)}")

    # Exactly 1 NoiseNode
    noise_lines = [i for i, ln in enumerate(lines) if ln.startswith("NoiseNode:")]
    if len(noise_lines) != 1:
        errors.append(f"Pattern A: expected 1 NoiseNode, found {len(noise_lines)}")

    # Ordering: x(world) < SourceNode < Elements < SensorNode < NoiseNode < y
    if src_lines and elem_nums and sensor_lines and noise_lines:
        first_elem_idx = _line_index(lines, lambda ln: _ELEMENT_RE.search(ln))
        last_elem_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if _ELEMENT_RE.search(lines[i]):
                last_elem_idx = i
                break
        order_ok = (
            0 < src_lines[0] < first_elem_idx <= last_elem_idx
            < sensor_lines[0] < noise_lines[0] < len(lines) - 1
        )
        if not order_ok:
            errors.append(
                "Pattern A: ordering violated "
                "(x < SourceNode < Elements < SensorNode < NoiseNode < y)"
            )

    # Arrows between stages
    if not any(_ARROW_RE.search(ln) for ln in lines):
        errors.append("Pattern A: no arrow characters (↓) found between stages")

    return errors


def check_pattern_b(lines: list[str]) -> list[str]:
    """Validate Pattern B (branch entry).  Returns list of error strings."""
    errors: list[str] = []
    if not lines:
        errors.append("Flowchart is empty")
        return errors

    # First non-blank line starts with "SourceNode:"
    if not lines[0].startswith("SourceNode:"):
        errors.append(
            f"Pattern B: first line must start with 'SourceNode:', got '{lines[0]}'"
        )

    # Exactly 1 line containing "← x (world) enters here"
    branch_indices = [
        i for i, ln in enumerate(lines) if "x (world) enters here" in ln
    ]
    if len(branch_indices) != 1:
        errors.append(
            f"Pattern B: expected exactly 1 branch-entry line, found {len(branch_indices)}"
        )

    # Branch-entry before first Element
    first_elem_idx = _line_index(lines, lambda ln: _ELEMENT_RE.search(ln))
    if branch_indices and first_elem_idx is not None:
        if branch_indices[0] >= first_elem_idx:
            errors.append("Pattern B: branch-entry must come BEFORE first Element line")
    elif first_elem_idx is None:
        errors.append("Pattern B: no Element lines found")

    # First element must have subrole=interaction
    if first_elem_idx is not None:
        m = _ELEMENT_RE.search(lines[first_elem_idx])
        if m and m.group(2) != "interaction":
            errors.append(
                f"Pattern B: first Element must have subrole=interaction, "
                f"got subrole={m.group(2)}"
            )

    # Elements >=1, sequential from 1
    elem_nums = _element_numbers(lines)
    if len(elem_nums) < 1:
        errors.append("Pattern B: need >=1 Element lines")
    elif elem_nums != list(range(1, len(elem_nums) + 1)):
        errors.append(f"Pattern B: element numbers not sequential from 1: {elem_nums}")

    # Exactly 1 SensorNode, 1 NoiseNode
    sensor_lines = [i for i, ln in enumerate(lines) if ln.startswith("SensorNode:")]
    if len(sensor_lines) != 1:
        errors.append(f"Pattern B: expected 1 SensorNode, found {len(sensor_lines)}")

    noise_lines = [i for i, ln in enumerate(lines) if ln.startswith("NoiseNode:")]
    if len(noise_lines) != 1:
        errors.append(f"Pattern B: expected 1 NoiseNode, found {len(noise_lines)}")

    # Last non-blank line == "y"
    if lines[-1] != "y":
        errors.append(f"Pattern B: last line must be 'y', got '{lines[-1]}'")

    # Order: SourceNode < branch-entry < Elements < SensorNode < NoiseNode < y
    if sensor_lines and noise_lines and branch_indices and first_elem_idx is not None:
        last_elem_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if _ELEMENT_RE.search(lines[i]):
                last_elem_idx = i
                break
        order_ok = (
            0 < branch_indices[0] < first_elem_idx
            and last_elem_idx < sensor_lines[0] < noise_lines[0] < len(lines) - 1
        )
        if not order_ok:
            errors.append(
                "Pattern B: ordering violated "
                "(SourceNode < branch-entry < Elements < SensorNode < NoiseNode < y)"
            )

    return errors


# ---------------------------------------------------------------------------
# Test class — parametrized over discovered reports.
# When _REPORTS is empty the parametrize list is empty and pytest marks
# the tests as skipped (zero failures, nothing to validate).
# ---------------------------------------------------------------------------

_REPORTS = _collect_reports()


@pytest.mark.parametrize(
    "report_path", _REPORTS, ids=lambda p: Path(p).stem
)
class TestReportContract:
    """Contract tests for each modality report."""

    # 1. Required headings ------------------------------------------------

    def test_required_headings(self, report_path):
        text = _read_report(report_path)
        found_headings = {
            ln.strip() for ln in text.splitlines() if ln.strip().startswith("## ")
        }
        for heading in REQUIRED_HEADINGS:
            assert heading in found_headings, (
                f"Missing required heading: '{heading}'"
            )

    # 2. Flowchart Pattern A or B -----------------------------------------

    def test_flowchart_pattern_a_or_b(self, report_path):
        text = _read_report(report_path)
        lines = extract_flowchart_block(text)
        assert lines, "Flowchart section is empty or missing"

        errors_a = check_pattern_a(lines)
        errors_b = check_pattern_b(lines)

        if errors_a and errors_b:
            msg = (
                "Flowchart matches neither Pattern A nor Pattern B.\n"
                f"  Pattern A errors: {errors_a}\n"
                f"  Pattern B errors: {errors_b}"
            )
            pytest.fail(msg)

    # 3. Element inventory ------------------------------------------------

    def test_element_inventory(self, report_path):
        text = _read_report(report_path)
        section = extract_section(text, "## Element inventory & mismatch parameters")
        assert section.strip(), "Element inventory section is empty"

        header = parse_table_header(section)
        header_lower = [h.lower() for h in header]
        for col in ("node_id", "primitive_id", "subrole"):
            assert col in header_lower, f"Missing column '{col}' in element inventory"

        rows = parse_table_rows(section)
        assert len(rows) >= 1, "Element inventory table must have >=1 data row"

        # At least one row with a non-empty mismatch knob (not "—" or empty)
        has_knob = False
        for row in rows:
            # Mismatch knob is typically the last column(s) after subrole
            for cell in row:
                if cell and cell not in ("\u2014", "—", "-", ""):
                    has_knob = True
                    break
            if has_knob:
                break
        # Check explicitly for a mismatch-related column having content
        # Look for any column beyond the base three that has real content
        mismatch_col_idx = None
        for i, h in enumerate(header_lower):
            if "mismatch" in h or "knob" in h:
                mismatch_col_idx = i
                break
        if mismatch_col_idx is not None:
            has_knob = any(
                len(row) > mismatch_col_idx
                and row[mismatch_col_idx] not in ("\u2014", "—", "-", "")
                for row in rows
            )
        assert has_knob, (
            "At least one row must have a non-empty mismatch knob (not '—')"
        )

    # 4. Node-by-node trace -----------------------------------------------

    def test_node_trace(self, report_path):
        text = _read_report(report_path)
        section = extract_section(text, "## Node-by-node trace (one sample)")
        assert section.strip(), "Node-by-node trace section is empty"

        rows = parse_table_rows(section)
        assert len(rows) >= 3, (
            f"Node-by-node trace table must have >=3 data rows, found {len(rows)}"
        )

        # Each row should reference an artifact path
        artifact_re = re.compile(r"artifacts/trace/\d+_")
        for i, row in enumerate(rows):
            row_text = " | ".join(row)
            assert artifact_re.search(row_text), (
                f"Trace row {i} missing artifact_path matching artifacts/trace/\\d+_"
            )

        # Section mentions png/PNG
        assert re.search(r"png", section, re.IGNORECASE), (
            "Node-by-node trace section must mention 'png' or 'PNG'"
        )

    # 5. W1 content -------------------------------------------------------

    def test_w1_content(self, report_path):
        text = _read_report(report_path)
        section = extract_section(
            text,
            "## Workflow W1: Prompt-driven simulation + reconstruction",
        )
        assert section.strip(), "W1 section is empty"

        assert "Prompt used:" in section, "W1 must contain 'Prompt used:'"
        # Check there is actual text after "Prompt used:"
        m = re.search(r"Prompt used:\s*(.+)", section)
        assert m and m.group(1).strip(), "W1 'Prompt used:' must have text after it"

        assert "Mode S results" in section, "W1 must contain 'Mode S results'"
        assert "Mode I results" in section, "W1 must contain 'Mode I results'"

        # Metrics table with PSNR/SSIM
        assert re.search(r"PSNR", section), "W1 must contain PSNR in metrics table"
        assert re.search(r"SSIM", section), "W1 must contain SSIM in metrics table"

    # 6. W2 operator spec -------------------------------------------------

    def test_w2_operator_spec(self, report_path):
        text = _read_report(report_path)
        section = extract_section(
            text,
            "## Workflow W2: Operator correction mode (measured y + operator A)",
        )
        assert section.strip(), "W2 section is empty"

        # A_definition
        m_def = re.search(r"A_definition:\s*(\S+)", section)
        assert m_def, "W2 must contain 'A_definition:'"
        assert m_def.group(1) in VALID_A_DEFINITIONS, (
            f"A_definition must be one of {VALID_A_DEFINITIONS}, "
            f"got '{m_def.group(1)}'"
        )

        # A_extraction_method
        m_method = re.search(r"A_extraction_method:\s*(\S+)", section)
        assert m_method, "W2 must contain 'A_extraction_method:'"
        assert m_method.group(1) in VALID_A_METHODS, (
            f"A_extraction_method must be one of {VALID_A_METHODS}, "
            f"got '{m_method.group(1)}'"
        )

        # If linearized, Notes must not be N/A
        if m_method.group(1) == "linearized":
            m_notes = re.search(
                r"Notes\s*\(if linearized\):\s*(.+)", section
            )
            assert m_notes, (
                "W2 must contain 'Notes (if linearized):' when method is linearized"
            )
            assert m_notes.group(1).strip() != "N/A", (
                "Notes (if linearized) must not be 'N/A' when method is linearized"
            )

        # A_sha256
        assert re.search(r"A_sha256:", section), "W2 must contain 'A_sha256:'"

        # Comparison table with A0 and A' columns
        header = parse_table_header(section)
        header_text = " ".join(header)
        assert re.search(r"A[₀0]", header_text), (
            "W2 comparison table must have an A₀ (or A0) column"
        )
        assert re.search(r"A['\u2032]", header_text), (
            "W2 comparison table must have an A' column"
        )

    def test_w2_nll_fields(self, report_path):
        text = _read_report(report_path)
        section = extract_section(
            text,
            "## Workflow W2: Operator correction mode (measured y + operator A)",
        )
        assert section.strip(), "W2 section is empty"

        # NLL before correction
        m_before = re.search(r"NLL before correction:\s*([\d.eE+\-]+)", section)
        assert m_before, "W2 must contain 'NLL before correction:' with a number"
        float(m_before.group(1))  # must parse as float

        # NLL after correction
        m_after = re.search(r"NLL after correction:\s*([\d.eE+\-]+)", section)
        assert m_after, "W2 must contain 'NLL after correction:' with a number"
        float(m_after.group(1))  # must parse as float

    # 7. Test results summary ---------------------------------------------

    def test_results_summary_quick_and_full(self, report_path):
        text = _read_report(report_path)
        section = extract_section(text, "## Test results summary")
        assert section.strip(), "Test results summary section is empty"

        # Quick gate with >=6 checklist items
        assert "### Quick gate" in section, "Missing '### Quick gate' sub-heading"
        quick_section = section.split("### Quick gate", 1)[1]
        # Stop at next ### if any
        if "### " in quick_section.split("\n", 1)[-1]:
            quick_section = quick_section.split("### ", 1)[0]
        checklist_items = re.findall(r"^\s*[-*]\s*\[[ xX]\]", quick_section, re.MULTILINE)
        assert len(checklist_items) >= 6, (
            f"Quick gate must have >=6 checklist items, found {len(checklist_items)}"
        )

        # Full metrics with table
        assert "### Full metrics" in section, "Missing '### Full metrics' sub-heading"
        full_section = section.split("### Full metrics", 1)[1]
        rows = parse_table_rows(full_section)
        assert len(rows) >= 5, (
            f"Full metrics table must have >=5 rows, found {len(rows)}"
        )

        # Check expected columns
        full_header = parse_table_header(full_section)
        full_header_lower = [h.lower() for h in full_header]
        for expected_col in ("check", "metric", "value", "threshold", "status"):
            assert expected_col in full_header_lower, (
                f"Full metrics table missing column '{expected_col}', "
                f"header: {full_header}"
            )

    # 8. Reproducibility --------------------------------------------------

    def test_reproducibility(self, report_path):
        text = _read_report(report_path)
        section = extract_section(text, "## Reproducibility")
        assert section.strip(), "Reproducibility section is empty"

        # Seed with integer
        m_seed = re.search(r"Seed:\s*(\d+)", section)
        assert m_seed, "Reproducibility must contain 'Seed:' with an integer"

        # PWM version non-empty
        m_ver = re.search(r"PWM version:\s*(\S+)", section)
        assert m_ver, "Reproducibility must contain 'PWM version:' with a value"

        # Output hash
        assert re.search(r"Output hash", section, re.IGNORECASE), (
            "Reproducibility must mention 'Output hash'"
        )
        # At least one hash-like value (hex string)
        assert re.search(r"[0-9a-fA-F]{8,}", section), (
            "Reproducibility must contain at least one hash value"
        )

        # pwm_cli run command
        assert "pwm_cli run" in section, (
            "Reproducibility must contain a 'pwm_cli run' command"
        )

    # 9. Saved artifacts --------------------------------------------------

    def test_saved_artifacts_section(self, report_path):
        text = _read_report(report_path)
        section = extract_section(text, "## Saved artifacts")
        assert section.strip(), "Saved artifacts section is empty"
