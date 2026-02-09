# PWM Project Makefile
# Run `make check` to verify project integrity before and after parallel work.

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

PKG_DIR     = packages/pwm_core
TESTS_DIR   = $(PKG_DIR)/tests
BENCH_DIR   = $(PKG_DIR)/benchmarks
AGENTS_DIR  = $(PKG_DIR)/pwm_core/agents

.PHONY: check test test-unit test-correction check-literals help

# --- Main target: run all checks ---
check: test-unit test-correction check-literals
	@echo ""
	@echo "========================================="
	@echo "  All checks passed."
	@echo "========================================="

# --- Unit tests ---
test-unit:
	$(PYTEST) $(TESTS_DIR)/ -x -q

# --- Operator correction regression tests ---
test-correction:
	$(PYTEST) $(BENCH_DIR)/test_operator_correction.py -x -q

# --- Verify generated literals are up-to-date ---
check-literals:
	$(PYTHON) $(AGENTS_DIR)/_generate_literals.py --check

# --- Run all tests (alias) ---
test: check

# --- Help ---
help:
	@echo "PWM Makefile targets:"
	@echo "  make check           - Run all checks (unit + correction + literals)"
	@echo "  make test-unit       - Run unit tests only"
	@echo "  make test-correction - Run operator correction tests only"
	@echo "  make check-literals  - Verify generated literals are up-to-date"
	@echo "  make help            - Show this message"
