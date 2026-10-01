# AI4Science PWM mode, v1 (test ledger)

Two add-ons OpenCode loads (AI4SCI-07):

1. **MCP server** `python -m pwm_mode.mcp_server` (stdio). It exposes the low-dose CT research agent (`ldct_solve`), its tools (`ldct_problem`, `ldct_field`), its judge (`ldct_judge`) and the reused `mcp_pwm.py` tools (`pwm_validate`, `pwm_judge_cassi`, `pwm_status`). The agent itself is served as the MCP prompt `low_dose_ct_agent`. Free calls: `pwm_catalog`, `pwm_quote`, `pwm_authorize`, `pwm_balance`, `pwm_receipt`, `ldct_describe`, `ldct_field_schemas`.
2. **PWM model proxy** `python -m pwm_mode.proxy` (OpenAI-compatible, `http://127.0.0.1:8795/v1`). It serves `claude-opus-5-5` (Anthropic SDK upstream) and `qwen3.8:27b` (the fleet's own GPUs).

Every paid use is one run on the ledger:

- quote → reserve → work → **settle** (delivered) or **release** (failed: 0 PWM);
- a settled run pays the service's agreed beneficiaries, with 3% to the treasury, and is final;
- one receipt per run, with a sha256 hash.

Model turns are metered, charged `min(usage, cap)`, and any excess is recorded as `over_cap`. Your own LLM is configured directly in OpenCode, never through the proxy, so it costs 0 PWM.

**Test ledger only.** `Ledger(network != "test")` refuses to start. Prices and beneficiaries in `config/services.json` are test placeholders until the owner sets them. There is no live charging before gate G3 and the owner's approval.

```bash
python3 -m venv .venv && .venv/bin/pip install mcp numpy scipy scikit-image pyyaml httpx anthropic uvicorn starlette pytest
export PYTHONPATH=$PWD                                 # runtime path: AI4SCIENCE_SRC (default: spiritai's checkout)
.venv/bin/python -m pwm_mode.admin fund user:test-owner 20
.venv/bin/python -m pwm_mode.admin new-key user:test-owner > config/.owner_test_key
.venv/bin/python -m pwm_mode.proxy &
cd e2e && opencode run -m ai4science-pwm/qwen3.8:27b "pwm_authorize 3 PWM, then ldct_solve seed 0; report the receipt"
.venv/bin/python -m pytest -q tests
```

`e2e/opencode.json` shows the OpenCode wiring: one provider and one MCP server. `e2e-fail/` has solver failure injected. `e2e-own/` uses the user's own LLM.
