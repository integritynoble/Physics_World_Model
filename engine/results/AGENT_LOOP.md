# The real agent loop (common vs PWM mode, OpenCode + model), 1 Oct 2026, agent-prod-restored: NOT RUN (blocked)

**Result for the paper: there is no agent-loop result yet.** The run stopped at its preflight: the only model
this host can give OpenCode, the fleet's Qwen (https://physicsworldmodel.org/qwen/v1), has answered HTTP 502 on
every request since 28 Sep (rechecked 1 Oct, 20:42 UTC). OpenCode has no other credentials here. The harness
wrote `agent_loop_blocked.jsonl` = one "blocked" record and claims nothing. No problem was attempted, no PWM moved.

What did run:

1. **RUN1 reproduced on a second machine.** `engine/run_engine.py` on agent-prod (8 workers, 145 s):
   identical held-out results (common mode 46/54; PWM mode L0 46, L3 49, L5 52, L8 53/54) and identical library
   state digests to the integrity-wsl run (`run1_repro_agentprod_report.json`). The scripted result is
   deterministic across machines.
2. **The agent-loop harness** (`engine/agent_loop.py`), ready to run:
   - both arms use the same model, prompt core, workspace (public data, the runtime's solver, the adopted
     defaults) and time limit; the PWM arm adds only the ai4science-pwm MCP server,
     reading the run's latest library state (L8 for run 1, L9 for run 3), on a run-local test ledger
     (never the production ledger), with a 1 PWM cap per problem;
   - scoring is outside the agent (validator-ldct-1 against the withheld key); a missing file is a fail;
   - it records per problem: pass, metrics, the agent's tool calls, seconds and PWM spent.
   Harness tests (`tests/test_agent_loop.py`, no model): the arms' configs differ only by the MCP server; the key
   never enters a workspace; outside scoring reproduces RUN1's common-mode verdicts and CNR exactly; an
   unreachable model writes "blocked" and nothing else.

Target: run 3 (qualified chest, 162 held-out problems, 324 agent sessions). Its corpus (ldct-v2, 24 TCIA
patients) is on integrity-wsl, not here; run it there, or fetch the corpus here first. Run 1 (54 problems,
108 sessions) runs here as is. When a model is available:

    python -m engine.agent_loop --run ~/.ai4science-pwm-mode/run1-repro --out <dir> \
        [--base <openai-compatible URL> --key-file <file> --model <id>]

What the paper may say today: the scripted engine results (runs 1-3) and run 1's exact reproduction on a second
machine.
It may not say anything about agents using the library.
