"""AI4Science PWM-mode MCP server (stdio): research agents, tools and judge, paid in PWM.

OpenCode (or any MCP host) starts it as a local server. Every paid call is one run on the
test ledger: quote -> reserve -> do the work -> settle if delivered, release if not. A
settled run pays the service's agreed beneficiaries with 3% to the treasury and is final;
a failed run is released and costs 0 PWM. Each run returns its receipt.

Consent: nothing is charged until the user authorises a session cap with pwm_authorize
(an OpenCode permission prompt). Paid calls then draw quotes inside that cap. A call
with an explicit quote_id uses that quote instead.

Reused from the AI4Science runtime: the LDCT benchmark, scorer and judge (see ldct.py),
the low-dose-ct agent's system prompt and field-store tools, and mcp_pwm.py's
pwm_validate and pwm_judge_cassi coroutines, called with the workspace pinned here,
never taken from the model (as harness/mcp_pwm.py does).
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from . import catalog, ldct
from .ledger import Ledger, LedgerError, micro, pwm
from .library import Library, LibraryError

WORKSPACE = Path(os.environ.get("PWM_MODE_WORKSPACE", os.getcwd())).resolve()
PAYER = os.environ.get("PWM_MODE_PAYER", "user:test-owner")

server = MCPServer(
    name="ai4science-pwm",
    version="0.1.0",
    instructions=(
        "AI4Science PWM mode (TEST LEDGER). Research agents, tools and judges priced in PWM. "
        "Start with pwm_catalog, then pwm_authorize a session cap the user agrees to. "
        "Every paid call returns a receipt; failed runs cost 0 PWM; settled runs are final. "
        "The judge's verdict is the source of truth: report it, never override it."),
)

_ledger: Optional[Ledger] = None
_session = {"cap_micro": 0, "spent_micro": 0, "id": "s_" + uuid.uuid4().hex[:10]}


def ledger() -> Ledger:
    global _ledger
    if _ledger is None:
        _ledger = Ledger(catalog.ledger_path())
    return _ledger


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def _in_workspace(path: str) -> Path:
    p = (WORKSPACE / path).resolve()
    if not p.is_relative_to(WORKSPACE):
        raise ValueError("path escapes the workspace: %s" % path)
    return p


def _charged(service_id: str, quote_id: Optional[str], work: Callable[[], Dict[str, Any]],
             delivered: Callable[[Dict[str, Any]], bool],
             evidence: Callable[[Dict[str, Any]], Dict[str, Any]]) -> str:
    """One paid run. `work` raising, or `delivered` returning False, releases the hold."""
    svc = catalog.service(service_id)
    L = ledger()
    auto = not quote_id          # drawn from the session cap; an explicit quote is its own consent
    try:
        if quote_id:
            q = L.db.execute("SELECT service,payer FROM quotes WHERE id=?", (quote_id,)).fetchone()
            if q is None or q[0] != service_id or q[1] != PAYER:
                return _dump({"error": "quote %s is not a %s quote for %s" % (quote_id, service_id, PAYER)})
        else:
            price = micro(svc["price_pwm"])
            left = _session["cap_micro"] - _session["spent_micro"]
            if price > left:
                return _dump({"error": "not authorised", "charged_pwm": "0",
                              "price_pwm": svc["price_pwm"], "session_cap_left_pwm": pwm(left),
                              "next": "ask the user, then pwm_authorize a cap, or pass a quote_id "
                                      "from pwm_quote"})
            quote_id = L.quote(PAYER, svc)["quote_id"]
        run_id = "r_" + uuid.uuid4().hex[:16]
        held = L.reserve(run_id, quote_id)
        cap_used = micro(held["reserved_pwm"]) if auto else 0
        _session["spent_micro"] += cap_used
    except LedgerError as exc:
        return _dump({"error": exc.code, "detail": str(exc), "charged_pwm": "0"})

    try:
        result = work()
    except Exception as exc:                      # failed run: not charged
        receipt = L.release(run_id, "failed: %s" % exc)
        _session["spent_micro"] -= cap_used
        return _dump({"status": "failed", "error": str(exc), "receipt": receipt})
    if delivered(result):
        receipt = L.settle(run_id, evidence=evidence(result))
        _session["spent_micro"] -= micro(receipt["refunded_pwm"]) if auto else 0
    else:
        receipt = L.release(run_id, "not delivered: %s" % result.get("status", "rejected"),
                            evidence=evidence(result))
        _session["spent_micro"] -= cap_used
    return _dump({"result": result, "receipt": receipt})


# ---- free: catalogue, quotes, authorisation, balance, receipts ------------------------

@server.tool()
def pwm_catalog() -> str:
    """List the PWM-priced research agents, tools and judges, with prices and who is paid."""
    c = catalog.load()
    return _dump({"network": c["network"], "payer": PAYER, "services": [
        {k: s[k] for k in ("id", "kind", "title", "price_pwm", "settles_only", "agreement")}
        | {"paid_to": [b["account"] for b in s["beneficiaries"]] + ["treasury (3%)"]}
        for s in c["services"]]})


@server.tool()
def pwm_quote(service: str) -> str:
    """Get a PWM quote for one use of a service, with the receipt lines it would settle into."""
    try:
        return _dump(ledger().quote(PAYER, catalog.service(service)))
    except (KeyError, LedgerError, ValueError) as exc:
        return _dump({"error": str(exc)})


@server.tool()
def pwm_authorize(cap_pwm: str) -> str:
    """Authorise paid calls in this session up to cap_pwm PWM in total. Ask the user first."""
    cap = micro(cap_pwm)
    _session["cap_micro"] = max(cap, _session["spent_micro"])
    return _dump({"session": _session["id"], "payer": PAYER, "cap_pwm": pwm(_session["cap_micro"]),
                  "spent_pwm": pwm(_session["spent_micro"]),
                  "balance": ledger().balance(PAYER)})


@server.tool()
def pwm_balance() -> str:
    """The payer's test-ledger PWM balance and this session's authorised cap and spend."""
    return _dump({"balance": ledger().balance(PAYER), "session_cap_pwm": pwm(_session["cap_micro"]),
                  "session_spent_pwm": pwm(_session["spent_micro"])})


@server.tool()
def pwm_receipt(run_id: str) -> str:
    """Fetch the receipt of one run (settled, released or open)."""
    try:
        r = ledger().receipt(run_id)
    except LedgerError as exc:
        return _dump({"error": str(exc)})
    if r.get("payer", PAYER) != PAYER:
        return _dump({"error": "not your run"})
    return _dump(r)


# ---- the PWM library (the knowledge engine) ------------------------------------------
# A session reads one pinned library state (PWM_LIBRARY_STATE, else the latest at start), so everything a run used is
# named by that state's digest. New work enters only through pwm_submit, and only the validator can accept it.

_library: Optional[Library] = None
_pinned: Dict[str, Any] = {}


def library() -> Library:
    global _library
    if _library is None:
        _library = Library(Path(os.environ.get("PWM_MODE_HOME", Path.home() / ".ai4science-pwm-mode")) / "library")
    return _library


def pinned_state() -> int:
    if "n" not in _pinned:
        want = os.environ.get("PWM_LIBRARY_STATE")
        _pinned["n"] = int(want) if want and want.isdigit() else library().head()
    return _pinned["n"]


@server.tool()
def pwm_library() -> str:
    """Free: the library state this session reads (pinned for the whole session) and what it holds."""
    if library().head() < 0:
        return _dump({"state": None, "note": "the library is empty on this machine"})
    st = library().state(pinned_state())
    layers: Dict[str, int] = {}
    for ref in st["artifacts"]:
        a = library().get(ref, st["n"])
        layers[a["layer"]] = layers.get(a["layer"], 0) + 1
    return _dump({"state": st["label"], "digest": st["digest"], "artifacts": layers, "latest": library().head()})


@server.tool()
def pwm_search(field: str = "ldct", layer: Optional[str] = None, query: str = "",
               conditions: Optional[Dict[str, Any]] = None, quote_id: Optional[str] = None) -> str:
    """Paid (pwm.library.search): search the pinned library state for principles (L1), twins (L2), benchmarks (L3) and
    accepted solutions (L4). Pass the problem's conditions (e.g. lesion_hu, lesion_r) to rank the closest first."""
    return _charged("pwm.library.search", quote_id,
                    lambda: {"state": f"L{pinned_state()}",
                             "results": library().search(field=field, layer=layer, query=query,
                                                         conditions=conditions, state=pinned_state())},
                    lambda r: True, lambda r: {"state": r["state"], "n": len(r["results"])})


@server.tool()
def pwm_get(ref: str, quote_id: Optional[str] = None) -> str:
    """Paid (pwm.library.get): one artifact from the pinned state (e.g. 'ldct-sol-c150-r5@1'): its payload (for a
    solution, the method and parameters), conditions, contributors and the validator's evidence."""
    def work():
        try:
            return {"artifact": library().get(ref, pinned_state())}
        except LibraryError as exc:
            raise ValueError(str(exc))
    return _charged("pwm.library.get", quote_id, work, lambda r: True, lambda r: {"ref": ref})


@server.tool()
def pwm_submit(field: str, layer: str, id: str, title: str, payload: Dict[str, Any],
               conditions: Dict[str, Any], source_problems: List[str], summary: str = "",
               based_on: Optional[str] = None) -> str:
    """Free: submit a solution (L4) or benchmark (L3) for the library. It is NOT added now: the external validator
    reruns it on protected data and only a pass adds it, as a new library state. Credit the solution you started from
    in based_on."""
    try:
        sub = library().submit(field=field, layer=layer, id=id, title=title, summary=summary, payload=payload,
                               conditions=conditions, source_problems=source_problems, submitter=PAYER,
                               based_on=based_on)
    except LibraryError as exc:
        return _dump({"error": str(exc)})
    return _dump({"submission": sub["id"], "state": "pending",
                  "next": "the validator decides; nothing you report about your own results counts"})


# ---- low-dose CT: agent, tools, judge -----------------------------------------------

@server.tool()
def ldct_describe() -> str:
    """Free: what the low-dose CT agent does, its acceptance criteria, parameters and data."""
    return _dump(ldct.describe())


@server.tool()
def ldct_solve(seed: int = 0, params: Optional[Dict[str, float]] = None,
               quote_id: Optional[str] = None, from_solution: Optional[str] = None) -> str:
    """Paid (ldct.solve): run the low-dose CT research agent end to end on a real patient
    scan (seed 0-3): restore it and have the LDCT judge score fidelity and lesion
    detectability against the withheld full-dose scan. Charged only if the judge passes.
    from_solution: start from an accepted library solution (e.g. 'ldct-sol-c150-r5@1'); its
    parameters are used unless params overrides them."""
    if from_solution:
        try:
            base = library().get(from_solution, pinned_state())["payload"].get("params") or {}
        except LibraryError as exc:
            return _dump({"error": str(exc), "charged_pwm": "0"})
        params = dict(base, **(params or {}))
    return _charged(
        "ldct.solve", quote_id,
        lambda: ldct.solve(seed=seed, params=params,
                           fail_injection=os.environ.get("PWM_MODE_FAIL_INJECTION") == "1"),
        lambda r: r.get("status") == "delivered",
        lambda r: {"status": r.get("status"), "metrics": r.get("metrics"),
                   "verdict_passed": (r.get("verdict") or {}).get("passed"), "job": r.get("job")})


@server.tool()
def ldct_problem(seed: int = 0, dest: str = "ldct_problem", quote_id: Optional[str] = None) -> str:
    """Paid (ldct.problem): copy a real low-dose CT case into the workspace (answer key
    withheld) so you can write your own restoration, then submit it with ldct_judge."""
    return _charged("ldct.problem", quote_id,
                    lambda: ldct.problem(seed=seed, dest=_in_workspace(dest)),
                    lambda r: bool(r.get("problem_id")),
                    lambda r: {"problem_id": r.get("problem_id"), "files": r.get("files")})


@server.tool()
def ldct_judge(problem_id: str, reconstruction: str = "ldct_problem/results/reconstruction.npy",
               quote_id: Optional[str] = None) -> str:
    """Paid (ldct.judge): judge your reconstruction (.npy, HU, same shape as low_dose.npy)
    against the withheld full-dose scan. Charged when a verdict is delivered, pass or fail;
    a malformed submission gets no verdict and costs 0 PWM."""
    return _charged("ldct.judge", quote_id,
                    lambda: ldct.judge(problem_id=problem_id,
                                       reconstruction=_in_workspace(reconstruction)),
                    lambda r: "passed" in r,
                    lambda r: {"problem_id": problem_id, "passed": r.get("passed"),
                               "metrics": r.get("metrics")})


def _field_tools() -> Dict[str, Any]:
    from ai4science.harness.agents.research_agents.stores import field_store_tools
    return {t.name: t for t in field_store_tools("low-dose-ct", SimpleNamespace(workspace=WORKSPACE))}


@server.tool()
def ldct_field(tool: str, args: Optional[Dict[str, Any]] = None,
               quote_id: Optional[str] = None) -> str:
    """Paid (ldct.field): the low-dose CT agent's field tools. tool is one of
    field_principles, field_check_claim, field_twin_status, field_benchmark_protocol,
    field_solutions, field_append_solution, field_provenance, field_validate_split.
    args follow that tool's schema (see ldct_field_schemas, free)."""
    tools = _field_tools()
    if tool not in tools:
        return _dump({"error": "unknown field tool", "choose": sorted(tools), "charged_pwm": "0"})
    return _charged("ldct.field", quote_id,
                    lambda: {"answer": json.loads(tools[tool].func(WORKSPACE, **(args or {})))},
                    lambda r: True, lambda r: {"tool": tool})


@server.tool()
def ldct_field_schemas() -> str:
    """Free: the argument schemas of the low-dose CT field tools."""
    return _dump({n: {"description": t.description, "parameters": t.parameters}
                  for n, t in _field_tools().items()})


# ---- reused from mcp_pwm.py -----------------------------------------------------------

def _mcp_pwm(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    from ai4science.agents import mcp_pwm
    out = asyncio.run(getattr(mcp_pwm, name)(dict(args, workspace=str(WORKSPACE))))
    text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
    if out.get("isError"):
        raise RuntimeError(text)
    return {"report": json.loads(text)}


@server.tool()
def pwm_validate(quote_id: Optional[str] = None) -> str:
    """Paid (pwm.validate): run the deterministic AI4Science artifact validator on the workspace."""
    return _charged("pwm.validate", quote_id, lambda: _mcp_pwm("pwm_validate", {}),
                    lambda r: True, lambda r: {"overall": r["report"].get("overall")})


@server.tool()
def pwm_judge_cassi(benchmark: Optional[str] = None, quote_id: Optional[str] = None) -> str:
    """Paid (pwm.judge_cassi): run the deterministic CASSI Physics Judge on the workspace."""
    args = {"benchmark": benchmark} if benchmark else {}
    return _charged("pwm.judge_cassi", quote_id, lambda: _mcp_pwm("pwm_judge_cassi", args),
                    lambda r: isinstance(r.get("report"), dict), lambda r: {"benchmark": benchmark})


@server.tool()
def pwm_status() -> str:
    """Free: workspace status (artifacts present, reports), from mcp_pwm.py."""
    try:
        return _dump(_mcp_pwm("pwm_status", {})["report"])
    except Exception as exc:
        return _dump({"error": str(exc)})


# ---- the agent itself, as a prompt the host model adopts --------------------------------

@server.prompt()
def low_dose_ct_agent() -> str:
    """Act as AI4Science's low-dose CT research agent, using its PWM-priced tools."""
    from ai4science.harness.agents.specs import low_dose_ct as spec
    d = ldct.describe()
    return (spec.SYSTEM_PROMPT.strip() + "\n\nIn PWM mode your tools are the MCP server "
            "ai4science-pwm: ldct_field (the field_* tools), ldct_problem, ldct_judge and "
            "ldct_solve. Each paid call returns a receipt; quote the price and get the user's "
            "cap with pwm_authorize before the first paid call. The judge's criteria:\n- "
            + "\n- ".join(d["criteria"]))


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
