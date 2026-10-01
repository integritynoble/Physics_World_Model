"""Drive the MCP server over real stdio (as OpenCode does) and check charging on the ledger."""
import asyncio, json, os, sys
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession
from pwm_mode.ledger import Ledger
from pwm_mode import catalog

async def call(fail, seed):
    env = dict(os.environ, PYTHONPATH=os.getcwd(), PWM_MODE_PAYER="user:test-owner",
               PWM_MODE_WORKSPACE=os.path.abspath("e2e-fail"), PWM_MODE_FAIL_INJECTION="1" if fail else "0")
    p = StdioServerParameters(command=sys.executable, args=["-m", "pwm_mode.mcp_server"], env=env)
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            await s.call_tool("pwm_authorize", {"cap_pwm": "3"})
            out = await s.call_tool("ldct_solve", {"seed": seed})
            return tools, json.loads(out.content[0].text)

L = Ledger(catalog.ledger_path())
bal = lambda: L.balance("user:test-owner")["pwm"]
b0 = bal()
tools, out = asyncio.run(call(True, 1))
print("tools over stdio:", len(tools), sorted(tools))
print("FAILED run:", out.get("status") or out["result"]["status"], (out.get("result") or {}).get("why"), "| receipt", out["receipt"]["run_id"], out["receipt"]["status"],
      "charged", out["receipt"]["charged_pwm"], "| payer", b0, "->", bal())
b1 = bal()
_, ok = asyncio.run(call(False, 3))
print("GOOD run:", ok["result"]["status"], "judge passed", ok["result"]["verdict"]["passed"],
      "cnr %.2f" % ok["result"]["metrics"]["lesion_cnr"], "| receipt", ok["receipt"]["run_id"],
      ok["receipt"]["status"], "charged", ok["receipt"]["charged_pwm"],
      {l["account"]: l["pwm"] for l in ok["receipt"]["lines"]}, "| payer", b1, "->", bal())
print("books", L.books_balance(), "open runs", L.open_runs())
