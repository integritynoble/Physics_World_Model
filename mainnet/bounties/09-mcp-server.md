# Bounty 9 — PWM MCP Server (Model Context Protocol)

- **Amount:** 25,000 PWM
- **Opens:** when `agent-web` reference impl exposes REST API endpoints (Bounty 2 winner declared OR Track 3c API surface published) — earliest ~D9 + 90 days
- **Reference implementation:** None — bounty implements an external spec
- **Reference specification:** [Model Context Protocol](https://modelcontextprotocol.io/) (Anthropic; widely adopted by Claude, ChatGPT, Gemini, Cursor, others)
- **Acceptance harness:** MCP test client + PWM-specific tool-use scenarios in `infrastructure/agent-mcp/tests/`

## What you build

A lightweight **PWM MCP Server** — a Python or TypeScript binary that:

1. Speaks the MCP wire protocol (stdio + JSON-RPC 2.0 transport per the spec)
2. Exposes PWM-specific **tools** that AI assistants can call to query PWM
3. Returns structured data from PWM's on-chain registry + indexer API
4. Handles authentication, rate limiting, and error reporting per MCP norms
5. Ships as a PyPI package (`pwm-mcp-server`) or npm package (`@pwm/mcp-server`) — installable in ~30 seconds

**Why this matters:** MCP is the standard wire protocol for AI assistants to call external tools. Anthropic Claude, OpenAI ChatGPT, Google Gemini, Cursor, Zed all support MCP. A PWM MCP server makes PWM queryable by every AI assistant that supports MCP — without partnership deals, without protocol changes, without per-AI-vendor integration work. **Single MCP server = N AI assistants.**

## Mandatory tools

Your MCP server must expose these 8 tools (using MCP `tools/list` + `tools/call`):

| Tool name | Inputs | Output |
|---|---|---|
| `pwm_get_principle` | `principle_id` (string) | Principle metadata: domain, δ, Pool_k, active benchmarks, top PSNR |
| `pwm_get_benchmark` | `benchmark_id` (string) | Benchmark metadata: Ω tier, pool size, rank-1 holder, ε, ρ weight |
| `pwm_get_certificate` | `cert_hash` (bytes32) | Certificate detail: S1-S4 verdicts, Q score, reward breakdown, status |
| `pwm_get_leaderboard` | `benchmark_id`, `top_n` (default 10) | Top-n ranks: PSNR, Q, SP wallet, draw per epoch |
| `pwm_search` | `query` (string) | Resolves principle / spec / benchmark / cert / wallet IDs |
| `pwm_get_pool_balance` | `principle_id` | Current Pool_k + T_k balance |
| `pwm_list_recent_finalizations` | `limit` (default 25) | Most recent CertificateFinalized events |
| `pwm_verify_artifact_priority` | `hash` (bytes32) | Returns block timestamp + creator address for the on-chain registration (the "claim-priority" check from `prevent_copy/PWM_COMPETITIVE_DEFENSE` §3.1) |

These 8 tools cover ~80% of expected AI-assistant queries to PWM. Adding extra tools is fine; missing any of the 8 voids the award.

## Interface contract

- **Data source:** Read from the public PWM web indexer API at `api.physicsworldmodel.org/v1/...` (or `physicsworldmodel.org` if migration pending)
- **Address mapping:** `addresses.json` from `interfaces/addresses.json` — must support both Sepolia + mainnet via env var `PWM_NETWORK={sepolia,mainnet}`
- **Rate limiting:** Respect MCP server best-practices (token bucket; ~10 req/sec default)
- **Authentication:** No auth required for public read APIs (PWM data is fully public)

## Distribution

Your binary must install in ≤ 30 seconds via:

```bash
# Python (preferred for AI-tooling ecosystem)
pip install pwm-mcp-server

# OR TypeScript
npm install -g @pwm/mcp-server
```

Configuration via Claude Desktop / ChatGPT / Cursor settings — provide:

- `~/.config/Claude/claude_desktop_config.json` example snippet
- `~/.config/cursor/mcp.json` example snippet
- Universal example for any MCP-compatible client

## What must pass

1. **MCP spec compliance.** All 8 tools must respond to `tools/list` and `tools/call` per [MCP spec v0.1+](https://modelcontextprotocol.io/specification). Validated via official MCP test client.

2. **Functional correctness.** Each tool returns data within 1 wei / 0.0001 Q of the on-chain or indexer-API value. Verified by running 100 random tool calls per week during shadow run.

3. **Latency.** P95 tool-call latency < 2 seconds (measured against `api.physicsworldmodel.org`). P99 < 5 seconds.

4. **Installation reliability.** ≥ 95% of users complete `pip install` (or npm install) + add to AI client + execute first tool call within 5 minutes. Tested with 20 sample users during shadow run.

5. **Error handling.** Network errors, malformed inputs, rate-limit hits all return structured MCP errors per spec (not raw stack traces).

6. **Documentation.** README must include:
   - 30-second install guide
   - Configuration snippets for Claude Desktop, Cursor, Zed (minimum 3 clients)
   - Tool descriptions with example queries
   - Troubleshooting section

7. **AI-assistant interop demo.** Submission must include a 2-3 minute screencast OR test transcript showing the MCP server working with:
   - Claude Desktop OR Claude CLI
   - At least one second AI client (Cursor, Zed, ChatGPT Desktop, etc.)

## What you may change

- Language: Python OR TypeScript (both ecosystems have mature MCP libraries)
- Transport: stdio (default) OR HTTP+SSE (also per MCP spec)
- DB caching: optional in-memory cache (~5 min TTL) to reduce indexer load
- Tool naming: prefix `pwm_` is required; specific names may be polished
- Internal structure: any code organization; layer your own way

## What you may not change

- The set of 8 mandatory tools (adding more is fine; removing any voids award)
- MCP spec compliance (no proprietary extensions for required tools)
- Data sources (must use public PWM indexer API; no direct contract reads outside `addresses.json`)
- Public install path (must be installable via pip OR npm; not git clone only)

## Shadow run

Your MCP server runs alongside any reference (if Bounty 2 winner has built one) for **30 days**. agent-coord:

- Runs 100 random tool calls per week and diffs your responses vs. the indexer-API
- Tests installation with 5 first-time users per week
- Monitors p95 latency continuously
- Disagreement > 0 wei / 0.0001 Q OR install failure rate > 5% OR latency p95 > 2s triggers investigation; three unexplained regressions void the award

## Strategic context

- This bounty is the **smallest, simplest pre-cursor** to Track 7 (full agent infrastructure)
- A successful MCP server makes PWM queryable by every AI assistant out of the box
- Opens potential Anthropic / OpenAI / Google partnership conversations (per `coordination/prevent_copy/PWM_COMPETITIVE_DEFENSE` §4 "walled-garden capture" mitigation)
- Per `coordination/PWM_API_VS_WEBSITE_2026-05-20.md` §6, an agent SDK partnership trigger accelerates Track 7 — but this MCP bounty is the lightweight version that can ship without that trigger

## Payment

- Paid from Reserve discretionary pool (Layer 3 per `coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` §4) OR from existing bounty allocation if Director / governance prefers
- Wallet listed in PR description; confirm on Sepolia before opening
- Mainnet swap: 1:1 at Phase 2 launch

## References

- MCP spec: https://modelcontextprotocol.io/
- MCP TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- Claude Desktop MCP config: https://docs.claude.com/en/docs/claude-desktop/mcp
- Example MCP servers: https://github.com/modelcontextprotocol/servers
