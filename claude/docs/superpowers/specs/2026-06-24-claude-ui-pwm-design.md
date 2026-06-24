# Claude UI PWM — Design Spec

**Date**: 2026-06-24
**Repo**: https://github.com/integritynoble/Claude_UI_PWM
**Domain**: claude.platformai.org
**Server**: 34.63.169.185

## Overview

A web-based Claude chat interface (like claude.ai in a browser) powered by PWM tokens. Users open `claude.platformai.org`, enter their `pwm_` API key once, and get a full Claude conversational UI with streaming responses, markdown rendering, and conversation history — all billed against their PWM token balance.

This is distinct from `claude-pwm` (which is the agentic Claude Code CLI). This is the conversational Claude experience in the browser.

---

## Section 1: Architecture & File Layout

**Stack**: FastAPI (Python) backend + static HTML/CSS/Vanilla JS frontend. No React, no npm build step. Consistent with existing PWM platform patterns.

```
Claude_UI_PWM/
  main.py                          # FastAPI app entry point
  config.py                        # Settings (exchange URL, port, CORS)
  routers/
    chat.py                        # POST /api/chat/stream → SSE streaming
    pages.py                       # GET / → serves index.html
  static/
    index.html                     # Full chat UI (single page)
    css/style.css                  # Claude-inspired dark/light theme
    js/app.js                      # Streaming chat logic (fetch + ReadableStream)
  Dockerfile
  docker-compose.yml               # Binds to 127.0.0.1:8102
  nginx/
    claude.platformai.org.conf     # nginx vhost config (SSL via certbot)
  requirements.txt
  .github/workflows/deploy.yml     # SSH deploy on push to main
```

**Streaming mechanism**: Backend exposes a Server-Sent Events (SSE) endpoint. JS frontend uses `fetch()` with a `ReadableStream` to render tokens as they arrive, identical to claude.ai feel.

**Deployment pattern**: Docker container on port `8102` behind the shared nginx on `34.63.169.185` — same as `testpwm.platformai.org → 8101`.

---

## Section 2: Auth & API Flow

Auth is fully stateless — no server-side database or session store needed.

### User flow
1. First visit → settings modal prompts for `pwm_` API key
2. Key stored in `localStorage` — only sent per-request in `Authorization` header, never stored server-side
3. User sends a message → frontend POSTs to `/api/chat/stream` with key + conversation history
4. Backend extracts key, forwards to PWM exchange with it as `x-api-key`
5. Exchange verifies PWM balance, deducts tokens, forwards to Anthropic
6. Anthropic SSE stream is piped directly back to browser

```
Browser
  └─► POST /api/chat/stream  (Authorization: Bearer pwm_...)
        FastAPI backend
          └─► physicsworldmodel.org/api/v1/exchange/anthropic  (x-api-key: pwm_...)
                └─► api.anthropic.com
                      └─► SSE stream back through the chain to browser
```

### Request body (frontend → backend)
```json
{
  "model": "claude-sonnet-4-6",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What is quantum mechanics?"}
  ],
  "system": "You are Claude, a helpful AI assistant made by Anthropic."
}
```

### Available models
| Model | Description |
|-------|-------------|
| `claude-sonnet-4-6` | Default — best balance of speed and quality |
| `claude-opus-4-8` | Most capable |
| `claude-haiku-4-5-20251001` | Fastest, most economical |
| `claude-3-5-sonnet-20241022` | Previous generation Sonnet |
| `claude-3-5-haiku-20241022` | Previous generation Haiku |
| `claude-3-opus-20240229` | Previous generation Opus |

---

## Section 3: UI/UX

Mirrors the claude.ai layout — clean, minimal, conversation-focused.

### Layout
```
┌─────────────────────────────────────────────────────┐
│  ☰  Claude  [PWM]              [model ▾]  [⚙ key]  │  ← top bar
├─────────────────────────────────────────────────────┤
│  [sidebar]  │                                        │
│  + New chat │      conversation thread               │
│  ─────────  │                                        │
│  Chat 1     │  [You]                                 │
│  Chat 2     │  Your message text                     │
│  Chat 3     │                                        │
│             │  [Claude]  (claude-sonnet-4-6)         │
│             │  Response with markdown, code, etc.    │
│             │                                        │
│             │  tokens: 420 in / 185 out              │
├─────────────┴───────────────────────────────────────┤
│  ┌──────────────────────────────────┐  [↑ Send]     │
│  │  Type a message...               │               │
│  └──────────────────────────────────┘               │
└─────────────────────────────────────────────────────┘
```

### Key UX details
- **Markdown rendering**: marked.js (~50 KB CDN, no build step)
- **Syntax highlighting**: highlight.js for code blocks
- **Streaming**: tokens render in real-time as they arrive via `ReadableStream`
- **Conversation history**: stored in `localStorage` — no backend DB
- **Sidebar**: collapsible list of past conversations; click to load
- **New chat**: clears the current thread, starts fresh
- **Settings modal**: enter/update PWM API key + choose default model
- **Responsive**: works on mobile (sidebar collapses to hamburger)
- **Theme**: Claude-inspired purple/white palette; optional dark mode toggle

---

## Section 4: Deployment

### Docker
```yaml
# docker-compose.yml
services:
  app:
    build: .
    restart: unless-stopped
    ports:
      - "127.0.0.1:8102:8000"
    environment:
      PWM_EXCHANGE_URL: https://physicsworldmodel.org/api/v1/exchange/anthropic
```

Port `8102` is the next free port after `8101` (testpwm/nonprofit).

### nginx vhost
```nginx
# /etc/nginx/sites-enabled/claude.platformai.org
server {
    server_name claude.platformai.org;
    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8102;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_buffering off;          # required for SSE streaming
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

TLS via: `certbot --nginx -d claude.platformai.org`

### CI/CD
`.github/workflows/deploy.yml` — on push to `main`:
1. SSH into `34.63.169.185`
2. `git pull`
3. `docker compose up -d --build`

### Python dependencies
```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
httpx>=0.27.0
python-multipart>=0.0.9
```

No `anthropic` SDK needed — `httpx` handles the raw HTTP proxy streaming directly to the exchange endpoint.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| No API key set | Settings modal auto-opens on first message attempt |
| 401 from exchange | Show "Invalid PWM key — check your settings" inline |
| Insufficient balance | Show "Insufficient PWM token balance" inline |
| Network error | Show "Connection error — please retry" inline |
| Stream interrupted | Show partial response + "(connection interrupted)" note |

---

## Out of Scope (v1)

- Extended thinking / tool use
- File / image attachments
- Server-side conversation persistence (DB)
- User accounts / login
- Usage dashboard
