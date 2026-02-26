# Spec-Builder Chatbox

The spec-builder chatbox is an interactive conversational interface on the PWM Dashboard that lets users design, validate, and simulate computational imaging forward-model specs through natural language. It is powered by Gemini 2.5 Flash and uses HTMX for a server-driven, no-SPA chat experience.

**Live URL:** `https://pwm.platformai.org/` (Dashboard page)

---

## Table of Contents

1. [Core Flows](#core-flows)
2. [File Map](#file-map)
3. [API Endpoints](#api-endpoints)
4. [Spec JSON Schema](#spec-json-schema)
5. [The 11 PWM Primitives](#the-11-pwm-primitives)
6. [LLM Integration](#llm-integration)
7. [Session Persistence](#session-persistence)
8. [Simulation Pipeline](#simulation-pipeline)
9. [Frontend Architecture](#frontend-architecture)

---

## Core Flows

### Flow 1: Prompt to Spec

A user describes an imaging system in natural language and the LLM generates a structured spec JSON.

```
User types description ──► HTMX POST /api/v1/spec-chat/{variant_key}
                                │
                    ┌───────────▼────────────┐
                    │  spec_chat.chat_message │
                    │  (routers/spec_chat.py) │
                    └───────────┬────────────┘
                                │
               create/retrieve session (PostgreSQL)
               append user turn to history
                                │
                    ┌───────────▼────────────────┐
                    │  build_system_prompt()      │
                    │  (services/spec_chat_prompts│
                    │   .py)                      │
                    │                             │
                    │  Injects:                   │
                    │  - 11 PWM primitives        │
                    │  - 3 few-shot example specs │
                    │  - Current variant context  │
                    │  - Output format rules      │
                    └───────────┬────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  call_gemini()          │
                    │  (services/gemini_client│
                    │   .py)                  │
                    │                         │
                    │  Gemini 2.5 Flash via   │
                    │  CompareGPT API         │
                    │  temp=0.3, 4096 tokens  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────────┐
                    │  parse_spec_from_response() │
                    │  Extracts JSON from ```json │
                    │  blocks; separates          │
                    │  explanation text from spec  │
                    └───────────┬────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Render HTML partial    │
                    │  _spec_chat_message.html│
                    │                         │
                    │  Shows: explanation,    │
                    │  spec notation, visual  │
                    │  DAG, mismatch table,   │
                    │  noise model, simulate  │
                    │  button                 │
                    └────────────────────────┘
```

**Step-by-step:**

1. User types a description (e.g. "Design an SD-CASSI system with a coded aperture and prism dispersion") in the chatbox textarea (input mode = `describe`).
2. The form submits via HTMX: `POST /api/v1/spec-chat/{variant_key}` with `message`, `session_id`, and `input_mode`.
3. The router creates or retrieves a conversation session from PostgreSQL.
4. The user turn is appended to the conversation history.
5. `build_system_prompt()` assembles a system prompt containing the 11 primitives, three example specs (SD-CASSI, SPC-Block, CACTI), the current variant context, and output-format instructions.
6. `call_gemini()` sends the system prompt + full history to Gemini 2.5 Flash (CompareGPT OpenAI-compatible API, temperature 0.3, max 4096 tokens, 120s timeout).
7. `parse_spec_from_response()` extracts the JSON spec from markdown fenced blocks and separates the explanation text.
8. The response renders as `_spec_chat_message.html` — a chat bubble with the explanation, spec notation, visual DAG pipeline, mismatch parameters table, noise model, measurement matrix, and a **Simulate this spec** button.

### Flow 2: Finetune the Spec

Users iteratively refine an existing spec through multi-turn conversation.

```
 Turn 1                         Turn 2                         Turn N
┌──────────┐               ┌──────────────┐               ┌──────────────┐
│ "Design  │               │ "Change the  │               │ "Add Poisson │
│  a CASSI │               │  dispersion  │               │  noise and   │
│  system" │               │  angle to    │               │  increase    │
│          │               │  0.2 rad"    │               │  mismatch"   │
└────┬─────┘               └──────┬───────┘               └──────┬───────┘
     │                            │                              │
     ▼                            ▼                              ▼
  history: [1 turn]         history: [3 turns]           history: [2N-1 turns]
     │                            │                              │
     ▼                            ▼                              ▼
  Gemini sees full          Gemini sees full              Gemini sees full
  context each turn         context each turn             context each turn
     │                            │                              │
     ▼                            ▼                              ▼
  Complete spec v1          Complete spec v2              Complete spec vN
```

- Uses the **same endpoint** as Flow 1; the only difference is that the conversation history already contains previous turns.
- The full history is persisted in PostgreSQL (`spec_chat_sessions` table) with an in-memory write-through cache.
- The LLM always returns the **complete updated spec** (not diffs), so every response is a self-contained, runnable spec.
- Users can ask to modify any aspect: forward-model DAG, noise model, mismatch parameters, measurement matrix, specific primitive parameters, etc.

### Flow 3: Paste Spec (Upload & Validate)

- The chatbox has an input-mode toggle: **Describe** (default) or **Paste Spec**.
- In "Paste Spec" mode, the user's JSON is wrapped with: *"Please validate and describe the following spec JSON, then suggest improvements if any."*
- The LLM validates the JSON structure, checks primitive usage, and returns suggestions.

### Flow 4: Load Example

- Three example buttons at the top of the chatbox: **SD-CASSI**, **SPC-Block**, **CACTI**.
- `POST /api/v1/spec-chat/{variant_key}/example` creates a new session with a synthetic first exchange (pre-built spec + explanation).
- The user can immediately start refining from the example.

### Flow 5: Simulate

- After any spec is generated, a green **Simulate this spec** button appears.
- `POST /api/v1/spec-chat/{variant_key}/simulate` runs the full forward-model pipeline.
- Returns ground truth / measurement / reconstructed images, PSNR/SSIM metrics, bottleneck analysis, and recommendations.
- See [Simulation Pipeline](#simulation-pipeline) for details.

---

## File Map

### Frontend Templates

| File | Purpose |
|------|---------|
| `templates/dashboard.html` | Dashboard page; includes the chatbox via `{% include "_spec_chat_box.html" %}` |
| `templates/_spec_chat_box.html` | Chat widget container — chat log, input area, example buttons, loading spinner |
| `templates/_spec_chat_message.html` | Individual message partial — user/assistant bubbles, spec notation, visual DAG, mismatch table, simulate button |
| `templates/_spec_simulation_result.html` | Simulation result card — images, PSNR/SSIM, bottleneck analysis, recommendations |

### Backend

| File | Purpose |
|------|---------|
| `routers/spec_chat.py` | API router (`/api/v1/spec-chat`) — 3 endpoints: chat, example, simulate |
| `services/spec_chat_prompts.py` | System prompt assembly + `parse_spec_from_response()` |
| `services/gemini_client.py` | Async Gemini client via CompareGPT API; conversation CRUD with in-memory cache |
| `services/spec_simulator.py` | Forward-model simulation pipeline (CASSI/SPC/CACTI) |
| `db/models.py` (`SpecChatSession`) | SQLAlchemy model for `spec_chat_sessions` table |

### Shared Data

| File | Purpose |
|------|---------|
| `services/benchmark_database/_primitives.py` | The 11 PWM primitive definitions |
| `services/benchmark_database/_variant_registry.py` | All 65 variant specs (SD-CASSI, SPC-Block, CACTI, etc.) |

All paths are relative to `platform/pwm_platform/`.

---

## API Endpoints

Base path: `/api/v1/spec-chat`

### `POST /{variant_key}` — Chat Message

Process a user message and return an LLM-generated spec.

| Parameter | Source | Type | Description |
|-----------|--------|------|-------------|
| `variant_key` | path | string | Imaging variant (e.g. `sd_cassi`) |
| `message` | form | string | User's natural-language description or JSON spec |
| `session_id` | form | string | UUID for continuing a conversation (empty = new session) |
| `input_mode` | form | string | `describe` (default) or `upload_spec` |

**Response:** HTML partial (`_spec_chat_message.html`) swapped into `#chat-log` via HTMX.

### `POST /{variant_key}/example` — Load Example

Load a pre-built example spec to bootstrap a conversation.

| Parameter | Source | Type | Description |
|-----------|--------|------|-------------|
| `variant_key` | path | string | Imaging variant |
| `example` | form | string | One of: `cassi`, `spc`, `cacti` |

**Response:** HTML partial with synthetic first exchange.

### `POST /{variant_key}/simulate` — Run Simulation

Run the forward-model pipeline on a spec.

| Parameter | Source | Type | Description |
|-----------|--------|------|-------------|
| `variant_key` | path | string | Imaging variant |
| `spec_json` | form | string | The spec JSON to simulate |

**Response:** HTML partial (`_spec_simulation_result.html`) with images, metrics, and analysis.

---

## Spec JSON Schema

Every spec generated by the chatbox conforms to this structure:

```json
{
  "spec_notation": "M(mask) → W(α, a) → Σ_λ → D(g, η₄)",
  "forward_model": [
    {
      "primitive": "M",
      "params": "mask",
      "label": "Coded Aperture"
    },
    {
      "primitive": "W",
      "params": "α, a",
      "label": "Prism Dispersion"
    },
    {
      "primitive": "Sigma",
      "params": "λ",
      "label": "Spectral Sum"
    },
    {
      "primitive": "D",
      "params": "g, η₄",
      "label": "Detector"
    }
  ],
  "mismatch_params": [
    {
      "name": "mask_dx",
      "symbol": "Δx",
      "description": "Mask lateral shift",
      "nominal": 0,
      "perturbed": 0.5
    }
  ],
  "noise_model": "Mixed Poisson-Gaussian η₄",
  "measurement_matrix": "Binary random mask with 50% transmittance"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `spec_notation` | string | Human-readable pipeline notation using primitive symbols and arrows |
| `forward_model` | array | Ordered DAG of primitives — each node has `primitive` (key), `params`, and `label` |
| `mismatch_params` | array | Parameters that can deviate from nominal — each has `name`, `symbol`, `description`, `nominal`, `perturbed` |
| `noise_model` | string | Noise model description (Poisson, Gaussian, mixed Poisson-Gaussian) |
| `measurement_matrix` | string | Description of the sensing/measurement matrix |

---

## The 11 PWM Primitives

These are the building blocks available for constructing forward-model DAGs:

| Symbol | Key | Name | Description |
|--------|-----|------|-------------|
| **P** | `P` | Propagation | Free-space or medium propagation kernel |
| **M** | `M` | Mask / Modulation | Spatial or spatio-temporal amplitude modulation |
| **Pi** | `Pi` | Projection | Geometric projection (Radon, fan-beam, cone-beam) |
| **F** | `F` | Fourier Sampling | k-space sampling (MRI, ptychography) |
| **C** | `C` | Convolution | Shift-invariant convolution with PSF |
| **Sigma** | `Sigma` | Summation / Integration | Summation along a physical dimension |
| **D** | `D` | Detector | Sensor readout with gain and noise model |
| **S** | `S` | Structured Illumination | Patterned illumination |
| **W** | `W` | Wavelength Dispersion | Spectral dispersion (prism, grating) |
| **R** | `R` | Rotation / Motion | Sample or gantry rotation |
| **Lambda** | `Lambda` | Wavelength Selection | Spectral filter or monochromator |

Defined in `services/benchmark_database/_primitives.py`.

---

## LLM Integration

| Setting | Value |
|---------|-------|
| **Model** | Gemini 2.5 Flash |
| **API** | CompareGPT OpenAI-compatible endpoint (`https://comparegpt.io/api/chat/completions`) |
| **API key env var** | `COMPAREGPT_API_KEY` |
| **Temperature** | 0.3 |
| **Max tokens** | 4096 |
| **Request timeout** | 120 seconds |

### System Prompt Contents

`build_system_prompt(variant)` in `services/spec_chat_prompts.py` assembles:

1. **Task description** — "You are a spec builder for computational imaging systems..."
2. **Primitive library** — All 11 primitives with symbols and descriptions
3. **Output format** — JSON schema with `spec_notation`, `forward_model`, `mismatch_params`, `noise_model`, `measurement_matrix`
4. **Few-shot examples** — Three complete specs (SD-CASSI, SPC-Block, CACTI)
5. **Current variant context** — Display name and existing spec notation
6. **Guidelines** — Use only library primitives, explain reasoning, always show the complete spec

### Response Parsing

`parse_spec_from_response(text)` extracts JSON from `` ```json ``` `` fenced blocks and returns a tuple: `(explanation_text, spec_dict | None)`.

---

## Session Persistence

| Component | Detail |
|-----------|--------|
| **Database table** | `spec_chat_sessions` |
| **Model** | `SpecChatSession` in `db/models.py` |
| **Primary key** | `session_id` (UUID) |
| **Fields** | `session_id`, `user_id` (optional), `variant_key`, `history` (JSONB), `created_at`, `updated_at` |
| **History format** | JSONB array: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]` |
| **Caching** | In-memory write-through cache in `gemini_client.py` |
| **Anonymous support** | `user_id` is optional; anonymous users can still have persistent sessions |

Key functions in `services/gemini_client.py`:
- `create_conversation(db, user_id, variant_key)` — Create new session
- `get_conversation(db, session_id)` — Retrieve history (cached)
- `append_to_conversation(db, session_id, role, text)` — Persist a turn
- `list_user_sessions(db, user_id, limit)` — List a user's sessions

---

## Simulation Pipeline

When the user clicks **Simulate this spec**, the spec is sent to `services/spec_simulator.py`.

```
spec JSON ──► modality detection ──► per-modality pipeline ──► metrics + analysis
                                           │
                               ┌───────────┼───────────┐
                               ▼           ▼           ▼
                            CASSI        SPC        CACTI
                               │           │           │
                     KAISTDataset    Set11Dataset  SyntheticCACTI
                     (256x256x28)    (64x64)       (256x256x8)
                               │           │           │
                     SDCASSIOperator SPCOperator  CACTIOperator
                               │           │           │
                     TVAL3 recon   ADMM-TV recon GAP-TV recon
                               │           │           │
                               └───────────┼───────────┘
                                           ▼
                                  PSNR, SSIM metrics
                                  Bottleneck analysis:
                                  - Photon severity
                                  - Mismatch severity
                                  - Recoverability
                                  - Solver fit
                                           │
                                           ▼
                              Save images to /static/simulations/{sim_id}/
                              Render _spec_simulation_result.html
```

The result card shows:
- **3-panel image grid:** Ground Truth, Measurement, Reconstructed
- **Metrics:** PSNR (dB) and SSIM
- **Solver info:** Algorithm name and iteration count
- **Bottleneck analysis:** Severity bars (red/amber/green)
- **Recommendations:** Actionable suggestions for spec improvement

---

## Frontend Architecture

| Technology | Role |
|------------|------|
| **Jinja2** | Server-side HTML templating |
| **Tailwind CSS v3** | Styling (loaded from CDN) |
| **HTMX 2.0.4** | Dynamic UI updates without JavaScript (loaded from CDN) |

### How HTMX drives the chat

The chatbox uses zero custom JavaScript for its core interactions:

1. **Form submission:** `hx-post="/api/v1/spec-chat/{variant_key}"` sends the form data.
2. **Response swapping:** `hx-target="#chat-log"` + `hx-swap="beforeend"` appends the returned HTML partial to the chat log.
3. **Loading state:** `hx-indicator="#loading-spinner"` shows a spinner during the request.
4. **Event hooks:**
   - `hx-on::before-request` — Removes the placeholder text from the chat log.
   - `hx-on::after-request` — Clears the textarea and re-focuses it.
5. **Session tracking:** A hidden `<input name="session_id">` is updated by inline JS in each response partial.

### Chat message rendering

`_spec_chat_message.html` renders:
- **User bubble** — Right-aligned, indigo background
- **Assistant bubble** — Left-aligned, white with border, containing:
  - Prose explanation
  - Spec notation (monospace, indigo background)
  - Visual DAG pipeline (primitive boxes connected by arrows, flexbox layout)
  - Noise model and measurement matrix cards (2-column grid)
  - Mismatch parameters table (symbol, parameter, description, nominal, perturbed)
  - Green **Simulate this spec** button
