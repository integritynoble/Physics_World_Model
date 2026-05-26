# PWM Three-Domain Architecture — physicsworldmodel.org / test. / explorer.

**Date:** 2026-05-23
**Audience:** sub-GPU + Director (implementation guide)
**Status:** Canonical architecture for the 3-domain split. Implementation pending sub-GPU's ~4-5 hour work.
**Purpose:** Defines what each of the 3 PWM domains serves + how they differ + how to implement the split.

This doc complements:
- `coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` — Phase 1a/1b/2/3/4 phasing
- `coordination/PWM_VALUE_FRAMING_2026-05-22.md` — AI4Science = product
- `infrastructure/agent-web/nginx.example.conf` — nginx template
- `infrastructure/agent-web/deploy/Dockerfile.production` — Docker build

---

## TL;DR

3 domains, 3 audiences, 1 container.

| Domain | Default chain | Mode | Audience |
|---|---|---|---|
| **physicsworldmodel.org** | Base mainnet (`base`) | Full app — AI4Science submission + wallet + leaderboard | Researchers, clinicians, paying users |
| **test.physicsworldmodel.org** | Base Sepolia (`baseSepolia`) | Full app + "TESTNET" banner | Developers, contributors, test miners |
| **explorer.physicsworldmodel.org** | Base mainnet (`base`) | Read-only browse (no wallet, no submission) | Auditors, analysts, journalists, casual browsers |

All 3 domains served by the **same `pwm-explorer` Docker container** on port 3000. nginx injects per-domain headers; Next.js server components read headers to set defaults + hide/show UI. The chain switcher dropdown still works on every domain — defaults differ; user can override.

Effort to implement: **~4-5 hours sub-GPU work** (nginx + frontend conditional rendering + 2 banner components).

---

## 1. Design rationale

### 1.1 Why three domains, not one

Three distinct audiences need different default experiences:

**Audience A: Researchers, clinicians, paying users → physicsworldmodel.org**
- They want to USE the platform (submit methods, view verified scores, pay for runs)
- They expect mainnet data by default (the production chain)
- They need wallet connect + AI4Science submission UI
- Most are non-technical; the URL "physicsworldmodel.org" is the obvious entry point

**Audience B: Developers, contributors, test miners → test.physicsworldmodel.org**
- They want to BUILD on the platform (test integrations, try new features)
- They need testnet by default (cheap, no real money at risk)
- They need full feature surface to test write paths
- The "test" subdomain makes the testnet context unambiguous

**Audience C: Auditors, analysts, journalists, casual browsers → explorer.physicsworldmodel.org**
- They want to LOOK at the platform (verify state, browse data, write articles)
- They don't need wallet connect or submission UI (just slows them down)
- They want mainnet data (the production state)
- The "explorer" subdomain matches the Etherscan/Basescan pattern they're familiar with

Conflating these audiences on one URL forces compromises: a researcher gets confused by wallet popups; an auditor wastes time figuring out how to skip past the submission UI; a developer accidentally tests against mainnet.

### 1.2 Why one container, not three

Three separate containers would mean:
- Three deployment pipelines (or one orchestrator)
- Three sets of indexer DBs (or shared volume + careful concurrency)
- Three sets of monitoring / log streams
- Three opportunities for version drift between environments

One container with header-based differentiation:
- One deployment pipeline (same Docker image, just different nginx routes)
- One set of indexer DBs (already serving all 3 chains)
- One log stream + one health endpoint
- Zero version drift — all 3 domains run the same code, just with different env

For PWM's scale (small protocol, single-founder ops), one container is the right choice. If we grow to multiple co-founders + sub-team-owned environments, we can split later.

### 1.3 Why the chain switcher still works on every domain

A user visiting `physicsworldmodel.org/principles?chain=baseSepolia` should see testnet principles. The DEFAULT chain differs per domain; the SUPPORTED chains are all 3 (Base mainnet, Base Sepolia, Eth Sepolia) on every domain. This preserves flexibility — a researcher who wants to test their method on testnet before mainnet submission can do that without changing domains.

---

## 2. What changes per domain

### 2.1 Default chain (when no `?chain=` query param)

The `ChainSwitcher` component reads a default from the server-side header injected by nginx:

| Domain | `X-PWM-Default-Chain` header value | UI default chain |
|---|---|---|
| physicsworldmodel.org | `base` | Base mainnet |
| test.physicsworldmodel.org | `baseSepolia` | Base Sepolia |
| explorer.physicsworldmodel.org | `base` | Base mainnet |

User can still switch via the dropdown — the default is just the initial state when arriving with no query param.

### 2.2 Feature mode (controls what UI elements are visible)

The `X-PWM-Mode` header controls what UI elements render:

| Domain | `X-PWM-Mode` value | What's shown | What's hidden |
|---|---|---|---|
| physicsworldmodel.org | `production` | Wallet connect, `/submit`, `/mine`, `/profile`, leaderboard, browse | (nothing hidden) |
| test.physicsworldmodel.org | `test` | All of above + "TESTNET" red banner at top | (nothing hidden) |
| explorer.physicsworldmodel.org | `read_only` | Browse (`/principles`, `/benchmarks`, `/leaderboard/*`, `/cert/*`), chain switcher, "READ-ONLY" gray pill in nav | Wallet connect button, `/submit`, `/mine`, `/profile/*` |

### 2.3 What does NOT change

- Indexer behavior (all 3 chains always indexed)
- API endpoints (same routes on every domain)
- SSL certificates (Let's Encrypt for all 3 domains; same auto-renewal)
- Chain switcher dropdown (always shows 3 chains; user can switch)
- Backend state (one container, one SQLite per chain in `/data/`)

---

## 3. Implementation plan (~4-5 hours)

### 3.1 nginx config (~15 min)

Modify the 3 existing nginx site configs at `/etc/nginx/sites-available/` on the VM to inject headers:

**`/etc/nginx/sites-available/physicsworldmodel.org`** (production mainnet):

```nginx
server {
    server_name physicsworldmodel.org;
    # ... existing SSL config ...
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header X-PWM-Default-Chain "base";
        proxy_set_header X-PWM-Mode "production";
        # ... existing proxy headers ...
    }
}
```

**`/etc/nginx/sites-available/test.physicsworldmodel.org`** (testnet dev):

```nginx
server {
    server_name test.physicsworldmodel.org;
    # ... existing SSL config ...
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header X-PWM-Default-Chain "baseSepolia";
        proxy_set_header X-PWM-Mode "test";
        # ... existing proxy headers ...
    }
}
```

**`/etc/nginx/sites-available/explorer.physicsworldmodel.org`** (read-only mainnet):

```nginx
server {
    server_name explorer.physicsworldmodel.org;
    # ... existing SSL config ...
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header X-PWM-Default-Chain "base";
        proxy_set_header X-PWM-Mode "read_only";
        # ... existing proxy headers ...
    }
}
```

**Note:** `test.physicsworldmodel.org` currently proxies to port 8101 (FastAPI pwm_nonprofit-app). Director needs to decide whether to keep that backend OR switch test. to port 3000 (the Next.js explorer with testnet default). Recommendation below in §6.

Then:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 3.2 Frontend: read headers in server components (~1 hour)

In `infrastructure/agent-web/frontend/app/layout.tsx` or a higher-order helper:

```typescript
import { headers } from 'next/headers';

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const h = headers();
  const defaultChain = h.get('X-PWM-Default-Chain') ?? 'base';
  const mode = h.get('X-PWM-Mode') ?? 'production';

  // Pass to client via a server-rendered <script> or React context
  return (
    <html lang="en">
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `window.__PWM_DEFAULT_CHAIN__ = ${JSON.stringify(defaultChain)};
                     window.__PWM_MODE__ = ${JSON.stringify(mode)};`,
          }}
        />
      </head>
      <body>
        {/* Pass via context too for SSR */}
        <ModeContext.Provider value={{ mode, defaultChain }}>
          {children}
        </ModeContext.Provider>
      </body>
    </html>
  );
}
```

### 3.3 Frontend: ChainSwitcher reads default from context (~30 min)

Update `infrastructure/agent-web/frontend/components/ChainSwitcher.tsx`:

```typescript
'use client';
import { useContext } from 'react';
import { ModeContext } from '@/context/ModeContext';

export function ChainSwitcher() {
  const searchParams = useSearchParams();
  const { defaultChain } = useContext(ModeContext);
  const activeChain = searchParams.get('chain') ?? defaultChain;
  // ... rest of component
}
```

### 3.4 Frontend: hide write UI when mode=read_only (~2 hours)

In nav component, `/submit` page, `/mine` page, `/profile/*` pages — wrap the write-flow UI with a mode check:

```typescript
'use client';
import { useContext } from 'react';
import { ModeContext } from '@/context/ModeContext';

export function WalletConnectButton() {
  const { mode } = useContext(ModeContext);
  if (mode === 'read_only') return null;  // hide entirely on explorer.physicsworldmodel.org
  return <RainbowKitButton />;
}
```

For routes — wrap the `/submit` page with a server-side redirect to `/principles` if mode is read_only:

```typescript
// app/submit/page.tsx
import { redirect } from 'next/navigation';
import { headers } from 'next/headers';

export default function SubmitPage() {
  const mode = headers().get('X-PWM-Mode') ?? 'production';
  if (mode === 'read_only') redirect('/principles');
  // ... rest of submit UI
}
```

### 3.5 Frontend: TESTNET banner on test.physicsworldmodel.org (~30 min)

Add a top-of-page banner component:

```typescript
'use client';
import { useContext } from 'react';
import { ModeContext } from '@/context/ModeContext';

export function ModeBanner() {
  const { mode } = useContext(ModeContext);
  if (mode === 'test') {
    return (
      <div className="bg-red-600 text-white text-center py-2 text-sm font-mono">
        🧪 TESTNET — You're on the test environment. Tokens here have NO real value.
        Production: <a href="https://physicsworldmodel.org" className="underline">physicsworldmodel.org</a>
      </div>
    );
  }
  if (mode === 'read_only') {
    return (
      <div className="bg-slate-700 text-slate-200 text-center py-1 text-xs font-mono">
        👁 READ-ONLY explorer. Submit your method at <a href="https://physicsworldmodel.org" className="underline">physicsworldmodel.org</a>
      </div>
    );
  }
  return null;
}
```

Then render `<ModeBanner />` in the root layout above the main content.

### 3.6 Test all 3 domains (~30 min)

Open in browser:
- https://physicsworldmodel.org/ → expect wallet connect visible; default chain = Base mainnet; no banner
- https://test.physicsworldmodel.org/ → expect TESTNET red banner; default chain = Base Sepolia
- https://explorer.physicsworldmodel.org/ → expect READ-ONLY gray pill; default chain = Base mainnet; wallet button hidden; `/submit` redirects to `/principles`

---

## 4. Edge cases + open questions

### 4.1 Should test.physicsworldmodel.org keep proxying to FastAPI (port 8101)?

Currently test.physicsworldmodel.org → port 8101 (the pwm_nonprofit-app FastAPI testing tier). This is unrelated to the Next.js explorer.

**Two options:**

| Option | Test domain serves | Pros | Cons |
|---|---|---|---|
| **A** | Next.js explorer with testnet default (port 3000) | Clean "test = testnet PWM explorer" semantic; matches the 3-domain design | Loses the FastAPI testing capability |
| **B** | Keep FastAPI testing app (port 8101) AS-IS | Preserves existing functionality; test = "non-prod platform tier" not "testnet explorer" | Confusing semantic — "test" doesn't mean testnet |

**Recommendation: Option A.** The "test" subdomain should mean "testnet PWM" — that's what new visitors expect. If the FastAPI testing app is still needed, move it to a different subdomain (e.g., `staging.physicsworldmodel.org` or `internal.physicsworldmodel.org`).

### 4.2 Should explorer.physicsworldmodel.org serve testnet too via `?chain=baseSepolia`?

**Yes** — the chain switcher still works on every domain. Visitors can browse testnet principles on the explorer subdomain by selecting `Base Sepolia` from the dropdown. The DEFAULT is mainnet, but users can override.

### 4.3 What if someone tries to submit on explorer.physicsworldmodel.org via direct URL (e.g., `https://explorer.physicsworldmodel.org/submit`)?

Server-side redirect to `/principles`. The `read_only` mode header is the source of truth, not the URL path.

### 4.4 What about API calls from a user who's looking at testnet on physicsworldmodel.org?

API routes already accept `?chain=base|baseSepolia|testnet` query parameter. The default fallback is `baseSepolia` (per `api/store.py` `_db_path()`). Frontend always passes the active chain in the API call, so this works regardless of which domain the user is on.

### 4.5 What if SSL certs expire on one of the 3 domains?

Let's Encrypt auto-renewal is scheduled (per certbot config from Step 5.5c). All 3 domains have certs from the same renewal cycle. If one fails, all 3 fail — but the failure is loud + email-notified.

---

## 5. What this DOESN'T solve

This design addresses URL routing + UI mode + chain defaults. It does NOT address:

- **Dev vs prod branch deployments.** If sub-GPU wants to deploy experimental code to test.physicsworldmodel.org without affecting physicsworldmodel.org, that requires running TWO containers (one per branch). See alternative architecture in §6.

- **A/B testing of new features.** Currently a feature either ships everywhere or nowhere. If we want gradual rollout (10% of physicsworldmodel.org users see new UI), that's a separate concern (feature flags + cookie-based variant selection).

- **Compute isolation.** All 3 domains share the same Node.js process + same SQLite files. If one domain has a traffic spike, all 3 are affected. For PWM's scale this is fine; for higher-scale protocols, separate containers would be needed.

---

## 6. Alternative architecture (two containers) — for future consideration

If Director wants stronger isolation between production and test:

| Container | Image | Port | nginx route | Branch |
|---|---|---|---|---|
| `pwm-explorer-prod` | `pwm-explorer:v2-2026-05-22` | 3000 | physicsworldmodel.org + explorer.physicsworldmodel.org | `master` |
| `pwm-explorer-test` | `pwm-explorer:v2-test-branch` | 3001 | test.physicsworldmodel.org | `test` (or `dev`, `next`) |

This means:
- Test container can deploy experimental code without affecting prod
- Test container can be restarted/crashed independently
- Test runs against Base Sepolia by default; prod against Base mainnet

But adds:
- 2x ops overhead (two containers to maintain)
- Risk of code drift between prod and test branches
- Doubles indexer resource use

**Recommendation: Stay with the single-container design for now.** Re-evaluate at Month 6+ when there's more dev activity OR when Director needs to test breaking changes safely.

---

## 7. Migration plan from current state to 3-domain design

### Current state (2026-05-23)

- physicsworldmodel.org → port 3000 (Next.js explorer; default chain `base`)
- explorer.physicsworldmodel.org → port 3000 (same backend; same default)
- test.physicsworldmodel.org → port 8101 (FastAPI pwm_nonprofit-app; unrelated)

### Target state (this doc)

- physicsworldmodel.org → port 3000 (Next.js explorer; mode=production, default chain=`base`)
- explorer.physicsworldmodel.org → port 3000 (same; mode=read_only, default chain=`base`)
- test.physicsworldmodel.org → port 3000 (same; mode=test, default chain=`baseSepolia`)

### Migration steps (in order)

1. **sub-GPU implements frontend changes** (§3.2-3.5) — 4 hours
2. **Build new Docker image** with the conditional UI — 15 min
3. **Update nginx configs** to inject headers (§3.1) — 15 min
4. **Restart container** with new image — 5 min
5. **Reload nginx** — 1 min
6. **Smoke test all 3 domains** (§3.6) — 30 min
7. **Decide on test.physicsworldmodel.org FastAPI app** (§4.1):
   - Option A: shut down port 8101 FastAPI container OR move to `staging.physicsworldmodel.org`
   - Option B: keep FastAPI; rename "test" subdomain purpose in this doc
8. **Update strategic docs** to reflect 3-domain design (Director or sub-GPU)

Total wall time: ~5-6 hours including testing + deploy. Single-day project for sub-GPU.

---

## 8. Decision points for Director

| # | Decision | Default | Notes |
|---|---|---|---|
| 1 | Approve 3-domain split (one container, header-based)? | YES (recommended) | This is the central architectural decision |
| 2 | "Static" = read-only-live (recommended) OR pre-rendered HTML? | **read-only-live** | Pre-rendered loses real-time indexer data |
| 3 | test.physicsworldmodel.org → Next.js explorer with testnet default? | YES (recommended) | Currently serves FastAPI; switch to align with "test" semantic |
| 4 | Where does the existing FastAPI testing app go if test. moves? | **`staging.physicsworldmodel.org`** | Or shut down if unused |
| 5 | Implementation owner | sub-GPU | ~4-5 hour single-day project |
| 6 | Timeline | After Wave 2 (D9+30+) | Not blocking; can wait until Phase 1b activation |

---

## 9. Cross-references

- `infrastructure/agent-web/nginx.example.conf` — nginx template (will be updated when sub-GPU implements §3.1)
- `infrastructure/agent-web/frontend/components/ChainSwitcher.tsx` — chain switcher (already exists; will be updated per §3.3)
- `infrastructure/agent-web/frontend/app/providers.tsx` — already sets `chains: [base, baseSepolia]`; default chain swap needed
- `coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` — phase sequencing context
- `coordination/PWM_VALUE_FRAMING_2026-05-22.md` — AI4Science framing context
- `mainnet/strategy/THREE_DOMAIN_ARCHITECTURE.md` — public-repo mirror (will be added when this commits)

---

## 10. The single sentence

**All 3 domains serve the same Docker container on port 3000 with the same data; nginx-injected headers differentiate the default chain (mainnet vs testnet) and feature mode (full vs read-only), and per-mode banners make the environment obvious — totalling ~4-5 hours of sub-GPU work.**

---

*This doc is the canonical architecture for the 3-domain split. Update when sub-GPU implements; mark decisions in §8 as RESOLVED post-implementation.*
