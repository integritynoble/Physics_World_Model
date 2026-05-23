# Bounty 10 — PWM Mobile UX

- **Amount:** 40,000 PWM
- **Opens:** when Bounty 2 (Web UI / Explorer) winner declared OR Track 3c verification routes shipped — earliest ~D9 + 60 days
- **Reference implementation:** Bounty 2 reference impl at `infrastructure/agent-web/` (mobile-first extension)
- **Acceptance harness:** Mobile-specific tests at `infrastructure/agent-web/tests/mobile/` + visual regression suite

## What you build

A mobile-first PWM dApp that lets researchers, miners, and explorers use PWM from their phones. Three subsystems:

1. **Mobile-responsive frontend** — every Bounty 2 route (`/`, `/principles`, `/benchmarks/<id>`, `/cert/<hash>`, `/submit`, etc.) renders correctly on mobile screens (iPhone SE through iPad Pro)
2. **Mobile wallet flows** — WalletConnect v2 + mobile-native MetaMask + Coinbase Wallet + Rainbow Wallet flows for stake/sign/submit transactions
3. **Mobile-optimized gas tooling** — gas estimation + signing UX that doesn't require desktop-class screen real estate

**Why this matters:** 60% of web traffic is mobile. PWM's Bounty 2 ships desktop-optimized; without mobile-quality UX, PWM excludes 60% of potential miners. Mobile is especially important for the academic research audience: junior PhD students + postdocs are heavy mobile users.

## Mandatory features

| Feature | Detail |
|---|---|
| Responsive across 3 device categories | iPhone SE (375×667), iPhone Pro (390×844), iPad (768×1024). All Bounty 2 routes legible + usable |
| WalletConnect v2 integration | Standard QR-code + deep-link flow for mobile wallets (MetaMask Mobile, Coinbase Wallet Mobile, Rainbow, Trust) |
| Mobile transaction signing UX | Confirm-dialog UX that fits mobile screens; gas estimation in human-readable format |
| Mobile-aware indexer / API consumption | Caching strategy that respects mobile data + battery (no aggressive polling on cell network) |
| Mobile-specific submission flow | `/submit` route adapted for mobile keyboard + form-field constraints (no horizontal scroll, no truncated copy-paste fields) |
| Accessibility | WCAG 2.1 AA compliance for color contrast + touch target size; screen-reader-tested with VoiceOver + TalkBack |
| PWA support | Installable as a Progressive Web App (manifest.json + service worker); standalone mode without browser chrome |
| iOS Safari compatibility | Specific testing for iOS Safari quirks (no infinite scroll on momentum-scroll; correct viewport handling; touch-action management) |
| Android Chrome compatibility | Specific testing for Android Chrome (back-button behavior; WebView limitations) |

## Interface contract

- **Source of truth for protocol data:** Bounty 2 indexer + API (no separate indexer required)
- **Address mapping:** `interfaces/addresses.json` — must support both Sepolia + mainnet
- **Wallet provider abstraction:** Use WalletConnect v2, Privy, or RainbowKit (any of three OK)
- **Routing:** All URL paths from Bounty 2 must work; mobile is responsive overlay, not separate site

## What must pass

1. **Mobile-first viewport tests.** All Bounty 2 mandatory routes tested at 320×568 (iPhone SE 1st gen — the smallest realistic device PWM supports) through 768×1024 (iPad Pro). Every page must be navigable, readable, and functional. Manual + automated visual regression tests during shadow run.

2. **Mobile wallet roundtrip.** Submit + sign + finalize a test L4 certificate end-to-end on:
   - iPhone 12 or later + Safari + MetaMask Mobile
   - Android 12 or later + Chrome + Coinbase Wallet
   - At least one tablet (iPad or Android tablet)
   Per device family, must complete the full miner-submission flow without errors.

3. **Lighthouse performance.** Lighthouse mobile score:
   - Performance ≥ 80
   - Accessibility ≥ 90 (this is the WCAG 2.1 AA target)
   - SEO ≥ 80
   - Best practices ≥ 90
   Measured on `https://m.physicsworldmodel.org` (or whatever subdomain the winner hosts at)

4. **Touch target compliance.** All interactive elements (buttons, links, form fields) ≥ 44×44 px touch target per Apple HIG / Material guidelines.

5. **No horizontal scroll.** Every page must fit horizontally in viewports as narrow as 320px without horizontal scrolling.

6. **PWA installability.** Browser shows "Install" prompt on supported mobile browsers; installed PWA opens in standalone mode without browser chrome.

7. **Availability.** ≥ 99% uptime measured over 30-day shadow run.

## What you may change

- Framework: React Native / Next.js mobile / Capacitor / SolidJS / Svelte / Vue all acceptable, IF they meet the responsive + PWA + Lighthouse criteria. **Recommended: Next.js + Tailwind + RainbowKit for fastest delivery.**
- Hosting: any free or paid tier; must be publicly reachable
- Styling: any approach; design quality not graded BUT must pass accessibility + touch-target requirements
- Component library: Tailwind, shadcn/ui, MUI, Chakra, custom — all OK
- Wallet provider stack: WalletConnect v2 + Privy OR RainbowKit OR Web3Modal — any major mainstream option

## What you may not change

- The Bounty 2 URL paths (mobile must use the same routes)
- The set of mandatory features above (adding more is fine; missing any voids award)
- WalletConnect v2 protocol compliance (no proprietary wallet protocol extensions)
- WCAG 2.1 AA accessibility minimums

## Shadow run

Your mobile dApp runs alongside any reference for **30 days**. agent-coord:

- Tests the full miner-submission flow weekly on 3 device families (iPhone latest gen, Android flagship, iPad)
- Monitors Lighthouse scores continuously
- Samples user feedback from 10+ first-time mobile users
- Tracks crash rate via standard error monitoring (Sentry / Rollbar / etc.) — < 1% crash rate target
- Disagreement with reference on protocol data > 0 wei / 0.0001 Q OR Lighthouse score regressions OR > 1% crash rate triggers investigation; three unexplained regressions void the award

## Why mobile-first matters for PWM specifically

The PWM target audience splits roughly:

| User type | Mobile traffic % | Why |
|---|---|---|
| Researchers browsing principles | 50-60% | Casual exploration on phone; deep work on desktop |
| Miners submitting certs | 20-30% | Submission requires keyboard + dataset access (desktop preferred) |
| Investors / journalists / grant reviewers | 70-80% | Quick due-diligence checks on mobile |
| AI assistants querying PWM | N/A (Bounty 9 territory) | — |

A desktop-only PWM excludes ~50% of the addressable audience for browsing + due-diligence. Mobile UX is **not optional** for a Year 1 priority protocol per `coordination/PWM_API_VS_WEBSITE_2026-05-20.md` §3 priority list (item #7 = "Public documentation portal" includes mobile-quality docs).

## Strategic context

- Bounty 2 (Web UI / Explorer) ships desktop-first; Bounty 10 extends it to mobile
- Bounty 10 can be claimed by Bounty 2's winner OR by a separate developer
- If a single developer ships both Bounty 2 + Bounty 10, total is 120,000 PWM (80K + 40K)
- Single winner is fine; encourages quality over fragmentation

## Payment

- Paid from Reserve discretionary pool (Layer 3 per `coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` §4)
- Wallet listed in PR description; confirm on Sepolia before opening
- Mainnet swap: 1:1 at Phase 2 launch

## References

- WalletConnect v2: https://docs.walletconnect.com/2.0
- Lighthouse: https://developer.chrome.com/docs/lighthouse/
- WCAG 2.1: https://www.w3.org/WAI/WCAG21/quickref/
- RainbowKit (recommended): https://rainbowkit.com/
- Privy (PWM's current desktop choice): https://privy.io/
- Apple HIG touch targets: https://developer.apple.com/design/human-interface-guidelines/
- Material Design touch targets: https://m3.material.io/foundations/accessible-design/accessibility-basics
