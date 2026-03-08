# BaaS (Benchmarking-as-a-Service) Revenue Playbook

BaaS only works if you create a situation where vendors **must** appear on your leaderboard. The mechanism is a **credibility flywheel**:

```
Academic citations → Procurement references → Vendor pressure → Revenue
```

Each stage must be deliberately built.

---

## Stage 1: Make PWM the Cited Standard (Months 0–12)

**What to do:**

1. **Publish the papers** — Nature (flagship) + Nature Methods (PWM-SyS). This is already in progress.

2. **Seed citations in other people's papers** — When research groups publish new reconstruction algorithms, they need a benchmark to compare against. Make PWM the easiest option:
   - Provide a one-line Python API: `pwm.evaluate(my_reconstruction, variant="ct")`
   - Pre-compute baseline scores so authors can write "Our method achieves 38.2 dB on PWM-CT, compared to 36.1 dB for the previous SOTA"
   - Reach out to 10–20 active computational imaging groups and offer to run their algorithms on PWM for free

3. **Conference presence** — Submit workshop papers / tutorials at SPIE Photonics West, IEEE ISBI, MICCAI, Optica Imaging Congress. These are where the vendor reps attend.

4. **Metrics**: Target 50+ papers citing PWM scores within 12 months.

**Why this ensures revenue later**: Once 50+ papers use PWM scores, it becomes the de facto reference. Reviewers start asking "What's your PWM score?" — at that point, vendors can't ignore it.

---

## Stage 2: Enter the Procurement Chain (Months 6–18)

This is the critical step most academic platforms miss. Citations alone don't generate revenue — **purchasing decisions** do.

**What to do:**

1. **Publish an annual "PWM Imaging Systems Report"** — A free PDF ranking all 168 systems across the 8 adequacy dimensions. Think of it like the Gartner Magic Quadrant for imaging:
   - "Top 5 systems for clinical brain imaging under $500K"
   - "Most cost-effective NDT systems for aerospace"
   - Make it downloadable with email capture (builds your lead list)

2. **Partner with hospital procurement committees** — Hospitals spend $1M–$50M on imaging equipment. Their procurement teams need objective comparisons. Offer PWM reports as decision-support documents. Start with 2–3 academic medical centers where you have connections.

3. **Partner with government/funding agencies** — If NIH or NSF grant applications start requiring "benchmark your system on PWM," every funded lab becomes a user, and every vendor whose equipment they're buying needs PWM scores.

4. **Write procurement-oriented content** — White papers like "How to Select a Computational Imaging System: A Data-Driven Framework." Target lab managers and department heads, not researchers.

**Why this ensures revenue**: When a hospital's procurement committee uses your report to decide between Siemens and GE, both vendors will want to ensure their scores are accurate and favorable. That's when they call you.

---

## Stage 3: Create the Revenue Trigger (Months 12–24)

**The key insight**: Vendors don't pay for benchmarking — they pay for **control over their narrative**. Structure your offering so that paying gives them something they can't get otherwise.

### Pricing Tiers

| Tier | Price | What They Get |
|------|-------|---------------|
| **Free** | $0 | Listed on public leaderboard with PWM-computed scores (using published data only) |
| **Verified** | $5K–$15K | Submit proprietary test data → PWM runs evaluation on hidden test sets → "PWM Verified" badge + detailed report |
| **Certified** | $15K–$50K | Full 8-dimension adequacy profile + annual re-certification + right to use "PWM Certified" logo in marketing materials + priority listing |
| **Custom** | $50K+ | Custom benchmark design for new product launches, white-label comparison reports for sales teams |

### Why Vendors Will Pay

1. **Hidden test sets** — Your hidden challenge tier data is something vendors cannot self-evaluate against. Only PWM controls access. This is your moat.

2. **The "PWM Certified" badge** — Once procurement committees reference PWM, vendors need the badge. Similar to how "Energy Star" certification drives appliance manufacturers to pay for testing.

3. **Competitive pressure** — When Siemens is "PWM Certified" and GE is not, GE's sales team will hear about it from every procurement committee. GE will pay to get certified.

4. **Accuracy disputes** — If PWM's free-tier score underestimates a vendor's system (because it uses published data, not their optimized configuration), the vendor's only option to correct this is to pay for a Verified evaluation. This naturally drives upgrades.

---

## Stage 4: Lock-In Mechanisms

Once revenue starts, make it recurring:

1. **Annual re-certification** — Scores expire after 12 months. Vendors must re-certify annually ($10K–$30K/year recurring).

2. **Version upgrades** — When PWM releases v2.0 benchmark (new test sets, new metrics), all existing certifications are marked "v1.0." Vendors must re-certify on v2.0 to stay current.

3. **Category expansion** — As you add new modalities or sub-challenges, vendors in those spaces must benchmark separately.

---

## Concrete First Actions (This Month)

| # | Action | Purpose |
|---|--------|---------|
| 1 | Add `pwm.evaluate()` Python package to PyPI | Make it trivial for researchers to cite PWM scores |
| 2 | Email 10 computational imaging PIs offering free benchmark runs | Seed early citations |
| 3 | Draft the first "PWM Imaging Systems Report" PDF | Procurement-ready artifact |
| 4 | Add "PWM Verified" badge placeholder to the leaderboard UI | Signal that paid verification exists |
| 5 | Submit Nature Methods paper | Credibility foundation |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Nobody cites PWM | Make the Python API so easy that it's less work than NOT using it |
| Vendors ignore PWM | Target procurement committees first — vendors follow buyer behavior |
| Vendors replicate PWM internally | Hidden test sets + independent third-party trust = cannot be replicated |
| Free tier is "good enough" | Free tier uses only published data; Verified tier uses vendor's optimized config on hidden sets — significant quality gap |
| Market too small | 168 modalities × ~5 vendors each = ~840 potential customers at $10K avg = $8.4M TAM |

---

## Bottom Line

**BaaS revenue is not automatic from having a benchmark. It requires deliberately inserting PWM into the procurement decision chain.** The papers create awareness, the annual report creates procurement relevance, and the hidden test sets create the moat that forces vendors to pay.
