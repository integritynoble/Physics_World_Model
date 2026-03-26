# Contributing to PWM

## How to grow PWM

First ask: **"Can this be added as a DomainProfile, diagnostic module, card, or registry entry?"**
Only if the answer is no should you propose a sun-level change.

## Adding a new domain (Stage A -> B -> C)

1. **Stage A (Explicit)**: Create `spec.<domain>.md` mapping your domain to (Omega, E, B, I, O, epsilon)
2. **Stage B (Executable)**: Add a DomainProfile + primitive mappings; run one end-to-end test
3. **Stage C (Certifiable)**: Add domain diagnostics; produce a golden RunBundle

See `dyson_swarm_strategy.md` S6 "How PWM turns a science into explicit, executable, certifiable objects"

## Proposing a sun-level change

If your change must enter the deep sun:
1. File an RFC using `docs/governance/RFC_TEMPLATE.md`
2. The RFC must pass the **sun admission checklist**:
   - [ ] Shared across multiple domains
   - [ ] Needed by semantic kernel, trust kernel, or both
   - [ ] Stable enough to semver-lock
   - [ ] Has compatibility-matrix implication
   - [ ] Has regression tests
3. Core team reviews within 2 weeks
