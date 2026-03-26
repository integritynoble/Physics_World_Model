# PWM Protocol Governance -- Decision Authorities

Per Dyson Swarm Strategy S14.

## Decision Matrix

| Decision | Authority | Process | Timeline |
|----------|-----------|---------|----------|
| **spec.core changes** | Core team (unanimous) | Written RFC -> 2-week review -> vote | 2-4 weeks |
| **Primitive registry changes** | Registry maintainer + core team approval | PR with CHANGELOG entry -> core team review | 1-2 weeks |
| **Judge kernel changes** (R1-R4 logic) | Core team (unanimous) | Written RFC -> backward-compat analysis -> regression test update | 2-4 weeks |
| **DomainProfile conflicts** | Domain maintainer + core team mediation | Issue filed -> mediation meeting -> resolution PR | 1-2 weeks |
| **Trust-tier rule changes** | Core team + benchmark reviewer consensus | Written RFC -> reviewer consultation -> joint decision | 2-4 weeks |
| **Certificate schema changes** | Core team (unanimous) | Written RFC -> semver analysis -> migration plan | 4+ weeks |

## Changelog Requirement

Every change to protocol semantics must be accompanied by:
1. A CHANGELOG entry in the affected artifact's directory
2. A version bump (minor for additions, major for breaking)
3. A compatibility-matrix update if cross-cutting

## Contact

Protocol governance questions: file an issue at github.com/integritynoble/Physics_World_Model
