# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in PWM, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, email: **integrityyang@gmail.com** with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact assessment
4. Any suggested fixes (optional)

We will acknowledge receipt within 48 hours and provide an initial assessment
within 7 days.

## DICOM and Patient Data

PWM's clinical modules (CT QC Copilot) are designed to process **phantom data
only**. The DICOM ingester includes PHI (Protected Health Information)
validation with 20 sensitive DICOM tags and 7 phantom-pattern regexes. Strict
mode rejects non-phantom studies.

**Important:**

- Never commit real patient DICOM data to this repository.
- Never include PHI in issue reports, pull requests, or discussions.
- The PHI filter is a safety net, not a guarantee. Always verify that data
  is de-identified before processing.
- Clinical deployment requires additional validation per IEC 62304 and
  institutional IRB approval.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Security Considerations for Contributors

- Do not commit credentials, API keys, or tokens.
- Do not commit patient or human-subject data.
- Use SHA-256 integrity hashing for all audit-trail artifacts.
- Follow the immutable baseline pattern (never overwrite, always version).
- Report any PHI filter bypasses immediately.

## Disclosure Policy

We follow coordinated disclosure. We will:

1. Confirm the vulnerability and determine its impact.
2. Develop and test a fix.
3. Release the fix and notify affected users.
4. Credit the reporter (unless anonymity is requested).
