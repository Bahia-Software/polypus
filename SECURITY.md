# Security Policy

## Supported versions

Security fixes are released for the latest `0.x` minor version only. Polypus is
pre-1.0, so older minors do not receive backports; upgrade to the latest release.

| Version | Supported |
| ------- | --------- |
| Latest `0.x` minor (currently 0.7) | Yes |
| Older releases | No |

## Reporting a vulnerability

Please do **not** open a public issue, pull request or discussion for a
suspected vulnerability.

Report it privately through GitHub:

1. Open the [Security tab](https://github.com/Bahia-Software/polypus/security) of the repository.
2. Choose **Report a vulnerability**.
3. Describe the problem and, where possible, include the affected version, a
   minimal reproduction and the expected impact.

If the **Report a vulnerability** option is not available, open a public issue
that asks the maintainers for a private contact channel and contains no
technical details.

## What to expect

- Acknowledgement of your report within 5 working days.
- An assessment and, if the report is confirmed, a fix and a coordinated
  disclosure date agreed with you.
- Credit in the release notes and the published advisory, unless you prefer to
  stay anonymous.

Polypus is maintained by a small team, so these are targets rather than
guarantees.

## Scope

In scope: the Rust crates and the `polypus-quantum` Python package in this
repository, including the backend bridges (QMIO, CUNQA, subprocess workers) and
the OpenQASM importers.

Out of scope: vulnerabilities in third-party dependencies (report them
upstream; we will update once a fix exists), and in the quantum services or
clusters that Polypus connects to.
