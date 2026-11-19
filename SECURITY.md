# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.4.x | Yes |
| < 1.0 | No |

## Reporting a vulnerability

Please report security issues privately to **neuratal@gmail.com** with the
subject line `SECURITY: NTransformer — <short description>`. Do not open a
public GitHub issue for vulnerabilities that could compromise users.

Include:

- Affected component and version (commit SHA).
- Impact assessment and, if applicable, a proof-of-concept.
- Any suggested mitigation.

You can expect an acknowledgement within 72 hours and, once the issue is
confirmed, a public disclosure timeline agreed with you or immediate patch
release — whichever applies.

## Scope

- Model checkpoints produced by this project are **not** signed; always
  verify `checksums`/`trainer_meta.json` fingerprints where available before
  loading third-party checkpoints.
- This project executes Python; treat any checkpoint from an untrusted source
  as untrusted input. `safetensors` is used precisely to avoid pickle
  deserialization; do not introduce pickle-based loading paths.
- Training data is explicitly out of scope: callers are responsible for the
  legality and ethics of their corpora.
- Do not report "model produces wrong/bad answers" as a vulnerability; this
  is a quality issue for the untrained development presets and is documented.

## Known non-vulnerabilities

- Training checkpoints embed optimizer state; they reveal nothing about
  private data unless the caller's data is itself sensitive.
- The development preset generates random output; this is expected.
