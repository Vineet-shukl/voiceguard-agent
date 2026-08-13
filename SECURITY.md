# Security

## Secure deployment defaults

VoiceGuard protects all POST endpoints unless one of these configurations is
present:

- Recommended: set a strong `AGENT_API_KEY` and send it in `X-API-Key`.
- Demo only: explicitly set `ALLOW_PUBLIC_DEMO=true`. Process-local rate and
  concurrency limits still apply, but an Internet-facing demo can still consume
  provider quotas.

Additional controls:

- `MAX_BODY_BYTES` defaults to 36 MB and is enforced against actual ASGI body
  bytes, including chunked requests.
- `RATE_LIMIT_REQUESTS` defaults to 20 requests per
  `RATE_LIMIT_WINDOW_SECONDS` (60 seconds) per worker process.
- `POST_CONCURRENCY` defaults to 2 protected requests per worker process.
- Audio is limited to 24 MB decoded and format/language fields are validated.
- `VOICE_API_AUTH_MODE` selects exactly one detector credential transport:
  `x-api-key` (default), `bearer`, or `body`. Credentials are never placed in
  detector URLs.

For multi-worker or multi-instance deployments, enforce distributed rate limits,
request-size limits, authentication, and quotas at the ingress/API gateway too.
The in-process controls are defense in depth, not a replacement for edge limits.

## If an older version was deployed

Older builds could run with unauthenticated POST endpoints, put detector keys in
query strings after authentication failures, disable TLS verification when CA
setup failed, and include `.env` in a Docker image built without `.dockerignore`.

Treat exploitation as unconfirmed until runtime evidence is reviewed. Check:

1. Reverse-proxy, Railway, Render, Hugging Face, and detector access logs for
   unexpected POST volume, repeated 401/403/413/429 responses, unusual source
   addresses, and URLs containing `key=` or `api_key=`.
2. Gemini, Google Cloud, Groq, OpenRouter, and detector provider usage/billing
   for quota spikes, unfamiliar regions, or requests outside expected hours.
3. Container registry and build history for images created before
   `.dockerignore` existed. Inspect or delete those images and build caches.
4. Deployment audit logs for secret reads/changes, environment changes, new
   collaborators, unexpected deploys, or altered domains.
5. Git hosting audit logs for unfamiliar tokens, SSH keys, force pushes,
   workflow changes, deploy keys, webhooks, or collaborator changes.
6. Running containers/hosts for unknown processes, scheduled tasks, outbound
   connections, mounts, or modified files. This cannot be established from the
   repository alone.

If any older image may have contained `.env`, or detector keys may have appeared
in URL logs, rotate all affected keys immediately. Revoke old keys rather than
only creating additional ones, purge retained logs/images where appropriate,
and redeploy from a clean build.

## Repository audit status

A source and Git-history review on 2026-08-13 found no tracked credentials,
recognized secret signatures, malicious persistence, command execution,
obfuscated payloads, or unexpected exfiltration code. The outbound detector and
AI-provider calls are intentional application behavior and remain external data
trust boundaries.

Dependency auditing is performed with:

```bash
python -m pip_audit -r requirements.txt
```

Run both offline suites after dependency or security-control changes:

```bash
python tests/test_offline.py
python tests/test_adapter_offline.py
```
