# Deploying VoiceGuard

> **Heads-up (2026):** Hugging Face now requires a PRO subscription to *create*
> new Docker/Gradio Spaces on free accounts, so the agent deploys elsewhere.
> Your existing detection Space keeps running — **do not rebuild it**.
> The same Dockerfile works on Railway, Render, and HF (it binds `$PORT`).

---

## Option A — Railway (recommended for the hackathon)

Free **$5 / 30-day trial, no credit card**. Always-on during the trial — no
cold starts in front of judges. Sign up with **GitHub** (unverified accounts
get restricted outbound network, which breaks web research).

### CLI path (no git repo needed)

```powershell
npm i -g @railway/cli        # or: scoop install railway
railway login                # opens the browser
railway init                 # create project "voiceguard-agent"
railway up                   # uploads this folder, builds the Dockerfile
railway domain               # generates the public URL
```

Then open the project dashboard → **Variables** → add:

| Variable | Value |
|---|---|
| `GEMINI_API_KEY` | your Google AI Studio key |
| `GROQ_API_KEY` | your Groq key |
| `VOICE_API_KEY` | your detection API key (see `.env`) |
| `VOICE_API_URL` | `https://pandaisop-voice-detection-api.hf.space/test` |
| `VOICE_API_AUTH_MODE` | detector auth transport: `x-api-key` (default), `bearer`, or `body` |
| `AGENT_API_KEY` | strong random key required by all POST endpoints |

Railway redeploys automatically after variables change. Verify:

```powershell
python verify_live.py https://<your-service>.up.railway.app
```

`.railwayignore` keeps `.env` out of uploads. `railway.json` sets the
`/health` healthcheck.

### No Node/npm?

Push the folder to a GitHub repo instead (GitHub → New repository → *uploading
an existing file* supports drag-and-drop of the whole folder, minus `.env`),
then railway.com → **New Project → Deploy from GitHub repo** → add the same
variables.

### After the trial

The free plan drops to $1/month of usage credit, which may not keep the
service alive. Either upgrade to Hobby ($5/mo) or switch to Render (below) —
same code, no changes.

---

## Option B — Render (free forever, sleeps when idle)

No card; 750 h/month; spins down after 15 min idle (first request takes
30-60 s to wake — open the UI a minute before demoing).

1. Push this folder to a GitHub repo (drag-and-drop upload works; exclude `.env`).
2. render.com → sign in with GitHub → **New + → Blueprint** → pick the repo —
   `render.yaml` configures everything and prompts for the secret values.
3. Verify: `python verify_live.py https://voiceguard-agent.onrender.com`

---

## Option C — Hugging Face PRO ($9/mo)

If you subscribe at <https://huggingface.co/pro>, the original one-command
deploy works unchanged and keeps both services on one platform:

```powershell
python deploy.py
# Or, only for an intentionally open rate-limited demo:
# python deploy.py --public-demo
```

(Creates the Space, sets secrets from `.env`, uploads, builds, smoke-tests.)

---

## After any deploy

- **UI** at the base URL · **Docs** at `/docs` · **Health** at `/agent/health`
- Integrating from other projects: see [INTEGRATE.md](INTEGRATE.md)
- Rotate/change keys: edit the platform's variables and redeploy/restart.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Application failed to respond" on the public URL while logs show the app healthy | Ensure the platform routes to its injected `$PORT`; `start.sh` binds that port (default 7860). |
| Build fails | Check the platform's build logs (usually a missing file — re-upload) |
| `/agent/health` shows `none (heuristic fallback)` | `GEMINI_API_KEY` variable missing or typo'd |
| `/detect` returns 502 | Detection Space asleep (first call wakes it, ~1 min) or request-shape mismatch — set variable `VOICE_API_AUDIO_FIELD`, or send Claude the detection API's request JSON |
| 401 on POSTs | Send the configured key in `X-API-Key`. |
| 503 `API authentication is not configured` | Set `AGENT_API_KEY`, or explicitly set `ALLOW_PUBLIC_DEMO=true` for a public demo. |
| 429 on POSTs | The process-level request budget was reached; honor `Retry-After` or tune `RATE_LIMIT_REQUESTS`. |
| Railway build has no network | Trial not GitHub-verified — link GitHub in account settings |
