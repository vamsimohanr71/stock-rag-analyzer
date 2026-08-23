# Deploying to Streamlit Community Cloud (free)

This folder is a **standalone, single-app** version of the Stock RAG
Analyzer - no separate FastAPI backend, no Docker required. Everything
runs as one Streamlit app, which is what Streamlit Community Cloud hosts
for free.

## What's different from the backend/ + frontend/ version

- No FastAPI, no `uvicorn`, no separate backend process
- `app.py` calls `ingest_ticker()`, `retrieve_context()`,
  `generate_analysis()` etc. directly as Python functions
- Config reads from Streamlit's built-in secrets manager on the cloud,
  and falls back to a local `.env` file for local testing - same code,
  both places

## Local test before deploying

```bash
cd streamlit_cloud
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your real NVIDIA_API_KEY and FINNHUB_API_KEY
streamlit run app.py
```

## Deploy steps

1. **Push this folder to a public (or private) GitHub repo.** Streamlit
   Community Cloud deploys directly from a GitHub repo.
   - Easiest: push the whole `stock_rag_app` repo, but tell Streamlit Cloud
     the app's "Main file path" is `streamlit_cloud/app.py` (see step 4).
   - Make sure `.streamlit/secrets.toml` is **not** committed (it's already
     in `.gitignore` here) - real keys never belong in git.

2. **Go to https://share.streamlit.io** and sign in with GitHub.

3. Click **"New app"**, pick your repo and branch.

4. Set:
   - **Main file path**: `streamlit_cloud/app.py` (if you pushed the whole
     project) or just `app.py` (if you pushed only this folder as its own repo)

5. Click **"Advanced settings" -> Secrets**, and paste:
   ```toml
   NVIDIA_API_KEY = "nvapi-your-real-key"
   FINNHUB_API_KEY = "your-real-key"
   ```

6. Click **Deploy**. First build takes a few minutes (installing
   dependencies). You'll get a public URL like
   `https://your-app-name.streamlit.app`.

## Known limitations of this free setup

- **Storage is ephemeral.** Streamlit Community Cloud's filesystem resets
  on redeploys, sleep/wake cycles (free apps sleep after inactivity), and
  periodic maintenance. `chroma_store/` will NOT persist long-term -
  every "cold start" effectively starts with an empty vector DB, and
  ingestion re-runs as needed when you analyze a ticker. This is fine for
  a demo/personal tool; it is not durable storage.
- **Free-tier rate limits still apply** - NVIDIA's free NIM tier (~40
  req/min) and Finnhub's free tier (60 calls/min) are shared across
  however many people use your deployed app at once. Fine for personal
  or small-scale use, not for real traffic.
- **No background scheduler.** `scheduler.py` (periodic watchlist
  ingestion) isn't wired into this version, since Streamlit Cloud doesn't
  run a persistent background process outside the app's own request
  handling. Ingestion here is on-demand only (when a ticker is analyzed).
- If you outgrow these limits: the `backend/` + `frontend/` (FastAPI +
  Docker) version in this project is the path to a "real" persistent
  deployment - but that requires paid hosting with persistent storage
  (a small VPS, Render's paid tier with a disk, etc.), since free tiers
  generally don't offer durable storage for two coordinated services.
