# Piano Performance Marker — Deployment & Data Ops

This app runs on Express (server.js) and serves the React production build from `build/`. It exposes JSON endpoints to save and browse evaluations.

Production URL (Railway)
- https://piano-performance-marker-production.up.railway.app

Local Paths
- Local evaluations (runtime): `piano-performance-marker/evaluations/`
- Downloaded evaluations (from Railway): `piano-performance-marker/downloaded_evaluations/`

## Run Locally
- Install deps: `npm install`
- Build: `CI=false npm run build`
- Start: `npm start` → http://localhost:3000
- Health: `GET /healthz`

## API Endpoints (Server)
- Save: `POST /api/save-evaluation` with JSON `{ username, filename, data }`
- List users: `GET /api/users`
- List by user: `GET /api/evaluations/:username` (use `?withData=1` to include bodies)
- Get one: `GET /api/get-evaluation?username=...&filename=...`
- Direct file: `GET /files/:username/:filename`
- Latest by audio: `GET /api/evaluation-latest?username=...&audio=...`
- Last index: `GET /api/last-evaluation/:username`

Admin (protected)
- Set `ADMIN_TOKEN` on the server (Railway Variables).
- Delete one file: `DELETE /api/evaluation/:username/:filename` with header `x-admin-token: <ADMIN_TOKEN>`
- Delete all for user: `DELETE /api/evaluations/:username` with header `x-admin-token: <ADMIN_TOKEN>`

## Railway Notes
- Build/Start: Railway Nixpacks builds React (`npm install`, `CI=false npm run build`) and starts `node server.js`.
- Persistence: attach a Volume and mount to `/app/evaluations` so saved JSON survives restarts.
- Variables: set `ADMIN_TOKEN` to enable admin delete APIs. Optional `EVALUATIONS_DIR` (default: `/app/evaluations`).

## Sync/Manage Data From Railway
Helper script: `scripts/sync_from_railway.py` (requires `requests`: `pip install requests`)

Defaults
- Base: `https://piano-performance-marker-production.up.railway.app`
- Output: `piano-performance-marker/downloaded_evaluations/`

Examples
- Sync everything:
  - `python piano-performance-marker/scripts/sync_from_railway.py`
- Sync one user into repo evaluations/ (merge):
  - `python piano-performance-marker/scripts/sync_from_railway.py --user test --merge-into-local-evals`
- Delete all for a user on Railway (admin):
  - `ADMIN_TOKEN=xxxx python piano-performance-marker/scripts/sync_from_railway.py --delete-user test --yes`
- Delete one file on Railway (admin):
  - `ADMIN_TOKEN=xxxx python piano-performance-marker/scripts/sync_from_railway.py --delete-file test amateur_piano_1.json --yes`

## Housekeeping
- This repo uses Railway for deployment. Artifacts/configs for other hosting providers are not required.
- Data you download from Railway is stored under `piano-performance-marker/downloaded_evaluations/` by default.
