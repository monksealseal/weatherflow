# WeatherFlow Deployment Guide

Deploy WeatherFlow publicly using one of four methods.

---

## Option 1: Railway (Recommended)

Railway provides the simplest deployment with automatic builds from GitHub.

### Deploy the API Backend

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **"New Project"** > **"Deploy from GitHub Repo"**
3. Select the `weatherflow` repository
4. Railway auto-detects `railway.json` and deploys the FastAPI API
5. Set environment variables (if needed):
   - `PYTHONUNBUFFERED=1`
6. Your API will be live at `https://<your-app>.up.railway.app`

### Deploy the Streamlit App (second service)

1. In the same Railway project, click **"New Service"** > **"GitHub Repo"**
2. Select `weatherflow` again
3. Override the config by setting:
   - **Start Command**: `streamlit run streamlit_app/Home.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Health Check Path**: `/_stcore/health`
4. Set environment variable:
   - `WEATHERFLOW_API_URL=https://<your-api-service>.up.railway.app`
5. Your Streamlit app will be live at `https://<your-streamlit>.up.railway.app`

> Alternative: Use `deploy/railway-streamlit.json` as the config file for the Streamlit service.

### Cost

Railway free tier: 500 hours/month, $5/month for the Starter plan.

---

## Option 2: Docker Compose (VPS / Local Server)

Deploy the full stack on any server with Docker installed.

### Prerequisites

- Docker and Docker Compose installed
- A server with at least 2GB RAM (4GB recommended for PyTorch)

### Deploy

```bash
# Clone the repo
git clone https://github.com/monksealseal/weatherflow.git
cd weatherflow

# Start all services (nginx + Streamlit + API)
make deploy

# Or use the deploy script directly
./scripts/deploy.sh docker
```

This starts three containers:
- **nginx** on port 80 (reverse proxy)
- **Streamlit** on internal port 8501
- **FastAPI** on internal port 8000

### Manage

```bash
make deploy-status   # Check service status
make deploy-logs     # View logs
make deploy-stop     # Stop everything
```

### Custom Port

```bash
PORT=3000 make deploy   # Run on port 3000 instead of 80
```

---

## Option 3: Fly.io

Deploy the Streamlit app to Fly.io's global edge network.

### Prerequisites

- Install the Fly CLI: `curl -L https://fly.io/install.sh | sh`
- Sign up: `flyctl auth signup`

### Deploy

```bash
# First time: create the app
flyctl launch --no-deploy --copy-config

# Deploy
flyctl deploy

# Or use Make
make deploy-fly
```

### Scale

```bash
flyctl scale count 2        # Run 2 instances
flyctl scale vm shared-cpu-2x  # Upgrade CPU
```

---

## Option 4: Vercel

Deploy the React frontend as a static site + FastAPI as a serverless function.

### Prerequisites

- Install the Vercel CLI: `npm install -g vercel`
- Sign up: `vercel login`

### Deploy

```bash
# First time: link to Vercel project
vercel

# Production deploy
vercel deploy --prod

# Or use Make
make deploy-vercel
```

Vercel auto-detects `vercel.json` and:
- Builds the React frontend from `frontend/` (served as static files)
- Deploys `api/index.py` as a serverless function (handles `/api/*` routes)

### How it works

- `/api/*` requests are routed to the FastAPI serverless function
- All other requests serve the React SPA
- Automatic HTTPS, CDN, and global edge caching

### Limitations

- Serverless functions have a 60s timeout (configurable up to 300s on Pro)
- 250MB unzipped size limit (CPU-only PyTorch fits)
- Streamlit is **not** deployed on Vercel (use Railway or Docker for Streamlit)

---

## Architecture

```
                    ┌─────────────┐
   Internet ──────>│    nginx    │ (port 80)
                    │  (reverse   │
                    │   proxy)    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │                         │
              v                         v
     ┌─────────────┐          ┌─────────────┐
     │  Streamlit  │          │  FastAPI    │
     │  (port 8501)│          │  (port 8000)│
     │  28 pages   │          │  20+ APIs   │
     └─────────────┘          └─────────────┘
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `80` | External port (Docker Compose) |
| `WEATHERFLOW_API_URL` | `http://api:8000` | API URL for Streamlit to call |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |
| `MPLBACKEND` | `Agg` | Matplotlib non-interactive backend |
