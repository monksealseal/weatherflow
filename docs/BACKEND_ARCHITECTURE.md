# Backend Architecture for Maintainers

This document explains the centralized backend architecture for WeatherFlow and how to maintain it.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   GitHub Pages (Frontend)                    │
│            https://monksealseal.github.io/weatherflow/       │
│                                                              │
│  • Static HTML/CSS/JavaScript                               │
│  • React + TypeScript application                           │
│  • No server-side code                                      │
│  • Deployed automatically on push to main                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTPS REST API
                            │ (CORS enabled)
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              Centralized Backend (Railway)                   │
│     https://weatherflow-api-production.up.railway.app        │
│                                                              │
│  • FastAPI application                                      │
│  • PyTorch for ML training                                  │
│  • Handles experiments for ALL users                        │
│  • Auto-deploys from GitHub                                 │
│  • Scales automatically                                     │
└──────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Single Centralized Backend

**Why:** 
- Simplifies deployment - maintainers deploy once
- No user configuration required
- Consistent experience for all users
- Easier to monitor and maintain
- Cost-effective through resource sharing

**Trade-offs:**
- Single point of failure (mitigated with Railway's uptime)
- Shared resources (mitigated with rate limiting)
- Cost scales with total usage (not per user)

### 2. Railway as Deployment Platform

**Why Railway:**
- Automatic GitHub integration
- Zero-config deployment
- Automatic HTTPS
- Built-in monitoring
- Reasonable pricing
- Easy scaling

**Alternatives Considered:**
- Heroku: More expensive, similar features
- Render: Good free tier, but slower cold starts
- AWS/GCP: Too complex for this use case
- Docker: Requires own infrastructure

### 3. FastAPI for Backend

**Why FastAPI:**
- High performance
- Automatic API documentation
- Type safety with Pydantic
- Async support for long-running tasks
- Great for ML/scientific computing

**Already in codebase:**
- `weatherflow/server/app.py` exists
- Provides `/api/experiments` endpoint
- Well-structured and tested

## Code Structure

### Backend (`weatherflow/server/app.py`)

```python
# Key components:

1. FastAPI application creation
   - create_app() function
   - CORS middleware for GitHub Pages
   - Health check endpoint

2. Configuration models (Pydantic)
   - ExperimentConfig
   - DatasetConfig
   - ModelConfig
   - TrainingConfig

3. API endpoints
   - GET /api/health - Health check
   - GET /api/options - Available options
   - POST /api/experiments - Run training

4. Training execution
   - _build_dataloaders() - Create synthetic data
   - _train_model() - PyTorch training loop
   - _run_prediction() - Generate predictions
```

### Frontend (`frontend/src/api/client.ts`)

```typescript
// Key components:

1. Backend URL configuration
   - Uses VITE_API_URL environment variable
   - Falls back to Railway URL

2. Axios client setup
   - 5-minute timeout for experiments
   - Error interceptors
   - Automatic request/response handling

3. API functions
   - fetchOptions() - Get server options
   - runExperiment() - Submit experiment
   - checkHealth() - Check backend status
```

### Configuration Files

1. **`railway.json`** - Railway deployment config
   - Start command
   - Health check path
   - Restart policy

2. **`Procfile`** - Process definition
   - Web process command
   - Worker count

3. **`frontend/.env.production`** - Production environment
   - Backend URL for GitHub Pages build

4. **`frontend/.env.development`** - Development environment
   - Local backend URL for development

## Deployment Workflow

### Frontend Deployment (Automatic)

```yaml
# .github/workflows/deploy-pages.yml

on push to main:
  1. Checkout code
  2. Install Node.js dependencies
  3. Build frontend with VITE_API_URL
  4. Upload to GitHub Pages
  5. Deploy to https://monksealseal.github.io/weatherflow/
```

### Backend Deployment (Automatic via Railway)

```
on push to main:
  1. Railway detects GitHub push
  2. Pulls latest code
  3. Installs Python dependencies (requirements.txt)
  4. Runs: uvicorn weatherflow.server.app:app --host 0.0.0.0 --port $PORT
  5. Performs health check at /api/health
  6. Routes traffic to new deployment
  7. Terminates old deployment
```

## Maintenance Tasks

### Daily

**Monitor Backend Status**
- Check Railway dashboard: https://railway.app
- Verify no errors in logs
- Check response times are < 3 seconds

**Quick Health Check**
```bash
curl https://weatherflow-api-production.up.railway.app/api/health
# Expected: {"status":"ok"}
```

### Weekly

**Review Metrics**
- Railway Dashboard → Metrics tab
- CPU usage (should be < 80%)
- Memory usage (should be < 80%)
- Request count and patterns
- Error rate (should be < 1%)

**Check Logs**
```bash
# Install Railway CLI
npm i -g @railway/cli
railway login
railway logs --tail 100
```

**Test Key Endpoints**
```bash
# Health
curl https://weatherflow-api-production.up.railway.app/api/health

# Options
curl https://weatherflow-api-production.up.railway.app/api/options

# Simple experiment (should complete in ~30 seconds)
curl -X POST https://weatherflow-api-production.up.railway.app/api/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": {"variables": ["t"], "pressureLevels": [500], "trainSamples": 8, "valSamples": 4},
    "training": {"epochs": 1, "batchSize": 2}
  }'
```

### Monthly

**Update Dependencies**
```bash
# Update Python packages
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt

# Test locally (if possible)
uvicorn weatherflow.server.app:app --reload

# Commit and push (triggers auto-deploy)
git add requirements.txt
git commit -m "Update dependencies"
git push origin main
```

**Review Costs**
- Railway Dashboard → Usage tab
- Check current month's usage
- Estimate next month's cost
- Optimize if costs increasing

**Test Disaster Recovery**
- Export Railway environment variables
- Document backup procedure
- Test health endpoint recovery

### Quarterly

**Security Audit**
- Update Python to latest patch version
- Review CORS origins
- Check for CVEs in dependencies
- Consider adding authentication

**Performance Optimization**
- Review slow endpoints
- Add caching where appropriate
- Optimize database queries (if added)
- Consider scaling up if needed

## Monitoring & Alerts

### Built-in Railway Monitoring

Railway provides:
- Request count and rate
- Response time (p50, p95, p99)
- Error rate
- CPU usage
- Memory usage
- Deployment history

### External Monitoring (Recommended)

**UptimeRobot (Free)**
1. Sign up at https://uptimerobot.com
2. Add monitor:
   - Type: HTTP(S)
   - URL: `https://weatherflow-api-production.up.railway.app/api/health`
   - Interval: 5 minutes
3. Add alert contacts (email, SMS, Slack)

**Sentry (Optional)**
For error tracking:
```python
# In weatherflow/server/app.py
import sentry_sdk
sentry_sdk.init(dsn="your-dsn", traces_sample_rate=0.1)
```

### Alert Thresholds

Set up alerts for:
- Downtime > 2 minutes
- Response time > 5 seconds
- Error rate > 5%
- Memory usage > 90%
- CPU usage > 90%

## Scaling Guide

### When to Scale

**Signs you need to scale:**
- Response times consistently > 3 seconds
- Memory usage > 80% regularly
- CPU usage > 80% regularly
- User reports of timeouts
- More than 100 concurrent requests

### Vertical Scaling

**Increase resources for single instance:**

1. Railway Dashboard → Settings → Resources
2. Increase memory: 512MB → 1GB → 2GB → 4GB
3. Increase CPU allocation
4. Cost: $10-50/month depending on resources

**When to use:** When single experiments are hitting resource limits

### Horizontal Scaling

**Add more instances:**

1. Upgrade to Railway Pro plan
2. Add replica services
3. Railway handles load balancing automatically
4. Cost: $20-100/month depending on replicas

**When to use:** When many concurrent users

### Cost Optimization

**Before scaling up:**
1. Add rate limiting (prevent abuse)
2. Optimize experiment configurations (smaller batches, fewer epochs)
3. Add result caching
4. Use background jobs for long experiments

## Security

### Current Security Measures

1. **HTTPS Only**
   - Railway provides automatic HTTPS
   - All traffic encrypted with TLS 1.3

2. **CORS Protection**
   - Only allows requests from GitHub Pages domain
   - Prevents unauthorized access from other websites

3. **Request Timeout**
   - 5-minute timeout prevents resource exhaustion
   - Experiments must complete within timeout

### Recommended Enhancements

1. **Rate Limiting**
```python
# Add to weatherflow/server/app.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/experiments")
@limiter.limit("10/hour")  # Max 10 experiments per hour per IP
async def run_experiment(config: ExperimentConfig):
    ...
```

2. **Authentication (for production)**
```python
# GitHub OAuth integration
from fastapi_oauth2 import OAuth2

oauth2 = OAuth2(
    client_id="github_client_id",
    client_secret="github_client_secret",
)

@app.post("/api/experiments")
async def run_experiment(
    config: ExperimentConfig,
    user: User = Depends(oauth2.get_current_user)
):
    ...
```

3. **Input Validation**
   - Already implemented with Pydantic models
   - Validates all input parameters
   - Rejects invalid requests

4. **Logging**
```python
# Add structured logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/api/experiments")
async def run_experiment(config: ExperimentConfig):
    logger.info(f"Experiment started: {config.model_dump()}")
    ...
```

## Troubleshooting

### Backend Not Responding

**Symptom:** Health check fails, frontend can't connect

**Diagnosis:**
1. Check Railway dashboard - is service running?
2. Check logs for errors: `railway logs`
3. Check recent deployments - did something break?

**Fix:**
```bash
# Restart service
railway restart

# Or redeploy from GitHub
git commit --allow-empty -m "Trigger redeploy"
git push origin main
```

### CORS Errors

**Symptom:** Browser console shows CORS errors

**Diagnosis:**
Check `weatherflow/server/app.py` CORS configuration

**Fix:**
```python
# Ensure GitHub Pages domain is allowed
allow_origins=[
    "https://monksealseal.github.io",
    ...
]
```

Then redeploy:
```bash
git add weatherflow/server/app.py
git commit -m "Fix CORS configuration"
git push origin main
```

### Slow Response Times

**Symptom:** Experiments take > 2 minutes

**Diagnosis:**
1. Check Railway metrics - is CPU/memory maxed?
2. Check experiment configurations - are they too large?
3. Check if backend is sleeping (free tier)

**Fix:**
- Upgrade Railway plan (always-on)
- Increase resources (more CPU/memory)
- Optimize experiment defaults

### Out of Memory

**Symptom:** 500 errors, "Out of memory" in logs

**Diagnosis:**
Check Railway metrics - memory usage at 100%?

**Fix:**
1. Increase memory allocation in Railway
2. Reduce batch sizes in experiments
3. Add memory monitoring

### High Costs

**Symptom:** Railway bill higher than expected

**Diagnosis:**
Check Railway usage tab - which resources are expensive?

**Fix:**
1. Add rate limiting to prevent abuse
2. Reduce resources if over-provisioned
3. Add usage quotas per user
4. Cache expensive computations

## Development Workflow

### Local Development

**Setup:**
```bash
# Backend
cd weatherflow
pip install -r requirements.txt
pip install -e .
uvicorn weatherflow.server.app:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev  # Runs on port 5173
```

Frontend will connect to local backend automatically (via `.env.development`).

### Testing Backend Changes

```bash
# Test health endpoint
curl http://localhost:8000/api/health

# Test options endpoint
curl http://localhost:8000/api/options

# Test experiment (minimal)
curl -X POST http://localhost:8000/api/experiments \
  -H "Content-Type: application/json" \
  -d @test_experiment.json
```

### Deployment Process

1. **Make changes locally**
2. **Test changes locally**
3. **Commit and push to GitHub**
4. **Railway automatically deploys**
5. **Verify deployment** via health check
6. **Test from frontend**

## Best Practices

### Code Changes

- ✅ Test locally before pushing
- ✅ Use small, incremental commits
- ✅ Write descriptive commit messages
- ✅ Update documentation when changing APIs
- ✅ Maintain backward compatibility when possible

### Deployment

- ✅ Deploy during low-traffic periods
- ✅ Monitor logs after deployment
- ✅ Keep previous deployment for quick rollback
- ✅ Test health endpoint immediately after deploy
- ✅ Announce downtime if needed

### Monitoring

- ✅ Check dashboard daily
- ✅ Review metrics weekly
- ✅ Set up alerts for critical issues
- ✅ Keep deployment history
- ✅ Document incidents and resolutions

## Emergency Contacts

**Railway Support:**
- Status: https://status.railway.app
- Discord: https://discord.gg/railway
- Email: team@railway.app

**Repository Issues:**
- GitHub: https://github.com/monksealseal/weatherflow/issues

## Change Log

Keep a log of major changes:

```
2026-01-04: Initial deployment to Railway
           - Backend: weatherflow-api-production.up.railway.app
           - Frontend: monksealseal.github.io/weatherflow
           - CORS configured for GitHub Pages
           - Auto-deploy from main branch enabled
```

## Future Enhancements

### Short-term (1-3 months)
- [ ] Add rate limiting
- [ ] Implement result caching
- [ ] Add user authentication
- [ ] Enhanced error messages
- [ ] Logging improvements

### Medium-term (3-6 months)
- [ ] Background job queue for long experiments
- [ ] Database for experiment storage
- [ ] User dashboards and history
- [ ] API versioning
- [ ] Load testing and optimization

### Long-term (6-12 months)
- [ ] Multi-region deployment
- [ ] Horizontal auto-scaling
- [ ] Advanced monitoring and analytics
- [ ] Custom domain setup
- [ ] CDN for static assets

## Summary

This architecture provides:
- ✅ Single backend for all users
- ✅ Automatic deployment from GitHub
- ✅ Zero user configuration
- ✅ Scalable infrastructure
- ✅ Easy maintenance
- ✅ Predictable costs

Maintainers need to:
- Monitor Railway dashboard daily
- Review logs and metrics weekly
- Update dependencies monthly
- Plan for scaling as usage grows

The system is designed to "just work" for users while being easy to maintain for the repository owner.
