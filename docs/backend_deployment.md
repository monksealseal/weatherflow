# WeatherFlow Centralized Backend

This document describes the centralized backend infrastructure for WeatherFlow that serves ALL users through the GitHub Pages frontend.

## Architecture Overview

```
┌─────────────────────────────────────┐
│   GitHub Pages Frontend             │
│   monksealseal.github.io/weatherflow│
│   (Static HTML/JS/CSS)              │
└─────────────┬───────────────────────┘
              │
              │ HTTPS API Calls
              │
              ▼
┌─────────────────────────────────────┐
│   Centralized Backend               │
│   weatherflow-api.railway.app       │
│   (FastAPI + PyTorch)               │
└─────────────────────────────────────┘
```

## Backend Responsibilities

The centralized backend provides:

1. **Training Execution**: Runs PyTorch training for flow matching models
2. **Data Loading**: Loads and preprocesses ERA5 reanalysis data
3. **Visualization Generation**: Creates plots and animations
4. **Experiment Tracking**: Manages experiment state and results
5. **Model Inference**: Generates predictions from trained models

## Deployment Platform: Railway

Railway is chosen as the single deployment platform because:

- ✅ **Easy automatic deployment** from GitHub
- ✅ **Automatic HTTPS** with custom domains
- ✅ **Auto-scaling** based on load
- ✅ **Built-in monitoring** and logs
- ✅ **GitHub integration** for CI/CD
- ✅ **Reasonable cost** for shared usage

## Production Deployment

### Initial Setup (One-Time)

1. **Create Railway Account**
   - Visit https://railway.app
   - Sign in with GitHub
   - Link to monksealseal/weatherflow repository

2. **Create New Project**
   - New Project → Deploy from GitHub
   - Select `monksealseal/weatherflow`
   - Railway auto-detects Python

3. **Configure Service**
   - Set start command: `uvicorn weatherflow.server.app:app --host 0.0.0.0 --port $PORT`
   - Set environment variables:
     - `PORT=8000`
     - `TORCH_NUM_THREADS=4`
     - `PYTHON_VERSION=3.11`

4. **Enable Public URL**
   - Settings → Networking → Generate Domain
   - Note the URL (e.g., `weatherflow-api-production.up.railway.app`)

5. **Set Custom Domain (Optional)**
   - Add custom domain: `api.weatherflow.dev`
   - Configure DNS CNAME record

### Automatic Deployment

Railway automatically deploys when code is pushed to `main`:

```bash
git push origin main
# Railway detects push and deploys automatically
```

### Configuration Files

The repository includes `railway.json` for Railway configuration:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn weatherflow.server.app:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/api/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

## Backend Configuration

### CORS Setup

The backend must allow requests from the GitHub Pages domain.

In `weatherflow/server/app.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app = create_app()

# Configure CORS for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://monksealseal.github.io",
        "http://localhost:5173",  # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Environment Variables

Set these in Railway dashboard:

```bash
PORT=8000                    # Railway assigns this automatically
TORCH_NUM_THREADS=4          # Limit CPU usage
PYTHON_VERSION=3.11          # Python version
LOG_LEVEL=info               # Logging level
MAX_WORKERS=2                # Uvicorn workers
```

## Frontend Integration

### Automatic Backend URL

The frontend automatically connects to the production backend.

In `frontend/.env.production`:

```bash
VITE_API_URL=https://weatherflow-api-production.up.railway.app
```

### Build-Time Configuration

In `.github/workflows/deploy-pages.yml`:

```yaml
- name: Build frontend
  working-directory: ./frontend
  env:
    GITHUB_PAGES: true
    VITE_API_URL: https://weatherflow-api-production.up.railway.app
  run: npm run build
```

### Frontend Client

The API client in `frontend/src/api/client.ts` uses the configured URL:

```typescript
const baseURL = import.meta.env.VITE_API_URL || 'https://weatherflow-api-production.up.railway.app';

const client = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json'
  }
});
```

## Monitoring & Maintenance

### Health Checks

Railway automatically monitors the health endpoint:

```bash
curl https://weatherflow-api-production.up.railway.app/api/health
# Expected: {"status":"ok"}
```

### Viewing Logs

Railway Dashboard → Project → Logs tab

Or use Railway CLI:

```bash
railway logs
```

### Metrics

Railway Dashboard shows:
- Request count
- Response times
- Memory usage
- CPU usage
- Error rates

### Alerts

Set up alerts in Railway dashboard:
- High error rate (> 5%)
- High response time (> 2s)
- Memory usage (> 80%)
- Service downtime

## Scaling

### Current Configuration

- **Memory**: 512 MB (starter)
- **CPU**: Shared
- **Concurrent Requests**: ~50
- **Monthly Cost**: ~$5-10

### Scaling Up

When traffic increases:

1. **Vertical Scaling** (Railway Dashboard)
   - Increase memory to 1-2 GB
   - More CPU allocation
   - Cost: $10-20/month

2. **Horizontal Scaling**
   - Add more replicas (Railway Pro)
   - Load balancing automatic
   - Cost: $20-50/month

### Performance Optimization

1. **Caching**
   - Cache preprocessed datasets
   - Store intermediate results
   - Use Redis for session data

2. **Request Limiting**
   - Rate limit: 100 requests/minute per IP
   - Prevents abuse and DoS

3. **Async Processing**
   - Long experiments run in background
   - Return job ID immediately
   - Poll for results

## Security

### HTTPS

- Railway provides automatic HTTPS
- All traffic encrypted with TLS 1.3
- Certificates auto-renewed

### Rate Limiting

Implemented in backend to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/experiments")
@limiter.limit("10/minute")  # Max 10 experiments per minute
async def run_experiment(config: ExperimentConfig):
    ...
```

### Authentication (Future)

For production with many users, consider:
- GitHub OAuth for user identification
- API keys for programmatic access
- Usage quotas per user

## Cost Management

### Current Costs

**Railway Pricing:**
- Development: $5/month (with $5 free credit)
- Light Production: $10-15/month
- Medium Production: $20-30/month
- Heavy Production: $50+/month

### Cost Optimization

1. **Resource Limits**
   - Limit experiment size (batch size, epochs)
   - Maximum timeout: 300 seconds
   - Memory limit: 1 GB per request

2. **Efficient Resource Usage**
   - Use smaller model configurations by default
   - Lazy load large datasets
   - Clear memory after experiments

3. **Monitoring**
   - Track usage per endpoint
   - Identify expensive operations
   - Optimize hot paths

## Disaster Recovery

### Backup Strategy

1. **Code Backup**
   - GitHub repository (source of truth)
   - Railway pulls from GitHub

2. **Configuration Backup**
   - Environment variables documented here
   - Export Railway config regularly

3. **Data Backup**
   - User experiments stored in database (if implemented)
   - Periodic backups to S3/GCS

### Recovery Process

If backend goes down:

1. **Check Railway Status**
   - Visit Railway dashboard
   - Check service logs
   - Look for errors

2. **Restart Service**
   ```bash
   railway restart
   ```

3. **Redeploy from GitHub**
   ```bash
   git push origin main --force
   ```

4. **Create New Service** (worst case)
   - Deploy new Railway service
   - Update frontend environment variable
   - Redeploy frontend

## Testing

### Integration Tests

Test the deployed backend:

```bash
# Health check
curl https://weatherflow-api-production.up.railway.app/api/health

# Get options
curl https://weatherflow-api-production.up.railway.app/api/options

# Run minimal experiment
curl -X POST https://weatherflow-api-production.up.railway.app/api/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": {"variables": ["t"], "pressureLevels": [500], "trainSamples": 8, "valSamples": 4},
    "training": {"epochs": 1, "batchSize": 2}
  }'
```

### Load Testing

Before major releases, test with realistic load:

```bash
# Using Apache Bench
ab -n 100 -c 10 https://weatherflow-api-production.up.railway.app/api/health

# Using k6
k6 run load_test.js
```

## Troubleshooting

### Backend Not Responding

1. Check Railway dashboard for service status
2. View logs for errors
3. Restart service if needed
4. Check CORS configuration

### Slow Response Times

1. Check Railway metrics for resource usage
2. Increase memory/CPU if needed
3. Optimize expensive operations
4. Add caching

### CORS Errors in Frontend

1. Verify CORS origins in backend code
2. Check that frontend URL matches allowed origins
3. Redeploy backend after CORS changes

### Out of Memory

1. Reduce batch sizes in experiments
2. Increase Railway memory allocation
3. Add memory monitoring and alerts

## Maintenance Schedule

### Daily
- Monitor error rates in Railway dashboard
- Check response times

### Weekly
- Review logs for issues
- Check resource usage trends
- Test key endpoints

### Monthly
- Review costs and optimize
- Update dependencies
- Deploy security patches
- Test disaster recovery

## Support & Contact

For backend issues:
- **GitHub Issues**: https://github.com/monksealseal/weatherflow/issues
- **Railway Status**: https://status.railway.app
- **Documentation**: This file

## Summary

The centralized backend approach provides:

✅ **Single deployment** - One backend serves all users
✅ **Automatic updates** - Push to main, auto-deploys
✅ **Zero user configuration** - Frontend connects automatically
✅ **Reliable infrastructure** - Railway handles scaling and uptime
✅ **Cost-effective** - Shared resources, predictable costs
✅ **Easy maintenance** - Single point of control

Users simply visit https://monksealseal.github.io/weatherflow/ and everything works!
