# WeatherFlow System Architecture

## Overview

WeatherFlow uses a **client-server architecture** where all users connect to a single centralized backend.

```
┌─────────────────────────────────────────────────────────────────┐
│                         USERS                                    │
│                                                                  │
│  User 1        User 2        User 3        User N               │
│    ↓             ↓             ↓             ↓                   │
└────┼─────────────┼─────────────┼─────────────┼───────────────────┘
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                   │
                   │ HTTPS
                   │
     ┌─────────────▼─────────────┐
     │   GitHub Pages Frontend   │
     │  (Static HTML/CSS/JS)     │
     │                           │
     │  monksealseal.github.io/  │
     │       weatherflow/        │
     └─────────────┬─────────────┘
                   │
                   │ REST API
                   │ (CORS enabled)
                   │
     ┌─────────────▼─────────────┐
     │  Centralized Backend      │
     │  (FastAPI + PyTorch)      │
     │                           │
     │  weatherflow-api-         │
     │  production.up.railway.app│
     │                           │
     │  • ML Training            │
     │  • Data Loading           │
     │  • Visualizations         │
     └───────────────────────────┘
```

## Components

### 1. Frontend (GitHub Pages)

**Location:** `https://monksealseal.github.io/weatherflow/`

**Technology:**
- React 18 (UI framework)
- TypeScript (type safety)
- Vite (build tool)
- Axios (HTTP client)

**Responsibilities:**
- User interface
- Experiment configuration
- Result visualization
- Experiment history tracking (localStorage)

**Deployment:**
- Automatic on push to `main` branch
- GitHub Actions workflow
- Zero cost (GitHub Pages is free)

### 2. Backend (Railway)

**Location:** `https://weatherflow-api-production.up.railway.app`

**Technology:**
- Python 3.11
- FastAPI (web framework)
- PyTorch (ML training)
- Uvicorn (ASGI server)

**Responsibilities:**
- Run PyTorch training
- Load and preprocess data
- Execute experiments
- Generate predictions
- Create visualizations

**Deployment:**
- Automatic on push to `main` branch
- Railway auto-deploy from GitHub
- Cost: ~$5-20/month depending on usage

## Data Flow

### Running an Experiment

```
1. User configures experiment in frontend
   ↓
2. Frontend sends POST /api/experiments
   ↓
3. Backend receives request
   ↓
4. Backend creates synthetic dataset
   ↓
5. Backend trains PyTorch model
   ↓
6. Backend generates predictions
   ↓
7. Backend returns results as JSON
   ↓
8. Frontend visualizes results
   ↓
9. Frontend saves to experiment history
```

### API Endpoints

```
GET  /api/health
     Returns: {"status": "ok"}
     Purpose: Health check

GET  /api/options
     Returns: Available variables, pressure levels, etc.
     Purpose: Populate UI dropdowns

POST /api/experiments
     Accepts: ExperimentConfig JSON
     Returns: ExperimentResult JSON
     Purpose: Run ML training experiment
```

## Deployment Architecture

### Development

```
┌──────────────────────┐       ┌──────────────────────┐
│  Frontend Dev Server │──────▶│  Local Backend       │
│  localhost:5173      │       │  localhost:8000      │
│                      │       │                      │
│  npm run dev         │       │  uvicorn ... --reload│
└──────────────────────┘       └──────────────────────┘
```

### Production

```
┌──────────────────────┐       ┌──────────────────────┐
│  GitHub Repository   │       │  GitHub Repository   │
│  (push to main)      │       │  (push to main)      │
└──────────┬───────────┘       └──────────┬───────────┘
           │                              │
           │ triggers                     │ triggers
           ▼                              ▼
┌──────────────────────┐       ┌──────────────────────┐
│  GitHub Actions      │       │  Railway Platform    │
│  Build frontend      │       │  Build backend       │
└──────────┬───────────┘       └──────────┬───────────┘
           │                              │
           │ deploy                       │ deploy
           ▼                              ▼
┌──────────────────────┐       ┌──────────────────────┐
│  GitHub Pages        │──────▶│  Railway Service     │
│  Static hosting      │ API   │  Docker container    │
└──────────────────────┘       └──────────────────────┘
```

## Security

### CORS Protection

The backend only accepts requests from:
- `https://monksealseal.github.io` (production)
- `http://localhost:5173` (development)
- `http://localhost:3000` (alternative dev)

### HTTPS Encryption

- All traffic encrypted with TLS 1.3
- Automatic certificate management
- No mixed content warnings

### Rate Limiting (Planned)

- Limit: 10 experiments per hour per IP
- Prevents abuse and DoS attacks
- Ensures fair resource sharing

## Monitoring

### Backend Health

```bash
# Check if backend is alive
curl https://weatherflow-api-production.up.railway.app/api/health

# Expected response
{"status":"ok"}
```

### Metrics (Railway Dashboard)

- Request count and rate
- Response time (p50, p95, p99)
- Error rate
- CPU usage
- Memory usage
- Deployment status

### Frontend Status

Check browser console for:
- API connection status
- Request/response errors
- Performance metrics

## Scaling

### Current Capacity

- **Concurrent users:** ~50
- **Requests per minute:** ~100
- **Experiments per hour:** ~500
- **Response time:** ~2-30 seconds
- **Uptime:** 99.9% (Railway SLA)

### Scaling Strategy

**Vertical (more resources):**
- Increase memory: 512MB → 1GB → 2GB
- Increase CPU allocation
- Cost: $10-20/month

**Horizontal (more instances):**
- Add replicas
- Automatic load balancing
- Cost: $20-50/month

## Cost Breakdown

### Frontend (GitHub Pages)
- **Cost:** $0/month
- **Storage:** Unlimited
- **Bandwidth:** 100 GB/month free
- **Build minutes:** 2,000/month free

### Backend (Railway)
- **Free tier:** $5 credit/month
- **Starter:** $5-10/month
- **Pro:** $20-50/month
- **Enterprise:** Custom

### Total Monthly Cost
- **Light usage:** $0-5
- **Medium usage:** $5-15
- **Heavy usage:** $20-50

## Maintenance

### Automated
- Frontend deployment (GitHub Actions)
- Backend deployment (Railway)
- HTTPS certificate renewal
- Health checks

### Manual (Monthly)
- Review metrics
- Update dependencies
- Check costs
- Test disaster recovery

### Emergency (As Needed)
- Restart backend
- Roll back deployment
- Scale resources
- Debug errors

## Troubleshooting

### Frontend Can't Connect to Backend

**Check:**
1. Backend health endpoint
2. CORS configuration
3. Frontend environment variables
4. Browser console for errors

**Fix:**
1. Restart backend on Railway
2. Verify CORS origins in code
3. Clear browser cache
4. Check Railway logs

### Backend Slow or Timing Out

**Check:**
1. Railway metrics (CPU/memory)
2. Experiment configuration size
3. Response time trends

**Fix:**
1. Increase resources
2. Reduce experiment complexity
3. Add caching
4. Scale horizontally

### High Costs

**Check:**
1. Railway usage dashboard
2. Request patterns
3. Resource allocation

**Fix:**
1. Add rate limiting
2. Optimize experiments
3. Right-size resources
4. Add usage quotas

## Future Enhancements

### Short-term
- [ ] Rate limiting
- [ ] Result caching
- [ ] Enhanced error messages
- [ ] User authentication

### Medium-term
- [ ] Background job queue
- [ ] Database for experiments
- [ ] User dashboards
- [ ] API versioning

### Long-term
- [ ] Multi-region deployment
- [ ] Auto-scaling
- [ ] Custom domains
- [ ] CDN integration

## Resources

### Documentation
- [Backend Quick Start](../BACKEND_QUICKSTART.md)
- [Deployment Guide](backend_deployment.md)
- [Architecture Details](BACKEND_ARCHITECTURE.md)

### Dashboards
- Frontend: https://monksealseal.github.io/weatherflow/
- Backend Health: https://weatherflow-api-production.up.railway.app/api/health
- Railway Dashboard: https://railway.app

### Support
- GitHub Issues: https://github.com/monksealseal/weatherflow/issues
- Railway Support: https://discord.gg/railway

---

**Last Updated:** 2026-01-04  
**Version:** 0.4.2  
**Status:** Production Ready ✅
