# Backend Deployment Checklist

Use this checklist to deploy the WeatherFlow backend for the first time.

## Pre-Deployment

- [ ] **Review Architecture**
  - Read [SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md)
  - Understand the centralized backend approach
  - Know the estimated costs

- [ ] **Prepare Accounts**
  - GitHub account with admin access to repository
  - Railway account (sign up at https://railway.app)
  - Credit card for Railway (required even for free tier)

- [ ] **Review Code**
  - Backend code in `weatherflow/server/app.py`
  - CORS configuration includes GitHub Pages domain
  - Health check endpoint exists

## Deployment Steps

### Step 1: Deploy Backend to Railway

- [ ] **Install Railway CLI** (optional, can use web UI instead)
  ```bash
  npm install -g @railway/cli
  ```

- [ ] **Option A: Automated Script**
  ```bash
  ./deploy_backend.sh
  ```

- [ ] **Option B: Manual via Railway Dashboard**
  1. Login to Railway at https://railway.app
  2. Click "New Project"
  3. Select "Deploy from GitHub repo"
  4. Choose `monksealseal/weatherflow`
  5. Railway auto-configures Python app
  6. Wait for initial deployment (~3-5 minutes)

- [ ] **Set Environment Variables**
  - `PORT=8000` (Railway sets this automatically)
  - `TORCH_NUM_THREADS=4`
  - `PYTHON_VERSION=3.11`

- [ ] **Generate Public Domain**
  - Railway Settings → Networking → Generate Domain
  - Copy the generated URL (e.g., `weatherflow-api-production.up.railway.app`)

### Step 2: Verify Backend

- [ ] **Test Health Endpoint**
  ```bash
  curl https://YOUR-RAILWAY-URL/api/health
  ```
  Expected: `{"status":"ok"}`

- [ ] **Test Options Endpoint**
  ```bash
  curl https://YOUR-RAILWAY-URL/api/options
  ```
  Should return JSON with variables, pressure levels, etc.

- [ ] **Test Simple Experiment** (optional, may take 30-60 seconds)
  ```bash
  curl -X POST https://YOUR-RAILWAY-URL/api/experiments \
    -H "Content-Type: application/json" \
    -d '{
      "dataset": {
        "variables": ["t"],
        "pressureLevels": [500],
        "trainSamples": 8,
        "valSamples": 4
      },
      "training": {
        "epochs": 1,
        "batchSize": 2
      }
    }'
  ```

- [ ] **Check Railway Logs**
  - Railway Dashboard → Logs tab
  - Should see startup messages
  - No error messages

### Step 3: Configure Frontend

- [ ] **Update Production Environment File**
  
  Edit `frontend/.env.production`:
  ```bash
  VITE_API_URL=https://YOUR-RAILWAY-URL
  ```

- [ ] **Update GitHub Pages Workflow**
  
  Edit `.github/workflows/deploy-pages.yml`:
  ```yaml
  - name: Build frontend
    working-directory: ./frontend
    env:
      GITHUB_PAGES: true
      VITE_API_URL: https://YOUR-RAILWAY-URL
    run: npm run build
  ```

- [ ] **Commit Changes**
  ```bash
  git add frontend/.env.production .github/workflows/deploy-pages.yml
  git commit -m "Configure production backend URL"
  git push origin main
  ```

### Step 4: Deploy Frontend

- [ ] **Wait for GitHub Actions**
  - Go to GitHub repository → Actions tab
  - Watch "Deploy to GitHub Pages" workflow
  - Should complete in ~2-3 minutes

- [ ] **Verify Frontend Deployment**
  - Visit https://monksealseal.github.io/weatherflow/
  - Page should load without errors
  - Check browser console for errors

### Step 5: End-to-End Testing

- [ ] **Test Frontend Connection to Backend**
  - Open browser developer tools (F12)
  - Visit https://monksealseal.github.io/weatherflow/
  - Check Network tab for API calls
  - Should see successful calls to Railway backend

- [ ] **Run Test Experiment from Frontend**
  1. Navigate to "Experiments" → "New Experiment"
  2. Keep default settings
  3. Click "Run Experiment"
  4. Wait 30-60 seconds
  5. Should see results with metrics and predictions
  6. Check "Experiment History" - should show completed experiment

- [ ] **Test on Different Browsers**
  - Chrome/Edge
  - Firefox
  - Safari (if available)

- [ ] **Test on Mobile** (optional)
  - iOS Safari
  - Android Chrome

## Post-Deployment

### Step 6: Set Up Monitoring

- [ ] **Railway Built-in Monitoring**
  - Railway Dashboard → Metrics tab
  - Verify metrics are being collected
  - Note baseline values for CPU, memory, requests

- [ ] **External Monitoring** (recommended)
  - Sign up for UptimeRobot (free): https://uptimerobot.com
  - Add monitor for `https://YOUR-RAILWAY-URL/api/health`
  - Set interval to 5 minutes
  - Add email notification

- [ ] **Set Up Alerts**
  - Email alerts for downtime
  - Slack/Discord webhook (optional)

### Step 7: Documentation

- [ ] **Update Repository README**
  - Add link to live demo
  - Add backend URL to documentation
  - Update deployment status

- [ ] **Document Deployment**
  - Save Railway URL
  - Document environment variables
  - Note deployment date
  - Create change log entry

- [ ] **Share Access** (if needed)
  - Invite team members to Railway project
  - Share monitoring dashboard access

### Step 8: Security

- [ ] **Review CORS Configuration**
  - Verify only GitHub Pages domain is allowed
  - No wildcards in production

- [ ] **Check HTTPS**
  - All traffic uses HTTPS
  - No mixed content warnings
  - Valid SSL certificate

- [ ] **Plan for Rate Limiting** (future)
  - Document rate limit strategy
  - Identify abuse prevention measures

## Ongoing Maintenance

### Daily

- [ ] Check Railway dashboard for errors
- [ ] Verify health endpoint responds

### Weekly

- [ ] Review Railway metrics
  - Response times
  - Error rates
  - Resource usage
- [ ] Check costs and usage
- [ ] Review logs for issues

### Monthly

- [ ] Update Python dependencies
- [ ] Review security advisories
- [ ] Test backup/restore procedures
- [ ] Review costs and optimize
- [ ] Document any incidents

## Troubleshooting

### Backend Not Deploying

**Symptoms:**
- Railway shows "Build failed"
- Deployment stuck

**Checks:**
- [ ] Check Railway logs for errors
- [ ] Verify `railway.json` exists
- [ ] Verify `Procfile` is correct
- [ ] Check `requirements.txt` has all dependencies

**Solutions:**
- [ ] Redeploy: `railway up --detach`
- [ ] Check Railway Discord for service issues
- [ ] Contact Railway support

### Frontend Can't Connect

**Symptoms:**
- CORS errors in browser console
- "Network error" messages
- Timeouts

**Checks:**
- [ ] Backend health endpoint responds
- [ ] CORS configuration in `weatherflow/server/app.py`
- [ ] Frontend environment variables correct
- [ ] Railway service is running

**Solutions:**
- [ ] Verify CORS origins match
- [ ] Clear browser cache
- [ ] Redeploy backend
- [ ] Check Railway logs

### High Costs

**Symptoms:**
- Railway bill higher than expected
- Usage alerts

**Checks:**
- [ ] Railway usage dashboard
- [ ] Request rate and patterns
- [ ] Resource allocation vs usage

**Solutions:**
- [ ] Add rate limiting
- [ ] Optimize experiment defaults
- [ ] Right-size resources
- [ ] Review and remove unused services

## Rollback Procedure

If deployment fails or causes issues:

### Backend Rollback

- [ ] Railway Dashboard → Deployments
- [ ] Find last working deployment
- [ ] Click "Redeploy"
- [ ] Wait for redeployment
- [ ] Verify health endpoint

### Frontend Rollback

- [ ] GitHub repository → Actions
- [ ] Find last successful deployment
- [ ] Re-run workflow
- [ ] Or revert commit and push

## Success Criteria

Deployment is successful when:

- [x] Backend deploys without errors
- [x] Health endpoint returns `{"status":"ok"}`
- [x] Frontend loads without errors
- [x] Test experiment completes successfully
- [x] No CORS errors in browser console
- [x] Monitoring is active
- [x] Documentation is updated

## Resources

- **Quick Start:** [BACKEND_QUICKSTART.md](../BACKEND_QUICKSTART.md)
- **Deployment Guide:** [backend_deployment.md](backend_deployment.md)
- **Architecture:** [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
- **Maintenance:** [BACKEND_ARCHITECTURE.md](BACKEND_ARCHITECTURE.md)

## Support

- **Railway:** https://discord.gg/railway
- **GitHub Issues:** https://github.com/monksealseal/weatherflow/issues
- **Status Page:** https://status.railway.app

---

## Checklist Summary

```
Pre-Deployment:     [ ] Accounts [ ] Review Code
Deployment:         [ ] Backend [ ] Verify [ ] Frontend
Testing:            [ ] E2E [ ] Browsers
Post-Deployment:    [ ] Monitoring [ ] Docs [ ] Security
Maintenance:        [ ] Daily [ ] Weekly [ ] Monthly
```

**Status:** ☐ Not Started | ⏳ In Progress | ✅ Complete

**Deployed By:** _____________  
**Deployment Date:** _____________  
**Backend URL:** _____________  
**Frontend URL:** https://monksealseal.github.io/weatherflow/

---

**Notes:**

_Use this space for deployment notes, issues encountered, or special configurations_
