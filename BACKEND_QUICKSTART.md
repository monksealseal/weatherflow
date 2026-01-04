# WeatherFlow Backend Quick Start

This guide will help you deploy the centralized WeatherFlow backend that serves all users.

## What You're Deploying

A **single, centralized FastAPI backend** that:
- Runs PyTorch training for all users
- Handles data loading and preprocessing  
- Generates visualizations
- Provides REST APIs for the GitHub Pages frontend

**All users** will connect to this one backend instance.

## Prerequisites

- GitHub account with admin access to the weatherflow repository
- Railway account (free - sign up at https://railway.app)

## Step-by-Step Deployment

### 1. Sign Up for Railway (2 minutes)

1. Visit https://railway.app
2. Click "Login" → "Login with GitHub"
3. Authorize Railway to access your GitHub account
4. You'll get $5 free credit per month (enough for testing)

### 2. Deploy the Backend (5 minutes)

1. **Create New Project**
   - In Railway dashboard, click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `monksealseal/weatherflow` repository
   - Railway will automatically detect it's a Python app

2. **Configure the Service**
   - Railway will auto-configure most settings
   - Click on the service card
   - Go to "Settings" tab

3. **Set Environment Variables**
   - Click "Variables" in the sidebar
   - Add these variables:
     ```
     PORT=8000
     TORCH_NUM_THREADS=4
     PYTHON_VERSION=3.11
     ```

4. **Set Start Command**
   - Go to "Settings" → "Deploy"
   - Set custom start command:
     ```
     uvicorn weatherflow.server.app:app --host 0.0.0.0 --port $PORT --workers 2
     ```
   
   Or simply ensure `railway.json` is in the repository (it already is).

5. **Generate Public Domain**
   - Go to "Settings" → "Networking"
   - Click "Generate Domain"
   - Railway will create a public URL like:
     ```
     weatherflow-api-production.up.railway.app
     ```
   - **COPY THIS URL** - you'll need it for the frontend

### 3. Verify Backend is Working (1 minute)

Test the health endpoint:

```bash
curl https://weatherflow-api-production.up.railway.app/api/health
```

Expected response:
```json
{"status":"ok"}
```

Test the options endpoint:
```bash
curl https://weatherflow-api-production.up.railway.app/api/options
```

Should return available variables, pressure levels, etc.

### 4. Update Frontend Configuration (2 minutes)

Update the backend URL in your repository:

1. **Update Environment File**
   
   Edit `frontend/.env.production`:
   ```bash
   VITE_API_URL=https://your-actual-railway-url.up.railway.app
   ```

2. **Update GitHub Workflow**
   
   Edit `.github/workflows/deploy-pages.yml`:
   ```yaml
   - name: Build frontend
     working-directory: ./frontend
     env:
       GITHUB_PAGES: true
       VITE_API_URL: https://your-actual-railway-url.up.railway.app
     run: npm run build
   ```

3. **Commit and Push**
   ```bash
   git add frontend/.env.production .github/workflows/deploy-pages.yml
   git commit -m "Configure backend URL for production"
   git push origin main
   ```

### 5. Deploy Frontend (Automatic)

GitHub Actions will automatically:
1. Build the frontend with the backend URL
2. Deploy to GitHub Pages
3. Frontend will now connect to your Railway backend

Visit: https://monksealseal.github.io/weatherflow/

## Verify Everything Works

### Test from Frontend

1. Visit https://monksealseal.github.io/weatherflow/
2. Check the footer/status bar - should show "Backend: Connected ✓"
3. Try running a simple experiment:
   - Navigate to "Experiments" → "New Experiment"
   - Use default settings
   - Click "Run Experiment"
   - Should complete successfully in 30-60 seconds

### Monitor Backend

In Railway dashboard:
- View real-time logs
- Check resource usage (CPU, memory)
- Monitor request counts and response times

## Automatic Deployments

Railway is now configured for **automatic deployments**:

1. Any push to `main` branch triggers deployment
2. Railway pulls latest code from GitHub
3. Rebuilds and restarts the backend
4. Zero downtime with health checks

```bash
# Make changes to backend code
git add weatherflow/server/app.py
git commit -m "Update backend feature"
git push origin main

# Railway automatically deploys in ~2-3 minutes
```

## Cost Estimation

### Railway Pricing

**Free Tier:**
- $5/month free credit
- Enough for ~500 requests/day
- Perfect for testing and demos
- Backend sleeps after 10 minutes of inactivity

**Starter Plan ($5/month):**
- $5 credit + $5 usage
- ~5,000 requests/day
- Good for light production usage

**Pro Plan ($20/month):**
- Unlimited projects
- More resources
- Always-on backend
- For heavy usage

### Typical Usage Costs

| Users/Day | Experiments/Day | Monthly Cost |
|-----------|-----------------|--------------|
| 1-5       | 10-50          | Free ($0)    |
| 10-50     | 50-500         | $5-10        |
| 50-200    | 500-2000       | $10-20       |
| 200+      | 2000+          | $20-50       |

## Monitoring & Alerts

### Set Up Health Monitoring

1. **Railway Built-in Monitoring**
   - Railway Dashboard → Your Project → Metrics
   - Shows CPU, memory, requests, errors

2. **External Monitoring (Optional)**
   - Use UptimeRobot (free): https://uptimerobot.com
   - Monitor: `https://your-backend-url/api/health`
   - Get alerts via email/SMS if backend goes down

### View Logs

Railway Dashboard → Your Project → Logs

Or use Railway CLI:
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# View logs
railway logs
```

## Troubleshooting

### Backend Not Responding

**Check 1: Railway Service Status**
- Railway Dashboard → Check if service is "Active"
- If crashed, check logs for errors

**Check 2: Health Endpoint**
```bash
curl https://your-backend-url/api/health
```

**Check 3: CORS Errors**
- View browser console for CORS errors
- Verify `weatherflow/server/app.py` has correct origins:
  ```python
  allow_origins=[
      "https://monksealseal.github.io",
      ...
  ]
  ```

**Fix: Restart Service**
```bash
railway restart
```

### Slow Response Times

**Cause:** Free tier may sleep after inactivity

**Solution 1:** Upgrade to Starter plan ($5/month)

**Solution 2:** Keep-alive ping (add to frontend)
```typescript
// Ping backend every 5 minutes to keep it awake
setInterval(async () => {
  await checkHealth();
}, 5 * 60 * 1000);
```

### Out of Memory

**Symptoms:** 
- Experiments failing with 500 errors
- Backend crashing during large experiments

**Solution:**
- Railway Dashboard → Settings → Increase memory limit
- Or optimize experiments (smaller batch sizes)

## Scaling for Production

### When to Scale

Scale up if you experience:
- Response times > 3 seconds consistently
- Memory errors during experiments
- More than 100 users per day
- Backend downtime due to resource limits

### How to Scale

**Vertical Scaling** (More resources for same instance):
1. Railway Dashboard → Settings → Resources
2. Increase Memory: 512MB → 1GB → 2GB
3. Increase CPU allocation
4. Cost: $10-20/month

**Horizontal Scaling** (Multiple instances):
1. Upgrade to Railway Pro plan
2. Add replica services
3. Railway handles load balancing
4. Cost: $20-50/month

## Security Checklist

- [x] HTTPS enabled (Railway default)
- [x] CORS configured for GitHub Pages domain only
- [ ] Add rate limiting (see backend_deployment.md)
- [ ] Add authentication for production (optional)
- [ ] Set up monitoring and alerts
- [ ] Regular security updates

## Maintenance

### Weekly Tasks
- Check Railway dashboard for errors
- Review resource usage trends
- Test key endpoints

### Monthly Tasks
- Review costs and optimize
- Update Python dependencies
- Check for security updates
- Test disaster recovery

### Update Dependencies

```bash
# Update requirements
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt

# Commit and push
git add requirements.txt
git commit -m "Update dependencies"
git push origin main

# Railway auto-deploys
```

## Getting Help

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **WeatherFlow Issues**: https://github.com/monksealseal/weatherflow/issues

## Success Checklist

- [ ] Railway account created
- [ ] Backend deployed successfully
- [ ] Public URL generated
- [ ] Health endpoint returns `{"status":"ok"}`
- [ ] Options endpoint returns data
- [ ] Frontend updated with backend URL
- [ ] GitHub Pages deployed
- [ ] Test experiment runs successfully
- [ ] Monitoring set up

## Summary

You now have a **centralized backend** that:

✅ Serves **all users** from a single deployment  
✅ **Auto-deploys** when you push to main  
✅ **Scales automatically** based on traffic  
✅ **Costs $0-10/month** for typical usage  
✅ **Zero user configuration** - they just visit the frontend

Users simply visit https://monksealseal.github.io/weatherflow/ and everything works!

## Next Steps

1. Deploy backend on Railway (10 minutes)
2. Update frontend environment files
3. Push to GitHub to deploy frontend
4. Test end-to-end functionality
5. Set up monitoring and alerts
6. Share with users!

**Questions?** Open an issue at https://github.com/monksealseal/weatherflow/issues
