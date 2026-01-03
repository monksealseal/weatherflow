# GCM Deployment Guide

Complete guide to deploying the General Circulation Model web application.

## Prerequisites

- Git installed
- Heroku CLI installed (`npm install -g heroku` or download from heroku.com)
- Heroku account (free tier works)

## Quick Deploy (One-Click)

The fastest way to deploy:

1. Click this button:
   [![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

2. Fill in your app name
3. Click "Deploy app"
4. Wait 2-3 minutes
5. Click "View app"

Done! Your GCM is now running in the cloud.

## Manual Deployment

### Step 1: Prepare Your Repository

```bash
# Clone the repository (if not already)
git clone <your-repo-url>
cd heroku

# Ensure you're on the correct branch
git checkout claude/build-gcm-physics-VaCFZ
```

### Step 2: Login to Heroku

```bash
heroku login
```

This will open your browser to login.

### Step 3: Create Heroku App

```bash
# Create a new app (name must be unique)
heroku create my-gcm-app

# Or let Heroku generate a name
heroku create
```

This will output something like:
```
Creating app... done, ⬢ my-gcm-app
https://my-gcm-app.herokuapp.com/ | https://git.heroku.com/my-gcm-app.git
```

### Step 4: Configure Buildpacks

```bash
# Python buildpack (should be automatic, but verify)
heroku buildpacks:add heroku/python
```

### Step 5: Deploy

```bash
# Push to Heroku
git push heroku claude/build-gcm-physics-VaCFZ:main

# Or if you're on main branch
git push heroku main
```

You'll see output like:
```
remote: -----> Building on the Heroku-22 stack
remote: -----> Using buildpack: heroku/python
remote: -----> Python app detected
remote: -----> Installing python-3.11.7
remote: -----> Installing pip dependencies
...
remote: -----> Compressing...
remote: -----> Launching...
remote: https://my-gcm-app.herokuapp.com/ deployed to Heroku
```

### Step 6: Open Your App

```bash
heroku open
```

Or visit: `https://my-gcm-app.herokuapp.com`

## Scaling and Configuration

### Check Status

```bash
heroku ps
```

### View Logs

```bash
# Real-time logs
heroku logs --tail

# Last 1000 lines
heroku logs -n 1000
```

### Scale Dynos

```bash
# Scale up to 2 workers (costs money)
heroku ps:scale web=2

# Scale down to save resources
heroku ps:scale web=1
```

### Set Environment Variables

```bash
heroku config:set FLASK_ENV=production
heroku config:set MAX_SIMULATION_DURATION=30
```

### Upgrade Dyno Type

For better performance:

```bash
# Hobby dyno ($7/month) - recommended for regular use
heroku ps:type hobby

# Standard dyno ($25-50/month) - for heavy use
heroku ps:type standard-1x
```

## Performance Optimization

### Dyno Types and Capabilities

| Dyno Type | RAM | CPU | Best For | Cost |
|-----------|-----|-----|----------|------|
| Free | 512 MB | Shared | Testing | Free |
| Hobby | 512 MB | Shared | Light use | $7/mo |
| Standard-1X | 512 MB | 1x | Regular use | $25/mo |
| Standard-2X | 1 GB | 2x | Heavy use | $50/mo |

### Recommended Settings by Dyno

**Free/Hobby:**
- Max resolution: 48×24×16
- Max duration: 10 days
- Timeout: 300s

**Standard-1X:**
- Max resolution: 64×32×20
- Max duration: 30 days
- Timeout: 600s

**Standard-2X:**
- Max resolution: 96×48×32
- Max duration: 60 days
- Timeout: 900s

### Optimize Slug Size

Already configured in `.slugignore`:
- Documentation excluded
- Test files excluded
- Example plots excluded

Current slug size: ~150-200 MB

## Troubleshooting

### App Won't Start

```bash
# Check logs
heroku logs --tail

# Restart app
heroku restart

# Check buildpack
heroku buildpacks
```

### Out of Memory

```bash
# Upgrade dyno type
heroku ps:type standard-1x

# Or reduce simulation resolution in app
```

### Timeout Errors

Edit `Procfile`:
```
web: gunicorn app:app --timeout 600 --workers 2
```

Then commit and push:
```bash
git add Procfile
git commit -m "Increase timeout"
git push heroku main
```

### Slow Performance

1. **Use smaller resolution** in web UI
2. **Upgrade dyno type**
3. **Enable worker dynos:**
   ```bash
   heroku ps:scale worker=1
   ```

### Database for Results

To persist results:

```bash
# Add PostgreSQL (free tier)
heroku addons:create heroku-postgresql:mini

# Get database URL
heroku config:get DATABASE_URL
```

## Monitoring

### View App Metrics

```bash
# Open metrics dashboard
heroku open --metrics
```

### Set Up Alerts

In Heroku Dashboard:
1. Go to your app
2. Click "Metrics"
3. Set up alerts for:
   - Response time > 2s
   - Memory > 80%
   - Errors > 10/hour

## Custom Domain

### Add Domain

```bash
# Add your domain
heroku domains:add www.my-gcm-app.com

# Get DNS target
heroku domains
```

### Configure DNS

Add CNAME record:
```
CNAME: www
Target: <shown-dns-target>.herokudns.com
```

### Enable SSL

```bash
# Automatic SSL (free)
heroku certs:auto:enable
```

## Backup and Restore

### Backup Code

```bash
# Download Heroku git repo
heroku git:clone -a my-gcm-app
```

### Restore from Backup

```bash
# Roll back to previous release
heroku releases
heroku rollback v123
```

## CI/CD Integration

### GitHub Integration

1. Go to Heroku Dashboard
2. Select your app
3. Click "Deploy" tab
4. Connect to GitHub
5. Enable automatic deploys
6. Choose branch: `claude/build-gcm-physics-VaCFZ`

Now every push automatically deploys!

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Heroku

on:
  push:
    branches: [ claude/build-gcm-physics-VaCFZ ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: akhileshns/heroku-deploy@v3.12.12
        with:
          heroku_api_key: ${{secrets.HEROKU_API_KEY}}
          heroku_app_name: "my-gcm-app"
          heroku_email: "your-email@example.com"
```

## Cost Estimation

### Free Tier
- 550 dyno hours/month
- Sleeps after 30 min inactivity
- **Cost: $0**

### Hobby Tier
- Always on
- Custom domains
- SSL included
- **Cost: $7/month**

### Standard Tier
- Better performance
- Higher memory
- Metrics & alerts
- **Cost: $25-50/month**

### Typical Usage Costs

**Light use (testing, demos):**
- Free or Hobby tier
- **$0-7/month**

**Regular use (small team):**
- Hobby or Standard-1X
- **$7-25/month**

**Production use (multiple users):**
- Standard-2X + worker dynos
- **$50-100/month**

## Security Best Practices

### Environment Variables

Never commit secrets. Use config vars:

```bash
heroku config:set SECRET_KEY=your-secret-key
heroku config:set API_TOKEN=your-api-token
```

### HTTPS Only

```python
# In app.py, add:
if not app.debug:
    @app.before_request
    def force_https():
        if request.headers.get('X-Forwarded-Proto') != 'https':
            return redirect(request.url.replace('http://', 'https://'))
```

### Rate Limiting

```bash
pip install flask-limiter
```

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    default_limits=["200 per day", "50 per hour"]
)
```

## Advanced: Custom Workers

For long simulations, use worker dynos.

### Create `worker.py`:

```python
import redis
from rq import Worker, Queue, Connection

listen = ['default']

redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
```

### Update `Procfile`:

```
web: gunicorn app:app
worker: python worker.py
```

### Scale:

```bash
heroku ps:scale worker=1
```

## Support

- **Documentation**: See README.md and WEB_README.md
- **Heroku Help**: https://help.heroku.com
- **Issues**: Open on GitHub

## Next Steps

1. ✅ Deploy to Heroku
2. 🔧 Configure custom domain
3. 📊 Set up monitoring
4. 🔐 Enable SSL
5. 🚀 Share with users!

Happy simulating! 🌍
