# Quick Deployment Guide

## Option 1: One-Click Deploy (Easiest!)

Click this button to deploy instantly:

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/YOUR_USERNAME/YOUR_REPO)

*Note: Update the URL above with your GitHub repository*

## Option 2: Automated Script (Recommended)

```bash
# Run the deployment script
./deploy.sh
```

The script will:
1. ✓ Check Heroku CLI installation
2. ✓ Login to Heroku
3. ✓ Create the app
4. ✓ Deploy the code
5. ✓ Open in browser

## Option 3: Manual Commands

```bash
# 1. Login to Heroku
heroku login

# 2. Create app
heroku create my-gcm-app

# 3. Deploy
git push heroku claude/build-gcm-physics-VaCFZ:main

# 4. Open app
heroku open
```

## Prerequisites

### Install Heroku CLI

**macOS:**
```bash
brew tap heroku/brew && brew install heroku
```

**Ubuntu/Debian:**
```bash
curl https://cli-assets.heroku.com/install.sh | sh
```

**Windows:**
Download from: https://devcenter.heroku.com/articles/heroku-cli

### Verify Installation

```bash
heroku --version
# Should show: heroku/8.x.x
```

## First Time Setup

```bash
# Login
heroku login

# Verify
heroku auth:whoami
```

## Deployment Steps

### 1. Create Heroku App

```bash
# Auto-generated name
heroku create

# Or with custom name
heroku create my-gcm-app
```

This outputs:
```
Creating app... done, ⬢ my-gcm-app
https://my-gcm-app.herokuapp.com/ | https://git.heroku.com/my-gcm-app.git
```

### 2. Deploy Code

```bash
# Ensure you're on the right branch
git checkout claude/build-gcm-physics-VaCFZ

# Push to Heroku
git push heroku claude/build-gcm-physics-VaCFZ:main
```

You'll see:
```
remote: -----> Building on the Heroku-22 stack
remote: -----> Using buildpack: heroku/python
remote: -----> Python app detected
remote: -----> Installing python-3.11.7
remote: -----> Installing pip dependencies
remote:        Collecting numpy>=1.24.0
remote:        Collecting scipy>=1.10.0
remote:        Collecting flask>=3.0.0
remote:        Collecting gunicorn>=21.2.0
...
remote: -----> Compressing...
remote: -----> Launching...
remote:        https://my-gcm-app.herokuapp.com/ deployed to Heroku
```

### 3. Verify Deployment

```bash
# Check status
heroku ps

# View logs
heroku logs --tail

# Open in browser
heroku open
```

## Configuration

### Set Environment Variables

```bash
heroku config:set FLASK_ENV=production
heroku config:set MAX_WORKERS=2
```

### Scale Dynos

```bash
# Start with 1 dyno (free tier)
heroku ps:scale web=1

# Scale up for more capacity
heroku ps:scale web=2
```

### Upgrade Dyno Type

```bash
# Hobby dyno ($7/month)
heroku ps:type hobby

# Standard dyno ($25/month)
heroku ps:type standard-1x
```

## Monitoring

### View Logs

```bash
# Real-time logs
heroku logs --tail

# Last 1000 lines
heroku logs -n 1000

# Filter by type
heroku logs --source app
```

### Check App Status

```bash
# See running dynos
heroku ps

# See app info
heroku info

# See releases
heroku releases
```

### Metrics Dashboard

```bash
# Open metrics in browser
heroku open --metrics
```

## Troubleshooting

### App Not Starting

```bash
# Check logs
heroku logs --tail

# Restart
heroku restart

# Check buildpack
heroku buildpacks
```

### Memory Issues

```bash
# Upgrade to larger dyno
heroku ps:type standard-1x

# Check current memory usage
heroku ps
```

### Timeout Errors

Edit `Procfile`:
```
web: gunicorn app:app --timeout 600 --workers 2
```

Commit and redeploy:
```bash
git add Procfile
git commit -m "Increase timeout"
git push heroku claude/build-gcm-physics-VaCFZ:main
```

### Deployment Fails

```bash
# Check git remote
git remote -v

# If heroku remote missing, add it
heroku git:remote -a my-gcm-app

# Try deploying again
git push heroku claude/build-gcm-physics-VaCFZ:main
```

## Post-Deployment

### Test the App

1. Visit: `https://your-app.herokuapp.com`
2. Try running a small simulation:
   - Resolution: 32×16×10
   - Duration: 5 days
   - Profile: Tropical

### Set Up Custom Domain

```bash
# Add domain
heroku domains:add www.my-gcm.com

# Get DNS target
heroku domains

# Enable SSL (free)
heroku certs:auto:enable
```

### Enable Monitoring

```bash
# Add free New Relic
heroku addons:create newrelic:wayne

# Add free Papertrail for logs
heroku addons:create papertrail:choklad
```

## Updating the App

```bash
# Make changes locally
git add .
git commit -m "Update feature"

# Push to Heroku
git push heroku claude/build-gcm-physics-VaCFZ:main

# Heroku will automatically redeploy
```

## Cost Management

### Free Tier

- 550 dyno hours/month
- Sleeps after 30 min inactivity
- Perfect for demos and testing

### Keep Free Dyno Awake

Add to your app:
```python
# In app.py
from threading import Thread
import time
import requests

def keep_alive():
    while True:
        time.sleep(25 * 60)  # 25 minutes
        try:
            requests.get('https://your-app.herokuapp.com')
        except:
            pass

Thread(target=keep_alive, daemon=True).start()
```

### Hobby Tier ($7/month)

- Always on
- No sleeping
- Custom domains
- SSL included

## Useful Commands Summary

```bash
# Create and deploy
heroku create && git push heroku claude/build-gcm-physics-VaCFZ:main

# View logs
heroku logs --tail

# Restart app
heroku restart

# Open app
heroku open

# Shell access
heroku run bash

# Database console (if using PostgreSQL)
heroku pg:psql

# Config vars
heroku config
heroku config:set KEY=value

# Rollback to previous version
heroku releases
heroku rollback v123
```

## Success Checklist

- [ ] Heroku CLI installed
- [ ] Logged into Heroku
- [ ] App created
- [ ] Code deployed successfully
- [ ] App opens in browser
- [ ] Test simulation runs
- [ ] Logs look clean
- [ ] No errors on homepage

## Getting Help

- **Heroku Status**: https://status.heroku.com
- **Documentation**: https://devcenter.heroku.com
- **Support**: https://help.heroku.com
- **Community**: https://stackoverflow.com/questions/tagged/heroku

## What's Next?

1. ✅ Deploy the app
2. 🧪 Run test simulations
3. 🎨 Customize the UI
4. 📊 Add more visualizations
5. 🔐 Set up authentication (if needed)
6. 📈 Monitor performance
7. 🚀 Share with users!

---

**Your GCM is ready to simulate the climate! 🌍**
