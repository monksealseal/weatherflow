# GitHub Pages Deployment Instructions

## Prerequisites

Before the GitHub Actions workflow can successfully deploy to GitHub Pages, you need to enable GitHub Pages in the repository settings.

## Step-by-Step Setup

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** (top menu)
3. In the left sidebar, click on **Pages**
4. Under **Build and deployment**:
   - **Source**: Select **"GitHub Actions"** from the dropdown
   - This tells GitHub to use the workflow in `.github/workflows/deploy-pages.yml`

### 2. Verify Workflow Permissions

The workflow already has the required permissions set:
```yaml
permissions:
  contents: read
  pages: write
  id-token: write
```

### 3. Trigger Deployment

Once GitHub Pages is configured:

- **Automatic**: Push to the `main` branch will trigger deployment
- **Manual**: Go to Actions → Deploy to GitHub Pages → Run workflow

### 4. Access Your Site

After successful deployment, your site will be available at:
```
https://<username>.github.io/weatherflow/
```

## Troubleshooting

### "Setup Pages" Step Fails

**Error**: `Error: Unable to get information about Pages`

**Solution**: GitHub Pages must be enabled in repository settings with "GitHub Actions" as the source.

1. Go to Settings → Pages
2. Change Source from "None" to "GitHub Actions"
3. Re-run the workflow

### Build Fails

**Error**: Build errors in the "Build frontend" step

**Solution**: Test the build locally first:
```bash
cd frontend
npm install
GITHUB_PAGES=true npm run build
```

### Page Shows 404

**Possible causes**:
1. Base path misconfiguration - check `vite.config.ts` has correct base path
2. Missing `.nojekyll` file - should be in `frontend/public/` directory
3. Deployment didn't complete - check the Actions tab for errors

### Assets Not Loading

**Cause**: Incorrect base path in production

**Solution**: The `GITHUB_PAGES` environment variable is set in the workflow, which configures the base path as `/weatherflow/`. This should work automatically.

## Technical Details

### What the Workflow Does

1. **Build**: Compiles the React app with Vite
2. **Configure**: Sets up GitHub Pages environment
3. **Upload**: Packages the build artifacts
4. **Deploy**: Publishes to GitHub Pages

### Files Involved

- `.github/workflows/deploy-pages.yml` - Deployment workflow
- `frontend/vite.config.ts` - Build configuration with base path
- `frontend/public/.nojekyll` - Disables Jekyll processing
- `frontend/dist/` - Build output (not in git)

### Environment Variables

- `GITHUB_PAGES=true` - Enables production base path in build

## First-Time Setup Checklist

- [ ] Enable GitHub Pages in Settings → Pages
- [ ] Set Source to "GitHub Actions"
- [ ] Push to main branch or manually run workflow
- [ ] Wait for deployment to complete (~2 minutes)
- [ ] Visit `https://<username>.github.io/weatherflow/`

## Need Help?

If deployment fails after following these steps:
1. Check the Actions tab for detailed error logs
2. Verify all settings match this guide
3. Ensure you have admin access to the repository
4. Check if the repository is private (Pages requires public repo or GitHub Pro)
