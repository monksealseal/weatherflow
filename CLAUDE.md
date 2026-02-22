# WeatherFlow Development Conventions

## Architecture
- **Frontend**: React 18 + TypeScript + Vite in `frontend/`
- **Backend**: FastAPI + PyTorch in `weatherflow/server/`
- **Streamlit**: Interactive app in `streamlit_app/`

## Commands
- `make dev-backend` — Start FastAPI on port 8000
- `make dev-frontend` — Start Vite dev server on port 5173
- `make deploy` — One-button cloud deploy (frontend + backend)
- `make deploy-status` — Check deployment health
- `make scale-up` — Scale Railway to 4GB/4vCPU
- `cd frontend && npm run build` — Build frontend for production

## Frontend Conventions
- Views go in `frontend/src/components/views/`
- Each view has a `.tsx` and matching `.css` file
- API types in `frontend/src/api/nhcTypes.ts`, client in `nhcClient.ts`
- Navigation defined in `frontend/src/utils/navigation.ts`
- Use `view-container` class for top-level view wrapper
- Follow existing patterns: fetch in useEffect, loading states, error handling

## NHC Products
- The Hurricane Center hub at `/nhc/hub` provides dual Public/Scientist modes
- Public mode: plain-language for affected communities
- Scientist mode: tabbed interface with all 10 NHC products
- All NHC views wrapped in `NHCErrorBoundary`
- API client has exponential backoff retry (3 attempts)

## Deployment
- Frontend deploys to GitHub Pages on push to main
- Backend deploys to Railway via `deploy-cloud.yml` workflow
- One-time setup: `bash scripts/setup-cloud-secrets.sh`
- Railway secrets: RAILWAY_TOKEN, RAILWAY_PROJECT_ID, RAILWAY_SERVICE_ID
