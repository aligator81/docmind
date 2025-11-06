# Coolify Deployment Guide for Docling v2

## Problem Analysis
The original Docker build failed because:
- The build process tried to run `npm run build` from the root directory
- Frontend dependencies were only available in the `frontend/` directory
- The Docker context didn't have proper separation between backend and frontend

## Solution
We've created separate Docker configurations for backend and frontend services:

### 1. Backend Service (FastAPI)
- **Dockerfile**: `backend/Dockerfile`
- **Port**: 8000
- **Environment**: Python 3.11 with FastAPI dependencies

### 2. Frontend Service (Next.js)
- **Dockerfile**: `frontend/Dockerfile`
- **Port**: 3000
- **Environment**: Node.js 20 with Next.js dependencies

## Deployment Instructions for Coolify

### Option 1: Use docker-compose.prod.yml
1. Upload your project to Coolify
2. Use the provided `docker-compose.prod.yml` file
3. Ensure your `.env` file contains:
   - `NEXTAUTH_SECRET` (for authentication)
   - `OPENAI_API_KEY` (for AI features)
   - `MISTRAL_API_KEY` (optional, for alternative AI provider)
   - `SECRET_KEY` (for application security)

### Option 2: Manual Service Setup
If Coolify doesn't support docker-compose, set up two separate services:

#### Backend Service:
- **Build Context**: `./backend`
- **Dockerfile**: `Dockerfile` (in backend directory)
- **Port**: 8000
- **Environment Variables**:
  - `API_HOST=0.0.0.0`
  - `API_PORT=8000`
  - `WORKERS=1`
  - `API_RELOAD=false`

#### Frontend Service:
- **Build Context**: `./frontend`
- **Dockerfile**: `Dockerfile` (in frontend directory)
- **Port**: 3000
- **Environment Variables**:
  - `NEXT_PUBLIC_API_URL=http://backend:8000` (use your backend service URL)
  - `NEXTAUTH_URL=http://your-domain.com` (your actual domain)
  - `NEXTAUTH_SECRET=your-secret-key`
  - `NODE_ENV=production`

## Environment Variables
Create a `.env` file with the following variables:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key
SECRET_KEY=your_secret_key

# NextAuth
NEXTAUTH_SECRET=your_nextauth_secret
NEXTAUTH_URL=http://your-domain.com

# Database (if using external database)
DATABASE_URL=your_database_url
```

## Health Checks
- Backend: `http://your-backend:8000/health`
- Frontend: `http://your-frontend:3000`

## Troubleshooting
1. **Build Fails**: Ensure all dependencies are properly installed in their respective directories
2. **Frontend Can't Connect to Backend**: Verify `NEXT_PUBLIC_API_URL` points to the correct backend service
3. **Authentication Issues**: Check `NEXTAUTH_SECRET` and `NEXTAUTH_URL` configuration

## Notes
- The frontend is built with Next.js standalone output for optimal performance
- Backend uses FastAPI with Uvicorn for high-performance async operations
- Both services are designed to be scalable and production-ready