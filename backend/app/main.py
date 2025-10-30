from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text
from .database import engine, SessionLocal
from .models import Base
import os
import logging
import signal
import sys
import atexit

# Import enhanced features
try:
    from .rate_limiting import rate_limit_middleware
    from .monitoring import structured_logger, health_check as enhanced_health_check
    RATE_LIMITING_ENABLED = True
except ImportError as e:
    logging.warning(f"Enhanced features not available: {e}")
    RATE_LIMITING_ENABLED = False

# Create necessary directories
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Document Q&A API",
    description="Enterprise Document Q&A System API with Advanced Features",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    # Increase maximum request body size for larger file uploads
    max_request_size=52428800  # 50MB
)

# CORS for frontend integration (enhanced)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "https://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware if available
if RATE_LIMITING_ENABLED:
    app.middleware("http")(rate_limit_middleware)

# Mount static files for uploaded documents
app.mount("/uploads", StaticFiles(directory="data/uploads"), name="uploads")

from .routers import auth, documents, chat, admin, hybrid_search
from .routers.processing import router as processing_router, process_document_complete
from .routers.search_feedback import router as search_feedback_router
from .routers.question_export import router as question_export_router
from .tasks import background_task_manager
from .services import DocumentProcessor
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from .auth import get_current_active_user
from .database import get_db, SessionLocal
from .models import User, Document

app.include_router(
    auth.router,
    prefix="/api/auth",
    tags=["authentication"]
)

app.include_router(
    documents.router,
    prefix="/api/documents",
    tags=["documents"]
)

app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["chat"]
)

app.include_router(
    admin.router,
    prefix="/api/admin",
    tags=["admin"]
)

app.include_router(
    processing_router,
    prefix="/api/processing",
    tags=["processing"]
)

app.include_router(
    hybrid_search.router,
    prefix="/api/hybrid-search",
    tags=["hybrid-search"]
)

app.include_router(
    search_feedback_router,
    prefix="/api/search-feedback",
    tags=["search-feedback"]
)

app.include_router(
    question_export_router,
    prefix="/api/question-export",
    tags=["question-export"]
)

# Initialize DocumentProcessor
document_processor = DocumentProcessor()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Q&A API v2.0",
        "status": "running",
        "features": [
            "Document Upload & Processing",
            "Advanced Search & Filtering",
            "Document Versioning",
            "Collaboration Features",
            "Background Processing",
            "Rate Limiting",
            "Enhanced Security",
            "Comprehensive Monitoring"
        ],
        "docs": "/api/docs"
    }

@app.get("/process_document/{document_id}")
async def process_document(document_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """Process a specific document by ID"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        # Process the document
        result = await process_document_complete(document_id, current_user, db)
        return {"message": f"Document {document_id} processing triggered", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Basic database connection test
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()

        # Use enhanced health check if available
        if 'enhanced_health_check' in globals():
            return enhanced_health_check()

        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        structured_logger.log_error(
            "health_check_failed",
            str(e),
            endpoint="/health"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/api/metrics")
async def get_metrics():
    """Get application metrics (requires authentication)"""
    # This would return detailed metrics
    # For now, return basic information
    return {
        "status": "Metrics endpoint ready",
        "features": ["Rate limiting", "Monitoring", "Background tasks"],
        "note": "Authentication required for detailed metrics"
    }

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Global exception handler with structured logging"""
    structured_logger.log_error(
        "unhandled_exception",
        str(exc),
        endpoint=request.url.path,
        method=request.method,
        client_ip=getattr(request.client, 'host', 'unknown')
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred", "message": "Internal server error"}
    )


from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette.requests import Request

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles validation errors and logs them."""
    structured_logger.log_error(
        "validation_error",
        str(exc),
        endpoint=request.url.path,
        method=request.method,
        client_ip=getattr(request.client, 'host', 'unknown'),
        detail=exc.errors()
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Validation error"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles HTTP exceptions and logs them."""
    structured_logger.log_error(
        "http_exception",
        str(exc),
        endpoint=request.url.path,
        method=request.method,
        client_ip=getattr(request.client, 'host', 'unknown'),
        status_code=exc.status_code,
        detail=exc.detail
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "message": str(exc.detail)}
    )

# Shutdown handling for background tasks
def shutdown_handler():
    """Handle graceful shutdown"""
    print("🛑 Received shutdown signal, stopping background task manager...")
    background_task_manager.shutdown()

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print(f"🛑 Received signal {sig}, shutting down gracefully...")
    shutdown_handler()
    sys.exit(0)

# Register shutdown handlers
atexit.register(shutdown_handler)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Enhanced health check with background processing status
@app.get("/health/detailed")
async def detailed_health_check():
    """Enhanced health check with background processing status"""
    try:
        # Basic database connection test
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()

        # Get background processing queue status
        queue_stats = background_task_manager.get_queue_stats()

        return {
            "status": "healthy",
            "database": "connected",
            "background_processing": {
                "queue_size": queue_stats["queue_size"],
                "active_jobs": queue_stats["active_jobs"],
                "max_workers": queue_stats["max_workers"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        structured_logger.log_error(
            "detailed_health_check_failed",
            str(e),
            endpoint="/health/detailed"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )