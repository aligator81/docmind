from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
import logging

from ..database import get_db
from ..auth import get_current_user
from ..models import User
from ..schemas import (
    HybridSearchRequest, 
    HybridSearchResponse, 
    SearchAnalyticsResponse
)
from ..services.hybrid_search_service import create_hybrid_search_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hybrid-search", tags=["hybrid-search"])

@router.post("/search", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Perform hybrid search combining vector similarity and full-text search
    """
    try:
        # Set user_id for the request
        request.user_id = current_user.id
        
        # Create search service and perform search
        search_service = create_hybrid_search_service(db)
        result = search_service.hybrid_search(request)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in hybrid search endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Search failed: {str(e)}"
        )

@router.get("/analytics", response_model=SearchAnalyticsResponse)
async def get_search_analytics(
    days: int = Query(30, description="Number of days to analyze", ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get search analytics for the specified period
    """
    try:
        # Only allow admins to access analytics
        if current_user.role not in ["admin", "super_admin"]:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions to access search analytics"
            )
        
        # Create search service and get analytics
        search_service = create_hybrid_search_service(db)
        result = search_service.get_search_analytics(days)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting search analytics: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get analytics: {str(e)}"
        )

@router.get("/health")
async def search_health_check(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Health check for hybrid search functionality
    """
    try:
        # Test basic search functionality
        search_service = create_hybrid_search_service(db)
        
        # Test search vector update
        search_service._update_search_vectors()
        
        # Test analytics (basic call)
        analytics = search_service.get_search_analytics(1)
        
        return {
            "status": "healthy",
            "search_vectors_updated": True,
            "analytics_available": analytics["success"]
        }
        
    except Exception as e:
        logger.error(f"Search health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Search service unhealthy: {str(e)}"
        )

@router.get("/config")
async def get_search_config(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current search configuration
    """
    try:
        # Get default configuration
        config_query = """
            SELECT config_value 
            FROM search_config 
            WHERE config_key = 'hybrid_search_weights' 
            AND is_active = true
        """
        
        result = db.execute(config_query).fetchone()
        
        if result:
            config = result[0]
        else:
            # Default configuration
            config = {
                "vector_weight": 0.6,
                "text_weight": 0.4,
                "similarity_threshold": 0.7,
                "max_results": 50
            }
        
        return {
            "success": True,
            "config": config
        }
        
    except Exception as e:
        logger.error(f"Error getting search config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search configuration: {str(e)}"
        )