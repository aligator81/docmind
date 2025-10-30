"""
Search Feedback Router
Handles user feedback collection and relevance tuning operations.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.relevance_tuner import RelevanceTuner, FeedbackType, TuningStrategy
from ..schemas import (
    FeedbackRequest, 
    FeedbackResponse, 
    AnalysisResponse,
    TuningRequest,
    TuningResponse,
    ABTestRequest,
    ABTestResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search-feedback", tags=["search-feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_search_feedback(
    feedback_request: FeedbackRequest,
    db: Session = Depends(get_db)
) -> FeedbackResponse:
    """
    Submit search feedback for relevance tuning.
    
    Args:
        feedback_request: Feedback submission request
        db: Database session
        
    Returns:
        Feedback submission response
    """
    try:
        tuner = RelevanceTuner(db)
        
        result = tuner.collect_feedback(
            query=feedback_request.query,
            document_ids=feedback_request.document_ids,
            feedback_type=FeedbackType(feedback_request.feedback_type),
            user_id=feedback_request.user_id
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Failed to collect feedback: {result.get('error', 'Unknown error')}"
            )
        
        return FeedbackResponse(
            success=True,
            feedback_id=result.get("feedback_id"),
            query_id=result.get("query_id"),
            message=result.get("message", "Feedback collected successfully")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid feedback type: {str(e)}")
    except Exception as e:
        logger.error(f"Error submitting search feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analysis", response_model=AnalysisResponse)
async def get_feedback_analysis(
    days_back: int = Query(30, description="Number of days to analyze", ge=1, le=365),
    db: Session = Depends(get_db)
) -> AnalysisResponse:
    """
    Get feedback analysis for search relevance.
    
    Args:
        days_back: Number of days to analyze
        db: Database session
        
    Returns:
        Feedback analysis response
    """
    try:
        tuner = RelevanceTuner(db)
        analysis = tuner.analyze_feedback_patterns(days_back)
        
        if "error" in analysis:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {analysis['error']}"
            )
        
        return AnalysisResponse(
            success=True,
            analysis_period=analysis.get("analysis_period"),
            feedback_statistics=analysis.get("feedback_statistics", {}),
            query_analysis=analysis.get("query_analysis", {}),
            relevance_issues=analysis.get("relevance_issues", []),
            tuning_recommendations=analysis.get("tuning_recommendations", []),
            summary=analysis.get("summary", {})
        )
        
    except Exception as e:
        logger.error(f"Error getting feedback analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/tune", response_model=TuningResponse)
async def tune_search_relevance(
    tuning_request: TuningRequest,
    db: Session = Depends(get_db)
) -> TuningResponse:
    """
    Tune search relevance parameters based on feedback.
    
    Args:
        tuning_request: Tuning request parameters
        db: Database session
        
    Returns:
        Tuning operation response
    """
    try:
        tuner = RelevanceTuner(db)
        
        # Get pre-computed analysis if provided
        feedback_analysis = None
        if tuning_request.use_precomputed_analysis:
            feedback_analysis = tuner.analyze_feedback_patterns()
        
        result = tuner.tune_relevance_parameters(
            strategy=TuningStrategy(tuning_request.strategy),
            feedback_analysis=feedback_analysis
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail=f"Tuning failed: {result['error']}"
            )
        
        return TuningResponse(
            success=True,
            strategy=result.get("strategy"),
            parameters_used=result.get("parameters_used", {}),
            tuning_results=result.get("tuning_results", {}),
            effectiveness_estimate=result.get("effectiveness_estimate", {}),
            recommendations=result.get("recommendations", [])
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tuning strategy: {str(e)}")
    except Exception as e:
        logger.error(f"Error tuning search relevance: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ab-test", response_model=ABTestResponse)
async def create_ab_test(
    ab_test_request: ABTestRequest,
    db: Session = Depends(get_db)
) -> ABTestResponse:
    """
    Create an A/B test for comparing tuning strategies.
    
    Args:
        ab_test_request: A/B test configuration
        db: Database session
        
    Returns:
        A/B test creation response
    """
    try:
        tuner = RelevanceTuner(db)
        
        result = tuner.create_ab_test(
            strategy_a=TuningStrategy(ab_test_request.strategy_a),
            strategy_b=TuningStrategy(ab_test_request.strategy_b),
            test_duration_days=ab_test_request.test_duration_days
        )
        
        return ABTestResponse(
            success=True,
            test_id=result.get("test_id"),
            strategy_a=result.get("strategy_a"),
            strategy_b=result.get("strategy_b"),
            test_duration_days=result.get("test_duration_days"),
            start_time=result.get("start_time"),
            end_time=result.get("end_time"),
            status=result.get("status"),
            metrics_to_track=result.get("metrics_to_track", [])
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tuning strategy: {str(e)}")
    except Exception as e:
        logger.error(f"Error creating A/B test: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tuning-history")
async def get_tuning_history(
    days_back: int = Query(7, description="Number of days to look back", ge=1, le=30),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get history of tuning operations.
    
    Args:
        days_back: Number of days to look back
        db: Database session
        
    Returns:
        Tuning history
    """
    try:
        tuner = RelevanceTuner(db)
        history = tuner.get_tuning_history(days_back)
        
        return {
            "success": True,
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error getting tuning history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/feedback-types")
async def get_feedback_types() -> Dict[str, Any]:
    """
    Get available feedback types.
    
    Returns:
        Available feedback types
    """
    return {
        "success": True,
        "feedback_types": [
            {"value": fb_type.value, "description": fb_type.name}
            for fb_type in FeedbackType
        ]
    }


@router.get("/tuning-strategies")
async def get_tuning_strategies() -> Dict[str, Any]:
    """
    Get available tuning strategies.
    
    Returns:
        Available tuning strategies
    """
    return {
        "success": True,
        "tuning_strategies": [
            {"value": strategy.value, "description": strategy.name}
            for strategy in TuningStrategy
        ]
    }


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for search feedback service.
    
    Returns:
        Health status
    """
    return {"status": "healthy", "service": "search-feedback"}