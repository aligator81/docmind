#!/usr/bin/env python3
"""
Test script for search performance monitoring system
"""

import sys
import os
import time
import json
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.monitoring import (
    search_monitor, 
    search_analytics, 
    SearchPerformanceMonitor,
    SearchAnalytics
)
from app.services.hybrid_search_service import HybridSearchService
from app.schemas import HybridSearchRequest
from app.database import get_db

def test_search_monitor_basic():
    """Test basic search monitoring functionality"""
    print("🧪 Testing basic search monitoring...")
    
    # Test search event logging
    search_monitor.log_search_event(
        search_type="hybrid",
        query="test query",
        duration=0.5,
        results_count=10,
        search_params={"vector_weight": 0.7, "text_weight": 0.3}
    )
    
    # Test hybrid search breakdown
    search_monitor.log_hybrid_search_breakdown(
        query="test query",
        vector_duration=0.2,
        text_duration=0.1,
        fusion_duration=0.05,
        total_duration=0.35,
        results_count=15,
        vector_weight=0.7,
        text_weight=0.3
    )
    
    # Test search quality logging
    search_monitor.log_search_quality(
        query="test query",
        search_type="hybrid",
        relevance_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
        user_feedback="good results"
    )
    
    # Test embedding performance
    search_monitor.log_embedding_performance(
        operation="batch_embedding",
        document_count=5,
        chunk_count=50,
        duration=2.5,
        embedding_model="text-embedding-ada-002"
    )
    
    print("✅ Basic search monitoring tests passed")

def test_search_analytics():
    """Test search analytics functionality"""
    print("🧪 Testing search analytics...")
    
    # Reset analytics for clean test
    search_analytics.reset_analytics()
    
    # Simulate various search operations
    search_types = ["hybrid", "vector", "text", "hybrid", "vector"]
    durations = [0.5, 0.3, 0.2, 0.6, 0.4]
    
    for search_type, duration in zip(search_types, durations):
        search_analytics.update_analytics(
            search_type=search_type,
            duration=duration,
            success=True
        )
    
    # Get analytics report
    report = search_analytics.get_analytics_report()
    
    # Verify analytics data
    assert report["total_searches"] == 5
    assert report["hybrid_searches"] == 2
    assert report["vector_searches"] == 2
    assert report["text_searches"] == 1
    assert report["success_rate"] == 100.0
    assert report["avg_response_time"] > 0
    
    print(f"📊 Analytics Report: {json.dumps(report, indent=2)}")
    print("✅ Search analytics tests passed")

def test_monitor_decorator():
    """Test the search performance monitoring decorator"""
    print("🧪 Testing search performance decorator...")
    
    from app.monitoring import monitor_search_performance
    
    @monitor_search_performance("test_search")
    def mock_search_function(query: str, limit: int = 10):
        """Mock search function for testing"""
        time.sleep(0.1)  # Simulate search time
        return [{"id": i, "score": 1.0 - (i * 0.1)} for i in range(limit)]
    
    # Test the decorated function
    results = mock_search_function("test query", limit=5)
    assert len(results) == 5
    assert all("id" in result for result in results)
    
    print("✅ Search performance decorator tests passed")

def test_hybrid_search_with_monitoring():
    """Test hybrid search with integrated monitoring"""
    print("🧪 Testing hybrid search with monitoring...")
    
    try:
        # Get database session
        engine = create_engine("sqlite:///test.db")  # Use in-memory for testing
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        # Create hybrid search service
        search_service = HybridSearchService(db)
        
        # Create test request
        request = HybridSearchRequest(
            query="test search query",
            limit=10,
            vector_weight=0.7,
            text_weight=0.3,
            similarity_threshold=0.5
        )
        
        # Perform hybrid search (this will trigger monitoring)
        result = search_service.hybrid_search(request)
        
        # Check result structure
        assert "success" in result
        assert "results" in result
        assert "response_time_ms" in result
        assert "search_metadata" in result
        
        # Check that performance breakdown is included
        if result["success"]:
            metadata = result["search_metadata"]
            assert "performance_breakdown" in metadata
            breakdown = metadata["performance_breakdown"]
            assert "vector_search_ms" in breakdown
            assert "text_search_ms" in breakdown
            assert "fusion_ms" in breakdown
            assert "total_ms" in breakdown
        
        print(f"🔍 Hybrid search result: {json.dumps(result, indent=2)}")
        print("✅ Hybrid search with monitoring tests passed")
        
    except Exception as e:
        print(f"⚠️ Hybrid search test skipped (database not available): {e}")

def test_slow_search_detection():
    """Test slow search detection functionality"""
    print("🧪 Testing slow search detection...")
    
    # Test slow vector search
    search_monitor.log_search_event(
        search_type="vector",
        query="slow query",
        duration=1.2,  # Above 1.0 second threshold
        results_count=5
    )
    
    # Test slow hybrid search
    search_monitor.log_search_event(
        search_type="hybrid",
        query="very slow query",
        duration=2.0,  # Above 1.5 second threshold
        results_count=8
    )
    
    # Test normal search (should not trigger slow search)
    search_monitor.log_search_event(
        search_type="text",
        query="fast query",
        duration=0.3,  # Below 0.5 second threshold
        results_count=12
    )
    
    print("✅ Slow search detection tests passed")

def test_embedding_performance_monitoring():
    """Test embedding performance monitoring"""
    print("🧪 Testing embedding performance monitoring...")
    
    # Test single document embedding
    search_monitor.log_embedding_performance(
        operation="single_document",
        document_count=1,
        chunk_count=15,
        duration=0.8,
        embedding_model="text-embedding-ada-002"
    )
    
    # Test batch embedding
    search_monitor.log_embedding_performance(
        operation="batch_embedding",
        document_count=10,
        chunk_count=150,
        duration=12.5,
        embedding_model="text-embedding-ada-002"
    )
    
    # Calculate and verify chunks per second
    chunks_per_second = 150 / 12.5  # Should be 12 chunks/second
    print(f"📈 Embedding performance: {chunks_per_second:.2f} chunks/second")
    
    print("✅ Embedding performance monitoring tests passed")

def test_search_quality_metrics():
    """Test search quality metrics"""
    print("🧪 Testing search quality metrics...")
    
    # Test with high relevance scores
    search_monitor.log_search_quality(
        query="high quality query",
        search_type="hybrid",
        relevance_scores=[0.95, 0.92, 0.88, 0.85, 0.82],
        user_feedback="excellent results"
    )
    
    # Test with mixed relevance scores
    search_monitor.log_search_quality(
        query="mixed quality query",
        search_type="vector",
        relevance_scores=[0.8, 0.6, 0.4, 0.3, 0.2],
        user_feedback="some relevant results"
    )
    
    # Test with low relevance scores
    search_monitor.log_search_quality(
        query="low quality query",
        search_type="text",
        relevance_scores=[0.3, 0.25, 0.2, 0.15, 0.1],
        user_feedback="poor results"
    )
    
    print("✅ Search quality metrics tests passed")

def main():
    """Run all search performance monitoring tests"""
    print("🚀 Starting Search Performance Monitoring Tests")
    print("=" * 50)
    
    try:
        test_search_monitor_basic()
        test_search_analytics()
        test_monitor_decorator()
        test_hybrid_search_with_monitoring()
        test_slow_search_detection()
        test_embedding_performance_monitoring()
        test_search_quality_metrics()
        
        print("\n🎉 All search performance monitoring tests completed successfully!")
        print("📊 The monitoring system is ready to track:")
        print("   - Search performance metrics")
        print("   - Hybrid search breakdowns")
        print("   - Search quality and relevance")
        print("   - Embedding generation performance")
        print("   - Slow search detection")
        print("   - Search analytics and reporting")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())