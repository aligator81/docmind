#!/usr/bin/env python3
"""
Test script for hybrid search functionality
"""

import sys
import os
sys.path.append('.')

def test_hybrid_search_imports():
    """Test that all required imports work"""
    try:
        from app.database import SessionLocal
        print("✅ Database import successful")
        
        from app.models import SearchHistory
        print("✅ SearchHistory model import successful")
        
        from app.schemas import HybridSearchRequest
        print("✅ HybridSearchRequest schema import successful")
        
        from app.services.hybrid_search_service import HybridSearchService
        print("✅ HybridSearchService import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_database_connection():
    """Test database connection and schema"""
    try:
        from app.database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        
        # Test if search_history table exists
        result = db.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'search_history')")).fetchone()
        if result[0]:
            print("✅ search_history table exists")
        else:
            print("❌ search_history table does not exist")
        
        # Test if search_vector column exists
        result = db.execute(text("SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'document_chunks' AND column_name = 'search_vector')")).fetchone()
        if result[0]:
            print("✅ search_vector column exists")
        else:
            print("❌ search_vector column does not exist")
        
        # Test if vector extension is enabled
        result = db.execute(text("SELECT EXISTS (SELECT FROM pg_extension WHERE extname = 'vector')")).fetchone()
        if result[0]:
            print("✅ pgvector extension is enabled")
        else:
            print("❌ pgvector extension is not enabled")
        
        db.close()
        return True
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def test_hybrid_search_service():
    """Test hybrid search service creation"""
    try:
        from app.database import SessionLocal
        from app.services.hybrid_search_service import create_hybrid_search_service
        
        db = SessionLocal()
        search_service = create_hybrid_search_service(db)
        
        print("✅ Hybrid search service created successfully")
        
        # Test search vector update
        search_service._update_search_vectors()
        print("✅ Search vectors updated successfully")
        
        db.close()
        return True
    except Exception as e:
        print(f"❌ Hybrid search service test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Hybrid Search Implementation...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_hybrid_search_imports():
        tests_passed += 1
    
    if test_database_connection():
        tests_passed += 1
    
    if test_hybrid_search_service():
        tests_passed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All hybrid search tests passed!")
    else:
        print("⚠️ Some tests failed. Check the implementation.")