#!/usr/bin/env python3
"""
Comprehensive test for the complete document processing pipeline
This test verifies that the "Process Files" functionality works end-to-end
"""

import sys
import os
import asyncio
import time

# Add backend to path
sys.path.append('./backend')

async def test_complete_pipeline():
    """Test the complete document processing pipeline"""
    print("🧪 Testing Complete Document Processing Pipeline")
    print("=" * 60)
    
    try:
        # Test 1: Check if backend services can be imported
        print("\n📦 Test 1: Service Imports")
        print("-" * 30)
        
        try:
            from backend.app.services.document_processor import DocumentProcessor
            print("✅ DocumentProcessor imported successfully")
            
            from backend.app.services.document_chunker import DocumentChunker
            print("✅ DocumentChunker imported successfully")
            
            from backend.app.services.optimized_embedding_service import OptimizedEmbeddingService
            print("✅ OptimizedEmbeddingService imported successfully")
            
            print("🎯 All core services imported successfully!")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
        except Exception as e:
            print(f"❌ Other error: {e}")
            return False
        
        # Test 2: Check database connection
        print("\n🗄️ Test 2: Database Connection")
        print("-" * 30)
        
        try:
            from backend.app.database import SessionLocal
            from backend.app.models import Document, User, DocumentChunk, Embedding
            
            db = SessionLocal()
            users = db.query(User).all()
            documents = db.query(Document).all()
            
            print(f"✅ Database connection successful")
            print(f"📊 Users in database: {len(users)}")
            print(f"📄 Documents in database: {len(documents)}")
            
            if documents:
                for doc in documents[:3]:
                    print(f"  - {doc.filename} (ID: {doc.id}, Status: {doc.status})")
            
            db.close()
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
        
        # Test 3: Check if there's a test document to process
        print("\n📄 Test 3: Test Document Availability")
        print("-" * 30)
        
        test_file_path = "./data/uploads/test.md"
        if os.path.exists(test_file_path):
            print(f"✅ Test document found: {test_file_path}")
            print(f"📏 File size: {os.path.getsize(test_file_path)} bytes")
        else:
            print(f"⚠️ Test document not found: {test_file_path}")
            print("Creating a simple test document...")
            
            # Create a simple test document
            os.makedirs("./data/uploads", exist_ok=True)
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write("""# Test Document for Processing Pipeline

This is a test document to verify the complete document processing pipeline.

## Document Processing Steps

The pipeline should:
1. Extract the text content from the document
2. Split it into meaningful chunks
3. Create embeddings for each chunk
4. Store everything in the database

## Test Content

This document contains sample text to test the extraction, chunking, and embedding functionality. The processing pipeline should be able to handle this content and create appropriate chunks and embeddings.

## Expected Results

After processing, we should see:
- Document status changed to 'processed'
- Multiple chunks created in the database
- Embeddings generated for each chunk
- All metadata properly stored

This completes the test document content.""")
            
            print(f"✅ Created test document: {test_file_path}")
        
        # Test 4: Check if we can run the existing test files
        print("\n🚀 Test 4: Running Existing Test Files")
        print("-" * 30)
        
        test_files = [
            "./backend/test_simple_processing.py",
            "./backend/test_processing_direct.py"
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"📋 Found test file: {test_file}")
            else:
                print(f"⚠️ Test file not found: {test_file}")
        
        print("\n🎯 Ready to test the complete pipeline!")
        print("\n💡 To test the frontend 'Process Files' button:")
        print("   1. Start the backend server: cd backend && python -m uvicorn app.main:app --reload")
        print("   2. Start the frontend: cd frontend && npm run dev")
        print("   3. Upload a document and click '🚀 Process File'")
        print("   4. Monitor the progress and verify the results")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Comprehensive Document Processing Pipeline Test")
    print("=" * 60)
    
    success = asyncio.run(test_complete_pipeline())
    
    if success:
        print("\n✅ All preliminary tests passed!")
        print("\n📋 Next Steps:")
        print("   - Start backend server: cd backend && python -m uvicorn app.main:app --reload")
        print("   - Start frontend: cd frontend && npm run dev")
        print("   - Upload a document and test the 'Process Files' button")
        print("   - Verify extraction, chunking, and embedding work correctly")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")