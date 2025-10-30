#!/usr/bin/env python3
"""
Simple test for document processing pipeline with a text file
"""

import sys
import os
import asyncio
sys.path.append('.')

from app.database import SessionLocal
from app.models import Document, DocumentChunk, Embedding, User
from app.services import DocumentProcessor, DocumentChunker
from app.services.optimized_embedding_service import OptimizedEmbeddingService

async def test_simple_document_processing():
    """Test the complete document processing pipeline with a simple text file"""
    db = SessionLocal()
    
    try:
        # Get a test user (use the first user)
        user = db.query(User).first()
        if not user:
            print("❌ No users found in database")
            return False
        
        print(f"👤 Using user: {user.username} (ID: {user.id})")
        
        # Create a test document record
        test_file_path = "../data/uploads/test_simple.txt"
        if not os.path.exists(test_file_path):
            print(f"❌ Test file not found: {test_file_path}")
            return False
        
        document = Document(
            filename="test_simple.txt",
            original_filename="test_simple.txt",
            file_path=test_file_path,
            file_size=os.path.getsize(test_file_path),
            mime_type="text/plain",
            user_id=user.id,
            status="not processed"
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        print(f"📄 Created test document: {document.filename} (ID: {document.id})")
        
        # Initialize services
        document_processor = DocumentProcessor()
        document_chunker = DocumentChunker()
        optimized_embedding_service = OptimizedEmbeddingService()
        
        print("\n🚀 Starting document processing pipeline...")
        
        # Step 1: Extract document content
        print("\n📝 Step 1: Document Extraction")
        extraction_result = await document_processor.extract_document(
            document.file_path, 
            original_filename=document.original_filename
        )
        
        if not extraction_result.success:
            print(f"❌ Extraction failed: {extraction_result.method}")
            document.status = "failed"
            db.commit()
            return False
        
        print(f"✅ Extraction successful using {extraction_result.method}")
        document.content = extraction_result.content
        document.status = "extracted"
        db.commit()
        
        # Step 2: Chunk the document
        print("\n✂️ Step 2: Document Chunking")
        chunking_result = await document_chunker.process_document_from_db(db, document.id)
        
        if not chunking_result.success:
            print(f"❌ Chunking failed: {chunking_result.metadata}")
            document.status = "failed"
            db.commit()
            return False
        
        print(f"✅ Chunking successful: {chunking_result.chunks_created} chunks created")
        
        # Step 3: Create embeddings
        print("\n🧠 Step 3: Embedding Creation (Optimized)")
        embedding_result = await optimized_embedding_service.process_embeddings_for_document(db, document.id)
        
        if not embedding_result.success:
            print(f"❌ Embedding creation failed: {embedding_result.metadata}")
            document.status = "failed"
            db.commit()
            return False
        
        print(f"✅ Embedding creation successful: {embedding_result.embeddings_created} embeddings created")
        
        # Update final status
        document.status = "processed"
        db.commit()
        
        # Verify results
        chunk_count = db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).count()
        embedding_count = db.query(Embedding).join(
            DocumentChunk, Embedding.chunk_id == DocumentChunk.id
        ).filter(DocumentChunk.document_id == document.id).count()
        
        print(f"\n🎉 Processing Complete!")
        print(f"📊 Final Status: {document.status}")
        print(f"📄 Content Length: {len(document.content) if document.content else 0} characters")
        print(f"✂️ Chunks Created: {chunk_count}")
        print(f"🧠 Embeddings Created: {embedding_count}")
        
        # Show some sample chunks and embeddings
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).limit(3).all()
        print(f"\n📋 Sample chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  {i+1}. {chunk.text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("🧪 Testing Simple Document Processing Pipeline")
    print("=" * 50)
    
    success = asyncio.run(test_simple_document_processing())
    
    if success:
        print("\n✅ Test completed successfully!")
        print("\n🎯 The document processing pipeline is now working correctly!")
        print("   - Document extraction ✓")
        print("   - Document chunking ✓") 
        print("   - Embedding creation (optimized) ✓")
    else:
        print("\n❌ Test failed!")