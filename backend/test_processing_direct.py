#!/usr/bin/env python3
"""
Direct test for document processing pipeline (chunking + embedding)
"""

import sys
import os
import asyncio
sys.path.append('.')

from app.database import SessionLocal
from app.models import Document, DocumentChunk, Embedding, User
from app.services import DocumentChunker
from app.services.optimized_embedding_service import OptimizedEmbeddingService

async def test_processing_direct():
    """Test the chunking and embedding pipeline directly"""
    db = SessionLocal()
    
    try:
        # Get a test user (use the first user)
        user = db.query(User).first()
        if not user:
            print("❌ No users found in database")
            return False
        
        print(f"👤 Using user: {user.username} (ID: {user.id})")
        
        # Create a test document record with pre-existing content
        test_content = """This is a simple test document to verify the document processing pipeline.

The document processing pipeline should:
1. Extract the text content
2. Split it into chunks
3. Create embeddings for the chunks

This is a test of the complete document processing workflow."""

        document = Document(
            filename="test_direct.txt",
            original_filename="test_direct.txt",
            file_path="test_direct.txt",
            file_size=len(test_content),
            mime_type="text/plain",
            user_id=user.id,
            status="extracted",  # Start with extracted content
            content=test_content
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        print(f"📄 Created test document: {document.filename} (ID: {document.id})")
        print(f"📄 Content length: {len(document.content)} characters")
        
        # Initialize services
        document_chunker = DocumentChunker()
        optimized_embedding_service = OptimizedEmbeddingService()
        
        print("\n🚀 Starting document processing pipeline...")
        
        # Step 1: Chunk the document
        print("\n✂️ Step 1: Document Chunking")
        chunking_result = await document_chunker.process_document_from_db(db, document.id)
        
        if not chunking_result.success:
            print(f"❌ Chunking failed: {chunking_result.metadata}")
            document.status = "failed"
            db.commit()
            return False
        
        print(f"✅ Chunking successful: {chunking_result.chunks_created} chunks created")
        
        # Step 2: Create embeddings
        print("\n🧠 Step 2: Embedding Creation (Optimized)")
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
            print(f"  {i+1}. {chunk.chunk_text[:100]}...")
        
        # Show embeddings
        embeddings = db.query(Embedding).join(
            DocumentChunk, Embedding.chunk_id == DocumentChunk.id
        ).filter(DocumentChunk.document_id == document.id).limit(2).all()
        
        print(f"\n🧠 Sample embeddings:")
        for i, embedding in enumerate(embeddings):
            print(f"  {i+1}. Provider: {embedding.embedding_provider}, Model: {embedding.embedding_model}")
            print(f"     Vector length: {len(embedding.embedding_vector) if embedding.embedding_vector else 0}")
            print(f"     Created at: {embedding.created_at}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("🧪 Testing Document Processing Pipeline (Direct)")
    print("=" * 50)
    
    success = asyncio.run(test_processing_direct())
    
    if success:
        print("\n✅ Test completed successfully!")
        print("\n🎯 The document processing pipeline is now working correctly!")
        print("   - Document chunking ✓")
        print("   - Embedding creation (optimized) ✓")
        print("\n💡 The frontend document processing should now work properly!")
    else:
        print("\n❌ Test failed!")