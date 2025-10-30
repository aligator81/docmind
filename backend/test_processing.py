#!/usr/bin/env python3
"""
Test script for document processing pipeline
"""

import sys
import os
import asyncio
sys.path.append('.')

from app.database import SessionLocal
from app.models import Document, DocumentChunk, Embedding
from app.services import DocumentProcessor, DocumentChunker
from app.services.optimized_embedding_service import OptimizedEmbeddingService

async def test_document_processing(document_id: int):
    """Test the complete document processing pipeline"""
    db = SessionLocal()
    
    try:
        # Get the document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            print(f"❌ Document {document_id} not found")
            return False
        
        print(f"📄 Testing document: {document.filename}")
        print(f"📊 Current status: {document.status}")
        
        # Reset document status if it's failed
        if document.status == "failed":
            print("🔄 Resetting document status from 'failed' to 'not processed'")
            document.status = "not processed"
            db.commit()
        
        # Initialize services
        document_processor = DocumentProcessor()
        document_chunker = DocumentChunker()
        optimized_embedding_service = OptimizedEmbeddingService()
        
        print("\n🚀 Starting document processing pipeline...")
        
        # Step 1: Extract document content
        print("\n📝 Step 1: Document Extraction")
        if not document.file_path or not os.path.exists(document.file_path):
            print(f"❌ Document file not found: {document.file_path}")
            return False
        
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
        chunking_result = await document_chunker.process_document_from_db(db, document_id)
        
        if not chunking_result.success:
            print(f"❌ Chunking failed: {chunking_result.metadata}")
            document.status = "failed"
            db.commit()
            return False
        
        print(f"✅ Chunking successful: {chunking_result.chunks_created} chunks created")
        
        # Step 3: Create embeddings
        print("\n🧠 Step 3: Embedding Creation (Optimized)")
        embedding_result = await optimized_embedding_service.process_embeddings_for_document(db, document_id)
        
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
        chunk_count = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).count()
        embedding_count = db.query(Embedding).join(
            DocumentChunk, Embedding.chunk_id == DocumentChunk.id
        ).filter(DocumentChunk.document_id == document_id).count()
        
        print(f"\n🎉 Processing Complete!")
        print(f"📊 Final Status: {document.status}")
        print(f"📄 Content Length: {len(document.content) if document.content else 0} characters")
        print(f"✂️ Chunks Created: {chunk_count}")
        print(f"🧠 Embeddings Created: {embedding_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    document_id = 67  # The document we found earlier
    print("🧪 Testing Document Processing Pipeline")
    print("=" * 50)
    
    success = asyncio.run(test_document_processing(document_id))
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")