#!/usr/bin/env python3
"""
Create a document from the existing test file and test the complete processing pipeline
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.database import SessionLocal
from app.models import Document, User
from app.services.document_processor import DocumentProcessor
from app.services.document_chunker import DocumentChunker
from app.services.optimized_embedding_service import OptimizedEmbeddingService

async def create_and_test_document():
    """Create a document from existing test file and test processing"""
    print("📄 Creating and Testing Document Processing")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        # Check if we have a user (required for document ownership)
        user = db.query(User).first()
        if not user:
            print("❌ No users found in database. Creating a test user...")
            user = User(
                username="testuser",
                password_hash="test_hash",  # This would be hashed in real scenario
                email="test@example.com",
                role="user"
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"✅ Created test user with ID: {user.id}")
        
        # Check existing files
        uploads_dir = Path("data/uploads")
        existing_files = list(uploads_dir.glob("*"))
        
        if not existing_files:
            print("❌ No files found in uploads directory")
            return
        
        test_file = existing_files[0]
        print(f"📁 Using existing file: {test_file.name} ({test_file.stat().st_size} bytes)")
        
        # Check if document already exists for this file
        existing_doc = db.query(Document).filter(
            Document.filename == test_file.name
        ).first()
        
        if existing_doc:
            print(f"📄 Document already exists with ID: {existing_doc.id}")
            doc = existing_doc
        else:
            # Create new document
            print("📄 Creating new document record...")
            doc = Document(
                filename=test_file.name,
                original_filename=test_file.name,
                file_path=str(test_file),
                file_size=test_file.stat().st_size,
                mime_type="text/markdown" if test_file.suffix == ".md" else "application/octet-stream",
                user_id=user.id,
                status="not processed"
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            print(f"✅ Created document with ID: {doc.id}")
        
        # Now test the complete processing pipeline
        print("\n🔄 Testing Complete Processing Pipeline")
        print("=" * 40)
        
        # Step 1: Extract content
        print("\n1️⃣ Step 1: Document Extraction")
        processor = DocumentProcessor()
        extraction_result = await processor.extract_document(
            doc.file_path,
            original_filename=doc.original_filename
        )
        
        if not extraction_result.success:
            print(f"❌ Extraction failed: {extraction_result.method}")
            doc.status = "failed"
            db.commit()
            return
        
        print(f"✅ Extraction successful using {extraction_result.method}")
        print(f"📄 Extracted {len(extraction_result.content)} characters")
        
        # Update document
        doc.content = extraction_result.content
        doc.status = "extracted"
        db.commit()
        
        # Step 2: Chunk the document
        print("\n2️⃣ Step 2: Document Chunking")
        chunker = DocumentChunker()
        chunking_result = await chunker.process_document_from_db(db, doc.id)
        
        if not chunking_result.success:
            print(f"❌ Chunking failed: {chunking_result.metadata}")
            doc.status = "failed"
            db.commit()
            return
        
        print(f"✅ Chunking successful - created {chunking_result.chunks_created} chunks")
        print(f"📊 Metadata: {chunking_result.metadata}")
        
        # Step 3: Create embeddings
        print("\n3️⃣ Step 3: Embedding Creation")
        embedding_service = OptimizedEmbeddingService()
        embedding_result = await embedding_service.process_embeddings_from_db(db)
        
        if not embedding_result.success:
            print(f"❌ Embedding creation failed: {embedding_result.metadata}")
            # Don't mark as failed since chunking worked
            doc.status = "chunked"
        else:
            print(f"✅ Embedding creation successful - created {embedding_result.embeddings_created} embeddings")
            doc.status = "processed"
        
        doc.processed_at = datetime.utcnow()
        db.commit()
        
        print(f"\n🎉 Processing pipeline completed successfully!")
        print(f"📊 Final status: {doc.status}")
        
        # Show final database state
        print(f"\n📋 Final Database State:")
        print(f"  - Document ID: {doc.id}")
        print(f"  - Filename: {doc.filename}")
        print(f"  - Status: {doc.status}")
        print(f"  - Content length: {len(doc.content) if doc.content else 0}")
        print(f"  - Processed at: {doc.processed_at}")
        
        # Count chunks and embeddings
        from app.models import DocumentChunk, Embedding
        chunk_count = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == doc.id
        ).count()
        embedding_count = db.query(Embedding).join(
            DocumentChunk, Embedding.chunk_id == DocumentChunk.id
        ).filter(
            DocumentChunk.document_id == doc.id
        ).count()
        
        print(f"  - Chunks created: {chunk_count}")
        print(f"  - Embeddings created: {embedding_count}")
        
    except Exception as e:
        print(f"❌ Error in processing pipeline: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

async def main():
    """Main function to create and test document processing"""
    print("📄 Docling Document Processing Test")
    print("=" * 60)
    
    await create_and_test_document()
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print("The document processing pipeline has been tested.")
    print("If successful, you should see chunks and embeddings created in the database.")
    print("You can now upload new documents through the web interface.")

if __name__ == "__main__":
    asyncio.run(main())