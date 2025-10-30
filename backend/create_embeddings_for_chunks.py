"""
Create embeddings for the newly created chunks
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(__file__))

from app.database import SessionLocal
from app.services.embedding_service import EmbeddingService

async def create_embeddings_for_document(document_id: int):
    """Create embeddings for chunks of a specific document"""
    db = SessionLocal()
    try:
        print(f"🧬 Creating embeddings for document ID {document_id}...")
        
        # Initialize embedding service
        embedding_service = EmbeddingService(provider="openai")
        
        # Process embeddings for the document
        result = await embedding_service.process_embeddings_for_document(db, document_id)
        
        print(f"\n📊 Embedding Results:")
        print(f"✅ Success: {result.success}")
        print(f"📄 Embeddings created: {result.embeddings_created}")
        print(f"⏱️ Processing time: {result.processing_time:.2f}s")
        print(f"📋 Metadata: {result.metadata}")
        
        # Verify embeddings were created
        from app.models import Embedding, DocumentChunk
        embeddings = db.query(Embedding).join(
            DocumentChunk, Embedding.chunk_id == DocumentChunk.id
        ).filter(
            DocumentChunk.document_id == document_id
        ).all()
        
        print(f"\n🔍 Database verification: {len(embeddings)} embeddings found")
        
        if embeddings:
            print("\n📝 Embedding previews:")
            for i, embedding in enumerate(embeddings[:3]):  # Show first 3 embeddings
                print(f"  Embedding {i}: Chunk ID {embedding.chunk_id}, Provider: {embedding.embedding_provider}")
                print(f"    Model: {embedding.embedding_model}, Vector length: {len(embedding.embedding_vector) if embedding.embedding_vector else 0}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = asyncio.run(create_embeddings_for_document(96))
    if success:
        print("\n🎉 Embedding creation completed successfully!")
    else:
        print("\n❌ Embedding creation failed!")