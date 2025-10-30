"""
Test script for improved chunking functionality
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(__file__))

from app.database import SessionLocal
from app.services.improved_chunker import ImprovedDocumentChunker

async def test_improved_chunking():
    """Test the improved chunking on document ID 96"""
    db = SessionLocal()
    try:
        print("🧪 Testing improved chunking on document ID 96...")
        
        # Initialize improved chunker
        chunker = ImprovedDocumentChunker()
        
        # Process the document
        result = await chunker.process_document_from_db(db, 96)
        
        print(f"\n📊 Results:")
        print(f"✅ Success: {result.success}")
        print(f"📄 Chunks created: {result.chunks_created}")
        print(f"⏱️ Processing time: {result.processing_time:.2f}s")
        print(f"📋 Metadata: {result.metadata}")
        
        # Verify the chunks were created
        from app.models import DocumentChunk
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == 96).all()
        print(f"\n🔍 Database verification: {len(chunks)} chunks found")
        
        if chunks:
            print("\n📝 Chunk previews:")
            for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                print(f"  Chunk {i}: {chunk.chunk_text[:100]}...")
                print(f"    Pages: {chunk.page_numbers}, Title: {chunk.section_title}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = asyncio.run(test_improved_chunking())
    if success:
        print("\n🎉 Improved chunking test completed successfully!")
    else:
        print("\n❌ Improved chunking test failed!")