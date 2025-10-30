#!/usr/bin/env python3
"""
Test script to verify chat functionality with the improved chunking and embeddings
"""

import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.database import get_db
from backend.app.services.embedding_service import EmbeddingService
from backend.app.routers.chat import get_context_from_db, get_embedding, generate_llm_response

async def test_chat_functionality():
    """Test if the AI can now answer 'what is the document about?'"""
    print("🧪 Testing Chat Functionality")
    print("=" * 50)
    
    # Test question
    question = "what is the document about?"
    print(f"📝 Question: '{question}'")
    
    try:
        # Get database session
        db = next(get_db())
        
        # Search for relevant chunks using the actual chat function
        print("🔍 Searching for relevant chunks...")
        context, references = await get_context_from_db(
            query=question,
            db=db,
            document_ids=[96],  # Test with document ID 96
            user_id=None  # No user filter for testing
        )
        
        print(f"📊 Found {len(references)} relevant chunks")
        
        if references:
            print("\n📄 Relevant Chunks Found:")
            for i, ref in enumerate(references[:3], 1):
                print(f"  {i}. Chunk ID: {ref['id']}")
                print(f"     Similarity: {ref['similarity']:.3f}")
                print(f"     Document: {ref['filename']}")
                print(f"     Section: {ref['section_title']}")
                print()
            
            # Test if we can generate a summary using the actual LLM
            print("🤖 Testing AI response generation...")
            
            # Use the actual LLM generation function
            system_prompt_template = """You are a helpful assistant that answers questions based on the provided document context.

Context:
{context}

{references_text}

Please provide a clear and concise summary of what the document is about based on the context provided."""

            response = await generate_llm_response(
                message=question,
                context=context,
                references=references,
                provider="openai",
                model="gpt-4o-mini",
                document_ids=[96],
                system_prompt_template=system_prompt_template
            )
            
            print("\n🤖 AI Response:")
            print("=" * 50)
            print(response)
            print("=" * 50)
            
            print("\n✅ Chat functionality test completed successfully!")
            print("   The AI can now answer 'what is the document about?'")
            
        else:
            print("❌ No relevant chunks found for the question")
            print("   This suggests the embeddings or search isn't working properly")
            
    except Exception as e:
        print(f"❌ Error testing chat functionality: {e}")
        import traceback
        traceback.print_exc()

async def verify_document_state():
    """Verify the current state of the document and its chunks"""
    print("\n📊 Document State Verification")
    print("=" * 50)
    
    try:
        db = next(get_db())
        
        # Check document
        from backend.app.models import Document, DocumentChunk, Embedding
        from sqlalchemy import func
        
        # Get document info
        document = db.query(Document).filter(Document.id == 96).first()
        if document:
            print(f"📄 Document ID: {document.id}")
            print(f"📝 Original Filename: {document.original_filename}")
            print(f"📁 Filename: {document.filename}")
            print(f"📊 Status: {document.status}")
            print(f"📏 Size: {document.file_size} bytes")
        
        # Count chunks
        chunk_count = db.query(DocumentChunk).filter(DocumentChunk.document_id == 96).count()
        print(f"🧩 Total chunks: {chunk_count}")
        
        # Count embeddings
        embedding_count = db.query(Embedding).join(DocumentChunk).filter(DocumentChunk.document_id == 96).count()
        print(f"🧬 Total embeddings: {embedding_count}")
        
        # Show chunk details
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == 96).all()
        print(f"\n📋 Chunk Details:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. Chunk ID: {chunk.id}")
            print(f"     Content length: {len(chunk.chunk_text)} characters")
            print(f"     Created: {chunk.created_at}")
            
        return chunk_count, embedding_count
        
    except Exception as e:
        print(f"❌ Error verifying document state: {e}")
        return 0, 0

async def main():
    """Main test function"""
    print("🚀 Starting Chat Functionality Test")
    print("=" * 50)
    
    # First verify the document state
    chunk_count, embedding_count = await verify_document_state()
    
    if chunk_count > 0 and embedding_count > 0:
        print(f"\n✅ Document has {chunk_count} chunks and {embedding_count} embeddings")
        print("   Ready to test chat functionality...")
        
        # Test chat functionality
        await test_chat_functionality()
    else:
        print("❌ Document doesn't have sufficient chunks or embeddings")
        print("   Please run the improved chunking and embedding scripts first")

if __name__ == "__main__":
    asyncio.run(main())