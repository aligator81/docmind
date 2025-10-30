#!/usr/bin/env python3
"""
Simple test to verify chat functionality works with the improved chunks
"""

import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.database import get_db
from backend.app.routers.chat import get_context_from_db, generate_llm_response

async def test_chat():
    """Test chat functionality with a specific question"""
    print("🧪 Testing Chat with Specific Question")
    print("=" * 50)
    
    try:
        db = next(get_db())
        
        # Test with a question that should match the content
        question = "What are the rules for the Sequence game?"
        print(f"📝 Question: '{question}'")
        
        # Search for relevant chunks
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
            
            # Generate AI response
            print("🤖 Generating AI response...")
            
            system_prompt_template = """You are a helpful assistant that answers questions based on the provided document context.

Context:
{context}

{references_text}

Please provide a clear and helpful answer based on the context provided."""

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
            print("   The AI can now answer questions about the document!")
            
        else:
            print("❌ No relevant chunks found for the question")
            print("   This suggests the embeddings or search isn't working properly")
            
    except Exception as e:
        print(f"❌ Error testing chat functionality: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_chat())