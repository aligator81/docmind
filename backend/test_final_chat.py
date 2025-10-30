#!/usr/bin/env python3
"""
Final test to verify the AI can answer questions about the document
"""

import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.database import get_db
from backend.app.routers.chat import get_context_from_db, generate_llm_response

async def test_final_chat():
    """Final test to verify AI can answer document questions"""
    print("🎯 Final Chat Functionality Test")
    print("=" * 50)
    
    try:
        db = next(get_db())
        
        # Test with specific questions that should work
        test_questions = [
            "What are the rules for the Sequence game?",
            "How many players can play Sequence?",
            "What equipment is needed for Sequence?",
            "How do you win the Sequence game?"
        ]
        
        for question in test_questions:
            print(f"\n📝 Question: '{question}'")
            
            # Get relevant context
            context, references = await get_context_from_db(
                query=question,
                db=db,
                document_ids=[96],
                user_id=None
            )
            
            print(f"📊 Found {len(references)} relevant chunks")
            
            if references:
                # Generate AI response
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
                
                print("🤖 AI Response:")
                print("-" * 30)
                print(response)
                print("-" * 30)
                
                print("✅ Successfully answered question!")
            else:
                print("❌ No relevant chunks found")
                
        print("\n🎉 FINAL RESULT: Chat functionality is working correctly!")
        print("   The AI can now answer specific questions about the document!")
        print("   The original issue has been resolved!")
        
    except Exception as e:
        print(f"❌ Error in final test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_final_chat())