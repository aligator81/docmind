#!/usr/bin/env python3
"""
Test similarity matching with different questions
"""

import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.database import get_db
from backend.app.routers.chat import get_context_from_db

async def test_similarity():
    """Test similarity matching with different questions"""
    print("🧪 Testing Similarity Matching")
    print("=" * 50)
    
    try:
        db = next(get_db())
        
        # Test with different questions
        questions = [
            'What are the rules for the Sequence game?',
            'How many players can play Sequence?',
            'What equipment is needed for Sequence?',
            'what is the document about?'
        ]
        
        for question in questions:
            print(f"\n📝 Testing: '{question}'")
            context, references = await get_context_from_db(question, db, [96], None)
            print(f"📊 Found {len(references)} relevant chunks")
            
            if references:
                for ref in references:
                    print(f"  - Similarity: {ref['similarity']:.3f}")
                    print(f"    Document: {ref['filename']}")
            else:
                print("  ❌ No relevant chunks found")
                
    except Exception as e:
        print(f"❌ Error testing similarity: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_similarity())